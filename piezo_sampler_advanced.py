from WF_SDK import device, scope
import time
import sys
import numpy as np
from collections import deque
from datetime import datetime

# =========================
# Configuration
# =========================
PRINT_HZ = 10                 # UI/print frequency (times per second)
CHANNEL = 1                   # AD3 channel
# Choose a safe default: if using MTE inputs start small (±2.5 V typical). With BNC+probe, larger is fine.
VOLTAGE_RANGE = 5             # Full-scale ±V (adjust to your hardware path)
SCOPE_FS = 100_000            # ADC sampling frequency [samples/second]
CHUNK_MS = 100                # Chunk length to process [ms]; 100 ms -> 10 chunks/s @ PRINT_HZ = 10
THRESHOLD_HI = 0.5            # Fixed high threshold [V]; set to None to auto-calibrate
HYSTERESIS_RATIO = 0.6        # TH_LO = HYSTERESIS_RATIO * THRESHOLD_HI
HOLDOFF_MS = 120              # Refractory period after an event [ms]
LOG_TO_FILE = False           # Event-level log
SAVE_EVENT_WAVEFORMS = False  # Save raw event waveforms (.npy)

# =========================
# Utility
# =========================
def now_ts():
    return time.strftime("%H:%M:%S", time.localtime())

class PiezoMonitor:
    def __init__(self):
        self.dev = None
        self.th_hi = THRESHOLD_HI
        self.th_lo = (HYSTERESIS_RATIO * THRESHOLD_HI) if THRESHOLD_HI is not None else None
        self.event_active = False
        self.last_event_end_t = -1e9
        self.hits = 0
        self.peak_voltage = 0.0
        self.chunk_samples = max(1, int(SCOPE_FS * (CHUNK_MS / 1000.0)))
        self.print_period = 1.0 / PRINT_HZ
        self.holdoff_s = HOLDOFF_MS / 1000.0
        self.log = None

    def connect(self):
        print("Connecting to Analog Discovery 3 ...")
        self.dev = device.open()

        scope.open(
            self.dev,
            sampling_frequency=SCOPE_FS,
            buffer_size=self.chunk_samples,
            offset=0,
            amplitude_range=VOLTAGE_RANGE,
        )

        if LOG_TO_FILE:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log = open(f"piezo_events_{ts}.csv", "w", buffering=1)
            self.log.write("timestamp,peak_v,ptp_v,rms_v,energy_v2,chunk_ms,event\n")
            print(f"Logging events to piezo_events_{ts}.csv")

        print(f"Connected on CH{CHANNEL}")
        print(f"ADC rate: {SCOPE_FS/1000:.1f} kS/s | chunk: {CHUNK_MS} ms ({self.chunk_samples} samples)")
        if self.th_hi is not None:
            print(f"Fixed thresholds: TH_HI={self.th_hi:.3f} V, TH_LO={self.th_hi*HYSTERESIS_RATIO:.3f} V")
        else:
            print("Auto threshold: will calibrate on idle noise")
        print("Press Ctrl+C to stop\n" + "-"*60)

    def _auto_calibrate_threshold(self, seconds=2.0, k_sigma=10.0):
        """
        Collect 'seconds' of idle data, estimate noise sigma via MAD, set TH_HI = k_sigma*sigma.
        """
        print(f"Calibrating threshold for {seconds:.1f}s ... please avoid impacts.")
        need = int(SCOPE_FS * seconds)
        buf = np.empty(0, dtype=np.float32)
        while buf.size < need:
            chunk = np.array(scope.record(self.dev, CHANNEL), dtype=np.float32)
            # Remove DC per chunk (robust)
            chunk = chunk - np.median(chunk)
            buf = np.concatenate((buf, chunk))
        # MAD-based sigma estimate (Gaussian: sigma ≈ MAD/0.6745)
        mad = np.median(np.abs(buf - np.median(buf)))
        sigma = mad / 0.6745 if mad > 0 else np.std(buf)
        self.th_hi = float(k_sigma * sigma)
        self.th_lo = float(HYSTERESIS_RATIO * self.th_hi)
        print(f"Auto thresholds: TH_HI={self.th_hi:.3f} V, TH_LO={self.th_lo:.3f} V\n" + "-"*60)

    def _process_chunk(self, x):
        """
        Return per-chunk features and event state transitions.
        """
        x = x.astype(np.float32, copy=False)
        # Remove slow offset if DC-coupled; safe even if AC-coupled
        x = x - np.median(x)

        peak = float(np.max(np.abs(x)))
        p2p = float(np.max(x) - np.min(x))
        rms = float(np.sqrt(np.mean(x*x)))
        energy = float(np.sum(x*x))  # not normalized by length -> "energy proxy"

        # Update global peak
        if peak > self.peak_voltage:
            self.peak_voltage = peak

        t_now = time.monotonic()
        event_started, event_ended = False, False

        # Event logic with Schmitt trigger and holdoff
        if not self.event_active:
            if self.th_hi is not None and (peak >= self.th_hi):
                if (t_now - self.last_event_end_t) >= self.holdoff_s:
                    self.event_active = True
                    self.hits += 1
                    event_started = True
        else:
            # End event when the whole chunk is below TH_LO (aggressive) or the peak drops below TH_LO (lenient)
            if self.th_lo is None:
                lo_cond = peak < (0.6 * self.th_hi)  # fallback
            else:
                lo_cond = peak < self.th_lo
            if lo_cond:
                self.event_active = False
                self.last_event_end_t = t_now
                event_ended = True

        return peak, p2p, rms, energy, event_started, event_ended, x

    def run(self):
        if self.th_hi is None:
            self._auto_calibrate_threshold()

        try:
            while True:
                t0 = time.monotonic()
                raw = np.array(scope.record(self.dev, CHANNEL), dtype=np.float32)
                peak, p2p, rms, energy, ev_start, ev_end, detrended = self._process_chunk(raw)

                # Print status line
                stamp = now_ts()
                line = (f"[{stamp}] "
                        f"peak:{peak:6.3f}V  p2p:{p2p:6.3f}V  rms:{rms:6.3f}V  "
                        f"E:{energy:9.1f}  max:{self.peak_voltage:6.3f}V  hits:{self.hits}")
                if ev_start:
                    line += "  *** COLLISION ***"
                print(line)

                # Event-level logging
                if self.log:
                    self.log.write(f"{stamp},{peak:.6f},{p2p:.6f},{rms:.6f},{energy:.3f},{CHUNK_MS},{1 if ev_start else 0}\n")

                # Save waveform if an event starts in this chunk
                if SAVE_EVENT_WAVEFORMS and ev_start:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    np.save(f"impact_{ts}.npy", detrended)  # centered waveform

                # Pace the loop for human-readable console output
                elapsed = time.monotonic() - t0
                if elapsed < self.print_period:
                    time.sleep(self.print_period - elapsed)

        except KeyboardInterrupt:
            print("\n" + "-"*60)
            print("Stopped by user.")
            print(f"Total hits: {self.hits}")
            print(f"Max observed voltage: {self.peak_voltage:.3f} V")

    def record_buffer(self, duration_seconds=1.0):
        """
        Capture a single long buffer (best effort within device buffer limits) and summarize it.
        """
        print(f"\nRecording {duration_seconds:.2f} s of data ...")
        buffer_size = int(SCOPE_FS * duration_seconds)
        # If your SDK exposes a max buffer size property, check here; otherwise just ask for it.
        scope.open(self.dev, sampling_frequency=SCOPE_FS, buffer_size=buffer_size,
                   offset=0, amplitude_range=VOLTAGE_RANGE)
        data = np.array(scope.record(self.dev, CHANNEL), dtype=np.float32)
        data = data - np.median(data)
        peak = float(np.max(np.abs(data)))
        p2p = float(np.max(data) - np.min(data))
        rms = float(np.sqrt(np.mean(data*data)))
        crossings = int(np.count_nonzero(np.abs(data) > (self.th_hi if self.th_hi else 0.5)))
        print("Buffer analysis:")
        print(f"  Samples: {len(data)}  | duration: {len(data)/SCOPE_FS:.3f}s")
        print(f"  Max amplitude: {peak:.4f} V  | p-p: {p2p:.4f} V  | RMS: {rms:.4f} V")
        print(f"  Threshold crossings: {crossings}")
        return data

    def cleanup(self):
        if self.log:
            self.log.close()
        if self.dev:
            try:
                scope.close(self.dev)
            finally:
                device.close(self.dev)
            print("Device disconnected.")

def main():
    mon = PiezoMonitor()
    try:
        mon.connect()

        print("\nSelect mode:")
        print("1. Continuous monitoring (chunked streaming)")
        print("2. One-shot buffered capture (then analyze)")
        print("3. Buffered capture, then monitor")
        choice = input("Enter choice (1-3): ").strip()

        if choice in ("2", "3"):
            dur = float(input("Enter recording duration in seconds: "))
            buf = mon.record_buffer(dur)
            save = input("Save buffer to file? (y/n): ").strip().lower()
            if save == "y":
                fname = f"buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                np.savetxt(fname, buf, fmt="%.6f")
                print(f"Saved {fname}")

        if choice in ("1", "3"):
            print("\nStarting continuous monitoring ...")
            time.sleep(0.5)
            mon.run()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        mon.cleanup()

if __name__ == "__main__":
    main()

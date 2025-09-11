from WF_SDK import device, scope
import time
import sys
import numpy as np
from collections import deque
from datetime import datetime

# Configuration
SAMPLE_RATE_HZ = 10  # How many times per second to process and print values
CHANNEL = 1  # AD3 channel where piezo is connected
VOLTAGE_RANGE = 5  # ±5V range (adjust based on your piezo output)
THRESHOLD = 0.5  # Voltage threshold for collision detection
AVERAGING_WINDOW = 10  # Number of samples to average
LOG_TO_FILE = False  # Set to True to log data to CSV file

class PiezoSampler:
    def __init__(self):
        self.device_data = None
        self.voltage_history = deque(maxlen=AVERAGING_WINDOW)
        self.peak_voltage = 0
        self.collision_count = 0
        self.log_file = None
        
        if LOG_TO_FILE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"piezo_log_{timestamp}.csv"
            self.log_file = open(filename, 'w')
            self.log_file.write("Timestamp,Voltage,Average,Peak,Collision\n")
            print(f"Logging to: {filename}")
    
    def connect(self):
        """Connect to the Analog Discovery 3"""
        print("Connecting to Analog Discovery 3...")
        self.device_data = device.open()
        
        # Initialize scope with optimal settings for piezo sensing
        scope.open(self.device_data,
                   sampling_frequency=100e03,  # 100 kHz for good time resolution
                   buffer_size=100,
                   offset=0,
                   amplitude_range=VOLTAGE_RANGE)
        
        print(f"Connected! Sampling on channel {CHANNEL}")
        print(f"Sample rate: {SAMPLE_RATE_HZ} Hz")
        print(f"Collision threshold: {THRESHOLD} V")
        print(f"Averaging window: {AVERAGING_WINDOW} samples")
        print("Press Ctrl+C to stop\n")
        print("-" * 60)
    
    def sample_continuous(self):
        """Continuously sample and display piezo signal"""
        sample_period = 1.0 / SAMPLE_RATE_HZ
        
        try:
            while True:
                start_time = time.time()
                
                # Get voltage reading
                voltage = scope.measure(self.device_data, CHANNEL)
                self.voltage_history.append(voltage)
                
                # Calculate statistics
                if len(self.voltage_history) > 0:
                    avg_voltage = np.mean(self.voltage_history)
                    std_voltage = np.std(self.voltage_history) if len(self.voltage_history) > 1 else 0
                else:
                    avg_voltage = voltage
                    std_voltage = 0
                
                # Update peak
                abs_voltage = abs(voltage)
                if abs_voltage > self.peak_voltage:
                    self.peak_voltage = abs_voltage
                
                # Check for collision
                collision_detected = abs_voltage > THRESHOLD
                if collision_detected:
                    self.collision_count += 1
                
                # Display data
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                print(f"[{timestamp}] ", end="")
                print(f"V: {voltage:+.4f}V | ", end="")
                print(f"Avg: {avg_voltage:+.4f}V | ", end="")
                print(f"Std: {std_voltage:.4f}V | ", end="")
                print(f"Peak: {self.peak_voltage:.4f}V | ", end="")
                print(f"Hits: {self.collision_count}", end="")
                
                if collision_detected:
                    print(" *** COLLISION! ***", end="")
                
                print()  # New line
                
                # Log to file if enabled
                if self.log_file:
                    self.log_file.write(f"{timestamp},{voltage:.6f},{avg_voltage:.6f},"
                                       f"{self.peak_voltage:.6f},{1 if collision_detected else 0}\n")
                    self.log_file.flush()
                
                # Maintain sample rate
                elapsed = time.time() - start_time
                if elapsed < sample_period:
                    time.sleep(sample_period - elapsed)
                    
        except KeyboardInterrupt:
            print("\n" + "-" * 60)
            print("Measurement stopped")
            print(f"Total collisions detected: {self.collision_count}")
            print(f"Maximum voltage recorded: {self.peak_voltage:.4f}V")
    
    def sample_buffered(self, duration_seconds=1):
        """Record a buffer of data for specified duration"""
        print(f"\nRecording {duration_seconds} second(s) of data...")
        
        # Configure for buffered recording
        buffer_size = int(100e03 * duration_seconds)  # Based on 100kHz sampling
        if buffer_size > self.device_data.analog.input.max_buffer_size:
            buffer_size = self.device_data.analog.input.max_buffer_size
            actual_duration = buffer_size / 100e03
            print(f"Adjusted duration to {actual_duration:.2f}s due to buffer limits")
        
        scope.open(self.device_data,
                   sampling_frequency=100e03,
                   buffer_size=buffer_size,
                   offset=0,
                   amplitude_range=VOLTAGE_RANGE)
        
        # Record data
        buffer = scope.record(self.device_data, CHANNEL)
        
        # Analyze buffer
        buffer_np = np.array(buffer)
        max_val = np.max(np.abs(buffer_np))
        mean_val = np.mean(buffer_np)
        std_val = np.std(buffer_np)
        
        # Find peaks (simple threshold crossing)
        peaks = np.where(np.abs(buffer_np) > THRESHOLD)[0]
        num_peaks = len(peaks)
        
        print(f"Buffer analysis:")
        print(f"  Samples: {len(buffer)}")
        print(f"  Max amplitude: {max_val:.4f}V")
        print(f"  Mean: {mean_val:.4f}V")
        print(f"  Std deviation: {std_val:.4f}V")
        print(f"  Threshold crossings: {num_peaks}")
        
        return buffer
    
    def cleanup(self):
        """Clean up resources"""
        if self.log_file:
            self.log_file.close()
        if self.device_data:
            scope.close(self.device_data)
            device.close(self.device_data)
            print("Device disconnected")

def main():
    sampler = PiezoSampler()
    
    try:
        sampler.connect()
        
        # Choose sampling mode
        print("\nSelect mode:")
        print("1. Continuous sampling (real-time display)")
        print("2. Buffered recording (record then analyze)")
        print("3. Both (record buffer first, then continuous)")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '2' or choice == '3':
            duration = float(input("Enter recording duration in seconds: "))
            buffer = sampler.sample_buffered(duration)
            
            # Optional: save buffer to file
            save = input("Save buffer to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"piezo_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                np.savetxt(filename, buffer, fmt='%.6f')
                print(f"Buffer saved to {filename}")
        
        if choice == '1' or choice == '3':
            print("\nStarting continuous sampling...")
            time.sleep(1)
            sampler.sample_continuous()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sampler.cleanup()

if __name__ == "__main__":
    main()
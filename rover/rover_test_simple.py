from megapi import *
import time
import threading
import csv
import os
from WF_SDK import device, scope

# Configuration
SAMPLE_RATE_HZ = 10  # How many times per second to print values
CHANNEL = 1  # AD3 channel where piezo is connected
VOLTAGE_RANGE = 5  # ±5V range (adjust based on your piezo output)
THRESHOLD = 0.025  # Voltage threshold for collision detection (adjust as needed)
DISTANCE = 0.89  # Distance traveled in meters before collision'

LEFT_SPEED = 57  # Speed for left motors
RIGHT_SPEED = 50  # Speed for right motors

# Sampling frequency
SAMPLING_FREQUENCY = 100e3  # 100 kHz from scope

# Outlier filtering
MAX_VOLTAGE_CHANGE = 0.2  # Maximum voltage change per sample (V) - discard spikes exceeding this

ad3 = None
bot = MegaPi()
voltage_data = []  # Store (timestamp, voltage) tuples
max_voltage = 0  # Track maximum voltage seen
start_time = 0 # Track start time
end_time = 0 # Track end time
elapsed_time = 0 # Track elapsed time
collision_detected_time = None  # Track when collision was detected
previous_voltage = None  # Track previous valid voltage for outlier filtering
outliers_filtered = 0  # Count of filtered outlier samples

def onRead(level):
	print(level)

# Function to set the speed of the left motors
def set_left(speed: float):
    bot.motorRun(1, -speed)
    bot.motorRun(9, -speed)
    time.sleep(0.01)

# Function to set the speed of the right motors
def set_right(speed: float):
    bot.motorRun(2, speed)
    bot.motorRun(10, speed)
    time.sleep(0.01)

# Callback function for collision detection
def on_collision():
    """Handle collision response (runs in background thread) with ramped motor speeds."""
    try:
        print("Collision detected! Stopping motors, reversing with ramped speed.")
        set_left(0)
        set_right(0)
        time.sleep(1)

        # Ramp down to -50 over 500ms in 10 steps
        print("Ramping to reverse speed...")
        for step in range(10):
            speed = -(step + 1) * 5  # -5, -10, -15, ..., -50
            set_left(speed)
            set_right(speed)
            time.sleep(0.05)  # 50ms per step = 500ms total

        print("Holding reverse speed for 1 second...")
        time.sleep(1)

        # Ramp back to 0 over 500ms in 10 steps
        print("Ramping back to stop...")
        for step in range(10):
            speed = -(50 - (step + 1) * 5)  # -45, -40, -35, ..., 0
            set_left(speed)
            set_right(speed)
            time.sleep(0.05)  # 50ms per step = 500ms total

        set_left(0)
        set_right(0)
    except Exception:
        # Silently handle errors (port may be closed by main thread)
        pass
    return


def open_ad3():
    global ad3
    try:
        # Connect to the device
        print("Connecting to Analog Discovery 3...")
        ad3 = device.open()

        # Initialize the scope with high sampling frequency for better resolution
        # Using a smaller buffer for faster readings
        scope.open(ad3,
                   sampling_frequency=100e03,  # 100 kHz sampling
                   buffer_size=100,  # Small buffer for quick readings
                   offset=0,
                   amplitude_range=VOLTAGE_RANGE)

        print("Scope initialized")

    except Exception as e:
        print(f"Error opening device: {e}")
        raise


def check_collision():
    global max_voltage, voltage_data, start_time, end_time, elapsed_time, previous_voltage, outliers_filtered

    # Capture multiple samples during interval to find peak
    # Increased from 10 to 100 to achieve better sampling rate
    samples = 100  # Take 100 samples per check interval
    peak_voltage = 0
    peak_time = 0

    for _ in range(samples):
        voltage = scope.measure(ad3, CHANNEL)
        curr_time = time.time()
        abs_voltage = abs(voltage)

        # Outlier filtering: Check voltage change from previous valid sample
        if previous_voltage is None:
            # Accept first sample unconditionally
            voltage_data.append((curr_time, voltage))
            previous_voltage = voltage
        else:
            # Check if voltage change exceeds threshold
            voltage_change = abs(voltage - previous_voltage)
            if voltage_change <= MAX_VOLTAGE_CHANGE:
                # Valid sample - accept and update previous voltage
                voltage_data.append((curr_time, voltage))
                previous_voltage = voltage
            else:
                # Outlier detected - skip this sample
                outliers_filtered += 1
                # Do NOT update previous_voltage; wait for next valid sample

        # Track peak in this interval
        if abs_voltage > abs(peak_voltage):
            peak_voltage = voltage
            peak_time = curr_time

        # Track global max
        if abs_voltage > abs(max_voltage):
            max_voltage = voltage

    print(f"Peak in interval: {peak_voltage:+.4f} V | Max seen: {max_voltage:+.4f} V")

    # Check threshold for collision detection
    if abs(peak_voltage) > THRESHOLD:
        print(" *** THRESHOLD EXCEEDED - Possible collision! ***", end="")
        print(f" Collision Voltage: {peak_voltage:+.4f} V")

        end_time = peak_time
        elapsed_time = end_time - start_time
        return True
    return False

def end_routine():
    global voltage_data, elapsed_time, outliers_filtered, start_time, collision_detected_time

    set_left(0)
    set_right(0)
    print("Motors stopped and bot disconnected.")

    # Write all voltage data to CSV with incrementing filename in data subdirectory
    try:
        # Create data directory if it doesn't exist
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Find next available filename in data subdirectory
        counter = 0
        while os.path.exists(os.path.join(data_dir, f'rover_run_data_{counter}_L{LEFT_SPEED}_R{RIGHT_SPEED}.csv')):
            counter += 1

        csv_filename = os.path.join(data_dir, f'rover_run_data_{counter}_L{LEFT_SPEED}_R{RIGHT_SPEED}.csv')
        with open(csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'voltage'])
            csv_writer.writerows(voltage_data)
        print(f"CSV file saved: {csv_filename} ({len(voltage_data)} samples, {outliers_filtered} outliers filtered)")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

    # Calculate and print speed using collision_detected_time for accuracy
    if collision_detected_time is not None and start_time > 0:
        elapsed_time = collision_detected_time - start_time
        speed = DISTANCE / elapsed_time
        print(f"\nCollision detected at {elapsed_time:.3f} seconds")
        print(f"Distance traveled: {DISTANCE} m")
        print(f"Average speed before collision: {speed:.3f} m/s")

    if ad3 is not None:
        scope.close(ad3)
        device.close(ad3)
        print("Device disconnected")


def main():
    global start_time, elapsed_time, end_time, previous_voltage, outliers_filtered, collision_detected_time

    try:
        print("start")
        open_ad3()
        bot.start('/dev/cu.usbserial-2110')
        print("connected")
        time.sleep(1)

        # Reset filtering state for this run
        previous_voltage = None
        outliers_filtered = 0
        collision_detected_time = None

        # Start moving forward
        set_left(LEFT_SPEED)
        set_right(RIGHT_SPEED)
        start_time = time.time()

        while True:
            if check_collision() and collision_detected_time is None:
                # Collision just detected
                collision_detected_time = time.time()
                print("Elapsed time: {:.2f} seconds".format(elapsed_time))
                # Run collision response in background thread (non-blocking)
                collision_thread = threading.Thread(target=on_collision, daemon=True)
                collision_thread.start()
                print("[Collision response started in background, recording post-collision data...]")

                # Continue sampling while collision response is running
                while collision_thread.is_alive():
                    check_collision()
                    # time.sleep(0.001)

                break

            # time.sleep(0.001)  # Reduced from 10ms to 1ms for faster sampling

    finally:
        end_routine()


if __name__ == '__main__':
    sys.exit(main())

from WF_SDK import device, scope
import time

# Configuration
SAMPLE_RATE_HZ = 10  # How many times per second to print values
CHANNEL = 1  # AD3 channel where piezo is connected
VOLTAGE_RANGE = 5  # ±5V range (adjust based on your piezo output)
THRESHOLD = 0.5  # Voltage threshold for collision detection (adjust as needed)

def main():
    try:
        # Connect to the device
        print("Connecting to Analog Discovery 3...")
        device_data = device.open()
        
        # Initialize the scope with high sampling frequency for better resolution
        # Using a smaller buffer for faster readings
        scope.open(device_data, 
                   sampling_frequency=100e03,  # 100 kHz sampling
                   buffer_size=100,  # Small buffer for quick readings
                   offset=0, 
                   amplitude_range=VOLTAGE_RANGE)
        
        print(f"Sampling piezo signal on channel {CHANNEL}")
        print(f"Sample rate: {SAMPLE_RATE_HZ} Hz")
        print(f"Threshold: {THRESHOLD} V")
        print("Press Ctrl+C to stop\n")
        
        sample_period = 1.0 / SAMPLE_RATE_HZ
        
        while True:
            start_time = time.time()
            
            # Take a single voltage measurement
            voltage = scope.measure(device_data, CHANNEL)
            
            # Print the voltage value
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] Voltage: {voltage:+.4f} V", end="")
            
            # Check threshold for collision detection
            if abs(voltage) > THRESHOLD:
                print(" *** THRESHOLD EXCEEDED - Possible collision! ***", end="")
            
            print()  # New line
            
            # Wait for the remainder of the sample period
            elapsed = time.time() - start_time
            if elapsed < sample_period:
                time.sleep(sample_period - elapsed)
                
    except KeyboardInterrupt:
        print("\n\nStopping measurement...")
        
    finally:
        # Clean up
        scope.close(device_data)
        device.close(device_data)
        print("Device disconnected")

if __name__ == "__main__":
    main()
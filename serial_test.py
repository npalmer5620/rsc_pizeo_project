#!/usr/bin/env python3
import serial
import time

if __name__ == '__main__':
    print("Testing serial connection...")
    try:
        ser = serial.Serial('/dev/cu.usbserial-1330', 115200, timeout=1, dsrdtr=False, rtscts=False)
        print(f"Serial port opened: {ser.is_open}")
        print(f"Port: {ser.port}")
        print(f"Baudrate: {ser.baudrate}")
        print(f"FD: {ser.fd if hasattr(ser, 'fd') else 'N/A'}")

        # Try to check in_waiting multiple times
        time.sleep(0.5)
        for i in range(5):
            try:
                waiting = ser.in_waiting
                print(f"Attempt {i+1}: {waiting} bytes waiting")
                time.sleep(0.1)
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
                break

        ser.close()
        print("Serial port closed successfully")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

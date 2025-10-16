#!/usr/bin/env python3
"""
Collision Event Detector for Piezo Sensor Data

This script analyzes voltage data from a piezo sensor to detect and visualize
collision events based on voltage thresholds.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ======================== CONFIGURATION ========================
# Adjust these parameters as needed

# Sampling parameters
SAMPLING_RATE = 63.98e3  # Hz (63.98 kHz)

# Collision detection parameters
VOLTAGE_THRESHOLD = 0.05  # Volts - voltages above this are considered collisions
MIN_EVENT_SEPARATION = 0.01  # Seconds - minimum time between separate collision events

# Plotting parameters
PLOT_FULL_TIMELINE = True  # Show full 20-second timeline
PLOT_INDIVIDUAL_EVENTS = True  # Show zoomed plots of each collision
EVENT_WINDOW = 0.05  # Seconds - time window around each event to plot

# ================================================================


def load_data(filepath):
    """
    Load voltage data from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        numpy array of voltage values
    """
    print(f"Loading data from {filepath}...")

    # Read file and skip empty lines
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(float(line))
                except ValueError:
                    continue  # Skip any non-numeric lines

    voltage = np.array(data, dtype=np.float64)
    print(f"Loaded {len(voltage)} samples")

    return voltage


def create_time_axis(num_samples, sampling_rate):
    """
    Create time axis in seconds.

    Args:
        num_samples: Number of data samples
        sampling_rate: Sampling rate in Hz

    Returns:
        numpy array of time values in seconds
    """
    dt = 1.0 / sampling_rate
    time = np.arange(num_samples) * dt
    return time


def detect_collisions(voltage, time, threshold, min_separation):
    """
    Detect collision events in voltage data.

    Args:
        voltage: Array of voltage values
        time: Array of time values
        threshold: Voltage threshold for detection (absolute value)
        min_separation: Minimum time separation between events (seconds)

    Returns:
        List of dictionaries, each containing:
            - 'start_idx': Starting index of event
            - 'end_idx': Ending index of event
            - 'peak_idx': Index of peak voltage (maximum absolute value)
            - 'peak_voltage': Peak voltage value (signed)
            - 'time': Time of peak (seconds)
    """
    print(f"\nDetecting collisions (threshold = ±{threshold} V)...")

    # Find all samples above threshold (absolute value)
    above_threshold = np.abs(voltage) > threshold

    # Find indices where voltage crosses threshold
    crossings = np.diff(above_threshold.astype(int))
    start_indices = np.where(crossings == 1)[0] + 1
    end_indices = np.where(crossings == -1)[0] + 1

    # Handle edge cases
    if above_threshold[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if above_threshold[-1]:
        end_indices = np.append(end_indices, len(voltage) - 1)

    # Group into events
    events = []
    min_sep_samples = int(min_separation * SAMPLING_RATE)

    for start_idx, end_idx in zip(start_indices, end_indices):
        # Find peak (maximum absolute value) in this segment
        segment = voltage[start_idx:end_idx+1]
        peak_idx_rel = np.argmax(np.abs(segment))
        peak_idx = start_idx + peak_idx_rel
        peak_voltage = voltage[peak_idx]

        # Check if this should be merged with previous event
        if events and (start_idx - events[-1]['end_idx']) < min_sep_samples:
            # Merge with previous event
            prev_event = events[-1]
            prev_event['end_idx'] = end_idx

            # Update peak if this one has higher absolute value
            if np.abs(peak_voltage) > np.abs(prev_event['peak_voltage']):
                prev_event['peak_idx'] = peak_idx
                prev_event['peak_voltage'] = peak_voltage
                prev_event['time'] = time[peak_idx]
        else:
            # New event
            events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peak_idx': peak_idx,
                'peak_voltage': peak_voltage,
                'time': time[peak_idx]
            })

    print(f"Detected {len(events)} collision events")

    return events


def print_statistics(events, voltage, time):
    """Print statistics about detected events."""
    print("\n" + "="*60)
    print("COLLISION DETECTION SUMMARY")
    print("="*60)

    print(f"\nTotal recording duration: {time[-1]:.2f} seconds")
    print(f"Total samples: {len(voltage)}")
    print(f"Voltage range: {voltage.min():.4f} V to {voltage.max():.4f} V")
    print(f"\nNumber of collision events detected: {len(events)}")

    if events:
        print("\nEvent details:")
        print("-" * 60)
        for i, event in enumerate(events, 1):
            duration = (event['end_idx'] - event['start_idx']) / SAMPLING_RATE
            print(f"Event {i}:")
            print(f"  Time: {event['time']:.4f} s")
            print(f"  Peak voltage: {event['peak_voltage']:.4f} V")
            print(f"  Duration: {duration*1000:.2f} ms")

        # Overall statistics
        peak_voltages = [e['peak_voltage'] for e in events]
        print("-" * 60)
        print(f"\nPeak voltage statistics:")
        print(f"  Mean: {np.mean(peak_voltages):.4f} V")
        print(f"  Std:  {np.std(peak_voltages):.4f} V")
        print(f"  Max:  {np.max(peak_voltages):.4f} V")
        print(f"  Min:  {np.min(peak_voltages):.4f} V")

    print("="*60 + "\n")


def plot_full_timeline(voltage, time, events, threshold, output_filename):
    """Plot the full timeline with collision events marked."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot voltage
    ax.plot(time, voltage, 'b-', linewidth=0.5, alpha=0.7, label='Voltage')

    # Mark collision events
    for i, event in enumerate(events):
        ax.axvline(x=event['time'], color='r', linestyle='-',
                  linewidth=1, alpha=0.5)
        ax.plot(event['time'], event['peak_voltage'], 'ro',
               markersize=8, zorder=5)
        ax.text(event['time'], event['peak_voltage'], f"  {i+1}",
               verticalalignment='bottom', fontsize=8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title('Piezo Sensor Data - Full Timeline with Collision Events',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_filename}")

    return fig


def plot_individual_events(voltage, time, events, window_size, threshold, output_filename):
    """Plot individual collision events with surrounding context."""
    if not events:
        return None

    # Determine grid layout
    n_events = len(events)
    n_cols = min(3, n_events)
    n_rows = (n_events + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    # Handle single event case
    if n_events == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    window_samples = int(window_size * SAMPLING_RATE)

    for i, event in enumerate(events):
        ax = axes[i]

        # Determine window around event
        center_idx = event['peak_idx']
        start_idx = max(0, center_idx - window_samples)
        end_idx = min(len(voltage), center_idx + window_samples)

        # Extract window
        window_time = time[start_idx:end_idx]
        window_voltage = voltage[start_idx:end_idx]

        # Plot
        ax.plot(window_time, window_voltage, 'b-', linewidth=1, alpha=0.8)
        ax.axhline(y=threshold, color='r', linestyle='--',
                  linewidth=1, alpha=0.7)
        ax.axvline(x=event['time'], color='r', linestyle='-',
                  linewidth=1, alpha=0.5)
        ax.plot(event['time'], event['peak_voltage'], 'ro',
               markersize=10, zorder=5)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Voltage (V)', fontsize=10)
        ax.set_title(f"Event {i+1}: {event['time']:.4f}s, Peak: {event['peak_voltage']:.4f}V",
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_events, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_filename}")

    return fig


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Detect and visualize collision events in piezo sensor data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dataset1.csv
  %(prog)s dataset2.csv --threshold 0.08
  %(prog)s dataset3.csv --no-show
        """
    )

    parser.add_argument(
        'filename',
        type=str,
        help='CSV file containing voltage data'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=VOLTAGE_THRESHOLD,
        help=f'Voltage threshold for collision detection (default: {VOLTAGE_THRESHOLD} V)'
    )

    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Save plots without displaying them'
    )

    args = parser.parse_args()

    # Check if data file exists
    data_path = Path(args.filename)
    if not data_path.exists():
        print(f"Error: Data file '{args.filename}' not found!")
        return

    # Generate output filenames based on input filename
    base_name = data_path.stem  # filename without extension
    timeline_output = f"{base_name}_timeline.png"
    events_output = f"{base_name}_events_detail.png"

    # Load data
    voltage = load_data(args.filename)

    # Create time axis
    time = create_time_axis(len(voltage), SAMPLING_RATE)

    # Detect collisions
    events = detect_collisions(voltage, time, args.threshold, MIN_EVENT_SEPARATION)

    # Print statistics
    print_statistics(events, voltage, time)

    # Generate plots
    print("\nGenerating plots...")

    if PLOT_FULL_TIMELINE:
        plot_full_timeline(voltage, time, events, args.threshold, timeline_output)

    if PLOT_INDIVIDUAL_EVENTS and events:
        plot_individual_events(voltage, time, events, EVENT_WINDOW, args.threshold, events_output)

    print("\nDone! Check the output PNG files.")

    # Show plots unless --no-show is specified
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

# Configuration
THRESHOLD = 0.1  # Voltage threshold for collision detection (must match rover_test_simple.py)
DISTANCE = 0.8  # Distance traveled in meters (adjust based on your setup)

# Publication-quality matplotlib configuration
# Settings optimized for single-column journal figures (3.25 in wide)
PUBLICATION_RCPARAMS = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'font.family': 'sans-serif',
    'lines.linewidth': 0.75,
    'lines.markersize': 4,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'text.usetex': False,
}

# Professional color palette (colorblind-friendly)
COLOR_PRIMARY = '#0173B2'      # Blue for primary data
COLOR_THRESHOLD = '#DE8F05'    # Orange for threshold
COLOR_EVENT_PRE = '#0173B2'    # Blue for pre-collision
COLOR_EVENT_POST = '#029E73'   # Green for post-collision
COLOR_PEAK = '#D55E00'         # Red for peak marker

# Trial colors for multi-file overlays (colorblind-friendly palette)
TRIAL_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00']  # Up to 4 trials


def load_data(csv_file):
    """Load timestamp and voltage data from CSV file."""
    timestamps = []
    voltages = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            voltages.append(float(row['voltage']))

    if not timestamps:
        raise ValueError("No data found in CSV file")

    return np.array(timestamps), np.array(voltages)


def find_collision_event(voltages, relative_time, threshold=THRESHOLD):
    """Find first collision event. Returns index and time, or None if not found."""
    for i, voltage in enumerate(voltages):
        if abs(voltage) > threshold:
            return i, relative_time[i]
    return None, None


def extract_trial_name(csv_filepath):
    """Extract trial name from CSV filepath. Example: 'data/rover_run_data_0.csv' -> 'rover_run_data_0'"""
    basename = os.path.basename(csv_filepath)
    trial_name = os.path.splitext(basename)[0]
    return trial_name


def generate_timestamp():
    """Generate YYYYMMDD_HHMMSS timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_timeline_figure(relative_time, voltages, collision_idx, collision_time, output_file):
    """Create publication-quality full timeline figure."""
    # Apply publication settings
    rcParams.update(PUBLICATION_RCPARAMS)

    # Create figure with journal-compliant dimensions (single column)
    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # Shade detail region (±300ms window around collision, same as detail figure)
    if collision_idx is not None:
        detail_window_s = 0.3  # 300ms in seconds
        detail_start = collision_time - detail_window_s
        detail_end = collision_time + detail_window_s
        ax.axvspan(detail_start, detail_end, alpha=0.25, color='#AAAAAA', zorder=0)

    # Plot voltage data
    ax.plot(relative_time, voltages, color=COLOR_PRIMARY, linewidth=0.75)

    # Plot threshold line (commented out for cleaner visualization)
    # ax.axhline(y=THRESHOLD, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.5,
    #            alpha=0.7, label='Collision threshold')
    # ax.axhline(y=-THRESHOLD, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.5, alpha=0.7)

    # Mark collision event if found (commented out for cleaner visualization)
    # if collision_idx is not None:
    #     ax.plot(collision_time, voltages[collision_idx], 'o', color=COLOR_PEAK,
    #             markersize=4, zorder=5)

    # Styling
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Voltage (V)', fontsize=8)

    # Grid: Y-axis only, light, minimal
    ax.grid(True, axis='y', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines (Tufte principle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout
    fig.tight_layout()

    # Save at publication quality (300 DPI for images)
    fig.savefig(output_file, dpi=600, bbox_inches='tight', format='png')
    print(f"Full timeline saved to: {output_file}")
    plt.close(fig)


def create_event_detail_figure(relative_time, voltages, collision_idx, collision_time,
                              output_file, window_ms=200):
    """Create publication-quality collision event detail figure (±window_ms)."""
    # Apply publication settings
    rcParams.update(PUBLICATION_RCPARAMS)

    if collision_idx is None:
        print("No collision event found, skipping detail figure")
        return

    # Check data sufficiency
    if len(relative_time) <= 1:
        print("Insufficient data for event detail")
        return

    # Convert window from ms to seconds
    window_s = window_ms / 1000.0

    # Find indices for the window (±window_ms around collision)
    start_time = collision_time - window_s
    end_time = collision_time + window_s

    mask = (relative_time >= start_time) & (relative_time <= end_time)

    if not np.any(mask):
        print(f"No data in ±{window_ms}ms window around collision")
        return

    # Extract windowed data
    windowed_time = relative_time[mask]
    windowed_voltage = voltages[mask]

    # Convert to relative time centered at collision (in ms)
    windowed_time_relative = (windowed_time - collision_time) * 1000  # Convert to ms

    # Create figure
    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # Shade pre-collision region (commented out for cleaner visualization)
    # pre_start_ms = -100
    # pre_end_ms = 0
    # ax.axvspan(pre_start_ms, pre_end_ms, alpha=0.15, color=COLOR_EVENT_PRE, zorder=1)

    # Shade post-collision region (commented out for cleaner visualization)
    # post_start_ms = 0
    # post_end_ms = 100
    # ax.axvspan(post_start_ms, post_end_ms, alpha=0.15, color=COLOR_EVENT_POST, zorder=1)

    # Plot voltage data
    ax.plot(windowed_time_relative, windowed_voltage, color=COLOR_PRIMARY,
            linewidth=0.75, zorder=2)

    # Mark collision peak (commented out for cleaner visualization)
    # peak_idx = np.argmin(np.abs(windowed_time_relative - 0))
    # ax.plot(0, windowed_voltage[peak_idx], 'o', color=COLOR_PEAK,
    #         markersize=5, zorder=5)

    # Styling
    ax.set_xlabel('Time relative to collision (ms)', fontsize=8)
    ax.set_ylabel('Voltage (V)', fontsize=8)

    # Grid: Y-axis only, light, minimal
    ax.grid(True, axis='y', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines (Tufte principle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout
    fig.tight_layout()

    # Save at publication quality
    fig.savefig(output_file, dpi=600, bbox_inches='tight', format='png')
    print(f"Event detail (±{window_ms}ms) saved to: {output_file}")
    plt.close(fig)


def create_multi_timeline_figure(all_trials_data, output_file, window_ms=300):
    """Create publication-quality timeline figure with multiple trials overlaid and aligned at collision."""
    rcParams.update(PUBLICATION_RCPARAMS)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # Plot each trial
    for trial_idx, trial_data in enumerate(all_trials_data):
        relative_time = trial_data['relative_time']
        voltages = trial_data['voltages']
        collision_time = trial_data['collision_time']
        trial_name = trial_data['trial_name']
        color = trial_data['color']

        # Shift time so collision is at t=0 for alignment
        shifted_time = relative_time - collision_time

        # Plot voltage data for this trial
        ax.plot(shifted_time, voltages, color=color, linewidth=0.75, label=trial_name, alpha=0.8)

        # Shade detail region for this trial (±300ms around collision)
        detail_window_s = window_ms / 1000.0
        detail_start = -detail_window_s
        detail_end = detail_window_s
        ax.axvspan(detail_start, detail_end, alpha=0.05, color=color, zorder=0)

    # Styling
    ax.set_xlabel('Time relative to collision (s)', fontsize=8)
    ax.set_ylabel('Voltage (V)', fontsize=8)

    # Grid: Y-axis only, light, minimal
    ax.grid(True, axis='y', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines (Tufte principle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.9, fontsize=7)

    # Tight layout
    fig.tight_layout()

    # Save at publication quality
    fig.savefig(output_file, dpi=600, bbox_inches='tight', format='png')
    print(f"Multi-trial timeline saved to: {output_file}")
    plt.close(fig)


def create_multi_detail_figure(all_trials_data, output_file, window_ms=300):
    """Create publication-quality detail figure with multiple trials overlaid, each centered on its collision.
    Trials are sorted by magnitude (smallest first) for better visibility."""
    rcParams.update(PUBLICATION_RCPARAMS)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # First pass: Calculate magnitude for each trial in the collision window
    for trial_data in all_trials_data:
        relative_time = trial_data['relative_time']
        voltages = trial_data['voltages']
        collision_idx = trial_data['collision_idx']
        collision_time = trial_data['collision_time']
        trial_name = trial_data['trial_name']

        if collision_idx is None:
            trial_data['max_magnitude'] = 0
            continue

        # Extract window around this trial's collision
        window_s = window_ms / 1000.0
        start_time = collision_time - window_s
        end_time = collision_time + window_s

        mask = (relative_time >= start_time) & (relative_time <= end_time)

        if not np.any(mask):
            trial_data['max_magnitude'] = 0
            continue

        # Extract windowed voltage and calculate magnitude
        windowed_voltage = voltages[mask]
        max_magnitude = np.max(np.abs(windowed_voltage))
        trial_data['max_magnitude'] = max_magnitude

    # Sort trials by magnitude (descending - largest first to draw in back, smallest last to draw on top)
    sorted_trials = sorted(all_trials_data, key=lambda x: x['max_magnitude'], reverse=True)

    # Re-label trials as Trial 1, 2, 3... in sorted (magnitude) order
    for idx, trial_data in enumerate(sorted_trials):
        trial_data['trial_name'] = f"Trial {idx + 1}"

    # Plot each trial in magnitude-sorted order (smallest first = on top)
    for trial_idx, trial_data in enumerate(sorted_trials):
        relative_time = trial_data['relative_time']
        voltages = trial_data['voltages']
        collision_idx = trial_data['collision_idx']
        collision_time = trial_data['collision_time']
        trial_name = trial_data['trial_name']
        color = trial_data['color']
        max_magnitude = trial_data['max_magnitude']

        if collision_idx is None:
            continue

        # Extract window around this trial's collision
        window_s = window_ms / 1000.0
        start_time = collision_time - window_s
        end_time = collision_time + window_s

        mask = (relative_time >= start_time) & (relative_time <= end_time)

        if not np.any(mask):
            continue

        # Extract windowed data
        windowed_time = relative_time[mask]
        windowed_voltage = voltages[mask]

        # Convert to relative time centered at collision (in ms)
        windowed_time_relative = (windowed_time - collision_time) * 1000

        # Plot this trial's data
        ax.plot(windowed_time_relative, windowed_voltage, color=color, linewidth=0.75,
                label=trial_name, alpha=0.8)

        # Mark maximum magnitude point with an X in trial color
        max_magnitude_idx = np.argmax(np.abs(windowed_voltage))
        max_magnitude_time = windowed_time_relative[max_magnitude_idx]
        max_magnitude_voltage = windowed_voltage[max_magnitude_idx]
        ax.plot(max_magnitude_time, max_magnitude_voltage, 'x', color=color, markersize=3, zorder=5)

    # Styling
    ax.set_xlabel('Time relative to collision (ms)', fontsize=8)
    ax.set_ylabel('Voltage (V)', fontsize=8)

    # Grid: Y-axis only, light, minimal
    ax.grid(True, axis='y', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines (Tufte principle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='lower left', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.9, fontsize=7)

    # Tight layout
    fig.tight_layout()

    # Save at publication quality
    fig.savefig(output_file, dpi=600, bbox_inches='tight', format='png')
    print(f"Multi-trial collision detail (±{window_ms}ms) saved to: {output_file}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python rover_test_simple_grapher.py <csv_file> [csv_file2] [csv_file3] ...")
        print("  Single file:   python rover_test_simple_grapher.py data/rover_run_data_0.csv")
        print("  Multiple files: python rover_test_simple_grapher.py data/rover_run_data_0.csv data/rover_run_data_1.csv data/rover_run_data_2.csv")
        sys.exit(1)

    csv_files = sys.argv[1:]

    # Create figures directory if it doesn't exist
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    try:
        # Single file mode
        if len(csv_files) == 1:
            print(f"Processing single file: {csv_files[0]}")
            csv_file = csv_files[0]

            # Load data
            timestamps, voltages = load_data(csv_file)

            # Calculate relative time (in seconds from start)
            start_time = timestamps[0]
            relative_time = timestamps - start_time

            # Find collision event
            collision_idx, collision_time = find_collision_event(voltages, relative_time)

            # Print collision information
            if collision_time is not None:
                speed = DISTANCE / collision_time
                print(f"\nCollision Analysis:")
                print(f"  Collision detected at: {collision_time:.3f} seconds")
                print(f"  Distance traveled: {DISTANCE} m")
                print(f"  Average speed: {speed:.3f} m/s")
                print(f"  Voltage at collision: {voltages[collision_idx]:.4f} V")
            else:
                print("No collision detected (voltage never exceeded threshold)")

            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            timeline_output = os.path.join(figures_dir,
                                           f"{base_name}_timeline_publication.png")
            event_output = os.path.join(figures_dir,
                                        f"{base_name}_collision_detail_publication.png")

            # Create figures
            print(f"\nGenerating publication-quality figures...")
            create_timeline_figure(relative_time, voltages, collision_idx, collision_time,
                                 timeline_output)
            create_event_detail_figure(relative_time, voltages, collision_idx, collision_time,
                                      event_output, window_ms=300)

            print(f"\nFigures saved successfully!")

        # Multi-file mode
        else:
            print(f"Processing {len(csv_files)} files...")

            all_trials_data = []

            # Load and process each file
            for file_idx, csv_file in enumerate(csv_files):
                print(f"\n  [{file_idx + 1}/{len(csv_files)}] Loading {csv_file}...")

                # Load data
                timestamps, voltages = load_data(csv_file)

                # Calculate relative time
                start_time = timestamps[0]
                relative_time = timestamps - start_time

                # Find collision event
                collision_idx, collision_time = find_collision_event(voltages, relative_time)

                # Get trial name and color
                trial_name = f"Trial {file_idx + 1}"
                color = TRIAL_COLORS[file_idx % len(TRIAL_COLORS)]

                # Store trial data
                trial_data = {
                    'csv_file': csv_file,
                    'trial_name': trial_name,
                    'trial_number': file_idx + 1,
                    'timestamps': timestamps,
                    'voltages': voltages,
                    'relative_time': relative_time,
                    'collision_idx': collision_idx,
                    'collision_time': collision_time,
                    'color': color
                }
                all_trials_data.append(trial_data)

                # Print info for this trial
                if collision_time is not None:
                    speed = DISTANCE / collision_time
                    print(f"    Collision: {collision_time:.3f}s | Speed: {speed:.3f} m/s | Voltage: {voltages[collision_idx]:.4f}V")
                else:
                    print(f"    No collision detected")

            # Generate timestamped output filenames
            timestamp = generate_timestamp()
            timeline_output = os.path.join(figures_dir, f"merged_{timestamp}_timeline.png")
            detail_output = os.path.join(figures_dir, f"merged_{timestamp}_detail.png")

            # Create multi-trial figures
            print(f"\nGenerating multi-trial publication-quality figures...")
            create_multi_timeline_figure(all_trials_data, timeline_output, window_ms=300)
            create_multi_detail_figure(all_trials_data, detail_output, window_ms=300)

            print(f"\nMulti-trial figures saved successfully!")

            # Print legend information
            print(f"\nTrial Legend:")
            for trial_data in all_trials_data:
                print(f"  {trial_data['trial_name']}: {trial_data['csv_file']}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

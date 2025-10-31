import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ============================================================================
# DATA CONFIGURATION - Edit this section with your data
# ============================================================================

# Define your data as a list of (x, y) tuples
# Example: DATA = [(0, 5), (1, 12), (2, 8), (3, 15)]
DATA = [
    (0.105, 0.131),
    (0.232, 0.392),
    (0.357, 0.550),
]

# ============================================================================
# PLOT CONFIGURATION - Customize your plot here
# ============================================================================

# Axis labels
X_AXIS_LABEL = "Velocity (m/s)"
Y_AXIS_LABEL = "abs(Voltage) (V)"

# Axis ranges (set to None for auto-scaling)
X_MIN = None
X_MAX = None
Y_MIN = None
Y_MAX = None

# Legend and markers
LEGEND_LABEL = "Data Series"
SHOW_MARKERS = True
MARKER_SIZE = 4

# Title and output
TITLE = None  # Set to None to omit title, or use a string like "My Figure"
OUTPUT_FILENAME = "line_graph.png"

# ============================================================================
# PUBLICATION-QUALITY MATPLOTLIB CONFIGURATION (from rover_test_simple_grapher)
# ============================================================================

PUBLICATION_RCPARAMS = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'font.family': 'sans-serif',
    'lines.linewidth': 0.75,
    'lines.markersize': MARKER_SIZE,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'text.usetex': False,
}

# Professional color palette (colorblind-friendly)
COLOR_PRIMARY = '#0173B2'  # Blue for primary data


def create_line_figure(x_data, y_data, x_label, y_label, x_min=None, x_max=None,
                       y_min=None, y_max=None, legend_label=None, show_markers=True,
                       title=None, output_file="line_graph.png"):
    """
    Create a publication-quality line graph.

    Parameters:
    -----------
    x_data : array-like
        X-axis data points
    y_data : array-like
        Y-axis data points
    x_label : str
        Label for X-axis
    y_label : str
        Label for Y-axis
    x_min, x_max : float or None
        X-axis range (None for auto-scale)
    y_min, y_max : float or None
        Y-axis range (None for auto-scale)
    legend_label : str or None
        Label for the data series (for legend)
    show_markers : bool
        Whether to show markers at data points
    title : str or None
        Figure title (None to omit)
    output_file : str
        Output filename for PNG
    """
    # Apply publication settings
    rcParams.update(PUBLICATION_RCPARAMS)

    # Create figure with journal-compliant dimensions (single column)
    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    # Convert data to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Plot data
    if show_markers:
        ax.plot(x_data, y_data, color=COLOR_PRIMARY, linewidth=0.75,
                marker='o', markersize=4, label=legend_label, zorder=2)
    else:
        ax.plot(x_data, y_data, color=COLOR_PRIMARY, linewidth=0.75,
                label=legend_label, zorder=2)

    # Set axis labels
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)

    # Set axis ranges if specified
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    # Grid: Y-axis only, light, minimal
    ax.grid(True, axis='y', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines (Tufte principle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend if label provided
    if legend_label is not None:
        ax.legend(loc='best', frameon=True, fancybox=False,
                  edgecolor='gray', framealpha=0.9, fontsize=7)

    # Add title if provided
    if title is not None:
        ax.set_title(title, fontsize=8, pad=5)

    # Tight layout
    fig.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save at publication quality (600 DPI for images)
    fig.savefig(output_file, dpi=600, bbox_inches='tight', format='png')
    print(f"Figure saved to: {output_file}")
    plt.close(fig)


def main():
    """Main entry point - creates figure from configured data."""

    if not DATA:
        print("Error: No data provided. Edit the DATA variable at the top of the script.")
        return

    # Extract x and y from data tuples
    x_values = [point[0] for point in DATA]
    y_values = [point[1] for point in DATA]

    print(f"Plotting {len(DATA)} data points...")
    print(f"X range: {min(x_values):.3f} to {max(x_values):.3f}")
    print(f"Y range: {min(y_values):.3f} to {max(y_values):.3f}")

    # Create the figure
    create_line_figure(
        x_data=x_values,
        y_data=y_values,
        x_label=X_AXIS_LABEL,
        y_label=Y_AXIS_LABEL,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        legend_label=LEGEND_LABEL,
        show_markers=SHOW_MARKERS,
        title=TITLE,
        output_file=OUTPUT_FILENAME
    )

    print("Done!")


if __name__ == '__main__':
    main()

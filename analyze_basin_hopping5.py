#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2025-, Huu Trong Phan (phanhuutrong93@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20250415 21:34:26
# ==============================================================
#!/usr/bin/env python3
"""
Basin Hopping Energy Analysis Tool

Analyzes energy distributions from Basin Hopping simulations, creating histograms
of accepted structures and overall energy landscape with relative energies in kJ/mol.
"""

import os
import glob
import logging
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator

# Constants
HARTREE_TO_KJ_MOL = 2625.5  # Conversion factor: 1 Hartree = 2625.5 kJ/mol
MAX_BINS = 1000  # Maximum number of bins to prevent overflow
DEFAULT_BINS = 50  # Default number of bins if automatic calculation fails

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basin_hopping_analysis')


def extract_energies_from_file(filepath):
    """
    Extract energy values from an XYZ file using ASE.

    Args:
        filepath: Path to the XYZ file

    Returns:
        List of energy values in Hartree
    """
    energies = []

    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return energies

    try:
        # Read all frames in the XYZ file
        frames = read(filepath, index=':')

        for frame in frames:
            if 'eng' in frame.info:
                energy = frame.info['eng']
                # Filter out invalid energy values
                if np.isfinite(energy):
                    energies.append(energy)
                else:
                    logger.warning(f"Skipping non-finite energy value: {energy}")
            else:
                logger.warning(f"No energy information found in frame from {filepath}")

        logger.info(f"Extracted {len(energies)} valid energy values from {filepath}")
        return energies

    except Exception as e:
        logger.error(f"Error extracting energies from {filepath}: {e}")
        return []


def analyze_run_directory(run_dir):
    """
    Analyze a single Basin Hopping run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dictionary containing accepted energies, all energies, and step count
    """
    results = {
        'accepted_energies': [],
        'rejected_energies': [],
        'all_energies': [],
        'step_count': 0,
        'directory': run_dir
    }

    try:
        # Define file paths
        accepted_file = os.path.join(run_dir, 'xyz_opt', 'accepted_structure.xyz')
        rejected_file = os.path.join(run_dir, 'xyz_opt', 'rejected_structure.xyz')

        # Extract energies from accepted structures
        accepted_energies = extract_energies_from_file(accepted_file)
        results['accepted_energies'] = accepted_energies
        results['all_energies'].extend(accepted_energies)

        # Extract energies from rejected structures if available
        rejected_energies = extract_energies_from_file(rejected_file)
        results['rejected_energies'] = rejected_energies
        results['all_energies'].extend(rejected_energies)

        # Calculate total step count
        results['step_count'] = len(results['all_energies'])
        logger.info(f"Directory {run_dir}: {len(accepted_energies)} accepted, {len(rejected_energies)} rejected, {results['step_count']} total")

        return results

    except Exception as e:
        logger.error(f"Error analyzing directory {run_dir}: {e}")
        return results


def get_safe_bin_count(min_val, max_val, bin_width):
    """
    Calculate a safe number of bins for histogram, handling edge cases.

    Args:
        min_val: Minimum value in the data
        max_val: Maximum value in the data
        bin_width: Desired bin width

    Returns:
        int: Number of bins to use
    """
    try:
        # Handle case where min and max are the same
        if np.isclose(min_val, max_val):
            logger.warning("Minimum and maximum energy values are nearly identical. Using default bin count.")
            return DEFAULT_BINS

        # Calculate range and ensure it's positive
        data_range = max_val - min_val
        if data_range <= 0:
            logger.warning(f"Invalid data range: {data_range}. Using default bin count.")
            return DEFAULT_BINS

        # Ensure bin_width is reasonable
        if bin_width <= 0:
            logger.warning(f"Invalid bin width: {bin_width}. Using 2.0 kJ/mol instead.")
            bin_width = 2.0

        # Calculate number of bins
        num_bins = int(data_range / bin_width) + 1

        # Cap at maximum bins
        if num_bins > MAX_BINS:
            logger.warning(f"Calculated {num_bins} bins exceeds maximum. Capping at {MAX_BINS}.")
            return MAX_BINS

        return num_bins

    except (OverflowError, ValueError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating bin count: {e}. Using default bin count.")
        return DEFAULT_BINS


def plot_energy_histogram(all_results, output_dir, reference_energy=None, bin_width=2.0):
    """
    Create a histogram plot of energies showing both accepted and total distributions.

    Args:
        all_results: List of result dictionaries from each run
        output_dir: Directory to save plots
        reference_energy: Energy value to use as zero point (in Hartree)
        bin_width: Width of histogram bins in kJ/mol
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    all_accepted = []
    all_total = []

    for run_results in all_results:
        all_accepted.extend(run_results['accepted_energies'])
        all_total.extend(run_results['all_energies'])

    if not all_accepted or not all_total:
        logger.warning("No energy data to plot")
        return

    # Filter out any non-finite values to be safe
    all_accepted = [e for e in all_accepted if np.isfinite(e)]
    all_total = [e for e in all_total if np.isfinite(e)]

    if not all_accepted or not all_total:
        logger.error("No finite energy values found")
        return

    # Use the minimum energy as reference if none provided
    if reference_energy is None:
        reference_energy = min(all_total)
        logger.info(f"Using minimum energy as reference: {reference_energy} Hartree")

    # Convert energies to relative values in kJ/mol
    relative_accepted = [(e - reference_energy) * HARTREE_TO_KJ_MOL for e in all_accepted]
    relative_total = [(e - reference_energy) * HARTREE_TO_KJ_MOL for e in all_total]

    # Calculate histogram range
    min_energy = min(relative_total)
    max_energy = max(relative_total)

    # Get safe bin count
    #num_bins = get_safe_bin_count(min_energy, max_energy, bin_width)
    bins = np.arange(0, 200+2, 2)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot histograms
    plt.hist(relative_total, bins=bins, alpha=0.5, color='blue',
             edgecolor='black', label='All Structures')
    plt.hist(relative_accepted, bins=bins, alpha=0.7, color='red',
             edgecolor='black', label='Accepted Structures')

    # Add vertical line at the minimum energy
    plt.axvline(x=min_energy, color='green', linestyle='--', linewidth=2,
                label=f'Lowest Energy: {min_energy:.2f} kJ/mol')

    # Add text annotation for the minimum energy
    y_max = plt.gca().get_ylim()[1]
    plt.xlim(0, 200)
    plt.annotate(f'Global Min: {min_energy:.2f} kJ/mol',
                 xy=(min_energy, y_max * 0.95),
                 xytext=(min_energy + (max_energy - min_energy) * 0.05, y_max * 0.95),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    # Add labels and title
    plt.xlabel('Relative Energy (kJ/mol)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Energy Distribution of Basin Hopping Structures', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Ensure integer values on y-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Save figure
    output_path = os.path.join(output_dir, 'energy_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Energy histogram saved to {output_path}")
    plt.close()

    # Save statistics to file
    stats_path = os.path.join(output_dir, 'energy_stats.txt')
    try:
        with open(stats_path, 'w') as f:
            f.write(f"Total structures: {len(relative_total)}\n")
            f.write(f"Accepted structures: {len(relative_accepted)}\n")
            f.write(f"Acceptance ratio: {len(relative_accepted)/len(relative_total):.2f}\n\n")

            f.write(f"Reference energy: {reference_energy} Hartree\n\n")

            f.write("Statistics (kJ/mol):\n")
            f.write(f"Min energy: {min_energy:.2f}\n")
            f.write(f"Max energy: {max_energy:.2f}\n")
            f.write(f"Energy range: {max_energy - min_energy:.2f}\n")

            # Add global minimum structure info
            f.write(f"\nGlobal minimum found: {min_energy:.2f} kJ/mol\n")

            # Add info about accepted structures
            accepted_min = min(relative_accepted)
            f.write(f"Lowest accepted structure: {accepted_min:.2f} kJ/mol\n")

        logger.info(f"Energy statistics saved to {stats_path}")
    except Exception as e:
        logger.error(f"Error writing energy statistics: {e}")


def plot_steps_sorted(all_results, output_dir):
    """
    Create a connected scatter plot of step counts sorted in increasing order.

    Args:
        all_results: List of result dictionaries from each run
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract step counts and directory names
    step_data = [(results['directory'], results['step_count']) for results in all_results]

    if not step_data:
        logger.warning("No step data to plot")
        return

    # Calculate total steps across all runs
    total_steps = sum(item[1] for item in step_data)

    # Sort by step count (ascending)
    step_data.sort(key=lambda x: x[1])

    # Separate into x and y data for plotting
    dir_names = [os.path.basename(item[0]) for item in step_data]
    step_counts = [item[1] for item in step_data]

    # Create indices for x-axis
    x_indices = np.arange(len(step_counts))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot connected scatter points
    ax.plot(x_indices, step_counts, 'o-', color='purple', markersize=8, linewidth=2)
    ax.grid(True, alpha=0.3)

    # Add labels and title
    ax.set_title('Basin Hopping Step Counts (Sorted)', fontsize=14)
    ax.set_xlabel('Run Index (sorted by increasing step count)', fontsize=12)
    ax.set_ylabel('Number of Steps', fontsize=12)
    ax.set_ylim(0, 5000)

    # Add data labels if there aren't too many points
    if len(step_counts) <= 20:
        for i, count in enumerate(step_counts):
            ax.annotate(f"{count}",
                      (x_indices[i], step_counts[i]),
                      textcoords="offset points",
                      xytext=(0, 10),
                      ha='center')

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Ensure integer values on y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add text for total steps
    # Place it in the upper left corner with a background box
    ax.text(0.02, 0.95, f'Total Steps: {total_steps:,}',
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
            verticalalignment='top')

    # Add text for number of runs
    ax.text(0.02, 0.88, f'Number of Runs: {len(step_counts)}',
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
            verticalalignment='top')

    # Add text for mean steps per run
    mean_steps = total_steps / len(step_counts) if step_counts else 0
    ax.text(0.02, 0.81, f'Mean Steps/Run: {mean_steps:.1f}',
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
            verticalalignment='top')

    # Save figure
    output_path = os.path.join(output_dir, 'steps_sorted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Step plot saved to {output_path}")

    # Save detailed step data to CSV
    try:
        csv_path = os.path.join(output_dir, 'step_counts.csv')
        with open(csv_path, 'w') as f:
            f.write("Directory,StepCount\n")
            for directory, count in step_data:
                f.write(f"{directory},{count}\n")
            # Add summary statistics
            f.write("\nSummary Statistics\n")
            f.write(f"Total Steps,{total_steps}\n")
            f.write(f"Number of Runs,{len(step_counts)}\n")
            f.write(f"Mean Steps/Run,{mean_steps:.1f}\n")
            f.write(f"Min Steps,{min(step_counts) if step_counts else 0}\n")
            f.write(f"Max Steps,{max(step_counts) if step_counts else 0}\n")

        logger.info(f"Step data saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error writing step count data: {e}")

    plt.close()


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze Basin Hopping results')
    parser.add_argument('-d', '--data-dir', default='.',
                        help='Base directory containing multiple run folders')
    parser.add_argument('-o', '--output-dir', default='./basin_hopping_analysis',
                        help='Directory to save output plots')
    parser.add_argument('-p', '--pattern', default='*',
                        help='Pattern to match run directories (e.g., "run_*")')
    parser.add_argument('-r', '--reference', type=float,
                        help='Reference energy in Hartree to use as zero point')
    parser.add_argument('-b', '--bin-width', type=float, default=2.0,
                        help='Width of histogram bins in kJ/mol (default: 2.0)')
    args = parser.parse_args()

    # Get list of run directories
    base_dir = Path(args.data_dir)
    run_pattern = os.path.join(base_dir, args.pattern)
    run_dirs = sorted(glob.glob(run_pattern))

    if not run_dirs:
        logger.error(f"No directories found matching pattern {run_pattern}")
        return

    logger.info(f"Found {len(run_dirs)} run directories to analyze")

    # Analyze each run directory
    all_results = []
    for run_dir in run_dirs:
        if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, 'xyz_opt')):
            results = analyze_run_directory(run_dir)
            if results['step_count'] > 0:
                all_results.append(results)

    if not all_results:
        logger.error("No valid results were collected")
        return

    # Create plots
    try:
        plot_energy_histogram(all_results, args.output_dir, args.reference, args.bin_width)
    except Exception as e:
        logger.error(f"Error creating energy histogram: {e}")

    try:
        plot_steps_sorted(all_results, args.output_dir)
    except Exception as e:
        logger.error(f"Error creating step plot: {e}")

    # Log summary
    total_steps = sum(result['step_count'] for result in all_results)
    total_accepted = sum(len(result['accepted_energies']) for result in all_results)
    logger.info(f"Analysis completed: {len(all_results)} runs, {total_steps} total steps, {total_accepted} accepted structures")


if __name__ == "__main__":
    main()

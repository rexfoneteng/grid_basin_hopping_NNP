#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-05-13 16:48:20
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-13 16:58:33
# @Description: Basin Hopping Analysis Tool
"""
This script analyzes the performance of Basin Hopping optimization runs by collecting
step information from multiple job directories, generating histograms, and providing 
statistical insights into the optimization process.

Usage:
    python analyze_basin_hopping.py [--output_dir OUTPUT_DIR] [--results_dir RESULTS_DIR]
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Basin Hopping step distributions")
    parser.add_argument("--output_dir", default="output", 
                        help="Directory containing job outputs (default: output)")
    parser.add_argument("--results_dir", default="analysis_results",
                        help="Directory to save analysis results (default: analysis_results)")
    return parser.parse_args()


def collect_step_data(output_dir):
    """
    Collect step data from all job directories.
    
    Args:
        output_dir: Path to directory containing job outputs
        
    Returns:
        DataFrame with collected step data
    """
    print(f"Collecting data from {output_dir}...")
    
    job_dirs = sorted(glob.glob(os.path.join(output_dir, "job_*")))
    if not job_dirs:
        raise ValueError(f"No job directories found in {output_dir}")
    
    data = []
    
    for job_dir in job_dirs:
        job_id = os.path.basename(job_dir)
        stats_file = os.path.join(job_dir, "statistics.json")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                
                # Extract step information
                total_steps = stats.get("accepted_steps", 0) + stats.get("rejected_steps", 0)
                
                # Determine if the job completed
                completed = bool(stats.get("stopping_reason"))
                
                # Extract grid and base names from job_id
                parts = job_id.split("_")
                if len(parts) >= 3:
                    # Assuming format: job_base#_grid_name
                    base_part = parts[1] if parts[1].startswith("part") else "unknown"
                    grid_name = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
                else:
                    base_part = "unknown"
                    grid_name = "unknown"
                
                # Collect data
                data.append({
                    "job_id": job_id,
                    "total_steps": total_steps,
                    "accepted_steps": stats.get("accepted_steps", 0),
                    "rejected_steps": stats.get("rejected_steps", 0),
                    "acceptance_ratio": stats.get("accepted_steps", 0) / max(1, total_steps),
                    "best_energy": stats.get("best_energy", float("nan")),
                    "completed": completed,
                    "base_part": base_part,
                    "grid_name": grid_name
                })
                
            except Exception as e:
                print(f"Error processing {stats_file}: {str(e)}")
    
    print(f"Collected data from {len(data)} jobs")
    return pd.DataFrame(data)


def analyze_step_distribution(df, results_dir):
    """
    Analyze the distribution of steps and create visualizations.
    
    Args:
        df: DataFrame with step data
        results_dir: Directory to save results
    """
    # Calculate statistics for completed jobs
    completed_df = df[df["completed"]]
    if len(completed_df) > 0:
        step_stats = {
            "count": len(completed_df),
            "mean": completed_df["total_steps"].mean(),
            "median": completed_df["total_steps"].median(),
            "min": completed_df["total_steps"].min(),
            "max": completed_df["total_steps"].max(),
            "std": completed_df["total_steps"].std(),
            "acceptance_ratio_mean": completed_df["acceptance_ratio"].mean()
        }
    else:
        step_stats = {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "acceptance_ratio_mean": np.nan
        }
    
    # Create step distribution histogram using matplotlib directly
    plt.figure(figsize=(10, 6))
    
    if len(completed_df) > 0:
        # Plot histogram with simple settings
        n, bins, patches = plt.hist(completed_df["total_steps"].values, bins=30, 
                                   alpha=0.7, color='blue', edgecolor='black')
        
        # Add vertical lines for important statistics
        plt.axvline(x=step_stats["median"], color='red', linestyle='--', 
                   label=f'Median: {step_stats["median"]:.0f}')
        plt.axvline(x=step_stats["mean"], color='green', linestyle='-', 
                   label=f'Mean: {step_stats["mean"]:.0f}')
        plt.axvline(x=step_stats["max"], color='purple', linestyle='-.', 
                   label=f'Max: {step_stats["max"]:.0f}')
    
    # Add labels and title
    plt.xlabel("Number of Steps")
    plt.ylabel("Frequency")
    plt.title("Basin Hopping Step Distribution")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics text box
    if len(completed_df) > 0:
        stats_text = (
            f"Completed Jobs: {step_stats['count']}\n"
            f"Mean Steps: {step_stats['mean']:.1f}\n"
            f"Median Steps: {step_stats['median']:.1f}\n"
            f"Min Steps: {step_stats['min']:.0f}\n"
            f"Max Steps: {step_stats['max']:.0f}\n"
            f"Std Dev: {step_stats['std']:.1f}\n"
            f"Avg. Acceptance Ratio: {step_stats['acceptance_ratio_mean']:.2f}"
        )
        
        # Simplified annotation
        plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="white", alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(results_dir, "step_distribution.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Step distribution histogram saved to {output_file}")
    
    return step_stats


def analyze_by_group(df, results_dir):
    """
    Analyze step distributions by grid and base groups.
    
    Args:
        df: DataFrame with step data
        results_dir: Directory to save results
    """
    completed_df = df[df["completed"]]
    if len(completed_df) < 5:  # Need enough data to make meaningful plots
        print("Not enough completed jobs for group analysis")
        return
    
    # Function to create simplified boxplot
    def create_boxplot(data, x_col, y_col, title, output_file):
        plt.figure(figsize=(12, 6))
        
        # Get unique groups and sort them
        groups = sorted(data[x_col].unique())
        
        # Prepare data for boxplot
        group_data = []
        for group in groups:
            group_data.append(data[data[x_col] == group][y_col].values)
        
        # Create box plot
        plt.boxplot(group_data, labels=groups)
        
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Analysis saved to {output_file}")
    
    # Analyze by grid name
    create_boxplot(
        completed_df, 
        "grid_name", 
        "total_steps", 
        "Step Distribution by Proton Grid Type",
        os.path.join(results_dir, "steps_by_grid.png")
    )
    
    # Analyze by base part
    create_boxplot(
        completed_df, 
        "base_part", 
        "total_steps", 
        "Step Distribution by Base Structure Group",
        os.path.join(results_dir, "steps_by_base.png")
    )


def save_summary_report(df, step_stats, results_dir):
    """
    Save a summary report of the analysis.
    
    Args:
        df: DataFrame with step data
        step_stats: Dictionary with step statistics
        results_dir: Directory to save results
    """
    completed_df = df[df["completed"]]
    total_jobs = len(df)
    completed_jobs = len(completed_df)
    
    report_file = os.path.join(results_dir, "analysis_summary.txt")
    
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"BASIN HOPPING ANALYSIS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("JOB STATISTICS:\n")
        f.write(f"Total Jobs: {total_jobs}\n")
        f.write(f"Completed Jobs: {completed_jobs} ({completed_jobs/max(1, total_jobs)*100:.1f}%)\n\n")
        
        if completed_jobs > 0:
            f.write("STEP STATISTICS (Completed Jobs Only):\n")
            f.write(f"Mean Steps: {step_stats['mean']:.1f}\n")
            f.write(f"Median Steps: {step_stats['median']:.1f}\n")
            f.write(f"Minimum Steps: {step_stats['min']:.0f}\n")
            f.write(f"Maximum Steps: {step_stats['max']:.0f}\n")
            f.write(f"Standard Deviation: {step_stats['std']:.1f}\n")
            f.write(f"Average Acceptance Ratio: {step_stats['acceptance_ratio_mean']:.2f}\n\n")
            
            # Add performance insights
            f.write("PERFORMANCE INSIGHTS:\n")
            if step_stats["std"] / step_stats["mean"] > 0.5:
                f.write("- High variation in step counts suggests inconsistent convergence behavior\n")
            else:
                f.write("- Consistent step counts across jobs indicate stable convergence\n")
                
            if step_stats["acceptance_ratio_mean"] < 0.2:
                f.write("- Low acceptance ratio may indicate high temperature or rough energy landscape\n")
            elif step_stats["acceptance_ratio_mean"] > 0.5:
                f.write("- High acceptance ratio may indicate low temperature or smooth energy landscape\n")
            
            # Add top performers
            f.write("\nTOP PERFORMING JOBS (BY LOWEST ENERGY):\n")
            top_jobs = completed_df.nsmallest(5, "best_energy")[["job_id", "best_energy", "total_steps"]]
            for idx, row in top_jobs.iterrows():
                f.write(f"- {row['job_id']}: Energy = {row['best_energy']:.6f}, Steps = {row['total_steps']:.0f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("GENERATED VISUALIZATIONS:\n")
        f.write("1. step_distribution.png - Histogram of basin hopping steps\n")
        f.write("2. steps_by_grid.png - Box plot of steps by proton grid type\n")
        f.write("3. steps_by_base.png - Box plot of steps by base structure group\n")
        
    print(f"Summary report saved to {report_file}")


def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Collect data
    try:
        df = collect_step_data(args.output_dir)
        
        # Analyze step distribution
        step_stats = analyze_step_distribution(df, args.results_dir)
        
        # Analyze by groups
        analyze_by_group(df, args.results_dir)
        
        # Save summary report
        save_summary_report(df, step_stats, args.results_dir)
        
        print(f"Analysis complete. Results saved to {args.results_dir}")
        
        # Print key statistics to console
        completed_jobs = len(df[df["completed"]])
        if completed_jobs > 0:
            print("\nKEY STATISTICS:")
            print(f"Median Steps: {step_stats['median']:.0f}")
            print(f"Maximum Steps: {step_stats['max']:.0f}")
            print(f"Mean Steps: {step_stats['mean']:.1f} ± {step_stats['std']:.1f}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
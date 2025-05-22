#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-17 16:34:44
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-21 16:16:51
# @Description: Utility functions for basin hopping simulations.

import os
import json
import logging
from utils.config_utils import parse_operation_sequence

logger = logging.getLogger(__name__)

def prepare_basin_hopping(config):
    """
    Prepare the environment and parameters for basin hopping.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of prepared parameters for basin hopping
    """
    # Setup directories
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    trajectories_dir = os.path.join(output_dir, "trajectories")
    if config["save_trajectories"]:
        os.makedirs(trajectories_dir, exist_ok=True)
        logger.info(f"Created trajectories directory: {trajectories_dir}")

    # Parse operations sequence
    operation_sequence = parse_operation_sequence(config["operation_sequence"])

    # Prepare optimizer parameters
    optimizer_params = config["optimizer"].get("params", {})
    optimizer_type = config["optimizer"]["type"]

    accepted_xyz = os.path.join(output_dir, "accepted_structure.xyz")
    rejected_xyz = os.path.join(output_dir, "rejected_structure.xyz")
    stats_file = os.path.join(output_dir, "statistics.json")

    return {
        "output_dir": output_dir,
        "trajectories_dir": trajectories_dir,
        "operation_sequence": operation_sequence,
        "operation_params": config["operation_params"],
        "optimizer_type": optimizer_type,
        "optimizer_params": optimizer_params,
        "optimize_params": config["optimizer"]["params"],
        "accepted_xyz": accepted_xyz,
        "rejected_xyz": rejected_xyz,
        "stats_file": stats_file,
        "physical_check": {
            "enabled": config["physical_check"]["enabled"],
            "params": config["physical_check"]["params"]
        },
        "attach_rotate_grid": config.get("attach_rotate_grid"),
        "proton_grid": config.get("proton_grid"),
        "skip_local_optimization": config.get("skip_local_optimization")
    }

def save_results(stats_file, stats, best_energy, output_xyz):
    """
    Save basin hopping results to file and log summary.
    
    Args:
        stats_file: Path to save statistics JSON
        stats: Basin hopping statistics dictionary
        best_energy: Best energy found
        output_xyz: Path where structures were saved
    """
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Log summary
    logger.info(f"Basin hopping completed: {stats['stopping_reason']}")
    logger.info(f"Best energy found: {best_energy:.6f}")
    logger.info(f"Total steps: {stats['total_steps']}")
    logger.info(f"Accepted steps: {stats['accepted_steps']}")
    logger.info(f"Rejected steps: {stats['rejected_steps']}")
    logger.info(f"Maximum consecutive rejections: {stats['max_consecutive_rejections']}")
    logger.info(f"Duration: {stats['duration']:.2f} seconds")
    logger.info(f"All structures have been saved to: {output_xyz}")
    logger.info(f"Statistics saved to: {stats_file}")

def get_optimizer_summary(stats):
    """
    Generate a summary of optimization statistics.
    
    Args:
        stats: Basin hopping statistics dictionary
        
    Returns:
        String containing optimization summary
    """
    summary = []
    summary.append(f"Basin Hopping Optimization Summary")
    summary.append(f"===================================")
    summary.append(f"Duration: {stats['duration']:.2f} seconds")
    summary.append(f"Total steps: {stats['total_steps']}")
    summary.append(f"Accepted moves: {stats['accepted_steps']}")
    summary.append(f"Rejected moves: {stats['rejected_steps']}")
    
    # Calculate acceptance ratio, avoiding division by zero
    total_attempts = max(1, stats["total_steps"])
    acceptance_ratio = stats["accepted_steps"] / total_attempts
    summary.append(f"Acceptance ratio: {acceptance_ratio:.2f}")
    
    summary.append(f"Stopping reason: {stats['stopping_reason']}")
    summary.append(f"Best energy found: {stats['best_energy']:.6f}")
    summary.append(f"Optimization failures: {stats['optimization_failures']}")
    summary.append(f"Generation failures: {stats['generation_failures']}")
    
    return "\n".join(summary)
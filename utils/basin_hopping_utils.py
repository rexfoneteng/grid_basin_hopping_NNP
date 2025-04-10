#!/usr/bin/env python3
"""
Utility functions for basin hopping simulations.
"""
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

    if config["save_trajectories"]:
        trajectories_dir = "xyz_traj"
        if not os.path.exists(trajectories_dir):
            os.makedirs(trajectories_dir, exist_ok=True)
            logger.info(f"Created trajectories directory: {trajectories_dir}")

    # Parse operations sequence
    operation_sequence = parse_operation_sequence(config["operations"])

    # Prepare model parameters
    model_params = {
        "state_dict": config['model']['state_dict'],
        "prop_stats": config['model']['prop_stats'],
        "device": config['model']['device'],
        "in_module": config['model']['in_module'],
        "interface_params": config['model']['interface_params']
    }

    accepted_xyz = os.path.join(output_dir, "accepted_structure.xyz")
    rejected_xyz = os.path.join(output_dir, "rejected_structure.xyz")
    stats_file = os.path.join(output_dir, "statistics.json")

    return {
        'output_dir': output_dir,
        'trajectories_dir': trajectories_dir,
        'operation_sequence': operation_sequence,
        'model_params': model_params,
        'optimize_params': config['optimization'],
        'accepted_xyz': accepted_xyz,
        'rejected_xyz': rejected_xyz,
        'stats_file': stats_file,
        'physical_check': {
            'enabled': config['physical_check']['enabled'],
            'params': config['physical_check']['params']
        },
        'flip_grid': config.get('flip_grid'),
        'attach_rotate_grid': config.get('attach_rotate_grid'),
        'proton_grid': config.get('proton_grid')
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

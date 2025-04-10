#!/usr/bin/env python3
"""
Basin Hopping - Structure Optimization with Neural Network Potentials
"""
import os
import sys
import argparse
import logging

from basin_hopping.basin_hopping_generator import BasinHoppingGenerator
from utils.logging_utils import setup_logger
from utils.config_utils import load_config, merge_cli_with_config, validate_config
from utils.basin_hopping_utils import prepare_basin_hopping, save_results


def parse_arguments():
    """Parse command-line arguments for basin hopping."""
    parser = argparse.ArgumentParser(
        description="Basin Hopping Optimization with Neural Network Potentials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--base-structures", "-b",
        nargs="+",
        help="List of base structure XYZ files"
    )
    
    parser.add_argument(
        "--seed-structure", "-s",
        help="Seed structure XYZ file for attachment operations"
    )
    
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=["flip", "attach_rotate", "add_proton"],
        help="List of operations to perform in sequence"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        help="Temperature for Metropolis criterion (K)"
    )
    
    parser.add_argument(
        "--steps", "-n",
        type=int,
        help="Number of basin hopping steps to perform"
    )
    
    parser.add_argument(
        "--max-rejected",
        type=int,
        help="Maximum consecutive rejected moves before stopping"
    )
    
    parser.add_argument(
        "--model-state",
        help="Path to NNP model state dictionary"
    )
    
    parser.add_argument(
        "--prop-stats",
        help="Path to property statistics file for NNP model"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run NNP model on"
    )
    
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save optimization trajectories"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )

    parser.add_argument(
        "--export-config",
        help="Export default configuration to specified YAML file and exit"
    )
    
    return parser.parse_args()

def run_basin_hopping(config):
    """
    Run basin hopping with the provided configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Prepare parameters
    params = prepare_basin_hopping(config)

    try:
        logger.info("Initializing basin hopping generator...")
        bh_generator = BasinHoppingGenerator(
            base_structures=config['base_structures'],
            seed_structure=config['seed_structure'],
            temperature=config['temperature'],
            check_physical=params['physical_check']['enabled'],
            check_physical_kwargs=params['physical_check']['params'],
            model_params=params['model_params'],
            optimize_params=params['optimize_params'],
            accepted_xyz=params['accepted_xyz'],
            rejected_xyz=params['rejected_xyz'],
            max_rejected=config['max_rejected'],
            operation_sequence=params['operation_sequence'],
            save_trj=config['save_trajectories'],
            trajectory_dir=params['trajectories_dir']
        )

        if params["flip_grid"]:
            bh_generator.set_flip_angles(params["flip_grid"])
            logger.info(f"Configured {len(params['flip_grid'])} flip grid points")

        if params["attach_rotate_grid"]:
            bh_generator.set_attach_angles(params["attach_rotate_grid"])
            logger.info(f"Configured {len(params['attach_rotate_grid'])} attach-rotate grid points")

        if params["proton_grid"]:
            params["proton_grid"] = [tuple(ele) for ele in params["proton_grid"]]
            bh_generator.set_proton_grid(params["proton_grid"])
            logger.info(f"Configured {len(params['proton_grid'])} proton grid points")

        # Run basin hopping
        logger.info(f"Starting basin hopping with {config['steps']} steps...")
        best_structure = bh_generator(n_steps=config['steps'])

        # Save results
        save_results(params["stats_file"], bh_generator.get_stats(),
                     bh_generator.best_energy, params["accepted_xyz"])

        return 0

    except Exception as e:
        logger.exception(f"Error during basin hopping: {e}")
        return 1

def main():
    """Main entry point for the basin hopping CLI."""
    global logger

    # Parse args
    args = parse_arguments()

    # Handle config export if requested
    if args.export_config:
        # Setup basic logging for the export operation
        logger = setup_logger(name="basin_hopping", level="INFO")
        from utils.config_utils import export_default_config
        success = export_default_config(args.export_config)
        return 0 if success else 1

    #Load config
    config = load_config(args.config)

    # Merge with CLI args
    config = merge_cli_with_config(args, config)

    # Setup logging
    log_file = config.get("logging", {}).get("file")
    log_level = config.get("logging", {}).get("level", "INFO")
    logger = setup_logger(name="basin_hopping", level=log_level, 
                          log_file=log_file)

    # Validate config
    if not validate_config(config):
        logger.error("Config validation failed")
        return 1

    # Run  basin hopping
    return run_basin_hopping(config)


if __name__ == "__main__":
    sys.exit(main())

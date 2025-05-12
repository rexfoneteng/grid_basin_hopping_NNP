#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-11 20:52:30
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-04-16 17:46:13
# @Description: Configuration utilities for basin hopping simulations. Handles loading, validation, and management of YAML configurations.

import os
import yaml
import logging
import yaml

from basin_hopping.operation_type import OperationType
from core.constants import PHYSICAL_DICT
from utils.file_utils import resolve_structures

logger = logging.getLogger(__name__)

def load_config(config_path=None):
    """
    Load configuration from YAML file with sensible defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    # Default configuration
    config = {
        'output_dir': 'basin_hopping_output',
        'base_structures': "./",
        'seed_structures': None,
        'temperature': 300.0,
        'max_rejected': 50,
        'save_trajectories': False,
        'steps': 100,
        'operations': ['flip', 'attach_rotate', 'add_proton'],
        'physical_check': {
            'enabled': True,
            'params': PHYSICAL_DICT
        }, 
        'attach_rotate_grid': [0, 60, 120, 180, 240, 300],
        'logging': {
            'level': 'INFO',
            'file': 'basin_hopping.log'
        }
    }
    # Load and merge with file configuration if provided
    if config_path:
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    deep_update(config, file_config)
                    logger.info(f"Configuration loaded from {config_path}")
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")

    # Resolve base_structures
    if config["base_structures"]:
        config["base_structures"] = resolve_structures(config["base_structures"])

    if config["seed_structures"]:
        config["seed_structures"] = resolve_structures(config["seed_structures"])

    return config

def deep_update(base_dict, update_dict):
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with values to update base_dict
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def merge_cli_with_config(args, config):
    """
    Merge command-line arguments with configuration.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # Update config with command-line args if specified
    if hasattr(args, 'output_dir') and args.output_dir:
        config['output_dir'] = args.output_dir
    
    if hasattr(args, 'base_structures') and args.base_structures:
        config['base_structures'] = args.base_structures
    
    if hasattr(args, 'seed_structures') and args.seed_structures:
        config['seed_structures'] = args.seed_structures
    
    if hasattr(args, 'operations') and args.operations:
        config['operations'] = args.operations
    
    if hasattr(args, 'temperature') and args.temperature:
        config['temperature'] = args.temperature
    
    if hasattr(args, 'steps') and args.steps:
        config['steps'] = args.steps
    
    if hasattr(args, 'max_rejected') and args.max_rejected:
        config['max_rejected'] = args.max_rejected
    
    if hasattr(args, 'model_state') and args.model_state:
        config['model']['state_dict'] = args.model_state
    
    if hasattr(args, 'prop_stats') and args.prop_stats:
        config['model']['prop_stats'] = args.prop_stats
    
    if hasattr(args, 'device') and args.device:
        config['model']['device'] = args.device
    
    if hasattr(args, 'save_trajectories') and args.save_trajectories:
        config['save_trajectories'] = args.save_trajectories
    
    if hasattr(args, 'log_level') and args.log_level:
        config['logging']['level'] = args.log_level

    # Resolve base_structures
    if args.base_structures:
        config["base_structures"] = resolve_structures(args.base_structures)
    
    return config

def validate_config(config):
    """
    Validate the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Boolean indicating if configuration is valid
    """
    errors = []

    # Check for required base structures
    if not config["base_structures"]:
        errors.append("No base structures specified")
    
    # Check for seed structure if needed
    if "attach_rotate" in config['operations']:
        if not config["seed_structures"]:
            errors.append("Seed structure required for attach_rotate operation")

    # Check model
    model_state = config["model"]["state_dict"]
    if model_state and not os.path.exists(model_state):
        errors.append(f"Model state dictionary not found: {model_state}")

    prop_stats = config["model"]["prop_stats"]
    if prop_stats and model_state.endswith(".pth.tar") and not os.path.exists(prop_stats):
        errors.append(f"Property stats file not found: {prop_stats}")

    try:
        parse_operation_sequence(config["operations"])
    except ValueError as e:
        errors.append(str(e))

    # Ensure output directory exists or can be created
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create output directory {output_dir}: {e}")

    #Report errors if any
    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True

def parse_operation_sequence(sequence_names):
    """
    Convert operation names to OperationType enum values.
    
    Args:
        sequence_names: List of operation names
        
    Returns:
        List of OperationType enum values
        
    Raises:
        ValueError: If an invalid operation name is provided
    """
    operation_map = {
        "flip": OperationType.FLIP,
        "attach_rotate": OperationType.ATTACH_ROTATE,
        "add_proton": OperationType.ADD_PROTON
    }

    operations = []
    for name in sequence_names:
        if name.lower() in operation_map:
            operations.append(operation_map[name.lower()])
        else:
            valid_ops = ', '.join(operation_map.keys())
            raise ValueError(f"Unknown operation: {name}. Valid operations are: {valid_ops}")

    return operations

def export_default_config(output_path):
    """
    Export default configuration to a YAML file.
    
    Args:
        output_path: Path to save the YAML configuration file
        include_header: Whether to include header text explaining the file
    
    Returns:
        Boolean indicating success
    """
    try:
        default_config = load_config(None)

        with open(output_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=None, 
                      sort_keys=False, width=88, indent=2)
        logger.info(f"Default configuration exported to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting default configuration: {e}")
        return False
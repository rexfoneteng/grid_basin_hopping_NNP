#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-17 16:34:44
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-21 16:18:07
# @Description: Configuration utilities for basin hopping simulations.

import os
import yaml
import logging

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
        "output_dir": "basin_hopping_output",
        "base_structures": "./",
        "seed_structures": None,
        "temperature": 300.0,
        "max_rejected": 50,
        "save_trajectories": False,
        "steps": 100,
        "operation_sequence": ["flip", "attach_rotate", "add_proton"],
        "operation_params": [{}, {}, {}],
        "optimizer": {"type": "nnp_gauss"},  # Options: "nnp", "gamess"
        "optimization": {
            "fmax": 5e-3,
            "steps": 1200
        },
        "physical_check": {
            "enabled": True,
            "params": PHYSICAL_DICT
        },
        "flip_grid": [0, 120, 240],
        "attach_rotate_grid": [0, 60, 120, 180, 240, 300],
        "proton_grid": [
            [16, -130, "OCC"],
            [16, 130, "OCC"],
            [0, -130, "OCC"],
            [0, 130, "OCC"],
            [21, -120, "OCH"],
            [21, 120, "OCH"],
            [19, -120, "OCH"],
            [19, 120, "OCH"],
            [17, -120, "OCH"],
            [17, 120, "OCH"],
            [14, -120, "OCH"],
            [14, 120, "OCH"],
            [39, -130, "OCC"],
            [39, 130, "OCC"],
            [24, -120, "OCH"],
            [24, 120, "OCH"],
            [43, -67.5, "N"],
            [43, 67.5, "N"],
            [50, 0, "OC"],
            [50, 60, "OC"],
            [50, 120, "OC"],
            [50, 180, "OC"],
            [50, 240, "OC"],
            [50, 300, "OC"],
            [40, -120, "OCH"],
            [40, 120, "OCH"],
            [37, -120, "OCH"],
            [37, 120, "OCH"]
        ],
        "skip_local_optimization": False,
        "logging": {
            "level": "INFO",
            "file": "basin_hopping.log"
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
    if hasattr(args, "output_dir") and args.output_dir:
        config["output_dir"] = args.output_dir
    
    if hasattr(args, "base_structures") and args.base_structures:
        config["base_structures"] = args.base_structures
    
    if hasattr(args, "seed_structures") and args.seed_structures:
        config["seed_structures"] = args.seed_structures
    
    if hasattr(args, "operation_sequence") and args.operation_sequence:
        config["operation_sequence"] = args.operation_sequence

    if hasattr(args, "operation_params") and args.operation_params:
        config["operation_params"] = args.operation_params
    
    if hasattr(args, "temperature") and args.temperature:
        config["temperature"] = args.temperature
    
    if hasattr(args, "steps") and args.steps:
        config["steps"] = args.steps
    
    if hasattr(args, "max_rejected") and args.max_rejected:
        config["max_rejected"] = args.max_rejected
    
    if hasattr(args, "model_state") and args.model_state:
        config["model"]["state_dict"] = args.model_state
    
    if hasattr(args, "prop_stats") and args.prop_stats:
        config["model"]["prop_stats"] = args.prop_stats
    
    if hasattr(args, "device") and args.device:
        config["model"]["device"] = args.device
    
    if hasattr(args, "save_trajectories") and args.save_trajectories:
        config["save_trajectories"] = args.save_trajectories

    if hasattr(args, "skip_local_optimization") and args.skip_local_optimization:
        config["skip_local_optimization"] = args.skip_local_optimization
    
    if hasattr(args, "log_level") and args.log_level:
        config["logging"]["level"] = args.log_level

    # Resolve base_structures
    if args.base_structures:
        config["base_structures"] = resolve_structures(args.base_structures)

    if args.seed_structures:
        config["seed_structures"] = resolve_structures(args.seed_structures)
    
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
    if "attach_rotate" in config["operation_sequence"]:
        if not config["seed_structures"]:
            errors.append("Seed structure required for attach_rotate operation")
        #elif not os.path.exists(config["seed_structures"]):
        #    errors.append(f"Seed structure file not found: {config["seed_structures"]}")

    # Check optimizer configuration
    optimizer_type = config["optimizer"]["type"]
    if optimizer_type not in ["nnp", "dftb3", "nnp_gauss"]:
        errors.append(f"Invalid optimizer type: {optimizer_type}. Supported types are 'nnp' and 'gamess'")
    
    # For NNP optimizer, check model state
    if optimizer_type == "nnp":
        model_state = config["optimizer"]["params"].get("state_dict")
        if not model_state:
            errors.append("NNP optimizer requires state_dict parameter")
        elif not os.path.exists(model_state):
            errors.append(f"Model state dictionary not found: {model_state}")
        
        # Check property stats for older model format
        if model_state and model_state.endswith(".pth.tar"):
            prop_stats = config["optimizer"]["params"].get("prop_stats")
            if not prop_stats:
                errors.append("Property stats required for .pth.tar model format")
            elif not os.path.exists(prop_stats):
                errors.append(f"Property stats file not found: {prop_stats}")
    
    # For GAMESS optimizer, check required parameters
    if optimizer_type == "gamess":
        gamess_params = config["optimizer"]["params"]
        if "method" not in gamess_params:
            errors.append("GAMESS optimizer requires 'method' parameter")
        if "basis_set" not in gamess_params:
            errors.append("GAMESS optimizer requires 'basis_set' parameter")

    try:
        parse_operation_sequence(config["operation_sequence"])
    except ValueError as e:
        errors.append(str(e))

    # Ensure output directory exists or can be created
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create output directory {output_dir}: {e}")

    # Report errors if any
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
        "add_proton": OperationType.ADD_PROTON,
        "functional_to_ch": OperationType.FUNCTIONAL_TO_CH,
        "ch_to_methyl": OperationType.CH_TO_METHYL
    }

    operations = []
    for name in sequence_names:
        if name.lower() in operation_map:
            operations.append(operation_map[name.lower()])
        else:
            valid_ops = ", ".join(operation_map.keys())
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
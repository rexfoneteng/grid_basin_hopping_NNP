#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple, Optional
import logging

# Import your BasinHoppingGenerator class
from basin_hopping.basin_hopping_generator import BasinHoppingGenerator
from basin_hopping.operation_type import OperationType
from core.molecular_structure import MolecularStructure
from xyz_tools import Xyz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_xyz(structure, filename, comment=""):
    """Save MolecularStructure or xyz_list to file."""
    if isinstance(structure, MolecularStructure):
        xyz_str = structure.to_xyz_str()
        lines = xyz_str.strip().split("\n")
        lines[1] = comment  # Replace the comment line
        xyz_str = "\n".join(lines)
        
        with open(filename, 'w') as f:
            f.write(xyz_str)
    else:
        # Assume it's an xyz_list
        with open(filename, 'w') as f:
            f.write(f"{len(structure)}\n")
            f.write(f"{comment}\n")
            for atom in structure:
                symbol = atom[0]
                x, y, z = atom[1:]
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

def debug_proton_placement(base_structure_path, output_dir="debug_output", proton_grid=None):
    """Debug the proton placement in BasinHoppingGenerator"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    custom_proton_grid = (40, 120, "OCH")

    optimize_params = {
    "fmax": 4e-3,
    "steps": 1000}



    # Physical geometry check parameters
    PHYSICAL_DICT = {
        "CO_min_threshold": 1.2,
        "CH_min_threshold": 0.88,
        "OO_min_threshold": 1.62,
        "OH_min_threshold": 0.85,
        "NC_min_threshold": 1.20,
        "NO_min_threshold": 1.62,
        "NH_min_threshold": 0.73,
        "HH_min_threshold": 0.66
    }

    model_params = {
    "state_dict": "/home/htphan/work/sugar_4/schnetpack_proj/force_pred/m062x_6-311_gpd/find_structure_di/08_add_bMan_14_GlcNAc/cl_learning_lr_8e-4/best_model",
    "prop_stats": "/beegfs/coldpool/htphan/sugar_4/schnetpack_proj/force_pred/m062x_6-311_gpd/property_stats/property_stats_atomrefs_all.pt",
    "device": "cpu",
    "in_module": {
        "n_atom_basis": 128,
        "n_filters": 128,
        "n_gaussians": 75,
        "charged_systems": True,
        "n_interactions": 4,
        "cutoff": 15.0
    },
    "interface_params": {
        "energy": "energy",
        "forces": "force",
        "energy_units": "Hartree",
        "forces_units": "Hartree/Angstrom"
    }
    }
    
    # Initialize BasinHoppingGenerator with minimal settings
    generator = BasinHoppingGenerator(
        base_structures=[base_structure_path],
        seed_structure=base_structure_path,  # Using same file as both base and seed
        operation_sequence=[OperationType.ADD_PROTON],  # Only focus on proton addition
        temperature=99999999.9,
        check_physical=True,
        check_physical_kwargs=PHYSICAL_DICT,
        optimize_params=optimize_params,
        output_xyz=os.path.join(output_dir, "basin_hopping_output.xyz")
    )
    
    # Use custom proton grid if provided
    if proton_grid:
        generator.set_proton_grid(proton_grid)

    print(f"PHYSIC DICT {PHYSICAL_DICT}")
    
    # Load base structure
    base_structure = generator._load_structure(base_structure_path)
    if not base_structure:
        logger.error(f"Failed to load base structure: {base_structure_path}")
        return
    
    # Save the base structure
    save_xyz(base_structure, os.path.join(output_dir, "base_structure.xyz"), 
             "Base structure for proton placement")
    
    # Get proton grid definitions from the generator
    all_proton_grids = generator.proton_grid
    
    # Create a file to record all results
    results_file = os.path.join(output_dir, "proton_placement_results.txt")
    with open(results_file, 'w') as f:
        f.write("Proton Placement Debug Results\n")
        f.write("==============================\n\n")
        f.write(f"Base structure: {base_structure_path}\n")
        f.write(f"Total proton grid points: {len(all_proton_grids)}\n\n")
        
        # Print the proton grid for reference
        f.write("Proton Grid Definitions:\n")
        f.write("=======================\n")
        for i, (at_idx, angle_val, atom_type) in enumerate(all_proton_grids):
            f.write(f"Grid {i}: atom_idx={at_idx}, angle={angle_val}, type={atom_type}\n")
        f.write("\n")
    
    # Test each proton grid point
    for i, (at_idx, angle_val, atom_type) in enumerate(all_proton_grids):
        logger.info(f"Testing proton grid {i}: atom_idx={at_idx}, angle={angle_val}, type={atom_type}")
        
        # Call _get_specific_protonated_structure directly
        try:
            protonated_structure = generator._get_specific_protonated_structure(base_structure, i)
            
            if protonated_structure:
                # Save the protonated structure
                output_file = os.path.join(output_dir, f"protonated_grid_{i}.xyz")
                save_xyz(protonated_structure, output_file, 
                         f"Protonated at atom {at_idx}, angle {angle_val}, type {atom_type}")
                
                # Append results to the results file
                with open(results_file, 'a') as f:
                    f.write(f"Grid {i} (atom={at_idx}, angle={angle_val}, type={atom_type}):\n")
                    f.write(f"  - Success: Yes\n")
                    f.write(f"  - Output file: {os.path.basename(output_file)}\n")
                    f.write(f"  - Metadata: {protonated_structure.metadata}\n\n")
                
                logger.info(f"  - Success: Saved to {output_file}")
            else:
                # Record failure
                with open(results_file, 'a') as f:
                    f.write(f"Grid {i} (atom={at_idx}, angle={angle_val}, type={atom_type}):\n")
                    f.write(f"  - Success: No\n")
                    f.write(f"  - Reason: Function returned None\n\n")
                
                logger.warning(f"  - Failed: Function returned None")
        except Exception as e:
            # Record error
            with open(results_file, 'a') as f:
                f.write(f"Grid {i} (atom={at_idx}, angle={angle_val}, type={atom_type}):\n")
                f.write(f"  - Success: No\n")
                f.write(f"  - Error: {str(e)}\n\n")
            
            logger.error(f"  - Error: {str(e)}")
    
    # Now test with specific angle variations for a selected atom
    # Find all OCH-type grid points
    och_grid_points = [(i, at_idx, angle_val) for i, (at_idx, angle_val, atom_type) in enumerate(all_proton_grids) 
                      if atom_type == "OCH"]
    
    if och_grid_points:
        # Take the first OCH point
        test_grid_idx, test_at_idx, _ = och_grid_points[0]
        
        # Test with various angles
        test_angles = [-180, -120, -60, 0, 60, 120, 180]
        
        with open(results_file, 'a') as f:
            f.write("\nAngle Variation Tests\n")
            f.write("====================\n")
            f.write(f"Testing atom_idx={test_at_idx} (OCH type) with various angles\n\n")
        
        # Create temporary modified proton grid
        for angle in test_angles:
            # Modify the grid for this test
            temp_grid = all_proton_grids.copy()
            temp_grid[test_grid_idx] = (test_at_idx, angle, "OCH")
            
            # Set the modified grid
            generator.set_proton_grid(temp_grid)
            
            logger.info(f"Testing atom {test_at_idx} with angle {angle}")
            
            try:
                protonated_structure = generator._get_specific_protonated_structure(base_structure, test_grid_idx)
                
                if protonated_structure:
                    # Save the protonated structure
                    output_file = os.path.join(output_dir, f"protonated_atom_{test_at_idx}_angle_{angle}.xyz")
                    save_xyz(protonated_structure, output_file, 
                             f"Protonated at atom {test_at_idx}, angle {angle}")
                    
                    # Append results
                    with open(results_file, 'a') as f:
                        f.write(f"Angle {angle}:\n")
                        f.write(f"  - Success: Yes\n")
                        f.write(f"  - Output file: {os.path.basename(output_file)}\n\n")
                    
                    logger.info(f"  - Success: Saved to {output_file}")
                else:
                    with open(results_file, 'a') as f:
                        f.write(f"Angle {angle}:\n")
                        f.write(f"  - Success: No\n")
                        f.write(f"  - Reason: Function returned None\n\n")
                    
                    logger.warning(f"  - Failed: Function returned None")
            except Exception as e:
                with open(results_file, 'a') as f:
                    f.write(f"Angle {angle}:\n")
                    f.write(f"  - Success: No\n")
                    f.write(f"  - Error: {str(e)}\n\n")
                
                logger.error(f"  - Error: {str(e)}")
    
    # Restore original grid
    generator.set_proton_grid(all_proton_grids)
    
    logger.info(f"Debug complete. Results saved to {output_dir}")
    logger.info(f"Summary of results saved to {results_file}")
    
    return generator

def main():
    parser = argparse.ArgumentParser(description='Debug proton placement in BasinHoppingGenerator')
    parser.add_argument('input_xyz', help='Input XYZ file')
    parser.add_argument('--output_dir', default="debug_proton_output", help='Output directory')
    
    args = parser.parse_args()
    
    # Run the debug function
    generator = debug_proton_placement(args.input_xyz, args.output_dir)
    
    if generator:
        print(f"Debug files saved to {args.output_dir}/")
    else:
        print("Error: Failed to initialize generator")

if __name__ == "__main__":
    main()
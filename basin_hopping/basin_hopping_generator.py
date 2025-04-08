#!/usr/bin/env python3

import numpy as np
import logging
import random
import os
import torch
import schnetpack as spk
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import tempfile
import time
from custom_interface import CustomInterface
from rotational_matrix import rotation_matrix_numpy
from ase.io import read
import math
import shutil

from generators.base_generator import BaseGenerator
from generators.flip_generator import FlipGenerator
from generators.attach_rotate_generator import AttachRotateGenerator
from generators.proton_generator import ProtonGenerator
from core.molecular_structure import MolecularStructure
from core.constants import DEFAULT_FLIP_PARAMS, DEFAULT_ATTACH_ROTATE_PARAMS, DEFAULT_ADD_PROTON_PARAMS, eV_TO_Ha_conversion
from basin_hopping.operation_type import OperationType
from utils.flatten import flatten_concatenation

from rotational_matrix import rotation_matrix_numpy
from geometry_1 import unit_vector

from os_tools import single_cmd
from xyz_tools import Xyz, attach, turn
from sugar_tools import sugar_stat
from rotational_matrix import rotation_matrix_numpy
from xyz_physical_geometry_tool_mod import is_physical_geometry


logger = logging.getLogger(__name__)

class BasinHoppingGenerator(BaseGenerator):
    """
    Generator that performs grid-based basin hopping by applying a sequence of operations
    in sequence to systematically explore the energy landscape, with local optimization using NNP model.
    """

    def __init__(self, 
                 base_structures: List[str],
                 seed_structure: str,
                 temperature: float = 300.0,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None,
                 model_params: Optional[Dict[str, Any]] = None,
                 optimize_params: Optional[Dict[str, Any]] = None,
                 output_xyz: str = "best_structure.xyz",
                 max_rejected: int = 50,
                 operation_sequence: List[OperationType] = None,
                 save_trj: bool = False,
                 trajectory_dir: str = "opt_trj"):
        """Initialize the basin hopping generator.
        
        Args:
            base_structures: List of paths to base structure files
            seed_structure: Path to the seed structure file for attachment
            temperature: Temperature parameter for Metropolis criterion (K)
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
            model_params: Parameters for the NNP model
            optimize_params: Parameters for the optimization
            output_xyz: Path to the output XYZ file where all local minima will be appended
            max_rejected: Maximum number of consecutive rejected moves before stopping
            operation_sequence: Sequence of operations to apply (default: FLIP, ATTACH_ROTATE)
        """
        super().__init__(check_physical, check_physical_kwargs)
        
        # Set operation sequence
        self.operation_sequence = operation_sequence
        if not self.operation_sequence:
            # Default operation sequence if none provided
            self.operation_sequence = [OperationType.FLIP, OperationType.ATTACH_ROTATE]
        
        self.base_structures = base_structures
        self.seed_structure = seed_structure
        self.temperature = temperature
        self.max_rejected = max_rejected
        self.output_xyz = output_xyz

        # Optimization trajectory saving
        self.save_trj = save_trj
        self.trajectory_dir = trajectory_dir
        if self.save_trj and not os.path.exists(trajectory_dir):
            os.makedirs(self.trajectory_dir, exist_ok=True)
        
        # Grid definitions for each operation
        self.flip_angles = [0, 120, 240]  # 3 grid points for rotation after flip (in degrees)
        self.attach_angles = [0, 60, 120, 180, 240, 300]  # 6 grid points for rotation after attach (in degrees)
        
        # Define proton addition grid points
        # Each grid point is a tuple of (atom_type, angle, description)
        # where atom_type is: 0=hydroxyl O, 1=ring O, 2=NAc O, 3=NAc N
        self.proton_grid = [
            (16, -130, "OCC"), # atom_index, angle in degree, atom type
            (16, 130, "OCC"),

            (0, -130, "OCC"),
            (0, 130, "OCC"),

            (21, -120, "OCH"),
            (21, 120, "OCH"),

            (19, -120, "OCH"),
            (19, 120, "OCH"),

            (17, -120, "OCH"),
            (17, 120, "OCH"),
            
            (14, -120, "OCH"),
            (14, 120, "OCH"),

            (39, -130, "OCC"),
            (39, 130, "OCC"),

            (24, -120, "OCH"),
            (24, 120, "OCH"),
            
            (43, -67.5, "N"),
            (43, 67.5, "N"),

            (50, 0, "OC"),
            (50, 60, "OC"),
            (50, 120, "OC"),
            (50, 180, "OC"),
            (50, 240, "OC"),
            (50, 300, "OC"),

            (40, -120, "OCH"),
            (40, 120, "OCH"),

            (37, -120, "OCH"),
            (37, 120, "OCH")
        ]
        
        # Setup NNP model parameters
        self.model_params = model_params or {}
        self.optimize_params = optimize_params or {}
        
        # Default optimization parameters if not provided
        if "fmax" not in self.optimize_params:
            self.optimize_params["fmax"] = 5e-3
        if "steps" not in self.optimize_params:
            self.optimize_params["steps"] = 1200
            
        # Load NNP model if parameters are provided
        self.nnp_model = None
        if "state_dict" in self.model_params:
            self._load_nnp_model()
            
        # Initialize operation generators
        self.flip_generator = FlipGenerator(
            base_structures=self.base_structures,
            params=DEFAULT_FLIP_PARAMS.copy(),
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)
            
        self.attach_rotate_generator = AttachRotateGenerator(
            base_structures=self.base_structures,
            seed_structures=[self.seed_structure],
            params=DEFAULT_ATTACH_ROTATE_PARAMS.copy(),
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)
        
        self.proton_generator = ProtonGenerator(
            base_structures=self.base_structures,
            params=DEFAULT_ADD_PROTON_PARAMS.copy(),
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)
        
        # Map operation types to generator instances
        self.generators = {
            OperationType.FLIP: self.flip_generator,
            OperationType.ATTACH_ROTATE: self.attach_rotate_generator,
            OperationType.ADD_PROTON: self.proton_generator
        }
        
        # Configure the specific parameters for each generator
        self.flip_generator.set_params(
            position="4NR",  # Position to flip (can be customized)
            rotate_after_flip=True,
            rotating_angle=self.flip_angles
        )
        
        self.attach_rotate_generator.set_params(
            angle_list=self.attach_angles
        )
        
        self.proton_generator.set_params(
            angle_list=[-120, 0, 120],  # Included for backward compatibility 
            angle_around_Oring=[-130, 0, 130],  # Included for backward compatibility
            proton_Oring_dist=0.982  # Default O-H bond length (Å)
        )
        
        # Current state of basin hopping
        self.current_structure = None
        self.current_optimized_structure = None
        self.current_energy = float('inf')
        self.current_grid_point = None
        
        # Best structure found
        self.best_structure = None
        self.best_energy = float('inf')
        
        # History and statistics
        self.history = []
        self.consecutive_rejections = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.stats = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "total_steps": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "optimization_failures": 0,
            "generation_failures": 0,
            "max_consecutive_rejections": 0,
            "final_energy": None,
            "best_energy": None,
            "stopping_reason": None
        }
        
        # Initialize the output XYZ file
        with open(self.output_xyz, 'w') as f:
            f.write("")  # Create an empty file
        
    def _load_nnp_model(self):
        """Load the Neural Network Potential model."""
        try:
            # Get model parameters
            state_dict_path = self.model_params.get("state_dict")
            prop_stats_path = self.model_params.get("prop_stats")
            device = self.model_params.get("device", "cpu")
            
            if not state_dict_path or not os.path.exists(state_dict_path):
                logger.error(f"Model state_dict file not found: {state_dict_path}")
                return
            
            # Load model
            state_dict_basename = os.path.basename(state_dict_path)
            
            if state_dict_basename.endswith(".pth.tar"):
                # Load property statistics
                if not prop_stats_path or not os.path.exists(prop_stats_path):
                    logger.error(f"Property stats file not found: {prop_stats_path}")
                    return
                
                prop_stats = torch.load(prop_stats_path)
                means, stddevs = prop_stats["means"], prop_stats["stddevs"]
                
                # Define SchNet model
                in_module_params = self.model_params.get("in_module", {
                    "n_atom_basis": 128,
                    "n_filters": 128,
                    "n_gaussians": 15,
                    "charged_systems": True,
                    "n_interactions": 4,
                    "cutoff": 15.0
                })
                
                in_module = spk.representation.SchNet(**in_module_params)
                
                out_module = spk.atomistic.Atomwise(
                    n_in=in_module.n_atom_basis,
                    property="energy",
                    mean=means["energy"],
                    stddev=stddevs["energy"],
                    derivative="force",
                    negative_dr=True
                )
                
                self.nnp_model = spk.AtomisticModel(
                    representation=in_module,
                    output_modules=out_module
                )
                
                # Load state dictionary
                state_dict = torch.load(state_dict_path, map_location=device)
                self.nnp_model.load_state_dict(state_dict["model"])
                
            elif state_dict_basename == "best_model":
                self.nnp_model = torch.load(state_dict_path, map_location=device)
                
            else:
                raise ValueError(f"Unknown model format: {state_dict_basename}")
                
            # Set model to evaluation mode
            self.nnp_model.eval()
            logger.info("NNP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NNP model: {str(e)}", exc_info=True)
            self.nnp_model = None
        
    def __call__(self, n_steps: int = 100) -> MolecularStructure:
        """Run basin hopping for n_steps and return the best structure found.
        
        Args:
            n_steps: Maximum number of basin hopping steps to perform
            
        Returns:
            The best structure found during basin hopping
        """
        # Reset statistics
        self._reset_stats()
        self.stats["start_time"] = time.time()
        
        # Initialize if not already
        if self.current_structure is None:
            self._initialize()
            
        # Perform basin hopping
        step = 0
        while step < n_steps:
            logger.info(f"Basin hopping step {step+1}/{n_steps} (consecutive rejections: {self.consecutive_rejections}/{self.max_rejected})")
            self.stats["total_steps"] += 1
            
            # Check if maximum consecutive rejections reached
            if self.consecutive_rejections >= self.max_rejected:
                logger.info(f"Stopping: Maximum consecutive rejections ({self.max_rejected}) reached")
                self.stats["stopping_reason"] = "max_consecutive_rejections"
                break
                
            # Propose a move
            proposed_structure, proposed_grid_point = self._propose_move()
            
            # Skip if no valid structure was proposed
            if proposed_structure is None:
                logger.warning(f"Step {step}: Failed to propose a valid structure")
                self.stats["generation_failures"] += 1
                continue
            
            #with open("./input.xyz", "a") as f:
            #    f.write(proposed_structure.to_xyz_str())  
            # Optimize the proposed structure
            optimized_structure, optimized_energy, trajectory_path = self._optimize_structure(proposed_structure)
            
            # Skip if optimization failed
            if optimized_structure is None:
                logger.warning(f"Step {step}: Optimization failed")
                self.stats["optimization_failures"] += 1
                continue
                
            # Accept or reject the move
            accepted = self._accept_move(optimized_energy)
            
            # Update state if accepted
            if accepted:
                self._append_to_output_file(optimized_structure, optimized_energy, step, accepted)
                self.current_structure = proposed_structure
                self.current_optimized_structure = optimized_structure
                self.current_energy = optimized_energy
                self.current_grid_point = proposed_grid_point
                self.consecutive_rejections = 0
                self.total_accepted += 1
                self.stats["accepted_steps"] += 1
                
                # Record in history
                self.history.append({
                    'step': step,
                    'grid_point': proposed_grid_point,
                    'energy': optimized_energy,
                    'structure': optimized_structure,
                    'unoptimized_structure': proposed_structure,
                    'operations': self._get_operations_description(proposed_grid_point)
                })
                
                logger.info(f"Step {step}: Accepted move to grid point {proposed_grid_point} with energy {optimized_energy:.6f}")
                
                # Update best structure if applicable
                if optimized_energy < self.best_energy:
                    self.best_structure = optimized_structure
                    self.best_energy = optimized_energy
                    logger.info(f"Step {step}: New best structure with energy {self.best_energy:.6f}")
            else:
                self.consecutive_rejections += 1
                self.total_rejected += 1
                self.stats["rejected_steps"] += 1
                self.stats["max_consecutive_rejections"] = max(
                    self.stats["max_consecutive_rejections"], 
                    self.consecutive_rejections
                )
                
                logger.info(f"Step {step}: Rejected move to grid point {proposed_grid_point} with energy {optimized_energy:.6f}")
                
            step += 1
            
        # If we completed all steps without hitting max rejections
        if step >= n_steps:
            self.stats["stopping_reason"] = "max_steps"
            
        # Update final statistics
        self.stats["end_time"] = time.time()
        self.stats["duration"] = self.stats["end_time"] - self.stats["start_time"]
        self.stats["final_energy"] = self.current_energy
        self.stats["best_energy"] = self.best_energy
        
        logger.info(f"Basin hopping completed: {self.stats['stopping_reason']}")
        logger.info(f"Best energy found: {self.best_energy:.6f}")
        logger.info(f"Accepted/rejected moves: {self.total_accepted}/{self.total_rejected}")
        
        return self.best_structure
    
    def _reset_stats(self):
        """Reset statistics for a new run."""
        self.consecutive_rejections = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.stats = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "total_steps": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "optimization_failures": 0,
            "generation_failures": 0,
            "max_consecutive_rejections": 0,
            "final_energy": None,
            "best_energy": None,
            "stopping_reason": None
        }
        
    def _initialize(self):
        """Initialize the basin hopping process with a random starting point."""
        # Generate a random grid point
        grid_point = self._get_random_grid_point()
        
        # Apply operations to generate a structure
        structure = self._apply_operations(grid_point)
        
        # Try again if we couldn't generate a valid structure
        attempts = 0
        while structure is None and attempts < 10:
            grid_point = self._get_random_grid_point()
            structure = self._apply_operations(grid_point)
            attempts += 1
            
        if structure is None:
            raise RuntimeError("Failed to initialize basin hopping with a valid structure")
            
        # Optimize the structure
        #with open("./input.xyz", "a") as f:
        #    f.write(structure.to_xyz_str())
        optimized_structure, optimized_energy, trajectory_path = self._optimize_structure(structure)
        
        if optimized_structure is None:
            logger.error("Initial structure optimization failed")
            raise RuntimeError("Failed to optimize initial structure")
            
        # Set initial state
        self.current_structure = structure
        self.current_optimized_structure = optimized_structure
        self.current_energy = optimized_energy
        self.current_grid_point = grid_point
        
        # Initialize best structure
        self.best_structure = optimized_structure
        self.best_energy = optimized_energy
        
        # Append initial structure to output file
        self._append_to_output_file(optimized_structure, optimized_energy, -1, True)
        
        logger.info(f"Initialized basin hopping at grid point {grid_point} with energy {self.current_energy:.6f}")
    
    def _optimize_structure(self, structure: MolecularStructure) -> Tuple[Optional[MolecularStructure], float]:
        """Optimize a structure using the NNP model.
        
        Args:
            structure: Structure to optimize
            
        Returns:
            Tuple of (optimized structure, energy)
        """
        if self.nnp_model is None:
            # If no NNP model, just return the structure with a random energy
            if hasattr(self, 'energy_function') and self.energy_function:
                energy = self.energy_function(structure)
            else:
                energy = random.random() * 100
            return structure, energy, None
            
        try:
            # Create a temporary directory for the optimization
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdirname:
                # Create a temporary file within the temporary directory
                # Save structure to XYZ file
                temp_xyz = f"{tmpdirname}/input.xyz"
                with open(temp_xyz, 'w') as f:
                    f.write(structure.to_xyz_str())
            
                # Set up interface with SchNetPack
                interface_params = self.model_params.get("interface_params", {
                    "energy": "energy",
                    "forces": "force",
                    "energy_units": "Hartree",
                    "forces_units": "Hartree/A"
                })
            
                device = self.model_params.get("device", "cpu")
            
                #ase_interface = AseInterface(
                ase_interface = CustomInterface(
                    temp_xyz,
                    self.nnp_model,
                    tmpdirname,
                    device,
                    **interface_params
                )
            
                # Run optimization
                ase_interface.optimize(**self.optimize_params)
                
                # Read optimized structure
                opt_traj = f"{tmpdirname}/optimization.traj"
                if not os.path.exists(opt_traj):
                    logger.error(f"Optimization failed: output file not found at {opt_traj}")
                    return None, float('inf'), None

                opt_frame = read(opt_traj)
                # The last frame is the optimized structure
                optimized_structure = MolecularStructure.from_ase_atoms(opt_frame)
            
                # Copy metadata from original structure
                optimized_structure.metadata = structure.metadata.copy()
                optimized_structure.metadata.update({
                    "optimized": True,
                    "optimize_params": self.optimize_params.copy()
                })
                
                # Get final energy
                local_min_at = read(opt_traj)
                energy = local_min_at.get_potential_energy() * eV_TO_Ha_conversion

                saved_trajectory_path = None
                if self.save_trj:
                    saved_trajectory_path = self._traj2xyz(structure, opt_traj)

            return optimized_structure, energy, saved_trajectory_path
            
        except Exception as e:
            logger.error(f"Error in structure optimization: {str(e)}", exc_info=True)
            return None, float('inf')
    
    def _propose_move(self) -> Tuple[Optional[MolecularStructure], Tuple]:
        """Propose a new move by selecting a random neighbor on the grid.
        
        Returns:
            Tuple of (proposed structure, grid point)
        """
        # Get a random grid point
        grid_point = self._get_random_grid_point()
        
        # Apply operations to generate a structure
        structure = self._apply_operations(grid_point)
        
        # If failed to generate a valid structure, try another random grid point
        attempts = 0
        while structure is None and attempts < 5:
            grid_point = self._get_random_grid_point()
            structure = self._apply_operations(grid_point)
            attempts += 1
            
        return structure, grid_point
    
    def _get_random_grid_point(self) -> Tuple:
        """Get a completely random grid point.
        
        Returns:
            Tuple of grid point indices for each operation in the sequence
        """
        grid_point = []
        
        # First component: random base structure index
        base_idx = random.randint(0, len(self.base_structures) - 1)
        grid_point.append(base_idx)
        
        # Add grid point components for each operation in the sequence
        for op in self.operation_sequence:
            if op == OperationType.FLIP:
                # Random flip angle index
                flip_angle_idx = random.randint(0, len(self.flip_angles) - 1)
                grid_point.append(flip_angle_idx)
            elif op == OperationType.ATTACH_ROTATE:
                # Random attach angle index
                attach_angle_idx = random.randint(0, len(self.attach_angles) - 1)
                grid_point.append(attach_angle_idx)
            elif op == OperationType.ADD_PROTON:
                # Random proton grid index (combines atom type and angle)
                proton_grid_idx = random.randint(0, len(self.proton_grid) - 1)
                grid_point.append(proton_grid_idx)
                
        return tuple(grid_point)
    
    def _apply_operations(self, grid_point: Tuple) -> Optional[MolecularStructure]:
        """Apply the operations in sequence to generate a structure.
        
        Args:
            grid_point: Tuple of grid point indices
            
        Returns:
            Generated structure or None if generation failed
        """
        try:
            # Extract base structure index
            base_idx = grid_point[0]
            
            # Load the base structure
            if base_idx >= len(self.base_structures):
                logger.error(f"Invalid base_idx {base_idx} (max: {len(self.base_structures)-1})")
                return None
                
            base_structure_path = self.base_structures[base_idx]
            structure = self._load_structure(base_structure_path)
            
            if not structure:
                logger.error(f"Failed to load base structure: {base_structure_path}")
                return None
                
            # Apply each operation in sequence
            grid_idx = 1  # Start after base_idx
            
            for op_idx, op_type in enumerate(self.operation_sequence):
                if op_type == OperationType.FLIP:
                    # Apply flip operation
                    flip_angle_idx = grid_point[grid_idx]
                    grid_idx += 1
                    
                    flip_angle = self.flip_angles[flip_angle_idx]
                    structure = self._get_specific_flipped_structure(structure, flip_angle)
                    
                    if not structure:
                        logger.warning(f"Failed to generate flipped structure with angle {flip_angle}")
                        return None
                        
                elif op_type == OperationType.ATTACH_ROTATE:
                    # Apply attach-rotate operation
                    attach_angle_idx = grid_point[grid_idx]
                    grid_idx += 1
                    
                    attach_angle = self.attach_angles[attach_angle_idx]
                    structure = self._get_specific_attached_structure(structure, attach_angle)
                    
                    if not structure:
                        logger.warning(f"Failed to generate attached structure with angle {attach_angle}")
                        return None
                        
                elif op_type == OperationType.ADD_PROTON:
                    # Apply add-proton operation with the combined grid
                    proton_grid_idx = grid_point[grid_idx]
                    grid_idx += 1
                    
                    structure = self._get_specific_protonated_structure(structure, proton_grid_idx)
                    
                    if not structure:
                        logger.warning(f"Failed to add proton to structure using grid point {proton_grid_idx}")
                        return None
            
            # Add grid point to metadata
            structure.metadata['grid_point'] = grid_point
            structure.metadata['base_structure'] = base_structure_path
            
            return structure
            
        except Exception as e:
            logger.error(f"Error applying operations: {str(e)}", exc_info=True)
            return None

    def _get_specific_flipped_structure(self, structure: MolecularStructure, flip_angle: float) -> Optional[MolecularStructure]:
        """Generate a structure with a specific flip angle."""
        try:
            # Convert MolecularStructure to xyz list format
            current_frame = structure.to_xyz_list()
            
            # Get sugar statistics

            sug_stat = sugar_stat(current_frame,
                                find_O_on_C_ring=True,
                                find_H_on_C_ring=True,
                                find_O_on_C_chain=True,
                                find_H_on_C_chain=True,
                                find_H_on_O=True,
                                find_H_on_C_orientation=True)

            # Determine ring ID and position
            ring_id = 0
            position_str = self.flip_generator.params["position"]
            position = int(position_str[0])

            if not position_str.endswith("NR"):
                ring_id = 1

            # Get atom IDs for flipping
            C_id = sug_stat["C_ring"][ring_id][position-1]
            HC_id = sug_stat["H_on_C_ring"][ring_id][position-1]
            O_id = sug_stat["O_on_C_ring"][ring_id][position-1]
            HO_id = sug_stat["H_on_O"][ring_id][position-1]

            C_flip_atom = (C_id, O_id)
            CH_bond = (C_id, HC_id)

            # Handle special case for N-Acetyl group
            if O_id == -1 and HO_id == -1:
                # Assuming C_NH_CO_CHHH contains N-Acetyl group information
                flip_group = sug_stat["C_NH_CO_CHHH"][0]
                # Find proton ID if needed
                from molecule_stat import molecule_stat
                from flatten_list import flatten_concatenation
                
                sugar_list = [s for s in current_frame]
                mol_stat = molecule_stat(sugar_list)
                ONAc_bond = [ele for ele in mol_stat["mol_bond_pattern"][0] if ele[1] == flip_group[4]]
                H_id = [idx for idx, ele in enumerate(sugar_list) if ele[0] == "H"]
                proton_id = set(flatten_concatenation(ONAc_bond)).intersection(H_id)
                
                flip_group.extend(proton_id)
                C_flip_atom = (C_id, flip_group[1])
            else:
                flip_group = (C_id, O_id, HO_id)

            # Perform the flip
            from xyz_toolsx import flip
            flipped_xyz = flip(current_frame, CH_bond, C_flip_atom, CH_bond, flip_group)
            
            # Rotate with the specified angle
            from xyz_tools import turn
            rotated_xyz = turn(flipped_xyz,
                             rotate_atom_list=flip_group,
                             rotate_bond=C_flip_atom,
                             angle=flip_angle)
            
            # Convert xyz_list to MolecularStructure obj
            flipped_structure = MolecularStructure.from_xyz_list(rotated_xyz)
            flipped_structure.metadata = structure.metadata.copy() if structure.metadata else {}
            flipped_structure.metadata.update({
                "operation": "flip",
                "base_structure": structure.metadata.get('source_file', ''),
                "ring_id": ring_id,
                "position": position,
                "angle_val": flip_angle
            })
            
            # Check if physically reasonable
            if self.check_physical:
                check_result = is_physical_geometry(rotated_xyz, **self.check_physical_kwargs)
                if check_result != "normal":
                    return None
            
            return flipped_structure
            
        except Exception as e:
            logger.error(f"Error generating flipped structure: {str(e)}", exc_info=True)
            return None

    def _get_specific_attached_structure(self, structure: MolecularStructure, attach_angle: float) -> Optional[MolecularStructure]:
        """Generate a structure with a specific attach rotation angle."""
        try:
            base_frame = structure.to_xyz_list()
            
            # Load seed structure

            seed_xyz_obj = Xyz(self.seed_structure)
            seed_frame = seed_xyz_obj.next()

            # Attach the structures
            attached = attach(base_frame, seed_xyz_list=seed_frame, **self.attach_rotate_generator.params["attach_kwargs"])

            # Get atom counts for rotation
            n_atom = len(base_frame)
            n_atom_1 = len(seed_frame)

            # Set up rotation parameters
            rot_bond = self.attach_rotate_generator.params["rotate_bond_list"][0]
            rot_atoms = set(list(rot_bond) + list(range(n_atom-1, n_atom+n_atom_1-3)))

            # Rotate with the specified angle
            rotated = turn(attached, rotate_bond=rot_bond, rotate_atom_list=rot_atoms, angle=attach_angle)
            
            # Convert to MolecularStructure
            attached_structure = MolecularStructure.from_xyz_list(rotated)
            attached_structure.metadata = structure.metadata.copy() if structure.metadata else {}
            attached_structure.metadata.update({
                "operation": "attach_rotate",
                "seed_structure": self.seed_structure,
                "rot_angle": attach_angle
            })
            
            # Check if physically reasonable
            if self.check_physical:
                check_result = is_physical_geometry(rotated, **self.check_physical_kwargs)
                if check_result != "normal":
                    return None
                    
            return attached_structure
            
        except Exception as e:
            logger.error(f"Error generating attached structure: {str(e)}", exc_info=True)
            return None
            
    def _get_specific_protonated_structure(self, structure: MolecularStructure, proton_grid_idx: int) -> Optional[MolecularStructure]:
        """Generate a structure with a specific proton added at a specific position.
        
        Args:
            structure: Input molecular structure
            proton_grid_idx: Index in self.proton_grid defining atom type and angle
            
        Returns:
            Protonated structure or None if generation failed
        """
        #with open("b4_add_H.xyz", 'a') as f:
        #    f.write(structure.to_xyz_str())
        try:
            # Get the atom type and angle from the proton grid
            if proton_grid_idx < 0 or proton_grid_idx >= len(self.proton_grid):
                logger.warning(f"Invalid proton_grid_idx: {proton_grid_idx}")
                return None
                
            at_idx, angle_val, atom_type = self.proton_grid[proton_grid_idx]
            
            # Get structure info
            info = self.proton_generator._get_structure_info(structure)
            current_frame = info['frame']
            coor = info['coor']


            O_chain = info.get("O_on_C_chain", [])
            for I0, O_ring_id in enumerate(info["O_ring"]):
                info["O_on_C_chain"][I0][4] = O_ring_id[0]

            flatten_O_ring = flatten_concatenation(info.get("O_ring", []))

            flatten_O_on_C_chain = flatten_concatenation(info.get("O_on_C_chain", []))

            flattened_C_chain = flatten_concatenation(info.get("C_chain", []))
            flattened_H_on_O = flatten_concatenation(info.get("H_on_O", []))

            
            
            # Handle different atom types
            if atom_type == "OCH":  # hydroxyl O
                try:
                    pos_idx = flatten_O_on_C_chain.index(at_idx)
                    c_idx = flattened_C_chain[pos_idx]
                    h_idx = flattened_H_on_O[pos_idx]
                except Exception as e:
                    raise
                
                
                # Calculate vectors for the C-O and O-H bonds
                CO_vec = coor[at_idx] - coor[c_idx]
                OH_vec = coor[h_idx] - coor[at_idx]
                
                # Convert angle from degrees to radians
                angle_rad = angle_val * (np.pi / 180)
                
                # Rotate the O-H vector around the C-O axis to get new proton position
                rotated_OH_vec = np.dot(rotation_matrix_numpy(CO_vec, angle_rad), OH_vec)
                proton_pos = coor[at_idx] + rotated_OH_vec
                
                atom_metadata = {
                    "site": "hydroxyl_O",
                    "pos_idx": pos_idx,
                    "o_idx": at_idx
                }
                
            elif atom_type == "OCC":  # ring O or glycosidic O
                if at_idx in flatten_O_ring:
                    # ring O, determine C1-C5 next
                    ring_idx = flatten_O_ring.index(at_idx)
                    neighbor_c1_idx = info["C_chain"][ring_idx][0] #C1
                    neighbor_c2_idx = info["C_chain"][ring_idx][4] #C5
                else:
                    # glycosidict O
                    #O_glycosidic = set(info["O_on_C_chain"][0], info["O_on_C_chain"][1])
                    neighbor_c1_pos = info["O_on_C_chain"][0].index(at_idx)
                    neighbor_c2_pos = info["O_on_C_chain"][1].index(at_idx)

                    neighbor_c1_idx = info["C_chain"][0][neighbor_c1_pos]
                    neighbor_c2_idx = info["C_chain"][1][neighbor_c2_pos]

                
                # Convert angle from degrees to radians
                angle_rad = angle_val * (np.pi / 180)
                
                # Calculate the proton position
                proton_pos = self._calculate_proton_position(at_idx, [neighbor_c1_idx, neighbor_c2_idx], coor, angle_rad,
                                                          self.proton_generator.params["proton_Oring_dist"])
                
                atom_metadata = {
                    "site": "ring_O",
                    "o_idx": at_idx
                }
                
            elif atom_type == "OC":  # NAc O
                # Check if NAc group exists
                if 'C_NH_CO_CHHH' not in info or not info['C_NH_CO_CHHH']:
                    logger.warning("No NAc groups found for protonation")
                    return None
                
                # Get a random NAc group
                nac_group = info["C_NH_CO_CHHH"][0]
                             
                # Extract atom indices from the NAc group
                C_id, N_id, HN_id, CO_id, OC_id, C_methyl_id, H_methyl_1, H_methyl_2, H_methyl_3 = nac_group
                
                # Set up a spherical coordinates around O carbonyl
                z_axis = unit_vector(coor[OC_id] - coor[CO_id])
                
                # Find the X-axis vector perpendicular to Z
                temp_vec = coor[C_id] - coor[CO_id]
                # Remove any component along z-axis to get the perpendicular vector
                temp_vec -= np.dot(temp_vec, z_axis) * z_axis
                
                if np.linalg.norm(temp_vec) < 1e-6:
                    # If temp_vec is too small, try another reference point
                    temp_vec = coor[N_id] - coor[CO_id]
                    temp_vec -= np.dot(temp_vec, z_axis) * z_axis
                
                x_axis = unit_vector(temp_vec)
                # y-axis completes the right-handed coordinate system
                y_axis = np.cross(z_axis, x_axis)
                
                # Define the theta angle (from z-axis)
                theta = 0.37 * np.pi
                
                # Define 5 evenly distributed phi angles around z-axis (use angle_idx to choose one)
                phi_values = [i * 2 * np.pi / 5 for i in range(5)]
                phi_val = phi_values[int(angle_val) % 5]  # angle_val is the index of phi angle
                
                # Define r values
                r = self.proton_generator.params["proton_Oring_dist"]
                
                # Convert spherical coordinates to Cartesian coordinates
                x_local = r * math.sin(theta) * math.cos(phi_val)
                y_local = r * math.sin(theta) * math.sin(phi_val)
                z_local = r * math.cos(theta)
                
                # Transform to global coordinates
                proton_vector = x_local * x_axis + y_local * y_axis + z_local * z_axis
                proton_pos = coor[OC_id] + proton_vector
                
                atom_metadata = {
                    "site": "NAc_O",
                    "phi_idx": int(angle_val)
                }
                
            elif atom_type == "N":  # NAc N
                # Check if NAc group exists
                if 'C_NH_CO_CHHH' not in info or not info['C_NH_CO_CHHH']:
                    logger.warning("No NAc groups found for protonation")
                    return None
                
                # Get a random NAc group
                nac_group = info["C_NH_CO_CHHH"][0]
                
                # Extract atom indices from the NAc group
                C_id, N_id, HN_id, CO_id, OC_id, C_methyl_id, H_methyl_1, H_methyl_2, H_methyl_3 = nac_group
                
                # Calculate vectors
                C2CO_vec = coor[CO_id] - coor[C_id]
                NH_vec = coor[HN_id] - coor[N_id]
                
                # Convert angle from degrees to radians
                angle_rad = angle_val * (np.pi / 180)
                
                # Rotate NH vector around C2CO axis
                rotated_NH_vec = np.dot(rotation_matrix_numpy(C2CO_vec, angle_rad), NH_vec)
                proton_pos = coor[N_id] + rotated_NH_vec
                
                atom_metadata = {
                    "site": "NAc_N"
                }
                
            else:
                logger.warning(f"Invalid atom_type: {atom_type}")
                return None
            
            if proton_pos is None:
                logger.warning(f"Failed to calculate proton position for atom_type {atom_type}, angle {angle_val}")
                return None
                
            # Create the protonated structure
            # Add the proton to the structure
            proton_coor = ("H", *proton_pos)
            protonated_frame = current_frame.copy() + [proton_coor]
            
            # Check if physically reasonable
            if self.check_physical:
                check_result = is_physical_geometry(protonated_frame, **self.check_physical_kwargs)
                if check_result != "normal":
                    logger.debug(f"Generated protonated structure is not physically reasonable")
                    return None
            
            # Convert to MolecularStructure
            mol_structure = MolecularStructure.from_xyz_list(protonated_frame)
            mol_structure.metadata = structure.metadata.copy() if structure.metadata else {}
            mol_structure.metadata.update({
                "operation": "add_proton",
                "proton_grid_idx": proton_grid_idx,
                "proton_description": "none",
                "angle_val": angle_val,
                **atom_metadata
            })
            
            return mol_structure
            
        except Exception as e:
            logger.error(f"Error generating protonated structure: {str(e)}", exc_info=True)
            return None
    
    def _calculate_proton_position(self, O_id, connecting_atoms, coor, angle, HO_len):
        """Calculate H+ position around O with specified parameters."""
        try:
            # Vector between connecting atoms
            c1_c2_vec = coor[connecting_atoms[1]] - coor[connecting_atoms[0]]
            
            # Get vector for rotation
            oc1_vec = coor[connecting_atoms[0]] - coor[O_id]
            oc2_vec = coor[connecting_atoms[1]] - coor[O_id]
            
            # Use average vector for better proton placement
            avg_oc_vec = (oc1_vec + oc2_vec) / 2
            
            # Add H+ at the specified angle around the oxygen

            rotated_vec = np.dot(rotation_matrix_numpy(c1_c2_vec, angle), avg_oc_vec)
            proton_pos = coor[O_id] + unit_vector(rotated_vec) * HO_len
            
            return proton_pos
            
        except Exception as e:
            logger.error(f"Error calculating H+ position around O: {str(e)}")
            return None
    
    def _load_structure(self, structure_path: str) -> Optional[MolecularStructure]:
        """Load a structure from file."""
        try:
            # Implementation depends on your file loading utilities
            atoms = read(structure_path)
            
            # Get a random frame if there are multiple
            structure = MolecularStructure.from_ase_atoms(atoms)
            structure.metadata['source_file'] = structure_path      
            return structure

        except Exception as e:
            logger.error(f"Error loading structure from {structure_path}: {str(e)}")
            return None
    
    def _accept_move(self, proposed_energy: float) -> bool:
        """Decide whether to accept a proposed move using the Metropolis criterion."""
        # Always accept if the energy is lower
        if proposed_energy < self.current_energy:
            return True
            
        # Metropolis criterion
        energy_diff = proposed_energy - self.current_energy
        boltzmann_factor = np.exp(-energy_diff / (8.314462618e-3 * self.temperature))  # kB in kJ/(mol·K)
        
        return random.random() < boltzmann_factor
    
    def _append_to_output_file(self, structure: MolecularStructure, energy: float, step: int, accepted: bool):
        """Append a structure to the output XYZ file."""
        try:
            xyz_str = structure.to_xyz_str()
            
            # Add a comment line with the step, energy, and acceptance
            lines = xyz_str.strip().split("\n")
            
            # Replace the blank comment line with information
            grid_point = structure.metadata.get('grid_point', 'unknown')
            base_file = os.path.basename(structure.metadata.get('base_structure', 'unknown'))

            # Get operation-specific information
            operations_info = []
            for op in self.operation_sequence:
                if op == OperationType.FLIP and "angle_val" in structure.metadata:
                    flip_angle = structure.metadata.get("angle_val", "unknown")
                    operations_info.append(f"FlipAngle= {flip_angle}")
                elif op == OperationType.ATTACH_ROTATE and "rot_angle" in structure.metadata:
                    attach_angle = structure.metadata.get("rot_angle", "unknown")
                    operations_info.append(f"AttachAngle= {attach_angle}")
                elif op == OperationType.ADD_PROTON and "proton_description" in structure.metadata:
                    proton_desc = structure.metadata.get("proton_description", "unknown")
                    proton_site = structure.metadata.get("site", "unknown")
                    operations_info.append(f"ProtonAt= {proton_site}")
                    operations_info.append(f"ProtonDesc= {proton_desc}")
            
            # Build comment line with all available information
            comment_parts = [f"Step= {step}", f"eng= {energy:.6f}", f"Base= {base_file}", 
                            f"GridPoint= {grid_point}"] + operations_info
            comment_line = " ".join(comment_parts)
            
            # Reassemble the XYZ file content
            lines[1] = comment_line
            xyz_str = "\n".join(lines)
            
            # Append to the output file
            with open(self.output_xyz, 'a') as f:
                f.write(xyz_str + "\n")
                
        except Exception as e:
            logger.error(f"Error appending to output file: {str(e)}")
    
    def _get_operations_description(self, grid_point: Tuple) -> Dict[str, Any]:
        """Generate a human-readable description of the operations at a grid point."""
        base_idx = grid_point[0]
        description = {
            "base_structure": self.base_structures[base_idx],
        }
        
        grid_idx = 1
        for op_idx, op_type in enumerate(self.operation_sequence):
            if op_type == OperationType.FLIP:
                flip_angle_idx = grid_point[grid_idx]
                grid_idx += 1
                description["flip_angle"] = self.flip_angles[flip_angle_idx]
            elif op_type == OperationType.ATTACH_ROTATE:
                attach_angle_idx = grid_point[grid_idx]
                grid_idx += 1
                description["attach_angle"] = self.attach_angles[attach_angle_idx]
            elif op_type == OperationType.ADD_PROTON:
                proton_grid_idx = grid_point[grid_idx]
                grid_idx += 1
                
                if 0 <= proton_grid_idx < len(self.proton_grid):
                    atom_type, angle_val, proton_desc = self.proton_grid[proton_grid_idx]
                    description["proton_grid_idx"] = proton_grid_idx
                    description["proton_description"] = proton_desc
                    description["proton_atom_type"] = "none"
                    description["proton_angle"] = angle_val
                else:
                    description["proton_grid_idx"] = "invalid"
        
        return description

    def _traj2xyz(self, structure: MolecularStructure, opt_traj: str):
        """Store traj file"""
        # grid-based naming scheme
        grid_point = structure.metadata.get("grid_point", "unknown")
        grid_point_str = "-".join(map(str, grid_point)) if isinstance(grid_point, tuple) else str(grid_point)
        trajectory_file_name = f"grid_{grid_point_str}.xyz"
        saved_trajectory_path = os.path.join(self.trajectory_dir, trajectory_file_name)

        traj = read(opt_traj, index=":")
        with open(saved_trajectory_path, "a") as f:
            for snapshot in traj:
                energy = snapshot.get_potential_energy() * eV_TO_Ha_conversion
                forces = snapshot.get_forces() * eV_TO_Ha_conversion
                snapshot.info.update({"eng": energy})

                snapshot_structure = MolecularStructure.from_ase_atoms(snapshot, include_force=True)
                snapshot_structure.set_forces(forces)

                xyz_str = snapshot_structure.to_xyz_str(include_forces=True)
                lines = xyz_str.strip().split("\n")
                lines[1] = f"eng= {energy} Properties=species:S:1:pos:R:3:force:R:3"

                xyz_str = "\n".join(lines) + "\n"

                f.write(xyz_str)

        #shutil.copy2(opt_traj, saved_trajectory_path)
        logger.info(f"Saved trajectory file to {saved_trajectory_path}")
        return saved_trajectory_path
    
    def get_best_structure(self) -> MolecularStructure:
        """Get the best structure found so far."""
        return self.best_structure
    
    def get_history(self) -> List[Dict]:
        """Get the history of accepted moves."""
        return self.history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from the basin hopping run."""
        return self.stats
    
    def set_temperature(self, temperature: float):
        """Set the temperature parameter for the Metropolis criterion."""
        self.temperature = temperature
        
    def set_base_structures(self, base_structures: List[str]):
        """Set the base structures."""
        self.base_structures = base_structures
        self.flip_generator.set_base_structures(base_structures)
        self.attach_rotate_generator.set_base_structures(base_structures)
        self.proton_generator.set_base_structures(base_structures)
        
    def set_seed_structure(self, seed_structure: str):
        """Set the seed structure."""
        self.seed_structure = seed_structure
        self.attach_rotate_generator.set_seed_structures([seed_structure])
        
    def set_model_params(self, model_params: Dict[str, Any]):
        """Set the NNP model parameters and reload the model."""
        self.model_params = model_params
        self._load_nnp_model()
        
    def set_optimize_params(self, optimize_params: Dict[str, Any]):
        """Set the optimization parameters."""
        self.optimize_params = optimize_params
        
    def set_max_rejected(self, max_rejected: int):
        """Set the maximum number of consecutive rejected moves."""
        self.max_rejected = max_rejected
        
    def set_flip_angles(self, angles: List[float]):
        """Set the grid points for rotation angles after flipping."""
        self.flip_angles = angles
        self.flip_generator.set_params(rotating_angle=angles)
        
    def set_attach_angles(self, angles: List[float]):
        """Set the grid points for rotation angles after attaching."""
        self.attach_angles = angles
        self.attach_rotate_generator.set_params(angle_list=angles)
        
    def set_proton_grid(self, proton_grid: List[Tuple[int, float, str]]):
        """Set the grid points for proton addition.
        
        Args:
            proton_grid: List of tuples (atom_type, angle_value, description)
                where atom_type is 0=hydroxyl O, 1=ring O, 2=NAc O, 3=NAc N
        """
        self.proton_grid = proton_grid
        
    def set_operation_sequence(self, operation_sequence: List[OperationType]):
        """Set the operation sequence."""
        self.operation_sequence = operation_sequence

    def set_save_trj(self, save_trj: bool):
        """Enable or disable trajectory saving."""
        self.save_trj = save_trj

    def set_trajectory_dir(self, trajectory_dir: str):
        """Set the directory where trajectory files will be saved."""
        self.trajectory_dir = trajectory_dir
        if self.save_trj and not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir, exist_ok=True)
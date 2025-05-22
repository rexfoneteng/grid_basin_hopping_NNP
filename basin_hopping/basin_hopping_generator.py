#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-17 16:34:44
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-21 17:04:32
# @Description: Basin Hopping Engine

import numpy as np
import logging
import random
import os
from typing import List, Dict, Any, Tuple, Optional, Union, Callable, Union
import time
from ase.io import read
from rotational_matrix import rotation_matrix_numpy

from generators.base_generator import BaseGenerator
from generators.flip_generator import FlipGenerator
from generators.attach_rotate_generator import AttachRotateGenerator
from generators.proton_generator import ProtonGenerator
from core.molecular_structure import MolecularStructure
from core.constants import DEFAULT_FLIP_PARAMS, DEFAULT_ATTACH_ROTATE_PARAMS, DEFAULT_ADD_PROTON_PARAMS, eV_TO_Ha_conversion
from basin_hopping.operation_type import OperationType
from models.optimizer_factory import OptimizerFactory

from xyz_physical_geometry_tool_mod import validate_structure
from xyz_tools import xyz2list

logger = logging.getLogger(__name__)

class BasinHoppingGenerator(BaseGenerator):
    """
    Generator that performs grid-based basin hopping by applying a sequence of operations
    in sequence to systematically explore the energy landscape, with local optimization.
    """

    def __init__(self, 
                 base_structures: List[str],
                 seed_structures: Union[str, None] = None,
                 temperature: float = 300.0,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None,
                 optimizer_type: str = "nnp",
                 optimizer_params: Optional[Dict[str, Any]] = None,
                 optimize_params: Optional[Dict[str, Any]] = None,
                 input_xyz: str = "input.xyz",
                 accepted_xyz: str = "accepted_structure.xyz",
                 rejected_xyz: str = "rejected_structure.xyz",
                 max_rejected: int = 100,
                 operation_sequence: List[OperationType] = None,
                 operation_params: Union[List, Dict] = None,
                 attach_params: Dict = {"angle_list": [(0, 0), (0, 120), (0, 240), (120, 0), (120, 120), (120, 240), (240, 0), (240, 120), (240, 240)]},
                 save_trj: bool = False,
                 trajectory_dir: str = "opt_trj",
                 energy_outliner_threshold=0.06,
                 skip_local_optimization: bool = False, **kwargs):
        """Initialize the basin hopping generator.

        Args:
            base_structures: List of paths to base structure files
            seed_structures: Path to the seed structure file for attachment
            temperature: Temperature parameter for Metropolis criterion (K)
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
            model_params: Parameters for the NNP model
            optimize_params: Parameters for the optimization
            accepted_xyz: Path to the output XYZ file where all local minima will be appended
            max_rejected: Maximum number of consecutive rejected moves before stopping
            operation_sequence: Sequence of operations to apply (default: FLIP, ATTACH_ROTATE)
            operation_params: List of parameters corresponds to the each operations listed in the operation_sequence
            skip_local_optimization: Bool whether to perform the local optimization at each BH move,
                                    this option is specifically useful when user simply want to generate initial structures.
        """
        super().__init__(check_physical, check_physical_kwargs)
        
        self.base_structures = base_structures
        self.seed_structures = seed_structures
        self.temperature = temperature
        self.max_rejected = max_rejected
        self.input_xyz = input_xyz
        self.accepted_xyz = accepted_xyz
        self.rejected_xyz = rejected_xyz

        # Optimization trajectory saving
        self.save_trj = save_trj
        self.trajectory_dir = trajectory_dir
        if self.save_trj and not os.path.exists(trajectory_dir):
            os.makedirs(self.trajectory_dir, exist_ok=True)

        # Set operation sequence
        self.operation_sequence = operation_sequence
        self.operation_params = operation_params
        if isinstance(self.operation_params, dict):
            self.operation_params = [self.operation_params]

        assert len(self.operation_sequence) == len(self.operation_params), \
            "the len of operation_sequence is not equal to the len of operation_params"

        self.energy_outliner_threshold = energy_outliner_threshold
        self.skip_local_optimization = skip_local_optimization
        
        # Grid definitions for each operation
        self.flip_angles = [0, 120, 240]  # 3 grid points for rotation after flip (in degrees)
        self.attach_params = attach_params

        if isinstance(self.seed_structures, str):
            self.seed_structures = [self.seed_structures]
        
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
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params or {}        
        self.optimize_params = optimize_params or {}

        self.optimizer = OptimizerFactory.create_optimizer(
            optimizer_type=self.optimizer_type,
            params=self.optimizer_params)

        if not self.optimizer:
            logger.warning(f"Failed to create optimizer of type '{optimizer_type}'. "
                          "Basin hopping will run without optimization.")
            
        self.attach_rotate_generator = AttachRotateGenerator(
            base_structures=self.base_structures,
            seed_structures=self.seed_structures,
            params=DEFAULT_ATTACH_ROTATE_PARAMS.copy(),
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)
        
        self.proton_generator = ProtonGenerator(
            base_structures=self.base_structures,
            params=DEFAULT_ADD_PROTON_PARAMS.copy(),
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)

        self.generators = {
            OperationType.ATTACH_ROTATE: self.attach_rotate_generator,
            OperationType.ADD_PROTON: self.proton_generator}

        if OperationType.FLIP in self.operation_sequence:
            from generators.flip_generator import FlipGenerator
            self.flip_generator = FlipGenerator(
                                input_structure=self.base_structures,
                                params=DEFAULT_FLIP_PARAMS.copy(),
                                check_physical=self.check_physical,
                                check_physical_kwargs=self.check_physical_kwargs)
            self.generators[OperationType.FLIP] = self.flip_generator

        if OperationType.FUNCTIONAL_TO_CH in self.operation_sequence:
            from generators.functional_to_ch_generator import FunctionalToChGenerator
            self.functional_to_ch_generator = FunctionalToChGenerator(
                                input_structure=self.base_structures,
                                check_physical=self.check_physical,
                                check_physical_kwargs=self.check_physical_kwargs)
            self.generators[OperationType.FUNCTIONAL_TO_CH] = self.functional_to_ch_generator

        if OperationType.CH_TO_METHYL in self.operation_sequence:
            from generators.ch_to_methyl_generator import ChToMethylGenerator
            self.ch_to_methyl_generator = ChToMethylGenerator(
                                input_structure=self.base_structures,
                                check_physical=self.check_physical,
                                check_physical_kwargs=self.check_physical_kwargs)
            self.generators[OperationType.CH_TO_METHYL] = self.ch_to_methyl_generator
        # Map operation types to generator instances

        self.attach_rotate_generator.set_params(
            angle_list=self.attach_params
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
        if self.current_structure is None and not self.skip_local_optimization:
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
            
            with open("./input.xyz", "a") as f:
                f.write(proposed_structure.to_xyz_str(info=f"Step= {step}"))

            if not self.skip_local_optimization:
                # Optimize the proposed structure
                optimized_structure, optimized_energy, trajectory_path = self._optimize_structure(proposed_structure)
            
                # Validate the optimized_structure
                optimized_structure = \
                    validate_structure(optimized_structure, 
                                       cutoff_sq_kwargs={"O_HO": 1.44, "O_H": 1.44})
                # Validate the energy drop, if the energy drop more than threshold, 
                # then considered it as outliner
                is_valid_energy = self._is_normal_energy_drop(optimized_energy)

                # Skip if optimization failed
                if optimized_structure is None or not is_valid_energy:
                    logger.warning(f"Step {step}: Optimization failed")
                    self.stats["optimization_failures"] += 1
                    continue
                    
                # Accept or reject the move
                accepted = self._accept_move(optimized_energy)
                
                # Update state if accepted
                if accepted:
                    self._append_to_output_file(optimized_structure, optimized_energy, step, True)
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
                    self._append_to_output_file(optimized_structure, optimized_energy, step, False)
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
            else: # When user simply want to generate the initial guesses
                step += 1
                if step >= n_steps:
                    return 0
    
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
        # Generate a valid structure with a random grid point
        structure, grid_point = self._get_valid_structure_and_grid_point()

        if structure is None:
            raise RuntimeError("Failed to initialize basin hopping with a valid structure")
            
        # Optimize the structure
        #with open("./input.xyz", "a") as f:
        #    f.write(structure.to_xyz_str())
        # Optimize the structure
        with open(self.input_xyz, "a") as f:
            f.write(structure.to_xyz_str(info="Step= -1"))

        optimized_structure, optimized_energy, trajectory_path = self._optimize_structure(structure)

        # Validate the optimized_structure 
        optimized_structure = \
            validate_structure(optimized_structure, 
                               cutoff_sq_kwargs={"O_HO": 1.44, "O_H": 1.44})
        
        if optimized_structure is None:
            logger.error("Initial structure optimization failed")
            optimized_energy = float("inf")
            #raise RuntimeError("Failed to optimize initial structure")
            
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
    
    def _optimize_structure(self, structure):
        """Optimize a structure using the configured optimizer.
        
        Args:
            structure: Structure to optimize
            
        Returns:
            Tuple of (optimized structure, energy, trajectory path)
        """
        if not self.optimizer:
            # If no optimizer, just return the structure with a random energy
            if hasattr(self, 'energy_function') and self.energy_function:
                energy = self.energy_function(structure)
            else:
                energy = random.random() * 100
            return structure, energy, None
        
        try:
            optimized_structure, energy, trajectory_path = self.optimizer.optimize(
                structure, **self.optimize_params)
            
            # Check if optimization was successful
            if optimized_structure is None:
                logger.error("Optimization failed")
                return None, float('inf'), None
            
            # Add optimizer type to metadata
            optimized_structure.metadata.update({
                "optimizer_type": self.optimizer_type
            })
            
            return optimized_structure, energy, trajectory_path
            
        except Exception as e:
            logger.error(f"Error in structure optimization: {str(e)}", exc_info=True)
            return None, float('inf'), None
    
    def _propose_move(self) -> Tuple[Optional[MolecularStructure], Tuple]:
        """Propose a new move by selecting a random neighbor on the grid.
        
        Returns:
            Tuple of (proposed structure, grid point)
        """
        # Get a random grid point
        structure, grid_point = self._get_valid_structure_and_grid_point()    
        return structure, grid_point

    def _get_valid_structure_and_grid_point(self, max_attempts=50) -> Tuple[Optional[MolecularStructure], Optional[Tuple]]:
        """Try to generate a valid structure with a random grid point.
        
        Args:
            max_attempts: Maximum number of attempts to generate a valid structure
            
        Returns:
            Tuple of (structure, grid_point) or (None, None) if all attempts failed
        """
        for attempt in range(max_attempts):
            grid_point = self._get_random_grid_point()
            structure = self._apply_operations(grid_point)
            
            if structure is not None:
                return structure, grid_point
                
        # If all attempts failed
        return None, None
    
    def _get_random_grid_point(self) -> Tuple:
        """Get a completely random grid point.
        
        Returns:
            Tuple of grid point indices for each operation in the sequence
        """
        grid_point = []
        
        # First component: random base structure index
        base_idx = random.randint(0, len(self.base_structures) - 1)

        if self.seed_structures:
            seed_idx = random.randint(0, len(self.seed_structures) - 1)
        else:
            seed_idx = 0

        grid_point.extend([base_idx, seed_idx])
        
        # Add grid point components for each operation in the sequence
        for op, op_params in zip(self.operation_sequence, self.operation_params):
            if op == OperationType.FLIP:
                # Random flip angle index
                flip_angle_idx = \
                    random.randint(0, len(op_params["rotating_angle"]) - 1)
                grid_point.append(flip_angle_idx)
            elif op == OperationType.ATTACH_ROTATE:
                # Random attach angle index
                attach_angle_idx = random.randint(0, len(self.attach_rotate_generator.params["angle_list"]) - 1)
                grid_point.append(attach_angle_idx)
            elif op == OperationType.ADD_PROTON:
                # Random proton grid index (combines atom type and angle)
                proton_grid_idx = random.randint(0, len(self.proton_grid) - 1)
                grid_point.append(proton_grid_idx)
            elif op == OperationType.FUNCTIONAL_TO_CH:
                grid_point.append(0) # Only 1 grid with id 0
            elif op == OperationType.CH_TO_METHYL:
                grid_point.append(0) # Only 1 grid with id 0
                
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
            base_idx, seed_idx = grid_point[:2]
            
            # Load the base structure
            if base_idx >= len(self.base_structures):
                logger.error(f"Invalid base_idx {base_idx} (max: {len(self.base_structures)-1})")
                return None

            if self.seed_structures  and seed_idx >= len(self.seed_structures):
                logger.error(f"Invalid seed index {seed_idx} (max: {len(self.seed_structures)-1})")
                
            base_structure_path = self.base_structures[base_idx]
            structure = self._load_structure(base_structure_path)

            if self.seed_structures:
                seed_structure_path = self.seed_structures[seed_idx]
            else:
                seed_structure_path = None
   
            if not structure:
                logger.error(f"Failed to load base structure: {base_structure_path}")
                return None
                
            # Apply each operation in sequence
            grid_idx = 2  # Start after base_idx and seed_idx
            
            for op_idx, (op_type, op_params) in \
                enumerate(zip(self.operation_sequence, self.operation_params)):
                if op_type == OperationType.FLIP:
                    # Apply flip operation
                    flip_angle_idx = grid_point[grid_idx]
                    grid_idx += 1
                    #flip_angle = self.flip_angles[flip_angle_idx]
                    flip_angle = op_params["rotating_angle"][flip_angle_idx]
                    self.flip_generator.set_params(**op_params)
                    self.flip_generator.set_input_structure(structure)
                    structure = self._get_specific_flipped_structure(flip_angle)
                    
                    if not structure:
                        logger.warning(f"Failed to generate flipped structure with angle {flip_angle}")
                        return None
                        
                elif op_type == OperationType.ATTACH_ROTATE:
                    # Apply attach-rotate operation
                    attach_angle_idx = grid_point[grid_idx]
                    grid_idx += 1
                    
                    attach_angle = self.attach_rotate_generator.params["angle_list"][attach_angle_idx]
                    structure = self._get_specific_attached_structure(structure, attach_angle, seed_structure_path)
                    
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

                elif op_type == OperationType.FUNCTIONAL_TO_CH:
                    grid_idx += 1
                    self.functional_to_ch_generator.set_input_structure(structure)
                    structure = self._get_specific_functional_to_ch()

                    if not structure:
                        logger.warning(f"Failed to add proton to structure using grid point {proton_grid_idx}")
                        return None

                elif op_type == OperationType.CH_TO_METHYL:
                    grid_idx += 1
                    #print(structure)
                    self.ch_to_methyl_generator.set_input_structure(structure)
                    structure = self._get_specific_ch_to_methyl()

                    if not structure:
                        logger.warning(f"Failed to add proton to structure using grid point {proton_grid_idx}")
                        return None
            # Add grid point to metadata
            structure.metadata['grid_point'] = grid_point
            structure.metadata['base_structure'] = base_structure_path
            structure.metadata["seed_structure"] = seed_structure_path
            
            return structure
            
        except Exception as e:
            logger.error(f"Error applying operations: {str(e)}", exc_info=True)
            return None

    def _get_specific_flipped_structure(self, flip_angle: float) -> Optional[MolecularStructure]:
        """Generate a structure with a specific flip angle."""
        return self.flip_generator.generate_grid(flip_angle)

    def _get_specific_attached_structure(self, structure: MolecularStructure, attach_angle: float, seed_structure: str) -> Optional[MolecularStructure]:
        """Generate a structure with a specific attach rotation angle."""
        return self.attach_rotate_generator.generate_grid(structure, attach_angle, seed_structure)
            
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
        if proton_grid_idx < 0 or proton_grid_idx >= len(self.proton_grid):
            logger.warning(f"Invalid proton grid index: {proton_grid_idx}")
            return None
        return self.proton_generator.generate_grid(structure, self.proton_grid[proton_grid_idx])

    def _get_specific_functional_to_ch(self) -> Optional[MolecularStructure]:
        """Generate a structure with a specific functional group to CH truncation"""
        return self.functional_to_ch_generator.generate_grid()

    def _get_specific_ch_to_methyl(self) -> Optional[MolecularStructure]:
        """Generate a structure with a specific functional group to CH truncation"""
        return self.ch_to_methyl_generator.generate_grid()
    
    def _load_structure(self, structure_path: str) -> Optional[MolecularStructure]:
        """Load a structure from file."""
        try:
            # Implementation depends on your file loading utilities
            try:
                atoms = read(structure_path)
                structure = MolecularStructure.from_ase_atoms(atoms)
            except ValueError:
                current_frame = xyz2list(structure_path)
                structure = MolecularStructure.from_xyz_list(current_frame)

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

    def _is_normal_energy_drop(self, proposed_energy: float) -> bool:
        """ Validate the energy drop if it is normal """           
        if proposed_energy < self.current_energy and self.current_energy != float("inf"):
            if (self.current_energy - proposed_energy) < self.energy_outliner_threshold:
                return True
            return False
        return True
    
    def _append_to_output_file(self, structure: MolecularStructure, energy: float, step: int, accept: bool):
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
            if accept:
                with open(self.accepted_xyz, "a") as f:
                    f.write(xyz_str + "\n")
            else:
                with open(self.rejected_xyz, "a") as f:
                    f.write(xyz_str + "\n")
                
        except Exception as e:
            logger.error(f"Error appending to output file: {str(e)}")
    
    def _get_operations_description(self, grid_point: Tuple) -> Dict[str, Any]:
        """Generate a human-readable description of the operations at a grid point."""
        base_idx, seed_idx = grid_point[:2]
        description = {
            "base_structure": self.base_structures[base_idx],
        }
        
        grid_idx = 2
        for op_idx, op_type in enumerate(self.operation_sequence):
            if op_type == OperationType.FLIP:
                flip_angle_idx = grid_point[grid_idx]
                grid_idx += 1
                description["flip_angle"] = op_params["rotating_angle"][flip_angle_idx]
            elif op_type == OperationType.ATTACH_ROTATE:
                attach_angle_idx = grid_point[grid_idx]
                grid_idx += 1
                description["attach_angle"] = self.attach_rotate_generator.params["angle_list"][attach_angle_idx]
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

                xyz_str = "\n".join(lines)

                f.write(xyz_str + "\n")

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
        
    def set_seed_structures(self, seed_structures: List):
        """Set the seed structure."""
        self.seed_structures = seed_structures
        if isinstance(self.seed_structures, str):
            self.seed_structures = [self.seed_structures]
        self.attach_rotate_generator.set_seed_structures(seed_structures)
        
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

    def set_flip_params(self, **kwargs):
        self.flip_generator.set_params(**kwargs)

    def set_attach_params(self, angles):
        """Set the grid points for rotation angles after attaching."""
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

    def set_kwargs(self, kwargs: Dict):
        """ Set extra parameters"""
        self.__dict__.update(kwargs)
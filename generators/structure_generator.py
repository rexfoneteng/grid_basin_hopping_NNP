#!/usr/bin/env python3

import os
import numpy as np
import logging
import math
from ase.io import read

from enum import Enum
from typing import List, Dict, Optional, Union, Any, Tuple, Iterator
from joblib import Parallel, delayed

from core.constants import (
    DEFAULT_FLIP_PARAMS,
    DEFAULT_ATTACH_ROTATE_PARAMS,
    DEFAULT_ADD_PROTON_PARAMS
    )
from generators.base_generator import BaseGenerator
from generators.flip_generator import FlipGenerator
from generators.attach_rotate_generator import AttachRotateGenerator
from generators.proton_generator import ProtonGenerator
from core.molecular_structure import MolecularStructure

# Configure logging
logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Enum for the type of structure generation operation"""
    FLIP = "flip"
    ATTACH_ROTATE = "attach_rotate"
    ADD_PROTON = "add_proton"

class StructureGenerator(BaseGenerator):
    """
    A unified structure generator that applies a sequence of operations
    to create a tree of molecular structures.

    Each operation in the sequence takes all structures from the previous step
    and produces multiple outputs for each input, resulting in a branching tree
    of structures.
    """

    def __init__(self,
                 operation_sequence: List[OperationType] = None,
                 base_structures: Optional[List[str]] = None,
                 seed_structures: Optional[List[str]] = None,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None,
                 tmp_dir: Optional[str] = None,
                 n_jobs: int = 1):
        """Initialize the structure generator.

        Args:
            operation_sequence: Sequence of operations to perform in order
            base_structures: List of paths to base structure files
            seed_structures: List of paths to seed structure files (for attach-rotate)
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
            tmp_dir: Directory for temporary files
            n_jobs: Number of parallel jobs for processing base structures
        """
        super().__init__(check_physical, check_physical_kwargs, tmp_dir)

        self.operation_sequence = operation_sequence or []

        if not self.operation_sequence:
            raise ValueError("No operation sequence provided")

        self.base_structures = base_structures or []
        self.seed_structures = seed_structures or []
        self.n_jobs = n_jobs

        # Default parameters for operations
        self.flip_params = DEFAULT_FLIP_PARAMS.copy()
        self.attach_rotate_params = DEFAULT_ATTACH_ROTATE_PARAMS.copy()
        self.add_proton_params = DEFAULT_ADD_PROTON_PARAMS.copy()

        # Convert angle from degrees to radians for proton addition
        self.add_proton_params["angle_list"] = \
            [math.radians(ele) for ele in self.add_proton_params["angle_list"]]
        self.add_proton_params["angle_around_Oring"] = \
            [math.radians(ele) for ele in self.add_proton_params["angle_around_Oring"]]

        # Initialize specific generators
        self.flip_generator = FlipGenerator(
            base_structures=self.base_structures,
            params=self.flip_params,
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)

        self.attach_rotate_generator = AttachRotateGenerator(
            base_structures=self.base_structures,
            seed_structures=self.seed_structures,
            params=self.attach_rotate_params,
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs)

        self.proton_generator = ProtonGenerator(
            base_structures=self.base_structures,
            params=self.add_proton_params,
            check_physical=self.check_physical,
            check_physical_kwargs=self.check_physical_kwargs,
            tmp_dir=self.tmp_dir)

        # Mapping from operation type to generator
        self.generators = {
            OperationType.FLIP: self.flip_generator,
            OperationType.ATTACH_ROTATE: self.attach_rotate_generator,
            OperationType.ADD_PROTON: self.proton_generator}

    def __call__(self) -> MolecularStructure:
        """Generate a structure by applying the operation sequence.
        
        Returns:
            A randomly selected structure from all generated structures
        """
        all_structures = self.generate_structure()
        if not all_structures:
            logger.warning("No structures were generated")
            return None
        
        # Return a random structure from all generated structures
        return np.random.choice(all_structures)

    def process_base_structure(self, structure_path: str) -> List[MolecularStructure]:
        """
        Process a single base structure through the operation sequence.
        
        Args:
            structure_path: Path to the base structure file
            
        Returns:
            List of generated structures after applying all operations
        """
        try:
            # Set the current base structure for all generators
            self.flip_generator.set_current_base_structure(structure_path)
            self.attach_rotate_generator.set_current_base_structure(structure_path)
            self.proton_generator.set_current_base_structure(structure_path)

            # Start with a single structure loaded from file
            initial_structure = self._load_structure(structure_path)
            current_structures = [initial_structure]

            # Apply each operation in sequence
            for operation_idx, operation_type in enumerate(self.operation_sequence):
                if not current_structures:
                    logger.warning(f"No structures to process after operation {operation_idx-1}")
                    break
                    
                generator = self.generators[operation_type]
                next_structures = []

                # Apply current operation to all structures from previous step
                for input_structure in current_structures:
                    # Get output variants from this input structure
                    output_variants = self._apply_operation(generator, operation_type, input_structure)
                    
                    # Skip if no variants were produced
                    if not output_variants:
                        continue

                    # Add operation index to metadata
                    for variant in output_variants:
                        variant.metadata["operation_idx"] = operation_idx
                        variant.metadata["base_structure"] = structure_path

                    next_structures.extend(output_variants)

                current_structures = next_structures
                logger.debug(f"After operation {operation_type.value}, generated {len(current_structures)} structures")

                # If no structures were generated, stop the sequence
                if not current_structures:
                    logger.warning(f"No valid structures after operation {operation_type.value} for {structure_path}")
                    break

            return current_structures

        except Exception as e:
            logger.error(f"Error in processing {structure_path}: {str(e)}", exc_info=True)
            return []

    def generate_structure_iterator(self) -> Iterator[MolecularStructure]:
        """Generate structures one by one as an iterator.
        
        Yields:
            MolecularStructure: Generated structures one at a time
        """
        if not self.base_structures:
            raise ValueError("No base structures provided")
            
        logger.info(f"Processing {len(self.base_structures)} base structures with {len(self.operation_sequence)} operations")
        
        for structure_path in self.base_structures:
            try:
                logger.debug(f"Processing base structure: {structure_path}")
                
                # Process this base structure
                structures = self.process_base_structure(structure_path)
                
                # Yield each structure one by one
                for structure in structures:
                    yield structure
                    
            except Exception as e:
                logger.error(f"Error processing {structure_path}: {e}", exc_info=True)

    def _load_structure(self, structure_path: str) -> MolecularStructure:
        """Load a structure from a file path using ASE.
        
        Args:
            structure_path: Path to the structure file
            
        Returns:
            A MolecularStructure object
            
        Raises:
            Exception: If the file cannot be read or processed
        """
        try:
            # Read the file using ASE
            atoms = read(structure_path)
            
            # Extract atomic symbols
            atom_symbols = atoms.get_chemical_symbols()
            
            # Extract coordinates
            coordinates = atoms.get_positions()
            
            # Create the molecular structure
            molecular_structure = MolecularStructure(
                atoms=atom_symbols,
                coordinates=np.array(coordinates),
                atomic_numbers=np.array(atoms.get_atomic_numbers()),
                metadata={"source_file": structure_path, "frame_id": 0}
            )
            
            return molecular_structure
            
        except Exception as e:
            logger.error(f"Error loading structure from {structure_path}: {str(e)}", exc_info=True)
            raise

    def _apply_operation(self,
                         generator: Any,
                         operation_type: OperationType,
                         input_structure: MolecularStructure) -> List[MolecularStructure]:
        """Apply an operation to a structure, generating multiple variants."""
        try:
            # Each generator should have a generate_variants method that returns multiple structures
            if hasattr(generator, "generate_variants"):
                variants = generator.generate_variants(input_structure)
                # Make sure to return an empty list instead of None
                return [] if variants is None else variants
            else:
                logger.warning(f"Generator for {operation_type} does not support generating variants")
                result = generator()
                return [result] if result is not None else []
        except Exception as e:
            logger.error(f"Error in _apply_operation: {str(e)}", exc_info=True)
            return []  # Return empty list on error

    def generate_structure(self) -> List[MolecularStructure]:
        """
        Generate structures by processing all base structures through the operation sequence.

        Returns:
            List of all generated structures
        """
        if not self.base_structures:
            raise ValueError("No base structures provided")

        logger.info(f"Processing {len(self.base_structures)} base structures with {len(self.operation_sequence)} operations")

        # Parallelize
        if self.n_jobs != 1:
            all_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_base_structure)(structure_path)
                for structure_path in self.base_structures)
        else:
            all_results = [
                self.process_base_structure(structure_path) for structure_path in self.base_structures]

        # List flattening
        all_structures = [structure for sublist in all_results for structure in sublist]

        logger.info(f"Generated {len(all_structures)} from {len(self.base_structures)} base structures")

        return all_structures

    def set_operation_sequence(self, operation_sequence: List[OperationType]):
        """Set the operation sequence."""
        self.operation_sequence = operation_sequence
        return self

    def set_base_structures(self, base_structures: List[str]):
        """Set the base structures."""
        self.base_structures = base_structures
        self.flip_generator.set_base_structures(base_structures)
        self.attach_rotate_generator.set_base_structures(base_structures)
        self.proton_generator.set_base_structures(base_structures)
        return self

    def set_seed_structures(self, seed_structures: List[str]):
        """Set the seed structures for attach-rotate operation."""
        self.seed_structures = seed_structures
        self.attach_rotate_generator.set_seed_structures(seed_structures)
        return self

    def set_flip_params(self, **kwargs):
        """Set parameters for flip operation."""
        self.flip_params.update(kwargs)
        self.flip_generator.set_params(**kwargs)
        return self

    def set_attach_rotate_params(self, **kwargs):
        """Set parameters for attach-rotate operation."""
        self.attach_rotate_params.update(kwargs)
        self.attach_rotate_generator.set_params(**kwargs)
        return self

    def set_add_proton_params(self, **kwargs):
        """Set parameters for add-proton operation."""
        self.add_proton_params.update(kwargs)
        
        # Convert angle lists from degrees to radians if provided
        if "angle_list" in kwargs:
            self.add_proton_params["angle_list"] = \
                [math.radians(ele) for ele in kwargs["angle_list"]]
        if "angle_around_Oring" in kwargs:
            self.add_proton_params["angle_around_Oring"] = \
                [math.radians(ele) for ele in kwargs["angle_around_Oring"]]
                
        self.proton_generator.set_params(**self.add_proton_params)
        return self

    def set_n_jobs(self, n_jobs: int):
        """Set the number of parallel jobs."""
        self.n_jobs = n_jobs
        return self
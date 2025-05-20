#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-05-19 14:10:12
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-20 10:10:17
# @Description: transforms CH to CH3 group

import numpy as np
import logging
import math
from typing import List, Dict, Any, Tuple, Optional, Set, Union

from generators.base_generator import BaseGenerator
from core.molecular_structure import MolecularStructure

from xyz_physical_geometry_tool_mod import is_physical_geometry
from xyz_tools import Xyz

logger = logging.getLogger(__name__)


class ChToMethylGenerator(BaseGenerator):
    """Generator for transforming CH groups to CH3.
    
    This generator takes a pair of atom indices defining the vector of the resulting group
    and automatically determines which hydrogen to maintain while adding two more hydrogens.
    """

    def __init__(self,
                 input_structure: Union[List[str], MolecularStructure, str, None] = None,
                 params: Optional[Dict[str, Any]] = \
                    {"c_idx": 9,
                     "attach_kwargs": {"seed_align_list": [[0, 1]],
                                       "seed_del_list": []}},
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the ChToMethylGenerator.

        Args:
            input_structure: List of paths to base structure files
            params: Parameters for CH to CH3 transformation
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
        """
        super().__init__(check_physical, check_physical_kwargs)

        self.input_structure = input_structure
        self.params = params or {}

        if isinstance(self.input_structure, str):
            self.input_structure = self._load_structure(self.input_structure)

    def __call__(self) -> MolecularStructure:
        """Generate a structure by transforming a CH group to CH3.

        Returns:
            A new MolecularStructure with a CH group transformed to CH3
        """
        if not self.input_structure:
            raise ValueError("No base structures available for transformation")
            
        # Select a random base structure if none is currently selected        
        # Generate variants and return a random one
        variants = self.generate_variants()
        if variants:
            return np.random.choice(variants)
        else:
            logger.warning(f"No valid variants generated")
            return self()

    def _load_structure(self, structure_path: str) -> MolecularStructure:
        """Load a structure from file."""
        current_frame = xyz2list(structure_path)
        return MolecularStructure.from_xyz_list(current_frame)

    def generate_variants(self) -> List[MolecularStructure]:
        """Generate new structures by transforming CH groups to CH3.

        Args:
            structure: Input molecular structure

        Returns:
            List of generated structures
        """
        try:
            # Convert to xyz list format
            xyz_list = self.input_structure.to_xyz_list()

            C_CH3_list = [('C', -3.008535, 1.29474, -0.181592),
                          ('C', -3.600732, 2.680895, -0.215513),
                          ('H', -3.362085, 3.130344, -1.182454),
                          ('H', -4.681806, 2.601987, -0.126612),
                          ('H', -3.205821, 3.320289, 0.574525)]

            # Get atom coordinates and types
            atom_coordinates = self.input_structure.coordinates
            atom_types = [atom[0] for atom in xyz_list]

            try:
                c_idx = self.params["c_idx"]
                if atom_types[c_idx] != 'C':
                    logger.warning(f"Atom at index {c_idx} is not a carbon atom")
                    return []
            except Exception as e:
                logger.error(f"Failed to retrieve the C index: {str(e)}")
                return []

            # Find the H atom attached to C
            h_idx = None
            typical_bond_length = 1.2  # Angstroms, typical for C-H bonds
            
            for i, coords in enumerate(atom_coordinates):
                if atom_types[i] != 'H':
                    continue
                
                distance = np.linalg.norm(atom_coordinates[c_idx] - coords)
                if distance < typical_bond_length:
                    h_idx = i
                    break

            if h_idx is None:
                logger.warning(f"No hydrogen atom found attached to carbon at index {c_idx}")
                return []

            base_align_list = [[c_idx, h_idx]]
            attached = attach(xyz_list, seed_xyz_list=C_CH3_list, 
                              base_align_list=base_align_list,
                              base_del_list=base_align_list,
                              **self.params["attach_kwargs"])
            
            # Create new MolecularStructure
            variant = MolecularStructure.from_xyz_list(attached)
            variant.metadata = self.input_structure.metadata.copy() if self.input_structure.metadata else {}
            variant.metadata.update({"operation": "ch_to_methyl"})
            
            # Check if physically reasonable
            if self.check_physical:
                check_result = is_physical_geometry(new_xyz_list, **self.check_physical_kwargs)
                if check_result != "normal":
                    logger.debug(f"Generated structure is not physically reasonable")
                    return []
            
            return [variant]
            
        except Exception as e:
            logger.error(f"Error in ch_to_methyl operation: {str(e)}", exc_info=True)
            return []

    def generate_grid(self):
        """
        Generate a structure with specific bond vector indices.
        
        Args:
            structure: Input molecular structure
            
        Returns:
            Transformed structure or None if generation failed
        """
        try:
            # Save the original bond vector indices
            
            # Generate variants
            variants = self.generate_variants()
            
            return variants[0] if variants else None
            
        except Exception as e:
            logger.error(f"Error generating structure with specific indices: {str(e)}", exc_info=True)
            return None

    def set_input_structure(self, input_structure: List[str]):
        """Set the base structures."""
        self.input_structure = input_structure

    def set_params(self, **kwargs):
        """Update parameters for the transformation operation."""
        self.params.update(kwargs)
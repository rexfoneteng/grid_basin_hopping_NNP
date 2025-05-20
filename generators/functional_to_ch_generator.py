#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-05-19 14:08:42
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-20 10:09:17
# @Description: transforms any functional group into CH

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union

from generators.base_generator import BaseGenerator
from core.molecular_structure import MolecularStructure

from xyz_physical_geometry_tool_mod import is_physical_geometry
from xyz_tools import xyz2list

logger = logging.getLogger(__name__)

class FunctionalToChGenerator(BaseGenerator):
    """Generator for transforming -CH2OH functional groups to -CH.
    
    This generator takes a pair of atom indices defining the vector of the resulting group
    (C atom of CH2 and its attachment point), and automatically determines which 
    atoms to remove (OH and one H from CH2).
    """

    def __init__(self,
                 input_structure: Union[str, List, MolecularStructure, None] = None,
                 params: Optional[Dict[str, Any]] = \
                    {"c_c": [9, 11]},
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the FunctionalToChGenerator.

        Args:
            base_structures: List of paths to base structure files
            params: Parameters for -CH2OH to -CH transformation
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
        """
        super().__init__(check_physical, check_physical_kwargs)

        self.input_structure = input_structure
        self.params = params
        # Load the base structure
        if isinstance(self.input_structure, str):
            self.input_structure = self._load_structure(self.input_structure)

    def __call__(self) -> MolecularStructure:
        """Generate a structure by transforming a -CH2OH group to -CH.

        Returns:
            A new MolecularStructure with a -CH2OH group transformed to -CH
        """
        if not self.input_structure:
            raise ValueError("No base structures available for transformation")
        
        
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
        """Generate new structures by transforming -CH2OH groups to -CH.

        Args:
            structure: Input molecular structure

        Returns:
            List of generated structures
        """
        try:
            # Convert to xyz list format
            xyz_list = self.input_structure.to_xyz_list()
            
            # Get atom coordinates and types
            atom_coordinates = self.input_structure.coordinates
            atom_types = [atom[0] for atom in xyz_list]

            try:
                c1_idx, c2_idx = self.params["c_c"]
                if atom_types[c1_idx] != 'C' or atom_types[c2_idx] != 'C':
                    logger.warning(f"Atom at index {c_idx} is not a carbon atom")
                    return []
            except Exception as e:
                logger.error(f"Failed to retrieve the C index: {str(e)}")
                return []
        
            # Find the connected atoms based on distance thresholds
            c1_coords = atom_coordinates[c1_idx]
            c2_coords = atom_coordinates[c2_idx]
            
            # Typical bond lengths (in Angstroms)
            c_o_bond_length = 1.43  # Typical C-O bond length
            c_h_bond_length = 1.1   # Typical C-H bond length
            o_h_bond_length = 0.96  # Typical O-H bond length
            
            # Lists to store indices of different atom types bonded to the carbon
            h_indices = []
            o_idx = None
            oh_h_idx = None
            
            # Find all atoms bonded to the carbon
            for i, coords in enumerate(atom_coordinates):
                if i in self.params["c_c"]:
                    continue
                
                distance = np.linalg.norm(c2_coords - coords)
                
                # Find hydrogen atoms directly bonded to carbon
                if atom_types[i] == 'H' and distance < c_h_bond_length + 0.1:
                    h_indices.append(i)
                
                # Find oxygen atom bonded to carbon
                if atom_types[i] == 'O' and distance < c_o_bond_length + 0.1:
                    o_idx = i
            
            # Find hydrogen bonded to oxygen
            if o_idx is not None:
                o_coords = atom_coordinates[o_idx]
                for i, coords in enumerate(atom_coordinates):
                    if atom_types[i] == 'H':
                        distance = np.linalg.norm(o_coords - coords)
                        if distance < o_h_bond_length + 0.1:
                            oh_h_idx = i
                            break
            
            # Verify we have a -CH2OH group
            if o_idx is None or oh_h_idx is None or len(h_indices) < 2:
                logger.warning("Could not identify a -CH2OH group")
                return []
            
            # Atoms to remove: both H atoms, O atom, and OH's H atom
            atoms_to_remove = set(h_indices + [o_idx, oh_h_idx] + [c2_idx])
            
            # Calculate vector for new H atom position
            # Use the attachment point to determine the orientation
            c1c2_vec = c2_coords - c1_coords
            c1c2_unit_vec = c1c2_vec / np.linalg.norm(c1c2_vec)
            
            # Position new H opposite to attachment direction
            new_h_coords = c1c2_unit_vec * 0.97 + c1_coords
            
            # Create new structure
            new_xyz_list = [xyz for i, xyz in enumerate(xyz_list) if i not in atoms_to_remove]
            new_xyz_list.append(('H', *new_h_coords))
            
            # Create new MolecularStructure
            variant = MolecularStructure.from_xyz_list(new_xyz_list)
            variant.metadata = structure.metadata.copy() if structure.metadata else {}
            variant.metadata.update({"operation": "ch2oh_to_ch"})
            
            # Check if physically reasonable
            if self.check_physical:
                check_result = is_physical_geometry(new_xyz_list, **self.check_physical_kwargs)
                if check_result != "normal":
                    logger.debug(f"Generated structure is not physically reasonable")
                    return []
            
            return [variant]
            
        except Exception as e:
            logger.error(f"Error in CH2OH to CH transformation: {str(e)}", exc_info=True)
            return []

    def generate_grid(self):
        """
        Generate a structure with specific bond vector indices.
        
        Args:
            structure: Input molecular structure
            bond_vector_indices: Tuple of (c_idx, attachment_idx)
            
        Returns:
            Transformed structure or None if generation failed
        """
        try:
            
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
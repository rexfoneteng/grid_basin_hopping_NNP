"""Generator for attaching and rotating structures."""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

from generators.base_generator import BaseGenerator
from core.molecular_structure import MolecularStructure
from core.constants import DEFAULT_ATTACH_ROTATE_PARAMS

# Import needed functions from your original tools
from xyz_tools import Xyz, attach, turn
from xyz_physical_geometry_tool_mod import is_physical_geometry

logger = logging.getLogger(__name__)

class AttachRotateGenerator(BaseGenerator):
    """Generator for attaching and rotating structures."""

    def __init__(self,
                 base_structures: Optional[List[str]] = None,
                 seed_structures: Optional[List[str]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the AttachRotateGenerator.

        Args:
            base_structures: List of paths to base structure files
            seed_structures: List of paths to seed structure files (for attachment)
            params: Parameters for attach-rotate operation
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
        """
        super().__init__(check_physical, check_physical_kwargs)

        self.base_structures = base_structures or []
        self.seed_structures = seed_structures or []
        self.params = params or DEFAULT_ATTACH_ROTATE_PARAMS.copy()
        self.current_base_structure = None
        self.current_seed_structure = None

    def __call__(self) -> MolecularStructure:
        """Generate a structure by attaching and rotating parts.

        Returns:
            A new MolecularStructure with attached and rotated parts
        """
        pass

    def set_current_base_structure(self, structure_path: str):
        """Set the current base structure."""
        self.current_base_structure = structure_path

    def set_current_seed_structure(self, structure_path: str):
        """Set the current seed structure."""
        self.current_seed_structure = structure_path

    def set_base_structures(self, base_structures: List[str]):
        """Set the base structures."""
        self.base_structures = base_structures

    def set_seed_structures(self, seed_structures: List[str]):
        """Set the seed structures."""
        self.seed_structures = seed_structures

    def set_params(self, **kwargs):
        """Update parameters for the attach-rotate operation."""
        self.params.update(kwargs)

    def generate_variants(self, structure: MolecularStructure) -> MolecularStructure:
        """Generate a new structure by flipping a functional group in an existing structure.

        Args:
            structure: Existing molecular structure

        Returns:
            A new MolecularStructure with flipped functional group
        """
        try:
            base_frame = structure.to_xyz_list()
            base_frame_id = 0
            seed_xyz_obj = Xyz(self.current_seed_structure)

            seed_frame_id = 0
            seed_frame = seed_xyz_obj.next()

            # Attach the structures
            attached = attach(base_frame, seed_xyz_list=seed_frame, **self.params["attach_kwargs"])

            # Get atom counts for rotation
            n_atom = len(base_frame)
            n_atom_1 = len(seed_frame)

            # Set up rotation parameters
            rot_bond_0 = self.params["rotate_bond_list"][0]
            rot_bond_1 = None if len(self.params["rotate_bond_list"]) < 2 else self.params["rotate_bond_list"][1]

            rot_atoms_0 = set(list(rot_bond_0) + list(range(n_atom-1, n_atom+n_atom_1-3)))

            variant_structures = []
            # Rotate around first bond
            for rot_angle_0 in self.params["angle_list"]:
                rotated = turn(attached, rotate_bond=rot_bond_0, rotate_atom_list=rot_atoms_0, angle=rot_angle_0)

                # Rotate around second bond if applicable
                if rot_bond_1:
                    rot_atoms_1 = set(list(rot_bond_1) + list(range(n_atom-1, n_atom+n_atom_1-3)))
                    for rot_angle_1 in self.params["angle_list"]:
                        rotated = turn(rotated, rotate_bond=rot_bond_1, rotate_atom_list=rot_atoms_1, angle=rot_angle_1)

                        # Convert to MolecularStructure
                        mol_structure = MolecularStructure.from_xyz_list(rotated)
                        mol_structure.metadata = {
                            "operation": "attach_rotate",
                            "base_structure": self.current_base_structure,
                            "seed_structure": self.current_seed_structure,
                            "base_frame_id": base_frame_id,
                            "seed_frame_id": seed_frame_id,
                            "rot_angle_0": rot_angle_0,
                            "rot_angle_1": rot_angle_1
                        }

                        if self.check_physical:
                            check_result = is_physical_geometry(rotated, **self.check_physical_kwargs)
                            if check_result == "normal":
                                variant_structures.extend(mol_structure)
                        else:
                            variant_structures.extend(mol_structure)
                else:
                    rot_angle_1 = -1

                    # Convert to MolecularStructure
                    mol_structure = MolecularStructure.from_xyz_list(rotated)
                    mol_structure.metadata = {
                            "operation": "attach_rotate",
                            "base_structure": self.current_base_structure,
                            "seed_structure": self.current_seed_structure,
                            "base_frame_id": base_frame_id,
                            "seed_frame_id": seed_frame_id,
                            "rot_angle_0": rot_angle_0,
                            "rot_angle_1": rot_angle_1
                        }

                    if self.check_physical:
                        check_result = is_physical_geometry(rotated, **self.check_physical_kwargs)
                        if check_result == "normal":
                            variant_structures.extend(mol_structure)
                    else:
                            variant_structures.extend(mol_structure)

            return mol_structure

        except (IndexError, KeyError) as e:
            logger.warning(f"Error in flip operation: {str(e)}")
            return None

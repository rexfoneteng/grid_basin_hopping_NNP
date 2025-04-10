import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union

from generators.base_generator import BaseGenerator
from core.molecular_structure import MolecularStructure
from core.constants import DEFAULT_FLIP_PARAMS
from utils.flatten import flatten_concatenation

# Import needed functions from your original tools
from xyz_tools import Xyz, turn, molecule_stat
from sugar_tools import sugar_stat
from xyz_physical_geometry_tool_mod import is_physical_geometry
from xyz_toolsx import flip

logger = logging.getLogger(__name__)

class FlipGenerator(BaseGenerator):
    """Generator for flipping functional groups in structures with grid-based sampling."""

    def __init__(self,
                 base_structures: Optional[List[str]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the FlipGenerator.

        Args:
            base_structures: List of paths to base structure files
            params: Parameters for flip operation
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
        """
        super().__init__(check_physical, check_physical_kwargs)

        self.base_structures = base_structures or []
        self.params = params or DEFAULT_FLIP_PARAMS.copy()
        self.current_base_structure = None
        
        # Define the grid for rotation angles (in degrees)
        self.angle_grid = [0, 120, 240]  # Default grid points
        if "rotating_angle" in self.params and isinstance(self.params["rotating_angle"], (list, tuple)):
            self.angle_grid = list(self.params["rotating_angle"])
        
        # Initialize grid state for basin hopping
        self.current_grid_point = None

    def __call__(self) -> MolecularStructure:
        """Generate a structure by flipping a functional group.

        Returns:
            A new MolecularStructure with flipped functional group
        """
        if not self.base_structures:
            raise ValueError("No base structures available for flipping")
            
        # Select a random base structure if none is currently selected
        if not self.current_base_structure:
            self.current_base_structure = np.random.choice(self.base_structures)
            
        # Load the base structure
        base_structure = self._load_structure(self.current_base_structure)
        
        # Generate variants and return a random one
        variants = self.generate_variants(base_structure)
        if variants:
            return np.random.choice(variants)
        else:
            logger.warning(f"No valid variants generated for {self.current_base_structure}")
            # Try a different base structure if this one failed
            previous = self.current_base_structure
            self.current_base_structure = np.random.choice([s for s in self.base_structures 
                                                          if s != previous])
            return self()

    def _load_structure(self, structure_path: str) -> MolecularStructure:
        """Load a structure from file.
        
        Args:
            structure_path: Path to the structure file
            
        Returns:
            Loaded molecular structure
        """
        # Implementation depends on your file loading utilities
        # This is a placeholder
        xyz_obj = Xyz(structure_path)
        frame_id = np.random.randint(0, xyz_obj.frame_num)
        current_frame = xyz_obj.get_frame(frame_id)
        
        return MolecularStructure.from_xyz_list(current_frame)

    def set_current_base_structure(self, structure_path: str):
        """Set the current base structure."""
        self.current_base_structure = structure_path

    def set_base_structures(self, base_structures: List[str]):
        """Set the base structures."""
        self.base_structures = base_structures

    def set_params(self, **kwargs):
        """Update parameters for the flip operation."""
        self.params.update(kwargs)
        # Update angle grid if rotating_angle was changed
        if "rotating_angle" in kwargs and isinstance(kwargs["rotating_angle"], (list, tuple)):
            self.angle_grid = list(kwargs["rotating_angle"])

    def set_grid_point(self, base_structure_idx: int, angle_idx: int):
        """Set the specific grid point for basin hopping.
        
        Args:
            base_structure_idx: Index of base structure in self.base_structures
            angle_idx: Index of rotation angle in self.angle_grid
        """
        if 0 <= base_structure_idx < len(self.base_structures) and 0 <= angle_idx < len(self.angle_grid):
            self.current_base_structure = self.base_structures[base_structure_idx]
            self.current_grid_point = (base_structure_idx, angle_idx)
        else:
            raise ValueError(f"Invalid grid point: ({base_structure_idx}, {angle_idx})")

    def get_random_grid_point(self) -> Tuple[int, int]:
        """Get a random grid point.
        
        Returns:
            Tuple of (base_structure_idx, angle_idx)
        """
        base_idx = np.random.randint(0, len(self.base_structures))
        angle_idx = np.random.randint(0, len(self.angle_grid))
        return (base_idx, angle_idx)

    def _find_proton(self, sugar, O_NAc_id) -> Set[int]:
        """Get the ID of H+."""

        sugar_list = [s for s in sugar]  # Convert to list format if needed
        mol_stat = molecule_stat(sugar_list)
        ONAc_bond = [ele for ele in mol_stat["mol_bond_pattern"][0] if ele[1] == O_NAc_id]
        H_id = [idx for idx, ele in enumerate(sugar_list) if ele[0] == "H"]
        proton_id = set(flatten_concatenation(ONAc_bond)).intersection(H_id)
        return proton_id

    def generate_grid(self, structure, flip_angle):
        """
        Generate a structure with a specific flip angle.
        
        Args:
            structure: Input molecular structure
            flip_angle: Specific rotation angle to use (degrees)
            
        Returns:
            Flipped structure or None if generation failed
        """
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
            position_str = self.params["position"]
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
                from utils.flatten import flatten_concatenation
                
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
            flipped_xyz = flip(current_frame, CH_bond, C_flip_atom, CH_bond, flip_group)
            
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

    def generate_variants(self, structure: MolecularStructure) -> List[MolecularStructure]:
        """Generate structure variants by flipping a functional group.

        Args:
            structure: Input molecular structure

        Returns:
            List of generated structures
        """
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
            position_str = self.params["position"]
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
                proton_id = self._find_proton(current_frame, flip_group[4])
                flip_group.extend(proton_id)
                C_flip_atom = (C_id, flip_group[1])
            else:
                flip_group = (C_id, O_id, HO_id)

            variant_structures = []
            # Perform the flip
            flipped_xyz = flip(current_frame, CH_bond, C_flip_atom, CH_bond, flip_group)

            # If rotate_after_flip is not specified, add the flipped structure without rotation
            if not self.params.get("rotate_after_flip", True):
                mol_structure = MolecularStructure.from_xyz_list(flipped_xyz)
                mol_structure.metadata = {
                    "operation": "flip",
                    "base_structure": self.current_base_structure,
                    "ring_id": ring_id,
                    "position": position,
                    "angle_val": 0.0
                }
                
                # Check if the structure is physically reasonable
                if self.check_physical:
                    check_result = is_physical_geometry(flipped_xyz, **self.check_physical_kwargs)
                    if check_result == "normal":
                        variant_structures.append(mol_structure)
                else:
                    variant_structures.append(mol_structure)

            # Rotate using the grid angles
            if self.params.get("rotate_after_flip", True):
                for angle_val in self.angle_grid:
                    flipped_xyz_0 = turn(flipped_xyz,
                                        rotate_atom_list=flip_group,
                                        rotate_bond=C_flip_atom,
                                        angle=angle_val)
                    
                    # Convert xyz_list to MolecularStructure obj
                    mol_structure = MolecularStructure.from_xyz_list(flipped_xyz_0)
                    mol_structure.metadata = {
                        "operation": "flip",
                        "base_structure": self.current_base_structure,
                        "ring_id": ring_id,
                        "position": position,
                        "angle_val": angle_val,
                        "grid_point": (self.base_structures.index(self.current_base_structure), 
                                     self.angle_grid.index(angle_val))
                    }

                    # Check if the structure is physically reasonable
                    if self.check_physical:
                        check_result = is_physical_geometry(flipped_xyz_0, **self.check_physical_kwargs)
                        if check_result == "normal":
                            variant_structures.append(mol_structure)
                    else:
                        variant_structures.append(mol_structure)

            return variant_structures

        except (IndexError, KeyError, Exception) as e:
            logger.error(f"Error in flip operation: {str(e)}", exc_info=True)
            return []
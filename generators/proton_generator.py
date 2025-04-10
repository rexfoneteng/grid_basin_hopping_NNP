import math
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from functools import lru_cache

from generators.base_generator import BaseGenerator
from core.molecular_structure import MolecularStructure
from core.constants import DEFAULT_ADD_PROTON_PARAMS
from utils.flatten import flatten_concatenation

from xyz_physical_geometry_tool_mod import is_physical_geometry
from rotational_matrix import rotation_matrix_numpy
from geometry_1 import unit_vector
from geometry import angle
from sugar_tools import Sugar, sugar_stat

logger = logging.getLogger(__name__)

class ProtonGenerator(BaseGenerator):
    """Generator for adding protons to structures using either user-defined or auto-detected structure info."""

    def __init__(self,
                 base_structures: Optional[List[str]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 sugar_info: Optional[Dict[str, List]] = None,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None,
                 tmp_dir: Optional[str] = None):
        """Initialize the ProtonGenerator.

        Args:
            base_structures: List of paths to base structure files
            params: Parameters for proton addition
            sugar_info: User-defined sugar structure information dict
                        (format as shown in example)
            check_physical: Whether to check if generated structures are physically reasonable
            check_physical_kwargs: Additional kwargs for physical geometry check
            tmp_dir: Directory for temporary files
        """
        super().__init__(check_physical, check_physical_kwargs, tmp_dir)

        self.base_structures = base_structures or []
        self.params = params or DEFAULT_ADD_PROTON_PARAMS.copy()
        self.current_base_structure = None
        self.sugar_info = sugar_info

        # Initialize cache for structure information
        self._structure_cache = {}

        # Convert angle lists from degrees to radians if needed
        if "angle_list" in self.params and isinstance(self.params["angle_list"][0], (int, float)):
            self.params["angle_list"] = [math.radians(ele) for ele in self.params["angle_list"]]
        if "angle_around_Oring" in self.params and isinstance(self.params["angle_around_Oring"][0], (int, float)):
            self.params["angle_around_Oring"] = [math.radians(ele) for ele in self.params["angle_around_Oring"]]

    def __call__(self) -> MolecularStructure:
        """Generate a structure by adding a proton.

        Returns:
            A new MolecularStructure with added proton
        """
        # Load a base structure if none is currently selected
        if not self.current_base_structure:
            if not self.base_structures:
                raise ValueError("No base structures available for protonation")
            self.current_base_structure = np.random.choice(self.base_structures)

        # Load the structure
        structure = self._load_structure(self.current_base_structure)

        # Generate variants for this structure
        variants = self.generate_variants(structure)

        # Return a random variant if any were generated
        if variants:
            return np.random.choice(variants)
        else:
            logger.warning(f"No valid variants generated for {self.current_base_structure}")
            # Try with a different structure if this one fails
            if len(self.base_structures) > 1:
                self.current_base_structure = np.random.choice([s for s in self.base_structures
                                                              if s != self.current_base_structure])
            return self()

    def _load_structure(self, structure_path: str) -> MolecularStructure:
        """Load a structure from file.

        Args:
            structure_path: Path to the structure file

        Returns:
            Loaded molecular structure
        """
        # This is a placeholder - in a real implementation, you'd use your existing
        # code to load a structure from file and potentially select a specific frame

        sug_obj = Sugar(structure_path)
        frame_id = np.random.randint(0, sug_obj.frame_num)
        current_frame = sug_obj.get_frame(frame_id)

        return MolecularStructure.from_xyz_list(current_frame)

    def _get_structure_info(self, structure: MolecularStructure) -> Dict[str, Any]:
        """Get or compute structure information (cached).

        Args:
            structure: The molecular structure to analyze

        Returns:
            Dictionary with structure information
        """
        # Use fingerprint as cache key
        fingerprint = structure.get_fingerprint()

        # Return cached data if available
        if fingerprint in self._structure_cache:
            logger.debug("Using cached structure information")
            return self._structure_cache[fingerprint]

        # Get the XYZ list and coordinates
        current_frame = structure.to_xyz_list()
        coor = np.array([ele[1:4] for ele in current_frame])
        atom_types = [atom[0] for atom in current_frame]

        # Prepare basic information
        info = {
            'frame': current_frame,
            'coor': coor,
            'atom_types': atom_types,
        }

        # Use user-provided sugar information if available, otherwise auto-detect
        if self.sugar_info:
            # Copy user-provided sugar information
            info.update(self.sugar_info)

            # Find hydrogens attached to oxygens
            if 'H_on_O' not in info:
                info['H_on_O'] = self._find_h_on_o(info, coor, atom_types)

            # Find glycosidic bonds if not provided and we have multiple rings
            if 'O_glycosidic' not in info and len(info.get('ring', [])) > 1:
                info['O_glycosidic'] = self._find_glycosidic_o(info, coor, atom_types)
        else:
            # Automatically determine sugar information
            #auto_info = self._determine_sugar_info(current_frame, coor, atom_types)
            auto_info = sugar_stat(current_frame)
            info.update(auto_info)

        # Cache the result
        self._structure_cache[fingerprint] = info

        # Limit cache size (keep only last 100 structures)
        if len(self._structure_cache) > 100:
            oldest_key = next(iter(self._structure_cache))
            self._structure_cache.pop(oldest_key)

        return info

    def _find_h_on_o(self, info: Dict[str, Any], coor: np.ndarray, atom_types: List[str]) -> List[List[int]]:
        """Find hydrogen atoms attached to oxygens in each ring.

        Args:
            info: Structure information dictionary
            coor: Atom coordinates array
            atom_types: List of atom types

        Returns:
            List of lists containing H indices for each ring's oxygens
        """
        h_on_o = []

        # Typical O-H bond length: ~0.95 Å
        o_h_bond_threshold = 1.2  # Angstroms, slightly larger than O-H bond length

        for ring_idx, o_atoms in enumerate(info.get('O_on_C_chain', [])):
            h_indices = []
            for o_idx in o_atoms:
                if o_idx < 0:  # Skip non-existing oxygens
                    h_indices.append(-1)
                    continue

                # Find the closest hydrogen to this oxygen
                closest_h_idx = -1
                min_distance = float('inf')

                for i, atom_type in enumerate(atom_types):
                    if atom_type != 'H':
                        continue

                    dist = np.linalg.norm(coor[o_idx] - coor[i])
                    if dist < o_h_bond_threshold and dist < min_distance:
                        min_distance = dist
                        closest_h_idx = i

                h_indices.append(closest_h_idx)

            h_on_o.append(h_indices)

        return h_on_o

    def _find_glycosidic_o(self, info: Dict[str, Any], coor: np.ndarray, atom_types: List[str]) -> List[int]:
        """Find glycosidic oxygen atoms between rings.

        Args:
            info: Structure information dictionary
            coor: Atom coordinates array
            atom_types: List of atom types

        Returns:
            List of glycosidic oxygen indices
        """
        # A glycosidic oxygen is an oxygen that bridges two rings
        # We can look at the O_on_C_chain entries to find common oxygens

        o_chains = info.get('O_on_C_chain', [])
        if len(o_chains) < 2:
            return []

        # Find oxygen indices that appear in different rings
        all_o_indices = []
        for chain in o_chains:
            all_o_indices.extend([o for o in chain if o >= 0])

        # Count occurrences of each oxygen
        o_counts = {}
        for o_idx in all_o_indices:
            o_counts[o_idx] = o_counts.get(o_idx, 0) + 1

        # Glycosidic oxygens appear in multiple rings
        glycosidic_o = [o_idx for o_idx, count in o_counts.items() if count > 1]

        return glycosidic_o

    def _determine_sugar_info(self, frame: List, coor: np.ndarray, atom_types: List[str]) -> Dict[str, Any]:
        """Determine sugar structure information automatically.

        Args:
            frame: XYZ frame list
            coor: Coordinates array
            atom_types: List of atom types

        Returns:
            Dictionary containing determined sugar information
        """
        # TODO: Implement a proper automatic sugar structure detection
        # This is a placeholder that would need to be replaced with a real implementation
        # For now, I'll just return a minimal structure

        # Find atom indices by type
        atom_indices = {
            'C': [i for i, atom_type in enumerate(atom_types) if atom_type == 'C'],
            'O': [i for i, atom_type in enumerate(atom_types) if atom_type == 'O'],
            'H': [i for i, atom_type in enumerate(atom_types) if atom_type == 'H'],
            'N': [i for i, atom_type in enumerate(atom_types) if atom_type == 'N'],
        }

        # Basic info for a simple ring structure
        # In a real implementation, this would be much more sophisticated
        info = {
            'ring': [[i for i in atom_indices['C'][:5] + [atom_indices['O'][0]]]],
            'C_ring': [[i for i in atom_indices['C'][:5]]],
            'O_ring': [[atom_indices['O'][0]]],
            'C_chain': [[i for i in atom_indices['C'][:6]]],
            'O_on_C_chain': [[i for i in atom_indices['O'][1:7]]],
            'O_num_on_C_chain': [[1, 1, 1, 1, 1, 1]],
        }

        # Find hydrogens attached to oxygens
        info['H_on_O'] = self._find_h_on_o(info, coor, atom_types)

        return info

    def generate_variants(self, structure: MolecularStructure, oh_ids_of_interest: List[int] = None) -> List[MolecularStructure]:
        """Generate multiple variant structures by adding protons at different positions.

        Args:
            structure: Input molecular structure
            oh_ids_of_interest: List of specific OH group IDs to protonate (if None, all OH groups are considered)

        Returns:
            List of generated variant structures with protons added
        """
        try:
            # Get structure information
            info = self._get_structure_info(structure)
            current_frame = info['frame']
            coor = info['coor']

            # List to collect all generated variants
            variant_structures = []

            # Add protons at different sites using helper functions
            variants = []

            # 1. Add protons to ring oxygens
            variants.extend(self._add_protons_to_ring_oxygens(info, coor, current_frame))

            # 2. Add protons to hydroxyl groups
            variants.extend(self._add_protons_to_hydroxyl_groups(info, coor, current_frame, oh_ids_of_interest))

            # 3. Add protons to NAc if present
            if 'C_NH_CO_CHHH' in info and info['C_NH_CO_CHHH']:
                variants.extend(self._add_protons_to_nac_groups(info, coor, current_frame))

            # 4. Add protons to glycosidic oxygen if there are multiple rings
            if len(info.get('ring', [])) > 1 and 'O_glycosidic' in info:
                variants.extend(self._add_protons_to_glycosidic_oxygen(info, coor, current_frame))

            # Filter out None values
            variant_structures = [v for v in variants if v is not None]

            logger.info(f"Generated {len(variant_structures)} protonated variants")
            return variant_structures

        except Exception as e:
            logger.error(f"Error in generate_variants: {str(e)}", exc_info=True)
            return []  # Return empty list on error

    def _add_protons_to_ring_oxygens(self, info: Dict[str, Any], coor: np.ndarray, current_frame: List) -> List[MolecularStructure]:
        """Add protons to ring oxygen atoms.

        Args:
            info: Structure information dictionary
            coor: Atom coordinates array
            current_frame: Current XYZ frame

        Returns:
            List of structures with protons added to ring oxygens
        """
        variants = []

        # Process each ring
        for ring_idx, o_ring_indices in enumerate(info.get('O_ring', [])):
            if not o_ring_indices:
                continue

            try:
                # Get ring oxygen (assume the first one is the main ring oxygen)
                o_idx = o_ring_indices[0]

                # Get connected carbon atoms
                c_ring = info.get('C_ring', [])[ring_idx]
                if len(c_ring) < 2:
                    continue

                # Get the two carbon atoms connected to the ring oxygen
                # In pyranoses, typically C1 and C5 are connected to the ring oxygen
                c1_idx = c_ring[0]  # Typically C1
                c5_idx = c_ring[-1]  # Typically C5

                # Add protons at different angles around the ring oxygen
                for angle_val in self.params["angle_around_Oring"]:
                    proton_pos = self._calculate_proton_position(o_idx, [c1_idx, c5_idx], coor, angle_val,
                                                              self.params["proton_Oring_dist"])

                    # Create the protonated structure
                    variant = self._create_protonated_structure(
                        current_frame, proton_pos,
                        {"operation": "add_proton", "site": "ring_O",
                         "ring_id": ring_idx, "angle_val": angle_val, "atom_id": o_idx}
                    )
                    if variant:
                        variants.append(variant)

            except (IndexError, KeyError) as e:
                logger.debug(f"Ring {ring_idx}: Failed to process ring oxygen: {str(e)}")

        return variants

    def _add_protons_to_hydroxyl_groups(self, info: Dict[str, Any], coor: np.ndarray, current_frame: List,
                                      oh_ids_of_interest: List[int] = None) -> List[MolecularStructure]:
        """Add protons to hydroxyl groups.

        Args:
            info: Structure information dictionary
            coor: Atom coordinates array
            current_frame: Current XYZ frame
            oh_ids_of_interest: List of specific OH group IDs to protonate

        Returns:
            List of structures with protons added to hydroxyl groups
        """
        variants = []

        # Process each ring's hydroxyl groups
        for ring_idx, (c_chain, o_chain, h_chain) in enumerate(
            zip(info.get('C_chain', []), info.get('O_on_C_chain', []), info.get('H_on_O', []))):

            # Process each position in the chain
            for pos_idx, (c_idx, o_idx, h_idx) in enumerate(zip(c_chain, o_chain, h_chain)):
                try:
                    # Skip if no oxygen or hydrogen atom
                    if o_idx < 0 or h_idx < 0:
                        continue

                    # Skip if oh_ids_of_interest is specified and this O_id is not in the list
                    if oh_ids_of_interest and o_idx not in oh_ids_of_interest:
                        continue

                    # Calculate vectors for the C-O and O-H bonds
                    CO_vec = coor[o_idx] - coor[c_idx]
                    OH_vec = coor[h_idx] - coor[o_idx]

                    # Add protons at different angles around the hydroxyl
                    for angle_val in self.params["angle_list"]:
                        # Rotate the O-H vector around the C-O axis to get new proton position
                        rotated_OH_vec = np.dot(rotation_matrix_numpy(CO_vec, angle_val), OH_vec)
                        proton_pos = coor[o_idx] + rotated_OH_vec

                        # Create the protonated structure
                        variant = self._create_protonated_structure(
                            current_frame, proton_pos,
                            {"operation": "add_proton", "site": "OH",
                             "ring_id": ring_idx, "position": pos_idx,
                             "angle_val": angle_val, "atom_id": o_idx}
                        )
                        if variant:
                            variants.append(variant)

                except Exception as e:
                    logger.debug(f"Failed to add proton to hydroxyl group at ring {ring_idx}, position {pos_idx}: {str(e)}")

        return variants

    def _add_protons_to_nac_groups(self, info: Dict[str, Any], coor: np.ndarray, current_frame: List) -> List[MolecularStructure]:
        """Add protons to N-Acetyl groups.

        Args:
            info: Structure information dictionary
            coor: Atom coordinates array
            current_frame: Current XYZ frame

        Returns:
            List of structures with protons added to N-Acetyl groups
        """
        variants = []

        # Process each N-Acetyl group
        for nac_idx, nac_group in enumerate(info.get('C_NH_CO_CHHH', [])):
            try:
                if len(nac_group) < 9:
                    continue

                # Extract atom indices from the NAc group
                C_id, N_id, HN_id, CO_id, OC_id, C_methyl_id, H_methyl_1, H_methyl_2, H_methyl_3 = nac_group

                # Add protons to NH part and carbonyl oxygen
                variants.extend(self._add_protons_to_nac_nh(nac_idx, C_id, N_id, HN_id, CO_id, coor, current_frame))
                variants.extend(self._add_protons_to_nac_carbonyl(nac_idx, C_id, N_id, CO_id, OC_id, coor, current_frame))

            except Exception as e:
                logger.debug(f"Failed to add proton to N-acetyl group {nac_idx}: {str(e)}")

        return variants

    def _add_protons_to_nac_nh(self, nac_idx: int, C_id: int, N_id: int, HN_id: int, CO_id: int,
                             coor: np.ndarray, current_frame: List) -> List[MolecularStructure]:
        """Add protons to NH group in N-Acetyl.

        Args:
            nac_idx: Index of the N-Acetyl group
            C_id, N_id, HN_id, CO_id: Atom indices
            coor: Atom coordinates array
            current_frame: Current XYZ frame

        Returns:
            List of structures with protons added to NH groups
        """
        variants = []

        try:
            # Calculate vectors
            C2CO_vec = coor[CO_id] - coor[C_id]
            NH_vec = coor[HN_id] - coor[N_id]

            angle_val_NAc_list = [math.radians(ele) for ele in (-67.5, 67.5)]
            for angle_val in angle_val_NAc_list:
                rotated_NH_vec = np.dot(rotation_matrix_numpy(C2CO_vec, angle_val), NH_vec)
                proton_pos = coor[N_id] + rotated_NH_vec

                # Create the protonated structure
                variant = self._create_protonated_structure(
                    current_frame, proton_pos,
                    {"operation": "add_proton", "site": "NAc_NH",
                     "nac_idx": nac_idx, "angle_val": angle_val, "atom_id": N_id}
                )
                if variant:
                    variants.append(variant)

        except Exception as e:
            logger.debug(f"Failed to add proton to NAc NH group {nac_idx}: {str(e)}")

        return variants

    def _add_protons_to_nac_carbonyl(self, nac_idx: int, C_id: int, N_id: int, CO_id: int, OC_id: int,
                                   coor: np.ndarray, current_frame: List) -> List[MolecularStructure]:
        """Add protons to carbonyl oxygen in N-Acetyl.

        Args:
            nac_idx: Index of the N-Acetyl group
            C_id, N_id, CO_id, OC_id: Atom indices
            coor: Atom coordinates array
            current_frame: Current XYZ frame

        Returns:
            List of structures with protons added to carbonyl oxygens
        """
        variants = []

        try:
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
            theta = 0.37 * math.pi

            # Define 5 evenly distributed phi angles around z-axis
            phi_values = [i * 2 * math.pi / 5 for i in range(5)]

            # Define r values
            r = self.params["proton_Oring_dist"]

            # Add proton at each phi value
            for phi_val in phi_values:
                # Convert spherical coordinates to Cartesian coordinates
                x_local = r * math.sin(theta) * math.cos(phi_val)
                y_local = r * math.sin(theta) * math.sin(phi_val)
                z_local = r * math.cos(theta)

                # Transform to global coordinates
                proton_vector = x_local * x_axis + y_local * y_axis + z_local * z_axis
                proton_pos = coor[OC_id] + proton_vector

                # Create the protonated structure
                variant = self._create_protonated_structure(
                    current_frame, proton_pos,
                    {"operation": "add_proton", "site": "NAc_carbonyl",
                     "nac_idx": nac_idx, "theta": theta, "phi": phi_val, "atom_id": OC_id}
                )
                if variant:
                    variants.append(variant)

        except Exception as e:
            logger.debug(f"Failed to add proton to NAc carbonyl group {nac_idx}: {str(e)}")

        return variants

    def _add_protons_to_glycosidic_oxygen(self, info: Dict[str, Any], coor: np.ndarray, current_frame: List) -> List[MolecularStructure]:
        """Add protons to glycosidic oxygen if present (for disaccharides).

        Args:
            info: Structure information dictionary
            coor: Atom coordinates array
            current_frame: Current XYZ frame

        Returns:
            List of structures with protons added to glycosidic oxygen
        """
        variants = []

        # Process each glycosidic oxygen
        for o_idx in info.get('O_glycosidic', []):
            try:
                # Find connected carbon atoms
                connected_c_indices = []

                # Find which ring's C_chain this oxygen is part of
                for ring_idx, o_chain in enumerate(info.get('O_on_C_chain', [])):
                    try:
                        pos_idx = o_chain.index(o_idx)
                        # Get the carbon at the same position
                        c_idx = info.get('C_chain', [])[ring_idx][pos_idx]
                        connected_c_indices.append(c_idx)
                    except ValueError:
                        continue

                if len(connected_c_indices) < 2:
                    logger.debug(f"Not enough connected C atoms found for glycosidic O {o_idx}")
                    continue

                # Add protons at different angles around the glycosidic oxygen
                for angle_val in self.params["angle_around_Oring"]:
                    proton_pos = self._calculate_proton_position(
                        o_idx, connected_c_indices, coor, angle_val,
                        self.params["proton_Oring_dist"]
                    )

                    # Create the protonated structure
                    variant = self._create_protonated_structure(
                        current_frame, proton_pos,
                        {"operation": "add_proton", "site": "glycosidic_O",
                         "angle_val": angle_val, "atom_id": o_idx}
                    )
                    if variant:
                        variants.append(variant)

            except Exception as e:
                logger.debug(f"Failed to add proton to glycosidic oxygen {o_idx}: {str(e)}")

        return variants

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
            logger.debug(f"Error calculating H+ position around O: {str(e)}")
            return None

    def _create_protonated_structure(self, current_frame, proton_pos, metadata):
        """Create a protonated structure with the given proton position and metadata.

        Args:
            current_frame: The original xyz frame
            proton_pos: The position to add the proton
            metadata: Metadata to include in the structure

        Returns:
            A MolecularStructure with the added proton, or None if invalid
        """
        if proton_pos is None:
            return None

        # Add the proton
        proton_coor = ("H", *list(proton_pos))
        protonated_frame = current_frame.copy() + [proton_coor]

        # Check if physically reasonable
        if self.check_physical:
            check_result = is_physical_geometry(protonated_frame, **self.check_physical_kwargs)
            if check_result != "normal":
                logger.debug(f"Generated structure is not physically reasonable: {check_result}")
                return None

        # Convert to MolecularStructure
        mol_structure = MolecularStructure.from_xyz_list(protonated_frame)
        mol_structure.metadata.update(metadata)

        return mol_structure

    def _calculate_proton_position(self, O_id, connecting_atoms, coor, angle, HO_len):
        """
        Calculate H+ position around O with specified parameters.
        
        Args:
            O_id: Index of oxygen atom
            connecting_atoms: List of indices of atoms connected to oxygen
            coor: Array of coordinates
            angle: Rotation angle (radians)
            HO_len: O-H bond length (Angstroms)
            
        Returns:
            Coordinates of the added proton or None on error
        """
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
            logger.debug(f"Error calculating H+ position around O: {str(e)}")
            return None

    def generate_grid(self, structure, proton_grid_point):
        """
        Generate a structure with a specific proton added at a specific position.
        
        Args:
            structure: Input molecular structure
            proton_grid_point: Tuple of (atom_idx, angle_val, atom_type) defining protonation
            
        Returns:
            Protonated structure or None if generation failed
        """
        try:
            # Extract proton grid parameters
            at_idx, angle_val, atom_type = proton_grid_point
            
            # Get structure info
            info = self._get_structure_info(structure)
            current_frame = info['frame']
            coor = info['coor']

            # Process structure info
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
                    # glycosidic O
                    neighbor_c1_pos = info["O_on_C_chain"][0].index(at_idx)
                    neighbor_c2_pos = info["O_on_C_chain"][1].index(at_idx)
                    neighbor_c1_idx = info["C_chain"][0][neighbor_c1_pos]
                    neighbor_c2_idx = info["C_chain"][1][neighbor_c2_pos]
                
                # Convert angle from degrees to radians
                angle_rad = angle_val * (np.pi / 180)
                
                # Calculate the proton position
                proton_pos = self._calculate_proton_position(at_idx, [neighbor_c1_idx, neighbor_c2_idx], coor, angle_rad,
                                                         self.params["proton_Oring_dist"])
                
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
                r = self.params["proton_Oring_dist"]
                
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
                "proton_description": "none",
                "angle_val": angle_val,
                **atom_metadata
            })
            
            return mol_structure
            
        except Exception as e:
            logger.error(f"Error generating protonated structure: {str(e)}", exc_info=True)
            return None

    def set_current_base_structure(self, structure_path: str):
        """Set the current base structure."""
        self.current_base_structure = structure_path
        # Clear cache when changing the base structure
        self._structure_cache.clear()

    def set_base_structures(self, base_structures: List[str]):
        """Set the base structures."""
        self.base_structures = base_structures
        # Clear cache when changing the base structures
        self._structure_cache.clear()

    def set_params(self, **kwargs):
        """Update parameters for the proton addition operation."""
        self.params.update(kwargs)
        # Convert angle lists from degrees to radians if they were updated
        if "angle_list" in kwargs:
            self.params["angle_list"] = [math.radians(ele) for ele in self.params["angle_list"]]
        if "angle_around_Oring" in kwargs:
            self.params["angle_around_Oring"] = [math.radians(ele) for ele in
                                               self.params["angle_around_Oring"]]

    def set_sugar_info(self, sugar_info: Dict[str, List]):
        """Set user-defined sugar structure information."""
        self.sugar_info = sugar_info
        # Clear cache when changing the sugar info
        self._structure_cache.clear()

    def clear_cache(self):
        """Clear the structure cache."""
        self._structure_cache.clear()

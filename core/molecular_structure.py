import numpy as np
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.constants import ATOMIC_NUMBER_MAP

@dataclass
class MolecularStructure:
    """Class to represent a molecular structure with XYZ coordinates."""
    coordinates: np.ndarray
    atomic_numbers: np.ndarray # Shapes: (n_atoms,)
    atoms: List[str] # symbols
    identifier: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_xyz_list(cls, xyz_list: List[Any]) -> 'MolecularStructure':
        """Create a MolecularStructure from an XYZ list format.

        Args:
            xyz_list: List of atoms in format [(symbol, x, y, z), ...]

        Returns:
            MolecularStructure instance
        """
        atoms = [atom[0] for atom in xyz_list]
        coordinates = np.array([atom[1:4] for atom in xyz_list])

        atomic_numbers = np.array([ATOMIC_NUMBER_MAP.get(atom, 0) for atom in atoms])

        return cls(coordinates=coordinates, atomic_numbers=atomic_numbers,
                   atoms=atoms, metadata={})

    @classmethod
    def from_ase_atoms(cls, atoms_obj) -> 'MolecularStructure':
        """Create a MolecularStructure from an ASE Atoms object.
        
        Args:
            atoms_obj: ASE Atoms object
            
        Returns:
            MolecularStructure instance
        """
        coordinates = atoms_obj.get_positions()
        atomic_numbers = atoms_obj.get_atomic_numbers()
        atoms = atoms_obj.get_chemical_symbols()

        metadata = {}
        if hasattr(atoms_obj, "info") and atoms_obj.info:
            metadata.update(atoms_obj.info)

        return cls(coordinates=coordinates, atomic_numbers=atomic_numbers,
                   atoms=atoms, metadata=metadata)

    def to_xyz_str(self) -> str:
        """ Convert structure to XYZ format string.

        Returns:
            XYZ format string representation of the structure
        """
        n_atoms = len(self.atoms)
        xyz_str = f"{n_atoms}\n\n"

        for i in range(n_atoms):
            atom = self.atoms[i]
            x, y, z = self.coordinates[i]
            xyz_str += f"{atom} {x:.6f} {y:.6f} {z:.6f}\n"

        return xyz_str

    def to_xyz_list(self) -> List:
        """Convert the MolecularStructure to an XYZ list format.
        
        Returns:
            List of atoms in format [(symbol, x, y, z), ...]
        """
        xyz_list = []
        
        for i in range(len(self.atoms)):
            atom = self.atoms[i]
            x, y, z = self.coordinates[i]
            xyz_list.append((atom, x, y, z))
            
        return xyz_list

    def get_fingerprint(self) -> str:
        """Generate a unique fingerprint for this structure.

        Returns:
            MD5 hash of the structure's coordinates and atomic numbers
        """
        data = self.coordinates.tobytes() + self.atomic_numbers.tobytes()
        return hashlib.md5(data).hexdigest()

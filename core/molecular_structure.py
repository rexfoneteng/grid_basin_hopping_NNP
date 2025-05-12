import numpy as np
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from core.constants import ATOMIC_NUMBER_MAP

@dataclass
class MolecularStructure:
    """Class to represent a molecular structure with XYZ coordinates."""
    coordinates: np.ndarray
    atomic_numbers: np.ndarray # Shapes: (n_atoms,)
    atoms: List[str] # symbols
    forces: Optional[np.ndarray] = None
    identifier: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_xyz_list(cls, xyz_list: List[Any], forces_list: Optional[List[Tuple[float, float, float]]] = None) -> 'MolecularStructure':
        """Create a MolecularStructure from an XYZ list format.

        Args:
            xyz_list: List of atoms in format [(symbol, x, y, z), ...]

        Returns:
            MolecularStructure instance
        """
        atoms = [atom[0] for atom in xyz_list]
        coordinates = np.array([atom[1:4] for atom in xyz_list])

        atomic_numbers = np.array([ATOMIC_NUMBER_MAP.get(atom, 0) for atom in atoms])

        forces = None
        if forces_list is not None:
            if len(forces_list) != len(xyz_list):
                raise ValueError(f"Forces list length ({len(forces_list)}) must match XYZ list length ({len(xyz_list)})")
            forces = np.array(forces_list)

        return cls(coordinates=coordinates, atomic_numbers=atomic_numbers,
                   atoms=atoms, forces=forces, metadata={})

    @classmethod
    def from_ase_atoms(cls, atoms_obj, include_force=False, conversion_const=1) -> 'MolecularStructure':
        """Create a MolecularStructure from an ASE Atoms object.
        
        Args:
            atoms_obj: ASE Atoms object
            
        Returns:
            MolecularStructure instance
        """
        coordinates = atoms_obj.get_positions()
        atomic_numbers = atoms_obj.get_atomic_numbers()
        atoms = atoms_obj.get_chemical_symbols()
        
        forces = None
        metadata = {}
        if hasattr(atoms_obj, "info") and atoms_obj.info:
            metadata.update(atoms_obj.info)

        if hasattr(atoms_obj, "calc") and atoms_obj.calc is not None:
            try:
                if include_force:
                    forces = atoms_obj.get_forces() * conversion_const
                else:
                    forces = None
            except:
                forces = None

        return cls(coordinates=coordinates, atomic_numbers=atomic_numbers,
                   atoms=atoms, forces=forces, metadata=metadata)

    def to_xyz_str(self, include_forces: bool = False, info="") -> str:
        """ Convert structure to XYZ format string.

        Returns:
            XYZ format string representation of the structure
        """
        n_atoms = len(self.atoms)
        xyz_str = f"{n_atoms}\n{info}\n"

        if self.forces is not None and include_forces:
            for i in range(n_atoms):
                atom = self.atoms[i]
                x, y, z = self.coordinates[i]
                fx, fy, fz = self.forces[i]
                xyz_str += f"{atom} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n"
        else:
            for i in range(n_atoms):
                atom = self.atoms[i]
                x, y, z = self.coordinates[i]
                xyz_str += f"{atom} {x:.6f} {y:.6f} {z:.6f}\n"

        return xyz_str

    def to_xyz_list(self, include_forces: bool = False) -> List:
        """Convert the MolecularStructure to an XYZ list format.
        
        Returns:
            List of atoms in format [(symbol, x, y, z), ...]
        """
        xyz_list = []

        if self.forces is not None and include_forces:
            for i in range(len(self.atoms)):
                atom = self.atoms[i]
                x, y, z = self.coordinates[i]
                fx, fy, fz = self.forces[i]
                xyz_list.append((atom, x, y, z, fx, fy, fz))
        else:        
            for i in range(len(self.atoms)):
                atom = self.atoms[i]
                x, y, z = self.coordinates[i]
                xyz_list.append((atom, x, y, z))
            
        return xyz_list

    def get_forces(self) -> Optional[np.ndarray]:
        """Get the forces array, if available.
        
        Returns:
            NumPy array of forces with shape (n_atoms, 3) or None if forces are not available
        """
        return self.forces

    def set_forces(self, forces: np.ndarray) -> None:
        """Set forces for the structure.
        
        Args:
            forces: NumPy array of forces with shape (n_atoms, 3)
            
        Raises:
            ValueError: If forces shape doesn't match coordinates shape
        """
        if forces.shape != self.coordinates.shape:
            raise ValueError(f"Forces shape {forces.shape} must match coordinates shape {self.coordinates.shape}")
        self.forces = forces
        
    def get_fingerprint(self) -> str:
        """Generate a unique fingerprint for this structure.

        Returns:
            MD5 hash of the structure's coordinates and atomic numbers
        """
        data = self.coordinates.tobytes() + self.atomic_numbers.tobytes()
        return hashlib.md5(data).hexdigest()
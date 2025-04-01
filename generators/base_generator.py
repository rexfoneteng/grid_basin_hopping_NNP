from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import tempfile
import os

from core.molecular_structure import MolecularStructure

class BaseGenerator(ABC):
    """Abstract base class for all structure generators."""

    def __init__(self,
                 check_physical: bool = True,
                 check_physical_kwargs: Optional[Dict[str, Any]] = None,
                 tmp_dir: Optional[str] = None):
        """Initialize the base generator.

            Args:
                check_physical: Whether to check if generated structures are physically reasonable
                check_physical_kwargs: Additional kwargs for physical geometry check
        """
        self.check_physical = check_physical
        self.check_physical_kwargs = check_physical_kwargs

    @abstractmethod
    def __call__(self) -> MolecularStructure:
        """Generate a new structure.

        Returns:
            A new MolecularStructure object
        """
        pass



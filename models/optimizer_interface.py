#!/usr/bin/env python3
"""
Abstract interface for molecular structure optimizers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

from core.molecular_structure import MolecularStructure

class StructureOptimizer(ABC):
    """
    Abstract interface for optimizing molecular structures.
    
    This interface defines the common API for different structure optimization 
    implementations (NNP, GAMESS, etc.).
    """
    
    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
                """
        Optimize a molecular structure.
        
        Args:
            structure: Structure to optimize
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Tuple of (optimized structure, energy, trajectory path)
            If optimization fails, the structure will be None
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the optimizer.
        
        Returns:
            String identifier of the optimizer
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the optimizer.
        
        Returns:
            Dictionary of optimizer parameters
        """
        pass
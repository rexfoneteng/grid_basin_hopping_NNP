# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-18 16:12:08
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-12 12:28:58
# @Description: GAMESS optimizer 

import os
import tempfile
import logging
from typing import Optional, Dict, Any, Tuple

from core.molecular_structure import MolecularStructure
from models.optimizer_interface import StructureOptimizer
from gamess_tools import gamess_vibmin

logger = logging.getLogger(__name__)

class GamessOptimizer(StructureOptimizer):
    """
    Structure optimizer using GAMESS quantum chemistry package.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the GAMESS optimizer.
        
        Args:
            params: Dictionary containing:
                - method: Computational method (e.g., 'B3LYP', 'MP2')
                - basis_set: Basis set (e.g., '6-31G(d)')
                - charge: Molecular charge
                - multiplicity: Spin multiplicity
                - memory: Memory in MW
                - ncpus: Number of CPU cores
                - additional_keywords: Additional GAMESS keywords
                - parm_loc: path of parameters
                - gamess_loc: path of GAMESS
                - preserve_inp: Preserve inp
        """
        self.params = params.copy()

        # Set default parameters
        if "method" not in self.params:
            self.params["method"] = "DFTB3"
        if "charge" not in self.params:
            self.params["charge"] = 0
        if "multiplicity" not in self.params:
            self.params["multiplicity"] = 1
        if "ncpus" not in self.params:
            self.params["ncpus"] = 1
        if "parm_loc" not in self.params:
            self.params["parm_loc"] = "/home/htphan/src/gamess/parameters"
        if "gamess_loc" not in self.params:
            self.params["gamess_loc"] = "/home/htphan/src/gamess/rungms"
        if "preserve_inp" not in self.params:
            self.params["preserve_inp"] = False

    def optimize(self, structure: MolecularStructure, **kwargs) -> Tuple[Optional[MolecularStructure], float, Optional[str]]:
        """
        Optimize a structure using GAMESS.
        
        Args:
            structure: Structure to optimize
            **kwargs: Additional parameters:
                - save_trajectory: Whether to save the trajectory
                - trajectory_dir: Directory to save trajectory in
            
        Returns:
            Tuple of (optimized structure, energy, trajectory path)
        """
        try:
            save_trajectory = kwargs.get("save_trajectory", False)
            trajectory_dir = kwargs.get("trajectory_dir", "trajectories")

            xyz_coor = structure.to_xyz_list()
            result = gamess_vibmin(xyz_coor,
                                   parm_loc=self.params["parm_loc"],
                                   gamess_loc=self.params["gamess_loc"],
                                   molecular_charge=self.params["charge"],
                                   preserve_inp=self.params["preserve_inp"])

            if "additional_keywords" in self.params:
                    gamess_args.update(self.params["additional_keywords"])

            optimized_xyz = result["xyz_list"]
            energy = gamess_opt.get("eng", float("inf"))
            optimized_structure = MolecularStructure.from_xyz_list(optimized_xyz)  

            # Copy metadata from original structure             
            optimized_structure.metadata = structure.metadata.copy()
            optimized_structure.metadata.update({
                                            "optimized": True,
                                            "optimizer": self.params["method"]})

            if not isinstance(energy, float):
                energy = float("inf")

            saved_trajectory_path = None

            return optimized_structure, energy, saved_trajectory_path
            
        except Exception as e:
            logger.error(f"Error in structure optimization: {str(e)}", exc_info=True)
            return None, float('inf'), "none"

    def get_name(self) -> str:
        """
        Get the name of the optimizer.
        
        Returns:
            String identifier of the optimizer
        """
        return "DFTB3"

    def get_params(self) -> str:
        """
        Get the parameters of the optimizer.
        
        Returns:
            Dictionary of optimizer parameters
        """
        return self.params


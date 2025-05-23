# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-17 16:34:44
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-22 17:11:39
# @Description: Neural Network Potential optimizer implementation.

import os
import tempfile
import logging
from typing import Optional, Dict, Any, Tuple, List

import torch
from ase.io import read

from core.molecular_structure import MolecularStructure
from core.constants import eV_TO_Ha_conversion
from models.optimizer_interface import StructureOptimizer
from custom_interface import CustomInterface
from interfaces.gaussian_tools import gaussian_job

from xyz_tools import Xyz

logger = logging.getLogger(__name__)

class NNPOptimizer(StructureOptimizer):
    """
    NNP Optimizer via SchNetPack v0.3.1
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the NNP optimizer.
        
        Args:
            params: Dictionary containing:
                - state_dict: Path to NNP model state dictionary
                - prop_stats: (Optional) Path to property statistics file
                - device: Device to run model on ('cpu' or 'cuda')
                - in_module: Parameters for SchNet input module
                - interface_params: Parameters for interface between ASE and SchNetPack
        """
        self.params = params.copy()
        self.model = None

        # Load NNP model for the checkpoint files
        if "state_dict" in self.params:
            self._load_model()

    def _load_model(self):
        """ Load NNP """
        try:
            from models.nnp_model import NNPModelLoader
            self.model = NNPModelLoader.load_model(self.params)
            if self.model:
                logger.info("NNP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NNP model: {str(e)}", exc_info=True)
            self.model = None

    def optimize(self, structure: MolecularStructure, **kwargs) -> Tuple[Optional[MolecularStructure], float, Optional[str]]:
        """
        Optimize a structure using the NNP model.
        
        Args:
            structure: Structure to optimize
            **kwargs: Additional parameters:
                - fmax: Maximum force criterion for convergence (eV/Å)
                - steps: Maximum number of steps
                - trajectory_dir: Directory to save trajectory in
            
        Returns:
            Tuple of (optimized structure, energy, trajectory path)
        """
        if self.model is None:
            logger.error("Cannot optimize: NNP model not loaded")
            return None, float("inf"), None

        # Get optimization parameters
        fmax = kwargs.get("fmax", self.params.get("fmax", 5e-3))
        steps = kwargs.get("steps", self.params.get("steps", 1000))
        trajectory_dir = kwargs.get("trajectory_dir")

        try:
            # Create a temporary directory for the optimization
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdirname:
                # Save structure to XYZ file
                temp_xyz = f"{tmpdirname}/input.xyz"
                with open(temp_xyz, "w") as f:
                    f.write(structure.to_xyz_str())

                # Set up interface with SchNetPack
                interface_params = self.params.get("interface_params", {
                    "energy": "energy",
                    "forces": "force",
                    "energy_units": "Hartree",
                    "forces_units": "Hartree/A"
                })

                device = self.params.get("device", "cpu")

                ase_interface = CustomInterface(
                    temp_xyz,
                    self.model,
                    tmpdirname,
                    device,
                    **interface_params
                )

                # Run optimization
                optimize_params = {"fmax": fmax, "steps": steps}
                ase_interface.optimize(**optimize_params)

                opt_traj = f"{tmpdirname}/optimization.traj"
                if not os.path.exists(opt_traj):
                    logger.error(f"Optimization failed: output file not found at {opt_traj}")
                    return None, float('inf'), None

                opt_frame = read(opt_traj)
                # The last frame is the optimized structure
                optimized_structure = MolecularStructure.from_ase_atoms(opt_frame)
            
                # Copy metadata from original structure
                optimized_structure.metadata = structure.metadata.copy()
                optimized_structure.metadata.update({
                    "optimized": True,
                    "optimizer": "nnp",
                    "optimize_params": optimize_params
                })
                
                # Get final energy
                local_min_at = read(opt_traj)
                energy = local_min_at.get_potential_energy() * eV_TO_Ha_conversion

                saved_trajectory_path = None
                if trajectory_dir:
                    # Save trajectory if requested
                    if not os.path.exists(trajectory_dir):
                        os.makedirs(trajectory_dir, exist_ok=True)
                    
                    # Generate a filename based on structure metadata
                    grid_point = structure.metadata.get("grid_point", "unknown")
                    grid_point_str = "-".join(map(str, grid_point)) if isinstance(grid_point, tuple) else str(grid_point)
                    trajectory_file_name = f"grid_{grid_point_str}.xyz"
                    saved_trajectory_path = os.path.join(trajectory_dir, trajectory_file_name)
                    
                    # Save the trajectory file
                    self._write_traj(opt_traj, saved_trajectory_path)

            return optimized_structure, energy, saved_trajectory_path

        except Exception as e:
            logger.error(f"Error in structure optimization: {str(e)}", exc_info=True)
            return None, float('inf'), None

    def _write_traj(self, opt_traj: str, saved_trajectory_path: str):
        """Write optimization trajectory in XYZ format"""
        with open(saved_trajectory_path, "a") as f:
            for snapshot in opt_traj:
                energy = snapshot.get_potential_energy() * eV_TO_Ha_conversion
                forces = snapshot.get_forces() * eV_TO_Ha_conversion
                snapshot.info.update({"eng": energy})
                snapshot_structure = MolecularStructure.from_ase_atoms(snapshot, include_force=True)
                snapshot_structure.set_forces(forces)
                xyz_str = snapshot_structure.to_xyz_str(include_forces=True)
                lines = xyz_str.strip().split("\n")
                lines[1] = f"eng= {energy} Properties=species:S:1:pos:R:3:force:R:3"
                xyz_str = "\n".join(lines)
                f.write(xyz_str + "\n")
        logger.info(f"Saved trajectory file to {saved_trajectory_path}")
        return saved_trajectory_path

    def get_name(self) -> str:
        """
        Get the name of the optimizer.
        
        Returns:
            String identifier of the optimizer
        """
        return "nnp"
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the optimizer.
        
        Returns:
            Dictionary of optimizer parameters
        """
        return self.params

class NNPGaussOptimizer(StructureOptimizer):
    """
    Structure optimizer using NNP-Gaussian Interface
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the GAMESS optimizer.
        
        Args:
            params: Dictionary containing:
        """

        self.params = params.copy()

        if "state_dict" in self.params:
            self.model = self.params.pop("state_dict")
        else:
            self.model = None

        if "header_template" not in self.params:
            self.params["header_template"] = \
                "%mem=2GB\n%nprocshared=1\n#opt(loose,nomicro,maxcyc=100) nosymmetry external='run.py g09_external_schnet_func.json {}'\n\n_\n\n1 1\n"

    def optimize(self, structure: MolecularStructure, 
                 convert_to_xyz: bool = True,
                 convert_frame_list= [-1],
                 preserve_out: bool = False,
                 check_out: bool = False, **kwargs) -> Tuple[Optional[MolecularStructure], float, Optional[str]]:
        """
        Optimize structure
        Args
        - convert_to_xyz: bool- Enable convert output to XYZ
        - convert_frame_list: list - list of frames to be exported
        - preserve_out: bool - whether to preserve the output files
        - check_out: bool - True means only export to XYZ if the output file is
                            normally terminated
        """
        if not self.model:
            raise ValueError("The best_model path is not defined")
        elif not self.model.endswith("best_model"):
            raise NotImplementedError("Current implementatation only support 'best_model' only")

        trajectory_dir = kwargs.get("trajectory_dir")

        opt_pref = {"convert_to_xyz": convert_to_xyz,
                    "convert_frame_list": convert_frame_list,
                    "header": self.params["header_template"].format(self.model),
                    "preserve_out": preserve_out}
        opt_pref.update(kwargs)

        try:
            opt_pref.update({"input": structure.to_xyz_list()})
            results = gaussian_job(pref=opt_pref)

            energy = results.get("eng")[-1]

            if not isinstance(energy, float):
                return None, float("inf"), "none"

            optimized_structure = MolecularStructure.from_xyz_list(results.get("xyz_list")[-1])

            # Copy metadata from original structure
            optimized_structure.metadata = structure.metadata.copy()
            optimized_structure.metadata.update({
                    "optimized": True,
                    "optimize_params": {"algorithm": "Gaussian Berny"}
                })

            saved_trajectory_path = None
            if trajectory_dir:
                #print(f"trajectory_dir: {trajectory_dir}")
                # Save trajectory if requested
                if not os.path.exists(trajectory_dir):
                    os.makedirs(trajectory_dir, exist_ok=True)
                    
                    # Generate a filename based on structure metadata
                grid_point = structure.metadata.get("grid_point", "unknown")
                grid_point_str = "-".join(map(str, grid_point)) if isinstance(grid_point, tuple) else str(grid_point)
                trajectory_file_name = f"grid_{grid_point_str}.xyz"
                saved_trajectory_path = os.path.join(trajectory_dir, trajectory_file_name)
                    
                # Save the trajectory file
                self._write_traj(results, saved_trajectory_path)

            return optimized_structure, energy, saved_trajectory_path

        except Exception as e:
            logger.error(f"Error in structure optimization: {str(e)}", exc_info=True)
            return None, float('inf'), None

    def _write_traj(self, opt_results: Dict, saved_trajectory_path: str):
        """Write optimization trajectory in XYZ format"""
        try:
            with Xyz(saved_trajectory_path, "w") as xyz_obj:
                for frame_id, (coord, eng) \
                    in enumerate(zip(opt_results["xyz_list"], opt_results["eng"])):
                    xyz_info = f"frame={frame_id} eng={eng}"
                    xyz_obj.write(coord, xyz_info)

            return saved_trajectory_path
        except Exception as e:
            logger.error(f"Error in writing trajectory: {str(e)}")
            return None

    def get_name(self) -> str:
        """
        Get the name of the optimizer.
        
        Returns:
            String identifier of the optimizer
        """
        return "nnp"
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the optimizer.
        
        Returns:
            Dictionary of optimizer parameters
        """
        return self.params

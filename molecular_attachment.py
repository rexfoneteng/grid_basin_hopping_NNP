#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-05-20 12:38:24
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-20 12:44:35
# @Description:
"""
Molecular attachment utilities for combining molecular fragments.

This module provides functions for attaching molecular fragments to base structures
with proper alignment and atom deletion capabilities.
"""

import numpy as np
from typing import List, Tuple, Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def attach_fragment(
    base_xyz_list: List[Tuple],
    seed_xyz_list: List[Tuple],
    base_align_list: Optional[List[List[int]]] = None,
    seed_align_list: Optional[List[List[int]]] = None,
    base_del_list: Optional[List[int]] = None,
    seed_del_list: Optional[List[int]] = None
) -> List[Tuple]:
    """
    Attach a molecular fragment (seed) to a base structure with proper alignment.
    
    Args:
        base_xyz_list: List of (element, x, y, z) tuples for the base structure
        seed_xyz_list: List of (element, x, y, z) tuples for the fragment to attach
        base_align_list: List of atom index pairs in the base structure used for alignment
        seed_align_list: List of atom index pairs in the seed fragment used for alignment
        base_del_list: List of atom indices to delete from the base structure
        seed_del_list: List of atom indices to delete from the seed fragment
        
    Returns:
        List of (element, x, y, z) tuples for the combined structure
    
    Example:
        To replace a CH group with CH3:
        ```
        attach_fragment(
            base_xyz_list=base_structure,
            seed_xyz_list=methyl_fragment,
            base_align_list=[[c_idx, h_idx]],  # C and H in base
            seed_align_list=[[0, 1]],          # C and first H in methyl
            base_del_list=[c_idx, h_idx],      # Delete original C-H
            seed_del_list=[]                   # Keep all of the methyl
        )
        ```
    """
    try:
        # Validate inputs
        if not base_xyz_list or not seed_xyz_list:
            raise ValueError("Base and seed structures cannot be empty")
        
        # Set defaults for optional arguments
        base_align_list = base_align_list or []
        seed_align_list = seed_align_list or [[0, 1]]  # Default to first two atoms
        base_del_list = base_del_list or []
        seed_del_list = seed_del_list or []
        
        # Convert to numpy arrays for easier manipulation
        base_elements = [atom[0] for atom in base_xyz_list]
        base_coords = np.array([atom[1:4] for atom in base_xyz_list])
        
        seed_elements = [atom[0] for atom in seed_xyz_list]
        seed_coords = np.array([atom[1:4] for atom in seed_xyz_list])
        
        # Process alignment
        if base_align_list and seed_align_list:
            # Get the transformation matrix
            transform_matrix = _get_alignment_transform(
                base_coords, seed_coords, base_align_list, seed_align_list
            )
            
            # Apply transformation to seed coordinates
            seed_coords = _apply_transformation(seed_coords, transform_matrix)
        
        # Create sets of atoms to delete
        base_atoms_to_delete = set(base_del_list)
        seed_atoms_to_delete = set(seed_del_list)
        
        # Create the new combined structure
        combined_xyz_list = []
        
        # Add atoms from base structure (excluding deleted atoms)
        for i, (element, *coords) in enumerate(base_xyz_list):
            if i not in base_atoms_to_delete:
                combined_xyz_list.append((element, *coords))
        
        # Add atoms from seed structure (excluding deleted atoms)
        for i, (element, *_) in enumerate(seed_xyz_list):
            if i not in seed_atoms_to_delete:
                combined_xyz_list.append((element, *seed_coords[i]))
        
        return combined_xyz_list
    
    except Exception as e:
        logger.error(f"Error in attach_fragment: {str(e)}", exc_info=True)
        raise

def _get_alignment_transform(
    base_coords: np.ndarray,
    seed_coords: np.ndarray,
    base_align_list: List[List[int]],
    seed_align_list: List[List[int]]
) -> Dict[str, np.ndarray]:
    """
    Calculate the transformation matrix to align seed fragment with base structure.
    
    Args:
        base_coords: Array of coordinates for the base structure
        seed_coords: Array of coordinates for the seed fragment
        base_align_list: List of atom index pairs in the base structure
        seed_align_list: List of atom index pairs in the seed fragment
        
    Returns:
        Dictionary containing translation vector and rotation matrix
    """
    # Validate alignment lists
    if len(base_align_list) != len(seed_align_list):
        raise ValueError("Base and seed alignment lists must have the same length")
    
    # For simplicity, we'll implement basic alignment using the first pair
    # More sophisticated alignment could use multiple points
    
    # Get the reference points in both structures
    base_point1 = base_coords[base_align_list[0][0]]
    base_point2 = base_coords[base_align_list[0][1]]
    
    seed_point1 = seed_coords[seed_align_list[0][0]]
    seed_point2 = seed_coords[seed_align_list[0][1]]
    
    # Calculate reference vectors
    base_vector = base_point2 - base_point1
    seed_vector = seed_point2 - seed_point1
    
    # Normalize vectors
    base_vector_norm = np.linalg.norm(base_vector)
    seed_vector_norm = np.linalg.norm(seed_vector)
    
    if base_vector_norm < 1e-6 or seed_vector_norm < 1e-6:
        raise ValueError("Reference vectors are too small for alignment")
    
    base_vector = base_vector / base_vector_norm
    seed_vector = seed_vector / seed_vector_norm
    
    # Calculate rotation matrix to align seed vector with base vector
    rotation_matrix = _rotation_matrix_from_vectors(seed_vector, base_vector)
    
    # Calculate translation to move seed_point1 to base_point1
    # First rotate seed_point1
    rotated_seed_point1 = np.dot(rotation_matrix, seed_point1)
    translation_vector = base_point1 - rotated_seed_point1
    
    return {
        "rotation": rotation_matrix,
        "translation": translation_vector
    }

def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Calculate rotation matrix that rotates vec1 to align with vec2.
    
    Args:
        vec1: Source vector to be rotated
        vec2: Target vector to align with
        
    Returns:
        3x3 rotation matrix
    """
    # Handle the case where vectors are parallel (identity rotation)
    if np.allclose(vec1, vec2):
        return np.eye(3)
    
    # Handle the case where vectors are anti-parallel (180° rotation)
    if np.allclose(vec1, -vec2):
        # Find any perpendicular axis
        perp_axis = np.array([1, 0, 0])
        if np.allclose(np.abs(np.dot(perp_axis, vec1)), 1.0):
            perp_axis = np.array([0, 1, 0])
        
        # Make it perpendicular to vec1
        perp_axis = perp_axis - np.dot(perp_axis, vec1) * vec1
        perp_axis = perp_axis / np.linalg.norm(perp_axis)
        
        # Rotate 180° around this axis
        R = _rotation_matrix_from_axis_angle(perp_axis, np.pi)
        return R
    
    # Cross product gives the rotation axis
    axis = np.cross(vec1, vec2)
    axis = axis / np.linalg.norm(axis)
    
    # Calculate the angle between vectors
    cos_angle = np.dot(vec1, vec2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Get rotation matrix from axis-angle representation
    R = _rotation_matrix_from_axis_angle(axis, angle)
    return R

def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create rotation matrix from axis-angle representation.
    
    Args:
        axis: Normalized rotation axis
        angle: Rotation angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    # Normalize axis if it's not already
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def _apply_transformation(
    coords: np.ndarray,
    transform: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Apply rotation and translation to coordinates.
    
    Args:
        coords: Array of coordinates to transform
        transform: Dictionary with 'rotation' matrix and 'translation' vector
        
    Returns:
        Transformed coordinates
    """
    # Apply rotation
    rotated_coords = np.dot(coords, transform["rotation"].T)
    
    # Apply translation
    transformed_coords = rotated_coords + transform["translation"]
    
    return transformed_coords

def scale_bond_length(
    coords: np.ndarray, 
    idx1: int, 
    idx2: int, 
    target_length: float
) -> np.ndarray:
    """
    Scale the distance between two atoms to a target bond length.
    
    Args:
        coords: Array of coordinates
        idx1, idx2: Indices of atoms defining the bond
        target_length: Desired bond length
        
    Returns:
        Coordinates with scaled bond length
    """
    vec = coords[idx2] - coords[idx1]
    current_length = np.linalg.norm(vec)
    
    if current_length < 1e-6:
        raise ValueError("Cannot scale zero-length bond")
    
    scale_factor = target_length / current_length
    
    # Scale the vector
    new_vec = vec * scale_factor
    
    # Update coordinates
    new_coords = coords.copy()
    new_coords[idx2] = coords[idx1] + new_vec
    
    return new_coords
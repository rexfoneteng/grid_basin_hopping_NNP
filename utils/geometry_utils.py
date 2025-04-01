import numpy as np
from typing import List, Tuple
import logging

from rotational_matrix import rotation_matrix_numpy
from geometry_1 import unit_vector

logger = logging.getLogger(__name__)

def add_proton_around_O(
    O_id: int, 
    connecting_atoms: List[int], 
    coor: np.ndarray, 
    angle: float, 
    HO_len: float
) -> np.ndarray:
    """Add H+ around O with specified parameters.
    
    Args:
        O_id: Index of oxygen atom
        connecting_atoms: List of indices of atoms connected to oxygen
        coor: Array of coordinates
        angle: Rotation angle
        HO_len: O-H bond length
        
    Returns:
        Coordinates of the added proton
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
        logger.debug(f"Error adding H+ around O: {str(e)}")
        raise
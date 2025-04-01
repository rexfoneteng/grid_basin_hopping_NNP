#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2023-, Huu Trong Phan (phanhuutrong93@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20230516 22:27:18
# ==============================================================
from collections import OrderedDict
from std_tools import std_err
from xyz_tools import xyz2list
from sklearn.metrics import pairwise_distances
from itertools import product
import numpy as np
#class DistMat:
#    r"""Class to check the physical geometries via the distance matrix
#    Args:
#
#    """
#    def __init__(self, xyz_frame):
#        self.xyz_frame = xyz_frame
#
#        atom_list = [], coor_list = []
#        for coor in self.xyz_frame:
#            atom_list.append(coor[0])
#            coor_list.append(coor[1:])
#        self.coor = np.array(coor_list)
#        self.atoms = atom_list
#
#
def is_physical_geometry(xyz_frame, triu_ids=None, at_at_triu=None, n_atoms=None,
                         brk_label="BRK",
                         CO_min_threshold=1.353, CC_min_threshold=1.4,
                         CH_min_threshold=.52, OH_min_threshold=.5,
                         OO_min_threshold=1.4, HH_min_threshold=.5,
                         catX_min_threshold=1.4, catH_min_threshold=.5,
                         NC_min_threshold=1.0, NO_min_threshold=1.4,
                         NH_min_threshold=.5):
    """
    Check a geometry via the distanace matrix
    Note:
    This script assume the system has zero or one cation absorp in the molecule only.
    More than one cation atoms may not work well by this script
    """
    atom_list, coor_list = [], []
    for coor in xyz_frame:
        atom_list.append(coor[0])
        coor_list.append(coor[1:4])

    cation_list = [ele for ele in atom_list if ele in ("Li", "Na")]
    #print(cation)
    coor_arr = np.array(coor_list)
    at_at = product(atom_list, atom_list)
    at_at_str = [f"{I0}-{I1}" for I0, I1 in at_at]

    dist_mat = pairwise_distances(coor_arr)
    if n_atoms is None:
        n_atoms = len(atom_list)
    at_at_str = np.array(at_at_str).reshape(n_atoms, n_atoms)
    if triu_ids is None:
        triu_ids = np.triu_indices(n_atoms, 1)
    dist_mat_triu = dist_mat[triu_ids]
    if at_at_triu is None:
        at_at_triu = at_at_str[triu_ids]

    CO_map = (at_at_triu == 'C-O') | (at_at_triu == 'O-C')
    CC_map = (at_at_triu == 'C-C')
    CH_map = (at_at_triu == 'C-H') | (at_at_triu == 'H-C')
    OH_map = (at_at_triu == 'O-H') | (at_at_triu == 'H-O')
    OO_map = (at_at_triu == 'O-O')
    HH_map = (at_at_triu == 'H-H')


    check_result = [dist_mat_triu[CO_map].min() < CO_min_threshold,
                    dist_mat_triu[CC_map].min() < CC_min_threshold,
                    dist_mat_triu[CH_map].min() < CH_min_threshold,
                    dist_mat_triu[OH_map].min() < OH_min_threshold,
                    dist_mat_triu[OO_map].min() < OO_min_threshold,
                    dist_mat_triu[HH_map].min() < HH_min_threshold]
    if "N" in atom_list:
        #print("yest")
        NH_map = (at_at_triu == 'N-H') | (at_at_triu == 'H-N')
        NO_map = (at_at_triu == 'N-O') | (at_at_triu == 'O-N')
        NC_map = (at_at_triu == 'N-C') | (at_at_triu == 'C-N')
        check_result.extend([dist_mat_triu[NH_map].min() < NH_min_threshold,
                             dist_mat_triu[NC_map].min() < NC_min_threshold,
                             dist_mat_triu[NO_map].min() < NO_min_threshold])

    if cation_list:
        cation = cation_list[0] #assuming only one cation
        catX_map = (at_at_triu == f'{cation}-O') | (at_at_triu == f'O-{cation}') \
            | (at_at_triu == f'{cation}-C') | (at_at_triu == f'C-{cation}')
        catH_map = (at_at_triu == f'{cation}-H') | (at_at_triu == f'H-{cation}')
        check_result.extend([dist_mat_triu[catX_map].min() < catX_min_threshold,
                            dist_mat_triu[catH_map].min() < catH_min_threshold])
        #print(catH_map)
    check_brk = np.any(check_result)
    if bool(check_brk) is False:
        return "normal"
    else:
        return brk_label

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2022-, Huu Trong Phan (phanhuutrong93@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20220507 18:33:26
# ==============================================================
from collections import OrderedDict
from std_tools import std_err
import numpy as np

from xyz_tools import xyz2list
from ase.io import read

import schnetpack as spk
import torch

from geometry_1 import unit_vector, angle_between
from vector_tools import rotation_matrix


def module_tools(*args, **kwargs):
    """
    Returns:
        return_val: (type)
    Args:
        input: input_folder
        pref: preference which includes all parameters (dict)
    Note:

    Example:

    """
    pref = OrderedDict()
    pref["input"] = "./"
    pref = update_pref(args, kwargs, pref)
    print(pref)
    return result(results=None, pref=pref)

def get_value_from_info(*args, **kwargs):
    """
    Extract values from the xyz info lines
    Return:
    Value (str)
    Args:
    info: (str) info line string
    kw: (str) keyword to extract the value

    Note:
    Examples:
    """
    pref = OrderedDict()
    pref["info"] = ""
    pref["kw"] = ""
    pref = update_pref(args, kwargs, pref)

    info_ls = pref["info"].split()
    val = info_ls[info_ls.index(pref["kw"]) +1]
    return result(results=val, pref=pref)

def si(*args, **kwargs):
    """
    Return:
    si: (float) The similarity index
    Args:
    usr_1, usr_2:: (array-like) The USR descriptors
    Note:
    Example:
    """
    pref = OrderedDict()
    pref["usr_0"] = None
    pref["usr_1"] = None
    pref = update_pref(args, kwargs, pref)

    pref["usr_0"], pref["usr_1"] = [np.array(ele) for ele in
                                    (pref["usr_0"], pref["usr_1"])
                                    if not isinstance(ele, np.ndarray)]
    si = 1 / (np.abs(pref["usr_0"] - pref["usr_1"]).mean() + 1)
    return result(results=si, pref=pref)

def get_coor(*args, **kwargs):
    """
    Return:
    coor: (list of float) Extract the coordinate part from the xyz body
    Args:
    xyz_body: (list, array-like) The xyz body
    Note:
    Example:
    """
    pref = OrderedDict()
    pref["xyz"] = None
    pref = update_pref(args, kwargs, pref)

    if isinstance(pref["xyz"], str):
        pref["xyz"] = xyz2list(pref["xyz"])

    coor = [ele[:4] for ele in pref["xyz"]]
    return result(results=coor, pref=pref)

def nnp_to_eng_forces(*args, **kwargs):
    """
    Returns:
        return_val: (type)
    Args:
        input: input_folder
        model: (spk model object/str) the NNP model
        atom: ase.Atoms
        pref: preference which includes all parameters (dict)
    Note:

    Example:

    """
    pref = OrderedDict()
    pref["model"] = ""
    pref["atoms"] = None
    pref = update_pref(args, kwargs, pref)
    #print(pref)

    if isinstance(pref["model"], str):
        pref["model"] = torch.load(pref["model"], map_location="cpu")

    calculator = spk.interfaces.SpkCalculator(model=pref["model"], device="cpu",
                                              energy="energy", forces="force")

    pref["atoms"].set_calculator(calculator)
    eng = pref["atoms"].get_potential_energy()
    forces = pref["atoms"].get_forces()

    return result(results=(eng, forces), pref=pref)

# pref_tools {
def update_pref(args=None, kwargs=None, pref=None):
    """
	Returns:
        pref: preference dict
    Args:
        args: tuple
        kwargs: dict
        pref: preference dict
    Note:
        kwargs will overwrite args.
        Once pref is used, both args and kwargs will be ignore.
    """
    if "pref" in kwargs:
        if "dict" not in type(kwargs["pref"]).__name__.lower():
            std_err("Error! pref={} is not a dict!".format(kwargs["pref"]))
        for key in kwargs["pref"]:
            pref[key] = kwargs["pref"][key]
        pref["use_pref"] = True
    else:
        pref_keys = list(pref.keys())
        for id, arg in enumerate(args):
            pref[pref_keys[id]] = arg
        for key in kwargs:
            pref[key] = kwargs[key]
        pref["use_pref"] = False
    return pref


def turn_1(*args, **kwargs):
    pref = OrderedDict()
    pref["input"] = [
        "O 1.367749 -1.367429 -0.013497",
        "O -1.368741 -1.368520 0.010409",
        "O 1.368487 1.368708 0.010409",
        "O -1.367495 1.367240 -0.007321",
        "H -0.387179 -1.510905 0.016404",
        "H -1.707588 -1.844689 -0.761917",
        "H 1.857414 -1.715512 0.746192",
        "H 0.386861 1.510873 0.013378",
        "H 1.710892 1.850722 -0.756732",
        "H 1.511250 -0.385951 -0.012002",
        "H -1.510960 0.385737 -0.005700",
        "H -1.852021 1.715084 0.755722",
    ]
    pref["rotate_atom_list"] = [0, 2, 6, 7, 8, 9]
    pref["rotate_bond"] = [0, 4]
    pref["angle"] = 0.5  # 0.5pi = 90 degree
    pref = update_pref(args, kwargs, pref)
    # Perform rotation for seed xyz around base_v
    xyz_list = xyz2list(pref["input"])
    xyz_array = []
    for xyz_info in xyz_list:
        xyz_array.append([xyz_info[1], xyz_info[2], xyz_info[3]])
    xyz_array = np.array(xyz_array)
    #bond_org = np.copy(xyz_array[pref["rotate_bond"][0]])
    if len(pref["rotate_bond"]) == 3:
        rotate_v = pref["rotate_bond"]
    else:
        rotate_v = xyz_array[pref["rotate_bond"][1]] - \
                xyz_array[pref["rotate_bond"][0]]

    rot_mat = rotation_matrix(rotate_v, pref["angle"])
    for rotate_id in pref["rotate_atom_list"]:
        xyz_array[rotate_id] = np.dot(rot_mat, xyz_array[rotate_id])
    #displace_v = bond_org - xyz_array[pref["rotate_bond"][0]]
    displace_v = np.array([0.0, 0.0, 0.0])
    for disp_id in pref["rotate_atom_list"]:
        xyz_array[disp_id] += displace_v
    turn_xyz_list = []
    for I0, xyz in enumerate(xyz_array):
        turn_xyz_list.append([xyz_list[I0][0],
                              xyz_array[I0][0],
                              xyz_array[I0][1],
                              xyz_array[I0][2]])
    return result(turn_xyz_list, pref)

def flip(xyz, bond_0, bond_1, atom_along_bond_0, atom_along_bond_1):
    """
    xyz: (str, list of list) xyz frame
    bond_0: (list, tuple of int) pair of indices of atom in bond_0
    bond_1: (list, tuple of int) pair of indices of atom in bond_1
    atom_along_bond_0: (list, tuple of int) list of atom indices to be flip
            together with those in bond_0
    atom_along_bond_1: (list, tuple of int) list of atom indices to be flip
            together with those in bond_1

    Note:
    For structure such tjat
        A-B
           \
            C
           /
          D
    bond_0 is C-B and bond_1 is C-D
    """
    current_frame = xyz2list(xyz)
    coor = np.array([ele[1:4] for ele in current_frame])

    vec_0 = unit_vector(coor[bond_0[1]] - coor[bond_0[0]])
    vec_1 = unit_vector(coor[bond_1[1]] - coor[bond_1[0]])

    angle = angle_between(vec_0, vec_1)
    perpendicular_vector = np.cross(vec_0, vec_1)

    tmp = turn_1(
        current_frame, rotate_bond=perpendicular_vector,
        rotate_atom_list=atom_along_bond_0, angle=(angle/np.pi))

    translate_vec = coor[bond_0[0]] - np.array(tmp[bond_0[0]][1:4])
    for atom_id in atom_along_bond_0:
        atom_coor = np.array(tmp[atom_id][1:4])
        atom_coor += translate_vec
        tmp[atom_id] = [tmp[atom_id][0]] + [*atom_coor]

    tmp_1 = turn_1(
        tmp, rotate_bond=perpendicular_vector,
        rotate_atom_list=atom_along_bond_1, angle=-(angle/np.pi))

    translate_vec = coor[bond_1[0]] - np.array(tmp_1[bond_1[0]][1:4])
    for atom_id in atom_along_bond_1:
        atom_coor = np.array(tmp_1[atom_id][1:4])
        atom_coor += translate_vec
        tmp_1[atom_id] = [tmp_1[atom_id][0]] + [*atom_coor]

    return tmp_1


def result(results=None, pref={"return": None}):
    """
	Returns:
        results: if pref["return"] == "pref", it will return preference dict
    Args:
        results: any (normal results)
        pref: preference dict
    Note:
        0 could be successful return value.
        1 or other none-zero values for fault results.
    """
    if pref["use_pref"]:
        pref["return"] = results
        return pref
    else:
        return results


# } pref_tools
def main():
    """
    Assign main function here.
    """


if __name__ == "__main__":
    main()



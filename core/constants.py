from ase.units import eV, Hartree

eV_TO_Ha_conversion = eV /  Hartree
# Atom symbol to atomic number mapping
ATOMIC_NUMBER_MAP = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20
}

# Default parameters for structure generation operations
DEFAULT_FLIP_PARAMS = {
    "position": "4NR",  # 1 at Reducing end
    "rotate_after_flip": True,
    "rotating_angle": (-120, 120),
    "cation_list": ("Na", "Li", "K")
}

DEFAULT_ATTACH_ROTATE_PARAMS = {
    "attach_kwargs": {
        "base_align_list": [[27, 43]], 
        "base_del_list": [27, 43, 44],
        "seed_align_list": [[0, 1]],
        "seed_del_list": []
    },
    "rotate_bond_list": [[42, 43]],
    "angle_list": (0, 1/3, 2/3, 1.0, 4/3, 5/3),
    "merge_xyz": False
}

DEFAULT_ADD_PROTON_PARAMS = {
    "angle_list": [-120, 120],  # Rotation angles for OH groups (degrees)
    "angle_around_Oring": [-130, 130],  # Rotation around ring O (degrees)
    "proton_Oring_dist": 0.982  # Distance between O and H+ (Angstroms)
}

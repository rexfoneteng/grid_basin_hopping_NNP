#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
import os
os.sys.path.insert(0, "/home/htphan/git_projects/bh_1")
from basin_hopping.basin_hopping_generator import BasinHoppingGenerator
from basin_hopping.operation_type import OperationType

import logging
import json
from os_tools import file_list


# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   filename='basin_hopping_proton.log')

# Define paths
output_dir = "xyz_opt"
os.makedirs(output_dir, exist_ok=True)

# Define structures
base_structures = file_list("/beegfs/hotpool/htphan/sugar/disaccharides/replace_and_rotate/bGal_14_aGlcNAc/find_structure_bh_ext/base_structure", keys=".lst", filter_type="file_list")
seed_structure = "/beegfs/coldpool/htphan/fragment/C_NAc/C_NAc.xyz"

# NNP model parameters
model_params = {
    "state_dict": "/beegfs/coldpool/htphan/sugar_4/schnetpack_proj/force_pred/m062x_6-311_gpd/find_structure_di/10_add_ext_H_position/01_add_bGal_14_bGlcNAc/cl_learning_lr_5e-4_resume/best_model",
    "prop_stats": "/beegfs/coldpool/htphan/sugar_4/schnetpack_proj/force_pred/m062x_6-311_gpd/property_stats/property_stats_atomrefs_all.pt",
    "device": "cpu",
    "in_module": {
        "n_atom_basis": 128,
        "n_filters": 128,
        "n_gaussians": 75,
        "charged_systems": True,
        "n_interactions": 4,
        "cutoff": 15.0
    },
    "interface_params": {
        "energy": "energy",
        "forces": "force",
        "energy_units": "Hartree",
        "forces_units": "Hartree/Angstrom"
    }
}

# Optimization parameters
optimize_params = {
    "fmax": 4e-3,
    "steps": 1000
}

# Output XYZ file where all local minima will be appended
output_xyz = f"{output_dir}/best_structure.xyz"

# Physical geometry check parameters
PHYSICAL_DICT = {
    "CO_min_threshold": 1.2,
    "CH_min_threshold": 0.88,
    "OO_min_threshold": 1.62,
    "OH_min_threshold": 0.85,
    "NC_min_threshold": 1.20,
    "NO_min_threshold": 1.62,
    "NH_min_threshold": 0.73,
    "HH_min_threshold": 0.66
}

# Define the operation sequence including ADD_PROTON
operation_sequence = [
    OperationType.FLIP,
    OperationType.ATTACH_ROTATE,
    OperationType.ADD_PROTON
]

custom_proton_grid = [
            (16, -130, "OCC"), # atom_index, angle in degree, atom type
            (16, 130, "OCC"),

            (0, -130, "OCC"),
            (0, 130, "OCC"),

            (21, -120, "OCH"),
            (21, 120, "OCH"),

            (19, -120, "OCH"),
            (19, 120, "OCH"),

            (17, -120, "OCH"),
            (17, 120, "OCH"),

            (14, -120, "OCH"),
            (14, 120, "OCH"),

            (39, -130, "OCC"),
            (39, 130, "OCC"),

            (24, -120, "OCH"),
            (24, 120, "OCH"),

            (43, -67.5, "N"),
            (43, 67.5, "N"),

            (50, 0, "OC"),
            (50, 60, "OC"),
            (50, 120, "OC"),
            (50, 180, "OC"),
            (50, 240, "OC"),
            (50, 300, "OC"),

            (40, -120, "OCH"),
            (40, 120, "OCH"),

            (37, -120, "OCH"),
            (37, 120, "OCH")
]

# Create basin hopping generator with the specified operation sequence
bh_generator = BasinHoppingGenerator(
    base_structures=base_structures,
    seed_structure=seed_structure,
    temperature=99999999.9,
    check_physical=True,
    check_physical_kwargs=PHYSICAL_DICT,
    model_params=model_params,
    optimize_params=optimize_params,
    output_xyz=output_xyz,
    max_rejected=1000,  # Stop after 50 consecutive rejections
    operation_sequence=operation_sequence  # Using the custom operation sequence
)

# Customize the grid points for each operation
bh_generator.set_flip_angles([0, 120, 240])  # 3 grid points for rotation after flip
bh_generator.set_attach_angles([0, 60, 120, 180, 240, 300])  # 6 grid points for rotation after attach
bh_generator.set_proton_grid(custom_proton_grid)  # Set custom proton grid

# Run basin hopping with maximum steps
print("Starting basin hopping with proton addition...")
best_structure = bh_generator(n_steps=300)

# Save statistics as JSON
with open(f"{output_dir}/statistics.json", "w") as f:
    json.dump(bh_generator.get_stats(), f, indent=2)

# Print summary
print(f"Basin hopping completed: {bh_generator.stats['stopping_reason']}")
print(f"Best energy found: {bh_generator.best_energy:.6f}")
print(f"Total steps: {bh_generator.stats['total_steps']}")
print(f"Accepted steps: {bh_generator.stats['accepted_steps']}")
print(f"Rejected steps: {bh_generator.stats['rejected_steps']}")
print(f"Maximum consecutive rejections: {bh_generator.stats['max_consecutive_rejections']}")
print(f"Duration: {bh_generator.stats['duration']:.2f} seconds")
print(f"All structures have been saved to: {output_xyz}")

# Print grid point statistics from accepted moves
grid_counts = {}
for entry in bh_generator.history:
    grid_point = entry['grid_point']
    if grid_point in grid_counts:
        grid_counts[grid_point] += 1
    else:
        grid_counts[grid_point] = 1

# Count operations
operation_counts = {op.name: 0 for op in operation_sequence}
for op in operation_sequence:
    for entry in bh_generator.history:
        op_desc = entry.get('operations', {})
        if op == OperationType.FLIP and 'flip_angle' in op_desc:
            operation_counts[op.name] += 1
        elif op == OperationType.ATTACH_ROTATE and 'attach_angle' in op_desc:
            operation_counts[op.name] += 1
        elif op == OperationType.ADD_PROTON and 'proton_description' in op_desc:
            operation_counts[op.name] += 1

# Calculate proton site distribution
proton_site_counts = {}
for entry in bh_generator.history:
    op_desc = entry.get('operations', {})
    if 'proton_description' in op_desc:
        proton_desc = op_desc['proton_description']
        if proton_desc in proton_site_counts:
            proton_site_counts[proton_desc] += 1
        else:
            proton_site_counts[proton_desc] = 1

print("\nOperation statistics from accepted moves:")
for op, count in operation_counts.items():
    print(f"{op}: {count}")

print("\nProton site distribution:")
for site, count in sorted(proton_site_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{site}: {count}")

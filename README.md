# Basin Hopping for Carbohydrate Molecular Systems

A structure optimization tool that performs grid-based sampling using Basin Hopping algorithm with multiple optimizer backends (Neural Network Potentials and GAMESS).

## Overview

This package provides tools for efficiently exploring the conformational space of carbohydrate molecules using a grid-based Basin Hopping algorithm. The implementation supports various operations (flip, attach-rotate, add-proton) and can use different optimization backends:

1. Neural Network Potentials (NNP) through SchNetPack
2. GAMESS quantum chemistry package

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/basin-hopping.git
cd basin-hopping
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- NumPy
- SchNetPack (for NNP optimizer)
- GAMESS tools (for GAMESS optimizer)
- ASE (Atomic Simulation Environment)
- PyTorch
- PyYAML

## Usage

The basic usage is:

```
python bh.py -c config.yaml
```

### Command-line Options

```
python bh.py --help
```

Key options:
- `-c/--config`: Path to YAML configuration file
- `-o/--output-dir`: Directory for output files
- `-b/--base-structures`: List of base structure XYZ files
- `-s/--seed-structure`: Seed structure XYZ file for attachment operations
- `--operations`: List of operations to perform (flip, attach_rotate, add_proton)
- `-t/--temperature`: Temperature for Metropolis criterion (K)
- `-n/--steps`: Number of basin hopping steps to perform
- `--max-rejected`: Maximum consecutive rejected moves before stopping
- `--optimizer-type`: Type of optimizer to use (nnp or gamess)
- `--save-trajectories`: Save optimization trajectories
- `--export-config`: Export default configuration to specified YAML file and exit

## Configuration

You can configure the Basin Hopping process using a YAML file. A minimal example:

```yaml
output_dir: xyz_opt 
base_structures: structures.lst
seed_structure: fragment.xyz 
temperature: 0.132 
max_rejected: 10
save_trajectories: true
steps: 100
operations: [flip, attach_rotate, add_proton]

# NNP optimizer configuration
optimizer:
  type: nnp
  params:
    state_dict: /path/to/model/best_model
    device: cpu
    in_module: 
      n_atom_basis: 128
      n_filters: 128
      n_gaussians: 75
      charged_systems: true
      n_interactions: 4
      cutoff: 15.0
```

### Configuring Different Optimizers

#### Neural Network Potentials (NNP)

```yaml
optimizer:
  type: nnp
  params:
    state_dict: /path/to/model/best_model
    prop_stats: /path/to/property_stats.pt  # Optional, for older model formats
    device: cpu  # or cuda
    in_module: 
      n_atom_basis: 128
      n_filters: 128
      n_gaussians: 75
      charged_systems: true
      n_interactions: 4
      cutoff: 15.0
    interface_params: 
      energy: energy
      forces: force
      energy_units: Hartree
      forces_units: Hartree/Angstrom
```

#### GAMESS

```yaml
optimizer:
  type: gamess
  params:
    method: B3LYP
    basis_set: 6-31G(d)
    charge: 0
    multiplicity: 1
    memory: 100
    ncpus: 4
    additional_keywords:
      nadvfn: 1  # Example of additional GAMESS keyword
```

### Exportable Default Configuration

You can export a default configuration file:

```bash
python bh.py --export-config default_config.yaml
```

## Output Files

- `accepted_structure.xyz`: XYZ file containing all accepted structures
- `rejected_structure.xyz`: XYZ file containing all rejected structures
- `statistics.json`: JSON file with statistics from the Basin Hopping run
- `trajectories/`: Directory containing optimization trajectories (if `save_trajectories` is enabled)

## Extending the System

### Adding a New Optimizer

To add a new optimizer backend:

1. Create a new class that inherits from `StructureOptimizer` in the `models` directory
2. Implement the required methods (`__init__`, `optimize`, `get_name`, `get_params`)
3. Update the `OptimizerFactory` class to include your new optimizer

Example for a new optimizer called "NewMethod":

```python
# models/new_method_optimizer.py
from models.optimizer_interface import StructureOptimizer

class NewMethodOptimizer(StructureOptimizer):
    def __init__(self, params):
        self.params = params
        # Initialize your optimizer
        
    def optimize(self, structure, **kwargs):
        # Implement optimization logic
        # Return (optimized_structure, energy, trajectory_path)
        
    def get_name(self):
        return "new_method"
        
    def get_params(self):
        return self.params
```

Then update the factory:

```python
# models/optimizer_factory.py
from models.new_method_optimizer import NewMethodOptimizer

# In OptimizerFactory.create_optimizer method:
elif optimizer_type.lower() == 'new_method':
    logger.info("Creating NewMethod optimizer")
    return NewMethodOptimizer(params)
```

### Adding a New Operation

To add a new operation type:

1. Add a new enum value in `OperationType`
2. Create a new generator class that inherits from `BaseGenerator`
3. Update the `BasinHoppingGenerator` to handle the new operation

## Examples

### Basic Basin Hopping with NNP

```bash
python bh.py -c examples/nnp_example.yaml
```

### Basin Hopping with GAMESS

```bash
python bh.py -c examples/gamess_example.yaml
```

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@misc{Basin-Hopping,
  author = {Phan, Huu Trong},
  title = {Basin Hopping for Carbohydrate Molecular Systems},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/basin-hopping}}
}
```

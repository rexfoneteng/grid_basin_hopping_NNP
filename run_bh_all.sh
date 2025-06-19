#!/bin/bash
# ==============================================================
# Copyright(c) 2025-, Po-Jen Hsu (clusterga@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20250508 15:50:04
# ==============================================================

# run_all.sh - Master script to execute the full workflow

set -e  # Exit on error

echo "======================="
echo "BASIN HOPPING WORKFLOW"
echo "======================="

# Step 1: Create directories
mkdir -p logs output base_structures/subset

# Step 2: Split base structures
echo "[1/5] Splitting base structures..."
#python split_base_structures.py
split -n 48 base_structures/nnp_minima.lst --suffix-length=2 --numeric-suffixes=0 base_structures/subset/nnp_minima_

# Step 3: Split proton grid
echo "[2/5] Creating proton grid variants..."
python /beegfs/hotpool/htphan/sugar/disaccharides/replace_and_rotate/bGal_14_aGlcNAc/find_structure_bh_ext_1/yaml_split_proton_grid.py

# Step 4: Generate job configs
echo "[3/5] Generating job configurations..."
python /beegfs/hotpool/htphan/sugar/disaccharides/replace_and_rotate/bGal_14_aGlcNAc/find_structure_bh_ext_1/yaml_generate_configs.py

# Count the number of configs
NUM_CONFIGS=$(ls configs/config_job_*.yaml | wc -l)
echo "[4/5] Preparing to submit $NUM_CONFIGS jobs..."

# Step 5: Submit jobs
echo "[5/5] Submitting job array..."
# Determine which job scheduler to use (SLURM or PBS)
if command -v sbatch &> /dev/null; then
    # SLURM environment
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=basin_hopping
#SBATCH --output=logs/basin_hopping_%A_%a.out
#SBATCH --error=logs/basin_hopping_%A_%a.err
#SBATCH --array=1-${NUM_CONFIGS}%20  # Limit to 20 concurrent jobs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# Load required modules
module load python/3.8

# Get the config file for this job
CONFIG_FILE=configs/config_job_\${SLURM_ARRAY_TASK_ID}.yaml

# Run the basin hopping
python bh.py -c \${CONFIG_FILE}
EOF
    echo "Jobs submitted with SLURM. Use 'squeue' to check status."
elif command -v qsub &> /dev/null; then
    # PBS environment
    qsub <<EOF
#!/bin/bash
#PBS -N basin_hopping
#PBS -o logs/bh.qerr
#PBS -e logs/bh.qout
#PBS -q mem192v2
#PBS -t 1-${NUM_CONFIGS}
#PBS -l nodes=1:ppn=1

##PBS -l select=1:ncpus=1:mem=4gb
##PBS -l walltime=24:00:00

# Move to the working directory
account=`whoami`
cd \$PBS_O_WORKDIR
source ~/.bashrc

pyscript=/beegfs/coldpool/htphan/sugar/disaccharides/script/src/bh_nnp_1/bh.py

# Load required modules
#module load python/3.8

# Get the config file for this job
#CONFIG_FILE=configs/config_job_\${PBS_ARRAY_INDEX}.yaml
CONFIG_FILE=configs/config_job_\${PBS_ARRAYID}.yaml

# Run the basin hopping
python \${pyscript} -c \${CONFIG_FILE}
EOF
    echo "Jobs submitted with PBS."
else
    echo "No supported job scheduler found (SLURM or PBS). Please submit jobs manually."
fi

echo "Setup complete. Start the progress monitor with: python monitor_progress.py"

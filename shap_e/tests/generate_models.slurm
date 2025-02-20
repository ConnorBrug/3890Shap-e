#!/bin/bash
#SBATCH --job-name=Shap-e_Generation
#SBATCH --output=/home/bruggecr/shap-e/shap_e/tests/generation_output.log
#SBATCH --error=/home/bruggecr/shap-e/shap_e/tests/generation_error.log
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=pascal
#SBATCH --account=es3890_acc

# Load Modules
module load GCCcore/.11.3.0 Python/3.10.4

# Activate Environment
source ~/shap-e/shap-e-env/bin/activate

# Set environment variables for DDP
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Debugging: Print Python version and paths
echo "Python location: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"
python -c "import site; print('Python site-packages:', site.getsitepackages())"

# Confirm GPU is available
echo "GPU Check: $(nvidia-smi)"

# Run the Python script
/panfs/accrepfs.vampire/home/bruggecr/shap-e/shap-e-env/bin/python ~/shap-e/shap_e/tests/generate.py


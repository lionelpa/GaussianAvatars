#!/bin/bash

# prettyfy util method
print_decorated_message() {
    echo
    echo "============================================"
    echo "$1"
    echo "============================================"
    echo
}


# Exit immediately if a command exits with a non-zero status
set -e

if ! command -v conda &> /dev/null; then
    echo "Conda is not initialized in the current shell. Running 'conda init'..."
    conda init bash
    # After running 'conda init', the script will need to be re-sourced.
    exec bash
fi

# Step 1: Create the conda environment with Python 3.10
print_decorated_message "Creating conda environment 'avatars'..."
conda create --name avatars -y python=3.10

# Step 2: Activate the conda environment
print_decorated_message "Activating the environment..."
source $(conda info --base)/etc/profile.d/conda.sh  # Ensure conda is recognized in scripts
conda activate avatars

# Step 3: Install the CUDA toolkit and Ninja
print_decorated_message "Installing CUDA toolkit and Ninja..."
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja -y

# Step 4: Create symbolic link to avoid linker error
print_decorated_message "Creating symlink for lib64..."
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"

# Step 5: Set CUDA_HOME environment variable
print_decorated_message "Setting CUDA_HOME environment variable..."
conda env config vars set CUDA_HOME=$CONDA_PREFIX
echo "Set CUDA_HOME to ${CUDA_HOME}!"

# Step 6: Install PyTorch with the matching CUDA version
print_decorated_message "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Step 7: Install other required packages from requirements.txt
print_decorated_message "Installing packages from requirements.txt"
pip install -r requirements.txt

print_decorated_message "Environment setup is complete!"

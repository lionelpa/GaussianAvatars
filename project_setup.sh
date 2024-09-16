#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Create the conda environment with Python 3.10
echo "Creating conda environment 'avatars'..."
conda create --name avatars -y python=3.10

# Step 2: Activate the conda environment
echo "Activating the environment..."
conda activate avatars

# Step 3: Install the CUDA toolkit and Ninja
echo "Installing CUDA toolkit and Ninja..."
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja -y

# Step 4: Create symbolic link to avoid linker error
echo "Creating symlink for lib64..."
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"

# Step 5: Set CUDA_HOME environment variable
echo "Setting CUDA_HOME environment variable..."
conda env config vars set CUDA_HOME=$CONDA_PREFIX
echo "Set CUDA_HOME to ${CUDA_HOME}!"

# Step 6: Install PyTorch with the matching CUDA version
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Step 7: Install other required packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Environment setup is complete!"

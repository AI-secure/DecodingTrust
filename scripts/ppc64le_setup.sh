#!/bin/bash

echo "Enter the file path to AutoGPTQ wheel (auto_gptq-0.5.0.dev0+cu118-cp39-cp39-linux_ppc64le.whl)"
read -r whl

# NOTE: install open-ce in a directory with plenty of free space, like /usr/workspace/$USER
envName="llm-dt"

# Load necessary modules (exact version may vary by system)
module load gcc/9.3.1

# install to conda_env subdirectory within current working directory
install_dir="$(pwd)/dt_conda_env"

## install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
sh Miniconda3-latest-Linux-ppc64le.sh -f -p "$install_dir"

# activate conda environment (SPECIFIC TO MY SYSTEM but this is the users .bashrc file after conda install)
source .bashrc_dt_env

# create an OpenCE environment in conda (Python-3.9)
conda config --prepend channels 'https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/'
conda config --prepend channels 'https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/'
conda config --prepend channels 'https://opence.mit.edu'
conda config --prepend channels 'https://ftp.osuosl.org/pub/open-ce/current/'
conda config --prepend channels 'conda-forge'   # highest priority
conda create --name "${envName}" python==3.9 pytorch=2.0.1 torchvision=0.15.2 spacy=3.5.3 scipy=1.10.1 fairlearn~=0.9.0 scikit-learn~=1.1.2 pandas~=2.0.3 pyarrow~=11.0.0 matplotlib=3.6.3 rust -c conda-forge

# activate the OpenCE environment
conda activate "${envName}"

# install precompiled AutoGPT-Q
pip install "${whl}"

# install some packages (need to install matplotlib specific version first; could probably include in conda create)
pip install "decoding-trust[gptq] @ git+https://github.com/AI-secure/DecodingTrust.git@release"

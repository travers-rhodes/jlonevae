#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate lil_disentanglement_challenge

export PYTHONPATH="..:../disentanglement_lib"
export DISENTANGLEMENT_LIB_DATA="../scratch/dataset"
export NDC_ROOT=$(realpath "../")
python3 train_standard_tf_models.py

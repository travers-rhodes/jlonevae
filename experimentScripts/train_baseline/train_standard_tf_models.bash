#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate lil_disentanglement_challenge

export PYTHONPATH="./:./disentanglement_lib"
export DISENTANGLEMENT_LIB_DATA="data"
export NDC_ROOT=$(realpath "./")
export TF_FORCE_GPU_ALLOW_GROWTH=true
python3 ./jlonevae_lib/baseline_lib/train/train_baseline_models.py

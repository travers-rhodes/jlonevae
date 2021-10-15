#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae 

export PYTHONPATH="./:./disentanglement_lib"
export DISENTANGLEMENT_LIB_DATA="data/"
export AICROWD_DATASET_NAME="threeDotsCache"
for seed in {11..20}
do
  for gamma in 0.1 0.05 0.2 0.4
  do
    echo "Running with seed: $seed"
    python ./jlonevae_lib/train/train_jlonevae_models.py --beta 4 --gamma $gamma --latent-dim 10 --epochs 60 --annealingBatches 100000 --lr 0.0001 --regularization_type jltwo --seed $seed
  done
done

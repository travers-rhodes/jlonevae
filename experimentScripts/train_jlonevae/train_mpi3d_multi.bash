#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae

export PYTHONPATH="./:./disentanglement_lib"
export DISENTANGLEMENT_LIB_DATA="data/"
export AICROWD_DATASET_NAME="mpi3d_multi_real"
for seed in {11..15}
do
  for latentDim in 10
  do
    for beta in 0.01
    do
      for gamma in 0 0.01
      do
        for lr in 0.0001
        do 
          echo $latentDim
          python ./jlonevae_lib/train/train_jlonevae_models.py --beta $beta --gamma $gamma --latent-dim $latentDim --epochs 60 --annealingBatches 100000 --lr $lr --seed $seed
          exit 1
	      done
      done
    done
  done
done


#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae

export PYTHONPATH="./"

experimentName="naturalImages"
commonFlags="--experimentName $experimentName --vaeShape tiny16 --numBatches 100000 --recordLossEvery 100 --batchSize 128 --lr 0.001 --annealingBatches 50000 --beta 0.01 --latentDim 10"

for runId in 0
do
  python3 ./jlonevae_lib/train/train_jlonevae_without_disentanglement_lib.py $commonFlags --runId _run$runId --gamma 0
  python3 ./jlonevae_lib/train/train_jlonevae_without_disentanglement_lib.py $commonFlags --runId _run$runId --gamma 0.01
done

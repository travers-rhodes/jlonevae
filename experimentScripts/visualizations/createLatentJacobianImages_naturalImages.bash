#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae

export PYTHONPATH="./"

experimentName=trainedNaturalImageModels

for latentDim in 10
do
python3 experimentScripts/visualizations/createLatentJacobianImages_naturalImages.py --experimentName $experimentName --datasetName naturalImages --beta 0.01 --annealingBatches 50000 --latentDim $latentDim --batchNumber 100000
python3 experimentScripts/visualizations/createLatentJacobianImages_naturalImages.py --experimentName $experimentName --datasetName naturalImages --beta 0.01 --annealingBatches 50000 --latentDim $latentDim --batchNumber 100000 --gamma 0.01
done

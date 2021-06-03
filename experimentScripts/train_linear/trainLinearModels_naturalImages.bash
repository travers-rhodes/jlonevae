#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae

export PYTHONPATH="./"

datasetName=naturalImage
experimentName=$datasetName
commonArgs="--experimentName $experimentName --datasetName $datasetName --batchSize 10000 --modelType PCA ICA --latentDim 5 10 25"
python3 ./experimentScripts/train_linear/trainLinearModels_naturalImages.py $commonArgs

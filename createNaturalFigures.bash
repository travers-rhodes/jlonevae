#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae

experimentName=naturalImages

# create data
if [[ ! -f ./data/$experimentName ]]
then
	printf "generating data"
	./data/download_natural_images_data.sh $experimentName
	./data/sampleNatualImagePatches.py --experimentName $experimentName
else
	printf "using previously-made data\n"
fi

# train models
	commonFlags="--experimentName $experimentName --vaeShape tiny16 --numBatches 15000 --recordLossEvery 1"
printf "(re)training models"
rm -rf trainedModels/$experimentName
# run 10 trials with BatchNorm (SGD) and without Batch Norm (SGD and Adam)
for runId in {0..9}
do
	  for lr in 0.01 0.001 0.0001 0.1
	  do
	  	python3 01-trainModels.py $commonFlags --useAdam 0 --runId _run$runId --batchNormalize 1 --lr $lr
	done
done

#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

#https://stackoverflow.com/questions/7069682/how-to-get-arguments-with-flags-in-bash/21128172
train_model=false
make_images=false
while getopts 'ti' flag; do
	case "${flag}" in
		t) train_model=true;;
		i) make_images=true;;
		*) printf "unknown flag\n"
		   exit 1;;
	esac
done

source ~/.bashrc

# activate conda within this child script
conda activate jlonevae 

datasetName=naturalImages
experimentName=analyticNaturalImage

commonArgs="--experimentName $experimentName --modelType PCA ICA --latentDim 5 25 50 100"

# train models
if "$train_model" = true ;
then
	printf "(re)training models"
	rm -rf analyticModels/$experimentName
  python3 01a-trainAnalyticModel.py --datasetName $datasetName --batchSize 10000 $commonArgs
else
	printf "using previously-trained models\n"
fi

# create images 
if "$make_images" = true ;
then
	printf "saving_images"
	rm -rf images/analyticModels/$experimentName
  python3 02a-createAnalyticFigures.py $commonArgs
else
	printf "not saving images\n"
fi

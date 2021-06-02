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
echo "root is $NDC_ROOT"


#if [ "$#" -ne 2 ]; then
#  echo "we require the model directory and the dataset name"
#  exit 1
#fi

export AICROWD_DATASET_NAME=threeDots

#export metricOutputFolder="combinedStandardOutputs/$AICROWD_DATASET_NAME"
#mkdir -p $metricOutputFolder

export AICROWD_OUTPUT_PATH="trainedStandardModels"
for folder in $AICROWD_OUTPUT_PATH/*/*;
do
  echo $folder
  # ignore the "trainedStandardModels" part of the path
  export AICROWD_EVALUATION_NAME=${folder:22}
  export experimentNameBase=$(echo "$AICROWD_EVALUATION_NAME" | cut -d/ -f1)
  echo "running on $AICROWD_EVALUATION_NAME"
  echo "running $folder"
  python postprocess_trained_standard_models.py
  #break # just for testing 
done

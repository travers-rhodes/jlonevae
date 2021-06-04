#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae 

export PYTHONPATH=".:./disentanglement_lib"
export DISENTANGLEMENT_LIB_DATA="data"
export NDC_ROOT=$(realpath "./")
export TF_FORCE_GPU_ALLOW_GROWTH=true
echo "root is $NDC_ROOT"


export AICROWD_DATASET_NAME=threeDots

export AICROWD_OUTPUT_PATH="trainedStandardModels"
for folder in $AICROWD_OUTPUT_PATH/*/*;
do
  echo $folder
  # ignore the "trainedStandardModels" part of the path
  export AICROWD_EVALUATION_NAME=${folder:22}
  export experimentNameBase=$(echo "$AICROWD_EVALUATION_NAME" | cut -d/ -f1)
  echo "running on $AICROWD_EVALUATION_NAME"
  echo "running $folder"
  python ./jlonevae_lib/baseline_lib/evaluate/evaluation.py 
done

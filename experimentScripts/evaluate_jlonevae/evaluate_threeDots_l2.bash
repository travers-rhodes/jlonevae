#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae 

# The name of the dataset to evaluate with
export AICROWD_DATASET_NAME="threeDots"

# to filter down which experiments are evaluated, add a search string here
export experimentNameFilter="defaultConv_lone_beta4_0000_ica0_1000_lat10_batch64_lr0_0001_anneal100000"

export PYTHONPATH="./:./disentanglement_lib"
export DISENTANGLEMENT_LIB_DATA="data"
export NDC_ROOT=$(realpath "./")
echo "root is $NDC_ROOT"
# the folder where trained models are stored
# note: if you change this variable, be sure to modify the code below
# that strips this part of the path to create AICROWD_EVALUATION_NAME
export AICROWD_OUTPUT_PATH="trainedModels"
for folder in $AICROWD_OUTPUT_PATH/*/*;
do
  echo $folder
  # only run if the training completed and saved a checkpoint at batch 300000
  if [[ -f $folder/representation/cache_batch_no300000/model_type.txt ]]; then
    # ignore the "trainedModels" part of the path
    export AICROWD_EVALUATION_NAME=${folder:14}
    export experimentNameBase=$(echo "$AICROWD_EVALUATION_NAME" | cut -d/ -f1)
    echo "running on $AICROWD_EVALUATION_NAME"
    echo "running $folder"
    # For now, only generate results for 10 latent dims (for all betas)
    if [[ "$experimentNameBase" == *"$experimentNameFilter"* ]]; then
      echo "Contains the right search string, so running."
      python ./jlonevae_lib/evaluate/evaluation.py 
    fi
  else
    echo "Model did not complete and save in $folder. Skipping"
  fi
done

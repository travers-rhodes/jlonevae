import os
import numpy as np
import datetime

from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.config.unsupervised_study_v1.sweep import UnsupervisedStudyV1

def train_on_three_dots(original_model_num, output_directory="trainedStandardModels"):
  overwrite = True
  study = UnsupervisedStudyV1()
  model_bindings, model_config_file = study.get_model_config(original_model_num)

  # simplest (hackiest) way to train exact same configs on new dataset
  # is to take the old config and just...change the dataset name
  if not model_bindings[0] == "dataset.name = 'dsprites_full'":
    raise RuntimeError("I was hoping the first arg would always be the dataset")
  model_bindings[0] = "dataset.name = 'threeDotsCache'"
  
  print(model_bindings)
  print(model_config_file)
  
  print("Training model...")
  model_dir = os.path.join(output_directory, f"model{original_model_num}_on3dots", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  model_bindings = [
      "model.name = '{}'".format(os.path.basename(model_config_file)).replace(
          ".gin", ""),
      "model.model_num = {}".format(original_model_num),
  ] + model_bindings
  train.train_with_gin(model_dir, overwrite, [model_config_file],
                           model_bindings)
  print(model_bindings)
  print(model_config_file)


# The UnsupervisedStudyV1 has (in lowest bit order first):
# 50 trials of each model.
# 6 hyperparameters for each model setting (300 so far)
# 6 model types (1800 so far)
# 6 datasets (10800 total).

# Thus, the following indices ranges gives the 3rd hyperparameter choice for
# each of the 6 models:

for seed_number_adder in np.arange(1,10):
  original_model_nums = np.arange(100,1800,300, dtype=np.int) + seed_number_adder
  for omn in original_model_nums:
    train_on_three_dots(omn)

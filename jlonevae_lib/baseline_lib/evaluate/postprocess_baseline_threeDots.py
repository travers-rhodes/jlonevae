# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
import gin.tf

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = os.getenv("AICROWD_OUTPUT_PATH","./scratch/shared")
experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name")
overwrite = True
experiment_output_path = os.path.join(base_path, experiment_name)
ROOT = os.getenv("NDC_ROOT", ".")

# 0.1 Helpers
# ------------------------------------------------------------------------------


# Extract the mean representation for both of these models.
representation_path = os.path.join(experiment_output_path, "representation")
model_path = os.path.join(experiment_output_path)
# This contains the settings:
postprocess_gin = [os.path.join(ROOT, "jlonevae_lib","baseline_lib","evaluate","postprocess_threeDots.gin")]
postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                 postprocess_gin)

print("Written output to : ", experiment_output_path)

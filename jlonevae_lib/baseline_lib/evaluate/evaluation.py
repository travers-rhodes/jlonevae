# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
# Copyright 2021 JLONEVAE_ANONYMOUS_AUTHORS.  All rights reserved.
#
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

# This file was modified by JLONEVAE_ANONYMOUS_AUTHORS 

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# trying so hard to mute tensorflow warnings...
# https://stackoverflow.com/questions/57539273/disable-tensorflow-logging-completely
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


import os
import disentanglement_lib

#try:
# Monkey patch in neurips2019_disentanglement_challengs's evaluate, 
# which supports pytorch *and* tensorflow.
if True: #JLONEVAE_ANONYMOUS_AUTHORS: we use pytorch, so don't hide any errors about it
    import jlonevae_lib.evaluate.evaluate_helper as evaluate
    disentanglement_lib.evaluation.evaluate = evaluate
    MONKEY = True
#except ImportError:
#    # No pytorch, no problem.
#    MONKEY = False

from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize import visualize_model
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1
import gin.config
import gin.tf
import json
import numpy as np

##############################################################################
# 0. Settings
# By default, we save all the results in subdirectories of the following path.
##############################################################################
base_path = os.getenv("AICROWD_OUTPUT_PATH","./scratch/shared")
experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name")
overwrite = True
experiment_output_path = os.path.join(base_path, experiment_name)
ROOT = os.getenv("NDC_ROOT", ".")

# Print the configuration for reference
if not MONKEY:
    print(f"Evaluating Experiment '{experiment_name}' from {base_path}.")
else:
    import jlonevae_lib.utils.utils_pytorch as utils_pytorch
    exp_config = utils_pytorch.get_config()
    print(f"Evaluating Experiment '{exp_config.experiment_name}' "
          f"from {exp_config.base_path} on dataset {exp_config.dataset_name}")

# ----- Helpers -----


def get_full_path(filename):
    return os.path.join(ROOT, filename)


##############################################################################
# Gather Evaluation Configs | Compute Metrics
##############################################################################
_study = unsupervised_study_v1.UnsupervisedStudyV1()
evaluation_configs = sorted(_study.get_eval_config_files())

#Add local config files from jlonevae package
jlonevaeMetricConfigPath=get_full_path("jlonevae_lib/config/metrics_configs")
for configFile in os.listdir(jlonevaeMetricConfigPath):
    evaluation_configs.append(get_full_path(os.path.join(jlonevaeMetricConfigPath, configFile)))

# allow the internal "import 'local_mig_base.gin'" to resolve file name correctly
# unhelpfully, it still will return an error that it can't find the file (the first place it searches)
# but print statements show it does get loaded (eventually)
gin.config.add_config_file_search_path(get_full_path("jlonevae_lib/config/metrics_configs"))

# Compute individual metrics
expected_evaluation_metrics = [
    'local_mig_0_1',
    'local_modularity_0_1',
]

# we need a separate dataset config because we need to
# be able to black out certain latent factors from the dataset which are not ordered
# (in particular, the cars3d factor on type of car)
gin_dataset_config = get_full_path(f"jlonevae_lib/config/dataset_configs/{exp_config.dataset_name}.gin")

for gin_eval_config in evaluation_configs:
    metric_name = gin_eval_config.split("/")[-1].replace(".gin", "")
    if  metric_name not in expected_evaluation_metrics:
        # Ignore unneeded evaluation configs
        continue
    print("Evaluating Metric : {}".format(metric_name))
    result_path = os.path.join(
                    experiment_output_path,
                    "metrics",
                    metric_name
                )
    representation_path = os.path.join(
                            experiment_output_path,
                            "representation"
                        )
    eval_bindings = [
        "evaluation.random_seed = {}".format(0),
        "evaluation.name = '{}'".format(metric_name)
    ]                        
    evaluate.evaluate_with_gin(
                representation_path,
                result_path,
                overwrite,
                [gin_eval_config, gin_dataset_config],
                eval_bindings
                )

# Gather evaluation results
evaluation_result_template = "{}/metrics/{}/results/aggregate/evaluation.json"
final_scores = {}
for _metric_name in expected_evaluation_metrics:
    evaluation_json_path = evaluation_result_template.format(
        experiment_output_path,
        _metric_name
    )
    evaluation_results = json.loads(
            open(evaluation_json_path, "r").read()
    )
    if _metric_name == "factor_vae_metric":
        _score = evaluation_results["evaluation_results.eval_accuracy"]
        final_scores["factor_vae_metric"] = _score
    elif _metric_name == "dci":
        _score = evaluation_results["evaluation_results.disentanglement"]
        final_scores["dci"] = _score
    elif "mig" in _metric_name:
        _score = evaluation_results["evaluation_results.discrete_mig"]
        final_scores[_metric_name] = _score
    elif "modularity" in _metric_name: 
        _score = evaluation_results["evaluation_results.modularity_score"]
        final_scores[_metric_name] = _score
    elif _metric_name == "sap_score":
        _score = evaluation_results["evaluation_results.SAP_score"]
        final_scores["sap_score"] = _score
    elif _metric_name == "irs":
        _score = evaluation_results["evaluation_results.IRS"]
        final_scores["irs"] = _score
    else:
        raise Exception("Unknown metric name : {}".format(_metric_name))

print("Final Scores : ", final_scores)


##############################################################################
# (Optional) Generate Visualizations
##############################################################################
# model_directory = os.path.join(experiment_output_path, "model")
# visualize_model.visualize(model_directory, "viz_output/")

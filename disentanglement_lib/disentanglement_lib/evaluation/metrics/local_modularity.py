# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
# Copyright 2021 Travers Rhodes.  All rights reserved.
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

# This file was modified by Travers Rhodes

"""Modularity and explicitness metrics from the F-statistic paper.

Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"
(https://arxiv.org/pdf/1802.05312.pdf).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation.metrics import modularity_explicitness
import numpy as np
import gin.tf


@gin.configurable(
    "local_modularity",
    denylist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_local_modularity(ground_truth_data,
                                    representation_function,
                                    random_state,
                                    artifact_dir=None,
                                    num_train=gin.REQUIRED,
                                    num_local_clusters=gin.REQUIRED,
                                    batch_size=16):
  """Computes the modularity metric according to Sec 3.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_local_clusters: how many times to run the local mig calculation.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with average modularity score and average explicitness
      (train and test).
  """
  del artifact_dir
  mod_results = []
  for modrun in range(num_local_clusters):
    #print("Generating training set %d." % modrun)
    mus_train, ys_train = utils.generate_local_batch_factor_code(
        ground_truth_data, representation_function, num_train,
        random_state, batch_size)
    discretized_mus = utils.make_discretizer(mus_train)
    #print(mus_train.shape, ys_train.shape)
    mutual_information = utils.discrete_mutual_info(discretized_mus, ys_train)
    # Mutual information should have shape [num_codes, num_factors].
    assert mutual_information.shape[0] == mus_train.shape[0]
    assert mutual_information.shape[1] == ys_train.shape[0]
    mod_results.append(modularity_explicitness.modularity(mutual_information))
  mod_results = np.array(mod_results)
  scores = {}
  scores["modularity_score"] = np.mean(mod_results) 
  scores["local_modularity_scores_samples"] = mod_results.tolist()
  return scores

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

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).

Modified from mig.py by Travers Rhodes to apply the MIG metric to 
a local sample of data rather than a global sample
"""
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation.metrics import mig 
import numpy as np
import gin.tf


@gin.configurable(
    "local_mig",
    denylist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_local_mig(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_local_clusters=gin.REQUIRED,
                batch_size=16):
  """Computes the mutual information gap.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.
    num_local_clusters: how many times to run the local mig calculation.

  Returns:
    Dict with average local mutual information gap across different local
    clusters.
  """
  del artifact_dir
  mig_results = []
  for migrun in range(num_local_clusters):
    logging.info("Generating training set %d." % migrun)
    mus_train, ys_train = utils.generate_local_batch_factor_code(
        ground_truth_data, representation_function, num_train,
        random_state, batch_size)
    assert mus_train.shape[1] == num_train
    score_dict = mig._compute_mig(mus_train, ys_train)
    logging.debug("local cluster mig score: %s" % score_dict)
    mig_results.append(score_dict["discrete_mig"])
  mig_results = np.array(mig_results)
  avg_score_dict = {}
  avg_score_dict["discrete_mig"] = np.mean(mig_results)
  avg_score_dict["local_discrete_migs_samples"] = mig_results.tolist()
  return avg_score_dict

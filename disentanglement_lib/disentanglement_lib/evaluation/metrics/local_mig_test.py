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

# Added/modified by Travers Rhodes
# Based on mig_test.py

"""Tests for local_mig.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from absl import logging
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import local_mig, utils
import numpy as np
import gin.tf


def _identity_discretizer(target, num_bins):
  del num_bins
  return np.array(target,dtype=np.int32)

def initialize_gin():
  gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
  #gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)
  gin.bind_parameter("discretizer.num_bins", 10)
  gin.bind_parameter("local_sample_factors.denylist_factors", []) 
  gin.bind_parameter("local_sample_factors.locality_proportion", 1.0)
  gin.bind_parameter("local_mig.num_train", 10000)
  gin.bind_parameter("local_mig.num_local_clusters", 20)
  gin.bind_parameter("local_sample_factors.continuity_cutoff", 10)

class LocalMIGTest(absltest.TestCase):

  def test_metric(self):
    initialize_gin()
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x,dtype=np.float)
    random_state = np.random.RandomState(0)
    scores = local_mig.compute_local_mig(
        ground_truth_data, representation_function, random_state)
    #print(scores)
    self.assertBetween(scores["discrete_mig"], 0.9, 1.0)

  def test_bad_metric(self):
    initialize_gin()
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = np.zeros_like
    random_state = np.random.RandomState(0)
    scores = local_mig.compute_local_mig(
        ground_truth_data, representation_function, random_state)
    self.assertBetween(scores["discrete_mig"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    initialize_gin()
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = local_mig.compute_local_mig(
        ground_truth_data, representation_function, random_state)
    self.assertBetween(scores["discrete_mig"], 0.0, 0.1)

  # the representation here folds the dataset in on itself.
  # it looks like MIG counts this as a relatively good discretization which
  # makes sense because mutual information is not correlation
  def test_locally_good_metric(self):
    initialize_gin()
    num_factors = 10
    num_possible_values = 40
    ground_truth_data = dummy_data.IdentityObservationsCustomSize(
        [num_possible_values] * num_factors)

    random_state = np.random.RandomState(30)
    # our representation function is the mod operator---good locally
    # and terrible globally
    def representation_function(x):
      return np.array(x % 6, dtype=np.float64)

    # if you sample globally (lp = 1) then you get a low MIG score
    # if you sample locally (lp=0.1) then you get a high MIG score
    for lp, exp_min_score, exp_max_score in [(1.0, 0.4, 0.6), (0.1, 0.7, 0.9)]:
        gin.bind_parameter("local_sample_factors.locality_proportion" ,lp)
        gin.bind_parameter("local_mig.num_local_clusters",3)
        scores = local_mig.compute_local_mig(
            ground_truth_data, representation_function, random_state, None,
            batch_size=30)
        self.assertBetween(scores["discrete_mig"], exp_min_score, exp_max_score)


if __name__ == "__main__":
  #logging.set_verbosity(logging.DEBUG)
  absltest.main()

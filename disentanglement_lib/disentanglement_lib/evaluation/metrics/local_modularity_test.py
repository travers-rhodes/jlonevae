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

"""Tests for local_modularity.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import utils, local_modularity
import numpy as np
from six.moves import range
import gin.tf


def _identity_discretizer(target, num_bins):
  del num_bins
  return target

def initialize_gin():
  gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
  #gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)
  gin.bind_parameter("discretizer.num_bins", 10)
  gin.bind_parameter("local_sample_factors.denylist_factors", []) 
  gin.bind_parameter("local_sample_factors.locality_proportion", 1.0)
  gin.bind_parameter("local_sample_factors.continuity_cutoff", 0)
  gin.bind_parameter("local_modularity.num_train", 20000)
  gin.bind_parameter("local_modularity.num_local_clusters", 3)

class ModularityExplicitnessTest(absltest.TestCase):

  def test_metric(self):
    initialize_gin()
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = local_modularity.compute_local_modularity(
        ground_truth_data, representation_function, random_state)
    self.assertBetween(scores["modularity_score"], 0.9, 1.0)

  def test_bad_metric(self):
    initialize_gin()
    ground_truth_data = dummy_data.IdentityObservationsData()
    random_state_rep = np.random.RandomState(0)
    # The representation which randomly permutes the factors, should have equal
    # non-zero MI which should give a low modularity score.
    def representation_function(x):
      code = np.array(x, dtype=np.float64)
      for i in range(code.shape[0]):
        code[i, :] = random_state_rep.permutation(code[i, :])
      return code
    random_state = np.random.RandomState(0)
    scores = local_modularity.compute_local_modularity(
        ground_truth_data, representation_function, random_state)
    self.assertBetween(scores["modularity_score"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    initialize_gin()
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = local_modularity.compute_local_modularity(
        ground_truth_data, representation_function, random_state)
    self.assertBetween(scores["modularity_score"], 0.9, 1.0)
  
  # the representation here folds the dataset in on itself.
  # it looks like MIG counts this as a relatively good discretization which
  # makes sense because mutual information is not correlation
  def test_locally_good_metric(self):
    initialize_gin()
    gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)
    num_factors = 3
    num_possible_values = 40
    ground_truth_data = dummy_data.IdentityObservationsCustomSize(
        [num_possible_values] * num_factors)

    random_state = np.random.RandomState(30)
    # modularity is so weird. We want a factor that globally
    # depends on multiple variables but locally depends on only
    # one variable. Easiest is something like Floor[(A+B) / 5] + C / 40
    # which for small changes of A,B,C only depends on changes of C
    # and globally depends jointly on A,B
    def representation_function(x):
      representation = np.floor((x[:,0:1] + x[:,1:2])/12) + x[:,2:3] / 40
      return representation

    # if you sample globally (lp = 1) then you get a low MIG score
    # if you sample locally (lp=0.1) then you get a high MIG score
    for lp, exp_min_score, exp_max_score in [(1.0, 0.0, 0.3), (0.1, 0.8, 1.0)]:
        gin.bind_parameter("local_sample_factors.locality_proportion" ,lp)
        gin.bind_parameter("local_modularity.num_local_clusters",3)
        scores = local_modularity.compute_local_modularity(
            ground_truth_data, representation_function, random_state,
            batch_size=30)
        self.assertBetween(scores["modularity_score"], exp_min_score, exp_max_score)

if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  absltest.main()

# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
# Copyright 2021 The DisentanglementLib Authors.  All rights reserved.
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

# This file was modified by Travers Rhodes in 2021

"""Dummy data sets used for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import ground_truth_data
import numpy as np


class IdentityObservationsData(ground_truth_data.GroundTruthData):
  """Data set where dummy factors are also the observations."""

  @property
  def num_factors(self):
    return 10

  @property
  def observation_shape(self):
    return 10

  @property
  def factors_num_values(self):
    return [10] * 10

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return random_state.randint(10, size=(num, self.num_factors))

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    return factors

  @property
  def factor_names(self):
    return ["Factor {}".format(i) for i in range(self.num_factors)]

class IdentityObservationsCustomSize(ground_truth_data.GroundTruthData):
  """Data set where dummy factors are also the observations.
     With customizable number of values and factors,
     and the same number of factors and values per factor"""
  def __init__(self, num_values_for_factors = [10] * 10):
    self._factors_num_values = num_values_for_factors

  @property
  def num_factors(self):
    return len(self._factors_num_values) 

  @property
  def observation_shape(self):
    return len(self._factors_num_values)

  @property
  def factors_num_values(self):
    return self._factors_num_values 

  def sample_factors(self, num, random_state):
    """Sample a batch of the latent factors.
       Copied and modified from util.py"""
    factors = np.zeros(
        shape=(num, self.num_factors), dtype=np.int64)
    for i, num_val in enumerate(self.factors_num_values):
      factors[:, i] = random_state.randint(num_val, size=num)
    return factors

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y.
       Since this is IdentityObservations, just return factors."""
    return factors

  @property
  def factor_names(self):
    return ["Factor {}".format(i) for i in range(self.num_factors)]


class DummyData(ground_truth_data.GroundTruthData):
  """Dummy image data set of random noise used for testing."""

  @property
  def num_factors(self):
    return 10

  @property
  def factors_num_values(self):
    return [5] * 10

  @property
  def observation_shape(self):
    return [64, 64, 1]

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return random_state.randint(5, size=(num, self.num_factors))

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    return random_state.random_sample(size=(factors.shape[0], 64, 64, 1))


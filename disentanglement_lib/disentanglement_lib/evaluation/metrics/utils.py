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

# This file was modified by Travers Rhodes in 2021

"""Utility functions that are useful for the different metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import sklearn
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
import gin.tf


def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations = \
        ground_truth_data.sample(num_points_iter, random_state)
    if i == 0:
      factors = current_factors
      representations = representation_function(current_observations)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def split_train_test(observations, train_percentage):
  """Splits observations into a train and test set.

  Args:
    observations: Observations to split in train and test. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]
  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test set."
  return observations_train, observations_test


def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.

  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations)
    else:
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h


@gin.configurable(
    "discretizer", denylist=["target"])
def make_discretizer(target, num_bins=gin.REQUIRED,
                     discretizer_fn=gin.REQUIRED):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)


@gin.configurable("histogram_discretizer", denylist=["target"])
def _histogram_discretize(target, num_bins=gin.REQUIRED):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized


def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


@gin.configurable("predictor")
def make_predictor_fn(predictor_fn=gin.REQUIRED):
  """Wrapper that creates classifiers."""
  return predictor_fn


@gin.configurable("logistic_regression_cv")
def logistic_regression_cv():
  """Logistic regression with 5 folds cross validation."""
  return linear_model.LogisticRegressionCV(Cs=10,
                                           cv=model_selection.KFold(n_splits=5))


@gin.configurable("gradient_boosting_classifier")
def gradient_boosting_classifier():
  """Default gradient boosting classifier."""
  return ensemble.GradientBoostingClassifier()


# TSR: This method generates a local sample from the factor space.
# It then randomly selects a batch of latent factor values that are "near" to 
# the latent value given by factor_centroid
# Inputs:
#  -num (int): the number of samples to draw
#  -locality_proportion (float): The normalized radius of random sampling for
#     continuous variables. For example, if a latent factor has 100 values and
#     the locality_proportion is 0.1 then we will sample latent values for that
#     factor within 10 of the center value for that factor.
#  -factors_num_values (int[]): The number of possible integer values for each 
#     factor.
#     IMPORTANT ASSUMPTION:
#        we assume that The latent factors are all integers in a range 0 to
#        factors_num_values[index]-1 (inclusive)
#  -factors_centroid: the factor set that we should sample near
#  -continuity_cutoff (int): factors with fewer than this many discrete values
#     are considered "discontinuous" and will have the same value sampled within 
#     each batch sampled
#  -random_state (obj): the numpy random_state object for deterministic sampling
@gin.configurable("local_sample_factors", 
        denylist=["num", "factors_num_values", "factor_centroid", "random_state"])
def local_sample_factors(num, factors_num_values, factor_centroid, random_state, 
        locality_proportion=gin.REQUIRED, continuity_cutoff=gin.REQUIRED, 
        denylist_factors=gin.REQUIRED):
  """Sample a batch of the latent factors locally around some central sampled
  point."""
  #print("Using locality_proportion of %f" % locality_proportion)
  #print("Using continuity_cutoff of %d" % continuity_cutoff)
  #print("Using blackout indices of %s" % denylist_factors)
  factors = np.zeros(
      shape=(num, len(factors_num_values)), dtype=np.int64)
  for i, num_values in enumerate(factors_num_values):
    center = factor_centroid[i] 
    # if this factor has too few factors to be considered "continuous"
    # then just return the center value for this factor
    if num_values >= continuity_cutoff and i not in denylist_factors:
      radius = np.floor(num_values * locality_proportion)
    else:
      #print("not sampling this i, num_values, center", i, num_values, center)
      radius = 0
    factors[:, i] = sample_integers_around_center(center, radius, minv=0,
        maxv = num_values-1, num=num, random_state=random_state)
  return factors


# sample num different integers within radius (inclusive) of center.
# clipped below to minv (usually 0) and above to maxv (usually num_values - 1)
# using the random_state object passed in 
def sample_integers_around_center(center, radius, minv, maxv, num, random_state):
  lower_bound = max(minv, center-radius)
  upper_bound = min(maxv, center+radius)
  return random_state.randint(lower_bound, upper_bound + 1, size=num,
      dtype=np.int32)

def generate_local_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.
     but ensure that the training sample is "close" in generative factor space.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  factors_num_values = ground_truth_data.factors_num_values

  # compute the centroid around which we sample this batch
  factor_centroid = []
  for i, num_values in enumerate(factors_num_values):
    factor_centroid.append(random_state.randint(num_values))

  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors = local_sample_factors(num_points_iter, 
        factors_num_values, factor_centroid, random_state)
    current_observations = ground_truth_data.sample_observations_from_factors(
        current_factors, random_state)
    if i == 0:
      factors = current_factors
      representations = representation_function(current_observations)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)

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

"""Tests for utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.data.ground_truth import dummy_data
import numpy as np


class UtilsTest(absltest.TestCase):

  def test_histogram_discretizer(self):
    # Input of 2D samples.
    target = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       [0.6, .5, .4, .3, .2, .1]])
    result = utils._histogram_discretize(target, num_bins=3)
    shouldbe = np.array([[1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1]])
    np.testing.assert_array_equal(result, shouldbe)

  def test_discrete_entropy(self):
    target = np.array([[1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1]])
    result = utils.discrete_entropy(target)
    shouldbe = np.log(3)
    np.testing.assert_allclose(result, [shouldbe, shouldbe])

  def test_discrete_mutual_info(self):
    xs = np.array([[1, 2, 1, 2], [1, 1, 2, 2]])
    ys = np.array([[1, 2, 1, 2], [2, 2, 1, 1]])
    result = utils.discrete_mutual_info(xs, ys)
    shouldbe = np.array([[np.log(2), 0.], [0., np.log(2)]])
    np.testing.assert_allclose(result, shouldbe)

  def test_split_train_test(self):
    xs = np.zeros([10, 100])
    xs_train, xs_test = utils.split_train_test(xs, 0.9)
    shouldbe_train = np.zeros([10, 90])
    shouldbe_test = np.zeros([10, 10])
    np.testing.assert_allclose(xs_train, shouldbe_train)
    np.testing.assert_allclose(xs_test, shouldbe_test)

  def test_local_sample_factors(self):
    random_state = np.random.RandomState(3)
    # sample range of 10% of num_factors
    factor_num_values = [1, 9, 10, 11, 100, 101]
    factor_centroid = np.array([0, 4, 9, 3, 10, 10])
    samps = utils.local_sample_factors(1000, 0.1, 
        factor_num_values, factor_centroid, 0, random_state)
    np.testing.assert_equal(samps.shape, (1000, 6))
    self.assertTrue(np.all(samps[:,0] == 0))
    # should all have the same value, since 0.1 * 9 < 1
    self.assertTrue(np.max(samps[:,1]) - np.min(samps[:,1]) == 0)
    # should have diameter of 2 for both these
    for inx in [2,3]:
      assert_correct_radius(self, samps[:,inx], 1, 0, factor_num_values[inx]-1)
    # should have diameter of 20 for both these
    for inx in [4,5]:
      assert_correct_radius(self, samps[:,inx], 10, 0, factor_num_values[inx]-1)
    # same experiment, but now we don't consider any factor 
    # with numfactors less than 11 to count as continuous (so 10 should now also
    # return all same values)
    # sample range of 10% of num_factors
    factor_num_values = [1, 9, 10, 11, 100, 110]
    samps = utils.local_sample_factors(1000, 0.15, 
        factor_num_values, factor_centroid, 11, random_state)
    np.testing.assert_equal(samps.shape, (1000, 6))
    self.assertTrue(np.all(samps[:,0] == 0))
    # should all have the same value
    for inx in [1,2]:
      self.assertTrue(np.max(samps[:,inx]) - np.min(samps[:,inx]) == 0)
    # should have radius 1 for this, since floor(0.15 * 11) = 1
    for inx in [3]:
      assert_correct_radius(self, samps[:,inx], 1, 0, factor_num_values[inx]-1)
    # should have diameter of 20 for both these
    for inx in [4]:
      assert_correct_radius(self, samps[:,inx], 15, 0, factor_num_values[inx]-1)
    for inx in [5]:
      assert_correct_radius(self, samps[:,inx], 16, 0, factor_num_values[inx]-1)

  def test_sample_integers_around_center(self):
    random_state = np.random.RandomState(3)
    for i in range(20):
      sample = utils.sample_integers_around_center(5, 3, 0, 10, 100, random_state)
      self.assertTrue(np.all(sample <= 8))
      self.assertTrue(np.all(sample >= 2))
      self.assertTrue(np.any(sample > 6))
      self.assertTrue(np.any(sample < 4))
    for i in range(20):
      sample = utils.sample_integers_around_center(5, 3, 4, 6, 100, random_state)
      self.assertTrue(np.all(sample <= 6))
      self.assertTrue(np.all(sample >= 4))
    sample = utils.sample_integers_around_center(5, 0, 4, 6, 100, random_state)
    self.assertTrue(np.all(sample == 5))
    self.assertTrue(len(sample) == 100)
    self.assertTrue(sample.dtype == np.int32)

  def test_generate_batch_factor_code(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    num_points = 100
    random_state = np.random.RandomState(3)
    batch_size = 192
    represents, factors = utils.generate_batch_factor_code(ground_truth_data, 
        representation_function, num_points, random_state, batch_size)
    # representation is identity
    for batch in [represents, factors]:
      np.testing.assert_equal(batch.shape, [10, num_points])
      for inx in range(10):
        self.assertEqual(np.min(batch[inx,:]), 0)
        self.assertEqual(np.max(batch[inx,:]), 10 - 1)

  # just for debugging
  #def test_print_sample(self):
  #  ground_truth_data = dummy_data.IdentityObservationsCustomSize([100] * 10)
  #  representation_function = lambda x: np.array(x % 50, dtype=np.float64)
  #  num_points = 10
  #  random_state = np.random.RandomState(3)
  #  batch_size = 192
  #  local_repr, local_facts = utils.generate_local_batch_factor_code(ground_truth_data, 
  #      representation_function, num_points, random_state, batch_size,
  #      locality_proportion=1.0, continuity_cutoff=0.0)
  #  print(local_repr)
  #  print(local_facts)

  def test_generate_local_batch_factor_code(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    num_points = 100
    random_state = np.random.RandomState(3)
    # you gotta test batch size smaller than num_points, silly
    batch_size = 13 
    local_repr, local_facts = utils.generate_local_batch_factor_code(ground_truth_data, 
        representation_function, num_points, random_state, batch_size,
        locality_proportion=1.0, continuity_cutoff=0.0)
    for local_batch in [local_repr, local_facts]:
      np.testing.assert_equal(local_batch.shape, [10,num_points])
      for inx in range(10):
        self.assertEqual(np.min(local_batch[inx,:]), 0)
        self.assertEqual(np.max(local_batch[inx,:]), 10 - 1)
    local_repr, local_facts = utils.generate_local_batch_factor_code(ground_truth_data, 
        representation_function, num_points, random_state, batch_size,
        locality_proportion=0.1, continuity_cutoff=0.0)
    # representation is identity
    for local_batch in [local_repr, local_facts]:
      np.testing.assert_equal(local_batch.shape, [10, num_points])
      for inx in range(10):
        assert_correct_radius(self, local_batch[inx,:], 1, 0, 10-1)
    


# used in the sampling test
# samples should span the full 2 * radius unless they hit an upper/lower bound
def assert_correct_radius(tester, array, radius, lowbound, upbound):
  minval = np.min(array)
  maxval = np.max(array)
  if minval == lowbound or maxval == upbound:
    tester.assertTrue(maxval - minval <= 2 * radius)
  else:
    tester.assertEqual(maxval - minval, 2 * radius)


if __name__ == '__main__':
  absltest.main()

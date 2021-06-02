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
from disentanglement_lib.data.ground_truth import mpi3d_multi
import numpy as np
import cv2
import gin.tf
import os

class MPI3DMultiTest(absltest.TestCase):
  def x_test_data_shape(self):
    os.environ["DISENTANGLEMENT_LIB_DATA"]= "../scratch/dataset/"
    random_state = np.random.RandomState(0)
    three_dots_data = mpi3d_multi.MPI3DMulti("mpi3d_real")
    num = 1
    sample = three_dots_data.sample_observations(num, random_state)
    np.testing.assert_equal(sample.shape, [num, 64, 64, 3])

  # remove the x before this method to have it run by unittest framework
  # this pops up some plots of the generated images so you can confirm they
  # "look correct"
  def test_data_visualize_x_y_axis(self):
    os.environ["DISENTANGLEMENT_LIB_DATA"]= "../scratch/dataset/"
    random_state = np.random.RandomState(0)
    three_dots_data = mpi3d_multi.MPI3DMulti("mpi3d_real")
    num = 3
    samples = three_dots_data.sample_observations(num, random_state)

    for sample in samples:
      cv2.imshow('image',sample)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
  # confirm the range 0-63 looks right
  def x_test_data_visualize_bounds(self):
    random_state = np.random.RandomState(0)
    three_dots_data = threeDots.ThreeDots()
    latent_factors_test = np.ones((2,6)) * 63
    latent_factors_test[0] *= 0
    samples = three_dots_data.sample_observations_from_factors(latent_factors_test, random_state)
    for sample in samples:
      cv2.imshow('image',sample)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

if __name__ == "__main__":
  #logging.set_verbosity(logging.DEBUG)
  absltest.main()

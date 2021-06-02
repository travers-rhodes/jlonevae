# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""MPI3D data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import tensorflow.compat.v1 as tf



class MPI3DMulti(ground_truth_data.GroundTruthData):
  """MPI3D dataset.

  MPI3D datasets have been introduced as a part of NEURIPS 2019 Disentanglement
  Competition.(http://www.disentanglement-challenge.com).
  There are three different datasets:
  1. Simplistic rendered images (mpi3d_toy).
  2. Realistic rendered images (mpi3d_realistic).
  3. Real world images (mpi3d_real).

  Currently only mpi3d_toy is publicly available. More details about this
  dataset can be found in "On the Transfer of Inductive Bias from Simulation to
  the Real World: a New Disentanglement Dataset"
  (https://arxiv.org/abs/1906.03292).

  The ground-truth factors of variation in the dataset are:
  0 - Object color (4 different values for the simulated datasets and 6 for the
    real one)
  1 - Object shape (4 different values for the simulated datasets and 6 for the
    real one)
  2 - Object size (2 different values)
  3 - Camera height (3 different values)
  4 - Background colors (3 different values)
  5 - First DOF (40 different values)
  6 - Second DOF (40 different values)
  """

  def __init__(self, mode="mpi3d_toy"):
    if mode == "mpi3d_toy":
      mpi3d_path = os.path.join(
          os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_toy",
          "mpi3d_toy.npz")
      if not tf.io.gfile.exists(mpi3d_path):
        raise ValueError(
            "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
            .format(mode))
      else:
        with tf.io.gfile.GFile(mpi3d_path, "rb") as f:
          data = np.load(f)
      self.factor_sizes = [4, 4, 2, 3, 3, 40, 40]
    elif mode == "mpi3d_realistic":
      mpi3d_path = os.path.join(
          os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_realistic",
          "mpi3d_realistic.npz")
      if not tf.io.gfile.exists(mpi3d_path):
        raise ValueError(
            "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
            .format(mode))
      else:
        with tf.io.gfile.GFile(mpi3d_path, "rb") as f:
          data = np.load(f)
      self.factor_sizes = [4, 4, 2, 3, 3, 40, 40]
    elif mode == "mpi3d_real":
      mpi3d_path = os.path.join(
          os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_real",
          "mpi3d_real.npz")
      if not tf.io.gfile.exists(mpi3d_path):
        raise ValueError(
            "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
            .format(mode))
      else:
        #print(mpi3d_path)
        with tf.io.gfile.GFile(mpi3d_path, "rb") as f:
          data = np.load(f)
      self.factor_sizes = [6, 6, 2, 3, 3, 40, 40]
    else:
      raise ValueError("Unknown mode provided.")

    self.images = data["images"]
    self.latent_factor_indices = range(7) 
    self.num_total_factors = 7*4
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes * 4

  @property
  def observation_shape(self):
    return [64, 64, 3]


  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    factors = []
    for innerindx in range(4):
      factors.append(self.state_space.sample_latent_factors(num, random_state))
    return np.concatenate(factors, axis=1)

  # input factors is of shape numsamples x numfactors
  def sample_observations_from_factors(self, factors, random_state):
    # build up a 2x2 grid of images by choosing indices for each of the 4 images
    # and then combining them into one larger image
    inner_imgs = []
    for innerindx in range(4):
      #print(innerindx,self.num_factors, factors.shape)
      all_factors = self.state_space.sample_all_factors(factors[:,
                                (innerindx*self.num_factors):
                                   ((innerindx+1)*self.num_factors)]
                              , random_state)
      # simple hack to only ever sample large blue squares
      all_factors[:,0] = 3
      all_factors[:,1] = 1
      all_factors[:,2] = 1 
      all_factors[:,3] = 0 
      all_factors[:,4] = 2 
      indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
      # pull out this indices image and take every other pixel (superfast
      # resolution reduction
      inner_imgs.append(self.images[indices][:,0:64:2,0:64:2,:] / 255.)
    # combine all 4 sampled images into one image
    # index 0 of inner_imgs is image sample number
    #print(inner_imgs[0].shape)
    return np.concatenate((
                 np.concatenate((inner_imgs[0],inner_imgs[1]),axis=1),
                 np.concatenate((inner_imgs[2],inner_imgs[3]),axis=1)), axis=2)

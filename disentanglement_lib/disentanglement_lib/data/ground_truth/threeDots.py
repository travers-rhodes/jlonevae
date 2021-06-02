# coding=utf-8
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

from disentanglement_lib.data.ground_truth import util, ground_truth_data
import numpy as np
import cv2
import os
from pathlib import Path

THREE_DOTS_CACHE_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "threeDots")
THREE_DOTS_CACHE_ZIP = os.path.join(THREE_DOTS_CACHE_PATH,
    "trainingCache.npz")

class ThreeDotsTrainingCache(ground_truth_data.GroundTruthData):
  """Three Dots Dataset Cache for Training
  
     This class generates a LARGE random batch of data
     and reads back random samples when sample_observations
     is called.

     This class should only be used for training, since it doesn't actually
     implement sample_observations_from_factors

     However if you want to train quickly, use this class, since the
     pre-computed images are faster than using ThreeDots class to calculate/draw
     the images as needed.

     Our fine-resolution state space has 64^6 datapoints, so is too large to
     practically/nicely store in memory. Thus, we use a large random sample
     instead.
  """
  def __init__(self, seed=0, latent_factor_granularity=64):
    self.random_state = np.random.RandomState(seed)
    self.threeDotsGenerator = ThreeDots(latent_factor_granularity)
    # if the training cache doesn't exist, create one!
    if not os.path.exists(THREE_DOTS_CACHE_ZIP):
      print("No three dots cache zip found...creating one")
      Path(THREE_DOTS_CACHE_PATH).mkdir(parents=True, exist_ok=True)
      numTotalSamples = int(500000) # comparable size to dsprites
      factors, imgs = self.threeDotsGenerator.sample(numTotalSamples,
        self.random_state)
      np.savez_compressed(THREE_DOTS_CACHE_ZIP, factors=factors, imgs=imgs)
    data = np.load(THREE_DOTS_CACHE_ZIP)
    self.fullDataset = data["imgs"]
    self.totalCacheSize = self.fullDataset.shape[0]
    print("Loaded dataset of threeDots data with shape", self.fullDataset.shape)
  
  @property
  def observation_shape(self):
    return [64, 64, 1]

  def sample_observations(self, num, random_state):
    # we have to reshape or else num==1 condenses to int, not array
    indexArray = self.random_state.choice(self.totalCacheSize, (num,))
    return self.fullDataset[indexArray]

class ThreeDots(ground_truth_data.GroundTruthData):
  """Three Dots dataset.

  The data set was originally introduced in "Bias and Generalization in 
  Deep Generative Models: An Empirical Study" and can be downloaded from
  https://github.com/ermongroup/BiasAndGeneralization/tree/master/DotsAndPie/dataset

  This class presents a modification/simplification of that dataset to allow dot
  overlap and to be in grayscale.

  The dataset is stored implicitly as a procedure to generate images, not
  explicitly as a numpy array.

  The generative function is _not_ injective. The same image can be constructed
  from multiple inputs. However, the generative function is a well-defined
  function, and is therefore useful for disentanglement studies.

  The ground-truth factors of variation are (in the default setting):
  0 - position x (dot 0) 
  1 - position y (dot 0)
  2 - position x (dot 1) 
  3 - position y (dot 1) 
  4 - position x (dot 2)
  5 - position y (dot 2)
  They all take in 64 different possible integer values 0 - 64
  Horizontal is x, vertical is y, the (0,0) value is top left
  """

  def __init__(self, latent_factor_granularity=64):
    self.latent_factor_indices = list(range(6))
    self.data_shape = [64, 64, 1]
    self.latent_factor_granularity = latent_factor_granularity
    self.factor_sizes = [latent_factor_granularity] * 6
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
  
  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    num = factors.shape[0]
    sample = np.zeros(([num] + self.data_shape), dtype=np.float32)
    radius = 0.08
    for sampind in range(num):
      # divide by granularity-1 to scale 0-1 (fencepost problem)
      dot_centers = factors[sampind].reshape(3,2)/(self.latent_factor_granularity-1)
      # last index is the single grayscale channel
      sample[sampind,:,:,0] = draw_dots_to_image_array(0.08, dot_centers)
    return sample 

# create an image with dots of the given radius (range 0 to 1)
# placed at the given locations (shape: (N, 2)) in the range 0 to 1
# the output image is shape (64, 64, 1)
# with a black dot on a white background
# and some gray colors on the border pixels
# (the gray color is created by making a purely black circle on a 10x larger image
# and then resizing the image smaller and averaging nearby pixels
def draw_dots_to_image_array(radius, dot_centers):
    num_object = dot_centers.shape[0]
    im_side_len = 64
    # in order to get nicely averaged/grayscale shape boundary
    # we draw on a larger canvas and scale down
    # Create a black image
    upscale_image_factor = 10
    precision_scale = 10
    black = 0 
    up_img = np.ones((im_side_len * upscale_image_factor,im_side_len *
      upscale_image_factor,1))
    up_radius = int(im_side_len * radius * upscale_image_factor * 2**precision_scale)
    for i in range(num_object):
        up_center = np.floor(im_side_len * dot_centers[i] * upscale_image_factor * 2**precision_scale).astype(int)
        # negative thickness means filled circle
        circle = cv2.circle(up_img, tuple(up_center), up_radius, color=black, thickness = -1, shift=precision_scale)
    arr = cv2.resize(up_img, (im_side_len,im_side_len), interpolation=cv2.INTER_AREA)
    return arr

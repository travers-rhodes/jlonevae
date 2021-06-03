#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import sklearn.feature_extraction
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description='Sample patches from figures')
parser.add_argument('--experimentName', default="naturalImages",
                    help='the folder name to expect/use for the data/model/results')
args = parser.parse_args()
experiment_name = args.experimentName

# The nature paper begins with ten 512x512 images
# from which 16x16 pixel image patches should be sampled
large_images = sio.loadmat("data/%s/IMAGES.mat" % experiment_name)['IMAGES']

# turns out those images are not in the scale from 0 to 1, so scale
# them accordingly first
minValue = np.min(large_images)
maxValue = np.max(large_images)

newmin = 0.0
newmax = 1.0
large_images = (large_images-minValue) / (maxValue - minValue) * (newmax - newmin) + newmin


def sample_patches(large_images, num_patches_per_image = 10000):
    num_images = large_images.shape[-1]
    num_patches = num_patches_per_image * num_images
    patches = np.zeros((num_patches, 1, 16, 16))
    for i in range(num_images):
        patches[(i * num_patches_per_image):((i+1) * num_patches_per_image),0,:,:] = sklearn.feature_extraction.image.extract_patches_2d(large_images[:,:,i], (16,16), max_patches = num_patches_per_image)
    return patches

train_patches = sample_patches(large_images)
test_patches = sample_patches(large_images, num_patches_per_image = 1000)

save_folder = "data/%s" % experiment_name

Path(save_folder + "/train").mkdir(parents=True, exist_ok=True)
np.savez_compressed(save_folder + "/train/data.npz", images=train_patches)

Path(save_folder + "/test").mkdir(parents=True, exist_ok=True)
np.savez_compressed(save_folder + "/test/data.npz", images=test_patches)









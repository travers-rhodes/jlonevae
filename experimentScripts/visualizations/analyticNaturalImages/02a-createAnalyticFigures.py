#!/usr/bin/env python3
import PIL.Image
import numpy as np
from pathlib import Path
import os.path

import argparse
parser = argparse.ArgumentParser(description='Create figures given a dataset')
parser.add_argument('--experimentName', required=True,
                    help='the folder name to expect/use for the model')
parser.add_argument('--latentDim', default=[5 ,10, 25], type=int, nargs="*",
        help='number of latent dimensions to use')
parser.add_argument('--modelType', default=["PCA", "ICA"], type=str, nargs="*",
        help='number of latent dimensions to use')
args = parser.parse_args()

experimentName = args.experimentName
latentDimVals = args.latentDim
modelTypeVals = args.modelType


model_folder = "analyticModels/%s" % experimentName
save_folder = "images/analyticModels/%s" % experimentName
Path(save_folder).mkdir(parents=True, exist_ok=True)

contrast_range = 255/2

for modelType in modelTypeVals:
    for latent_dim in latentDimVals:
        modelData = np.load(os.path.join(model_folder,"%s_lat%d.npz" %  (modelType, latent_dim)))
        latent_components = modelData["latent_components"]
        contrast = contrast_range / (np.max(np.abs(latent_components)))

        for i in range(latent_components.shape[1]):
            pixvals = latent_components[:,i].reshape(16,16)
            #print(np.max(pixvals),np.min(pixvals))
            pixscaled = pixvals*contrast + 255/2
            pic = PIL.Image.fromarray(pixscaled)
            # https://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg
            pic = pic.convert("L")
            pic.save(os.path.join(save_folder, "%s_lat%d_image%d.png" % (modelType, latent_dim, i)))



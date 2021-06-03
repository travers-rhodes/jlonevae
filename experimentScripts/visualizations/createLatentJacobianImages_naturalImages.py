#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
from jlonevae_lib.architecture.load_model import load_model
import jlonevae_lib.architecture.vae_jacobian as vj
import torch
import PIL.Image
from pathlib import Path
import os.path
import glob

import argparse
parser = argparse.ArgumentParser(description='Create figures given a dataset')
parser.add_argument('--datasetName', 
                    help='the folder name to expect to contain the data IMAGES.mat file')
parser.add_argument('--experimentName', 
                    help='the folder name to expect for the trained model')
parser.add_argument('--vaeShape', default="tiny16",
                    help='the folder name to expect/use for the data/model/results')
parser.add_argument('--lossFunction', default="bernoulli",
                    help='the loss function to use for the results, either (unit) gaussian or bernoulli).')
parser.add_argument('--runId', default="",
                    help='runId string to add to run name')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate to use')
parser.add_argument('--batchNumber', default=100000, type=int,
        help='which saved model to display')
parser.add_argument('--batchSize', default=128, type=int,
        help='number of images per batch during training')
parser.add_argument('--beta', default=[1.000], type=float, nargs="*",
        help='one or more beta values to use in training')
parser.add_argument('--gamma', default=[0.000], type=float, nargs="*",
        help='one or more gamma values to use in training')
parser.add_argument('--latentDim', default=[10], type=int, nargs="*",
        help='number of latent dimensions to use')
parser.add_argument('--annealingBatches', default=0, type=int,
        help='number of batches for which to perform linear beta annealing (starting at 0, ramping up to set beta)')
args = parser.parse_args()

experimentName = args.experimentName
datasetName = args.datasetName
vaeShape= args.vaeShape
lossfunc = args.lossFunction
lr = args.lr
runId = args.runId
batchNumber = args.batchNumber
batch_size = args.batchSize
betaVals = args.beta
gammaVals = args.gamma
latentDimVals = args.latentDim
annealingBatches = args.annealingBatches

def getModelName(beta, gamma, latent_dim): 
  beta_string = ("%.4f"%beta).replace(".","_")
  ica_factor_name = ("%0.4f"%gamma).replace(".","_")
  pca_factor = None
  scaling = "lone"
  runId = "_run0"
  batch_normalize = False
  useAdam = True
  useDoublePrecision = False
  lr_string = ("%0.4f"%lr).replace(".","_")
  model_name = "%s_%s_beta%s%s_ica%s_lat%d_batch%d_lr%s%s%s%s%s%s" % (
    vaeShape,
    scaling, 
    beta_string, 
    "_pca%s" % (("%0.4f"%pca_factor).replace(".","_")) if pca_factor is not None else "", 
    ica_factor_name, 
    latent_dim, 
    batch_size, 
    lr_string,
    "_norm" if batch_normalize else "", 
    "_d" if useDoublePrecision else "",
    "" if useAdam else "_sgd",
    "_anneal%d" % annealingBatches if annealingBatches != 0 else "", 
    runId)
  return(model_name)

device="cuda"

def get_model_and_enc(latent_dim, beta, ica_factor, data):
  modelname = getModelName(beta, ica_factor, latent_dim)
  search_path = "trainedModels/%s/%s/*/cache_batch_no%d" % (experimentName, modelname,batchNumber)
  print(search_path)
  matching_model_paths = glob.glob(search_path)
  model_path = matching_model_paths[0]
  model = load_model(model_path,device);
  recon, encoding, logvar, noisy_encoding = model(torch.tensor(data).to(device).float());
  return model, recon, encoding, logvar, noisy_encoding

save_folder = "experimentScripts/visualizations/latentJacobianImages/%s" % experimentName
Path(save_folder).mkdir(parents=True, exist_ok=True)

large_images = sio.loadmat("data/%s/IMAGES.mat" % datasetName)['IMAGES']
# turns out those images are not in the scale from 0 to 1, so scale
# them accordingly first
minValue = np.min(large_images)
maxValue = np.max(large_images)
newmin = 0.0
newmax = 1.0
large_images = (large_images-minValue) / (maxValue - minValue) * (newmax - newmin) + newmin

def getImageCrop(imageId, xstart, ystart):
    # return shape is (1,1,16,16)---the first index indicates "batchsize", the second channel
    return np.array([[large_images[xstart:(xstart+16), ystart:(ystart+16), imageId]]]).copy()

imageId = 0
firstxstart = 40
ystart = 100

# scaling jacobian images for visibility
contrast_range = 255/2



for beta in betaVals: 
  for ica_factor in gammaVals:
    for latent_dim in latentDimVals:
      # for each model (each beta/ica/latent_dim combination) pick a single ordering of latent indices
      # (purely for aesthetic display purposes)
      # if you have multiple beta/ica factors/latent_dims, you'll resave the same source image multiple times,
      # but this ordering of for loops allows us to easily remember _for each model_
      # the aesthetically pleasing jacobian index ordering
      top_jac_inds = None
      contrast = None
      for xstart in range(firstxstart,firstxstart+10):
            data = getImageCrop(imageId, xstart, ystart)
            # save the relevant image crop so we can display it in LaTeX (fine to overwrite this multiple times as needed
            pic = PIL.Image.fromarray(data[0,0]*255)
            # https://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg
            pic = pic.convert("L")
            imagename = os.path.join(save_folder, "naturalImageCrop_image%d_x%d_y%d.png" % (imageId, xstart, ystart))
            pic.save(imagename)
 
            model, recon, encoding, logvar, noisy_encoding = get_model_and_enc(latent_dim, beta, ica_factor, data)

            jacs = vj.compute_generator_jacobian_optimized(model, encoding, device=device).detach().cpu().numpy()

            # pick ordering for this model's latents based on the first image (associated with the first xstart)
            if top_jac_inds is None:
                # Sort the latent variables by the norm for aesthetics
                activities = [np.sum(np.square(jac)) for jac in jacs]
                top_jac_inds = np.flip(np.argsort(activities))

            # scale contrast across all jacobian values
            # don't cache the contrast from the first image as well
            contrast = contrast_range / (np.max(np.abs(jacs)))

            for i, ind in enumerate(top_jac_inds):
              pixvals = jacs[ind,0,0] # switch from (1,1,16,16) back to (16,16)
              #print(np.max(pixvals),np.min(pixvals))
              pixscaled = pixvals*contrast + 255/2
              pic = PIL.Image.fromarray(pixscaled)
              # https://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg
              pic = pic.convert("L")
              betastring = ("%.3f"%beta).replace(".","_")
              ica_factor_name = ("%0.3f"%ica_factor).replace(".","_")
              modelinfo = "beta%s_ica%s_lat%d" % (betastring, ica_factor_name, latent_dim)
              imagename = os.path.join(save_folder, "%s_im%d_latind%d_x%d_y%d.png" % (modelinfo, imageId, i, xstart, ystart))
              print(imagename)
              pic.save(imagename)

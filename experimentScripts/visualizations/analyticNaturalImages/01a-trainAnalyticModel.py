#!/usr/bin/env python3
import numpy as np
import datetime
import glob
import sys
sys.path.append("../../..") # include base dir

from jlonevae_lib.utils.pytorch_npz_dataset import PytorchNpzDataset 
from sklearn.decomposition import FastICA, PCA 
import torch.utils
from pathlib import Path
import os.path

import argparse
parser = argparse.ArgumentParser(description='Create figures given a dataset')
parser.add_argument('--datasetName',  required=True,
                    help='the folder name to expect/use for the data')
parser.add_argument('--experimentName', required=True,
                    help='the folder name to expect/use for the model')
parser.add_argument('--latentDim', default=[10], type=int, nargs="*",
        help='number of latent dimensions to use')
parser.add_argument('--modelType', default=["PCA"], type=str, nargs="*",
        help='number of latent dimensions to use')
parser.add_argument('--batchSize', default=10000, type=int,
        help='number of images to use to train ICA')
args = parser.parse_args()

dataset_name = args.datasetName
experimentName = args.experimentName
latentDimVals = args.latentDim
modelTypeVals = args.modelType
batch_size = args.batchSize
device = "cpu"
useDoublePrecision = False

train_data_paths = glob.glob("../../../data/%s/train/*.npz" % dataset_name)
print(train_data_paths)
dataset = PytorchNpzDataset(train_data_paths, useDoublePrecision)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
dataiter = iter(data_loader)
data = dataiter.next()
data = data.detach().cpu().numpy()
assert batch_size == data.shape[0], "we weren't able to sample the requested num images"
data = data.reshape(batch_size,-1)

# loop over several hyperparameters and plot results
for latent_dim in latentDimVals:
    for modelType in modelTypeVals:
        if modelType == "ICA":
            transformer = FastICA(n_components=latent_dim,
                random_state=0)
            transformer.fit(data)
            latent_components = transformer.mixing_
        elif modelType == "PCA":
            transformer = PCA(n_components=latent_dim,
                random_state=0)
            transformer.fit(data)
            latent_components = transformer.components_.T

        X = transformer.transform(data)
        imgrecon = transformer.inverse_transform(X)
        print(np.max(imgrecon), np.min(imgrecon))
        imgrecon_clip = np.clip(imgrecon, 0, 1)
        # binary cross entropy loss
        MSE_loss = torch.nn.functional.mse_loss(torch.tensor(imgrecon_clip, dtype=torch.float), torch.tensor(data, dtype=torch.float), reduction='sum')/batch_size
        print(MSE_loss.item())
        NLL_loss = torch.nn.functional.binary_cross_entropy(torch.tensor(imgrecon_clip, dtype=torch.float), torch.tensor(data, dtype=torch.float), reduction='sum')/batch_size
        print(NLL_loss.item())
        save_folder = "analyticModels/%s" % experimentName
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(save_folder,"%s_lat%d.npz" %  (modelType, latent_dim)), 
                MSE_loss = MSE_loss.item(), 
                NLL_loss = NLL_loss.item(),
                latent_components = latent_components) 
        print(MSE_loss.item())

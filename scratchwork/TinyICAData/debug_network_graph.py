#!/usr/bin/env python3
import torch
import torch.utils.tensorboard
import os
import sys
sys.path.append("../..") # include base dir
import numpy as np
from jlonevae_lib.architecture.save_model import save_conv_vae
import jlonevae_lib.architecture.vae_jacobian as vj
from jlonevae_lib.train.loss_function import vae_loss_function 
from jlonevae_lib.architecture.vae import ConvVAE

class LossComputingVaeModule(ConvVAE):
  def __init__(self, **args):
    super(LossComputingVaeModule, self).__init__(**args)

  def forward(self, x):
    recon_batch, mu, logvar, noisy_mu = super(LossComputingVaeModule, self).forward(x)
    loss, NegLogLikelihood, KLD, mu_error, logvar_error = vae_loss_function(recon_batch, 
                                      x, mu, logvar, 1, "gaussian")
    emb_ICA_loss = vj.embedding_jacobian_loss_function(self, data, "cpu")
    loss += emb_ICA_loss
    return loss


# define our convolutional network parameters
# see jlonevae_lib.architecture.vae.ConvVAE for how these parameters define the VAE network.
emb_conv_layers_channels = []
emb_conv_layers_strides = []
emb_conv_layers_kernel_sizes = []
emb_fc_layers_num_features = [10,10,10]
gen_conv_layers_channels = []
gen_conv_layers_kernel_sizes = []
gen_fc_layers_num_features = [2]
gen_first_im_side_len=1
gen_conv_layers_strides = []
im_channels = 2
im_side_len = 1
latent_dim=3
device="cpu"
model = LossComputingVaeModule(
         latent_dim = latent_dim,
         im_side_len = im_side_len,
         im_channels = im_channels,
         emb_conv_layers_channels = emb_conv_layers_channels,
         emb_conv_layers_strides = emb_conv_layers_strides,
         emb_conv_layers_kernel_sizes = emb_conv_layers_kernel_sizes,
         emb_fc_layers_num_features = emb_fc_layers_num_features,
         gen_fc_layers_num_features = gen_fc_layers_num_features,
         gen_first_im_side_len = gen_first_im_side_len,
         gen_conv_layers_channels = gen_conv_layers_channels,
         gen_conv_layers_strides = gen_conv_layers_strides,
         gen_conv_layers_kernel_sizes = gen_conv_layers_kernel_sizes,
         final_activation="linear"
        ).to(device)
writer = torch.utils.tensorboard.SummaryWriter(log_dir="debug_architecture")
datafile = "icaData.npz"
dataset = np.load(datafile)["dataset"]
data = torch.tensor(dataset[:10].astype(np.float32))

writer.add_graph(model, data)
writer.close()

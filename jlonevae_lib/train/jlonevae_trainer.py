# if you want to train without using the disentanglement_lib infrastructure
# you can train using this file.
# This file's train method is very, very similar to 
# jlonevae_lib/train/train_jlonevae_models.py's train method
# This is just an object-oriented version of that more script-based version
# this file's train method also has a "save model every" and expects the caller
# to do the parameter annealing already.
# If you're using disentanglement_lib you might as well use that file.
# If you're using a custom dataloader it probably makes sense to use this file.

import torch
import torch.utils.tensorboard
import os
import numpy as np
from jlonevae_lib.architecture.save_model import save_conv_vae
import jlonevae_lib.architecture.vae_jacobian as vj
from jlonevae_lib.train.loss_function import vae_loss_function 

class JLOneVAETrainer(object):
    def __init__(self, model, data_loader, beta, gamma, device, 
        log_dir, lr, annealingBatches, record_loss_every=100):
      self.model = model
      self.data_loader = data_loader
      self.optimizer= torch.optim.Adam(self.model.parameters(), lr=lr)
      self.device = device
      self.log_dir = log_dir
      self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.log_dir)
      self.num_batches_seen = 0
      self.annealingBatches = annealingBatches
      self.record_loss_every = record_loss_every

    def train(self):
      # set model to "training mode"
      self.model.train()

      # for each chunk of data in an epoch
      for data in self.data_loader:
        # do annealing and store annealed value to tmp value
        if self.num_batches_seen < self.annealingBatches:
          tmp_beta = self.beta * self.num_batches_seen / self.annealingBatches
          tmp_gamma = self.gamma * self.num_batches_seen / self.annealingBatches
        else:
          tmp_beta = self.beta
          tmp_gamma = self.gamma

        # move data to device, initialize optimizer, model data, compute loss,
        # and perform one optimizer step
        data = data.to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar, noisy_mu = self.model(data)
        loss, NegLogLikelihood, KLD, mu_error, logvar_error,
        noiselessNegLogLikelihood = vae_loss_function(recon_batch, data, mu,
            logvar, tmp_beta)

        # short-circuit calc if gamma is 0 (no JL1-VAE loss)
        if tmp_gamma == 0:
            ICA_loss = torch.tensor(0)
        else:
            ICA_loss = vj.jacobian_loss_function(self.model, noisy_mu, logvar, self.device, scaling = self.scaling)
        loss += tmp_gamma * ICA_loss
        loss.backward()
        self.optimizer.step()

        # log to tensorboard
        self.num_batches_seen += 1
        if self.num_batches_seen % self.record_loss_every == 0:
            self.writer.add_scalar("ICALoss/train", ICA_loss.item(), self.num_batches_seen) 

            self.writer.add_scalar("ELBO/train", loss.item(), self.num_batches_seen) 
            self.writer.add_scalar("KLD/train", KLD.item(), self.num_batches_seen) 
            self.writer.add_scalar("MuDiv/train", mu_error.item(), self.num_batches_seen) 
            self.writer.add_scalar("VarDiv/train", logvar_error.item(), self.num_batches_seen) 
            self.writer.add_scalar("NLL/train", NegLogLikelihood.item(), self.num_batches_seen) 
            self.writer.add_scalar("beta", tmp_beta, self.num_batches_seen) 
            self.writer.add_scalar("gamma", tmp_gamma, self.num_batches_seen) 

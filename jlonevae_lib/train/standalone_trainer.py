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
from jlonevae_lib.train.loss_function import vae_loss_function 
import jlonevae_lib.train.vae_jacobian as vj

#https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class StandaloneTrainer(object):
    def __init__(self, model, dataset, batch_size=64, device="cpu", log_dir="./runs/", lr = 0.001):
        self.model = model
        self.data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        #https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations
        self.data_iterator = iter(cycle(self.data_loader))
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.log_dir = log_dir
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.log_dir)
        self.num_batches_seen = 0

    def train(self, beta=1.0, gamma=0.0, record_loss_every=100, save_model_every=10000):
        # set model to "training mode"
        self.model.train()

        # get chunk of data 
        data =  next(self.data_iterator)

        # move data to device, initialize optimizer, model data, compute loss,
        # and perform one optimizer step
        data = data.to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar, noisy_mu = self.model(data)
        loss, NegLogLikelihood, KLD, mu_error, logvar_error, noiselessNegLogLikelihood = vae_loss_function(recon_batch, data, mu, logvar, beta)

        # short-circuit calc if gamma is 0 (no JL1-VAE loss)
        if gamma == 0:
            ICA_loss = torch.tensor(0)
        else:
            ICA_loss = vj.jacobian_loss_function(self.model, noisy_mu, logvar, self.device, scaling = self.scaling)
        loss += gamma * ICA_loss
        loss.backward()
        self.optimizer.step()

        # log to tensorboard
        self.num_batches_seen += 1
        if self.num_batches_seen % record_loss_every == 0:
            self.writer.add_scalar("ICALoss/train", ICA_loss.item(), self.num_batches_seen) 

            self.writer.add_scalar("ELBO/train", loss.item(), self.num_batches_seen) 
            self.writer.add_scalar("KLD/train", KLD.item(), self.num_batches_seen) 
            self.writer.add_scalar("MuDiv/train", mu_error.item(), self.num_batches_seen) 
            self.writer.add_scalar("VarDiv/train", logvar_error.item(), self.num_batches_seen) 
            self.writer.add_scalar("NLL/train", NegLogLikelihood.item(), self.num_batches_seen) 
            self.writer.add_scalar("beta", beta, self.num_batches_seen) 
            self.writer.add_scalar("gamma", gamma, self.num_batches_seen) 
        # save a cached version of this model
        if self.num_batches_seen % save_model_every == 0:
            save_conv_vae(self.model, os.path.join(self.log_dir, "cache_batch_no%d" % self.num_batches_seen))

__doc__ = """
This code was taken and modified
from https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit
"""

# Note: _we_ don't use tensorflow, but we call data-loading code that does
# trying so hard to mute tensorflow warnings...
# https://stackoverflow.com/questions/57539273/disable-tensorflow-logging-completely
import tensorflow as tf
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import argparse
import torch
import torch.utils.data
import torch.utils.tensorboard
from torch import nn, optim

from jlonevae_lib.architecture.vae import ConvVAE
from jlonevae_lib.architecture.save_model import save_conv_vae
import jlonevae_lib.train.vae_jacobian as vj
from jlonevae_lib.train.loss_function import vae_loss_function
import jlonevae_lib.utils.utils_pytorch as pyu

from pathlib import Path
import pickle
import os

import datetime

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=1,
                    help='the beta value to train with')
parser.add_argument('--gamma', type=float, default=1,
                    help='the gamma value (LIL-VAE hyperparameter) to train with')
parser.add_argument('--lr', type=float, default=0.001,
                    help='the learning rate to use') 
parser.add_argument('--latent-dim', type=int, default=10,
                    help='the number of latent dimensions to train with')
parser.add_argument('--annealingBatches', default=0, type=int,
        help='number of batches for which to perform linear annealing (starting at 0, ramping up to beta and gamma values)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
# iterator_len defines length of an epoch
# "epoch" is an illusion for disentanglement_lib's pytorch implementation
# (it's always just random).
# however, we want to be able to nicely print out our final answers,
# so we make this a nice round number
batch_size = args.batch_size
num_batches_per_epoch = 5000
# we want a total of 300000 training steps (batches). Where each batch size is 64.
# That/s 300000 / 5000 = 60 epochs
train_loader = pyu.get_loader(batch_size=batch_size, iterator_len=batch_size * num_batches_per_epoch, **kwargs)

data_sample = iter(train_loader).next()
im_channels = data_sample.shape[1]


class RepresentationExtractor(nn.Module):
    VALID_MODES = ['mean', 'sample']

    def __init__(self, model, mode='mean'):
        super(RepresentationExtractor, self).__init__()
        assert mode in self.VALID_MODES, f'`mode` must be one of {self.VALID_MODES}'
        self.wrapped_model = model
        self.mode = mode

    def forward(self, x):
        mu, logvar = self.wrapped_model.encode(x)
        if self.mode == 'mean':
            return mu
        elif self.mode == 'sample':
            return self.reparameterize(mu, logvar)
        else:
            raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std




# Changes from Locatello:
# we have sigmoid activation for reconstruction
# we have consistent 4x4 kernels (they have two 2x2's in their code)
vaeShape = "defaultConv"
emb_conv_layers_channels = [32,32,64,64]
emb_conv_layers_strides = [2,2,2,2]
emb_conv_layers_kernel_sizes = [4,4,4,4]
emb_fc_layers_num_features = [256]
gen_conv_layers_channels = [64,32,32,im_channels]
gen_conv_layers_kernel_sizes = [4,4,4,4]
gen_fc_layers_num_features = [256,1024]
gen_first_im_side_len = 4
gen_conv_layers_strides = [2,2,2,2]
#im_channels = # set above by taking sample from dataset
im_side_len = 64 
latent_dim = args.latent_dim
lr = args.lr
annealingBatches = args.annealingBatches

# name to use for this model
def getModelName(beta, gamma, latent_dim):
  beta_string = ("%.4f"%beta).replace(".","_")
  ica_factor_name = ("%0.4f"%gamma).replace(".","_")
  pca_factor = None
  scaling = "lone"
  batch_normalize = False
  useAdam = True
  useDoublePrecision = False
  lr_string = ("%0.4f"%lr).replace(".","_")
  model_name = "%s_%s_beta%s%s_ica%s_lat%d_batch%d_lr%s%s%s%s%s" % (
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
    "_anneal%d" % annealingBatches if annealingBatches != 0 else "")
  return(model_name)

model = ConvVAE(
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
  ).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


record_loss_every = args.log_interval
beta=args.beta
gamma=args.gamma
experimentName = getModelName(beta, gamma, latent_dim)
logdir = "logs/%s/%s" % (experimentName, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
modelDir = "trainedModels/%s/%s/representation" % (experimentName, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
Path(logdir).mkdir(parents=True, exist_ok=True)
Path(modelDir).mkdir(parents=True, exist_ok=True)
# empty "results" folder here required for evaluation.py
Path(modelDir+"/results").mkdir(parents=True, exist_ok=True)
writer = torch.utils.tensorboard.SummaryWriter(log_dir=logdir)

def train(epoch, num_batches_seen):
  # set model to "training mode"
  model.train()

  # for each chunk of data 
  for batch_idx, data in enumerate(train_loader):
    # do annealing and store annealed value to tmp value
    if num_batches_seen < annealingBatches:
      tmp_beta = beta * num_batches_seen / annealingBatches
      tmp_gamma = gamma * num_batches_seen / annealingBatches
    else:
      tmp_beta = beta
      tmp_gamma = gamma

    # move data to device, initialize optimizer, model data, compute loss,
    # and perform one optimizer step
    data = data.to(device).float()
    optimizer.zero_grad()
    recon_batch, mu, logvar, noisy_mu = model(data)
    loss, NLL, KLD, mu_err, logvar_err = vae_loss_function(recon_batch, data, mu, logvar, tmp_beta)

    # short-circuit calc if gamma is 0 (no JL1-VAE loss)
    if tmp_gamma == 0: 
      ICA_loss = torch.tensor(0)
    else:
      ICA_loss = vj.jacobian_loss_function(model, noisy_mu, logvar, device,
          scaling = "lone")
    loss += tmp_gamma * ICA_loss
    loss.backward()
    optimizer.step()

    # log to tensorboard
    num_batches_seen += 1
    if num_batches_seen % record_loss_every == 0:
      writer.add_scalar("ICALoss/train", ICA_loss.item(), num_batches_seen) 

      writer.add_scalar("ELBO/train", loss.item(), num_batches_seen) 
      writer.add_scalar("KLD/train", KLD.item(), num_batches_seen) 
      writer.add_scalar("MuDiv/train", mu_err.item(), num_batches_seen) 
      writer.add_scalar("VarDiv/train", logvar_err.item(), num_batches_seen) 
      writer.add_scalar("NLL/train", NLL.item(), num_batches_seen) 
      writer.add_scalar("beta", tmp_beta, num_batches_seen) 
      writer.add_scalar("gamma", tmp_gamma, num_batches_seen) 

  print('====> Epoch: {}'.format(epoch))
  return num_batches_seen

if __name__ == '__main__':
    print("Starting training. Logging to %s" % logdir)
    num_batches_seen = 0
    for epoch in range(1, args.epochs + 1):
        num_batches_seen = train(epoch, num_batches_seen)
    full_save_model_dir = modelDir + "/cache_batch_no%d" % num_batches_seen
    print("Saving full model info to %s" % full_save_model_dir)
    save_conv_vae(model, full_save_model_dir)
    print("Saving trained model to %s" % modelDir)
    pyu.export_model(RepresentationExtractor(model),
            path=modelDir + "/pytorch_model.pt",
            input_shape=(1, im_channels, 64, 64))

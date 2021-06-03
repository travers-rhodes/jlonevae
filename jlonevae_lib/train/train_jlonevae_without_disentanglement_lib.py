# If you want to train a jlonevae model directly
# without using disentanglement_lib, you can do so by using this file.
# For our paper, this is the implementation we use for the naturalImage results
# for which we do not have ground-truth factors of variation.
import datetime
import torch
import glob
import math
import os

from jlonevae_lib.architecture.vae import ConvVAE
from jlonevae_lib.architecture.save_model import save_conv_vae
from jlonevae_lib.train.jlonevae_trainer import JLOneVAETrainer
from jlonevae_lib.utils.pytorch_npz_dataset import PytorchNpzDataset 

import argparse
parser = argparse.ArgumentParser(description='Create figures given a dataset')
parser.add_argument('--experimentName', 
                    help='the folder name to expect/use for the data/model/results')
parser.add_argument('--vaeShape', default="tiny15",
                    help='the folder name to expect/use for the data/model/results')
parser.add_argument('--lossFunction', default="bernoulli",
                    help='the loss function to use for the results, either (unit) gaussian or bernoulli).')
parser.add_argument('--useDouble', default=0, type=int,
                    help='whether to use double precision for model')
parser.add_argument('--runId', default="",
                    help='runId string to add to run name')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate to use')
parser.add_argument('--numBatches', default=100000, type=int,
        help='number of batches to train for')
parser.add_argument('--recordLossEvery', default=100, type=int,
        help='how frequently to record loss to tensorboard (units are number of batches)')
parser.add_argument('--batchSize', default=128, type=int,
        help='number of images per batch during training')
parser.add_argument('--beta', default=[1.000], type=float, nargs="*",
        help='one or more beta values to use in training')
parser.add_argument('--gamma', default=[0.000], type=float, nargs="*",
        help='one or more gamma values to use in training')
parser.add_argument('--latentDim', default=[10], type=int, nargs="*",
        help='number of latent dimensions to use')
parser.add_argument('--annealingBatches', default=0, type=int,
        help='number of batches for which to perform linear annealing (starting at 0, ramping up to set value)')
args = parser.parse_args()

dataset_name = args.experimentName
vaeShape= args.vaeShape
lossfunc = args.lossFunction
useDoublePrecision = bool(args.useDouble)
lr = args.lr
runId = args.runId
num_batches = args.numBatches
batch_size = args.batchSize
betaVals = args.beta
gammaVals = args.gamma
latentDimVals = args.latentDim
# ...how often we log loss to tensorboard
record_loss_every = args.recordLossEvery
annealingBatches = args.annealingBatches


train_data_paths = glob.glob("data/%s/train/*.npz" % dataset_name)
# since we aren't using disentanglement_lib's ground-truth data
# infrastrucutre, we just create a simple "read in a numpy zip file
# class and use that
dataset = PytorchNpzDataset(train_data_paths, useDoublePrecision)

# ...and how of often we save the current model
save_model_every=300000
# (units above are in "number of batches"---not number of epochs.

# Train on the CUDA device for speed
device="cuda" if torch.cuda.is_available() else "cpu"

# define our convolutional network parameters
# see jlonevae_lib.architecture.vae.ConvVAE for how these parameters define the VAE network.
if vaeShape == "tiny16":
    emb_conv_layers_channels = [64,128]
    emb_conv_layers_strides = [2,2]
    emb_conv_layers_kernel_sizes = [3,3]
    emb_fc_layers_num_features = [128]
    gen_conv_layers_channels = [64,32,1]
    gen_conv_layers_kernel_sizes = [3,3,3]
    gen_fc_layers_num_features = [128, 4*4*64]
    gen_first_im_side_len=4
    gen_conv_layers_strides = [2,2,1]
    im_channels = 1
    im_side_len = 16

# loop over hyperparameters and train models
for run_beta in betaVals:
  for ica_factor in gammaVals:
    for latent_dim in latentDimVals:
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
         gen_conv_layers_kernel_sizes = gen_conv_layers_kernel_sizes
        ).to(device)

      if useDoublePrecision:
          model = model.double()

      beta_string = ("%.4f"%run_beta).replace(".","_")
      ica_factor_name = ("%0.4f"%ica_factor).replace(".","_")
      lr_string = ("%0.4f"%lr).replace(".","_")
      scaling = "lone"
      new_experiment_name = "%s_%s_beta%s_ica%s_lat%d_batch%d_lr%s%s%s%s/" % (
              vaeShape,
              scaling, 
              beta_string, 
              ica_factor_name, 
              latent_dim, 
              batch_size, 
              lr_string,
              "_d" if useDoublePrecision else "",
              "_anneal%d" % annealingBatches if annealingBatches != 0 else "", 
              runId)
      print("starting " + new_experiment_name)
      modelDir = "trainedModels/%s/%s/%s/" % (dataset_name, new_experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
      data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
      trainer = JLOneVAETrainer(model, data_loader,
          beta=run_beta, gamma=ica_factor,
          device=device, log_dir=modelDir, lr = lr,
          annealingBatches=annealingBatches)
      print(len(dataset))
      num_epochs = int(math.ceil(num_batches * batch_size/len(dataset)))
      print(num_epochs)
      for i in range(num_epochs):
        trainer.train()
      # save a cached version of this model
      save_conv_vae(trainer.model, os.path.join(trainer.log_dir,
        "cache_batch_no%d" % trainer.num_batches_seen))

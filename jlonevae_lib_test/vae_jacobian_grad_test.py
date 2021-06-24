import unittest
import numpy as np
import torch
from jlonevae_lib.architecture.vae import ConvVAE
from jlonevae_lib.architecture.vae_jacobian import *

if torch.cuda.is_available():
  available_devices = ["cuda", "cpu"]
else:
  available_devices = ["cpu"]

def get_linear_model(device):
  # define our convolutional network parameters
  # see jlonevae_lib.architecture.vae.ConvVAE for how these parameters define the VAE network.
  emb_conv_layers_channels = []
  emb_conv_layers_strides = []
  emb_conv_layers_kernel_sizes = []
  emb_fc_layers_num_features = []
  gen_conv_layers_channels = []
  gen_conv_layers_kernel_sizes = []
  gen_fc_layers_num_features = [2]
  gen_first_im_side_len=1
  gen_conv_layers_strides = []
  im_channels = 2
  im_side_len = 1
  latent_dim=3
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
         final_activation="linear"
        ).to(device)
  for name, p in model.named_parameters():
    if name == "gen_fcs.0.weight":
      p.data.copy_(torch.tensor(np.array([[1,2],[3,4],[5,6]], dtype=np.float32).T*2))
    elif name=="gen_fcs.0.bias":
      p.data.copy_(torch.tensor(np.array([10,20], dtype=np.float32)*2))
    elif name == "fcmu.weight":
      p.data.copy_(torch.tensor(np.array([[1,2],[3,4],[5,6]], dtype=np.float32)))
    elif name=="fcmu.bias":
      p.data.copy_(torch.tensor(np.array([10,20,30], dtype=np.float32)))

  return model
  

class TestEmbedding(unittest.TestCase):
    def test_embedding_jacobian_grad(self):
      batch_size = 20
      images = torch.tensor(np.zeros(shape=(batch_size,2,1,1), dtype=np.float32))
      for device in ["cuda"]:#available_devices:
        model = get_linear_model(device)
        loss = embedding_jacobian_loss_function(model, images, device)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
          print(name, param.grad)
        

if __name__ == '__main__':
    unittest.main()

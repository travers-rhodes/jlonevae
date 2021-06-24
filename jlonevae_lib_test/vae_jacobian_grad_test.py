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
  emb_fc_layers_num_features = [2]
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
    elif name == "emb_fcs.0.weight":
      p.data.copy_(torch.tensor(np.array([[1,0],[0,2]], dtype=np.float32)))
    elif name=="emb_fcs.0.bias":
      p.data.copy_(torch.tensor(np.array([0,0], dtype=np.float32)))
    elif name == "fcmu.weight":
      p.data.copy_(torch.tensor(np.array([[1,2],[3,4],[5,6]], dtype=np.float32)))
    elif name=="fcmu.bias":
      p.data.copy_(torch.tensor(np.array([10,20,30], dtype=np.float32)))

  return model
  

class TestEmbedding(unittest.TestCase):
    def ignore_test_embedding_jacobian_loss_grad(self):
      batch_size = 20
      images = torch.tensor(np.ones(shape=(batch_size,2,1,1), dtype=np.float32))
      device = "cuda"
      for jac_batch_size in [1,2,3,128]:#available_devices:
        model = get_linear_model(device)
        loss = embedding_jacobian_loss_function(model, images, device, jac_batch_size=jac_batch_size)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
          if "fcmu" in name or "emb" in name:
            print(name, param.grad, param.grad.dtype)
    
    def test_loss_plus_embedding_jacobian_loss_grad(self):
      batch_size = 20
      images = torch.tensor(np.ones(shape=(batch_size,2,1,1), dtype=np.float32))
      device = "cuda"
      for jac_batch_size in [1]:#available_devices:
        images = images.to(device)
        model = get_linear_model(device)
        # do it twice as a way to make sure that it's properly clearing grad before and after calc
        loss = embedding_jacobian_loss_function(model, images, device, jac_batch_size=jac_batch_size)
        ys = model.encode(images)[0][0][0]
        model.zero_grad()
        ys.backward(retain_graph=True)
        for name, param in model.named_parameters():
          if "fcmu.weight" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [[1., 2.],
                [0., 0.],
                [0., 0.]])
          if "fcmu.bias" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [1., 0., 0.])
          if "emb_fcs.0.weight" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [[1., 1.],
                [2., 2.]])
          if "emb_fcs.0.bias" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [1., 2.])
        model.zero_grad()
        loss.backward(retain_graph=True)
        for name, param in model.named_parameters():
          if "fcmu.weight" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [[1., 2.],
                [1., 2.],
                [1., 2.]], decimal=5)
          if "fcmu.bias" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [0., 0., 0.])
          if "emb_fcs.0.weight" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [[9., 9.],
                [12., 12.]], decimal=5)
          if "emb_fcs.0.bias" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [0., 0.])
        ys_plus_loss = ys + loss
        model.zero_grad()
        ys_plus_loss.backward()
        for name, param in model.named_parameters():
          if "fcmu.weight" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [[2., 4.],
                [1., 2.],
                [1., 2.]], decimal=5)
          if "fcmu.bias" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [1., 0., 0.])
          if "emb_fcs.0.weight" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [[10., 10.],
                [14., 14.]], decimal=5)
          if "emb_fcs.0.bias" in name:
            np.testing.assert_almost_equal(param.grad.cpu(),
               [1., 2.])

        

if __name__ == '__main__':
    unittest.main()

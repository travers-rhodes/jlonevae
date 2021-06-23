import unittest
import numpy as np
import torch
from jlonevae_lib.architecture.vae import ConvVAE
from jlonevae_lib.architecture.vae_jacobian import compute_embedding_jacobian_analytic, \
         embedding_jacobian_loss_function, compute_generator_jacobian_optimized, \
         compute_generator_jacobian_analytic

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
    def test_embedding_jacobian(self):
      image = torch.tensor(np.zeros(shape=(1,2,1,1), dtype=np.float32))
      device = "cpu"
      model = get_linear_model(device)
      jac = compute_embedding_jacobian_analytic(model, image, device, jac_batch_size=1)
      np.testing.assert_almost_equal(jac[0],np.array([[1,2],[3,4],[5,6]], dtype=np.float32).reshape(3,2,1,1))
    
    def test_embedding_jacobian_batch(self):
      images = torch.tensor(np.zeros(shape=(10,2,1,1), dtype=np.float32))
      device = "cpu"
      model = get_linear_model(device)
      jacs = compute_embedding_jacobian_analytic(model, images, device, jac_batch_size=1)
      for jac in jacs:
        np.testing.assert_almost_equal(jac,np.array([[1,2],[3,4],[5,6]], dtype=np.float32).reshape(3,2,1,1))

class TestEmbeddingJacobianCost(unittest.TestCase):
    def test_embedding_jacobian_cost(self):
      batch_size = 20
      images = torch.tensor(np.zeros(shape=(batch_size,2,1,1), dtype=np.float32))
      device = "cpu"
      model = get_linear_model(device)
      loss = embedding_jacobian_loss_function(model, images, device)
      expected_loss = np.sum(np.array([[1,2],[3,4],[5,6]]))
      np.testing.assert_almost_equal(loss.detach().cpu().numpy(), expected_loss)

class TestGenerator(unittest.TestCase):
    def test_generator_jacobian(self):
      embedding = torch.tensor(np.zeros(shape=(1,3), dtype=np.float32))
      device = "cpu"
      model = get_linear_model(device)
      jacs = compute_generator_jacobian_analytic(model, embedding, im_channels=2, im_side_len=1, device=device)
      np.testing.assert_almost_equal(jacs[:,0],np.array([[1,2],[3,4],[5,6]], dtype=np.float32).reshape(3,2,1,1)*2)
    
    def test_generation_jacobian_optimized(self):
      embedding = torch.tensor(np.zeros(shape=(13,3), dtype=np.float32))
      device = "cpu"
      model = get_linear_model(device)
      # output shape is (latent_dim, batch_size, model_output_shape)
      # weirdly, you need large epsilon in order to get good precision (large epsilon is fine for linear functions).
      jacs = compute_generator_jacobian_optimized(model, embedding, epsilon_scale=1, device=device) 
      for jac in torch.transpose(jacs,0,1).detach().cpu().numpy():
        np.testing.assert_almost_equal(jac,np.array([[1,2],[3,4],[5,6]], dtype=np.float32).reshape(3,2,1,1)*2)

class TestEmbeddingJacobianCost(unittest.TestCase):
    def test_embedding_jacobian_cost(self):
      batch_size = 20
      images = torch.tensor(np.zeros(shape=(batch_size,2,1,1), dtype=np.float32))
      device = "cpu"
      model = get_linear_model(device)
      loss = embedding_jacobian_loss_function(model, images, device)
      expected_loss = np.sum(np.array([[1,2],[3,4],[5,6]]))
      np.testing.assert_almost_equal(loss.detach().cpu().numpy(), expected_loss)

if __name__ == '__main__':
    unittest.main()

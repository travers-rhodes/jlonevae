import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import os
from opt_einsum import contract

TESTING = True

def compute_generator_jacobian_image_optimized(model, embedding, epsilon_scale = 0.001, device="cpu"):
    raw_jacobian = compute_generator_jacobian_optimized(model, embedding, epsilon_scale, device)
    # shape is (latent_size, batch_size, numchannels = 1, im_size, im_size)
    jacobian = torch.sum(raw_jacobian, dim=2,keepdim = True)
    return(jacobian)

# output shape is (latent_dim, batch_size, model_output_shape)
def compute_generator_jacobian_optimized(model, embedding, epsilon_scale = 0.001, device="cpu"):
    batch_size = embedding.shape[0]
    latent_dim = embedding.shape[1]
    # repeat "tiles" like ABCABCABC (not AAABBBCCC)
    # note that we detach the embedding here, so we should hopefully
    # not be pulling our gradients further back than we intend
    encoding_rep = embedding.repeat(latent_dim + 1,1).detach().clone()
    # define our own repeat to work like "AAABBBCCC"
    delta = torch.eye(latent_dim)\
                .reshape(latent_dim, 1, latent_dim)\
                .repeat(1, batch_size, 1)\
                .reshape(latent_dim*batch_size, latent_dim)
    delta = torch.cat((delta, torch.zeros(batch_size,latent_dim))).to(device)
    # we randomized this before up to epsilon_scale,
    # but for now let's simplify and just have this equal to epsilon_scale.
    # I'd be _very_ impressed if the network can figure out to make the results
    # periodic with this frequency in order to get around this gradient check.
    epsilon = epsilon_scale     
    encoding_rep += epsilon * delta
    recons = model.decode(encoding_rep)
    temp_calc_shape = [latent_dim+1,batch_size] + list(recons.shape[1:])
    recons = recons.reshape(temp_calc_shape)
    recons = (recons[:-1] - recons[-1])/epsilon
    return(recons)


# assume VAE generation functin has input of embedding numpy array
# -----which has shape (batch_size, latent_dim)
# and output shape (batch_size, imchannels, imsidelen(row), imsidelen(col))
# RETURNS: Jacobian matrix, with shape
# latent_dim, batch_size, im_channels, imsize, imsize
def compute_generator_jacobian_analytic(model, 
                                        embedding, 
                                        im_channels=3, 
                                        im_side_len=64,
                                        device="cpu",
                                        jac_batch_size=128):
    batch_size = embedding.shape[0]
    assert batch_size == 1, "for now assert batch size one"
    latent_dim = embedding.shape[1]
    imsize = im_side_len
    encoding_rep = torch.tensor(
            embedding.repeat(imsize*imsize*im_channels,1)\
                     .detach().cpu().numpy(), 
        requires_grad=True, 
        device=device)    
    
 
    total_rows = imsize * imsize * im_channels
    #want_to_increase_these.shape
    gradvec = torch.zeros((total_rows, im_channels, imsize, imsize)).to(device)
    for row in range(imsize):
        #if row != 2: 
        #    continue #debugging. only update row 2
        for col in range(imsize):
            #if col != 1:
            #    continue #debugging. only update column 1
            for chan in range(im_channels):
                #if chan != 0:
                #    continue #debugging. only update channel 0
                gradvec[chan * imsize * imsize + row * imsize + col, chan, row, col] = 1
                
    #print("set ones:", np.where(gradvec.detach().cpu().numpy() == 1))
                
    encoding_rep_splits = encoding_rep.split(jac_batch_size)
    gradvec_splits = gradvec.split(jac_batch_size)
    jac_num_batches = len(gradvec_splits)
    startInd = 0
    jacobians_array = []
    for splitInd in range(jac_num_batches):
        encoding_rep_split = encoding_rep_splits[splitInd].detach()
        encoding_rep_split.requires_grad = True
        #print("ers shape", encoding_rep_split.shape)
        recons = model.decode(encoding_rep_split)

        recons.backward(gradvec_splits[splitInd])
        jacobians = encoding_rep_split.grad.detach()\
                    .cpu()\
                    .numpy()
        #print("jacshape:", jacobians.shape)
        jacobians_array.append(jacobians)
    jacobians_array = np.concatenate(jacobians_array,axis=0)
    #print("jacarrayshape:", jacobians_array.shape)
    #print(jacobians_array.shape)
    # wait. really? transpose(0,1) does nothing? Wow.
    jacobians_array_reshaped = jacobians_array\
                    .transpose(1,0)\
                    .reshape(-1, batch_size, im_channels, imsize, imsize)
    return(jacobians_array_reshaped)

def jacobian_loss_function(model, mu, logvar, device):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(model, mu, epsilon_scale = 0.001, device=device)
    #print(jacobian.shape)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, -1))
    obs_dim = jacobian.shape[2]
    loss = torch.sum(torch.abs(jacobian))/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)

def embedding_jacobian_loss_function(model, images, device, jac_batch_size=128):
    latent_dim = model.latent_dim
    batch_size = images.shape[0]
    #jac output latent_dim x image_shape
    loss = None
    jacobian = compute_embedding_jacobian_analytic(model, 
                images, 
                device=device, 
                jac_batch_size=jac_batch_size)
    loss = torch.sum(torch.abs(jacobian))/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)

def embedding_jacobian_twoone_loss_function(model, images, device, jac_batch_size=128):
    latent_dim = model.latent_dim
    batch_size = images.shape[0]
    loss = None
    # jacobian matrix has shape batch_size, latent_dim, im_channels, im_side_len, im_side_len
    jacobian = compute_embedding_jacobian_analytic(model, 
                images, 
                device=device, 
                jac_batch_size=jac_batch_size)
    # sum over all but batch_size and latent_dim (which are indices 0,1)
    row_norms = torch.sqrt(jacobian.pow(2).sum((2,3,4)))
    # take the L1 norm over those row_norms
    loss = torch.sum(row_norms)/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)


# assume VAE embedding function has output of embedding numpy array
# -----which has shape (batch_size, latent_dim)
# and input shape (batch_size, imchannels, imsidelen(row), imsidelen(col))
# RETURNS: jacobian matrix with shape batch_size, latent_dim, im_channels, im_side_len, im_side_len
def compute_embedding_jacobian_analytic(model, 
                                        images,
                                        device="cpu",
                                        jac_batch_size=128):
    latent_dim = model.latent_dim
    batch_size = images.shape[0]
    im_channels = images.shape[1]
    im_side_len = images.shape[2]
    assert im_side_len == images.shape[3], "image should be square"
    imsize = im_side_len
    image_rep = torch.tensor(
            images.repeat(latent_dim,1,1,1)\
                     .detach().cpu().numpy(), 
            requires_grad=True, 
            device=device)
    
 
    total_rows = batch_size * latent_dim
    #want_to_increase_these.shape
    gradvec = torch.zeros((total_rows, latent_dim)).to(device)
    for bind in range(batch_size):
      for ind in range(latent_dim):
        gradvec[bind * latent_dim + ind, ind] = 1
    #print("set ones:", np.where(gradvec.detach().cpu().numpy() == 1))
                
    image_rep_splits = image_rep.split(jac_batch_size)
    gradvec_splits = gradvec.split(jac_batch_size)
    jac_num_batches = len(gradvec_splits)
    startInd = 0
    jacobians_array = torch.empty((0,im_channels, im_side_len, im_side_len), device=device)
    for splitInd in range(jac_num_batches):
        image_rep_split = image_rep_splits[splitInd].detach()
        image_rep_split.requires_grad = True
        #print("ers shape", encoding_rep_split.shape)
        mu, logvar = model.encode(image_rep_split)
        # I feel like you should always call zero_grad before backward
        model.zero_grad()
        # by creating the graph we allow subsequent backward on grad
        mu.backward(gradvec_splits[splitInd], create_graph=True)
        jacobians = image_rep_split.grad
        jacobians_array = torch.cat((jacobians_array,jacobians), dim=0)
    if TESTING:
     np.testing.assert_almost_equal(jacobians_array.shape, (total_rows, im_channels, im_side_len, im_side_len))
    jacobians_array_reshaped = jacobians_array.reshape(batch_size, latent_dim, im_channels, im_side_len, im_side_len)
    return(jacobians_array_reshaped)

# RETURNS: jacobian matrix with shape batch_size, latent_dim, im_channels, im_side_len, im_side_len
def compute_embedding_jacobian_optimized(model, images, epsilon_scale = 0.001, device="cpu"):
    batch_size = images.shape[0]
    im_channels = images.shape[1]
    imsize = images.shape[2]
    assert imsize == images.shape[3], "image should be square"
    latent_dim = model.latent_dim

    total_rows = im_channels * imsize * imsize
    # torch's repeat "tiles" like ABCABCABC (not AAABBBCCC)
    # note that we detach the embedding here, so we should hopefully
    # not be pulling our gradients further back than we intend
    images_rep = images.repeat(total_rows+1,1,1,1).detach().clone().to(device)
    
    delta = torch.zeros((total_rows+1, im_channels, imsize, imsize)).to(device)
    for row in range(imsize):
        #if row != 2: 
        #    continue #debugging. only update row 2
        for col in range(imsize):
            #if col != 1:
            #    continue #debugging. only update column 1
            for chan in range(im_channels):
                #if chan != 0:
                #    continue #debugging. only update channel 0
                delta[chan * imsize * imsize + row * imsize + col, chan, row, col] = 1
    # this is like numpy repeat on first axis (AAABBBCCC)
    delta = torch.repeat_interleave(delta,batch_size, dim=0)
    # we randomized this before up to epsilon_scale,
    # but for now let's simplify and just have this equal to epsilon_scale.
    # I'd be _very_ impressed if the network can figure out to make the results
    # periodic with this frequency in order to get around this gradient check.
    epsilon = epsilon_scale     
    images_rep += epsilon * delta
    embs = model.encode(images_rep)[0]
    temp_calc_shape = [total_rows+1, batch_size] + list(embs.shape[1:])
    embs = embs.view(temp_calc_shape)
    embs = (embs[:-1] - embs[-1])/epsilon
    embs = embs.view(im_channels, imsize, imsize, batch_size, latent_dim)
    # RETURNS: jacobian matrix with shape batch_size, latent_dim, im_channels, im_side_len, im_side_len
    embs = embs.permute(3,4,0,1,2) 
    return(embs)

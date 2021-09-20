import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import os
from opt_einsum import contract


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

## This function was added based on reviewer feedback to check the effect of regularizing the L2
## to various extents (on top of the implicit (weighted) L2 loss that is already part of the VAE
def jacobian_l2_loss_function(model, mu, logvar, device):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(model, mu, epsilon_scale = 0.001, device=device)
    #print(jacobian.shape)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, -1))
    obs_dim = jacobian.shape[2]
    loss = torch.sum(torch.square(jacobian))/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)

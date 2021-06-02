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

def compute_generator_jacobian_image(model, embedding, epsilon_scale = 0.001, device="cpu"):
    raw_jacobian = compute_generator_jacobian(model, embedding, epsilon_scale, device)
    # shape is (latent_size, batch_size, numchannels = 1, im_size, im_size)
    jacobian = torch.sum(raw_jacobian, dim=2,keepdim = True)
    return(jacobian)

# output shape is (latent_dim, batch_size, model_output_shape)
def compute_generator_jacobian(model, embedding, epsilon_scale = 0.001, device="cpu"):
    batch_size = embedding.shape[0]
    latent_dim = embedding.shape[1]
    # repeat "tiles" like ABCABCABC (not AAABBBCCC)
    encoding_rep = torch.tensor(
        embedding.repeat(latent_dim + 1,1)
                                   .detach().cpu().numpy(), 
        requires_grad=False, 
        device=device)
    epsilon = np.random.uniform() * epsilon_scale
    for ii in range(latent_dim):
        encoding_rep[(ii*batch_size):((ii+1)*batch_size),ii] += epsilon
    recons = model.decode(encoding_rep)
    jacobian_shape = [latent_dim,batch_size] + list(recons.shape[1:])
    jacobian = torch.empty(size=jacobian_shape, dtype=embedding.dtype).to(device)
    for ii in range(latent_dim):
        jacobian[ii,:] = \
                (recons[(ii*batch_size):((ii+1)*batch_size),:] - recons[-batch_size:,:]) / epsilon
    return(jacobian)

def jacobian_loss_function(model, mu, logvar, device, scaling="none"):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(model, mu, epsilon_scale = 0.001, device=device)
    #print(jacobian.shape)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, -1))
    obs_dim = jacobian.shape[2]
    if scaling=="lone":
        # loss is a scalar
        loss = torch.sum(torch.abs(jacobian))/batch_size
    elif scaling=="lonesquared":
        # loss is a scalar
        loss = torch.square((torch.sum(torch.abs(jacobian))/batch_size))
    elif scaling=="scaledlone":
        scaled_jacobian = contract("lbo,bl->lbo",jacobian, torch.exp(-logvar/2))
        loss = torch.sum(torch.abs(scaled_jacobian))/batch_size
    else:
        print("UNKNOWN SCALING TYPE %s" % scaling)

    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)


# this is the "overly complicated" calculation that looks at J^T tanh(J), etc.
# instead of just looking at J and making J sparse
def correlation_jacobian_loss_functions(model, mu, logvar, device, scaling="none"):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(model, mu, epsilon_scale = 0.001, device=device)
    #print(jacobian.shape)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, -1))
    obs_dim = jacobian.shape[2]
    pca_corr      = contract("xbo,ybo->bxy",jacobian, jacobian)/(obs_dim)
    tanh_corr = torch.einsum("xbo,ybo->bxy",torch.tanh(jacobian), jacobian)/(obs_dim)
    if scaling == "diagonal":
        pca_diag  = 1/(contract("bxx->b",pca_corr))
        pca_corr = contract("b,bxy->bxy",pca_diag,pca_corr)
        tanh_diag     = 1/(contract("bxx->b",tanh_corr))
        tanh_corr = contract("b,bxy->bxy",tanh_diag,tanh_corr)
    elif scaling == "vardiagonal":
        pca_diag  = 1/(contract("bxx,bx->b",pca_corr, torch.exp(logvar)))
        pca_corr = contract("b,bxy->bxy",pca_diag,pca_corr)
        tanh_diag     = 1/(contract("bxx,bx->b",tanh_corr, torch.exp(logvar)))
        tanh_corr = contract("b,bxy->bxy",tanh_diag,tanh_corr)
    elif scaling == "individual":
        pca_diag  = 1/(torch.sqrt(contract("bxx->bx",pca_corr)) + 1e-8)
        pca_corr = contract("bx,bxy,by->bxy",pca_diag,pca_corr,pca_diag)
        tanh_diag     = 1/(torch.sqrt(contract("bxx->bx",tanh_corr)) + 1e-8)
        tanh_corr = contract("bx,bxy,by->bxy",tanh_diag,tanh_corr,tanh_diag)
    elif scaling == "sixth":
        pca_diag  = 1/(contract("bxx->bx",pca_corr))
        pca_sixth_diag = torch.topk(pca_diag, 6, dim=1,sorted=True)[0][:,5]
        pca_corr = contract("b,bxy->bxy",pca_sixth_diag,pca_corr)
        tanh_diag  = 1/(contract("bxx->bx",tanh_corr))
        tanh_sixth_diag = torch.topk(tanh_diag, 6, dim=1,sorted=True)[0][:,5]
        tanh_corr = contract("b,bxy->bxy",tanh_sixth_diag,tanh_corr)
    elif scaling == "sixthgeommean":
        pca_diag  = 1/(contract("bxx->bx",pca_corr))
        # we have one over pca_diag above, so we want to pick the _smallest_ elements below...sillyyyyy
        pca_sixth_diag = torch.topk(pca_diag, 6, dim=1, largest = False, sorted=True)[0]
        pca_sixth_diag = torch.exp(torch.mean(torch.log(pca_sixth_diag),dim=1))
        pca_corr = contract("b,bxy->bxy",pca_sixth_diag,pca_corr)
        tanh_diag  = 1/(contract("bxx->bx",tanh_corr))
        # see note above on why largest = False
        tanh_sixth_diag = torch.topk(tanh_diag, 6, dim=1,largest = False, sorted=True)[0]
        tanh_sixth_diag = torch.exp(torch.mean(torch.log(tanh_sixth_diag),dim=1))
        tanh_corr = contract("b,bxy->bxy",tanh_sixth_diag,tanh_corr)
    elif scaling == "none":
        pass
    else:
        print("scaling of '%s' is not defined"%scaling)

    pca_loss  = torch.sqrt(torch.sum(torch.square(pca_corr * (1 - torch.eye(jacobian.shape[0], device=device)))))
    
    tanh_loss = torch.sqrt(torch.sum(torch.square(tanh_corr * (1 - torch.eye(jacobian.shape[0], device=device)))))
    
    return(pca_loss, tanh_loss)
   

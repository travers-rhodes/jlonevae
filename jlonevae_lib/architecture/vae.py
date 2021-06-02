import torch
from torch import nn, optim
from torch.nn import functional as F
import math

class VAE(nn.Module):
    def __init__(self, beta=1.0):
        super(VAE, self).__init__()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn((mu.shape[0], mu.shape[1]), device=mu.device)
        return mu + eps*std
    
    def noisy_decode(self, mu, logvar):
        z = self.reparameterize(mu,logvar)
        x = self.decode(z)
        return(x, z)
    
    def noiseless_autoencode(self,x):
        mu, logvar = self.encode(x)
        x = self.decode(mu)
        return(x,mu,logvar)
        
    def forward(self,x):
        mu, logvar = self.encode(x)
        x, noisy_mu = self.noisy_decode(mu,logvar)
        return(x,mu,logvar, noisy_mu)
    
class ConvVAE(VAE):
    def __init__(self, 
            latent_dim, 
            im_side_len=64, 
            im_channels=3,
            emb_conv_layers_channels = [64, 128, 192, 256], 
            emb_conv_layers_strides = [2,2,2,2], 
            emb_conv_layers_kernel_sizes = [4,4,4,4], 
            emb_fc_layers_num_features = [1024], 
            gen_fc_layers_num_features = [4*4*256], 
            gen_first_im_side_len=4,
            gen_conv_layers_channels = [192, 128, 128, 64,3], 
            gen_conv_layers_strides = [2,2,1,2,2], 
            gen_conv_layers_kernel_sizes = [4,4,4,4,4]):
        super(ConvVAE, self).__init__()
        self.architecture="conv_vae"
        self.latent_dim = latent_dim
        self.im_side_len = int(im_side_len)
        self.im_channels = im_channels
        self.emb_conv_layers_channels = emb_conv_layers_channels
        self.emb_conv_layers_strides = emb_conv_layers_strides
        self.emb_conv_layers_kernel_sizes = emb_conv_layers_kernel_sizes 
        self.emb_fc_layers_num_features = emb_fc_layers_num_features
        self.gen_fc_layers_num_features = gen_fc_layers_num_features
        self.gen_first_im_side_len = gen_first_im_side_len
        self.gen_conv_layers_channels = gen_conv_layers_channels
        self.gen_conv_layers_strides = gen_conv_layers_strides
        self.gen_conv_layers_kernel_sizes = gen_conv_layers_kernel_sizes 

        prev_channels = im_channels
        side_len = self.im_side_len
        layer_channels = self.im_channels
        # construct the parameters for all the embedding convolutions
        self.emb_convs = []
        for i, layer_channels in enumerate(self.emb_conv_layers_channels):
            self.emb_convs.append(
                    nn.Conv2d(prev_channels, 
                              layer_channels, 
                              self.emb_conv_layers_kernel_sizes[i], 
                              self.emb_conv_layers_strides[i]
                              ))
            side_len = int(math.floor(
                (side_len - (self.emb_conv_layers_kernel_sizes[i]- 1) - 1)/
                  self.emb_conv_layers_strides[i] 
                + 1))
            prev_channels = layer_channels
        # construct the parameters for all the embedding fully-connected layers 
        self.emb_fcs = []
        # layer channels is the last-used layer channels
        prev_features = int(side_len * side_len * layer_channels)
        layer_features = prev_features
        for layer_features in self.emb_fc_layers_num_features:
            self.emb_fcs.append(nn.Linear(prev_features, layer_features))
            prev_features = layer_features
        # separately save the mu and logvar layers
        self.fcmu = nn.Linear(layer_features, self.latent_dim)
        self.fclogvar = nn.Linear(layer_features, self.latent_dim)
        # INITIALIZATION CHANGE: don't initialize logvar to average to 0
        # Instead, initialize to average to -3
        self.fclogvar.bias.data.sub_(3.)
        
        # construct the parameters for all the generative fully-connected layers 
        self.gen_fcs = []
        prev_features = self.latent_dim
        for layer_features in self.gen_fc_layers_num_features:
            self.gen_fcs.append(nn.Linear(prev_features, layer_features))
            prev_features = layer_features
        # construct the parameters for all the generative convolutions
        side_len = self.gen_first_im_side_len
        self.gen_first_conv_channels = int(layer_features/(side_len * side_len))
        prev_channels = self.gen_first_conv_channels
        self.gen_convs = []
        self.gen_conv_crops = []
        for i, layer_channels in enumerate(self.gen_conv_layers_channels):
            pytorch_side_len = (side_len - 1) * self.gen_conv_layers_strides[i] + (self.gen_conv_layers_kernel_sizes[i] - 1) + 1
            desired_side_len = side_len * self.gen_conv_layers_strides[i] # to match tensorflow
            self.gen_convs.append(
                    nn.ConvTranspose2d(prev_channels,
                              layer_channels, 
                              self.gen_conv_layers_kernel_sizes[i], 
                              self.gen_conv_layers_strides[i]
                              ))
            # This self.gen_conv_crops logic is soooo annoying, but it's necessary in order to
            # match tensorflow's padding="SAME"
            smaller_crop = math.floor((pytorch_side_len - desired_side_len)/2)
            larger_crop = pytorch_side_len - desired_side_len - smaller_crop
            if smaller_crop + larger_crop > 0:
                #print(smaller_crop, larger_crop) 
                self.gen_conv_crops.append(
                      torch.nn.ZeroPad2d((-smaller_crop, -larger_crop, -smaller_crop, -larger_crop)))
            else:
                self.gen_conv_crops.append(None)
            side_len = desired_side_len
            prev_channels = layer_channels
            #print("side len of ", side_len, self.gen_conv_layers_channels)
        assert self.im_side_len == side_len, "Your reconstructed image (side size %d) is a different size from input (side size %d) for architecture %s" % (side_len, self.im_side_len, self.architecture)
        assert self.im_channels == layer_channels, "Your reconstructed image has a different number of channels (%d) from your input (%d)" % (layer_channels, self.im_channels)

        self.emb_convs = nn.ModuleList(self.emb_convs)
        self.emb_fcs = nn.ModuleList(self.emb_fcs)
        self.gen_fcs = nn.ModuleList(self.gen_fcs)
        self.gen_convs = nn.ModuleList(self.gen_convs)
        
    # x is expected to be a Tensor of the form
    # batchsize x self.im_channels x self.im_side_len x self.im_side_len 
    # and the output is a pair of Tensors of sizes
    # batchsize x self.latent_dim
    # and
    # batchsize x self.latent_dim
    def encode(self,x):
        layer = x
        for conv in self.emb_convs:
            layer = F.relu(conv(layer))
        # flatten all but the 0th dimension
        layer = torch.flatten(layer, 1)
        for fc in self.emb_fcs:
            layer = F.relu(fc(layer))
        
        mu = self.fcmu(layer)
        logvar = self.fclogvar(layer)
        return(mu,logvar)
    
    def decode(self,z):
        layer = z
        num_fcs = len(self.gen_fcs)
        num_convs = len(self.gen_convs)
        for i, fc in enumerate(self.gen_fcs):
            layer = fc(layer)
            # special logic to not do ReLu on last FC layer if we only have Fcs
            if not (num_convs == 0 and i+1 == num_fcs):
              layer = F.relu(layer)
        layer = layer.view(-1,
                self.gen_first_conv_channels,
                self.gen_first_im_side_len,
                self.gen_first_im_side_len)
        for i, conv in enumerate(self.gen_convs):
            layer = conv(layer)
            # special logic to not do ReLu on last conv layer 
            if not (i+1 == num_convs):
              layer = F.relu(layer)
            # This self.gen_conv_crops logic is soooo annoying, but it's necessary in order to
            # match tensorflow's padding="SAME"
            if self.gen_conv_crops[i] is not None:
                layer = self.gen_conv_crops[i](layer)
        g = torch.sigmoid(layer)
        return(g)

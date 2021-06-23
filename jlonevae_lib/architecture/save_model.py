import pickle
import os
import torch

# Save the model in an custom-code-readable way
def save_conv_vae(convvae, model_folder_path):
    kwargs = {"latent_dim": convvae.latent_dim,
              "im_side_len": convvae.im_side_len,
              "im_channels": convvae.im_channels,
              "emb_conv_layers_channels": convvae.emb_conv_layers_channels,
              "emb_conv_layers_strides": convvae.emb_conv_layers_strides,
              "emb_conv_layers_kernel_sizes": convvae.emb_conv_layers_kernel_sizes,
              "emb_fc_layers_num_features": convvae.emb_fc_layers_num_features,
              "gen_fc_layers_num_features": convvae.gen_fc_layers_num_features,
              "gen_first_im_side_len": convvae.gen_first_im_side_len,
              "gen_conv_layers_channels": convvae.gen_conv_layers_channels,
              "gen_conv_layers_strides": convvae.gen_conv_layers_strides,
              "gen_conv_layers_kernel_sizes": convvae.gen_conv_layers_kernel_sizes,
              "final_activation": convvae.final_activation
             }
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    with open(model_folder_path + "/model_args.p", "wb") as f:
        pickle.dump(kwargs, f)
    with open(model_folder_path + "/model_type.txt", "w") as f:
        f.write(convvae.architecture)
    torch.save(convvae.state_dict(), model_folder_path + "/model_state_dict.pt")

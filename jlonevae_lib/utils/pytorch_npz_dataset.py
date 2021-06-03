#### If, instead of using disentanglement_lib's infrastructure
#### you want to just train on a npz file of data directly
#### (eg: no ground-truth factors, like for natural images)
#### you can use this dataloader
import torch
import numpy as np

# maybe I'm just being overly fancy here,
# but this is basically just a constructor to 
# create and return a torch.utils.data.TensorDataset
# based on an .npz file. 
# We only have one tensor, so instead of __getitem__
# returning list of tensors it just returns first (single) tensor
class PytorchNpzDataset(torch.utils.data.TensorDataset):
    # Constructor inputs:
    #   * compressed_npz_files: the path of the the compressed .npz files (a list)
    #   * key: the key where our data tensor is stored inside the .npz
    def __init__(self, compressed_npz_files, useDoublePrecision=False, key="images"):
        if useDoublePrecision:
            value_type = torch.double
        else:
            value_type = torch.float
        loaded_tensor = None
        for file_name in compressed_npz_files:
            print("Loading %s into memory" % file_name)
            nparraychunk = np.load(file_name)[key]
            tensorchunk = torch.tensor(nparraychunk, dtype=value_type)
            if loaded_tensor is None:
                loaded_tensor = tensorchunk
            else:
                loaded_tensor = torch.cat((loaded_tensor, tensorchunk), 0)

        super(PytorchNpzDataset, self).__init__(loaded_tensor)

    # return tensor, not list of tensors, since we only have one dataset
    # (eg we don't have labeled pairs of data in this dataset, just one input data tensor)
    def __getitem__(self, index):
        return (super(PytorchNpzDataset, self).__getitem__(index)[0])

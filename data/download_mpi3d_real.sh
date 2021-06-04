#!/bin/bash
# download the MPI3d-real data from https://github.com/rr-learning/disentanglement_dataset
# That data is licensed under a Creative Commons Attribution 4.0 International License 
# (https://creativecommons.org/licenses/by/4.0/).
# Thus, we cite
# @article{gondal2019transfer,
#   title={On the Transfer of Inductive Bias from Simulation to the Real World: a New Disentanglement Dataset},
#   author={Gondal, Muhammad Waleed and W{\"u}thrich, Manuel and Miladinovi{\'c}, {\DJ}or{\dj}e and Locatello, Francesco and Breidt, Martin and Volchkov, Valentin and Akpo, Joel and Bachem, Olivier and Sch{\"o}lkopf, Bernhard and Bauer, Stefan},
#   journal={arXiv preprint arXiv:1906.03292},
#   year={2019}
# }

echo "Downloading mpi3d_real dataset."
if [[ ! -d "mpi3d_real" ]]; then
  mkdir mpi3d_real
  wget -O mpi3d_real/mpi3d_real.npz https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz
fi
echo "Downloading mpi3d_real completed!"

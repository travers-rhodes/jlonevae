echo "Downloading mpi3d_real dataset."
if [[ ! -d "mpi3d_real" ]]; then
  mkdir mpi3d_real
  wget -O mpi3d_real/mpi3d_real.npz https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz
fi
echo "Downloading mpi3d_real completed!"


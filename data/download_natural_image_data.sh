#!/bin/bash

# Download and convert to a numpy npz the natural image data used in Olshausen and Field in Nature, vol. 381, pp. 607-609.
# Currently, this data is available at http://www.rctn.org/bruno/sparsenet/
# Which is a website called "Sparse coding simulation software written by Bruno Olshausen"

if ! test -f naturalImages/IMAGES.mat; then
  wget http://www.rctn.org/bruno/sparsenet/IMAGES.mat -P naturalImages/
fi

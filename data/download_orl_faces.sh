#!/bin/bash
# download the orl_faces data from https://web.archive.org/web/20160527000913/https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html 
# That data is licensed under "When using these images, please give credit to AT&T Laboratories Cambridge" 
# Thus, we additionally need to cite ATT somehow
#@INPROCEEDINGS{341300,
#  author={Samaria, F.S. and Harter, A.C.},
#  booktitle={Proceedings of 1994 IEEE Workshop on Applications of Computer Vision},
#  title={Parameterisation of a stochastic model for human face identification},
#  year={1994},
#  volume={},
#  number={},
#  pages={138-142},
#  doi={10.1109/ACV.1994.341300}}

echo "Downloading orl_faces dataset."
if [[ ! -d "orl_faces" ]]; then
  mkdir orl_faces 
  wget -O orl_faces/orl_faces.zip https://web.archive.org/web/20160426011203/http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip
  pushd orl_faces
  unzip orl_faces.zip
  rm orl_faces.zip
  popd
fi
echo "Downloading orl_faces completed!"

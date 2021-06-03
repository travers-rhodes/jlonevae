#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

#https://stackoverflow.com/questions/7069682/how-to-get-arguments-with-flags-in-bash/21128172
reload_conda=false
while getopts 'r' flag; do
	case "${flag}" in
                r) reload_conda=true;;
		*) printf "unknown flag\n"
		   exit 1;;
	esac
done

if "$reload_conda" = true;
then
  # create our conda environment (if it doesn't exist). If it does exist, update it
  # https://github.com/conda/conda/issues/7819
  # once we know our environment exists, activate it
  conda env create -f environment.yml || conda env update -f environment.yml
  # make sure that conda is set up to run with bash scripts
  # (even in this local bash script)
  conda init bash
else
	printf "assuming conda environment already up-to-date\n"
fi

source ~/.bashrc

# activate conda within this child script
conda activate local_ica_vae

experimentName=naturalImageNorm

rm images/latentJacobianImages/$experimentName/*
for latentDim in 10 50 100
do
python3 02-createLatentJacobianImages.py --experimentName $experimentName --datasetName naturalImage --beta 0.01 --betaAnnealingBatches 50000 --latentDim $latentDim
python3 02-createLatentJacobianImages.py --experimentName $experimentName --datasetName naturalImage --beta 0.01 --betaAnnealingBatches 50000 --latentDim $latentDim --gamma 0.01
done

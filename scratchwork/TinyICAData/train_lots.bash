#! /bin/bash -i
# on my Debian you need to run this interactively (with the -i flag)
# or else conda doesn't get loaded properly from ~/.bashrc (even
# when you source ./bashrc directly) because of short-circuit logic
# in ~./bashrc

# activate conda within this child script
conda activate jlonevae 

for beta in 0.001 0.005 0.01 0.1
do
  for gamma in 0.01 0.001 0.005
  do
    echo $beta
    echo $gamma
    ./trainSimple.py --beta $beta --gamma $gamma
  done
done


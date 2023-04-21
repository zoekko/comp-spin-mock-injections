#!/bin/bash

source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate igwn-py39

job=$1
json=$2
outdir=$3

mkdir -p $outdir

python /home/simona.miller/comp-spin-mock-injections/Code/IndividualInference/launchBilby_justSpin.py \
        -job $job \
        -json $json \
        -outdir $outdir

conda deactivate

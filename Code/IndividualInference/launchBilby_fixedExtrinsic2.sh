#!/bin/bash

. /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate /home/simona.miller/.conda/envs/bilby_201

job=$1
json=$2
outdir=$3

mkdir -p $outdir

python /home/simona.miller/comp-spin-mock-injections/Code/IndividualInference/launchBilby_fixedExtrinsic.py \
        -job $job \
        -json $json \
        -outdir $outdir

conda deactivate

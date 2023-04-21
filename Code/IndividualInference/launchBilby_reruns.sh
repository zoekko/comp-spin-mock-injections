#!/bin/bash

source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate igwn-py39

job=$1
json=$2
outdir=$3

# format job into str and remove the corrupt file
if [[ ${#job} -eq 1 ]]
then
  jobstr=0000$job
elif [[ ${#job} -eq 2 ]]
then
  jobstr=000$job
elif [[ ${#job} -eq 3 ]]
then
  jobstr=00$job
elif [[ ${#job} -eq 4 ]]
then
  jobstr=0$job
else
  jobstr=$job
fi

rm ${outdir}/.job_${jobstr}_generate_posterior_cache.pickle

# then relaunch bilby
python /home/simona.miller/comp-spin-mock-injections/Code/IndividualInference/launchBilby.py \
        -job $job \
        -json $json \
        -outdir $outdir

conda deactivate

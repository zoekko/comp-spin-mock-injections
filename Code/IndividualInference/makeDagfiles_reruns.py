import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.special import erf
import sys
import pandas as pd

# repository root
root = '/home/simona.miller/comp-spin-mock-injections/'

# cycle through populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

# jobs that need to be rerun for each population
reruns = {
    pop_names[0]:[26961, 28049, 38360, 39079], 
    pop_names[1]:[857, 13575, 14440, 17940, 27128, 36080, 36927, 47970], 
    pop_names[2]:[5786, 5813, 8393, 16452, 19419, 26073, 28571, 29912, 34313, 36401, 42181, 49802], 
}

for j,pop_name in enumerate(pop_names):

    # Choose random set to inject
    to_inject = reruns[pop_name]

    # Write dag file in the condor subfolder
    dagfile=f'./condor/bilby_{pop_name}_reruns.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/bilby_{2}_reruns.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}.json" outdir="{1}Data/IndividualInferenceOutput/{2}"\n\n'.format(int(i),root,pop_name))

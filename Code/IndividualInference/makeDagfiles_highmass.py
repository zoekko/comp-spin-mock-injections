import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.special import erf
import sys
import pandas as pd
from os.path import exists

# Pass number of of events to inject via commandline 
nevents = sys.argv[1]

# repository root
root = '/home/simona.miller/comp-spin-mock-injections/'

# cycle through populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

for j,pop_name in enumerate(pop_names):

    fname = f'injlist_{pop_name}_highmass_{nevents}events.txt'
    if exists(fname): 
        print('Loading in existing job numbers')
        to_inject = np.loadtxt(fname)
    else:
        print('Generating new random job numbers')
        # Load in injected population
        injections = pd.read_json(f'../../Data/InjectedPopulationParameters/{pop_name}_highmass.json')
        injections.sort_index(inplace=True)
        n_total = len(injections)

        # Choose random set to inject
        to_inject = np.random.choice(range(n_total),size=int(nevents),replace=False)

        # Save as text file for reference
        np.savetxt(fname,to_inject,fmt="%d")

    # Write dag file in the condor subfolder
    dagfile=f'./condor/new_bilby/bilby_{pop_name}_highmass2.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/new_bilby/bilby_{2}.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}_highmass.json" outdir="{1}Data/IndividualInferenceOutput/new_bilby/{2}_highmass"\n\n'.format(int(i),root,pop_name))
            
            
    dagfile=f'./condor/new_bilby/bilby_{pop_name}_fixedExtrinsic_highmass2.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/new_bilby/bilby_{2}_fixedExtrinsic.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}_highmass.json" outdir="{1}Data/IndividualInferenceOutput/new_bilby/{2}_highmass"\n\n'.format(int(i),root,pop_name))
            
            
    dagfile=f'./condor/new_bilby/bilby_{pop_name}_justSpin_highmass2.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/new_bilby/bilby_{2}_justSpin.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}_highmass.json" outdir="{1}Data/IndividualInferenceOutput/new_bilby/{2}_highmass"\n\n'.format(int(i),root,pop_name))

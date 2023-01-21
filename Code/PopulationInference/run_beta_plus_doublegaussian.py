import numpy as np
import glob
import emcee as mc
import json
import sys
from posterior_helper_functions import draw_initial_walkers_uniform
from posteriors import betaPlusDoubleGaussian
from postprocessing import processEmceeChain 

# set seed for reproducibility (number chosen arbitrarily)
np.random.seed(2648)

"""
Definitions and loading data
"""

# Pass population and number of events via commandline 
pop = sys.argv[1]
nevents = sys.argv[2]

# Model
model = "betaPlusDoubleGaussian"
model_savename = model + f"_pop{pop}_{nevents}events_temp" ## TODO: get rid of temp

# File path root for where to store data 
froot_input = "/home/simona.miller/Xeff_injection_campaign/for_hierarchical_inf/"
froot = "../../Data/" ## TODO: eventually input and output should both be this

# Define emcee parameters
nWalkers = 20       # number of walkers 
dim = 8             # dimension of parameter space (number hyper params)
nSteps = 50000      # number of steps for chain

# Set prior bounds (same as Table XII in https://arxiv.org/pdf/2111.03634.pdf)
priorDict = {
    'mu_chi':(0., 1.),
    'sigma_chi':(0.07, 0.5),
    'mu_cost':(-1., 1.),
    'sigma_cost':(0.07, 1),
    'MF_cost':(0., 1.)
}

# Load sampleDict
with open(froot_input+f"sampleDict_pop{pop}_gaussian_spin_posteriors_sigma_meas_0.1_300events.json", 'r') as f: ## TODO: update with "real" data
    sampleDict_full = json.load(f)

# Choose subset of sampleDict if necessary
if int(nevents)<300: 
    events = np.random.choice(sampleDict_full.keys(), size=int(nevents), replace=False)
    sampleDict = {event:sampleDict_full[event] for event in events}
else: 
    sampleDict = sampleDict_full
    
# Load injectionDict
with open(froot_input+"injectionDict_flat.json", 'r') as f: 
    injectionDict = json.load(f)

# Will save emcee chains temporarily in the .tmp folder in this directory
output_folder_tmp = froot+"PopulationInferenceOutput/.tmp/"
output_tmp = output_folder_tmp+model_savename


"""
Initializing emcee walkers or picking up where an old chain left off
"""

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))

# If no chain already exists, begin a new one
if len(old_chains)==0:
    
    print('\nNo old chains found, generating initial walkers ... ')

    run_version = 0

    # Initialize walkers
    initial_mu_chis = draw_initial_walkers_uniform(nWalkers, (0.2,0.4)) # TODO: change these?
    initial_sigma_chis = draw_initial_walkers_uniform(nWalkers, (0.17,0.25))
    initial_mu_costs = draw_initial_walkers_uniform(2*nWalkers, (0.2,0.4))
    initial_sigma_costs = draw_initial_walkers_uniform(2*nWalkers, (0.17,0.25))
    initial_MF_costs = draw_initial_walkers_uniform(nWalkers, (0,1))
    initial_Bqs = np.random.normal(loc=0, scale=3, size=nWalkers)
    
    # Put together all initial walkers into a single array
    initial_walkers = np.transpose(
        [initial_mu_chis, initial_sigma_chis, initial_mu_costs[:nWalkers], initial_sigma_costs[:nWalkers], 
         initial_mu_costs[nWalkers:], initial_sigma_costs[nWalkers:], initial_MF_costs, initial_Bqs]
    )
            
# Otherwise resume existing chain
else:
    
    print('\nOld chains found, loading and picking up where they left off ... ' )
    
    # Load existing file and iterate run version
    old_chain = np.concatenate([np.load(chain) for chain in old_chains], axis=1)
    run_version = int(old_chains[-1][-6:-4])+1

    # Strip off any trailing zeros due to incomplete run
    goodInds = np.where(old_chain[0,:,0]!=0.0)[0]
    old_chain = old_chain[:,goodInds,:]

    # Initialize new walker locations to final locations from old chain
    initial_walkers = old_chain[:,-1,:]
    
    # Figure out how many more steps we need to take 
    nSteps = nSteps - old_chain.shape[1]
    
        
print('Initial walkers:')
print(initial_walkers)


"""
Launching emcee
"""

if nSteps>0: # if the run hasn't already finished

    assert dim==initial_walkers.shape[1], "'dim' = wrong number of dimensions for 'initial_walkers'"

    print(f'\nLaunching emcee with {dim} hyper-parameters, {nSteps} steps, and {nWalkers} walkers ...')

    sampler = mc.EnsembleSampler(
        nWalkers,
        dim,
        betaPlusDoubleGaussian, # model in posteriors.py
        args=[sampleDict,injectionDict,priorDict], # arguments passed to betaPlusDoubleGaussian
        threads=16
    )

    print('\nRunning emcee ... ')

    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):

        # Save every 10 iterations
        if i%10==0:
            np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)

        # Print progress every 100 iterations
        if i%100==0:
            print(f'On step {i} of {nSteps}', end='\r')

    # Save raw output chains
    np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)


"""
Running post processing and saving results
"""

print('\nDoing post processing ...')

if nSteps>0: 

    # If this is the only run, just process this one directly 
    if run_version==0:
        chainRaw = sampler.chain

    # otherwise, put chains from all previous runs together 
    else:
        previous_chains = [np.load(chain) for chain in old_chains]
        previous_chains.append(sampler.chain)
        chainRaw = np.concatenate(previous_chains, axis=1)

else: 
    chainRaw = old_chain

# Run post-processing
chainDownsampled = processEmceeChain(chainRaw) 

# Format output into an easily readable format 
results = {
    'mu_chi':{'unprocessed':chainRaw[:,:,0].tolist(), 'processed':chainDownsampled[:,0].tolist()},
    'sigma_chi':{'unprocessed':chainRaw[:,:,1].tolist(), 'processed':chainDownsampled[:,1].tolist()},
    'mu1_cost':{'unprocessed':chainRaw[:,:,2].tolist(), 'processed':chainDownsampled[:,2].tolist()},
    'sigma1_cost':{'unprocessed':chainRaw[:,:,3].tolist(), 'processed':chainDownsampled[:,3].tolist()},
    'mu2_cost':{'unprocessed':chainRaw[:,:,4].tolist(), 'processed':chainDownsampled[:,4].tolist()},
    'sigma2_cost':{'unprocessed':chainRaw[:,:,5].tolist(), 'processed':chainDownsampled[:,5].tolist()},
    'MF_cost':{'unprocessed':chainRaw[:,:,6].tolist(), 'processed':chainDownsampled[:,6].tolist()},
    'Bq':{'unprocessed':chainRaw[:,:,7].tolist(), 'processed':chainDownsampled[:,7].tolist()}
} 

# Save
savename = froot+f"PopulationInferenceOutput/{model}/{model_savename}.json"
with open(savename, "w") as outfile:
    json.dump(results, outfile)
print(f'Done! Run saved at {savename}')

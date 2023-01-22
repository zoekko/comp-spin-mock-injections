import numpy as np
import sys
from scipy.special import erf
from scipy.special import beta
import pickle
import json 

import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

from posterior_helper_functions import * 

pop = 'pop1'
num_injections = '70'
model = 'gaussianPlusGaussian' # either gaussianPlusGaussian or betaPlusDoubleGaussian

"""
Function to do posterior reweighting for beta plus double Gaussain model 
    sampleDict = dictionary containing individual event samples
    hyperPEDict = dictionary containing hyperparemeter samples: from json file that emcee outputs: 
        samples are the individual curves from the final trace plots (less than the actual number of samples
        for each ind event)
"""
def pop_reweight(sampleDict, hyperPEDict): 
    
    # Number of hyperparameter samples
    nHyperPESamps = len(hyperPEDict['mu_chi']['processed'])
    
    # dict in which to put reweighted individual event samples
    sampleDict_rw = {}
    
    # cycle through events
    for k, event in enumerate(sampleDict): 
        print(f"event {k+1} of {len(sampleDict)}: {event}")
        
        # Unpack posterior samples for this event
        chi1_samples = sampleDict[event]['a1']
        chi2_samples =  sampleDict[event]['a2']
        cost1_samples = sampleDict[event]['cost1']
        cost2_samples = sampleDict[event]['cost2']
        m1_samples = sampleDict[event]['m1']
        m2_samples = sampleDict[event]['m2']
        z_samples = sampleDict[event]['z']
        z_prior_samples = sampleDict[event]['z_prior']
        dVdz_samples = sampleDict[event]['dVc_dz']

        # indices corresponding to each sample for these events (will be used below in the for loop)
        nSamples = len(chi1_samples)
        indices = np.arange(nSamples)
        
        # arrays in which to store reweighted samples for this event
        new_chi1_samps = np.zeros(nHyperPESamps)
        new_chi2_samps = np.zeros(nHyperPESamps)
        new_cost1_samps = np.zeros(nHyperPESamps)
        new_cost2_samps = np.zeros(nHyperPESamps)
        new_mass1_samps = np.zeros(nHyperPESamps)
        new_mass2_samps = np.zeros(nHyperPESamps)
        
        # cycle through hyper PE samps
        for i in range(nHyperPESamps):
            '''iterating through each curve of the trace plot'''
            
            if model == 'gaussianPlusGaussian':
                
                # Fetch i^th hyper PE sample: 
                mu_chi = hyperPEDict['mu_chi']['processed'][i]
                sigma_chi = hyperPEDict['sigma_chi']['processed'][i]
                mu_cost = hyperPEDict['mu_cost']['processed'][i]
                sigma_cost = hyperPEDict['sigma_cost']['processed'][i]
                Bq = hyperPEDict['Bq']['processed'][i]


                # Evaluate model at the locations of samples for this event
                p_chi1 = calculate_Gaussian_1D(chi1_samples, mu_chi, sigma_chi, 0, 1)
                p_chi2 = calculate_Gaussian_1D(chi2_samples, mu_chi, sigma_chi, 0, 1)
                p_cost1 = calculate_Gaussian_1D(cost1_samples, mu_cost, sigma_cost, -1, 1)
                p_cost2 = calculate_Gaussian_1D(cost2_samples, mu_cost, sigma_cost, -1, 1)
                
            elif model == 'betaPlusDoubleGaussian':
                # Fetch i^th hyper PE sample: 
                mu_chi = hyperPEDict['mu_chi']['processed'][i]
                sigma_chi = hyperPEDict['sigma_chi']['processed'][i]
                mu1_cost = hyperPEDict['mu1_cost']['processed'][i]
                sigma1_cost = hyperPEDict['sigma1_cost']['processed'][i]
                mu2_cost = hyperPEDict['mu2_cost']['processed'][i]
                sigma2_cost = hyperPEDict['sigma2_cost']['processed'][i]
                MF_cost = hyperPEDict['MF_cost']['processed'][i]
                Bq = hyperPEDict['Bq']['processed'][i]

                # Translate mu_chi and sigma_chi to beta function parameters a and b 
                # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
                a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)

                # Evaluate model at the locations of samples for this event
                p_chi1 = betaDistribution(chi1_samples, a, b)
                p_chi2 = betaDistribution(chi2_samples, a, b)
                p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1)
                p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1)
            
        
            # Pop dist for all four params combined is a product of each four individual dists
            pSpins = p_chi1*p_chi2*p_cost1*p_cost2
            
            # PE priors for chi_i and cost_i are all uniform, so we set them to unity here
            nSamples = pSpins.size
            spin_PE_prior = np.ones(nSamples)
            
            # Need to reweight by astrophysical priors on m1, m2, z ...
            # - p(m1)*p(m2)
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq)
            old_m1_m2_prior = np.ones(nSamples) # PE prior on masses is uniform in component masses
            # - p(z)
            p_astro_redshift = p_astro_z(z_samples, dVdz_samples)
            # - For full m1, m2, z prior reweighting: 
            m1_m2_z_prior_ratio = (p_astro_m1_m2/old_m1_m2_prior)*(p_astro_redshift/z_prior_samples)
            
            # calculate weights for this hyper parameter
            weights = pSpins*m1_m2_z_prior_ratio/spin_PE_prior
            weights = weights/np.sum(weights)
            
            # select a random sample from the event posterior subject to these weights
            j = np.random.choice(indices, p=weights)
            
            # populate the new sample arrays with this random sample
            new_chi1_samps[i] = chi1_samples[j]
            new_chi2_samps[i] = chi2_samples[j]
            new_cost1_samps[i] = cost1_samples[j]
            new_cost2_samps[i] = cost2_samples[j]
            new_mass1_samps[i] = m1_samples[j]
            new_mass2_samps[i] = m2_samples[j]
        
        # Add into reweighted sampleDict
        sampleDict_rw[event] = {
            'chi1':new_chi1_samps,
            'chi2':new_chi2_samps,
            'cost1':new_cost1_samps,
            'cost2':new_cost2_samps,
            'm1':new_mass1_samps,
            'm2':new_mass2_samps
        }

    return sampleDict_rw


"""
Actually loading and running pop reweighting
"""


if __name__=="__main__":
    # Repository root 
    froot = '/home/zoe.ko/comp-spin-mock-injections/'
    
    # Load dict with individual event PE samples (Load sampleDict):
    # f = open(f'{froot}Data/PopulationInferenceInput/sampleDict_{pop}.json')
    f = open(f'/home/zoe.ko/LIGOSURF22/input/sampleDict_{pop}.json')
    sampleDict = json.load(f) 
    
    if num_injections != '300':
        length = len(sampleDict)
        to_pop = length - int(num_injections)
        keys = list(sampleDict.keys())
        keys_to_pop = np.random.choice(keys, size=to_pop, replace=False)
        for key in keys_to_pop:
            sampleDict.pop(key)
    
    # Load population parameter PE samps
    if model == 'gaussianPlusGaussian':
        fname = 'gaus_gaus'
    elif model == 'betaPlusDoubleGaussian':
        fname = 'double_gaussian'
    hyperPEDict_fname = "/home/zoe.ko/LIGOSURF22/data/" + num_injections + "injections/" + num_injections + model + "/" + num_injections + pop + fname
    with open(f'{hyperPEDict_fname}.json', 'r') as f:
        hyperPEDict = json.load(f)
    
    # Run reweighting for default model
    print("Running reweighting ... ")
    sampleDict_rw = pop_reweight(sampleDict, hyperPEDict)   
    
    # Save results
    savename = f'{froot}Data/PopulationInferenceOutput/{model}/{num_injections}{model}_{pop}rw_sampleDict.pickle'
    with open(savename, 'wb') as f:
        pickle.dump(sampleDict_rw,f)
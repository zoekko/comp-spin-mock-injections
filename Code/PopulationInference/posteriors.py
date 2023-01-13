import numpy as np
from posterior_helper_functions import *

def betaPlusDoubleGaussian(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the beta+doubleGaussian model: a component spin distribution with spin 
    magnitude as beta distribution and the cosine of the tilt angle as a double gaussian, for 
    inference within `emcee`.
    
    Model used in Section III of **PAPER TITLE**
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: 
            [ mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, Bq ] 
        where 
        
        - mu_chi = mean of spin magnitude beta distribution
        
        - sigma_chi = std. dev. of spin magnitude distribution
        
        - mu1_cost = mean of gaussian 1 for cosine tilt distribution
        
        - sigma1_cost = std. dev. gaussian 1 for cosine tilt distribution
        
        - mu2_cost = mean of gaussian 2 for cosine tilt distribution
        
        - sigma2_cost = std. dev. of gaussian 2 for cosine tilt distribution
        
        - MF_cost = fraction in gaussian 1 for cosine tilt distribution
                
        - Bq = power law slope of the mass ratio distribution    
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==8, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu1_cost = c[2] 
    sigma1_cost = c[3]
    mu2_cost = c[4] 
    sigma2_cost = c[5]
    MF_cost = c[6]
    Bq = c[7]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf
    if mu1_cost < priorDict['mu_cost'][0] or mu1_cost > priorDict['mu_cost'][1]:
        return -np.inf
    elif sigma1_cost < priorDict['sigma_cost'][0] or sigma1_cost > priorDict['sigma_cost'][1]:
        return -np.inf
    if mu2_cost < priorDict['mu_cost'][0] or mu2_cost > priorDict['mu_cost'][1]:
        return -np.inf
    elif sigma2_cost < priorDict['sigma_cost'][0] or sigma2_cost > priorDict['sigma_cost'][1]:
        return -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > priorDict['MF_cost'][1]:
        return -np.inf
    
    # Additionally, to break the degeneracy between the two gaussians in the tilt distribution, 
    # impose that mu1_cost <= mu2_cost
    elif mu1_cost > mu2_cost:
        return -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-posterior
        logP = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Double_Gaussian(cost1_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        p_cost2_det = calculate_Double_Gaussian(cost2_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff
        
        # --- Loop across BBH events ---
        for event in sampleDict:

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            pSpins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            # - p(m1)*p(m2)
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq)
            old_m1_m2_prior = np.power(1.+z_samples, 2) # PE prior on masses is uniform in DETECTOR FRAME component masses
            # - p(z)
            p_astro_redshift = p_astro_z(z_samples, dVdz_samples)
            old_z_prior = p_astro_z(z_samples, dVdz_samples, kappa=0) # see: bilby.gw.prior.UniformSourceFrame prior 
            # - For full m1, m2, z prior reweighting: 
            m1_m2_z_prior_ratio = (p_astro_m1_m2/old_m1_m2_prior)*(p_astro_redshift/old_z_prior) 
            
            # Sum over probabilities to get the marginalized likelihood for this event
            # PE priors for chi_i and cost_i are all uniform so "dividing them out" = dividing by 1
            nSamples = pSpins.size
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP

def gaussianPlusGaussian(c,sampleDict,injectionDict,priorDict): 
    
    """
    Implementation of the Gaussian+Gaussian model: a component spin distribution with spin 
    magnitude as 1D gaussian and the cosine of the tilt angle as another 1D gaussian, for 
    inference within `emcee`. We do not include variance between the two gaussians.
    
    Model used in Section IV of **PAPER TITLE**
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: 
            [ mu_chi, sigma_chi, mu_cost, sigma_cost, Bq ] 
        where 
        
        - mu_chi = mean of spin magnitude gaussian
        
        - sigma_chi = std. dev. of spin magnitude gaussian
        
        - mu_cost = mean of cosine tilt gaussian
        
        - sigma_cost = std. dev. of cosine tilt gaussian
                
        - Bq = power law slope of the mass ratio distribution    
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu_cost = c[2] 
    sigma_cost = c[3]
    Bq = c[4]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf
    if mu_cost < priorDict['mu_cost'][0] or mu_cost > priorDict['mu_cost'][1]:
        return -np.inf
    elif sigma_cost < priorDict['sigma_cost'][0] or sigma_cost > priorDict['sigma_cost'][1]:
        return -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-posterior
        logP = 0.
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logP -= (Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_Gaussian_1D(chi1_det, mu_chi, sigma_chi, 0, 1)
        p_chi2_det = calculate_Gaussian_1D(chi2_det, mu_chi, sigma_chi, 0, 1)
        p_cost1_det = calculate_Gaussian_1D(cost1_det, mu_cost, sigma_cost, -1, 1.)
        p_cost2_det = calculate_Gaussian_1D(cost2_det, mu_cost, sigma_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq)
        pdet_z = p_astro_z(z_det, dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Nsamp = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Nsamp<=4*nEvents:
            return -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff
        
        # --- Loop across BBH events ---
        for event in sampleDict:

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_Gaussian_1D(chi1_samples, mu_chi, sigma_chi, 0, 1)
            p_chi2 = calculate_Gaussian_1D(chi2_samples, mu_chi, sigma_chi, 0, 1)
            p_cost1 = calculate_Gaussian_1D(cost1_samples, mu_cost, sigma_cost, -1, 1.)
            p_cost2 = calculate_Gaussian_1D(cost2_samples, mu_cost, sigma_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            pSpins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            # - p(m1)*p(m2)
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq)
            old_m1_m2_prior = np.power(1.+z_samples, 2) # PE prior on masses is uniform in DETECTOR FRAME component masses
            # - p(z)
            p_astro_redshift = p_astro_z(z_samples, dVdz_samples)
            old_z_prior = p_astro_z(z_samples, dVdz_samples, kappa=0) # see: bilby.gw.prior.UniformSourceFrame prior 
            # - For full m1, m2, z prior reweighting: 
            m1_m2_z_prior_ratio = (p_astro_m1_m2/old_m1_m2_prior)*(p_astro_redshift/old_z_prior) 
            
            # Sum over probabilities to get the marginalized likelihood for this event
            # PE priors for chi_i and cost_i are all uniform so "dividing them out" = dividing by 1
            nSamples = pSpins.size
            pEvidence = (1.0/nSamples)*np.sum(pSpins*m1_m2_z_prior_ratio)

            # Add to our running total
            logP += np.log(pEvidence)

        if logP!=logP:
            return -np.inf

        else:
            return logP
import numpy as np
import bilby
import argparse
import sys
import pandas as pd
from bilby.core.prior.base import Constraint
from lalsimulation import SimInspiralTransformPrecessingWvf2PE
from lal import GreenwichMeanSiderealTime

# Parser to load args from commandline
parser = argparse.ArgumentParser()
parser.add_argument('-json',help='Json file with population instantiation')
parser.add_argument('-job',help='Job number',type=int)
parser.add_argument('-outdir',help="Output directory")
args = parser.parse_args()

# Directory 
directory = '/home/simona.miller/comp-spin-mock-injections/'

# Specify the output directory and the name of the simulation.
label = "job_{0:05d}_fixedExtrinsic".format(args.job)
bilby.core.utils.setup_logger(outdir=args.outdir, label=label)

# Load dataframe and select injection
injections = pd.read_json(args.json)
injections.sort_index(inplace=True)
injection = injections.loc[args.job]

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(int(injection.seed))

# Reference frequency and phase
fRef = 50.
phiRef = injection.phase

# Source frame -> detector frame masses
m1_det_inj = injection.m1*(1.+injection.z)
m2_det_inj = injection.m2*(1.+injection.z)

# Convert spin parameters from components --> angles and magnitudes using the  
# lalsimulation function XLALSimInspiralTransformPrecessingWvf2PE()
theta_jn_inj, phi_jl_inj, tilt_1_inj, tilt_2_inj, phi_12_inj, a1_inj, a2_inj = SimInspiralTransformPrecessingWvf2PE(
    injection.inc, 
    injection.s1x, injection.s1y, injection.s1z, 
    injection.s2x, injection.s2y, injection.s2z, 
    m1_det_inj, m2_det_inj, 
    fRef, phiRef
)

# Make dictionary of BBH parameters that includes all of the different waveform
# parameters, including masses and spins of both black holes
injection_parameters = dict(
    mass_1=m1_det_inj,\
    mass_2=m2_det_inj,\
    a_1=a1_inj,\
    a_2=a2_inj,\
    tilt_1=tilt_1_inj,\
    tilt_2=tilt_2_inj,\
    phi_12=phi_12_inj,\
    phi_jl=phi_jl_inj,
    luminosity_distance=injection.Dl,\
    theta_jn=theta_jn_inj,\
    psi=injection.pol,\
    phase=phiRef,\
    geocent_time=1126259642.413,\
    ra=injection.ra, \
    dec=injection.dec
)

print(injection_parameters)

# Sampling frequency and mininmum frequency of the data segment that we're going to 
# inject the signal into
sampling_frequency = 2048.
fMin = 15.

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant='IMRPhenomXPHM',\
    reference_frequency=fRef,\
    minimum_frequency=fMin
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos[0].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=directory+"Code/GeneratePopulations/aligo_O3actual_H1.txt")
ifos[1].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=directory+"Code/GeneratePopulations/aligo_O3actual_L1.txt")
ifos[2].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=directory+"Code/GeneratePopulations/avirgo_O3actual.txt")

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters and inject the signal into the
# ifos. 
# For a small fraction of the signals, the given duration will be too small so we
# increase as need be, in increments of half a second.
duration = 8
durationFlag=False
while durationFlag==False:
    try:
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, 
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments
        )
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, 
            duration=duration,
            start_time=injection_parameters['geocent_time'] - (duration-2) # canonically, geocenter time is set to be 2 sec from the end of the segment
        )
        ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)
        
        durationFlag=True
    except:
        duration = duration + 0.5
          
print('duration:',duration)
        
# For this analysis, we implement the standard precessing BBH priors defined in the prior_justIntrinsic.prior
# for the intrinsic parameters, and fix the extrinsic parameters to truth
priors =  bilby.gw.prior.BBHPriorDict(directory+"Code/IndividualInference/prior_justIntrinsic.prior")
extrinsic_params = [
    'ra','dec', 'geocent_time', 'phase', 'psi', 'luminosity_distance', 'theta_jn'
]
for extrinsic_param in extrinsic_params: 
    priors[extrinsic_param] = injection_parameters[extrinsic_param]

# Additionally, constrain component masses to be +/- 10 solar masses about the injected values, 
# such that we don't run into convergence issues with bilby. 
mMin=5.00*(1.+injection.z)
mMax=88.21*(1.+injection.z)
priors['mass_1'] = Constraint( # detector frame masses           
    name='mass_1',                   
    minimum=max(mMin, injection_parameters['mass_1']-10),                               
    maximum=min(mMax, injection_parameters['mass_1']+10)
)
priors['mass_2'] = Constraint(
    name='mass_2',                               
    minimum=max(mMin, injection_parameters['mass_2']-10),                               
    maximum=min(mMax, injection_parameters['mass_2']+10)
)


# Initialize the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, 
    waveform_generator=waveform_generator, 
    priors=priors, 
    distance_marginalization=False,  # don't need these on since we fix them to the true values
    phase_marginalization=False, 
    time_marginalization=False
)

# Run sampler. In this case we're going to use the `cpnest` sampler
# Note that the maxmcmc parameter is increased so that between each iteration of
# the nested sampler approach, the walkers will move further using an mcmc
# approach, searching the full parameter space.
result = bilby.run_sampler(
    likelihood=likelihood, 
    priors=priors, 
    sampler='dynesty', 
    nlive=2000,
    nact=5,
    npool=8,
    injection_parameters=injection_parameters, 
    outdir=args.outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
)
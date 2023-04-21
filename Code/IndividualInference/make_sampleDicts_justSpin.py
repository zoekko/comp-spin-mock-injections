import numpy as np
import os
import json
import sys

"""
Genearate sampleDict for justSpin bilby outputs 
"""

# Filepath for where bilby outputs saved
#individual_inference_output_folder = '../../Data/IndividualInferenceOutput/'
individual_inference_output_folder = '../../Data/IndividualInferenceOutput/new_bilby/'


# Cycle through the three populations
pop_names = ['population1_highSpinPrecessing_highmass', 'population2_mediumSpin_highmass', 'population3_lowSpinAligned_highmass']
for pop in pop_names: 
    
    print('\nCalculating for '+pop+' ...')

    # Read list of event names for this population
    pop_injlist = np.sort(np.loadtxt(f'injlist_{pop}_100events.txt'))

    sampleDict = {}
    
    # Cycle through events
    for event in pop_injlist: 
        print(str(int(event))+'        ', end='\r')
        
        job_name = "job_{0:05d}_justSpin_result.json".format(int(event))
        fname = individual_inference_output_folder+f'{pop}/'+job_name

        # If the result exists, load in data + format correctly    
        if os.path.exists(fname): 
            
            with open(fname,'r') as jf:
                result = json.load(jf)
            
            try:
                
                # Fetch injected parameters
                injected_params = {
                    'm1':result['injection_parameters']['mass_1_source'],
                    'm2':result['injection_parameters']['mass_2_source'],
                    'z':result['injection_parameters']['redshift'],
                    'chi1':result['injection_parameters']['a_1'],
                    'chi2':result['injection_parameters']['a_2'],
                    'cost1':result['injection_parameters']['cos_tilt_1'],
                    'cost2':result['injection_parameters']['cos_tilt_2'],
                }
                
                # Fetch samples
                chi1 = np.asarray(result['posterior']['content']['a_1'])
                chi2 = np.asarray(result['posterior']['content']['a_2'])
                cost1 =  np.asarray(result['posterior']['content']['cos_tilt_1'])
                cost2 =  np.asarray(result['posterior']['content']['cos_tilt_2'])
            
                # Downsample to 5000 samples per event
                idxs = np.random.choice(len(chi1), size=min(len(chi1),5000))

                sampleDict[str(int(event))] = {
                    'a1':chi1[idxs].tolist(),
                    'a2':chi2[idxs].tolist(),
                    'cost1':cost1[idxs].tolist(),
                    'cost2':cost2[idxs].tolist(),
                    'injected_params':injected_params
                }
                                
            except Exception as e:
                print(e, end='\r')
                sys.exit()
                
        else:
            print(f"event {int(event)} not found")
            
    # Save sampleDict in folder where population inference input goes 
    #output_fname = f'../../Data/PopulationInferenceInput/sampleDict_{pop}_justSpin.json'
    output_fname = f'../../Data/PopulationInferenceInput/sampleDict_{pop}_justSpin_newbilby.json'
    with open(output_fname, 'w') as f:
        json.dump(sampleDict, f)
    
        
        
     
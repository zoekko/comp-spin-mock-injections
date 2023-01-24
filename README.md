# comp-spin-mock-injections

TODO: FLESH OUT THE README WITH DETAILED INSTRUCTIONS FOR HOW TO RUN EVERYTHING

This repository contains all the code to reproduce the results in [insert paper title and name]. None of the specific data used is pushed to the repo because the files are large but we give step by step instructions on how to recreate all data used below. The repo is set up with all the requisite folders / organization. Folders that our scripts write data to currently just contain .tmp files as placeholders. 

## 1. Generate Mock Population Parameters 

Organization:
- Scripts: `Code/GeneratePopulations/`
- Outputs saved: `Data/InjectedPopulationParameters`

The first step to generating mock catalogs of gravitational-wave events is to generate the parameters for each BBH in each population.
First, to generate `.json` files containing the underlying distributions for each of the three populations, run 
```$ python generate_underlying_pops.py``` 
These underlying populations are plotted in Figure 1. 
To then generate the *found* injections for each population, i.e. those from the underlying distributions that pass a network signal-to-noise-ratio threshold of 10, run 
```$ python generate_population1.py``` 
and repeat with `generate_population2.py` and `generate_population3.py`.
Finally, to generate the `.json` file with the sensitivity injections from a flat distribution, which is needed for the selection effects term in population inference (see section 3), run 
```$ python generate_flat_pop_for_injDict.py```

## 2. Perform Individual Event Inference 

Once the parameters for the populations are generated, on the signals generated from the mock population parameters mentioned above using bilby

Organization:
- Scripts: `Code/IndivdualInference/`
- Inputs read from: `Data/InjectedPopulationParameters`
- Outputs saved: `Data/IndividualInferenceOutput` and `Data/PopulationInferenceInput`

## 3. Perform Population Level Inference

using the output individual event posteriors from bilby described above

Organization:
- Scripts: `Code/PopulationInference/`
- Inputs read from: `Data/PopulationInferenceInput`
- Outputs saved: `Data/PopulationInferenceOutput`

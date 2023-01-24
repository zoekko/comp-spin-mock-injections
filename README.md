# comp-spin-mock-injections

TODO: FLESH OUT THE README WITH DETAILED INSTRUCTIONS FOR HOW TO RUN EVERYTHING

This repository contains all the code to reproduce the results in [insert paper title and name]. None of the specific data used is pushed to the repo because the files are large but we give step by step instructions on how to recreate all data used below. The repo is set up with all the requisite folders / organization. Folders that our scripts write data to currently just contain .tmp files as placeholders. 

## 1. Generate Mock Population Parameters 

Organization:
- Scripts: `Code/GeneratePopulations/`
- Outputs saved: `Data/InjectedPopulationParameters`

The first step to generating mock catalogs of gravitational-wave events is to generate the parameters for each BBH in each population.
First, to generate `.json` files containing the underlying distributions for each of the three populations, run 
```
$ python generate_underlying_pops.py
``` 
These underlying populations are plotted in Figure 1. 
To then generate 50,0000 *found* injections for each population, i.e. those from the underlying distributions that pass a network signal-to-noise-ratio  (SNR) threshold of 10, run 
```
$ python generate_population1.py
``` 
and repeat with `generate_population2.py` and `generate_population3.py`. 
Note that this will take hours to run since most randomly generated parameter combinations to not produce signals that pass the SNR cut.
Finally, to generate the `.json` file with the sensitivity injections from a flat distribution, which is needed for the selection effects term in population inference (see section 3), run 
```
$ python generate_flat_pop_for_injDict.py
```

## 2. Perform Individual Event Inference 

Organization:
- Scripts: `Code/IndivdualInference/`
- Inputs read from: `Data/InjectedPopulationParameters`
- Outputs saved: `Data/IndividualInferenceOutput` and `Data/PopulationInferenceInput`

Next, from the 50,000 events we generated from each population, we want to choose a much smaller subset of events that we will inject into LIGO data. These will be our "catalogs" analogous to the actual events LIGO has detected. In the `makeDagFiles.py` and `launchBilby.py` scripts, we select a subset of the 50,000 found events, inject them Gaussian noise realiziations using O3 actual noise PSDs from LIGO Livingston, LIGO Hanford, and Virgo, and use `bilby` to perform parameter estimation on the signals. 

Run
```
$ python makeDagFiles.py
```
to make a text file containing 300 ID numbers of the randomly selected events for each population, and corresponding `.dag` file in the `condor` sub-folder. These files are used to submit all 300 jobs per population to run `bilby` with `HTCondor` on the LIGO computing cluster.
Also, in the `condor` sub-folder are the necessary `.sub` files that submit the `launchBilby.py` script.

*IMPORTANT NOTE* In line 12 of `makeDagFiles.py`, line 16 of `launchBilby.py`, and line 12 of `launchBilby.sh` you will need to change the repository root to be your own.

Once you have this set up, you are ready to submit to condor. To do this, simply run 
```
$ condor_submit_dag bilby_population1_highSpinPrecessing.dag
``` 
for population 1 (and `bilby_population2_mediumSpin.dag` and `bilby_population3_lowSpinAligned.dag` analogously for the others).

Helpful condor commands to monitor your jobs: 
* `condor_q` to see the overview of how many are running
* `condor_q -nobatch -dag` for more detailed info about each job
* `watch condor_q` to wtch the overview of the runs
* If any jobs are held, use `condor_q -hold` to see the reason they're hold. Usually its a memory issue, in which case you can do `condor_qedit JOB# RequestMemory=XXXXXX` and then `condor_release JOB#`. 

Individual event parameter estimation will take days to weeks to run. Once jobs have finished, turn the `bilby` outputs into the correct format to be read into to population inference by running 
```
$ python make_sampleDicts.py
```
Finally, to format the sensitivity injections correctly, run 
```
$ python make_injectionDict_flat.py
```

## 3. Perform Population Level Inference

using the output individual event posteriors from bilby described above

Organization:
- Scripts: `Code/PopulationInference/`
- Inputs read from: `Data/PopulationInferenceInput`
- Outputs saved: `Data/PopulationInferenceOutput`

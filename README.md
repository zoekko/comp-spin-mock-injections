# comp-spin-mock-injections

TODO: FLESH OUT THE README WITH DETAILED INSTRUCTIONS FOR HOW TO RUN EVERYTHING

This repository contains all the code to reproduce the results in [insert paper title and name]. None of the specific data used is pushed to the repo because the files are large but we give step by step instructions on how to recreate all data used below. The repo is set up with all the requisite folders / organization. Folders that our scripts write data to currently just contain .tmp files as placeholders. 

## 1. Generate Mock Population Parameters 

Organization:
- Scripts: `Code/GeneratePopulations/`
- Outputs saved: `Data/InjectedPopulationParameters`

## 2. Perform Individual Event Inference 

on the signals generated from the mock population parameters mentioned above using bilby

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

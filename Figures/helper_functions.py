import numpy as np

def calculate_chiEff(chi1, chi2, cost1, cost2, q): 
    chieff = (chi1*cost1 + q*chi2*cost2)/(1+q)
    return chieff

def calculate_ChiP(chi1, chi2, sint1, sint2, q): 
    term1 = chi1*sint1
    term2 = (2 + 4*q)/(4 + 3*q)*q*chi2*sint2
    chip = np.maximum(term1,term2)
    return chip 
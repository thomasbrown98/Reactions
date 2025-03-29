# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:34:59 2025

@author: 11708
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


species = ['A', 'B']
initial_conc = [1.0, 2]
stoich = [-1, -2]  # reactants consumed, so use negative
k = 0.1
time_span = np.linspace(0, 100, 100)

def elementary_irreversible_rate_law(t, concentrations, k, stoich):
    conc_array = np.clip(np.array(concentrations), 1e-12, None)  # Avoid zero or negative values
    powers = -np.array(stoich)  # Flip signs so powers are positive
    rate = k * np.prod(conc_array ** powers)
    return rate

def elementary_irreversible_reactive_system(conc, t,k,stoich):
    rate = elementary_irreversible_rate_law(t, conc, k, stoich)
    return [rate * coeff for coeff in stoich]

def elementary_reversible_rate_law(t, forward_concentrations, reverse_concentrations, k, Keq, stoich_forward, stoich_reverse):
    forward_conc_array = np.clip(np.array(forward_concentrations), 1e-12, None)
    reverse_conc_array = np.clip(np.array(reverse_concentrations), 1e-12, None)
    forward_powers = np.array(stoich_forward)
    reverse_powers = np.array(stoich_reverse)
    
    rate_forward = np.prod(forward_conc_array ** forward_powers)
    rate_reverse = np.prod(reverse_conc_array ** reverse_powers)
    
    rate = k * (rate_forward - rate_reverse / Keq)
    return rate


def solve_reactive_system(
    elementary_irreversible_reactive_system, 
    initial_conc, 
    time_span, 
    k, 
    stoich, 
    plot=False, 
    species_names=None
):
    results = odeint(
        elementary_irreversible_reactive_system,
        initial_conc,
        time_span,
        args=(k, stoich)
    )
    
    if plot:
        num_species = len(initial_conc)
        if species_names is None:
            species_names = [f"Species {i+1}" for i in range(num_species)]
        
        for i in range(num_species):
            plt.plot(time_span, results[:, i], label=species_names[i])
        
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.title("Concentration vs. Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return results

def fit_reaction(model, time, initial_concentration, initial_guesses):
    p_opt, pcov = curve_fit(
        model,
        time,
        initial_concentration,
        p0=initial_guesses,
        bounds=([0, 0], [np.inf, np.inf])
    )
    return p_opt

concentrations = solve_reactive_system(elementary_irreversible_reactive_system, initial_conc, time_span, k, stoich, plot = True, species_names= species)

'''
def second_order(t, CA, k, C_limiting=None):
    if C_limiting is None:
        return -k * CA**2
    else:
        k_prime = k * CA
        return k_prime * C_limiting  # pseudo first-order version

def langmuir_singlespecies(t, CA, k_prime, K):
    return (-k_prime * CA) / (1 + K * CA)




def fit_reaction(model, time, concentration, initial_guesses):
    p_opt, pcov = curve_fit(
        model,
        time,
        concentration,
        p0=initial_guesses,
        bounds=([0, 0], [np.inf, np.inf])
    )
    return p_opt


model = make_pseudo_first_order_model(C_limiting=0.25)
k = fit_reaction(model, t_data, C_B_noisy, [0.01,8])


stoich = [1,2]
concentrations = [1,3]

for conc in concentrations:
    exp_conc = concentrations**(stoich)
'''   
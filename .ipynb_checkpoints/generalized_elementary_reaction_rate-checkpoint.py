# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:34:59 2025

@author: 11708
"""


'''
species = ['A', 'B']
initial_conc = [1.0, 2]
stoich = [-1, -2]  # reactants consumed, so use negative
k = 0.1
time_span = np.linspace(0, 100, 100)
'''
def generate_synthetic_data(k_true, stoich, initial_conc, time_span, noise_level=0.01):
    import numpy as np
    clean_data = solve_reactive_system(
        elementary_irreversible_reactive_system,
        initial_conc,
        time_span,
        k_true,
        stoich
    )
    
    noisy_data = clean_data + np.random.normal(0, noise_level, clean_data.shape)
    return noisy_data


def elementary_irreversible_rate_law(t, concentrations, k, stoich):
    import numpy as np
    conc_array = np.clip(np.array(concentrations), 1e-12, None)  # Avoid zero or negative values
    powers = -np.array(stoich)  # Flip signs so powers are positive
    rate = k * np.prod(conc_array ** powers)
    return rate

def elementary_irreversible_reactive_system(conc, t,k,stoich):
    rate = elementary_irreversible_rate_law(t, conc, k, stoich)
    return [rate * coeff for coeff in stoich]

def elementary_reversible_rate_law(t, forward_concentrations, reverse_concentrations, k, Keq, stoich_forward, stoich_reverse):
    import numpy as np
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
    from scipy.integrate import odeint
    
    results = odeint(
        elementary_irreversible_reactive_system,
        initial_conc,
        time_span,
        args=(k, stoich)
    )
    
    if plot:
        import matplotlib.pyplot as plt
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

def reaction_model_to_fit(t, k, stoich, initial_conc):
    from scipy.integrate import odeint
    result = odeint(
        elementary_irreversible_reactive_system,
        initial_conc,
        t,
        args=(k, stoich)
    )
    return result[:, 0]  # Fit to species A (you can change this)



def fit_reaction_to_data(t, data, stoich, initial_conc, k_guess=0.1):
    from scipy.optimize import curve_fit
    def wrapper(t, k):
        return reaction_model_to_fit(t, k, stoich, initial_conc)
    
    popt, pcov = curve_fit(wrapper, t, data, p0=[k_guess])
    return popt[0], pcov

  
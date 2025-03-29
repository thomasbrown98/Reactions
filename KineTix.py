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
    """
    Generate synthetic concentration data with added Gaussian noise for an elementary irreversible reaction.

    Parameters:
        k_true (float): The true rate constant used for data generation.
        stoich (list of float): Stoichiometric coefficients of the reactants (negative values).
        initial_conc (list of float): Initial concentrations of the species.
        time_span (array-like): Time points at which the system is evaluated.
        noise_level (float): Standard deviation of the Gaussian noise added to the clean data.

    Returns:
        np.ndarray: Simulated concentration data with added noise.
    """
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
    """
    Calculate the reaction rate for an elementary irreversible reaction.

    Parameters:
        t (float): Time (unused but required for ODE solver compatibility).
        concentrations (list or np.ndarray): Current concentrations of reactants.
        k (float): Rate constant.
        stoich (list of float): Stoichiometric coefficients (negative for reactants).

    Returns:
        float: Reaction rate based on the rate law.
    """
    import numpy as np
    conc_array = np.clip(np.array(concentrations), 1e-12, None)  # Avoid zero or negative values
    powers = -np.array(stoich)  # Flip signs so powers are positive
    rate = k * np.prod(conc_array ** powers)
    return rate


def elementary_irreversible_reactive_system(conc, t, k, stoich):
    """
    Define the system of ODEs for an elementary irreversible reaction.

    Parameters:
        conc (list or np.ndarray): Concentrations at time t.
        t (float): Current time (required by odeint).
        k (float): Rate constant.
        stoich (list of float): Stoichiometric coefficients.

    Returns:
        list: Derivatives of concentrations with respect to time.
    """
    rate = elementary_irreversible_rate_law(t, conc, k, stoich)
    return [rate * coeff for coeff in stoich]


def elementary_reversible_rate_law(t, forward_concentrations, reverse_concentrations, k, Keq, stoich_forward, stoich_reverse):
    """
    Calculate the net reaction rate for an elementary reversible reaction.

    Parameters:
        t (float): Time (unused).
        forward_concentrations (list or np.ndarray): Concentrations of forward reactants.
        reverse_concentrations (list or np.ndarray): Concentrations of reverse reactants.
        k (float): Forward rate constant.
        Keq (float): Equilibrium constant.
        stoich_forward (list of float): Stoichiometric coefficients of forward reactants.
        stoich_reverse (list of float): Stoichiometric coefficients of reverse reactants.

    Returns:
        float: Net reaction rate.
    """
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
    """
    Solve the system of ODEs for a chemical reaction and optionally plot the results.

    Parameters:
        elementary_irreversible_reactive_system (function): Function defining the ODE system.
        initial_conc (list of float): Initial concentrations of the species.
        time_span (array-like): Time points for evaluation.
        k (float): Rate constant.
        stoich (list of float): Stoichiometric coefficients.
        plot (bool): If True, plots concentration vs. time.
        species_names (list of str): Names of species for plotting.

    Returns:
        np.ndarray: Concentration profiles for all species over time.
    """
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
    """
    Wrapper function that solves the ODE system and returns the concentration of a single species for curve fitting.

    Parameters:
        t (array-like): Time points for evaluation.
        k (float): Rate constant to fit.
        stoich (list of float): Stoichiometric coefficients.
        initial_conc (list of float): Initial concentrations.

    Returns:
        np.ndarray: Concentration of species A (index 0) over time.
    """
    from scipy.integrate import odeint
    result = odeint(
        elementary_irreversible_reactive_system,
        initial_conc,
        t,
        args=(k, stoich)
    )
    return result[:, 0]  # Fit to species A (you can change this)


def fit_reaction_to_data(t, data, stoich, initial_conc, k_guess=0.1):
    """
    Fit the rate constant k to experimental or synthetic concentration data using nonlinear least squares.

    Parameters:
        t (array-like): Time points corresponding to the data.
        data (array-like): Measured concentrations of species A.
        stoich (list of float): Stoichiometric coefficients.
        initial_conc (list of float): Initial concentrations of the species.
        k_guess (float): Initial guess for the rate constant.

    Returns:
        tuple: Fitted rate constant (float) and the covariance matrix (2D array).
    """
    from scipy.optimize import curve_fit

    def wrapper(t, k):
        return reaction_model_to_fit(t, k, stoich, initial_conc)
    
    popt, pcov = curve_fit(wrapper, t, data, p0=[k_guess])
    return popt[0], pcov

  
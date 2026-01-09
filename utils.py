import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class TT_params(object) : 
    """
    Parameters for simulations of T-cell clone growth in presence of antigens and with
    TT inhibition
    """
    def __init__(self, beta, gamma, lambd, P0, mu, alpha):
        # Rate of conversion from MHC bind to TCR growth
        self.beta = beta
        # TCR death rate
        self.gamma = gamma
        # Rate of aquisition of MHC from TCRs
        self.lambd = lambd
        # Number of initial MHC
        self.P0 = P0
        # Degradation rate MHC
        self.mu = mu
        # Inibition-factor growth rate constant
        self.alpha = alpha

    def print_on_file(self, path):
        sr = pd.Series({
            'beta':self.beta, 
            'gamma':self.gamma, 
            'lambda':self.lambd, 
            'P0':self.P0, 
            'mu':self.mu, 
            'alpha':self.alpha
        })
        sr = sr[sr.notna()]
        sr.to_csv(path, sep='\t', header=None)
        


def nsolve(init_vars, vars_dots, pars, t_steps, dt, traj_steps=10):
    """
    Numerical solver for a dynamical system of equations.
    
    Parameters
    ----------
    init_vars (list): initial conditions for each of the variables
    vars_dots: list of dot equations for each of the variables. The dot equation
        is a function of the list of variables and the parameters
    pars: parameters to be passed to the vars_dots functions
    t_steps (int): Number of time steps
    dt (float): infinitesimal length of the time step
    traj_steps (int): the values of the variables are stored every 'traj_steps'

    Notes: the type of the variables is free and need to be consistent with the
    initial conditions and the dot equations. E.g. can be a float or a list of 
    floats.
    
    Returns
    -------
    trajs (list of list): Trajectories of the variables stored every 'traj_steps'.
        First dimension is time, second dimension is the variable space.
    """
    
    _vars = [np.copy(var) for var in init_vars]
    trajs = []
    for ti in range(t_steps):
        dots = [vars_dot(_vars, pars) for vars_dot in vars_dots]
        for i in range(len(_vars)):
            _vars[i] += (np.array(dots[i]) * dt)
        if ti % traj_steps == 0:
            trajs.append([np.copy(var) for var in _vars])
    return trajs
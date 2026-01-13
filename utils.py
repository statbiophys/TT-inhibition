import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm


class TT_params(object) : 
    """
    Parameters for simulations of T-cell clone growth in presence of antigens and with
    TT inhibition
    """
    def __init__(self, taus, beta0=1, tau_crit=1, gamma=0.03, lambd=0.001, P0=1.0, mu=0.03, alpha0=5e-4):
        
        # List of dissotiation constants for all the T-cells in simulation
        self.taus = np.array(taus)
        # Rate of conversion from MHC bind to TCR growth
        self.beta0 = beta0
        # Typical time of activation
        self.tau_crit = tau_crit
        # T cell death rate
        self.gamma = gamma
        # Rate of aquisition of MHC from TCRs
        self.lambd = lambd
        # Number of initial MHC-presented antigens
        self.P0 = P0
        # Antigen degradation rate
        self.mu = mu
        # Inibition-factor growth rate constant
        self.alpha0 = alpha0

        # Number of T cells
        self.n = len(taus)
        # Activation rates for all the T cells
        self.betas = self.beta0 * np.exp(- self.tau_crit / self.taus)
        self.mean_beta = np.mean(self.betas)
        # Inhibition rates
        self.alphas = self.alpha0 * self.taus
        self.mean_alpha = np.mean(self.alphas)
        

    def print_on_file(self, folder, file_name, other_pars={}):
        """
        Print the parameters on a tsv file at "path". Other parameters can be added 
        to the file if passed in other_pars dictionary.
        """
        sr = pd.Series({
            'beta0':self.beta0, 
            'tau_crit':self.tau_crit, 
            'gamma':self.gamma, 
            'lambda':self.lambd, 
            'P0':self.P0, 
            'mu':self.mu, 
            'alpha0':self.alpha0,
            **other_pars
        })
        sr = sr[sr.notna()]
        sr.to_csv(folder+file_name+'.tsv', sep='\t', header=None)
        

def sample_taus_lognorm(logmean=-4, logstd=2.5, n_samples=100):
    """
    Return n_samples from a lognormal distributio having given log mean and log
    standard deviation
    """
    ln = lognorm(s=logstd, scale=np.exp(logmean))
    return ln.rvs(n_samples)


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


### FULL SYSTEM EQUATIONS

def Ts_dot(var, pars):
    Ts, P, Ss = var
    res = Ts * ( pars.betas * P * (1 - Ss) - pars.gamma )
    #res[Ts < 1] = 0
    return res

def Ss_dot(var, pars):
    Ts, P, Ss = var
    res = pars.alphas * np.sum(Ts)
    res[Ss >= 1] = 0
    return res

def P_dot(var, pars):
    Ts, P, Ss = var
    return - P * ( pars.lambd * np.sum(pars.betas * (1 - Ss) * Ts) + pars.mu )
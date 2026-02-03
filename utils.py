import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm


class TT_params(object) : 
    """
    Parameters for simulations of T-cell clone growth in presence of antigens and with
    TT inhibition
    """
    def __init__(self, taus, beta0=2, tau_crit=1, gamma=0.05, lambd=0.001, P0=1.0, mu=0.05, alpha0=5e-4, inhib_T_th=2):
        
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
        # Threshold of T cell abundance above which inhibition starts
        self.inhib_T_th = inhib_T_th
        # List of dissotiation constants for all the T-cells in simulation
        self.set_taus(np.array(taus))
        

    def set_taus(self, taus):
        self.taus = np.array(taus)
        # Number of T cells
        self.n = len(taus)
        # Activation rates for all the T cells
        self.betas = self.beta0 * np.exp(- self.tau_crit / self.taus)
        self.mean_beta = np.mean(self.betas)
        # Inhibition rates
        self.alphas = self.alpha0 * self.taus
        self.mean_alpha = np.mean(self.alphas)


    def get_pars(self):
        """
        Return the parameters as a dictionary
        """
        return {
            'beta0':self.beta0, 
            'tau_crit':self.tau_crit, 
            'gamma':self.gamma, 
            'lambda':self.lambd, 
            'P0':self.P0, 
            'mu':self.mu, 
            'alpha0':self.alpha0,
            'inhib_T_th':self.inhib_T_th,
        }
        
    def print_on_file(self, folder, file_name, other_pars={}):
        """
        Print the parameters on a tsv file at "path". Other parameters can be added 
        to the file if passed in other_pars dictionary.
        """
        sr = pd.Series({**self.get_pars(), **other_pars})
        sr = sr[sr.notna()]
        sr.to_csv(folder+'/'+file_name+'.tsv', sep='\t', header=None)


class tau_sampler_lognorm(object):
    """
    Generate n_samples from a lognormal distribution having given log mean and log
    standard deviation in the variable taus
    """
    def __init__(self, logmean=-4, logstd=2.5, n_samples=100):
        self.logmean = logmean
        self.logstd = logstd
        self.n_samples = n_samples
        self.ln = lognorm(s=logstd, scale=np.exp(logmean))
        self.sample()

    def sample(self):
        self.taus = self.ln.rvs(self.n_samples)
        
    def get_pars(self):
        """
        Return the parameters as a dictionary
        """
        return {
            'logmean':self.logmean, 
            'logstd':self.logstd, 
            'n_samples':self.n_samples
        }

    def get_av_std_max(self):
        """
        Average and standard deviation of a maximum sampled from the lognormal
        """
        lN, lm, lstd = np.log(self.n_samples), self.logmean, self.logstd
        av = np.exp(lm + lstd*(np.sqrt(2*lN) - (np.log(lN) + np.log(4*np.pi))/(2*np.sqrt(2*lN))))
        std = np.pi * lstd * av / np.sqrt(12 * lN)
        return av, std

        
def nsolve(init_vars, vars_dots, pars, t_steps, dt, stop_cond=None, traj_steps=10):
    """
    Numerical solver for a dynamical system of equations.
    
    Parameters
    ----------
    init_vars (list): initial conditions for each of the variables.
    vars_dots: list of dot equations for each of the variables. The dot equation
        is a function of the list of variables and the parameters.
    pars: parameters to be passed to the vars_dots functions.
    t_steps (int): Number of time steps. If stop_cond is specified, this is the maximum
        number of steps after which the simulation is stopped anyway.
    dt (float): infinitesimal length of the time step.
    traj_steps (int): the values of the variables are stored every 'traj_steps'.
    stop_cond (boolean funct of the trajs and pars): if not None it is tested at the end of
        a batch of t_steps and the simulations goes on with another batch until the condition
        is satisfied or after max_iterations.
    max_iterations (int): maximum number of iteration of the stop condition
        (not used if stop_cond=None).

    Notes: the type of the variables is free and need to be consistent with the
    initial conditions and the dot equations. E.g. can be a float or a list of 
    floats.
    
    Returns
    -------
    times (list of float): time at which the trajectory is stored
    trajs (list of list): Trajectories of the variables stored every 'traj_steps'.
        First dimension is time, second dimension is the variable space.
    """
    
    _vars = [np.copy(var) for var in init_vars]
    trajs, times = [], []
    if stop_cond is None: stop_cond = lambda x, y : False # Stop_cond always true if None
        
    for ti in range(t_steps):
        dots = [vars_dot(_vars, pars) for vars_dot in vars_dots]
        for i in range(len(_vars)): _vars[i] += (np.array(dots[i]) * dt)
        if ti % traj_steps == 0:
            times.append(ti*dt)
            trajs.append([np.copy(var) for var in _vars])
            if stop_cond(trajs, pars): # Stop condition tested at end of batch
                break
    
    return times, trajs

        
def dt_adapted(pars, tau_sampler, scale_factor=100):
    """
    Finding an estimate of dt as scale_factor times smaller than the shorter
    time scale of the system
    """
    av_max_tau, std_max_tau = tau_sampler.get_av_std_max()
    av_max_alpha = pars.alpha0 * (av_max_tau + std_max_tau)
    scales = [1/av_max_alpha, 1/pars.beta0]
    dt = np.min(scales) / scale_factor
    return min(dt, 0.01)
    
    
### FULL SYSTEM EQUATIONS

def Ts_dot(var, pars):
    Ts, P, Ss = var
    res = Ts * ( pars.betas * P * (1 - Ss) - pars.gamma )
    #res[Ts < 1] = 0
    return res

def Ss_dot(var, pars):
    Ts, P, Ss = var
    exper_T_mask = Ts >= pars.inhib_T_th
    res = pars.alphas * np.sum(Ts[exper_T_mask])
    res[Ss >= 1] = 0
    res[~exper_T_mask] = 0
    return res

def P_dot(var, pars):
    Ts, P, Ss = var
    return - P * ( pars.lambd * np.sum(pars.betas * (1 - Ss) * Ts) + pars.mu )


def run_setting(pars, tau_sampler, t_steps, dt, stop_cond=None, traj_steps=1):
    """
    It runs the numerical integration of the full setting given the TT_params and the 
    integration parameters: t_steps (number of time steps), dt (infinitesimal time
    step), traj_steps (after how many steps the trajectories are saved).
    After t_steps the stop_condition (function of trajectories and parameters) is tested 
    and the result is returned only if satisfied. Otherwise the simulation goes on for 
    other t_steps until success or the number of max_iterations is reached.
    stop_cond=None does not test any condition and the simulation stops after t_steps
    """

    if dt == 'adapt': dt = dt_adapted(pars, tau_sampler, scale_factor=50)
        
    # Initial conditions of T (list for each clone), P and S (list for each clone)
    init_vars = [np.ones(len(pars.taus)), pars.P0, np.zeros(len(pars.taus))]
    
    # Dot equations for T, P and S
    dot_eqs = [Ts_dot, P_dot, Ss_dot]

    # Integrating the trajectories
    times, trajs = nsolve(init_vars, dot_eqs, pars, t_steps, dt, stop_cond, traj_steps)
    
    # Defining useful variables
    T_trajs = np.array([tr[0] for tr in trajs])
    P_trajs = np.array([tr[1] for tr in trajs])
    S_trajs = np.array([tr[2] for tr in trajs])

    return times, T_trajs, P_trajs, S_trajs


### OTHER GENERIC UTILITIES

def binning_x(xs, ys, x_bins):
    """
    Given the coordinates x and y, it returns averages and standard deviations 
    in a binning over the x-axis defined in x_bins
    """
    av_x, av_y = [], []
    std_x, std_y = [], []
    for i in range(len(x_bins)-1):
        mask = np.logical_and(xs >= x_bins[i], xs < x_bins[i+1])
        if mask.sum() > 1:
            av_x.append(np.mean(xs[mask]))
            av_y.append(np.mean(ys[mask]))
            std_x.append(np.std(xs[mask]))
            std_y.append(np.std(ys[mask]))
    return np.array(av_x), np.array(av_y), np.array(std_x), np.array(std_y)


def bisection(function, a, b, tol):
    """
    Bisection method for finding the zero of the function in the range [a,b] with 
    given tollerance
    """
    fa, fb = function(a), function(b)
    if fa * fb >= 0:
        print("You have not assumed right a and b in bisection:")
        print(f"f(a)={fa}, f(b)={fb}")
        return a
    c = a
    while (b-a) >= tol:
        #Find middle point
        c = (a+b)/2
        fc = function(c)
        #print(fc)
        #Check if middle point is root
        if fc == 0.0: break
        #Decide the side to repeat the steps
        else:
            if fc*fa < 0:
                b = c
            else:
                a = c
                fa = fc
    return c

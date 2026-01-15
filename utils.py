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


def nsolve(init_vars, vars_dots, pars, t_steps, dt, stop_cond=None, max_iterations=20, traj_steps=10):
    """
    Numerical solver for a dynamical system of equations.
    
    Parameters
    ----------
    init_vars (list): initial conditions for each of the variables.
    vars_dots: list of dot equations for each of the variables. The dot equation
        is a function of the list of variables and the parameters.
    pars: parameters to be passed to the vars_dots functions.
    t_steps (int): Number of time steps.
    dt (float): infinitesimal length of the time step.
    traj_steps (int): the values of the variables are stored every 'traj_steps'.
    stop_cond (boolean funct of the vars and pars): if not None it is tested at the end of
        t_steps and the simulations goes on in blocks of t_steps until the condition
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
    if stop_cond is None: stop_cond = lambda x, y : True # Stop_cond always true if None
        
    for n_iter in range(max_iterations): # Iterations over stop_condition
        
        for ti_at_iter in range(t_steps):
            ti = n_iter*t_steps + ti_at_iter
            dots = [vars_dot(_vars, pars) for vars_dot in vars_dots]
            for i in range(len(_vars)):
                _vars[i] += (np.array(dots[i]) * dt)
            if ti % traj_steps == 0:
                times.append(ti*dt)
                trajs.append([np.copy(var) for var in _vars])

        if stop_cond(_vars, pars): break # Stop condition tested
                
    return times, trajs


### SAMPLING TAUS

def sample_taus_lognorm(logmean=-4, logstd=2.5, n_samples=100):
    """
    Return n_samples from a lognormal distributio having given log mean and log
    standard deviation
    """
    ln = lognorm(s=logstd, scale=np.exp(logmean))
    return ln.rvs(n_samples)

def av_max_lognorm(logmean=-4, logstd=2.5, n_samples=100):
    """
    Average maximum sampled from a lognormal
    """
    lN = np.log(n_samples)
    return np.exp(logmean + logstd*(np.sqrt(2*lN) - (np.log(lN) + np.log(4*np.pi))/(2*np.sqrt(2*lN))))

    
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


def run_setting(pars, t_steps, dt, stop_cond=None, max_iterations=20, traj_steps=1):
    """
    It runs the numerical integration of the full setting given the TT_params and the 
    integration parameters: t_steps (number of time steps), dt (infinitesimal time
    step), traj_steps (after how many steps the trajectories are saved).
    After t_steps the stop_condition (function of trajectories and parameters) is tested 
    and the result is returned only if satisfied. Otherwise the simulation goes on for 
    other t_steps until success or the number of max_iterations is reached.
    stop_cond=None does not test any condition and the simulation stops after t_steps
    """
    
    # Initial conditions of T (list for each clone), P and S (list for each clone)
    init_vars = [np.ones(len(pars.taus)), pars.P0, np.zeros(len(pars.taus))]
    
    # Dot equations for T, P and S
    dot_eqs = [Ts_dot, P_dot, Ss_dot]

    # Integrating the trajectories
    times, trajs = nsolve(init_vars, dot_eqs, pars, t_steps, dt, stop_cond, max_iterations, traj_steps)
    
    # Defining useful variables
    T_trajs = np.array([tr[0] for tr in trajs])
    P_trajs = np.array([tr[1] for tr in trajs])
    S_trajs = np.array([tr[2] for tr in trajs])

    return times, T_trajs, P_trajs, S_trajs


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
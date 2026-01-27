import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm
import utils as ut


def stop_cond_memory(trajs, pars):
    """
    Stop condition of having the positive growth rate theta times smaller 
    than gamma
    """
    Ts, P, Ss = trajs[-1]
    growth_rates = pars.betas * P * (1 - Ss)
    gr_small_than_th = np.sum(growth_rates / pars.gamma < pars.theta_stop)
    return gr_small_than_th >= pars.n


def stop_cond_peak(trajs, pars):
    """
    Stop condition of being at the stationary point of total T cell abundance
    """
    if len(trajs) < 2: return False
    Ts1, _, _ = trajs[-1]
    Ts0, _, _ = trajs[-2]
    delta_T_tot = np.sum(Ts1) - np.sum(Ts0)
    return delta_T_tot < 0


def generate_ensamble(R, tau_sampler, t_steps, dt, traj_steps, theta_stop, **params):
    """
    Generating an ensamble of R simulations with a stop condition in the "memory regime" defined by theta_stop,
    and collecting the final T frequencies and taus of the surviving clones.
    """
    
    # Initializing parameters with default values or corrected through **params
    pars = ut.TT_params([1], **params)
    pars.theta_stop = theta_stop

    progress = tqdm(total=R)
    taus_mat, Tfreqs_mat, T_tot_mat, times_mat = [], [], [], []
    for _ in range(R):
        tau_sampler.sample()
        pars.set_taus(tau_sampler.taus)
        times, T_trajs, _, _ = ut.run_setting(pars, tau_sampler, t_steps, dt, traj_steps=traj_steps, stop_cond=stop_cond_memory)
    
        T_tot_mat.append(np.sum(T_trajs, axis=1))
        times_mat.append(times)
        no_extinct = T_trajs[-1] >= 1
        freqs = T_trajs[-1] / T_trajs[-1].sum()
        Tfreqs_mat.append(freqs[no_extinct])
        taus_mat.append(tau_sampler.taus[no_extinct])
        progress.update(1)
        
    return taus_mat, Tfreqs_mat, T_tot_mat, times_mat


def find_first_zero(xs, ys):
    for i in range(1, len(xs)):
        x0, y0, x1, y1 = xs[i-1], ys[i-1], xs[i], ys[i]
        if y0*y1 < 0:
            # Linear interpolation to get the zero
            x_max = x1 + y1*(x0-x1)/(y1-y0)
            return x_max
    return float('nan')


def find_max(taus_mat, Tfreqs_mat, smooth_bins=30):
    """
    Method for detecting an average maximum in an ensemble of tau-freq curves
    """
    if any(np.array([len(t) < 2 for t in taus_mat])):
        return float('nan'), float('nan'), 0, 'extinct'
    
    # Moving to the log transformed variables, sorting them and taking the derivative
    logtaus, logTfreq = [np.log10(t) for t in taus_mat], [np.log10(t) for t in Tfreqs_mat]
    tau_sort_i = [np.argsort(t) for t in logtaus]
    logtaus = [logtaus[i][tau_sort_i[i]] for i in range(len(logtaus))]
    logTfreq = [logTfreq[i][tau_sort_i[i]] for i in range(len(logTfreq))]
    ders = [(logTfreq[i][1:] - logTfreq[i][:-1]) / (logtaus[i][1:] - logtaus[i][:-1]) for i in range(len(logtaus))]
    mid_logtaus = [(logtaus[i][1:] + logtaus[i][:-1]) / 2.0 for i in range(len(logtaus))]

    # Computing the average trajectory
    bins = np.linspace(np.min(np.concatenate(mid_logtaus)), np.max(np.concatenate(mid_logtaus)), smooth_bins)
    smooth_x, smooth_y, _, _ = ut.binning_x(np.concatenate(mid_logtaus), np.concatenate(ders), bins)

    # Finding the zero of the smoothed trajectory
    x_max = find_first_zero(smooth_x, smooth_y)
    if np.isnan(x_max):
        return x_max, float('nan'), 0, 'no_maximum'

    # Counting how many trajectories cross the average maximum and are decreasing at the end
    logtaus_end = np.array([t[-1] for t in logtaus])
    ders_end = np.array([d[-1] for d in ders])
    max_mask = np.logical_and(logtaus_end > x_max, ders_end < 0)
    n_x_max = np.sum(max_mask)
    
    # Estimating the average y at max
    y_maxs = []
    for r in range(len(taus_mat)):
        if max_mask[r]:
            y_maxs.append(np.interp(x_max, logtaus[r], logTfreq[r]))
    y_max = np.mean(y_maxs)
    
    return x_max, y_max, n_x_max, 'maximum'


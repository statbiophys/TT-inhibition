import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
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


def generate_ensamble(R, tau_sampler, t_steps, dt, traj_steps, theta_stop, print_progress=True, **params):
    """
    Generating an ensamble of R simulations with a stop condition in the "memory regime" defined by theta_stop,
    and collecting the final T frequencies and taus of the surviving clones.
    """

    # Initializing parameters with default values or corrected through **params
    pars = ut.TT_params([1], **params)
    pars.theta_stop = theta_stop

    if print_progress: progress = tqdm(total=R)
    taus_mat, Tfreqs_mat, T_tot_mat, times_mat = [], [], [], []
    for _ in range(R):
        tau_sampler.sample()
        pars.set_taus(tau_sampler.taus)
        times, T_trajs, _, _ = ut.run_setting(pars, tau_sampler, t_steps, dt, traj_steps=traj_steps, stop_cond=stop_cond_memory)
    
        T_tot_mat.append(np.sum(T_trajs, axis=1))
        times_mat.append(times)
        #no_extinct = T_trajs[-1] >= 1
        freqs = T_trajs[-1]
        Tfreqs_mat.append(freqs)
        taus_mat.append(tau_sampler.taus)
        if print_progress: progress.update(1)
        
    return taus_mat, Tfreqs_mat, T_tot_mat, times_mat


def find_first_zero(xs, ys):
    for i in range(1, len(xs)):
        x0, y0, x1, y1 = xs[i-1], ys[i-1], xs[i], ys[i]
        if y0*y1 < 0:
            # Linear interpolation to get the zero
            x_max = x1 + y1*(x0-x1)/(y1-y0)
            return x_max
    return float('nan')


def find_max(taus_mat, Ts_mat, smooth_bins=100, quantile_taus=0.998):
    """
    Method for detecting an average maximum in an ensemble of tau-freq curves
    """
    if any(np.array([len(t) < 2 for t in taus_mat])):
        return float('nan'), float('nan'), 0, 'extinct'
    
    # Moving to the log transformed variables, sorting them and taking the derivative
    logtaus, logT = [np.log10(t) for t in taus_mat], [np.log10(t) for t in Ts_mat]
    tau_sort_i = [np.argsort(t) for t in logtaus]
    logtaus = [logtaus[i][tau_sort_i[i]] for i in range(len(logtaus))]
    logT = [logT[i][tau_sort_i[i]] for i in range(len(logT))]
    ders = [(logT[i][1:] - logT[i][:-1]) / (logtaus[i][1:] - logtaus[i][:-1]) for i in range(len(logtaus))]
    mid_logtaus = [(logtaus[i][1:] + logtaus[i][:-1]) / 2.0 for i in range(len(logtaus))]

    # Computing the average trajectory
    all_taus = np.concatenate(mid_logtaus)
    quantile_mask = all_taus <= np.quantile(all_taus, quantile_taus)
    all_taus = all_taus[quantile_mask] # Removing largest tau for outliers
    all_ders = np.concatenate(ders)[quantile_mask]
    bins = np.linspace(np.min(all_taus), np.max(all_taus), smooth_bins)
    smooth_x, smooth_y, _, _ = ut.binning_x(all_taus, all_ders, bins)

    # Finding the zero of the smoothed trajectory
    x_max = find_first_zero(smooth_x[::-1], smooth_y[::-1])
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
            y_maxs.append(np.interp(x_max, logtaus[r], logT[r]))
    y_max = np.mean(y_maxs)
    
    return x_max, y_max, n_x_max, 'maximum'


def compute_inv_simps(Ts_mat):
    """
    Compute the inverse Simpson's index for a list of audaces arrays.
    It discards abundances smaller than 1.
    """
    T_freqs_mat = [T[T >= 1] / np.sum(T[T >= 1]) for T in Ts_mat]
    inv_simps = np.array([1 / np.sum(freqs**2) for freqs in T_freqs_mat])
    inv_simps = inv_simps[~np.isnan(inv_simps)]
    inv_simps = inv_simps[~np.isinf(inv_simps)]
    return np.mean(inv_simps)


def add_phases(ax, outcome_mat, extent, color1='#033500', color2='k', add_increasing=True, legend=False):
    """Add phase regions to a given axis."""
    cm1 = plt.cm.colors.ListedColormap([color1])
    cm2 = plt.cm.colors.ListedColormap([color2])
    aux_mat = np.where(outcome_mat == 'extinct', 1.0, float('nan'))
    ax.imshow(aux_mat[::-1], cmap=cm1, vmin=0, extent=extent, aspect='auto')

    if add_increasing:
        aux_mat = np.where(outcome_mat == 'no_maximum', 1.0, float('nan'))
        ax.imshow(aux_mat[::-1], cmap=cm2, vmin=0, extent=extent, aspect='auto')

    patch1 = patches.Patch(color=color1, label='no expansion')
    patch2 = patches.Patch(color=color2, label='only increasing')
    if legend:
        ax.legend(handles=[patch1, patch2], bbox_to_anchor=(0.05, 0.05), loc=3, borderaxespad=0.)
    
    return ax


import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import utils_string as uts
import utils as ut


def get_av_t_Tmax(alpha0, beta0, logtau_ens, R):

    ts = []
    for _ in range(R):

        logtaus = logtau_ens[np.random.randint(0, len(logtau_ens))]
        pars = ut.TT_params(np.exp(logtaus), alpha0=alpha0, beta0=beta0)
        times, T_trajs, _, _ = ut.run_setting(pars, ut.tau_sampler_lognorm(), 5000, 'adapt', traj_steps=50)

        # Tricky computation of max expansion because of trajectories that get extinct (for low alpha0)
        # We remove from computation trajectories that were never above 1
        extinct_traj = np.all(T_trajs <= 1, axis=0)
        T_tot = np.sum(T_trajs[:,~extinct_traj], axis=1)
        ts.append(times[np.argmax(T_tot)])

    return np.mean(ts)


out_dir = 'out_data/string_model/'
str_name = 'gamma=1.0_L=40'
mean_logtau, std_logtau = -2, 2
t_bounds = [0.4, 1.8]
R_beta0 = 10
t_T_max = 9
t_T_max_tol = 0.05
alpha0s = np.logspace(-5.5, -3, 4)

name = str_name + f'_meanlt={mean_logtau}_stdlt={std_logtau}'


with open(out_dir + f'string_ens_{str_name}.pkl', 'rb') as f:
    string_ens = pickle.load(f)

pars = pd.read_csv(out_dir + f'string_pars_{str_name}.tsv', sep='\t', header=None, index_col=0)[1].to_dict()
string_model = uts.TT_string(N_tot=pars['N_tot'], L_string=int(pars['L_string']), N_recruit=pars['N_recruit'])


logtau_ens = []
for i, strings in enumerate(string_ens):
    h_dists = uts.comp_h_dist(strings, np.zeros(len(strings[0])))
    logtau_ens.append(string_model.get_logtau(h_dists, mean_logtau, std_logtau))


beta0s = []
progress = tqdm(total=len(alpha0s))
for a0 in alpha0s:
    func = lambda x : get_av_t_Tmax(a0, x, logtau_ens, R_beta0) - t_T_max
    beta0s.append(ut.bisection(func, t_bounds[0], t_bounds[1], t_T_max_tol))
    progress.update(1)

with open(out_dir + f'a0s_b0s_{name}.pkl', 'wb') as f:
    pickle.dump((alpha0s, beta0s), f)

# Export params as well
pars = pd.read_csv(out_dir + f'string_pars_{str_name}.tsv', sep='\t', index_col=0, header=None)
sr = pd.Series({
    'mean_logtau':mean_logtau, 
    'std_logtau':std_logtau, 
    'R_beta0':R_beta0, 
    't_T_max':t_T_max, 
    't_T_max_tol':t_T_max_tol}, 
    name=1)
pd.concat((pars, sr)).to_csv(out_dir + f'pars_{name}.tsv', sep='\t', header=None)
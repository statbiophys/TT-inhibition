import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import sys

import utils_string as uts
import utils as ut
import utils_phase_diagr as ut_pd



out_dir_str = 'out_data/string_model/'
str_name = 'gamma=2.0_L=40'
out_dir = 'out_data/string_model_Tmax/'
name = str_name + '_meanlt=-4_stdlt=2.5'

label = sys.argv[1]
chunk_start = int(sys.argv[2])
chunk_size = int(sys.argv[3])
mem_time = 30
dt = 0.005


with open(out_dir_str + f'string_ens_{str_name}.pkl', 'rb') as f:
    string_ens = pickle.load(f)

with open(out_dir + f'a0s_results_{name}.pkl', 'rb') as f:
    alpha0s, results = pickle.load(f)

pars = pd.read_csv(out_dir_str + f'pars_{name}.tsv', sep='\t', header=None, index_col=0)[1].to_dict()
string_model = uts.TT_string(N_tot=pars['N_tot'], L_string=int(pars['L_string']), N_recruit=pars['N_recruit'])
n_time_steps = int(mem_time / dt)

logtau_ens = []
for i, strings in enumerate(string_ens):
    h_dists = uts.comp_h_dist(strings, np.zeros(len(strings[0])))
    logtau_ens.append(string_model.get_logtau(h_dists, pars['mean_logtau'], pars['std_logtau']))


string_ids_ens, Ts_final_ens,  = [], []
x_maxs_ens, n_xmax_ens = [], []

progress = tqdm(total=len(alpha0s))
for i in range(len(alpha0s)):

    string_aux, Ts_aux, taus_mat = [], [], []
    for j in range(chunk_size):
        
        i_string_ens = j + chunk_start
        string_aux.append(i_string_ens)

        # Generating a trajectory until memory
        taus_mat.append(np.exp(logtau_ens[i_string_ens]))
        TT_pars = ut.TT_params(taus_mat[-1], alpha0=alpha0s[i], beta0=results[i][0], gamma=results[i][1])
        _, Ts_mat, _, _ = ut.run_setting(TT_pars, ut.tau_sampler_lognorm(), n_time_steps, dt, traj_steps=50)
        Ts_aux.append(Ts_mat[-1])

    # Computing some observables
    if alpha0s[i] > 5e-4: qt = 0.995 # Setting the sensitivty to outliers depending on alpha0)
    else: qt = 0.999
    x_max, y_max, n_x_max, outcome = ut_pd.find_max(taus_mat, Ts_aux, smooth_bins=50, quantile_taus=qt)
    x_maxs_ens.append(x_max)
    n_xmax_ens.append(n_x_max)

    string_ids_ens.append(string_aux)
    Ts_final_ens.append(Ts_aux)
    progress.update(1)

with open(out_dir + f'Ts_final_{name}_{label}.pkl', 'wb') as f:
    pickle.dump((string_ids_ens, Ts_final_ens, x_maxs_ens, n_xmax_ens), f)

# Export params as well
sr = pd.Series({'chunk_start':chunk_start, 'chunk_size':chunk_size, 'mem_time':mem_time, 'dt':dt}, name=1)
pars = pd.Series(pars, name=1)
pd.concat((pars, sr)).to_csv(out_dir + f'pars_{name}_{label}.tsv', sep='\t', header=None)
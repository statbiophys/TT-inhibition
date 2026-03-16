import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import utils_string as uts


out_dir = 'out_data/string_model/'

string_model = uts.TT_string(N_tot=10**7, L_string=40, N_recruit=100, gamma=1.0)
name = f'gamma={string_model.gamma_recr}_L={string_model.L}'

R_string = 10
string_ens = []
progress = tqdm(total=R_string)
for _ in range(R_string):
    recruit_strings, h_recruit = string_model.gen_and_recruit_string()
    string_ens.append(recruit_strings)
    progress.update(1)

with open(out_dir + f'string_ens_{name}.pkl', 'wb') as f:
    pickle.dump(string_ens, f)

sr = pd.Series(string_model.get_pars())
sr.to_csv(out_dir + f'string_pars_{name}.tsv', sep='\t', header=None)

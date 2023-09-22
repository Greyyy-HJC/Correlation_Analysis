# %%
'''
Standard error 
'''


import numpy as np
import gvar as gv
import liblattice.preprocess.resampling as resamp

N_conf = 96

# %%
#! raw data without resampling

corr_data = gv.load('samp_data.pkl')
meff = np.log( corr_data[:,:-1] / corr_data[:,1:] )
meff_avg = gv.dataset.avg_data(meff, bstrap=False)

print(gv.sdev(meff_avg))
print(np.std(meff, axis=0) / np.sqrt(N_conf-1))

# %%
#! Jackknife

corr_jk = resamp.jackknife(corr_data)
meff_jk = np.log( corr_jk[:,:-1] / corr_jk[:,1:] )
meff_jk_avg = resamp.jk_ls_avg(meff_jk)

print(gv.sdev(meff_jk_avg))
print(np.std(meff_jk, axis=0) * np.sqrt(N_conf-1))

# %%
#! Bootstrap

corr_bs = resamp.bootstrap(corr_data, samp_times=100)
meff_bs = np.log( corr_bs[:,:-1] / corr_bs[:,1:] )
meff_bs_avg = resamp.bs_ls_avg(meff_bs)
# meff_bs_avg = gv.dataset.avg_data(meff_bs, bstrap=True)

print(gv.sdev(meff_bs_avg))
print(np.std(meff_bs, axis=0))
# %%

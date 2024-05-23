# %%
"""

We have a bootstrapped sample data set with different momenta, each momentum case has 100 configurations and 11 time slices. 

Correlation is determined when averaging samples
------------------------------------------------

    1. If average each mom separately (average on tseq axis), different moms will be treated as independent variables without correlations.

    2. If average all mom and tseq together, every (mom, tseq) will be treated as correlated variables.

"""

import gvar as gv
import numpy as np

# ignore the warning of log(0)
np.seterr(invalid='ignore')

samp_data = gv.load("data/samp_data_mom_mix.pkl")

n_mom = 7
n_t = 11
n_samp = 100

# %%
#! correlation is determined when averaging samples

# * avg each mom separately
if True:
    data_avg_sep = []

    for key in samp_data.keys():
        data_avg_sep.append(gv.dataset.avg_data(samp_data[key], bstrap=True))

    data_avg_sep = np.array(data_avg_sep)

    corr_sep = gv.evalcorr(data_avg_sep)
    print("\n shape of corr_sep: ", np.shape(corr_sep))
    print("\n correlation matrix between mom 0 and mom 1 when averaging separately: ")
    print(corr_sep[0, :, 1, :]) #? there is no correlation between different momenta


# * avg all mom and time slices together
if True:
    data_col = np.array([samp_data[key] for key in samp_data.keys()])
    # print(np.shape(data_col))

    data_col = np.swapaxes(data_col, 0, 1)  # swap the sample axis to the 0-th axis
    # print(np.shape(data_col))

    data_col = list(data_col)
    data_avg_col = gv.dataset.avg_data(data_col, bstrap=True)

    corr_col = gv.evalcorr(data_avg_col)
    print("\n shape of corr_col: ", np.shape(corr_col))
    print("\n correlation matrix between mom 0 and mom 1 when averaging all together: ")
    print(corr_col[0, :, 1, :]) #? it can be seen that the correlation between different momenta is non-zero

# %%
#! show the correlation matrix of the average all together case

if True:
    import matplotlib.pyplot as plt

    data_avg_col_plot = data_avg_col.reshape(77)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.imshow(gv.evalcorr(data_avg_col_plot))

    # add the colorbar for both plots
    fig.colorbar(ax.imshow(gv.evalcorr(data_avg_col_plot)), ax=ax)

    # add the title for both plots
    ax.set_title("correlation matrix when averaging all together")

    plt.show() #? the correlation matrix shows a non-zero correlation between different momenta


# %%
"""

Gvar list can preserve all correlations as well as reconstruct distributions
----------------------------------------------------------------------------

    If we have distributions with N samples, after averaging it to gvar list and reconstructing distributions with mean and correlations, the reconstructed distributions will be almost same as the original distributions.

    In other words, gvar list with correlations approx distributions. [The difference is below 1% .]

"""

#! use gvar list to reconstruct distributions and compare with the original distributions

import module.resampling as resamp

if True:
    data_test = np.array(samp_data["mom_0"])
    print("\n shape of data_test: ", np.shape(data_test))
    data_test_avg = gv.dataset.avg_data(data_test, bstrap=True)

    print("\n data_test_avg: ")
    print(data_test_avg)

    # * reconstruct distributions
    data_test_recon = resamp.gv_ls_to_samples_corr(data_test_avg, N_samp=100)

    # * compare the reconstructed distributions with the original distributions
    diff = (data_test - data_test_recon) / data_test
    print("\n normalized difference of the reconstructed distributions: ")
    print(diff)
    print("\n max difference: ")
    print(np.max(np.abs(diff)))  #? we can see that the largest difference is below 1%

#! the difference is below 1%


# %%

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
""" 


If we construct gvar list with the completed correlation matrix, gvar list behaves exactly like the distribution of bs samples, as long as you did the bootstrap resampling, no matter in linear or non-linear (including lsqfit, check "4_lsq_fit.py") operations. Therefore, you can choose one of two methods according to your situation, bs samples take more time but less memory, while gvar list takes less time but you may need more memory to store the large correlation matrix.

Let's check the behavior in linear and non-linear operations using fake data, to see if it can reproduce the correlation between samples.
"""

#! generate two random data sets, one is sin with amplitude 1, the other is sin with amplitude 2, add small gaussian noise to them

if True:
    N_conf = 100
    N_t = 10
    noise_amp = 0.1

    #? When we do the 3pt / 2pt ratio, we expect the fluctuation will be partly canceled out on each config/sample via ratio, so we add a large fixed noise to imitate this situation.
    noise_fix = np.random.normal(0, 0.1 + noise_amp, (N_conf, N_t))
    # noise_fix = np.zeros_like(noise_fix) # test without noise_fix

    #? If we do not use bootstrap, the existance of noise_fix will make the error of bs sample method smaller than the gvar method because of the fluctuation cancellation on each config/sample.
    if_bstrap = True 


    #* generate two data sets
    t = np.arange(N_t)
    # sin with amplitude 1
    data_1 = np.sin(t) + noise_fix + np.random.normal(0, noise_amp, (N_conf, N_t))
    print("\n shape of data_1: ", np.shape(data_1))
    # sin with amplitude 2
    data_2 = 2 * np.sin(t) + noise_fix + np.random.normal(0, noise_amp, (N_conf, N_t))
    print("\n shape of data_2: ", np.shape(data_2))

    #* do the bootstrap resampling
    if if_bstrap:
        data_1 = resamp.bootstrap(data_1, samp_times=100)
        data_2 = resamp.bootstrap(data_2, samp_times=100)


    #* process operations with sample averaging by gvar
    print("\n >>> Linear and non-linear operations with gvar sample averaging:")
    data_1_gv = resamp.bs_ls_avg(data_1)
    # print("\n shape of data_1_gv: ", np.shape(data_1_gv))

    data_2_gv = resamp.bs_ls_avg(data_2)
    # print("\n shape of data_2_gv: ", np.shape(data_2_gv))

    # subtract data_1 from data_2
    subtraction_gv = data_2_gv - data_1_gv
    print("\n gvar subtraction: ")
    print(subtraction_gv) #* note here if no bstrap, the error needs to be divided by sqrt(N_conf)

    # divide data_2 by data_1
    division_gv = data_2_gv / data_1_gv
    print("\n gvar division: ")
    print(division_gv) #* note here if no bstrap, the error needs to be divided by sqrt(N_conf)


    #* process operations sample by sample
    print("\n >>> Linear and non-linear operations sample by sample:")

    # subtract data_1 from data_2
    subtraction_sample = data_2 - data_1
    print("\n bs sample subtraction: ")
    print(resamp.bs_ls_avg(subtraction_sample)) #* note here if no bstrap, the error needs to be divided by sqrt(N_conf)

    # divide data_2 by data_1
    division_sample = data_2 / data_1
    print("\n bs sample division: ")
    print(resamp.bs_ls_avg(division_sample)) #* note here if no bstrap, the error needs to be divided by sqrt(N_conf)


# %%
#! plot the subtraction results and division results of the two methods for comparison, use errorbar plot
if True:
    from module.plot_settings import *
    from module.general_plot_funcs import errorbar_ls_plot

    #* comparison plot of subtraction
    x_ls = [np.arange(N_t), np.arange(N_t)+0.2]
    y_ls = [gv.mean(subtraction_gv), gv.mean( resamp.bs_ls_avg(subtraction_sample) )]
    yerr_ls = [gv.sdev(subtraction_gv), gv.sdev( resamp.bs_ls_avg(subtraction_sample) )]

    if not if_bstrap:
        yerr_ls = [ err / np.sqrt(N_conf) for err in yerr_ls ]

    errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls=["gvar", "sample"], title="subtraction comparison", save=False)

    #* comparison plot of division
    x_ls = [np.arange(N_t), np.arange(N_t)+0.2]
    y_ls = [gv.mean(division_gv), gv.mean( resamp.bs_ls_avg(division_sample) )]
    yerr_ls = [gv.sdev(division_gv), gv.sdev( resamp.bs_ls_avg(division_sample) )]

    if not if_bstrap:
        yerr_ls = [ err / np.sqrt(N_conf) for err in yerr_ls ]

    errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls=["gvar", "sample"], title="division comparison", save=False, ylim=[1, 3]) #? the two methods are consistent, even for non-linear operations



# %%

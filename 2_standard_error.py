# %%
"""

NOTE: When I say configuration, I mean the raw data, when I say sample, I mean the resampled data.


Standard deviation v.s. standard error of the mean
--------------------------------------------------

    The standard error (SE) of a statistic (usually an estimate of a parameter) is the standard deviation of its sampling distribution or an estimate of that standard deviation. If the statistic is the sample mean, it is called the standard error of the mean (SEM).

    The standard deviation (SD) reflects variability within a sample, while the standard error (SE) estimates the variability across samples of a population.

    SE: The standard error measures the precision of an estimate, it tells you how much the sample mean is likely to vary from the population mean.

    SD: The standard deviation measures the average amount of variability or dispersion in a dataset, it tells you how spread out the values in a dataset are around the mean.

    
How to calculate the standard error of the mean (SEM)?
-----------------------------------------------

    Variance: Var(X) = Cov(X, X) = < (X - <X>)^2 > = <X^2> - <X>^2
    Var( aX + bY ) = a^2 Var(X) + b^2 Var(Y) + 2ab Cov(X, Y)

    1. For raw data without resampling 

        Var( <X> ) = Var( sum(X_i) / N ) = Var( sum(X_i) ) / N^2 = N * Var(X_i) / N^2 = Var(X_i) / N

        here Var(X_i) is the variance of each sample, which is approximated with SD of the whole set, Var(X_i) = SD^2

    SEM = sqrt( Var( <X> ) ) = SD / sqrt(N), where N is the number of configurations

    2. For jackknife

        New distribution of X_{-i} = sum(X_j) / (N-1), where j != i

        Var( X_{-i} ) = < ( X_{-i} - <X> )^2 >, 
            in which X_{-i} - <X> = [( N <X> - X_i ) / (N-1)] - <X> = ( <X> - X_i ) / (N-1),
            so Var( X_{-i} ) = Var( X_i ) / (N-1)^2

        Or you can think of Var( X_{-i} ) = Var( sum(X_i) - X_i ) / (N-1)^2, while sum(X_i) is a constant for each sample.

        Var( <X> ) = Var(X_i) / N = Var( X_{-i} ) * (N-1)^2 / N

    SEM = SD( X_{-i} ) * sqrt(N-1)

    3. For bootstrap

        New distribution is exactly the distribution of the mean, so the standard error is the standard deviation of the bootstrap distribution.

    SEM = SD, note here SD is the SD of the new distribution


"""


import numpy as np
import gvar as gv
import module.resampling as resamp

# ignore the warning of log(0)
np.seterr(invalid='ignore')

N_conf = 96

# %%
#! raw data without resampling

corr_data = gv.load("data/samp_data.pkl")
meff = np.log(corr_data[:, :-1] / corr_data[:, 1:])
# average over all configurations
meff_avg = gv.dataset.avg_data(meff, bstrap=False) 

print("\n Raw data SEM: ")
print(gv.sdev(meff_avg))
print("\n Raw data SD / N: ")
print(np.std(meff, axis=0) / np.sqrt(N_conf)) #? they are exactly consistent, verified our analysis for the raw data above

# %%
#! Jackknife

corr_jk = resamp.jackknife(corr_data)
meff_jk = np.log(corr_jk[:, :-1] / corr_jk[:, 1:])
meff_jk_avg = resamp.jk_ls_avg(meff_jk)

print("\n Jackknife SEM: ")
print(gv.sdev(meff_jk_avg))
print("\n Jackknife SD * sqrt(N-1): ")
print(np.std(meff_jk, axis=0) * np.sqrt(N_conf - 1)) #? they are approximately consistent, verified our analysis for the jackknife above

# %%
#! Bootstrap

corr_bs = resamp.bootstrap(corr_data, samp_times=100)
meff_bs = np.log(corr_bs[:, :-1] / corr_bs[:, 1:])
meff_bs_avg = resamp.bs_ls_avg(meff_bs)
# meff_bs_avg = gv.dataset.avg_data(meff_bs, bstrap=True)

print("\n Bootstrap SEM: ")
print(gv.sdev(meff_bs_avg))
print("\n Bootstrap SD: ")
print(np.std(meff_bs, axis=0)) #? they are approximately consistent, verified our analysis for the bootstrap above

# %%
#! The small difference above partly comes from correlation, "resamp.jk_ls_avg" and "resamp.bs_ls_avg" considered correlation while "np.std" does not.

# * generate a random data set
test = np.random.normal(size=(N_conf, 10))
test_jk = resamp.jackknife(test)
test_jk_avg = resamp.jk_ls_avg(test_jk)

print("\n shape of data jk avg: ")
print(np.shape(test_jk_avg))

# * show the comparison between the two correlation matrices
if True:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(gv.evalcorr(test_jk_avg))
    ax[1].imshow(gv.evalcorr(meff_jk_avg))

    # add the colorbar for both plots
    fig.colorbar(ax[0].imshow(gv.evalcorr(test_jk_avg)), ax=ax[0])
    fig.colorbar(ax[1].imshow(gv.evalcorr(meff_jk_avg)), ax=ax[1])

    # add the title for both plots
    ax[0].set_title("Random data")
    ax[1].set_title("Correlation matrix of meff")

    plt.show() #? compared with the random data, the correlation matrix of meff shows a non-zero correlation between different time slices

# %%

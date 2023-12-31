# %%
"""
What is resampling?
    Resampling is like repeatedly taking mini-versions of our data to see how much a specific number changes each time we do it.

    In general, we use resampling to give a distribution of mean values.

Why do we need resampling?
    1. Improves the overall accuracy and estimates any uncertainty within a population.

    2. Avoids overfitting: how our models perform across various subsets of the data.

    3. Physical meaning: our correlation functions are defined in the sense of average over all possible paths.

Two common resampling methods:
    1. Jackknife

    2. Bootstrap

"""

import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import liblattice.preprocess.resampling as resamp

# %%
#! check the sample data ##########################################

if True:
    corr_data = gv.load("samp_data.pkl")
    print(np.shape(corr_data))

    # calculate the effective mass
    # meff(t) = np.log( corr_data(t) / corr_data(t+1) )
    meff = np.log(corr_data[:, :-1] / corr_data[:, 1:])

    # average over configs
    meff_avg = gv.dataset.avg_data(meff, bstrap=False)

    # check the plot
    plt.errorbar(
        np.arange(len(meff_avg)),
        gv.mean(meff_avg),
        yerr=gv.sdev(meff_avg),
        fmt="o",
        capsize=3,
        label="sample data",
    )
    plt.ylim([0.6, 1])
    plt.legend()
    plt.show()

    # check the data of correlation function
    print(corr_data[:, 8])


# %%
#! Jackknife #####################################################

# * An example:

if True:
    # Suppose I have 5 integer measurements of a quantity
    data = np.random.randint(0, 10, 5)
    print(data)

    # Jackknife will give me 5 new data sets, each of which has 49 measurements
    data_jk = [np.delete(data, i) for i in range(5)]
    print(data_jk)

    # Then I can compute the mean of each data set
    data_jk_mean = np.mean(data_jk, axis=1)
    print(data_jk_mean)


def jackknife(data):
    """Do jackknife resampling on the data, drop one data each time and average the rest.

    Args:
        data (list): data to be resampled, resample on the axis 0.

    Returns:
        array: jackknife samples
    """
    data = np.array(data)  # Convert the input data into a numpy array

    N_conf = data.shape[
        0
    ]  # Get the number of elements in the first dimension of the array

    conf_jk = [
        np.delete(data, i, axis=0) for i in range(N_conf)
    ]  # Create a list of arrays where each array is obtained by deleting one element along the first dimension

    jk_ls = np.mean(conf_jk, axis=1)  # Compute the mean along the second dimension

    return jk_ls  # Return the jackknife samples


# %%
#! Bootstrap #####################################################

# * An example:

if True:
    # Suppose I have 5 integer measurements of a quantity
    data = np.random.randint(0, 10, 5)
    print(data)

    # Bootstrap will give me n_times new data sets, each of which has n_samp measurements
    n_times = 3
    n_samp = 4

    data_bs = [
        np.random.choice(data, n_samp, replace=True) for i in range(n_times)
    ]  # 有放回的抽样
    print(data_bs)

    # Then I can compute the mean of each data set
    data_bs_mean = np.mean(data_bs, axis=1)
    print(data_bs_mean)


def bootstrap(data, samp_times):
    """Do bootstrap resampling on the data, take random samples from the data and average them.

    Args:
        data (list): data to be resampled, resample on the axis 0.
        samp_times (int): how many times to sample, i.e. how many bootstrap samples to generate

    Returns:
        array: bootstrap samples
    """
    data = np.array(data)  # Convert the input data into a numpy array

    N_conf = data.shape[
        0
    ]  # Get the number of elements in the first dimension of the array

    conf_bs = np.random.choice(
        N_conf, (samp_times, N_conf), replace=True
    )  # Generate random indices to select samples from the data with replacement

    bs_ls = np.take(
        data, conf_bs, axis=0
    )  # Select the samples using the generated indices

    bs_ls = np.mean(
        bs_ls, axis=0 + 1
    )  # Compute the mean along both the first and second dimensions

    return bs_ls  # Return the bootstrap samples


# %%
#! Jackknife and bootstrap on sample data ########################

# * Jackknife

if True:
    corr_jk = jackknife(corr_data)

    # calculate the effective mass
    # meff(t) = np.log( corr_jk(t) / corr_jk(t+1) )
    meff = np.log(corr_jk[:, :-1] / corr_jk[:, 1:])

    # average over configs
    meff_avg = resamp.jk_ls_avg(meff)

    # check the plot
    plt.errorbar(
        np.arange(len(meff_avg)),
        gv.mean(meff_avg),
        yerr=gv.sdev(meff_avg),
        fmt="o",
        capsize=3,
        label="jackknife",
    )
    plt.ylim([0.6, 1])
    plt.legend()
    plt.show()


# * Bootstrap

if True:
    corr_bs = bootstrap(corr_data, 100)

    # calculate the effective mass
    # meff(t) = np.log( corr_bs(t) / corr_bs(t+1) )
    meff = np.log(corr_bs[:, :-1] / corr_bs[:, 1:])

    # average over configs
    meff_avg = resamp.bs_ls_avg(meff)

    # check the plot
    plt.errorbar(
        np.arange(len(meff_avg)),
        gv.mean(meff_avg),
        yerr=gv.sdev(meff_avg),
        fmt="o",
        capsize=3,
        label="bootstrap",
    )
    plt.ylim([0.6, 1])
    plt.legend()
    plt.show()


# %%

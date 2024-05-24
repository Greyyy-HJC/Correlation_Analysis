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
        Each calculation of the statistic is called a "leave-one-out" estimate.

    2. Bootstrap
        Bootstrapping is a topic that has been studied extensively for many different population parameters and many different situations. There are parametric bootstrap, nonparametric bootstraps, weighted bootstraps, etc. We merely introduce the very basics of the bootstrap method.

        The most common implementation of bootstrap involves generating resampled datasets that have the same sample size as the original dataset.

Comparison between jackknife and bootstrap:
    - Jackknife:
        Pros: 
        1. Simple and easy to implement.
        2. Particularly useful for estimating bias and linear statistics (e.g., mean, linear regression coefficients).
        

        Cons:
        1. May not be accurate for non-linear statistics (e.g., median or percentiles), better for smooth and differentiable statistics.
        2. Only two points are different in each two jackknife samples.
        3. The estimates may be unstable for small sample sizes.

    - Bootstrap:
        Pros:
        1. Extremely flexible, suitable for almost any type of statistic, whether linear or non-linear.
        2. Makes fewer assumptions about the sample distribution and can be used for complex estimation problems.
        3. Generally more stable than the jackknife method for small sample sizes.

        Cons:
        1. It needs more samples and is computationally expensive.

    In summary, Jackknife is a special case of the bootstrap method, and the bootstrap method is more general and flexible.

"""

import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import module.resampling as resamp

# ignore the warning of log(0)
np.seterr(invalid='ignore')

# %%
#! check the sample data ##########################################

if True:
    corr_data = gv.load("data/samp_data.pkl")
    print("\n shape of corr_data: ", np.shape(corr_data)) #? (96, 11) means 96 configurations and 11 time slices

    # calculate the effective mass
    # meff(t) = np.log( corr_data(t) / corr_data(t+1) )
    meff = np.log(corr_data[:, :-1] / corr_data[:, 1:]) # make this no warning


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
    plt.show() #? but only 7 time slices are shown, why?

    # check the data of correlation function
    print("\n the t=8 time slice of the 2pt: ")
    print(corr_data[20:40, 8]) #? the 9th time slice of the 2pt looks good, but some of configs are negative, so we need resampling

# %%
#! Jackknife #####################################################

# * An example:

if True:
    # Suppose I have 5 integer measurements of a quantity
    data = np.random.randint(0, 10, 5)
    print("\n original data: ")
    print(data)

    # Jackknife will give me 5 new data sets, each of which has 4 measurements
    data_jk = [np.delete(data, i) for i in range(5)]
    print("\n jackknife data: ")
    print(data_jk)

    # Then I can compute the mean of each data set
    data_jk_mean = np.mean(data_jk, axis=1)
    print("\n jackknife mean: ")
    print(data_jk_mean)


# package the jackknife function
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
    print("\n original data: ")
    print(data)

    # Bootstrap will give me n_times new data sets, each of which has n_samp measurements
    n_times = 5
    n_samp = 4

    data_bs = [
        np.random.choice(data, n_samp, replace=True) for i in range(n_times)
    ]  # Sampling with replacement
    print("\n bootstrap data: ")
    print(data_bs)

    # Then I can compute the mean of each data set
    data_bs_mean = np.mean(data_bs, axis=1)
    print("\n bootstrap mean: ")
    print(data_bs_mean)

# package the bootstrap function
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
    plt.show() #? now we can see all time slices, note here meff only has 10 time slices because of the definition


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
    plt.show() #? bootstrap gives a similar plot to jackknife, as we expected


# %%
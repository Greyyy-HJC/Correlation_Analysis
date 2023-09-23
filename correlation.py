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

samp_data = gv.load("samp_data_mom_mix.pkl")

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
    print("shape of corr_sep: ", np.shape(corr_sep))
    print(corr_sep[0, :, 1, :])


# * avg all mom and time slices together
if True:
    data_col = np.array([samp_data[key] for key in samp_data.keys()])
    print(np.shape(data_col))

    data_col = np.swapaxes(data_col, 0, 1)  # swap the sample axis to the 0-th axis
    print(np.shape(data_col))

    data_col = list(data_col)
    data_avg_col = gv.dataset.avg_data(data_col, bstrap=True)

    corr_col = gv.evalcorr(data_avg_col)
    print("shape of corr_col: ", np.shape(corr_col))
    print(corr_col[0, :, 1, :])

#! it can be seen that the correlation between different momenta is different in these two cases


# %%
"""

Gvar list can preserve all correlations as well as reconstruct distributions
----------------------------------------------------------------------------

    If we have distributions with N samples, after averaging it to gvar list and reconstructing distributions with mean and correlations, the reconstructed distributions will be almost same as the original distributions.

    In other words, gvar list with correlations approx distributions. [The difference is below 1% .]

"""

#! use gvar list to reconstruct distributions and compare with the original distributions

import liblattice.preprocess.resampling as resamp

if True:
    data_test = np.array(samp_data["mom_0"])
    print(np.shape(data_test))
    data_test_avg = gv.dataset.avg_data(data_test, bstrap=True)

    # * reconstruct distributions
    data_test_recon = resamp.gv_ls_to_samples_corr(data_test_avg, N_samp=100)

    # * compare the reconstructed distributions with the original distributions
    diff = (data_test - data_test_recon) / data_test
    print(diff)
    print(np.max(np.abs(diff)))  # find the larest difference

#! the difference is below 1%

# %%
"""

Fit once with gvar list is better than fit N times on each sample
-----------------------------------------------------------------

We can fit the gvar list once, instead of fit each sample for N_samp times, but we may have a large correlation matrix lies in the gvar list.

    1. Fit once can give almost same distributions as fit N times as long as we reconstruct with correlations.

    2. Fit once means less parameter tuning and better fit quality.

"""

import lsqfit as lsf


#! do once fit with gvar list and compare with the N times fit on each sample


# * lsqfit setup
if True:

    def fcn(x, p):
        return p["m"] + p["c1"] * x + p["c2"] * x**2

    priors = gv.BufferDict()
    priors["m"] = gv.gvar(1, 10)
    priors["c1"] = gv.gvar(1, 10)
    priors["c2"] = gv.gvar(1, 10)

    # * data with full correlations
    data_all = np.array([samp_data[key] for key in samp_data.keys()])
    data_all = np.swapaxes(data_all, 0, 1)  # swap the sample axis to the 0-th axis
    data_all = list(data_all)
    data_all_avg = gv.dataset.avg_data(data_all, bstrap=True)


# * fit once
if True:
    res_fit_once = []

    for mom in range(n_mom):
        t_ls = np.arange(n_t)
        gv_ls = data_all_avg[mom]

        res = lsf.nonlinear_fit(
            data=(t_ls, gv_ls),
            prior=priors,
            fcn=fcn,
            maxit=10000,
            svdcut=1e-100,
            fitter="scipy_least_squares",
        )

        if res.Q < 0.1:  # * bad fit
            print(">>> Bad gvar fit for mom = {} with p value {}".format(mom, res.Q))

        res_fit_once.append(res.p["m"])


# * fit N times
if True:
    res_fit_n_times = []

    for mom in range(n_mom):
        res_fit_n_times.append([])
        gv_ls = data_all_avg[mom]

        for n in range(n_samp):
            samp_ls = data_all[n][mom]
            samp_gv = gv.gvar(
                samp_ls, gv.evalcov(gv_ls)
            )  # add correlation to each sample for fit

            t_ls = np.arange(n_t)

            res = lsf.nonlinear_fit(
                data=(t_ls, samp_gv),
                prior=priors,
                fcn=fcn,
                maxit=10000,
                svdcut=1e-100,
                fitter="scipy_least_squares",
            )

            if res.Q < 0.1:  # * bad fit
                print(
                    ">>> Bad sample fit for mom = {} with p value {}".format(mom, res.Q)
                )

            res_fit_n_times[mom].append(res.p["m"].mean)

    res_fit_n_times = np.swapaxes(
        res_fit_n_times, 0, 1
    )  # swap the sample axis to the 0-th axis


# * compare the fit results
if True:
    res_fit_once_recon = resamp.gv_ls_to_samples_corr(
        res_fit_once, N_samp=n_samp
    )  # reconstruct distributions

    diff = (res_fit_n_times - res_fit_once_recon) / res_fit_n_times
    print(diff)
    print(np.max(np.abs(diff)))  # find the larest difference

#! fit N times means worse fit, and the difference is about 1%

# %%

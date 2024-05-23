# %%
"""

Fit once with gvar list is better than fit N times on each sample
-----------------------------------------------------------------

We can fit the gvar list once, instead of fit each sample for N_samp times, but we may have a large correlation matrix lies in the gvar list.

    1. Fit once can give almost same distributions as fit N times as long as we reconstruct with correlations.

    2. Fit once means less parameter tuning and better fit quality.

"""

import lsqfit as lsf
import numpy as np
import gvar as gv
import module.resampling as resamp

# ignore the warning of log(0)
np.seterr(invalid='ignore')

samp_data = gv.load("data/samp_data_mom_mix.pkl")

n_mom = 7
n_t = 11
n_samp = 100


# %%
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
    bad_fit_count_once = 0

    for mom in range(n_mom):
        t_ls = np.arange(n_t)
        gv_ls = data_all_avg[mom]

        # gv_ls = gv.gvar(gv.mean(gv_ls), gv.sdev(gv_ls)) #! remove correlations

        res = lsf.nonlinear_fit(
            data=(t_ls, gv_ls),
            prior=priors,
            fcn=fcn,
            maxit=10000,
            svdcut=1e-100,
            fitter="scipy_least_squares",
        )

        if res.Q < 0.05:  # * bad fit
            bad_fit_count_once += 1

        res_fit_once.append(res.p["m"])


# * fit N times
if True:
    res_fit_n_times = []
    bad_fit_count_n_times = 0

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

            if res.Q < 0.05:  # * bad fit
                bad_fit_count_n_times += 1

            res_fit_n_times[mom].append(res.p["m"].mean)

    res_fit_n_times = np.swapaxes(
        res_fit_n_times, 0, 1
    )  # swap the sample axis to the 0-th axis


# * compare the fit results
if True:
    print('\n>>> Bad fit count: ')
    print('Fit once: ' + str(bad_fit_count_once))
    print('Fit N times: ' + str(bad_fit_count_n_times))

    res_fit_once_recon = resamp.gv_ls_to_samples_corr(
        res_fit_once, N_samp=n_samp
    )  # reconstruct distributions

    # * check the difference of the fit results distribution
    diff = (res_fit_n_times - res_fit_once_recon) / res_fit_n_times
    print('\n>>> Difference of the fit results distribution: ')
    print(np.max(np.abs(diff)))  # find the larest difference

    # * check the difference of the fit results
    fit_1 = resamp.bs_ls_avg(res_fit_once_recon)
    fit_2 = resamp.bs_ls_avg(res_fit_n_times)

    print('\n>>> Difference of the fit results: ')
    print('Fit once: ' + str(fit_1))
    print('Fit N times: ' + str(fit_2))

#! fit N times means worse fit, and the difference is within the error.
#! if we do the gvar fit without correlations, the difference will be larger.

# %%

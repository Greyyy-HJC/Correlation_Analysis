"""
Here are general plot functions for liblattice.
"""
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from .plot_settings import *


# * default plot axes for general plots
plt_axes = [0.12, 0.12, 0.8, 0.8]  # left, bottom, width, height
fs_p = {"fontsize": 13}  # font size of text, label, ticks
ls_p = {"labelsize": 13}


def errorbar_plot(x, y, yerr, title, ylim=None, save=True, head=None):
    """Make a general errorbar plot, default save to `output/plots/`.

    Args:
        x (list): list of float x values
        y (list): list of float y values
        yerr (list): list of float yerr values
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    ax.errorbar(x, y, yerr, marker='x', **errorb)
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)

    if save == True:
        plt.savefig("../output/plots/" + title + "_err.pdf", transparent=True)


def fill_between_plot(x, y, yerr, title, ylim=None, save=True, head=None):
    """Make a general fill_between plot, default save to `output/plots/`.

    Args:
        x (list): list of float x values
        y (list): list of float y values
        yerr (list): list of float yerr values
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    ax.fill_between(
        x,
        [y[i] + yerr[i] for i in range(len(y))],
        [y[i] - yerr[i] for i in range(len(y))],
        alpha=0.4,
    )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)

    if save == True:
        plt.savefig("../output/plots/" + title + "_fill.pdf", transparent=True)


def errorbar_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title, ylim=None, save=True, head=None):
    """Make a general errorbar plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    for x_ls, y_ls, yerr_ls, label_ls in zip(x_ls, y_ls, yerr_ls, label_ls):
        ax.errorbar(x_ls, y_ls, yerr_ls, marker='x', label=label_ls, **errorb)
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("../output/plots/" + title + "_err_ls.pdf", transparent=True)


def fill_between_ls_plot(x_ls, y_ls, yerr_ls, label_ls, title, ylim=None, save=True, head=None):
    """Make a general fill_between plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    for x_ls, y_ls, yerr_ls, label_ls in zip(x_ls, y_ls, yerr_ls, label_ls):
        ax.fill_between(
            x_ls,
            [y_ls[i] + yerr_ls[i] for i in range(len(y_ls))],
            [y_ls[i] - yerr_ls[i] for i in range(len(y_ls))],
            alpha=0.4,
            label=label_ls,
        )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("../output/plots/" + title + "_fill_ls.pdf", transparent=True)


def errorbar_fill_between_ls_plot(
    x_ls, y_ls, yerr_ls, label_ls, plot_style_ls, title, ylim=None, save=True, head=None
):
    """Make a general errorbar & fill_between plot with multiple lines, default save to `output/plots/`.

    Args:
        x_ls (list): list of list of float x values
        y_ls (list): list of list of float y values
        yerr_ls (list): list of list of float yerr values
        label_ls (list): list of str labels
        plot_style_ls (list): list of str plot styles, 'errorbar' or 'fill_between'
        title (str): title of the plot, and also the name of the plot file
        ylim (tuple, optional): set the ylim of the plot. Defaults to None.
        save (bool, optional): whether save it. Defaults to True.
        head (ax, optional): whether make a new figure. Defaults to None, which means make a new figure. If not None, then set head to be the ax of the figure.
    """

    if head == None:
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
    else:
        ax = head
    for x_ls, y_ls, yerr_ls, label_ls, plot_style in zip(
        x_ls, y_ls, yerr_ls, label_ls, plot_style_ls
    ):
        if plot_style == "errorbar":
            ax.errorbar(x_ls, y_ls, yerr_ls, label=label_ls, **errorb)
        elif plot_style == "fill_between":
            ax.fill_between(
                x_ls,
                [y_ls[i] + yerr_ls[i] for i in range(len(y_ls))],
                [y_ls[i] - yerr_ls[i] for i in range(len(y_ls))],
                alpha=0.4,
                label=label_ls,
            )
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_ylim(ylim)
    plt.title(title, **fs_p)
    plt.legend()

    if save == True:
        plt.savefig("../output/plots/" + title + "_err_fill_ls.pdf", transparent=True)


def plot_fit_on_data_log(get_ratio_data, ra_fit_res, ra_re_fcn, ra_im_fcn, px, py, pz, b, z, err_tsep_ls, fill_tsep_ls, Ls, err_tau_cut=1, fill_tau_cut=1):
    """
    Plot the ratio fit on data.

    Args:
        px (float): Momentum in the x-direction.
        py (float): Momentum in the y-direction.
        pz (float): Momentum in the z-direction.
        b (float): Impact parameter.
        z (float): Light-cone momentum fraction.
        ss_fit_res (FitResult): Fit result for the 2pt SS fit.
        err_tsep_ls (list): List of time separations for error bars.
        fill_tsep_ls (list): List of time separations for filled regions.
        Ls (list): List of lattice sizes.
        err_tau_cut (int, optional): Cut for the range of tau values used for error bars. Defaults to 1.
        fill_tau_cut (int, optional): Cut for the range of tau values used for filled regions. Defaults to 1.

    Returns:
        None
    """
    from liblattice.preprocess.resampling import bs_ls_avg

    tsep_ls = [6, 8, 10, 12]
    ra_re, ra_im = get_ratio_data(px, py, pz, b, z, tsep_ls, jk_bs="bs")

    # Reshape and average the data only once.
    ra_re_avg = bs_ls_avg(ra_re.reshape(len(ra_re), -1)).reshape(len(tsep_ls), -1)  # (tsep, tau)
    ra_im_avg = bs_ls_avg(ra_im.reshape(len(ra_im), -1)).reshape(len(tsep_ls), -1)  # (tsep, tau)

    ra_re_avg_dic = {}
    ra_im_avg_dic = {}
    for id, tsep in enumerate(tsep_ls):
        ra_re_avg_dic[f'tsep_{tsep}'] = ra_re_avg[id]
        ra_im_avg_dic[f'tsep_{tsep}'] = ra_im_avg[id]

    def plot_part(part, ra_avg, ra_fcn, pdf_key):
        x_ls = []
        y_ls = []
        yerr_ls = []
        label_ls = []
        plot_style_ls = []

        for id, tsep in enumerate(err_tsep_ls):
            tau_range = np.arange(err_tau_cut, tsep + 1 - err_tau_cut)
            x_ls.append(tau_range - tsep / 2)
            y_ls.append(gv.mean(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut]))
            yerr_ls.append(gv.sdev(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut]))
            label_ls.append(f'tsep = {tsep}')
            plot_style_ls.append('errorbar')

        for id, tsep in enumerate(fill_tsep_ls):
            fit_tau = np.linspace(fill_tau_cut - 0.5, tsep - fill_tau_cut + 0.5, 100)
            fit_t = np.ones_like(fit_tau) * tsep
            fit_ratio = ra_fcn(fit_t, fit_tau, ra_fit_res.p, Ls)

            x_ls.append(fit_tau - tsep / 2)
            y_ls.append(gv.mean(fit_ratio))
            yerr_ls.append(gv.sdev(fit_ratio))
            label_ls.append(None)
            plot_style_ls.append('fill_between')

        band_x = np.arange(-6, 7)
        x_ls.append(band_x)
        y_ls.append(np.ones_like(band_x) * gv.mean(ra_fit_res.p[pdf_key]))
        yerr_ls.append(np.ones_like(band_x) * gv.sdev(ra_fit_res.p[pdf_key]))
        label_ls.append('fit')
        plot_style_ls.append('fill_between')

        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(plt_axes)
        errorbar_fill_between_ls_plot( 
            x_ls, y_ls, yerr_ls, label_ls, plot_style_ls,
            title=f'Ratio_fit_on_data_P{px}_{part}_b{b}_z{z}', save=False, head=ax
        )
        plt.savefig(f"../log/gsfit/Ratio_fit_on_data_P{px}_{part}_b{b}_z{z}.pdf", transparent=True)

    # Plot real part
    plot_part('real', ra_re_avg, ra_re_fcn, 'pdf_re')

    # Plot imaginary part
    plot_part('imag', ra_im_avg, ra_im_fcn, 'pdf_im')
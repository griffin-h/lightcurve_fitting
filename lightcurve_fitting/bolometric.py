from .filters import all_filters, filtdict, extinction_law
from .models import planck_fast, UniformPrior, LogUniformPrior, GaussianPrior
from .lightcurve import LC

from astropy import constants as const
from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import gaussian_kde
import emcee
import corner
import os
from pkg_resources import resource_filename
import warnings

for filt in all_filters:
    filt.read_curve()

plt.style.use(resource_filename('lightcurve_fitting', 'serif.mplstyle'))


def pseudo(temp, radius, z, filter0=filtdict['I'], filter1=filtdict['U'], cutoff_freq=np.inf):
    """
    Integrate a blackbody spectrum between two broadband filters to produce a pseudobolometric luminosity

    Parameters
    ----------
    temp : float, array-like
        Blackbody temperatures in kilokelvins
    radius : float, array-like
        Blackbody radii in units of 1000 solar radii
    z : float
        Redshift between the blackbody (rest frame) and the filters (observed frame)
    filter0, filter1 : lightcurve_fitting.filters.Filter, optional
        Filters to integrate between, where ``filter1`` is bluer than ``filter0``. Default: U to I.
    cutoff_freq : float, optional
        Cutoff frequency (in terahertz) for a modified blackbody spectrum (see https://doi.org/10.3847/1538-4357/aa9334)

    Returns
    -------
    L_opt : float, array-like
        Pseudobolometric luminosity in watts
    """
    freq0 = (filter0.freq_eff - filter0.dfreq / 2.).value
    freq1 = (filter1.freq_eff + filter1.dfreq / 2.).value
    x_optical = np.arange(freq0, freq1)
    y_optical = planck_fast(x_optical * (1. + z), temp, radius, cutoff_freq)
    L_opt = np.trapz(y_optical) * 1e12  # dx = 1 THz
    return L_opt


def plot_chain(chain, labels=None):
    """
    Plot the chain histories for an MCMC run

    Parameters
    ----------
    chain : array-like
        Array containing the MCMC chain history, e.g., from :meth:`emcee.EnsembleSampler.chain`
    labels : iterable, optional
        A list of axis labels for each parameter in the chain

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure object containing the chain history plots
    """
    ndim = chain.shape[-1]
    fig, ax = plt.subplots(ndim, figsize=(6., 2. * ndim))
    for i in range(ndim):
        ax[i].plot(chain[:, :, i].T, 'k', alpha=0.2)
        if labels:
            ax[i].set_ylabel(labels[i])
    return fig


def spectrum_mcmc(spectrum, epoch1, priors, starting_guesses, z=0., ebv=0., spectrum_kwargs=None, show=False, outpath='.',
                  nwalkers=10, burnin_steps=200, steps=100, save_chains=False, use_sigma=False, sigma_type='relative',
                  labels=None, freq_min=100., freq_max=1000.):
    """
    Fit the given spectral energy distribution to an epoch of photometry using a Markov-chain Monte Carlo routine

    Parameters
    ----------
    spectrum : function
        Function describing the spectrum. The first argument must be frequency in THz, and it must return spectral
        luminosity in watts per hertz. For a blackbody, you can use :func:`.models.planck_fast`.
    epoch1 : lightcurve_fitting.lightcurve.LC
        A single "epoch" of photometry that defines the observed spectral energy distribution
    priors : list
        Prior probability distributions for each model parameter (and sigma, if used). Available priors:
        :class:`.models.UniformPrior` (default), :class:`.models.LogUniformPrior`, :class:`.models.GaussianPrior`
    starting_guesses : array-like
        Initial guesses for each input parameter to ``spectrum``. Must have shape (nwalkers, nparameters).
    z : float, optional
        Redshift between the emission source and the observed filter. Default: 0.
    ebv : float, array-like, optional
        Selective extinction :math:`E(B-V)` in magnitudes, evaluated using a Fitzpatrick (1999) extinction law with
        :math:`R_V=3.1`. Its shape must be broadcastable to any array-like arguments. Default: 0.
    spectrum_kwargs : dict, optional
        Keyword arguments to be passed to the ``spectrum`` function.
    show : bool, optional
        Plot the chain histories, and display all plots at the end of the script. Default: only save the corner plot
    outpath : str, optional
        Directory to which to save the corner plots. Default: current directory
    nwalkers : int, optional
        Number of walkers (chains) to use for fitting. Default: 10
    burnin_steps : int, optional
        Number of MCMC steps before convergence. This part of the history is discarded. Default: 200
    steps : int, optional
        Number of MCMC steps after convergence. This part of the history is used to calculate parameters. Default: 100.
    save_chains : bool, optional
        If True, save the MCMC chain histories to the directory ``outpath``. Default: only save the corner plot
    use_sigma : bool, optional
        Include an intrinsic scatter parameter. Default: False.
    sigma_type : str, optional
        If 'relative' (default), sigma will be in units of the individual photometric uncertainties.
        If 'absolute', sigma will be in units of the median photometric uncertainty.
    labels : list, optional
        Axis labels for the chain histories and corner plot.
    freq_min, freq_max : float, optional
        Minimum and maximum frequencies for the SED panel in the corner plot. Default: (100., 1000.) or observed range.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        Sampler object containing the results of the MCMC fit
    """
    y = epoch1['lum'].data
    dy = epoch1['dlum'].data
    filtobj = epoch1['filter'].data
    mjdavg = np.mean(epoch1['MJD'].data)

    if spectrum_kwargs is None:
        spectrum_kwargs = {}

    if sigma_type == 'relative':
        sigma_units = dy
    elif sigma_type == 'absolute':
        sigma_units = np.median(dy)
    else:
        raise Exception('sigma_type must either be "relative" or "absolute"')

    def log_posterior(p):
        log_prior = 0.
        for prior, p_i in zip(priors, p):
            log_prior += prior(p_i)
        if np.isinf(log_prior):
            return log_prior
        y_fit = np.array([f.synthesize(spectrum, *p[:-1 if use_sigma else None], z=z, ebv=ebv, **spectrum_kwargs)
                          for f in filtobj])
        sigma = np.sqrt(dy ** 2. + (p[-1] * sigma_units) ** 2.) if use_sigma else dy
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2.) + ((y - y_fit) / sigma) ** 2.)
        return log_prior + log_likelihood

    ndim = len(priors)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    pos, _, _ = sampler.run_mcmc(starting_guesses, burnin_steps)

    # Plotting
    if show:
        plot_chain(sampler.chain, labels)
    sampler.reset()
    sampler.run_mcmc(pos, steps)
    if show:
        plot_chain(sampler.chain, labels)

    os.makedirs(outpath, exist_ok=True)
    if save_chains:
        chain_filename = os.path.join(outpath, f'{mjdavg:.3f}.npy')
        np.save(chain_filename, sampler.flatchain)

    f4 = spectrum_corner(spectrum, epoch1, sampler.flatchain, z, ebv, spectrum_kwargs, use_sigma, labels,
                         freq_min=freq_min, freq_max=freq_max, save_plot_as=os.path.join(outpath, f'{mjdavg:.3f}.pdf'))
    if show:
        plt.show()
    else:
        plt.close(f4)

    return sampler


def spectrum_corner(spectrum, epoch1, sampler_flatchain, z=0., ebv=0., spectrum_kwargs=None, use_sigma=False,
                    labels=None, freq_min=100., freq_max=1000., save_plot_as=''):
    """
    Plot the posterior distributions in a corner (pair) plot, with an inset showing the observed and model SEDs.

    Parameters
    ----------
    spectrum : function
        Function describing the spectrum. The first argument must be frequency in THz, and it must return spectral
        luminosity in watts per hertz. For a blackbody, you can use :func:`.models.planck_fast`.
    epoch1 : lightcurve_fitting.lightcurve.LC
        A single "epoch" of photometry that defines the observed spectral energy distribution
    sampler_flatchain : array-like
        2D array containing the aggregated MCMC chain histories
    z : float, optional
        Redshift between the emission source and the observed filter. Default: 0.
    ebv : float, array-like, optional
        Selective extinction :math:`E(B-V)` in magnitudes, evaluated using a Fitzpatrick (1999) extinction law with
        :math:`R_V=3.1`. Its shape must be broadcastable to any array-like arguments. Default: 0.
    spectrum_kwargs : dict, optional
        Keyword arguments to be passed to the ``spectrum`` function.
    use_sigma : bool, optional
        Include an intrinsic scatter parameter. Default: False.
    labels : list, optional
        Axis labels for the chain histories and corner plot.
    freq_min, freq_max : float, optional
        Minimum and maximum frequencies for the SED panel in the corner plot. Default: (100., 1000.) or observed range.
    save_plot_as : str, optional
        Filename to which to save the resulting plot

    Returns
    -------
    f4 : matplotlib.pyplot.Figure
        Figure containing the corner and SED plots.
    """

    ndim = sampler_flatchain.shape[-1]
    if spectrum_kwargs is None:
        spectrum_kwargs = {}

    f4 = corner.corner(sampler_flatchain, labels=labels)

    ps = sampler_flatchain[np.random.choice(sampler_flatchain.shape[0], 100)].T
    xmin = min(freq_min, epoch1['filter'].max().freq_eff.value)
    xmax = max(freq_max, epoch1['filter'].min().freq_eff.value)
    xfit = np.arange(xmin, xmax)
    freq = xfit * (1. + z)
    yfit = spectrum(freq, *ps[:-1 if use_sigma else None], **spectrum_kwargs) * extinction_law(freq, ebv)
    yscale = 10. ** np.floor(np.log10(yfit.max()))

    # add tick labels and axis labels to the top right figure so tight_layout works
    all_axes = np.reshape(f4.get_axes(), (ndim, ndim))
    top_right = all_axes[0, -1]
    top_right.set_frame_on(True)
    top_right.xaxis.set_major_locator(plt.AutoLocator())
    top_right.xaxis.tick_top()
    top_right.set_xlabel('Frequency (THz)')
    top_right.xaxis.set_label_position('top')
    top_right.yaxis.set_major_locator(plt.AutoLocator())
    top_right.yaxis.tick_right()
    top_right.set_ylabel(f'Luminosity $L_\\nu$ (10$^{{{np.log10(yscale):.0f}}}$ W Hz$^{{-1}}$)')
    top_right.yaxis.set_label_position('right')
    f4.tight_layout(h_pad=0.05, w_pad=0.05)

    bottom_left = all_axes[ndim // 2 - 1, (ndim + 1) // 2]
    if bottom_left is top_right:
        ax = top_right
    else:
        # add a new axis that takes up the entire top right of the figure
        bbox0 = bottom_left.bbox.transformed(f4.transFigure.inverted())
        bbox1 = top_right.bbox.transformed(f4.transFigure.inverted())
        ax = f4.add_axes([bbox0.xmin, bbox0.ymin, bbox1.xmax - bbox0.xmin, bbox1.ymax - bbox0.ymin])
        ax.xaxis.tick_top()
        ax.set_xlabel('Frequency (THz)')
        ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
        ax.set_ylabel(f'Luminosity $L_\\nu$ (10$^{{{np.log10(yscale):.0f}}}$ W Hz$^{{-1}}$)')
        ax.yaxis.set_label_position('right')

        # turn the original top right axis back off
        top_right.set_frame_on(False)
        top_right.xaxis.set_major_locator(plt.NullLocator())
        top_right.set_xlabel('')
        top_right.yaxis.set_major_locator(plt.NullLocator())
        top_right.set_ylabel('')

    for row in epoch1:
        ax.errorbar(row['freq'], row['lum'] / yscale, row['dlum'] / yscale, marker='o', **row['filter'].plotstyle)
    ax.plot(xfit, yfit.T / yscale, color='k', alpha=0.05)

    if save_plot_as:
        f4.savefig(save_plot_as)
        print('saving figure as ' + save_plot_as)

    return f4


def plot_bolometric_results(t0, save_plot_as=None):
    """
    Plot the parameters and bolometric light curves that result from fitting the spectral energy distribution

    Parameters
    ----------
    t0 : lightcurve_fitting.lightcurve.LC
        Table containing the results from fitting the spectral energy distribution
    save_plot_as : str, optional
        Filename to which to save the plot.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure object containing the plot
    """
    fig, axarr = plt.subplots(3, figsize=(6, 12), sharex=True)

    axarr[0].errorbar(t0['MJD'], t0['lum'], t0['dlum'], marker='.', ls='none', color='k', label='bolometric, curve_fit')
    axarr[0].plot(t0['MJD'], t0['L_opt'], marker='.', ls='none', color='C0', label='pseudobolometric, curve_fit')
    axarr[0].errorbar(t0['MJD'], t0['L_mcmc'], (t0['dL_mcmc0'], t0['dL_mcmc1']), marker='.', ls='none', color='C1',
                      label='pseudobolometric, MCMC')
    axarr[0].plot(t0['MJD'], t0['L_int'], marker='.', ls='none', color='C2', label='pseudobolometric, integration')
    axarr[0].legend()

    axarr[0].set_yscale('log')
    axarr[0].set_ylabel('Luminosity (W)')

    axarr[1].errorbar(t0['MJD'], t0['radius'], t0['dradius'], color='C0', marker='.', ls='none')
    axarr[1].errorbar(t0['MJD'], t0['radius_mcmc'], (t0['dradius0'], t0['dradius1']), color='C1', marker='.', ls='none')
    axarr[1].set_ylabel('Radius ($1000 R_\\odot$)')

    axarr[2].errorbar(t0['MJD'], t0['temp'], t0['dtemp'], color='C0', marker='.', ls='none')
    axarr[2].errorbar(t0['MJD'], t0['temp_mcmc'], (t0['dtemp0'], t0['dtemp1']), color='C1', marker='.', ls='none')
    axarr[2].set_ylabel('Temperature (kK)')
    axarr[2].set_xlabel('MJD')

    fig.tight_layout()
    if save_plot_as is not None:
        fig.savefig(save_plot_as)

    return fig


def group_by_epoch(lc, res=1., also_group_by=()):
    """
    Group a light curve into epochs that will be treated as single spectral energy distributions.

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table containing the observed photometry. Must contain times in an "MJD" column.
    res : float, optional
        Approximate resolution for grouping, in days. Default: 1 day.

    Returns
    -------
    epochs.groups : astropy.table.TableGroups
        Iterable containing single-epoch spectral energy distributions
    """
    x = lc['MJD'].data / res
    frac = np.median(x - np.trunc(x))
    lc['bin'] = np.round(x - frac + np.round(frac)) * res
    group_by = ['bin'] + list(also_group_by)
    for col in also_group_by:
        if np.ma.is_masked(lc[col]):
            lc[col] = lc[col].filled()
    epochs = lc.group_by(group_by)
    return epochs.groups


sigma_sb = const.sigma_sb.to(u.W / (1000. * u.Rsun) ** 2 / u.kK ** 4).value


def stefan_boltzmann(temp, radius, dtemp, drad, covTR):
    """
    Calculate blackbody luminosity and associated uncertainty using the Stefan-Boltzmann law

    Parameters
    ----------
    temp : float, array-like
        Temperature in kilokelvins
    radius : float, array-like
        Radius in units of 1000 solar radii
    dtemp : float, array-like
        Uncertainty in the temperature in kilokelvins
    drad : float, array-like
        Uncertainty in the radius in units of 1000 solar radii
    covTR : float, array-like
        Covariance between the temperature and radius

    Returns
    -------
    lum : float, array-like
        Luminosity in watts
    dlum : float, array-like
        Uncertainty in the luminosity in watts
    """
    lum = 4 * np.pi * radius ** 2 * sigma_sb * temp ** 4
    dlum = 8 * np.pi * sigma_sb * (radius ** 2 * temp ** 8 * drad ** 2
                                   + 4 * radius ** 4 * temp ** 6 * dtemp ** 2
                                   + 4 * radius ** 3 * temp ** 7 * covTR) ** 0.5
    return lum, dlum


def median_and_unc(x, perc_contained=68.):
    """
    Calculate the equal-tailed credible interval, centered on the median, for a sample of data

    Parameters
    ----------
    x : array-like
        The data sample
    perc_contained : float, optional
        The percentage of the probability to be contained in the interval. Default: 68% (1σ)

    Returns
    -------
    median : float
        The median of ``x``
    lower : float
        The lower boundary of the credible interval
    upper : float
        The upper boundary of the credible interval
    """
    q = 50. + np.array([-perc_contained / 2., 0., perc_contained / 2.])
    percentiles = np.percentile(x, q, axis=0)
    median = percentiles[1]
    lower, upper = np.diff(percentiles, axis=0)
    return median, lower, upper


def blackbody_lstsq(epoch1, z, p0=None, T_range=(1., 100.), R_range=(0.01, 1000.), cutoff_freq=np.inf):
    """
    Fit a blackbody spectrum to a spectral energy distribution using :math:`χ^2` minimization

    Parameters
    ----------
    epoch1 : lightcurve_fitting.lightcurve.LC
        A single "epoch" of photometry that defines the observed spectral energy distribution
    z : float
        Redshift between the blackbody (rest frame) and the filters (observed frame)
    p0 : list, tuple, array-like, optional
        Initial guess for [temperature (kK), radius (1000 Rsun)]. Default: ``[10., 10.]``
    T_range : tuple, list, array-like, optional
        Range of allowed temperatures (in kilokelvins) in the prior. Default: ``(1., 100.)``
    R_range : tuple, list, array-like, optional
        Range of allowed radii (in 1000 solar radii) in the prior. Default: ``(0.01, 1000.)``
    cutoff_freq : float, optional
        Cutoff frequency (in terahertz) for a modified blackbody spectrum (see https://doi.org/10.3847/1538-4357/aa9334)

    Returns
    -------
    temp : float
        Best-fit blackbody temperature in kilokelvins
    radius : float
        Best-fit blackbody radius in units of 1000 solar radii
    dtemp : float
        Uncertainty in the temperature in kilokelvins
    drad : float
        Uncertainty in the radius in units of 1000 solar radii
    lum : float
        Blackbody luminosity implied by the best-fit temperature and radius
    dlum : float
        Uncertainty in the blackbody luminosity
    L_opt : float
        Pseudobolometric luminosity from integrating the best-fit blackbody spectrum over the U-I filters
    """
    if p0 is None:
        p0 = [10., 10.]

    def planck_cutoff(nu, T, R):
        return planck_fast(nu, T, R, cutoff_freq)

    with warnings.catch_warnings():
        if len(epoch1) <= 2:
            warnings.simplefilter('ignore', OptimizeWarning)
        p0, cov = curve_fit(planck_cutoff, epoch1['freq'] * (1. + z), epoch1['lum'], p0=p0,
                            bounds=([T_range[0], R_range[0]], [T_range[1], R_range[1]]))
    temp, radius = p0
    dtemp, drad = np.sqrt(np.diag(cov))
    lum, dlum = stefan_boltzmann(temp, radius, dtemp, drad, cov[0, 1])
    L_opt = pseudo(temp, radius, z, cutoff_freq=cutoff_freq)
    return temp, radius, dtemp, drad, lum, dlum, L_opt


def integrate_sed(epoch1):
    """
    Directly integrate a spectral energy distribution using the trapezoidal rule

    Parameters
    ----------
    epoch1 : lightcurve_fitting.lightcurve.LC
        A single "epoch" of photometry that defines the observed spectral energy distribution

    Returns
    -------
    L_int : float
        Luminosity in watts
    """
    epoch1.sort('freq')
    freqs = np.insert(epoch1['freq'], 0, epoch1['freq'][0] - epoch1['dfreq'][0])
    lums = np.insert(epoch1['lum'], 0, 0)
    freqs = np.append(freqs, epoch1['freq'][-1] + epoch1['dfreq'][-1])
    lums = np.append(lums, 0)
    L_int = np.trapz(lums * epoch1['lum'].unit, freqs * epoch1['freq'].unit).to(u.W).value
    return L_int


def calc_colors(epoch1, colors):
    """
    Calculate colors from a spectral energy distribution

    Parameters
    ----------
    epoch1 : lightcurve_fitting.lightcurve.LC
        A single "epoch" of photometry that defines the observed spectral energy distribution
    colors : list
        A list of colors to calculate, e.g., ``['U-B', 'B-V', 'g-r', 'r-i']``

    Returns
    -------
    mags : list
        A list of the calculated colors
    dmags : list
        A list of uncertainties on the calculated colors, from propagating errors on the photometry points
    lolims : list
        A list of booleans: True if the first filter is a nondetection or is absent from ``epoch1``
    uplims : list
        A list of booleans: True if the second filter is a nondetection or is absent from ``epoch1``
    """
    mags = []
    dmags = []
    lolims = []
    uplims = []
    for color in colors:
        f0, f1 = [filtdict[f] for f in color.split('-')]
        if f0 in epoch1['filter'] and f1 in epoch1['filter']:
            m0, dm0, n0 = epoch1.where(filter=f0)[['absmag', 'dmag', 'nondet']][0]
            m1, dm1, n1 = epoch1.where(filter=f1)[['absmag', 'dmag', 'nondet']][0]
            if n0 and n1:
                m0_m1 = np.nan
            else:
                m0_m1 = m0 - m1
            dm0_m1 = (dm0 ** 2. + dm1 ** 2.) ** 0.5
            mags.append(m0_m1)
            dmags.append(dm0_m1)
            lolims.append(n0)
            uplims.append(n1)
        else:
            mags.append(np.nan)
            dmags.append(np.nan)
            lolims.append(True)
            uplims.append(True)
    return mags, dmags, lolims, uplims


def plot_color_curves(t, colors=None, fmt='o', limit_length=0.1, xcol='MJD'):
    """
    Plot the color curves calculated by :func:`.calculate_bolometric`.

    Parameters
    ----------
    t : lightcurve_fitting.lightcurve.LC
        Output table from :func:`.calculate_bolometric`
    colors : list, optional
        List of colors to plot, e.g., ``['g-r', 'r-i']``. By default, plot all recognizable colors.
    fmt : str, optional
        Format string, passed to :func:`matplotlib.pyplot.errorbar`
    limit_length : float, optional
        Length (in data units) of the upper and lower limit markers

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure object containing the plot
    """
    if colors is None:
        colors = []
        for col in t.colnames:
            if col.split('-')[0] in filtdict and not (t.has_masked_values and t.mask[col].all()):
                colors.append(col)
    fig = plt.figure()
    for c in colors:
        dcolor_colname = f'd({c})'
        if t.has_masked_values and t.mask[dcolor_colname].any():
            dcolor = t[dcolor_colname].filled(limit_length)
        else:
            dcolor = t[dcolor_colname]
        plt.errorbar(t[xcol], t[c], dcolor, (t[f'd{xcol}0'], t[f'd{xcol}1']), fmt=fmt,
                     lolims=t[f'lolims({c})'], uplims=t[f'uplims({c})'], label=f'${c}$')
    plt.xlabel(xcol)
    plt.ylabel('Color (mag)')
    plt.legend()
    return fig


def calculate_bolometric(lc, z=0., outpath='.', res=1., nwalkers=10, burnin_steps=200, steps=100, priors=None,
                         save_table_as=None, min_nfilt=3, cutoff_freq=np.inf, show=False, colors=None, do_mcmc=True,
                         save_chains=False, use_sigma=False, sigma_type='relative', also_group_by=()):
    """
    Calculate the full bolometric light curve from a table of broadband photometry

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table of broadband photometry including columns "MJD", "mag", "dmag", "filt"
    z : float, optional
        Redshift between the emission source and the observed filter. Default: 0.
    outpath : str, optional
        Directory to which to save the corner plots and MCMC chains. Default: current directory
    res : float, optional
        Approximate resolution for grouping, in days. Default: 1 day.
    nwalkers : int, optional
        Number of walkers (chains) to use for fitting. Default: 10
    burnin_steps : int, optional
        Number of MCMC steps before convergence. This part of the history is discarded. Default: 200
    steps : int, optional
        Number of MCMC steps after convergence. This part of the history is used to calculate paramers. Default: 100.
    priors : list, optional
        Prior probability distributions for temperature (in kilokelvins) and radius (in 1000 solar radii). Available
        priors: :class:`.models.UniformPrior`, :class:`.models.LogUniformPrior`, :class:`.models.GaussianPrior`.
        Default: ``T = Uniform(1., 100.)``, ``R = LogUniform(0.01, 1000.)``, ``σ = Gaussian(0., 10., stddev=1.)``.
    save_table_as : str, optional
        Filename to which to save the output table of blackbody parameters and bolometric luminosities
    min_nfilt : int, optional
        Minimum number of distinct observed filters required to calculate a luminosity. Default: 3
    cutoff_freq : float, optional
        Cutoff frequency (in terahertz) for a modified blackbody spectrum (see https://doi.org/10.3847/1538-4357/aa9334)
    show : bool, optional
        Plot the chain histories, and display all plots at the end of the script. Default: only save the corner plot
    colors : list, optional
        A list of colors to calculate, e.g., ``['U-B', 'B-V', 'g-r', 'r-i']``
    do_mcmc : bool, optional
        If True (default), also fit the spectral energy distribution with an MCMC routine. This is slower but gives
        more realistic uncertainties.
    save_chains : bool, optional
        If True, save the MCMC chain histories to the directory ``outpath``. Default: only save the corner plot
    use_sigma : bool, optional
        Include an intrinsic scatter parameter in the MCMC fit. Default: False.
    sigma_type : str, optional
        If 'relative' (default), sigma will be in units of the individual photometric uncertainties.
        If 'absolute', sigma will be in units of the median photometric uncertainty.
    also_group_by : list, optional
        Group by these columns in addition to epoch

    Returns
    -------
    t0 : lightcurve_fitting.lightcurve.LC
        Table containing the blackbody parameters, bolometric luminosities, and (optionally) colors
    """

    if colors is None:
        colors = []

    use_src = 'source' in lc.colnames
    t0 = LC(names=['MJD', 'dMJD0', 'dMJD1',
                   'temp', 'radius', 'dtemp', 'dradius',  # best fit from scipy.curve_fit
                   'lum', 'dlum',  # total bolometric luminosity from scipy.curve_fit
                   'L_opt',  # pseudobolometric luminosity from scipy.curve_fit
                   'temp_mcmc', 'radius_mcmc', 'dtemp0', 'dtemp1', 'dradius0', 'dradius1',  # best fit from MCMC
                   'L_mcmc', 'dL_mcmc0', 'dL_mcmc1',  # pseudobolometric luminosity from MCMC
                   'L_int',  # pseudobolometric luminosity from direct integration of the SED
                   'npoints']
            + colors + ['d({})'.format(c) for c in colors] + ['lolims({})'.format(c) for c in colors]
            + ['uplims({})'.format(c) for c in colors] + ['filts'] + (['source'] if use_src else []),
            dtype=[float, float, float, float, float, float, float, float, float, float, float, float, float, float,
                   float, float, float, float, float, float, int]
            + [float] * 2 * len(colors) + [bool] * 2 * len(colors) + ['S6'] + ([lc['source'].dtype] if use_src else []),
            masked=True)

    if priors is None:
        priors = [UniformPrior(1., 100.), LogUniformPrior(0.01, 1000.)]
        if use_sigma:
            priors.append(GaussianPrior(0., 10.))

    sampler = None
    lc = lc[np.isfinite(lc['dmag']) & (lc['dmag'] > 0.)]
    for epoch1 in group_by_epoch(lc, res, also_group_by):
        epoch1.sn = lc.sn
        epoch1.calcFlux()
        epoch1 = epoch1.bin(delta=np.inf)
        epoch1.calcMag()
        epoch1.calcAbsMag()
        epoch1.calcLum()

        epoch1['freq'] = u.Quantity([f.freq_eff for f in epoch1['filter']])
        epoch1['dfreq'] = u.Quantity([f.dfreq for f in epoch1['filter']])

        epoch1['lum'].unit = u.W / u.Hz
        epoch1['dlum'].unit = u.W / u.Hz

        filts = set(epoch1.where(nondet=False)['filter'].data)
        nfilt = len(filts)
        if nfilt < min_nfilt:
            continue

        if nfilt > 1:
            p0 = np.array([10., 10.])
        elif sampler is not None:
            priors[0] = gaussian_kde(sampler.flatchain[:, 0]).logpdf
            p0 = np.median(sampler.flatchain, axis=0)
        else:
            continue

        mjdavg, dmjd0, dmjd1 = median_and_unc(epoch1['MJD'], 100.)
        filtstr = ''.join([f.char for f in sorted(filts)])

        # blackbody - least squares
        T_range = (priors[0].p_min, priors[0].p_max)
        R_range = (priors[1].p_min, priors[1].p_max)
        try:
            temp, radius, dtemp, drad, lum, dlum, L_opt = blackbody_lstsq(epoch1, z, p0, T_range, R_range, cutoff_freq)
            p0 = np.array([temp, radius])
        except RuntimeError:  # optimization failed
            temp = radius = dtemp = drad = lum = dlum = L_opt = np.nan

        rng = np.random.default_rng()
        starting_guesses = rng.normal(size=(nwalkers, 2)) + p0
        starting_guesses[starting_guesses <= 0.] = 1.
        labels = ['T (kK)', 'R (1000 R$_\\odot$)']
        if use_sigma:
            starting_guesses = np.append(starting_guesses, np.abs(rng.normal(size=(nwalkers, 1))), axis=1)
            labels.append('$\\sigma$')

        # blackbody - MCMC
        try:
            if not do_mcmc:
                raise ValueError
            spectrum_kwargs = {'cutoff_freq': cutoff_freq}
            sampler = spectrum_mcmc(planck_fast, epoch1, priors, starting_guesses, z=z, spectrum_kwargs=spectrum_kwargs,
                                    outpath=outpath, nwalkers=nwalkers, burnin_steps=burnin_steps, steps=steps,
                                    show=show, save_chains=save_chains, use_sigma=use_sigma, sigma_type=sigma_type,
                                    labels=labels)
            L_mcmc_opt = pseudo(sampler.flatchain[:, 0], sampler.flatchain[:, 1], z, cutoff_freq=cutoff_freq)
            (T_mcmc, R_mcmc), (dT0_mcmc, dR0_mcmc), (dT1_mcmc, dR1_mcmc) = median_and_unc(sampler.flatchain[:, :2])
            L_mcmc, dL_mcmc0, dL_mcmc1 = median_and_unc(L_mcmc_opt)
        except ValueError as e:
            print(e)
            T_mcmc = R_mcmc = dT0_mcmc = dR0_mcmc = dT1_mcmc = dR1_mcmc = L_mcmc = dL_mcmc0 = dL_mcmc1 = np.nan

        # direct integration
        L_int = integrate_sed(epoch1)

        # color calculation
        if colors is None:
            colors = []
        color_mags, color_dmags, color_lolims, color_uplims = calc_colors(epoch1, colors)

        row = [mjdavg, dmjd0, dmjd1,
               temp, radius, dtemp, drad, lum, dlum, L_opt,
               T_mcmc, R_mcmc, dT0_mcmc, dT1_mcmc, dR0_mcmc, dR1_mcmc, L_mcmc, dL_mcmc0, dL_mcmc1,
               L_int, nfilt] + color_mags + color_dmags
        row_bool = color_lolims + color_uplims
        row_string = [filtstr] + ([epoch1['source'][0]] if use_src else [])
        mask = np.concatenate([np.isnan(row), np.zeros_like(row_bool, dtype=bool),
                               ~np.array([bool(rs) for rs in row_string])])
        t0.add_row(row + row_bool + row_string, mask=mask)

    if save_table_as is not None and t0:
        t0.write(save_table_as, format='ascii.fixed_width_two_line', overwrite=True)

    return t0

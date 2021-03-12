import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
import corner
from .models import CompanionShocking, scale_sifto, flat_prior
from pkg_resources import resource_filename


def lightcurve_mcmc(lc, model, priors=None, p_min=None, p_max=None, p_lo=None, p_up=None,
                    nwalkers=100, nsteps=1000, nsteps_burnin=1000, model_kwargs=None,
                    show=False, save_sampler_as='', use_sigma=False):
    """
    Fit an analytical model to observed photometry using a Markov-chain Monte Carlo routine

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table of broadband photometry including columns "MJD", "mag", "dmag", "filt"
    model : lightcurve_fitting.models.Model
        The model to fit to the light curve. Available models: :class:`models.ShockCooling`,
        :class:`models.ShockCooling2`, :class:`models.CompanionShocking`
    priors : list, optional
        Prior probability distributions for each model parameter. Available priors: :func:`models.flat_prior` (default),
        :func:`models.log_flat_prior`
    p_min, p_max : list, optional
        Lower bounds on the priors for each parameter. Omit individual bounds using :mod:`-numpy.inf`.
    p_max : list, optional
        Upper bounds on the priors for each parameter. Omit individual bounds using :mod:`numpy.inf`.
    p_lo : list, optional
        Lower bounds on the starting guesses for each paramter. Default: equal to ``p_min``.
    p_up : list, optional
        Upper bounds on the starting guesses for each parameter. Default: equal to ``p_max``.
    nwalkers : int, optional
        Number of walkers (chains) for the MCMC routine. Default: 100
    nsteps : int, optional
        Number of steps (iterations) for the MCMC routine, excluding burn-in. Default: 1000
    nsteps_burnin : int, optional
        Number of steps (iterations) for the MCMC routine during burn-in. Default: 1000
    model_kwargs : dict, optional
        Keyword arguments to be passed to the model
    show : bool, optional
        If True, plot and display the chain histories
    save_sampler_as : str, optional
        Save the aggregated chain histories to this filename
    use_sigma : bool, optional
        If True, treat the last parameter as an intrinsic scatter parameter that does not get passed to the model

    Returns
    -------
    sampler : emcee.EnsembleSampler
        EnsembleSampler object containing the results of the fit
    """

    if model_kwargs is None:
        model_kwargs = {}

    lc.calcAbsMag()
    lc.calcLum()

    f = lc['filter'].data
    t = lc['MJD'].data
    y = lc['lum'].data
    dy = lc['dlum'].data

    if model == CompanionShocking:
        scale_sifto(lc)

    if use_sigma:
        model.axis_labels.append('$\\sigma$')

    ndim = model.nparams + use_sigma

    if priors is None:
        priors = [flat_prior] * ndim
    elif len(priors) != ndim:
        raise Exception('priors must have length {:d}'.format(ndim))

    if p_min is None:
        p_min = np.tile(-np.inf, ndim)
    elif len(p_min) == ndim:
        p_min = np.array(p_min, float)
    else:
        raise Exception('p_min must have length {:d}'.format(ndim))

    if p_max is None:
        p_max = np.tile(np.inf, ndim)
    elif len(p_max) == ndim:
        p_max = np.array(p_max, float)
    else:
        raise Exception('p_max must have length {:d}'.format(ndim))

    if p_lo is None:
        p_lo = p_min
    elif len(p_lo) == ndim:
        p_lo = np.array(p_lo, float)
    else:
        raise Exception('p_lo must have length {:d}'.format(ndim))

    if p_up is None:
        p_up = p_max
    elif len(p_up) == ndim:
        p_up = np.array(p_up, float)
    else:
        raise Exception('p_up must have length {:d}'.format(ndim))

    def log_posterior(p):
        if np.any(p < p_min) or np.any(p > p_max):
            return -np.inf
        else:
            log_prior = 0.
            for prior, p_i in zip(priors, p):
                log_prior += np.log(prior(p_i))
            y_fit = model(t, f, *p, **model_kwargs)
            sigma = dy * np.sqrt(1. + p[-1] ** 2.) if use_sigma else dy
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2.) + ((y - y_fit) / sigma) ** 2.)
            return log_prior + log_likelihood

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    starting_guesses = np.random.rand(nwalkers, ndim) * (p_up - p_lo) + p_lo
    pos, _, _ = sampler.run_mcmc(starting_guesses, nsteps_burnin)
    if show:
        f1, ax1 = plt.subplots(ndim, figsize=(6, 2 * ndim))
        for i in range(ndim):
            ax1[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax1[i].set_ylabel(model.axis_labels[i])
        ax1[0].set_title('During Burn In')
        ax1[-1].set_xlabel('Step Number')

    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    if save_sampler_as:
        np.save(save_sampler_as, sampler.flatchain)
        print('saving sampler.flatchain as ' + save_sampler_as)

    if show:
        f2, ax2 = plt.subplots(ndim, figsize=(6, 2 * ndim))
        for i in range(ndim):
            ax2[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax2[i].set_ylabel(model.axis_labels[i])
        ax2[0].set_title('After Burn In')
        ax2[-1].set_xlabel('Step Number')
        plt.show()

    return sampler


def lightcurve_corner(lc, model, sampler_flatchain, model_kwargs=None,
                      num_models_to_plot=100, lcaxis_posn=(0.7, 0.55, 0.2, 0.4),
                      filter_spacing=0.5, tmin=None, tmax=None, t0_offset=None, save_plot_as=''):
    """
    Plot the posterior distributions in a corner (pair) plot, with an inset showing the observed and model light curves.

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table of broadband photometry including columns "MJD", "mag", "dmag", "filt"
    model : lightcurve_fitting.models.Model
        The model that was fit to the light curve.
    sampler_flatchain : array-like
        2D array containing the aggregated MCMC chain histories
    model_kwargs : dict, optional
        Keyword arguments to be passed to the model
    num_models_to_plot : int, optional
        Number of model realizations to plot in the light curve inset. Default: 100
    lcaxis_posn : tuple, optional
        Light curve inset position and size specification in figure units: (left, bottom, width, height)
    filter_spacing : float, optional
        Spacing between filters in the light curve inset, in units determined by the order of magnitude of the
        luminosities. Default: 0.5
    tmin, tmax : float, optional
        Starting and ending times for which to plot the models in the light curve inset. Default: determined by the
        time range of the observed light curve.
    t0_offset : float, optional
        Reference time on the horizontal axis of the light curve inset. Default: determined by the starting time of
        the model light curve.
    save_plot_as : str, optional
        Filename to which to save the resulting plot

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure object containing the plot
    """
    if model_kwargs is None:
        model_kwargs = {}
    plt.style.use(resource_filename('lightcurve_fitting', 'serif.mplstyle'))

    choices = np.random.choice(sampler_flatchain.shape[0], num_models_to_plot)
    ps = sampler_flatchain[choices].T

    sampler_flatchain_corner = sampler_flatchain.copy()
    if 't_0' in model.input_names:
        i_t0 = model.input_names.index('t_0')
        if t0_offset is None:
            t0_offset = np.floor(sampler_flatchain_corner[:, i_t0].min())
        if t0_offset != 0.:
            sampler_flatchain_corner[:, i_t0] -= t0_offset
            model.axis_labels[i_t0] = '$t_0 - {:.0f}$ (d)'.format(t0_offset)

    fig = corner.corner(sampler_flatchain_corner, labels=model.axis_labels)
    corner_axes = np.array(fig.get_axes()).reshape(sampler_flatchain.shape[-1], sampler_flatchain.shape[-1])

    for ax in np.diag(corner_axes):
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')

    ax = fig.add_axes(lcaxis_posn)
    if tmin is None:
        tmin = np.min(lc['MJD'])
    if tmax is None:
        tmax = np.max(lc['MJD'])
    xfit = np.arange(tmin, tmax, 0.1)
    ufilts = np.unique(lc['filter'])
    y_fit = model(xfit, ufilts, *ps, **model_kwargs)

    mjd_offset = np.floor(tmin)
    if y_fit.max() > 0.:
        yscale = 10. ** np.round(np.log10(y_fit.max()))
    else:
        yscale = 1.
    offset = -len(ufilts) // 2 * filter_spacing
    for filt, yfit in zip(ufilts, y_fit):
        offset += filter_spacing
        lc_filt = lc.where(filter=filt)
        ax.errorbar(lc_filt['MJD'] - mjd_offset, lc_filt['lum'] / yscale + offset, lc_filt['dlum'] / yscale,
                    ls='none', marker='o', **filt.plotstyle)
        ax.plot(xfit - mjd_offset, yfit / yscale + offset, color=filt.linecolor, alpha=0.05)
        txt = '${}{:+.1f}$'.format(filt.name, offset) if offset else filt.name
        ax.text(1.03, yfit[-1, 0] / yscale + offset, txt, color=filt.textcolor,
                ha='left', va='center', transform=ax.get_yaxis_transform())
    ax.set_xlabel('MJD $-$ {:.0f}'.format(mjd_offset))
    ax.set_ylabel('Luminosity $L_\\nu$ (10$^{{{:.0f}}}$ erg s$^{{-1}}$ Hz$^{{-1}}$) + Offset'
                  .format(np.log10(yscale) + 7))  # W --> erg / s

    paramtexts = format_credible_interval(sampler_flatchain, varnames=model.input_names, units=model.units)
    fig.text(0.45, 0.95, '\n'.join(paramtexts), va='top', ha='center', fontdict={'size': 'large'})
    if save_plot_as:
        fig.savefig(save_plot_as)
        print('saving figure as ' + save_plot_as)

    return fig


def format_credible_interval(x, sigfigs=1, percentiles=(15.87, 50., 84.14), axis=0, varnames=None, units=None):
    """
    Use LaTeX to format an equal-tailed credible interval with a given number of significant figures in the uncertainty

    Parameters
    ----------
    x : array-like
        Data from which to calculate the credible interval
    sigfigs : int, optional
        Number of significant figures in the uncertainty. Default: 1
    percentiles : tuple, optional
        Percentiles for the (lower, center, upper) of the credible interval. Default: ``(15.87, 50., 84.14)``
        (median +/- 1σ)
    axis : int, optional
        Axis of ``x`` along which to calculate the credible intervals. Default: 0
    varnames : list, optional
        Variable names to be equated with the credible intervals. Default: no variable names
    units : list, optional
        Units to be applied to the credible intervals. Default: no units

    Returns
    -------
    paramtexts : str
        The formatted credible intervals
    """
    quantiles = np.percentile(x, percentiles, axis=axis).T
    uncertainties = np.diff(quantiles)
    smaller_unc = np.amin(uncertainties, axis=-1)
    log_unc = np.log10(smaller_unc)
    roundto = sigfigs - np.ceil(log_unc).astype(int)
    quantiles = np.atleast_2d(quantiles)
    uncertainties = np.atleast_2d(uncertainties)
    roundto = np.atleast_1d(roundto)
    texstrings = []
    for quant, unc, dec in zip(quantiles, uncertainties, roundto):
        center = np.round(quant[1], dec)
        lower, upper = np.round(unc, dec)
        if dec < 0:
            dec = 0
        if upper == lower:
            texstring = '{{:.{0:d}f}} \\pm {{:.{0:d}f}}'.format(dec).format(center, upper)
        else:
            texstring = '{{:.{0:d}f}}^{{{{+{{:.{0:d}f}}}}}}_{{{{-{{:.{0:d}f}}}}}}'.format(dec).format(center, upper,
                                                                                                      lower)
        texstrings.append(texstring)

    if varnames is None or units is None:
        paramtexts = texstrings
    else:
        paramtexts = []
        for var, value, unit in zip(varnames, texstrings, units):
            if isinstance(unit, u.quantity.Quantity):
                value = '({}) \\times 10^{{{:.1f}}}'.format(value, np.log10(unit.value)).replace('.0}', '}')
                unit = unit.unit
            paramtexts.append('${} = {}$ {:latex_inline}'.format(var, value, unit))

    return paramtexts

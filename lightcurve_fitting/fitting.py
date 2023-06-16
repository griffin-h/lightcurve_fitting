import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
import corner
from .models import UniformPrior, CompanionShocking, BaseCompanionShocking
from .lightcurve import filter_legend, flux2mag
from .filters import filtdict
from pkg_resources import resource_filename
import warnings

PRIOR_WARNING = 'The p_max/p_min keywords are deprecated. Use the priors keyword instead.'
MODEL_KWARGS_WARNING = 'The model_kwargs keyword is deprecated. These are now included in the model intialization.'


def lightcurve_mcmc(lc, model, priors=None, p_min=None, p_max=None, p_lo=None, p_up=None,
                    nwalkers=100, nsteps=1000, nsteps_burnin=1000, model_kwargs=None,
                    show=False, save_plot_as='', save_sampler_as='', use_sigma=False, sigma_type='relative'):
    """
    Fit an analytical model to observed photometry using a Markov-chain Monte Carlo routine

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table of broadband photometry including columns "MJD", "mag", "dmag", "filter"
    model : lightcurve_fitting.models.Model
        The model to fit to the light curve. Available models: :class:`.models.ShockCooling`,
        :class:`.models.ShockCooling2`, :class:`.models.ShockCooling3`, :class:`.models.CompanionShocking`,
        :class:`.models.CompanionShocking2`, :class:`.models.CompanionShocking3`
    priors : list, optional
        Prior probability distributions for each model parameter. Available priors:
        :class:`.models.UniformPrior` (default), :class:`.models.LogUniformPrior`, :class:`.models.GaussianPrior`
    p_min, p_max : list, optional
        DEPRECATED: Use `priors` instead
    p_lo : list
        Lower bounds on the starting guesses for each paramter
    p_up : list
        Upper bounds on the starting guesses for each parameter
    nwalkers : int, optional
        Number of walkers (chains) for the MCMC routine. Default: 100
    nsteps : int, optional
        Number of steps (iterations) for the MCMC routine, excluding burn-in. Default: 1000
    nsteps_burnin : int, optional
        Number of steps (iterations) for the MCMC routine during burn-in. Default: 1000
    model_kwargs : dict, optional
        DEPRECATED: Keyword arguments are now included in the model initialization
    show : bool, optional
        If True, plot and display the chain histories
    save_plot_as : str, optional
        Save a plot of the chain histories to this filename
    save_sampler_as : str, optional
        Save the aggregated chain histories to this filename
    use_sigma : bool, optional
        If True, treat the last parameter as an intrinsic scatter parameter that does not get passed to the model
    sigma_type : str, optional
        If 'relative' (default), sigma will be in units of the individual photometric uncertainties.
        If 'absolute', sigma will be in units of the median photometric uncertainty.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        EnsembleSampler object containing the results of the fit
    """

    if model_kwargs is not None:
        raise Exception(MODEL_KWARGS_WARNING)

    if model.output_quantity == 'flux':
        lc.calcFlux()
    elif model.output_quantity == 'lum':
        lc.calcAbsMag()
        lc.calcLum()

    if use_sigma and model.input_names[-1] != '\\sigma':
        model.input_names.append('\\sigma')
        model.units.append(u.dimensionless_unscaled)

    ndim = model.nparams

    # DEPRECATED
    if p_min is None:
        p_min = np.tile(-np.inf, ndim)
    elif len(p_min) == ndim:
        p_min = np.array(p_min, float)
        warnings.warn(PRIOR_WARNING)
    else:
        raise Exception(PRIOR_WARNING)

    # DEPRECATED
    if p_max is None:
        p_max = np.tile(np.inf, ndim)
    elif len(p_max) == ndim:
        p_max = np.array(p_max, float)
        warnings.warn(PRIOR_WARNING)
    else:
        raise Exception(PRIOR_WARNING)

    if p_lo is None:
        p_lo = p_min
    elif len(p_lo) == ndim:
        p_lo = np.array(p_lo, float)
    else:
        raise Exception('p_lo must have length {:d}'.format(ndim))

    if len(p_up) == ndim:
        p_up = np.array(p_up, float)
    else:
        raise Exception('p_up must have length {:d}'.format(ndim))

    if priors is None:
        priors = [UniformPrior(p0, p1) for p0, p1 in zip(p_min, p_max)]
    elif len(priors) != ndim:
        raise Exception('priors must have length {:d}'.format(ndim))

    for param, prior, p0, p1 in zip(model.input_names, priors, p_lo, p_up):
        if p0 < prior.p_min:
            raise Exception(f'starting guess for {param} (p_lo = {p0}) is outside prior (p_min = {prior.p_min})')
        if p1 > prior.p_max:
            raise Exception(f'starting guess for {param} (p_up = {p1}) is outside prior (p_max = {prior.p_max})')

    def log_posterior(p):
        log_prior = 0.
        for prior, p_i in zip(priors, p):
            log_prior += prior(p_i)
        if np.isinf(log_prior):
            return log_prior
        log_likelihood = model.log_likelihood(lc, p, use_sigma=use_sigma, sigma_type=sigma_type)
        return log_prior + log_likelihood

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    starting_guesses = np.random.rand(nwalkers, ndim) * (p_up - p_lo) + p_lo
    pos, _, _ = sampler.run_mcmc(starting_guesses, nsteps_burnin, progress=True, progress_kwargs={'desc': ' Burn-in'})

    if show or save_plot_as:
        fig, ax = plt.subplots(ndim, 2, figsize=(12., 2. * ndim))
        ax1 = ax[:, 0]
        for i in range(ndim):
            ax1[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax1[i].set_ylabel(model.axis_labels[i])
        ax1[0].set_title('During Burn In')
        ax1[-1].set_xlabel('Step Number')

    sampler.reset()
    sampler.run_mcmc(pos, nsteps, progress=True, progress_kwargs={'desc': 'Sampling'}, skip_initial_state_check=True)
    if save_sampler_as:
        np.save(save_sampler_as, sampler.flatchain)
        print('saving sampler.flatchain as ' + save_sampler_as)

    if show or save_plot_as:
        ax2 = ax[:, 1]
        for i in range(ndim):
            ax2[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax2[i].set_ylabel(model.axis_labels[i])
            ax2[i].yaxis.set_label_position('right')
            ax2[i].yaxis.tick_right()
        ax2[0].set_title('After Burn In')
        ax2[-1].set_xlabel('Step Number')
        fig.tight_layout()

        if save_plot_as:
            print('saving chain plot as ' + save_plot_as)
            fig.savefig(save_plot_as)

        if show:
            plt.show()

    return sampler


def lightcurve_corner(lc, model, sampler_flatchain, model_kwargs=None,
                      num_models_to_plot=100, lcaxis_posn=(0.7, 0.55, 0.2, 0.4),
                      filter_spacing=1., tmin=None, tmax=None, t0_offset=None, save_plot_as='', ycol=None,
                      textsize='medium', param_textsize='large', use_sigma=False, xscale='linear',
                      filters_to_model=None):
    """
    Plot the posterior distributions in a corner (pair) plot, with an inset showing the observed and model light curves.

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table of broadband photometry including columns "MJD", "mag", "dmag", "filter"
    model : lightcurve_fitting.models.Model
        The model that was fit to the light curve.
    sampler_flatchain : array-like
        2D array containing the aggregated MCMC chain histories
    model_kwargs : dict, optional
        DEPRECATED: Keyword arguments are now included in the model initialization
    num_models_to_plot : int, optional
        Number of model realizations to plot in the light curve inset. Default: 100
    lcaxis_posn : tuple, optional
        Light curve inset position and size specification in figure units: (left, bottom, width, height)
    filter_spacing : float, optional
        Spacing between filters in the light curve inset, in units determined by the order of magnitude of the
        luminosities. Default: 1.
    tmin, tmax : float, optional
        Starting and ending times for which to plot the models in the light curve inset. Default: determined by the
        time range of the observed light curve.
    t0_offset : float, optional
        Reference time for the explosion time in the corner plot and the horizontal axis of the light curve inset.
         Default: the earliest explosion time in `sampler_flatchain`, rounded down.
    save_plot_as : str, optional
        Filename to which to save the resulting plot
    ycol : str, optional
        Quantity to plot on the light curve inset. Choices: "lum", "flux", or "absmag". Default: model.output_quantity
    textsize : str, optional
        Font size for the x- and y-axis labels, as well as the tick labels. Default: 'medium'
    param_textsize : str, optional
        Font size for the parameter text. Default: 'large'
    use_sigma : bool, optional
        If True, treat the last parameter as an intrinsic scatter parameter that does not get passed to the model
    xscale : str, optional
        Scale for the x-axis of the model plot. Choices: "linear" (default) or "log".
    filters_to_model : list, set, optional
        (Unique) list of filters for which to calculate the model light curves. Default: all filters in `lc`.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure object containing the plot
    corner_ax : array-like
        Array of matplotlib.pyplot.Axes objects corresponding to the corner plot
    ax : matplotlib.pyplot.Axes
        Axes object for the light curve inset
    """
    if model_kwargs is not None:
        raise Exception(MODEL_KWARGS_WARNING)
    if ycol is None:
        ycol = model.output_quantity
    plt.style.use(resource_filename('lightcurve_fitting', 'serif.mplstyle'))
    if use_sigma and model.input_names[-1] != '\\sigma':
        model.input_names.append('\\sigma')
        model.units.append(u.dimensionless_unscaled)

    sampler_flatchain_corner = sampler_flatchain.copy()
    axis_labels_corner = model.axis_labels
    for var in ['t_0', 't_\\mathrm{max}']:
        if var in model.input_names:
            i_t0 = model.input_names.index(var)
            if t0_offset is None:
                t0_offset = np.floor(sampler_flatchain_corner[:, i_t0].min())
            if t0_offset != 0.:
                sampler_flatchain_corner[:, i_t0] -= t0_offset
                t0_offset_formatted = '{:f}'.format(t0_offset).rstrip('0').rstrip('.')
                axis_labels_corner[i_t0] = f'${var} - {t0_offset_formatted}$ (d)'

    fig = corner.corner(sampler_flatchain_corner, labels=axis_labels_corner, label_kwargs={'size': textsize})
    corner_axes = np.array(fig.get_axes()).reshape(sampler_flatchain.shape[-1], sampler_flatchain.shape[-1])
    for i in range(sampler_flatchain.shape[-1]):
        corner_axes[i, 0].tick_params(labelsize=textsize)
        corner_axes[-1, i].tick_params(labelsize=textsize)

    for ax in np.diag(corner_axes):
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')

    ax = fig.add_axes(lcaxis_posn)
    lightcurve_model_plot(lc, model, sampler_flatchain, model_kwargs, num_models_to_plot, filter_spacing,
                          tmin, tmax, ycol, textsize, ax, t0_offset, use_sigma, xscale, filters_to_model)

    paramtexts = format_credible_interval(sampler_flatchain, varnames=model.input_names, units=model.units)
    fig.text(0.45, 0.95, '\n'.join(paramtexts), va='top', ha='center', fontdict={'size': param_textsize})
    if save_plot_as:
        fig.savefig(save_plot_as)
        print('saving figure as ' + save_plot_as)

    return fig, corner_axes, ax


def lightcurve_model_plot(lc, model, sampler_flatchain, model_kwargs=None, num_models_to_plot=100, filter_spacing=1.,
                          tmin=None, tmax=None, ycol=None, textsize='medium', ax=None, mjd_offset=None, use_sigma=False,
                          xscale='linear', filters_to_model=None):
    """
    Plot the observed and model light curves.

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC
        Table of broadband photometry including columns "MJD", "mag", "dmag", "filter"
    model : lightcurve_fitting.models.Model
        The model that was fit to the light curve.
    sampler_flatchain : array-like
        2D array containing the aggregated MCMC chain histories
    model_kwargs : dict, optional
        DEPRECATED: Keyword arguments are now included in the model initialization.
    num_models_to_plot : int, optional
        Number of model realizations to plot in the light curve inset. Default: 100
    filter_spacing : float, optional
        Spacing between filters in the light curve inset, in units determined by the order of magnitude of the
        luminosities. Default: 1.
    tmin, tmax : float, optional
        Starting and ending times for which to plot the models in the light curve inset. Default: determined by the
        time range of the observed light curve.
    ycol : str, optional
        Quantity to plot on the light curve inset. Choices: "lum", "flux", or "absmag". Default: model.output_quantity
    textsize : str, optional
        Font size for the x- and y-axis labels, as well as the tick labels. Default: 'medium'
    ax : matplotlib.pyplot.Axes
        Axis on which to plot the light curves
    mjd_offset : float, optional
        Reference time on the horizontal axis of the light curve inset. Default: determined by the starting time of
        the model light curve.
    use_sigma : bool, optional
        If True, treat the last parameter as an intrinsic scatter parameter that does not get passed to the model
    xscale : str, optional
        Scale for the x-axis. Choices: "linear" (default) or "log".
    filters_to_model : list, set, optional
        (Unique) list of filters for which to calculate the model light curves. Default: all filters in `lc`.
    """
    if model_kwargs is not None:
        raise Exception(MODEL_KWARGS_WARNING)
    if ycol is None:
        ycol = model.output_quantity
    if ax is None:
        ax = plt.axes()
    if use_sigma and model.input_names[-1] != '\\sigma':
        model.input_names.append('\\sigma')
        model.units.append(u.dimensionless_unscaled)

    choices = np.random.choice(sampler_flatchain.shape[0], num_models_to_plot)
    ps = sampler_flatchain[choices].T

    if tmin is None:
        tmin = np.min(lc['MJD'])
    if tmax is None:
        tmax = np.max(lc['MJD'])
    xfit = np.geomspace(tmin, tmax, 1000) if xscale == 'log' else np.linspace(tmin, tmax, 1000)
    if filters_to_model is None:
        ufilts = np.unique(lc['filter'])
    else:
        ufilts = np.array([filtdict[f] for f in filters_to_model])
    if use_sigma:
        y_fit = model(xfit, ufilts, *ps[:-1])
    else:
        y_fit = model(xfit, ufilts, *ps)

    # for CompanionShocking, add SiFTO model as dashed lines
    if isinstance(model, CompanionShocking):
        y_fit1 = model.stretched_sifto(xfit, ufilts, *ps[3:5])
        y_fit1[ufilts == filtdict['r']] *= ps[5]
        y_fit1[ufilts == filtdict['i']] *= ps[6]
    elif isinstance(model, BaseCompanionShocking):
        y_fit1 = model.stretched_sifto(xfit, ufilts, *ps[3:7])
    else:
        y_fit1 = [None] * len(ufilts)

    if mjd_offset is None:
        mjd_offset = np.floor(tmin)
    if ycol == 'lum':
        dycol = 'dlum'
        yscale = 10. ** np.round(np.log10(y_fit.max()))
        ylabel = 'Luminosity $L_\\nu$ (10$^{{{:.0f}}}$ erg s$^{{-1}}$ Hz$^{{-1}}$) + Offset'.format(
            np.log10(yscale) + 7)  # W --> erg / s
    elif ycol == 'absmag':
        dycol = 'dmag'
        yscale = 1.
        ylabel = 'Absolute Magnitude + Offset'
        y_fit, _ = flux2mag(y_fit, zp=[[[filt.M0]] for filt in ufilts])
        if y_fit1[0] is not None:
            y_fit1, _ = flux2mag(y_fit1, zp=[[[filt.M0]] for filt in ufilts])
        ax.invert_yaxis()
    elif ycol == 'flux':
        dycol = 'dflux'
        yscale = 10. ** np.round(np.log10(y_fit.max()))
        ylabel = 'Flux $F_\\nu$ (10$^{{{:.0f}}}$ erg s$^{{-1}}$ m$^{{-2}}$ Hz$^{{-1}}$) + Offset'.format(
            np.log10(yscale) + 7)  # W --> erg / s
    else:
        raise ValueError(f'ycol="{ycol}" is not recognized. Use "lum", "absmag", "flux".')

    if xscale == 'log':
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
        lc = lc.where(MJD_min=mjd_offset)
    else:
        lc = lc.copy()
    lc['MJD'] -= mjd_offset
    lc[ycol] /= yscale
    lc[dycol] /= yscale
    lc.plot(xcol='MJD', ycol=ycol, offset_factor=filter_spacing, appmag_axis=False, tight_layout=False)
    plt.autoscale(False)
    _, labels, _ = filter_legend(np.array(ufilts), filter_spacing)
    for yfit, yfit1, filt, txt in zip(y_fit, y_fit1, ufilts, labels):
        offset = -filt.offset * filter_spacing
        ax.plot(xfit - mjd_offset, yfit / yscale + offset, color=filt.linecolor, alpha=0.05)
        if yfit1 is not None:
            ax.plot(xfit - mjd_offset, np.median(yfit1, axis=1) / yscale + offset, color=filt.linecolor, ls='--')
        ax.text(1.03, yfit[-1, 0] / yscale + offset, txt, color=filt.textcolor, fontdict={'size': textsize},
                ha='left', va='center', transform=ax.get_yaxis_transform())
    ax.set_xlabel('MJD $-$ {:f}'.format(mjd_offset).rstrip('0').rstrip('.'), size=textsize)
    ax.set_ylabel(ylabel, size=textsize)
    ax.tick_params(labelsize=textsize)


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
    roundto = sigfigs - np.floor(log_unc).astype(int) - 1

    # catch numbers that will have more sigfigs after rounding
    smaller_unc_round = [np.round(unc, dec) for unc, dec in zip(smaller_unc, roundto)]
    log_unc_round = np.log10(smaller_unc_round)
    roundto = sigfigs - np.floor(log_unc_round).astype(int) - 1

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

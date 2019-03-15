import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
import corner
import models


def lightcurve_mcmc(lc, model, priors=None, p_min=None, p_max=None, p_lo=None, p_up=None,
                    nwalkers=100, nsteps=1000, nsteps_burnin=1000, model_kwargs=None,
                    show=False, save_sampler_as=''):
    """
    Available models: models.ShockCooling, models.ShockCooling2

    Available priors: models.flat_prior (default), models.log_flat_prior

    p_min & p_max (optional) are bounds on the priors. Omit individual bounds using +/-np.inf.
    p_lo & p_up (optional) are bounds on the starting guesses. These default to p_min & p_max.
    You must specify either p_min & p_max or p_up & p_lo for each parameter!
    """

    if model_kwargs is None:
        model_kwargs = {}

    f = lc['filter'].data
    t = lc['MJD'].data
    y = lc['lum'].data
    dy = lc['dlum'].data

    if model == models.CompanionShocking:
        models.scale_sifto(lc)

    ndim = model.nparams

    if priors is None:
        priors = [models.flat_prior] * ndim
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
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + ((y - y_fit) / dy) ** 2)
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
    if show:
        f2, ax2 = plt.subplots(ndim, figsize=(6, 2 * ndim))
        for i in range(ndim):
            ax2[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax2[i].set_ylabel(model.axis_labels[i])
        ax2[0].set_title('After Burn In')
        ax2[-1].set_xlabel('Step Number')

    if save_sampler_as:
        np.save(save_sampler_as, sampler.flatchain)
        print('saving sampler.flatchain as ' + save_sampler_as)

    return sampler


def lightcurve_corner(lc, model, sampler_flatchain, model_kwargs={},
                      num_models_to_plot=100, lcaxis_posn=(0.7, 0.55, 0.2, 0.4),
                      filter_spacing=0.5, tmin=None, tmax=None, t0_offset=None, save_plot_as=''):
    plt.style.use('serif.mplstyle')

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
    corner_axes = np.array(fig.get_axes()).reshape(model.nparams, model.nparams)

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
    yscale = 10 ** np.round(np.log10(np.max(y_fit)))
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
    ax.set_xlabel('MJD - {:.0f}'.format(mjd_offset))
    ax.set_ylabel('Luminosity $L_\\nu$ (10$^{{{:.0f}}}$ erg s$^{{-1}}$ Hz$^{{-1}}$) + Offset'
                  .format(np.log10(yscale) + 7))  # W --> erg / s

    paramtexts = format_credible_interval(sampler_flatchain, varnames=model.input_names, units=model.units)
    fig.text(0.45, 0.95, '\n'.join(paramtexts), va='top', ha='center', fontdict={'size': 'large'})
    if save_plot_as:
        fig.savefig(save_plot_as)
        print('saving figure as ' + save_plot_as)

    return fig


def format_credible_interval(x, sigfigs=1, percentiles=(15.87, 50., 84.14), axis=0, varnames=None, units=None):
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

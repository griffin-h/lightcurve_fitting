import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
import emcee
import corner
import models

def lightcurve_mcmc(lc, model, priors=None, p_min=None, p_max=None, p_lo=None, p_up=None,
                    nwalkers=100, nsteps=1000, nsteps_burnin=1000, model_kwargs={},
                    show=False, save_sampler_as=''):
    '''
    Available models: models.ShockCooling, models.ShockCooling2

    Available priors: models.flat_prior (default), models.log_flat_prior

    p_min & p_max (optional) are bounds on the priors. Omit individual bounds using +/-np.inf.
    p_lo & p_up (optional) are bounds on the starting guesses. These default to p_min & p_max.
    You must specify either p_min & p_max or p_up & p_lo for each parameter!
    '''
    
    f = lc['filter'].data
    t = lc['MJD'].data
    y = lc['lum'].data
    dy = lc['dlum'].data
    
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
            for i, prior in enumerate(priors):
                log_prior += np.log(prior(p[i]))
            T, R, t_min, t_max = model(t, *p, **model_kwargs)
            y_fit = blackbody_to_filters(f, T, R, pointwise=True)
            log_likelihood = -0.5 * np.sum(np.log(2*np.pi*dy**2) + ((y - y_fit)/dy)**2)        
            return log_prior + log_likelihood

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    starting_guesses = np.random.rand(nwalkers, ndim) * (p_up - p_lo) + p_lo
    pos, _, _ = sampler.run_mcmc(starting_guesses, nsteps_burnin)
    if show:
        f1, ax1 = plt.subplots(ndim, figsize=(6, 2*ndim))
        for i in range(ndim):
            ax1[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax1[i].set_ylabel(model.axis_labels[i])
        ax1[0].set_title('During Burn In')
        ax1[-1].set_xlabel('Step Number')

    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    if show:
        f2, ax2 = plt.subplots(ndim, figsize=(6, 2*ndim))
        for i in range(ndim):
            ax2[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
            ax2[i].set_ylabel(model.axis_labels[i])
        ax2[0].set_title('After Burn In')
        ax2[-1].set_xlabel('Step Number')

    if save_sampler_as:
    	np.save(save_sampler_as, sampler.flatchain)
    	print('saving sampler.flatchain as ' + save_sampler_as)

    return sampler

def lightcurve_corner(lc, model, sampler_flatchain,
                      num_models_to_plot=100, lcaxis_posn=[0.7, 0.55, 0.2, 0.4],
                      filter_spacing=0.5, save_plot_as=''):
	
    if 'serif' in plt.style.available:
        plt.style.use('serif')
    
    choices = np.random.choice(sampler_flatchain.shape[0], num_models_to_plot)
    ps = sampler_flatchain[choices].T

    fig = corner.corner(sampler_flatchain, labels=model.axis_labels)
    corner_axes = np.array(fig.get_axes()).reshape(model.nparams, model.nparams)

    for ax in np.diag(corner_axes):
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')

    ax = fig.add_axes(lcaxis_posn)
    xfit = np.arange(np.min(lc['MJD']), np.max(lc['MJD']), 0.1)
    ufilts = np.unique(lc['filter'])
    T, R, tmin, tmax = model(xfit, *ps)
    y_fit = blackbody_to_filters(ufilts, T.flatten(), R.flatten())
    y_fit = np.rollaxis(y_fit.reshape(T.shape[0], T.shape[1], len(ufilts)), -1)

    yscale = 10**np.round(np.log10(np.max(y_fit)))
    offset = -len(ufilts) // 2 * filter_spacing
    for filt, yfit in zip(ufilts, y_fit):
        offset += filter_spacing
        lc_filt = lc.where(filter=filt)
        ax.errorbar(lc_filt['MJD'], lc_filt['lum'] / yscale + offset, lc_filt['dlum'] / yscale,
                    ls='none', marker='o', **filt.plotstyle)
        ax.plot(xfit, yfit / yscale + offset, color=filt.linecolor, alpha=0.05)
        txt = '${}{:+.1f}$'.format(filt.name, offset) if offset else filt.name
        ax.text(1.03, yfit[-1, 0] / yscale + offset, txt, color=filt.textcolor,
                ha='left', va='center', transform=ax.get_yaxis_transform())
    ax.set_xlabel('MJD')
    ax.set_ylabel('Luminosity $L_\\nu$ (10$^{{{:.0f}}}$ erg s$^{{-1}}$ Hz$^{{-1}}$) + Offset'
                  .format(np.log10(yscale) + 7)) # W --> erg / s

    paramtexts = format_credible_interval(sampler_flatchain, varnames=model.input_names, units=model.units)
    fig.text(0.45, 0.95, '\n'.join(paramtexts), va='top', ha='center', fontdict={'size': 'large'})
    if save_plot_as:
        fig.savefig(save_plot_as)
        print('saving figure as ' + save_figure_as)
    
    return fig

c1 = (const.h / const.k_B).to(u.kK / u.THz).value
c2 = 8 * np.pi**2 * (const.h / const.c**2).to(u.W / u.Hz / (1000 * u.Rsun)**2 / u.THz**3).value

def planck_fast(nu, T, R):
    return c2 * np.squeeze(np.outer(R**2, nu**3) / (np.exp(c1 * np.outer(T**-1, nu)) - 1)) # shape = (len(T), len(nu))

def blackbody_to_filters(filtobj, T, R, pointwise=False):
    if pointwise:
        return np.array([np.trapz(planck_fast(f.trans['freq'].data, t, r) * f.trans['T_norm_per_freq'].data,
                                   f.trans['freq'].data) for t, r, f in zip(T, R, filtobj)]) # shape = (len(T),)
    else:
        return np.array([np.trapz(planck_fast(f.trans['freq'].data, T, R) * f.trans['T_norm_per_freq'].data,
                                   f.trans['freq'].data) for f in filtobj]).T # shape = (len(T), len(filtobj))

def planck(nu, T, R, dT=0., dR=0., cov=0.):
    Lnu = planck_fast(nu, T, R)
    if not dT and not dR and not cov:
        return Lnu
    dlogLdT = c1 * nu * T**-2 / (1 - np.exp(-c1 * nu / T))
    dlogLdR = 2. / R
    dLnu = Lnu * (dlogLdT**2 * dT**2 + dlogLdR**2 * dR**2 + 2. * dlogLdT * dlogLdR * cov)**0.5
    return Lnu, dLnu

def format_credible_interval(x, sigfigs=2, percentiles=[15.87, 50., 84.14], axis=0, varnames=None, units=None):
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
            texstring = '{{:.{0:d}f}}^{{{{+{{:.{0:d}f}}}}}}_{{{{-{{:.{0:d}f}}}}}}'.format(dec).format(center, upper, lower)
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

import filters
import models
from lightcurve import LC

from astropy import constants as const
from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from scipy.optimize import curve_fit
import emcee
import corner
import os

for filt in filters.all_filters:
    filt.read_curve()

plt.style.use('serif')


def pseudo(temp, radius, z, filter0=filters.filtdict['I'], filter1=filters.filtdict['U'], cutoff_freq=np.inf):
    freq0 = (filter0.freq_eff - filter0.dfreq / 2.).value
    freq1 = (filter1.freq_eff + filter1.dfreq / 2.).value
    x_optical = np.arange(freq0, freq1)
    y_optical = models.planck_fast(x_optical * (1. + z), temp, radius, cutoff_freq)
    L_opt = np.trapz(y_optical) * 1e12  # dx = 1 THz
    return L_opt


def blackbody_mcmc(epoch1, z, p0=None, show=False, outpath='.', nwalkers=10, burnin_steps=200, steps=100,
                   T_range=(1., 100.), R_range=(0.01, 1000.), cutoff_freq=np.inf):
    y = epoch1['lum'].data
    dy = epoch1['dlum'].data
    filtobj = epoch1['filter'].data
    mjdavg = np.mean(epoch1['MJD'].data)

    if p0 is None:
        p0 = [10., 10.]

    def log_prior(p):  # p = [T, R]
        if (np.any(p[0] < T_range[0]) or np.any(p[0] > T_range[1])
                or np.any(p[1] < R_range[0]) or np.any(p[1] > R_range[1])):
            return -np.inf
        else:
            return -np.sum(np.log(p))

    def log_likelihood(p, filtobj, y, dy):
        y_fit = models.blackbody_to_filters(filtobj, p[0], p[1], z, cutoff_freq)
        return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + ((y - y_fit) / dy) ** 2, -1)

    def log_posterior(p, filtobj, y, dy):
        return log_prior(p) + log_likelihood(p, filtobj, y, dy)

    ndim = 2
    starting_guesses = np.random.randn(nwalkers, ndim) + p0
    starting_guesses[starting_guesses <= 0.] = 1.

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[filtobj, y, dy])
    pos, _, _ = sampler.run_mcmc(starting_guesses, burnin_steps)

    # Plotting
    if show:
        f1, ax2 = plt.subplots(ndim)
        for i in range(len(ax2)):
            ax2[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
    sampler.reset()
    sampler.run_mcmc(pos, steps)
    if show:
        f2, ax3 = plt.subplots(ndim)
        for i in range(len(ax3)):
            ax3[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)

    f4 = corner.corner(sampler.flatchain, labels=['T (kK)', 'R (1000 R$_\\odot$)'])
    ax = f4.get_axes()[1]
    ps = sampler.flatchain[np.random.choice(sampler.flatchain.shape[0], 100)].T
    xfit = np.arange(100., max(1000., min(filtobj).freq_eff.value))
    yfit = models.planck_fast(xfit * (1. + z), ps[0], ps[1], cutoff_freq)
    plt.sca(ax)
    epoch1.plot(xcol='freq', ycol='lum', offset_factor=0.)
    ax.plot(xfit, yfit.T, color='k', alpha=0.05)
    ax.set_frame_on(True)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.tick_top()
    ax.set_xlabel('Frequency (THz)')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.tick_right()
    ax.set_ylabel('$L_\\nu$ (W Hz$^{-1}$)')
    ax.yaxis.set_label_position('right')
    f4.tight_layout()

    os.makedirs(outpath, exist_ok=True)
    filename = os.path.join(outpath, f'{mjdavg:.1f}.png')
    print(filename)
    f4.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close(f4)

    return sampler


def plot_bolometric_results(t0, save_plot_as=None):
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


def group_by_epoch(lc, res=1.):
    x = lc['MJD'].data / res
    frac = np.median(x - np.trunc(x))
    lc['bin'] = np.round(x - frac + np.round(frac)) * res
    epochs = lc.group_by(['bin', 'source'])
    return epochs.groups


sigma_sb = const.sigma_sb.to(u.W / (1000. * u.Rsun) ** 2 / u.kK ** 4).value


def stefan_boltzmann(temp, radius, dtemp, drad, covTR):
    lum = 4 * np.pi * radius ** 2 * sigma_sb * temp ** 4
    dlum = 8 * np.pi * sigma_sb * (radius ** 2 * temp ** 8 * drad ** 2
                                   + 4 * radius ** 4 * temp ** 6 * dtemp ** 2
                                   + 4 * radius ** 3 * temp ** 7 * covTR) ** 0.5
    return lum, dlum


def median_and_unc(x, perc_contained=68.):
    q = 50. + np.array([-perc_contained / 2., 0., perc_contained / 2.])
    percentiles = np.percentile(x, q, axis=0)
    median = percentiles[1]
    lower, upper = np.diff(percentiles, axis=0)
    return median, lower, upper


def blackbody_lstsq(epoch1, z, p0=None, T_range=(1., 100.), R_range=(0.01, 1000.), cutoff_freq=np.inf):
    if p0 is None:
        p0 = [10., 10.]

    def planck_cutoff(nu, T, R):
        return models.planck_fast(nu, T, R, cutoff_freq)

    p0, cov = curve_fit(planck_cutoff, epoch1['freq'] * (1. + z), epoch1['lum'], p0=p0,
                        bounds=([T_range[0], R_range[0]], [T_range[1], R_range[1]]))
    temp, radius = p0
    dtemp, drad = np.sqrt(np.diag(cov))
    lum, dlum = stefan_boltzmann(temp, radius, dtemp, drad, cov[0, 1])
    L_opt = pseudo(temp, radius, z, cutoff_freq=cutoff_freq)
    return temp, radius, dtemp, drad, lum, dlum, L_opt


def integrate_sed(epoch1):
    epoch1_to_agg = epoch1[['filter', 'freq', 'dfreq', 'lum']]  # remove other columns to avoid warnings
    epoch1_to_agg = epoch1_to_agg.group_by('filter').groups.aggregate(np.mean)
    epoch1_to_agg.sort('freq')
    freqs = np.insert(epoch1_to_agg['freq'], 0, epoch1_to_agg['freq'][0] - epoch1_to_agg['dfreq'][0])
    lums = np.insert(epoch1_to_agg['lum'], 0, 0)
    freqs = np.append(freqs, epoch1_to_agg['freq'][-1] + epoch1_to_agg['dfreq'][-1])
    lums = np.append(lums, 0)
    L_int = np.trapz(lums * epoch1_to_agg['lum'].unit, freqs * epoch1_to_agg['freq'].unit).to(u.W)
    return L_int


def calculate_bolometric(lc, z, outpath='.', res=1., nwalkers=10, burnin_steps=200, steps=100,
                         T_range=(1., 100.), R_range=(0.01, 1000.), save_table_as=None, min_nfilt=3,
                         cutoff_freq=np.inf, show=False):

    t0 = LC(names=('MJD', 'dMJD0', 'dMJD1',
                   'temp', 'radius', 'dtemp', 'dradius',  # best fit from scipy.curve_fit
                   'lum', 'dlum',  # total bolometric luminosity from scipy.curve_fit
                   'L_opt',  # pseudobolometric luminosity from scipy.curve_fit
                   'temp_mcmc', 'radius_mcmc', 'dtemp0', 'dtemp1', 'dradius0', 'dradius1',  # best fit from MCMC
                   'L_mcmc', 'dL_mcmc0', 'dL_mcmc1',  # pseudobolometric luminosity from MCMC
                   'L_int',  # pseudobolometric luminosity from direct integration of the SED
                   'npoints', 'filts', 'source'),
            dtype=(float, float, float, float, float, float, float, float, float, float, float, float, float,
                   float, float, float, float, float, float, float, int, 'S6', lc['source'].dtype), masked=True)

    lc['freq'] = u.Quantity([f.freq_eff for f in lc['filter']])
    lc['dfreq'] = u.Quantity([f.dfreq for f in lc['filter']])

    lc['lum'].unit = u.W / u.Hz
    lc['dlum'].unit = u.W / u.Hz

    for epoch1 in group_by_epoch(lc, res):
        filts = set(epoch1.where(nondet=False)['filter'].data)
        nfilt = len(filts)
        if nfilt < min_nfilt:
            continue

        mjdavg, dmjd0, dmjd1 = median_and_unc(epoch1['MJD'], 100.)
        filtstr = ''.join([f.char for f in sorted(filts)])

        # blackbody - least squares
        p0 = [10., 10.]
        try:
            temp, radius, dtemp, drad, lum, dlum, L_opt = blackbody_lstsq(epoch1, z, p0, T_range, R_range, cutoff_freq)
            p0 = [temp, radius]
        except RuntimeError:  # optimization failed
            temp = radius = dtemp = drad = lum = dlum = L_opt = np.nan

        # blackbody - MCMC
        sampler = blackbody_mcmc(epoch1, z, p0, outpath=outpath, nwalkers=nwalkers, burnin_steps=burnin_steps,
                                 steps=steps, T_range=T_range, R_range=R_range, cutoff_freq=cutoff_freq, show=show)
        L_mcmc_opt = pseudo(sampler.flatchain[:, 0], sampler.flatchain[:, 1], z, cutoff_freq=cutoff_freq)
        (T_mcmc, R_mcmc), (dT0_mcmc, dR0_mcmc), (dT1_mcmc, dR1_mcmc) = median_and_unc(sampler.flatchain)
        L_mcmc, dL_mcmc0, dL_mcmc1 = median_and_unc(L_mcmc_opt)

        # direct integration
        L_int = integrate_sed(epoch1)

        t0.add_row([mjdavg, dmjd0, dmjd1,
                    temp, radius, dtemp, drad,
                    lum, dlum,
                    L_opt,
                    T_mcmc, R_mcmc, dT0_mcmc, dT1_mcmc, dR0_mcmc, dR1_mcmc,
                    L_mcmc, dL_mcmc0, dL_mcmc1,
                    L_int,
                    nfilt, filtstr, epoch1['source'][0]])

    if save_table_as is not None:
        t0.write(save_table_as, format='ascii.fixed_width', overwrite=True)

    return t0

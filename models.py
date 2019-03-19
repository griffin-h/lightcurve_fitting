import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.table import Table
import os

k_B = const.k_B.to("eV / kK").value
c3 = (4 * np.pi * const.sigma_sb.to("erg s-1 Rsun-2 kK-4").value) ** -0.5 / 1000.  # Rsun --> kiloRsun


def format_unit(unit):
    if isinstance(unit, u.Quantity):
        value = np.log10(unit.value)
        unit = unit.unit
        if value % 1.:
            unit_str = '$10^{{{value:.1f}}}$ {unit:latex_inline}'
        else:
            unit_str = '$10^{{{value:.0f}}}$ {unit:latex_inline}'
    else:
        value = None
        unit_str = '{unit:latex_inline}'
    return unit_str.format(value=value, unit=unit)


class Model:
    def __init__(self, func, input_names, units):
        self.func = func
        self.input_names = input_names
        self.units = units
        self.nparams = len(input_names)
        self.axis_labels = ['${}$ ({})'.format(var, format_unit(unit)) for var, unit in zip(input_names, units)]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def shock_cooling(t_in, f, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1., n=1.5, RW=False, z=0.):
    """time in days, velocity in 10**8.5 cm/s, masses in M_sun,
       progenitor radius in 10**13 cm, opacity in 0.34 cm**2/g,
       color temperature in kK, blackbody radius in 1000 R_sun"""

    if n == 1.5:
        A = 0.94
        a = 1.67
        alpha = 0.8
        epsilon_1 = 0.027
        epsilon_2 = 0.086
        L_0 = 2.0e42
        T_0 = 1.61
        Tph_to_Tcol = 1.1
    elif n == 3.:
        A = 0.79
        a = 4.57
        alpha = 0.73
        epsilon_1 = 0.016
        epsilon_2 = 0.175
        L_0 = 2.1e42
        T_0 = 1.69
        Tph_to_Tcol = 1.0
    else:
        raise ValueError('n can only be 1.5 or 3')

    if RW:
        a = 0.
        Tph_to_Tcol = 1.2

    t = t_in.reshape(-1, 1) - t_exp
    L_RW = L_0 * (t ** 2 * v_s / (f_rho_M * kappa)) ** -epsilon_2 * v_s ** 2 * R / kappa
    t_tr = 19.5 * (kappa * M_env / v_s) ** 0.5
    L = L_RW * A * np.exp(-(a * t / t_tr) ** alpha)
    T_ph = T_0 * (t ** 2 * v_s ** 2 / (f_rho_M * kappa)) ** epsilon_1 * kappa ** -0.25 * t ** -0.5 * R ** 0.25
    T_col = T_ph * Tph_to_Tcol
    T_K = np.squeeze(T_col) / k_B
    R_bb = c3 * np.squeeze(L) ** 0.5 * T_K ** -2

    y_fit = blackbody_to_filters(f, T_K, R_bb, z)

    return y_fit


def t_min(v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
    return 0.2 * R / v_s * np.minimum(0.5, R ** 0.4 * (f_rho_M * kappa) ** -0.2 * v_s ** -0.7) + t_exp


def t_max(v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
    return 7.4 * (R / kappa) ** 0.55 + t_exp


ShockCooling = Model(shock_cooling,
                     [
                         'v_\\mathrm{s*}',
                         'M_\\mathrm{env}',
                         'f_\\rho M',
                         'R',
                         't_0'
                     ],
                     [
                         10. ** 8.5 * u.cm / u.s,
                         u.Msun,
                         u.Msun,
                         1e13 * u.cm,
                         u.d
                     ]
                     )
ShockCooling.t_min = t_min
ShockCooling.t_max = t_max


def shock_cooling2(t_in, f, T_1, L_1, t_tr, t_exp=0., n=1.5, RW=False, z=0.):
    if n == 1.5:
        a = 1.67
        alpha = 0.8
        epsilon_1 = 0.027
        epsilon_2 = 0.086
    elif n == 3.:
        a = 4.57
        alpha = 0.73
        epsilon_1 = 0.016
        epsilon_2 = 0.175
    else:
        raise ValueError('n can only be 1.5 or 3')

    if RW:
        a = 0.

    t = t_in.reshape(-1, 1) - t_exp

    epsilon_T = 2 * epsilon_1 - 0.5
    epsilon_L = -2 * epsilon_2

    T_K = np.squeeze(T_1 * t ** epsilon_T)
    L = np.squeeze(L_1 * np.exp(-(a * t / t_tr) ** alpha) * t ** epsilon_L) * 1e42
    R_bb = c3 * L ** 0.5 * T_K ** -2

    y_fit = blackbody_to_filters(f, T_K, R_bb, z)

    return y_fit


def t_min2(*args):
    raise NotImplementedError('t_min cannot be translated to these parameters')


def t_max2(T_1, L_1=0., t_tr=0., t_exp=0., n=1.5):
    if n == 1.5:
        epsilon_1 = 0.027
    elif n == 3.:
        epsilon_1 = 0.016
    else:
        raise ValueError('n can only be 1.5 or 3')

    epsilon_T = 2 * epsilon_1 - 0.5

    return (8.12 / T_1) ** (epsilon_T ** -1) + t_exp


ShockCooling2 = Model(shock_cooling2,
                      [
                          'T_1',
                          'L_1',
                          't_\\mathrm{tr}',
                          't_0'
                      ],
                      [
                          u.kK,
                          1e42 * u.erg / u.s,
                          u.d,
                          u.d
                      ]
                      )
ShockCooling2.t_min = t_min2
ShockCooling2.t_max = t_max2

sifto_filename = os.path.join(os.path.dirname(__file__), 'models', 'sifto.dat')
sifto = Table.read(sifto_filename, format='ascii')


def scale_sifto(sn_lc):
    """Run this function before using the CompanionShocking model
       to scale the SiFTO model to match your supernova's luminosity
       and colors. The argument is your supernova's light curve."""
    for filt in set(sn_lc['filter']):
        if filt.char not in sifto.colnames:
            raise Exception('No SiFTO template for filter ' + filt.char)
        lc_filt = sn_lc.where(filter=filt)
        sifto[filt.char] *= np.max(lc_filt['lum']) / np.max(sifto[filt.char])


def companion_shocking(t_in, f, t_exp, a13, Mc_v9_7, t_peak, stretch, rr, ri, rU, kappa=1., z=0.):
    t_wrt_exp = t_in.reshape(-1, 1) - t_exp
    T_kasen = np.squeeze(25. * (a13 ** 36 * Mc_v9_7 * kappa ** -35 * t_wrt_exp ** -74) ** (1 / 144.))  # kK
    R_kasen = np.squeeze(2.7 * (kappa * Mc_v9_7 * t_wrt_exp ** 7) ** (1 / 9.))  # kiloRsun
    Lnu_kasen = blackbody_to_filters(f, T_kasen, R_kasen, z)

    t_wrt_peak = np.squeeze(t_in.reshape(-1, 1) - t_peak)
    if t_wrt_peak.ndim <= 1 and len(t_wrt_peak) == len(f):  # pointwise
        Lnu_sifto = np.array([np.interp(t, sifto['Epoch'] * stretch, sifto[filt.char])
                              for t, filt in zip(t_wrt_peak, f)])
    elif t_wrt_peak.ndim <= 1:
        Lnu_sifto = np.array([np.interp(t_wrt_peak, sifto['Epoch'] * stretch, sifto[filt.char])
                              for filt in f])
    else:
        Lnu_sifto = np.array([np.array([np.interp(t, sifto['Epoch'] * s, sifto[filt.char])
                                        for t, s in zip(t_wrt_peak.T, stretch)]).T
                              for filt in f])

    sifto_factors = {'r': rr, 'i': ri}
    kasen_factors = {'U': rU}
    y_fit = np.array([L1 * kasen_factors.get(filt.char, 1.) + L2 * sifto_factors.get(filt.char, 1.)
                      for L1, L2, filt in zip(Lnu_kasen, Lnu_sifto, f)])

    return y_fit


CompanionShocking = Model(companion_shocking,
                          [
                              't_0',
                              'a',
                              'M v^7',
                              't_\\mathrm{max}',
                              's',
                              'r_r',
                              'r_i',
                              'r_U'
                          ],
                          [
                              u.d,
                              10. ** 13. * u.cm,
                              1.4 * u.Msun * (1e9 * u.cm / u.s) ** 7,
                              u.d,
                              u.dimensionless_unscaled,
                              u.dimensionless_unscaled,
                              u.dimensionless_unscaled,
                              u.dimensionless_unscaled
                          ]
                          )


def log_flat_prior(p):
    return p ** -1


def flat_prior(p):
    return np.ones_like(p)


c1 = (const.h / const.k_B).to(u.kK / u.THz).value
c2 = 8 * np.pi ** 2 * (const.h / const.c ** 2).to(u.W / u.Hz / (1000 * u.Rsun) ** 2 / u.THz ** 3).value


def planck_fast(nu, T, R):
    return c2 * np.squeeze(
        np.outer(R ** 2, nu ** 3) / (np.exp(c1 * np.outer(T ** -1, nu)) - 1))  # shape = (len(T), len(nu))


def blackbody_to_filters(filtobj, T, R, z=0.):
    if T.shape != R.shape:
        raise Exception('T & R must have the same shape')
    if T.ndim == 1 and len(T) == len(filtobj):  # pointwise
        y_fit = np.array([np.trapz(planck_fast(f.trans['freq'].data * (1. + z), t, r) * f.trans['T_norm_per_freq'].data,
                                   f.trans['freq'].data) for t, r, f in zip(T, R, filtobj)])  # shape = (len(T),)
    elif T.ndim <= 1:
        y_fit = np.array([np.trapz(planck_fast(f.trans['freq'].data * (1. + z), T, R) * f.trans['T_norm_per_freq'].data,
                                   f.trans['freq'].data) for f in filtobj]).T  # shape = (len(T), len(filtobj))
    else:
        y_fit = np.array([np.trapz(planck_fast(f.trans['freq'].data * (1. + z), T.flatten(), R.flatten())
                                   * f.trans['T_norm_per_freq'].data,
                                   f.trans['freq'].data) for f in filtobj]).T  # shape = (T.size, len(filtobj))
        y_fit = np.rollaxis(y_fit.reshape(T.shape[0], T.shape[1], len(filtobj)), -1)
    return y_fit


def planck(nu, T, R, dT=0., dR=0., cov=0.):
    Lnu = planck_fast(nu, T, R)
    if not dT and not dR and not cov:
        return Lnu
    dlogLdT = c1 * nu * T ** -2 / (1 - np.exp(-c1 * nu / T))
    dlogLdR = 2. / R
    dLnu = Lnu * (dlogLdT ** 2 * dT ** 2 + dlogLdR ** 2 * dR ** 2 + 2. * dlogLdT * dlogLdR * cov) ** 0.5
    return Lnu, dLnu

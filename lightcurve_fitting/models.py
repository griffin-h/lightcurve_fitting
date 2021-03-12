import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.table import Table
from pkg_resources import resource_filename

k_B = const.k_B.to("eV / kK").value
c3 = (4 * np.pi * const.sigma_sb.to("erg s-1 Rsun-2 kK-4").value) ** -0.5 / 1000.  # Rsun --> kiloRsun


def format_unit(unit):
    """
    Use LaTeX to format a physical unit, which may consist of an order of magnitude times a base unit

    Parameters
    ----------
    unit : astropy.units.Unit, astropy.units.Quantity
        The unit to format

    Returns
    -------
    unit_str : str
        Formatted unit
    """
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
    """
    An analytical model, defined by a function and its parameters

    Parameters
    ----------
    func : function
        A function that defines the analytical model
    input_names : list
        A list of parameter names
    units : list
        A list of units (:class:`astropy.unit.Unit`) for each parameter

    Attributes
    ----------
    func : function
        The function that defines the analytical model
    input_names : list
        A list of the parameter names
    units : list
        A list of the units for each parameter
    nparams : int
        The number of parameters in the model
    axis_labels : list
        Axis labels for each paramter (including name and unit)
    """
    def __init__(self, func, input_names, units):
        self.func = func
        self.input_names = input_names
        self.units = units
        self.nparams = len(input_names)
        self.axis_labels = ['${}$ ({})'.format(var, format_unit(unit))
                            if unit is not u.dimensionless_unscaled else '${}$'.format(var)
                            for var, unit in zip(input_names, units)]

    def __call__(self, *args, **kwargs):
        return self.func(*args[:self.nparams+2], **kwargs)  # +2 for times and filters


def shock_cooling(t_in, f, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1., n=1.5, RW=False, z=0.):
    """
    The shock cooling model of Sapir & Waxman (https://doi.org/10.3847/1538-4357/aa64df).

    This version of the model is written in terms of physical parameters :math:`v_s, M_\\mathrm{env}, f_ρ M, R`:

    :math:`T(t) = \\frac{T_\\mathrm{col}}{T_\\mathrm{ph}} T_0 \\left(\\frac{v_s^2 t^2}{f_ρ M κ}\\right)^{ε_1}
    \\frac{R^{1/4}}{κ^{1/4}} t^{-1/2}` (Eq. 23)

    :math:`L(t) = A \\exp\\left[-\\left(\\frac{a t}{t_\\mathrm{tr}}\\right)^α\\right]
    L_0 \\left(\\frac{v_s t^2}{f_ρ M κ}\\right)^{-ε_2} \\frac{v_s^2 R}{κ}` (Eq. 18-19)

    :math:`t_\\mathrm{tr} = (19.5\\,\\mathrm{d}) \\left(\\frac{κ * M_\\mathrm{env}}{v_s} \\right)^{1/2}` (Eq. 20)

    Parameters
    ----------
    t_in : float, array-like
        Time in days
    f : lightcurve_fitting.filter.Filter, array-like
        Filters for which to calculate the model
    v_s : float, array-like
        The shock speed in :math:`10^{8.5}` cm/s
    M_env : float, array-like
        The envelope mass in solar masses
    f_rho_M : float, array-like
        The product :math:`f_ρ M`, where ":math:`f_ρ` is a numerical factor of order unity that depends on the inner
        envelope structure" and :math:`M` is the ejecta mass in solar masses
    R : float, array-like
        The progenitor radius in :math:`10^{13}` cm
    t_exp : float, array-like
        The explosion epoch
    kappa : float, array-like
        The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)
    n : float, array-like
        The polytropic index of the progenitor. Must be either 1.5 or 3.
    RW : bool, optional
        Reduce the model to the simpler form of Rabinak & Waxman (https://doi.org/10.1088/0004-637X/728/1/63)
    z : float, optional
        The redshift between blackbody source and the observed filters

    Returns
    -------
    y_fit : array-like
        The filtered model light curves
    """
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
    """
    The minimum validity time for the :func:`shock_cooling` model

    :math:`t_\\mathrm{min} = (0.2\\,\\mathrm{d}) \\frac{R}{v_s}
    \\max\\left[0.5, \\frac{R^{0.4}}{(f_ρ M κ)^{0.2} v_s^{-0.7}}\\right] + t_\\mathrm{exp}` (Eq. 17)
    """
    return 0.2 * R / v_s * np.maximum(0.5, R ** 0.4 * (f_rho_M * kappa) ** -0.2 * v_s ** -0.7) + t_exp


def t_max(v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
    """
    The maximum validity time for the :func:`shock_cooling` model

    :math:`t_\\mathrm{max} = (7.4\,\\mathrm{d}) \\left(\\frac{R}{κ}\\right)^{0.55} + t_\\mathrm{exp}` (Eq. 24)
    """
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
    """
    The shock cooling model of Sapir & Waxman (https://doi.org/10.3847/1538-4357/aa64df).

    This version of the model is written in terms of scaling parameters :math:`T_1, L_1, t_\\mathrm{tr}`:

    :math:`T(t) = T_1 t^{2 ε_1 - 0.5}` and
    :math:`L(t) = L_1 \\exp\\left[-\\left(\\frac{a t}{t_\\mathrm{tr}}\\right)^α\\right] t^{-2 ε_2}`

    Parameters
    ----------
    t_in : float, array-like
        Time in days
    f : lightcurve_fitting.filter.Filter, array-like
        Filters for which to calculate the model
    T_1 : float, array-like
        The blackbody temperature 1 day after explosion in kilokelvins
    L_1 : float, array-like
        The approximate blackbody luminosity 1 day after explosion in :math:`10^{42}` erg/s
    t_tr : float, array-like
        The time at which the envelope becomes transparent in rest-frame days
    t_exp : float, array-like
        The explosion epoch
    n : float, array-like
        The polytropic index of the progenitor. Must be either 1.5 or 3.
    RW : bool, optional
        Reduce the model to the simpler form of Rabinak & Waxman (https://doi.org/10.1088/0004-637X/728/1/63)
    z : float, optional
        The redshift between blackbody source and the observed filters

    Returns
    -------
    y_fit : array-like
        The filtered model light curves
    """
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
    """
    The maximum validity time for the :func:`shock_cooling2` model

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError('t_min cannot be translated to these parameters')


def t_max2(T_1, L_1=0., t_tr=0., t_exp=0., n=1.5):
    """
    The maximum validity time for the :func:`shock_cooling2` model

    :math:`t_\\mathrm{max} = \\left(\\frac{8.12\\,\\mathrm{kK}}{T_1}\\right)^{1/(2 ε_1 - 0.5)} + t_\\mathrm{exp}`
    """
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

sifto_filename = resource_filename('lightcurve_fitting', 'models/sifto.dat')
sifto = Table.read(sifto_filename, format='ascii')


def scale_sifto(sn_lc):
    """
    Scale the SiFTO model to match your supernova's luminosity and colors

    Parameters
    ----------
    sn_lc : lightcurve_fitting.lightcurve.LC
        Your supernova's light curve
    """
    for filt in set(sn_lc['filter']):
        if filt.char not in sifto.colnames:
            raise Exception('No SiFTO template for filter ' + filt.char)
        lc_filt = sn_lc.where(filter=filt)
        sifto[filt.char] *= np.max(lc_filt['lum']) / np.max(sifto[filt.char])


def companion_shocking(t_in, f, t_exp, a13, Mc_v9_7, t_peak, stretch, rr, ri, rU, kappa=1., z=0.):
    """
    The companion shocking model of Kasen (https://doi.org/10.1088/0004-637X/708/2/1025) plus the SiFTO SN Ia model

    As written by Hosseinzadeh et al. (https://doi.org/10.3847/2041-8213/aa8402), the shock component is defined by:

    :math:`R_\\mathrm{phot}(t) = (2700\\,R_\\odot) (M_c v_9^7)^{1/9} κ^{1/9} t^{7/9}` (Eq. 1)

    :math:`T_\\mathrm{eff}(t) = (25\\,\\mathrm{kK}) a_{13}^{1/4} (M_c v_9^7)^{1/144} κ^{-35/144} t^{37/72}` (Eq. 2)

    The SiFTO model (https://doi.org/10.1086/588518) is currently only available in the UBVgri filters.

    Parameters
    ----------
    t_in : float, array-like
        Time in days
    f : lightcurve_fitting.filter.Filter, array-like
        Filters for which to calculate the model
    t_exp : float, array-like
        The explosion epoch
    a13 : float, array-like
        The binary separation in :math:`10^{13}` cm
    Mc_v9_7 : float, array-like
        The product :math:`M_c v_9^7`, where :math:`M_c` is the ejecta mass in Chandrasekhar masses and :math:`v_9` is
        the ejecta velocity in units of :math:`10^9` cm/s
    t_peak : float, array-like
        The epoch of maximum light for the SiFTO model
    stretch : float, array-like
        The stretch for the SiFTO model
    rr : float, array-like
        A scale factor for the r-band SiFTO model
    ri : float, array-like
        A scale factor for the i-band SiFTO model
    rU : float, aray-like
        A scale factor for the U-band of the shock component
    kappa : float, array-like
        The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)
    z : float, optional
        The redshift between blackbody source and the observed filters

    Returns
    -------
    y_fit : array-like
        The filtered model light curves
    """
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


M_chandra = u.def_unit('M_chandra', 1.4 * u.Msun, format={'latex': 'M_\\mathrm{Ch}'})
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
                              M_chandra * (1e9 * u.cm / u.s) ** 7,
                              u.d,
                              u.dimensionless_unscaled,
                              u.dimensionless_unscaled,
                              u.dimensionless_unscaled,
                              u.dimensionless_unscaled
                          ]
                          )


def log_flat_prior(p):
    """
    A uniform prior in the logarithm of the parameter, i.e., :math:`\\frac{dP}{dp} \\propto \\frac{1}{p}`
    """
    return p ** -1


def flat_prior(p):
    """
    A uniform prior in the parameter, i.e., :math:`\\frac{dP}{dp} \\propto 1`
    """
    return np.ones_like(p)


c1 = (const.h / const.k_B).to(u.kK / u.THz).value
c2 = 8 * np.pi ** 2 * (const.h / const.c ** 2).to(u.W / u.Hz / (1000 * u.Rsun) ** 2 / u.THz ** 3).value


def planck_fast(nu, T, R, cutoff_freq=np.inf):
    """
    The Planck spectrum for a blackbody source

    :math:`L_ν = 4 π R^2 \\frac{2 π h ν^3 c^{-2}}{\\exp(h ν / k_B T) - 1}`

    Parameters
    ----------
    nu : float, array-like
        Frequency in terahertz
    T : float, array-like
        Temperature in kilokelvins
    R : float, array-like
        Radius in units of 1000 solar radii
    cutoff_freq : float, optional
        Cutoff frequency (in terahertz) for a modified blackbody spectrum (see https://doi.org/10.3847/1538-4357/aa9334)

    Returns
    -------
    float, array-like
        The spectral luminosity density (:math:`L_ν`) of the source in watts per hertz
    """
    return c2 * np.squeeze(np.outer(R ** 2, nu ** 3 * np.minimum(1., cutoff_freq / nu))
                           / (np.exp(c1 * np.outer(T ** -1, nu)) - 1))


def blackbody_to_filters(filters, T, R, z=0., cutoff_freq=np.inf):
    """
    The average spectral luminosity density (:math:`L_ν`) of a blackbody as observed through one or more filters

    Parameters
    ----------
    filters : lightcurve_fitting.filters.Filter, array-like
        One or more broadband filters
    T : float, array-like
        Blackbody temperatures in kilokelvins
    R : float, array-like
        Blackbody radii in units of 1000 solar radii
    z : float
        Redshift between the blackbody source and the observed filters
    cutoff_freq : float, optional
        Cutoff frequency (in terahertz) for a modified blackbody spectrum (see https://doi.org/10.3847/1538-4357/aa9334)

    Returns
    -------
    y_fit : float, array-like
        The average spectral luminosity density in each filter in watts per hertz
    """
    T = np.array(T)
    R = np.array(R)
    if T.shape != R.shape:
        raise Exception('T & R must have the same shape')
    if T.ndim == 1 and len(T) == len(filters):  # pointwise
        y_fit = np.array([f.blackbody(t, r, z, cutoff_freq) for f, t, r in zip(filters, T, R)])
    else:
        y_fit = np.array([f.blackbody(T.flatten(), R.flatten(), z, cutoff_freq) for f in filters])
        y_fit = y_fit.reshape(len(filters), *T.shape)
    return y_fit


def planck(nu, T, R, dT=0., dR=0., cov=0.):
    """
    The Planck spectrum for a blackbody source, including propagation of uncertainties

    Parameters
    ----------
    nu : float, array-like
        Frequency in terahertz
    T : float, array-like
        Temperature in kilokelvins
    R : float, array-like
        Radius in units of 1000 solar radii
    dT : float, array-like, optional
        Uncertainty in the temperature in kilokelvins
    dR : float, array-like, optional
        Uncertainty in the radius in units of 1000 solar radii
    cov : float, array-like, optional
        The covariance between the temperature and radius

    Returns
    -------
    Lnu : float, array-like
        The spectral luminosity density (:math:`L_ν`) of the source in watts per hertz
    dLnu : float, array-like
        The uncertainty in the spectral luminosity density in watts per hertz
    """
    Lnu = planck_fast(nu, T, R)
    if not dT and not dR and not cov:
        return Lnu
    dlogLdT = c1 * nu * T ** -2 / (1 - np.exp(-c1 * nu / T))
    dlogLdR = 2. / R
    dLnu = Lnu * (dlogLdT ** 2 * dT ** 2 + dlogLdR ** 2 * dR ** 2 + 2. * dlogLdT * dlogLdR * cov) ** 0.5
    return Lnu, dLnu

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.table import Table
from pkg_resources import resource_filename
from abc import ABCMeta, abstractmethod
from scipy.interpolate import CubicSpline
from .filters import filtdict

k_B = const.k_B.to("eV / kK").value
c3 = (4. * np.pi * const.sigma_sb.to("erg s-1 Rsun-2 kK-4").value) ** -0.5 / 1000.  # Rsun --> kiloRsun
c4 = 1. / (4. * np.pi * u.Mpc.to(u.m) ** 2.)


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


def power(base, exp):
    """Power function that returns zero for any nonpositive base"""
    broadcast = np.broadcast(base, exp)
    zeros = np.zeros(broadcast.shape, float)
    positive = base > 0.
    power = np.power(base, exp, out=zeros, where=positive)
    return power


class Model:
    """An analytical model, defined by a function and its parameters"""

    input_names = []
    """A list of the parameter names"""

    units = []
    """A list of the units for each parameter"""

    output_quantity = 'lum'
    """Quantity output by the model: 'lum' or 'flux'"""

    @property
    def nparams(self):
        """The number of parameters in the model"""
        return len(self.input_names)

    @property
    def axis_labels(self):
        """Axis labels for each paramter (including name and unit)"""
        return ['${}$ ({})'.format(var, format_unit(unit))
                if unit is not u.dimensionless_unscaled else '${}$'.format(var)
                for var, unit in zip(self.input_names, self.units)]

    def __init__(self, lc=None, redshift=0.):
        if redshift:
            self.z = redshift
        elif lc is not None and 'redshift' in lc.meta:
            self.z = lc.meta['redshift']
        else:
            self.z = 0.

    def __repr__(self):
        return f'<{self.__class__.__name__}: z={self.z:.3f}>'

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        return NotImplemented

    def log_likelihood(self, lc, p, use_sigma=False, sigma_type='relative'):
        """
        The log of the likelihood function of the model given data contained in `lc` and parameters contained in `p`

        Parameters
        ----------
        lc : lightcurve_fitting.lightcurve.LC
            Table of broadband photometry including columns "MJD", "mag", "dmag", "filter"
        p : array-like
            Model parameters for which to calculate the likelihood. The first dimension should have the same length as
            the number of parameters, or one more if sigma is used.
        use_sigma : bool, optional
            If True, treat the last parameter as an intrinsic scatter parameter that does not get passed to the model
        sigma_type : str, optional
            If 'relative' (default), sigma will be in units of the individual photometric uncertainties.
            If 'absolute', sigma will be in units of the median photometric uncertainty.

        Returns
        -------
        log_likelihood : float, array-like
            The natural log of the likelihood function. If `p` is 1D, `log_likelihood` is a float. Otherwise, its shape
            is determined by the remaining dimensions of `p`.
        """
        f = lc['filter'].data
        t = lc['MJD'].data
        y = lc[self.output_quantity].data
        dy = lc['d' + self.output_quantity].data

        if sigma_type == 'relative':
            sigma_units = dy
        elif sigma_type == 'absolute':
            sigma_units = np.median(dy)
        else:
            raise Exception('sigma_type must either be "relative" or "absolute"')

        if use_sigma:
            y_fit = self(t, f, *p[:-1])
            sigma = np.sqrt(dy ** 2. + (p[-1] * sigma_units) ** 2.)
        else:
            y_fit = self(t, f, *p)
            sigma = dy

        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2.) + ((y - y_fit) / sigma) ** 2.)
        return log_likelihood


class BaseShockCooling(Model):
    """
    The shock cooling model of Sapir & Waxman (https://doi.org/10.3847/1538-4357/aa64df).

    :math:`T(t) = \\frac{T_\\mathrm{col}}{T_\\mathrm{ph}} T_0 \\left(\\frac{v_s^2 t^2}{f_ρ M κ}\\right)^{ε_1}
    \\frac{R^{1/4}}{κ^{1/4}} t^{-1/2}` (Eq. 23)

    :math:`L(t) = A \\exp\\left[-\\left(\\frac{a t}{t_\\mathrm{tr}}\\right)^α\\right]
    L_0 \\left(\\frac{v_s t^2}{f_ρ M κ}\\right)^{-ε_2} \\frac{v_s^2 R}{κ}` (Eq. 18-19)

    :math:`t_\\mathrm{tr} = (19.5\\,\\mathrm{d}) \\left(\\frac{κ * M_\\mathrm{env}}{v_s} \\right)^{1/2}` (Eq. 20)

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC, optional
        The light curve to which the model will be fit. Only used to get the redshift if `redshift` is not given.
    redshift : float, optional
        The redshift between blackbody source and the observed filters. Default: 0.
    n : float, optional
        The polytropic index of the progenitor. Must be either 1.5 (default) or 3.
    RW : bool, optional
        Reduce the model to the simpler form of Rabinak & Waxman (https://doi.org/10.1088/0004-637X/728/1/63).
        Default: False.

    Attributes
    ----------
    z : float
        The redshift between blackbody source and the observed filters
    n : float
        The polytropic index of the progenitor
    A : float
        Coefficient on the luminosity suppression factor (Eq. 19)
    a : float
        Coefficient on the transparency timescale (Eq. 19)
    alpha : float
        Exponent on the transparency timescale (Eq. 19)
    epsilon_1 : float
        Exponent in the temperature expression (Eq. 23)
    epsilon_2 : float
        Exponent in the luminosity expression (Eq. 18)
    L_0 : float
        Coefficient on the luminosity expression in erg/s (Eq. 18)
    T_0 : float
        Coefficient on the temperature expression in eV (Eq. 23)
    Tph_to_Tcol : float
        Ratio of the color temperature to the photospheric temperature
    epsilon_T : float
        Exponent on time in the temperature expression
    epsilon_L : float
        Exponent on time in the luminosity expression
    RW : bool
        Reduce the model to the simpler form of Rabinak & Waxman (https://doi.org/10.1088/0004-637X/728/1/63)
    """
    def __init__(self, lc=None, redshift=0., n=1.5, RW=False):
        super().__init__(lc, redshift=redshift)

        if n == 1.5:
            self.n = 1.5
            self.A = 0.94
            self.a = 1.67
            self.alpha = 0.8
            self.epsilon_1 = 0.027
            self.epsilon_2 = 0.086
            self.L_0 = 2.0e42  # erg / s
            self.T_0 = 1.61  # eV
            self.Tph_to_Tcol = 1.1
        elif n == 3.:
            self.n = 3.
            self.A = 0.79
            self.a = 4.57
            self.alpha = 0.73
            self.epsilon_1 = 0.016
            self.epsilon_2 = 0.175
            self.L_0 = 2.1e42  # erg / s
            self.T_0 = 1.69  # eV
            self.Tph_to_Tcol = 1.0
        else:
            raise ValueError('n can only be 1.5 or 3')

        self.epsilon_T = 2 * self.epsilon_1 - 0.5
        self.epsilon_L = -2 * self.epsilon_2

        if RW:
            self.RW = True
            self.a = 0.
            self.Tph_to_Tcol = 1.2
        else:
            self.RW = False

    def __repr__(self):
        return f'<{self.__class__.__name__}: z={self.z:.3f}, n={self.n:.1f}, RW={self.RW}>'

    def temperature_radius(self, t_in, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
        """
        Evaluate the color temperature and photospheric radius as a function of time for a set of parameters

        Parameters
        ----------
        t_in : float, array-like
            Time in days
        v_s : float, array-like
            The shock speed in :math:`10^{8.5}` cm/s
        M_env : float, array-like
            The envelope mass in solar masses
        f_rho_M : float, array-like
            The product :math:`f_ρ M`, where :math:`f_ρ` is a numerical factor of order unity that depends on the inner
            envelope structure and :math:`M` is the ejecta mass in solar masses
        R : float, array-like
            The progenitor radius in :math:`10^{13}` cm
        t_exp : float, array-like
            The explosion epoch
        kappa : float, array-like
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)

        Returns
        -------
        T_K : array-like
            The model blackbody temperatures in units of kilokelvins
        R_bb : array-like
            The model blackbody radii in units of 1000 solar radii
        """
        t = np.reshape(t_in, (-1, 1)) - t_exp
        L_RW = self.L_0 * power(t ** 2 * v_s / (f_rho_M * kappa), -self.epsilon_2) * v_s ** 2 * R / kappa
        t_tr = 19.5 * (kappa * M_env / v_s) ** 0.5
        L = L_RW * self.A * np.exp(-power(self.a * t / t_tr, self.alpha))
        T_ph = self.T_0 * power(t ** 2 * v_s ** 2 / (f_rho_M * kappa), self.epsilon_1) \
               * kappa ** -0.25 * power(t, -0.5) * R ** 0.25
        T_col = T_ph * self.Tph_to_Tcol
        T_K = np.squeeze(T_col) / k_B
        R_bb = c3 * np.squeeze(L) ** 0.5 * power(T_K, -2.)
        return T_K, R_bb

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        return NotImplemented

    @staticmethod
    def t_min(p, kappa=1.):
        """
        The minimum time at which the model is valid

        :math:`t_\\mathrm{min} = (0.2\\,\\mathrm{d}) \\frac{R}{v_s}
        \\max\\left[0.5, \\frac{R^{0.4}}{(f_ρ M κ)^{0.2} v_s^{-0.7}}\\right] + t_\\mathrm{exp}` (Eq. 17)
        """
        v_s = p[0]
        f_rho_M = p[2]
        R = p[3]
        t_exp = p[4] if len(p) > 4 else 0.
        return 0.2 * R / v_s * np.maximum(0.5, R ** 0.4 * (f_rho_M * kappa) ** -0.2 * v_s ** -0.7) + t_exp

    @staticmethod
    def t_max(p, kappa=1.):
        """
        The maximum time at which the model is valid

        :math:`t_\\mathrm{max} = (7.4\\,\\mathrm{d}) \\left(\\frac{R}{κ}\\right)^{0.55} + t_\\mathrm{exp}` (Eq. 24)
        """
        R = p[3]
        t_exp = p[4] if len(p) > 4 else 0.
        return 7.4 * (R / kappa) ** 0.55 + t_exp


class ShockCooling(BaseShockCooling):
    """
    The shock cooling model of Sapir & Waxman (https://doi.org/10.3847/1538-4357/aa64df).

    This version of the model is written in terms of physical parameters :math:`v_s, M_\\mathrm{env}, f_ρ M, R`.
    """
    input_names = [
        'v_\\mathrm{s*}',
        'M_\\mathrm{env}',
        'f_\\rho M',
        'R',
        't_0'
    ]
    units = [
        10. ** 8.5 * u.cm / u.s,
        u.Msun,
        u.Msun,
        1e13 * u.cm,
        u.d
    ]

    def evaluate(self, t_in, f, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
        """
        Evaluate this model at a range of times and filters

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
        t_exp : float, array-like, optional
            The explosion epoch. Default: 0.
        kappa : float, array-like, optional
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g). Default: 1.

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        T_K, R_bb = self.temperature_radius(t_in, v_s, M_env, f_rho_M, R, t_exp, kappa)
        y_fit = blackbody_to_filters(f, T_K, R_bb, self.z)
        return y_fit


class ShockCooling2(BaseShockCooling):
    """
    The shock cooling model of Sapir & Waxman (https://doi.org/10.3847/1538-4357/aa64df).

    This version of the model is written in terms of scaling parameters :math:`T_1, L_1, t_\\mathrm{tr}`:

    :math:`T(t) = T_1 t^{ε_T}` and
    :math:`L(t) = L_1 t^{ε_L} \\exp\\left[-\\left(\\frac{a t}{t_\\mathrm{tr}}\\right)^α\\right]`
    """
    input_names = [
        'T_1',
        'L_1',
        't_\\mathrm{tr}',
        't_0'
    ]
    units = [
        u.kK,
        1e42 * u.erg / u.s,
        u.d,
        u.d
    ]

    def evaluate(self, t_in, f, T_1, L_1, t_tr, t_exp=0.):
        """
        Evaluate this model at a range of times and filters

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

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """

        t = np.reshape(t_in, (-1, 1)) - t_exp

        T_K = np.squeeze(T_1 * power(t, self.epsilon_T))
        L = np.squeeze(L_1 * np.exp(-power(self.a * t / t_tr, self.alpha)) * power(t, self.epsilon_L)) * 1e42
        R_bb = c3 * L ** 0.5 * power(T_K, -2.)

        y_fit = blackbody_to_filters(f, T_K, R_bb, self.z)

        return y_fit

    @staticmethod
    def t_min(p, kappa=1.):
        """
        The minimum time at which the model is valid

        This expression cannot be translated to the parameters of :class:`ShockCooling2`
        """
        return NotImplemented

    def t_max(self, p, kappa=1.):
        """
        The maximum time at which the model is valid

        :math:`t_\\mathrm{max} = \\left(\\frac{8.12\\,\\mathrm{kK}}{T_1}\\right)^{1/(2 ε_1 - 0.5)} + t_\\mathrm{exp}`
        """
        T_1 = p[0]
        t_exp = p[3] if len(p) > 3 else 0.
        return (8.12 / T_1) ** (self.epsilon_T ** -1) + t_exp


class ShockCooling3(BaseShockCooling):
    """
    The shock cooling model of Sapir & Waxman (https://doi.org/10.3847/1538-4357/aa64df).

    This version of the model is written in terms of physical parameters :math:`v_s, M_\\mathrm{env}, f_ρ M, R` and
    includes distance :math:`d_L` and reddening :math:`E(B-V)` as free parameters.
    """
    input_names = [
        'v_\\mathrm{s*}',
        'M_\\mathrm{env}',
        'f_\\rho M',
        'R',
        'd_L',
        'E(B-V)',
        't_0',
    ]
    units = [
        10. ** 8.5 * u.cm / u.s,
        u.Msun,
        u.Msun,
        1e13 * u.cm,
        u.Mpc,
        u.mag,
        u.d,
    ]
    output_quantity = 'flux'

    def evaluate(self, t_in, f, v_s, M_env, f_rho_M, R, dist, ebv=0., t_exp=0., kappa=1.):
        """
        Evaluate this model at a range of times and filters

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
        dist : float, array-like
            The luminosity distance in Mpc
        ebv : float, array-like, optional
            The reddening :math:`E(B-V)` to apply to the blackbody spectrum before integration. Default: 0.
        t_exp : float, array-like, optional
            The explosion epoch. Default: 0.
        kappa : float, array-like, optional
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g). Default: 1.

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        T_K, R_bb = self.temperature_radius(t_in, v_s, M_env, f_rho_M, R, t_exp, kappa)
        lum = blackbody_to_filters(f, T_K, R_bb, self.z, ebv=ebv)
        flux = c4 * lum / dist ** 2.
        return flux

    @staticmethod
    def t_min(p, kappa=1.):
        return super().t_min([p[0], p[1], p[2], p[3], p[6] if len(p) > 6 else 0.], kappa=kappa)

    @staticmethod
    def t_max(p, kappa=1.):
        return super().t_max([p[0], p[1], p[2], p[3], p[6] if len(p) > 6 else 0.], kappa=kappa)


class ShockCooling4(Model):
    """
    The shock cooling model of Morag, Sapir, & Waxman (https://doi.org/10.1093/mnras/stad899).

    :math:`L(\\tilde{t}) = L_\\mathrm{br}\\left\{\\tilde{t}^{-4/3} + 0.9\\exp\\left[-\\left(\\frac{2.0t}{t_\\mathrm{tr}}\\right)^{0.5}\\right] \\tilde{t}^{-0.17}\\right\}` (Eq. A1)

    :math:`T_\\mathrm{col}(\\tilde{t}) = T_\\mathrm{col,br} \\min(0.97\\tilde{t}^{-1/3}, \\tilde{t}^{-0.45})` (Eq. A2)

    :math:`\\tilde{t} \\equiv \\frac{t}{t_\\mathrm{br}}`, where :math:`t_\\mathrm{br} = (0.86\\,\\mathrm{h}) R^{1.26} v_\\mathrm{s*}^{-1.13} (f_\\rho M \\kappa)^{-0.13}` (Eq. A5)

    :math:`L_\\mathrm{br} = (3.69 \\times 10^{42}\\,\\mathrm{erg}\\,\\mathrm{s}^{-1}) R^{0.78} v_\\mathrm{s*}^{2.11} (f_\\rho M)^{0.11} \\kappa^{-0.89}` (Eq. A6)

    :math:`T_\\mathrm{col,br} = (8.19\\,\\mathrm{eV}) R^{-0.32} v_\\mathrm{s*}^{0.58} (f_\\rho M)^{0.03} \\kappa^{-0.22}` (Eq. A7)

    :math:`t_\\mathrm{tr} = (19.5\\,\\mathrm{d}) \\sqrt{\\frac{\\kappa M}{v_\\mathrm{s*}}}` (Eq. A9)

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC, optional
        The light curve to which the model will be fit. Only used to get the redshift if `redshift` is not given.
    redshift : float, optional
        The redshift between blackbody source and the observed filters. Default: 0.

    Attributes
    ----------
    z : float
        The redshift between blackbody source and the observed filters
    n : float
        The polytropic index of the progenitor
    A : float
        Coefficient on the luminosity suppression factor (Eq. A1)
    a : float
        Coefficient on the transparency timescale (Eq. A1)
    alpha : float
        Exponent on the transparency timescale (Eq. A1)
    L_br_0 : float
        Coefficient on the luminosity expression in erg/s (Eq. A6)
    T_col_br_0 : float
        Coefficient on the temperature expression in eV (Eq. A7)
    t_min_0 : float
        Coefficient on the minimum validity time in days (Eq. A3)
    t_br_0 : float
        Coefficient on the :math:`\\tilde{t}` timescale in days (Eq. A5)
    t_07eV_0 : float
        Coefficient on the time at which the ejecta reach 0.7 eV in days (Eq. A8)
    t_tr_0 : float
        Coefficient on the transparency timescale in days (Eq. A9)
    """
    input_names = [
        'v_\\mathrm{s*}',
        'M_\\mathrm{env}',
        'f_\\rho M',
        'R',
        't_0',
    ]
    units = [
        10. ** 8.5 * u.cm / u.s,
        u.Msun,
        u.Msun,
        1e13 * u.cm,
        u.d,
    ]

    def __init__(self, lc=None, redshift=0.):
        super().__init__(lc, redshift=redshift)

        self.A = 0.9
        self.a = 2.
        self.alpha = 0.5
        self.L_br_0 = 3.69e42  # erg / s
        self.T_col_br_0 = 8.19  # eV
        self.t_min_0 = 0.012  # d (17 min)
        self.t_br_0 = 0.036  # d (0.86 h)
        self.t_07eV_0 = 6.86  # d
        self.t_tr_0 = 19.5  # d

    def temperature_radius(self, t_in, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
        t_br = self.t_br_0 * R ** 1.26 * v_s ** -1.13 * f_rho_M ** -0.13  # Eq. A5
        L_br = self.L_br_0 * R ** 0.78 * v_s ** 2.11 * f_rho_M ** 0.11 * kappa ** -0.89  # Eq. A6
        T_col_br = self.T_col_br_0 * R ** -0.32 * v_s ** 0.58 ** f_rho_M ** 0.03 * kappa ** -0.22  # Eq. A7
        t_tr = self.t_tr_0 * np.sqrt(kappa * M_env / v_s)  # Eq. A9

        t = np.reshape(t_in, (-1, 1)) - t_exp
        ttilde = t / t_br
        L = L_br * (power(ttilde, -4. / 3.)
                    + self.A * np.exp(-power(self.a * t / t_tr, self.alpha)) * power(ttilde, -0.17))  # Eq. A1
        T_col = T_col_br * np.minimum(0.97 * power(ttilde, -1. / 3.), power(ttilde, -0.45))  # Eq. A2

        T_K = np.squeeze(T_col) / k_B
        R_bb = c3 * np.squeeze(L) ** 0.5 * power(T_K, -2.)
        return T_K, R_bb

    def evaluate(self, t_in, f, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
        """
        Evaluate this model at a range of times and filters

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
        t_exp : float, array-like, optional
            The explosion epoch. Default: 0.
        kappa : float, array-like, optional
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g). Default: 1.

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        T_K, R_bb = self.temperature_radius(t_in, v_s, M_env, f_rho_M, R, t_exp, kappa)
        lum_blackbody = blackbody_to_filters(f, T_K, R_bb, self.z)
        lum_suppressed = blackbody_to_filters(f, 0.74 * T_K, 0.74 ** -2. * R_bb, self.z)
        lum = np.minimum(lum_blackbody, lum_suppressed)  # Eq. A4
        return lum

    def t_min(self, p, kappa=1.):
        """
        The minimum time at which the model is valid

        :math:`t_\\mathrm{min} = (17\\,\\mathrm{min}) R + t_\\mathrm{exp}` (Eq. A3)
        """
        R = p[3]
        t_exp = p[4] if len(p) > 4 else 0.
        return self.t_min_0 * R + t_exp  # Eq. A3

    def t_max(self, p, kappa=1.):
        """
        The maximum time at which the model is valid

        :math:`t_\\mathrm{max} = \\min(t_\\mathrm{0.7\\,eV}, 0.5 t_\\mathrm{tr})` (Eq. A3)

        :math:`t_\\mathrm{0.7\\,eV} = (6.86\\,\\mathrm{d}) R^{0.56} v_\\mathrm{s*}^{0.16} \\kappa^{-0.61} (f_\\rho M)^{-0.06}` (Eq. A8)

        :math:`t_\\mathrm{tr} = (19.5\\,\\mathrm{d}) \\sqrt{\\frac{\\kappa M}{v_\\mathrm{s*}}}` (Eq. A9)
        """
        v_s, M_env, f_rho_M, R, t_exp, *_ = p
        t_07eV = self.t_07eV_0 * R ** 0.56 * v_s ** 0.16 * kappa ** -0.61 * f_rho_M ** -0.06  # Eq. A8
        t_tr = self.t_tr_0 ** np.sqrt(kappa * M_env / v_s)  # Eq. A9
        return np.minimum(t_07eV, t_tr / self.a) + t_exp  # Eq. A3


sifto_filename = resource_filename('lightcurve_fitting', 'models/sifto.dat')
sifto = Table.read(sifto_filename, format='ascii')[3:]  # the first three points are ~0
M_chandra = u.def_unit('M_chandra', 1.4 * u.Msun, format={'latex': 'M_\\mathrm{Ch}'})


class BaseCompanionShocking(Model):
    """
    The companion shocking model of Kasen (https://doi.org/10.1088/0004-637X/708/2/1025) plus the SiFTO SN Ia model.

    As written by Hosseinzadeh et al. (https://doi.org/10.3847/2041-8213/aa8402), the shock component is defined by:

    :math:`R_\\mathrm{phot}(t) = (2700\\,R_\\odot) (M_c v_9^7)^{1/9} κ^{1/9} t^{7/9}` (Eq. 1)

    :math:`T_\\mathrm{eff}(t) = (25\\,\\mathrm{kK}) a_{13}^{1/4} (M_c v_9^7)^{1/144} κ^{-35/144} t^{37/72}` (Eq. 2)

    The SiFTO model (https://doi.org/10.1086/588518) is currently only available in the UBVgri filters.

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC, optional
        The light curve to which the model will be fit. Only used for `lc.meta['redshift']` if `z` is not given.
    redshift : float, optional
        The redshift between blackbody source and the observed filters. Default: 0.

    Attributes
    ----------
    z : float
        The redshift between blackbody source and the observed filters
    sifto : astropy.table.Table
        A copy of the SiFTO model scaled to match the observed peak luminosity in each filter

    """
    def __init__(self, lc, redshift=0.):
        super().__init__(lc, redshift=redshift)

        # make sure input light curve has luminosities
        if 'lum' not in lc.colnames:
            if 'absmag' not in lc.colnames:
                lc.calcAbsMag()
            lc.calcLum()

        self.sifto = {}
        dlt40 = filtdict['DLT40']
        unfilt = filtdict['unfilt.']
        for filt in set(lc['filter']):
            if filt.char in sifto.colnames or filt == dlt40:
                char = 'r' if filt == dlt40 else filt.char
                lc_filt = lc.where(filter=filt)
                sifto_scaled = sifto[char] * np.max(lc_filt['lum']) / np.max(sifto[char])
                self.sifto[filt] = CubicSpline(sifto['Epoch'], sifto_scaled, extrapolate=False)
            elif filt != unfilt:
                raise Exception('No SiFTO template for filter ' + filt.name)

        # assume unfiltered = DLT40 for now
        self.sifto[unfilt] = self.sifto[dlt40]

    def __repr__(self):
        return f'<{self.__class__.__name__}: z={self.z:.3f}>'

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        return NotImplemented

    @staticmethod
    def temperature_radius(t_in, t_exp, a13, Mc_v9_7, kappa=1.):
        """
        Evaluate the color temperature and photospheric radius of the shock component as a function of time

        Parameters
        ----------
        t_in : float, array-like
            Time in days
        t_exp : float, array-like
            The explosion epoch
        a13 : float, array-like
            The binary separation in :math:`10^{13}` cm
        Mc_v9_7 : float, array-like
            The product :math:`M_c v_9^7`, where :math:`M_c` is the ejecta mass in Chandrasekhar masses and :math:`v_9`
            is the ejecta velocity in units of :math:`10^9` cm/s
        kappa : float, array-like
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)

        Returns
        -------
        T_kasen : array-like
            The model blackbody temperatures in units of kilokelvins
        R_kasen : array-like
            The model blackbody radii in units of 1000 solar radii
        """
        t = np.reshape(t_in, (-1, 1)) - t_exp
        T_kasen = np.squeeze(25. * power(a13 ** 36. * Mc_v9_7 * kappa ** -35. * power(t, -74.), 1. / 144.))  # kK
        R_kasen = np.squeeze(2.7 * power(kappa * Mc_v9_7 * t ** 7., 1. / 9.))  # kiloRsun
        return T_kasen, R_kasen

    def companion_shocking(self, t_in, f, t_exp, a13, Mc_v9_7, kappa=1.):
        """
        Evaluate the shock component only at a range of times and filters

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
        kappa : float, array-like
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        T_kasen, R_kasen = self.temperature_radius(t_in, t_exp, a13, Mc_v9_7, kappa)
        Lnu_kasen = blackbody_to_filters(f, T_kasen, R_kasen, self.z)
        return Lnu_kasen

    def stretched_sifto(self, t_in, f, t_peak, stretch, dtU=None, dti=None):
        """
        The SiFTO SN Ia model (https://doi.org/10.1086/588518), offset and stretched by the input parameters

        The SiFTO model is currently only available in the UBVgri filters. We assume unfiltered photometry can be
        modeled as r.

        Parameters
        ----------
        t_in : float, array-like
            Time in days
        f : lightcurve_fitting.filter.Filter, array-like
            Filters for which to calculate the model
        t_peak : float, array-like
            The epoch of maximum light for the SiFTO model
        stretch : float, array-like
            The stretch for the SiFTO model
        dtU, dti : float, array-like
            Time offsets for the U- and i-band models relative to the other bands

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        dt_peak = {}
        if dtU is not None:
            dt_peak[filtdict['U']] = dtU
        if dti is not None:
            dt_peak[filtdict['i']] = dti
        t_wrt_peak = np.squeeze(np.reshape(t_in, (-1, 1)) - t_peak)
        if t_wrt_peak.ndim <= 1 and len(t_wrt_peak) == len(f):  # pointwise
            Lnu_sifto = np.array([self.sifto[filt]((t - dt_peak.get(filt, 0.)) / stretch)
                                  for t, filt in zip(t_wrt_peak, f)])
        elif t_wrt_peak.ndim <= 1:
            Lnu_sifto = np.array([self.sifto[filt]((t_wrt_peak - dt_peak.get(filt, 0.)) / stretch) for filt in f])
        else:
            Lnu_sifto = np.array([np.transpose([self.sifto[filt]((t - dt) / s) for t, dt, s in
                                                zip(t_wrt_peak.T, dt_peak.get(filt, np.zeros_like(stretch)), stretch)])
                                  for filt in f])
        Lnu_sifto[np.isnan(Lnu_sifto)] = 0.  # extrapolate all filters as zero
        return Lnu_sifto

    @staticmethod
    def t_min(p):
        """
        The minimum time at which the model is valid

        This is the first epoch at which the stretched SiFTO model is computed
        """
        return p[3] + p[4] * sifto['Epoch'].min()

    @staticmethod
    def t_max(p):
        """
        The maximum time at which the model is valid

        This is the last epoch at which the stretched SiFTO model is computed
        """
        return p[3] + p[4] * sifto['Epoch'].max()


class CompanionShocking(BaseCompanionShocking):
    """
    The companion shocking model of Kasen (https://doi.org/10.1088/0004-637X/708/2/1025) plus the SiFTO SN Ia model.

    This version of the model includes factors on the r and i SiFTO models, and a factor on the U shock component.
    """
    input_names = [
        't_0',
        'a',
        'M v^7',
        't_\\mathrm{max}',
        's',
        'r_r',
        'r_i',
        'r_U'
    ]
    units = [
        u.d,
        10. ** 13. * u.cm,
        M_chandra * (1e9 * u.cm / u.s) ** 7,
        u.d,
        u.dimensionless_unscaled,
        u.dimensionless_unscaled,
        u.dimensionless_unscaled,
        u.dimensionless_unscaled
    ]

    def evaluate(self, t_in, f, t_exp, a13, Mc_v9_7, t_peak, stretch, rr=1., ri=1., rU=1., kappa=1.):
        """
        Evaluate this model at a range of times and filters

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

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        Lnu_kasen = self.companion_shocking(t_in, f, t_exp, a13, Mc_v9_7, kappa)
        Lnu_sifto = self.stretched_sifto(t_in, f, t_peak, stretch)

        sifto_factors = {'r': rr, 'i': ri}
        kasen_factors = {'U': rU}
        y_fit = np.array([L1 * kasen_factors.get(filt.char, 1.) + L2 * sifto_factors.get(filt.char, 1.)
                          for L1, L2, filt in zip(Lnu_kasen, Lnu_sifto, f)])

        return y_fit


class CompanionShocking2(BaseCompanionShocking):
    """
    The companion shocking model of Kasen (https://doi.org/10.1088/0004-637X/708/2/1025) plus the SiFTO SN Ia model.

    This version of the model includes time offsets for the U and i SiFTO models.
    """
    input_names = [
        't_0',
        'a',
        'M v^7',
        't_\\mathrm{max}',
        's',
        '\\Delta t_U',
        '\\Delta t_i',
    ]
    units = [
        u.d,
        10. ** 13. * u.cm,
        M_chandra * (1e9 * u.cm / u.s) ** 7,
        u.d,
        u.dimensionless_unscaled,
        u.d,
        u.d,
    ]

    def evaluate(self, t_in, f, t_exp, a13, Mc_v9_7, t_peak, stretch, dtU=0., dti=0., kappa=1.):
        """
        Evaluate this model at a range of times and filters

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
        dtU, dti : float, array-like
            Time offsets for the U- and i-band SiFTO models relative to the other bands
        kappa : float, array-like
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        Lnu_kasen = self.companion_shocking(t_in, f, t_exp, a13, Mc_v9_7, kappa)
        Lnu_sifto = self.stretched_sifto(t_in, f, t_peak, stretch, dtU, dti)
        y_fit = Lnu_kasen + Lnu_sifto
        return y_fit


class CompanionShocking3(BaseCompanionShocking):
    """
    The companion shocking model of Kasen (https://doi.org/10.1088/0004-637X/708/2/1025) plus the SiFTO SN Ia model.

    This version of the model includes time offsets for the U and i SiFTO models, as well as the viewing angle
    dependence parametrized by Brown et al. (https://doi.org/10.1088/0004-637X/749/1/18).
    """
    input_names = [
        't_0',
        'a',
        '\\theta',
        't_\\mathrm{max}',
        's',
        '\\Delta t_U',
        '\\Delta t_i',
    ]
    units = [
        u.d,
        10. ** 13. * u.cm,
        u.deg,
        u.d,
        u.dimensionless_unscaled,
        u.d,
        u.d,
    ]

    def evaluate(self, t_in, f, t_exp, a13, theta, t_peak, stretch, dtU, dti, kappa=1.):
        """
        Evaluate this model at a range of times and filters

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
        theta : float, array-like
            The angle between the binary axis and the line to the observer in degrees. 0° means the binary companion
            is along the line of sight.
        t_peak : float, array-like
            The epoch of maximum light for the SiFTO model
        stretch : float, array-like
            The stretch for the SiFTO model
        dtU, dti : float, array-like
            Time offsets for the U- and i-band SiFTO models relative to the other bands
        kappa : float, array-like
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g)

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        Lnu_kasen = self.companion_shocking(t_in, f, t_exp, a13, 1., kappa)
        Lnu_sifto = self.stretched_sifto(t_in, f, t_peak, stretch, dtU, dti)
        theta_rad = np.deg2rad(theta)
        fractional_flux = (0.5 * np.cos(theta_rad) + 0.5) * (0.14 * theta_rad ** 2. - 0.4 * theta_rad + 1.)
        y_fit = Lnu_kasen * fractional_flux + Lnu_sifto
        return y_fit


class Prior:
    __metaclass__ = ABCMeta

    def __init__(self, p_min=-np.inf, p_max=np.inf):
        self.p_min = p_min
        self.p_max = p_max

    def __call__(self, p):
        if self.p_min < p < self.p_max:
            return self.logp(p)
        else:
            return -np.inf

    @abstractmethod
    def logp(self, p):
        pass


class UniformPrior(Prior):
    """
    A uniform prior in the parameter, i.e., :math:`\\frac{dP}{dp} \\propto 1`
    """
    def logp(self, p):
        return np.zeros_like(p)


class LogUniformPrior(Prior):
    """
    A uniform prior in the logarithm of the parameter, i.e., :math:`\\frac{dP}{dp} \\propto \\frac{1}{p}`
    """
    def __init__(self, p_min=0., p_max=np.inf):
        if p_min < 0.:
            raise ValueError('a log-uniform prior cannot have negative limits')
        super().__init__(p_min, p_max)

    def logp(self, p):
        return -np.log(p)


class GaussianPrior(Prior):
    """
    A Gaussian prior centered at `mean` with standard deviation `stddev`, i.e.,
    :math:`\\frac{dP}{dp} \\propto \\exp \\left( \\frac{(p - \\mu)^2}{2 \\sigma^2} \\right)`
    """
    def __init__(self, p_min=-np.inf, p_max=np.inf, mean=0., stddev=1.):
        super().__init__(p_min, p_max)
        self.mean = mean
        self.stddev = stddev

    def logp(self, p):
        return -0.5 * ((p - self.mean) / self.stddev) ** 2.


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
    return c2 * np.squeeze(np.multiply.outer(R ** 2, nu ** 3 * np.minimum(1., cutoff_freq / nu))
                           * power(np.exp(c1 * np.multiply.outer(power(T, -1.), nu)) - 1., -1.))


def blackbody_to_filters(filters, T, R, z=0., cutoff_freq=np.inf, ebv=0.):
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
    ebv : float, array-like, optional
        Selective extinction E(B-V) in magnitudes, evaluated using a Fitzpatrick (1999) extinction law with R_V=3.1.
        Its shape must be broadcastable to T and R. Default: 0.

    Returns
    -------
    y_fit : float, array-like
        The average spectral luminosity density in each filter in watts per hertz
    """
    T = np.array(T)
    R = np.array(R)
    if T.shape != R.shape:
        raise Exception('T & R must have the same shape')
    np.broadcast(T, ebv)  # check if T and ebv are broadcastable, otherwise raise a ValueError
    if T.ndim == 1 and len(T) == len(filters):  # pointwise
        y_fit = np.array([f.synthesize(planck_fast, t, r, cutoff_freq, z=z, ebv=ebv) for f, t, r in zip(filters, T, R)])
    else:
        y_fit = np.array([f.synthesize(planck_fast, T, R, cutoff_freq, z=z, ebv=ebv) for f in filters])
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

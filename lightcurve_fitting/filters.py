import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
import os
from pkg_resources import resource_filename
from functools import total_ordering
from extinction import fitzpatrick99

c = const.c.to(u.angstrom * u.THz).value


def extinction_law(freq, ebv, rv=3.1):
    """
    A vectorized version of the Fitzpatrick (1999) extinction law from the ``extinction`` package

    Parameters
    ----------
    freq : array-like
        Frequencies in THz in the frame of the dust
    ebv : array-like
        Selective extinction :math:`E(B-V)` in magnitudes
    rv : float, optional
        Ratio of total to selective extinction :math:`R_V`. Default: 3.1.

    Returns
    -------
    extinction : array-like
        Extinction factor :math:`10^{A/-2.5}` at each input wavelength.
    """
    A = np.squeeze([fitzpatrick99(c / freq, rv * e) for e in np.atleast_1d(ebv)])
    return 10. ** (A / -2.5)


@total_ordering
class Filter:
    """
    A broadband photometric filter described by its transmission function and its associated photometric system

    Parameters
    ----------
    names : str, list
        One or more names for the filter. The first name is used by default but other names are recognized.
    color : str, tuple, optional
        The color used when plotting photometry in this filter. Default: black.
    offset : float, optional
        When plotting, offset photometry in this filter by this amount (in magnitudes if plotting magnitudes)
    system : str, optional
        Photometric system. If one of ``['Gunn', 'ATLAS', 'Gaia', 'MOSFiT']``, magnitudes are assumed to be AB.
        If ``'Johnson'``, plot markers with a black edge. Otherwise, ignored except for grouping filters in legends.
    fnu : float, optional
        Zero-point flux for magnitudes in this filter, in W/(m^2 Hz). If ``fnu = None`` and ``system in ['Gunn',
        'ATLAS', 'Gaia', 'MOSFiT']``, assume the AB zero point. Otherwise if ``fnu = None``, converting to flux will raise
        an error.
    filename : str, optional
        Filename containing the transmission function. If none is supplied, converting to flux will raise an error.
    angstrom : bool, optional
        If False (default), the transmission function wavelengths are assumed to be in nanometers. If True, they are
        given in ångströms.
    linecolor : str, tuple, optional
        The line color used when plotting photometry in this filter. Default: same as ``color``.
    textcolor : str, tuple, optional
        The color used when printing the name of this filter. Default: same as ``linecolor``.
    italics : bool, optional
        Italicize the filter name when used with LaTeX. Default: True.

    Attributes
    ----------
    name : str
        The default name of the filter
    names : list
        A list of aliases for the filter
    char : str
        A single-character identifier for the filter
    color : str, tuple
        The color used when plotting photometry in this filter
    linecolor : str, tuple
        The line color used when plotting photometry in this filter
    textcolor : str, tuple
        The color used when printing the name of this filter
    italics : bool
        Italicize the filter name when used with LaTeX
    mec : str, tuple
        The marker edge color used when plotting photometry in this filter
    plotstyle : dict
        Contains all keyword arguments for plotting photometry in this filter
    offset : float
        When plotting, offset photometry in this filter by this amount (in magnitudes if plotting magnitudes)
    system : str
        The photometric system for magnitudes in this filter
    fnu : float
        The zero-point flux for magnitudes in this filter, in watts per square meter per hertz
    m0 : float
        The zero point for magnitudes in this filter, i.e., :math:`m_0 = 2.5 \\log_{10}(F_ν)`
    M0 : float
        The zero point for absolute magnitudes in this filter, i.e., :math:`M_0 = m_0 + 90.19`
    filename : str
        Filename containing the transmission function
    angstrom : bool
        If False (default), the transmission function wavelengths are assumed to be in nanometers. If True, they are
        given in ångströms.
    trans : astropy.table.Table
        The transmission function
    freq_eff : float
        The effective frequency of the filter (in terahertz), i.e., :math:`ν_0 = \\frac{1}{Δν} \\int ν T(ν) dν`
    dfreq : float
        The effective bandwidth of the filter (in terahertz), i.e., :math:`Δν = \\int T(ν) dν`
    freq_range : tuple
        The approximate edges of the filter transmission function (in terahertz), i.e., :math:`ν_0 \\pm Δν`
    """

    order = None
    """The names of recognized filters listed in (approximate) decreasing order of effective frequency"""

    def __init__(self, names, color='k', offset=0, system=None, fnu=3.631e-23, filename='', angstrom=False,
                 linecolor=None, textcolor=None, italics=True):
        if type(names) == list:
            self.name = names[0]
            self.names = names
        else:
            self.name = names
            self.names = [names]
        if len(self.name) == 1:
            self.char = self.name
        else:
            shortest = sorted(self.names, key=len)[0]
            if len(shortest) == 1:
                self.char = shortest
            else:
                self.char = 'x'
        self.color = color
        if linecolor:
            self.linecolor = linecolor
        else:
            self.linecolor = self.color
        if textcolor:
            self.textcolor = textcolor
        else:
            self.textcolor = self.linecolor
        self.italics = italics
        self.system = system
        if self.system in ['Johnson', 'JWST MIRI']:
            self.mec = 'k'
        else:
            self.mec = self.linecolor
        self.offset = offset
        self.system = system
        self.plotstyle = {'color': self.linecolor, 'mfc': self.color, 'mec': self.mec}
        self.fnu = fnu  # * u.W * u.m**-2 * u.Hz**-1
        if self.fnu is None:
            self.m0 = np.nan
            self.M0 = np.nan
        else:
            self.m0 = 2.5 * np.log10(self.fnu)
            self.M0 = self.m0 + 90.19
        if filename:
            self.filename = resource_filename('lightcurve_fitting', os.path.join('filters', filename))
        else:
            self.filename = ''
        self.angstrom = angstrom
        self.trans = None
        self.freq_eff = None
        self.dfreq = None
        self.freq_range = None
        self.wl_eff = None
        self.dwl = None
        self.wl_range = None

    def read_curve(self, show=False, force=False):
        """
        Read the transmission function from ``self.filename`` and store it in ``self.trans``

        Parameters
        ----------
        show : bool, optional
            If True, also plot the transmission function
        force : bool, optional
            If True, reread the transmission function from ``self.filename`` even if is already stored in ``self.trans``
        """
        if (self.trans is None or force) and self.filename:
            i = Filter.order.index(self.name) / float(len(Filter.order))
            self.trans = Table.read(self.filename, format='ascii', names=('wl', 'T'))
            if self.angstrom:
                self.trans['wl'] = self.trans['wl'] / 10.
            self.trans['wl'].unit = u.nm
            self.trans.sort('wl')
            self.trans['T'] /= np.max(self.trans['T'])
            self.trans['freq'] = (const.c / self.trans['wl']).to(u.THz)

            dwl = np.trapz(self.trans['T'].quantity, self.trans['wl'].quantity)
            wl_eff = np.trapz(self.trans['T'].quantity * self.trans['wl'].quantity, self.trans['wl'].quantity) / dwl
            left = self.trans[(self.trans['wl'] < wl_eff.value) & (self.trans['T'] >= 0.1) * (self.trans['T'] <= 0.9)]
            left.sort('T')
            wl0 = np.interp(0.5, left['T'], left['wl'])
            right = self.trans[(self.trans['wl'] > wl_eff.value) & (self.trans['T'] >= 0.1) * (self.trans['T'] <= 0.9)]
            right.sort('T')
            wl1 = np.interp(0.5, right['T'], right['wl'])
            if show:
                plt.figure(1)
                ax1 = plt.gca()
                ax1.plot(self.trans['wl'], self.trans['T'], self.color if self.color != 'w' else 'k',
                         label=self.system + ' ' + self.name)
                ax1.errorbar(wl_eff.value, i, xerr=[[wl_eff.value - wl0], [wl1 - wl_eff.value]], marker='o',
                             **self.plotstyle)
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('Transmission')

            dfreq = np.trapz(self.trans['T'].quantity, self.trans['freq'].quantity)
            freq_eff = np.trapz(self.trans['T'].quantity * self.trans['freq'].quantity,
                                self.trans['freq'].quantity) / dfreq
            freq0 = np.interp(0.5, right['T'], right['freq'])
            freq1 = np.interp(0.5, left['T'], left['freq'])
            T_per_freq = self.trans['T'].quantity / self.trans['freq'].quantity
            self.trans['T_norm_per_freq'] = (T_per_freq / np.trapz(T_per_freq, self.trans['freq'].quantity))
            if show:
                plt.figure(2)
                ax2 = plt.gca()
                ax2.plot(self.trans['freq'], self.trans['T'], self.color if self.color != 'w' else 'k',
                         label=self.system + ' ' + self.name)
                ax2.errorbar(freq_eff.value, i, xerr=[[freq_eff.value - freq0], [freq1 - freq_eff.value]], marker='o',
                             **self.plotstyle)
                ax2.set_xlabel('Frequency (THz)')
                ax2.set_ylabel('Transmission')

            self.wl_eff = wl_eff
            self.dwl = dwl
            self.wl_range = (wl_eff.value - wl0, wl1 - wl_eff.value)
            self.freq_eff = freq_eff
            self.dfreq = -dfreq
            self.freq_range = (freq_eff.value - freq0, freq1 - freq_eff.value)

    def synthesize(self, spectrum, *args, z=0., ebv=0., **kwargs):
        """
        Returns the average Lnu of the given spectrum in this filter

        Parameters
        ----------
        spectrum : function
            Function describing the spectrum. The first argument must be frequency in THz, and it must return spectral
            luminosity in watts per hertz. All arguments are passed to this function except for keywords z and ebv.
        z : float, optional
            Redshift between the emission source and the observed filter. Default: 0.
        ebv : float, array-like, optional
            Selective extinction :math:`E(B-V)` in magnitudes, evaluated using a Fitzpatrick (1999) extinction law with
            :math:`R_V=3.1`. Its shape must be broadcastable to any array-like arguments. Default: 0.

        Returns
        -------
        Lnu : float or array-like
            Average spectral luminosity in the filter in watts per hertz
        """
        freq = self.trans['freq'].value * (1. + z)
        return np.trapz(spectrum(freq, *args, **kwargs) * extinction_law(freq, ebv)
                        * self.trans['T_norm_per_freq'].data, self.trans['freq'].data)

    def spectrum(self, freq, lum, z=0., ebv=0.):
        """
        Return the average value of the given spectrum in this filter

        The limits of integration come from the input spectrum, so if the spectrum does not cover the full filter, this
        function returns the average Lnu within the overlapping region, i.e., it does not extrapolate.

        Parameters
        ----------
        freq : float or array-like
            Frequency of the input spectrum in THz
        lum : float or array-like
            Spectral luminosity (Lnu) or flux (Fnu) of the input spectrum
        z : float, optional
            Redshift between the spectrum and the filter. Default: 0.
        ebv : float, array-like, optional
            Selective extinction :math:`E(B-V)` in magnitudes, evaluated using a Fitzpatrick (1999) extinction law with
            :math:`R_V=3.1`. Default: 0.

        Returns
        -------
        Lnu : float or array-like
            Average spectral luminosity (Lnu) or flux (Fnu) in the filter
        """
        freq *= (1. + z)
        T_per_freq = self.trans['T'].value / self.trans['freq'].value
        T_interp = np.interp(freq, self.trans['freq'][::-1].value, T_per_freq[::-1], left=0., right=0.)
        T_norm_per_freq = T_interp / np.trapz(T_interp, freq)
        return np.trapz(lum * extinction_law(freq, ebv) * T_norm_per_freq, freq)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<filter ' + self.name + '>'

    def __eq__(self, other):
        return isinstance(other, Filter) and self.name == other.name

    def __lt__(self, other):
        return isinstance(other, Filter) and Filter.order.index(self.name) < Filter.order.index(other.name)

    def __hash__(self):
        return self.name.__hash__()


def _resample_filter_curve(filename, outfile):
    orig = np.loadtxt(filename)
    wl = np.arange(1225., 274., -1.)
    resampled = np.interp(wl, orig[:, 0], orig[:, 1], left=0, right=0)
    output = np.array([wl, resampled]).T
    np.savetxt(outfile, output, fmt=['%.0f', '%.16f'])


# Vega zero points are from
#  * Table A2 of Bessell et al. 1998, A&A, 333, 231 for UBVRIJHK
#  * Table 1 of https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/swift/docs/uvot/uvot_caldb_AB_10wa.pdf for Swift
all_filters = [
    Filter('FUV', 'b', 8, 'GALEX', filename='GALEX_GALEX.FUV.dat', angstrom=True),
    Filter('NUV', 'r', 8, 'GALEX', filename='GALEX_GALEX.NUV.dat', angstrom=True),
    Filter(['UVW2', 'uvw2', 'W2', '2', 'uw2'], '#FF007F', 8, 'Swift', 7.379e-24, 'Swift_UVOT.UVW2.dat', angstrom=True),
    Filter(['UVM2', 'uvm2', 'M2', 'M', 'um2'], 'm', 8, 'Swift', 7.656e-24, 'Swift_UVOT.UVM2.dat', angstrom=True),
    Filter(['UVW1', 'uvw1', 'W1', '1', 'uw1'], '#7F00FF', 4, 'Swift', 9.036e-24, 'Swift_UVOT.UVW1.dat', angstrom=True),
    Filter(['u', "u'", 'up'], '#4700CC', 3, 'Gunn', filename='SLOAN_SDSS.u.dat', angstrom=True),  # brightened from '#080017'
    Filter(['U_S', 's', 'us'], '#230047', 3, 'Swift', 1.419e-23, filename='Swift_UVOT.U.dat', angstrom=True),
    Filter('U', '#3C0072', 3, 'Johnson', 1.790e-23, filename='Generic_Johnson.U.dat', angstrom=True),
    Filter('B', '#0057FF', 2, 'Johnson', 4.063e-23, filename='Generic_Johnson.B.dat', angstrom=True),
    Filter(['B_S', 'b', 'bs'], '#4B00FF', 2, 'Swift', 4.093e-23, filename='Swift_UVOT.B.dat', angstrom=True),
    Filter(['g', "g'", 'gp', 'F475W'], '#00CCFF', 1, 'Gunn', filename='SLOAN_SDSS.g.dat', angstrom=True),
    Filter('g-DECam', '#00CCFF', 1, 'DECam', filename='CTIO_DECam.g.dat', angstrom=True),
    Filter(['c', 'cyan'], 'c', 1, 'ATLAS', filename='ATLAS_cyan.txt'),
    Filter('V', '#79FF00', 1, 'Johnson', 3.636e-23, filename='Generic_Johnson.V.dat', angstrom=True, textcolor='#46CC00'),
    Filter(['V_S', 'v', 'vs'], '#00FF30', 1, 'Swift', 3.664e-23, filename='Swift_UVOT.V.dat', angstrom=True),
    Filter('Itagaki', 'w', 0, 'Itagaki', filename='KAF-1001E.asci', linecolor='k', italics=False),
    Filter('white', 'w', 0, 'MOSFiT', filename='white.txt', linecolor='k', italics=False),
    Filter(['unfilt.', '0', 'Clear', 'C', 'clear', 'pseudobolometric', 'griz'], 'w', 0, 'MOSFiT',
           filename='pseudobolometric.txt', linecolor='k', italics=False),
    Filter('G', 'w', 0, 'Gaia', filename='GAIA_GAIA0.G.dat', angstrom=True, linecolor='k'),
    Filter('Kepler', 'r', 0, 'Kepler', filename='Kepler_Kepler.K.dat', angstrom=True, italics=False),
    Filter('TESS', 'r', 0, 'TESS', filename='TESS_TESS.Red.dat', angstrom=True, italics=False),
    Filter('DLT40', 'w', 0, 'DLT40', filename='QE_E2V_MBBBUV_Broadband.csv', linecolor='k', italics=False),
    Filter('w', 'w', 0, 'Gunn', filename='PAN-STARRS_PS1.w.dat', angstrom=True, linecolor='k'),
    Filter(['o', 'orange'], 'orange', 0, 'ATLAS', filename='ATLAS_orange.txt'),
    Filter(['r', "r'", 'rp', 'F625W'], '#FF7D00', 0, 'Gunn', filename='SLOAN_SDSS.r.dat', angstrom=True),
    Filter('r-DECam', '#FF7D00', 0, 'DECam', filename='CTIO_DECam.r.dat', angstrom=True),
    Filter(['R', 'Rc', 'R_s'], '#FF7000', 0, 'Johnson', 3.064e-23, filename='Generic_Cousins.R.dat', angstrom=True),  # '#CC5900'
    Filter(['i', "i'", 'ip', 'F775W'], '#90002C', -1, 'Gunn', filename='SLOAN_SDSS.i.dat', angstrom=True),
    Filter('i-DECam', '#90002C', -1, 'DECam', filename='CTIO_DECam.i.dat', angstrom=True),
    Filter(['I', 'Ic'], '#66000B', -1, 'Johnson', 2.416e-23, filename='Generic_Cousins.I.dat', angstrom=True),  # brightened from '#1C0003'
    Filter('zs', '#000000', -2, 'Gunn', filename='PAN-STARRS_PS1.z.dat', angstrom=True),
    Filter(['z', "z'", 'zp'], '#000000', -2, 'Gunn', filename='SLOAN_SDSS.z.dat', angstrom=True),
    Filter('z-DECam', '#000000', -2, 'DECam', filename='CTIO_DECam.z.dat', angstrom=True),
    Filter('y', 'y', -3, 'Gunn', filename='PAN-STARRS_PS1.y.dat', angstrom=True),
    Filter('y-DECam', 'y', -3, 'DECam', filename='CTIO_DECam.Y.dat', angstrom=True),
    Filter('J', '#444444', -2, 'UKIRT', 1.589e-23, filename='Gemini_Flamingos2.J.dat', angstrom=True),
    Filter('H', '#888888', -3, 'UKIRT', 1.021e-23, filename='Gemini_Flamingos2.H.dat', angstrom=True),
    Filter(['K', 'Ks'], '#CCCCCC', -4, 'UKIRT', 0.640e-23, filename='Gemini_Flamingos2.Ks.dat', angstrom=True),
    Filter('L', 'r', -4, 'UKIRT', 0.285e-23),
    # JWST
    Filter('F070W', 'C7', 0, 'JWST NIRCam', filename='JWST_NIRCam.F070W.dat', angstrom=True, italics=False),
    Filter('F090W', 'C0', 0, 'JWST NIRCam', filename='JWST_NIRCam.F090W.dat', angstrom=True, italics=False),
    Filter('F115W', 'C8', 0, 'JWST NIRCam', filename='JWST_NIRCam.F115W.dat', angstrom=True, italics=False),
    Filter('F150W', 'C1', 0, 'JWST NIRCam', filename='JWST_NIRCam.F150W.dat', angstrom=True, italics=False),
    Filter('F200W', 'C2', 0, 'JWST NIRCam', filename='JWST_NIRCam.F200W.dat', angstrom=True, italics=False),
    Filter('F277W', 'C3', 0, 'JWST NIRCam', filename='JWST_NIRCam.F277W.dat', angstrom=True, italics=False),
    Filter('F356W', 'C4', 0, 'JWST NIRCam', filename='JWST_NIRCam.F356W.dat', angstrom=True, italics=False),
    Filter('F444W', 'C5', 0, 'JWST NIRCam', filename='JWST_NIRCam.F444W.dat', angstrom=True, italics=False),
    Filter('F560W', 'C9', 0, 'JWST MIRI', filename='JWST_MIRI.F560W.dat', angstrom=True, italics=False),
    Filter('F770W', 'C6', 0, 'JWST MIRI', filename='JWST_MIRI.F770W.dat', angstrom=True, italics=False),
    Filter('F1000W', 'C7', 0, 'JWST MIRI', filename='JWST_MIRI.F1000W.dat', angstrom=True, italics=False),
    Filter('F1130W', 'C0', 0, 'JWST MIRI', filename='JWST_MIRI.F1130W.dat', angstrom=True, italics=False),
    Filter('F1280W', 'C8', 0, 'JWST MIRI', filename='JWST_MIRI.F1280W.dat', angstrom=True, italics=False),
    Filter('F1500W', 'C1', 0, 'JWST MIRI', filename='JWST_MIRI.F1500W.dat', angstrom=True, italics=False),
    Filter('F1800W', 'C9', 0, 'JWST MIRI', filename='JWST_MIRI.F1800W.dat', angstrom=True, italics=False),
    Filter('F2100W', 'C2', 0, 'JWST MIRI', filename='JWST_MIRI.F2100W.dat', angstrom=True, italics=False),
    Filter('F2550W', 'C3', 0, 'JWST MIRI', filename='JWST_MIRI.F2550W.dat', angstrom=True, italics=False),
    Filter(['unknown', '?'], 'w', 0, 'unknown', linecolor='k', italics=False)]
Filter.order = [f.name for f in all_filters]
filtdict = {}
for filt in all_filters:
    for name in filt.names:
        filtdict[name] = filt

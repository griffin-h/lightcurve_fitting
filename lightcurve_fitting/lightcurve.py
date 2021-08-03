import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from astropy.table import Table, vstack, MaskedColumn
from .filters import filtdict
import itertools
from matplotlib.markers import MarkerStyle
try:
    from config import markers
except ModuleNotFoundError:
    markers = {}


class Arrow(Path):
    """
    An downward-pointing arrow-shaped ``Path``, whose head has half-width ``hx`` and height ``hy`` (in units of length)
    """
    def __init__(self, hx, hy):
        verts = [(0, 0),
                 (0, -1),
                 (-hx, -1 + hy),
                 (0, -1),
                 (hx, -1 + hy),
                 (0, -1),
                 (0, 0)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        Path.__init__(self, verts, codes)


arrow = Arrow(0.2, 0.3)
othermarkers = ('o', *MarkerStyle.filled_markers[2:])
itermarkers = itertools.cycle(othermarkers)
usedmarkers = []

# if you edit this list, also add the new names to usage.rst
column_names = {
    'Filter': ['filt', 'filter', 'Filter', 'band', 'FLT', 'Band'],
    'Telescope': ['telescope', 'Telescope', 'Tel', 'tel+inst'],
    'Source': ['source', 'Source'],
    'Apparent Magnitude': ['mag', 'Magnitude', 'Mag', 'ab_mag', 'PSFmag', 'MAG', 'omag', 'magnitude', 'apparent_mag'],
    'Apparent Magnitude Uncertainty': [
        'dmag', 'Magnitude_Error', 'magerr', 'MagErr', 'mag_err', 'e_mag', 'Error', 'err', 'PSFerr', 'MAGERR', 'e_omag',
        'e_magnitude', 'apparent_mag_err', 'Mag_Err',
    ],
    'MJD': ['MJD', 'mjd'],
    'JD': ['JD', 'jd'],
    'Phase (rest days)': ['phase', 'Phase', 'PHASE'],
    'Flux $F_ν$ (W m$^{-2}$ Hz$^{-1}$)': ['flux', 'FLUXCAL'],
    'Flux Uncertainty': ['dflux', 'FLUXCALERR'],
    'Nondetection': ['nondet', 'Is_Limit', 'UL', 'l_omag', 'upper_limit', 'upperlimit'],
    'Absolute Magnitude': ['absmag'],
    'Luminosity $L_ν$ (W Hz$^{-1}$)': ['lum'],
    'Luminosity Uncertainty': ['dlum'],
}


class LC(Table):
    """
    A broadband light curve, stored as an :class:`astropy.table.Table` (see ``help(Table)`` for more details)

    Attributes
    ----------
    sn : object
        Astronomical transient with which this light curve is associated
    nondetSigmas : float
        Significance level implied by nondetections in the light curve. Default: 3σ
    groupby : set
        Column names to group by when binning the light curve. Default: ``{'filt', 'filter', 'source'}``
    """
    def __init__(self, *args, **kwargs):
        Table.__init__(self, *args, **kwargs)
        self.sn = None
        self.nondetSigmas = 3.
        self.groupby = {'filt', 'filter', 'source'}
        self.markers = markers.copy()

    def where(self, **kwargs):
        """
        Select the subset of a light curve matching some criteria, given as keyword arguments, e.g., ``colname=value``.

        The keyword ``colname`` can be any of the following:
          * a column in the table, in which case rows must match ``value`` in that column
          * a column in the table + ``_not``, in which case rows must *not* match ``value`` in that column
          * a column in the table + ``_min``, in which case rows must be >= ``value`` in that column
          * a column in the table + ``_max``, in which case rows must be <= ``value`` in that column

        ``value`` must match the data type of the column ``colname`` and can either be a single value or a list of
        values. If ``value`` is a list, rows must match at least one of the values. If ``value`` is a list and
        ``colname`` ends in ``_not``, rows must not match any of the values.
        """
        use = np.tile(True, len(self))
        for col, val in kwargs.items():
            if isinstance(val, list):
                if '_not' in col:
                    use1 = np.tile(True, len(self))
                    for v in val:
                        use1 &= self[col.replace('_not', '')] != v
                else:
                    use1 = np.tile(False, len(self))
                    for v in val:
                        use1 |= self[col] == v
            elif '_min' in col:
                use1 = self[col.replace('_min', '')] >= val
            elif '_max' in col:
                use1 = self[col.replace('_max', '')] <= val
            elif '_not' in col:
                if val is None:
                    use1 = np.array([v is not None for v in self[col.replace('_not', '')]])
                else:
                    use1 = self[col.replace('_not', '')] != val
            else:
                if val is None:
                    use1 = np.array([v is None for v in self[col]])
                else:
                    use1 = self[col] == val
            use &= use1
        selected = self[use]
        selected.sn = self.sn
        selected.meta = self.meta
        selected.markers = self.markers
        return selected

    def get(self, key, default=None):
        return self[key] if key in self.colnames else default

    def normalize_column_names(self):
        """
        Rename any recognizable columns to their standard names for this package (see `lightcurve.column_names`).
        """
        for good_key, *bad_keys in column_names.values():
            if good_key not in self.colnames:
                for bad_key in bad_keys:
                    if bad_key in self.colnames:
                        self.rename_column(bad_key, good_key)
                        break
        if 'MJD' not in self.colnames and 'JD' in self.colnames:
            self['MJD'] = self['JD'] - 2400000.5
            self.remove_column('JD')
        if 'nondet' in self.colnames and self['nondet'].dtype != bool:
            if isinstance(self['nondet'], MaskedColumn):
                self['nondet'] = self['nondet'].filled()
            nondet = (self['nondet'] == 'True') | (self['nondet'] == 'T') | (self['nondet'] == '>')
            self.remove_column('nondet')
            self['nondet'] = nondet

    def filters_to_objects(self, read_curve=True):
        """
        Parse the ``'filt'`` column into :class:`filters.Filter` objects and store in the ``'filter'`` column

        Parameters
        ----------
        read_curve : bool, optional
            Read in the transmission function for each filter encountered (default)
        """
        self['filter'] = [filtdict['?'] if np.ma.is_masked(f) or str(f) not in filtdict else filtdict[str(f)]
                          for f in self['filt']]
        is_swift = np.zeros(len(self), bool)
        if 'telescope' in self.colnames:
            is_swift |= self['telescope'] == 'Swift'
            is_swift |= self['telescope'] == 'UVOT'
            is_swift |= self['telescope'] == 'Swift/UVOT'
            is_swift |= self['telescope'] == 'Swift+UVOT'
        if 'source' in self.colnames:
            is_swift |= self['source'] == 'SOUSA'
        if is_swift.any():
            for filt, swiftfilt in zip('UBV', 'sbv'):
                self['filter'][is_swift & (self['filt'] == filt)] = filtdict[swiftfilt]
        if read_curve:
            for filt in np.unique(self['filter']):
                filt.read_curve()

    @property
    def zp(self):
        """
        Returns an array of zero points for each filter in the ``'filter'`` column
        """
        return np.array([f.m0 for f in self['filter']])

    def calcFlux(self, nondetSigmas=None, zp=None):
        """
        Calculate the ``'flux'`` and ``'dflux'`` columns from the ``'mag'`` and ``'dmag'`` columns

        Parameters
        ----------
        nondetSigmas : float, optional
            Significance level implied by nondetections in the light curve. Default: 3σ
        zp : float, array-like, optional
            Array of zero points for each magnitude. Default: standard for each filter
        """
        if nondetSigmas is not None:
            self.nondetSigmas = nondetSigmas
        if zp is None:
            zp = self.zp
        self['flux'], self['dflux'] = mag2flux(self['mag'], self['dmag'], zp, self.get('nondet'), self.nondetSigmas)

    def bin(self, delta=0.3, groupby=None):
        """
        Bin the light curve by averaging points within ``delta`` days of each other

        Parameters
        ----------
        delta : float, optional
            Bin size, in days. Default: 0.3 days
        groupby : set, optional
            Column names to group by before binning. Default: ``{'filt', 'filter', 'source'}``

        Returns
        -------
        lc : lightcurve_fitting.lightcurve.LC
            Binned light curve
        """
        if groupby is not None:
            self.groupby = groupby
        subtabs = []
        self.groupby = list(set(self.groupby) & set(self.colnames))
        if self.groupby:
            grouped = self.group_by(self.groupby)
        else:
            grouped = self
        for g, k in zip(grouped.groups, grouped.groups.keys):
            mjd, flux, dflux = binflux(g['MJD'], g['flux'], g['dflux'], delta)
            binned = LC([mjd, flux, dflux], names=['MJD', 'flux', 'dflux'])
            for key in self.groupby:
                binned[key] = k[key]
            subtabs.append(binned)
        lc = vstack(subtabs)
        lc.sn = self.sn
        lc.meta = self.meta
        return lc

    def findNondet(self, nondetSigmas=None):
        """
        Add a boolean column ``'nondet'`` indicating flux measurements that are below the detection threshold

        Parameters
        ----------
        nondetSigmas : float, optional
            Significance level implied by nondetections in the light curve. Default: 3σ
        """
        if nondetSigmas is not None:
            self.nondetSigmas = nondetSigmas
        self['nondet'] = self['flux'] < self.nondetSigmas * self['dflux']

    def calcMag(self, nondetSigmas=None, zp=None):
        """
        Calculate the ``'mag'`` and ``'dmag'`` columns from the ``'flux'`` and ``'dflux'`` columns

        Parameters
        ----------
        nondetSigmas : float, optional
            Significance level implied by nondetections in the light curve. Default: 3σ
        zp : float, array-like, optional
            Array of zero points for each magnitude. Default: standard for each filter
        """
        if nondetSigmas is not None:
            self.nondetSigmas = nondetSigmas
        self.findNondet()
        if zp is None:
            zp = self.zp
        self['mag'], self['dmag'] = flux2mag(self['flux'], self['dflux'], zp, self.get('nondet'), self.nondetSigmas)

    def calcAbsMag(self, dm=None, extinction=None, hostext=None):
        """
        Calculate the ``'absmag'`` column from the ``'mag'`` column by correcting for distance and extinction

        Parameters
        ----------
        dm : float, optional
            Distance modulus. Default: use the distance modulus of ``self.sn``, if any. Otherwise do not correct for
            distance.
        extinction : dict, optional
            Milky Way extinction coefficients :math:`A_λ` for each filter. Default: use the extinction of ``self.sn``,
            if any. Otherwise do not correct for Milky Way extinction.
        hostext : dict, optional
            Host galaxy extinction coefficients :math:`A_λ` for each filter. Default: use the extinction of ``self.sn``,
            if any. Otherwise do not correct for host galaxy extinction.
        """
        if dm is not None:
            self.meta['dm'] = dm
        elif self.sn is not None:
            self.meta['dm'] = self.sn.dm
        elif 'dm' not in self.meta:
            self.meta['dm'] = 0.

        if extinction is not None:
            self.meta['extinction'] = extinction
        elif self.sn is not None:
            self.meta['extinction'] = self.sn.extinction
        elif 'extinction' not in self.meta:
            self.meta['extinction'] = {}

        if hostext is not None:
            self.meta['hostext'] = hostext
        elif self.sn is not None:
            self.meta['hostext'] = self.sn.hostext
        elif 'hostext' not in self.meta:
            self.meta['hostext'] = {}

        self['absmag'] = self['mag'].data - self.meta['dm']
        for filt, A in self.meta['extinction'].items():
            self['absmag'][self['filter'] == filtdict[filt]] -= A
        for filt, A in self.meta['hostext'].items():
            self['absmag'][self['filter'] == filtdict[filt]] -= A

    def calcLum(self, nondetSigmas=None):
        """
        Calculate the ``'lum'`` and ``'dlum'`` columns from the ``'absmag'`` and ``'dmag'`` columns

        Parameters
        ----------
        nondetSigmas : float, optional
            Significance level implied by nondetections in the light curve. Default: 3σ
        """
        if nondetSigmas is not None:
            self.nondetSigmas = nondetSigmas
        self['lum'], self['dlum'] = mag2flux(self['absmag'], self['dmag'], self.zp + 90.19, self.get('nondet'),
                                             self.nondetSigmas)

    def findPeak(self, **criteria):
        """
        Find the peak of the light curve and store it in ``self.sn.peakdate``

        Parameters
        ----------
        criteria : dict, optional
            Use only a subset of the light curve matching some criteria when calculating the peak date (stored in
            ``self.sn.peakcriteria``
        """
        useforpeak = np.ones_like(self, bool)
        if 'nondet' in self.colnames:
            useforpeak &= ~self['nondet']
        for key, value in criteria.items():
            if isinstance(value, list):
                subuseforpeak = np.tile(False, len(self))
                for val in value:
                    subuseforpeak |= self[key] == val
            else:
                subuseforpeak = self[key] == value
            useforpeak &= subuseforpeak
        if np.any(useforpeak):
            peaktable = self[useforpeak]
            imin = np.argmin(peaktable['mag'])
            self.sn.peakdate = peaktable['MJD'][imin]
        else:
            self.sn.peakdate = np.nan
        self.sn.peakcriteria = criteria

    def calcPhase(self, rdsp=False):
        """
        Calculate the rest-frame ``'phase'`` column from the ``'MJD'`` column and ``self.sn.refmjd`` and ``self.sn.z``

        Parameters
        ----------
        rdsp : bool, optional
            Define phase as rest-frame days since peak, rather than rest-frame days since explosion
        """
        if 'refmjd' not in self.meta and self.sn is not None:
            if rdsp and self.sn.peakdate is None:
                raise Exception('must run sn.findPeak() first')
            elif rdsp:
                self.sn.refmjd = self.sn.peakdate
            elif self.sn.explosion is not None:
                self.sn.refmjd = self.sn.explosion
            else:
                if 'nondet' in self.colnames:
                    detections = self.where(nondet=False)
                else:
                    detections = self
                self.sn.refmjd = np.min(detections['MJD'].data)
            self.meta['refmjd'] = self.sn.refmjd
            self.meta['redshift'] = self.sn.z
        self['phase'] = (self['MJD'].data - self.meta['refmjd']) / (1 + self.meta['redshift'])
        if 'dMJD0' in self.colnames:
            self['dphase0'] = self['dMJD0'] / (1. + self.meta['redshift'])
        if 'dMJD1' in self.colnames:
            self['dphase1'] = self['dMJD1'] / (1. + self.meta['redshift'])

    def plot(self, xcol='phase', ycol='absmag', offset_factor=1., color='filter', marker=None, use_lines=False,
             normalize=False, fillmark=True, **kwargs):
        """
        Plot the light curve, with nondetections marked with a downward-pointing arrow

        Parameters
        ----------
        xcol : str, optional
            Column to plot on the horizontal axis. Default: ``'phase'``
        ycol : str, optional
            Column to plot on the vertical axis. Default: ``'absmag'``
        offset_factor : float, optional
            Increase or decrease the filter offsets by a constant factor. Default: 1.
        color : str, optional
            Column that controls the color of the lines and points. Default: ``'filter'``
        marker : str, optional
            Column that controls the marker shape. Default: ``'source'`` or ``'telescope'``
        use_lines : bool, optional
            Connect light curve points with lines. Default: False
        normalize : bool, optional
            Normalize all light curves to peak at 0. Default: False
        fillmark : bool, optional
            Fill each marker with color. Default: True
        kwargs : dict, optional
            Keyword arguments matching column names in the light curve are used to specify a subset of points to plot.
            Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`.
        """
        xchoices = ['phase', 'MJD']
        while xcol not in self.keys():
            xchoices.remove(xcol)
            if xchoices:
                xcol = xchoices[0]
            else:
                raise Exception('no columns found for x-axis')
        ychoices = ['absmag', 'mag']
        while ycol not in self.keys():
            ychoices.remove(ycol)
            if ychoices:
                ycol = ychoices[0]
            else:
                raise Exception('no columns found for y-axis')
        if marker is None:
            if 'source' in self.colnames:
                marker = 'source'
            elif 'telescope' in self.colnames:
                marker = 'telescope'
            else:
                marker = 'o'
        plotthese = np.tile(True, len(self))
        criteria = {key: val for key, val in kwargs.items() if key in self.colnames}
        plot_kwargs = {key: val for key, val in kwargs.items() if key not in self.colnames}
        for key, value in criteria.items():
            if isinstance(value, list):
                subplotthese = np.tile(False, len(self))
                for val in value:
                    subplotthese |= self[key] == val
            else:
                subplotthese = self[key] == value
            plotthese &= subplotthese
        if np.any(plotthese):
            plottable = self[plotthese]
        else:
            return
        groupby = []
        if color in plottable.keys():
            groupby.append(color)
        if marker in plottable.keys():
            groupby.append(marker)
        if groupby:
            plottable = plottable.group_by(groupby)
        for g in plottable.groups:
            filt = g['filter'][0]
            if color == 'filter':
                col = filt.color
                mec = 'k' if filt.system == 'Johnson' else filt.linecolor
            elif color == 'name':
                col = self.sn.plotcolor
                mec = col if col not in ['w', '#FFFFFF'] else 'k'
            else:
                col = 'k'
                mec = 'k'
            mfc = col if fillmark else 'none'
            if marker == 'name':
                mark = self.sn.marker
            elif marker in plottable.keys():
                if g[marker][0] not in self.markers:
                    for nextmarker in othermarkers:
                        if nextmarker not in usedmarkers:
                            self.markers[g[marker][0]] = nextmarker
                            break
                    else:
                        for nextmarker in itermarkers:
                            self.markers[g[marker][0]] = nextmarker
                            break
                mark = self.markers[g[marker][0]]
            elif marker in MarkerStyle.markers:
                mark = marker
            else:
                mark = MarkerStyle.markers[0]
            usedmarkers.append(mark)
            if use_lines:
                g.sort(xcol)
            elif 'mag' in ycol:
                yerr = g['dmag']
            else:
                yerr = g['d' + ycol]
            x = g[xcol].data
            y = g[ycol].data - filt.offset * offset_factor
            if normalize and ycol == 'mag':
                y -= self.sn.peakmag
            elif normalize and ycol == 'absmag':
                y -= self.sn.peakabsmag
            if 'mag' in ycol and 'nondet' in g.keys() and marker:  # don't plot if no markers used
                plt.plot(x[g['nondet']], y[g['nondet']], marker=arrow, linestyle='none', ms=25, mec=mec, **plot_kwargs)
            if len(filt.name) >= 4 and not filt.offset:
                label = filt.name
            elif offset_factor:
                label = '${}{:+.0f}$'.format(filt.name, -filt.offset * offset_factor)
            else:
                label = '${}$'.format(filt.name)
            if self.sn is None:
                linestyle = None
                linewidth = None
            else:
                linestyle = self.sn.linestyle
                linewidth = self.sn.linewidth
            if not use_lines:
                plt.errorbar(x, y, yerr, color=mec, mfc=mfc, mec=mec, marker=mark, linestyle='none', label=label,
                             **plot_kwargs)
            elif 'mag' in ycol and 'nondet' in g.colnames:
                plt.plot(x[~g['nondet']], y[~g['nondet']], color=col, mfc=mfc, mec=mec, marker=mark, label=label,
                         linestyle=linestyle, linewidth=linewidth, **plot_kwargs)
                plt.plot(x[g['nondet']], y[g['nondet']], color=mec, mfc=mfc, mec=mec, marker=mark, linestyle='none',
                         **plot_kwargs)
            else:
                plt.plot(x, y, color=mec, mfc=mfc, mec=mec, marker=mark, label=label, linestyle=linestyle,
                         linewidth=linewidth, **plot_kwargs)
        ymin, ymax = plt.ylim()
        if 'mag' in ycol and ymax > ymin:
            plt.ylim(ymax, ymin)
        for axlabel, keys in column_names.items():
            if xcol in keys:
                plt.xlabel(axlabel)
            elif ycol in keys:
                plt.ylabel(axlabel)

    @classmethod
    def read(cls, filepath, format='ascii', fill_values=None, **kwargs):
        if fill_values is None:
            fill_values = [('--', '0'), ('', '0')]
        t = super(LC, cls).read(filepath, format=format, fill_values=fill_values, **kwargs)
        t.normalize_column_names()
        if 'filt' in t.colnames:
            t.filters_to_objects()
        return t


def flux2mag(flux, dflux=np.array(np.nan), zp=0., nondet=None, nondetSigmas=3.):
    """
    Convert flux (and uncertainty) to magnitude (and uncertainty). Nondetections are converted to limiting magnitudes.

    Parameters
    ----------
    flux : float, array-like
        Flux to be converted
    dflux : float, array-like, optional
        Uncertainty on the flux to be converted. Default: :mod:`numpy.nan`
    zp : float, array-like, optional
        Zero point to be applied to the magnitudes
    nondet : array-like
        Boolean or index array indicating the nondetections among the fluxes. Default: no nondetections
    nondetSigmas : float, optional
        Significance level implied by nondetections in the light curve. Default: 3σ

    Returns
    -------
    mag : float, array-like
        Magnitude corresponding to the input flux
    dmag : float, array-like
        Uncertainty on the output magnitude
    """
    flux = flux.copy()
    dflux = dflux.copy()
    if nondet is not None:
        flux[nondet] = nondetSigmas * dflux[nondet]
        dflux[nondet] = np.nan
    mag = -2.5 * np.log10(flux) + zp
    dmag = 2.5 * dflux / (flux * np.log(10))
    return mag, dmag


def mag2flux(mag, dmag=np.nan, zp=0., nondet=None, nondetSigmas=3.):
    """
    Convert magnitude (and uncertainty) to flux (and uncertainty). Nondetections are assumed to imply zero flux.

    Parameters
    ----------
    mag : float, array-like
        Magnitude to be converted
    dmag : float, array-like, optional
        Uncertainty on the magnitude to be converted. Default: :mod:`numpy.nan`
    zp : float, array-like, optional
        Zero point to be applied to the magnitudes
    nondet : array-like
        Boolean or index array indicating the nondetections among the fluxes. Default: no nondetections
    nondetSigmas : float, optional
        Significance level implied by nondetections in the light curve. Default: 3σ

    Returns
    -------
    flux : float, array-like
        Flux corresponding to the input magnitude
    dflux : float, array-like
        Uncertainty on the output flux
    """
    flux = 10 ** ((zp - mag) / 2.5)
    dflux = np.log(10) / 2.5 * flux * dmag
    if nondet is not None:
        dflux[nondet] = flux[nondet] / nondetSigmas
        flux[nondet] = 0
    return flux, dflux


def binflux(time, flux, dflux, delta=0.2, include_zero=True):
    """
    Bin a light curve by averaging points within ``delta`` of each other in time

    Parameters
    ----------
    time, flux, dflux : array-like
        Arrays of times, fluxes, and uncertainties comprising the observed light curve
    delta : float, optional
        Bin size, in the same units as ``time``. Default: 0.2
    include_zero : bool, optional
        Include data points with no error bar

    Returns
    -------
    time, flux, dflux : array-like
        Binned arrays of times, fluxes, and uncertainties
    """
    bin_time = []
    bin_flux = []
    bin_dflux = []
    while len(flux) > 0:
        grp = np.array(abs(time - time[0]) <= delta)
        time_grp = time[grp]
        flux_grp = flux[grp]
        dflux_grp = dflux[grp]

        # Indices with no error bar
        zeros = (dflux_grp == 0) | (dflux_grp == 999) | (dflux_grp == 9999) | (dflux_grp == -1) | np.isnan(dflux_grp)
        if np.ma.is_masked(dflux_grp):
            zeros = zeros.data | dflux_grp.mask

        if any(zeros) and include_zero:
            x = np.mean(time_grp)
            y = np.mean(flux_grp)
            z = 0.
        else:
            # Remove points with no error bars
            time_grp = time_grp[~zeros]
            flux_grp = flux_grp[~zeros]
            dflux_grp = dflux_grp[~zeros]

            x = np.mean(time_grp)
            y = np.sum(flux_grp * dflux_grp ** -2) / np.sum(dflux_grp ** -2)
            z = np.sum(dflux_grp ** -2) ** -0.5
        # Append result 
        bin_time.append(x)
        bin_flux.append(y)
        bin_dflux.append(z)
        # Remove data points already used
        time = time[~grp]
        flux = flux[~grp]
        dflux = dflux[~grp]
    time = np.array(bin_time)
    flux = np.array(bin_flux)
    dflux = np.array(bin_dflux)
    return time, flux, dflux

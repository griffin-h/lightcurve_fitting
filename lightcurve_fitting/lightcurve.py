import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from astropy.table import Table, vstack, MaskedColumn
from astropy.cosmology import Planck18
from .filters import filtdict
import itertools
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
from functools import partial
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
itercolors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# if you edit this list, also add the new names to usage.rst
column_names = {
    'Filter': ['filter', 'filt', 'Filter', 'band', 'FLT', 'Band'],
    'Telescope': ['telescope', 'Telescope', 'Tel', 'tel+inst'],
    'Source': ['source', 'Source'],
    'Apparent Magnitude': ['mag', 'Magnitude', 'Mag', 'ab_mag', 'PSFmag', 'MAG', 'omag', 'magnitude', 'apparent_mag'],
    'Apparent Magnitude Uncertainty': [
        'dmag', 'Magnitude_Error', 'magerr', 'MagErr', 'mag_err', 'e_mag', 'Error', 'err', 'PSFerr', 'MAGERR', 'e_omag',
        'e_magnitude', 'apparent_mag_err', 'Mag_Err', 'emag', 'error',
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
    'Effective Wavelength (nm)': ['wl_eff'],  # calculated from filters; does not need to be in usage.rst
}


class LC(Table):
    """
    A broadband light curve, stored as an :class:`astropy.table.Table`

    Attributes
    ----------
    nondetSigmas : float
        Significance level implied by nondetections in the light curve. Default: 3σ
    groupby : set
        Column names to group by when binning the light curve. Default: ``{'filter', 'source'}``
    markers : dict
        Mapping of some light curve property (default: ``'source'`` or ``'telescope'``) to marker shapes
    """
    def __init__(self, *args, **kwargs):
        Table.__init__(self, *args, **kwargs)
        self.normalize_column_names()
        if 'filter' in self.colnames and self['filter'].dtype != object:
            self.filters_to_objects()
        self.nondetSigmas = 3.
        self.groupby = {'filter', 'source'}
        self.markers = markers.copy()
        self.colors = {}

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
            if col.startswith('filter'):  # allow constraints like filter='r' so the user does not need to use filtdict
                if isinstance(val, str):
                    val = filtdict[val]
                elif isinstance(val, list):
                    val = [filtdict[v] if isinstance(v, str) else v for v in val]
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
            self.replace_column('nondet', nondet)

    def filters_to_objects(self):
        """
        Parse the ``'filter'`` column into :class:`filters.Filter` objects
        """
        filters = np.array([filtdict['0'] if np.ma.is_masked(f) else filtdict.get(str(f), filtdict['?'])
                            for f in self['filter']])
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
                filters[is_swift & (self['filter'] == filt)] = filtdict[swiftfilt]
        self.replace_column('filter', filters)

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
            Column names to group by before binning. Default: ``{'filter', 'source'}``

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

    def calcAbsMag(self, dm=None, extinction=None, hostext=None, ebv=None, rv=None, host_ebv=None, host_rv=None,
                   redshift=None):
        """
        Calculate the ``'absmag'`` column from the ``'mag'`` column by correcting for distance and extinction

        Parameters
        ----------
        dm : float, optional
            Distance modulus. Default: calculate from ``redshift``.
        extinction : dict, optional
            Milky Way extinction coefficients :math:`A_λ` for each filter. Default: calculate from ``ebv`` and ``rv``.
        hostext : dict, optional
            Host galaxy extinction coefficients :math:`A_λ` for each filter. Default: calculate from ``host_ebv`` and
            ``host_rv``.
        ebv : float, optional
            Milky Way selective extinction :math:`E(B-V)`, used if ``extinction`` is not given. Default: 0.
        host_ebv : float, optional
            Host galaxy selective extinction :math:`E(B-V)`, used if ``hostext`` is not given. Default: 0.
        rv : float, optional
            Ratio of total to selective Milky Way extinction :math:`R_V`, used with the ``ebv`` argument. Default: 3.1.
        host_rv : float, optional
            Ratio of total to selective host-galaxy extinction :math:`R_V`, used with the ``host_ebv`` argument.
            Default: 3.1.
        redshift : float, optional
            Redshift of the host galaxy. Used to redshift the filters for host galaxy extinction. If no distance
            modulus is given, a redshift-dependent distance is calculated using the Planck18 cosmology. Default: 0.
        """
        if redshift is not None:
            self.meta['redshift'] = redshift
        elif 'redshift' not in self.meta:
            self.meta['redshift'] = 0.

        if dm is not None:
            self.meta['dm'] = dm
        elif 'dm' not in self.meta and self.meta.get('redshift'):
            self.meta['dm'] = Planck18.distmod(self.meta['redshift']).value
            print('using a redshift-dependent distance modulus')
        elif 'dm' not in self.meta:
            self.meta['dm'] = 0.

        if ebv is None:
            ebv = self.meta.get('ebv')
        if host_ebv is None:
            host_ebv = self.meta.get('host_ebv')
        if rv is None:
            rv = self.meta.get('rv', 3.1)
        if host_rv is None:
            host_rv = self.meta.get('host_rv', 3.1)

        if extinction is not None:
            self.meta['extinction'] = extinction
        elif 'extinction' not in self.meta:
            self.meta['extinction'] = {f.name: f.extinction(ebv, rv)
                                       for f in set(self['filter']) if f.wl_eff is not None and ebv is not None}

        if hostext is not None:
            self.meta['hostext'] = hostext
        elif 'hostext' not in self.meta:
            self.meta['hostext'] = {f.name: f.extinction(host_ebv, host_rv, self.meta.get('z', 0.))
                                    for f in set(self['filter']) if f.wl_eff is not None and host_ebv is not None}

        self['absmag'] = self['mag'].data - self.meta['dm']
        for filtobj in set(self['filter']):
            for filt in filtobj.names:
                if filt in self.meta['extinction']:
                    self['absmag'][self['filter'] == filtobj] -= self.meta['extinction'][filt]
                    break
            else:
                print('MW extinction not applied to filter', filtobj)
            for filt in filtobj.names:
                if filt in self.meta['hostext']:
                    self['absmag'][self['filter'] == filtobj] -= self.meta['hostext'][filt]
                    break
            else:
                print('host extinction not applied to filter', filtobj)

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
        Find the peak of the light curve and store it in ``.meta['peakdate']``

        Parameters
        ----------
        criteria : dict, optional
            Use only a subset of the light curve matching some criteria when calculating the peak date (stored in
            ``.meta['peakcriteria']``)
        """
        if 'nondet' in self.colnames:
            criteria['nondet'] = False
        peaktable = self.where(**criteria)
        if len(peaktable):
            imin = np.argmin(peaktable['mag'])
            self.meta['peakdate'] = peaktable['MJD'][imin]
            self.meta['peakcriteria'] = criteria
        else:
            print(f'no data match these criteria: {criteria}')

    def calcPhase(self, rdsp=False, hours=False):
        """
        Calculate the rest-frame ``'phase'`` column from ``'MJD'``, ``.meta['refmjd']``, and ``.meta['redshift']``

        Parameters
        ----------
        rdsp : bool, optional
            Define phase as rest-frame days since peak, rather than rest-frame days since explosion
        hours : bool, optional
            Give the phase in rest-frame hours instead of rest-frame days
        """
        if 'refmjd' not in self.meta:
            if rdsp and self.meta.get('peakdate') is None:
                raise Exception('must run lc.findPeak() first')
            elif rdsp:
                self.meta['refmjd'] = self.meta['peakdate']
            elif self.meta.get('explosion') is not None:
                self.meta['refmjd'] = self.meta['explosion']
            else:
                if 'nondet' in self.colnames:
                    detections = self.where(nondet=False)
                else:
                    detections = self
                self.meta['refmjd'] = np.min(detections['MJD'].data)
        self['phase'] = (self['MJD'].data - self.meta['refmjd']) / (1 + self.meta['redshift'])
        if 'dMJD0' in self.colnames:
            self['dphase0'] = self['dMJD0'] / (1. + self.meta['redshift'])
        if 'dMJD1' in self.colnames:
            self['dphase1'] = self['dMJD1'] / (1. + self.meta['redshift'])
        if hours:
            self['phase'] *= 24.
            if 'dphase0' in self.colnames:
                self['dphase0'] *= 24.
            if 'dphase1' in self.colnames:
                self['dphase1'] *= 24.

    def plot(self, xcol='phase', ycol='absmag', offset_factor=1., color='filter', marker=None, use_lines=False,
             normalize=False, fillmark=True, mjd_axis=True, appmag_axis=True, loc_mark=None, loc_filt=None, ncol_mark=1,
             lgd_filters=None, tight_layout=True, phase_hours=False, return_axes=False, **kwargs):
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
        mjd_axis : bool, optional
            Plot MJD on the upper x-axis. Must have ``.meta['redshift']`` and ``.meta['refmjd']``. Default: True.
        appmag_axis : bool, optional
            Plot extinction-corrected apparent magnitude on the right y-axis. Must have ``.meta['dm']``. Default: True.
        loc_mark, loc_filt : str, optional
            Location for the marker and filter legends, respectively. Set to 'none' to omit them. Three new options are
            available: 'above', 'above left', and 'above right'. ``mjd_axis`` and/or ``appmag_axis`` must be used to
            add these legends. Otherwise run ``plt.legend()`` after plotting for a single simple legend. Default: no
            legend.
        ncol_mark : int, optional
            Number of columns in the marker legend. Default: 1.
        lgd_filters : list, array-like, optional
            Customize the arrangement of filters in the legend by providing a list of filters for each column. ``None``
            can be used to leave a blank space in the column. Only filters given here will be used. The default
            arrangement shows all filters arranged by ``.system`` (columns) and ``.offset`` (rows).
        tight_layout : bool, optional
            Adjust the figure margins to look beautiful. Default: True.
        phase_hours : bool, optional
            Plot the phase in units of rest-frame hours instead of rest-frame days. Default: False.
        return_axes : bool, optional
            Return the newly created axes if ``mjd_axis=True`` or ``appmag_axis=True``. Default: False.
        kwargs
            Keyword arguments matching column names in the light curve are used to specify a subset of points to plot.
            Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        top : matplotlib.pyplot.Axes, optional
            The upper x-axis, if ``mjd_axis=True`` and ``return_axes=True``. Otherwise, None.
        right : matplotlib.pyplot.Axes, optional
            The right y-axis, if ``appmag_axis=True`` and ``return_axes=True``. Otherwise, None.
        """
        if xcol.startswith('filter'):
            unit = xcol.split(':')[-1] if ':' in xcol else None
            xcol = 'wl_eff'
            self[xcol] = [f.wl_eff.to(unit) if unit else f.wl_eff for f in self['filter']]
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
        criteria = {key: val for key, val in kwargs.items() if key in self.colnames}
        plot_kwargs = {key: val for key, val in kwargs.items() if key not in self.colnames}
        plottable = self.where(**criteria)
        if len(plottable) == 0:
            return
        groupby = set()
        if color in plottable.keys():
            groupby.add(color)
        if marker in plottable.keys():
            groupby.add(marker)
        if groupby:
            plottable = plottable.group_by(list(groupby))
            keys = plottable.groups.keys
        else:
            keys = [Table()]
        linestyle = plot_kwargs.pop('linestyle', plot_kwargs.pop('ls', self.meta.get('linestyle', self.meta.get('ls'))))
        linewidth = plot_kwargs.pop('linewidth', plot_kwargs.pop('lw', self.meta.get('linewidth', self.meta.get('lw'))))
        ms = plot_kwargs.pop('markersize', plot_kwargs.pop('ms', plt.rcParams['lines.markersize']))
        for g, k in zip(plottable.groups, keys):
            filt = g['filter'][0]
            if color == 'filter':
                col = filt.color
                mec = filt.mec
            elif color == 'name' and 'plotcolor' in self.meta:
                col = self.meta['plotcolor']
                mec = col if col not in ['w', '#FFFFFF'] else 'k'
            elif g[color][0] in self.colors:
                col = self.colors[g[color][0]]
                mec = col if col not in ['w', '#FFFFFF'] else 'k'
            else:
                col = mec = next(itercolors)
            self.colors[g[color][0]] = col
            mfc = col if fillmark else 'none'
            if marker == 'name' and 'marker' in self.meta:
                mark = self.meta['marker']
            elif marker in plottable.keys():
                if g[marker][0] not in self.markers:
                    for nextmarker in othermarkers:
                        if nextmarker not in usedmarkers:
                            self.markers[g[marker][0]] = nextmarker
                            break
                    else:
                        self.markers[g[marker][0]] = next(itermarkers)
                mark = self.markers[g[marker][0]]
            elif marker in MarkerStyle.markers:
                mark = marker
            elif marker == 'none':
                mark = None
            else:
                mark = next(itermarkers)
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
                if 'peakmag' in self.meta:
                    y -= self.meta['peakmag']
                else:
                    print("must set .meta['peakmag'] to use normalize")
            elif normalize and ycol == 'absmag':
                if 'peakabsmag' in self.meta:
                    y -= self.meta['peakmag']
                else:
                    print("must set .meta['peakabsmag'] to use normalize")
            if 'mag' in ycol and 'nondet' in g.keys() and marker:  # don't plot if no markers used
                plt.plot(x[g['nondet']], y[g['nondet']], marker=arrow, linestyle='none', ms=ms / 6. * 25., mec=mec,
                         **plot_kwargs)
            if 'filter' in k.colnames:
                if len(filt.name) >= 4 and not filt.offset:
                    k['filter'] = filt.name
                elif offset_factor:
                    k['filter'] = '${}{:+.0f}$'.format(filt.name, -filt.offset * offset_factor)
                else:
                    k['filter'] = '${}$'.format(filt.name)
            label = ' '.join([str(kv) for kv in k.values()])
            if not use_lines:
                plt.errorbar(x, y, yerr, color=mec, mfc=mfc, mec=mec, ms=ms, marker=mark, linestyle='none', label=label,
                             **plot_kwargs)
            elif 'mag' in ycol and 'nondet' in g.colnames:
                plt.plot(x[~g['nondet']], y[~g['nondet']], color=col, mfc=mfc, mec=mec, ms=ms, marker=mark, label=label,
                         linestyle=linestyle, linewidth=linewidth, **plot_kwargs)
                plt.plot(x[g['nondet']], y[g['nondet']], color=mec, mfc=mfc, mec=mec, ms=ms, marker=mark,
                         linestyle='none', **plot_kwargs)
            else:
                plt.plot(x, y, color=col, mfc=mfc, mec=mec, ms=ms, marker=mark, label=label, linestyle=linestyle,
                         linewidth=linewidth, **plot_kwargs)

        # format axes
        ymin, ymax = plt.ylim()
        if 'mag' in ycol and ymax > ymin:
            plt.ylim(ymax, ymin)
        lgd_title = None
        for axlabel, keys in column_names.items():
            if xcol in keys:
                if xcol == 'phase' and phase_hours:
                    axlabel = axlabel.replace('days', 'hours')
                plt.xlabel(axlabel)
            elif ycol in keys:
                plt.ylabel(axlabel)
            elif marker in keys:
                lgd_title = axlabel

        # add auxiliary axes
        mjd_axis = mjd_axis and xcol == 'phase' and 'redshift' in self.meta and 'refmjd' in self.meta
        appmag_axis = appmag_axis and ycol == 'absmag' and 'dm' in self.meta
        if mjd_axis or appmag_axis:
            xfunc = partial(self._phase2mjd, hours=phase_hours)
            top, right = aux_axes(xfunc if mjd_axis else None, self._abs2app if appmag_axis else None)
            if mjd_axis:
                top.xaxis.get_major_formatter().set_useOffset(False)
                top.set_xlabel('MJD')
            if appmag_axis:
                right.set_ylabel('Apparent Magnitude')

        # add legends
            if marker in self.colnames:
                labels = sorted(set(self[marker]), key=lambda s: s.lower())
                lines = []
                for label in labels:
                    if marker == color:
                        mec = mfc = self.colors[label]
                    else:
                        mec = 'k'
                        mfc = 'none'
                    line = plt.Line2D([], [], mec=mec, mfc=mfc, ms=ms, marker=self.markers[label], linestyle='none')
                    lines.append(line)
                custom_legend(top, lines, labels, ncol=ncol_mark, loc=loc_mark, title=lgd_title, frameon=True)

            if color == 'filter':
                if lgd_filters is None:
                    lgd_filters = set(self['filter'])
                lines, labels, ncol = filter_legend(lgd_filters, offset_factor)
                custom_legend(right, lines, labels, loc=loc_filt, ncol=ncol, title='Filter', frameon=True)

        if tight_layout:
            plt.tight_layout()

        if return_axes and (mjd_axis or appmag_axis):
            return top, right

    def _phase2mjd(self, phase, hours=False):
        return phase * (1. + self.meta['redshift']) / (24. if hours else 1.) + self.meta['refmjd']

    def _abs2app(self, absmag):
        return absmag + self.meta['dm']  # extinction-corrected apparent magnitude

    @classmethod
    def read(cls, filepath, format='ascii', fill_values=None, **kwargs):
        if fill_values is None:
            fill_values = [('--', '0'), ('', '0')]
        t = super(LC, cls).read(filepath, format=format, fill_values=fill_values, **kwargs)
        return t

    def write(self, *args, **kwargs):
        # Filter is not serializable, so produce a copy of the LC object with 'filter' as a string
        out = Table(self)
        if 'filter' in out.colnames:
            out.replace_column('filter', self['filter'].astype(str))
        out.write(*args, **kwargs)


def aux_axes(xfunc=None, yfunc=None, ax0=None, xfunc_args=None, yfunc_args=None):
    """
    Add auxiliary axes to a plot that are linear transformations of the existing axes

    Parameters
    ----------
    xfunc : function, optional
        Function that transforms the lower x-axis to the upper x-axis. Default: do not add an upper x-axis.
    yfunc : function, optional
        Function that transforms the left y-axis to the right y-axis. Default: do not add a right y-axis.
    ax0 : matplotlib.pyplot.Axes, optional
        Existing axes object. Default: use the current Axes.
    xfunc_args, yfunc_args : dict, optional
        Keyword arguments for ``xfunc`` and ``yfunc``, respectively

    Returns
    -------
    top : matplotlib.pyplot.Axes
        The upper x-axis, if any. Otherwise, None.
    right : matplotlib.pyplot.Axes
        The right y-axis, if any. Otherwise, None.
    """
    if xfunc_args is None:
        xfunc_args = {}
    if yfunc_args is None:
        yfunc_args = {}
    if not ax0:
        ax0 = plt.gca()
    lims = np.array(ax0.axis())
    if xfunc is not None:
        ax0.xaxis.tick_bottom()
        lims[:2] = xfunc(lims[:2], **xfunc_args)
        top = ax0.twiny()
        top.axis(lims)
    else:
        top = ax0
    if yfunc is not None:
        ax0.yaxis.tick_left()
        lims[2:] = yfunc(lims[2:], **yfunc_args)
        right = top.twinx()
        right.axis(lims)
    else:
        right = None
    plt.sca(ax0)
    return top, right


def custom_legend(ax, handles, labels, top_axis=True, **kwargs):
    """
    Add a legend to the axes with options for ``loc='above'``, ``loc='above left'``, and ``loc='above right'``

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes, matplotlib.pyplot.Figure
        Axes or Figure object to which to add the legend
    handles : list of matplotlib.pyplot.Artist
        A list of Artists (lines, patches) to be added to the legend
    labels : list of str
        A list of labels to show next to the handles
    top_axis : bool, optional
        For legends above the top of the plot, add extra padding to the upper x-axis labels. Default: True.
    kwargs
        Keyword arguments to be passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    lgd : matplotlib.legend.Legend
        The Legend object
    """
    loc = kwargs.pop('loc', None)
    bbox_to_anchor = kwargs.pop('bbox_to_anchor', None)
    if top_axis:
        top_of_axis = 1.15
    else:
        top_of_axis = 1.
    if loc is None or loc.lower() == 'none':
        return
    elif loc == 'above':
        loc = 'lower center'
        bbox_to_anchor = (0.5, top_of_axis)
    elif loc == 'above left':
        loc = 'lower left'
        bbox_to_anchor = (0., top_of_axis)
    elif loc == 'above right':
        loc = 'lower right'
        bbox_to_anchor = (1., top_of_axis)
    lgd = ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, **kwargs)
    plt.tight_layout()  # adjusts the top of the axes to make room for 'above' legends
    return lgd


def filter_legend(filts, offset_factor=1.):
    """
    Creates dummy artists and labels for the filter legend using the filter properties

    Parameters
    ----------
    filts : set, list, array-like
        If a list or array of strings, the arrangement of filters in the legend, with columns in the first dimension and
        rows in the second. If a set of :class:`.Filter` objects, they will first be arranged using :func:`.filtsetup`.
    offset_factor : float, optional
        Increase or decrease the filter offsets by a constant factor. Default: 1.

    Returns
    -------
    lines : list of matplotlib.pyplot.Artist
        A list of Artists (lines, patches) to be added to the legend
    labels : list of str
        A list of labels to show next to the handles
    ncol : int
        Number of columns needed for the filter legend
    """
    lines = []
    labels = []
    if isinstance(filts, set):
        filts = filtsetup(filts)
    elif isinstance(filts[0], str) or (isinstance(filts[0], list) and isinstance(filts[0][0], str)):
        filts = np.vectorize(filtdict.get)(filts)
    for filt in filts.flatten():
        if filt is None:
            labels.append('')
            lines.append(Patch(color='none', ec='none'))
        else:
            col = filt.color
            ec = filt.mec
            off = filt.offset * offset_factor
            if not filt.italics:
                labels.append(filt.name)
            elif offset_factor:
                labels.append('${}{:+g}$'.format(filt.name, -off))
            else:
                labels.append('${}$'.format(filt.name))
            lines.append(Patch(fc=col, ec=ec))
    return lines, labels, filts.shape[0]


def filtsetup(filts):
    """
    Arrange filters in a grid according to their system (columns) and offset (rows)

    Parameters
    ----------
    filts : set
        A set of :class:`.Filter` objects to be arranged

    Returns
    -------
    lgnd : numpy.array
        A 2D array of :class:`.Filter` objects
    """
    sysrows = dict()
    for filt in filts:
        if filt.system in sysrows:
            sysrows[filt.system].add(filt.offset)
        else:
            sysrows[filt.system] = {filt.offset}
    syscols = dict()
    rowcols = []
    for sys in list(sysrows.keys()):
        for i, rows in enumerate(rowcols):
            if not rows & sysrows[sys]:
                syscols[sys] = i
                rows |= sysrows[sys]
                break
        else:
            syscols[sys] = len(rowcols)
            rowcols.append(sysrows[sys])
    offs = sorted({filt.offset for filt in filts}, reverse=True)
    lgnd = np.tile(None, (len(rowcols), len(offs)))
    for filt in filts:
        if lgnd[syscols[filt.system], offs.index(filt.offset)] is None:
            lgnd[syscols[filt.system], offs.index(filt.offset)] = filt
        else:
            offind = offs.index(filt.offset) + 1
            offs.insert(offind, filt.offset)
            newrow = np.tile(None, lgnd.shape[0])
            newrow[syscols[filt.system]] = filt
            lgnd = np.insert(lgnd, offind, newrow, 1)
    while lgnd[0, 0] is None:
        lgnd = np.roll(lgnd, 1, axis=0)
    return lgnd


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
    mag = -2.5 * np.log10(flux, out=np.full_like(flux, -np.inf), where=flux > 0.) + zp
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

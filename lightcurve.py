import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from astropy.table import Table, vstack
import filters
import itertools
from matplotlib.markers import MarkerStyle
try:
    from config import markers
except ModuleNotFoundError:
    markers = {}


class Arrow(Path):
    def __init__(self, hx, hy):
        """Create an down-pointing arrow-shaped path.

        *hx* is the half-width of the arrow head (in units of the arrow length).

        *hy* is the height of the arrow head (in units of the arrow length)."""
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
othermarkers = itertools.cycle(MarkerStyle.filled_markers)
usedmarkers = []


class LC(Table):
    def __init__(self, *args, **kwargs):
        Table.__init__(self, *args, **kwargs)
        self.sn = None

    def where(self, **kwargs):
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
        return selected

    def filters_to_objects(self, read_curve=True):
        self['filter'] = [filters.filtdict[f] for f in self['filt']]
        if read_curve:
            for filt in np.unique(self['filter']):
                filt.read_curve()

    def zp(self):
        self.filters_to_objects()
        return np.array([f.m0 for f in self['filter']])

    def calcFlux(self, nondetSigmas=3, zp=None):
        self.nondetSigmas = nondetSigmas
        if zp is None:
            zp = self.zp()
        self['flux'], self['dflux'] = mag2flux(self['mag'], self['dmag'], zp, self['nondet'], self.nondetSigmas)

    def bin(self, delta=0.3, groupby=None):
        if groupby is None:
            groupby = {'filt', 'source'}
        subtabs = []
        self.groupby = list(set(groupby) & set(self.colnames))
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
        return lc

    def findNondet(self, nondetSigmas=3):
        self.nondetSigmas = nondetSigmas
        self['nondet'] = self['flux'] < self.nondetSigmas * self['dflux']

    def calcMag(self, nondetSigmas=3, zp=None):
        self.findNondet(nondetSigmas)
        if zp is None:
            zp = self.zp()
        self['mag'], self['dmag'] = flux2mag(self['flux'], self['dflux'], zp, self['nondet'], self.nondetSigmas)

    def calcAbsMag(self, dm=None, extinction=None, hostext=None):
        if self.sn is None:
            if dm is None:
                dm = 0.
            if extinction is None:
                extinction = {}
            if hostext is None:
                hostext = {}
        else:
            if dm is None:
                dm = self.sn.dm
            if extinction is None:
                extinction = self.sn.extinction
            if hostext is None:
                hostext = self.sn.hostext

        self['absmag'] = self['mag'].data - dm
        for filt, A in extinction.items():
            for filtname in filters.filtdict[filt].names:
                self['absmag'][self['filt'] == filtname] -= A
        for filt, A in hostext.items():
            for filtname in filters.filtdict[filt].names:
                self['absmag'][self['filt'] == filtname] -= A

    def calcLum(self, nondetSigmas=3):
        self.nondetSigmas = nondetSigmas
        self['lum'], self['dlum'] = mag2flux(self['absmag'], self['dmag'], self.zp() + 90.19, self['nondet'],
                                             self.nondetSigmas)

    def findPeak(self, **criteria):
        useforpeak = ~self['nondet']
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
        phase = (self['MJD'].data - self.sn.refmjd) / (1 + self.sn.z)
        self['phase'] = phase

    def plot(self, xcol='phase', ycol='absmag', offset_factor=1, color='filt', marker='source', use_lines=False,
             normalize=False, fillmark=True, **kwargs):
        global markers
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
            filt = filters.filtdict[g['filt'][0]]
            if color == 'filt':
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
                if g[marker][0] not in markers:
                    for nextmarker in othermarkers:
                        if nextmarker not in usedmarkers:
                            markers[g[marker][0]] = nextmarker
                            break
                mark = markers[g[marker][0]]
            elif not marker:
                mark = None
            else:
                mark = marker
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
            if self.sn is None:
                label = None
                linestyle = None
                linewidth = None
            else:
                label = self.sn.name
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

    @classmethod
    def read(cls, filepath, format='ascii.fixed_width', fill_values=None, **kwargs):
        if fill_values is None:
            fill_values = [('--', '0'), ('', '0')]
        t = super(LC, cls).read(filepath, format=format, fill_values=fill_values, **kwargs)
        # make nondetections booleans
        if 'nondet' in t.keys():
            nondets = t['nondet'] == 'True'
            t.remove_column('nondet')
            t['nondet'] = nondets
        else:
            t['nondet'] = False
        # make filters strings (in case they are all the '0' filter)
        if 'filt' in t.colnames:
            filts = np.array(t['filt'], dtype=str)
            t.remove_column('filt')
            t['filt'] = filts
        return t


def flux2mag(flux, dflux=np.array(np.nan), zp=0, nondet=None, nondetSigmas=3):
    flux = flux.copy()
    dflux = dflux.copy()
    if nondet is not None:
        flux[nondet] = nondetSigmas * dflux[nondet]
        dflux[nondet] = np.nan
    mag = -2.5 * np.log10(flux) + zp
    dmag = 2.5 * dflux / (flux * np.log(10))
    return mag, dmag


def mag2flux(mag, dmag=np.nan, zp=0, nondet=None, nondetSigmas=3):
    flux = 10 ** ((zp - mag) / 2.5)
    dflux = np.log(10) / 2.5 * flux * dmag
    if nondet is not None:
        dflux[nondet] = flux[nondet] / nondetSigmas
        flux[nondet] = 0
    return flux, dflux


def binflux(time, flux, dflux, delta=0.2):
    bin_time = []
    bin_flux = []
    bin_dflux = []
    while len(flux) > 0:
        grp = np.array(abs(time - time[0]) <= delta)
        time_grp = time[grp]
        flux_grp = flux[grp]
        dflux_grp = dflux[grp]
        if any(dflux_grp == 0) or any(dflux_grp == 9999) or any(np.isnan(dflux_grp)) or (
                np.ma.is_masked(dflux_grp) and any(dflux_grp.mask)):
            x = np.mean(time_grp)
            y = np.mean(flux_grp)
            z = 0.
        else:
            x = np.mean(time_grp)
            y = np.sum(flux_grp * dflux_grp ** -2) / np.sum(dflux_grp ** -2)
            z = np.sum(dflux_grp ** -2) ** -0.5
        bin_time.append(x)  # indent these lines
        bin_flux.append(y)  # to exlude points
        bin_dflux.append(z)  # with no error
        time = time[~grp]
        flux = flux[~grp]
        dflux = dflux[~grp]
    time = np.array(bin_time)
    flux = np.array(bin_flux)
    dflux = np.array(bin_dflux)
    return time, flux, dflux

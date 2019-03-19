import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
import os
try:
    from config import filters_dir
except ModuleNotFoundError:
    filters_dir = 'filters/'


class Filter:
    def __init__(self, names, color='k', offset=0, system=None, fnu=None, filename='', angstrom=False, linecolor=None,
                 textcolor=None):
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
        self.system = system
        if self.system == 'Johnson':
            self.mec = 'k'
        else:
            self.mec = self.linecolor
        self.offset = offset
        self.system = system
        self.plotstyle = {'color': self.linecolor, 'mfc': self.color, 'mec': self.mec}
        if fnu is not None:
            self.fnu = fnu  # * u.W * u.m**-2 * u.Hz**-1
        elif self.system in ['Gunn', 'ATLAS', 'Gaia']:  # AB magnitudes
            self.fnu = 3.631e-23  # * u.W * u.m**-2 * u.Hz**-1
        else:
            self.fnu = None
        if self.fnu is None:
            self.m0 = np.nan
            self.M0 = np.nan
        else:
            self.m0 = 2.5 * np.log10(self.fnu)
            self.M0 = self.m0 + 90.19
        if filename:
            self.filename = os.path.join(filters_dir, filename)
        else:
            self.filename = ''
        self.angstrom = angstrom
        self.trans = None

    def read_curve(self, show=False, force=False):
        if (self.trans is None or force) and self.filename:
            i = Filter.order.index(self.name) / float(len(Filter.order))
            self.trans = Table.read(self.filename, format='ascii', names=('wl', 'T'))
            if self.angstrom:
                self.trans['wl'] /= 10.
            self.trans['wl'].unit = u.nm
            self.trans.sort('wl')
            self.trans['T'] /= np.max(self.trans['T'])
            self.trans['freq'] = (const.c / self.trans['wl']).to(u.THz)

            dwl = np.trapz(self.trans['T'].quantity, self.trans['wl'].quantity)
            wl_eff = np.trapz(self.trans['T'].quantity * self.trans['wl'].quantity, self.trans['wl'].quantity) / dwl
            left = self.trans[(self.trans['wl'] < wl_eff.value) & (self.trans['T'] > 0.1) * (self.trans['T'] < 0.9)]
            left.sort('T')
            wl0 = np.interp(0.5, left['T'], left['wl'])
            right = self.trans[(self.trans['wl'] > wl_eff.value) & (self.trans['T'] > 0.1) * (self.trans['T'] < 0.9)]
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

            self.freq_eff = freq_eff
            self.dfreq = -dfreq
            self.freq_range = [[freq_eff.value - freq0], [freq1 - freq_eff.value]]

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<filter ' + self.name + '>'

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return Filter.order.index(self.name) == Filter.order.index(other.name)

    def __nq__(self, other):
        if other is None:
            return False
        else:
            return Filter.order.index(self.name) != Filter.order.index(other.name)

    def __lt__(self, other):
        if other is None:
            return False
        else:
            return Filter.order.index(self.name) < Filter.order.index(other.name)

    def __le__(self, other):
        if other is None:
            return False
        else:
            return Filter.order.index(self.name) <= Filter.order.index(other.name)

    def __gt__(self, other):
        if other is None:
            return False
        else:
            return Filter.order.index(self.name) > Filter.order.index(other.name)

    def __ge__(self, other):
        if other is None:
            return False
        else:
            return Filter.order.index(self.name) >= Filter.order.index(other.name)

    def __hash__(self):
        return self.name.__hash__()


def resample_filter_curve(filename, outfile):
    orig = np.loadtxt(filename)
    wl = np.arange(1225., 274., -1.)
    resampled = np.interp(wl, orig[:, 0], orig[:, 1], left=0, right=0)
    output = np.array([wl, resampled]).T
    np.savetxt(outfile, output, fmt=['%.0f', '%.16f'])


all_filters = [
    Filter(['UVW2', 'uvw2', 'W2', '2', 'uw2'], '#FF007F', 8, 'Swift', 7.379e-24, 'Swift_UVOT.UVW2.dat', angstrom=True),
    Filter(['UVM2', 'uvm2', 'M2', 'M', 'um2'], 'm', 8, 'Swift', 7.656e-24, 'Swift_UVOT.UVM2.dat', angstrom=True),
    Filter(['UVW1', 'uvw1', 'W1', '1', 'uw1'], '#7F00FF', 4, 'Swift', 9.036e-24, 'Swift_UVOT.UVW1.dat', angstrom=True),
    Filter(['u', "u'", 'up'], '#4700CC', 1, 'Gunn', filename='sdss-up-183.asci'),  # brightened from '#080017'
    Filter(['us', 's'], '#230047', 1, 'Swift', 1.419e-23, filename='Swift_UVOT.U.dat', angstrom=True),
    Filter('U', '#3C0072', 3, 'Johnson', 1.79e-23, filename='jnsn-uv-183.asci'),
    Filter('B', '#0057FF', 2, 'Johnson', 4.063e-23, filename='jnsn-bu-183.asci'),
    Filter(['b', 'bs'], '#4B00FF', 0, 'Swift', 4.093e-23, filename='Swift_UVOT.B.dat', angstrom=True),
    Filter(['g', "g'", 'gp'], '#00CCFF', 1, 'Gunn', filename='sdss-gp-183.asci'),
    Filter('V', '#79FF00', 0, 'Johnson', 3.636e-23, filename='jnsn-vx-183.asci', textcolor='#46CC00'),
    Filter(['v', 'vs'], '#00FF30', -2, 'Swift', 3.664e-23, filename='Swift_UVOT.V.dat', angstrom=True),
    Filter(['unfilt.', '0', 'Clear', 'C'], 'w', 0, 'Itagaki', 3.631e-23, filename='KAF-1001E.asci', linecolor='k'),
    Filter('G', 'w', 0, 'Gaia', filename='GAIA_GAIA0.G.dat', angstrom=True, linecolor='k'),
    Filter('o', 'orange', -1, 'ATLAS', filename='orange.asci'),
    Filter(['r', "r'", 'rp'], '#FF7D00', -1, 'Gunn', filename='sdss-rp-183.asci'),
    Filter(['R', 'Rc'], '#FF7000', -1, 'Johnson', 3.064e-23, filename='cous-rs-183.asci'),  # '#CC5900'
    Filter(['i', "i'", 'ip'], '#90002C', -2, 'Gunn', filename='sdss-ip-183.asci'),
    Filter(['I', 'Ic'], '#66000B', -2, 'Johnson', 2.416e-23, filename='cous-ic-183.asci'),  # brightened from '#1C0003'
    Filter(['z', "z'", 'Z', 'zs'], '#000000', -1, 'Gunn', filename='pstr-zs-183.asci'),
    Filter('w', '#FFFFFF', -1, 'Johnson', filename='pstr-wx-183.asci'),
    Filter('y', 'y', -2, 'Gunn', filename='pstr-yx-183.asci'),
    Filter('J', '#444444', -2, 'UKIRT', 1.589e-23, filename='Gemini_Flamingos2.J.dat', angstrom=True),
    Filter('H', '#888888', -3, 'UKIRT', 1.021e-23, filename='Gemini_Flamingos2.H.dat', angstrom=True),
    Filter(['K', 'Ks'], '#CCCCCC', -4, 'UKIRT', 0.64e-23, filename='Gemini_Flamingos2.Ks.dat', angstrom=True),
    Filter('L', 'r', -4, 'UKIRT')]
Filter.order = [f.name for f in all_filters]
filtdict = {}
for filt in all_filters:
    for name in filt.names:
        filtdict[name] = filt

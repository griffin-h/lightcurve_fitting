#!/usr/bin/env python

import numpy as np
from .lightcurve import LC
from astropy import constants as const, units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
import argparse
import matplotlib.pyplot as plt
import os
import warnings
import json


def removebadcards(hdr):
    """
    Remove problematic entries from a :class:`astropy.io.fits.header.Header`
    """
    for card in hdr.cards:
        try:
            card.verify('fix')
        except fits.verify.VerifyError as e:
            print(e)
            hdr.remove(card.keyword)
        except Exception as e:
            print(e)
            hdr.remove(card.keyword)
    return hdr


def remove_duplicate_wcs(hdr, keep_number=0):
    """
    Remove duplicate world coordinate system header keywords from a :class:`astropy.io.fits.header.Header`
    """
    for key in ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD2_2', 'CD1_2', 'CD2_1']:
        if key in hdr and hdr.count(key) > 1:
            card = hdr.cards[(key, keep_number)]
            hdr.remove(card.keyword, remove_all=True)
            hdr[card.keyword] = (card.value, card.comment)


def readfitsspec(filename, header=False, ext=None):
    """
    Read a spectrum from a FITS file

    Parameters
    ----------
    filename : str
        Filename from which to read the spectrum
    header : bool, optional
        If True, also return the FITS header
    ext : int, str, optional
        FITS extension number or name in which the spectrum is stored

    Returns
    -------
    wl : array-like
        Wavelengths, typically in ångströms
    flux : array-like
        Observed fluxes in erg / (s cm2 angstrom), if units are identifiable
    hdr : astropy.io.fits.header.Header, optional
        FITS header, returned if ``header=True``
    """
    hdulist = fits.open(filename)
    if ext is None:
        for hdu in hdulist:  # try to find SCI extension
            if hdu.header.get('extname') == 'SCI':
                break
        else:  # if no SCI extension, pick the first extension with data
            for hdu in hdulist:
                if hdu.data is not None:
                    break
            else:
                raise Exception('no extensions have any data')
    else:
        hdu = hdulist[ext]
    data = hdu.data
    hdr = hdu.header
    flux = data.flatten()[:max(data.shape)]
    remove_duplicate_wcs(hdr)  # some problem with Gemini pipeline
    if hdr.get('CUNIT1') in ['Angstroms', 'angstroms', 'deg', 'pixel']:
        hdr['CUNIT1'] = 'Angstrom'  # WCS object needs recognizable units
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(removebadcards(hdr), naxis=1, relax=False, fix=False)
    wl = wcs.wcs_pix2world(np.arange(len(flux)), 0)[0]
    if header:
        return wl, flux, hdr
    else:
        return wl, flux


def convert_spectrum_units(wl, flux, hdr, default_bunit='erg / (Angstrom cm2 s)', default_cunit='Angstrom'):
    """
    Convert a spectrum to standard units, if information is available in the header to establish units (BUNIT & CUNIT1)

    Parameters
    ----------
    wl: array-like
        Input wavelengths
    flux: array-like
        Input fluxes
    hdr: dict-like
        Metadata from which to determine the units of wl and flux
    default_bunit: str, optional
        Units of the output flux. Default: 'erg / (Angstrom cm2 s)'
    default_cunit: str, optional
        Units of the output wavelengths. Default: 'Angstrom'

    Returns
    -------
    wl: numpy.ndarray
        Input wavelengths converted to `default_cunit`
    flux: numpy.ndarray
        Input fluxes converted to `default_bunit`
    """
    bunit = hdr.get('BUNIT', default_bunit)
    if 'Angstrom' not in bunit:
        bunit = bunit.replace('Ang', 'Angstrom')
    if 'Angstrom' not in bunit:
        bunit = bunit.replace('A', 'Angstrom')
    cunit = hdr.get('CUNIT1', default_cunit)
    if cunit.lower() == 'angstroms':
        cunit = cunit.rstrip('s')
    wl = u.Quantity(wl, cunit).to(default_cunit)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flux = u.Quantity(flux, bunit).to(default_bunit, u.equivalencies.spectral_density(wl))
    return wl.value, flux.value


def readOSCspec(filepath):
    """
    Read spectra from a JSON file from the Open Supernova Catalog (https://sne.space)

    Parameters
    ----------
    filepath : str
        Path to the JSON file containing the spectra

    Returns
    -------
    filenames : list
        List of filenames corresponding to each spectrum
    times : list
        List of :class:`astropy.time.Time` when each spectrum was observed
    tel : list
        List of telescope names (if given) where each spectrum was observed
    inst : list
        List of instrument names (if given) with which each spectrum was observed
    wl : list
        List of wavelength arrays for each spectrum
    fx : list
        List of flux arrays for each spectrum
    scales : list
        List of scaling factors for each spectrum (all ones for the OSC)
    """
    with open(filepath) as f:
        json_dict = json.load(f)
    rows = json_dict[os.path.splitext(os.path.basename(filepath))[0]]
    if 'spectra' in rows:
        rows = rows['spectra']
    else:
        return [], [], [], [], [], [], []
    superdict = {key: [d[key] if key in d else '0' for d in rows] for key in
                 set(np.concatenate([list(d.keys()) for d in rows]))}
    times = [Time(float(t), format=u.lower()) for t, u in zip(superdict['time'], superdict['u_time'])]
    wl = [0.1 * np.array(d, dtype=float)[:, 0] for d in superdict['data']]
    fx = [np.array(d, dtype=float)[:, 1] for d in superdict['data']]
    if 'telescope' in superdict:
        tel = superdict['telescope']
    else:
        tel = np.tile('', len(rows))
    if 'instrument' in superdict:
        inst = superdict['instrument']
    else:
        inst = np.tile('', len(rows))
    #    t = Table([superdict['filename'], times, tel, inst, wl, fx, np.ones(len(rows))],
    #               names=['filename', 'date', 'telescope', 'instrument', 'wl', 'flux', 'scale'])
    return superdict['filename'], times, tel, inst, wl, fx, np.ones(len(rows))


def readspec(f, verbose=False):
    """
    Read a spectrum from a FITS or ASCII file and try to identify where and when it was observed

    Parameters
    ----------
    f : str
        Filename from which to read the spectrum
    verbose : bool, optional
        If True, print the date and filename of the spectrum

    Returns
    -------
    x : array-like
        Wavelengths, typically in ångströms
    y : array-like
        Fluxes in erg / (s cm2 angstrom), if units are identifiable
    date : astropy.time.Time
        Time at which the spectrum was observed, if identifiable (otherwise ``None``)
    telescope : str
        Name of the telescope at which the spectrum was observed, if identifiable (otherwise ``''``)
    instrument: str
        Name of the instrument used to observe the spectrum, if identifiable (otherwise ``''``)
    """
    ext = os.path.splitext(f)[1]
    if ext == '.fits':
        x, y, hdr = readfitsspec(f, header=True)
    elif ext == '.json':
        x, y, hdr = readOSCspec(f)
    elif ext in ['.ascii', '.asci', '.flm', '.txt', '.dat']:
        t = Table.read(f, format='ascii')
        # assume it's the first two columns, regardless of if column names are given
        x = t.columns[0]
        y = t.columns[1]
        if 'comments' in t.meta:
            hdr = {kwd.strip(' #'): val.strip(' "\'') for kwd, val in
                   [(line.split('=')[0], '='.join(line.split(' / ')[0].split('=')[1:])) for line in t.meta['comments']
                    if '=' in line]}
            # (anything before first '=', anything between first '=' and ' / ')
        else:
            hdr = {}
    else:
        raise Exception('ext not recognized:', f)

    for kwd in ['MJD-OBS', 'MJD_OBS', 'MJD', 'JD', 'DATE-AVG', 'UTMIDDLE', 'DATE-OBS', 'DATE_BEG', 'UTSHUT']:
        if kwd in hdr and hdr[kwd]:
            if 'MJD' in kwd:
                date = Time(float(hdr[kwd]), format='mjd')
            elif 'JD' in kwd and float(hdr['JD']) > 2400000:
                date = Time(float(hdr[kwd]), format='jd')
            elif 'JD' in kwd:
                date = Time(float(hdr[kwd]) + 2400000, format='jd')
            elif 'T' in hdr[kwd]:
                date = Time(hdr[kwd])
            elif '-' in hdr[kwd]:
                for kwd2 in ['UTMIDDLE', 'EXPSTART', 'UT']:
                    if kwd2 in hdr and type(hdr[kwd2]) == str and ':' in hdr[kwd2]:
                        date = Time(hdr[kwd] + 'T' + hdr[kwd2])
                        break
                    elif kwd2 in hdr:
                        h = int(np.floor(hdr[kwd2]))
                        m = int(np.floor((hdr[kwd2] * 60) % 60))
                        s = int(np.floor((hdr[kwd2] * 3600) % 60))
                        date = Time(hdr[kwd] + 'T{:02d}:{:02d}:{:02d}'.format(h, m, s))
                        break
                else:
                    date = Time(hdr[kwd])
            else:
                continue
            break
    else:  # hope it's in the filename
        import re
        m1 = re.search('24[0-9][0-9][0-9][0-9][0-9]\.[0-9]+', f)  # JD w/1 or more decimals
        m2 = re.search('([12][90][0-9][0-9])-?(0[0-9]|1[0-2])-?(0[1-9]|[12][0-9]|3[01])', f)  # YYYYMMDD
        m_tns = re.search(
            '(19|20)[0-9][0-9]-(0[0-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])_([01][0-9]|2[0-4])-[0-5][0-9]-[0-5][0-9]',
            f)  # YYYY-MM-DD_HH:MM:SS
        m3 = re.search('[0-9][0-9][0-9]d', f)  # integer phase followed by 'd'
        if m1 is not None:
            m = m1.group()
            date = Time(float(m), format='jd')
        elif m2 is not None:
            date = Time('-'.join(m2.groups()))
        elif m_tns is not None:
            m = m_tns.group()
            d, t = m.split('_')
            date = Time(d + 'T' + t.replace('-', ':'))
        elif m3 is not None:
            m = m3.group()
            date = Time(float(m[:-1]), format='mjd')
        else:
            date = None

    if 'TELESCOP' in hdr:
        telescope = hdr['TELESCOP'].strip()
    else:
        telescope = ''
    if 'INSTRUME' in hdr:
        instrument = hdr['INSTRUME'].strip()
    else:
        instrument = ''

    x, y = convert_spectrum_units(x, y, hdr)

    if verbose:
        print(date.isot, f)
    return x, y, date, telescope, instrument


def calibrate_spectra(spectra, lc, filters=None, order=0, subtract_percentile=None, show=False):
    """
    Calibrate a set of spectra to an observed broadband light curve.

    Each calibrated spectrum is saved to a text file that corresponds to the original filename prefixed by ``photcal_``.

    Parameters
    ----------
    spectra : list
        List of filenames containing the spectra
    lc : lightcurve_fitting.lightcurve.LC
        Photometry table containing the observed light curve
    filters : list, optional
        Only use this subset of filters for calibration
    order : int, optional
        Polynomial order for the calibration function. Default: 0 (constant factor)
    subtract_percentile : float, optional
        Subtract flux corresponding to this percentile of the spectrum before calibration. Default: no subtraction
    show : bool, optional
        Plot the observed light curve and the uncalibrated and calibrated spectra, and ask whether to save the results
    """
    if filters is not None:
        lc = lc.where(filt=filters)
    lc.calcFlux()
    lc.sort('MJD')
    filts = set(lc['filter'])

    for filt in filts:
        filt.read_curve()
        filt.trans.sort('freq')

    for spec in spectra:
        wl, flux, time, _, _ = readspec(spec)
        mjd = time.mjd
        if show:
            fig = plt.figure(figsize=(8., 6.))
            ax1 = plt.subplot(211)
            lc.plot(xcol='MJD', ycol='flux', offset_factor=0)
            ax1.axvline(mjd)
            ax1.set_xlabel('MJD')
            ax1.set_ylabel('$F_\\nu$ (W Hz$^{-1}$)')
            ax2 = plt.subplot(212)
        good = ~np.isnan(flux)
        wl = wl[good] * u.angstrom
        Flam = flux[good] * u.erg / u.s / u.angstrom / u.cm**2
        nu = const.c / wl
        Fnu = (Flam * wl / nu).to(u.W / u.Hz / u.m**2).value[::-1]
        nu = nu.to(u.THz).value[::-1]
        if subtract_percentile is not None:
            Fnu -= np.nanpercentile(Fnu, subtract_percentile)
        freqs = []
        ratios = []
        for filt in filts:
            freq0 = filt.freq_eff.value - filt.freq_range[0]
            freq1 = filt.freq_range[1] + filt.freq_eff.value
            if freq1 < np.min(nu) or freq0 > np.max(nu):
                print(filt, "and spectrum don't overlap")
                continue  # filter and spectrum don't overlap
            lc_filt = lc.where(filter=filt, nondet=False)
            if len(lc_filt) == 0 or mjd - np.max(lc_filt['MJD']) > 1. or mjd < np.min(lc_filt['MJD']):
                print(filt, "not observed before and after spectrum")
                continue
            flux_lc = np.interp(mjd, lc_filt['MJD'], lc_filt['flux'])
            trans_interp = np.interp(nu, filt.trans['freq'], filt.trans['T_norm_per_freq'])
            flux_spec = np.trapz(Fnu * trans_interp, nu) / np.trapz(trans_interp, nu)
            ratio = flux_lc / flux_spec
            if show:
                ax2.axvspan(freq0, freq1, color=filt.color, alpha=0.2)
                ax2.plot(filt.freq_eff, flux_lc, marker='o', color=filt.color, zorder=5)
            ratios.append(ratio)
            freqs.append(filt.freq_eff.value)
        if not ratios:
            print('no filters for', spec)
            plt.close(fig)
            continue
        scale = np.mean(ratios)
        if order:
            p = np.polyfit(freqs, np.array(ratios) / scale, order)
            corr = np.polyval(p, nu) * scale
            print(spec, scale, p[:-1])
        else:
            corr = scale
            print(spec, scale)
        if show:
            ax2.plot(nu, Fnu * scale, label='rescaled')
            ax2.set_xlabel('Frequency (THz)')
            ax2.set_ylabel('$F_\\nu$ (W Hz$^{-1}$)')
            if order:
                ax2.plot(nu, Fnu * corr, color='C2', label='rescaled & warped')
                plt.legend(loc='best')
            plt.show()
            ans = input('accept this scale? [Y/n] ')
        if not show or ans.lower() != 'n':
            data_out = np.array([wl[good], flux[good] * corr]).T
            path_in, filename_in = os.path.split(spec)
            filename_out = os.path.join(path_in, 'photcal_' + filename_in).replace('.fits', '.txt')
            np.savetxt(filename_out, data_out, fmt='%.1f %.2e')
            print(filename_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate spectra of a supernova to photometry.')
    parser.add_argument('spectra', nargs='+', help='filenames of spectra')
    parser.add_argument('--lc', help='filename of photometry table (must have columns "MJD", "filt", "mag"/"flux", and'
                                     '"dmag"/"dflux")')
    parser.add_argument('--lc-format', default='ascii',
                        help='format of photometry table (passed to :func:`astropy.table.Table.read`)')
    parser.add_argument('-f', '--filters', nargs='+', help='filters to use for calibration')
    parser.add_argument('-o', '--order', type=int, default=0, help='polynomial order of correction function')
    parser.add_argument('--subtract-percentile', type=float, help='subtract continuum from spectrum before correcting')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    lc = LC.read(args.lc, format=args.format)
    calibrate_spectra(args.spectra, lc, args.filters, args.order, args.subtract_percentile, args.show)

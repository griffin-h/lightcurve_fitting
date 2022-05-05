=====
Usage
=====

Light Curves
------------
The core of this package is the ``lightcurve.LC`` object. This is an extension of the Astropy ``Table`` object, which contains
tabular data, in this case broadband photometric observations. You can read light curve data from a file using the
standard ``Table.read()`` method. I'm going to read an example light curve of SN 2016bkv to be used from here on:

.. code-block:: python

    from lightcurve_fitting.lightcurve import LC
    from pkg_resources import resource_filename

    filename = resource_filename('lightcurve_fitting', 'example/SN2016bkv.txt')
    lc = LC.read(filename)
    print(lc)

The following column names are used by the package, although the light curve can have extra columns
[alternative column names are given in square brackets]:

 * MJD (required): modified Julian date of the observation [mjd, JD, jd (JD/jd are converted)]
 * mag (required): magnitude of the observation [Magnitude, Mag, ab_mag, PSFmag, MAG, omag, magnitude, apparent_mag]
 * dmag (required): uncertainty on the magnitude [Magnitude_Error, magerr, MagErr, mag_err, e_mag, Error, err, PSFerr,
   MAGERR, e_omag, e_magnitude, apparent_mag_err, Mag_Err]
 * filt (required): name of the filter [filter, Filter, band, FLT, Band]
 * filter (automatic): the filter object (see :ref:`Filters` below)
 * nondet: True if the magnitude is an upper limit, False otherwise [Is_Limit, UL, l_omag, upper_limit, upperlimit]
 * flux: the spectral flux density (:math:`F_ν`, arbitrary units) of the observation [FLUXCAL]
 * dflux: uncertainty on the flux [FLUXCALERR]
 * phase: time since a reference date (e.g., peak or explosion) in rest-frame days [Phase, PHASE]
 * absmag: absolute magnitude of the observation
 * lum: the spectral luminosity density (:math:`L_ν`, in watts/hertz) of the observation
 * dlum: the uncertainty on the spectral luminosity density
 * telescope: the name of the telescope/instrument where this observation was carried out [Telescope, Tel, tel+inst]
 * source: the data source, either a telescope/instrument name or a literature reference [Source]

The ``LC.meta`` attribute contains information needed to calculate absolute magnitudes and luminosities:

 * dm: the distance modulus
 * extinction: a dictionary containing Milky Way extinction corrections for each filter
 * hostext: a dictionary containing host galaxy extinction corrections for each filter

.. code-block:: python

    lc.meta['dm'] = 30.79
    lc.meta['extinction'] = {
     'U': 0.069,
     'B': 0.061,
     'g': 0.055,
     'V': 0.045,
     '0': 0.035,
     'r': 0.038,
     'R': 0.035,
     'i': 0.028,
     'I': 0.025,
    }

The ``LC`` object has several methods for converting between the columns above (see API Documentation)
as well as a method for plotting the light curve in a single command:

.. code-block:: python

    lc.calcAbsMag()
    lc.plot(xcol='MJD')

Filters
-------
The ``filters`` submodule defines a ``Filter`` object that stores information about the broadband filters: transmission
function, photometric system, and styles for plotting. You mostly won't have to touch this module, unless you are
adding new filters.

Bolometric Light Curves
-----------------------
You can make a bolometric light curve and color curves from the photometry table with the ``bolometric`` module.

.. code-block:: python

    from lightcurve_fitting.bolometric import calculate_bolometric, plot_bolometric_results, plot_color_curves

    redshift = 0.002
    outpath = '/Users/griffin/Desktop/SN2016bkv_bolometric'
    t = calculate_bolometric(lc, redshift, outpath, colors=['B-V', 'g-r', 'r-i'])
    print(t)
    plot_bolometric_results(t)
    plot_color_curves(t)

The light curve is divided into epochs (defined by the ``bin`` and ``also_group_by`` arguments to ``calculate_bolometric``), and processed four different ways:

 * Fitting the Planck function using ``scipy.curve_fit``. This is very fast but may not give reliable uncertainties.
   The columns ``temp``, ``radius``, ``dtemp``, and ``dradius`` come from this fit.
 * The Stefan-Bolzmann law gives the total bolometric luminosity, ``lum`` and ``dlum``.
 * Integrating the Planck function between :math:`U` and :math:`I` band (observed) gives ``L_opt``.
 * Fitting the Planck function using an MCMC routine.
   This is slower, depending on how many walkers (``nwalkers``) and steps (``burnin_steps`` and ``steps``) you use,
   but gives more robust uncertainties.
   The columns ``temp_mcmc``, ``radius_mcmc``, ``dtemp0``, ``dtemp1``, ``dradius0``, ``dradius1`` come from this fit.
   My convention for non-Gaussian uncertainties is that 0 is the lower uncertainty and 1 is the upper uncertainty.
 * Integrating the Planck function between :math:`U` and :math:`I` band (observed) gives
   ``L_mcmc``, ``dL_mcmc0``, and ``dL_mcmc1``.
 * Directly integrating the observed SED, assuming 0 flux outside of :math:`U` to :math:`I`.
   Use this if you do not want to assume the SED is a blackbody. This yields the column ``L_int``.

The MCMC routine saves a corner plot for each fit in the folder you specify (``outpath``).
I highly recommend looking through these to make sure the fits converged.
If they didn't, try adjusting the number of burn-in steps (``burnin_steps``).
To save the table, give ``save_table_as='filename.table'`` as an argument to ``calculate_bolometric``.
To save the plot, give ``save_plot_as='filename.pdf'`` as an argument to ``plot_bolometric_results``.

Beware of the units I'm using:

 * Temperatures are in kilokelvins (kK).
 * Radii are in thousands of solar radii (:math:`1000R_\odot`).
 * Luminosities are in watts (W). :math:`1\,\mathrm{W} = 10^7\,\mathrm{erg}\,\mathrm{s}^{-1}`

Optionally, you can calculate colors at each epoch by giving the argument ``colors`` to ``calculate_bolometric``). These get saved in the same output table in four columns per color, e.g., for :math:`B-V`:

 * the color itself, ``B-V``,
 * the uncertainty on the color, ``d(B-V)``,
 * whether the color is a lower limit, ``lolims(B-V)`` (i.e., :math:`B` was an upper limit), and
 * whether the color is an upper limit, ``uplims(B-V)`` (i.e., :math:`V` was an upper limit).

Intrinsic Scatter
^^^^^^^^^^^^^^^^^

You can include an intrinsic scatter term (:math:`\sigma`) in your MCMC fits by setting ``use_sigma=True``. :math:`\sigma` is added in quadrature to the photometric uncertainty on each point (:math:`\sigma_i`). If you choose ``sigma_type='relative'``, :math:`\sigma` will be in units of the individual photometric uncertainties, i.e.,

.. math::
    \sigma_{i,\mathrm{eff}} = \sqrt{ \sigma_i^2 + \left( \sigma * \sigma_i \right)^2 }

If you choose ``sigma_type='absolute'``, :math:`\sigma` will be in units of the median photometric uncertainty (:math:`\bar\sigma`), i.e.,

.. math::
    \sigma_{i,\mathrm{eff}} = \sqrt{ \sigma_i^2 + \left( \sigma * \bar{\sigma} \right)^2 }

For bolometric light curve fitting, you can also set a maximum for this intrinsic scatter using the ``sigma_max`` keyword (default: 10). (For model fitting, you can set a maximum using the ``priors`` keyword.)

Model Fitting
-------------
The ``models`` and ``fitting`` submodules allow you to fit analytical models to the observed data. Right now, the only choices are:

 * ``CompanionShocking``, which is the SiFTO Type Ia supernova template (Conley et al. `2008 <https://doi.org/10.1086/588518>`_) plus a shock component from Kasen (`2010 <https://doi.org/10.1088/0004-637X/708/2/1025>`_), with factors on the r and i SiFTO models and a factor on the U shock component.
   This was used in my paper on SN 2017cbv: https://doi.org/10.3847/2041-8213/aa8402.
 * ``CompanionShocking2``, which is the same SiFTO Type Ia supernova template plus a shock component, but with time offsets for the U and i SiFTO models instead of the three multiplicative factors.
   This was used in my paper on SN 2021aefx (submitted).
 * ``ShockCooling``, which is the Sapir & Waxman (`2017 <https://doi.org/10.3847/1538-4357/aa64df>`_) model for shock cooling in a core-collapse supernova,
   formulated in terms of :math:`v_s, M_\mathrm{env}, f_ρ M, R`
 * ``ShockCooling2``, which is the same Sapir & Waxman model but formulated in terms of scaling parameters :math:`T_1, L_1, t_\mathrm{tr}`.
   This was used in my paper on SN 2016bkv: https://doi.org/10.3847/1538-4357/aac5f6.
 * ``ShockCooling3``, which is the same as ``ShockCooling`` but with :math:`d_L` and :math:`E(B-V)` as free parameters. (Therefore it fits the flux instead of the luminosity.) This was used in my paper on SN 2021yja (submitted).

**Note on the shock cooling models:**
There are degeneracies between many of the physical parameters that make them difficult to fit independently.
This led us to fit develop the ``ShockCooling2`` model just to see if the model could fit the data at all.
Since it did not fit well, we concluded that the physical parameters we could have obtained by fitting the ``ShockCooling`` model were irrelevant.
However, in order to measure, for example, the progenitor radius, one must use the ``ShockCooling`` model.


.. code-block:: python

    from lightcurve_fitting.models import ShockCooling2, UniformPrior
    from lightcurve_fitting.fitting import lightcurve_mcmc, lightcurve_corner

    # Fit only the early light curve
    lc_early = lc.where(MJD_min=57468., MJD_max=57485.)

    # Define the priors and initial guesses
    priors = [
        UniformPrior(0., 100.),
        UniformPrior(0., 100.),
        UniformPrior(0., 100.),
        UniformPrior(57468., 57468.7),
    ]
    p_lo = [20., 2., 20., 57468.5]
    p_up = [50., 5., 50., 57468.7]

    redshift = 0.002

    sampler = fitting.lightcurve_mcmc(lc_early, ShockCooling2, model_kwargs={'z': redshift},
                                      priors=priors, p_lo=p_lo, p_up=p_up,
                                      nwalkers=10, nsteps=100, nsteps_burnin=100, show=True)
    lightcurve_corner(lc_early, ShockCooling2, sampler.flatchain, model_kwargs={'z': redshift})

**Another note on the shock cooling models:**
The shock cooling models are only valid for temperatures above 0.7 eV = 8120 K (Sapir & Waxman 2017),
so you should check that you have not included observations where the model goes below that.
If you have, you should rerun the fit without those points.
If you used the Rabinak & Waxman option, the model fails even earlier, but you will have to check that manually.

.. code-block:: python

    p_mean = sampler.flatchain.mean(axis=0)
    t_max = ShockCooling2.t_max(p_mean)
    print(t_max)
    if lc_early['MJD'].max() > t_max:
        print('Warning: your model is not valid for all your observations')

Note that you can add an :ref:`Intrinsic Scatter` to your model fits as well.

Calibrating Spectra to Photometry
---------------------------------
The ``speccal`` module (somewhat experimental right now) can be used to calibrate spectra to observed photometry.

.. code-block:: python

    from lightcurve_fitting.speccal import calibrate_spectra

    spectra_filenames = ['blah.fits', 'blah.txt', 'blah.dat']
    calibrate_spectra(spectra_filenames, lc, show=True)

Each spectrum is multiplied by the filter transmission function and integrated to produce a synthetic flux measurement.
Each magnitude in the light curve is also converted to flux.
The ratios of these two flux measurements (for each filter) are fit with a polynomial (order 0 by default).
Multiplying by this best-fit polynomial calibrates the spectrum to the photometry.
Each calibrated spectrum is saved to a text file with the prefix ``photcal_``.
I recommend using ``show=True`` to visualize the process.
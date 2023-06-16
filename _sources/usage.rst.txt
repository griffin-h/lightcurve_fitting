=====
Usage
=====

Light Curves
------------
The core of this package is the :class:`.lightcurve.LC` object. This is an extension of the :class:`astropy.table.Table` object, which contains
tabular data, in this case broadband photometric observations. You can read light curve data from a file using the
standard :meth:`astropy.table.Table.read` method. I'm going to read an example light curve of SN 2016bkv to be used from here on:

.. code-block:: python

    from lightcurve_fitting.lightcurve import LC

    filename = 'example/SN2016bkv.txt'
    lc = LC.read(filename)
    print(lc)

The following column names are used by the package, although the light curve can have extra columns
[alternative column names are given in square brackets]:

 * MJD (required): modified Julian date of the observation [mjd, JD, jd (JD/jd are converted)]
 * mag (required): magnitude of the observation [Magnitude, Mag, ab_mag, PSFmag, MAG, omag, magnitude, apparent_mag]
 * dmag (required): uncertainty on the magnitude [Magnitude_Error, magerr, MagErr, mag_err, e_mag, Error, err, PSFerr,
   MAGERR, e_omag, e_magnitude, apparent_mag_err, Mag_Err, emag, error]
 * filter (required): name of the filter [filter, filt Filter, band, FLT, Band] (see :ref:`Filters` below)
 * nondet: True if the magnitude is an upper limit, False otherwise [Is_Limit, UL, l_omag, upper_limit, upperlimit]
 * flux: the spectral flux density (:math:`F_ν`, arbitrary units) of the observation [FLUXCAL]
 * dflux: uncertainty on the flux [FLUXCALERR]
 * phase: time since a reference date (e.g., peak or explosion) in rest-frame days [Phase, PHASE]
 * absmag: absolute magnitude of the observation
 * lum: the spectral luminosity density (:math:`L_ν`, in watts/hertz) of the observation
 * dlum: the uncertainty on the spectral luminosity density
 * telescope: the name of the telescope/instrument where this observation was carried out [Telescope, Tel, tel+inst]
 * source: the data source, either a telescope/instrument name or a literature reference [Source]

The :attr:`LC.meta` attribute contains information needed to calculate absolute magnitudes, luminosities, and rest-frame phases:

 * dm: the distance modulus
 * ebv: the selective extinction due to dust in the Milky Way
 * host_ebv: the selective extinction due to dust in the host galaxy (applied at the target redshift)
 * redshift: the redshift (also used to calculate distance if the distance modulus is not given)

.. code-block:: python

    lc.meta['dm'] = 30.79
    lc.meta['ebv'] = 0.016
    lc.meta['host_ebv'] = 0.
    lc.meta['redshift'] = 0.002

The :class:`.LC` object has several methods for converting between the columns above,
as well as a method for plotting the light curve in a single command:

.. code-block:: python

    lc.calcAbsMag()
    lc.calcPhase()
    lc.plot()

Filters
-------
The :mod:`.filters` submodule defines a :class:`.Filter` object that stores information about the broadband filters: transmission function, photometric system, and styles for plotting.
You mostly won't have to touch this module, unless you are adding or modifying filters.

The ``'filter'`` column in a :class:`.LC` object contains :class:`.Filter` objects, rather than strings.
However, you can use filter names directly in most places, including the :meth:`.LC.where` method, and they will be parsed into :class:`.Filter` objects.
For example, ``lc.where(filter='r')`` will return photometry points in bands labeled both 'r' and 'rp' in your input file.

If you ever need direct access to the :class:`.Filter` objects by name, you can use the filter lookup dictionary.

.. code-block:: python

    from lightcurve_fitting.filters import filtdict

    g = filtdict['g']
    print(g)

Bolometric Light Curves
-----------------------
You can make a bolometric light curve and color curves from the photometry table with the :mod:`.bolometric` module.

.. code-block:: python

    from lightcurve_fitting.bolometric import calculate_bolometric, plot_bolometric_results, plot_color_curves

    t = calculate_bolometric(lc, outpath='./SN2016bkv_bolometric', colors=['B-V', 'g-r', 'r-i'])
    print(t)
    plot_bolometric_results(t)
    plot_color_curves(t)

The light curve is divided into epochs (defined by the ``bin`` and ``also_group_by`` arguments to :func:`.calculate_bolometric`), and processed four different ways:

 * Fitting the Planck function using :func:`scipy.optimize.curve_fit`. This is very fast but may not give reliable uncertainties.
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
To save the table, give ``save_table_as='filename.table'`` as an argument to :func:`.calculate_bolometric`.
To save the plot, give ``save_plot_as='filename.pdf'`` as an argument to :func:`.plot_bolometric_results`.

Beware of the units I'm using:

 * Temperatures are in kilokelvins (kK).
 * Radii are in thousands of solar radii (:math:`1000R_\odot`).
 * Luminosities are in watts (W). :math:`1\,\mathrm{W} = 10^7\,\mathrm{erg}\,\mathrm{s}^{-1}`

Optionally, you can calculate colors at each epoch by giving the argument ``colors`` to :func:`.calculate_bolometric`. These get saved in the same output table in four columns per color, e.g., for :math:`B-V`:

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
The :mod:`.models` and :mod:`.fitting` submodules allow you to fit analytical models to the observed data.
Right now, there are three classes of models: :class:`.BaseCompanionShocking`, which is the SiFTO Type Ia supernova template [C08]_ plus a shock component from [K10]_; :class:`.BaseShockCooling`, which is the [SW17]_ model for shock cooling in a core-collapse supernova; and :class:`.ShockCooling4`, which is the updated shock cooling model of [MSW23]_.
The variations on these classes are as follows:

 * :class:`.CompanionShocking` uses factors on the r and i SiFTO models and a factor on the U shock component.
   This was used in my paper on SN 2017cbv [H17]_.
 * :class:`.CompanionShocking2` uses time offsets for the U and i SiFTO models.
   This was used in my paper on SN 2021aefx [H22a]_.
 * :class:`.CompanionShocking3` is the same as :class:`.CompanionShocking2` but includes viewing angle dependence.
   This was used in my paper on SN 2023bee [H23a]_.
 * :class:`.ShockCooling` is formulated in terms of physical parameters :math:`v_s, M_\mathrm{env}, f_ρ M, R`.
 * :class:`.ShockCooling2` is formulated in terms of scaling parameters :math:`T_1, L_1, t_\mathrm{tr}`.
   This was used in my paper on SN 2016bkv [H18]_.
 * :class:`.ShockCooling3` is the same as :class:`.ShockCooling` but with :math:`d_L` and :math:`E(B-V)` as free parameters. (Therefore it fits the flux instead of the luminosity.)
   This was used in my paper on SN 2021yja [H22b]_.
 * :class:`.ShockCooling4` is the updated shock cooling model of [MSW23]_.
   This was used in my paper on SN 2023ixf [H23b]_.

**Note on the shock cooling models:**
There are degeneracies between many of the physical parameters that make them difficult to fit independently.
This led us to fit develop the :data:`.ShockCooling2` model just to see if the model could fit the data at all.
Since it did not fit well, we concluded that the physical parameters we could have obtained by fitting the :data:`.ShockCooling` model were irrelevant.
However, in order to measure, for example, the progenitor radius, one must use the :data:`.ShockCooling` model.


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

    # Initialize the model
    model = ShockCooling2(lc_early)

    # Run the fit
    sampler = lightcurve_mcmc(lc_early, model, priors=priors, p_lo=p_lo, p_up=p_up,
                              nwalkers=10, nsteps=100, nsteps_burnin=100, show=True)

    # Plot the results
    fig, ax_corner, ax_model = lightcurve_corner(lc_early, model, sampler.flatchain)

**Another note on the shock cooling models:**
The shock cooling models are only valid for temperatures above 0.7 eV = 8120 K [SW17]_,
so you should check that you have not included observations where the model goes below that.
If you have, you should rerun the fit without those points.
If you used the [RW11]_ option, the model fails even earlier, but you will have to check that manually.

.. code-block:: python

    p_mean = sampler.flatchain.mean(axis=0)
    t_max = model.t_max(p_mean)
    print(t_max)
    if lc_early['MJD'].max() > t_max:
        print('Warning: your model is not valid for all your observations')

Note that you can add an :ref:`Intrinsic Scatter` to your model fits as well.

Defining New Models (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to define a new model, all you need to do is subclass the :class:`.Model` class.
Implement the model in the :meth:`.Model.evaluate` method, which takes an array of times and an array of filters as the first two arguments, followed by the physical parameters of the model.
If there are keyword arguments (parameters that are *not* fit for) that need to be specified, you may have to override the :meth:`.Model.__init__` method.
You must also provide ``input_names`` and ``units`` as class variables.

Calibrating Spectra to Photometry
---------------------------------
The :mod:`.speccal` module (somewhat experimental right now) can be used to calibrate spectra to observed photometry.

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

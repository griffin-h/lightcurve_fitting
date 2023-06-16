===============
Release History
===============

v0.9.0 (2023-06-16)
-------------------
* Add :class:`.ShockCooling4` model from [MSW23]_
* Add ``filters_to_model`` option to :func:`.lightcurve_corner` and :func:`.lightcurve_model_plot`
* Propagate ``t0_offset`` option from :func:`.lightcurve_corner` to ``mjd_offset`` option of :func:`.lightcurve_model_plot`
* Recognize filters RGB, LRGB, and z_s
* Assume an empty/masked value for filter means "unfiltered"
* Use colors in marker legend if ``color`` = ``marker`` in :meth:`.LC.plot`
* Add option to plot phase in hours in :meth:`.LC.plot`
* Add option to return axes objects in :meth:`.LC.plot`

v0.8.0 (2023-04-27)
-------------------
This release includes a major refactor of the :mod:`models` module. Models now require initialization with the :class:`.LC` object to be fit. Updated examples are given in the documentation and the Jupyter notebook. This alleviates the need for model keyword arguments, which are now deprecated. (A warning will be issued if the user tries to supply a ``model_kwargs`` argument.) This also allows for much easier specification of new models by subclassing the existing models.

* Add :class:`.CompanionShocking3` model, which includes viewing angle dependence
* Added :meth:`.BaseCompanionShocking.t_min` and :meth:`.BaseCompanionShocking.t_max` to indicate when the SiFTO model is computed
* Require Astropy version 5 (there were already features that required this)
* Automatically calculate extinction based on :math:`E(B-V)` and :math:`R_V` if :math:`A_V` are not given
* Automatically parse filters even when :class:`.LC` is initialized without reading (removes need for separate "filt" column)
* Parse filters in :meth:`.LC.where` (removes need for user to access ``filtdict``)
* Add options to add auxiliary axes and fancy legends to light curve plots
* Avoid warnings/NaN when models are evaluated at negative phases
* Update :func:`.calculate_bolometric` to use redshift from :class:`.LC` object. A deprecation warning is issued if the ``z`` parameter is used, although it will still work for now.
* Bug fix in :class:`.ShockCooling3` when :math:`R_V \ne 3.1`
* Read the filter curves on the fly when accessing :class:`.Filter` properties ``trans``, ``wl_eff``, ``dwl``, ``wl_range``, ``freq_eff``, ``dfreq``, or ``freq_range``
* Finish removing deprecated way of storing supernova properties in :class:`Supernova` object
* Allow :class:`.LC` to be written as ECSV, FITS, and HDF5 files by converting :class:`.Filter` to strings
* Add a few more JWST filters
* Return the figure object in :func:`.calibrate_spectra`
* Fix bug in :func:`.calibrate_spectra` that reverses the correction when warping
* Reimplement :func:`.lightcurve_model_plot` using :meth:`.LC.plot` to get markers, nondetections, offsets, etc.
* Better matching of filters to SiFTO template (not just by name)
* Interpolate SiFTO with a cubic spline instead of a line
* Recognize uprime, gprime, rprime, iprime, and zprime as filter names
* Allow adjustments to marker size
* Prevent log(0) warning from :func:`lightcurve_model_plot`
* Automatically plot SiFTO model in dashed lines on :func:`lightcurve_model_plot`
* Add option for logarithmic x-axis in :func:`lightcurve_model_plot`
* Add offset to peak time (in addition to explosion time) in :func:`lightcurve_corner`
* Fix bug in :meth:`.Filter.wl_range` and :meth:`.Filter.freq_range` when filter curve has non-smooth features
* Move likelihood function to be a method: :meth:`.Model.log_likelihood`

v0.7.0 (2022-10-25)
-------------------
* Generalize :meth:`Filter.blackbody` to :meth:`.Filter.synthesize` and :func:`blackbody_mcmc` to :func:`.spectrum_mcmc`
* Allow for arbitrary priors in bolometric light curve fitting (see note at :ref:`v0.5.0 <v050>`)
* Add ability to plot :class:`.LC` data vs. filter effective wavelength (SED)
* Add JWST filters
* Raise an error if the initial parameter guesses are outside the prior
* Add convenience function for preparing spectra to upload to WISeREP
* Allow use of :meth:`.LC.findPeak` without :class:`Supernova` object
* Fix minor bug in rounding to display a given number of significant figures
* Minor change to :meth:`.Filter.spectrum` to avoid inadvertently re-sorting transmission tables
* When calibrating spectra to photometry, assume constant flux in a filter for a configurable amount of time after the last observed point
* Plot SED over configurable range in :func:`.spectrum_mcmc`
* Reoptimize SED corner plot for any number of parameters, and save as PDF instead of PNG
* Refactor SED corner plots into its own function: :func:`.spectrum_corner`

v0.6.0 (2022-05-04)
-------------------
* Add :class:`.CompanionShocking2` model: similar to :class:`.CompanionShocking` model but with time shifts on U and i SiFTO tempates instead of the three multipicative factors
* Separate out the :func:`.lightcurve_model_plot` function to allow plotting only the observed vs. model light curves (the inset from :func:`.lightcurve_corner`)
* Add the :meth:`.Filter.spectrum` method to calculate synthetic photometry on an arbitrary spectrum
* Skip initial state check for post-burn-in MCMC (so it doesn't crash half way through the fit)
* Treat the DLT40 filter as r when fitting the SiFTO model
* Minor changes to plot formatting (remove trailing zeros)
* Add missing docstring to :func:`.shock_cooling3`

.. _v050:

v0.5.0 (2022-03-16)
-------------------
For the first time, this release introduces a change that is not backward compatible.
To enable the use of Gaussian priors, I have had to make the prior specification a little more complex.
Instead of using ``p_min`` and ``p_max`` to specify the bounds on a uniform prior, users will have to define the shape and bounds on each prior using the ``priors`` keyword.
This takes a list of :class:`.Prior` objects, e.g., :class:`.models.UniformPrior`, :class:`.models.LogUniformPrior`, or :class:`.models.GaussianPrior`.
See the updated example in :ref:`Model Fitting`.
For now, the code will still work if you use ``p_min`` and ``p_max``, but a warning will be issued to encourage you to switch.

* Add intrinsic scatter option to bolometric light curve creation
* Add more MJD digits in bolometric output files
* Add option to consider other columns when dividing light curves into epochs
* Recognize spectra stored as FITS tables
* Don't crash when plotting ungrouped light curve
* Allow linewidth/linestyle to be passed as ``plot_kwargs``
* Recognize ``marker='none'`` when plotting a light curve
* Do not plot black lines for Johnson filters when using ``plot_lines``
* Allow adjustment of font sizes in light curve corner plots
* Change priors from functions to classes (see above)
* Allow for a reddened blackbody SED in models
* Add :class:`.ShockCooling3` model: same as :class:`.ShockCooling` but with :math:`d_L` and :math:`E(B-V)` as free parameters
* Add option to make sigma an absolute intrinsic scatter

v0.4.0 (2022-02-08)
-------------------
* Fix bug in min/max validity times when using intrinsic scatter parameter
* Change prior on blackbody temperature from log-uniform to uniform
* Don't italicize some filter names
* Return axes objects in light curve corner plot
* Give option to plot magnitudes in light curve corner plot
* Fix plotting of wavelength when units are supplied
* Add option to calculate phase in rest-frame hours
* Issue warning when filters do not receive extinction correction
* Switch from to generic filter curves from the SVO Filter Profile Service where possible
* Add progress bars for MCMC fitting
* Add option to save chain plots. Burn-in and sampling plots are combined into the same figure.
* Add option to save chain in bolometric light curve fitting

v0.3.0 (2021-09-22)
-------------------
* Switch the default table format from ``'ascii.fixed_width'`` to just ``'ascii'``
* Add more recognized column names for light curves
* Add more recognized filter names, including an "unknown" filter
* Add option to include intrinsic scatter in model fitting
* Do not require ``'nondet'`` or ``'source'`` columns
* Improve handling of units in spectra files
* Include automatic axis labels and filter legend in light curve plot
* Make bolometric module compatible with numpy 1.20
* Allow :meth:`.LC.calcPhase` to function without a :class:`.Supernova` object
* Allow color curves to be plotted against phase (in addition to MJD)

v0.2.0 (2020-12-08)
-------------------
* Recognize several other names for LC columns (e.g., "filter" for "filt")
* When binning a light curve, if one point has no uncertainty, ignore only that point
* Recognize "Swift+UVOT" as a telescope (in addition to "Swift/UVOT")
* Recognize the full names of the ATLAS cyan and orange filters
* Fix bug causing a crash when some photometry points are missing a filter
* Fix bug in recognizing wavelength unit for spectra when "Angstrom" is spelled out

v0.1.0 (2020-06-25)
-------------------
Initial release on PyPI.

v0.0.0 (2019-04-14)
-------------------
Initial release on GitHub and Zenodo.

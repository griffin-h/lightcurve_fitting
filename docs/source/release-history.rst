===============
Release History
===============

v0.6.0 (2022-05-04)
-------------------
* Add CompanionShocking2 model: similar to CompanionShocking model but with time shifts on U and i SiFTO tempates instead of the three multipicative factors
* Separate out the :func:`lightcurve_model_plot` function to allow plotting only the observed vs. model light curves (the inset from :func:`lightcurve_corner`)
* Add the :func:`Filter.spectrum` method to calculate synthetic photometry on an arbitrary spectrum
* Skip initial state check for post-burn-in MCMC (so it doesn't crash half way through the fit)
* Treat the DLT40 filter as r when fitting the SiFTO model
* Minor changes to plot formatting (remove trailing zeros)
* Add missing docstring to :func:`shock_cooling3`

v0.5.0 (2022-03-16)
-------------------
For the first time, this release introduces a change that is not backward compatible.
To enable the use of Gaussian priors, I have had to make the prior specification a little more complex.
Instead of using ``p_min`` and ``p_max`` to specify the bounds on a uniform prior, users will have to define the shape and bounds on each prior using the ``priors`` keyword.
This takes a list of ``Prior`` objects, e.g., ``models.UniformPrior``, ``models.LogUniformPrior``, or ``models.GaussianPrior``.
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
* Add ShockCooling3 model: same as ShockCooling but with :math:`d_L` and :math:`E(B-V)` as free parameters
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
* Allow ``calcPhase`` to function without a ``Supernova`` object
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
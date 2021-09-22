===============
Release History
===============

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
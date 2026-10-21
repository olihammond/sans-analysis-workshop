# Fitting helper tests

These tests cover the small `sans_fitter` package bundled with the workshop.
The package is supporting infrastructure for the notebooks rather than a
separate end-user project.

Run the suite from the repository root:

```bash
python -m pytest
```

The tests exercise data loading and q-range selection, SasModels setup,
parameter handling, BUMPS and SciPy optimization, plotting, result export, and
structure factors. Temporary datasets and output files are removed
automatically.

To execute the complete teaching material as well:

```bash
python scripts/generate_workshop.py --check
```

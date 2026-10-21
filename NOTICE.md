# Provenance

This is a standalone teaching repository for a Masters-level SANS data analysis
workshop.

The bundled `src/sans_fitter` helper and its original tests were adapted from
the BSD-licensed [`rozyczko/SANS-fitter`](https://github.com/rozyczko/SANS-fitter)
project. They are retained here as internal workshop infrastructure so that the
student exercises have a small, stable interface to SasModels, BUMPS, and
SciPy. Subsequent workshop-specific fixes include q-range masking,
double-precision fitting, and explicit structure-factor radius handling.

The repository as a whole is distributed under the BSD 3-Clause license in
[`LICENSE`](LICENSE).

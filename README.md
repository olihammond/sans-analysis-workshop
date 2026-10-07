# SANS Data Analysis Workshop

A self-contained, Masters-level practical for learning small-angle neutron
scattering (SANS) model fitting in JupyterLab. The material is designed for a
four-hour afternoon session on VISA, but it can also run on a local computer.

Students work from inspecting reduced data through parameter fitting, residual
diagnosis, model ambiguity, q-range sensitivity, and inter-particle structure
factors. All datasets and the small fitting interface used by the notebooks are
included in this repository.

## Learning outcomes

By the end of the workshop, students should be able to:

- interpret the main regions and features of a one-dimensional SANS curve;
- choose initial values, bounds, and fixed or varying parameters deliberately;
- fit SasModels form factors and inspect normalized residuals;
- compare models using fit quality and physical plausibility;
- explain how limited q coverage creates parameter and model ambiguity;
- recognize when particle interactions require a structure factor.
- interpret posterior intervals, chain diagnostics, and predictive checks.

## Start on VISA

Open a terminal in the cloned repository:

```bash
pixi install
pixi run workshop
```

In JupyterLab, open [`exercises/00_start_here.ipynb`](exercises/00_start_here.ipynb).
The first code cell prints the loaded `sans_fitter` path. It must point to this
repository's `src/sans_fitter` directory, not a package under `site-packages`.

If the VISA image does not provide the Prefix.dev Pixi executable:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
jupyter lab exercises
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Workshop schedule

| Time | Workbook | Focus |
|---:|---|---|
| 15 min | `00_start_here.ipynb` | Environment check, data conventions, plotting |
| 65 min | `01_fit_a_sphere.ipynb` | Form factors, bounds, fitting, residuals |
| 10 min | Break | |
| 70 min | `02_model_detective.ipynb` | Sphere vs ellipsoid and q-range ambiguity |
| 55 min | `03_when_particles_interact.ipynb` | Structure factors and physical interpretation |
| 25 min | Discussion | Challenge review and plenary |
| 60–75 min | `04_bayesian_uncertainty.ipynb` | Optional advanced DREAM sampling and posterior checks |

Student workbooks are in [`exercises/`](exercises/). They first demonstrate the
fitting API in small steps, then use `YOUR CODE HERE` cells and deliberate
`NotImplementedError` stops to require students to complete each analysis.
Fully executable reference copies are in [`solutions/`](solutions/).

## Repository layout

```text
.
├── data/             # Checked-in synthetic SANS datasets
├── exercises/        # Student workbooks
├── instructor/       # Dataset ground truth
├── scripts/          # Reproducible material generator
├── solutions/        # Worked and executable notebooks
├── src/sans_fitter/  # Bundled teaching-oriented fitting helper
└── tests/            # Tests for the fitting helper
```

The three datasets use `Q,I,dI` columns, with q in inverse angstrom and
intensity in arbitrary units:

- `sphere_reference.csv`: dilute, nearly monodisperse spherical particles;
- `mystery_shape.csv`: a deliberately noisy anisotropic-particle dataset;
- `interacting_spheres.csv`: spheres with a hard-sphere interaction peak.

Their fixed generation seed and source parameters are recorded in
[`instructor/ground_truth.json`](instructor/ground_truth.json).

## Instructor guidance

- Pair students for the model-selection exercise and ask each partner to defend
  one model before revealing the ground truth.
- Emphasize residual structure, parameter plausibility, and information content;
  a lower objective value alone is not a scientific conclusion.
- In exercise 2, removing high-q information makes the competing models nearly
  indistinguishable and provides a concrete example of non-identifiability.
- In exercise 3, allow students to inspect the failed sphere-only fit before
  introducing the structure factor.
- End by asking what additional measurement would distinguish each pair of
  competing physical explanations.

## Maintaining the materials

Regenerate datasets and clean notebooks:

```bash
python scripts/generate_workshop.py
```

Regenerate the materials, validate student task cells, and execute every
solution notebook:

```bash
python scripts/generate_workshop.py --check
```

Run library tests and code-quality checks:

```bash
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
```

Notebook checks use a non-interactive Matplotlib backend and write executed
solution copies to a temporary directory, leaving the teaching copies clean.

## Bundled fitting helper

The `sans_fitter` Python package is included to give the notebooks a concise,
stable interface to SasModels, BUMPS, and SciPy. It is supporting workshop code,
not the identity or primary deliverable of this repository. See
[`NOTICE.md`](NOTICE.md) for provenance.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).

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
- recognize when particle interactions require a structure factor;
- interpret posterior intervals, chain diagnostics, and predictive checks.

## Connection to the lecture

The workshop is the practical half of the lecture *Small-angle scattering: from
principles to practice* (22 October 2026; slide numbers follow the
`221026_OH_SANS1_ed` version of the deck). Each notebook opens with a
**Lecture connection** box that names the slides it builds on. Slide titles are
quoted too, so the links survive renumbering.

| Workbook | Builds on lecture slides |
|---|---|
| `00_start_here` | 9–10 Q and real space; 21–22 Guinier and power laws; 41 resolution; 43–46 data reduction |
| `01_fit_a_sphere` | 20 master equation; 23 form factor; 30–33 sample knowledge, fitting strategy, fit quality |
| `02_model_detective` | 21 Guinier universality; 24–25 the SANS zoo, "name that shape"; 34 model choice as a prior; 50 same Rg, different shape |
| `03_when_particles_interact` | 10 P(Q)·S(Q); 21 low-Q warning signs; 25–26 structure factor; 51 micelles as P(Q)·S(Q) |
| `04_bayesian_uncertainty` | 20 and 35 parameter degeneracy; 34 priors and bounds |
| `mystery_data` | 30 Step 0; 21–26 model-free analysis and models; 33–34 fit quality and uncertainty; 50–54 real examples |

Recommended reading for students:

- C. J. Gommes, S. Jaksch & H. Frielinghaus, [*Small-angle scattering for
  beginners*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8662971/), J. Appl.
  Cryst. 54, 1832 (2021).
- C. A. Jacques & J. Trewhella, [*Small-angle scattering for structural
  biology: expanding the frontier while avoiding the
  pitfalls*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2867006/), Protein Sci.
  19, 642 (2010).
- J. S. Pedersen, *Analysis of small-angle scattering data from colloids and
  polymer solutions: modeling and least-squares fitting*, Adv. Colloid
  Interface Sci. 70, 171 (1997).
- [SasView model documentation](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html)
  and the [SasView model marketplace](https://marketplace.sasview.org/).

## Start on VISA

Open a terminal in the cloned repository:

```bash
pixi install
pixi run workshop
```

The first `pixi install` resolves the environment (including sans-fitter from
PyPI) and writes `pixi.lock`; commit that file so every later install matches.

In JupyterLab, open [`exercises/00_start_here.ipynb`](exercises/00_start_here.ipynb).
The first code cell prints the installed sans-fitter version, which must be 0.4
or newer.

If the VISA image does not provide the Prefix.dev Pixi executable:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev]"
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

Student workbooks are in [`exercises/`](exercises/). This is a SANS workshop,
not a programming course. The only code students write is sans-fitter calls
they have just been shown (`set_model`, `set_param`, `fit`, `plot_results`,
`plot_model`, `get_fit_report`, `set_structure_factor`, `fit_bayesian` and the
posterior plots), in cells
marked `YOUR CODE HERE` that list the exact lines to add. Everything else
(functions, loops, reading values out of results, NumPy formulas) is provided
and commented. Scientific answers go in **Your answer** text cells. Fully
executable copies with model answers are in [`solutions/`](solutions/).

### Mystery-sample report

After the workbooks, each student picks one of six unlabelled datasets in
[`mystery_data/`](mystery_data/), works out what is in the beam, and writes a
short report on the fit and the physical interpretation. The folder README
gives each sample's composition, conditions and SLDs, the hints students need
(including `set_polydispersity` for size distributions), and the report
structure. These datasets are in absolute units (cm⁻¹), so fitted volume
fractions can be checked against the stated concentrations.

## Repository layout

```text
.
├── data/             # Checked-in synthetic SANS datasets
├── exercises/        # Student workbooks
├── instructor/       # Dataset ground truth and mystery-sample answer key
├── mystery_data/     # Six unlabelled datasets for the report exercise
├── scripts/          # Reproducible material generator
└── solutions/        # Worked and executable notebooks
```

The three datasets use `Q,I,dI` columns, with q in inverse angstrom and
intensity in arbitrary units:

- `sphere_reference.csv`: dilute, nearly monodisperse spherical particles;
- `mystery_shape.csv`: a deliberately noisy anisotropic-particle dataset;
- `interacting_spheres.csv`: spheres with a hard-sphere interaction peak.

Their fixed generation seed and source parameters are recorded in
[`instructor/ground_truth.json`](instructor/ground_truth.json). The models,
parameters and expected interpretation of the mystery samples are in
[`instructor/mystery_ground_truth.json`](instructor/mystery_ground_truth.json).

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
- For the mystery-sample report, check that each sample has a student
  before the session. Samples D (silica) and F (polymer) are the most direct.
  A (block-copolymer micelles with interactions) and E (bicontinuous
  microemulsion) are the most demanding.

## Maintaining the materials

Regenerate datasets and clean notebooks:

```bash
python scripts/generate_workshop.py
```

Regenerate the materials and execute every student and solution notebook
(student workbooks must run end to end without edits):

```bash
python scripts/generate_workshop.py --check
```

Check code quality:

```bash
ruff check scripts
ruff format --check scripts
```

Notebook checks use a non-interactive Matplotlib backend and write executed
solution copies to a temporary directory, leaving the teaching copies clean.

## Fitting package

The notebooks use [sans-fitter](https://pypi.org/project/sans-fitter/) 0.4 or
newer, installed from PyPI ([documentation](https://ai4se1dk.github.io/SANS-fitter/),
[source](https://github.com/ai4se1dk/SANS-fitter)). It gives a concise interface
to SasModels, BUMPS and SciPy. Students can keep using it after the workshop.

Two settings matter for the workshop data:

- The CSV files have no `dQ` column, so every fitter calls
  `set_resolution('none')`. Without it, sans-fitter warns that it cannot smear
  the model.
- For a hard-sphere structure factor with a freely fitted `radius_effective`,
  notebook 03 sets `radius_effective_mode` to 0 after `set_structure_factor()`.
  In sans-fitter 0.4 the "unconstrained" mode leaves sasmodels' default (1) in
  place, which computes the interaction radius from the particle radius and
  ignores `radius_effective`.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).

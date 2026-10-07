"""Generate deterministic workshop datasets and Jupyter notebooks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
NOTEBOOK_DIR = ROOT / 'exercises'
SOLUTION_DIR = ROOT / 'solutions'
INSTRUCTOR_DIR = ROOT / 'instructor'
SEED = 20260929


def evaluate(model_name: str, q: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    """Evaluate a SasModels model on q."""
    data = empty_data1D(q)
    calculator = DirectModel(data, load_model(model_name))
    return np.asarray(calculator(**parameters))


def noisy_dataset(
    model_name: str,
    parameters: dict[str, float],
    *,
    points: int,
    qmin: float,
    qmax: float,
    relative_error: float,
    error_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return q, noisy intensity, and one-sigma uncertainty columns."""
    q = np.logspace(np.log10(qmin), np.log10(qmax), points)
    truth = evaluate(model_name, q, parameters)
    uncertainty = np.sqrt((relative_error * truth) ** 2 + error_floor**2)
    observed = np.maximum(truth + rng.normal(0.0, uncertainty), np.finfo(float).tiny)
    return np.column_stack((q, observed, uncertainty))


def write_datasets() -> dict[str, dict[str, object]]:
    """Create the workshop datasets and return their ground truth."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUCTOR_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    specifications: dict[str, dict[str, object]] = {
        'sphere_reference': {
            'model': 'sphere',
            'parameters': {
                'radius': 58.0,
                'sld': 4.2,
                'sld_solvent': 1.0,
                'scale': 0.065,
                'background': 0.0015,
            },
            'sampling': {
                'points': 90,
                'qmin': 0.004,
                'qmax': 0.35,
                'relative_error': 0.04,
                'error_floor': 0.00035,
            },
        },
        'mystery_shape': {
            'model': 'ellipsoid',
            'parameters': {
                'radius_polar': 125.0,
                'radius_equatorial': 34.0,
                'sld': 4.2,
                'sld_solvent': 1.0,
                'scale': 0.04,
                'background': 0.0018,
            },
            'sampling': {
                'points': 105,
                'qmin': 0.004,
                'qmax': 0.32,
                'relative_error': 0.055,
                'error_floor': 0.00045,
            },
        },
        'interacting_spheres': {
            'model': 'sphere@hardsphere',
            'parameters': {
                'radius': 43.0,
                'radius_effective': 48.0,
                'radius_effective_mode': 0,
                'volfraction': 0.27,
                'sld': 4.2,
                'sld_solvent': 1.0,
                'scale': 0.055,
                'background': 0.0012,
            },
            'sampling': {
                'points': 100,
                'qmin': 0.004,
                'qmax': 0.38,
                'relative_error': 0.045,
                'error_floor': 0.0003,
            },
        },
    }

    for name, specification in specifications.items():
        values = noisy_dataset(
            str(specification['model']),
            dict(specification['parameters']),
            rng=rng,
            **dict(specification['sampling']),
        )
        np.savetxt(
            DATA_DIR / f'{name}.csv',
            values,
            delimiter=',',
            header='Q,I,dI',
            comments='',
            fmt='%.10e',
        )

    truth = {'seed': SEED, 'datasets': specifications}
    (INSTRUCTOR_DIR / 'ground_truth.json').write_text(
        json.dumps(truth, indent=2) + '\n', encoding='utf-8'
    )
    return specifications


def markdown(source: str) -> dict[str, object]:
    """Create a markdown notebook cell."""
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    """Create a code notebook cell."""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source.splitlines(keepends=True),
    }


def connection(source: str) -> dict[str, object]:
    """Create a highlighted markdown cell linking the notebook to the lecture.

    Slide numbers refer to the lecture deck 'Small-angle scattering: from
    principles to practice' (22 October 2026). Slide titles are given as well
    so the links survive renumbering.
    """
    quoted = '\n'.join(f'> {line}'.rstrip() for line in source.strip().splitlines())
    return markdown(f'> **Lecture connection**\n>\n{quoted}')


# Reading and reference links used across the notebooks.
GOMMES = (
    '[Gommes, Jaksch & Frielinghaus, *Small-angle scattering for beginners*, '
    'J. Appl. Cryst. 54, 1832 (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8662971/)'
)
JACQUES = (
    '[Jacques & Trewhella, *Small-angle scattering for structural biology: expanding the '
    'frontier while avoiding the pitfalls*, Protein Sci. 19, 642 (2010)]'
    '(https://pmc.ncbi.nlm.nih.gov/articles/PMC2867006/)'
)
SASVIEW_MODEL = 'https://www.sasview.org/docs/user/models/{}.html'
SASVIEW_FIT_QUALITY = (
    'https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/residuals_help.html'
)
SASVIEW_POLYDISPERSITY = (
    'https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html'
)
SASVIEW_RESOLUTION = 'https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/resolution.html'
SASVIEW_SLD = 'https://www.sasview.org/docs/user/qtgui/Calculators/sld_calculator_help.html'
SASVIEW_MARKETPLACE = 'https://marketplace.sasview.org/'
BUMPS_OPTIMIZERS = 'https://bumps.readthedocs.io/en/latest/guide/optimizer.html'
SCIPY_LEAST_SQUARES = (
    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html'
)
PHASE_MAP_VIDEO = 'https://www.youtube.com/watch?v=4VKR-0b25Og'


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    """Build a minimal Jupyter notebook document."""
    for index, cell in enumerate(cells):
        cell['id'] = f'cell-{index:02d}'
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python', 'version': '3.10'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


SETUP = """from pathlib import Path
import sys

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'pyproject.toml').exists())
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import sans_fitter
from sans_fitter import SANSFitter

DATA = ROOT / 'data'
print(f'Using sans_fitter from {Path(sans_fitter.__file__).resolve()}')
%matplotlib inline
"""


def start_notebook(solution: bool) -> dict[str, object]:
    """Create the orientation notebook."""
    label = 'Solution' if solution else 'Student workbook'
    cells = [
        markdown(
            f"""# 00 — Start here ({label})

**Time:** 15 minutes
**Goal:** confirm the environment, load reduced SANS data, and recognize the
three-column `Q,I,dI` convention.

Work in pairs. Discuss every prediction before running the next cell."""
        ),
        connection(
            """This practical picks up exactly where last week's lecture stopped: at a
reduced, radially averaged curve. Keep the lecture slides open beside you.

| Today's task | Lecture slides |
|---|---|
| Convert q to a real-space length | 8 *Scattering geometry*, 9 *Real and Q-space* |
| Read the shape of a log–log curve | 19 *Guinier analysis*, 20 *Power law slopes* |
| Know what "reduced data" has already been through | 41–44 *Data reduction* … *When data look wrong* |

**Notation.** The lecture writes the scattering vector as $Q$; the code and data
files write `q` or `Q`. They are the same quantity,
$Q = 4\\pi\\sin\\theta/\\lambda$ with $2\\theta$ the scattering angle, here in Å⁻¹.
Always check θ versus 2θ and Å⁻¹ versus nm⁻¹ when you open someone else's data."""
        ),
        code(SETUP),
        code(
            """filename = DATA / 'sphere_reference.csv'
values = np.genfromtxt(filename, delimiter=',', names=True)
q, intensity, uncertainty = values['Q'], values['I'], values['dI']
print(f'{len(q)} points; q = {q.min():.4f}–{q.max():.3f} Å⁻¹')
assert np.all(np.diff(q) > 0)
assert np.all(uncertainty > 0)
"""
        ),
        markdown(
            """## From q to real space

The lecture rule of thumb is that a feature of size $d$ shows up near
$Q \\approx 2\\pi/d$: **big in real space, small in reciprocal space**. The cell
below converts the measured q window into the real-space distances it can see.
Before running it, predict roughly what the largest and smallest distances will be."""
        ),
        code(
            """d_max = 2 * np.pi / q.min()
d_min = 2 * np.pi / q.max()
print(f'Largest distance probed:  {d_max:6.0f} Å = {d_max / 10:5.1f} nm')
print(f'Smallest distance probed: {d_min:6.1f} Å = {d_min / 10:5.1f} nm')
"""
        ),
        markdown(
            """## Predict before plotting

1. Why are both axes usually logarithmic for SANS?
2. What information does `dI` carry?
3. Which part of the curve is most sensitive to overall particle size?

Write a one-sentence prediction in the cell below.

*Hints from the lecture:* intensity typically falls by several orders of
magnitude over the measured range, and power laws $I \\propto Q^{-m}$ become
straight lines of slope $-m$ on log–log axes (slide 20). Size information sits
where $Q \\sim 2\\pi/d$ for the whole particle, i.e. at the low-q end (slide 19)."""
        ),
        code(
            """prediction = (
    'Low q should contain information about the overall size; logarithmic axes '
    'show features spanning several orders of magnitude.'
)
print(prediction)
"""
            if solution
            else """# TODO: Replace this with your prediction.
prediction = 'I expect intensity to decrease as q increases.'
print(prediction)
"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(q, intensity, yerr=uncertainty, fmt='o', ms=3, capsize=2)
ax.set(xscale='log', yscale='log', xlabel=r'$q$ / Å$^{-1}$', ylabel=r'$I(q)$')
ax.grid(alpha=0.25)
plt.show()
"""
        ),
        markdown(
            """## Reading the curve

Use the lecture's reading guide on the plot above, moving from left to right:

- **Low-q plateau (Guinier region).** For $Q R_g \\lesssim 1.3$ the whole
  particle scatters almost in phase, $I(Q) \\approx I(0)\\exp(-Q^2R_g^2/3)$, and
  the curve flattens. For a solid sphere $R_g = \\sqrt{3/5}\\,R$ (slide 19).
- **Oscillations (form factor).** Sharp minima mean nearly identical particles.
  For a sphere the first minimum sits at $Q_1 R \\approx 4.49$ (slide 21).
- **High-q decay.** Smooth, sharp interfaces give the Porod envelope
  $I \\propto Q^{-4}$ (slide 20).
- **Flat floor at the highest q.** Background, mostly incoherent scattering
  from hydrogen, which is $Q$-independent to a good approximation (slide 14).

Mark each region on a sketch of your plot. Read off $Q_1$ as accurately as you
can and estimate the particle radius. You will need this number as a starting
value in notebook 01."""
        ),
        markdown(
            """## Quick check

The curve is already reduced: detector corrections, normalization, radial
averaging, and uncertainty propagation happened upstream. This workshop starts
at the point where a scientific model is compared with those reduced data.

Two consequences matter for the rest of the afternoon:

1. **Units.** In the lecture, $I(Q)$ is an absolute cross-section in cm⁻¹ and
   the master equation (slide 18) lets you check the fitted volume fraction.
   These workshop data are in *arbitrary units*, so the fitted `scale`
   parameter absorbs the volume fraction *and* any calibration factor.
2. **Reduction artefacts look like physics.** Slide 44 (*When data look
   wrong…*) is a useful diagnostic when a residual pattern puzzles you: a
   discrepancy at high q only suggests a flat or weak-signal term (cell,
   background); a constant factor everywhere suggests normalization, thickness,
   transmission or calibration; a change of shape suggests inconsistent
   subtraction. The synthetic data here are clean, but real data rarely are."""
        ),
        markdown(
            f"""## Further reading

- {GOMMES}: an almost mathematics-free introduction that matches the lecture's
  phase-map picture of the form factor.
- {JACQUES}: the classic checklist of what can go wrong between sample and curve.
- [SasView SLD calculator]({SASVIEW_SLD}): compute the scattering length
  densities that the models in this workshop take as fixed inputs."""
        ),
    ]
    return notebook(cells)


def sphere_notebook(solution: bool) -> dict[str, object]:
    """Create the introductory fitting notebook."""
    label = 'Solution' if solution else 'Student workbook'
    sphere_function = (
        """def sphere_amplitude(q, radius):
    qr = q * radius
    return 3 * (np.sin(qr) - qr * np.cos(qr)) / qr**3


trial_radius = 58.0
trial_shape = sphere_amplitude(q, trial_radius) ** 2
"""
        if solution
        else """def sphere_amplitude(q, radius):
    qr = q * radius
    # TODO: implement 3 * (sin(qr) - qr*cos(qr)) / (qr**3)
    return np.ones_like(qr)


trial_radius = 50.0  # TODO: change this and rerun the plot
trial_shape = sphere_amplitude(q, trial_radius) ** 2
"""
    )
    model_setup = (
        """model_name = 'sphere'
fitter.set_model(model_name)
fitter.get_params()
"""
        if solution
        else """# `set_model` takes a SasModels model name as a string.
# TODO: replace None with the model name for a homogeneous sphere.
model_name = None
if model_name is None:
    raise NotImplementedError('Choose a SasModels model')

fitter.set_model(model_name)

# `get_params` lists every parameter supplied by that model.
fitter.get_params()
"""
    )
    params = (
        """fitter.set_param('sld', value=4.2, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)
fitter.set_param('radius', value=50, min=20, max=100, vary=True)
fitter.set_param('scale', value=0.05, min=0.001, max=0.2, vary=True)
fitter.set_param('background', value=0.002, min=0, max=0.01, vary=True)
"""
        if solution
        else """# One fixed parameter is demonstrated below. `vary=False` tells the
# optimizer to retain this independently known value.
fitter.set_param('sld', value=4.2, vary=False)

# YOUR CODE HERE
# 1. Fix `sld_solvent` at 1.0.
# 2. Configure radius, scale, and background with an initial value and bounds.
# 3. Set `vary=True` for those three fitted parameters.
raise NotImplementedError('Configure the remaining model parameters')
"""
    )
    fit_call = (
        """result = fitter.fit(engine='lmfit', method='least_squares')
fitter.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# `fit` accepts an optimization engine and an engine-specific method.
# For a bounded local fit, use engine='lmfit' and method='least_squares'.
# Store the returned dictionary in `result`, then call `plot_results`.

# YOUR CODE HERE
raise NotImplementedError('Fit the configured model and plot its residuals')
"""
    )
    cells = [
        markdown(
            f"""# 01 — Fit a sphere ({label})

**Time:** 65 minutes
**Question:** what can a SANS curve tell us about a dilute spherical particle?

You will connect the analytical sphere amplitude to a SasModels fit, choose
bounds, and diagnose the result using normalized residuals."""
        ),
        connection(
            """- Slide 18, *The SANS master equation*: the model you fit is this equation
  with $S(Q)=1$ (dilute particles).
- Slide 21, *The form factor, P(Q)*: the phase-map picture of why a sphere
  gives minima at $QR \\approx 4.49, 7.73, \\ldots$ Rewatch the
  [phase-map video]("""
            + PHASE_MAP_VIDEO
            + """) if the formula below feels abstract.
- Slides 28 and 30, *Step 0: understand your sample* and *Fitting strategy*:
  where defensible starting values and bounds come from.
- Slide 31, *Is my fit good or not?*: what this notebook asks you to judge."""
        ),
        code(SETUP),
        code(
            """values = np.genfromtxt(DATA / 'sphere_reference.csv', delimiter=',', names=True)
q, intensity, uncertainty = values['Q'], values['I'], values['dI']
"""
        ),
        markdown(
            r"""## Exercise 1 — Reduce the model

For a homogeneous sphere,

$$
F(q)=3\frac{\sin(qr)-qr\cos(qr)}{(qr)^3}, \qquad
I(q)=\mathrm{scale}\,V(\Delta\rho)^2F(q)^2+\mathrm{background}.
$$

Explain why volume is not an independent fifth shape parameter, then complete
the amplitude function. Ignore absolute scaling in this first visual test.

**How this maps onto the lecture.** SasModels writes the sphere as
$I(q) = \frac{\mathrm{scale}}{V}\,\left[3V\Delta\rho\,\frac{\sin(qr)-qr\cos(qr)}{(qr)^3}\right]^2
+ \mathrm{background}$, which is the expression above. Compare it with the
master equation on slide 18,
$I(Q) = \phi\,V_p\,(\Delta\rho)^2\,P(Q)\,S(Q) + \mathrm{bkg}$, where
$P(Q) = F(Q)^2$ is normalized so that $P(0)=1$. For dilute particles $S(Q)=1$,
and on an absolute scale `scale` *is* the volume fraction $\phi$. Because the
workshop data are in arbitrary units, `scale` here also absorbs a calibration
factor. That is one face of the $\phi \leftrightarrow \Delta\rho \leftrightarrow
\mathrm{scale}$ degeneracy from the lecture's part 2 wrap-up (slide 34). We fix
both SLDs so that `scale` carries the rest.

*Hint for the volume question:* write $V$ in terms of $r$."""
        ),
        code(sphere_function),
        code(
            """fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(q, intensity / intensity.max(), 'o', ms=3, label='normalized data')
ax.loglog(q, trial_shape / trial_shape.max(), label=f'trial r = {trial_radius:g} Å')
ax.set(xlabel=r'$q$ / Å$^{-1}$', ylabel='normalized intensity')
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""
        ),
        markdown(
            """### Reading the size off the curve

Change `trial_radius` until the first minimum of your trial curve lines up
with the first minimum of the data. Then check yourself with the lecture's
rule of thumb (slide 21): the first zero of $F(qr)$ is at $qr \\approx 4.493$,
so $r \\approx 4.49/q_1$.

- The lecture's worked example used $R = 80$ Å, with minima at
  $Q = 0.0562, 0.0966, 0.136$ Å⁻¹. Are the data minima to the left or right of
  those? Is this sphere bigger or smaller than 80 Å?
- Why are the data minima not exactly zero? Think about noise, the background,
  and, for real samples, polydispersity and instrumental resolution (slide 21's
  dQ/Q panels and slide 39).

The radius you read off here is a *model-free* first estimate. Use it as the
starting value in the fit below rather than guessing."""
        ),
        markdown(
            """## Exercise 2 — The fitting workflow

Every fit follows the same sequence:

1. create `SANSFitter()`;
2. load a reduced dataset with `load_data(path)`;
3. select a SasModels model with `set_model('model_name')`;
4. inspect its parameters with `get_params()`;
5. configure each parameter with `set_param(...)`;
6. call `fit(...)` and inspect the residuals.

The first two steps are demonstrated below. You will choose the model."""
        ),
        code(
            """fitter = SANSFitter()
fitter.load_data(str(DATA / 'sphere_reference.csv'))
"""
        ),
        code(model_setup),
        markdown(
            """### Changing parameters

The general form is:

```python
fitter.set_param(
    'parameter_name',
    value=initial_guess,
    min=lower_bound,
    max=upper_bound,
    vary=True,
)
```

- `value` is the optimizer's starting point, not the answer.
- `min` and `max` express physically defensible limits.
- `vary=True` lets the optimizer change the parameter.
- `vary=False` fixes independently known quantities such as SLD.

Keep both SLDs fixed. Choose and justify initial values and bounds for radius,
scale, and background. Afterward, call `get_params()` to audit your setup.

**Where do defensible numbers come from?** The lecture's *Step 0* (slide 28)
is to know what you put in the beam. In practice:

| Parameter | Lecture source for a starting value | Sensible bounds reflect… |
|---|---|---|
| `sld`, `sld_solvent` | Composition and density via an SLD calculator (slide 11) | fixed: known independently |
| `radius` | First minimum, $r \\approx 4.49/q_1$ (slide 21), or Guinier, $R = \\sqrt{5/3}\\,R_g$ (slide 19) | synthesis, TEM, or DLS knowledge |
| `scale` | Height of the low-q plateau, $I(0) \\propto \\phi V \\Delta\\rho^2$ (slide 18) | concentration and calibration |
| `background` | Level of the flat high-q floor (slide 14) | must be ≥ 0 and below the data |

The SLD values used here are 4.2 for the particle and 1.0 for the solvent, in
units of 10⁻⁶ Å⁻², the "SLD units" of slide 11."""
        ),
        code(params + '\nfitter.get_params()\n'),
        markdown(
            """### Running the optimizer

`fit(engine='lmfit', method='least_squares')` performs a bounded local
least-squares fit. The returned dictionary contains the objective value and
fitted parameters. `plot_results(...)` overlays the model and displays
normalized residuals.

**What the optimizer actually does.** It searches for the parameters $\\theta$
that minimize

$$
\\chi^2(\\theta) = \\sum_i \\left(\\frac{I_i - I_{\\mathrm{model}}(q_i;\\theta)}{\\mathrm{d}I_i}\\right)^2 .
$$

Each term in brackets is a **normalized residual**: the misfit at one point,
measured in units of its own uncertainty. That is why `dI` matters (notebook
00). For a good model with correct error bars, about 68 % of normalized
residuals should lie between −1 and +1, few should exceed ±3, and they should
show no pattern along q ([SasView: *Assessing fit quality*]("""
            + SASVIEW_FIT_QUALITY
            + """)).

Two practical notes:

- The engine label `'lmfit'` is historical. Underneath, the helper calls
  [SciPy `least_squares`]("""
            + SCIPY_LEAST_SQUARES
            + """), a bounded trust-region method that only searches downhill from the
  starting point.
- With `engine='lmfit'`, `result['chisq']` is the **sum** above. The BUMPS
  engine reports the **reduced** value $\\chi^2/(N-p)$, which is ≈ 1 for a good
  fit ($N$ data points, $p$ fitted parameters). Compare like with like."""
        ),
        code(fit_call),
        markdown(
            """## Exercise 3 — Criticize, do not merely accept

Record:

- fitted radius and uncertainty;
- whether residuals look randomly distributed around zero;
- any fitted parameter touching a bound;
- one experimental systematic that this model ignores.

**Checklist for "is my fit good?" (slide 31).**

1. *Residuals:* scattered around zero with no runs or waves? A wave that
   follows the form-factor oscillations means the shape or size distribution
   is wrong, not the noise.
2. *Statistic:* is the reduced χ², $\\chi^2/(N-p)$, close to 1? Much larger
   means a missing ingredient or underestimated `dI`. Much smaller means
   overestimated `dI` or overfitting.
3. *Bounds:* a parameter pinned at a bound has not been determined. The
   bound has decided it.
4. *Plausibility:* does the radius agree with your model-free estimate from
   the first minimum? Is the background below the lowest data points?

**Systematics a single monodisperse sphere ignores:** a size distribution
([SasView polydispersity]("""
            + SASVIEW_POLYDISPERSITY
            + """)) and instrumental smearing ([SasView resolution]("""
            + SASVIEW_RESOLUTION
            + """)). Both fill in the minima (slide 21 shows 0, 5 and 10 % dQ/Q). The
lecture also covered reduction errors in transmission, empty-cell subtraction
and thickness (slides 42 and 44), and interparticle interactions, $S(Q) \\neq 1$
(notebook 03). These synthetic data were generated without any of them. That is
why the minima are so sharp. Say which one you would expect to matter most for a
real measurement, and why."""
        ),
        code(
            """for name in ['radius', 'scale', 'background']:
    print(name, result['parameters'][name]['formatted'])
print('Fit statistic:', result['chisq'])
"""
            if solution
            else """# YOUR CODE HERE
# Extract the fitted radius, scale, and background from `result`.
# Then write your scientific interpretation in a new Markdown cell.
raise NotImplementedError('Summarize and interpret the fit')
"""
        ),
        markdown(
            """## Extension — Bad starting values

Change the initial radius to 15 Å while retaining broad bounds. Does the local
optimizer reach the same answer? If not, try BUMPS differential evolution with
`fitter.fit(engine='bumps', method='de', steps=200)`. What does this reveal
about optimization versus inference?

*Why this happens:* the χ² surface of a form factor is full of local minima.
A sphere of 15 Å puts its first minimum near $4.49/15 \\approx 0.3$ Å⁻¹, so
a downhill search starting there can lock onto the wrong oscillation.
[Differential evolution]("""
            + BUMPS_OPTIMIZERS
            + """#differential-evolution) keeps a population of candidates spread over
the whole prior box, so it is far less sensitive to the start. Neither method
tells you *how well* the radius is determined. That question, the theme of slide
32 (*How can uncertainties help*), is taken up in notebook 04.

Remember the χ² note above when you compare the two engines' numbers."""
        ),
        markdown(
            f"""## Further reading

- {GOMMES}, sections on the form factor and the sphere.
- [SasView `sphere` model documentation]({SASVIEW_MODEL.format('sphere')}): the exact
  equation, parameter units, and the meaning of `scale` on an absolute scale.
- J. S. Pedersen, *Analysis of small-angle scattering data from colloids and polymer
  solutions: modeling and least-squares fitting*, Adv. Colloid Interface Sci. 70, 171
  (1997): the standard reference for form factors, polydispersity and smearing in fits."""
        ),
    ]
    return notebook(cells)


def detective_notebook(solution: bool) -> dict[str, object]:
    """Create the model-selection notebook."""
    label = 'Solution' if solution else 'Student workbook'
    conclusion = (
        """conclusion = (
    'The ellipsoid is preferred on the full q-range: its residuals lose the '
    'systematic oscillatory pattern left by the sphere. After truncation the '
    'models are harder to distinguish, showing that fit quality depends on '
    'the information measured, not only the optimizer.'
)
print(conclusion)
"""
        if solution
        else """# TODO: defend a model using residuals and physical plausibility.
conclusion = 'Write your evidence-based conclusion here.'
print(conclusion)
"""
    )
    sphere_fit = (
        """def fit_sphere(qmin=None, qmax=None):
    fitter = SANSFitter()
    fitter.load_data(str(filename))
    if qmin is not None or qmax is not None:
        fitter.set_q_range(qmin=qmin, qmax=qmax)
    fitter.set_model('sphere')
    fitter.set_param('sld', value=4.2, vary=False)
    fitter.set_param('sld_solvent', value=1.0, vary=False)
    fitter.set_param('radius', value=45, min=10, max=180, vary=True)
    fitter.set_param('scale', value=0.04, min=0.001, max=0.2, vary=True)
    fitter.set_param('background', value=0.002, min=0, max=0.02, vary=True)
    return fitter, fitter.fit(engine='lmfit', method='least_squares')


sphere, sphere_result = fit_sphere()
sphere.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """def fit_sphere(qmin=None, qmax=None):
    # Fit a sphere, optionally over only part of the measured q range.
    fitter = SANSFitter()
    fitter.load_data(str(filename))
    if qmin is not None or qmax is not None:
        fitter.set_q_range(qmin=qmin, qmax=qmax)

    # YOUR CODE HERE
    # Select the model, configure its parameters, and return the fitter and
    # result. Choose your own defensible initial values and bounds.
    raise NotImplementedError('Implement the sphere fitting workflow')


sphere, sphere_result = fit_sphere()
sphere.plot_results(show_residuals=True, log_scale=True)
"""
    )
    ellipsoid_fit = (
        """ellipsoid = SANSFitter()
ellipsoid.load_data(str(filename))
ellipsoid.set_model('ellipsoid')
ellipsoid.set_param('sld', value=4.2, vary=False)
ellipsoid.set_param('sld_solvent', value=1.0, vary=False)
ellipsoid.set_param('radius_polar', value=100, min=15, max=220, vary=True)
ellipsoid.set_param('radius_equatorial', value=40, min=10, max=120, vary=True)
ellipsoid.set_param('scale', value=0.04, min=0.001, max=0.2, vary=True)
ellipsoid.set_param('background', value=0.002, min=0, max=0.02, vary=True)
ellipsoid_result = ellipsoid.fit(engine='lmfit', method='least_squares')
ellipsoid.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """ellipsoid = SANSFitter()
ellipsoid.load_data(str(filename))

# Select 'ellipsoid', call `get_params()` to discover its two radius
# parameters, then configure and fit it.

# YOUR CODE HERE
raise NotImplementedError('Configure and fit the ellipsoid model')
"""
    )
    short_range = (
        """sphere_short, sphere_short_result = fit_sphere(qmin=0.004, qmax=0.025)
sphere_short.plot_results(show_residuals=True, log_scale=True)

ellipsoid_short = SANSFitter()
ellipsoid_short.load_data(str(filename))
ellipsoid_short.set_q_range(qmin=0.004, qmax=0.025)
ellipsoid_short.set_model('ellipsoid')
ellipsoid_short.set_param('sld', value=4.2, vary=False)
ellipsoid_short.set_param('sld_solvent', value=1.0, vary=False)
ellipsoid_short.set_param('radius_polar', value=100, min=15, max=220, vary=True)
ellipsoid_short.set_param('radius_equatorial', value=40, min=10, max=120, vary=True)
ellipsoid_short.set_param('scale', value=0.04, min=0.001, max=0.2, vary=True)
ellipsoid_short.set_param('background', value=0.002, min=0, max=0.02, vary=True)
ellipsoid_short_result = ellipsoid_short.fit(engine='lmfit', method='least_squares')
ellipsoid_short.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# Reuse `fit_sphere` with qmin=0.004 and qmax=0.025.
# Then build an ellipsoid fit with the same q limits.

# YOUR CODE HERE
raise NotImplementedError('Fit both models over the restricted q range')
"""
    )
    cells = [
        markdown(
            f"""# 02 — Model detective ({label})

**Time:** 70 minutes
**Question:** when does the dataset contain enough information to distinguish
a sphere from an ellipsoid?

The file name deliberately reveals nothing about the generating model. Split
the work: one partner advocates for a sphere, the other for an ellipsoid."""
        ),
        connection(
            """- Slide 22, *Welcome to the SANS zoo*: many shapes share a plateau and a
  decay and differ in oscillation depth and mid-q slope. Browse the
  [SasView model marketplace]("""
            + SASVIEW_MARKETPLACE
            + """) for the full zoo.
- Slide 23, *Name that shape!*: panels B and C "look alike over a narrow
  Q-window", and an extended Q-range shows the difference. This notebook is
  that statement made quantitative.
- Slide 19, *Guinier analysis*: the $R_g$ table. Every shape looks the same at
  low Q apart from $I(0)$ and $R_g$.
- Slide 33, *Different models, similar fits*: bring your figures from this
  notebook to the plenary discussion."""
        ),
        code(SETUP),
        code(
            """filename = DATA / 'mystery_shape.csv'
values = np.genfromtxt(filename, delimiter=',', names=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(values['Q'], values['I'], yerr=values['dI'], fmt='o', ms=3)
ax.set(xscale='log', yscale='log', xlabel=r'$q$ / Å$^{-1}$', ylabel=r'$I(q)$')
ax.grid(alpha=0.25)
plt.show()
"""
        ),
        markdown(
            """### Look before you fit

The lecture's fitting strategy (slide 30) starts with reading the data, not
with the optimizer. Compare this curve with `sphere_reference.csv` from
notebook 01:

- Are the form-factor minima as deep and sharp? Shallow, washed-out minima
  can mean polydispersity or smearing, but also **anisotropy**: an
  orientationally averaged ellipsoid has a different dimension in every
  direction, so its minima fall at different q and blur together.
- Where does the plateau bend over, and what slope follows it (slide 20)?
  A very long object would show an extended $Q^{-1}$ region. A moderately
  elongated one shows only a hint of it.
- Notice that these data are noisier than in notebook 01. How might that
  affect your ability to tell two models apart?"""
        ),
        markdown(
            """## Exercise 1 — Competing explanations

You now know the full workflow, so this notebook gives less code. Implement a
reusable sphere fit first. Keep SLD values fixed and vary only radius, scale,
and background. Compare residual patterns, not just the numerical objective."""
        ),
        code(sphere_fit),
        markdown(
            """### A competing model

Create a **new** `SANSFitter` for the ellipsoid so the two fits remain
independent. After `set_model('ellipsoid')`, use `get_params()` to discover the
model's parameter names. Decide whether each should be fixed or varied."""
        ),
        code(ellipsoid_fit),
        code(
            """print('Sphere objective:', sphere_result['chisq'])
print('Ellipsoid objective:', ellipsoid_result['chisq'])
for name in ['radius_polar', 'radius_equatorial']:
    print(name, ellipsoid_result['parameters'][name]['formatted'])
"""
        ),
        markdown(
            """## Exercise 2 — Remove information

Repeat both fits using only `0.004 ≤ q ≤ 0.025 Å⁻¹`. Predict what will happen
before running the cells. Which shape features have disappeared from view?

*Before you run:* convert the window to real space with $d \\approx 2\\pi/Q$,
as in notebook 00. What is the smallest distance still resolved, and how does
it compare with the particle dimensions you just fitted?"""
        ),
        code(short_range),
        markdown(
            """### Why the models become indistinguishable

The lecture called the Guinier law *universal*: at low Q every shape gives
$I(Q) \\approx I(0)\\exp(-Q^2R_g^2/3)$, which is valid for $QR_g \\lesssim 1.3$
(slide 19). In that region the data carry essentially **two numbers**, $I(0)$
and $R_g$. Any model that can reproduce those two numbers fits equally well.

Test this with your own fitted parameters:

1. For a spheroid with equatorial radius $R_e$ and polar radius $R_p$,
   $R_g^2 = (2R_e^2 + R_p^2)/5$. For a sphere, $R_g^2 = \\tfrac{3}{5}R^2$.
   Compute both from your **restricted-range** fits. How close are they?
2. Compute $Q_{\\max}R_g$ for the restricted window. Where does it sit relative
   to the Guinier limit of 1.3?
3. Using the full-range ellipsoid fit, what radius would a sphere *with the
   same* $R_g$ have? Is this close to the radius your restricted sphere fit
   found?

This is **non-identifiability**: the data are compatible with a family of
parameter sets, so the optimizer's single answer is not a measurement of
shape. Notebook 04 shows the same idea as correlated posterior distributions."""
        ),
        *(
            [
                markdown(
                    """**Solution note.** With the generating parameters
($R_e = 34$ Å, $R_p = 125$ Å) the ellipsoid has $R_g \\approx 60$ Å. An
equal-$R_g$ sphere has $R = \\sqrt{5/3}\\,R_g \\approx 77$ Å, and
$Q_{\\max}R_g \\approx 0.025 \\times 60 \\approx 1.5$: the restricted window is
essentially only the Guinier region. Your fitted values will differ slightly
because of noise."""
                )
            ]
            if solution
            else []
        ),
        markdown(
            """## Verdict

Write a short referee-style verdict. Include one reason based on residuals, one
based on parameter plausibility, and one follow-up measurement that would make
the distinction stronger.

Follow-up measurements from the lecture to consider:

- **Extend q.** A shorter detector distance or a shorter wavelength raises
  $Q_{\\max}$ (slides 8 and 37–39). Ask which q range the shape information
  lives in.
- **Change the contrast.** H/D variation (slides 13 and 15) changes $\\Delta\\rho$
  and so constrains the scale–contrast degeneracy, but does contrast alone
  change the *shape* of the curve for a homogeneous particle?
- **Complementary techniques.** Slide 6 asked when SANS and TEM would disagree
  about a size. Here, what could a micrograph settle that SANS cannot, and
  what could it get wrong?
- **Field or flow alignment.** Anisotropic objects give anisotropic 2D
  patterns when aligned (compare slide 27): a radially averaged 1D curve has
  thrown that information away."""
        ),
        code(conclusion),
        markdown(
            f"""## Further reading

- [SasView `ellipsoid` model]({SASVIEW_MODEL.format('ellipsoid')}): parameter
  definitions. `radius_polar` lies along the rotation axis, so $R_p > R_e$ is
  prolate (cigar-like) and $R_p < R_e$ is oblate (disc-like).
- {JACQUES}: the discussion of model ambiguity and of reporting what the data can
  and cannot support."""
        ),
    ]
    return notebook(cells)


def interactions_notebook(solution: bool) -> dict[str, object]:
    """Create the structure-factor notebook."""
    label = 'Solution' if solution else 'Student workbook'
    interpretation = (
        """interpretation = (
    'A form factor alone cannot reproduce the suppressed low-q intensity and '
    'correlation peak. The hard-sphere structure factor captures excluded-volume '
    'correlations; the fitted volume fraction should be treated as model-dependent.'
)
print(interpretation)
"""
        if solution
        else """# TODO: explain what changed physically, not only statistically.
interpretation = 'Write your physical interpretation here.'
print(interpretation)
"""
    )
    dilute_fit = (
        """dilute = SANSFitter()
dilute.load_data(str(filename))
dilute.set_model('sphere')
dilute.set_param('sld', value=4.2, vary=False)
dilute.set_param('sld_solvent', value=1.0, vary=False)
dilute.set_param('radius', value=40, min=15, max=90, vary=True)
dilute.set_param('scale', value=0.05, min=0.001, max=0.2, vary=True)
dilute.set_param('background', value=0.002, min=0, max=0.02, vary=True)
dilute_result = dilute.fit(engine='lmfit', method='least_squares')
dilute.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# Build the simplest plausible explanation: a dilute sphere.
# Keep both SLD values fixed and vary radius, scale, and background.

# YOUR CODE HERE
raise NotImplementedError('Build and fit the dilute-sphere model')
"""
    )
    interacting_fit = (
        """interacting = SANSFitter()
interacting.load_data(str(filename))
interacting.set_model('sphere')
interacting.set_param('sld', value=4.2, vary=False)
interacting.set_param('sld_solvent', value=1.0, vary=False)
interacting.set_param('radius', value=40, min=15, max=90, vary=True)
interacting.set_param('scale', value=0.05, min=0.001, max=0.2, vary=True)
interacting.set_param('background', value=0.002, min=0, max=0.02, vary=True)
interacting.set_structure_factor('hardsphere')
interacting.set_param('radius_effective', value=45, min=20, max=100, vary=True)
interacting.set_param('volfraction', value=0.2, min=0.01, max=0.55, vary=True)
interacting_result = interacting.fit(engine='lmfit', method='least_squares')
interacting.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """interacting = SANSFitter()
interacting.load_data(str(filename))

# Configure the same sphere form factor, add 'hardsphere', and call
# `get_params()` before configuring radius_effective and volfraction.

# YOUR CODE HERE
raise NotImplementedError('Add, inspect, configure, and fit the structure factor')
"""
    )
    comparison = (
        """print('Form factor only:', dilute_result['chisq'])
print('With structure factor:', interacting_result['chisq'])
for name in ['radius', 'radius_effective', 'volfraction']:
    print(name, interacting_result['parameters'][name]['formatted'])
"""
        if solution
        else """# Report both objective values and the fitted radius,
# effective radius, and volume fraction.

# YOUR CODE HERE
raise NotImplementedError('Compare the two physical explanations')
"""
    )
    cells = [
        markdown(
            f"""# 03 — When particles interact ({label})

**Time:** 55 minutes
**Question:** can a poor fit be repaired by changing particle shape, or is
inter-particle physics missing?

Start with the simplest plausible particle model. Use the residual pattern as a
clue before adding complexity."""
        ),
        connection(
            """- Slide 9, *Real and Q-space*: "the blob sets the envelope, the arrangement
  sets the sharp structure". This is the $P(Q)\\cdot S(Q)$ split.
- Slide 24, *The structure factor, S(Q)*: hard-sphere repulsion suppresses
  low-Q intensity and creates a correlation peak near $Q^* \\approx 2\\pi/\\sigma$.
- Slide 19, *Guinier analysis*: a low-Q **downturn** is the warning sign for
  repulsive interactions. An upturn suggests aggregation.
- Slide 23, *Name that shape!*, panel D: concentrated spheres. Dilute the
  sample, and if the peak disappears it was $S(Q)$."""
        ),
        code(SETUP),
        code(
            """filename = DATA / 'interacting_spheres.csv'
values = np.genfromtxt(filename, delimiter=',', names=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(values['Q'], values['I'], yerr=values['dI'], fmt='o', ms=3)
ax.set(xscale='log', yscale='log', xlabel=r'$q$ / Å$^{-1}$', ylabel=r'$I(q)$')
ax.grid(alpha=0.25)
plt.show()
"""
        ),
        markdown(
            """### Look before you fit

Compare this curve with the dilute sphere from notebook 01.

1. What happens at the lowest q: plateau, downturn or upturn? What does
   slide 19 say each one means?
2. Find the q position of the bump, $Q^*$. Convert it to a real-space
   distance, $2\\pi/Q^*$. Is that distance closer to a particle *radius*, a
   *diameter*, or neither? (Slide 24: the first peak sits near
   $Q^* \\approx 6/\\sigma$, where $\\sigma = 2R_{hs}$ is the contact distance.)
3. Is the bump a feature of one particle or of the arrangement of many?
   Name the experiment that would tell you (slide 9's discussion prompt)."""
        ),
        markdown(
            """## Exercise 1 — Deliberately incomplete model

From the previous notebooks, build a dilute-sphere fit without copying a
complete solution. Locate the structured residual region. Why can changing
radius or scale not independently repair it?

*Hint:* the lecture's "four knobs" (slide 18). Concentration, contrast and
scale only move the whole curve **up or down**. Size moves features **left or
right** and also changes $I(0) \\propto V$. Neither move can lower the
intensity at low q while raising it at $Q^*$. Also predict what a Guinier
analysis of these data would give for the size (slide 24: "Guinier analysis on
a concentrated sample under-estimates the size")."""
        ),
        code(dilute_fit),
        markdown(
            """## Exercise 2 — Add excluded-volume interactions

`set_structure_factor('hardsphere')` combines the selected form factor with a
hard-sphere interaction model. Calling `get_params()` afterward reveals the
additional parameters:

- `volfraction`: fraction of the sample volume occupied by interacting spheres;
- `radius_effective`: interaction distance, which may include a solvation layer.

Fit both independently. The mode-control parameters should remain fixed.

**What is being multiplied in.** This is the full master equation of slide 18,
$I(Q) = \\phi V \\Delta\\rho^2 P(Q)\\,S(Q) + \\mathrm{bkg}$, now with $S(Q) \\neq 1$.
$S(Q)$ is the Fourier transform of the pair correlation function $g(r)$:
"the form factor of the neighbours" (slide 24). The `hardsphere` model uses
the Percus–Yevick closure for particles that cannot overlap
([SasView `hardsphere`]("""
            + SASVIEW_MODEL.format('hardsphere')
            + """)). Two limits from the lecture to check against your fit:

- $S(0) = (1-\\eta)^4/(1+2\\eta)^2$. It falls from 0.92 at $\\eta = 0.01$ to
  0.09 at $\\eta = 0.3$. Use your fitted `volfraction` to predict how strongly
  the low-q intensity is suppressed.
- $Q^* \\approx 6/\\sigma$ with $\\sigma = 2\\times$`radius_effective`. Does this
  match the bump position you read off above?

*Physicists:* this is the same $S(Q)$ as for an atomic liquid, with each
colloidal particle playing the role of an atom (slide 24)."""
        ),
        code(interacting_fit),
        code(comparison),
        markdown(
            """## Exercise 3 — Interpret and challenge

1. What real-space correlation creates the peak?
2. Why might `radius_effective` differ from the form-factor radius?
3. What assumptions in the hard-sphere model could fail?
4. Try `radius_effective_mode='link_radius'`. Is that constraint supported by
   these data?

Notes for these questions:

- For question 2, think about what surrounds a real colloid: a stabilizer or
  surfactant shell, a hydration layer, or a counter-ion cloud. Charged
  particles repel over longer distances, and slide 24 notes that screened
  charges (Hayter–Penfold/MSA) produce peaks at much lower concentration.
  Is the shell visible to the *form factor* at the given contrast?
- For question 3, the hard-sphere model assumes monodisperse, spherical,
  purely repulsive particles. Attractive interactions *raise* $S(0)$ instead
  (slide 24), and the "decoupling" of $P(Q)$ and $S(Q)$ is an approximation
  for polydisperse or anisotropic particles.
- For question 4, `radius_effective_mode` is an argument of
  `set_structure_factor`, not of `set_param`. Build a new fitter and call
  `set_structure_factor('hardsphere', radius_effective_mode='link_radius')`,
  as on slide 24. Compare the residuals and reduced χ² with your free fit."""
        ),
        code(interpretation),
        markdown(
            """## Exit ticket

Sketch a decision tree for an unfamiliar curve: data checks, simplest form
factor, residual diagnosis, alternative shapes, q-range sensitivity, and only
then interaction models. Name one conclusion that SANS alone cannot establish.

Use the lecture's *Step 0* and *Fitting strategy* (slides 28 and 30) as the
root of your tree: know the composition, concentration, charge and likely
sizes first. Then use the model-free looks (Guinier, power-law slopes,
Kratky, P(r); slides 19, 20 and 25) before model fitting. Bring your tree to
the plenary discussion."""
        ),
        markdown(
            f"""## Further reading

- [SasView `hardsphere` model]({SASVIEW_MODEL.format('hardsphere')}): the
  Percus–Yevick structure factor used here.
- [SasView `hayter_msa` model]({SASVIEW_MODEL.format('hayter_msa')}): the
  Hayter–Penfold structure factor for charged particles mentioned on slide 24.
- J. S. Pedersen, Adv. Colloid Interface Sci. 70, 171 (1997): the section on
  concentrated systems and the decoupling approximation."""
        ),
    ]
    return notebook(cells)


def bayesian_notebook(solution: bool) -> dict[str, object]:
    """Create the Bayesian fitting extension notebook."""
    label = 'Solution' if solution else 'Student workbook'
    configure_model = (
        """fitter = SANSFitter()
fitter.load_data(str(filename))
fitter.set_model('sphere')
fitter.set_param('sld', value=4.2, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)
fitter.set_param('radius', value=50, min=30, max=90, vary=True)
fitter.set_param('scale', value=0.05, min=0.01, max=0.12, vary=True)
fitter.set_param('background', value=0.002, min=0, max=0.008, vary=True)
fitter.get_params()
"""
        if solution
        else """fitter = SANSFitter()
fitter.load_data(str(filename))

# Select a sphere model and inspect its parameter names. Keep both SLDs fixed,
# then give radius, scale, and background proper bounded uniform priors.
# Avoid bounds that are so broad that the short classroom chain cannot explore.

# YOUR CODE HERE
raise NotImplementedError('Configure the sphere model and its priors')
"""
    )
    run_sampler = (
        """np.random.seed(2026)
result = fitter.fit_bayesian(samples=1500, burn=100, thin=1, pop=10)
fitter.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# For a classroom-scale exploratory run, use:
#   samples=1500, burn=100, thin=1, pop=10
# Store the result and plot the best fit with normalized residuals.

# YOUR CODE HERE
raise NotImplementedError('Run DREAM and inspect the fitted curve')
"""
    )
    diagnostics = (
        """posterior = fitter.get_posterior()
print('Parameters:', posterior.labels)
print('Combined sample shape:', posterior.samples.shape)
for name in posterior.labels:
    low, high = posterior.ci_95[name]
    r_hat = posterior.diagnostics.get(name, {}).get('r_hat')
    r_hat_text = f'{r_hat:.3f}' if r_hat is not None else 'n/a'
    print(f'{name:>12}: 95% CI [{low:.6g}, {high:.6g}], R-hat={r_hat_text}')

fitter.plot_trace()
"""
        if solution
        else """posterior = fitter.get_posterior()
print('Parameters:', posterior.labels)
print('Combined sample shape:', posterior.samples.shape)

# YOUR CODE HERE
# Print each 95% credible interval and R-hat, then call `plot_trace()`.
# Identify any parameter for which the chains have not mixed well.
raise NotImplementedError('Inspect convergence and credible intervals')
"""
    )
    posterior_plots = (
        """fitter.plot_param_distribution('radius')
fitter.plot_posterior_pairs()
fitter.plot_param_correlations()
"""
        if solution
        else """# Use:
#   plot_param_distribution('radius')
#   plot_posterior_pairs()
#   plot_param_correlations()
# Explain one correlation physically in a Markdown cell.

# YOUR CODE HERE
raise NotImplementedError('Explore marginal and joint posterior structure')
"""
    )
    predictive = (
        """fitter.plot_posterior_predictive(style='band+draws', n_draws=40)
"""
        if solution
        else """# Plot a 95% posterior predictive band with 40 individual draws.
# Decide whether uncertainty is equally large across the measured q range.

# YOUR CODE HERE
raise NotImplementedError('Create and interpret a posterior predictive check')
"""
    )
    interpretation = (
        """interpretation = (
    'The radius is tightly constrained by the oscillation positions, while scale '
    'and background show stronger trade-offs with intensity normalization. The '
    'R-hat values remain above the convergence target, so this classroom chain '
    'must be lengthened and rechecked before reporting any credible intervals.'
)
print(interpretation)
"""
        if solution
        else """# Summarize what DREAM adds beyond the point estimate. Refer to the
# credible intervals, trace behavior, parameter correlations, and predictive band.
interpretation = 'Write your evidence-based interpretation here.'
print(interpretation)
"""
    )
    cells = [
        markdown(
            f"""# 04 — Bayesian fitting with DREAM ({label})

**Time:** 60–75 minutes (advanced extension)
**Question:** what can a posterior distribution tell us that a single best fit
cannot?

Earlier notebooks returned one optimum and a local uncertainty estimate. Here
you will sample a distribution of plausible parameter combinations using the
DREAM Markov chain Monte Carlo algorithm."""
        ),
        connection(
            """- Slide 32, *How can uncertainties help: posteriors, correlations and bounds*:
  this notebook produces exactly those three objects for the sphere from
  notebook 01.
- Slide 34, *Wrap-up for part 2*: "several of these parameters are
  degenerate ($\\phi \\leftrightarrow \\Delta\\rho \\leftrightarrow$ scale)". Here
  you will *see* a degeneracy as a tilted cloud in a pair plot.
- Slide 18's discussion: concentration and contrast move the curve up and
  down, while size moves it left and right. Keep that picture in mind when
  you predict the correlations below.
- Notebook 02's non-identifiability is the extreme case of what you measure
  here: a posterior that is long and flat in some direction."""
        ),
        code(SETUP),
        code(
            """filename = DATA / 'sphere_reference.csv'
values = np.genfromtxt(filename, delimiter=',', names=True)
q, intensity, uncertainty = values['Q'], values['I'], values['dI']
"""
        ),
        markdown(
            r"""## 1 — From bounds to priors

Bayes' theorem combines a likelihood with prior information:

$$
p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta).
$$

For this workshop, `set_param(..., min=a, max=b, vary=True)` defines a
**uniform prior** between `a` and `b`: every value inside is initially equally
plausible and values outside are forbidden. This is not merely an optimizer
safety rail. Bounds must therefore be scientifically defensible.

Configure the same sphere model as before. Keep known SLD values fixed and use
bounded priors for radius, scale, and background.

**Where the likelihood comes from.** If each measured point is Gaussian with
standard deviation `dI`, then
$p(D \\mid \\theta) \\propto \\exp(-\\chi^2(\\theta)/2)$, with the same $\\chi^2$ you
minimized in notebook 01. The best fit is the peak of this function. Sampling
maps out its whole shape. This is why correct error bars from data reduction
(slides 42–43) matter even more here than for a point estimate.

**Fixing the SLDs is itself a prior.** It is a delta-function prior that
breaks the $\\phi \\leftrightarrow \\Delta\\rho$ degeneracy of slide 34 by fiat.
Ask yourself how confident you really are in those numbers (slide 11's
caveats about density and exchangeable hydrogens)."""
        ),
        code(configure_model),
        markdown(
            """## 2 — Sampling with DREAM

`fit_bayesian` uses several controls:

- `samples`: approximate posterior sampling budget;
- `burn`: early generations discarded while chains approach high probability;
- `thin`: retain every nth draw;
- `pop`: population size multiplier for DREAM's parallel chains.

A larger budget is not automatically trustworthy. Chains must mix, explore the
same region, and be insensitive to starting values. The short run below is
chosen for class time and is **exploratory, not publication quality**.

DREAM is a population method, like the differential-evolution optimizer from
the notebook 01 extension. The difference is that it sometimes *accepts worse*
parameter sets, in proportion to their probability, so the chains end up
visiting regions in proportion to their posterior probability instead of
converging to a single point ([BUMPS optimizer guide: DREAM]("""
            + BUMPS_OPTIMIZERS
            + """#dream))."""
        ),
        code(run_sampler),
        markdown(
            """## 3 — Posterior data and convergence

`get_posterior()` returns:

- `labels` and a two-dimensional `samples` array;
- 68% and 95% credible intervals;
- separate `chains` where available;
- R-hat diagnostics comparing within-chain and between-chain variation.

R-hat near 1 is desirable. Values above roughly 1.05 are a warning to run
longer, revisit priors, or investigate a multimodal posterior. Trace plots
should look stationary and well mixed rather than drifting or separating."""
        ),
        code(diagnostics),
        markdown(
            """## 4 — Marginal distributions and correlations

A marginal distribution shows uncertainty in one parameter after averaging
over the others. A pair plot and correlation matrix reveal combinations that
the data constrain together. For example, particle contrast, concentration,
and scale can compensate for one another.

**Predict before plotting.** With both SLDs fixed, the plateau height is
$I(0) \\propto \\mathrm{scale}\\cdot V \\propto \\mathrm{scale}\\cdot r^3$ (slide 18).
If the data pin $I(0)$ tightly, which sign of correlation do you expect between
`scale` and `radius`? Is it strong, given that the *positions* of the minima
pin the radius independently (slide 21)? Which parameter should be most
correlated with `background`, and in which q region do they compete?"""
        ),
        code(posterior_plots),
        markdown(
            """## 5 — Posterior predictive checking

Parameter intervals are not the final scientific target. A posterior
predictive check propagates sampled parameters through the scattering model.
The resulting band asks whether plausible parameter sets reproduce the
observations across q, and where predictions remain uncertain.

Look especially at the form-factor minima and at the highest q, where the
signal approaches the background. Relate what you see to the lecture's point
that resolution and polydispersity fill in minima (slide 21). If those effects
were present but missing from the model, would the posterior band be wider,
narrower, or simply in the wrong place?"""
        ),
        code(predictive),
        markdown(
            """## 6 — Interpretation

Write a short report addressing:

1. Which parameter is best constrained, and what feature of the curve does it control?
2. Which parameters are correlated, and why?
3. Do the traces and R-hat values support convergence?
4. Where is the predictive band widest?
5. What would you change before publishing these intervals?"""
        ),
        code(interpretation),
        markdown(
            """## Optional export

The raw chain can be analyzed with other tools:

```python
posterior.save_posterior_csv('sphere_posterior.csv')
fitter.save_results('sphere_bayesian_fit.csv')
```

Try changing one prior range or restricting q. Rerun the sampler and explain
whether the posterior changed because of the data or because of the prior.

*Suggested experiment:* restrict q to `0.004–0.025 Å⁻¹`, as in notebook 02,
with `fitter.set_q_range(qmin=0.004, qmax=0.025)`, and rerun. Which posterior
widens most, and does the pair plot now show the Guinier-only degeneracy
between size and scale?"""
        ),
        markdown(
            f"""## Further reading

- [BUMPS optimizer guide]({BUMPS_OPTIMIZERS}): DREAM, differential
  evolution, and how BUMPS reports uncertainties.
- J. A. Vrugt, *Markov chain Monte Carlo simulation using the DREAM software
  package*, Environ. Model. Softw. 75, 273 (2016): the algorithm behind
  `fit_bayesian`.
- A. Vehtari et al., *Rank-normalization, folding, and localization: an improved
  R-hat for assessing convergence of MCMC*, Bayesian Anal. 16, 667 (2021): why
  R-hat thresholds have tightened from 1.1 to about 1.01.
- {JACQUES}: on reporting uncertainty and model ambiguity in SAS results."""
        ),
    ]
    return notebook(cells)


def write_notebooks() -> None:
    """Write student and solution notebook sets."""
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        '00_start_here.ipynb': start_notebook,
        '01_fit_a_sphere.ipynb': sphere_notebook,
        '02_model_detective.ipynb': detective_notebook,
        '03_when_particles_interact.ipynb': interactions_notebook,
        '04_bayesian_uncertainty.ipynb': bayesian_notebook,
    }
    for filename, builder in builders.items():
        (NOTEBOOK_DIR / filename).write_text(
            json.dumps(builder(False), indent=1) + '\n', encoding='utf-8'
        )
        (SOLUTION_DIR / filename).write_text(
            json.dumps(builder(True), indent=1) + '\n', encoding='utf-8'
        )


def check_notebooks() -> None:
    """Validate student tasks and execute solutions without changing source notebooks."""
    for notebook_path in sorted(NOTEBOOK_DIR.glob('*.ipynb')):
        notebook_data = json.loads(notebook_path.read_text(encoding='utf-8'))
        if notebook_path.name != '00_start_here.ipynb':
            sources = '\n'.join(
                ''.join(cell.get('source', [])) for cell in notebook_data.get('cells', [])
            )
            if 'YOUR CODE HERE' not in sources or 'NotImplementedError' not in sources:
                raise ValueError(f'{notebook_path} does not contain required student tasks')

    environment = dict(os.environ)
    environment['MPLBACKEND'] = 'Agg'
    source_path = str(ROOT / 'src')
    existing_pythonpath = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = (
        f'{source_path}{os.pathsep}{existing_pythonpath}' if existing_pythonpath else source_path
    )
    with tempfile.TemporaryDirectory(prefix='sans-workshop-') as output_dir:
        for notebook_path in sorted(SOLUTION_DIR.glob('*.ipynb')):
            print(f'Executing {notebook_path.relative_to(ROOT)}')
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'jupyter',
                    'nbconvert',
                    '--to',
                    'notebook',
                    '--execute',
                    '--ExecutePreprocessor.timeout=300',
                    '--output-dir',
                    output_dir,
                    str(notebook_path),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )


def main() -> None:
    """Generate workshop assets and optionally execute all solutions."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    write_datasets()
    if not args.data_only:
        write_notebooks()
    if args.check:
        check_notebooks()


if __name__ == '__main__':
    main()

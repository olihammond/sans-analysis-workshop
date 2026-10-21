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
from generate_mystery_data import write_mystery_datasets
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


def answer(solution: bool, model_answer: str, prompt: str) -> dict[str, object]:
    """Create a written-answer cell: a prompt for students, a model answer in solutions."""
    if solution:
        return markdown(f'**Model answer.** {model_answer}')
    return markdown(
        '**Your answer** *(double-click this cell to edit it, then press Shift + Enter)*'
        f'\n\n{prompt}'
    )


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
| Convert q to a real-space length | 9 *Scattering geometry*, 10 *Real and Q-space* |
| Read the shape of a log–log curve | 21 *Guinier analysis*, 22 *Power law slopes* |
| Know what "reduced data" has already been through | 41 *Q-range and resolution*, 43–46 *Data reduction* … *When data look wrong* |

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

Write a one-sentence prediction for each in the answer cell below.

*Hints from the lecture:* intensity typically falls by several orders of
magnitude over the measured range, and power laws $I \\propto Q^{-m}$ become
straight lines of slope $-m$ on log–log axes (slide 22). Size information sits
where $Q \\sim 2\\pi/d$ for the whole particle, i.e. at the low-q end (slide 21)."""
        ),
        answer(
            solution,
            """Intensity falls by several orders of magnitude, and power laws become
straight lines on log–log axes, so both axes are logarithmic. `dI` is the
one-standard-deviation uncertainty of each point; it tells a fit how much to
trust that point. The overall particle size is encoded at low q, where the
curve bends over from the plateau (the Guinier region).""",
            'Write a one-sentence prediction for each of the three questions.',
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
  the curve flattens. For a solid sphere $R_g = \\sqrt{3/5}\\,R$ (slide 21).
- **Oscillations (form factor).** Sharp minima mean nearly identical particles.
  For a sphere the first minimum sits at $Q_1 R \\approx 4.49$ (slide 23).
- **High-q decay.** Smooth, sharp interfaces give the Porod envelope
  $I \\propto Q^{-4}$ (slide 22).
- **Flat floor at the highest q.** Background, mostly incoherent scattering
  from hydrogen, which is $Q$-independent to a good approximation (slide 15).

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
   the master equation (slide 20) lets you check the fitted volume fraction.
   These workshop data are in *arbitrary units*, so the fitted `scale`
   parameter absorbs the volume fraction *and* any calibration factor.
2. **Reduction artefacts look like physics.** Slide 46 (*When data look
   wrong…*) is a useful diagnostic when a residual pattern puzzles you: a
   discrepancy at high q only suggests a flat or weak-signal term (cell,
   background); a constant factor everywhere suggests normalization, thickness,
   transmission or calibration; a change of shape suggests inconsistent
   subtraction. The synthetic data here are clean, but real data rarely are.
3. **Resolution.** Reduced files from a real instrument often carry a fourth
   column, `dQ`, the q resolution of each point (slide 41). When it is present,
   `sans_fitter` (like SasView) smears the model with it automatically, so sharp
   minima are compared fairly. The workshop files have no `dQ` column, so no
   smearing is applied."""
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
    sphere_function = f"""def sphere_amplitude(q, radius):
    # Sphere form-factor amplitude F(q), written exactly as in the equation above.
    qr = q * radius
    return 3 * (np.sin(qr) - qr * np.cos(qr)) / qr**3


# The measured intensity follows the amplitude squared, P(q) = F(q)^2.
# trial_radius is in Å. Change this number, then rerun this cell and the plot
# below, until the minima of the model line up with the minima of the data.
trial_radius = {58.0 if solution else 50.0}
trial_shape = sphere_amplitude(q, trial_radius) ** 2
"""
    model_setup = (
        """# SasModels models are chosen by name, written as a lower-case string.
# 'sphere' is a homogeneous sphere. All names are listed at
# https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html
model_name = 'sphere'
fitter.set_model(model_name)

# get_params() prints every parameter of the chosen model: its current value,
# its bounds, and whether it will be fitted (the Vary column).
fitter.get_params()
"""
        if solution
        else """# `set_model` takes a SasModels model name as a string, in quotes.
# Names are lower case and match the SasView model list, e.g. 'cylinder':
# https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html
# YOUR CODE HERE: replace None with the model name for a homogeneous sphere.
model_name = None
if model_name is None:
    raise NotImplementedError('Replace None with a model name in quotes, then rerun')

fitter.set_model(model_name)

# `get_params` lists every parameter supplied by that model.
fitter.get_params()
"""
    )
    params = (
        """# Contrast: both SLDs (in 10^-6 Å^-2) are known from the sample composition,
# so they are fixed with vary=False and the optimizer will not change them.
fitter.set_param('sld', value=4.2, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)

# Fitted parameters: value = starting guess, min/max = physically sensible limits,
# vary=True = let the optimizer adjust it.
# Radius in Å: start near the estimate from the first minimum.
fitter.set_param('radius', value=50, min=20, max=100, vary=True)
# Scale: the overall intensity level (volume fraction x calibration here).
fitter.set_param('scale', value=0.05, min=0.001, max=0.2, vary=True)
# Background: the flat level at high q. It cannot be negative.
fitter.set_param('background', value=0.002, min=0, max=0.01, vary=True)
"""
        if solution
        else """# One fixed parameter is done for you. `vary=False` keeps this
# independently known value unchanged during the fit.
fitter.set_param('sld', value=4.2, vary=False)

# YOUR CODE HERE: add four more set_param lines, in the same form as above
# and as in the box "Changing parameters".
#   1. Fix 'sld_solvent' at 1.0.
#   2. 'radius', 'scale' and 'background': give each a value, min and max,
#      and vary=True. Use the table above to choose the numbers.
# When your lines are in place, delete the `raise` line below and run the cell.
raise NotImplementedError('Add the set_param lines, then delete this line')
"""
    )
    fit_call = (
        """# Run a bounded least-squares fit, starting from the values set above.
# The returned dictionary, stored in `result`, holds the fitted parameters.
result = fitter.fit(engine='lmfit', method='least_squares')

# Top panel: data and model. Bottom panel: normalized residuals,
# (data - model) / dI, i.e. the misfit in units of the error bar.
fitter.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# YOUR CODE HERE: two lines.
#   1. Fit with  result = fitter.fit(engine='lmfit', method='least_squares')
#   2. Plot with fitter.plot_results(show_residuals=True, log_scale=True)
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Add the fit and plot lines, then delete this line')
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
            """- Slide 20, *The SANS master equation*: the model you fit is this equation
  with $S(Q)=1$ (dilute particles).
- Slide 23, *The form factor, P(Q)*: the phase-map picture of why a sphere
  gives minima at $QR \\approx 4.49, 7.73, \\ldots$ Rewatch the
  [phase-map video]("""
            + PHASE_MAP_VIDEO
            + """) if the formula below feels abstract.
- Slides 30 and 32, *Step 0: understand your sample* and *Fitting strategy*:
  where defensible starting values and bounds come from.
- Slide 33, *Is my fit good or not?*: what this notebook asks you to judge."""
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
master equation on slide 20,
$I(Q) = \phi\,V_p\,(\Delta\rho)^2\,P(Q)\,S(Q) + \mathrm{bkg}$, where
$P(Q) = F(Q)^2$ is normalized so that $P(0)=1$. For dilute particles $S(Q)=1$,
and on an absolute scale `scale` *is* the volume fraction $\phi$. Because the
workshop data are in arbitrary units, `scale` here also absorbs a calibration
factor. That is one face of the $\phi \leftrightarrow \Delta\rho \leftrightarrow
\mathrm{scale}$ degeneracy from the lecture's part 2 wrap-up (slide 35). We fix
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
rule of thumb (slide 23): the first zero of $F(qr)$ is at $qr \\approx 4.493$,
so $r \\approx 4.49/q_1$.

- The lecture's worked example used $R = 80$ Å, with minima at
  $Q = 0.0562, 0.0966, 0.136$ Å⁻¹. Are the data minima to the left or right of
  those? Is this sphere bigger or smaller than 80 Å?
- Why are the data minima not exactly zero? Think about noise, the background,
  and, for real samples, polydispersity and instrumental resolution (slide 23's
  dQ/Q panels and slide 41).

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

The first two steps are done for you below. You will then choose the model,
set the parameters and run the fit yourself, using only the commands listed
here. Each task cell says exactly which lines to add."""
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

Keep both SLDs fixed. Choose and justify a starting value and bounds for
radius, scale and background, using the table below. There is no single right
answer; defensible numbers are what matter.

**Where do defensible numbers come from?** Slide 33 says it directly: *start
from the data, not a guess*. The lecture's *Step 0* (slide 30)
is to know what you put in the beam. In practice:

| Parameter | Lecture source for a starting value | Sensible bounds reflect… |
|---|---|---|
| `sld`, `sld_solvent` | Composition and density via an SLD calculator (slide 12) | fixed: known independently |
| `radius` | First minimum, $r \\approx 4.49/q_1$ (slide 23), or Guinier, $R = \\sqrt{5/3}\\,R_g$ (slide 21) | synthesis, TEM, or DLS knowledge |
| `scale` | Height of the low-q plateau, $I(0) \\propto \\phi V \\Delta\\rho^2$ (slide 20) | concentration and calibration |
| `background` | Level of the flat high-q floor (slide 15) | must be ≥ 0 and below the data |

The SLD values used here are 4.2 for the particle and 1.0 for the solvent, in
units of 10⁻⁶ Å⁻², the "SLD units" of slide 12."""
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
00). As slide 33 puts it, good residuals scatter randomly around 0, mostly
within ±2, and change sign roughly every other point (about $N/2$ sign changes
for $N$ points). Structure in the residuals is the diagnostic, more than the χ²
number ([SasView: *Assessing fit quality*]("""
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

**Checklist for "is my fit good?" (slide 33).**

1. *Residuals:* scattered around zero with no runs or waves? A wave that
   follows the form-factor oscillations means the shape or size distribution
   is wrong, not the noise.
2. *Statistic:* is the reduced χ², $\\chi^2_\\nu = \\chi^2/\\nu$ with
   $\\nu = N-p$, within about $1 \\pm \\sqrt{2/\\nu}$ (slide 33)? Much larger
   means a missing ingredient or underestimated `dI`. Much smaller means
   overestimated `dI` or overfitting. A χ²ν of 1 with a long run of positive
   residuals is still a bad fit: χ² mixes the model and the error bars.
3. *Bounds:* a parameter pinned at a bound has not been determined. The
   bound has decided it.
4. *Plausibility:* does the radius agree with your model-free estimate from
   the first minimum? Is the background below the lowest data points?

**Systematics a single monodisperse sphere ignores:** a size distribution
([SasView polydispersity]("""
            + SASVIEW_POLYDISPERSITY
            + """)) and instrumental smearing ([SasView resolution]("""
            + SASVIEW_RESOLUTION
            + """)). Both fill in the minima (slide 23 shows 0, 5 and 10 % dQ/Q). The
lecture also covered reduction errors in transmission, empty-cell subtraction
and thickness (slides 44 and 46), and interparticle interactions, $S(Q) \\neq 1$
(notebook 03). These synthetic data were generated without any of them. That is
why the minima are so sharp. Say which one you would expect to matter most for a
real measurement, and why."""
        ),
        code(
            """# result['parameters'][name]['formatted'] is the fitted value with its
# uncertainty, written as text.
for name in ['radius', 'scale', 'background']:
    print(name, result['parameters'][name]['formatted'])

# Reduced chi-squared: the sum of squared normalized residuals divided by
# nu = (number of data points - number of fitted parameters).
n_points = np.count_nonzero(~fitter.data.mask)
nu = n_points - 3
print(f"Reduced chi-squared: {result['chisq'] / nu:.2f} "
      f'(expected 1 ± {np.sqrt(2 / nu):.2f} for nu = {nu})')

# The two residual checks from slide 33, using the data and fitted model.
_, data_I, data_dI, model_I = fitter._fit_arrays()
residuals = (data_I - model_I) / data_dI
sign_changes = np.count_nonzero(np.diff(np.sign(residuals)))
print(f'Residuals within ±2: {np.mean(np.abs(residuals) < 2):.0%}')
print(f'Sign changes: {sign_changes} (random scatter gives about {n_points // 2})')
"""
        ),
        answer(
            solution,
            """The fitted radius is about 58.0 ± 0.02 Å, matching the first-minimum
estimate $4.49/q_1$. The residuals scatter randomly around zero with no wave
following the oscillations: 97 % lie within ±2 and they change sign 49 times,
close to the $N/2 \\approx 45$ expected for random scatter. The reduced χ² of
0.76 is slightly below the expected 1 ± 0.15, so the error bars are, if
anything, a little generous. No parameter sits on a bound. The model ignores polydispersity and instrumental resolution. For a real
sample these would fill in the minima, and the quoted ± 0.02 Å would be far too
optimistic.""",
            'Record the four points listed above and interpret the fit in two or three sentences.',
        ),
        markdown(
            """## Extension — Bad starting values

What happens if the starting radius is poor, say 15 Å, with broad bounds? Does
the local optimizer reach the same answer? Compare it with BUMPS differential
evolution, `fitter.fit(engine='bumps', method='de', steps=200)`. What does this
reveal about optimization versus inference?

*Why this happens:* as slide 33 warns, a fit started far from the answer can be
trapped in a local minimum. The χ² surface of a form factor is full of them.
A sphere of 15 Å puts its first minimum near $4.49/15 \\approx 0.3$ Å⁻¹, so
a downhill search starting there can lock onto the wrong oscillation.
[Differential evolution]("""
            + BUMPS_OPTIMIZERS
            + """#differential-evolution) keeps a population of candidates spread over
the whole prior box, so it is far less sensitive to the start. Neither method
tells you *how well* the radius is determined. That question, the theme of slide
34 (*Deeper into uncertainty*), is taken up in notebook 04.

Remember the χ² note above when you compare the two engines' numbers.

The cell below runs both experiments for you. Read the two printed radii and
compare the two residual plots."""
        ),
        code(
            """# 1. The same local optimizer, from a deliberately poor start of 15 Å.
fitter.set_param('radius', value=15, min=10, max=200, vary=True)
local_result = fitter.fit(engine='lmfit', method='least_squares')
fitter.plot_results(show_residuals=True, log_scale=True)

# 2. Differential evolution, a population search over the whole min-max range.
global_result = fitter.fit(engine='bumps', method='de', steps=200)
fitter.plot_results(show_residuals=True, log_scale=True)

print('Local optimizer radius:       ', local_result['parameters']['radius']['formatted'])
print('Differential evolution radius:', global_result['parameters']['radius']['formatted'])
"""
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
    conclusion = answer(
        solution,
        """The ellipsoid is preferred on the full q range: its residuals lose the
systematic oscillatory pattern left by the sphere, and its fitted radii are
physically reasonable. After truncation to the Guinier region the two models are
hard to distinguish, showing that the fit quality depends on the information
measured, not only on the optimizer. A measurement to higher q, or of aligned
particles, would make the distinction stronger.""",
        'Write your referee-style verdict: one reason from the residuals, one from '
        'parameter plausibility, and one follow-up measurement.',
    )
    sphere_fit = """def fit_sphere(qmin=None, qmax=None):
    # The notebook-01 workflow wrapped in a function, so it can be rerun
    # on a different q range. qmin and qmax are in Å^-1; None means "use all data".
    fitter = SANSFitter()
    fitter.load_data(str(filename))
    if qmin is not None or qmax is not None:
        fitter.set_q_range(qmin=qmin, qmax=qmax)
    fitter.set_model('sphere')
    # Same contrast as before, so the SLDs stay fixed.
    fitter.set_param('sld', value=4.2, vary=False)
    fitter.set_param('sld_solvent', value=1.0, vary=False)
    # Broad bounds: we do not yet know whether a sphere is even the right shape.
    fitter.set_param('radius', value=45, min=10, max=180, vary=True)
    fitter.set_param('scale', value=0.04, min=0.001, max=0.2, vary=True)
    fitter.set_param('background', value=0.002, min=0, max=0.02, vary=True)
    return fitter, fitter.fit(engine='lmfit', method='least_squares')


# Fit the full q range and show the residuals.
sphere, sphere_result = fit_sphere()
sphere.plot_results(show_residuals=True, log_scale=True)
"""
    ellipsoid_fit = (
        """# A new, separate fitter, so the sphere fit above is left untouched.
ellipsoid = SANSFitter()
ellipsoid.load_data(str(filename))
ellipsoid.set_model('ellipsoid')

# The ellipsoid has two radii: radius_polar along its symmetry axis and
# radius_equatorial perpendicular to it. Equal radii would make it a sphere.
ellipsoid.set_param('sld', value=4.2, vary=False)
ellipsoid.set_param('sld_solvent', value=1.0, vary=False)
ellipsoid.set_param('radius_polar', value=100, min=15, max=220, vary=True)
ellipsoid.set_param('radius_equatorial', value=40, min=10, max=120, vary=True)
ellipsoid.set_param('scale', value=0.04, min=0.001, max=0.2, vary=True)
ellipsoid.set_param('background', value=0.002, min=0, max=0.02, vary=True)
ellipsoid.get_params()

ellipsoid_result = ellipsoid.fit(engine='lmfit', method='least_squares')
ellipsoid.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# A new, separate fitter, so the sphere fit above is left untouched.
ellipsoid = SANSFitter()
ellipsoid.load_data(str(filename))

# YOUR CODE HERE, following the same steps as for the sphere:
#   1. ellipsoid.set_model('ellipsoid'), then ellipsoid.get_params() to see
#      its parameters (it has radius_polar and radius_equatorial).
#   2. set_param lines: fix both SLDs as before; give the two radii, scale and
#      background starting values, bounds and vary=True.
#   3. ellipsoid_result = ellipsoid.fit(engine='lmfit', method='least_squares')
#   4. ellipsoid.plot_results(show_residuals=True, log_scale=True)
# Keep the names `ellipsoid` and `ellipsoid_result`: later cells use them.
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Configure and fit the ellipsoid, then delete this line')
"""
    )
    short_range = (
        """# Ellipsoid: the same setup as before, restricted to the same q window.
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
        else """# Ellipsoid on the restricted window.
ellipsoid_short = SANSFitter()
ellipsoid_short.load_data(str(filename))
ellipsoid_short.set_q_range(qmin=0.004, qmax=0.025)

# YOUR CODE HERE: repeat your ellipsoid setup from Exercise 1 for
# `ellipsoid_short` (set_model, set_param lines), then
#   ellipsoid_short_result = ellipsoid_short.fit(engine='lmfit', method='least_squares')
#   ellipsoid_short.plot_results(show_residuals=True, log_scale=True)
# Keep the name `ellipsoid_short_result`: the next cells use it.
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Fit the ellipsoid on the restricted range, then delete this line')
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
            """- Slide 24, *Welcome to the SANS zoo*: many shapes share a plateau and a
  decay and differ in oscillation depth and mid-q slope. Browse the
  [SasView model marketplace]("""
            + SASVIEW_MARKETPLACE
            + """) for the full zoo.
- Slide 25, *Name that shape!*: panels B and C "look alike over a narrow
  Q-window", and an extended Q-range shows the difference. This notebook is
  that statement made quantitative.
- Slide 21, *Guinier analysis*: the $R_g$ table. Every shape looks the same at
  low Q apart from $I(0)$ and $R_g$.
- Slide 34, *Deeper into uncertainty*: "choosing a model is also a prior. When
  two models fit equally well, physics and sample knowledge decide, not χ²."
  Bring your figures from this notebook to the plenary discussion.
- Slide 50, *Polymers*: a sphere, a coil and a rod with the *same* $R_g$ cannot
  be told apart at low Q. Guinier only sees size; the shape lives at higher Q."""
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

The lecture's fitting strategy (slide 32) starts with reading the data, not
with the optimizer. Compare this curve with `sphere_reference.csv` from
notebook 01:

- Are the form-factor minima as deep and sharp? Shallow, washed-out minima
  can mean polydispersity or smearing, but also **anisotropy**: an
  orientationally averaged ellipsoid has a different dimension in every
  direction, so its minima fall at different q and blur together.
- Where does the plateau bend over, and what slope follows it (slide 22)?
  A very long object would show an extended $Q^{-1}$ region. A moderately
  elongated one shows only a hint of it.
- Notice that these data are noisier than in notebook 01. How might that
  affect your ability to tell two models apart?"""
        ),
        markdown(
            """## Exercise 1 — Competing explanations

The first cell repeats the notebook-01 workflow for a sphere, wrapped in a
function so it can be rerun later on part of the q range. The SLDs stay fixed;
only radius, scale and background vary. Compare residual patterns, not just
the numerical objective."""
        ),
        code(sphere_fit),
        markdown(
            """### A competing model

Set up the ellipsoid with its own `SANSFitter`, so the two fits stay
independent. After `set_model('ellipsoid')`, `get_params()` shows its two size
parameters. Decide which parameters to fix and which to vary. Hint: if both
radii start at the same value, the optimizer begins from a sphere; is that a
fair start for testing an elongated particle?"""
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

Repeat both fits using only `0.004 ≤ q ≤ 0.025 Å⁻¹`. The sphere fit is done
for you; set up the ellipsoid fit yourself. Predict what will happen before
running the cells. Which shape features have disappeared from view?

*Before you run:* convert the window to real space with $d \\approx 2\\pi/Q$,
as in notebook 00. What is the smallest distance still resolved, and how does
it compare with the particle dimensions you just fitted?"""
        ),
        code(
            """# Sphere: reuse fit_sphere from Exercise 1, now with only 0.004 <= q <= 0.025 Å^-1.
sphere_short, sphere_short_result = fit_sphere(qmin=0.004, qmax=0.025)
sphere_short.plot_results(show_residuals=True, log_scale=True)
print('Restricted sphere radius:', sphere_short_result['parameters']['radius']['formatted'])
"""
        ),
        code(short_range),
        markdown(
            """### Why the models become indistinguishable

The lecture called the Guinier law *universal*: at low Q every shape gives
$I(Q) \\approx I(0)\\exp(-Q^2R_g^2/3)$, which is valid for $QR_g \\lesssim 1.3$
(slide 21). In that region the data carry essentially **two numbers**, $I(0)$
and $R_g$. Any model that can reproduce those two numbers fits equally well.

The cell below tests this with your fitted parameters, using
$R_g^2 = (2R_e^2 + R_p^2)/5$ for a spheroid with equatorial radius $R_e$ and
polar radius $R_p$, and $R_g^2 = \\tfrac{3}{5}R^2$ for a sphere. Look at the
printed numbers and answer:

1. How close are the two restricted-range $R_g$ values?
2. Where does $Q_{\\max}R_g$ sit relative to the Guinier limit of 1.3?
3. Is the equal-$R_g$ sphere radius close to the radius the restricted sphere
   fit found?

This is **non-identifiability**: the data are compatible with a family of
parameter sets, so the optimizer's single answer is not a measurement of
shape. Notebook 04 shows the same idea as correlated posterior distributions."""
        ),
        code(
            """def value(result, name):
    # The fitted number for one parameter.
    return result['parameters'][name]['value']


def rg_spheroid(r_equatorial, r_polar):
    return np.sqrt((2 * r_equatorial**2 + r_polar**2) / 5)


def rg_sphere(radius):
    return np.sqrt(3 / 5) * radius


rg_sphere_short = rg_sphere(value(sphere_short_result, 'radius'))
rg_ellipsoid_short = rg_spheroid(
    value(ellipsoid_short_result, 'radius_equatorial'),
    value(ellipsoid_short_result, 'radius_polar'),
)
rg_ellipsoid_full = rg_spheroid(
    value(ellipsoid_result, 'radius_equatorial'), value(ellipsoid_result, 'radius_polar')
)

print(f'1. Restricted range: Rg(sphere) = {rg_sphere_short:.1f} Å, '
      f'Rg(ellipsoid) = {rg_ellipsoid_short:.1f} Å')
print(f'2. Qmax * Rg = 0.025 * {rg_ellipsoid_full:.1f} = {0.025 * rg_ellipsoid_full:.2f}')
print(f'3. Sphere with the full-range ellipsoid Rg: R = {np.sqrt(5 / 3) * rg_ellipsoid_full:.1f} Å; '
      f'restricted sphere fit: R = {value(sphere_short_result, "radius"):.1f} Å')
"""
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
  $Q_{\\max}$ (slides 9 and 39–41). Ask which q range the shape information
  lives in.
- **Change the contrast.** H/D variation (slides 14 and 16) changes $\\Delta\\rho$
  and so constrains the scale–contrast degeneracy, but does contrast alone
  change the *shape* of the curve for a homogeneous particle?
- **Complementary techniques.** Slide 7 asked when SANS and TEM would disagree
  about a size. Here, what could a micrograph settle that SANS cannot, and
  what could it get wrong?
- **Field or flow alignment.** Anisotropic objects give anisotropic 2D
  patterns when aligned (compare slide 29): a radially averaged 1D curve has
  thrown that information away."""
        ),
        conclusion,
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
    interpretation = answer(
        solution,
        """A form factor alone cannot reproduce the suppressed low-q intensity and the
correlation peak: changing radius or scale moves the whole curve, not one region.
The hard-sphere structure factor describes excluded-volume correlations between
neighbouring particles. The peak comes from the typical nearest-neighbour
distance, about twice `radius_effective`. The effective radius exceeds the
form-factor radius because the interaction range can include a solvation or
stabilizer layer that is invisible at this contrast. Linking the two radii makes
χ² several times worse, so these data do not support that constraint. The fitted
volume fraction
is model-dependent: charge, attraction, polydispersity, or anisotropy would
change it. A dilution series would confirm the interpretation, since the peak
should weaken and the low-q suppression disappear on dilution.""",
        'Explain what changed physically, not only statistically, and answer the four '
        'questions above.',
    )
    dilute_fit = (
        """# The simplest plausible explanation: dilute spheres with no interactions,
# set up exactly as in notebook 01.
dilute = SANSFitter()
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
        else """dilute = SANSFitter()
dilute.load_data(str(filename))

# YOUR CODE HERE: the simplest plausible explanation, a dilute sphere,
# with the same steps as in notebook 01:
#   1. dilute.set_model(...)
#   2. set_param lines: fix both SLDs (4.2 and 1.0); give radius, scale and
#      background a value, bounds and vary=True.
#   3. dilute_result = dilute.fit(engine='lmfit', method='least_squares')
#   4. dilute.plot_results(show_residuals=True, log_scale=True)
# Keep the names `dilute` and `dilute_result`: later cells use them.
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Build and fit the dilute-sphere model, then delete this line')
"""
    )
    interacting_fit = (
        """# Same sphere form factor as above...
interacting = SANSFitter()
interacting.load_data(str(filename))
interacting.set_model('sphere')
interacting.set_param('sld', value=4.2, vary=False)
interacting.set_param('sld_solvent', value=1.0, vary=False)
interacting.set_param('radius', value=40, min=15, max=90, vary=True)
interacting.set_param('scale', value=0.05, min=0.001, max=0.2, vary=True)
interacting.set_param('background', value=0.002, min=0, max=0.02, vary=True)

# ...now multiplied by a hard-sphere structure factor S(q).
interacting.set_structure_factor('hardsphere')
# Two new parameters appear: the interaction distance and the volume fraction.
# A volume fraction above ~0.55 is impossible for hard spheres, hence the bound.
interacting.set_param('radius_effective', value=45, min=20, max=100, vary=True)
interacting.set_param('volfraction', value=0.2, min=0.01, max=0.55, vary=True)
interacting.get_params()

interacting_result = interacting.fit(engine='lmfit', method='least_squares')
interacting.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """interacting = SANSFitter()
interacting.load_data(str(filename))

# YOUR CODE HERE:
#   1. The same sphere setup as `dilute` above (set_model and set_param lines).
#   2. interacting.set_structure_factor('hardsphere')
#   3. interacting.get_params() to see the new parameters.
#   4. set_param lines for 'radius_effective' and 'volfraction' with vary=True.
#      A hard-sphere volume fraction cannot exceed about 0.55.
#   5. interacting_result = interacting.fit(engine='lmfit', method='least_squares')
#   6. interacting.plot_results(show_residuals=True, log_scale=True)
# Keep the name `interacting_result`: later cells use it.
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Add, configure, and fit the structure factor, then delete this line')
"""
    )
    comparison = """# Fit statistic for each model: lower is better, but only residuals and
# physical plausibility make it a scientific conclusion.
print('Form factor only:      ', dilute_result['chisq'])
print('With structure factor: ', interacting_result['chisq'])

# The fitted particle radius, interaction radius, and volume fraction.
for name in ['radius', 'radius_effective', 'volfraction']:
    print(name, interacting_result['parameters'][name]['formatted'])
"""
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
            """- Slide 10, *Real and Q-space*: "the blob sets the envelope, the arrangement
  sets the sharp structure". This is the $P(Q)\\cdot S(Q)$ split.
- Slide 26, *The structure factor, S(Q)*: hard-sphere repulsion suppresses
  low-Q intensity and creates a correlation peak near $Q^* \\approx 2\\pi/\\sigma$.
- Slide 21, *Guinier analysis*: a low-Q **downturn** is the warning sign for
  repulsive interactions. An upturn suggests aggregation.
- Slide 25, *Name that shape!*, panel D: concentrated spheres. Dilute the
  sample, and if the peak disappears it was $S(Q)$.
- Slide 51, *Self-assembly*: real micelles are fitted with exactly this
  combination, core–shell $P(Q)$ × hard-sphere $S(Q)$."""
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
   slide 21 say each one means?
2. Find the q position of the bump, $Q^*$. Convert it to a real-space
   distance, $2\\pi/Q^*$. Is that distance closer to a particle *radius*, a
   *diameter*, or neither? (Slide 26: the first peak sits near
   $Q^* \\approx 6/\\sigma$, where $\\sigma = 2R_{hs}$ is the contact distance.)
3. Is the bump a feature of one particle or of the arrangement of many?
   Name the experiment that would tell you (slide 10's discussion prompt)."""
        ),
        markdown(
            """## Exercise 1 — Deliberately incomplete model

Using the same steps as in notebook 01, fit the simplest plausible model, a
dilute sphere. Locate the structured residual region. Why can changing
radius or scale not independently repair it?

*Hint:* the lecture's "four knobs" (slide 20). Concentration, contrast and
scale only move the whole curve **up or down**. Size moves features **left or
right** and also changes $I(0) \\propto V$. Neither move can lower the
intensity at low q while raising it at $Q^*$. Also predict what a Guinier
analysis of these data would give for the size (slide 26: "Guinier analysis on
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

Fit both new parameters freely. A third new parameter,
`radius_effective_mode`, only selects how the effective radius is calculated;
leave it fixed.

**What is being multiplied in.** This is the full master equation of slide 20,
$I(Q) = \\phi V \\Delta\\rho^2 P(Q)\\,S(Q) + \\mathrm{bkg}$, now with $S(Q) \\neq 1$.
$S(Q)$ is the Fourier transform of the pair correlation function $g(r)$:
"the form factor of the neighbours" (slide 26). The `hardsphere` model uses
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
colloidal particle playing the role of an atom (slide 26)."""
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
  particles repel over longer distances, and slide 26 notes that screened
  charges (Hayter–Penfold/MSA) produce peaks at much lower concentration.
  Is the shell visible to the *form factor* at the given contrast?
- For question 3, the hard-sphere model assumes monodisperse, spherical,
  purely repulsive particles. Attractive interactions *raise* $S(0)$ instead
  (slide 26), and the "decoupling" of $P(Q)$ and $S(Q)$ is an approximation
  for polydisperse or anisotropic particles.
- For question 4, the cell below repeats the fit with the interaction radius
  tied to the particle radius, as on slide 26. Compare its residuals and χ²
  with the free fit above."""
        ),
        code(
            """# Same model, but radius_effective is forced to equal the particle radius.
linked = SANSFitter()
linked.load_data(str(filename))
linked.set_model('sphere')
linked.set_param('sld', value=4.2, vary=False)
linked.set_param('sld_solvent', value=1.0, vary=False)
linked.set_param('radius', value=40, min=15, max=90, vary=True)
linked.set_param('scale', value=0.05, min=0.001, max=0.2, vary=True)
linked.set_param('background', value=0.002, min=0, max=0.02, vary=True)
linked.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
linked.set_param('volfraction', value=0.2, min=0.01, max=0.55, vary=True)
linked_result = linked.fit(engine='lmfit', method='least_squares')
linked.plot_results(show_residuals=True, log_scale=True)

print('Free effective radius:  ', interacting_result['chisq'])
print('Linked effective radius:', linked_result['chisq'])
"""
        ),
        interpretation,
        markdown(
            """## Exit ticket

Sketch a decision tree for an unfamiliar curve: data checks, simplest form
factor, residual diagnosis, alternative shapes, q-range sensitivity, and only
then interaction models. Name one conclusion that SANS alone cannot establish.

Use the lecture's *Step 0* and *Fitting strategy* (slides 30 and 32) as the
root of your tree: know the composition, concentration, charge and likely
sizes first. Then use the model-free looks (Guinier, power-law slopes,
Kratky, P(r); slides 21, 22, 50 and 27) before model fitting. Bring your tree to
the plenary discussion."""
        ),
        markdown(
            f"""## Further reading

- [SasView `hardsphere` model]({SASVIEW_MODEL.format('hardsphere')}): the
  Percus–Yevick structure factor used here.
- [SasView `hayter_msa` model]({SASVIEW_MODEL.format('hayter_msa')}): the
  Hayter–Penfold structure factor for charged particles mentioned on slide 26.
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

# Known contrast: fixed, exactly as in notebook 01.
fitter.set_param('sld', value=4.2, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)

# For DREAM, min and max define uniform priors: every value inside is equally
# plausible beforehand. They are narrower than in notebook 01 so that the short
# classroom run can explore them, but still wide enough not to cut the answer.
fitter.set_param('radius', value=50, min=30, max=90, vary=True)
fitter.set_param('scale', value=0.05, min=0.01, max=0.12, vary=True)
fitter.set_param('background', value=0.002, min=0, max=0.008, vary=True)
fitter.get_params()
"""
        if solution
        else """fitter = SANSFitter()
fitter.load_data(str(filename))

# YOUR CODE HERE:
#   1. fitter.set_model('sphere') and fitter.get_params()
#   2. Fix both SLDs (4.2 and 1.0) with vary=False.
#   3. Give radius, scale and background a value, min, max and vary=True.
#      The min-max range is now your uniform prior. Keep it defensible but not
#      so broad that the short classroom run cannot explore it.
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Configure the sphere model and its priors, then delete this line')
"""
    )
    run_sampler = (
        """# Fix the random seed so everyone in the room gets the same chains.
np.random.seed(2026)

# A short, exploratory DREAM run sized for class time (about a minute).
result = fitter.fit_bayesian(samples=1500, burn=100, thin=1, pop=10)

# Best sample from the posterior, with normalized residuals.
fitter.plot_results(show_residuals=True, log_scale=True)
"""
        if solution
        else """# YOUR CODE HERE: two lines.
#   1. result = fitter.fit_bayesian(samples=1500, burn=100, thin=1, pop=10)
#   2. fitter.plot_results(show_residuals=True, log_scale=True)
# The seed below makes everyone's chains identical; leave it in place.
np.random.seed(2026)
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Run DREAM and plot the result, then delete this line')
"""
    )
    diagnostics = """posterior = fitter.get_posterior()
print('Parameters:', posterior.labels)
print('Combined sample shape (draws, parameters):', posterior.samples.shape)

# For each parameter: the 95 % credible interval and the R-hat convergence check
# (close to 1 is good; above about 1.05 means the chains disagree).
for name in posterior.labels:
    low, high = posterior.ci_95[name]
    r_hat = posterior.diagnostics.get(name, {}).get('r_hat')
    r_hat_text = f'{r_hat:.3f}' if r_hat is not None else 'n/a'
    print(f'{name:>12}: 95% CI [{low:.6g}, {high:.6g}], R-hat={r_hat_text}')

# Trace plots: each chain's value against step number. Look for flat, well-mixed
# "fuzzy caterpillars" rather than drifts or chains that stay apart.
fitter.plot_trace()
"""
    posterior_plots = (
        """# The marginal distribution of the radius: its uncertainty after
# averaging over all values of the other parameters.
fitter.plot_param_distribution('radius')

# Pair plot: every pair of parameters against each other. A tilted, elongated
# cloud means the data constrain a combination of the two, not each separately.
fitter.plot_posterior_pairs()

# The same information as a single matrix of correlation coefficients (-1 to 1).
fitter.plot_param_correlations()
"""
        if solution
        else """# YOUR CODE HERE: three lines.
#   fitter.plot_param_distribution('radius')   # uncertainty in one parameter
#   fitter.plot_posterior_pairs()              # every pair of parameters
#   fitter.plot_param_correlations()           # correlation coefficients, -1 to 1
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Plot the posterior distributions, then delete this line')
"""
    )
    predictive = (
        """# Draw parameter sets from the posterior, compute the model for each, and
# plot the 95 % band they span together with 40 individual model curves.
fitter.plot_posterior_predictive(style='band+draws', n_draws=40)
"""
        if solution
        else """# YOUR CODE HERE: one line.
#   fitter.plot_posterior_predictive(style='band+draws', n_draws=40)
# It draws parameter sets from the posterior, computes the model for each, and
# shows the 95 % band they span plus 40 individual curves.
# Delete the `raise` line below when your lines are in place.
raise NotImplementedError('Plot the posterior predictive check, then delete this line')
"""
    )
    interpretation = answer(
        solution,
        """The radius is the best-constrained parameter, because it is fixed by the
positions of the form-factor minima. Scale and background are the most
correlated pair (r ≈ −0.3): both lift the curve where the signal is weakest, so
one can partly compensate for the other. Scale and radius are only weakly
anti-correlated (r ≈ −0.1). The plateau $I(0) \\propto \\mathrm{scale}\\cdot r^3$
couples them, but the minima fix the radius independently. The predictive band
is widest near the minima and at high q, where the signal approaches the
background. The R-hat values of this short classroom run remain above the
convergence target, so the chain must be lengthened and rechecked before any
credible interval is reported.""",
        'Summarize what DREAM adds beyond the best fit, answering questions 1–5 above. '
        'Refer to the credible intervals, the trace plots, the correlations, and the '
        'predictive band.',
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
            """- Slide 34, *Deeper into uncertainty: priors and bounds*: every fit has
  priors (the bounds, the model, the q range you kept); a bound is a uniform
  prior; a posterior pressed against a bound means the bound, not the data,
  chose the answer; a restricted q range destroys information. This notebook
  lets you test each of these on the sphere from notebook 01.
- Slide 35, *Wrap-up for part 2*: "several of these parameters are
  degenerate ($\\phi \\leftrightarrow \\Delta\\rho \\leftrightarrow$ scale)". Here
  you will *see* a degeneracy as a tilted cloud in a pair plot.
- Slide 20's discussion: concentration and contrast move the curve up and
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

Configure the same sphere model as in notebook 01. Keep the SLDs fixed and give
radius, scale and background bounded uniform priors.

**Where the likelihood comes from.** If each measured point is Gaussian with
standard deviation `dI`, then
$p(D \\mid \\theta) \\propto \\exp(-\\chi^2(\\theta)/2)$, with the same $\\chi^2$ you
minimized in notebook 01. The best fit is the peak of this function. Sampling
maps out its whole shape. This is why correct error bars from data reduction
(slides 44–45) matter even more here than for a point estimate.

**Fixing the SLDs is itself a prior.** It is a delta-function prior that
breaks the $\\phi \\leftrightarrow \\Delta\\rho$ degeneracy of slide 35 by fiat.
Ask yourself how confident you really are in those numbers (slide 12's
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
$I(0) \\propto \\mathrm{scale}\\cdot V \\propto \\mathrm{scale}\\cdot r^3$ (slide 20).
If the data pin $I(0)$ tightly, which sign of correlation do you expect between
`scale` and `radius`? Is it strong, given that the *positions* of the minima
pin the radius independently (slide 23)? Which parameter should be most
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
that resolution and polydispersity fill in minima (slide 23). If those effects
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
        interpretation,
        markdown(
            """## Optional export

The raw chain can be analyzed with other tools:

```python
posterior.save_posterior_csv('sphere_posterior.csv')
fitter.save_results('sphere_bayesian_fit.csv')
```

Try changing one prior range or restricting q. Rerun the sampler and explain
whether the posterior changed because of the data or because of the prior.

Two experiments from slide 34, each a one-line change before rerunning the
sampler cell:

1. *A bound that chooses the answer.* Cap the radius below its true value with
   `fitter.set_param('radius', max=55)`. Does χ² warn you? Where does the
   radius posterior pile up?
2. *Restricted q destroys information.* Keep only `0.004–0.025 Å⁻¹` with
   `fitter.set_q_range(qmin=0.004, qmax=0.025)`. Which posterior widens most,
   and does the pair plot now show a "banana" between size and scale?"""
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
    """Validate student tasks, then execute solutions and task-free student notebooks.

    Student tasks may only ask for sans_fitter calls, so every task cell must be
    marked `YOUR CODE HERE`. Source notebooks are executed into a temporary
    directory and stay clean.
    """
    runnable = []
    for notebook_path in sorted(NOTEBOOK_DIR.glob('*.ipynb')):
        notebook_data = json.loads(notebook_path.read_text(encoding='utf-8'))
        has_task = False
        for cell in notebook_data.get('cells', []):
            source = ''.join(cell.get('source', []))
            if 'NotImplementedError' in source:
                has_task = True
                if 'YOUR CODE HERE' not in source:
                    raise ValueError(f'{notebook_path} has an unmarked student task')
        if not has_task:
            runnable.append(notebook_path)
    runnable += sorted(SOLUTION_DIR.glob('*.ipynb'))

    environment = dict(os.environ)
    environment['MPLBACKEND'] = 'Agg'
    source_path = str(ROOT / 'src')
    existing_pythonpath = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = (
        f'{source_path}{os.pathsep}{existing_pythonpath}' if existing_pythonpath else source_path
    )
    with tempfile.TemporaryDirectory(prefix='sans-workshop-') as output_root:
        for notebook_path in runnable:
            output_dir = Path(output_root) / notebook_path.parent.name
            output_dir.mkdir(exist_ok=True)
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
                    str(output_dir),
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
    write_mystery_datasets()
    if not args.data_only:
        write_notebooks()
    if args.check:
        check_notebooks()


if __name__ == '__main__':
    main()

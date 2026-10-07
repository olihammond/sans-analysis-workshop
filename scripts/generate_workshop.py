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
            """## Predict before plotting

1. Why are both axes usually logarithmic for SANS?
2. What information does `dI` carry?
3. Which part of the curve is most sensitive to overall particle size?

Write a one-sentence prediction in the cell below."""
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
            """## Quick check

The curve is already reduced: detector corrections, normalization, radial
averaging, and uncertainty propagation happened upstream. This workshop starts
at the point where a scientific model is compared with those reduced data."""
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
the amplitude function. Ignore absolute scaling in this first visual test."""
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
scale, and background. Afterward, call `get_params()` to audit your setup."""
        ),
        code(params + '\nfitter.get_params()\n'),
        markdown(
            """### Running the optimizer

`fit(engine='lmfit', method='least_squares')` performs a bounded local
least-squares fit. The returned dictionary contains the objective value and
fitted parameters. `plot_results(...)` overlays the model and displays
normalized residuals."""
        ),
        code(fit_call),
        markdown(
            """## Exercise 3 — Criticize, do not merely accept

Record:

- fitted radius and uncertainty;
- whether residuals look randomly distributed around zero;
- any fitted parameter touching a bound;
- one experimental systematic that this model ignores."""
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
about optimization versus inference?"""
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
before running the cells. Which shape features have disappeared from view?"""
        ),
        code(short_range),
        markdown(
            """## Verdict

Write a short referee-style verdict. Include one reason based on residuals, one
based on parameter plausibility, and one follow-up measurement that would make
the distinction stronger."""
        ),
        code(conclusion),
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
            """## Exercise 1 — Deliberately incomplete model

From the previous notebooks, build a dilute-sphere fit without copying a
complete solution. Locate the structured residual region. Why can changing
radius or scale not independently repair it?"""
        ),
        code(dilute_fit),
        markdown(
            """## Exercise 2 — Add excluded-volume interactions

`set_structure_factor('hardsphere')` combines the selected form factor with a
hard-sphere interaction model. Calling `get_params()` afterward reveals the
additional parameters:

- `volfraction`: fraction of the sample volume occupied by interacting spheres;
- `radius_effective`: interaction distance, which may include a solvation layer.

Fit both independently. The mode-control parameters should remain fixed."""
        ),
        code(interacting_fit),
        code(comparison),
        markdown(
            """## Exercise 3 — Interpret and challenge

1. What real-space correlation creates the peak?
2. Why might `radius_effective` differ from the form-factor radius?
3. What assumptions in the hard-sphere model could fail?
4. Try `radius_effective_mode='link_radius'`. Is that constraint supported by
   these data?"""
        ),
        code(interpretation),
        markdown(
            """## Exit ticket

Sketch a decision tree for an unfamiliar curve: data checks, simplest form
factor, residual diagnosis, alternative shapes, q-range sensitivity, and only
then interaction models. Name one conclusion that SANS alone cannot establish."""
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
bounded priors for radius, scale, and background."""
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
chosen for class time and is **exploratory, not publication quality**."""
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
and scale can compensate for one another."""
        ),
        code(posterior_plots),
        markdown(
            """## 5 — Posterior predictive checking

Parameter intervals are not the final scientific target. A posterior
predictive check propagates sampled parameters through the scattering model.
The resulting band asks whether plausible parameter sets reproduce the
observations across q, and where predictions remain uncertain."""
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
whether the posterior changed because of the data or because of the prior."""
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

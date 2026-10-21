"""Generate the unlabelled 'mystery sample' datasets for the report exercise.

Each sample is simulated with SasModels in absolute units (cm^-1), with
realistic compositions, scattering length densities, and backgrounds. The
student-facing sample cards are written to ``mystery_data/README.md``; the
generating models and parameters go to ``instructor/mystery_ground_truth.json``.

Run on its own with ``python scripts/generate_mystery_data.py`` or as part of
``python scripts/generate_workshop.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

ROOT = Path(__file__).resolve().parents[1]
MYSTERY_DIR = ROOT / 'mystery_data'
INSTRUCTOR_DIR = ROOT / 'instructor'
MYSTERY_SEED = 20261021

# SLDs in 10^-6 Å^-2, calculated from composition and bulk density.
SLD = {
    'H2O': -0.56,
    'D2O': 6.36,
    'silica': 3.47,
    'PPO': 0.34,
    'PEO': 0.64,
    'C16 tail': -0.37,
    'DOPC': 0.30,
    'n-decane': -0.49,
}

SAMPLES: list[dict[str, object]] = [
    {
        'label': 'A',
        'model': 'core_shell_sphere@hardsphere',
        'parameters': {
            'radius': 48.0,
            'radius_pd': 0.10,
            'radius_pd_n': 35,
            'radius_pd_nsigma': 3,
            'thickness': 32.0,
            'sld_core': SLD['PPO'],
            'sld_shell': 5.75,
            'sld_solvent': SLD['D2O'],
            'radius_effective': 80.0,
            'radius_effective_mode': 0,
            'volfraction': 0.16,
            'scale': 1.0,
            'background': 0.07,
        },
        'sampling': {'qmin': 0.004, 'qmax': 0.45, 'points': 120, 'relative_error': 0.03},
        'card': {
            'sample': 'Pluronic P123 triblock copolymer, EO20–PO70–EO20 '
            '(M ≈ 5800 g/mol), 5 wt% in D2O',
            'conditions': 'Measured at 35 °C, above the critical micellization temperature',
            'slds': [
                ('PPO block (C3H6O, 1.00 g/cm³)', SLD['PPO']),
                ('PEO block (C2H4O, 1.13 g/cm³)', SLD['PEO']),
                ('D2O', SLD['D2O']),
            ],
            'notes': 'PEO is water soluble and strongly hydrated at this temperature; PPO is not.',
        },
        'answer': 'Spherical block-copolymer micelles: dry PPO core (R ≈ 48 Å, 10 % '
        'polydisperse) with a highly hydrated PEO corona (≈ 32 Å, ≈ 89 % D2O, so weak '
        'contrast). Concentrated enough for a hard-sphere S(Q) with effective radius '
        '≈ 80 Å (core + corona) and micelle volume fraction ≈ 0.16. Aggregation number '
        '≈ 70 from the core volume. Pitfall: sphere@hardsphere with the core contrast fails '
        '(reduced χ² ≈ 50) because one volfraction cannot be both the core and the micelle '
        'volume fraction; the corona must be in the model.',
    },
    {
        'label': 'B',
        'model': 'cylinder',
        'parameters': {
            'radius': 21.0,
            'length': 800.0,
            'length_pd': 0.30,
            'length_pd_n': 35,
            'length_pd_nsigma': 3,
            'sld': SLD['C16 tail'],
            'sld_solvent': SLD['D2O'],
            'scale': 0.0041,
            'background': 0.05,
        },
        'sampling': {'qmin': 0.003, 'qmax': 0.45, 'points': 130, 'relative_error': 0.03},
        'card': {
            'sample': 'Cetyltrimethylammonium bromide (CTAB, M = 364.5 g/mol), 15 mM in D2O '
            'with 0.1 M KBr',
            'conditions': 'Measured at 30 °C',
            'slds': [
                ('Hexadecyl tail (C16H33, 457.8 Å³ per chain)', SLD['C16 tail']),
                ('D2O', SLD['D2O']),
            ],
            'notes': 'Treat the hydrocarbon tails as the only scattering component; the '
            'small headgroups and counter-ions are hydrated and close to solvent.',
        },
        'answer': 'Rod-like (short wormlike) micelles: cylinder radius ≈ 21 Å, matching '
        'the extended C16 chain length (Tanford: 1.5 + 1.265 × 16 ≈ 21.7 Å); length '
        '≈ 800 Å with 30 % polydispersity. Volume fraction of tails ≈ 0.004, matching '
        '15 mM CTAB. Q⁻¹ region between the two Guinier knees.',
    },
    {
        'label': 'C',
        'model': 'vesicle',
        'parameters': {
            'radius': 240.0,
            'radius_pd': 0.22,
            'radius_pd_n': 35,
            'radius_pd_nsigma': 3,
            'thickness': 38.0,
            'sld': SLD['DOPC'],
            'sld_solvent': SLD['D2O'],
            'volfraction': 0.0099,
            'scale': 1.0,
            'background': 0.05,
        },
        'sampling': {'qmin': 0.003, 'qmax': 0.40, 'points': 130, 'relative_error': 0.03},
        'card': {
            'sample': 'Dioleoylphosphatidylcholine (DOPC, C44H84NO8P, M = 786 g/mol), '
            '10 mg/mL in D2O, extruded through a 50 nm membrane',
            'conditions': 'Measured at 25 °C',
            'slds': [
                ('DOPC (1.01 g/cm³, averaged over the molecule)', SLD['DOPC']),
                ('D2O', SLD['D2O']),
            ],
            'notes': 'Extrusion produces a fairly broad size distribution.',
        },
        'answer': 'Unilamellar vesicles: core radius ≈ 240 Å (22 % polydisperse), '
        'bilayer thickness ≈ 38 Å (typical for DOPC). Q⁻² mid-q region from the thin '
        'shell; the bilayer thickness sets the high-q drop. Shell volume fraction ≈ 0.01, '
        'matching 10 mg/mL. In the vesicle model, keep scale = 1 and fit volfraction.',
    },
    {
        'label': 'D',
        'model': 'sphere',
        'parameters': {
            'radius': 110.0,
            'radius_pd': 0.12,
            'radius_pd_n': 35,
            'radius_pd_nsigma': 3,
            'sld': SLD['silica'],
            'sld_solvent': SLD['H2O'],
            'scale': 0.0092,
            'background': 0.95,
        },
        'sampling': {'qmin': 0.003, 'qmax': 0.35, 'points': 120, 'relative_error': 0.03},
        'card': {
            'sample': 'Colloidal silica (amorphous SiO2, 2.2 g/cm³), 2 wt% in H2O',
            'conditions': 'Measured at 25 °C, pH 9, 1 mm path length',
            'slds': [
                ('Amorphous SiO2', SLD['silica']),
                ('H2O', SLD['H2O']),
            ],
            'notes': 'The supplier quotes a "nominal diameter of 22 nm".',
        },
        'answer': 'Dilute polydisperse spheres: R ≈ 110 Å (diameter 22 nm), 12 % Gaussian '
        'polydispersity, which smears the minima. Volume fraction ≈ 0.009 (2 wt% / 2.2 '
        'g/cm³). High flat background (≈ 0.95 cm⁻¹) from H2O incoherent scattering.',
    },
    {
        'label': 'E',
        'model': 'teubner_strey',
        'parameters': {
            'volfraction_a': 0.45,
            'sld_a': SLD['n-decane'],
            'sld_b': SLD['D2O'],
            'd': 260.0,
            'xi': 95.0,
            'scale': 1.0,
            'background': 0.25,
        },
        'sampling': {'qmin': 0.004, 'qmax': 0.40, 'points': 120, 'relative_error': 0.03},
        'card': {
            'sample': 'D2O / n-decane / C12E5 (pentaethylene glycol monododecyl ether); '
            '45 vol% n-decane and 55 vol% D2O, with about 8 wt% surfactant',
            'conditions': 'Measured at 32 °C, near the phase-inversion temperature where '
            'the surfactant has no preferred curvature',
            'slds': [
                ('n-decane (C10H22, 0.73 g/cm³)', SLD['n-decane']),
                ('D2O', SLD['D2O']),
            ],
            'notes': 'The thin surfactant film can be neglected at this contrast. The '
            'sample is a single, transparent, isotropic phase.',
        },
        'answer': 'Bicontinuous microemulsion (Teubner–Strey): broad peak at '
        'Q ≈ 0.022 Å⁻¹, repeat distance d ≈ 260 Å, correlation length ξ ≈ 95 Å '
        '(d/ξ ≈ 2.7, a weakly ordered sponge). Q⁻⁴ (Porod) decay from sharp oil–water '
        'interfaces. Not a droplet form factor: no Guinier plateau, peak without '
        'dilute-particle oscillations. The model depends on volfraction_a only through '
        'φ(1 − φ), so 0.45 and 0.55 fit equally well; the card decides which.',
    },
    {
        'label': 'F',
        'model': 'polymer_excl_volume',
        'parameters': {
            'rg': 68.0,
            'porod_exp': 1.67,
            'scale': 0.85,
            'background': 0.06,
        },
        'sampling': {'qmin': 0.004, 'qmax': 0.50, 'points': 120, 'relative_error': 0.035},
        'card': {
            'sample': 'Poly(ethylene glycol) (PEG, M ≈ 20 000 g/mol), 10 mg/mL in D2O',
            'conditions': 'Measured at 25 °C',
            'slds': [
                ('PEG (C2H4O, 1.13 g/cm³)', SLD['PEO']),
                ('D2O', SLD['D2O']),
            ],
            'notes': 'Water is a good solvent for PEG at room temperature.',
        },
        'answer': 'Polymer coil in good solvent: Rg ≈ 68 Å and power-law exponent ≈ 5/3 '
        '(Flory ν = 3/5, swollen coil). I(0) ≈ 0.85 cm⁻¹ is consistent with '
        'M ≈ 20 kg/mol via I(0) = φ V Δρ². In polymer_excl_volume, scale is I(0), not a '
        'volume fraction.',
    },
]


def simulate(model_name: str, q: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    """Evaluate a SasModels model on q in double precision."""
    kernel = load_model(model_name, dtype='double', platform='dll')
    return np.asarray(DirectModel(empty_data1D(q), kernel)(**parameters))


def sample_card(sample: dict[str, object]) -> str:
    """Return the student-facing Markdown card for one sample."""
    card = sample['card']
    filename = f'sample_{sample["label"]}.csv'
    q_min = sample['sampling']['qmin']
    q_max = sample['sampling']['qmax']
    rows = '\n'.join(f'| {name} | {value:.2f} |' for name, value in card['slds'])
    return f"""### Sample {sample['label']} — `{filename}`

- **What was put in the beam:** {card['sample']}.
- **Conditions:** {card['conditions']}.
- **Measured q range:** {q_min}–{q_max} Å⁻¹.
- **Notes:** {card['notes']}

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
{rows}
"""


def readme(samples: list[dict[str, object]]) -> str:
    """Return the README for the mystery-data folder."""
    cards = '\n'.join(sample_card(sample) for sample in samples)
    return f"""# Mystery samples

Six reduced SANS datasets, one per file, with no labels and no model. Pick one,
work out what is in the beam, and write a short report that defends your answer.

The files use the same `Q,I,dI` columns as the workshop data, but intensities
are now in **absolute units, cm⁻¹**. That means the fitted parameters must agree
with the sample composition, not only with the curve. Use this as a check.

## Your task

1. Choose a sample and read its card below. The card tells you what was put
   in the beam, but not what structure it formed.
2. Before fitting, write down what you expect and why (lecture slide 30,
   *Step 0: understand your sample*). Which length scales can you estimate from
   the chemistry alone?
3. Analyse the curve model-free first: Guinier ($R_g$, $I(0)$), power-law
   slopes, and peak positions (slides 21, 22, and 26).
4. Choose candidate models from the
   [SasView model list](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html)
   and fit them with `sans_fitter`. Fit at least two competing models.
5. Write the report (structure below).

## Loading a sample

```python
fitter = SANSFitter()
fitter.load_data(str(ROOT / 'mystery_data' / 'sample_A.csv'))

# Look before you fit: the data with a candidate model at its current values.
fitter.set_model('sphere')
fig = fitter.plot_model()

# ...set parameters, then fit and summarize:
fitter.fit(engine='lmfit', method='least_squares')
fitter.plot_results(show_residuals=True, log_scale=True)
report = fitter.get_fit_report()
report                  # a table in the notebook; print(report) for plain text
```

`report.to_markdown()` gives the same table in Markdown, ready to paste into
your report.

Use the setup cell from any workshop notebook to define `ROOT` and import
`SANSFitter`.

## Practical hints

- **Absolute scale.** For most form-factor models, `scale` is the volume
  fraction of the scattering objects. Compare it with what the card implies.
- **Models with their own volume fraction.** Some models (for example those with
  a `volfraction` parameter, and all form factor × structure factor products)
  already contain the concentration. Keep `scale` fixed at 1 for those, or the
  two parameters will trade off against each other.
- **Polydispersity.** Real samples are rarely monodisperse, and a size
  distribution fills in form-factor minima (slide 23). You can give any size
  parameter a Gaussian distribution and fit its width:

  ```python
  fitter.set_polydispersity('radius', width=0.1, min=0.0, max=0.4, vary=True)
  ```

  Size and polydispersity are strongly correlated (slide 34): a larger particle
  with a narrower distribution can look like a smaller one with a broader
  distribution. Report both, and say whether outside knowledge constrains either.

- **Background.** The flat high-q level comes mainly from incoherent
  scattering of hydrogen (slide 15). Is its size consistent with the solvent?
- **Structure factors.** `set_structure_factor('hardsphere')` and the others
  from notebook 03 work with any spherical form factor.
- **Reduced χ².** The fit report gives the reduced χ², its expected range and
  the residual checks for every engine. Judge them together with the residual
  plot (slide 33), and look for parameters the report flags as stuck on a bound.
- **Kratky plot.** Plotting $I\\cdot Q^2$ against $Q$ (slide 50) quickly
  separates compact objects (a peak), flexible chains (a plateau) and rods (a
  steady rise).
- **Real examples.** Part 4 of the lecture (slides 50–54) shows how polymers,
  self-assembled systems, lipid particles and precipitates are analysed in
  practice. Use it to choose candidate models.

## Report structure

Aim for two to three pages including figures.

1. **Sample and expectations.** What is in the beam, and what sizes, shapes and
   concentrations do you expect from the chemistry? Show one calculation (for
   example a molecular length, a volume fraction or a contrast).
2. **Model-free analysis.** Plot the data on log–log axes. Report a Guinier
   $R_g$ and $I(0)$ if a Guinier region exists, any characteristic slopes, and
   any peak positions converted to real-space distances.
3. **Model fitting.** State the models you tried, which parameters you fixed and
   why, the bounds, the fitted values with uncertainties, the reduced χ², and a
   residual plot for each model.
4. **Physical consistency.** Check fitted sizes against molecular dimensions,
   and fitted `scale` or volume fraction against the composition. Derive one
   further quantity, for example an aggregation number, molecular weight,
   specific surface area or repeat distance.
5. **Verdict.** What do you think the object is? Which alternative explanations
   did you reject, and on what evidence? What remains ambiguous?
6. **Next measurement.** Name one additional experiment (contrast variation,
   dilution series, wider q range, another technique) that would test your
   conclusion, and say what you would expect to see.

## Sample cards

All SLD values are calculated from the chemical formula and bulk density.
Treat them as good estimates, not exact values: hydration, H/D exchange and
packing density can change the effective SLD of a real component.

{cards}"""


def write_mystery_datasets() -> list[dict[str, object]]:
    """Write the mystery datasets, their README, and the instructor answer key."""
    MYSTERY_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUCTOR_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MYSTERY_SEED)

    for sample in SAMPLES:
        sampling = sample['sampling']
        parameters = sample['parameters']
        q = np.logspace(np.log10(sampling['qmin']), np.log10(sampling['qmax']), sampling['points'])
        truth = simulate(str(sample['model']), q, parameters)
        # Counting-like relative error plus a floor set by the flat background level.
        uncertainty = np.sqrt(
            (sampling['relative_error'] * truth) ** 2
            + (sampling['relative_error'] * parameters['background']) ** 2
        )
        observed = np.maximum(truth + rng.normal(0.0, uncertainty), np.finfo(float).tiny)
        np.savetxt(
            MYSTERY_DIR / f'sample_{sample["label"]}.csv',
            np.column_stack((q, observed, uncertainty)),
            delimiter=',',
            header='Q,I,dI',
            comments='',
            fmt='%.10e',
        )

    (MYSTERY_DIR / 'README.md').write_text(readme(SAMPLES), encoding='utf-8')
    answer_key = {
        'seed': MYSTERY_SEED,
        'units': 'I in cm^-1, q in Å^-1, SLD in 10^-6 Å^-2',
        'samples': {
            sample['label']: {
                'model': sample['model'],
                'parameters': sample['parameters'],
                'answer': sample['answer'],
            }
            for sample in SAMPLES
        },
    }
    (INSTRUCTOR_DIR / 'mystery_ground_truth.json').write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    return SAMPLES


if __name__ == '__main__':
    write_mystery_datasets()
    print(f'Wrote {len(SAMPLES)} mystery samples to {MYSTERY_DIR}')

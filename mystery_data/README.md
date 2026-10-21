# Mystery samples

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
fitter.set_resolution('none')

# Look before you fit: the data with a candidate model at its current values.
fitter.set_model('sphere')
fitter.plot_model()
```

Then set the parameters, fit and summarize, in separate cells:

```python
fitter.fit(engine='lmfit', method='least_squares')
report = fitter.get_fit_report()
report                  # a table in the notebook; print(report) for plain text
```

```python
fitter.plot_results(show_residuals=True, log_scale=True)
```

A figure is displayed when the plotting command is the last line of its cell.
`report.to_markdown()` gives the fit report in Markdown, ready to paste into
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
  fitter.enable_polydispersity(True)
  fitter.set_pd_param('radius', pd_width=0.1, vary=True)
  ```

  Size and polydispersity are strongly correlated (slide 34): a larger particle
  with a narrower distribution can look like a smaller one with a broader
  distribution. Report both, and say whether outside knowledge constrains either.

- **Background.** The flat high-q level comes mainly from incoherent
  scattering of hydrogen (slide 15). Is its size consistent with the solvent?
- **Structure factors.** `set_structure_factor('hardsphere')` and the others
  from notebook 03 work with any spherical form factor. As in notebook 03, call
  `set_param('radius_effective_mode', value=0)` afterwards if you want to fit
  `radius_effective` yourself.
- **Reduced χ².** The fit report gives χ²/dof and the correlations between
  fitted parameters for every engine. Judge them together with the residual plot
  (slide 33), and look for parameters the report flags as on a bound.
- **Kratky plot.** Plotting $I\cdot Q^2$ against $Q$ (slide 50) quickly
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

### Sample A — `sample_A.csv`

- **What was put in the beam:** Pluronic P123 triblock copolymer, EO20–PO70–EO20 (M ≈ 5800 g/mol), 5 wt% in D2O.
- **Conditions:** Measured at 35 °C, above the critical micellization temperature.
- **Measured q range:** 0.004–0.45 Å⁻¹.
- **Notes:** PEO is water soluble and strongly hydrated at this temperature; PPO is not.

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
| PPO block (C3H6O, 1.00 g/cm³) | 0.34 |
| PEO block (C2H4O, 1.13 g/cm³) | 0.64 |
| D2O | 6.36 |

### Sample B — `sample_B.csv`

- **What was put in the beam:** Cetyltrimethylammonium bromide (CTAB, M = 364.5 g/mol), 15 mM in D2O with 0.1 M KBr.
- **Conditions:** Measured at 30 °C.
- **Measured q range:** 0.003–0.45 Å⁻¹.
- **Notes:** Treat the hydrocarbon tails as the only scattering component; the small headgroups and counter-ions are hydrated and close to solvent.

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
| Hexadecyl tail (C16H33, 457.8 Å³ per chain) | -0.37 |
| D2O | 6.36 |

### Sample C — `sample_C.csv`

- **What was put in the beam:** Dioleoylphosphatidylcholine (DOPC, C44H84NO8P, M = 786 g/mol), 10 mg/mL in D2O, extruded through a 50 nm membrane.
- **Conditions:** Measured at 25 °C.
- **Measured q range:** 0.003–0.4 Å⁻¹.
- **Notes:** Extrusion produces a fairly broad size distribution.

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
| DOPC (1.01 g/cm³, averaged over the molecule) | 0.30 |
| D2O | 6.36 |

### Sample D — `sample_D.csv`

- **What was put in the beam:** Colloidal silica (amorphous SiO2, 2.2 g/cm³), 2 wt% in H2O.
- **Conditions:** Measured at 25 °C, pH 9, 1 mm path length.
- **Measured q range:** 0.003–0.35 Å⁻¹.
- **Notes:** The supplier quotes a "nominal diameter of 22 nm".

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
| Amorphous SiO2 | 3.47 |
| H2O | -0.56 |

### Sample E — `sample_E.csv`

- **What was put in the beam:** D2O / n-decane / C12E5 (pentaethylene glycol monododecyl ether); 45 vol% n-decane and 55 vol% D2O, with about 8 wt% surfactant.
- **Conditions:** Measured at 32 °C, near the phase-inversion temperature where the surfactant has no preferred curvature.
- **Measured q range:** 0.004–0.4 Å⁻¹.
- **Notes:** The thin surfactant film can be neglected at this contrast. The sample is a single, transparent, isotropic phase.

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
| n-decane (C10H22, 0.73 g/cm³) | -0.49 |
| D2O | 6.36 |

### Sample F — `sample_F.csv`

- **What was put in the beam:** Poly(ethylene glycol) (PEG, M ≈ 20 000 g/mol), 10 mg/mL in D2O.
- **Conditions:** Measured at 25 °C.
- **Measured q range:** 0.004–0.5 Å⁻¹.
- **Notes:** Water is a good solvent for PEG at room temperature.

| Component | SLD / 10⁻⁶ Å⁻² |
|---|---:|
| PEG (C2H4O, 1.13 g/cm³) | 0.64 |
| D2O | 6.36 |

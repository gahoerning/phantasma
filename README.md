# PHANTaSMA

**P**latform of **H**igh-level **A**nalysis, **N**umerical **T**echniques **a**nd **S**ky **M**ap **A**pplications

PHANTaSMA is a precision-oriented framework for astrophysical sky-map analysis.  
Its purpose is to integrate established libraries (`healpy`, `astropy`, `numpy`, `scipy`) into rigorously controlled pipelines that minimise numerical inconsistencies and preprocessing biases.

The project does not aim to replace existing libraries. It enforces correct usage patterns, explicit assumptions, and reproducible workflows.

---

## Installation

```bash
pip install -e /path/to/phantasma   # editable install (recommended for development)
```

Dependencies (`numpy`, `astropy`, `scipy`, `healpy`, `reproject`, `tqdm`) are installed automatically.

---

## Modules

### `phantasma.cutoff_processing` — Map preprocessing

Reprojection, smoothing, and cutout utilities for 2-D sky maps.

```python
from phantasma import smooth_cutout, make_target_wcs

target_wcs = make_target_wcs(...)
result = smooth_cutout(...)
```

---

### `phantasma.template_fitting` — Weighted linear template fitting

Fits the model:

```
data_map = Σ_i  a_i · template_i  +  Σ_j  b_j · geom_j  +  residual
```

Features:
- **Iterative uncertainty propagation** — total pixel variance is updated at each iteration to account for calibration errors and template noise.
- **SVD-based solver** (`scipy`) for numerical stability with ill-conditioned design matrices.
- **Bootstrap uncertainty estimation** — pixel resampling with a `tqdm` progress bar.
- **Geometric foreground templates** — monopole, x/y gradients, quadratic r².
- **`template_fit`** — fast version without bootstrap, for exploratory fits.

#### Quick start

```python
from phantasma.template_fitting import template_fit_bootstrap

result = template_fit_bootstrap(
    data_map,
    template_maps,          # shape (Ntemp, ny, nx)
    data_rms=rms_map,
    template_names=["dust", "co"],
    n_bootstrap=1000,
)

result.summary()
# ======================================================================
# Template Fit Summary
# ======================================================================
# Parameter                   Value      Formal σ   Bootstrap σ
# ----------------------------------------------------------------------
# dust                       1.4991    0.00075491     0.0007952
# co                         3.0002    0.00074608    0.00074665
# ...

# Per-template contribution maps
contrib = result.component_maps(template_maps)   # shape (Ntemp, ny, nx)
```

#### Geometric foreground removal

```python
from phantasma.template_fitting import make_geometric_templates, template_fit_bootstrap

geom, gnames = make_geometric_templates(data_map.shape, include=("monopole", "x", "y"))

result = template_fit_bootstrap(
    data_map, template_maps, data_rms=rms_map,
    geom_templates=geom, geom_names=gnames,
)
```

#### Synthetic simulation (for validation)

```python
from phantasma.template_fitting import simulate_template_fit, template_fit_bootstrap

data, templates, rms, true_a = simulate_template_fit(
    shape=(64, 64),
    true_amplitudes=[1.5, 3.0],
    noise_level=0.05,
)

result = template_fit_bootstrap(data, templates, data_rms=rms)
result.summary()
```

---

## Planned Features

- RMS map smoothing (HEALPix and Cartesian)
- Safe HEALPix resolution changes (`ud_grade` with variance and pixel window control)
- Simple SED fitting (synchrotron, dust, free–free)
- Gaussian fitting via MCMC
- Aperture photometry utilities
- High-performance C/C++ modules for heavy simulations

---

## Principles

- Numerical exactness
- Explicit beam and resolution handling
- Reproducibility
- Modular design

---

## Status

Active development. Implemented modules: `cutoff_processing`, `template_fitting`.


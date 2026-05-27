# PHANTaSMA

**P**latform of **H**igh-level **A**nalysis, **N**umerical **T**echniques **a**nd **S**ky **M**ap **A**pplications

PHANTaSMA is a precision-oriented framework for astronomical sky-map preprocessing, analysis, and linear template fitting. Its core goal is to integrate established astrophysics and astronomy libraries (`healpy`, `astropy`, `numpy`, `scipy`, `reproject`) into rigorously controlled, mathematically exact pipelines that minimize numerical inconsistencies, coordinate reprojection issues, and preprocessing biases.

The framework is built around **exact physical principles** (e.g., pixel window correction, Wiener-regularized beam transfers, flux conservation) and **robust uncertainty propagation** (Monte Carlo noise projection and Bootstrap template fitting).

---

## Key Features

- 🌌 **Unified Preprocessing (`cutoff_processing`)**: Reproject, smooth, and extract cutouts from both FITS flat-sky maps (data + WCS) and HEALPix full-sky maps in a single, clean function call.
- 📐 **Non-Gaussian / Arbitrary Beam Support**: Smooth maps using either Gaussian beams (FWHM) or arbitrary tabulated beam profiles (e.g., $B(\ell)$ tables from survey websites like Planck or COMAP), or custom mathematical callables.
- 🧮 **Fourier-Space Transfer Functions**: Implements Wiener-regularized 2D FFT transfer functions:
  $$T(\ell) = \frac{B_{\text{target}}(\ell) P_{\text{new}}(\ell)}{B_{\text{input}}(\ell) P_{\text{old}}(\ell)}$$
  which exactly mirrors spherical harmonic operations on flat-sky projection patches.
- 📉 **Rigorous Noise/RMS Propagation (`propagate_rms_cutout`)**:
  - **Constant RMS**: Fast, exact analytical propagation formula including pixel window functions:
    $$\sigma_{\text{out}} = \sigma_{\text{in}} \cdot \left(\frac{p_0}{p_1}\right) \cdot \left(\frac{\theta_{\text{eff, orig}}}{\theta_{\text{eff, target}}}\right)$$
  - **Spatially-Varying RMS**: Memory-efficient Monte Carlo simulation patch propagation (HEALPix and FITS). Simulates white noise *only* on the local cutout patch, preventing memory bloat on large HEALPix maps.
- ⚖️ **Weighted Linear Template Fitting (`template_fitting`)**:
  - Stable SVD-based multi-template linear solver.
  - Accounts for geometric foregrounds (monopoles, x/y linear gradients, quadratic gradients).
  - Iterative pixel-variance updates incorporating template noise and calibration uncertainties.
  - Bootstrap-based parameter uncertainty estimation with a clean progress bar.

---

## Installation

```bash
pip install -e /path/to/phantasma   # editable install (recommended for development)
```

Dependencies (`numpy`, `astropy`, `scipy`, `healpy`, `reproject`, `tqdm`) are installed automatically.

---

## Modules & Usage Guide

### 1. `phantasma.cutoff_processing` — Precision Map Preprocessing

This module provides tools to smooth, reproject, and crop astronomical maps to a target grid defined by an `astropy.wcs.WCS` object.

#### Basic Usage (Gaussian Beams)

```python
import phantasma as ph

# Create target WCS centered at (l, b) = (17.0, 0.8) with 2.0 arcmin pixel size and 2x2 deg field of view
target_wcs, target_shape = ph.make_target_wcs(
    center_l=17.0,
    center_b=0.8,
    pixel_size_arcmin=2.0,
    cutout_size_deg=2.0
)

# Smooth and reproject a flat-sky FITS map
data_out, wcs_out = ph.smooth_cutout(
    data=fits_data,
    map_format="fits",
    original_wcs=fits_wcs,
    input_beam=5.0,        # original beam FWHM in arcmin
    target_beam=10.0,      # desired target resolution FWHM in arcmin
    pixel_size_arcmin=2.0,
    target_wcs=target_wcs,
    target_shape=target_shape
)
```

#### Advanced: Tabulated/Non-Gaussian Beams $B(\ell)$

If you are using a survey with a measured non-Gaussian beam, you can pass the tabulated beam transfer function directly:

```python
import numpy as np
import phantasma as ph

# Load measured beam profile from a survey (e.g. Planck, Effelsberg, COMAP)
# B_ell is the beam transfer function normalized to 1 at ell=0
ell, B_ell = np.loadtxt("survey_beam_profile.txt", unpack=True)

data_out, wcs_out = ph.smooth_cutout(
    data=hpx_map,
    map_format="healpix",
    healpix_coord="G",
    center_l=17.0,
    center_b=0.8,
    cutout_size_deg=2.0,
    input_beam=(ell, B_ell),  # Non-Gaussian input beam specification!
    target_beam=10.0,         # Target beam FWHM (always Gaussian)
    pixel_size_arcmin=2.0,
    target_wcs=target_wcs,
    target_shape=target_shape,
    beam_regularization=1e-4  # Wiener regularization parameter
)
```

#### Inverse-Variance Weighted Preprocessing

When a noise/RMS map is provided, `smooth_cutout` performs **inverse-variance weighted smoothing** and **weighted reprojection**, ensuring high-noise pixels (or masked regions) do not bias the output:

```python
data_out, wcs_out = ph.smooth_cutout(
    data=fits_data,
    rms_data=rms_data,       # Provide the original RMS map
    map_format="fits",
    original_wcs=fits_wcs,
    input_beam=5.0,
    target_beam=10.0,
    pixel_size_arcmin=2.0,
    target_wcs=target_wcs,
    target_shape=target_shape
)
```

#### Precision RMS Propagation (`propagate_rms_cutout`)

Propagates the associated RMS/noise maps through the exact same smoothing and reprojection pipelines.

##### Case A: Spatially Constant RMS (Exact Analytical Formula)
```python
# Propagates a single constant RMS noise level using exact analytical formula
rms_out = ph.propagate_rms_cutout(
    rms_data=0.02,           # Constant noise standard deviation
    rms_is_constant=True,
    map_format="fits",
    input_beam=5.0,
    target_beam=10.0,
    pixel_size_arcmin=2.0,
    target_wcs=target_wcs,
    target_shape=target_shape
)
```

##### Case B: Spatially Varying RMS (Memory-Efficient Monte Carlo)
For varying FITS or HEALPix RMS maps, a Monte Carlo simulation runs *only* on the extracted padded cutout grid. This keeps memory usage extremely low.
```python
# Spatially-varying RMS propagation via Wiener-correct local MC realisations
rms_out = ph.propagate_rms_cutout(
    rms_data=rms_map,        # 1-D HEALPix array or 2-D FITS array
    rms_is_constant=False,
    map_format="healpix",
    center_l=17.0,
    center_b=0.8,
    cutout_size_deg=2.0,
    input_beam=(ell, B_ell),
    target_beam=10.0,
    pixel_size_arcmin=2.0,
    target_wcs=target_wcs,
    target_shape=target_shape,
    n_mc=300,                # Number of MC simulation realisations
    random_seed=42
)
```

---

### 2. `phantasma.template_fitting` — Multi-Template Linear Regression

Fits multi-component astrophysical models with optional spatial gradients and full parameter covariance:

$$\mathbf{d} = \sum_{i} a_i \mathbf{t}_i + \sum_{j} b_j \mathbf{g}_j + \mathbf{r}$$

where:
- $\mathbf{d}$ is the data map (`data_map`)
- $\mathbf{t}_i$ are the foreground template maps (`template_maps`)
- $\mathbf{g}_j$ are the geometric templates (`geom_templates`)
- $\mathbf{r}$ is the residual map

#### Setup and Fitting (Quickstart)

```python
from phantasma.template_fitting import template_fit_bootstrap, make_geometric_templates

# 1. Create geometric baseline templates (monopole + linear gradients)
geom_templates, geom_names = make_geometric_templates(
    shape=data_map.shape, 
    include=("monopole", "x", "y")
)

# 2. Run the iterative fitting + bootstrap analysis
result = template_fit_bootstrap(
    data_map=data_map,
    template_maps=template_maps,                  # shape: (N_templates, ny, nx)
    data_rms=rms_map,
    template_rms=template_rms_maps,               # noise in the templates themselves
    data_calib_frac=0.05,                         # 5% calibration uncertainty on data
    template_calib_frac=[0.10, 0.05],             # calibration uncertainty per template
    geom_templates=geom_templates,
    geom_names=geom_names,
    template_names=["Eff. 2.7 GHz", "IRIS 25um"],
    n_bootstrap=1000,
    random_seed=42
)

# 3. Print a comprehensive statistics report
result.summary()
```

Sample output:
```
======================================================================
Template Fit Summary
======================================================================
Parameter                   Value      Formal σ   Bootstrap σ
----------------------------------------------------------------------
Eff. 2.7 GHz               1.4991    0.00075491     0.0007952
IRIS 25um                  3.0002    0.00074608    0.00074665
monopole                  -0.0520    0.01240500     0.0135021
x                          0.0031    0.00140200     0.0015112
y                         -0.0084    0.00135200     0.0014021
======================================================================
Reduced Chi-Square: 1.042
```

#### Extract Component and Residual Maps

```python
# Compute individual foreground contribution maps
contrib_maps = result.component_maps(template_maps) # shape: (N_templates, ny, nx)

# Extract fitted amplitudes directly
amplitudes = result.amplitudes
errors = result.bootstrap_errors
```

---

## Core Mathematical Principles

1. **Pixel Window Conservation**:
   We model pixels as square top-hat filters. When smoothing from beam $\theta_A$ on grid $p_0$ to target beam $\theta_B$ on grid $p_1$, the required smoothing kernel FWHM $\theta_K$ is:
   $$\theta_K^2 = \theta_B^2 - \theta_A^2 - \theta_{\text{pix}, 0}^2 - \theta_{\text{pix}, 1}^2$$
   where $\theta_{\text{pix}} \approx 0.6798 \cdot p$ is the FWHM of the pixel's Gaussian approximation.
2. **Wiener-Deconvolution**:
   When using tabulated beams, we reconstruct the Fourier-space signal before applying the target beam to avoid multi-pixel leakage, regularised by parameter $\epsilon$:
   $$T(\ell) = \frac{B_{\text{target}}(\ell) P_{\text{new}}(\ell)}{B_{\text{input}}(\ell) P_{\text{old}}(\ell) + \epsilon}$$
3. **Rigorous Calibration & Noise Iteration**:
   In linear fits, templates themselves carry noise and calibration scaling errors. We solve this by iteratively updating the weights matrix using the fitted amplitudes:
   $$\sigma^2_{k} = \sigma^2_{\text{data}, k} + (\alpha_{\text{data}} d_k)^2 + \sum_{i} a_i^2 \sigma^2_{\text{temp}, i, k} + \sum_{i} (a_i \alpha_{\text{temp}, i} t_{i, k})^2$$

---

## Status & Roadmap

PHANTaSMA is in active scientific deployment.

- [x] Cartesian/FITS and HEALPix support for reprojection, cutout, and exact smoothing.
- [x] Complete support for arbitrary/tabulated non-Gaussian beam profiles $B(\ell)$.
- [x] Exact analytical & Monte Carlo local patch RMS propagation.
- [x] Iterative, weighted SVD template fitting with bootstrap statistics.
- [ ] Simple SED fitting (synchrotron, dust, free–free)
- [ ] MCMC-based high-dimensional parameter estimation.
- [ ] Aperture photometry suite.
- [ ] C++ accelerations for huge-scale Monte Carlo noise realisations.

---

## License

This project is licensed under the MIT License - see the [LICENSE](file:///Users/user/code/phantasma/LICENSE) file for details.

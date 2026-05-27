"""
Template fitting module for PHANTaSMA.

Implements weighted linear template fitting with:

- Iterative uncertainty propagation from calibration errors and template noise.
- Bootstrap uncertainty estimation (pixel resampling).
- SVD-based linear solver for numerical robustness.
- Optional geometric foreground templates (monopole, x/y gradients, quadratic).

Typical usage
-------------
>>> from phantasma.template_fitting import template_fit_bootstrap
>>> result = template_fit_bootstrap(data_map, template_maps, data_rms=rms_map)
>>> result.summary()

For a quick fit without bootstrap overhead:

>>> from phantasma.template_fitting import template_fit
>>> result = template_fit(data_map, template_maps, data_rms=rms_map)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import svd as _svd

try:
    from tqdm.auto import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


__all__ = [
    "TemplateFitResult",
    "make_geometric_templates",
    "template_fit",
    "template_fit_bootstrap",
    "simulate_template_fit",
]


# ============================================================
# Scientific rounding utility
# ============================================================


def _round_to_uncertainty(value, sigma, n_sig=2):
    """
    Round *value* and *sigma* so that *sigma* retains *n_sig* significant
    figures.  The value is rounded to the same decimal position.

    This follows the standard scientific convention for reporting
    measurements with uncertainties.

    Parameters
    ----------
    value : float
        Central value.
    sigma : float
        Uncertainty.  Must be positive and finite.
    n_sig : int
        Number of significant figures to keep in *sigma* (default: 2).

    Returns
    -------
    value_rounded : float
    sigma_rounded : float
    value_str : str
    sigma_str : str
    combined_str : str
        Paper-ready string, e.g. ``"1.50 \u00b1 0.17"``.

    Examples
    --------
    >>> _round_to_uncertainty(1.4991, 0.1676)
    (1.5, 0.17, '1.50', '0.17', '1.50 \u00b1 0.17')
    >>> _round_to_uncertainty(3.0002e4, 2.121e2)
    (30000.0, 200.0, '3.000e+04', '2.0e+02', '3.000e+04 \u00b1 2.0e+02')
    """
    import math

    # Fall back to generic formatting for edge cases
    if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(value)):
        v_s = f"{value:.6g}"
        s_s = f"{sigma:.6g}"
        return value, sigma, v_s, s_s, f"{v_s} \u00b1 {s_s}"

    # Power of 10 of the last significant digit we want to keep in sigma
    mag = math.floor(math.log10(sigma))  # e.g. sigma=0.167 → mag=-1
    last_pos = mag - n_sig + 1           # e.g. n_sig=2      → last_pos=-2

    # Round both to that precision
    scale = 10.0 ** last_pos
    sigma_r = round(sigma / scale) * scale
    value_r = round(value / scale) * scale

    # Choose fixed vs scientific notation based on magnitude
    use_fixed = (last_pos <= 0) and (-4 <= mag)

    if use_fixed:
        decimals = -last_pos
        s_s = f"{sigma_r:.{decimals}f}"
        v_s = f"{value_r:.{decimals}f}"
    else:
        # Scientific notation for very large / very small numbers
        s_s = f"{sigma_r:.{n_sig - 1}e}"
        if value_r != 0:
            val_mag = math.floor(math.log10(abs(value_r)))
            val_n_sig = val_mag - last_pos + 1
            v_s = f"{value_r:.{max(val_n_sig - 1, 0)}e}"
        else:
            v_s = "0"

    return value_r, sigma_r, v_s, s_s, f"{v_s} \u00b1 {s_s}"


# ============================================================
# Result dataclass
# ============================================================


@dataclass
class TemplateFitResult:
    """
    Result of a template fitting run.

    Attributes
    ----------
    coefficients : dict
        Per-parameter summary::

            {name: {
                "value",
                "formal_error",
                "bootstrap_mean",   # None if no bootstrap
                "bootstrap_error",  # None if no bootstrap
                "calib_error",      # calibration-only contribution
                "total_error",      # sqrt(stat² + calib²)
            }}

    beta : ndarray, shape (Npar,)
        Best-fit coefficient vector (physical templates first, then geometric).

    beta_names : list of str
        Name of each element of ``beta``.

    beta_formal_error : ndarray, shape (Npar,)
        Formal (analytic) 1-sigma uncertainties from the scaled covariance
        matrix.  Based on white-noise weights only.

    beta_bootstrap_mean : ndarray or None, shape (Npar,)
        Mean of bootstrap coefficient samples.

    beta_bootstrap_error : ndarray or None, shape (Npar,)
        Standard deviation of bootstrap samples.  Captures the statistical
        (white-noise) uncertainty only — calibration is NOT included here.

    beta_bootstrap_cov : ndarray or None, shape (Npar, Npar)
        Covariance matrix estimated from bootstrap samples.

    beta_calib_error : ndarray, shape (Npar,)
        Uncertainty contribution from globally-correlated calibration errors,
        propagated analytically after the fit::

            σ_calib(aᵢ)² = (aᵢ · f_data_cal)² + (aᵢ · f_template_cal_i)²

        For geometric coefficients only the data-calibration term enters
        (geometric templates carry no calibration error of their own).

    beta_total_error : ndarray, shape (Npar,)
        Total 1-sigma uncertainty — statistical and calibration added in
        quadrature::

            σ_total = sqrt(σ_stat² + σ_calib²)

        where ``σ_stat`` is ``beta_bootstrap_error`` (bootstrap run) or
        ``beta_formal_error`` (``template_fit``).

    model_map : ndarray, shape (ny, nx)
        Best-fit model map (NaN outside the valid mask).

    residual_map : ndarray, shape (ny, nx)
        Residual map: ``data_map - model_map``.

    sigma_eff_map : ndarray, shape (ny, nx)
        Effective per-pixel white-noise uncertainty used in the last iteration.

    valid_mask : ndarray of bool, shape (ny, nx)
        ``True`` where pixels were included in the fit.

    chi2 : float
        Chi-squared statistic of the best fit.

    chi2_red : float
        Reduced chi-squared: ``chi2 / ndof``.

    ndof : int
        Number of degrees of freedom (effective, based on SVD rank).

    n_pix : int
        Number of valid pixels used.

    n_iter_done : int
        Number of iterations actually performed.

    converged : bool
        Whether the iterative fit converged within ``tol``.

    bootstrap_chi2_red : ndarray or None, shape (n_bootstrap,)
        Reduced chi-squared for each bootstrap realisation.

    bootstrap_converged : ndarray of bool or None, shape (n_bootstrap,)
        Per-realisation convergence flag.

    bootstrap_mode : str or None
        Bootstrap sampling mode (currently always ``"pixel"``).

    beta_cov : ndarray, shape (Npar, Npar)
        Parameter covariance matrix. Holds bootstrap covariance matrix if
        bootstrap was performed, otherwise the formal covariance matrix.

    beta_corr : ndarray, shape (Npar, Npar)
        Parameter correlation matrix derived from ``beta_cov``.

    pearson_r : float
        Pearson correlation coefficient between the actual data map and the
        best-fit model map, computed within the valid-pixel mask.

    r2 : float
        Coefficient of determination (R-squared) between the data map and
        the best-fit model map, computed within the valid-pixel mask.

    template_corr : ndarray, shape (Npar, Npar)
        Spatial correlation matrix between the template maps (including
        geometric templates, if any), computed within the valid-pixel mask.

    beta_bootstrap_samples : ndarray or None, shape (n_bootstrap, Npar)
        Raw bootstrap samples (only when ``return_bootstrap_samples=True``).
    """

    coefficients: dict
    beta: np.ndarray
    beta_names: list
    beta_formal_error: np.ndarray
    beta_bootstrap_mean: Optional[np.ndarray]
    beta_bootstrap_error: Optional[np.ndarray]
    beta_bootstrap_cov: Optional[np.ndarray]
    beta_calib_error: np.ndarray
    beta_total_error: np.ndarray
    model_map: np.ndarray
    residual_map: np.ndarray
    sigma_eff_map: np.ndarray
    valid_mask: np.ndarray
    chi2: float
    chi2_red: float
    ndof: int
    n_pix: int
    n_iter_done: int
    converged: bool
    bootstrap_chi2_red: Optional[np.ndarray]
    bootstrap_converged: Optional[np.ndarray]
    bootstrap_mode: Optional[str]
    beta_cov: np.ndarray
    beta_corr: np.ndarray
    pearson_r: float
    r2: float
    template_corr: np.ndarray
    beta_bootstrap_samples: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def _format_correlation_matrix(self, names: list[str], matrix: np.ndarray) -> str:
        lines = []
        header = f"{'':<16}" + "".join(f"{name[:11]:>12}" for name in names)
        lines.append(header)
        for i, name in enumerate(names):
            row = f"{name[:16]:<16}" + "".join(f"{matrix[i, j]:>12.3f}" for j in range(len(names)))
            lines.append(row)
        return "\n".join(lines)

    def summary(self, n_sig: int = 2) -> dict:
        """
        Print a paper-ready summary of the fit and return the rounded values.

        The uncertainty column used for rounding is always ``total_error``
        (bootstrap + calibration in quadrature).  The central value is
        rounded to the same decimal position as the rounded uncertainty.

        Parameters
        ----------
        n_sig : int
            Number of significant figures to keep in the uncertainty
            (default: 2, standard in most astronomy journals).

        Returns
        -------
        out : dict with keys:

            ``"text"``
                The formatted summary string that was printed.

            ``"rounded"``
                Per-parameter dict::

                    {name: {
                        "value"           : float  (rounded),
                        "total_error"     : float  (rounded),
                        "value_str"       : str,
                        "total_error_str" : str,
                        "paper_str"       : str,   # e.g. "1.50 \u00b1 0.17"
                    }}

            ``"raw"``
                Per-parameter dict with full-precision values::

                    {name: {
                        "value"       : float,
                        "stat_error"  : float,   # bootstrap or formal
                        "calib_error" : float,
                        "total_error" : float,
                    }}

            ``"chi2"``
                Best-fit absolute chi-squared.

            ``"chi2_red"``
                Reduced chi-squared.

            ``"ndof"``
                Degrees of freedom.

            ``"pearson_r"``
                Pearson correlation coefficient.

            ``"r2"``
                R-squared.

            ``"beta_cov"``
                Covariance matrix of parameters.

            ``"beta_corr"``
                Correlation matrix of parameters.

            ``"template_corr"``
                Template spatial correlation matrix.
        """
        has_boot = self.beta_bootstrap_error is not None

        # ---- Build rounded / raw representations per parameter ----
        rounded = {}
        raw = {}

        for name, info in self.coefficients.items():
            stat_err = (
                info["bootstrap_error"]
                if (has_boot and info["bootstrap_error"] is not None)
                else info["formal_error"]
            )
            v_r, s_r, v_s, s_s, paper = _round_to_uncertainty(
                info["value"], info["total_error"], n_sig=n_sig
            )
            rounded[name] = {
                "value": v_r,
                "total_error": s_r,
                "value_str": v_s,
                "total_error_str": s_s,
                "paper_str": paper,
            }
            raw[name] = {
                "value": info["value"],
                "stat_error": stat_err,
                "calib_error": info["calib_error"],
                "total_error": info["total_error"],
            }

        # ---- Format printed table ----
        col_w = 88
        paper_col = 28
        stat_label = "Stat \u03c3" if has_boot else "Formal \u03c3"

        lines = []
        lines.append("=" * col_w)
        lines.append("Template Fit Summary")
        lines.append("=" * col_w)
        lines.append(
            f"{'Parameter':<24} {'Paper value':<{paper_col}}"
            f" {stat_label:>13} {'Calib \u03c3':>13} {'Total \u03c3':>13}"
        )
        lines.append("-" * col_w)

        for name, info in self.coefficients.items():
            stat_err = (
                info["bootstrap_error"]
                if (has_boot and info["bootstrap_error"] is not None)
                else info["formal_error"]
            )
            row = (
                f"{name:<24}"
                f" {rounded[name]['paper_str']:<{paper_col}}"
                f" {stat_err:>13.3g}"
                f" {info['calib_error']:>13.3g}"
                f" {info['total_error']:>13.3g}"
            )
            lines.append(row)

        lines.append("-" * col_w)
        lines.append(f"\u03c7\u00b2        = {self.chi2:.4f}")
        lines.append(f"\u03c7\u00b2_red    = {self.chi2_red:.4f}  (white-noise weights only)")
        lines.append(f"ndof      = {self.ndof}")
        lines.append(f"Pearson r = {self.pearson_r:.4f}  (data vs model)")
        lines.append(f"R\u00b2        = {self.r2:.4f}  (coefficient of determination)")
        lines.append(f"n_pix     = {self.n_pix}")
        lines.append(f"n_iter    = {self.n_iter_done}")
        lines.append(
            f"converged = {'yes' if self.converged else 'NO  \u2190 consider increasing n_iter'}"
        )
        if has_boot and self.bootstrap_converged is not None:
            n_boot_conv = int(np.sum(self.bootstrap_converged))
            n_boot = len(self.bootstrap_converged)
            lines.append(f"bootstrap convergence: {n_boot_conv}/{n_boot}")

        # Add correlation matrices if we have multiple parameters
        if len(self.beta_names) > 1:
            lines.append("-" * col_w)
            lines.append("Parameter Correlation Matrix:")
            lines.append(self._format_correlation_matrix(self.beta_names, self.beta_corr))
            lines.append("-" * col_w)
            lines.append("Template Spatial Correlation Matrix (within mask):")
            lines.append(self._format_correlation_matrix(self.beta_names, self.template_corr))

        lines.append("-" * col_w)
        lines.append(
            f"rounding: {n_sig} significant figure(s) in the uncertainty  "
            f"[use result.summary(n_sig=N) to change]"
        )
        lines.append("=" * col_w)

        text = "\n".join(lines)
        print(text)
        return {
            "text": text,
            "rounded": rounded,
            "raw": raw,
            "chi2": self.chi2,
            "chi2_red": self.chi2_red,
            "ndof": self.ndof,
            "pearson_r": self.pearson_r,
            "r2": self.r2,
            "beta_cov": self.beta_cov,
            "beta_corr": self.beta_corr,
            "template_corr": self.template_corr,
        }

    def component_maps(self, template_maps: np.ndarray) -> np.ndarray:
        """
        Compute the individual template contribution maps.

        For each physical template *i*, the contribution is
        ``beta[i] * template_maps[i]``.

        Parameters
        ----------
        template_maps : ndarray, shape (Ntemp, ny, nx) or (ny, nx)
            The physical template maps that were used in the fit.

        Returns
        -------
        maps : ndarray, shape (Ntemp, ny, nx)
            Scaled template contribution maps.

        Notes
        -----
        Only the physical-template amplitudes are used. Geometric-template
        contributions are not included here.
        """
        template_maps = _as_template_cube(template_maps, name="template_maps")
        ntemp = template_maps.shape[0]
        beta_phys = self.beta[:ntemp]
        return beta_phys[:, None, None] * template_maps

    @property
    def fractional_residual_map(self) -> np.ndarray:
        """
        Compute the fractional residual map: ``(data_map - model_map) / data_map``.

        Returns
        -------
        frac_map : ndarray, shape (ny, nx)
            Fractional residual map (NaN outside valid mask or where data is 0).
        """
        data = self.model_map + self.residual_map
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = self.residual_map / data
        return frac


# ============================================================
# Geometric templates
# ============================================================


def make_geometric_templates(shape, include=("monopole", "x", "y")):
    """
    Create simple geometric templates for 2D maps.

    Parameters
    ----------
    shape : tuple
        Map shape: ``(ny, nx)``.

    include : tuple of str
        Templates to include. Options: ``"monopole"``, ``"x"``,
        ``"y"``, ``"r2"``.

    Returns
    -------
    geom_templates : ndarray, shape (Ngeom, ny, nx)
        Geometric template cube, or ``None`` if *include* is empty.

    geom_names : list of str
        Names corresponding to each template.
    """
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)

    xx = (xx - np.nanmean(xx)) / np.nanstd(xx)
    yy = (yy - np.nanmean(yy)) / np.nanstd(yy)

    templates = []
    names = []

    if "monopole" in include:
        templates.append(np.ones(shape, dtype=float))
        names.append("monopole")

    if "x" in include:
        templates.append(xx)
        names.append("gradient_x")

    if "y" in include:
        templates.append(yy)
        names.append("gradient_y")

    if "r2" in include:
        r2 = xx**2 + yy**2
        r2 = (r2 - np.nanmean(r2)) / np.nanstd(r2)
        templates.append(r2)
        names.append("quadratic_r2")

    if not templates:
        return None, []

    return np.asarray(templates, dtype=float), names


# ============================================================
# Input validation helpers
# ============================================================


def _as_template_cube(arr, name="array"):
    """
    Ensure an array has shape ``(N, ny, nx)``.

    Accepted inputs:
        ``(ny, nx)``     → converted to ``(1, ny, nx)``.
        ``(N, ny, nx)``  → returned unchanged.

    Any other dimensionality raises ``ValueError``.
    """
    if arr is None:
        return None

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (ny, nx) or (N, ny, nx). "
            f"Received shape {arr.shape}."
        )

    return arr


def _check_same_spatial_shape(reference_shape, arr, name):
    """
    Raise ``ValueError`` if *arr* has incompatible spatial dimensions.

    Parameters
    ----------
    reference_shape : tuple
        Expected spatial shape ``(ny, nx)``.
    arr : ndarray or None
        Array to check (2D or 3D). Ignored when ``None``.
    name : str
        Label used in error messages.
    """
    if arr is None:
        return

    arr = np.asarray(arr)

    if arr.ndim == 2:
        spatial_shape = arr.shape
    elif arr.ndim == 3:
        spatial_shape = arr.shape[1:]
    else:
        raise ValueError(
            f"{name} must be either 2D or 3D. Received shape {arr.shape}."
        )

    if spatial_shape != reference_shape:
        raise ValueError(
            f"{name} has incompatible spatial dimensions. "
            f"Expected {reference_shape}, received {spatial_shape}."
        )


def _validate_all_input_shapes(
    data_map,
    template_maps,
    geom_templates=None,
    data_rms=None,
    template_rms=None,
    mask=None,
):
    """
    Validate that all input maps share the same spatial dimensions.

    Also verifies that *template_rms*, when provided, has exactly the
    same shape as *template_maps*.
    """
    data_map = np.asarray(data_map)

    if data_map.ndim != 2:
        raise ValueError(
            f"data_map must be 2D, with shape (ny, nx). "
            f"Received shape {data_map.shape}."
        )

    reference_shape = data_map.shape

    _check_same_spatial_shape(reference_shape, template_maps, "template_maps")
    _check_same_spatial_shape(reference_shape, geom_templates, "geom_templates")
    _check_same_spatial_shape(reference_shape, data_rms, "data_rms")
    _check_same_spatial_shape(reference_shape, template_rms, "template_rms")
    _check_same_spatial_shape(reference_shape, mask, "mask")

    template_maps_cube = _as_template_cube(template_maps, name="template_maps")

    if template_rms is not None:
        template_rms_cube = _as_template_cube(template_rms, name="template_rms")

        if template_rms_cube.shape != template_maps_cube.shape:
            raise ValueError(
                "template_rms must have exactly the same shape as template_maps. "
                f"template_maps has shape {template_maps_cube.shape}, "
                f"template_rms has shape {template_rms_cube.shape}."
            )

    if geom_templates is not None:
        _as_template_cube(geom_templates, name="geom_templates")

    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != reference_shape:
            raise ValueError(
                f"mask must have shape {reference_shape}. "
                f"Received shape {mask.shape}."
            )

    return reference_shape


def _validate_calib_frac(calib_frac, n, name):
    """
    Normalise calibration fractional uncertainties to an array of length *n*.

    Accepted inputs:
        ``None``       → all zeros.
        scalar         → repeated for all *n* templates.
        array of len n → used directly.
    """
    if calib_frac is None:
        return np.zeros(n, dtype=float)

    calib_frac = np.asarray(calib_frac, dtype=float)

    if calib_frac.ndim == 0:
        return np.full(n, float(calib_frac), dtype=float)

    if calib_frac.size != n:
        raise ValueError(
            f"{name} must be either a scalar or have length {n}. "
            f"Received length {calib_frac.size}."
        )

    return calib_frac.astype(float)


# ============================================================
# Mask and fitting helpers
# ============================================================


def _build_valid_mask(
    data_map,
    template_maps,
    geom_templates=None,
    data_rms=None,
    template_rms=None,
    mask=None,
):
    """
    Build the final valid-pixel mask.

    Convention:
        ``True``  → pixel is included in the fit.
        ``False`` → pixel is excluded.

    A pixel is valid only when *all* of the following hold:

    * ``data_map`` is finite.
    * ``data_rms`` (if given) is finite and positive.
    * Every ``template_maps[i]`` is finite.
    * Every ``template_rms[i]`` (if given) is finite and non-negative.
    * Every ``geom_templates[j]`` (if given) is finite.
    * The external *mask* (if given) is ``True``.
    """
    _validate_all_input_shapes(
        data_map=data_map,
        template_maps=template_maps,
        geom_templates=geom_templates,
        data_rms=data_rms,
        template_rms=template_rms,
        mask=mask,
    )

    data_map = np.asarray(data_map, dtype=float)
    template_maps = _as_template_cube(template_maps, name="template_maps")

    if geom_templates is not None:
        geom_templates = _as_template_cube(geom_templates, name="geom_templates")

    if template_rms is not None:
        template_rms = _as_template_cube(template_rms, name="template_rms")

    valid = np.isfinite(data_map)

    if data_rms is not None:
        data_rms = np.asarray(data_rms, dtype=float)
        valid &= np.isfinite(data_rms)
        valid &= data_rms > 0

    for i in range(template_maps.shape[0]):
        valid &= np.isfinite(template_maps[i])

    if template_rms is not None:
        for i in range(template_rms.shape[0]):
            valid &= np.isfinite(template_rms[i])
            valid &= template_rms[i] >= 0

    if geom_templates is not None:
        for i in range(geom_templates.shape[0]):
            valid &= np.isfinite(geom_templates[i])

    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    return valid


def _weighted_linear_fit(y, X, sigma):
    """
    Solve the weighted linear model ``y = X beta`` via SVD.

    Uses the equivalent scaled ordinary least-squares formulation:

        ``y' = y / sigma``,   ``X' = X / sigma[:, None]``

    and computes ``beta`` via the SVD of ``X'``.  This approach is
    numerically more stable than the normal-equation route
    (``X^T W X beta = X^T W y``) when the design matrix is
    ill-conditioned.

    Parameters
    ----------
    y : ndarray, shape (Npix,)
        Data vector.
    X : ndarray, shape (Npix, Npar)
        Design matrix.
    sigma : ndarray, shape (Npix,)
        Per-pixel 1-sigma uncertainty.

    Returns
    -------
    result : dict
        Keys: ``beta``, ``cov``, ``cov_scaled``, ``formal_error``,
        ``model``, ``residual``, ``chi2``, ``chi2_red``, ``ndof``,
        ``n_pix``, ``rank``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if y.ndim != 1:
        raise ValueError(f"y must be 1D. Received shape {y.shape}.")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D. Received shape {X.shape}.")
    if sigma.ndim != 1:
        raise ValueError(f"sigma must be 1D. Received shape {sigma.shape}.")
    if X.shape[0] != y.size:
        raise ValueError(
            f"X and y are incompatible: X has {X.shape[0]} rows, "
            f"but y has length {y.size}."
        )
    if sigma.size != y.size:
        raise ValueError(
            f"sigma and y are incompatible: sigma has length {sigma.size}, "
            f"but y has length {y.size}."
        )

    good = (
        np.isfinite(y)
        & np.all(np.isfinite(X), axis=1)
        & np.isfinite(sigma)
        & (sigma > 0)
    )

    y = y[good]
    X = X[good]
    sigma = sigma[good]

    if y.size <= X.shape[1]:
        raise ValueError(
            "The number of valid data points is insufficient for the number of parameters. "
            f"Npix = {y.size}, Npar = {X.shape[1]}."
        )

    # Scale to standard form so that WLS becomes OLS
    y_s = y / sigma
    X_s = X / sigma[:, None]

    # Thin SVD of the scaled design matrix
    U, s, Vt = _svd(X_s, full_matrices=False)

    # Effective rank (numerical threshold)
    rcond = np.finfo(float).eps * max(X_s.shape) * s[0]
    s_inv = np.where(s > rcond, 1.0 / s, 0.0)
    rank = int(np.sum(s > rcond))

    # Solution: beta = V S^{-1} U^T y'
    beta = Vt.T @ (s_inv * (U.T @ y_s))

    # Covariance of beta: (X'^T X')^{-1} = V S^{-2} V^T
    cov = (Vt.T * s_inv**2) @ Vt

    model = X @ beta
    residual = y - model

    chi2 = np.sum((residual / sigma) ** 2)
    ndof = y.size - rank

    if ndof > 0:
        chi2_red = chi2 / ndof
        cov_scaled = cov * chi2_red
    else:
        chi2_red = np.nan
        cov_scaled = cov.copy()

    formal_error = np.sqrt(np.diag(cov_scaled))

    return {
        "beta": beta,
        "cov": cov,
        "cov_scaled": cov_scaled,
        "formal_error": formal_error,
        "model": model,
        "residual": residual,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "ndof": ndof,
        "n_pix": y.size,
        "rank": rank,
    }


def _compute_white_noise_sigmas(data_rms_vec, template_rms_vecs):
    """
    Compute per-pixel white-noise variances for data and physical templates.

    These are the *uncorrelated* noise contributions that enter the WLS
    weights and the bootstrap.  Globally-correlated calibration errors are
    handled separately by :func:`_compute_calib_uncertainty_on_beta`.

    Parameters
    ----------
    data_rms_vec : ndarray, shape (Npix,)
        Per-pixel white-noise RMS of the target map.
    template_rms_vecs : ndarray, shape (Npix, Ntemp)
        Per-pixel white-noise RMS of each physical template.

    Returns
    -------
    sigma2_data_white : ndarray, shape (Npix,)
    sigma2_template_white : ndarray, shape (Npix, Ntemp)
    """
    sigma2_data_white = data_rms_vec ** 2
    sigma2_template_white = template_rms_vecs ** 2
    return sigma2_data_white, sigma2_template_white


def _compute_calib_uncertainty_on_beta(
    beta, ntemp, data_calib_frac, template_calib_frac
):
    """
    Compute the calibration contribution to the uncertainty on each coefficient.

    Calibration errors are *globally correlated* across all pixels — the
    entire map (or template) is scaled by the same unknown factor.  They
    therefore cannot be captured by pixel resampling (bootstrap) and must
    be propagated analytically.

    For a data-calibration error ``ε_data`` (σ = f_data_cal):

        All fitted coefficients shift by ``β · ε_data``:

            ``δβᵢ = βᵢ · ε_data``   →   ``σ_calib_data(βᵢ)² = (βᵢ · f_data_cal)²``

    For a template-calibration error ``εᵢ`` on template *i* (σ = f_template_cal_i):

        Only the *i*-th physical coefficient shifts:

            ``δaᵢ = -aᵢ · εᵢ``   →   ``σ_calib_template(aᵢ)² = (aᵢ · f_template_cal_i)²``

    Both contributions are added in quadrature.  Geometric template
    coefficients (indices ≥ ntemp) are only affected by the data calibration.

    Parameters
    ----------
    beta : ndarray, shape (Npar,)
        Best-fit coefficient vector.
    ntemp : int
        Number of physical templates (``beta[:ntemp]`` are physical coefficients).
    data_calib_frac : float
        Fractional calibration uncertainty of the target map.
    template_calib_frac : ndarray, shape (Ntemp,)
        Fractional calibration uncertainties of the physical templates.

    Returns
    -------
    sigma_calib : ndarray, shape (Npar,)
        1-sigma calibration uncertainty on each coefficient.
    """
    sigma2_calib = (beta * data_calib_frac) ** 2  # data cal affects all params

    for i in range(ntemp):                         # template cal only for physical
        sigma2_calib[i] += (beta[i] * template_calib_frac[i]) ** 2

    return np.sqrt(sigma2_calib)


def _run_iterative_fit(
    y, X, sigma2_data_white, sigma2_template_white, ntemp, n_iter, tol
):
    """
    Run the iterative weighted fit until convergence.

    Uses *white-noise-only* variances in the per-pixel weights::

        σ²_tot(p) = σ²_data_white(p) + Σᵢ aᵢ² · σ²_template_white_i(p)

    Calibration errors are globally correlated and are therefore NOT included
    here; they are propagated analytically by the caller via
    :func:`_compute_calib_uncertainty_on_beta`.

    The iteration is needed because the template white-noise contribution
    depends on the fitted amplitudes ``aᵢ``, which change each iteration.

    Returns
    -------
    fit : dict
        Last :func:`_weighted_linear_fit` result.
    sigma : ndarray, shape (Npix,)
        Final effective white-noise sigma vector.
    n_iter_done : int
    converged : bool
    """
    # Seed fit using data white noise only
    sigma = np.sqrt(sigma2_data_white)
    fit = _weighted_linear_fit(y, X, sigma)
    beta_old = fit["beta"].copy()

    n_iter_done = 0
    converged = False

    for iteration in range(n_iter):
        n_iter_done = iteration + 1

        a_templates = beta_old[:ntemp]

        sigma2_tot = sigma2_data_white.copy()
        for i in range(ntemp):
            sigma2_tot += (a_templates[i] ** 2) * sigma2_template_white[:, i]

        sigma = np.sqrt(sigma2_tot)
        fit = _weighted_linear_fit(y, X, sigma)
        beta_new = fit["beta"].copy()

        denom = np.maximum(np.abs(beta_old), 1e-30)
        rel_change = np.max(np.abs((beta_new - beta_old) / denom))
        beta_old = beta_new

        if rel_change < tol:
            converged = True
            break

    return fit, sigma, n_iter_done, converged


# ============================================================
# Public fitting functions
# ============================================================


def template_fit(
    data_map,
    template_maps,
    geom_templates=None,
    data_rms=None,
    template_rms=None,
    data_calib_frac=0.0,
    template_calib_frac=None,
    mask=None,
    template_names=None,
    geom_names=None,
    n_iter=20,
    tol=1e-6,
):
    """
    Weighted linear template fitting without bootstrap.

    Fits the model:

        ``data_map = Σ_i a_i template_i + Σ_j b_j geom_j + residual``

    Iteratively updates the total per-pixel uncertainty to account for
    calibration errors and template noise.  Use this function for fast
    exploratory fits; for rigorous uncertainty estimation use
    :func:`template_fit_bootstrap`.

    Parameters
    ----------
    data_map : ndarray, shape (ny, nx)
        Target map.

    template_maps : ndarray, shape (Ntemp, ny, nx) or (ny, nx)
        Physical template maps.

    geom_templates : ndarray, shape (Ngeom, ny, nx) or (ny, nx) or None
        Geometric foreground templates (e.g. from
        :func:`make_geometric_templates`).

    data_rms : ndarray, shape (ny, nx)
        Per-pixel white-noise RMS of the target map.

    template_rms : ndarray, shape (Ntemp, ny, nx) or None
        Per-pixel white-noise RMS of the physical templates.
        Assumed zero when ``None``.

    data_calib_frac : float
        Fractional calibration uncertainty of the target map
        (e.g. ``0.05`` for 5 %).

    template_calib_frac : float or array-like or None
        Fractional calibration uncertainties of the physical templates.
        Scalar or array of length *Ntemp*.

    mask : ndarray of bool, shape (ny, nx) or None
        External mask. ``True`` → pixel is used.

    template_names : list of str or None
        Names for the physical templates.

    geom_names : list of str or None
        Names for the geometric templates.

    n_iter : int
        Maximum number of iterations used to update ``σ_tot``.

    tol : float
        Relative convergence tolerance for the fitted coefficients.

    Returns
    -------
    result : TemplateFitResult
        Fit result. Bootstrap fields are ``None``.
    """
    # ---- Validate & standardise inputs ----
    _validate_all_input_shapes(
        data_map=data_map,
        template_maps=template_maps,
        geom_templates=geom_templates,
        data_rms=data_rms,
        template_rms=template_rms,
        mask=mask,
    )

    data_map = np.asarray(data_map, dtype=float)
    template_maps = _as_template_cube(template_maps, name="template_maps")
    ntemp = template_maps.shape[0]

    if geom_templates is not None:
        geom_templates = _as_template_cube(geom_templates, name="geom_templates")
        ngeom = geom_templates.shape[0]
    else:
        ngeom = 0

    if data_rms is None:
        raise ValueError("data_rms must be provided.")
    data_rms = np.asarray(data_rms, dtype=float)

    if template_rms is None:
        template_rms = np.zeros_like(template_maps, dtype=float)
    else:
        template_rms = _as_template_cube(template_rms, name="template_rms")

    data_calib_frac = float(data_calib_frac)
    template_calib_frac = _validate_calib_frac(
        template_calib_frac, ntemp, "template_calib_frac"
    )

    if template_names is None:
        template_names = [f"template_{i}" for i in range(ntemp)]
    if len(template_names) != ntemp:
        raise ValueError(
            f"template_names must have length {ntemp}. "
            f"Received length {len(template_names)}."
        )

    if geom_templates is not None:
        if geom_names is None:
            geom_names = [f"geom_{i}" for i in range(ngeom)]
        if len(geom_names) != ngeom:
            raise ValueError(
                f"geom_names must have length {ngeom}. "
                f"Received length {len(geom_names)}."
            )
    else:
        geom_names = []

    beta_names = list(template_names) + list(geom_names)

    # ---- Valid-pixel mask ----
    valid = _build_valid_mask(
        data_map=data_map,
        template_maps=template_maps,
        geom_templates=geom_templates,
        data_rms=data_rms,
        template_rms=template_rms,
        mask=mask,
    )

    n_valid = int(np.sum(valid))
    n_par = ntemp + ngeom

    if n_valid <= n_par:
        raise ValueError(
            "The number of valid pixels is insufficient for the number of parameters. "
            f"Nvalid = {n_valid}, Npar = {n_par}."
        )

    # ---- Flatten maps ----
    y = data_map[valid]
    data_rms_vec = data_rms[valid]
    template_vecs = template_maps[:, valid].T
    template_rms_vecs = template_rms[:, valid].T

    if geom_templates is not None:
        geom_vecs = geom_templates[:, valid].T
        X = np.column_stack([template_vecs, geom_vecs])
    else:
        X = template_vecs

    # ---- White-noise variances (used in the fit and bootstrap) ----
    sigma2_data_white, sigma2_template_white = _compute_white_noise_sigmas(
        data_rms_vec=data_rms_vec,
        template_rms_vecs=template_rms_vecs,
    )

    # ---- Iterative fit ----
    fit, sigma_best, n_iter_done, converged = _run_iterative_fit(
        y, X, sigma2_data_white, sigma2_template_white, ntemp, n_iter, tol
    )

    if not converged:
        warnings.warn(
            f"template_fit did not converge in {n_iter} iterations. "
            "Consider increasing n_iter or checking your inputs.",
            RuntimeWarning,
            stacklevel=2,
        )

    beta_best = fit["beta"]

    # ---- Reconstruct 2-D maps ----
    model_map = np.full(data_map.shape, np.nan)
    residual_map = np.full(data_map.shape, np.nan)
    sigma_eff_map = np.full(data_map.shape, np.nan)

    model_map[valid] = X @ beta_best
    residual_map[valid] = y - model_map[valid]
    sigma_eff_map[valid] = sigma_best

    # ---- Calibration uncertainty (analytic, globally correlated) ----
    beta_calib_error = _compute_calib_uncertainty_on_beta(
        beta_best, ntemp, data_calib_frac, template_calib_frac
    )
    beta_total_error = np.sqrt(fit["formal_error"] ** 2 + beta_calib_error ** 2)

    # ---- Additional statistics ----
    y_model = X @ beta_best
    if y.size > 1 and np.std(y) > 0 and np.std(y_model) > 0:
        pearson_r = float(np.corrcoef(y, y_model)[0, 1])
    else:
        pearson_r = 0.0

    if y.size > 1 and np.std(y) > 0:
        r2 = float(1.0 - np.sum((y - y_model) ** 2) / np.sum((y - np.mean(y)) ** 2))
    else:
        r2 = 0.0

    beta_cov = np.atleast_2d(fit["cov_scaled"])
    diag = np.diag(beta_cov)
    std = np.sqrt(np.where(diag > 0, diag, 1.0))
    beta_corr = np.clip(beta_cov / std[:, None] / std[None, :], -1.0, 1.0)

    if X.shape[0] > 1:
        template_corr = np.corrcoef(X, rowvar=False)
        template_corr = np.atleast_2d(template_corr)
    else:
        template_corr = np.eye(n_par)

    # ---- Organise output ----
    coefficients = {
        name: {
            "value": beta_best[i],
            "formal_error": fit["formal_error"][i],
            "bootstrap_mean": None,
            "bootstrap_error": None,
            "calib_error": beta_calib_error[i],
            "total_error": beta_total_error[i],
        }
        for i, name in enumerate(beta_names)
    }

    return TemplateFitResult(
        coefficients=coefficients,
        beta=beta_best,
        beta_names=beta_names,
        beta_formal_error=fit["formal_error"],
        beta_bootstrap_mean=None,
        beta_bootstrap_error=None,
        beta_bootstrap_cov=None,
        beta_calib_error=beta_calib_error,
        beta_total_error=beta_total_error,
        model_map=model_map,
        residual_map=residual_map,
        sigma_eff_map=sigma_eff_map,
        valid_mask=valid,
        chi2=fit["chi2"],
        chi2_red=fit["chi2_red"],
        ndof=fit["ndof"],
        n_pix=fit["n_pix"],
        n_iter_done=n_iter_done,
        converged=converged,
        bootstrap_chi2_red=None,
        bootstrap_converged=None,
        bootstrap_mode=None,
        beta_cov=beta_cov,
        beta_corr=beta_corr,
        pearson_r=pearson_r,
        r2=r2,
        template_corr=template_corr,
    )


def template_fit_bootstrap(
    data_map,
    template_maps,
    geom_templates=None,
    data_rms=None,
    template_rms=None,
    data_calib_frac=0.0,
    template_calib_frac=None,
    mask=None,
    template_names=None,
    geom_names=None,
    n_iter=20,
    tol=1e-6,
    n_bootstrap=1000,
    bootstrap_mode="pixel",
    random_seed=1234,
    return_bootstrap_samples=False,
    show_progress=True,
):
    """
    Weighted linear template fitting with bootstrap uncertainty estimation.

    Fits the model:

        ``data_map = Σ_i a_i template_i + Σ_j b_j geom_j + residual``

    **Noise model for the iterative fit and bootstrap:**

    Only *uncorrelated* (white) noise enters the per-pixel weights::

        σ²_tot(p) = σ²_data_rms(p) + Σᵢ aᵢ² · σ²_template_rms_i(p)

    **Calibration uncertainties** (globally correlated across all pixels)
    are propagated *analytically* after the fit and added in quadrature to
    the bootstrap uncertainty::

        σ_calib(aᵢ)² = (aᵢ · f_data_cal)² + (aᵢ · f_template_cal_i)²
        σ_total(aᵢ)  = sqrt(σ_bootstrap(aᵢ)² + σ_calib(aᵢ)²)

    This separation is physically motivated: calibration errors shift all
    pixels by the same factor and therefore cannot be captured by pixel
    resampling.

    Parameters
    ----------
    data_map : ndarray, shape (ny, nx)
        Target map.

    template_maps : ndarray, shape (Ntemp, ny, nx) or (ny, nx)
        Physical template maps.

    geom_templates : ndarray, shape (Ngeom, ny, nx) or (ny, nx) or None
        Geometric foreground templates (e.g. from
        :func:`make_geometric_templates`).

    data_rms : ndarray, shape (ny, nx)
        Per-pixel white-noise RMS of the target map.

    template_rms : ndarray, shape (Ntemp, ny, nx) or None
        Per-pixel white-noise RMS of the physical templates.
        Assumed zero when ``None``.

    data_calib_frac : float
        Fractional calibration uncertainty of the target map
        (e.g. ``0.05`` for 5 %).

    template_calib_frac : float or array-like or None
        Fractional calibration uncertainties of the physical templates.
        Scalar or array of length *Ntemp*.

    mask : ndarray of bool, shape (ny, nx) or None
        External mask. ``True`` → pixel is used.

    template_names : list of str or None
        Names for the physical templates.

    geom_names : list of str or None
        Names for the geometric templates.

    n_iter : int
        Maximum number of iterations for updating ``σ_tot``.

    tol : float
        Relative convergence tolerance for the fitted coefficients.

    n_bootstrap : int
        Number of bootstrap pixel-resampling realisations.

    bootstrap_mode : str
        Resampling strategy.  Currently only ``"pixel"`` is supported.

    random_seed : int
        Random seed for reproducibility.

    return_bootstrap_samples : bool
        If ``True``, attach the full ``(n_bootstrap, Npar)`` sample array
        to the result as ``beta_bootstrap_samples``.

    show_progress : bool
        Show a ``tqdm`` progress bar during bootstrap (requires tqdm).

    Returns
    -------
    result : TemplateFitResult
        Fit result including bootstrap statistics.
    """
    rng = np.random.default_rng(random_seed)

    # ---- Validate & standardise inputs ----
    _validate_all_input_shapes(
        data_map=data_map,
        template_maps=template_maps,
        geom_templates=geom_templates,
        data_rms=data_rms,
        template_rms=template_rms,
        mask=mask,
    )

    data_map = np.asarray(data_map, dtype=float)
    template_maps = _as_template_cube(template_maps, name="template_maps")
    ntemp = template_maps.shape[0]

    if geom_templates is not None:
        geom_templates = _as_template_cube(geom_templates, name="geom_templates")
        ngeom = geom_templates.shape[0]
    else:
        ngeom = 0

    if data_rms is None:
        raise ValueError("data_rms must be provided.")
    data_rms = np.asarray(data_rms, dtype=float)

    if template_rms is None:
        template_rms = np.zeros_like(template_maps, dtype=float)
    else:
        template_rms = _as_template_cube(template_rms, name="template_rms")

    data_calib_frac = float(data_calib_frac)
    template_calib_frac = _validate_calib_frac(
        template_calib_frac, ntemp, "template_calib_frac"
    )

    if template_names is None:
        template_names = [f"template_{i}" for i in range(ntemp)]
    if len(template_names) != ntemp:
        raise ValueError(
            f"template_names must have length {ntemp}. "
            f"Received length {len(template_names)}."
        )

    if geom_templates is not None:
        if geom_names is None:
            geom_names = [f"geom_{i}" for i in range(ngeom)]
        if len(geom_names) != ngeom:
            raise ValueError(
                f"geom_names must have length {ngeom}. "
                f"Received length {len(geom_names)}."
            )
    else:
        geom_names = []

    beta_names = list(template_names) + list(geom_names)

    # ---- Valid-pixel mask ----
    valid = _build_valid_mask(
        data_map=data_map,
        template_maps=template_maps,
        geom_templates=geom_templates,
        data_rms=data_rms,
        template_rms=template_rms,
        mask=mask,
    )

    n_valid = int(np.sum(valid))
    n_par = ntemp + ngeom

    if n_valid <= n_par:
        raise ValueError(
            "The number of valid pixels is insufficient for the number of parameters. "
            f"Nvalid = {n_valid}, Npar = {n_par}."
        )

    # ---- Flatten maps ----
    y = data_map[valid]
    data_rms_vec = data_rms[valid]
    template_vecs = template_maps[:, valid].T
    template_rms_vecs = template_rms[:, valid].T

    if geom_templates is not None:
        geom_vecs = geom_templates[:, valid].T
        X = np.column_stack([template_vecs, geom_vecs])
    else:
        X = template_vecs

    # ---- White-noise variances (used in the fit and bootstrap) ----
    sigma2_data_white, sigma2_template_white = _compute_white_noise_sigmas(
        data_rms_vec=data_rms_vec,
        template_rms_vecs=template_rms_vecs,
    )

    # ---- Iterative fit on full data ----
    fit, sigma_best, n_iter_done, converged = _run_iterative_fit(
        y, X, sigma2_data_white, sigma2_template_white, ntemp, n_iter, tol
    )

    if not converged:
        warnings.warn(
            f"template_fit_bootstrap did not converge in {n_iter} iterations. "
            "Consider increasing n_iter or checking your inputs.",
            RuntimeWarning,
            stacklevel=2,
        )

    beta_best = fit["beta"]

    # ---- Reconstruct 2-D maps ----
    model_map = np.full(data_map.shape, np.nan)
    residual_map = np.full(data_map.shape, np.nan)
    sigma_eff_map = np.full(data_map.shape, np.nan)

    model_map[valid] = X @ beta_best
    residual_map[valid] = y - model_map[valid]
    sigma_eff_map[valid] = sigma_best

    # ---- Bootstrap (white-noise weights only) ----
    if bootstrap_mode != "pixel":
        raise ValueError("Currently, bootstrap_mode must be 'pixel'.")

    n_pix = y.size
    n_beta = X.shape[1]

    beta_boot = np.zeros((n_bootstrap, n_beta), dtype=float)
    chi2_boot = np.zeros(n_bootstrap, dtype=float)
    converged_boot = np.zeros(n_bootstrap, dtype=bool)

    iterator = range(n_bootstrap)
    if show_progress and _HAS_TQDM:
        iterator = _tqdm(iterator, desc="Bootstrap", unit="iter")
    elif show_progress and not _HAS_TQDM:
        warnings.warn(
            "tqdm is not installed; install it with `pip install tqdm` to see a progress bar.",
            ImportWarning,
            stacklevel=2,
        )

    for k in iterator:
        idx = rng.integers(0, n_pix, size=n_pix)

        y_b = y[idx]
        X_b = X[idx]
        sigma2_data_b = sigma2_data_white[idx]
        sigma2_template_b = sigma2_template_white[idx]

        fit_b, _, _, converged_k = _run_iterative_fit(
            y_b, X_b, sigma2_data_b, sigma2_template_b, ntemp, n_iter, tol
        )

        beta_boot[k] = fit_b["beta"]
        chi2_boot[k] = fit_b["chi2_red"]
        converged_boot[k] = converged_k

    beta_boot_mean = np.nanmean(beta_boot, axis=0)
    beta_boot_std = np.nanstd(beta_boot, axis=0, ddof=1)
    beta_boot_cov = np.cov(beta_boot, rowvar=False)

    # ---- Calibration uncertainty (analytic, globally correlated) ----
    beta_calib_error = _compute_calib_uncertainty_on_beta(
        beta_best, ntemp, data_calib_frac, template_calib_frac
    )
    beta_total_error = np.sqrt(beta_boot_std ** 2 + beta_calib_error ** 2)

    # ---- Additional statistics ----
    y_model = X @ beta_best
    if y.size > 1 and np.std(y) > 0 and np.std(y_model) > 0:
        pearson_r = float(np.corrcoef(y, y_model)[0, 1])
    else:
        pearson_r = 0.0

    if y.size > 1 and np.std(y) > 0:
        r2 = float(1.0 - np.sum((y - y_model) ** 2) / np.sum((y - np.mean(y)) ** 2))
    else:
        r2 = 0.0

    beta_cov = np.atleast_2d(beta_boot_cov)
    diag = np.diag(beta_cov)
    std = np.sqrt(np.where(diag > 0, diag, 1.0))
    beta_corr = np.clip(beta_cov / std[:, None] / std[None, :], -1.0, 1.0)

    if X.shape[0] > 1:
        template_corr = np.corrcoef(X, rowvar=False)
        template_corr = np.atleast_2d(template_corr)
    else:
        template_corr = np.eye(n_par)

    # ---- Organise output ----
    coefficients = {
        name: {
            "value": beta_best[i],
            "formal_error": fit["formal_error"][i],
            "bootstrap_mean": beta_boot_mean[i],
            "bootstrap_error": beta_boot_std[i],
            "calib_error": beta_calib_error[i],
            "total_error": beta_total_error[i],
        }
        for i, name in enumerate(beta_names)
    }

    result = TemplateFitResult(
        coefficients=coefficients,
        beta=beta_best,
        beta_names=beta_names,
        beta_formal_error=fit["formal_error"],
        beta_bootstrap_mean=beta_boot_mean,
        beta_bootstrap_error=beta_boot_std,
        beta_bootstrap_cov=beta_boot_cov,
        beta_calib_error=beta_calib_error,
        beta_total_error=beta_total_error,
        model_map=model_map,
        residual_map=residual_map,
        sigma_eff_map=sigma_eff_map,
        valid_mask=valid,
        chi2=fit["chi2"],
        chi2_red=fit["chi2_red"],
        ndof=fit["ndof"],
        n_pix=fit["n_pix"],
        n_iter_done=n_iter_done,
        converged=converged,
        bootstrap_chi2_red=chi2_boot,
        bootstrap_converged=converged_boot,
        bootstrap_mode=bootstrap_mode,
        beta_cov=beta_cov,
        beta_corr=beta_corr,
        pearson_r=pearson_r,
        r2=r2,
        template_corr=template_corr,
    )

    if return_bootstrap_samples:
        result.beta_bootstrap_samples = beta_boot

    return result


# ============================================================
# Simulation helper
# ============================================================


def simulate_template_fit(
    shape=(64, 64),
    true_amplitudes=None,
    noise_level=0.1,
    template_smooth_sigma=5.0,
    random_seed=42,
):
    """
    Generate a synthetic dataset to validate :func:`template_fit_bootstrap`.

    Creates *Ntemp* spatially-smooth random templates, builds a ground-truth
    signal as a linear combination with *true_amplitudes*, and adds white
    Gaussian noise.

    Parameters
    ----------
    shape : tuple of int
        Map shape ``(ny, nx)``.

    true_amplitudes : array-like or None
        True linear coefficients for each template.
        Defaults to ``[1.0, 2.0]`` (two templates).

    noise_level : float
        RMS amplitude of the white noise added to the data map.

    template_smooth_sigma : float
        Gaussian smoothing scale (in pixels) applied to each random template
        before normalisation.

    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    data_map : ndarray, shape (ny, nx)
        Simulated noisy data map.

    template_maps : ndarray, shape (Ntemp, ny, nx)
        Normalised smooth template maps.

    data_rms : ndarray, shape (ny, nx)
        Uniform noise map (constant = *noise_level*).

    true_amplitudes : ndarray, shape (Ntemp,)
        Ground-truth coefficients used to build the signal.

    Example
    -------
    >>> from phantasma.template_fitting import simulate_template_fit, template_fit_bootstrap
    >>> data, templates, rms, true_a = simulate_template_fit()
    >>> result = template_fit_bootstrap(data, templates, data_rms=rms,
    ...                                 n_bootstrap=200, show_progress=False)
    >>> result.summary()
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(random_seed)

    if true_amplitudes is None:
        true_amplitudes = np.array([1.0, 2.0])

    true_amplitudes = np.asarray(true_amplitudes, dtype=float)
    ntemp = true_amplitudes.size

    # Build smooth random templates
    template_list = []
    for _ in range(ntemp):
        raw = rng.standard_normal(shape)
        smooth = gaussian_filter(raw, sigma=template_smooth_sigma)
        smooth /= np.std(smooth)  # unit variance
        template_list.append(smooth)

    template_maps = np.array(template_list)

    # Ground-truth signal
    signal = np.zeros(shape, dtype=float)
    for a, t in zip(true_amplitudes, template_list):
        signal += a * t

    # Add noise
    noise = rng.standard_normal(shape) * noise_level
    data_map = signal + noise
    data_rms = np.full(shape, noise_level, dtype=float)

    return data_map, template_maps, data_rms, true_amplitudes

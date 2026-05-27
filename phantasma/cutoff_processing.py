"""
cutoff_processing.py
Utilities to preprocess astronomical maps: smooth to a target resolution
accounting for pixel window functions, reproject to a target WCS, and
extract a cutout.

Supports both FITS (flat-sky, provided as data+WCS) and HEALPix input maps.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel, convolve
from reproject import reproject_exact, reproject_from_healpix
import healpy as hp
import warnings


# Conversion factor: FWHM = 2 * sqrt(2 * ln2) * sigma
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247


def _pixel_window_fwhm(pixel_size_arcmin):
    """
    Approximate FWHM of the pixel window function for a square pixel.

    A square pixel of angular size delta acts as a 2D top-hat convolution.
    Its Gaussian approximation has sigma = delta / sqrt(12) per axis,
    giving FWHM = 2*sqrt(2*ln2) * delta / sqrt(12) ≈ 0.6798 * delta.

    Parameters
    ----------
    pixel_size_arcmin : float
        Pixel angular size in arcmin.

    Returns
    -------
    float
        Approximate pixel window FWHM in arcmin.
    """
    # Gaussian approximation of the square pixel top-hat:
    # sigma_pixel = pixel / sqrt(12), FWHM = 2*sqrt(2*ln2) * sigma_pixel
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_size_arcmin / np.sqrt(12.0)


def _compute_smoothing_kernel_sigma_pix(
    original_res_arcmin,
    target_res_arcmin,
    orig_pixel_size_arcmin,
    new_pixel_size_arcmin,
    current_pixel_scale_arcmin,
):
    """
    Compute the Gaussian smoothing kernel sigma in pixels.

    The kernel accounts for:
    - The original Gaussian beam (already applied to the data)
    - The original pixel window (already baked into the data)
    - The new pixel window (which ``reproject_exact`` will apply)

    .. math::

        \\mathrm{FWHM_{kernel}}^2 =
            \\mathrm{FWHM_{target}}^2
          - \\mathrm{FWHM_{beam}}^2
          - \\mathrm{FWHM_{pix,orig}}^2
          - \\mathrm{FWHM_{pix,new}}^2

    Parameters
    ----------
    original_res_arcmin : float
        Original *beam* FWHM in arcmin (Gaussian; does **not** include the
        pixel window — that is accounted for separately).
    target_res_arcmin : float
        Desired total effective FWHM after smoothing + reprojection [arcmin].
    orig_pixel_size_arcmin : float
        Pixel size of the *original* map [arcmin].  Its pixel window is
        already present in the data.
    new_pixel_size_arcmin : float
        Pixel size of the *output* grid [arcmin].  ``reproject_exact`` will
        effectively convolve with this pixel window.
    current_pixel_scale_arcmin : float
        Pixel scale of the intermediate grid on which the smoothing kernel
        is applied [arcmin].  Used to convert the kernel from arcmin to
        pixels.

    Returns
    -------
    sigma_pix : float or None
        Gaussian kernel sigma in pixels.  ``None`` if no smoothing is needed.
    fwhm_kernel_arcmin : float
        FWHM of the smoothing kernel in arcmin (0 if no smoothing needed).
    """
    pw_orig = _pixel_window_fwhm(orig_pixel_size_arcmin)
    pw_new  = _pixel_window_fwhm(new_pixel_size_arcmin)

    fwhm_sq = (
        target_res_arcmin**2
        - original_res_arcmin**2
        - pw_orig**2
        - pw_new**2
    )

    if fwhm_sq <= 0:
        eff = np.sqrt(original_res_arcmin**2 + pw_orig**2 + pw_new**2)
        warnings.warn(
            f"Target resolution ({target_res_arcmin:.2f}') is not larger than "
            f"sqrt(beam^2 + pix_window_orig^2 + pix_window_new^2) = "
            f"{eff:.2f}'.  No additional smoothing will be applied.",
            stacklevel=3,
        )
        return None, 0.0

    fwhm_kernel = np.sqrt(fwhm_sq)
    sigma_kernel_arcmin = fwhm_kernel * _FWHM_TO_SIGMA
    sigma_pix = sigma_kernel_arcmin / current_pixel_scale_arcmin

    return sigma_pix, fwhm_kernel


# ======================================================================
#  Beam-profile helpers (non-Gaussian beam support)
# ======================================================================

def _make_beam_evaluator(beam_spec):
    """
    Convert a beam specification to a callable ``B(ell) -> ndarray``.

    Parameters
    ----------
    beam_spec : float, tuple, or callable
        - **float** — Gaussian FWHM in arcmin.
          ``B(ℓ) = exp(−ℓ(ℓ+1) σ²/2)`` with ``σ = FWHM_rad × _FWHM_TO_SIGMA``.
        - **tuple** ``(ell_array, B_ell_array)`` — tabulated beam from a
          survey or simulation; interpolated linearly, extrapolated as
          ``B(0) = B_ell_array[0]`` and ``B(ℓ > ℓ_max) = 0``.
        - **callable** — used directly as ``B(ell)``.

    Returns
    -------
    callable : B(ell) -> ndarray
    """
    if isinstance(beam_spec, (int, float)):
        fwhm_rad = float(beam_spec) * (np.pi / (180.0 * 60.0))   # arcmin → rad
        sigma2   = (fwhm_rad * _FWHM_TO_SIGMA) ** 2
        def _gauss(ell, _s2=sigma2):
            ell = np.asarray(ell, dtype=float)
            return np.exp(-0.5 * ell * (ell + 1.0) * _s2)
        return _gauss
    elif callable(beam_spec):
        return beam_spec
    else:
        # (ell_array, B_ell_array) tuple — tabulated beam
        from scipy.interpolate import interp1d
        ell_arr = np.asarray(beam_spec[0], dtype=float)
        b_arr   = np.asarray(beam_spec[1], dtype=float)
        _interp  = interp1d(ell_arr, b_arr, kind='linear',
                            bounds_error=False,
                            fill_value=(b_arr[0], 0.0))
        return lambda ell: _interp(np.asarray(ell, dtype=float))


def _beam_effective_fwhm(beam_eval, ell_max=10_000):
    """
    Estimate the effective FWHM [arcmin] of a beam B(ℓ) as the
    angular scale where B first drops to 0.5 (half-power beam width).

    Parameters
    ----------
    beam_eval : callable
        ``B(ell) -> ndarray``
    ell_max : int
        Maximum multipole to search over.

    Returns
    -------
    float
        FWHM in arcmin.  Returns 0 if B stays ≥ 0.5 everywhere.
    """
    ell  = np.arange(0, ell_max + 1, dtype=float)
    B    = np.asarray(beam_eval(ell), dtype=float)
    below = np.where(B < 0.5)[0]
    if len(below) == 0:
        return 0.0
    ell_half = float(below[0])
    if ell_half <= 0:
        return float('inf')
    # For a Gaussian beam B(ℓ) = exp(-0.5 ℓ² σ²) = 0.5 at ℓ_half = sqrt(2·ln2)/σ
    # => FWHM_rad = 2·sqrt(2·ln2)·σ = 4·ln2 / ℓ_half
    fwhm_arcmin = (4.0 * np.log(2.0) / ell_half) * (180.0 * 60.0 / np.pi)
    return fwhm_arcmin


def _apply_beam_transfer_2d(
    data_2d,
    pixel_arcmin,
    input_beam_eval,
    target_beam_eval,
    regularization=1e-4,
):
    """
    Apply a beam transfer function in 2D Fourier space.

    **Mathematical equivalence with HEALPix**

    On the sphere, the standard operation is to multiply the spherical
    harmonic coefficients by::

        T_HEALPix(ℓ) = [B_new(ℓ) × P_new(ℓ)] / [B_old(ℓ) × P_old(ℓ)]

    where *P* is the pixel window function and *P_new* is added implicitly
    by ``hp.alm2map``.  The flat-sky analogue (used here) multiplies the 2D
    Fourier modes of the intermediate map by::

        T_applied(k) = B_target(ℓ) × H_in(k) / (H_in(k)² + ε²)

    where ``H_in(k) = B_input(ℓ(k)) × sinc(kx p) × sinc(ky p)`` is the
    combined input beam + pixel window, and ``ε`` provides Wiener
    regularisation.  ``reproject_exact`` then adds the output pixel window
    ``P_new`` implicitly, so the total effective transfer is::

        T_total = B_target × P_new    ✓

    The flat-sky ↔ ℓ relation is ``ℓ = 2π k / arcmin_per_rad``.

    Parameters
    ----------
    data_2d : 2-D ndarray
        Input map on a flat-sky intermediate grid.  NaN pixels are
        inpainted before the FFT (using a 1-pixel Gaussian) and restored
        afterwards.
    pixel_arcmin : float
        Pixel size of ``data_2d`` [arcmin].
    input_beam_eval : callable
        ``B_input(ell) -> ndarray`` — total effective beam of the input map,
        including any pixel window of the original survey pixelisation.
    target_beam_eval : callable
        ``B_target(ell) -> ndarray`` — desired output beam.  Typically a
        Gaussian obtained from ``_make_beam_evaluator(target_fwhm_arcmin)``.
    regularization : float
        Wiener regularisation parameter.  The noise floor is set to
        ``regularization × max|H_in|``.  Default ``1e-4``.

    Returns
    -------
    smoothed : 2-D ndarray, same shape as ``data_2d``.
        NaN pixels from the input are preserved in the output.
    """
    ny, nx = data_2d.shape

    # --- Inpaint NaN pixels before FFT ---
    nan_mask = ~np.isfinite(data_2d)
    if nan_mask.any():
        fill_kernel = Gaussian2DKernel(x_stddev=1.0)
        data_filled = convolve(
            data_2d, fill_kernel,
            boundary='fill', fill_value=0.0,
            nan_treatment='interpolate', preserve_nan=False,
        )
        data_filled = np.where(nan_mask, data_filled, data_2d)
    else:
        data_filled = data_2d

    # --- 2D frequency grid (cycles / arcmin) ---
    fy = np.fft.fftfreq(ny, d=pixel_arcmin)   # shape (ny,)
    fx = np.fft.fftfreq(nx, d=pixel_arcmin)   # shape (nx,)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')  # both (ny, nx)

    # Multipole ℓ (flat-sky approximation: ℓ = 2π k_rad⁻¹)
    arcmin_per_rad = 180.0 * 60.0 / np.pi
    K_mag = np.sqrt(FX**2 + FY**2)              # cycles/arcmin
    ELL   = 2.0 * np.pi * K_mag * arcmin_per_rad  # dimensionless ℓ

    # --- Evaluate beams on the 2D ℓ grid ---
    B_in  = np.clip(np.asarray(input_beam_eval(ELL),  dtype=float), 0.0, None)
    B_tgt = np.clip(np.asarray(target_beam_eval(ELL), dtype=float), 0.0, None)

    # --- Exact pixel window of the input intermediate grid ---
    # For a square pixel of side p: P(fx,fy) = sinc(fx·p) × sinc(fy·p)
    # (numpy sinc is normalized: sinc(x) = sin(πx)/(πx))
    P_orig = np.sinc(FX * pixel_arcmin) * np.sinc(FY * pixel_arcmin)

    # --- Combined input transfer H_in and Wiener-regularised T ---
    H_in = B_in * P_orig
    eps  = regularization * float(np.max(np.abs(H_in))) if H_in.max() > 0 else 1e-10
    T    = B_tgt * H_in / (H_in**2 + eps**2)

    # --- Apply in Fourier space ---
    D_fft     = np.fft.fft2(data_filled)
    smoothed  = np.real(np.fft.ifft2(D_fft * T))

    # Restore NaN mask
    if nan_mask.any():
        smoothed[nan_mask] = np.nan

    return smoothed


def _create_intermediate_wcs(center_l, center_b, size_deg, pixel_size_arcmin):
    """
    Create a gnomonic (TAN) WCS in Galactic coordinates centred on (l, b).

    Parameters
    ----------
    center_l, center_b : float
        Centre of the projection in Galactic degrees.
    size_deg : float
        Angular size of the field in degrees.
    pixel_size_arcmin : float
        Pixel size in arcmin.

    Returns
    -------
    wcs : astropy.wcs.WCS
    shape : tuple of int
        (ny, nx)
    """
    pixel_size_deg = pixel_size_arcmin / 60.0
    n_pix = int(np.ceil(size_deg / pixel_size_deg))
    if n_pix % 2 == 1:
        n_pix += 1  # keep even for symmetry

    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2.0 + 0.5, n_pix / 2.0 + 0.5]
    w.wcs.cdelt = [-pixel_size_deg, pixel_size_deg]
    w.wcs.crval = [center_l, center_b]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    return w, (n_pix, n_pix)


def make_target_wcs(center_l, center_b, pixel_size_arcmin, cutout_size_deg):
    """
    Create a gnomonic (TAN) WCS in Galactic coordinates to use as the
    output grid for ``preprocess_and_cutout``.

    Parameters
    ----------
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    pixel_size_arcmin : float
        Desired output pixel size [arcmin].
    cutout_size_deg : float
        Angular size of the cutout [deg].  The grid will be square with
        ``ceil(cutout_size_deg / pixel_size_deg)`` pixels per side
        (rounded up to the nearest even number).

    Returns
    -------
    wcs : astropy.wcs.WCS
        2-D WCS centred on (center_l, center_b) with TAN projection.
    shape : tuple of int
        ``(ny, nx)`` shape of the corresponding output grid.

    Examples
    --------
    >>> target_wcs, shape = make_target_wcs(17.0, 0.8, pixel_size_arcmin=2.0, cutout_size_deg=2.0)
    >>> data_out, wcs_out = ph.smooth_cutout(..., target_wcs=target_wcs, target_shape=shape)
    """
    return _create_intermediate_wcs(center_l, center_b, cutout_size_deg, pixel_size_arcmin)


def _sanitise(data, remove_healpix_unseen=False):
    """Replace non-finite values (inf, etc.) with NaN.

    Parameters
    ----------
    data : ndarray
    remove_healpix_unseen : bool
        If True, also replace the HEALPix UNSEEN sentinel (~-1.6e30) with NaN.
        Only needed for HEALPix input.
    """
    out = np.array(data, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    if remove_healpix_unseen:
        out[np.isclose(out, hp.UNSEEN, atol=1e-6)] = np.nan
    return out


def smooth_cutout(
    data,
    map_format="fits",
    rms_data=None,
    original_wcs=None,
    center_l=0.0,
    center_b=0.0,
    cutout_size_deg=1.0,
    original_res_arcmin=None,
    target_res_arcmin=None,
    input_beam=None,
    target_beam=None,
    pixel_size_arcmin=1.0,
    target_wcs=None,
    target_shape=None,
    healpix_coord="G",
    beam_regularization=1e-4,
):
    """
    Smooth an astronomical map to a target resolution and reproject it onto
    a target WCS grid.

    The function handles both flat-sky FITS images (provided as a numpy array
    + WCS) and HEALPix maps (provided as a 1-D pixel array).  Reprojection is
    done with ``reproject_exact`` (flux-conserving).

    The original pixel scale is read automatically from ``original_wcs`` (for
    FITS) or computed from the HEALPix Nside (for HEALPix).  The output grid
    shape is computed from ``pixel_size_arcmin`` and ``cutout_size_deg``.

    Parameters
    ----------
    data : numpy.ndarray
        - For ``map_format='fits'``: 2-D (or higher, extra axes are dropped)
          pixel array of the original map.
        - For ``map_format='healpix'``: 1-D HEALPix pixel array.
    map_format : {'fits', 'healpix'}
        Format of the input map.
    rms_data : numpy.ndarray, optional
        RMS (noise) map associated with ``data``, same shape and pixelisation.
        When provided, pixels are weighted by inverse variance (``w = 1/rms²``)
        during both the Gaussian smoothing and the reprojection steps.  Pixels
        where the RMS is zero or NaN receive zero weight and do not contribute
        to the output.
    original_wcs : astropy.wcs.WCS, optional
        WCS of the original FITS map.  Required when ``map_format='fits'``;
        ignored for ``'healpix'`` (Nside is inferred from the array length).
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    cutout_size_deg : float
        Angular size of the cutout [deg].
    original_res_arcmin : float, optional
        **Deprecated alias** for ``input_beam`` — Gaussian FWHM of the input
        beam [arcmin].  Kept for backward compatibility.
    target_res_arcmin : float, optional
        **Deprecated alias** for ``target_beam`` — Gaussian FWHM of the
        desired output beam [arcmin].  Kept for backward compatibility.
    input_beam : float, tuple, or callable, optional
        Beam profile of the input map.  Accepts:

        - **float** — Gaussian FWHM [arcmin].  Equivalent to the old
          ``original_res_arcmin`` parameter.
        - **tuple** ``(ell_array, B_ell_array)`` — tabulated beam transfer
          function as measured for the survey (e.g. downloaded from a survey
          website).  Interpolated linearly; extrapolated as ``B(ℓ → ∞) = 0``.
        - **callable** ``B(ell) -> ndarray`` — arbitrary beam function.

        When a Gaussian float is given, the code follows the original
        Gaussian-kernel path (backward compatible, exact NaN handling).
        For any other type the beam transfer is applied in 2D Fourier space
        using Wiener regularisation (see ``beam_regularization``).
    target_beam : float, optional
        Target beam FWHM [arcmin] (always Gaussian).  Equivalent to the old
        ``target_res_arcmin`` parameter.
    pixel_size_arcmin : float
        Pixel size of the *output* grid [arcmin].
    target_wcs : astropy.wcs.WCS
        WCS of the output grid.
    target_shape : tuple of int, optional
        Shape ``(ny, nx)`` of the output grid.  Computed automatically if not
        provided.
    healpix_coord : str, optional
        Coordinate system of the HEALPix map (``'G'``, ``'C'``, ``'E'``).
    beam_regularization : float, optional
        Wiener regularisation strength for the Fourier-space beam transfer.
        The noise floor is set to ``beam_regularization × max|H_in|``.
        Only used when ``input_beam`` is non-Gaussian.  Default ``1e-4``.

    Returns
    -------
    cutout_data : numpy.ndarray
        2-D map reprojected onto ``target_wcs`` with NaN for invalid pixels.
    target_wcs : astropy.wcs.WCS
        The WCS of the returned map.

    Notes
    -----
    **Pixel window correction**

    In the Gaussian path the smoothing kernel is computed from::

        FWHM_kernel^2 = FWHM_target^2 - FWHM_beam^2
                      - FWHM_pix_orig^2 - FWHM_pix_new^2

    In the Fourier path the exact sinc pixel window is used, exactly
    mirroring the HEALPix operation of multiplying alms by::

        T(ℓ) = [B_target(ℓ) × P_new(ℓ)] / [B_input(ℓ) × P_old(ℓ)]

    **Weighted smoothing (when rms_data is provided)**

    Gaussian path: ``data_smooth = convolve(w·data, K) / convolve(w, K)``.
    Fourier path: NaN pixels are inpainted before the FFT; ``w`` is used
    unchanged for the weighted ``reproject_exact`` step.
    """
    # ------------------------------------------------------------------
    # Resolve beam specifications (backward compatibility)
    # ------------------------------------------------------------------
    if input_beam is None and original_res_arcmin is not None:
        input_beam = float(original_res_arcmin)
    if target_beam is None and target_res_arcmin is not None:
        target_beam = float(target_res_arcmin)
    if input_beam is None:
        raise ValueError(
            "Provide input_beam (float FWHM in arcmin, (ell,B_ell) tuple, "
            "or callable).  The deprecated original_res_arcmin is also accepted."
        )
    if target_beam is None:
        raise ValueError(
            "Provide target_beam (float FWHM in arcmin).  "
            "The deprecated target_res_arcmin is also accepted."
        )

    # Build evaluators; detect whether we can use the fast Gaussian path
    _input_is_gaussian = isinstance(input_beam, (int, float))
    input_beam_eval    = _make_beam_evaluator(input_beam)
    target_beam_eval   = _make_beam_evaluator(target_beam)   # always Gaussian

    # Effective FWHMs for padding and Gaussian-path kernel computation
    _input_fwhm  = float(input_beam) if _input_is_gaussian else _beam_effective_fwhm(input_beam_eval)
    _target_fwhm = float(target_beam)   # target is always a Gaussian float

    if target_wcs is None:
        raise ValueError("target_wcs must be provided.")
    if _target_fwhm <= 0:
        raise ValueError("target_beam must be a positive FWHM [arcmin].")
    if pixel_size_arcmin <= 0:
        raise ValueError("pixel_size_arcmin must be positive.")
    if cutout_size_deg <= 0:
        raise ValueError("cutout_size_deg must be positive.")

    # Output grid shape derived from output pixel scale and cutout size.
    # Can be overridden by passing target_shape explicitly (e.g. from make_target_wcs).
    if target_shape is None:
        pixel_size_deg = pixel_size_arcmin / 60.0
        n_pix = int(np.ceil(cutout_size_deg / pixel_size_deg))
        if n_pix % 2 == 1:
            n_pix += 1
        target_shape = (n_pix, n_pix)

    # ------------------------------------------------------------------
    # 1.  Read / prepare the input map on a flat-sky intermediate grid
    # ------------------------------------------------------------------
    # Padding around the cutout to avoid edge effects from convolution.
    # 5× the target FWHM is generous (kernel is ~4σ ≈ 1.7 FWHM).
    # For non-Gaussian beams we use the estimated effective FWHM.
    _pad_fwhm   = max(_target_fwhm, _input_fwhm if _input_fwhm > 0 else _target_fwhm)
    padding_deg = 5.0 * _pad_fwhm / 60.0
    padded_size_deg = cutout_size_deg + 2.0 * padding_deg

    if map_format == "healpix":
        data_proc, current_wcs, current_pixel_arcmin = _read_healpix(
            data, healpix_coord,
            center_l, center_b, padded_size_deg, pixel_size_arcmin,
        )
        if rms_data is not None:
            rms_proc, _, _ = _read_healpix(
                rms_data, healpix_coord,
                center_l, center_b, padded_size_deg, pixel_size_arcmin,
            )
        else:
            rms_proc = None

    elif map_format == "fits":
        if original_wcs is None:
            raise ValueError(
                "original_wcs must be provided when map_format='fits'."
            )
        data_proc, current_wcs, current_pixel_arcmin = _read_fits(
            data, original_wcs,
            center_l, center_b, padded_size_deg,
        )
        if rms_data is not None:
            rms_proc, _, _ = _read_fits(
                rms_data, original_wcs,
                center_l, center_b, padded_size_deg,
            )
        else:
            rms_proc = None

    else:
        raise ValueError(f"Unknown map_format: '{map_format}'")

    # ------------------------------------------------------------------
    # 2.  Sanitise – replace sentinels / inf with NaN
    # ------------------------------------------------------------------
    data_proc = _sanitise(data_proc, remove_healpix_unseen=(map_format == "healpix"))

    # Compute inverse-variance weights from the RMS map (if provided).
    # w = 0 where RMS is NaN or zero so those pixels never contribute.
    if rms_proc is not None:
        rms_proc = _sanitise(rms_proc, remove_healpix_unseen=(map_format == "healpix"))
        w = np.where(np.isfinite(rms_proc) & (rms_proc > 0),
                     1.0 / rms_proc**2, 0.0)
    else:
        w = None

    # ------------------------------------------------------------------
    # 3.  Smooth to the target resolution
    # ------------------------------------------------------------------
    if _input_is_gaussian:
        # ---- Fast Gaussian path (backward compatible, exact NaN handling) ----
        sigma_pix, _ = _compute_smoothing_kernel_sigma_pix(
            _input_fwhm,
            _target_fwhm,
            orig_pixel_size_arcmin=current_pixel_arcmin,
            new_pixel_size_arcmin=pixel_size_arcmin,
            current_pixel_scale_arcmin=current_pixel_arcmin,
        )
        if sigma_pix is not None and sigma_pix > 0:
            kernel = Gaussian2DKernel(x_stddev=sigma_pix)
            if w is not None:
                # Weighted Gaussian smoothing: convolve(w*data,K) / convolve(w,K)
                valid  = np.isfinite(data_proc) & (w > 0)
                w_data = np.where(valid, w * data_proc, 0.0)
                w_only = np.where(valid, w, 0.0)
                numer = convolve(w_data, kernel, boundary="fill", fill_value=0.0,
                                 nan_treatment="fill")
                denom = convolve(w_only, kernel, boundary="fill", fill_value=0.0,
                                 nan_treatment="fill")
                data_proc = np.where(denom > 0, numer / denom, np.nan)
                w = denom   # smoothed weight map — used for weighted reprojection
            else:
                data_proc = convolve(
                    data_proc, kernel,
                    boundary="fill", fill_value=np.nan,
                    nan_treatment="interpolate", preserve_nan=True,
                )
        elif w is not None:
            w = np.where(np.isfinite(data_proc) & (w > 0), w, 0.0)
    else:
        # ---- Fourier-space beam transfer (non-Gaussian input_beam) ----
        # Applies T(k) = B_target × H_in / (H_in² + ε²) in 2D Fourier space.
        # reproject_exact (step 4) adds the output pixel window P_new,
        # mirroring the HEALPix formula: T = (B_new × P_new)/(B_old × P_old).
        data_proc = _apply_beam_transfer_2d(
            data_proc, current_pixel_arcmin,
            input_beam_eval, target_beam_eval,
            regularization=beam_regularization,
        )
        if w is not None:
            # Zero out weight where the transfer produced NaN
            w = np.where(np.isfinite(data_proc) & (w > 0), w, 0.0)

    # ------------------------------------------------------------------
    # 4.  Reproject to the target WCS  (exact / flux-conserving)
    # ------------------------------------------------------------------
    footprint = None  # will be set below in both branches
    if w is not None:
        # Weighted reprojection: reproject numerator (w*data) and denominator
        # (w) separately, then divide — equivalent to inverse-variance co-adding.
        w_data = np.where(np.isfinite(data_proc), w * data_proc, 0.0)
        hdu_wd = fits.PrimaryHDU(data=w_data,    header=current_wcs.to_header())
        hdu_w  = fits.PrimaryHDU(data=w,         header=current_wcs.to_header())

        repr_wd, fp_wd = reproject_exact(hdu_wd, target_wcs, shape_out=target_shape)
        repr_w,  fp_w  = reproject_exact(hdu_w,  target_wcs, shape_out=target_shape)

        footprint   = fp_w
        reprojected = np.where((fp_w > 0) & (repr_w > 0),
                               repr_wd / repr_w, np.nan)
    else:
        input_hdu = fits.PrimaryHDU(data=data_proc, header=current_wcs.to_header())
        reprojected, footprint = reproject_exact(
            input_hdu, target_wcs, shape_out=target_shape,
        )
        reprojected = np.where(footprint > 0, reprojected, np.nan)

    # ------------------------------------------------------------------
    # 5.  Mask invalid / zero-footprint pixels with NaN
    # ------------------------------------------------------------------
    reprojected = np.where(footprint > 0, reprojected, np.nan)
    reprojected[~np.isfinite(reprojected)] = np.nan

    return reprojected, target_wcs


# ======================================================================
#  Private helpers for reading different input formats
# ======================================================================

def _read_healpix(
    input_map, coord,
    center_l, center_b, padded_size_deg, pixel_size_arcmin,
):
    """
    Take a 1-D HEALPix pixel array and project it onto an intermediate
    flat-sky grid (gnomonic / TAN projection in Galactic coordinates).

    The intermediate pixel scale is set to half the smaller of the original
    HEALPix pixel size and the requested output pixel size, so that the
    HEALPix resolution is well-sampled.

    Parameters
    ----------
    input_map : 1-D numpy.ndarray
        HEALPix pixel values.
    coord : str
        Coordinate system of the HEALPix map ('G', 'C', or 'E').

    Returns
    -------
    data : 2-D ndarray
    wcs  : WCS
    pixel_scale_arcmin : float
    """
    hpx_data = np.asarray(input_map, dtype=np.float64).ravel()

    nside = hp.npix2nside(len(hpx_data))
    hpx_pixel_arcmin = np.degrees(hp.nside2resol(nside)) * 60.0

    # Intermediate pixel scale: well-sample the original HEALPix grid
    intermediate_pix = min(hpx_pixel_arcmin, pixel_size_arcmin) / 2.0

    # --- build intermediate WCS ---
    inter_wcs, inter_shape = _create_intermediate_wcs(
        center_l, center_b, padded_size_deg, intermediate_pix,
    )

    # --- reproject from HEALPix using nearest-neighbour sampling ---
    # (smoothing is done afterwards on the flat grid)
    hpx_input = (hpx_data, coord)
    data, footprint = reproject_from_healpix(
        hpx_input, inter_wcs, shape_out=inter_shape,
        order="nearest-neighbor", nested=False,
    )
    data[footprint == 0] = np.nan

    return data, inter_wcs, intermediate_pix


def _read_fits(data, wcs, center_l, center_b, padded_size_deg):
    """
    Extract a padded cutout from a FITS map (provided as array + WCS).

    The original pixel scale is read directly from the provided WCS.

    Parameters
    ----------
    data : 2-D (or higher) ndarray
        Pixel data of the original FITS map.
    wcs : astropy.wcs.WCS
        WCS of the original FITS map (used to determine pixel scale and
        to convert sky coordinates to pixel positions).

    Returns
    -------
    data : 2-D ndarray
    wcs  : WCS
    pixel_scale_arcmin : float
    """
    raw_data = np.asarray(data, dtype=np.float64)
    raw_wcs = wcs

    # Collapse extra dimensions (e.g. freq / Stokes)
    while raw_data.ndim > 2:
        raw_data = raw_data[0]

    # Pixel scale — read from WCS, no user input required
    pixel_scales_deg = proj_plane_pixel_scales(raw_wcs)  # deg per pixel
    pixel_scale_arcmin = np.mean(pixel_scales_deg) * 60.0

    # --- cut out a padded region around the requested centre ---
    center = SkyCoord(l=center_l * u.deg, b=center_b * u.deg, frame="galactic")

    # Convert Galactic → whatever frame the map WCS uses
    try:
        pix_x, pix_y = raw_wcs.world_to_pixel(center)
    except Exception:
        center_icrs = center.icrs
        pix_x, pix_y = raw_wcs.world_to_pixel(center_icrs)

    size_pix = int(np.ceil(padded_size_deg / np.mean(pixel_scales_deg)))
    position = (float(pix_x), float(pix_y))

    try:
        from astropy.nddata.utils import PartialOverlapError
        cutout = Cutout2D(
            raw_data, position=position, size=size_pix,
            wcs=raw_wcs, mode="partial", fill_value=np.nan,
        )
        data_out = cutout.data
        wcs_out = cutout.wcs
    except PartialOverlapError:
        # Centre is outside the map — return NaN array with original WCS
        warnings.warn(
            "Cutout region falls outside the input FITS map; "
            "returning NaN-filled array.",
            stacklevel=3,
        )
        data_out = np.full((size_pix, size_pix), np.nan)
        wcs_out = raw_wcs

    return data_out, wcs_out, pixel_scale_arcmin


# ======================================================================
#  Private helper: smooth + reproject a single flat-sky noise realization
# ======================================================================

def _smooth_and_reproject_flat(
    noise_flat,
    flat_wcs,
    pixel_arcmin,
    target_beam_eval,
    target_wcs,
    target_shape,
):
    """
    Apply **forward** beam smoothing and exact reprojection to one MC realisation.

    ``noise_flat`` is raw white noise (not yet smoothed by any beam).  This
    function applies **only** the target beam ``B_target(ℓ)`` in 2D Fourier
    space (forward-only, no deconvolution of the input pipeline), then
    reprojects with ``reproject_exact`` which adds the output pixel window
    ``P_new`` implicitly.

    The underlying amplitude ``sigma_WN = rms_flat / sqrt(mean(|H_in|^2))``
    already encodes the full input pipeline (beam + pixel window).  Applying
    only ``B_target`` here gives::

        sigma_out = sigma_WN * sqrt(mean(|B_target|^2)) * p0/p1 correction
                  = rms_flat * sqrt(mean(|B_target|^2) / mean(|H_in|^2))

    which matches the exact analytical formula for any beam type.

    .. note::

        **Do not** use ``_apply_beam_transfer_2d`` here.  That function applies
        a Wiener deconvolution + reconvolution (correct for DATA smoothing) but
        would overestimate noise by dividing by ``H_in < 1``.

    Parameters
    ----------
    noise_flat : 2-D ndarray
        Raw WN realisation; amplitude = ``sigma_WN`` per pixel.  Must be
        fully finite (no NaN).
    flat_wcs : astropy.wcs.WCS
        WCS of ``noise_flat``.
    pixel_arcmin : float
        Pixel size of ``noise_flat`` [arcmin].
    target_beam_eval : callable
        ``B_target(ell) -> ndarray`` — desired output beam.
    target_wcs : astropy.wcs.WCS
        Output WCS.
    target_shape : tuple of int
        Output shape ``(ny, nx)``.

    Returns
    -------
    out : 2-D ndarray, shape ``target_shape``
        Beam-smoothed and reprojected noise realisation.
    """
    ny, nx = noise_flat.shape

    # 2D frequency grid (cycles / arcmin)
    fy = np.fft.fftfreq(ny, d=pixel_arcmin)
    fx = np.fft.fftfreq(nx, d=pixel_arcmin)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    # Multipole ℓ (flat-sky)
    arcmin_per_rad = 180.0 * 60.0 / np.pi
    ELL = 2.0 * np.pi * np.sqrt(FX**2 + FY**2) * arcmin_per_rad

    # Apply ONLY B_target (forward smoothing — no deconvolution of H_in)
    B_tgt = np.clip(np.asarray(target_beam_eval(ELL), dtype=float), 0.0, None)

    D_fft    = np.fft.fft2(noise_flat)   # no NaN in WN realisation
    smoothed = np.real(np.fft.ifft2(D_fft * B_tgt))

    input_hdu = fits.PrimaryHDU(data=smoothed, header=flat_wcs.to_header())
    reprojected, footprint = reproject_exact(
        input_hdu, target_wcs, shape_out=target_shape,
    )
    reprojected = np.where(footprint > 0, reprojected, np.nan)
    return reprojected



# ======================================================================
#  Public function: RMS propagation
# ======================================================================

def propagate_rms_cutout(
    rms_data,
    map_format,
    rms_is_constant,
    original_res_arcmin=None,
    target_res_arcmin=None,
    input_beam=None,
    target_beam=None,
    pixel_size_arcmin=1.0,
    target_wcs=None,
    target_shape=None,
    center_l=0.0,
    center_b=0.0,
    cutout_size_deg=1.0,
    original_wcs=None,
    n_mc=None,
    random_seed=None,
    healpix_coord="G",
    beam_regularization=1e-4,
):
    """
    Propagate an RMS map through the smooth-and-reproject pipeline.

    Three regimes are supported:

    **Constant RMS** (``rms_is_constant=True``)
        Applies the exact analytical formula for white noise through a
        Gaussian beam and flux-conserving reprojection::

            σ_out = σ_in × (p₀/p₁) × (θ_eff_orig / θ_eff_target)

        where ``θ_eff = sqrt(θ_beam² + θ_pix²)`` is the effective resolution
        including the pixel window function.  ``rms_data`` must be a scalar
        float.  Returns a constant 2-D map of shape ``target_shape``.

    **Spatially varying FITS RMS** (``rms_is_constant=False``,
    ``map_format='fits'``)
        Uses Monte Carlo (``n_mc`` realisations) to propagate the noise.
        A padded cutout of the RMS map is extracted first so that
        simulations run only on the relevant sky patch.  Each realisation
        is white noise drawn pixel-by-pixel from the RMS values, then
        processed through the same smooth-and-reproject pipeline as the
        data.  The output RMS is the standard deviation of the
        realisations.

    **Spatially varying HEALPix RMS** (``rms_is_constant=False``,
    ``map_format='healpix'``)
        The HEALPix RMS map is first reprojected onto a small flat-sky
        intermediate grid (same padded region used by ``smooth_cutout``).
        From that point on, the computation is identical to the FITS case
        above, keeping memory usage proportional to the cutout area rather
        than the full-sky HEALPix array.

    Parameters
    ----------
    rms_data : float or ndarray
        - ``float`` when ``rms_is_constant=True``.
        - 2-D ndarray (FITS) or 1-D ndarray (HEALPix) otherwise.
    map_format : {'fits', 'healpix'}
        Format of the input RMS map (ignored when ``rms_is_constant=True``).
    rms_is_constant : bool
        If ``True``, uses the exact analytical formula (Gaussian beams only).
        ``rms_data`` must be a scalar.
    original_res_arcmin : float, optional
        **Deprecated alias** for ``input_beam``.
    target_res_arcmin : float, optional
        **Deprecated alias** for ``target_beam``.
    input_beam : float, tuple, or callable, optional
        Beam profile of the input map (same format as in ``smooth_cutout``).
    target_beam : float, optional
        Target beam FWHM [arcmin] (always Gaussian).
    pixel_size_arcmin : float
        Output pixel size [arcmin].
    target_wcs : astropy.wcs.WCS
        WCS of the output grid.
    target_shape : tuple of int
        Output shape ``(ny, nx)``.
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    cutout_size_deg : float
        Angular size of the final cutout [deg].
    original_wcs : astropy.wcs.WCS, optional
        WCS of the FITS input map.
    n_mc : int
        Number of Monte Carlo realisations.  **Required** when
        ``rms_is_constant=False``.
    random_seed : int or None
        Seed for the random number generator.
    healpix_coord : {'G', 'C', 'E'}
        Coordinate system of the HEALPix map.  Default ``'G'``.
    beam_regularization : float
        Wiener regularisation for the Fourier beam transfer.  Default ``1e-4``.

    Returns
    -------
    rms_out : ndarray, shape ``target_shape``
        Propagated RMS map on the output grid.  NaN outside the valid
        footprint.

    Examples
    --------
    Constant RMS — exact formula (Gaussian beam):

    >>> rms_out = propagate_rms_cutout(
    ...     rms_data=0.02, map_format='fits', rms_is_constant=True,
    ...     input_beam=5.0, target_beam=10.0,
    ...     pixel_size_arcmin=2.0, target_wcs=wcs, target_shape=(60, 60),
    ... )

    Spatially varying FITS RMS — Monte Carlo with custom beam:

    >>> rms_out = propagate_rms_cutout(
    ...     rms_data=rms_map, map_format='fits', rms_is_constant=False,
    ...     original_wcs=fits_wcs,
    ...     center_l=17.0, center_b=0.8, cutout_size_deg=2.0,
    ...     input_beam=(ell_arr, B_ell_arr), target_beam=10.0,
    ...     pixel_size_arcmin=2.0, target_wcs=wcs, target_shape=(60, 60),
    ...     n_mc=300, random_seed=42,
    ... )
    """
    # ------------------------------------------------------------------
    # Resolve beam specifications (backward compatibility)
    # ------------------------------------------------------------------
    if input_beam is None and original_res_arcmin is not None:
        input_beam = float(original_res_arcmin)
    if target_beam is None and target_res_arcmin is not None:
        target_beam = float(target_res_arcmin)
    if input_beam is None:
        raise ValueError("Provide input_beam (or the deprecated original_res_arcmin).")
    if target_beam is None:
        raise ValueError("Provide target_beam (or the deprecated target_res_arcmin).")

    _input_is_gaussian = isinstance(input_beam, (int, float))
    input_beam_eval    = _make_beam_evaluator(input_beam)
    target_beam_eval   = _make_beam_evaluator(target_beam)
    _input_fwhm        = float(input_beam) if _input_is_gaussian else _beam_effective_fwhm(input_beam_eval)
    _target_fwhm       = float(target_beam)

    # ------------------------------------------------------------------
    # CASE 1 — Constant RMS: exact analytical formula (Gaussian beams only)
    # ------------------------------------------------------------------
    if rms_is_constant:
        sigma_in = float(rms_data)

        if not _input_is_gaussian:
            warnings.warn(
                "propagate_rms_cutout: rms_is_constant=True uses the exact "
                "Gaussian analytical formula.  For a non-Gaussian input_beam "
                "the effective FWHM is used as an approximation.  Use "
                "rms_is_constant=False with n_mc for a rigorous Monte Carlo.",
                stacklevel=2,
            )

        # Derive the original pixel scale from the WCS when available.
        if map_format == "fits" and original_wcs is not None:
            pixel_scales_deg = proj_plane_pixel_scales(original_wcs)
            p0 = float(np.mean(pixel_scales_deg)) * 60.0   # arcmin
        elif map_format == "healpix":
            warnings.warn(
                "propagate_rms_cutout: for HEALPix + rms_is_constant=True, "
                "the original pixel scale cannot be inferred without data.  "
                "Use rms_is_constant=False with Monte Carlo instead.",
                stacklevel=2,
            )
            p0 = pixel_size_arcmin
        else:
            p0 = pixel_size_arcmin

        p1 = pixel_size_arcmin

        # σ_out = σ_in × (p₀/p₁) × (θ_eff_orig / θ_eff_target)
        pw_orig          = _pixel_window_fwhm(p0)
        pw_new           = _pixel_window_fwhm(p1)
        theta_eff_orig   = np.sqrt(_input_fwhm**2  + pw_orig**2)
        theta_eff_target = np.sqrt(_target_fwhm**2 + pw_new**2)

        sigma_out = sigma_in * (p0 / p1) * (theta_eff_orig / theta_eff_target)
        return np.full(target_shape, sigma_out, dtype=float)

    # ------------------------------------------------------------------
    # CASES 2 & 3 — Spatially varying RMS: Monte Carlo
    # ------------------------------------------------------------------
    if n_mc is None:
        raise ValueError(
            "n_mc must be provided when rms_is_constant=False."
        )

    rng = np.random.default_rng(random_seed)

    # Padding around the cutout to avoid edge effects (same as smooth_cutout)
    _pad_fwhm       = max(_target_fwhm, _input_fwhm if _input_fwhm > 0 else _target_fwhm)
    padding_deg     = 5.0 * _pad_fwhm / 60.0
    padded_size_deg = cutout_size_deg + 2.0 * padding_deg

    # ------------------------------------------------------------------
    # Step 1 — Read / reproject the RMS map to a flat-sky intermediate grid
    # ------------------------------------------------------------------
    if map_format == "healpix":
        # Reproject HEALPix RMS to a small flat-sky patch.
        # This is memory-efficient: only the patch pixels are used,
        # not the full-sky HEALPix array.
        rms_flat, flat_wcs, flat_pixel_arcmin = _read_healpix(
            rms_data, healpix_coord,
            center_l, center_b, padded_size_deg, pixel_size_arcmin,
        )
    elif map_format == "fits":
        if original_wcs is None:
            raise ValueError(
                "original_wcs must be provided when map_format='fits'."
            )
        rms_flat, flat_wcs, flat_pixel_arcmin = _read_fits(
            rms_data, original_wcs,
            center_l, center_b, padded_size_deg,
        )
    else:
        raise ValueError(f"Unknown map_format: '{map_format}'")

    # Sanitise the RMS flat patch (replace infs, HEALPix UNSEEN, etc.)
    rms_flat = _sanitise(rms_flat, remove_healpix_unseen=(map_format == "healpix"))

    # Zero or NaN RMS pixels are treated as masked — noise amplitude = 0
    rms_flat = np.where(np.isfinite(rms_flat) & (rms_flat > 0), rms_flat, 0.0)

    # ------------------------------------------------------------------
    # Step 2 — Compute sigma_WN from the input beam power on the 2D grid
    # ------------------------------------------------------------------
    # The input RMS map (rms_flat) is the noise of the beam-smoothed map.
    # The underlying white noise per pixel is:
    #
    #     sigma_WN = rms_flat / sqrt(mean(|H_in(k)|²))
    #
    # where H_in(k) = B_input(ℓ) × P_pixel_orig(k) is the combined transfer
    # of the input beam and the intermediate pixel window, evaluated on the
    # 2D Fourier grid of the flat patch.  This generalises the Gaussian case
    # and is equivalent to the user's suggestion of simulating WN and
    # applying T = (B_new/B_old) × (P_new/P_old) in Fourier space.
    ny_flat, nx_flat = rms_flat.shape
    fy_grid = np.fft.fftfreq(ny_flat, d=flat_pixel_arcmin)
    fx_grid = np.fft.fftfreq(nx_flat, d=flat_pixel_arcmin)
    FX_g, FY_g = np.meshgrid(fx_grid, fy_grid, indexing='xy')

    arcmin_per_rad = 180.0 * 60.0 / np.pi
    ELL_g  = 2.0 * np.pi * np.sqrt(FX_g**2 + FY_g**2) * arcmin_per_rad
    B_in_g = np.clip(np.asarray(input_beam_eval(ELL_g), dtype=float), 0.0, None)
    P_or_g = np.sinc(FX_g * flat_pixel_arcmin) * np.sinc(FY_g * flat_pixel_arcmin)
    H_in_g = B_in_g * P_or_g

    # Parseval: sum(K_b²) ≡ mean(|H_in|²)  in the discrete Fourier sense
    sum_Kb_sq = float(np.mean(H_in_g**2))
    if sum_Kb_sq <= 0:
        sum_Kb_sq = 1.0   # fallback: no beam → sigma_WN = rms_flat

    sigma_WN_flat = rms_flat / np.sqrt(sum_Kb_sq)

    # ------------------------------------------------------------------
    # Step 3 — Monte Carlo: generate WN realisations and propagate
    # ------------------------------------------------------------------
    out_stack = np.empty((n_mc, *target_shape), dtype=float)

    for k in range(n_mc):
        # Draw raw white noise at the native amplitude
        noise_k = rng.standard_normal((ny_flat, nx_flat)) * sigma_WN_flat

        # Apply ONLY the target beam B_tgt in Fourier space (forward smoothing).
        # reproject_exact adds the output pixel window P_new implicitly.
        # sigma_WN already encodes the full input pipeline, so no deconvolution
        # is needed here — that would overestimate noise (divides by H_in < 1).
        out_stack[k] = _smooth_and_reproject_flat(
            noise_k,
            flat_wcs,
            flat_pixel_arcmin,
            target_beam_eval,
            target_wcs,
            target_shape,
        )

    # ------------------------------------------------------------------
    # Step 4 — RMS = std of realisations (ddof=1 for unbiased estimate)
    # ------------------------------------------------------------------
    rms_out = np.nanstd(out_stack, axis=0, ddof=1)

    # Pixels where all realisations are NaN → keep NaN
    all_nan = np.all(~np.isfinite(out_stack), axis=0)
    rms_out[all_nan] = np.nan

    return rms_out

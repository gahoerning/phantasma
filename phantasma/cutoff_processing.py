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
    return (1.0 / _FWHM_TO_SIGMA) * pixel_size_arcmin / np.sqrt(12.0)


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
    original_res_arcmin=5.0,
    target_res_arcmin=10.0,
    pixel_size_arcmin=1.0,
    target_wcs=None,
    target_shape=None,
    healpix_coord="G",
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
        to the output.  The RMS map itself is **not** smoothed or propagated;
        only the original values are used as weights.
    original_wcs : astropy.wcs.WCS, optional
        WCS of the original FITS map.  Required when ``map_format='fits'``;
        ignored for ``'healpix'`` (Nside is inferred from the array length).
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    cutout_size_deg : float
        Angular size of the cutout [deg].
    original_res_arcmin : float
        Original *beam* FWHM of the input map [arcmin].  This is the
        Gaussian beam only — the pixel window is accounted for separately
        using the pixel scales.
    target_res_arcmin : float
        Desired *total effective* FWHM after smoothing and reprojection
        [arcmin].
    pixel_size_arcmin : float
        Pixel size of the *output* grid [arcmin].  Used both to compute the
        pixel window correction and to determine the output grid shape.
    target_wcs : astropy.wcs.WCS
        WCS of the output grid.
    target_shape : tuple of int, optional
        Shape ``(ny, nx)`` of the output grid.  Computed automatically from
        ``pixel_size_arcmin`` and ``cutout_size_deg`` if not provided.
    healpix_coord : str, optional
        Coordinate system of the HEALPix map: ``'G'`` (Galactic), ``'C'``
        (Celestial/Equatorial), ``'E'`` (Ecliptic).  Default ``'G'``.

    Returns
    -------
    cutout_data : numpy.ndarray
        2-D map reprojected onto ``target_wcs`` with NaN for invalid pixels.
    target_wcs : astropy.wcs.WCS
        The WCS of the returned map (same object as the input ``target_wcs``).

    Notes
    -----
    **Pixel window correction**

    The input data has already been smoothed by the original beam *and* the
    original pixel window.  ``reproject_exact`` will add the output pixel
    window.  The smoothing kernel subtracts all three contributions so that
    the final effective resolution equals ``target_res_arcmin``::

        FWHM_kernel^2 = FWHM_target^2 - FWHM_beam^2
                      - FWHM_pix_orig^2 - FWHM_pix_new^2

    where  FWHM_pix ≈ 0.68 × pixel_size  (Gaussian approximation of a
    square top-hat pixel window).

    **Weighted smoothing (when rms_data is provided)**

    The weighted convolution is computed as::

        w = 1 / rms²
        data_smooth = convolve(w * data, K) / convolve(w, K)

    For reprojection, numerator and denominator are reprojected separately::

        data_out = reproject(w_smooth * data_smooth) / reproject(w_smooth)

    where ``w_smooth = convolve(w, K)`` are the weights after smoothing.
    """
    if target_wcs is None:
        raise ValueError("target_wcs must be provided.")
    if original_res_arcmin <= 0:
        raise ValueError("original_res_arcmin must be positive.")
    if target_res_arcmin <= 0:
        raise ValueError("target_res_arcmin must be positive.")
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
    padding_deg = 5.0 * target_res_arcmin / 60.0
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
        rms_proc = _sanitise(rms_proc)
        w = np.where(np.isfinite(rms_proc) & (rms_proc > 0),
                     1.0 / rms_proc**2, 0.0)
    else:
        w = None

    # ------------------------------------------------------------------
    # 3.  Smooth to the target resolution
    # ------------------------------------------------------------------
    sigma_pix, fwhm_kernel = _compute_smoothing_kernel_sigma_pix(
        original_res_arcmin,
        target_res_arcmin,
        orig_pixel_size_arcmin=current_pixel_arcmin,
        new_pixel_size_arcmin=pixel_size_arcmin,
        current_pixel_scale_arcmin=current_pixel_arcmin,
    )

    if sigma_pix is not None and sigma_pix > 0:
        kernel = Gaussian2DKernel(x_stddev=sigma_pix)
        if w is not None:
            # Weighted Gaussian smoothing: convolve(w*data, K) / convolve(w, K)
            # Only count pixels that are both valid in data AND have positive weight.
            valid = np.isfinite(data_proc) & (w > 0)
            w_data = np.where(valid, w * data_proc, 0.0)
            w_only = np.where(valid, w, 0.0)
            numer = convolve(w_data, kernel, boundary="fill", fill_value=0.0,
                             nan_treatment="fill")
            denom = convolve(w_only, kernel, boundary="fill", fill_value=0.0,
                             nan_treatment="fill")
            data_proc = np.where(denom > 0, numer / denom, np.nan)
            w = denom  # smoothed weight map — used for weighted reprojection
        else:
            # Standard (unweighted) smoothing
            data_proc = convolve(
                data_proc, kernel,
                boundary="fill",
                fill_value=np.nan,
                nan_treatment="interpolate",
                preserve_nan=True,
            )
    elif w is not None:
        # No kernel needed, but mask pixels invalid in data
        w = np.where(np.isfinite(data_proc) & (w > 0), w, 0.0)

    # ------------------------------------------------------------------
    # 4.  Reproject to the target WCS  (exact / flux-conserving)
    # ------------------------------------------------------------------
    if w is not None:
        # Weighted reprojection: reproject numerator (w*data) and denominator
        # (w) separately, then divide — equivalent to inverse-variance co-adding.
        w_data = np.where(np.isfinite(data_proc), w * data_proc, 0.0)
        hdu_wd = fits.PrimaryHDU(data=w_data,    header=current_wcs.to_header())
        hdu_w  = fits.PrimaryHDU(data=w,         header=current_wcs.to_header())

        repr_wd, fp_wd = reproject_exact(hdu_wd, target_wcs, shape_out=target_shape)
        repr_w,  fp_w  = reproject_exact(hdu_w,  target_wcs, shape_out=target_shape)

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
    total_sigma_pix,
    target_wcs,
    target_shape,
):
    """
    Apply Gaussian smoothing and exact reprojection to one flat-sky 2-D array.

    The ``noise_flat`` array is treated as raw white noise (i.e. it has
    **not** yet been smoothed by any beam).  The function convolves with
    the **total** target Gaussian kernel (beam + extra smoothing combined in
    quadrature via ``total_sigma_pix``) and then reprojects to the output
    WCS.  This ensures the output amplitude correctly reflects the noise
    after the full pipeline.

    Parameters
    ----------
    noise_flat : 2-D ndarray
        Raw white-noise realisation on the intermediate flat-sky grid.
        Amplitude is the underlying WN std per pixel (``sigma_WN``).
    flat_wcs : astropy.wcs.WCS
        WCS of ``noise_flat``.
    total_sigma_pix : float or None
        Sigma [pixels] of the **total** Gaussian kernel to apply
        (combining original beam and additional smoothing).  Pass ``None``
        or ≤ 0 if no smoothing is needed.
    target_wcs : astropy.wcs.WCS
        Output WCS.
    target_shape : tuple of int
        Output grid shape ``(ny, nx)``.

    Returns
    -------
    out : 2-D ndarray, shape ``target_shape``
        Smoothed and reprojected noise realisation (NaN outside footprint).
    """
    proc = noise_flat.copy()

    if total_sigma_pix is not None and total_sigma_pix > 0:
        kernel = Gaussian2DKernel(x_stddev=total_sigma_pix)
        proc = convolve(
            proc, kernel,
            boundary="fill",
            fill_value=0.0,         # WN outside the patch contributes zero
            nan_treatment="fill",   # NaN → 0 to avoid contaminating interior
            preserve_nan=False,
        )

    input_hdu = fits.PrimaryHDU(data=proc, header=flat_wcs.to_header())
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
    original_res_arcmin,
    target_res_arcmin,
    pixel_size_arcmin,
    target_wcs,
    target_shape,
    center_l=0.0,
    center_b=0.0,
    cutout_size_deg=1.0,
    original_wcs=None,
    n_mc=None,
    random_seed=None,
    healpix_coord="G",
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
        If ``True``, uses the exact analytical formula.  ``rms_data`` must
        be a scalar.
    original_res_arcmin : float
        Beam FWHM of the input map [arcmin].
    target_res_arcmin : float
        Target beam FWHM after smoothing [arcmin].
    pixel_size_arcmin : float
        Output pixel size [arcmin].
    target_wcs : astropy.wcs.WCS
        WCS of the output grid (same object used in ``smooth_cutout``).
    target_shape : tuple of int
        Output shape ``(ny, nx)`` (same as used in ``smooth_cutout``).
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    cutout_size_deg : float
        Angular size of the *final* cutout [deg].  Used to determine the
        minimum padded area for the MC simulations.
    original_wcs : astropy.wcs.WCS, optional
        WCS of the FITS input map.  Required when ``map_format='fits'``
        and ``rms_is_constant=False``; ignored otherwise.
    n_mc : int
        Number of Monte Carlo realisations.  **Required** when
        ``rms_is_constant=False``.
    random_seed : int or None
        Seed for the random number generator (for reproducibility).
    healpix_coord : {'G', 'C', 'E'}
        Coordinate system of the HEALPix map.  Default ``'G'`` (Galactic).

    Returns
    -------
    rms_out : ndarray, shape ``target_shape``
        Propagated RMS map on the output grid.  NaN outside the valid
        footprint.

    Examples
    --------
    Constant RMS — exact formula:

    >>> rms_out = propagate_rms_cutout(
    ...     rms_data=0.02, map_format='fits', rms_is_constant=True,
    ...     original_res_arcmin=5.0, target_res_arcmin=10.0,
    ...     pixel_size_arcmin=2.0, target_wcs=wcs, target_shape=(60, 60),
    ... )

    Spatially varying FITS RMS — Monte Carlo with 300 realisations:

    >>> rms_out = propagate_rms_cutout(
    ...     rms_data=rms_map, map_format='fits', rms_is_constant=False,
    ...     original_wcs=fits_wcs,
    ...     center_l=17.0, center_b=0.8, cutout_size_deg=2.0,
    ...     original_res_arcmin=5.0, target_res_arcmin=10.0,
    ...     pixel_size_arcmin=2.0, target_wcs=wcs, target_shape=(60, 60),
    ...     n_mc=300, random_seed=42,
    ... )
    """
    # ------------------------------------------------------------------
    # CASE 1 — Constant RMS: exact analytical formula
    # ------------------------------------------------------------------
    if rms_is_constant:
        sigma_in = float(rms_data)

        # We need the original pixel scale.  For a constant-RMS map we
        # cannot read it from a WCS, so we derive it from the intermediate
        # grid that _read_healpix / _read_fits would produce.  As a robust
        # proxy, we use the output pixel size as the reference scale.
        # The formula only needs p₀; we obtain it from original_wcs (FITS)
        # or from a dummy HEALPix nside estimate.
        if map_format == "fits" and original_wcs is not None:
            pixel_scales_deg = proj_plane_pixel_scales(original_wcs)
            p0 = float(np.mean(pixel_scales_deg)) * 60.0        # arcmin
        elif map_format == "healpix":
            # Cannot determine p0 without data; warn and use pixel_size_arcmin
            warnings.warn(
                "propagate_rms_cutout: for HEALPix + rms_is_constant=True, "
                "the original pixel scale cannot be inferred without data.  "
                "Pass original_wcs=None and provide a FITS WCS, or use "
                "rms_is_constant=False with Monte Carlo instead.",
                stacklevel=2,
            )
            p0 = pixel_size_arcmin
        else:
            p0 = pixel_size_arcmin

        p1 = pixel_size_arcmin

        # Effective resolutions including pixel window functions
        pw_orig = _pixel_window_fwhm(p0)
        pw_new  = _pixel_window_fwhm(p1)
        theta_eff_orig   = np.sqrt(original_res_arcmin**2   + pw_orig**2)
        theta_eff_target = np.sqrt(target_res_arcmin**2     + pw_new**2)

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
    padding_deg = 5.0 * target_res_arcmin / 60.0
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
    # Step 2 — Compute kernels needed for the Monte Carlo
    # ------------------------------------------------------------------
    # The input RMS map (rms_flat) is the noise of a beam-smoothed map.
    # The underlying white-noise per pixel is:
    #     sigma_WN = rms_flat / sqrt(sum(K_b^2))
    # where K_b is the Gaussian kernel of the original beam.
    # Each MC realisation draws from WN with that amplitude, then convolves
    # with the TOTAL kernel K_total = Gaussian(sigma_b ⊕ sigma_extra) and
    # reprojects.  This is equivalent to simulating the full pipeline from
    # scratch on raw WN, which correctly propagates correlated noise.

    fwhm_to_sigma = _FWHM_TO_SIGMA  # 1 / (2*sqrt(2*ln2)) ≈ 0.4247

    # Original beam sigma [pixels on the flat intermediate grid]
    sigma_b_arcmin = original_res_arcmin * fwhm_to_sigma
    sigma_b_pix    = sigma_b_arcmin / flat_pixel_arcmin

    # Extra smoothing kernel sigma [pixels] — needed to reach target resolution
    sigma_k_pix, _ = _compute_smoothing_kernel_sigma_pix(
        original_res_arcmin,
        target_res_arcmin,
        orig_pixel_size_arcmin=flat_pixel_arcmin,
        new_pixel_size_arcmin=pixel_size_arcmin,
        current_pixel_scale_arcmin=flat_pixel_arcmin,
    )

    # Total kernel: combine beam and extra kernel in quadrature
    if sigma_k_pix is not None and sigma_k_pix > 0:
        sigma_total_pix = np.sqrt(sigma_b_pix**2 + sigma_k_pix**2)
    else:
        sigma_total_pix = sigma_b_pix  # no additional smoothing needed

    # Amplitude correction: convert rms_flat (noise of beam-smoothed map)
    # to sigma_WN (noise of the underlying raw WN field).
    # For a Gaussian kernel: sum(K^2) = 1 / (4*pi*sigma_pix^2)  [continuous limit]
    # Using the actual discrete kernel avoids small numerical errors.
    if sigma_b_pix > 0:
        K_b = Gaussian2DKernel(x_stddev=sigma_b_pix)
        sum_Kb_sq = np.sum(K_b.array ** 2)
    else:
        sum_Kb_sq = 1.0   # no beam → sigma_WN = rms_flat

    # sigma_WN per pixel (spatially varying field)
    sigma_WN_flat = rms_flat / np.sqrt(sum_Kb_sq)

    # ------------------------------------------------------------------
    # Step 3 — Monte Carlo: generate WN realisations and propagate
    # ------------------------------------------------------------------
    ny_flat, nx_flat = rms_flat.shape
    out_stack = np.empty((n_mc, *target_shape), dtype=float)

    for k in range(n_mc):
        # Draw raw white noise scaled by sigma_WN per pixel
        noise_k = rng.standard_normal((ny_flat, nx_flat)) * sigma_WN_flat

        # Apply total kernel (beam + extra) and reproject to target WCS
        out_stack[k] = _smooth_and_reproject_flat(
            noise_k,
            flat_wcs,
            sigma_total_pix,
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

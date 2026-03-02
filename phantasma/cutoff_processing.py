"""
cutoff_processing.py
Utilities to preprocess astronomical maps: smooth to a target resolution
accounting for pixel window functions, reproject to a target WCS, and
extract a cutout.

Supports both FITS (flat-sky) and HEALPix input maps.
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


def _sanitise(data):
    """Replace non-finite values (inf, UNSEEN, etc.) with NaN."""
    out = np.array(data, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    # HEALPix UNSEEN sentinel
    out[np.isclose(out, hp.UNSEEN, atol=1e-6)] = np.nan
    return out


def preprocess_and_cutout(
    input_map,
    map_format="fits",
    center_l=0.0,
    center_b=0.0,
    cutout_size_deg=1.0,
    original_res_arcmin=5.0,
    target_res_arcmin=10.0,
    pixel_size_arcmin=1.0,
    target_wcs=None,
    target_shape=None,
    healpix_coord="G",
    fits_hdu=0,
):
    """
    Read an astronomical map, smooth it to a target resolution (accounting for
    pixel window functions), and reproject it onto a target WCS grid.

    The function handles both flat-sky FITS images and HEALPix maps.
    Reprojection is done with ``reproject_exact`` (flux-conserving).

    Parameters
    ----------
    input_map : str or numpy.ndarray or tuple
        - For ``map_format='fits'``: a file path (str) or a ``(data, wcs)``
          tuple.
        - For ``map_format='healpix'``: a 1-D numpy array of HEALPix pixel
          values (the caller is responsible for reading the file).
    map_format : {'fits', 'healpix'}
        Format of the input map.
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    cutout_size_deg : float
        Angular size of the cutout [deg].
    original_res_arcmin : float
        Original *beam* FWHM of the input map [arcmin].  This is the
        Gaussian beam only — the pixel window is accounted for separately
        using the pixel sizes.
    target_res_arcmin : float
        Desired *total effective* FWHM after smoothing and reprojection
        [arcmin].
    pixel_size_arcmin : float
        Pixel size of the *output* grid [arcmin].  Used to compute the pixel
        window correction.
    target_wcs : astropy.wcs.WCS
        WCS of the output grid.  Together with ``target_shape`` this fully
        defines the output pixelisation.
    target_shape : tuple of int
        Shape ``(ny, nx)`` of the output grid.
    healpix_coord : str, optional
        Coordinate system of the HEALPix map: ``'G'`` (Galactic), ``'C'``
        (Celestial/Equatorial), ``'E'`` (Ecliptic).  Default ``'G'``.
    fits_hdu : int, optional
        HDU index to read from a FITS file (default 0).

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
    """
    if target_wcs is None:
        raise ValueError("target_wcs must be provided.")

    if target_shape is None:
        pixel_size_deg = pixel_size_arcmin / 60.0
        n_pix = int(np.ceil(cutout_size_deg / pixel_size_deg))
        if n_pix % 2 == 1:
            n_pix += 1  # keep even for symmetry
        target_shape = (n_pix, n_pix)

    # ------------------------------------------------------------------
    # 1.  Read / prepare the input map on a flat-sky intermediate grid
    # ------------------------------------------------------------------
    # Padding around the cutout to avoid edge effects from convolution.
    # 5× the target FWHM is generous (kernel is ~4σ ≈ 1.7 FWHM).
    padding_deg = 5.0 * target_res_arcmin / 60.0
    padded_size_deg = cutout_size_deg + 2.0 * padding_deg

    if map_format == "healpix":
        data, current_wcs, current_pixel_arcmin = _read_healpix(
            input_map, healpix_coord,
            center_l, center_b, padded_size_deg, pixel_size_arcmin,
        )

    elif map_format == "fits":
        data, current_wcs, current_pixel_arcmin = _read_fits(
            input_map, fits_hdu,
            center_l, center_b, padded_size_deg,
        )

    else:
        raise ValueError(f"Unknown map_format: '{map_format}'")

    # ------------------------------------------------------------------
    # 2.  Sanitise – replace sentinels / inf with NaN
    # ------------------------------------------------------------------
    data = _sanitise(data)

    # ------------------------------------------------------------------
    # 3.  Smooth to the target resolution
    # ------------------------------------------------------------------
    sigma_pix, fwhm_kernel = _compute_smoothing_kernel_sigma_pix(
        original_res_arcmin,
        target_res_arcmin,
        orig_pixel_size_arcmin=current_pixel_arcmin,   # original pixel window
        new_pixel_size_arcmin=pixel_size_arcmin,        # new pixel window
        current_pixel_scale_arcmin=current_pixel_arcmin, # for arcmin → px conversion
    )

    if sigma_pix is not None and sigma_pix > 0:
        # Use astropy convolve to properly handle NaN pixels.
        kernel = Gaussian2DKernel(x_stddev=sigma_pix)
        data = convolve(
            data, kernel,
            boundary="fill",
            fill_value=np.nan,
            nan_treatment="interpolate",
            preserve_nan=True,
        )

    # ------------------------------------------------------------------
    # 4.  Reproject to the target WCS  (exact / flux-conserving)
    # ------------------------------------------------------------------
    input_hdu = fits.PrimaryHDU(data=data, header=current_wcs.to_header())

    reprojected, footprint = reproject_exact(
        input_hdu, target_wcs, shape_out=target_shape,
    )

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


def _read_fits(input_map, hdu_idx, center_l, center_b, padded_size_deg):
    """
    Read a FITS map and extract a padded cutout around (l, b).

    Returns
    -------
    data : 2-D ndarray
    wcs  : WCS
    pixel_scale_arcmin : float
    """
    if isinstance(input_map, str):
        hdu_list = fits.open(input_map)
        raw_data = hdu_list[hdu_idx].data.astype(np.float64)
        raw_wcs = WCS(hdu_list[hdu_idx].header, naxis=2)
        hdu_list.close()
    elif isinstance(input_map, tuple) and len(input_map) == 2:
        raw_data, raw_wcs = input_map
        raw_data = np.asarray(raw_data, dtype=np.float64)
    else:
        raise ValueError(
            "For 'fits' format, provide a file path (str) or (data, wcs) tuple."
        )

    # Collapse extra dimensions (e.g. freq / Stokes)
    while raw_data.ndim > 2:
        raw_data = raw_data[0]

    # Pixel scale
    pixel_scales_deg = proj_plane_pixel_scales(raw_wcs)  # deg per pixel
    pixel_scale_arcmin = np.mean(pixel_scales_deg) * 60.0

    # --- cut out a padded region around the requested centre ---
    center = SkyCoord(l=center_l * u.deg, b=center_b * u.deg, frame="galactic")

    # Cutout2D needs the position in pixel coords of the *current* WCS.
    # Convert Galactic → whatever the map WCS frame is.
    try:
        pix_x, pix_y = raw_wcs.world_to_pixel(center)
    except Exception:
        # If the WCS frame doesn't directly accept Galactic, convert to ICRS
        center_icrs = center.icrs
        pix_x, pix_y = raw_wcs.world_to_pixel(center_icrs)

    size_pix = int(np.ceil(padded_size_deg / np.mean(pixel_scales_deg)))
    position = (float(pix_x), float(pix_y))

    try:
        cutout = Cutout2D(
            raw_data, position=position, size=size_pix,
            wcs=raw_wcs, mode="partial", fill_value=np.nan,
        )
        data = cutout.data
        wcs_out = cutout.wcs
    except Exception:
        # If cutout fails (e.g. completely outside), return NaN array
        warnings.warn(
            "Cutout region falls outside the input FITS map; "
            "returning NaN-filled array.",
            stacklevel=3,
        )
        data = np.full((size_pix, size_pix), np.nan)
        wcs_out = raw_wcs  # fallback

    return data, wcs_out, pixel_scale_arcmin

"""
PHANTaSMA — Standard astronomy map preprocessing and analysis utilities.

Modules
-------
cutoff_processing
    Map reprojection, smoothing, and cutout utilities.
template_fitting
    Weighted linear template fitting with bootstrap uncertainty estimation.

Usage
-----
    import phantasma as ph

    # Map preprocessing
    result = ph.smooth_cutout(...)

    # Template fitting
    from phantasma.template_fitting import template_fit_bootstrap
    result = template_fit_bootstrap(data_map, template_maps, data_rms=rms)
    result.summary()
"""

from .cutoff_processing import smooth_cutout, make_target_wcs, propagate_rms_cutout
from .template_fitting import (
    TemplateFitResult,
    make_geometric_templates,
    template_fit,
    template_fit_bootstrap,
    simulate_template_fit,
)

__all__ = [
    # cutoff_processing
    "smooth_cutout",
    "make_target_wcs",
    "propagate_rms_cutout",
    # template_fitting
    "TemplateFitResult",
    "make_geometric_templates",
    "template_fit",
    "template_fit_bootstrap",
    "simulate_template_fit",
]

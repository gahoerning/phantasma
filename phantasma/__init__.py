"""
PHANTaSMA — Standard astronomy map preprocessing utilities.

Usage
-----
    import phantasma as ph
    result, wcs = ph.preprocess_and_cutout(...)

    from phantasma import cutoff_processing
    from phantasma.cutoff_processing import preprocess_and_cutout
"""

from .cutoff_processing import preprocess_and_cutout

__all__ = ["preprocess_and_cutout"]

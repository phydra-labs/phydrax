#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import abc


class SpectralDiscretizationProvider(abc.ABC):
    """Validated spectral analysis/synthesis plan consumable by spatial solvers."""


__all__ = ["SpectralDiscretizationProvider"]

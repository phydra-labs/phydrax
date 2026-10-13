#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Global polynomial collocation discretizations."""

from ._chebyshev import (
    chebyshev_lobatto_matrices,
    ChebyshevCollocation,
    ChebyshevDifferentiationAction,
)


__all__ = [
    "ChebyshevCollocation",
    "ChebyshevDifferentiationAction",
    "chebyshev_lobatto_matrices",
]

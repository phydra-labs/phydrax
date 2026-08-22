#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Global polynomial collocation discretizations."""

from ._chebyshev import chebyshev_lobatto_matrices, ChebyshevCollocation


__all__ = ["ChebyshevCollocation", "chebyshev_lobatto_matrices"]

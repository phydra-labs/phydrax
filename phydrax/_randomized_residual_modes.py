#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Loss-mode vocabulary shared by randomized residual terms and PDE compilation.

A dependency-free leaf: the randomized residual terms and the PDE compiler both
validate against this alias, and the compiler cannot import the terms package at
module load without an import cycle.
"""

from typing import Literal, TypeAlias


RandomizedResidualLossMode: TypeAlias = Literal[
    "u_statistic",
    "independent_product",
    "plug_in",
]


__all__ = ["RandomizedResidualLossMode"]

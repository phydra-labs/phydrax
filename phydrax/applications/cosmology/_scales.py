#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ..._physical import DimensionalScaleContract


CosmologyScaleContract = DimensionalScaleContract

CODE_COSMOLOGY_SCALE = CosmologyScaleContract(
    "code_length",
    "code_mass",
    "code_time",
    length_coordinate_kind="comoving",
)


__all__ = ["CODE_COSMOLOGY_SCALE", "CosmologyScaleContract"]

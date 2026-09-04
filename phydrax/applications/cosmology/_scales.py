#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ..._physical import DimensionalScaleContract
from ...units import LENGTH, MASS, TIME, UnitDefinition


CosmologyScaleContract = DimensionalScaleContract

_COSMOLOGY_CODE_REFERENCE_SYSTEM_ID = "phydrax:cosmology-code"
_CODE_LENGTH = UnitDefinition("code_length", LENGTH, _COSMOLOGY_CODE_REFERENCE_SYSTEM_ID)
_CODE_MASS = UnitDefinition("code_mass", MASS, _COSMOLOGY_CODE_REFERENCE_SYSTEM_ID)
_CODE_TIME = UnitDefinition("code_time", TIME, _COSMOLOGY_CODE_REFERENCE_SYSTEM_ID)

CODE_COSMOLOGY_SCALE = CosmologyScaleContract(
    _CODE_LENGTH,
    _CODE_MASS,
    _CODE_TIME,
    length_coordinate_kind="comoving",
)


__all__ = ["CODE_COSMOLOGY_SCALE", "CosmologyScaleContract"]

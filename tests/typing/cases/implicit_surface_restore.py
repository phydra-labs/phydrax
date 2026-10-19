#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import assert_type

from phydrax.geometry import CompiledGeometry, ImplicitSurfacePlan


def surface_restore_contract(
    value: ImplicitSurfacePlan, geometry: CompiledGeometry
) -> None:
    assert_type(
        ImplicitSurfacePlan.from_stored_fields(value, geometry=geometry),
        ImplicitSurfacePlan,
    )

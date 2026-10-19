import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import (
    brep_box,
    BRepTessellationPolicy,
    FixedTopologyBRepSource,
    PlanePatch,
)
from phydrax.units import METER, MILLIMETER, UnitDefinition


@pytest.mark.parametrize(
    ("unit", "scale"),
    [(METER, 1.0), (MILLIMETER, 1000.0)],
    ids=["meters", "millimeters"],
)
def test_compatible_realized_seams_accept_in_both_units(
    unit: UnitDefinition, scale: float
) -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (scale, 2.0 * scale, 3.0 * scale),
        coordinate_contract=SpatialCoordinateContract(unit),
        tessellation=BRepTessellationPolicy(linear_deflection=0.05 * scale),
    )
    geometry = FixedTopologyBRepSource(model).compile(coordinate_unit=unit)
    points = jnp.asarray([[0.5 * scale, scale, 1.5 * scale]], dtype=jnp.float64)
    np.testing.assert_array_equal(
        geometry.kernel.contains(geometry.state, points), [True]
    )


@pytest.mark.parametrize(
    ("unit", "scale"),
    [(METER, 1.0), (MILLIMETER, 1000.0)],
    ids=["meters", "millimeters"],
)
def test_incompatible_realized_seams_refuse_in_both_units(
    unit: UnitDefinition, scale: float
) -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (scale, 2.0 * scale, 3.0 * scale),
        coordinate_contract=SpatialCoordinateContract(unit),
        tessellation=BRepTessellationPolicy(linear_deflection=0.05 * scale),
    )
    geometry = FixedTopologyBRepSource(model).compile(coordinate_unit=unit)
    origin_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "origin"
    )
    # One nanometer is below the former absolute meter threshold, but its
    # equivalent millimeter displacement exceeds it. Both break the same seam.
    patch = model.patches[0]
    assert isinstance(patch, PlanePatch)
    origin = jnp.asarray(patch.origin, dtype=jnp.float64)
    state = geometry.state.replace_at(
        origin_index, origin + jnp.asarray([1.0e-9 * scale, 0.0, 0.0], dtype=jnp.float64)
    )
    points = jnp.asarray([[0.5 * scale, scale, 1.5 * scale]], dtype=jnp.float64)
    with pytest.raises(RuntimeError, match="seam_residual.*bound"):
        geometry.kernel.contains(state, points).block_until_ready()

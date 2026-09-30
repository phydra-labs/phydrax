from typing import assert_type

from jax import Array

from phydrax.discretization import BoundaryTraceSpaceCapability
from phydrax.discretization.bem import (
    BuffaChristiansenDualSpace3D,
    prepare_boundary_form_query,
    PreparedBoundaryFormQuery,
    RWGSurfaceCurrentSpace3D,
)
from phydrax.exterior import FormType, FormValueSpec


def surface_values(
    rwg: RWGSurfaceCurrentSpace3D,
    dual: BuffaChristiansenDualSpace3D,
    capability: BoundaryTraceSpaceCapability,
    points: Array,
    coefficients: Array,
) -> None:
    assert_type(rwg.form_type, FormType)
    assert_type(dual.value_spec, FormValueSpec)
    assert_type(capability.form_type, FormType)
    query = prepare_boundary_form_query(rwg, points)
    assert_type(query, PreparedBoundaryFormQuery)
    assert_type(query.apply(coefficients), Array)
    assert_type(query.transpose(points), Array)

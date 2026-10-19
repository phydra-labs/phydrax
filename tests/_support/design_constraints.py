#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from phydrax.geometry import CompiledGeometry, DesignConstraintSystem, Sphere
from phydrax.geometry.design import AbstractDesignConstraint, ParameterId, ParameterTarget
from phydrax.units import METER


def sphere_constraint_problem(
    *, constraint: AbstractDesignConstraint | None = None
) -> tuple[
    CompiledGeometry,
    DesignConstraintSystem,
    dict[str, ParameterId],
    dict[ParameterId, tuple[float, float]],
]:
    """Build a sphere residual system with a finite center/radius search box."""
    geometry = Sphere((0.0, 0.0, 0.0), 1.0).compile(coordinate_unit=METER)
    parameter_ids = {
        parameter_id.name: parameter_id for parameter_id in geometry.schema.parameter_ids
    }
    if constraint is None:
        constraint = ParameterTarget(parameter_ids["radius"], 1.5)
    system = DesignConstraintSystem(geometry, (constraint,))
    bounds = {
        parameter_ids["center"]: (-0.5, 0.5),
        parameter_ids["radius"]: (0.25, 2.5),
    }
    return geometry, system, parameter_ids, bounds

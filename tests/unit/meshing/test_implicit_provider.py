import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(count=9):
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))


def test_native_implicit_provider_preserves_fixed_topology_gradients():
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="sphere",
    ).compile()
    source_id = "sphere"
    source_revision = "r1"
    scope = phx.meshing.MeshingScope(
        source_id,
        source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2,
            3,
            phx.meshing.CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.3,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
    )
    plan = phx.meshing.NativeImplicitProvider().plan(
        geometry,
        _grid(),
        specification,
        source_id=source_id,
        source_revision=source_revision,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
    )
    result = plan.execute()
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))

    def vertex_sum(radius):
        state = geometry.state.replace_at(radius_index, radius)
        return jnp.sum(plan.surface_plan.realize(state).proposed_vertices)

    derivative = jax.grad(vertex_sum)(jnp.asarray(0.75))

    assert result.audit.passed
    assert result.associations[0].complete
    assert "wall_seconds" in result.runtime.enforced_limits
    assert (
        result.derivative_mode is phx.meshing.MeshingDerivativeMode.FIXED_ROUTE_PIECEWISE
    )
    assert jnp.isfinite(derivative)
    assert derivative != 0.0


def _implicit_case(*, size_strength, limits=None, periodic_constraints=()):
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="admission-sphere",
    ).compile()
    scope = phx.meshing.MeshingScope(
        "admission-sphere",
        "r1",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2,
            3,
            phx.meshing.CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.3,
                strength=size_strength,
            ),
        ),
        periodic_constraints=periodic_constraints,
        limits=limits,
    )
    return geometry, scope, specification


def _plan_case(geometry, specification):
    return phx.meshing.NativeImplicitProvider().plan(
        geometry,
        _grid(),
        specification,
        source_id="admission-sphere",
        source_revision="r1",
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
    )


def test_native_implicit_provider_rejects_unsupported_periodic_controls():
    geometry, scope, _ = _implicit_case(
        size_strength=phx.meshing.SizeControlStrength.SOFT
    )
    periodic = phx.meshing.PeriodicConstraint(scope, scope, np.eye(4))
    _, _, specification = _implicit_case(
        size_strength=phx.meshing.SizeControlStrength.SOFT,
        periodic_constraints=(periodic,),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as error:
        _plan_case(geometry, specification)
    assert (
        error.value.category is phx.meshing.MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    )


def test_native_implicit_provider_enforces_limits_and_hard_size_compliance():
    limits = phx.meshing.MeshingLimits(maximum_vertices=3)
    geometry, _, limited = _implicit_case(
        size_strength=phx.meshing.SizeControlStrength.SOFT,
        limits=limits,
    )
    with pytest.raises(phx.meshing.MeshingFailure) as resource_error:
        _plan_case(geometry, limited)
    assert (
        resource_error.value.category
        is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    )

    geometry, _, hard = _implicit_case(size_strength=phx.meshing.SizeControlStrength.HARD)
    plan = _plan_case(geometry, hard)
    with pytest.raises(phx.meshing.MeshingFailure) as compliance_error:
        plan.execute()
    assert (
        compliance_error.value.category
        is phx.meshing.MeshingFailureCategory.COMPLIANCE_FAILED
    )


def test_native_implicit_provider_enforces_wall_deadline_in_worker():
    limits = phx.meshing.MeshingLimits(maximum_wall_seconds=1.0e-6)
    geometry, _, specification = _implicit_case(
        size_strength=phx.meshing.SizeControlStrength.SOFT,
        limits=limits,
    )
    plan = _plan_case(geometry, specification)

    with pytest.raises(phx.meshing.MeshingFailure) as error:
        plan.execute()
    assert error.value.category is phx.meshing.MeshingFailureCategory.TIMED_OUT

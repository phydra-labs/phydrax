import jax
import jax.numpy as jnp
import numpy as np

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
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.3),),
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
    assert (
        result.derivative_mode is phx.meshing.MeshingDerivativeMode.FIXED_ROUTE_PIECEWISE
    )
    assert jnp.isfinite(derivative)
    assert derivative != 0.0

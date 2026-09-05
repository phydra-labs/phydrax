import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._distribution import MeshDistribution


def _cell_part(name="cells", scale=1.0, single=False):
    points = scale * np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32)
    if single:
        points = points[[0, 1, 3]]
        cells = np.asarray(((0, 1, 2),), dtype=np.int32)
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return MeshPart(
        name, phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )


def _fe(part):
    return phx.discretization.fem.FiniteElementPlan(
        part.carrier.mesh,
        phx.discretization.fem.FiniteElementFieldSpec(
            "u", phx.discretization.fem.discontinuous_element("triangle", 1)
        ),
        coordinate_spec=part.carrier.geometry,
    ).prepare()


def _grid_part(name="grid", *, periodic=True, scale=1.0):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [scale]]))
    return MeshPart(name, grid, coordinate_contract=phx.SpatialCoordinateContract.si())


def _compiled_fv(part):
    discretization = phx.discretization.FiniteVolumePlan(part.carrier).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="meshing-distribution-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "meshing-distribution-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    return phx.equations.compile_conservation_problem(problem, discretization, method)


def test_global_id_ownership_normalization_lowers_to_conservative_fe_execution():
    part = _cell_part()
    native_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in part.carrier.mesh.blocks]
    )
    distribution = MeshDistribution(
        part,
        phx.discretization.CellPartition(np.asarray([1, 0]), 2),
        cell_global_ids=native_ids[::-1],
    )
    field = jnp.asarray([3.0, 7.0])
    np.testing.assert_allclose(distribution.gather(0, field), [3.0, 7.0])
    np.testing.assert_allclose(distribution.gather(1, field), [7.0, 3.0])
    phases = distribution.lower_finite_element(part, _fe(part))
    np.testing.assert_allclose(
        sum(phases.local_contribution(rank, field) for rank in range(2)), 10.0
    )
    flux = jnp.asarray([2.0])
    routed = sum(phases.facet_ownership.route_partition(rank, flux) for rank in range(2))
    np.testing.assert_allclose(routed, [2.0, -2.0])
    with pytest.raises(ValueError, match="stale"):
        distribution.lower_finite_element(_cell_part(scale=2.0), _fe(part))
    with pytest.raises(ValueError, match="exact distribution mesh"):
        distribution.lower_finite_element(part, _fe(_cell_part(scale=2.0)))


def test_fe_distribution_handles_no_interior_interfaces_without_fake_facets():
    part = _cell_part(single=True)
    distribution = MeshDistribution(
        part, phx.discretization.CellPartition(np.asarray([0]), 1)
    )
    phases = distribution.lower_finite_element(part, _fe(part))
    np.testing.assert_allclose(phases.local_contribution(0, jnp.asarray([4.0])), 4.0)
    np.testing.assert_allclose(
        phases.facet_ownership.route_equal_opposite(jnp.empty((0,))), [0.0]
    )


def test_distribution_rejects_fractional_ownership_missing_halos_and_wrong_partition_geometry():
    part = _cell_part()
    with pytest.raises(TypeError, match="integer vector"):
        phx.discretization.CellPartition(np.asarray([0.2, 1.0]), 2)
    with pytest.raises(ValueError, match="every partition|Every partition"):
        phx.discretization.CellPartition(np.asarray([0, 0]), 2)
    with pytest.raises(ValueError, match="adjacency reach"):
        MeshDistribution(
            part,
            phx.discretization.CellPartition(np.asarray([0, 1]), 2),
            halo_global_ids=([], []),
        )
    with pytest.raises(ValueError, match="locally owned"):
        MeshDistribution(
            part,
            phx.discretization.CellPartition(np.asarray([0, 1]), 2),
            halo_global_ids=([0, 1], [0]),
        )
    grid = _grid_part()
    with pytest.raises(ValueError, match="reproduce"):
        MeshDistribution(
            grid,
            phx.discretization.CellPartition(np.asarray([0, 1] * 4), 2),
            split_factors=(2,),
        )


def test_tensor_halos_respect_periodic_topology_and_lower_to_real_fv_residual():
    part = _grid_part()
    distribution = MeshDistribution.cartesian(part, (2,))
    np.testing.assert_array_equal(distribution.halo_global_ids[0], [4, 7])
    np.testing.assert_array_equal(distribution.halo_global_ids[1], [0, 3])
    bounded = _grid_part(periodic=False)
    bounded_distribution = MeshDistribution.cartesian(bounded, (2,))
    np.testing.assert_array_equal(bounded_distribution.halo_global_ids[0], [4])
    bounded_plan = bounded_distribution.lower_finite_volume(
        bounded, phx.discretization.FiniteVolumePlan(bounded.carrier).prepare()
    )
    assert not bounded_plan.periodic[0]
    compiled = _compiled_fv(part)
    serial_distribution = MeshDistribution.cartesian(part, (1,))
    plan = serial_distribution.lower_finite_volume(part, compiled.discretization)
    runtime = plan.prepare((jax.devices()[0],))
    values = jnp.sin(2 * jnp.pi * part.carrier.structured_axes[0].interval_centers)[
        :, None
    ]
    np.testing.assert_allclose(
        runtime.residual(compiled.dynamics, 0.0, runtime.shard_state(values)),
        compiled(0.0, values),
        rtol=1e-12,
        atol=1e-12,
    )
    changed = _grid_part(scale=2.0)
    with pytest.raises(ValueError, match="stale"):
        serial_distribution.lower_finite_volume(changed, compiled.discretization)
    with pytest.raises(ValueError, match="grid revision"):
        runtime.compile_residual(_compiled_fv(changed).dynamics, 0.0)

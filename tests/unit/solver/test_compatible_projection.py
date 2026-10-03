#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import CochainDiscretization
from phydrax.discretization.meshfree._exterior import MeshfreeExteriorCalculusPlan
from phydrax.discretization.meshfree._exterior_metric import MeshfreeMetricPolicy
from phydrax.solver import CompatibleProjectionStatus


def _structured_bridge(points: int) -> phx.discretization.StructuredCochainBridge:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(points) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    return phx.discretization.StructuredCochainBridge(grid)


def _structured_cochain(points: int) -> CochainDiscretization:
    return _structured_bridge(points).cochain


@pytest.fixture(scope="module")
def meshfree_cochain() -> CochainDiscretization:
    # Perturbed 4x4 cloud whose exact nonnegative edge metric is admitted as a
    # native positive one-complex (graph of 16 vertices and 26 edges).
    rng = np.random.default_rng(3)
    lattice = np.stack(
        np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    points = lattice + 0.08 * rng.uniform(-1.0, 1.0, lattice.shape)
    boundary = np.any((lattice == 0.0) | (lattice == 3.0), axis=1)
    count = points.shape[0]
    prepared = MeshfreeExteriorCalculusPlan(
        points,
        1.5,
        count * (count - 1) // 2,
        node_volumes=np.ones(count),
        dirichlet=boundary,
        metric_policy=MeshfreeMetricPolicy("nonnegative", tolerance=1e-7),
    ).prepare()
    assert bool(prepared.metric_result.accepted)
    return prepared.to_cochain()


def _dense_reference(
    cochain: CochainDiscretization, velocity: Array, edge_coefficients: Array
) -> np.ndarray:
    """Independent dense oracle: minimum-norm least squares of δ(w d p) = δ u."""
    size = cochain.cell_counts[0]

    def action(pressure: Array) -> Array:
        return cochain.codifferential(
            1, edge_coefficients * cochain.exterior_derivative(0, pressure)
        )

    matrix = np.asarray(jax.vmap(action)(jnp.eye(size, dtype=jnp.float64)).T)
    divergence = np.asarray(cochain.codifferential(1, velocity))
    pressure = np.linalg.lstsq(matrix, divergence, rcond=1e-12)[0]
    gradient = np.asarray(cochain.exterior_derivative(0, jnp.asarray(pressure)))
    return np.asarray(velocity) - np.asarray(edge_coefficients) * gradient


def _arithmetic_edge_inverse_density(
    cochain: CochainDiscretization, density: Array
) -> Array:
    relation = cochain.topology.incidences[0].relation
    sums = np.zeros((relation.target_size,))
    counts = np.zeros((relation.target_size,))
    valid = np.asarray(relation.valid)
    source = np.asarray(relation.source_indices)[valid]
    target = np.asarray(relation.target_indices)[valid]
    np.add.at(sums, target, np.asarray(density)[source])
    np.add.at(counts, target, 1.0)
    return jnp.asarray(counts / sums)


def _assert_accepted(result: phx.solver.IncompressibleProjectionResult) -> None:
    assert int(result.status) == CompatibleProjectionStatus.SUCCESS
    assert bool(result.successful)
    assert bool(result.linear.successful)
    assert bool(result.kernel_valid)
    assert float(result.divergence_defect_norm) <= float(result.acceptance_threshold)
    assert float(result.pressure_residual_norm) <= float(result.acceptance_threshold)
    assert float(result.compatibility_residual) <= float(result.acceptance_threshold)
    assert int(result.linear.diagnostics.iterations) > 0
    assert result.precision == "float64"
    np.testing.assert_array_equal(result.velocity, result.candidate_velocity)


def test_structured_projection_matches_dense_pseudoinverse_route() -> None:
    bridge = _structured_bridge(4)
    cochain = bridge.cochain
    velocity = jnp.sin(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64) / 3.0)
    projection = phx.solver.CompatibleIncompressibleProjection(bridge)

    result = eqx.filter_jit(projection.project)(velocity)

    _assert_accepted(result)
    assert result.nullity == 1
    reference = _dense_reference(
        cochain, velocity, jnp.ones((cochain.cell_counts[1],), dtype=jnp.float64)
    )
    np.testing.assert_allclose(result.velocity, reference, rtol=0.0, atol=1e-9)
    assert float(jnp.linalg.norm(result.divergence_before)) > 1e-2
    np.testing.assert_allclose(result.divergence_after, 0.0, atol=1e-9)
    # Minimum-norm gauge: the pressure is Hodge-orthogonal to the constant kernel.
    volumes = cochain.hodge_diagonal(0)
    np.testing.assert_allclose(jnp.sum(volumes * result.pressure), 0.0, atol=1e-12)


def test_structured_variable_density_matches_dense_route() -> None:
    cochain = _structured_cochain(4)
    velocity = jnp.sin(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64))
    density = 1.0 + 0.4 * jnp.cos(jnp.arange(cochain.cell_counts[0], dtype=jnp.float64))
    projection = phx.solver.CompatibleVariableDensityProjection(cochain)

    result = projection.project(velocity, density)

    _assert_accepted(result)
    coefficients = _arithmetic_edge_inverse_density(cochain, density)
    np.testing.assert_allclose(result.edge_coefficients, coefficients, rtol=1e-14)
    np.testing.assert_allclose(
        result.velocity,
        _dense_reference(cochain, velocity, coefficients),
        rtol=0.0,
        atol=1e-9,
    )


def test_meshfree_positive_one_complex_projection_is_solenoidal(
    meshfree_cochain: CochainDiscretization,
) -> None:
    cochain = meshfree_cochain
    assert cochain.dimension == 1
    velocity = jnp.asarray(
        np.random.default_rng(11).standard_normal(cochain.cell_counts[1])
    )
    projection = phx.solver.CompatibleIncompressibleProjection(cochain)

    result = projection.project(velocity)

    _assert_accepted(result)
    assert result.nullity == 1
    assert float(jnp.linalg.norm(result.divergence_before)) > 1e-1
    np.testing.assert_allclose(
        result.velocity,
        _dense_reference(cochain, velocity, jnp.ones_like(velocity)),
        rtol=0.0,
        atol=1e-9,
    )
    # The removed part is exactly the gradient of the reported pressure.
    np.testing.assert_allclose(
        velocity - result.velocity,
        cochain.exterior_derivative(0, result.pressure),
        atol=1e-12,
    )


def test_meshfree_variable_density_projection_under_jit(
    meshfree_cochain: CochainDiscretization,
) -> None:
    cochain = meshfree_cochain
    rng = np.random.default_rng(5)
    velocity = jnp.asarray(rng.standard_normal(cochain.cell_counts[1]))
    density = jnp.asarray(rng.uniform(0.5, 3.0, cochain.cell_counts[0]))
    projection = phx.solver.CompatibleVariableDensityProjection(cochain)

    result = eqx.filter_jit(projection.project)(velocity, density)

    _assert_accepted(result)
    coefficients = _arithmetic_edge_inverse_density(cochain, density)
    np.testing.assert_allclose(
        result.velocity,
        _dense_reference(cochain, velocity, coefficients),
        rtol=0.0,
        atol=1e-9,
    )


def test_compatible_target_divergence_is_reached() -> None:
    cochain = _structured_cochain(4)
    velocity = jnp.cos(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64))
    volumes = cochain.hodge_diagonal(0)
    source = jnp.sin(jnp.arange(cochain.cell_counts[0], dtype=jnp.float64))
    source = source - jnp.sum(volumes * source) / jnp.sum(volumes)
    projection = phx.solver.CompatibleIncompressibleProjection(cochain)

    result = projection.project(velocity, target_divergence=source)

    _assert_accepted(result)
    np.testing.assert_allclose(result.divergence_after, source, atol=1e-9)


def test_incompatible_target_divergence_is_refused_and_rolled_back() -> None:
    cochain = _structured_cochain(4)
    velocity = jnp.cos(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64))
    projection = phx.solver.CompatibleIncompressibleProjection(cochain)

    # A net source on a closed (natural-boundary) complex violates ∫ δu = 0.
    result = projection.project(
        velocity, target_divergence=jnp.ones((cochain.cell_counts[0],))
    )

    assert int(result.status) == CompatibleProjectionStatus.INCOMPATIBLE_SOURCE
    assert not bool(result.successful)
    assert float(result.compatibility_residual) > float(result.acceptance_threshold)
    np.testing.assert_array_equal(result.velocity, velocity)
    np.testing.assert_array_equal(result.divergence_after, result.divergence_before)
    assert bool(jnp.all(jnp.isnan(result.pressure)))
    assert bool(jnp.all(jnp.isfinite(result.candidate_pressure)))


def test_iteration_limited_pressure_solve_reports_failure_without_values(
    meshfree_cochain: CochainDiscretization,
) -> None:
    cochain = meshfree_cochain
    velocity = jnp.asarray(
        np.random.default_rng(13).standard_normal(cochain.cell_counts[1])
    )
    projection = phx.solver.CompatibleIncompressibleProjection(
        cochain,
        solve_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.ProjectedPCG(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-12, absolute=1e-14, max_steps=1
            ),
            require_device_binding=True,
        ),
    )

    result = projection.project(velocity)

    assert int(result.status) == CompatibleProjectionStatus.SOLVE_FAILED
    assert int(result.linear.status) == int(
        phx.linalg.LinearSolveStatus.MAXIMUM_STEPS_REACHED
    )
    assert float(result.divergence_defect_norm) > float(result.acceptance_threshold)
    np.testing.assert_array_equal(result.velocity, velocity)
    assert bool(jnp.all(jnp.isnan(result.pressure)))


def test_variable_density_refuses_nonpositive_active_density() -> None:
    cochain = _structured_cochain(3)
    projection = phx.solver.CompatibleVariableDensityProjection(cochain)
    density = jnp.ones((cochain.cell_counts[0],)).at[2].set(0.0)

    with pytest.raises(eqx.EquinoxRuntimeError, match="finite and positive"):
        result = projection.project(jnp.ones((cochain.cell_counts[1],)), density)
        jax.block_until_ready(result.velocity)


def test_non_projected_krylov_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="ProjectedPCG"):
        phx.solver.CompatibleIncompressibleProjection(
            _structured_cochain(3),
            solve_policy=phx.linalg.LinearSolvePolicy(phx.linalg.MINRES()),
        )


@pytest.fixture(scope="module")
def preconditioned_bridge() -> tuple[
    phx.discretization.StructuredCochainBridge,
    phx.solver.CompatibleIncompressibleProjection,
]:
    # 17x17 vertices: large enough that unpreconditioned CG needs tens of steps.
    bridge = _structured_bridge(16)
    return bridge, phx.solver.CompatibleIncompressibleProjection(
        bridge, preconditioner="smoothed-aggregation"
    )


def _structured_iterations(
    bridge: phx.discretization.StructuredCochainBridge,
    projection: phx.solver.CompatibleIncompressibleProjection,
) -> int:
    count = bridge.cochain.cell_counts[1]
    result = projection.project(jnp.sin(jnp.arange(count, dtype=jnp.float64) / 3.0))
    _assert_accepted(result)
    return int(result.linear.diagnostics.iterations)


def test_smoothed_aggregation_projection_matches_unpreconditioned_route(
    preconditioned_bridge: tuple[
        phx.discretization.StructuredCochainBridge,
        phx.solver.CompatibleIncompressibleProjection,
    ],
) -> None:
    bridge, preconditioned = preconditioned_bridge
    cochain = bridge.cochain
    velocity = jnp.sin(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64) / 3.0)

    reference = phx.solver.CompatibleIncompressibleProjection(bridge).project(velocity)
    result = eqx.filter_jit(preconditioned.project)(velocity)

    _assert_accepted(reference)
    _assert_accepted(result)
    assert reference.preconditioner == "none"
    assert reference.multigrid is None
    assert result.preconditioner == "smoothed-aggregation"
    assert result.multigrid is not None
    levels = result.multigrid.level_dimensions
    assert levels[0] == cochain.cell_counts[0]
    assert len(levels) >= 2 and levels[-1] < levels[0]
    assert int(result.linear.diagnostics.iterations) < int(
        reference.linear.diagnostics.iterations
    )
    np.testing.assert_allclose(result.velocity, reference.velocity, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(result.pressure, reference.pressure, rtol=0.0, atol=1e-9)


def test_smoothed_aggregation_iterations_grow_sublinearly(
    preconditioned_bridge: tuple[
        phx.discretization.StructuredCochainBridge,
        phx.solver.CompatibleIncompressibleProjection,
    ],
) -> None:
    small = _structured_bridge(8)
    small_iterations = _structured_iterations(
        small,
        phx.solver.CompatibleIncompressibleProjection(
            small, preconditioner="smoothed-aggregation"
        ),
    )
    large_iterations = _structured_iterations(*preconditioned_bridge)

    # 81 -> 289 unknowns (3.6x); a mesh-independent preconditioner keeps the
    # count far below the sqrt(N) ~ 1.9x growth of unpreconditioned CG.
    assert large_iterations < 2 * small_iterations


def test_meshfree_variable_density_frozen_hierarchy_accepts_two_densities(
    meshfree_cochain: CochainDiscretization,
) -> None:
    cochain = meshfree_cochain
    rng = np.random.default_rng(17)
    velocity = jnp.asarray(rng.standard_normal(cochain.cell_counts[1]))
    mild = jnp.asarray(rng.uniform(0.5, 3.0, cochain.cell_counts[0]))
    contrast = jnp.asarray(rng.uniform(1.0, 100.0, cochain.cell_counts[0]))
    projection = phx.solver.CompatibleVariableDensityProjection(
        cochain, preconditioner="smoothed-aggregation"
    )
    project = eqx.filter_jit(projection.project)

    first = project(velocity, mild)
    second = project(velocity, contrast)

    _assert_accepted(first)
    _assert_accepted(second)
    assert first.multigrid is not None
    assert first.multigrid == second.multigrid
    np.testing.assert_allclose(
        first.velocity,
        _dense_reference(
            cochain, velocity, _arithmetic_edge_inverse_density(cochain, mild)
        ),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        second.velocity,
        _dense_reference(
            cochain, velocity, _arithmetic_edge_inverse_density(cochain, contrast)
        ),
        rtol=0.0,
        atol=1e-9,
    )


def test_smoothed_aggregation_refuses_non_diagonal_vertex_hodge() -> None:
    base = _structured_cochain(3)
    count = base.cell_counts[0]
    indices = np.arange(count)
    cochain = CochainDiscretization(
        base.topology,
        (
            phx.discretization.SparseHodge(
                indices, indices, np.asarray(base.hodge_diagonal(0)), count
            ),
            *base.hodges[1:],
        ),
        coordinates=base.coordinates,
        boundary_masks=base.boundary_masks,
        primal_measures=base.primal_measures,
        dual_measures=base.dual_measures,
    )

    with pytest.raises(ValueError, match="diagonal degree-zero and degree-one Hodges"):
        phx.solver.CompatibleIncompressibleProjection(
            cochain, preconditioner="smoothed-aggregation"
        )


def test_smoothed_aggregation_refuses_policy_preconditioning() -> None:
    with pytest.raises(ValueError, match="owns the pressure preconditioning"):
        phx.solver.CompatibleIncompressibleProjection(
            _structured_cochain(3),
            solve_policy=phx.linalg.LinearSolvePolicy(
                phx.linalg.ProjectedPCG(),
                preconditioning=phx.linalg.PreconditioningPolicy(
                    phx.linalg.JacobiPreconditionerBuilder()
                ),
            ),
            preconditioner="smoothed-aggregation",
        )

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Consumer-visible reproduction, admission, spatial and Hilbert contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.meshfree._stencils import chart_stencil_kernel
from phydrax.discretization.meshfree._types import StencilWeightKernel


def _points() -> Array:
    return jnp.asarray([(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])


def _polynomial(points: Array) -> Array:
    x, y = points[:, 0], points[:, 1]
    return x * x + 2 * x * y + 3 * y * y + 4 * x - 2 * y + 1


@pytest.mark.parametrize(
    "policy",
    [
        phx.discretization.LocalStencilPolicy(weight_kernel="inverse-square"),
        phx.discretization.LocalStencilPolicy(weight_kernel="wendland-c2"),
        phx.discretization.LocalStencilPolicy(approximation="phs-rbf-fd"),
    ],
    ids=["gmls-inverse-square", "gmls-wendland-c2", "phs-rbf-fd"],
)
def test_cross_target_polynomial_and_normal_mixed_reproduction(
    policy: phx.discretization.LocalStencilPolicy,
) -> None:
    d = phx.discretization
    points = _points()
    targets = jnp.asarray(((0.2, -0.3), (-0.4, 0.6)))
    neighborhood = d.MeshfreeNeighborhoodPlan(points, 9, targets=targets).prepare()
    normal = np.asarray(((0.6, 0.8), (-0.8, 0.6)))
    functionals = (
        d.MeshfreeFunctional(((0, 0),), (1.0,), name="value"),
        d.MeshfreeFunctional(
            ((1, 0), (0, 1)), (1.0, 1.0), row_coefficients=normal, name="normal"
        ),
        d.MeshfreeFunctional(((1, 1),), (1.0,), name="mixed"),
        d.MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0), name="laplacian"),
    )
    stencils = d.prepare_local_stencils(
        neighborhood,
        points,
        targets,
        functionals,
        policy,
    )
    values = _polynomial(points)
    gx = 2 * targets[:, 0] + 2 * targets[:, 1] + 4
    gy = 2 * targets[:, 0] + 6 * targets[:, 1] - 2
    expected = (
        _polynomial(targets),
        normal[:, 0] * gx + normal[:, 1] * gy,
        jnp.full((2,), 2.0),
        jnp.full((2,), 8.0),
    )
    for index, reference in enumerate(expected):
        operator = d.MeshfreeOperator(stencils, index)
        np.testing.assert_allclose(
            eqx.filter_jit(operator.apply)(values), reference, atol=2e-10
        )
    vector = jnp.stack((values, 2 * values), axis=-1)
    np.testing.assert_allclose(
        d.MeshfreeOperator(stencils).apply(vector),
        jnp.stack((expected[0], 2 * expected[0]), axis=-1),
        atol=2e-10,
    )


@pytest.mark.parametrize("kernel", ["inverse-square", "wendland-c2"])
def test_overdetermined_gmls_matches_weighted_noisy_data_fit(
    kernel: StencilWeightKernel,
) -> None:
    d = phx.discretization
    points = _points()
    targets = jnp.asarray(((0.2, -0.3), (-0.4, 0.6)))
    neighborhood = d.MeshfreeNeighborhoodPlan(points, 9, targets=targets).prepare()
    stencils = d.prepare_local_stencils(
        neighborhood,
        points,
        targets,
        (d.MeshfreeFunctional(((0, 0),), (1.0,)),),
        d.LocalStencilPolicy(polynomial_degree=1, weight_kernel=kernel),
    )
    values = np.asarray((0.1, 2.0, -0.7, 0.2, 1.4, -1.8, 0.9, 3.1, -0.3))
    expected = []
    for row, target in enumerate(np.asarray(targets)):
        selected = np.asarray(neighborhood.relation.source_indices[row])
        offsets = np.asarray(points)[selected] - target
        radius = np.linalg.norm(offsets, axis=1)
        scale = np.max(radius)
        ratio = radius / scale
        weight = (
            (1.0 - ratio / 1.1) ** 4 * (4.0 * ratio / 1.1 + 1.0)
            if kernel == "wendland-c2"
            else 1.0 / np.maximum(ratio, 0.25) ** 2
        )
        design = np.concatenate((np.ones((9, 1)), offsets / scale), axis=1)
        fitted = np.linalg.lstsq(
            np.sqrt(weight)[:, None] * design,
            np.sqrt(weight) * values[selected],
            rcond=None,
        )[0]
        expected.append(fitted[0])
    np.testing.assert_allclose(
        d.MeshfreeOperator(stencils).apply(jnp.asarray(values)), expected, atol=1e-12
    )


def test_rank_condition_and_amplification_refusal_are_distinct() -> None:
    d = phx.discretization
    # A genuinely full-rank, nearly-collinear cloud violates a low condition
    # threshold, whereas a collinear cloud fails rank even with a loose limit.
    nearly = jnp.asarray(((-1.0, -1.0), (0.0, 1e-4), (1.0, 1.0), (0.3, 0.3 - 2e-4)))
    value = (d.MeshfreeFunctional(((0, 0),), (1.0,)),)
    neighborhood = d.MeshfreeNeighborhoodPlan(
        nearly, 4, targets=jnp.zeros((1, 2))
    ).prepare()
    policy = d.LocalStencilPolicy(
        polynomial_degree=1, condition_limit=2.0, acceptance="mask"
    )
    masked = d.prepare_local_stencils(
        neighborhood, nearly, jnp.zeros((1, 2)), value, policy
    )
    assert int(masked.evidence.rank[0]) == 3
    assert int(masked.evidence.status[0]) == int(d.MeshfreeRowStatus.ILL_CONDITIONED)
    assert np.isfinite(float(masked.evidence.condition[0]))
    np.testing.assert_array_equal(masked.weights[0], 0.0)
    with pytest.raises(ValueError, match="ILL_CONDITIONED"):
        d.prepare_local_stencils(
            neighborhood,
            nearly,
            jnp.zeros((1, 2)),
            value,
            d.LocalStencilPolicy(polynomial_degree=1, condition_limit=2.0),
        )
    line = nearly.at[:, 1].set(0.0)
    relation = d.MeshfreeNeighborhoodPlan(line, 4, targets=jnp.zeros((1, 2))).prepare()
    rank_failed = d.prepare_local_stencils(
        relation,
        line,
        jnp.zeros((1, 2)),
        value,
        d.LocalStencilPolicy(polynomial_degree=1, acceptance="mask"),
    )
    assert int(rank_failed.evidence.status[0]) == int(d.MeshfreeRowStatus.RANK_DEFICIENT)
    points = _points()
    relation = d.MeshfreeNeighborhoodPlan(points, 9, targets=jnp.zeros((1, 2))).prepare()
    with pytest.raises(ValueError, match="EXCESSIVE_AMPLIFICATION"):
        d.prepare_local_stencils(
            relation,
            points,
            jnp.zeros((1, 2)),
            value,
            d.LocalStencilPolicy(amplification_limit=0.1),
        )


def test_device_fixed_support_kernel_has_same_reproduction_and_masked_rank_failure() -> (
    None
):
    d = phx.discretization
    points = _points()
    offsets = jnp.stack((points, points.at[:, 1].set(0.0)))
    coefficients = jnp.ones((2, 1, 1))
    weights, evidence = eqx.filter_jit(chart_stencil_kernel)(
        offsets,
        jnp.ones((2, 9), dtype=bool),
        ((1, 1),),
        coefficients,
        d.LocalStencilPolicy(acceptance="mask"),
    )
    np.testing.assert_allclose(
        jnp.sum(weights[0, 0] * _polynomial(points)), 2.0, atol=1e-10
    )
    assert int(evidence.status[1]) == int(d.MeshfreeRowStatus.RANK_DEFICIENT)
    np.testing.assert_array_equal(weights[1], 0.0)


def test_cross_target_coordinate_transpose_and_weighted_hilbert_adjoint() -> None:
    d = phx.discretization
    points = _points()
    targets = jnp.asarray(((0.2, -0.3), (-0.4, 0.6)))
    relation = d.MeshfreeNeighborhoodPlan(points, 9, targets=targets).prepare()
    stencils = d.prepare_local_stencils(
        relation,
        points,
        targets,
        (d.MeshfreeFunctional(((1, 0),), (1.0,)),),
        d.LocalStencilPolicy(),
    )
    source_mass = jnp.linspace(0.5, 2.0, 9)
    target_mass = jnp.asarray((3.0, 0.7))
    source = phx.linalg.ArraySpace((9,), pairing=phx.linalg.DiagonalPairing(source_mass))
    target = phx.linalg.ArraySpace((2,), pairing=phx.linalg.DiagonalPairing(target_mass))
    operator = d.MeshfreeOperator(stencils, source=source, target=target)
    x = jnp.sin(jnp.arange(9, dtype=jnp.float64))
    y = jnp.asarray((0.3, -0.7))
    transpose = operator.transpose_mv(y)
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(x), y), jnp.vdot(x, transpose), atol=1e-12
    )
    adjoint = operator.adjoint_mv(y)
    np.testing.assert_allclose(
        jnp.vdot(operator.mv(x), target_mass * y),
        jnp.vdot(x, source_mass * adjoint),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adjoint, operator.transpose_mv(target_mass * y) / source_mass, atol=1e-12
    )
    _, pullback = jax.vjp(operator.apply, x)
    np.testing.assert_allclose(pullback(y)[0], transpose, atol=1e-12)


def test_neighbor_gap_certifies_motion_of_sources_and_targets() -> None:
    d = phx.discretization
    sources = jnp.asarray(((0.0,), (1.0,), (3.0,)))
    targets = jnp.asarray(((0.2,),))
    prepared = d.MeshfreeNeighborhoodPlan(sources, 2, targets=targets).prepare()
    np.testing.assert_allclose(prepared.trust_margin, 0.5)
    np.testing.assert_array_equal(prepared.relation.source_indices, ((0, 1),))
    # At delta .4 < .5 both source and target movement preserve the selected set.
    moved = sources + jnp.asarray(((-0.4,), (-0.4,), (-0.4,)))
    moved_target = targets + 0.4
    again = d.MeshfreeNeighborhoodPlan(moved, 2, targets=moved_target).prepare()
    assert set(np.asarray(again.relation.source_indices[0])) == {0, 1}
    tied = d.MeshfreeNeighborhoodPlan(
        jnp.asarray(((-1.0,), (1.0,), (2.0,))), 1, targets=jnp.zeros((1, 1))
    ).prepare()
    np.testing.assert_array_equal(tied.trust_margin, 0.0)


def test_spatial_admission_duplicates_active_sources_and_radius_capacity() -> None:
    d = phx.discretization
    with pytest.raises(ValueError, match="duplicates"):
        d.MeshfreeNeighborhoodPlan(jnp.asarray(((0.0,), (0.0,))), 1)
    with pytest.raises(ValueError, match="finite"):
        d.MeshfreeNeighborhoodPlan(jnp.asarray(((0.0,), (jnp.nan,))), 1)
    with pytest.raises(TypeError, match="integer"):
        d.MeshfreeNeighborhoodPlan(jnp.asarray(((0.0,), (1.0,))), True)
    points = jnp.asarray(((0.0,), (1.0,), (2.0,), (4.0,)))
    prepared = d.MeshfreeNeighborhoodPlan(
        points,
        2,
        targets=jnp.asarray(((1.1,),)),
        source_active=jnp.asarray((True, False, True, True)),
    ).prepare()
    assert set(np.asarray(prepared.relation.source_indices[0])) == {0, 2}
    edges = d.MeshfreeEdgeRelationPlan(points, 1.1, 2).prepare().relation
    valid = np.asarray(edges.valid)
    pairs = set(
        zip(
            np.asarray(edges.source_indices)[valid],
            np.asarray(edges.target_indices)[valid],
            strict=True,
        )
    )
    assert pairs == {(0, 1), (1, 2)}
    with pytest.raises(ValueError, match="capacity"):
        d.MeshfreeEdgeRelationPlan(points, 1.1, 1).prepare()


def test_point_cloud_mixed_derivative_transpose_and_dual_refusal() -> None:
    d = phx.discretization
    points = _points()
    prepared = d.PointCloudPlan(points, jnp.arange(1.0, 10.0), neighbors=9).prepare()
    values = _polynomial(points)
    np.testing.assert_allclose(
        prepared.mixed_partial_derivative(values, multi_index=(1, 1)), 2.0, atol=1e-10
    )
    np.testing.assert_allclose(prepared.laplacian(values), 8.0, atol=1e-10)
    y = jnp.cos(jnp.arange(9, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(prepared.partial_derivative(values, axis=0), y),
        jnp.vdot(values, prepared.transpose_partial_derivative(y, axis=0)),
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="dual"):
        prepared.divergence(jnp.ones((9, 2)), dual=True)


def _brute_shell(
    points: np.ndarray, radius: float, half_width: float, periodic: bool
) -> tuple[float, int]:
    relative = points[:, None, :] - points[None, :, :]
    if periodic:
        relative = relative - np.round(relative)
    deviation = np.abs(np.linalg.norm(relative, axis=-1) - radius)
    deviation = deviation[~np.eye(points.shape[0], dtype=bool)]
    return min(float(deviation.min()), half_width), int(np.sum(deviation <= half_width))


@pytest.mark.parametrize("periodic", [False, True], ids=["bounded", "periodic"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_radius_shell_witness_matches_brute_force(dimension: int, periodic: bool) -> None:
    spatial = phx.discretization.spatial
    points = np.random.default_rng(dimension).uniform(0.0, 1.0, (40, dimension))
    address = spatial.MortonAddressPlan(
        (0.0,) * dimension, (1.0,) * dimension, 10, periodic_axes=(periodic,) * dimension
    )
    radius = 0.3
    for half_width in (0.002, 0.05):
        gap, incidences = _brute_shell(points, radius, half_width, periodic)
        witness = spatial.MortonRadiusShellWitnessPlan(
            address, 40, 40, maximum_candidates=40
        ).certify(
            jnp.asarray(points),
            jnp.asarray(points),
            radius,
            half_width,
            exclude_self=True,
        )
        evidence = witness.evidence
        assert bool(evidence.successful)
        assert int(evidence.shell_incidences) == incidences
        assert bool(evidence.certified) == (incidences == 0)
        # Conservative: never above the exact gap, below it only by rounding.
        assert float(evidence.certified_gap) <= gap
        np.testing.assert_allclose(float(evidence.certified_gap), gap, atol=1e-12)
        assert int(evidence.candidate_evaluations) > 0


def test_radius_shell_witness_refuses_ties_overflow_and_missing_owners() -> None:
    spatial = phx.discretization.spatial
    address = spatial.MortonAddressPlan((-1.0,), (3.0,), 10)
    points = jnp.asarray(((0.0,), (0.5,), (1.25,), (2.5,)))
    plan = spatial.MortonRadiusShellWitnessPlan(address, 4, 4, maximum_candidates=4)
    tie = plan.certify(points, points, 0.5, 0.1, exclude_self=True)
    assert bool(tie.evidence.successful)
    assert not bool(tie.evidence.certified)
    assert float(tie.evidence.certified_gap) == 0.0
    overflow = spatial.MortonRadiusShellWitnessPlan(
        address, 4, 4, maximum_candidates=1
    ).certify(points, points, 0.4, 0.05, exclude_self=True)
    assert not bool(overflow.evidence.successful)
    assert int(overflow.evidence.overflow_rows) > 0
    assert float(overflow.evidence.certified_gap) == 0.0
    assert not bool(overflow.evidence.certified)
    outside = plan.certify(points, points.at[3, 0].set(5.0), 0.4, 0.05, exclude_self=True)
    assert int(outside.evidence.invalid_targets) == 1
    assert not bool(outside.evidence.successful)
    assert float(outside.evidence.certified_gap) == 0.0


def test_exterior_topology_trust_uses_the_bounded_shell_witness() -> None:
    from phydrax.discretization.meshfree._exterior import MeshfreeExteriorCalculusPlan

    rng = np.random.default_rng(4)
    grid = np.asarray([(x, y) for x in range(5) for y in range(5)], dtype=np.float64)
    points = 0.25 * grid + rng.uniform(-0.02, 0.02, grid.shape)
    radius = 0.3
    prepared = MeshfreeExteriorCalculusPlan(
        points, radius, 200, node_volumes=np.full(25, 0.0625)
    ).prepare()
    gap, _ = _brute_shell(points, radius, 0.5 * radius, periodic=False)
    assert prepared.topology_trust_margin <= 0.5 * gap
    np.testing.assert_allclose(prepared.topology_trust_margin, 0.5 * gap, atol=1e-12)


def test_periodic_neighborhood_uses_minimum_images_and_stable_tie_order() -> None:
    d = phx.discretization
    spatial = d.spatial
    sources = np.asarray(((0.05,), (0.3,), (0.6,), (0.95,)))
    targets = np.asarray(((0.01,),))
    periodic = spatial.MortonAddressPlan((0.0,), (1.0,), 12, periodic_axes=(True,))
    wrapped = d.MeshfreeNeighborhoodPlan(
        sources, 2, targets=targets, address=periodic
    ).prepare()
    assert set(np.asarray(wrapped.relation.source_indices[0])) == {0, 3}
    offsets = np.asarray(wrapped.offsets(sources, targets))[0, :, 0]
    np.testing.assert_allclose(sorted(offsets), (-0.06, 0.04), atol=1e-12)
    bounded = d.MeshfreeNeighborhoodPlan(sources, 2, targets=targets).prepare()
    assert set(np.asarray(bounded.relation.source_indices[0])) == {0, 1}
    tie = np.asarray(((-1.0,), (1.0,), (3.0,)))
    origin = np.zeros((1, 1))
    default = d.MeshfreeNeighborhoodPlan(tie, 1, targets=origin).prepare()
    relabeled = d.MeshfreeNeighborhoodPlan(
        tie, 1, targets=origin, source_ids=np.asarray((7, 2, 9))
    ).prepare()
    assert int(default.relation.source_indices[0, 0]) == 0
    assert int(relabeled.relation.source_indices[0, 0]) == 1
    slab = spatial.MortonAddressPlan(
        (0.0, 0.0), (1.0, 10.0), 10, periodic_axes=(True, False)
    )
    column = np.stack((np.full(6, 0.5), np.arange(6.0)), axis=1)
    with pytest.raises(ValueError, match="half the periodic cell"):
        d.MeshfreeNeighborhoodPlan(column, 2, address=slab).prepare()
    with pytest.raises(ValueError, match="inside the declared address"):
        d.MeshfreeNeighborhoodPlan(column + 20.0, 2, address=slab)
    float32 = phx.discretization.meshfree.MeshfreePrecisionPolicy(
        geometry_dtype="float32"
    )
    single = d.MeshfreeNeighborhoodPlan(sources, 2, precision=float32).prepare()
    assert single.precision.policy_id == float32.policy_id
    assert single.distances.dtype == np.float32


def test_capacity_map_keeps_stable_ids_for_bulk_padding() -> None:
    meshfree = phx.discretization.meshfree
    mapping = meshfree.MeshfreeCapacityPolicy((4, 8)).allocate(
        5, stable_ids=np.asarray((30, 10, 50, 20, 40))
    )
    assert mapping.capacity == 8
    np.testing.assert_array_equal(mapping.storage_ids, (30, 10, 50, 20, 40, -1, -1, -1))
    np.testing.assert_array_equal(
        mapping.compact_positions(np.asarray((20, 99, 30))), (3, -1, 0)
    )
    np.testing.assert_array_equal(mapping.active_mask, (True,) * 5 + (False,) * 3)
    with pytest.raises(ValueError, match="strictly positive"):
        mapping.positive_space(np.asarray((1.0, 1.0, 0.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="unique"):
        meshfree.MeshfreeCapacityMap(4, np.asarray((0, 2)), stable_ids=np.asarray((1, 1)))


def test_point_cloud_fixed_support_refresh_is_status_returning() -> None:
    d = phx.discretization
    rng = np.random.default_rng(8)
    grid = np.asarray([(x, y) for x in range(6) for y in range(6)], dtype=np.float64)
    points = 0.2 * grid + rng.uniform(-0.03, 0.03, grid.shape)
    prepared = d.PointCloudPlan(
        points, np.full(36, 0.04), point_ids=np.arange(100, 136)
    ).prepare()
    assert isinstance(prepared.support.topology, phx.discretization.PointTopology)
    assert prepared.support.topology.refreshable_neighborhoods
    np.testing.assert_array_equal(prepared.stable_ids, np.arange(100, 136))
    trust = float(jnp.min(prepared.trust_radius))
    assert trust > 0
    moved = points + 0.5 * trust * np.asarray((0.8, -0.6)) * np.cos(points)
    refresh = prepared.refresh(moved)
    assert int(refresh.status) == int(d.meshfree.LocalStencilRefreshStatus.ACCEPTED)
    candidate = refresh.discretization
    np.testing.assert_array_equal(candidate.plan.points, points)
    np.testing.assert_array_equal(candidate.points, moved)
    np.testing.assert_array_equal(
        candidate.quadrature_weights, prepared.quadrature_weights
    )
    quadratic = (
        1.0 + moved[:, 0] ** 2 - 2.0 * moved[:, 0] * moved[:, 1] + 3 * moved[:, 1] ** 2
    )
    np.testing.assert_allclose(candidate.laplacian(quadratic), 8.0, atol=1e-7)
    np.testing.assert_allclose(
        candidate.mixed_partial_derivative(quadratic, multi_index=(1, 1)), -2.0, atol=1e-7
    )

    def laplacian_at(coordinates: Array) -> Array:
        return prepared.refresh(coordinates).discretization.laplacian(
            jnp.sin(coordinates[:, 0]) * coordinates[:, 1]
        )

    _, tangent = jax.jvp(
        laplacian_at, (jnp.asarray(points),), (jnp.asarray(np.cos(points)),)
    )
    assert bool(jnp.all(jnp.isfinite(tangent)))
    far = prepared.refresh(points + 4.0 * trust)
    assert int(far.status) == int(d.meshfree.LocalStencilRefreshStatus.SUPPORT_EXCEEDED)
    assert not bool(far.accepted)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from fractions import Fraction
from itertools import combinations, permutations
from time import monotonic
from typing import assert_never, Literal

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, jvp, vjp

from phydrax._meshcore import meshcore_available, MeshcoreError, MeshcoreStatus, TetMesh3D
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem import (
    discontinuous_element,
    form_element,
    lagrange_element,
    prepare_l2_projection_target,
    prepare_projection_field_transfer,
)
from phydrax.geometry import CommonRefinementPolicy, prepare_common_refinement
from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._metric import metric_simplex_quality
from phydrax.meshing._tetra_metric import (
    _append_global_size_branches,
    _global_size_branch_gradients,
    _global_size_branch_targets,
    _global_size_canonical_residual_bound,
    _global_size_residual,
    _global_size_shape_residual,
    _GlobalSizeArgs,
    _GlobalSizeBranchBank,
    _GlobalSizeTrialPolicy,
    _size_quantile,
    execute_tetra_metric_adaptation,
    MetricRemeshingStatus,
)
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.optim._least_squares import _prepare_residual_model


def _trial_policy(
    parameters: Array, direction: Array, residual: Array, args: _GlobalSizeArgs
) -> tuple[Array, Array, Array, Array]:
    model = _prepare_residual_model(
        lambda value: _global_size_residual(value, args), parameters
    )
    work_limit = jnp.asarray(1_000_000_000, dtype=jnp.int64)
    result = _GlobalSizeTrialPolicy()(
        parameters, direction, residual, model.jacobian, work_limit, args
    )
    assert bool(result.receipts_valid())
    assert int(result.jvp_actions) == int(result.vjp_actions) == 1
    assert 0 <= int(result.model_visits) <= 11
    assert 0 <= int(result.model_work_units) <= int(work_limit)
    assert not bool(result.resource_refused)
    return result.direction, result.scale, result.model_image, result.valid


def _box() -> CellMesh:
    points = np.asarray(tuple(np.ndindex(2, 2, 2)), dtype=np.float64)
    cells = []
    for order in permutations(range(3)):
        corner = np.zeros((3,), dtype=np.int32)
        path = [0]
        for axis in order:
            corner[axis] = 1
            path.append(int(np.ravel_multi_index(tuple(corner), (2, 2, 2))))
        cells.append(path)
    tetrahedra = np.asarray(cells, dtype=np.int32)
    negative = np.linalg.det(points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]]) < 0.0
    tetrahedra[negative, 0], tetrahedra[negative, 1] = (
        tetrahedra[negative, 1].copy(),
        tetrahedra[negative, 0].copy(),
    )
    points *= np.asarray((2.0, 0.5, 1.0), dtype=np.float64)
    return CellMesh(points, (CellBlock("box", "tetrahedron", tetrahedra),))


def test_metric_quality_is_regular_after_anisotropic_coordinate_pullback() -> None:
    regular = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, np.sqrt(3.0) / 2.0, 0.0),
            (0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)),
        ),
        dtype=np.float64,
    )
    points = regular * np.asarray((2.0, 0.5, 1.0), dtype=np.float64)
    metric = np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (4, 3, 3))
    quality = metric_simplex_quality(
        metric, points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    np.testing.assert_allclose(quality, 1.0, rtol=1.0e-12, atol=1.0e-12)


def test_unit_edges_do_not_hide_tetrahedral_slivers() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0e-9)),
        dtype=np.float64,
    )
    edges = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    assert np.all((lengths >= 1.0) & (lengths <= np.sqrt(2.0) + 1.0e-12))
    quality = np.asarray(
        metric_simplex_quality(
            np.broadcast_to(np.eye(3), (4, 3, 3)),
            points,
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        )
    )
    assert quality[0] < 1.0e-5


@pytest.mark.parametrize("crossing_length", (3.0, np.nextafter(2.0, np.inf)))
def test_statistical_rank_residual_tracks_crossing_edge_and_ignores_padding(
    crossing_length: float,
) -> None:
    """A formerly unweighted edge must govern after it crosses the size rank."""
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
        )
    )
    basis = jnp.broadcast_to(jnp.eye(3), (5, 3, 3))
    args = _GlobalSizeArgs(
        origins,
        basis,
        jnp.asarray(((0, 1), (0, 4), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins).at[1, 0].set(crossing_length - 1.0)
    direction = jnp.zeros_like(origins).at[1, 0].set(1.0)
    residual, derivative = jvp(
        lambda value: _global_size_residual(value, args),
        (parameters,),
        (direction,),
    )
    # The old fixed-identity support would remain at edge (0, 4), with
    # residual one and zero derivative in the crossing edge's direction.
    np.testing.assert_array_equal(residual[:1], (crossing_length - 1.0,))
    np.testing.assert_array_equal(derivative[:1], (1.0,))
    _, pullback = vjp(lambda value: _global_size_residual(value, args), parameters)
    seed = jnp.zeros_like(residual).at[0].set(1.0)
    reverse = pullback(seed)[0]
    np.testing.assert_array_equal(reverse, direction.at[0, 0].set(-1.0))
    reordered = args._replace(edges=args.edges[jnp.asarray((1, 0, 2))])
    np.testing.assert_array_equal(
        _global_size_residual(parameters, reordered)[:1],
        residual[:1],
    )


@pytest.mark.parametrize("quantile", (0.5, 0.95))
def test_statistical_residual_uses_interpolated_statistic_not_each_support_edge(
    quantile: float,
) -> None:
    lower = 0.5
    upper = lower + (1.0 - lower) / quantile
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (lower, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (upper, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (5, 3, 3)),
        jnp.asarray(((0, 1), (0, 4), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((quantile,)),
        jnp.asarray((1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((lower,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    residual = _global_size_residual(jnp.zeros_like(origins), args)
    expected = np.quantile(np.asarray((lower, upper)), quantile) - 1.0
    np.testing.assert_array_equal(residual[:1], (expected,))
    assert lower != 1.0 and upper != 1.0


@pytest.mark.parametrize("edge_velocities", ((-1.0, -2.0, 1.0), (-1.0, -2.0, 0.0)))
@pytest.mark.parametrize("quantile", (0.5, 0.95, 1.0))
def test_statistical_repeated_tie_uses_outgoing_rank_for_finite_movement(
    quantile: float,
    edge_velocities: tuple[float, float, float],
) -> None:
    """No fixed tied-row identity can replace the actual outgoing statistic."""
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (7, 3, 3)),
        jnp.asarray(((0, 4), (0, 5), (0, 6), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((quantile,)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    velocities = np.asarray(edge_velocities)
    step = 0.125
    parameters = jnp.zeros_like(origins).at[4:, 0].set(step * velocities)
    baseline = _global_size_residual(parameters, args)
    for order in permutations(range(3)):
        reordered = args._replace(edges=args.edges[jnp.asarray((*order, 3))])
        residual = _global_size_residual(parameters, reordered)
        np.testing.assert_array_equal(residual, baseline)
        assert bool(jnp.all(jnp.isfinite(residual)))
    # Reversing movement must reselect the outgoing ranks as well; it
    # cannot reuse the branch selected for the positive displacement.
    reverse = _global_size_residual(-parameters, args)[0]
    initial = _global_size_residual(jnp.zeros_like(origins), args)[0]
    assert float(reverse) > float(initial)
    if quantile == 0.5 and edge_velocities[-1] == 0.0:
        # A tied support that does not move has zero branch directional
        # derivative, yet it must not falsely prevent median descent.
        assert float(baseline[0] * baseline[0]) < float(initial * initial)


def test_statistical_tie_linear_model_is_symmetric_not_directional_rank_proxy() -> None:
    lengths = jnp.asarray((2.0, 2.0, 2.0, jnp.nan))
    quantiles = jnp.asarray((0.5, 0.95))
    direction = jnp.asarray((-1.0, -2.0, 0.0, 0.0))
    primal, image = jvp(
        lambda value: _size_quantile(value, quantiles), (lengths,), (direction,)
    )
    np.testing.assert_array_equal(primal, (2.0, 2.0))
    np.testing.assert_array_equal(image, (-1.0, -1.0))
    _, pullback = vjp(lambda value: _size_quantile(value, quantiles), lengths)
    gradient = pullback(jnp.asarray((1.0, 0.0)))[0]
    np.testing.assert_allclose(
        gradient, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0), rtol=0.0, atol=0.0
    )
    # The outgoing p95 derivative is -0.1, not the linear model's -1.
    # This distinction is essential to native trial merit prediction.
    step = 0.125
    actual = _size_quantile(lengths + step * direction, quantiles)
    expected = np.quantile(
        np.asarray((2.0, 2.0, 2.0)) + step * np.asarray(direction[:3]),
        np.asarray(quantiles),
    )
    np.testing.assert_array_equal(actual, expected)
    assert float(actual[1]) != float(primal[1] + step * image[1])
    order = jnp.asarray((2, 0, 1, 3))
    reordered, reordered_image = jvp(
        lambda value: _size_quantile(value, quantiles),
        (lengths[order],),
        (direction[order],),
    )
    np.testing.assert_array_equal(reordered, primal)
    np.testing.assert_array_equal(reordered_image, image)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_statistical_tolerance_hinge_preserves_one_sided_finite_movement(
    sign: float,
) -> None:
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.25, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (5, 3, 3)),
        jnp.asarray(((0, 4), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.asarray((1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.25),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    origin = _global_size_residual(jnp.zeros_like(origins), args)
    np.testing.assert_array_equal(origin[:2], (0.0, 0.0))
    parameters = jnp.zeros_like(origins).at[4, 0].set(sign * 0.125)
    residual = _global_size_residual(parameters, args)
    np.testing.assert_array_equal(residual[:2], (max(sign * 0.125, 0.0),) * 2)
    assert bool(jnp.all(jnp.isfinite(residual)))


def test_statistical_trial_policy_uses_exact_outgoing_tied_rank_merit() -> None:
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (7, 3, 3)),
        jnp.asarray(((0, 4), (0, 5), (0, 6), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[4:, 0].set(jnp.asarray((-1.0, -2.0, 0.0)))
    residual, image = jvp(
        lambda value: _global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, scale, outgoing, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    actual = _global_size_residual(parameters + scale * direction, args)
    assert 0.0 < float(scale) <= 1.0
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    np.testing.assert_allclose(image[:2], (-1.0, -1.0), rtol=0.0, atol=1e-15)
    # The rank-frontier policy may replace the authored direction; its
    # published model image belongs to that selected finite trajectory.
    np.testing.assert_allclose(
        outgoing, (actual - residual) / scale, rtol=0.0, atol=1e-15
    )
    assert float(jnp.sum(residual * outgoing)) < 0.0
    assert not bool(_GlobalSizeTrialPolicy().termination_valid(residual, args))
    order = jnp.asarray((2, 0, 1, 3))
    reordered = args._replace(edges=args.edges[order], edge_valid=args.edge_valid[order])
    direction, repeated_scale, repeated_image, repeated_valid = _trial_policy(
        parameters, direction, residual, reordered
    )
    np.testing.assert_array_equal(repeated_scale, scale)
    np.testing.assert_array_equal(repeated_image, outgoing)
    assert bool(repeated_valid)


def test_statistical_trial_merit_selects_outgoing_shortest_edge_in_active_radius_hinge() -> (
    None
):
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (4, 3, 3)),
        jnp.asarray(((0, 1), (0, 2), (0, 3), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5,)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(0.5),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[1, 0].set(1.0).at[2, 1].set(-2.0)
    residual, image = jvp(
        lambda value: _global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, _, outgoing, valid = _trial_policy(parameters, direction, residual, args)
    # Size is already satisfied; the chosen direction increases the actual
    # active shape merit, so refusal is truthful, not scientific infeasibility.
    assert not bool(valid)
    radius_row = args.quantiles.size + args.cells.shape[0]
    assert float(residual[radius_row]) > 0.0
    step = 1e-6
    actual = _global_size_residual(parameters + step * direction, args)
    np.testing.assert_allclose(
        outgoing[radius_row],
        (actual[radius_row] - residual[radius_row]) / step,
        rtol=1e-5,
        atol=1e-5,
    )
    assert float(outgoing[radius_row]) > float(image[radius_row])


def test_statistical_finite_rank_model_crosses_subulp_merit_event_with_real_descent() -> (
    None
):
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.8, 0.0, 0.0),
            (0.9, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (np.nextafter(1.0, np.inf), 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (11, 3, 3)),
        jnp.asarray(((0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.asarray((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[8, 0].set(-0.01).at[10, 0].set(-0.001)
    residual, linear_image = jvp(
        lambda value: _global_size_residual(value, args), (parameters,), (direction,)
    )
    # The adjacent-to-unit edge meets the fixed unit block at this authored
    # rank event. That represented movement cannot change either residual.
    event_scale = np.spacing(1.0) / 0.01
    clipped = _global_size_residual(parameters + event_scale * direction, args)
    np.testing.assert_array_equal(clipped, residual)
    direction, scale, finite_image, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    actual = _global_size_residual(parameters + scale * direction, args)
    # The collinear affine model admits finite progress without disturbing
    # the already satisfied median or adding any scientific tolerance.
    np.testing.assert_array_equal(scale * finite_image, actual - residual)
    assert float(actual[0]) == 0.0
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    model = _prepare_residual_model(
        lambda value: _global_size_residual(value, args), parameters
    )
    original_direction = parameters.at[8, 0].set(-0.01).at[10, 0].set(-0.001)
    receipt = _GlobalSizeTrialPolicy()(
        parameters,
        original_direction,
        residual,
        model.jacobian,
        jnp.asarray(1_000_000_000, dtype=jnp.int64),
        args,
    )
    assert bool(receipt.receipts_valid())
    assert not bool(receipt.resource_refused)
    allowance = receipt.model_work_units - 1
    refused = _GlobalSizeTrialPolicy()(
        parameters, original_direction, residual, model.jacobian, allowance, args
    )
    assert bool(refused.resource_refused)
    assert not bool(refused.valid)
    assert bool(refused.receipts_valid())
    assert int(refused.model_work_units) <= int(allowance)


def test_statistical_finite_rank_model_reselects_support_after_crossing() -> None:
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (np.nextafter(2.0, np.inf), 0.0, 0.0),
            (3.0, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (7, 3, 3)),
        jnp.asarray(((0, 4), (0, 5), (0, 6), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5,)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[4, 0].set(-0.125).at[5, 0].set(-0.25)
    residual, linear_image = jvp(
        lambda value: _global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, scale, finite_image, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    actual = _global_size_residual(parameters + scale * direction, args)
    np.testing.assert_array_equal(scale * finite_image, actual - residual)
    # This is a genuine finite support switch, not a rounding difference.
    assert float(finite_image[0]) > float(linear_image[0])
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))


def test_statistical_finite_rank_model_selects_intermediate_descent_not_full_ascent() -> (
    None
):
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (2.1, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (7, 3, 3)).at[:4].set(0.0),
        jnp.asarray(((0, 4), (0, 5), (0, 6), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5,)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[4, 0].set(4.0).at[5, 0].set(-1.0)
    original_direction = direction
    residual, image = jvp(
        lambda value: _global_size_residual(value, args), (parameters,), (direction,)
    )
    full = _global_size_residual(direction, args)
    direction, scale, secant, valid = _trial_policy(parameters, direction, residual, args)
    actual = _global_size_residual(scale * direction, args)
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    assert bool(jnp.any(direction != original_direction))
    np.testing.assert_array_equal(direction[:4], jnp.zeros((4, 3)))
    assert float(jnp.sum(full * full)) > float(jnp.sum(residual * residual))
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    # A finite secant is a distinct model, not a bit-identical actual
    # residual after subtract/divide/multiply/reconstruct operations.
    prediction = -scale * jnp.sum(residual * secant) - 0.5 * scale**2 * jnp.sum(
        secant * secant
    )
    assert float(prediction) > 0.0
    assert bool(jnp.all(jnp.isfinite(secant)))


def test_statistical_finite_radius_model_reselects_all_cell_edges() -> None:
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, np.nextafter(1.0, np.inf), 0.0),
            (0.0, 0.0, 2.5),
        )
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (4, 3, 3)),
        jnp.asarray(((0, 1), (0, 2), (0, 3), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((2.5,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(1.2),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[1, 0].set(0.02).at[2, 1].set(-0.04).at[3, 2].set(-0.5)
    residual, linear_image = jvp(
        lambda value: _global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, scale, finite_image, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    radius_row = args.quantiles.size + args.cells.shape[0]
    # The x edge is initially shortest but the y edge becomes shortest.
    # Freezing x would falsely deactivate this finite modeled radius hinge.
    assert float(residual[radius_row] + scale * finite_image[radius_row]) > 0.0
    actual = _global_size_residual(parameters + scale * direction, args)
    assert float(actual[radius_row]) > 0.0
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))


def test_statistical_rank_model_zero_work_budget_refuses_before_actions() -> None:
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (4, 3, 3)),
        jnp.asarray(((0, 1), (0, 2), (0, 3), (0, 0))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(0.5),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    model = _prepare_residual_model(
        lambda value: _global_size_residual(value, args), parameters
    )
    result = _GlobalSizeTrialPolicy()(
        parameters,
        -jnp.ones_like(parameters),
        model.residual,
        model.jacobian,
        jnp.asarray(0, dtype=jnp.int64),
        args,
    )
    assert bool(result.resource_refused)
    assert not bool(result.valid)
    assert bool(result.receipts_valid())
    assert int(result.jvp_actions) == int(result.vjp_actions) == 0
    assert int(result.model_work_units) == int(result.model_visits) == 0


def test_statistical_branch_bank_retains_tied_rows_and_refuses_overflow() -> None:
    empty = jnp.zeros((2,), dtype=jnp.int32)
    bank = _GlobalSizeBranchBank(empty, empty, empty, jnp.asarray(0), jnp.asarray(False))
    bank, work = _append_global_size_branches(
        bank,
        jnp.asarray((1, 1, 1)),
        jnp.asarray((7, 8, 7)),
        jnp.asarray((7, 8, 7)),
        jnp.asarray((True, True, True)),
    )
    assert int(bank.count) == 2
    assert not bool(bank.overflow)
    assert int(work) > 0
    np.testing.assert_array_equal(bank.lower, (7, 8))
    previous = bank
    bank, work = _append_global_size_branches(
        bank,
        jnp.asarray((1,)),
        jnp.asarray((9,)),
        jnp.asarray((9,)),
        jnp.asarray((True,)),
    )
    assert bool(bank.overflow)
    assert int(bank.count) == 2
    assert int(work) > 0
    np.testing.assert_array_equal(bank.lower, previous.lower)
    np.testing.assert_array_equal(bank.upper, previous.upper)


def test_statistical_branch_rows_preserve_individual_targets_and_source_tangents() -> (
    None
):
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.6, 0.0, 0.0),
            (0.7, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (1.2, 0.0, 0.0),
            (1.2, 0.0, 0.0),
        )
    )
    edges = jnp.asarray(((0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 0)))
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (9, 3, 3)).at[:4].set(0.0),
        edges,
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.asarray((1.0, 1.0, 1.0, 1.0, 1.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    residual = _global_size_residual(jnp.zeros_like(origins), args)
    delta = origins[edges[:, 1]] - origins[edges[:, 0]]
    lengths = jnp.sqrt(
        jnp.where(args.edge_valid > 0.0, jnp.sum(delta * delta, axis=-1), 1.0)
    )
    bank = _GlobalSizeBranchBank(
        jnp.asarray((0, 1, 1)),
        jnp.asarray((2, 3, 4)),
        jnp.asarray((2, 3, 4)),
        jnp.asarray(3),
        jnp.asarray(False),
    )
    gradients, active, equality = _global_size_branch_gradients(
        bank, delta, lengths, residual, jnp.zeros_like(origins), args
    )
    expected = (
        jnp.zeros((3, 9, 3))
        .at[0, 6, 0]
        .set(1.0)
        .at[1, 7, 0]
        .set(-1.0)
        .at[2, 8, 0]
        .set(-1.0)
    )
    np.testing.assert_array_equal(gradients, expected)
    np.testing.assert_array_equal(active, (True, True, True))
    np.testing.assert_array_equal(equality, (False, False, False))
    gradients, active, equality = _global_size_branch_gradients(
        bank, delta, lengths, residual.at[0].set(0.0), jnp.zeros_like(origins), args
    )
    np.testing.assert_array_equal(gradients[0], expected[0])
    np.testing.assert_array_equal(active, (False, True, True))
    np.testing.assert_array_equal(equality, (True, False, False))


def test_statistical_branch_targets_preserve_actual_component_gaps_and_padding() -> None:
    statistics = jnp.asarray((1, -1, 0, 0, 1, -1, 0, 1), dtype=jnp.int32)
    empty = jnp.zeros((8,), dtype=jnp.int32)
    bank = _GlobalSizeBranchBank(
        statistics, empty, empty, jnp.asarray(3), jnp.asarray(False)
    )
    inequality = jnp.asarray((True, True, True, False, False, False, False, False))
    residual = jnp.asarray((-1.0e-6, 0.18, 3.0, 4.0))
    targets = _global_size_branch_targets(bank, residual, inequality, 2)
    np.testing.assert_array_equal(targets, (0.18, 25.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0))
    equality_residual = residual.at[0].set(0.0)
    targets = _global_size_branch_targets(
        bank, equality_residual, inequality.at[2].set(False), 2
    )
    np.testing.assert_array_equal(targets, (0.18, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_statistical_canonical_primal_bound_uses_original_evaluation_guard() -> None:
    # Original 32 steps / four trials can make 162 real primals, but the
    # original 128-evaluation outer guard permits only a last complete step
    # and final model beyond that guard. Finite policy models are separate.
    assert _global_size_canonical_residual_bound() == 128 + 4 + 1
    assert _global_size_canonical_residual_bound() >= 1 + 32 * 2 + 1


def test_statistical_finite_model_does_not_trade_hard_dihedral_for_size_descent() -> None:
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 1.5))
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (4, 3, 3)),
        jnp.asarray(tuple(combinations(range(4), 2))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.ones((6,)),
        jnp.ones((1,)),
        jnp.asarray((3.375,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(np.cos(np.deg2rad(10.0))),
        jnp.asarray(1000.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[1, 0].set(-0.9).at[2, 1].set(-0.9).at[3, 2].set(-1.49)
    residual = _global_size_residual(parameters, args)
    full = _global_size_shape_residual(origins + direction, args)
    assert bool(jnp.any(full[1:] > 0.0))
    selected, scale, _, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    assert float(scale) < 1.0
    actual = _global_size_shape_residual(origins + scale * selected, args)
    np.testing.assert_array_equal(actual[1:], 0.0)


def test_statistical_refused_full_point_recovers_with_distinct_finite_scale() -> None:
    height = 1.5 * np.tan(np.deg2rad(10.01)) / np.sqrt(2.0)
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, height))
    )
    args = _GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (4, 3, 3)),
        jnp.asarray(tuple(combinations(range(4), 2))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.ones((6,)),
        jnp.ones((1,)),
        jnp.asarray((2.25 * height,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(np.cos(np.deg2rad(10.0))),
        jnp.asarray(1000.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = (
        parameters.at[1, 0].set(-1.0e-5).at[2, 1].set(-1.0e-5).at[3, 2].set(-3.0e-4)
    )
    residual = _global_size_residual(parameters, args)
    assert bool(jnp.all(residual[2:] == 0.0))
    full = _global_size_residual(direction, args)
    assert bool(jnp.all(jnp.abs(full[:2]) < jnp.abs(residual[:2])))
    assert bool(jnp.any(full[4:] > 0.0))
    selected, scale, image, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    assert float(scale) == 0.5
    actual = _global_size_residual(scale * selected, args)
    np.testing.assert_array_equal(actual[4:], 0.0)
    np.testing.assert_array_equal(scale * image, actual - residual)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_anisotropic_box_remesh_keeps_exact_domain_and_p1_source_coverage() -> None:
    source = _box()
    coordinates = np.asarray(source.coordinates).copy()
    metric = np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (8, 3, 3))
    outcome = execute_tetra_metric_adaptation(
        source, metric, maximum_passes=12, relocation=False
    )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    target, lineage, stencil = assemble_topology_edit(
        source, outcome.edit, numeric_version="box-remesh"
    )
    corners = np.asarray(target.coordinates)[np.asarray(target.blocks[0].vertices)]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.all(volumes > 0.0)
    assert np.sum(volumes) == pytest.approx(1.0, abs=1.0e-12)
    assert outcome.evidence.out_of_range_edges == 0
    assert lineage.source_topology_id == source.topology_id
    sources = np.asarray(outcome.edit.stencil_sources)
    weights = np.asarray(outcome.edit.stencil_weights)
    source_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source.vertex_global_ids))
    }
    reproduced = np.sum(
        coordinates[
            np.asarray(
                [[source_rows[int(value)] for value in row] for row in sources],
                dtype=np.int32,
            )
        ]
        * weights[..., None],
        axis=1,
    )
    np.testing.assert_allclose(reproduced, target.coordinates, atol=1.0e-12)
    np.testing.assert_array_equal(source.coordinates, coordinates)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_exhausted_metric_work_keeps_source_and_reports_unmet_edges() -> None:
    source = _box()
    before = np.asarray(source.coordinates).copy()
    outcome = execute_tetra_metric_adaptation(
        source,
        np.broadcast_to(np.diag((1.0, 16.0, 4.0)), (8, 3, 3)),
        maximum_operations=0,
        maximum_passes=4,
    )
    assert outcome.evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
    assert outcome.evidence.out_of_range_edges > 0
    assert outcome.evidence.splits == 0
    np.testing.assert_array_equal(outcome.edit.coordinates, before)
    np.testing.assert_array_equal(source.coordinates, before)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
@pytest.mark.parametrize("family", ["h1", "dg", "hcurl", "hdiv"])
def test_repeated_native_remesh_transfers_each_declared_field_family(
    family: Literal["h1", "dg", "hcurl", "hdiv"],
) -> None:
    match family:
        case "h1":
            element = lagrange_element("tetrahedron", 1)
        case "dg":
            element = discontinuous_element("tetrahedron", 0)
        case "hcurl":
            element = form_element("tetrahedron", 1, 1, proxy="circulation")
        case "hdiv":
            element = form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux")
        case unknown:
            assert_never(unknown)

    def field(points: Array, args: object) -> Array:
        del args
        if family == "hcurl":
            return jnp.asarray((0.5, -0.2, 1.3), dtype=jnp.float64) + jnp.cross(
                jnp.asarray((-0.7, 0.9, 0.4), dtype=jnp.float64),
                points,
            )
        if family == "hdiv":
            return jnp.asarray((0.5, -0.2, 1.3), dtype=jnp.float64) + 2.0 * points
        if family == "dg":
            return 1.0 + points[..., 0]
        return 0.75 + 2.0 * points[..., 0] - 0.3 * points[..., 1] + points[..., 2]

    def dg_content(
        discretization: FiniteElementDiscretization, coefficients: Array
    ) -> float:
        total = 0.0
        for block, routes in zip(
            discretization.mesh.blocks, discretization.dof_maps[0].cell_dofs, strict=True
        ):
            corners = np.asarray(discretization.mesh.coordinates)[
                np.asarray(block.vertices)
            ]
            volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
            averages = np.asarray(coefficients)[np.asarray(routes)[:, 0]]
            total += float(volumes @ averages)
        return total

    mesh = _box()
    specification = FiniteElementFieldSpec("u", element)
    prepared = FiniteElementPlan(mesh, specification).prepare()
    values = prepared.project("u", field)
    for epoch, scale in enumerate((1.0, 4.0), start=1):
        metric = np.broadcast_to(
            scale * np.diag((0.25, 4.0, 1.0)), (mesh.coordinates.shape[0], 3, 3)
        )
        outcome = execute_tetra_metric_adaptation(
            mesh, metric, maximum_passes=16, relocation=False
        )
        assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
        target, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version=f"field-remesh:{epoch}"
        )
        successor = FiniteElementPlan(target, specification).prepare()
        common = prepare_common_refinement(
            mesh, target, policy=CommonRefinementPolicy(overlap_simplices=True)
        )
        transfer = prepare_projection_field_transfer(
            prepared,
            prepare_l2_projection_target(successor, field_name="u"),
            common,
            field_name="u",
        )
        source_content = dg_content(prepared, values) if family == "dg" else 0.0
        source_minimum, source_maximum = float(np.min(values)), float(np.max(values))
        values = transfer.transfer.apply(values)
        assert transfer.evidence.passed
        assert transfer.evidence.defect("coverage") <= transfer.evidence.tolerance
        assert transfer.evidence.defect("reproduction") <= transfer.evidence.tolerance
        if family in ("hcurl", "hdiv"):
            assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
        if family == "dg":
            assert dg_content(successor, values) == pytest.approx(
                source_content, abs=1.0e-11
            )
            assert np.min(values) >= source_minimum - 1.0e-11
            assert np.max(values) <= source_maximum + 1.0e-11
        else:
            np.testing.assert_allclose(
                values, successor.project("u", field), atol=1.0e-10, rtol=1.0e-10
            )
        mesh, prepared = target, successor


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_metric_ring_reconnection_improves_quality_without_changing_boundary() -> None:
    radius, height = 0.75, 0.6
    points = np.asarray(
        (
            (0.0, 0.0, -height),
            (0.0, 0.0, height),
            (radius, 0.0, 0.0),
            (-radius / 2, radius * np.sqrt(3.0) / 2, 0.0),
            (-radius / 2, -radius * np.sqrt(3.0) / 2, 0.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3), (0, 1, 3, 4), (0, 1, 4, 2)), dtype=np.int32)
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    source = CellMesh.from_tetrahedra(points, cells)
    expected = np.asarray(((0, 2, 3, 4), (1, 2, 3, 4)), dtype=np.int32)

    def mean_ratio(tetrahedra: np.ndarray) -> np.ndarray:
        corners = points[tetrahedra]
        determinant = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
        edges = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
        squares = np.sum(
            (corners[:, edges[:, 1]] - corners[:, edges[:, 0]]) ** 2, axis=-1
        )
        return 6.0 * (2.0 * determinant**2) ** (1.0 / 3.0) / np.sum(squares, axis=1)

    requested = float((np.min(mean_ratio(cells)) + np.min(mean_ratio(expected))) / 2.0)
    outcome = execute_tetra_metric_adaptation(
        source,
        np.broadcast_to(np.eye(3), (5, 3, 3)),
        minimum_metric_quality=requested,
        relocation=False,
        maximum_passes=4,
    )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    target, _, _ = assemble_topology_edit(
        source, outcome.edit, numeric_version="ring-remesh"
    )
    actual_cells = np.asarray(target.blocks[0].vertices)
    assert {tuple(row) for row in np.sort(actual_cells, axis=1)} == {
        tuple(row) for row in np.sort(expected, axis=1)
    }
    assert outcome.evidence.minimum_metric_quality >= requested
    assert outcome.evidence.out_of_range_edges == 0
    old_faces = np.sort(
        cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3)), axis=1
    )
    new_faces = np.sort(
        actual_cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3)),
        axis=1,
    )
    old_keys, old_uses = np.unique(old_faces, axis=0, return_counts=True)
    new_keys, new_uses = np.unique(new_faces, axis=0, return_counts=True)
    np.testing.assert_array_equal(old_keys[old_uses == 1], new_keys[new_uses == 1])


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_source_location_budget_refuses_atomically_with_resource_evidence() -> None:
    source = _box()
    before = np.asarray(source.coordinates).copy()
    with pytest.raises(MeshingFailure) as caught:
        execute_tetra_metric_adaptation(
            source,
            np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (8, 3, 3)),
            maximum_location_pairs=0,
        )
    assert caught.value.evidence.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert caught.value.evidence.requested == (("source_location_candidates", 0.0),)
    assert caught.value.evidence.achieved[0][1] > 0.0
    np.testing.assert_array_equal(source.coordinates, before)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_rotated_decimal_constraint_splits_at_an_exact_nonmidpoint() -> None:
    points = np.asarray(
        ((-1.6, 1.4, 0.9), (0.6, 0.5, -0.3), (1.0, 2.0, 1.0), (2.0, 0.0, 3.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    if np.linalg.det(points[cells[0, 1:]] - points[cells[0, :1]]) < 0.0:
        cells = cells[:, (1, 0, 2, 3)]
    faces = cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3))
    segments = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    source_volume = abs(np.linalg.det(points[cells[0, 1:]] - points[cells[0, :1]])) / 6.0
    native = TetMesh3D(
        points,
        cells,
        np.zeros((1,), dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        segments,
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    try:
        construction = native.edge_split_point(0, 1, work_limit=10000)
        assert construction is not None
        point, fraction = construction.position, construction.parameter
        assert 0.0 < fraction < 1.0
        assert fraction != 0.5
        inserted = native.split_edge(
            0, 1, point, source_fraction=fraction, work_limit=10000
        )
        assert inserted is not None
        target = native.arrays()
        corners = target.points[target.tetrahedra]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        assert np.all(volumes > 0.0)
        assert np.sum(volumes) == pytest.approx(source_volume, abs=1.0e-12)
        np.testing.assert_array_equal(target.points[:4], points)
        np.testing.assert_array_equal(target.points[inserted], point)
    finally:
        native.close()


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_coordinated_relocation_commits_once_and_refuses_the_entire_cavity_union() -> (
    None
):
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.2, 0.2),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(
        ((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)), dtype=np.int32
    )
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    segments = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    native = TetMesh3D(
        points,
        cells,
        np.zeros((4,), dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        segments,
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
        max_vertices=32,
        max_tetrahedra=128,
    )
    try:
        inserted = native.split_edge(0, 4, points[4] / 2.0, work_limit=100000)
        assert inserted is not None
        before = native.arrays()
        rows = np.asarray((4, inserted), dtype=np.int32)
        positions = before.points[rows] + np.asarray((0.01, 0.01, 0.01), dtype=np.float64)
        assert native.relocate_vertices(rows, positions, work_limit=100000)
        accepted = native.arrays()
        np.testing.assert_array_equal(accepted.points[rows], positions)
        np.testing.assert_array_equal(accepted.points[:4], points[:4])
        np.testing.assert_array_equal(accepted.tetrahedra, before.tetrahedra)
        np.testing.assert_array_equal(accepted.faces, before.faces)
        np.testing.assert_array_equal(accepted.face_sources, before.face_sources)
        corners = accepted.points[accepted.tetrahedra]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        assert np.all(volumes > 0.0)
        assert np.sum(volumes) == pytest.approx(1.0 / 6.0, abs=1.0e-14)

        inverted = positions.copy()
        inverted[0] += 0.001
        inverted[1] = -1.0
        assert not native.relocate_vertices(rows, inverted, work_limit=100000)
        np.testing.assert_array_equal(native.arrays().points, accepted.points)
        assert not native.relocate_vertices(
            rows,
            positions + 0.001,
            radius_edge_bound=0.5,
            work_limit=100000,
        )
        np.testing.assert_array_equal(native.arrays().points, accepted.points)
        with pytest.raises(MeshcoreError) as caught:
            native.relocate_vertices(rows, positions + 0.001, work_limit=0)
        assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        np.testing.assert_array_equal(native.arrays().points, accepted.points)
        np.testing.assert_array_equal(native.arrays().tetrahedra, accepted.tetrahedra)
    finally:
        native.close()


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_batch_curve_construction_retains_exact_decimal_source_and_reports_refused_rows() -> (
    None
):
    points = np.asarray(
        ((-1.6, 1.4, 0.9), (0.6, 0.5, -0.3), (1.0, 2.0, 1.0), (2.0, 0.0, 3.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    if np.linalg.det(points[cells[0, 1:]] - points[cells[0, :1]]) < 0.0:
        cells = cells[:, (1, 0, 2, 3)]
    faces = cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3))
    native = TetMesh3D(
        points,
        cells,
        np.zeros((1,), dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32),
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    try:
        seed = native.edge_split_point(0, 1, work_limit=10000)
        assert seed is not None
        inserted = native.split_edge(
            0, 1, seed.position, source_fraction=seed.parameter, work_limit=10000
        )
        assert inserted is not None
        before = native.arrays()
        constructed, coordinates, mask = native.construct_curve_points(
            np.asarray((inserted, 0), dtype=np.int32),
            np.asarray(((0, 1), (1, 2)), dtype=np.int32),
            np.asarray(
                (points[0] + 0.625 * (points[1] - points[0]), points[0]), dtype=np.float64
            ),
            work_limit=10000,
        )
        np.testing.assert_array_equal(mask, np.asarray((True, False), dtype=np.bool_))
        assert np.all(np.isnan(constructed[1])) and np.isnan(coordinates[1])
        axis = int(np.argmax(np.abs(points[1] - points[0])))
        assert coordinates[0] == constructed[0, axis]
        source_delta = [
            Fraction(float(points[1, k])) - Fraction(float(points[0, k]))
            for k in range(3)
        ]
        target_delta = [
            Fraction(float(constructed[0, k])) - Fraction(float(points[0, k]))
            for k in range(3)
        ]
        for first, second in combinations(range(3), 2):
            assert (
                target_delta[first] * source_delta[second]
                == target_delta[second] * source_delta[first]
            )
        assert (
            min(points[0, axis], points[1, axis])
            < constructed[0, axis]
            < max(points[0, axis], points[1, axis])
        )
        reversed_points, reversed_coordinates, reversed_mask = (
            native.construct_curve_points(
                np.asarray((inserted, 0), dtype=np.int32),
                np.asarray(((1, 0), (2, 1)), dtype=np.int32),
                np.asarray(
                    (points[0] + 0.625 * (points[1] - points[0]), points[0]),
                    dtype=np.float64,
                ),
                work_limit=10000,
            )
        )
        np.testing.assert_array_equal(reversed_mask, mask)
        np.testing.assert_array_equal(reversed_points, constructed)
        np.testing.assert_array_equal(reversed_coordinates, coordinates)
        np.testing.assert_array_equal(native.arrays().points, before.points)
        np.testing.assert_array_equal(native.arrays().tetrahedra, before.tetrahedra)
        np.testing.assert_array_equal(
            native.arrays().segment_sources, before.segment_sources
        )
    finally:
        native.close()


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_metric_epoch_checks_original_host_deadline_before_preparation() -> None:
    from phydrax.meshing._tetra_metric import (
        _constraints,
        execute_native_tetra_metric,
        TetraMetricSource,
    )

    mesh = _box()
    points = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices)
    regions = np.zeros((cells.shape[0],), dtype=np.int32)
    faces, face_ids, segments, segment_ids = _constraints(
        mesh,
        cells,
        regions,
        None,
        None,
        np.zeros((mesh.entity_set(1).count,), dtype=np.bool_),
    )
    native = TetMesh3D(
        points,
        cells,
        regions,
        faces,
        face_ids,
        segments,
        segment_ids,
        boundary_policy="fixed",
    )
    source = TetraMetricSource(
        points,
        cells,
        np.arange(points.shape[0], dtype=np.int64),
        np.arange(cells.shape[0], dtype=np.int64),
    )
    before = native.arrays()
    work = native.work_units()
    try:
        with pytest.raises(MeshingFailure) as caught:
            execute_native_tetra_metric(
                native,
                source,
                np.broadcast_to(np.eye(3, dtype=np.float64), (points.shape[0], 3, 3)),
                fixed_vertices=np.zeros((points.shape[0],), dtype=np.bool_),
                protected_edges=set(),
                maximum_passes=16,
                topology_operations=True,
                relocation=True,
                minimum_metric_quality=0.05,
                lower_metric_length=0.0,
                upper_metric_length=np.sqrt(2.0),
                maximum_vertices=4096,
                maximum_cells=4096,
                maximum_operations=100000,
                maximum_work_units=1000000,
                maximum_location_pairs=1000000,
                maximum_cavity_cells=512,
                maximum_cavity_work=1000000,
                operation_started=monotonic() - 2.0,
                maximum_wall_seconds=1.0,
            )
        assert caught.value.evidence.category is MeshingFailureCategory.TIMED_OUT
        assert caught.value.evidence.requested == (("maximum_wall_seconds", 1.0),)
        assert dict(caught.value.evidence.achieved)["elapsed_seconds"] > 1.0
        np.testing.assert_array_equal(native.arrays().points, before.points)
        np.testing.assert_array_equal(native.arrays().tetrahedra, before.tetrahedra)
        assert native.work_units() == work
    finally:
        native.close()

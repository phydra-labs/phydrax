#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""White-box contracts for tetrahedral size trial kernels and work admission."""

from itertools import combinations, permutations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, jvp, vjp

from phydrax.meshing._tetra_metric import (
    _append_global_size_joint_rows,
    _GlobalSizeJointRows,
    append_global_size_branches,
    global_size_branch_gradients,
    global_size_branch_targets,
    global_size_canonical_residual_bound,
    global_size_residual,
    global_size_shape_residual,
    GlobalSizeArgs,
    GlobalSizeBranchBank,
    GlobalSizeTrialPolicy,
    size_quantile,
)
from phydrax.optim._least_squares import _prepare_residual_model
from tests._support.metric_meshes import (
    TETRA_RADIUS_DIFFERENCE_STEP,
    TETRA_RADIUS_DIFFERENCE_TOLERANCE,
    TETRA_TRIAL_IMAGE_TOLERANCE,
)


def _trial_policy(
    parameters: Array, direction: Array, residual: Array, args: GlobalSizeArgs
) -> tuple[Array, Array, Array, Array]:
    model = _prepare_residual_model(
        lambda value: global_size_residual(value, args), parameters
    )
    work_limit = jnp.asarray(1_000_000_000, dtype=jnp.int64)
    result = GlobalSizeTrialPolicy()(
        parameters, direction, residual, model.jacobian, work_limit, args
    )
    assert bool(result.receipts_valid())
    assert int(result.jvp_actions) == int(result.vjp_actions) == 1
    assert 0 <= int(result.model_visits) <= 11
    assert 0 <= int(result.model_work_units) <= int(work_limit)
    assert not bool(result.resource_refused)
    return result.direction, result.scale, result.model_image, result.valid


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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args),
        (parameters,),
        (direction,),
    )
    # The old fixed-identity support would remain at edge (0, 4), with
    # residual one and zero derivative in the crossing edge's direction.
    np.testing.assert_array_equal(residual[:1], (crossing_length - 1.0,))
    np.testing.assert_array_equal(derivative[:1], (1.0,))
    _, pullback = vjp(lambda value: global_size_residual(value, args), parameters)
    seed = jnp.zeros_like(residual).at[0].set(1.0)
    reverse = pullback(seed)[0]
    np.testing.assert_array_equal(reverse, direction.at[0, 0].set(-1.0))
    reordered = args._replace(edges=args.edges[jnp.asarray((1, 0, 2))])
    np.testing.assert_array_equal(
        global_size_residual(parameters, reordered)[:1],
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
    args = GlobalSizeArgs(
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
    residual = global_size_residual(jnp.zeros_like(origins), args)
    expected = np.quantile(np.asarray((lower, upper)), quantile) - 1.0
    np.testing.assert_array_equal(residual[:1], (expected,))
    assert lower != 1.0 and upper != 1.0


@pytest.mark.parametrize("edge_velocities", ((-1.0, -2.0, 1.0), (-1.0, -2.0, 0.0)))
@pytest.mark.parametrize("quantile", (0.5, 0.95, 1.0))
def test_statistical_repeated_tie_uses_outgoing_rank_for_finite_movement(
    quantile: float,
    edge_velocities: tuple[float, float, float],
) -> None:
    """Reselect tied support rows using the outgoing statistic."""
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
    args = GlobalSizeArgs(
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
    baseline = global_size_residual(parameters, args)
    for order in permutations(range(3)):
        reordered = args._replace(edges=args.edges[jnp.asarray((*order, 3))])
        residual = global_size_residual(parameters, reordered)
        np.testing.assert_array_equal(residual, baseline)
        assert bool(jnp.all(jnp.isfinite(residual)))
    # Reversing movement must reselect the outgoing ranks as well; it
    # cannot reuse the branch selected for the positive displacement.
    reverse = global_size_residual(-parameters, args)[0]
    initial = global_size_residual(jnp.zeros_like(origins), args)[0]
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
        lambda value: size_quantile(value, quantiles), (lengths,), (direction,)
    )
    np.testing.assert_array_equal(primal, (2.0, 2.0))
    np.testing.assert_array_equal(image, (-1.0, -1.0))
    _, pullback = vjp(lambda value: size_quantile(value, quantiles), lengths)
    gradient = pullback(jnp.asarray((1.0, 0.0)))[0]
    np.testing.assert_allclose(
        gradient, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0), rtol=0.0, atol=0.0
    )
    # The outgoing p95 derivative is -0.1, not the linear model's -1.
    # This distinction is essential to native trial merit prediction.
    step = 0.125
    actual = size_quantile(lengths + step * direction, quantiles)
    expected = np.quantile(
        np.asarray((2.0, 2.0, 2.0)) + step * np.asarray(direction[:3]),
        np.asarray(quantiles),
    )
    np.testing.assert_array_equal(actual, expected)
    assert float(actual[1]) != float(primal[1] + step * image[1])
    order = jnp.asarray((2, 0, 1, 3))
    reordered, reordered_image = jvp(
        lambda value: size_quantile(value, quantiles),
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
    args = GlobalSizeArgs(
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
    origin = global_size_residual(jnp.zeros_like(origins), args)
    np.testing.assert_array_equal(origin[:2], (0.0, 0.0))
    parameters = jnp.zeros_like(origins).at[4, 0].set(sign * 0.125)
    residual = global_size_residual(parameters, args)
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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, scale, outgoing, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    actual = global_size_residual(parameters + scale * direction, args)
    assert 0.0 < float(scale) <= 1.0
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    np.testing.assert_allclose(
        image[:2], (-1.0, -1.0), rtol=0.0, atol=TETRA_TRIAL_IMAGE_TOLERANCE
    )
    # The rank-frontier policy may replace the authored direction; its
    # published model image belongs to that selected finite trajectory.
    np.testing.assert_allclose(
        outgoing, (actual - residual) / scale, rtol=0.0, atol=TETRA_TRIAL_IMAGE_TOLERANCE
    )
    assert float(jnp.sum(residual * outgoing)) < 0.0
    assert not bool(GlobalSizeTrialPolicy().termination_valid(residual, args))
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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, _, outgoing, valid = _trial_policy(parameters, direction, residual, args)
    # Satisfied size constraints leave an increasing active shape merit,
    # which requires refusal of this trial rather than a feasibility claim.
    assert not bool(valid)
    radius_row = args.quantiles.size + args.cells.shape[0]
    assert float(residual[radius_row]) > 0.0
    step = TETRA_RADIUS_DIFFERENCE_STEP
    actual = global_size_residual(parameters + step * direction, args)
    np.testing.assert_allclose(
        outgoing[radius_row],
        (actual[radius_row] - residual[radius_row]) / step,
        rtol=TETRA_RADIUS_DIFFERENCE_TOLERANCE,
        atol=TETRA_RADIUS_DIFFERENCE_TOLERANCE,
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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), (parameters,), (direction,)
    )
    # The adjacent-to-unit edge meets the fixed unit block at this authored
    # rank event. That represented movement cannot change either residual.
    event_scale = np.spacing(1.0) / 0.01
    clipped = global_size_residual(parameters + event_scale * direction, args)
    np.testing.assert_array_equal(clipped, residual)
    direction, scale, finite_image, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    actual = global_size_residual(parameters + scale * direction, args)
    # The collinear affine model admits finite progress without disturbing
    # the already satisfied median or adding any scientific tolerance.
    np.testing.assert_array_equal(scale * finite_image, actual - residual)
    assert float(actual[0]) == 0.0
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    model = _prepare_residual_model(
        lambda value: global_size_residual(value, args), parameters
    )
    original_direction = parameters.at[8, 0].set(-0.01).at[10, 0].set(-0.001)
    receipt = GlobalSizeTrialPolicy()(
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
    refused = GlobalSizeTrialPolicy()(
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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), (parameters,), (direction,)
    )
    direction, scale, finite_image, valid = _trial_policy(
        parameters, direction, residual, args
    )
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    actual = global_size_residual(parameters + scale * direction, args)
    np.testing.assert_array_equal(scale * finite_image, actual - residual)
    # Finite movement changes the support, rather than only rounding.
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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), (parameters,), (direction,)
    )
    full = global_size_residual(direction, args)
    direction, scale, secant, valid = _trial_policy(parameters, direction, residual, args)
    actual = global_size_residual(scale * direction, args)
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    assert bool(jnp.any(direction != original_direction))
    np.testing.assert_array_equal(direction[:4], jnp.zeros((4, 3)))
    assert float(jnp.sum(full * full)) > float(jnp.sum(residual * residual))
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    # A finite secant and its reconstructed residual can round differently
    # through subtract, divide, multiply, and reconstruction operations.
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
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), (parameters,), (direction,)
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
    actual = global_size_residual(parameters + scale * direction, args)
    assert float(actual[radius_row]) > 0.0
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))


def test_statistical_rank_model_zero_work_budget_refuses_before_actions() -> None:
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    args = GlobalSizeArgs(
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
        lambda value: global_size_residual(value, args), parameters
    )
    result = GlobalSizeTrialPolicy()(
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
    bank = GlobalSizeBranchBank(empty, empty, empty, jnp.asarray(0), jnp.asarray(False))
    bank, work = append_global_size_branches(
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
    bank, work = append_global_size_branches(
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
    args = GlobalSizeArgs(
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
    residual = global_size_residual(jnp.zeros_like(origins), args)
    delta = origins[edges[:, 1]] - origins[edges[:, 0]]
    lengths = jnp.sqrt(
        jnp.where(args.edge_valid > 0.0, jnp.sum(delta * delta, axis=-1), 1.0)
    )
    bank = GlobalSizeBranchBank(
        jnp.asarray((0, 1, 1)),
        jnp.asarray((2, 3, 4)),
        jnp.asarray((2, 3, 4)),
        jnp.asarray(3),
        jnp.asarray(False),
    )
    gradients, active, equality = global_size_branch_gradients(
        bank,
        delta,
        lengths,
        residual,
        jnp.zeros_like(origins),
        jnp.zeros((1, *origins.shape), dtype=origins.dtype),
        args,
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
    gradients, active, equality = global_size_branch_gradients(
        bank,
        delta,
        lengths,
        residual.at[0].set(0.0),
        jnp.zeros_like(origins),
        jnp.zeros((1, *origins.shape), dtype=origins.dtype),
        args,
    )
    np.testing.assert_array_equal(gradients[0], expected[0])
    np.testing.assert_array_equal(active, (False, True, True))
    np.testing.assert_array_equal(equality, (True, False, False))


def test_statistical_branch_targets_preserve_component_gaps_and_padding() -> None:
    statistics = jnp.asarray((1, -1, 0, 0, 1, -1, 0, 1), dtype=jnp.int32)
    empty = jnp.zeros((8,), dtype=jnp.int32)
    bank = GlobalSizeBranchBank(
        statistics, empty, empty, jnp.asarray(3), jnp.asarray(False)
    )
    inequality = jnp.asarray((True, True, True, False, False, False, False, False))
    residual = jnp.asarray((-1.0e-6, 0.18, 3.0, 4.0))
    targets = global_size_branch_targets(bank, residual, inequality, 2)
    np.testing.assert_array_equal(targets, (0.18, 25.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0))
    equality_residual = residual.at[0].set(0.0)
    targets = global_size_branch_targets(
        bank, equality_residual, inequality.at[2].set(False), 2
    )
    np.testing.assert_array_equal(targets, (0.18, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_statistical_primal_bound_respects_evaluation_guard() -> None:
    # The 32-step/four-trial bound permits 162 primals. The 128-evaluation
    # guard permits a final full step and final model beyond that guard;
    # finite policy models have separate accounting.
    assert global_size_canonical_residual_bound() == 128 + 4 + 1
    assert global_size_canonical_residual_bound() >= 1 + 32 * 2 + 1


def test_statistical_finite_model_does_not_trade_hard_dihedral_for_size_descent() -> None:
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 1.5))
    )
    args = GlobalSizeArgs(
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
    residual = global_size_residual(parameters, args)
    full = global_size_shape_residual(origins + direction, args)
    assert bool(jnp.any(full[1:] > 0.0))
    selected, scale, _, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    assert float(scale) < 1.0
    actual = global_size_shape_residual(origins + scale * selected, args)
    np.testing.assert_array_equal(actual[1:], 0.0)


def test_statistical_refused_full_point_recovers_with_distinct_finite_scale() -> None:
    height = 1.5 * np.tan(np.deg2rad(10.01)) / np.sqrt(2.0)
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, height))
    )
    args = GlobalSizeArgs(
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
    residual = global_size_residual(parameters, args)
    assert bool(jnp.all(residual[2:] == 0.0))
    full = global_size_residual(direction, args)
    assert bool(jnp.all(jnp.abs(full[:2]) < jnp.abs(residual[:2])))
    assert bool(jnp.any(full[4:] > 0.0))
    selected, scale, image, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    assert float(scale) == 0.5
    actual = global_size_residual(scale * selected, args)
    np.testing.assert_array_equal(actual[4:], 0.0)
    np.testing.assert_array_equal(scale * image, actual - residual)


@pytest.mark.parametrize("quantile,unit_edges", ((0.5, 1), (0.95, 19)))
def test_statistical_future_fixed_rank_does_not_block_exact_finite_descent(
    quantile: float, unit_edges: int
) -> None:
    """A newly governing fixed unit edge needs no speculative strict progress."""
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            *((1.0, 0.0, 0.0),) * unit_edges,
            (1.2, 0.0, 0.0),
            (1.4, 0.0, 0.0),
        )
    )
    moving = origins.shape[0] - 2
    edges = jnp.asarray((*((0, vertex) for vertex in range(4, origins.shape[0])), (0, 0)))
    args = GlobalSizeArgs(
        origins,
        jnp.zeros((origins.shape[0], 3, 3)).at[moving].set(jnp.eye(3)),
        edges,
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((quantile,)),
        jnp.asarray((1.0,) * (unit_edges + 2) + (0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[moving, 0].set(-0.5)
    residual = global_size_residual(parameters, args)
    assert float(residual[0]) > 0.0
    selected, scale, image, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    np.testing.assert_array_equal(selected, direction)
    assert float(scale) == 1.0
    actual = global_size_residual(parameters + scale * selected, args)
    np.testing.assert_array_equal(actual, 0.0)
    np.testing.assert_array_equal(scale * image, actual - residual)
    assert bool(GlobalSizeTrialPolicy().termination_valid(actual, args))
    reverse = jnp.arange(edges.shape[0] - 1, -1, -1)
    reordered = args._replace(edges=edges[reverse], edge_valid=args.edge_valid[reverse])
    repeated = _trial_policy(parameters, direction, residual, reordered)
    np.testing.assert_array_equal(repeated[0], selected)
    np.testing.assert_array_equal(repeated[1], scale)
    np.testing.assert_array_equal(repeated[2], image)
    assert bool(repeated[3])


def test_statistical_finite_scale_brackets_tiny_positive_radius_margin() -> None:
    """A legal near-boundary cell needs a shape event, not another rank trial."""
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 1.5))
    )
    args = GlobalSizeArgs(
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
        jnp.asarray(1.0),
        jnp.asarray(np.sqrt(3.0) / 2.0 + 1.0e-6),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[1, 0].set(-0.01).at[2, 1].set(-0.01).at[3, 2].set(-0.5)
    residual = global_size_residual(parameters, args)
    np.testing.assert_array_equal(residual[2:], 0.0)
    full = global_size_residual(direction, args)
    half = global_size_residual(0.5 * direction, args)
    radius_row = args.quantiles.size + args.cells.shape[0]
    assert float(full[radius_row]) > 0.0
    assert float(half[radius_row]) > 0.0
    selected, scale, image, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    assert 0.0 < float(scale) <= 1.0
    if float(scale) >= 0.5:
        assert bool(jnp.any(selected != direction))
    actual = global_size_residual(parameters + scale * selected, args)
    np.testing.assert_array_equal(actual[2:], 0.0)
    assert bool(jnp.all(jnp.abs(actual[:2]) < jnp.abs(residual[:2])))
    np.testing.assert_allclose(
        scale * image, actual - residual, rtol=0.0, atol=TETRA_TRIAL_IMAGE_TOLERANCE
    )


def test_statistical_joint_merit_admits_failed_median_tradeoff_across_unit_ties() -> None:
    """Upper-rank unit ties are not artificial lower-rank Pareto constraints."""
    near_unit = 1.0 - 1.5e-8
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            *((0.5, 0.0, 0.0),) * 49,
            (near_unit, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            *((1.0, 0.0, 0.0),) * 32,
            *((-0.2, 0.0, 0.0),) * 17,
        )
    )
    moving = 54
    edges = jnp.asarray(
        (
            *((0, vertex) for vertex in range(4, 87)),
            *((moving, vertex) for vertex in range(87, 104)),
        )
    )
    basis = jnp.zeros((origins.shape[0], 3, 3)).at[moving, 0, 0].set(1.0)
    args = GlobalSizeArgs(
        origins,
        basis,
        edges,
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5, 0.95)),
        jnp.ones((100,)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    parameters = jnp.zeros_like(origins)
    direction = parameters.at[moving, 0].set(-1.0e-3)
    residual = global_size_residual(parameters, args)
    assert float(residual[0]) < 0.0
    assert float(residual[1]) > 0.0
    selected, scale, image, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    actual = global_size_residual(parameters + scale * selected, args)
    assert abs(float(actual[0])) > abs(float(residual[0]))
    assert float(jnp.sum(jnp.abs(actual[:2]))) < float(jnp.sum(jnp.abs(residual[:2])))
    assert float(jnp.sum(actual * actual)) < float(jnp.sum(residual * residual))
    np.testing.assert_array_equal(actual[2:], 0.0)
    np.testing.assert_allclose(
        scale * image, actual - residual, rtol=0.0, atol=TETRA_TRIAL_IMAGE_TOLERANCE
    )


@pytest.mark.parametrize("radius_bound", (1000.0, 0.5))
def test_statistical_third_trial_restores_near_active_dihedral_with_raw_tangent(
    radius_bound: float,
) -> None:
    height = 1.5 * np.tan(np.deg2rad(10.000000000000016)) / np.sqrt(2.0)
    origins = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, height))
    )
    args = GlobalSizeArgs(
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
        jnp.asarray(radius_bound),
    )
    parameters = jnp.zeros_like(origins)
    direction = (
        parameters.at[1, 0].set(-1.0e-5).at[2, 1].set(-1.0e-5).at[3, 2].set(-3.0e-4)
    )
    residual = global_size_residual(parameters, args)
    if radius_bound == 1000.0:
        np.testing.assert_array_equal(residual[2:], 0.0)
    else:
        assert float(residual[3]) > 0.0
        np.testing.assert_array_equal(residual[4:], 0.0)
    full = global_size_residual(direction, args)
    assert bool(jnp.any(full[2:] > 0.0))
    selected, scale, image, valid = _trial_policy(parameters, direction, residual, args)
    assert bool(valid)
    assert bool(jnp.any(selected != direction))
    actual = global_size_residual(parameters + scale * selected, args)
    assert bool(jnp.all(actual[2:] <= residual[2:]))
    if radius_bound == 1000.0:
        np.testing.assert_array_equal(actual[2:], 0.0)
    else:
        assert not bool(GlobalSizeTrialPolicy().termination_valid(actual, args))
    assert float(jnp.sum(jnp.abs(actual[:2]))) < float(jnp.sum(jnp.abs(residual[:2])))
    np.testing.assert_allclose(
        scale * image, actual - residual, rtol=0.0, atol=TETRA_TRIAL_IMAGE_TOLERANCE
    )


def test_statistical_joint_frontier_retains_support_history_and_refuses_overflow() -> (
    None
):
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
        )
    )
    args = GlobalSizeArgs(
        origins,
        jnp.broadcast_to(jnp.eye(3), (5, 3, 3)),
        jnp.asarray(((0, 1), (0, 4))),
        jnp.asarray(((0, 1, 2, 3),)),
        jnp.asarray((0.5,)),
        jnp.ones((2,)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(10.0),
    )
    residual = global_size_residual(jnp.zeros_like(origins), args)
    delta = origins[args.edges[:, 1]] - origins[args.edges[:, 0]]
    lengths = jnp.sqrt(jnp.sum(delta * delta, axis=1))
    bank = _GlobalSizeJointRows(
        jnp.full((2, 2), -1, dtype=jnp.int32),
        jnp.zeros((2, *origins.shape), dtype=origins.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )
    limit = jnp.asarray(1_000_000, dtype=jnp.int64)
    bank, slots, work, resource = _append_global_size_joint_rows(
        bank,
        jnp.asarray(((0,), (1,))),
        jnp.asarray(((0,), (1,))),
        jnp.asarray((True, False)),
        delta,
        lengths,
        residual,
        limit,
        args,
    )
    first = bank.gradients[0]
    assert int(bank.count) == 1 and not bool(resource)
    bank, slots, repeated_work, resource = _append_global_size_joint_rows(
        bank,
        jnp.asarray(((1,), (0,))),
        jnp.asarray(((1,), (0,))),
        jnp.asarray((True, True)),
        delta,
        lengths,
        residual,
        limit - work,
        args,
    )
    assert int(bank.count) == 2 and not bool(bank.overflow) and not bool(resource)
    np.testing.assert_array_equal(slots, (1, 0))
    np.testing.assert_array_equal(bank.keys, ((0, 0), (1, 1)))
    np.testing.assert_array_equal(bank.gradients[0], first)
    assert bool(jnp.any(bank.gradients[1] != first))
    retained = bank.gradients
    bank, _, _, resource = _append_global_size_joint_rows(
        bank,
        jnp.asarray(((0,), (0,))),
        jnp.asarray(((1,), (1,))),
        jnp.asarray((True, True)),
        delta,
        lengths,
        residual,
        limit - work - repeated_work,
        args,
    )
    assert bool(bank.overflow) and int(bank.count) == 2 and not bool(resource)
    np.testing.assert_array_equal(bank.gradients, retained)

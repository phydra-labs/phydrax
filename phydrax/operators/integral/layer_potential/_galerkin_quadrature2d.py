#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Weak panel-pair quadrature for the straight-panel 2-D Laplace Galerkin operator.

Panel pairs are classified as coincident, shared-endpoint, near, or regular.
Coincident pairs use closed forms. Shared-endpoint pairs use the Duffy map about
the shared vertex, which extracts ``ρ log ρ`` analytically and leaves smooth
angular integrands for adaptive Gauss--Kronrod quadrature. Near pairs integrate
the exact straight-panel inner integrals with adaptive Gauss--Kronrod over the
test panel. Regular pairs use a fixed tensor Gauss--Legendre rule at run time;
every regular pair is certified at preparation against the exact-inner Kronrod
reference, and a pair whose estimate exceeds the tolerance is promoted to the
near class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

import phydrax.ein as ein

from ...._numerics import gauss_kronrod_data, gauss_legendre_data
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


PanelLayerKind2D: TypeAlias = Literal["single", "double"]

type _Floats = npt.NDArray[np.float64]
type _Integers = npt.NDArray[np.int64]
type _Integrand = Callable[[_Integers, _Floats], _Floats]

_TWO_PI = 2.0 * np.pi
_KRONROD_ORDER = 21
_FLOAT_BYTES = np.dtype(np.float64).itemsize
EXCEPTION_ENTRY_BYTES = 8 + 4 + 4 + 4 + 8 + 16

PAIR_CLASS_NAMES_2D = ("coincident", "shared-endpoint", "near", "regular")
PAIR_CLASS_METHODS_2D = (
    "closed-form-straight-panel",
    "duffy-analytic-log-moment-adaptive-gauss-kronrod-21",
    "exact-inner-adaptive-gauss-kronrod-21-outer",
    "tensor-gauss-legendre-certified-by-exact-inner-kronrod-21",
)


class PanelGeometry2D(NamedTuple):
    """Host straight-panel geometry in declared panel order."""

    starts: _Floats
    tangents: _Floats
    normals: _Floats
    lengths: _Floats


class _AdaptiveIntegrals(NamedTuple):
    values: _Floats
    errors: _Floats
    evaluations: _Integers
    depth: int
    workspace_bytes: int


class _PairValues(NamedTuple):
    single: _Floats
    double: _Floats
    single_errors: _Floats
    double_errors: _Floats
    evaluations: int
    depth: int
    workspace_bytes: int


class _RegularSweep(NamedTuple):
    near: _Integers
    promoted: _Integers
    candidate_count: int
    maximum_error: float
    evaluations: int
    workspace_bytes: int


class PanelPairData2D(StrictModule, NonTrainableState):
    """Sparse non-regular pair values and preparation evidence."""

    exception_keys: Array
    targets: Array
    sources: Array
    classes: Array
    single_values: Array
    double_values: Array
    maximum_errors: Array
    evaluations: Array
    pair_counts: tuple[int, int, int, int] = eqx.field(static=True)
    promoted_count: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)


def unit_gauss_rule(order: int, /) -> tuple[_Floats, _Floats]:
    """Gauss--Legendre nodes and weights on ``[0, 1]``."""
    rule = gauss_legendre_data(order)
    nodes = np.asarray(rule.nodes, dtype=np.float64)
    weights = np.asarray(rule.weights, dtype=np.float64)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _unit_kronrod_rule() -> tuple[_Floats, _Floats, _Floats]:
    rule = gauss_kronrod_data(_KRONROD_ORDER)
    if rule.embedded_weights is None:
        raise RuntimeError("The Gauss--Kronrod rule has no embedded Gauss weights.")
    nodes = np.asarray(rule.nodes, dtype=np.float64)
    return (
        0.5 * (nodes + 1.0),
        0.5 * np.asarray(rule.weights, dtype=np.float64),
        0.5 * np.asarray(rule.embedded_weights, dtype=np.float64),
    )


def segment_layer_moments(
    targets: Array,
    starts: Array,
    tangents: Array,
    normals: Array,
    lengths: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Exact straight-panel single-layer and P1 double-layer integrals.

    Returns ``∫ G(x, y) ds_y`` and ``∫ ∂G/∂n_y(x, y) λ(y) ds_y`` for the start
    and end hat functions ``λ`` of each panel, with ``G = -log|x - y|/(2π)`` and
    the panel's outward normal. All arguments broadcast; targets on the line of
    a panel receive the principal-value double layer, which is zero there.
    """
    relative = targets - starts
    along = jnp.sum(relative * tangents, axis=-1)
    across = jnp.sum(relative * normals, axis=-1)
    distance = jnp.abs(across)
    before = -along
    after = lengths - along
    before_squared = before * before + across * across
    after_squared = after * after + across * across

    def primitive(offset: Array, squared: Array) -> Array:
        positive = squared > 0.0
        logarithm = jnp.log(jnp.where(positive, squared, 1.0))
        return (
            jnp.where(positive, 0.5 * offset * logarithm, 0.0)
            - offset
            + distance * jnp.arctan2(offset, distance)
        )

    single = (primitive(before, before_squared) - primitive(after, after_squared)) / (
        _TWO_PI
    )
    angle = jnp.arctan2(
        lengths * across, across * across + along * along - along * lengths
    )
    log_ratio = 0.5 * (
        jnp.log(jnp.where(after_squared > 0.0, after_squared, 1.0))
        - jnp.log(jnp.where(before_squared > 0.0, before_squared, 1.0))
    )
    first_moment = along * angle + jnp.where(across != 0.0, across * log_ratio, 0.0)
    end = first_moment / (_TWO_PI * lengths)
    start = angle / _TWO_PI - end
    return single, start, end


def regular_pair_values(
    target_points: Array,
    target_weights: Array,
    source_points: Array,
    source_weights: Array,
    source_normals: Array,
    basis: Array,
    active: Array,
    layer: PanelLayerKind2D,
    /,
) -> Array:
    """Tensor Gauss--Legendre pair integrals ``(target, source, basis)``.

    Inactive pairs (exceptions or padding) are replaced by a harmless unit
    separation before the kernel is evaluated and contribute exactly zero.
    """
    differences = target_points[:, :, None, None, :] - source_points[None, None, :, :, :]
    differences = jnp.where(active[:, None, :, None, None], differences, 1.0)
    squared = jnp.sum(differences * differences, axis=-1)
    match layer:
        case "single":
            kernel = -jnp.log(squared) / (2.0 * _TWO_PI)
        case "double":
            kernel = jnp.sum(
                differences * source_normals[None, None, :, None, :], axis=-1
            ) / (_TWO_PI * squared)
        case _:
            assert_never(layer)
    values = ein.contract(
        "ta,sb,tasb,bk->tsk",
        target_weights,
        source_weights,
        kernel,
        basis,
        backend="jax",
    )
    return jnp.where(active[:, :, None], values, 0.0)


def _point_segment_distance(points: _Floats, starts: _Floats, ends: _Floats) -> _Floats:
    direction = ends - starts
    squared = np.sum(direction * direction, axis=-1)
    parameter = np.clip(
        np.sum((points - starts) * direction, axis=-1) / squared, 0.0, 1.0
    )
    closest = starts + parameter[..., None] * direction
    return np.linalg.norm(points - closest, axis=-1)


def _cross(left: _Floats, right: _Floats) -> _Floats:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def segment_distances(
    first_starts: _Floats,
    first_ends: _Floats,
    second_starts: _Floats,
    second_ends: _Floats,
    /,
) -> _Floats:
    """Exact Euclidean distance between broadcast straight segments."""
    endpoint = np.minimum(
        np.minimum(
            _point_segment_distance(first_starts, second_starts, second_ends),
            _point_segment_distance(first_ends, second_starts, second_ends),
        ),
        np.minimum(
            _point_segment_distance(second_starts, first_starts, first_ends),
            _point_segment_distance(second_ends, first_starts, first_ends),
        ),
    )
    first_direction = first_ends - first_starts
    second_direction = second_ends - second_starts
    crossing = (
        _cross(first_direction, second_starts - first_starts)
        * _cross(first_direction, second_ends - first_starts)
        < 0.0
    ) & (
        _cross(second_direction, first_starts - second_starts)
        * _cross(second_direction, first_ends - second_starts)
        < 0.0
    )
    return np.where(crossing, 0.0, endpoint)


def _interval_evaluation_bytes(components: int, /) -> int:
    """Scratch of one interval's integrand evaluation.

    Twelve live ``(node, component)`` arrays: the samples, the integrand's
    geometric temporaries, and the Kronrod and Gauss rule products.
    """
    return _FLOAT_BYTES * _KRONROD_ORDER * components * 12


def _interval_state_bytes(components: int, /) -> int:
    """Bookkeeping of one active interval: owner, bounds, rule values, and error."""
    return _FLOAT_BYTES * (3 + 3 * components)


def adaptive_unit_integrals(
    integrand: _Integrand,
    tolerances: _Floats,
    /,
    *,
    max_depth: int,
    max_intervals: int,
    max_workspace_bytes: int,
    label: str,
) -> _AdaptiveIntegrals:
    """Vector adaptive Gauss--Kronrod integration of many owners on ``[0, 1]``.

    Each owner component must satisfy its absolute tolerance. An interval of
    width ``h`` is accepted when every Kronrod--Gauss difference is at most
    ``h`` times the owner tolerance, so the accepted differences sum to at most
    the tolerance. Non-finite samples are never accepted.

    Every level is accounted before it is allocated: its active-interval
    bookkeeping plus one chunk of integrand evaluations must fit
    ``max_workspace_bytes``, and the integrand is evaluated chunk by chunk.
    A level whose bookkeeping and a single interval's evaluation do not fit is
    refused. The returned ``workspace_bytes`` is the accounted peak.
    """
    nodes, kronrod, gauss = _unit_kronrod_rule()
    count, components = tolerances.shape
    evaluation = _interval_evaluation_bytes(components)
    state = _interval_state_bytes(components)

    def admit(active: int, /) -> tuple[int, int]:
        """Chunk size and accounted bytes of a level, refused before allocation."""
        if active > max_intervals:
            raise ValueError(
                f"[quadrature-{label}] Adaptive intervals exceed max_adaptive_intervals."
            )
        chunk = min(active, (max_workspace_bytes - active * state) // evaluation)
        if chunk < 1:
            raise ValueError(
                f"[preparation-bytes] Adaptive {label} quadrature of {active} "
                "intervals exceeds max_preparation_workspace_bytes."
            )
        return chunk, active * state + chunk * evaluation

    values = np.zeros((count, components), dtype=np.float64)
    errors = np.zeros((count, components), dtype=np.float64)
    evaluations = np.zeros((count,), dtype=np.int64)
    chunk, peak = admit(count) if count else (0, 0)
    owners = np.arange(count, dtype=np.int64)
    lower = np.zeros((count,), dtype=np.float64)
    width = np.ones((count,), dtype=np.float64)
    depth = 0
    while owners.size:
        kronrod_value = np.empty((owners.size, components), dtype=np.float64)
        gauss_value = np.empty((owners.size, components), dtype=np.float64)
        for start in range(0, owners.size, chunk):
            part = slice(start, start + chunk)
            samples = integrand(
                owners[part], lower[part, None] + width[part, None] * nodes[None, :]
            )
            kronrod_value[part] = width[part, None] * ein.contract(
                "n,mnk->mk", kronrod, samples
            )
            gauss_value[part] = width[part, None] * ein.contract(
                "n,mnk->mk", gauss, samples
            )
        error = np.abs(kronrod_value - gauss_value)
        accepted = np.all(error <= tolerances[owners] * width[:, None], axis=1)
        np.add.at(values, owners[accepted], kronrod_value[accepted])
        np.add.at(errors, owners[accepted], error[accepted])
        np.add.at(evaluations, owners, nodes.size)
        refine = ~accepted
        depth += 1
        if not np.any(refine):
            break
        if depth > max_depth:
            raise ValueError(
                f"[quadrature-{label}] Adaptive quadrature exhausted max_depth "
                "before meeting its tolerance."
            )
        chunk, level_bytes = admit(2 * int(np.count_nonzero(refine)))
        peak = max(peak, level_bytes)
        half = 0.5 * width[refine]
        owners = np.repeat(owners[refine], 2)
        lower = np.stack((lower[refine], lower[refine] + half), axis=1).reshape((-1,))
        width = np.repeat(half, 2)
    return _AdaptiveIntegrals(values, errors, evaluations, depth, peak)


def _shared_endpoint_frames(
    geometry: PanelGeometry2D,
    targets: _Integers,
    sources: _Integers,
    /,
) -> tuple[_Floats, _Floats, _Integers]:
    """Directions from each shared vertex and the shared source-hat slot."""
    count = geometry.lengths.shape[0]
    forward = sources == (targets + 1) % count
    target_direction = np.where(
        forward[:, None], -geometry.tangents[targets], geometry.tangents[targets]
    )
    source_direction = np.where(
        forward[:, None], geometry.tangents[sources], -geometry.tangents[sources]
    )
    shared_slot = np.where(forward, 0, 1).astype(np.int64)
    return target_direction, source_direction, shared_slot


def shared_endpoint_values(
    geometry: PanelGeometry2D,
    targets: _Integers,
    sources: _Integers,
    /,
    *,
    tolerance: float,
    max_depth: int,
    max_intervals: int,
    max_workspace_bytes: int,
) -> _PairValues:
    """Duffy-transformed shared-endpoint weak V and K entries.

    With both panels parametrized from their shared vertex and the square split
    along its diagonal, the Duffy map ``(ρ, w)`` factors ``|x - y| = ρ R(w)``. The
    ``ρ log ρ`` part of V and the ``ρ`` cancellation of K are integrated exactly;
    the smooth ``w`` remainders use adaptive Gauss--Kronrod quadrature.
    """
    target_direction, source_direction, shared_slot = _shared_endpoint_frames(
        geometry, targets, sources
    )
    first = geometry.lengths[targets]
    second = geometry.lengths[sources]
    cosine = np.sum(target_direction * source_direction, axis=1)
    sine = np.sum(target_direction * geometry.normals[sources], axis=1)

    def integrand(owners: _Integers, points: _Floats) -> _Floats:
        a = first[owners][:, None]
        b = second[owners][:, None]
        cross = 2.0 * a * b * cosine[owners][:, None] * points
        left = a * a - cross + b * b * points * points
        right = a * a * points * points - cross + b * b
        product = a * b
        return np.stack(
            (
                np.log(left) + np.log(right),
                product * (1.0 / left - 0.5 * points / left + 0.5 * points / right),
                product * (0.5 * points / left + 0.5 * points / right),
            ),
            axis=-1,
        )

    kernel_tolerance = np.where(
        sine != 0.0, _TWO_PI * tolerance / np.maximum(np.abs(sine), 1.0e-300), np.inf
    )
    tolerances = np.stack(
        (
            np.full(first.shape, 4.0 * _TWO_PI * tolerance),
            kernel_tolerance,
            kernel_tolerance,
        ),
        axis=1,
    )
    result = adaptive_unit_integrals(
        integrand,
        tolerances,
        max_depth=max_depth,
        max_intervals=max_intervals,
        max_workspace_bytes=max_workspace_bytes,
        label="shared-endpoint",
    )
    integrals = result.values
    single = -(first * second / _TWO_PI) * (-0.5 + 0.25 * integrals[:, 0])
    scale = first * sine / _TWO_PI
    shared = scale * integrals[:, 1]
    far = scale * integrals[:, 2]
    ordered = np.stack((shared, far), axis=1)
    double = np.where(shared_slot[:, None] == 0, ordered, ordered[:, ::-1])
    return _PairValues(
        single=single,
        double=double,
        single_errors=result.errors[:, 0] / (4.0 * _TWO_PI),
        double_errors=np.abs(sine) / _TWO_PI * np.max(result.errors[:, 1:], axis=1),
        evaluations=int(np.sum(result.evaluations)),
        depth=result.depth,
        workspace_bytes=result.workspace_bytes,
    )


def near_pair_values(
    geometry: PanelGeometry2D,
    targets: _Integers,
    sources: _Integers,
    /,
    *,
    tolerance: float,
    max_depth: int,
    max_intervals: int,
    max_workspace_bytes: int,
) -> _PairValues:
    """Exact inner straight-panel integrals with adaptive outer quadrature."""
    starts = geometry.starts
    tangents = geometry.tangents
    lengths = geometry.lengths

    def integrand(owners: _Integers, points: _Floats) -> _Floats:
        test = targets[owners]
        trial = sources[owners]
        locations = (
            starts[test][:, None, :]
            + (points * lengths[test][:, None])[..., None] * tangents[test][:, None, :]
        )
        single, start, end = segment_layer_moments(
            jnp.asarray(locations),
            jnp.asarray(starts[trial][:, None, :]),
            jnp.asarray(tangents[trial][:, None, :]),
            jnp.asarray(geometry.normals[trial][:, None, :]),
            jnp.asarray(lengths[trial][:, None]),
        )
        return np.stack(
            (
                np.asarray(single) / lengths[trial][:, None],
                np.asarray(start),
                np.asarray(end),
            ),
            axis=-1,
        )

    result = adaptive_unit_integrals(
        integrand,
        np.full((targets.shape[0], 3), tolerance, dtype=np.float64),
        max_depth=max_depth,
        max_intervals=max_intervals,
        max_workspace_bytes=max_workspace_bytes,
        label="near",
    )
    test_lengths = lengths[targets]
    return _PairValues(
        single=test_lengths * lengths[sources] * result.values[:, 0],
        double=test_lengths[:, None] * result.values[:, 1:],
        single_errors=result.errors[:, 0],
        double_errors=np.max(result.errors[:, 1:], axis=1),
        evaluations=int(np.sum(result.evaluations)),
        depth=result.depth,
        workspace_bytes=result.workspace_bytes,
    )


def regular_panel_rule(
    geometry: PanelGeometry2D,
    order: int,
    /,
) -> tuple[_Floats, _Floats, _Floats]:
    """Physical Gauss--Legendre points, arc-length weights, and hat values."""
    nodes, weights = unit_gauss_rule(order)
    points = (
        geometry.starts[:, None, :]
        + (nodes[None, :] * geometry.lengths[:, None])[..., None]
        * geometry.tangents[:, None, :]
    )
    hats = np.stack((1.0 - nodes, nodes), axis=1)
    return points, weights[None, :] * geometry.lengths[:, None], hats


def sweep_workspace_bytes(panel_count: int, rows: int, order: int, /) -> int:
    """Peak scratch of one classification/certification row block."""
    tensor = rows * order * panel_count * order
    reference = rows * _KRONROD_ORDER * panel_count
    return _FLOAT_BYTES * (12 * tensor + 16 * reference + 8 * rows * panel_count)


def _block_classes(
    geometry: PanelGeometry2D,
    rows: _Integers,
    near_ratio: float,
    /,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    count = geometry.lengths.shape[0]
    columns = np.arange(count, dtype=np.int64)
    ends = geometry.starts + geometry.lengths[:, None] * geometry.tangents
    distances = segment_distances(
        geometry.starts[rows][:, None, :],
        ends[rows][:, None, :],
        geometry.starts[None, :, :],
        ends[None, :, :],
    )
    offset = (columns[None, :] - rows[:, None]) % count
    singular = (offset == 0) | (offset == 1) | (offset == count - 1)
    scale = np.maximum(geometry.lengths[rows][:, None], geometry.lengths[None, :])
    near = ~singular & (distances < near_ratio * scale)
    return near, ~singular & ~near


def _regular_block_errors(
    geometry: PanelGeometry2D,
    rule: tuple[_Floats, _Floats, _Floats],
    rows: _Integers,
    regular: npt.NDArray[np.bool_],
    /,
) -> _Floats:
    """Normalized run-time rule error plus the reference's own Kronrod estimate."""
    points, weights, hats = rule
    active = jnp.asarray(regular)
    arguments = (
        jnp.asarray(points[rows]),
        jnp.asarray(weights[rows]),
        jnp.asarray(points),
        jnp.asarray(weights),
        jnp.asarray(geometry.normals),
    )
    single_rule = np.asarray(
        regular_pair_values(*arguments, jnp.ones((hats.shape[0], 1)), active, "single")
    )[..., 0]
    double_rule = np.asarray(
        regular_pair_values(*arguments, jnp.asarray(hats), active, "double")
    )
    nodes, kronrod, gauss = _unit_kronrod_rule()
    lengths = geometry.lengths
    locations = (
        geometry.starts[rows][:, None, :]
        + (nodes[None, :] * lengths[rows][:, None])[..., None]
        * geometry.tangents[rows][:, None, :]
    )
    single, start, end = segment_layer_moments(
        jnp.asarray(locations[:, :, None, :]),
        jnp.asarray(geometry.starts[None, None, :, :]),
        jnp.asarray(geometry.tangents[None, None, :, :]),
        jnp.asarray(geometry.normals[None, None, :, :]),
        jnp.asarray(lengths[None, None, :]),
    )
    moments = np.stack((np.asarray(single), np.asarray(start), np.asarray(end)), axis=-1)
    moments = np.where(regular[:, None, :, None], moments, 0.0)
    test_lengths = lengths[rows][:, None, None]
    kronrod_value = test_lengths * ein.contract("n,rnck->rck", kronrod, moments)
    gauss_value = test_lengths * ein.contract("n,rnck->rck", gauss, moments)
    pair_scale = lengths[rows][:, None] * lengths[None, :]
    reference_error = np.maximum(
        np.abs(kronrod_value[..., 0] - gauss_value[..., 0]) / pair_scale,
        np.max(np.abs(kronrod_value[..., 1:] - gauss_value[..., 1:]), axis=-1)
        / lengths[rows][:, None],
    )
    rule_error = np.maximum(
        np.abs(single_rule - kronrod_value[..., 0]) / pair_scale,
        np.max(np.abs(double_rule - kronrod_value[..., 1:]), axis=-1)
        / lengths[rows][:, None],
    )
    return np.where(regular, rule_error + reference_error, 0.0)


def _regular_sweep(
    geometry: PanelGeometry2D,
    *,
    regular_order: int,
    near_ratio: float,
    tolerance: float,
    max_preparation_workspace_bytes: int,
) -> _RegularSweep:
    """Classify separated pairs and certify every regular pair's run-time rule."""
    count = geometry.lengths.shape[0]
    rows_per_block = count
    while rows_per_block > 1 and (
        sweep_workspace_bytes(count, rows_per_block, regular_order)
        > max_preparation_workspace_bytes
    ):
        rows_per_block = (rows_per_block + 1) // 2
    workspace = sweep_workspace_bytes(count, rows_per_block, regular_order)
    if workspace > max_preparation_workspace_bytes:
        raise ValueError(
            "[preparation-bytes] Panel-pair certification exceeds its workspace budget."
        )
    rule = regular_panel_rule(geometry, regular_order)
    near_pairs: list[_Integers] = []
    promoted_pairs: list[_Integers] = []
    candidates = 0
    maximum = 0.0
    for first in range(0, count, rows_per_block):
        rows = np.arange(first, min(first + rows_per_block, count), dtype=np.int64)
        near, regular = _block_classes(geometry, rows, near_ratio)
        estimates = _regular_block_errors(geometry, rule, rows, regular)
        promote = regular & ~(estimates <= tolerance)
        accepted = regular & ~promote
        candidates += int(np.count_nonzero(regular))
        if np.any(accepted):
            maximum = max(maximum, float(np.max(estimates[accepted])))
        near_rows, near_columns = np.nonzero(near)
        near_pairs.append(np.stack((rows[near_rows], near_columns), axis=1))
        promoted_rows, promoted_columns = np.nonzero(promote)
        promoted_pairs.append(np.stack((rows[promoted_rows], promoted_columns), axis=1))
    per_pair = regular_order * regular_order + _KRONROD_ORDER
    return _RegularSweep(
        near=np.concatenate(near_pairs, axis=0).astype(np.int64),
        promoted=np.concatenate(promoted_pairs, axis=0).astype(np.int64),
        candidate_count=candidates,
        maximum_error=maximum,
        evaluations=candidates * per_pair,
        workspace_bytes=workspace,
    )


def _symmetric_pairs(pairs: _Integers, count: int, /) -> _Integers:
    keys = np.concatenate(
        (pairs[:, 0] * count + pairs[:, 1], pairs[:, 1] * count + pairs[:, 0])
    )
    unique = np.unique(keys)
    return np.stack((unique // count, unique % count), axis=1).astype(np.int64)


def _coincident_values(geometry: PanelGeometry2D, /) -> _PairValues:
    lengths = geometry.lengths
    logarithm = np.log(lengths)
    single = lengths * lengths * (1.5 - logarithm) / _TWO_PI
    rounding = 16.0 * np.finfo(np.float64).eps * (1.5 + np.abs(logarithm)) / _TWO_PI
    return _PairValues(
        single=single,
        double=np.zeros((lengths.shape[0], 2), dtype=np.float64),
        single_errors=rounding,
        double_errors=np.zeros_like(lengths),
        evaluations=lengths.shape[0],
        depth=0,
        workspace_bytes=0,
    )


def _mirror_single_values(
    targets: _Integers,
    sources: _Integers,
    single: _Floats,
    count: int,
    /,
) -> _Floats:
    """Copy each upper-triangle exceptional V entry to its transpose position."""
    keys = targets * count + sources
    order = np.argsort(keys, kind="stable")
    canonical = np.minimum(targets, sources) * count + np.maximum(targets, sources)
    positions = order[np.minimum(np.searchsorted(keys[order], canonical), keys.size - 1)]
    if np.any(keys[positions] != canonical):
        raise RuntimeError("Panel-pair exceptions are not closed under transposition.")
    return single[positions]


def _exception_pairs(
    count: int,
    sweep: _RegularSweep,
    /,
) -> tuple[_Integers, _Integers, _Integers]:
    """Ordered adjacent pairs, separated exception pairs, and promoted pairs."""
    panels = np.arange(count, dtype=np.int64)
    adjacent = np.stack(
        (
            np.concatenate((panels, panels)),
            np.concatenate(((panels + 1) % count, (panels - 1) % count)),
        ),
        axis=1,
    )
    promoted = (
        _symmetric_pairs(sweep.promoted, count)
        if sweep.promoted.size
        else np.zeros((0, 2), dtype=np.int64)
    )
    return adjacent, np.concatenate((sweep.near, promoted), axis=0), promoted


def prepare_panel_pairs_2d(
    geometry: PanelGeometry2D,
    /,
    *,
    regular_order: int,
    near_ratio: float,
    tolerance: float,
    max_depth: int,
    max_adaptive_intervals: int,
    max_exception_pairs: int,
    max_preparation_workspace_bytes: int,
) -> PanelPairData2D:
    """Classify, integrate, and certify every panel pair of a closed polygon."""
    count = geometry.lengths.shape[0]
    sweep = _regular_sweep(
        geometry,
        regular_order=regular_order,
        near_ratio=near_ratio,
        tolerance=tolerance,
        max_preparation_workspace_bytes=max_preparation_workspace_bytes,
    )
    adjacent, separated, promoted = _exception_pairs(count, sweep)
    exception_count = count + adjacent.shape[0] + separated.shape[0]
    if exception_count > max_exception_pairs:
        raise ValueError(
            "[exception-capacity] Panel-pair exceptions exceed max_exception_pairs."
        )
    panels = np.arange(count, dtype=np.int64)
    classes = (
        _coincident_values(geometry),
        shared_endpoint_values(
            geometry,
            adjacent[:, 0],
            adjacent[:, 1],
            tolerance=tolerance,
            max_depth=max_depth,
            max_intervals=max_adaptive_intervals,
            max_workspace_bytes=max_preparation_workspace_bytes,
        ),
        near_pair_values(
            geometry,
            separated[:, 0],
            separated[:, 1],
            tolerance=tolerance,
            max_depth=max_depth,
            max_intervals=max_adaptive_intervals,
            max_workspace_bytes=max_preparation_workspace_bytes,
        ),
    )
    targets = np.concatenate((panels, adjacent[:, 0], separated[:, 0]))
    sources = np.concatenate((panels, adjacent[:, 1], separated[:, 1]))
    labels = np.concatenate(
        [
            np.full(values.single.shape, index, dtype=np.int32)
            for index, values in enumerate(classes)
        ]
    )
    single = _mirror_single_values(
        targets, sources, np.concatenate([values.single for values in classes]), count
    )
    double = np.concatenate([values.double for values in classes], axis=0)
    order = np.argsort(targets * count + sources, kind="stable")
    class_errors = [
        max(
            float(np.max(values.single_errors, initial=0.0)),
            float(np.max(values.double_errors, initial=0.0)),
        )
        for values in classes
    ]
    return PanelPairData2D(
        exception_keys=jnp.asarray(
            targets[order] * count + sources[order], dtype=jnp.int64
        ),
        targets=jnp.asarray(targets[order], dtype=jnp.int32),
        sources=jnp.asarray(sources[order], dtype=jnp.int32),
        classes=jnp.asarray(labels[order], dtype=jnp.int32),
        single_values=jnp.asarray(single[order], dtype=jnp.float64),
        double_values=jnp.asarray(double[order], dtype=jnp.float64),
        maximum_errors=jnp.asarray(
            (*class_errors, sweep.maximum_error), dtype=jnp.float64
        ),
        evaluations=jnp.asarray(
            (*(values.evaluations for values in classes), sweep.evaluations),
            dtype=jnp.int64,
        ),
        pair_counts=(
            count,
            adjacent.shape[0],
            separated.shape[0],
            sweep.candidate_count - promoted.shape[0],
        ),
        promoted_count=promoted.shape[0],
        maximum_depth=max(values.depth for values in classes),
        preparation_workspace_bytes=max(
            sweep.workspace_bytes, *(values.workspace_bytes for values in classes)
        ),
        resident_bytes=exception_count * EXCEPTION_ENTRY_BYTES,
    )


__all__: list[str] = []

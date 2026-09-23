#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._compressible_contracts import CompressibleKineticModelKind


class CompressibleVelocityRule(StrictModule, NonTrainableState):
    velocities: Array
    base_probabilities: Array
    opposite: Array
    guided_features: Array
    model_kind: CompressibleKineticModelKind = eqx.field(static=True)
    name: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    population_count: int = eqx.field(static=True)
    dual_dimension: int = eqx.field(static=True)
    maximum_reach: tuple[int, ...] = eqx.field(static=True)
    frame_shift: tuple[int, ...] = eqx.field(static=True)
    exact_streaming: bool = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocities: ArrayLike,
        base_probabilities: ArrayLike,
        opposite: ArrayLike,
        guided_features: ArrayLike,
        /,
        *,
        model_kind: CompressibleKineticModelKind,
        name: str,
        frame_shift: tuple[int, ...] | None = None,
        exact_streaming: bool = True,
    ):
        velocity_host = np.asarray(velocities)
        base_host = np.asarray(base_probabilities)
        opposite_host = np.asarray(opposite, dtype=np.int32)
        feature_host = np.asarray(guided_features)
        if velocity_host.ndim != 2 or velocity_host.shape[0] < 2:
            raise ValueError("velocities must have shape (population, dimension).")
        q, dimension = velocity_host.shape
        if base_host.shape != (q,) or opposite_host.shape != (q,):
            raise ValueError("Velocity weights and opposite map must match Q.")
        if feature_host.ndim != 2 or feature_host.shape[0] != q:
            raise ValueError("guided_features must have shape (Q, dual_dimension).")
        if not np.all(np.isfinite(velocity_host)):
            raise ValueError("velocities must be finite.")
        if exact_streaming and not np.all(velocity_host == np.rint(velocity_host)):
            raise ValueError("Exact-streaming velocities must be integral.")
        if len({tuple(row) for row in velocity_host.tolist()}) != q:
            raise ValueError("velocities must be unique.")
        if not np.all(np.isfinite(base_host)) or np.any(base_host <= 0.0):
            raise ValueError("base_probabilities must be finite and positive.")
        base_host = base_host / np.sum(base_host)
        if np.any(opposite_host < 0) or np.any(opposite_host >= q):
            raise ValueError("opposite indices are outside the population range.")
        if not np.array_equal(opposite_host[opposite_host], np.arange(q)):
            raise ValueError("opposite must be an involution.")
        if not np.all(np.isfinite(feature_host)):
            raise ValueError("guided_features must be finite.")
        augmented = np.column_stack((np.ones(q), feature_host))
        if np.linalg.matrix_rank(augmented) != feature_host.shape[1] + 1:
            raise ValueError("guided_features must be affinely full rank.")
        shift = (
            (0,) * dimension
            if frame_shift is None
            else tuple(int(v) for v in frame_shift)
        )
        if len(shift) != dimension:
            raise ValueError("frame_shift must match the velocity dimension.")
        if model_kind not in (
            "guided-d3q39",
            "entropic-d3q343",
            "filtered-d3q33",
            "adaptive-gauge",
        ):
            raise ValueError(f"Unknown kinetic model kind {model_kind!r}.")
        if not name:
            raise ValueError("Velocity rule name must be non-empty.")
        dtype = np.result_type(velocity_host.dtype, base_host.dtype, feature_host.dtype)
        maximum_reach = tuple(
            int(np.max(np.abs(velocity_host[:, axis]))) for axis in range(dimension)
        )
        self.velocities = jnp.asarray(velocity_host, dtype=dtype)
        self.base_probabilities = jnp.asarray(base_host, dtype=dtype)
        self.opposite = jnp.asarray(opposite_host, dtype=jnp.int32)
        self.guided_features = jnp.asarray(feature_host, dtype=dtype)
        self.model_kind = model_kind
        self.name = str(name)
        self.dimension = dimension
        self.population_count = q
        self.dual_dimension = feature_host.shape[1]
        self.maximum_reach = maximum_reach
        self.frame_shift = shift
        self.exact_streaming = bool(exact_streaming)
        self.rule_id = canonical_fingerprint(
            {
                "kind": "compressible-velocity-rule",
                "model": model_kind,
                "name": name,
                "velocities": velocity_host.tolist(),
                "base_probabilities": base_host.tolist(),
                "opposite": opposite_host.tolist(),
                "guided_features": feature_host.tolist(),
                "frame_shift": list(shift),
                "exact_streaming": bool(exact_streaming),
            }
        )


def _opposite_indices(velocities: np.ndarray, center: np.ndarray) -> np.ndarray:
    lookup = {tuple(row.tolist()): index for index, row in enumerate(velocities)}
    result = []
    for velocity in velocities:
        opposite = tuple((2.0 * center - velocity).tolist())
        if opposite not in lookup:
            raise ValueError("Velocity set is not inversion-closed about its frame.")
        result.append(lookup[opposite])
    return np.asarray(result, dtype=np.int32)


def _guided_features(velocities: np.ndarray) -> np.ndarray:
    cx, cy, cz = velocities.T
    speed_squared = np.sum(velocities * velocities, axis=1)
    return np.column_stack(
        (
            cx,
            cy,
            cz,
            cx * cx,
            cy * cy,
            cz * cz,
            cx * cy,
            cx * cz,
            cy * cz,
            cx * speed_squared,
            cy * speed_squared,
            cz * speed_squared,
        )
    )


def _energy_features(velocities: np.ndarray) -> np.ndarray:
    speed_squared = np.sum(velocities * velocities, axis=1)
    return np.column_stack((velocities, speed_squared))


def _signed_permutations(base: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    points: set[tuple[int, int, int]] = set()
    for permutation in set(itertools.permutations(base)):
        active = tuple(index for index, value in enumerate(permutation) if value != 0)
        for signs in itertools.product((-1, 1), repeat=len(active)):
            point = list(permutation)
            for index, sign in zip(active, signs, strict=True):
                point[index] *= sign
            points.add(tuple(point))
    return points


def d3q39_guided_rule(*, dtype: np.dtype | str = np.float64) -> CompressibleVelocityRule:
    points = {(0, 0, 0)}
    for shell in ((1, 0, 0), (1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 0, 0)):
        points.update(_signed_permutations(shell))
    velocities = np.asarray(
        sorted(
            points,
            key=lambda value: (sum(component * component for component in value), value),
        ),
        dtype=dtype,
    )
    if velocities.shape != (39, 3):
        raise RuntimeError("Canonical D3Q39 construction did not produce 39 velocities.")
    center = np.zeros(3, dtype=velocities.dtype)
    return CompressibleVelocityRule(
        velocities,
        np.ones(39, dtype=dtype),
        _opposite_indices(velocities, center),
        _guided_features(velocities),
        model_kind="guided-d3q39",
        name="D3Q39-guided",
    )


def d3q343_entropic_rule(
    *,
    reference_temperature: float = 0.6979533220196831,
    dtype: np.dtype | str = np.float64,
) -> CompressibleVelocityRule:
    temperature = float(reference_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("reference_temperature must be finite and positive.")
    support = np.arange(-3, 4, dtype=np.float64)
    matrix = np.asarray(
        (
            (1.0, 2.0, 2.0, 2.0),
            (0.0, 2.0, 8.0, 18.0),
            (0.0, 2.0, 32.0, 162.0),
            (0.0, 2.0, 128.0, 1458.0),
        )
    )
    target = np.asarray((1.0, temperature, 3.0 * temperature**2, 15.0 * temperature**3))
    orbit_weights = np.linalg.solve(matrix, target)
    if np.any(orbit_weights <= 0.0):
        raise ValueError("reference_temperature does not yield positive V7 weights.")
    weights_1d = np.asarray(
        (
            orbit_weights[3],
            orbit_weights[2],
            orbit_weights[1],
            orbit_weights[0],
            orbit_weights[1],
            orbit_weights[2],
            orbit_weights[3],
        )
    )
    velocities = np.asarray(list(itertools.product(support, repeat=3)), dtype=dtype)
    weights = np.asarray(
        [
            weights_1d[int(x + 3)] * weights_1d[int(y + 3)] * weights_1d[int(z + 3)]
            for x, y, z in velocities
        ],
        dtype=dtype,
    )
    center = np.zeros(3, dtype=velocities.dtype)
    return CompressibleVelocityRule(
        velocities,
        weights,
        _opposite_indices(velocities, center),
        _energy_features(velocities),
        model_kind="entropic-d3q343",
        name="D3Q343-entropic",
    )


def d3q33_filtered_rule(
    *,
    reference_temperature: float = 1.0 / 3.0,
    dtype: np.dtype | str = np.float64,
) -> CompressibleVelocityRule:
    temperature = float(reference_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("reference_temperature must be finite and positive.")
    shells = (
        {(0, 0, 0)},
        _signed_permutations((1, 0, 0)),
        _signed_permutations((2, 0, 0)),
        _signed_permutations((1, 1, 0)),
        _signed_permutations((1, 1, 1)),
    )
    orbit_weights = (
        (360.0 * temperature**3 - 540.0 * temperature**2 + 222.0 * temperature - 19.0)
        / 108.0,
        temperature * (75.0 * temperature**2 - 125.0 * temperature + 54.0) / 648.0,
        temperature * (25.0 * temperature**2 - 25.0 * temperature + 9.0) / 1296.0,
        temperature**2 * (10.0 - 15.0 * temperature) / 216.0,
        temperature**3 / 8.0,
    )
    pairs = []
    for shell, weight in zip(shells, orbit_weights, strict=True):
        pairs.extend((point, weight) for point in shell)
    pairs.sort(
        key=lambda item: (sum(component * component for component in item[0]), item[0])
    )
    velocities = np.asarray([point for point, _ in pairs], dtype=dtype)
    weights = np.asarray([weight for _, weight in pairs], dtype=dtype)
    if velocities.shape != (33, 3):
        raise RuntimeError("Canonical D3Q33 construction did not produce 33 velocities.")
    center = np.zeros(3, dtype=velocities.dtype)
    return CompressibleVelocityRule(
        velocities,
        weights,
        _opposite_indices(velocities, center),
        _energy_features(velocities),
        model_kind="filtered-d3q33",
        name="D3Q33-filtered",
    )


def shift_compressible_velocity_rule(
    rule: CompressibleVelocityRule,
    shift: tuple[int, ...],
    /,
) -> CompressibleVelocityRule:
    if not isinstance(rule, CompressibleVelocityRule):
        raise TypeError("rule must be a CompressibleVelocityRule.")
    frame = tuple(int(value) for value in shift)
    if len(frame) != rule.dimension:
        raise ValueError("shift must match the rule dimension.")
    velocities = np.asarray(rule.velocities) + np.asarray(frame)[None, :]
    center = np.asarray(frame, dtype=velocities.dtype)
    features = (
        _guided_features(velocities)
        if rule.model_kind == "guided-d3q39"
        else _energy_features(velocities)
    )
    return CompressibleVelocityRule(
        velocities,
        np.asarray(rule.base_probabilities),
        _opposite_indices(velocities, center),
        features,
        model_kind=rule.model_kind,
        name=f"{rule.name}-shifted",
        frame_shift=frame,
        exact_streaming=rule.exact_streaming,
    )


__all__ = [
    "CompressibleVelocityRule",
    "d3q33_filtered_rule",
    "d3q39_guided_rule",
    "d3q343_entropic_rule",
    "shift_compressible_velocity_rule",
]

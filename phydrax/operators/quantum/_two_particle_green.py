#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit fermionic two-particle Matsubara channels and crossing evidence."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FermionicTwoParticleChannel = Literal[
    "particle-hole-direct",
    "particle-hole-crossed",
    "particle-particle",
]

_CHANNELS = (
    "particle-hole-direct",
    "particle-hole-crossed",
    "particle-particle",
)
_OPERATOR_ORDER = ("annihilation", "creation", "annihilation", "creation")


class FermionicTwoParticleChannelConvention(StrictModule, NonTrainableState):
    """One three-frequency routing for ``<T c c† c c†>_connected``.

    The four external legs always use the operator order
    ``(annihilation, creation, annihilation, creation)``. Integer fermionic
    labels ``n`` denote ``(2 n + 1) pi / beta`` and integer transfer labels
    ``m`` denote ``2 m pi / beta``. The channel changes only which three
    independent labels parameterize the same four external frequencies.
    """

    channel: FermionicTwoParticleChannel = eqx.field(static=True)
    operator_order: tuple[str, str, str, str] = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self, channel: FermionicTwoParticleChannel, /):
        if channel not in _CHANNELS:
            raise ValueError("Unknown fermionic two-particle channel.")
        self.channel = channel
        self.operator_order = _OPERATOR_ORDER
        self.convention_id = canonical_fingerprint(
            {
                "kind": "fermionic-two-particle-channel-convention",
                "channel": channel,
                "operator_order": _OPERATOR_ORDER,
                "fermionic_labels": "omega_n=(2n+1)pi/beta",
                "bosonic_labels": "nu_m=2mpi/beta",
            }
        )

    def route(
        self,
        transfer_labels: ArrayLike,
        left_labels: ArrayLike,
        right_labels: ArrayLike,
        /,
    ) -> Array:
        """Return external fermionic labels in canonical operator order."""
        transfer, left, right = jnp.broadcast_arrays(
            jnp.asarray(transfer_labels),
            jnp.asarray(left_labels),
            jnp.asarray(right_labels),
        )
        if not all(
            jnp.issubdtype(value.dtype, jnp.integer) for value in (transfer, left, right)
        ):
            raise TypeError("Two-particle Matsubara route labels must be integers.")
        if self.channel == "particle-hole-direct":
            routed = (left + transfer, left, right, right + transfer)
        elif self.channel == "particle-hole-crossed":
            routed = (right + transfer, left, left - transfer, right)
        else:
            routed = (left, right, transfer - left - 1, transfer - right - 1)
        return jnp.stack(routed, axis=-1).astype(jnp.int32)

    def coordinates(self, external_labels: ArrayLike, /) -> Array:
        """Invert this channel's routing for conserving external labels."""
        external = jnp.asarray(external_labels)
        if external.shape[-1:] != (4,) or not jnp.issubdtype(external.dtype, jnp.integer):
            raise TypeError("external_labels must end in four integer labels.")
        first, second, third, fourth = [external[..., i] for i in range(4)]
        if self.channel == "particle-hole-direct":
            coordinates = (first - second, second, third)
        elif self.channel == "particle-hole-crossed":
            coordinates = (first - fourth, second, fourth)
        else:
            coordinates = (first + third + 1, first, second)
        return jnp.stack(coordinates, axis=-1).astype(jnp.int32)


class FermionicTwoParticleRouting(StrictModule, NonTrainableState):
    """Unwrapped and cyclic-bank external routes with explicit wrap counts."""

    external_labels: Array
    wrapped_external_labels: Array
    wraps: Array
    conservation_residual: Array
    exact: Array
    fermion_label_minimum: int = eqx.field(static=True)
    fermion_label_count: int = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)


class FermionicCrossingEvidence(StrictModule, NonTrainableState):
    annihilation_exchange_residual: Array
    creation_exchange_residual: Array
    simultaneous_exchange_residual: Array
    coverage_fraction: Array
    finite: Array
    satisfied: Array
    representation_id: str = eqx.field(static=True)


class TwoParticleChannelPermutation(StrictModule, NonTrainableState):
    """Cyclic coordinate gather from a source channel into a target channel."""

    source_flat_indices: Array
    coordinate_wraps: Array
    routing_residual: Array
    exact: Array
    source_convention_id: str = eqx.field(static=True)
    target_convention_id: str = eqx.field(static=True)


class MatsubaraTwoParticleGreenFunction(StrictModule):
    """Connected fermionic four-point samples on a bounded cyclic label bank.

    ``values`` has shape ``(transfer, left, right, mode, mode, mode, mode)``.
    The final four axes follow the convention's immutable ``c c† c c†`` order.
    No disconnected subtraction, crossing completion, or frequency continuation
    is inferred by this type.
    """

    transfer_labels: Array
    left_labels: Array
    right_labels: Array
    values: Array
    routing: FermionicTwoParticleRouting
    finite: Array
    valid: Array
    beta: float = eqx.field(static=True)
    connected: bool = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    maximum_elements: int = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    convention: FermionicTwoParticleChannelConvention

    def __init__(
        self,
        convention: FermionicTwoParticleChannelConvention,
        beta: float,
        transfer_labels: ArrayLike,
        left_labels: ArrayLike,
        right_labels: ArrayLike,
        values: ArrayLike,
        /,
        *,
        connected: bool = True,
        fermion_label_minimum: int | None = None,
        fermion_label_count: int | None = None,
        maximum_elements: int = 16_777_216,
        representation_id: str | None = None,
    ):
        if not isinstance(convention, FermionicTwoParticleChannelConvention):
            raise TypeError("convention must be FermionicTwoParticleChannelConvention.")
        beta_ = float(beta)
        capacity = int(maximum_elements)
        if not np.isfinite(beta_) or beta_ <= 0.0 or capacity <= 0:
            raise ValueError("beta and maximum_elements must be positive.")
        labels = tuple(
            np.asarray(value) for value in (transfer_labels, left_labels, right_labels)
        )
        if any(
            value.ndim != 1
            or value.size == 0
            or not np.issubdtype(value.dtype, np.integer)
            or np.unique(value).size != value.size
            for value in labels
        ):
            raise ValueError("Frequency axes must be non-empty unique integer labels.")
        transfer, left, right = (value.astype(np.int32) for value in labels)
        payload = np.asarray(values)
        prefix = (transfer.size, left.size, right.size)
        if (
            payload.ndim != 7
            or payload.shape[:3] != prefix
            or len(set(payload.shape[3:])) != 1
            or payload.shape[3] == 0
        ):
            raise ValueError(
                "values must have shape (transfer, left, right, mode, mode, mode, mode)."
            )
        if payload.size > capacity:
            raise ValueError("Two-particle Green payload exceeds maximum_elements.")
        minimum = (
            int(np.min(np.concatenate((left, right))))
            if fermion_label_minimum is None
            else int(fermion_label_minimum)
        )
        count = (
            int(max(np.max(left), np.max(right)) - minimum + 1)
            if fermion_label_count is None
            else int(fermion_label_count)
        )
        if (
            count <= 0
            or np.any(left < minimum)
            or np.any(left >= minimum + count)
            or np.any(right < minimum)
            or np.any(right >= minimum + count)
        ):
            raise ValueError("Independent fermionic labels lie outside the cyclic bank.")
        if representation_id is not None and not str(representation_id):
            raise ValueError("representation_id must be non-empty when provided.")
        routing = route_matsubara_two_particle(
            convention,
            transfer[:, None, None],
            left[None, :, None],
            right[None, None, :],
            fermion_label_minimum=minimum,
            fermion_label_count=count,
        )
        finite = jnp.all(jnp.isfinite(jnp.asarray(payload)))
        valid = finite & routing.exact
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "matsubara-two-particle-green-function",
                    "convention": convention.convention_id,
                    "beta": beta_,
                    "transfer": array_tree_fingerprint(transfer),
                    "left": array_tree_fingerprint(left),
                    "right": array_tree_fingerprint(right),
                    "mode_count": int(payload.shape[3]),
                    "connected": bool(connected),
                    "fermion_bank": (minimum, count),
                }
            )
            if representation_id is None
            else str(representation_id)
        )
        self.convention = convention
        self.beta = beta_
        self.transfer_labels = jnp.asarray(transfer)
        self.left_labels = jnp.asarray(left)
        self.right_labels = jnp.asarray(right)
        self.values = jnp.asarray(payload)
        self.routing = routing
        self.finite = finite
        self.valid = valid
        self.connected = bool(connected)
        self.mode_count = int(payload.shape[3])
        self.maximum_elements = capacity
        self.representation_id = identifier


def route_matsubara_two_particle(
    convention: FermionicTwoParticleChannelConvention,
    transfer_labels: ArrayLike,
    left_labels: ArrayLike,
    right_labels: ArrayLike,
    /,
    *,
    fermion_label_minimum: int,
    fermion_label_count: int,
) -> FermionicTwoParticleRouting:
    """Route and wrap four fermionic legs without hiding aliasing."""
    if not isinstance(convention, FermionicTwoParticleChannelConvention):
        raise TypeError("convention must be FermionicTwoParticleChannelConvention.")
    minimum = int(fermion_label_minimum)
    count = int(fermion_label_count)
    if count <= 0:
        raise ValueError("fermion_label_count must be positive.")
    external = convention.route(transfer_labels, left_labels, right_labels)
    wraps = jnp.floor_divide(external - minimum, count)
    wrapped = external - count * wraps
    conservation = (
        external[..., 0] + external[..., 2] - external[..., 1] - external[..., 3]
    )
    residual = jnp.max(jnp.abs(conservation))
    exact = residual == 0
    return FermionicTwoParticleRouting(
        external,
        wrapped.astype(jnp.int32),
        wraps.astype(jnp.int32),
        residual,
        exact,
        minimum,
        count,
        convention.convention_id,
    )


def _axis_lookup(labels: np.ndarray, value: int) -> int | None:
    matches = np.flatnonzero(labels == value)
    return None if matches.size == 0 else int(matches[0])


def _crossing_residual(
    green: MatsubaraTwoParticleGreenFunction,
    leg_permutation: tuple[int, int, int, int],
    sign: int,
) -> tuple[Array, int, int]:
    transfer = np.asarray(green.transfer_labels)
    left = np.asarray(green.left_labels)
    right = np.asarray(green.right_labels)
    routes = np.asarray(green.routing.external_labels)
    values = green.values
    maximum = jnp.asarray(0.0, dtype=values.real.dtype)
    covered = 0
    total = int(np.prod(routes.shape[:-1]))
    orbital_axes = tuple(3 + value for value in leg_permutation)
    for index in np.ndindex(routes.shape[:-1]):
        crossed_route = routes[index][list(leg_permutation)]
        coordinate = np.asarray(green.convention.coordinates(crossed_route))
        target = (
            _axis_lookup(transfer, int(coordinate[0])),
            _axis_lookup(left, int(coordinate[1])),
            _axis_lookup(right, int(coordinate[2])),
        )
        if any(item is None for item in target):
            continue
        source_value = values[index]
        target_value = values[tuple(int(item) for item in target)]
        target_value = jnp.transpose(
            target_value, tuple(value - 3 for value in orbital_axes)
        )
        maximum = jnp.maximum(
            maximum, jnp.max(jnp.abs(source_value - sign * target_value))
        )
        covered += 1
    return maximum, covered, total


def fermionic_crossing_evidence(
    green: MatsubaraTwoParticleGreenFunction,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> FermionicCrossingEvidence:
    """Check Pauli exchange identities wherever the finite grid has coverage."""
    if not isinstance(green, MatsubaraTwoParticleGreenFunction):
        raise TypeError("green must be MatsubaraTwoParticleGreenFunction.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    annihilation, covered_a, total = _crossing_residual(green, (2, 1, 0, 3), -1)
    creation, covered_c, _ = _crossing_residual(green, (0, 3, 2, 1), -1)
    simultaneous, covered_b, _ = _crossing_residual(green, (2, 3, 0, 1), 1)
    coverage = jnp.asarray(min(covered_a, covered_c, covered_b) / total)
    residuals = jnp.stack((annihilation, creation, simultaneous))
    finite = green.finite & jnp.all(jnp.isfinite(residuals))
    satisfied = finite & (coverage > 0.0) & (jnp.max(residuals) <= tolerance_)
    return FermionicCrossingEvidence(
        annihilation,
        creation,
        simultaneous,
        coverage,
        finite,
        satisfied,
        green.representation_id,
    )


def _cyclic_axis_index(
    labels: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(labels)
    if not np.array_equal(ordered, np.arange(ordered[0], ordered[0] + ordered.size)):
        raise ValueError("Channel conversion requires a complete contiguous cyclic axis.")
    minimum = int(ordered[0])
    count = int(ordered.size)
    wraps = np.floor_divide(values - minimum, count)
    wrapped = values - count * wraps
    positions = {int(value): index for index, value in enumerate(labels)}
    indices = np.asarray(
        [positions[int(value)] for value in wrapped.reshape(-1)], dtype=np.int32
    )
    return indices.reshape(values.shape), wraps.astype(np.int32)


def two_particle_channel_permutation(
    green: MatsubaraTwoParticleGreenFunction,
    target: FermionicTwoParticleChannelConvention,
    /,
) -> TwoParticleChannelPermutation:
    """Prepare an exact cyclic gather that represents the same external legs."""
    if not isinstance(green, MatsubaraTwoParticleGreenFunction):
        raise TypeError("green must be MatsubaraTwoParticleGreenFunction.")
    if not isinstance(target, FermionicTwoParticleChannelConvention):
        raise TypeError("target must be FermionicTwoParticleChannelConvention.")
    transfer = np.asarray(green.transfer_labels)
    left = np.asarray(green.left_labels)
    right = np.asarray(green.right_labels)
    mesh = np.meshgrid(transfer, left, right, indexing="ij")
    target_routes = np.asarray(target.route(mesh[0], mesh[1], mesh[2]))
    source_coordinates = np.asarray(green.convention.coordinates(target_routes))
    transfer_index, transfer_wrap = _cyclic_axis_index(
        transfer, source_coordinates[..., 0]
    )
    left_index, left_wrap = _cyclic_axis_index(left, source_coordinates[..., 1])
    right_index, right_wrap = _cyclic_axis_index(right, source_coordinates[..., 2])
    flat = np.ravel_multi_index(
        (transfer_index, left_index, right_index),
        (transfer.size, left.size, right.size),
    ).astype(np.int32)
    source_routes = np.asarray(green.routing.external_labels).reshape((-1, 4))[
        flat.reshape(-1)
    ]
    wrapped_source = source_routes - green.routing.fermion_label_count * np.floor_divide(
        source_routes - green.routing.fermion_label_minimum,
        green.routing.fermion_label_count,
    )
    wrapped_target = target_routes.reshape(
        (-1, 4)
    ) - green.routing.fermion_label_count * np.floor_divide(
        target_routes.reshape((-1, 4)) - green.routing.fermion_label_minimum,
        green.routing.fermion_label_count,
    )
    residual = np.max(np.abs(wrapped_source - wrapped_target), initial=0)
    return TwoParticleChannelPermutation(
        jnp.asarray(flat),
        jnp.asarray(np.stack((transfer_wrap, left_wrap, right_wrap), axis=-1)),
        jnp.asarray(residual),
        jnp.asarray(residual == 0),
        green.convention.convention_id,
        target.convention_id,
    )


def rechannel_matsubara_two_particle(
    green: MatsubaraTwoParticleGreenFunction,
    target: FermionicTwoParticleChannelConvention,
    /,
) -> MatsubaraTwoParticleGreenFunction:
    """Represent one cyclic finite bank in another channel without leg reordering."""
    permutation = two_particle_channel_permutation(green, target)
    if not bool(np.asarray(permutation.exact)):
        raise ValueError("Two-particle channel permutation does not preserve routing.")
    shape = green.values.shape
    flat = green.values.reshape((-1,) + shape[3:])
    converted = flat[permutation.source_flat_indices.reshape(-1)].reshape(shape)
    return MatsubaraTwoParticleGreenFunction(
        target,
        green.beta,
        green.transfer_labels,
        green.left_labels,
        green.right_labels,
        converted,
        connected=green.connected,
        fermion_label_minimum=green.routing.fermion_label_minimum,
        fermion_label_count=green.routing.fermion_label_count,
        maximum_elements=green.maximum_elements,
    )


__all__ = [
    "FermionicCrossingEvidence",
    "FermionicTwoParticleChannel",
    "FermionicTwoParticleChannelConvention",
    "FermionicTwoParticleRouting",
    "MatsubaraTwoParticleGreenFunction",
    "TwoParticleChannelPermutation",
    "fermionic_crossing_evidence",
    "rechannel_matsubara_two_particle",
    "route_matsubara_two_particle",
    "two_particle_channel_permutation",
]

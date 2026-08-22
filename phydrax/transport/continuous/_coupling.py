#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import prod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._frozendict import frozendict
from ..._strict import StrictModule
from .._results import AbstractBalancedTransportPlan, require_converged


def _positive_count(value: int, name: str, /) -> int:
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive.")
    return count


def _atoms(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or int(array.shape[0]) <= 0:
        raise ValueError(f"{name} must contain a non-empty leading atom axis.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _probabilities(value: ArrayLike | None, count: int, name: str, /) -> Array:
    if value is None:
        return jnp.full((count,), 1.0 / float(count), dtype=float)
    probabilities = jnp.asarray(value, dtype=float)
    if probabilities.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}; got {probabilities.shape}.")
    probabilities = eqx.error_if(
        probabilities,
        jnp.any(~jnp.isfinite(probabilities) | (probabilities < 0.0)),
        f"{name} must be finite and nonnegative.",
    )
    mass = jnp.sum(probabilities)
    mass = eqx.error_if(
        mass,
        ~(jnp.isfinite(mass) & (mass > 0.0)),
        f"{name} must have positive finite mass.",
    )
    return probabilities / mass


def _context(
    values: Mapping[str, ArrayLike] | None,
    atom_count: int,
    indices: Array,
    /,
) -> frozendict[str, Array]:
    if values is None:
        return frozendict()
    gathered: dict[str, Array] = {}
    for name, value in values.items():
        label = str(name)
        if not label:
            raise ValueError("Endpoint context labels must be non-empty.")
        if label in ("x", "t", "source", "target"):
            raise ValueError(f"Endpoint context label {label!r} is reserved.")
        array = jnp.asarray(value)
        if array.ndim < 1 or int(array.shape[0]) != atom_count:
            raise ValueError(
                f"Endpoint context {label!r} must begin with atom count "
                f"{atom_count}; got {array.shape}."
            )
        gathered[label] = array[indices]
    return frozendict(gathered)


class EndpointCouplingSample(StrictModule):
    """Finite samples from one explicit endpoint coupling."""

    source: Array
    target: Array
    source_indices: Array
    target_indices: Array
    valid: Array
    log_weights: Array
    context: frozendict[str, Array]
    coupling_status: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    num_pairs: int = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: ArrayLike,
        target: ArrayLike,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        valid: ArrayLike,
        log_weights: ArrayLike,
        context: Mapping[str, ArrayLike] | None = None,
        coupling_status: ArrayLike = 0,
        coupling_id: str,
        provenance: str,
    ):
        source_array = jnp.asarray(source)
        target_array = jnp.asarray(target, dtype=source_array.dtype)
        if source_array.shape != target_array.shape or source_array.ndim < 1:
            raise ValueError(
                "Coupled source and target values must have matching non-scalar shapes."
            )
        count = int(source_array.shape[0])
        source_index = jnp.asarray(source_indices, dtype=jnp.int32)
        target_index = jnp.asarray(target_indices, dtype=jnp.int32)
        validity = jnp.asarray(valid, dtype=bool)
        weights = jnp.asarray(log_weights, dtype=float)
        expected = (count,)
        if not (
            source_index.shape
            == target_index.shape
            == validity.shape
            == weights.shape
            == expected
        ):
            raise ValueError("Coupling indices, validity, and log weights must align.")
        weights = eqx.error_if(
            weights,
            jnp.any(validity & ~jnp.isfinite(weights)),
            "Valid endpoint-pair log weights must be finite.",
        )
        active_count = jnp.sum(validity)
        active_count = eqx.error_if(
            active_count,
            active_count <= 0,
            "Endpoint coupling contains no valid pairs.",
        )
        safe = jnp.where(validity, weights, -jnp.inf)
        normalizer = logsumexp(safe) + 0.0 * active_count
        normalized = jnp.where(validity, safe - normalizer, -jnp.inf)
        resolved_context = frozendict(
            {}
            if context is None
            else {str(name): jnp.asarray(value) for name, value in context.items()}
        )
        for name, value in resolved_context.items():
            if not name or name in ("x", "t", "source", "target"):
                raise ValueError(
                    f"Endpoint context label {name!r} is invalid or reserved."
                )
            if value.ndim < 1 or int(value.shape[0]) != count:
                raise ValueError(
                    f"Pair-aligned context {name!r} must begin with {count}; "
                    f"got {value.shape}."
                )
        if not isinstance(coupling_id, str) or not coupling_id:
            raise ValueError("coupling_id must be a non-empty string.")
        if not isinstance(provenance, str) or not provenance:
            raise ValueError("provenance must be a non-empty string.")
        status = jnp.asarray(coupling_status, dtype=jnp.int32).reshape(())
        self.source = source_array
        self.target = target_array
        self.source_indices = source_index
        self.target_indices = target_index
        self.valid = validity
        self.log_weights = normalized
        self.context = resolved_context
        self.coupling_status = status
        self.event_shape = tuple(source_array.shape[1:])
        self.num_pairs = count
        self.coupling_id = coupling_id
        self.provenance = provenance

    @property
    def event_size(self) -> int:
        return prod(self.event_shape)

    @property
    def probabilities(self) -> Array:
        return jnp.where(self.valid, jnp.exp(self.log_weights), 0.0)


def independent_endpoint_coupling(
    source: ArrayLike,
    target: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    num_pairs: int,
    source_probabilities: ArrayLike | None = None,
    target_probabilities: ArrayLike | None = None,
    target_context: Mapping[str, ArrayLike] | None = None,
    coupling_id: str | None = None,
) -> EndpointCouplingSample:
    """Sample endpoint pairs from the independent empirical product coupling."""
    source_atoms = _atoms(source, "source")
    target_atoms = _atoms(target, "target")
    if tuple(source_atoms.shape[1:]) != tuple(target_atoms.shape[1:]):
        raise ValueError("Source and target endpoint event shapes must match.")
    count = _positive_count(num_pairs, "num_pairs")
    source_weights = _probabilities(
        source_probabilities, int(source_atoms.shape[0]), "source_probabilities"
    )
    target_weights = _probabilities(
        target_probabilities, int(target_atoms.shape[0]), "target_probabilities"
    )
    source_key, target_key = jr.split(key)
    source_indices = jr.categorical(
        source_key, jnp.log(source_weights), shape=(count,)
    ).astype(jnp.int32)
    target_indices = jr.categorical(
        target_key, jnp.log(target_weights), shape=(count,)
    ).astype(jnp.int32)
    resolved_id = (
        canonical_fingerprint(
            {
                "kind": "independent-endpoint-coupling-v1",
                "event_shape": list(source_atoms.shape[1:]),
                "source_atoms": int(source_atoms.shape[0]),
                "target_atoms": int(target_atoms.shape[0]),
            }
        )
        if coupling_id is None
        else str(coupling_id)
    )
    return EndpointCouplingSample(
        source=source_atoms[source_indices],
        target=target_atoms[target_indices],
        source_indices=source_indices,
        target_indices=target_indices,
        valid=jnp.ones((count,), dtype=bool),
        log_weights=jnp.full((count,), -jnp.log(float(count))),
        context=_context(target_context, int(target_atoms.shape[0]), target_indices),
        coupling_status=jnp.asarray(0, dtype=jnp.int32),
        coupling_id=resolved_id,
        provenance="independent-empirical-product-coupling",
    )


def transport_plan_endpoint_coupling(
    plan: AbstractBalancedTransportPlan,
    source: ArrayLike,
    target: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    num_pairs: int,
    target_context: Mapping[str, ArrayLike] | None = None,
    coupling_id: str | None = None,
) -> EndpointCouplingSample:
    """Draw joint endpoint pairs from one converged balanced transport plan."""
    if not isinstance(plan, AbstractBalancedTransportPlan):
        raise TypeError("plan must implement AbstractBalancedTransportPlan.")
    checked = require_converged(plan)
    source_atoms = _atoms(source, "source")
    target_atoms = _atoms(target, "target")
    if tuple(source_atoms.shape[1:]) != tuple(target_atoms.shape[1:]):
        raise ValueError("Source and target endpoint event shapes must match.")
    matrix = jnp.asarray(checked.dense_plan(), dtype=float)
    expected = (int(source_atoms.shape[0]), int(target_atoms.shape[0]))
    if matrix.shape != expected:
        raise ValueError(
            f"Transport plan shape must match endpoint atom counts {expected}; "
            f"got {matrix.shape}."
        )
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix) | (matrix < 0.0)),
        "Transport coupling mass must be finite and nonnegative.",
    )
    flat = matrix.reshape((-1,))
    mass = jnp.sum(flat)
    mass = eqx.error_if(
        mass,
        ~(jnp.isfinite(mass) & (mass > 0.0)),
        "Transport coupling must have positive finite mass.",
    )
    probabilities = flat / mass
    count = _positive_count(num_pairs, "num_pairs")
    joint = jr.categorical(key, jnp.log(probabilities), shape=(count,)).astype(jnp.int32)
    target_count = int(target_atoms.shape[0])
    source_indices = joint // target_count
    target_indices = joint % target_count
    resolved_id = (
        canonical_fingerprint(
            {
                "kind": "balanced-plan-endpoint-coupling-v1",
                "plan_type": f"{type(plan).__module__}.{type(plan).__name__}",
                "shape": list(matrix.shape),
            }
        )
        if coupling_id is None
        else str(coupling_id)
    )
    return EndpointCouplingSample(
        source=source_atoms[source_indices],
        target=target_atoms[target_indices],
        source_indices=source_indices,
        target_indices=target_indices,
        valid=jnp.ones((count,), dtype=bool),
        log_weights=jnp.full((count,), -jnp.log(float(count))),
        context=_context(target_context, target_count, target_indices),
        coupling_status=jnp.asarray(0, dtype=jnp.int32),
        coupling_id=resolved_id,
        provenance="sampled-balanced-transport-plan",
    )


__all__ = [
    "EndpointCouplingSample",
    "independent_endpoint_coupling",
    "transport_plan_endpoint_coupling",
]

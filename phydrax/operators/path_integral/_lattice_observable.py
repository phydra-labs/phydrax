#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._scalar_lattice import Phi4LatticeAction


LatticeObservableNormalization: TypeAlias = Literal[
    "extensive",
    "site-mean",
    "physical-measure",
    "pair-mean",
]
LatticeObservableKind: TypeAlias = Literal["real", "complex"]


class LatticeObservablePlan(StrictModule):
    """One topology-bound per-configuration lattice observable."""

    evaluate: Callable[[Array], Array] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    normalization: LatticeObservableNormalization = eqx.field(static=True)
    output_kind: LatticeObservableKind = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluate: Callable[[Array], Array],
        /,
        *,
        output_shape: Sequence[int] = (),
        topology_id: str,
        field_space_id: str,
        normalization: LatticeObservableNormalization,
        output_kind: LatticeObservableKind,
        observable_id: str,
    ):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable.")
        shape = tuple(output_shape)
        if any(value <= 0 for value in shape):
            raise ValueError("output_shape dimensions must be positive.")
        if normalization not in (
            "extensive",
            "site-mean",
            "physical-measure",
            "pair-mean",
        ):
            raise ValueError("Unknown lattice-observable normalization.")
        if output_kind not in ("real", "complex"):
            raise ValueError("output_kind must be 'real' or 'complex'.")
        identifiers = str(topology_id), str(field_space_id), str(observable_id)
        if any(not value for value in identifiers):
            raise ValueError("Lattice observable identifiers must be non-empty.")
        self.evaluate = evaluate
        self.output_shape = shape
        self.topology_id, self.field_space_id, self.observable_id = identifiers
        self.normalization = normalization
        self.output_kind = output_kind


class LatticeObservableValue(StrictModule):
    """One observable value with finite-value evidence."""

    value: Array
    valid: Array
    observable_id: str = eqx.field(static=True)
    normalization: LatticeObservableNormalization = eqx.field(static=True)


def evaluate_lattice_observable(
    plan: LatticeObservablePlan,
    configuration: ArrayLike,
    /,
) -> LatticeObservableValue:
    """Evaluate one declared observable and enforce its output contract."""
    if not isinstance(plan, LatticeObservablePlan):
        raise TypeError("plan must be LatticeObservablePlan.")
    value = jnp.asarray(plan.evaluate(jnp.asarray(configuration)))
    if value.shape != plan.output_shape:
        raise ValueError(
            f"Observable output must have shape {plan.output_shape}; got {value.shape}."
        )
    if plan.output_kind == "real" and jnp.iscomplexobj(value):
        raise TypeError("A real lattice observable returned complex values.")
    return LatticeObservableValue(
        value=value,
        valid=jnp.all(jnp.isfinite(value)),
        observable_id=plan.observable_id,
        normalization=plan.normalization,
    )


def phi4_observable_plans(
    action: Phi4LatticeAction,
    /,
) -> tuple[LatticeObservablePlan, ...]:
    """Return canonical action and magnetization observables for one phi4 model."""
    if not isinstance(action, Phi4LatticeAction):
        raise TypeError("action must be Phi4LatticeAction.")
    weights = action.discretization.dual_measures[0]
    volume = jnp.sum(weights)

    def magnetization(field):
        return jnp.sum(weights * field) / volume

    common = {
        "topology_id": action.topology_id,
        "field_space_id": action.field_space_id,
        "output_kind": "real",
    }
    return (
        LatticeObservablePlan(
            lambda field: action.action(field) / volume,
            normalization="physical-measure",
            observable_id=f"{action.action_id}:action-density",
            **common,
        ),
        LatticeObservablePlan(
            magnetization,
            normalization="physical-measure",
            observable_id=f"{action.action_id}:magnetization",
            **common,
        ),
        LatticeObservablePlan(
            lambda field: jnp.abs(magnetization(field)),
            normalization="physical-measure",
            observable_id=f"{action.action_id}:absolute-magnetization",
            **common,
        ),
        LatticeObservablePlan(
            lambda field: magnetization(field) ** 2,
            normalization="physical-measure",
            observable_id=f"{action.action_id}:magnetization-squared",
            **common,
        ),
        LatticeObservablePlan(
            lambda field: magnetization(field) ** 4,
            normalization="physical-measure",
            observable_id=f"{action.action_id}:magnetization-fourth",
            **common,
        ),
    )


def phi4_pair_correlation_plan(
    action: Phi4LatticeAction,
    source_indices: ArrayLike,
    target_indices: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    observable_id: str | None = None,
) -> LatticeObservablePlan:
    """Build a translation/pair-averaged scalar two-point observable."""
    if not isinstance(action, Phi4LatticeAction):
        raise TypeError("action must be Phi4LatticeAction.")
    source = np.asarray(source_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    if source.ndim != 1 or target.shape != source.shape or source.size == 0:
        raise ValueError("source_indices and target_indices must be non-empty vectors.")
    site_count = action.configuration_shape[0]
    if np.any(source < 0) or np.any(source >= site_count):
        raise ValueError("source_indices lie outside the scalar field.")
    if np.any(target < 0) or np.any(target >= site_count):
        raise ValueError("target_indices lie outside the scalar field.")
    weight = (
        np.ones(source.shape, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weight.shape != source.shape or np.any(~np.isfinite(weight)):
        raise ValueError("Pair weights must be finite and match the index vectors.")
    normalization = float(np.sum(weight))
    if not np.isfinite(normalization) or normalization == 0.0:
        raise ValueError("Pair weights must have a finite nonzero sum.")
    source_array = jnp.asarray(source, dtype=jnp.int32)
    target_array = jnp.asarray(target, dtype=jnp.int32)
    weight_array = jnp.asarray(weight)
    resolved_id = (
        canonical_fingerprint(
            {
                "kind": "phi4-pair-correlation",
                "action": action.action_id,
                "source": array_tree_fingerprint(source),
                "target": array_tree_fingerprint(target),
                "weights": array_tree_fingerprint(weight),
            }
        )
        if observable_id is None
        else str(observable_id)
    )
    if not resolved_id:
        raise ValueError("observable_id must be non-empty.")

    def evaluate(field):
        values = jnp.asarray(field)
        if values.shape != action.configuration_shape:
            raise ValueError("Scalar field shape does not match the phi4 action.")
        return (
            jnp.sum(weight_array * values[source_array] * values[target_array])
            / normalization
        )

    return LatticeObservablePlan(
        evaluate,
        topology_id=action.topology_id,
        field_space_id=action.field_space_id,
        normalization="pair-mean",
        output_kind="real",
        observable_id=resolved_id,
    )


__all__ = [
    "LatticeObservableKind",
    "LatticeObservableNormalization",
    "LatticeObservablePlan",
    "LatticeObservableValue",
    "evaluate_lattice_observable",
    "phi4_observable_plans",
    "phi4_pair_correlation_plan",
]

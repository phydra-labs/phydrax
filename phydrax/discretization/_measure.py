#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import Bool, Dim, Float, parse, Scalar, Scope
from ._core import nonempty_identifier, resolved_identifier


MeasureNormalization: TypeAlias = Literal[
    "physical",
    "probability",
    "counting",
    "signed",
]


class MeasureEntityDim(Dim):
    """Capacity of the owning measure's entity set."""


def _validate_host_measure(
    values: ArrayLike,
    mask: ArrayLike,
    normalization: MeasureNormalization,
    tolerance: float,
    /,
) -> None:
    active = np.asarray(values)[np.asarray(mask)]
    if np.any(~np.isfinite(active)):
        raise ValueError("Active measure weights must be finite.")
    if normalization != "signed" and np.any(active < 0):
        raise ValueError("Unsigned measure weights must be non-negative.")
    if normalization == "counting" and np.any(active != 1):
        raise ValueError("Counting-measure active weights must equal one.")
    total = float(np.sum(active, dtype=np.float64))
    if normalization in ("physical", "probability") and not total > 0:
        raise ValueError("Physical and probability measures require positive mass.")
    if normalization == "probability" and not np.isclose(
        total, 1.0, rtol=tolerance, atol=tolerance
    ):
        raise ValueError("Probability-measure active weights must sum to one.")


@final
class DiscreteMeasure(StrictModule, NonTrainableState):
    """Identity-bearing finite measure with dynamic weights and total mass.

    Traced construction requires an explicit stable `measure_id`. Its numerical
    admissibility is exposed by `valid` and can be checked eagerly with `admit`.
    """

    __strict_contract__ = True

    name: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    normalization: MeasureNormalization = eqx.field(static=True)
    weights: Float[MeasureEntityDim]
    active_mask: Bool[MeasureEntityDim]
    total_mass: Float[Scalar]
    measure_id: str = eqx.field(static=True)
    probability_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        support_id: str,
        entity_set_id: str,
        weights: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        normalization: MeasureNormalization = "physical",
        probability_tolerance: float = 1e-10,
        measure_id: str | None = None,
    ) -> None:
        name_ = nonempty_identifier("name", name)
        support_id_ = nonempty_identifier("support_id", support_id)
        entity_set_id_ = nonempty_identifier("entity_set_id", entity_set_id)
        normalization = parse(normalization, MeasureNormalization, "normalization")
        values = jnp.asarray(weights)
        if values.ndim != 1:
            raise ValueError("Discrete measure weights must be rank-1.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(jnp.float64)
        mask = (
            jnp.ones(values.shape, dtype=jnp.bool_)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        if mask.shape != values.shape:
            raise ValueError(
                f"active_mask must have shape {values.shape}; got {mask.shape}."
            )
        scope = Scope()
        values = parse(values, Float[MeasureEntityDim], "weights", scope=scope)
        mask = parse(mask, Bool[MeasureEntityDim], "active_mask", scope=scope)
        tolerance = float(probability_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("probability_tolerance must be finite and non-negative.")
        traced = isinstance(values, jax_core.Tracer) or isinstance(mask, jax_core.Tracer)
        if traced:
            if measure_id is None:
                raise ValueError(
                    "Traced discrete measures require an explicit measure_id."
                )
            identifier = nonempty_identifier("measure_id", measure_id)
        else:
            host_values = np.asarray(weights)
            if not np.issubdtype(host_values.dtype, np.inexact):
                host_values = host_values.astype(np.float64)
            _validate_host_measure(host_values, mask, normalization, tolerance)
            if measure_id is not None:
                identifier = nonempty_identifier("measure_id", measure_id)
            else:
                identifier = resolved_identifier(
                    "measure_id",
                    None,
                    {
                        "kind": "discrete-measure",
                        "name": name_,
                        "support": support_id_,
                        "entity_set": entity_set_id_,
                        "normalization": normalization,
                        "weights": array_tree_fingerprint(host_values),
                        "active_mask": array_tree_fingerprint(np.asarray(mask)),
                    },
                )
        total = jnp.sum(
            jnp.where(mask, values, jnp.zeros((), dtype=values.dtype)), dtype=jnp.float64
        )
        self.name = name_
        self.support_id = support_id_
        self.entity_set_id = entity_set_id_
        self.normalization = normalization
        self.weights = values
        self.active_mask = mask
        self.total_mass = total
        self.measure_id = identifier
        self.probability_tolerance = tolerance

    @property
    def valid(self) -> Bool[Scalar]:
        """Device-resident admissibility of active weights and normalization."""
        active = self.active_mask
        valid = jnp.all(~active | jnp.isfinite(self.weights))
        match self.normalization:
            case "signed":
                return valid
            case "counting":
                return valid & jnp.all(~active | (self.weights == 1))
            case "physical":
                return (
                    valid & jnp.all(~active | (self.weights >= 0)) & (self.total_mass > 0)
                )
            case "probability":
                return (
                    valid
                    & jnp.all(~active | (self.weights >= 0))
                    & (self.total_mass > 0)
                    & jnp.isclose(
                        self.total_mass,
                        1.0,
                        rtol=self.probability_tolerance,
                        atol=self.probability_tolerance,
                    )
                )
            case _ as unreachable:
                assert_never(unreachable)

    def admit(self) -> DiscreteMeasure:
        """Validate numerical admissibility at an explicit eager boundary."""
        if isinstance(self.weights, jax_core.Tracer) or isinstance(
            self.active_mask, jax_core.Tracer
        ):
            raise ValueError("Discrete measure admission requires concrete weights.")
        _validate_host_measure(
            self.weights, self.active_mask, self.normalization, self.probability_tolerance
        )
        return self

    def masked_weights(self, /) -> Array:
        """Return weights with inactive payloads replaced before multiplication."""
        return jnp.where(
            self.active_mask,
            self.weights,
            jnp.zeros((), dtype=self.weights.dtype),
        )

    def integrate(self, values: ArrayLike, /) -> Array:
        """Integrate values whose leading axis is the measure axis."""
        array = jnp.asarray(values)
        if not array.shape or array.shape[0] != self.weights.shape[0]:
            raise ValueError(
                "Integrand leading axis must match the discrete measure capacity."
            )
        safe = jnp.where(
            self.active_mask.reshape(self.active_mask.shape + (1,) * (array.ndim - 1)),
            array,
            jnp.zeros((), dtype=array.dtype),
        )
        weights = self.masked_weights().reshape(
            self.weights.shape + (1,) * (array.ndim - 1)
        )
        return jnp.sum(weights * safe, axis=0)


__all__ = ["DiscreteMeasure", "MeasureNormalization"]

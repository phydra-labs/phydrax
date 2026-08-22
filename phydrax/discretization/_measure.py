#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import nonempty_identifier, resolved_identifier


MeasureNormalization: TypeAlias = Literal[
    "physical",
    "probability",
    "counting",
    "signed",
]


class DiscreteMeasure(StrictModule, NonTrainableState):
    """Identity-bearing finite measure over one discrete entity set."""

    name: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    normalization: MeasureNormalization = eqx.field(static=True)
    weights: Array
    active_mask: Array
    total_mass: float = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)

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
    ):
        name_ = nonempty_identifier("name", name)
        support_id_ = nonempty_identifier("support_id", support_id)
        entity_set_id_ = nonempty_identifier("entity_set_id", entity_set_id)
        if normalization not in ("physical", "probability", "counting", "signed"):
            raise ValueError("Unknown measure normalization.")
        values = np.asarray(weights)
        if values.ndim != 1:
            raise ValueError("Discrete measure weights must be rank-1.")
        if not np.issubdtype(values.dtype, np.inexact):
            values = values.astype(float)
        mask = (
            np.ones(values.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if mask.shape != values.shape:
            raise ValueError(
                f"active_mask must have shape {values.shape}; got {mask.shape}."
            )
        active = values[mask]
        if np.any(~np.isfinite(active)):
            raise ValueError("Active measure weights must be finite.")
        if normalization != "signed" and np.any(active < 0):
            raise ValueError("Unsigned measure weights must be non-negative.")
        if normalization == "counting" and np.any(active != 1):
            raise ValueError("Counting-measure active weights must equal one.")
        total = float(np.sum(active, dtype=np.float64))
        if normalization in ("physical", "probability") and not total > 0:
            raise ValueError("Physical and probability measures require positive mass.")
        tolerance = float(probability_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("probability_tolerance must be finite and non-negative.")
        if normalization == "probability" and not np.isclose(
            total, 1.0, rtol=tolerance, atol=tolerance
        ):
            raise ValueError("Probability-measure active weights must sum to one.")
        self.name = name_
        self.support_id = support_id_
        self.entity_set_id = entity_set_id_
        self.normalization = normalization
        self.weights = jnp.asarray(values)
        self.active_mask = jnp.asarray(mask)
        self.total_mass = total
        self.measure_id = resolved_identifier(
            "measure_id",
            measure_id,
            {
                "kind": "discrete-measure",
                "name": name_,
                "support": support_id_,
                "entity_set": entity_set_id_,
                "normalization": normalization,
                "weights": array_tree_fingerprint(values),
                "active_mask": array_tree_fingerprint(mask),
            },
        )

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
        if not array.shape or int(array.shape[0]) != int(self.weights.shape[0]):
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical ascending energy groups with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import conversion_factor, ENERGY, JOULE, UnitDefinition


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


class EnergyGroupLocation(StrictModule, NonTrainableState):
    """Fixed-shape energy-bin indices and support evidence."""

    indices: Array
    valid: Array


class PreparedEnergyGroupStructure(StrictModule, NonTrainableState):
    """JAX energy-group boundaries in ascending joules."""

    edges_j: Array
    source_id: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    @property
    def group_count(self) -> int:
        return int(self.edges_j.shape[0] - 1)

    def locate(self, energies_j: ArrayLike, /) -> EnergyGroupLocation:
        values = jnp.asarray(energies_j, dtype=self.edges_j.dtype)
        valid = (
            jnp.isfinite(values)
            & (values >= self.edges_j[0])
            & (values <= self.edges_j[-1])
        )
        indices = jnp.searchsorted(self.edges_j, values, side="right") - 1
        indices = jnp.clip(indices, 0, self.group_count - 1)
        return EnergyGroupLocation(indices, valid)


@dataclass(frozen=True, slots=True, init=False)
class EnergyGroupStructure:
    """One immutable energy-group axis, canonicalized to ascending joules."""

    edges_j: np.ndarray
    source_id: str
    group_id: str = field(init=False)

    def __init__(
        self,
        edges: ArrayLike,
        unit: UnitDefinition,
        /,
        *,
        source_id: str,
    ) -> None:
        if not isinstance(unit, UnitDefinition) or unit.dimension != ENERGY:
            raise ValueError("Energy-group edges require an energy unit.")
        factor = float(conversion_factor(unit, JOULE))
        values = np.array(edges, dtype=np.float64, copy=True) * factor
        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                "Energy-group edges must be a rank-one array of size at least two."
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Energy-group edges must be finite and nonnegative.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("Energy-group edges must be strictly increasing.")
        source = _text(source_id, "source_id")
        values.setflags(write=False)
        object.__setattr__(self, "edges_j", values)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "group_id",
            canonical_fingerprint(
                {
                    "kind": "energy-group-structure",
                    "edges_j": array_tree_fingerprint(values),
                    "source": source,
                    "intervals": "left-closed-right-open-last-closed",
                }
            ),
        )

    @property
    def group_count(self) -> int:
        return int(self.edges_j.size - 1)

    def edges_in(self, unit: UnitDefinition, /) -> np.ndarray:
        factor = float(conversion_factor(JOULE, unit))
        result = np.asarray(self.edges_j * factor)
        result.setflags(write=False)
        return result

    def prepare(self) -> PreparedEnergyGroupStructure:
        return PreparedEnergyGroupStructure(
            jnp.asarray(self.edges_j), self.source_id, self.group_id
        )


__all__ = [
    "EnergyGroupLocation",
    "EnergyGroupStructure",
    "PreparedEnergyGroupStructure",
]

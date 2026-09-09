#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....units import AMPERE, convert_value, UnitDefinition


class ElectrodePatch(StrictModule, NonTrainableState):
    """An exact exterior-facet support, not a point electrode or snapped coordinate.

    ``facet_indices`` index ``CellMesh.connectivity.faces``. All facets of a
    patch carry the same prescribed current density. Distinct patches may share
    edges/vertices but not facets. Mesh validation occurs in FinitePatchDCPlan.
    """

    name: str = eqx.field(static=True)
    facet_indices: Array
    patch_id: str = eqx.field(static=True)

    def __init__(self, name: str, facet_indices: ArrayLike, /):
        name_ = str(name).strip()
        indices = np.asarray(facet_indices)
        if (
            not name_
            or indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices > np.iinfo(np.int32).max)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError(
                "Electrode patches require a name and unique nonnegative facet indices."
            )
        self.name = name_
        self.facet_indices = jnp.asarray(np.sort(indices), dtype=jnp.int32)
        self.patch_id = canonical_fingerprint(
            {
                "kind": "dc-electrode-patch",
                "name": name_,
                "facets": array_tree_fingerprint(self.facet_indices),
            }
        )


class ElectricalSurvey(StrictModule, NonTrainableState):
    """Signed, balanced current patterns and gauge-invariant voltage channels.

    ``currents[s, e]`` is integrated current entering the domain through patch e,
    in ``current_unit`` (converted to amperes). ``receiver_weights[m, e]`` is a
    dimensionless signed combination of area-average electrode potentials;
    ``source_indices[m]`` selects its excitation. Each current and receiver row
    must sum to zero. A dipole uses +1 and -1 entries. Arbitrary balanced patterns
    are supported, preserving signs and reciprocity of matched patch supports.
    """

    patches: tuple[ElectrodePatch, ...]
    currents: Array
    receiver_weights: Array
    source_indices: Array
    survey_id: str = eqx.field(static=True)

    def __init__(
        self,
        patches: Sequence[ElectrodePatch],
        currents: ArrayLike,
        receiver_weights: ArrayLike,
        source_indices: ArrayLike,
        /,
        *,
        current_unit: UnitDefinition = AMPERE,
    ):
        patches_ = tuple(patches)
        if len(patches_) < 2 or not all(isinstance(p, ElectrodePatch) for p in patches_):
            raise ValueError("A DC survey requires at least two electrode patches.")
        if len({p.name for p in patches_}) != len(patches_):
            raise ValueError("Electrode names must be unique.")
        if np.iscomplexobj(currents) or np.iscomplexobj(receiver_weights):
            raise TypeError("DC currents and receiver weights must be real.")
        current = np.asarray(
            convert_value(currents, source=current_unit, target=AMPERE), dtype=float
        )
        receivers = np.asarray(receiver_weights, dtype=float)
        sources = np.asarray(source_indices)
        electrode_count = len(patches_)
        if (
            current.ndim != 2
            or current.shape[0] == 0
            or current.shape[1] != electrode_count
        ):
            raise ValueError(
                "currents must have shape (source_count > 0, electrode_count)."
            )
        if (
            receivers.ndim != 2
            or receivers.shape[0] == 0
            or receivers.shape[1] != electrode_count
        ):
            raise ValueError(
                "receiver_weights must have shape (measurement_count > 0, electrode_count)."
            )
        if (
            sources.shape != (receivers.shape[0],)
            or not np.issubdtype(sources.dtype, np.integer)
            or np.any(sources < 0)
            or np.any(sources >= current.shape[0])
        ):
            raise ValueError(
                "source_indices must select one declared source per measurement."
            )
        for values, label in ((current, "current"), (receivers, "receiver")):
            magnitude = np.sum(np.abs(values), axis=1)
            if np.any(~np.isfinite(values)) or np.any(magnitude == 0):
                raise ValueError(f"Every {label} pattern must be finite and nonzero.")
            # Relative to the actual pattern, never to an arbitrary unit current.
            tolerance = 64.0 * np.finfo(float).eps * magnitude
            if np.any(np.abs(np.sum(values, axis=1)) > tolerance):
                raise ValueError(f"Every {label} pattern must be balanced (sum to zero).")
        self.patches = patches_
        self.currents = jnp.asarray(current)
        self.receiver_weights = jnp.asarray(receivers)
        self.source_indices = jnp.asarray(sources, dtype=jnp.int32)
        self.survey_id = canonical_fingerprint(
            {
                "kind": "electrical-survey",
                "patches": [patch.patch_id for patch in patches_],
                "currents": array_tree_fingerprint(self.currents),
                "receivers": array_tree_fingerprint(self.receiver_weights),
                "sources": array_tree_fingerprint(self.source_indices),
            }
        )

    @property
    def source_count(self) -> int:
        return self.currents.shape[0]

    @property
    def measurement_count(self) -> int:
        return self.receiver_weights.shape[0]


__all__ = ["ElectricalSurvey", "ElectrodePatch"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import DifferentiationContract, ScientificArtifactEnvelope
from ...observation import (
    CoordinateLayout,
    CorrelatedGaussianPlan,
    CorrelatedGaussianResult,
    LinearObservationPlan,
    PrecisionCovarianceAction,
    TheoryVector,
)


class CosmologyPhysicalState(StrictModule):
    values: Array
    names: tuple[str, ...] = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    categorical_ids: tuple[str, ...] = eqx.field(static=True)
    state_form_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        names: tuple[str, ...],
        scale_id: str,
        /,
        *,
        categorical_ids: tuple[str, ...] = (),
    ):
        names_ = tuple(str(name).strip() for name in names)
        values_ = jnp.asarray(values).reshape((-1,))
        scale_id_ = str(scale_id).strip()
        categories = tuple(str(value).strip() for value in categorical_ids)
        if (
            not names_
            or len(names_) != values_.size
            or len(set(names_)) != len(names_)
            or any(not name for name in names_)
            or not scale_id_
            or any(not value for value in categories)
        ):
            raise ValueError(
                "Canonical cosmology physical-state coordinates are invalid."
            )
        values_ = eqx.error_if(
            values_,
            jnp.any(~jnp.isfinite(values_)),
            "Cosmology physical parameters must be finite.",
        )
        self.values = values_
        self.names = names_
        self.scale_id = scale_id_
        self.categorical_ids = categories
        self.state_form_id = canonical_fingerprint(
            {
                "kind": "cosmology-physical-state-form",
                "names": list(names_),
                "scale_id": scale_id_,
                "categorical_ids": list(categories),
            }
        )

    def content_id(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "cosmology-physical-state",
                "form": self.state_form_id,
                "values": array_tree_fingerprint(self.values),
            }
        )


class PhysicalDependencyProjection(StrictModule, NonTrainableState):
    names: tuple[str, ...] = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(self, names: tuple[str, ...], /):
        names_ = tuple(str(name).strip() for name in names)
        if (
            not names_
            or len(set(names_)) != len(names_)
            or any(not name for name in names_)
        ):
            raise ValueError("Physical dependency names must be non-empty and unique.")
        self.names = names_
        self.projection_id = canonical_fingerprint(
            {"kind": "physical-dependency-projection", "names": list(names_)}
        )

    def project(self, state: CosmologyPhysicalState, /) -> CosmologyRealizationSignature:
        if not isinstance(state, CosmologyPhysicalState):
            raise TypeError("state must be CosmologyPhysicalState.")
        missing = tuple(name for name in self.names if name not in state.names)
        if missing:
            raise ValueError(f"Physical state is missing projected parameters {missing}.")
        indices = tuple(state.names.index(name) for name in self.names)
        return CosmologyRealizationSignature(
            state.values[jnp.asarray(indices)],
            self.names,
            state.state_form_id,
            state.scale_id,
            self.projection_id,
        )


class CosmologyRealizationSignature(StrictModule):
    parameter_values: Array
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    physical_state_form_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_values: ArrayLike,
        parameter_names: tuple[str, ...],
        physical_state_form_id: str,
        scale_id: str,
        projection_id: str,
        /,
    ):
        names = tuple(str(name).strip() for name in parameter_names)
        values = jnp.asarray(parameter_values).reshape((-1,))
        identities = tuple(
            str(value).strip()
            for value in (physical_state_form_id, scale_id, projection_id)
        )
        if (
            not names
            or len(names) != values.size
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or any(not value for value in identities)
        ):
            raise ValueError("Projected realization contract is invalid.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Projected physical parameters must be finite.",
        )
        self.parameter_values = values
        self.parameter_names = names
        self.physical_state_form_id = identities[0]
        self.scale_id = identities[1]
        self.projection_id = identities[2]

    def require_compatible(
        self, other: CosmologyRealizationSignature, token: ArrayLike, /
    ) -> Array:
        shared = tuple(
            name for name in self.parameter_names if name in other.parameter_names
        )
        if self.scale_id != other.scale_id or not shared:
            raise ValueError(
                "Projected realization contracts have no compatible overlap."
            )
        left = jnp.stack(
            tuple(
                self.parameter_values[self.parameter_names.index(name)] for name in shared
            )
        )
        right = jnp.stack(
            tuple(
                other.parameter_values[other.parameter_names.index(name)]
                for name in shared
            )
        )
        return eqx.error_if(
            jnp.asarray(token),
            jnp.any(left != right),
            "Cosmology products come from different physical realizations.",
        )

    def content_id(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "projected-cosmology-realization",
                "physical_state_form_id": self.physical_state_form_id,
                "projection_id": self.projection_id,
                "scale_id": self.scale_id,
                "names": list(self.parameter_names),
                "values": array_tree_fingerprint(self.parameter_values),
            }
        )


__all__ = [
    "CoordinateLayout",
    "CorrelatedGaussianPlan",
    "CorrelatedGaussianResult",
    "CosmologyPhysicalState",
    "CosmologyRealizationSignature",
    "DifferentiationContract",
    "LinearObservationPlan",
    "PhysicalDependencyProjection",
    "PrecisionCovarianceAction",
    "ScientificArtifactEnvelope",
    "TheoryVector",
]

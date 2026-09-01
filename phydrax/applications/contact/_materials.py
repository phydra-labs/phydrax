#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ContactPairParameters(StrictModule):
    normal_stiffness: Array
    static_friction: Array
    dynamic_friction: Array
    restitution: Array
    adhesion_energy: Array
    thermal_conductance: Array
    electrical_conductance: Array
    wear_coefficient: Array
    hardness: Array
    roughness: Array
    mechanical_available: Array
    transport_available: Array


class ContactMaterialPairTable(StrictModule, NonTrainableState):
    """Explicit material-pair closure parameters without hidden mixing rules."""

    normal_stiffness: Array
    static_friction: Array
    dynamic_friction: Array
    restitution: Array
    adhesion_energy: Array
    thermal_conductance: Array
    electrical_conductance: Array
    wear_coefficient: Array
    hardness: Array
    roughness: Array
    mechanical_available: Array
    transport_available: Array
    material_count: int = eqx.field(static=True)
    symmetric: bool = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        normal_stiffness: ArrayLike,
        /,
        *,
        static_friction: ArrayLike,
        dynamic_friction: ArrayLike,
        restitution: ArrayLike,
        adhesion_energy: ArrayLike,
        thermal_conductance: ArrayLike,
        electrical_conductance: ArrayLike,
        wear_coefficient: ArrayLike,
        hardness: ArrayLike,
        roughness: ArrayLike,
        mechanical_available: ArrayLike | None = None,
        transport_available: ArrayLike | None = None,
        symmetric: bool = True,
    ):
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                normal_stiffness,
                static_friction,
                dynamic_friction,
                restitution,
                adhesion_energy,
                thermal_conductance,
                electrical_conductance,
                wear_coefficient,
                hardness,
                roughness,
            )
        )
        shape = arrays[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0:
            raise ValueError("Contact material parameters require one square table.")
        if any(value.shape != shape for value in arrays[1:]):
            raise ValueError("All contact material parameter tables must agree.")
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise ValueError("Contact material parameters must be finite.")
        (
            stiffness,
            static_mu,
            dynamic_mu,
            restitution_,
            adhesion,
            thermal,
            electrical,
            wear,
            hardness_,
            roughness_,
        ) = arrays
        if (
            np.any(stiffness < 0.0)
            or np.any(static_mu < 0.0)
            or np.any(dynamic_mu < 0.0)
            or np.any(dynamic_mu > static_mu)
            or np.any((restitution_ < 0.0) | (restitution_ > 1.0))
            or np.any(adhesion < 0.0)
            or np.any(thermal < 0.0)
            or np.any(electrical < 0.0)
            or np.any(wear < 0.0)
            or np.any(hardness_ <= 0.0)
            or np.any(roughness_ < 0.0)
        ):
            raise ValueError("Contact material parameters violate physical bounds.")
        mechanical = (
            np.ones(shape, dtype=bool)
            if mechanical_available is None
            else np.asarray(mechanical_available, dtype=bool)
        )
        transport = (
            np.ones(shape, dtype=bool)
            if transport_available is None
            else np.asarray(transport_available, dtype=bool)
        )
        if mechanical.shape != shape or transport.shape != shape:
            raise ValueError("Contact availability masks must match the material table.")
        symmetric_ = bool(symmetric)
        if symmetric_:
            for value in arrays:
                if not np.array_equal(value, value.T):
                    raise ValueError(
                        "Symmetric contact material tables must equal their transpose."
                    )
            if not np.array_equal(mechanical, mechanical.T) or not np.array_equal(
                transport, transport.T
            ):
                raise ValueError(
                    "Symmetric contact availability masks must be symmetric."
                )
        self.normal_stiffness = jnp.asarray(stiffness)
        self.static_friction = jnp.asarray(static_mu)
        self.dynamic_friction = jnp.asarray(dynamic_mu)
        self.restitution = jnp.asarray(restitution_)
        self.adhesion_energy = jnp.asarray(adhesion)
        self.thermal_conductance = jnp.asarray(thermal)
        self.electrical_conductance = jnp.asarray(electrical)
        self.wear_coefficient = jnp.asarray(wear)
        self.hardness = jnp.asarray(hardness_)
        self.roughness = jnp.asarray(roughness_)
        self.mechanical_available = jnp.asarray(mechanical)
        self.transport_available = jnp.asarray(transport)
        self.material_count = int(shape[0])
        self.symmetric = symmetric_
        self.table_id = canonical_fingerprint(
            {
                "kind": "contact-material-pair-table",
                "parameters": array_tree_fingerprint(arrays),
                "mechanical": array_tree_fingerprint(mechanical),
                "transport": array_tree_fingerprint(transport),
                "symmetric": symmetric_,
            }
        )

    @classmethod
    def uniform(
        cls,
        *,
        normal_stiffness: float,
        static_friction: float = 0.0,
        dynamic_friction: float | None = None,
        restitution: float = 0.0,
        adhesion_energy: float = 0.0,
        thermal_conductance: float = 0.0,
        electrical_conductance: float = 0.0,
        wear_coefficient: float = 0.0,
        hardness: float = 1.0,
        roughness: float = 0.0,
    ) -> ContactMaterialPairTable:
        dynamic = static_friction if dynamic_friction is None else dynamic_friction
        scalar = lambda value: np.asarray(((float(value),),))
        return cls(
            scalar(normal_stiffness),
            static_friction=scalar(static_friction),
            dynamic_friction=scalar(dynamic),
            restitution=scalar(restitution),
            adhesion_energy=scalar(adhesion_energy),
            thermal_conductance=scalar(thermal_conductance),
            electrical_conductance=scalar(electrical_conductance),
            wear_coefficient=scalar(wear_coefficient),
            hardness=scalar(hardness),
            roughness=scalar(roughness),
        )

    def lookup(
        self, left_material: ArrayLike, right_material: ArrayLike, /
    ) -> ContactPairParameters:
        left = jnp.asarray(left_material, dtype=jnp.int32)
        right = jnp.asarray(right_material, dtype=jnp.int32)
        if left.shape != right.shape:
            raise ValueError("Contact material lookup indices must agree.")
        safe_left = jnp.clip(left, 0, self.material_count - 1)
        safe_right = jnp.clip(right, 0, self.material_count - 1)
        declared = (
            (left >= 0)
            & (left < self.material_count)
            & (right >= 0)
            & (right < self.material_count)
        )
        gather = lambda table: table[safe_left, safe_right]
        return ContactPairParameters(
            gather(self.normal_stiffness),
            gather(self.static_friction),
            gather(self.dynamic_friction),
            gather(self.restitution),
            gather(self.adhesion_energy),
            gather(self.thermal_conductance),
            gather(self.electrical_conductance),
            gather(self.wear_coefficient),
            gather(self.hardness),
            gather(self.roughness),
            declared & gather(self.mechanical_available),
            declared & gather(self.transport_available),
        )


__all__ = ["ContactMaterialPairTable", "ContactPairParameters"]

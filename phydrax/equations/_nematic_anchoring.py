#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._nematic import NematicTensorBasis


class NematicAnchoringKind(StrEnum):
    FIXED = "fixed"
    HOMEOTROPIC = "homeotropic"
    PLANAR_DEGENERATE = "planar_degenerate"


class NematicAnchoringFields(StrictModule):
    energy_density: Array
    molecular_field: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class NematicAnchoringPlan(StrictModule, NonTrainableState):
    basis: NematicTensorBasis
    kind: NematicAnchoringKind = eqx.field(static=True)
    boundary_mask: Array
    preferred_compact: Array
    normals: Array
    strength: Array
    scalar_order: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: NematicTensorBasis,
        kind: NematicAnchoringKind,
        boundary_mask: ArrayLike,
        /,
        *,
        preferred_compact: ArrayLike | None = None,
        normals: ArrayLike | None = None,
        strength: ArrayLike = 1.0,
        scalar_order: ArrayLike = 1.0,
    ):
        if not isinstance(basis, NematicTensorBasis):
            raise TypeError("basis must be NematicTensorBasis.")
        if not isinstance(kind, NematicAnchoringKind):
            raise TypeError("kind must be NematicAnchoringKind.")
        mask = jnp.asarray(boundary_mask, dtype=bool)
        strength_ = jnp.asarray(strength)
        order = jnp.asarray(scalar_order, dtype=strength_.dtype)
        if mask.ndim < 1 or strength_.shape != () or order.shape != ():
            raise ValueError("Anchoring mask and parameters have incompatible shapes.")
        normal_values = (
            jnp.zeros(mask.shape + (basis.orientation_dimension,), dtype=strength_.dtype)
            if normals is None
            else jnp.asarray(normals, dtype=strength_.dtype)
        )
        if normal_values.shape != mask.shape + (basis.orientation_dimension,):
            raise ValueError(
                "Anchoring normals must match mask and orientation dimension."
            )
        norm = jnp.sqrt(jnp.sum(normal_values * normal_values, axis=-1))
        normal_values = jnp.where(
            mask[..., None],
            normal_values / jnp.where(norm > 0.0, norm, 1.0)[..., None],
            0.0,
        )
        if kind is NematicAnchoringKind.FIXED:
            if preferred_compact is None:
                raise ValueError("Fixed anchoring requires preferred_compact.")
            preferred = jnp.asarray(preferred_compact, dtype=strength_.dtype)
        elif kind is NematicAnchoringKind.HOMEOTROPIC:
            identity = jnp.eye(basis.orientation_dimension, dtype=strength_.dtype)
            preferred_tensor = order * (
                contract("...i,...j->...ij", normal_values, normal_values)
                - identity / basis.orientation_dimension
            )
            preferred = basis.encode(preferred_tensor)
        else:
            preferred = jnp.zeros(
                mask.shape + (basis.component_count,), dtype=strength_.dtype
            )
        if preferred.shape != mask.shape + (basis.component_count,):
            raise ValueError("preferred_compact must match mask and basis components.")
        self.basis = basis
        self.kind = kind
        self.boundary_mask = mask
        self.preferred_compact = preferred
        self.normals = normal_values
        self.strength = strength_
        self.scalar_order = order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nematic-anchoring",
                "basis": basis.basis_id,
                "anchoring_kind": kind.value,
                "shape": list(mask.shape),
                "strength": float(strength_),
                "scalar_order": float(order),
            }
        )

    def evaluate(self, compact_q: ArrayLike, /) -> NematicAnchoringFields:
        compact = jnp.asarray(compact_q)
        if compact.shape != self.boundary_mask.shape + (self.basis.component_count,):
            raise ValueError("compact_q must match anchoring mask and basis.")
        if self.kind in (
            NematicAnchoringKind.FIXED,
            NematicAnchoringKind.HOMEOTROPIC,
        ):
            difference = compact - self.preferred_compact
            energy = 0.5 * self.strength * jnp.sum(difference * difference, axis=-1)
            molecular = -self.strength * difference
        else:
            tensor = self.basis.decode(compact)
            q_normal = contract("...ij,...j->...i", tensor, self.normals)
            target = -self.scalar_order * self.normals / self.basis.orientation_dimension
            difference = q_normal - target
            energy = 0.5 * self.strength * jnp.sum(difference * difference, axis=-1)
            derivative = (
                0.5
                * self.strength
                * (
                    contract("...i,...j->...ij", difference, self.normals)
                    + contract("...i,...j->...ij", self.normals, difference)
                )
            )
            molecular = -self.basis.encode(self.basis.project(derivative))
        energy = jnp.where(self.boundary_mask, energy, 0.0)
        molecular = jnp.where(self.boundary_mask[..., None], molecular, 0.0)
        successful = (
            jnp.isfinite(self.strength)
            & (self.strength >= 0.0)
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(molecular))
            & jnp.all(
                (~self.boundary_mask)
                | (jnp.sqrt(jnp.sum(self.normals * self.normals, axis=-1)) > 0.0)
            )
        )
        return NematicAnchoringFields(
            energy,
            molecular,
            successful,
            self.plan_id,
        )


__all__ = [
    "NematicAnchoringFields",
    "NematicAnchoringKind",
    "NematicAnchoringPlan",
]

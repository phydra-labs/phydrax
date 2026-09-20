#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical-cell periodic features for finite electronic amplitudes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell


class PeriodicCellFeatureResult(StrictModule):
    """Fixed-shape periodic features and the selected physical image branch."""

    fractional_coordinates: Array
    wrapped_coordinates: Array
    coordinate_images: Array
    reciprocal_features: Array
    pair_displacements: Array
    pair_distances: Array
    pair_image_shifts: Array
    twist_phase: Array
    valid: Array


class PeriodicCellFeatures(StrictModule, NonTrainableState):
    """Evaluate periodic electronic features in a certified physical cell.

    ``reciprocal_modes`` are integer coefficients of the reciprocal lattice, not
    Cartesian wavevectors. Inputs are Cartesian coordinates. Pair displacements
    use :class:`PeriodicCell`'s metric-aware bounded image search, including for
    oblique cells. Image and wrapping choices are stopped discrete decisions;
    coordinate derivatives are meaningful only while those choices stay fixed.
    """

    cell: PeriodicCell
    reciprocal_modes: Array
    twist: Array
    reciprocal_mode_count: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    differentiation_scope: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        reciprocal_modes: ArrayLike,
        /,
        *,
        twist: ArrayLike,
    ):
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        mode_host = np.asarray(reciprocal_modes)
        if (
            mode_host.ndim != 2
            or mode_host.shape[0] < 1
            or mode_host.shape[1] != cell.rank
        ):
            raise ValueError(
                "reciprocal_modes must have shape (modes, cell.rank) with at least one mode."
            )
        if not np.issubdtype(mode_host.dtype, np.integer):
            raise TypeError("reciprocal_modes must contain integer lattice coefficients.")
        mode_host = mode_host.astype(np.int32, copy=False)
        twist_host = np.asarray(twist, dtype=np.asarray(cell.vectors).dtype)
        if twist_host.shape != (cell.rank,) or np.any(~np.isfinite(twist_host)):
            raise ValueError(
                "twist must be a finite vector with one entry per periodic axis."
            )
        boundary_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-boundary",
                "cell": cell.cell_id,
                "twist": array_tree_fingerprint(twist_host),
            }
        )
        self.cell = cell
        self.reciprocal_modes = jnp.asarray(mode_host, dtype=jnp.int32)
        self.twist = jnp.asarray(twist_host)
        self.reciprocal_mode_count = mode_host.shape[0]
        self.boundary_id = boundary_id
        self.differentiation_scope = (
            "fixed-cell coordinate derivatives within fixed wrap and minimum-image "
            "selections; no gradient across image selection"
        )

    @property
    def ambient_dimension(self) -> int:
        return self.cell.ambient_dimension

    def __call__(self, coordinates: ArrayLike, /) -> PeriodicCellFeatureResult:
        value = jnp.asarray(coordinates)
        if value.ndim < 2 or value.shape[-1] != self.cell.ambient_dimension:
            raise ValueError(
                "Periodic feature inputs must end in (particles, cell.ambient_dimension)."
            )
        fractional = self.cell.fractional(value)
        wrapped, coordinate_images = self.cell.wrap(value)
        reciprocal_argument = (
            2.0
            * jnp.pi
            * contract(
                "...ir,mr->...im",
                fractional,
                self.reciprocal_modes.astype(value.dtype),
                backend="jax",
            )
        )
        reciprocal_features = jnp.exp(1.0j * reciprocal_argument)
        displacement = value[..., :, None, :] - value[..., None, :, :]
        minimum_displacement = self.cell.minimum_image(displacement)
        pair_squared_distance = jnp.sum(
            minimum_displacement * minimum_displacement, axis=-1
        )
        identity = jnp.eye(value.shape[-2], dtype=jnp.bool_)
        pair_distances = jnp.sqrt(jnp.where(identity, 1.0, pair_squared_distance))
        pair_distances = jnp.where(identity, 0.0, pair_distances)
        selected_translation = displacement - minimum_displacement
        pair_image_shifts = jax.lax.stop_gradient(
            jnp.rint(self.cell.fractional(selected_translation)).astype(jnp.int32)
        )
        twist_argument = jnp.sum(
            fractional * self.twist.astype(value.dtype), axis=(-2, -1)
        )
        twist_phase = jnp.exp(1.0j * twist_argument)
        valid = (
            jnp.all(jnp.isfinite(value), axis=(-2, -1))
            & jnp.all(jnp.isfinite(pair_distances), axis=(-2, -1))
            & jnp.all(jnp.isfinite(reciprocal_features), axis=(-2, -1))
        )
        return PeriodicCellFeatureResult(
            fractional_coordinates=fractional,
            wrapped_coordinates=wrapped,
            coordinate_images=coordinate_images,
            reciprocal_features=reciprocal_features,
            pair_displacements=minimum_displacement,
            pair_distances=pair_distances,
            pair_image_shifts=pair_image_shifts,
            twist_phase=twist_phase,
            valid=valid,
        )


__all__ = ["PeriodicCellFeatureResult", "PeriodicCellFeatures"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._discretization import LatticeBoltzmannDiscretization


def _default_body_names(count: int, /) -> tuple[str, ...]:
    if count == 0:
        return ()
    if count == 1:
        return ("__solid__",)
    return tuple(f"body:{index}" for index in range(count))


class FixedSDFLinkGeometry(StrictModule, NonTrainableState):
    """Frozen exact-SDF intersections for every fluid-to-solid lattice link.

    Fractions measure distance from the fluid cell centre toward the solid source
    in lattice-link units. They are compiled once and never regenerated at run time.
    """

    fluid_mask: Array
    signed_distance: Array
    body_labels: Array
    body_index: Array
    link_fraction: Array
    body_names: tuple[str, ...] = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        signed_distance: ArrayLike,
        /,
        *,
        body_labels: ArrayLike | None = None,
        body_names: Sequence[str] | None = None,
        link_fractions: ArrayLike | None = None,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("Fixed SDF link geometry requires an LBM discretization.")
        shape = discretization.grid.shape
        phi = np.asarray(signed_distance, dtype=np.float64)
        if phi.shape != shape or np.any(~np.isfinite(phi)):
            raise ValueError("signed_distance must be finite and match the LBM grid.")
        if np.any(phi == 0.0):
            raise ValueError(
                "Cell centres may not lie exactly on the frozen SDF interface."
            )
        fluid = phi > 0.0
        if not np.any(fluid):
            raise ValueError("Fixed SDF geometry must contain at least one fluid cell.")

        if body_labels is None:
            labels = np.where(fluid, -1, 0).astype(np.int32)
        else:
            labels = np.asarray(body_labels, dtype=np.int32)
            if labels.shape != shape:
                raise ValueError("body_labels must match the LBM grid shape.")
            if np.any(labels[fluid] != -1) or np.any(labels[~fluid] < 0):
                raise ValueError("Fluid labels must be -1 and solid labels nonnegative.")
        label_count = 0 if np.all(fluid) else int(np.max(labels[~fluid])) + 1
        if np.any(np.unique(labels[~fluid]) != np.arange(label_count)):
            raise ValueError("Solid body labels must be contiguous from zero.")
        names = (
            _default_body_names(label_count)
            if body_names is None
            else tuple(str(name) for name in body_names)
        )
        if len(names) != label_count or any(not name for name in names):
            raise ValueError(
                "body_names must provide one non-empty name per solid label."
            )
        if len(set(names)) != len(names):
            raise ValueError("body_names must be unique.")

        velocities = discretization.velocity_set.velocity_tuples
        q_count = len(velocities)
        population_shape = (*shape, q_count)
        blocked = np.zeros(population_shape, dtype=bool)
        link_body = np.full(population_shape, -1, dtype=np.int32)
        inferred = np.full(population_shape, np.nan, dtype=np.float64)
        for cell in np.ndindex(shape):
            if not fluid[cell]:
                continue
            for direction, velocity in enumerate(velocities):
                if not any(velocity):
                    continue
                source_values = []
                supported = True
                for axis, component in enumerate(velocity):
                    source = cell[axis] - component
                    if source < 0 or source >= shape[axis]:
                        if discretization.periodic[axis]:
                            source %= shape[axis]
                        else:
                            supported = False
                            break
                    source_values.append(source)
                if not supported:
                    continue
                source_cell = tuple(source_values)
                if fluid[source_cell]:
                    continue
                blocked[cell + (direction,)] = True
                link_body[cell + (direction,)] = labels[source_cell]
                denominator = phi[cell] - phi[source_cell]
                fraction = phi[cell] / denominator
                if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                    raise ValueError(
                        "Exact-SDF link intersection lies outside its lattice link."
                    )
                inferred[cell + (direction,)] = fraction

        if link_fractions is None:
            fractions = inferred
        else:
            fractions = np.asarray(link_fractions, dtype=np.float64)
            if fractions.shape != population_shape:
                raise ValueError(f"link_fractions must have shape {population_shape}.")
            if np.any(~np.isfinite(fractions[blocked])) or np.any(
                (fractions[blocked] <= 0.0) | (fractions[blocked] > 1.0)
            ):
                raise ValueError("Every blocked link fraction must lie in (0, 1].")
            if np.any(np.isfinite(fractions[~blocked])):
                raise ValueError(
                    "Link fractions may only be supplied on fluid-to-solid links."
                )

        self.fluid_mask = jnp.asarray(fluid, dtype=bool)
        self.signed_distance = jnp.asarray(phi)
        self.body_labels = jnp.asarray(labels, dtype=jnp.int32)
        self.body_index = jnp.asarray(link_body, dtype=jnp.int32)
        self.link_fraction = jnp.asarray(fractions)
        self.body_names = names
        self.discretization_id = discretization.prepared_id
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "fixed-sdf-lattice-link-geometry",
                "discretization": discretization.prepared_id,
                "signed_distance": array_tree_fingerprint(phi),
                "body_labels": array_tree_fingerprint(labels),
                "body_names": list(names),
                "link_fraction": array_tree_fingerprint(fractions),
            }
        )


__all__ = ["FixedSDFLinkGeometry"]

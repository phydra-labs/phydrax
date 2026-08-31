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
from ...geometry import CompiledGeometry, GeometryCapability
from ._discretization import LatticeBoltzmannDiscretization


def _default_body_names(count: int, /) -> tuple[str, ...]:
    if count == 0:
        return ()
    if count == 1:
        return ("__solid__",)
    return tuple(f"body:{index}" for index in range(count))


class FixedSDFLinkGeometry(StrictModule, NonTrainableState):
    """Frozen SDF-reconstructed intersections for fluid-to-solid lattice links.

    Fractions linearly reconstruct the signed-distance zero from the fluid cell
    centre toward the solid source. They never regenerate at run time.
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


class LatticeBoltzmannGeometryImportEvidence(StrictModule, NonTrainableState):
    blocked_link_count: int = eqx.field(static=True)
    fluid_cell_count: int = eqx.field(static=True)
    solid_cell_count: int = eqx.field(static=True)
    minimum_cell_distance_margin: float = eqx.field(static=True)
    maximum_normal_residual: float = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedLatticeBoltzmannLinkGeometry(StrictModule, NonTrainableState):
    link_geometry: FixedSDFLinkGeometry
    boundary_fraction: Array
    boundary_normals: Array
    evidence: LatticeBoltzmannGeometryImportEvidence
    source_id: str = eqx.field(static=True)


def prepare_lattice_boltzmann_link_geometry(
    discretization: LatticeBoltzmannDiscretization,
    geometry: CompiledGeometry,
    /,
    *,
    fluid_region: str = "outside",
    body_name: str = "body",
) -> PreparedLatticeBoltzmannLinkGeometry:
    """Compile exact signed-distance geometry into fixed per-link LBM metadata."""

    if not isinstance(discretization, LatticeBoltzmannDiscretization):
        raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    if geometry.ambient_dimension != discretization.velocity_set.dimension:
        raise ValueError("Geometry and lattice dimensions do not match.")
    if fluid_region not in ("outside", "inside"):
        raise ValueError("fluid_region must be 'outside' or 'inside'.")
    name = str(body_name)
    if not name:
        raise ValueError("body_name must be nonempty.")
    geometry.require(GeometryCapability.SIGNED_DISTANCE)
    geometry.require(GeometryCapability.BOUNDARY_NORMAL)
    points = jnp.asarray(discretization.grid.points)
    signed_distance = np.asarray(geometry.signed_distance(points), dtype=np.float64)
    shape = discretization.grid.shape
    dimension = discretization.velocity_set.dimension
    if signed_distance.shape != (discretization.grid.size,):
        raise ValueError("Geometry signed distance must return one scalar per cell.")
    if fluid_region == "inside":
        signed_distance = -signed_distance
    signed_distance = signed_distance.reshape(shape)
    cell_points = np.asarray(points, dtype=np.float64).reshape(shape + (dimension,))
    link_geometry = FixedSDFLinkGeometry(
        discretization,
        signed_distance,
        body_names=(name,),
    )
    raw_fractions = np.asarray(link_geometry.link_fraction)
    blocked = np.isfinite(raw_fractions)
    fractions = np.where(blocked, raw_fractions, 0.0)
    velocities = np.asarray(
        discretization.velocity_set.velocities,
        dtype=np.float64,
    )
    boundary_points = cell_points[..., None, :] - (
        fractions[..., None]
        * float(discretization.cell_size)
        * velocities.reshape((1,) * len(shape) + velocities.shape)
    )
    active_normals = np.asarray(
        geometry.boundary_normal(jnp.asarray(boundary_points[blocked])),
        dtype=np.float64,
    )
    if active_normals.shape != (int(np.count_nonzero(blocked)), dimension):
        raise ValueError("Geometry normal must return one vector per blocked link.")
    if fluid_region == "inside":
        active_normals = -active_normals
    active_normal_norm = np.sqrt(np.sum(active_normals**2, axis=-1))
    safe_norm = np.where(active_normal_norm > 0.0, active_normal_norm, 1.0)
    link_normals = np.zeros((*shape, len(velocities), dimension), dtype=np.float64)
    link_normals[blocked] = active_normals / safe_norm[..., None]
    active_normal_residual = np.abs(np.sqrt(np.sum(link_normals**2, axis=-1)) - 1.0)
    maximum_normal_residual = float(np.max(active_normal_residual[blocked], initial=0.0))
    minimum_margin = float(np.min(np.abs(signed_distance)))
    finite = bool(
        np.all(np.isfinite(signed_distance))
        and np.all(np.isfinite(link_normals))
        and np.all(np.isfinite(fractions))
    )
    passed = bool(
        finite
        and minimum_margin > 0.0
        and maximum_normal_residual <= 1.0e-10
        and np.all((fractions[blocked] > 0.0) & (fractions[blocked] <= 1.0))
        and np.all(fractions[~blocked] == 0.0)
    )
    source_id = canonical_fingerprint(
        {
            "kind": "compiled-geometry-to-lattice-boltzmann-links",
            "kernel": type(geometry.kernel).__name__,
            "schema": repr(geometry.schema),
            "tolerance": repr(geometry.tolerance),
            "state": array_tree_fingerprint(
                tuple(np.asarray(value) for value in geometry.state.values)
            ),
            "fluid_region": fluid_region,
            "body_name": name,
            "link_geometry": link_geometry.geometry_id,
        }
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "lattice-boltzmann-geometry-import-evidence",
            "source": source_id,
            "blocked_links": int(np.count_nonzero(blocked)),
            "fluid_cells": int(np.count_nonzero(link_geometry.fluid_mask)),
            "solid_cells": int(np.count_nonzero(~np.asarray(link_geometry.fluid_mask))),
            "minimum_margin": minimum_margin,
            "maximum_normal_residual": maximum_normal_residual,
            "finite": finite,
            "passed": passed,
        }
    )
    evidence = LatticeBoltzmannGeometryImportEvidence(
        int(np.count_nonzero(blocked)),
        int(np.count_nonzero(link_geometry.fluid_mask)),
        int(np.count_nonzero(~np.asarray(link_geometry.fluid_mask))),
        minimum_margin,
        maximum_normal_residual,
        finite,
        passed,
        evidence_id,
    )
    if not passed:
        raise ValueError("Compiled geometry failed LBM link metadata certification.")
    return PreparedLatticeBoltzmannLinkGeometry(
        link_geometry,
        jnp.asarray(fractions),
        jnp.asarray(link_normals),
        evidence,
        source_id,
    )


__all__ = [
    "FixedSDFLinkGeometry",
    "LatticeBoltzmannGeometryImportEvidence",
    "PreparedLatticeBoltzmannLinkGeometry",
    "prepare_lattice_boltzmann_link_geometry",
]

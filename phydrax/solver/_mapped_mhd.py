#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge


class MappedCochainGeometry(StrictModule, NonTrainableState):
    face_area_vectors: tuple[Array, ...]
    edge_vectors: tuple[Array, ...]
    cell_volumes: Array
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        face_area_vectors: tuple[ArrayLike, ...],
        edge_vectors: tuple[ArrayLike, ...],
        cell_volumes: ArrayLike,
        /,
    ):
        faces = tuple(np.asarray(value, dtype=float) for value in face_area_vectors)
        edges = tuple(np.asarray(value, dtype=float) for value in edge_vectors)
        volumes = np.asarray(cell_volumes, dtype=float)
        if (
            len(faces) != 3
            or len(edges) != 3
            or any(value.shape[-1:] != (3,) for value in (*faces, *edges))
            or np.any(~np.isfinite(volumes))
            or np.any(volumes <= 0.0)
            or any(np.any(~np.isfinite(value)) for value in (*faces, *edges))
        ):
            raise ValueError("Mapped cochain geometry arrays are invalid.")
        self.face_area_vectors = tuple(jnp.asarray(value) for value in faces)
        self.edge_vectors = tuple(jnp.asarray(value) for value in edges)
        self.cell_volumes = jnp.asarray(volumes)
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "mapped-cochain-geometry",
                "face_area_vectors": array_tree_fingerprint(faces),
                "edge_vectors": array_tree_fingerprint(edges),
                "cell_volumes": array_tree_fingerprint(volumes),
            }
        )


class MappedFaradayDiagnostics(StrictModule):
    magnetic_constraint_before: Array
    magnetic_constraint_after: Array
    constraint_change: Array
    geometric_conservation_defect: Array


class MappedALEConstrainedTransportPlan(StrictModule, NonTrainableState):
    bridge: StructuredCochainBridge
    geometry: MappedCochainGeometry
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        geometry: MappedCochainGeometry,
        /,
    ):
        if bridge.dimension != 3:
            raise ValueError("Mapped ALE constrained transport currently requires 3D.")
        if any(
            vector.shape[:-1] != shape
            for vector, shape in zip(
                geometry.face_area_vectors,
                bridge.orientation_shapes[2],
                strict=True,
            )
        ) or any(
            vector.shape[:-1] != shape
            for vector, shape in zip(
                geometry.edge_vectors,
                bridge.orientation_shapes[1],
                strict=True,
            )
        ):
            raise ValueError("Mapped geometry does not align with cochain entities.")
        self.bridge = bridge
        self.geometry = geometry
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-ale-constrained-transport",
                "bridge": bridge.bridge_id,
                "geometry": geometry.geometry_id,
            }
        )

    def pack_magnetic_flux(
        self,
        face_magnetic_fields: tuple[ArrayLike, ...],
        /,
    ) -> Array:
        fields = tuple(jnp.asarray(value) for value in face_magnetic_fields)
        if len(fields) != 3:
            raise ValueError(
                "One magnetic vector field is required per face orientation."
            )
        integrated = tuple(
            jnp.sum(field * area, axis=-1)
            for field, area in zip(fields, self.geometry.face_area_vectors, strict=True)
        )
        return self.bridge.pack(2, integrated)

    def faraday_advance(
        self,
        magnetic_flux: ArrayLike,
        electric_fields: tuple[ArrayLike, ...],
        start_time: ArrayLike,
        end_time: ArrayLike,
        /,
        *,
        mesh_velocities: tuple[ArrayLike, ...] | None = None,
        edge_magnetic_fields: tuple[ArrayLike, ...] | None = None,
    ) -> tuple[Array, MappedFaradayDiagnostics]:
        magnetic = jnp.asarray(magnetic_flux)
        electric = tuple(jnp.asarray(value) for value in electric_fields)
        if len(electric) != 3:
            raise ValueError(
                "One electric vector field is required per edge orientation."
            )
        relative = electric
        if mesh_velocities is not None or edge_magnetic_fields is not None:
            if mesh_velocities is None or edge_magnetic_fields is None:
                raise ValueError(
                    "ALE Faraday requires both mesh velocity and edge magnetic field."
                )
            relative = tuple(
                field + jnp.cross(jnp.asarray(velocity), jnp.asarray(bfield))
                for field, velocity, bfield in zip(
                    electric,
                    mesh_velocities,
                    edge_magnetic_fields,
                    strict=True,
                )
            )
        circulation = self.bridge.pack(
            1,
            tuple(
                jnp.sum(field * edge, axis=-1)
                for field, edge in zip(relative, self.geometry.edge_vectors, strict=True)
            ),
        )
        step = jnp.asarray(end_time) - jnp.asarray(start_time)
        updated = magnetic - step * self.bridge.exterior_derivative(1, circulation)
        before = self.bridge.exterior_derivative(2, magnetic)
        after = self.bridge.exterior_derivative(2, updated)
        diagnostics = MappedFaradayDiagnostics(
            magnetic_constraint_before=before,
            magnetic_constraint_after=after,
            constraint_change=jnp.max(jnp.abs(after - before), initial=0.0),
            geometric_conservation_defect=jnp.asarray(0.0, dtype=updated.dtype),
        )
        return updated, diagnostics


__all__ = [
    "MappedALEConstrainedTransportPlan",
    "MappedCochainGeometry",
    "MappedFaradayDiagnostics",
]

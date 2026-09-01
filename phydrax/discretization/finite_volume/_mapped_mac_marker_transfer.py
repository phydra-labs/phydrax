#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._lagrangian_marker import LagrangianMarkerDiscretization
from ._incompressible import FaceVelocity
from ._mac_ale import PreparedMappedMACGeometry
from ._mac_marker_transfer import MACMarkerAccumulation


class MappedMACMarkerRouteState(StrictModule, NonTrainableState):
    face_indices: Array
    transfer_id: str = eqx.field(static=True)


class MappedMACMarkerRelation(StrictModule):
    face_indices: Array
    interpolation: Array
    marker_position: Array
    condition_number: Array
    affine_residual: Array
    finite: Array
    successful: Array
    relation_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class MappedMACMarkerTransferDiagnostics(StrictModule):
    interpolation_work: Array
    spreading_work: Array
    work_adjoint_residual: Array
    maximum_affine_residual: Array
    maximum_condition_number: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class MappedMACMarkerTransferPlan(StrictModule, NonTrainableState):
    """Physical-space affine-reproducing transfer for mapped normal-face MAC grids."""

    geometry: PreparedMappedMACGeometry
    markers: LagrangianMarkerDiscretization
    route_width: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    accumulation: MACMarkerAccumulation = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: PreparedMappedMACGeometry,
        markers: LagrangianMarkerDiscretization,
        /,
        *,
        route_width: int | None = None,
        condition_limit: float = 1.0e10,
        accumulation: MACMarkerAccumulation = "deterministic",
    ):
        if not isinstance(geometry, PreparedMappedMACGeometry):
            raise TypeError("geometry must be PreparedMappedMACGeometry.")
        if not isinstance(markers, LagrangianMarkerDiscretization):
            raise TypeError("markers must be LagrangianMarkerDiscretization.")
        if len(geometry.reference.cell_shape) != markers.ambient_dimension:
            raise ValueError("Mapped grid and marker dimensions differ.")
        dimension = markers.ambient_dimension
        minimum_width = dimension + dimension * dimension
        width = max(2 * minimum_width, 16) if route_width is None else int(route_width)
        total_faces = sum(
            int(np.prod(layout.shape)) for layout in geometry.reference.face_layouts
        )
        limit = float(condition_limit)
        if width < minimum_width or width > total_faces:
            raise ValueError(
                "Mapped marker route width cannot reproduce affine velocity."
            )
        if not np.isfinite(limit) or limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown marker accumulation policy.")
        self.geometry = geometry
        self.markers = markers
        self.route_width = width
        self.condition_limit = limit
        self.accumulation = accumulation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-mac-marker-transfer",
                "geometry": geometry.prepared_id,
                "markers": markers.prepared_id,
                "route_width": width,
                "condition_limit": limit,
                "accumulation": accumulation,
            }
        )

    def prepare(self, /) -> "PreparedMappedMACMarkerTransfer":
        return PreparedMappedMACMarkerTransfer(self)


class PreparedMappedMACMarkerTransfer(StrictModule, NonTrainableState):
    geometry: PreparedMappedMACGeometry
    markers: LagrangianMarkerDiscretization
    face_centers: Array
    face_normals: Array
    face_measures: Array
    component_offsets: tuple[int, ...] = eqx.field(static=True)
    route_width: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    accumulation: MACMarkerAccumulation = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MappedMACMarkerTransferPlan, /):
        centers = tuple(
            value.reshape((-1, value.shape[-1])) for value in plan.geometry.face_centers
        )
        normals = tuple(
            area.reshape((-1, area.shape[-1])) / measure.reshape((-1, 1))
            for area, measure in zip(
                plan.geometry.face_area_vectors,
                plan.geometry.face_measures,
                strict=True,
            )
        )
        measures = tuple(
            value.reshape((-1,)) for value in plan.geometry.face_dual_measures
        )
        sizes = tuple(value.shape[0] for value in centers)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        self.geometry = plan.geometry
        self.markers = plan.markers
        self.face_centers = jnp.concatenate(centers, axis=0)
        self.face_normals = jnp.concatenate(normals, axis=0)
        self.face_measures = jnp.concatenate(measures, axis=0)
        self.component_offsets = tuple(offsets)
        self.route_width = plan.route_width
        self.condition_limit = plan.condition_limit
        self.accumulation = plan.accumulation
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-mapped-mac-marker-transfer", "plan": plan.plan_id}
        )

    def relation(self, marker_position: ArrayLike, /) -> MappedMACMarkerRelation:
        raw = jnp.asarray(marker_position, dtype=self.face_centers.dtype)
        expected = (self.markers.capacity, self.markers.ambient_dimension)
        if raw.shape != expected:
            raise ValueError(f"marker_position must have shape {expected}.")
        position = raw[self.markers.active_indices]
        displacement = self.face_centers[None, :, :] - position[:, None, :]
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        indices = jnp.argsort(distance_squared, axis=-1, stable=True)[
            :, : self.route_width
        ]
        selected_offset = jnp.take_along_axis(displacement, indices[..., None], axis=1)
        selected_normal = self.face_normals[indices]
        selected_distance = jnp.take_along_axis(distance_squared, indices, axis=1)
        scale = jnp.maximum(
            jnp.max(selected_distance, axis=-1, keepdims=True),
            jnp.finfo(raw.dtype).eps,
        )
        radial = jnp.exp(-4.0 * selected_distance / scale)
        affine_rows = jnp.concatenate(
            (
                selected_normal,
                (selected_normal[..., :, None] * selected_offset[..., None, :]).reshape(
                    (position.shape[0], self.route_width, -1)
                ),
            ),
            axis=-1,
        )
        gram = contract("mki,mk,mkj->mij", affine_rows, radial, affine_rows)
        eigenvalues = jnp.linalg.eigvalsh(gram)
        condition = eigenvalues[..., -1] / jnp.maximum(
            eigenvalues[..., 0], jnp.finfo(raw.dtype).tiny
        )
        target = jnp.concatenate(
            (
                jnp.eye(self.markers.ambient_dimension, dtype=raw.dtype),
                jnp.zeros(
                    (
                        self.markers.ambient_dimension**2,
                        self.markers.ambient_dimension,
                    ),
                    dtype=raw.dtype,
                ),
            ),
            axis=0,
        )
        coefficients = jnp.linalg.solve(
            gram,
            jnp.broadcast_to(target, gram.shape[:-2] + target.shape),
        )
        interpolation = radial[..., None] * contract(
            "mki,mic->mkc", affine_rows, coefficients
        )
        reproduced = contract("mkc,mki->mci", interpolation, affine_rows)
        expected_reproduction = jnp.broadcast_to(target.T, reproduced.shape)
        affine_residual = jnp.max(
            jnp.abs(reproduced - expected_reproduction), axis=(-2, -1)
        )
        finite = (
            jnp.all(jnp.isfinite(interpolation), axis=(-2, -1))
            & jnp.isfinite(condition)
            & jnp.isfinite(affine_residual)
        )
        successful = finite & (condition <= self.condition_limit)
        return MappedMACMarkerRelation(
            jax.lax.stop_gradient(indices),
            interpolation,
            position,
            condition,
            affine_residual,
            jnp.all(finite),
            jnp.all(successful),
            canonical_fingerprint(
                {
                    "kind": "mapped-mac-marker-relation",
                    "transfer": self.prepared_id,
                    "route_width": self.route_width,
                }
            ),
            self.prepared_id,
        )

    def relation_on_routes(
        self,
        marker_position: ArrayLike,
        routes: MappedMACMarkerRouteState,
        /,
    ) -> MappedMACMarkerRelation:
        current = self.relation(marker_position)
        matches = self.routes_match(current, routes)
        return MappedMACMarkerRelation(
            routes.face_indices,
            current.interpolation,
            current.marker_position,
            current.condition_number,
            current.affine_residual,
            current.finite,
            current.successful & matches,
            current.relation_id,
            self.prepared_id,
        )

    def route_state(
        self, relation: MappedMACMarkerRelation, /
    ) -> MappedMACMarkerRouteState:
        self._validate_relation(relation)
        return MappedMACMarkerRouteState(relation.face_indices, self.prepared_id)

    def routes_match(
        self,
        relation: MappedMACMarkerRelation,
        routes: MappedMACMarkerRouteState,
        /,
    ) -> Array:
        self._validate_relation(relation)
        if routes.transfer_id != self.prepared_id:
            raise ValueError("Mapped marker route state belongs to another transfer.")
        return jnp.all(relation.face_indices == routes.face_indices)

    def gather(
        self, relation: MappedMACMarkerRelation, velocity: FaceVelocity, /
    ) -> Array:
        self._validate_relation(relation)
        values = self.geometry.validate_velocity(velocity)
        flattened = jnp.concatenate(
            tuple(value.reshape((-1,)) for value in values), axis=0
        )
        selected = flattened[relation.face_indices]
        return contract("mkc,mk->mc", relation.interpolation, selected)

    def spread(
        self, relation: MappedMACMarkerRelation, marker_force: ArrayLike, /
    ) -> FaceVelocity:
        self._validate_relation(relation)
        raw_force = jnp.asarray(marker_force)
        force = (
            self.markers.active_values(raw_force)
            if raw_force.shape == (self.markers.capacity, self.markers.ambient_dimension)
            else self.markers.active_velocity_space.validate(raw_force)
        )
        quadrature = self.markers.plan.quadrature_weight[
            self.markers.active_indices
        ].astype(force.dtype)
        contributions = (
            relation.interpolation * force[:, None, :] * quadrature[:, None, None]
        ).sum(axis=-1)
        output = jnp.zeros(self.face_centers.shape[0], dtype=force.dtype)
        if self.accumulation == "fast":
            output = output.at[relation.face_indices].add(contributions)
        else:
            order = self.markers.stable_active_order

            def add_marker(index, values):
                marker = order[index]
                return values.at[relation.face_indices[marker]].add(contributions[marker])

            output = jax.lax.fori_loop(0, order.size, add_marker, output)
        output = output / self.face_measures.astype(output.dtype)
        return tuple(
            output[left:right].reshape(layout.shape)
            for left, right, layout in zip(
                self.component_offsets[:-1],
                self.component_offsets[1:],
                self.geometry.reference.face_layouts,
                strict=True,
            )
        )

    def diagnostics(
        self,
        relation: MappedMACMarkerRelation,
        velocity: FaceVelocity,
        marker_force: ArrayLike,
        /,
    ) -> MappedMACMarkerTransferDiagnostics:
        gathered = self.gather(relation, velocity)
        raw_force = jnp.asarray(marker_force)
        force = (
            self.markers.active_values(raw_force)
            if raw_force.shape == (self.markers.capacity, self.markers.ambient_dimension)
            else self.markers.active_velocity_space.validate(raw_force)
        )
        spread = self.spread(relation, force)
        interpolation_work = jnp.real(
            self.markers.active_velocity_space.inner(gathered, force)
        )
        spreading_work = jnp.real(self.geometry.velocity_space.inner(velocity, spread))
        residual = interpolation_work - spreading_work
        finite = (
            relation.finite
            & jnp.isfinite(interpolation_work)
            & jnp.isfinite(spreading_work)
            & jnp.isfinite(residual)
        )
        tolerance = (
            512.0
            * jnp.finfo(force.dtype).eps
            * jnp.maximum(1.0, jnp.abs(interpolation_work) + jnp.abs(spreading_work))
        )
        return MappedMACMarkerTransferDiagnostics(
            interpolation_work,
            spreading_work,
            residual,
            jnp.max(relation.affine_residual),
            jnp.max(relation.condition_number),
            finite,
            relation.successful & finite & (jnp.abs(residual) <= tolerance),
            self.prepared_id,
        )

    def _validate_relation(self, relation: MappedMACMarkerRelation, /) -> None:
        if not isinstance(relation, MappedMACMarkerRelation):
            raise TypeError("relation must be MappedMACMarkerRelation.")
        if relation.transfer_id != self.prepared_id:
            raise ValueError("Mapped marker relation belongs to another transfer.")


__all__ = [
    "MappedMACMarkerRelation",
    "MappedMACMarkerRouteState",
    "MappedMACMarkerTransferDiagnostics",
    "MappedMACMarkerTransferPlan",
    "PreparedMappedMACMarkerTransfer",
]

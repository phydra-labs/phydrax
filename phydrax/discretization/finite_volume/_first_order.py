#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace, DiagonalPairing
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import CellComplexTopology
from .._triangular import triangle_cell_complex, triangle_connectivity


def triangular_finite_volume_geometry(
    vertices: ArrayLike,
    faces: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Return cell area/centroid and local edge length/outward unit normal."""
    points = jnp.asarray(vertices)
    cells = jnp.asarray(faces, dtype=jnp.int32)
    triangles = points[cells]
    signed_double_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
        triangles[:, 2, 0] - triangles[:, 0, 0]
    )
    signed_double_area = eqx.error_if(
        signed_double_area,
        jnp.any(~jnp.isfinite(signed_double_area)) | jnp.any(signed_double_area == 0.0),
        "Finite-volume triangles require finite nonzero oriented area.",
    )
    area = 0.5 * jnp.abs(signed_double_area)
    orientation = jnp.sign(signed_double_area)
    starts = triangles
    stops = triangles[:, [1, 2, 0]]
    tangents = stops - starts
    lengths = jnp.linalg.norm(tangents, axis=-1)
    lengths = eqx.error_if(
        lengths,
        jnp.any(~jnp.isfinite(lengths)) | jnp.any(lengths <= 0.0),
        "Finite-volume faces require finite positive length.",
    )
    right_normals = jnp.stack((tangents[..., 1], -tangents[..., 0]), axis=-1)
    outward_normals = orientation[:, None, None] * right_normals / lengths[..., None]
    centroids = jnp.mean(triangles, axis=1)
    return area, centroids, lengths, outward_normals


class FiniteVolumePlan(AbstractDiscretizationPlan):
    """First-order cell-centered triangular finite-volume plan."""

    vertices: Array
    faces: Array
    field_name: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        faces: ArrayLike,
        /,
        *,
        field_name: str = "state",
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        points = np.asarray(vertices, dtype=float)
        cells = np.asarray(faces, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
            raise ValueError("FiniteVolumePlan requires vertices shaped (count, 2).")
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != 3:
            raise ValueError("FiniteVolumePlan requires faces shaped (count, 3).")
        if np.any(~np.isfinite(points)):
            raise ValueError("Finite-volume vertices must be finite.")
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise ValueError("Finite-volume cells contain out-of-range vertices.")
        if np.unique(cells).size != points.shape[0]:
            raise ValueError("Every finite-volume vertex must belong to a cell.")
        if np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]:
            raise ValueError("Finite-volume meshes cannot contain duplicate cells.")
        triangle_connectivity(cells, points.shape[0])
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        key_ = (
            DiscretizationKey(
                "finite_volume",
                DiscretizationRole.PHYSICAL,
                domain_labels=("space",),
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.SPARSE_ASSEMBLY,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "first-order-finite-volume-plan",
                    "vertices": array_tree_fingerprint(points),
                    "faces": array_tree_fingerprint(cells),
                    "field": field,
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.vertices = jnp.asarray(points)
        self.faces = jnp.asarray(cells, dtype=jnp.int32)
        self.field_name = field
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(self, /, *, numeric_version: str = "0") -> "FiniteVolumeDiscretization":
        return FiniteVolumeDiscretization(self, numeric_version=numeric_version)


class FiniteVolumeDiscretization(AbstractPreparedDiscretization):
    """Prepared first-order scalar cell-average finite-volume geometry."""

    vertices: Array
    faces: Array
    topology: CellComplexTopology
    cell_areas: Array
    cell_centroids: Array
    face_vertices: Array
    face_centers: Array
    face_lengths: Array
    face_normals: Array
    left_cells: Array
    right_cells: Array
    boundary_face_mask: Array
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(
        self,
        plan: FiniteVolumePlan,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(plan, FiniteVolumePlan):
            raise TypeError("plan must be a FiniteVolumePlan.")
        points = np.asarray(plan.vertices, dtype=float)
        cells = np.asarray(plan.faces, dtype=np.int32)
        connectivity = triangle_connectivity(cells, points.shape[0])
        topology = triangle_cell_complex(cells, points.shape[0])
        area, centroids, local_lengths, local_normals = triangular_finite_volume_geometry(
            plan.vertices,
            plan.faces,
        )
        area_host = np.asarray(area)
        if np.any(~np.isfinite(area_host)) or np.any(area_host <= 0.0):
            raise ValueError("Finite-volume mesh contains degenerate cells.")
        cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
        edge_count = int(connectivity.edges.shape[0])
        left_cells = np.full((edge_count,), -1, dtype=np.int32)
        right_cells = np.full((edge_count,), -1, dtype=np.int32)
        left_local = np.full((edge_count,), -1, dtype=np.int32)
        for cell_index in range(cells.shape[0]):
            for local_index in range(3):
                edge = int(cell_edges[cell_index, local_index])
                if left_cells[edge] < 0:
                    left_cells[edge] = cell_index
                    left_local[edge] = local_index
                elif right_cells[edge] < 0:
                    right_cells[edge] = cell_index
                else:
                    raise ValueError("Finite-volume faces may have at most two cells.")
        if np.any(left_cells < 0):
            raise ValueError("Every finite-volume face requires an incident cell.")
        face_lengths = local_lengths[
            jnp.asarray(left_cells),
            jnp.asarray(left_local),
        ]
        face_normals = local_normals[
            jnp.asarray(left_cells),
            jnp.asarray(left_local),
        ]
        face_vertices = connectivity.edges
        face_centers = jnp.mean(plan.vertices[face_vertices], axis=1)
        interior = right_cells >= 0
        if np.any(interior):
            right_centers = np.asarray(centroids)[right_cells[interior]]
            left_centers = np.asarray(centroids)[left_cells[interior]]
            normals_host = np.asarray(face_normals)[interior]
            if np.any(
                np.sum((right_centers - left_centers) * normals_host, axis=1) <= 0.0
            ):
                raise ValueError("Finite-volume face normals must point left-to-right.")
        support = DiscreteSupport(
            topology,
            2,
            canonical_fingerprint(
                {
                    "kind": "finite-volume-embedding",
                    "vertices": array_tree_fingerprint(points),
                }
            ),
        )
        cell_entities = topology.entity_sets[2]
        field_space = DiscreteFieldSpace(
            plan.field_name,
            support.support_id,
            EntityDofLayout(
                cell_entities.entity_set_id,
                cell_entities.count,
                cell_entities.count,
            ),
            ArraySpace(
                (cells.shape[0],),
                pairing=DiagonalPairing(area),
            ),
            representation="cell_average",
            conformity="discontinuous",
            reconstruction_id=canonical_fingerprint(
                {"kind": "piecewise-constant-reconstruction", "plan": plan.plan_id}
            ),
        )
        measure = DiscreteMeasure(
            "cell_area",
            support.support_id,
            cell_entities.entity_set_id,
            area,
            normalization="physical",
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            resource_counts={
                "cells": cells.shape[0],
                "faces": edge_count,
                "boundary_faces": int(np.count_nonzero(right_cells < 0)),
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(field_space,),
            measures=(measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-first-order-finite-volume",
                "plan": plan.plan_id,
                "embedding": support.embedding_id,
                "numeric_version": version,
            }
        )
        self.vertices = plan.vertices
        self.faces = plan.faces
        self.topology = topology
        self.cell_areas = area
        self.cell_centroids = centroids
        self.face_vertices = face_vertices
        self.face_centers = face_centers
        self.face_lengths = face_lengths
        self.face_normals = face_normals
        self.left_cells = jnp.asarray(left_cells)
        self.right_cells = jnp.asarray(right_cells)
        self.boundary_face_mask = jnp.asarray(right_cells < 0)
        self.key = plan.key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation

    def first_order_dynamics(
        self,
        flux: Callable[[Array, Array, Array, Any], ArrayLike],
        wave_speed: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        exterior_state: Callable[[Array, Array, Array, Array, Any], ArrayLike]
        | None = None,
        source: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
    ) -> "FirstOrderFiniteVolumeDynamics":
        return FirstOrderFiniteVolumeDynamics(
            self,
            flux,
            wave_speed,
            exterior_state=exterior_state,
            source=source,
        )


class FirstOrderFiniteVolumeDynamics(StrictModule):
    """Conservative first-order Rusanov semidiscretization for one scalar law."""

    discretization: FiniteVolumeDiscretization
    flux: Callable[[Array, Array, Array, Any], ArrayLike] = eqx.field(static=True)
    wave_speed: Callable[[Array, Array, Array, Any], ArrayLike] = eqx.field(static=True)
    exterior_state: Callable[[Array, Array, Array, Array, Any], ArrayLike] | None = (
        eqx.field(static=True)
    )
    source: Callable[[Array, Array, Array, Any], ArrayLike] | None = eqx.field(
        static=True
    )

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        flux: Callable[[Array, Array, Array, Any], ArrayLike],
        wave_speed: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        exterior_state: Callable[[Array, Array, Array, Array, Any], ArrayLike]
        | None = None,
        source: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
    ):
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be a FiniteVolumeDiscretization.")
        if not callable(flux) or not callable(wave_speed):
            raise TypeError("flux and wave_speed must be callable.")
        if exterior_state is not None and not callable(exterior_state):
            raise TypeError("exterior_state must be callable or None.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        if (
            bool(np.any(np.asarray(discretization.boundary_face_mask)))
            and exterior_state is None
        ):
            raise ValueError("Boundary faces require an explicit exterior_state policy.")
        self.discretization = discretization
        self.flux = flux
        self.wave_speed = wave_speed
        self.exterior_state = exterior_state
        self.source = source

    def face_flux(self, time: Array, state: Array, args: Any, /) -> Array:
        discretization = self.discretization
        value = jnp.asarray(state)
        cell_count = int(discretization.cell_areas.shape[0])
        if value.shape != (cell_count,):
            raise ValueError(f"Finite-volume state must have shape {(cell_count,)}.")
        left = value[discretization.left_cells]
        safe_right = jnp.where(
            discretization.right_cells >= 0,
            discretization.right_cells,
            0,
        )
        right = value[safe_right]
        if self.exterior_state is not None:
            boundary_right = jnp.asarray(
                self.exterior_state(
                    jnp.asarray(time),
                    left,
                    discretization.face_centers,
                    discretization.face_normals,
                    args,
                )
            )
            if boundary_right.shape != left.shape:
                raise ValueError("exterior_state must return one state per face.")
            right = jnp.where(discretization.boundary_face_mask, boundary_right, right)
        left_flux = jnp.asarray(
            self.flux(jnp.asarray(time), left, discretization.face_centers, args)
        )
        right_flux = jnp.asarray(
            self.flux(jnp.asarray(time), right, discretization.face_centers, args)
        )
        expected_flux = (int(discretization.face_lengths.shape[0]), 2)
        if left_flux.shape != expected_flux or right_flux.shape != expected_flux:
            raise ValueError(f"Physical flux must return shape {expected_flux}.")
        speed = jnp.asarray(
            self.wave_speed(
                left,
                right,
                discretization.face_normals,
                args,
            )
        )
        if speed.shape == ():
            speed = jnp.broadcast_to(speed, left.shape)
        if speed.shape != left.shape:
            raise ValueError("wave_speed must return scalar or one value per face.")
        speed = eqx.error_if(
            speed,
            jnp.any(~jnp.isfinite(speed)) | jnp.any(speed < 0.0),
            "Rusanov wave speeds must be finite and non-negative.",
        )
        normal_physical_flux = 0.5 * jnp.sum(
            (left_flux + right_flux) * discretization.face_normals,
            axis=-1,
        )
        return normal_physical_flux - 0.5 * speed * (right - left)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        discretization = self.discretization
        flux = self.face_flux(jnp.asarray(time), state, args)
        integrated = flux * discretization.face_lengths
        residual = (
            jnp.zeros_like(jnp.asarray(state))
            .at[discretization.left_cells]
            .add(-integrated)
        )
        interior = discretization.right_cells >= 0
        safe_right = jnp.where(interior, discretization.right_cells, 0)
        residual = residual.at[safe_right].add(
            jnp.where(interior, integrated, jnp.zeros_like(integrated))
        )
        derivative = residual / discretization.cell_areas
        if self.source is not None:
            source = jnp.asarray(
                self.source(
                    jnp.asarray(time),
                    jnp.asarray(state),
                    discretization.cell_centroids,
                    args,
                )
            )
            if source.shape != derivative.shape:
                raise ValueError("Finite-volume source must match the cell state shape.")
            derivative = derivative + source
        return derivative


__all__ = [
    "FiniteVolumeDiscretization",
    "FiniteVolumePlan",
    "FirstOrderFiniteVolumeDynamics",
    "triangular_finite_volume_geometry",
]

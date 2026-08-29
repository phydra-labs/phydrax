#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh
from ...equations import CellResidualAction, FiniteElementForm
from ...equations.fem import symmetric_gradient


class PhaseFieldFractureParameters(StrictModule, NonTrainableState):
    lame_lambda: Array
    shear_modulus: Array
    critical_energy_release_rate: Array
    length_scale: Array
    residual_stiffness: Array

    def __init__(
        self,
        lame_lambda: ArrayLike,
        shear_modulus: ArrayLike,
        critical_energy_release_rate: ArrayLike,
        length_scale: ArrayLike,
        /,
        *,
        residual_stiffness: ArrayLike = 1.0e-8,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                lame_lambda,
                shear_modulus,
                critical_energy_release_rate,
                length_scale,
                residual_stiffness,
            )
        )
        if any(value.shape != () or not bool(jnp.isfinite(value)) for value in values):
            raise ValueError("Fracture parameters must be finite scalars.")
        if any(value <= 0.0 for value in values[1:4]) or values[0] < 0.0:
            raise ValueError("Fracture elastic and regularization data are inadmissible.")
        if not 0.0 <= values[4] < 1.0:
            raise ValueError("Residual fracture stiffness must lie in [0, 1).")
        (
            self.lame_lambda,
            self.shear_modulus,
            self.critical_energy_release_rate,
            self.length_scale,
            self.residual_stiffness,
        ) = values

    def degradation(self, damage: ArrayLike, /) -> Array:
        damage_ = jnp.asarray(damage)
        return (1.0 - damage_) ** 2 + self.residual_stiffness

    def tensile_energy(self, strain: ArrayLike, /) -> Array:
        strain_ = jnp.asarray(strain)
        dimension = strain_.shape[-1]
        trace = jnp.trace(strain_, axis1=-2, axis2=-1)
        deviator = strain_ - trace[..., None, None] * jnp.eye(dimension) / dimension
        bulk = self.lame_lambda + 2.0 * self.shear_modulus / dimension
        return 0.5 * bulk * jnp.maximum(trace, 0.0) ** 2 + self.shear_modulus * jnp.sum(
            deviator**2, axis=(-2, -1)
        )


class FractureHistoryState(StrictModule):
    history: Array
    accepted_damage: Array
    state_version: int = eqx.field(static=True)

    def __init__(
        self,
        history: ArrayLike,
        accepted_damage: ArrayLike,
        /,
        *,
        state_version: int = 0,
    ):
        history_ = jnp.asarray(history)
        damage = jnp.asarray(accepted_damage)
        version = int(state_version)
        if history_.ndim != 2 or damage.ndim != 1 or version < 0:
            raise ValueError("Fracture history/damage shapes or version are invalid.")
        if bool(jnp.any(history_ < 0.0) | jnp.any((damage < 0.0) | (damage > 1.0))):
            raise ValueError("Fracture history and damage must be admissible.")
        self.history = history_
        self.accepted_damage = damage
        self.state_version = version

    def promote(
        self,
        tensile_energy: ArrayLike,
        damage: ArrayLike,
        /,
    ) -> FractureHistoryState:
        energy = jnp.asarray(tensile_energy)
        damage_ = jnp.asarray(damage)
        if (
            energy.shape != self.history.shape
            or damage_.shape != self.accepted_damage.shape
        ):
            raise ValueError("Fracture promotion must preserve state layouts.")
        if bool(jnp.any(damage_ < self.accepted_damage) | jnp.any(damage_ > 1.0)):
            raise ValueError("Accepted fracture damage must be irreversible and bounded.")
        return FractureHistoryState(
            jnp.maximum(self.history, energy),
            damage_,
            state_version=self.state_version + 1,
        )


def phase_field_fracture_form(
    displacement_field: str,
    damage_field: str,
    parameters: PhaseFieldFractureParameters,
    history: ArrayLike,
    /,
    *,
    form_id: str = "phase-field-fracture",
) -> FiniteElementForm:
    if not isinstance(parameters, PhaseFieldFractureParameters):
        raise TypeError("parameters must be PhaseFieldFractureParameters.")
    history_ = jnp.asarray(history)
    if history_.ndim != 2:
        raise ValueError("Fracture history must have cell/quadrature shape.")

    def equilibrium(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        displacement_gradient, _ = gradients
        _, damage = values
        strain = symmetric_gradient(displacement_gradient)
        dimension = strain.shape[-1]
        trace = jnp.trace(strain, axis1=-2, axis2=-1)
        identity = jnp.eye(dimension, dtype=strain.dtype)
        positive_trace = jnp.maximum(trace, 0.0)
        negative_trace = jnp.minimum(trace, 0.0)
        deviator = strain - trace[..., None, None] * identity / dimension
        stress_plus = (
            parameters.lame_lambda + 2.0 * parameters.shear_modulus / dimension
        ) * positive_trace[
            ..., None, None
        ] * identity + 2.0 * parameters.shear_modulus * deviator
        stress_minus = (
            (parameters.lame_lambda + 2.0 * parameters.shear_modulus / dimension)
            * negative_trace[..., None, None]
            * identity
        )
        stress = (
            parameters.degradation(damage)[..., None, None] * stress_plus + stress_minus
        )
        return oe.contract("cq,cqib,cqab->cia", weights, test_gradients, stress)

    def damage_residual(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        _, damage = values
        _, damage_gradient = gradients
        if history_.shape[0] != damage.shape[0] or history_.shape[1] not in (
            1,
            damage.shape[1],
        ):
            raise ValueError("Fracture history must match damage cells/quadrature.")
        history_values = jnp.broadcast_to(history_, damage.shape)
        local = (
            parameters.critical_energy_release_rate / parameters.length_scale * damage
            - 2.0 * (1.0 - damage) * history_values
        )
        return (
            oe.contract("cq,cq,qi->ci", weights, local, test_basis)
            + parameters.critical_energy_release_rate
            * parameters.length_scale
            * oe.contract("cq,cqid,cqd->ci", weights, test_gradients, damage_gradient)
        )

    return FiniteElementForm(
        form_id,
        (displacement_field, damage_field),
        (
            CellResidualAction(
                displacement_field,
                (displacement_field, damage_field),
                equilibrium,
                action_id="fracture-equilibrium",
            ),
            CellResidualAction(
                damage_field,
                (displacement_field, damage_field),
                damage_residual,
                action_id="fracture-damage",
            ),
        ),
    )


class CrackGeometry(StrictModule, NonTrainableState):
    start: Array
    end: Array
    tangent: Array
    normal: Array
    crack_id: str = eqx.field(static=True)

    def __init__(self, start: ArrayLike, end: ArrayLike, /, *, crack_id: str = "crack"):
        start_ = jnp.asarray(start)
        end_ = jnp.asarray(end)
        if start_.shape != (2,) or end_.shape != (2,):
            raise ValueError(
                "Initial XFEM crack geometry requires two-dimensional endpoints."
            )
        tangent = end_ - start_
        length = jnp.sqrt(jnp.sum(tangent**2))
        if not bool(jnp.isfinite(length)) or length <= 0.0:
            raise ValueError("Crack segment must have positive finite length.")
        tangent = tangent / length
        normal = jnp.asarray((-tangent[1], tangent[0]))
        identifier = str(crack_id)
        if not identifier:
            raise ValueError("crack_id must be non-empty.")
        self.start = start_
        self.end = end_
        self.tangent = tangent
        self.normal = normal
        self.crack_id = canonical_fingerprint(
            {"kind": "xfem-crack", "declared_id": identifier}
        )

    def signed_distance(self, points: ArrayLike, /) -> Array:
        points_ = jnp.asarray(points)
        return jnp.sum((points_ - self.start) * self.normal, axis=-1)

    def heaviside(self, points: ArrayLike, /) -> Array:
        return jnp.where(self.signed_distance(points) >= 0.0, 1.0, -1.0)


class XFEMEnrichmentState(StrictModule, NonTrainableState):
    active_cell_ids: Array
    active_vertex_ids: Array
    topology_version: int = eqx.field(static=True)
    crack_id: str = eqx.field(static=True)
    enrichment_id: str = eqx.field(static=True)

    def __init__(
        self,
        active_cell_ids: ArrayLike,
        active_vertex_ids: ArrayLike,
        crack_id: str,
        /,
        *,
        topology_version: int = 0,
    ):
        cells = jnp.asarray(active_cell_ids, dtype=jnp.int64)
        vertices = jnp.asarray(active_vertex_ids, dtype=jnp.int64)
        version = int(topology_version)
        if cells.ndim != 1 or vertices.ndim != 1 or version < 0:
            raise ValueError("XFEM enrichment IDs and version are invalid.")
        self.active_cell_ids = cells
        self.active_vertex_ids = vertices
        self.topology_version = version
        self.crack_id = str(crack_id)
        self.enrichment_id = canonical_fingerprint(
            {
                "kind": "xfem-enrichment-state",
                "crack": self.crack_id,
                "cells": cells.tolist(),
                "vertices": vertices.tolist(),
                "version": version,
            }
        )


def classify_crack_cells(mesh: CellMesh, crack: CrackGeometry, /) -> XFEMEnrichmentState:
    if not isinstance(mesh, CellMesh) or not isinstance(crack, CrackGeometry):
        raise TypeError("XFEM classification requires a CellMesh and CrackGeometry.")
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "triangle":
        raise ValueError("XFEM classification currently requires one T3 block.")
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    distances = np.asarray(crack.signed_distance(mesh.coordinates))
    cut = (np.min(distances[cells], axis=1) <= 0.0) & (
        np.max(distances[cells], axis=1) >= 0.0
    )
    active_cells = np.flatnonzero(cut)
    active_vertices = np.unique(cells[active_cells].reshape((-1,)))
    return XFEMEnrichmentState(
        np.asarray(mesh.blocks[0].global_ids)[active_cells],
        np.asarray(mesh.vertex_global_ids)[active_vertices],
        crack.crack_id,
    )


class CutCellQuadrature(StrictModule, NonTrainableState):
    cell_ids: Array
    points: Array
    weights: Array
    side: Array
    valid: Array

    def __init__(
        self,
        cell_ids: ArrayLike,
        points: ArrayLike,
        weights: ArrayLike,
        side: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        cells = jnp.asarray(cell_ids, dtype=jnp.int64)
        points_ = jnp.asarray(points)
        weights_ = jnp.asarray(weights)
        side_ = jnp.asarray(side, dtype=jnp.int8)
        valid_ = jnp.asarray(valid, dtype=bool)
        if (
            cells.ndim != 1
            or points_.shape != (cells.size, 4, 2)
            or weights_.shape != (cells.size, 4)
            or side_.shape != weights_.shape
            or valid_.shape != weights_.shape
        ):
            raise ValueError("Cut-cell quadrature layouts are incompatible.")
        self.cell_ids = cells
        self.points = points_
        self.weights = weights_
        self.side = side_
        self.valid = valid_


def _clip_polygon(vertices, distances, positive):
    polygon = [
        (np.asarray(vertex, dtype=float), float(distance))
        for vertex, distance in zip(vertices, distances, strict=True)
    ]
    result = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_inside = current[1] >= 0.0 if positive else current[1] <= 0.0
        previous_inside = previous[1] >= 0.0 if positive else previous[1] <= 0.0
        if current_inside != previous_inside:
            fraction = previous[1] / (previous[1] - current[1])
            point = previous[0] + fraction * (current[0] - previous[0])
            result.append((point, 0.0))
        if current_inside:
            result.append(current)
    return [value[0] for value in result]


def cut_cell_quadrature(
    mesh: CellMesh,
    crack: CrackGeometry,
    /,
) -> CutCellQuadrature:
    enrichment = classify_crack_cells(mesh, crack)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    global_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    active = set(np.asarray(enrichment.active_cell_ids).tolist())
    coordinates = np.asarray(mesh.coordinates)
    distance = np.asarray(crack.signed_distance(mesh.coordinates))
    selected_ids = []
    all_points = []
    all_weights = []
    all_sides = []
    all_valid = []
    for local_cell, cell_id in enumerate(global_ids):
        if int(cell_id) not in active:
            continue
        triangle = coordinates[cells[local_cell]]
        signed = distance[cells[local_cell]]
        points = np.zeros((4, 2), dtype=coordinates.dtype)
        weights = np.zeros((4,), dtype=coordinates.dtype)
        sides = np.zeros((4,), dtype=np.int8)
        valid = np.zeros((4,), dtype=bool)
        cursor = 0
        for positive, side in ((True, 1), (False, -1)):
            polygon = _clip_polygon(triangle, signed, positive)
            for index in range(1, len(polygon) - 1):
                subtriangle = np.stack((polygon[0], polygon[index], polygon[index + 1]))
                first_edge = subtriangle[1] - subtriangle[0]
                second_edge = subtriangle[2] - subtriangle[0]
                cross = (
                    first_edge[0] * second_edge[1]
                    - first_edge[1] * second_edge[0]
                )
                points[cursor] = np.mean(subtriangle, axis=0)
                weights[cursor] = 0.5 * abs(cross)
                sides[cursor] = side
                valid[cursor] = True
                cursor += 1
        selected_ids.append(int(cell_id))
        all_points.append(points)
        all_weights.append(weights)
        all_sides.append(sides)
        all_valid.append(valid)
    return CutCellQuadrature(
        np.asarray(selected_ids, dtype=np.int64),
        np.asarray(all_points),
        np.asarray(all_weights),
        np.asarray(all_sides),
        np.asarray(all_valid),
    )


class FixedMeshEnrichmentLayout(StrictModule, NonTrainableState):
    base_dof_count: int = eqx.field(static=True)
    enriched_vertex_ids: Array
    enriched_dofs: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        enrichment: XFEMEnrichmentState,
        /,
    ):
        if not isinstance(mesh, CellMesh) or not isinstance(
            enrichment, XFEMEnrichmentState
        ):
            raise TypeError("Enrichment layout requires mesh and enrichment state.")
        base_count = int(mesh.vertex_global_ids.size)
        vertex_by_id = {
            int(identifier): index
            for index, identifier in enumerate(np.asarray(mesh.vertex_global_ids))
        }
        local_vertices = np.asarray(
            [vertex_by_id[int(value)] for value in enrichment.active_vertex_ids],
            dtype=np.int32,
        )
        enriched_dofs = base_count + np.arange(local_vertices.size, dtype=np.int32)
        self.base_dof_count = base_count
        self.enriched_vertex_ids = jnp.asarray(enrichment.active_vertex_ids)
        self.enriched_dofs = jnp.asarray(enriched_dofs)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "fixed-mesh-enrichment-layout",
                "crack": enrichment.crack_id,
                "vertices": enrichment.active_vertex_ids.tolist(),
                "base_dof_count": base_count,
            }
        )


def enriched_field_value(
    base_value: ArrayLike,
    enrichment_coefficient: ArrayLike,
    points: ArrayLike,
    crack: CrackGeometry,
    /,
) -> Array:
    base = jnp.asarray(base_value)
    enrichment = jnp.asarray(enrichment_coefficient)
    heaviside = crack.heaviside(points)
    return base + heaviside[..., None] * enrichment


__all__ = [
    "CrackGeometry",
    "CutCellQuadrature",
    "FixedMeshEnrichmentLayout",
    "FractureHistoryState",
    "PhaseFieldFractureParameters",
    "XFEMEnrichmentState",
    "classify_crack_cells",
    "cut_cell_quadrature",
    "enriched_field_value",
    "phase_field_fracture_form",
]

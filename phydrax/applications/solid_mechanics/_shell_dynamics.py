#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...discretization.contact import (
    CollisionSurfacePlan,
    ContactPairPolicy,
    ContactPrecisionPolicy,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from ...discretization.particle._rigid_contact import RigidContactGeometry
from ...linalg import ArraySpace


class ShellMaterialParameters(StrictModule, NonTrainableState):
    """Plane-stress and hinge moduli before thickness integration.

    ``membrane_matrix`` acts on ``(E11, E22, 2 E12)``. The membrane energy is
    linear in thickness, while ``bending_modulus`` is multiplied by thickness
    cubed. Both parameters may be positive semidefinite, including zero.
    """

    membrane_matrix: Array
    bending_modulus: Array
    minimum_membrane_eigenvalue: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        membrane_matrix: ArrayLike,
        bending_modulus: ArrayLike,
        /,
        *,
        parameter_id: str | None = None,
    ):
        raw_matrix = np.asarray(membrane_matrix)
        if raw_matrix.shape != (3, 3):
            raise ValueError("membrane_matrix must have shape (3, 3).")
        if not np.issubdtype(raw_matrix.dtype, np.number) or np.issubdtype(
            raw_matrix.dtype, np.complexfloating
        ):
            raise TypeError("membrane_matrix must be real numerical data.")
        matrix = np.asarray(raw_matrix, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("membrane_matrix must be finite.")
        symmetry_scale = max(float(np.max(np.abs(matrix))), 1.0)
        if float(np.max(np.abs(matrix - matrix.T))) > 1.0e-12 * symmetry_scale:
            raise ValueError("membrane_matrix must be symmetric.")
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        psd_tolerance = 1.0e-12 * max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(eigenvalues[0]) < -psd_tolerance:
            raise ValueError("membrane_matrix must be positive semidefinite.")

        raw_bending = np.asarray(bending_modulus)
        if (
            raw_bending.shape != ()
            or not np.issubdtype(raw_bending.dtype, np.number)
            or np.issubdtype(raw_bending.dtype, np.complexfloating)
        ):
            raise TypeError("bending_modulus must be a real numerical scalar.")
        bending = float(raw_bending)
        if not isfinite(bending) or bending < 0.0:
            raise ValueError("bending_modulus must be finite and nonnegative.")

        matrix_array = jnp.asarray(matrix)
        bending_array = jnp.asarray(bending, dtype=matrix_array.dtype)
        generated = canonical_fingerprint(
            {
                "kind": "shell-material-parameters",
                "membrane": array_tree_fingerprint(matrix),
                "bending_modulus": bending,
            }
        )
        resolved = generated if parameter_id is None else str(parameter_id)
        if not resolved:
            raise ValueError("parameter_id must be nonempty.")
        self.membrane_matrix = matrix_array
        self.bending_modulus = bending_array
        self.minimum_membrane_eigenvalue = max(float(eigenvalues[0]), 0.0)
        self.parameter_id = resolved

    @classmethod
    def isotropic(
        cls,
        young_modulus: float,
        poisson_ratio: float,
        *,
        parameter_id: str | None = None,
    ) -> ShellMaterialParameters:
        young = float(young_modulus)
        poisson = float(poisson_ratio)
        if not isfinite(young) or young < 0.0:
            raise ValueError("young_modulus must be finite and nonnegative.")
        if not isfinite(poisson) or not (-1.0 < poisson < 1.0):
            raise ValueError("poisson_ratio must lie strictly between -1 and one.")
        denominator = 1.0 - poisson * poisson
        scale = young / denominator
        membrane = scale * np.asarray(
            [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]],
            dtype=np.float64,
        )
        bending = young / (12.0 * denominator)
        return cls(membrane, bending, parameter_id=parameter_id)


class ShellGeometryEvidence(StrictModule):
    triangle_area: Array
    area_ratio: Array
    orientation_ratio: Array
    degenerate: Array
    inverted: Array
    finite: Array
    minimum_area_ratio: Array
    minimum_orientation_ratio: Array
    valid: Array


class ShellEvaluation(StrictModule):
    forces: Array
    membrane_energy: Array
    bending_energy: Array
    stored_energy: Array
    kinetic_energy: Array
    total_energy: Array
    geometry: ShellGeometryEvidence
    self_contact: RigidContactGeometry
    finite: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)


class ShellState(StrictModule):
    positions: Array
    velocities: Array


class ShellStepDiagnostics(StrictModule):
    stable_step_size: Array
    requested_step_size: Array
    maximum_displacement_ratio: Array
    fixed_position_defect: Array
    fixed_velocity_defect: Array
    finite: Array
    locally_valid: Array


class ShellStepResult(StrictModule):
    candidate_state: ShellState
    accepted_state: ShellState
    initial_evaluation: ShellEvaluation
    evaluation: ShellEvaluation
    diagnostics: ShellStepDiagnostics
    successful: Array
    rejection_reasons: Array
    prepared_id: str = eqx.field(static=True)


class ShellRejectionReason(IntFlag):
    NONE = 0
    INVALID_STEP = 1 << 0
    UNSTABLE_STEP = 1 << 1
    INVALID_STATE = 1 << 2
    NONFINITE = 1 << 3
    DEGENERATE = 1 << 4
    INVERTED = 1 << 5
    DISPLACEMENT_BOUND = 1 << 6
    FIXED_NODE_DEFECT = 1 << 7
    SELF_CONTACT_GEOMETRY = 1 << 8
    EXTERNAL_FORCE = 1 << 9


def _constant_triangle_array(
    value: ArrayLike,
    triangle_count: int,
    name: str,
    /,
    *,
    positive: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be real numerical data.")
    values = np.asarray(raw, dtype=np.float64)
    if values.shape == ():
        values = np.full((triangle_count,), float(values), dtype=np.float64)
    if values.shape != (triangle_count,):
        raise ValueError(f"{name} must be scalar or have shape ({triangle_count},).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return values


def _fixed_topology(
    triangles: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_uses: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for triangle_index, triangle in enumerate(triangles.tolist()):
        a, b, c = (int(value) for value in triangle)
        directed = ((a, b, c), (b, c, a), (c, a, b))
        for start, end, opposite in directed:
            key = (min(start, end), max(start, end))
            uses = edge_uses.setdefault(key, [])
            uses.append((triangle_index, start, end, opposite))
            if len(uses) > 2:
                raise ValueError("Shell topology must be a two-manifold at every edge.")

    stencils: list[tuple[int, int, int, int]] = []
    hinge_triangles: list[tuple[int, int]] = []
    hinge_edges: list[tuple[int, int]] = []
    for key in sorted(edge_uses):
        uses = edge_uses[key]
        if len(uses) != 2:
            continue
        left, right = uses
        if left[1] != right[2] or left[2] != right[1]:
            raise ValueError(
                "Adjacent shell triangles must use opposite interior-edge orientations."
            )
        stencils.append((left[1], left[2], left[3], right[3]))
        hinge_triangles.append((left[0], right[0]))
        hinge_edges.append((left[1], left[2]))
    return (
        np.asarray(stencils, dtype=np.int32).reshape((-1, 4)),
        np.asarray(hinge_triangles, dtype=np.int32).reshape((-1, 2)),
        np.asarray(hinge_edges, dtype=np.int32).reshape((-1, 2)),
    )


class TriangularShellPlan(StrictModule, NonTrainableState):
    """Fixed triangular topology and constitutive data for a shell in 3-D."""

    triangles: Array
    material: ShellMaterialParameters
    thickness: Array
    density: Array
    fixed_mask: Array
    fixed_mask_provided: bool = eqx.field(static=True)
    hinge_stencils: Array
    hinge_triangles: Array
    hinge_edges: Array
    self_contact_pairs: Array
    self_contact_slot_valid: Array
    contact_keys: Array
    triangle_count: int = eqx.field(static=True)
    hinge_count: int = eqx.field(static=True)
    self_contact_capacity: int = eqx.field(static=True)
    rest_area_tolerance: float = eqx.field(static=True)
    minimum_area_ratio: float = eqx.field(static=True)
    minimum_orientation_ratio: float = eqx.field(static=True)
    contact_distance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        triangles: ArrayLike,
        material: ShellMaterialParameters,
        /,
        *,
        thickness: ArrayLike,
        density: ArrayLike,
        fixed_mask: ArrayLike | None = None,
        self_contact_pairs: ArrayLike | None = None,
        rest_area_tolerance: float = 1.0e-12,
        minimum_area_ratio: float = 1.0e-8,
        minimum_orientation_ratio: float = 1.0e-8,
        contact_distance_tolerance: float = 1.0e-10,
        plan_id: str | None = None,
    ):
        raw_triangles = np.asarray(triangles)
        if raw_triangles.ndim != 2 or raw_triangles.shape[1:] != (3,):
            raise ValueError("triangles must have shape (triangle_count, 3).")
        if raw_triangles.shape[0] == 0:
            raise ValueError("A triangular shell requires at least one triangle.")
        if not np.issubdtype(raw_triangles.dtype, np.integer):
            raise TypeError("triangles must contain integer node indices.")
        topology = np.asarray(raw_triangles, dtype=np.int32)
        if np.any(topology < 0):
            raise ValueError("triangle node indices must be nonnegative.")
        if np.any(
            (topology[:, 0] == topology[:, 1])
            | (topology[:, 1] == topology[:, 2])
            | (topology[:, 2] == topology[:, 0])
        ):
            raise ValueError("A triangle cannot repeat a node.")
        triangle_count = int(topology.shape[0])
        if not isinstance(material, ShellMaterialParameters):
            raise TypeError("material must be ShellMaterialParameters.")
        thickness_values = _constant_triangle_array(
            thickness, triangle_count, "thickness", positive=True
        )
        density_values = _constant_triangle_array(
            density, triangle_count, "density", positive=True
        )
        tolerances = tuple(
            float(value)
            for value in (
                rest_area_tolerance,
                minimum_area_ratio,
                minimum_orientation_ratio,
                contact_distance_tolerance,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Shell geometry tolerances must be positive and finite.")

        stencils, hinge_triangles, hinge_edges = _fixed_topology(topology)

        if fixed_mask is None:
            fixed = np.zeros((0,), dtype=bool)
            fixed_provided = False
        else:
            fixed_raw = np.asarray(fixed_mask)
            if fixed_raw.ndim != 1 or not np.issubdtype(fixed_raw.dtype, np.bool_):
                raise TypeError("fixed_mask must be a one-dimensional Boolean array.")
            fixed = np.asarray(fixed_raw, dtype=bool)
            fixed_provided = True

        if self_contact_pairs is None:
            pairs = np.zeros((0, 2), dtype=np.int32)
            slot_valid = np.zeros((0,), dtype=bool)
        else:
            raw_pairs = np.asarray(self_contact_pairs)
            if raw_pairs.ndim != 2 or raw_pairs.shape[1:] != (2,):
                raise ValueError("self_contact_pairs must have shape (capacity, 2).")
            if not np.issubdtype(raw_pairs.dtype, np.integer):
                raise TypeError(
                    "self_contact_pairs must contain integer triangle indices."
                )
            pairs = np.asarray(raw_pairs, dtype=np.int32)
            padding = np.all(pairs == -1, axis=1)
            partially_padded = np.any(pairs == -1, axis=1) & ~padding
            if np.any(partially_padded):
                raise ValueError(
                    "Padded self-contact slots must contain the pair (-1, -1)."
                )
            valid_indices = ~padding
            if np.any(
                valid_indices & np.any((pairs < 0) | (pairs >= triangle_count), axis=1)
            ):
                raise ValueError("self-contact triangle indices are out of range.")
            canonical_pairs = np.sort(np.where(padding[:, None], 0, pairs), axis=1)
            if np.any(valid_indices & (canonical_pairs[:, 0] == canonical_pairs[:, 1])):
                raise ValueError("A triangle cannot be a self-contact pair with itself.")
            valid_rows = canonical_pairs[valid_indices]
            if (
                valid_rows.shape[0] > 1
                and np.unique(valid_rows, axis=0).shape[0] != valid_rows.shape[0]
            ):
                raise ValueError("self_contact_pairs cannot contain duplicate pairs.")
            pairs = canonical_pairs.astype(np.int32)
            shared_node = np.zeros((pairs.shape[0],), dtype=bool)
            for slot in range(pairs.shape[0]):
                if valid_indices[slot]:
                    left_nodes = topology[pairs[slot, 0]]
                    right_nodes = topology[pairs[slot, 1]]
                    shared_node[slot] = bool(
                        np.any(left_nodes[:, None] == right_nodes[None, :])
                    )
            slot_valid = valid_indices & ~shared_node

        contact_capacity = int(pairs.shape[0])
        keys = np.arange(contact_capacity, dtype=np.int32)
        triangles_array = jnp.asarray(topology)
        thickness_array = jnp.asarray(
            thickness_values, dtype=material.membrane_matrix.dtype
        )
        density_array = jnp.asarray(density_values, dtype=material.membrane_matrix.dtype)
        generated = canonical_fingerprint(
            {
                "kind": "triangular-shell-plan",
                "triangles": array_tree_fingerprint(topology),
                "material": material.parameter_id,
                "material_values": array_tree_fingerprint(
                    (
                        np.asarray(material.membrane_matrix),
                        np.asarray(material.bending_modulus),
                    )
                ),
                "thickness": array_tree_fingerprint(thickness_values),
                "density": array_tree_fingerprint(density_values),
                "fixed_mask": array_tree_fingerprint(fixed),
                "hinges": array_tree_fingerprint(stencils),
                "self_contact_pairs": array_tree_fingerprint(pairs),
                "self_contact_valid": array_tree_fingerprint(slot_valid),
                "tolerances": tolerances,
            }
        )
        resolved = generated if plan_id is None else str(plan_id)
        if not resolved:
            raise ValueError("plan_id must be nonempty.")

        self.triangles = triangles_array
        self.material = material
        self.thickness = thickness_array
        self.density = density_array
        self.fixed_mask = jnp.asarray(fixed)
        self.fixed_mask_provided = fixed_provided
        self.hinge_stencils = jnp.asarray(stencils)
        self.hinge_triangles = jnp.asarray(hinge_triangles)
        self.hinge_edges = jnp.asarray(hinge_edges)
        self.self_contact_pairs = jnp.asarray(pairs)
        self.self_contact_slot_valid = jnp.asarray(slot_valid)
        self.contact_keys = jnp.asarray(keys)
        self.triangle_count = triangle_count
        self.hinge_count = int(stencils.shape[0])
        self.self_contact_capacity = contact_capacity
        (
            self.rest_area_tolerance,
            self.minimum_area_ratio,
            self.minimum_orientation_ratio,
            self.contact_distance_tolerance,
        ) = tolerances
        self.plan_id = resolved

    def prepare(self, reference_positions: ArrayLike, /) -> PreparedTriangularShell:
        return PreparedTriangularShell(self, reference_positions)


def _numpy_dihedral(
    positions: np.ndarray,
    triangles: np.ndarray,
    triangle_pair: np.ndarray,
    edge: np.ndarray,
    /,
) -> float:
    left_nodes = triangles[int(triangle_pair[0])]
    right_nodes = triangles[int(triangle_pair[1])]
    left_edges = positions[left_nodes]
    right_edges = positions[right_nodes]
    left_normal = np.cross(left_edges[1] - left_edges[0], left_edges[2] - left_edges[0])
    right_normal = np.cross(
        right_edges[1] - right_edges[0], right_edges[2] - right_edges[0]
    )
    left_normal = left_normal / np.linalg.norm(left_normal)
    right_normal = right_normal / np.linalg.norm(right_normal)
    edge_vector = positions[int(edge[1])] - positions[int(edge[0])]
    edge_vector = edge_vector / np.linalg.norm(edge_vector)
    sine = float(np.dot(edge_vector, np.cross(left_normal, right_normal)))
    cosine = float(np.clip(np.dot(left_normal, right_normal), -1.0, 1.0))
    return float(np.arctan2(sine, cosine))


class PreparedTriangularShell(StrictModule, NonTrainableState):
    plan: TriangularShellPlan
    reference_positions: Array
    rest_metric: Array
    rest_shape_inverse: Array
    rest_area: Array
    rest_unit_normal: Array
    rest_anchor_frame: Array
    rest_dihedral: Array
    hinge_weight: Array
    nodal_mass: Array
    nodal_stiffness: Array
    fixed_mask: Array
    characteristic_length: Array
    node_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: TriangularShellPlan, reference_positions: ArrayLike, /):
        raw_reference = np.asarray(reference_positions)
        if raw_reference.ndim != 2 or raw_reference.shape[1:] != (3,):
            raise ValueError("reference_positions must have shape (node_count, 3).")
        if not np.issubdtype(raw_reference.dtype, np.floating):
            raise TypeError(
                "reference_positions must be real floating-point coordinates."
            )
        reference = np.asarray(raw_reference, dtype=np.float64)
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_positions must be finite.")
        node_count = int(reference.shape[0])
        triangles = np.asarray(plan.triangles)
        if int(np.max(triangles)) >= node_count:
            raise ValueError("triangle node indices exceed the reference node count.")
        if plan.fixed_mask_provided and tuple(plan.fixed_mask.shape) != (node_count,):
            raise ValueError("fixed_mask must match the reference node count.")
        fixed = (
            np.asarray(plan.fixed_mask)
            if plan.fixed_mask_provided
            else np.zeros((node_count,), dtype=bool)
        )

        vertices = reference[triangles]
        first = vertices[:, 1] - vertices[:, 0]
        second = vertices[:, 2] - vertices[:, 0]
        double_normal = np.cross(first, second)
        double_area = np.linalg.norm(double_normal, axis=1)
        area = 0.5 * double_area
        if np.any(area <= plan.rest_area_tolerance):
            raise ValueError("Reference shell contains a degenerate triangle.")
        unit_normal = double_normal / double_area[:, None]
        anchor_tangent = first[0] / np.linalg.norm(first[0])
        anchor_binormal = np.cross(unit_normal[0], anchor_tangent)
        anchor_frame = np.stack((anchor_tangent, anchor_binormal, unit_normal[0]), axis=1)
        metric = np.stack(
            (
                np.stack(
                    (np.sum(first * first, axis=1), np.sum(first * second, axis=1)),
                    axis=1,
                ),
                np.stack(
                    (np.sum(first * second, axis=1), np.sum(second * second, axis=1)),
                    axis=1,
                ),
            ),
            axis=1,
        )
        first_length = np.linalg.norm(first, axis=1)
        first_tangent = first / first_length[:, None]
        projection = np.sum(first_tangent * second, axis=1)
        height = np.sqrt(
            np.maximum(np.sum(second * second, axis=1) - projection * projection, 0.0)
        )
        if np.any(height <= plan.rest_area_tolerance):
            raise ValueError("Reference shell contains a singular triangle metric.")
        shape_inverse = np.zeros((plan.triangle_count, 2, 2), dtype=np.float64)
        shape_inverse[:, 0, 0] = 1.0 / first_length
        shape_inverse[:, 0, 1] = -projection / (first_length * height)
        shape_inverse[:, 1, 1] = 1.0 / height

        hinge_triangles = np.asarray(plan.hinge_triangles)
        hinge_edges = np.asarray(plan.hinge_edges)
        rest_dihedral = np.asarray(
            [
                _numpy_dihedral(reference, triangles, pair, edge)
                for pair, edge in zip(hinge_triangles, hinge_edges, strict=True)
            ],
            dtype=np.float64,
        )
        if plan.hinge_count == 0:
            rest_dihedral = np.zeros((0,), dtype=np.float64)
            hinge_weight = np.zeros((0,), dtype=np.float64)
        else:
            hinge_length_squared = np.sum(
                (reference[hinge_edges[:, 1]] - reference[hinge_edges[:, 0]]) ** 2,
                axis=1,
            )
            adjacent_area = area[hinge_triangles[:, 0]] + area[hinge_triangles[:, 1]]
            hinge_weight = 3.0 * hinge_length_squared / adjacent_area

        thickness = np.asarray(plan.thickness)
        density = np.asarray(plan.density)
        triangle_mass = density * thickness * area
        nodal_mass = np.zeros((node_count,), dtype=np.float64)
        np.add.at(nodal_mass, triangles.reshape((-1,)), np.repeat(triangle_mass / 3.0, 3))
        if np.any(nodal_mass <= 0.0):
            raise ValueError("Every shell node must belong to a positive-mass triangle.")

        eigenvalues = np.linalg.eigvalsh(np.asarray(plan.material.membrane_matrix))
        membrane_scale = float(np.max(eigenvalues))
        edge_lengths_squared = np.stack(
            (
                np.sum((vertices[:, 1] - vertices[:, 0]) ** 2, axis=1),
                np.sum((vertices[:, 2] - vertices[:, 1]) ** 2, axis=1),
                np.sum((vertices[:, 0] - vertices[:, 2]) ** 2, axis=1),
            ),
            axis=1,
        )
        maximum_edge_squared = np.max(edge_lengths_squared, axis=1)
        minimum_altitude_squared = 4.0 * area * area / maximum_edge_squared
        triangle_stiffness = membrane_scale * thickness * area / minimum_altitude_squared
        nodal_stiffness = np.zeros((node_count,), dtype=np.float64)
        np.add.at(
            nodal_stiffness, triangles.reshape((-1,)), np.repeat(triangle_stiffness, 3)
        )
        if plan.hinge_count:
            bending_rigidity = (
                float(np.asarray(plan.material.bending_modulus))
                * 0.5
                * (
                    thickness[hinge_triangles[:, 0]] ** 3
                    + thickness[hinge_triangles[:, 1]] ** 3
                )
            )
            hinge_length_squared = np.sum(
                (reference[hinge_edges[:, 1]] - reference[hinge_edges[:, 0]]) ** 2,
                axis=1,
            )
            hinge_stiffness = bending_rigidity * hinge_weight / hinge_length_squared
            np.add.at(
                nodal_stiffness,
                np.asarray(plan.hinge_stencils).reshape((-1,)),
                np.repeat(hinge_stiffness, 4),
            )
        characteristic_length = float(np.sqrt(np.min(edge_lengths_squared)))

        dtype = plan.material.membrane_matrix.dtype
        reference_array = jnp.asarray(reference, dtype=dtype)
        self.plan = plan
        self.reference_positions = reference_array
        self.rest_metric = jnp.asarray(metric, dtype=dtype)
        self.rest_shape_inverse = jnp.asarray(shape_inverse, dtype=dtype)
        self.rest_area = jnp.asarray(area, dtype=dtype)
        self.rest_unit_normal = jnp.asarray(unit_normal, dtype=dtype)
        self.rest_anchor_frame = jnp.asarray(anchor_frame, dtype=dtype)
        self.rest_dihedral = jnp.asarray(rest_dihedral, dtype=dtype)
        self.hinge_weight = jnp.asarray(hinge_weight, dtype=dtype)
        self.nodal_mass = jnp.asarray(nodal_mass, dtype=dtype)
        self.nodal_stiffness = jnp.asarray(nodal_stiffness, dtype=dtype)
        self.fixed_mask = jnp.asarray(fixed)
        self.characteristic_length = jnp.asarray(characteristic_length, dtype=dtype)
        self.node_count = node_count
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-triangular-shell",
                "plan": plan.plan_id,
                "reference": array_tree_fingerprint(reference),
                "rest_metric": array_tree_fingerprint(metric),
                "rest_anchor_frame": array_tree_fingerprint(anchor_frame),
                "rest_area": array_tree_fingerprint(area),
                "rest_dihedral": array_tree_fingerprint(rest_dihedral),
                "hinge_weight": array_tree_fingerprint(hinge_weight),
                "nodal_mass": array_tree_fingerprint(nodal_mass),
            }
        )

    def collision_surface(
        self,
        /,
        *,
        vertex_ids: ArrayLike | None = None,
        body_id: int = 0,
        patch_id: int = 0,
        physical_radius: float | None = None,
    ) -> PreparedCollisionSurface:
        """Expose the shell through the shared exact-map collision interface."""
        identifiers = (
            np.arange(self.node_count, dtype=np.int64)
            if vertex_ids is None
            else np.asarray(vertex_ids)
        )
        if identifiers.shape != (self.node_count,) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise TypeError("vertex_ids must be one integer ID per shell node.")
        radius = (
            0.5 * float(jnp.max(self.plan.thickness))
            if physical_radius is None
            else float(physical_radius)
        )
        policy = ContactPairPolicy(self.node_count)
        topology = CollisionSurfacePlan(
            identifiers,
            ambient_dimension=3,
            faces=self.plan.triangles,
            pair_policy=policy,
            participant_ids=0,
            body_ids=int(body_id),
            patch_ids=int(patch_id),
            static_mask=np.asarray(self.fixed_mask, dtype=bool),
            physical_radius=radius,
        )
        dtype = np.dtype(self.reference_positions.dtype)
        source = ArraySpace((self.node_count, 3), dtype=dtype)
        precision = ContactPrecisionPolicy(
            geometry_dtype=dtype,
            accumulation_dtype=np.float64,
            certification_dtype=np.float64,
            output_dtype=dtype,
        )
        return PreparedCollisionSurface(
            topology,
            self.reference_positions,
            selection_collision_operator(
                source, np.arange(self.node_count, dtype=np.int32)
            ),
            precision=precision,
        )

    def _validate_kinematics(
        self, positions: ArrayLike, velocities: ArrayLike | None, /
    ) -> tuple[Array, Array]:
        position_values = jnp.asarray(positions, dtype=self.reference_positions.dtype)
        if position_values.shape != (self.node_count, 3):
            raise ValueError("positions must match the prepared shell shape.")
        if velocities is None:
            velocity_values = jnp.zeros_like(position_values)
        else:
            velocity_values = jnp.asarray(velocities, dtype=position_values.dtype)
            if velocity_values.shape != position_values.shape:
                raise ValueError("velocities must match positions.")
        return position_values, velocity_values

    def _triangle_normals(self, positions: Array, /) -> tuple[Array, Array, Array]:
        vertices = positions[self.plan.triangles]
        raw = jnp.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        squared = jnp.sum(raw * raw, axis=1)
        tolerance = jnp.asarray(self.plan.rest_area_tolerance, dtype=positions.dtype)
        length = jnp.sqrt(jnp.maximum(squared, tolerance * tolerance))
        return raw / length[:, None], raw, jnp.sqrt(jnp.maximum(squared, 0.0))

    def _stored_energy(self, positions: Array, /) -> tuple[Array, tuple[Array, Array]]:
        vertices = positions[self.plan.triangles]
        current_edges = jnp.stack(
            (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=-1,
        )
        deformation = contract("tij,tjk->tik", current_edges, self.rest_shape_inverse)
        right_cauchy_green = contract("tki,tkj->tij", deformation, deformation)
        strain_tensor = 0.5 * (
            right_cauchy_green - jnp.eye(2, dtype=positions.dtype)[None, :, :]
        )
        strain = jnp.stack(
            (
                strain_tensor[:, 0, 0],
                strain_tensor[:, 1, 1],
                2.0 * strain_tensor[:, 0, 1],
            ),
            axis=1,
        )
        membrane_density = 0.5 * contract(
            "ti,ij,tj->t", strain, self.plan.material.membrane_matrix, strain
        )
        membrane = jnp.sum(membrane_density * self.plan.thickness * self.rest_area)

        if self.plan.hinge_count == 0:
            bending = jnp.asarray(0.0, dtype=positions.dtype)
        else:
            normals, _, _ = self._triangle_normals(positions)
            left_normal = normals[self.plan.hinge_triangles[:, 0]]
            right_normal = normals[self.plan.hinge_triangles[:, 1]]
            edges = (
                positions[self.plan.hinge_edges[:, 1]]
                - positions[self.plan.hinge_edges[:, 0]]
            )
            edge_squared = jnp.sum(edges * edges, axis=1)
            tolerance = jnp.asarray(self.plan.rest_area_tolerance, dtype=positions.dtype)
            edge_length = jnp.sqrt(jnp.maximum(edge_squared, tolerance * tolerance))
            edge_tangent = edges / edge_length[:, None]
            sine = contract(
                "hi,hi->h", edge_tangent, jnp.cross(left_normal, right_normal)
            )
            cosine = jnp.clip(contract("hi,hi->h", left_normal, right_normal), -1.0, 1.0)
            angle = jnp.arctan2(sine, cosine)
            difference = jnp.arctan2(
                jnp.sin(angle - self.rest_dihedral), jnp.cos(angle - self.rest_dihedral)
            )
            left_thickness = self.plan.thickness[self.plan.hinge_triangles[:, 0]]
            right_thickness = self.plan.thickness[self.plan.hinge_triangles[:, 1]]
            rigidity = (
                self.plan.material.bending_modulus
                * 0.5
                * (left_thickness**3 + right_thickness**3)
            )
            bending = 0.5 * jnp.sum(
                rigidity * self.hinge_weight * difference * difference
            )
        total = membrane + bending
        return total, (membrane, bending)

    def _geometry_evidence(self, positions: Array, /) -> ShellGeometryEvidence:
        _, raw_normal, double_area = self._triangle_normals(positions)
        triangle_area = 0.5 * double_area
        area_ratio = triangle_area / self.rest_area
        anchor_vertices = positions[self.plan.triangles[0]]
        anchor_edge = anchor_vertices[1] - anchor_vertices[0]
        tolerance = jnp.asarray(self.plan.rest_area_tolerance, dtype=positions.dtype)
        anchor_length = jnp.sqrt(
            jnp.maximum(jnp.sum(anchor_edge * anchor_edge), tolerance * tolerance)
        )
        anchor_tangent = anchor_edge / anchor_length
        anchor_raw_normal = jnp.cross(
            anchor_vertices[1] - anchor_vertices[0],
            anchor_vertices[2] - anchor_vertices[0],
        )
        anchor_normal_length = jnp.sqrt(
            jnp.maximum(
                jnp.sum(anchor_raw_normal * anchor_raw_normal),
                tolerance * tolerance,
            )
        )
        anchor_normal = anchor_raw_normal / anchor_normal_length
        anchor_binormal = jnp.cross(anchor_normal, anchor_tangent)
        current_anchor_frame = jnp.stack(
            (anchor_tangent, anchor_binormal, anchor_normal), axis=1
        )
        corotation = contract("ij,kj->ik", current_anchor_frame, self.rest_anchor_frame)
        transported_normal = contract("ij,tj->ti", corotation, self.rest_unit_normal)
        orientation_ratio = contract("ti,ti->t", raw_normal, transported_normal) / (
            2.0 * self.rest_area
        )
        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.isfinite(triangle_area)
            & jnp.isfinite(area_ratio)
            & jnp.isfinite(orientation_ratio)
        )
        degenerate = area_ratio <= self.plan.minimum_area_ratio
        inverted = orientation_ratio <= self.plan.minimum_orientation_ratio
        minimum_area = jnp.min(area_ratio)
        minimum_orientation = jnp.min(orientation_ratio)
        valid = jnp.all(finite & ~degenerate & ~inverted)
        return ShellGeometryEvidence(
            triangle_area,
            area_ratio,
            orientation_ratio,
            degenerate,
            inverted,
            finite,
            minimum_area,
            minimum_orientation,
            valid,
        )

    def self_contact_geometry(
        self, positions: ArrayLike, velocities: ArrayLike | None = None, /
    ) -> RigidContactGeometry:
        position_values, velocity_values = self._validate_kinematics(
            positions, velocities
        )
        pairs = self.plan.self_contact_pairs
        left_triangles = self.plan.triangles[pairs[:, 0]]
        right_triangles = self.plan.triangles[pairs[:, 1]]
        left_vertices = position_values[left_triangles]
        right_vertices = position_values[right_triangles]
        tolerance = jnp.asarray(
            self.plan.contact_distance_tolerance, dtype=position_values.dtype
        )
        closest = jax.vmap(_triangle_pair_closest, in_axes=(0, 0, None))(
            left_vertices, right_vertices, tolerance
        )
        (
            left_point,
            right_point,
            left_weights,
            right_weights,
            distance_squared,
            left_feature,
            right_feature,
            feature_margin,
        ) = closest
        distance = jnp.sqrt(jnp.maximum(distance_squared, 0.0))
        displacement = left_point - right_point
        positive_distance = distance > tolerance
        safe_distance = jnp.where(positive_distance, distance, 1.0)
        normal = displacement / safe_distance[:, None]

        triangle_normals, _, current_double_area = self._triangle_normals(position_values)
        left_normal = triangle_normals[pairs[:, 0]]
        right_normal = triangle_normals[pairs[:, 1]]
        fallback = left_normal - right_normal
        fallback_length = jnp.sqrt(
            jnp.maximum(jnp.sum(fallback * fallback, axis=1), tolerance * tolerance)
        )
        fallback = fallback / fallback_length[:, None]
        normal = jnp.where(positive_distance[:, None], normal, fallback)

        left_velocity = contract(
            "ci,cij->cj", left_weights, velocity_values[left_triangles]
        )
        right_velocity = contract(
            "ci,cij->cj", right_weights, velocity_values[right_triangles]
        )
        relative_velocity = left_velocity - right_velocity
        normal_velocity = contract("ci,ci->c", relative_velocity, normal)
        tangential_velocity = relative_velocity - normal_velocity[:, None] * normal
        contact_point = 0.5 * (left_point + right_point)
        left_centroid = jnp.mean(left_vertices, axis=1)
        right_centroid = jnp.mean(right_vertices, axis=1)
        left_arm = contact_point - left_centroid
        right_arm = contact_point - right_centroid

        left_thickness = self.plan.thickness[pairs[:, 0]]
        right_thickness = self.plan.thickness[pairs[:, 1]]
        left_radius = 0.5 * left_thickness
        right_radius = 0.5 * right_thickness
        radius_sum = left_radius + right_radius
        effective_radius = jnp.where(
            radius_sum > 0.0, left_radius * right_radius / radius_sum, 0.0
        )
        gap = distance - radius_sum
        overlap = jnp.maximum(-gap, 0.0)
        pair_area_valid = (current_double_area[pairs[:, 0]] > tolerance) & (
            current_double_area[pairs[:, 1]] > tolerance
        )
        finite = (
            jnp.isfinite(distance)
            & jnp.all(jnp.isfinite(left_point), axis=1)
            & jnp.all(jnp.isfinite(right_point), axis=1)
            & jnp.all(jnp.isfinite(relative_velocity), axis=1)
        )
        degenerate = (
            self.plan.self_contact_slot_valid
            & pair_area_valid
            & finite
            & (~positive_distance & (overlap > 0.0))
        )
        valid = (
            self.plan.self_contact_slot_valid
            & pair_area_valid
            & finite
            & positive_distance
        )
        mask_vector = valid[:, None]
        zero_vector = jnp.zeros_like(normal)
        zero_scalar = jnp.zeros_like(gap)
        zero_angular = jnp.zeros_like(normal)
        successful = jnp.all(finite | ~self.plan.self_contact_slot_valid) & ~jnp.any(
            degenerate
        )
        return RigidContactGeometry(
            jnp.where(mask_vector, normal, zero_vector),
            jnp.where(valid, gap, zero_scalar),
            jnp.where(valid, overlap, zero_scalar),
            jnp.where(valid, effective_radius, zero_scalar),
            jnp.where(mask_vector, contact_point, zero_vector),
            jnp.where(mask_vector, left_arm, zero_vector),
            jnp.where(mask_vector, right_arm, zero_vector),
            jnp.where(mask_vector, left_arm, zero_vector),
            jnp.where(mask_vector, right_arm, zero_vector),
            jnp.where(mask_vector, relative_velocity, zero_vector),
            jnp.where(valid, normal_velocity, zero_scalar),
            jnp.where(mask_vector, tangential_velocity, zero_vector),
            zero_angular,
            zero_angular,
            self.plan.contact_keys,
            jnp.where(valid, left_feature, 0),
            jnp.where(valid, right_feature, 0),
            valid,
            degenerate.astype(jnp.int32),
            jnp.where(valid, feature_margin, jnp.inf),
            successful,
            f"shell-self-contact/{self.prepared_id}",
        )

    def evaluate(
        self, positions: ArrayLike, velocities: ArrayLike | None = None, /
    ) -> ShellEvaluation:
        position_values, velocity_values = self._validate_kinematics(
            positions, velocities
        )
        (stored, (membrane, bending)), gradient = jax.value_and_grad(
            self._stored_energy, has_aux=True
        )(position_values)
        forces = -gradient
        kinetic = 0.5 * jnp.sum(
            self.nodal_mass[:, None] * velocity_values * velocity_values
        )
        geometry = self._geometry_evidence(position_values)
        contact = self.self_contact_geometry(position_values, velocity_values)
        finite = (
            tree_allfinite((forces, membrane, bending, stored, kinetic))
            & jnp.isfinite(stored + kinetic)
            & jnp.all(geometry.finite)
            & contact.successful
        )
        valid = finite & geometry.valid
        return ShellEvaluation(
            forces,
            membrane,
            bending,
            stored,
            kinetic,
            stored + kinetic,
            geometry,
            contact,
            finite,
            valid,
            self.prepared_id,
        )


def _closest_point_on_triangle(
    point: Array, triangle: Array, tolerance: Array, /
) -> tuple[Array, Array, Array, Array]:
    a, b, c = triangle[0], triangle[1], triangle[2]
    first = b - a
    second = c - a
    normal = jnp.cross(first, second)
    normal_squared = jnp.sum(normal * normal)
    safe_normal_squared = jnp.where(
        normal_squared > tolerance * tolerance, normal_squared, 1.0
    )
    projected = point - normal * (jnp.sum((point - a) * normal) / safe_normal_squared)
    d00 = jnp.sum(first * first)
    d01 = jnp.sum(first * second)
    d11 = jnp.sum(second * second)
    relative = projected - a
    d20 = jnp.sum(relative * first)
    d21 = jnp.sum(relative * second)
    determinant = d00 * d11 - d01 * d01
    safe_determinant = jnp.where(
        jnp.abs(determinant) > tolerance * tolerance, determinant, 1.0
    )
    v = (d11 * d20 - d01 * d21) / safe_determinant
    w = (d00 * d21 - d01 * d20) / safe_determinant
    face_weights = jnp.stack((1.0 - v - w, v, w))
    face_valid = (determinant > tolerance * tolerance) & jnp.all(face_weights >= 0.0)

    edge_indices = jnp.asarray(((0, 1), (1, 2), (2, 0)), dtype=jnp.int32)

    def edge_candidate(indices: Array) -> tuple[Array, Array]:
        start = triangle[indices[0]]
        end = triangle[indices[1]]
        edge = end - start
        denominator = jnp.sum(edge * edge)
        safe_denominator = jnp.where(
            denominator > tolerance * tolerance, denominator, 1.0
        )
        parameter = jnp.clip(jnp.sum((point - start) * edge) / safe_denominator, 0.0, 1.0)
        closest_point = start + parameter * edge
        weights = jnp.zeros((3,), dtype=point.dtype)
        weights = weights.at[indices[0]].set(1.0 - parameter)
        weights = weights.at[indices[1]].set(parameter)
        return closest_point, weights

    edge_points, edge_weights = jax.vmap(edge_candidate)(edge_indices)
    candidate_points = jnp.concatenate((projected[None, :], edge_points), axis=0)
    candidate_weights = jnp.concatenate((face_weights[None, :], edge_weights), axis=0)
    distance_squared = jnp.sum((candidate_points - point[None, :]) ** 2, axis=1)
    distance_squared = distance_squared.at[0].set(
        jnp.where(face_valid, distance_squared[0], jnp.inf)
    )
    selected = jnp.argmin(distance_squared)
    features = jnp.asarray((0, 4, 5, 6), dtype=jnp.int32)
    return (
        candidate_points[selected],
        candidate_weights[selected],
        features[selected],
        distance_squared[selected],
    )


def _interior_edge_pair(
    left: Array,
    right: Array,
    left_edge_index: Array,
    right_edge_index: Array,
    tolerance: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    edge_indices = jnp.asarray(((0, 1), (1, 2), (2, 0)), dtype=jnp.int32)
    left_indices = edge_indices[left_edge_index]
    right_indices = edge_indices[right_edge_index]
    left_start = left[left_indices[0]]
    left_end = left[left_indices[1]]
    right_start = right[right_indices[0]]
    right_end = right[right_indices[1]]
    left_direction = left_end - left_start
    right_direction = right_end - right_start
    offset = left_start - right_start
    a = jnp.sum(left_direction * left_direction)
    b = jnp.sum(left_direction * right_direction)
    c = jnp.sum(right_direction * right_direction)
    d = jnp.sum(left_direction * offset)
    e = jnp.sum(right_direction * offset)
    determinant = a * c - b * b
    safe_determinant = jnp.where(
        jnp.abs(determinant) > tolerance * tolerance, determinant, 1.0
    )
    left_parameter = (b * e - c * d) / safe_determinant
    right_parameter = (a * e - b * d) / safe_determinant
    valid = (
        (determinant > tolerance * tolerance)
        & (left_parameter >= 0.0)
        & (left_parameter <= 1.0)
        & (right_parameter >= 0.0)
        & (right_parameter <= 1.0)
    )
    left_point = left_start + left_parameter * left_direction
    right_point = right_start + right_parameter * right_direction
    left_weights = jnp.zeros((3,), dtype=left.dtype)
    left_weights = left_weights.at[left_indices[0]].set(1.0 - left_parameter)
    left_weights = left_weights.at[left_indices[1]].set(left_parameter)
    right_weights = jnp.zeros((3,), dtype=right.dtype)
    right_weights = right_weights.at[right_indices[0]].set(1.0 - right_parameter)
    right_weights = right_weights.at[right_indices[1]].set(right_parameter)
    distance_squared = jnp.sum((left_point - right_point) ** 2)
    return (
        left_point,
        right_point,
        left_weights,
        right_weights,
        jnp.where(valid, distance_squared, jnp.inf),
    )


def _triangle_pair_closest(
    left: Array, right: Array, tolerance: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    left_eye = jnp.eye(3, dtype=left.dtype)
    right_eye = jnp.eye(3, dtype=right.dtype)
    right_results = jax.vmap(_closest_point_on_triangle, in_axes=(0, None, None))(
        left, right, tolerance
    )
    right_points, right_weights, right_features, left_to_right_distance = right_results
    left_results = jax.vmap(_closest_point_on_triangle, in_axes=(0, None, None))(
        right, left, tolerance
    )
    left_points, left_weights, left_features, right_to_left_distance = left_results

    edge_pair_indices = jnp.stack(
        jnp.meshgrid(
            jnp.arange(3, dtype=jnp.int32), jnp.arange(3, dtype=jnp.int32), indexing="ij"
        ),
        axis=-1,
    ).reshape((-1, 2))
    edge_results = jax.vmap(_interior_edge_pair, in_axes=(None, None, 0, 0, None))(
        left, right, edge_pair_indices[:, 0], edge_pair_indices[:, 1], tolerance
    )
    (
        edge_left_points,
        edge_right_points,
        edge_left_weights,
        edge_right_weights,
        edge_distance,
    ) = edge_results

    candidate_left_points = jnp.concatenate((left, left_points, edge_left_points), axis=0)
    candidate_right_points = jnp.concatenate(
        (right_points, right, edge_right_points), axis=0
    )
    candidate_left_weights = jnp.concatenate(
        (left_eye, left_weights, edge_left_weights), axis=0
    )
    candidate_right_weights = jnp.concatenate(
        (right_weights, right_eye, edge_right_weights), axis=0
    )
    candidate_distance = jnp.concatenate(
        (left_to_right_distance, right_to_left_distance, edge_distance), axis=0
    )
    left_candidate_features = jnp.concatenate(
        (
            jnp.asarray((1, 2, 3), dtype=jnp.int32),
            left_features,
            4 + edge_pair_indices[:, 0],
        )
    )
    right_candidate_features = jnp.concatenate(
        (
            right_features,
            jnp.asarray((1, 2, 3), dtype=jnp.int32),
            4 + edge_pair_indices[:, 1],
        )
    )
    selected = jnp.argmin(candidate_distance)
    ordered_distance = jnp.sort(candidate_distance)
    margin = jnp.sqrt(jnp.maximum(ordered_distance[1], 0.0)) - jnp.sqrt(
        jnp.maximum(ordered_distance[0], 0.0)
    )
    return (
        candidate_left_points[selected],
        candidate_right_points[selected],
        candidate_left_weights[selected],
        candidate_right_weights[selected],
        candidate_distance[selected],
        left_candidate_features[selected],
        right_candidate_features[selected],
        margin,
    )


class ShellDynamicsPlan(StrictModule, NonTrainableState):
    shell: TriangularShellPlan
    damping: float = eqx.field(static=True)
    maximum_step_size: float = eqx.field(static=True)
    stability_safety: float = eqx.field(static=True)
    maximum_displacement_ratio: float = eqx.field(static=True)
    fixed_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell: TriangularShellPlan,
        /,
        *,
        damping: float = 0.0,
        maximum_step_size: float = 1.0,
        stability_safety: float = 0.5,
        maximum_displacement_ratio: float = 0.25,
        fixed_tolerance: float = 1.0e-10,
        plan_id: str | None = None,
    ):
        if not isinstance(shell, TriangularShellPlan):
            raise TypeError("shell must be a TriangularShellPlan.")
        damping_ = float(damping)
        maximum_step = float(maximum_step_size)
        safety = float(stability_safety)
        displacement = float(maximum_displacement_ratio)
        fixed = float(fixed_tolerance)
        if not isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and nonnegative.")
        if not isfinite(maximum_step) or maximum_step <= 0.0:
            raise ValueError("maximum_step_size must be positive and finite.")
        if not isfinite(safety) or not (0.0 < safety <= 1.0):
            raise ValueError("stability_safety must lie in (0, 1].")
        if not isfinite(displacement) or displacement <= 0.0:
            raise ValueError("maximum_displacement_ratio must be positive and finite.")
        if not isfinite(fixed) or fixed <= 0.0:
            raise ValueError("fixed_tolerance must be positive and finite.")
        generated = canonical_fingerprint(
            {
                "kind": "shell-dynamics-plan",
                "shell": shell.plan_id,
                "damping": damping_,
                "maximum_step_size": maximum_step,
                "stability_safety": safety,
                "maximum_displacement_ratio": displacement,
                "fixed_tolerance": fixed,
            }
        )
        resolved = generated if plan_id is None else str(plan_id)
        if not resolved:
            raise ValueError("plan_id must be nonempty.")
        self.shell = shell
        self.damping = damping_
        self.maximum_step_size = maximum_step
        self.stability_safety = safety
        self.maximum_displacement_ratio = displacement
        self.fixed_tolerance = fixed
        self.plan_id = resolved

    def prepare(self, reference_positions: ArrayLike, /) -> PreparedShellDynamics:
        return PreparedShellDynamics(self, self.shell.prepare(reference_positions))


class PreparedShellDynamics(StrictModule, NonTrainableState):
    plan: ShellDynamicsPlan
    shell: PreparedTriangularShell
    stable_step_size: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ShellDynamicsPlan, shell: PreparedTriangularShell, /):
        masses = np.asarray(shell.nodal_mass)
        stiffness = np.asarray(shell.nodal_stiffness)
        mobile = ~np.asarray(shell.fixed_mask)
        active_stiffness = mobile & (stiffness > 0.0)
        if np.any(active_stiffness):
            estimated = plan.stability_safety * float(
                np.sqrt(np.min(masses[active_stiffness] / stiffness[active_stiffness]))
            )
            stable = min(plan.maximum_step_size, estimated)
        else:
            stable = plan.maximum_step_size
        if not isfinite(stable) or stable <= 0.0:
            raise ValueError("Prepared shell has no positive stable explicit step size.")
        self.plan = plan
        self.shell = shell
        self.stable_step_size = jnp.asarray(stable, dtype=shell.reference_positions.dtype)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-shell-dynamics",
                "plan": plan.plan_id,
                "shell": shell.prepared_id,
                "stable_step_size": stable,
            }
        )

    def initialize_state(
        self,
        positions: ArrayLike | None = None,
        velocities: ArrayLike | None = None,
        /,
    ) -> ShellState:
        position_values = (
            self.shell.reference_positions
            if positions is None
            else jnp.asarray(positions, dtype=self.shell.reference_positions.dtype)
        )
        if position_values.shape != self.shell.reference_positions.shape:
            raise ValueError("positions must match the prepared shell shape.")
        velocity_values = (
            jnp.zeros_like(position_values)
            if velocities is None
            else jnp.asarray(velocities, dtype=position_values.dtype)
        )
        if velocity_values.shape != position_values.shape:
            raise ValueError("velocities must match positions.")
        fixed = self.shell.fixed_mask[:, None]
        position_values = jnp.where(
            fixed, self.shell.reference_positions, position_values
        )
        velocity_values = jnp.where(fixed, 0.0, velocity_values)
        return ShellState(position_values, velocity_values)

    def step(
        self,
        state: ShellState,
        time: ArrayLike,
        step_size: ArrayLike,
        external_force: ArrayLike | None = None,
        /,
    ) -> ShellStepResult:
        if not isinstance(state, ShellState):
            raise TypeError("state must be a ShellState.")
        if state.positions.shape != self.shell.reference_positions.shape or (
            state.velocities.shape != self.shell.reference_positions.shape
        ):
            raise ValueError("state arrays must match the prepared shell shape.")
        dtype = self.shell.reference_positions.dtype
        time_value = jnp.asarray(time, dtype=dtype)
        requested_step = jnp.asarray(step_size, dtype=dtype)
        scalar_step = time_value.shape == () and requested_step.shape == ()
        if not scalar_step:
            raise ValueError("time and step_size must be scalars.")
        valid_step = (
            jnp.isfinite(time_value)
            & jnp.isfinite(requested_step)
            & (requested_step > 0.0)
        )
        stable_step = valid_step & (requested_step <= self.stable_step_size)
        safe_step = jnp.where(
            valid_step,
            jnp.minimum(requested_step, self.stable_step_size),
            jnp.asarray(0.0, dtype=dtype),
        )
        if external_force is None:
            load = jnp.zeros_like(state.positions)
        else:
            load = jnp.asarray(external_force, dtype=dtype)
            if load.shape != state.positions.shape:
                raise ValueError("external_force must match shell positions.")
        external_finite = jnp.all(jnp.isfinite(load))

        initial = self.shell.evaluate(state.positions, state.velocities)
        fixed = self.shell.fixed_mask[:, None]
        fixed_position_defect = jnp.max(
            jnp.where(
                fixed,
                jnp.abs(state.positions - self.shell.reference_positions),
                0.0,
            ),
            initial=0.0,
        )
        fixed_velocity_defect = jnp.max(
            jnp.where(fixed, jnp.abs(state.velocities), 0.0), initial=0.0
        )
        fixed_valid = (fixed_position_defect <= self.plan.fixed_tolerance) & (
            fixed_velocity_defect <= self.plan.fixed_tolerance
        )
        input_finite = tree_allfinite(state)
        input_valid = input_finite & initial.valid & fixed_valid

        acceleration = (initial.forces + load) / self.shell.nodal_mass[:, None]
        acceleration = jnp.where(fixed, 0.0, acceleration)
        damped_velocity = jnp.exp(-self.plan.damping * safe_step) * state.velocities
        half_velocity = damped_velocity + 0.5 * safe_step * acceleration
        candidate_positions = state.positions + safe_step * half_velocity
        candidate_positions = jnp.where(
            fixed, self.shell.reference_positions, candidate_positions
        )
        provisional = self.shell.evaluate(candidate_positions, half_velocity)
        closing_acceleration = (provisional.forces + load) / self.shell.nodal_mass[
            :, None
        ]
        closing_acceleration = jnp.where(fixed, 0.0, closing_acceleration)
        candidate_velocities = half_velocity + 0.5 * safe_step * closing_acceleration
        candidate_velocities = jnp.where(fixed, 0.0, candidate_velocities)
        candidate = ShellState(candidate_positions, candidate_velocities)
        evaluation = self.shell.evaluate(candidate_positions, candidate_velocities)

        increment = candidate_positions - state.positions
        displacement_ratio = (
            jnp.max(jnp.sqrt(jnp.sum(increment * increment, axis=1)))
            / self.shell.characteristic_length
        )
        displacement_valid = displacement_ratio <= self.plan.maximum_displacement_ratio
        finite = tree_allfinite(candidate) & evaluation.finite
        self_contact_valid = evaluation.self_contact.successful
        successful = (
            valid_step
            & stable_step
            & input_valid
            & external_finite
            & finite
            & evaluation.geometry.valid
            & self_contact_valid
            & displacement_valid
        )

        reasons = jnp.asarray(int(ShellRejectionReason.NONE), dtype=jnp.int32)
        reasons |= jnp.where(
            valid_step, 0, int(ShellRejectionReason.INVALID_STEP)
        ).astype(jnp.int32)
        reasons |= jnp.where(
            stable_step | ~valid_step, 0, int(ShellRejectionReason.UNSTABLE_STEP)
        ).astype(jnp.int32)
        reasons |= jnp.where(
            input_valid, 0, int(ShellRejectionReason.INVALID_STATE)
        ).astype(jnp.int32)
        reasons |= jnp.where(finite, 0, int(ShellRejectionReason.NONFINITE)).astype(
            jnp.int32
        )
        reasons |= jnp.where(
            ~jnp.any(evaluation.geometry.degenerate),
            0,
            int(ShellRejectionReason.DEGENERATE),
        ).astype(jnp.int32)
        reasons |= jnp.where(
            ~jnp.any(evaluation.geometry.inverted),
            0,
            int(ShellRejectionReason.INVERTED),
        ).astype(jnp.int32)
        reasons |= jnp.where(
            displacement_valid, 0, int(ShellRejectionReason.DISPLACEMENT_BOUND)
        ).astype(jnp.int32)
        reasons |= jnp.where(
            fixed_valid, 0, int(ShellRejectionReason.FIXED_NODE_DEFECT)
        ).astype(jnp.int32)
        reasons |= jnp.where(
            self_contact_valid, 0, int(ShellRejectionReason.SELF_CONTACT_GEOMETRY)
        ).astype(jnp.int32)
        reasons |= jnp.where(
            external_finite, 0, int(ShellRejectionReason.EXTERNAL_FORCE)
        ).astype(jnp.int32)
        accepted = tree_where(successful, candidate, state)
        diagnostics = ShellStepDiagnostics(
            self.stable_step_size,
            requested_step,
            displacement_ratio,
            fixed_position_defect,
            fixed_velocity_defect,
            finite,
            successful,
        )
        return ShellStepResult(
            candidate,
            accepted,
            initial,
            evaluation,
            diagnostics,
            successful,
            reasons,
            self.prepared_id,
        )


__all__ = [
    "PreparedShellDynamics",
    "PreparedTriangularShell",
    "ShellDynamicsPlan",
    "ShellEvaluation",
    "ShellGeometryEvidence",
    "ShellMaterialParameters",
    "ShellRejectionReason",
    "ShellState",
    "ShellStepDiagnostics",
    "ShellStepResult",
    "TriangularShellPlan",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....geometry import MeshRegion, TriangleMesh
from ....geometry.surface import SurfaceModel, SurfaceRealization
from ....linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    DualSpace,
    LinearCapabilityError,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    MaterializationPolicy,
    solve as solve_linear,
)
from ._galerkin3d import LaplaceSingleLayerDP0GalerkinPolicy3D
from ._galerkin_quadrature3d import _prepare_surface_pairs_3d
from ._scalar_calderon3d import (
    _BlockedScalarWeakOperator3D,
    _scalar_exception_values,
    _triangle_normal,
    ScalarKernelFamily3D,
)
from ._scalar_trace import UnsupportedScalarBoundarySpaceError


ScalarCrackSide3D = Literal["minus", "plus"]
ScalarScreenSupport3D = SurfaceModel | SurfaceRealization | TriangleMesh | MeshRegion


class UnsupportedScalarScreenJunctionError(ValueError):
    """A requested solve crosses a non-manifold screen junction."""

    evidence: ScalarScreenJunctionEvidence3D

    def __init__(self, message: str, evidence: ScalarScreenJunctionEvidence3D, /):
        self.evidence = evidence
        super().__init__(message)


class ScalarScreenJunctionEvidence3D(StrictModule, NonTrainableState):
    """Exact edge-to-face incidence for non-manifold screen junctions."""

    support_id: str = eqx.field(static=True)
    junction_edges: tuple[tuple[int, int], ...] = eqx.field(static=True)
    incident_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    maximum_incidence: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def junction_edge_count(self) -> int:
        return len(self.junction_edges)


class ScalarScreenTopologyEvidence3D(StrictModule, NonTrainableState):
    """Oriented open-triangle topology and boundary-edge evidence."""

    support_id: str = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    interior_edge_count: int = eqx.field(static=True)
    boundary_edge_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    edge_vertices: Array
    edge_incidence_counts: Array
    boundary_edges: Array
    boundary_edge_lengths: Array
    face_component_ids: Array
    junctions: ScalarScreenJunctionEvidence3D
    orientation: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)


class ScalarCrackSideMetadata3D(StrictModule, NonTrainableState):
    """Two semantically distinct sides of an oriented zero-thickness screen.

    The face normal points from ``minus`` to ``plus``. For the fundamental
    solutions used here, the same-normal derivative traces of a single-layer
    density satisfy ``q_plus - q_minus = -density``. This algebraic jump is the
    one supported crack-side evaluation route; separate one-sided W/K' traces
    are deliberately not claimed.
    """

    minus_name: str = eqx.field(static=True)
    plus_name: str = eqx.field(static=True)
    side_names: tuple[str, str] = eqx.field(static=True)
    oriented_normal: str = eqx.field(static=True)
    derivative_jump: str = eqx.field(static=True)
    evaluation_route: str = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    def __init__(self, minus_name: str = "minus", plus_name: str = "plus", /):
        minus = str(minus_name)
        plus = str(plus_name)
        if not minus or not plus:
            raise ValueError("Crack side names must be non-empty.")
        if minus == plus:
            raise ValueError("Crack sides must have distinct names.")
        self.minus_name = minus
        self.plus_name = plus
        self.side_names = (minus, plus)
        self.oriented_normal = "oriented-face-normal-from-minus-to-plus"
        self.derivative_jump = "q_plus-q_minus=-density"
        self.evaluation_route = "single-layer-same-normal-derivative-jump"
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "scalar-crack-side-metadata-3d-v1",
                "minus": minus,
                "plus": plus,
                "normal": self.oriented_normal,
                "jump": self.derivative_jump,
            }
        )

    def name(self, side: ScalarCrackSide3D, /) -> str:
        if side == "minus":
            return self.minus_name
        if side == "plus":
            return self.plus_name
        raise ValueError("side must be 'minus' or 'plus'.")

    def jump_density(self, density: ArrayLike, /) -> Array:
        values = jnp.asarray(density)
        if values.ndim != 1:
            raise ValueError("Crack jump density must be one-dimensional.")
        return -values


class ScalarScreenAssemblyReport3D(StrictModule, NonTrainableState):
    """Quadrature, topology, resource, accuracy, and restriction evidence."""

    pde: str = eqx.field(static=True)
    kernel_family: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    boundary_edge_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    pair_counts: tuple[int, int, int, int, int] = eqx.field(static=True)
    exception_count: int = eqx.field(static=True)
    quadrature_maximum_errors: Array
    quadrature_evaluations: Array
    preparation_workspace_bytes: int = eqx.field(static=True)
    transient_quadrature_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    dense_operator_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    solve_workspace_bound_bytes: int = eqx.field(static=True)
    finite: Array
    accuracy_supported: Array
    exact_dense_actions: bool = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    restrictions: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class ScalarScreenDirichletResult3D(StrictModule, NonTrainableState):
    """Solved DP0 screen density with a true dense boundary residual."""

    density: Array
    boundary_values: Array
    predicted_boundary_values: Array
    crack_normal_derivative_jump: Array
    residual: Array
    residual_norm: Array
    relative_residual: Array
    linear_result: LinearSolveResult
    assembly_report: ScalarScreenAssemblyReport3D
    valid: Array


class ScalarScreenSingleLayerDP0Galerkin3D(StrictModule, NonTrainableState):
    """Bounded exact-dense DP0 Galerkin screen single-layer preparation."""

    weak_operator: DenseLinearOperator
    strong_operator: DenseLinearOperator
    space: ArraySpace
    face_areas: Array
    topology: ScalarScreenTopologyEvidence3D
    crack_sides: ScalarCrackSideMetadata3D
    kernel: ScalarKernelFamily3D
    assembly_report: ScalarScreenAssemblyReport3D
    preparation_id: str = eqx.field(static=True)

    @property
    def face_count(self) -> int:
        return self.topology.face_count

    def forward(self, density: ArrayLike, /) -> Array:
        """Apply the exact stored dense strong-form single-layer matrix."""

        return self.strong_operator.mv(density)

    def transpose(self, values: ArrayLike, /) -> Array:
        """Apply the exact algebraic transpose of the stored dense matrix."""

        return self.strong_operator.transpose_mv(values)

    def adjoint(self, values: ArrayLike, /) -> Array:
        """Apply the exact conjugate transpose of the stored dense matrix."""

        return self.strong_operator.adjoint_mv(values)

    def crack_jump_density(self, density: ArrayLike, /) -> Array:
        """Evaluate ``q_plus-q_minus=-density`` on the prepared DP0 support."""

        values = self.space.validate(density)
        return self.space.validate(self.crack_sides.jump_density(values))

    def solve_dirichlet(
        self,
        boundary_values: ArrayLike,
        /,
        *,
        linear: LinearSolvePolicy | None = None,
    ) -> ScalarScreenDirichletResult3D:
        """Solve the supported open-screen Dirichlet single-layer equation."""

        values = self.space.validate(boundary_values)
        policy = LinearSolvePolicy(DenseLU()) if linear is None else linear
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        problem = LinearSystem(
            self.strong_operator,
            problem_id=f"scalar-screen-dirichlet:{self.preparation_id}",
        )
        solved = solve_linear(problem, values, policy=policy)
        density = self.space.validate(solved.value)
        predicted = self.forward(density)
        residual = predicted - values
        residual_norm = jnp.linalg.norm(residual)
        scale = jnp.maximum(
            jnp.linalg.norm(values), jnp.asarray(1.0, dtype=residual_norm.dtype)
        )
        relative = residual_norm / scale
        finite = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(relative)
        )
        return ScalarScreenDirichletResult3D(
            density=density,
            boundary_values=values,
            predicted_boundary_values=predicted,
            crack_normal_derivative_jump=self.crack_jump_density(density),
            residual=residual,
            residual_norm=residual_norm,
            relative_residual=relative,
            linear_result=solved,
            assembly_report=self.assembly_report,
            valid=solved.successful & finite & self.assembly_report.accuracy_supported,
        )


def _support_arrays(
    support: ScalarScreenSupport3D, /
) -> tuple[np.ndarray, np.ndarray, str]:
    if isinstance(support, SurfaceRealization):
        support = support.model
    if isinstance(support, SurfaceModel):
        mesh = support.mesh
        vertices = np.asarray(mesh.coordinates, dtype=float)
        faces = np.asarray(mesh.connectivity.cell_vertices, dtype=np.int32)
        kinds = np.asarray(mesh.connectivity.cell_kinds, dtype=np.int32)
        if faces.ndim != 2 or kinds.shape != (faces.shape[0],) or np.any(kinds != 3):
            raise ValueError("Scalar screens require affine triangle cells only.")
        return vertices, faces[:, :3], support.model_id
    if isinstance(support, MeshRegion):
        support = support.triangle_mesh
    if isinstance(support, TriangleMesh):
        return (
            np.asarray(support.vertices, dtype=float),
            np.asarray(support.faces, dtype=np.int32),
            support.source_id,
        )
    raise TypeError(
        "support must be SurfaceModel, SurfaceRealization, TriangleMesh, or MeshRegion."
    )


def _edge_records(faces: np.ndarray, /) -> dict[tuple[int, int], list[tuple[int, int]]]:
    records: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for local in range(3):
            start = int(face[local])
            stop = int(face[(local + 1) % 3])
            edge = (min(start, stop), max(start, stop))
            direction = 1 if (start, stop) == edge else -1
            records.setdefault(edge, []).append((face_index, direction))
    return records


def _components(
    face_count: int,
    records: dict[tuple[int, int], list[tuple[int, int]]],
    /,
) -> tuple[np.ndarray, int]:
    neighbours: list[list[int]] = [[] for _ in range(face_count)]
    for incidents in records.values():
        incident_faces = tuple(item[0] for item in incidents)
        for index, left in enumerate(incident_faces):
            for right in incident_faces[index + 1 :]:
                neighbours[left].append(right)
                neighbours[right].append(left)
    labels = np.full((face_count,), -1, dtype=np.int32)
    count = 0
    for first in range(face_count):
        if labels[first] >= 0:
            continue
        labels[first] = count
        pending = [first]
        while pending:
            face = pending.pop()
            for neighbour in neighbours[face]:
                if labels[neighbour] < 0:
                    labels[neighbour] = count
                    pending.append(neighbour)
        count += 1
    return labels, count


def scalar_screen_junction_evidence_3d(
    support: ScalarScreenSupport3D, /
) -> ScalarScreenJunctionEvidence3D:
    """Return exact non-manifold edge incidence without preparing quadrature."""

    _, faces, support_id = _support_arrays(support)
    records = _edge_records(faces)
    junctions = tuple(
        (edge, tuple(face for face, _ in incidents))
        for edge, incidents in sorted(records.items())
        if len(incidents) > 2
    )
    edges = tuple(item[0] for item in junctions)
    incident_faces = tuple(item[1] for item in junctions)
    maximum = max((len(value) for value in incident_faces), default=0)
    return ScalarScreenJunctionEvidence3D(
        support_id=support_id,
        junction_edges=edges,
        incident_faces=incident_faces,
        maximum_incidence=maximum,
        evidence_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-junction-incidence-3d-v1",
                "support": support_id,
                "edges": edges,
                "incident_faces": incident_faces,
            }
        ),
    )


def _topology_evidence(
    vertices: np.ndarray, faces: np.ndarray, support_id: str, /
) -> ScalarScreenTopologyEvidence3D:
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[0] == 0
        or faces.shape[1] != 3
    ):
        raise ValueError(
            "Scalar screens require non-empty three-dimensional triangle arrays."
        )
    if (
        np.any(~np.isfinite(vertices))
        or np.any(faces < 0)
        or np.any(faces >= vertices.shape[0])
    ):
        raise ValueError("Scalar screen geometry and incidence must be finite and valid.")
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    doubled_areas = np.linalg.norm(crosses, axis=1)
    scale = max(float(np.max(np.ptp(vertices, axis=0))), 1.0)
    if np.any(~np.isfinite(doubled_areas)) or np.any(
        doubled_areas <= 64.0 * np.finfo(float).eps * scale * scale
    ):
        raise ValueError("Scalar screen triangles must be finite and nondegenerate.")

    records = _edge_records(faces)
    junction_pairs = tuple(
        (edge, tuple(face for face, _ in incidents))
        for edge, incidents in sorted(records.items())
        if len(incidents) > 2
    )
    junctions = ScalarScreenJunctionEvidence3D(
        support_id=support_id,
        junction_edges=tuple(item[0] for item in junction_pairs),
        incident_faces=tuple(item[1] for item in junction_pairs),
        maximum_incidence=max((len(item[1]) for item in junction_pairs), default=0),
        evidence_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-junction-incidence-3d-v1",
                "support": support_id,
                "junctions": junction_pairs,
            }
        ),
    )
    if junctions.junction_edge_count:
        raise UnsupportedScalarScreenJunctionError(
            "Scalar screen junction solves are unsupported; edge incidence evidence is attached.",
            junctions,
        )
    if any(
        len(incidents) == 2 and incidents[0][1] == incidents[1][1]
        for incidents in records.values()
    ):
        raise ValueError(
            "Scalar screen triangles must have a consistent global orientation."
        )

    labels, component_count = _components(faces.shape[0], records)
    boundary = tuple(
        edge for edge, incidents in sorted(records.items()) if len(incidents) == 1
    )
    if not boundary:
        raise ValueError(
            "Scalar screen preparation requires an open support; closed surfaces must use a closed-surface formulation."
        )
    component_has_boundary = np.zeros((component_count,), dtype=bool)
    for incidents in records.values():
        if len(incidents) == 1:
            component_has_boundary[labels[incidents[0][0]]] = True
    if not np.all(component_has_boundary):
        raise ValueError(
            "Every scalar screen component must be open; mixed closed components are rejected."
        )
    boundary_degrees: dict[int, int] = {}
    for start, stop in boundary:
        boundary_degrees[start] = boundary_degrees.get(start, 0) + 1
        boundary_degrees[stop] = boundary_degrees.get(stop, 0) + 1
    if any(degree != 2 for degree in boundary_degrees.values()):
        raise ValueError(
            "Scalar screen boundary edges must form regular closed boundary loops."
        )

    edge_vertices_host = np.asarray(tuple(sorted(records)), dtype=np.int32)
    incidences = np.asarray(
        tuple(len(records[tuple(edge)]) for edge in edge_vertices_host), dtype=np.int32
    )
    boundary_host = np.asarray(boundary, dtype=np.int32).reshape((-1, 2))
    boundary_lengths = np.linalg.norm(
        vertices[boundary_host[:, 1]] - vertices[boundary_host[:, 0]], axis=1
    )
    topology_id = canonical_fingerprint(
        {
            "kind": "oriented-open-scalar-screen-topology-3d-v1",
            "support": support_id,
            "faces": array_tree_fingerprint(faces),
            "edges": array_tree_fingerprint(edge_vertices_host),
            "incidence": array_tree_fingerprint(incidences),
        }
    )
    return ScalarScreenTopologyEvidence3D(
        support_id=support_id,
        vertex_count=vertices.shape[0],
        face_count=faces.shape[0],
        edge_count=len(records),
        interior_edge_count=sum(len(value) == 2 for value in records.values()),
        boundary_edge_count=len(boundary),
        component_count=component_count,
        edge_vertices=jnp.asarray(edge_vertices_host),
        edge_incidence_counts=jnp.asarray(incidences),
        boundary_edges=jnp.asarray(boundary_host),
        boundary_edge_lengths=jnp.asarray(boundary_lengths),
        face_component_ids=jnp.asarray(labels),
        junctions=junctions,
        orientation="consistent-oriented-face-normal-from-minus-to-plus",
        topology_id=topology_id,
    )


def prepare_scalar_screen_single_layer_dp0_3d(
    support: ScalarScreenSupport3D,
    /,
    *,
    kernel: ScalarKernelFamily3D | None = None,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
    dense_policy: MaterializationPolicy | None = None,
    max_solve_workspace_bytes: int = 256 * 1024 * 1024,
    crack_sides: ScalarCrackSideMetadata3D | None = None,
) -> ScalarScreenSingleLayerDP0Galerkin3D:
    """Prepare a bounded exact-dense DP0 open-screen Dirichlet single layer."""

    family = ScalarKernelFamily3D.laplace() if kernel is None else kernel
    selected = LaplaceSingleLayerDP0GalerkinPolicy3D() if policy is None else policy
    dense = MaterializationPolicy() if dense_policy is None else dense_policy
    sides = ScalarCrackSideMetadata3D() if crack_sides is None else crack_sides
    if not isinstance(family, ScalarKernelFamily3D):
        raise TypeError("kernel must be ScalarKernelFamily3D or None.")
    if not isinstance(selected, LaplaceSingleLayerDP0GalerkinPolicy3D):
        raise TypeError("policy must be LaplaceSingleLayerDP0GalerkinPolicy3D or None.")
    if not isinstance(dense, MaterializationPolicy):
        raise TypeError("dense_policy must be MaterializationPolicy or None.")
    if not isinstance(sides, ScalarCrackSideMetadata3D):
        raise TypeError("crack_sides must be ScalarCrackSideMetadata3D or None.")
    solve_limit = int(max_solve_workspace_bytes)
    if solve_limit <= 0:
        raise ValueError("max_solve_workspace_bytes must be positive.")

    vertices, faces, support_id = _support_arrays(support)
    topology = _topology_evidence(vertices, faces, support_id)
    face_count = topology.face_count
    entries = face_count * face_count
    scalar_dtype = np.dtype(
        np.complex128 if family.scalar_dtype == "complex" else np.float64
    )
    matrix_bytes = entries * scalar_dtype.itemsize
    if entries > dense.max_entries:
        raise LinearCapabilityError("Scalar screen dense matrix exceeds max_entries.")
    if matrix_bytes > dense.max_bytes:
        raise LinearCapabilityError("Scalar screen dense matrix exceeds max_bytes.")
    solve_workspace = 4 * matrix_bytes
    if solve_workspace > solve_limit:
        raise LinearCapabilityError(
            "Scalar screen dense solve exceeds max_solve_workspace_bytes."
        )

    max_diameter = max(
        float(
            np.max(np.linalg.norm(triangle[:, None, :] - triangle[None, :, :], axis=-1))
        )
        for triangle in vertices[faces]
    )
    if family.parameter * max_diameter > float(selected.regular_order):
        raise ValueError(
            "Kernel parameter exceeds the bounded screen panel-frequency envelope."
        )
    pair_data = _prepare_surface_pairs_3d(
        jnp.asarray(vertices),
        jnp.asarray(faces),
        regular_order=selected.regular_order,
        singular_order=selected.singular_order,
        near_order=selected.near_order,
        near_ratio=selected.near_ratio,
        near_max_depth=selected.near_max_depth,
        absolute_tolerance=selected.absolute_tolerance,
        relative_tolerance=selected.relative_tolerance,
        max_exception_pairs=selected.max_exception_pairs,
        max_preparation_workspace_bytes=selected.max_preparation_workspace_bytes,
        max_resident_bytes=selected.max_resident_bytes,
    )
    if family.family == "laplace":
        exception_values = selected.precision.accumulation(pair_data.values)
        maximum_errors = selected.precision.decision(pair_data.maximum_errors)
        evaluations = pair_data.evaluations
        quadrature_supported = jnp.all(pair_data.supported)
    else:
        single_host, _, errors_host, evaluations_host = _scalar_exception_values(
            vertices, faces, pair_data, family, selected
        )
        exception_values = selected.precision.accumulation(
            jnp.asarray(single_host, dtype=scalar_dtype)
        )
        maximum_errors = selected.precision.decision(jnp.asarray(errors_host[:1]))
        evaluations = jnp.asarray(evaluations_host[:1], dtype=jnp.int64)
        quadrature_supported = jnp.asarray(True)
    pair_data = eqx.tree_at(
        lambda data: (data.regular_points, data.regular_weights),
        pair_data,
        (
            selected.precision.evaluation(pair_data.regular_points),
            selected.precision.accumulation(pair_data.regular_weights),
        ),
    )
    areas_host = 0.5 * np.linalg.norm(
        np.cross(
            vertices[faces][:, 1] - vertices[faces][:, 0],
            vertices[faces][:, 2] - vertices[faces][:, 0],
        ),
        axis=1,
    )
    areas = selected.precision.accumulation(jnp.asarray(areas_host))
    normals = selected.precision.evaluation(
        jnp.asarray(
            np.stack(tuple(_triangle_normal(triangle) for triangle in vertices[faces]))
        )
    )
    space = ArraySpace(
        (face_count,),
        dtype=jnp.asarray(exception_values).dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-dp0-space-3d-v1",
                "topology": topology.topology_id,
                "kernel-dtype": family.scalar_dtype,
            }
        ),
    )
    weak_blocked = _BlockedScalarWeakOperator3D(
        pair_data,
        exception_values,
        normals,
        family,
        "single",
        space,
        target_block_size=selected.target_block_size,
        source_block_size=selected.source_block_size,
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-single-layer-weak-blocked-3d-v1",
                "topology": topology.topology_id,
                "policy": selected.policy_id,
                "kernel": family.kernel_id,
            }
        ),
    )
    basis = jnp.eye(face_count, dtype=space.dtype)
    weak_matrix = jax.vmap(weak_blocked.mv, in_axes=1, out_axes=1)(basis)
    strong_matrix = weak_matrix / areas[:, None]
    weak = DenseLinearOperator(
        weak_matrix,
        source=space,
        target=DualSpace(space),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-single-layer-weak-dense-3d-v1",
                "topology": topology.topology_id,
                "kernel": family.kernel_id,
            }
        ),
    )
    strong = DenseLinearOperator(
        strong_matrix,
        source=space,
        target=space,
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-single-layer-strong-dense-3d-v1",
                "topology": topology.topology_id,
                "kernel": family.kernel_id,
            }
        ),
    )
    dense_operator_bytes = int(weak_matrix.nbytes + strong_matrix.nbytes)
    topology_bytes = int(
        topology.edge_vertices.nbytes
        + topology.edge_incidence_counts.nbytes
        + topology.boundary_edges.nbytes
        + topology.boundary_edge_lengths.nbytes
        + topology.face_component_ids.nbytes
    )
    resident_bytes = dense_operator_bytes + int(areas.nbytes) + topology_bytes
    if resident_bytes > selected.max_resident_bytes:
        raise LinearCapabilityError(
            "Scalar screen prepared state exceeds max_resident_bytes."
        )
    finite = (
        jnp.all(jnp.isfinite(weak_matrix))
        & jnp.all(jnp.isfinite(strong_matrix))
        & jnp.all(jnp.isfinite(areas))
        & jnp.all(areas > 0.0)
        & jnp.all(jnp.isfinite(maximum_errors))
    )
    supported = finite & quadrature_supported
    restrictions = (
        "open-oriented-manifold-triangle-screens-only",
        "Dirichlet-single-layer-density-solve-only",
        "no-hypersingular-W-or-closed-Calderon-route",
        "no-junction-or-nonmanifold-solve",
        "no-edge-singularity-enrichment-or-continuum-error-certificate",
        "dense-resource-bounded-route-only",
    )
    report_id = canonical_fingerprint(
        {
            "kind": "scalar-screen-assembly-report-3d-v1",
            "topology": topology.topology_id,
            "kernel": family.kernel_id,
            "policy": selected.policy_id,
            "matrix": array_tree_fingerprint(strong_matrix),
            "errors": array_tree_fingerprint(maximum_errors),
        }
    )
    report = ScalarScreenAssemblyReport3D(
        pde=family.pde,
        kernel_family=family.family,
        formulation="open-screen-DP0-Galerkin-Dirichlet-single-layer",
        support_id=support_id,
        topology_id=topology.topology_id,
        policy_id=selected.policy_id,
        kernel_id=family.kernel_id,
        face_count=face_count,
        boundary_edge_count=topology.boundary_edge_count,
        component_count=topology.component_count,
        pair_counts=pair_data.counts,
        exception_count=int(pair_data.targets.shape[0]),
        quadrature_maximum_errors=maximum_errors,
        quadrature_evaluations=evaluations,
        preparation_workspace_bytes=max(
            pair_data.preparation_workspace_bytes,
            int(weak_blocked.action_workspace_bytes + matrix_bytes),
        ),
        transient_quadrature_bytes=pair_data.resident_bytes,
        resident_bytes=resident_bytes,
        dense_operator_bytes=dense_operator_bytes,
        action_workspace_bytes_per_rhs=2 * face_count * scalar_dtype.itemsize,
        solve_workspace_bound_bytes=solve_workspace,
        finite=finite,
        accuracy_supported=supported,
        exact_dense_actions=True,
        continuum_certified=False,
        restrictions=restrictions,
        report_id=report_id,
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "scalar-screen-single-layer-preparation-3d-v1",
            "operator": strong.operator_id,
            "report": report.report_id,
            "crack-sides": sides.metadata_id,
        }
    )
    return ScalarScreenSingleLayerDP0Galerkin3D(
        weak_operator=weak,
        strong_operator=strong,
        space=space,
        face_areas=areas,
        topology=topology,
        crack_sides=sides,
        kernel=family,
        assembly_report=report,
        preparation_id=preparation_id,
    )


def prepare_scalar_screen_hypersingular_dp0_3d(
    support: ScalarScreenSupport3D, /, **kwargs
):
    """Reject W before any screen geometry or quadrature preparation."""

    del support, kwargs
    raise UnsupportedScalarBoundarySpaceError(
        "Open-screen hypersingular W requires an H^1/2-conforming edge-aware space; DP0 is rejected before preparation."
    )


def prepare_scalar_screen_calderon_dp0_3d(support: ScalarScreenSupport3D, /, **kwargs):
    """Reject closed Calderón semantics on a two-sided open screen."""

    del support, kwargs
    raise UnsupportedScalarBoundarySpaceError(
        "Closed-surface Calderón preparation is not defined on an open screen; use the Dirichlet single-layer route."
    )


def prepare_scalar_screen_junction_solve_3d(support: ScalarScreenSupport3D, /, **kwargs):
    """Retain junction incidence and reject before operator preparation."""

    del kwargs
    evidence = scalar_screen_junction_evidence_3d(support)
    raise UnsupportedScalarScreenJunctionError(
        "Scalar screen junction solves are unsupported; no operator was prepared.",
        evidence,
    )


__all__ = [
    "ScalarCrackSide3D",
    "ScalarCrackSideMetadata3D",
    "ScalarScreenAssemblyReport3D",
    "ScalarScreenDirichletResult3D",
    "ScalarScreenJunctionEvidence3D",
    "ScalarScreenSingleLayerDP0Galerkin3D",
    "ScalarScreenSupport3D",
    "ScalarScreenTopologyEvidence3D",
    "UnsupportedScalarScreenJunctionError",
    "prepare_scalar_screen_calderon_dp0_3d",
    "prepare_scalar_screen_hypersingular_dp0_3d",
    "prepare_scalar_screen_junction_solve_3d",
    "prepare_scalar_screen_single_layer_dp0_3d",
    "scalar_screen_junction_evidence_3d",
]

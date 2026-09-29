#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization._boundary_trace_space import (
    boundary_geometry_revision,
    BoundaryTraceSpaceCapability,
    CauchyTraceCapability,
    diagonal_gram_trace_space,
    sparse_gram_trace_space,
)
from ....geometry import MeshRegion
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    DualSpace,
    FunctionLinearOperator,
    OperatorProperties,
)
from ....sparse import EdgeRelation, SparseCoordinateOperator
from ._galerkin3d import LaplaceSingleLayerDP0GalerkinPolicy3D
from ._galerkin_quadrature3d import (
    _duffy_rule,
    _map_triangle,
    _regular_rule,
    _remap_edge,
    _remap_vertex,
)
from ._scalar_calderon3d import (
    prepare_scalar_calderon_dp0_3d,
    ScalarCalderonDP0Galerkin3D,
    ScalarKernelFamily3D,
)
from ._scalar_trace import SCALAR_TRACE_CONVENTION_3D


class ScalarBoundarySpaces3D(StrictModule, NonTrainableState):
    """Canonical continuous-P1 Dirichlet and DP0 Neumann trace pairing."""

    vertices: Array
    faces: Array
    face_areas: Array
    cross_mass: Array
    dirichlet_space: ArraySpace
    neumann_space: ArraySpace
    vertex_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    spaces_id: str = eqx.field(static=True)

    def cauchy_trace_capability(
        self, /, *, gram_tolerance: float = 1.0e-13
    ) -> CauchyTraceCapability:
        """Publish the P1 Dirichlet / DP0 Neumann Cauchy data with area pairings.

        The Dirichlet coordinates are paired by the exact P1 area Gram map
        (local stencil `A (1 + δ_ij) / 12`) with a prepared conjugate-gradient
        Riesz solve to `gram_tolerance`, the Neumann coordinates by the face
        areas, and the duality is the sparse cross mass `(B φ)_f = ∫_f φ dA`
        between the native spaces. The Neumann trace differentiates along the
        interior-to-exterior normal: `MeshRegion` orients every closed surface
        component outward, whatever winding was declared.
        """
        cells = np.asarray(self.faces, dtype=np.int32)
        areas = np.asarray(self.face_areas, dtype=np.float64)
        stencil = (areas / 12.0)[:, None, None] * (1.0 + np.eye(3))[None, :, :]
        dirichlet_gram, dirichlet_mass = sparse_gram_trace_space(
            np.repeat(cells, 3, axis=1).reshape((-1,)),
            np.tile(cells, (1, 3)).reshape((-1,)),
            stencil.reshape((-1,)),
            size=self.vertex_count,
            dtype=self.dirichlet_space.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "closed-surface-continuous-p1-trace-3d",
                    "spaces": self.spaces_id,
                }
            ),
            gram_tolerance=gram_tolerance,
        )
        neumann_gram, neumann_mass = diagonal_gram_trace_space(
            areas,
            dtype=self.neumann_space.dtype,
            space_id=canonical_fingerprint(
                {"kind": "closed-surface-dp0-trace-3d", "spaces": self.spaces_id}
            ),
        )
        duality = SparseCoordinateOperator(
            EdgeRelation(
                cells.reshape((-1,)),
                np.repeat(np.arange(self.face_count, dtype=np.int32), 3),
                source_size=self.vertex_count,
                target_size=self.face_count,
            ),
            jnp.asarray(np.repeat(areas / 3.0, 3), dtype=jnp.float64),
            source=self.dirichlet_space,
            target=DualSpace(self.neumann_space),
            operator_id=f"{self.spaces_id}:cauchy-duality",
            accumulation_dtype=np.result_type(
                self.dirichlet_space.dtype, self.neumann_space.dtype
            ),
        )
        revision = boundary_geometry_revision(self.vertices, cells)
        return CauchyTraceCapability(
            BoundaryTraceSpaceCapability(
                owner_id=self.spaces_id,
                quantity="dirichlet",
                representation="continuous-p1",
                coefficient_space=self.dirichlet_space,
                gram_space=dirichlet_gram,
                mass=dirichlet_mass,
                ambient_dimension=3,
                revision_id=revision,
            ),
            BoundaryTraceSpaceCapability(
                owner_id=self.spaces_id,
                quantity="neumann",
                representation="dp0",
                coefficient_space=self.neumann_space,
                gram_space=neumann_gram,
                mass=neumann_mass,
                ambient_dimension=3,
                revision_id=revision,
            ),
            duality,
            interior=SCALAR_TRACE_CONVENTION_3D.interior,
            convention_id=SCALAR_TRACE_CONVENTION_3D.convention_id,
        )


class ScalarConformingAssemblyEvidence3D(StrictModule, NonTrainableState):
    maximum_pair_order_defect: float = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    maue_regularized: bool = eqx.field(static=True)
    dp0_hypersingular_supported: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class _P1HypersingularAction3D(StrictModule, NonTrainableState):
    single_layer: AbstractLinearOperator
    correction: AbstractLinearOperator | None
    faces: Array
    surface_curls: Array
    vertex_count: int = eqx.field(static=True)

    def mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape != (self.vertex_count,):
            raise ValueError("P1 hypersingular input must have one value per vertex.")
        local = value[self.faces]
        curl_density = ein.contract(
            "fid,fi->fd", self.surface_curls, local, backend="jax"
        )
        acted = jax.vmap(self.single_layer.mv, in_axes=1, out_axes=1)(curl_density)
        local_result = ein.contract(
            "fid,fd->fi", self.surface_curls, acted, backend="jax"
        )
        result = jnp.zeros((self.vertex_count,), dtype=local_result.dtype)
        result = result.at[self.faces.reshape((-1,))].add(local_result.reshape((-1,)))
        if self.correction is not None:
            result = result + self.correction.mv(value)
        return result

    def transpose_mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape != (self.vertex_count,):
            raise ValueError("P1 hypersingular input must have one value per vertex.")
        local = value[self.faces]
        curl_density = ein.contract(
            "fid,fi->fd", self.surface_curls, local, backend="jax"
        )
        acted = jax.vmap(self.single_layer.transpose_mv, in_axes=1, out_axes=1)(
            curl_density
        )
        local_result = ein.contract(
            "fid,fd->fi", self.surface_curls, acted, backend="jax"
        )
        result = jnp.zeros((self.vertex_count,), dtype=local_result.dtype)
        result = result.at[self.faces.reshape((-1,))].add(local_result.reshape((-1,)))
        if self.correction is not None:
            result = result + self.correction.transpose_mv(value)
        return result


class ScalarCalderonGalerkin3D(StrictModule, NonTrainableState):
    """Conforming mixed P1/DP0 scalar Calderón operators on a closed surface."""

    spaces: ScalarBoundarySpaces3D
    single_layer_weak: AbstractLinearOperator
    double_layer_weak: AbstractLinearOperator
    adjoint_double_layer_weak: AbstractLinearOperator
    hypersingular_weak: AbstractLinearOperator
    cross_mass: AbstractLinearOperator
    kernel: ScalarKernelFamily3D
    dp0_calderon: ScalarCalderonDP0Galerkin3D
    evidence: ScalarConformingAssemblyEvidence3D
    prepared_id: str = eqx.field(static=True)


class ClosedScalarCalderon3D(StrictModule, NonTrainableState):
    """Closed-surface Calderón product with explicit P1 constant gauge."""

    calderon: ScalarCalderonGalerkin3D
    p1_constant: Array
    gauge_mass: Array
    product_id: str = eqx.field(static=True)


def _barycentric(reference: np.ndarray, /) -> np.ndarray:
    return np.stack(
        (1.0 - reference[:, 0] - reference[:, 1], reference[:, 0], reference[:, 1]),
        axis=1,
    )


def _pair_rule(
    faces: np.ndarray,
    target: int,
    source: int,
    regular_order: int,
    singular_order: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = sorted(set(map(int, faces[target])) & set(map(int, faces[source])))
    if target == source:
        return _duffy_rule(singular_order, "coincident")
    if len(shared) == 2:
        test_reference, source_reference, weights = _duffy_rule(
            singular_order, "shared-edge"
        )
        test_local = tuple(
            int(np.flatnonzero(faces[target] == value)[0]) for value in shared
        )
        source_local = tuple(
            int(np.flatnonzero(faces[source] == value)[0]) for value in shared
        )
        return (
            _remap_edge(test_reference, *test_local),
            _remap_edge(source_reference, *source_local),
            weights,
        )
    if len(shared) == 1:
        test_reference, source_reference, weights = _duffy_rule(
            singular_order, "shared-vertex"
        )
        test_local = int(np.flatnonzero(faces[target] == shared[0])[0])
        source_local = int(np.flatnonzero(faces[source] == shared[0])[0])
        return (
            _remap_vertex(test_reference, test_local),
            _remap_vertex(source_reference, source_local),
            weights,
        )
    points, weights = _regular_rule(regular_order)
    count = points.shape[0]
    return (
        np.repeat(points, count, axis=0),
        np.tile(points, (count, 1)),
        (weights[:, None] * weights[None, :]).reshape((-1,)),
    )


def _kernel_values(
    differences: np.ndarray,
    source_normal: np.ndarray,
    kernel: ScalarKernelFamily3D,
    derivative: bool,
    /,
) -> np.ndarray:
    radius = np.linalg.norm(differences, axis=-1)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("Conforming scalar quadrature encountered a singular point.")
    if kernel.family == "laplace":
        factor = np.ones_like(radius)
    elif kernel.family == "modified-helmholtz":
        scaled = kernel.parameter * radius
        factor = np.exp(-scaled) * ((1.0 + scaled) if derivative else 1.0)
    else:
        scaled = kernel.parameter * radius
        factor = np.exp(1j * scaled) * ((1.0 - 1j * scaled) if derivative else 1.0)
    if not derivative:
        return factor / (4.0 * pi * radius)
    return factor * (differences @ source_normal) / (4.0 * pi * radius**3)


def _assemble_p1_blocks(
    vertices: np.ndarray,
    faces: np.ndarray,
    kernel: ScalarKernelFamily3D,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    triangles = vertices[faces]
    normals_raw = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    jacobians = np.linalg.norm(normals_raw, axis=1)
    normals = normals_raw / jacobians[:, None]
    vertex_count = vertices.shape[0]
    double_layer = np.zeros(
        (faces.shape[0], vertex_count),
        dtype=np.complex128 if kernel.family == "outgoing-helmholtz" else np.float64,
    )
    normal_single = np.zeros((vertex_count, vertex_count), dtype=double_layer.dtype)
    maximum_defect = 0.0
    for target in range(faces.shape[0]):
        for source in range(faces.shape[0]):
            orders = (policy.singular_order, policy.singular_order + 2)
            pair_values = []
            normal_values = []
            for order in orders:
                test_reference, source_reference, weights = _pair_rule(
                    faces,
                    target,
                    source,
                    max(policy.regular_order, order),
                    order,
                )
                test_points = _map_triangle(triangles[target], test_reference)
                source_points = _map_triangle(triangles[source], source_reference)
                differences = test_points - source_points
                source_basis = _barycentric(source_reference)
                test_basis = _barycentric(test_reference)
                scale = weights * jacobians[target] * jacobians[source]
                double_kernel = _kernel_values(differences, normals[source], kernel, True)
                single_kernel = _kernel_values(
                    differences, normals[source], kernel, False
                )
                pair_values.append(
                    np.sum((scale * double_kernel)[:, None] * source_basis, axis=0)
                )
                normal_values.append(
                    ein.contract(
                        "q,qi,qj->ij",
                        scale
                        * single_kernel
                        * float(np.dot(normals[target], normals[source])),
                        test_basis,
                        source_basis,
                    )
                )
            defect = max(
                float(np.max(np.abs(pair_values[1] - pair_values[0]))),
                float(np.max(np.abs(normal_values[1] - normal_values[0]))),
            )
            maximum_defect = max(maximum_defect, defect)
            double_layer[target, faces[source]] += pair_values[1]
            normal_single[np.ix_(faces[target], faces[source])] += normal_values[1]
    return double_layer, normal_single, maximum_defect


def _surface_curls(vertices: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    triangles = vertices[faces]
    normal_raw = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    doubled_areas = np.linalg.norm(normal_raw, axis=1)
    normals = normal_raw / doubled_areas[:, None]
    gradients = (
        np.stack(
            (
                np.cross(normals, triangles[:, 2] - triangles[:, 1]),
                np.cross(normals, triangles[:, 0] - triangles[:, 2]),
                np.cross(normals, triangles[:, 1] - triangles[:, 0]),
            ),
            axis=1,
        )
        / doubled_areas[:, None, None]
    )
    return np.cross(normals[:, None, :], gradients)


def prepare_scalar_calderon_3d(
    region: MeshRegion,
    /,
    *,
    kernel: ScalarKernelFamily3D | None = None,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
    numeric_version: str = "0",
) -> ScalarCalderonGalerkin3D:
    """Prepare conforming P1 W/K and DP0 V/K' without a DP0-W projection."""

    family = ScalarKernelFamily3D.laplace() if kernel is None else kernel
    if not isinstance(family, ScalarKernelFamily3D):
        raise TypeError("kernel must be ScalarKernelFamily3D or None.")
    selected = LaplaceSingleLayerDP0GalerkinPolicy3D() if policy is None else policy
    if not isinstance(selected, LaplaceSingleLayerDP0GalerkinPolicy3D):
        raise TypeError("policy must be LaplaceSingleLayerDP0GalerkinPolicy3D or None.")
    dp0 = prepare_scalar_calderon_dp0_3d(
        region,
        kernel=family,
        policy=selected,
        numeric_version=numeric_version,
    )
    vertices = np.asarray(region.triangle_mesh.vertices, dtype=np.float64)
    faces = np.asarray(region.triangle_mesh.faces, dtype=np.int32)
    face_areas = np.asarray(dp0.face_areas, dtype=np.float64)
    dense_entries = faces.shape[0] * vertices.shape[0] + vertices.shape[0] ** 2
    dense_bytes = (
        dense_entries
        * np.dtype(
            np.complex128 if family.family == "outgoing-helmholtz" else np.float64
        ).itemsize
    )
    if dense_bytes > selected.max_resident_bytes:
        raise ValueError("Conforming scalar blocks exceed max_resident_bytes.")
    double_matrix, normal_single, maximum_defect = _assemble_p1_blocks(
        vertices, faces, family, selected
    )
    if maximum_defect > max(
        selected.absolute_tolerance,
        selected.relative_tolerance * max(float(np.max(np.abs(double_matrix))), 1.0),
    ):
        raise ValueError(
            "Conforming scalar pair quadrature did not meet the declared tolerance."
        )
    cross_mass_matrix = np.zeros((faces.shape[0], vertices.shape[0]), dtype=np.float64)
    for face_index, face in enumerate(faces):
        cross_mass_matrix[face_index, face] = face_areas[face_index] / 3.0
    correction_matrix = None
    if family.family == "modified-helmholtz":
        correction_matrix = family.parameter**2 * normal_single
    elif family.family == "outgoing-helmholtz":
        correction_matrix = -(family.parameter**2) * normal_single
    correction = (
        None
        if correction_matrix is None
        else DenseLinearOperator(
            jnp.asarray(correction_matrix),
            operator_id=canonical_fingerprint(
                {
                    "kind": "scalar-hypersingular-normal-correction",
                    "kernel": family.kernel_id,
                    "matrix": array_tree_fingerprint(correction_matrix),
                }
            ),
        )
    )
    curls = _surface_curls(vertices, faces)
    hypersingular_action = _P1HypersingularAction3D(
        single_layer=dp0.single_layer_weak,
        correction=correction,
        faces=jnp.asarray(faces),
        surface_curls=jnp.asarray(curls),
        vertex_count=vertices.shape[0],
    )
    p1_space = ArraySpace(
        (vertices.shape[0],),
        dtype=dp0.space.dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "closed-surface-continuous-p1-space-3d",
                "binding": dp0._binding.binding_id,
                "faces": array_tree_fingerprint(faces),
                "dtype": np.dtype(dp0.space.dtype).str,
            }
        ),
    )
    hypersingular = FunctionLinearOperator(
        hypersingular_action.mv,
        source=p1_space,
        target=p1_space,
        transpose_action=hypersingular_action.transpose_mv,
        properties=OperatorProperties(evidence={}),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-hypersingular-p1-3d",
                "kernel": family.kernel_id,
                "faces": array_tree_fingerprint(faces),
            }
        ),
        closure_convert=False,
    )
    double_layer = DenseLinearOperator(
        jnp.asarray(double_matrix),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-double-layer-p1-dp0-3d",
                "kernel": family.kernel_id,
                "matrix": array_tree_fingerprint(double_matrix),
            }
        ),
    )
    adjoint_double = DenseLinearOperator(
        jnp.asarray(double_matrix.T.conj()),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-adjoint-double-layer-dp0-p1-3d",
                "kernel": family.kernel_id,
                "matrix": array_tree_fingerprint(double_matrix.T.conj()),
            }
        ),
    )
    cross_mass = DenseLinearOperator(
        jnp.asarray(cross_mass_matrix),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-p1-dp0-cross-mass-3d",
                "matrix": array_tree_fingerprint(cross_mass_matrix),
            }
        ),
    )
    spaces = ScalarBoundarySpaces3D(
        vertices=jnp.asarray(vertices),
        faces=jnp.asarray(faces),
        face_areas=jnp.asarray(face_areas),
        cross_mass=jnp.asarray(cross_mass_matrix),
        dirichlet_space=p1_space,
        neumann_space=dp0.space,
        vertex_count=vertices.shape[0],
        face_count=faces.shape[0],
        spaces_id=canonical_fingerprint(
            {
                "kind": "scalar-boundary-spaces-3d",
                "binding": dp0._binding.binding_id,
                "faces": array_tree_fingerprint(faces),
            }
        ),
    )
    resident_bytes = int(
        double_matrix.nbytes
        + normal_single.nbytes
        + cross_mass_matrix.nbytes
        + curls.nbytes
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "scalar-conforming-assembly-evidence-3d",
            "kernel": family.kernel_id,
            "spaces": spaces.spaces_id,
            "maximum_pair_order_defect": maximum_defect,
            "resident_bytes": resident_bytes,
            "maue_regularized": True,
            "dp0_hypersingular_supported": False,
        }
    )
    evidence = ScalarConformingAssemblyEvidence3D(
        maximum_pair_order_defect=maximum_defect,
        resident_bytes=resident_bytes,
        maue_regularized=True,
        dp0_hypersingular_supported=False,
        evidence_id=evidence_id,
    )
    return ScalarCalderonGalerkin3D(
        spaces=spaces,
        single_layer_weak=dp0.single_layer_weak,
        double_layer_weak=double_layer,
        adjoint_double_layer_weak=adjoint_double,
        hypersingular_weak=hypersingular,
        cross_mass=cross_mass,
        kernel=family,
        dp0_calderon=dp0,
        evidence=evidence,
        prepared_id=canonical_fingerprint(
            {
                "kind": "scalar-calderon-galerkin-3d",
                "spaces": spaces.spaces_id,
                "kernel": family.kernel_id,
                "evidence": evidence_id,
            }
        ),
    )


def prepare_scalar_hypersingular_p1_3d(
    region: MeshRegion,
    /,
    *,
    kernel: ScalarKernelFamily3D | None = None,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
    numeric_version: str = "0",
) -> AbstractLinearOperator:
    """Prepare the conforming P1 hypersingular weak operator."""

    return prepare_scalar_calderon_3d(
        region,
        kernel=kernel,
        policy=policy,
        numeric_version=numeric_version,
    ).hypersingular_weak


def prepare_closed_scalar_calderon_3d(
    region: MeshRegion,
    /,
    *,
    kernel: ScalarKernelFamily3D | None = None,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
    numeric_version: str = "0",
) -> ClosedScalarCalderon3D:
    """Prepare the closed mixed Calderón tuple with an explicit P1 gauge."""

    calderon = prepare_scalar_calderon_3d(
        region,
        kernel=kernel,
        policy=policy,
        numeric_version=numeric_version,
    )
    constant = jnp.ones(
        (calderon.spaces.vertex_count,), dtype=calderon.spaces.face_areas.dtype
    )
    gauge_mass = calderon.cross_mass.mv(constant)
    return ClosedScalarCalderon3D(
        calderon=calderon,
        p1_constant=constant,
        gauge_mass=gauge_mass,
        product_id=canonical_fingerprint(
            {
                "kind": "closed-scalar-calderon-3d",
                "calderon": calderon.prepared_id,
                "gauge_mass": array_tree_fingerprint(np.asarray(gauge_mass)),
            }
        ),
    )


__all__ = [
    "ClosedScalarCalderon3D",
    "ScalarBoundarySpaces3D",
    "ScalarCalderonGalerkin3D",
    "ScalarConformingAssemblyEvidence3D",
    "prepare_closed_scalar_calderon_3d",
    "prepare_scalar_calderon_3d",
    "prepare_scalar_hypersingular_p1_3d",
]

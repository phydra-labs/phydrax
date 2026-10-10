#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import exact_orient3d
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._form_type import FormType, FormValueSpec
from ...linalg import FunctionLinearOperator, OperatorProperties
from ...linalg._small_batched import SmallLinearSolvePlan, solve_small_linear
from ...sparse import scatter_local
from .._cell_complex import PolyhedralConnectivity
from .._cell_geometry import CellGeometrySpec
from .._cell_mesh import CellMesh
from .._polygon_geometry import (
    PolyhedralFaceTriangulation,
    prepare_polyhedral_face_triangulation,
)
from ._operator import FactorizedVirtualElementOperator
from ._precision import VirtualElementResourceBudget


if TYPE_CHECKING:
    from .._spaces import DiscreteFieldSpace
    from .._transfer import FieldTransfer, TransferGeometryBinding
    from ..fem._topology_transfer import FiniteElementTopologyTransfer


class PolyhedralVEMEvidence3D(StrictModule, NonTrainableState):
    """Admissibility and polynomial-consistency evidence for degree-one H1 VEM."""

    cell_volumes: Array
    cell_volume_error_bounds: Array
    projector_rank_margins: Array
    polynomial_reproduction_defects: Array
    minimum_volume: float = eqx.field(static=True)
    minimum_rank_margin: float = eqx.field(static=True)
    maximum_reproduction_defect: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    face_triangulation_id: str = eqx.field(static=True)


class PreparedPolyhedralH1VirtualElement3D(StrictModule, NonTrainableState):
    """Degree-one scalar 0-form H1 VEM on root polyhedral topology.

    This qualified 3-D scalar route is not a vector VEM or de Rham complex.
    """

    mesh: CellMesh
    cell_geometry: CellGeometrySpec | None
    operator: FactorizedVirtualElementOperator
    evidence: PolyhedralVEMEvidence3D
    value_spec: FormValueSpec = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)
    cell_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def mv(self, values: ArrayLike, /) -> Array:
        return self.operator.mv(values)

    def transpose_mv(self, values: ArrayLike, /) -> Array:
        return self.operator.transpose_mv(values)

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        return self.operator.as_linear_operator()

    def vertex_field_space(self, name: str, /) -> DiscreteFieldSpace:
        """Canonical scalar H1 vertex field on the actual quotient entities."""
        from ...linalg import ArraySpace
        from .._spaces import DiscreteFieldSpace, EntityDofLayout

        topology = (
            self.mesh.topology
            if self.mesh.periodic_topology is None
            else self.mesh.periodic_topology.quotient
        )
        return DiscreteFieldSpace(
            name,
            self.mesh.support.support_id,
            EntityDofLayout(
                topology.entities(0).entity_set_id,
                self.dof_count,
                self.dof_count,
                dofs_per_entity=1,
            ),
            ArraySpace((self.dof_count,), dtype=self.operator.coefficient_maps[0].dtype),
            representation="functional",
            conformity="H1",
            form_type=self.value_spec.form_type,
            projection_id=self.prepared_id,
        )

    def vertex_field_transfer(
        self,
        target: PreparedPolyhedralH1VirtualElement3D,
        physical_transfer: FiniteElementTopologyTransfer,
        geometry: TransferGeometryBinding,
        /,
        *,
        field_name: str = "u",
    ) -> FieldTransfer:
        """Compose a checked physical vertex stencil with scalar quotient maps.

        A stencil must intertwine the actual SCI orbits. Representative
        restriction is not coordinate matching or averaging. Only interpolation
        is claimed; arbitrary VEM interior/face interpolation is not admitted.
        """
        from ...linalg import ComposedLinearOperator, transpose
        from ...sparse import RowRelation, SparseLinearMap
        from .._cell_geometry_validity import cell_geometry_id
        from .._transfer import FieldTransfer, TransferGeometryBinding, TransferProperties
        from ..fem._topology_transfer import FiniteElementTopologyTransfer

        if not isinstance(target, PreparedPolyhedralH1VirtualElement3D) or not isinstance(
            physical_transfer, FiniteElementTopologyTransfer
        ):
            raise TypeError(
                "Scalar VEM transfer requires actual prepared endpoints and a canonical physical topology transfer."
            )
        if not isinstance(geometry, TransferGeometryBinding):
            raise TypeError("Scalar VEM transfer requires its actual geometry binding.")
        if (
            physical_transfer.source_topology_id != self.mesh.topology_id
            or physical_transfer.target_topology_id != target.mesh.topology_id
            or geometry.source_topology_id != self.mesh.topology_id
            or geometry.target_topology_id != target.mesh.topology_id
        ):
            raise ValueError(
                "The physical transfer and geometry binding must name the actual VEM topologies."
            )
        source_geometry = (
            self.mesh.geometry_id
            if self.cell_geometry is None
            else cell_geometry_id(self.cell_geometry)
        )
        target_geometry = (
            target.mesh.geometry_id
            if target.cell_geometry is None
            else cell_geometry_id(target.cell_geometry)
        )
        if (
            geometry.source_geometry_id != source_geometry
            or geometry.target_geometry_id != target_geometry
        ):
            raise ValueError(
                "The VEM transfer differs from its actual scientific geometry maps."
            )
        primal = physical_transfer.primal
        if not isinstance(primal, SparseLinearMap) or not isinstance(
            primal.relation, RowRelation
        ):
            raise TypeError(
                "Scalar VEM vertex traces require a canonical sparse physical vertex stencil."
            )
        source_periodic, target_periodic = (
            self.mesh.periodic_topology,
            target.mesh.periodic_topology,
        )
        source_orbits = (
            np.arange(self.mesh.coordinates.shape[0], dtype=np.int32)
            if source_periodic is None
            else np.asarray(source_periodic.orbits(0)[0])
        )
        target_orbits = (
            np.arange(target.mesh.coordinates.shape[0], dtype=np.int32)
            if target_periodic is None
            else np.asarray(target_periodic.orbits(0)[0])
        )
        representatives = (
            np.arange(target.dof_count, dtype=np.int32)
            if target_periodic is None
            else np.asarray(target_periodic.orbit_representatives(0))
        )
        if (
            primal.source.size != source_orbits.size
            or primal.target.size != target_orbits.size
        ):
            raise ValueError(
                "Physical transfer rows must cover the complete lifted vertex carriers."
            )
        routes, valid, weights = (
            np.asarray(primal.relation.source_indices),
            np.asarray(primal.relation.valid),
            np.asarray(primal.coefficients),
        )
        signatures = []
        for row, mask, coefficients in zip(routes, valid, weights, strict=True):
            signature: dict[int, Fraction] = {}
            for vertex, coefficient in zip(row[mask], coefficients[mask], strict=True):
                orbit = int(source_orbits[vertex])
                signature[orbit] = signature.get(
                    orbit, Fraction(0)
                ) + Fraction.from_float(float(coefficient))
            signatures.append(
                tuple(sorted((key, value) for key, value in signature.items() if value))
            )
        if any(
            signatures[index] != signatures[int(representatives[orbit])]
            for index, orbit in enumerate(target_orbits)
        ):
            raise ValueError(
                "The physical vertex stencil does not intertwine the scientific scalar quotient orbits."
            )
        lift = SparseLinearMap(
            RowRelation(source_orbits[:, None], source_size=self.dof_count),
            np.ones((source_orbits.size, 1)),
        )
        restriction = SparseLinearMap(
            RowRelation(representatives[:, None], source_size=target_orbits.size),
            np.ones((target.dof_count, 1)),
        )
        quotient = ComposedLinearOperator(
            restriction, ComposedLinearOperator(primal, lift)
        )
        reverse = transpose(quotient)
        return FieldTransfer(
            self.vertex_field_space(field_name),
            target.vertex_field_space(field_name),
            quotient,
            dual_pullback_operator=reverse,
            hilbert_adjoint_operator=reverse,
            properties=TransferProperties(
                constant_preserving=physical_transfer.preserves_constants,
                positivity_preserving=physical_transfer.positivity_preserving,
                adjoint_paired=True,
                semantics="interpolation",
            ),
            geometry=geometry,
        )

    def assemble_cellwise_constant_load(self, cell_forcing: ArrayLike, /) -> Array:
        """Volume vertex-lump a physical-cell scalar force into quotient DOFs.

        Each original cell contributes once, using its admitted source volume.
        This explicitly supported lumped load is not an exact higher-order VEM
        moment rule. No incompatible-load mean is silently removed.
        """
        forcing = jnp.asarray(cell_forcing)
        count = self.mesh.connectivity.cell_count
        if forcing.shape == ():
            forcing = jnp.broadcast_to(forcing, (count,))
        if forcing.shape != (count,) or jnp.iscomplexobj(forcing):
            raise ValueError(
                "Scalar load assembly requires one real force per physical cell."
            )
        result = jnp.zeros(
            (self.dof_count,), dtype=jnp.result_type(forcing, self.evidence.cell_volumes)
        )
        for gather, indices in zip(self.operator.gathers, self.cell_indices, strict=True):
            arity = gather.shape[1]
            cells = jnp.asarray(indices, dtype=jnp.int32)
            loads = forcing[cells] * self.evidence.cell_volumes[cells] / arity
            result = scatter_local(
                result,
                gather,
                jnp.broadcast_to(loads[:, None], gather.shape),
                self.operator.accumulation,
            )
        return result

    def bind_scalar_diffusion(
        self, cell_coefficients: ArrayLike, /
    ) -> FactorizedVirtualElementOperator:
        """Rebind positive scalar cell material without rebuilding source projectors."""
        coefficients = np.asarray(cell_coefficients)
        count = self.mesh.connectivity.cell_count
        if coefficients.shape == ():
            coefficients = np.broadcast_to(coefficients, (count,))
        if coefficients.shape != (count,) or coefficients.dtype.kind not in "fiu":
            raise ValueError(
                "Scalar diffusion requires one real coefficient per physical cell."
            )
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients <= 0):
            raise ValueError(
                "Scalar diffusion coefficients must be finite and strictly positive."
            )
        polynomials = []
        stabilizations = []
        for polynomial, stabilization, indices in zip(
            self.operator.polynomial_matrices,
            self.operator.stabilization_matrices,
            self.cell_indices,
            strict=True,
        ):
            weights = jnp.asarray(coefficients[np.asarray(indices, dtype=np.int32)])[
                :, None, None
            ]
            polynomials.append(polynomial * weights)
            stabilizations.append(stabilization * weights)
        return FactorizedVirtualElementOperator(
            self.operator.coefficient_maps,
            tuple(polynomials),
            tuple(stabilizations),
            self.operator.gathers,
            self.dof_count,
            accumulation=self.operator.accumulation,
            properties=self.operator.properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "bound-scalar-polyhedral-diffusion",
                    "prepared": self.prepared_id,
                    "coefficients": array_tree_fingerprint(coefficients),
                }
            ),
        )

    def compatible_pinned_load(
        self, load: ArrayLike, pinned_dof: int, /, *, compatibility_tolerance: float
    ) -> Array:
        """Admit the physical Neumann/periodic load before fixing its constant gauge."""
        values = np.asarray(load)
        tolerance = float(compatibility_tolerance)
        pin = int(pinned_dof)
        if values.shape != (self.dof_count,) or not np.all(np.isfinite(values)):
            raise ValueError("Diffusion load must be a finite quotient scalar vector.")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "Load compatibility tolerance must be finite and nonnegative."
            )
        if not 0 <= pin < self.dof_count:
            raise ValueError("The diffusion gauge must name an actual quotient DOF.")
        if abs(math.fsum(map(float, values))) > tolerance:
            raise ValueError(
                "Periodic/Neumann diffusion requires a compatible zero-total physical load."
            )
        return jnp.asarray(load)[jnp.asarray(np.delete(np.arange(self.dof_count), pin))]

    def pinned_diffusion_operator(
        self,
        pinned_dof: int,
        /,
        *,
        cell_coefficients: ArrayLike | None = None,
    ) -> FunctionLinearOperator:
        """Explicit zero-value gauge for a connected scalar diffusion component.

        The physical load must satisfy ``sum(load) == 0`` before restriction.
        This fixes the additive constant; it does not add a mass or use a
        pseudoinverse. Disconnected components require independent gauges.
        """
        pin = int(pinned_dof)
        if not 0 <= pin < self.dof_count:
            raise ValueError("The diffusion gauge must name an actual quotient DOF.")
        parents = np.arange(self.dof_count)

        def root(index: int) -> int:
            while parents[index] != index:
                index = int(parents[index])
            return index

        for bucket in self.operator.gathers:
            for row in np.asarray(bucket):
                first = root(int(row[0]))
                for index in row[1:]:
                    parents[root(int(index))] = first
        if len({root(index) for index in range(self.dof_count)}) != 1:
            raise ValueError("A single diffusion gauge requires a connected quotient.")
        if self.dof_count == 1:
            raise ValueError("The gauged quotient has no nonconstant scalar modes.")
        free = jnp.asarray(np.delete(np.arange(self.dof_count, dtype=np.int32), pin))
        from ...linalg import ArraySpace

        reduced = ArraySpace(
            (self.dof_count - 1,), dtype=self.operator.coefficient_maps[0].dtype
        )
        diffusion = (
            self.operator
            if cell_coefficients is None
            else self.bind_scalar_diffusion(cell_coefficients)
        )

        def action(values: Array) -> Array:
            full = jnp.zeros((self.dof_count,), dtype=values.dtype).at[free].set(values)
            return diffusion.mv(full)[free]

        return FunctionLinearOperator(
            action,
            source=reduced,
            target=reduced,
            transpose_action=action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "pinned-polyhedral-diffusion",
                    "operator": diffusion.operator_id,
                    "pin": pin,
                }
            ),
        )


def _cell_matrix(
    coordinates: np.ndarray,
    connectivity: PolyhedralConnectivity,
    cell_index: int,
    prepared_faces: PolyhedralFaceTriangulation,
    /,
) -> tuple[np.ndarray, float, float, float]:
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int32)
    cell_face_values = np.asarray(connectivity.cell_face_values, dtype=np.int32)
    cell_face_signs = np.asarray(connectivity.cell_face_sign_values)
    triangle_offsets = prepared_faces.triangle_offsets
    triangle_vertices = prepared_faces.triangle_vertices
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    cell_vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
    cell_start, cell_stop = cell_vertex_offsets[cell_index : cell_index + 2]
    vertices = cell_vertex_values[int(cell_start) : int(cell_stop)]
    local = {int(vertex): index for index, vertex in enumerate(vertices)}
    points = np.asarray(coordinates[vertices], dtype=np.float64)
    centroid = (
        np.asarray(
            tuple(
                sum((coordinates[vertex, axis] for vertex in vertices), Fraction(0))
                / len(vertices)
                for axis in range(3)
            ),
            dtype=object,
        )
        if coordinates.dtype == object
        else np.mean(coordinates[vertices], axis=0)
    )
    numerical_centroid = np.asarray(centroid, dtype=np.float64)
    characteristic_length = float(
        np.max(np.linalg.norm(points - numerical_centroid[None, :], axis=1))
    )
    if not np.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("Polyhedral VEM cells require positive diameter.")

    exact_source = coordinates.dtype == object
    gradients = np.zeros((3, vertices.size), dtype=object if exact_source else np.float64)
    volume = Fraction(0) if exact_source else 0.0
    face_start, face_stop = cell_face_offsets[cell_index : cell_index + 2]
    for face_index, sign in zip(
        cell_face_values[int(face_start) : int(face_stop)],
        cell_face_signs[int(face_start) : int(face_stop)],
        strict=True,
    ):
        start, stop = triangle_offsets[int(face_index) : int(face_index) + 2]
        face_points = coordinates[triangle_vertices[start:stop]]
        if not exact_source:
            signs = exact_orient3d(
                np.broadcast_to(centroid, face_points[:, 0].shape),
                face_points[:, 0],
                face_points[:, 1],
                face_points[:, 2],
            )
            if np.any(signs * int(sign) <= 0):
                raise ValueError(
                    "Polyhedral VEM has unresolved/nonpositive exact star visibility."
                )
        for triangle in triangle_vertices[start:stop]:
            triangle_vertices_ = tuple(map(int, triangle))
            first, second, third = coordinates[triangle]
            factor = Fraction(int(sign), 2) if exact_source else float(sign) * 0.5
            area_vector = factor * np.cross(second - first, third - first)
            signed_volume = np.dot(first - centroid, area_vector) / 3
            if not math.isfinite(float(signed_volume)) or signed_volume <= 0:
                raise ValueError(
                    "Polyhedral VEM requires outward faces visible from the cell centroid."
                )
            volume += signed_volume
            for vertex in triangle_vertices_:
                gradients[:, local[vertex]] += area_vector / 3
    if not math.isfinite(float(volume)) or volume <= 0.0:
        raise ValueError("Polyhedral VEM requires positive finite cell volume.")
    gradients = np.asarray(gradients / volume, dtype=np.float64)
    volume = float(volume)
    consistency = volume * gradients.T @ gradients

    monomials = np.concatenate(
        (
            np.ones((vertices.size, 1), dtype=np.float64),
            (points - numerical_centroid[None, :]) / characteristic_length,
        ),
        axis=1,
    )
    singular_values = np.linalg.svd(monomials, compute_uv=False)
    rank_margin = float(singular_values[-1])
    if rank_margin <= np.finfo(np.float64).eps * singular_values[0] * vertices.size:
        raise ValueError("Polyhedral VEM affine projector is rank deficient.")
    solve = solve_small_linear(
        SmallLinearSolvePlan(4), monomials.T @ monomials, monomials.T
    )
    if not bool(solve.successful):
        raise ValueError(
            "Polyhedral VEM affine projector fails native rank/conditioning/residual admission."
        )
    projector = monomials @ np.asarray(solve.value)
    kernel = np.eye(vertices.size) - projector
    scale = float(np.trace(consistency) / max(vertices.size, 1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "Polyhedral VEM consistency energy has no admissible stabilization scale."
        )
    stabilization = scale * (kernel.T @ kernel)
    expected_gradient = np.column_stack((np.zeros(3), np.eye(3)))
    reproduction = max(
        float(np.linalg.norm(kernel @ monomials, ord=np.inf)),
        float(
            np.linalg.norm(
                characteristic_length * gradients @ monomials - expected_gradient,
                ord=np.inf,
            )
        ),
    )
    return consistency + stabilization, volume, rank_margin, reproduction


def prepare_polyhedral_h1_virtual_element_3d(
    mesh: CellMesh,
    /,
    *,
    degree: int = 1,
    resource_budget: VirtualElementResourceBudget | None = None,
    accumulation: str = "fast",
    cell_geometry: CellGeometrySpec | None = None,
) -> PreparedPolyhedralH1VirtualElement3D:
    """Prepare the first root-topology 3-D H1 VEM consumer.

    The supported tuple is deliberately exact: affine geometry, star-visible
    oriented polyhedra, and degree one. Higher-degree 2-D VEM remains available
    through ``VirtualElementPlan``; unsupported 3-D degree requests fail rather
    than substituting a low-order cell.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Polyhedral H1 VEM requires a 3-D CellMesh in R3.")
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral H1 VEM requires root PolyhedralConnectivity.")
    if int(degree) != 1:
        raise ValueError("The qualified polyhedral H1 VEM tuple has degree one.")
    budget = (
        VirtualElementResourceBudget() if resource_budget is None else resource_budget
    )
    if not isinstance(budget, VirtualElementResourceBudget):
        raise TypeError("resource_budget must be VirtualElementResourceBudget.")
    value_spec = FormValueSpec(FormType(3, 0), proxy="scalar")
    connectivity = mesh.connectivity
    periodic = mesh.periodic_topology
    vertex_route = (
        np.arange(connectivity.vertex_count, dtype=np.int32)
        if periodic is None
        else np.asarray(periodic.orbits(0)[0], dtype=np.int32)
    )
    dof_count = (
        connectivity.vertex_count
        if periodic is None
        else len(periodic.orbit_representatives(0))
    )
    if periodic is not None and cell_geometry is None:
        cell_geometry = periodic.actual_geometry
        if cell_geometry is None:
            raise ValueError(
                "Periodic polyhedral VEM requires its authoritative scientific cell geometry."
            )
    if connectivity.cell_count > budget.maximum_cells:
        raise ValueError("Polyhedral VEM cell capacity exceeded.")

    if cell_geometry is not None:
        from ...geometry._exact_polyhedral_geometry import exact_vertices

        coordinates = exact_vertices(mesh, cell_geometry)
    else:
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    prepared_faces = prepare_polyhedral_face_triangulation(
        mesh,
        cell_geometry=cell_geometry,
        maximum_entries=budget.maximum_projector_bytes // np.dtype(np.int32).itemsize,
    )
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    cell_vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
    cell_arities = np.diff(cell_vertex_offsets)
    buckets: dict[int, list[int]] = {}
    matrices: dict[int, list[np.ndarray]] = {}
    volumes = np.empty((connectivity.cell_count,), dtype=np.float64)
    rank_margins = np.empty_like(volumes)
    defects = np.empty_like(volumes)
    estimated_bytes = 0
    for cell_index in range(connectivity.cell_count):
        arity = int(cell_arities[cell_index])
        if arity > budget.maximum_local_dofs:
            raise ValueError("Polyhedral VEM local-DOF capacity exceeded.")
        matrix, volume, margin, defect = _cell_matrix(
            coordinates, connectivity, cell_index, prepared_faces
        )
        buckets.setdefault(arity, []).append(cell_index)
        matrices.setdefault(arity, []).append(matrix)
        volumes[cell_index] = volume
        rank_margins[cell_index] = margin
        defects[cell_index] = defect
        estimated_bytes += matrix.nbytes * 3
    if estimated_bytes > budget.maximum_projector_bytes:
        raise ValueError("Polyhedral VEM projector byte capacity exceeded.")
    volume_errors = np.zeros_like(volumes)
    if cell_geometry is not None:
        from .._cell_geometry_transfer import _certified_cell_measures

        volumes, volume_errors, _ = _certified_cell_measures(mesh, cell_geometry)

    coefficient_maps = []
    polynomial_matrices = []
    stabilization_matrices = []
    gathers = []
    for arity in sorted(buckets):
        indices = np.asarray(buckets[arity], dtype=np.int32)
        local = np.stack(matrices[arity])
        coefficient_maps.append(
            np.broadcast_to(np.eye(arity), (indices.size, arity, arity)).copy()
        )
        polynomial_matrices.append(local)
        stabilization_matrices.append(np.zeros_like(local))
        gathers.append(
            np.stack(
                [
                    vertex_route[
                        cell_vertex_values[
                            cell_vertex_offsets[cell] : cell_vertex_offsets[cell + 1]
                        ]
                    ]
                    for cell in indices
                ]
            )
        )
    properties = OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = FactorizedVirtualElementOperator(
        tuple(coefficient_maps),
        tuple(polynomial_matrices),
        tuple(stabilization_matrices),
        tuple(gathers),
        dof_count,
        accumulation=accumulation,
        properties=properties,
        operator_id=canonical_fingerprint(
            {
                "kind": "polyhedral-h1-vem-operator",
                "mesh": mesh.topology_id,
                "degree": 1,
                "periodic": None if periodic is None else periodic.periodic_topology_id,
                "gathers": array_tree_fingerprint(tuple(gathers)),
                "matrices": array_tree_fingerprint(tuple(polynomial_matrices)),
            }
        ),
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "polyhedral-h1-vem-evidence",
            "mesh": mesh.geometry_id,
            "volumes": array_tree_fingerprint(volumes),
            "volume_errors": array_tree_fingerprint(volume_errors),
            "rank_margins": array_tree_fingerprint(rank_margins),
            "face_triangulation": prepared_faces.triangulation_id,
            "defects": array_tree_fingerprint(defects),
        }
    )
    evidence = PolyhedralVEMEvidence3D(
        cell_volumes=jnp.asarray(volumes),
        cell_volume_error_bounds=jnp.asarray(volume_errors),
        projector_rank_margins=jnp.asarray(rank_margins),
        polynomial_reproduction_defects=jnp.asarray(defects),
        minimum_volume=float(np.min(volumes)),
        minimum_rank_margin=float(np.min(rank_margins)),
        maximum_reproduction_defect=float(np.max(defects)),
        evidence_id=evidence_id,
        face_triangulation_id=prepared_faces.triangulation_id,
    )
    return PreparedPolyhedralH1VirtualElement3D(
        mesh=mesh,
        cell_geometry=cell_geometry,
        operator=operator,
        evidence=evidence,
        value_spec=value_spec,
        degree=1,
        dof_count=dof_count,
        cell_indices=tuple(tuple(buckets[arity]) for arity in sorted(buckets)),
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-polyhedral-h1-vem",
                "mesh": mesh.mesh_id,
                "operator": operator.operator_id,
                "evidence": evidence_id,
                "value_spec": value_spec.value_spec_id,
                "cell_indices": tuple(tuple(buckets[arity]) for arity in sorted(buckets)),
            }
        ),
    )


__all__ = [
    "PolyhedralVEMEvidence3D",
    "PreparedPolyhedralH1VirtualElement3D",
    "prepare_polyhedral_h1_virtual_element_3d",
]

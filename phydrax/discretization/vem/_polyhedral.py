#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
from phydrax.discretization._polygon_types import PolyhedralFaceTriangulation

from ... import linalg as linalg_owner, sparse as sparse_owner
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import exact_orient3d
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._form_type import FormType, FormValueSpec
from ...geometry import _exact_polyhedral_geometry as exact_polyhedral_geometry_owner
from ...linalg import (
    FunctionLinearOperator,
    OperatorProperties,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...sparse import RelationAccumulation, scatter_local
from ...typing import checked
from .. import (
    _cell_geometry_validity as cell_geometry_validity_owner,
    _spaces as spaces_owner,
    _transfer as transfer_owner,
)
from .._cell_complex import PolyhedralConnectivity
from .._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    CellGeometrySpec,
    owned_coordinate_preparation,
)
from .._cell_mesh import CellMesh
from .._coordinate_budget import CoordinateEnclosureBudget
from .._polygon_geometry import prepare_polyhedral_face_triangulation
from ..fem import _topology_transfer as topology_transfer_owner
from ._operator import FactorizedVirtualElementOperator
from ._precision import VirtualElementResourceBudget


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

    def vertex_field_space(self, name: str, /) -> spaces_owner.DiscreteFieldSpace:
        """Return a scalar H1 vertex field on quotient entities."""

        topology = (
            self.mesh.topology
            if self.mesh.periodic_topology is None
            else self.mesh.periodic_topology.quotient
        )
        return spaces_owner.DiscreteFieldSpace(
            name,
            self.mesh.support.support_id,
            spaces_owner.EntityDofLayout(
                topology.entities(0).entity_set_id,
                self.dof_count,
                self.dof_count,
                dofs_per_entity=1,
            ),
            linalg_owner.ArraySpace(
                (self.dof_count,), dtype=self.operator.coefficient_maps[0].dtype
            ),
            representation="functional",
            conformity="H1",
            form_type=self.value_spec.form_type,
            projection_id=self.prepared_id,
        )

    @checked
    def vertex_field_transfer(
        self,
        target: PreparedPolyhedralH1VirtualElement3D,
        physical_transfer: topology_transfer_owner.FiniteElementTopologyTransfer,
        geometry: transfer_owner.TransferGeometryBinding,
        /,
        *,
        field_name: str = "u",
    ) -> transfer_owner.FieldTransfer:
        """Compose a checked physical vertex stencil with scalar quotient maps.

        A stencil must intertwine the SCI orbits. Representative
        restriction is not coordinate matching or averaging. Only interpolation
        is claimed; arbitrary VEM interior/face interpolation is not admitted.
        """

        if (
            physical_transfer.source_topology_id != self.mesh.topology_id
            or physical_transfer.target_topology_id != target.mesh.topology_id
            or geometry.source_topology_id != self.mesh.topology_id
            or geometry.target_topology_id != target.mesh.topology_id
        ):
            raise ValueError(
                "PreparedPolyhedralH1VirtualElement3D.geometry must name the VEM topologies; got mismatched transfer endpoints"
            )
        source_geometry = (
            self.mesh.geometry_id
            if self.cell_geometry is None
            else cell_geometry_validity_owner.cell_geometry_id(self.cell_geometry)
        )
        target_geometry = (
            target.mesh.geometry_id
            if target.cell_geometry is None
            else cell_geometry_validity_owner.cell_geometry_id(target.cell_geometry)
        )
        if (
            geometry.source_geometry_id != source_geometry
            or geometry.target_geometry_id != target_geometry
        ):
            raise ValueError(
                "PreparedPolyhedralH1VirtualElement3D.geometry must match the scientific geometry maps; got mismatched geometry IDs"
            )
        primal = physical_transfer.primal
        if not isinstance(primal, sparse_owner.SparseLinearMap) or not isinstance(
            primal.relation, sparse_owner.RowRelation
        ):
            raise TypeError(
                "PreparedPolyhedralH1VirtualElement3D.physical_transfer must carry a sparse row vertex stencil; got incompatible primal route"
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
                f"PreparedPolyhedralH1VirtualElement3D.physical_transfer must cover lifted vertex carriers; got source={primal.source.size}, target={primal.target.size}"
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
                "PreparedPolyhedralH1VirtualElement3D.physical_transfer must intertwine scalar quotient orbits; got different orbit row signatures"
            )
        lift = sparse_owner.SparseLinearMap(
            sparse_owner.RowRelation(source_orbits[:, None], source_size=self.dof_count),
            np.ones((source_orbits.size, 1)),
        )
        restriction = sparse_owner.SparseLinearMap(
            sparse_owner.RowRelation(
                representatives[:, None], source_size=target_orbits.size
            ),
            np.ones((target.dof_count, 1)),
        )
        quotient = linalg_owner.ComposedLinearOperator(
            restriction, linalg_owner.ComposedLinearOperator(primal, lift)
        )
        reverse = linalg_owner.transpose(quotient)
        return transfer_owner.FieldTransfer(
            self.vertex_field_space(field_name),
            target.vertex_field_space(field_name),
            quotient,
            dual_pullback_operator=reverse,
            hilbert_adjoint_operator=reverse,
            properties=transfer_owner.TransferProperties(
                constant_preserving=physical_transfer.preserves_constants,
                positivity_preserving=physical_transfer.positivity_preserving,
                adjoint_paired=True,
                semantics="interpolation",
            ),
            geometry=geometry,
        )

    def assemble_cellwise_constant_load(self, cell_forcing: ArrayLike, /) -> Array:
        """Volume vertex-lump a physical-cell scalar force into quotient DOFs.

        Each physical cell contributes once, using its admitted source volume.
        The rule distributes that volume equally among the cell's vertex DOFs.
        """
        forcing = jnp.asarray(cell_forcing)
        count = self.mesh.connectivity.cell_count
        if forcing.shape == ():
            forcing = jnp.broadcast_to(forcing, (count,))
        if forcing.shape != (count,) or jnp.iscomplexobj(forcing):
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.cell_forcing must have one real force per physical cell; got shape={forcing.shape}, dtype={forcing.dtype}"
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
                f"PreparedPolyhedralH1VirtualElement3D.cell_coefficients must have one real coefficient per physical cell; got shape={coefficients.shape}, dtype={coefficients.dtype}"
            )
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients <= 0):
            raise ValueError(
                "PreparedPolyhedralH1VirtualElement3D.cell_coefficients must be finite and strictly positive; got nonfinite or nonpositive coefficients"
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
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.load must be a finite quotient scalar vector; got shape={values.shape}"
            )
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.compatibility_tolerance must be finite and nonnegative; got {tolerance}"
            )
        if not 0 <= pin < self.dof_count:
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.pinned_dof must name a quotient DOF; got {pin}"
            )
        if abs(math.fsum(map(float, values))) > tolerance:
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.load must have compatible zero-total physical load; got total={math.fsum(map(float, values))}, tolerance={tolerance}"
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
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.pinned_dof must name a quotient DOF; got {pin}"
            )
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
            raise ValueError(
                "PreparedPolyhedralH1VirtualElement3D.pinned_dof must gauge a connected quotient; got disconnected components"
            )
        if self.dof_count == 1:
            raise ValueError(
                f"PreparedPolyhedralH1VirtualElement3D.dof_count must exceed one for a gauged quotient; got {self.dof_count}"
            )
        free = jnp.asarray(np.delete(np.arange(self.dof_count, dtype=np.int32), pin))

        reduced = linalg_owner.ArraySpace(
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


class _PolyhedralHostTopology(StrictModule):
    cell_face_offsets: np.ndarray
    cell_face_values: np.ndarray
    cell_face_signs: np.ndarray
    cell_vertex_offsets: np.ndarray
    cell_vertex_values: np.ndarray


def _polyhedral_host_topology(
    connectivity: PolyhedralConnectivity, /
) -> _PolyhedralHostTopology:
    """Snapshot packed topology once for host projector preparation."""
    face_offsets, face_values, face_signs, vertex_offsets, vertex_values = jax.device_get(
        (
            connectivity.cell_face_offsets,
            connectivity.cell_face_values,
            connectivity.cell_face_sign_values,
            connectivity.cell_vertex_offsets,
            connectivity.cell_vertex_values,
        )
    )
    return _PolyhedralHostTopology(
        np.asarray(face_offsets, dtype=np.int32),
        np.asarray(face_values, dtype=np.int32),
        np.asarray(face_signs),
        np.asarray(vertex_offsets, dtype=np.int32),
        np.asarray(vertex_values, dtype=np.int32),
    )


class _PolyhedralVEMSource(StrictModule):
    coordinates: np.ndarray
    faces: PolyhedralFaceTriangulation
    topology: _PolyhedralHostTopology


def _prepare_polyhedral_vem_source(
    mesh: CellMesh,
    connectivity: PolyhedralConnectivity,
    cell_geometry: CellGeometrySpec | None,
    resource_limits: VirtualElementResourceBudget,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> _PolyhedralVEMSource:
    """Prepare source vertices, face triangulation, and packed host topology."""
    if cell_geometry is not None:
        if budget is None:
            work_upper, bytes_upper = cell_geometry_preparation_bounds(cell_geometry)
            source_request = owned_coordinate_preparation(
                2 * work_upper,
                2 * bytes_upper,
                maximum_geometry_queries=2
                * cell_geometry_preparation_query_upper(cell_geometry),
            )
        else:
            source_request = nullcontext(budget)
        with source_request as source_budget:
            coordinates = exact_polyhedral_geometry_owner.exact_vertices(
                mesh, cell_geometry, budget=source_budget
            )
    else:
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    faces = prepare_polyhedral_face_triangulation(
        mesh,
        cell_geometry=cell_geometry,
        maximum_entries=resource_limits.maximum_projector_bytes
        // np.dtype(np.int32).itemsize,
        budget=budget,
    )
    return _PolyhedralVEMSource(
        coordinates, faces, _polyhedral_host_topology(connectivity)
    )


def _cell_matrix(
    coordinates: np.ndarray,
    topology: _PolyhedralHostTopology,
    cell_index: int,
    prepared_faces: PolyhedralFaceTriangulation,
    /,
) -> tuple[np.ndarray, float, float, float]:
    cell_face_offsets = topology.cell_face_offsets
    cell_face_values = topology.cell_face_values
    cell_face_signs = topology.cell_face_signs
    triangle_offsets = prepared_faces.triangle_offsets
    triangle_vertices = prepared_faces.triangle_vertices
    cell_vertex_offsets = topology.cell_vertex_offsets
    cell_vertex_values = topology.cell_vertex_values
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
        raise ValueError(
            f"PolyhedralVEM.cell_diameter must be finite and positive; got {characteristic_length}"
        )

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
                    "PolyhedralVEM.cell_faces must have positive star visibility under exact predicates; got unresolved or nonpositive orientation"
                )
        for triangle in triangle_vertices[start:stop]:
            triangle_vertices_ = tuple(map(int, triangle))
            first, second, third = coordinates[triangle]
            factor = Fraction(int(sign), 2) if exact_source else float(sign) * 0.5
            area_vector = factor * np.cross(second - first, third - first)
            signed_volume = np.dot(first - centroid, area_vector) / 3
            if not math.isfinite(float(signed_volume)) or signed_volume <= 0:
                raise ValueError(
                    f"PolyhedralVEM.cell_faces must point outward and be visible from the centroid; got signed_volume={signed_volume}"
                )
            volume += signed_volume
            for vertex in triangle_vertices_:
                gradients[:, local[vertex]] += area_vector / 3
    if not math.isfinite(float(volume)) or volume <= 0.0:
        raise ValueError(
            f"PolyhedralVEM.cell_volume must be finite and positive; got {volume}"
        )
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
        raise ValueError(
            f"PolyhedralVEM.projector must have full affine rank; got rank_margin={rank_margin}"
        )
    solve = solve_small_linear(
        SmallLinearSolvePlan(4), monomials.T @ monomials, monomials.T
    )
    if not bool(solve.successful):
        raise ValueError(
            "PolyhedralVEM.projector must pass native rank, conditioning, and residual admission; got failed small solve"
        )
    projector = monomials @ np.asarray(solve.value)
    kernel = np.eye(vertices.size) - projector
    scale = float(np.trace(consistency) / max(vertices.size, 1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"PolyhedralVEM.stabilization_scale must be finite and positive; got {scale}"
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


@checked
def prepare_polyhedral_h1_virtual_element_3d(
    mesh: CellMesh,
    /,
    *,
    degree: int = 1,
    resource_budget: VirtualElementResourceBudget | None = None,
    accumulation: RelationAccumulation = "fast",
    cell_geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedPolyhedralH1VirtualElement3D:
    """Prepare the first root-topology 3-D H1 VEM consumer.

    The supported tuple has affine geometry, star-visible oriented polyhedra,
    and degree one. Higher-degree 2D VEM uses ``VirtualElementPlan``.
    """

    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            f"prepare_polyhedral_h1_virtual_element_3d.mesh must be a 3D CellMesh in R3; got topological={mesh.topological_dimension}, ambient={mesh.ambient_dimension}"
        )
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise TypeError(
            "prepare_polyhedral_h1_virtual_element_3d.mesh.connectivity must be root PolyhedralConnectivity; got incompatible connectivity"
        )
    if int(degree) != 1:
        raise ValueError(
            f"prepare_polyhedral_h1_virtual_element_3d.degree must be one; got {degree}"
        )
    resource_limits = (
        VirtualElementResourceBudget() if resource_budget is None else resource_budget
    )
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
                "prepare_polyhedral_h1_virtual_element_3d.cell_geometry must be supplied for periodic cells; got None"
            )
    if connectivity.cell_count > resource_limits.maximum_cells:
        raise ValueError(
            f"prepare_polyhedral_h1_virtual_element_3d.resource_budget must admit the cell capacity; got cells={connectivity.cell_count}, maximum={resource_limits.maximum_cells}"
        )

    source = _prepare_polyhedral_vem_source(
        mesh, connectivity, cell_geometry, resource_limits, budget
    )
    cell_vertex_offsets = source.topology.cell_vertex_offsets
    cell_vertex_values = source.topology.cell_vertex_values
    cell_arities = np.diff(cell_vertex_offsets)
    buckets: dict[int, list[int]] = {}
    matrices: dict[int, list[np.ndarray]] = {}
    volumes = np.empty((connectivity.cell_count,), dtype=np.float64)
    rank_margins = np.empty_like(volumes)
    defects = np.empty_like(volumes)
    estimated_bytes = 0
    for cell_index in range(connectivity.cell_count):
        arity = int(cell_arities[cell_index])
        if arity > resource_limits.maximum_local_dofs:
            raise ValueError(
                f"prepare_polyhedral_h1_virtual_element_3d.resource_budget must admit the local-DOF capacity; got local_dofs={arity}, maximum={resource_limits.maximum_local_dofs}"
            )
        matrix, volume, margin, defect = _cell_matrix(
            source.coordinates, source.topology, cell_index, source.faces
        )
        buckets.setdefault(arity, []).append(cell_index)
        matrices.setdefault(arity, []).append(matrix)
        volumes[cell_index] = volume
        rank_margins[cell_index] = margin
        defects[cell_index] = defect
        estimated_bytes += matrix.nbytes * 3
    if estimated_bytes > resource_limits.maximum_projector_bytes:
        raise ValueError(
            f"prepare_polyhedral_h1_virtual_element_3d.resource_budget must admit projector byte capacity; got bytes={estimated_bytes}, maximum={resource_limits.maximum_projector_bytes}"
        )
    volume_errors = np.zeros_like(volumes)
    if cell_geometry is not None:
        if budget is None:
            # These are the existing standalone certified-measure allowances;
            # the separate VEM projector capacity bounds projector storage.
            measure_work = 100_000_000
            measure_request = owned_coordinate_preparation(
                measure_work,
                sys.maxsize,
                maximum_geometry_queries=(measure_work + 1)
                * cell_geometry_preparation_query_upper(cell_geometry),
            )
        else:
            measure_request = nullcontext(budget)
        with measure_request as measure_budget:
            volumes, volume_errors, _ = mapped_measure_owner.certified_cell_measures(
                mesh, cell_geometry, budget=measure_budget
            )

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
            "face_triangulation": source.faces.triangulation_id,
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
        face_triangulation_id=source.faces.triangulation_id,
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

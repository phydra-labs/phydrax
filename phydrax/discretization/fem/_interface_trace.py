#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import MeshRegion
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    FunctionLinearOperator,
)
from .._cell_complex import TetrahedralConnectivity
from .._topology import EntitySelection
from ._generic import FiniteElementDiscretization


class PreparedMatchingScalarInterfaceTrace3D(StrictModule, NonTrainableState):
    """Exact affine P1-tetrahedron routes to one matching oriented DP0 surface.

    The bounded envelope is a scalar three-dimensional interior Poisson problem
    on an affine tetrahedral H1 P1 mesh coupled through its complete exterior
    boundary to a homogeneous exterior Laplace problem.  The geometry is one
    vertex-matching, outward-oriented triangular ``MeshRegion``; the trace is
    the exact facet average, never point interpolation.  Conormal values use
    the normal pointing from the FEM interior into the exterior on both PDE
    sides.  The provider is the fixed-topology FEM discretization and the
    supplied DP0 boundary vector space.  Precision, coordinate/orientation
    error, flux-closure evidence, and fixed action resources are recorded
    below.  This is discrete preparation evidence, not continuum certification.

    Non-goals are nonmatching/mortar interfaces, curved or moving geometry,
    partial/open boundaries, higher-order or vector fields, two dimensions,
    acoustics, and continuum discretization-error estimation.
    """

    trace_operator: AbstractLinearOperator
    boundary_load_operator: AbstractLinearOperator
    conormal_operator: AbstractLinearOperator
    interface_coordinates: Array
    interface_faces: Array
    fem_vertex_indices: Array
    fem_face_indices: Array
    owner_cells: Array
    outward_normals: Array
    face_areas: Array
    coordinate_max_error: Array
    minimum_orientation_cosine: Array
    flux_closure_norm: Array
    spatial_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry_contract: str = eqx.field(static=True)
    formulation_role: str = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    provider_ids: tuple[str, str, str] = eqx.field(static=True)
    precision_evidence: tuple[str, str] = eqx.field(static=True)
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def trace(self, coefficients: Array, /) -> Array:
        """Return the exact DP0 facet-average Dirichlet trace."""

        return self.trace_operator.mv(coefficients)

    def trace_transpose(self, facet_values: Array, /) -> Array:
        """Apply the exact algebraic transpose of the facet-average trace."""

        return self.trace_operator.transpose_mv(facet_values)

    def boundary_load(self, conormal_values: Array, /) -> Array:
        """Assemble ``∫Γ q γv`` in the FEM dual coordinates."""

        return self.boundary_load_operator.mv(conormal_values)

    def boundary_load_transpose(self, coefficients: Array, /) -> Array:
        """Apply the exact transpose of the conormal-to-FEM load route."""

        return self.boundary_load_operator.transpose_mv(coefficients)

    def conormal(self, coefficients: Array, /) -> Array:
        """Return the exact outward normal derivative of affine P1 cells."""

        return self.conormal_operator.mv(coefficients)

    def conormal_transpose(self, facet_values: Array, /) -> Array:
        """Apply the exact algebraic transpose of the conormal route."""

        return self.conormal_operator.transpose_mv(facet_values)

    def integrated_flux(self, conormal_values: Array, /) -> Array:
        """Integrate one DP0 conormal density over the interface."""

        values = self.trace_operator.target.validate(conormal_values)
        return jnp.sum(self.face_areas * values)


def _field_index(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
) -> int:
    name = str(field_name)
    matches = tuple(
        index
        for index, field_space in enumerate(discretization.field_spaces)
        if field_space.name == name
    )
    if len(matches) != 1:
        raise ValueError("field_name must select exactly one prepared FEM field.")
    return matches[0]


def _coordinate_tolerance(
    coordinate_tolerance: float | None,
    coordinates: np.ndarray,
    /,
) -> tuple[float, float]:
    scale = max(float(np.max(np.ptp(coordinates, axis=0))), 1.0)
    epsilon = np.finfo(coordinates.dtype).eps
    maximum = math.sqrt(epsilon) * scale
    tolerance = (
        128.0 * epsilon * scale
        if coordinate_tolerance is None
        else float(coordinate_tolerance)
    )
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > maximum:
        raise ValueError(
            "coordinate_tolerance must be finite, nonnegative, and no larger "
            "than sqrt(machine epsilon) times the interface scale."
        )
    return tolerance, scale


def _surface_vertex_route(
    fem_coordinates: np.ndarray,
    boundary_vertices: np.ndarray,
    surface_coordinates: np.ndarray,
    tolerance: float,
    /,
) -> tuple[np.ndarray, float]:
    if surface_coordinates.shape != (boundary_vertices.size, 3):
        raise ValueError(
            "The matching interface must contain every FEM boundary vertex exactly once."
        )
    candidates = fem_coordinates[boundary_vertices]
    route = np.full((surface_coordinates.shape[0],), -1, dtype=np.int32)
    errors = np.zeros((surface_coordinates.shape[0],), dtype=float)
    if tolerance == 0.0:
        exact = {
            tuple(float(value) for value in point): int(vertex)
            for point, vertex in zip(candidates, boundary_vertices, strict=True)
        }
        if len(exact) != boundary_vertices.size:
            raise ValueError("FEM boundary coordinates must be unique.")
        for index, point in enumerate(surface_coordinates):
            route[index] = exact.get(tuple(float(value) for value in point), -1)
    else:
        origin = np.minimum(
            np.min(candidates, axis=0), np.min(surface_coordinates, axis=0)
        )
        candidate_bins = np.floor((candidates - origin) / tolerance).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for local, key in enumerate(candidate_bins):
            buckets.setdefault(tuple(int(value) for value in key), []).append(local)
        offsets = tuple(product((-1, 0, 1), repeat=3))
        for index, point in enumerate(surface_coordinates):
            key = np.floor((point - origin) / tolerance).astype(np.int64)
            local_candidates = tuple(
                local
                for offset in offsets
                for local in buckets.get(
                    tuple(int(key[axis] + offset[axis]) for axis in range(3)), ()
                )
            )
            accepted = tuple(
                local
                for local in local_candidates
                if float(np.linalg.norm(point - candidates[local])) <= tolerance
            )
            if len(accepted) != 1:
                raise ValueError(
                    "Surface coordinates do not bijectively match the FEM exterior "
                    "vertices."
                )
            local = accepted[0]
            route[index] = boundary_vertices[local]
            errors[index] = float(np.linalg.norm(point - candidates[local]))
    if np.any(route < 0) or np.unique(route).size != boundary_vertices.size:
        raise ValueError(
            "Surface coordinates do not bijectively match the FEM exterior vertices."
        )
    return route, float(np.max(errors, initial=0.0))


def _face_routes(
    connectivity: TetrahedralConnectivity,
    surface_faces: np.ndarray,
    surface_to_fem_vertices: np.ndarray,
    selected_faces: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    fem_faces = np.asarray(connectivity.faces, dtype=np.int32)
    keys: Mapping[tuple[int, int, int], int] = {
        tuple(sorted(int(value) for value in fem_faces[face])): int(face)
        for face in selected_faces
    }
    if len(keys) != selected_faces.size:
        raise ValueError("FEM exterior facets do not have unique vertex triples.")
    mapped_faces = surface_to_fem_vertices[surface_faces]
    route = np.asarray(
        [
            keys.get(tuple(sorted(int(value) for value in face)), -1)
            for face in mapped_faces
        ],
        dtype=np.int32,
    )
    if (
        np.any(route < 0)
        or np.unique(route).size != selected_faces.size
        or set(int(value) for value in route)
        != set(int(value) for value in selected_faces)
    ):
        raise ValueError(
            "Surface triangles do not bijectively match the selected FEM exterior facets."
        )
    return route, mapped_faces


def prepare_matching_scalar_interface_trace_3d(
    discretization: FiniteElementDiscretization,
    surface: MeshRegion,
    boundary_space: AbstractVectorSpace,
    /,
    *,
    field_name: str = "u",
    interface: EntitySelection | None = None,
    coordinate_tolerance: float | None = None,
) -> PreparedMatchingScalarInterfaceTrace3D:
    """Prepare exact trace, boundary-load, and conormal routes for 3D P1/DP0.

    Geometry validation is intentionally authoritative here: all coupling
    preparation consumes this one result rather than repeating or weakening
    coordinate, topology, or orientation checks.
    """

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be a FiniteElementDiscretization.")
    if not isinstance(surface, MeshRegion):
        raise TypeError("surface must be a MeshRegion.")
    if not isinstance(boundary_space, ArraySpace):
        raise TypeError("boundary_space must be a one-dimensional ArraySpace.")
    mesh = discretization.mesh
    if mesh.ambient_dimension != 3 or mesh.topological_dimension != 3:
        raise ValueError("Matching scalar FEM–BEM preparation is three-dimensional.")
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "tetrahedron":
        raise ValueError("The supported FEM envelope is one affine tetrahedron block.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("Matching scalar FEM–BEM requires tetrahedral connectivity.")

    index = _field_index(discretization, field_name)
    elements = discretization.elements[index]
    dof_map = discretization.dof_maps[index]
    field_space = discretization.field_spaces[index].vector_space
    if (
        any(
            element.cell_kind != "tetrahedron"
            or element.family != "Lagrange"
            or element.degree != 1
            or element.conformity != "H1"
            or element.mapping != "identity"
            or element.value_shape
            for element in elements
        )
        or dof_map.association != "vertex"
        or dof_map.component_shape
        or field_space.shape != (mesh.coordinates.shape[0],)
    ):
        raise ValueError(
            "The supported scalar trace envelope is nodal H1 Lagrange P1 on "
            "affine tetrahedra."
        )

    block = mesh.blocks[0]
    cell_dofs = np.asarray(dof_map.cell_dofs[0], dtype=np.int32)
    cells = np.asarray(block.vertices, dtype=np.int32)
    if cell_dofs.shape != cells.shape or not np.array_equal(cell_dofs, cells):
        raise ValueError("P1 FEM DOF routes must coincide with tetrahedron vertices.")
    coordinate_element = discretization.coordinate_elements[0]
    coordinate_dofs = np.asarray(discretization.coordinate_dofs[0], dtype=np.int32)
    if (
        coordinate_element.cell_kind != "tetrahedron"
        or coordinate_element.family != "Lagrange"
        or coordinate_element.degree != 1
        or coordinate_element.conformity != "H1"
        or coordinate_element.mapping != "identity"
        or coordinate_element.value_shape
        or coordinate_dofs.shape != cells.shape
        or not np.array_equal(coordinate_dofs, cells)
    ):
        raise ValueError(
            "Matching scalar FEM-BEM requires affine P1 tetrahedral geometry."
        )

    domain = discretization.integration_domain("exterior_facet", interface)
    selected_faces = np.asarray(domain.entity_indices, dtype=np.int32)
    all_exterior = np.asarray(
        discretization.exterior_facet_domain.entity_indices, dtype=np.int32
    )
    if selected_faces.size == 0 or not np.array_equal(
        np.sort(selected_faces), np.sort(all_exterior)
    ):
        raise ValueError(
            "The bounded coupling envelope requires the complete FEM exterior boundary."
        )
    if (
        boundary_space.shape != (selected_faces.size,)
        or boundary_space.dtype != field_space.dtype
    ):
        raise ValueError(
            "The DP0 boundary space must have one scalar per surface face and "
            "the FEM scalar dtype."
        )

    fem_coordinates = np.asarray(discretization.default_runtime.coordinates)
    surface_coordinates = np.asarray(surface.vertices)
    surface_faces = np.asarray(surface.faces, dtype=np.int32)
    if (
        fem_coordinates.shape != mesh.coordinates.shape
        or fem_coordinates.shape[1:] != (3,)
        or surface_coordinates.ndim != 2
        or surface_coordinates.shape[1:] != (3,)
        or surface_faces.shape != (selected_faces.size, 3)
        or np.any(~np.isfinite(fem_coordinates))
        or np.any(~np.isfinite(surface_coordinates))
    ):
        raise ValueError("Interface coordinates/faces must be finite triangles in 3D.")
    tolerance, scale = _coordinate_tolerance(coordinate_tolerance, fem_coordinates)
    boundary_vertices = np.flatnonzero(
        np.asarray(connectivity.boundary_vertices, dtype=bool)
    ).astype(np.int32)
    surface_to_fem, coordinate_error = _surface_vertex_route(
        fem_coordinates,
        boundary_vertices,
        surface_coordinates,
        tolerance,
    )
    fem_face_route, mapped_faces = _face_routes(
        connectivity,
        surface_faces,
        surface_to_fem,
        selected_faces,
    )

    owner_by_face = dict(
        zip(
            np.asarray(
                discretization.exterior_facet_domain.entity_indices, dtype=np.int32
            ).tolist(),
            np.asarray(
                discretization.exterior_facet_domain.owner_cells, dtype=np.int32
            ).tolist(),
            strict=True,
        )
    )
    owner_cells = np.asarray(
        [owner_by_face[int(face)] for face in fem_face_route], dtype=np.int32
    )
    triangles = fem_coordinates[mapped_faces]
    area_vectors = 0.5 * np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(area_vectors, axis=1)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("Matching interface triangles must have positive finite area.")
    normals = area_vectors / areas[:, None]
    face_centres = np.mean(triangles, axis=1)
    cell_centres = np.mean(fem_coordinates[cells[owner_cells]], axis=1)
    directions = face_centres - cell_centres
    direction_norms = np.linalg.norm(directions, axis=1)
    orientation_cosines = np.sum(normals * directions, axis=1) / direction_norms
    orientation_tolerance = 128.0 * np.finfo(fem_coordinates.dtype).eps
    if np.any(~np.isfinite(orientation_cosines)) or np.any(
        orientation_cosines <= orientation_tolerance
    ):
        raise ValueError(
            "Surface faces must use the FEM-interior-to-exterior orientation."
        )
    flux_closure = float(np.linalg.norm(np.sum(areas[:, None] * normals, axis=0)))
    boundary_measure = float(np.sum(areas))
    closure_tolerance = (
        512.0
        * np.finfo(fem_coordinates.dtype).eps
        * max(boundary_measure, scale * scale, 1.0)
    )
    if not math.isfinite(flux_closure) or flux_closure > closure_tolerance:
        raise ValueError("Interface area vectors do not close to zero flux.")

    owner_routes = cell_dofs[owner_cells]
    unique_owners, owner_inverse_route = np.unique(owner_cells, return_inverse=True)
    affine = np.concatenate(
        (
            np.ones((unique_owners.size, 4, 1), dtype=fem_coordinates.dtype),
            fem_coordinates[cells[unique_owners]],
        ),
        axis=2,
    )
    inverse_affine = np.linalg.inv(affine)[owner_inverse_route]
    conormal_weights = np.sum(normals[:, :, None] * inverse_affine[:, 1:, :], axis=1)

    face_dofs = jnp.asarray(mapped_faces, dtype=jnp.int32)
    owner_dofs = jnp.asarray(owner_routes, dtype=jnp.int32)
    conormal_weights_ = jnp.asarray(conormal_weights, dtype=field_space.dtype)
    face_areas = jnp.asarray(areas, dtype=boundary_space.dtype)
    trace_scale = jnp.asarray(1.0 / 3.0, dtype=field_space.dtype)

    def trace_action(values: Array) -> Array:
        return jnp.sum(values[face_dofs], axis=1) * trace_scale

    def trace_transpose(values: Array) -> Array:
        contributions = jnp.broadcast_to(values[:, None] * trace_scale, face_dofs.shape)
        return (
            jnp.zeros(field_space.shape, dtype=values.dtype)
            .at[face_dofs]
            .add(contributions)
        )

    trace_id = canonical_fingerprint(
        {
            "kind": "matching-scalar-interface-average-trace-3d",
            "fem": discretization.prepared_id,
            "surface": surface.feature_id,
            "field": field_name,
            "face_dofs": array_tree_fingerprint(face_dofs),
        }
    )
    trace_operator = FunctionLinearOperator(
        trace_action,
        source=field_space,
        target=boundary_space,
        transpose_action=trace_transpose,
        operator_id=trace_id,
    )

    boundary_load_operator = FunctionLinearOperator(
        lambda values: trace_operator.transpose_mv(face_areas * values),
        source=boundary_space,
        target=field_space,
        transpose_action=lambda values: face_areas * trace_operator.mv(values),
        operator_id=canonical_fingerprint(
            {
                "kind": "matching-scalar-interface-boundary-load-3d",
                "trace": trace_id,
                "areas": array_tree_fingerprint(face_areas),
            }
        ),
    )

    def conormal_action(values: Array) -> Array:
        return jnp.sum(conormal_weights_ * values[owner_dofs], axis=1)

    def conormal_transpose(values: Array) -> Array:
        contributions = values[:, None] * conormal_weights_
        return (
            jnp.zeros(field_space.shape, dtype=values.dtype)
            .at[owner_dofs]
            .add(contributions)
        )

    conormal_operator = FunctionLinearOperator(
        conormal_action,
        source=field_space,
        target=boundary_space,
        transpose_action=conormal_transpose,
        operator_id=canonical_fingerprint(
            {
                "kind": "matching-scalar-interface-conormal-3d",
                "fem": discretization.prepared_id,
                "surface": surface.feature_id,
                "owner_dofs": array_tree_fingerprint(owner_dofs),
                "weights": array_tree_fingerprint(conormal_weights_),
                "normal": "fem-interior-to-exterior",
            }
        ),
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-matching-scalar-interface-trace-3d",
            "fem": discretization.prepared_id,
            "surface": surface.feature_id,
            "boundary_space": boundary_space.space_id,
            "trace": trace_operator.operator_id,
            "conormal": conormal_operator.operator_id,
        }
    )
    itemsize = np.dtype(field_space.dtype).itemsize
    return PreparedMatchingScalarInterfaceTrace3D(
        trace_operator=trace_operator,
        boundary_load_operator=boundary_load_operator,
        conormal_operator=conormal_operator,
        interface_coordinates=jnp.asarray(surface_coordinates),
        interface_faces=jnp.asarray(surface_faces, dtype=jnp.int32),
        fem_vertex_indices=jnp.asarray(surface_to_fem, dtype=jnp.int32),
        fem_face_indices=jnp.asarray(fem_face_route, dtype=jnp.int32),
        owner_cells=jnp.asarray(owner_cells, dtype=jnp.int32),
        outward_normals=jnp.asarray(normals, dtype=field_space.dtype),
        face_areas=face_areas,
        coordinate_max_error=jnp.asarray(coordinate_error, dtype=field_space.dtype),
        minimum_orientation_cosine=jnp.asarray(
            float(np.min(orientation_cosines)), dtype=field_space.dtype
        ),
        flux_closure_norm=jnp.asarray(flux_closure, dtype=field_space.dtype),
        spatial_dimension=3,
        pde="scalar interior Poisson / homogeneous exterior Laplace",
        geometry_contract=(
            "complete affine tetrahedral exterior matched bijectively to one closed "
            "outward triangular surface"
        ),
        formulation_role="exact P1 facet-average trace and outward P1 conormal to DP0",
        normal_convention=(
            "normal points from the FEM interior into the exterior; both conormals "
            "use this same geometric normal"
        ),
        provider_ids=(
            discretization.prepared_id,
            surface.feature_id,
            boundary_space.space_id,
        ),
        precision_evidence=(
            str(discretization.precision_policy.policy_id),
            str(boundary_space.dtype),
        ),
        resource_evidence=(
            ("fem_dofs", field_space.size),
            ("interface_faces", selected_faces.size),
            ("trace_route_entries", 3 * selected_faces.size),
            ("conormal_route_entries", 4 * selected_faces.size),
            (
                "action_route_and_coefficient_bytes",
                int(
                    selected_faces.size * (7 * np.dtype(np.int32).itemsize + 5 * itemsize)
                ),
            ),
        ),
        error_evidence=(
            f"maximum matched-coordinate error {coordinate_error:.17g}",
            f"minimum outward-orientation cosine {float(np.min(orientation_cosines)):.17g}",
            f"closed-surface area-vector defect {flux_closure:.17g}",
            "continuum discretization error is not estimated",
        ),
        non_goals=(
            "nonmatching or mortar coupling",
            "partial or open interfaces",
            "curved, moving, higher-order, or two-dimensional interfaces",
            "vector or acoustic equations",
            "continuum certification",
        ),
        prepared_id=prepared_id,
    )


__all__ = [
    "PreparedMatchingScalarInterfaceTrace3D",
    "prepare_matching_scalar_interface_trace_3d",
]

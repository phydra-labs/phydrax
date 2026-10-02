#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared manifold surface geometry shared by thin-film transport routes.

The film routes use vertex-centered barycentric control volumes on a manifold
``TriangleTopology``. Extensive vertex content (liquid volume, surfactant
amount, momentum) exchanges only through antisymmetric fluxes on primal edges,
so totals are conserved to roundoff by construction. The topology is prepared
once on the host; geometry refresh for new coordinates is pure JAX and never
rebuilds connectivity.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..geometry.simplicial import DDGOperators, TriangleMesh
from ..sparse import EdgeRelation
from ..typing import checked


# Cotangent conductances are dimensionless; roundoff on right angles gives
# |w| of order machine epsilon, which must not be read as an obtuse violation.
_CONDUCTANCE_ROUNDOFF = 1.0e3 * float(np.finfo(np.float64).eps)


class FilmSurfaceTopology(StrictModule, NonTrainableState):
    """Host-prepared manifold connectivity for vertex-centered film transport.

    ``edges`` are the canonical ``TriangleTopology`` edges ``(low, high)``.
    ``edge_relation`` routes vertices to edges with ``edge_vertex_signs``
    ``(-1, +1)``, the oriented coboundary of the canonical cell complex, so an
    edge flux is positive from ``edges[:, 0]`` to ``edges[:, 1]``.
    """

    mesh: TriangleMesh
    edges: Array
    edge_faces: Array
    edge_relation: EdgeRelation
    edge_vertex_signs: Array
    halfedge_edge: Array
    halfedge_edge_sign: Array
    edge_orientation: Array
    boundary_vertices: Array
    boundary_edges: Array
    topology_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(self, mesh: TriangleMesh, /) -> None:
        topology = mesh.topology
        faces = np.asarray(topology.faces, dtype=np.int32)
        edges = np.asarray(topology.edges, dtype=np.int32)
        halfedge_edge = np.asarray(topology.halfedge_edge, dtype=np.int32)
        edge_halfedges = np.asarray(topology.edge_halfedges, dtype=np.int32)
        origin = faces.reshape((-1,))
        halfedge_sign = np.where(origin == edges[halfedge_edge, 0], 1.0, -1.0)
        edge_faces = np.where(edge_halfedges >= 0, edge_halfedges // 3, -1).astype(
            np.int32
        )
        boundary_edges = edge_halfedges[:, 1] < 0
        boundary_vertices = np.zeros((topology.num_vertices,), dtype=np.bool_)
        boundary_vertices[np.unique(edges[boundary_edges].reshape((-1,)))] = True
        if np.any(
            np.bincount(faces.reshape((-1,)), minlength=topology.num_vertices) == 0
        ):
            raise ValueError("Film surfaces cannot contain isolated vertices.")
        self.mesh = mesh
        self.edges = jnp.asarray(edges, dtype=jnp.int32)
        self.edge_faces = jnp.asarray(edge_faces, dtype=jnp.int32)
        self.edge_relation = EdgeRelation(
            edges.reshape((-1,)),
            np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
            source_size=topology.num_vertices,
            target_size=edges.shape[0],
        )
        self.edge_vertex_signs = jnp.asarray(
            np.tile(np.asarray([-1.0, 1.0]), edges.shape[0]), dtype=jnp.float64
        )
        self.halfedge_edge = jnp.asarray(halfedge_edge, dtype=jnp.int32)
        self.halfedge_edge_sign = jnp.asarray(halfedge_sign, dtype=jnp.float64)
        # +1 when the first incident face traverses the edge from low to high.
        self.edge_orientation = jnp.asarray(
            halfedge_sign[edge_halfedges[:, 0]], dtype=jnp.float64
        )
        self.boundary_vertices = jnp.asarray(boundary_vertices)
        self.boundary_edges = jnp.asarray(boundary_edges)
        self.topology_id = canonical_fingerprint(
            {
                "kind": "film-surface-topology",
                "faces": faces,
                "num_vertices": topology.num_vertices,
            }
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "film-surface-operators",
                "topology_id": self.topology_id,
                "stiffness": "cotangent",
                "mass": "barycentric-lumped",
                "curvature": "normal-cycle-vertex-star",
                "advection": "barycentric-dual-segment-p1",
            }
        )

    @property
    def num_vertices(self) -> int:
        return self.mesh.topology.num_vertices

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def watertight(self) -> bool:
        return self.mesh.topology.watertight


class FilmSurfaceEvidence(StrictModule):
    """Conductance and measure admissibility of one prepared film geometry.

    ``conductance_admissible`` holds when every cotangent edge conductance is
    nonnegative up to roundoff: the stiffness is then an M-matrix Laplacian
    (intrinsically Delaunay on interior edges, non-obtuse opposite angles on
    boundary edges), the premise of every film positivity claim.
    """

    minimum_edge_conductance: Array
    negative_conductance_count: Array
    minimum_vertex_area: Array
    minimum_face_area: Array
    conductance_admissible: Array
    measures_admissible: Array

    def __init__(
        self,
        *,
        minimum_edge_conductance: Array,
        negative_conductance_count: Array,
        minimum_vertex_area: Array,
        minimum_face_area: Array,
        conductance_admissible: Array,
        measures_admissible: Array,
    ) -> None:
        self.minimum_edge_conductance = jnp.asarray(minimum_edge_conductance)
        self.negative_conductance_count = jnp.asarray(
            negative_conductance_count, dtype=jnp.int32
        )
        self.minimum_vertex_area = jnp.asarray(minimum_vertex_area)
        self.minimum_face_area = jnp.asarray(minimum_face_area)
        self.conductance_admissible = jnp.asarray(conductance_admissible, dtype=jnp.bool_)
        self.measures_admissible = jnp.asarray(measures_admissible, dtype=jnp.bool_)

    @property
    def admissible(self) -> Array:
        return self.conductance_admissible & self.measures_admissible


class PreparedFilmSurface(StrictModule):
    """Current embedding, DDG operators and film geometry on fixed topology.

    ``vertex_area`` is the barycentric dual measure (the lumped DDG mass).
    ``curvature_squared`` is ``kappa_1^2 + kappa_2^2`` from the normal-cycle
    shape-operator estimate on each barycentric vertex cell; it vanishes
    exactly on planar meshes, including boundary vertices, where cotangent
    mean-curvature identities do not apply. ``geometry_revision`` is a
    dynamic counter so refreshed geometry stays traceable inside compiled
    loops; ``topology.topology_id`` and ``topology.operator_id`` are static.
    """

    topology: FilmSurfaceTopology
    coordinates: Array
    operators: DDGOperators
    vertex_area: Array
    vertex_normal: Array
    curvature_squared: Array
    geometry_revision: Array
    evidence: FilmSurfaceEvidence

    @checked
    def __init__(
        self,
        topology: FilmSurfaceTopology,
        coordinates: ArrayLike,
        /,
        *,
        geometry_revision: ArrayLike = 0,
    ) -> None:
        points = jnp.asarray(coordinates, dtype=jnp.float64)
        if points.shape != (topology.num_vertices, 3):
            raise ValueError("coordinates must have shape (num_vertices, 3).")
        revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        if revision.shape != ():
            raise ValueError("geometry_revision must be a scalar.")
        operators = DDGOperators(topology.mesh, vertices=points)
        weights = operators.edge_weights
        negative = weights < -_CONDUCTANCE_ROUNDOFF
        evidence = FilmSurfaceEvidence(
            minimum_edge_conductance=jnp.min(weights),
            negative_conductance_count=jnp.sum(negative),
            minimum_vertex_area=jnp.min(operators.vertex_mass),
            minimum_face_area=jnp.min(operators.face_area),
            conductance_admissible=~jnp.any(negative) & jnp.all(jnp.isfinite(weights)),
            measures_admissible=jnp.all(jnp.isfinite(operators.vertex_mass))
            & jnp.all(operators.vertex_mass > 0)
            & jnp.all(operators.face_area > 0),
        )
        self.topology = topology
        self.coordinates = points
        self.operators = operators
        self.vertex_area = operators.vertex_mass
        self.vertex_normal = operators.vertex_normals
        self.curvature_squared = _normal_cycle_curvature_squared(
            topology, points, operators
        )
        self.geometry_revision = revision
        self.evidence = evidence

    def refresh(self, coordinates: ArrayLike, /) -> PreparedFilmSurface:
        """Recompute numeric geometry for new coordinates on identical topology."""
        return PreparedFilmSurface(
            self.topology,
            coordinates,
            geometry_revision=self.geometry_revision + 1,
        )

    def edge_gradient(self, vertex_values: Array, /) -> Array:
        """Return ``u[edges[:, 0]] - u[edges[:, 1]]`` for vertex values ``u``."""
        values = jnp.asarray(vertex_values)
        return values[self.topology.edges[:, 0]] - values[self.topology.edges[:, 1]]

    def edge_divergence(self, edge_flux: Array, /) -> Array:
        """Return the net outflow per vertex of antisymmetric edge fluxes.

        ``edge_flux[e]`` flows from ``edges[e, 0]`` to ``edges[e, 1]``; the
        vertex sum of the result is exactly zero up to roundoff.
        """
        flux = jnp.asarray(edge_flux)
        if flux.shape[:1] != (self.topology.num_edges,):
            raise ValueError("edge_flux must have leading dimension num_edges.")
        result = jnp.zeros((self.topology.num_vertices, *flux.shape[1:]), flux.dtype)
        result = result.at[self.topology.edges[:, 0]].add(flux)
        return result.at[self.topology.edges[:, 1]].add(-flux)

    def edge_area_flux(self, vertex_velocity: Array, /) -> Array:
        """Area flux (m^2/s) of a P1 velocity through each barycentric dual edge.

        Inside face ``f`` the dual segment between corners ``a`` and ``b``
        runs from the edge midpoint to the barycenter; its conormal times
        length is ``A_f (grad phi_b - grad phi_a) / 3`` and the P1 velocity
        averaged over it is ``5/12 (v_a + v_b) + 1/6 v_c``. The flux is
        positive from ``edges[:, 0]`` to ``edges[:, 1]``, and its divergence is
        the exact integral of the tangential P1 divergence over each
        barycentric cell on planar meshes.
        """
        velocity = jnp.asarray(vertex_velocity)
        if velocity.shape != (self.topology.num_vertices, 3):
            raise ValueError("vertex_velocity must have shape (num_vertices, 3).")
        faces = self.operators.faces
        corner_velocity = velocity[faces]
        following = jnp.roll(corner_velocity, -1, axis=1)
        opposite = jnp.roll(corner_velocity, -2, axis=1)
        segment_velocity = (5.0 / 12.0) * (corner_velocity + following) + (
            1.0 / 6.0
        ) * opposite
        gradients = self.operators.basis_gradients
        conormal = (
            self.operators.face_area[:, None, None]
            * (jnp.roll(gradients, -1, axis=1) - gradients)
            / 3.0
        )
        halfedge_flux = jnp.sum(segment_velocity * conormal, axis=-1).reshape((-1,))
        edge_flux = jnp.zeros((self.topology.num_edges,), dtype=velocity.dtype)
        return edge_flux.at[self.topology.halfedge_edge].add(
            self.topology.halfedge_edge_sign * halfedge_flux
        )


def _normal_cycle_curvature_squared(
    topology: FilmSurfaceTopology, points: Array, operators: DDGOperators, /
) -> Array:
    """Return ``|S_v|_F^2`` from the Cohen-Steiner--Morvan edge estimate.

    ``S_v = (1 / A_v) sum_{e ∋ v} theta_e (l_e / 2) e_hat e_hat^T`` with
    signed dihedral angle ``theta_e``; boundary edges carry no dihedral angle.
    """
    first_face = topology.edge_faces[:, 0]
    second_face = jnp.maximum(topology.edge_faces[:, 1], 0)
    interior = ~topology.boundary_edges
    normal_first = operators.face_normal[first_face]
    normal_second = operators.face_normal[second_face]
    vector = points[topology.edges[:, 1]] - points[topology.edges[:, 0]]
    length = jnp.linalg.norm(vector, axis=-1)
    direction = vector / length[:, None]
    oriented = topology.edge_orientation[:, None] * direction
    angle = jnp.arctan2(
        jnp.sum(jnp.cross(normal_first, normal_second) * oriented, axis=-1),
        jnp.sum(normal_first * normal_second, axis=-1),
    )
    weight = jnp.where(interior, 0.5 * angle * length, 0.0)
    tensor = weight[:, None, None] * contract("ea,eb->eab", direction, direction)
    shape = jnp.zeros((topology.num_vertices, 3, 3), dtype=points.dtype)
    shape = shape.at[topology.edges[:, 0]].add(tensor)
    shape = shape.at[topology.edges[:, 1]].add(tensor)
    shape = shape / operators.vertex_mass[:, None, None]
    return jnp.sum(shape**2, axis=(-2, -1))


def prepare_film_surface(mesh: TriangleMesh, /) -> PreparedFilmSurface:
    """Prepare manifold film topology once and assemble the mesh geometry."""
    return PreparedFilmSurface(FilmSurfaceTopology(mesh), mesh.vertices)


__all__ = [
    "FilmSurfaceEvidence",
    "FilmSurfaceTopology",
    "PreparedFilmSurface",
    "prepare_film_surface",
]

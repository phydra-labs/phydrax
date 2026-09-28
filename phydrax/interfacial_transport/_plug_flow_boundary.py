#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Open, wall and inflow boundaries of plug flow on bordered film surfaces.

Vertex-centered barycentric cells of boundary vertices are closed by the
halves of boundary edges. For a P1 velocity the outward area flux through
the half of boundary edge ``(a, b)`` next to ``a`` is exactly

``phi_a = (l_e nu_e / 2) . (3 u_a + u_b) / 4``,

with the outward in-plane conormal ``l_e nu_e = -2 A_f grad phi_c`` of the
incident face and opposite corner ``c``. Together with the interior
dual-edge fluxes of ``PreparedFilmSurface.edge_area_flux`` this closes the
discrete divergence theorem cell by cell on planar meshes, so a uniform film
translating through open boundaries stays uniform. Open edges transport
content by donor cell: outflow and backflow on ``outflow`` edges carry the
interior density (zero gradient), inflow on ``inflow`` edges carries the
prescribed exterior density.

Velocity constraints enter the implicit momentum block as rows
``Q (u - u_D)`` with a per-vertex projector ``Q``: the tangent projector for
held vertices (``no-slip`` walls and ``inflow``) and ``nu nu^T`` with the
averaged wall conormal for ``free-slip`` vertices. The force a constraint
exerts on the film is reported with the boundary line tension
``oint phi_i T nu ds``, which the strong-form Marangoni vertex force omits,
so a film pulling with uniform tension on a closed rim exerts no net force.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import get_args, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, parameter_field, ParameterOwner
from ..typing import ConvertibleToArray, parse
from ._film_contracts import PreparedFilmSurface


PlugFlowBoundaryKind: TypeAlias = Literal[
    "no-flux", "no-slip", "free-slip", "inflow", "outflow"
]
"""Kind of one boundary edge of a plug-flow film.

``no-flux`` closes the edge and leaves the velocity traction-free;
``no-slip`` closes the edge and holds its vertices at rest (wire or obstacle
rim); ``free-slip`` closes the edge and holds only the boundary-normal
velocity at zero; ``inflow`` opens the edge with the prescribed exterior
state and holds its vertices at the prescribed velocity; ``outflow`` opens
the edge with the interior state and leaves the velocity traction-free.
Vertex constraints take precedence ``no-slip``, ``inflow``, ``free-slip``."""

_KINDS: tuple[PlugFlowBoundaryKind, ...] = get_args(PlugFlowBoundaryKind)

# Per-vertex velocity constraint codes, in ascending precedence.
_NATURAL = 0
_NORMAL = 1
_PRESCRIBED = 2
_REST = 3


def _kind_code(kind: PlugFlowBoundaryKind, /) -> int:
    return _KINDS.index(kind)


class PlugFlowBoundaryEvidence(StrictModule):
    """Per-vertex boundary exchange of one plug-flow step.

    Outflows are the step-integrated amounts leaving each vertex cell through
    open boundary edges (negative for inflow). ``surfactant_outflow_mol``
    counts both interfaces and dissolved surfactant. ``constraint_force_n`` is
    the force the velocity constraint exerts on the film during the step,
    including the boundary line tension; it is zero on unconstrained
    vertices. The force of the film on a wall or obstacle is its negative sum
    over the wall vertices.
    """

    volume_outflow_m3: Array
    surfactant_outflow_mol: Array
    momentum_outflow_n_s: Array
    constraint_force_n: Array

    def __init__(
        self,
        *,
        volume_outflow_m3: Array,
        surfactant_outflow_mol: Array,
        momentum_outflow_n_s: Array,
        constraint_force_n: Array,
    ) -> None:
        self.volume_outflow_m3 = jnp.asarray(volume_outflow_m3)
        self.surfactant_outflow_mol = jnp.asarray(surfactant_outflow_mol)
        self.momentum_outflow_n_s = jnp.asarray(momentum_outflow_n_s)
        self.constraint_force_n = jnp.asarray(constraint_force_n)


class PlugFlowBoundary(StrictModule, ParameterOwner):
    """Boundary-edge kinds and prescribed inflow state of a bordered film.

    ``boundary_vertices`` maps each declared kind to vertex ids on the
    surface boundary; sets may overlap at corners. A boundary edge takes kind
    ``k`` when both endpoints belong to set ``k``, and every boundary edge
    must take exactly one kind. Inflow values broadcast to every vertex and
    are used on inflow edges and held inflow vertices only; the exterior
    dissolved concentration is required exactly for soluble films.
    """

    edges: Array = fixed_field()
    faces: Array = fixed_field()
    opposite_corner: Array = fixed_field()
    edge_kind: Array = fixed_field()
    vertex_constraint: Array = fixed_field()
    inflow_velocity_m_s: Array = parameter_field()
    inflow_thickness_m: Array = parameter_field()
    inflow_surface_concentration_mol_m2: Array = parameter_field()
    inflow_dissolved_concentration_mol_m3: Array | None = parameter_field()
    topology_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedFilmSurface,
        boundary_vertices: Mapping[PlugFlowBoundaryKind, ArrayLike],
        /,
        *,
        inflow_velocity_m_s: ConvertibleToArray = (0.0, 0.0, 0.0),
        inflow_thickness_m: ArrayLike = 0.0,
        inflow_surface_concentration_mol_m2: ArrayLike = 0.0,
        inflow_dissolved_concentration_mol_m3: ArrayLike | None = None,
    ) -> None:
        if not isinstance(surface, PreparedFilmSurface):
            raise TypeError("surface must be a PreparedFilmSurface.")
        if not isinstance(boundary_vertices, Mapping):
            raise TypeError("boundary_vertices must map boundary kinds to vertex ids.")
        topology = surface.topology
        if topology.watertight:
            raise ValueError("Plug-flow boundaries require a bordered surface.")
        count = topology.num_vertices
        members = _membership(
            np.asarray(topology.boundary_vertices), boundary_vertices, count
        )
        mask = np.asarray(topology.boundary_edges)
        edges = np.asarray(topology.edges)[mask]
        edge_kind = _edge_kinds(edges, members)
        faces = np.asarray(topology.edge_faces)[mask, 0]
        corners = np.asarray(surface.operators.faces)[faces]
        opposite = np.argmax(
            (corners != edges[:, :1]) & (corners != edges[:, 1:]), axis=1
        ).astype(np.int32)
        velocity = _broadcast(inflow_velocity_m_s, (count, 3), "inflow_velocity_m_s")
        thickness = _broadcast(inflow_thickness_m, (count,), "inflow_thickness_m")
        concentration = _broadcast(
            inflow_surface_concentration_mol_m2,
            (count,),
            "inflow_surface_concentration_mol_m2",
        )
        dissolved = (
            None
            if inflow_dissolved_concentration_mol_m3 is None
            else _broadcast(
                inflow_dissolved_concentration_mol_m3,
                (count,),
                "inflow_dissolved_concentration_mol_m3",
            )
        )
        inflow_vertices = np.unique(edges[edge_kind == _kind_code("inflow")])
        if np.any(thickness[inflow_vertices] <= 0.0):
            raise ValueError("Inflow thickness must be positive on inflow edges.")
        if np.any(concentration[inflow_vertices] < 0.0) or (
            dissolved is not None and np.any(dissolved[inflow_vertices] < 0.0)
        ):
            raise ValueError("Inflow concentrations must be nonnegative.")
        self.edges = jnp.asarray(edges, dtype=jnp.int32)
        self.faces = jnp.asarray(faces, dtype=jnp.int32)
        self.opposite_corner = jnp.asarray(opposite, dtype=jnp.int32)
        self.edge_kind = jnp.asarray(edge_kind, dtype=jnp.int32)
        self.vertex_constraint = jnp.asarray(
            _vertex_constraints(edges, edge_kind, count), dtype=jnp.int32
        )
        self.inflow_velocity_m_s = jnp.asarray(velocity)
        self.inflow_thickness_m = jnp.asarray(thickness)
        self.inflow_surface_concentration_mol_m2 = jnp.asarray(concentration)
        self.inflow_dissolved_concentration_mol_m3 = (
            None if dissolved is None else jnp.asarray(dissolved)
        )
        self.topology_id = topology.topology_id
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "plug-flow-boundary",
                "topology_id": topology.topology_id,
                "edges": edges,
                "edge_kind": [_KINDS[code] for code in edge_kind],
                "dissolved_inflow": dissolved is not None,
            }
        )

    @property
    def open_edges(self) -> Array:
        return (self.edge_kind == _kind_code("inflow")) | (
            self.edge_kind == _kind_code("outflow")
        )

    @property
    def held_vertices(self) -> Array:
        """Vertices whose full tangential velocity is prescribed."""
        return self.vertex_constraint >= _PRESCRIBED

    @property
    def constrained_vertices(self) -> Array:
        return self.vertex_constraint != _NATURAL

    def conormals(self, surface: PreparedFilmSurface, /) -> Array:
        """Return outward in-plane conormals times length ``l_e nu_e`` (m)."""
        operators = surface.operators
        gradient = operators.basis_gradients[self.faces, self.opposite_corner]
        return -2.0 * operators.face_area[self.faces, None] * gradient

    def area_flux(self, surface: PreparedFilmSurface, velocity: Array, /) -> Array:
        """Return outward area flux (m^2/s) through both halves of open edges.

        The result has shape ``(num_boundary_edges, 2)``; column ``j`` is the
        half next to ``edges[:, j]``. Closed edges carry zero flux.
        """
        conormal = self.conormals(surface)
        first = velocity[self.edges[:, 0]]
        second = velocity[self.edges[:, 1]]
        halves = (
            jnp.stack(
                (
                    jnp.sum(conormal * (3.0 * first + second), axis=1),
                    jnp.sum(conormal * (first + 3.0 * second), axis=1),
                ),
                axis=1,
            )
            / 8.0
        )
        return jnp.where(self.open_edges[:, None], halves, 0.0)

    def donor_density(
        self,
        surface: PreparedFilmSurface,
        content: Array,
        exterior_density: Array,
        area_flux: Array,
        /,
    ) -> Array:
        """Return the donor-cell density on every boundary half-edge.

        Outflow halves and every half of an ``outflow`` edge carry the cell
        density ``content / A``; inflow halves of ``inflow`` edges carry the
        exterior density of their vertex.
        """
        trailing = (1,) * (content.ndim - 1)
        density = content / surface.vertex_area.reshape((-1,) + trailing)
        inflow = (self.edge_kind == _kind_code("inflow"))[:, None] & (area_flux < 0.0)
        return jnp.where(
            inflow.reshape(inflow.shape + trailing),
            exterior_density[self.edges],
            density[self.edges],
        )

    def vertex_sum(self, surface: PreparedFilmSurface, values: Array, /) -> Array:
        """Scatter half-edge values ``(num_boundary_edges, 2, ...)`` to vertices."""
        trailing = values.shape[2:]
        result = jnp.zeros(
            (surface.topology.num_vertices,) + trailing, dtype=values.dtype
        )
        return result.at[self.edges.reshape((-1,))].add(values.reshape((-1,) + trailing))

    def outflow(
        self,
        surface: PreparedFilmSurface,
        content: Array,
        exterior_density: Array,
        area_flux: Array,
        /,
    ) -> Array:
        """Return the donor-cell content rate leaving each vertex cell."""
        density = self.donor_density(surface, content, exterior_density, area_flux)
        flux = area_flux.reshape(area_flux.shape + (1,) * (content.ndim - 1))
        return self.vertex_sum(surface, flux * density)

    def outflow_rate(self, surface: PreparedFilmSurface, area_flux: Array, /) -> Array:
        """Return the outgoing boundary area flux per vertex (m^2/s)."""
        return self.vertex_sum(surface, jnp.maximum(area_flux, 0.0))

    def line_tension_force(
        self, surface: PreparedFilmSurface, tension: Array, /
    ) -> Array:
        """Return ``oint_boundary phi_i T nu ds`` per vertex (N) for P1 tension ``T``.

        The integral is exact for piecewise-linear tension (weights one third
        and one sixth) and vanishes on interior vertices; for uniform tension
        its sum over a closed boundary loop is zero.
        """
        conormal = self.conormals(surface)
        first = tension[self.edges[:, 0]]
        second = tension[self.edges[:, 1]]
        halves = jnp.stack(
            (
                conormal * (first / 3.0 + second / 6.0)[:, None],
                conormal * (first / 6.0 + second / 3.0)[:, None],
            ),
            axis=1,
        )
        return self.vertex_sum(surface, halves)

    def velocity_constraint(self, surface: PreparedFilmSurface, /) -> tuple[Array, Array]:
        """Return per-vertex projectors ``Q`` and prescribed velocities ``u_D``.

        ``Q`` is the tangent projector on held vertices, ``nu nu^T`` with the
        tangential unit average of the incident free-slip conormals on
        free-slip vertices, and zero elsewhere. ``u_D`` is the tangential
        inflow velocity on held inflow vertices and zero elsewhere.
        """
        normal = surface.vertex_normal
        tangent = jnp.eye(3, dtype=normal.dtype) - normal[:, :, None] * normal[:, None, :]
        slip = (self.edge_kind == _kind_code("free-slip"))[:, None, None]
        conormal = jnp.where(slip, self.conormals(surface)[:, None, :], 0.0)
        wall = self.vertex_sum(
            surface, jnp.broadcast_to(conormal, (conormal.shape[0], 2, 3))
        )
        wall = wall - jnp.sum(wall * normal, axis=1, keepdims=True) * normal
        slipping = (self.vertex_constraint == _NORMAL)[:, None]
        wall = wall / jnp.where(
            slipping, jnp.linalg.norm(wall, axis=1, keepdims=True), 1.0
        )
        code = self.vertex_constraint[:, None, None]
        projector = jnp.where(
            code >= _PRESCRIBED,
            tangent,
            jnp.where(code == _NORMAL, wall[:, :, None] * wall[:, None, :], 0.0),
        )
        inflow = self.inflow_velocity_m_s
        prescribed = jnp.where(
            (self.vertex_constraint == _PRESCRIBED)[:, None],
            inflow - jnp.sum(inflow * normal, axis=1, keepdims=True) * normal,
            0.0,
        )
        return projector, prescribed


def _membership(
    boundary: np.ndarray,
    boundary_vertices: Mapping[PlugFlowBoundaryKind, ArrayLike],
    count: int,
    /,
) -> np.ndarray:
    members = np.zeros((len(_KINDS), count), dtype=np.bool_)
    for key, value in boundary_vertices.items():
        kind = parse(key, PlugFlowBoundaryKind, "boundary kind")
        ids = np.asarray(value)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"Vertex ids of {kind!r} must be a 1-D integer array.")
        if np.any((ids < 0) | (ids >= count)):
            raise ValueError(f"Vertex ids of {kind!r} are out of range.")
        if not np.all(boundary[ids]):
            raise ValueError(f"Vertex ids of {kind!r} must lie on the surface boundary.")
        members[_kind_code(kind), ids] = True
    return members


def _edge_kinds(edges: np.ndarray, members: np.ndarray, /) -> np.ndarray:
    covered = members[:, edges[:, 0]] & members[:, edges[:, 1]]
    counts = np.sum(covered, axis=0)
    if np.any(counts == 0):
        raise ValueError(f"{np.sum(counts == 0)} boundary edges have no declared kind.")
    if np.any(counts > 1):
        raise ValueError(f"{np.sum(counts > 1)} boundary edges match several kinds.")
    return np.argmax(covered, axis=0).astype(np.int32)


def _vertex_constraints(
    edges: np.ndarray, edge_kind: np.ndarray, count: int, /
) -> np.ndarray:
    constraint = np.full((count,), _NATURAL, dtype=np.int32)
    for kind, code in (
        ("free-slip", _NORMAL),
        ("inflow", _PRESCRIBED),
        ("no-slip", _REST),
    ):
        constraint[np.unique(edges[edge_kind == _kind_code(kind)])] = code
    return constraint


def _broadcast(
    value: ConvertibleToArray, shape: tuple[int, ...], name: str, /
) -> np.ndarray:
    host = np.asarray(value, dtype=np.float64)
    if host.shape not in {(), shape[1:], shape}:
        raise ValueError(f"{name} must have shape {shape[1:]} or {shape}.")
    host = np.broadcast_to(host, shape).copy()
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    return host


__all__ = [
    "PlugFlowBoundary",
    "PlugFlowBoundaryEvidence",
    "PlugFlowBoundaryKind",
]

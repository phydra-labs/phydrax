#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit conservative upwind transport of extensive vertex content.

Content ``X_i`` lives on barycentric cells of area ``A_i`` with density
``X_i / A_i``. Given dual-edge area fluxes ``phi_e`` (m^2/s, positive from
``edges[:, 0]`` to ``edges[:, 1]``), the donor-cell flux is
``phi+ X_0/A_0 - phi- X_1/A_1``. The update is exactly conservative, and it
preserves nonnegativity when the outflow Courant number
``max_i dt sum_out phi / A_i`` is at most one.

The limited second-order flux reconstructs the donor density at the edge
midpoint with the edge-based MUSCL scheme of vertex-centered finite volumes
(Dervieux and coworkers): the upwind-extended difference
``2 g_d . (x_c - x_d) - (rho_c - rho_d)`` from the lumped P1 vertex gradient
``g_d`` is minmod-limited against the central difference, so the face
density stays between the donor and the edge mean. It is still exactly
conservative; it removes the first-order numerical viscosity
``|u| dx / 2`` of the donor cell in smooth regions but guarantees neither
positivity nor extremum diminishing on unstructured meshes, so consumers
check the transported content.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import jax.numpy as jnp
from jax import Array

from ..ein import contract
from ._film_contracts import PreparedFilmSurface


FilmTransportScheme: TypeAlias = Literal["donor-cell", "limited-muscl"]
"""Interior edge reconstruction of explicit film transport: first-order
``donor-cell`` (positivity for outflow Courant numbers up to one) or
second-order ``limited-muscl`` (``limited_edge_flux``; stable for Courant
numbers up to one half, positivity not guaranteed)."""


def transported_edge_flux(
    scheme: FilmTransportScheme,
    surface: PreparedFilmSurface,
    content: Array,
    area_flux: Array,
    area: Array,
    /,
) -> Array:
    """Return interior edge fluxes of ``content`` under the selected scheme."""
    match scheme:
        case "donor-cell":
            return upwind_edge_flux(surface, content, area_flux, area)
        case "limited-muscl":
            return limited_edge_flux(surface, content, area_flux, area)
        case _:
            raise ValueError(f"Unknown film transport scheme {scheme!r}.")


def courant_limit(scheme: FilmTransportScheme, /) -> float:
    """Return the largest admitted outflow Courant number of ``scheme``."""
    match scheme:
        case "donor-cell":
            return 1.0
        case "limited-muscl":
            return 0.5
        case _:
            raise ValueError(f"Unknown film transport scheme {scheme!r}.")


def upwind_edge_flux(
    surface: PreparedFilmSurface, content: Array, area_flux: Array, area: Array, /
) -> Array:
    """Return donor-cell edge fluxes of ``content`` (leading vertex axis)."""
    edges = surface.topology.edges
    density = content / area.reshape((-1,) + (1,) * (content.ndim - 1))
    shape = (-1,) + (1,) * (content.ndim - 1)
    forward = jnp.maximum(area_flux, 0.0).reshape(shape)
    backward = jnp.maximum(-area_flux, 0.0).reshape(shape)
    return forward * density[edges[:, 0]] - backward * density[edges[:, 1]]


def limited_edge_flux(
    surface: PreparedFilmSurface, content: Array, area_flux: Array, area: Array, /
) -> Array:
    """Return minmod-limited MUSCL edge fluxes of ``content`` (leading vertex axis)."""
    edges = surface.topology.edges
    operators = surface.operators
    trailing = (1,) * (content.ndim - 1)
    density = content / area.reshape((-1,) + trailing)
    # Lumped L2 projection of the P1 face gradients onto vertices.
    weighted = (operators.face_area / 3.0).reshape(
        (-1, 1) + trailing
    ) * operators.gradient(density)
    gradient = (
        jnp.zeros((surface.topology.num_vertices,) + weighted.shape[1:], weighted.dtype)
        .at[operators.faces.reshape((-1,))]
        .add(jnp.repeat(weighted, 3, axis=0))
        / area.reshape((-1, 1) + trailing)
    )
    tail, head = edges[:, 0], edges[:, 1]
    span = surface.coordinates[head] - surface.coordinates[tail]
    difference = density[head] - density[tail]
    tail_slope = 2.0 * contract("ea...,ea->e...", gradient[tail], span) - difference
    head_slope = 2.0 * contract("ea...,ea->e...", gradient[head], span) - difference
    forward_face = density[tail] + 0.5 * _minmod(tail_slope, difference)
    backward_face = density[head] - 0.5 * _minmod(head_slope, difference)
    shape = (-1,) + trailing
    forward = jnp.maximum(area_flux, 0.0).reshape(shape)
    backward = jnp.maximum(-area_flux, 0.0).reshape(shape)
    return forward * forward_face - backward * backward_face


def _minmod(first: Array, second: Array, /) -> Array:
    same = first * second > 0.0
    return jnp.where(
        same, jnp.sign(first) * jnp.minimum(jnp.abs(first), jnp.abs(second)), 0.0
    )


def outflow_courant(
    surface: PreparedFilmSurface,
    area_flux: Array,
    area: Array,
    step_size: Array,
    /,
    *,
    boundary_outflow: Array | None = None,
) -> Array:
    """Return ``max_i dt sum_out phi / A_i`` for donor-cell transport.

    ``boundary_outflow`` adds the outgoing area flux of open boundary edges
    per vertex.
    """
    edges = surface.topology.edges
    outflow = jnp.zeros((surface.topology.num_vertices,), dtype=area_flux.dtype)
    outflow = outflow.at[edges[:, 0]].add(jnp.maximum(area_flux, 0.0))
    outflow = outflow.at[edges[:, 1]].add(jnp.maximum(-area_flux, 0.0))
    if boundary_outflow is not None:
        outflow = outflow + boundary_outflow
    return jnp.max(step_size * outflow / area)


__all__ = [
    "FilmTransportScheme",
    "courant_limit",
    "limited_edge_flux",
    "outflow_courant",
    "transported_edge_flux",
    "upwind_edge_flux",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite compact U(1) Wilson measures on explicit oriented cell incidence."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


_TWO_PI = 2.0 * jnp.pi


def wrap_u1(angle: ArrayLike, /) -> Array:
    value = jnp.asarray(angle)
    return jnp.mod(value + jnp.pi, _TWO_PI) - jnp.pi


class CompactU1GaugeMeasure(StrictModule):
    """Static finite U(1) Wilson action; no continuum/non-Abelian claim."""

    plaquette_edge_incidence: Array
    vertex_edge_incidence: Array
    beta: Array
    topology_residual: Array
    valid: Array
    num_edges: int = eqx.field(static=True)
    num_plaquettes: int = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        plaquette_edge_incidence: ArrayLike,
        vertex_edge_incidence: ArrayLike,
        /,
        *,
        beta: float,
        topology_tolerance: float = 1e-12,
    ):
        plaquettes = jnp.asarray(plaquette_edge_incidence)
        vertices = jnp.asarray(vertex_edge_incidence)
        if (
            plaquettes.ndim != 2
            or vertices.ndim != 2
            or plaquettes.shape[1] != vertices.shape[1]
        ):
            raise ValueError("Incidence arrays must be rank two and share the edge axis.")
        if min(plaquettes.shape) < 1 or vertices.shape[0] < 1:
            raise ValueError("Gauge complexes require vertices, edges, and plaquettes.")
        if not jnp.issubdtype(plaquettes.dtype, jnp.integer) or not jnp.issubdtype(
            vertices.dtype, jnp.integer
        ):
            raise TypeError("Gauge incidence arrays must be integer-valued.")
        beta_ = float(beta)
        if not np.isfinite(beta_) or beta_ < 0.0:
            raise ValueError("beta must be finite and non-negative.")
        boundary_squared = plaquettes.astype(float) @ vertices.astype(float).T
        residual = jnp.max(jnp.abs(boundary_squared))
        self.plaquette_edge_incidence = plaquettes
        self.vertex_edge_incidence = vertices
        self.beta = jnp.asarray(beta_)
        self.topology_residual = residual
        self.valid = residual <= topology_tolerance
        self.num_edges = int(plaquettes.shape[1])
        self.num_plaquettes = int(plaquettes.shape[0])
        self.claim = "finite-compact-u1-wilson-measure"

    def plaquette_angles(self, link_angles: ArrayLike, /) -> Array:
        links = jnp.asarray(link_angles)
        if links.shape != (self.num_edges,):
            raise ValueError(f"link_angles must have shape ({self.num_edges},).")
        return wrap_u1(self.plaquette_edge_incidence @ links)

    def action(self, link_angles: ArrayLike, /) -> Array:
        return -self.beta * jnp.sum(jnp.cos(self.plaquette_angles(link_angles)))

    def gauge_transform(
        self, link_angles: ArrayLike, vertex_phases: ArrayLike, /
    ) -> Array:
        links = jnp.asarray(link_angles)
        phases = jnp.asarray(vertex_phases, dtype=links.dtype)
        if links.shape != (self.num_edges,) or phases.shape != (
            self.vertex_edge_incidence.shape[0],
        ):
            raise ValueError(
                "link_angles/vertex_phases do not match the prepared topology."
            )
        return wrap_u1(links + self.vertex_edge_incidence.T @ phases)

    def local_delta_action(
        self,
        link_angles: ArrayLike,
        edge: ArrayLike,
        proposed_angle: ArrayLike,
        /,
    ) -> Array:
        links = jnp.asarray(link_angles)
        edge_ = jnp.asarray(edge, dtype=jnp.int32)
        proposed = wrap_u1(jnp.asarray(proposed_angle, dtype=links.dtype))
        if links.shape != (self.num_edges,) or edge_.shape != () or proposed.shape != ():
            raise ValueError(
                "local update requires link vector and scalar edge/proposed angle."
            )
        incidence = self.plaquette_edge_incidence[:, edge_]
        old_plaquettes = self.plaquette_angles(links)
        difference = proposed - links[edge_]
        new_plaquettes = wrap_u1(old_plaquettes + incidence * difference)
        active = incidence != 0
        return -self.beta * jnp.sum(
            jnp.where(active, jnp.cos(new_plaquettes) - jnp.cos(old_plaquettes), 0.0)
        )


class U1GaugeState(StrictModule):
    link_angles: Array
    plaquette_angles: Array
    action: Array
    valid: Array


def initialize_u1_gauge_state(
    measure: CompactU1GaugeMeasure, link_angles: ArrayLike, /
) -> U1GaugeState:
    if not isinstance(measure, CompactU1GaugeMeasure):
        raise TypeError("measure must be CompactU1GaugeMeasure.")
    links = wrap_u1(link_angles)
    plaquettes = measure.plaquette_angles(links)
    return U1GaugeState(
        link_angles=links,
        plaquette_angles=plaquettes,
        action=-measure.beta * jnp.sum(jnp.cos(plaquettes)),
        valid=measure.valid & jnp.all(jnp.isfinite(links)),
    )


def wilson_loop(
    link_angles: ArrayLike, oriented_edge_coefficients: ArrayLike, /
) -> Array:
    links = jnp.asarray(link_angles)
    loop = jnp.asarray(oriented_edge_coefficients)
    if links.ndim != 1 or loop.shape != links.shape:
        raise ValueError("Wilson loop coefficients must match the rank-one link vector.")
    if not jnp.issubdtype(loop.dtype, jnp.integer):
        raise TypeError("Wilson loop coefficients must be integer oriented counts.")
    return jnp.exp(1j * jnp.sum(loop * links))


__all__ = [
    "CompactU1GaugeMeasure",
    "U1GaugeState",
    "initialize_u1_gauge_state",
    "wilson_loop",
    "wrap_u1",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ChargedGaugeState(StrictModule):
    scalar: Array
    vector_potential: Array
    plan_id: str = eqx.field(static=True)


class ChargedGaugeEvidence(StrictModule):
    scalar_covariance_residual: Array
    curvature_invariance_residual: Array
    norm_invariance_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ChargedScalarGaugePlan(StrictModule, NonTrainableState):
    edges: Array
    edge_lengths: Array
    face_edge_incidence: Array
    coupling: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        faces: ArrayLike,
        edges: ArrayLike,
        /,
        *,
        coupling: float,
    ):
        vertices_ = np.asarray(vertices, dtype=np.float64)
        faces_ = np.asarray(faces, dtype=np.int32)
        edges_ = np.asarray(edges, dtype=np.int32)
        coupling_ = float(coupling)
        if (
            vertices_.ndim != 2
            or vertices_.shape[1] != 3
            or faces_.ndim != 2
            or faces_.shape[1] != 3
            or edges_.ndim != 2
            or edges_.shape[1] != 2
            or np.any(~np.isfinite(vertices_))
            or not isfinite(coupling_)
            or coupling_ == 0.0
        ):
            raise ValueError("Charged gauge mesh or coupling is invalid.")
        incidence = np.zeros((faces_.shape[0], edges_.shape[0]), dtype=np.float64)
        lookup = {tuple(edge): index for index, edge in enumerate(edges_)}
        for face_index, face in enumerate(faces_):
            for first, second in zip(face, np.roll(face, -1), strict=True):
                canonical = tuple(sorted((int(first), int(second))))
                edge_index = lookup[canonical]
                incidence[face_index, edge_index] = 1.0 if first < second else -1.0
        lengths = np.linalg.norm(
            vertices_[edges_[:, 1]] - vertices_[edges_[:, 0]], axis=-1
        )
        if np.any(lengths <= 0.0):
            raise ValueError("Charged gauge edges must have positive length.")
        self.edges = jnp.asarray(edges_)
        self.edge_lengths = jnp.asarray(lengths)
        self.face_edge_incidence = jnp.asarray(incidence)
        self.coupling = coupling_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charged-scalar-local-u1-gauge",
                "vertices": array_tree_fingerprint(vertices_),
                "faces": array_tree_fingerprint(faces_),
                "edges": array_tree_fingerprint(edges_),
                "coupling": coupling_,
            }
        )

    def state(
        self, scalar: ArrayLike, vector_potential: ArrayLike, /
    ) -> ChargedGaugeState:
        scalar_ = jnp.asarray(scalar)
        potential = jnp.asarray(vector_potential, dtype=jnp.real(scalar_).dtype)
        vertex_count = int(jnp.max(self.edges)) + 1
        if scalar_.shape != (vertex_count,) or potential.shape != (self.edges.shape[0],):
            raise ValueError("Charged scalar or vector-potential shape is invalid.")
        if not jnp.iscomplexobj(scalar_):
            raise TypeError("Charged scalar field must be complex-valued.")
        return ChargedGaugeState(scalar_, potential, self.plan_id)

    def links(self, state: ChargedGaugeState, /) -> Array:
        if state.plan_id != self.plan_id:
            raise ValueError("Charged gauge state belongs to another plan.")
        return jnp.exp(-1.0j * self.coupling * state.vector_potential)

    def covariant_difference(self, state: ChargedGaugeState, /) -> Array:
        link = self.links(state)
        first, second = self.edges[:, 0], self.edges[:, 1]
        return (link * state.scalar[second] - state.scalar[first]) / self.edge_lengths

    def curvature(self, state: ChargedGaugeState, /) -> Array:
        return self.face_edge_incidence @ state.vector_potential

    def transform(
        self, state: ChargedGaugeState, parameter: ArrayLike, /
    ) -> ChargedGaugeState:
        chi = jnp.asarray(parameter, dtype=state.vector_potential.dtype)
        if chi.shape != state.scalar.shape:
            raise ValueError("Gauge parameter must match charged scalar vertices.")
        first, second = self.edges[:, 0], self.edges[:, 1]
        scalar = jnp.exp(1.0j * self.coupling * chi) * state.scalar
        potential = state.vector_potential + chi[second] - chi[first]
        return ChargedGaugeState(scalar, potential, self.plan_id)

    def validate_transform(
        self, state: ChargedGaugeState, parameter: ArrayLike, /
    ) -> ChargedGaugeEvidence:
        transformed = self.transform(state, parameter)
        chi = jnp.asarray(parameter, dtype=state.vector_potential.dtype)
        phase = jnp.exp(1.0j * self.coupling * chi[self.edges[:, 0]])
        covariance = jnp.max(
            jnp.abs(
                self.covariant_difference(transformed)
                - phase * self.covariant_difference(state)
            ),
            initial=0.0,
        )
        curvature = jnp.max(
            jnp.abs(self.curvature(transformed) - self.curvature(state)), initial=0.0
        )
        norm = jnp.max(
            jnp.abs(jnp.abs(transformed.scalar) - jnp.abs(state.scalar)), initial=0.0
        )
        finite = jnp.isfinite(covariance) & jnp.isfinite(curvature) & jnp.isfinite(norm)
        tolerance = 256.0 * jnp.finfo(state.vector_potential.dtype).eps
        successful = (
            finite
            & (covariance <= tolerance)
            & (curvature <= tolerance)
            & (norm <= tolerance)
        )
        return ChargedGaugeEvidence(
            covariance, curvature, norm, finite, successful, self.plan_id
        )


__all__ = [
    "ChargedGaugeEvidence",
    "ChargedGaugeState",
    "ChargedScalarGaugePlan",
]

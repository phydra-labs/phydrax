#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, DenseLinearOperator
from .._spaces import EntityDofLayout
from ._surface_complex import OrientedTriangleSurfaceComplex3D


class TangentialTracePairing3D(StrictModule, NonTrainableState):
    """Metadata for the implemented RWG tangential-current weak pairing."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    trace_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    bilinear_pairing: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)


class RWGSurfaceCurrentSpace3D(StrictModule, NonTrainableState):
    """One oriented Rao-Wilton-Glisson surface-current DOF per mesh edge."""

    surface: OrientedTriangleSurfaceComplex3D
    layout: EntityDofLayout
    vector_space: ArraySpace
    centroid_basis: Array
    divergence_matrix: Array
    divergence_operator: DenseLinearOperator
    trace_pairing: TangentialTracePairing3D
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: OrientedTriangleSurfaceComplex3D,
        /,
        *,
        coefficient_dtype: Any = np.complex128,
    ):
        if not isinstance(surface, OrientedTriangleSurfaceComplex3D):
            raise TypeError("surface must be OrientedTriangleSurfaceComplex3D.")
        dtype = np.dtype(coefficient_dtype)
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError("RWG Maxwell coefficients require a native complex dtype.")
        dtype = np.dtype(jax.dtypes.canonicalize_dtype(dtype))
        edges = surface.topology.entities(1)
        layout = EntityDofLayout(
            edges.entity_set_id,
            surface.edge_count,
            surface.edge_count,
        )
        local_edges = surface.face_edges
        lengths = surface.edge_lengths[local_edges]
        signs = surface.face_edge_signs
        opposite = surface.vertices[surface.opposite_vertices]
        scale = signs * lengths / (2.0 * surface.face_areas[:, None])
        basis = scale[:, :, None] * (surface.face_centroids[:, None, :] - opposite)
        divergence_local = signs * lengths / surface.face_areas[:, None]
        divergence = jnp.zeros(
            (surface.face_count, surface.edge_count), dtype=surface.vertices.dtype
        )
        face_ids = jnp.repeat(jnp.arange(surface.face_count), 3)
        divergence = divergence.at[face_ids, local_edges.reshape(-1)].set(
            divergence_local.reshape(-1)
        )
        space_id = canonical_fingerprint(
            {
                "kind": "rwg-surface-current-space-3d-v1",
                "surface": surface.complex_id,
                "layout": layout.layout_id,
                "coefficient_dtype": dtype.str,
            }
        )
        pairing_id = canonical_fingerprint(
            {
                "kind": "rwg-tangential-trace-pairing-3d-v1",
                "space": space_id,
            }
        )
        coefficient_space = ArraySpace(
            (surface.edge_count,), dtype=dtype, space_id=space_id
        )
        divergence_space = ArraySpace(
            (surface.face_count,),
            dtype=coefficient_space.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "rwg-surface-divergence-range-3d-v1",
                    "surface": surface.complex_id,
                    "coefficient_dtype": coefficient_space.dtype.str,
                }
            ),
        )
        divergence_operator = DenseLinearOperator(
            divergence.astype(coefficient_space.dtype),
            source=coefficient_space,
            target=divergence_space,
        )
        trace_pairing = TangentialTracePairing3D(
            ambient_dimension=3,
            pde="time-harmonic Maxwell electric field integral equation",
            geometry="oriented closed piecewise-planar triangular surface",
            formulation="RWG H(div_Gamma) trial/test functions with unconjugated Galerkin transpose pairing",
            provider="phydrax.discretization.bem",
            precision=dtype.name,
            resource_evidence=f"one complex coefficient per {surface.edge_count} edges",
            error_evidence="exact signed edge assembly; piecewise-linear geometric approximation only",
            non_goals=(
                "BC/RBC dual spaces",
                "Calderon products",
                "continuum trace certification",
            ),
            trace_kind="tangential electric surface current",
            conformity="H(div_Gamma), single-valued signed edge-normal trace",
            bilinear_pairing="integral test_dot_field without test conjugation; Hermitian adjoint is separate",
            pairing_id=pairing_id,
        )
        self.surface = surface
        self.layout = layout
        self.vector_space = coefficient_space
        self.centroid_basis = basis
        self.divergence_matrix = divergence
        self.divergence_operator = divergence_operator
        self.trace_pairing = trace_pairing
        self.space_id = space_id

    @property
    def size(self) -> int:
        return self.surface.edge_count

    def validate(self, coefficients: ArrayLike, /) -> Array:
        return self.vector_space.validate(coefficients)

    def local_basis(self, points: ArrayLike, /) -> Array:
        """Evaluate the three incident RWG pieces at one point per triangle."""
        values = jnp.asarray(points, dtype=self.surface.vertices.dtype)
        if values.shape != (self.surface.face_count, 3):
            raise ValueError(
                f"points must have shape {(self.surface.face_count, 3)}; got {values.shape}."
            )
        opposite = self.surface.vertices[self.surface.opposite_vertices]
        lengths = self.surface.edge_lengths[self.surface.face_edges]
        scale = (
            self.surface.face_edge_signs
            * lengths
            / (2.0 * self.surface.face_areas[:, None])
        )
        return scale[:, :, None] * (values[:, None, :] - opposite)

    def current_at_centroids(self, coefficients: ArrayLike, /) -> Array:
        values = self.validate(coefficients)
        local = values[self.surface.face_edges]
        return jnp.sum(self.centroid_basis * local[:, :, None], axis=1)

    def surface_divergence(self, coefficients: ArrayLike, /) -> Array:
        return self.divergence_operator.mv(coefficients)

    def tangential_conformity_defect(self, /) -> Array:
        """Return the maximum signed co-normal trace jump of all RWG basis pieces."""
        defects = []
        for edge_id in range(self.surface.edge_count):
            locations = np.argwhere(np.asarray(self.surface.face_edges) == edge_id)
            traces = []
            start, stop = self.surface.edge_vertices[edge_id]
            tangent = self.surface.vertices[stop] - self.surface.vertices[start]
            tangent = tangent / jnp.linalg.norm(tangent)
            midpoint = 0.5 * (self.surface.vertices[start] + self.surface.vertices[stop])
            for face_id_host, local_id_host in locations:
                face_id, local_id = int(face_id_host), int(local_id_host)
                sign = self.surface.face_edge_signs[face_id, local_id]
                boundary_tangent = sign * tangent
                outward_conormal = jnp.cross(
                    boundary_tangent, self.surface.face_normals[face_id]
                )
                opposite = self.surface.vertices[
                    self.surface.opposite_vertices[face_id, local_id]
                ]
                value = (
                    sign
                    * self.surface.edge_lengths[edge_id]
                    / (2.0 * self.surface.face_areas[face_id])
                    * (midpoint - opposite)
                )
                traces.append(jnp.dot(value, outward_conormal))
            defects.append(jnp.abs(traces[0] + traces[1]))
        return jnp.max(jnp.stack(defects))

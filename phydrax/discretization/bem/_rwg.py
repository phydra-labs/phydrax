#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import lru_cache
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...exterior._algebra import map_reference_values
from ...exterior._form_type import FormType, FormValueSpec
from ...linalg import ArraySpace
from ...sparse import EdgeRelation, SparseCoordinateOperator
from .._boundary_trace_space import (
    boundary_geometry_revision,
    BoundaryTraceSpaceCapability,
)
from .._gram import sparse_gram_space
from .._spaces import EntityDofLayout
from ..fem._form_elements import form_element
from ..fem._reference import FiniteElementSpec
from ._surface_complex import OrientedTriangleSurfaceComplex3D


@lru_cache(maxsize=1)
def _rwg_element() -> FiniteElementSpec:
    return form_element("triangle", 1, 1, twist="twisted", proxy="flux")


def _reference_edge_dofs(element: FiniteElementSpec, /) -> Array:
    basis = element.form_basis
    if basis is None:
        raise ValueError("RWG requires a canonical polynomial form basis.")
    cyclic_edges = ((0, 1), (1, 2), (0, 2))
    dofs: list[int] = []
    for edge in cyclic_edges:
        matches = tuple(
            index for index, label in enumerate(basis.dof_labels) if label[0] == edge
        )
        if len(matches) != 1:
            raise ValueError(
                "RWG requires exactly one constant form moment per triangle edge."
            )
        dofs.append(matches[0])
    return jnp.asarray(dofs, dtype=jnp.int32)


def _rwg_tabulate(
    surface: OrientedTriangleSurfaceComplex3D,
    points: Array,
    element: FiniteElementSpec,
    /,
) -> tuple[Array, Array]:
    corners = surface.vertices[surface.triangles]
    jacobians = jnp.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
    offsets = points - corners[:, :1]
    gram = jnp.swapaxes(jacobians, -1, -2) @ jacobians
    rhs = jnp.swapaxes(jacobians, -1, -2) @ jnp.swapaxes(offsets, -1, -2)
    reference = jnp.swapaxes(jnp.linalg.solve(gram, rhs), -1, -2)
    reference_values, gradients = jax.vmap(element.tabulate)(reference)

    def map_basis(values: Array, jacobian: Array) -> Array:
        return map_reference_values(values, element.value_spec, jacobian, coorientation=1)

    mapped = jax.vmap(map_basis)(reference_values, jacobians)
    dofs = _reference_edge_dofs(element)
    lengths = surface.edge_lengths[surface.face_edges]
    scale = surface.face_edge_signs * lengths * jnp.asarray((1.0, 1.0, -1.0))
    basis_values = mapped[:, :, dofs] * scale[:, None, :, None]
    reference_divergence = jnp.trace(gradients, axis1=-2, axis2=-1)[:, :, dofs]
    divergence = (
        reference_divergence
        * scale[:, None, :]
        / (2.0 * surface.face_areas[:, None, None])
    )
    return basis_values, divergence


def rwg_gram_entries(
    surface: OrientedTriangleSurfaceComplex3D, /
) -> tuple[np.ndarray, np.ndarray, Array]:
    """Exact area Gram of the canonical surface flux element in RWG coordinates."""
    corners = surface.vertices[surface.triangles]
    points = 0.5 * (corners + jnp.roll(corners, -1, axis=1))
    basis, _ = _rwg_tabulate(surface, points, _rwg_element())
    local = contract("fqic,fqjc,f->fij", basis, basis, surface.face_areas / 3.0)
    edges = np.asarray(surface.face_edges, dtype=np.int32)
    targets = np.repeat(edges, 3, axis=1).reshape((-1,))
    sources = np.tile(edges, (1, 3)).reshape((-1,))
    return targets, sources, local.reshape((-1,))


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


@final
class RWGSurfaceCurrentSpace3D(StrictModule, NonTrainableState):
    """One oriented Rao-Wilton-Glisson surface-current DOF per mesh edge."""

    surface: OrientedTriangleSurfaceComplex3D
    element: FiniteElementSpec
    layout: EntityDofLayout
    vector_space: ArraySpace
    centroid_basis: Array
    divergence_matrix: Array
    divergence_operator: SparseCoordinateOperator
    trace_pairing: TangentialTracePairing3D
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: OrientedTriangleSurfaceComplex3D,
        /,
        *,
        coefficient_dtype: DTypeLike = np.complex128,
    ) -> None:
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
        element = _rwg_element()
        basis_values, divergence_values = _rwg_tabulate(
            surface, surface.face_centroids[:, None], element
        )
        basis = basis_values[:, 0]
        divergence_local = divergence_values[:, 0]
        divergence = jnp.zeros(
            (surface.face_count, surface.edge_count), dtype=surface.vertices.dtype
        )
        face_ids = jnp.repeat(jnp.arange(surface.face_count), 3)
        divergence = divergence.at[face_ids, local_edges.reshape(-1)].set(
            divergence_local.reshape(-1)
        )
        space_id = canonical_fingerprint(
            {
                "kind": "rwg-surface-current-space-3d",
                "surface": surface.complex_id,
                "layout": layout.layout_id,
                "coefficient_dtype": dtype.str,
            }
        )
        pairing_id = canonical_fingerprint(
            {
                "kind": "rwg-tangential-trace-pairing-3d",
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
                    "kind": "rwg-surface-divergence-range-3d",
                    "surface": surface.complex_id,
                    "coefficient_dtype": coefficient_space.dtype.str,
                }
            ),
        )
        divergence_operator = SparseCoordinateOperator(
            EdgeRelation(
                np.asarray(local_edges).reshape((-1,)),
                np.repeat(np.arange(surface.face_count, dtype=np.int32), 3),
                source_size=surface.edge_count,
                target_size=surface.face_count,
            ),
            divergence_local.reshape((-1,)).astype(coefficient_space.dtype),
            source=coefficient_space,
            target=divergence_space,
            operator_id=f"{space_id}:surface-divergence",
            accumulation_dtype=coefficient_space.dtype,
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
                "Calderon products",
                "continuum trace certification",
            ),
            trace_kind="tangential electric surface current",
            conformity="H(div_Gamma), single-valued signed edge-normal trace",
            bilinear_pairing="integral test_dot_field without test conjugation; Hermitian adjoint is separate",
            pairing_id=pairing_id,
        )
        self.surface = surface
        self.element = element
        self.layout = layout
        self.vector_space = coefficient_space
        self.centroid_basis = basis
        self.divergence_matrix = divergence
        self.divergence_operator = divergence_operator
        self.trace_pairing = trace_pairing
        self.space_id = space_id

    @property
    def value_spec(self) -> FormValueSpec:
        """Twisted intrinsic surface flux with an embedded Cartesian proxy."""
        return FormValueSpec(
            FormType(2, 1, twist="twisted", ambient_dimension=3), proxy="flux"
        )

    @property
    def form_type(self) -> FormType:
        return self.value_spec.form_type

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
        basis, _ = _rwg_tabulate(self.surface, values[:, None], self.element)
        return basis[:, 0]

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

    def trace_capability(
        self, /, *, gram_tolerance: float = 1.0e-13, numeric_revision: str | None = None
    ) -> BoundaryTraceSpaceCapability:
        """Publish the RWG surface-current trace space with its area Gram pairing.

        The capability's `gram_space` pairs the native RWG coordinates by
        `∫ conj(j)·k dA` through the exact RWG Gram map; its inverse is a
        prepared conjugate-gradient solve to `gram_tolerance`. RWG currents are
        an H(div_Γ) representation and never a scalar Cauchy trace.
        Supply a stable ``numeric_revision`` binding for differentiable numeric
        refreshes; omitted revisions fingerprint an eager geometry admission.
        """
        targets, sources, values = rwg_gram_entries(self.surface)
        gram_space, mass = sparse_gram_space(
            targets,
            sources,
            values,
            size=self.size,
            dtype=self.vector_space.dtype,
            space_id=canonical_fingerprint(
                {"kind": "rwg-surface-current-trace-space-3d", "space": self.space_id}
            ),
            gram_tolerance=gram_tolerance,
        )
        return BoundaryTraceSpaceCapability(
            owner_id=self.space_id,
            quantity="surface-current",
            representation="rwg",
            coefficient_space=self.vector_space,
            gram_space=gram_space,
            mass=mass,
            ambient_dimension=3,
            revision_id=(
                boundary_geometry_revision(self.surface.vertices, self.surface.triangles)
                if numeric_revision is None
                else canonical_identifier(numeric_revision, "numeric_revision")
            ),
        )

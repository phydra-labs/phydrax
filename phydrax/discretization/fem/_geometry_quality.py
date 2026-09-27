#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_geometry import CellGeometrySpec
from .._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ._generic import (
    _degree_aware_reference_rule,
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
)


class FiniteElementGeometryQualityEvidence(StrictModule, NonTrainableState):
    """Certified geometric validity plus sampled conditioning of FE coordinates.

    ``valid_cells`` and ``minimum_jacobian`` (the certified determinant lower
    bound) come from the Bernstein validity certificate; the scaled Jacobian
    and condition number are quality samples at reference quadrature points.
    """

    minimum_jacobian: Array
    minimum_scaled_jacobian: Array
    maximum_condition_number: Array
    valid_cells: Array
    certificate: CellValidityCertificate
    maximum_face_coordinate_defect: Array
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def passed(self) -> Array:
        return jnp.all(self.valid_cells)


def finite_element_geometry_quality(
    discretization: FiniteElementDiscretization,
    runtime: FiniteElementRuntimeData | None = None,
    /,
    *,
    probe_degree_increment: int = 2,
    policy: CellValidityPolicy | None = None,
) -> FiniteElementGeometryQualityEvidence:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    runtime_ = discretization.default_runtime if runtime is None else runtime
    if not isinstance(runtime_, FiniteElementRuntimeData):
        raise TypeError("runtime must be FiniteElementRuntimeData or None.")
    increment = int(probe_degree_increment)
    if increment < 0:
        raise ValueError("probe_degree_increment must be non-negative.")
    mesh = discretization.mesh
    geometry = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(
                mesh.blocks, discretization.coordinate_elements, strict=True
            )
        },
        {
            block.name: routes
            for block, routes in zip(
                mesh.blocks, discretization.coordinate_dofs, strict=True
            )
        },
        runtime_.coordinates,
    )
    certificate = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
    tiny = jnp.finfo(runtime_.coordinates.dtype).tiny
    minimum_scaled = []
    maximum_conditions = []
    for block_index, block in enumerate(mesh.blocks):
        coordinate_element = discretization.coordinate_elements[block_index]
        # ty: ignore[unresolved-attribute]
        degree = max(coordinate_element.degree + increment, 2)
        points, _weights = _degree_aware_reference_rule(block.cell_kind, degree)
        # ty: ignore[unresolved-attribute]
        _basis, gradients = coordinate_element.tabulate(points)
        local_coordinates = runtime_.coordinates[
            discretization.coordinate_dofs[block_index]
        ]
        jacobian = ein.contract(
            "qid,cia->cqad", gradients, local_coordinates, backend="jax"
        )
        # Singular values of the tiny per-point Jacobians are the sampled
        # conditioning evidence; validity itself is owned by the certificate.
        singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
        column_norms = jnp.linalg.norm(jacobian, axis=-2)
        volume = jnp.prod(singular_values, axis=-1)
        scaled = volume / jnp.maximum(jnp.prod(column_norms, axis=-1), tiny)
        condition = singular_values[..., 0] / jnp.maximum(singular_values[..., -1], tiny)
        minimum_scaled.append(jnp.min(scaled, axis=1))
        maximum_conditions.append(jnp.max(condition, axis=1))
    valid = np.asarray(certificate.status) == CellValidityStatus.CERTIFIED_VALID
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-element-geometry-quality",
            "topology": mesh.topology_id,
            "geometry": mesh.geometry_id,
            "runtime": runtime_.runtime_id,
            "probe_degree_increment": increment,
            "certificate": certificate.certificate_id,
        }
    )
    return FiniteElementGeometryQualityEvidence(
        certificate.determinant_lower,
        jnp.concatenate(minimum_scaled),
        jnp.concatenate(maximum_conditions),
        jnp.asarray(valid),
        certificate,
        jnp.asarray(0.0),
        runtime_.runtime_id,
        evidence_id,
    )


__all__ = [
    "FiniteElementGeometryQualityEvidence",
    "finite_element_geometry_quality",
]

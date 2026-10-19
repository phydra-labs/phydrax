#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify chart endpoints or authenticate their retained geometry proofs."""

from .._strict import StrictModule
from ..geometry._embedding_state import MeshCertificateLimits
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..typing import checked
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget


class ChartEndpointCertificates(StrictModule):
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate


@checked
def certify_chart_endpoint(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    policy: CellValidityPolicy,
    limits: MeshCertificateLimits,
    prepared: tuple[CellValidityCertificate, GlobalEmbeddingCertificate] | None,
    *,
    budget: CoordinateEnclosureBudget,
    stale_message: str,
) -> ChartEndpointCertificates:
    """Return validity and embedding proofs bound to the declared controls.

    Status interpretation stays with the family so its certification and measure
    refusal ordering is preserved.
    """
    if prepared is None:
        validity = certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=policy, budget=budget
        )
        embedding = certify_global_embedding(
            mesh, geometry, validity, limits=limits, budget=budget
        )
    else:
        validity, embedding = prepared
        validity.require_bound(geometry, mesh=mesh)
        embedding.binding.require(mesh, geometry)
        if (
            validity.policy_id != policy.policy_id
            or embedding.binding.limits_id != limits.limits_id
        ):
            raise ValueError(stale_message)
    return ChartEndpointCertificates(validity=validity, embedding=embedding)

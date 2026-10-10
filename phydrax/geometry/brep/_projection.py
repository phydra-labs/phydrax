#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native projection preparation bound to an exact source revision."""

from __future__ import annotations

from .._planar_embedding import PlanarEmbedding
from ._model import BRepModel
from ._projection_contracts import BRepProjectionPolicy
from ._query import BRepQueryPolicy, NativeBRepProjection


def prepare_brep_projection(
    model: BRepModel,
    /,
    *,
    policy: BRepProjectionPolicy | None = None,
    embedding: PlanarEmbedding | None = None,
    query_policy: BRepQueryPolicy | None = None,
) -> NativeBRepProjection:
    """Prepare exact native projection; external comparison is explicit interchange."""
    if not isinstance(model, BRepModel):
        raise TypeError("model must be BRepModel.")
    if model.geometry is None:
        raise ValueError("Native projection requires exact native B-Rep geometry.")
    policy_ = BRepProjectionPolicy() if policy is None else policy
    query_policy_ = BRepQueryPolicy() if query_policy is None else query_policy
    if not isinstance(policy_, BRepProjectionPolicy):
        raise TypeError("policy must be BRepProjectionPolicy or None.")
    if not isinstance(query_policy_, BRepQueryPolicy):
        raise TypeError("query_policy must be BRepQueryPolicy or None.")
    return NativeBRepProjection(model, policy_, embedding, query_policy_)


__all__ = ["prepare_brep_projection"]

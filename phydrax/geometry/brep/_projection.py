#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare native projection bound to a B-Rep source revision."""

from __future__ import annotations

from ...typing import checked
from .._planar_embedding import PlanarEmbedding
from ._model import BRepModel
from ._projection_contracts import BRepProjectionPolicy
from ._query import BRepQueryPolicy
from ._query_projection import NativeBRepProjection


@checked
def prepare_brep_projection(
    model: BRepModel,
    /,
    *,
    policy: BRepProjectionPolicy | None = None,
    embedding: PlanarEmbedding | None = None,
    query_policy: BRepQueryPolicy | None = None,
) -> NativeBRepProjection:
    """Prepare native projection bound to the model's source revision."""
    model.require_geometry()
    policy_ = BRepProjectionPolicy() if policy is None else policy
    query_policy_ = BRepQueryPolicy() if query_policy is None else query_policy
    return NativeBRepProjection(model, policy_, embedding, query_policy_)


__all__ = ["prepare_brep_projection"]

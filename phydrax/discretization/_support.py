#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import nonempty_identifier, resolved_identifier
from ._topology import (
    CellComplexTopology,
    DiscreteTopology,
    PointTopology,
    TensorTopology,
)


class DiscreteSupport(StrictModule, NonTrainableState):
    """A discrete topology bound to one geometric embedding identity."""

    topology: DiscreteTopology
    ambient_dimension: int = eqx.field(static=True)
    embedding_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: DiscreteTopology,
        ambient_dimension: int,
        embedding_id: str,
        /,
        *,
        support_id: str | None = None,
    ):
        if not isinstance(topology, (TensorTopology, CellComplexTopology, PointTopology)):
            raise TypeError("topology must be a supported DiscreteTopology value.")
        dimension = int(ambient_dimension)
        if dimension <= 0:
            raise ValueError("ambient_dimension must be positive.")
        embedding_id_ = nonempty_identifier("embedding_id", embedding_id)
        self.topology = topology
        self.ambient_dimension = dimension
        self.embedding_id = embedding_id_
        self.support_id = resolved_identifier(
            "support_id",
            support_id,
            {
                "kind": "discrete-support",
                "topology": topology.topology_id,
                "ambient_dimension": dimension,
                "embedding": embedding_id_,
            },
        )


__all__ = ["DiscreteSupport"]

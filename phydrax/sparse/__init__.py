#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""JAX-native sparse relations, routing kernels, and linear actions."""

from ._linear import LinearAction, SparseLinearMap
from ._ops import (
    gather_routes,
    linear_adjoint_apply,
    linear_apply,
    linear_transpose_apply,
    mask_routes,
    route_reduce,
    RouteReduction,
)
from ._relation import EdgeRelation, RowRelation, SparseRelation


__all__ = [
    "EdgeRelation",
    "LinearAction",
    "RouteReduction",
    "RowRelation",
    "SparseLinearMap",
    "SparseRelation",
    "gather_routes",
    "linear_adjoint_apply",
    "linear_apply",
    "linear_transpose_apply",
    "mask_routes",
    "route_reduce",
]

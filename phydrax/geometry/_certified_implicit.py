#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._topology import CellComplexTopology


class CertifiedImplicitCover(StrictModule, NonTrainableState):
    """Validated boxwise sign and regular-value evidence for an implicit set."""

    boxes: Array
    value_lower: Array
    value_upper: Array
    gradient_norm_lower: Array
    intersects: Array
    excluded: Array
    regular: Array
    certified: Array
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        boxes: ArrayLike,
        value_lower: ArrayLike,
        value_upper: ArrayLike,
        gradient_norm_lower: ArrayLike,
        /,
    ):
        boxes_ = jnp.asarray(boxes)
        lower = jnp.asarray(value_lower)
        upper = jnp.asarray(value_upper)
        gradient = jnp.asarray(gradient_norm_lower)
        if boxes_.ndim != 3 or boxes_.shape[1] != 2:
            raise ValueError("Implicit cover boxes require shape (box, 2, dimension).")
        if (
            lower.shape != (boxes_.shape[0],)
            or upper.shape != lower.shape
            or gradient.shape != lower.shape
        ):
            raise ValueError("Implicit cover bound vectors do not match the boxes.")
        if not bool(jnp.all(jnp.isfinite(boxes_))) or not bool(
            jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper) & jnp.isfinite(gradient))
        ):
            raise ValueError("Implicit cover bounds must be finite.")
        if not bool(jnp.all(boxes_[:, 0] < boxes_[:, 1])):
            raise ValueError("Implicit cover boxes require strict lower/upper bounds.")
        if not bool(jnp.all(lower <= upper)) or bool(jnp.any(gradient < 0)):
            raise ValueError("Implicit value/gradient bounds are inconsistent.")
        intersects = (lower <= 0) & (upper >= 0)
        excluded = ~intersects
        regular = excluded | (gradient > 0)
        self.boxes = boxes_
        self.value_lower = lower
        self.value_upper = upper
        self.gradient_norm_lower = gradient
        self.intersects = intersects
        self.excluded = excluded
        self.regular = regular
        self.certified = jnp.all(regular)
        self.cover_id = canonical_fingerprint(
            {
                "kind": "certified-implicit-cover",
                "boxes": array_tree_fingerprint(boxes_),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
                "gradient": array_tree_fingerprint(gradient),
            }
        )


class CertifiedImplicitTopology(StrictModule, NonTrainableState):
    """Finite topology bound to separately supplied implicit-geometry evidence."""

    cover: CertifiedImplicitCover
    topology: CellComplexTopology
    theorem: str = eqx.field(static=True)
    certified: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        cover: CertifiedImplicitCover,
        topology: CellComplexTopology,
        /,
        *,
        theorem: str,
    ):
        theorem_ = str(theorem)
        if not theorem_:
            raise ValueError("Implicit topology requires an explicit theorem identifier.")
        self.cover = cover
        self.topology = topology
        self.theorem = theorem_
        self.certified = cover.certified
        self.result_id = canonical_fingerprint(
            {
                "kind": "certified-implicit-topology",
                "cover": cover.cover_id,
                "topology": topology.topology_id,
                "theorem": theorem_,
                "certified": bool(cover.certified),
            }
        )


__all__ = ["CertifiedImplicitCover", "CertifiedImplicitTopology"]

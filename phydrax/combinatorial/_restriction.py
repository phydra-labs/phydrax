#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._problem import AbstractCombinatorialSpace
from ._types import CombinatorialResult


class AbstractBoundableCombinatorialSpace(AbstractCombinatorialSpace):
    """Combinatorial space with compact feature bounds and exact restrictions."""

    __strict_abstract__ = True

    @abc.abstractmethod
    def feature_bounds(self, /) -> tuple[PyTree[Array], PyTree[Array]]:
        raise NotImplementedError

    @abc.abstractmethod
    def integral_feature_mask(self, /) -> PyTree[Array]:
        raise NotImplementedError


class CombinatorialFeatureRestriction(StrictModule, NonTrainableState):
    """Immutable feature-coordinate bounds for one boundable space."""

    lower: PyTree[Array]
    upper: PyTree[Array]
    space_id: str = eqx.field(static=True)
    restriction_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: AbstractBoundableCombinatorialSpace,
        /,
        *,
        lower: Any | None = None,
        upper: Any | None = None,
    ):
        if not isinstance(space, AbstractBoundableCombinatorialSpace):
            raise TypeError("space must be an AbstractBoundableCombinatorialSpace.")
        specification = space.feature_spec()
        spec_leaves, treedef = jax.tree_util.tree_flatten(specification)
        global_lower, global_upper = space.feature_bounds()
        lower_ = global_lower if lower is None else lower
        upper_ = global_upper if upper is None else upper
        lower_leaves, lower_tree = jax.tree_util.tree_flatten(lower_)
        upper_leaves, upper_tree = jax.tree_util.tree_flatten(upper_)
        global_lower_leaves, global_lower_tree = jax.tree_util.tree_flatten(global_lower)
        global_upper_leaves, global_upper_tree = jax.tree_util.tree_flatten(global_upper)
        mask_leaves, mask_tree = jax.tree_util.tree_flatten(space.integral_feature_mask())
        if not (
            lower_tree
            == upper_tree
            == global_lower_tree
            == global_upper_tree
            == mask_tree
            == treedef
        ):
            raise ValueError("Restriction trees must exactly match feature_spec().")
        normalized_lower = []
        normalized_upper = []
        for spec, lo, hi, root_lo, root_hi, integral in zip(
            spec_leaves,
            lower_leaves,
            upper_leaves,
            global_lower_leaves,
            global_upper_leaves,
            mask_leaves,
            strict=True,
        ):
            if not isinstance(spec, jax.ShapeDtypeStruct):
                raise TypeError("feature_spec leaves must be ShapeDtypeStruct values.")
            dtype = jnp.dtype(spec.dtype)
            lo_ = jnp.asarray(lo, dtype=dtype)
            hi_ = jnp.asarray(hi, dtype=dtype)
            root_lo_ = jnp.asarray(root_lo, dtype=dtype)
            root_hi_ = jnp.asarray(root_hi, dtype=dtype)
            integral_ = jnp.asarray(integral, dtype=bool)
            if any(
                value.shape != spec.shape
                for value in (lo_, hi_, root_lo_, root_hi_, integral_)
            ):
                raise ValueError("Restriction leaf shapes must match feature_spec().")
            host = tuple(np.asarray(value) for value in (lo_, hi_, root_lo_, root_hi_))
            if not all(np.all(np.isfinite(value)) for value in host):
                raise ValueError("Boundable feature bounds must be finite.")
            if np.any(host[0] > host[1]):
                raise ValueError("Restriction lower bounds cannot exceed upper bounds.")
            if np.any(host[0] < host[2]) or np.any(host[1] > host[3]):
                raise ValueError("Restriction bounds must remain inside root bounds.")
            integral_host = np.asarray(integral_)
            if np.any(
                host[0][integral_host] != np.rint(host[0][integral_host])
            ) or np.any(host[1][integral_host] != np.rint(host[1][integral_host])):
                raise ValueError("Integral feature restrictions require integral bounds.")
            normalized_lower.append(lo_)
            normalized_upper.append(hi_)
        lower_tree_value = jax.tree_util.tree_unflatten(treedef, normalized_lower)
        upper_tree_value = jax.tree_util.tree_unflatten(treedef, normalized_upper)
        self.lower = lower_tree_value
        self.upper = upper_tree_value
        self.space_id = space.structure_id
        self.restriction_id = canonical_fingerprint(
            {
                "kind": "combinatorial-feature-restriction",
                "space": space.structure_id,
                "lower": array_tree_fingerprint(lower_tree_value),
                "upper": array_tree_fingerprint(upper_tree_value),
            }
        )

    @classmethod
    def root(
        cls,
        space: AbstractBoundableCombinatorialSpace,
        /,
    ) -> CombinatorialFeatureRestriction:
        return cls(space)


class BoundedCombinatorialExecution(StrictModule, NonTrainableState):
    """Restricted solve result plus an independent feature-bound audit."""

    result: CombinatorialResult
    restriction: CombinatorialFeatureRestriction
    restriction_violation: Array
    valid: Array


def audit_combinatorial_restriction(
    space: AbstractBoundableCombinatorialSpace,
    features: Any,
    restriction: CombinatorialFeatureRestriction,
    /,
    *,
    tolerance: float,
) -> Array:
    if not isinstance(space, AbstractBoundableCombinatorialSpace):
        raise TypeError("space must be an AbstractBoundableCombinatorialSpace.")
    if not isinstance(restriction, CombinatorialFeatureRestriction):
        raise TypeError("restriction must be a CombinatorialFeatureRestriction.")
    if restriction.space_id != space.structure_id:
        raise ValueError("Restriction does not belong to this combinatorial space.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    specification = space.feature_spec()
    feature_leaves, feature_tree = jax.tree_util.tree_flatten(features)
    spec_leaves, spec_tree = jax.tree_util.tree_flatten(specification)
    lower_leaves, lower_tree = jax.tree_util.tree_flatten(restriction.lower)
    upper_leaves, upper_tree = jax.tree_util.tree_flatten(restriction.upper)
    if not feature_tree == spec_tree == lower_tree == upper_tree:
        raise ValueError("Feature and restriction trees do not match feature_spec().")
    violations = []
    for value, spec, lower, upper in zip(
        feature_leaves,
        spec_leaves,
        lower_leaves,
        upper_leaves,
        strict=True,
    ):
        array = jnp.asarray(value)
        if tuple(array.shape[-len(spec.shape) :]) != tuple(spec.shape):
            raise ValueError("Feature leaf trailing shape does not match feature_spec().")
        residual = jnp.maximum(
            jnp.maximum(jnp.asarray(lower) - array, array - jnp.asarray(upper)),
            0.0,
        )
        axes = tuple(range(array.ndim - len(spec.shape), array.ndim))
        violations.append(jnp.max(residual, axis=axes, initial=0.0))
    maximum = jnp.asarray(0.0)
    for violation in violations:
        maximum = jnp.maximum(maximum, violation)
    return maximum


__all__ = [
    "AbstractBoundableCombinatorialSpace",
    "BoundedCombinatorialExecution",
    "CombinatorialFeatureRestriction",
    "audit_combinatorial_restriction",
]

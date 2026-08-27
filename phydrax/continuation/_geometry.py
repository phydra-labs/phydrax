#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, validate_inexact_tree
from ..linalg import AbstractRealCoordinateMap, AbstractVectorSpace, PyTreeSpace


def _same_structure(left: AbstractVectorSpace, right: AbstractVectorSpace, /) -> bool:
    left_structure = left.structure()
    right_structure = right.structure()
    if jax.tree.structure(left_structure) != jax.tree.structure(right_structure):
        return False
    return all(
        left_spec.shape == right_spec.shape
        and np.dtype(left_spec.dtype) == np.dtype(right_spec.dtype)
        for left_spec, right_spec in zip(
            jax.tree.leaves(left_structure),
            jax.tree.leaves(right_structure),
            strict=True,
        )
    )


def _coordinate_dtype(space: AbstractVectorSpace, /) -> np.dtype:
    value = space.flatten(space.zeros())
    return np.dtype(value.dtype)


def _real_space(space: AbstractVectorSpace, /, *, name: str) -> None:
    dtype = _coordinate_dtype(space)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"{name} must use real floating-point execution coordinates.")


class ContinuationRepresentationPolicy(StrictModule, NonTrainableState):
    """Public-to-real coordinate maps and execution-space overrides."""

    state_coordinates: AbstractRealCoordinateMap | None
    residual_coordinates: AbstractRealCoordinateMap | None
    state_execution_space: AbstractVectorSpace | None
    residual_execution_space: AbstractVectorSpace | None
    defect_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_coordinates: AbstractRealCoordinateMap | None = None,
        residual_coordinates: AbstractRealCoordinateMap | None = None,
        state_execution_space: AbstractVectorSpace | None = None,
        residual_execution_space: AbstractVectorSpace | None = None,
        defect_tolerance: float = 1e-10,
        policy_id: str | None = None,
    ):
        for value, name in (
            (state_coordinates, "state_coordinates"),
            (residual_coordinates, "residual_coordinates"),
        ):
            if value is not None and not isinstance(value, AbstractRealCoordinateMap):
                raise TypeError(f"{name} must be an AbstractRealCoordinateMap or None.")
        for value, name in (
            (state_execution_space, "state_execution_space"),
            (residual_execution_space, "residual_execution_space"),
        ):
            if value is not None and not isinstance(value, AbstractVectorSpace):
                raise TypeError(f"{name} must be an AbstractVectorSpace or None.")
        tolerance = float(defect_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("defect_tolerance must be finite and non-negative.")
        if (
            state_coordinates is not None
            and state_execution_space is not None
            and not _same_structure(
                state_coordinates.coordinate_space, state_execution_space
            )
        ):
            raise ValueError(
                "state_execution_space must match the state coordinate-map structure."
            )
        if (
            residual_coordinates is not None
            and residual_execution_space is not None
            and not _same_structure(
                residual_coordinates.coordinate_space, residual_execution_space
            )
        ):
            raise ValueError(
                "residual_execution_space must match the residual coordinate-map "
                "structure."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "continuation-representation-v1",
                    "state_coordinates": (
                        "identity"
                        if state_coordinates is None
                        else state_coordinates.evidence.evidence_id
                    ),
                    "residual_coordinates": (
                        "identity"
                        if residual_coordinates is None
                        else residual_coordinates.evidence.evidence_id
                    ),
                    "state_execution_space": (
                        "inferred"
                        if state_execution_space is None
                        else state_execution_space.space_id
                    ),
                    "residual_execution_space": (
                        "inferred"
                        if residual_execution_space is None
                        else residual_execution_space.space_id
                    ),
                    "defect_tolerance": tolerance,
                }
            )
            if policy_id is None
            else str(policy_id)
        )
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        self.state_coordinates = state_coordinates
        self.residual_coordinates = residual_coordinates
        self.state_execution_space = state_execution_space
        self.residual_execution_space = residual_execution_space
        self.defect_tolerance = tolerance
        self.policy_id = identifier


class ContinuationGeometry(StrictModule, NonTrainableState):
    """Resolved public spaces, real execution spaces, and arclength metric."""

    public_state_space: AbstractVectorSpace
    public_residual_space: AbstractVectorSpace
    execution_state_space: AbstractVectorSpace
    execution_residual_space: AbstractVectorSpace
    representation: ContinuationRepresentationPolicy
    coordinate_scale: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        public_state_space: AbstractVectorSpace,
        public_residual_space: AbstractVectorSpace,
        representation: ContinuationRepresentationPolicy,
        /,
        *,
        coordinate_scale: float = 1.0,
    ):
        if not isinstance(public_state_space, AbstractVectorSpace):
            raise TypeError("public_state_space must be an AbstractVectorSpace.")
        if not isinstance(public_residual_space, AbstractVectorSpace):
            raise TypeError("public_residual_space must be an AbstractVectorSpace.")
        if not isinstance(representation, ContinuationRepresentationPolicy):
            raise TypeError("representation must be a ContinuationRepresentationPolicy.")
        state_map = representation.state_coordinates
        residual_map = representation.residual_coordinates
        if state_map is not None and not public_state_space.compatible(
            state_map.source_space
        ):
            raise ValueError(
                "State coordinate map does not match the public state space."
            )
        if residual_map is not None and not public_residual_space.compatible(
            residual_map.source_space
        ):
            raise ValueError(
                "Residual coordinate map does not match the public residual space."
            )
        execution_state_space = (
            representation.state_execution_space
            if representation.state_execution_space is not None
            else (public_state_space if state_map is None else state_map.coordinate_space)
        )
        execution_residual_space = (
            representation.residual_execution_space
            if representation.residual_execution_space is not None
            else (
                public_residual_space
                if residual_map is None
                else residual_map.coordinate_space
            )
        )
        _real_space(execution_state_space, name="execution_state_space")
        _real_space(execution_residual_space, name="execution_residual_space")
        if execution_state_space.size != execution_residual_space.size:
            raise ValueError(
                "Continuation execution state and residual dimensions must match."
            )
        if _coordinate_dtype(execution_state_space) != _coordinate_dtype(
            execution_residual_space
        ):
            raise TypeError(
                "Continuation execution state and residual dtypes must match."
            )
        scale = float(coordinate_scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("coordinate_scale must be finite and positive.")
        self.public_state_space = public_state_space
        self.public_residual_space = public_residual_space
        self.execution_state_space = execution_state_space
        self.execution_residual_space = execution_residual_space
        self.representation = representation
        self.coordinate_scale = scale
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "continuation-geometry-v1",
                "public_state_space": public_state_space.space_id,
                "public_residual_space": public_residual_space.space_id,
                "execution_state_space": execution_state_space.space_id,
                "execution_residual_space": execution_residual_space.space_id,
                "representation": representation.policy_id,
                "coordinate_scale": scale,
            }
        )

    @classmethod
    def resolve(
        cls,
        state: PyTree[Any],
        residual: PyTree[Any],
        /,
        *,
        state_space: AbstractVectorSpace | None = None,
        residual_space: AbstractVectorSpace | None = None,
        representation: ContinuationRepresentationPolicy | None = None,
        coordinate_scale: float = 1.0,
    ) -> ContinuationGeometry:
        policy = (
            ContinuationRepresentationPolicy()
            if representation is None
            else representation
        )
        if not isinstance(policy, ContinuationRepresentationPolicy):
            raise TypeError(
                "representation must be ContinuationRepresentationPolicy or None."
            )
        public_state_space = (
            policy.state_coordinates.source_space
            if state_space is None and policy.state_coordinates is not None
            else (
                PyTreeSpace(validate_inexact_tree(state, name="continuation state"))
                if state_space is None
                else state_space
            )
        )
        public_residual_space = (
            policy.residual_coordinates.source_space
            if residual_space is None and policy.residual_coordinates is not None
            else (
                PyTreeSpace(validate_inexact_tree(residual, name="continuation residual"))
                if residual_space is None
                else residual_space
            )
        )
        geometry = cls(
            public_state_space,
            public_residual_space,
            policy,
            coordinate_scale=coordinate_scale,
        )
        state_coordinates = geometry.state_to_execution(state)
        residual_coordinates = geometry.residual_to_execution(residual)
        if not bool(
            tree_allfinite(state_coordinates) & tree_allfinite(residual_coordinates)
        ):
            raise ValueError("Initial continuation state and residual must be finite.")
        return geometry

    @property
    def coordinate_dtype(self) -> np.dtype:
        return _coordinate_dtype(self.execution_state_space)

    def _validate_defect(
        self,
        coordinate_map: AbstractRealCoordinateMap,
        value: Any,
        /,
        *,
        name: str,
    ) -> Any:
        defect = jnp.asarray(coordinate_map.defect(value))
        if defect.shape != () or not jnp.issubdtype(defect.dtype, jnp.floating):
            raise TypeError(f"{name} coordinate defect must be one real scalar.")
        return eqx.error_if(
            value,
            ~jnp.isfinite(defect) | (defect > self.representation.defect_tolerance),
            f"{name} lies outside the declared coordinate domain.",
        )

    def state_to_execution(self, state: PyTree[Any], /) -> PyTree[Array]:
        value = self.public_state_space.validate(state)
        coordinate_map = self.representation.state_coordinates
        if coordinate_map is not None:
            value = self._validate_defect(
                coordinate_map,
                value,
                name="Continuation state",
            )
            value = coordinate_map.to_real_coordinates(value)
        return self.execution_state_space.validate(value)

    def state_from_execution(self, state: PyTree[Any], /) -> PyTree[Array]:
        value = self.execution_state_space.validate(state)
        coordinate_map = self.representation.state_coordinates
        if coordinate_map is not None:
            value = coordinate_map.from_real_coordinates(value)
        return self.public_state_space.validate(value)

    def residual_to_execution(self, residual: PyTree[Any], /) -> PyTree[Array]:
        value = self.public_residual_space.validate(residual)
        coordinate_map = self.representation.residual_coordinates
        if coordinate_map is not None:
            value = self._validate_defect(
                coordinate_map,
                value,
                name="Continuation residual",
            )
            value = coordinate_map.to_real_coordinates(value)
        return self.execution_residual_space.validate(value)

    def residual_from_execution(self, residual: PyTree[Any], /) -> PyTree[Array]:
        value = self.execution_residual_space.validate(residual)
        coordinate_map = self.representation.residual_coordinates
        if coordinate_map is not None:
            value = coordinate_map.from_real_coordinates(value)
        return self.public_residual_space.validate(value)

    def state_tangent_to_execution(
        self,
        state: PyTree[Any],
        tangent: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        state_ = self.public_state_space.validate(state)
        tangent_ = self.public_state_space.validate(tangent)
        coordinate_map = self.representation.state_coordinates
        if coordinate_map is None:
            return self.execution_state_space.validate(tangent_)
        return self.execution_state_space.validate(
            jax.jvp(
                coordinate_map.to_real_coordinates,
                (state_,),
                (tangent_,),
            )[1]
        )

    def state_tangent_from_execution(
        self,
        state: PyTree[Any],
        tangent: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        state_ = self.execution_state_space.validate(state)
        tangent_ = self.execution_state_space.validate(tangent)
        coordinate_map = self.representation.state_coordinates
        if coordinate_map is None:
            return self.public_state_space.validate(tangent_)
        return self.public_state_space.validate(
            jax.jvp(
                coordinate_map.from_real_coordinates,
                (state_,),
                (tangent_,),
            )[1]
        )

    def state_inner(
        self,
        left: PyTree[Any],
        right: PyTree[Any],
        /,
    ) -> Array:
        return jnp.real(
            self.execution_state_space.inner(
                self.execution_state_space.validate(left),
                self.execution_state_space.validate(right),
            )
        )

    def state_norm(self, value: PyTree[Any], /) -> Array:
        squared = self.state_inner(value, value)
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    def residual_norm(self, value: PyTree[Any], /) -> Array:
        value_ = self.execution_residual_space.validate(value)
        squared = jnp.real(self.execution_residual_space.inner(value_, value_))
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    def augmented_inner(
        self,
        left_state: PyTree[Any],
        left_coordinate: Any,
        right_state: PyTree[Any],
        right_coordinate: Any,
        /,
    ) -> Array:
        left_scalar = jnp.asarray(left_coordinate)
        right_scalar = jnp.asarray(right_coordinate)
        return self.state_inner(left_state, right_state) + (
            left_scalar * right_scalar / self.coordinate_scale**2
        )

    def augmented_norm(self, state: PyTree[Any], coordinate: Any, /) -> Array:
        squared = self.augmented_inner(state, coordinate, state, coordinate)
        return jnp.sqrt(jnp.maximum(jnp.real(squared), 0.0))


__all__ = ["ContinuationGeometry", "ContinuationRepresentationPolicy"]

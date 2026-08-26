#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from .._state_geometry import AbstractStateGeometry
from ._cayley_dickson import ComplexAlgebraSpec, QuaternionAlgebraSpec
from ._core import AbstractFiniteRealAlgebraSpec


class _AbstractUnitCoordinateStateGeometry(AbstractStateGeometry):
    coordinate_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    @abc.abstractmethod
    def _algebra_marker(self) -> str:
        raise NotImplementedError

    def __init__(self, algebra: AbstractFiniteRealAlgebraSpec, tolerance: float, /):
        tolerance_ = float(tolerance)
        if tolerance_ <= 0.0:
            raise ValueError("Unit algebra geometry tolerance must be positive.")
        self.coordinate_dimension = algebra.coordinate_dimension
        self.tolerance = tolerance_
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "unit-algebra-state-geometry-v1",
                "algebra": algebra.algebra_id,
                "tolerance": tolerance_,
            }
        )
        self.retraction_method = "normalized-addition"
        self.trivial = False
        self.supports_exact_pullback = False
        self.supports_commutator_free = False

    def _value(self, value: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != (self.coordinate_dimension,):
            raise ValueError(
                f"{owner} must have shape {(self.coordinate_dimension,)}; got {array.shape}."
            )
        if jnp.iscomplexobj(array):
            raise TypeError(f"{owner} must use real algebra coordinates.")
        return array

    def contains(self, state: ArrayLike, /) -> Array:
        value = self._value(state, "Unit algebra state")
        norm = jnp.linalg.norm(value)
        return jnp.all(jnp.isfinite(value)) & (jnp.abs(norm - 1.0) <= self.tolerance)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        point = self._value(state, "Unit algebra state")
        tangent = self._value(vector, "Unit algebra tangent")
        return tangent - jnp.vdot(point, tangent) * point

    def to_local(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        return self.project_tangent(state, tangent)

    def from_local(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.project_tangent(state, local_tangent)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        point = self._value(state, "Unit algebra state")
        local = self.project_tangent(point, local_tangent)
        candidate = point + local
        norm = jnp.linalg.norm(candidate)
        return candidate / norm

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        base = self._value(state, "Unit algebra state")
        target = self._value(point, "Unit algebra target")
        overlap = jnp.vdot(base, target)
        overlap = eqx.error_if(
            overlap,
            overlap <= self.tolerance,
            "Normalized retraction inverse requires a positive local overlap.",
        )
        return target / overlap - base

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del state, local_tangent, tangent
        raise ValueError(
            "Normalized unit-algebra retraction has no exact pullback contract."
        )


class UnitComplexStateGeometry(_AbstractUnitCoordinateStateGeometry):
    def __init__(self, *, tolerance: float = 1e-9):
        super().__init__(ComplexAlgebraSpec(), tolerance)

    def _algebra_marker(self) -> str:
        return "complex"


class UnitQuaternionStateGeometry(_AbstractUnitCoordinateStateGeometry):
    def __init__(self, *, tolerance: float = 1e-9):
        super().__init__(QuaternionAlgebraSpec(), tolerance)

    def _algebra_marker(self) -> str:
        return "quaternion"


def unit_algebra_state_geometry(
    algebra: AbstractFiniteRealAlgebraSpec,
    /,
    *,
    tolerance: float = 1e-9,
) -> AbstractStateGeometry:
    if isinstance(algebra, ComplexAlgebraSpec):
        return UnitComplexStateGeometry(tolerance=tolerance)
    if isinstance(algebra, QuaternionAlgebraSpec):
        return UnitQuaternionStateGeometry(tolerance=tolerance)
    if not algebra.properties.proven("associative"):
        raise ValueError(
            "Nonassociative unit algebras do not inherit a Lie-group state geometry."
        )
    raise ValueError("No unit state geometry is prepared for this algebra family.")


__all__ = [
    "UnitComplexStateGeometry",
    "UnitQuaternionStateGeometry",
    "unit_algebra_state_geometry",
]

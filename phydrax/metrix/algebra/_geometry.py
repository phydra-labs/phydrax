#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax
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
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
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
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
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

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        base = self._value(state, "Unit algebra state")
        target = self._value(point, "Unit algebra target")
        return jnp.abs(jnp.vdot(base, target))

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        base = self._value(state, "Unit algebra state")
        local = self._value(local_tangent, "Unit algebra local tangent")
        direction = self._value(local_velocity, "Unit algebra local velocity")
        return jax.jvp(
            lambda value: self.retract(base, value),
            (local,),
            (direction,),
        )[1]

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        base = self._value(state, "Unit algebra state")
        target = self._value(point, "Unit algebra target")
        velocity = self.project_tangent(target, tangent)
        return jax.jvp(
            lambda value: self.inverse_retract(base, value),
            (target,),
            (velocity,),
        )[1]

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        base = self._value(state, "Unit algebra state")
        local = self._value(local_tangent, "Unit algebra local tangent")
        target_cotangent = self._value(cotangent, "Unit algebra physical cotangent")
        return jax.linear_transpose(
            lambda direction: self.retraction_jvp(base, local, direction),
            jnp.zeros_like(local),
        )(target_cotangent)[0]

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        source = self._value(state, "Unit algebra transport source")
        target = self._value(point, "Unit algebra transport target")
        source_tangent = self.project_tangent(source, tangent)
        overlap = jnp.vdot(source, target)
        denominator = eqx.error_if(
            1.0 + overlap,
            1.0 + overlap <= self.tolerance,
            "Unit algebra transport reaches the antipodal cut locus.",
        )
        return source_tangent - (jnp.vdot(target, source_tangent) / denominator) * (
            source + target
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        source = self._value(state, "Unit algebra transport source")
        target = self._value(point, "Unit algebra transport target")
        target_cotangent = self._value(cotangent, "Unit algebra physical cotangent")
        return jax.linear_transpose(
            lambda tangent: self.transport_tangent(source, target, tangent),
            jnp.zeros_like(source),
        )(target_cotangent)[0]


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

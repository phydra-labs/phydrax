#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_dtype_name
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._algebra_spaces import AlgebraArraySpace
from ._real_coordinates import AbstractRealCoordinateMap, RealCoordinateEvidence
from ._spaces import ArraySpace


AlgebraCoordinateStorage: TypeAlias = Literal["native_complex", "real_coordinates"]


class AlgebraCoordinatePlan(StrictModule, NonTrainableState):
    algebra: Any
    public_storage: AlgebraCoordinateStorage = eqx.field(static=True)
    public_axis: int = eqx.field(static=True)
    backend_axis: int = eqx.field(static=True)
    public_dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: Any,
        /,
        *,
        public_storage: AlgebraCoordinateStorage = "real_coordinates",
        public_axis: int = -1,
        backend_axis: int = 0,
        public_dtype: Any = np.float64,
    ):
        from ..metrix.algebra import AbstractFiniteRealAlgebraSpec, ComplexAlgebraSpec

        if not isinstance(algebra, AbstractFiniteRealAlgebraSpec):
            raise TypeError("algebra must implement AbstractFiniteRealAlgebraSpec.")
        if public_storage not in ("native_complex", "real_coordinates"):
            raise ValueError("Unknown algebra coordinate storage kind.")
        if public_storage == "native_complex" and not isinstance(
            algebra, ComplexAlgebraSpec
        ):
            raise ValueError("Native complex storage requires ComplexAlgebraSpec.")
        dtype = precision_dtype_name(public_dtype)
        if public_storage == "native_complex":
            if dtype not in ("complex64", "complex128"):
                raise TypeError("Native complex coordinates require complex64/128 dtype.")
        elif dtype not in ("float32", "float64"):
            raise TypeError("Real algebra coordinates require float32/64 dtype.")
        self.algebra = algebra
        self.public_storage = public_storage
        self.public_axis = int(public_axis)
        self.backend_axis = int(backend_axis)
        self.public_dtype = dtype
        self.plan_id = canonical_fingerprint(
            {
                "kind": "algebra-coordinate-plan-v1",
                "algebra": algebra.algebra_id,
                "public_storage": public_storage,
                "public_axis": int(public_axis),
                "backend_axis": int(backend_axis),
                "public_dtype": dtype,
            }
        )

    def prepare(self, base_shape: Sequence[int], /) -> "PreparedAlgebraCoordinates":
        return PreparedAlgebraCoordinates(self, base_shape)


class PreparedAlgebraCoordinates(AbstractRealCoordinateMap, NonTrainableState):
    plan: AlgebraCoordinatePlan
    base_shape: tuple[int, ...] = eqx.field(static=True)
    public_shape: tuple[int, ...] = eqx.field(static=True)
    public_axis: int = eqx.field(static=True)
    backend_axis: int = eqx.field(static=True)

    def __init__(self, plan: AlgebraCoordinatePlan, base_shape: Sequence[int], /):
        if not isinstance(plan, AlgebraCoordinatePlan):
            raise TypeError("plan must be AlgebraCoordinatePlan.")
        base = tuple(int(size) for size in base_shape)
        if any(size <= 0 for size in base):
            raise ValueError("Algebra coordinate base shape must be positive.")
        real_dtype = (
            jnp.empty((), dtype=jnp.dtype(plan.public_dtype)).real.dtype
            if plan.public_storage == "native_complex"
            else jnp.dtype(plan.public_dtype)
        )
        coordinate_space = AlgebraArraySpace(
            base,
            plan.algebra,
            algebra_axis=plan.backend_axis,
            dtype=real_dtype,
        )
        if plan.public_storage == "native_complex":
            source_space = ArraySpace(base, dtype=jnp.dtype(plan.public_dtype))
            public_shape = base
            public_axis = -1
        else:
            source_space = AlgebraArraySpace(
                base,
                plan.algebra,
                algebra_axis=plan.public_axis,
                dtype=jnp.dtype(plan.public_dtype),
            )
            public_shape = source_space.shape
            public_axis = source_space.algebra_axis
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-algebra-coordinates-v1",
                "plan": plan.plan_id,
                "base_shape": list(base),
                "source_space": source_space.space_id,
                "coordinate_space": coordinate_space.space_id,
            }
        )
        evidence = RealCoordinateEvidence(
            domain_kind="full",
            source_space_id=source_space.space_id,
            coordinate_space_id=coordinate_space.space_id,
            source_dtype=precision_dtype_name(source_space.structure().dtype),
            coordinate_dtype=precision_dtype_name(coordinate_space.dtype),
            source_shape=public_shape,
            coordinate_shape=coordinate_space.shape,
            norm_relation="isometry",
            projection_kind="identity",
            map_id=identifier,
        )
        self.plan = plan
        self.source_space = source_space
        self.coordinate_space = coordinate_space
        self.evidence = evidence
        self.coordinate_id = identifier
        self.base_shape = base
        self.public_shape = public_shape
        self.public_axis = public_axis
        self.backend_axis = coordinate_space.algebra_axis

    def validate_state(self, state: ArrayLike, /) -> Array:
        return self.source_space.validate(state)

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        return self.coordinate_space.validate(coordinates)

    def to_real_coordinates(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        if self.plan.public_storage == "native_complex":
            return jnp.stack((jnp.real(value), jnp.imag(value)), axis=self.backend_axis)
        return jnp.moveaxis(value, self.public_axis, self.backend_axis)

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = self.validate_coordinates(coordinates)
        if self.plan.public_storage == "native_complex":
            real = jnp.take(value, 0, axis=self.backend_axis)
            imag = jnp.take(value, 1, axis=self.backend_axis)
            return jax.lax.complex(real, imag).astype(jnp.dtype(self.plan.public_dtype))
        return jnp.moveaxis(value, self.backend_axis, self.public_axis)

    def pack_diffusion(self, value: ArrayLike, noise_shape: Sequence[int], /) -> Array:
        array = jnp.asarray(value)
        noise = tuple(int(size) for size in noise_shape)
        expected = self.public_shape + noise
        if array.shape != expected:
            raise ValueError(
                f"Algebra diffusion must have shape {expected}; got {array.shape}."
            )
        if self.plan.public_storage == "native_complex":
            array = array.astype(jnp.dtype(self.plan.public_dtype))
            return jnp.stack((jnp.real(array), jnp.imag(array)), axis=self.backend_axis)
        return jnp.moveaxis(array, self.public_axis, self.backend_axis)

    def unpack_values(self, value: ArrayLike, backend_axis: int, /) -> Array:
        array = jnp.asarray(value)
        axis = int(backend_axis)
        if axis < 0:
            axis += array.ndim
        if (
            axis < 0
            or axis >= array.ndim
            or array.shape[axis] != self.plan.algebra.coordinate_dimension
        ):
            raise ValueError("Backend values expose the wrong algebra coordinate axis.")
        if self.plan.public_storage == "native_complex":
            return jax.lax.complex(
                jnp.take(array, 0, axis=axis),
                jnp.take(array, 1, axis=axis),
            ).astype(jnp.dtype(self.plan.public_dtype))
        leading = axis
        target = leading + self.public_axis
        if self.public_axis < 0:
            target = array.ndim + self.public_axis
        return jnp.moveaxis(array, axis, target)

    def project(self, state: ArrayLike, /) -> Array:
        return self.validate_state(state)

    def defect(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        return jnp.zeros((), dtype=self.coordinate_space.dtype)


__all__ = [
    "AlgebraCoordinatePlan",
    "AlgebraCoordinateStorage",
    "PreparedAlgebraCoordinates",
]

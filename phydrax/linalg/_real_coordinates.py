#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_dtype_name
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._spaces import AbstractVectorSpace, ArraySpace


RealCoordinateDomainKind: TypeAlias = Literal["full", "constrained_subspace"]
RealCoordinateNormRelation: TypeAlias = Literal[
    "isometry",
    "scaled_isometry",
    "coordinate_equivalence",
    "unknown",
]


class RealCoordinateEvidence(StrictModule, NonTrainableState):
    """Identity and mathematical claim for one public-to-real coordinate map."""

    domain_kind: RealCoordinateDomainKind = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    coordinate_space_id: str = eqx.field(static=True)
    source_dtype: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    norm_relation: RealCoordinateNormRelation = eqx.field(static=True)
    projection_kind: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        domain_kind: RealCoordinateDomainKind,
        source_space_id: str,
        coordinate_space_id: str,
        source_dtype: str,
        coordinate_dtype: str,
        source_shape: tuple[int, ...],
        coordinate_shape: tuple[int, ...],
        norm_relation: RealCoordinateNormRelation,
        projection_kind: str,
        map_id: str,
    ):
        if domain_kind not in ("full", "constrained_subspace"):
            raise ValueError("Unknown real-coordinate domain kind.")
        if norm_relation not in (
            "isometry",
            "scaled_isometry",
            "coordinate_equivalence",
            "unknown",
        ):
            raise ValueError("Unknown real-coordinate norm relation.")
        identifiers = tuple(
            str(value)
            for value in (
                source_space_id,
                coordinate_space_id,
                source_dtype,
                coordinate_dtype,
                projection_kind,
                map_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Real-coordinate evidence identifiers must be non-empty.")
        source_shape_ = tuple(int(size) for size in source_shape)
        coordinate_shape_ = tuple(int(size) for size in coordinate_shape)
        if any(size <= 0 for size in source_shape_ + coordinate_shape_):
            raise ValueError("Real-coordinate shapes must contain positive dimensions.")
        self.domain_kind = domain_kind
        (
            self.source_space_id,
            self.coordinate_space_id,
            self.source_dtype,
            self.coordinate_dtype,
            self.projection_kind,
            self.map_id,
        ) = identifiers
        self.source_shape = source_shape_
        self.coordinate_shape = coordinate_shape_
        self.norm_relation = norm_relation
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "real-coordinate-evidence-v1",
                "domain_kind": domain_kind,
                "source_space": self.source_space_id,
                "coordinate_space": self.coordinate_space_id,
                "source_dtype": self.source_dtype,
                "coordinate_dtype": self.coordinate_dtype,
                "source_shape": list(source_shape_),
                "coordinate_shape": list(coordinate_shape_),
                "norm_relation": norm_relation,
                "projection_kind": self.projection_kind,
                "map": self.map_id,
            }
        )


class AbstractRealCoordinateMap(StrictModule):
    """Map a public numerical state onto explicit real execution coordinates."""

    source_space: AbstractVectorSpace
    coordinate_space: AbstractVectorSpace
    evidence: RealCoordinateEvidence
    coordinate_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def validate_state(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def to_real_coordinates(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def project(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def defect(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError


class ComplexCartesianCoordinates(AbstractRealCoordinateMap, NonTrainableState):
    """Full Cartesian real coordinates for one native-complex array space."""

    pair_axis: int = eqx.field(static=True)

    def __init__(self, source_space: ArraySpace, /, *, pair_axis: int = 0):
        if not isinstance(source_space, ArraySpace):
            raise TypeError("source_space must be an ArraySpace.")
        if not jnp.issubdtype(source_space.dtype, jnp.complexfloating):
            raise TypeError("Complex Cartesian coordinates require a complex ArraySpace.")
        axis = int(pair_axis)
        backend_rank = len(source_space.shape) + 1
        if axis < 0:
            axis += backend_rank
        if axis < 0 or axis >= backend_rank:
            raise ValueError("pair_axis lies outside the real-coordinate rank.")
        coordinate_shape = list(source_space.shape)
        coordinate_shape.insert(axis, 2)
        real_dtype = jnp.empty((), dtype=source_space.dtype).real.dtype
        identifier = canonical_fingerprint(
            {
                "kind": "complex-cartesian-coordinates-v1",
                "source_space": source_space.space_id,
                "pair_axis": axis,
            }
        )
        coordinate_space = ArraySpace(
            tuple(coordinate_shape),
            dtype=real_dtype,
            space_id=f"complex-cartesian-space:{identifier}",
        )
        evidence = RealCoordinateEvidence(
            domain_kind="full",
            source_space_id=source_space.space_id,
            coordinate_space_id=coordinate_space.space_id,
            source_dtype=precision_dtype_name(source_space.dtype),
            coordinate_dtype=precision_dtype_name(real_dtype),
            source_shape=source_space.shape,
            coordinate_shape=tuple(coordinate_shape),
            norm_relation="isometry",
            projection_kind="identity",
            map_id=identifier,
        )
        self.source_space = source_space
        self.coordinate_space = coordinate_space
        self.evidence = evidence
        self.coordinate_id = identifier
        self.pair_axis = axis

    def validate_state(self, state: ArrayLike, /) -> Array:
        return self.source_space.validate(state)

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        return self.coordinate_space.validate(coordinates)

    def to_real_coordinates(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        return jnp.stack((jnp.real(value), jnp.imag(value)), axis=self.pair_axis)

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = self.validate_coordinates(coordinates)
        real = jnp.take(value, 0, axis=self.pair_axis)
        imag = jnp.take(value, 1, axis=self.pair_axis)
        return jax.lax.complex(real, imag).astype(self.source_space.dtype)

    def to_real_array(
        self,
        value: ArrayLike,
        trailing_shape: tuple[int, ...] = (),
        /,
    ) -> Array:
        array = jnp.asarray(value)
        expected = self.source_space.shape + tuple(int(size) for size in trailing_shape)
        if array.shape != expected:
            raise ValueError(
                f"Complex value must have shape {expected}; got {array.shape}."
            )
        array = array.astype(self.source_space.dtype)
        return jnp.stack((jnp.real(array), jnp.imag(array)), axis=self.pair_axis)

    def from_real_array(self, value: ArrayLike, pair_axis: int, /) -> Array:
        array = jnp.asarray(value)
        axis = int(pair_axis)
        if axis < 0:
            axis += array.ndim
        if axis < 0 or axis >= array.ndim or int(array.shape[axis]) != 2:
            raise ValueError("Real coordinates require one size-two Cartesian axis.")
        return jax.lax.complex(
            jnp.take(array, 0, axis=axis),
            jnp.take(array, 1, axis=axis),
        ).astype(self.source_space.dtype)

    def pack_diffusion(
        self,
        value: ArrayLike,
        noise_shape: tuple[int, ...],
        /,
    ) -> Array:
        return self.to_real_array(value, noise_shape)

    def unpack_values(self, value: ArrayLike, pair_axis: int, /) -> Array:
        return self.from_real_array(value, pair_axis)

    def project(self, state: ArrayLike, /) -> Array:
        return self.validate_state(state)

    def defect(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        return jnp.zeros((), dtype=value.real.dtype)


__all__ = [
    "AbstractRealCoordinateMap",
    "ComplexCartesianCoordinates",
    "RealCoordinateDomainKind",
    "RealCoordinateEvidence",
    "RealCoordinateNormRelation",
]

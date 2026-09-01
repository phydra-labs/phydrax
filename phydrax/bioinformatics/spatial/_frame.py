#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


@dataclass(frozen=True, slots=True)
class SpatialUnit:
    """Host description of one spatial unit.

    ``micrometre_scale`` is the number of micrometres represented by one unit.
    Pixel units deliberately have no physical scale; converting them therefore
    requires an explicit calibrated transform rather than an invented convention.
    """

    name: str
    symbol: str
    micrometre_scale: float | None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("SpatialUnit name must be non-empty.")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("SpatialUnit symbol must be non-empty.")
        if self.micrometre_scale is not None:
            scale = float(self.micrometre_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "SpatialUnit micrometre_scale must be finite and positive."
                )
            object.__setattr__(self, "micrometre_scale", scale)

    @property
    def physical(self) -> bool:
        return self.micrometre_scale is not None

    def conversion_factor_to(self, target: SpatialUnit, /) -> float:
        if not isinstance(target, SpatialUnit):
            raise TypeError("target must be a SpatialUnit.")
        if self.micrometre_scale is None or target.micrometre_scale is None:
            raise ValueError(
                "Pixel and other uncalibrated units need an explicit transform."
            )
        return self.micrometre_scale / target.micrometre_scale


NANOMETRE = SpatialUnit("nanometre", "nm", 1.0e-3)
MICROMETRE = SpatialUnit("micrometre", "µm", 1.0)
MILLIMETRE = SpatialUnit("millimetre", "mm", 1.0e3)
PIXEL = SpatialUnit("pixel", "px", None)


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """Host identity of an ordered spatial coordinate frame."""

    frame_id: str
    axes: tuple[str, ...]
    unit: SpatialUnit
    semantic_id: str = field(init=False)
    code: tuple[int, ...] = field(init=False, repr=False)

    def __init__(
        self,
        frame_id: str,
        axes: Sequence[str],
        unit: SpatialUnit,
        /,
    ):
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a non-empty string.")
        axes_ = tuple(axes)
        if not axes_ or any(
            not isinstance(axis, str) or not axis.strip() for axis in axes_
        ):
            raise ValueError("SpatialFrame axes must be non-empty strings.")
        if len(set(axes_)) != len(axes_):
            raise ValueError("SpatialFrame axes must be unique and ordered.")
        if not isinstance(unit, SpatialUnit):
            raise TypeError("unit must be a SpatialUnit.")
        semantic_id = canonical_fingerprint(
            {
                "kind": "spatial-frame-v1",
                "frame_id": frame_id,
                "axes": list(axes_),
                "unit": {
                    "name": unit.name,
                    "symbol": unit.symbol,
                    "micrometre_scale": unit.micrometre_scale,
                },
            }
        )
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "axes", axes_)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "semantic_id", semantic_id)
        object.__setattr__(self, "code", tuple(bytes.fromhex(semantic_id)))

    @property
    def dimension(self) -> int:
        return len(self.axes)

    def same_coordinate_system(self, other: SpatialFrame, /) -> bool:
        return (
            isinstance(other, SpatialFrame)
            and self.frame_id == other.frame_id
            and self.axes == other.axes
        )


class SpatialCoordinates(StrictModule):
    """Numeric coordinates tagged by a dynamic, nondifferentiable frame code."""

    values: Array
    frame_code: Array
    unit_scale: Array
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        values: Any,
        frame: SpatialFrame | None = None,
        /,
        *,
        frame_code: Any | None = None,
        unit_scale: Any | None = None,
    ):
        array = jnp.asarray(values, dtype=float)
        if array.ndim < 1 or int(array.shape[-1]) < 1:
            raise ValueError(
                "Spatial coordinates must end in a non-empty coordinate axis."
            )
        if frame is not None:
            if not isinstance(frame, SpatialFrame):
                raise TypeError("frame must be a SpatialFrame.")
            if int(array.shape[-1]) != frame.dimension:
                raise ValueError(
                    f"Coordinate dimension {array.shape[-1]} does not match frame dimension "
                    f"{frame.dimension}."
                )
            code = jnp.asarray(frame.code, dtype=jnp.uint8)
            scale = (
                0.0
                if frame.unit.micrometre_scale is None
                else frame.unit.micrometre_scale
            )
        else:
            if frame_code is None or unit_scale is None:
                raise ValueError(
                    "Internal coordinates require frame_code and unit_scale."
                )
            code = jnp.asarray(frame_code, dtype=jnp.uint8)
            scale = unit_scale
        if code.shape != (32,):
            raise ValueError("Spatial frame codes must contain 32 digest bytes.")
        self.values = array
        self.frame_code = code
        self.unit_scale = jnp.asarray(scale, dtype=array.dtype)
        self.dimension = int(array.shape[-1])

    def matches(self, frame: SpatialFrame, /) -> Array:
        if not isinstance(frame, SpatialFrame):
            raise TypeError("frame must be a SpatialFrame.")
        return jnp.all(self.frame_code == jnp.asarray(frame.code, dtype=jnp.uint8))


class AffineSpatialTransform(StrictModule):
    """Differentiable affine map between two explicit spatial frames.

    For row-major point arrays the action is ``points @ matrix.T + offset``.
    Frame identities and unit semantics are dynamic nondifferentiable leaves, so
    changing a frame never triggers accidental recompilation of the numeric map.
    """

    matrix: Array
    offset: Array
    source_code: Array
    target_code: Array
    source_unit_scale: Array
    target_unit_scale: Array
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        matrix: Any,
        offset: Any,
        source: SpatialFrame,
        target: SpatialFrame,
        /,
    ):
        if not isinstance(source, SpatialFrame) or not isinstance(target, SpatialFrame):
            raise TypeError("source and target must be SpatialFrame instances.")
        matrix_ = jnp.asarray(matrix, dtype=float)
        offset_ = jnp.asarray(offset, dtype=matrix_.dtype)
        expected = (target.dimension, source.dimension)
        if matrix_.shape != expected:
            raise ValueError(
                f"Transform matrix must have shape {expected}; got {matrix_.shape}."
            )
        if offset_.shape != (target.dimension,):
            raise ValueError(
                f"Transform offset must have shape {(target.dimension,)}; got {offset_.shape}."
            )
        self.matrix = matrix_
        self.offset = offset_
        self.source_code = jnp.asarray(source.code, dtype=jnp.uint8)
        self.target_code = jnp.asarray(target.code, dtype=jnp.uint8)
        self.source_unit_scale = jnp.asarray(
            0.0 if source.unit.micrometre_scale is None else source.unit.micrometre_scale,
            dtype=matrix_.dtype,
        )
        self.target_unit_scale = jnp.asarray(
            0.0 if target.unit.micrometre_scale is None else target.unit.micrometre_scale,
            dtype=matrix_.dtype,
        )
        self.source_dimension = source.dimension
        self.target_dimension = target.dimension

    @classmethod
    def _from_numeric(
        cls,
        matrix: Any,
        offset: Any,
        source_code: Any,
        target_code: Any,
        source_unit_scale: Any,
        target_unit_scale: Any,
    ) -> AffineSpatialTransform:
        result = object.__new__(cls)
        matrix_ = jnp.asarray(matrix)
        object.__setattr__(result, "matrix", matrix_)
        object.__setattr__(result, "offset", jnp.asarray(offset, dtype=matrix_.dtype))
        object.__setattr__(
            result, "source_code", jnp.asarray(source_code, dtype=jnp.uint8)
        )
        object.__setattr__(
            result, "target_code", jnp.asarray(target_code, dtype=jnp.uint8)
        )
        object.__setattr__(
            result,
            "source_unit_scale",
            jnp.asarray(source_unit_scale, dtype=matrix_.dtype),
        )
        object.__setattr__(
            result,
            "target_unit_scale",
            jnp.asarray(target_unit_scale, dtype=matrix_.dtype),
        )
        object.__setattr__(result, "source_dimension", int(matrix_.shape[1]))
        object.__setattr__(result, "target_dimension", int(matrix_.shape[0]))
        return result

    @classmethod
    def identity(cls, frame: SpatialFrame, /) -> AffineSpatialTransform:
        return cls(jnp.eye(frame.dimension), jnp.zeros((frame.dimension,)), frame, frame)

    def apply_array(self, values: Any, /) -> Array:
        points = jnp.asarray(values, dtype=self.matrix.dtype)
        if points.ndim < 1 or int(points.shape[-1]) != self.source_dimension:
            raise ValueError(
                f"Transform input must end in dimension {self.source_dimension}."
            )
        return points @ self.matrix.T + self.offset

    def apply(self, coordinates: SpatialCoordinates, /) -> SpatialCoordinates:
        if not isinstance(coordinates, SpatialCoordinates):
            raise TypeError("coordinates must be SpatialCoordinates.")
        mismatch = jnp.any(coordinates.frame_code != self.source_code)
        if not isinstance(mismatch, jax_core.Tracer) and bool(mismatch):
            raise ValueError("Coordinate frame/unit does not match transform source.")
        checked = eqx.error_if(
            coordinates.values,
            mismatch,
            "Coordinate frame/unit does not match transform source.",
        )
        values = self.apply_array(checked)
        return SpatialCoordinates(
            values,
            frame_code=self.target_code,
            unit_scale=self.target_unit_scale,
        )

    def compose(self, prior: AffineSpatialTransform, /) -> AffineSpatialTransform:
        """Return ``self(prior(x))`` after checking the intermediate frame."""
        if not isinstance(prior, AffineSpatialTransform):
            raise TypeError("prior must be an AffineSpatialTransform.")
        mismatch = jnp.any(prior.target_code != self.source_code)
        if not isinstance(mismatch, jax_core.Tracer) and bool(mismatch):
            raise ValueError("Transform composition has a frame or unit mismatch.")
        matrix = self.matrix @ prior.matrix
        offset = self.matrix @ prior.offset + self.offset
        return AffineSpatialTransform._from_numeric(
            matrix,
            offset,
            prior.source_code,
            self.target_code,
            prior.source_unit_scale,
            self.target_unit_scale,
        )

    def then(self, following: AffineSpatialTransform, /) -> AffineSpatialTransform:
        return following.compose(self)


def unit_conversion_transform(
    source: SpatialFrame,
    target: SpatialFrame,
    /,
) -> AffineSpatialTransform:
    """Create the exact scale conversion between two views of one frame."""
    if not source.same_coordinate_system(target):
        raise ValueError("Unit conversion requires matching frame identity and axes.")
    factor = source.unit.conversion_factor_to(target.unit)
    return AffineSpatialTransform(
        factor * jnp.eye(source.dimension),
        jnp.zeros((target.dimension,)),
        source,
        target,
    )


__all__ = [
    "AffineSpatialTransform",
    "MICROMETRE",
    "MILLIMETRE",
    "NANOMETRE",
    "PIXEL",
    "SpatialCoordinates",
    "SpatialFrame",
    "SpatialUnit",
    "unit_conversion_transform",
]

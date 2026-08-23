#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


BoundaryTarget = Callable[[Array, Array, Array, Array, Any], ArrayLike]


def _boundary_value(value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.shape == () or array.shape == (shape[-1],):
        return jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(f"Boundary state must have shape {shape}, scalar, or components.")
    return array


class AbstractFiniteVolumeBoundary(StrictModule, NonTrainableState):
    """Physical boundary policy that constructs an exterior face state."""

    boundary_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        raise NotImplementedError


class ExtrapolationBoundary(AbstractFiniteVolumeBoundary):
    """Zero-normal-gradient exterior state."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "fv-extrapolation"})

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, coordinates, outward_normal, axis, args
        return jnp.asarray(interior)


class ConstantStateBoundary(AbstractFiniteVolumeBoundary):
    """Constant exterior conservative state."""

    value: Array

    def __init__(self, value: ArrayLike, /):
        value_ = jnp.asarray(value)
        if value_.ndim > 1:
            raise ValueError("Constant boundary state must be scalar or component vector.")
        self.value = value_
        self.boundary_id = canonical_fingerprint(
            {"kind": "fv-constant-state", "value": array_tree_fingerprint(value_)}
        )

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, coordinates, outward_normal, axis, args
        return _boundary_value(self.value, interior.shape)


class PrescribedStateBoundary(AbstractFiniteVolumeBoundary):
    """Time-, state-, coordinate-, and parameter-dependent exterior state."""

    target: BoundaryTarget = eqx.field(static=True)

    def __init__(self, target: BoundaryTarget, /, *, boundary_id: str):
        if not callable(target):
            raise TypeError("target must be callable.")
        identifier = str(boundary_id)
        if not identifier:
            raise ValueError("boundary_id must be non-empty.")
        self.target = target
        self.boundary_id = identifier

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, axis
        value = self.target(time, interior, coordinates, outward_normal, args)
        return _boundary_value(value, interior.shape)


class ReflectiveBoundary(AbstractFiniteVolumeBoundary):
    """Equation-owned reflective state transformation."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "fv-reflective"})

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, outward_normal, args
        return system.reflect_state(interior, axis)


class PrescribedNormalFluxBoundary(AbstractFiniteVolumeBoundary):
    """Direct outward integrated-flux-density policy."""

    target: BoundaryTarget = eqx.field(static=True)

    def __init__(self, target: BoundaryTarget, /, *, boundary_id: str):
        if not callable(target):
            raise TypeError("target must be callable.")
        identifier = str(boundary_id)
        if not identifier:
            raise ValueError("boundary_id must be non-empty.")
        self.target = target
        self.boundary_id = identifier

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, interior, coordinates, outward_normal, axis, args
        raise ValueError("PrescribedNormalFluxBoundary supplies flux, not exterior state.")

    def normal_flux(
        self,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        args: Any,
        /,
    ) -> Array:
        value = self.target(time, interior, coordinates, outward_normal, args)
        return _boundary_value(value, interior.shape)


class FiniteVolumeBoundaryPair(StrictModule, NonTrainableState):
    """Lower and upper physical boundaries for one bounded axis."""

    lower: AbstractFiniteVolumeBoundary
    upper: AbstractFiniteVolumeBoundary
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: AbstractFiniteVolumeBoundary,
        upper: AbstractFiniteVolumeBoundary,
        /,
    ):
        if not isinstance(lower, AbstractFiniteVolumeBoundary) or not isinstance(
            upper, AbstractFiniteVolumeBoundary
        ):
            raise TypeError("Boundary pairs require finite-volume boundary policies.")
        self.lower = lower
        self.upper = upper
        self.pair_id = canonical_fingerprint(
            {
                "kind": "fv-boundary-pair",
                "lower": lower.boundary_id,
                "upper": upper.boundary_id,
            }
        )


class FiniteVolumeBoundarySet(StrictModule, NonTrainableState):
    """Axis-ordered bounded policies; periodic axes use ``None``."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    pairs: tuple[FiniteVolumeBoundaryPair | None, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        pairs: Sequence[FiniteVolumeBoundaryPair | None],
        /,
    ):
        names = tuple(str(name) for name in axis_names)
        pairs_ = tuple(pairs)
        if (
            not names
            or len(names) != len(pairs_)
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Boundary axes and pairs must align with unique names.")
        if any(
            pair is not None and not isinstance(pair, FiniteVolumeBoundaryPair)
            for pair in pairs_
        ):
            raise TypeError("Boundary entries must be FiniteVolumeBoundaryPair or None.")
        self.axis_names = names
        self.pairs = pairs_
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "fv-boundary-set",
                "axes": list(names),
                "pairs": [None if pair is None else pair.pair_id for pair in pairs_],
            }
        )

    @classmethod
    def periodic(cls, axis_names: Sequence[str], /) -> "FiniteVolumeBoundarySet":
        names = tuple(axis_names)
        return cls(names, (None,) * len(names))


__all__ = [
    "AbstractFiniteVolumeBoundary",
    "ConstantStateBoundary",
    "ExtrapolationBoundary",
    "FiniteVolumeBoundaryPair",
    "FiniteVolumeBoundarySet",
    "PrescribedNormalFluxBoundary",
    "PrescribedStateBoundary",
    "ReflectiveBoundary",
]

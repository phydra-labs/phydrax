# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Atmospheric vertical-coordinate identity and state-dependent pressure measures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class HybridPressureCoordinate(StrictModule, NonTrainableState):
    """Top-to-bottom interfaces p=A p_ref+B p_s (or pressure-valued A).

    Coefficients are fixed preparation state. Surface pressure stays differentiable.
    The final array axis is vertical. Invalid columns are reported, never clipped.
    """

    a: Array
    b: Array
    reference_pressure: float = eqx.field(static=True)
    a_is_pressure: bool = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(
        self,
        a: Any,
        b: Any,
        *,
        reference_pressure: float = 100000.0,
        a_is_pressure: bool = False,
        dtype: Any = None,
    ):
        dtype_ = (
            np.dtype(np.float64 if jax.config.x64_enabled else np.float32)
            if dtype is None
            else np.dtype(dtype)
        )
        if dtype_ not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Hybrid coefficients require float32 or float64 precision.")
        if dtype_ == np.dtype("float64") and not jax.config.x64_enabled:
            raise ValueError(
                "Enable JAX double precision to restore float64 hybrid coordinates."
            )
        a_ = np.asarray(a, dtype=dtype_)
        b_ = np.asarray(b, dtype=dtype_)
        if (
            a_.ndim != 1
            or a_.size < 2
            or a_.shape != b_.shape
            or not np.all(np.isfinite(a_))
            or not np.all(np.isfinite(b_))
        ):
            raise ValueError(
                "Hybrid coefficients require equally sized finite 1-D interface arrays."
            )
        if not np.isfinite(reference_pressure) or reference_pressure <= 0:
            raise ValueError("Reference pressure must be finite and positive.")
        pressure = (
            a_ if a_is_pressure else a_ * reference_pressure
        ) + b_ * reference_pressure
        if np.any(pressure < 0) or np.any(np.diff(pressure) <= 0):
            raise ValueError(
                "Hybrid interfaces must increase top-to-bottom at reference pressure."
            )
        self.a = jnp.asarray(a_, dtype=dtype_)
        self.b = jnp.asarray(b_, dtype=dtype_)
        self.reference_pressure = float(reference_pressure)
        self.a_is_pressure = bool(a_is_pressure)
        self.coordinate_id = canonical_fingerprint(
            {
                "kind": "hybrid-pressure-coordinate",
                "a": a_.tolist(),
                "b": b_.tolist(),
                "reference_pressure": self.reference_pressure,
                "dtype": str(dtype_),
                "a_is_pressure": self.a_is_pressure,
                "orientation": "top-to-bottom",
            }
        )

    @property
    def level_count(self) -> int:
        return self.a.shape[0] - 1

    def interfaces(self, surface_pressure: Any) -> Array:
        ps = jnp.asarray(surface_pressure)
        base = self.a if self.a_is_pressure else self.a * self.reference_pressure
        return base + ps[..., None] * self.b

    def valid(self, surface_pressure: Any) -> Array:
        ps = jnp.asarray(surface_pressure)
        pressure = self.interfaces(ps)
        return (
            jnp.isfinite(ps)
            & (ps > 0)
            & jnp.all(jnp.isfinite(pressure), axis=-1)
            & jnp.all(pressure >= 0, axis=-1)
            & jnp.all(jnp.diff(pressure, axis=-1) > 0, axis=-1)
        )

    def layer_mass(self, surface_pressure: Any, gravity: Any = 9.80665) -> Array:
        """Mass per horizontal area; caller must admit valid pressure and gravity."""
        gravity_ = jnp.asarray(gravity)
        gravity_ = eqx.error_if(
            gravity_,
            jnp.any(~jnp.isfinite(gravity_) | (gravity_ <= 0)),
            "Gravity must be finite and positive.",
        )
        return jnp.diff(self.interfaces(surface_pressure), axis=-1) / gravity_[..., None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": np.asarray(self.a).tolist(),
            "b": np.asarray(self.b).tolist(),
            "reference_pressure": self.reference_pressure,
            "a_is_pressure": self.a_is_pressure,
            "coordinate_id": self.coordinate_id,
            "dtype": str(self.a.dtype),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HybridPressureCoordinate:
        if set(payload) != {
            "a",
            "b",
            "reference_pressure",
            "a_is_pressure",
            "coordinate_id",
            "dtype",
        }:
            raise ValueError("Vertical descriptor must use exactly the canonical fields.")
        result = cls(
            payload["a"],
            payload["b"],
            reference_pressure=payload["reference_pressure"],
            a_is_pressure=payload["a_is_pressure"],
            dtype=payload["dtype"],
        )
        if result.coordinate_id != payload["coordinate_id"]:
            raise ValueError(
                "Vertical descriptor fingerprint does not match its content."
            )
        return result


__all__ = ["HybridPressureCoordinate"]

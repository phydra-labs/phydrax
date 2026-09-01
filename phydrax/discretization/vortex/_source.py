#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


def _identifier(name: str, value: str | None, payload: object, /) -> str:
    identifier = canonical_fingerprint(payload) if value is None else str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class VortexSourceState(StrictModule):
    """Canonical fixed-capacity integrated-vorticity source state.

    ``strength`` is circulation in two dimensions and integrated vector vorticity
    in three dimensions. It is never material-particle mass.
    """

    positions: Array
    strength: Array
    core_radius: Array | None
    volume: Array | None
    active_mask: Array
    dimension: int = eqx.field(static=True)
    source_kind: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        strength: ArrayLike,
        /,
        *,
        core_radius: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        dimension: int | None = None,
        source_kind: str = "particle",
        source_id: str | None = None,
    ):
        positions_ = jnp.asarray(positions)
        if positions_.ndim != 2 or positions_.shape[0] == 0:
            raise ValueError(
                "Vortex source positions require non-empty shape (source, dim)."
            )
        dimension_ = int(positions_.shape[1] if dimension is None else dimension)
        if dimension_ not in (2, 3) or positions_.shape[1] != dimension_:
            raise ValueError(
                "Vortex source dimension must be 2 or 3 and match positions."
            )
        if not jnp.issubdtype(positions_.dtype, jnp.floating):
            raise TypeError("Vortex source positions must use a real floating dtype.")
        strength_ = jnp.asarray(strength, dtype=positions_.dtype)
        expected_strength = (
            (positions_.shape[0],)
            if dimension_ == 2
            else (positions_.shape[0], dimension_)
        )
        if strength_.shape != expected_strength:
            raise ValueError(
                f"Vortex source strength must have shape {expected_strength}, "
                f"got {strength_.shape}."
            )
        active = (
            jnp.ones((positions_.shape[0],), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != (positions_.shape[0],):
            raise ValueError("Vortex source active_mask must have source-capacity shape.")
        core = (
            None
            if core_radius is None
            else jnp.asarray(core_radius, dtype=positions_.dtype)
        )
        volume_ = None if volume is None else jnp.asarray(volume, dtype=positions_.dtype)
        if core is not None and core.shape != (positions_.shape[0],):
            raise ValueError("Vortex source core_radius must have source-capacity shape.")
        if volume_ is not None and volume_.shape != (positions_.shape[0],):
            raise ValueError("Vortex source volume must have source-capacity shape.")
        strength_mask = active if dimension_ == 2 else active[:, None]
        positions_ = eqx.error_if(
            positions_,
            jnp.any(jnp.where(active[:, None], ~jnp.isfinite(positions_), False)),
            "Active vortex source positions must be finite.",
        )
        strength_ = eqx.error_if(
            strength_,
            jnp.any(jnp.where(strength_mask, ~jnp.isfinite(strength_), False)),
            "Active vortex source strengths must be finite.",
        )
        if core is not None:
            core = eqx.error_if(
                core,
                jnp.any(jnp.where(active, ~jnp.isfinite(core) | (core <= 0.0), False)),
                "Active vortex source core radii must be finite and positive.",
            )
        if volume_ is not None:
            volume_ = eqx.error_if(
                volume_,
                jnp.any(
                    jnp.where(active, ~jnp.isfinite(volume_) | (volume_ <= 0.0), False)
                ),
                "Active vortex source volumes must be finite and positive.",
            )
        kind = str(source_kind).strip()
        if not kind:
            raise ValueError("source_kind must be non-empty.")
        identifier = _identifier(
            "source_id",
            source_id,
            {
                "kind": "vortex-source-state",
                "dimension": dimension_,
                "source_kind": kind,
                "capacity": int(positions_.shape[0]),
                "coordinate_dtype": str(positions_.dtype),
                "strength_shape": list(strength_.shape),
                "has_core_radius": core is not None,
                "has_volume": volume_ is not None,
            },
        )
        self.positions = positions_
        self.strength = strength_
        self.core_radius = core
        self.volume = volume_
        self.active_mask = active
        self.dimension = dimension_
        self.source_kind = kind
        self.source_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.positions.shape[0])

    def safe_positions(self) -> Array:
        return jnp.where(self.active_mask[:, None], self.positions, 0.0)

    def safe_strength(self) -> Array:
        mask = self.active_mask if self.dimension == 2 else self.active_mask[:, None]
        return jnp.where(mask, self.strength, 0.0)

    def safe_core_radius(self) -> Array:
        if self.core_radius is None:
            raise ValueError("Vortex source core_radius is unavailable.")
        return jnp.where(self.active_mask, self.core_radius, 1.0)

    def safe_volume(self) -> Array:
        if self.volume is None:
            raise ValueError("Vortex source volume is unavailable.")
        return jnp.where(self.active_mask, self.volume, 1.0)


class VortexTargetState(StrictModule):
    """Canonical target positions with optional explicit source-index identity."""

    positions: Array
    source_indices: Array | None
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        /,
        *,
        source_indices: ArrayLike | None = None,
        target_id: str | None = None,
    ):
        positions_ = jnp.asarray(positions)
        if (
            positions_.ndim != 2
            or positions_.shape[0] == 0
            or positions_.shape[1] not in (2, 3)
        ):
            raise ValueError(
                "Vortex target positions require non-empty shape (target, 2|3)."
            )
        if not jnp.issubdtype(positions_.dtype, jnp.floating):
            raise TypeError("Vortex target positions must use a real floating dtype.")
        positions_ = eqx.error_if(
            positions_,
            jnp.any(~jnp.isfinite(positions_)),
            "Vortex target positions must be finite.",
        )
        identity = None if source_indices is None else jnp.asarray(source_indices)
        if identity is not None:
            if identity.shape != (positions_.shape[0],):
                raise ValueError(
                    "Vortex target source_indices must have target-capacity shape."
                )
            if not jnp.issubdtype(identity.dtype, jnp.integer):
                raise TypeError("Vortex target source_indices must contain integers.")
            identity = identity.astype(jnp.int32)
            identity = eqx.error_if(
                identity,
                jnp.any(identity < -1),
                "Vortex target source_indices entries must be -1 or nonnegative.",
            )
        identifier = _identifier(
            "target_id",
            target_id,
            {
                "kind": "vortex-target-state",
                "dimension": int(positions_.shape[1]),
                "capacity": int(positions_.shape[0]),
                "coordinate_dtype": str(positions_.dtype),
                "has_source_identity": identity is not None,
            },
        )
        self.positions = positions_
        self.source_indices = identity
        self.target_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.positions.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.positions.shape[1])


__all__ = ["VortexSourceState", "VortexTargetState"]

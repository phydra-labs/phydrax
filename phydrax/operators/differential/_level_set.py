#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from phydrax.domain import DomainFunction, UnaryFieldEvaluator
from phydrax.geometry import regularized_delta_values, regularized_heaviside_values

from ..._strict import StrictModule
from ._domain_ops import div, dt, grad


class _CompactHeaviside(StrictModule):
    width: float = eqx.field(static=True)

    def __call__(self, value):
        return regularized_heaviside_values(value, width=self.width)


class _CompactDelta(StrictModule):
    width: float = eqx.field(static=True)

    def __call__(self, value):
        return regularized_delta_values(value, width=self.width)


class _GradientNorm(StrictModule):
    gradient: DomainFunction

    def __call__(self, *args, key=None, **kwargs):
        values = _real_values(
            self.gradient.func(*args, key=key, **kwargs),
            "Level-set gradient",
        )
        return jnp.sqrt(jnp.sum(values * values, axis=-1))


class _NormalizedGradient(StrictModule):
    gradient: DomainFunction
    gradient_floor: float = eqx.field(static=True)

    def __call__(self, *args, key=None, **kwargs):
        values = _real_values(
            self.gradient.func(*args, key=key, **kwargs),
            "Level-set gradient",
        )
        magnitude = jnp.sqrt(jnp.sum(values * values, axis=-1))
        denominator = jnp.maximum(magnitude, self.gradient_floor)
        return values / denominator[..., None]


class _LowerBound(StrictModule):
    minimum: float = eqx.field(static=True)

    def __call__(self, value):
        return jnp.maximum(_real_values(value, "Level-set gradient norm"), self.minimum)


def _real_values(value, name: str, /):
    values = jnp.asarray(value)
    if jnp.iscomplexobj(values):
        raise TypeError(f"{name} must be real-valued.")
    return values


def _positive_finite(value: float, name: str, /) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _field(value: DomainFunction, name: str, /) -> DomainFunction:
    if not isinstance(value, DomainFunction):
        raise TypeError(f"{name} must be a DomainFunction.")
    return value


def regularized_heaviside(
    level_set: DomainFunction,
    /,
    *,
    width: float,
) -> DomainFunction:
    r"""Return a compact regularized Heaviside field.

    The transition is zero for ``phi <= -width``, one for
    ``phi >= width``, and uses the standard cosine regularization inside the
    band. The sign convention is therefore positive-side occupancy.
    """

    field = _field(level_set, "level_set")
    width_ = _positive_finite(width, "width")
    return DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=UnaryFieldEvaluator(field.func, _CompactHeaviside(width_)),
        metadata=field.metadata,
    )


def regularized_delta(
    level_set: DomainFunction,
    /,
    *,
    width: float,
) -> DomainFunction:
    r"""Return the compact derivative of :func:`regularized_heaviside`.

    The result is nonnegative, even in the level-set value, integrates to one
    along the scalar level-set coordinate, and vanishes outside ``|phi| < width``.
    """

    field = _field(level_set, "level_set")
    width_ = _positive_finite(width, "width")
    return DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=UnaryFieldEvaluator(field.func, _CompactDelta(width_)),
        metadata=field.metadata,
    )


def level_set_phase_indicator(
    level_set: DomainFunction,
    /,
    *,
    width: float,
    phase: Literal["inside", "outside"] = "inside",
) -> DomainFunction:
    r"""Return a smooth phase indicator using ``phi < 0`` as the inside phase."""

    if phase == "inside":
        return regularized_heaviside(-_field(level_set, "level_set"), width=width)
    if phase == "outside":
        return regularized_heaviside(level_set, width=width)
    raise ValueError("phase must be 'inside' or 'outside'.")


def level_set_gradient_norm(
    level_set: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Return ``|grad(phi)|`` for a scalar level-set field."""

    field = _field(level_set, "level_set")
    gradient = grad(field, var=var, mode=mode)
    return DomainFunction(
        domain=gradient.domain,
        deps=gradient.deps,
        func=_GradientNorm(gradient),
        metadata=gradient.metadata,
    )


def level_set_normal(
    level_set: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
) -> DomainFunction:
    r"""Return the outward level-set normal ``grad(phi) / |grad(phi)|``.

    The convention ``phi < 0`` inside makes the normal point toward increasing
    ``phi``. ``gradient_floor`` prevents undefined values away from a regular
    zero set; callers must still verify that the interface gradient is nonzero.
    """

    field = _field(level_set, "level_set")
    floor = _positive_finite(gradient_floor, "gradient_floor")
    gradient = grad(field, var=var, mode=mode)
    return DomainFunction(
        domain=gradient.domain,
        deps=gradient.deps,
        func=_NormalizedGradient(gradient, floor),
        metadata=gradient.metadata,
    )


def level_set_curvature(
    level_set: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
) -> DomainFunction:
    r"""Return ``div(grad(phi) / |grad(phi)|)``.

    This is the sum of principal curvatures. For a signed-distance sphere of
    radius ``r`` with negative-inside convention, the value is ``(d - 1) / r``.
    """

    normal = level_set_normal(
        level_set,
        var=var,
        mode=mode,
        gradient_floor=gradient_floor,
    )
    return div(normal, var=var, mode=mode)


def level_set_normal_velocity(
    level_set: DomainFunction,
    /,
    *,
    spatial_var: str | None = None,
    time_var: str = "t",
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
) -> DomainFunction:
    r"""Return the normal interface velocity ``-partial_t(phi) / |grad(phi)|``."""

    field = _field(level_set, "level_set")
    floor = _positive_finite(gradient_floor, "gradient_floor")
    magnitude = level_set_gradient_norm(field, var=spatial_var, mode=mode)
    safe_magnitude = DomainFunction(
        domain=magnitude.domain,
        deps=magnitude.deps,
        func=UnaryFieldEvaluator(magnitude.func, _LowerBound(floor)),
        metadata=magnitude.metadata,
    )
    return -dt(field, var=time_var, mode=mode) / safe_magnitude


def level_set_coarea_density(
    level_set: DomainFunction,
    /,
    *,
    width: float,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Return ``delta_width(phi) * |grad(phi)|`` for diffuse surface integrals."""

    field = _field(level_set, "level_set")
    return regularized_delta(field, width=width) * level_set_gradient_norm(
        field,
        var=var,
        mode=mode,
    )


__all__ = [
    "level_set_coarea_density",
    "level_set_curvature",
    "level_set_gradient_norm",
    "level_set_normal",
    "level_set_normal_velocity",
    "level_set_phase_indicator",
    "regularized_delta",
    "regularized_heaviside",
]

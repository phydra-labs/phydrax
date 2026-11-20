"""Derivative bounds of traced surface carriers across kinks and jumps.

A traced carrier's derivative programs nest ``jacfwd`` through its point map,
which differentiates ``abs`` and selections piecewise: the Hessian of
``z = |u - 1/2|`` is identically zero as traced. Bounds built from those
programs must hold only where the point map's branch ledger proves the needed
regularity, fall back to first order across a kink and refuse across a jump.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._patches import AbstractSurfacePatch, LineCurve


@final
class _KinkedSurface(AbstractSurfacePatch):
    """``(u, v, |u - 1/2|)``."""

    def evaluate(self, parameters: Array, /) -> Array:
        u, v = parameters[..., 0], parameters[..., 1]
        return jnp.stack((u, v, jnp.abs(u - 0.5)), axis=-1)


@final
class _SteppedSurface(AbstractSurfacePatch):
    """``(u, v, [u > 1/2])``."""

    def evaluate(self, parameters: Array, /) -> Array:
        u, v = parameters[..., 0], parameters[..., 1]
        return jnp.stack((u, v, jnp.where(u > 0.5, 1.0, 0.0)), axis=-1)


def _square_domain(surface: AbstractSurfacePatch, /) -> MeshingDomain:
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(3, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    corners = np.asarray(surface.evaluate(jnp.asarray(uv)), dtype=np.float64)
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(0, 3),
        ),
        4,
        source_id=f"{type(surface).__name__}-square",
        source_revision="1",
        corner_points=corners,
    )


# S is 1/2 at every corner of this triangle; S(1/2, 0) = (1/2, 0, 0) lies 1/2
# below the interpolant there.
_ACROSS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
_BESIDE = ((0.625, 0.0), (1.0, 0.0), (0.625, 0.375))


def test_interpolation_bound_across_a_kink_encloses_the_true_deviation() -> None:
    domain = _square_domain(_KinkedSurface())
    bounds = domain.interpolation_bounds(
        0, np.asarray((_ACROSS, _BESIDE), dtype=np.float64)
    )
    assert 0.5 <= bounds[0] < math.inf
    # The carrier is affine beside the kink: the Hessian bound applies there.
    assert bounds[1] <= 2.0**-40


def test_adaptive_interpolation_bound_across_a_kink_encloses_the_true_deviation() -> None:
    domain = _square_domain(_KinkedSurface())
    chart = tuple((Fraction(u), Fraction(v)) for u, v in _ACROSS)
    (bound,) = domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 64),))
    assert bound is None or bound >= Fraction(1, 2)


def test_normal_turn_bound_across_a_kink_is_refused() -> None:
    domain = _square_domain(_KinkedSurface())
    turns = domain.normal_turn_bounds(0, np.asarray((_ACROSS, _BESIDE), dtype=np.float64))
    # The normal turns by pi / 2 across u = 1/2; none beside it.
    assert turns[0] == math.inf
    assert turns[1] <= 2.0**-20


def test_interpolation_bounds_across_a_jump_are_refused() -> None:
    domain = _square_domain(_SteppedSurface())
    bounds = domain.interpolation_bounds(0, np.asarray((_ACROSS,), dtype=np.float64))
    assert bounds[0] == math.inf
    chart = tuple((Fraction(u), Fraction(v)) for u, v in _ACROSS)
    assert domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 64),)) == (
        None,
    )

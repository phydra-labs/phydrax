# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Prepared rational IGA fixtures with explicit parameter-axis identities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax import geometry
from phydrax._strict import StrictModule
from phydrax.discretization.iga import (
    BSplineGrid,
    IsogeometricPlan,
    IsogeometricQuadraturePolicy,
    NURBSGeometryState,
    PreparedIsogeometricDiscretization,
)
from phydrax.discretization.iga._basis import IsogeometricFieldSpec, TensorSplineBasisSpec
from phydrax.geometry import CompiledGeometry
from phydrax.units import METER


class QuarterAnnulusFixture(StrictModule):
    plan: IsogeometricPlan
    prepared: PreparedIsogeometricDiscretization
    controls: np.ndarray


class _QuadraticKnotRefinement(StrictModule):
    knots: np.ndarray
    points: np.ndarray
    weights: np.ndarray


def _insert_quadratic_knot(
    knots: np.ndarray, controls: np.ndarray, weights: np.ndarray, value: float
) -> _QuadraticKnotRefinement:
    """Insert one knot into a quadratic NURBS curve by Boehm's rule."""
    homogeneous = np.concatenate((controls * weights[:, None], weights[:, None]), axis=1)
    span = np.searchsorted(knots, value, side="right") - 1
    refined = []
    for index in range(homogeneous.shape[0] + 1):
        if index <= span - 2:
            refined.append(homogeneous[index])
        elif index > span:
            refined.append(homogeneous[index - 1])
        else:
            alpha = (value - knots[index]) / (knots[index + 2] - knots[index])
            refined.append(
                alpha * homogeneous[index] + (1 - alpha) * homogeneous[index - 1]
            )
    points = np.asarray(refined)
    return _QuadraticKnotRefinement(
        np.sort(np.append(knots, value)), points[:, :2] / points[:, 2:], points[:, 2]
    )


def quarter_annulus_query(scale: float = 1.0) -> QuarterAnnulusFixture:
    """Prepare the query fixture on parameter axes ``r`` and ``theta``."""
    half = np.sqrt(0.5)
    refinement = _insert_quadratic_knot(
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        np.asarray(((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray((1.0, half, 1.0), dtype=np.float64),
        0.5,
    )
    radii = 1.0 + np.asarray((0.0, 0.25, 0.75, 1.0), dtype=np.float64)
    controls = scale * radii[:, None, None] * refinement.points[None, :, :]
    plan = IsogeometricPlan.isoparametric(
        (
            BSplineGrid(jnp.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)), 2),
            BSplineGrid(jnp.asarray(refinement.knots), 2),
        ),
        NURBSGeometryState(
            jnp.asarray(controls), jnp.asarray(np.ones(4)[:, None] * refinement.weights)
        ),
        axis_names=("r", "theta"),
        quadrature_policy=IsogeometricQuadraturePolicy(3),
    )
    return QuarterAnnulusFixture(plan, plan.prepare(numeric_version="annulus"), controls)


def quarter_annulus_trace(*, vector: bool = False) -> QuarterAnnulusFixture:
    """Prepare the trace fixture on parameter axes ``r`` and ``t``."""
    # The trace fixture retains its authored half-knot arithmetic and axis IDs;
    # the query fixture has a distinct theta axis and general insertion rule.
    half = np.sqrt(0.5)
    homogeneous = np.asarray(((1.0, 0.0, 1.0), (half, half, half)), dtype=np.float64)
    homogeneous = np.concatenate((homogeneous, ((0.0, 1.0, 1.0),)))
    refined = np.stack(
        (
            homogeneous[0],
            0.5 * (homogeneous[0] + homogeneous[1]),
            0.5 * (homogeneous[1] + homogeneous[2]),
            homogeneous[2],
        )
    )
    arc, arc_weights = refined[:, :2] / refined[:, 2:], refined[:, 2]
    radii = 1.0 + np.asarray((0.0, 0.25, 0.75, 1.0), dtype=np.float64)
    controls = radii[:, None, None] * arc[None, :, :]
    knots = jnp.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0))
    basis = TensorSplineBasisSpec(
        (BSplineGrid(knots, 2), BSplineGrid(knots, 2)), axis_names=("r", "t")
    )
    geometry_state = NURBSGeometryState(
        jnp.asarray(controls), jnp.asarray(np.ones(4)[:, None] * arc_weights)
    )
    field = IsogeometricFieldSpec(
        "v" if vector else "u",
        basis,
        component_shape=(2,) if vector else (),
        weights_from_geometry=True,
    )
    plan = IsogeometricPlan(
        basis, geometry_state, field, quadrature_policy=IsogeometricQuadraturePolicy(3)
    )
    return QuarterAnnulusFixture(plan, plan.prepare(numeric_version="annulus"), controls)


def annulus_region() -> CompiledGeometry:
    """Compile the first-quadrant region with radii one and two."""
    ring = geometry.Ball((0.0, 0.0), 2.0) - geometry.Ball((0.0, 0.0), 1.0)
    return (ring & geometry.Orthotope((1.0, 1.0), (2.0, 2.0))).compile(
        coordinate_unit=METER
    )


def annulus_points(count: int, seed: int) -> np.ndarray:
    """Sample the quarter annulus, including one site on each interior knot line."""
    rng = np.random.default_rng(seed)
    radius = rng.uniform(1.0, 2.0, count)
    angle = rng.uniform(0.0, 0.5 * np.pi, count)
    radius = np.append(radius, (1.5, 1.3))
    angle = np.append(angle, (0.3, 0.25 * np.pi))
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def affine_box_field(
    degree: int,
) -> tuple[PreparedIsogeometricDiscretization, np.ndarray]:
    """Prepare the [0, 2] by [0, 1] box with one interior knot per axis."""
    knots = np.concatenate((np.zeros(degree + 1), (0.5,), np.ones(degree + 1)))
    grid = BSplineGrid(jnp.asarray(knots), degree)
    greville = np.asarray(grid.greville_abscissae)
    xx, yy = np.meshgrid(2.0 * greville, greville, indexing="ij")
    plan = IsogeometricPlan.isoparametric(
        (grid, grid),
        NURBSGeometryState(jnp.asarray(np.stack((xx, yy), axis=-1)), jnp.ones(xx.shape)),
        axis_names=("xi", "eta"),
        quadrature_policy=IsogeometricQuadraturePolicy(degree + 1),
    )
    return plan.prepare(numeric_version="box"), knots

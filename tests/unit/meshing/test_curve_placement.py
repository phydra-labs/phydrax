"""Cell-local placement preserves global density residuals and native refusals."""

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.geometry._atlas import BoundaryAtlas
from phydrax.geometry._meshing_domain import PhysicalCurveMap
from phydrax.geometry.brep._patches import BSplineCurve
from phydrax.meshing.providers._native_curve import (
    CurveCharts,
    CurveSizing,
    placement_roots,
)
from phydrax.nonlinear import NonlinearStatus, NonlinearTermination


class PlacementCase(NamedTuple):
    charts: CurveCharts
    sizing: CurveSizing
    lower: float
    upper: float
    fraction: float
    nodes: Array
    weights: Array


@pytest.fixture(scope="module")
def captured_cell() -> PlacementCase:
    """Reconstruct the stored binary64 coefficients and inputs of the STEP refusal."""
    path = (
        Path(__file__).resolve().parents[2]
        / "fixtures/meshing/curve_density_placement.json"
    )
    data = json.loads(path.read_text())
    curve = BSplineCurve(
        np.asarray(data["control_points"], dtype=np.float64),
        np.asarray(data["weights"], dtype=np.float64),
        np.asarray(data["knots"], dtype=np.float64),
        data["degree"],
    )
    mapping = PhysicalCurveMap((curve,), np.asarray(data["ranges"], dtype=np.float64))
    atlas = BoundaryAtlas(
        mapping,
        source_entity_ids=np.asarray((0,), dtype=np.int32),
        source_id="captured-step-placement",
    )
    charts = CurveCharts(atlas, jnp.zeros((1,), dtype=jnp.int32))
    sizing = CurveSizing(
        np.asarray((data["uniform_size"],), dtype=np.float64),
        np.full((1,), np.inf, dtype=np.float64),
        np.zeros((1,), dtype=np.float64),
        np.full((1,), np.inf, dtype=np.float64),
        np.zeros((1,), dtype=np.float64),
    )
    nodes, weights = np.polynomial.legendre.leggauss(data["quadrature_order"])
    return PlacementCase(
        charts,
        sizing,
        data["lower"],
        data["upper"],
        data["fraction"],
        jax.device_put(nodes),
        jax.device_put(weights),
    )


def _global_integral(case: PlacementCase, parameter: float) -> float:
    """Integrate the source density on the represented global parameter interval."""
    half = 0.5 * (parameter - case.lower)
    points = case.lower + half * (case.nodes + 1.0)
    density, _ = case.sizing.density(
        case.charts, jnp.zeros(points.shape, dtype=jnp.int32), points
    )
    return float(half * jnp.sum(case.weights * density))


def _place(case: PlacementCase, fraction: float) -> tuple[Array, Array, Array]:
    return placement_roots(
        case.charts,
        case.sizing,
        jnp.asarray((case.lower,), dtype=jnp.float64),
        jnp.asarray((case.upper,), dtype=jnp.float64),
        jnp.asarray((fraction,), dtype=jnp.float64),
        case.nodes,
        case.weights,
    )


def test_captured_step_placement_cell_converges(captured_cell: PlacementCase) -> None:
    root, status, residual = _place(captured_cell, captured_cell.fraction)
    assert int(status[0]) == int(NonlinearStatus.SUCCESS)
    parameter = float(root[0])
    assert captured_cell.lower <= parameter <= captured_cell.upper
    mass = _global_integral(captured_cell, captured_cell.upper)
    target = captured_cell.fraction * mass
    global_residual = _global_integral(captured_cell, parameter) - target
    initial_residual = min(abs(target), abs(mass - target))
    termination = NonlinearTermination(absolute_residual=1.0e-12, maximum_steps=64)
    threshold = float(termination.residual_threshold(initial_residual))
    assert abs(global_residual) <= threshold
    assert abs(float(residual[0])) <= threshold
    # Both paths use the mapped global parameter. Their only difference is the
    # compiled/reduction grouping; gamma(64) bounds those binary64 operations.
    epsilon = np.finfo(np.float64).eps / 2.0
    roundoff = (64 * epsilon / (1 - 64 * epsilon)) * mass
    assert abs(float(residual[0]) - global_residual) <= roundoff


def test_nonbracketing_placement_cell_preserves_native_refusal(
    captured_cell: PlacementCase,
) -> None:
    root, status, residual = _place(captured_cell, 1.25)
    assert int(status[0]) == int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)
    assert int(status[0]) != int(NonlinearStatus.SUCCESS)
    assert np.isfinite(float(root[0])) and np.isfinite(float(residual[0]))
    mass = _global_integral(captured_cell, captured_cell.upper)
    assert -1.25 * mass < 0.0 and (1.0 - 1.25) * mass < 0.0

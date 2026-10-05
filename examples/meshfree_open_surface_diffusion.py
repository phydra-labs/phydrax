# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Diffusion on authoritative open and creased surface patches.

1. Transient diffusion ``u_t = k Lap_G u`` on a curved latitude-longitude patch
   of the unit sphere with time-dependent Dirichlet data. ``x y`` is a degree-2
   spherical harmonic, so ``u = exp(-6 k t) x y`` is an independent exact
   solution. Node measures come from authoritative chart cubature; boundary
   quadrature and conormals come from the declared chart box. The discrete
   balance ``d/dt int u = int_dGamma k du/dnu`` is audited at every step.
2. A sheet folded along a sharp crease is two smooth patches joined by an
   oriented interface (continuity plus conormal-flux balance, no normal
   averaging). The folded sheet is intrinsically flat, so the unfolded
   coordinate gives an exact manufactured solution.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.linalg as la
from phydrax.discretization.meshfree import (
    chart_box_atlas,
    ChartSurfaceGeometry,
    LocalStencilPolicy,
    PreparedSurfacePointCloud,
    SurfaceEllipticSystem,
    SurfacePatchInterface,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
)
from phydrax.metrix import CoordinateChart, EmbeddedChart


WorkflowMetric = float | int | bool | str


class CreaseReport(TypedDict):
    nodes: int
    solver_successful: bool
    iterations: int
    maximum_error: float
    seam_jump: float
    normal_angle_across_crease: float


PATCH_LOWER = (0.0, -0.6)
PATCH_UPPER = (1.2, 0.6)


def _latlong(u: Array) -> Array:
    return jnp.stack(
        (jnp.cos(u[1]) * jnp.cos(u[0]), jnp.cos(u[1]) * jnp.sin(u[0]), jnp.sin(u[1]))
    )


LATLONG_CHART = EmbeddedChart(CoordinateChart("latlong", ("lambda", "mu")), _latlong, 3)


def box_samples(
    lower: tuple[float, float], upper: tuple[float, float], count: int, seed: int
) -> np.ndarray:
    """Jittered tensor samples with exact boundary rows on the box faces."""
    if count < 5:
        raise ValueError("Each box direction needs at least five samples.")
    low, high = np.asarray(lower), np.asarray(upper)
    first, second = np.meshgrid(
        np.linspace(low[0], high[0], count),
        np.linspace(low[1], high[1], count),
        indexing="ij",
    )
    spacing = (high - low) / (count - 1)
    inside = (first > low[0]) & (first < high[0]) & (second > low[1]) & (second < high[1])
    jitter = np.random.default_rng(seed).uniform(-0.2, 0.2, (2, *first.shape))
    first = first + np.where(inside, jitter[0] * spacing[0], 0)
    second = second + np.where(inside, jitter[1] * spacing[1], 0)
    return np.column_stack((first.ravel(), second.ravel()))


def chart_patch(
    chart: EmbeddedChart,
    lower: tuple[float, float],
    upper: tuple[float, float],
    *,
    count: int,
    seed: int = 0,
    neighbors: int = 20,
    degree: int = 3,
) -> PreparedSurfacePointCloud:
    """Open authoritative chart patch with chart-box boundary and cubature."""
    coordinates = box_samples(lower, upper, count, seed)
    geometry = ChartSurfaceGeometry(
        chart, coordinates, closed=False, geometry_id=f"{chart.chart.name}-patch"
    )
    points = geometry.embed(geometry.chart_indices, geometry.chart_coordinates)
    return SurfacePointCloudPlan(
        points,
        geometry,
        neighbors,
        quadrature=SurfaceQuadraturePolicy(
            "chart-cubature",
            atlas=chart_box_atlas(chart, lower, upper, source_id=chart.chart.name),
            subdivisions=max(8, count),
            reconstruction_neighbors=12,
        ),
        boundary=geometry.box_boundary(lower, upper),
        stencil_policy=LocalStencilPolicy(polynomial_degree=degree, chunk_rows=128),
    ).prepare()


def _policy() -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.GMRES(restart=300),
        tolerance=la.TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=3000),
    )


def run_open_patch(
    *, count: int = 17, steps: int = 20, time: float = 0.1, diffusivity: float = 1.0
) -> dict[str, WorkflowMetric]:
    patch = chart_patch(LATLONG_CHART, PATCH_LOWER, PATCH_UPPER, count=count)
    boundary = patch.boundary
    if boundary is None:
        raise RuntimeError("The open patch must carry its declared boundary.")
    dt = time / steps
    system = SurfaceEllipticSystem(
        (patch,), diffusivity=diffusivity, reaction=1 / dt, system_id="open-latlong"
    )
    prepared = la.prepare(system.linear_system, _policy())
    harmonic = patch.points[:, 0] * patch.points[:, 1]
    conormal = patch.conormal_derivative
    value = harmonic
    balance, successful, iterations = 0.0, True, 0
    for step in range(1, steps + 1):
        boundary_data = math.exp(-6 * diffusivity * step * dt) * harmonic
        result = la.solve(prepared, system.rhs((value / dt,), (boundary_data,)))
        successful &= bool(result.successful)
        iterations = max(iterations, int(result.diagnostics.iterations))
        new = system.split(result.value)[0]
        rate = jnp.sum(patch.measures * (new - value)) / dt
        flux = diffusivity * jnp.sum(boundary.measures * conormal.mv(new))
        balance = max(balance, abs(float(rate - flux)) / max(abs(float(flux)), 1e-30))
        value = new
    exact = math.exp(-6 * diffusivity * time) * harmonic
    area = 1.2 * 2 * math.sin(0.6)
    perimeter = 2 * 1.2 * math.cos(0.6) + 2 * 1.2
    return {
        "nodes": int(patch.points.shape[0]),
        "steps": steps,
        "solver_successful": successful,
        "maximum_iterations": iterations,
        "final_relative_error": float(
            jnp.sqrt(
                jnp.sum(patch.measures * (value - exact) ** 2)
                / jnp.sum(patch.measures * exact**2)
            )
        ),
        "temporal_error_scale": 18 * diffusivity**2 * dt,
        "area_error": abs(float(patch.quadrature_evidence.total_area) - area),
        "atlas_area_error": abs(float(patch.quadrature_evidence.reference_area) - area),
        "perimeter_error": abs(float(jnp.sum(boundary.measures)) - perimeter),
        "measure_transfer_residual": float(patch.quadrature_evidence.transfer_residual),
        "maximum_balance_defect": balance,
        "geometry_source": patch.geometry_evidence.source,
    }


def _flat(u: Array) -> Array:
    return jnp.stack((u[0], u[1], jnp.zeros_like(u[0])))


def _folded(angle: float) -> EmbeddedChart:
    def embedding(u: Array) -> Array:
        return jnp.stack((u[0] * math.cos(angle), u[1], u[0] * math.sin(angle)))

    return EmbeddedChart(CoordinateChart("folded", ("t", "y")), embedding, 3)


def crease_exact(arclength: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(0.5 * math.pi * arclength + 0.3) * np.cos(y)


def run_crease(*, count: int = 13, angle: float = math.pi / 3) -> CreaseReport:
    """``-Lap u + u = f`` on two folded flat patches joined along the crease."""
    flat = EmbeddedChart(CoordinateChart("flat", ("s", "y")), _flat, 3)
    pieces = (
        (flat, (-1.0, 0.0), (0.0, 1.0), 1),
        (_folded(angle), (0.0, 0.0), (1.0, 1.0), 2),
    )
    patches, coordinates = [], []
    for chart, lower, upper, seed in pieces:
        patch = chart_patch(chart, lower, upper, count=count, seed=seed, neighbors=16)
        geometry = patch.plan.geometry
        if not isinstance(geometry, ChartSurfaceGeometry):
            raise RuntimeError("Crease pieces are authoritative chart patches.")
        patches.append(patch)
        coordinates.append(np.asarray(geometry.chart_coordinates))
    seam = []
    for values in coordinates:
        # Crease nodes away from the outer corners; corners keep Dirichlet data.
        rows = np.flatnonzero(
            (np.abs(values[:, 0]) < 1e-12)
            & (values[:, 1] > 1e-12)
            & (values[:, 1] < 1 - 1e-12)
        )
        seam.append(rows[np.argsort(values[rows, 1])])
    interface = SurfacePatchInterface(0, seam[0], 1, seam[1], interface_id="crease")
    system = SurfaceEllipticSystem(
        patches, (interface,), diffusivity=1.0, reaction=1.0, system_id="crease"
    )
    exact = [
        jnp.asarray(crease_exact(values[:, 0], values[:, 1])) for values in coordinates
    ]
    shift = (0.5 * math.pi) ** 2 + 2
    result = la.solve(
        system.linear_system,
        system.rhs([shift * value for value in exact], exact),
        policy=_policy(),
    )
    solution = system.split(result.value)
    normals = [patch.normals for patch in patches]
    return {
        "nodes": int(sum(patch.points.shape[0] for patch in patches)),
        "solver_successful": bool(result.successful),
        "iterations": int(result.diagnostics.iterations),
        "maximum_error": max(
            float(jnp.max(jnp.abs(value - reference)))
            for value, reference in zip(solution, exact, strict=True)
        ),
        "seam_jump": float(jnp.max(jnp.abs(solution[0][seam[0]] - solution[1][seam[1]]))),
        "normal_angle_across_crease": float(
            jnp.arccos(
                jnp.clip(jnp.sum(normals[0][seam[0]] * normals[1][seam[1]], -1), -1, 1)
            ).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=17)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "open_patch": run_open_patch(count=args.count, steps=args.steps),
                "crease": run_crease(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

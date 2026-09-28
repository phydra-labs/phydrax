#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Threshold-dynamics qualification campaigns at feasible CPU scale.

Scenarios (``--scenario`` selects a subset; default runs all):

- ``circle`` / ``sphere``: two-phase mean-curvature flow rates under joint
  refinement (curve shortening ``dA/dt = -2 pi``; sphere ``d(r^2)/dt = -4``);
- ``herring``: triple-junction angles for tensions ``(1, 1, sqrt 2)``;
- ``pressure``: research gate for the auction-price normalization
  ``P = -sqrt(pi) p`` against the Laplace law on circles and spheres;
- ``bubble-diffusion``: isolated-bubble gas diffusion ``dA/dt = -2 pi sigma k_eff``;
- ``von-neumann``: 2D grain growth and gas-diffusive foam coarsening rates versus
  ``(pi/3) c (n - 6)``;
- ``grain-statistics-2d`` / ``grain-statistics-3d``: sparse-route grain growth
  topology versus Euler (mean sides 6) and Mason, Lazar, MacPherson and Srolovitz
  (Phys. Rev. E 92, 063308, 2015: mean faces 13.766 +- 0.009);
- ``kelvin-weaire-phelan``: explicit periodic BCC/A15 foams, exact discrete
  volumes and zero-kernel-width surface costs ``A / V^(2/3)`` versus Kelvin
  (5.306) and Weaire-Phelan (5.288; Weaire and Phelan, Phil. Mag. Lett. 69,
  107, 1994);
- ``sparse-scaling``: sparse-route cost varying sites and declared labels
  separately;
- ``mesh``: mesh heat-action route against geodesic-curvature flow on the sphere,
  with time-step, Taylor-tolerance, mass-lumping and spatial-error isolation.

Every record carries runtime identity, discretization, measurements, the pinned
reference, the criterion and whether it passed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from scipy import sparse as scipy_sparse
from scipy.sparse import linalg as scipy_sparse_linalg
from scipy.spatial import ConvexHull, Voronoi

import phydrax as phx
import phydrax.threshold_dynamics as td
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix
from phydrax.linalg import TaylorExponentialPolicy


ROOT2 = float(np.sqrt(2.0))
KELVIN_COST = 5.306
WEAIRE_PHELAN_COST = 5.288
MEAN_FACES_3D = 13.766


class _PeriodicSeedEvidence(TypedDict):
    region_volumes: list[float]
    face_counts: list[int]
    total_volume: float
    volume_residual: float
    planar_seed_cost: float


class _MeshRow(TypedDict):
    level: int
    vertices: int
    h: float
    dt: float
    dt_over_h: float
    steps: int
    time: float
    cos_theta: float
    predicted: float
    error: float
    polyhedral_area: float
    sphere_area_relative_defect: float
    status: int
    committed_steps: int
    successful: bool
    heat_error_estimate: float
    requested_tolerance: NotRequired[float]


@eqx.filter_jit
def _run(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState, steps: int
) -> td.ThresholdDynamicsRunResult:
    return prepared.run(state, steps)


@eqx.filter_jit
def _energy(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState
) -> jax.Array:
    return prepared.energy(state)


@eqx.filter_jit
def _coarsen(
    coarsening: td.GasDiffusionCoarsening, state: td.LabelFieldState, steps: int
) -> td.GasDiffusionRunResult:
    return coarsening.run(state, steps)


def _runtime_identity() -> dict[str, object]:
    build_id = canonical_fingerprint(
        {
            "kind": "threshold-dynamics-qualification-build",
            "phydrax": importlib.metadata.version("phydrax"),
            "phydrax_path": str(Path(phx.__file__).resolve().parent),
        }
    )
    environment_id = canonical_fingerprint(
        {
            "kind": "threshold-dynamics-qualification-environment",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        }
    )
    identity = phx.qualification.QualificationRuntimeIdentity(
        build_id,
        environment_id,
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return dict(identity.to_record())


def _uniform_plan(
    count: int,
    dt: float,
    *,
    constraint: td.LabelVolumeConstraint | None = None,
    mobility: float = 1.0,
) -> td.ThresholdDynamicsPlan:
    labels = tuple(f"label{index}" for index in range(count))
    return td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, mobility, structure="uniform"),
        dt,
        volume_constraint=constraint,
        minimum_resolution_ratio=0.0,
    )


def _centers(n: int, dimension: int) -> list[np.ndarray]:
    x = (np.arange(n) + 0.5) / n
    return list(np.meshgrid(*([x] * dimension), indexing="ij"))


def _ball(n: int, dimension: int, radius: float) -> np.ndarray:
    squared = sum((axis - 0.5) ** 2 for axis in _centers(n, dimension))
    return np.where(squared < radius**2, 0, 1)


def _periodic_voronoi(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    nearest = np.empty((points.shape[0],), dtype=np.int64)
    for start in range(0, points.shape[0], 8_192):
        chunk = points[start : start + 8_192]
        offsets = np.abs(chunk[:, None, :] - seeds[None])
        offsets = np.minimum(offsets, 1.0 - offsets)
        nearest[start : start + 8_192] = np.argmin(np.sum(offsets**2, axis=-1), axis=1)
    return nearest


def _grid_voronoi(n: int, dimension: int, seeds: np.ndarray) -> np.ndarray:
    points = np.stack(_centers(n, dimension), axis=-1).reshape(-1, dimension)
    return _periodic_voronoi(points, seeds).reshape((n,) * dimension)


def _history(result: td.ThresholdDynamicsRunResult, initial: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (initial[None], np.asarray(result.evidence.label_counts, dtype=np.float64))
    )


def circle() -> dict[str, object]:
    rows = []
    errors: list[float] = []
    for n, dt in ((128, 4e-3), (256, 2e-3), (512, 1e-3), (1024, 5e-4)):
        steps = int(round(0.024 / dt))
        prepared = _uniform_plan(2, dt).prepare(
            td.PeriodicGridHeatKernel((n, n), (1.0, 1.0))
        )
        labels = _ball(n, 2, 0.3)
        result = _run(prepared, prepared.initial_state(labels), steps)
        areas = _history(result, np.bincount(labels.reshape(-1)))[:, 0] / n**2
        slope = float(np.polyfit(dt * np.arange(steps + 1), areas, 1)[0])
        errors.append(abs(slope / (-2.0 * np.pi) - 1.0))
        status = int(result.status)
        committed_steps = int(result.committed_steps)
        rows.append(
            {
                "grid": n,
                "dt": dt,
                "steps": steps,
                "committed_steps": committed_steps,
                "rate": slope,
                "relative_error": errors[-1],
                "status": status,
                "successful": status == int(td.ThresholdDynamicsStatus.SUCCESS)
                and committed_steps == steps,
            }
        )
    orders = [float(np.log2(errors[k] / errors[k + 1])) for k in range(len(errors) - 1)]
    successful = all(row["successful"] for row in rows)
    return {
        "reference": "curve shortening dA/dt = -2 pi mu sigma",
        "refinement": "joint h and dt halving (pinning ratio dt/(R h) fixed)",
        "rows": rows,
        "observed_orders": orders,
        "criterion": "finest relative error < 1% and finest < coarsest",
        "successful": successful,
        "passed": bool(successful and errors[-1] < 0.01 and errors[-1] < errors[0]),
    }


def sphere() -> dict[str, object]:
    rows = []
    errors: list[float] = []
    for n, dt in ((32, 4e-3), (64, 2e-3), (128, 1e-3)):
        steps = int(round(0.016 / dt))
        prepared = _uniform_plan(2, dt).prepare(
            td.PeriodicGridHeatKernel((n,) * 3, (1.0,) * 3)
        )
        labels = _ball(n, 3, 0.3)
        result = _run(prepared, prepared.initial_state(labels), steps)
        volumes = _history(result, np.bincount(labels.reshape(-1)))[:, 0] / n**3
        squared = (3.0 * volumes / (4.0 * np.pi)) ** (2.0 / 3.0)
        slope = float(np.polyfit(dt * np.arange(steps + 1), squared, 1)[0])
        errors.append(abs(slope / -4.0 - 1.0))
        status = int(result.status)
        committed_steps = int(result.committed_steps)
        rows.append(
            {
                "grid": n,
                "dt": dt,
                "steps": steps,
                "committed_steps": committed_steps,
                "rate": slope,
                "relative_error": errors[-1],
                "status": status,
                "successful": status == int(td.ThresholdDynamicsStatus.SUCCESS)
                and committed_steps == steps,
            }
        )
    successful = all(row["successful"] for row in rows)
    return {
        "reference": "mean curvature flow of a sphere d(r^2)/dt = -4 mu sigma",
        "rows": rows,
        "criterion": "finest relative error < 2%",
        "successful": successful,
        "passed": bool(successful and errors[-1] < 0.02),
    }


def _herring_labels(n: int) -> np.ndarray:
    x, _ = _centers(n, 2)
    field = np.where(x < 0.5, 1, 2)
    return np.where(_ball(n, 2, 0.2) == 0, 0, field)


def herring() -> dict[str, object]:
    labels = ("lens", "left", "right")
    sigma = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, ROOT2], [1.0, ROOT2, 0.0]])
    mobility = np.where(sigma > 0.0, 1.0 / np.where(sigma > 0.0, sigma, 1.0), 0.0)
    rows = []
    errors: list[float] = []
    for n, dt, steps in ((256, 1e-3, 10), (512, 5e-4, 20), (1024, 2.5e-4, 40)):
        plan = td.ThresholdDynamicsPlan(
            InterfaceTensionMatrix(labels, sigma),
            InterfaceMobilityMatrix(labels, mobility),
            dt,
        )
        prepared = plan.prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))
        result = _run(prepared, prepared.initial_state(_herring_labels(n)), steps)
        final = np.asarray(result.state.labels)
        angles = [
            td.triple_junction_angles(
                final,
                (1.0, 1.0),
                (0, 1, 2),
                center=center,
                inner_radius=0.02,
                outer_radius=0.06,
            ).tolist()
            for center in ((0.5, 0.7), (0.5, 0.3))
        ]
        error = float(np.max(np.abs(np.asarray(angles) - [90.0, 135.0, 135.0])))
        errors.append(error)
        status = int(result.status)
        committed_steps = int(result.committed_steps)
        rows.append(
            {
                "grid": n,
                "dt": dt,
                "steps": steps,
                "committed_steps": committed_steps,
                "angles": angles,
                "maximum_error_degrees": error,
                "status": status,
                "successful": status == int(td.ThresholdDynamicsStatus.SUCCESS)
                and committed_steps == steps,
            }
        )
    successful = all(row["successful"] for row in rows)
    return {
        "reference": "Herring/Young sine law: (90, 135, 135) degrees",
        "rows": rows,
        "criterion": "maximum angle error < 5 degrees at the finest grid",
        "successful": successful,
        "passed": bool(successful and errors[-1] < 5.0),
    }


def _price_ratio(
    dimension: int, n: int, dt: float, radius: float, steps: int
) -> tuple[dict[str, object], float]:
    labels = _ball(n, dimension, radius)
    counts = np.bincount(labels.reshape(-1))
    plan = _uniform_plan(2, dt, constraint=td.LabelVolumeConstraint(counts))
    prepared = plan.prepare(
        td.PeriodicGridHeatKernel((n,) * dimension, (1.0,) * dimension)
    )
    result = _run(prepared, prepared.initial_state(labels), steps)
    volume = result.evidence.volume
    if volume is None:
        raise RuntimeError("volume evidence missing")
    prices = np.asarray(volume.prices)
    jump = np.sqrt(np.pi) * (prices[:, 1] - prices[:, 0])
    fraction = counts[0] / n**dimension
    effective = (
        np.sqrt(fraction / np.pi)
        if dimension == 2
        else (3.0 * fraction / (4.0 * np.pi)) ** (1 / 3)
    )
    laplace = (dimension - 1) / effective
    mean = float(np.mean(jump / laplace))
    status = int(result.status)
    committed_steps = int(result.committed_steps)
    auction_status = np.asarray(volume.auction_status)
    return {
        "dimension": dimension,
        "grid": n,
        "dt": dt,
        "radius": float(effective),
        "pinning_ratio": dt * (dimension - 1) / effective / (1.0 / n),
        "steps": steps,
        "committed_steps": committed_steps,
        "per_step_ratio": (jump / laplace).tolist(),
        "mean_ratio": mean,
        "status": status,
        "auction_status": auction_status.tolist(),
        "successful": status == int(td.ThresholdDynamicsStatus.SUCCESS)
        and committed_steps == steps
        and bool(np.all(auction_status == 0)),
    }, mean


def pressure() -> dict[str, object]:
    resolved = [_price_ratio(2, 512, 4e-4, radius, 4) for radius in (0.1, 0.15, 0.2)] + [
        _price_ratio(3, 64, 1e-3, radius, 3) for radius in (0.2, 0.3)
    ]
    pinned = [_price_ratio(2, 512, 1e-4, 0.3, 4)]
    successful = all(record["successful"] for record, _ in (*resolved, *pinned))
    return {
        "reference": "Laplace law P_in - P_out = sigma (d - 1) / R with P = -sqrt(pi) p",
        "resolved_regime": [record for record, _ in resolved],
        "pinned_regime": [record for record, _ in pinned],
        "criterion": "resolved regime (tau kappa >= h): every mean ratio within 10% of 1",
        "successful": successful,
        "passed": bool(
            successful and all(abs(ratio - 1.0) <= 0.1 for _, ratio in resolved)
        ),
    }


def bubble_diffusion() -> dict[str, object]:
    n, dt, steps = 256, 1e-3, 16
    labels = _ball(n, 2, 0.2)
    rows = []
    errors: list[float] = []
    for permeance in (0.25, 1.0):
        prepared = _uniform_plan(
            2, dt, constraint=td.LabelVolumeConstraint(np.bincount(labels.reshape(-1)))
        ).prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))
        result = _coarsen(
            td.GasDiffusionCoarsening(prepared, permeance),
            prepared.initial_state(labels),
            steps,
        )
        areas = np.asarray(result.evidence.step.label_counts)[:, 0] / n**2
        slope = float(np.polyfit(dt * np.arange(1, steps + 1), areas, 1)[0])
        expected = -2.0 * np.pi * permeance / (permeance + 1.0)
        errors.append(abs(slope / expected - 1.0))
        statuses = np.asarray(result.evidence.step.status)
        successful = statuses.size == steps and bool(
            np.all(statuses == int(td.ThresholdDynamicsStatus.SUCCESS))
        )
        rows.append(
            {
                "permeance": permeance,
                "rate": slope,
                "expected": expected,
                "relative_error": errors[-1],
                "step_status": statuses.tolist(),
                "successful": successful,
            }
        )
    successful = all(row["successful"] for row in rows)
    return {
        "reference": "isolated bubble dA/dt = -2 pi sigma k mu / (k + mu)",
        "rows": rows,
        "criterion": "relative error < 5%",
        "successful": successful,
        "passed": bool(successful and all(error < 0.05 for error in errors)),
    }


def _von_neumann_window(
    history: np.ndarray,
    sides: np.ndarray,
    dt: float,
    start: int,
    stop: int,
    final_sides: np.ndarray,
) -> tuple[float, float, int]:
    stable = (sides == final_sides) & np.all(history[start:stop] > 0.0, axis=0)
    selected = history[start:stop][:, stable]
    slope, crossover = td.von_neumann_mullins_fit(
        dt * np.arange(start, stop), selected, sides[stable].astype(np.float64)
    )
    return slope, crossover, int(np.count_nonzero(stable))


def _trend(slope: float, expected: float, crossover: float) -> bool:
    return abs(slope / expected - 1.0) <= 0.3 and abs(crossover - 6.0) <= 1.0


def von_neumann() -> dict[str, object]:
    rows = []
    checks: list[bool] = []
    # Grain growth on the sparse route: 600 grains on 512^2, sqrt(tau) = 3 h, a
    # 20-step warm-up, then a 12-step window over topologically stable grains.
    depth, cells, window = 9, 600, 12
    resolution = 2**depth
    dt = (3.0 / resolution) ** 2
    grid = _sparse_grid(depth, 2, 12, np.float32, 8)
    prepared = _uniform_plan(cells, dt).prepare(grid, candidate_capacity=32)
    sites = np.asarray(grid.site_coordinates, dtype=np.int64)
    seeds = np.random.default_rng(21).random((cells, 2))
    labels0 = _periodic_voronoi((sites + 0.5) / resolution, seeds)
    warm = _run(prepared, prepared.initial_state(labels0), 20)
    later = _run(prepared, warm.state, window)

    def as_grid(values: np.ndarray, /) -> np.ndarray:
        field = np.zeros((resolution, resolution), dtype=np.int64)
        field[sites[:, 0], sites[:, 1]] = values
        return field

    start_labels = np.asarray(warm.state.labels)
    history = (
        np.concatenate(
            (
                np.bincount(start_labels, minlength=cells)[None],
                np.asarray(later.evidence.label_counts),
            )
        ).astype(np.float64)
        / resolution**2
    )
    sides = td.label_neighbor_counts(as_grid(start_labels), cells, minimum_contact=3)
    final_sides = td.label_neighbor_counts(
        as_grid(np.asarray(later.state.labels)), cells, minimum_contact=3
    )
    stable = (sides == final_sides) & np.all(history > 0.0, axis=0)
    times = dt * np.arange(window + 1)
    rates = np.polyfit(times, history[:, stable], 1)[0]
    coefficients, covariance = np.polyfit(
        sides[stable].astype(np.float64), rates, 1, cov=True
    )
    slope = float(coefficients[0])
    crossover = float(-coefficients[1] / coefficients[0])
    checks.append(_trend(slope, float(np.pi / 3.0), crossover))
    grain_successful = bool(
        int(warm.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
        and int(warm.committed_steps) == 20
        and int(later.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
        and int(later.committed_steps) == window
    )
    rows.append(
        {
            "model": "grain-growth",
            "passed": bool(grain_successful and checks[-1]),
            "successful": grain_successful,
            "route": "sparse",
            "grid": resolution,
            "cells": cells,
            "dt": dt,
            "sqrt_tau_over_h": 3.0,
            "slope": slope,
            "slope_standard_error": float(np.sqrt(covariance[0, 0])),
            "expected_slope": float(np.pi / 3.0),
            "crossover_sides": crossover,
            "cells_used": int(np.count_nonzero(stable)),
            "cells_per_side_count": np.bincount(sides[stable]).tolist(),
            "warm_committed_steps": int(warm.committed_steps),
            "measured_committed_steps": int(later.committed_steps),
            "worst_status": int(max(int(warm.status), int(later.status))),
        }
    )

    n, cells, dt, permeance = 128, 24, (2.5 / 128) ** 2, 0.25
    seeds = np.random.default_rng(7).random((cells, 2))
    field = _grid_voronoi(n, 2, seeds)
    counts = np.bincount(field.reshape(-1), minlength=cells)
    constrained = _uniform_plan(
        cells, dt, constraint=td.LabelVolumeConstraint(counts)
    ).prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))
    relaxation = _run(constrained, constrained.initial_state(field), 5)
    relaxed = relaxation.state
    coarsening = _coarsen(td.GasDiffusionCoarsening(constrained, permeance), relaxed, 40)
    history = (
        np.concatenate(
            (counts[None], np.asarray(coarsening.evidence.step.label_counts))
        ).astype(np.float64)
        / n**2
    )
    sides = td.label_neighbor_counts(np.asarray(relaxed.labels), cells, minimum_contact=3)
    final_sides = td.label_neighbor_counts(
        np.asarray(coarsening.state.labels), cells, minimum_contact=3
    )
    slope, crossover, used = _von_neumann_window(history, sides, dt, 3, 41, final_sides)
    effective = permeance / (permeance + 1.0)
    checks.append(_trend(slope, float(np.pi * effective / 3.0), crossover))
    coarsening_status = np.asarray(coarsening.evidence.step.status)
    foam_successful = bool(
        int(relaxation.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
        and int(relaxation.committed_steps) == 5
        and coarsening_status.size == 40
        and np.all(coarsening_status == int(td.ThresholdDynamicsStatus.SUCCESS))
    )
    rows.append(
        {
            "model": "gas-diffusion-foam",
            "passed": bool(foam_successful and checks[-1]),
            "successful": foam_successful,
            "grid": n,
            "cells": cells,
            "dt": dt,
            "permeance": permeance,
            "slope": slope,
            "expected_slope": float(np.pi * effective / 3.0),
            "crossover_sides": crossover,
            "cells_used": used,
            "relaxation_committed_steps": int(relaxation.committed_steps),
            "coarsening_step_status": coarsening_status.tolist(),
        }
    )
    successful = all(row["successful"] for row in rows)
    return {
        "reference": "von Neumann-Mullins dA/dt = (pi/3) c (n - 6)",
        "rows": rows,
        "criterion": "trend: slope within 30% and crossover within 1 side of 6",
        "successful": successful,
        "passed": bool(successful and all(checks)),
    }


def _sparse_grid(
    depth: int, dimension: int, radius: int, dtype: DTypeLike, brick: int = 8
) -> td.SparseLabelGrid:
    resolution = 2**depth
    address = MortonAddressPlan(
        (0.0,) * dimension, (1.0,) * dimension, depth, periodic_axes=(True,) * dimension
    )
    coordinates = np.asarray(
        tuple(itertools.product(range(resolution), repeat=dimension))
    )
    return td.SparseLabelGrid(
        address,
        coordinates,
        brick_size=brick,
        brick_capacity=(resolution // brick) ** dimension,
        stencil_radius=radius,
        dtype=dtype,
    )


def _sparse_growth(
    depth: int,
    dimension: int,
    grains: int,
    steps: int,
    capacity: int,
    radius: int,
    brick: int,
    root_tau: float = 1.2,
) -> tuple[np.ndarray, td.ThresholdDynamicsRunResult, float, int]:
    resolution = 2**depth
    grid = _sparse_grid(depth, dimension, radius, np.float32, brick)
    dt = (root_tau / resolution) ** 2
    prepared = _uniform_plan(grains, dt).prepare(grid, candidate_capacity=capacity)
    sites = np.asarray(grid.site_coordinates, dtype=np.int64)
    seeds = np.random.default_rng(13).random((grains, dimension))
    labels = _periodic_voronoi((sites + 0.5) / resolution, seeds)
    begin = time.perf_counter()
    result = _run(prepared, prepared.initial_state(labels), steps)
    jax.block_until_ready(result.state.labels)
    final = np.zeros((resolution,) * dimension, dtype=np.int64)
    final[tuple(sites.T)] = np.asarray(result.state.labels)
    return final, result, time.perf_counter() - begin, prepared.working_bytes


def _topology(final: np.ndarray, contact: int) -> tuple[np.ndarray, int]:
    present = np.unique(final)
    compact = np.searchsorted(present, final)
    return td.label_neighbor_counts(
        compact, present.size, minimum_contact=contact
    ), present.size


def grain_statistics_2d() -> dict[str, object]:
    # A 4-site brick keeps the (brick + 2 radius)^2 candidate halo local enough
    # for capacity 32 at this grain density. The prior 8-site brick required more
    # than 32 labels in at least one halo and correctly failed before step one.
    steps, initial, capacity, brick = 100, 3000, 32, 4
    final, result, seconds, working_bytes = _sparse_growth(
        9, 2, initial, steps, capacity, 12, brick, 3.0
    )
    sides, grains = _topology(final, 2)
    mean = float(np.mean(sides))
    deviation = float(np.std(sides, ddof=1))
    standard_error = deviation / np.sqrt(sides.size)
    sparse = result.evidence.sparse
    if sparse is None:
        raise RuntimeError("sparse evidence missing")
    successful = bool(
        int(result.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
        and int(result.committed_steps) == steps
        and not np.any(np.asarray(sparse.candidate_overflow))
    )
    return {
        "reference": "Euler: mean sides of a periodic trivalent network = 6",
        "grid": 512,
        "sqrt_tau_over_h": 3.0,
        "initial_grains": initial,
        "final_grains": grains,
        "steps": steps,
        "committed_steps": int(result.committed_steps),
        "mean_sides": mean,
        "side_standard_deviation": deviation,
        "standard_error": standard_error,
        "mean_sides_95_percent_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "side_histogram": np.bincount(sides).tolist(),
        "brick_size": brick,
        "candidate_capacity": capacity,
        "maximum_candidates": int(np.max(np.asarray(sparse.required_candidates))),
        "overflowed_sites": int(np.max(np.asarray(sparse.overflowed_sites))),
        "truncated_kernel_mass": float(np.max(np.asarray(sparse.truncated_kernel_mass))),
        "kernel_symbol_minimum": float(np.min(np.asarray(sparse.kernel_symbol_minimum))),
        "working_bytes": working_bytes,
        "worst_status": int(result.status),
        "seconds": seconds,
        "criterion": "all steps commit, grains coarsen below 70%, |mean sides - 6| <= 0.3",
        "successful": successful,
        "passed": bool(
            successful
            and grains <= 0.7 * initial
            and abs(mean - 6.0) <= 0.3
        ),
    }


def grain_statistics_3d() -> dict[str, object]:
    steps = 40
    final, result, seconds, _ = _sparse_growth(6, 3, 500, steps, 32, 4, 4)
    faces, grains = _topology(final, 3)
    mean = float(np.mean(faces))
    error = float(np.std(faces) / np.sqrt(faces.size))
    sparse = result.evidence.sparse
    if sparse is None:
        raise RuntimeError("sparse evidence missing")
    successful = bool(
        int(result.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
        and int(result.committed_steps) == steps
        and not np.any(np.asarray(sparse.candidate_overflow))
    )
    return {
        "reference": f"steady-state mean faces {MEAN_FACES_3D} +- 0.009 (Mason et al. 2015)",
        "grid": 64,
        "initial_grains": 500,
        "final_grains": grains,
        "steps": steps,
        "committed_steps": int(result.committed_steps),
        "mean_faces": mean,
        "standard_error": error,
        "maximum_candidates": int(np.max(np.asarray(sparse.required_candidates))),
        "truncated_kernel_mass": float(np.max(np.asarray(sparse.truncated_kernel_mass))),
        "worst_status": int(result.status),
        "seconds": seconds,
        "criterion": "within 3 standard errors + 0.5 of the reference (transient, small sample)",
        "successful": successful,
        "passed": bool(
            successful and abs(mean - MEAN_FACES_3D) <= 3.0 * error + 0.5
        ),
    }


def _foam_cell_cost(
    energy: Sequence[float] | np.ndarray, cell_count: int, domain_volume: float = 1.0
) -> np.ndarray:
    """Nondimensional mean cell area from energy that counts each film once."""
    values = np.asarray(energy, dtype=np.float64)
    cell_volume = domain_volume / cell_count
    mean_cell_area = 2.0 * values / cell_count
    return mean_cell_area / cell_volume ** (2.0 / 3.0)


def _zero_width_cost(
    kernel_widths: Sequence[float] | np.ndarray,
    finite_width_costs: Sequence[float] | np.ndarray,
    /,
) -> tuple[float, float]:
    """Linear heat-content perimeter extrapolation and fit standard error."""
    coefficients, covariance = np.polyfit(
        np.asarray(kernel_widths, dtype=np.float64),
        np.asarray(finite_width_costs, dtype=np.float64),
        1,
        cov=True,
    )
    return float(coefficients[1]), float(np.sqrt(covariance[1, 1]))


def _kelvin_bcc_seeds() -> np.ndarray:
    """Two-by-two-by-two BCC supercell (16 truncated-octahedron cells)."""
    basis = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)), dtype=np.float64)
    return np.concatenate(
        [
            (basis + np.asarray(shift, dtype=np.float64)) / 2.0
            for shift in itertools.product((0, 1), repeat=3)
        ]
    )


def _weaire_phelan_a15_seeds() -> np.ndarray:
    """A15 cubic cell: two 12-faced and six 14-faced Voronoi regions."""
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.25, 0.5, 0.0),
            (0.75, 0.5, 0.0),
            (0.0, 0.25, 0.5),
            (0.0, 0.75, 0.5),
            (0.5, 0.0, 0.25),
            (0.5, 0.0, 0.75),
        ),
        dtype=np.float64,
    )


def _periodic_seed_evidence(
    seeds: np.ndarray, expected_faces: Sequence[int]
) -> _PeriodicSeedEvidence:
    """Independent periodic Voronoi volume/topology certificate for a seed cell."""
    shifts = np.asarray(tuple(itertools.product((-1, 0, 1), repeat=3)), dtype=np.float64)
    tiled = np.concatenate([seeds + shift for shift in shifts])
    tiled_shifts = np.repeat(shifts, seeds.shape[0], axis=0)
    central = np.flatnonzero(np.all(tiled_shifts == 0.0, axis=1))
    diagram = Voronoi(tiled)
    ridge_points = np.asarray(diagram.ridge_points)
    volumes = []
    areas = []
    faces = []
    for index in central:
        region = diagram.regions[diagram.point_region[index]]
        if not region or -1 in region:
            raise RuntimeError(
                "Periodic seed replication left an unbounded central cell."
            )
        hull = ConvexHull(diagram.vertices[region])
        volumes.append(float(hull.volume))
        areas.append(float(hull.area))
        faces.append(int(np.count_nonzero(np.any(ridge_points == index, axis=1))))
    expected = list(expected_faces)
    if faces != expected:
        raise RuntimeError(f"Periodic seed topology {faces} does not match {expected}.")
    total_volume = float(np.sum(volumes))
    if not np.isclose(total_volume, 1.0, rtol=0.0, atol=2.0e-12):
        raise RuntimeError(f"Periodic seed volumes sum to {total_volume}, not one.")
    count = seeds.shape[0]
    return {
        "region_volumes": volumes,
        "face_counts": faces,
        "total_volume": total_volume,
        "volume_residual": abs(total_volume - 1.0),
        "planar_seed_cost": float(np.mean(areas) / (1.0 / count) ** (2.0 / 3.0)),
    }


def _equal_volume_foam(
    n: int, seeds: np.ndarray, steps: int
) -> tuple[dict[str, object], float]:
    cells = seeds.shape[0]
    field = _grid_voronoi(n, 3, seeds)
    if (n**3) % cells:
        raise ValueError("grid must split into equal cells")
    counts = np.full((cells,), n**3 // cells)
    route = td.PeriodicGridHeatKernel((n,) * 3, (1.0,) * 3)
    dt = (2.0 / n) ** 2
    prepared = _uniform_plan(
        cells, dt, constraint=td.LabelVolumeConstraint(counts)
    ).prepare(route)
    begin = time.perf_counter()
    result = _run(prepared, prepared.initial_state(field), steps)
    jax.block_until_ready(result.state.labels)
    seconds = time.perf_counter() - begin
    energies = np.asarray(result.evidence.energy_after)
    raw_history = _foam_cell_cost(energies, cells)
    final_labels = np.asarray(result.state.labels)
    kernel_widths = np.asarray((2.0, 3.0, 4.0), dtype=np.float64) / n
    width_costs = []
    for root_tau in (2.0, 3.0, 4.0):
        evaluator = _uniform_plan(cells, (root_tau / n) ** 2).prepare(route)
        state = evaluator.initial_state(final_labels)
        energy = float(_energy(evaluator, state))
        width_costs.append(float(_foam_cell_cost([energy], cells)[0]))
    cost, extrapolation_error = _zero_width_cost(kernel_widths, width_costs)
    cell_volume = 1.0 / cells
    mean_cell_area = cost * cell_volume ** (2.0 / 3.0)
    interface_area = 0.5 * cells * mean_cell_area
    final_counts = np.bincount(final_labels.reshape(-1), minlength=cells)
    volume_error = int(np.max(np.abs(final_counts - counts)))
    status = int(result.status)
    committed_steps = int(result.committed_steps)
    return {
        "grid": n,
        "cells": cells,
        "voxels_per_cell": n**3 // cells,
        "sqrt_tau_over_h": 2.0,
        "steps": steps,
        "committed_steps": committed_steps,
        "raw_cost_history_at_sqrt_tau_over_h_2": raw_history.tolist(),
        "finite_kernel_widths": kernel_widths.tolist(),
        "finite_kernel_costs": width_costs,
        "cost": cost,
        "zero_width_fit_standard_error": extrapolation_error,
        "interface_area": interface_area,
        "mean_cell_surface_area": mean_cell_area,
        "cell_volume": cell_volume,
        "maximum_volume_count_error": volume_error,
        "active_cells": int(np.count_nonzero(final_counts)),
        "worst_status": status,
        "successful": status == int(td.ThresholdDynamicsStatus.SUCCESS)
        and committed_steps == steps,
        "seconds": seconds,
    }, cost


def kelvin_weaire_phelan() -> dict[str, object]:
    kelvin_seeds = _kelvin_bcc_seeds()
    phelan_seeds = _weaire_phelan_a15_seeds()
    kelvin_seed = _periodic_seed_evidence(kelvin_seeds, (14,) * 16)
    phelan_seed = _periodic_seed_evidence(phelan_seeds, (12, 12, 14, 14, 14, 14, 14, 14))
    refinements = []
    kelvin_costs = []
    phelan_costs = []
    valid = True
    for kelvin_grid, phelan_grid in ((32, 26), (48, 38), (64, 50)):
        kelvin, kelvin_cost = _equal_volume_foam(kelvin_grid, kelvin_seeds, 20)
        phelan, phelan_cost = _equal_volume_foam(phelan_grid, phelan_seeds, 20)
        refinements.append({"kelvin": kelvin, "weaire_phelan": phelan})
        kelvin_costs.append(kelvin_cost)
        phelan_costs.append(phelan_cost)
        valid &= bool(
            kelvin["successful"]
            and phelan["successful"]
            and kelvin["maximum_volume_count_error"] == 0
            and phelan["maximum_volume_count_error"] == 0
        )
    kelvin_cost = kelvin_costs[-1]
    phelan_cost = phelan_costs[-1]
    kelvin_error = abs(kelvin_cost / KELVIN_COST - 1.0)
    phelan_error = abs(phelan_cost / WEAIRE_PHELAN_COST - 1.0)
    kelvin_converged = kelvin_error <= 0.03 and abs(kelvin_cost - KELVIN_COST) < abs(
        kelvin_costs[0] - KELVIN_COST
    )
    phelan_converged = phelan_error <= 0.03 and abs(
        phelan_cost - WEAIRE_PHELAN_COST
    ) < abs(phelan_costs[0] - WEAIRE_PHELAN_COST)
    kelvin_uncertainty = abs(kelvin_costs[-1] - kelvin_costs[-2])
    phelan_uncertainty = abs(phelan_costs[-1] - phelan_costs[-2])
    ordering_uncertainty = kelvin_uncertainty + phelan_uncertainty
    observed_gap = kelvin_cost - phelan_cost
    reference_gap = KELVIN_COST - WEAIRE_PHELAN_COST
    ordering_resolved = bool(
        observed_gap > ordering_uncertainty and ordering_uncertainty < reference_gap
    )
    return {
        "reference": {"kelvin": KELVIN_COST, "weaire_phelan": WEAIRE_PHELAN_COST},
        "cost_definition": (
            "zero-kernel-width extrapolation of A/V^(2/3) = "
            "(2 E/N)/(V_box/N)^(2/3); E counts every periodic film once"
        ),
        "seed_cells": {
            "kelvin_bcc_supercell": kelvin_seed,
            "weaire_phelan_a15_cell": phelan_seed,
        },
        "refinements": refinements,
        "kelvin": refinements[-1]["kelvin"],
        "weaire_phelan": refinements[-1]["weaire_phelan"],
        "kelvin_relative_error": kelvin_error,
        "weaire_phelan_relative_error": phelan_error,
        "kelvin_converged_within_3_percent": kelvin_converged,
        "weaire_phelan_converged_within_3_percent": phelan_converged,
        "kelvin_refinement_uncertainty": kelvin_uncertainty,
        "weaire_phelan_refinement_uncertainty": phelan_uncertainty,
        "observed_cost_gap": observed_gap,
        "ordering_uncertainty": ordering_uncertainty,
        "ordering_resolved": ordering_resolved,
        "ordering_claim": (
            "Weaire-Phelan lower than Kelvin"
            if ordering_resolved
            else "nonclaim: the refinement uncertainty does not resolve the 0.3% gap"
        ),
        "criterion": "both finest costs within 3%; ordering only if refinement intervals separate",
        "successful": bool(valid),
        "passed": bool(valid and kelvin_converged and phelan_converged),
    }


def _timed_sparse(
    depth: int, labels: int, grains: int
) -> tuple[dict[str, object], float, int]:
    resolution = 2**depth
    grid = _sparse_grid(depth, 2, 4, np.float32)
    prepared = _uniform_plan(labels, (1.2 / resolution) ** 2).prepare(
        grid, candidate_capacity=16
    )
    sites = np.asarray(grid.site_coordinates, dtype=np.int64)
    seeds = np.random.default_rng(3).random((grains, 2))
    field = _periodic_voronoi((sites + 0.5) / resolution, seeds) * (labels // grains)
    state = prepared.initial_state(field)
    compiled = eqx.filter_jit(lambda item, value: item.step(value))
    begin = time.perf_counter()
    jax.block_until_ready(compiled(prepared, state).state.labels)
    first = time.perf_counter() - begin
    begin = time.perf_counter()
    for _ in range(3):
        result = compiled(prepared, state)
    jax.block_until_ready(result.state.labels)
    warm = (time.perf_counter() - begin) / 3.0
    status = int(result.status)
    sparse = result.evidence.sparse
    successful = bool(
        status == int(td.ThresholdDynamicsStatus.SUCCESS)
        and sparse is not None
        and not np.any(np.asarray(sparse.candidate_overflow))
    )
    return (
        {
            "sites": resolution**2,
            "declared_labels": labels,
            "compile_and_first_step": first,
            "warm_step": warm,
            "working_bytes": prepared.working_bytes,
            "status": status,
            "successful": successful,
        },
        warm,
        prepared.working_bytes,
    )


def _uniform_dense_resource() -> dict[str, object]:
    label_count = 50_000
    prepared = _uniform_plan(label_count, 1.0e-3).prepare(
        td.PeriodicGridHeatKernel((4,), (1.0,))
    )
    state = prepared.initial_state(
        np.linspace(0, label_count - 1, 4, dtype=np.int32)
    )
    decomposition = prepared.plan.decomposition()
    potentials = prepared.potentials(state)
    result = prepared.step(state)
    coefficient_shape = decomposition.coefficients.shape
    avoided_pairwise_bytes = (
        2 * label_count * label_count * np.dtype(np.float64).itemsize
    )
    values = np.asarray(potentials.values)
    valid = np.asarray(potentials.valid)
    finite = bool(
        np.all(np.isfinite(values[valid]))
        and np.all(np.isfinite(np.asarray(potentials.own)))
    )
    successful = bool(
        int(result.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
        and finite
        and coefficient_shape == (2,)
        and prepared.coefficient_bytes == 0
        and prepared.working_bytes < avoided_pairwise_bytes
    )
    return {
        "declared_labels": label_count,
        "sites": prepared.site_count,
        "kernel_coefficient_shape": list(coefficient_shape),
        "coefficient_bytes": prepared.coefficient_bytes,
        "working_bytes": prepared.working_bytes,
        "avoided_pairwise_materialization_bytes": avoided_pairwise_bytes,
        "potentials_finite": finite,
        "status": int(result.status),
        "successful": successful,
    }


def _nonuniform_resource_refusal() -> dict[str, object]:
    label_count = 128
    labels = tuple(f"label{index}" for index in range(label_count))
    values = np.ones((label_count, label_count), dtype=np.float64)
    np.fill_diagonal(values, 0.0)
    expected_coefficient_bytes = 3 * values.nbytes
    refused = False
    try:
        td.ThresholdDynamicsPlan(
            InterfaceTensionMatrix(labels, values),
            InterfaceMobilityMatrix(labels, values),
            1.0e-3,
            resource_policy=td.ThresholdDynamicsResourcePolicy(
                maximum_working_bytes=expected_coefficient_bytes - 1
            ),
            minimum_resolution_ratio=0.0,
        )
    except MemoryError:
        refused = True
    return {
        "declared_labels": label_count,
        "expected_coefficient_bytes": expected_coefficient_bytes,
        "maximum_working_bytes": expected_coefficient_bytes - 1,
        "refused_before_decomposition": refused,
        "successful": refused,
    }


def sparse_scaling() -> dict[str, object]:
    uniform_dense = _uniform_dense_resource()
    nonuniform_refusal = _nonuniform_resource_refusal()
    by_sites = [_timed_sparse(depth, 64, 32) for depth in (6, 7, 8)]
    by_labels = [_timed_sparse(7, labels, 32) for labels in (64, 10_000, 1_000_000)]
    warm = [row[1] for row in by_labels]
    spread = max(warm) / min(warm)
    same_storage = len({row[2] for row in by_labels}) == 1
    site_rows = [row[0] for row in by_sites]
    label_rows = [row[0] for row in by_labels]
    successful = (
        bool(uniform_dense["successful"])
        and bool(nonuniform_refusal["successful"])
        and all(row["successful"] for row in (*site_rows, *label_rows))
    )
    return {
        "uniform_dense": uniform_dense,
        "nonuniform_resource_refusal": nonuniform_refusal,
        "by_sites": site_rows,
        "by_labels": label_rows,
        "label_time_spread": spread,
        "criterion": "uniform coefficients avoid label-squared storage; sparse working storage is label-count independent and warm time spread is within 1.5x",
        "successful": successful,
        "passed": bool(successful and same_storage and spread <= 1.5),
    }


def _icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    ratio = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, ratio, 0),
            (1, ratio, 0),
            (-1, -ratio, 0),
            (1, -ratio, 0),
            (0, -1, ratio),
            (0, 1, ratio),
            (0, -1, -ratio),
            (0, 1, -ratio),
            (ratio, 0, -1),
            (ratio, 0, 1),
            (-ratio, 0, -1),
            (-ratio, 0, 1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoints = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        count = faces.shape[0]
        ab, bc, ca = (
            vertices.shape[0] + inverse.reshape(-1)[k * count : (k + 1) * count]
            for k in range(3)
        )
        a, b, c = faces.T
        faces = np.concatenate(
            (
                np.stack((a, ab, ca), 1),
                np.stack((b, bc, ab), 1),
                np.stack((c, ca, bc), 1),
                np.stack((ab, bc, ca), 1),
            )
        )
        vertices = np.concatenate((vertices, midpoints))
    return vertices, faces


def _mesh_cosine(mass: np.ndarray, labels: np.ndarray, /) -> float:
    """Cap cosine from the fraction of the same discrete polyhedral sphere area."""
    return 1.0 - 2.0 * float(np.sum(mass[labels == 0])) / float(np.sum(mass))


def _mesh_case(
    level: int,
    dt: float,
    steps: int,
    *,
    policy: TaylorExponentialPolicy | None = None,
) -> tuple[
    _MeshRow,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    vertices, faces = _icosphere(level)
    kernel = td.MeshHeatKernel(vertices, faces, policy=policy)
    prepared = _uniform_plan(2, dt).prepare(kernel)
    labels = np.where(vertices[:, 2] > 0.5, 0, 1)
    result = _run(prepared, prepared.initial_state(labels), steps)
    mass = np.asarray(prepared.site_measures)
    final = np.asarray(result.state.labels)
    cos_initial = _mesh_cosine(mass, labels)
    cos_final = _mesh_cosine(mass, final)
    elapsed = dt * int(result.committed_steps)
    predicted = cos_initial * np.exp(elapsed)
    row: _MeshRow = {
        "level": level,
        "vertices": vertices.shape[0],
        "h": kernel.resolution_length,
        "dt": dt,
        "dt_over_h": dt / kernel.resolution_length,
        "steps": steps,
        "time": elapsed,
        "cos_theta": cos_final,
        "predicted": float(predicted),
        "error": abs(cos_final - predicted),
        "polyhedral_area": float(np.sum(mass)),
        "sphere_area_relative_defect": abs(float(np.sum(mass)) / (4.0 * np.pi) - 1.0),
        "status": int(result.status),
        "committed_steps": int(result.committed_steps),
        "successful": bool(
            int(result.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
            and int(result.committed_steps) == steps
            and np.all(np.asarray(result.evidence.heat.successful))
            and np.all(np.asarray(result.evidence.heat.native_status) == 0)
            and np.all(np.asarray(result.evidence.heat.converged))
            and np.all(np.asarray(result.evidence.heat.derivative_valid))
        ),
        "heat_error_estimate": float(
            np.max(np.asarray(result.evidence.heat.error_estimate))
        ),
    }
    return row, vertices, faces, labels, final, mass


def _surface_p1_operators(
    vertices: np.ndarray, faces: np.ndarray, /
) -> tuple[scipy_sparse.csr_matrix, scipy_sparse.csr_matrix]:
    """Independent consistent P1 mass and stiffness matrices on a triangle surface."""
    corners = vertices[faces]
    edges = corners[:, 1:] - corners[:, :1]
    gram = np.einsum("fie,fje->fij", edges, edges)
    areas = 0.5 * np.sqrt(np.linalg.det(gram))
    tails = np.linalg.solve(gram, edges)
    gradients = np.concatenate((-np.sum(tails, axis=1, keepdims=True), tails), axis=1)
    stiffness_local = areas[:, None, None] * np.einsum(
        "fie,fje->fij", gradients, gradients
    )
    mass_template = np.ones((3, 3), dtype=np.float64) + np.eye(3, dtype=np.float64)
    mass_local = areas[:, None, None] * mass_template[None] / 12.0
    rows = np.repeat(faces, 3, axis=1).reshape(-1)
    columns = np.tile(faces, (1, 3)).reshape(-1)
    shape = (vertices.shape[0], vertices.shape[0])
    mass = scipy_sparse.coo_matrix(
        (mass_local.reshape(-1), (rows, columns)), shape=shape
    ).tocsr()
    stiffness = scipy_sparse.coo_matrix(
        (stiffness_local.reshape(-1), (rows, columns)), shape=shape
    ).tocsr()
    return mass, stiffness


def _consistent_mass_probe(
    level: int,
    dt: float,
    lumped_error: float,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    lumped_final: np.ndarray,
) -> dict[str, object]:
    """One-step consistent-mass reference isolates the lumping contribution."""
    mass, stiffness = _surface_p1_operators(vertices, faces)
    generator = -scipy_sparse_linalg.spsolve(mass.tocsc(), stiffness.toarray())
    smoothed = scipy_sparse_linalg.expm_multiply(
        dt * generator, (labels == 0).astype(np.float64)
    )
    consistent = np.where(smoothed >= 0.5, 0, 1)
    lumped_mass = np.asarray(mass.sum(axis=1)).reshape(-1)
    initial_cosine = _mesh_cosine(lumped_mass, labels)
    predicted = initial_cosine * np.exp(dt)
    consistent_error = abs(_mesh_cosine(lumped_mass, consistent) - predicted)
    return {
        "level": level,
        "dt": dt,
        "lumped_error": lumped_error,
        "consistent_mass_error": consistent_error,
        "different_label_fraction": float(np.mean(consistent != lumped_final)),
    }


def mesh() -> dict[str, object]:
    rows = []
    cases = []
    errors: list[float] = []
    # Fixed final time with dt/h nearly constant jointly refines the MBO step and
    # polyhedral P1 space without entering the hard-label pinning regime.
    for level, steps in zip((3, 4, 5), (1, 2, 4), strict=True):
        row, vertices, faces, labels, final, mass = _mesh_case(level, 0.2 / steps, steps)
        rows.append(row)
        cases.append((vertices, faces, labels, final, mass))
        errors.append(float(row["error"]))
    monotone = all(
        fine < coarse for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    )
    spatial_orders = (
        [
            float(
                np.log(errors[index] / errors[index + 1])
                / np.log(float(rows[index]["h"]) / float(rows[index + 1]["h"]))
            )
            for index in range(len(errors) - 1)
        ]
        if monotone and all(error > 0.0 for error in errors)
        else None
    )

    h = float(rows[1]["h"])
    time_rows = []
    for factor, steps in ((1.5, 2), (1.0, 3), (0.75, 4)):
        row, *_ = _mesh_case(4, factor * h, steps)
        time_rows.append(row)
    time_errors = [float(row["error"]) for row in time_rows]
    time_monotone = all(
        fine < coarse
        for coarse, fine in zip(time_errors[:-1], time_errors[1:], strict=True)
    )
    time_orders = (
        [
            float(
                np.log(time_errors[index] / time_errors[index + 1])
                / np.log(
                    float(time_rows[index]["dt"]) / float(time_rows[index + 1]["dt"])
                )
            )
            for index in range(len(time_errors) - 1)
        ]
        if time_monotone and all(error > 0.0 for error in time_errors)
        else None
    )

    tolerance_rows = []
    tolerance_labels = []
    for tolerance in (1.0e-6, 1.0e-8, 1.0e-10):
        policy = TaylorExponentialPolicy(error_tolerance=tolerance, norm_mode="estimate")
        row, _, _, _, final, _ = _mesh_case(3, 0.2, 1, policy=policy)
        row["requested_tolerance"] = tolerance
        tolerance_rows.append(row)
        tolerance_labels.append(final)
    tolerance_difference = max(
        float(np.mean(labels != tolerance_labels[-1])) for labels in tolerance_labels[:-1]
    )

    coarse_vertices, coarse_faces, coarse_labels, coarse_final, _ = cases[0]
    mass_probe = _consistent_mass_probe(
        3, 0.2, errors[0], coarse_vertices, coarse_faces, coarse_labels, coarse_final
    )
    error_floor = max(errors[-2:]) if spatial_orders is None else None
    order_claim = (
        "local observed orders under joint dt/h refinement; no asymptotic order"
        if spatial_orders is not None
        else "none: nonmonotone critical-regime time/space discretization floor"
    )
    all_rows = (*rows, *time_rows, *tolerance_rows)
    successful = all(row["successful"] for row in all_rows)
    return {
        "reference": "geodesic curvature flow of a circle on the unit sphere: cos(theta) = cos(theta0) exp(t)",
        "area_normalization": "cos(theta) = 1 - 2 A_cap/A_polyhedron",
        "rows": rows,
        "spatial_orders": spatial_orders,
        "order_claim": order_claim,
        "discretization_error_floor": error_floor,
        "finest_joint_refinement_error": errors[-1],
        "time_step_isolation": {
            "fixed_level": 4,
            "fixed_time_over_h": 3.0,
            "rows": time_rows,
            "observed_orders": time_orders,
        },
        "matrix_exponential_tolerance_isolation": {
            "rows": tolerance_rows,
            "maximum_changed_label_fraction": tolerance_difference,
        },
        "mass_lumping_isolation": mass_probe,
        "error_source": (
            "The former sequence mixed polyhedral cap areas with the continuum "
            "4 pi sphere area, adding a spatial geometry-normalization bias. "
            "Polyhedral-area normalization removes that bias. The joint refinement "
            "and isolation probes distinguish the remaining critical-regime "
            "time/space error from Taylor-action tolerance and mass lumping."
        ),
        "criterion": "finest error < 0.02 and finest < coarsest",
        "successful": successful,
        "passed": bool(
            successful and errors[-1] < 0.02 and errors[-1] < errors[0]
        ),
    }


SCENARIOS: dict[str, Callable[[], dict[str, object]]] = {
    "circle": circle,
    "sphere": sphere,
    "herring": herring,
    "pressure": pressure,
    "bubble-diffusion": bubble_diffusion,
    "von-neumann": von_neumann,
    "grain-statistics-2d": grain_statistics_2d,
    "grain-statistics-3d": grain_statistics_3d,
    "kelvin-weaire-phelan": kelvin_weaire_phelan,
    "sparse-scaling": sparse_scaling,
    "mesh": mesh,
}


def run_qualification(selected: tuple[str, ...]) -> dict[str, object]:
    records = {}
    for name in selected:
        begin = time.perf_counter()
        record = SCENARIOS[name]()
        record["wall_seconds"] = time.perf_counter() - begin
        records[name] = record
    return {
        "kind": "threshold-dynamics-qualification",
        "runtime": _runtime_identity(),
        "profiles": [
            profile.to_record() for profile in td.threshold_dynamics_candidate_profiles()
        ],
        "scenarios": records,
        "successful": bool(records)
        and all(
            record["successful"] is True and record["passed"] is True
            for record in records.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", choices=tuple(SCENARIOS))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    selected = tuple(arguments.scenario) if arguments.scenario else tuple(SCENARIOS)
    report = run_qualification(selected)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=float)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

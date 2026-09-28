#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coarse Hysing test case 1: a gas bubble rising in liquid (structured VOF).

Problem definition from Hysing et al., Int. J. Numer. Meth. Fluids 60 (2009)
1259-1288, doi:10.1002/fld.1934, Section 2 and Table I: domain
``[0, 1] x [0, 2]``, bubble of radius 0.25 at ``(0.5, 0.5)``, densities
1000/100, viscosities 10/1, gravity 0.98, surface tension 24.5, free-slip
sides, and no-slip top and bottom walls.  This smoke-scale run (24 cells per
unit, 6 per radius, to ``t = 1``) reports the published observables
(circularity, centroid, mean rise velocity) through
:func:`phydrax.applications.free_boundary.hysing_bubble_benchmark`.  It is not
a qualification: the grid is far coarser than the reference solutions and
the run stops before the circularity minimum.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


CELLS_PER_UNIT = 24
FINAL_TIME = 1.0
SAMPLE_INTERVAL = 0.1
LIQUID_DENSITY = 1000.0
GAS_DENSITY = 100.0
SURFACE_TENSION = 24.5
# Hysing et al. (2009) Table XII, finest TP2D grid.
REFERENCE_MAXIMUM_RISE_VELOCITY = 0.2417
REFERENCE_MAXIMUM_RISE_VELOCITY_TIME = 0.9213


def _liquid_fraction(cells_per_unit: int, samples: int = 16) -> np.ndarray:
    """Area-accurate liquid fraction outside the bubble (samples² points per cell)."""

    offsets = (np.arange(samples) + 0.5) / (samples * cells_per_unit)
    x = (np.arange(cells_per_unit)[:, None] / cells_per_unit + offsets).reshape(-1)
    y = (np.arange(2 * cells_per_unit)[:, None] / cells_per_unit + offsets).reshape(-1)
    gas = (x[:, None] - 0.5) ** 2 + (y[None, :] - 0.5) ** 2 < 0.25**2
    shape = (cells_per_unit, samples, 2 * cells_per_unit, samples)
    return 1.0 - gas.reshape(shape).mean(axis=(1, 3))


def _prepared() -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(CELLS_PER_UNIT),
            phx.discretization.UniformCellAxisSpec(2 * CELLS_PER_UNIT),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 2.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=LIQUID_DENSITY,
        gas_density=GAS_DENSITY,
        liquid_viscosity=10.0,
        gas_viscosity=1.0,
        surface_tension=SURFACE_TENSION,
    )
    walls = (
        phx.discretization.MACBoundarySide("x", "lower", "free-slip"),
        phx.discretization.MACBoundarySide("x", "upper", "free-slip"),
        phx.discretization.MACBoundarySide("y", "lower", "no-slip"),
        phx.discretization.MACBoundarySide("y", "upper", "no-slip"),
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=4000,
        gravity=(0.0, -0.98),
        hydrostatic_reference=(0.5, 2.0),
        wall_sides=walls,
    ).prepare()
    return discretization, two_phase


def _observables(discretization: Any, view: Any) -> dict[str, float]:
    """Hysing observables from PLIC facet centroids ordered by angle.

    Angular ordering around the gas centroid is valid for the star-shaped
    bubble of test case 1 before its circularity minimum.  Facet centroids lie
    on chords of the convex interface, so the contour slightly underestimates
    the perimeter (circularity about 1.002 for the initial circle at 6 cells per
    radius).
    """

    alpha = np.asarray(view.alpha)
    # Accepted alpha may leave [0, 1] only by the rounding excursion the step
    # admits; anything larger is a boundedness failure, never clipped away.
    band = phx.applications.two_phase_flow.alpha_bound_tolerance(view.alpha.dtype)
    if alpha.min() < -band or alpha.max() > 1.0 + band:
        raise RuntimeError("Accepted alpha left the admitted rounding band.")
    gas = np.clip(1.0 - alpha, 0.0, 1.0)
    centers = np.asarray(discretization.cell_centers)
    volumes = np.asarray(discretization.cell_volumes)
    centroid = np.sum((gas * volumes)[..., None] * centers, axis=(0, 1)) / np.sum(
        gas * volumes
    )
    points = np.asarray(view.plic.interface_point)[np.asarray(view.plic.mixed_cell)]
    angle = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    vertical = np.asarray(view.velocity[1])
    report = phx.applications.free_boundary.hysing_bubble_benchmark(
        gas.reshape(-1),
        centers.reshape((-1, 2)),
        volumes.reshape(-1),
        (0.5 * (vertical[:, :-1] + vertical[:, 1:])).reshape(-1),
        points[np.argsort(angle)],
    )
    return {
        "circularity": float(report.circularity),
        "centroid_y": float(report.centroid[1]),
        "rise_velocity": float(report.mean_rise_velocity),
        "gas_area": float(report.area),
    }


def run() -> Any:
    discretization, two_phase = _prepared()
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(
        two_phase.initial_state(jnp.asarray(_liquid_fraction(CELLS_PER_UNIT)))
    )
    width = 1.0 / CELLS_PER_UNIT
    # Below the Brackbill capillary limit sqrt(rho_mean h^3 / (2 pi sigma)) and
    # at directional Courant number 1/4 for an expected speed of 0.3.
    capillary_limit = np.sqrt(
        0.5 * (LIQUID_DENSITY + GAS_DENSITY) * width**3 / (2.0 * np.pi * SURFACE_TENSION)
    )
    step_count = int(np.ceil(FINAL_TIME / min(0.8 * capillary_limit, 0.25 * width / 0.3)))
    step_size = FINAL_TIME / step_count
    stride = round(SAMPLE_INTERVAL / step_size)
    step = eqx.filter_jit(method.step)
    view = eqx.filter_jit(two_phase.view)
    samples = [{"time": 0.0, **_observables(discretization, view(continuation.state))}]
    underresolved = 0
    fallback = 0
    maximum_divergence = 0.0
    for index in range(step_count):
        result = step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * step_size),
            continuation,
            jnp.asarray(step_size),
            None,
        )
        evidence = result.candidate_state.evidence
        if not bool(result.successful):
            refusal = {
                field.name: np.asarray(getattr(evidence, field.name)).item()
                for field in dataclasses.fields(evidence)
            }
            raise RuntimeError(f"Rising-bubble step {index} was refused: {refusal}")
        continuation = result.accepted_state
        underresolved += int(evidence.curvature_underresolved_count)
        fallback += int(evidence.curvature_fallback_count)
        maximum_divergence = max(maximum_divergence, float(evidence.divergence_residual))
        if (index + 1) % stride == 0 or index + 1 == step_count:
            samples.append(
                {
                    "time": (index + 1) * step_size,
                    **_observables(discretization, view(continuation.state)),
                }
            )
    fastest = max(samples, key=lambda sample: sample["rise_velocity"])
    return {
        "successful": True,
        "cells": (CELLS_PER_UNIT, 2 * CELLS_PER_UNIT),
        "steps": step_count,
        "step_size": step_size,
        "samples": samples,
        "sampled_maximum_rise_velocity": fastest["rise_velocity"],
        "sampled_maximum_rise_velocity_time": fastest["time"],
        "reference_maximum_rise_velocity": REFERENCE_MAXIMUM_RISE_VELOCITY,
        "reference_maximum_rise_velocity_time": REFERENCE_MAXIMUM_RISE_VELOCITY_TIME,
        "gas_area_drift": samples[-1]["gas_area"] - samples[0]["gas_area"],
        "curvature_fallback_cell_steps": fallback,
        "curvature_underresolved_cell_steps": underresolved,
        "maximum_divergence_residual": maximum_divergence,
        "gravity_work": float(continuation.ledger.gravity_work),
        "gravitational_energy_change": float(
            continuation.ledger.gravitational_energy_change
        ),
        "viscous_dissipation": float(continuation.ledger.viscous_dissipation),
    }


if __name__ == "__main__":
    print(run())

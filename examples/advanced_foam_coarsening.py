#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two-dimensional dry foam by threshold dynamics: coarsening and exact volumes.

A periodic Voronoi foam of 16 cells on a 128^2 grid with one film tension is advanced three ways
on the exact periodic Fourier heat kernel:

1. unconstrained multiphase threshold dynamics (curvature-driven coarsening,
   the capillarity-only grain-growth limit): cells vanish and the film energy
   decreases monotonically;
2. exact integer cell volumes by capacitated auction: every cell keeps its site
   count while films straighten toward 120-degree Plateau junctions;
3. gas-diffusive coarsening: auction prices give cell pressures, gas crosses films
   with permeance ``k``, and the von Neumann rate ``dA/dt = k_eff pi sigma (n - 6)/3``
   is fitted with ``k_eff = k mu / (k + mu)``.
"""

from typing import Any

import equinox as eqx
import numpy as np

import phydrax.threshold_dynamics as td
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix


GRID = 128
CELLS = 16
TIME_STEP = (2.5 / GRID) ** 2


@eqx.filter_jit
def _run(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState, steps: int
) -> td.ThresholdDynamicsRunResult:
    return prepared.run(state, steps)


@eqx.filter_jit
def _coarsen(
    coarsening: td.GasDiffusionCoarsening, state: td.LabelFieldState, steps: int
) -> td.GasDiffusionRunResult:
    return coarsening.run(state, steps)


def _voronoi_foam(seed: int) -> np.ndarray:
    x = (np.arange(GRID) + 0.5) / GRID
    points = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1)
    centers = np.random.default_rng(seed).random((CELLS, 2))
    offsets = np.abs(points[:, :, None, :] - centers[None, None])
    offsets = np.minimum(offsets, 1.0 - offsets)
    return np.argmin(np.sum(offsets**2, axis=-1), axis=-1)


def _plan(
    labels: tuple[str, ...], constraint: td.LabelVolumeConstraint | None
) -> td.ThresholdDynamicsPlan:
    return td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
        TIME_STEP,
        volume_constraint=constraint,
    )


def _energies(result: td.ThresholdDynamicsRunResult) -> np.ndarray:
    return np.concatenate(
        ([float(result.initial_energy)], np.asarray(result.evidence.energy_after))
    )


def run() -> dict[str, Any]:
    labels = tuple(f"cell{index}" for index in range(CELLS))
    foam = _voronoi_foam(7)
    kernel = td.PeriodicGridHeatKernel((GRID, GRID), (1.0, 1.0))
    counts = np.bincount(foam.reshape(-1), minlength=CELLS)

    free = _plan(labels, None).prepare(kernel)
    coarsened = _run(free, free.initial_state(foam), 40)
    free_energy = _energies(coarsened)

    constrained = _plan(labels, td.LabelVolumeConstraint(counts)).prepare(kernel)
    relaxed = _run(constrained, constrained.initial_state(foam), 8)
    relaxed_energy = _energies(relaxed)

    permeance = 0.25
    diffusion = td.GasDiffusionCoarsening(constrained, permeance)
    coarsening = _coarsen(diffusion, constrained.initial_state(foam), 30)
    history = np.asarray(coarsening.evidence.step.label_counts, dtype=np.float64) / GRID**2
    window = slice(3, 31)
    sides = td.label_neighbor_counts(
        np.asarray(foam), CELLS, minimum_contact=3
    )
    slope, crossover = td.von_neumann_mullins_fit(
        TIME_STEP * np.arange(history.shape[0])[window], history[window], sides
    )
    effective = permeance / (permeance + 1.0)
    return {
        "cells": CELLS,
        "unconstrained_surviving_cells": int(np.sum(np.asarray(coarsened.state.active_labels))),
        "unconstrained_energy": (float(free_energy[0]), float(free_energy[-1])),
        "unconstrained_energy_monotone": bool(
            np.all(np.diff(free_energy) <= np.asarray(coarsened.evidence.energy_tolerance))
        ),
        "unconstrained_extinction_epochs": int(coarsened.state.epoch),
        "constrained_counts_exact": bool(
            np.all(np.asarray(relaxed.evidence.label_counts) == counts)
        ),
        "constrained_energy": (float(relaxed_energy[0]), float(relaxed_energy[-1])),
        "constrained_status": int(relaxed.status),
        "diffusion_surviving_cells": int(np.sum(history[-1] > 0.0)),
        "diffusion_total_area": float(np.max(np.abs(history.sum(axis=1) - 1.0))),
        "von_neumann_slope": slope,
        "von_neumann_slope_expected": float(np.pi * effective / 3.0),
        "von_neumann_crossover_sides": crossover,
        "diffusion_status": int(np.max(np.asarray(coarsening.evidence.step.status))),
    }


if __name__ == "__main__":
    print(run())

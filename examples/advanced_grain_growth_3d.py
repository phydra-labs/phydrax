#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Three-dimensional capillarity-driven grain growth on the sparse label route.

A periodic Voronoi polycrystal with 200 grains declared among 100 000 stable
grain labels evolves by isotropic multiphase threshold dynamics on a 64^3 sparse
voxel grid. Each 4^3 brick keeps at most 32 candidate labels from its halo, so
storage and work never scale with the declared label count; overflow, truncated
kernel mass and kernel positivity are reported as evidence.

This is capillarity-only grain growth: no anisotropic boundary energy, solute
drag, pinning particles or recrystallization are modelled, and the steady-state
topology it approaches (mean faces per grain about 13.77, Mason et al. 2015) is a
qualification campaign, not a claim of this short run.
"""

from typing import Any

import equinox as eqx
import numpy as np

import phydrax.threshold_dynamics as td
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix


DEPTH = 6
GRAINS = 200
DECLARED_LABELS = 100_000
TIME_STEP = (1.1 / 2**DEPTH) ** 2


@eqx.filter_jit
def _run(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState, steps: int
) -> td.ThresholdDynamicsRunResult:
    return prepared.run(state, steps)


def _polycrystal(coordinates: np.ndarray, seed: int) -> np.ndarray:
    resolution = 2**DEPTH
    points = (coordinates + 0.5) / resolution
    rng = np.random.default_rng(seed)
    centers = rng.random((GRAINS, 3))
    ids = rng.choice(DECLARED_LABELS, size=GRAINS, replace=False)
    nearest = np.empty((points.shape[0],), dtype=np.int64)
    for start in range(0, points.shape[0], 16_384):
        chunk = points[start : start + 16_384]
        offsets = np.abs(chunk[:, None, :] - centers[None])
        offsets = np.minimum(offsets, 1.0 - offsets)
        nearest[start : start + 16_384] = np.argmin(np.sum(offsets**2, axis=-1), axis=1)
    return ids[nearest]


def run() -> dict[str, Any]:
    resolution = 2**DEPTH
    labels = tuple(f"grain{index}" for index in range(DECLARED_LABELS))
    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
        TIME_STEP,
    )
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), DEPTH, periodic_axes=(True, True, True)
    )
    coordinates = np.stack(
        np.meshgrid(*([np.arange(resolution)] * 3), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid = td.SparseLabelGrid(
        address,
        coordinates,
        brick_size=4,
        brick_capacity=(resolution // 4) ** 3,
        stencil_radius=4,
        dtype=np.float32,
    )
    prepared = plan.prepare(grid, candidate_capacity=32)
    sites = np.asarray(grid.site_coordinates, dtype=np.int64)
    state = prepared.initial_state(_polycrystal(sites, 5))
    result = _run(prepared, state, 12)
    evidence = result.evidence
    sparse = evidence.sparse
    if sparse is None:
        raise RuntimeError("The sparse route must report candidate evidence.")
    final = np.zeros((resolution,) * 3, dtype=np.int64)
    final[sites[:, 0], sites[:, 1], sites[:, 2]] = np.asarray(result.state.labels)
    present = np.unique(final)
    compact = np.searchsorted(present, final)
    faces = td.label_neighbor_counts(compact, present.size, minimum_contact=4)
    energies = np.concatenate(
        ([float(result.initial_energy)], np.asarray(evidence.energy_after))
    )
    return {
        "declared_labels": DECLARED_LABELS,
        "initial_grains": GRAINS,
        "final_grains": int(present.size),
        "mean_faces": float(np.mean(faces)),
        "energy": (float(energies[0]), float(energies[-1])),
        "energy_monotone": bool(
            np.all(np.diff(energies) <= np.asarray(evidence.energy_tolerance))
        ),
        "worst_status": int(result.status),
        "committed_steps": int(result.committed_steps),
        "maximum_candidates": int(np.max(np.asarray(sparse.required_candidates))),
        "candidate_capacity": sparse.candidate_capacity,
        "truncated_kernel_mass": float(np.max(np.asarray(sparse.truncated_kernel_mass))),
        "kernel_symbol_minimum": float(np.min(np.asarray(sparse.kernel_symbol_minimum))),
        "working_bytes": prepared.working_bytes,
    }


if __name__ == "__main__":
    print(run())

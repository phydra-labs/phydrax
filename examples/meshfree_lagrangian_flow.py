# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Lagrangian GMLS Taylor-Green flow with ledgers, a refused step and a transfer.

Material points of a periodic unit box carry quadrature volumes and constant
masses; every step refreshes (or re-prepares) the GMLS cloud at the particle
positions, projects the predicted velocity with the weak GMLS pressure
projection and moves the points. A deliberately oversized step is refused by
the Courant check and rolled back; the run then continues. The final state is
transferred to a fixed GMLS cloud with conservative positive coefficients.
Run with ``JAX_ENABLE_X64=1 python -m examples.meshfree_lagrangian_flow``.
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.solver import (
    MeshfreeLagrangianEvidence,
    MeshfreeLagrangianFlowPlan,
    MeshfreeLagrangianStatus,
    MeshfreeMeasureTransferPlan,
)


_WAVE = 2.0 * np.pi
# The unit torus: both coordinate seams of the closed unit square identified.
_UNIT_SQUARE = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_UNIT_SQUARE, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)


def lattice(side: int) -> NDArray[np.float64]:
    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=1)


def taylor_green(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = points[:, 0], points[:, 1]
    return np.stack(
        (
            np.sin(_WAVE * x) * np.cos(_WAVE * y),
            -np.cos(_WAVE * x) * np.sin(_WAVE * y),
        ),
        axis=1,
    )


def ledger(label: str, evidence: MeshfreeLagrangianEvidence) -> str:
    status = MeshfreeLagrangianStatus(int(evidence.status)).name
    momentum = np.asarray(evidence.momentum_after) - np.asarray(evidence.momentum_before)
    return (
        f"{label}: status={status} reprepared={bool(evidence.reprepared)} "
        f"courant={float(evidence.courant):.3f} "
        f"mass_defect={float(evidence.mass_after - evidence.mass_before):.1e} "
        f"momentum_change={np.array2string(momentum, precision=2)} "
        f"pressure_impulse={np.array2string(np.asarray(evidence.pressure_impulse), precision=2)} "
        f"weak_div={float(evidence.divergence_before):.2e}->{float(evidence.divergence_after):.2e} "
        f"strong_div={float(evidence.strong_divergence_after):.2e} "
        f"energy={float(evidence.kinetic_energy_after):.6f} "
        f"pressure_its={int(evidence.pressure_iterations)} "
        f"pressure_normal_residual={float(evidence.pressure_residual):.2e} "
        f"pressure_condition={float(evidence.pressure_condition):.2e}"
    )


def run(*, side: int, steps: int, step_size: float, viscosity: float) -> None:
    points = lattice(side)
    flow = MeshfreeLagrangianFlowPlan(
        "quadrature-volume",
        reference_density=1.0,
        address=_PERIODIC,
        neighbors=21,  # complete square-lattice distance shells
        viscosity=viscosity,
    ).prepare(points, volumes=np.full(points.shape[0], 1.0 / points.shape[0]))
    state = flow.initialize(taylor_green(points))
    for index in range(steps):
        result = flow.step(state, step_size)
        print(ledger(f"step {index}", result.evidence))
        flow, state = result.flow, result.state
        if index == steps // 2:
            refused = flow.step(state, 40.0 * step_size)
            unchanged = bool(np.array_equal(refused.state.positions, state.positions))
            print(
                ledger("oversized step", refused.evidence) + f" rolled_back={unchanged}"
            )
            flow, state = refused.flow, refused.state
    grid = lattice(side // 2)
    transfer = MeshfreeMeasureTransferPlan(
        state.positions,
        state.volumes,
        grid,
        np.full(grid.shape[0], 1.0 / grid.shape[0]),
        source_measure="quadrature-volume",
        target_measure="quadrature-volume",
        address=_PERIODIC,
    ).prepare()
    if not transfer.admitted:
        evidence = transfer.transfer.evidence
        print(
            f"transfer: refused status={transfer.status.name} "
            f"provider={evidence.provider} provider_status={evidence.provider_status}"
        )
        return
    moved = transfer.apply(state.masses, state.velocity)
    print(
        f"transfer: status={transfer.status.name} "
        f"mass_defect={float(moved.mass_defect):.1e} "
        f"momentum_defect={np.array2string(np.asarray(moved.momentum_defect), precision=2)} "
        f"density_mismatch={float(moved.density_mismatch):.2e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--viscosity", type=float, default=0.01)
    args = parser.parse_args()
    run(
        side=args.side,
        steps=args.steps,
        step_size=args.step_size,
        viscosity=args.viscosity,
    )


if __name__ == "__main__":
    main()

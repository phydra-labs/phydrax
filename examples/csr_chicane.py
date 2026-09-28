#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""CSR in a four-bend chicane: steady versus transient 1-D models.

A 1 nC, 50 µm electron bunch at γ = 1000 crosses a rectangular-bend chicane.
The steady Saldin–Derbenev wake ignores the bend entrance and exit
transients that the retarded Mayes–Hoffstaetter model resolves.
"""

import math

import numpy as np

import phydrax as phx
from phydrax.applications import accelerator
from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec


scale = phx.ElectromagneticScaleContract.si()
charge = float(scale.elementary_charge)
rest_energy = float(scale.electron_mass) * float(scale.speed_of_light) ** 2
gamma = 1000.0
momentum = rest_energy * math.sqrt(gamma * gamma - 1.0)

theta, bend, drift = 0.05, 0.5, 2.0
curvature = theta / bend
lattice = accelerator.CSRLattice(
    [bend, drift, bend, 0.5, bend, drift, bend, 0.5],
    [curvature, 0.0, -curvature, 0.0, -curvature, 0.0, curvature, 0.0],
    element_ids=["b1", "d1", "b2", "d2", "b3", "d3", "b4", "d4"],
    entrance_edges=[0.0, 0.0, -theta, 0.0, 0.0, 0.0, theta, 0.0],
    exit_edges=[theta, 0.0, 0.0, 0.0, -theta, 0.0, 0.0, 0.0],
)

count, sigma_z, emittance, beta_x = 2000, 5.0e-5, 1.0e-9, 10.0
rng = np.random.default_rng(0)
coordinates = np.zeros((count, 6))
coordinates[:, 0] = math.sqrt(emittance * beta_x) * rng.standard_normal(count)
coordinates[:, 1] = math.sqrt(emittance / beta_x) * rng.standard_normal(count)
coordinates[:, 2] = math.sqrt(emittance * beta_x) * rng.standard_normal(count)
coordinates[:, 3] = math.sqrt(emittance / beta_x) * rng.standard_normal(count)
coordinates[:, 4] = sigma_z * rng.standard_normal(count)
bunch = accelerator.AcceleratorBunch(
    coordinates,
    np.full((count,), 1.0e-9 / charge / count),
    np.arange(count),
    reference_rest_energy=rest_energy,
    reference_momentum=momentum,
    reference_charge=-1.0,
    bunch_id="chicane",
)
grid = TensorGridPlan((UniformCellAxisSpec(48),)).prepare(
    np.asarray([[-8.0 * sigma_z], [8.0 * sigma_z]])
)
substeps = [6, 3, 6, 1, 6, 3, 6, 1]
initial = accelerator.beam_diagnostics(bunch).geometric_emittance_x
report = {"initial_emittance_x_m": float(initial)}
for model in ("1d-steady", "1d-transient-shielded"):
    plan = accelerator.CSRPlan(
        model,
        lattice,
        scale,
        grid,
        reference_rest_energy=rest_energy,
        reference_momentum=momentum,
        capacity=count,
        smoothing=sigma_z / 6.0,
        history_capacity=64,
        far_nodes=48,
    )
    result = accelerator.track_csr(
        accelerator.CSRTrackingPlan(plan, substeps=substeps), bunch
    )
    status = int(np.bitwise_or.reduce(np.asarray(result.status)))
    report[model] = {
        "emittance_x_m": float(
            accelerator.beam_diagnostics(result.bunch).geometric_emittance_x
        ),
        "energy_change_J": float(np.sum(result.energy_change)),
        "final_rms_delta": float(result.rms_delta[-1]),
        "status_union": str(accelerator.CSRStatus(status)),
        "accepted": bool(result.accepted),
    }
print(report)

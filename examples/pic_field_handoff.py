#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hand a running PIC plasma from the Yee cochain solver to order-2 staggered PSATD.

A cold electron plasma (ω_p² = 16, c = 1) oscillates along x while a transverse
standing wave rings in B_z. It runs 20 steps on the cochain (Yee) solver, is
handed off to PSATD with the Yee divergence (``grid="staggered"``,
``stencil="finite-order"``, ``stencil_order=2``), and continues 20 steps. The
handoff evidence shows the converted state satisfying Gauss's law and
``∇·B = 0`` within the PSATD plan's ``constraint_tolerance``, the relative
Gauss residual and field energy preserved to roundoff, and the leapfrog energy
term by which the ledger's field energy changes meaning.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


D = phx.discretization
spectral = phx.solver.maxwell.spectral

COUNTS = (8, 4, 4)
SPACING = 0.125
STEP = 0.04
STEPS = 20
PLASMA_FREQUENCY_SQUARED = 16.0
WAVENUMBER = 2.0 * np.pi / (COUNTS[0] * SPACING)

grid = D.TensorGridPlan(
    tuple(D.UniformCellAxisSpec(n, periodic=True) for n in COUNTS),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * SPACING for n in COUNTS]]))
bridge = D.StructuredCochainBridge(grid)

ix, iy, iz, ip = np.meshgrid(*(np.arange(n) for n in COUNTS), (0, 1), indexing="ij")
ions = np.stack(
    ((ix + (ip + 0.5) / 2) * SPACING, (iy + 0.5) * SPACING, (iz + 0.5) * SPACING), -1
).reshape(-1, 3)
electrons = ions.copy()
electrons[:, 0] += 0.02 * np.sin(WAVENUMBER * ions[:, 0])
count = ions.shape[0]
weight = PLASMA_FREQUENCY_SQUARED * float(np.prod(COUNTS)) * SPACING**3 / count

species, charged = [], []
for offset, specific, name, mass in (
    (0, -1.0, "electrons", weight),
    (10**6, 1.0 / 1836.0, "ions", 1836.0 * weight),
):
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count), mass * jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    charged.append(
        D.ChargedParticlePlan(specific * mass * jnp.ones((count,)), name).prepare(support)
    )
    species.append(
        D.pic.PICSpeciesPlan(
            D.ParticlePopulationPlan(support),
            D.pic.PICChargeModelPlan(
                specific,
                name,
                minimum_charge_number=1,
                maximum_charge_number=1,
                initial_charge_number=1,
            ),
        )
    )
transfer_plan = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=2)
transfers = tuple(transfer_plan.prepare(value) for value in charged)
currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)

maxwell = phx.solver.CompatibleMaxwellPlan(
    bridge, sources=(phx.solver.PICMaxwellCurrentSourcePlan(),)
).prepare()
electrostatic = phx.solver.CochainElectrostaticPlan(
    bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
)
cochain = phx.solver.ElectromagneticPICPlan(
    phx.solver.CochainMaxwellPICFieldSolver(maxwell, electrostatic, transfers, currents),
    species=species,
)
psatd = phx.solver.ElectromagneticPICPlan(
    spectral.SpectralMaxwellPlan(
        bridge, grid="staggered", stencil="finite-order", stencil_order=2
    ).prepare(transfers, currents),
    species=species,
)


@eqx.filter_jit
def run(plan: Any, state: Any) -> tuple[Any, Any]:
    def body(value: Any, _: None) -> tuple[Any, Any]:
        result = plan.step_detailed(value, STEP)
        ledger = result.diagnostics.energy
        return result.accepted_state, (
            result.successful,
            ledger.previous_total,
            ledger.total,
        )

    return jax.lax.scan(body, state, None, length=STEPS)


centers = (np.arange(COUNTS[0]) + 0.5) * SPACING
zero = np.zeros(COUNTS)
magnetic = np.broadcast_to((0.3 * np.cos(WAVENUMBER * centers))[:, None, None], COUNTS)
velocity = np.zeros((count, 3))
state = cochain.initialize(
    (electrons, ions),
    (velocity, velocity),
    STEP,
    magnetic=bridge.pack_face_flux((zero, zero, magnetic)),
)
state, (before, _, before_total) = run(cochain, state)
handoff = phx.solver.hand_off_pic_state(cochain, psatd, state, STEP)
evidence = handoff.evidence
_, (after, after_previous, _) = run(psatd, handoff.state)
print(
    {
        "route": evidence.route,
        "handoff_successful": bool(evidence.successful),
        "psatd_constraints_satisfied": bool(evidence.constraint_satisfied),
        "cochain_gauss_residual": float(evidence.source_electric_constraint),
        "psatd_gauss_residual": float(evidence.target_electric_constraint),
        "psatd_magnetic_divergence": float(evidence.target_magnetic_constraint),
        "energy_relative_difference": float(evidence.energy_relative_difference),
        "leapfrog_energy_correction": float(evidence.leapfrog_energy_correction),
        "cochain_steps_successful": bool(jnp.all(before)),
        "psatd_steps_successful": bool(jnp.all(after)),
        "ledger_jump": float(after_previous[0] - before_total[-1]),
    }
)

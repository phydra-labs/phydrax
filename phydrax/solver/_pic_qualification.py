#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""PIC qualification gates and unreleased PIC capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

import jax.numpy as jnp
import numpy as np

from ..discretization import (
    ChargedParticlePlan,
    ParticlePopulationPlan,
    ParticleSetPlan,
    PreparedTensorGrid,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from ..discretization.pic import (
    ChargeConservingCurrentPlan,
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICParticleCochainTransferPlan,
    PICSpeciesPlan,
    RelativisticPusher,
    RelativisticPushPlan,
)
from ..qualification import CapabilityProfile, SupportTuple
from ._cochain_electrostatic import (
    CochainElectrostaticBoundaryPlan,
    CochainElectrostaticPlan,
)
from ._cochain_pic_field import CochainMaxwellPICFieldSolver
from ._electromagnetic_pic import ElectromagneticPICPlan
from ._electrostatic_pic import ElectrostaticPICPlan
from ._maxwell import CompatibleMaxwellPlan
from ._pic_current_source import PICMaxwellCurrentSourcePlan


_RADIATION_REACTION_GATES = (
    "planar-landau-lifshitz-cooling-analytic",
    "first-order-step-convergence",
    "radiated-equals-kinetic-loss",
    "reduced-equals-full-in-uniform-fields",
    "quantum-to-classical-limit",
    "niel-quantum-corrections",
    "fokker-planck-moment-equations",
    "identity-addressed-wiener-increments",
    "support-and-ownership-refusals",
    "species-charge-to-mass-refusal",
    "documentation-nonclaims",
)


@dataclass(frozen=True, slots=True)
class PICQualificationReport:
    """Host qualification evidence; every gate must pass for ``successful``."""

    poisson_residual: float
    charge_balance_defect: float
    electrostatic_energy_defect: float
    continuity_defect: float
    electromagnetic_continuity_defect: float
    particle_field_charge_defect: float
    gauss_defect: float
    deposit_gauss_pairing_defect: float
    pusher_speed_defect: float
    successful: bool


def _periodic_grid(count: int, dimension: int, /) -> PreparedTensorGrid:
    names = ("x", "y", "z")[:dimension]
    return TensorGridPlan(
        tuple(UniformCellAxisSpec(count, periodic=True) for _ in range(dimension)),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))


def _electrostatic_gates(smoke: bool, /) -> tuple[float, float, float, float, bool]:
    count = 16 if smoke else 32
    particle_count = 8 if smoke else 32
    bridge = StructuredCochainBridge(_periodic_grid(count, 1))
    base = (jnp.arange(particle_count, dtype=jnp.float64)[:, None] + 0.5) / particle_count
    transfers = []
    for offset, sign, name in ((0, -1.0, "negative"), (1000, 1.0, "positive")):
        support = ParticleSetPlan(
            jnp.arange(offset, offset + particle_count),
            jnp.ones((particle_count,)),
            ambient_dimension=1,
        ).prepare()
        charged = ChargedParticlePlan(sign * jnp.ones((particle_count,)), name).prepare(
            support
        )
        transfers.append(PICParticleCochainTransferPlan(bridge).prepare(charged))
    plan = ElectrostaticPICPlan(
        CochainElectrostaticPlan(
            bridge, CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        tuple(transfers),
    )
    state = plan.initialize(
        (base + 0.002 * jnp.sin(2.0 * jnp.pi * base), base),
        (jnp.zeros((particle_count, 1)), jnp.zeros((particle_count, 1))),
    )
    step = plan.step_detailed(state, 5.0e-4)
    diagnostics = step.diagnostics
    values = (
        float(diagnostics.poisson_residual),
        float(diagnostics.charge_balance_defect),
        float(diagnostics.energy.defect),
        float(diagnostics.continuity_defect),
    )
    passed = (
        bool(step.successful)
        and values[0] < 1.0e-8
        and values[1] < 1.0e-10
        and abs(values[2]) < 1.0e-8
        and values[3] < 1.0e-10
    )
    return (*values, passed)


def _electromagnetic_gates(smoke: bool, /) -> tuple[float, float, float, float, bool]:
    count = 3 if smoke else 4
    particle_count = 2 if smoke else 4
    bridge = StructuredCochainBridge(_periodic_grid(count, 3))
    transfer_plan = PICParticleCochainTransferPlan(bridge)
    transfers = []
    species = []
    for offset, sign, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        support = ParticleSetPlan(
            jnp.arange(offset, offset + particle_count),
            jnp.ones((particle_count,)),
            ambient_dimension=3,
        ).prepare()
        transfers.append(
            transfer_plan.prepare(
                ChargedParticlePlan(sign * jnp.ones((particle_count,)), name).prepare(
                    support
                )
            )
        )
        species.append(
            PICSpeciesPlan(
                ParticlePopulationPlan(support),
                PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    maxwell = CompatibleMaxwellPlan(
        bridge,
        sources=(PICMaxwellCurrentSourcePlan(),),
        plan_id="pic-qualification-maxwell",
    ).prepare()
    solver = CochainMaxwellPICFieldSolver(
        maxwell,
        CochainElectrostaticPlan(
            bridge, CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        tuple(transfers),
        tuple(ChargeConservingCurrentPlan(value) for value in transfers),
    )
    plan = ElectromagneticPICPlan(solver, species=species)
    slot = (np.arange(particle_count)[:, None] + np.asarray([0.3, 0.5, 0.7])) / (
        particle_count + 1
    )
    position = jnp.asarray(slot)
    velocity = jnp.zeros((particle_count, 3)).at[:, 0].set(0.1)
    dt = 0.01 * maxwell.stable_dt
    state = plan.initialize(
        (position + jnp.asarray([0.002, 0.0, 0.0]), position),
        (jnp.zeros((particle_count, 3)), velocity),
        dt,
    )
    step = plan.step_detailed(state, dt)
    diagnostics = step.diagnostics
    values = (
        float(diagnostics.continuity_defect),
        float(diagnostics.particle_field_charge_defect),
        float(diagnostics.electric_constraint),
        plan.pairing_defect,
    )
    passed = bool(step.successful) and max(values) < 1.0e-10
    return (*values, passed)


def _pusher_speed_defect() -> float:
    """Worst ``| |u⁺|² − |u⁻|² |`` of every pusher in a pure magnetic field."""
    proper = jnp.asarray([[0.2, 0.1, 0.0]])
    worst = 0.0
    for method in get_args(RelativisticPusher):
        pushed = RelativisticPushPlan(PIC_CODE_RELATIVITY, method=method).push(
            proper,
            jnp.zeros_like(proper),
            jnp.asarray([[0.0, 0.0, 0.7]]),
            jnp.asarray([1.0]),
            jnp.asarray([True]),
            1.0e-3,
        )
        if not bool(pushed.successful):
            return float("inf")
        worst = max(
            worst,
            float(jnp.abs(jnp.sum(pushed.proper_velocity**2) - jnp.sum(proper**2))),
        )
    return worst


def run_pic_qualification(*, smoke: bool = False) -> PICQualificationReport:
    """Evaluate the electrostatic, electromagnetic, and pusher PIC gates."""
    poisson, balance, energy, continuity, electrostatic_passed = _electrostatic_gates(
        smoke
    )
    em_continuity, charge_defect, gauss, pairing, electromagnetic_passed = (
        _electromagnetic_gates(smoke)
    )
    speed = _pusher_speed_defect()
    return PICQualificationReport(
        poisson,
        balance,
        energy,
        continuity,
        em_continuity,
        charge_defect,
        gauss,
        pairing,
        speed,
        electrostatic_passed and electromagnetic_passed and speed < 1.0e-10,
    )


def pic_radiation_reaction_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.radiation-reaction",
            {
                "models": "landau-lifshitz-reduced-full-quantum-corrected-fokker-planck",
                "coupling": "pic-momentum-stage-process-and-detector-propagation",
                "time_integration": "first-order-explicit-split-after-push",
                "field_derivatives": "order-one-gather-and-staggered-field-history",
                "quantum_corrections": "baier-katkov-g-and-h-tables-from-synchrotron-kernels",
                "stochasticity": "euler-maruyama-identity-addressed-wiener-increments",
                "ownership": "subgrid-reaction-with-grid-cutoff-scale-separation",
                "validity": "lcfa-declared-maximum-chi-and-minimum-gamma",
                "precision": "float64",
            },
        ),
    )


def pic_radiation_reaction_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_RADIATION_REACTION_GATES,
            released=False,
        )
        for support in pic_radiation_reaction_support_tuples()
    )


__all__ = [
    "PICQualificationReport",
    "pic_radiation_reaction_candidate_profiles",
    "pic_radiation_reaction_support_tuples",
    "run_pic_qualification",
]

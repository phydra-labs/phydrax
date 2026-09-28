#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A QED pair cascade in a rotating electric field (the node of two circularly
polarized counter-propagating lasers), run as a PIC creation-stage process."""

import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import pic
from phydrax.units import CHARGE, UnitDefinition


# Code units c = ε₀ = 1, time 1/ω₀ and fields m ω₀ c/e for a 1 μm laser:
# a_S = mc²/(ħω₀) = 4.1e5 and α = 1/137.036 with q = m = ħ a_S.
schwinger = 410000
hbar = 4 * Fraction(math.pi) * Fraction(1000, 137036) / schwinger**2
mass = float(hbar) * schwinger
scale = phx.ElectromagneticScaleContract.code_units(
    pic.PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=hbar,
    boltzmann_constant=1,
    elementary_charge=Fraction(mass),
    electron_mass=Fraction(mass),
    vacuum_permittivity=1,
    constant_set_id="qed-cascade-example",
)


class RotatingField(phx.StrictModule):
    amplitude: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return f"rotating-electric-{self.amplitude!r}"

    def external_fields(
        self, positions: jax.Array, times: jax.Array, /
    ) -> pic.ExternalFieldSample:
        del positions
        electric = self.amplitude * jnp.stack(
            (jnp.cos(times), jnp.sin(times), jnp.zeros_like(times)), axis=-1
        )
        return pic.ExternalFieldSample(
            electric, jnp.zeros_like(electric), jnp.ones(times.shape, dtype=bool)
        )


def species(sign: float, name: str, capacity: int, offset: int) -> pic.PICSpeciesPlan:
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + capacity), jnp.ones((capacity,)), ambient_dimension=1
    ).prepare()
    return pic.PICSpeciesPlan(
        phx.discretization.ParticlePopulationPlan(support),
        pic.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


compton = pic.NonlinearComptonPlan(
    "improved-lcfa",
    scale,
    -mass,
    mass,
    pic.QEDTable("nonlinear-compton", maximum_chi=100.0),
    maximum_chi=100.0,
    minimum_gamma=1.0,
    maximum_event_probability=0.2,
)
pairs = pic.NonlinearBreitWheelerPlan(
    scale,
    mass,
    mass,
    pic.QEDTable("nonlinear-breit-wheeler", maximum_chi=100.0),
    maximum_chi=100.0,
)
capacity = 512
photons = pic.QEDPhotonSpeciesPlan(
    8 * capacity,
    1,
    escape_lower=(0.0,),
    escape_upper=(100.0,),
    energy_edges=tuple(np.geomspace(1.0, 1.0e5, 11) * mass),
)
cascade = pic.QEDCascadeProcess(
    compton,
    photons,
    emitters=(0, 1),
    breit_wheeler=pairs,
    electron=0,
    positron=1,
    gather_species=0,
    minimum_photon_energy=50.0 * mass,
)
grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(64, periodic=True),), axis_names=("x",)
).prepare(jnp.asarray([[0.0], [100.0]]))
run = phx.solver.ElectromagneticPICPlan(
    phx.solver.ReducedMaxwellPICFieldSolver(
        phx.solver.CompatibleMaxwell1DPlan(grid), pic.ReducedPICTransferPlan(grid)
    ),
    species=(
        species(-1.0, "electrons", capacity, 0),
        species(1.0, "positrons", capacity, 10**5),
    ),
    processes=(cascade,),
    ownership="subgrid-reaction",
    external_fields=(RotatingField(1000.0),),
    key=jax.random.key(0),
)

# Eight electron–positron seed pairs at rest; negligible macroparticle weight.
seeds, dt = 8, 0.01
position = jnp.zeros((capacity, 1)).at[:seeds, 0].set(jnp.linspace(30.0, 70.0, seeds))
live = jnp.arange(capacity) < seeds
weight = jnp.where(live, 1.0e-9 * mass, 0.0)
state = run.initialize(
    (position, position),
    (jnp.zeros((capacity, 3)),) * 2,
    dt,
    active_masks=(live, live),
    masses=(weight, weight),
)


@eqx.filter_jit
def advance(
    plan: phx.solver.ElectromagneticPICPlan, state: phx.solver.ElectromagneticPICState
) -> tuple[phx.solver.ElectromagneticPICState, tuple[jax.Array, ...]]:
    def body(
        carry: phx.solver.ElectromagneticPICState, _: None
    ) -> tuple[phx.solver.ElectromagneticPICState, tuple[jax.Array, ...]]:
        result = plan.step_detailed(carry, dt)
        (evidence,) = result.diagnostics.process_evidence
        energy = result.diagnostics.energy
        return result.accepted_state, (
            result.successful,
            evidence.decays,
            evidence.outside_lcfa_validity,
            energy.radiated,
            energy.field_exchange,
            energy.defect,
        )

    return jax.lax.scan(body, state, None, length=100)


pairs_created = outside = 0
radiated = exchange = defect = 0.0
for _ in range(6):
    state, (successful, decays, invalid, rad, ex, err) = advance(run, state)
    if not bool(jnp.all(successful)):
        raise RuntimeError("a cascade step was rejected")
    pairs_created += int(jnp.sum(decays))
    outside += int(jnp.sum(invalid))
    radiated += float(jnp.sum(rad))
    exchange += float(jnp.sum(ex))
    defect = max(defect, float(jnp.max(jnp.abs(err))))
    count = int(jnp.sum(state.species[0].population.active))
    print(
        f"t = {float(state.time):4.1f}: {count} electrons, {pairs_created} pairs created"
    )

bank = state.processes[0].photons
print(
    f"photons banked {int(jnp.sum(bank.population.active))}, "
    f"events outside LCFA validity {outside}"
)
print(
    f"radiated {radiated:.3e}, field exchange {exchange:.3e}, "
    f"largest ledger defect {defect:.1e} (work of the external field)"
)

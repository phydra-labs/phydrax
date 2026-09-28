#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Self-consistent PIC Cherenkov radiation of a relativistic beam in a dispersive medium.

A weakly coupled macroparticle (huge mass, unit charge) moves at ``β = 0.9``
along the periodic ``z`` axis through a Lorentz dielectric
``ε(ω) = ε∞ + f/(ω₀² − ω² − iγω)``; an immobile compensating charge at its start
makes the run begin from zero field. ``y`` is four periodic cells (a line
charge) and CPML absorbs along the bounded ``x`` axis, where grounded walls
initialize the electrostatic field. Quadratic spline shapes make the gathered
field continuous across the cell faces the particle crosses, so the energy
ledger is second order in ``Δt``.

The run is repeated at ``Δt`` and ``Δt/2``. The synchronized energy ledger
(field, medium, particle, and the CPML and pole losses as dissipated energy)
must fall fourfold, so the fine defect is bounded by the Richardson estimate
``|D(Δt) − D(Δt/2)|/3``. The cone angle of the dominant band, the first
harmonic ``ω₁ = 2πv/L_z`` (higher harmonics sit in the pole's absorption band
or below the Cherenkov threshold), is measured from the phase gradient of its
DFT phasor and compared with the discrete cone of the dispersion audit of the
executed update at the same ``Δt`` and with the continuum ``cos θ = 1/(β n(ω₁))``.
"""

from __future__ import annotations

import equinox as eqx
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

cells_x, cells_y, cells_z, spacing = 32, 4, 12, 0.05
beta, mass = 0.9, 1.0e12
grid = D.TensorGridPlan(
    (
        D.UniformCellAxisSpec(cells_x),
        D.UniformCellAxisSpec(cells_y, periodic=True),
        D.UniformCellAxisSpec(cells_z, periodic=True),
    ),
    axis_names=("x", "y", "z"),
).prepare(
    jnp.asarray(
        [[0.0, 0.0, 0.0], [cells_x * spacing, cells_y * spacing, cells_z * spacing]]
    )
)
bridge = D.StructuredCochainBridge(grid)
period = cells_z * spacing / beta
omega = 2.0 * np.pi / period
constitutive = mx.LorentzDrudeMaxwellConstitutivePlan(
    mx.MaxwellLorentzPoles([20.0], [0.5], [240.0]), permittivity_infinity=1.5
)
species, charged = [], []
for offset, charge, name in ((0, 1.0, "beam"), (10, -1.0, "compensator")):
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + 1), mass * jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    charged.append(D.ChargedParticlePlan(jnp.asarray([charge]), name).prepare(support))
    species.append(
        PIC.PICSpeciesPlan(
            D.ParticlePopulationPlan(support),
            PIC.PICChargeModelPlan(
                charge / mass,
                name,
                minimum_charge_number=1,
                maximum_charge_number=1,
                initial_charge_number=1,
            ),
        )
    )
# E_z along x in the beam plane, between the beam's near field and the CPML.
probe_x = np.arange(cells_x // 2 + 3, cells_x - 8)
probe = bridge.orientation_offsets[1][2] + np.ravel_multi_index(
    (probe_x, np.zeros_like(probe_x), np.zeros_like(probe_x)),
    bridge.orientation_shapes[1][2],
)


def maxwell_runtime(start_time: float, stop_time: float) -> mx.PreparedCompatibleMaxwell:
    return phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=constitutive,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        observers=(
            mx.DFTObserverPlan(
                mx.FieldProbePlan("electric", jnp.asarray(probe)),
                mx.MaxwellSpectralAcquisition(
                    jnp.asarray([omega]),
                    sign="positive",
                    measure="sample-mean",
                    start_time=start_time,
                    stop_time=stop_time,
                ),
            ),
        ),
        pml=mx.MaxwellCPMLPlan((6, 0, 0)),
    ).prepare()


State = phx.solver.ElectromagneticPICState
Diagnostics = phx.solver.ElectromagneticPICDiagnostics


@eqx.filter_jit
def run(
    plan: phx.solver.ElectromagneticPICPlan,
    state: State,
    step_size: jax.Array,
    steps: int,
) -> tuple[State, Diagnostics]:
    def body(current: State, _: None) -> tuple[State, Diagnostics]:
        result = plan.step_detailed(current, step_size)
        return result.accepted_state, result.diagnostics

    return jax.lax.scan(body, state, None, length=steps)


def cherenkov(per_period: int) -> dict[str, float]:
    """Run four beam periods with ``per_period`` steps each; ledger and cone."""
    dt = period / per_period
    steps = 4 * per_period
    end = steps * dt
    # The DFT acquires the last period, after the start-up transient has left.
    maxwell = maxwell_runtime(end - period + 0.5 * dt, end + 0.5 * dt)
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge,
        phx.solver.CochainElectrostaticBoundaryPlan.dirichlet(bridge),
        permittivity=maxwell.constitutive.electric_displacement(
            jnp.ones((maxwell.layout.electric_count,)),
            maxwell.constitutive.initialize_state(),
        ),
    )
    transfers = tuple(
        PIC.PICParticleCochainTransferPlan(bridge, shape_order=2).prepare(value)
        for value in charged
    )
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        electrostatic,
        transfers,
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )
    # The absolute continuity roundoff grows with the unwrapped periodic coordinate.
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=species,
        maximum_displacement_fraction=1.0,
        continuity_tolerance=1.0e-8,
    )
    start = jnp.asarray(
        [[0.5 * cells_x * spacing, 0.5 * cells_y * spacing, 0.3 * spacing]]
    )
    state = pic.initialize(
        (start, start), (jnp.asarray([[0.0, 0.0, beta]]), jnp.zeros((1, 3))), dt
    )
    initial = pic.synchronized_energy(state, dt)
    state, diagnostics = run(pic, state, jnp.asarray(dt), steps)
    final = pic.synchronized_energy(state, dt)
    dissipated_steps = diagnostics.energy.dissipated
    if dissipated_steps is None or final.material is None:
        raise RuntimeError(
            "The cochain PIC field solver accounts medium energy and losses."
        )
    if not bool(jnp.all(diagnostics.successful)):
        raise RuntimeError("Every PIC step must be accepted.")
    dissipated = float(jnp.sum(dissipated_steps))
    phasor = np.asarray(maxwell.observe(state.field)[0]).reshape(-1)
    wavenumber_x = np.polyfit(probe_x * spacing, np.unwrap(np.angle(phasor)), 1)[0]
    # Discrete cone of the executed update: the dispersion audit's Bloch branches
    # of a homogeneous periodic block at the same h and Δt.
    block = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(8, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [8.0 * spacing] * 3]))
    audit = mx.CompatibleMaxwellDispersionAudit(
        phx.solver.CompatibleMaxwellPlan(
            D.StructuredCochainBridge(block), constitutive=constitutive
        ).prepare(),
        mx.MaxwellMaterialRegion((0, 0, 0), (8, 8, 8)),
        dt,
    )
    regime = mx.CherenkovRegimePlan(
        audit, jnp.asarray([0.0, 0.0, beta]), jnp.asarray([omega]), 1
    ).evaluate()
    cones = np.asarray(regime.numerical_cone_angle).reshape(-1)
    return {
        "dt": dt,
        "gauss": float(jnp.max(diagnostics.electric_constraint)),
        "charge_ledger": float(jnp.max(diagnostics.charge_ledger_defect)),
        "material": float(final.material),
        "dissipated": dissipated,
        "ledger": float(final.total - initial.total) + dissipated,
        "measured": float(np.degrees(np.arctan2(abs(wavenumber_x), omega / beta))),
        "audited": float(np.degrees(cones[np.isfinite(cones)][0])),
        "continuum": float(np.degrees(np.asarray(regime.physical_cone_angle).ravel()[0])),
        "index": float(np.asarray(regime.continuum_index).ravel()[0]),
    }


stable = float(maxwell_runtime(0.0, 1.0).stable_dt)
per_period = int(np.ceil(period / (0.9 * stable)))
coarse, fine = cherenkov(per_period), cherenkov(2 * per_period)
for label, result in (("Δt", coarse), ("Δt/2", fine)):
    print(
        f"{label:>4}: dt={result['dt']:.3e} gauss={result['gauss']:.1e} "
        f"charge_ledger={result['charge_ledger']:.1e} material={result['material']:.3e} "
        f"dissipated={result['dissipated']:.3e} energy_ledger={result['ledger']:+.3e}"
    )
bound = abs(coarse["ledger"] - fine["ledger"]) / 3.0
print(
    f"ledger ratio={coarse['ledger'] / fine['ledger']:.2f} (second order: 4); "
    f"|ledger(Δt/2)|={abs(fine['ledger']):.3e} <= Richardson bound "
    f"{bound:.3e}: {abs(fine['ledger']) <= bound}"
)
print(
    f"n(ω₁)={fine['index']:.4f} cone measured={fine['measured']:.2f} deg "
    f"audited discrete={fine['audited']:.2f} deg continuum={fine['continuum']:.2f} deg"
)

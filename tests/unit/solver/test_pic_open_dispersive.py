#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Open-boundary, dispersive, and magnetized self-consistent electromagnetic PIC.

References are independent of the runtime:

* Gauss and charge ledgers: ``−δD`` (the Maxwell Gauss charge) against the
  particle charge redeposited from the particle state, and the particle count
  and macrocharge before and after absorbing faces.
* Exit ledger: ``(γ − 1)mc²`` of the absorbed particles from their proper
  velocities (NumPy), interpolated to the face.
* Energy ledger: the synchronized total energy plus the dissipated and exited
  energy is conserved to the order of the leapfrog, measured by halving ``Δt``
  at fixed ``h`` over one physical interval.
* Cherenkov: a macroparticle of huge mass (weak coupling: its self-field cannot
  bend the path) with an immobile compensating charge is the coincident-neutral
  prescribed charge of `solve_prescribed_charge_maxwell` (B4), so the PIC field,
  its DFT phasors, and the cone ``cos θ = 1/(βn)`` must reproduce that run.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell


def _species(
    count: int,
    charge: float,
    mass: float,
    name: str,
    offset: int,
    dimension: int = 3,
) -> tuple[Any, Any]:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count),
        mass * jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()
    species = PIC.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        PIC.PICChargeModelPlan(
            charge / mass,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )
    charged = D.ChargedParticlePlan(charge * jnp.ones((count,)), name).prepare(support)
    return species, charged


def _cochain_solver(
    bridge: Any, maxwell: Any, charged: tuple[Any, ...], *, periodic: bool = False
) -> Any:
    constitutive = maxwell.constitutive
    permittivity = constitutive.electric_displacement(
        jnp.ones((maxwell.layout.electric_count,)), constitutive.initialize_state()
    )
    boundary = (
        phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        if periodic
        else phx.solver.CochainElectrostaticBoundaryPlan.dirichlet(bridge)
    )
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge, boundary, permittivity=permittivity
    )
    transfers = tuple(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(value) for value in charged
    )
    return phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        electrostatic,
        transfers,
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )


@eqx.filter_jit
def _run(plan: Any, state: Any, step_size: Any, steps: int) -> tuple[Any, Any]:
    def body(current: Any, _: None) -> tuple[Any, Any]:
        result = plan.step_detailed(current, step_size)
        return result.accepted_state, result.diagnostics

    return jax.lax.scan(body, state, None, length=steps)


# -- reduced 2-D nonperiodic pairing ------------------------------------------


@pytest.mark.parametrize(
    "periodic",
    [(False, True), (True, False), (False, False)],
    ids=["x-open", "y-open", "both-open"],
)
def test_reduced_2d_nonperiodic_pic_pairs_gauss_with_the_deposit(
    periodic: tuple[bool, bool],
) -> None:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(12, periodic=periodic[0]),
            D.UniformCellAxisSpec(10, periodic=periodic[1]),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    wall = mx.MaxwellBoundaryPlan("pmc")
    field = phx.solver.CompatibleMaxwell2DPlan(
        grid,
        boundaries=tuple((None, None) if value else (wall, wall) for value in periodic),
    )
    solver = phx.solver.ReducedMaxwellPICFieldSolver(
        field, PIC.ReducedPICTransferPlan(grid)
    )
    electrons, _ = _species(4, -1.0, 1.0, "electrons", 0, 2)
    ions, _ = _species(4, 1.0, 1.0, "ions", 10, 2)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=(electrons, ions))
    assert pic.pairing_defect < 1e-13
    position = jnp.asarray([[0.3, 0.4], [0.5, 0.5], [0.62, 0.31], [0.45, 0.77]])
    velocity = jnp.zeros((4, 3)).at[:, 0].set(0.1).at[:, 1].set(-0.05)
    dt = 0.4 * field.stable_dt
    state = pic.initialize(
        (position + 0.013, position), (velocity, jnp.zeros((4, 3))), dt
    )
    state, diagnostics = _run(pic, state, dt, 20)
    assert bool(jnp.all(diagnostics.successful))
    assert float(jnp.max(diagnostics.particle_field_charge_defect)) < 1e-11
    # Gauss's law ε∇·E = ρ against the redeposited particle charge; PMC walls
    # induce no charge, so it holds on every cell.
    charge, _ = pic.species_charge(state.species)
    gauss = field.divergence_electric(state.field)
    scale = float(jnp.max(jnp.abs(charge)))
    np.testing.assert_allclose(gauss, charge, rtol=0.0, atol=1e-12 * scale)


def test_reduced_2d_open_pic_initializes_gauss_from_filtered_charge() -> None:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(16), D.UniformCellAxisSpec(12)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    wall = mx.MaxwellBoundaryPlan("pmc")
    field = phx.solver.CompatibleMaxwell2DPlan(
        grid, boundaries=((wall, wall), (wall, wall))
    )
    solver = phx.solver.ReducedMaxwellPICFieldSolver(
        field, PIC.ReducedPICTransferPlan(grid)
    )
    electrons, _ = _species(3, -1.0, 1.0, "electrons", 0, 2)
    filters = (phx.solver.PICFilterPlan(passes=2),)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=(electrons,), filters=filters)
    position = jnp.asarray([[0.41, 0.37], [0.55, 0.52], [0.63, 0.48]])
    state = pic.initialize((position,), (jnp.zeros((3, 3)),), 0.3 * field.stable_dt)
    charge, _ = pic.species_charge(state.species)
    filtered = filters[0].filter_charge(solver, charge)
    # A bounded 2-D field carries non-neutral charge: Gauss holds cell by cell
    # against the filtered, not the raw, deposit.
    gauss = field.divergence_electric(state.field)
    scale = float(jnp.max(jnp.abs(filtered)))
    np.testing.assert_allclose(gauss, filtered, rtol=0.0, atol=1e-11 * scale)
    assert float(jnp.max(jnp.abs(filtered - charge))) > 1e-3 * scale


# -- open 3-D cochain PIC: exit, charge, and Gauss ledgers ---------------------


def _open_box(cells: int, width: int, constitutive: Any) -> tuple[Any, Any]:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(cells) for _ in range(3)), axis_names=("x", "y", "z")
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = D.StructuredCochainBridge(grid)
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=constitutive,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        pml=mx.MaxwellCPMLPlan((width,) * 3),
    ).prepare()
    return bridge, maxwell


def test_open_pic_absorbs_particles_with_charge_mass_energy_and_gauss_ledgers() -> None:
    cells, width = 12, 3
    h = 1.0 / cells
    bridge, maxwell = _open_box(cells, width, mx.DiagonalMaxwellConstitutivePlan())
    # Huge masses keep velocities fixed, so every exit energy is (γ − 1)mc² of
    # the launch velocity while the unit charges still radiate into the CPML.
    mass = 1.0e9
    electrons, electron_charge = _species(4, -1.0, mass, "electrons", 0)
    ions, ion_charge = _species(4, 1.0, mass, "ions", 10)
    solver = _cochain_solver(bridge, maxwell, (electron_charge, ion_charge))
    kind = PIC.PICBoundaryKind
    inset = width * h
    boundary = PIC.PICOpenBoundaryPlan(
        jnp.full((3,), inset), jnp.full((3,), 1.0 - inset), kinds=(kind.ABSORB,) * 6
    )
    pic = phx.solver.ElectromagneticPICPlan(
        solver, species=(electrons, ions), boundaries=boundary
    )
    position = (
        jnp.asarray(
            [[5.4, 6.45, 5.5], [6.5, 5.4, 6.55], [5.55, 6.52, 6.45], [6.48, 5.55, 5.6]]
        )
        * h
    )
    # Launch directions +z, −x, +y, −z at distinct speeds.
    velocity = jnp.asarray(
        [[0.0, 0.0, 0.8], [-0.6, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, -0.9]]
    )
    dt = 0.5 * maxwell.stable_dt
    state = pic.initialize((position, position), (velocity, jnp.zeros((4, 3))), dt)
    state, diagnostics = _run(pic, state, dt, 30)
    assert bool(jnp.all(diagnostics.successful))
    assert not bool(jnp.any(state.species[0].population.active))
    assert bool(jnp.all(state.species[1].population.active))
    # Charge ledger: every step's particle charge change is its exit charge.
    assert float(jnp.max(diagnostics.charge_ledger_defect)) < 1e-15
    np.testing.assert_allclose(jnp.sum(diagnostics.exit.charge), -4.0, rtol=1e-14)
    np.testing.assert_allclose(jnp.sum(diagnostics.exit.mass), 4.0 * mass, rtol=1e-14)
    surface = state.boundaries[0]
    gamma = 1.0 / np.sqrt(1.0 - np.sum(np.asarray(velocity) ** 2, axis=1))
    kinetic = mass * (gamma - 1.0)
    # Faces are (−x, +x, −y, +y, −z, +z).
    expected_charge = np.asarray([-1.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    expected_energy = np.asarray(
        [kinetic[1], 0.0, 0.0, kinetic[2], kinetic[3], kinetic[0]]
    )
    np.testing.assert_allclose(surface.collected_charge, expected_charge, atol=1e-14)
    np.testing.assert_allclose(
        surface.collected_kinetic_energy, expected_energy, rtol=1e-6, atol=0.0
    )
    np.testing.assert_allclose(
        jnp.sum(diagnostics.exit.kinetic_energy), np.sum(kinetic), rtol=1e-6
    )
    np.testing.assert_allclose(
        jnp.sum(diagnostics.energy.exited), np.sum(kinetic), rtol=1e-6
    )
    # Gauss: −δD equals the redeposited particle charge plus the frozen exit
    # charge away from the absorber, whose stretched curl carries bookkeeping
    # charge only inside the layer.
    particles, _ = pic.species_charge(state.species)
    gauss = -bridge.codifferential(1, state.field.primary.electric_displacement)
    residual = np.asarray(gauss - particles - state.wall_charge).reshape((cells + 1,) * 3)
    interior = residual[
        width + 1 : -width - 1, width + 1 : -width - 1, width + 1 : -width - 1
    ]
    scale = float(jnp.max(jnp.abs(particles)))
    assert np.max(np.abs(interior)) < 1e-10 * scale
    assert float(jnp.max(diagnostics.electric_constraint)) < 1e-10 * scale
    # The absorber removed radiated field energy, which the ledger accounts.
    assert float(jnp.sum(diagnostics.energy.dissipated)) > 0.0


# -- weak-coupling relativistic beam in a dielectric = prescribed charge -------


def test_weak_coupling_beam_in_dielectric_reproduces_prescribed_cherenkov() -> None:
    cells_x, cells_z, h = 24, 12, 0.05
    beta, index = 0.9, 1.5
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(cells_x),
            D.UniformCellAxisSpec(2, periodic=True),
            D.UniformCellAxisSpec(cells_z, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [cells_x * h, 2 * h, cells_z * h]]))
    bridge = D.StructuredCochainBridge(grid)
    medium: dict[str, Any] = {
        "constitutive": mx.DiagonalMaxwellConstitutivePlan(permittivity=index**2),
        "pml": mx.MaxwellCPMLPlan((6, 0, 0)),
    }
    stable = float(phx.solver.CompatibleMaxwellPlan(bridge, **medium).prepare().stable_dt)
    period = cells_z * h / beta
    per_period = int(np.ceil(period / (0.9 * stable)))
    dt = period / per_period
    periods = 3
    steps = periods * per_period
    times = dt * np.arange(steps + 1)
    start = np.asarray([0.5 * cells_x * h, 0.5 * h, 0.3 * h])
    omega = 2.0 * np.pi / period
    probe_x = np.arange(cells_x // 2 + 2, cells_x - 7)
    probe = bridge.orientation_offsets[1][2] + np.ravel_multi_index(
        (probe_x, np.zeros_like(probe_x), np.zeros_like(probe_x)),
        bridge.orientation_shapes[1][2],
    )
    observer = mx.DFTObserverPlan(
        mx.FieldProbePlan("electric", jnp.asarray(probe)),
        mx.MaxwellSpectralAcquisition(
            jnp.asarray([omega]),
            sign="positive",
            measure="sample-mean",
            # Half-step margins keep the window's samples (one period) the same
            # under the two runtimes' accumulated clocks.
            start_time=float(times[-1] - period + 0.5 * dt),
            stop_time=float(times[-1] + 0.5 * dt),
        ),
    )
    # Prescribed charge (B4): unit charge on the Maxwell step grid.
    positions = start[None, None, :] + beta * times[:, None, None] * np.asarray(
        [0.0, 0.0, 1.0]
    )
    single, single_charge = _species(1, 1.0, 1.0, "prescribed", 0)
    current = PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(single_charge)
    )
    trajectory = mx.PrescribedChargeTrajectory(times, positions)
    prescribed = mx.solve_prescribed_charge_maxwell(
        mx.PrescribedChargeMaxwellPlan(
            phx.solver.CompatibleMaxwellPlan(
                bridge,
                sources=(mx.PrescribedChargeCurrentSourcePlan(trajectory, current),),
                observers=(observer,),
                **medium,
            ).prepare(),
            current,
            trajectory,
            np.asarray([1.0]),
        )
    )
    # Self-consistent PIC: beam macroparticle plus immobile compensator, both of
    # mass 1e12, start coincident (zero field, as the prescribed run).
    beam, beam_charge = _species(1, 1.0, 1.0e12, "beam", 0)
    compensator, compensator_charge = _species(1, -1.0, 1.0e12, "compensator", 10)
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        observers=(observer,),
        **medium,
    ).prepare()
    solver = _cochain_solver(bridge, maxwell, (beam_charge, compensator_charge))
    pic = phx.solver.ElectromagneticPICPlan(
        solver, species=(beam, compensator), maximum_displacement_fraction=1.0
    )
    initial = jnp.asarray(start)[None, :]
    state = pic.initialize(
        (initial, initial),
        (jnp.asarray([[0.0, 0.0, beta]]), jnp.zeros((1, 3))),
        dt,
    )
    state, diagnostics = _run(pic, state, jnp.asarray(dt), steps)
    assert bool(jnp.all(diagnostics.successful))
    reference = np.asarray(prescribed.final_state.primary.electric_displacement)
    field = np.asarray(state.field.primary.electric_displacement)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0.0, atol=1e-7 * scale)
    phasor = np.asarray(maxwell.observe(state.field)[0]).reshape(-1)
    expected = np.asarray(prescribed.observations[0]).reshape(-1)
    np.testing.assert_allclose(
        phasor, expected, rtol=0.0, atol=1e-7 * np.max(np.abs(expected))
    )
    # The radiated Bloch wave leaves at the Cherenkov angle cos θ = 1/(βn).
    wavenumber_x = np.polyfit(probe_x * h, np.unwrap(np.angle(phasor)), 1)[0]
    angle = np.degrees(np.arctan2(abs(wavenumber_x), omega / beta))
    assert abs(angle - np.degrees(np.arccos(1.0 / (beta * index)))) < 3.0


# -- dispersive and magnetized energy ledgers -----------------------------------


# Second-order ledger in every medium: halving Δt divides the defect by ≈4.
_MEDIA: dict[str, Any] = {
    "cpml-vacuum": (mx.DiagonalMaxwellConstitutivePlan(), True),
    "lossy": (mx.ConductiveMaxwellConstitutivePlan(electric_conductivity=0.5), False),
    "lorentz-drude": (
        mx.LorentzDrudeMaxwellConstitutivePlan(
            mx.MaxwellLorentzPoles([3.0], [0.5], [4.0]), permittivity_infinity=1.5
        ),
        False,
    ),
    "negative-index": (
        mx.LorentzDrudeMaxwellConstitutivePlan(
            mx.MaxwellLorentzPoles([0.0], [0.3], [9.0]),
            magnetic_poles=mx.MaxwellLorentzPoles([0.0], [0.3], [9.0]),
        ),
        False,
    ),
    "magnetized-plasma": (
        mx.MagnetizedColdPlasmaMaxwellConstitutivePlan(
            jnp.asarray([2.0]), jnp.asarray([[0.0, 0.0, 3.0]]), collision_frequency=0.2
        ),
        False,
    ),
}


@pytest.mark.parametrize("medium", sorted(_MEDIA))
def test_self_consistent_energy_ledger_converges_under_step_halving(medium: str) -> None:
    constitutive, absorbing = _MEDIA[medium]
    cells = 12
    h = 1.0 / cells
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(cells) for _ in range(3)), axis_names=("x", "y", "z")
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = D.StructuredCochainBridge(grid)
    walls: dict[str, Any] = (
        {"pml": mx.MaxwellCPMLPlan((3, 3, 3))}
        if absorbing
        else {"boundaries": (mx.MaxwellBoundaryPlan("pec"),)}
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=constitutive,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        **walls,
    ).prepare()
    electrons, electron_charge = _species(4, -0.05, 1.0, "electrons", 0)
    ions, ion_charge = _species(4, 0.05, 1.0, "ions", 10)
    solver = _cochain_solver(bridge, maxwell, (electron_charge, ion_charge))
    pic = phx.solver.ElectromagneticPICPlan(solver, species=(electrons, ions))
    # Paths stay inside their cells (the Whitney gather is smooth there).
    position = (
        jnp.asarray(
            [[5.4, 6.45, 5.5], [6.5, 5.4, 6.55], [5.55, 6.52, 6.45], [6.48, 5.55, 5.6]]
        )
        * h
    )
    velocity = jnp.zeros((4, 3)).at[:, 2].set(0.05).at[:, 0].set(0.05 / 6.0)
    interval = 12.0 * float(maxwell.stable_dt)
    defects = []
    for steps in (60, 120):
        dt = interval / steps
        state = pic.initialize((position, position), (velocity, jnp.zeros((4, 3))), dt)
        first = pic.synchronized_energy(state, dt)
        state, diagnostics = _run(pic, state, jnp.asarray(dt), steps)
        assert bool(jnp.all(diagnostics.successful))
        dissipated = diagnostics.energy.dissipated
        assert dissipated is not None
        last = pic.synchronized_energy(state, dt)
        defects.append(
            abs(float(last.total - first.total + jnp.sum(dissipated)))
            / float(first.total)
        )
        if medium != "cpml-vacuum" and medium != "lossy":
            assert last.material is not None and float(last.material) > 0.0
        # Passive media and absorbers only remove energy.
        assert float(jnp.sum(dissipated)) >= 0.0
    assert defects[1] < 1e-4, defects
    assert 3.6 < defects[0] / defects[1] < 4.4, defects

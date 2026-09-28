#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic


def _species(
    capacity: int,
    sign: float,
    name: str,
    dimension: int,
    offset: int,
    *,
    maximum_charge_number: int = 1,
) -> Any:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + capacity),
        jnp.ones((capacity,)),
        ambient_dimension=dimension,
    ).prepare()
    return PIC.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        PIC.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=maximum_charge_number,
            initial_charge_number=1,
        ),
    )


def _grid_1d(count: int, *, periodic: bool) -> Any:
    return D.TensorGridPlan(
        (D.UniformCellAxisSpec(count, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _reduced_solver(count: int = 16, *, periodic: bool = True) -> Any:
    grid = _grid_1d(count, periodic=periodic)
    boundaries = None
    if not periodic:
        pec = phx.solver.maxwell.MaxwellBoundaryPlan("pec")
        boundaries = ((pec, pec),)
    field = phx.solver.CompatibleMaxwell1DPlan(grid, boundaries=boundaries)
    return phx.solver.ReducedMaxwellPICFieldSolver(
        field, PIC.ReducedPICTransferPlan(grid)
    )


def _neutral_pair() -> tuple[Any, Any]:
    return _species(4, -1.0, "electrons", 1, 0), _species(4, 1.0, "ions", 1, 10)


_POSITIONS = jnp.asarray([[0.1], [0.3], [0.55], [0.8]])


def _initial_state(pic: Any, step_size: Any) -> Any:
    velocity = jnp.zeros((4, 3)).at[:, 0].set(0.05)
    return pic.initialize(
        (_POSITIONS + 0.01, _POSITIONS), (velocity, jnp.zeros((4, 3))), step_size
    )


def test_order_one_gather_matches_in_cell_difference_of_the_interpolant() -> None:
    solver = _reduced_solver(16)
    x = (jnp.arange(16) + 0.5) / 16
    zero = jnp.zeros((16,))
    field = solver.field.initialize(
        electric=(zero, jnp.sin(2.0 * jnp.pi * x), zero),
        magnetic=(zero, zero, jnp.cos(2.0 * jnp.pi * x)),
    )
    # Between two cell centers the CIC interpolant is linear in x.
    position = jnp.asarray([[0.21], [0.62]])
    active = jnp.asarray([True, True])
    sample = solver.gather(0, position, active, field, derivative_order=1)
    h = 1.0e-4
    plus = solver.gather(0, position + h, active, field)
    minus = solver.gather(0, position - h, active, field)
    assert sample.successful
    assert sample.electric_gradient.shape == (2, 3, 3)
    np.testing.assert_allclose(
        sample.electric_gradient[:, :, 0],
        (plus.electric - minus.electric) / (2.0 * h),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        sample.magnetic_gradient[:, :, 0],
        (plus.magnetic - minus.magnetic) / (2.0 * h),
        rtol=1e-8,
        atol=1e-8,
    )
    # dD3V fields are invariant along the unresolved directions.
    np.testing.assert_array_equal(sample.electric_gradient[:, :, 1:], 0.0)
    with pytest.raises(ValueError):
        solver.gather(0, position, active, field, derivative_order=2)


def test_spectral_symbol_predicts_the_standing_wave_frequency_of_the_update() -> None:
    solver = _reduced_solver(32)
    plan = solver.field
    mode = 3
    k = 2.0 * jnp.pi * mode
    x = (jnp.arange(32) + 0.5) / 32
    zero = jnp.zeros((32,))
    state = plan.initialize(electric=(zero, jnp.cos(k * x), zero))
    dt = 0.9 * plan.stable_dt
    steps = 40
    for _ in range(steps):
        state, diagnostics = plan.step(state, (zero, zero, zero), dt)
        assert diagnostics.successful
    omega = solver.dispersion_frequency(jnp.asarray([[k]]), dt)[0]
    # A Störmer–Verlet eigenmode started with B = 0 evolves as cos(n ω Δt).
    np.testing.assert_allclose(
        state.electric[1], jnp.cos(steps * omega * dt) * jnp.cos(k * x), atol=1e-10
    )
    assert not np.isclose(float(omega), float(k))


def test_restart_components_round_trip_and_admit_only_matching_owners() -> None:
    electrons, ions = _neutral_pair()
    solver = _reduced_solver()
    pic = phx.solver.ElectromagneticPICPlan(solver, species=(electrons, ions))
    dt = 0.2 * solver.field.stable_dt
    state = pic.step_detailed(_initial_state(pic, dt), dt).accepted_state
    checkpoint = pic.checkpoint(state)
    restored = pic.restore(checkpoint)
    for left, right in zip(
        jax.tree.leaves(restored), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    continued = pic.step_detailed(restored, dt).accepted_state
    reference = pic.step_detailed(state, dt).accepted_state
    for left, right in zip(
        jax.tree.leaves(continued), jax.tree.leaves(reference), strict=True
    ):
        np.testing.assert_array_equal(left, right)

    # Stateless processes change the run plan but not any restart component owner.
    collided = phx.solver.ElectromagneticPICPlan(
        solver,
        species=(electrons, ions),
        processes=(
            PIC.collisions.CoulombCollisionProcess(
                PIC.collisions.CoulombCollisionPlan(1.0, maximum_probability=0.2), 0
            ),
        ),
        key=jr.key(0),
    )
    assert collided.plan_id != pic.plan_id
    np.testing.assert_array_equal(
        collided.restore(checkpoint).species[0].particles.position,
        state.species[0].particles.position,
    )
    other_field = phx.solver.ElectromagneticPICPlan(
        _reduced_solver(32), species=(electrons, ions)
    )
    with pytest.raises(ValueError, match="not admitted"):
        other_field.restore(checkpoint)
    other_species = phx.solver.ElectromagneticPICPlan(
        solver, species=(electrons, _species(4, 1.0, "protons", 1, 10))
    )
    with pytest.raises(ValueError, match="not admitted"):
        other_species.restore(checkpoint)


def test_field_ionization_process_creates_identified_neutral_electrons() -> None:
    ions = _species(4, 1.0, "ions", 1, 0, maximum_charge_number=2)
    electrons = _species(8, -1.0, "electrons", 1, 100)
    process = PIC.ionization.FieldIonizationProcess(
        PIC.ionization.FieldIonizationPlan(
            1.0e12,
            field_power=1.0,
            ionization_energy=0.1,
            maximum_probability=1.0,
            maximum_events=2,
        ),
        0,
        1,
    )
    solver = _reduced_solver()
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=(ions, electrons),
        processes=(process,),
        key=jr.key(3),
    )
    active = jnp.asarray([True] * 4 + [False] * 4)
    dt = 0.2 * solver.field.stable_dt
    state = pic.initialize(
        (_POSITIONS, jnp.concatenate((_POSITIONS + 0.02, jnp.zeros((4, 1))))),
        (jnp.zeros((4, 3)), jnp.zeros((8, 3)).at[:4, 0].set(0.05)),
        dt,
        active_masks=(None, active),
        masses=(None, jnp.where(active, 1.0, 0.0)),
    )
    result = pic.step_detailed(state, dt)
    assert result.successful
    (ledger,) = result.diagnostics.processes
    assert ledger.successful
    assert int(ledger.event_count) == 2
    assert result.diagnostics.process_charge_defect < 1e-12
    assert result.diagnostics.particle_field_charge_defect < 1e-12
    ion_state, electron_state = result.accepted_state.species
    born = electron_state.population.active & ~active
    assert int(jnp.sum(born)) == 2
    assert int(jnp.sum(ion_state.charge.charge_number)) == 4 + 2
    # Every new electron is born at its parent ion with that ion's identity.
    ionized = ion_state.charge.charge_number == 2
    np.testing.assert_array_equal(
        np.sort(np.asarray(electron_state.population.parent_lo[born])),
        np.sort(np.asarray(ion_state.population.id_lo[ionized])),
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(electron_state.particles.position[born, 0])),
        np.sort(np.asarray(ion_state.particles.position[ionized, 0])),
    )


class _SubgridClaim(PIC.AbstractPICProcess):
    process_id: str = eqx.field(static=True)
    stage: str = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: str | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, identifier: str) -> None:
        self.process_id = identifier
        self.stage = "momentum"
        self.stochastic = False
        self.radiation_ownership = "subgrid-reaction"
        self.species_indices = (0,)

    def apply(self, species: Any, context: Any, /) -> Any:
        del species
        return PIC.PICProcessResult(
            context.species,
            PIC.PICProcessLedger(
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(True),
                self.process_id,
            ),
        )


def test_overlapping_radiation_ownership_claims_are_refused() -> None:
    solver = _reduced_solver()
    species = _neutral_pair()
    with pytest.raises(ValueError, match="overlaps"):
        phx.solver.ElectromagneticPICPlan(
            solver, species=species, processes=(_SubgridClaim("reaction"),)
        )
    with pytest.raises(ValueError, match="exactly one"):
        phx.solver.ElectromagneticPICPlan(
            solver,
            species=species,
            processes=(_SubgridClaim("first"), _SubgridClaim("second")),
            ownership="subgrid-reaction",
        )
    plan = phx.solver.ElectromagneticPICPlan(
        solver,
        species=species,
        processes=(_SubgridClaim("reaction"),),
        ownership="subgrid-reaction",
    )
    assert plan.ownership == "subgrid-reaction"


def test_unpaired_deposit_and_gauss_charge_are_refused_at_preparation() -> None:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(8, periodic=False),
            D.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    pec = phx.solver.maxwell.MaxwellBoundaryPlan("pec")
    solver = phx.solver.ReducedMaxwellPICFieldSolver(
        phx.solver.CompatibleMaxwell2DPlan(grid, boundaries=((pec, pec), (None, None))),
        PIC.ReducedPICTransferPlan(grid),
    )
    with pytest.raises(ValueError, match="pairing defect"):
        phx.solver.ElectromagneticPICPlan(
            solver, species=(_species(4, -1.0, "electrons", 2, 0),)
        )


def test_absorbed_charge_stays_on_the_grid_as_wall_charge() -> None:
    electrons, ions = _neutral_pair()
    boundary = PIC.PICOpenBoundaryPlan(
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        kinds=(PIC.PICBoundaryKind.ABSORB, PIC.PICBoundaryKind.ABSORB),
    )
    solver = _reduced_solver(periodic=False)
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=(electrons, ions),
        boundaries=boundary,
    )
    assert pic.pairing_defect < 1e-12
    dt = 0.9 * solver.field.stable_dt
    position = jnp.asarray([[0.1], [0.3], [0.55], [0.99]])
    velocity = jnp.zeros((4, 3)).at[3, 0].set(0.5)
    state = pic.initialize((position, position), (velocity, jnp.zeros((4, 3))), dt)
    result = pic.step_detailed(state, dt)
    assert result.successful
    electron_state = result.accepted_state.species[0]
    assert not electron_state.population.active[3]
    np.testing.assert_allclose(
        result.accepted_state.boundaries[0].collected_charge, [0.0, -1.0]
    )
    wall = result.accepted_state.wall_charge
    np.testing.assert_allclose(jnp.sum(wall) * solver.transfer.cell_volume, -1.0)
    assert result.diagnostics.particle_field_charge_defect < 1e-10
    assert result.diagnostics.electric_constraint < 1e-10
    # The next step starts without the absorbed particle; the field keeps its charge.
    follow = pic.step_detailed(result.accepted_state, dt)
    assert follow.successful
    assert follow.diagnostics.particle_field_charge_defect < 1e-10


def test_moving_window_translates_field_and_retires_trailing_particles() -> None:
    electrons, ions = _neutral_pair()
    solver = _reduced_solver()
    pic = phx.solver.ElectromagneticPICPlan(solver, species=(electrons, ions))
    window = phx.solver.PICMovingWindowPlan(pic, 0, shift_cells=2)
    state = window.initialize(_initial_state(pic, 0.2 * solver.field.stable_dt))
    result = window.shift(state)
    assert result.successful
    shifted = result.accepted_state
    np.testing.assert_allclose(shifted.origin, 2.0 / 16.0)
    for species in shifted.pic.species:
        # Particles at 0.1 and 0.11 leave through the trailing face.
        np.testing.assert_array_equal(
            species.population.active, [False, True, True, True]
        )
    np.testing.assert_allclose(result.outflow_charge, [-1.0, 1.0])
    electric = state.pic.field.electric[0]
    np.testing.assert_array_equal(shifted.pic.field.electric[0][:-2], electric[2:])
    np.testing.assert_array_equal(shifted.pic.field.electric[0][-2:], 0.0)
    rejected = window.shift(state, apply_shift=False)
    np.testing.assert_array_equal(rejected.accepted_state.origin, state.origin)


def _tetrahedral_solver() -> Any:
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        + ((0.25, 0.25, 0.25),)
    )
    cells = jnp.asarray(((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)))
    mesh = D.CellMesh(coordinates, (D.CellBlock("tet", "tetrahedron", cells),))
    element = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1))
    ).prepare()
    locator = D.PreparedSimplicialCellLocator(
        D.fem.prepare_finite_element_cell_map(element, 0),
        element.default_runtime.coordinates,
        D.SimplicialLocationPolicy(4, 8, 4),
    )
    hodge = phx.solver.maxwell.tetrahedral_maxwell_hodge(mesh.coordinates, locator.cells)
    maxwell = phx.solver.maxwell.UnstructuredMaxwellPlan(
        hodge.cochain, phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(), 100.0
    ).prepare()
    return phx.solver.UnstructuredMaxwellPICFieldSolver(
        maxwell, PIC.UnstructuredWhitneyCurrentPlan(locator, maximum_segments=4)
    )


def test_tetrahedral_maxwell_charge_follows_the_whitney_deposit() -> None:
    solver = _tetrahedral_solver()
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=(
            _species(2, -1.0, "electrons", 3, 0),
            _species(2, 1.0, "ions", 3, 10),
        ),
    )
    assert pic.pairing_defect < 1e-9
    # Generic interior points: off every shared face and edge of the split tetrahedron.
    position = jnp.asarray([[0.17, 0.23, 0.11], [0.31, 0.12, 0.2]])
    dt = 0.5 * float(solver.maxwell.stable_dt)
    state = pic.initialize(
        (position + jnp.asarray([0.01, 0.0, 0.0]), position),
        (jnp.zeros((2, 3)).at[:, 0].set(0.01), jnp.zeros((2, 3))),
        dt,
    )
    gauss, _ = solver.maxwell.constraints(state.field)
    np.testing.assert_allclose(gauss, 0.0, atol=1e-10)
    result = pic.step_detailed(state, dt)
    assert result.successful
    assert result.diagnostics.continuity_defect < 1e-9
    assert result.diagnostics.particle_field_charge_defect < 1e-10
    assert result.diagnostics.electric_constraint < 1e-10

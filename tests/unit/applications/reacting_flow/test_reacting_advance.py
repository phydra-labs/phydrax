#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.reacting_flow._advance import (
    ReactiveAdvanceState,
    ReactiveIMEXPlan,
    ReactiveStrangPlan,
)
from phydrax.applications.reacting_flow._finite_volume import (
    ReactiveStructuredFiniteVolumePlan,
)
from phydrax.applications.reacting_flow._mechanism import ChemicalMechanismCompiler
from phydrax.applications.reacting_flow._state import ReactiveConservedLayout
from phydrax.applications.reacting_flow._thermodynamics import ReactingGasModel
from phydrax.equations._chemical_mechanism import (
    ChemicalMechanismIR,
    ChemicalReactionSpec,
)
from phydrax.equations._chemical_rates import ArrheniusRatePlan
from phydrax.equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)


def _problem(rate=1.0):
    schema = ChemicalSpeciesSchema(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
    )
    thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -5.0e4)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    gas = ReactingGasModel(schema, thermodynamics)
    mechanism = ChemicalMechanismCompiler().compile(
        ChemicalMechanismIR(
            "A-to-B",
            schema,
            thermodynamics,
            (
                ChemicalReactionSpec(
                    "A->B",
                    {"A": 1.0},
                    {"B": 1.0},
                    ArrheniusRatePlan(rate),
                ),
            ),
        ),
        gas_model=gas,
    )
    layout = ReactiveConservedLayout(gas, 1)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(2, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0e6,))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=layout.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    dynamics = ReactiveStructuredFiniteVolumePlan(
        layout, method, boundaries
    ).prepare_runtime(discretization, positivity_iterations=2)
    one_cell = layout.from_thermodynamic_state(
        jnp.asarray(1.0),
        jnp.asarray((0.0,)),
        jnp.asarray(800.0),
        jnp.asarray((0.7, 0.3)),
    )
    conserved = jnp.broadcast_to(one_cell, discretization.state_shape)
    return layout, mechanism, dynamics, conserved


def _mass_fraction_a(layout, state):
    return layout.split(state.conserved).mass_fractions[0, 0]


def test_strang_fixed_schedule_is_second_order_for_uniform_reaction():
    layout, mechanism, dynamics, conserved = _problem()
    plan = ReactiveStrangPlan(dynamics, mechanism)
    coarse = plan.advance(plan.initial_state(conserved), jnp.asarray(0.2))
    half = plan.advance(plan.initial_state(conserved), jnp.asarray(0.1))
    fine = plan.advance(half.state, jnp.asarray(0.1))
    exact = 0.7 * np.exp(-0.2)
    coarse_error = abs(float(_mass_fraction_a(layout, coarse.state)) - exact)
    fine_error = abs(float(_mass_fraction_a(layout, fine.state)) - exact)

    assert coarse.evidence.accepted
    assert half.evidence.accepted
    assert fine.evidence.accepted
    assert coarse_error > 3.0 * fine_error
    assert int(coarse.state.schedule_index) == 1
    assert int(fine.state.schedule_index) == 2


def test_strang_restart_is_deterministic_and_failed_macro_step_fully_rolls_back():
    _, mechanism, dynamics, conserved = _problem()
    plan = ReactiveStrangPlan(dynamics, mechanism, schedule_substeps=2)
    first = plan.advance(plan.initial_state(conserved), jnp.asarray(0.02))
    restarted = ReactiveAdvanceState(
        first.state.time,
        first.state.conserved,
        first.state.transport_runtime_state,
        accepted_macro_steps=first.state.accepted_macro_steps,
        schedule_index=first.state.schedule_index,
        state_id=first.state.state_id,
    )
    continuous = plan.advance(first.state, jnp.asarray(0.02))
    resumed = plan.advance(restarted, jnp.asarray(0.02))

    assert continuous.evidence.accepted
    assert resumed.evidence.accepted
    np.testing.assert_array_equal(continuous.state.time, resumed.state.time)
    np.testing.assert_array_equal(continuous.state.conserved, resumed.state.conserved)
    np.testing.assert_array_equal(
        continuous.state.schedule_index, resumed.state.schedule_index
    )
    np.testing.assert_array_equal(
        continuous.state.transport_runtime_state.accepted_step,
        resumed.state.transport_runtime_state.accepted_step,
    )
    np.testing.assert_array_equal(
        continuous.state.transport_runtime_state.content_state.conservative_content,
        resumed.state.transport_runtime_state.content_state.conservative_content,
    )

    _, fast_mechanism, fast_dynamics, fast_conserved = _problem(rate=1.0e6)
    rejecting = ReactiveStrangPlan(fast_dynamics, fast_mechanism)
    initial = rejecting.initial_state(fast_conserved)
    failed = rejecting.advance(initial, jnp.asarray(0.01))
    assert failed.evidence.rolled_back
    assert not failed.evidence.accepted
    np.testing.assert_array_equal(failed.state.time, initial.time)
    np.testing.assert_array_equal(failed.state.conserved, initial.conserved)
    np.testing.assert_array_equal(
        failed.state.accepted_macro_steps, initial.accepted_macro_steps
    )
    np.testing.assert_array_equal(failed.state.schedule_index, initial.schedule_index)
    np.testing.assert_array_equal(
        failed.state.transport_runtime_state.accepted_step,
        initial.transport_runtime_state.accepted_step,
    )
    np.testing.assert_array_equal(
        failed.state.transport_runtime_state.content_state.conservative_content,
        initial.transport_runtime_state.content_state.conservative_content,
    )


def test_coupled_imex_agrees_with_strang_in_small_step_limit():
    layout, mechanism, dynamics, conserved = _problem()
    strang = ReactiveStrangPlan(dynamics, mechanism)
    imex = ReactiveIMEXPlan(
        dynamics,
        mechanism,
        nonlinear_iterations=16,
        nonlinear_tolerance=1.0e-11,
    )
    step = jnp.asarray(1.0e-4)
    split_result = strang.advance(strang.initial_state(conserved), step)
    imex_result = imex.advance(imex.initial_state(conserved), step)

    assert split_result.evidence.accepted
    assert imex_result.evidence.accepted
    np.testing.assert_allclose(
        _mass_fraction_a(layout, imex_result.state),
        _mass_fraction_a(layout, split_result.state),
        rtol=1.0e-8,
        atol=1.0e-11,
    )

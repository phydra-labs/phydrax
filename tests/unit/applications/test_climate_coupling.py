#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples.coupled_slab_ocean import slab_ocean_scenario
from phydrax.applications.geophysics._coupling import (
    BoussinesqOceanCouplingSubsystem,
    coupling_surface_field,
    read_geophysical_coupling_checkpoint,
    write_geophysical_coupling_checkpoint,
)
from phydrax.applications.ocean._step import OceanBoussinesqContinuationState
from phydrax.solver._partitioned_coupling_adaptive import (
    CouplingEpochTransitionPlan,
    CouplingTopologyRequest,
    IdentityCouplingEpochTransfer,
    PreparedCouplingEpoch,
    transition_coupling_epoch,
)
from phydrax.solver._partitioned_coupling_graph import CouplingGraph, prepare_coupling
from phydrax.solver._partitioned_coupling_runtime import advance_coupling_window
from phydrax.solver._partitioned_coupling_types import (
    CallableCouplingSubsystem,
    CouplingExchange,
    CouplingPort,
    CouplingQuantity,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
    CouplingSweep,
    CouplingTransferRequirement,
    CouplingWindow,
    ExplicitCouplingPolicy,
)


_AREA = phx.units.derived_unit("m²", ((phx.units.METER, 2),))
_HEAT = phx.units.derived_unit("J/m²", ((phx.units.JOULE, 1), (phx.units.METER, -2)))
_KILOHEAT = phx.units.derived_unit(
    "kJ/m²", ((phx.units.KILOJOULE, 1), (phx.units.METER, -2))
)


def _capabilities():
    return CouplingSubsystemCapabilities(
        jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
    )


def _typed_exchange(
    *,
    fail=False,
    jacobi=False,
    bad_measure=False,
    target_kind="enthalpy_per_area",
    target_unit=_HEAT,
    source_kind="enthalpy_per_area",
):
    source, source_measure = coupling_surface_field(jnp.asarray([1.0, 3.0]), "source")
    target, target_measure = coupling_surface_field(
        jnp.asarray([1.0, 1.0, 2.0]), "target"
    )
    output = CouplingPort(
        "out",
        "output",
        source.vector_space,
        field_space=source,
        measure=source_measure,
        measure_unit=_AREA,
        quantity=CouplingQuantity(source_kind, _HEAT),
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )
    input_ = CouplingPort(
        "in",
        "input",
        target.vector_space,
        field_space=target,
        measure=target_measure,
        measure_unit=_AREA,
        quantity=CouplingQuantity(target_kind, target_unit),
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )
    matrix = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0] if bad_measure else [0.0, 1.0]]
    )
    transfer = phx.discretization.FieldTransfer(
        source,
        target,
        phx.linalg.DenseLinearOperator(
            matrix, source=source.vector_space, target=target.vector_space
        ),
        properties=phx.discretization.TransferProperties(
            conservative=True, constant_preserving=True, positivity_preserving=True
        ),
    )

    def produce(window, state, inputs, args):
        del inputs, args
        integral = jnp.asarray([2.0, 3.0]) * window.size
        return CouplingSubsystemResult(
            state - integral, (integral,), successful=True, status=0
        )

    def consume(window, state, inputs, args):
        del window, args
        return CouplingSubsystemResult(
            state + inputs[0], (), successful=not fail, status=int(fail)
        )

    a = CallableCouplingSubsystem(
        produce, subsystem_id="a", output_ports=(output,), capabilities=_capabilities()
    )
    b = CallableCouplingSubsystem(
        consume, subsystem_id="b", input_ports=(input_,), capabilities=_capabilities()
    )
    exchange = CouplingExchange(
        "energy",
        "out",
        "in",
        transfer=transfer,
        requirement=CouplingTransferRequirement(
            conservative=True,
            positivity_preserving=True,
            constant_preserving=True,
            frame_action="preserve",
        ),
    )
    sweep = (
        CouplingSweep("jacobi")
        if jacobi
        else CouplingSweep("gauss-seidel", subsystem_order=("a", "b"))
    )
    return prepare_coupling(
        CouplingGraph((a, b), (exchange,)),
        (jnp.asarray([20.0, 30.0]), jnp.zeros(3)),
        (jnp.zeros(3),),
        policy=ExplicitCouplingPolicy(sweep),
    )


@pytest.mark.parametrize("requested", [True, False])
def test_epoch_transition_cannot_relabel_foreign_physical_state_or_budgets(requested):
    original = _typed_exchange()
    accepted = advance_coupling_window(
        original, original.reference_state, 1.0
    ).accepted_state
    incompatible = _typed_exchange(
        source_kind="temperature-inventory", target_kind="temperature-inventory"
    )
    epoch = PreparedCouplingEpoch(incompatible, ("a-epoch", "b-epoch"), ())
    identity = IdentityCouplingEpochTransfer()
    transition = CouplingEpochTransitionPlan(
        (identity, identity),
        (identity,),
        (),
        (),
        source_subsystem_ids=accepted.subsystem_ids,
        target_subsystem_ids=accepted.subsystem_ids,
        source_exchange_ids=accepted.exchange_ids,
        target_exchange_ids=accepted.exchange_ids,
        transition_id="same-local-ids",
    )
    request = CouplingTopologyRequest(requested, (0, 0), (0, 0), 0)
    with pytest.raises(ValueError):
        transition_coupling_epoch(
            epoch, accepted, epoch, transition, request, accepted_window=True
        )
    np.testing.assert_array_equal(accepted.cumulative_exchange_budget, [[-11.0, 11.0]])


def test_nonmatching_integrals_convert_units_and_close_measured_budget():
    prepared = _typed_exchange(target_unit=_KILOHEAT)
    result = eqx.filter_jit(advance_coupling_window)(
        prepared, prepared.reference_state, 2.0
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.accepted_state.participant_states[0], [16.0, 24.0])
    np.testing.assert_allclose(
        result.accepted_state.participant_states[1], [0.004, 0.006, 0.006]
    )
    np.testing.assert_allclose(result.accepted_exchange_budget, [[-22.0, 22.0]])


def test_physical_type_and_measure_claims_are_checked_not_inferred_from_storage():
    with pytest.raises(ValueError):
        _typed_exchange(target_kind="temperature-inventory")
    with pytest.raises(ValueError):
        _typed_exchange(bad_measure=True)


@pytest.mark.parametrize("failure", ["participant", "stale-proposal"])
def test_rejected_windows_preserve_every_state_and_accepted_budget(failure):
    prepared = _typed_exchange(
        fail=failure == "participant", jacobi=failure == "stale-proposal"
    )
    result = advance_coupling_window(prepared, prepared.reference_state, 1.0)
    assert not bool(result.successful)
    assert bool(eqx.tree_equal(result.accepted_state, prepared.reference_state))
    np.testing.assert_array_equal(result.accepted_exchange_budget, 0.0)
    assert float(result.candidate_state.time) == 1.0


def test_replay_and_window_refinement_do_not_double_spend_integrals():
    prepared = _typed_exchange()
    step = eqx.filter_jit(advance_coupling_window)
    full = step(prepared, prepared.reference_state, 2.0)
    replay = step(prepared, prepared.reference_state, 2.0)
    assert bool(eqx.tree_equal(full.accepted_state, replay.accepted_state))
    first = step(prepared, prepared.reference_state, 1.0)
    second = step(prepared, first.accepted_state, 1.0)
    np.testing.assert_allclose(
        second.accepted_state.cumulative_exchange_budget,
        full.accepted_state.cumulative_exchange_budget,
    )
    for actual, expected in zip(
        second.accepted_state.participant_states,
        full.accepted_state.participant_states,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)


def test_slab_ocean_real_freshwater_enthalpy_and_native_restart(tmp_path):
    prepared, slab, receiver = slab_ocean_scenario()
    initial = prepared.reference_state
    step = eqx.filter_jit(advance_coupling_window)
    first = step(prepared, initial, 0.5)
    assert bool(first.successful)
    state = first.accepted_state
    slab_index = state.subsystem_ids.index(slab.subsystem_id)
    ocean_index = state.subsystem_ids.index(receiver.subsystem_id)
    old_slab, new_slab = (
        initial.participant_states[slab_index],
        state.participant_states[slab_index],
    )
    ocean = state.participant_states[ocean_index]
    coefficient = receiver.method.ocean.plan.reference_density * receiver.heat_capacity
    heat = coefficient * ocean.ledger.tracer_change["conservative_temperature"]
    water = receiver.freshwater_density * ocean.ledger.freshwater_volume
    np.testing.assert_allclose(
        slab.measure.integrate(new_slab.enthalpy - old_slab.enthalpy) + heat,
        0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        slab.measure.integrate(new_slab.water_mass - old_slab.water_mass) + water,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ocean.state.eta, 0.5e-4 / receiver.freshwater_density, atol=1e-14
    )
    np.testing.assert_allclose(
        state.cumulative_exchange_budget.sum(axis=1), 0, atol=1e-12
    )
    path = write_geophysical_coupling_checkpoint(
        tmp_path / "coupled.npz", prepared, state
    )
    restarted = read_geophysical_coupling_checkpoint(path, prepared)
    continued = step(prepared, restarted, 0.5)
    uninterrupted = step(prepared, state, 0.5)
    assert bool(eqx.tree_equal(continued.accepted_state, uninterrupted.accepted_state))
    other, _, _ = slab_ocean_scenario(conductance=21.0)
    with pytest.raises(ValueError):
        read_geophysical_coupling_checkpoint(path, other)


def test_ocean_evaporation_removes_water_and_enthalpy_without_removing_salt():
    prepared, _, receiver = slab_ocean_scenario()
    initial = prepared.reference_state
    state = initial.participant_states[initial.subsystem_ids.index(receiver.subsystem_id)]
    heat = jnp.full((4,), -250.0)
    water = jnp.full((4,), -1.0e-4)
    result = receiver.advance_window(
        CouplingWindow(0, 0.0, 1.0), state, (heat, water), None
    )
    assert bool(result.successful)
    candidate = result.candidate_state
    coefficient = receiver.method.ocean.plan.reference_density * receiver.heat_capacity
    np.testing.assert_allclose(
        candidate.state.eta, -1.0e-4 / receiver.freshwater_density, atol=1e-14
    )
    np.testing.assert_allclose(
        candidate.state.tracer_inventory["absolute_salinity"],
        state.state.tracer_inventory["absolute_salinity"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        coefficient * candidate.ledger.tracer_change["conservative_temperature"],
        receiver.measure.integrate(heat),
        atol=1e-5,
    )


def test_slab_depletion_rejects_native_ocean_ledger_atomically():
    prepared, _, _ = slab_ocean_scenario(water_rate=1.0, water_mass=0.1)
    result = advance_coupling_window(prepared, prepared.reference_state, 1.0)
    assert not bool(result.successful)
    assert bool(eqx.tree_equal(result.accepted_state, prepared.reference_state))
    np.testing.assert_array_equal(result.accepted_exchange_budget, 0)


def test_slab_ocean_time_refinement_converges_to_two_reservoir_solution():
    prepared, slab, receiver = slab_ocean_scenario(water_rate=0, conductance=1000.0)
    step = eqx.filter_jit(advance_coupling_window)
    index = prepared.reference_state.subsystem_ids.index(slab.subsystem_id)
    slab_capacity = slab.dry_heat_capacity + slab.water_heat_capacity * 10.0
    ocean_capacity = (
        receiver.method.ocean.plan.reference_density * receiver.heat_capacity * 5.0
    )
    total_time = 4.0
    difference = 5 * np.exp(
        -slab.conductance * (1 / slab_capacity + 1 / ocean_capacity) * total_time
    )
    mean = (slab_capacity * 288.15 + ocean_capacity * 283.15) / (
        slab_capacity + ocean_capacity
    )
    exact = mean + ocean_capacity * difference / (slab_capacity + ocean_capacity)
    errors = []
    for count in (4, 8):
        state = prepared.reference_state
        for _ in range(count):
            result = step(prepared, state, total_time / count)
            assert bool(result.successful)
            state = result.accepted_state
        errors.append(
            abs(float(slab.temperature(state.participant_states[index])[0]) - exact)
        )
        np.testing.assert_allclose(
            state.cumulative_exchange_budget.sum(axis=1), 0, atol=1e-10
        )
    assert 1.8 < errors[0] / errors[1] < 2.2


def test_rigid_lid_adapter_uses_native_heat_and_stress_quadrature_without_water_port():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(
                3 if axis == 2 else 2, periodic=axis < 2
            )
            for axis in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        temperature_diffusivity=0.0,
        salinity_diffusivity=0.0,
    ).prepare(discretization)
    receiver = BoussinesqOceanCouplingSubsystem(ocean, stress=True)
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    coordinates = ocean.initial_state(
        velocity, jnp.full(grid.shape, 10.0), jnp.full(grid.shape, 35.0)
    )
    state = OceanBoussinesqContinuationState.initialize(coordinates)
    window = phx.solver.coupling.CouplingWindow(0, 0.0, 0.01)
    inputs = (jnp.full(4, 1.0), jnp.full(4, 1e-5), jnp.full(4, -2e-5))
    result = eqx.filter_jit(receiver.advance_window)(window, state, inputs, None)
    assert bool(result.successful)
    coefficient = (
        ocean.plan.reference.reference_density * ocean.plan.reference.heat_capacity
    )
    np.testing.assert_allclose(
        result.candidate_state.temperature_boundary_content * coefficient, 1.0, rtol=1e-10
    )
    _, scalars = ocean.dynamics.unpack_state(result.candidate_state.coordinates)
    np.testing.assert_allclose(
        scalars[ocean.plan.reference.salinity_name], 35.0, atol=1e-12
    )
    assert float(result.candidate_state.surface_stress_work) > 0
    assert all(
        port.quantity.quantity_kind != "water_mass_per_area"
        for port in receiver.input_ports
    )

"""Independent fixed-grid finite differences of the actual coupled rod."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from examples.nuclear_fuel_calibration import prepare_synthetic_inference_rod
from phydrax.applications.nuclear_fuel import (
    fuel_parameterization,
    FuelFixedGrid,
    FuelModelBundle,
    FuelParameterization,
    FuelParameterName,
    FuelReplayResult,
    FuelRodState,
    FuelSensitivityPlan,
    FuelSensitivityResult,
    HeliumTransportTable,
    PreparedFuelRod,
    replay_fuel_grid,
)
from phydrax.materials import PropertyValidityRange
from phydrax.observation import CoordinateLayout
from tests._support.nuclear_fuel import synthetic_model_bundle


_WORKING_SET_BYTES = 64 * 1024 * 1024


def _observable_history(state: FuelRodState) -> Array:
    # Physical fields plus irreversible history and gas stock: omitting local
    # candidate differentiation cannot pass this independent replay reference.
    return jnp.concatenate(
        (
            state.fields.temperature_fuel,
            jnp.asarray([state.fields.pressure]),
            state.fuel_sites.maximum_temperature,
            state.gas.retained_moles,
            state.fields.free_moles,
        )
    )


@dataclass(frozen=True)
class _CheckpointTangent:
    prepared: PreparedFuelRod
    startup: FuelReplayResult
    checkpoint: FuelRodState
    grid: FuelFixedGrid
    parameterization: FuelParameterization
    values: Array
    baseline: FuelReplayResult
    derivative: FuelSensitivityResult


@pytest.fixture(scope="module")
def checkpoint_tangent() -> _CheckpointTangent:
    """Phonon-resistivity tangent of a 5-20 s history from a frozen 5 s checkpoint."""
    with jax.enable_x64():
        # Deliberately synthetic transport data makes pressure-driven helium
        # absorption resolvable on this small fixed grid; it is not UO2 evidence.
        source = "synthetic-implicit-tangent-verification:helium-coupling"
        transport = HeliumTransportTable(
            temperature=(293.0, 2500.0),
            diffusivity=(1.0e-12, 1.0e-12),
            henry_constant=(2.0e18, 4.0e18),
            validity=PropertyValidityRange(
                {"temperature": (293.0, 2500.0, "K")}, source_id=source
            ),
            source_id=source,
        )
        # The plenum fill (about 9.5e-5 mol He at 0.103 MPa) loses about
        # 4.1e-6 mol to this Henry absorption per 5 s step, so each Picard
        # pass between the gas and thermomechanical blocks contracts the error
        # by about 4.1e-6 / 9.5e-5 = 0.044 (measured: 0.0438). From the
        # first-pass change of 1.1e-4 (scaled coordinates), reaching the 1e-9
        # coupling tolerance takes ceil(ln(1e-9 / 1.1e-4) / ln(0.044)) = 4 more
        # passes: 6 evaluations, above the default budget of 4.
        prepared = prepare_synthetic_inference_rod(
            helium_transport=transport, maximum_coupling_evaluations=6
        )
        startup = replay_fuel_grid(
            prepared, FuelFixedGrid((0.0, 5.0), prepared.prepared_id)
        )
        checkpoint = startup.states[-1]
        grid = FuelFixedGrid((5.0, 10.0, 20.0), prepared.prepared_id)
        parameterization, values = fuel_parameterization(
            prepared.admission.bundle, ("fuel-phonon-resistivity",)
        )
        baseline = replay_fuel_grid(
            prepared,
            grid,
            initial_state=checkpoint,
            parameters=parameterization.bundle(prepared.admission.bundle, values),
        )

        # This is a conditional checkpoint derivative: previous *complete* local
        # histories are fixed, not fabricated again from burnup or temperature.
        def fixed_checkpoint(parameters: Array) -> FuelRodState:
            del parameters
            return checkpoint

        initial_tangent = eqx.filter_jvp(
            fixed_checkpoint, (values,), (jnp.ones_like(values),)
        )[1]
        derivative = FuelSensitivityPlan(
            prepared,
            grid,
            parameterization,
            values,
            parameter_capacity=1,
            maximum_working_set_bytes=_WORKING_SET_BYTES,
        ).evaluate(baseline.states, initial_tangents=(initial_tangent,))
        return _CheckpointTangent(
            prepared=prepared,
            startup=startup,
            checkpoint=checkpoint,
            grid=grid,
            parameterization=parameterization,
            values=values,
            baseline=baseline,
            derivative=derivative,
        )


def test_checkpoint_replay_is_accepted_and_event_free_on_its_frozen_grid(
    checkpoint_tangent: _CheckpointTangent,
) -> None:
    case = checkpoint_tangent
    assert bool(case.startup.successful)
    assert bool(case.baseline.successful)
    np.testing.assert_array_equal(
        [float(state.time) for state in case.baseline.states], case.grid.times
    )
    # A Richardson reference is only meaningful on an event-free frozen grid.
    assert not np.any(np.asarray(case.baseline.branch_changed))


def test_checkpoint_forward_tangent_is_linearly_solved_and_branchwise_valid(
    checkpoint_tangent: _CheckpointTangent,
) -> None:
    derivative = checkpoint_tangent.derivative
    assert bool(derivative.successful)
    assert np.all(np.asarray(derivative.linear_success))
    assert np.all(np.asarray(derivative.branchwise_valid))
    assert len(derivative.tangents) == len(checkpoint_tangent.grid.times)


def test_full_history_forward_tangent_matches_independent_richardson_replay(
    checkpoint_tangent: _CheckpointTangent,
) -> None:
    case = checkpoint_tangent
    assert bool(case.baseline.successful) and bool(case.derivative.successful)
    actual = np.asarray(_observable_history(case.derivative.tangents[-1][0]))
    h = float(case.values[0]) * 0.002

    def solved_observables(offset: float) -> NDArray[np.float64]:
        parameters = case.parameterization.bundle(
            case.prepared.admission.bundle, case.values + offset
        )
        replay = replay_fuel_grid(
            case.prepared, case.grid, parameters=parameters, initial_state=case.checkpoint
        )
        assert bool(replay.successful)
        return np.asarray(_observable_history(replay.states[-1]))

    with jax.enable_x64():
        coarse = (solved_observables(h) - solved_observables(-h)) / (2.0 * h)
        fine = (solved_observables(h / 2.0) - solved_observables(-h / 2.0)) / h
    reference = (4.0 * fine - coarse) / 3.0
    # Explicit per-coordinate SI scales avoid treating moles and kelvin as
    # interchangeable solely because the flattened coordinates have a shape.
    temperature_count = case.prepared.geometry.fuel.node_count
    history_count = case.prepared.geometry.fuel.site_count
    atol = np.asarray(
        [2.0e-3] * temperature_count  # fuel nodal temperature [K per m*K/W]
        + [0.2]  # rod pressure [Pa per m*K/W]
        + [2.0e-3] * history_count  # site maximum temperature [K per m*K/W]
        + [2.0e-12] * 10  # retained and free gas moles [mol per m*K/W]
    )
    np.testing.assert_array_less(
        np.abs(actual - reference), atol + 0.01 * np.abs(reference)
    )


def test_contact_closure_is_reported_as_branchwise_derivative_event() -> None:
    with jax.enable_x64():
        # At 1000 W/m the steady pellet runs 313 K at the centre and 303 K at
        # the surface (area mean 307 K), so UO2 thermal expansion moves the
        # pellet surface by 9.8e-6 /K * 4.095e-3 m * 7.1 K = 284 nm, plus
        # 0.0275 * gap of permanent relocation, while the cladding bore moves
        # out by about 23 nm. A 0.3 um gap therefore never closes (31.7 nm
        # left at 60 s). With a 0.15 um gap the net closure of about 109 nm at
        # 5 s leaves the gap open, and about 231 nm at 10 s closes it.
        prepared = prepare_synthetic_inference_rod(fabricated_gap=0.15e-6)
        grid = FuelFixedGrid((0.0, 5.0, 10.0, 20.0, 30.0, 60.0), prepared.prepared_id)
        parameterization, values = fuel_parameterization(
            prepared.admission.bundle, ("fuel-phonon-resistivity",)
        )
        replay = replay_fuel_grid(prepared, grid)
        assert bool(replay.successful)
        active = np.asarray(
            [
                np.any(np.asarray(state.fields.contact_pressure) > 0.0)
                for state in replay.states
            ]
        )
        closure = (~active[:-1]) & active[1:]
        assert np.any(closure), (
            "The physical small-gap trajectory must actually close contact."
        )
        tangent = FuelSensitivityPlan(
            prepared,
            grid,
            parameterization,
            values,
            parameter_capacity=1,
            maximum_working_set_bytes=_WORKING_SET_BYTES,
        ).evaluate(replay.states)
        assert np.all(np.asarray(tangent.branch_changed)[closure])
        assert not np.any(np.asarray(tangent.branchwise_valid)[closure])


@pytest.mark.parametrize(
    ("parameter_capacity", "maximum_working_set_bytes", "message"),
    [
        pytest.param(1, _WORKING_SET_BYTES, "parameter capacity", id="coordinates"),
        pytest.param(2, 1, "working set", id="working-set-bytes"),
    ],
)
def test_tangent_capacity_refuses_before_tangent_allocation(
    parameter_capacity: int, maximum_working_set_bytes: int, message: str
) -> None:
    with jax.enable_x64():
        prepared = prepare_synthetic_inference_rod()
        grid = FuelFixedGrid((0.0, 5.0), prepared.prepared_id)
        parameterization, values = fuel_parameterization(
            prepared.admission.bundle,
            ("fuel-phonon-resistivity", "fuel-electronic-temperature"),
        )
        with pytest.raises(ValueError, match=message):
            FuelSensitivityPlan(
                prepared,
                grid,
                parameterization,
                values,
                parameter_capacity=parameter_capacity,
                maximum_working_set_bytes=maximum_working_set_bytes,
            )


@pytest.fixture(scope="module")
def bundle() -> FuelModelBundle:
    with jax.enable_x64():
        return synthetic_model_bundle(mode_count=4)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("fuel-phonon-resistivity", -0.001, id="negative-resistivity"),
        pytest.param("fuel-phonon-resistivity", 0.0, id="zero-resistivity"),
        pytest.param("fuel-phonon-resistivity", np.nan, id="nonfinite-resistivity"),
        pytest.param(
            "fuel-porosity-normalization", 0.0, id="zero-porosity-normalization"
        ),
        pytest.param(
            "gas-intrinsic-diffusivity-prefactor", -1.0, id="negative-diffusivity"
        ),
        pytest.param(
            "gas-intrinsic-activation-energy", -1.0, id="negative-activation-energy"
        ),
        pytest.param(
            "fuel-electronic-coefficient", np.inf, id="nonfinite-signed-coefficient"
        ),
    ],
)
def test_canonical_coordinates_preserve_source_parameter_domains(
    bundle: FuelModelBundle, name: FuelParameterName, value: float
) -> None:
    with jax.enable_x64():
        parameterization, _ = fuel_parameterization(bundle, (name,))
        with pytest.raises(eqx.EquinoxRuntimeError):
            parameterization.bundle(bundle, jnp.asarray([value], dtype=jnp.float64))


def test_zero_intrinsic_diffusivity_remains_a_valid_physical_boundary(
    bundle: FuelModelBundle,
) -> None:
    with jax.enable_x64():
        parameterization, _ = fuel_parameterization(
            bundle, ("gas-intrinsic-diffusivity-prefactor",)
        )
        updated = parameterization.bundle(bundle, jnp.zeros((1,), dtype=jnp.float64))
        assert float(updated.fission_gas.diffusivity.diffusivity(300.0, 0.0)) == 0.0


@pytest.mark.parametrize(
    "custom", [False, True], ids=["canonical-signed-coefficient", "custom-signed-map"]
)
def test_signed_coordinates_preserve_physical_temperature_response(
    bundle: FuelModelBundle, custom: bool
) -> None:
    with jax.enable_x64():
        if custom:

            def update(base: FuelModelBundle, values: Array) -> FuelModelBundle:
                return eqx.tree_at(
                    lambda model: model.fuel_conductivity.phonon_temperature_coefficient,
                    base,
                    values[0],
                )

            parameterization = FuelParameterization(
                CoordinateLayout(("signed-source-slope",)),
                ("m/W",),
                update,
                parameterization_id="signed-source-coefficient-regression",
            )
        else:
            parameterization, _ = fuel_parameterization(
                bundle, ("fuel-phonon-temperature-coefficient",)
            )
        updated = parameterization.bundle(
            bundle, jnp.asarray([-1.0e-8], dtype=jnp.float64)
        )
        cold = updated.fuel_conductivity.conductivity(300.0, 0.0, 0.05, 0.0)
        hot = updated.fuel_conductivity.conductivity(600.0, 0.0, 0.05, 0.0)
        # Clipping the signed coordinate to zero loses this resolvable response;
        # the source electronic term alone is negligible at these temperatures.
        assert float(hot - cold) > 1.0e-4


@pytest.mark.strict_jax
def test_float32_coordinates_use_float64_source_execution_without_value_change(
    bundle: FuelModelBundle,
) -> None:
    with jax.enable_x64():
        parameterization, nominal = fuel_parameterization(
            bundle, ("fuel-phonon-resistivity",)
        )
        inputs = jnp.asarray(nominal, dtype=jnp.float32)
        updated = parameterization.bundle(bundle, inputs)

        def conductivity(values: Array) -> Array:
            model = parameterization.bundle(bundle, values)
            return model.fuel_conductivity.conductivity(300.0, 0.0, 0.05, 0.0)

        result = jax.jit(conductivity)(inputs)
        assert updated.fuel_conductivity.phonon_constant.dtype == jnp.dtype(jnp.float64)
        assert result.dtype == jnp.dtype(jnp.float64)
        assert float(updated.fuel_conductivity.phonon_constant) == float(inputs[0])

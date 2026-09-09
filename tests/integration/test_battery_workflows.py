#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.applications.battery as battery
from phydrax.applications.battery._ageing_empirical import (
    EmpiricalAgeingSupport,
    EmpiricalAgeingTopology,
)
from phydrax.applications.battery._calibration import BatteryCalibrationFailurePolicy
from phydrax.applications.battery._current_control import (
    BatteryCurrentControlBounds,
    BatteryCurrentControlObjective,
    BatteryCurrentControlTerminalTarget,
    BatteryPiecewiseCurrent,
)
from phydrax.applications.battery._dfn_entry import evaluate_dfn_entry_gate
from phydrax.applications.battery._ecm import (
    ThermalEquivalentCircuitAdapter,
    ThermalEquivalentCircuitInitialCondition,
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitPlan,
)
from phydrax.applications.battery._estimation import ExactAffineECMEstimationPlan
from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
    BatteryRuntimeInputs,
)
from phydrax.applications.battery._oed import BatteryOEDModelContract
from phydrax.applications.battery._pack_entry import evaluate_series_pack_entry_gate
from phydrax.applications.battery._properties import (
    ConcentrationTemperaturePropertyLaw,
    ConstantPropertyLaw,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._qualification import (
    BATTERY_OED_CANDIDATE,
    MARQUIS_2019_SPME_CANDIDATE,
    THERMAL_ECM_CANDIDATE,
    THERMAL_ECM_SUPPORT,
)
from phydrax.applications.battery._spm import (
    PrescribedCurrentSpmAdapter,
    PrescribedCurrentSpmPlan,
    SpmInitialCondition,
    SpmParameters,
)
from phydrax.applications.battery._spme_marquis2019 import (
    Marquis2019SpmeAdapter,
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmeParameters,
    Marquis2019SpmePlan,
)
from phydrax.applications.battery._spme_side_reactions import (
    BrosaPlanellaSpmeSeiAdapter,
    BrosaPlanellaSpmeSeiPlan,
)
from phydrax.applications.battery._tspme_brosa_planella import (
    BrosaPlanellaTspmeAdapter,
    BrosaPlanellaTspmePlan,
)


def _law(value, support, *, quantity, unit, coordinate="temperature"):
    return ConstantPropertyLaw(
        jnp.asarray(value),
        jnp.asarray(support),
        value_bounds=(0.0, jnp.inf),
        quantity=quantity,
        coordinate=coordinate,
        value_unit=unit,
        coordinate_unit=(
            "1" if coordinate in ("state_of_charge", "stoichiometry") else "K"
        ),
        source_id=f"integration:{quantity}",
    )


def _ecm_parameters():
    return ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1,)),
        jnp.asarray((20.0,)),
        1000.0,
        100.0,
        0.5,
        300.0,
        300.0,
        _law(
            3.7,
            (0.0, 1.0),
            quantity="reference-open-circuit-voltage",
            unit="V",
            coordinate="state_of_charge",
        ),
        _law(
            0.0,
            (0.0, 1.0),
            quantity="entropic-coefficient",
            unit="V/K",
            coordinate="state_of_charge",
        ),
    )


def _spm_parameters():
    return SpmParameters(
        electrode_area_m2=0.1,
        negative_electrode_thickness_m=1.0e-4,
        positive_electrode_thickness_m=1.0e-4,
        negative_active_material_volume_fraction=0.6,
        positive_active_material_volume_fraction=0.6,
        negative_particle_radius_m=5.0e-6,
        positive_particle_radius_m=4.0e-6,
        negative_maximum_concentration_mol_m3=3.0e4,
        positive_maximum_concentration_mol_m3=3.0e4,
        temperature_k=298.15,
        maximum_absolute_current_a=20.0,
        negative_stoichiometry_at_empty=0.1,
        negative_stoichiometry_at_full=0.9,
        positive_stoichiometry_at_empty=0.9,
        positive_stoichiometry_at_full=0.1,
        negative_solid_diffusivity=_law(
            2.0e-14,
            (250.0, 350.0),
            quantity="negative-solid-diffusivity",
            unit="m2/s",
        ),
        positive_solid_diffusivity=_law(
            1.0e-14,
            (250.0, 350.0),
            quantity="positive-solid-diffusivity",
            unit="m2/s",
        ),
        negative_exchange_current_density=_law(
            5.0,
            (250.0, 350.0),
            quantity="negative-exchange-current",
            unit="A/m2",
        ),
        positive_exchange_current_density=_law(
            4.0,
            (250.0, 350.0),
            quantity="positive-exchange-current",
            unit="A/m2",
        ),
        negative_open_circuit_potential=_law(
            0.1,
            (0.0, 1.0),
            quantity="negative-ocp",
            unit="V",
            coordinate="stoichiometry",
        ),
        positive_open_circuit_potential=_law(
            4.1,
            (0.0, 1.0),
            quantity="positive-ocp",
            unit="V",
            coordinate="stoichiometry",
        ),
    )


def _electrolyte_law(value, *, quantity, unit):
    return ConcentrationTemperaturePropertyLaw(
        jnp.asarray((500.0, 1500.0)),
        jnp.asarray((273.15, 323.15)),
        jnp.full((2, 2), value),
        value_bounds=(0.0, 1.0 if unit == "1" else jnp.inf),
        quantity=quantity,
        value_unit=unit,
        source_id=f"integration:{quantity}",
    )


def _spme_parameters():
    return Marquis2019SpmeParameters(
        _spm_parameters(),
        separator_thickness_m=5.0e-5,
        negative_electrolyte_porosity=0.3,
        separator_electrolyte_porosity=0.5,
        positive_electrolyte_porosity=0.3,
        bruggeman_coefficient=1.5,
        typical_electrolyte_concentration_mol_m3=1000.0,
        electrolyte_diffusivity=_electrolyte_law(
            2.0e-10,
            quantity="electrolyte-diffusivity",
            unit="m2/s",
        ),
        electrolyte_conductivity=_electrolyte_law(
            1.1,
            quantity="electrolyte-conductivity",
            unit="S/m",
        ),
        transference_number=_electrolyte_law(
            0.4,
            quantity="transference-number",
            unit="1",
        ),
        negative_solid_conductivity_s_m=100.0,
        positive_solid_conductivity_s_m=80.0,
    )


def _runtime(parameters, current):
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0),))
    values = BatteryProtocolValues(protocol, jnp.asarray((current,)))
    return BatteryRuntimeInputs(
        parameters,
        protocol.input_policy(values, node_side="right"),
        values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(values),
    )


def test_ecm_run_identity_and_native_transition_states_bind_concrete_execution():
    parameters = _ecm_parameters()
    model = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(1))
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0), RestStepPlan(1.0)))
    prepared = BatteryExperimentPlan(
        model,
        protocol,
        BatteryOutputPlan(("voltage_v", "current_a")),
        BatteryDiffraxSolvePlan(dt0=0.2),
        jnp.asarray((0.0, 2.0)),
        THERMAL_ECM_CANDIDATE,
        THERMAL_ECM_SUPPORT,
    ).prepare()
    initial = ThermalEquivalentCircuitInitialCondition(500.0, 300.0, jnp.zeros((1,)))
    first = prepared.run(
        parameters,
        initial,
        BatteryProtocolValues(protocol, jnp.asarray((1.0,))),
    )
    changed_current = prepared.run(
        parameters,
        initial,
        BatteryProtocolValues(protocol, jnp.asarray((2.0,))),
    )
    changed_initial = prepared.run(
        parameters,
        ThermalEquivalentCircuitInitialCondition(550.0, 300.0, jnp.zeros((1,))),
        BatteryProtocolValues(protocol, jnp.asarray((1.0,))),
    )
    changed_parameters_value = eqx.tree_at(
        lambda value: value.series_resistance_ohm,
        parameters,
        1.1 * parameters.series_resistance_ohm,
    )
    changed_parameters = prepared.run(
        changed_parameters_value,
        initial,
        BatteryProtocolValues(protocol, jnp.asarray((1.0,))),
    )
    assert (
        len(
            {
                first.run_id,
                changed_current.run_id,
                changed_initial.run_id,
                changed_parameters.run_id,
            }
        )
        == 4
    )
    assert first.parameters_digest != changed_parameters.parameters_digest
    assert first.require_evidence_identity() == first.run_id
    np.testing.assert_allclose(first.native_solution.times, np.asarray((0.0, 1.0, 2.0)))
    np.testing.assert_allclose(first.outputs.times_s, np.asarray((0.0, 2.0)))


def test_private_model_candidates_execute_their_native_physics_boundaries():
    ecm_parameters = _ecm_parameters()
    ecm = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(1))
    prepared_ecm = ecm.prepare()
    ecm_state = ecm.initial_state(
        prepared_ecm,
        ecm_parameters,
        ThermalEquivalentCircuitInitialCondition(500.0, 300.0, jnp.zeros((1,))),
    )
    ecm_runtime = _runtime(ecm_parameters, 2.0)
    ecm_rate = ecm.problem(prepared_ecm, ecm_state, ecm_runtime).drift(
        jnp.asarray(0.0), ecm_state, ecm_runtime
    )
    ecm_output = ecm.observe(prepared_ecm, jnp.asarray(0.0), ecm_state, ecm_runtime)
    np.testing.assert_allclose(ecm_rate.charge_c, 2.0)
    assert bool(ecm_output.domain_valid)
    assert ecm.model_id == dict(THERMAL_ECM_SUPPORT.attributes)["model_id"]

    spm_parameters = _spm_parameters()
    spm = PrescribedCurrentSpmAdapter(PrescribedCurrentSpmPlan(3))
    prepared_spm = spm.prepare()
    spm_state = spm.initial_state(
        prepared_spm, spm_parameters, SpmInitialCondition(0.5, 0.5)
    )
    spm_runtime = _runtime(spm_parameters, 4.0)
    spm_rate = spm.problem(prepared_spm, spm_state, spm_runtime).drift(
        jnp.asarray(0.0), spm_state, spm_runtime
    )
    np.testing.assert_allclose(
        jnp.sum(spm_rate.negative_amount_mol) + jnp.sum(spm_rate.positive_amount_mol),
        0.0,
        atol=1.0e-15,
    )

    spme_parameters = _spme_parameters()
    spme = Marquis2019SpmeAdapter(
        Marquis2019SpmePlan(
            3,
            negative_electrolyte_cell_count=2,
            separator_electrolyte_cell_count=1,
            positive_electrolyte_cell_count=2,
        )
    )
    prepared_spme = spme.prepare()
    spme_state = spme.initial_state(
        prepared_spme,
        spme_parameters,
        Marquis2019SpmeInitialCondition(0.5, 0.5),
    )
    evaluation = prepared_spme.evaluate(spme_state, spme_parameters, -2.0)
    np.testing.assert_allclose(
        jnp.sum(evaluation.electrolyte_transport.amount_rate_mol_s),
        0.0,
        atol=1.0e-15,
    )
    assert bool(evaluation.domain_valid)


def test_private_workflow_contracts_retain_real_content_boundaries():
    ageing_support = EmpiricalAgeingSupport(
        (273.15, 333.15),
        (0.0, 1.0),
        (0.0, 20.0),
        (0.0, 1.0e9),
        maximum_time_gap_s=60.0,
        maximum_macrostep_s=3600.0,
        source_id="integration:empirical-ageing",
    )
    ageing = EmpiricalAgeingTopology(ageing_support, 3)
    assert ageing.topology_id

    calibration_policy = BatteryCalibrationFailurePolicy()
    assert calibration_policy.policy_id

    estimation = ExactAffineECMEstimationPlan(
        ThermalEquivalentCircuitPlan(1),
        jnp.asarray((0.1, 0.9)),
        temperature_mode="isothermal",
    )
    assert estimation.plan_id

    assert BatteryOEDModelContract.__module__.endswith("._oed")
    assert (
        dict(BATTERY_OED_CANDIDATE.support_tuples[0].attributes)["code_id"]
        == "phydrax.applications.battery._oed.evaluate_battery_oed"
    )

    bounds = BatteryCurrentControlBounds(
        current_a=(-5.0, 5.0),
        voltage_v=(2.5, 4.5),
        stoichiometry=(0.0, 1.0),
        temperature_k=(273.15, 333.15),
        voltage_name="voltage_v",
        stoichiometry_names=("negative_stoichiometry",),
        temperature_name="temperature_k",
    )
    target = BatteryCurrentControlTerminalTarget(
        0.5, state_index=0, name="terminal-charge"
    )
    objective = BatteryCurrentControlObjective(
        current_squared_weight=1.0,
        terminal_target_squared_weight=2.0,
    )
    current = BatteryPiecewiseCurrent(
        jnp.asarray((1.0, -1.0)),
        knot_times_s=(0.0, 1.0, 2.0),
        lowering_id="integration:piecewise-current",
    )
    assert bounds.bounds_id and target.target_id and objective.objective_id
    np.testing.assert_allclose(current.coefficients, jnp.asarray(((1.0,), (-1.0,))))

    assert BrosaPlanellaTspmePlan.__module__.endswith("._tspme_brosa_planella")
    assert BrosaPlanellaTspmeAdapter.__module__.endswith("._tspme_brosa_planella")
    assert BrosaPlanellaSpmeSeiPlan.__module__.endswith("._spme_side_reactions")
    assert BrosaPlanellaSpmeSeiAdapter.__module__.endswith("._spme_side_reactions")


def test_unsigned_expansion_requests_cannot_authorize_implementation():
    dfn = evaluate_dfn_entry_gate(
        release_index=None,
        trust_policy=None,
        marquis_profile_id=MARQUIS_2019_SPME_CANDIDATE.profile_id,
        marquis_support=MARQUIS_2019_SPME_CANDIDATE.support_tuples[0],
        assessment=None,
        at_time=50,
    )
    pack = evaluate_series_pack_entry_gate(
        release_index=None,
        profile_id=THERMAL_ECM_CANDIDATE.profile_id,
        trust_policy=None,
        assessment=None,
        at_time=50,
    )
    assert (dfn.eligible, dfn.conclusive) == (False, False)
    assert (pack.eligible, pack.conclusive) == (False, False)
    assert dfn.reason and pack.reason


def test_public_release_builder_rejects_opaque_evidence_ids():
    with pytest.raises(TypeError):
        battery.build_thermal_ecm_release_profile(
            "opaque:qualification-evidence", trust_policy=None, at_time=10, expires_at=20
        )

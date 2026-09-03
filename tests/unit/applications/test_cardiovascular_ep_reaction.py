#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.electrophysiology._membrane_scaling import (
    CardiacMembraneScaling,
)
from phydrax.applications.cardiovascular.electrophysiology._reaction import (
    CardiacReactionModel,
    plan_reaction,
    prepare_reaction,
)
from phydrax.applications.cardiovascular.electrophysiology._reaction_ir import (
    compile_reaction_ir,
    interpret_reaction_ir,
    PinnedReactionIR,
    ReactionBinaryOperator,
    ReactionIRBinary,
    ReactionIRInput,
    ReactionIRLiteral,
    ReactionIROutput,
    ReactionIRSelect,
    ReactionIRUnary,
    ReactionUnaryOperator,
)
from phydrax.applications.cardiovascular.electrophysiology._ventricular_models import (
    ORdVentricularModel,
    TenTusscherPanfilov2006Model,
    VentricularCellPhenotype,
)


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def tp06() -> TenTusscherPanfilov2006Model:
    return TenTusscherPanfilov2006Model()


@pytest.fixture
def ord_model() -> ORdVentricularModel:
    return ORdVentricularModel()


def _tp06_plateau(model: TenTusscherPanfilov2006Model) -> jax.Array:
    return jnp.asarray(
        (
            20.0,
            0.4,
            0.5,
            0.2,
            0.99,
            0.01,
            0.01,
            0.8,
            0.5,
            0.7,
            0.8,
            0.2,
            0.8,
            0.0006,
            1.0,
            0.0008,
            0.7,
            9.0,
            135.0,
        ),
        dtype=jnp.float64,
    )


def _ord_plateau(model: ORdVentricularModel) -> jax.Array:
    del model
    return jnp.asarray(
        (
            20.0,
            0.05,
            8.0,
            8.0,
            140.0,
            140.0,
            0.0006,
            0.001,
            1.3,
            1.0,
            0.99,
            0.01,
            0.02,
            0.01,
            0.02,
            0.01,
            0.9,
            0.2,
            0.2,
            0.8,
            0.2,
            0.3,
            0.8,
            0.2,
            0.3,
            0.8,
            0.4,
            0.6,
            0.5,
            0.6,
            0.8,
            0.2,
            0.4,
            0.5,
            0.3,
            0.4,
            0.2,
            0.2,
            0.7,
            0.01,
            0.012,
        ),
        dtype=jnp.float64,
    )


def test_model_specific_named_layouts_have_no_padded_union(tp06, ord_model):
    assert isinstance(tp06, CardiacReactionModel)
    assert isinstance(ord_model, CardiacReactionModel)
    assert tp06.state_layout.state_count == 19
    assert ord_model.state_layout.state_count == 41
    assert tp06.state_layout.state_names != ord_model.state_layout.state_names
    assert tp06.state_layout.index("calcium_i_mM") == 13
    assert ord_model.state_layout.index("calcium_i_mM") == 6
    assert tp06.parameter_layout.index("g_Na_mS_per_uF") == 4
    assert ord_model.parameter_layout.index("g_Na_mS_per_uF") == 4
    packed = tp06.state_layout.pack(tp06.state_layout.unpack(tp06.initialize()))
    np.testing.assert_array_equal(packed, tp06.initialize())


@pytest.mark.parametrize(
    "model,cell_count,current_count",
    [
        (TenTusscherPanfilov2006Model(), 4, 12),
        (ORdVentricularModel(), 3, 16),
    ],
)
def test_models_are_jittable_over_fixed_homogeneous_blocks(
    model, cell_count, current_count
):
    state = model.initialize((cell_count,), dtype=jnp.float64)
    evaluation = jax.jit(model.evaluate)(state)
    assert evaluation.state_rate.shape == state.shape
    assert evaluation.gate_steady_state.shape == (
        cell_count,
        model.state_layout.gate_count,
    )
    assert evaluation.current_density_uA_per_mm2.shape == (cell_count, current_count)
    assert evaluation.calcium_cytosol_mM.shape == (cell_count,)
    assert evaluation.model_id == model.model_id
    assert np.all(np.asarray(evaluation.valid))
    np.testing.assert_allclose(
        evaluation.total_outward_current_uA_per_mm2,
        np.sum(evaluation.current_density_uA_per_mm2, axis=-1),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_tp06_rest_and_plateau_rate_reference_fixtures(tp06):
    rest = tp06.evaluate(tp06.initialize(dtype=jnp.float64))
    np.testing.assert_allclose(
        np.asarray(
            (
                rest.state_rate[0],
                rest.calcium_cytosol_rate_mM_per_ms,
                rest.total_outward_current_uA_per_mm2,
                rest.calcium_sr_flux_mM_per_ms,
            )
        ),
        np.asarray(
            (
                0.041318510118333346,
                -7.439573927407132e-08,
                -0.00041318510118333346,
                4.511807929238334e-06,
            )
        ),
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    plateau = tp06.evaluate(_tp06_plateau(tp06))
    np.testing.assert_allclose(
        np.asarray(
            (
                plateau.state_rate[0],
                plateau.calcium_cytosol_rate_mM_per_ms,
                plateau.total_outward_current_uA_per_mm2,
                plateau.calcium_sr_flux_mM_per_ms,
            )
        ),
        np.asarray(
            (
                2.105287318427654,
                -4.353370998227344e-06,
                -0.02105287318427654,
                -0.005072077533410167,
            )
        ),
        rtol=2.0e-11,
        atol=2.0e-13,
    )


def test_ord_rest_and_plateau_rate_reference_fixtures(ord_model):
    rest = ord_model.evaluate(ord_model.initialize(dtype=jnp.float64))
    np.testing.assert_allclose(
        np.asarray(
            (
                rest.state_rate[0],
                rest.calcium_cytosol_rate_mM_per_ms,
                rest.total_outward_current_uA_per_mm2,
                rest.calcium_sr_flux_mM_per_ms,
            )
        ),
        np.asarray(
            (
                -0.16394414539125418,
                -1.0253746778419257e-07,
                0.0016394414539125418,
                -0.00013405462184873943,
            )
        ),
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    plateau = ord_model.evaluate(_ord_plateau(ord_model))
    np.testing.assert_allclose(
        np.asarray(
            (
                plateau.state_rate[0],
                plateau.calcium_cytosol_rate_mM_per_ms,
                plateau.total_outward_current_uA_per_mm2,
                plateau.calcium_sr_flux_mM_per_ms,
            )
        ),
        np.asarray(
            (
                1.6320924338934706,
                -3.490256949255947e-06,
                -0.016320924338934707,
                0.00810378705038853,
            )
        ),
        rtol=2.0e-11,
        atol=2.0e-13,
    )


def test_ord_nca_relaxation_uses_jca_as_the_backward_rate(ord_model):
    state = _ord_plateau(ord_model)
    evaluation = ord_model.evaluate(state)
    jca = state[ord_model.state_layout.index("j_ca")]
    cass = state[ord_model.state_layout.index("calcium_ss_mM")]
    anca = jca / (1000.0 + jca * (1.0 + 0.002 / cass) ** 4)
    forward = anca * 1000.0
    expected_tau = 1.0 / (forward + jca)
    expected_steady = forward * expected_tau
    np.testing.assert_allclose(
        evaluation.gate_time_constant_ms[21],
        expected_tau,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        evaluation.gate_steady_state[21],
        expected_steady,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_ord_epicardial_delta_scales_both_transient_outward_inactivation_times():
    endocardial = ORdVentricularModel(VentricularCellPhenotype.ENDOCARDIAL)
    epicardial = ORdVentricularModel(VentricularCellPhenotype.EPICARDIAL)
    state = endocardial.initialize(dtype=jnp.float64).at[0].set(-60.0)
    endocardial_evaluation = endocardial.evaluate(state)
    epicardial_evaluation = epicardial.evaluate(state)
    delta_epi = 1.0 - 0.95 / (1.0 + np.exp((-60.0 + 70.0) / 5.0))
    np.testing.assert_allclose(
        epicardial_evaluation.gate_time_constant_ms[10:12],
        endocardial_evaluation.gate_time_constant_ms[10:12] * delta_epi,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    "phenotype,release_factor",
    [
        (VentricularCellPhenotype.ENDOCARDIAL, 1.0),
        (VentricularCellPhenotype.MIDMYOCARDIAL, 1.7),
    ],
)
def test_ord_release_gain_and_time_constant_include_bt_and_m_cell_factor(
    phenotype, release_factor
):
    model = ORdVentricularModel(phenotype)
    state = _ord_plateau(model)
    evaluation = model.evaluate(state)
    bt = 4.75 * release_factor
    cajsr = state[model.state_layout.index("calcium_jsr_mM")]
    denominator = 1.0 + (1.5 / cajsr) ** 8
    normalized_ical = evaluation.current("I_CaL") / model.membrane_capacitance_uF_per_mm2
    release_scale = model.default_parameters[
        model.parameter_layout.index("calcium_release_scale_per_ms")
    ]
    expected_steady = -0.5 * bt * release_scale * normalized_ical / denominator
    expected_tau = bt / (1.0 + 0.0123 / cajsr)
    np.testing.assert_allclose(
        evaluation.gate_steady_state[29],
        expected_steady,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        evaluation.gate_time_constant_ms[29],
        expected_tau,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        evaluation.gate_steady_state[30],
        1.25 * expected_steady,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        evaluation.gate_time_constant_ms[30],
        1.25 * expected_tau,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    "model,state_factory,sodium_gate",
    [
        (
            TenTusscherPanfilov2006Model(),
            _tp06_plateau,
            lambda state: state[4] ** 3 * state[5] * state[6],
        ),
        (
            ORdVentricularModel(),
            _ord_plateau,
            lambda state: (
                state[10] ** 3 * (0.99 * state[11] + 0.01 * state[12]) * state[13]
            ),
        ),
    ],
)
def test_fast_sodium_current_has_independent_direct_formula(
    model, state_factory, sodium_gate
):
    state = state_factory(model)
    evaluation = model.evaluate(state)
    parameters = np.asarray(model.default_parameters)
    temperature = parameters[model.parameter_layout.index("temperature_K")]
    sodium_o = parameters[model.parameter_layout.index("sodium_o_mM")]
    sodium_i = float(state[model.state_layout.index("sodium_i_mM")])
    voltage = float(state[0])
    reversal = 8314.46261815324 * temperature / 96485.33212 * np.log(sodium_o / sodium_i)
    conductance = parameters[model.parameter_layout.index("g_Na_mS_per_uF")]
    if isinstance(model, ORdVentricularModel):
        cass = float(state[model.state_layout.index("calcium_ss_mM")])
        camkt = float(state[1])
        camkb = 0.05 * (1.0 - camkt) / (1.0 + 0.0015 / cass)
        fraction = 1.0 / (1.0 + 0.15 / (camkb + camkt))
        h = 0.99 * float(state[11]) + 0.01 * float(state[12])
        hp = 0.99 * float(state[11]) + 0.01 * float(state[14])
        gate = float(state[10]) ** 3 * (
            (1.0 - fraction) * h * float(state[13]) + fraction * hp * float(state[15])
        )
    else:
        gate = float(sodium_gate(state))
    direct = (
        conductance * gate * (voltage - reversal) * model.membrane_capacitance_uF_per_mm2
    )
    np.testing.assert_allclose(
        evaluation.current("I_Na"), direct, rtol=2.0e-12, atol=2.0e-14
    )


@pytest.mark.parametrize(
    "model,state,voltage",
    [
        (
            TenTusscherPanfilov2006Model(),
            _tp06_plateau(TenTusscherPanfilov2006Model()),
            15.0,
        ),
        (ORdVentricularModel(), _ord_plateau(ORdVentricularModel()), 0.0),
    ],
)
def test_ghk_singular_voltage_is_value_and_gradient_safe(model, state, voltage):
    state = state.at[0].set(voltage)

    def calcium_current(v):
        candidate = state.at[0].set(v)
        return model.evaluate(candidate).current("I_CaL")

    value = calcium_current(jnp.asarray(voltage, dtype=jnp.float64))
    derivative = jax.grad(calcium_current)(jnp.asarray(voltage, dtype=jnp.float64))
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


@pytest.mark.parametrize("model", [TenTusscherPanfilov2006Model(), ORdVentricularModel()])
def test_exact_gate_update_matches_reported_affine_solution(model):
    state = model.initialize((2,), dtype=jnp.float64)
    state = state.at[1, 0].set(state[1, 0] + 5.0)
    evaluation = model.evaluate(state)
    dt = jnp.asarray(0.17)
    updated = model.exact_gate_update(state, dt)
    gate_indices = jnp.asarray(model.state_layout.gate_indices)
    expected = evaluation.gate_steady_state + (
        state[..., gate_indices] - evaluation.gate_steady_state
    ) * jnp.exp(-dt / evaluation.gate_time_constant_ms)
    np.testing.assert_allclose(
        updated[..., gate_indices], expected, rtol=2.0e-13, atol=2.0e-13
    )
    nongates = tuple(
        index
        for index in range(model.state_layout.state_count)
        if index not in model.state_layout.gate_indices
    )
    np.testing.assert_array_equal(
        updated[..., jnp.asarray(nongates)], state[..., jnp.asarray(nongates)]
    )


@pytest.mark.parametrize("model", [TenTusscherPanfilov2006Model(), ORdVentricularModel()])
def test_invalid_concentration_fails_closed_and_host_validation_refuses(model):
    state = model.initialize(dtype=jnp.float64)
    invalid = state.at[model.state_layout.index("calcium_i_mM")].set(0.0)
    evaluation = model.evaluate(invalid)
    assert not bool(evaluation.valid)
    assert np.all(np.isnan(np.asarray(evaluation.state_rate)))
    with pytest.raises(ValueError, match="concentrations|admissible"):
        model.validate_state(invalid)
    bad_parameters = model.default_parameters.at[0].set(-1.0)
    invalid_parameters = model.evaluate(state, bad_parameters)
    assert not bool(invalid_parameters.valid)
    with pytest.raises(ValueError, match="physical domain"):
        model.validate_state(state, bad_parameters)


def test_typed_phenotype_routes_refuse_runtime_strings():
    with pytest.raises(TypeError, match="VentricularCellPhenotype"):
        TenTusscherPanfilov2006Model(phenotype="epicardial")
    epicardial = ORdVentricularModel(VentricularCellPhenotype.EPICARDIAL)
    endocardial = ORdVentricularModel(VentricularCellPhenotype.ENDOCARDIAL)
    assert epicardial.model_id != endocardial.model_id
    assert not np.array_equal(
        epicardial.default_parameters, endocardial.default_parameters
    )


def test_charge_evidence_closes_with_outward_current_and_stimulus(tp06, ord_model):
    for model in (tp06, ord_model):
        stimulus = jnp.asarray(0.003, dtype=jnp.float64)
        evaluation = model.evaluate(
            model.initialize(dtype=jnp.float64),
            stimulus_current_uA_per_mm2=stimulus,
        )
        np.testing.assert_allclose(
            evaluation.charge_balance_residual_uA_per_mm2,
            0.0,
            atol=2.0e-15,
        )
        assert np.isfinite(float(evaluation.calcium_membrane_current_uA_per_mm2))
        assert np.isfinite(float(evaluation.calcium_sr_flux_mM_per_ms))


def test_membrane_scaling_has_exact_kernel_and_si_factors():
    scaling = CardiacMembraneScaling(140.0, 0.01)
    assert scaling.volumetric_capacitance_uF_per_mm3 == pytest.approx(1.4)
    assert scaling.membrane_surface_to_volume_per_m == 140_000.0
    assert scaling.membrane_capacitance_F_per_m2 == 0.01
    assert scaling.volumetric_capacitance_F_per_m3 == pytest.approx(1400.0)
    np.testing.assert_allclose(scaling.conductivity_mS_per_mm_to_S_per_m(0.2), 0.2)
    np.testing.assert_allclose(scaling.surface_current_uA_per_mm2_to_A_per_m2(0.4), 0.4)
    np.testing.assert_allclose(scaling.volume_current_uA_per_mm3_to_A_per_m3(0.4), 400.0)
    np.testing.assert_allclose(
        scaling.outward_surface_current_to_voltage_rate(0.02), -2.0
    )
    np.testing.assert_allclose(scaling.outward_volume_current_to_voltage_rate(1.4), -1.0)
    np.testing.assert_allclose(scaling.applied_volume_current_to_voltage_rate(1.4), 1.0)
    np.testing.assert_allclose(scaling.conductivity_to_diffusivity_mm2_per_ms(0.14), 0.1)
    with pytest.raises(ValueError, match="positive"):
        CardiacMembraneScaling(0.0, 0.01)


def test_prepared_reaction_exposes_pinned_split_block_contract(tp06):
    plan = plan_reaction(tp06, 5, dtype=np.float64)
    prepared = prepare_reaction(plan)
    voltage, local = prepared.initialize()
    assert prepared.model_id == tp06.model_id
    assert prepared.gate_count == 18
    assert prepared.true_gate_count == 12
    assert voltage.shape == (5,)
    assert local.shape == (5, 18)
    d_voltage, d_local = jax.jit(prepared.rates)(
        voltage,
        local,
        jnp.full((5,), 0.14),
    )
    assert d_voltage.shape == voltage.shape
    assert d_local.shape == local.shape
    outward = prepared.currents(voltage, local)
    surface = tp06.evaluate(
        tp06.initialize((5,), dtype=jnp.float64)
    ).total_outward_current_uA_per_mm2
    np.testing.assert_allclose(outward, 140.0 * surface)
    with pytest.raises(ValueError, match="pinned"):
        prepared.initialize(node_count=4)


def test_reaction_ir_compiler_matches_independent_tree_interpreter():
    x = ReactionIRInput(0)
    threshold = ReactionIRLiteral(0.0)
    positive = ReactionIRBinary(
        ReactionBinaryOperator.LESS,
        threshold,
        x,
    )
    smooth_positive = ReactionIRUnary(
        ReactionUnaryOperator.LOG1P,
        ReactionIRUnary(ReactionUnaryOperator.EXP, x),
    )
    expression = ReactionIRSelect(
        positive,
        smooth_positive,
        ReactionIRBinary(ReactionBinaryOperator.MULTIPLY, x, x),
    )
    source = PinnedReactionIR(
        "n-version-algebra-v1",
        ("x",),
        (ReactionIROutput("value", expression),),
    )
    compiled = compile_reaction_ir(source)
    values = (jnp.asarray((-2.0, -0.1, 0.5, 2.0)),)
    direct = interpret_reaction_ir(source, values)
    lowered = jax.jit(compiled)(values)
    np.testing.assert_allclose(lowered[0], direct[0], rtol=2.0e-13, atol=2.0e-13)
    assert compiled.program_id == source.program_id
    assert compiled.inspect()
    assert all("eval" not in row[1].lower() for row in compiled.inspect())
    with pytest.raises(ValueError, match="outside"):
        PinnedReactionIR(
            "invalid-input-v1",
            ("x",),
            (ReactionIROutput("bad", ReactionIRInput(1)),),
        )


@pytest.mark.parametrize("model", [TenTusscherPanfilov2006Model(), ORdVentricularModel()])
def test_model_pinned_ir_is_an_independent_ohmic_current_route(model):
    conductance = jnp.asarray((0.1, 0.3))
    gate = jnp.asarray((0.4, 0.8))
    voltage = jnp.asarray((-80.0, 20.0))
    reversal = jnp.asarray((-90.0, -85.0))
    (compiled,) = model.reaction_ir((conductance, gate, voltage, reversal))
    direct = conductance * gate * (voltage - reversal)
    np.testing.assert_allclose(compiled, direct, rtol=2.0e-13, atol=2.0e-13)

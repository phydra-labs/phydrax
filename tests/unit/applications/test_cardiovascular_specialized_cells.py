#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.electrophysiology._atrial_models import (
    AtrialAdmissibilityStatus,
    AtrialCurrents,
    AtrialPhenotype,
    AtrialState,
    CourtemancheAtrialParameters,
    CourtemancheAtrialReactionAdapter,
)
from phydrax.applications.cardiovascular.electrophysiology._nodal_models import (
    AtrioventricularCurrents,
    AtrioventricularPhenotype,
    AtrioventricularState,
    InadaAtrioventricularParameters,
    InadaAtrioventricularReactionAdapter,
    NodalAdmissibilityStatus,
    SinoatrialCurrents,
    SinoatrialPhenotype,
    SinoatrialState,
    ZhangSinoatrialParameters,
    ZhangSinoatrialReactionAdapter,
)
from phydrax.applications.cardiovascular.electrophysiology._purkinje_models import (
    PurkinjeAdmissibilityStatus,
    PurkinjeCurrents,
    PurkinjePhenotype,
    PurkinjeState,
    StewartPurkinjeParameters,
    StewartPurkinjeReactionAdapter,
)
from phydrax.applications.cardiovascular.electrophysiology._reaction import (
    CardiacReactionModel,
    plan_reaction,
    prepare_reaction,
)


def _models():
    return (
        CourtemancheAtrialParameters().prepare(),
        ZhangSinoatrialParameters().prepare(),
        InadaAtrioventricularParameters().prepare(),
        StewartPurkinjeParameters().prepare(),
    )


def test_specialized_families_have_typed_fixed_soa_layouts() -> None:
    atrial, san, av, purkinje = _models()

    states = tuple(model.initialize((2, 3)) for model in (atrial, san, av, purkinje))
    assert isinstance(states[0], AtrialState)
    assert isinstance(states[1], SinoatrialState)
    assert isinstance(states[2], AtrioventricularState)
    assert isinstance(states[3], PurkinjeState)
    assert len({type(state) for state in states}) == 4

    for model, state in zip((atrial, san, av, purkinje), states, strict=True):
        packed = model.layout.pack(state)
        assert packed.shape == (model.layout.state_size, 2, 3)
        unpacked = model.layout.unpack(packed)
        assert type(unpacked) is type(state)
        assert jnp.array_equal(model.layout.pack(unpacked), packed)
        assert len(model.layout.names) == len(set(model.layout.names))
        assert model.layout.index("voltage_mV") == 0

    assert atrial.layout.names != san.layout.names
    assert san.layout.names != av.layout.names
    assert av.layout.names != purkinje.layout.names
    assert len({model.layout.layout_id for model in (atrial, san, av, purkinje)}) == 4


def test_phenotypes_are_explicit_identities_and_ids_cover_coefficients() -> None:
    plans = (
        CourtemancheAtrialParameters(),
        ZhangSinoatrialParameters(),
        InadaAtrioventricularParameters(),
        StewartPurkinjeParameters(),
    )
    expected_phenotypes = (
        AtrialPhenotype.HUMAN_WORKING_MYOCYTE_CRN1998_REDUCED,
        SinoatrialPhenotype.RABBIT_PERIPHERAL_ZHANG2000_REDUCED,
        AtrioventricularPhenotype.RABBIT_COMPACT_INADA2009_REDUCED,
        PurkinjePhenotype.HUMAN_STEWART2009_REDUCED,
    )
    assert tuple(plan.phenotype for plan in plans) == expected_phenotypes
    assert all(isinstance(plan.phenotype, Enum) for plan in plans)
    assert tuple(type(plan.phenotype) for plan in plans) == tuple(
        type(value) for value in expected_phenotypes
    )
    assert tuple(plan.parameter_id for plan in plans) == tuple(
        type(plan)().parameter_id for plan in plans
    )

    changed = (
        CourtemancheAtrialParameters(g_na=7.9),
        ZhangSinoatrialParameters(g_f=0.013),
        InadaAtrioventricularParameters(g_f=0.005),
        StewartPurkinjeParameters(g_f=0.013),
    )
    assert all(
        original.parameter_id != modified.parameter_id
        for original, modified in zip(plans, changed, strict=True)
    )
    assert all(
        original.prepare().model_id != modified.prepare().model_id
        for original, modified in zip(plans, changed, strict=True)
    )

    with pytest.raises(TypeError, match="not bool"):
        CourtemancheAtrialParameters(g_na=True)
    with pytest.raises(TypeError, match="not bool"):
        ZhangSinoatrialParameters(g_f=True)
    with pytest.raises(TypeError, match="not bool"):
        InadaAtrioventricularParameters(g_f=True)
    with pytest.raises(TypeError, match="not bool"):
        StewartPurkinjeParameters(g_f=True)


@pytest.mark.parametrize(
    ("index", "expected_total", "expected_ca_current", "expected_ca_flux"),
    (
        (0, 0.0006063971100267948, -0.0003403879392709641, 1.1440280849473818e-06),
        (1, -0.1756208912047558, -0.18685466371777448, 0.00012783096659294444),
        (2, -0.001999786886188182, -0.025957759268736825, 5.191551853747366e-07),
        (3, -0.01741531307956351, 0.06608411961641592, 4.730656804435075e-05),
    ),
)
def test_specialized_reference_fixtures(
    index: int,
    expected_total: float,
    expected_ca_current: float,
    expected_ca_flux: float,
) -> None:
    model = _models()[index]
    evaluation = model.rates(model.initialize(dtype=jnp.float64))

    assert float(evaluation.currents.total_ionic) == pytest.approx(
        expected_total, rel=2.0e-7, abs=1.0e-10
    )
    assert float(evaluation.calcium.membrane_current_pA_per_pF) == pytest.approx(
        expected_ca_current, rel=2.0e-7, abs=1.0e-10
    )
    assert float(evaluation.calcium.net_cytosolic_flux_mM_per_ms) == pytest.approx(
        expected_ca_flux, rel=2.0e-7, abs=1.0e-11
    )
    assert bool(evaluation.evidence.successful)


def test_resting_and_pacemaking_current_balance_is_phenotype_specific() -> None:
    atrial, san, av, purkinje = _models()
    atrial_rate = float(atrial.rates(atrial.initialize()).state_rate.voltage_mV_per_ms)
    san_rate = float(san.rates(san.initialize()).state_rate.voltage_mV_per_ms)
    av_rate = float(av.rates(av.initialize()).state_rate.voltage_mV_per_ms)
    purkinje_rate = float(
        purkinje.rates(purkinje.initialize()).state_rate.voltage_mV_per_ms
    )

    assert abs(atrial_rate) < 0.01
    assert san_rate > 0.10
    assert 0.0 < av_rate < san_rate
    assert 0.0 < purkinje_rate < san_rate

    atrial_currents = atrial.rates(atrial.initialize()).currents
    san_currents = san.rates(san.initialize()).currents
    av_currents = av.rates(av.initialize()).currents
    purkinje_currents = purkinje.rates(purkinje.initialize()).currents
    assert isinstance(atrial_currents, AtrialCurrents)
    assert isinstance(san_currents, SinoatrialCurrents)
    assert isinstance(av_currents, AtrioventricularCurrents)
    assert isinstance(purkinje_currents, PurkinjeCurrents)
    assert san_currents.t_type_calcium.shape == ()
    assert av_currents.fast_sodium.shape == ()
    assert purkinje_currents.funny.shape == ()
    assert atrial_currents.ultrarapid_potassium.shape == ()


def test_vectorized_rate_systems_are_finite_under_jit() -> None:
    for model in _models():
        state = model.initialize((4,), dtype=jnp.float32)
        evaluation = jax.jit(model.rates)(state)
        leaves = jax.tree_util.tree_leaves(evaluation)
        assert leaves
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
        assert evaluation.currents.total_ionic.shape == (4,)
        assert evaluation.calcium.net_cytosolic_flux_mM_per_ms.shape == (4,)
        assert jnp.all(evaluation.evidence.successful)


def test_admissibility_is_fail_closed_for_each_typed_layout() -> None:
    statuses = (
        AtrialAdmissibilityStatus,
        NodalAdmissibilityStatus,
        NodalAdmissibilityStatus,
        PurkinjeAdmissibilityStatus,
    )
    for model, status_type in zip(_models(), statuses, strict=True):
        packed = model.layout.pack(model.initialize((2,)))
        invalid = packed.at[1, 0].set(1.2).at[-1, 1].set(-1.0)
        evidence = model.admissibility(model.layout.unpack(invalid))

        assert not bool(evidence.successful[0])
        assert int(evidence.status[0]) & int(status_type.GATE_OUT_OF_RANGE)
        assert not bool(evidence.successful[1])
        assert int(evidence.status[1]) & int(status_type.NONPOSITIVE_CALCIUM)


@pytest.mark.parametrize(
    ("adapter", "state_count", "gate_count", "current_count"),
    (
        (CourtemancheAtrialReactionAdapter(), 15, 12, 12),
        (ZhangSinoatrialReactionAdapter(), 10, 7, 7),
        (InadaAtrioventricularReactionAdapter(), 10, 8, 7),
        (StewartPurkinjeReactionAdapter(), 13, 10, 11),
    ),
)
def test_specialized_family_prepares_and_evaluates_as_reaction_model(
    adapter: CardiacReactionModel,
    state_count: int,
    gate_count: int,
    current_count: int,
) -> None:
    assert isinstance(adapter, CardiacReactionModel)
    prepared = prepare_reaction(plan_reaction(adapter, node_count=4, dtype=np.float64))
    voltage, local_state = prepared.initialize()
    state = jnp.concatenate((voltage[..., None], local_state), axis=-1)
    evaluation = adapter.evaluate(state)

    assert state.shape == (4, state_count)
    assert local_state.shape == (4, state_count - 1)
    assert prepared.true_gate_count == gate_count
    assert evaluation.gate_steady_state.shape == (4, gate_count)
    assert evaluation.gate_time_constant_ms.shape == (4, gate_count)
    assert evaluation.current_density_uA_per_mm2.shape == (4, current_count)
    assert len(evaluation.current_names) == current_count
    assert jnp.all(evaluation.valid)
    assert jnp.all(jnp.isfinite(evaluation.state_rate))
    assert jnp.max(jnp.abs(evaluation.charge_balance_residual_uA_per_mm2)) < 1.0e-12

    outward_stimulus = jnp.asarray(0.001, dtype=state.dtype)
    stimulated = adapter.evaluate(state, stimulus_current_uA_per_mm2=outward_stimulus)
    expected_delta = -outward_stimulus / adapter.membrane_capacitance_uF_per_mm2
    assert jnp.allclose(
        stimulated.state_rate[..., 0] - evaluation.state_rate[..., 0],
        expected_delta,
    )

    gate_updated = adapter.exact_gate_update(state, 0.2)
    concentration_indices = jnp.asarray(adapter.state_layout.concentration_indices)
    assert jnp.allclose(
        gate_updated[..., concentration_indices],
        state[..., concentration_indices],
    )
    assert jnp.array_equal(gate_updated[..., 0], state[..., 0])
    assert jnp.all(jnp.isnan(adapter.exact_gate_update(state, -0.1)))

    mismatched_parameters = adapter.default_parameters.at[0].add(1.0)
    failed = adapter.evaluate(state, mismatched_parameters)
    assert not jnp.any(failed.valid)
    assert jnp.all(jnp.isnan(failed.state_rate))

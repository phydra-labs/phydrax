#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.systems_biology import (
    bind_biological_evidence,
    BiologicalCondition,
    BiologicalFact,
    BiologicalReference,
    CompartmentSpec,
    CountMeasurementPlan,
    ExchangeFieldSpec,
    HillPropensity,
    MassActionPropensity,
    MichaelisMentenPropensity,
    MultirateScheduleEntry,
    PlanFieldAssertion,
    PromoterTransitionPropensity,
    SpeciesSpec,
    StoichiometricNetworkPlan,
    StoichiometricProcessSpec,
    StoichiometricRuntime,
    TelegraphFitTarget,
    TelegraphGeneExpressionPlan,
    WholeCellAssemblyPlan,
    WholeCellProcessBinding,
)
from phydrax.equations import (
    ArrheniusRatePlan,
    ChemicalMechanismIR,
    ChemicalPhaseKind,
    ChemicalReactionSpec,
    ChemicalSpeciesSchema,
    PolynomialSpeciesThermodynamicsPlan,
)
from phydrax.solver import solve_direct_ssa
from phydrax.stochastic import PoissonClockRealization


def _closed_conversion():
    compartment = CompartmentSpec("cell", 1.0)
    return StoichiometricNetworkPlan(
        "closed-conversion",
        (compartment,),
        (
            SpeciesSpec("a", "cell"),
            SpeciesSpec("b", "cell"),
        ),
        (
            StoichiometricProcessSpec(
                "forward", {"a": -1, "b": 1}, MassActionPropensity(0.1, {"a": 1})
            ),
            StoichiometricProcessSpec(
                "reverse", {"a": 1, "b": -1}, MassActionPropensity(0.05, {"b": 1})
            ),
        ),
        stoichiometry_capacity=2,
    ).prepare()


def _whole_cell_assembly():
    compartment = CompartmentSpec("cell", 1.0)
    uptake = StoichiometricNetworkPlan(
        "uptake",
        (compartment,),
        (
            SpeciesSpec("external_nutrient", "cell", reservoir=True),
            SpeciesSpec("metabolite", "cell"),
        ),
        (
            StoichiometricProcessSpec(
                "uptake",
                {"external_nutrient": -1, "metabolite": 1},
                MassActionPropensity(1.0, {"external_nutrient": 1}),
            ),
        ),
    ).prepare()
    growth = StoichiometricNetworkPlan(
        "growth",
        (compartment,),
        (SpeciesSpec("metabolite", "cell"), SpeciesSpec("biomass", "cell")),
        (
            StoichiometricProcessSpec(
                "growth",
                {"metabolite": -1, "biomass": 1},
                MassActionPropensity(1.0, {"metabolite": 1}),
            ),
        ),
    ).prepare()
    fields = (
        ExchangeFieldSpec("external_nutrient", reservoir=True),
        ExchangeFieldSpec("metabolite"),
        ExchangeFieldSpec("biomass"),
    )
    bindings = (
        WholeCellProcessBinding(
            "uptake",
            uptake,
            {
                "external_nutrient": "external_nutrient",
                "metabolite": "metabolite",
            },
        ),
        WholeCellProcessBinding(
            "growth",
            growth,
            {"metabolite": "metabolite", "biomass": "biomass"},
        ),
    )
    return WholeCellAssemblyPlan(
        "minimal-cell",
        fields,
        bindings,
        (
            MultirateScheduleEntry("uptake", 2, require_regime_valid=False),
            MultirateScheduleEntry("growth", 4, require_regime_valid=False),
        ),
        field_capacity=5,
        process_capacity=3,
    ).prepare()


def test_sparse_stoichiometry_conservation_and_nonnegative_propensities():
    network = _closed_conversion()
    np.testing.assert_array_equal(network.stoichiometry, np.asarray([[-1, 1], [1, -1]]))
    np.testing.assert_allclose(
        np.asarray(network.conservation.basis) @ np.asarray(network.stoichiometry).T,
        0.0,
        atol=1.0e-12,
    )
    evaluation = network.evaluate(jnp.asarray([20.0, 5.0]))
    assert bool(evaluation.successful)
    assert bool(jnp.all(evaluation.propensities >= 0.0))
    np.testing.assert_allclose(evaluation.conservation_residual, 0.0, atol=1.0e-12)
    boundary = network.evaluate(jnp.asarray([0.0, 5.0]))
    assert float(boundary.propensities[0]) == 0.0


def test_thermochemical_binding_requires_exact_reactants_orders_and_content():
    schema = ChemicalSpeciesSchema(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        np.asarray([1.0, 1.0]),
        ("E",),
        np.asarray([[1, 1]], dtype=int),
        np.asarray([0, 0], dtype=int),
    )
    thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 0.0]),
    )

    def mechanism(pre_exponential):
        reaction = ChemicalReactionSpec(
            "association-view",
            {"A": 2},
            {"A": 1, "B": 1},
            ArrheniusRatePlan(pre_exponential),
        )
        return ChemicalMechanismIR(
            "association", schema, thermodynamics, (reaction,)
        ).prepare()

    compartment = CompartmentSpec("cell", 1.0)
    species = (
        SpeciesSpec("a", "cell", thermochemical_name="A"),
        SpeciesSpec("b", "cell", thermochemical_name="B"),
    )
    mismatched = StoichiometricNetworkPlan(
        "mismatched-chemical-view",
        (compartment,),
        species,
        (
            StoichiometricProcessSpec(
                "association-view",
                {"a": -1, "b": 1},
                MassActionPropensity(1.0, {"a": 1}),
                thermochemical_reaction="association-view",
            ),
        ),
    ).prepare()
    with pytest.raises(ValueError, match="do not exactly match"):
        mismatched.bind_thermochemical(mechanism(1.0))
    matched = StoichiometricNetworkPlan(
        "exact-chemical-view",
        (compartment,),
        species,
        (
            StoichiometricProcessSpec(
                "association-view",
                {"a": -1, "b": 1},
                MassActionPropensity(1.0, {"a": 2}),
                thermochemical_reaction="association-view",
            ),
        ),
    ).prepare()
    first = matched.bind_thermochemical(mechanism(1.0))
    second = matched.bind_thermochemical(mechanism(2.0))
    assert bool(first.compatible)
    assert first.mechanism_content_id != second.mechanism_content_id
    assert first.binding_id != second.binding_id


def test_plan_capacities_and_stoichiometric_units_are_exact():
    network = _closed_conversion()
    with pytest.raises(ValueError, match="stoichiometry_capacity"):
        StoichiometricNetworkPlan(
            network.plan.name,
            network.plan.compartments,
            network.plan.species,
            network.plan.processes,
            stoichiometry_capacity=2.0,
        )
    with pytest.raises(ValueError, match="substeps"):
        MultirateScheduleEntry("process", 2.0)
    compartment = CompartmentSpec("cell", 1.0)
    with pytest.raises(ValueError, match="incompatible"):
        StoichiometricNetworkPlan(
            "mixed-units",
            (compartment,),
            (
                SpeciesSpec("molecules", "cell", unit="molecule"),
                SpeciesSpec("amount", "cell", unit="mol"),
            ),
            (
                StoichiometricProcessSpec(
                    "invalid-conversion",
                    {"molecules": -1, "amount": 1},
                    MassActionPropensity(1.0, {"molecules": 1}),
                ),
            ),
        )


def test_invalid_runtime_parameters_fail_closed():
    network = _closed_conversion()
    parameters = network.propensity_parameters.at[0, 0].set(jnp.nan)
    evaluation = network.evaluate(
        jnp.asarray([20.0, 5.0]), StoichiometricRuntime(parameters)
    )
    assert not bool(evaluation.successful)
    assert not bool(evaluation.parameter_valid)
    assert bool(jnp.all(jnp.isnan(evaluation.propensities)))


def test_opposing_reservoir_flows_remain_separate_in_ledgers():
    compartment = CompartmentSpec("cell", 1.0)
    network = StoichiometricNetworkPlan(
        "reservoir-ledgers",
        (compartment,),
        (
            SpeciesSpec("reservoir", "cell", reservoir=True),
            SpeciesSpec("internal", "cell"),
        ),
        (
            StoichiometricProcessSpec(
                "inflow",
                {"reservoir": -1, "internal": 1},
                MassActionPropensity(1.0, {"reservoir": 1}),
            ),
            StoichiometricProcessSpec(
                "outflow",
                {"reservoir": 1, "internal": -1},
                MassActionPropensity(1.0, {"internal": 1}),
            ),
        ),
    ).prepare()
    evaluation = network.evaluate(jnp.asarray([10.0, 8.0]), mode="deterministic")
    np.testing.assert_allclose(evaluation.source_rate, [10.0, 0.0])
    np.testing.assert_allclose(evaluation.sink_rate, [8.0, 0.0])
    np.testing.assert_allclose(evaluation.conservation_residual, 0.0, atol=1.0e-12)


def test_supported_nonlinear_propensities_are_finite_and_nonnegative():
    compartment = CompartmentSpec("cell", 2.0)
    species = tuple(
        SpeciesSpec(name, "cell")
        for name in ("off", "on", "regulator", "substrate", "product")
    )
    network = StoichiometricNetworkPlan(
        "propensity-laws",
        (compartment,),
        species,
        (
            StoichiometricProcessSpec(
                "switch",
                {"off": -1, "on": 1},
                PromoterTransitionPropensity(0.3, "off"),
            ),
            StoichiometricProcessSpec(
                "hill", {"product": 1}, HillPropensity(4.0, 2.0, 2.0, "regulator")
            ),
            StoichiometricProcessSpec(
                "enzyme",
                {"substrate": -1, "product": 1},
                MichaelisMentenPropensity(5.0, 3.0, "substrate"),
            ),
        ),
    ).prepare()
    result = eqx.filter_jit(network.evaluate)(jnp.asarray([1.0, 0.0, 4.0, 8.0, 0.0]))
    assert bool(result.successful)
    assert bool(jnp.all(jnp.isfinite(result.propensities)))
    assert bool(jnp.all(result.propensities >= 0.0))


def test_exact_ssa_reuses_native_realization_reproducibly():
    network = _closed_conversion()
    process = network.exact_jump_process()
    realization = PoissonClockRealization(
        jax.random.key(19),
        process.num_channels,
        support=(0.0, 2.0),
        max_events_per_channel=64,
        sample_shape=(4,),
        process_id=process.process_id,
    )
    arguments = dict(
        t0=0.0,
        t1=2.0,
        save_times=jnp.linspace(0.0, 2.0, 9),
        args=network.default_runtime(),
    )
    first = solve_direct_ssa(process, realization, jnp.asarray([40.0, 10.0]), **arguments)
    second = solve_direct_ssa(
        process, realization, jnp.asarray([40.0, 10.0]), **arguments
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.events.channels, second.events.channels)
    np.testing.assert_allclose(jnp.sum(first.states, axis=-1), 50.0)


def test_cle_ensemble_mean_agrees_with_deterministic_step_in_large_count_regime():
    network = _closed_conversion()
    state = jnp.asarray([10_000.0, 10_000.0])
    duration = jnp.asarray(0.05)
    deterministic = network.deterministic_step(state, duration)
    keys = jax.random.split(jax.random.key(3), 4096)
    candidates = jax.vmap(lambda key: network.cle_step(state, duration, key).candidate)(
        keys
    )
    np.testing.assert_allclose(
        jnp.mean(candidates, axis=0),
        deterministic.candidate,
        rtol=2.0e-4,
        atol=0.25,
    )
    assert bool(deterministic.evidence.regime_valid)
    cle = network.cle_step(state, duration, jax.random.key(4))
    assert bool(cle.evidence.regime_valid)
    endpoint_limited = network.deterministic_step(jnp.asarray([100.0, 21.0]), 9.0)
    assert bool(endpoint_limited.evidence.numerical_valid)
    assert not bool(endpoint_limited.evidence.regime_valid)
    zero_intensity = network.cle_step(jnp.asarray([0.0, 100.0]), 0.01, jax.random.key(5))
    assert not bool(zero_intensity.evidence.differentiable)


def test_telegraph_stationary_moments_and_gradient_sanity():
    model = TelegraphGeneExpressionPlan(2.0, 3.0, 12.0, 4.0, 1.5).prepare()
    moments = model.stationary_moments()
    np.testing.assert_allclose(moments.promoter_mean, 0.4)
    np.testing.assert_allclose(moments.nascent_mean, 1.2)
    np.testing.assert_allclose(moments.mature_mean, 3.2)
    assert float(moments.nascent_variance) > float(moments.nascent_mean)
    assert float(moments.mature_variance) > float(moments.mature_mean)
    moment_equation_residuals = jnp.stack(
        (
            9.0 * moments.promoter_nascent_covariance - 12.0 * moments.promoter_variance,
            6.5 * moments.promoter_mature_covariance
            - 4.0 * moments.promoter_nascent_covariance,
            4.0 * moments.nascent_variance
            - 12.0 * moments.promoter_mean
            - 12.0 * moments.promoter_nascent_covariance,
            5.5 * moments.nascent_mature_covariance
            - 12.0 * moments.promoter_mature_covariance
            - 4.0 * moments.nascent_variance
            + 12.0 * moments.promoter_mean,
            1.5 * moments.mature_variance
            - 4.0 * moments.nascent_mature_covariance
            - 12.0 * moments.promoter_mean,
        )
    )
    np.testing.assert_allclose(moment_equation_residuals, 0.0, atol=1.0e-6)
    target = TelegraphFitTarget(
        1.1 * moments.fitting_vector,
        jnp.maximum(0.1 * moments.fitting_vector, 0.1),
    )
    logs = jnp.log(model.rates)
    gradient = jax.grad(model.fitting_objective)(logs, target)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.sqrt(jnp.sum(gradient * gradient))) > 0.0
    objective = model.fitting_objective(logs, target)
    improved = model.fitting_objective(logs - 1.0e-4 * gradient, target)
    assert float(improved) < float(objective)
    identifiability = model.identifiability_evidence()
    assert identifiability.rank == 4
    assert not bool(identifiability.locally_identifiable)


def test_capture_and_background_count_likelihood_is_exact():
    measurement = CountMeasurementPlan(0.5, 0.0, observation_capacity=8).prepare()
    evaluation = measurement.log_likelihood(1, 2)
    assert bool(evaluation.valid)
    np.testing.assert_allclose(evaluation.log_likelihood, np.log(0.5), atol=1.0e-6)
    observed_mean, observed_variance = measurement.observed_moments(4, 6)
    np.testing.assert_allclose(observed_mean, 2.0)
    np.testing.assert_allclose(observed_variance, 2.5)
    perfect = CountMeasurementPlan(1.0, 0.0, observation_capacity=8).prepare()
    np.testing.assert_allclose(perfect.log_likelihood(3, 3).log_likelihood, 0.0)
    background = CountMeasurementPlan(1.0, 2.0, observation_capacity=8).prepare()
    np.testing.assert_allclose(
        background.log_likelihood(1, 0).log_likelihood,
        np.log(2.0) - 2.0,
        atol=1.0e-6,
    )
    rare_capture = CountMeasurementPlan(1.0e-30, 0.0, observation_capacity=8).prepare()
    large_latent = rare_capture.log_likelihood(1.0, 1.0e30)
    assert bool(large_latent.valid)
    np.testing.assert_allclose(large_latent.log_likelihood, -1.0, atol=1.0e-4)
    assert not bool(measurement.log_likelihood(9, 10).valid)


def test_evidence_bindings_reject_conflicts_and_cover_prepared_identity():
    model = TelegraphGeneExpressionPlan(2.0, 3.0, 12.0, 4.0, 1.5).prepare()
    reference = BiologicalReference("doi", "10.example/gene", "table:2")
    fact = BiologicalFact("ecoli", "transcription-rate", 12.0, "s^-1", reference)
    condition = BiologicalCondition("culture", "temperature-k", 310.0)
    assertion = PlanFieldAssertion(
        fact.key,
        "telegraph.transcription_rate",
        condition_key=condition.key,
    )
    first = bind_biological_evidence(model, (fact,), (condition,), (assertion,))
    second = bind_biological_evidence(model, (fact,), (condition,), (assertion,))
    assert bool(first.valid)
    assert first.target_id == model.model_id
    assert first.binding_id == second.binding_id
    boolean_fact = BiologicalFact(
        "ecoli", "boolean-rate", True, "dimensionless", reference
    )
    boolean_binding = bind_biological_evidence(
        model,
        (boolean_fact,),
        (),
        (PlanFieldAssertion(boolean_fact.key, "telegraph.activation_rate"),),
    )
    assert not bool(boolean_binding.valid)
    integer_reservoir_fact = BiologicalFact(
        "ecoli", "integer-reservoir", 0, "dimensionless", reference
    )
    reservoir_binding = bind_biological_evidence(
        model,
        (integer_reservoir_fact,),
        (),
        (
            PlanFieldAssertion(
                integer_reservoir_fact.key,
                "network.species.promoter_off.reservoir",
            ),
        ),
    )
    assert not bool(reservoir_binding.valid)
    wrong_unit = BiologicalFact("ecoli", "wrong-unit-rate", 2.0, "kg", reference)
    unit_binding = bind_biological_evidence(
        model,
        (wrong_unit,),
        (),
        (PlanFieldAssertion(wrong_unit.key, "telegraph.activation_rate"),),
    )
    assert not bool(unit_binding.valid)
    with pytest.raises(ValueError, match="reserved"):
        SpeciesSpec("ambiguous.name", "nucleus-cytosol")
    with pytest.raises(ValueError, match="reserved"):
        BiologicalFact(
            "ambiguous:namespace",
            "fact",
            1.0,
            "dimensionless",
            reference,
        )
    conflicting = BiologicalFact("ecoli", "transcription-rate", 11.0, "s^-1", reference)
    with pytest.raises(ValueError, match="Conflicting biological facts"):
        bind_biological_evidence(model, (fact, conflicting), (condition,), (assertion,))


def test_whole_cell_rechecks_regime_after_shared_deltas_are_coupled():
    compartment = CompartmentSpec("cell", 1.0)
    drain = StoichiometricNetworkPlan(
        "drain",
        (compartment,),
        (SpeciesSpec("x", "cell"),),
        (
            StoichiometricProcessSpec(
                "decay",
                {"x": -1},
                MassActionPropensity(0.5, {"x": 1}),
            ),
        ),
    ).prepare()
    processes = (
        WholeCellProcessBinding("first-drain", drain, {"x": "x"}),
        WholeCellProcessBinding("second-drain", drain, {"x": "x"}),
    )
    assembly = WholeCellAssemblyPlan(
        "coupled-drains",
        (ExchangeFieldSpec("x"),),
        processes,
        (
            MultirateScheduleEntry("first-drain", 1),
            MultirateScheduleEntry("second-drain", 1),
        ),
        field_capacity=1,
        process_capacity=2,
    ).prepare()
    state = assembly.initial_state([100.0])
    evaluation = assembly.step(state, 1.0)
    np.testing.assert_allclose(evaluation.candidate, [0.0])
    assert not bool(evaluation.regime_valid)
    assert not bool(evaluation.valid)
    assert not bool(evaluation.commit(state).committed)


def test_whole_cell_exchange_conservation_atomic_commit_and_rollback():
    assembly = _whole_cell_assembly()
    with pytest.raises(ValueError, match="reservoir"):
        WholeCellAssemblyPlan(
            "mixed-reservoir-semantics",
            tuple(
                ExchangeFieldSpec(name)
                for name in ("external_nutrient", "metabolite", "biomass")
            ),
            assembly.plan.processes,
            assembly.plan.schedule,
            field_capacity=5,
            process_capacity=3,
        )
    state = assembly.initial_state(jnp.asarray([100.0, 10.0, 0.0]))
    before = assembly.checkpoint(state)
    evaluation = assembly.step(state, 0.01)
    assert bool(evaluation.valid)
    assert not bool(evaluation.regime_valid)
    np.testing.assert_allclose(evaluation.conservation_residual, 0.0, atol=1.0e-6)
    committed = evaluation.commit(state)
    assert bool(committed.committed)
    np.testing.assert_allclose(committed.state.source_ledger[0], 1.0, atol=3.0e-3)
    after = assembly.checkpoint(committed.state)
    other_lineage = type(state)(
        state.values,
        state.source_ledger,
        state.sink_ledger,
        state.epoch,
        state.assembly_id,
        "other-lineage",
    )
    with pytest.raises(ValueError, match="lineage"):
        evaluation.commit(other_lineage)
    assert before.checkpoint_id != after.checkpoint_id
    failed = assembly.step(committed.state, 10.0)
    assert not bool(failed.valid)
    rolled_back = failed.commit(committed.state)
    assert not bool(rolled_back.committed)
    np.testing.assert_array_equal(rolled_back.state.values, committed.state.values)
    np.testing.assert_array_equal(
        rolled_back.state.source_ledger, committed.state.source_ledger
    )
    np.testing.assert_array_equal(
        rolled_back.state.sink_ledger, committed.state.sink_ledger
    )

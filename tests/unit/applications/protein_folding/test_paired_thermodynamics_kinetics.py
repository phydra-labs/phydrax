from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.protein_folding.experiments import ThermodynamicConvention
from phydrax.applications.protein_folding.thermodynamics import (
    close_free_energy_at_reference,
    EnthalpyReplica,
    fit_heat_capacity_slope,
    paired_state_enthalpy,
    ProteinEnsembleComposition,
)
from phydrax.applications.protein_folding.workflows import (
    ProteinBasinDefinitions,
    ProteinFreeEnergyWorkflow,
    ProteinKineticWorkflow,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.dynamics import StateLayout, TrajectoryData
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.stochastic.path_sampling import StateRegionPlan
from phydrax.units import KILOJOULE_PER_MOLE, PICOSECOND


def _replicas():
    composition = ProteinEnsembleComposition(
        "ordered-protein",
        "fixed-protonation",
        (("protein", 1), ("water-model", 100), ("sodium", 2)),
        "parameter-artifact",
    )
    result = []
    for basin in ("independent-folded-basin", "independent-unfolded-basin"):
        ensemble = []
        for temperature in (280.0, 300.0, 320.0):
            for index, offset in enumerate((-1.0, 1.0)):
                identity = f"{basin}:{temperature}:{index}"
                source = ScientificArtifactEnvelope(
                    artifact_kind="synthetic-enthalpy-ensemble",
                    content_digest=identity,
                    producer="analytic-unit-fixture",
                    producer_version="native",
                    build_id="no-experimental-claim",
                    license_id="CC0-1.0",
                    resource_id=identity,
                    status="complete",
                )
                support = SeriesSupport(jnp.arange(8.0), coordinate_id=PICOSECOND.unit_id)
                delta = 30.0 + 0.5 * (temperature - 300.0) if "unfolded" in basin else 0.0
                values = 1000.0 + delta + offset + jnp.repeat(jnp.asarray([-0.5, 0.5]), 4)
                series = SampledSeries(support, values, series_id=identity)
                ensemble.append(
                    EnthalpyReplica(
                        series,
                        composition,
                        basin,
                        temperature,
                        KILOJOULE_PER_MOLE,
                        PICOSECOND,
                        identity,
                        identity,
                        source,
                        4,
                        0.1,
                        "declared-analytic-correlation-bound",
                        "stationary-analytic-input",
                        "one-pressure",
                    )
                )
        result.append(tuple(ensemble))
    return tuple(result)


def _reference():
    return ReferenceArtifactManifest(
        "synthetic-melting-experiment",
        checksum_algorithm="sha256",
        checksum="1" * 64,
        size_bytes=32,
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"kelvin": 1.0},
        uncertainty={"temperature_standard_error_kelvin": 0.5},
        lineage_ids=("explicit-synthetic-reference",),
    )


def test_matched_state_enthalpy_cp_and_experimental_closure_uncertainty():
    folded, unfolded = _replicas()
    estimate = paired_state_enthalpy(
        folded, unfolded, convention=ThermodynamicConvention()
    )
    np.testing.assert_allclose(estimate.delta_enthalpy, [20.0, 30.0, 40.0])
    # Between-replica uncertainty survives block averaging.
    np.testing.assert_allclose(estimate.standard_errors, np.sqrt(2.0))
    fit = fit_heat_capacity_slope(estimate)
    np.testing.assert_allclose(fit.delta_heat_capacity, 0.5, atol=1e-12)
    np.testing.assert_allclose(fit.residuals, 0.0, atol=1e-12)
    closed = close_free_energy_at_reference(
        fit,
        jnp.asarray([280.0, 300.0, 320.0, 330.0]),
        reference_temperature=300.0,
        reference_delta_g=0.0,
        experimental_covariance=[[0.25, 0.0], [0.0, 0.0]],
        reference=_reference(),
        closure_kind="measured-melting-temperature",
    )
    np.testing.assert_allclose(closed.delta_free_energy[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(closed.standard_errors[1], 0.05, atol=1e-12)
    np.testing.assert_allclose(
        closed.delta_free_energy[0],
        30.0 * (1 - 280 / 300) + 0.5 * (280 - 300 - 280 * np.log(280 / 300)),
        atol=1e-12,
    )
    assert not bool(closed.valid[-1]) and bool(jnp.isnan(closed.delta_free_energy[-1]))
    assert closed.experimental_dependencies == (_reference().manifest_id,)


def test_composition_duplicate_replica_and_underblocked_data_are_not_accepted():
    folded, unfolded = _replicas()
    wrong = replace(
        unfolded[0].composition,
        components=(("protein", 1), ("water-model", 101), ("sodium", 2)),
    )
    with pytest.raises(ValueError, match="composition mismatch"):
        paired_state_enthalpy(
            folded,
            (replace(unfolded[0], composition=wrong), *unfolded[1:]),
            convention=ThermodynamicConvention(),
        )
    with pytest.raises(ValueError, match="independent realization"):
        paired_state_enthalpy(
            folded,
            (
                replace(unfolded[0], independence_id=folded[0].independence_id),
                *unfolded[1:],
            ),
            convention=ThermodynamicConvention(),
        )
    with pytest.raises(ValueError, match="correlation-time"):
        replace(folded[0], correlation_time_bound=1.0)


def test_free_energy_adapters_agree_with_exact_constant_energy_shift():
    convention = ThermodynamicConvention()
    workflow = ProteinFreeEnergyWorkflow(
        ("state-A", "state-B"),
        "matched-box",
        300.0,
        convention,
        "independent-equilibrium-samples",
        "analytically-independent",
    )
    delta = 2.4
    fep = workflow.fep(jnp.full((12,), delta), energy_unit=KILOJOULE_PER_MOLE)
    bar = workflow.bar(
        jnp.full((12,), delta), jnp.full((12,), -delta), energy_unit=KILOJOULE_PER_MOLE
    )
    potentials = jnp.stack((jnp.zeros(12), jnp.full(12, delta)))
    mbar = workflow.mbar(
        potentials, [6, 6], [0] * 6 + [1] * 6, energy_unit=KILOJOULE_PER_MOLE
    )
    for result in (fep, bar, mbar):
        np.testing.assert_allclose(
            result.free_energies[1] - result.free_energies[0], delta, atol=1e-8
        )
    with pytest.raises(ValueError, match="origin labels"):
        workflow.mbar(potentials, [6, 6], [0] * 12, energy_unit=KILOJOULE_PER_MOLE)


def test_kinetic_lag_pairs_never_cross_resets_and_irregular_times_refuse():
    coordinates = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    states = jnp.asarray([[0.0], [0.0], [1.0], [0.0], [1.0], [1.0]])
    data = TrajectoryData(
        coordinates,
        states,
        state_layout=StateLayout((1,)),
        reset_mask=[False, False, True, False, False],
        coordinate_id=PICOSECOND.unit_id,
        source_id="independent-physical-trajectories",
    )
    workflow = ProteinKineticWorkflow(
        data, PICOSECOND, "fixed-condition", "physical-md-time"
    )
    np.testing.assert_array_equal(data.transitions(2).valid, [True, False, False, True])
    assert workflow.require_uniform_lag(2) == 2.0
    basins = ProteinBasinDefinitions(
        ("A", "B"),
        (
            StateRegionPlan.half_open([-0.5], [0.5]),
            StateRegionPlan.half_open([0.5], [1.5]),
        ),
        "held-out-basin-definition",
    )
    model = workflow.markov(states, basins)
    np.testing.assert_allclose(model.diagnostics.counts, [[1.0, 2.0], [0.0, 1.0]])
    irregular = TrajectoryData(
        [0.0, 1.0, 3.0],
        [[0.0], [1.0], [0.0]],
        state_layout=StateLayout((1,)),
        coordinate_id=PICOSECOND.unit_id,
        source_id="irregular-md",
    )
    with pytest.raises(ValueError, match="Irregular physical lags"):
        ProteinKineticWorkflow(
            irregular, PICOSECOND, "same-condition", "physical-md-time"
        ).tica(n_modes=1)
    with pytest.raises(ValueError, match="Configuration bias"):
        replace(workflow, configuration_bias_id="umbrella-bias")
    with pytest.raises(ValueError, match="optimizer traces"):
        replace(workflow, source_kind="optimizer-trace")

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_support_release_requires_all_independent_gates():
    support = phx.applications.aerothermodynamics.AerothermodynamicSupportTuple(
        gas_system_id="gas",
        transport_id="transport",
        discretization_id="fv",
        topology_id="mesh",
        backend="jax-cpu",
        precision="float64",
        species_count=11,
        mode_count=2,
        radiation_group_count=8,
        thermochemistry_id="chemistry",
        radiation_id="radiation",
    )
    with pytest.raises(ValueError, match="all independent gates"):
        phx.applications.aerothermodynamics.AerothermodynamicCapabilityStatus(
            support,
            ("scientific",),
            scientific=True,
            performance=True,
            operational=False,
            security=True,
            released=True,
        )
    status = phx.applications.aerothermodynamics.AerothermodynamicCapabilityStatus(
        support,
        ("scientific", "performance", "operational", "security"),
        scientific=True,
        performance=True,
        operational=True,
        security=True,
        released=True,
    )
    assert status.released


def test_validation_campaign_uses_reference_uncertainty_and_exact_order():
    manifest = phx.qualification.ReferenceArtifactManifest(
        "ram-c-reference",
        checksum_algorithm="sha256",
        checksum="2" * 64,
        size_bytes=1,
        license_id="reference-test",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"electron_density": 1.0},
        uncertainty=None,
        lineage_ids=("ram-c",),
    )
    case = phx.applications.aerothermodynamics.AerothermodynamicValidationCase(
        "ram-c",
        "ionized-plasma",
        manifest,
        ("electron_density", "shock_standoff"),
        jnp.asarray((1.0e18, 0.1)),
        jnp.asarray((1.0e17, 0.01)),
    )
    support = phx.applications.aerothermodynamics.AerothermodynamicSupportTuple(
        gas_system_id="gas",
        transport_id="transport",
        discretization_id="fv",
        topology_id="mesh",
        backend="jax-cpu",
        precision="float64",
        species_count=11,
    )
    campaign = (
        phx.applications.aerothermodynamics.AerothermodynamicValidationCampaignPlan(
            support, (case,), acceptance_sigma=3.0
        )
    )
    evidence = campaign.evaluate(
        {
            "ram-c": {
                "electron_density": jnp.asarray(1.1e18),
                "shock_standoff": jnp.asarray(0.11),
            }
        }
    )
    assert bool(evidence.successful)
    np.testing.assert_allclose(evidence.maximum_normalized_error, 1.0)


def test_sst_and_delayed_detached_eddy_limits_are_finite():
    sst = phx.equations.SSTTurbulencePlan("sst-2003-m")
    evaluation = sst.evaluate(
        1.0,
        1.8e-5,
        0.1,
        10.0,
        jnp.asarray(((0.0, 20.0), (0.0, 0.0))),
        jnp.asarray((0.01, 0.0)),
        jnp.asarray((0.1, 0.0)),
        0.01,
    )
    assert bool(evaluation.successful)
    assert evaluation.eddy_viscosity >= 0.0

    grid = phx.equations.HybridRANSLESGridScalePlan("volume")
    hybrid = phx.equations.DelayedDetachedEddyPlan("sa-iddes", grid)
    result = hybrid.evaluate(
        0.01,
        jnp.asarray((0.002, 0.003, 0.004)),
        1.0e-3,
        1.8e-5,
        100.0,
    )
    assert bool(result.successful)
    assert 0.0 <= result.rans_fraction <= 1.0


def test_multicomponent_filter_preserves_mean_and_recovers_admissibility():
    system = phx.equations.EulerSystem(1)
    mean_primitive = jnp.asarray((1.0, 0.0, 1.0))
    mean = system.primitive_to_conserved(mean_primitive)
    perturbation = jnp.asarray((1.4, 0.0, 0.0))
    state = jnp.stack((mean + perturbation, mean - perturbation))[None, ...]
    plan = phx.equations.fem.MulticomponentAdmissibilityFilterPlan(
        jnp.asarray((0.5, 0.5))
    )
    result = plan.apply(system, state)

    assert bool(result.successful[0])
    assert bool(result.admissible[0])
    np.testing.assert_allclose(jnp.mean(result.state[0], axis=0), mean, atol=2.0e-7)


def test_high_enthalpy_amr_retains_named_indicator_evidence():
    plan = phx.solver.HighEnthalpyAMRIndicatorPlan(
        ("shock", "chemistry", "rarefaction"),
        jnp.asarray((0.5, 0.4, 0.05)),
        jnp.asarray((0.1, 0.1, 0.01)),
    )
    evidence = plan.evaluate(
        {
            "shock": jnp.asarray((0.8, 0.0)),
            "chemistry": jnp.asarray((0.0, 0.0)),
            "rarefaction": jnp.asarray((0.0, 0.0)),
        }
    )
    assert bool(evidence.successful)
    assert evidence.refine_mask.tolist() == [True, False]
    assert evidence.coarsen_mask.tolist() == [False, True]

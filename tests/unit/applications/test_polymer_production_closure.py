import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.polymer_liquids import (
    entanglement as ent,
    production as prod,
    reptation as rep,
)


def _supported_regime():
    return prod.PolymerProductionRegime(
        "hard-sphere-colloid",
        "none",
        "none",
        "free-space-rpy",
        "equilibrium",
    )


def test_exact_support_matrix_admits_only_declared_regimes():
    supported = prod.decide_polymer_production_regime(_supported_regime())
    assert supported.supported
    assert supported.require_supported() is supported.regime

    excluded = prod.decide_polymer_production_regime(
        prod.PolymerProductionRegime(
            "kremer-grest-particle",
            "native-ppa",
            "particle-observables",
            "periodic-rpy",
            "overdamped-affine",
        )
    )
    assert not excluded.supported
    assert "long-range hydrodynamic" in excluded.exclusions[0]
    with pytest.raises(ValueError, match="Unsupported polymer production regime"):
        excluded.require_supported()


def test_entanglement_adapter_parameterizes_named_tube_models():
    result = ent.estimate_entanglement(
        ent.EntanglementEstimatorPlan(
            1.0,
            2.0,
            block_count=4,
            minimum_frames=8,
            maximum_relative_standard_error=1.0e-10,
        ),
        [20, 20],
        jnp.full((8, 2), 40.0),
        jnp.full((8, 2), 20.0),
    )
    lm = prod.likhtman_mcleish_from_entanglement(result, 200, 1.0)
    de = prod.doi_edwards_from_entanglement(result, 100.0)
    assert lm.entanglement_count == 100
    np.testing.assert_allclose(lm.plateau_modulus, result.plateau_modulus)
    np.testing.assert_allclose(de.plateau_modulus, result.plateau_modulus)


def test_primitive_path_contacts_seed_particle_slip_springs():
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
    )
    snapshot = ent.PrimitivePathSnapshot(
        positions,
        jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        jnp.ones((2, 2), dtype=bool),
        jnp.arange(4),
        jnp.ones((2,)),
        jnp.zeros((4, 0), dtype=jnp.int32),
        jnp.zeros((0, 3)),
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        "state",
        "snapshot",
        "prepared",
    )
    contacts = ent.PrimitivePathContactState(
        jnp.asarray([0, -1]),
        jnp.asarray([1, -1]),
        jnp.asarray([1, -1]),
        jnp.asarray([0, -1]),
        jnp.asarray([1.0, jnp.nan]),
        jnp.asarray([True, False]),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    )
    primitive_path = ent.ForcePrimitivePathResult(
        positions,
        jnp.asarray([1.0, 1.0]),
        contacts,
        jnp.asarray([0.0]),
        jnp.asarray([True]),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        "snapshot",
        "ppa",
    )
    seed = prod.primitive_path_contacts_to_slip_spring_seed(snapshot, primitive_path)
    plan, allowed = prod.slip_spring_plan_from_primitive_path(
        seed,
        maximum_particles=4,
        inverse_temperature=1.0,
        stiffness=2.0,
        chemical_potential=0.0,
        maximum_extension=2.0,
    )
    assert int(seed.count) == 1
    np.testing.assert_array_equal(allowed, [[1, 2]])
    assert plan.maximum_springs == 1


def test_composite_checkpoint_roundtrip_and_replay(tmp_path):
    glamm = rep.GLAMMPlan(3, 1.0e-3, 1.0, 10.0, 1.0, contour_diffusivity=0.0)
    component = glamm.initialize()
    plan = prod.CompositePolymerCheckpointPlan(
        {"constitutive": glamm.plan_id}, _supported_regime().regime_id
    )
    state = plan.state(
        {"constitutive": component},
        component_ids={"constitutive": glamm.plan_id},
        step_index=3,
        accepted_steps=3,
    )
    checkpoint = prod.write_composite_polymer_checkpoint(
        tmp_path / "polymer-checkpoint.npz", plan, state
    )
    restored = prod.read_composite_polymer_checkpoint(
        tmp_path / "polymer-checkpoint.npz", plan, state
    )
    evidence = prod.compare_composite_polymer_replay(state, restored.state)
    assert checkpoint.payload_id == restored.payload_id
    assert evidence.identical


def test_qualification_campaign_is_fail_closed_and_evidence_bearing():
    cases = prod.default_polymer_qualification_cases()
    campaign = prod.PolymerQualificationCampaignPlan(_supported_regime(), cases)
    observations = {case.name: case.reference for case in cases}
    qualified = prod.evaluate_polymer_qualification(campaign, observations)
    assert qualified.qualified
    failed_observations = dict(observations)
    failed_observations["rpy-symmetry-residual"] = jnp.asarray(1.0)
    failed = prod.evaluate_polymer_qualification(campaign, failed_observations)
    assert not failed.qualified


def test_bounded_production_smoke_closes_all_gates():
    result = prod.run_polymer_production_smoke()
    assert result.successful
    assert bool(jnp.all(result.gate_passed))

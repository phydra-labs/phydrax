#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._admissibility import AdmissibilityReason, DOMAIN_REASON_SHIFT
from tests._ported_models import full_port, PortedAffine


def _manifest(name, *, training=True):
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=("a" if name == "model" else "b") * 64,
        size_bytes=1,
        license_id="test-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=training,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty={"synthetic": 0.0},
        lineage_ids=(f"source:{name}",),
    )


def _mechanism():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.zeros((2,)),
        minimum_temperature=200.0,
        maximum_temperature=6000.0,
    )
    return phx.equations.ChemicalMechanismIR(
        "A-to-B",
        schema,
        species,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(1.0),
            ),
        ),
    ).prepare()


def _schema():
    return phx.applications.reacting_flow.LearnedChemicalFeatureSchema(
        ("log_A", "log_B", "temperature", "pressure", "log_step"),
        ("log(mol/m3)", "log(mol/m3)", "K", "Pa", "log(s)"),
        jnp.asarray((-10.0, -10.0, 300.0, 5.0e4, -10.0)),
        jnp.asarray((10.0, 10.0, 2000.0, 2.0e5, 0.0)),
    )


def _uncertainty(features):
    return jnp.zeros(features.shape[:-1])


def test_supported_learned_extent_preserves_invariants_without_fallback():
    def model(features):
        return 0.01 * jnp.ones(features.shape[:-1] + (1,))

    plan = phx.applications.reacting_flow.LearnedChemicalTransitionPlan(
        _mechanism(),
        _schema(),
        model,
        _uncertainty,
        _manifest("model"),
        (_manifest("training"),),
        model_id="extent-model",
        maximum_uncertainty=0.1,
    )
    result = plan.advance(jnp.asarray((0.9, 0.1)), 1000.0, 101325.0, 0.01)

    assert bool(result.successful)
    assert not bool(result.fallback_used)
    np.testing.assert_allclose(result.accepted_concentrations, (0.89, 0.11))
    np.testing.assert_allclose(result.element_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.charge_residual, 0.0, atol=1e-12)


def test_out_of_support_and_nonphysical_models_use_exact_mechanism():
    def bad_model(features):
        return 2.0 * jnp.ones(features.shape[:-1] + (1,))

    plan = phx.applications.reacting_flow.LearnedChemicalTransitionPlan(
        _mechanism(),
        _schema(),
        bad_model,
        _uncertainty,
        _manifest("model"),
        (_manifest("training"),),
        model_id="bad-extent-model",
        maximum_uncertainty=0.1,
        exact_subcycles=8,
        exact_iterations=4,
    )
    negative = plan.advance(jnp.asarray((0.9, 0.1)), 1000.0, 101325.0, 0.01)
    out_of_support = plan.advance(jnp.asarray((0.9, 0.1)), 3000.0, 101325.0, 0.01)

    assert bool(negative.successful)
    assert bool(negative.fallback_used)
    assert int(negative.fallback_reason) == int(
        phx.applications.reacting_flow.LearnedChemicalFallbackReason.NEGATIVE_SPECIES
    )
    assert negative.accepted_concentrations[1] > 0.1
    assert bool(out_of_support.fallback_used)
    assert int(out_of_support.fallback_reason) == int(
        phx.applications.reacting_flow.LearnedChemicalFallbackReason.OUT_OF_SUPPORT
    )


def _small_extent_plan(model_id="extent-model"):
    def model(features):
        return 0.01 * jnp.ones(features.shape[:-1] + (1,))

    return phx.applications.reacting_flow.LearnedChemicalTransitionPlan(
        _mechanism(),
        _schema(),
        model,
        _uncertainty,
        _manifest("model"),
        (_manifest("training"),),
        model_id=model_id,
        maximum_uncertainty=0.1,
    )


_LANES = jnp.asarray(((0.9, 0.1), (0.9, 0.1)))
_TEMPERATURES = jnp.asarray((1000.0, 3000.0))


def test_header_marks_learned_lanes_eligible_and_records_out_of_support_fallback():
    plan = _small_extent_plan()
    result = plan.advance(_LANES, _TEMPERATURES, 101325.0, 0.01)

    np.testing.assert_array_equal(result.fallback_used, (False, True))
    np.testing.assert_array_equal(result.header.eligible, ~result.fallback_used)
    np.testing.assert_array_equal(
        result.header.reason_bits,
        (0, int(AdmissibilityReason.OUTSIDE_SUPPORT)),
    )
    assert float(result.header.margin[0]) >= 0.0 > float(result.header.margin[1])
    assert result.header.model_id == plan.component_id
    assert result.header.evidence_id == plan.plan_id
    assert _small_extent_plan("other-model").component_id != plan.component_id
    assert result.derivative_contract.conditions == ("decisions-frozen",)


def test_header_sets_domain_bit_for_nonphysical_learned_extent():
    def bad_model(features):
        return 2.0 * jnp.ones(features.shape[:-1] + (1,))

    plan = phx.applications.reacting_flow.LearnedChemicalTransitionPlan(
        _mechanism(),
        _schema(),
        bad_model,
        _uncertainty,
        _manifest("model"),
        (_manifest("training"),),
        model_id="bad-extent-model",
        maximum_uncertainty=0.1,
    )
    result = plan.advance(jnp.asarray((0.9, 0.1)), 1000.0, 101325.0, 0.01)
    negative_bit = 1 << (
        DOMAIN_REASON_SHIFT
        + int(
            phx.applications.reacting_flow.LearnedChemicalFallbackReason.NEGATIVE_SPECIES
        )
    )

    assert not bool(result.header.eligible)
    assert int(result.header.reason_bits) & negative_bit
    assert not int(result.header.reason_bits) & int(AdmissibilityReason.OUTSIDE_SUPPORT)


def test_invalid_derivative_lanes_are_poisoned_without_changing_primals():
    plan = _small_extent_plan()
    result = plan.advance(_LANES, _TEMPERATURES, 101325.0, 0.01)

    np.testing.assert_array_equal(result.derivative_valid, (True, False))
    np.testing.assert_allclose(result.accepted_concentrations[0], (0.89, 0.11))
    assert bool(jnp.all(jnp.isfinite(result.accepted_concentrations)))

    jacobian = jax.jacfwd(
        lambda values: (
            plan.advance(values, _TEMPERATURES, 101325.0, 0.01).accepted_concentrations
        )
    )(_LANES)
    assert bool(jnp.all(jnp.isfinite(jacobian[0])))
    assert bool(jnp.all(jnp.isnan(jacobian[1])))


class _ConstantExtent(phx.AbstractArrayModel):
    extent: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, extent):
        self.extent = jnp.asarray(extent)
        self.in_size = 5
        self.out_size = 1

    def __call__(self, features, /, *, key=None):
        return self.extent * jnp.ones(features.shape[:-1] + (1,))


def _artifact(model=None):
    return phx.applications.reacting_flow.LearnedChemicalTransitionPlan(
        _mechanism(),
        _schema(),
        phx.uq.FrozenModel(_ConstantExtent(0.01)) if model is None else model,
        _uncertainty,
        _manifest("model"),
        (_manifest("training"),),
        model_id="extent-model",
        maximum_uncertainty=0.1,
    )


def _target_loss(plan):
    result = plan.advance(jnp.asarray((0.9, 0.1)), 1000.0, 101325.0, 0.01)
    return jnp.sum((result.accepted_concentrations - jnp.asarray((0.87, 0.13))) ** 2)


def test_frozen_chemistry_artifact_exposes_no_parameters():
    artifact = _artifact()
    resolution = phx.require_parameter_roles(artifact, context="chemistry artifact")

    assert phx.ArrayRole.PARAMETER not in resolution.roles
    assert not jax.tree_util.tree_leaves(phx.partition_parameters(artifact)[0])


def test_explicit_trainable_chemistry_binding_trains_without_touching_the_artifact():
    artifact = _artifact()
    trainable = artifact.as_trainable_binding()

    assert trainable.plan_id == artifact.plan_id
    assert trainable.component_id == artifact.component_id
    (parameter,) = jax.tree_util.tree_leaves(phx.partition_parameters(trainable)[0])
    assert float(parameter) == 0.01

    @eqx.filter_jit
    def step(plan):
        parameters, model_state, fixed = phx.partition_parameters(plan)
        gradient = jax.grad(
            lambda values: _target_loss(
                phx.combine_parameters(values, model_state, fixed)
            )
        )(parameters)
        updated = jax.tree_util.tree_map(
            lambda value, slope: value - 0.1 * slope, parameters, gradient
        )
        return phx.combine_parameters(updated, model_state, fixed)

    trained = step(step(trainable))
    assert float(_target_loss(trained)) < float(_target_loss(trainable))
    assert float(artifact.model.model.extent) == 0.01
    assert float(trained.model.extent) > 0.01


class _FixedExtent(phx.AbstractArrayModel):
    extent: jax.Array = phx.fixed_field()
    in_size: int = eqx.field(static=True, default=5)
    out_size: int = eqx.field(static=True, default=1)

    def __call__(self, features, /, *, key=None):
        return self.extent * jnp.ones(features.shape[:-1] + (1,))


def test_trainable_binding_requires_a_parameter_model_array():
    def model(features):
        return 0.01 * jnp.ones(features.shape[:-1] + (1,))

    for untrainable in (model, _FixedExtent(jnp.asarray(0.01))):
        with pytest.raises(ValueError, match="no PARAMETER leaf"):
            _artifact(untrainable).as_trainable_binding()


def test_port_declaring_extent_models_are_refused_without_owner_ports():
    ported = PortedAffine(
        phx.ModelPorts(
            inputs=(full_port("chemistry.features", (5,)),),
            outputs=(full_port("chemistry.extent", (1,)),),
        ),
        out_size=1,
    )
    for model in (ported, phx.uq.FrozenModel(ported)):
        with pytest.raises(ValueError, match="no owner value ports; its model"):
            _artifact(model)

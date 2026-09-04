from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.closure_data import (
    LearnedClosureBindingPlan,
    LearnedStressBindingPlan,
    LearnedStressFeatureSchema,
    LearnedStressOutputContract,
    LearnedStressResult,
    NormalizerProvenance,
    PreparedLearnedStressBinding,
    TrainOnlyNormalizer,
)
from phydrax.equations import LESParameterProvenance, ResolvedLESFilter


_FEATURES = jnp.asarray(((3.0, 2.0), (-1.0, 2.0)), dtype=jnp.float32)
_STRAIN = jnp.asarray(
    (
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0)),
    ),
    dtype=jnp.float32,
)


def _predict_stress(features, _args):
    a = features[..., 0]
    b = features[..., 1]
    zero = jnp.zeros_like(a)
    return jnp.stack(
        (
            jnp.stack((a, b, zero), axis=-1),
            jnp.stack((b, -a, zero), axis=-1),
            jnp.stack((zero, zero, zero), axis=-1),
        ),
        axis=-2,
    )


def _asymmetric_prediction(features, args):
    stress = _predict_stress(features, args)
    return stress.at[0, 0, 1].add(1.0)


def _traceful_prediction(features, args):
    stress = _predict_stress(features, args)
    return stress + jnp.eye(3, dtype=stress.dtype)


def _nonfinite_prediction(features, args):
    return _predict_stress(features, args).at[0, 0, 0].set(jnp.nan)


def _wrong_shape_prediction(features, args):
    return _predict_stress(features, args)[..., 0]


def _wrong_dtype_prediction(features, args):
    return _predict_stress(features, args).astype(jnp.float16)


def _resolved_filter(name="cell filter"):
    return ResolvedLESFilter(
        name,
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="volume-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="unmodeled",
    )


def _normalizer(*, feature_name="resolved-gradient", schema_id="flow-schema"):
    provenance = NormalizerProvenance(
        partition_id="train-partition",
        training_assignment_ids=("assignment-0",),
        training_sample_ids=("sample-0",),
        feature_name=feature_name,
        schema_id=schema_id,
    )
    return TrainOnlyNormalizer(
        jnp.asarray((1.0, 2.0), dtype=jnp.float32),
        jnp.asarray((2.0, 1.0), dtype=jnp.float32),
        provenance,
        epsilon=1e-6,
    )


def _feature_schema():
    return LearnedStressFeatureSchema(
        name="resolved-gradient",
        component_names=("s_xx", "s_xy"),
        component_units=("1/s", "1/s"),
        shape=(2, 2),
        dtype=jnp.float32,
        flow_schema_id="flow-schema",
    )


def _output_contract(
    resolved_filter,
    *,
    filter_id=None,
    discretization_id="mesh-32-cells",
    regime="constant-density-incompressible",
):
    return LearnedStressOutputContract(
        shape=(2, 3, 3),
        dtype=jnp.float32,
        units="(m/s)^2",
        target_id="deviatoric-specific-stress-target",
        filter_id=resolved_filter.filter_id if filter_id is None else filter_id,
        discretization_id=discretization_id,
        regime=regime,
        stress_convention="deviatoric",
        symmetry_tolerance=1e-6,
        trace_tolerance=1e-6,
    )


def _prepared(
    *,
    policy="signed",
    fraction=None,
    predictor=_predict_stress,
):
    resolved_filter = _resolved_filter()
    provenance = LESParameterProvenance(
        resolved_filter,
        "mesh-32-cells",
        "constant-density-incompressible",
        source_kind="user",
        evidence_ids=(),
    )
    normalizer = _normalizer()
    plan = LearnedStressBindingPlan(
        _feature_schema(),
        _output_contract(resolved_filter),
        resolved_filter,
        provenance,
        model_artifact_id="stress-model-sha256",
        normalizer_id=normalizer.normalizer_id,
        energy_policy=policy,
        maximum_backscatter_fraction=fraction,
    )
    prepared = plan.prepare(
        predictor,
        normalizer,
        model_artifact_id="stress-model-sha256",
        target_id="deviatoric-specific-stress-target",
        output_units="(m/s)^2",
    )
    return prepared


def test_signed_stress_validates_contract_and_preserves_backscatter():
    prepared = _prepared()
    result = prepared(_FEATURES, _STRAIN)

    assert isinstance(prepared, PreparedLearnedStressBinding)
    assert isinstance(result, LearnedStressResult)
    np.testing.assert_allclose(result.local_transfer, (-2.0, 2.0))
    np.testing.assert_allclose(
        jnp.trace(result.stress, axis1=-2, axis2=-1), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(result.stress, jnp.swapaxes(result.stress, -1, -2))
    assert not bool(result.evidence.correction_applied)
    assert result.evidence.differentiation_semantics == "smooth_discrete"
    assert result.evidence.target_id == prepared.plan.output_contract.target_id
    assert result.evidence.filter_id == prepared.plan.resolved_filter.filter_id
    assert result.evidence.valid


def test_dissipative_projection_removes_only_negative_local_transfer():
    result = _prepared(policy="dissipative")(_FEATURES, _STRAIN)

    np.testing.assert_allclose(result.evidence.raw_local_transfer, (-2.0, 2.0))
    np.testing.assert_allclose(result.local_transfer, (0.0, 2.0), atol=1e-6)
    assert bool(result.evidence.correction_applied)
    assert bool(result.evidence.correction_active[0])
    assert not bool(result.evidence.correction_active[1])
    assert result.evidence.differentiation_semantics == "branchwise"
    np.testing.assert_allclose(result.evidence.selected_backscatter_transfer, 0.0)


def test_bounded_backscatter_caps_aggregate_without_claiming_pointwise_dissipation():
    result = _prepared(policy="bounded_backscatter", fraction=0.25)(_FEATURES, _STRAIN)

    np.testing.assert_allclose(result.evidence.raw_forward_transfer, 2.0)
    np.testing.assert_allclose(result.evidence.raw_backscatter_transfer, 2.0)
    np.testing.assert_allclose(result.evidence.backscatter_limit, 0.5)
    np.testing.assert_allclose(result.evidence.selected_backscatter_transfer, 0.5)
    np.testing.assert_allclose(result.local_transfer, (-0.5, 2.0), atol=1e-6)
    assert float(result.local_transfer[0]) < 0.0
    np.testing.assert_allclose(
        jnp.trace(result.stress, axis1=-2, axis2=-1), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(result.stress, jnp.swapaxes(result.stress, -1, -2))


@pytest.mark.parametrize(
    ("predictor", "message"),
    (
        (_asymmetric_prediction, "symmetric"),
        (_traceful_prediction, "trace-free"),
        (_nonfinite_prediction, "nonfinite"),
    ),
)
def test_invalid_prediction_is_refused_without_a_zero_fallback(predictor, message):
    prepared = _prepared(predictor=predictor)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match=message):
        result = prepared(_FEATURES, _STRAIN)
        jax.block_until_ready(result.stress)


@pytest.mark.parametrize(
    ("predictor", "error", "message"),
    (
        (_wrong_shape_prediction, ValueError, "shape"),
        (_wrong_dtype_prediction, TypeError, "dtype"),
    ),
)
def test_prediction_shape_and_dtype_must_match_exactly(predictor, error, message):
    prepared = _prepared(predictor=predictor)

    with pytest.raises(error, match=message):
        prepared(_FEATURES, _STRAIN)


def test_feature_shape_dtype_and_output_convention_are_exact():
    prepared = _prepared()

    with pytest.raises(ValueError, match="features.*shape"):
        prepared(_FEATURES[:1], _STRAIN)
    with pytest.raises(TypeError, match="features.*dtype"):
        prepared(_FEATURES.astype(jnp.float16), _STRAIN)
    with pytest.raises(ValueError, match="deviatoric"):
        LearnedStressOutputContract(
            shape=(2, 3, 3),
            dtype=jnp.float32,
            units="(m/s)^2",
            target_id="target",
            filter_id="filter",
            discretization_id="mesh",
            regime="constant-density",
            stress_convention="full",
        )


def test_prepare_refuses_artifact_normalizer_target_and_units_mismatches():
    prepared = _prepared()
    plan = prepared.plan
    normalizer = prepared.normalizer

    mismatches = (
        {"model_artifact_id": "other-artifact"},
        {"target_id": "other-target"},
        {"output_units": "Pa"},
    )
    defaults = {
        "model_artifact_id": "stress-model-sha256",
        "target_id": "deviatoric-specific-stress-target",
        "output_units": "(m/s)^2",
    }
    for mismatch in mismatches:
        with pytest.raises(ValueError):
            plan.prepare(
                _predict_stress,
                normalizer,
                **(defaults | mismatch),
            )

    other_normalizer = TrainOnlyNormalizer(
        normalizer.mean + 1.0,
        normalizer.scale,
        normalizer.provenance,
        epsilon=normalizer.epsilon,
    )
    with pytest.raises(ValueError, match="normalizer"):
        plan.prepare(_predict_stress, other_normalizer, **defaults)


def test_plan_refuses_filter_provenance_and_feature_identity_mismatches():
    resolved_filter = _resolved_filter()
    other_filter = _resolved_filter("other cell filter")
    provenance = LESParameterProvenance(
        resolved_filter,
        "mesh-32-cells",
        "constant-density-incompressible",
        source_kind="user",
        evidence_ids=(),
    )
    normalizer = _normalizer()

    with pytest.raises(ValueError, match="filter identities"):
        LearnedStressBindingPlan(
            _feature_schema(),
            _output_contract(resolved_filter, filter_id=other_filter.filter_id),
            resolved_filter,
            provenance,
            model_artifact_id="stress-model-sha256",
            normalizer_id=normalizer.normalizer_id,
        )

    other_provenance = LESParameterProvenance(
        other_filter,
        "mesh-32-cells",
        "constant-density-incompressible",
        source_kind="user",
        evidence_ids=(),
    )
    with pytest.raises(ValueError, match="provenance identities"):
        LearnedStressBindingPlan(
            _feature_schema(),
            _output_contract(resolved_filter),
            resolved_filter,
            other_provenance,
            model_artifact_id="stress-model-sha256",
            normalizer_id=normalizer.normalizer_id,
        )

    for mismatch in (
        {"discretization_id": "other-mesh"},
        {"regime": "other-regime"},
    ):
        with pytest.raises(ValueError, match="provenance identities"):
            LearnedStressBindingPlan(
                _feature_schema(),
                _output_contract(resolved_filter, **mismatch),
                resolved_filter,
                provenance,
                model_artifact_id="stress-model-sha256",
                normalizer_id=normalizer.normalizer_id,
            )

    bad_normalizer = _normalizer(schema_id="other-flow-schema")
    plan = LearnedStressBindingPlan(
        _feature_schema(),
        _output_contract(resolved_filter),
        resolved_filter,
        provenance,
        model_artifact_id="stress-model-sha256",
        normalizer_id=bad_normalizer.normalizer_id,
    )
    with pytest.raises(ValueError, match="provenance"):
        plan.prepare(
            _predict_stress,
            bad_normalizer,
            model_artifact_id="stress-model-sha256",
            target_id="deviatoric-specific-stress-target",
            output_units="(m/s)^2",
        )


def test_prepared_binding_is_jittable_and_has_a_finite_jvp():
    prepared = _prepared()
    evaluate = eqx.filter_jit(lambda values: prepared(values, _STRAIN))
    compiled = evaluate(_FEATURES)
    np.testing.assert_allclose(compiled.local_transfer, (-2.0, 2.0))

    primal, tangent = jax.jvp(
        lambda values: prepared(values, _STRAIN).local_transfer,
        (_FEATURES,),
        (jnp.ones_like(_FEATURES),),
    )
    assert jnp.all(jnp.isfinite(primal))
    assert jnp.all(jnp.isfinite(tangent))


def test_existing_generic_binding_contract_remains_unchanged():
    predictor = lambda values, args: values if args is None else args * values
    binding = LearnedClosureBindingPlan(
        predictor,
        deployment_kind="conservative_face",
        schema_id="legacy-schema",
        input_component_names=("rho",),
        output_component_names=("rho",),
        model_artifact_id="legacy-model",
        normalizer_provenance_id="legacy-normalizer",
    )

    assert binding.predictor is predictor
    assert binding.deployment_kind == "conservative_face"
    assert binding.differentiability == "smooth_discrete"

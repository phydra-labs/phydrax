from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.closure_data import (
    LearnedStressBindingPlan,
    LearnedStressFeatureSchema,
    LearnedStressOutputContract,
    NormalizerProvenance,
    TrainOnlyNormalizer,
)
from phydrax.equations._learned_stress import (
    LEARNED_STRESS_FEATURE_NAME,
    LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
    LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
    MACLearnedStressPlan,
    MACLearnedStressStage,
    PeriodicLearnedStressPlan,
    PeriodicLearnedStressStage,
    PreparedMACLearnedStress,
    PreparedPeriodicLearnedStress,
)
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)


def _deviatoric_strain(features):
    gradient = features.reshape(features.shape[:-1] + (3, 3))
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    return strain - (trace / 3.0)[..., None, None] * jnp.eye(3, dtype=strain.dtype)


def _signed_viscosity_predictor(features, args):
    strain = _deviatoric_strain(features)
    coefficient = features[..., 1] if args is None else jnp.asarray(args)
    return -2.0 * coefficient[..., None, None] * strain


def _positive_viscosity_predictor(features, args):
    strain = _deviatoric_strain(features)
    coefficient = jnp.asarray(0.2 if args is None else args, dtype=features.dtype)
    return -2.0 * coefficient * strain


def _nonfinite_predictor(features, args):
    return _positive_viscosity_predictor(features, args).at[0, 0, 0, 0, 0].set(jnp.nan)


def _periodic_space(count=5):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in range(3))
    )


def _periodic_velocity(space):
    x, y, z = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        space.axes[2].nodes,
        indexing="ij",
    )
    return jnp.stack(
        (
            jnp.sin(y) * jnp.cos(z),
            jnp.sin(z) * jnp.cos(x),
            jnp.sin(x) * jnp.cos(y),
        ),
        axis=-1,
    )


def _periodic_filter():
    return ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )


def _mac_grid(*, counts=(4, 4, 4), nonuniform=False):
    specs = (
        tuple(
            phx.discretization.NonuniformCellAxisSpec(
                jnp.asarray((0.0, 0.1, 0.35, 0.7, 1.0)), periodic=True
            )
            for _ in range(3)
        )
        if nonuniform
        else tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in counts
        )
    )
    bounds = (
        jnp.asarray([[0.0] * 3, [1.0] * 3])
        if nonuniform
        else jnp.asarray([[0.0] * 3, [2.0 * jnp.pi] * 3])
    )
    grid = phx.discretization.TensorGridPlan(specs, axis_names=("x", "y", "z")).prepare(
        bounds
    )
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="iterative" if nonuniform else "transform"
    )
    return discretization, operators, momentum, projection


def _mac_velocity(discretization):
    x_faces, y_faces, z_faces = discretization.face_centers
    return (
        jnp.sin(x_faces[..., 1]) * jnp.cos(x_faces[..., 2]),
        jnp.sin(y_faces[..., 2]) * jnp.cos(y_faces[..., 0]),
        jnp.sin(z_faces[..., 0]) * jnp.cos(z_faces[..., 1]),
    )


def _mac_filter():
    return ResolvedLESFilter(
        "mac-cell-volume",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )


def _binding(
    *,
    sample_shape,
    dtype,
    resolved_filter,
    discretization_id,
    regime,
    policy="signed",
    fraction=None,
    predictor=_positive_viscosity_predictor,
    artifact="learned-stress-model",
    component_names=LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
):
    flow_schema_id = f"flow-{discretization_id}"
    schema = LearnedStressFeatureSchema(
        name=LEARNED_STRESS_FEATURE_NAME,
        component_names=component_names,
        component_units=LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
        shape=sample_shape + (9,),
        dtype=dtype,
        flow_schema_id=flow_schema_id,
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization_id,
        regime,
        source_kind="user",
        evidence_ids=(),
    )
    output = LearnedStressOutputContract(
        shape=sample_shape + (3, 3),
        dtype=dtype,
        units="(m/s)^2",
        target_id="deviatoric-specific-stress-target",
        filter_id=resolved_filter.filter_id,
        discretization_id=discretization_id,
        regime=regime,
        symmetry_tolerance=2.0e-6,
        trace_tolerance=2.0e-6,
    )
    normalizer_provenance = NormalizerProvenance(
        partition_id="training-partition",
        training_assignment_ids=("assignment",),
        training_sample_ids=("sample",),
        feature_name=LEARNED_STRESS_FEATURE_NAME,
        schema_id=flow_schema_id,
    )
    normalizer = TrainOnlyNormalizer(
        jnp.zeros((9,), dtype=dtype),
        jnp.ones((9,), dtype=dtype),
        normalizer_provenance,
        epsilon=1.0e-12,
    )
    plan = LearnedStressBindingPlan(
        schema,
        output,
        resolved_filter,
        provenance,
        model_artifact_id=artifact,
        normalizer_id=normalizer.normalizer_id,
        energy_policy=policy,
        maximum_backscatter_fraction=fraction,
    )
    return plan.prepare(
        predictor,
        normalizer,
        model_artifact_id=artifact,
        target_id=output.target_id,
        output_units=output.units,
    )


def _periodic_prepared(
    *,
    policy="signed",
    fraction=None,
    predictor=_positive_viscosity_predictor,
    artifact="periodic-learned-stress",
):
    space = _periodic_space()
    binding = _binding(
        sample_shape=space.physical_shape,
        dtype=jnp.dtype(space.plan.precision.physical_dtype),
        resolved_filter=_periodic_filter(),
        discretization_id=space.prepared_id,
        regime="three-dimensional-periodic-unit-density",
        policy=policy,
        fraction=fraction,
        predictor=predictor,
        artifact=artifact,
    )
    projector = phx.discretization.PeriodicLerayProjector(space)
    return space, PeriodicLearnedStressPlan(binding).prepare(space, projector)


def _mac_prepared(
    *,
    policy="signed",
    fraction=None,
    predictor=_positive_viscosity_predictor,
    artifact="mac-learned-stress",
):
    discretization, operators, momentum, projection = _mac_grid()
    binding = _binding(
        sample_shape=discretization.cell_shape,
        dtype=operators.pressure_space.dtype,
        resolved_filter=_mac_filter(),
        discretization_id=discretization.prepared_id,
        regime="incompressible-unit-density",
        policy=policy,
        fraction=fraction,
        predictor=predictor,
        artifact=artifact,
    )
    prepared = MACLearnedStressPlan(binding).prepare(momentum, projection)
    return discretization, operators, momentum, prepared


def test_periodic_backend_owns_conservative_divergence_projection_and_work():
    space, prepared = _periodic_prepared(
        policy="dissipative", predictor=_signed_viscosity_predictor
    )
    state = prepared.projector.project(space.project(_periodic_velocity(space)))
    result = prepared(state)

    assert isinstance(prepared, PreparedPeriodicLearnedStress)
    assert isinstance(result, PeriodicLearnedStressStage)
    assert result.prepared_id == prepared.prepared_id
    assert result.feature_schema_id == prepared.feature_schema_id
    assert result.energy_policy_active
    assert jnp.all(result.learned_result.local_transfer >= -1.0e-10)
    assert result.successful
    assert result.energy_policy_satisfied
    assert result.integrated_transfer >= 0.0
    assert result.momentum_conservation_defect < 1.0e-12
    assert result.divergence_norm < 2.0e-10
    assert jnp.abs(result.energy_identity_defect) < 2.0e-9
    assert jnp.abs(result.projection_work_defect) < 2.0e-9
    np.testing.assert_allclose(
        result.learned_result.stress,
        jnp.swapaxes(result.learned_result.stress, -1, -2),
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        jnp.trace(result.learned_result.stress, axis1=-2, axis2=-1),
        0.0,
        atol=2.0e-10,
    )


def test_periodic_bounded_backscatter_preserves_policy_activity_jit_and_jvp():
    space, prepared = _periodic_prepared(
        policy="bounded_backscatter",
        fraction=0.2,
        predictor=_signed_viscosity_predictor,
    )
    state = prepared.projector.project(space.project(_periodic_velocity(space)))
    eager = prepared(state)
    compiled = eqx.filter_jit(lambda value: prepared(value))(state)
    tangent = 0.1 * state
    _, derivative = jax.jvp(
        lambda value: prepared(value).projected_rate,
        (state,),
        (tangent,),
    )

    assert eager.energy_policy_active
    assert eager.energy_policy_satisfied
    assert eager.learned_result.evidence.selected_backscatter_transfer <= (
        eager.learned_result.evidence.backscatter_limit + 1.0e-9
    )
    np.testing.assert_allclose(compiled.projected_rate, eager.projected_rate, atol=2e-9)
    assert jnp.all(jnp.isfinite(derivative))
    assert prepared.projector.divergence_norm(derivative) < 2.0e-9


def test_mac_backend_owns_conservative_divergence_projection_and_work():
    discretization, operators, momentum, prepared = _mac_prepared(policy="dissipative")
    velocity = _mac_velocity(discretization)
    result = prepared(velocity, momentum.boundaries.homogeneous_stage())

    assert isinstance(prepared, PreparedMACLearnedStress)
    assert isinstance(result, MACLearnedStressStage)
    assert result.prepared_id == prepared.prepared_id
    assert result.successful
    assert result.conservative
    assert result.energy_policy_satisfied
    assert result.integrated_transfer >= 0.0
    assert jnp.max(result.momentum_conservation_defect) < 2.0e-10
    assert jnp.max(jnp.abs(result.projection.divergence_after)) < 2.0e-10
    assert jnp.abs(result.energy_identity_defect) < 2.0e-9
    work = jnp.real(operators.velocity_space.inner(velocity, result.unprojected_rate))
    np.testing.assert_allclose(work, -result.integrated_transfer, atol=2.0e-9)
    np.testing.assert_allclose(work, result.unprojected_work, atol=2.0e-12)
    for physical, projected in zip(
        result.physical_rate, result.projected_rate, strict=True
    ):
        np.testing.assert_array_equal(physical, projected)
    np.testing.assert_array_equal(result.integrated_work, result.projected_work)


def test_mac_signed_backend_is_jittable_and_forward_differentiable():
    discretization, _, momentum, prepared = _mac_prepared()
    velocity = _mac_velocity(discretization)
    boundary_stage = momentum.boundaries.homogeneous_stage()
    eager = prepared(velocity, boundary_stage)
    compiled = eqx.filter_jit(lambda value, stage: prepared(value, stage))(
        velocity, boundary_stage
    )
    tangent = tuple(0.1 * component for component in velocity)
    _, derivative = jax.jvp(
        lambda value: prepared(value, boundary_stage).projected_rate,
        (velocity,),
        (tangent,),
    )

    for actual, expected in zip(
        compiled.projected_rate, eager.projected_rate, strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2.0e-9)
    assert all(jnp.all(jnp.isfinite(component)) for component in derivative)


@pytest.mark.parametrize("backend", ("periodic", "mac"))
def test_invalid_learned_prediction_is_refused_without_zero_fallback(backend):
    if backend == "periodic":
        space, prepared = _periodic_prepared(predictor=_nonfinite_predictor)
        arguments = (
            prepared.projector.project(space.project(_periodic_velocity(space))),
        )
    else:
        discretization, _, momentum, prepared = _mac_prepared(
            predictor=_nonfinite_predictor
        )
        arguments = (
            _mac_velocity(discretization),
            momentum.boundaries.homogeneous_stage(),
        )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonfinite"):
        result = prepared(*arguments)
        jax.block_until_ready(result.learned_result.stress)


def test_adapters_refuse_incompatible_abi_layout_filter_and_mac_grid():
    space = _periodic_space()
    wrong_order = tuple(reversed(LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS))
    wrong_abi = _binding(
        sample_shape=space.physical_shape,
        dtype=jnp.dtype(space.plan.precision.physical_dtype),
        resolved_filter=_periodic_filter(),
        discretization_id=space.prepared_id,
        regime="three-dimensional-periodic-unit-density",
        component_names=wrong_order,
    )
    with pytest.raises(ValueError, match="row-major nine-component"):
        PeriodicLearnedStressPlan(wrong_abi)

    wrong_filter = _binding(
        sample_shape=space.physical_shape,
        dtype=jnp.dtype(space.plan.precision.physical_dtype),
        resolved_filter=_mac_filter(),
        discretization_id=space.prepared_id,
        regime="three-dimensional-periodic-unit-density",
    )
    with pytest.raises(ValueError, match="sharp Fourier projection"):
        PeriodicLearnedStressPlan(wrong_filter)

    projector = phx.discretization.PeriodicLerayProjector(space)
    wrong_layout = _binding(
        sample_shape=(4, 4, 4),
        dtype=jnp.dtype(space.plan.precision.physical_dtype),
        resolved_filter=_periodic_filter(),
        discretization_id=space.prepared_id,
        regime="three-dimensional-periodic-unit-density",
    )
    with pytest.raises(ValueError, match="feature layout"):
        PeriodicLearnedStressPlan(wrong_layout).prepare(space, projector)

    wrong_provenance = _binding(
        sample_shape=space.physical_shape,
        dtype=jnp.dtype(space.plan.precision.physical_dtype),
        resolved_filter=_periodic_filter(),
        discretization_id="different-fourier-discretization",
        regime="three-dimensional-periodic-unit-density",
    )
    with pytest.raises(ValueError, match="provenance"):
        PeriodicLearnedStressPlan(wrong_provenance).prepare(space, projector)

    discretization, operators, momentum, projection = _mac_grid(nonuniform=True)
    binding = _binding(
        sample_shape=discretization.cell_shape,
        dtype=operators.pressure_space.dtype,
        resolved_filter=_mac_filter(),
        discretization_id=discretization.prepared_id,
        regime="incompressible-unit-density",
    )
    with pytest.raises(ValueError, match="periodic-uniform"):
        MACLearnedStressPlan(binding).prepare(momentum, projection)


def test_learned_adapters_remain_separate_from_static_algebraic_models():
    space, first = _periodic_prepared(artifact="learned-artifact-a")
    _, second = _periodic_prepared(artifact="learned-artifact-b")
    provenance = LESParameterProvenance(
        _periodic_filter(),
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    static_model = SmagorinskyLESPlan(0.17).prepare(provenance)

    assert first.prepared_id != second.prepared_id
    assert first.binding.predictor is _positive_viscosity_predictor
    assert not isinstance(first.binding, type(static_model))
    with pytest.raises(TypeError, match="PreparedLearnedStressBinding"):
        PeriodicLearnedStressPlan(static_model)

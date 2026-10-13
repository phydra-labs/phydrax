from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.conditions import (
    ArrayCodomain,
    bind_condition,
    Condition,
    Equality,
    FieldCodomain,
    FieldSpec,
    MatrixLinearFunctional,
    PointJetAction,
    ProductFieldSpec,
)
from phydrax.domain import Interval1d
from phydrax.enforcement import (
    AffineProjectionPolicy,
    CanonicalKernelCorrectionProvider,
    compile as compile_enforcement,
    ConditionEvaluationContext,
    ConstraintLinearCorrectionProvider,
    EnforcementSpec,
    ExactAffineProjector,
    MatrixFreeKernelCorrectionProvider,
    PointKernelRepresenter,
    prepare_affine_projector,
    RealizationStatus,
)
from phydrax.kernels import ProductFieldKernelMetric, SquaredExponentialKernel
from phydrax.linalg import (
    DenseLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    solve,
    TolerancePolicy,
)


def _condition(field_names: Any, matrices: Any, target: Any) -> Any:
    shape = tuple(jnp.asarray(target).shape)
    source = ProductFieldSpec(
        tuple(
            FieldSpec(name, ArrayCodomain.from_shape((2,), dtype="float64"))
            for name in field_names
        )
    )
    operator = MatrixLinearFunctional(
        field_names,
        tuple((2,) for _ in field_names),
        matrices,
        output_shape=shape,
    )
    return Condition(
        "finite-linear-test",
        source,
        operator,
        ArrayCodomain.from_shape(shape, dtype="float64"),
        Equality(jnp.asarray(target, dtype="float64")),
    )


def test_affine_projector_closure_scenario_1() -> None:
    condition = _condition(
        ("u",),
        (jnp.eye(2),),
        jnp.asarray([1.0, -2.0]),
    )
    fields = {"u": jnp.asarray([4.0, 3.0])}
    bound = bind_condition(condition, fields)
    prepared = prepare_affine_projector(
        (bound,),
        ConstraintLinearCorrectionProvider(),
        correction_fields=("u",),
    )
    projected = prepared.apply(fields)
    assert jnp.allclose(projected["u"], jnp.asarray([1.0, -2.0]))
    repeated = prepared.apply(projected)
    assert jnp.allclose(repeated["u"], projected["u"])
    exact = ExactAffineProjector(prepared)
    context = ConditionEvaluationContext(condition)
    result = exact.realize(fields, context=context)
    assert result.successful
    assert result.evidence.verified

    spec = EnforcementSpec(condition, realization=exact)
    program = compile_enforcement(fields, (spec,))
    program_projected = program.apply(fields)
    assert jnp.allclose(program_projected["u"], jnp.asarray([1.0, -2.0]))
    condition = _condition(
        ("u", "v"),
        (jnp.eye(2), jnp.eye(2)),
        jnp.asarray([2.0, 4.0]),
    )
    fields = {
        "u": jnp.asarray([5.0, -1.0]),
        "v": jnp.asarray([0.0, 0.0]),
    }
    bound = bind_condition(condition, fields)
    prepared = prepare_affine_projector(
        (bound,),
        ConstraintLinearCorrectionProvider(),
        correction_fields=("u", "v"),
        policy=AffineProjectionPolicy(compatibility="strict"),
    )
    projected = prepared.apply(fields)
    assert jnp.allclose(projected["u"] + projected["v"], jnp.asarray([2.0, 4.0]))
    assert prepared.correction.evidence.identity_defect < 1e-6
    condition = _condition(
        ("u",),
        (jnp.eye(2),),
        jnp.asarray([1.0, -2.0]),
    )
    fields = {"u": jnp.asarray([4.0, 3.0])}
    prepared = prepare_affine_projector(
        (bind_condition(condition, fields),),
        ConstraintLinearCorrectionProvider(),
        correction_fields=("u",),
        policy=AffineProjectionPolicy(compatibility="generalized"),
    )

    approximate = prepared.realize(
        fields,
        context=ConditionEvaluationContext(condition, exact_required=False),
    )
    required = prepared.realize(
        fields,
        context=ConditionEvaluationContext(condition, exact_required=True),
    )

    assert approximate.successful
    # ty: ignore[unresolved-attribute]
    assert not approximate.stamp.exact
    assert required.status is RealizationStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="certifies exact semantics"):
        ExactAffineProjector(prepared)


def test_matrix_free_kernel_projection_retains_its_prepared_solve() -> None:
    domain = Interval1d(-1.0, 1.0)
    support = domain.component()
    field_codomain = FieldCodomain(support)
    field_spec = ProductFieldSpec((FieldSpec("u", field_codomain),))
    points = support.points({"x": jnp.asarray([[0.0]])})
    condition = Condition(
        "matrix-free-kernel-test",
        field_spec,
        PointJetAction("u", points, jnp.ones((1, 1))),
        ArrayCodomain.from_shape((1,), dtype="float64"),
        Equality(jnp.asarray([1.0])),
    )
    field = domain.Function("x")(lambda x: jnp.asarray(0.0))
    metric = ProductFieldKernelMetric.independent(
        field_spec,
        {"u": SquaredExponentialKernel()},
    )
    prepared = prepare_affine_projector(
        (bind_condition(condition, {"u": field}),),
        MatrixFreeKernelCorrectionProvider(
            metric,
            PointKernelRepresenter("u", jnp.asarray([[0.0]])),
        ),
        correction_fields=("u",),
        policy=AffineProjectionPolicy(
            compatibility="generalized",
            exactness_scope="realization",
        ),
    )

    projected = prepared.apply({"u": field})

    assert jnp.allclose(projected["u"](points).data, jnp.asarray([1.0]))
    assert not prepared.correction.evidence.exact


@pytest.mark.parametrize(
    "count", [3, 35], ids=["single-probe-block", "partial-probe-block"]
)
def test_matrix_free_kernel_correction_dense_parity(count: int) -> None:
    domain = Interval1d(-1.0, 1.0)
    support = domain.component()
    spec = ProductFieldSpec((FieldSpec("u", FieldCodomain(support)),))
    coordinates = jnp.linspace(-0.9, 0.9, count, dtype=jnp.float64)[:, None]
    points = support.points({"x": coordinates})
    condition = Condition(
        "blocked-kernel-parity",
        spec,
        PointJetAction("u", points, jnp.eye(count, dtype=jnp.float64)),
        ArrayCodomain.from_shape((count,), dtype="float64"),
        Equality(jnp.sin(coordinates[:, 0])),
    )
    field = domain.Function("x")(lambda x: jnp.asarray(0.0, dtype=jnp.float64))
    fields = {"u": field}
    metric = ProductFieldKernelMetric.independent(
        spec, {"u": SquaredExponentialKernel(length_scale=0.025)}
    )
    functional = PointKernelRepresenter("u", coordinates)
    solve_policy = LinearSolvePolicy(
        GMRES(restart=8),
        tolerance=TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=64),
    )
    provider = MatrixFreeKernelCorrectionProvider(
        metric, functional, solve_policy=solve_policy
    )
    approximate = prepare_affine_projector(
        (bind_condition(condition, fields),),
        provider,
        correction_fields=("u",),
        policy=AffineProjectionPolicy(
            compatibility="generalized", exactness_scope="realization"
        ),
    )
    canonical = prepare_affine_projector(
        (bind_condition(condition, fields),),
        CanonicalKernelCorrectionProvider(metric, functional),
        correction_fields=("u",),
    )
    queries = support.points(
        {"x": jnp.linspace(-0.95, 0.95, 11, dtype=jnp.float64)[:, None]}
    )
    np.testing.assert_allclose(
        approximate.apply(fields)["u"](queries).data,
        canonical.apply(fields)["u"](queries).data,
        rtol=2e-9,
        atol=2e-11,
    )
    gram = metric.functional_gram(provider.plan.functional).matrix
    dense_prepared = prepare(LinearSystem(DenseLinearOperator(gram)), solve_policy)
    identity = jnp.eye(count, dtype=jnp.float64)
    probe = solve(dense_prepared, identity)
    assert jnp.all(probe.successful)
    reference_defect = jnp.max(jnp.abs(gram @ jnp.asarray(probe.value) - identity))
    evidence = approximate.correction.evidence
    np.testing.assert_allclose(evidence.identity_defect, reference_defect, atol=2e-12)
    np.testing.assert_allclose(evidence.range_defect, reference_defect, atol=2e-12)
    assert evidence.generalized
    assert not evidence.exact
    assert evidence.solve_evidence.finite


def test_matrix_free_kernel_preparation_preserves_failed_solve() -> None:
    domain = Interval1d(-1.0, 1.0)
    support = domain.component()
    spec = ProductFieldSpec((FieldSpec("u", FieldCodomain(support)),))
    coordinates = jnp.asarray([[-0.7], [0.0], [0.7]], dtype=jnp.float64)
    points = support.points({"x": coordinates})
    condition = Condition(
        "failed-kernel-probes",
        spec,
        PointJetAction("u", points, jnp.eye(3, dtype=jnp.float64)),
        ArrayCodomain.from_shape((3,), dtype="float64"),
        Equality(jnp.zeros(3, dtype=jnp.float64)),
    )
    field = domain.Function("x")(lambda x: jnp.asarray(0.0, dtype=jnp.float64))
    metric = ProductFieldKernelMetric.independent(
        spec, {"u": SquaredExponentialKernel(length_scale=1.0)}
    )
    provider = MatrixFreeKernelCorrectionProvider(
        metric,
        PointKernelRepresenter("u", coordinates),
        solve_policy=LinearSolvePolicy(
            GMRES(restart=1),
            tolerance=TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=1),
        ),
    )
    with pytest.raises(RuntimeError, match="Gram preparation did not converge"):
        prepare_affine_projector(
            (bind_condition(condition, {"u": field}),),
            provider,
            correction_fields=("u",),
            policy=AffineProjectionPolicy(
                compatibility="generalized", exactness_scope="realization"
            ),
        )

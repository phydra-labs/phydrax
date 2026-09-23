import jax.numpy as jnp
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


def _condition(field_names, matrices, target):
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


def test_exact_affine_projector_is_idempotent_and_fixes_feasible_values():
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


def test_joint_projector_resolves_cyclic_coupled_fields_without_a_pivot():
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


def test_generalized_affine_projection_is_not_stamped_exact():
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
    assert not approximate.stamp.exact
    assert required.status is RealizationStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="certifies exact semantics"):
        ExactAffineProjector(prepared)


def test_matrix_free_kernel_projection_retains_its_prepared_solve():
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

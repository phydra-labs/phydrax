import jax.numpy as jnp

from phydrax.conditions import (
    ArrayCodomain,
    bind_condition,
    Condition,
    Equality,
    FieldSpec,
    MatrixLinearFunctional,
    ProductFieldSpec,
)
from phydrax.enforcement import (
    AffineProjectionPolicy,
    compile as compile_enforcement,
    ConditionEvaluationContext,
    ConstraintLinearCorrectionProvider,
    EnforcementSpec,
    ExactAffineProjector,
    prepare_affine_projector,
)


def _condition(field_names, matrices, target):
    shape = tuple(jnp.asarray(target).shape)
    source = ProductFieldSpec(
        tuple(
            FieldSpec(name, ArrayCodomain.from_shape((2,), dtype=float))
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
        ArrayCodomain.from_shape(shape, dtype=float),
        Equality(jnp.asarray(target, dtype=float)),
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from dataclasses import fields
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_definite_properties() -> Any:
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _coordinate_term(global_space: Any, local_space: Any, row: Any) -> Any:
    restriction = la.DenseLinearOperator(
        jnp.asarray(row, dtype=jnp.float64)[None, :],
        source=global_space,
        target=local_space,
    )
    prolongation = la.DenseLinearOperator(
        jnp.asarray(row, dtype=jnp.float64)[:, None],
        source=local_space,
        target=global_space,
    )
    return la.SubspaceCorrectionTerm(
        restriction,
        prolongation,
        la.DenseInversePreconditionerBuilder(),
    )


def test_linalg_preconditioning_scenario_1() -> None:
    space = la.ArraySpace((2,), dtype=jnp.float64)
    local = la.ArraySpace((1,), dtype=jnp.float64)
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    terms = (
        _coordinate_term(space, local, [1.0, 0.0]),
        _coordinate_term(space, local, [0.0, 1.0]),
    )
    materialization = la.MaterializationPolicy(max_entries=16, max_bytes=1024)
    additive = la.AdditiveSubspaceCorrectionBuilder(terms).prepare(
        operator,
        materialization=materialization,
    )
    forward = la.MultiplicativeSubspaceCorrectionBuilder(
        terms,
        sweep="forward",
    ).prepare(operator, materialization=materialization)
    backward = la.MultiplicativeSubspaceCorrectionBuilder(
        terms,
        sweep="backward",
    ).prepare(operator, materialization=materialization)
    residual = jnp.asarray([2.0, -1.0])

    additive_expected = residual / jnp.diag(matrix)
    forward_expected = jnp.asarray(
        [
            residual[0] / matrix[0, 0],
            (residual[1] - matrix[1, 0] * residual[0] / matrix[0, 0]) / matrix[1, 1],
        ]
    )
    backward_expected = jnp.asarray(
        [
            (residual[0] - matrix[0, 1] * residual[1] / matrix[1, 1]) / matrix[0, 0],
            residual[1] / matrix[1, 1],
        ]
    )

    assert jnp.allclose(additive.apply(residual), additive_expected)
    assert jnp.allclose(forward.apply(residual), forward_expected)
    assert jnp.allclose(backward.apply(residual), backward_expected)
    assert jnp.allclose(
        jax.jit(lambda value: additive.apply(value))(residual),
        additive_expected,
    )
    assert jnp.allclose(
        jax.jit(lambda value: forward.apply(value))(residual),
        forward_expected,
    )
    # ty: ignore[unresolved-attribute]
    cost = additive.cost_for(operator)
    assert cost.storage_bytes == 112
    assert cost.setup_matvec_count == 0
    accepted_plan = la.plan(
        la.LinearSystem(operator),
        la.LinearSolvePolicy(
            la.GMRES(restart=2),
            preconditioning=la.PreconditioningPolicy(additive),
            differentiation=la.DifferentiationPolicy("none"),
            resources=la.SolveResourcePolicy(preconditioner_bytes=112),
        ),
    )
    assert accepted_plan.preconditioner_plan is not None
    assert accepted_plan.preconditioner_plan.cost.storage_bytes == 112
    with pytest.raises(ValueError, match="requires 112 preconditioner state bytes"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.GMRES(restart=2),
                preconditioning=la.PreconditioningPolicy(additive),
                differentiation=la.DifferentiationPolicy("none"),
                resources=la.SolveResourcePolicy(preconditioner_bytes=111),
            ),
        )
    with pytest.raises(ValueError, match="at least one"):
        la.AdditiveSubspaceCorrectionBuilder(())
    diagonal = jnp.asarray([2.0, 3.0, 5.0])
    operator = la.DiagonalLinearOperator(
        diagonal,
        properties=_positive_definite_properties(),
    )
    builder = la.ChebyshevPreconditionerBuilder(
        8,
        interval=(2.0, 5.0),
    )
    action = builder.prepare(operator, materialization=la.MaterializationPolicy())
    residual = jnp.asarray([1.0, -2.0, 0.5])
    value = action.apply(residual)
    compiled = jax.jit(lambda value: action.apply(value))(residual)

    assert jnp.allclose(compiled, value)
    assert jnp.linalg.norm(operator.mv(value) - residual) < 1e-4
    # ty: ignore[unresolved-attribute]
    assert action.degree == 8
    assert builder.cost_for(operator).setup_matvec_count == 0
    with pytest.raises(ValueError, match="degree"):
        la.ChebyshevPreconditionerBuilder(0, interval=(1.0, 2.0))
    with pytest.raises(ValueError, match="interval"):
        la.ChebyshevPreconditionerBuilder(2, interval=(2.0, 1.0))


def _two_by_two_block_operator(
    coefficients: tuple[float, float, float, float] = (4.0, 1.0, 2.0, 3.0),
) -> Any:
    first = la.ArraySpace((1,), dtype=jnp.float64)
    second = la.ArraySpace((1,), dtype=jnp.float64)
    block_space = la.BlockSpace((first, second))
    blocks = (
        (
            la.DenseLinearOperator(
                jnp.asarray([[coefficients[0]]], dtype=jnp.float64),
                source=first,
                target=first,
            ),
            la.DenseLinearOperator(
                jnp.asarray([[coefficients[1]]], dtype=jnp.float64),
                source=second,
                target=first,
            ),
        ),
        (
            la.DenseLinearOperator(
                jnp.asarray([[coefficients[2]]], dtype=jnp.float64),
                source=first,
                target=second,
            ),
            la.DenseLinearOperator(
                jnp.asarray([[coefficients[3]]], dtype=jnp.float64),
                source=second,
                target=second,
            ),
        ),
    )
    return la.BlockLinearOperator(blocks, source=block_space, target=block_space)


def test_block_factorization_forms_match_block_algebra_and_require_typed_schur_action() -> (
    None
):
    operator = _two_by_two_block_operator()
    rhs = (jnp.asarray([2.0]), jnp.asarray([-1.0]))
    matrix = la.materialize(operator, la.MaterializationPolicy(max_entries=4))
    pivot = la.DenseInversePreconditionerBuilder()
    schur = la.DenseInversePreconditionerBuilder()
    actions = {
        form: la.BlockFactorizationPreconditionerBuilder(
            pivot,
            schur,
            form=form,
        ).prepare(operator, materialization=la.MaterializationPolicy(max_entries=16))
        for form in ("diagonal", "lower", "upper", "ldu")
    }
    pivot_value = 1.0 / matrix[0, 0]
    schur_value = matrix[1, 1] - matrix[1, 0] * pivot_value * matrix[0, 1]
    diagonal_expected = jnp.asarray([pivot_value * rhs[0][0], rhs[1][0] / schur_value])
    lower_expected = jnp.asarray(
        [
            pivot_value * rhs[0][0],
            (rhs[1][0] - matrix[1, 0] * pivot_value * rhs[0][0]) / schur_value,
        ]
    )
    upper_expected = jnp.asarray(
        [
            pivot_value * (rhs[0][0] - matrix[0, 1] * rhs[1][0] / schur_value),
            rhs[1][0] / schur_value,
        ]
    )
    exact = jnp.linalg.solve(matrix, jnp.asarray([rhs[0][0], rhs[1][0]]))

    assert jnp.allclose(
        operator.source.flatten(actions["diagonal"].apply(rhs)), diagonal_expected
    )
    assert jnp.allclose(
        operator.source.flatten(actions["lower"].apply(rhs)), lower_expected
    )
    assert jnp.allclose(
        operator.source.flatten(actions["upper"].apply(rhs)), upper_expected
    )
    assert jnp.allclose(operator.source.flatten(actions["ldu"].apply(rhs)), exact)
    assert jnp.allclose(
        operator.source.flatten(jax.jit(lambda value: actions["ldu"].apply(value))(rhs)),
        exact,
    )
    with pytest.raises(TypeError, match="AbstractPreconditioner"):
        la.SchurComplementLinearOperator(
            operator.blocks[1][1],
            operator.blocks[1][0],
            operator.blocks[0][1],
            # ty: ignore[invalid-argument-type]
            lambda value: value,
        )


def _weighted_scalar_action(weights: jax.Array, value: jax.Array) -> jax.Array:
    return jnp.sum(weights) * value


@pytest.mark.parametrize(
    ("form", "retained_payload_indices"),
    [
        ("diagonal", ()),
        ("lower", (1,)),
        ("upper", (0,)),
        ("ldu", (0, 1)),
    ],
    ids=["diagonal", "lower", "upper", "ldu"],
)
def test_block_action_callback_storage_is_admitted_and_setup_peak_remains_charged(
    form: la.BlockFactorizationForm,
    retained_payload_indices: tuple[int, ...],
) -> None:
    scalar = la.ArraySpace((1,), dtype=jnp.float64)
    block_space = la.BlockSpace((scalar, scalar))

    def callback(weights: jax.Array, index: int) -> la.FunctionLinearOperator:
        function = jax.tree_util.Partial(_weighted_scalar_action, weights)
        return la.FunctionLinearOperator(
            function,
            transpose_action=function,
            source=scalar,
            target=scalar,
            operator_id=f"declared-coupling-{index}",
        )

    payloads = tuple(
        jnp.full((32,), coefficient / 32, dtype=jnp.float64)
        for coefficient in (1.0, 2.0, 3.0)
    )
    callbacks = tuple(callback(weights, index) for index, weights in enumerate(payloads))
    upper, lower, diagonal = callbacks
    pivot = la.DenseLinearOperator(
        jnp.asarray([[4.0]], dtype=jnp.float64),
        source=scalar,
        target=scalar,
    )
    operator = la.BlockLinearOperator(
        ((pivot, upper), (lower, diagonal)),
        source=block_space,
        target=block_space,
    )
    local = la.DenseInversePreconditionerBuilder()
    builder = la.BlockFactorizationPreconditionerBuilder(local, local, form)
    materialization = la.MaterializationPolicy(max_entries=16)
    action = builder.prepare(operator, materialization=materialization)
    planned = builder.cost_for(operator, materialization=materialization)
    retained = action.cost_for(operator, materialization=materialization)
    schur = la.SchurComplementLinearOperator(diagonal, lower, action.pivot_action, upper)
    pivot_cost = local.cost_for(pivot, materialization=materialization)
    schur_cost = local.cost_for(schur, materialization=materialization)
    # Resident host ownership counts each declared callback payload once; compiled
    # transport may have different aliases and is measured independently.
    expected_storage = (
        pivot_cost.storage_bytes
        + schur_cost.storage_bytes
        + sum(payloads[index].nbytes for index in retained_payload_indices)
    )
    assert retained.storage_bytes == planned.storage_bytes == expected_storage
    expected_peak = (
        pivot_cost.storage_bytes
        + schur_cost.storage_bytes
        + pivot_cost.preparation_workspace_bytes
        + schur_cost.preparation_workspace_bytes
        + sum(payload.nbytes for payload in payloads)
    )
    assert planned.storage_bytes + planned.preparation_workspace_bytes == expected_peak
    assert not bool(action.preparation_failed)
    accepted = la.plan(
        la.LinearSystem(operator),
        la.LinearSolvePolicy(
            la.GMRES(restart=2),
            preconditioning=la.PreconditioningPolicy(action),
            resources=la.SolveResourcePolicy(preconditioner_bytes=expected_storage),
        ),
    )
    assert accepted.preconditioner_plan is not None
    assert accepted.preconditioner_plan.cost.storage_bytes == expected_storage
    with pytest.raises(ValueError, match="preconditioner state bytes"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.GMRES(restart=2),
                preconditioning=la.PreconditioningPolicy(action),
                resources=la.SolveResourcePolicy(
                    preconditioner_bytes=expected_storage - 1
                ),
            ),
        )


@pytest.mark.parametrize("form", ["diagonal", "lower", "upper", "ldu"])
def test_block_action_refresh_matches_fresh_factors_and_preserves_form(
    form: la.BlockFactorizationForm,
) -> None:
    builder = la.BlockFactorizationPreconditionerBuilder(
        la.DenseInversePreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        form,
    )
    materialization = la.MaterializationPolicy(max_entries=16)
    original = builder.prepare(
        _two_by_two_block_operator(), materialization=materialization
    )
    changed = _two_by_two_block_operator((6.0, 1.0, 2.0, 4.0))
    refreshed = builder.refresh(original, changed, materialization=materialization)
    fresh = builder.prepare(changed, materialization=materialization)
    rhs = (jnp.asarray([2.0], dtype=jnp.float64), jnp.asarray([-1.0], dtype=jnp.float64))
    assert jnp.array_equal(
        changed.source.flatten(refreshed.apply(rhs)),
        changed.source.flatten(fresh.apply(rhs)),
    )
    assert (
        refreshed.cost_for(changed).storage_bytes == fresh.cost_for(changed).storage_bytes
    )
    with pytest.raises(ValueError, match="preserve its form"):
        la.BlockFactorizationPreconditionerBuilder(
            la.DenseInversePreconditionerBuilder(),
            la.DenseInversePreconditionerBuilder(),
            "lower" if form == "diagonal" else "diagonal",
        ).refresh(original, changed, materialization=materialization)


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("diagonal", (0.0, 0.0)),
        ("lower", (0.0, -0.2)),
        ("upper", (0.1, 0.0)),
        ("ldu", (0.2, 0.0)),
    ],
)
def test_block_action_execution_couplings_remain_dynamic(
    form: la.BlockFactorizationForm,
    expected: tuple[float, float],
) -> None:
    operator = _two_by_two_block_operator()
    seed = la.BlockFactorizationPreconditionerBuilder(
        la.DenseInversePreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        form,
    ).prepare(operator, materialization=la.MaterializationPolicy(max_entries=16))
    first, second = operator.source.spaces
    rhs = (jnp.asarray([2.0], dtype=jnp.float64), jnp.asarray([-1.0], dtype=jnp.float64))

    def apply(coefficient: jax.Array) -> jax.Array:
        lower = la.DenseLinearOperator(
            coefficient.reshape((1, 1))
            if form == "lower"
            else jnp.asarray([[2.0]], dtype=jnp.float64),
            source=first,
            target=second,
            operator_id="declared-lower-coupling",
        )
        upper = la.DenseLinearOperator(
            jnp.asarray([[1.0]], dtype=jnp.float64)
            if form == "lower"
            else coefficient.reshape((1, 1)),
            source=second,
            target=first,
            operator_id="declared-upper-coupling",
        )
        schur = la.SchurComplementLinearOperator(
            operator.blocks[1][1],
            lower,
            seed.pivot_action,
            upper,
            operator_id="declared-execution-schur",
        )
        action = la.BlockFactorizationPreconditioner(
            schur,
            seed.schur_action,
            form,
            properties=seed.properties,
            preconditioner_id="declared-execution-action",
        )
        return operator.source.flatten(action.apply(rhs))

    _, derivative = jax.jvp(
        apply,
        (jnp.asarray(2.0 if form == "lower" else 1.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    assert jnp.allclose(
        derivative, jnp.asarray(expected, dtype=jnp.float64), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "coefficients",
    [(0.0, 0.0, 0.0, 3.0), (4.0, 1.0, 2.0, 0.5)],
    ids=["singular-pivot", "singular-schur"],
)
def test_block_action_propagates_native_factor_failure_after_setup_is_discarded(
    coefficients: tuple[float, float, float, float],
) -> None:
    local = la.BoundedLUPreconditionerBuilder(
        max_size=1,
        probe_chunk_size=1,
        max_workspace_bytes=65536,
        max_output_bytes=65536,
    )
    action = la.BlockFactorizationPreconditionerBuilder(local, local, "diagonal").prepare(
        _two_by_two_block_operator(coefficients),
        materialization=la.MaterializationPolicy(max_entries=16),
    )
    assert bool(action.preparation_failed)
    assert bool(
        action.pivot_action.preparation_failed | action.schur_action.preparation_failed
    )


def test_block_action_registered_decoder_refuses_obsolete_prepared_fieldset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._model._artifacts as artifact_registry
    from phydrax._model._artifacts import register_artifact_value
    from phydrax._model._structure import model_from_array_recipe

    monkeypatch.setattr(artifact_registry, "_ARTIFACT_VALUES_BY_ID", {})
    monkeypatch.setattr(artifact_registry, "_ARTIFACT_IDS_BY_VALUE", {})

    identity = "test.linalg:block-factorization-action"
    register_artifact_value(identity, la.BlockFactorizationPreconditioner)
    obsolete_fields = {
        field.name: {"kind": "none"}
        for field in fields(la.BlockFactorizationPreconditioner)
    }
    for name in (
        "pivot_action",
        "lower_block",
        "upper_block",
        "schur_operator_id",
        "schur_input_ids",
    ):
        obsolete_fields.pop(name)
    obsolete_fields["schur_operator"] = {"kind": "none"}
    # A legacy field inventory is refused before restoring any arrays or module.
    recipe = {
        "kind": "dataclass",
        "type": identity,
        "items": list(obsolete_fields.values()),
    }
    with pytest.raises(ValueError, match=r"\.items"):
        model_from_array_recipe(recipe, {}, prefix="model")

from typing import Any

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.linalg import MaterializationPolicy, materialize
from phydrax.solver._functional_run import partition_functional_parameters
from phydrax.solver._functional_surrogate import prepare_functional_update


def test_neural_tangent_scenario_1() -> None:
    design = jnp.asarray([[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]])
    parameters = jnp.asarray([0.3, -0.7])
    prepared = phx.nn.neural_tangent.prepare_empirical_ntk(
        lambda value: design @ value,
        parameters,
    )
    matrix = materialize(
        prepared.kernel,
        MaterializationPolicy(max_entries=100, max_bytes=4096),
    )
    expected = design @ design.T

    assert jnp.allclose(matrix, expected)
    tangent = jnp.asarray([0.4, -0.2])
    cotangent = jnp.asarray([1.0, -2.0, 0.5])
    assert jnp.allclose(prepared.jvp(tangent), design @ tangent)
    assert jnp.allclose(prepared.vjp(cotangent), design.T @ cotangent)
    assert jnp.allclose(prepared.kernel.mv(cotangent), expected @ cotangent)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(prepared.parameter_gram.mv(tangent), design.T @ design @ tangent)
    design = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    prepared = phx.nn.neural_tangent.prepare_empirical_ntk(
        lambda value: design @ value,
        jnp.asarray([0.0, 0.0]),
    )
    diagnostics = phx.nn.neural_tangent.analyze_ntk(
        prepared,
        policy=phx.nn.neural_tangent.NTKDiagnosticsPolicy(
            dense_max_dimension=8,
            eigenvalue_count=3,
        ),
    )
    expected = jnp.linalg.eigvalsh(design @ design.T)[::-1]

    assert diagnostics.dense
    assert jnp.allclose(diagnostics.leading_eigenvalues, expected)
    assert diagnostics.numerical_rank == 2
    assert diagnostics.nullity == 1
    assert jnp.allclose(diagnostics.trace, jnp.sum(expected))
    assert bool(diagnostics.finite)
    design = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))
    prepared = phx.nn.neural_tangent.prepare_empirical_ntk(
        lambda value: design @ value,
        jnp.zeros((3,)),
    )
    diagnostics = phx.nn.neural_tangent.analyze_ntk(
        prepared,
        policy=phx.nn.neural_tangent.NTKDiagnosticsPolicy(
            dense_max_dimension=1,
            num_probes=4,
            eigenvalue_count=1,
            max_krylov_steps=3,
        ),
        key=jr.key(10),
    )

    assert not diagnostics.dense
    assert jnp.allclose(diagnostics.diagonal, jnp.asarray([1.0, 4.0, 9.0]))
    assert jnp.allclose(diagnostics.trace, 14.0)
    assert jnp.allclose(diagnostics.trace_square, 98.0)
    assert bool(diagnostics.finite)


def test_neural_tangent_scenario_2() -> None:
    first = jnp.asarray([[1.0, 2.0], [0.5, -1.0]])
    second = jnp.asarray([[3.0, 0.25]])
    point = jnp.asarray([0.2, -0.4])
    left = phx.nn.neural_tangent.prepare_empirical_ntk(lambda value: first @ value, point)
    right = phx.nn.neural_tangent.prepare_empirical_ntk(
        lambda value: second @ value, point
    )
    cross = left.cross_kernel(right)

    assert jnp.allclose(
        cross.mv(jnp.asarray([2.0])), first @ second.T @ jnp.asarray([2.0])
    )
    assert jnp.allclose(
        cross.adjoint_mv(jnp.asarray([1.0, -0.5])),
        second @ first.T @ jnp.asarray([1.0, -0.5]),
    )
    prepared = phx.solver.prepare_functional_ntk(_functional_solver(), key=jr.key(1))
    full = materialize(
        prepared.kernel,
        MaterializationPolicy(max_entries=100, max_bytes=4096),
    )
    first = prepared.block(phx.terms.ResidualBlockRef(0, "a"))
    second = prepared.block(phx.terms.ResidualBlockRef(0, "b"))

    assert full.shape == (8, 8)
    assert first.output_space.size == second.output_space.size == 4
    assert jnp.allclose(jnp.trace(full), 2.0)
    assert prepared.layout.logical_blocks == ((0, "a"), (0, "b"))
    solver = _functional_solver()
    params, non_trainable = partition_functional_parameters(solver.functions)
    physical = solver.objective.prepare_training(
        (0,),
        scale=1.0,
        evaluation_key=jr.key(2),
        sampling_key=jr.key(3),
        iteration=1,
    )
    training = phx.solver.FunctionalTrainingPlan(
        pseudo_transient=(
            phx.solver.PseudoTransientPolicy(
                0,
                phx.solver.ResidualRelaxationMap(
                    "u",
                    lambda value: value,
                    operator_semantic_id="identity-map",
                    operator_numeric_id="identity-map",
                ),
                freshness="experimental_fixed",
            ),
        )
    )
    update = prepare_functional_update(
        physical,
        params,
        non_trainable,
        solver.enforcement,
        training=training,
        previous_functions=solver.functions,
        pseudo_inverse_steps=(1.0,),
    )
    physical_ntk = phx.solver.prepare_functional_ntk(
        solver,
        prepared_update=update,
        view="physical",
    )
    surrogate_ntk = phx.solver.prepare_functional_ntk(
        solver,
        prepared_update=update,
        view="surrogate",
    )
    policy = MaterializationPolicy(max_entries=100, max_bytes=4096)
    physical_matrix = materialize(physical_ntk.kernel, policy)
    surrogate_matrix = materialize(surrogate_ntk.kernel, policy)

    assert jnp.allclose(surrogate_matrix, 4.0 * physical_matrix)


def _functional_solver() -> Any:
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray([1.0, -1.0]))
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda value: value)
    batch = component.sample(
        phx.domain.PointSampling(4, layout=phx.domain.SampleLayout((("x",),))),
        key=jr.key(0),
    )
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    term = phx.terms.ResidualPenalty(
        condition,
        source,
        blocks=phx.terms.ResidualBlockLayout(("a", "b")),
    )
    return phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))

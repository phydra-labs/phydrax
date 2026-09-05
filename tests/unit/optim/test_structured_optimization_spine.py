import sys

import jax.numpy as jnp
import pytest

import phydrax as phx


opt = phx.optim


def _problem():
    return opt.MinimizationProblem(
        lambda value, target: jnp.sum((value - target) ** 2),
        bounds=opt.Bounds(0.0, 1.0),
        constraints=(
            opt.NonlinearConstraint(
                lambda value, _: jnp.asarray([jnp.sum(value)]),
                lower=1.0,
                upper=1.0,
                constraint_id="sum-one",
            ),
        ),
        problem_id="structured-spine-quadratic",
    )


def _termination():
    return opt.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=80,
    )


def _compilation(target=None):
    target_ = jnp.asarray([0.25, 0.75]) if target is None else target
    return opt.compile_structured_minimization(
        _problem(),
        jnp.asarray([0.5, 0.5]),
        sample_args=target_,
    )


def test_structured_template_refresh_preserves_topology_and_changes_binding():
    compilation = _compilation()
    prepared = compilation.prepared
    refreshed = opt.refresh_structured_nonlinear(
        prepared,
        jnp.asarray([0.4, 0.6]),
    )
    assert refreshed.template.template_id == prepared.template.template_id
    assert refreshed.numeric_binding_id != prepared.numeric_binding_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1

    with pytest.raises(ValueError, match="bound roles changed"):
        opt.refresh_structured_nonlinear(
            prepared,
            constraint_lower=jnp.asarray([-jnp.inf]),
        )


def test_structured_dense_method_returns_portable_warm_start():
    compilation = _compilation()
    method = opt.PrimalDualInteriorPoint(
        mode="dense-filter",
        max_dense_dimension=32,
    )
    result = opt.solve_structured_minimization(
        compilation,
        method=method,
        termination=_termination(),
    )
    assert bool(result.successful)
    assert result.structured.warm_start.structure_id == compilation.program.structure_id
    assert jnp.allclose(
        result.optimization.parameters,
        jnp.asarray([0.25, 0.75]),
        atol=3e-4,
    )
    assert result.optimization.certificate is not None


def test_sparse_augmented_method_uses_exact_structured_derivatives():
    compilation = _compilation()
    method = opt.PrimalDualInteriorPoint(mode="sparse-augmented")
    result = opt.solve_structured_minimization(
        compilation,
        method=method,
        termination=_termination(),
    )
    assert bool(result.successful)
    assert method.structured_capabilities.exact_sparse_jacobian
    assert method.structured_capabilities.exact_sparse_hessian
    assert jnp.allclose(
        result.optimization.parameters,
        jnp.asarray([0.25, 0.75]),
        atol=2e-3,
    )


def test_structured_pool_is_input_ordered_and_exactly_once():
    compilation = _compilation()
    method = opt.PrimalDualInteriorPoint(mode="sparse-augmented")
    initial = jnp.asarray(
        [
            [0.5, 0.5],
            [0.1, 0.9],
            [0.8, 0.2],
        ]
    )
    pooled = opt.solve_pooled_structured_nonlinear(
        compilation.prepared,
        initial,
        method=method,
        termination=_termination(),
        lane_count=2,
    )
    assert len(pooled.results) == 3
    assert sorted(pooled.evidence.completion_order.tolist()) == [0, 1, 2]
    assert int(pooled.evidence.refills) == 1
    assert all(bool(result.successful) for result in pooled.results)


def test_kkt_plan_reports_executed_dense_form_and_reuses_factorization():
    plan = opt.plan_kkt(2, 1)
    assert plan.form == "dense-augmented"
    factor = opt.factor_kkt(
        jnp.diag(jnp.asarray([2.0, 4.0])),
        jnp.asarray([[1.0, 1.0]]),
        plan,
    )
    first = opt.solve_factored_kkt(
        factor,
        jnp.asarray([-2.0, -4.0]),
        jnp.asarray([0.0]),
    )
    second = opt.solve_factored_kkt(
        factor,
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([0.5]),
    )
    assert bool(first.finite & first.inertia_matches)
    assert float(second.residual_norm) <= 1e-10


def test_spineax_provider_is_explicit_and_reports_unreliable_zero_inertia():
    capabilities = phx.linalg.sparse_provider_capabilities("spineax-cudss")
    assert capabilities.factorization == "ldlt"
    assert capabilities.numeric_refactorization
    assert capabilities.inertia
    assert not capabilities.reliable_zero_inertia
    method = phx.linalg.SparseLDLT()
    assert method.provider == "spineax-cudss"
    availability = phx.backends.spineax_availability()
    assert availability.capabilities.backend == "spineax-cudss"
    if not availability.available:
        assert "spineax.cudss" not in sys.modules
    unsafe = opt.PrimalDualInteriorPoint(
        mode="sparse-augmented",
        linear_policy=phx.linalg.LinearSolvePolicy(method),
    )
    with pytest.raises(ValueError, match="zero-inertia"):
        opt.solve_structured_minimization(
            _compilation(),
            method=unsafe,
            termination=_termination(),
        )


def test_structured_sensitivity_and_continuation_use_certified_kkt_state():
    compilation = _compilation()
    result = opt.solve_structured_minimization(
        compilation,
        method=opt.PrimalDualInteriorPoint(
            mode="dense-filter",
            max_dense_dimension=32,
        ),
        termination=_termination(),
    )
    tangent = opt.structured_solution_jvp(
        compilation.prepared,
        result.structured,
        jnp.asarray([1.0, 0.0]),
    )
    assert bool(tangent.regular)
    assert jnp.allclose(tangent.value, jnp.asarray([0.5, -0.5]), atol=2e-4)

    seed = opt.structured_parameter_continuation(
        compilation.prepared,
        result.structured,
        lambda coordinate: jnp.asarray([0.25 + coordinate, 0.75 - coordinate]),
        parameter_lower=-0.2,
        parameter_upper=0.2,
    )
    assert jnp.linalg.norm(seed.problem.residual(seed.state, 0.0, None)) < 1e-5


def test_structured_state_design_recovers_all_at_once_kkt_solution():
    problem = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum((state - 1.0) ** 2 + design**2),
        problem_id="structured-state-design-quadratic",
    )
    compilation = opt.compile_structured_state_design(
        problem,
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
    )
    solved = opt.solve_structured_state_design(
        compilation,
        method=opt.PrimalDualInteriorPoint(mode="sparse-augmented"),
        termination=_termination(),
    )
    assert bool(solved.successful)
    assert jnp.allclose(solved.state, jnp.asarray([0.5]), atol=2e-3)
    assert jnp.allclose(solved.design, jnp.asarray([0.5]), atol=2e-3)


def test_structured_state_design_lowers_declared_vector_constraints():
    constraint = opt.StateDesignConstraint(
        lambda state, design, scale: jnp.stack((state[0] + design[0], scale * design[0])),
        lower=jnp.asarray((1.0, -jnp.inf)),
        upper=jnp.asarray((1.0, 2.0)),
        constraint_id="state-design-vector",
    )
    problem = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum(state**2 + design**2),
        constraints=(constraint,),
        problem_id="structured-state-design-constraints",
    )
    compilation = opt.compile_structured_state_design(
        problem,
        jnp.asarray((0.5,)),
        jnp.asarray((0.5,)),
        sample_args=jnp.asarray(2.0),
        exact_hessian=False,
    )
    program = compilation.optimization.program
    values = program.constraints(
        jnp.asarray((0.5, 0.5)),
        jnp.asarray(2.0),
    )

    assert program.num_constraints == 3
    assert jnp.allclose(values, jnp.asarray((0.0, 1.0, 1.0)))
    assert jnp.array_equal(
        program.constraint_lower,
        jnp.asarray((0.0, 1.0, -jnp.inf)),
    )
    assert jnp.array_equal(
        program.constraint_upper,
        jnp.asarray((0.0, 1.0, 2.0)),
    )
    assert program.equality_indices.tolist() == [0, 1]
    assert program.upper_indices.tolist() == [2]


@pytest.mark.parametrize(
    "lower,upper,point,multiplier,valid",
    [
        (0.5, 1.5, 0.5, -1.0, True),
        (0.5, 1.5, 1.5, 1.0, True),
        (0.5, float("inf"), 0.5, 1.0, False),
        (-float("inf"), 1.5, 1.5, -1.0, False),
    ],
)
def test_bound_form_certificate_splits_two_sided_net_duals_but_rejects_one_sided_wrong_signs(
    lower, upper, point, multiplier, valid
):
    coordinates = jnp.asarray([point])
    constraints = lambda value, _: value
    space = phx.linalg.ArraySpace((1,), dtype=coordinates.dtype)
    pattern = phx.sparse.SparsePattern.from_coo([0], [0], (1, 1))
    jacobian = phx.sparse.compile_sparse_jacobian(
        constraints,
        coordinates,
        source=space,
        target=space,
        structure=pattern,
        compiler="native",
    )
    program = opt.StructuredNonlinearProgram(
        lambda value, _: -multiplier * value[0],
        constraints,
        jacobian,
        variable_lower=[-jnp.inf],
        variable_upper=[jnp.inf],
        constraint_lower=[lower],
        constraint_upper=[upper],
        constraint_sources=("physical-range",),
        program_id="bound-form-dual-certificate",
        structure_id="scalar-range",
    )
    certificate = program.certificate(
        coordinates,
        jnp.asarray([multiplier]),
        jnp.zeros(1),
        jnp.zeros(1),
        active_tolerance=1e-8,
    )
    assert jnp.allclose(certificate.stationarity_residual, 0.0)
    assert certificate.primal_feasibility == 0.0
    if valid:
        assert certificate.dual_feasibility == 0.0
        assert certificate.complementarity == 0.0
    else:
        assert certificate.dual_feasibility > 0.0

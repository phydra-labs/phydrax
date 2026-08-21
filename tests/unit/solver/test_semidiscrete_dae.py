import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _spatial(points=8):
    axis = phx.domain.FourierAxisSpec(points).materialize(0.0, 1.0)
    return axis, phx.solver.TensorGridDiscretization((axis,))


def _mixed_problem():
    coordinate = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    time = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    fields = (
        phx.equations.PDEField("u", coordinates=("x", "t")),
        phx.equations.PDEField("p", coordinates=("x", "t")),
    )
    u = phx.equations.PDEExpression.field("u")
    p = phx.equations.PDEExpression.field("p")
    return phx.equations.PDEProblemIR(
        coordinates=(coordinate, time),
        fields=fields,
        parameters=(phx.equations.PDEParameter("kappa", value=0.1),),
        equations=(
            phx.equations.PDEEquation(
                "diffusion",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("kappa") * u.laplacian("x"),
            ),
            phx.equations.PDEEquation("constraint", p, u),
        ),
    )


def _compile_mixed(*, points=8):
    axis, spatial = _spatial(points)
    compiled = phx.equations.compile_semidiscrete_dae(
        _mixed_problem(),
        spatial,
        equation_targets={"diffusion": "u", "constraint": "p"},
        state_scale=2.0,
        state_rate_scale=3.0,
        residual_scale=0.5,
    )
    return axis, spatial, compiled


def test_semidiscrete_dae_compiles_aligned_residual_and_honest_structure():
    axis, spatial, compiled = _compile_mixed()
    u = jnp.sin(2.0 * jnp.pi * axis.nodes)
    state = compiled.layout.pack({"u": u, "p": u})
    rate = compiled.layout.pack({"u": 0.1 * spatial.laplacian(u), "p": jnp.zeros_like(u)})

    residual = compiled(0.0, state, rate, None)
    rate_jacobian = compiled.rate_jacobian(0.0, state, rate)

    assert compiled.state_shape == (axis.nodes.size, 2)
    assert compiled.structure.variable_roles == ("differential", "algebraic")
    assert compiled.structural_report.equation_targets == (
        ("diffusion", "u"),
        ("constraint", "p"),
    )
    assert compiled.structural_report.temporal_derivative_counts == (1, 0)
    assert not compiled.structural_report.regularity_verified
    assert (
        compiled.structural_report.index_assumption
        == "regular-index-1-required-unverified"
    )
    assert jnp.max(jnp.abs(residual)) < 1e-11
    assert rate_jacobian.shape == (2 * axis.nodes.size,) * 2
    assert jnp.array_equal(
        compiled.system.state_rate_scale,
        jnp.full(compiled.state_shape, 3.0),
    )


def test_semidiscrete_dae_requires_bijective_targets_and_direct_time_rates():
    _, spatial = _spatial()
    problem = _mixed_problem()

    with pytest.raises(ValueError, match="every PDE equation"):
        phx.equations.compile_semidiscrete_dae(
            problem,
            spatial,
            equation_targets={"diffusion": "u"},
        )
    with pytest.raises(ValueError, match="bijectively"):
        phx.equations.compile_semidiscrete_dae(
            problem,
            spatial,
            equation_targets={"diffusion": "u", "constraint": "u"},
        )

    u = phx.equations.PDEExpression.field("u")
    invalid = phx.equations.PDEProblemIR(
        coordinates=problem.coordinates,
        fields=problem.fields,
        parameters=problem.parameters,
        equations=(
            phx.equations.PDEEquation(
                "diffusion",
                u.derivative("t", order=2),
                u.laplacian("x"),
            ),
            problem.equations[1],
        ),
    )
    with pytest.raises(ValueError, match="unsupported temporal derivative"):
        phx.equations.compile_semidiscrete_dae(
            invalid,
            spatial,
            equation_targets={"diffusion": "u", "constraint": "p"},
        )


def test_explicit_and_implicit_compilers_agree_for_eliminable_heat_equation():
    base = _mixed_problem()
    heat = phx.equations.PDEProblemIR(
        coordinates=base.coordinates,
        fields=(base.fields[0],),
        parameters=base.parameters,
        equations=(base.equations[0],),
    )
    axis, spatial = _spatial()
    explicit = phx.equations.compile_semidiscrete_pde(heat, spatial, method="direct")
    implicit = phx.equations.compile_semidiscrete_dae(
        heat,
        spatial,
        equation_targets={"diffusion": "u"},
    )
    state = jnp.sin(2.0 * jnp.pi * axis.nodes)
    rate = explicit(0.0, state, None)

    assert implicit.structure.component_axis is None
    assert jnp.max(jnp.abs(implicit(0.0, state, rate, None))) < 1e-11


def test_compiled_dae_solve_is_jittable_and_parameter_differentiable():
    axis, spatial, compiled = _compile_mixed(points=6)
    initial_u = jnp.sin(2.0 * jnp.pi * axis.nodes)
    initial_state = compiled.layout.pack({"u": initial_u, "p": jnp.zeros_like(initial_u)})
    problem = phx.solver.DifferentialAlgebraicProblem(
        compiled.system,
        initial_state,
        problem_id="compiled-semipde",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.01, 3),
        time_id="compiled-semipde",
    )
    prepared = phx.solver.prepare_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(integration_method="bdf1"),
    )

    def terminal_amplitude(kappa):
        solution = phx.solver.solve_dae(prepared, args={"kappa": kappa})
        terminal = compiled.layout.field(solution.states[-1], "u")
        return jnp.vdot(initial_u, terminal) / jnp.vdot(initial_u, initial_u)

    kappa = jnp.asarray(0.1)
    value, gradient = jax.jit(jax.value_and_grad(terminal_amplitude))(kappa)
    eigenvalue = jnp.vdot(initial_u, spatial.laplacian(initial_u)) / jnp.vdot(
        initial_u, initial_u
    )
    step = grid.durations[0]
    denominator = 1.0 - step * kappa * eigenvalue
    expected = denominator**-grid.num_steps
    expected_gradient = (
        grid.num_steps * step * eigenvalue * denominator ** (-grid.num_steps - 1)
    )

    assert jnp.allclose(value, expected, rtol=1e-7, atol=1e-9)
    assert jnp.allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-8)


def test_dae_trajectory_adapter_retains_rates_validity_and_provenance():
    axis, _, compiled = _compile_mixed(points=6)
    initial_u = jnp.sin(2.0 * jnp.pi * axis.nodes)
    problem = phx.solver.DifferentialAlgebraicProblem(
        compiled.system,
        compiled.layout.pack({"u": initial_u, "p": jnp.zeros_like(initial_u)}),
        problem_id="semipde-adapter",
    )
    solution = phx.solver.solve_dae(
        problem,
        phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.01, 3), time_id="adapter"),
        policy=phx.solver.DAESolvePolicy(integration_method="bdf1"),
    )

    data = phx.dynamics.identification.trajectory_data_from_differential_solution(
        solution
    )

    assert jnp.array_equal(data.derivatives, solution.state_rates)
    assert not data.derivative_valid[0]
    assert jnp.all(data.derivative_valid[1:])
    assert jnp.array_equal(data.sample_valid, solution.valid)
    assert jnp.array_equal(
        data.transition_valid,
        solution.valid[:-1] & solution.valid[1:],
    )
    assert data.coordinate_id == solution.time_id
    assert data.source_id.startswith("dae:")

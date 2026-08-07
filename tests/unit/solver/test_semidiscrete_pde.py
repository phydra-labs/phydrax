import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _heat_problem(*, periodic=False, target=None, reaction=False):
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=periodic,
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    kappa = phx.equations.PDEParameter("kappa", value=0.2)
    u = phx.equations.PDEExpression.field("u")
    rhs = phx.equations.PDEExpression.parameter("kappa") * u.laplacian("x")
    if reaction:
        rhs = rhs + u * (1.0 - u)
    conditions = ()
    regions = ()
    if target is not None:
        regions = (phx.equations.PDERegion("boundary", "boundary", ("x",)),)
        conditions = (
            phx.equations.PDECondition(
                "dirichlet",
                "boundary",
                u,
                target=phx.equations.PDEExpression.constant(target),
                region="boundary",
                coordinate="x",
            ),
        )
    return phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(kappa,),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                rhs,
            ),
        ),
        conditions=conditions,
        regions=regions,
    )


def test_spatial_calculus_preserves_trailing_components_and_basis_semantics():
    x_axis = phx.domain.FourierAxisSpec(24).materialize(0.0, 1.0)
    y_axis = phx.domain.FourierAxisSpec(20).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((x_axis, y_axis))
    x = x_axis.nodes[:, None]
    y = y_axis.nodes[None, :]
    scalar = jnp.sin(2.0 * jnp.pi * x) * jnp.cos(4.0 * jnp.pi * y)
    channels = jnp.stack((scalar, -2.0 * scalar), axis=-1)

    derivative = spatial.partial_derivative(channels, axis=0)
    expected = jnp.stack(
        (
            2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x) * jnp.cos(4.0 * jnp.pi * y),
            -4.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x) * jnp.cos(4.0 * jnp.pi * y),
        ),
        axis=-1,
    )
    assert derivative.shape == channels.shape
    assert jnp.allclose(derivative, expected, rtol=1e-9, atol=1e-9)

    gradient = spatial.gradient(channels)
    assert gradient.shape == channels.shape + (2,)
    assert jnp.allclose(
        spatial.divergence(gradient, dual=True),
        spatial.laplacian(channels),
    )
    assert spatial.integral(channels).shape == (2,)
    assert jnp.allclose(spatial.integral(channels), jnp.zeros((2,)), atol=1e-12)

    partial_integral = spatial.integral(channels, axes=(0,))
    assert partial_integral.shape == (y_axis.nodes.size, 2)
    assert jnp.allclose(
        partial_integral,
        jnp.tensordot(x_axis.quad_weights, channels, axes=((0,), (0,))),
    )

    uniform_axis = phx.domain.UniformAxisSpec(
        16,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    uniform = phx.solver.TensorGridDiscretization((uniform_axis,))
    uniform_state = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * uniform_axis.nodes),
            jnp.cos(2.0 * jnp.pi * uniform_axis.nodes),
        ),
        axis=-1,
    )
    assert jnp.allclose(
        uniform.divergence(uniform.gradient(uniform_state), dual=True),
        uniform.laplacian(uniform_state),
    )
    assert jnp.allclose(
        uniform.partial_derivative(uniform_state, axis=0, order=2),
        uniform.laplacian(uniform_state),
    )

    sine_axis = phx.domain.SineAxisSpec(32).materialize(0.0, 1.0)
    cosine_axis = phx.domain.CosineAxisSpec(33).materialize(0.0, 1.0)
    sine = phx.solver.TensorGridDiscretization((sine_axis,))
    cosine = phx.solver.TensorGridDiscretization((cosine_axis,))
    assert jnp.allclose(
        sine.partial_derivative(jnp.sin(jnp.pi * sine_axis.nodes), axis=0),
        jnp.pi * jnp.cos(jnp.pi * sine_axis.nodes),
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(
        cosine.partial_derivative(jnp.cos(jnp.pi * cosine_axis.nodes), axis=0),
        -jnp.pi * jnp.sin(jnp.pi * cosine_axis.nodes),
        rtol=1e-9,
        atol=1e-9,
    )
    sine_mode = jnp.sin(jnp.pi * sine_axis.nodes)
    cosine_mode = jnp.cos(jnp.pi * cosine_axis.nodes)
    assert jnp.allclose(
        sine.divergence(sine.gradient(sine_mode), dual=True),
        sine.laplacian(sine_mode),
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(
        cosine.divergence(cosine.gradient(cosine_mode), dual=True),
        cosine.laplacian(cosine_mode),
        rtol=1e-9,
        atol=1e-9,
    )


def test_spatial_curl_uses_trailing_vector_axis():
    axes = tuple(
        phx.domain.FourierAxisSpec(8).materialize(0.0, 1.0) for _ in range(3)
    )
    spatial = phx.solver.TensorGridDiscretization(axes)
    x = axes[0].nodes[:, None, None]
    y = axes[1].nodes[None, :, None]
    z = axes[2].nodes[None, None, :]
    zero = jnp.zeros(spatial.state_shape)
    vector = jnp.stack(
        (
            jnp.broadcast_to(jnp.sin(2.0 * jnp.pi * z), spatial.state_shape),
            jnp.broadcast_to(jnp.sin(2.0 * jnp.pi * x), spatial.state_shape),
            jnp.broadcast_to(jnp.sin(2.0 * jnp.pi * y), spatial.state_shape),
        ),
        axis=-1,
    )
    expected = jnp.stack(
        (
            2.0 * jnp.pi * jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * y), spatial.state_shape),
            2.0 * jnp.pi * jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * z), spatial.state_shape),
            2.0 * jnp.pi * jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * x), spatial.state_shape),
        ),
        axis=-1,
    )
    assert zero.shape == spatial.state_shape
    assert jnp.allclose(spatial.curl(vector), expected, atol=1e-9, rtol=1e-9)


def test_compiled_heat_matches_handwritten_jit_and_parameter_gradient():
    axis = phx.domain.SineAxisSpec(32).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    problem = _heat_problem(target=0.0)
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    state = jnp.sin(jnp.pi * axis.nodes)

    expected = 0.35 * spatial.laplacian(state)
    actual = compiled(jnp.asarray(0.0), state, {"kappa": jnp.asarray(0.35)})
    jitted = jax.jit(
        lambda time, value, coefficient: compiled(
            time,
            value,
            {"kappa": coefficient},
        )
    )(jnp.asarray(0.0), state, jnp.asarray(0.35))
    sensitivity = jax.grad(
        lambda coefficient: jnp.sum(
            compiled(0.0, state, {"kappa": coefficient}) ** 2
        )
    )(jnp.asarray(0.35))

    decayed = phx.solver.matrix_exponential_action(
        compiled.semilinear_drift.linear_operator,
        state,
        0.4,
        self_adjoint=True,
        mass_weights=spatial.quadrature_weights,
    )
    assert jnp.allclose(
        decayed,
        jnp.exp(-0.2 * jnp.pi**2 * 0.4) * state,
        rtol=1e-8,
        atol=1e-8,
    )
    assert compiled.state_shape == spatial.state_shape
    assert compiled.semilinear_drift is not None
    assert compiled.resolved_method == "semilinear-matrix-free"
    assert compiled.compilation_id == phx.equations.compile_semidiscrete_pde(
        problem, spatial
    ).compilation_id
    assert jnp.allclose(actual, expected)
    assert jnp.allclose(jitted, expected)
    reparameterized = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        parameter_values={"kappa": 0.25},
    )
    assert reparameterized.compilation_id != compiled.compilation_id
    assert jnp.isfinite(sensitivity) and sensitivity > 0.0
    assert jnp.allclose(
        compiled.semilinear_drift.linear(state),
        -0.2 * jnp.pi**2 * state,
        rtol=1e-9,
        atol=1e-9,
    )


def test_compiled_reaction_diffusion_matches_handwritten_drift():
    axis = phx.domain.FourierAxisSpec(24).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    problem = _heat_problem(periodic=True, reaction=True)
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    state = 0.2 + 0.1 * jnp.sin(2.0 * jnp.pi * axis.nodes)
    args = {"kappa": jnp.asarray(0.07)}
    expected = 0.07 * spatial.laplacian(state) + state * (1.0 - state)

    assert jnp.allclose(compiled(0.4, state, args), expected)
    assert compiled.semilinear_drift is not None


def test_compiler_executes_gradient_divergence_integral_and_coordinate_nodes():
    axes = tuple(
        phx.domain.FourierAxisSpec(12).materialize(0.0, 1.0)
        for _ in range(2)
    )
    spatial = phx.solver.TensorGridDiscretization(axes)
    x = phx.equations.PDECoordinate(
        "x", "space", size=2, bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    u_field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    coordinate = phx.equations.PDEExpression.coordinate_value("x").component(0)
    region = phx.equations.PDERegion("domain", "interior", ("x",))
    gradient_component = u.gradient("x").component(0)
    rhs = (
        u.gradient("x").divergence("x")
        + 0.1 * u.derivative("x", axis=0)
        + 0.01 * u.gradient("x").dot(u.gradient("x"))
        + 0.001 * gradient_component
        + 0.02 * u.integrate("domain")
        + 0.03 * coordinate.sin()
        + 0.04 * (u.exp().log())
        + 0.05 * ((u + 2.0).sqrt() ** 2.0)
        + 0.06 * coordinate.cos()
    )
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(u_field,),
        equations=(
            phx.equations.PDEEquation("all_nodes", u.derivative("t"), rhs),
        ),
        regions=(region,),
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        method="direct",
    )
    state = 0.5 + 0.1 * jnp.sin(2.0 * jnp.pi * axes[0].nodes[:, None])
    state = jnp.broadcast_to(state, spatial.state_shape)
    result = jax.jit(lambda value: compiled(0.0, value, None))(state)

    assert result.shape == state.shape
    assert jnp.all(jnp.isfinite(result))


def test_boundary_basis_failures_and_explicit_nonhomogeneous_lift():
    sine_axis = phx.domain.SineAxisSpec(16).materialize(0.0, 1.0)
    sine = phx.solver.TensorGridDiscretization((sine_axis,))
    problem = _heat_problem(target=1.0)
    with pytest.raises(ValueError, match="explicit BoundaryLift"):
        phx.equations.compile_semidiscrete_pde(problem, sine)

    lift = phx.equations.BoundaryLift(
        "u",
        jnp.ones(sine.state_shape),
        lift_id="constant-one-boundary",
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        sine,
        boundary_lifts=(lift,),
    )
    residual_state = jnp.zeros(sine.state_shape)
    assert jnp.allclose(compiled.physical_state(0.0, residual_state), 1.0)
    assert jnp.allclose(compiled(0.0, residual_state, None), 0.0, atol=1e-10)

    periodic_axis = phx.domain.FourierAxisSpec(16).materialize(0.0, 1.0)
    periodic = phx.solver.TensorGridDiscretization((periodic_axis,))
    with pytest.raises(ValueError, match="periodic=False"):
        phx.equations.compile_semidiscrete_pde(problem, periodic)

    cosine_axis = phx.domain.CosineAxisSpec(16).materialize(0.0, 1.0)
    cosine = phx.solver.TensorGridDiscretization((cosine_axis,))
    homogeneous_dirichlet = _heat_problem(target=0.0)
    with pytest.raises(ValueError, match="requires homogeneous neumann"):
        phx.equations.compile_semidiscrete_pde(homogeneous_dirichlet, cosine)
    u = phx.equations.PDEExpression.field("u")
    neumann_region = phx.equations.PDERegion("boundary", "boundary", ("x",))
    neumann_problem = phx.equations.PDEProblemIR(
        coordinates=homogeneous_dirichlet.coordinates,
        fields=homogeneous_dirichlet.fields,
        parameters=homogeneous_dirichlet.parameters,
        equations=homogeneous_dirichlet.equations,
        conditions=(
            phx.equations.PDECondition(
                "homogeneous-neumann",
                "boundary",
                u.derivative("x"),
                region="boundary",
                coordinate="x",
            ),
        ),
        regions=(neumann_region,),
    )
    assert phx.equations.compile_semidiscrete_pde(
        neumann_problem,
        cosine,
    ).state_shape == cosine.state_shape


def test_compiler_requires_one_evolution_equation_per_field_and_static_packing():
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    fields = (
        phx.equations.PDEField("u", coordinates=("x", "t")),
        phx.equations.PDEField(
            "v",
            representation="vector",
            components=2,
            coordinates=("x", "t"),
        ),
    )
    u = phx.equations.PDEExpression.field("u")
    axis = phx.domain.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    invalid = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=fields,
        equations=(phx.equations.PDEEquation("u", u.derivative("t"), u),),
    )
    with pytest.raises(ValueError, match="missing"):
        phx.equations.compile_semidiscrete_pde(invalid, spatial)

    layout = phx.equations.SemidiscreteFieldLayout(fields, spatial.state_shape)
    packed = layout.pack(
        {
            "u": jnp.ones(spatial.state_shape),
            "v": jnp.ones(spatial.state_shape + (2,)),
        }
    )
    assert layout.state_shape == spatial.state_shape + (3,)
    assert packed.shape == layout.state_shape
    assert layout.field(packed, "u").shape == spatial.state_shape
    assert layout.field(packed, "v").shape == spatial.state_shape + (2,)
    v = phx.equations.PDEExpression.field("v")
    valid = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=fields,
        equations=(
            phx.equations.PDEEquation("u", u.derivative("t"), u),
            phx.equations.PDEEquation("v", v.derivative("t"), v),
        ),
    )
    compiled = phx.equations.compile_semidiscrete_pde(valid, spatial)
    assert compiled.state_shape == layout.state_shape
    assert jnp.allclose(compiled(0.0, packed, None), packed)


def test_spectral_laplacian_compilation_preserves_exact_representation():
    eigenvalues = jnp.asarray([0.0, 1.0, 4.0])
    plan = phx.nn.SpectralDiscretization.from_eigenpairs(
        eigenvalues,
        jnp.eye(3),
        jnp.ones((3,)),
        basis_id="compiler-plan",
    )
    spatial = phx.solver.SpectralSpatialDiscretization(plan)
    problem = _heat_problem()
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    state = jnp.asarray([1.0, 2.0, 3.0])

    assert compiled.resolved_method == "semilinear-spectral"
    assert compiled.semilinear_drift.spectral_representation is not None
    assert jnp.allclose(compiled(0.0, state, None), -0.2 * eigenvalues * state)


def test_semidiscrete_pde_benchmark_tracks_parity_and_provenance():
    from tools.high_dimensional_pde_benchmarks import (
        run_semidiscrete_pde_compiler_benchmark,
    )

    record = run_semidiscrete_pde_compiler_benchmark(8, repeats=1)

    assert record.passed
    assert record.compilation_id
    assert record.operator_id
    assert record.resolved_method == "semilinear-matrix-free"
    assert record.compiled_jit_ms >= 0.0
    assert record.compiled_mean_wall_ms >= 0.0



def test_divergence_distinguishes_primal_vectors_from_gradient_duals():
    sine_axis = phx.domain.SineAxisSpec(32).materialize(0.0, 1.0)
    cosine_axis = phx.domain.CosineAxisSpec(33).materialize(0.0, 1.0)
    sine = phx.solver.TensorGridDiscretization((sine_axis,))
    cosine = phx.solver.TensorGridDiscretization((cosine_axis,))

    sine_vector = jnp.sin(jnp.pi * sine_axis.nodes)[..., None]
    cosine_vector = jnp.cos(jnp.pi * cosine_axis.nodes)[..., None]
    assert jnp.allclose(
        sine.divergence(sine_vector),
        jnp.pi * jnp.cos(jnp.pi * sine_axis.nodes),
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(
        cosine.divergence(cosine_vector),
        -jnp.pi * jnp.sin(jnp.pi * cosine_axis.nodes),
        rtol=1e-9,
        atol=1e-9,
    )


def test_coordinate_and_time_scalar_coefficients_broadcast_over_fields():
    axis = phx.domain.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    coordinate_rhs = phx.equations.PDEExpression.coordinate_value("x") * u
    time_rhs = phx.equations.PDEExpression.coordinate_value("t") * u

    def problem(rhs):
        return phx.equations.PDEProblemIR(
            coordinates=(x, t),
            fields=(field,),
            equations=(
                phx.equations.PDEEquation("evolution", u.derivative("t"), rhs),
            ),
        )

    state = 1.0 + jnp.sin(2.0 * jnp.pi * axis.nodes)
    coordinate_compiled = phx.equations.compile_semidiscrete_pde(
        problem(coordinate_rhs),
        spatial,
    )
    time_compiled = phx.equations.compile_semidiscrete_pde(
        problem(time_rhs),
        spatial,
    )
    assert jnp.allclose(
        coordinate_compiled(0.0, state, None),
        axis.nodes * state,
    )
    assert jnp.allclose(time_compiled(0.25, state, None), 0.25 * state)
    assert time_compiled.resolved_method == "direct"


def test_partial_integrals_reinsert_integrated_axes_for_field_arithmetic():
    axes = (
        phx.domain.FourierAxisSpec(8).materialize(0.0, 1.0),
        phx.domain.FourierAxisSpec(10).materialize(0.0, 1.0),
    )
    spatial = phx.solver.TensorGridDiscretization(axes)
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    y = phx.equations.PDECoordinate(
        "y", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "y", "t"))
    u = phx.equations.PDEExpression.field("u")
    region = phx.equations.PDERegion("x-domain", "interior", ("x",))
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, y, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "partial-integral",
                u.derivative("t"),
                u + u.integrate("x-domain"),
            ),
        ),
        regions=(region,),
    )
    state = (
        1.0
        + jnp.sin(2.0 * jnp.pi * axes[0].nodes[:, None])
        + 0.2 * jnp.cos(2.0 * jnp.pi * axes[1].nodes[None, :])
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        method="direct",
    )
    integrated = spatial.integral(state, axes=(0,))
    expected = state + integrated[None, :]
    assert jnp.allclose(compiled(0.0, state, None), expected)


def test_temporal_derivatives_are_rejected_anywhere_on_evolution_rhs():
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "implicit-time",
                u.derivative("t"),
                u + u.derivative("t"),
            ),
        ),
    )
    axis = phx.domain.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    with pytest.raises(ValueError, match="temporal derivative"):
        phx.equations.compile_semidiscrete_pde(problem, spatial)


def test_functional_parameters_validate_shapes_and_broadcast_components():
    axis = phx.domain.FourierAxisSpec(12).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    parameters = (
        phx.equations.PDEParameter("a", functional=True),
        phx.equations.PDEParameter("b", components=2, functional=True),
    )
    u = phx.equations.PDEExpression.field("u")
    a = phx.equations.PDEExpression.parameter("a")
    b = phx.equations.PDEExpression.parameter("b")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=parameters,
        equations=(
            phx.equations.PDEEquation(
                "functional-coefficients",
                u.derivative("t"),
                a * u + b.component(0) + b.component(1) * u,
            ),
        ),
    )
    state = 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * axis.nodes)
    a_value = 0.5 + axis.nodes
    b_value = jnp.stack((axis.nodes, 2.0 * axis.nodes), axis=-1)
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        parameter_values={"a": a_value, "b": b_value},
    )
    expected = a_value * state + b_value[..., 0] + b_value[..., 1] * state
    assert jnp.allclose(compiled(0.0, state, None), expected)
    assert jnp.allclose(
        compiled(
            0.0,
            state,
            {"a": jnp.asarray(2.0), "b": jnp.asarray((3.0, 4.0))},
        ),
        6.0 * state + 3.0,
    )
    with pytest.raises(ValueError, match="expected one of"):
        phx.equations.compile_semidiscrete_pde(
            problem,
            spatial,
            parameter_values={"a": jnp.ones((2, 2)), "b": b_value},
        )


def test_missing_and_complex_parameter_bindings_fail_explicitly():
    axis = phx.domain.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    parameter = phx.equations.PDEParameter("a")
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(parameter,),
        equations=(
            phx.equations.PDEEquation(
                "parameterized",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("a") * u,
            ),
        ),
    )
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    state = jnp.ones(spatial.state_shape)
    with pytest.raises(KeyError, match="No value supplied"):
        compiled(0.0, state, None)
    with pytest.raises(TypeError, match="real-valued"):
        compiled(0.0, state, {"a": jnp.asarray(1.0 + 1.0j)})
    with pytest.raises(TypeError, match="real-valued"):
        phx.equations.compile_semidiscrete_pde(
            problem,
            spatial,
            parameter_values={"a": 1.0 + 1.0j},
        )


def test_additive_evolution_isolation_and_nonlinearity_guard():
    axis = phx.domain.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=True
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    isolated = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "additive-time",
                u.derivative("t") + 0.2 * u.laplacian("x"),
                u * (1.0 - u),
            ),
        ),
    )
    compiled = phx.equations.compile_semidiscrete_pde(isolated, spatial)
    state = 0.3 + 0.1 * jnp.sin(2.0 * jnp.pi * axis.nodes)
    assert compiled.semilinear_drift is not None
    assert jnp.allclose(
        compiled(0.0, state, None),
        state * (1.0 - state) - 0.2 * spatial.laplacian(state),
    )

    nonlinear_division = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "nonlinear-division",
                u.derivative("t"),
                u / (u + 1.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="could not conservatively isolate"):
        phx.equations.compile_semidiscrete_pde(
            nonlinear_division,
            spatial,
            method="semilinear",
        )


def test_boundary_lifts_require_derivatives_and_match_constant_targets():
    with pytest.raises(ValueError, match="time_derivative"):
        phx.equations.BoundaryLift(
            "u",
            lambda time, args: jnp.ones((8,)),
            lift_id="time-dependent",
        )

    axis = phx.domain.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    problem = _heat_problem(target=1.0)
    mismatched = phx.equations.BoundaryLift(
        "u",
        jnp.zeros(spatial.state_shape),
        lift_id="wrong-constant",
    )
    with pytest.raises(ValueError, match="does not match boundary target"):
        phx.equations.compile_semidiscrete_pde(
            problem,
            spatial,
            boundary_lifts=(mismatched,),
        )


def test_boundary_regions_and_field_spatial_layouts_must_match():
    axes = (
        phx.domain.SineAxisSpec(8).materialize(0.0, 1.0),
        phx.domain.SineAxisSpec(10).materialize(0.0, 1.0),
    )
    spatial = phx.solver.TensorGridDiscretization(axes)
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    y = phx.equations.PDECoordinate("y", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    u_field = phx.equations.PDEField("u", coordinates=("x", "y", "t"))
    u = phx.equations.PDEExpression.field("u")
    mismatched_boundary = phx.equations.PDEProblemIR(
        coordinates=(x, y, t),
        fields=(u_field,),
        equations=(
            phx.equations.PDEEquation(
                "evolution",
                u.derivative("t"),
                u.laplacian("x") + u.laplacian("y"),
            ),
        ),
        conditions=(
            phx.equations.PDECondition(
                "wrong-normal",
                "boundary",
                u,
                region="y-boundary",
                coordinate="x",
            ),
        ),
        regions=(
            phx.equations.PDERegion(
                "y-boundary",
                "boundary",
                ("y",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not match region"):
        phx.equations.compile_semidiscrete_pde(mismatched_boundary, spatial)

    v_field = phx.equations.PDEField("v", coordinates=("y", "t"))
    v = phx.equations.PDEExpression.field("v")
    heterogeneous = phx.equations.PDEProblemIR(
        coordinates=(x, y, t),
        fields=(u_field, v_field),
        equations=(
            phx.equations.PDEEquation("u", u.derivative("t"), u),
            phx.equations.PDEEquation("v", v.derivative("t"), v),
        ),
    )
    with pytest.raises(ValueError, match="same complete spatial coordinate layout"):
        phx.equations.compile_semidiscrete_pde(heterogeneous, spatial)


def test_uniform_tensor_grid_rejects_nonperiodic_roll_semantics():
    axis = phx.domain.UniformAxisSpec(
        8,
        endpoint=True,
        periodic=False,
    ).materialize(0.0, 1.0)
    with pytest.raises(ValueError, match="periodic=True"):
        phx.solver.TensorGridDiscretization((axis,))


def test_spectral_compilation_rejects_frames_and_only_preserves_full_bases():
    full_plan = phx.nn.SpectralDiscretization.from_eigenpairs(
        jnp.asarray((0.0, 1.0, 4.0)),
        jnp.eye(3),
        jnp.ones((3,)),
        basis_id="full-frame-check",
    )
    full = phx.solver.SpectralSpatialDiscretization(full_plan)
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    coordinate_problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "coordinate",
                u.derivative("t"),
                phx.equations.PDEExpression.coordinate_value("x") * u,
            ),
        ),
    )
    with pytest.raises(ValueError, match="has no coordinate frame"):
        phx.equations.compile_semidiscrete_pde(coordinate_problem, full)

    truncated_plan = phx.nn.SpectralDiscretization.from_eigenpairs(
        jnp.asarray((0.0, 1.0)),
        jnp.asarray(
            (
                (1.0, 1.0),
                (1.0, 0.0),
                (1.0, -1.0),
            )
        ),
        jnp.ones((3,)),
        basis_id="truncated-plan",
    )
    truncated = phx.solver.SpectralSpatialDiscretization(truncated_plan)
    compiled = phx.equations.compile_semidiscrete_pde(
        _heat_problem(),
        truncated,
    )
    assert compiled.resolved_method == "semilinear-matrix-free"
    assert compiled.semilinear_drift.spectral_representation is None


def test_compiled_operator_identity_includes_parameters_and_lifts():
    axis = phx.domain.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    problem = _heat_problem(target=0.0)
    first = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        parameter_values={"kappa": 0.2},
    )
    rebound = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        parameter_values={"kappa": 0.3},
    )
    lifted = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        parameter_values={"kappa": 0.2},
        boundary_lifts=(
            phx.equations.BoundaryLift(
                "u",
                jnp.zeros(spatial.state_shape),
                lift_id="zero-lift",
            ),
        ),
    )
    assert first.semilinear_drift.operator_id != rebound.semilinear_drift.operator_id
    assert first.semilinear_drift.operator_id != lifted.semilinear_drift.operator_id
    assert first.compilation_id != rebound.compilation_id
    assert first.compilation_id != lifted.compilation_id
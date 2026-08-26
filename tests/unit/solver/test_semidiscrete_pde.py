import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.discretization as spectral


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
    x_axis = phx.discretization.FourierAxisSpec(24).materialize(0.0, 1.0)
    y_axis = phx.discretization.FourierAxisSpec(20).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((x_axis, y_axis))
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

    uniform_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                16,
                endpoint=False,
                periodic=True,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    uniform_axis = uniform_grid.axes[0]
    uniform = phx.discretization.periodic_finite_difference(uniform_grid)
    uniform_state = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * uniform_axis.nodes),
            jnp.cos(2.0 * jnp.pi * uniform_axis.nodes),
        ),
        axis=-1,
    )
    assert uniform.gradient(uniform_state).shape == (16, 2, 1)
    assert jnp.allclose(
        uniform.partial_derivative(uniform_state, axis=0, order=2),
        uniform.laplacian(uniform_state),
    )

    sine_axis = phx.discretization.SineAxisSpec(32).materialize(0.0, 1.0)
    cosine_axis = phx.discretization.CosineAxisSpec(33).materialize(0.0, 1.0)
    sine = phx.discretization.TensorSpectralDiscretization.from_axes((sine_axis,))
    cosine = phx.discretization.TensorSpectralDiscretization.from_axes((cosine_axis,))
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
        phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0) for _ in range(3)
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes(axes)
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
            2.0
            * jnp.pi
            * jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * y), spatial.state_shape),
            2.0
            * jnp.pi
            * jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * z), spatial.state_shape),
            2.0
            * jnp.pi
            * jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * x), spatial.state_shape),
        ),
        axis=-1,
    )
    assert zero.shape == spatial.state_shape
    assert jnp.allclose(spatial.curl(vector), expected, atol=1e-9, rtol=1e-9)


def test_compiled_heat_matches_handwritten_jit_and_parameter_gradient():
    axis = phx.discretization.SineAxisSpec(32).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    problem = _heat_problem(target=0.0)
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    assert isinstance(
        compiled,
        phx.equations.CompiledDiscreteDynamics,
    )
    assert (
        compiled.discretization_bundle.record(spatial.key).artifact_id
        == spatial.prepared_id
    )
    assert compiled.layout.field_spaces[0].representation == "point_value"
    assert (
        compiled.layout.field_spaces[0].vector_space.dtype
        == spatial.physical_space.vector_space.dtype
    )
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
        lambda coefficient: jnp.sum(compiled(0.0, state, {"kappa": coefficient}) ** 2)
    )(jnp.asarray(0.35))
    assert compiled.semilinear_drift is not None

    decayed = phx.linalg.matrix_exponential_action(
        compiled.semilinear_drift.linear_operator,
        state,
        0.4,
    ).value
    assert jnp.allclose(
        decayed,
        jnp.exp(-0.2 * jnp.pi**2 * 0.4) * state,
        rtol=1e-8,
        atol=1e-8,
    )
    assert compiled.state_shape == spatial.state_shape
    assert compiled.semilinear_drift is not None
    assert compiled.resolved_method == "semilinear-matrix-free"
    assert (
        compiled.compilation_id
        == phx.equations.compile_semidiscrete_pde(problem, spatial).compilation_id
    )
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
    axis = phx.discretization.FourierAxisSpec(24).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    problem = _heat_problem(periodic=True, reaction=True)
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    state = 0.2 + 0.1 * jnp.sin(2.0 * jnp.pi * axis.nodes)
    args = {"kappa": jnp.asarray(0.07)}
    expected = 0.07 * spatial.laplacian(state) + state * (1.0 - state)

    assert jnp.allclose(compiled(0.4, state, args), expected)
    assert compiled.semilinear_drift is not None


def test_compiler_executes_gradient_divergence_integral_and_coordinate_nodes():
    axes = tuple(
        phx.discretization.FourierAxisSpec(12).materialize(0.0, 1.0) for _ in range(2)
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes(axes)
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
        equations=(phx.equations.PDEEquation("all_nodes", u.derivative("t"), rhs),),
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
    sine_axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    sine = phx.discretization.TensorSpectralDiscretization.from_axes((sine_axis,))
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

    periodic_axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    periodic = phx.discretization.TensorSpectralDiscretization.from_axes((periodic_axis,))
    with pytest.raises(ValueError, match="periodic=False"):
        phx.equations.compile_semidiscrete_pde(problem, periodic)

    cosine_axis = phx.discretization.CosineAxisSpec(16).materialize(0.0, 1.0)
    cosine = phx.discretization.TensorSpectralDiscretization.from_axes((cosine_axis,))
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
    assert (
        phx.equations.compile_semidiscrete_pde(
            neumann_problem,
            cosine,
        ).state_shape
        == cosine.state_shape
    )


def test_compiler_requires_one_evolution_equation_per_field_and_static_packing():
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
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
    axis = phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    invalid = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=fields,
        equations=(phx.equations.PDEEquation("u", u.derivative("t"), u),),
    )
    with pytest.raises(ValueError, match="missing"):
        phx.equations.compile_semidiscrete_pde(invalid, spatial)

    layout = phx.equations.DiscreteStateLayout(fields, spatial)
    packed = layout.pack(
        {
            "u": jnp.ones(spatial.state_shape),
            "v": jnp.ones(spatial.state_shape + (2,)),
        }
    )
    assert layout.state_shape == spatial.state_shape + (3,)
    assert tuple(space.name for space in layout.field_spaces) == ("u", "v")
    assert all(
        space.support_id == spatial.support.support_id for space in layout.field_spaces
    )
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
    plan = spectral.SpectralDecomposition.from_eigenpairs(
        eigenvalues,
        jnp.eye(3),
        jnp.ones((3,)),
        decomposition_id="compiler-plan",
    )
    spatial = phx.discretization.EigenbasisDiscretization(plan)
    problem = _heat_problem()
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    state = jnp.asarray([1.0, 2.0, 3.0])
    assert compiled.semilinear_drift is not None

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
    sine_axis = phx.discretization.SineAxisSpec(32).materialize(0.0, 1.0)
    cosine_axis = phx.discretization.CosineAxisSpec(33).materialize(0.0, 1.0)
    sine = phx.discretization.TensorSpectralDiscretization.from_axes((sine_axis,))
    cosine = phx.discretization.TensorSpectralDiscretization.from_axes((cosine_axis,))

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
    axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    coordinate_rhs = phx.equations.PDEExpression.coordinate_value("x") * u
    time_rhs = phx.equations.PDEExpression.coordinate_value("t") * u

    def problem(rhs):
        return phx.equations.PDEProblemIR(
            coordinates=(x, t),
            fields=(field,),
            equations=(phx.equations.PDEEquation("evolution", u.derivative("t"), rhs),),
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
        phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0),
        phx.discretization.FourierAxisSpec(10).materialize(0.0, 1.0),
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes(axes)
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    y = phx.equations.PDECoordinate("y", "space", bounds=(0.0, 1.0), periodic=True)
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
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
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
    axis = phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    with pytest.raises(ValueError, match="temporal derivative"):
        phx.equations.compile_semidiscrete_pde(problem, spatial)


def test_functional_parameters_validate_shapes_and_broadcast_components():
    axis = phx.discretization.FourierAxisSpec(12).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
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
    axis = phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
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
    axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
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

    axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
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
        phx.discretization.SineAxisSpec(8).materialize(0.0, 1.0),
        phx.discretization.SineAxisSpec(10).materialize(0.0, 1.0),
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes(axes)
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
    axis = phx.discretization.UniformAxisSpec(
        8,
        endpoint=True,
        periodic=False,
    ).materialize(0.0, 1.0)
    with pytest.raises(ValueError, match="require FiniteDifferencePlan"):
        phx.discretization.TensorSpectralDiscretization.from_axes((axis,))


def test_spectral_compilation_rejects_frames_and_only_preserves_full_bases():
    full_plan = spectral.SpectralDecomposition.from_eigenpairs(
        jnp.asarray((0.0, 1.0, 4.0)),
        jnp.eye(3),
        jnp.ones((3,)),
        decomposition_id="full-frame-check",
    )
    full = phx.discretization.EigenbasisDiscretization(full_plan)
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

    truncated_plan = spectral.SpectralDecomposition.from_eigenpairs(
        jnp.asarray((0.0, 1.0)),
        jnp.asarray(
            (
                (1.0, 1.0),
                (1.0, 0.0),
                (1.0, -1.0),
            )
        ),
        jnp.ones((3,)),
        decomposition_id="truncated-plan",
    )
    truncated = phx.discretization.EigenbasisDiscretization(truncated_plan)
    compiled = phx.equations.compile_semidiscrete_pde(
        _heat_problem(),
        truncated,
    )
    assert compiled.semilinear_drift is not None
    assert compiled.resolved_method == "semilinear-matrix-free"
    assert compiled.semilinear_drift.spectral_representation is None


def test_compiled_operator_identity_includes_parameters_and_lifts():
    axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
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
    assert first.semilinear_drift is not None
    assert rebound.semilinear_drift is not None
    assert lifted.semilinear_drift is not None
    assert first.semilinear_drift.operator_id != rebound.semilinear_drift.operator_id
    assert first.semilinear_drift.operator_id != lifted.semilinear_drift.operator_id
    assert first.compilation_id != rebound.compilation_id
    assert first.compilation_id != lifted.compilation_id


def test_one_dimensional_gradient_products_keep_the_vector_axis():
    axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    flux = (1.0 + coordinate) * u.gradient("x")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "variable-flux",
                u.derivative("t"),
                flux.divergence("x"),
            ),
        ),
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        method="direct",
    )
    state = jnp.sin(2.0 * jnp.pi * axis.nodes)
    expected_flux = (1.0 + axis.nodes)[..., None] * spatial.gradient(state)
    assert jnp.allclose(
        compiled(0.0, state, None),
        spatial.divergence(expected_flux),
    )


def test_partial_integrals_are_full_spatial_fields_for_parent_nodes():
    axes = (
        phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0),
        phx.discretization.FourierAxisSpec(10).materialize(0.0, 1.0),
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes(axes)
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    y = phx.equations.PDECoordinate("y", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "y", "t"))
    u = phx.equations.PDEExpression.field("u")
    region = phx.equations.PDERegion("x-domain", "interior", ("x",))

    def problem(rhs):
        return phx.equations.PDEProblemIR(
            coordinates=(x, y, t),
            fields=(field,),
            equations=(phx.equations.PDEEquation("integral", u.derivative("t"), rhs),),
            regions=(region,),
        )

    state = (
        1.0
        + jnp.sin(2.0 * jnp.pi * axes[0].nodes[:, None])
        + 0.2 * jnp.cos(2.0 * jnp.pi * axes[1].nodes[None, :])
    )
    integrated = spatial.integral(state, axes=(0,))[None, :]
    integrated = jnp.broadcast_to(integrated, spatial.state_shape)
    direct = phx.equations.compile_semidiscrete_pde(
        problem(u.integrate("x-domain")),
        spatial,
        method="direct",
    )
    differentiated = phx.equations.compile_semidiscrete_pde(
        problem(u.integrate("x-domain").derivative("y")),
        spatial,
        method="direct",
    )
    assert jnp.allclose(direct(0.0, state, None), integrated)
    assert jnp.allclose(
        differentiated(0.0, state, None),
        spatial.partial_derivative(integrated, axis=1),
    )


def test_nested_derivatives_honor_spectral_and_uniform_duals():
    sine_axes = tuple(
        phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0) for _ in range(2)
    )
    sine = phx.discretization.TensorSpectralDiscretization.from_axes(sine_axes)
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        size=2,
        bounds=(0.0, 1.0),
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    sine_problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "nested-sine",
                u.derivative("t"),
                u.derivative("x", axis=0).gradient("x").component(0),
            ),
        ),
    )
    sine_compiled = phx.equations.compile_semidiscrete_pde(
        sine_problem,
        sine,
        method="direct",
    )
    sine_state = jnp.sin(jnp.pi * sine_axes[0].nodes[:, None]) * jnp.sin(
        jnp.pi * sine_axes[1].nodes[None, :]
    )
    assert jnp.allclose(
        sine_compiled(0.0, sine_state, None),
        sine.partial_derivative(sine_state, axis=0, order=2),
        rtol=1e-9,
        atol=1e-9,
    )

    uniform_axes = tuple(
        phx.discretization.UniformAxisSpec(
            16,
            endpoint=False,
            periodic=True,
        ).materialize(0.0, 1.0)
        for _ in range(2)
    )
    uniform = phx.discretization.PreparedTensorGrid(uniform_axes)
    periodic_x = phx.equations.PDECoordinate(
        "x",
        "space",
        size=2,
        bounds=(0.0, 1.0),
        periodic=True,
    )
    uniform_problem = phx.equations.PDEProblemIR(
        coordinates=(periodic_x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "nested-uniform",
                u.derivative("t"),
                u.gradient("x").component(0).derivative("x", axis=0),
            ),
        ),
    )
    uniform_compiled = phx.equations.compile_semidiscrete_pde(
        uniform_problem,
        uniform,
        method="direct",
    )
    uniform_fd = phx.discretization.periodic_finite_difference(uniform)
    uniform_state = jnp.broadcast_to(
        jnp.sin(2.0 * jnp.pi * uniform_axes[0].nodes[:, None]),
        uniform.shape,
    )
    assert jnp.allclose(
        uniform_compiled(0.0, uniform_state, None),
        uniform_fd.partial_derivative(
            uniform_fd.partial_derivative(uniform_state, axis=0, order=1),
            axis=0,
            order=1,
        ),
    )


def test_nonperiodic_composite_parity_is_rejected_but_coordinates_are_exact():
    axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")

    def problem(rhs):
        return phx.equations.PDEProblemIR(
            coordinates=(x, t),
            fields=(field,),
            equations=(phx.equations.PDEEquation("parity", u.derivative("t"), rhs),),
        )

    for expression in ((u + 1.0).derivative("x"), (u * u).derivative("x")):
        with pytest.raises(ValueError, match="extension parity"):
            phx.equations.compile_semidiscrete_pde(problem(expression), spatial)

    coordinate_derivative = phx.equations.compile_semidiscrete_pde(
        problem(phx.equations.PDEExpression.coordinate_value("x").derivative("x")),
        spatial,
        method="direct",
    )
    assert jnp.allclose(
        coordinate_derivative(0.0, jnp.zeros(spatial.state_shape), None),
        jnp.ones(spatial.state_shape),
    )


def test_one_component_vector_fields_keep_semantic_component_axes():
    axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    fields = (
        phx.equations.PDEField("u", coordinates=("x", "t")),
        phx.equations.PDEField(
            "v",
            representation="vector",
            components=1,
            coordinates=("x", "t"),
        ),
    )
    u = phx.equations.PDEExpression.field("u")
    v = phx.equations.PDEExpression.field("v")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=fields,
        equations=(
            phx.equations.PDEEquation(
                "scalar",
                u.derivative("t"),
                v.divergence("x"),
            ),
            phx.equations.PDEEquation("vector", v.derivative("t"), v),
        ),
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        method="direct",
    )
    scalar = jnp.zeros(spatial.state_shape)
    vector = jnp.sin(2.0 * jnp.pi * axis.nodes)[..., None]
    state = compiled.layout.pack({"u": scalar, "v": vector})
    result = compiled.layout.unpack(compiled(0.0, state, None))
    assert compiled.layout.field_shape("v") == spatial.state_shape + (1,)
    assert jnp.allclose(
        result["u"],
        spatial.partial_derivative(vector[..., 0], axis=0),
    )
    assert jnp.allclose(result["v"], vector)


def test_lifts_propagate_through_gradient_divergence_composition():
    axis = phx.discretization.SineAxisSpec(32).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    base = _heat_problem(target=1.0)
    u = phx.equations.PDEExpression.field("u")
    composed = phx.equations.PDEProblemIR(
        coordinates=base.coordinates,
        fields=base.fields,
        parameters=(),
        equations=(
            phx.equations.PDEEquation(
                "composed",
                u.derivative("t"),
                u.gradient("x").divergence("x"),
            ),
        ),
        conditions=base.conditions,
        regions=base.regions,
    )
    laplacian = phx.equations.PDEProblemIR(
        coordinates=base.coordinates,
        fields=base.fields,
        parameters=(),
        equations=(
            phx.equations.PDEEquation(
                "laplacian",
                u.derivative("t"),
                u.laplacian("x"),
            ),
        ),
        conditions=base.conditions,
        regions=base.regions,
    )
    lift_value = 1.0 + axis.nodes * (1.0 - axis.nodes)
    lift = phx.equations.BoundaryLift(
        "u",
        lift_value,
        lift_id="curved-lift",
    )
    composed_compiled = phx.equations.compile_semidiscrete_pde(
        composed,
        spatial,
        boundary_lifts=(lift,),
        method="direct",
    )
    laplacian_compiled = phx.equations.compile_semidiscrete_pde(
        laplacian,
        spatial,
        boundary_lifts=(lift,),
        method="direct",
    )
    residual = 0.2 * jnp.sin(jnp.pi * axis.nodes)
    assert jnp.allclose(
        composed_compiled(0.0, residual, None),
        laplacian_compiled(0.0, residual, None),
        rtol=1e-9,
        atol=1e-9,
    )


def test_unsupported_region_and_condition_semantics_fail_at_compile_time():
    sine_axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    sine = phx.discretization.TensorSpectralDiscretization.from_axes((sine_axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    boundary_region = phx.equations.PDERegion(
        "boundary",
        "boundary",
        ("x",),
    )
    boundary_integral = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "boundary-integral",
                u.derivative("t"),
                u.integrate("boundary"),
            ),
        ),
        regions=(boundary_region,),
    )
    with pytest.raises(ValueError, match="only supports unpartitioned interior"):
        phx.equations.compile_semidiscrete_pde(boundary_integral, sine)

    component_region = phx.equations.PDERegion(
        "left-boundary",
        "boundary",
        ("x",),
        component="left",
    )
    component_condition = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(phx.equations.PDEEquation("evolution", u.derivative("t"), u),),
        conditions=(
            phx.equations.PDECondition(
                "left-only",
                "boundary",
                u,
                region="left-boundary",
                coordinate="x",
            ),
        ),
        regions=(component_region,),
    )
    with pytest.raises(ValueError, match="enforce both boundary sides"):
        phx.equations.compile_semidiscrete_pde(component_condition, sine)

    interface_region = phx.equations.PDERegion(
        "interface",
        "interface",
        ("x",),
    )
    interface_problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(phx.equations.PDEEquation("evolution", u.derivative("t"), u),),
        conditions=(
            phx.equations.PDECondition(
                "continuity",
                "interface",
                u,
                region="interface",
                coordinate="x",
            ),
        ),
        regions=(interface_region,),
    )
    with pytest.raises(ValueError, match="does not support interface conditions"):
        phx.equations.compile_semidiscrete_pde(interface_problem, sine)


def test_coordinate_bounds_and_grouped_derivative_capabilities_are_validated():
    mismatched_axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 2.0)
    mismatched = phx.discretization.TensorSpectralDiscretization.from_axes(
        (mismatched_axis,)
    )
    with pytest.raises(ValueError, match="do not match discretization axis bounds"):
        phx.equations.compile_semidiscrete_pde(
            _heat_problem(periodic=True),
            mismatched,
        )

    axes = tuple(
        phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0) for _ in range(2)
    )
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes(axes)
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        size=2,
        bounds=(0.0, 1.0),
        periodic=True,
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "grouped",
                u.derivative("t"),
                u.derivative("x"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="require an explicit axis"):
        phx.equations.compile_semidiscrete_pde(problem, spatial)


def test_full_integrals_and_second_uniform_gradients_preserve_field_shape():
    axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    region = phx.equations.PDERegion("domain", "interior", ("x",))
    integral_problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "integral-derivative",
                u.derivative("t"),
                u.integrate("domain").derivative("x"),
            ),
        ),
        regions=(region,),
    )
    integral_compiled = phx.equations.compile_semidiscrete_pde(
        integral_problem,
        spatial,
        method="direct",
    )
    assert jnp.allclose(
        integral_compiled(0.0, jnp.ones(spatial.state_shape), None),
        jnp.zeros(spatial.state_shape),
        atol=1e-12,
    )

    axes = tuple(
        phx.discretization.UniformAxisSpec(
            16,
            endpoint=False,
            periodic=True,
        ).materialize(0.0, 1.0)
        for _ in range(2)
    )
    uniform = phx.discretization.PreparedTensorGrid(axes)
    grouped = phx.equations.PDECoordinate(
        "x",
        "space",
        size=2,
        bounds=(0.0, 1.0),
        periodic=True,
    )
    grouped_field = phx.equations.PDEField("u", coordinates=("x", "t"))
    second_gradient = u.gradient("x").component(0).gradient("x").component(0)
    uniform_problem = phx.equations.PDEProblemIR(
        coordinates=(grouped, t),
        fields=(grouped_field,),
        equations=(
            phx.equations.PDEEquation(
                "second-gradient",
                u.derivative("t"),
                second_gradient,
            ),
        ),
    )
    uniform_compiled = phx.equations.compile_semidiscrete_pde(
        uniform_problem,
        uniform,
        method="direct",
    )
    uniform_fd = phx.discretization.periodic_finite_difference(uniform)
    state = jnp.broadcast_to(
        jnp.sin(2.0 * jnp.pi * axes[0].nodes[:, None]),
        uniform.shape,
    )
    assert jnp.allclose(
        uniform_compiled(0.0, state, None),
        uniform_fd.partial_derivative(
            uniform_fd.partial_derivative(state, axis=0, order=1),
            axis=0,
            order=1,
        ),
    )


def test_affine_lifts_and_coordinate_calculus_are_composition_safe():
    axis = phx.discretization.SineAxisSpec(32).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    base = _heat_problem(target=1.0)
    u = phx.equations.PDEExpression.field("u")
    lift = phx.equations.BoundaryLift(
        "u",
        1.0 + axis.nodes * (1.0 - axis.nodes),
        lift_id="affine-composition-lift",
    )

    def problem(rhs):
        return phx.equations.PDEProblemIR(
            coordinates=base.coordinates,
            fields=base.fields,
            equations=(
                phx.equations.PDEEquation(
                    "calculus",
                    u.derivative("t"),
                    rhs,
                ),
            ),
            conditions=base.conditions,
            regions=base.regions,
        )

    laplacian = phx.equations.compile_semidiscrete_pde(
        problem(u.laplacian("x")),
        spatial,
        boundary_lifts=(lift,),
        method="direct",
    )
    negated = phx.equations.compile_semidiscrete_pde(
        problem((-u).laplacian("x")),
        spatial,
        boundary_lifts=(lift,),
        method="direct",
    )
    residual = 0.2 * jnp.sin(jnp.pi * axis.nodes)
    assert jnp.allclose(
        negated(0.0, residual, None),
        -laplacian(0.0, residual, None),
        rtol=1e-9,
        atol=1e-9,
    )

    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    coordinate_compiled = phx.equations.compile_semidiscrete_pde(
        phx.equations.PDEProblemIR(
            coordinates=base.coordinates,
            fields=base.fields,
            equations=(
                phx.equations.PDEEquation(
                    "coordinate-calculus",
                    u.derivative("t"),
                    coordinate.gradient("x").divergence("x") + coordinate.laplacian("x"),
                ),
            ),
        ),
        spatial,
        method="direct",
    )
    assert jnp.allclose(
        coordinate_compiled(0.0, jnp.zeros(spatial.state_shape), None),
        jnp.zeros(spatial.state_shape),
    )


def test_functional_parameter_parity_and_integral_region_contracts_are_explicit():
    axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    parameter = phx.equations.PDEParameter("a", functional=True)
    functional = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(parameter,),
        equations=(
            phx.equations.PDEEquation(
                "functional-derivative",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("a").derivative("x"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="extension parity"):
        phx.equations.compile_semidiscrete_pde(
            functional,
            spatial,
            parameter_values={"a": jnp.ones(spatial.state_shape)},
        )

    invalid_region = phx.equations.PDERegion(
        "space-time-slice",
        "interior",
        ("x", "t"),
        component="subdomain",
    )
    invalid_integral = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "invalid-integral",
                u.derivative("t"),
                u.integrate("space-time-slice"),
            ),
        ),
        regions=(invalid_region,),
    )
    with pytest.raises(ValueError, match="unpartitioned interior spatial regions"):
        phx.equations.compile_semidiscrete_pde(invalid_integral, spatial)


def test_variable_flux_parity_and_scalar_divergence_fail_explicitly():
    axis = phx.discretization.SineAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0))
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    coordinate = phx.equations.PDEExpression.coordinate_value("x")

    def problem(rhs, *, parameters=()):
        return phx.equations.PDEProblemIR(
            coordinates=(x, t),
            fields=(field,),
            parameters=parameters,
            equations=(phx.equations.PDEEquation("flux", u.derivative("t"), rhs),),
        )

    coordinate_flux = ((1.0 + coordinate) * u.gradient("x")).divergence("x")
    with pytest.raises(ValueError, match="extension parity"):
        phx.equations.compile_semidiscrete_pde(
            problem(coordinate_flux),
            spatial,
        )

    parameter = phx.equations.PDEParameter("a", functional=True)
    coefficient = phx.equations.PDEExpression.parameter("a")
    parameter_flux = (coefficient * u.gradient("x")).divergence("x")
    with pytest.raises(ValueError, match="extension parity"):
        phx.equations.compile_semidiscrete_pde(
            problem(parameter_flux, parameters=(parameter,)),
            spatial,
            parameter_values={"a": jnp.ones(spatial.state_shape)},
        )

    with pytest.raises(ValueError, match="requires a vector-like operand"):
        phx.equations.compile_semidiscrete_pde(
            problem(u.divergence("x")),
            spatial,
        )


def test_vector_lift_multiplication_uses_semantic_axis_alignment():
    axis = phx.discretization.FourierAxisSpec(16).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=True)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField(
        "v",
        representation="vector",
        components=2,
        coordinates=("x", "t"),
    )
    v = phx.equations.PDEExpression.field("v")
    coordinate = phx.equations.PDEExpression.coordinate_value("x")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        equations=(
            phx.equations.PDEEquation(
                "aligned-lift",
                v.derivative("t"),
                (coordinate * v).derivative("x"),
            ),
        ),
    )
    lift_value = jnp.stack(
        (
            1.0 + axis.nodes,
            2.0 - axis.nodes,
        ),
        axis=-1,
    )
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        spatial,
        boundary_lifts=(
            phx.equations.BoundaryLift(
                "v",
                lift_value,
                lift_id="vector-coordinate-lift",
            ),
        ),
        method="direct",
    )
    residual = jnp.zeros(spatial.state_shape + (2,))
    expected = spatial.partial_derivative(
        axis.nodes[..., None] * lift_value,
        axis=0,
    )
    assert jnp.allclose(compiled(0.0, residual, None), expected)

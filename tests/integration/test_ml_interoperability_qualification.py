#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""ML-interoperability qualification gates.

Each gate exercises one end-to-end contract between learned components and
Phydrax owners through public API only.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax.domain import SampleLayout
from phydrax.kernels import SquaredExponentialKernel
from phydrax.ml.kernel_methods import KernelRidgeRecipe
from phydrax.ml.linear import RidgeRecipe
from phydrax.ml.preprocessing import StandardScaler


# G1: statistical closure -> PDE ------------------------------------------------


def _conductivity_port():
    return phx.ValuePort(
        "thermal-conductivity",
        event_shape=(),
        component_ids=("kappa",),
        representation="coefficient-field",
        dimensions=(phx.units.DIMENSIONLESS,),
    )


def _fit_closure(recipe, space, slope):
    """Fit kappa(x) = 1 + slope * x on the owner's own coordinate port."""
    features = jnp.linspace(0.0, 1.0, 16)[:, None]
    return phx.ml.fit(
        recipe,
        features,
        1.0 + slope * features[:, 0],
        feature_schema=phx.ml.FeatureSchema.from_ports((space.value_port("x"),)),
        target_schema=phx.ml.TargetSchema.from_port(_conductivity_port()),
    )


def _input_mapping(model_port, owner_port):
    return phx.PortMapping(inputs=[(model_port.port_id, owner_port.port_id)])


def _heat_residual(kappa, u):
    """Steady heat residual -(kappa u')' - f with the source of kappa = 1 + x/2, u = x^2."""
    source = kappa.domain.Function("x")(lambda x: -(2.0 + 2.0 * x))
    flux = kappa * phx.operators.partial_n(u, var="x", order=1)
    return -phx.operators.partial_n(flux, var="x", order=1) - source


def _penalty(space, fields, residual):
    condition = phx.conditions.Residual(fields, space.component(), residual)
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(64),
    )
    return phx.terms.ResidualPenalty(condition, source)


def _array_leaves(tree):
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf)]


@pytest.mark.parametrize(
    "recipe",
    (
        RidgeRecipe(alpha=1e-10),
        KernelRidgeRecipe(SquaredExponentialKernel(length_scale=0.5), alpha=1e-6),
    ),
    ids=("ridge", "kernel-ridge"),
)
def test_g1_fitted_closure_keeps_ports_and_binds_frozen_into_a_pde(recipe):
    space = phx.domain.Interval1d(0.0, 1.0)
    x_port = space.value_port("x")
    result = _fit_closure(recipe, space, 0.5)

    ports = result.model_ports()
    assert ports.inputs == (x_port,)
    assert ports.outputs == (_conductivity_port(),)

    kappa = space.Model("x", port_mapping=_input_mapping(x_port, x_port))(result.model)
    evidence = kappa.port_binding
    assert evidence.inputs == ((x_port.port_id, x_port.port_id),)
    # The coordinate's semantic axis is declared on both sides and verified; the
    # domain declares no coordinate units, so dimensions stay unverified.
    assert ("input", x_port.port_id, "axes") not in evidence.unverified
    assert ("input", x_port.port_id, "dimensions") in evidence.unverified
    assert not evidence.dimensions_verified

    network = phx.nn.models.MLP(
        in_size="scalar", out_size="scalar", width_size=8, depth=1, key=jr.key(0)
    )
    u = space.Model("x")(network)
    solver = phx.solver.FunctionalSolver(
        functions={"kappa": kappa, "u": u},
        terms=(_penalty(space, ("kappa", "u"), _heat_residual),),
        evaluation_terms=(),
        enforcement=None,
    )
    parameters, _, fixed = solver.partition_functions()
    assert _array_leaves(parameters["kappa"]) == []
    assert _array_leaves(parameters["u"])

    trained = solver.solve(num_iter=5, optim=optax.adam(1e-2), seed=0)

    for before, after in zip(
        _array_leaves(fixed["kappa"]),
        _array_leaves(trained.partition_functions()[2]["kappa"]),
        strict=True,
    ):
        assert jnp.array_equal(before, after)
    assert not all(
        jnp.array_equal(before, after)
        for before, after in zip(
            _array_leaves(parameters["u"]),
            _array_leaves(trained.partition_functions()[0]["u"]),
            strict=True,
        )
    )


def test_g1_explicit_parameter_subspace_trains_a_coefficient_subset():
    space = phx.domain.Interval1d(0.0, 1.0)
    x_port = space.value_port("x")
    # The observed closure has the right intercept but a biased slope.
    result = _fit_closure(RidgeRecipe(alpha=1e-10), space, 0.2)
    kappa = space.Model("x", port_mapping=_input_mapping(x_port, x_port))(
        result.as_trainable()
    )
    u = space.Function("x")(lambda x: x**2)
    functions = {"kappa": kappa}
    solver = phx.solver.FunctionalSolver(
        functions=functions,
        terms=(_penalty(space, "kappa", lambda k: _heat_residual(k, u)),),
        evaluation_terms=(),
        enforcement=None,
    )
    paths = phx.nn.parameters.ParameterSubspace.array_leaf_paths(solver.functions)
    (slope_path,) = (path for path in paths if path.endswith("coefficients"))
    (intercept_path,) = (path for path in paths if path.endswith("intercept"))
    subspace = phx.nn.parameters.ParameterSubspace.from_leaf_paths(
        solver.functions, (slope_path,)
    )

    def leaf(tree, path):
        return dict(
            (jax.tree_util.keystr(key), value)
            for key, value in jax.tree_util.tree_flatten_with_path(tree)[0]
        )[path]

    loss_before = solver.loss(key=jr.key(1))
    trained = solver.solve(
        num_iter=200,
        optim=optax.adam(2e-2),
        parameter_subspace=subspace,
        seed=0,
    )
    before, after = solver.functions, trained.functions

    assert trained.loss(key=jr.key(1)) < 1e-2 * loss_before
    assert jnp.array_equal(leaf(after, intercept_path), leaf(before, intercept_path))
    assert jnp.allclose(leaf(after, slope_path), 0.5, atol=2e-2)
    # The trained closure still carries its port binding.
    assert trained.functions["kappa"].port_binding == kappa.port_binding


def test_g1_mismatched_port_mapping_is_rejected_before_execution():
    space_time = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    x_port, t_port = space_time.value_port("x"), space_time.value_port("t")
    features = jr.uniform(jr.key(0), (16, 2))
    result = phx.ml.fit(
        RidgeRecipe(alpha=1e-10),
        features,
        1.0 + 0.5 * features[:, 0] - 0.1 * features[:, 1],
        feature_schema=phx.ml.FeatureSchema.from_ports((x_port, t_port)),
        target_schema=phx.ml.TargetSchema.from_port(_conductivity_port()),
    )
    in_order = phx.PortMapping(
        inputs=[(x_port.port_id, x_port.port_id), (t_port.port_id, t_port.port_id)]
    )
    crossed = phx.PortMapping(
        inputs=[(x_port.port_id, t_port.port_id), (t_port.port_id, x_port.port_id)]
    )

    # Every rejection happens while binding, before any closure evaluation.
    with pytest.raises(ValueError, match="requires an explicit port_mapping"):
        space_time.Model("x", "t")(result.model)
    with pytest.raises(ValueError, match="semantic_id mismatch"):
        space_time.Model("x", "t", port_mapping=crossed)(result.model)
    # Dependencies are packed in `deps` order and never repacked to match features.
    with pytest.raises(ValueError, match="never repacked"):
        space_time.Model("t", "x", port_mapping=in_order)(result.model)
    bound = space_time.Model("x", "t", port_mapping=in_order)(result.model)
    assert bound.port_binding.inputs == (
        (x_port.port_id, x_port.port_id),
        (t_port.port_id, t_port.port_id),
    )


# G14: ensemble lanes -----------------------------------------------------------


class _NormalizedMember(eqx.Module):
    scaler: Any
    network: Any

    def __call__(self, x):
        return self.network(self.scaler(x))


def _member(index):
    shift = jnp.asarray([float(index), -float(index)])
    features = jr.normal(jr.key(10 + index), (32, 2)) * (1.0 + index) + shift
    scaler = phx.ml.fit(StandardScaler(), features).model
    network = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=8, depth=1, key=jr.key(index)
    )
    return _NormalizedMember(scaler, network)


def test_g14_ensemble_members_keep_fixed_normalizers_and_serial_equals_vmap():
    members = tuple(_member(index) for index in range(4))
    ensemble = phx.uq.HomogeneousFunctionEnsemble.from_members(members)
    layout = ensemble.layout
    stacked = ensemble.model
    inputs = jnp.asarray([0.3, -0.7])

    parameters, _, fixed = phx.partition_parameters(stacked)
    assert _array_leaves(parameters.scaler) == []
    assert _array_leaves(fixed.scaler)
    assert _array_leaves(parameters.network)
    # Every member keeps its own FIXED normalizer statistics on the member lane.
    scaler_paths = {
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(stacked)[0]
        if eqx.is_array(leaf) and jax.tree_util.keystr(path).startswith(".scaler")
    }
    assert scaler_paths <= set(layout.mapped_paths)
    for index, member in enumerate(members):
        for mine, stored in zip(
            _array_leaves(member.scaler),
            _array_leaves(layout.take(stacked, index).scaler),
            strict=True,
        ):
            assert jnp.array_equal(mine, stored)

    serial = jnp.stack([member(inputs) for member in members])
    vmapped = eqx.filter_vmap(
        lambda member, x: member(x), in_axes=(layout.in_axes(stacked), None)
    )(stacked, inputs)
    assert jnp.allclose(vmapped, serial, rtol=1e-12, atol=1e-12)

    # Training a lane moves parameters only; FIXED normalizers are not updated.
    parameters, model_state, fixed = phx.partition_parameters(stacked)

    def loss(parameters):
        model = phx.combine_parameters(parameters, model_state, fixed)
        outputs = eqx.filter_vmap(
            lambda member, x: member(x), in_axes=(layout.in_axes(model), None)
        )(model, inputs)
        return jnp.sum(outputs**2)

    gradient = eqx.filter_grad(loss)(parameters)
    assert _array_leaves(gradient.scaler) == []
    assert any(jnp.any(leaf != 0.0) for leaf in _array_leaves(gradient.network))


# G17: axis identity --------------------------------------------------------------


def _square_batch():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x") @ phx.domain.ScalarInterval(
        0.0, 1.0, label="y"
    )
    batch = domain.component().sample(
        phx.domain.PointSampling((3, 3), layout=SampleLayout((("x",), ("y",)))),
        key=jr.key(3),
    )
    axes = tuple(batch.structure.axis_for(label) for label in ("x", "y"))
    return domain, batch, axes


def _blockwise(**layout):
    return phx.domain.ModelBinding.blockwise("structured", pass_key=False, **layout)


def test_g17_equal_extents_never_create_axis_identity():
    domain, batch, (x_axis, y_axis) = _square_batch()

    def channels(values):
        x, y = values
        # Three channels over the 3 x 3 grid: width equals both spatial extents.
        grid = x[:, None] + 10.0 * y[None, :]
        return jnp.stack((grid, 2.0 * grid, 3.0 * grid), axis=-1)

    out = domain.Model("x", "y", binding=_blockwise())(channels)(batch)
    assert out.dims == (x_axis, y_axis, None)

    def x_profile(values):
        x, _ = values
        # Leading axis is x by declaration; the width-3 trailing axis is a channel.
        return jnp.stack((x, x**2, x**3), axis=-1)

    subset = _blockwise(output_layout="dependency_subset", output_labels=("x",))
    profile = domain.Model("x", "y", binding=subset)(x_profile)(batch)
    assert profile.dims == (x_axis, y_axis, None)
    # The channel axis was broadcast over y, not identified with y.
    assert jnp.allclose(
        jnp.asarray(profile.data)[:, 0, :], jnp.asarray(profile.data)[:, 2, :]
    )

    relabeled = _blockwise(output_layout="dependency_subset", output_labels=("y",))
    swapped = domain.Model("x", "y", binding=relabeled)(x_profile)(batch)
    # Declaration alone decides identity: the same raw array now indexes y.
    assert swapped.dims == (x_axis, y_axis, None)
    assert jnp.allclose(
        jnp.asarray(swapped.data)[0, :, :], jnp.asarray(swapped.data)[2, :, :]
    )


def test_g17_raw_output_is_validated_only_against_its_declaration():
    domain, batch, _ = _square_batch()

    def reduced(values):
        x, y = values
        return jnp.stack((jnp.sum(x), jnp.sum(y), jnp.sum(x * y)))

    # A reduction of width 3 is not the x axis just because the extent is 3.
    with pytest.raises(ValueError, match="declared leading axes"):
        domain.Model("x", "y", binding=_blockwise())(reduced)(batch)

    def named(values):
        x, y = values
        return phx.axes.AxisArray(x[:, None] + y[None, :], dims=("rows", "cols"))

    array_layout = _blockwise(output_layout="axis_array")
    with pytest.raises(ValueError, match="named dims must be dependency batch axes"):
        domain.Model("x", "y", binding=array_layout)(named)(batch)


def test_g17_ports_with_equal_extents_do_not_share_identity():
    channels = phx.ml.FeatureSchema.anonymous(3).value_ports()[0]
    position = phx.ValuePort(
        "x",
        event_shape=(3,),
        component_ids=("x[0]", "x[1]", "x[2]"),
        representation="domain-coordinates",
        axis_keys=(phx.axes.AxisKey("domain-label:x", "0"),),
    )
    time_series = phx.ValuePort(
        "x",
        event_shape=(3,),
        component_ids=("x[0]", "x[1]", "x[2]"),
        representation="domain-coordinates",
        axis_keys=(phx.axes.AxisKey("domain-label:t", "0"),),
    )
    owner = phx.ModelPorts(inputs=(position,), outputs=())
    with pytest.raises(ValueError, match="semantic_id mismatch"):
        phx.resolve_port_mapping(
            phx.ModelPorts(inputs=(channels,), outputs=()),
            owner,
            phx.PortMapping(inputs=[(channels.port_id, position.port_id)]),
        )
    with pytest.raises(ValueError, match="axes mismatch"):
        phx.resolve_port_mapping(
            phx.ModelPorts(inputs=(time_series,), outputs=()),
            owner,
            phx.PortMapping(inputs=[(time_series.port_id, position.port_id)]),
        )


# G11: regularity mismatch ------------------------------------------------------


def _smooth_derivative():
    return phx.DerivativeContract.smooth(
        (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
    )


class _PlanningOnly(phx.AbstractArrayModel):
    """Carries a network's contract and fails if anything evaluates it."""

    network: phx.AbstractArrayModel
    in_size: Any = eqx.field(static=True)
    out_size: Any = eqx.field(static=True)

    def __init__(self, network):
        self.network = network
        self.in_size = network.in_size
        self.out_size = network.out_size

    def __call__(self, x, /, *, key=None):
        raise AssertionError("The model was evaluated during planning.")

    def model_execution_contract(self):
        return self.network.model_execution_contract()


def _relu_network(final_activation=None):
    return phx.nn.models.MLP(
        in_size="scalar",
        out_size="scalar",
        width_size=16,
        depth=2,
        activation=jax.nn.relu,
        final_activation=final_activation,
        key=jr.key(0),
    )


def _poisson_solver(network, **options):
    space = phx.domain.Interval1d(0.0, 1.0)
    u = space.Model("x")(network)
    residual = _penalty(
        space, "u", lambda field: phx.operators.laplacian(field, var="x") + 2.0
    )
    return phx.solver.FunctionalSolver(functions={"u": u}, terms=(residual,), **options)


def test_g11_relu_linear_pinn_laplacian_is_rejected_at_planning():
    network = _PlanningOnly(_relu_network())

    with pytest.raises(ValueError, match="Order-2 .*regularity-degenerate"):
        _poisson_solver(network)
    # Acknowledging almost-everywhere derivatives never admits proven degeneracy.
    with pytest.raises(ValueError, match="regularity-degenerate"):
        _poisson_solver(
            network,
            regularity_policy=phx.RegularityPolicy(allow_almost_everywhere=True),
        )


def test_g11_relu_tanh_composite_needs_almost_everywhere_acknowledgment():
    network = _relu_network(final_activation=jnp.tanh)

    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        _poisson_solver(_PlanningOnly(network))

    solver = _poisson_solver(
        network, regularity_policy=phx.RegularityPolicy(allow_almost_everywhere=True)
    )
    (request,) = solver.derivative_requests
    assert request.order == 2
    assert request.admission.request.authority is phx.ComponentAuthority.SURROGATE
    assert request.admission.levels == (phx.GradientLevel.ALMOST_EVERYWHERE,)
    assert request.admission.conditions == ("singular-part-ignored",)

    trained = solver.solve(num_iter=3, optim=optax.adam(1e-3), seed=0)
    assert bool(jnp.isfinite(trained.loss()))


# G15: precision floor ----------------------------------------------------------


class _CubicResponse(phx.AbstractArrayModel):
    """Learned cubic response evaluated in the declared compute dtype."""

    precision: phx.ComponentPrecisionContract
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, precision):
        self.precision = precision
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        compute = jnp.dtype(self.precision.compute_dtype)
        return (x.astype(compute) ** 3).astype(x.dtype)

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=_smooth_derivative(),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=self.precision,
            randomness=phx.RandomnessContract("deterministic"),
        )


def _precision(compute_dtype, **floors):
    return phx.ComponentPrecisionContract(
        input_dtype="float64",
        parameter_dtype=compute_dtype,
        compute_dtype=compute_dtype,
        accumulation_dtype=compute_dtype,
        output_dtype="float64",
        cast_boundary_evidence=(f"float64-to-{compute_dtype}-input",),
        **floors,
    )


class _CubicResidual(eqx.Module):
    response: phx.AbstractArrayModel

    def __call__(self, state, target):
        return self.response(state) - target


def _newton(response, components, tolerance):
    return phx.nonlinear.NewtonKrylov().solve(
        phx.nonlinear.NonlinearSystemProblem(_CubicResidual(response)),
        jnp.ones(2),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=tolerance, relative_residual=0.0, maximum_steps=50
        ),
        args=jnp.asarray([2.0, 3.0]),
        precision=phx.nonlinear.NonlinearPrecisionPolicy(
            components=(
                phx.bind_component(model, authority).contract()
                for model, authority in components
            )
        ),
    )


@pytest.mark.parametrize(
    "authority",
    (phx.ComponentAuthority.SURROGATE, phx.ComponentAuthority.MODEL),
)
def test_g15_undeclared_float32_residual_component_rejects_newton(authority):
    response = _CubicResponse(_precision("float32"))

    with pytest.raises(ValueError, match="float32.*declares no error floor"):
        _newton(response, ((response, authority),), 1e-12)


def test_g15_declared_floor_derives_the_achievable_tolerance():
    response = _CubicResponse(_precision("float32", absolute_error_floor=1e-5))
    components = ((response, phx.ComponentAuthority.SURROGATE),)
    policy = phx.nonlinear.NonlinearPrecisionPolicy(
        components=(
            phx.bind_component(response, phx.ComponentAuthority.SURROGATE).contract(),
        )
    )
    floor = policy.residual_floor()
    assert floor == pytest.approx(1e-5)

    with pytest.raises(ValueError, match="below the declared component residual"):
        _newton(response, components, 1e-12)
    result = _newton(response, components, floor)
    assert bool(result.successful)
    exact = result.state**3 - jnp.asarray([2.0, 3.0])
    assert float(jnp.max(jnp.abs(exact))) <= floor


def test_g15_float32_accelerator_does_not_raise_the_residual_floor():
    physical = _CubicResponse(_precision("float64", absolute_error_floor=1e-13))
    preconditioner = _CubicResponse(_precision("float32", absolute_error_floor=1e-6))
    components = (
        (physical, phx.ComponentAuthority.MODEL),
        (preconditioner, phx.ComponentAuthority.ACCELERATOR),
    )
    policy = phx.nonlinear.NonlinearPrecisionPolicy(
        components=(
            phx.bind_component(model, authority).contract()
            for model, authority in components
        )
    )
    assert policy.residual_floor() == pytest.approx(1e-13)
    assert bool(_newton(physical, components, 1e-12).successful)

    # Authority, not dtype, decides: the same undeclared float32 contract is
    # rejected when it defines the residual and admitted as an accelerator.
    undeclared = _CubicResponse(_precision("float32"))
    admitted = _newton(
        physical,
        (
            (physical, phx.ComponentAuthority.MODEL),
            (undeclared, phx.ComponentAuthority.ACCELERATOR),
        ),
        1e-12,
    )
    assert bool(admitted.successful)
    with pytest.raises(ValueError, match="declares no error floor"):
        _newton(physical, ((undeclared, phx.ComponentAuthority.SURROGATE),), 1e-12)


# G16: frozen randomness ----------------------------------------------------------


class _NoisyResponse(phx.AbstractArrayModel):
    """Smooth response with additive noise drawn from its evaluation key."""

    gain: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, gain):
        self.gain = jnp.asarray(gain)
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        noise = jr.normal(key, jnp.shape(x), dtype=x.dtype)
        return self.gain * jnp.tanh(x) + 0.05 * noise

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=_smooth_derivative(),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("resampled"),
        )


class _StochasticResidual(eqx.Module):
    response: phx.AbstractArrayModel

    def __call__(self, state, target):
        return state + self.response(state, key=jr.key(11)) - target


def _implicit_state(response, target):
    return phx.nonlinear.implicit_root_result(
        phx.nonlinear.NonlinearSystemProblem(_StochasticResidual(response)),
        jnp.zeros(2),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-12, relative_residual=0.0
        ),
        args=target,
    )


def test_g16_stochastic_residual_is_rejected_on_certified_root_maps():
    problem = phx.nonlinear.NonlinearSystemProblem(
        _StochasticResidual(_NoisyResponse(0.5))
    )

    with pytest.raises(ValueError, match="resampled-randomness-not-admitted"):
        _implicit_state(_NoisyResponse(0.5), jnp.ones(2))
    with pytest.raises(ValueError, match="resampled-randomness-not-admitted"):
        phx.nonlinear.prepare_nonlinear(problem, jnp.zeros(2), args=jnp.ones(2))


def test_g16_frozen_realization_reproduces_primal_and_derivative():
    target = jnp.asarray([0.3, -0.2])
    frozen = phx.FrozenRealization(
        _NoisyResponse(0.5), jr.key(7), realization_id="noise-draw-7"
    )

    def root(values):
        return _implicit_state(frozen, values).state

    first = _implicit_state(frozen, target)
    second = _implicit_state(frozen, target)
    assert bool(first.successful)
    assert first.component_evidence == ("residual.response:fixed-realization",)
    assert jnp.array_equal(first.state, second.state)

    jacobian = jax.jacrev(root)(target)
    assert jnp.array_equal(jacobian, jax.jacrev(root)(target))
    step = 1e-6
    finite_difference = jnp.stack(
        [
            (root(target + step * basis) - root(target - step * basis)) / (2 * step)
            for basis in jnp.eye(2)
        ],
        axis=1,
    )
    assert jnp.allclose(jacobian, finite_difference, rtol=1e-6, atol=1e-8)

    redrawn = phx.FrozenRealization(
        _NoisyResponse(0.5), jr.key(8), realization_id="noise-draw-8"
    )
    assert not jnp.allclose(_implicit_state(redrawn, target).state, first.state)

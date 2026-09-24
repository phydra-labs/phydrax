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
from phydrax import model_state_field, parameter_field
from phydrax._training import ExponentialMovingAverageTargetPolicy
from phydrax._training_kernel import (
    BacktrackingLineSearchRule,
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    TrainingKernelSpec,
)
from phydrax._training_objective import _ObjectiveContribution
from phydrax.domain import SampleLayout
from phydrax.dynamics import (
    AbstractDiscretePlant,
    ArrayPyTreeSchema,
    ExecutableSignature,
    NumericRevision,
    PlantParameters,
    PlantProposal,
    PlantStepContext,
    SemanticProvenance,
)
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


# G2: learned constitutive -> implicit mechanics ---------------------------------


class _LearnedStress(phx.AbstractArrayModel):
    """Learned uniaxial stress `E * strain + k * strain**3` with a declared contract."""

    modulus: jax.Array
    stiffening: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, modulus, stiffening):
        self.modulus = jnp.asarray(modulus)
        self.stiffening = jnp.asarray(stiffening)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        return self.modulus * x + self.stiffening * x**3

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=_smooth_derivative(),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


_BAR_ELEMENTS = 4
_BAR_LENGTH = 1.0 / _BAR_ELEMENTS


def _uniaxial_port(name, unit):
    return phx.ValuePort(
        f"g2.{name}",
        event_shape=(1,),
        component_ids=(name,),
        representation="engineering",
        dimensions=(phx.units.parse_unit(unit).dimension,),
    )


def _learned_bar_law(model):
    return phx.equations.LearnedConstitutiveModel(
        model,
        _uniaxial_port("strain", "1"),
        _uniaxial_port("stress", "Pa"),
        lower=[-0.2],
        upper=[0.2],
        model_id="g2-learned-bar-law",
    )


def _bar_strain(displacement):
    nodal = jnp.concatenate((jnp.zeros(1), displacement))
    return ((nodal[1:] - nodal[:-1]) / _BAR_LENGTH)[:, None]


class _BarEquilibrium(eqx.Module):
    """Clamped bar internal-minus-external nodal force with a learned law."""

    law: phx.equations.AbstractConstitutiveModel

    def __call__(self, displacement, load):
        response = self.law.evaluate(
            _bar_strain(displacement), jnp.zeros((_BAR_ELEMENTS, 0)), None, 0.0, 1.0
        )
        stress = response.response[:, 0]
        internal = jnp.zeros(_BAR_ELEMENTS + 1).at[1:].add(stress).at[:-1].add(-stress)
        return internal[1:] - jnp.zeros(_BAR_ELEMENTS).at[-1].set(load)


def _bar_root(model, load):
    return phx.nonlinear.implicit_root_result(
        phx.nonlinear.NonlinearSystemProblem(_BarEquilibrium(_learned_bar_law(model))),
        jnp.zeros(_BAR_ELEMENTS),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-12, relative_residual=0.0
        ),
        args=load,
    )


def test_g2_native_newton_owns_acceptance_of_the_learned_law():
    accepted = _bar_root(_LearnedStress(1.0, 2.0), 0.1)
    assert bool(accepted.successful)
    assert float(jnp.max(jnp.abs(accepted.residual))) <= 1e-12
    assert accepted.component_evidence == ("residual.law.binding:deterministic",)
    law = _learned_bar_law(_LearnedStress(1.0, 2.0))
    at_root = law.evaluate(
        _bar_strain(accepted.state), jnp.zeros((_BAR_ELEMENTS, 0)), None, 0.0, 1.0
    )
    assert bool(jnp.all(at_root.header.eligible))

    # A load whose equilibrium leaves the learned support is never accepted.
    refused = _bar_root(_LearnedStress(1.0, 2.0), 0.5)
    assert not bool(refused.successful)
    # A law without classical C1 regularity cannot enter implicit mechanics.
    relu = phx.nn.models.MLP(
        in_size=1,
        out_size=1,
        width_size=8,
        depth=1,
        activation=jax.nn.relu,
        key=jr.key(0),
    )
    with pytest.raises(ValueError, match="implicit-requires-c1"):
        _learned_bar_law(relu)


def test_g2_implicit_parameter_gradient_matches_finite_differences():
    def tip(parameters):
        return _bar_root(_LearnedStress(parameters[0], parameters[1]), 0.1).state[-1]

    parameters = jnp.asarray([1.0, 2.0])
    gradient = jax.grad(tip)(parameters)
    step = 1e-6
    finite_difference = jnp.stack(
        [
            (tip(parameters + step * basis) - tip(parameters - step * basis))
            / (2.0 * step)
            for basis in jnp.eye(2)
        ]
    )
    assert jnp.allclose(gradient, finite_difference, rtol=1e-6, atol=1e-10)
    law = _learned_bar_law(_LearnedStress(1.0, 2.0))
    assert len(jax.tree_util.tree_leaves(phx.partition_parameters(law)[0])) == 2


def test_g2_invalid_tangent_poisons_the_derivative():
    equilibrium = _BarEquilibrium(_learned_bar_law(_LearnedStress(1.0, 2.0)))
    # Element 2 strains to 0.6, outside the learned support [-0.2, 0.2].
    displacement = jnp.asarray([0.05, 0.1, 0.25, 0.28])
    residual = equilibrium(displacement, 0.1)
    assert bool(jnp.all(jnp.isfinite(residual)))
    jacobian = jax.jacfwd(lambda value: equilibrium(value, 0.1))(displacement)
    coupled = jnp.asarray([False, True, True, False])
    assert bool(jnp.all(jnp.isnan(jacobian[coupled])))
    assert bool(jnp.all(jnp.isfinite(jacobian[~coupled])))

    # The parameter derivative of the equilibrium at that state is poisoned too.
    parameter_gradient = jax.grad(
        lambda parameters: _BarEquilibrium(
            _learned_bar_law(_LearnedStress(parameters[0], parameters[1]))
        )(displacement, 0.1)[1]
    )(jnp.asarray([1.0, 2.0]))
    assert bool(jnp.all(jnp.isnan(parameter_gradient)))


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


# G8 / G9 / G18: learned face closures in finite-volume owners -----------------


class _FaceGenerator(phx.StrictModule):
    """Unconstrained learned generator g(a, b, n) of a face correction."""

    network: phx.nn.models.MLP

    def __call__(self, system, left, right, baseline, context, args=None):
        del system, baseline, args
        features = jnp.concatenate((left, right, context.unit_normal), axis=-1)
        flat = features.reshape((-1, features.shape[-1]))
        return 0.05 * jax.vmap(self.network)(flat).reshape(left.shape)


def _learned_face_closure(dimension, key=0):
    components = dimension + 2
    generator = _FaceGenerator(
        phx.nn.models.MLP(
            in_size=2 * components + dimension,
            out_size=components,
            width_size=8,
            depth=1,
            key=jr.key(key),
        )
    )
    return phx.discretization.ArbitraryNormalFaceClosurePlan(
        phx.discretization.SymmetrizedFaceClosure(generator),
        closure_id="learned-face-closure",
    )


def _unit_grid(shape, *, periodic):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for count in shape
        ),
        axis_names=tuple("xy"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def _extrapolation():
    return phx.discretization.ExtrapolationBoundary()


def _structured_euler(shape, closure, *, mapped=False, periodic=False):
    system = phx.equations.EulerSystem(len(shape))
    discretization = phx.discretization.FiniteVolumePlan(
        _unit_grid(shape, periodic=periodic), component_names=system.component_names
    ).prepare()
    if mapped:
        discretization = phx.discretization.MappedFiniteVolumePlan(
            discretization, lambda point: point, mapping_id="identity"
        ).prepare()
    names = tuple("xy"[: len(shape)])
    boundaries = (
        phx.discretization.FiniteVolumeBoundarySet.periodic(names)
        if periodic
        else phx.discretization.FiniteVolumeBoundarySet(
            names,
            tuple(
                phx.discretization.FiniteVolumeBoundaryPair(
                    _extrapolation(), _extrapolation()
                )
                for _ in names
            ),
        )
    )
    problem = phx.equations.ConservationProblemIR(
        "g-closure", "state", system, boundaries
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
        closure=closure,
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics


def _quadrilateral_mesh(system, nx, ny):
    vertices = [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            upper_left = lower_left + nx + 1
            cells.append((lower_left, lower_left + 1, upper_left + 1, upper_left))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        jnp.asarray(vertices),
        quadrilaterals=jnp.asarray(cells, dtype=jnp.int32),
        component_names=system.component_names,
    )


def _unstructured_euler(
    nx, ny, closure, *, interface_solver=None, motion=None, motion_id="static"
):
    system = phx.equations.EulerSystem(2)
    plan = _quadrilateral_mesh(system, nx, ny)
    discretization = plan.prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {name: _extrapolation() for name in discretization.boundary_patch_names},
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan()
        if interface_solver is None
        else interface_solver,
        closure=closure,
    )
    coupling = (
        None
        if motion is None
        else phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
            motion=phx.discretization.FixedConnectivityMotionPlan(
                plan, motion, mapping_id=motion_id
            )
        )
    )
    problem = phx.equations.ConservationProblemIR(
        f"g-closure-unstructured:{motion_id}", "state", system, boundaries
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method, coupling=coupling
    ).dynamics


def _euler_state(shape):
    system = phx.equations.EulerSystem(2)
    x = jnp.linspace(0.0, 1.0, shape[0] * shape[1]).reshape(shape)
    primitive = jnp.stack(
        (1.0 + 0.2 * jnp.sin(3.0 * x), 0.2 * x, -0.1 * x**2, 1.0 + 0.1 * x), axis=-1
    )
    return system.primitive_to_conserved(primitive)


def test_g8_plan_embedded_closure_trains_inside_prepared_dynamics():
    reference = phx.discretization.ArbitraryNormalFaceClosurePlan(
        lambda system, left, right, baseline, context, args: 0.03 * (right - left),
        closure_id="reference-jump-dissipation",
    )
    target_dynamics = _structured_euler((16,), reference, periodic=True)
    dynamics = _structured_euler((16,), _learned_face_closure(1), periodic=True)
    system = dynamics.system
    x = dynamics.discretization.cell_centers[..., 0]
    initial = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x),
                0.1 * jnp.cos(2.0 * jnp.pi * x),
                jnp.ones_like(x),
            ),
            axis=-1,
        )
    )

    def rollout(model):
        state = initial
        for _ in range(4):
            state = state + 2e-3 * model(0.0, state)
        return state

    target = rollout(target_dynamics)
    parameters, model_state, fixed = phx.partition_parameters(dynamics)
    network = dynamics.method.closure.correction.generator.network
    network_leaves = jax.tree.leaves(phx.partition_parameters(network)[0])
    assert len(jax.tree.leaves(parameters)) == len(network_leaves)
    assert all(
        leaf is expected
        for leaf, expected in zip(
            jax.tree.leaves(parameters), network_leaves, strict=True
        )
    )
    phx.require_parameter_roles(dynamics, context="G8 plan-embedded closure")

    def loss(parameters):
        model = phx.combine_parameters(parameters, model_state, fixed)
        return jnp.sum((rollout(model) - target) ** 2)

    optimizer = optax.adam(3e-2)

    @jax.jit
    def step(parameters, optimizer_state):
        value, gradient = jax.value_and_grad(loss)(parameters)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state)
        return optax.apply_updates(parameters, updates), optimizer_state, value

    optimizer_state = optimizer.init(parameters)
    trained = parameters
    first = None
    for _ in range(25):
        trained, optimizer_state, value = step(trained, optimizer_state)
        first = value if first is None else first
    final = loss(trained)

    assert float(final) < 0.5 * float(first)
    model = phx.combine_parameters(trained, model_state, fixed)
    _, _, fixed_after = phx.partition_parameters(model)
    for before, after in zip(
        jax.tree.leaves(fixed), jax.tree.leaves(fixed_after), strict=True
    ):
        assert before is after
    assert model.dynamics_id == dynamics.dynamics_id


def test_g9_one_face_closure_serves_cartesian_mapped_and_unstructured_owners():
    closure = _learned_face_closure(2)
    grid_state = _euler_state((4, 3))
    contributions = {
        name: _structured_euler((4, 3), closure, mapped=mapped)(0.0, grid_state)
        - _structured_euler((4, 3), None, mapped=mapped)(0.0, grid_state)
        for name, mapped in (("cartesian", False), ("mapped", True))
    }
    # Unstructured cell j * nx + i is structured cell (i, j).
    cell_state = jnp.swapaxes(grid_state, 0, 1).reshape((-1, 4))
    unstructured = _unstructured_euler(4, 3, closure)(0.0, cell_state) - (
        _unstructured_euler(4, 3, None)(0.0, cell_state)
    )

    assert float(jnp.max(jnp.abs(contributions["cartesian"]))) > 1e-5
    assert jnp.allclose(contributions["mapped"], contributions["cartesian"], atol=1e-12)
    assert jnp.allclose(
        unstructured,
        jnp.swapaxes(contributions["cartesian"], 0, 1).reshape((-1, 4)),
        atol=1e-12,
    )


class _RecordingClosure(phx.StrictModule):
    """Zero correction that records every context it receives (eager only)."""

    contexts: list = eqx.field(static=True)

    def __call__(self, system, left, right, baseline, context, args=None):
        self.contexts.append(context)
        return jnp.zeros_like(baseline)


def _deformation(time, vertices, args):
    del args
    return vertices.at[4, 0].add(0.15 * time)


def test_g18_ale_closure_receives_exact_geometry_and_cartesian_parity():
    recorder = _RecordingClosure([])
    closure = phx.discretization.ArbitraryNormalFaceClosurePlan(
        recorder, closure_id="recording-closure"
    )
    dynamics = _unstructured_euler(
        2,
        2,
        closure,
        interface_solver=phx.discretization.HLLCFluxPlan(),
        motion=_deformation,
        motion_id="g18-deformation",
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics, phx.discretization.FluxPositivityPlan()
    )
    system = dynamics.system
    uniform = system.primitive_to_conserved(
        jnp.broadcast_to(jnp.asarray((1.0, 0.2, -0.1, 1.0)), (4, 4))
    )
    initial = runtime.initialize_state(uniform, 0.0, 2e-2)
    stage = runtime.advance(initial).ale.geometry.stage_1
    recorder.contexts.clear()
    dynamics.evaluate_stage(initial.content_state, stage)

    (block,) = stage.face_blocks
    (context,) = recorder.contexts
    active = context.active
    expected_normal = jnp.broadcast_to(
        (block.area_vectors / block.face_measures[:, None])[:, None, :],
        context.unit_normal.shape,
    )
    expected_measure = jnp.broadcast_to(
        block.face_measures[:, None], context.face_measure.shape
    )
    assert bool(jnp.any(active))
    assert jnp.array_equal(context.unit_normal[active], expected_normal[active])
    assert jnp.array_equal(context.face_measure[active], expected_measure[active])
    assert jnp.array_equal(
        context.grid_normal_velocity, block.quadrature_grid_normal_velocity
    )
    assert float(jnp.max(jnp.abs(context.grid_normal_velocity))) > 0.0
    assert context.geometry_id == stage.geometry_layout_id

    recorder.contexts.clear()
    cartesian = _structured_euler((3, 2), closure)
    baseline = _structured_euler((3, 2), None)
    state = _euler_state((3, 2))
    assert jnp.array_equal(cartesian(0.0, state), baseline(0.0, state))
    for axis, context in enumerate(recorder.contexts):
        assert context.axis == axis
        assert jnp.array_equal(
            context.unit_normal,
            jnp.broadcast_to(jnp.eye(2)[axis], context.unit_normal.shape),
        )
        assert jnp.array_equal(
            context.face_measure, cartesian.discretization.face_measures[axis]
        )
        assert not bool(jnp.any(context.grid_normal_velocity))


# G12 / G24: in-situ coupled training --------------------------------------------


class _DriftPlant(AbstractDiscretePlant):
    """x <- x + gain * command; a negative command is a physical rejection."""

    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: dict
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)

    def __init__(self):
        semantic = SemanticProvenance({"kind": "g12-drift-plant"})
        self.state_schema = ArrayPyTreeSchema.from_tree(
            {"x": jnp.zeros((1,))}, case_ndim=1
        )
        self.control_schema = ArrayPyTreeSchema.from_tree(jnp.zeros((1,)), case_ndim=1)
        self.parameter_schema = ArrayPyTreeSchema.from_tree(
            {"gain": jnp.asarray(1.0)}, case_ndim=0
        )
        self.reset_fallback = {"x": jnp.asarray(0.0)}
        self.semantic_provenance = semantic
        self.numeric_revision = NumericRevision(semantic, {"gain": jnp.asarray(2.0)})
        self.execution_signature = ExecutableSignature(
            shapes={"x": ()}, algorithm_facts={"method": "drift"}
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True

    def propose_reset(self, keys, parameters, /, *, case_shape, initial_time):
        del keys, parameters, initial_time
        state = {"x": jnp.zeros(case_shape)}
        ok = jnp.ones(case_shape, dtype=jnp.bool_)
        status = jnp.zeros(case_shape, dtype=jnp.int32)
        return PlantProposal(state, state, ok, ok, status, status, None)

    def propose_step(self, context, source, commands, parameters, keys, /):
        del context, keys
        successful = commands >= 0.0
        status = jnp.where(successful, 0, 37).astype(jnp.int32)
        candidate = {"x": source["x"] + parameters["gain"] * commands}
        attempted = jnp.ones(successful.shape, dtype=jnp.bool_)
        return PlantProposal(
            candidate, candidate, attempted, successful, status, status, None
        )


class _DriftEstimator(phx.StrictModule):
    rate: jax.Array = parameter_field()
    updates: jax.Array = model_state_field()


def _drift_error(parameters, model_state, fixed, payload, keys):
    del fixed, keys
    residual = payload["delta"] - parameters.rate
    next_state = eqx.tree_at(
        lambda state: state.updates, model_state, model_state.updates + 1.0
    )
    contribution = _ObjectiveContribution(jnp.sum(residual**2), residual.size)
    return contribution, next_state, {}


def _observed_drift(source, step):
    return {"delta": step.candidate_state.payload["x"] - source.payload["x"]}


def _drift_setup(rule, **spec_options):
    plant = _DriftPlant()
    parameters = PlantParameters(
        {"gain": jnp.asarray(2.0)},
        plant.parameter_schema.schema_id,
        plant.numeric_revision,
    )
    plant_state = plant.reset(
        jr.split(jr.key(1), 2), parameters, case_shape=(2,)
    ).accepted_state
    tree = _DriftEstimator(jnp.asarray(0.0), jnp.asarray(0.0))
    kernel = prepare_training_kernel(
        tree,
        (
            KernelObjective(
                objective_id="drift",
                kind=phx.ObjectiveKind.DATA_FIT,
                route=phx.DerivativeRoute.DIRECT,
                fn=_drift_error,
            ),
        ),
        TrainingKernelSpec(
            rule, context="in-situ drift", rejection_budget=4, **spec_options
        ),
        root_authority=phx.ComponentAuthority.MODEL,
    )
    return plant, parameters, plant_state, kernel, kernel.init(tree, jr.key(0))


def _coupled(setup, plant_state, kernel_state, commands, policy, hooks=()):
    plant, parameters, _, kernel, _ = setup
    return phx.lifecycle.coupled_training_step(
        plant,
        plant_state,
        kernel,
        kernel_state,
        _observed_drift,
        policy,
        context=PlantStepContext(
            plant_state.time, plant_state.time + 1.0, plant_state.step_index
        ),
        commands=jnp.asarray(commands),
        plant_parameters=parameters,
        hooks=hooks,
    )


def _assert_bitwise(actual, expected):
    def data(leaf):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            return jr.key_data(leaf)
        return leaf

    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        assert jnp.array_equal(data(left), data(right))


@pytest.mark.parametrize("policy", list(phx.lifecycle.CoupledTrainingPolicy))
def test_g12_physical_rejection_restores_every_training_quantity_exactly(policy):
    setup = _drift_setup(
        OptaxUpdateRule(optax.adam(0.1), rule_id="adam"),
        target_policy=ExponentialMovingAverageTargetPolicy(decay=0.5),
    )
    _, _, plant_state, _, kernel_state = setup
    # One committed joint step makes optimizer moments, targets, model state,
    # and cursors nontrivial before the rejected step.
    warm = _coupled(setup, plant_state, kernel_state, [1.0, 1.0], policy)
    assert int(warm.kernel_state.accepted_cursor) == 1

    hooks = []
    rejected = _coupled(
        setup,
        warm.plant_state,
        warm.kernel_state,
        [1.0, -1.0],
        policy,
        hooks=(lambda *args: hooks.append(args),),
    )
    # The derived update was acceptable on its own, yet it was discarded with
    # the rejected physical step: nothing of the attempt survives.
    assert int(rejected.evidence.training.outcome) == 0
    assert not bool(rejected.evidence.training_committed)
    _assert_bitwise(rejected.kernel_state, warm.kernel_state)
    assert hooks == []
    x = rejected.plant_state.payload["x"]
    assert float(x[1]) == float(warm.plant_state.payload["x"][1])
    if policy is phx.lifecycle.CoupledTrainingPolicy.JOINTLY_REQUIRED:
        _assert_bitwise(rejected.plant_state, warm.plant_state)
    else:
        assert float(x[0]) == float(warm.plant_state.payload["x"][0]) + 2.0


def test_g24_policy_decides_whether_a_valid_step_survives_a_training_rejection():
    # One huge trial step: the line search rejects finitely and is authorized to
    # commit only its shrunken step size.
    setup = _drift_setup(BacktrackingLineSearchRule(initial_step=100.0, max_trials=1))
    _, _, plant_state, _, kernel_state = setup
    outcomes = {}
    for policy in phx.lifecycle.CoupledTrainingPolicy:
        result = _coupled(setup, plant_state, kernel_state, [1.0, 1.0], policy)
        outcomes[policy] = result
        assert int(result.evidence.training.outcome) == 1
        assert bool(result.evidence.training_committed)
        _assert_bitwise(result.kernel_state.parameters, kernel_state.parameters)
        _assert_bitwise(result.kernel_state.model_state, kernel_state.model_state)
        assert float(result.kernel_state.rule_state.step_size) == 50.0
        assert int(result.kernel_state.accepted_cursor) == 0
        assert int(result.kernel_state.attempt_cursor) == 1

    may_commit = outcomes[phx.lifecycle.CoupledTrainingPolicy.PHYSICAL_MAY_COMMIT]
    jointly = outcomes[phx.lifecycle.CoupledTrainingPolicy.JOINTLY_REQUIRED]
    assert jnp.array_equal(may_commit.plant_state.payload["x"], jnp.asarray([2.0, 2.0]))
    assert jnp.array_equal(
        may_commit.evidence.physical_committed, jnp.asarray([True, True])
    )
    _assert_bitwise(jointly.plant_state, plant_state)
    assert not bool(jnp.any(jointly.evidence.physical_committed))

    # With an acceptable update both policies commit plant and parameters together.
    accepting = _drift_setup(BacktrackingLineSearchRule(initial_step=0.5, max_trials=4))
    _, _, plant_state, _, kernel_state = accepting
    for policy in phx.lifecycle.CoupledTrainingPolicy:
        result = _coupled(accepting, plant_state, kernel_state, [1.0, 1.0], policy)
        assert int(result.evidence.training.outcome) == 0
        assert float(result.kernel_state.parameters.rate) > 0.0
        assert jnp.array_equal(result.plant_state.payload["x"], jnp.asarray([2.0, 2.0]))


# G8 (training) / G20: closures train through rollout objectives -----------------


class _PreparedRollout(eqx.Module):
    """FIXED prepared finite-volume dynamics and the rollout it is scored on."""

    dynamics: Any
    initial: jax.Array
    target: jax.Array


def _bind_face_closure(solve, closure):
    """Rebind a trained closure into the FIXED prepared dynamics' closure slot."""
    if closure.closure_id != solve.dynamics.method.closure.closure_id:
        raise ValueError("The trained closure must keep the prepared slot identity.")
    return eqx.tree_at(lambda dynamics: dynamics.method.closure, solve.dynamics, closure)


def _euler_rollout(dynamics, state):
    for _ in range(4):
        state = state + 2e-3 * dynamics(0.0, state)
    return state


def _rollout_misfit(owner, case):
    del case
    dynamics, solve = owner
    final = _euler_rollout(dynamics, solve.initial)
    return phx.solver.SolverCaseResult(
        residual=final - solve.target, accepted=jnp.all(jnp.isfinite(final))
    )


def _closure_rollout_objective(component=None):
    prepared = _structured_euler((16,), _learned_face_closure(1), periodic=True)
    reference = _structured_euler(
        (16,),
        phx.discretization.ArbitraryNormalFaceClosurePlan(
            lambda system, left, right, baseline, context, args: 0.03 * (right - left),
            closure_id="reference-jump-dissipation",
        ),
        periodic=True,
    )
    x = prepared.discretization.cell_centers[..., 0]
    initial = prepared.system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x),
                0.1 * jnp.cos(2.0 * jnp.pi * x),
                jnp.ones_like(x),
            ),
            axis=-1,
        )
    )
    return phx.solver.RolloutObjective(
        _PreparedRollout(prepared, initial, _euler_rollout(reference, initial)),
        lambda solve, closure: (_bind_face_closure(solve, closure), solve),
        _rollout_misfit,
        objective_id="face-closure-rollout",
        component=component,
    )


def test_g8_closure_trains_through_a_rollout_objective_bound_into_fixed_dynamics():
    closure = _learned_face_closure(1)
    objective = _closure_rollout_objective()
    before = objective.evaluate(closure)
    assert before.trained and all(
        authority == "discretization" for _, authority in before.trained
    )

    result = phx.solver.train_components(
        closure, (objective,), optimizer=optax.adam(3e-2), steps=25, key=jr.key(8)
    )
    after = objective.evaluate(result.tree)
    assert float(after.value) < 0.5 * float(before.value)
    network = result.tree.correction.generator.network
    assert result.authorities == tuple(
        (path, "discretization") for path, _ in before.trained
    )
    assert len(result.authorities) == len(
        jax.tree.leaves(phx.partition_parameters(network)[0])
    )


# G3 / G10: learned preconditioners train only through fixed Krylov work ---------


_KRYLOV_SIZE = 48
_KRYLOV_WORK = 8


def _scaled_diagonal(contrast, seed):
    """Badly scaled diagonal `10**u`, `u` uniform on `[-contrast, contrast]`."""
    exponents = jr.uniform(
        jr.key(seed), (_KRYLOV_SIZE,), minval=-contrast, maxval=contrast
    )
    return 10.0**exponents


def _scaled_system(diagonal):
    """Symmetric `D^{1/2} (I + 0.3 T) D^{1/2}` with `T` the unit tridiagonal coupling."""
    root = jnp.sqrt(diagonal)
    coupling = jnp.diag(jnp.ones(_KRYLOV_SIZE - 1), 1)
    matrix = (
        root[:, None]
        * (jnp.eye(_KRYLOV_SIZE) + 0.3 * (coupling + coupling.T))
        * root[None, :]
    )
    space = phx.linalg.ArraySpace((_KRYLOV_SIZE,), dtype=matrix.dtype)
    return phx.linalg.LinearSystem(
        phx.linalg.DenseLinearOperator(matrix, source=space, target=space)
    )


class _LearnedScalingPreconditioner(phx.linalg.AbstractPreconditioner):
    """Learned local scaling from log10-diagonal features, supported on [-2, 2]."""

    network: phx.nn.models.MLP
    features: jax.Array = phx.fixed_field()

    def __init__(self, network, diagonal):
        self.network = network
        self.features = jnp.log10(diagonal)[:, None]
        self.space = phx.linalg.ArraySpace((_KRYLOV_SIZE,), dtype=jnp.float64)
        self.properties = phx.linalg.PreconditionerProperties(
            stationary=True, evidence={"stationary": "construction"}
        )
        self.preconditioner_id = "g3-learned-scaling"

    def apply(self, residual, /, *, iteration=None):
        del iteration
        scale = jnp.exp(jax.vmap(self.network)(self.features))
        supported = jnp.all(jnp.abs(self.features) <= 2.0)
        # Outside its training support the learned action is not a number.
        return self.space.validate(jnp.where(supported, scale * residual, jnp.nan))


def _scaling_network(seed=0):
    return phx.nn.models.MLP(
        in_size=1, out_size="scalar", width_size=16, depth=2, key=jr.key(seed)
    )


class _KrylovSolve(eqx.Module):
    """FIXED prepared problem of a fixed-work preconditioned FGMRES solve."""

    system: Any


def _fixed_work_policy(preconditioner):
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=_KRYLOV_WORK),
        preconditioning=phx.linalg.PreconditioningPolicy(preconditioner, side="right"),
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0, absolute=0.0, max_steps=_KRYLOV_WORK
        ),
        differentiation=phx.linalg.DifferentiationPolicy("algorithmic"),
        failure=phx.linalg.FailurePolicy("status"),
    )


def _fixed_work_measure(owner, rhs):
    system, policy = owner
    result = phx.linalg.solve(system, rhs, policy=policy)
    return phx.solver.AlgorithmicWorkResult(
        initial_residual=rhs,
        final_residual=rhs - system.operator.mv(result.value),
        iterations=result.diagnostics.iterations,
        accepted=jnp.all(jnp.isfinite(result.value)),
    )


def _krylov_work_objective(diagonal, component=None):
    return phx.solver.AlgorithmicWorkObjective(
        _KrylovSolve(_scaled_system(diagonal)),
        lambda solve, preconditioner: (
            solve.system,
            _fixed_work_policy(preconditioner),
        ),
        _fixed_work_measure,
        work=_KRYLOV_WORK,
        objective_id="fgmres-fixed-work",
        cases=jr.normal(jr.key(21), (6, _KRYLOV_SIZE)),
        component=component,
    )


def _production_solve(system, preconditioner, rhs):
    return phx.linalg.solve(
        system,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.FGMRES(restart=_KRYLOV_SIZE),
            preconditioning=phx.linalg.PreconditioningPolicy(
                preconditioner, side="right"
            ),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-10, absolute=0.0, max_steps=3 * _KRYLOV_SIZE
            ),
            failure=phx.linalg.FailurePolicy("status"),
        ),
    )


def _never_bound(*_):
    raise AssertionError("a refused objective must not bind or measure")


def test_g10_solution_map_only_accelerator_is_refused_before_tracing():
    diagonal = _scaled_diagonal(2.0, 3)
    preconditioner = _LearnedScalingPreconditioner(_scaling_network(), diagonal)
    refused = phx.solver.SolverObjective(
        _KrylovSolve(_scaled_system(diagonal)),
        _never_bound,
        _never_bound,
        objective_id="solution-map-only",
    )
    with pytest.raises(ValueError, match=r"no admissible training signal.*accelerator"):
        refused.evaluate(preconditioner)
    with pytest.raises(ValueError, match=r"no admissible training signal.*accelerator"):
        phx.solver.train_components(
            preconditioner,
            (refused,),
            optimizer=optax.adam(1e-2),
            steps=1,
            key=jr.key(0),
        )

    work = _krylov_work_objective(diagonal)
    evaluation = work.evaluate(preconditioner)
    assert evaluation.work.tolist() == [_KRYLOV_WORK] * 6
    gradient = eqx.filter_grad(lambda tree: work.evaluate(tree).value)(preconditioner)
    norms = [float(jnp.linalg.norm(leaf)) for leaf in _array_leaves(gradient.network)]
    assert max(norms) > 0.0
    assert all(jnp.isfinite(jnp.asarray(norms)))


def test_g3_learned_preconditioner_reduces_fixed_krylov_work_without_changing_answers():
    diagonal = _scaled_diagonal(2.0, 3)
    system = _scaled_system(diagonal)
    untrained = _LearnedScalingPreconditioner(_scaling_network(), diagonal)
    objective = _krylov_work_objective(diagonal)

    result = phx.solver.train_components(
        untrained, (objective,), optimizer=optax.adam(3e-2), steps=80, key=jr.key(5)
    )
    trained = result.tree
    before = objective.evaluate(untrained)
    after = objective.evaluate(trained)
    assert result.accepted_updates == 80
    assert float(after.value) < float(before.value) - 0.5
    assert result.selection == (tuple(path for path, _ in before.trained),)

    rhs = jr.normal(jr.key(99), (_KRYLOV_SIZE,))
    reference = phx.linalg.solve(
        system, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    ).value
    slow = _production_solve(system, untrained, rhs)
    fast = _production_solve(system, trained, rhs)
    assert bool(fast.successful)
    assert int(fast.diagnostics.iterations) < int(slow.diagnostics.iterations)
    # Same answer up to the conditioning of the badly scaled system.
    assert float(jnp.linalg.norm(fast.value - reference)) <= 1e-5 * float(
        jnp.linalg.norm(reference)
    )
    for solved in (slow, fast):
        true_residual = jnp.linalg.norm(rhs - system.operator.mv(solved.value))
        assert not bool(solved.successful) or float(true_residual) <= 1e-9 * float(
            jnp.linalg.norm(rhs)
        )

    # Outside its support the learned action is not a number: the native solve
    # re-evaluates the original residual and cannot report success.
    outside = _scaled_diagonal(4.0, 4)
    shifted = _LearnedScalingPreconditioner(trained.network, outside)
    rejected = _production_solve(_scaled_system(outside), shifted, rhs)
    assert not bool(rejected.successful)


# G20: mixed authority needs one compatible contribution per group ----------------


def test_g20_closure_gets_rollout_signal_and_preconditioner_gets_fixed_work_signal():
    diagonal = _scaled_diagonal(2.0, 3)
    tree = {
        "closure": _learned_face_closure(1),
        "preconditioner": _LearnedScalingPreconditioner(_scaling_network(), diagonal),
    }
    rollout = _closure_rollout_objective(component=lambda value: value["closure"])
    work = _krylov_work_objective(
        diagonal, component=lambda value: value["preconditioner"]
    )

    with pytest.raises(ValueError, match="no admissible training signal for accelerator"):
        phx.solver.train_components(
            tree, (rollout,), optimizer=optax.adam(3e-2), steps=1, key=jr.key(6)
        )
    result = phx.solver.train_components(
        tree, (rollout, work), optimizer=optax.adam(3e-2), steps=12, key=jr.key(6)
    )

    closure_paths, preconditioner_paths = result.selection
    assert closure_paths and all(path.startswith("['closure']") for path in closure_paths)
    assert preconditioner_paths and all(
        path.startswith("['preconditioner']") for path in preconditioner_paths
    )
    assert {authority for _, authority in result.authorities} == {
        "discretization",
        "accelerator",
    }
    first, last = result.objective_values[0], result.objective_values[-1]
    assert float(last[0]) < float(first[0])
    assert float(last[1]) < float(first[1])


# G5 (rollout): neural operator proposals corrected by a native Newton solve -----


class _ImplicitEulerStep(eqx.Module):
    """Native residual `v - u + dt (v^3 + v)` of one implicit Euler step."""

    step: float = eqx.field(static=True)

    def __call__(self, state, previous):
        return state - previous + self.step * (state**3 + state)


class _CorrectedRollout(eqx.Module):
    proposal: phx.ComponentBinding
    step: float = eqx.field(static=True)


def _corrected_rollout(owner, initial):
    problem = phx.nonlinear.NonlinearSystemProblem(_ImplicitEulerStep(owner.step))
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-10, relative_residual=0.0, maximum_steps=50
    )
    state = initial
    proposals, corrected, residuals, accepted = [], [], [], jnp.asarray(True)
    for _ in range(6):
        proposal = state + owner.proposal.model(state)
        result = phx.nonlinear.NewtonKrylov().solve(
            problem,
            jax.lax.stop_gradient(proposal),
            termination=termination,
            args=state,
        )
        proposals.append(proposal)
        corrected.append(result.state)
        residuals.append(_ImplicitEulerStep(owner.step)(result.state, state))
        accepted = accepted & result.successful
        state = jax.lax.stop_gradient(result.state)
    proposals, corrected = jnp.stack(proposals), jnp.stack(corrected)
    return phx.solver.SolverCaseResult(
        residual=proposals - jax.lax.stop_gradient(corrected),
        accepted=accepted,
        aux={
            "proposal": proposals,
            "corrected": corrected,
            "residual": jnp.stack(residuals),
        },
    )


def test_g5_operator_proposals_stay_inspectable_beside_native_corrections():
    objective = phx.solver.RolloutObjective(
        None,
        lambda solve, proposal: _CorrectedRollout(proposal, 0.2),
        _corrected_rollout,
        objective_id="corrected-rollout",
        cases=jnp.asarray([[0.8, -0.5], [0.3, 0.9], [-0.6, 0.2]]),
    )
    operator = phx.bind_component(
        phx.nn.models.MLP(in_size=2, out_size=2, width_size=16, depth=1, key=jr.key(2)),
        phx.ComponentAuthority.SURROGATE,
    )

    before = objective.evaluate(operator)
    result = phx.solver.train_components(
        operator, (objective,), optimizer=optax.adam(2e-2), steps=60, key=jr.key(7)
    )
    after = objective.evaluate(result.tree)

    assert bool(jnp.all(before.accepted)) and bool(jnp.all(after.accepted))
    assert float(after.value) < 0.2 * float(before.value)
    # The corrector owns the answer: it re-evaluated the native residual, and
    # the corrected trajectory does not depend on the proposal quality.
    for evaluation in (before, after):
        assert float(jnp.max(jnp.abs(evaluation.aux["residual"]))) <= 1e-10
        assert evaluation.aux["proposal"].shape == evaluation.aux["corrected"].shape
    assert jnp.allclose(before.aux["corrected"], after.aux["corrected"], atol=1e-9)
    assert not jnp.allclose(before.aux["proposal"], before.aux["corrected"], atol=1e-3)


# G4: neural dynamics -> control (MPC sensitivity through a learned linearization) -


class _LearnedDiscreteDynamics(phx.AbstractArrayModel):
    """Learned step `x + dt * (W x + tanh(V x) + g u)` on the (state, control) port."""

    drift: jax.Array
    coupling: jax.Array
    input_gain: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, drift, coupling, input_gain):
        self.drift = jnp.asarray(drift)
        self.coupling = jnp.asarray(coupling)
        self.input_gain = jnp.asarray(input_gain)
        self.in_size = 3
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        state, control = x[:2], x[2:]
        rate = (
            self.drift @ state
            + jnp.tanh(self.coupling @ state)
            + self.input_gain * control[0]
        )
        return state + 0.1 * rate

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=_smooth_derivative(),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


_G4_HORIZON = 4
_G4_TIMES = phx.dynamics.TimeGrid(
    0.1 * jnp.arange(_G4_HORIZON + 1, dtype=jnp.float64), time_id="g4-mpc-time"
)
_G4_DENSE = phx.linalg.MaterializationPolicy(max_entries=256)
_G4_QP = phx.optim.ConvexSolvePolicy(
    termination=phx.optim.ConvexTermination(
        absolute=1e-12, relative=1e-12, maximum_steps=200
    )
)
_G4_INITIAL = jnp.asarray([1.0, -0.5])


def _g4_learned_plant():
    """The learned model is the transition; its parameters travel as ``args``."""
    return phx.control.DiscreteControlDynamics(
        phx.dynamics.DiscreteSystem(
            lambda context, state, control, model: model(
                jnp.concatenate((state, control))
            ),
            state_layout=phx.dynamics.StateLayout((2,)),
            input_layout=phx.dynamics.InputLayout((1,), roles="control"),
            system_id="g4-learned-plant",
        )
    )


def _g4_operating_trajectory(model):
    """Nominal zero-control rollout of the learned model (stages 0..H-1)."""
    states = [_G4_INITIAL]
    for _ in range(_G4_HORIZON - 1):
        states.append(model(jnp.concatenate((states[-1], jnp.zeros(1)))))
    return jax.lax.stop_gradient(jnp.stack(states)), jnp.zeros((_G4_HORIZON, 1))


def _g4_affine_dynamics(model, operating_states, operating_controls):
    """Traceable (A, B, c) of the learned model along the operating trajectory."""
    linearization = phx.control.linearize_discrete_dynamics(
        _g4_learned_plant(),
        _G4_TIMES.times[:-1],
        operating_states,
        operating_controls,
        materialization=_G4_DENSE,
        args=model,
        target_time=_G4_TIMES.times[1:],
        step_index=jnp.arange(_G4_HORIZON),
    )
    return (
        linearization.state_matrix,
        linearization.control_matrix,
        linearization.affine_offset,
    )


def _g4_controller(model, operating_states, operating_controls):
    problem = phx.control.linear_quadratic_problem_from_discrete_dynamics(
        _g4_learned_plant(),
        _G4_TIMES,
        operating_states,
        operating_controls,
        _G4_INITIAL,
        jnp.broadcast_to(jnp.eye(2), (_G4_HORIZON, 2, 2)),
        0.05 * jnp.ones((_G4_HORIZON, 1, 1)),
        4.0 * jnp.eye(2),
        materialization=_G4_DENSE,
        args=model,
        control_lower_bounds=-2.0 * jnp.ones((_G4_HORIZON, 1)),
        control_upper_bounds=2.0 * jnp.ones((_G4_HORIZON, 1)),
        problem_id="g4-learned-lq",
    )
    return phx.control.RecedingHorizonMPC(
        problem, prediction_horizon=2, terminal_policy="global", policy=_G4_QP
    )


def _g4_model(input_gain=(0.0, 1.0)):
    return _LearnedDiscreteDynamics(
        [[0.0, 1.0], [-0.5, -0.1]], [[0.3, 0.0], [0.0, 0.2]], input_gain
    )


def test_g4_mpc_sensitivity_through_the_learned_linearization():
    model = _g4_model()
    operating_states, operating_controls = _g4_operating_trajectory(model)
    sensitivity = phx.control.prepare_receding_horizon_mpc_sensitivity(
        _g4_controller(model, operating_states, operating_controls)
    )
    assert sensitivity.refusal is None
    # Closed-loop regulation loss 0.5 * ||x||^2 over the realized MPC states.
    cotangent = sensitivity.vjp(sensitivity.states, jnp.zeros_like(sensitivity.controls))
    _, pullback = jax.vjp(
        lambda candidate: _g4_affine_dynamics(
            candidate, operating_states, operating_controls
        ),
        model,
    )
    (model_gradient,) = pullback(
        (
            cotangent.dynamics_matrices,
            cotangent.control_matrices,
            cotangent.dynamics_bias,
        )
    )

    def closed_loop_loss(input_gain):
        candidate = _g4_model(input_gain)
        result = _g4_controller(candidate, operating_states, operating_controls).solve()
        assert bool(result.successful)
        return 0.5 * jnp.sum(result.states**2)

    step = 1e-6
    base = jnp.asarray([0.0, 1.0])
    finite_difference = jnp.stack(
        [
            (
                closed_loop_loss(base + step * direction)
                - closed_loop_loss(base - step * direction)
            )
            / (2.0 * step)
            for direction in jnp.eye(2)
        ]
    )
    assert jnp.allclose(
        model_gradient.input_gain, finite_difference, rtol=1e-5, atol=1e-7
    )
    assert bool(jnp.all(jnp.isfinite(model_gradient.drift)))
    assert bool(jnp.all(jnp.isfinite(model_gradient.coupling)))


# G19: differentiable MPC --------------------------------------------------------


_G19_HORIZON = 5
_G19_QP = phx.optim.ConvexSolvePolicy(
    termination=phx.optim.ConvexTermination(
        absolute=1e-12, relative=1e-12, maximum_steps=200
    )
)


def _g19_cost_blocks(weights):
    """Traceable MPC stage costs from the tuned log weights."""
    state_weight, control_weight = jnp.exp(weights[0]), jnp.exp(weights[1])
    return (
        state_weight * jnp.broadcast_to(jnp.eye(2), (_G19_HORIZON, 2, 2)),
        control_weight * jnp.ones((_G19_HORIZON, 1, 1)),
    )


def _g19_controller(weights, *, upper=0.3):
    state_costs, control_costs = _g19_cost_blocks(weights)
    problem = phx.control.LinearQuadraticControlProblem(
        jnp.broadcast_to(jnp.array([[1.0, 0.2], [0.0, 1.0]]), (_G19_HORIZON, 2, 2)),
        jnp.broadcast_to(jnp.array([[0.02], [0.2]]), (_G19_HORIZON, 2, 1)),
        jnp.array([1.0, 0.0]),
        state_costs,
        control_costs,
        jnp.eye(2),
        control_lower_bounds=-0.35 * jnp.ones((_G19_HORIZON, 1)),
        control_upper_bounds=upper * jnp.ones((_G19_HORIZON, 1)),
        problem_id="g19-tuned-mpc",
    )
    return phx.control.RecedingHorizonMPC(
        problem, prediction_horizon=3, terminal_policy="global", policy=_G19_QP
    )


def _g19_performance(states, controls):
    """Closed-loop performance judged independently of the MPC's own weights."""
    return 0.5 * jnp.sum(states**2) + 0.05 * jnp.sum(controls**2)


def test_g19_closed_loop_gradient_tunes_mpc_weights_through_active_bounds():
    weights = jnp.asarray([0.0, jnp.log(0.1)])
    sensitivity = phx.control.prepare_receding_horizon_mpc_sensitivity(
        _g19_controller(weights)
    )
    assert sensitivity.refusal is None
    # Some window meets a bound strictly: the derivative crosses active sets.
    assert bool(jnp.any(jnp.isclose(sensitivity.controls, -0.35, atol=1e-8)))
    states_cotangent, controls_cotangent = jax.grad(_g19_performance, argnums=(0, 1))(
        sensitivity.states, sensitivity.controls
    )
    cotangent = sensitivity.vjp(states_cotangent, controls_cotangent)
    _, pullback = jax.vjp(_g19_cost_blocks, weights)
    (gradient,) = pullback((cotangent.state_costs, cotangent.control_costs))

    def performance(candidate):
        result = _g19_controller(candidate).solve()
        assert bool(result.successful)
        return _g19_performance(result.states, result.controls)

    step = 1e-5
    finite_difference = jnp.stack(
        [
            (performance(weights + step * basis) - performance(weights - step * basis))
            / (2.0 * step)
            for basis in jnp.eye(2)
        ]
    )
    assert jnp.allclose(gradient, finite_difference, rtol=1e-5, atol=1e-8)

    # One descent step on the tuned weights improves the audited closed loop.
    improved = weights - 0.2 * gradient / jnp.linalg.norm(gradient)
    assert performance(improved) < performance(weights)


def test_g19_one_weak_window_refuses_the_complete_closed_loop_derivative():
    # Window 0's unconstrained optimum lies exactly on the control bound.
    problem = phx.control.LinearQuadraticControlProblem(
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.zeros((1, 1)),
        control_linear=jnp.asarray([[-0.5], [-0.2]]),
        control_upper_bounds=0.5 * jnp.ones((2, 1)),
    )
    sensitivity = phx.control.prepare_receding_horizon_mpc_sensitivity(
        phx.control.RecedingHorizonMPC(
            problem, prediction_horizon=1, terminal_policy="none", policy=_G19_QP
        )
    )
    assert bool(sensitivity.result.successful)
    assert not bool(sensitivity.regular)
    with pytest.raises(ValueError, match="refused"):
        sensitivity.vjp(
            jnp.ones_like(sensitivity.states), jnp.zeros_like(sensitivity.controls)
        )
    sparse = phx.control.RecedingHorizonMPC(
        problem,
        prediction_horizon=1,
        terminal_policy="none",
        compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
    )
    with pytest.raises(ValueError, match="dense-only"):
        phx.control.prepare_receding_horizon_mpc_sensitivity(sparse)

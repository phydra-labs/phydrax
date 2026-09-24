import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, ModelBinding, ModelPorts


class _ScaledField(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale):
        self.scale = jnp.asarray(scale)
        self.in_size = 2
        self.out_size = 2

    def __call__(self, state, /, *, key=None):
        del key
        return self.scale * state


class _ControlledStep(AbstractArrayModel):
    gain: jax.Array
    in_size: tuple[int, int] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    _input_binding = ModelBinding.pointwise("structured")

    def __init__(self, gain):
        self.gain = jnp.asarray(gain)
        self.in_size = (2, 1)
        self.out_size = 2

    def __call__(self, values, /, *, key=None):
        del key
        state, control = values
        return state + self.gain * control[0]


class _AxisStep(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    _input_binding = ModelBinding.axis("flat")

    def __init__(self, scale):
        self.scale = jnp.asarray(scale)
        self.in_size = 2
        self.out_size = 2

    def __call__(self, state, /, *, key=None):
        del key
        return self.scale * state


class _PortedIntervalStep(AbstractArrayModel):
    rate: jax.Array
    ports: ModelPorts
    in_size: tuple[int, str, str] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    _input_binding = ModelBinding.pointwise("structured")

    def __init__(self, rate, ports):
        self.rate = jnp.asarray(rate)
        self.ports = ports
        self.in_size = (2, "scalar", "scalar")
        self.out_size = 2

    def model_ports(self):
        return self.ports

    def __call__(self, values, /, *, key=None):
        del key
        state, source, target = values
        return state + self.rate * (target - source) * state


def _step_time_port(name):
    return phx.ValuePort(
        f"discrete-step:{name}",
        event_shape=(),
        component_ids=(name,),
        representation="step-time",
    )


def _identity_mapping(inputs, outputs):
    return phx.PortMapping(
        inputs=tuple((port.port_id, port.port_id) for port in inputs),
        outputs=tuple((port.port_id, port.port_id) for port in outputs),
    )


def _fitted_state_model(layout, target_port):
    features = jr.normal(jr.key(3), (16, 2))
    targets = features @ jnp.asarray([[0.5, 0.1], [0.0, -1.0]])
    return phx.ml.fit(
        phx.ml.linear.RidgeRecipe(1e-3),
        features,
        targets,
        feature_schema=phx.ml.FeatureSchema.from_ports(
            (layout.value_port(role="point"),)
        ),
        target_schema=phx.ml.TargetSchema.from_port(target_port),
    ).model


def test_fitted_model_binds_to_continuous_system_through_explicit_ports():
    layout = phx.dynamics.StateLayout((2,))
    point = layout.value_port(role="point")
    tangent = layout.value_port(role="tangent")
    model = _fitted_state_model(layout, tangent)

    with pytest.raises(ValueError, match="requires an explicit port_mapping"):
        phx.dynamics.continuous_model_system(
            model, state_layout=layout, system_id="fitted-field"
        )
    system = phx.dynamics.continuous_model_system(
        model,
        state_layout=layout,
        system_id="fitted-field",
        port_mapping=_identity_mapping((point,), (tangent,)),
    )
    state = jnp.asarray([1.0, -3.0])
    evidence = system.vector_field.port_binding

    assert jnp.allclose(system(0.0, state), model(state))
    assert evidence.inputs == ((point.port_id, point.port_id),)
    assert evidence.outputs == ((tangent.port_id, tangent.port_id),)
    assert not evidence.dimensions_verified
    assert ("input", point.port_id, "dimensions") in evidence.unverified
    assert ("output", tangent.port_id, "dimensions") in evidence.unverified


def test_fitted_model_binding_rejects_a_mismatched_owner_port():
    layout = phx.dynamics.StateLayout((2,))
    point = layout.value_port(role="point")
    tangent = layout.value_port(role="tangent")
    next_state_model = _fitted_state_model(layout, point)

    # A next-state estimator is not a vector field: its point output cannot be
    # bound to the owner's tangent port.
    with pytest.raises(ValueError, match="semantic_id mismatch"):
        phx.dynamics.continuous_model_system(
            next_state_model,
            state_layout=layout,
            system_id="fitted-field",
            port_mapping=phx.PortMapping(
                inputs=((point.port_id, point.port_id),),
                outputs=((point.port_id, tangent.port_id),),
            ),
        )
    system = phx.dynamics.discrete_model_system(
        next_state_model,
        state_layout=layout,
        system_id="fitted-step",
        step_size=0.1,
        port_mapping=_identity_mapping((point,), (point,)),
    )
    state = jnp.asarray([2.0, -4.0])

    assert system.transition.port_binding.outputs == ((point.port_id, point.port_id),)
    assert jnp.allclose(
        system(phx.dynamics.DiscreteStepContext(0.0, 0.1, 0), state),
        next_state_model(state),
    )


def test_discrete_interval_ports_bind_only_in_owner_order():
    layout = phx.dynamics.StateLayout((2,))
    point = layout.value_port(role="point")
    source = _step_time_port("source-time")
    target = _step_time_port("target-time")
    model = _PortedIntervalStep(
        0.5, ModelPorts(inputs=(point, source, target), outputs=(point,))
    )
    system = phx.dynamics.discrete_model_system(
        model,
        state_layout=layout,
        system_id="interval-step",
        input_mode="interval",
        port_mapping=_identity_mapping((point, source, target), (point,)),
    )
    state = jnp.asarray([1.0, -2.0])
    context = phx.dynamics.DiscreteStepContext(1.0, 1.5, 0)

    assert system.transition.port_binding.inputs == tuple(
        (port.port_id, port.port_id) for port in (point, source, target)
    )
    assert jnp.allclose(system(context, state), 1.25 * state)

    swapped = _PortedIntervalStep(
        0.5, ModelPorts(inputs=(point, target, source), outputs=(point,))
    )
    with pytest.raises(ValueError, match="never repacked"):
        phx.dynamics.discrete_model_system(
            swapped,
            state_layout=layout,
            system_id="interval-step",
            input_mode="interval",
            port_mapping=_identity_mapping((point, target, source), (point,)),
        )


def test_portless_model_takes_no_port_mapping():
    layout = phx.dynamics.StateLayout((2,))
    point = layout.value_port(role="point")
    tangent = layout.value_port(role="tangent")
    model = phx.nn.models.MLP(in_size=2, out_size=2, width_size=4, depth=1, key=jr.key(2))

    system = phx.dynamics.continuous_model_system(
        model, state_layout=layout, system_id="mlp-field"
    )
    assert system.vector_field.port_binding is None
    with pytest.raises(ValueError, match="declares no model ports"):
        phx.dynamics.continuous_model_system(
            model,
            state_layout=layout,
            system_id="mlp-field",
            port_mapping=_identity_mapping((point,), (tangent,)),
        )


def test_continuous_model_system_preserves_trainable_model_leaves():
    system = phx.dynamics.continuous_model_system(
        _ScaledField(2.0),
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="scaled-field",
    )
    state = jnp.asarray([1.0, -3.0])

    assert jnp.array_equal(system(0.5, state), 2.0 * state)
    assert jnp.array_equal(jax.jit(system)(0.5, state), 2.0 * state)
    gradient = eqx.filter_grad(lambda candidate: jnp.sum(candidate(0.5, state)))(system)
    assert jnp.allclose(gradient.vector_field.model.scale, -2.0)


def test_controlled_port_hamiltonian_binds_to_continuous_system():
    model = phx.nn.models.PortHamiltonianVectorField(
        state_size=2,
        control_size=1,
        energy_width=5,
        energy_depth=1,
        key=jr.key(0),
    )
    system = phx.dynamics.continuous_model_system(
        model,
        state_layout=phx.dynamics.StateLayout((2,)),
        input_layout=phx.dynamics.InputLayout((1,)),
        system_id="controlled-port-hamiltonian",
    )
    state = jnp.asarray([0.2, -0.4])
    control = jnp.asarray([0.3])

    assert jnp.allclose(
        system(0.0, state, inputs=control),
        model((state, control)),
        atol=1e-12,
        rtol=1e-12,
    )


def test_controlled_model_system_rejects_flat_input_binding():
    model = phx.nn.models.MLP(
        in_size=3,
        out_size=2,
        width_size=4,
        depth=1,
        key=jr.key(1),
    )
    with pytest.raises(ValueError, match="structured"):
        phx.dynamics.continuous_model_system(
            model,
            state_layout=phx.dynamics.StateLayout((2,)),
            input_layout=phx.dynamics.InputLayout((1,)),
            system_id="invalid-flat-model",
        )


def test_discrete_model_system_binds_complete_autonomous_next_state():
    model = _ScaledField(0.5)
    system = phx.dynamics.discrete_model_system(
        model,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="scaled-step",
        step_size=0.25,
    )
    state = jnp.asarray([2.0, -4.0])
    context = phx.dynamics.DiscreteStepContext(1.0, 1.25, 0)

    assert isinstance(system.transition, phx.dynamics.DiscreteModelTransition)
    assert system.step_size == 0.25
    assert jnp.array_equal(system(context, state), model(state))
    assert jnp.array_equal(jax.jit(system)(context, state), model(state))


def test_discrete_model_system_uses_structured_interval_control():
    model = _ControlledStep(2.0)
    system = phx.dynamics.discrete_model_system(
        model,
        state_layout=phx.dynamics.StateLayout((2,)),
        input_layout=phx.dynamics.InputLayout((1,)),
        system_id="controlled-step",
        step_size=1.0,
    )
    state = jnp.asarray([1.0, 3.0])
    control = jnp.asarray([0.5])
    context = phx.dynamics.DiscreteStepContext(0.0, 1.0, 0)

    assert jnp.array_equal(
        system(context, state, inputs=control),
        model((state, control)),
    )


def test_discrete_model_system_rejects_axis_models_and_invalid_step_contracts():
    layout = phx.dynamics.StateLayout((2,))

    with pytest.raises(ValueError, match="pointwise"):
        phx.dynamics.discrete_model_system(
            _AxisStep(1.0),
            state_layout=layout,
            system_id="axis-step",
            step_size=1.0,
        )
    with pytest.raises(ValueError, match="step_size"):
        phx.dynamics.discrete_model_system(
            _ScaledField(1.0),
            state_layout=layout,
            system_id="invalid-step",
            step_size=0.0,
        )

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, ModelBinding


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

    assert isinstance(system.transition, phx.dynamics.DiscreteModelTransition)
    assert system.step_size == 0.25
    assert jnp.array_equal(system(1.0, state), model(state))
    assert jnp.array_equal(jax.jit(system)(1.0, state), model(state))


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

    assert jnp.array_equal(
        system(0.0, state, inputs=control),
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

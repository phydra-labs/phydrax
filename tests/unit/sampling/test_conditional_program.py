import jax
import jax.numpy as jnp

import phydrax as phx


def _increment(_key, parameters, _tails, current, kernel_state):
    return current + parameters, kernel_state


def test_conditional_program_updates_arbitrary_pytree_state_by_stage():
    conditional = phx.sampling.conditional
    specification = {
        "position": jax.ShapeDtypeStruct((2,), jnp.float64),
        "label": jax.ShapeDtypeStruct((), jnp.int32),
    }
    group = conditional.ConditionalVariableGroup("particles", 2, specification)

    def update(_key, parameters, tails, current, kernel_state):
        neighbor = tails[0]
        output = {
            "position": current["position"] + parameters * neighbor["position"],
            "label": current["label"],
        }
        return output, kernel_state

    interaction = conditional.ConditionalInteractionGroup(
        "particles",
        jnp.asarray([0]),
        ("particles",),
        (jnp.asarray([1]),),
        jnp.asarray(0.5),
        interaction_id="neighbor-drift",
    )
    program = conditional.prepare_conditional_program(
        (group,),
        (
            conditional.ConditionalUpdate(
                interaction,
                conditional.CallableConditionalKernel(
                    update,
                    kernel_id="deterministic-neighbor-drift",
                ),
            ),
        ),
        (conditional.ConditionalUpdateStage((0,), stage_id="drift"),),
    )
    state = conditional.initialize_conditional_program(
        program,
        {
            "particles": {
                "position": jnp.asarray(
                    [[[1.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [0.0, 2.0]]]
                ),
                "label": jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32),
            }
        },
    )
    result = conditional.sample_conditional_program(
        program,
        state,
        key=jax.random.key(1),
        warmup_steps=0,
        num_draws=2,
    )

    samples = result.samples[0]
    assert samples["position"].shape == (2, 2, 2, 2)
    assert jnp.allclose(
        samples["position"][:, 0, 0], jnp.asarray([[2.0, 0.0], [0.0, 2.0]])
    )
    assert jnp.allclose(
        samples["position"][:, 1, 0], jnp.asarray([[3.0, 0.0], [0.0, 3.0]])
    )
    assert jnp.array_equal(samples["label"][:, 0], state.values[0]["label"])


def test_conditional_program_supports_stateful_callable_kernels():
    conditional = phx.sampling.conditional
    group = conditional.ConditionalVariableGroup(
        "x",
        2,
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    interaction = conditional.ConditionalInteractionGroup(
        "x",
        jnp.asarray([0, 1]),
        (),
        (),
        jnp.asarray(1.0),
        interaction_id="increment-all",
    )
    kernel = conditional.CallableConditionalKernel(
        _increment,
        kernel_id="increment",
        initialize=lambda _spec: jnp.asarray(0, dtype=jnp.int32),
    )
    program = conditional.prepare_conditional_program(
        (group,),
        (conditional.ConditionalUpdate(interaction, kernel),),
        (conditional.ConditionalUpdateStage((0,), stage_id="all"),),
    )
    state = conditional.initialize_conditional_program(
        program,
        {"x": jnp.zeros((3, 2))},
    )
    result = conditional.sample_conditional_program(
        program,
        state,
        key=jax.random.key(3),
        warmup_steps=1,
        num_draws=2,
    )

    assert result.samples[0].shape == (3, 2, 2)
    assert jnp.all(result.samples[0][:, 0] == 2)
    assert jnp.all(result.samples[0][:, 1] == 3)

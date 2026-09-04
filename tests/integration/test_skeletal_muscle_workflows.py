import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications import skeletal_muscle


def test_public_motor_unit_workflow_advances_and_retains_population_evidence():
    plan = skeletal_muscle.motor_units.PotvinFuglevand2017Plan()
    runtime = plan.prepare()
    initial = runtime.initialize()

    def step(state, _):
        candidate = runtime.candidate(state, 40.0, 0.1)
        return candidate.commit(), jnp.stack(
            (
                candidate.output.total_force,
                candidate.output.total_force_capacity_fraction,
                candidate.evidence.minimum_recruitment_margin,
                candidate.evidence.successful.astype(candidate.output.total_force.dtype),
            )
        )

    final, history = jax.lax.scan(step, initial, xs=None, length=100)

    assert history.shape == (100, 4)
    assert bool(jnp.all(history[:, 3] == 1.0))
    assert history[-1, 0] < history[0, 0]
    assert history[-1, 1] < history[0, 1]
    assert bool(jnp.any(final.recruitment_duration_s > 0.0))
    assert bool(jnp.any(final.current_twitch_force < initial.current_twitch_force))
    np.testing.assert_allclose(
        skeletal_muscle.skeletal_muscle_quantity("time").to_si(0.1), 0.1
    )

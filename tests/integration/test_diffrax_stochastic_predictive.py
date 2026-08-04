import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_diffrax_ensemble_converts_to_process_predictive_field():
    problem = phx.solver.DifferentialProblem(
        lambda t, state, args: -0.3 * state,
        jnp.asarray([1.0, -1.0]),
        t0=0.0,
        t1=0.5,
        wiener_terms=(
            phx.solver.WienerTerm(
                "state-space",
                lambda t, state, args: 0.2 * jnp.eye(2),
                (2,),
                structure="additive",
                basis_id="state-space",
            ),
        ),
        interpretation="ito",
    )
    solution = phx.solver.solve_diffrax_ensemble(
        problem,
        save_times=jnp.asarray([0.0, 0.25, 0.5]),
        realization=phx.stochastic.WienerRealization(
            jr.key(12),
            (2,),
            support=(0.0, 0.5),
            sample_shape=(32,),
            tolerance=1e-3,
            noise_id="state-space",
            label="integration-test",
        ),
        dt0=0.01,
    )
    predictive = solution.to_predictive(
        sample_dim="path",
        time_dim="time",
        state_dims=("state",),
    )

    assert predictive.samples.dims == ("path", "time", "state")
    assert predictive.samples.shape == (32, 3, 2)
    assert predictive.sample_axes == (phx.uq.SampleAxis("path", "process"),)
    assert predictive.valid.dims == ("path",)
    assert jnp.all(predictive.valid.data)
    assert predictive.process_variance().dims == ("time", "state")
    assert jnp.allclose(
        predictive.total_variance().data,
        predictive.process_variance().data,
    )
    with pytest.raises(ValueError, match="no sample axes for"):
        predictive.epistemic_variance()
    with pytest.raises(ValueError, match="no sample axes for"):
        predictive.numerical_variance()

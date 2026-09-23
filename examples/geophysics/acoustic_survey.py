"""Checkpoint-replayed two-dimensional line-source acoustic survey."""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    grid = phx.applications.geophysics.AcousticGrid((25, 25), (5.0, 5.0))
    acquisition = phx.applications.geophysics.SeismicAcquisition(
        grid,
        ((60.0, 60.0),),
        ((80.0, 60.0), (60.0, 80.0)),
    )
    plan = phx.applications.geophysics.ConstantDensityAcousticPlan(
        grid,
        0.001,
        48,
        2000.0,
        1000.0,
        absorber_cells=5,
        absorber_strength=5.0,
    )
    times = 0.001 * jnp.arange(plan.step_count)
    rates = (
        1.0e-3 * phx.applications.geophysics.ricker_wavelet(times, 25.0, delay=0.025)
    )[:, None]

    def prediction(speed):
        return plan.simulate(
            speed,
            acquisition,
            rates,
            replay="block",
            block_size=12,
        ).traces.values

    traces, tangent = jax.jvp(
        prediction,
        (jnp.asarray(1500.0),),
        (jnp.asarray(1.0),),
    )
    if not bool(jnp.all(jnp.isfinite(tangent))):
        raise RuntimeError("Acoustic survey tangent is nonfinite")
    print(
        {
            "cfl_number": plan.cfl_number,
            "maximum_pressure_Pa": float(jnp.max(jnp.abs(traces))),
            "finite_tangent": True,
            "trace_shape": np.asarray(traces.shape).tolist(),
        }
    )


if __name__ == "__main__":
    main()

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


system = phx.atomistic.AtomisticSystemPlan(
    list(range(64)), [1] * 64, [1.0] * 64, phx.atomistic.AtomisticUnitSystem.reduced()
).prepare()
positions = jnp.reshape(jnp.arange(192, dtype=jnp.float32), (64, 3)) * 0.01
cv = phx.atomistic.sampling.CollectiveVariablePlan(
    phx.atomistic.sampling.CollectiveVariableKind.RADIUS_OF_GYRATION,
    jnp.arange(64),
).prepare(system)
evaluate = eqx.filter_jit(cv.evaluate)
started = time.perf_counter()
first = evaluate(positions)
jax.block_until_ready(first.value)
compile_seconds = time.perf_counter() - started
started = time.perf_counter()
for _ in range(100):
    result = evaluate(positions)
jax.block_until_ready(result.value)
steady_seconds = (time.perf_counter() - started) / 100
print(
    json.dumps(
        {
            "compile_seconds": compile_seconds,
            "steady_seconds": steady_seconds,
            "value": float(result.value),
            "successful": bool(result.successful),
            "cv_id": cv.prepared_id,
        },
        indent=2,
        sort_keys=True,
    )
)

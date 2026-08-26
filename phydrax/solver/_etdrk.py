#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle
from ..linalg import DiagonalLinearOperator
from ..linalg._matrix_functions import _phi_function_value
from ._differential import DifferentialSolution
from ._semilinear_drift import SemilinearDrift
from ._temporal_method import TemporalMethodCapabilities


class ETDRKMethod(StrictModule, NonTrainableState):
    """Fixed-step diagonal exponential time-differencing Runge--Kutta method."""

    order: Literal[2, 4] = eqx.field(static=True)
    capabilities: TemporalMethodCapabilities
    method_id: str = eqx.field(static=True)

    def __init__(self, order: Literal[2, 4] = 4):
        order_ = int(order)
        if order_ not in (2, 4):
            raise ValueError("ETDRK order must be two or four.")
        identifier = canonical_fingerprint(
            {"kind": "etdrk-method", "order": order_, "linear_path": "diagonal"}
        )
        self.order = order_
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("additive-ode",),
            method_class="exponential",
            order=order_,
            adaptive=False,
            history_depth=1,
            stage_abscissae=(0.0, 1.0) if order_ == 2 else (0.0, 0.5, 0.5, 1.0),
            causal_stage_extent=1.0,
            noise_requirement="none",
            method_id=identifier,
        )
        self.method_id = identifier

    def _diagonal(self, drift: SemilinearDrift, /) -> Array:
        operator = drift.linear_operator
        if not isinstance(operator, DiagonalLinearOperator) or operator.batch_shape:
            raise ValueError(
                "ETDRK initially requires an unbatched DiagonalLinearOperator."
            )
        return operator.diagonal.reshape(drift.state_shape)

    def step(
        self,
        drift: SemilinearDrift,
        time: ArrayLike,
        state: ArrayLike,
        dt: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Advance one semilinear state by one fixed ETDRK step."""
        if not isinstance(drift, SemilinearDrift):
            raise TypeError("drift must be a SemilinearDrift.")
        value = jnp.asarray(state)
        if value.shape != drift.state_shape:
            raise ValueError(
                f"ETDRK state must have shape {drift.state_shape}; got {value.shape}."
            )
        step = jnp.asarray(dt, dtype=value.real.dtype).reshape(())
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step > 0)),
            "ETDRK step size must be finite and positive.",
        )
        start = jnp.asarray(time, dtype=step.dtype).reshape(())
        diagonal = self._diagonal(drift).astype(value.dtype)
        z = step * diagonal
        exponential = jnp.exp(z)
        n1 = drift.nonlinear(start, value, args)
        if self.order == 2:
            predictor = exponential * value + step * _phi_function_value(z, 1) * n1
            n2 = drift.nonlinear(start + step, predictor, args)
            return predictor + step * _phi_function_value(z, 2) * (n2 - n1)
        half_exponential = jnp.exp(0.5 * z)
        q = 0.5 * _phi_function_value(0.5 * z, 1)
        a = half_exponential * value + step * q * n1
        n2 = drift.nonlinear(start + 0.5 * step, a, args)
        b = half_exponential * value + step * q * n2
        n3 = drift.nonlinear(start + 0.5 * step, b, args)
        c = half_exponential * a + step * q * (2.0 * n3 - n1)
        n4 = drift.nonlinear(start + step, c, args)
        phi1 = _phi_function_value(z, 1)
        phi2 = _phi_function_value(z, 2)
        phi3 = _phi_function_value(z, 3)
        f1 = phi1 - 3.0 * phi2 + 4.0 * phi3
        f2 = phi2 - 2.0 * phi3
        f3 = 4.0 * phi3 - phi2
        return exponential * value + step * (f1 * n1 + 2.0 * f2 * (n2 + n3) + f3 * n4)


def solve_etdrk(
    method: ETDRKMethod,
    drift: SemilinearDrift,
    initial_state: ArrayLike,
    times: ArrayLike,
    /,
    *,
    args: Any = None,
    discretization_bundle: DiscretizationBundle | None = None,
    problem_id: str | None = None,
) -> DifferentialSolution:
    """Integrate a diagonal semilinear system on an explicit increasing time grid."""
    if not isinstance(method, ETDRKMethod):
        raise TypeError("method must be an ETDRKMethod.")
    if not isinstance(drift, SemilinearDrift):
        raise TypeError("drift must be a SemilinearDrift.")
    saved = jnp.asarray(times)
    if jnp.iscomplexobj(saved):
        raise TypeError("ETDRK times must be real-valued.")
    if saved.ndim != 1 or saved.size < 2:
        raise ValueError("ETDRK times must be a rank-one grid with at least two values.")
    saved_host = np.asarray(saved, dtype=float)
    if np.any(~np.isfinite(saved_host)) or np.any(np.diff(saved_host) <= 0.0):
        raise ValueError("ETDRK times must be finite and strictly increasing.")
    initial = jnp.asarray(initial_state)
    if initial.shape != drift.state_shape:
        raise ValueError(
            f"initial_state must have shape {drift.state_shape}; got {initial.shape}."
        )
    starts = saved[:-1]
    durations = jnp.diff(saved)

    def advance(state: Array, data: tuple[Array, Array]):
        time, duration = data
        following = method.step(drift, time, state, duration, args)
        return following, following

    _, advanced = jax.lax.scan(advance, initial, (starts, durations))
    states = jnp.concatenate((initial[None], advanced), axis=0)
    valid = jnp.all(jnp.isfinite(states.reshape((states.shape[0], -1))), axis=-1)
    name = f"ETDRK{method.order}"
    return DifferentialSolution(
        times=saved,
        states=states,
        valid=valid,
        backend_result=method,
        stats={"num_steps": int(saved.size - 1), "order": method.order},
        solver_name=name,
        interpretation="ito",
        solver_id=f"solver:etdrk:{method.order}",
        resolved_method=f"etdrk{method.order}:diagonal",
        discretization_bundle=discretization_bundle,
        backend_successful=jnp.all(valid),
        problem_id=problem_id,
    )


__all__ = ["ETDRKMethod", "solve_etdrk"]

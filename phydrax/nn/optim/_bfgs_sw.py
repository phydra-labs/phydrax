#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
Full-memory BFGS with strong Wolfe line search.

Uses a dense inverse-Hessian approximation in the flattened parameter space.
Best suited for low/moderate dimensional problems; for large models prefer L-BFGS.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from optax._src import base, transform

from ._extra_args import GradientTransformationExtraArgs


def _tree_dot(a, b):
    leaves = jax.tree_util.tree_map(
        lambda x, y: jnp.vdot(jnp.asarray(x), jnp.asarray(y)), a, b
    )
    s = jax.tree_util.tree_reduce(lambda acc, v: acc + v, leaves, 0.0)
    return jnp.real(s) if jnp.iscomplexobj(s) else s


def _tree_add_scaled(x, alpha, y):
    return jax.tree_util.tree_map(lambda a, b: a + alpha * b, x, y)


def _line_search_strong_wolfe(
    value_fn, params, f0, g0, p, *, c1=1e-4, c2=0.9, alpha0=1.0, max_iters=20
):
    g0p = _tree_dot(g0, p)
    if g0p >= 0:
        p = jax.tree_util.tree_map(lambda x: -x, g0)
        g0p = _tree_dot(g0, p)

    alpha = alpha0
    for _ in range(max_iters):
        x_new = _tree_add_scaled(params, alpha, p)
        f_new = value_fn(x_new)
        if f_new > f0 + c1 * alpha * g0p:
            alpha *= 0.5
            continue
        g_new = jax.grad(value_fn)(x_new)
        gnp = _tree_dot(g_new, p)
        if jnp.abs(gnp) <= c2 * jnp.abs(g0p):
            return alpha, p
        if gnp < 0:
            alpha *= 1.1
        else:
            alpha *= 0.5
    return alpha, p


class BFGSState(NamedTuple):
    prev_params: base.Params
    prev_grad: base.Updates
    H: jnp.ndarray  # inverse Hessian in flattened space (n,n)
    step: jnp.ndarray


def _as_bfgs_state(state: base.OptState, /) -> BFGSState:
    if not isinstance(state, BFGSState):
        raise TypeError("Expected BFGSState for optimizer state.")
    return state


def _split_decay_state(state: base.OptState, /) -> tuple[BFGSState, base.OptState]:
    if (
        not isinstance(state, tuple)
        or len(state) != 2
        or not isinstance(state[0], BFGSState)
    ):
        raise TypeError("Expected (BFGSState, decay_state) for optimizer state.")
    assert isinstance(state[1], base.OptState)
    return state[0], state[1]


def bfgs_sw_core(
    *, eps: float = 1e-12, init_lr: float = 1.0, c1: float = 1e-4, c2: float = 0.9
):
    def init_fn(params: base.Params) -> BFGSState:
        flat, _ = ravel_pytree(params)
        n = flat.size
        H0 = jnp.eye(n, dtype=flat.dtype)
        zero_like = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        return BFGSState(
            prev_params=params,
            prev_grad=zero_like,
            H=H0,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: base.Updates,
        state: base.OptState,
        params: base.Params | None = None,
        **extra_args: Any,
    ) -> tuple[base.Updates, base.OptState]:
        grads = updates
        state_bfgs = _as_bfgs_state(state)
        params_ = params if params is not None else state_bfgs.prev_params
        step = state_bfgs.step + 1
        # Flatten
        p_flat, unravel = ravel_pytree(params_)
        g_flat, _ = ravel_pytree(grads)
        pp_flat, _ = ravel_pytree(state_bfgs.prev_params)
        pg_flat, _ = ravel_pytree(state_bfgs.prev_grad)
        H = state_bfgs.H

        s = p_flat - pp_flat
        y = g_flat - pg_flat
        sty = jnp.vdot(s, y).real
        yHy = jnp.vdot(y, H @ y).real

        # BFGS update if curvature is good
        def do_bfgs(H):
            rho = 1.0 / (sty + 1e-30)
            I = jnp.eye(H.shape[0], dtype=H.dtype)
            V = I - rho * jnp.outer(s, y)
            H_new = V @ H @ V.T + rho * jnp.outer(s, s)
            return H_new

        cond = (sty > eps) & (yHy > eps)
        H = jax.lax.cond(cond, do_bfgs, lambda H: H, H)

        # Direction and descent check
        d_flat = -(H @ g_flat)
        descent = jnp.vdot(d_flat, g_flat).real < 0
        d_flat = jax.lax.cond(descent, lambda v: v, lambda v: -g_flat, d_flat)
        direction = unravel(d_flat)

        # Strong-Wolfe line search if value_fn provided
        if ("value_fn" in extra_args) and ("value" in extra_args):
            alpha, direction = _line_search_strong_wolfe(
                extra_args["value_fn"],
                params_,
                extra_args["value"],
                grads,
                direction,
                c1=c1,
                c2=c2,
                alpha0=init_lr,
            )
            updates = jax.tree_util.tree_map(lambda d: alpha * d, direction)
        else:
            updates = jax.tree_util.tree_map(lambda d: init_lr * d, direction)

        new_state = BFGSState(prev_params=params_, prev_grad=grads, H=H, step=step)
        return updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def bfgs_sw(
    learning_rate: base.ScalarOrSchedule,
    *,
    eps: float = 1e-12,
    weight_decay: float = 0.0,
    mask=None,
    wolfe_c1: float = 1e-4,
    wolfe_c2: float = 0.9,
):
    """Create a full-memory BFGS optimizer with strong Wolfe line search.

    This optimizer maintains a dense inverse-Hessian approximation in the flattened
    parameter space and optionally applies decoupled weight decay.

    **Arguments:**

    - `learning_rate`: Step size or schedule used as the initial line-search step.
    - `eps`: Curvature threshold for BFGS updates.
    - `weight_decay`: Decoupled weight decay coefficient (0 disables).
    - `mask`: Optional mask for weight decay.
    - `wolfe_c1`: Armijo parameter for strong Wolfe line search.
    - `wolfe_c2`: Curvature parameter for strong Wolfe line search.

    **Returns:**

    - An Optax `GradientTransformationExtraArgs` implementing BFGS with strong Wolfe
      line search.
    """
    lr0 = learning_rate if isinstance(learning_rate, (float, int)) else 1.0
    core = bfgs_sw_core(eps=eps, init_lr=float(lr0), c1=wolfe_c1, c2=wolfe_c2)
    if weight_decay == 0.0:
        return core

    decay = transform.add_decayed_weights(weight_decay, mask)

    def init_fn(params: base.Params) -> tuple[BFGSState, base.OptState]:
        return core.init(params), decay.init(params)

    def update_fn(
        updates: base.Updates,
        state: base.OptState,
        params: base.Params | None = None,
        **extra_args: Any,
    ) -> tuple[base.Updates, base.OptState]:
        grads = updates
        core_state, decay_state = _split_decay_state(state)
        updates, core_state = core.update(grads, core_state, params, **extra_args)
        updates, decay_state = decay.update(updates, decay_state, params)
        return updates, (core_state, decay_state)

    return GradientTransformationExtraArgs(init_fn, update_fn)

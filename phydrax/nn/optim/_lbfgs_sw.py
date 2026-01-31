#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
Limited-memory BFGS with strong Wolfe line search (custom implementation).

Two-loop recursion with ring-buffer history and scalar H0 scaling
  gamma = (s^T y) / (y^T y)
and a backtracking-ish strong Wolfe line search.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
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
    def tree_add(x, a, p):
        return jax.tree_util.tree_map(lambda xi, pi: xi + a * pi, x, p)

    g0p = _tree_dot(g0, p)
    if g0p >= 0:
        # fallback: steepest descent
        p = jax.tree_util.tree_map(lambda x: -x, g0)
        g0p = _tree_dot(g0, p)

    alpha = alpha0
    for _ in range(max_iters):
        x_new = tree_add(params, alpha, p)
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


class LBFGSState(NamedTuple):
    prev_params: base.Params
    prev_grad: base.Updates
    s_hist: tuple[base.Params, ...]
    y_hist: tuple[base.Updates, ...]
    rho_hist: jnp.ndarray
    size: jnp.ndarray
    head: jnp.ndarray
    memory: int
    gamma: jnp.ndarray


def _as_lbfgs_state(state: base.OptState, /) -> LBFGSState:
    if not isinstance(state, LBFGSState):
        raise TypeError("Expected LBFGSState for optimizer state.")
    return state


def _split_decay_state(state: base.OptState, /) -> tuple[LBFGSState, base.OptState]:
    if (
        not isinstance(state, tuple)
        or len(state) != 2
        or not isinstance(state[0], LBFGSState)
    ):
        raise TypeError("Expected (LBFGSState, decay_state) for optimizer state.")
    assert isinstance(state[1], base.OptState)
    return state[0], state[1]


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), tree)


def lbfgs_sw_core(
    *,
    memory: int = 10,
    eps: float = 1e-12,
    gamma_min: float = 1e-12,
    gamma_max: float = 1e12,
    init_lr: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
):
    def init_fn(params: base.Params) -> LBFGSState:
        zero = _zeros_like_tree(params)
        s_hist = tuple(zero for _ in range(memory))
        y_hist = tuple(zero for _ in range(memory))
        rho_hist = jnp.zeros((memory,), dtype=jnp.result_type(0.0))
        return LBFGSState(
            prev_params=params,
            prev_grad=zero,
            s_hist=s_hist,
            y_hist=y_hist,
            rho_hist=rho_hist,
            size=jnp.array(0, dtype=jnp.int32),
            head=jnp.array(0, dtype=jnp.int32),
            memory=memory,
            gamma=jnp.array(1.0),
        )

    def update_fn(
        updates: base.Updates,
        state: base.OptState,
        params: base.Params | None = None,
        **extra_args: Any,
    ) -> tuple[base.Updates, base.OptState]:
        grads = updates
        state_lbfgs = _as_lbfgs_state(state)
        params_ = params if params is not None else state_lbfgs.prev_params
        s_hist = list(state_lbfgs.s_hist)
        y_hist = list(state_lbfgs.y_hist)
        rho_hist = state_lbfgs.rho_hist
        size = int(state_lbfgs.size.item())
        head = int(state_lbfgs.head.item())
        memory = state_lbfgs.memory
        gamma = state_lbfgs.gamma

        # New pair
        s = jax.tree_util.tree_map(lambda p, p0: p - p0, params_, state_lbfgs.prev_params)
        y = jax.tree_util.tree_map(lambda g, g0: g - g0, grads, state_lbfgs.prev_grad)
        sty = _tree_dot(s, y)
        yty = _tree_dot(y, y)
        if sty > eps and yty > eps:
            rho = 1.0 / (sty + 1e-30)
            s_hist[head] = s
            y_hist[head] = y
            rho_hist = rho_hist.at[head].set(rho)
            head = (head + 1) % memory
            size = min(size + 1, memory)
            gamma = jnp.clip(sty / (yty + 1e-30), gamma_min, gamma_max)

        # Two-loop recursion
        q = grads
        alphas = []
        for k in range(size):
            idx = (head - 1 - k) % memory
            s_k = s_hist[idx]
            y_k = y_hist[idx]
            rho_k = rho_hist[idx]
            alpha_k = rho_k * _tree_dot(s_k, q)
            q = _tree_add_scaled(q, -alpha_k, y_k)
            alphas.append(alpha_k)

        r = jax.tree_util.tree_map(lambda qi: gamma * qi, q)

        for k in range(size - 1, -1, -1):
            idx = (head - size + k) % memory
            s_k = s_hist[idx]
            y_k = y_hist[idx]
            rho_k = rho_hist[idx]
            beta_k = rho_k * _tree_dot(y_k, r)
            r = _tree_add_scaled(r, alphas[size - 1 - k] - beta_k, s_k)

        direction = jax.tree_util.tree_map(lambda t: -t, r)

        # Strong Wolfe; expects value_fn + value provided
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

        new_state = LBFGSState(
            prev_params=params_,
            prev_grad=grads,
            s_hist=tuple(s_hist),
            y_hist=tuple(y_hist),
            rho_hist=rho_hist,
            size=jnp.array(size, dtype=jnp.int32),
            head=jnp.array(head, dtype=jnp.int32),
            memory=memory,
            gamma=gamma,
        )
        return updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def lbfgs_sw(
    learning_rate: base.ScalarOrSchedule,
    *,
    memory: int = 10,
    eps: float = 1e-12,
    gamma_min: float = 1e-12,
    gamma_max: float = 1e12,
    weight_decay: float = 0.0,
    mask=None,
    wolfe_c1: float = 1e-4,
    wolfe_c2: float = 0.9,
):
    """Create an L-BFGS optimizer with strong Wolfe line search.

    Uses a limited-memory two-loop recursion with ring-buffer history and optional
    decoupled weight decay.

    **Arguments:**

    - `learning_rate`: Step size or schedule used as the initial line-search step.
    - `memory`: Number of correction pairs stored in the history.
    - `eps`: Curvature threshold for L-BFGS updates.
    - `gamma_min`: Lower bound for initial inverse-Hessian scaling.
    - `gamma_max`: Upper bound for initial inverse-Hessian scaling.
    - `weight_decay`: Decoupled weight decay coefficient (0 disables).
    - `mask`: Optional mask for weight decay.
    - `wolfe_c1`: Armijo parameter for strong Wolfe line search.
    - `wolfe_c2`: Curvature parameter for strong Wolfe line search.

    **Returns:**

    - An Optax `GradientTransformationExtraArgs` implementing L-BFGS with strong Wolfe
      line search.
    """
    lr0 = learning_rate if isinstance(learning_rate, (float, int)) else 1.0
    core = lbfgs_sw_core(
        memory=memory,
        eps=eps,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        init_lr=float(lr0),
        c1=wolfe_c1,
        c2=wolfe_c2,
    )
    if weight_decay == 0.0:
        return core

    decay = transform.add_decayed_weights(weight_decay, mask)

    def init_fn(params: base.Params) -> tuple[LBFGSState, base.OptState]:
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

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
Spectral-Scaled Broyden (scalar inverse-Hessian) gradient transformation.

Implements a simple spectral step based on the alternate Barzilai–Borwein
formula often associated with Broyden-type updates:

  gamma_k = (s_k^T s_k) / (s_k^T y_k)

where s_k = x_k - x_{k-1}, y_k = g_k - g_{k-1}. The resulting update is

  u_k = - gamma_k * g_k

This keeps the Optax interface and can be chained with standard learning-rate
and weight-decay transformations.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from optax._src import base, transform

from ._extra_args import GradientTransformationExtraArgs


def _line_search_strong_wolfe(
    value_fn, params, f0, g0, p, *, c1=1e-4, c2=0.9, alpha0=1.0, max_iters=20
):
    def tree_add(x, a, p):
        return jax.tree_util.tree_map(lambda xi, pi: xi + a * pi, x, p)

    def tree_dot(a, b):
        leaves = jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), a, b)
        s = jax.tree_util.tree_reduce(lambda acc, v: acc + v, leaves, 0.0)
        return jnp.real(s) if jnp.iscomplexobj(s) else s

    g0p = tree_dot(g0, p)
    if g0p >= 0:
        p = jax.tree_util.tree_map(lambda x: -x, g0)
        g0p = tree_dot(g0, p)

    alpha = alpha0
    for _ in range(max_iters):
        x_new = tree_add(params, alpha, p)
        f_new = value_fn(x_new)
        if f_new > f0 + c1 * alpha * g0p:
            alpha *= 0.5
            continue
        g_new = jax.grad(value_fn)(x_new)
        gnp = tree_dot(g_new, p)
        if jnp.abs(gnp) <= c2 * jnp.abs(g0p):
            return alpha, p
        if gnp < 0:
            alpha *= 1.1
        else:
            alpha *= 0.5
    return alpha, p


def _tree_dot(a, b):
    def dot(x, y):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        v = jnp.vdot(x, y)
        return jnp.real(v) if jnp.iscomplexobj(v) else v

    leaves = jax.tree_util.tree_map(dot, a, b)
    return jax.tree_util.tree_reduce(lambda acc, v: acc + v, leaves, 0.0)


class SSBroydenState(NamedTuple):
    prev_params: base.Params
    prev_grad: base.Updates
    gamma: jnp.ndarray
    step: jnp.ndarray


def _as_ssbroyden_state(state: base.OptState, /) -> SSBroydenState:
    if not isinstance(state, SSBroydenState):
        raise TypeError("Expected SSBroydenState for optimizer state.")
    return state


def _split_decay_state(state: base.OptState, /) -> tuple[SSBroydenState, base.OptState]:
    if (
        not isinstance(state, tuple)
        or len(state) != 2
        or not isinstance(state[0], SSBroydenState)
    ):
        raise TypeError("Expected (SSBroydenState, decay_state) for optimizer state.")
    assert isinstance(state[1], base.OptState)
    return state[0], state[1]


def _ssbroyden_core(
    eps: float = 1e-12,
    gamma0: float = 1.0,
    gamma_min: float = 1e-6,
    gamma_max: float = 1e6,
    init_lr: float = 1.0,
    wolfe: bool = True,
    c1: float = 1e-4,
    c2: float = 0.9,
):
    def init_fn(params: base.Params) -> SSBroydenState:
        zeros_like = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        return SSBroydenState(
            prev_params=params,
            prev_grad=zeros_like,
            gamma=jnp.array(gamma0),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: base.Updates,
        state: base.OptState,
        params: base.Params | None = None,
        **extra_args: Any,
    ) -> tuple[base.Updates, base.OptState]:
        grads = updates
        state_ss = _as_ssbroyden_state(state)
        params_ = params if params is not None else state_ss.prev_params
        step = state_ss.step + 1
        gamma = state_ss.gamma

        s = jax.tree_util.tree_map(lambda p, p0: p - p0, params_, state_ss.prev_params)
        y = jax.tree_util.tree_map(lambda g, g0: g - g0, grads, state_ss.prev_grad)

        sts = _tree_dot(s, s)
        sty = _tree_dot(s, y)

        # gamma = (s^T s) / (s^T y); protect against division by ~0 using cond
        def _upd(_):
            return jnp.clip(sts / (sty + jnp.sign(sty) * eps), gamma_min, gamma_max)

        gamma_new = jax.lax.cond(jnp.abs(sty) > eps, _upd, lambda _: gamma, operand=None)

        direction = jax.tree_util.tree_map(lambda g: -gamma_new * g, grads)
        if wolfe and ("value_fn" in extra_args) and ("value" in extra_args):
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
        new_state = SSBroydenState(
            prev_params=params_,
            prev_grad=grads,
            gamma=gamma_new,
            step=step,
        )
        return updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def ssbroyden(
    learning_rate: base.ScalarOrSchedule,
    *,
    eps: float = 1e-12,
    gamma0: float = 1.0,
    gamma_min: float = 1e-6,
    gamma_max: float = 1e6,
    weight_decay: float = 0.0,
    mask=None,
    strong_wolfe: bool = True,
    wolfe_c1: float = 1e-4,
    wolfe_c2: float = 0.9,
):
    """Spectral-Scaled Broyden (scalar H0) optimizer as an Optax transform.
    def _line_search_strong_wolfe(value_fn, params, f0, g0, p, *, c1=1e-4, c2=0.9, alpha0=1.0, max_iters=20):
        def tree_add(x, a, p):
            return jax.tree_util.tree_map(lambda xi, pi: xi + a * pi, x, p)

        def tree_dot(a, b):
            leaves = jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), a, b)
            s = jax.tree_util.tree_reduce(lambda acc, v: acc + v, leaves, 0.0)
            return jnp.real(s) if jnp.iscomplexobj(s) else s

        g0p = tree_dot(g0, p)
        if g0p >= 0:
            p = jax.tree_util.tree_map(lambda x: -x, g0)
            g0p = tree_dot(g0, p)

        alpha = alpha0
        for _ in range(max_iters):
            x_new = tree_add(params, alpha, p)
            f_new = value_fn(x_new)
            if f_new > f0 + c1 * alpha * g0p:
                alpha *= 0.5
                continue
            g_new = jax.grad(value_fn)(x_new)
            gnp = tree_dot(g_new, p)
            if jnp.abs(gnp) <= c2 * jnp.abs(g0p):
                return alpha, p
            if gnp < 0:
                alpha *= 1.1
            else:
                alpha *= 0.5
        return alpha, p


        - Produces updates = -H g with H = gamma I; gamma is BB/Broyden spectral estimate.
        - Chain with learning rate and weight decay.
    """
    lr0 = learning_rate if isinstance(learning_rate, (float, int)) else 1.0
    core = _ssbroyden_core(
        eps=eps,
        gamma0=gamma0,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        init_lr=float(lr0),
        wolfe=strong_wolfe,
        c1=wolfe_c1,
        c2=wolfe_c2,
    )
    if weight_decay == 0.0:
        return core

    decay = transform.add_decayed_weights(weight_decay, mask)

    def init_fn(params: base.Params) -> tuple[SSBroydenState, base.OptState]:
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

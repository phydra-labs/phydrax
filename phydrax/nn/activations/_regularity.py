#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared value regularity of the package's known activation callables."""

import functools
from collections.abc import Callable
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp

from ..._differentiation import DerivativeRegularity
from .._utils import _identity
from ._adaptive_activation import AdaptiveActivation
from ._functions import squared_relu
from ._stan import Stan


_LINEAR = DerivativeRegularity.smooth(degree_bound=1)
_SMOOTH = DerivativeRegularity.smooth()
_C0_PIECEWISE_LINEAR = DerivativeRegularity.piecewise_polynomial(
    continuity=0, degree_bound=1
)
_C1_PIECEWISE_QUADRATIC = DerivativeRegularity.piecewise_polynomial(
    continuity=1, degree_bound=2
)
_C0_PIECEWISE_SMOOTH = DerivativeRegularity.piecewise_smooth(continuity=0)
_C1_PIECEWISE_SMOOTH = DerivativeRegularity.piecewise_smooth(continuity=1)

# Matched by object identity; `jax.nn.tanh` is `jnp.tanh` and `jax.nn.swish` is
# `jax.nn.silu`, so each alias is covered by one entry.
_KNOWN_ACTIVATIONS: tuple[tuple[Callable[..., Any], DerivativeRegularity], ...] = (
    (_identity, _LINEAR),
    (jax.nn.identity, _LINEAR),
    (jnp.tanh, _SMOOTH),
    (jax.nn.sigmoid, _SMOOTH),
    (jax.nn.softplus, _SMOOTH),
    (jax.nn.silu, _SMOOTH),
    (jax.nn.gelu, _SMOOTH),
    (jnp.sin, _SMOOTH),
    (jnp.cos, _SMOOTH),
    (jnp.exp, _SMOOTH),
    (jax.nn.relu, _C0_PIECEWISE_LINEAR),
    (jax.nn.leaky_relu, _C0_PIECEWISE_LINEAR),
    (jax.nn.hard_tanh, _C0_PIECEWISE_LINEAR),
    (jax.nn.relu6, _C0_PIECEWISE_LINEAR),
    (squared_relu, _C1_PIECEWISE_QUADRATIC),
    # elu(alpha=1) and celu match slopes at the kink but not curvature.
    (jax.nn.elu, _C1_PIECEWISE_SMOOTH),
    (jax.nn.celu, _C1_PIECEWISE_SMOOTH),
    (jax.nn.selu, _C0_PIECEWISE_SMOOTH),
)


def _partial_regularity(fn: functools.partial, /) -> DerivativeRegularity | None:
    # Only keyword specializations of shape parameters keep a declared regularity;
    # positional arguments would bind the activation input itself. The classes
    # below hold for every parameter value except `elu`, which is C^1 only at a
    # static alpha of exactly one.
    if fn.args:
        return None
    keywords = set(fn.keywords)
    if fn.func is jax.nn.gelu and keywords <= {"approximate"}:
        return _SMOOTH
    if fn.func is jax.nn.leaky_relu and keywords <= {"negative_slope"}:
        return _C0_PIECEWISE_LINEAR
    if fn.func is jax.nn.celu and keywords <= {"alpha"}:
        return _C1_PIECEWISE_SMOOTH
    if fn.func is jax.nn.elu and keywords <= {"alpha"}:
        alpha = fn.keywords.get("alpha", 1.0)
        unit = not isinstance(alpha, bool) and isinstance(alpha, Real) and alpha == 1.0
        return _C1_PIECEWISE_SMOOTH if unit else _C0_PIECEWISE_SMOOTH
    return None


def activation_regularity(fn: Callable[..., Any], /) -> DerivativeRegularity | None:
    r"""Return the declared value regularity of a known activation callable.

    Activations are matched by identity (or, for trainable activations, by
    class), never by name:

    - identity: smooth polynomial of degree 1;
    - `tanh`, `sigmoid`, `softplus`, `silu`/`swish`, `gelu`, `sin`, `cos`,
      `exp`, and `Stan`: smooth;
    - `relu`, `leaky_relu`, `hard_tanh`, and `relu6`: $C^0$ piecewise linear;
    - `squared_relu`: $C^1$ piecewise quadratic;
    - `elu` with $\alpha = 1$ and `celu`: $C^1$ with smooth pieces; `elu` with
      $\alpha \ne 1$ and `selu`: $C^0$ with smooth pieces;
    - ModReLU (the complex activation of the Feynman path models): $C^0$ with
      smooth branches separated at $|z| = -b$;
    - `AdaptiveActivation`: the regularity of its wrapped activation.

    `functools.partial` specializations are recognized only for the shape
    keywords of `gelu`, `leaky_relu`, `elu`, and `celu`; `elu` is $C^1$ only for
    a static $\alpha = 1$.
    Spiking cells such as `ArtificialLIFCell` fire through a hard threshold and
    differentiate through a surrogate on the `RELAXED` route; they declare no
    classical regularity. Unknown callables return `None` (undeclared).
    """
    # Imported here: the Feynman models import this package's activations.
    from ..models._feynmann import _ModReLU

    if isinstance(fn, functools.partial):
        return _partial_regularity(fn)
    if isinstance(fn, Stan):
        return _SMOOTH
    if isinstance(fn, AdaptiveActivation):
        # x -> fn(a x): a linear inner map preserves the regularity of `fn`.
        return activation_regularity(fn.fn)
    if isinstance(fn, _ModReLU):
        return _C0_PIECEWISE_SMOOTH
    for known, regularity in _KNOWN_ACTIVATIONS:
        if fn is known:
            return regularity
    return None

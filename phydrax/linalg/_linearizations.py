#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._spaces import AbstractVectorSpace, PyTreeSpace


RematerializationPolicy: TypeAlias = Literal["store", "rematerialize"]


class LinearizationPolicy(StrictModule):
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    def __init__(self, rematerialization: RematerializationPolicy = "store", /):
        if rematerialization not in ("store", "rematerialize"):
            raise ValueError("Unknown linearization rematerialization policy.")
        self.rematerialization = rematerialization


class PreparedLinearization(StrictModule):
    """Reusable primal value, pushforward, and pullback at one fixed point."""

    source: AbstractVectorSpace
    target: AbstractVectorSpace
    point: PyTree[Array]
    primal: PyTree[Array]
    pushforward: Callable[[PyTree[Any]], PyTree[Array]]
    pullback: Callable[[PyTree[Any]], PyTree[Array]]
    auxiliary: Any
    policy: LinearizationPolicy
    linearization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        point: PyTree[Array],
        primal: PyTree[Array],
        pushforward: Callable[[PyTree[Any]], PyTree[Array]],
        pullback: Callable[[PyTree[Any]], PyTree[Array]],
        policy: LinearizationPolicy,
        linearization_id: str,
        auxiliary: Any = None,
    ):
        self.source = source
        self.target = target
        self.point = point
        self.primal = primal
        self.pushforward = pushforward
        self.pullback = pullback
        self.auxiliary = auxiliary
        self.policy = policy
        self.linearization_id = linearization_id

    def jvp(self, tangent: PyTree[Any], /) -> PyTree[Array]:
        return self.target.validate(self.pushforward(self.source.validate(tangent)))

    def vjp(self, cotangent: PyTree[Any], /) -> PyTree[Array]:
        return self.source.validate(self.pullback(self.target.validate(cotangent)))




def prepare_linearization(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    point: PyTree[Any],
    /,
    *,
    source: AbstractVectorSpace | None = None,
    target: AbstractVectorSpace | None = None,
    has_aux: bool = False,
    policy: LinearizationPolicy | None = None,
    linearization_id: str | None = None,
) -> PreparedLinearization:
    """Evaluate once and retain reusable JVP/VJP actions."""
    if not callable(function):
        raise TypeError("function must be callable.")
    policy_ = LinearizationPolicy() if policy is None else policy
    if not isinstance(policy_, LinearizationPolicy):
        raise TypeError("policy must be a LinearizationPolicy or None.")
    source_ = PyTreeSpace(point) if source is None else source
    if not isinstance(source_, AbstractVectorSpace):
        raise TypeError("source must be an AbstractVectorSpace or None.")
    point_ = source_.validate(point)
    converted = eqx.filter_closure_convert(function, source_.structure())
    executed = (
        jax.checkpoint(converted)
        if policy_.rematerialization == "rematerialize"
        else converted
    )
    if has_aux:
        primal, pushforward, auxiliary = jax.linearize(executed, point_, has_aux=True)
    else:
        primal, pushforward = jax.linearize(executed, point_)
        auxiliary = None
    pushforward = eqx.filter_closure_convert(pushforward, source_.structure())
    target_ = PyTreeSpace(primal) if target is None else target
    if not isinstance(target_, AbstractVectorSpace):
        raise TypeError("target must be an AbstractVectorSpace or None.")
    primal_ = target_.validate(primal)
    transposed = jax.linear_transpose(pushforward, source_.zeros())

    def apply_transpose(cotangent):
        return transposed(cotangent)[0]

    pullback = eqx.filter_closure_convert(apply_transpose, target_.structure())

    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-linearization",
                "source": source_.space_id,
                "target": target_.space_id,
                "rematerialization": policy_.rematerialization,
            }
        )
        if linearization_id is None
        else str(linearization_id)
    )
    if not identifier:
        raise ValueError("linearization_id must be non-empty.")
    return PreparedLinearization(
        source=source_,
        target=target_,
        point=point_,
        primal=primal_,
        pushforward=pushforward,
        pullback=pullback,
        auxiliary=auxiliary,
        policy=policy_,
        linearization_id=identifier,
    )


__all__ = [
    "LinearizationPolicy",
    "PreparedLinearization",
    "RematerializationPolicy",
    "prepare_linearization",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.extend import core as jax_core
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..typing import parse
from ._spaces import AbstractVectorSpace, PyTreeSpace


RematerializationPolicy: TypeAlias = Literal["store", "rematerialize"]


class LinearizationPolicy(StrictModule):
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    def __init__(self, rematerialization: RematerializationPolicy = "store", /) -> None:
        rematerialization = parse(
            rematerialization, RematerializationPolicy, "rematerialization"
        )
        self.rematerialization = rematerialization


@final
class _LinearizedPullback(StrictModule):
    """Transpose the one prepared tangent only when an adjoint is requested."""

    source: AbstractVectorSpace
    pushforward: Callable[[PyTree[Any]], PyTree[Array]]

    def __call__(self, cotangent: PyTree[Any], /) -> PyTree[Array]:
        return _linearized_pullback(self.pushforward, self.source.structure(), cotangent)


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
    ) -> None:
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
        return self.target.validate(
            _linearized_pushforward(self.pushforward, self.source.validate(tangent))
        )

    def vjp(self, cotangent: PyTree[Any], /) -> PyTree[Array]:
        return self.source.validate(self.pullback(self.target.validate(cotangent)))


@runtime_checkable
class _OutputStructuredCallable(Protocol):
    @property
    def out_struct(self) -> PyTree[Any]: ...

    def __call__(self, point: PyTree[Any], /) -> PyTree[Any]: ...


def _restore_auxiliary_arrays(
    auxiliary: PyTree[Any], structure: PyTree[Any], /
) -> PyTree[Any]:
    def restore(value: Any, specification: Any) -> Any:
        if isinstance(specification, jax.ShapeDtypeStruct):
            if isinstance(value, (Array, Tracer)):
                result = jnp.asarray(value, dtype=specification.dtype)
            else:
                result = jax.device_put(np.asarray(value, dtype=specification.dtype))
            if result.shape != specification.shape:
                raise ValueError("Linearization auxiliary array shape changed.")
            return result
        return value

    return jax.tree.map(
        restore,
        auxiliary,
        structure,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )


def _closure_convert(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    structure: PyTree[Any],
    /,
) -> Callable[[PyTree[Any]], PyTree[Any]]:
    converted = eqx.filter_closure_convert(function, structure)

    def open_program(leaf: Any) -> Any:
        # Unified Jaxprs duplicate Equinox's explicit numerical const leaves.
        # Only those dynamic leaves may own captured arrays across compilation.
        if isinstance(leaf, jax_core.Jaxpr) and leaf.consts:
            return leaf.with_consts(())
        return leaf

    return jax.tree.map(open_program, converted)


@eqx.filter_jit
def _evaluate_converted(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    value: PyTree[Any],
    /,
    *,
    has_aux: bool,
    auxiliary_structure: PyTree[Any],
) -> PyTree[Any]:
    output = function(value)
    if has_aux:
        primal, auxiliary = output
        return primal, _restore_auxiliary_arrays(auxiliary, auxiliary_structure)
    return output


@eqx.filter_jit
def _linearized_pushforward(
    pushforward: Callable[[PyTree[Any]], PyTree[Array]],
    tangent: PyTree[Any],
    /,
) -> PyTree[Array]:
    return pushforward(tangent)


@eqx.filter_jit
def _linearized_pullback(
    pushforward: Callable[[PyTree[Any]], PyTree[Array]],
    structure: PyTree[jax.ShapeDtypeStruct],
    cotangent: PyTree[Any],
    /,
) -> PyTree[Array]:
    # Transposition needs shape/dtype examples, not allocated numerical zeros.
    return jax.linear_transpose(pushforward, structure)(cotangent)[0]


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
    """Evaluate once and retain reusable JVP/VJP actions.

    Auxiliary outputs retain their declared array/static roles, including
    constant boolean and integer diagnostic arrays. Static metadata is unchanged.
    """
    if not callable(function):
        raise TypeError("function must be callable.")
    policy_ = LinearizationPolicy() if policy is None else policy
    if not isinstance(policy_, LinearizationPolicy):
        raise TypeError("policy must be a LinearizationPolicy or None.")
    source_ = PyTreeSpace(point) if source is None else source
    if not isinstance(source_, AbstractVectorSpace):
        raise TypeError("source must be an AbstractVectorSpace or None.")
    point_ = source_.validate(point)
    converted = _closure_convert(function, source_.structure())
    executed = (
        eqx.filter_checkpoint(converted)
        if policy_.rematerialization == "rematerialize"
        else converted
    )
    auxiliary_structure = None
    if has_aux:
        if not isinstance(converted, _OutputStructuredCallable):
            raise TypeError("Closure conversion must retain its output structure.")
        output_structure = converted.out_struct
        if not isinstance(output_structure, tuple) or len(output_structure) != 2:
            raise ValueError("Auxiliary linearization must return (value, auxiliary).")
        auxiliary_structure = output_structure[1]
    evaluate = eqx.Partial(
        _evaluate_converted,
        executed,
        has_aux=has_aux,
        auxiliary_structure=auxiliary_structure,
    )
    if has_aux:
        primal, pushforward, auxiliary = jax.linearize(evaluate, point_, has_aux=True)
        # AD can re-literalize constant array outputs of the compiled primal.
        # Restore their declared roles at the retained-tape boundary as well.
        auxiliary = _restore_auxiliary_arrays(auxiliary, auxiliary_structure)
    else:
        primal, pushforward = jax.linearize(evaluate, point_)
        auxiliary = None
    pushforward = _closure_convert(pushforward, source_.structure())
    target_ = PyTreeSpace(primal) if target is None else target
    if not isinstance(target_, AbstractVectorSpace):
        raise TypeError("target must be an AbstractVectorSpace or None.")
    primal_ = target_.validate(primal)
    pullback = _LinearizedPullback(source_, pushforward)

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

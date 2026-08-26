#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_dtype_name
from .._strict import StrictModule
from .._trainable import NonTrainableState


DiffraxComplexStateStrategy: TypeAlias = Literal["real_imag", "native", "reject"]
RealizedDiffraxStateStrategy: TypeAlias = Literal["real_imag", "native"]
ToleranceGeometry: TypeAlias = Literal["componentwise_real", "backend_native"]


class DiffraxComplexStatePolicy(StrictModule, NonTrainableState):
    """Select the backend representation used for a complex Diffrax state."""

    strategy: DiffraxComplexStateStrategy = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, strategy: DiffraxComplexStateStrategy = "real_imag"):
        if strategy not in ("real_imag", "native", "reject"):
            raise ValueError(
                "Diffrax complex-state strategy must be 'real_imag', 'native', or "
                "'reject'."
            )
        self.strategy = strategy
        self.policy_id = canonical_fingerprint(
            {
                "kind": "diffrax-complex-state-policy",
                "strategy": strategy,
            }
        )


class ComplexStatePackingEvidence(StrictModule, NonTrainableState):
    """Realized public-to-backend representation for one complex state."""

    strategy: RealizedDiffraxStateStrategy = eqx.field(static=True)
    public_dtype: str = eqx.field(static=True)
    backend_dtype: str = eqx.field(static=True)
    public_shape: tuple[int, ...] = eqx.field(static=True)
    backend_shape: tuple[int, ...] = eqx.field(static=True)
    tolerance_geometry: ToleranceGeometry = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        strategy: RealizedDiffraxStateStrategy,
        public_dtype: str,
        backend_dtype: str,
        public_shape: tuple[int, ...],
        backend_shape: tuple[int, ...],
        tolerance_geometry: ToleranceGeometry,
        policy_id: str,
    ):
        if strategy not in ("real_imag", "native"):
            raise ValueError("Unknown realized Diffrax state strategy.")
        if tolerance_geometry not in ("componentwise_real", "backend_native"):
            raise ValueError("Unknown Diffrax tolerance geometry.")
        public_shape_ = tuple(int(size) for size in public_shape)
        backend_shape_ = tuple(int(size) for size in backend_shape)
        if any(size <= 0 for size in public_shape_ + backend_shape_):
            raise ValueError("State packing shapes must contain positive dimensions.")
        if not public_dtype or not backend_dtype or not policy_id:
            raise ValueError("State packing dtype and policy IDs must be non-empty.")
        self.strategy = strategy
        self.public_dtype = public_dtype
        self.backend_dtype = backend_dtype
        self.public_shape = public_shape_
        self.backend_shape = backend_shape_
        self.tolerance_geometry = tolerance_geometry
        self.policy_id = policy_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "complex-state-packing-evidence",
                "strategy": strategy,
                "public_dtype": public_dtype,
                "backend_dtype": backend_dtype,
                "public_shape": list(public_shape_),
                "backend_shape": list(backend_shape_),
                "tolerance_geometry": tolerance_geometry,
                "policy": policy_id,
            }
        )


class _PackedComplexLeaf(StrictModule):
    real: Array
    imag: Array

    def __init__(self, value: ArrayLike, /):
        array = jnp.asarray(value)
        if not jnp.iscomplexobj(array):
            raise TypeError("Packed argument leaves must be complex-valued.")
        self.real = jnp.real(array)
        self.imag = jnp.imag(array)


def _is_complex_arraylike(value: Any, /) -> bool:
    return bool(eqx.is_array_like(value) and jnp.iscomplexobj(value))


def _pack_complex_tree(tree: Any, /) -> Any:
    return jax.tree.map(
        lambda value: (
            _PackedComplexLeaf(value) if _is_complex_arraylike(value) else value
        ),
        tree,
        is_leaf=_is_complex_arraylike,
    )


def _unpack_complex_tree(tree: Any, /) -> Any:
    return jax.tree.map(
        lambda value: (
            jax.lax.complex(value.real, value.imag)
            if isinstance(value, _PackedComplexLeaf)
            else value
        ),
        tree,
        is_leaf=lambda value: isinstance(value, _PackedComplexLeaf),
    )


class _PackedEventCondition(eqx.Module):
    condition: Any
    adapter: "_PreparedDiffraxStateAdapter"

    def __init__(self, condition: Any, adapter: "_PreparedDiffraxStateAdapter", /):
        self.condition = _pack_complex_tree(condition)
        self.adapter = adapter

    def __call__(self, t, y, args, **kwargs):
        condition = _unpack_complex_tree(self.condition)
        return condition(
            t,
            self.adapter.unpack_state(y),
            self.adapter.unpack_args(args),
            **kwargs,
        )


class _PreparedDiffraxStateAdapter(StrictModule, NonTrainableState):
    """Prepared identity or leading-axis real/imaginary state representation."""

    mode: RealizedDiffraxStateStrategy = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    backend_shape: tuple[int, ...] = eqx.field(static=True)
    public_dtype: str = eqx.field(static=True)
    backend_dtype: str = eqx.field(static=True)
    evidence: ComplexStatePackingEvidence | None

    def __init__(
        self,
        *,
        mode: RealizedDiffraxStateStrategy,
        state_shape: tuple[int, ...],
        public_dtype: str,
        backend_dtype: str,
        evidence: ComplexStatePackingEvidence | None,
    ):
        shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("Diffrax state shape must contain positive dimensions.")
        if mode not in ("real_imag", "native"):
            raise ValueError("Unknown prepared Diffrax state mode.")
        self.mode = mode
        self.state_shape = shape
        self.backend_shape = (2,) + shape if mode == "real_imag" else shape
        self.public_dtype = public_dtype
        self.backend_dtype = backend_dtype
        self.evidence = evidence

    @property
    def active(self) -> bool:
        return self.mode == "real_imag"

    def _public_value(self, value: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(value)
        if tuple(array.shape) != self.state_shape:
            raise ValueError(
                f"{owner} must have public state shape {self.state_shape}; "
                f"got {array.shape}."
            )
        return array.astype(jnp.dtype(self.public_dtype))

    def pack_state(self, value: ArrayLike, /, *, owner: str = "State") -> Array:
        array = self._public_value(value, owner)
        if not self.active:
            return array
        return jnp.stack((jnp.real(array), jnp.imag(array)), axis=0)

    def unpack_state(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if tuple(array.shape) != self.backend_shape:
            raise ValueError(
                f"Packed backend state must have shape {self.backend_shape}; "
                f"got {array.shape}."
            )
        if not self.active:
            return array
        return jax.lax.complex(array[0], array[1]).astype(jnp.dtype(self.public_dtype))

    def pack_diffusion(
        self,
        value: ArrayLike,
        noise_shape: tuple[int, ...],
        /,
    ) -> Array:
        array = jnp.asarray(value)
        expected = self.state_shape + tuple(int(size) for size in noise_shape)
        if tuple(array.shape) != expected:
            raise ValueError(
                f"Diffusion must have public state-plus-noise shape {expected}; "
                f"got {array.shape}."
            )
        array = array.astype(jnp.dtype(self.public_dtype))
        if not self.active:
            return array
        return jnp.stack((jnp.real(array), jnp.imag(array)), axis=0)

    def unpack_values(self, value: ArrayLike, pair_axis: int, /) -> Array:
        array = jnp.asarray(value)
        if not self.active:
            return array
        axis = int(pair_axis)
        if axis < 0:
            axis += array.ndim
        if axis < 0 or axis >= array.ndim or int(array.shape[axis]) != 2:
            raise ValueError(
                "Packed Diffrax values must expose one size-two real/imaginary axis."
            )
        real = jnp.take(array, 0, axis=axis)
        imag = jnp.take(array, 1, axis=axis)
        return jax.lax.complex(real, imag).astype(jnp.dtype(self.public_dtype))

    def pack_args(self, args: Any, /) -> Any:
        return _pack_complex_tree(args) if self.active else args

    def unpack_args(self, args: Any, /) -> Any:
        return _unpack_complex_tree(args) if self.active else args

    def wrap_event(self, event: Any | None, /) -> Any | None:
        if event is None or not self.active:
            return event
        if not isinstance(event, dfx.Event):
            raise TypeError(
                "Complex packed Diffrax solves require event to be diffrax.Event or None."
            )
        conditions = jax.tree.map(
            lambda condition: _PackedEventCondition(condition, self),
            event.cond_fn,
            is_leaf=callable,
        )
        return dfx.Event(
            conditions,
            root_finder=event.root_finder,
            direction=event.direction,
        )


def _prepare_diffrax_state_adapter(
    initial_state: ArrayLike,
    policy: DiffraxComplexStatePolicy | None,
    state_geometry: Any | None,
    /,
) -> _PreparedDiffraxStateAdapter:
    state = jnp.asarray(initial_state)
    shape = tuple(int(size) for size in state.shape)
    public_dtype = precision_dtype_name(state.dtype)
    resolved = DiffraxComplexStatePolicy() if policy is None else policy
    if not isinstance(resolved, DiffraxComplexStatePolicy):
        raise TypeError("complex_state_policy must be DiffraxComplexStatePolicy or None.")
    if not jnp.iscomplexobj(state):
        return _PreparedDiffraxStateAdapter(
            mode="native",
            state_shape=shape,
            public_dtype=public_dtype,
            backend_dtype=public_dtype,
            evidence=None,
        )
    if resolved.strategy == "reject":
        raise ValueError("Complex Diffrax state was rejected by the selected policy.")
    if resolved.strategy == "native":
        evidence = ComplexStatePackingEvidence(
            strategy="native",
            public_dtype=public_dtype,
            backend_dtype=public_dtype,
            public_shape=shape,
            backend_shape=shape,
            tolerance_geometry="backend_native",
            policy_id=resolved.policy_id,
        )
        return _PreparedDiffraxStateAdapter(
            mode="native",
            state_shape=shape,
            public_dtype=public_dtype,
            backend_dtype=public_dtype,
            evidence=evidence,
        )
    if state_geometry is not None and not state_geometry.trivial:
        raise ValueError(
            "Real/imaginary Diffrax packing requires trivial Euclidean state geometry; "
            "select native complex execution explicitly or provide a real formulation."
        )
    backend_dtype = precision_dtype_name(state.real.dtype)
    evidence = ComplexStatePackingEvidence(
        strategy="real_imag",
        public_dtype=public_dtype,
        backend_dtype=backend_dtype,
        public_shape=shape,
        backend_shape=(2,) + shape,
        tolerance_geometry="componentwise_real",
        policy_id=resolved.policy_id,
    )
    return _PreparedDiffraxStateAdapter(
        mode="real_imag",
        state_shape=shape,
        public_dtype=public_dtype,
        backend_dtype=backend_dtype,
        evidence=evidence,
    )


def _validate_real_backend_tree(tree: Any, /) -> None:
    if any(
        eqx.is_array_like(value) and jnp.iscomplexobj(value)
        for value in jax.tree.leaves(tree)
    ):
        raise ValueError(
            "Real/imaginary Diffrax packing left a visible complex backend leaf."
        )


__all__ = [
    "ComplexStatePackingEvidence",
    "DiffraxComplexStatePolicy",
    "DiffraxComplexStateStrategy",
]

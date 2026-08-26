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
from ..linalg import (
    ArraySpace,
    ComplexCartesianCoordinates,
    PreparedAlgebraCoordinates,
)


DiffraxComplexStateStrategy: TypeAlias = Literal["real_imag", "native", "reject"]
RealizedDiffraxStateStrategy: TypeAlias = Literal[
    "real_imag",
    "native",
    "algebra_coordinates",
]
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


class DiffraxAlgebraStatePolicy(StrictModule, NonTrainableState):
    """Bind an explicit algebra-coordinate map to a Diffrax solve."""

    coordinates: PreparedAlgebraCoordinates
    policy_id: str = eqx.field(static=True)

    def __init__(self, coordinates: PreparedAlgebraCoordinates, /):
        if not isinstance(coordinates, PreparedAlgebraCoordinates):
            raise TypeError("coordinates must be PreparedAlgebraCoordinates.")
        self.coordinates = coordinates
        self.policy_id = canonical_fingerprint(
            {
                "kind": "diffrax-algebra-state-policy-v1",
                "coordinates": coordinates.coordinate_id,
                "algebra": coordinates.plan.algebra.algebra_id,
            }
        )


class AlgebraStatePackingEvidence(StrictModule, NonTrainableState):
    algebra_id: str = eqx.field(static=True)
    coordinate_evidence_id: str = eqx.field(static=True)
    public_dtype: str = eqx.field(static=True)
    backend_dtype: str = eqx.field(static=True)
    public_shape: tuple[int, ...] = eqx.field(static=True)
    backend_shape: tuple[int, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinates: PreparedAlgebraCoordinates,
        policy_id: str,
    ):
        if not isinstance(coordinates, PreparedAlgebraCoordinates):
            raise TypeError("coordinates must be PreparedAlgebraCoordinates.")
        self.algebra_id = coordinates.plan.algebra.algebra_id
        self.coordinate_evidence_id = coordinates.evidence.evidence_id
        self.public_dtype = coordinates.evidence.source_dtype
        self.backend_dtype = coordinates.evidence.coordinate_dtype
        self.public_shape = coordinates.public_shape
        self.backend_shape = coordinates.coordinate_space.shape
        self.policy_id = str(policy_id)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "algebra-state-packing-evidence-v1",
                "algebra": self.algebra_id,
                "coordinates": self.coordinate_evidence_id,
                "public_dtype": self.public_dtype,
                "backend_dtype": self.backend_dtype,
                "public_shape": list(self.public_shape),
                "backend_shape": list(self.backend_shape),
                "policy": self.policy_id,
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
    evidence: ComplexStatePackingEvidence | AlgebraStatePackingEvidence | None
    coordinates: ComplexCartesianCoordinates | PreparedAlgebraCoordinates | None

    def __init__(
        self,
        *,
        mode: RealizedDiffraxStateStrategy,
        state_shape: tuple[int, ...],
        public_dtype: str,
        backend_dtype: str,
        evidence: ComplexStatePackingEvidence | AlgebraStatePackingEvidence | None,
        coordinates: PreparedAlgebraCoordinates | None = None,
    ):
        shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("Diffrax state shape must contain positive dimensions.")
        if mode not in ("real_imag", "native", "algebra_coordinates"):
            raise ValueError("Unknown prepared Diffrax state mode.")
        resolved_coordinates: (
            ComplexCartesianCoordinates | PreparedAlgebraCoordinates | None
        ) = None
        backend_shape = shape
        if mode == "real_imag":
            source_space = ArraySpace(
                shape,
                dtype=jnp.dtype(public_dtype),
                space_id=canonical_fingerprint(
                    {
                        "kind": "diffrax-complex-source-space-v1",
                        "shape": list(shape),
                        "dtype": public_dtype,
                    }
                ),
            )
            resolved_coordinates = ComplexCartesianCoordinates(source_space, pair_axis=0)
            backend_shape = resolved_coordinates.coordinate_space.shape
        elif mode == "algebra_coordinates":
            if not isinstance(coordinates, PreparedAlgebraCoordinates):
                raise TypeError("Algebra Diffrax mode requires prepared coordinates.")
            if coordinates.public_shape != shape:
                raise ValueError(
                    "Algebra coordinates do not match the public state shape."
                )
            resolved_coordinates = coordinates
            backend_shape = coordinates.coordinate_space.shape
        self.mode = mode
        self.state_shape = shape
        self.backend_shape = backend_shape
        self.public_dtype = public_dtype
        self.backend_dtype = backend_dtype
        self.evidence = evidence
        self.coordinates = resolved_coordinates

    @property
    def active(self) -> bool:
        return self.mode != "native"

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
        if self.coordinates is None:
            return array
        return self.coordinates.to_real_coordinates(array)

    def unpack_state(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if tuple(array.shape) != self.backend_shape:
            raise ValueError(
                f"Packed backend state must have shape {self.backend_shape}; "
                f"got {array.shape}."
            )
        if self.coordinates is None:
            return array
        return self.coordinates.from_real_coordinates(array)

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
        if self.coordinates is None:
            return array
        return self.coordinates.pack_diffusion(
            array, tuple(int(size) for size in noise_shape)
        )

    def unpack_values(self, value: ArrayLike, pair_axis: int, /) -> Array:
        array = jnp.asarray(value)
        if self.coordinates is None:
            return array
        return self.coordinates.unpack_values(array, pair_axis)

    def pack_args(self, args: Any, /) -> Any:
        return _pack_complex_tree(args) if self.active else args

    def unpack_args(self, args: Any, /) -> Any:
        return _unpack_complex_tree(args) if self.active else args

    def wrap_event(self, event: Any | None, /) -> Any | None:
        if event is None or not self.active:
            return event
        if not isinstance(event, dfx.Event):
            raise TypeError(
                "Packed Diffrax solves require event to be diffrax.Event or None."
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
    algebra_policy: DiffraxAlgebraStatePolicy | None,
    state_geometry: Any | None,
    /,
) -> _PreparedDiffraxStateAdapter:
    state = jnp.asarray(initial_state)
    shape = tuple(int(size) for size in state.shape)
    public_dtype = precision_dtype_name(state.dtype)
    if algebra_policy is not None:
        if policy is not None:
            raise ValueError(
                "complex_state_policy and algebra_state_policy are mutually exclusive."
            )
        if not isinstance(algebra_policy, DiffraxAlgebraStatePolicy):
            raise TypeError(
                "algebra_state_policy must be DiffraxAlgebraStatePolicy or None."
            )
        if state_geometry is not None and not state_geometry.trivial:
            raise ValueError(
                "Algebra-coordinate Diffrax execution requires trivial Euclidean "
                "state geometry."
            )
        coordinates = algebra_policy.coordinates
        coordinates.validate_state(state)
        evidence = AlgebraStatePackingEvidence(
            coordinates=coordinates,
            policy_id=algebra_policy.policy_id,
        )
        return _PreparedDiffraxStateAdapter(
            mode="algebra_coordinates",
            state_shape=shape,
            public_dtype=public_dtype,
            backend_dtype=evidence.backend_dtype,
            evidence=evidence,
            coordinates=coordinates,
        )
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
    "AlgebraStatePackingEvidence",
    "ComplexStatePackingEvidence",
    "DiffraxAlgebraStatePolicy",
    "DiffraxComplexStatePolicy",
    "DiffraxComplexStateStrategy",
]

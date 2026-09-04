#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
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
    AbstractRealCoordinateMap,
    ArraySpace,
    ComplexCartesianCoordinates,
    RealCoordinateEvidence,
)


DiffraxComplexStateStrategy: TypeAlias = Literal["real_coordinates", "native", "reject"]
RealizedDiffraxStateStrategy: TypeAlias = Literal["real_coordinates", "native"]


class DiffraxComplexStatePolicy(StrictModule, NonTrainableState):
    """Select the declared backend representation for a complex Diffrax state."""

    strategy: DiffraxComplexStateStrategy = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, strategy: DiffraxComplexStateStrategy = "real_coordinates", /):
        if strategy not in ("real_coordinates", "native", "reject"):
            raise ValueError(
                "Diffrax complex-state strategy must be 'real_coordinates', "
                "'native', or 'reject'."
            )
        self.strategy = strategy
        self.policy_id = canonical_fingerprint(
            {
                "kind": "diffrax-complex-state-policy",
                "strategy": strategy,
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
    state_adapter: "_PreparedDiffraxStateAdapter"

    def __call__(self, t, y, args, **kwargs):
        return self.condition(
            t,
            self.state_adapter.unpack_state(y),
            self.state_adapter.unpack_args(args),
            **kwargs,
        )


class _PreparedDiffraxStateAdapter(StrictModule, NonTrainableState):
    """One immutable public-state to backend-coordinate action."""

    mode: RealizedDiffraxStateStrategy = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    backend_shape: tuple[int, ...] = eqx.field(static=True)
    public_dtype: str = eqx.field(static=True)
    backend_dtype: str = eqx.field(static=True)
    tree_mode: bool = eqx.field(static=True)
    evidence: RealCoordinateEvidence | None
    coordinates: AbstractRealCoordinateMap | None

    def __init__(
        self,
        *,
        mode: RealizedDiffraxStateStrategy,
        state_shape: tuple[int, ...],
        public_dtype: str,
        backend_dtype: str,
        evidence: RealCoordinateEvidence | None,
        coordinates: AbstractRealCoordinateMap | None = None,
    ):
        shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("Diffrax state shape must contain positive dimensions.")
        if mode not in ("real_coordinates", "native"):
            raise ValueError("Unknown prepared Diffrax state mode.")
        if mode == "real_coordinates":
            if not isinstance(coordinates, AbstractRealCoordinateMap):
                raise TypeError(
                    "Real-coordinate mode requires a declared coordinate map."
                )
            if not isinstance(evidence, RealCoordinateEvidence):
                raise TypeError("Real-coordinate mode requires canonical evidence.")
            source_spec = coordinates.source_space.structure()
            coordinate_spec = coordinates.coordinate_space.structure()
            tree_mode = not isinstance(source_spec, jax.ShapeDtypeStruct)
            if not isinstance(coordinate_spec, jax.ShapeDtypeStruct):
                raise TypeError(
                    "Diffrax real-coordinate backends require one flat array target."
                )
            if tree_mode:
                backend_shape = tuple(int(size) for size in coordinate_spec.shape)
            else:
                if tuple(source_spec.shape) != shape:
                    raise ValueError(
                        "The coordinate-map source shape does not match the state."
                    )
                backend_shape = tuple(int(size) for size in coordinate_spec.shape)
            if not jnp.issubdtype(coordinate_spec.dtype, jnp.floating):
                raise TypeError(
                    "Diffrax backend coordinates must be real floating arrays."
                )
        else:
            if coordinates is not None or evidence is not None:
                raise ValueError("Native mode cannot retain real-coordinate evidence.")
            backend_shape = shape
            tree_mode = False
        self.mode = mode
        self.state_shape = shape
        self.backend_shape = backend_shape
        self.public_dtype = str(public_dtype)
        self.backend_dtype = str(backend_dtype)
        self.evidence = evidence
        self.coordinates = coordinates
        self.tree_mode = tree_mode

    @property
    def active(self) -> bool:
        return self.mode == "real_coordinates"

    def _public_value(self, value: Any, owner: str, /) -> Any:
        if self.tree_mode:
            assert self.coordinates is not None
            return self.coordinates.validate_state(value)
        array = jnp.asarray(value)
        if tuple(array.shape) != self.state_shape:
            raise ValueError(
                f"{owner} must have public state shape {self.state_shape}; "
                f"got {array.shape}."
            )
        return array.astype(jnp.dtype(self.public_dtype))

    def pack_state(self, value: Any, /, *, owner: str = "State") -> Any:
        public = self._public_value(value, owner)
        if self.coordinates is None:
            return public
        defect = self.coordinates.defect(public)
        leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(public))
        epsilon = max(float(jnp.finfo(leaf.real.dtype).eps) for leaf in leaves)
        predicate = ~jnp.isfinite(defect) | (defect > 128.0 * epsilon)
        checked = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                predicate,
                f"{owner} violates its declared real-coordinate domain.",
            ),
            public,
        )
        return self.coordinates.to_real_coordinates(checked)

    def unpack_state(self, value: Any, /) -> Any:
        if self.coordinates is None:
            array = jnp.asarray(value)
            if tuple(array.shape) != self.backend_shape:
                raise ValueError(
                    "Packed backend state does not match its prepared shape."
                )
            return array.astype(jnp.dtype(self.public_dtype))
        if self.tree_mode:
            return self.coordinates.from_real_coordinates(
                self.coordinates.validate_coordinates(value)
            )
        array = jnp.asarray(value)
        if tuple(array.shape) != self.backend_shape:
            raise ValueError(
                f"Packed backend state must have shape {self.backend_shape}; "
                f"got {array.shape}."
            )
        return self.coordinates.from_real_coordinates(array)

    def pack_tangent(
        self,
        value: ArrayLike,
        tangent_shape: tuple[int, ...],
        /,
        *,
        owner: str = "Tangent",
    ) -> Array:
        array = jnp.asarray(value)
        expected = tuple(int(size) for size in tangent_shape)
        if tuple(array.shape) != expected:
            raise ValueError(
                f"{owner} must have public tangent shape {expected}; got {array.shape}."
            )
        if self.coordinates is None:
            return array.astype(jnp.dtype(self.public_dtype))
        if expected != self.state_shape:
            raise ValueError(
                "Real-coordinate Diffrax packing requires equal point and tangent "
                "shapes unless a tangent coordinate map is declared."
            )
        return self.pack_state(array, owner=owner)

    def unpack_tangent(
        self,
        value: ArrayLike,
        tangent_shape: tuple[int, ...],
        /,
    ) -> Array:
        expected = tuple(int(size) for size in tangent_shape)
        if self.coordinates is None:
            array = jnp.asarray(value)
            if tuple(array.shape) != expected:
                raise ValueError(
                    f"Packed backend tangent must have shape {expected}; "
                    f"got {array.shape}."
                )
            return array.astype(jnp.dtype(self.public_dtype))
        if expected != self.state_shape:
            raise ValueError(
                "Real-coordinate Diffrax unpacking requires equal point and tangent "
                "shapes unless a tangent coordinate map is declared."
            )
        return self.unpack_state(value)

    def unpack_tangent_values(
        self,
        value: ArrayLike,
        sample_rank: int,
        tangent_shape: tuple[int, ...],
        /,
    ) -> Array:
        rank = int(sample_rank)
        expected = tuple(int(size) for size in tangent_shape)
        array = jnp.asarray(value)
        if rank < 0 or rank > array.ndim:
            raise ValueError("sample_rank lies outside the tangent value rank.")
        if self.coordinates is None:
            if tuple(array.shape[rank:]) != expected:
                raise ValueError(
                    "Saved backend tangents do not end with the prepared tangent shape."
                )
            return array.astype(jnp.dtype(self.public_dtype))
        if expected != self.state_shape:
            raise ValueError(
                "Real-coordinate Diffrax unpacking requires equal point and tangent "
                "shapes unless a tangent coordinate map is declared."
            )
        return self.unpack_values(array, rank)

    def pack_diffusion(
        self,
        value: ArrayLike,
        noise_shape: tuple[int, ...],
        /,
        *,
        output_shape: tuple[int, ...] | None = None,
    ) -> Array:
        if self.tree_mode:
            raise ValueError(
                "PyTree stochastic diffusion requires a declared tree noise layout."
            )
        array = jnp.asarray(value)
        trailing = tuple(int(size) for size in noise_shape)
        leading = (
            self.state_shape
            if output_shape is None
            else tuple(int(size) for size in output_shape)
        )
        expected = leading + trailing
        if tuple(array.shape) != expected:
            raise ValueError(
                f"Diffusion must have public tangent-plus-noise shape {expected}; "
                f"got {array.shape}."
            )
        array = array.astype(jnp.dtype(self.public_dtype))
        if self.coordinates is None:
            return array
        if leading != self.state_shape:
            raise ValueError(
                "Real-coordinate Diffrax diffusion packing requires equal point and "
                "tangent shapes unless a tangent coordinate map is declared."
            )
        noise_size = math.prod(trailing) if trailing else 1
        columns = array.reshape(leading + (noise_size,))
        packed = jax.vmap(
            self.coordinates.to_real_coordinates,
            in_axes=-1,
            out_axes=-1,
        )(columns)
        return packed.reshape(self.backend_shape + trailing)

    def unpack_values(self, value: Any, sample_rank: int, /) -> Any:
        rank = int(sample_rank)
        if self.coordinates is not None and self.tree_mode:
            array = jnp.asarray(value)
            if rank < 0 or rank > array.ndim:
                raise ValueError("sample_rank lies outside the saved backend value rank.")
            if tuple(array.shape[rank:]) != self.backend_shape:
                raise ValueError(
                    "Saved backend values do not end with the prepared coordinate shape."
                )
            sample_shape = tuple(array.shape[:rank])
            flattened = array.reshape((-1,) + self.backend_shape)
            public = jax.vmap(self.coordinates.from_real_coordinates)(flattened)
            source_specs = self.coordinates.source_space.structure()
            return jax.tree.map(
                lambda leaf, spec: leaf.reshape(sample_shape + tuple(spec.shape)),
                public,
                source_specs,
            )
        array = jnp.asarray(value)
        if rank < 0 or rank > array.ndim:
            raise ValueError("sample_rank lies outside the saved backend value rank.")
        if self.coordinates is None:
            return array.astype(jnp.dtype(self.public_dtype))
        if tuple(array.shape[rank:]) != self.backend_shape:
            raise ValueError(
                "Saved backend values do not end with the prepared coordinate shape."
            )
        sample_shape = tuple(array.shape[:rank])
        flattened = array.reshape((-1,) + self.backend_shape)
        public = jax.vmap(self.coordinates.from_real_coordinates)(flattened)
        return public.reshape(sample_shape + self.state_shape)

    def pack_args(self, args: Any, /) -> Any:
        return _pack_complex_tree(args) if self.active else args

    def unpack_args(self, args: Any, /) -> Any:
        return _unpack_complex_tree(args) if self.active else args

    def wrap_event(self, event: Any | None, /) -> Any | None:
        if event is None or not self.active:
            return event
        if not isinstance(event, dfx.Event):
            raise TypeError(
                "Real-coordinate Diffrax solves require event to be diffrax.Event or None."
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
    initial_state: Any,
    policy: DiffraxComplexStatePolicy | None,
    state_coordinates: AbstractRealCoordinateMap | None,
    state_geometry: Any | None,
    /,
) -> _PreparedDiffraxStateAdapter:
    state_is_array = eqx.is_array_like(initial_state)
    state = (
        jnp.asarray(initial_state)
        if state_is_array
        else jax.tree.map(jnp.asarray, initial_state)
    )
    resolved = DiffraxComplexStatePolicy() if policy is None else policy
    if not isinstance(resolved, DiffraxComplexStatePolicy):
        raise TypeError("complex_state_policy must be DiffraxComplexStatePolicy or None.")
    if state_coordinates is not None and not isinstance(
        state_coordinates, AbstractRealCoordinateMap
    ):
        raise TypeError("state_coordinates must be AbstractRealCoordinateMap or None.")
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(state))
    if not leaves:
        raise ValueError("Diffrax state must contain array leaves.")
    complex_state = any(jnp.iscomplexobj(leaf) for leaf in leaves)
    if not state_is_array and state_coordinates is None:
        raise ValueError(
            "PyTree Diffrax states require an explicit PreparedRealCoordinateTree."
        )
    if state_is_array:
        shape = tuple(int(size) for size in state.shape)
        public_dtype = precision_dtype_name(leaves[0].dtype)
    else:
        assert state_coordinates is not None
        shape = state_coordinates.evidence.source_shape
        public_dtype = state_coordinates.evidence.source_dtype
    if resolved.strategy == "native":
        if state_coordinates is not None:
            raise ValueError("Native complex execution cannot accept state_coordinates.")
        return _PreparedDiffraxStateAdapter(
            mode="native",
            state_shape=shape,
            public_dtype=public_dtype,
            backend_dtype=public_dtype,
            evidence=None,
        )
    if resolved.strategy == "reject" and complex_state:
        raise ValueError("Complex Diffrax state was rejected by the selected policy.")
    if resolved.strategy == "reject" and state_coordinates is not None:
        raise ValueError("The reject strategy cannot accept state_coordinates.")
    coordinates = state_coordinates
    if coordinates is None and complex_state:
        coordinates = ComplexCartesianCoordinates(
            ArraySpace(
                shape,
                dtype=leaves[0].dtype,
                space_id=canonical_fingerprint(
                    {
                        "kind": "diffrax-complex-source-space",
                        "shape": list(shape),
                        "dtype": public_dtype,
                    }
                ),
            ),
            pair_axis=0,
        )
    if coordinates is None:
        return _PreparedDiffraxStateAdapter(
            mode="native",
            state_shape=shape,
            public_dtype=public_dtype,
            backend_dtype=public_dtype,
            evidence=None,
        )
    if (
        state_geometry is not None
        and not state_geometry.trivial
        and state_coordinates is None
    ):
        raise ValueError(
            "Nontrivial state geometry requires an explicit linear real-coordinate map."
        )
    if state_geometry is not None and coordinates is not None:
        point = jnp.asarray(state)
        tangent = jnp.asarray(
            state_geometry.project_tangent(point, jnp.zeros_like(point))
        )
        if tangent.shape != point.shape:
            raise ValueError(
                "Unequal point and tangent spaces require native Diffrax state "
                "packing until a tangent coordinate map is declared."
            )
    validated = coordinates.validate_state(state)
    defect = coordinates.defect(validated)
    if not bool(jnp.isfinite(defect)):
        raise ValueError("Real-coordinate state defect must be finite.")
    coordinate_spec = coordinates.coordinate_space.structure()
    coordinate_leaves = tuple(
        jnp.asarray(leaf)
        for leaf in jax.tree.leaves(coordinates.to_real_coordinates(validated))
    )
    if any(not jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in coordinate_leaves):
        raise TypeError("Diffrax backend coordinates must be real floating arrays.")
    backend_dtype = coordinates.evidence.coordinate_dtype
    return _PreparedDiffraxStateAdapter(
        mode="real_coordinates",
        state_shape=shape,
        public_dtype=public_dtype,
        backend_dtype=backend_dtype,
        evidence=coordinates.evidence,
        coordinates=coordinates,
    )


def _validate_real_backend_tree(tree: Any, /) -> None:
    if any(
        eqx.is_array_like(value) and jnp.iscomplexobj(value)
        for value in jax.tree.leaves(tree)
    ):
        raise ValueError(
            "Real-coordinate Diffrax packing left a visible complex backend leaf."
        )


__all__ = [
    "DiffraxComplexStatePolicy",
    "DiffraxComplexStateStrategy",
]

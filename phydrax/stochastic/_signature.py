#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import pairwise, product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ._fractional import FractionalGaussianRealization
from ._rough import AbstractRoughControl


Word = tuple[int, ...]


def _validate_depth(depth: int, /) -> int:
    resolved = int(depth)
    if resolved <= 0:
        raise ValueError("depth must be positive.")
    return resolved


def _inexact_array(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.result_type(array, jnp.asarray(0.0)))
    return array


def _signature_shape(signature: Sequence[ArrayLike], /) -> tuple[tuple[Array, ...], int]:
    levels = tuple(jnp.asarray(level) for level in signature)
    if not levels:
        raise ValueError("A signature must contain at least one level.")
    if levels[0].ndim < 1 or int(levels[0].shape[-1]) <= 0:
        raise ValueError("Signature level one must end in a non-empty driver axis.")
    dimension = int(levels[0].shape[-1])
    batch_shape = levels[0].shape[:-1]
    for degree, level in enumerate(levels, start=1):
        expected = batch_shape + (dimension,) * degree
        if level.shape != expected:
            raise ValueError(
                f"Signature level {degree} must have shape {expected}; got {level.shape}."
            )
    return levels, dimension


def _tensor_product(
    left: Array, right: Array, left_degree: int, right_degree: int
) -> Array:
    batch_shape = left.shape[:-left_degree]
    if right.shape[:-right_degree] != batch_shape:
        raise ValueError("Tensor factors must have matching batch shapes.")
    return left.reshape(
        batch_shape + left.shape[-left_degree:] + (1,) * right_degree
    ) * right.reshape(batch_shape + (1,) * left_degree + right.shape[-right_degree:])


def _signature_identity(
    batch_shape: tuple[int, ...],
    dimension: int,
    depth: int,
    dtype,
    /,
) -> tuple[Array, ...]:
    return tuple(
        jnp.zeros(batch_shape + (dimension,) * degree, dtype=dtype)
        for degree in range(1, depth + 1)
    )


def _chen_multiply_increment(
    signature: tuple[Array, ...],
    increment: Array,
    /,
) -> tuple[Array, ...]:
    """Multiply a validated signature by one straight-segment signature."""
    straight = tensor_exponential(increment, len(signature))
    result: list[Array] = []
    for degree in range(1, len(signature) + 1):
        level = signature[degree - 1] + straight[degree - 1]
        for split in range(1, degree):
            level = level + _tensor_product(
                signature[split - 1],
                straight[degree - split - 1],
                split,
                degree - split,
            )
        result.append(level)
    return tuple(result)


def tensor_exponential(increment: ArrayLike, depth: int, /) -> tuple[Array, ...]:
    """Return the truncated tensor exponential of one or batched increments."""
    value = _inexact_array(increment)
    resolved_depth = _validate_depth(depth)
    if value.ndim < 1 or int(value.shape[-1]) <= 0:
        raise ValueError("increment must end in a non-empty driver axis.")
    levels: list[Array] = [value]
    for degree in range(2, resolved_depth + 1):
        levels.append(_tensor_product(levels[-1], value, degree - 1, 1) / float(degree))
    return tuple(levels)


def chen_multiply(
    left: Sequence[ArrayLike], right: Sequence[ArrayLike], /
) -> tuple[Array, ...]:
    """Multiply equal-depth truncated tensor signatures using Chen's identity."""
    left_levels, left_dimension = _signature_shape(left)
    right_levels, right_dimension = _signature_shape(right)
    if len(left_levels) != len(right_levels) or left_dimension != right_dimension:
        raise ValueError("Chen factors must have equal depth and driver dimension.")
    if left_levels[0].shape[:-1] != right_levels[0].shape[:-1]:
        raise ValueError("Chen factors must have matching batch shapes.")
    result: list[Array] = []
    for degree in range(1, len(left_levels) + 1):
        level = left_levels[degree - 1] + right_levels[degree - 1]
        for split in range(1, degree):
            level = level + _tensor_product(
                left_levels[split - 1],
                right_levels[degree - split - 1],
                split,
                degree - split,
            )
        result.append(level)
    return tuple(result)


def _nonunital_product(
    left: tuple[Array, ...], right: tuple[Array, ...], /
) -> tuple[Array, ...]:
    depth = len(left)
    result: list[Array] = []
    for degree in range(1, depth + 1):
        level = jnp.zeros_like(left[degree - 1])
        for split in range(1, degree):
            level = level + _tensor_product(
                left[split - 1], right[degree - split - 1], split, degree - split
            )
        result.append(level)
    return tuple(result)


def tensor_logarithm(signature: Sequence[ArrayLike], /) -> tuple[Array, ...]:
    """Compute ``log(1 + S)`` in the truncated tensor algebra."""
    levels, _ = _signature_shape(signature)
    power = levels
    result = tuple(jnp.zeros_like(level) for level in levels)
    for exponent in range(1, len(levels) + 1):
        coefficient = (1.0 if exponent % 2 else -1.0) / float(exponent)
        result = tuple(
            accumulated + coefficient * term for accumulated, term in zip(result, power)
        )
        power = _nonunital_product(power, levels)
    return result


def piecewise_linear_signature(
    increments: ArrayLike,
    depth: int,
    /,
    *,
    stream: bool = False,
) -> tuple[Array, ...]:
    """Aggregate straight-segment signatures along the penultimate axis.

    Streaming output retains the signature after each supplied increment. The
    empty signature is not included because increments do not declare an
    initial knot.
    """
    values = _inexact_array(increments)
    resolved_depth = _validate_depth(depth)
    if values.ndim < 2 or int(values.shape[-1]) <= 0:
        raise ValueError(
            "increments must have shape batch_shape + (num_segments, dimension)."
        )
    dimension = int(values.shape[-1])
    batch_shape = tuple(int(size) for size in values.shape[:-2])
    initial = _signature_identity(
        batch_shape,
        dimension,
        resolved_depth,
        values.dtype,
    )
    scan_values = jnp.moveaxis(values, -2, 0)

    if stream:

        def combine_stream(carry, increment):
            updated = _chen_multiply_increment(carry, increment)
            return updated, updated

        _, history = jax.lax.scan(combine_stream, initial, scan_values)
        sequence_axis = len(batch_shape)
        return tuple(jnp.moveaxis(level, 0, sequence_axis) for level in history)

    def combine_terminal(carry, increment):
        return _chen_multiply_increment(carry, increment), None

    return jax.lax.scan(combine_terminal, initial, scan_values)[0]


def _is_lyndon(word: Word, /) -> bool:
    return all(word < word[index:] for index in range(1, len(word)))


def _concatenate_expansions(
    left: dict[Word, int], right: dict[Word, int], /
) -> dict[Word, int]:
    result: dict[Word, int] = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            word = left_word + right_word
            result[word] = result.get(word, 0) + left_coefficient * right_coefficient
    return result


def _subtract_expansions(
    left: dict[Word, int], right: dict[Word, int], /
) -> dict[Word, int]:
    result = dict(left)
    for word, coefficient in right.items():
        result[word] = result.get(word, 0) - coefficient
        if result[word] == 0:
            del result[word]
    return result


class PrimitiveBasis(StrictModule):
    """Standard-bracketed Lyndon basis constructed statically in Python."""

    dimension: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    words: tuple[Word, ...] = eqx.field(static=True)
    degrees: tuple[int, ...] = eqx.field(static=True)
    children: tuple[tuple[int, int] | None, ...] = eqx.field(static=True)
    word_expansions: tuple[tuple[tuple[Word, int], ...], ...] = eqx.field(static=True)
    degree_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    conversion_inverses: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(
        static=True
    )
    expansion_matrices: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(static=True)

    def __init__(self, dimension: int, depth: int, /):
        resolved_dimension = int(dimension)
        resolved_depth = _validate_depth(depth)
        if resolved_dimension <= 0:
            raise ValueError("dimension must be positive.")
        words = tuple(
            word
            for degree in range(1, resolved_depth + 1)
            for word in product(range(resolved_dimension), repeat=degree)
            if _is_lyndon(word)
        )
        word_to_index = {word: index for index, word in enumerate(words)}
        lyndon_set = frozenset(words)
        children: list[tuple[int, int] | None] = []
        expansions: list[dict[Word, int]] = []
        for word in words:
            if len(word) == 1:
                children.append(None)
                expansions.append({word: 1})
                continue
            split = next(
                index for index in range(1, len(word)) if word[index:] in lyndon_set
            )
            left_word = word[:split]
            right_word = word[split:]
            left_index = word_to_index[left_word]
            right_index = word_to_index[right_word]
            children.append((left_index, right_index))
            left_right = _concatenate_expansions(
                expansions[left_index], expansions[right_index]
            )
            right_left = _concatenate_expansions(
                expansions[right_index], expansions[left_index]
            )
            expansions.append(_subtract_expansions(left_right, right_left))
        degree_indices: list[tuple[int, ...]] = []
        inverses: list[tuple[tuple[float, ...], ...]] = []
        expansion_matrices: list[tuple[tuple[float, ...], ...]] = []
        for degree in range(1, resolved_depth + 1):
            indices = tuple(
                index for index, word in enumerate(words) if len(word) == degree
            )
            degree_indices.append(indices)
            if indices:
                restricted = np.asarray(
                    [
                        [expansions[column].get(words[row], 0) for column in indices]
                        for row in indices
                    ],
                    dtype=float,
                )
                inverse = tuple(
                    map(
                        tuple,
                        np.linalg.solve(
                            restricted,
                            np.eye(restricted.shape[0], dtype=restricted.dtype),
                        ).tolist(),
                    )
                )
            else:
                inverse = ()
            inverses.append(inverse)
            all_words = tuple(product(range(resolved_dimension), repeat=degree))
            matrix = tuple(
                tuple(float(expansions[index].get(word, 0)) for index in indices)
                for word in all_words
            )
            expansion_matrices.append(matrix)
        self.dimension = resolved_dimension
        self.depth = resolved_depth
        self.words = words
        self.degrees = tuple(len(word) for word in words)
        self.children = tuple(children)
        self.word_expansions = tuple(
            tuple(sorted(expansion.items())) for expansion in expansions
        )
        self.degree_indices = tuple(degree_indices)
        self.conversion_inverses = tuple(inverses)
        self.expansion_matrices = tuple(expansion_matrices)

    @property
    def size(self) -> int:
        return len(self.words)

    def tensor_to_primitive(self, tensor_log: Sequence[ArrayLike], /) -> Array:
        """Convert tensor-log word coefficients to standard Lyndon coefficients."""
        levels, dimension = _signature_shape(tensor_log)
        if dimension != self.dimension or len(levels) != self.depth:
            raise ValueError(
                "Tensor log must match the primitive basis depth and dimension."
            )
        coefficients: list[Array] = []
        for degree, indices in enumerate(self.degree_indices, start=1):
            if not indices:
                continue
            selected = jnp.stack(
                tuple(
                    levels[degree - 1][(...,) + self.words[index]] for index in indices
                ),
                axis=-1,
            )
            inverse = jnp.asarray(
                self.conversion_inverses[degree - 1], dtype=selected.dtype
            )
            coefficients.append(ein.contract("ij,...j->...i", inverse, selected))
        return jnp.concatenate(coefficients, axis=-1)

    def primitive_to_tensor(self, coefficients: ArrayLike, /) -> tuple[Array, ...]:
        """Expand packed primitive coefficients into tensor word coefficients."""
        values = jnp.asarray(coefficients)
        if values.ndim < 1 or int(values.shape[-1]) != self.size:
            raise ValueError(f"coefficients must end in primitive axis {self.size}.")
        levels: list[Array] = []
        for degree, indices in enumerate(self.degree_indices, start=1):
            selected = values[..., jnp.asarray(indices)]
            matrix = jnp.asarray(self.expansion_matrices[degree - 1], dtype=values.dtype)
            flattened = ein.contract("wi,...i->...w", matrix, selected)
            levels.append(
                flattened.reshape(values.shape[:-1] + (self.dimension,) * degree)
            )
        return tuple(levels)


def _digest_array(digest: hashlib._Hash, value: ArrayLike, /) -> None:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _control_id(
    fine_times: Array,
    coarse_times: Array,
    levels: tuple[Array, ...],
    depth: int,
    joint_time: bool,
    source_id: str | None,
    /,
) -> str:
    digest = hashlib.sha256(b"phydrax-log-signature-control\0")
    _digest_array(digest, fine_times)
    _digest_array(digest, coarse_times)
    for level in levels:
        _digest_array(digest, level)
    digest.update(str(depth).encode("ascii"))
    digest.update(str(joint_time).encode("ascii"))
    digest.update(repr(source_id).encode("utf-8"))
    return digest.hexdigest()


class LogSignatureControl(AbstractRoughControl):
    """Piecewise-linear control stored as exact interval log signatures."""

    times: Array
    fine_times: Array
    signature_levels: tuple[Array, ...]
    log_coefficients: Array
    primitive_basis: PrimitiveBasis
    realization: FractionalGaussianRealization | None
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_dimension: int = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    joint_time: bool = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: Array,
        fine_times: Array,
        signature_levels: tuple[Array, ...],
        log_coefficients: Array,
        primitive_basis: PrimitiveBasis,
        sample_shape: tuple[int, ...],
        source_dimension: int,
        joint_time: bool,
        source_id: str | None,
        realization: FractionalGaussianRealization | None,
    ):
        self.times = times
        self.fine_times = fine_times
        self.signature_levels = signature_levels
        self.log_coefficients = log_coefficients
        self.primitive_basis = primitive_basis
        self.realization = realization
        self.sample_shape = sample_shape
        self.dimension = primitive_basis.dimension
        self.source_dimension = source_dimension
        self.num_steps = int(times.size) - 1
        self.depth = primitive_basis.depth
        self.joint_time = joint_time
        self.source_id = source_id
        self.control_id = _control_id(
            fine_times,
            times,
            signature_levels,
            self.depth,
            joint_time,
            source_id,
        )

    @property
    def levels(self) -> tuple[Array, ...]:
        return self.signature_levels

    @classmethod
    def from_values(
        cls,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        depth: int,
        coarse_indices: Sequence[int] | None = None,
        coarse_times: ArrayLike | None = None,
        sample_shape: Sequence[int] = (),
        joint_time: bool = False,
        source_id: str | None = None,
        realization: FractionalGaussianRealization | None = None,
    ) -> LogSignatureControl:
        """Lift fine path values and aggregate them exactly onto coarse knots."""
        resolved_depth = _validate_depth(depth)
        nodes = jnp.asarray(times, dtype=float)
        samples = tuple(int(size) for size in sample_shape)
        path_values = _inexact_array(values)
        if nodes.ndim != 1 or int(nodes.size) < 2:
            raise ValueError("times must contain at least two fine knots.")
        if bool(jnp.any(~jnp.isfinite(nodes))) or bool(jnp.any(jnp.diff(nodes) <= 0.0)):
            raise ValueError("times must be finite and strictly increasing.")
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        if path_values.ndim != len(samples) + 2:
            raise ValueError(
                "values must have shape sample_shape + (num_times, dimension)."
            )
        if path_values.shape[: len(samples)] != samples or path_values.shape[
            len(samples)
        ] != int(nodes.size):
            raise ValueError("values must align with sample_shape and fine times.")
        source_dimension = int(path_values.shape[-1])
        if source_dimension <= 0 or bool(jnp.any(~jnp.isfinite(path_values))):
            raise ValueError("values must have a non-empty finite driver axis.")
        if coarse_indices is not None and coarse_times is not None:
            raise ValueError("Specify coarse_indices or coarse_times, not both.")
        if coarse_times is not None:
            requested = np.asarray(coarse_times, dtype=float)
            fine_host = np.asarray(jax.device_get(nodes))
            positions = np.searchsorted(fine_host, requested)
            if (
                requested.ndim != 1
                or requested.size < 2
                or np.any(positions >= fine_host.size)
                or not np.allclose(
                    fine_host[np.minimum(positions, fine_host.size - 1)],
                    requested,
                    rtol=0.0,
                    atol=100.0 * np.finfo(float).eps,
                )
            ):
                raise ValueError("coarse_times must be fine knots.")
            indices = tuple(int(index) for index in positions)
        elif coarse_indices is None:
            indices = tuple(range(int(nodes.size)))
        else:
            indices = tuple(int(index) for index in coarse_indices)
        if (
            len(indices) < 2
            or indices[0] != 0
            or indices[-1] != int(nodes.size) - 1
            or any(right <= left for left, right in pairwise(indices))
        ):
            raise ValueError(
                "Coarse knots must increase from the first through the final fine knot."
            )
        resolved_source_id = None if source_id is None else str(source_id)
        if resolved_source_id == "":
            raise ValueError("source_id must be non-empty or None.")
        if realization is not None:
            if not isinstance(realization, FractionalGaussianRealization):
                raise TypeError(
                    "realization must be a FractionalGaussianRealization or None."
                )
            if realization.sample_shape != samples:
                raise ValueError("realization sample_shape must match values.")
            if realization.process.dimension != source_dimension:
                raise ValueError("realization dimension must match values.")
            if not jnp.array_equal(realization.grid, nodes) or not jnp.array_equal(
                realization.values, path_values
            ):
                raise ValueError("realization path must match times and values.")
            resolved_source_id = realization.realization_id
        if bool(joint_time):
            time_values = jnp.broadcast_to(
                nodes.reshape((1,) * len(samples) + (int(nodes.size), 1)),
                samples + (int(nodes.size), 1),
            )
            lifted_values = jnp.concatenate((time_values, path_values), axis=-1)
        else:
            lifted_values = path_values
        increments = jnp.diff(lifted_values, axis=len(samples))
        interval_signatures = tuple(
            piecewise_linear_signature(increments[..., left:right, :], resolved_depth)
            for left, right in pairwise(indices)
        )
        step_axis = len(samples)
        signature_levels = tuple(
            jnp.stack(
                tuple(interval[level] for interval in interval_signatures),
                axis=step_axis,
            )
            for level in range(resolved_depth)
        )
        basis = PrimitiveBasis(int(lifted_values.shape[-1]), resolved_depth)
        tensor_log = tensor_logarithm(signature_levels)
        log_coefficients = basis.tensor_to_primitive(tensor_log)
        return cls(
            times=nodes[jnp.asarray(indices)],
            fine_times=nodes,
            signature_levels=signature_levels,
            log_coefficients=log_coefficients,
            primitive_basis=basis,
            sample_shape=samples,
            source_dimension=source_dimension,
            joint_time=bool(joint_time),
            source_id=resolved_source_id,
            realization=realization,
        )

    @classmethod
    def from_fractional_gaussian(
        cls,
        realization: FractionalGaussianRealization,
        /,
        *,
        depth: int,
        coarse_indices: Sequence[int] | None = None,
        coarse_times: ArrayLike | None = None,
        joint_time: bool = False,
    ) -> LogSignatureControl:
        if not isinstance(realization, FractionalGaussianRealization):
            raise TypeError("realization must be a FractionalGaussianRealization.")
        return cls.from_values(
            realization.grid,
            realization.values,
            depth=depth,
            coarse_indices=coarse_indices,
            coarse_times=coarse_times,
            sample_shape=realization.sample_shape,
            joint_time=joint_time,
            realization=realization,
        )

    def signature(self, start_index: int, end_index: int, /) -> tuple[Array, ...]:
        start = int(start_index)
        end = int(end_index)
        if start < 0 or end > self.num_steps or end <= start:
            raise ValueError(
                "signature indices must satisfy 0 <= start < end <= num_steps."
            )
        sliced = tuple(
            level[..., start:end, *((slice(None),) * degree)]
            for degree, level in enumerate(self.signature_levels, start=1)
        )
        initial = tuple(
            jnp.zeros(
                self.sample_shape + (self.dimension,) * degree,
                dtype=sliced[degree - 1].dtype,
            )
            for degree in range(1, self.depth + 1)
        )

        def combine(carry, segment):
            return chen_multiply(carry, segment), None

        scan_values = tuple(
            jnp.moveaxis(level, len(self.sample_shape), 0) for level in sliced
        )
        return jax.lax.scan(combine, initial, scan_values)[0]


__all__ = [
    "LogSignatureControl",
    "PrimitiveBasis",
    "chen_multiply",
    "piecewise_linear_signature",
    "tensor_exponential",
    "tensor_logarithm",
]

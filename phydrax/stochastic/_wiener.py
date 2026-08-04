#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from math import isfinite, prod
from typing import Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule


LevyAreaKind: TypeAlias = Literal["brownian", "space_time", "space_time_time"]
WienerAlgorithm: TypeAlias = Literal["virtual_tree"]


def _validated_key(key: Key[Array, ""], /) -> Array:
    try:
        key_data = jr.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "WienerRealization root_key must be a scalar JAX PRNG key."
        ) from exc
    if key_data.shape != (2,):
        raise ValueError("WienerRealization root_key must be one scalar JAX PRNG key.")
    return key


def _digest_array(digest: hashlib._Hash, value: Array, /) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _fingerprint(
    root_key: Array,
    *,
    support: tuple[float, float],
    noise_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    path_indices: Array,
    path_signs: Array,
    tolerance: float,
    levy_area: LevyAreaKind,
    algorithm: WienerAlgorithm,
    noise_id: str | None,
) -> str:
    digest = hashlib.sha256()
    _digest_array(digest, jr.key_data(root_key))
    for value in (
        support,
        noise_shape,
        sample_shape,
        tolerance,
        levy_area,
        algorithm,
        noise_id,
    ):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    _digest_array(digest, path_indices)
    _digest_array(digest, path_signs)
    return digest.hexdigest()


def _coupling_fingerprint(
    root_key: Array,
    *,
    support: tuple[float, float],
    noise_shape: tuple[int, ...],
    tolerance: float,
    levy_area: LevyAreaKind,
    algorithm: WienerAlgorithm,
    noise_id: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"phydrax-wiener-coupling\0")
    _digest_array(digest, jr.key_data(root_key))
    for value in (support, noise_shape, tolerance, levy_area, algorithm, noise_id):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class WienerRealization(StrictModule):
    """A global, reproducible Wiener path or coupled batch of Wiener paths.

    The Brownian construction is defined on ``support``. Solves over subintervals query
    this same global path instead of constructing unrelated local Brownian trees.
    ``path_indices`` and ``path_signs`` encode independent or antithetic coupling while
    preserving prefix-stable path keys derived with :func:`jax.random.fold_in`.
    """

    root_key: Array
    path_indices: Array
    path_signs: Array
    support: tuple[float, float] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    levy_area: LevyAreaKind = eqx.field(static=True)
    algorithm: WienerAlgorithm = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_key: Key[Array, ""],
        noise_shape: Sequence[int],
        /,
        *,
        support: tuple[float, float],
        sample_shape: Sequence[int] = (),
        tolerance: float = 1e-3,
        levy_area: LevyAreaKind = "brownian",
        algorithm: WienerAlgorithm = "virtual_tree",
        noise_id: str | None = None,
        label: str | None = None,
        coupling_id: str | None = None,
        _path_indices: Array | None = None,
        _path_signs: Array | None = None,
    ):
        key = _validated_key(root_key)
        if len(support) != 2:
            raise ValueError("WienerRealization support must contain exactly two bounds.")
        start, end = (float(value) for value in support)
        if not isfinite(start) or not isfinite(end):
            raise ValueError("WienerRealization support bounds must be finite.")
        if not end > start:
            raise ValueError("WienerRealization support requires end > start.")

        noise = tuple(int(size) for size in noise_shape)
        if not noise or any(size <= 0 for size in noise):
            raise ValueError(
                "WienerRealization noise_shape must contain positive dimensions."
            )
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("WienerRealization sample dimensions must be positive.")

        tolerance_value = float(tolerance)
        if not isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("WienerRealization tolerance must be finite and positive.")
        if levy_area not in ("brownian", "space_time", "space_time_time"):
            raise ValueError(
                "levy_area must be 'brownian', 'space_time', or 'space_time_time'."
            )
        if algorithm != "virtual_tree":
            raise ValueError("The only supported Wiener algorithm is 'virtual_tree'.")
        if noise_id is not None and (not isinstance(noise_id, str) or not noise_id):
            raise ValueError("WienerRealization noise_id must be non-empty or None.")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("WienerRealization label must be non-empty or None.")
        if coupling_id is not None and (
            not isinstance(coupling_id, str) or not coupling_id
        ):
            raise ValueError("WienerRealization coupling_id must be non-empty or None.")

        count = prod(samples) if samples else 1
        expected_shape = samples
        if _path_indices is None:
            indices = jnp.arange(count, dtype=jnp.uint32).reshape(expected_shape)
        else:
            indices = jnp.asarray(_path_indices, dtype=jnp.uint32)
            if tuple(indices.shape) != expected_shape:
                raise ValueError(
                    "WienerRealization path indices must match sample_shape; "
                    f"got {indices.shape} and {expected_shape}."
                )
        if _path_signs is None:
            signs = jnp.ones(expected_shape, dtype=float)
        else:
            signs = jnp.asarray(_path_signs, dtype=float)
            if tuple(signs.shape) != expected_shape:
                raise ValueError(
                    "WienerRealization path signs must match sample_shape; "
                    f"got {signs.shape} and {expected_shape}."
                )
            if not bool(jnp.all(jnp.isin(signs, jnp.asarray([-1.0, 1.0])))):
                raise ValueError("WienerRealization path signs must be +1 or -1.")

        support_value = (start, end)
        resolved_coupling_id = coupling_id or _coupling_fingerprint(
            key,
            support=support_value,
            noise_shape=noise,
            tolerance=tolerance_value,
            levy_area=levy_area,
            algorithm=algorithm,
            noise_id=noise_id,
        )
        realization_id = _fingerprint(
            key,
            support=support_value,
            noise_shape=noise,
            sample_shape=samples,
            path_indices=indices,
            path_signs=signs,
            tolerance=tolerance_value,
            levy_area=levy_area,
            algorithm=algorithm,
            noise_id=noise_id,
        )

        self.root_key = key
        self.path_indices = indices
        self.path_signs = signs
        self.support = support_value
        self.noise_shape = noise
        self.sample_shape = samples
        self.tolerance = tolerance_value
        self.levy_area = levy_area
        self.algorithm = algorithm
        self.noise_id = noise_id
        self.label = label
        self.realization_id = realization_id
        self.coupling_id = resolved_coupling_id

    @classmethod
    def independent(
        cls,
        root_key: Key[Array, ""],
        noise_shape: Sequence[int],
        /,
        *,
        support: tuple[float, float],
        sample_shape: Sequence[int] = (),
        tolerance: float = 1e-3,
        levy_area: LevyAreaKind = "brownian",
        noise_id: str | None = None,
        label: str | None = None,
        coupling_id: str | None = None,
    ) -> WienerRealization:
        """Construct independent, prefix-stable paths."""
        return cls(
            root_key,
            noise_shape,
            support=support,
            sample_shape=sample_shape,
            tolerance=tolerance,
            levy_area=levy_area,
            noise_id=noise_id,
            label=label,
            coupling_id=coupling_id,
        )

    @classmethod
    def antithetic(
        cls,
        root_key: Key[Array, ""],
        noise_shape: Sequence[int],
        /,
        *,
        support: tuple[float, float],
        num_pairs: int,
        tolerance: float = 1e-3,
        levy_area: LevyAreaKind = "brownian",
        noise_id: str | None = None,
        label: str | None = None,
        coupling_id: str | None = None,
    ) -> WienerRealization:
        """Construct ``(+W, -W)`` pairs from prefix-stable Brownian keys."""
        pairs = int(num_pairs)
        if pairs <= 0:
            raise ValueError("num_pairs must be positive.")
        indices = jnp.repeat(jnp.arange(pairs, dtype=jnp.uint32), 2)
        signs = jnp.tile(jnp.asarray([1.0, -1.0]), pairs)
        return cls(
            root_key,
            noise_shape,
            support=support,
            sample_shape=(2 * pairs,),
            tolerance=tolerance,
            levy_area=levy_area,
            noise_id=noise_id,
            label=label,
            coupling_id=coupling_id,
            _path_indices=indices,
            _path_signs=signs,
        )

    @property
    def num_paths(self) -> int:
        """Number of paths represented by this realization."""
        return prod(self.sample_shape) if self.sample_shape else 1

    @property
    def path_keys(self) -> Array:
        """PRNG keys for each path, aligned with ``sample_shape``."""
        if not self.sample_shape:
            return jr.fold_in(self.root_key, self.path_indices)
        flat_indices = self.path_indices.reshape((-1,))
        keys = jax.vmap(lambda index: jr.fold_in(self.root_key, index))(flat_indices)
        return keys.reshape(self.sample_shape + tuple(self.root_key.shape))

    def increments(
        self,
        starts: Array,
        ends: Array,
        /,
        *,
        dtype: jnp.dtype | type = float,
    ) -> Array:
        """Evaluate reproducible Brownian increments on matching interval arrays.

        The result has shape ``sample_shape + interval_shape + noise_shape``.
        Repeated or overlapping intervals query the same global paths. This method
        intentionally supports Brownian increments only; higher-order Lévy-area
        objects are solver data, not additive driver increments.
        """
        if self.levy_area != "brownian":
            raise ValueError(
                "WienerRealization.increments requires levy_area='brownian'."
            )
        start = jnp.asarray(starts)
        end = jnp.asarray(ends, dtype=start.dtype)
        if start.shape != end.shape:
            raise ValueError("Wiener increment bounds must have matching shapes.")
        support_start, support_end = self.support
        if bool(
            jnp.any(~jnp.isfinite(start))
            | jnp.any(~jnp.isfinite(end))
            | jnp.any(start < support_start)
            | jnp.any(end > support_end)
            | jnp.any(end < start)
        ):
            raise ValueError(
                "Wiener increment intervals must lie in support with end >= start."
            )
        resolved_dtype = jnp.dtype(dtype)
        flat_starts = start.reshape((-1,))
        flat_ends = end.reshape((-1,))
        keys = self.path_keys.reshape((-1,) + tuple(self.root_key.shape))
        signs = self.path_signs.reshape((-1,))

        def evaluate_path(path_key: Array, sign: Array) -> Array:
            path = dfx.VirtualBrownianTree(
                t0=support_start,
                t1=support_end,
                tol=self.tolerance,
                shape=jax.ShapeDtypeStruct(self.noise_shape, resolved_dtype),
                key=path_key,
                levy_area=dfx.BrownianIncrement,
            )
            values = jax.vmap(path.evaluate)(flat_starts, flat_ends)
            return jnp.asarray(sign, dtype=resolved_dtype) * values

        values = jax.vmap(evaluate_path)(keys, signs)
        return values.reshape(self.sample_shape + start.shape + self.noise_shape)


__all__ = ["LevyAreaKind", "WienerAlgorithm", "WienerRealization"]

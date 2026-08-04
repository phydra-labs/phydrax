#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key
from scipy.stats.qmc import Halton, LatinHypercube, Sobol

from ._types import (
    design_capabilities,
    design_name,
    DesignLike,
    HaltonDesign,
    HammersleyDesign,
    IIDDesign,
    LatinHypercubeDesign,
    RandomizedQMCDesign,
    resolve_design,
    SobolDesign,
)


def seed_from_key(key: ArrayLike, /) -> int:
    """Derive a deterministic host seed from a concrete JAX key."""
    words = np.asarray(jr.key_data(key), dtype=np.uint32).reshape(-1)
    value = 1469598103934665603
    for word in words:
        value ^= int(word)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return int(value)


def _first_primes(count: int, /) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < int(count):
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _radical_inverse(indices: np.ndarray, base: int, /) -> np.ndarray:
    values = np.zeros(indices.shape, dtype=float)
    remaining = np.asarray(indices, dtype=np.int64).copy()
    factor = 1.0 / float(base)
    while np.any(remaining > 0):
        values += factor * (remaining % base)
        remaining //= base
        factor /= float(base)
    return values


def _qmc_engine(
    design: HaltonDesign | SobolDesign | RandomizedQMCDesign,
    dimension: int,
    seed: int | np.random.Generator,
):
    if isinstance(design, RandomizedQMCDesign):
        sequence = design.sequence
        scrambled = design.scrambled
    else:
        sequence = "halton" if isinstance(design, HaltonDesign) else "sobol"
        scrambled = design.scrambled
    if sequence == "halton":
        return Halton(dimension, scramble=scrambled, seed=seed)
    return Sobol(dimension, scramble=scrambled, seed=seed)


def host_design(
    design: DesignLike,
    *,
    count: int,
    dimension: int,
    seed: int | np.random.Generator = 0,
    start: int = 0,
) -> np.ndarray:
    """Materialize a reference design in the unit cube on the host."""
    resolved = resolve_design(design)
    count_ = int(count)
    dimension_ = int(dimension)
    start_ = int(start)
    if count_ < 0:
        raise ValueError("count must be non-negative.")
    if dimension_ < 1:
        raise ValueError("dimension must be positive.")
    if start_ < 0:
        raise ValueError("start must be non-negative.")
    capabilities = design_capabilities(resolved)
    if start_ and not capabilities.random_access:
        raise ValueError(f"{design_name(resolved)} does not support a start index.")
    if count_ == 0:
        return np.empty((0, dimension_), dtype=float)
    if isinstance(resolved, IIDDesign):
        generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        return np.asarray(generator.random((count_, dimension_)), dtype=float)
    if isinstance(resolved, HammersleyDesign):
        indices = np.arange(1, count_ + 1, dtype=np.int64)
        columns = [(indices - 0.5) / float(count_)]
        columns.extend(
            _radical_inverse(indices, prime)
            for prime in _first_primes(max(dimension_ - 1, 0))
        )
        return np.stack(columns, axis=1)
    if isinstance(resolved, LatinHypercubeDesign):
        engine = LatinHypercube(dimension_, seed=seed)
    else:
        engine = _qmc_engine(resolved, dimension_, seed)
        if start_:
            engine.fast_forward(start_)
    return np.asarray(engine.random(count_), dtype=float).reshape((count_, dimension_))


def host_design_factory(
    design: DesignLike,
    *,
    dimension: int,
    seed: int | np.random.Generator,
) -> Callable[[int], np.ndarray]:
    """Return a stateful host design materializer for rejection samplers."""
    resolved = resolve_design(design)
    dimension_ = int(dimension)
    if dimension_ < 1:
        raise ValueError("dimension must be positive.")
    if isinstance(resolved, IIDDesign):
        generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        return lambda count: np.asarray(
            generator.random((int(count), dimension_)), dtype=float
        )
    if isinstance(resolved, HammersleyDesign):
        return lambda count: host_design(
            resolved,
            count=int(count),
            dimension=dimension_,
            seed=seed,
        )
    if isinstance(resolved, LatinHypercubeDesign):
        engine = LatinHypercube(dimension_, seed=seed)
    else:
        engine = _qmc_engine(resolved, dimension_, seed)
    return lambda count: np.asarray(engine.random(int(count)), dtype=float).reshape(
        (int(count), dimension_)
    )


def materialize_design(
    design: DesignLike,
    *,
    count: int,
    dimension: int,
    key: Key[Array, ""] | None,
    start: int = 0,
) -> Array:
    """Materialize unit-cube points with JIT-safe host generation when required."""
    resolved = resolve_design(design)
    count_ = int(count)
    dimension_ = int(dimension)
    start_ = int(start)
    if count_ < 0:
        raise ValueError("count must be non-negative.")
    if dimension_ < 1:
        raise ValueError("dimension must be positive.")
    if start_ < 0:
        raise ValueError("start must be non-negative.")
    capabilities = design_capabilities(resolved)
    if start_ and not capabilities.random_access:
        raise ValueError(f"{design_name(resolved)} does not support a start index.")
    if count_ == 0:
        return jnp.zeros((0, dimension_), dtype=float)
    if isinstance(resolved, IIDDesign):
        if key is None:
            raise ValueError("uniform randomized designs require a key.")
        return jr.uniform(key, (count_, dimension_), dtype=float)
    if capabilities.randomized and key is None:
        raise ValueError(f"{design_name(resolved)} requires a key.")
    if key is None:
        return jnp.asarray(
            host_design(
                resolved,
                count=count_,
                dimension=dimension_,
                seed=0,
                start=start_,
            ),
            dtype=float,
        )

    prototype = jnp.zeros((count_, dimension_), dtype=float)
    result_spec = jax.ShapeDtypeStruct(prototype.shape, prototype.dtype)

    def materialize_host(key_value):
        return np.asarray(
            host_design(
                resolved,
                count=count_,
                dimension=dimension_,
                seed=seed_from_key(key_value),
                start=start_,
            ),
            dtype=np.dtype(prototype.dtype),
        )

    return jax.pure_callback(materialize_host, result_spec, key)


def unit_design(
    name: str,
    *,
    count: int,
    dimension: int,
    key: Key[Array, ""] | None,
) -> Array:
    """Compatibility wrapper for the former numerical unit-design API."""
    resolved = resolve_design(name)
    capabilities = design_capabilities(resolved)
    if not capabilities.randomized and key is not None:
        raise ValueError(f"deterministic {design_name(resolved)} does not accept a key.")
    return materialize_design(
        resolved,
        count=count,
        dimension=dimension,
        key=key,
    )


def get_sampler_host(
    design: DesignLike,
    *,
    dim: int,
    seed: int | np.random.Generator,
) -> Callable[[int], np.ndarray]:
    """Compatibility host sampler for points in a unit cube."""
    return host_design_factory(design, dimension=int(dim), seed=seed)


def get_sampler(design: DesignLike):
    """Return the historical `(count, dimension, key)` sampler callable."""
    resolved = resolve_design(design)

    def sample(n: int, dim: int, key: Key[Array, ""]):
        return materialize_design(
            resolved,
            count=int(n),
            dimension=int(dim),
            key=key,
        )

    return sample


__all__ = [
    "get_sampler",
    "get_sampler_host",
    "host_design",
    "host_design_factory",
    "materialize_design",
    "seed_from_key",
    "unit_design",
]

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


_SUPPORTED_DESIGNS = (
    "uniform",
    "latin_hypercube",
    "halton",
    "halton_scrambled",
    "hammersley",
    "sobol",
    "sobol_scrambled",
)


def normalize_design_name(name: str, /) -> str:
    name_ = str(name).lower()
    if name_ not in _SUPPORTED_DESIGNS:
        raise ValueError(f"design must be one of {_SUPPORTED_DESIGNS}; got {name!r}.")
    return name_


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


def host_design(
    name: str,
    *,
    count: int,
    dimension: int,
    seed: int | np.random.Generator = 0,
) -> np.ndarray:
    """Materialize a deterministic or randomized design in the unit cube."""
    name_ = normalize_design_name(name)
    count_ = int(count)
    dimension_ = int(dimension)
    if count_ < 1:
        raise ValueError("count must be positive.")
    if dimension_ < 1:
        raise ValueError("dimension must be positive.")
    if name_ == "uniform":
        generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        return np.asarray(generator.random((count_, dimension_)), dtype=float)
    if name_ == "hammersley":
        indices = np.arange(1, count_ + 1, dtype=np.int64)
        columns = [(indices - 0.5) / float(count_)]
        columns.extend(
            _radical_inverse(indices, prime)
            for prime in _first_primes(max(dimension_ - 1, 0))
        )
        return np.stack(columns, axis=1)
    if name_ == "latin_hypercube":
        engine = LatinHypercube(dimension_, seed=seed)
    elif name_ == "halton":
        engine = Halton(dimension_, scramble=False, seed=seed)
    elif name_ == "halton_scrambled":
        engine = Halton(dimension_, scramble=True, seed=seed)
    elif name_ == "sobol":
        engine = Sobol(dimension_, scramble=False, seed=seed)
    else:
        engine = Sobol(dimension_, scramble=True, seed=seed)
    return np.asarray(engine.random(count_), dtype=float).reshape((count_, dimension_))


def host_design_factory(
    name: str,
    *,
    dimension: int,
    seed: int | np.random.Generator,
) -> Callable[[int], np.ndarray]:
    """Return a stateful host design factory used by domain sampling adapters."""
    name_ = normalize_design_name(name)
    dimension_ = int(dimension)
    if name_ == "uniform":
        generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        return lambda count: np.asarray(
            generator.random((int(count), dimension_)), dtype=float
        )
    if name_ == "hammersley":
        return lambda count: host_design(
            name_, count=int(count), dimension=dimension_, seed=seed
        )
    if name_ == "latin_hypercube":
        engine = LatinHypercube(dimension_, seed=seed)
    elif name_ == "halton":
        engine = Halton(dimension_, scramble=False, seed=seed)
    elif name_ == "halton_scrambled":
        engine = Halton(dimension_, scramble=True, seed=seed)
    elif name_ == "sobol":
        engine = Sobol(dimension_, scramble=False, seed=seed)
    else:
        engine = Sobol(dimension_, scramble=True, seed=seed)
    return lambda count: np.asarray(engine.random(int(count)), dtype=float).reshape(
        (int(count), dimension_)
    )


def unit_design(
    name: str,
    *,
    count: int,
    dimension: int,
    key: Key[Array, ""] | None,
) -> Array:
    """Materialize unit-cube points with JIT-safe host generation when required."""
    name_ = normalize_design_name(name)
    count_ = int(count)
    dimension_ = int(dimension)
    if name_ == "uniform":
        if key is None:
            raise ValueError("uniform randomized designs require a key.")
        return jr.uniform(key, (count_, dimension_), dtype=float)
    randomized = name_.endswith("_scrambled") or name_ == "latin_hypercube"
    if randomized and key is None:
        raise ValueError(f"{name_} requires a key.")
    if not randomized and key is not None:
        raise ValueError(f"deterministic {name_} does not accept a key.")
    if key is None:
        return jnp.asarray(
            host_design(name_, count=count_, dimension=dimension_, seed=0), dtype=float
        )
    prototype = jnp.zeros((count_, dimension_), dtype=float)
    result_spec = jax.ShapeDtypeStruct(prototype.shape, prototype.dtype)

    def materialize_host(key_value):
        return np.asarray(
            host_design(
                name_,
                count=count_,
                dimension=dimension_,
                seed=seed_from_key(key_value),
            ),
            dtype=np.dtype(prototype.dtype),
        )

    return jax.pure_callback(materialize_host, result_spec, key)


__all__ = [
    "host_design",
    "host_design_factory",
    "normalize_design_name",
    "seed_from_key",
    "unit_design",
]

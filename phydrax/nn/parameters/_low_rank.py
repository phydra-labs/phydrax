#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key, PyTree
from opt_einsum import contract

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ._selection import ParameterSubspace


LowRankScaling: TypeAlias = Literal["rank", "sqrt_rank"]


@dataclass(frozen=True, slots=True)
class LowRankSpec:
    """Construction policy for one low-rank affine weight update."""

    rank: int
    alpha: float | None = None
    scaling: LowRankScaling = "rank"
    stddev: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, Integral):
            raise TypeError("Low-rank adaptation rank must be an integer.")
        rank = int(self.rank)
        if rank <= 0:
            raise ValueError("Low-rank adaptation rank must be positive.")
        alpha = float(rank if self.alpha is None else self.alpha)
        scaling = str(self.scaling)
        stddev = float(self.stddev)
        if not isfinite(alpha) or alpha <= 0.0:
            raise ValueError("Low-rank adaptation alpha must be finite and positive.")
        if scaling not in ("rank", "sqrt_rank"):
            raise ValueError("Low-rank scaling must be 'rank' or 'sqrt_rank'.")
        if not isfinite(stddev) or stddev <= 0.0:
            raise ValueError(
                "Low-rank initialization stddev must be finite and positive."
            )
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "scaling", scaling)
        object.__setattr__(self, "stddev", stddev)


class LowRankUpdate(StrictModule):
    """Dense base weight plus one trainable low-rank update.

    For a base weight with shape ``(out, in)``, ``left`` has shape
    ``(out, rank)`` and ``right`` has shape ``(rank, in)``. Evaluation keeps the
    update factorized; :meth:`materialize` is reserved for diagnostics, export,
    and deployment merging.
    """

    base: Array
    left: Array
    right: Array
    alpha: float = eqx.field(static=True)
    scaling: LowRankScaling = eqx.field(static=True)

    def __init__(
        self,
        base: Array,
        /,
        *,
        rank: int,
        alpha: float | None = None,
        scaling: LowRankScaling = "rank",
        stddev: float = 0.01,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        value = jnp.asarray(base)
        if not eqx.is_inexact_array(value) or jnp.iscomplexobj(value):
            raise TypeError("Low-rank base weights must be real inexact JAX arrays.")
        if value.ndim != 2:
            raise ValueError("Low-rank base weights must be rank-two arrays.")
        spec = LowRankSpec(
            rank=rank,
            alpha=alpha,
            scaling=scaling,
            stddev=stddev,
        )
        output_size, input_size = (int(size) for size in value.shape)
        if spec.rank > min(output_size, input_size):
            raise ValueError(
                "Low-rank adaptation rank must not exceed either weight dimension."
            )
        self.base = value
        self.left = jnp.zeros((output_size, spec.rank), dtype=value.dtype)
        self.right = (
            jr.normal(key, (spec.rank, input_size), dtype=value.dtype) * spec.stddev
        )
        assert spec.alpha is not None
        self.alpha = spec.alpha
        self.scaling = spec.scaling

    @classmethod
    def from_factors(
        cls,
        base: Array,
        left: Array,
        right: Array,
        /,
        *,
        alpha: float,
        scaling: LowRankScaling = "rank",
    ) -> LowRankUpdate:
        """Construct an update from validated persisted factors without randomness."""
        base_ = jnp.asarray(base)
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if not eqx.is_inexact_array(base_) or jnp.iscomplexobj(base_):
            raise TypeError("Low-rank base weights must be real inexact JAX arrays.")
        if base_.ndim != 2 or left_.ndim != 2 or right_.ndim != 2:
            raise ValueError("Low-rank bases and factors must be rank-two arrays.")
        if left_.shape[0] != base_.shape[0] or right_.shape[1] != base_.shape[1]:
            raise ValueError("Low-rank factor outer dimensions must match the base.")
        if left_.shape[1] != right_.shape[0]:
            raise ValueError("Low-rank factors must share one positive rank dimension.")
        rank = int(left_.shape[1])
        if rank <= 0 or rank > min(base_.shape):
            raise ValueError("Low-rank factor rank is invalid for the base weight.")
        if left_.dtype != base_.dtype or right_.dtype != base_.dtype:
            raise TypeError("Low-rank factors must have the exact base dtype.")
        alpha_ = float(alpha)
        scaling_ = str(scaling)
        if not isfinite(alpha_) or alpha_ <= 0.0:
            raise ValueError("Low-rank adaptation alpha must be finite and positive.")
        if scaling_ not in ("rank", "sqrt_rank"):
            raise ValueError("Low-rank scaling must be 'rank' or 'sqrt_rank'.")
        instance = object.__new__(cls)
        object.__setattr__(instance, "base", base_)
        object.__setattr__(instance, "left", left_)
        object.__setattr__(instance, "right", right_)
        object.__setattr__(instance, "alpha", alpha_)
        object.__setattr__(instance, "scaling", scaling_)
        object.__setattr__(instance, "_strict_initialized", True)
        return instance

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.base.shape[0]), int(self.base.shape[1])

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def scale(self) -> float:
        denominator = self.rank if self.scaling == "rank" else sqrt(self.rank)
        return self.alpha / denominator

    @property
    def base_parameter_count(self) -> int:
        return int(self.base.size)

    @property
    def adapter_parameter_count(self) -> int:
        return int(self.left.size + self.right.size)

    def apply(self, value: Array, /) -> Array:
        """Apply the effective weight without materializing its dense update."""
        argument = jnp.asarray(value)
        base_output = contract(
            "oi,...i->...o", jax.lax.stop_gradient(self.base), argument
        )
        latent = contract("ri,...i->...r", self.right, argument)
        update = contract("or,...r->...o", self.left, latent)
        return (base_output + self.scale * update).astype(base_output.dtype)

    def materialize(self) -> Array:
        """Return the dense effective weight for diagnostics, export, or deployment."""
        update = contract("or,ri->oi", self.left, self.right)
        return (self.base + self.scale * update).astype(self.base.dtype)


@dataclass(frozen=True, slots=True)
class LowRankAdaptationSite:
    """Static accounting for one adapted affine weight path."""

    path: str
    shape: tuple[int, int]
    dtype: str
    rank: int
    alpha: float
    scaling: LowRankScaling
    scale: float
    base_parameter_count: int
    adapter_parameter_count: int


@dataclass(frozen=True, slots=True)
class LowRankAdaptationReport:
    """Deterministic site and parameter accounting for one adaptation operation."""

    sites: tuple[LowRankAdaptationSite, ...]
    base_parameter_count: int
    adapter_parameter_count: int

    @property
    def parameter_ratio(self) -> float:
        return self.adapter_parameter_count / self.base_parameter_count


@dataclass(frozen=True, slots=True)
class _LinearSite:
    path: str
    layer: Any


def _linear_sites(tree: PyTree[Any], /) -> tuple[_LinearSite, ...]:
    from ..layers._linear import Linear

    sites: list[_LinearSite] = []
    aliases: dict[int, str] = {}
    leaves = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=lambda value: isinstance(value, Linear),
    )[0]
    for path, leaf in leaves:
        if not isinstance(leaf, Linear):
            continue
        prefix = jax.tree_util.keystr(path)
        weight_path = f"{prefix}.weight" if prefix else ".weight"
        previous = aliases.get(id(leaf))
        if previous is not None:
            raise ValueError(
                "Low-rank adaptation does not support aliased Linear modules at "
                f"{previous!r} and {weight_path!r}."
            )
        aliases[id(leaf)] = weight_path
        sites.append(_LinearSite(weight_path, leaf))
    return tuple(sites)


def _is_adaptable(site: _LinearSite, /) -> bool:
    layer = site.layer
    weight = layer.weight
    return bool(
        not isinstance(weight, LowRankUpdate)
        and layer.weight_transform is None
        and eqx.is_inexact_array(weight)
        and not jnp.iscomplexobj(weight)
        and weight.ndim == 2
    )


def _validate_adaptable(site: _LinearSite, /) -> Array:
    layer = site.layer
    if isinstance(layer.weight, LowRankUpdate):
        raise TypeError(f"Linear weight {site.path!r} is already low-rank adapted.")
    if layer.weight_transform is not None:
        raise ValueError(f"Linear weight {site.path!r} uses a weight transform.")
    weight = jnp.asarray(layer.weight)
    if not eqx.is_inexact_array(weight) or jnp.iscomplexobj(weight) or weight.ndim != 2:
        raise TypeError(
            f"Linear weight {site.path!r} must be a real inexact rank-two array."
        )
    return weight


def low_rank_sites(tree: PyTree[Any], /) -> tuple[str, ...]:
    """Return deterministic paths of native Linear weights eligible for adaptation."""
    return tuple(site.path for site in _linear_sites(tree) if _is_adaptable(site))


def adapt_low_rank(
    tree: PyTree[Any],
    specs: Mapping[str, LowRankSpec],
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> tuple[PyTree[Any], LowRankAdaptationReport]:
    """Adapt exact native Linear weight paths and return immutable accounting."""
    from ..layers._linear import Linear

    requested = {str(path): spec for path, spec in specs.items()}
    if not requested:
        raise ValueError("Low-rank adaptation requires at least one weight path.")
    if any(not isinstance(spec, LowRankSpec) for spec in requested.values()):
        raise TypeError("Every low-rank adaptation value must be a LowRankSpec.")
    sites = _linear_sites(tree)
    sites_by_path = {site.path: site for site in sites}
    missing = tuple(path for path in requested if path not in sites_by_path)
    if missing:
        raise ValueError(f"Unknown native Linear weight paths: {missing!r}.")

    ordered = tuple(site for site in sites if site.path in requested)
    weights: dict[str, Array] = {}
    for site in ordered:
        weight = _validate_adaptable(site)
        spec = requested[site.path]
        if spec.rank > min(weight.shape):
            raise ValueError(
                f"Low-rank rank {spec.rank} exceeds weight shape {weight.shape} "
                f"at {site.path!r}."
            )
        weights[site.path] = weight

    keys = jr.split(key, len(ordered))
    replacements: dict[str, LowRankUpdate] = {}
    records: list[LowRankAdaptationSite] = []
    for site, site_key in zip(ordered, keys, strict=True):
        spec = requested[site.path]
        weight = weights[site.path]
        update = LowRankUpdate(
            weight,
            rank=spec.rank,
            alpha=spec.alpha,
            scaling=spec.scaling,
            stddev=spec.stddev,
            key=site_key,
        )
        replacements[site.path] = update
        records.append(
            LowRankAdaptationSite(
                path=site.path,
                shape=update.shape,
                dtype=jnp.dtype(update.dtype).str,
                rank=update.rank,
                alpha=update.alpha,
                scaling=update.scaling,
                scale=update.scale,
                base_parameter_count=update.base_parameter_count,
                adapter_parameter_count=update.adapter_parameter_count,
            )
        )

    def replace(path, value):
        if not isinstance(value, Linear):
            return value
        prefix = jax.tree_util.keystr(path)
        weight_path = f"{prefix}.weight" if prefix else ".weight"
        update = replacements.get(weight_path)
        if update is None:
            return value
        return eqx.tree_at(lambda layer: layer.weight, value, update)

    adapted = jax.tree_util.tree_map_with_path(
        replace,
        tree,
        is_leaf=lambda value: isinstance(value, Linear),
    )
    records_ = tuple(records)
    return adapted, LowRankAdaptationReport(
        sites=records_,
        base_parameter_count=sum(site.base_parameter_count for site in records_),
        adapter_parameter_count=sum(site.adapter_parameter_count for site in records_),
    )


def _low_rank_nodes(tree: PyTree[Any], /) -> tuple[tuple[str, LowRankUpdate], ...]:
    nodes: list[tuple[str, LowRankUpdate]] = []
    leaves = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=lambda value: isinstance(value, LowRankUpdate),
    )[0]
    for path, leaf in leaves:
        if isinstance(leaf, LowRankUpdate):
            nodes.append((jax.tree_util.keystr(path), leaf))
    return tuple(nodes)


def _low_rank_factor_paths(tree: PyTree[Any], /) -> tuple[str, ...]:
    paths: list[str] = []
    for prefix, _ in _low_rank_nodes(tree):
        paths.extend(
            (
                f"{prefix}.left" if prefix else ".left",
                f"{prefix}.right" if prefix else ".right",
            )
        )
    return tuple(paths)


def contains_low_rank_updates(tree: PyTree[Any], /) -> bool:
    """Return whether a PyTree contains at least one low-rank update."""
    return bool(_low_rank_nodes(tree))


def low_rank_parameter_subspace(tree: PyTree[Any], /) -> ParameterSubspace:
    """Select exactly the trainable factors of every low-rank update in a PyTree."""
    paths = _low_rank_factor_paths(tree)
    if not paths:
        raise ValueError("The supplied PyTree contains no low-rank updates.")
    return ParameterSubspace.from_leaf_paths(tree, paths)


def validate_low_rank_subspace(
    tree: PyTree[Any],
    subspace: ParameterSubspace,
    /,
) -> None:
    """Require all and only low-rank factor leaves in a training subspace."""
    expected = _low_rank_factor_paths(tree)
    if not expected:
        return
    if subspace.leaf_paths != expected:
        raise ValueError(
            "Low-rank training requires a ParameterSubspace selecting exactly every "
            "left and right adapter factor."
        )


def merge_low_rank(tree: PyTree[Any], /) -> PyTree[Any]:
    """Return a dense model with every low-rank update merged into its base."""
    return jax.tree_util.tree_map(
        lambda value: value.materialize() if isinstance(value, LowRankUpdate) else value,
        tree,
        is_leaf=lambda value: isinstance(value, LowRankUpdate),
    )


def _strip_low_rank(tree: PyTree[Any], /) -> PyTree[Any]:
    return jax.tree_util.tree_map(
        lambda value: value.base if isinstance(value, LowRankUpdate) else value,
        tree,
        is_leaf=lambda value: isinstance(value, LowRankUpdate),
    )


__all__ = [
    "LowRankAdaptationReport",
    "LowRankAdaptationSite",
    "LowRankSpec",
    "LowRankScaling",
    "LowRankUpdate",
    "adapt_low_rank",
    "contains_low_rank_updates",
    "low_rank_parameter_subspace",
    "low_rank_sites",
    "merge_low_rank",
    "validate_low_rank_subspace",
]

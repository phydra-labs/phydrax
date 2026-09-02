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

from phydrax.ein import contract

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


LowRankSiteHandler: TypeAlias = Literal["identity", "symmetric", "skew"]


@dataclass(frozen=True, slots=True)
class LowRankAdaptationPlan:
    """Host-prepared low-rank sites, handlers, and semantic alias groups."""

    specs: tuple[tuple[str, LowRankSpec], ...]
    site_handlers: tuple[tuple[str, LowRankSiteHandler], ...]
    alias_groups: tuple[tuple[str, ...], ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(path for path, _ in self.specs)


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
        if not eqx.is_inexact_array(value):
            raise TypeError("Low-rank base weights must be inexact JAX arrays.")
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
        if jnp.iscomplexobj(value):
            real_dtype = value.real.dtype
            normal = jr.normal(key, (2, spec.rank, input_size), dtype=real_dtype) * (
                spec.stddev / sqrt(2.0)
            )
            self.right = (normal[0] + 1j * normal[1]).astype(value.dtype)
        else:
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
        if not eqx.is_inexact_array(base_):
            raise TypeError("Low-rank base weights must be inexact JAX arrays.")
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

    def apply_linear_transform(
        self,
        value: Array,
        /,
        *,
        mode: Literal["symmetric", "skew"],
    ) -> Array:
        """Apply a symmetric/skew raw-space update without dense materialization."""
        if self.shape[0] != self.shape[1] or jnp.iscomplexobj(self.base):
            raise ValueError(
                "Symmetric/skew low-rank updates require square real weights."
            )
        argument = jnp.asarray(value)
        sign = 1.0 if mode == "symmetric" else -1.0
        base_effective = 0.5 * (self.base + sign * self.base.T)
        base_output = contract(
            "oi,...i->...o",
            jax.lax.stop_gradient(base_effective),
            argument,
        )
        forward = contract(
            "or,ri,...i->...o",
            self.left,
            self.right,
            argument,
        )
        transpose = contract(
            "ir,ro,...i->...o",
            self.left,
            self.right,
            argument,
        )
        return base_output + 0.5 * self.scale * (forward + sign * transpose)

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
    handler: LowRankSiteHandler = "identity"
    raw_update_space: bool = True
    materializes_effective_weight: bool = False
    complex_representation: Literal["real", "native_complex"] = "real"
    alias_group: tuple[str, ...] = ()


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
    module_id: int


def _linear_sites(tree: PyTree[Any], /) -> tuple[_LinearSite, ...]:
    from ..layers._linear import Linear

    sites: list[_LinearSite] = []
    leaves = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=lambda value: isinstance(value, Linear),
    )[0]
    for path, leaf in leaves:
        if not isinstance(leaf, Linear):
            continue
        prefix = jax.tree_util.keystr(path)
        weight_path = f"{prefix}.weight" if prefix else ".weight"
        sites.append(_LinearSite(weight_path, leaf, id(leaf)))
    return tuple(sites)


def _is_adaptable(site: _LinearSite, /) -> bool:
    layer = site.layer
    weight = layer.weight
    if isinstance(weight, LowRankUpdate) or not eqx.is_inexact_array(weight):
        return False
    if jnp.asarray(weight).ndim != 2:
        return False
    if layer.weight_transform is None:
        return True
    from ._transforms import SkewSymmetricTransform, SymmetricTransform

    return isinstance(
        layer.weight_transform, (SymmetricTransform, SkewSymmetricTransform)
    )


def _validate_adaptable(site: _LinearSite, /) -> Array:
    layer = site.layer
    if isinstance(layer.weight, LowRankUpdate):
        raise TypeError(f"Linear weight {site.path!r} is already low-rank adapted.")
    weight = jnp.asarray(layer.weight)
    if not eqx.is_inexact_array(weight) or weight.ndim != 2:
        raise TypeError(f"Linear weight {site.path!r} must be an inexact rank-two array.")
    if layer.weight_transform is not None:
        from ._transforms import SkewSymmetricTransform, SymmetricTransform

        if not isinstance(
            layer.weight_transform,
            (SymmetricTransform, SkewSymmetricTransform),
        ):
            raise ValueError(
                f"Linear weight {site.path!r} uses an unsupported weight transform."
            )
        if jnp.iscomplexobj(weight) or weight.shape[0] != weight.shape[1]:
            raise ValueError(
                "Symmetric/skew low-rank sites require square real raw weights."
            )
    return weight


def low_rank_sites(tree: PyTree[Any], /) -> tuple[str, ...]:
    """Return deterministic paths, rejecting undeclared incidental aliases."""
    sites = _linear_sites(tree)
    identities = [site.module_id for site in sites]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "Low-rank sites contain aliased modules; declare alias_groups during prepare."
        )
    return tuple(site.path for site in sites if _is_adaptable(site))


def prepare_low_rank_adaptation(
    tree: PyTree[Any],
    specs: Mapping[str, LowRankSpec],
    /,
    *,
    alias_groups: tuple[tuple[str, ...], ...] = (),
) -> LowRankAdaptationPlan:
    """Validate low-rank surgery and freeze site/alias semantics on the host."""
    from ._transforms import SkewSymmetricTransform, SymmetricTransform

    requested = {str(path): spec for path, spec in specs.items()}
    if not requested:
        raise ValueError("Low-rank adaptation requires at least one weight path.")
    if any(not isinstance(spec, LowRankSpec) for spec in requested.values()):
        raise TypeError("Every low-rank adaptation value must be a LowRankSpec.")
    sites_by_path = {site.path: site for site in _linear_sites(tree)}
    missing = tuple(path for path in requested if path not in sites_by_path)
    if missing:
        raise ValueError(f"Unknown native Linear weight paths: {missing!r}.")
    handlers: list[tuple[str, LowRankSiteHandler]] = []
    weights_by_path: dict[str, Array] = {}
    for path, spec in requested.items():
        site = sites_by_path[path]
        weight = _validate_adaptable(site)
        if spec.rank > min(weight.shape):
            raise ValueError(
                f"Low-rank rank {spec.rank} exceeds weight shape {weight.shape} "
                f"at {path!r}."
            )
        transform = site.layer.weight_transform
        weights_by_path[path] = weight
        handler: LowRankSiteHandler
        if transform is None:
            handler = "identity"
        elif isinstance(transform, SymmetricTransform):
            handler = "symmetric"
        elif isinstance(transform, SkewSymmetricTransform):
            handler = "skew"
        else:
            raise ValueError(f"Unsupported low-rank site transform at {path!r}.")
        handlers.append((path, handler))
    handler_map = dict(handlers)
    normalized_groups: list[tuple[str, ...]] = []
    used: set[str] = set()
    duplicate_paths = {
        site.path
        for site in sites_by_path.values()
        if sum(other.module_id == site.module_id for other in sites_by_path.values()) > 1
    }
    declared_alias_paths = {path for declared in alias_groups for path in declared}
    if duplicate_paths != declared_alias_paths:
        raise ValueError(
            "Incidental module aliases must be declared exactly in alias_groups."
        )
    for group in alias_groups:
        paths = tuple(str(path) for path in group)
        if len(paths) < 2 or len(set(paths)) != len(paths):
            raise ValueError("Alias groups require at least two distinct paths.")
        if any(path not in requested for path in paths):
            raise ValueError("Every alias path must be an adapted site.")
        if any(path in used for path in paths):
            raise ValueError("Low-rank alias groups must be disjoint.")
        first = requested[paths[0]]
        if any(requested[path] != first for path in paths[1:]):
            raise ValueError("Aliased low-rank sites must use identical specs.")
        if any(handler_map[path] != handler_map[paths[0]] for path in paths[1:]):
            raise ValueError("Aliased low-rank sites must use identical handlers.")
        canonical_weight = weights_by_path[paths[0]]
        if any(
            not bool(jnp.array_equal(weights_by_path[path], canonical_weight))
            for path in paths[1:]
        ):
            raise ValueError("Aliased low-rank sites must have identical base content.")
        used.update(paths)
        normalized_groups.append(paths)
    return LowRankAdaptationPlan(
        specs=tuple(requested.items()),
        site_handlers=tuple(handlers),
        alias_groups=tuple(normalized_groups),
    )


def adapt_low_rank(
    tree: PyTree[Any],
    plan: LowRankAdaptationPlan | Mapping[str, LowRankSpec],
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> tuple[PyTree[Any], LowRankAdaptationReport]:
    """Apply a prepared adaptation plan and return immutable accounting."""
    from ..layers._linear import Linear

    prepared = (
        prepare_low_rank_adaptation(tree, plan) if isinstance(plan, Mapping) else plan
    )
    if not isinstance(prepared, LowRankAdaptationPlan):
        raise TypeError("plan must be a LowRankAdaptationPlan.")
    requested = dict(prepared.specs)
    handlers = dict(prepared.site_handlers)
    sites = _linear_sites(tree)
    sites_by_path = {site.path: site for site in sites}
    missing = tuple(path for path in requested if path not in sites_by_path)
    if missing:
        raise ValueError(
            f"Prepared low-rank plan does not match this model tree: missing={missing!r}."
        )
    ordered = tuple(site for site in sites if site.path in requested)
    weights = {site.path: _validate_adaptable(site) for site in ordered}

    alias_for: dict[str, tuple[str, ...]] = {}
    canonical_for: dict[str, str] = {}
    for group in prepared.alias_groups:
        for path in group:
            alias_for[path] = group
            canonical_for[path] = group[0]
    canonical_paths = tuple(
        site.path
        for site in ordered
        if canonical_for.get(site.path, site.path) == site.path
    )
    canonical_updates: dict[str, LowRankUpdate] = {}
    for index, path in enumerate(canonical_paths):
        spec = requested[path]
        canonical_updates[path] = LowRankUpdate(
            weights[path],
            rank=spec.rank,
            alpha=spec.alpha,
            scaling=spec.scaling,
            stddev=spec.stddev,
            key=jr.fold_in(key, index),
        )
    replacements: dict[str, LowRankUpdate] = {}
    records: list[LowRankAdaptationSite] = []
    for site in ordered:
        canonical = canonical_for.get(site.path, site.path)
        update = canonical_updates[canonical]
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
                adapter_parameter_count=(
                    update.adapter_parameter_count if canonical == site.path else 0
                ),
                handler=handlers[site.path],
                complex_representation=(
                    "native_complex" if jnp.iscomplexobj(update.base) else "real"
                ),
                alias_group=alias_for.get(site.path, ()),
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
        base_parameter_count=sum(
            site.base_parameter_count
            for site in records_
            if not site.alias_group or site.path == site.alias_group[0]
        ),
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


def low_rank_parameter_subspace(
    tree: PyTree[Any],
    /,
    *,
    plan: LowRankAdaptationPlan | None = None,
) -> ParameterSubspace:
    """Select canonical factors and reconstruct every declared semantic alias."""
    paths = _low_rank_factor_paths(tree)
    if not paths:
        raise ValueError("The supplied PyTree contains no low-rank updates.")
    alias_groups: tuple[tuple[str, ...], ...] = ()
    if plan is not None:
        if not isinstance(plan, LowRankAdaptationPlan):
            raise TypeError("plan must be LowRankAdaptationPlan or None.")
        groups = []
        for weights in plan.alias_groups:
            groups.append(tuple(f"{path}.left" for path in weights))
            groups.append(tuple(f"{path}.right" for path in weights))
        alias_groups = tuple(groups)
    return ParameterSubspace.from_leaf_paths(
        tree,
        paths,
        alias_groups=alias_groups,
    )


def validate_low_rank_subspace(
    tree: PyTree[Any],
    subspace: ParameterSubspace,
    /,
) -> None:
    """Require all and only low-rank factor leaves in a training subspace."""
    expected = _low_rank_factor_paths(tree)
    if not expected:
        return
    alias_paths = frozenset(path for group in subspace.alias_groups for path in group[1:])
    canonical_expected = tuple(path for path in expected if path not in alias_paths)
    if subspace.leaf_paths != canonical_expected:
        raise ValueError(
            "Low-rank training requires exactly canonical left/right factors and "
            "declared semantic alias reconstruction."
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
    "LowRankAdaptationPlan",
    "LowRankAdaptationSite",
    "LowRankSpec",
    "LowRankScaling",
    "LowRankSiteHandler",
    "LowRankUpdate",
    "adapt_low_rank",
    "contains_low_rank_updates",
    "low_rank_parameter_subspace",
    "low_rank_sites",
    "prepare_low_rank_adaptation",
    "merge_low_rank",
    "validate_low_rank_subspace",
]

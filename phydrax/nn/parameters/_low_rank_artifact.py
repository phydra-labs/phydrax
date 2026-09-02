#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_mapping,
)
from ..._model import artifact_value_id
from ..._model._structure import model_structure_recipe
from ._low_rank import (
    _low_rank_nodes,
    _strip_low_rank,
    LowRankAdaptationPlan,
    LowRankAdaptationSite,
    LowRankUpdate,
)


_LOW_RANK_ADAPTER_FORMAT = "phydrax-low-rank-adapter"


@dataclass(frozen=True, slots=True)
class LowRankAdapterManifest:
    """Verified identity and site metadata for one adapter-only artifact."""

    base_model_type: str
    base_structure_sha256: str
    base_array_sha256: str
    base_array_signature: tuple[Mapping[str, Any], ...]
    sites: tuple[LowRankAdaptationSite, ...]

    @property
    def alias_groups(self) -> tuple[tuple[str, ...], ...]:
        groups = []
        for site in self.sites:
            if site.alias_group and site.alias_group not in groups:
                groups.append(site.alias_group)
        return tuple(groups)

    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LowRankAdapterArtifact:
    """A restored adapted model and its verified base-binding manifest."""

    model: Any

    def parameter_subspace(self):
        from ._selection import ParameterSubspace

        paths = []
        groups = []
        for site in self.manifest.sites:
            paths.extend((f"{site.path}.left", f"{site.path}.right"))
        for weights in self.manifest.alias_groups:
            groups.append(tuple(f"{path}.left" for path in weights))
            groups.append(tuple(f"{path}.right" for path in weights))
        return ParameterSubspace.from_leaf_paths(
            self.model,
            tuple(paths),
            alias_groups=tuple(groups),
        )

    manifest: LowRankAdapterManifest


def _base_identity(model: Any, /) -> tuple[str, str, dict[str, Any]]:
    recipe = model_structure_recipe(model)
    return (
        artifact_value_id(type(model)),
        canonical_fingerprint(recipe),
        array_tree_fingerprint(model),
    )


def save_low_rank_adapter(
    path: str | Path,
    model: Any,
    /,
    *,
    provenance: Mapping[str, Any] | None = None,
    plan: LowRankAdaptationPlan | None = None,
) -> Path:
    """Write only low-rank factors, bound to the exact dense base model."""
    nodes = _low_rank_nodes(model)
    if not nodes:
        raise ValueError("Cannot save a low-rank adapter from a model without adapters.")
    base_model = _strip_low_rank(model)
    base_type, structure_sha256, array_fingerprint = _base_identity(base_model)
    arrays: dict[str, Any] = {}
    if plan is not None and not isinstance(plan, LowRankAdaptationPlan):
        raise TypeError("plan must be LowRankAdaptationPlan or None.")
    alias_for: dict[str, tuple[str, ...]] = {}
    canonical_for: dict[str, str] = {}
    handlers = {} if plan is None else dict(plan.site_handlers)
    if plan is not None:
        for group in plan.alias_groups:
            for path in group:
                alias_for[path] = group
                canonical_for[path] = group[0]
    factor_names: dict[str, tuple[str, str]] = {}
    sites: list[dict[str, Any]] = []
    for index, (weight_path, update) in enumerate(nodes):
        canonical = canonical_for.get(weight_path, weight_path)
        if canonical not in factor_names:
            left_name = f"site/{len(factor_names):06d}/left"
            right_name = f"site/{len(factor_names):06d}/right"
            arrays[left_name] = update.left
            arrays[right_name] = update.right
            factor_names[canonical] = (left_name, right_name)
        left_name, right_name = factor_names[canonical]
        sites.append(
            {
                "path": weight_path,
                "shape": list(update.shape),
                "dtype": jnp.dtype(update.dtype).str,
                "rank": update.rank,
                "alpha": update.alpha,
                "scaling": update.scaling,
                "scale": update.scale,
                "left": left_name,
                "right": right_name,
                "handler": handlers.get(weight_path, "identity"),
                "alias_group": list(alias_for.get(weight_path, ())),
            }
        )
    manifest = {
        "format": _LOW_RANK_ADAPTER_FORMAT,
        "base_model_type": base_type,
        "base_structure_sha256": structure_sha256,
        "base_arrays": array_fingerprint,
        "sites": sites,
        "provenance": canonical_mapping({} if provenance is None else provenance),
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def _validated_sites(
    value: Any,
    arrays: Mapping[str, np.ndarray],
    /,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ArrayArchiveCorruptionError("Low-rank adapter sites are missing.")
    expected_fields = {
        "path",
        "shape",
        "dtype",
        "rank",
        "scaling",
        "alpha",
        "scale",
        "left",
        "right",
        "handler",
        "alias_group",
    }
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_arrays: set[str] = set()
    for record in value:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ArrayArchiveCorruptionError(
                "Low-rank adapter site metadata is invalid."
            )
        path = record["path"]
        if not isinstance(path, str) or not path or path in seen_paths:
            raise ArrayArchiveCorruptionError("Low-rank adapter paths are invalid.")
        left_name = record["left"]
        right_name = record["right"]
        if (
            not isinstance(left_name, str)
            or not isinstance(right_name, str)
            or left_name == right_name
            or left_name not in arrays
            or right_name not in arrays
        ):
            raise ArrayArchiveCorruptionError(
                "Low-rank adapter factor inventory is invalid."
            )
        shape = record["shape"]
        rank = record["rank"]
        alpha = record["alpha"]
        scaling = record["scaling"]
        scale = record["scale"]
        dtype = record["dtype"]
        handler = record["handler"]
        alias_group = record["alias_group"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or not isinstance(dtype, str)
            or scaling not in ("rank", "sqrt_rank")
            or not isinstance(alpha, (int, float))
            or not isinstance(scale, (int, float))
            or not np.isfinite(alpha)
            or not np.isfinite(scale)
            or alpha <= 0.0
            or scale <= 0.0
            or float(scale)
            != float(alpha) / (rank if scaling == "rank" else np.sqrt(rank))
            or handler not in ("identity", "symmetric", "skew")
            or not isinstance(alias_group, list)
            or any(not isinstance(item, str) or not item for item in alias_group)
            or (alias_group and path not in alias_group)
        ):
            raise ArrayArchiveCorruptionError("Low-rank adapter site values are invalid.")
        seen_paths.add(path)
        seen_arrays.update((left_name, right_name))
        records.append(record)
    if seen_arrays != set(arrays):
        raise ArrayArchiveCorruptionError(
            "Low-rank adapter contains unreferenced factor arrays."
        )
    return tuple(records)


def read_low_rank_adapter(
    path: str | Path,
    base_model: Any,
    /,
) -> LowRankAdapterArtifact:
    """Restore an adapter only after verifying the exact supplied base model."""
    manifest, arrays = read_array_archive(path)
    expected_fields = {
        "format",
        "base_model_type",
        "base_structure_sha256",
        "base_arrays",
        "sites",
        "provenance",
        "arrays",
    }
    if set(manifest) != expected_fields or manifest["format"] != _LOW_RANK_ADAPTER_FORMAT:
        raise ArrayArchiveCorruptionError(
            "Archive is not a canonical Phydrax low-rank adapter."
        )
    base_type, structure_sha256, array_fingerprint = _base_identity(base_model)
    if manifest["base_model_type"] != base_type:
        raise ValueError("Low-rank adapter base model type mismatch.")
    if manifest["base_structure_sha256"] != structure_sha256:
        raise ValueError("Low-rank adapter base model structure mismatch.")
    if manifest["base_arrays"] != array_fingerprint:
        raise ValueError("Low-rank adapter base model content mismatch.")
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict):
        raise ArrayArchiveCorruptionError("Low-rank adapter provenance is invalid.")
    records = _validated_sites(manifest["sites"], arrays)
    records_by_path = {record["path"]: record for record in records}

    from ..layers._linear import Linear
    from ._transforms import SkewSymmetricTransform, SymmetricTransform

    restored_paths: list[str] = []

    def restore(path_, value):
        if not isinstance(value, Linear):
            return value
        prefix = jax.tree_util.keystr(path_)
        weight_path = f"{prefix}.weight" if prefix else ".weight"
        record = records_by_path.get(weight_path)
        if record is None:
            return value
        if isinstance(value.weight, LowRankUpdate):
            raise TypeError("Low-rank adapters require a dense supplied base model.")
        handler = record["handler"]
        transform_valid = (
            (handler == "identity" and value.weight_transform is None)
            or (
                handler == "symmetric"
                and isinstance(value.weight_transform, SymmetricTransform)
            )
            or (
                handler == "skew"
                and isinstance(value.weight_transform, SkewSymmetricTransform)
            )
        )
        if not transform_valid:
            raise ValueError(
                f"Low-rank adapter site {weight_path!r} transform handler changed."
            )
        weight = jnp.asarray(value.weight)
        shape = tuple(int(size) for size in record["shape"])
        dtype = str(record["dtype"])
        if tuple(weight.shape) != shape or jnp.dtype(weight.dtype).str != dtype:
            raise ValueError(
                f"Low-rank adapter site {weight_path!r} shape or dtype mismatch."
            )
        left = jnp.asarray(arrays[record["left"]])
        right = jnp.asarray(arrays[record["right"]])
        update = LowRankUpdate.from_factors(
            weight,
            left,
            right,
            alpha=float(record["alpha"]),
            scaling=record["scaling"],
        )
        if update.rank != int(record["rank"]) or update.scale != float(record["scale"]):
            raise ArrayArchiveCorruptionError(
                f"Low-rank adapter site {weight_path!r} factors are inconsistent."
            )
        restored_paths.append(weight_path)
        return eqx.tree_at(lambda layer: layer.weight, value, update)

    restored = jax.tree_util.tree_map_with_path(
        restore,
        base_model,
        is_leaf=lambda value: isinstance(value, Linear),
    )
    if tuple(restored_paths) != tuple(record["path"] for record in records):
        raise ValueError("Low-rank adapter contains unknown or reordered weight paths.")
    sites = tuple(
        LowRankAdaptationSite(
            path=record["path"],
            shape=tuple(record["shape"]),
            dtype=record["dtype"],
            rank=int(record["rank"]),
            alpha=float(record["alpha"]),
            scaling=record["scaling"],
            scale=float(record["scale"]),
            base_parameter_count=int(np.prod(record["shape"])),
            adapter_parameter_count=(
                int(
                    np.prod(arrays[record["left"]].shape)
                    + np.prod(arrays[record["right"]].shape)
                )
                if not record["alias_group"] or record["path"] == record["alias_group"][0]
                else 0
            ),
            handler=record["handler"],
            complex_representation=(
                "native_complex"
                if np.issubdtype(np.dtype(record["dtype"]), np.complexfloating)
                else "real"
            ),
            alias_group=tuple(record["alias_group"]),
        )
        for record in records
    )
    parsed = LowRankAdapterManifest(
        base_model_type=base_type,
        base_structure_sha256=structure_sha256,
        base_array_sha256=array_fingerprint["sha256"],
        base_array_signature=tuple(
            MappingProxyType(dict(item)) for item in array_fingerprint["signature"]
        ),
        sites=sites,
        provenance=MappingProxyType(dict(provenance)),
    )
    return LowRankAdapterArtifact(model=restored, manifest=parsed)


__all__ = [
    "LowRankAdapterArtifact",
    "LowRankAdapterManifest",
    "read_low_rank_adapter",
    "save_low_rank_adapter",
]

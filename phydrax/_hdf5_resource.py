#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded structural admission for profile-specific HDF5 decoders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from ._external_resource import OpenedResource, ResourceManifest, ResourceReadError
from ._fingerprint import canonical_fingerprint


@dataclass(frozen=True, slots=True)
class HDF5Limits:
    """Finite structural and logical-allocation bounds for one HDF5 resource."""

    max_objects: int
    max_datasets: int
    max_attributes: int
    max_attribute_bytes: int
    max_rank: int
    max_axis_length: int
    max_dataset_elements: int
    max_dataset_bytes: int
    max_total_dataset_bytes: int
    max_dtype_itemsize: int
    allowed_filter_ids: frozenset[int] = frozenset({1, 2, 3})
    allow_virtual_datasets: bool = False
    allow_soft_links: bool = False
    allow_reference_or_object_dtypes: bool = False

    def __post_init__(self) -> None:
        values = (
            self.max_objects,
            self.max_datasets,
            self.max_attributes,
            self.max_attribute_bytes,
            self.max_rank,
            self.max_axis_length,
            self.max_dataset_elements,
            self.max_dataset_bytes,
            self.max_total_dataset_bytes,
            self.max_dtype_itemsize,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("HDF5 limits must be positive integers.")
        if self.max_dataset_bytes > self.max_total_dataset_bytes:
            raise ValueError("HDF5 dataset bytes cannot exceed aggregate bytes.")
        if any(
            type(identifier) is not int or identifier < 0
            for identifier in self.allowed_filter_ids
        ):
            raise ValueError("HDF5 allowed filter IDs must be nonnegative integers.")


@dataclass(frozen=True, slots=True)
class HDF5DatasetManifest:
    """Pre-allocation metadata for one admitted HDF5 dataset."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    logical_size_bytes: int
    filter_ids: tuple[int, ...]
    manifest_id: str


@dataclass(frozen=True, slots=True)
class HDF5Manifest:
    """Bounded structural inventory for one exact HDF5 resource."""

    resource: ResourceManifest
    groups: int
    attributes: int
    attribute_bytes: int
    datasets: tuple[HDF5DatasetManifest, ...]
    total_dataset_bytes: int
    manifest_id: str


def inspect_hdf5_resource(
    resource: OpenedResource,
    /,
    *,
    limits: HDF5Limits,
) -> HDF5Manifest:
    """Inspect links, shapes, dtypes, filters, and logical bytes without data reads."""

    if not isinstance(resource, OpenedResource):
        raise TypeError("resource must be an OpenedResource.")
    h5py = import_module("h5py")
    resource.stream.seek(0)
    datasets: list[HDF5DatasetManifest] = []
    counts = {"objects": 1, "groups": 1, "attributes": 0, "attribute_bytes": 0}
    total_dataset_bytes = 0
    try:
        with h5py.File(resource.stream, mode="r") as handle:
            counts["attributes"], counts["attribute_bytes"] = _attribute_counts(
                handle,
                limits,
                counts["attributes"],
                counts["attribute_bytes"],
            )
            pending: list[tuple[str, Any]] = [("", handle)]
            while pending:
                prefix, group = pending.pop()
                for name in sorted(group.keys()):
                    link = group.get(name, getlink=True)
                    path = f"{prefix}/{name}"
                    if isinstance(link, h5py.ExternalLink):
                        raise ResourceReadError(
                            "policy", "HDF5 external links are disabled."
                        )
                    if isinstance(link, h5py.SoftLink) and not limits.allow_soft_links:
                        raise ResourceReadError("policy", "HDF5 soft links are disabled.")
                    if not isinstance(link, (h5py.HardLink, h5py.SoftLink)):
                        raise ResourceReadError(
                            "policy", "HDF5 link type is not admitted."
                        )
                    item = group[name]
                    counts["objects"] += 1
                    if counts["objects"] > limits.max_objects:
                        raise ResourceReadError("limit", "HDF5 exceeds its object limit.")
                    counts["attributes"], counts["attribute_bytes"] = _attribute_counts(
                        item,
                        limits,
                        counts["attributes"],
                        counts["attribute_bytes"],
                    )
                    if isinstance(item, h5py.Group):
                        counts["groups"] += 1
                        pending.append((path, item))
                        continue
                    if not isinstance(item, h5py.Dataset):
                        raise ResourceReadError(
                            "policy", "HDF5 object type is not admitted."
                        )
                    if len(datasets) >= limits.max_datasets:
                        raise ResourceReadError(
                            "limit", "HDF5 exceeds its dataset limit."
                        )
                    if item.is_virtual and not limits.allow_virtual_datasets:
                        raise ResourceReadError(
                            "policy", "HDF5 virtual datasets are disabled."
                        )
                    dtype = np.dtype(item.dtype)
                    if dtype.itemsize > limits.max_dtype_itemsize or (
                        dtype.hasobject and not limits.allow_reference_or_object_dtypes
                    ):
                        raise ResourceReadError(
                            "policy", "HDF5 dataset dtype is not admitted."
                        )
                    shape = tuple(int(extent) for extent in item.shape)
                    if len(shape) > limits.max_rank:
                        raise ResourceReadError(
                            "limit", "HDF5 dataset exceeds its rank limit."
                        )
                    elements = 1
                    for extent in shape:
                        if extent < 0 or extent > limits.max_axis_length:
                            raise ResourceReadError(
                                "limit", "HDF5 dataset axis is not admitted."
                            )
                        if extent and elements > limits.max_dataset_elements // extent:
                            raise ResourceReadError(
                                "limit", "HDF5 dataset exceeds its element limit."
                            )
                        elements *= extent
                    logical_bytes = elements * dtype.itemsize
                    if logical_bytes > limits.max_dataset_bytes:
                        raise ResourceReadError(
                            "limit", "HDF5 dataset exceeds its byte limit."
                        )
                    if (
                        total_dataset_bytes
                        > limits.max_total_dataset_bytes - logical_bytes
                    ):
                        raise ResourceReadError(
                            "limit", "HDF5 exceeds its aggregate dataset byte limit."
                        )
                    total_dataset_bytes += logical_bytes
                    properties = item.id.get_create_plist()
                    filter_ids = tuple(
                        int(properties.get_filter(index)[0])
                        for index in range(properties.get_nfilters())
                    )
                    if any(
                        identifier not in limits.allowed_filter_ids
                        for identifier in filter_ids
                    ):
                        raise ResourceReadError(
                            "policy", "HDF5 dataset filter is not admitted."
                        )
                    payload = {
                        "kind": "hdf5-dataset-manifest",
                        "path": path,
                        "shape": list(shape),
                        "dtype": dtype.str,
                        "logical_size_bytes": logical_bytes,
                        "filter_ids": list(filter_ids),
                    }
                    datasets.append(
                        HDF5DatasetManifest(
                            path,
                            shape,
                            dtype.str,
                            logical_bytes,
                            filter_ids,
                            canonical_fingerprint(payload),
                        )
                    )
    except ResourceReadError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ResourceReadError(
            "malformed", "HDF5 resource structure is invalid."
        ) from error
    ordered = tuple(sorted(datasets, key=lambda dataset: dataset.path))
    payload = {
        "kind": "hdf5-manifest",
        "resource_manifest_id": resource.manifest.manifest_id,
        "groups": counts["groups"],
        "attributes": counts["attributes"],
        "attribute_bytes": counts["attribute_bytes"],
        "datasets": [dataset.manifest_id for dataset in ordered],
        "total_dataset_bytes": total_dataset_bytes,
    }
    return HDF5Manifest(
        resource.manifest,
        counts["groups"],
        counts["attributes"],
        counts["attribute_bytes"],
        ordered,
        total_dataset_bytes,
        canonical_fingerprint(payload),
    )


def _attribute_counts(
    item: Any,
    limits: HDF5Limits,
    attributes: int,
    attribute_bytes: int,
    /,
) -> tuple[int, int]:
    for name in item.attrs:
        attributes += 1
        if attributes > limits.max_attributes:
            raise ResourceReadError("limit", "HDF5 exceeds its attribute limit.")
        value = np.asarray(item.attrs[name])
        attribute_bytes += int(value.nbytes)
        if attribute_bytes > limits.max_attribute_bytes:
            raise ResourceReadError("limit", "HDF5 exceeds its attribute byte limit.")
    return attributes, attribute_bytes


__all__ = [
    "HDF5DatasetManifest",
    "HDF5Limits",
    "HDF5Manifest",
    "inspect_hdf5_resource",
]

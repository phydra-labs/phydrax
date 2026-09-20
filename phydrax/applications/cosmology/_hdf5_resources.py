#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded HDF5 dataset admission shared by cosmology products."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from ..._external_resource import open_bounded_resource, ResourceLimits
from ..._hdf5_resource import HDF5Limits, inspect_hdf5_resource


def read_cosmology_hdf5_arrays(
    path: str | Path,
    dataset_names: Sequence[str],
    /,
    *,
    trusted_root: str | Path | None,
    maximum_file_bytes: int,
    maximum_decoded_bytes: int,
) -> dict[str, np.ndarray]:
    """Read exact named datasets only after structural and allocation preflight."""

    names = tuple(str(name) for name in dataset_names)
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Cosmology HDF5 dataset names must be non-empty and unique.")
    source = Path(path).expanduser().absolute()
    root = source.parent if trusted_root is None else Path(trusted_root)
    with open_bounded_resource(
        source,
        trusted_root=root,
        limits=ResourceLimits(maximum_file_bytes, 64, 1_000_000, 1_000_000, 128),
    ) as resource:
        inspection = inspect_hdf5_resource(
            resource,
            limits=HDF5Limits(
                100_000,
                10_000,
                100_000,
                64 * 1024 * 1024,
                8,
                1_000_000_000,
                2_000_000_000,
                maximum_decoded_bytes,
                maximum_decoded_bytes,
                32,
            ),
        )
        available = {dataset.path.lstrip("/") for dataset in inspection.datasets}
        if any(name.lstrip("/") not in available for name in names):
            raise ValueError("Cosmology HDF5 source is missing a required dataset.")
        resource.stream.seek(0)
        with h5py.File(resource.stream, "r") as handle:
            arrays = {name: np.asarray(handle[name]) for name in names}
    total = sum(value.nbytes for value in arrays.values())
    if total > maximum_decoded_bytes:
        raise ValueError("Cosmology HDF5 selected datasets exceed their byte limit.")
    return arrays


__all__ = ["read_cosmology_hdf5_arrays"]

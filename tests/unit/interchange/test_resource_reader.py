#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import phydrax.interchange._resource as resource_module
from phydrax.interchange._resource import (
    account_bounded_resource,
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)


def _limits(*, max_bytes: int = 1024, max_depth: int = 8) -> ResourceLimits:
    return ResourceLimits(max_bytes, max_depth, 32, 64, 8)


def test_descriptor_relative_read_retains_exact_immutable_manifest(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "resource.bin"
    source.write_bytes(b"exact\x00bytes")

    loaded = read_bounded_resource(
        "nested/resource.bin", trusted_root=tmp_path, limits=_limits()
    )
    accounted = account_bounded_resource(
        loaded, depth=3, nodes=7, attributes=5, losses=1
    )

    assert loaded.data == b"exact\x00bytes"
    assert loaded.manifest.content_sha256 == hashlib.sha256(loaded.data).hexdigest()
    assert loaded.manifest.source_path == str(source)
    assert loaded.manifest.relative_components == ("nested", "resource.bin")
    assert loaded.manifest.file_inode == source.stat().st_ino
    assert accounted.manifest.observed_depth == 3
    assert accounted.manifest.observed_nodes == 7
    assert accounted.manifest.observed_attributes == 5
    assert accounted.manifest.observed_losses == 1
    with pytest.raises(FrozenInstanceError):
        accounted.manifest.size_bytes = 0  # type: ignore[misc]


@pytest.mark.parametrize("path", ("../outside.bin", "nested/../../outside.bin"))
def test_traversal_is_rejected_before_descriptor_walk(tmp_path: Path, path: str):
    with pytest.raises(ResourceReadError) as caught:
        read_bounded_resource(path, trusted_root=tmp_path, limits=_limits())
    assert caught.value.reason == "policy"


def test_symlink_roots_components_and_files_are_rejected(tmp_path: Path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "resource.bin").write_bytes(b"outside")
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    component_link = real_root / "component"
    component_link.symlink_to(outside, target_is_directory=True)
    file_link = real_root / "resource.bin"
    file_link.symlink_to(outside / "resource.bin")

    for path, root in (
        ("resource.bin", root_link),
        ("component/resource.bin", real_root),
        ("resource.bin", real_root),
    ):
        with pytest.raises(ResourceReadError) as caught:
            read_bounded_resource(path, trusted_root=root, limits=_limits())
        assert caught.value.reason == "policy"


def test_special_file_and_oversize_resource_fail_closed(tmp_path: Path):
    fifo = tmp_path / "stream"
    os.mkfifo(fifo)
    oversized = tmp_path / "large.bin"
    oversized.write_bytes(b"x" * 17)

    with pytest.raises(ResourceReadError) as special:
        read_bounded_resource("stream", trusted_root=tmp_path, limits=_limits())
    assert special.value.reason == "policy"

    with pytest.raises(ResourceReadError) as too_large:
        read_bounded_resource(
            "large.bin", trusted_root=tmp_path, limits=_limits(max_bytes=16)
        )
    assert too_large.value.reason == "limit"


def test_component_path_swap_is_detected(tmp_path: Path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "resource.bin").write_bytes(b"trusted")
    held = tmp_path / "held"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "resource.bin").write_bytes(b"substituted")
    real_open = resource_module.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "nested" and not swapped:
            nested.rename(held)
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(resource_module.os, "open", swapping_open)

    with pytest.raises(ResourceReadError) as caught:
        read_bounded_resource(
            "nested/resource.bin", trusted_root=tmp_path, limits=_limits()
        )
    assert caught.value.reason == "inconsistent"

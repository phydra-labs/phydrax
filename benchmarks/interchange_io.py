#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scaling evidence for bounded admission and crash-consistent publication."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from phydrax._external_resource import (
    open_bounded_resource,
    read_bounded_resource,
    ResourceLimits,
)
from phydrax._publication import publish_bytes, publish_resource_set
from phydrax._resource_set import read_bounded_resource_set, ResourceSetLimits


def _elapsed(operation):
    start = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - start


def _single_file(root: Path, size: int) -> dict[str, object]:
    payload = bytes((index % 251 for index in range(size)))
    destination = root / f"single-{size}.bin"
    receipt, publish_seconds = _elapsed(
        lambda: publish_bytes(destination, payload, maximum_bytes=size)
    )
    limits = ResourceLimits(size, 4, 1, 0, 0)
    resident, resident_seconds = _elapsed(
        lambda: read_bounded_resource(
            destination.name,
            trusted_root=root,
            limits=limits,
        )
    )

    def stream_identity():
        with open_bounded_resource(
            destination.name,
            trusted_root=root,
            limits=limits,
        ) as resource:
            while resource.stream.read(1024 * 1024):
                pass
            return resource.manifest

    manifest, stream_seconds = _elapsed(stream_identity)
    if resident.data != payload or manifest.content_sha256 != receipt.content_sha256:
        raise RuntimeError("Single-file benchmark identity mismatch.")
    return {
        "bytes": size,
        "publish_seconds": publish_seconds,
        "resident_read_seconds": resident_seconds,
        "stream_admit_and_read_seconds": stream_seconds,
        "content_sha256": receipt.content_sha256,
    }


def _resource_set(root: Path, member_count: int, member_bytes: int) -> dict[str, object]:
    members = {
        f"group-{index // 64:04d}/member-{index:08d}.bin": bytes(
            ((index + offset) % 251 for offset in range(member_bytes))
        )
        for index in range(member_count)
    }
    destination = root / f"set-{member_count}"
    limits = ResourceSetLimits(
        max_total_bytes=member_count * member_bytes,
        max_member_bytes=member_bytes,
        max_members=member_count,
        max_depth=2,
    )
    receipt, publish_seconds = _elapsed(
        lambda: publish_resource_set(destination, members, limits=limits)
    )
    admitted, admission_seconds = _elapsed(
        lambda: read_bounded_resource_set(
            destination.name,
            trusted_root=root,
            limits=limits,
        )
    )
    if admitted.manifest.aggregate_sha256 != receipt.aggregate_sha256:
        raise RuntimeError("Resource-set benchmark identity mismatch.")
    return {
        "members": member_count,
        "member_bytes": member_bytes,
        "total_bytes": receipt.total_size_bytes,
        "publish_seconds": publish_seconds,
        "admission_seconds": admission_seconds,
        "aggregate_sha256": receipt.aggregate_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="1048576,16777216,67108864",
        help="Comma-separated single-file byte sizes.",
    )
    parser.add_argument(
        "--member-counts",
        default="1,64,512",
        help="Comma-separated resource-set member counts.",
    )
    parser.add_argument("--member-bytes", type=int, default=4096)
    arguments = parser.parse_args()
    sizes = tuple(int(value) for value in arguments.sizes.split(","))
    counts = tuple(int(value) for value in arguments.member_counts.split(","))
    if any(value <= 0 for value in (*sizes, *counts, arguments.member_bytes)):
        raise ValueError("Benchmark capacities must be positive.")
    with tempfile.TemporaryDirectory(prefix="phydrax-io-benchmark-") as temporary:
        root = Path(temporary)
        result = {
            "single_files": [_single_file(root, size) for size in sizes],
            "resource_sets": [
                _resource_set(root, count, arguments.member_bytes) for count in counts
            ],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

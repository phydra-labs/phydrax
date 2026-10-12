"""Host scratch scaling and canonical archive identity evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import capture_benchmark_identity, capture_environment
from phydrax._array_archive import (
    array_collection_digest,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from phydrax._external_resource import bounded_resource_from_bytes, ResourceLimits
from phydrax._identity import NumericRevision, SemanticProvenance
from phydrax._numpy_resource import decode_npz_resource, NumpyFormatLimits
from phydrax.lifecycle._archive import (
    create,
    export,
    LifecycleArchive,
    support_bundle,
    SupportBundleAuthorization,
)
from phydrax.lifecycle._distributed_checkpoint import (
    _read_shard_payload,
    publish_process_checkpoint,
)
from phydrax.lifecycle._models import RevisionLineage
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)


def _support_snapshot(
    archive: LifecycleArchive,
    destination: Path,
    authorization: SupportBundleAuthorization,
) -> dict[str, str]:
    try:
        support_bundle(archive, destination, authorization=authorization)
    except ValueError as error:
        if "axis-length limit" not in str(error):
            raise
        return {"status": "refused", "reason": str(error)}
    return {"status": "ok"}


def run(counts: tuple[int, ...], member_bytes: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as directory:
        directory = str(Path(directory).resolve())
        for count in counts:
            arrays = {
                f"a{i:04d}": np.arange(member_bytes // 8, dtype=np.float64)
                for i in range(count)
            }
            path = Path(directory) / "archive.zip"
            started = time.perf_counter()
            _, peak = host_allocation_peak(
                lambda: write_array_archive(
                    path, manifest={"kind": "memory-benchmark"}, arrays=arrays
                )
            )
            records.append(
                {
                    "case": "archive-write",
                    "capacity": count,
                    "input_bytes": count * member_bytes,
                    "output_bytes": path.stat().st_size,
                    "scratch_peak_bytes": peak,
                    "wall_seconds": time.perf_counter() - started,
                    "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                    "collection_digest": array_collection_digest(arrays),
                }
            )
            decoded, peak = host_allocation_peak(lambda: read_array_archive(path))
            output_bytes = sum(a.nbytes for a in decoded[1].values())
            records.append(
                {
                    "case": "archive-read",
                    "capacity": count,
                    "input_bytes": path.stat().st_size,
                    "output_bytes": output_bytes,
                    "scratch_peak_bytes": max(0, peak - output_bytes),
                }
            )
            buffer = io.BytesIO()
            np.savez(buffer, allow_pickle=False, **arrays)
            resource = bounded_resource_from_bytes(
                buffer.getvalue(),
                limits=ResourceLimits(128 * 1024 * 1024, 16, 1000, 1000, 1000),
            )
            decoded_npz, peak = host_allocation_peak(
                lambda: decode_npz_resource(
                    resource, limits=NumpyFormatLimits(max_arrays=257)
                )
            )
            output_bytes = sum(a.nbytes for a in decoded_npz.arrays.values())
            records.append(
                {
                    "case": "npz-read",
                    "capacity": count,
                    "input_bytes": resource.manifest.size_bytes,
                    "output_bytes": output_bytes,
                    "scratch_peak_bytes": max(0, peak - output_bytes),
                }
            )
            revision = RevisionLineage(
                NumericRevision(SemanticProvenance({"kind": "memory-benchmark"}), arrays)
            )
            archive = create(
                Path(directory) / "lifecycle.zip", manifest=revision, arrays=arrays
            )
            destination = Path(directory) / "export.npz"
            _, peak = host_allocation_peak(
                lambda: export(archive, destination, format="npz")
            )
            records.append(
                {
                    "case": "npz-export",
                    "capacity": count,
                    "input_bytes": count * member_bytes,
                    "output_bytes": destination.stat().st_size,
                    "scratch_peak_bytes": peak,
                }
            )
            authorization = SupportBundleAuthorization(
                "memory-authorization",
                "memory-owner",
                archive.archive_id,
                1,
                frozenset(
                    {"arrays", "payloads", "paths", "identifiers", "free-text", "secrets"}
                ),
            )
            destination = Path(directory) / "support.zip"
            status, peak = host_allocation_peak(
                lambda: _support_snapshot(archive, destination, authorization)
            )
            records.append(
                {
                    "case": "support-snapshot",
                    "capacity": count,
                    "input_bytes": archive.path.stat().st_size,
                    "output_bytes": destination.stat().st_size
                    if status["status"] == "ok"
                    else 0,
                    "scratch_peak_bytes": peak,
                    **status,
                }
            )
            profile = HPCFilesystemProfile(
                "posix.memory",
                "local-posix",
                atomic_rename_same_filesystem=True,
                file_fsync=True,
                directory_fsync=True,
                advisory_locking=True,
                attempt_private_staging=True,
            )
            repository = POSIXArtifactRepository(
                Path(directory) / f"repository-{count}",
                POSIXRepositoryPolicy(
                    profile,
                    maximum_chunk_bytes=member_bytes,
                    maximum_metadata_bytes=65536,
                ),
            )
            value = jnp.arange(count * member_bytes // 8, dtype=jnp.float64)
            publication = publish_process_checkpoint(
                repository,
                "memory-checkpoint",
                "memory-execution",
                {"state": value},
                analysis_plan_id="memory-analysis",
                numeric_revision_id="memory-revision",
                writer_id="memory-writer",
            )
            decoded_shard, peak = host_allocation_peak(
                lambda: _read_shard_payload(
                    repository,
                    publication.shards[0],
                    value.shape,
                    np.dtype(np.float64),
                    DEFAULT_ARRAY_ARCHIVE_LIMITS,
                )
            )
            records.append(
                {
                    "case": "checkpoint-read",
                    "capacity": count,
                    "input_bytes": publication.shards[0].byte_count,
                    "output_bytes": decoded_shard.nbytes,
                    "scratch_peak_bytes": max(0, peak - decoded_shard.nbytes),
                    "payload_digest": publication.shards[0].payload_digest,
                }
            )
    scaling = {}
    for case in (
        "archive-write",
        "archive-read",
        "npz-read",
        "npz-export",
        "support-snapshot",
        "checkpoint-read",
    ):
        accepted = [
            r for r in records if r["case"] == case and r.get("status") != "refused"
        ]
        series = [r["scratch_peak_bytes"] for r in accepted]
        scaling[case] = {
            "exponent": fitted_scaling_exponent(
                [r["capacity"] for r in accepted], series
            ),
            "fourfold_ratios": [b / a for a, b in zip(series, series[1:])],
            "scratch_bound_met": all(b <= 1.5 * a for a, b in zip(series, series[1:])),
        }
    root = Path(__file__).resolve().parents[1]
    return {
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            (
                "scratch_peak_bytes",
                "input_bytes",
                "output_bytes",
                "sha256",
                "collection_digest",
            ),
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "member_bytes": member_bytes,
        "records": records,
        "scaling": scaling,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/memory_archive_streaming.json")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    write_json_atomic(
        args.output, run((4, 16) if args.smoke else (4, 16, 64), 1024 * 1024)
    )


if __name__ == "__main__":
    main()

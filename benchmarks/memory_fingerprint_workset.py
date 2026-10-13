"""Host scratch evidence for fingerprints and canonical workset identity."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import capture_benchmark_identity, capture_environment
from phydrax._execution_workset import _prepared_identity
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.execution import ExecutionWorksetPlan, PoolExecutionSignature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/memory_fingerprint_workset.json")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sizes = (1, 4, 16) if args.smoke else (1, 4, 16, 64)
    records: list[dict[str, Any]] = []
    for mib in sizes:
        source = np.arange(mib * 1024 * 1024 // 8, dtype=np.float64)
        for route, value in (
            ("contiguous", source),
            ("strided", source.reshape(-1, 2).T),
        ):
            start = perf_counter()
            result, peak = host_allocation_peak(lambda: array_tree_fingerprint(value))
            records.append(
                {
                    "case": route,
                    "capacity": mib,
                    "input_bytes": source.nbytes,
                    "scratch_peak_bytes": peak,
                    "seconds": perf_counter() - start,
                    "sha256": result["sha256"],
                }
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "retained.bin"
            with path.open("wb") as stream:
                stream.write(memoryview(source).cast("B"))
            for route in ("file-legacy-reference", "file-stream"):

                def file_hash() -> str:
                    if route == "file-legacy-reference":
                        return hashlib.sha256(path.read_bytes()).hexdigest()
                    with path.open("rb") as stream:
                        return hashlib.file_digest(stream, "sha256").hexdigest()

                start = perf_counter()
                result, peak = host_allocation_peak(file_hash)
                records.append(
                    {
                        "case": route,
                        "capacity": mib,
                        "input_bytes": source.nbytes,
                        "scratch_peak_bytes": peak,
                        "seconds": perf_counter() - start,
                        "sha256": result,
                    }
                )
    signature = PoolExecutionSignature(
        topology_id="benchmark",
        method_id="map",
        precision_id="float32",
        backend_id="jax-cpu",
    )
    for buckets in (16, 64, 256) if args.smoke else (64, 256, 1024, 4096):
        plan = ExecutionWorksetPlan(
            tuple(f"item-{i:06d}" for i in range(buckets * 8)),
            (signature,) * (buckets * 8),
        )
        start = perf_counter()
        prepared, peak = host_allocation_peak(plan.prepare)
        output_bytes = sum(
            value.nbytes
            for value in (
                prepared.item_indices,
                prepared.valid_mask,
                prepared.bucket_rng_indices,
                prepared.item_bucket,
                prepared.item_slot,
            )
        )
        records.append(
            {
                "case": "workset",
                "capacity": buckets,
                "input_bytes": plan.semantic_rng_indices.nbytes,
                "output_bytes": output_bytes,
                "scratch_peak_bytes": peak,
                "seconds": perf_counter() - start,
                "prepared_id": prepared.prepared_id,
            }
        )
        indices, valid = (
            np.asarray(prepared.item_indices),
            np.asarray(prepared.valid_mask),
        )
        for route in ("identity-legacy-reference", "identity-stream"):

            def identity_hash() -> str:
                if route == "identity-legacy-reference":
                    return canonical_fingerprint(
                        {
                            "kind": "prepared-fixed-shape-execution-worksets",
                            "plan": plan.plan_id,
                            "bucket_signatures": [
                                value.signature_id for value in prepared.bucket_signatures
                            ],
                            "item_indices": indices.tolist(),
                            "valid_mask": valid.tolist(),
                        }
                    )
                return _prepared_identity(
                    plan.plan_id, prepared.bucket_signatures, indices, valid
                )

            start = perf_counter()
            result, peak = host_allocation_peak(identity_hash)
            if result != prepared.prepared_id:
                raise ValueError("Prepared identity differs from its legacy reference.")
            records.append(
                {
                    "case": route,
                    "capacity": buckets,
                    "input_bytes": indices.nbytes + valid.nbytes,
                    "scratch_peak_bytes": peak,
                    "seconds": perf_counter() - start,
                    "prepared_id": result,
                }
            )
    root = Path(__file__).resolve().parents[1]
    write_json_atomic(
        args.output,
        {
            "identity": capture_benchmark_identity(
                root, Path(__file__), ("scratch_peak_bytes", "input_bytes", "seconds")
            ).to_dict(),
            "environment": capture_environment().to_dict(),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "records": records,
            "scaling_exponents": {
                case: fitted_scaling_exponent(
                    [row["capacity"] for row in records if row["case"] == case],
                    [row["scratch_peak_bytes"] for row in records if row["case"] == case],
                )
                for case in sorted({row["case"] for row in records})
            },
        },
    )


if __name__ == "__main__":
    main()

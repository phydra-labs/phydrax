#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from benchmarks._runtime import (
    capture_environment,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_kernels import (
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _source_kwargs(speeds, cosines, differential):
    payload = TwoBodyDifferentialKernelPlan.canonical_table_bytes(
        speeds, cosines, differential
    )
    digest = hashlib.sha256(payload).hexdigest()
    lineage = ("synthetic-sidm-kernel-benchmark",)
    manifest = ReferenceArtifactManifest(
        "synthetic-sidm-kernel-benchmark",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="unrestricted",
        nondimensionalization={"speed": 1.0, "cross_section": 1.0},
        uncertainty={"tabulation": 0.0},
        lineage_ids=lineage,
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="synthetic-differential-kernel",
        content_digest=digest,
        producer="benchmark-fixture",
        producer_version="native",
        build_id="hand-authored",
        license_id=manifest.license_id,
        parent_artifact_ids=lineage,
        resource_id="synthetic-sidm-kernel-benchmark",
        status="complete",
    )
    return {
        "source_artifact": artifact,
        "reference_manifest": manifest,
        "commercial_use": True,
        "redistribution": False,
        "training_use": False,
        "export": False,
    }


def _setup(node_count: int):
    species = DarkSectorSpeciesPlan("benchmark-chi", 1.0)
    speeds = jnp.asarray((0.25, 1.0, 4.0))
    cosines = jnp.linspace(-1.0, 1.0, node_count)
    forward_shape = 1.0 / (1.05 - cosines) ** 2
    speed_scale = jnp.asarray((0.5, 1.0, 2.0))
    differential = speed_scale[:, None] * forward_shape[None, :]
    kernel = TwoBodyDifferentialKernelPlan(
        species,
        species,
        speeds,
        cosines,
        differential,
        **_source_kwargs(speeds, cosines, differential),
        identical_particle_convention="labeled-full-sphere",
    )
    split = SmallAngleSplitPlan(kernel, 0.8)
    return kernel, split


def run(
    sample_count: int,
    node_count: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    environment = capture_environment().to_dict()
    (kernel, split), setup_seconds = measure_synchronized(lambda: _setup(node_count))
    speed = jnp.asarray(1.5)
    keys = jr.split(jr.key(20260915), sample_count)

    def sample_many(sample_keys):
        return jax.vmap(lambda key: kernel.sample_angles(key, speed))(sample_keys)

    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(sample_many).lower(keys),
        lambda lowered: lowered.compile(),
    )
    samples, execution = measure_repeated(
        lambda: compiled(keys), warmup=warmup, repeats=repeats
    )

    moments = kernel.moments(speed)
    split_evidence = split.moments(speed)
    expected_mean_cosine = 1.0 - moments.transfer / moments.total
    expected_mean_cosine_squared = 1.0 - moments.viscosity / moments.total
    expected_mean_absolute_cosine = 1.0 - moments.modified_transfer / moments.total
    empirical_mean_cosine = jnp.mean(samples.cosine)
    empirical_mean_cosine_squared = jnp.mean(samples.cosine**2)
    empirical_mean_absolute_cosine = jnp.mean(jnp.abs(samples.cosine))
    moment_errors = jnp.asarray(
        (
            jnp.abs(empirical_mean_cosine - expected_mean_cosine),
            jnp.abs(empirical_mean_cosine_squared - expected_mean_cosine_squared),
            jnp.abs(empirical_mean_absolute_cosine - expected_mean_absolute_cosine),
        )
    )
    reconstruction = jnp.asarray(split_evidence.reconstruction_error)
    mean_seconds = execution.mean_seconds
    sampling_tolerance = 8.0 / jnp.sqrt(jnp.asarray(sample_count, dtype=speed.dtype))
    successful = bool(
        jnp.all(samples.supported)
        & split_evidence.successful
        & jnp.all(jnp.isfinite(moment_errors))
        & jnp.all(moment_errors <= sampling_tolerance)
    )

    return {
        "environment": environment,
        "configuration": {
            "sample_count": sample_count,
            "node_count": node_count,
            "warmup": warmup,
            "repeats": repeats,
            "relative_speed": float(speed),
            "kernel_id": kernel.kernel_id,
            "reference_manifest_id": kernel.reference_manifest.manifest_id,
            "requested_use_id": kernel.requested_use_id,
            "split_id": split.split_id,
        },
        "physics": {
            "expected_moments": {
                "mean_cosine": float(expected_mean_cosine),
                "mean_cosine_squared": float(expected_mean_cosine_squared),
                "mean_absolute_cosine": float(expected_mean_absolute_cosine),
            },
            "empirical_moments": {
                "mean_cosine": float(empirical_mean_cosine),
                "mean_cosine_squared": float(empirical_mean_cosine_squared),
                "mean_absolute_cosine": float(empirical_mean_absolute_cosine),
            },
            "absolute_sampling_errors": [float(value) for value in moment_errors],
            "sampling_tolerance": float(sampling_tolerance),
            "split_relative_reconstruction_errors": [
                float(value) for value in reconstruction
            ],
            "maximum_split_relative_reconstruction_error": float(jnp.max(reconstruction)),
            "split_no_gap": bool(split.no_gap),
            "split_no_overlap": bool(split.no_overlap),
            "all_samples_supported": bool(jnp.all(samples.supported)),
            "successful": successful,
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "compilation": asdict(compilation),
            "execution": execution.to_seconds_dict(),
            "samples_per_second": (
                None if mean_seconds in (None, 0.0) else sample_count / mean_seconds
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=131_072)
    parser.add_argument("--nodes", type=int, default=129)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1_024 <= arguments.samples <= 10_000_000:
        raise ValueError("samples must be between 1,024 and 10,000,000.")
    if not 5 <= arguments.nodes <= 4_097:
        raise ValueError("nodes must be between 5 and 4,097.")
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive.")

    payload = run(
        arguments.samples,
        arguments.nodes,
        arguments.warmup,
        arguments.repeats,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()

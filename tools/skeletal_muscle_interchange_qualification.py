#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify strict skeletal-model identity and deterministic execution worksets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.interchange import (
    ExternalModelAsset,
    ExternalModelChannelBinding,
    ExternalModelDescriptor,
    ExternalModelDimensionalContract,
    ExternalModelHostInventory,
    ExternalModelPreparationError,
    ExternalModelQuantity,
    ExternalModelSource,
    ExternalModelTransformation,
    prepare_external_model_descriptor,
)
from phydrax.execution import (
    evaluate_execution_worksets_serial,
    evaluate_execution_worksets_vmap,
    ExecutionWorksetCheckpoint,
    ExecutionWorksetPlan,
    PoolExecutionSignature,
    restore_execution_workset_checkpoint,
)


def _descriptor_and_inventory(compiled: str = "2" * 64):
    asset = "1" * 64
    descriptor = ExternalModelDescriptor(
        source=ExternalModelSource(
            package="qualification-provider",
            revision="1.0+model.1",
            source_uri="https://example.invalid/qualification-provider",
            license_expression="LicenseRef-Qualification-Provider",
            license_uri="https://example.invalid/qualification-license",
            provenance_reference="Qualification provider release manifest, model 1",
            provenance_sha256="3" * 64,
        ),
        assets=(
            ExternalModelAsset(
                "muscle.xml",
                "https://example.invalid/muscle.xml",
                "application/xml",
                asset,
                1024,
            ),
        ),
        transformations=(
            ExternalModelTransformation(
                transformation_name="compile-muscle",
                tool_package="qualification-provider",
                tool_revision="1.0",
                specification_sha256="4" * 64,
                input_sha256=(asset,),
                output_sha256="2" * 64,
            ),
        ),
        dimensions=ExternalModelDimensionalContract(
            (
                ExternalModelQuantity(
                    "musculotendon_length",
                    "coordinate",
                    (0, 1, 0, 0, 0, 0, 0),
                    "cm",
                    "m",
                ),
                ExternalModelQuantity(
                    "independent_excitation",
                    "actuator",
                    (0, 0, 0, 0, 0, 0, 0),
                    "1",
                    "1",
                ),
                ExternalModelQuantity(
                    "raw_provider_force",
                    "sensor",
                    (1, 1, -2, 0, 0, 0, 0),
                    "N",
                    "N",
                ),
            ),
            spatial_axes=("musculotendon-line-of-action",),
            support="one lumped musculotendon actuator",
            reference="SI Brochure, 9th edition (2019), sections 2.3 and 2.4",
        ),
        provider_versions=(("qualification-provider", "1.0"),),
        compiled_sha256="2" * 64,
        coordinate_map=(
            ExternalModelChannelBinding(
                "length_cm", "musculotendon_length", "musculotendon_length", scale=0.01
            ),
        ),
        actuator_map=(
            ExternalModelChannelBinding(
                "independent_excitation",
                "excitation",
                "independent_excitation",
                scale=1.0,
            ),
        ),
        sensor_map=(
            ExternalModelChannelBinding(
                "signed_force",
                "raw_provider_force",
                "raw_provider_force",
                scale=-1.0,
            ),
        ),
        force_owner="provider-native",
    )
    inventory = ExternalModelHostInventory(
        source_package="qualification-provider",
        source_revision="1.0+model.1",
        asset_hashes=(("muscle.xml", asset),),
        provider_versions=(("qualification-provider", "1.0"),),
        compiled_sha256=compiled,
        coordinate_channels=("length_cm",),
        actuator_channels=("excitation",),
        sensor_channels=("signed_force",),
    )
    return descriptor, inventory


def _operation(signature, item, key, semantic_index):
    factor = 2.0 if signature.topology_id == "fast" else 3.0
    random_value = jax.random.uniform(key, item.shape, dtype=item.dtype)
    return factor * item + random_value + 0.0 * semantic_index.astype(item.dtype)


def qualification() -> dict[str, object]:
    descriptor, inventory = _descriptor_and_inventory()
    prepared_descriptor = prepare_external_model_descriptor(descriptor, inventory)
    mapped_length = prepared_descriptor.coordinate_to_phydrax(jnp.asarray([125.0]))
    mapped_force = prepared_descriptor.sensor_to_phydrax(jnp.asarray([-42.0]))
    _, mismatched_inventory = _descriptor_and_inventory("5" * 64)
    rejected = False
    failure_reasons: tuple[str, ...] = ()
    try:
        prepare_external_model_descriptor(descriptor, mismatched_inventory)
    except ExternalModelPreparationError as error:
        rejected = True
        failure_reasons = error.evidence.failure_reasons

    fast = PoolExecutionSignature(
        topology_id="fast",
        method_id="qualification-map",
        precision_id="float32",
        backend_id=jax.default_backend(),
    )
    slow = PoolExecutionSignature(
        topology_id="slow",
        method_id="qualification-map",
        precision_id="float32",
        backend_id=jax.default_backend(),
    )
    semantic_ids = tuple(f"unit-{index}" for index in range(9))
    signatures = tuple(fast if index % 2 else slow for index in range(9))
    first = ExecutionWorksetPlan(
        tuple(reversed(semantic_ids)), tuple(reversed(signatures)), bucket_capacity=4
    ).prepare()
    second = ExecutionWorksetPlan(semantic_ids, signatures, bucket_capacity=7).prepare()
    values = jnp.arange(27, dtype=jnp.float32).reshape((9, 3)) / 11.0
    counters = jnp.arange(9, dtype=jnp.uint32)
    root_key = jax.random.key(104729)
    serial = evaluate_execution_worksets_serial(
        first, _operation, values, root_key, counters
    )
    vectorized = evaluate_execution_worksets_vmap(
        first, _operation, values, root_key, counters
    )
    first_key_data = first.scatter(
        jax.random.key_data(first.semantic_keys(root_key, counters))
    )
    second_key_data = second.scatter(
        jax.random.key_data(second.semantic_keys(root_key, counters))
    )
    checkpoint = ExecutionWorksetCheckpoint(
        first, vectorized.values, vectorized.next_rng_counters
    )
    restarted, restarted_counters = restore_execution_workset_checkpoint(first, checkpoint)
    scatter_error = float(jnp.max(jnp.abs(first.scatter(first.gather(values)) - values)))
    serial_vmap_error = float(jnp.max(jnp.abs(serial.values - vectorized.values)))
    restart_error = float(jnp.max(jnp.abs(restarted - vectorized.values)))
    key_capacity_invariant = bool(jnp.array_equal(first_key_data, second_key_data))
    local_devices = tuple(str(device) for device in jax.local_devices())
    passed = bool(
        prepared_descriptor.evidence.successful
        and rejected
        and failure_reasons
        == ("compiled SHA-256 digest does not match the descriptor",)
        and jnp.allclose(mapped_length, 1.25)
        and jnp.allclose(mapped_force, 42.0)
        and scatter_error == 0.0
        and serial_vmap_error == 0.0
        and restart_error == 0.0
        and jnp.array_equal(restarted_counters, vectorized.next_rng_counters)
        and key_capacity_invariant
        and bool(serial.evidence.successful)
        and bool(vectorized.evidence.successful)
    )
    return {
        "descriptor": {
            "descriptor_id": descriptor.descriptor_id,
            "prepared_id": prepared_descriptor.prepared_id,
            "identity_mismatch_rejected": rejected,
            "identity_failure_reasons": failure_reasons,
            "mapped_length_m": float(mapped_length[0]),
            "mapped_raw_provider_force_N": float(mapped_force[0]),
            "force_owner": descriptor.force_owner,
        },
        "worksets": {
            "plan_id": first.plan.plan_id,
            "prepared_id": first.prepared_id,
            "bucket_count": first.bucket_count,
            "bucket_capacity": first.bucket_capacity,
            "padded_lane_count": int(serial.evidence.padded_lane_count),
            "gather_scatter_maximum_absolute_error": scatter_error,
            "serial_vmap_maximum_absolute_error": serial_vmap_error,
            "checkpoint_restart_maximum_absolute_error": restart_error,
            "semantic_keys_capacity_invariant": key_capacity_invariant,
            "checkpoint_id": checkpoint.checkpoint_id,
        },
        "distributed_gate": {
            "released": False,
            "local_device_count": jax.local_device_count(),
            "local_devices": local_devices,
            "reason": (
                "No distributed API is released without at least two local JAX devices; "
                "emulation is intentionally absent."
            ),
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = qualification()
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

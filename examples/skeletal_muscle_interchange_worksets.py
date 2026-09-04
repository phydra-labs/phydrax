#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Verify an external model identity and execute deterministic motor-unit worksets."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.interchange import (
    ExternalModelAsset,
    ExternalModelChannelBinding,
    ExternalModelDescriptor,
    ExternalModelDimensionalContract,
    ExternalModelHostInventory,
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


def verified_external_model():
    asset = "1" * 64
    compiled = "2" * 64
    descriptor = ExternalModelDescriptor(
        source=ExternalModelSource(
            package="example-provider",
            revision="2.0+model.4",
            source_uri="https://example.invalid/example-provider",
            license_expression="LicenseRef-Example-Provider",
            license_uri="https://example.invalid/example-license",
            provenance_reference="Example provider release manifest, model 4",
            provenance_sha256="3" * 64,
        ),
        assets=(
            ExternalModelAsset(
                "muscle.xml",
                "https://example.invalid/muscle.xml",
                "application/xml",
                asset,
                4096,
            ),
        ),
        transformations=(
            ExternalModelTransformation(
                transformation_name="compile-example-muscle",
                tool_package="example-provider",
                tool_revision="2.0",
                specification_sha256="4" * 64,
                input_sha256=(asset,),
                output_sha256=compiled,
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
        provider_versions=(("example-provider", "2.0"),),
        compiled_sha256=compiled,
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
    observed = ExternalModelHostInventory(
        source_package="example-provider",
        source_revision="2.0+model.4",
        asset_hashes=(("muscle.xml", asset),),
        provider_versions=(("example-provider", "2.0"),),
        compiled_sha256=compiled,
        coordinate_channels=("length_cm",),
        actuator_channels=("excitation",),
        sensor_channels=("signed_force",),
    )
    return prepare_external_model_descriptor(descriptor, observed)


def main() -> None:
    external = verified_external_model()
    signature = PoolExecutionSignature(
        topology_id=external.prepared_id,
        method_id="provider-sensor-map",
        precision_id="float32",
        backend_id=jax.default_backend(),
    )
    worksets = ExecutionWorksetPlan(
        ("motor-unit-3", "motor-unit-1", "motor-unit-2", "motor-unit-0"),
        (signature,) * 4,
        bucket_capacity=3,
    ).prepare()
    provider_force_by_id = {
        "motor-unit-0": -60.0,
        "motor-unit-1": -80.0,
        "motor-unit-2": -100.0,
        "motor-unit-3": -120.0,
    }
    signed_provider_forces = jnp.asarray(
        [[provider_force_by_id[item]] for item in worksets.plan.semantic_ids]
    )

    def map_force(signature_, value, key, semantic_index):
        del signature_, key, semantic_index
        return external.sensor_to_phydrax(value)

    counters = jnp.zeros((4,), dtype=jnp.uint32)
    key = jax.random.key(7)
    serial = evaluate_execution_worksets_serial(
        worksets, map_force, signed_provider_forces, key, counters
    )
    vectorized = evaluate_execution_worksets_vmap(
        worksets, map_force, signed_provider_forces, key, counters
    )
    if not jnp.array_equal(serial.values, vectorized.values):
        raise RuntimeError("Serial and vmap workset routes differ.")
    checkpoint = ExecutionWorksetCheckpoint(
        worksets, vectorized.values, vectorized.next_rng_counters
    )
    restarted, restarted_counters = restore_execution_workset_checkpoint(
        worksets, checkpoint
    )
    print("descriptor:", external.descriptor.descriptor_id)
    print("prepared:", external.prepared_id)
    print("force owner:", external.descriptor.force_owner)
    print("canonical IDs:", worksets.plan.semantic_ids)
    print("mapped forces [N]:", restarted[:, 0])
    print("restart RNG counters:", restarted_counters)
    print("checkpoint:", checkpoint.checkpoint_id)
    print("distributed API released: False (no emulated fallback)")


if __name__ == "__main__":
    main()

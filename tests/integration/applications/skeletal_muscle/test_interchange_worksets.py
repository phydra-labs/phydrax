#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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


def test_verified_signed_provider_force_routes_identically_through_worksets() -> None:
    asset = "1" * 64
    compiled = "2" * 64
    source = ExternalModelSource(
        package="provider-runtime",
        revision="5.0+muscle.9",
        source_uri="https://example.invalid/provider-runtime",
        license_expression="LicenseRef-Provider-Runtime",
        license_uri="https://example.invalid/provider-license",
        provenance_reference="Provider runtime release manifest, muscle 9",
        provenance_sha256="3" * 64,
    )
    dimensions = ExternalModelDimensionalContract(
        (
            ExternalModelQuantity(
                "musculotendon_length", "coordinate", (0, 1, 0, 0, 0, 0, 0), "m", "m"
            ),
            ExternalModelQuantity(
                "independent_excitation", "actuator", (0, 0, 0, 0, 0, 0, 0), "1", "1"
            ),
            ExternalModelQuantity(
                "raw_provider_force", "sensor", (1, 1, -2, 0, 0, 0, 0), "N", "N"
            ),
        ),
        spatial_axes=("musculotendon-line-of-action",),
        support="one lumped musculotendon actuator",
        reference="SI Brochure, 9th edition (2019), sections 2.3 and 2.4",
    )
    descriptor = ExternalModelDescriptor(
        source=source,
        assets=(
            ExternalModelAsset(
                "provider-model.xml",
                "https://example.invalid/provider-model.xml",
                "application/xml",
                asset,
                4096,
            ),
        ),
        transformations=(
            ExternalModelTransformation(
                transformation_name="provider-compile",
                tool_package="provider-runtime",
                tool_revision="5.0",
                specification_sha256="4" * 64,
                input_sha256=(asset,),
                output_sha256=compiled,
            ),
        ),
        dimensions=dimensions,
        provider_versions=(("provider-runtime", "5.0"),),
        compiled_sha256=compiled,
        coordinate_map=(
            ExternalModelChannelBinding(
                "length", "musculotendon_length", "musculotendon_length", scale=1.0
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
                "actuator_force",
                "raw_provider_force",
                "raw_provider_force",
                scale=-1.0,
            ),
        ),
        force_owner="provider-native",
    )
    inventory = ExternalModelHostInventory(
        source_package="provider-runtime",
        source_revision="5.0+muscle.9",
        asset_hashes=(("provider-model.xml", asset),),
        provider_versions=(("provider-runtime", "5.0"),),
        compiled_sha256=compiled,
        coordinate_channels=("length",),
        actuator_channels=("excitation",),
        sensor_channels=("actuator_force",),
    )
    prepared_descriptor = prepare_external_model_descriptor(descriptor, inventory)
    signature = PoolExecutionSignature(
        topology_id=prepared_descriptor.prepared_id,
        method_id="provider-sensor-map",
        precision_id="float32",
        backend_id="jax",
    )
    worksets = ExecutionWorksetPlan(
        ("motor-unit-2", "motor-unit-0", "motor-unit-1"),
        (signature, signature, signature),
        bucket_capacity=2,
    ).prepare()
    signed_provider_force = jnp.asarray([[-10.0], [-20.0], [-30.0]])

    def route_force(signature_, item, key, semantic_index):
        del signature_, key, semantic_index
        return prepared_descriptor.sensor_to_phydrax(item)

    counters = jnp.zeros((3,), dtype=jnp.uint32)
    root_key = jax.random.key(0)
    serial = evaluate_execution_worksets_serial(
        worksets, route_force, signed_provider_force, root_key, counters
    )
    vectorized = evaluate_execution_worksets_vmap(
        worksets, route_force, signed_provider_force, root_key, counters
    )
    assert jnp.array_equal(serial.values, vectorized.values)
    assert jnp.array_equal(vectorized.values[:, 0], jnp.asarray([10.0, 20.0, 30.0]))
    checkpoint = ExecutionWorksetCheckpoint(
        worksets, vectorized.values, vectorized.next_rng_counters
    )
    restarted, restarted_counters = restore_execution_workset_checkpoint(
        worksets, checkpoint
    )
    assert jnp.array_equal(restarted, vectorized.values)
    assert jnp.array_equal(restarted_counters, vectorized.next_rng_counters)

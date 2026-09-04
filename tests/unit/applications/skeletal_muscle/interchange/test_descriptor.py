#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

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


_ASSET = "a" * 64
_SPECIFICATION = "b" * 64
_COMPILED = "c" * 64
_PROVENANCE = "d" * 64


def _descriptor(*, force_owner: str = "provider-native") -> ExternalModelDescriptor:
    source = ExternalModelSource(
        package="qualified-provider",
        revision="1.4.2+model.7",
        source_uri="https://example.invalid/qualified-provider",
        license_expression="LicenseRef-Qualified-Provider",
        license_uri="https://example.invalid/license",
        provenance_reference="Qualified provider release manifest, model 7",
        provenance_sha256=_PROVENANCE,
    )
    dimensions = ExternalModelDimensionalContract(
        (
            ExternalModelQuantity(
                "musculotendon_length", "coordinate", (0, 1, 0, 0, 0, 0, 0), "cm", "m"
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
    return ExternalModelDescriptor(
        source=source,
        assets=(
            ExternalModelAsset(
                "model.xml",
                "https://example.invalid/model.xml",
                "application/xml",
                _ASSET,
                2048,
            ),
        ),
        transformations=(
            ExternalModelTransformation(
                transformation_name="compile-provider-model",
                tool_package="qualified-compiler",
                tool_revision="3.1.0",
                specification_sha256=_SPECIFICATION,
                input_sha256=(_ASSET,),
                output_sha256=_COMPILED,
            ),
        ),
        dimensions=dimensions,
        provider_versions=(("qualified-compiler", "3.1.0"), ("qualified-provider", "1.4.2")),
        compiled_sha256=_COMPILED,
        coordinate_map=(
            ExternalModelChannelBinding(
                "q_length_cm",
                "musculotendon_length",
                "musculotendon_length",
                scale=0.01,
            ),
        ),
        actuator_map=(
            ExternalModelChannelBinding(
                "independent_excitation",
                "ctrl_excitation",
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
        force_owner=force_owner,
    )


def _inventory(
    *,
    compiled_sha256: str = _COMPILED,
    coordinate_channels: tuple[str, ...] = ("q_length_cm",),
) -> ExternalModelHostInventory:
    return ExternalModelHostInventory(
        source_package="qualified-provider",
        source_revision="1.4.2+model.7",
        asset_hashes=(("model.xml", _ASSET),),
        provider_versions=(("qualified-provider", "1.4.2"), ("qualified-compiler", "3.1.0")),
        compiled_sha256=compiled_sha256,
        coordinate_channels=coordinate_channels,
        actuator_channels=("ctrl_excitation",),
        sensor_channels=("actuator_force",),
    )


def test_descriptor_is_immutable_and_force_owner_is_identity() -> None:
    provider = _descriptor()
    native = _descriptor(force_owner="de-groote")
    assert provider.descriptor_id != native.descriptor_id
    with pytest.raises(AttributeError):
        provider.force_owner = "de-groote"  # type: ignore[misc]
    with pytest.raises(ValueError, match="atomic"):
        _descriptor(force_owner="provider-native+de-groote")


def test_host_preparation_verifies_identity_and_lowers_maps() -> None:
    prepared = prepare_external_model_descriptor(_descriptor(), _inventory())
    assert prepared.evidence.successful
    assert jnp.allclose(prepared.coordinate_to_phydrax(jnp.asarray([125.0])), 1.25)
    assert jnp.allclose(prepared.actuator_to_external(jnp.asarray([0.4])), 0.4)
    assert jnp.allclose(prepared.sensor_to_phydrax(jnp.asarray([[-12.0]])), 12.0)
    leaves = jax.tree_util.tree_leaves(prepared)
    assert any(leaf is prepared.sensor_scale for leaf in leaves)


def test_host_preparation_rejects_compiled_identity_with_evidence() -> None:
    with pytest.raises(ExternalModelPreparationError) as raised:
        prepare_external_model_descriptor(_descriptor(), _inventory(compiled_sha256="e" * 64))
    assert not raised.value.evidence.successful
    assert raised.value.evidence.descriptor_id == _descriptor().descriptor_id
    assert raised.value.evidence.failure_reasons == (
        "compiled SHA-256 digest does not match the descriptor",
    )


def test_host_inventory_rejects_incomplete_channel_identity() -> None:
    with pytest.raises(ValueError, match="coordinate_channels"):
        _inventory(coordinate_channels=())


def test_descriptor_rejects_a_broken_compile_chain() -> None:
    descriptor = _descriptor()
    with pytest.raises(ValueError, match="ordered chain"):
        ExternalModelDescriptor(
            source=descriptor.source,
            assets=descriptor.assets,
            transformations=(
                ExternalModelTransformation(
                    transformation_name="unrelated-output",
                    tool_package="qualified-compiler",
                    tool_revision="3.1.0",
                    specification_sha256=_SPECIFICATION,
                    input_sha256=("f" * 64,),
                    output_sha256=_COMPILED,
                ),
            ),
            dimensions=descriptor.dimensions,
            provider_versions=descriptor.provider_versions,
            compiled_sha256=_COMPILED,
            coordinate_map=descriptor.coordinate_map,
            actuator_map=descriptor.actuator_map,
            sensor_map=descriptor.sensor_map,
            force_owner=descriptor.force_owner,
        )


def test_prepared_affine_maps_are_jittable() -> None:
    prepared = prepare_external_model_descriptor(_descriptor(), _inventory())
    mapped = eqx.filter_jit(prepared.coordinate_to_phydrax)(jnp.asarray([[100.0], [150.0]]))
    assert jnp.allclose(mapped[:, 0], jnp.asarray([1.0, 1.5]))

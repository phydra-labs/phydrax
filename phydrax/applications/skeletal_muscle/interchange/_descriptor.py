#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


ExternalChannelRole = Literal["coordinate", "actuator", "sensor"]
_SI_DIMENSION_COUNT = 7
_FORCE_OWNER_PUNCTUATION = frozenset("-._:")


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _atomic_owner(value: str, /) -> str:
    owner = _identifier(value, "force_owner")
    allowed = _FORCE_OWNER_PUNCTUATION
    if not owner[0].isalnum() or any(
        not (character.isalnum() or character in allowed) for character in owner
    ):
        raise ValueError(
            "force_owner must be one atomic alphanumeric, hyphenated, or namespaced ID."
        )
    return owner


def _sha256(value: str, name: str, /) -> str:
    digest = _identifier(value, name).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest.")
    return digest


def _version_pairs(
    values: tuple[tuple[str, str], ...], name: str, /
) -> tuple[tuple[str, str], ...]:
    normalized = tuple(
        (_identifier(package, f"{name} package"), _identifier(version, f"{name} version"))
        for package, version in values
    )
    if not normalized:
        raise ValueError(f"{name} must contain at least one exact package version.")
    if len({package for package, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} package names must be unique.")
    return tuple(sorted(normalized))


class ExternalModelSource(StrictModule, NonTrainableState):
    """Immutable package, revision, license, and provenance identity."""

    package: str = eqx.field(static=True)
    revision: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    license_expression: str = eqx.field(static=True)
    license_uri: str = eqx.field(static=True)
    provenance_reference: str = eqx.field(static=True)
    provenance_sha256: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        package: str,
        revision: str,
        source_uri: str,
        license_expression: str,
        license_uri: str,
        provenance_reference: str,
        provenance_sha256: str,
    ):
        values = tuple(
            _identifier(value, name)
            for value, name in (
                (package, "package"),
                (revision, "revision"),
                (source_uri, "source_uri"),
                (license_expression, "license_expression"),
                (license_uri, "license_uri"),
                (provenance_reference, "provenance_reference"),
            )
        )
        provenance_digest = _sha256(provenance_sha256, "provenance_sha256")
        (
            self.package,
            self.revision,
            self.source_uri,
            self.license_expression,
            self.license_uri,
            self.provenance_reference,
        ) = values
        self.provenance_sha256 = provenance_digest
        self.source_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-source",
                "package": values[0],
                "revision": values[1],
                "source_uri": values[2],
                "license_expression": values[3],
                "license_uri": values[4],
                "provenance_reference": values[5],
                "provenance_sha256": provenance_digest,
            }
        )


class ExternalModelAsset(StrictModule, NonTrainableState):
    """One immutable external asset and its independently checkable content hash."""

    asset_name: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    media_type: str = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)

    def __init__(
        self,
        asset_name: str,
        source_uri: str,
        media_type: str,
        sha256: str,
        byte_count: int,
        /,
    ):
        name = _identifier(asset_name, "asset_name")
        uri = _identifier(source_uri, "asset source_uri")
        media = _identifier(media_type, "asset media_type")
        digest = _sha256(sha256, "asset sha256")
        size = int(byte_count)
        if size < 1:
            raise ValueError("asset byte_count must be positive.")
        self.asset_name = name
        self.source_uri = uri
        self.media_type = media
        self.sha256 = digest
        self.byte_count = size
        self.asset_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-asset",
                "asset_name": name,
                "source_uri": uri,
                "media_type": media,
                "sha256": digest,
                "byte_count": size,
            }
        )


class ExternalModelTransformation(StrictModule, NonTrainableState):
    """One ordered, content-addressed host transformation in the compile chain."""

    transformation_name: str = eqx.field(static=True)
    tool_package: str = eqx.field(static=True)
    tool_revision: str = eqx.field(static=True)
    specification_sha256: str = eqx.field(static=True)
    input_sha256: tuple[str, ...] = eqx.field(static=True)
    output_sha256: str = eqx.field(static=True)
    transformation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        transformation_name: str,
        tool_package: str,
        tool_revision: str,
        specification_sha256: str,
        input_sha256: tuple[str, ...],
        output_sha256: str,
    ):
        name = _identifier(transformation_name, "transformation_name")
        package = _identifier(tool_package, "tool_package")
        revision = _identifier(tool_revision, "tool_revision")
        specification = _sha256(specification_sha256, "specification_sha256")
        inputs = tuple(_sha256(value, "input_sha256") for value in input_sha256)
        if not inputs or len(set(inputs)) != len(inputs):
            raise ValueError("Transformation input digests must be non-empty and unique.")
        output = _sha256(output_sha256, "output_sha256")
        self.transformation_name = name
        self.tool_package = package
        self.tool_revision = revision
        self.specification_sha256 = specification
        self.input_sha256 = inputs
        self.output_sha256 = output
        self.transformation_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-transformation",
                "name": name,
                "tool_package": package,
                "tool_revision": revision,
                "specification_sha256": specification,
                "input_sha256": list(inputs),
                "output_sha256": output,
            }
        )


class ExternalModelQuantity(StrictModule, NonTrainableState):
    """One channel quantity with an explicit SI-dimension and unit contract."""

    quantity_name: str = eqx.field(static=True)
    role: ExternalChannelRole = eqx.field(static=True)
    si_dimensions: tuple[int, int, int, int, int, int, int] = eqx.field(static=True)
    external_unit: str = eqx.field(static=True)
    phydrax_unit: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantity_name: str,
        role: ExternalChannelRole,
        si_dimensions: tuple[int, int, int, int, int, int, int],
        external_unit: str,
        phydrax_unit: str,
        /,
    ):
        name = _identifier(quantity_name, "quantity_name")
        if role not in ("coordinate", "actuator", "sensor"):
            raise ValueError("role must be coordinate, actuator, or sensor.")
        dimensions_ = tuple(si_dimensions)
        if len(dimensions_) != _SI_DIMENSION_COUNT or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in dimensions_
        ):
            raise ValueError(
                "si_dimensions must contain seven integer SI base exponents in "
                "(mass, length, time, current, temperature, amount, luminous intensity) order."
            )
        dimensions = tuple(int(value) for value in dimensions_)
        source_unit = _identifier(external_unit, "external_unit")
        target_unit = _identifier(phydrax_unit, "phydrax_unit")
        self.quantity_name = name
        self.role = role
        self.si_dimensions = dimensions  # type: ignore[assignment]
        self.external_unit = source_unit
        self.phydrax_unit = target_unit
        self.quantity_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-quantity",
                "quantity_name": name,
                "role": role,
                "si_dimensions": list(dimensions),
                "external_unit": source_unit,
                "phydrax_unit": target_unit,
            }
        )


class ExternalModelDimensionalContract(StrictModule, NonTrainableState):
    """Complete mapped-channel dimensions, axes, support, and reference convention."""

    quantities: tuple[ExternalModelQuantity, ...]
    spatial_axes: tuple[str, ...] = eqx.field(static=True)
    support: str = eqx.field(static=True)
    reference: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantities: tuple[ExternalModelQuantity, ...],
        /,
        *,
        spatial_axes: tuple[str, ...],
        support: str,
        reference: str,
    ):
        values = tuple(quantities)
        if not values or not all(isinstance(value, ExternalModelQuantity) for value in values):
            raise TypeError(
                "quantities must contain at least one ExternalModelQuantity."
            )
        names = tuple(value.quantity_name for value in values)
        if len(set(names)) != len(names):
            raise ValueError("Dimensional-contract quantity names must be unique.")
        axes = tuple(_identifier(value, "spatial axis") for value in spatial_axes)
        if not axes or len(set(axes)) != len(axes):
            raise ValueError("spatial_axes must be non-empty and unique.")
        support_value = _identifier(support, "support")
        reference_value = _identifier(reference, "reference")
        ordered = tuple(sorted(values, key=lambda value: value.quantity_name))
        self.quantities = ordered
        self.spatial_axes = axes
        self.support = support_value
        self.reference = reference_value
        self.contract_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-dimensional-contract",
                "quantities": [value.quantity_id for value in ordered],
                "spatial_axes": list(axes),
                "support": support_value,
                "reference": reference_value,
            }
        )

    def quantity(self, name: str, /) -> ExternalModelQuantity:
        identifier = _identifier(name, "quantity name")
        matches = tuple(value for value in self.quantities if value.quantity_name == identifier)
        if len(matches) != 1:
            raise KeyError(f"Unknown dimensional-contract quantity {identifier!r}.")
        return matches[0]


class ExternalModelChannelBinding(StrictModule, NonTrainableState):
    """Named affine channel map ``target = scale * source + offset``."""

    source_name: str = eqx.field(static=True)
    target_name: str = eqx.field(static=True)
    quantity_name: str = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    offset: float = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_name: str,
        target_name: str,
        quantity_name: str,
        /,
        *,
        scale: float,
        offset: float = 0.0,
    ):
        source = _identifier(source_name, "channel source_name")
        target = _identifier(target_name, "channel target_name")
        quantity = _identifier(quantity_name, "channel quantity_name")
        scale_value = float(scale)
        offset_value = float(offset)
        if not np.isfinite(scale_value) or scale_value == 0.0 or not np.isfinite(offset_value):
            raise ValueError("Channel scale must be finite and nonzero; offset must be finite.")
        self.source_name = source
        self.target_name = target
        self.quantity_name = quantity
        self.scale = scale_value
        self.offset = offset_value
        self.binding_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-channel-binding",
                "source": source,
                "target": target,
                "quantity": quantity,
                "scale": scale_value,
                "offset": offset_value,
            }
        )


class ExternalModelDescriptor(StrictModule, NonTrainableState):
    """Complete immutable identity and force ownership for one external model."""

    source: ExternalModelSource
    assets: tuple[ExternalModelAsset, ...]
    transformations: tuple[ExternalModelTransformation, ...]
    dimensions: ExternalModelDimensionalContract
    provider_versions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    compiled_sha256: str = eqx.field(static=True)
    coordinate_map: tuple[ExternalModelChannelBinding, ...]
    actuator_map: tuple[ExternalModelChannelBinding, ...]
    sensor_map: tuple[ExternalModelChannelBinding, ...]
    force_owner: str = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: ExternalModelSource,
        assets: tuple[ExternalModelAsset, ...],
        transformations: tuple[ExternalModelTransformation, ...],
        dimensions: ExternalModelDimensionalContract,
        provider_versions: tuple[tuple[str, str], ...],
        compiled_sha256: str,
        coordinate_map: tuple[ExternalModelChannelBinding, ...],
        actuator_map: tuple[ExternalModelChannelBinding, ...],
        sensor_map: tuple[ExternalModelChannelBinding, ...],
        force_owner: str,
    ):
        if not isinstance(source, ExternalModelSource):
            raise TypeError("source must be ExternalModelSource.")
        asset_values = tuple(assets)
        if not asset_values or not all(
            isinstance(value, ExternalModelAsset) for value in asset_values
        ):
            raise TypeError("assets must contain at least one ExternalModelAsset.")
        asset_names = tuple(value.asset_name for value in asset_values)
        if len(set(asset_names)) != len(asset_names):
            raise ValueError("External asset names must be unique.")
        transformations_ = tuple(transformations)
        if not transformations_ or not all(
            isinstance(value, ExternalModelTransformation) for value in transformations_
        ):
            raise TypeError(
                "transformations must contain the explicit ordered compile chain."
            )
        if not isinstance(dimensions, ExternalModelDimensionalContract):
            raise TypeError("dimensions must be ExternalModelDimensionalContract.")
        versions = _version_pairs(provider_versions, "provider_versions")
        compiled = _sha256(compiled_sha256, "compiled_sha256")
        coordinate = self._mapping(coordinate_map, "coordinate")
        actuator = self._mapping(actuator_map, "actuator")
        sensor = self._mapping(sensor_map, "sensor")
        owner = _atomic_owner(force_owner)
        quantities = {value.quantity_name: value for value in dimensions.quantities}
        mapped_quantities: set[str] = set()
        for role, bindings in (
            ("coordinate", coordinate),
            ("actuator", actuator),
            ("sensor", sensor),
        ):
            for binding in bindings:
                if binding.quantity_name not in quantities:
                    raise ValueError(
                        f"{role} binding references unknown quantity "
                        f"{binding.quantity_name!r}."
                    )
                if quantities[binding.quantity_name].role != role:
                    raise ValueError(
                        f"{role} binding quantity {binding.quantity_name!r} has another role."
                    )
                mapped_quantities.add(binding.quantity_name)
        if mapped_quantities != set(quantities):
            raise ValueError("Every dimensional-contract quantity must be mapped.")
        available_digests = {value.sha256 for value in asset_values}
        used_assets: set[str] = set()
        previous_output: str | None = None
        for transformation in transformations_:
            inputs = set(transformation.input_sha256)
            allowed = available_digests | ({previous_output} if previous_output else set())
            if not inputs <= allowed or (previous_output is not None and previous_output not in inputs):
                raise ValueError(
                    "Transformation inputs must form one ordered chain from declared assets."
                )
            used_assets.update(inputs & available_digests)
            previous_output = transformation.output_sha256
        if used_assets != available_digests:
            raise ValueError("Every declared asset must participate in the transformation chain.")
        if previous_output != compiled:
            raise ValueError("Final transformation output must equal compiled_sha256.")
        ordered_assets = tuple(sorted(asset_values, key=lambda value: value.asset_name))
        self.source = source
        self.assets = ordered_assets
        self.transformations = transformations_
        self.dimensions = dimensions
        self.provider_versions = versions
        self.compiled_sha256 = compiled
        self.coordinate_map = coordinate
        self.actuator_map = actuator
        self.sensor_map = sensor
        self.force_owner = owner
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-descriptor",
                "source": source.source_id,
                "assets": [value.asset_id for value in ordered_assets],
                "transformations": [
                    value.transformation_id for value in transformations_
                ],
                "dimensions": dimensions.contract_id,
                "provider_versions": [list(value) for value in versions],
                "compiled_sha256": compiled,
                "coordinate_map": [value.binding_id for value in coordinate],
                "actuator_map": [value.binding_id for value in actuator],
                "sensor_map": [value.binding_id for value in sensor],
                "force_owner": owner,
            }
        )

    @staticmethod
    def _mapping(
        values: tuple[ExternalModelChannelBinding, ...], role: ExternalChannelRole, /
    ) -> tuple[ExternalModelChannelBinding, ...]:
        bindings = tuple(values)
        if not bindings or not all(
            isinstance(value, ExternalModelChannelBinding) for value in bindings
        ):
            raise TypeError(f"{role}_map must contain at least one channel binding.")
        sources = tuple(value.source_name for value in bindings)
        targets = tuple(value.target_name for value in bindings)
        if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError(f"{role}_map source and target names must be unique.")
        key = (
            (lambda value: value.source_name)
            if role == "actuator"
            else (lambda value: value.target_name)
        )
        return tuple(sorted(bindings, key=key))


class ExternalModelHostInventory(StrictModule, NonTrainableState):
    """Exact identity and external channel names observed by a host adapter."""

    source_package: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    asset_hashes: tuple[tuple[str, str], ...] = eqx.field(static=True)
    provider_versions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    compiled_sha256: str = eqx.field(static=True)
    coordinate_channels: tuple[str, ...] = eqx.field(static=True)
    actuator_channels: tuple[str, ...] = eqx.field(static=True)
    sensor_channels: tuple[str, ...] = eqx.field(static=True)
    inventory_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_package: str,
        source_revision: str,
        asset_hashes: tuple[tuple[str, str], ...],
        provider_versions: tuple[tuple[str, str], ...],
        compiled_sha256: str,
        coordinate_channels: tuple[str, ...],
        actuator_channels: tuple[str, ...],
        sensor_channels: tuple[str, ...],
    ):
        package = _identifier(source_package, "source_package")
        revision = _identifier(source_revision, "source_revision")
        hashes = tuple(
            (_identifier(name, "asset hash name"), _sha256(digest, "asset hash"))
            for name, digest in asset_hashes
        )
        if not hashes or len({name for name, _ in hashes}) != len(hashes):
            raise ValueError("asset_hashes must be non-empty with unique names.")
        versions = _version_pairs(provider_versions, "provider_versions")
        compiled = _sha256(compiled_sha256, "compiled_sha256")
        channels = []
        for values, name in (
            (coordinate_channels, "coordinate_channels"),
            (actuator_channels, "actuator_channels"),
            (sensor_channels, "sensor_channels"),
        ):
            normalized = tuple(_identifier(value, name) for value in values)
            if not normalized or len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} must be non-empty and unique.")
            channels.append(normalized)
        self.source_package = package
        self.source_revision = revision
        self.asset_hashes = tuple(sorted(hashes))
        self.provider_versions = versions
        self.compiled_sha256 = compiled
        self.coordinate_channels = channels[0]
        self.actuator_channels = channels[1]
        self.sensor_channels = channels[2]
        self.inventory_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-host-inventory",
                "source_package": package,
                "source_revision": revision,
                "asset_hashes": [list(value) for value in sorted(hashes)],
                "provider_versions": [list(value) for value in versions],
                "compiled_sha256": compiled,
                "coordinate_channels": list(channels[0]),
                "actuator_channels": list(channels[1]),
                "sensor_channels": list(channels[2]),
            }
        )


class ExternalModelPreparationEvidence(StrictModule, NonTrainableState):
    """Host-only identity and map resolution evidence, including exact failures."""

    descriptor_id: str = eqx.field(static=True)
    inventory_id: str = eqx.field(static=True)
    failure_reasons: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        descriptor_id: str,
        inventory_id: str,
        failure_reasons: tuple[str, ...],
        /,
    ):
        failures = tuple(str(value) for value in failure_reasons)
        self.descriptor_id = descriptor_id
        self.inventory_id = inventory_id
        self.failure_reasons = failures
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "external-skeletal-model-preparation-evidence",
                "descriptor": descriptor_id,
                "inventory": inventory_id,
                "failure_reasons": list(failures),
            }
        )

    @property
    def successful(self) -> bool:
        return not self.failure_reasons


class ExternalModelPreparationError(ValueError):
    """Fail-closed preparation error retaining machine-readable identity evidence."""

    evidence: ExternalModelPreparationEvidence

    def __init__(self, evidence: ExternalModelPreparationEvidence, /):
        self.evidence = evidence
        super().__init__("; ".join(evidence.failure_reasons))


class PreparedExternalModelDescriptor(StrictModule, NonTrainableState):
    """Verified descriptor with fixed channel permutations and affine maps."""

    descriptor: ExternalModelDescriptor
    inventory: ExternalModelHostInventory
    coordinate_external_indices: Array
    coordinate_scale: Array
    coordinate_offset: Array
    actuator_external_indices: Array
    actuator_scale: Array
    actuator_offset: Array
    sensor_external_indices: Array
    sensor_scale: Array
    sensor_offset: Array
    evidence: ExternalModelPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def coordinate_to_phydrax(self, external_coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(external_coordinates)
        if values.ndim < 1 or values.shape[-1] != len(self.inventory.coordinate_channels):
            raise ValueError("External coordinates do not match the prepared coordinate map.")
        return (
            values[..., self.coordinate_external_indices] * self.coordinate_scale
            + self.coordinate_offset
        )

    def actuator_to_external(self, phydrax_actuators: ArrayLike, /) -> Array:
        values = jnp.asarray(phydrax_actuators)
        if values.ndim < 1 or values.shape[-1] != len(self.descriptor.actuator_map):
            raise ValueError("Phydrax actuators do not match the prepared actuator map.")
        mapped = values * self.actuator_scale + self.actuator_offset
        output = jnp.zeros(
            values.shape[:-1] + (len(self.inventory.actuator_channels),),
            dtype=mapped.dtype,
        )
        return output.at[..., self.actuator_external_indices].set(mapped)

    def sensor_to_phydrax(self, external_sensors: ArrayLike, /) -> Array:
        values = jnp.asarray(external_sensors)
        if values.ndim < 1 or values.shape[-1] != len(self.inventory.sensor_channels):
            raise ValueError("External sensors do not match the prepared sensor map.")
        return (
            values[..., self.sensor_external_indices] * self.sensor_scale
            + self.sensor_offset
        )


def _external_indices(
    names: tuple[str, ...], channel_names: tuple[str, ...], /
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray([lookup[name] for name in channel_names], dtype=np.int32)


def prepare_external_model_descriptor(
    descriptor: ExternalModelDescriptor,
    inventory: ExternalModelHostInventory,
    /,
) -> PreparedExternalModelDescriptor:
    """Verify complete external identity and lower fixed affine channel maps."""
    if not isinstance(descriptor, ExternalModelDescriptor):
        raise TypeError("descriptor must be ExternalModelDescriptor.")
    if not isinstance(inventory, ExternalModelHostInventory):
        raise TypeError("inventory must be ExternalModelHostInventory.")
    expected_assets = tuple((value.asset_name, value.sha256) for value in descriptor.assets)
    expected_coordinates = tuple(value.source_name for value in descriptor.coordinate_map)
    expected_actuators = tuple(value.target_name for value in descriptor.actuator_map)
    expected_sensors = tuple(value.source_name for value in descriptor.sensor_map)
    failures: list[str] = []
    if (inventory.source_package, inventory.source_revision) != (
        descriptor.source.package,
        descriptor.source.revision,
    ):
        failures.append("source package or revision does not match the descriptor")
    if inventory.asset_hashes != expected_assets:
        failures.append("asset names or SHA-256 digests do not match the descriptor")
    if inventory.provider_versions != descriptor.provider_versions:
        failures.append("provider versions do not match the descriptor")
    if inventory.compiled_sha256 != descriptor.compiled_sha256:
        failures.append("compiled SHA-256 digest does not match the descriptor")
    if set(inventory.coordinate_channels) != set(expected_coordinates):
        failures.append("coordinate channels do not exactly cover the descriptor map")
    if set(inventory.actuator_channels) != set(expected_actuators):
        failures.append("actuator channels do not exactly cover the descriptor map")
    if set(inventory.sensor_channels) != set(expected_sensors):
        failures.append("sensor channels do not exactly cover the descriptor map")
    evidence = ExternalModelPreparationEvidence(
        descriptor.descriptor_id,
        inventory.inventory_id,
        tuple(failures),
    )
    if not evidence.successful:
        raise ExternalModelPreparationError(evidence)
    coordinate_indices = _external_indices(
        inventory.coordinate_channels, expected_coordinates
    )
    actuator_indices = _external_indices(
        inventory.actuator_channels, expected_actuators
    )
    sensor_indices = _external_indices(inventory.sensor_channels, expected_sensors)
    coordinate_scale = np.asarray(
        [value.scale for value in descriptor.coordinate_map], dtype=float
    )
    coordinate_offset = np.asarray(
        [value.offset for value in descriptor.coordinate_map], dtype=float
    )
    actuator_scale = np.asarray(
        [value.scale for value in descriptor.actuator_map], dtype=float
    )
    actuator_offset = np.asarray(
        [value.offset for value in descriptor.actuator_map], dtype=float
    )
    sensor_scale = np.asarray(
        [value.scale for value in descriptor.sensor_map], dtype=float
    )
    sensor_offset = np.asarray(
        [value.offset for value in descriptor.sensor_map], dtype=float
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-external-skeletal-model-descriptor",
            "descriptor": descriptor.descriptor_id,
            "inventory": inventory.inventory_id,
            "evidence": evidence.evidence_id,
            "coordinate_indices": coordinate_indices.tolist(),
            "actuator_indices": actuator_indices.tolist(),
            "sensor_indices": sensor_indices.tolist(),
        }
    )
    return PreparedExternalModelDescriptor(
        descriptor,
        inventory,
        jnp.asarray(coordinate_indices),
        jnp.asarray(coordinate_scale),
        jnp.asarray(coordinate_offset),
        jnp.asarray(actuator_indices),
        jnp.asarray(actuator_scale),
        jnp.asarray(actuator_offset),
        jnp.asarray(sensor_indices),
        jnp.asarray(sensor_scale),
        jnp.asarray(sensor_offset),
        evidence,
        prepared_id,
    )


__all__ = [
    "ExternalModelAsset",
    "ExternalModelChannelBinding",
    "ExternalModelDescriptor",
    "ExternalModelDimensionalContract",
    "ExternalModelHostInventory",
    "ExternalModelPreparationError",
    "ExternalModelPreparationEvidence",
    "ExternalModelQuantity",
    "ExternalModelSource",
    "ExternalModelTransformation",
    "PreparedExternalModelDescriptor",
    "prepare_external_model_descriptor",
]

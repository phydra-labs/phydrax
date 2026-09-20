#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dependency-free array interchange for GR image and interferometry products.

The payloads are semantic array bundles, not FITS or UVFITS files. Optional I/O
adapters may map them to a concrete container without making core products depend
on a display library, an astronomy file package, or filesystem state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
    canonical_mapping,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import DifferentiationContract
from ...units import UnitDefinition
from ._gr_products import GRImageScreen, StokesImage
from ._interferometry import StokesVisibilityData, VisibilitySampling
from ._photometry import ObservationDataProvenance


class NeutralArrayPayload(StrictModule, NonTrainableState):
    """Named immutable arrays plus canonical JSON metadata and a content identity."""

    arrays: tuple[Array, ...]
    array_names: tuple[str, ...] = eqx.field(static=True)
    metadata_json: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        arrays: Sequence[ArrayLike],
        array_names: Sequence[str],
        metadata: Mapping[str, object],
        /,
    ):
        names = tuple(str(value).strip() for value in array_names)
        values = tuple(np.asarray(value) for value in arrays)
        if (
            not names
            or len(names) != len(values)
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or any(value.dtype.hasobject for value in values)
        ):
            raise ValueError(
                "Neutral array payloads need unique names and non-object numeric arrays."
            )
        normalized = canonical_mapping(metadata)
        metadata_json = canonical_json(normalized)
        self.arrays = tuple(jnp.asarray(value) for value in values)
        self.array_names = names
        self.metadata_json = metadata_json
        self.payload_id = canonical_fingerprint(
            {
                "kind": "gr-neutral-array-payload",
                "names": list(names),
                "arrays": array_tree_fingerprint(self.arrays),
                "metadata": normalized,
            }
        )

    @property
    def metadata(self) -> dict[str, object]:
        result = json.loads(self.metadata_json)
        if not isinstance(result, dict):
            raise ValueError("Neutral payload metadata must decode to an object.")
        return result

    def array(self, name: str, /) -> Array:
        identifier = str(name).strip()
        if identifier not in self.array_names:
            raise KeyError(identifier)
        return self.arrays[self.array_names.index(identifier)]


def _differentiation_payload(value: DifferentiationContract, /) -> dict[str, object]:
    return {
        "upstream_physical_parameters": value.upstream_physical_parameters,
        "stored_values": value.stored_values,
        "query_coordinates": value.query_coordinates,
        "local_parameters": value.local_parameters,
        "stochastic_realization": value.stochastic_realization,
        "higher_order": value.higher_order,
        "contract_id": value.contract_id,
    }


def _differentiation_from_payload(value: object, /) -> DifferentiationContract:
    if not isinstance(value, Mapping):
        raise TypeError("Differentiation metadata must be a mapping.")
    expected = {
        "upstream_physical_parameters",
        "stored_values",
        "query_coordinates",
        "local_parameters",
        "stochastic_realization",
        "higher_order",
        "contract_id",
    }
    if set(value) != expected:
        raise ValueError("Differentiation metadata does not use canonical fields.")
    boolean_names = (
        "upstream_physical_parameters",
        "stored_values",
        "query_coordinates",
        "local_parameters",
        "stochastic_realization",
        "higher_order",
    )
    if any(type(value[name]) is not bool for name in boolean_names):
        raise TypeError("Differentiation capability values must be booleans.")
    result = DifferentiationContract(
        upstream_physical_parameters=value["upstream_physical_parameters"],
        stored_values=value["stored_values"],
        query_coordinates=value["query_coordinates"],
        local_parameters=value["local_parameters"],
        stochastic_realization=value["stochastic_realization"],
        higher_order=value["higher_order"],
    )
    if value["contract_id"] != result.contract_id:
        raise ValueError("Differentiation contract identity does not match metadata.")
    return result


def _provenance_payload(value: ObservationDataProvenance, /) -> dict[str, object]:
    return {
        "producer": value.producer,
        "producer_version": value.producer_version,
        "source_id": value.source_id,
        "checksum": value.checksum,
        "license_id": value.license_id,
        "differentiation": _differentiation_payload(value.differentiation),
        "provenance_id": value.provenance_id,
    }


def _provenance_from_payload(value: object, /) -> ObservationDataProvenance:
    if not isinstance(value, Mapping):
        raise TypeError("Provenance metadata must be a mapping.")
    expected = {
        "producer",
        "producer_version",
        "source_id",
        "checksum",
        "license_id",
        "differentiation",
        "provenance_id",
    }
    if set(value) != expected:
        raise ValueError("Provenance metadata does not use canonical fields.")
    string_names = (
        "producer",
        "producer_version",
        "source_id",
        "checksum",
        "license_id",
        "provenance_id",
    )
    if any(not isinstance(value[name], str) for name in string_names):
        raise TypeError("Provenance text fields must be strings.")
    result = ObservationDataProvenance(
        producer=value["producer"],
        producer_version=value["producer_version"],
        source_id=value["source_id"],
        checksum=value["checksum"],
        license_id=value["license_id"],
        differentiation=_differentiation_from_payload(value["differentiation"]),
    )
    if value["provenance_id"] != result.provenance_id:
        raise ValueError("Provenance identity does not match metadata.")
    return result


def _unit_payload(value: UnitDefinition, /) -> dict[str, object]:
    return value.to_dict()


def _unit_from_payload(value: object, name: str, /) -> UnitDefinition:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} unit metadata must be a mapping.")
    return UnitDefinition.from_dict(value)


def _require_metadata(
    payload: NeutralArrayPayload,
    expected_names: tuple[str, ...],
    kind: str,
    /,
) -> dict[str, object]:
    if not isinstance(payload, NeutralArrayPayload):
        raise TypeError("payload must be NeutralArrayPayload.")
    if payload.array_names != expected_names:
        raise ValueError("Neutral payload array names or order are not canonical.")
    metadata = payload.metadata
    if metadata.get("kind") != kind:
        raise ValueError(f"Neutral payload is not {kind!r}.")
    return metadata


_IMAGE_ARRAYS = (
    "stokes",
    "screen_coordinates",
    "solid_angle",
    "screen_valid",
    "redshift",
    "redshift_valid",
    "lensing_masks",
    "frequency",
)


def stokes_image_to_fits_payload(image: StokesImage, /) -> NeutralArrayPayload:
    """Convert a physical Stokes image to a FITS-writer-neutral array bundle."""

    if not isinstance(image, StokesImage):
        raise TypeError("image must be StokesImage.")
    return NeutralArrayPayload(
        (
            image.stokes,
            image.screen.screen_coordinates,
            image.screen.solid_angle,
            image.screen.valid_mask,
            image.redshift,
            image.redshift_valid,
            image.lensing_masks,
            image.frequency,
        ),
        _IMAGE_ARRAYS,
        {
            "kind": "physical-stokes-image",
            "axis_order": ["stokes", "screen_y", "screen_x"],
            "stokes_components": list(image.stokes_components),
            "lensing_labels": list(image.lensing_labels),
            "units": {
                "angular": _unit_payload(image.screen.angular_unit),
                "solid_angle": _unit_payload(image.screen.solid_angle_unit),
                "intensity": _unit_payload(image.intensity_unit),
                "flux_density": _unit_payload(image.flux_density_unit),
                "frequency": _unit_payload(image.frequency_unit),
                "redshift": "dimensionless-nu-observer-over-nu-emitter",
            },
            "provenance": _provenance_payload(image.provenance),
            "screen_id": image.screen.screen_id,
            "source_content_id": image.content_id,
        },
    )


def stokes_image_from_fits_payload(payload: NeutralArrayPayload, /) -> StokesImage:
    """Reconstruct and identity-check a Stokes image from a neutral array bundle."""

    metadata = _require_metadata(payload, _IMAGE_ARRAYS, "physical-stokes-image")
    expected = {
        "kind",
        "axis_order",
        "stokes_components",
        "lensing_labels",
        "units",
        "provenance",
        "screen_id",
        "source_content_id",
    }
    if set(metadata) != expected:
        raise ValueError("Stokes image metadata does not use the canonical contract.")
    if metadata["axis_order"] != ["stokes", "screen_y", "screen_x"] or metadata[
        "stokes_components"
    ] != [
        "I",
        "Q",
        "U",
        "V",
    ]:
        raise ValueError("Stokes image axis semantics are not canonical.")
    units = metadata["units"]
    if not isinstance(units, Mapping) or set(units) != {
        "angular",
        "solid_angle",
        "intensity",
        "flux_density",
        "frequency",
        "redshift",
    }:
        raise ValueError("Stokes image unit metadata is incomplete.")
    if units["redshift"] != "dimensionless-nu-observer-over-nu-emitter":
        raise ValueError("Stokes image redshift convention is unsupported.")
    labels = metadata["lensing_labels"]
    if not isinstance(labels, list) or any(
        not isinstance(value, str) for value in labels
    ):
        raise TypeError("Stokes image lensing labels must be a string list.")
    screen = GRImageScreen(
        payload.array("screen_coordinates"),
        payload.array("solid_angle"),
        payload.array("screen_valid"),
        angular_unit=_unit_from_payload(units["angular"], "angular"),
        solid_angle_unit=_unit_from_payload(units["solid_angle"], "solid_angle"),
    )
    if metadata["screen_id"] != screen.screen_id:
        raise ValueError("Screen identity does not match payload content.")
    image = StokesImage(
        payload.array("stokes"),
        screen,
        _provenance_from_payload(metadata["provenance"]),
        frequency=payload.array("frequency"),
        redshift=payload.array("redshift"),
        redshift_valid=payload.array("redshift_valid"),
        lensing_masks=payload.array("lensing_masks"),
        lensing_labels=labels,
        intensity_unit=_unit_from_payload(units["intensity"], "intensity"),
        flux_density_unit=_unit_from_payload(units["flux_density"], "flux_density"),
        frequency_unit=_unit_from_payload(units["frequency"], "frequency"),
    )
    if metadata["source_content_id"] != image.content_id:
        raise ValueError("Stokes image identity does not match payload content.")
    return image


_VISIBILITY_ARRAYS = (
    "visibility_real",
    "visibility_imaginary",
    "uv_coordinates",
    "station_pairs",
    "frequencies",
)


def visibility_data_to_uvfits_payload(
    data: StokesVisibilityData, /
) -> NeutralArrayPayload:
    """Convert complex Stokes visibilities to a UVFITS-writer-neutral bundle."""

    if not isinstance(data, StokesVisibilityData):
        raise TypeError("data must be StokesVisibilityData.")
    return NeutralArrayPayload(
        (
            data.visibilities.real,
            data.visibilities.imag,
            data.sampling.uv_coordinates,
            data.sampling.station_pairs,
            data.sampling.frequencies,
        ),
        _VISIBILITY_ARRAYS,
        {
            "kind": "stokes-visibility-data",
            "axis_order": ["stokes", "visibility"],
            "stokes_components": ["I", "Q", "U", "V"],
            "complex_encoding": "separate-real-imaginary-arrays",
            "fourier_kernel": "exp(-2pi*i*(u*l+v*m))",
            "station_ids": list(data.sampling.station_ids),
            "uv_unit": data.sampling.uv_unit,
            "frequency_unit": _unit_payload(data.sampling.frequency_unit),
            "visibility_unit": _unit_payload(data.visibility_unit),
            "provenance": _provenance_payload(data.provenance),
            "parent_product_ids": list(data.parent_product_ids),
            "topology_id": data.sampling.topology_id,
            "source_content_id": data.content_id,
        },
    )


def visibility_data_from_uvfits_payload(
    payload: NeutralArrayPayload, /
) -> StokesVisibilityData:
    """Reconstruct and identity-check visibility data from a neutral array bundle."""

    metadata = _require_metadata(payload, _VISIBILITY_ARRAYS, "stokes-visibility-data")
    expected = {
        "kind",
        "axis_order",
        "stokes_components",
        "complex_encoding",
        "fourier_kernel",
        "station_ids",
        "uv_unit",
        "frequency_unit",
        "visibility_unit",
        "provenance",
        "parent_product_ids",
        "topology_id",
        "source_content_id",
    }
    if set(metadata) != expected:
        raise ValueError("Visibility metadata does not use the canonical contract.")
    if (
        metadata["axis_order"] != ["stokes", "visibility"]
        or metadata["stokes_components"] != ["I", "Q", "U", "V"]
        or metadata["complex_encoding"] != "separate-real-imaginary-arrays"
        or metadata["fourier_kernel"] != "exp(-2pi*i*(u*l+v*m))"
    ):
        raise ValueError("Visibility array or Fourier semantics are not canonical.")
    station_ids = metadata["station_ids"]
    parents = metadata["parent_product_ids"]
    if not isinstance(station_ids, list) or any(
        not isinstance(value, str) for value in station_ids
    ):
        raise TypeError("Visibility station_ids must be a string list.")
    if not isinstance(parents, list) or any(
        not isinstance(value, str) for value in parents
    ):
        raise TypeError("Visibility parent_product_ids must be a string list.")
    if not isinstance(metadata["uv_unit"], str):
        raise TypeError("Visibility uv_unit must be a string.")
    sampling = VisibilitySampling(
        payload.array("uv_coordinates"),
        payload.array("station_pairs"),
        payload.array("frequencies"),
        station_ids,
        frequency_unit=_unit_from_payload(metadata["frequency_unit"], "frequency"),
        uv_unit=metadata["uv_unit"],
    )
    if metadata["topology_id"] != sampling.topology_id:
        raise ValueError("Visibility topology identity does not match payload content.")
    visibilities = payload.array("visibility_real") + 1j * payload.array(
        "visibility_imaginary"
    )
    data = StokesVisibilityData(
        visibilities,
        sampling,
        _unit_from_payload(metadata["visibility_unit"], "visibility"),
        _provenance_from_payload(metadata["provenance"]),
        parent_product_ids=parents,
    )
    if metadata["source_content_id"] != data.content_id:
        raise ValueError("Visibility identity does not match payload content.")
    return data


__all__ = [
    "NeutralArrayPayload",
    "stokes_image_from_fits_payload",
    "stokes_image_to_fits_payload",
    "visibility_data_from_uvfits_payload",
    "visibility_data_to_uvfits_payload",
]

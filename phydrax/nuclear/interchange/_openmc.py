#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned OpenMC execution and one explicit multigroup-flux statepoint profile."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral

import h5py
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...interchange import (
    account_bounded_resource,
    AdapterReport,
    AdapterStatus,
    BoundedResource,
)
from ...interchange.energy_runtime import (
    EnergyRunResult,
    PinnedExecutable,
    run_energy_command,
)
from ...measurement import (
    DataOrigin,
    DataStage,
    DerivationRecord,
    IndependentStandardUncertainty,
    IndexSampleSupport,
    MeasurementAsset,
    QuantityField,
    SamplingSemantics,
    SpatialSamplingKind,
    ValueLayout,
)
from ...qualification import ReferenceArtifactManifest
from ...units import derived_unit, METER, SECOND
from .._energy import EnergyGroupStructure
from .._identity import NuclearParticleKind, NuclearSpeciesKey
from .._quantity import resolve_nuclear_quantity
from .._spectrum import MultigroupScalarFlux


_SCALAR_FLUX = derived_unit("1/m2/s-openmc", ((METER, -2), (SECOND, -1)))


@dataclass(frozen=True, slots=True)
class OpenMCStatepointProfile:
    tally_id: int
    value_shape: tuple[int, ...]
    axis_labels: tuple[str, ...]
    source_rate_s: float
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.tally_id, bool) or not isinstance(self.tally_id, Integral):
            raise TypeError("tally_id must be an integer.")
        tally = int(self.tally_id)
        shape = tuple(int(value) for value in self.value_shape)
        labels = tuple(str(value).strip() for value in self.axis_labels)
        rate = float(self.source_rate_s)
        if tally < 1 or not shape or any(value < 1 for value in shape):
            raise ValueError("OpenMC tally identity and value shape are invalid.")
        if (
            len(labels) != len(shape)
            or len(set(labels)) != len(labels)
            or any(not value for value in labels)
        ):
            raise ValueError("OpenMC tally axis labels must uniquely label value_shape.")
        if labels[-1] != "energy_group":
            raise ValueError(
                "OpenMC multigroup flux requires trailing energy_group axis."
            )
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("source_rate_s must be finite and positive.")
        object.__setattr__(self, "tally_id", tally)
        object.__setattr__(self, "value_shape", shape)
        object.__setattr__(self, "axis_labels", labels)
        object.__setattr__(self, "source_rate_s", rate)
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "openmc-statepoint-flux-profile",
                    "tally": tally,
                    "shape": list(shape),
                    "axes": list(labels),
                    "source_rate_s": rate,
                    "result_semantics": "sum-sum_sq-per-source-volume-normalized-flux",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OpenMCFluxImportResult:
    flux: MultigroupScalarFlux
    asset: MeasurementAsset
    resource: BoundedResource
    reference: ReferenceArtifactManifest
    report: AdapterReport
    openmc_version: str


def _scalar_integer(handle, name: str, /) -> int:
    if name in handle:
        value = np.asarray(handle[name][()])
    elif name in handle.attrs:
        value = np.asarray(handle.attrs[name])
    else:
        raise ValueError(f"OpenMC statepoint is missing {name}.")
    if value.size != 1:
        raise ValueError(f"OpenMC statepoint {name} must be scalar.")
    result = float(value.reshape(()))
    if not np.isfinite(result) or result < 1.0 or result != math.floor(result):
        raise ValueError(f"OpenMC statepoint {name} must be a positive integer.")
    return int(result)


def _text_attribute(handle, name: str, /) -> str:
    if name not in handle.attrs:
        raise ValueError(f"OpenMC statepoint is missing attribute {name}.")
    value = handle.attrs[name]
    if isinstance(value, bytes):
        result = value.decode("ascii", errors="strict")
    elif isinstance(value, np.ndarray) and value.size == 1:
        scalar = value.reshape(()).item()
        result = (
            scalar.decode("ascii", errors="strict")
            if isinstance(scalar, bytes)
            else str(scalar)
        )
    else:
        result = str(value)
    result = result.strip().strip("b'")
    if not result:
        raise ValueError(f"OpenMC statepoint attribute {name} is empty.")
    return result


def import_openmc_multigroup_flux(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    profile: OpenMCStatepointProfile,
    energy_groups: EnergyGroupStructure,
    /,
) -> OpenMCFluxImportResult:
    """Import one dedicated volume-normalized flux tally from an OpenMC statepoint."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be BoundedResource.")
    if not isinstance(reference, ReferenceArtifactManifest):
        raise TypeError("reference must be ReferenceArtifactManifest.")
    if not isinstance(profile, OpenMCStatepointProfile):
        raise TypeError("profile must be OpenMCStatepointProfile.")
    if not isinstance(energy_groups, EnergyGroupStructure):
        raise TypeError("energy_groups must be EnergyGroupStructure.")
    if profile.value_shape[-1] != energy_groups.group_count:
        raise ValueError("OpenMC profile and energy-group count disagree.")
    reference.verify_bytes(resource.data)
    reference.require_rights()
    with h5py.File(BytesIO(resource.data), "r") as handle:
        filetype = _text_attribute(handle, "filetype")
        if filetype != "statepoint":
            raise ValueError("HDF5 resource is not an OpenMC statepoint.")
        openmc_version = _text_attribute(handle, "openmc_version")
        realizations = _scalar_integer(handle, "n_realizations")
        path = f"tallies/tally {profile.tally_id}/results"
        if path not in handle:
            raise ValueError("Requested OpenMC tally result is absent.")
        dataset = handle[path]
        if dataset.is_virtual or dataset.external:
            raise ValueError(
                "OpenMC statepoint tally must be resident in the bounded resource."
            )
        expected = profile.value_shape + (2,)
        if dataset.shape != expected:
            raise ValueError(f"OpenMC tally results must have shape {expected}.")
        if dataset.dtype.hasobject or not np.issubdtype(dataset.dtype, np.number):
            raise TypeError("OpenMC tally results must use fixed-width numeric storage.")
        results = np.asarray(dataset[()], dtype=np.float64)
    if np.any(~np.isfinite(results)):
        raise ValueError("OpenMC tally sums must be finite.")
    tally_sum = results[..., 0]
    tally_sum_sq = results[..., 1]
    mean_per_source = tally_sum / realizations
    variance_numerator = tally_sum_sq / realizations - mean_per_source**2
    variance_scale = np.maximum(
        1.0,
        np.maximum(
            np.abs(tally_sum_sq / realizations),
            np.abs(mean_per_source**2),
        ),
    )
    variance_tolerance = 512.0 * np.finfo(np.float64).eps * variance_scale
    if np.any(variance_numerator < -variance_tolerance):
        raise ValueError("OpenMC tally sum of squares implies negative variance.")
    variance_of_mean = np.maximum(variance_numerator, 0.0) / max(realizations - 1, 1)
    mean = profile.source_rate_s * mean_per_source
    standard_error = profile.source_rate_s * np.sqrt(variance_of_mean)
    if np.any(mean < 0.0) or np.any(~np.isfinite(standard_error)):
        raise ValueError(
            "OpenMC flux mean and uncertainty must be finite and nonnegative."
        )
    target_id = canonical_fingerprint(
        {
            "kind": "openmc-multigroup-flux-values",
            "profile": profile.profile_id,
            "mean": array_tree_fingerprint(mean),
            "standard_error": array_tree_fingerprint(standard_error),
            "realizations": realizations,
            "openmc_version": openmc_version,
        }
    )
    accounted = account_bounded_resource(
        resource,
        depth=3,
        nodes=int(np.prod(expected)),
        attributes=3,
        losses=0,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "OpenMC-statepoint",
        "MeasurementAsset",
        source_id=accounted.manifest.manifest_id,
        target_id=target_id,
        coordinate_mapping=(
            "statepoint tally axes supplied explicitly by OpenMCStatepointProfile",
            "trailing energy groups use canonical ascending-energy Phydrax order",
        ),
        preserved_fields=(
            "tally sum",
            "tally sum of squares",
            "realization count",
            "OpenMC release identity",
        ),
        assumptions=(
            "tally is a dedicated cell-average scalar-flux score",
            "tally values are volume-normalized per source particle",
            "source_rate_s converts per-source tally values to physical scalar flux",
        ),
    )
    support = IndexSampleSupport(
        profile.value_shape,
        profile.axis_labels,
        frame_id=target_id,
    )
    quantity = resolve_nuclear_quantity(
        "openmc-scalar-flux",
        "scalar_flux",
        _SCALAR_FLUX,
        axes=profile.axis_labels,
        support_association="cell",
    )
    field = QuantityField(
        target_id,
        quantity,
        ValueLayout.scalar(),
        support,
        SamplingSemantics(SpatialSamplingKind.CELL_AVERAGE),
        mean,
        uncertainty=IndependentStandardUncertainty(standard_error, _SCALAR_FLUX),
    )
    asset = MeasurementAsset.from_single_reference(
        target_id,
        field,
        reference,
        DerivationRecord(
            DataOrigin.EXTERNAL,
            DataStage.DERIVED,
            transformation_id=report.report_id,
            adapter_report_ids=(report.report_id,),
        ),
        metadata={
            "openmc_version": openmc_version,
            "tally_id": profile.tally_id,
            "realization_count": realizations,
            "source_rate_s": profile.source_rate_s,
        },
    )
    neutron = NuclearSpeciesKey.from_particle(NuclearParticleKind.NEUTRON)
    flux = MultigroupScalarFlux(field, energy_groups, neutron, realizations)
    return OpenMCFluxImportResult(
        flux, asset, accounted, reference, report, openmc_version
    )


def run_openmc(
    executable: PinnedExecutable,
    /,
    *,
    inputs: dict[str, bytes],
    statepoint_path: str,
    args: tuple[str, ...] = (),
    timeout: float = 120.0,
    max_output_bytes: int = 512 * 1024 * 1024,
) -> EnergyRunResult:
    """Run a pinned OpenMC executable in the shared private energy runtime."""

    return run_energy_command(
        executable,
        args,
        inputs=inputs,
        outputs=(statepoint_path,),
        timeout=timeout,
        max_output_bytes=max_output_bytes,
    )


__all__ = [
    "OpenMCFluxImportResult",
    "OpenMCStatepointProfile",
    "import_openmc_multigroup_flux",
    "run_openmc",
]

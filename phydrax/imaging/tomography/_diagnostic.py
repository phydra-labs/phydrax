#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic spectral forward models for research diagnostic CT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    DiagnosticPhotonCoefficientRole,
    DiagnosticPhotonCoefficientTable,
    PhotonEnergyGrid,
)
from ...units import conversion_factor, derived_unit, KILOGRAM, METER, UnitDefinition
from ._core import BeerLambertResult, ProjectionSupport, VoxelXRayTransformPlan


ExposureBasis = Literal["relative", "absolute"]
_AREAL_MASS_UNIT = derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2)))
_MASS_ATTENUATION_UNIT = derived_unit("m2/kg", ((METER, 2), (KILOGRAM, -1)))


def _readonly(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use object dtype.")
    array.setflags(write=False)
    return array


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result or result != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return result


def _exposure_basis(value: str, /) -> ExposureBasis:
    if value not in ("relative", "absolute"):
        raise ValueError("exposure_basis must be 'relative' or 'absolute'.")
    return value


def _grid(value: PhotonEnergyGrid, /) -> PhotonEnergyGrid:
    if not isinstance(value, PhotonEnergyGrid):
        raise TypeError("energy_grid must be PhotonEnergyGrid.")
    return value


@dataclass(frozen=True, slots=True)
class TubeSpectrum:
    """Per-bin tube fluence with an explicit relative or absolute basis."""

    energy_grid: PhotonEnergyGrid
    fluence: np.ndarray
    exposure_basis: ExposureBasis
    spectrum_id: str = field(init=False)

    def __post_init__(self) -> None:
        grid = _grid(self.energy_grid)
        fluence = _readonly(self.fluence, "fluence")
        basis = _exposure_basis(self.exposure_basis)
        if fluence.shape != grid.energy_j.shape:
            raise ValueError("fluence must have one value per photon-energy bin.")
        if (
            np.any(~np.isfinite(fluence))
            or np.any(fluence < 0.0)
            or not np.any(fluence > 0.0)
        ):
            raise ValueError("fluence must be finite, non-negative, and nonzero.")
        object.__setattr__(self, "fluence", fluence)
        object.__setattr__(self, "exposure_basis", basis)
        object.__setattr__(
            self,
            "spectrum_id",
            canonical_fingerprint(
                {
                    "kind": "ct-tube-spectrum",
                    "energy_grid": grid.grid_id,
                    "fluence": array_tree_fingerprint(fluence),
                    "exposure_basis": basis,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FilterStack:
    """Net energy-dependent transmission of the source-side filter stack."""

    energy_grid: PhotonEnergyGrid
    transmission: np.ndarray
    filter_id: str = field(init=False)

    def __post_init__(self) -> None:
        grid = _grid(self.energy_grid)
        transmission = _readonly(self.transmission, "transmission")
        if transmission.shape != grid.energy_j.shape:
            raise ValueError("Filter transmission must have one value per energy bin.")
        if np.any(~np.isfinite(transmission)) or np.any(
            (transmission < 0.0) | (transmission > 1.0)
        ):
            raise ValueError("Filter transmission must be finite and lie in [0, 1].")
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(
            self,
            "filter_id",
            canonical_fingerprint(
                {
                    "kind": "ct-filter-stack",
                    "energy_grid": grid.grid_id,
                    "transmission": array_tree_fingerprint(transmission),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BowtieTransmission:
    """Energy-resolved bowtie transmission, optionally varying over detector bins."""

    energy_grid: PhotonEnergyGrid
    transmission: np.ndarray
    bowtie_id: str = field(init=False)

    def __post_init__(self) -> None:
        grid = _grid(self.energy_grid)
        transmission = _readonly(self.transmission, "transmission")
        if transmission.ndim < 1 or transmission.shape[-1] != grid.energy_j.size:
            raise ValueError("Bowtie transmission must end in the photon-energy axis.")
        if np.any(~np.isfinite(transmission)) or np.any(
            (transmission < 0.0) | (transmission > 1.0)
        ):
            raise ValueError("Bowtie transmission must be finite and lie in [0, 1].")
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(
            self,
            "bowtie_id",
            canonical_fingerprint(
                {
                    "kind": "ct-bowtie-transmission",
                    "energy_grid": grid.grid_id,
                    "transmission": array_tree_fingerprint(transmission),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class AECSetting:
    """One view's deterministic exposure modulation in a declared basis."""

    exposure: float
    exposure_basis: ExposureBasis
    setting_id: str = field(init=False)

    def __post_init__(self) -> None:
        exposure = float(self.exposure)
        basis = _exposure_basis(self.exposure_basis)
        if not np.isfinite(exposure) or exposure < 0.0:
            raise ValueError("AEC exposure must be finite and non-negative.")
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "exposure_basis", basis)
        object.__setattr__(
            self,
            "setting_id",
            canonical_fingerprint(
                {
                    "kind": "ct-aec-setting",
                    "exposure": exposure,
                    "exposure_basis": basis,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DetectorResponse:
    """Spectral detector sensitivity and deterministic readout calibration."""

    energy_grid: PhotonEnergyGrid
    response: np.ndarray
    dark_signal: float = 0.0
    gain: float = 1.0
    saturation: float = np.inf
    response_id: str = field(init=False)

    def __post_init__(self) -> None:
        grid = _grid(self.energy_grid)
        response = _readonly(self.response, "response")
        dark = float(self.dark_signal)
        gain = float(self.gain)
        saturation = float(self.saturation)
        if response.shape != grid.energy_j.shape:
            raise ValueError("Detector response must have one value per energy bin.")
        if np.any(~np.isfinite(response)) or np.any(response < 0.0):
            raise ValueError("Detector response must be finite and non-negative.")
        if not np.any(response > 0.0):
            raise ValueError("Detector response must contain a positive value.")
        if not np.isfinite(dark) or dark < 0.0:
            raise ValueError("dark_signal must be finite and non-negative.")
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError("gain must be finite and strictly positive.")
        if np.isnan(saturation) or saturation <= 0.0:
            raise ValueError("saturation must be strictly positive or infinity.")
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "dark_signal", dark)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(
            self,
            "response_id",
            canonical_fingerprint(
                {
                    "kind": "ct-detector-response",
                    "energy_grid": grid.grid_id,
                    "response": array_tree_fingerprint(response),
                    "dark_signal": dark,
                    "gain": gain,
                    "saturation": "unbounded" if np.isinf(saturation) else saturation,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ScatterLabel:
    """Provenance label for scatter added at the detector-input signal stage."""

    model_id: str
    exposure_basis: ExposureBasis
    signal_domain: str = "detector_input"
    label_id: str = field(init=False)

    def __post_init__(self) -> None:
        model = _identifier(self.model_id, "model_id")
        basis = _exposure_basis(self.exposure_basis)
        if self.signal_domain != "detector_input":
            raise ValueError("Scatter signal_domain must be 'detector_input'.")
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "exposure_basis", basis)
        object.__setattr__(
            self,
            "label_id",
            canonical_fingerprint(
                {
                    "kind": "ct-scatter-label",
                    "model": model,
                    "exposure_basis": basis,
                    "signal_domain": "detector_input",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CTViewAcquisition:
    """All deterministic source and detector settings for one projection view."""

    view_id: str
    spectrum: TubeSpectrum
    filters: FilterStack
    bowtie: BowtieTransmission
    aec: AECSetting
    detector: DetectorResponse
    acquisition_id: str = field(init=False)

    def __post_init__(self) -> None:
        view = _identifier(self.view_id, "view_id")
        if not isinstance(self.spectrum, TubeSpectrum):
            raise TypeError("spectrum must be TubeSpectrum.")
        if not isinstance(self.filters, FilterStack):
            raise TypeError("filters must be FilterStack.")
        if not isinstance(self.bowtie, BowtieTransmission):
            raise TypeError("bowtie must be BowtieTransmission.")
        if not isinstance(self.aec, AECSetting):
            raise TypeError("aec must be AECSetting.")
        if not isinstance(self.detector, DetectorResponse):
            raise TypeError("detector must be DetectorResponse.")
        grid_id = self.spectrum.energy_grid.grid_id
        if any(
            component.energy_grid.grid_id != grid_id
            for component in (self.filters, self.bowtie, self.detector)
        ):
            raise ValueError("All view components must share one photon-energy grid.")
        if self.spectrum.exposure_basis != self.aec.exposure_basis:
            raise ValueError(
                "Relative and absolute source/AEC exposure declarations cannot be mixed."
            )
        object.__setattr__(self, "view_id", view)
        object.__setattr__(
            self,
            "acquisition_id",
            canonical_fingerprint(
                {
                    "kind": "ct-view-acquisition",
                    "view": view,
                    "spectrum": self.spectrum.spectrum_id,
                    "filters": self.filters.filter_id,
                    "bowtie": self.bowtie.bowtie_id,
                    "aec": self.aec.setting_id,
                    "detector": self.detector.response_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CTAcquisitionProtocol:
    """Ordered per-view CT acquisition settings."""

    views: tuple[CTViewAcquisition, ...]
    protocol_id: str = field(init=False)

    def __post_init__(self) -> None:
        views = tuple(self.views)
        if not views or any(not isinstance(view, CTViewAcquisition) for view in views):
            raise TypeError("views must contain CTViewAcquisition values.")
        identifiers = tuple(view.view_id for view in views)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("CT view identifiers must be unique.")
        grid_id = views[0].spectrum.energy_grid.grid_id
        if any(view.spectrum.energy_grid.grid_id != grid_id for view in views[1:]):
            raise ValueError("Every protocol view must share one photon-energy grid.")
        basis = views[0].spectrum.exposure_basis
        if any(view.spectrum.exposure_basis != basis for view in views[1:]):
            raise ValueError("A protocol cannot mix relative and absolute exposures.")
        object.__setattr__(self, "views", views)
        object.__setattr__(
            self,
            "protocol_id",
            canonical_fingerprint(
                {
                    "kind": "ct-acquisition-protocol",
                    "views": [view.acquisition_id for view in views],
                }
            ),
        )

    @property
    def view_ids(self) -> tuple[str, ...]:
        return tuple(view.view_id for view in self.views)

    @property
    def energy_grid(self) -> PhotonEnergyGrid:
        return self.views[0].spectrum.energy_grid

    @property
    def exposure_basis(self) -> ExposureBasis:
        return self.views[0].spectrum.exposure_basis


class MaterialBasisProjectionPlan(StrictModule):
    """Project density-weighted material fractions to ordered areal masses."""

    transform: VoxelXRayTransformPlan
    path_length_to_meter: Array
    material_ids: tuple[str, ...] = eqx.field(static=True)
    areal_mass_unit: UnitDefinition = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transform: VoxelXRayTransformPlan,
        material_ids: tuple[str, ...],
        /,
        *,
        path_length_unit: UnitDefinition = METER,
    ):
        if not isinstance(transform, VoxelXRayTransformPlan):
            raise TypeError("transform must be VoxelXRayTransformPlan.")
        materials = tuple(
            _identifier(material_id, "material_id") for material_id in material_ids
        )
        if not materials or len(materials) != len(set(materials)):
            raise ValueError("material_ids must be nonempty and unique.")
        if not isinstance(path_length_unit, UnitDefinition):
            raise TypeError("path_length_unit must be UnitDefinition.")
        scale = float(conversion_factor(path_length_unit, METER))
        self.transform = transform
        self.path_length_to_meter = jnp.asarray(scale)
        self.material_ids = materials
        self.areal_mass_unit = _AREAL_MASS_UNIT
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ct-material-basis-projection",
                "transform": transform.operator_id,
                "materials": list(materials),
                "path_length_unit": path_length_unit.unit_id,
            }
        )

    def project(
        self,
        density_kg_m3: ArrayLike,
        material_fractions: ArrayLike,
        /,
    ) -> Array:
        density = jnp.asarray(density_kg_m3)
        fractions = jnp.asarray(material_fractions)
        if density.shape != self.transform.volume_shape:
            raise ValueError(
                f"density_kg_m3 must have shape {self.transform.volume_shape}."
            )
        expected = self.transform.volume_shape + (len(self.material_ids),)
        if fractions.shape != expected:
            raise ValueError(f"material_fractions must have shape {expected}.")
        material_density = ein.contract("...,...m->...m", density, fractions)
        projections = tuple(
            self.transform.forward(material_density[..., index]).values
            for index in range(len(self.material_ids))
        )
        return jnp.stack(projections, axis=-1) * self.path_length_to_meter


class PolychromaticDetectorPlan(StrictModule, NonTrainableState):
    """Contract material attenuation and deterministic CT acquisition response."""

    incident_fluence: Array
    detector_response: Array
    mass_attenuation_m2_kg: Array
    dark_signal: Array
    gain: Array
    saturation: Array
    projection_shape: tuple[int, ...] = eqx.field(static=True)
    material_ids: tuple[str, ...] = eqx.field(static=True)
    exposure_basis: ExposureBasis = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    coefficient_table_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: ProjectionSupport,
        protocol: CTAcquisitionProtocol,
        coefficients: DiagnosticPhotonCoefficientTable,
        /,
    ):
        if not isinstance(support, ProjectionSupport):
            raise TypeError("support must be ProjectionSupport.")
        if not isinstance(protocol, CTAcquisitionProtocol):
            raise TypeError("protocol must be CTAcquisitionProtocol.")
        if not isinstance(coefficients, DiagnosticPhotonCoefficientTable):
            raise TypeError("coefficients must be DiagnosticPhotonCoefficientTable.")
        if protocol.view_ids != support.view_ids:
            raise ValueError(
                "Protocol view ordering must exactly match ProjectionSupport."
            )
        if coefficients.role is not DiagnosticPhotonCoefficientRole.MASS_ATTENUATION:
            raise ValueError("CT attenuation requires MASS_ATTENUATION coefficients.")
        detector_shape = support.projection_shape[1:]
        energy_count = protocol.energy_grid.energy_j.size
        fluence_by_view = []
        response_by_view = []
        dark_by_view = []
        gain_by_view = []
        saturation_by_view = []
        for view in protocol.views:
            bowtie = np.asarray(view.bowtie.transmission)
            allowed = {(energy_count,), detector_shape + (energy_count,)}
            if bowtie.shape not in allowed:
                raise ValueError(
                    "Bowtie transmission must be energy-only or match the detector shape."
                )
            bowtie = np.broadcast_to(bowtie, detector_shape + (energy_count,))
            incident = (
                view.aec.exposure
                * view.spectrum.fluence
                * view.filters.transmission
                * bowtie
            )
            fluence_by_view.append(incident)
            response_by_view.append(
                np.broadcast_to(view.detector.response, incident.shape)
            )
            dark_by_view.append(np.full(detector_shape, view.detector.dark_signal))
            gain_by_view.append(np.full(detector_shape, view.detector.gain))
            saturation_by_view.append(np.full(detector_shape, view.detector.saturation))
        evaluation = coefficients.evaluate(
            protocol.energy_grid.energy_j, coefficients.material_ids
        )
        if not bool(np.all(np.asarray(evaluation.evidence.supported))):
            raise ValueError("The coefficient table does not support the protocol grid.")
        if evaluation.coefficient.shape != (
            len(coefficients.material_ids),
            energy_count,
        ):
            raise ValueError(
                "Mass-attenuation coefficients must have material-by-energy shape."
            )
        coefficient_scale = float(
            conversion_factor(evaluation.unit, _MASS_ATTENUATION_UNIT)
        )
        self.incident_fluence = jnp.asarray(np.stack(fluence_by_view, axis=0))
        self.detector_response = jnp.asarray(np.stack(response_by_view, axis=0))
        self.mass_attenuation_m2_kg = jnp.asarray(
            np.asarray(evaluation.coefficient) * coefficient_scale
        )
        self.dark_signal = jnp.asarray(np.stack(dark_by_view, axis=0))
        self.gain = jnp.asarray(np.stack(gain_by_view, axis=0))
        self.saturation = jnp.asarray(np.stack(saturation_by_view, axis=0))
        self.projection_shape = support.projection_shape
        self.material_ids = tuple(coefficients.material_ids)
        self.exposure_basis = protocol.exposure_basis
        self.protocol_id = protocol.protocol_id
        self.coefficient_table_id = coefficients.table_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ct-polychromatic-detector-plan",
                "support": support.support_id,
                "protocol": protocol.protocol_id,
                "coefficient_table": coefficients.table_id,
                "materials": list(coefficients.material_ids),
            }
        )

    def evaluate(
        self,
        areal_mass_kg_m2: ArrayLike,
        /,
        *,
        scatter_signal: ArrayLike | None = None,
        scatter_label: ScatterLabel | None = None,
    ) -> BeerLambertResult:
        areal_mass = jnp.asarray(areal_mass_kg_m2)
        expected_shape = self.projection_shape + (len(self.material_ids),)
        if areal_mass.shape != expected_shape:
            raise ValueError(f"areal_mass_kg_m2 must have shape {expected_shape}.")
        if scatter_signal is None:
            if scatter_label is not None:
                raise ValueError("scatter_label requires an explicit scatter_signal.")
            scatter = jnp.zeros(self.projection_shape, dtype=areal_mass.dtype)
        else:
            if not isinstance(scatter_label, ScatterLabel):
                raise TypeError("Explicit scatter_signal requires a ScatterLabel.")
            if scatter_label.exposure_basis != self.exposure_basis:
                raise ValueError(
                    "Relative and absolute primary/scatter signal bases cannot be mixed."
                )
            scatter = jnp.broadcast_to(
                jnp.asarray(scatter_signal, dtype=areal_mass.dtype),
                self.projection_shape,
            )
        optical_depth = ein.contract(
            "...m,me->...e", areal_mass, self.mass_attenuation_m2_kg
        )
        transmitted_spectrum = self.incident_fluence * jnp.exp(-optical_depth)
        primary = ein.contract(
            "...e,...e->...", transmitted_spectrum, self.detector_response
        )
        expected = self.dark_signal + self.gain * (primary + scatter)
        saturated = expected > self.saturation
        output = jnp.minimum(expected, self.saturation)
        finite = (
            jnp.all(jnp.isfinite(output))
            & jnp.all(jnp.isfinite(areal_mass))
            & jnp.all(areal_mass >= 0.0)
            & jnp.all(jnp.isfinite(scatter))
            & jnp.all(scatter >= 0.0)
        )
        return BeerLambertResult(
            output,
            primary,
            scatter,
            saturated,
            finite,
            finite,
        )


__all__ = [
    "AECSetting",
    "BowtieTransmission",
    "CTAcquisitionProtocol",
    "CTViewAcquisition",
    "DetectorResponse",
    "FilterStack",
    "MaterialBasisProjectionPlan",
    "PolychromaticDetectorPlan",
    "ScatterLabel",
    "TubeSpectrum",
]

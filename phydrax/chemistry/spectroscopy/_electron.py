#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-element ARPES and provider-vacuum-LDOS Tersoff--Hamann profiles."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition
from ._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


class PhotoemissionMatrixElementRequest(StrictModule, NonTrainableState):
    kpoints: Array
    photon_energy: Array
    polarization: Array
    spectral_source_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        kpoints: ArrayLike,
        photon_energy: ArrayLike,
        polarization: ArrayLike,
        spectral_source_id: str,
        geometry_id: str,
        /,
    ):
        points = jnp.asarray(kpoints, dtype=jnp.float64)
        photon = jnp.asarray(photon_energy, dtype=jnp.float64).reshape(())
        vector = jnp.asarray(polarization)
        source = str(spectral_source_id).strip()
        geometry = str(geometry_id).strip()
        if (
            points.ndim != 2
            or points.shape[1] not in (1, 2, 3)
            or vector.shape != (3,)
            or bool(jnp.any(~jnp.isfinite(points)))
            or bool(~jnp.isfinite(photon))
            or bool(photon <= 0.0)
            or bool(jnp.any(~jnp.isfinite(vector)))
            or bool(jnp.abs(jnp.linalg.norm(vector) - 1.0) > 1.0e-8)
            or not source
            or not geometry
        ):
            raise ValueError("Photoemission matrix-element request is invalid.")
        self.kpoints = points
        self.photon_energy = photon
        self.polarization = vector
        self.spectral_source_id = source
        self.geometry_id = geometry
        self.request_id = canonical_fingerprint(
            {
                "kind": "photoemission-matrix-element-request",
                "source": source,
                "geometry": geometry,
                "arrays": array_tree_fingerprint(
                    {
                        "kpoints": np.asarray(points),
                        "photon": np.asarray(photon),
                        "polarization": np.asarray(vector),
                    }
                ),
            }
        )


class PhotoemissionMatrixElementResult(StrictModule, NonTrainableState):
    matrix_elements: Array
    request: PhotoemissionMatrixElementRequest
    provider_id: str = eqx.field(static=True)
    source_hashes: tuple[str, ...] = eqx.field(static=True)
    converged: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix_elements: ArrayLike,
        request: PhotoemissionMatrixElementRequest,
        provider_id: str,
        source_hashes: tuple[str, ...],
        converged: ArrayLike,
        /,
    ):
        elements = jnp.asarray(matrix_elements)
        provider = str(provider_id).strip()
        hashes = tuple(str(value).strip() for value in source_hashes)
        if (
            elements.ndim != 2
            or elements.shape[0] != request.kpoints.shape[0]
            or elements.shape[1] == 0
            or bool(jnp.any(~jnp.isfinite(elements)))
            or not provider
            or not hashes
            or any(not value for value in hashes)
        ):
            raise ValueError("Photoemission matrix elements or provenance are invalid.")
        self.matrix_elements = elements
        self.request = request
        self.provider_id = provider
        self.source_hashes = hashes
        self.converged = jnp.asarray(converged, dtype=jnp.bool_).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "photoemission-matrix-element-result",
                "request": request.request_id,
                "provider": provider,
                "source_hashes": list(hashes),
                "converged": bool(self.converged),
                "matrix_elements": array_tree_fingerprint(np.asarray(elements)),
            }
        )


class ARPESEvidence(StrictModule, NonTrainableState):
    spectral_moment_residual: Array
    passivity_residual: Array
    selection_rule_residual: Array
    successful: Array


class ARPESResult(StrictModule, NonTrainableState):
    raw_response: SpectralResponseProduct
    evidence: ARPESEvidence
    matrix_element_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_response: SpectralResponseProduct,
        evidence: ARPESEvidence,
        matrix_element_result_id: str,
        /,
    ):
        self.raw_response = raw_response
        self.evidence = evidence
        self.matrix_element_result_id = str(matrix_element_result_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "matrix-element-arpes-result",
                "raw": raw_response.product_id,
                "matrix_elements": self.matrix_element_result_id,
                "successful": bool(evidence.successful),
            }
        )


class ARPESPlan(StrictModule, NonTrainableState):
    energy: Array
    chemical_potential: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    kpoint_capacity: int = eqx.field(static=True)
    band_capacity: int = eqx.field(static=True)
    moment_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        /,
        *,
        chemical_potential: float,
        temperature: float,
        kpoint_capacity: int,
        band_capacity: int,
        moment_tolerance: float = 5.0e-3,
    ):
        grid = jnp.asarray(energy, dtype=jnp.float64)
        chemical = float(chemical_potential)
        thermal = float(temperature)
        tolerance = float(moment_tolerance)
        if (
            grid.ndim != 1
            or grid.size < 2
            or bool(jnp.any(~jnp.isfinite(grid)))
            or bool(jnp.any(jnp.diff(grid) <= 0.0))
            or not all(isfinite(value) for value in (chemical, thermal, tolerance))
            or thermal <= 0.0
            or tolerance <= 0.0
            or int(kpoint_capacity) <= 0
            or int(band_capacity) <= 0
        ):
            raise ValueError("ARPES plan grid, thermodynamics, or capacity is invalid.")
        self.energy = grid
        self.chemical_potential = chemical
        self.temperature = thermal
        self.kpoint_capacity = int(kpoint_capacity)
        self.band_capacity = int(band_capacity)
        self.moment_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "matrix-element-arpes-plan",
                "approximation": "sudden-provider-matrix-element",
                "energy": array_tree_fingerprint(np.asarray(grid)),
                "chemical_potential": chemical,
                "temperature": thermal,
                "kpoint_capacity": self.kpoint_capacity,
                "band_capacity": self.band_capacity,
                "moment_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        spectral_density: ArrayLike,
        spectral_source_id: str,
        matrix_elements: PhotoemissionMatrixElementResult,
        energy_unit: UnitDefinition,
        response_unit: UnitDefinition,
        /,
    ) -> ARPESResult:
        density = jnp.asarray(spectral_density, dtype=jnp.float64)
        kpoints, bands = matrix_elements.matrix_elements.shape
        if density.shape != (kpoints, bands, self.energy.size):
            raise ValueError("ARPES spectral density must have shape (k, band, energy).")
        if kpoints > self.kpoint_capacity or bands > self.band_capacity:
            raise ValueError("ARPES input exceeds the planned k-point or band capacity.")
        if matrix_elements.request.spectral_source_id != str(spectral_source_id).strip():
            raise ValueError("Matrix elements do not belong to this spectral source.")
        if not bool(matrix_elements.converged):
            raise ValueError(
                "Unconverged matrix elements cannot enter the production profile."
            )
        if bool(jnp.any(~jnp.isfinite(density))) or bool(jnp.any(density < 0.0)):
            raise ValueError("ARPES requires a finite non-negative spectral density.")
        moments = jnp.trapezoid(density, self.energy, axis=-1)
        moment_residual = jnp.max(jnp.abs(moments - 1.0))
        fermi = 1.0 / (
            jnp.exp((self.energy - self.chemical_potential) / self.temperature) + 1.0
        )
        weights = jnp.real(
            matrix_elements.matrix_elements * jnp.conj(matrix_elements.matrix_elements)
        )
        resolved = jnp.sum(weights[:, :, None] * density, axis=1) * fermi[None, :]
        passivity = jnp.maximum(-jnp.min(resolved), 0.0)
        forbidden = weights == 0.0
        forbidden_leakage = jnp.max(
            jnp.where(forbidden[:, :, None], weights[:, :, None] * density, 0.0)
        )
        successful = (
            bool(moment_residual <= self.moment_tolerance)
            and bool(passivity <= self.moment_tolerance)
            and bool(forbidden_leakage <= self.moment_tolerance)
        )
        evidence = ARPESEvidence(
            moment_residual,
            passivity,
            forbidden_leakage,
            jnp.asarray(successful),
        )
        raw_evidence = SpectralResponseEvidence(
            passivity,
            moment_residual,
            0.0,
            forbidden_leakage,
            successful,
        )
        raw = SpectralResponseProduct(
            self.energy,
            resolved,
            jnp.ones(self.energy.shape, dtype=jnp.bool_),
            energy_unit,
            response_unit,
            tuple(f"k[{index}]" for index in range(kpoints)),
            SpectralResponseRepresentation.DENSITY,
            "provider-matrix-element-sudden-arpes",
            matrix_elements.result_id,
            raw_evidence,
        )
        return ARPESResult(raw, evidence, matrix_elements.result_id)


class VacuumLDOSRequest(StrictModule, NonTrainableState):
    tip_positions: Array
    energy: Array
    electronic_source_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self, tip_positions: ArrayLike, energy: ArrayLike, electronic_source_id: str, /
    ):
        positions = jnp.asarray(tip_positions, dtype=jnp.float64)
        grid = jnp.asarray(energy, dtype=jnp.float64)
        source = str(electronic_source_id).strip()
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or grid.ndim != 1
            or grid.size < 2
            or bool(jnp.any(~jnp.isfinite(positions)))
            or bool(jnp.any(~jnp.isfinite(grid)))
            or bool(jnp.any(jnp.diff(grid) <= 0.0))
            or not source
        ):
            raise ValueError("Vacuum-LDOS request coordinates are invalid.")
        self.tip_positions = positions
        self.energy = grid
        self.electronic_source_id = source
        self.request_id = canonical_fingerprint(
            {
                "kind": "vacuum-ldos-request",
                "source": source,
                "arrays": array_tree_fingerprint(
                    {"positions": np.asarray(positions), "energy": np.asarray(grid)}
                ),
            }
        )


class VacuumLDOSResult(StrictModule, NonTrainableState):
    ldos: Array
    request: VacuumLDOSRequest
    provider_id: str = eqx.field(static=True)
    source_hashes: tuple[str, ...] = eqx.field(static=True)
    moment_residual: Array
    converged: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        ldos: ArrayLike,
        request: VacuumLDOSRequest,
        provider_id: str,
        source_hashes: tuple[str, ...],
        moment_residual: ArrayLike,
        converged: ArrayLike,
        /,
    ):
        density = jnp.asarray(ldos, dtype=jnp.float64)
        provider = str(provider_id).strip()
        hashes = tuple(str(value).strip() for value in source_hashes)
        residual = jnp.asarray(moment_residual, dtype=jnp.float64).reshape(())
        if (
            density.shape != (request.tip_positions.shape[0], request.energy.size)
            or bool(jnp.any(~jnp.isfinite(density)))
            or bool(jnp.any(density < 0.0))
            or bool(~jnp.isfinite(residual))
            or bool(residual < 0.0)
            or not provider
            or not hashes
            or any(not value for value in hashes)
        ):
            raise ValueError("Vacuum LDOS or provider provenance is invalid.")
        self.ldos = density
        self.request = request
        self.provider_id = provider
        self.source_hashes = hashes
        self.moment_residual = residual
        self.converged = jnp.asarray(converged, dtype=jnp.bool_).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "vacuum-ldos-result",
                "request": request.request_id,
                "provider": provider,
                "hashes": list(hashes),
                "moment_residual": float(residual),
                "converged": bool(self.converged),
                "ldos": array_tree_fingerprint(np.asarray(density)),
            }
        )


class TersoffHamannEvidence(StrictModule, NonTrainableState):
    passivity_residual: Array
    current_derivative_residual: Array
    spectral_moment_residual: Array
    successful: Array


class TersoffHamannResult(StrictModule, NonTrainableState):
    raw_didv: SpectralResponseProduct
    current: Array
    biases: Array
    evidence: TersoffHamannEvidence
    provider_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_didv: SpectralResponseProduct,
        current: ArrayLike,
        biases: ArrayLike,
        evidence: TersoffHamannEvidence,
        provider_result_id: str,
        /,
    ):
        self.raw_didv = raw_didv
        self.current = jnp.asarray(current)
        self.biases = jnp.asarray(biases)
        self.evidence = evidence
        self.provider_result_id = str(provider_result_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "ters-off-hamann-result",
                "raw": raw_didv.product_id,
                "provider": self.provider_result_id,
                "successful": bool(evidence.successful),
            }
        )


class TersoffHamannPlan(StrictModule, NonTrainableState):
    biases: Array
    temperature: float = eqx.field(static=True)
    current_scale: float = eqx.field(static=True)
    position_capacity: int = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        biases: ArrayLike,
        /,
        *,
        temperature: float,
        current_scale: float,
        position_capacity: int,
        closure_tolerance: float = 1.0e-3,
    ):
        voltage = jnp.asarray(biases, dtype=jnp.float64)
        thermal = float(temperature)
        scale = float(current_scale)
        tolerance = float(closure_tolerance)
        if (
            voltage.ndim != 1
            or voltage.size < 3
            or bool(jnp.any(~jnp.isfinite(voltage)))
            or bool(jnp.any(jnp.diff(voltage) <= 0.0))
            or not all(isfinite(value) for value in (thermal, scale, tolerance))
            or thermal <= 0.0
            or scale <= 0.0
            or tolerance <= 0.0
            or int(position_capacity) <= 0
        ):
            raise ValueError("Tersoff--Hamann plan is invalid.")
        if not bool(jnp.any(voltage == 0.0)):
            raise ValueError(
                "Tersoff--Hamann bias grid must contain an exact zero reference."
            )
        self.biases = voltage
        self.temperature = thermal
        self.current_scale = scale
        self.position_capacity = int(position_capacity)
        self.closure_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ters-off-hamann-plan",
                "biases": array_tree_fingerprint(np.asarray(voltage)),
                "temperature": thermal,
                "current_scale": scale,
                "position_capacity": self.position_capacity,
                "closure_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        vacuum_ldos: VacuumLDOSResult,
        bias_unit: UnitDefinition,
        response_unit: UnitDefinition,
        /,
    ) -> TersoffHamannResult:
        if vacuum_ldos.request.tip_positions.shape[0] > self.position_capacity:
            raise ValueError("Vacuum LDOS exceeds the planned position capacity.")
        if not bool(vacuum_ldos.converged):
            raise ValueError("Unconverged provider vacuum LDOS cannot enter STM/STS.")
        energy = vacuum_ldos.request.energy
        offsets = energy[None, :, None] - self.biases[None, None, :]
        x = offsets / (2.0 * self.temperature)
        minus_fermi_derivative = 1.0 / (4.0 * self.temperature * jnp.cosh(x) ** 2)
        didv = self.current_scale * jnp.trapezoid(
            vacuum_ldos.ldos[:, :, None] * minus_fermi_derivative,
            energy,
            axis=1,
        )
        delta_bias = jnp.diff(self.biases)
        trapezoids = 0.5 * (didv[:, 1:] + didv[:, :-1]) * delta_bias[None, :]
        cumulative = jnp.concatenate(
            (
                jnp.zeros((didv.shape[0], 1), dtype=didv.dtype),
                jnp.cumsum(trapezoids, axis=1),
            ),
            axis=1,
        )
        zero_index = int(jnp.argmax(self.biases == 0.0))
        current = cumulative - cumulative[:, zero_index : zero_index + 1]
        numerical_derivative = jnp.gradient(current, self.biases, axis=-1)
        denominator = jnp.maximum(jnp.max(jnp.abs(didv)), jnp.finfo(didv.dtype).tiny)
        closure = jnp.max(jnp.abs(numerical_derivative - didv)) / denominator
        passivity = jnp.maximum(-jnp.min(didv), 0.0)
        successful = (
            bool(passivity <= self.closure_tolerance)
            and bool(closure <= self.closure_tolerance)
            and bool(vacuum_ldos.moment_residual <= self.closure_tolerance)
        )
        evidence = TersoffHamannEvidence(
            passivity, closure, vacuum_ldos.moment_residual, jnp.asarray(successful)
        )
        raw_evidence = SpectralResponseEvidence(
            passivity, vacuum_ldos.moment_residual, 0.0, 0.0, successful
        )
        raw = SpectralResponseProduct(
            self.biases,
            didv,
            jnp.ones(self.biases.shape, dtype=jnp.bool_),
            bias_unit,
            response_unit,
            tuple(f"tip[{index}]" for index in range(didv.shape[0])),
            SpectralResponseRepresentation.DENSITY,
            "provider-vacuum-ldos-ters-off-hamann",
            vacuum_ldos.result_id,
            raw_evidence,
        )
        return TersoffHamannResult(
            raw, current, self.biases, evidence, vacuum_ldos.result_id
        )


__all__ = [
    "ARPESEvidence",
    "ARPESPlan",
    "ARPESResult",
    "PhotoemissionMatrixElementRequest",
    "PhotoemissionMatrixElementResult",
    "TersoffHamannEvidence",
    "TersoffHamannPlan",
    "TersoffHamannResult",
    "VacuumLDOSRequest",
    "VacuumLDOSResult",
]

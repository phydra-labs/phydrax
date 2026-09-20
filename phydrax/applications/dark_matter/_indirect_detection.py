#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..astrophysics._operators import (
    BinnedResponsePlan,
    BinnedResponseResult,
    SpectralField,
)
from ..astrophysics._photometry import ObservationDataProvenance
from ._yields import (
    AnnihilationProcessDescriptor,
    DecayProcessDescriptor,
    ParticleYieldSpectrum,
)


class IndirectDetectionStatus(IntEnum):
    SUCCESS = 0
    NONPHYSICAL_INPUT = 1
    NUMERICAL_FAILURE = 2


def _factor_arrays(
    value: ArrayLike, standard_deviation: ArrayLike, name: str, /
) -> tuple[Array, Array]:
    host = np.asarray(value, dtype=np.float64)
    uncertainty_host = np.asarray(standard_deviation, dtype=np.float64)
    if host.shape != uncertainty_host.shape:
        raise ValueError(f"{name} values and standard deviations must have equal shape.")
    if (
        np.any(~np.isfinite(host))
        or np.any(~np.isfinite(uncertainty_host))
        or np.any(host < 0.0)
        or np.any(uncertainty_host < 0.0)
    ):
        raise ValueError(
            f"{name} values and uncertainties must be finite and non-negative."
        )
    return jnp.asarray(value), jnp.asarray(standard_deviation)


class JFactor(StrictModule, NonTrainableState):
    """Line-of-sight integral of density squared in GeV^2 cm^-5."""

    value_gev2_cm5: Array
    standard_deviation_gev2_cm5: Array
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        value_gev2_cm5: ArrayLike,
        standard_deviation_gev2_cm5: ArrayLike,
        /,
        *,
        target_id: str,
    ):
        value, uncertainty = _factor_arrays(
            value_gev2_cm5, standard_deviation_gev2_cm5, "J factor"
        )
        identifier = str(target_id).strip()
        if not identifier:
            raise ValueError("target_id must be non-empty.")
        self.value_gev2_cm5 = value
        self.standard_deviation_gev2_cm5 = uncertainty
        self.target_id = canonical_fingerprint(
            {
                "kind": "annihilation-j-factor",
                "name": identifier,
                "unit": "GeV^2 cm^-5",
                "arrays": array_tree_fingerprint((value, uncertainty)),
            }
        )


class DFactor(StrictModule, NonTrainableState):
    """Line-of-sight integral of density in GeV cm^-2."""

    value_gev_cm2: Array
    standard_deviation_gev_cm2: Array
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        value_gev_cm2: ArrayLike,
        standard_deviation_gev_cm2: ArrayLike,
        /,
        *,
        target_id: str,
    ):
        value, uncertainty = _factor_arrays(
            value_gev_cm2, standard_deviation_gev_cm2, "D factor"
        )
        identifier = str(target_id).strip()
        if not identifier:
            raise ValueError("target_id must be non-empty.")
        self.value_gev_cm2 = value
        self.standard_deviation_gev_cm2 = uncertainty
        self.target_id = canonical_fingerprint(
            {
                "kind": "decay-d-factor",
                "name": identifier,
                "unit": "GeV cm^-2",
                "arrays": array_tree_fingerprint((value, uncertainty)),
            }
        )


class ExactLineFluxTable(StrictModule, NonTrainableState):
    energy_gev: Array
    integrated_flux_cm2_s: Array
    standard_deviation_cm2_s: Array
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_gev: ArrayLike,
        integrated_flux_cm2_s: ArrayLike,
        standard_deviation_cm2_s: ArrayLike,
        /,
    ):
        energies = np.asarray(energy_gev, dtype=np.float64)
        values = np.asarray(integrated_flux_cm2_s, dtype=np.float64)
        uncertainty = np.asarray(standard_deviation_cm2_s, dtype=np.float64)
        if (
            energies.ndim != 1
            or values.shape[-1:] != energies.shape
            or uncertainty.shape != values.shape
        ):
            raise ValueError("Line flux arrays must end in the exact line-energy axis.")
        if (
            np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(values))
            or np.any(~np.isfinite(uncertainty))
            or np.any(energies <= 0.0)
            or np.any(values < 0.0)
            or np.any(uncertainty < 0.0)
        ):
            raise ValueError("Line flux values must be finite and non-negative.")
        self.energy_gev = jnp.asarray(energy_gev)
        self.integrated_flux_cm2_s = jnp.asarray(integrated_flux_cm2_s)
        self.standard_deviation_cm2_s = jnp.asarray(standard_deviation_cm2_s)
        self.table_id = canonical_fingerprint(
            {
                "kind": "exact-line-flux-table",
                "units": {"energy": "GeV", "flux": "cm^-2 s^-1"},
                "arrays": array_tree_fingerprint((energies, values, uncertainty)),
            }
        )


class FluxNormalizationEvidence(StrictModule, NonTrainableState):
    particle_physics_coefficient: Array
    target_factor: Array
    target_standard_deviation: Array
    continuum_nonnegative: Array
    lines_nonnegative: Array
    valid: Array
    convention: str = eqx.field(static=True)


class IndirectFluxResult(StrictModule, NonTrainableState):
    continuum: SpectralField
    continuum_standard_deviation: Array
    lines: ExactLineFluxTable
    evidence: FluxNormalizationEvidence
    valid: Array
    status: Array
    result_id: str = eqx.field(static=True)


def _flux_result(
    coefficient: Array,
    target: Array,
    target_standard_deviation: Array,
    spectrum: ParticleYieldSpectrum,
    /,
    *,
    convention: str,
    identity: dict[str, str],
) -> IndirectFluxResult:
    continuum_values = coefficient * target[..., None] * spectrum.continuum.values
    continuum_variance = (
        coefficient
        * target[..., None]
        * spectrum.uncertainty.continuum_standard_deviation
    ) ** 2 + (
        coefficient * target_standard_deviation[..., None] * spectrum.continuum.values
    ) ** 2
    line_values = coefficient * target[..., None] * spectrum.lines.multiplicity
    line_variance = (
        coefficient * target[..., None] * spectrum.uncertainty.line_standard_deviation
    ) ** 2 + (
        coefficient * target_standard_deviation[..., None] * spectrum.lines.multiplicity
    ) ** 2
    result_id = canonical_fingerprint(
        {
            "kind": "dark-matter-indirect-flux",
            "convention": convention,
            **identity,
        }
    )
    provenance = ObservationDataProvenance.native(result_id)
    continuum = SpectralField(
        spectrum.continuum.coordinate,
        continuum_values,
        provenance,
        coordinate_unit="GeV",
        value_unit="cm^-2 s^-1 GeV^-1",
        field_id=result_id,
    )
    lines = ExactLineFluxTable(
        spectrum.lines.energy_gev,
        line_values,
        jnp.sqrt(line_variance),
    )
    continuum_nonnegative = jnp.all(jnp.isfinite(continuum_values)) & jnp.all(
        continuum_values >= 0.0
    )
    lines_nonnegative = jnp.all(jnp.isfinite(line_values)) & jnp.all(line_values >= 0.0)
    valid = continuum_nonnegative & lines_nonnegative
    status = jnp.where(
        valid,
        int(IndirectDetectionStatus.SUCCESS),
        int(IndirectDetectionStatus.NUMERICAL_FAILURE),
    ).astype(jnp.int32)
    evidence = FluxNormalizationEvidence(
        coefficient,
        target,
        target_standard_deviation,
        continuum_nonnegative,
        lines_nonnegative,
        valid,
        convention,
    )
    return IndirectFluxResult(
        continuum,
        jnp.sqrt(continuum_variance),
        lines,
        evidence,
        valid,
        status,
        result_id,
    )


def annihilation_flux(
    process: AnnihilationProcessDescriptor,
    spectrum: ParticleYieldSpectrum,
    target: JFactor,
    /,
) -> IndirectFluxResult:
    """Compute dPhi/dE using J <sigma v>/(8 pi m^2) for self-conjugate DM."""

    if not isinstance(process, AnnihilationProcessDescriptor):
        raise TypeError("process must be an AnnihilationProcessDescriptor.")
    if not isinstance(spectrum, ParticleYieldSpectrum) or not isinstance(target, JFactor):
        raise TypeError("Annihilation flux requires a particle yield and JFactor.")
    denominator = (8.0 if process.self_conjugate else 16.0) * jnp.pi * process.mass_gev**2
    coefficient = process.velocity_averaged_cross_section_cm3_s / denominator
    convention = (
        "self-conjugate:<sigma-v>*J/(8*pi*m_chi^2)"
        if process.self_conjugate
        else "symmetric-non-self-conjugate:<sigma-v>*J/(16*pi*m_chi^2)"
    )
    return _flux_result(
        coefficient,
        target.value_gev2_cm5,
        target.standard_deviation_gev2_cm5,
        spectrum,
        convention=convention,
        identity={
            "process": process.process_id,
            "yield": spectrum.yield_id,
            "target": target.target_id,
        },
    )


def decay_flux(
    process: DecayProcessDescriptor,
    spectrum: ParticleYieldSpectrum,
    target: DFactor,
    /,
) -> IndirectFluxResult:
    """Compute dPhi/dE using D/(4 pi m tau)."""

    if not isinstance(process, DecayProcessDescriptor):
        raise TypeError("process must be a DecayProcessDescriptor.")
    if not isinstance(spectrum, ParticleYieldSpectrum) or not isinstance(target, DFactor):
        raise TypeError("Decay flux requires a particle yield and DFactor.")
    coefficient = 1.0 / (4.0 * jnp.pi * process.mass_gev * process.lifetime_s)
    return _flux_result(
        coefficient,
        target.value_gev_cm2,
        target.standard_deviation_gev_cm2,
        spectrum,
        convention="decay:D/(4*pi*m_chi*tau)",
        identity={
            "process": process.process_id,
            "yield": spectrum.yield_id,
            "target": target.target_id,
        },
    )


def _piecewise_linear_bin_weights(coordinate: Array, edges: Array, /) -> Array:
    coordinate = eqx.error_if(
        coordinate,
        jnp.any(~jnp.isfinite(coordinate))
        | (edges[0] < coordinate[0])
        | (edges[-1] > coordinate[-1]),
        "Response energy bins lie outside the continuum yield domain.",
    )
    left = coordinate[:-1][None, :]
    right = coordinate[1:][None, :]
    lower = edges[:-1, None]
    upper = edges[1:, None]
    overlap_left = jnp.maximum(left, lower)
    overlap_right = jnp.minimum(right, upper)
    active = overlap_right > overlap_left
    width = right - left
    first_moment = ((overlap_right - left) ** 2 - (overlap_left - left) ** 2) / (
        2.0 * width
    )
    first_moment = jnp.where(active, first_moment, 0.0)
    overlap = jnp.where(active, overlap_right - overlap_left, 0.0)
    left_weight = overlap - first_moment
    right_weight = first_moment
    weights = jnp.zeros((edges.size - 1, coordinate.size), dtype=coordinate.dtype)
    return weights.at[:, :-1].add(left_weight).at[:, 1:].add(right_weight)


class BinnedIndirectDetectionEvidence(StrictModule, NonTrainableState):
    continuum_bin_flux_cm2_s: Array
    line_bin_flux_cm2_s: Array
    source_bin_flux_cm2_s: Array
    response_count: Array
    finite_uncertainty: Array
    valid: Array


class BinnedIndirectDetectionResult(StrictModule, NonTrainableState):
    source_bin_flux_cm2_s: Array
    source_standard_deviation_cm2_s: Array
    response: BinnedResponseResult
    predicted_standard_deviation: Array
    evidence: BinnedIndirectDetectionEvidence
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class BinnedIndirectDetectionPlan(StrictModule, NonTrainableState):
    """Exact piecewise-linear continuum binning, exact lines, then BinnedResponse."""

    energy_bin_edges_gev: Array
    response: BinnedResponsePlan
    exposure_cm2_s: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_bin_edges_gev: ArrayLike,
        response: BinnedResponsePlan,
        /,
        *,
        exposure_cm2_s: ArrayLike,
    ):
        edges = np.asarray(energy_bin_edges_gev, dtype=np.float64)
        exposure = np.asarray(exposure_cm2_s, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(edges <= 0.0)
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("Energy-bin edges must be positive, finite, and increasing.")
        if not isinstance(response, BinnedResponsePlan):
            raise TypeError("response must be a BinnedResponsePlan.")
        if response.matrix.shape[1] != edges.size - 1:
            raise ValueError(
                "BinnedResponse input count must equal the number of energy bins."
            )
        if exposure.shape != () or not np.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("Exposure must be a finite positive scalar in cm^2 s.")
        self.energy_bin_edges_gev = jnp.asarray(energy_bin_edges_gev)
        self.response = response
        self.exposure_cm2_s = jnp.asarray(exposure_cm2_s)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "binned-indirect-detection",
                "energy_bin_edges_gev": array_tree_fingerprint(edges),
                "response": response.plan_id,
                "exposure_cm2_s": array_tree_fingerprint(exposure),
            }
        )

    def evaluate(self, flux: IndirectFluxResult, /) -> BinnedIndirectDetectionResult:
        if not isinstance(flux, IndirectFluxResult):
            raise TypeError("flux must be an IndirectFluxResult.")
        weights = _piecewise_linear_bin_weights(
            flux.continuum.coordinate, self.energy_bin_edges_gev
        )
        continuum = contract("be,...e->...b", weights, flux.continuum.values)
        continuum_variance = contract(
            "be,...e->...b", weights**2, flux.continuum_standard_deviation**2
        )
        line_energy = flux.lines.energy_gev
        lower = self.energy_bin_edges_gev[:-1, None]
        upper = self.energy_bin_edges_gev[1:, None]
        membership = (line_energy[None, :] >= lower) & (line_energy[None, :] < upper)
        membership = membership.at[-1].set(
            (line_energy >= self.energy_bin_edges_gev[-2])
            & (line_energy <= self.energy_bin_edges_gev[-1])
        )
        membership_values = membership.astype(flux.lines.integrated_flux_cm2_s.dtype)
        line = contract(
            "bl,...l->...b", membership_values, flux.lines.integrated_flux_cm2_s
        )
        line_variance = contract(
            "bl,...l->...b",
            membership_values,
            flux.lines.standard_deviation_cm2_s**2,
        )
        source = continuum + line
        source_standard_deviation = jnp.sqrt(continuum_variance + line_variance)
        expected_source_counts = self.exposure_cm2_s * source
        response = self.response.evaluate(expected_source_counts)
        predicted_variance = contract(
            "ob,...b->...o",
            self.response.matrix**2,
            (self.exposure_cm2_s * source_standard_deviation) ** 2,
        )
        predicted_standard_deviation = jnp.sqrt(predicted_variance)
        finite_uncertainty = jnp.all(jnp.isfinite(predicted_standard_deviation), axis=-1)
        valid = flux.valid & response.valid & finite_uncertainty
        status = jnp.where(
            valid,
            int(IndirectDetectionStatus.SUCCESS),
            int(IndirectDetectionStatus.NUMERICAL_FAILURE),
        ).astype(jnp.int32)
        evidence = BinnedIndirectDetectionEvidence(
            continuum,
            line,
            source,
            response.predicted,
            finite_uncertainty,
            valid,
        )
        return BinnedIndirectDetectionResult(
            source,
            source_standard_deviation,
            response,
            predicted_standard_deviation,
            evidence,
            valid,
            status,
            self.plan_id,
        )


__all__ = [
    "BinnedIndirectDetectionEvidence",
    "BinnedIndirectDetectionPlan",
    "BinnedIndirectDetectionResult",
    "DFactor",
    "ExactLineFluxTable",
    "FluxNormalizationEvidence",
    "IndirectDetectionStatus",
    "IndirectFluxResult",
    "JFactor",
    "annihilation_flux",
    "decay_flux",
]

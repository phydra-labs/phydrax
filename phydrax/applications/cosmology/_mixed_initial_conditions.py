#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coupled particle and wave initial conditions with explicit mode identity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from fractions import Fraction
from math import isfinite, pi, prod
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...discretization.spectral._coordinates import HermitianSpectralCoordinates
from ...discretization.spectral._space import TensorSpectralDiscretization
from ...qualification import ReferenceArtifactManifest
from ...stochastic._random_field import GaussianCoefficientRealization
from ...units import derived_unit, DIMENSIONLESS, UnitDefinition
from ._background import FLRWBackground
from ._closure import CosmologyRealizationSignature
from ._initial_conditions import (
    LagrangianInitialConditionResult,
    LagrangianPerturbationInitialConditionPlan,
)
from ._products import (
    CosmologyProductProvenance,
    LagrangianGrowthHistory,
    MatterPowerDescriptor,
    MatterPowerTable,
    TransferGauge,
)
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract
from ._wave_dark_matter import PreparedPeriodicWaveDarkMatter, WaveDarkMatterState


def _canonical_array_payload(kind: str, arrays: Sequence[np.ndarray], /) -> bytes:
    contiguous = tuple(np.ascontiguousarray(value) for value in arrays)
    metadata = {
        "kind": kind,
        "arrays": [
            {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "nbytes": value.nbytes,
            }
            for value in contiguous
        ],
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        len(header).to_bytes(8, "big")
        + header
        + b"".join(value.tobytes(order="C") for value in contiguous)
    )


def component_transfer_payload_bytes(
    scale_factors: ArrayLike,
    wavenumbers: ArrayLike,
    transfer_values: ArrayLike,
    /,
) -> bytes:
    """Return the canonical bytes governed by a component-transfer manifest."""
    scales = np.asarray(scale_factors, dtype=float).reshape((-1,))
    wavenumbers_ = np.asarray(wavenumbers, dtype=float).reshape((-1,))
    values = np.asarray(transfer_values)
    if np.iscomplexobj(values):
        raise TypeError("Component transfer matrices must be real-valued.")
    dtype = np.result_type(scales.dtype, wavenumbers_.dtype, values.dtype)
    return _canonical_array_payload(
        "component-transfer-matrix-v1",
        (
            scales.astype(dtype, copy=False),
            wavenumbers_.astype(dtype, copy=False),
            values.astype(dtype, copy=False),
        ),
    )


def imported_complex_field_payload_bytes(
    psi: ArrayLike,
    scale_factor: ArrayLike,
    declared_mass: ArrayLike,
    /,
) -> bytes:
    """Return canonical field, coordinate-time, and mass bytes for admission."""
    field = np.asarray(psi)
    if not np.issubdtype(field.dtype, np.complexfloating):
        raise TypeError("Imported wavefunction must have a complex dtype.")
    scale = np.asarray(scale_factor, dtype=field.real.dtype)
    mass = np.asarray(declared_mass, dtype=field.real.dtype)
    if scale.shape != () or mass.shape != ():
        raise ValueError("Imported scale factor and declared mass must be scalars.")
    return _canonical_array_payload(
        "imported-complex-wave-field-v1",
        (field, scale, mass),
    )


def _requested_reference_use(
    manifest: ReferenceArtifactManifest,
    /,
    *,
    commercial_use: bool,
    redistribution: bool,
    training_use: bool,
    export: bool,
) -> str:
    flags = (commercial_use, redistribution, training_use, export)
    if any(not isinstance(value, bool) for value in flags):
        raise TypeError("Requested reference-use flags must be booleans.")
    manifest.require_rights(
        commercial_use=commercial_use,
        redistribution=redistribution,
        training_use=training_use,
        export=export,
    )
    return canonical_fingerprint(
        {
            "kind": "reference-use-request",
            "manifest": manifest.manifest_id,
            "commercial_use": commercial_use,
            "redistribution": redistribution,
            "training_use": training_use,
            "export": export,
        }
    )


def _validate_reference_envelope(
    artifact: ScientificArtifactEnvelope,
    manifest: ReferenceArtifactManifest,
    /,
) -> None:
    required_lineage = {manifest.manifest_id, *manifest.lineage_ids}
    if (
        artifact.status != "complete"
        or artifact.content_digest != manifest.checksum
        or artifact.license_id != manifest.license_id
        or not required_lineage.issubset(artifact.parent_artifact_ids)
    ):
        raise ValueError(
            "Reference manifest and artifact digest/license/lineage disagree."
        )


def _verify_reference_binding(
    artifact: ScientificArtifactEnvelope,
    manifest: ReferenceArtifactManifest,
    payload: bytes,
    /,
) -> None:
    _validate_reference_envelope(artifact, manifest)
    manifest.verify_bytes(payload)


def _names(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{owner} must contain unique non-empty names.")
    return result


def _dimensionless_unit(scale: CosmologyScaleContract, /) -> UnitDefinition:
    return UnitDefinition(
        "1",
        DIMENSIONLESS,
        scale.length_unit.reference_system_id,
    )


def _mass_density_unit(
    scale: CosmologyScaleContract, dimension: int, /
) -> UnitDefinition:
    return derived_unit(
        f"{scale.mass_unit.symbol}/{scale.length_unit.symbol}^{dimension}",
        ((scale.mass_unit, 1), (scale.length_unit, -dimension)),
    )


def _mass_current_unit(
    scale: CosmologyScaleContract, dimension: int, /
) -> UnitDefinition:
    return derived_unit(
        (
            f"{scale.mass_unit.symbol}/"
            f"({scale.length_unit.symbol}^{dimension - 1}*{scale.time_unit.symbol})"
        ),
        (
            (scale.mass_unit, 1),
            (scale.length_unit, 1 - dimension),
            (scale.time_unit, -1),
        ),
    )


def _wavefunction_unit(
    scale: CosmologyScaleContract, dimension: int, /
) -> UnitDefinition:
    return derived_unit(
        f"1/{scale.length_unit.symbol}^{dimension}/2",
        ((scale.length_unit, Fraction(-dimension, 2)),),
    )


def _periodic_geometry(
    discretization: TensorSpectralDiscretization, /
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be TensorSpectralDiscretization.")
    if not discretization.axes or any(
        axis.family != "fourier" or not axis.periodic for axis in discretization.axes
    ):
        raise ValueError(
            "Mixed initial conditions require a periodic tensor Fourier grid."
        )
    origins = tuple(float(np.asarray(axis.domain.lower)) for axis in discretization.axes)
    lengths = tuple(float(np.asarray(axis.length)) for axis in discretization.axes)
    volume = float(prod(lengths))
    if not isfinite(volume) or volume <= 0.0:
        raise ValueError("Periodic initial-condition volume is invalid.")
    return origins, lengths, volume


def _wavenumber_components(
    discretization: TensorSpectralDiscretization,
    /,
    *,
    gradient_compatible: bool,
) -> tuple[Array, ...]:
    rank = len(discretization.axes)
    dtype = jnp.empty(
        (), dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype)
    ).real.dtype
    values = []
    for axis_index, axis in enumerate(discretization.axes):
        component = (
            2.0
            * jnp.pi
            * axis.modes.mode_numbers.astype(dtype)
            / axis.length.astype(dtype)
        )
        if gradient_compatible:
            component = jnp.where(axis.modes.nyquist_mask, 0.0, component)
        shape = [1] * rank
        shape[axis_index] = axis.mode_count
        values.append(
            jnp.broadcast_to(component.reshape(tuple(shape)), discretization.modal_shape)
        )
    return tuple(values)


def _wavenumber_magnitude(discretization: TensorSpectralDiscretization, /) -> Array:
    components = _wavenumber_components(discretization, gradient_compatible=False)
    return jnp.sqrt(sum(component * component for component in components))


def _coupled_mode_mask(discretization: TensorSpectralDiscretization, /) -> Array:
    mask = _wavenumber_magnitude(discretization) > 0.0
    rank = len(discretization.axes)
    for axis_index, axis in enumerate(discretization.axes):
        shape = [1] * rank
        shape[axis_index] = axis.mode_count
        mask = mask & ~jnp.broadcast_to(
            axis.modes.nyquist_mask.reshape(tuple(shape)),
            discretization.modal_shape,
        )
    return mask


def _hermitian_residual(
    discretization: TensorSpectralDiscretization, modes: Array, /
) -> Array:
    conjugate_partner = modes
    for axis_index, axis in enumerate(discretization.axes):
        conjugate_partner = jnp.take(
            conjugate_partner,
            axis.modes.conjugate_indices,
            axis=axis_index,
        )
    return jnp.max(jnp.abs(modes - jnp.conj(conjugate_partner)), initial=0.0)


def _weighted_sum(
    discretization: TensorSpectralDiscretization, values: Array, /
) -> Array:
    weights = discretization.quadrature_weights
    while weights.ndim < values.ndim:
        weights = weights[..., None]
    weights = jnp.broadcast_to(weights, values.shape)
    return jnp.real(ein.contract("i,i->", weights.reshape((-1,)), values.reshape((-1,))))


def _weighted_l2(discretization: TensorSpectralDiscretization, values: Array, /) -> Array:
    return jnp.sqrt(jnp.maximum(_weighted_sum(discretization, jnp.abs(values) ** 2), 0.0))


def _relative_residual(numerator: Array, denominator: Array, /) -> Array:
    safe = jnp.where(denominator > 0.0, denominator, jnp.ones_like(denominator))
    return jnp.where(denominator > 0.0, numerator / safe, numerator)


def _square_contour_indices(
    discretization: TensorSpectralDiscretization,
    center: tuple[float, ...],
    transverse: tuple[int, int],
    /,
) -> tuple[int, ...]:
    left, right = transverse
    counts = discretization.physical_shape
    radius = max(1, min(counts[left], counts[right]) // 4)
    center_indices = [
        int(np.argmin(np.abs(np.asarray(axis.nodes) - center[axis_index])))
        for axis_index, axis in enumerate(discretization.axes)
    ]
    offsets = [
        *((value, -radius) for value in range(-radius, radius)),
        *((radius, value) for value in range(-radius, radius)),
        *((value, radius) for value in range(radius, -radius, -1)),
        *((-radius, value) for value in range(radius, -radius, -1)),
    ]
    flat = []
    for left_offset, right_offset in offsets:
        index = list(center_indices)
        index[left] = (index[left] + left_offset) % counts[left]
        index[right] = (index[right] + right_offset) % counts[right]
        flat.append(int(np.ravel_multi_index(tuple(index), counts)))
    return tuple(flat)


def _contour_winding(field: Array, indices: tuple[int, ...], /) -> Array:
    contour = field.reshape((-1,))[jnp.asarray(indices, dtype=jnp.int32)]
    unit = contour / jnp.where(jnp.abs(contour) > 0.0, jnp.abs(contour), 1.0)
    increments = jnp.angle(jnp.roll(unit, -1) * jnp.conj(unit))
    return jnp.sum(increments) / (2.0 * jnp.pi)


def _semantic_coordinate_layout(
    discretization: TensorSpectralDiscretization,
    components: tuple[str, ...],
    /,
) -> tuple[HermitianSpectralCoordinates, tuple[str, ...], np.ndarray]:
    coordinates = discretization.real_coordinates(component_shape=(len(components),))
    fixed = np.asarray(coordinates.fixed_indices, dtype=np.int64)
    representatives = np.asarray(coordinates.representative_indices, dtype=np.int64)
    state_indices = np.concatenate((fixed, representatives, representatives))
    parts = (
        ("real",) * fixed.size
        + ("real",) * representatives.size
        + ("imaginary",) * representatives.size
    )
    mode_numbers = tuple(
        np.asarray(axis.modes.mode_numbers, dtype=np.int64)
        for axis in discretization.axes
    )
    nyquist = tuple(
        np.asarray(axis.modes.nyquist_mask, dtype=bool) for axis in discretization.axes
    )
    component_count = len(components)
    identifiers: list[str] = []
    positions: list[int] = []
    for coordinate_index, (state_index, part) in enumerate(
        zip(state_indices, parts, strict=True)
    ):
        modal_flat = int(state_index) // component_count
        component_index = int(state_index) % component_count
        multi = np.unravel_index(modal_flat, discretization.modal_shape)
        wave = tuple(
            int(mode_numbers[axis_index][index]) for axis_index, index in enumerate(multi)
        )
        is_zero = all(value == 0 for value in wave)
        is_nyquist = any(
            bool(nyquist[axis_index][index]) for axis_index, index in enumerate(multi)
        )
        if is_zero or is_nyquist:
            continue
        wave_label = ",".join(str(value) for value in wave)
        identifiers.append(
            f"cosmology-primordial:{components[component_index]}:k={wave_label}:{part}"
        )
        positions.append(coordinate_index)
    if not identifiers:
        raise ValueError(
            "Primordial realization grid has no shared nonzero non-Nyquist modes."
        )
    return coordinates, tuple(identifiers), np.asarray(positions, dtype=np.int32)


class PrimordialModeRealization(StrictModule):
    """Hermitian primordial modes derived from stable Gaussian mode identities.

    Mean and Nyquist modes are deliberately absent.  Consequently a finer realization
    can be restricted to a coarser Fourier grid by semantic mode ID without assigning a
    different random number to any shared resolvable mode.
    """

    modes: Array
    wavenumber_magnitude: Array
    active_mode_mask: Array
    gaussian: GaussianCoefficientRealization
    discretization: TensorSpectralDiscretization
    scale: CosmologyScaleContract
    primordial_components: tuple[str, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    hermitian_residual: Array
    finite: Array
    successful: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=("accepted", "non_finite", "non_hermitian"),
    )

    @classmethod
    def required_mode_ids(
        cls,
        discretization: TensorSpectralDiscretization,
        primordial_components: Sequence[str],
        /,
    ) -> tuple[str, ...]:
        components = _names(primordial_components, "primordial_components")
        _periodic_geometry(discretization)
        _, identifiers, _ = _semantic_coordinate_layout(discretization, components)
        return identifiers

    @classmethod
    def from_gaussian_modes(
        cls,
        discretization: TensorSpectralDiscretization,
        gaussian: GaussianCoefficientRealization,
        primordial_components: Sequence[str],
        /,
        *,
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
    ) -> PrimordialModeRealization:
        if not isinstance(gaussian, GaussianCoefficientRealization):
            raise TypeError("gaussian must be GaussianCoefficientRealization.")
        if gaussian.sample_shape != ():
            raise ValueError(
                "Primordial Gaussian modes must have scalar sample shape; component and "
                "spectral identities are already encoded by mode_ids."
            )
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be CosmologyScaleContract.")
        components = _names(primordial_components, "primordial_components")
        _periodic_geometry(discretization)
        coordinates, identifiers, positions = _semantic_coordinate_layout(
            discretization, components
        )
        selected = gaussian.select(identifiers)
        real_coordinates = jnp.zeros(
            (coordinates.coordinate_size,), dtype=coordinates.coordinate_space.dtype
        )
        real_coordinates = real_coordinates.at[jnp.asarray(positions)].set(
            selected.coefficients
        )
        modes = coordinates.from_real_coordinates(real_coordinates)
        residual = coordinates.reality_defect(modes)
        finite = jnp.all(jnp.isfinite(modes))
        hermitian = residual <= coordinates.reality_tolerance
        successful = finite & hermitian
        status = jnp.where(~finite, 1, jnp.where(~hermitian, 2, 0)).astype(jnp.int32)
        realization_id = canonical_fingerprint(
            {
                "kind": "primordial-mode-realization",
                "gaussian": selected.realization_id,
                "coupling": selected.coupling_id,
                "discretization": discretization.prepared_id,
                "scale": scale.scale_id,
                "components": list(components),
                "mode_ids": list(identifiers),
                "coordinate_map": coordinates.coordinate_id,
            }
        )
        return cls(
            modes=modes,
            wavenumber_magnitude=_wavenumber_magnitude(discretization),
            active_mode_mask=_coupled_mode_mask(discretization),
            gaussian=selected,
            discretization=discretization,
            scale=scale,
            primordial_components=components,
            mode_ids=identifiers,
            source_realization_id=selected.realization_id,
            coupling_id=selected.coupling_id,
            realization_id=realization_id,
            hermitian_residual=residual,
            finite=finite,
            successful=successful,
            status=status,
        )

    def at_resolution(
        self, discretization: TensorSpectralDiscretization, /
    ) -> PrimordialModeRealization:
        """Restrict the same semantic Gaussian modes to another periodic resolution."""
        return PrimordialModeRealization.from_gaussian_modes(
            discretization,
            self.gaussian,
            self.primordial_components,
            scale=self.scale,
        )


class ComponentModeRealization(StrictModule):
    """Correlated component modes and their exact matrix auto/cross spectrum."""

    modes: Array
    covariance: Array
    mode_mask: Array
    scale_factor: Array
    primordial: PrimordialModeRealization
    components: tuple[str, ...] = eqx.field(static=True)
    component_units: tuple[UnitDefinition, ...] = eqx.field(static=True)
    gauge: TransferGauge = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    artifact_id: str | None = eqx.field(static=True)
    manifest_id: str | None = eqx.field(static=True)
    requested_use_id: str = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    hermitian_residual: Array
    finite: Array
    successful: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "primordial_modes_failed",
            "non_finite",
            "non_hermitian",
        ),
    )

    def component_index(self, component: str, /) -> int:
        name = str(component).strip()
        if name not in self.components:
            raise ValueError(f"Unknown component {name!r}.")
        return self.components.index(name)

    def component_modes(self, component: str, /) -> Array:
        return self.modes[..., self.component_index(component)]

    def auto_power(self, component: str, /) -> Array:
        index = self.component_index(component)
        return self.covariance[..., index, index]

    def cross_power(self, left: str, right: str, /) -> Array:
        return self.covariance[
            ..., self.component_index(left), self.component_index(right)
        ]

    def density_contrast(self, component: str, /) -> Array:
        return self.primordial.discretization.reconstruct(
            self.component_modes(component), real_output=True
        )


class ComponentTransferMatrixProduct(StrictModule):
    """Immutable component-by-primordial transfer factor on ``(a, k)`` nodes.

    ``transfer_values[c, p, a, k]`` carries square-root power-spectrum units.  For
    unit-covariance primordial modes the exact component covariance is ``T T^T``;
    no cross term is discarded or reconstructed from correlation coefficients.
    """

    scale_factors: Array
    wavenumbers: Array
    transfer_values: Array
    components: tuple[str, ...] = eqx.field(static=True)
    primordial_components: tuple[str, ...] = eqx.field(static=True)
    component_units: tuple[UnitDefinition, ...] = eqx.field(static=True)
    transfer_units: tuple[UnitDefinition, ...] = eqx.field(static=True)
    gauge: TransferGauge = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature
    artifact: ScientificArtifactEnvelope | None
    manifest: ReferenceArtifactManifest | None
    requested_use_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        transfer_values: ArrayLike,
        *,
        components: Sequence[str],
        primordial_components: Sequence[str],
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        artifact: ScientificArtifactEnvelope | None,
        gauge: TransferGauge,
        manifest: ReferenceArtifactManifest | None = None,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
        normalization: str = "dimensionless-density-contrast",
        component_units: Sequence[UnitDefinition] | None = None,
        spatial_dimension: int = 3,
    ):
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be CosmologyScaleContract.")
        if not isinstance(provenance, CosmologyProductProvenance):
            raise TypeError("provenance must be CosmologyProductProvenance.")
        if not isinstance(realization, CosmologyRealizationSignature):
            raise TypeError("realization must be CosmologyRealizationSignature.")
        if artifact is not None and not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope or None.")
        if manifest is not None and not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("manifest must be ReferenceArtifactManifest or None.")
        if (
            scale.scale_id != provenance.scale_id
            or scale.scale_id != realization.scale_id
        ):
            raise ValueError("Component transfer scale identities disagree.")
        if gauge not in ("synchronous", "newtonian", "gauge-invariant"):
            raise ValueError("Unknown component transfer gauge.")
        normalization_ = str(normalization).strip()
        if not normalization_:
            raise ValueError("Component transfer normalization must be non-empty.")
        dimension = int(spatial_dimension)
        if dimension not in (1, 2, 3):
            raise ValueError("Component transfer dimension must be 1, 2, or 3.")
        component_names = _names(components, "components")
        primordial_names = _names(primordial_components, "primordial_components")
        units = (
            tuple(_dimensionless_unit(scale) for _ in component_names)
            if component_units is None
            else tuple(component_units)
        )
        if len(units) != len(component_names) or not all(
            isinstance(unit, UnitDefinition) for unit in units
        ):
            raise ValueError(
                "component_units must provide one UnitDefinition per component."
            )
        if any(
            unit.reference_system_id != scale.length_unit.reference_system_id
            for unit in units
        ):
            raise ValueError("Component transfer units and cosmology scale disagree.")
        scales_host = np.asarray(scale_factors, dtype=float).reshape((-1,))
        wavenumbers_host = np.asarray(wavenumbers, dtype=float).reshape((-1,))
        values_host = np.asarray(transfer_values)
        if np.iscomplexobj(values_host):
            raise TypeError("Component transfer matrices must be real-valued.")
        expected = (
            len(component_names),
            len(primordial_names),
            scales_host.size,
            wavenumbers_host.size,
        )
        if (
            scales_host.size < 2
            or wavenumbers_host.size < 2
            or np.any(~np.isfinite(scales_host))
            or np.any(~np.isfinite(wavenumbers_host))
            or np.any(scales_host <= 0.0)
            or np.any(wavenumbers_host <= 0.0)
            or np.any(np.diff(scales_host) <= 0.0)
            or np.any(np.diff(wavenumbers_host) <= 0.0)
            or values_host.shape != expected
            or np.any(~np.isfinite(values_host))
        ):
            raise ValueError(
                "Component transfer nodes and matrix values are finite, ordered, and shape-exact."
            )
        use_flags = (commercial_use, redistribution, training_use, export)
        if any(not isinstance(value, bool) for value in use_flags):
            raise TypeError("Requested reference-use flags must be booleans.")
        payload = component_transfer_payload_bytes(
            scales_host,
            wavenumbers_host,
            values_host,
        )
        if provenance.source_kind == "external":
            if artifact is None or manifest is None:
                raise ValueError(
                    "External component transfers require an artifact and rights manifest."
                )
            if (
                artifact.producer != provenance.producer
                or artifact.producer_version != provenance.producer_version
            ):
                raise ValueError(
                    "Component transfer provider and artifact identities disagree."
                )
            if not {
                artifact.artifact_id,
                manifest.manifest_id,
            }.issubset(provenance.parent_product_ids):
                raise ValueError(
                    "External transfer provenance must bind artifact and manifest IDs."
                )
            requested_use_id = _requested_reference_use(
                manifest,
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )
            _verify_reference_binding(artifact, manifest, payload)
        else:
            if manifest is not None or any(use_flags):
                raise ValueError(
                    "Native component transfers cannot claim external manifest rights."
                )
            if artifact is not None and (
                artifact.status != "complete"
                or artifact.producer != provenance.producer
                or artifact.producer_version != provenance.producer_version
                or artifact.artifact_id not in provenance.parent_product_ids
            ):
                raise ValueError(
                    "Native component transfer artifact and provenance disagree."
                )
            requested_use_id = "native-generated"
        dtype = jnp.result_type(jnp.asarray(scales_host), jnp.asarray(values_host))
        scales = jax.lax.stop_gradient(jnp.asarray(scales_host, dtype=dtype))
        wavenumber = jax.lax.stop_gradient(jnp.asarray(wavenumbers_host, dtype=dtype))
        values = jax.lax.stop_gradient(jnp.asarray(values_host, dtype=dtype))
        transfer_units = tuple(
            derived_unit(
                f"{unit.symbol}*{scale.length_unit.symbol}^{dimension}/2",
                ((unit, 1), (scale.length_unit, Fraction(dimension, 2))),
            )
            for unit in units
        )
        product_id = canonical_fingerprint(
            {
                "kind": "component-transfer-matrix-product",
                "components": list(component_names),
                "primordial_components": list(primordial_names),
                "component_units": [unit.unit_id for unit in units],
                "transfer_units": [unit.unit_id for unit in transfer_units],
                "gauge": gauge,
                "normalization": normalization_,
                "spatial_dimension": dimension,
                "scale": scale.scale_id,
                "provenance": provenance.provenance_id,
                "realization": realization.content_id(),
                "artifact": None if artifact is None else artifact.artifact_id,
                "manifest": None if manifest is None else manifest.manifest_id,
                "requested_use": requested_use_id,
                "scale_factors": array_tree_fingerprint(scales_host),
                "wavenumbers": array_tree_fingerprint(wavenumbers_host),
                "values": array_tree_fingerprint(values_host),
            }
        )
        self.scale_factors = scales
        self.wavenumbers = wavenumber
        self.transfer_values = values
        self.components = component_names
        self.primordial_components = primordial_names
        self.component_units = units
        self.transfer_units = transfer_units
        self.gauge = gauge
        self.normalization = normalization_
        self.spatial_dimension = dimension
        self.scale = scale
        self.provenance = provenance
        self.realization = realization
        self.artifact = artifact
        self.manifest = manifest
        self.requested_use_id = requested_use_id
        self.product_id = product_id

    @property
    def covariance_values(self) -> Array:
        return ein.contract("cpak,dpak->cdak", self.transfer_values, self.transfer_values)

    def covariance_unit(self, left: str, right: str, /) -> UnitDefinition:
        left_index = self.components.index(str(left).strip())
        right_index = self.components.index(str(right).strip())
        left_unit = self.component_units[left_index]
        right_unit = self.component_units[right_index]
        return derived_unit(
            (
                f"{left_unit.symbol}*{right_unit.symbol}*"
                f"{self.scale.length_unit.symbol}^{self.spatial_dimension}"
            ),
            (
                (left_unit, 1),
                (right_unit, 1),
                (self.scale.length_unit, self.spatial_dimension),
            ),
        )

    def _query(
        self, wavenumber: ArrayLike, scale_factor: ArrayLike, /
    ) -> tuple[Array, Array]:
        query_k = jnp.asarray(wavenumber, dtype=self.wavenumbers.dtype)
        query_a = jnp.asarray(scale_factor, dtype=self.scale_factors.dtype)
        if query_a.shape != ():
            raise ValueError("Component transfer scale-factor query must be scalar.")
        invalid = (
            jnp.any(~jnp.isfinite(query_k))
            | jnp.any(query_k < self.wavenumbers[0])
            | jnp.any(query_k > self.wavenumbers[-1])
            | ~jnp.isfinite(query_a)
            | (query_a < self.scale_factors[0])
            | (query_a > self.scale_factors[-1])
        )
        query_k = eqx.error_if(
            query_k, invalid, "Component transfer query is outside the tabulated support."
        )
        return query_k, query_a

    def evaluate_matrix(self, wavenumber: ArrayLike, scale_factor: ArrayLike, /) -> Array:
        query_k, query_a = self._query(wavenumber, scale_factor)
        flat_k = query_k.reshape((-1,))
        rows = []
        for component_index in range(len(self.components)):
            columns = []
            for primordial_index in range(len(self.primordial_components)):
                at_each_scale = jax.vmap(
                    lambda row: linear_interpolate(self.wavenumbers, row, flat_k).values
                )(self.transfer_values[component_index, primordial_index])
                evaluated = jax.vmap(
                    lambda column: (
                        linear_interpolate(self.scale_factors, column, query_a).values
                    ),
                    in_axes=1,
                    out_axes=0,
                )(at_each_scale)
                columns.append(evaluated.reshape(query_k.shape))
            rows.append(jnp.stack(tuple(columns), axis=-1))
        return jnp.stack(tuple(rows), axis=-2)

    def evaluate_covariance(
        self, wavenumber: ArrayLike, scale_factor: ArrayLike, /
    ) -> Array:
        matrix = self.evaluate_matrix(wavenumber, scale_factor)
        return ein.contract("...cp,...dp->...cd", matrix, matrix)

    def realize(
        self,
        primordial: PrimordialModeRealization,
        scale_factor: ArrayLike,
        /,
    ) -> ComponentModeRealization:
        if not isinstance(primordial, PrimordialModeRealization):
            raise TypeError("primordial must be PrimordialModeRealization.")
        if primordial.scale.scale_id != self.scale.scale_id:
            raise ValueError("Primordial and component transfer units disagree.")
        if primordial.primordial_components != self.primordial_components:
            raise ValueError("Primordial component identities or ordering disagree.")
        if len(primordial.discretization.axes) != self.spatial_dimension:
            raise ValueError("Primordial and component transfer dimensions disagree.")
        magnitude = primordial.wavenumber_magnitude
        active = primordial.active_mode_mask
        safe = jnp.where(active, magnitude, self.wavenumbers[0])
        matrix = self.evaluate_matrix(safe, scale_factor)
        matrix = jnp.where(active[..., None, None], matrix, 0.0)
        covariance = ein.contract("...cp,...dp->...cd", matrix, matrix)
        modes = ein.contract("...cp,...p->...c", matrix, primordial.modes)
        residual = _hermitian_residual(primordial.discretization, modes)
        finite = jnp.all(jnp.isfinite(modes)) & jnp.all(jnp.isfinite(covariance))
        tolerance = (
            64.0
            * jnp.finfo(modes.real.dtype).eps
            * prod(primordial.discretization.modal_shape)
        )
        hermitian = residual <= tolerance
        successful = primordial.successful & finite & hermitian
        status = jnp.where(
            ~primordial.successful,
            1,
            jnp.where(~finite, 2, jnp.where(~hermitian, 3, 0)),
        ).astype(jnp.int32)
        return ComponentModeRealization(
            modes=modes,
            covariance=covariance,
            mode_mask=active,
            scale_factor=jnp.asarray(scale_factor, dtype=magnitude.dtype),
            primordial=primordial,
            components=self.components,
            component_units=self.component_units,
            gauge=self.gauge,
            normalization=self.normalization,
            product_id=self.product_id,
            provenance_id=self.provenance.provenance_id,
            artifact_id=None if self.artifact is None else self.artifact.artifact_id,
            manifest_id=None if self.manifest is None else self.manifest.manifest_id,
            requested_use_id=self.requested_use_id,
            source_realization_id=primordial.realization_id,
            coupling_id=primordial.coupling_id,
            hermitian_residual=residual,
            finite=finite,
            successful=successful,
            status=status,
        )


class ParticleInitialConditionProjection(StrictModule):
    component: str = eqx.field(static=True)
    initial_conditions: LagrangianInitialConditionResult
    target_modes: Array
    recovered_modes: Array
    mode_relative_residual: Array
    particle_mass: Array
    target_particle_mass: Array
    mass_relative_residual: Array
    source_product_id: str = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    finite: Array
    successful: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "component_modes_failed",
            "lpt_failed",
            "non_finite",
            "mode_reconstruction_unclosed",
            "particle_mass_unclosed",
        ),
    )

    @property
    def state(self):
        return self.initial_conditions.state


class MixedInitialConditionPlan(StrictModule, NonTrainableState):
    """Bind one correlated component product to the existing periodic LPT owner."""

    transfer: ComponentTransferMatrixProduct
    particle_lpt: LagrangianPerturbationInitialConditionPlan
    particle_component: str = eqx.field(static=True)
    particle_target_mass: float = eqx.field(static=True)
    gauge: TransferGauge = eqx.field(static=True)
    mode_relative_tolerance: float = eqx.field(static=True)
    mass_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: ComponentTransferMatrixProduct,
        particle_lpt: LagrangianPerturbationInitialConditionPlan,
        /,
        *,
        particle_component: str = "cold_baryon",
        particle_target_mass: float | None = None,
        gauge: TransferGauge | None = None,
        mode_relative_tolerance: float = 1.0e-9,
        mass_relative_tolerance: float = 1.0e-12,
    ):
        if not isinstance(transfer, ComponentTransferMatrixProduct):
            raise TypeError("transfer must be ComponentTransferMatrixProduct.")
        if not isinstance(particle_lpt, LagrangianPerturbationInitialConditionPlan):
            raise TypeError(
                "particle_lpt must be LagrangianPerturbationInitialConditionPlan."
            )
        component = str(particle_component).strip()
        if component not in transfer.components:
            raise ValueError("particle_component is absent from the transfer product.")
        if component != "cold_baryon":
            raise ValueError("Existing LPT projection is qualified only for cold_baryon.")
        if particle_lpt.scale.scale_id != transfer.scale.scale_id:
            raise ValueError("Particle LPT and component transfer units disagree.")
        if len(particle_lpt.shape) != transfer.spatial_dimension:
            raise ValueError("Particle LPT and component transfer dimensions disagree.")
        selected_gauge = transfer.gauge if gauge is None else gauge
        if selected_gauge != transfer.gauge:
            raise ValueError("Particle LPT and component transfer gauges disagree.")
        unit = transfer.component_units[transfer.components.index(component)]
        if unit.unit_id != _dimensionless_unit(transfer.scale).unit_id:
            raise ValueError(
                "Particle LPT requires dimensionless density-contrast component units."
            )
        current_mass = float(
            np.sum(
                np.asarray(particle_lpt.particles.safe_masses)[
                    np.asarray(particle_lpt.particles.active_mask, dtype=bool)
                ]
            )
        )
        target_mass = (
            current_mass if particle_target_mass is None else float(particle_target_mass)
        )
        mode_tolerance = float(mode_relative_tolerance)
        mass_tolerance = float(mass_relative_tolerance)
        if (
            not isfinite(target_mass)
            or target_mass <= 0.0
            or not isfinite(mode_tolerance)
            or mode_tolerance < 0.0
            or not isfinite(mass_tolerance)
            or mass_tolerance < 0.0
        ):
            raise ValueError(
                "Mixed initial-condition tolerances and target mass are invalid."
            )
        self.transfer = transfer
        self.particle_lpt = particle_lpt
        self.particle_component = component
        self.particle_target_mass = target_mass
        self.gauge = selected_gauge
        self.mode_relative_tolerance = mode_tolerance
        self.mass_relative_tolerance = mass_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mixed-initial-condition-plan",
                "transfer": transfer.product_id,
                "particle_lpt": particle_lpt.plan_id,
                "particle_component": component,
                "particle_target_mass": target_mass,
                "gauge": selected_gauge,
                "mode_relative_tolerance": mode_tolerance,
                "mass_relative_tolerance": mass_tolerance,
            }
        )

    def realize_modes(
        self,
        primordial: PrimordialModeRealization,
        scale_factor: ArrayLike,
        /,
    ) -> ComponentModeRealization:
        if primordial.discretization.physical_shape != self.particle_lpt.shape:
            raise ValueError("Primordial and particle LPT resolutions disagree.")
        lengths = jnp.stack(tuple(axis.length for axis in primordial.discretization.axes))
        expected = jnp.asarray(self.particle_lpt.box_size, dtype=lengths.dtype)
        checked_modes = eqx.error_if(
            primordial.modes,
            jnp.any(lengths != expected),
            "Primordial and particle LPT periodic boxes disagree.",
        )
        checked = eqx.tree_at(lambda value: value.modes, primordial, checked_modes)
        return self.transfer.realize(checked, scale_factor)

    def _particle_power(self, /) -> MatterPowerTable:
        index = self.transfer.components.index(self.particle_component)
        covariance = self.transfer.covariance_values[index, index]
        descriptor = MatterPowerDescriptor(
            "cold_baryon",
            "cold_baryon",
            gauge=self.transfer.gauge,
            normalization=self.transfer.normalization,
            spatial_dimension=self.transfer.spatial_dimension,
        )
        return MatterPowerTable(
            self.transfer.scale_factors,
            self.transfer.wavenumbers,
            covariance,
            descriptor,
            self.transfer.scale,
            self.transfer.provenance,
            self.transfer.realization,
        )

    def project_particles(
        self,
        modes: ComponentModeRealization,
        background: FLRWBackground,
        growth: LagrangianGrowthHistory,
        /,
    ) -> ParticleInitialConditionProjection:
        if not isinstance(modes, ComponentModeRealization):
            raise TypeError("modes must be ComponentModeRealization.")
        if modes.product_id != self.transfer.product_id:
            raise ValueError("Component modes do not belong to this mixed IC plan.")
        if modes.gauge != self.gauge:
            raise ValueError("Component modes and mixed IC gauges disagree.")
        discretization = modes.primordial.discretization
        index = modes.component_index(self.particle_component)
        target = modes.modes[..., index]
        power = modes.covariance[..., index, index]
        volume = float(prod(self.particle_lpt.box_size))
        count = prod(discretization.physical_shape)
        scale = jnp.sqrt(power * count / volume)
        nonzero = power > 0.0
        noise_modes = jnp.where(nonzero, target / jnp.where(nonzero, scale, 1.0), 0.0)
        white_noise = discretization.reconstruct(noise_modes, real_output=True)
        result = self.particle_lpt.realize(
            background,
            growth,
            self._particle_power(),
            white_noise,
            modes.scale_factor,
        )
        recovered = discretization.project(result.density_contrast)
        residual = _relative_residual(
            jnp.sqrt(jnp.sum(jnp.abs(recovered - target) ** 2)),
            jnp.sqrt(jnp.sum(jnp.abs(target) ** 2)),
        )
        active = self.particle_lpt.particles.active_mask
        particle_mass = jnp.sum(
            jnp.where(active, self.particle_lpt.particles.safe_masses, 0.0)
        )
        target_mass = jnp.asarray(self.particle_target_mass, dtype=particle_mass.dtype)
        mass_residual = jnp.abs(particle_mass - target_mass) / target_mass
        finite = (
            jnp.isfinite(residual)
            & jnp.isfinite(particle_mass)
            & jnp.isfinite(mass_residual)
        )
        successful = (
            modes.successful
            & result.successful
            & finite
            & (residual <= self.mode_relative_tolerance)
            & (mass_residual <= self.mass_relative_tolerance)
        )
        status = jnp.where(
            ~modes.successful,
            1,
            jnp.where(
                ~result.successful,
                2,
                jnp.where(
                    ~finite,
                    3,
                    jnp.where(
                        residual > self.mode_relative_tolerance,
                        4,
                        jnp.where(mass_residual > self.mass_relative_tolerance, 5, 0),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return ParticleInitialConditionProjection(
            component=self.particle_component,
            initial_conditions=result,
            target_modes=target,
            recovered_modes=recovered,
            mode_relative_residual=residual,
            particle_mass=particle_mass,
            target_particle_mass=target_mass,
            mass_relative_residual=mass_residual,
            source_product_id=modes.product_id,
            source_realization_id=modes.source_realization_id,
            plan_id=self.plan_id,
            finite=finite,
            successful=successful,
            status=status,
        )


class WavePhaseSeedEvidence(StrictModule):
    curl_relative_residual: Array
    current_relative_residual: Array
    density_relative_residual: Array
    phase_gauge_absolute: Array
    minimum_density: Array
    node_free: Array
    de_broglie_nyquist_fraction: Array
    mass: Array
    wave_mass: Array
    mass_relative_residual: Array
    time_level_valid: Array
    finite: Array
    status: Array
    successful: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "non_finite",
            "density_node_or_negative",
            "curl_incompatible",
            "current_projection_incompatible",
            "phase_gauge_unclosed",
            "de_broglie_unresolved",
            "mass_unclosed",
            "invalid_time_level",
        ),
    )


class WavePhaseSeedResult(StrictModule):
    state: WaveDarkMatterState
    density: Array
    target_current: Array
    reconstructed_current: Array
    phase: Array
    evidence: WavePhaseSeedEvidence
    seed_kind: str = eqx.field(static=True, default="irrotational-density-current")
    plan_id: str = eqx.field(static=True, default="")

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class WavePhaseSeedPlan(StrictModule, NonTrainableState):
    """Reconstruct the unique mean-zero periodic phase from mass current.

    The supplied current is comoving mass density times ``dx/dt`` at the declared
    scale factor.  The phase law is ``dx/dt = hbar grad(theta)/(m a^2)``.  Only the
    irrotational, zero-circulation part is representable by a single-valued,
    node-free phase.  Curl, harmonic circulation, and density nodes fail closed.
    """

    prepared: PreparedPeriodicWaveDarkMatter
    density_unit: UnitDefinition = eqx.field(static=True)
    current_unit: UnitDefinition = eqx.field(static=True)
    current_convention: str = eqx.field(static=True)
    fundamental_wavenumber: float = eqx.field(static=True)
    minimum_density: float = eqx.field(static=True)
    curl_relative_tolerance: float = eqx.field(static=True)
    current_relative_tolerance: float = eqx.field(static=True)
    phase_gauge_tolerance: float = eqx.field(static=True)
    mass_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedPeriodicWaveDarkMatter,
        /,
        *,
        minimum_density: float = 0.0,
        curl_relative_tolerance: float = 1.0e-8,
        current_relative_tolerance: float = 1.0e-8,
        phase_gauge_tolerance: float = 1.0e-10,
        mass_relative_tolerance: float = 1.0e-10,
    ):
        if not isinstance(prepared, PreparedPeriodicWaveDarkMatter):
            raise TypeError("prepared must be PreparedPeriodicWaveDarkMatter.")
        dimension = len(prepared.discretization.axes)
        values = tuple(
            float(value)
            for value in (
                minimum_density,
                curl_relative_tolerance,
                current_relative_tolerance,
                phase_gauge_tolerance,
                mass_relative_tolerance,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Wave phase seed thresholds must be finite and nonnegative.")
        self.prepared = prepared
        self.density_unit = _mass_density_unit(prepared.plan.scale, dimension)
        self.current_unit = _mass_current_unit(prepared.plan.scale, dimension)
        self.fundamental_wavenumber = min(
            2.0 * pi / length for length in _periodic_geometry(prepared.discretization)[1]
        )
        self.current_convention = "comoving-mass-current=rho_c*dx/dt"
        (
            self.minimum_density,
            self.curl_relative_tolerance,
            self.current_relative_tolerance,
            self.phase_gauge_tolerance,
            self.mass_relative_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-phase-seed-plan",
                "prepared": prepared.prepared_id,
                "density_unit": self.density_unit.unit_id,
                "current_unit": self.current_unit.unit_id,
                "current_convention": self.current_convention,
                "thresholds": list(values),
                "phase_gauge": "quadrature-mean-zero",
            }
        )

    def realize(
        self,
        density: ArrayLike,
        current: ArrayLike,
        scale_factor: ArrayLike,
        /,
        *,
        density_unit: UnitDefinition,
        current_unit: UnitDefinition,
        current_convention: str,
    ) -> WavePhaseSeedResult:
        if not isinstance(density_unit, UnitDefinition) or not isinstance(
            current_unit, UnitDefinition
        ):
            raise TypeError("Wave phase seed units must be UnitDefinition values.")
        if density_unit.unit_id != self.density_unit.unit_id:
            raise ValueError("Wave phase seed density units disagree.")
        if current_unit.unit_id != self.current_unit.unit_id:
            raise ValueError("Wave phase seed current units disagree.")
        if str(current_convention).strip() != self.current_convention:
            raise ValueError("Wave phase seed current convention disagrees.")
        discretization = self.prepared.discretization
        rho = jnp.asarray(density)
        flux = jnp.asarray(current, dtype=rho.dtype)
        expected = discretization.physical_shape
        dimension = len(expected)
        if rho.shape != expected or flux.shape != expected + (dimension,):
            raise ValueError("Wave phase density/current shapes disagree with the grid.")
        scale = jnp.asarray(scale_factor, dtype=rho.dtype)
        if scale.shape != ():
            raise ValueError("Wave phase seed scale factor must be scalar.")
        scale_matches = jnp.abs(scale - self.prepared.scale_factors[0]) <= (
            32.0 * jnp.finfo(scale.dtype).eps
        )
        safe_scale = jnp.where(
            jnp.isfinite(scale) & (scale > 0.0),
            scale,
            self.prepared.scale_factors[0],
        )
        finite_inputs = (
            jnp.all(jnp.isfinite(rho)) & jnp.all(jnp.isfinite(flux)) & jnp.isfinite(scale)
        )
        minimum = jnp.min(rho)
        node_free = minimum > self.minimum_density
        safe_density = jnp.where(
            rho > self.minimum_density,
            rho,
            jnp.ones_like(rho),
        )
        velocity = flux / safe_density[..., None]
        velocity = jnp.where(node_free & finite_inputs, velocity, 0.0)
        coefficients = jnp.stack(
            tuple(
                discretization.project(velocity[..., axis]) for axis in range(dimension)
            ),
            axis=-1,
        )
        wavevectors = _wavenumber_components(discretization, gradient_compatible=True)
        squared = sum(component * component for component in wavevectors)
        divergence = sum(
            1j * component * coefficients[..., axis]
            for axis, component in enumerate(wavevectors)
        )
        safe_squared = jnp.where(squared > 0.0, squared, 1.0)
        phase_coefficients = jnp.where(
            squared > 0.0,
            -(
                self.prepared.boson_mass
                * safe_scale**2
                / self.prepared.reduced_planck_constant
            )
            * divergence
            / safe_squared,
            0.0,
        )
        phase = discretization.reconstruct(phase_coefficients, real_output=True)
        volume = self.prepared.cell_volume
        phase_mean = _weighted_sum(discretization, phase) / volume
        phase = phase - phase_mean
        phase_coefficients = discretization.project(phase)
        reconstructed_velocity = jnp.stack(
            tuple(
                (
                    self.prepared.reduced_planck_constant
                    / (self.prepared.boson_mass * safe_scale**2)
                )
                * discretization.reconstruct(
                    1j * component * phase_coefficients,
                    real_output=True,
                )
                for component in wavevectors
            ),
            axis=-1,
        )
        reconstructed_current = rho[..., None] * reconstructed_velocity
        current_residual = _relative_residual(
            _weighted_l2(discretization, reconstructed_current - flux),
            _weighted_l2(discretization, flux),
        )
        curl_terms = []
        for left in range(dimension):
            for right in range(left + 1, dimension):
                derivative_right = discretization.reconstruct(
                    1j * wavevectors[left] * coefficients[..., right],
                    real_output=True,
                )
                derivative_left = discretization.reconstruct(
                    1j * wavevectors[right] * coefficients[..., left],
                    real_output=True,
                )
                curl_terms.append(derivative_right - derivative_left)
        curl_norm = (
            jnp.asarray(0.0, dtype=rho.dtype)
            if not curl_terms
            else _weighted_l2(discretization, jnp.stack(tuple(curl_terms), axis=-1))
        )
        fundamental = self.fundamental_wavenumber
        curl_residual = _relative_residual(
            curl_norm,
            fundamental * _weighted_l2(discretization, velocity),
        )
        amplitude = jnp.sqrt(jnp.where(rho > 0.0, rho, 0.0) / self.prepared.boson_mass)
        psi = amplitude.astype(
            jnp.dtype(discretization.plan.precision.coefficient_dtype)
        ) * jnp.exp(1j * phase)
        reconstructed_density = self.prepared.boson_mass * jnp.abs(psi) ** 2
        density_residual = _relative_residual(
            _weighted_l2(discretization, reconstructed_density - rho),
            _weighted_l2(discretization, rho),
        )
        mass = _weighted_sum(discretization, rho)
        wave_mass = self.prepared.boson_mass * _weighted_sum(
            discretization, jnp.abs(psi) ** 2
        )
        mass_residual = _relative_residual(jnp.abs(wave_mass - mass), jnp.abs(mass))
        speed = jnp.sqrt(jnp.sum(reconstructed_velocity**2, axis=-1))
        de_broglie = (
            self.prepared.boson_mass
            * safe_scale**2
            * jnp.max(speed)
            / self.prepared.reduced_planck_constant
            / self.prepared.nyquist_wavenumber
        )
        de_broglie_limit = 2.0 / self.prepared.step_policy.minimum_de_broglie_cells
        gauge_absolute = jnp.abs(_weighted_sum(discretization, phase) / volume)
        finite = (
            finite_inputs
            & jnp.all(jnp.isfinite(phase))
            & jnp.all(jnp.isfinite(psi))
            & jnp.isfinite(curl_residual)
            & jnp.isfinite(current_residual)
            & jnp.isfinite(density_residual)
            & jnp.isfinite(gauge_absolute)
            & jnp.isfinite(de_broglie)
            & jnp.isfinite(mass_residual)
        )
        curl_closed = curl_residual <= self.curl_relative_tolerance
        current_closed = current_residual <= self.current_relative_tolerance
        gauge_closed = gauge_absolute <= self.phase_gauge_tolerance
        resolution_closed = de_broglie <= de_broglie_limit
        mass_closed = mass_residual <= self.mass_relative_tolerance
        successful = (
            finite
            & scale_matches
            & node_free
            & curl_closed
            & current_closed
            & gauge_closed
            & resolution_closed
            & mass_closed
        )
        status = jnp.where(
            ~finite,
            1,
            jnp.where(
                ~scale_matches,
                8,
                jnp.where(
                    ~node_free,
                    2,
                    jnp.where(
                        ~curl_closed,
                        3,
                        jnp.where(
                            ~current_closed,
                            4,
                            jnp.where(
                                ~gauge_closed,
                                5,
                                jnp.where(
                                    ~resolution_closed,
                                    6,
                                    jnp.where(~mass_closed, 7, 0),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        evidence = WavePhaseSeedEvidence(
            curl_relative_residual=curl_residual,
            current_relative_residual=current_residual,
            density_relative_residual=density_residual,
            phase_gauge_absolute=gauge_absolute,
            minimum_density=minimum,
            node_free=node_free,
            de_broglie_nyquist_fraction=de_broglie,
            mass=mass,
            wave_mass=wave_mass,
            mass_relative_residual=mass_residual,
            finite=finite,
            time_level_valid=scale_matches,
            status=status,
            successful=successful,
        )
        return WavePhaseSeedResult(
            state=WaveDarkMatterState(psi, scale),
            density=rho,
            target_current=flux,
            reconstructed_current=reconstructed_current,
            phase=phase,
            evidence=evidence,
            plan_id=self.plan_id,
        )


class LocalizedWaveSeedEvidence(StrictModule):
    target_mass: Array
    wave_mass: Array
    mass_relative_residual: Array
    minimum_amplitude: Array
    maximum_amplitude: Array
    finite: Array
    status: Array
    successful: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=("accepted", "non_finite", "mass_unclosed", "invalid_time_level"),
    )


class SolitonSeedResult(StrictModule):
    state: WaveDarkMatterState
    density: Array
    radial_profile: Array
    evidence: LocalizedWaveSeedEvidence
    seed_kind: str = eqx.field(static=True, default="soliton")
    plan_id: str = eqx.field(static=True, default="")

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class SolitonSeedPlan(StrictModule, NonTrainableState):
    """Normalized, node-free solitonic-core profile on a periodic grid."""

    prepared: PreparedPeriodicWaveDarkMatter
    center: tuple[float, ...] = eqx.field(static=True)
    box_size: tuple[float, ...] = eqx.field(static=True)
    core_radius: float = eqx.field(static=True)
    target_mass: float = eqx.field(static=True)
    shape_coefficient: float = eqx.field(static=True)
    profile_exponent: float = eqx.field(static=True)
    mass_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedPeriodicWaveDarkMatter,
        center: Sequence[float],
        core_radius: float,
        target_mass: float,
        /,
        *,
        shape_coefficient: float = 0.091,
        profile_exponent: float = 8.0,
        mass_relative_tolerance: float = 1.0e-10,
    ):
        if not isinstance(prepared, PreparedPeriodicWaveDarkMatter):
            raise TypeError("prepared must be PreparedPeriodicWaveDarkMatter.")
        origins, lengths, _ = _periodic_geometry(prepared.discretization)
        center_ = tuple(float(value) for value in center)
        values = (
            float(core_radius),
            float(target_mass),
            float(shape_coefficient),
            float(profile_exponent),
            float(mass_relative_tolerance),
        )
        if (
            len(center_) != len(lengths)
            or any(not isfinite(value) for value in center_)
            or any(
                value < origin or value >= origin + length
                for value, origin, length in zip(center_, origins, lengths, strict=True)
            )
            or any(not isfinite(value) for value in values)
            or any(value <= 0.0 for value in values[:4])
            or values[4] < 0.0
        ):
            raise ValueError("Soliton seed geometry and normalization are invalid.")
        self.prepared = prepared
        self.center = center_
        self.box_size = lengths
        (
            self.core_radius,
            self.target_mass,
            self.shape_coefficient,
            self.profile_exponent,
            self.mass_relative_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "soliton-seed-plan",
                "prepared": prepared.prepared_id,
                "center": list(center_),
                "core_radius": values[0],
                "target_mass": values[1],
                "shape_coefficient": values[2],
                "profile_exponent": values[3],
                "mass_relative_tolerance": values[4],
            }
        )

    def realize(self, scale_factor: ArrayLike, /) -> SolitonSeedResult:
        discretization = self.prepared.discretization
        lengths = self.box_size
        points = discretization.points.reshape(
            discretization.physical_shape + (len(lengths),)
        )
        displacement = points - jnp.asarray(self.center, dtype=points.dtype)
        box = jnp.asarray(lengths, dtype=points.dtype)
        displacement = displacement - box * jnp.round(displacement / box)
        radius_squared = jnp.sum(displacement**2, axis=-1)
        profile = (
            1.0 + self.shape_coefficient * radius_squared / self.core_radius**2
        ) ** (-self.profile_exponent)
        raw_mass = _weighted_sum(discretization, profile)
        density = profile * (self.target_mass / raw_mass)
        amplitude = jnp.sqrt(density / self.prepared.boson_mass)
        psi = amplitude.astype(jnp.dtype(discretization.plan.precision.coefficient_dtype))
        scale = jnp.asarray(scale_factor, dtype=amplitude.dtype)
        if scale.shape != ():
            raise ValueError("Soliton seed scale factor must be scalar.")
        time_valid = jnp.abs(scale - self.prepared.scale_factors[0]) <= (
            32.0 * jnp.finfo(scale.dtype).eps
        )
        wave_mass = self.prepared.boson_mass * _weighted_sum(
            discretization, jnp.abs(psi) ** 2
        )
        target = jnp.asarray(self.target_mass, dtype=wave_mass.dtype)
        residual = jnp.abs(wave_mass - target) / target
        finite = (
            jnp.all(jnp.isfinite(psi))
            & jnp.all(jnp.isfinite(density))
            & jnp.isfinite(wave_mass)
            & jnp.isfinite(residual)
        )
        successful = finite & time_valid & (residual <= self.mass_relative_tolerance)
        status = jnp.where(
            ~finite, 1, jnp.where(~time_valid, 3, jnp.where(successful, 0, 2))
        ).astype(jnp.int32)
        evidence = LocalizedWaveSeedEvidence(
            target_mass=target,
            wave_mass=wave_mass,
            mass_relative_residual=residual,
            minimum_amplitude=jnp.min(amplitude),
            maximum_amplitude=jnp.max(amplitude),
            finite=finite,
            status=status,
            successful=successful,
        )
        return SolitonSeedResult(
            state=WaveDarkMatterState(psi, scale),
            density=density,
            radial_profile=profile,
            evidence=evidence,
            plan_id=self.plan_id,
        )


class VortexSeedEvidence(StrictModule):
    target_mass: Array
    wave_mass: Array
    mass_relative_residual: Array
    prescribed_winding: Array
    measured_winding: Array
    measured_antivortex_winding: Array
    net_winding: Array
    winding_residual: Array
    node_present: Array
    finite: Array
    status: Array
    successful: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "non_finite",
            "mass_unclosed",
            "winding_or_periodic_neutrality_unclosed",
            "vortex_nodes_missing",
            "invalid_time_level",
        ),
    )


class VortexSeedResult(StrictModule):
    state: WaveDarkMatterState
    density: Array
    phase: Array
    evidence: VortexSeedEvidence
    winding_number: int = eqx.field(static=True)
    seed_kind: str = eqx.field(static=True, default="vortex")
    plan_id: str = eqx.field(static=True, default="")

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class VortexSeedPlan(StrictModule, NonTrainableState):
    """Normalized periodic vortex-antivortex pair with measured integer winding."""

    prepared: PreparedPeriodicWaveDarkMatter
    center: tuple[float, ...] = eqx.field(static=True)
    antivortex_center: tuple[float, ...] = eqx.field(static=True)
    box_size: tuple[float, ...] = eqx.field(static=True)
    transverse_axes: tuple[int, int] = eqx.field(static=True)
    vortex_contour_indices: tuple[int, ...] = eqx.field(static=True)
    antivortex_contour_indices: tuple[int, ...] = eqx.field(static=True)
    core_radius: float = eqx.field(static=True)
    target_mass: float = eqx.field(static=True)
    winding_number: int = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    mass_relative_tolerance: float = eqx.field(static=True)
    winding_tolerance: float = eqx.field(static=True)
    node_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedPeriodicWaveDarkMatter,
        center: Sequence[float],
        core_radius: float,
        target_mass: float,
        winding_number: int,
        /,
        *,
        antivortex_center: Sequence[float] | None = None,
        axis: int = 2,
        mass_relative_tolerance: float = 1.0e-10,
        winding_tolerance: float = 1.0e-10,
        node_relative_tolerance: float = 1.0e-10,
    ):
        if not isinstance(prepared, PreparedPeriodicWaveDarkMatter):
            raise TypeError("prepared must be PreparedPeriodicWaveDarkMatter.")
        if isinstance(winding_number, bool) or not isinstance(winding_number, Integral):
            raise TypeError("Vortex winding_number must be an integer.")
        winding = int(winding_number)
        if winding == 0:
            raise ValueError("Vortex winding_number must be nonzero.")
        origins, lengths, _ = _periodic_geometry(prepared.discretization)
        dimension = len(lengths)
        if dimension < 2:
            raise ValueError("Vortex seeds require at least two spatial dimensions.")
        axis_ = int(axis)
        if dimension == 2:
            axis_ = 2
            transverse = (0, 1)
        elif axis_ < 0 or axis_ >= dimension:
            raise ValueError("Vortex axis is outside the spatial dimension.")
        else:
            transverse = tuple(index for index in range(dimension) if index != axis_)[:2]
        center_ = tuple(float(value) for value in center)
        if antivortex_center is None and len(center_) == dimension:
            partner_values = list(center_)
            transverse_axis = transverse[0]
            partner_values[transverse_axis] = origins[transverse_axis] + (
                (
                    center_[transverse_axis]
                    - origins[transverse_axis]
                    + 0.5 * lengths[transverse_axis]
                )
                % lengths[transverse_axis]
            )
            partner = tuple(partner_values)
        else:
            partner = tuple(
                float(value)
                for value in (() if antivortex_center is None else antivortex_center)
            )
        values = (
            float(core_radius),
            float(target_mass),
            float(mass_relative_tolerance),
            float(winding_tolerance),
            float(node_relative_tolerance),
        )
        centers_valid = all(
            len(candidate) == dimension
            and all(isfinite(value) for value in candidate)
            and all(
                value >= origin and value < origin + length
                for value, origin, length in zip(candidate, origins, lengths, strict=True)
            )
            for candidate in (center_, partner)
        )
        if (
            not centers_valid
            or center_ == partner
            or any(not isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or any(value < 0.0 for value in values[2:])
        ):
            raise ValueError("Vortex-pair seed geometry and thresholds are invalid.")
        vortex_contour = _square_contour_indices(
            prepared.discretization, center_, transverse
        )
        antivortex_contour = _square_contour_indices(
            prepared.discretization, partner, transverse
        )
        self.prepared = prepared
        self.center = center_
        self.antivortex_center = partner
        self.box_size = lengths
        self.transverse_axes = transverse
        self.vortex_contour_indices = vortex_contour
        self.antivortex_contour_indices = antivortex_contour
        self.core_radius = values[0]
        self.target_mass = values[1]
        self.winding_number = winding
        self.axis = axis_
        self.mass_relative_tolerance = values[2]
        self.winding_tolerance = values[3]
        self.node_relative_tolerance = values[4]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-vortex-antivortex-seed-plan",
                "prepared": prepared.prepared_id,
                "center": list(center_),
                "antivortex_center": list(partner),
                "core_radius": values[0],
                "target_mass": values[1],
                "winding_number": winding,
                "axis": axis_,
                "vortex_contour_indices": list(vortex_contour),
                "antivortex_contour_indices": list(antivortex_contour),
                "thresholds": list(values[2:]),
            }
        )

    def realize(self, scale_factor: ArrayLike, /) -> VortexSeedResult:
        discretization = self.prepared.discretization
        lengths = self.box_size
        dimension = len(lengths)
        left, right = self.transverse_axes
        points = discretization.points.reshape(
            discretization.physical_shape + (dimension,)
        )
        box = jnp.asarray(lengths, dtype=points.dtype)
        primary_raw = points - jnp.asarray(self.center, dtype=points.dtype)
        partner_raw = points - jnp.asarray(self.antivortex_center, dtype=points.dtype)
        primary = primary_raw - box * jnp.round(primary_raw / box)
        partner = partner_raw - box * jnp.round(partner_raw / box)
        primary_radius_squared = primary[..., left] ** 2 + primary[..., right] ** 2
        partner_radius_squared = partner[..., left] ** 2 + partner[..., right] ** 2
        exponent = 0.5 * abs(self.winding_number)
        primary_amplitude = (
            primary_radius_squared / (primary_radius_squared + self.core_radius**2)
        ) ** exponent
        partner_amplitude = (
            partner_radius_squared / (partner_radius_squared + self.core_radius**2)
        ) ** exponent
        profile = (primary_amplitude * partner_amplitude) ** 2
        raw_mass = _weighted_sum(discretization, profile)
        density = profile * (self.target_mass / raw_mass)
        primary_angle = jnp.arctan2(primary_raw[..., right], primary_raw[..., left])
        partner_angle = jnp.arctan2(partner_raw[..., right], partner_raw[..., left])
        phase = self.winding_number * (primary_angle - partner_angle)
        amplitude = jnp.sqrt(density / self.prepared.boson_mass)
        psi = amplitude.astype(
            jnp.dtype(discretization.plan.precision.coefficient_dtype)
        ) * jnp.exp(1j * phase)
        scale = jnp.asarray(scale_factor, dtype=amplitude.dtype)
        if scale.shape != ():
            raise ValueError("Vortex seed scale factor must be scalar.")
        time_valid = jnp.abs(scale - self.prepared.scale_factors[0]) <= (
            32.0 * jnp.finfo(scale.dtype).eps
        )
        wave_mass = self.prepared.boson_mass * _weighted_sum(
            discretization, jnp.abs(psi) ** 2
        )
        target = jnp.asarray(self.target_mass, dtype=wave_mass.dtype)
        mass_residual = jnp.abs(wave_mass - target) / target
        prescribed = jnp.asarray(self.winding_number, dtype=amplitude.dtype)
        measured = _contour_winding(psi, self.vortex_contour_indices)
        measured_antivortex = _contour_winding(psi, self.antivortex_contour_indices)
        net_winding = measured + measured_antivortex
        winding_residual = (
            jnp.abs(measured - prescribed)
            + jnp.abs(measured_antivortex + prescribed)
            + jnp.abs(net_winding)
        )
        node_count = jnp.sum(
            amplitude <= self.node_relative_tolerance * jnp.max(amplitude)
        )
        node_present = node_count >= 2
        finite = (
            jnp.all(jnp.isfinite(psi))
            & jnp.all(jnp.isfinite(density))
            & jnp.isfinite(wave_mass)
            & jnp.isfinite(mass_residual)
            & jnp.isfinite(measured)
            & jnp.isfinite(measured_antivortex)
            & jnp.isfinite(net_winding)
            & jnp.isfinite(winding_residual)
        )
        mass_closed = mass_residual <= self.mass_relative_tolerance
        winding_closed = winding_residual <= self.winding_tolerance
        successful = finite & time_valid & mass_closed & winding_closed & node_present
        status = jnp.where(
            ~finite,
            1,
            jnp.where(
                ~time_valid,
                5,
                jnp.where(
                    ~mass_closed,
                    2,
                    jnp.where(~winding_closed, 3, jnp.where(~node_present, 4, 0)),
                ),
            ),
        ).astype(jnp.int32)
        evidence = VortexSeedEvidence(
            target_mass=target,
            wave_mass=wave_mass,
            mass_relative_residual=mass_residual,
            prescribed_winding=prescribed,
            measured_winding=measured,
            measured_antivortex_winding=measured_antivortex,
            net_winding=net_winding,
            winding_residual=winding_residual,
            node_present=node_present,
            finite=finite,
            status=status,
            successful=successful,
        )
        return VortexSeedResult(
            state=WaveDarkMatterState(psi, scale),
            density=density,
            phase=phase,
            evidence=evidence,
            winding_number=self.winding_number,
            plan_id=self.plan_id,
        )


class ImportedComplexFieldEvidence(StrictModule):
    declared_mass: Array
    wave_mass: Array
    mass_relative_residual: Array
    occupied_fraction: Array
    node_fraction: Array
    de_broglie_nyquist_fraction: Array
    finite: Array
    status: Array
    successful: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "non_finite",
            "zero_field",
            "mass_unclosed",
            "de_broglie_unresolved",
            "invalid_time_level",
        ),
    )


class ImportedComplexFieldValidationResult(StrictModule):
    state: WaveDarkMatterState
    evidence: ImportedComplexFieldEvidence
    artifact_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)
    requested_use_id: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    wavefunction_unit_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    validation_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class ImportedComplexFieldValidationPlan(StrictModule, NonTrainableState):
    """Admit exact imported field bytes with rights and convention identity."""

    prepared: PreparedPeriodicWaveDarkMatter
    artifact: ScientificArtifactEnvelope
    manifest: ReferenceArtifactManifest
    requested_use_id: str = eqx.field(static=True)
    expected_artifact_kind: str = eqx.field(static=True)
    wavefunction_unit: UnitDefinition = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    mass_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedPeriodicWaveDarkMatter,
        artifact: ScientificArtifactEnvelope,
        manifest: ReferenceArtifactManifest,
        /,
        *,
        expected_artifact_kind: str,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
        mass_relative_tolerance: float = 1.0e-8,
    ):
        if not isinstance(prepared, PreparedPeriodicWaveDarkMatter):
            raise TypeError("prepared must be PreparedPeriodicWaveDarkMatter.")
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope.")
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("manifest must be ReferenceArtifactManifest.")
        artifact_kind = str(expected_artifact_kind).strip()
        if not artifact_kind or artifact.artifact_kind != artifact_kind:
            raise ValueError("Imported complex-field artifact kind disagrees.")
        requested_use_id = _requested_reference_use(
            manifest,
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        _validate_reference_envelope(artifact, manifest)
        tolerance = float(mass_relative_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Imported field mass tolerance must be finite and nonnegative."
            )
        dimension = len(prepared.discretization.axes)
        self.prepared = prepared
        self.artifact = artifact
        self.manifest = manifest
        self.requested_use_id = requested_use_id
        self.expected_artifact_kind = artifact_kind
        self.wavefunction_unit = _wavefunction_unit(prepared.plan.scale, dimension)
        self.coordinate_convention = prepared.coordinate_convention
        self.normalization = "comoving-number-density=abs(psi)^2"
        self.mass_relative_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "imported-complex-field-validation-plan",
                "prepared": prepared.prepared_id,
                "artifact": artifact.artifact_id,
                "manifest": manifest.manifest_id,
                "requested_use": requested_use_id,
                "expected_artifact_kind": artifact_kind,
                "wavefunction_unit": self.wavefunction_unit.unit_id,
                "coordinate_convention": self.coordinate_convention,
                "normalization": self.normalization,
                "mass_relative_tolerance": tolerance,
            }
        )

    def validate(
        self,
        psi: ArrayLike,
        scale_factor: ArrayLike,
        declared_mass: ArrayLike,
        /,
        *,
        wavefunction_unit: UnitDefinition,
        coordinate_convention: str,
        normalization: str,
    ) -> ImportedComplexFieldValidationResult:
        if not isinstance(wavefunction_unit, UnitDefinition):
            raise TypeError("wavefunction_unit must be UnitDefinition.")
        if wavefunction_unit.unit_id != self.wavefunction_unit.unit_id:
            raise ValueError("Imported complex-field units disagree.")
        if str(coordinate_convention).strip() != self.coordinate_convention:
            raise ValueError("Imported complex-field coordinate convention disagrees.")
        if str(normalization).strip() != self.normalization:
            raise ValueError("Imported complex-field normalization disagrees.")
        payload = imported_complex_field_payload_bytes(
            psi,
            scale_factor,
            declared_mass,
        )
        self.manifest.verify_bytes(payload)
        field = jax.lax.stop_gradient(jnp.asarray(np.asarray(psi)))
        if field.shape != self.prepared.discretization.physical_shape:
            raise ValueError(
                "Imported complex field shape disagrees with the prepared grid."
            )
        scale = jax.lax.stop_gradient(jnp.asarray(scale_factor, dtype=field.real.dtype))
        mass = jax.lax.stop_gradient(jnp.asarray(declared_mass, dtype=field.real.dtype))
        discretization = self.prepared.discretization
        amplitude_squared = jnp.abs(field) ** 2
        maximum = jnp.max(amplitude_squared)
        occupied = amplitude_squared >= (
            self.prepared.step_policy.relative_amplitude_floor * maximum
        )
        safe = jnp.where(occupied, amplitude_squared, 1.0)
        coefficients = discretization.project(field)
        wave_squared = jnp.zeros_like(amplitude_squared)
        for axis in range(len(discretization.axes)):
            derivative = discretization.reconstruct(
                discretization.modal_derivative(coefficients, axis=axis),
                real_output=False,
            )
            component = jnp.imag(jnp.conj(field) * derivative) / safe
            wave_squared = wave_squared + jnp.where(occupied, component**2, 0.0)
        de_broglie = jnp.sqrt(jnp.max(wave_squared)) / self.prepared.nyquist_wavenumber
        wave_mass = self.prepared.boson_mass * _weighted_sum(
            discretization, amplitude_squared
        )
        mass_residual = _relative_residual(jnp.abs(wave_mass - mass), jnp.abs(mass))
        occupied_fraction = jnp.mean(occupied.astype(field.real.dtype))
        node_fraction = 1.0 - occupied_fraction
        time_valid = jnp.abs(scale - self.prepared.scale_factors[0]) <= (
            32.0 * jnp.finfo(scale.dtype).eps
        )
        finite = (
            jnp.all(jnp.isfinite(field))
            & jnp.isfinite(scale)
            & jnp.isfinite(mass)
            & jnp.isfinite(wave_mass)
            & jnp.isfinite(mass_residual)
            & jnp.isfinite(de_broglie)
        )
        nonzero = (maximum > 0.0) & (mass > 0.0)
        mass_closed = mass_residual <= self.mass_relative_tolerance
        de_broglie_closed = de_broglie <= (
            2.0 / self.prepared.step_policy.minimum_de_broglie_cells
        )
        successful = finite & nonzero & mass_closed & de_broglie_closed & time_valid
        status = jnp.where(
            ~finite,
            1,
            jnp.where(
                ~nonzero,
                2,
                jnp.where(
                    ~mass_closed,
                    3,
                    jnp.where(~de_broglie_closed, 4, jnp.where(~time_valid, 5, 0)),
                ),
            ),
        ).astype(jnp.int32)
        evidence = ImportedComplexFieldEvidence(
            declared_mass=mass,
            wave_mass=wave_mass,
            mass_relative_residual=mass_residual,
            occupied_fraction=occupied_fraction,
            node_fraction=node_fraction,
            de_broglie_nyquist_fraction=de_broglie,
            finite=finite,
            status=status,
            successful=successful,
        )
        validation_id = canonical_fingerprint(
            {
                "kind": "imported-complex-field-validation-result",
                "plan": self.plan_id,
                "manifest": self.manifest.manifest_id,
                "payload_size": len(payload),
            }
        )
        return ImportedComplexFieldValidationResult(
            state=WaveDarkMatterState(field, scale),
            evidence=evidence,
            artifact_id=self.artifact.artifact_id,
            manifest_id=self.manifest.manifest_id,
            requested_use_id=self.requested_use_id,
            provider=self.artifact.producer,
            provider_version=self.artifact.producer_version,
            wavefunction_unit_id=self.wavefunction_unit.unit_id,
            coordinate_convention=self.coordinate_convention,
            normalization=self.normalization,
            validation_id=validation_id,
        )


__all__ = [
    "ComponentModeRealization",
    "ComponentTransferMatrixProduct",
    "component_transfer_payload_bytes",
    "ImportedComplexFieldEvidence",
    "ImportedComplexFieldValidationPlan",
    "ImportedComplexFieldValidationResult",
    "imported_complex_field_payload_bytes",
    "LocalizedWaveSeedEvidence",
    "MixedInitialConditionPlan",
    "ParticleInitialConditionProjection",
    "PrimordialModeRealization",
    "SolitonSeedPlan",
    "SolitonSeedResult",
    "VortexSeedEvidence",
    "VortexSeedPlan",
    "VortexSeedResult",
    "WavePhaseSeedEvidence",
    "WavePhaseSeedPlan",
    "WavePhaseSeedResult",
]

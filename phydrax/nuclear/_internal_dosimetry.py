#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Research-only regional and voxel-kernel internal dosimetry."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..imaging._core import MedicalImageSupport
from ..measurement import (
    DataStage,
    DerivationRecord,
    IndexSampleSupport,
    MeasurementAsset,
    QualityFlag,
    QuantityField,
    RadiationQuantityKind,
    resolve_radiation_quantity,
    SamplingSemantics,
    SpatialSamplingKind,
)
from ..qualification import ReferenceArtifactManifest
from ..units import (
    BECQUEREL,
    BECQUEREL_SECOND,
    BECQUEREL_SECOND_PER_CUBIC_METER,
    conversion_factor,
    derived_unit,
    GRAY,
    METER,
    SECOND,
    UnitDefinition,
)
from ._activation import InventoryTransition
from ._provenance import NuclearDataProvenance
from ._time_activity import TimeActivityIntegrationResult


S_VALUE_UNIT = derived_unit("Gy/(Bq*s)", ((GRAY, 1), (BECQUEREL, -1), (SECOND, -1)))
_DOSE_KINDS = frozenset(
    {
        RadiationQuantityKind.ABSORBED_DOSE,
        RadiationQuantityKind.DOSE_TO_WATER,
        RadiationQuantityKind.DOSE_TO_MEDIUM,
    }
)


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text.")
    return value


def _identifiers(values, name: str, /) -> tuple[str, ...]:
    result = tuple(_text(value, name) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique canonical identifiers.")
    return result


def _readonly_nonnegative(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    result = np.array(value, dtype=np.result_type(original.dtype, np.float64), copy=True)
    if (
        not np.issubdtype(result.dtype, np.floating)
        or np.any(~np.isfinite(result))
        or np.any(result < 0.0)
    ):
        raise ValueError(f"{name} must be finite, real-valued, and non-negative.")
    result.setflags(write=False)
    return result


def _s_values(value: ArrayLike, unit: UnitDefinition, name: str, /) -> np.ndarray:
    if not isinstance(unit, UnitDefinition) or unit.unit_id != S_VALUE_UNIT.unit_id:
        raise ValueError("S-values must use the explicit Gy/(Bq*s) unit.")
    return _readonly_nonnegative(value, name)


def _dose_kind(value: RadiationQuantityKind | str, /) -> RadiationQuantityKind:
    kind = (
        value
        if isinstance(value, RadiationQuantityKind)
        else RadiationQuantityKind(value)
    )
    if kind not in _DOSE_KINDS:
        raise ValueError(
            "Internal dosimetry outputs must be absorbed dose, dose to water, or dose to medium."
        )
    return kind


def _references(
    asset: MeasurementAsset, data: NuclearDataProvenance, /
) -> tuple[ReferenceArtifactManifest, ...]:
    result = list(asset.references)
    known = {value.manifest_id for value in result}
    if data.reference.manifest_id not in known:
        result.append(data.reference)
    return tuple(result)


def _require_decay_transition(value: InventoryTransition, /) -> InventoryTransition:
    if not isinstance(value, InventoryTransition):
        raise TypeError("transition must be InventoryTransition.")
    if value.flux_driven:
        raise ValueError("Internal dosimetry requires a radionuclide decay transition.")
    return value


@dataclass(frozen=True, slots=True)
class InternalDosimetryEvidence:
    """Source, target, radionuclide, coefficient, and method lineage."""

    mode: str
    source_id: str
    target_id: str
    radionuclide_id: str
    transition_id: str
    kernel_id: str
    kernel_data_id: str
    method: str
    boundary_policy: str
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        mode = _text(self.mode, "mode")
        source = _text(self.source_id, "source_id")
        target = _text(self.target_id, "target_id")
        radionuclide = _text(self.radionuclide_id, "radionuclide_id")
        transition = _text(self.transition_id, "transition_id")
        kernel = _text(self.kernel_id, "kernel_id")
        data = _text(self.kernel_data_id, "kernel_data_id")
        method = _text(self.method, "method")
        boundary = _text(self.boundary_policy, "boundary_policy")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "radionuclide_id", radionuclide)
        object.__setattr__(self, "transition_id", transition)
        object.__setattr__(self, "kernel_id", kernel)
        object.__setattr__(self, "kernel_data_id", data)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "boundary_policy", boundary)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "internal-dosimetry-evidence",
                    "mode": mode,
                    "source": source,
                    "target": target,
                    "radionuclide": radionuclide,
                    "transition": transition,
                    "kernel": kernel,
                    "kernel_data": data,
                    "method": method,
                    "boundary": boundary,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RegionalSValueTable:
    """Target-by-source regional S-values for exactly one radionuclide."""

    source_region_ids: tuple[str, ...]
    target_region_ids: tuple[str, ...]
    values: np.ndarray
    unit: UnitDefinition
    source_region_set_id: str
    target_region_set_id: str
    transition: InventoryTransition
    data: NuclearDataProvenance
    dose_kind: RadiationQuantityKind | str = RadiationQuantityKind.ABSORBED_DOSE
    dose_reference_configuration: str = "regional-mean-absorbed-dose"
    table_id: str = field(init=False)

    def __post_init__(self) -> None:
        sources = _identifiers(self.source_region_ids, "source_region_id")
        targets = _identifiers(self.target_region_ids, "target_region_id")
        values = _s_values(self.values, self.unit, "regional S-values")
        if values.shape != (len(targets), len(sources)):
            raise ValueError(
                "Regional S-values must have shape (target_region, source_region)."
            )
        source_set = _text(self.source_region_set_id, "source_region_set_id")
        target_set = _text(self.target_region_set_id, "target_region_set_id")
        transition = _require_decay_transition(self.transition)
        if not isinstance(self.data, NuclearDataProvenance):
            raise TypeError("data must be NuclearDataProvenance.")
        kind = _dose_kind(self.dose_kind)
        reference = _text(
            self.dose_reference_configuration, "dose_reference_configuration"
        )
        resolve_radiation_quantity(
            "regional-internal-dose",
            kind,
            GRAY,
            support_association="regional-mean",
            reference_configuration=reference,
        )
        object.__setattr__(self, "source_region_ids", sources)
        object.__setattr__(self, "target_region_ids", targets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_region_set_id", source_set)
        object.__setattr__(self, "target_region_set_id", target_set)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "dose_kind", kind)
        object.__setattr__(self, "dose_reference_configuration", reference)
        object.__setattr__(
            self,
            "table_id",
            canonical_fingerprint(
                {
                    "kind": "regional-s-value-table",
                    "source_regions": list(sources),
                    "target_regions": list(targets),
                    "source_region_set": source_set,
                    "target_region_set": target_set,
                    "values_gy_per_bq_s": array_tree_fingerprint(values),
                    "unit": S_VALUE_UNIT.unit_id,
                    "radionuclide": transition.parent.nuclide_id,
                    "transition": transition.transition_id,
                    "data": self.data.provenance_id,
                    "dose_kind": kind.value,
                    "dose_reference": reference,
                }
            ),
        )


class RegionalSValueEvaluation(StrictModule):
    dose_gy: Array
    valid: Array


class PreparedRegionalSValuePlan(StrictModule, NonTrainableState):
    values_gy_per_bq_s: Array
    dependencies: Array
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def evaluate(
        self, time_integrated_activity_bq_s: ArrayLike, valid: ArrayLike, /
    ) -> RegionalSValueEvaluation:
        activity = jnp.asarray(time_integrated_activity_bq_s)
        validity = jnp.asarray(valid, dtype=bool)
        if activity.shape != (self.source_count,) or validity.shape != activity.shape:
            raise ValueError("Regional activity must match the source-region axis.")
        source_valid = validity & jnp.isfinite(activity) & (activity >= 0.0)
        safe = jnp.where(source_valid, activity, 0.0)
        dose = self.values_gy_per_bq_s @ safe
        target_valid = jnp.all(
            jnp.where(self.dependencies, source_valid[None, :], True), axis=1
        )
        target_valid = target_valid & jnp.isfinite(dose) & (dose >= 0.0)
        return RegionalSValueEvaluation(dose, target_valid)

    def affected_targets(self, source_mask: ArrayLike, /) -> Array:
        mask = jnp.asarray(source_mask, dtype=bool)
        if mask.shape != (self.source_count,):
            raise ValueError("Regional quality masks must match the source-region axis.")
        return jnp.any(self.dependencies & mask[None, :], axis=1)


@dataclass(frozen=True, slots=True)
class RegionalDoseResult:
    asset: MeasurementAsset
    target_region_ids: tuple[str, ...]
    table_id: str
    evidence: InternalDosimetryEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MeasurementAsset):
            raise TypeError("asset must be MeasurementAsset.")
        targets = _identifiers(self.target_region_ids, "target_region_id")
        table = _text(self.table_id, "table_id")
        if not isinstance(self.evidence, InternalDosimetryEvidence):
            raise TypeError("evidence must be InternalDosimetryEvidence.")
        if self.asset.intended_use != "research":
            raise ValueError("Regional dose results are research-only.")
        if self.asset.field.uncertainty is not None:
            raise ValueError("Regional dose uncertainty must not be fabricated.")
        if self.asset.field.values.shape != (len(targets),):
            raise ValueError("Regional dose values do not match target_region_ids.")
        object.__setattr__(self, "target_region_ids", targets)
        object.__setattr__(self, "table_id", table)

    @property
    def dose_gy(self) -> np.ndarray:
        return self.asset.field.values

    @property
    def valid_mask(self) -> np.ndarray:
        return self.asset.field.valid_mask


@dataclass(frozen=True, slots=True)
class RegionalSValuePlan:
    table: RegionalSValueTable
    source_support: IndexSampleSupport
    target_support: IndexSampleSupport = field(init=False)
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.table, RegionalSValueTable):
            raise TypeError("table must be RegionalSValueTable.")
        if not isinstance(self.source_support, IndexSampleSupport):
            raise TypeError("source_support must be IndexSampleSupport.")
        if self.source_support.time_axis is not None:
            raise ValueError("Regional S-value plans require time-integrated support.")
        if self.source_support.sample_shape != (len(self.table.source_region_ids),):
            raise ValueError("source_support does not match the source-region axis.")
        if self.source_support.frame_id != self.table.source_region_set_id:
            raise ValueError("source_support uses a different region reference system.")
        target = IndexSampleSupport(
            (len(self.table.target_region_ids),),
            ("target_region",),
            frame_id=self.table.target_region_set_id,
        )
        object.__setattr__(self, "target_support", target)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "regional-s-value-plan",
                    "table": self.table.table_id,
                    "source_support": self.source_support.support_id,
                    "target_support": target.support_id,
                }
            ),
        )

    def prepare(self) -> PreparedRegionalSValuePlan:
        return PreparedRegionalSValuePlan(
            jnp.asarray(self.table.values),
            jnp.asarray(self.table.values != 0.0),
            len(self.table.source_region_ids),
            len(self.table.target_region_ids),
            self.plan_id,
        )

    def apply(self, source: TimeActivityIntegrationResult, /) -> RegionalDoseResult:
        if not isinstance(source, TimeActivityIntegrationResult):
            raise TypeError("source must be TimeActivityIntegrationResult.")
        asset = source.asset
        field_ = asset.field
        if (
            field_.quantity.quantity_kind
            != RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY.value
        ):
            raise ValueError("Regional S-values require time-integrated activity.")
        if source.transition.transition_id != self.table.transition.transition_id:
            raise ValueError("Regional activity and S-values name different transitions.")
        if field_.support.support_id != self.source_support.support_id:
            raise ValueError("Regional activity does not use the plan source support.")
        scale = float(conversion_factor(field_.quantity.unit, BECQUEREL_SECOND))
        evaluation = self.prepare().evaluate(
            np.asarray(field_.values) * scale, field_.valid_mask
        )
        values = np.asarray(evaluation.dose_gy)
        valid = np.asarray(evaluation.valid)
        prepared = self.prepare()
        flags = tuple(
            QualityFlag(
                value.name,
                np.asarray(prepared.affected_targets(value.mask)),
                value.meaning,
            )
            for value in field_.quality_flags
        )
        quantity = resolve_radiation_quantity(
            "regional-internal-dose",
            self.table.dose_kind,
            GRAY,
            axes=field_.quantity.axes,
            support_association="regional-mean",
            reference_configuration=self.table.dose_reference_configuration,
        )
        sampling = SamplingSemantics(
            SpatialSamplingKind.CELL_AVERAGE,
            field_.sampling.temporal,
            self.table.table_id,
            "regional-s-value-matrix",
        )
        dose_field = QuantityField(
            f"{field_.field_id}:regional-dose:{self.plan_id[:12]}",
            quantity,
            field_.layout,
            self.target_support,
            sampling,
            values,
            valid,
            None,
            flags,
        )
        result_asset = MeasurementAsset(
            f"{asset.asset_id}:regional-dose:{self.plan_id[:12]}",
            dose_field,
            asset.acquisition,
            _references(asset, self.table.data),
            DerivationRecord(
                asset.derivation.origin,
                DataStage.DERIVED,
                (asset.content_id,),
                self.plan_id,
            ),
            "research",
            {
                "source_region_ids": self.table.source_region_ids,
                "target_region_ids": self.table.target_region_ids,
                "source_region_set_id": self.table.source_region_set_id,
                "target_region_set_id": self.table.target_region_set_id,
                "radionuclide_id": self.table.transition.parent.nuclide_id,
                "transition_id": self.table.transition.transition_id,
                "s_value_table_id": self.table.table_id,
                "kernel_data_id": self.table.data.provenance_id,
                "uncertainty_propagation": "not-performed-covariance-unspecified",
            },
        )
        evidence = InternalDosimetryEvidence(
            "regional-s-value",
            asset.content_id,
            self.target_support.support_id,
            self.table.transition.parent.nuclide_id,
            self.table.transition.transition_id,
            self.table.table_id,
            self.table.data.provenance_id,
            "target-by-source-matrix-product",
            "not-applicable-regional-table",
        )
        return RegionalDoseResult(
            result_asset, self.table.target_region_ids, self.table.table_id, evidence
        )


@dataclass(frozen=True, slots=True)
class SpatialSValueKernel:
    """Centered nonperiodic voxel S-value kernel on one exact lattice."""

    values: np.ndarray
    unit: UnitDefinition
    source_support: MedicalImageSupport
    target_support: MedicalImageSupport
    source_grid_id: str
    target_grid_id: str
    transition: InventoryTransition
    data: NuclearDataProvenance
    dose_kind: RadiationQuantityKind | str = RadiationQuantityKind.ABSORBED_DOSE
    dose_reference_configuration: str = "voxel-mean-absorbed-dose"
    kernel_id: str = field(init=False)
    voxel_volume_m3: float = field(init=False)

    def __post_init__(self) -> None:
        values = _s_values(self.values, self.unit, "spatial S-value kernel")
        if values.ndim != 3 or any(size % 2 != 1 for size in values.shape):
            raise ValueError("Spatial S-value kernels must have three odd-sized axes.")
        if not isinstance(self.source_support, MedicalImageSupport) or not isinstance(
            self.target_support, MedicalImageSupport
        ):
            raise TypeError("Spatial S-value kernels require MedicalImageSupport grids.")
        if (
            self.source_support.time_axis is not None
            or self.target_support.time_axis is not None
        ):
            raise ValueError("Spatial S-value kernel grids must not carry a time axis.")
        if (
            self.source_support.spatial_shape != self.target_support.spatial_shape
            or self.source_support.spatial_affine.affine_id
            != self.target_support.spatial_affine.affine_id
        ):
            raise ValueError(
                "Spatial S-value source and target grids must be exactly compatible."
            )
        source_grid = _text(self.source_grid_id, "source_grid_id")
        target_grid = _text(self.target_grid_id, "target_grid_id")
        transition = _require_decay_transition(self.transition)
        if not isinstance(self.data, NuclearDataProvenance):
            raise TypeError("data must be NuclearDataProvenance.")
        kind = _dose_kind(self.dose_kind)
        reference = _text(
            self.dose_reference_configuration, "dose_reference_configuration"
        )
        resolve_radiation_quantity(
            "voxel-internal-dose",
            kind,
            GRAY,
            support_association="voxel-cell-average",
            reference_configuration=reference,
        )
        affine = self.source_support.spatial_affine
        length_scale = float(
            conversion_factor(affine.coordinate_contract.length_unit, METER)
        )
        volume = abs(float(np.linalg.det(affine.matrix[:3, :3]))) * length_scale**3
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("The source grid must have finite positive voxel volume.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_grid_id", source_grid)
        object.__setattr__(self, "target_grid_id", target_grid)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "dose_kind", kind)
        object.__setattr__(self, "dose_reference_configuration", reference)
        object.__setattr__(self, "voxel_volume_m3", volume)
        object.__setattr__(
            self,
            "kernel_id",
            canonical_fingerprint(
                {
                    "kind": "spatial-s-value-kernel",
                    "values_gy_per_bq_s": array_tree_fingerprint(values),
                    "unit": S_VALUE_UNIT.unit_id,
                    "source_support": self.source_support.support_id,
                    "target_support": self.target_support.support_id,
                    "source_grid": source_grid,
                    "target_grid": target_grid,
                    "voxel_volume_m3": volume,
                    "radionuclide": transition.parent.nuclide_id,
                    "transition": transition.transition_id,
                    "data": self.data.provenance_id,
                    "dose_kind": kind.value,
                    "dose_reference": reference,
                    "boundary": "zero-outside-nonperiodic",
                }
            ),
        )


def _convolve_nonperiodic(values: Array, kernel: Array, /) -> Array:
    padding = tuple((int(size) // 2, int(size) // 2) for size in kernel.shape)
    result = lax.conv_general_dilated(
        values[None, ..., None],
        jnp.flip(kernel, axis=(0, 1, 2))[..., None, None],
        window_strides=(1, 1, 1),
        padding=padding,
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    return result[0, ..., 0]


class SpatialSValueEvaluation(StrictModule):
    dose_gy: Array
    valid: Array


class PreparedSpatialSValueConvolution(StrictModule, NonTrainableState):
    kernel_gy_per_bq_s: Array
    footprint: Array
    voxel_volume_m3: Array
    volume_shape: tuple[int, int, int] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def evaluate(
        self, time_integrated_activity_concentration: ArrayLike, valid: ArrayLike, /
    ) -> SpatialSValueEvaluation:
        concentration = jnp.asarray(time_integrated_activity_concentration)
        validity = jnp.asarray(valid, dtype=bool)
        if (
            concentration.shape != self.volume_shape
            or validity.shape != self.volume_shape
        ):
            raise ValueError(
                "Spatial activity and validity must match the exact source grid."
            )
        source_valid = validity & jnp.isfinite(concentration) & (concentration >= 0.0)
        activity_bq_s = jnp.where(source_valid, concentration, 0.0) * self.voxel_volume_m3
        dose = _convolve_nonperiodic(activity_bq_s, self.kernel_gy_per_bq_s)
        invalid_contributors = _convolve_nonperiodic(
            (~source_valid).astype(dose.dtype), self.footprint.astype(dose.dtype)
        )
        target_valid = (invalid_contributors == 0.0) & jnp.isfinite(dose) & (dose >= 0.0)
        return SpatialSValueEvaluation(dose, target_valid)

    def affected_targets(self, source_mask: ArrayLike, /) -> Array:
        mask = jnp.asarray(source_mask, dtype=bool)
        if mask.shape != self.volume_shape:
            raise ValueError("Spatial quality masks must match the exact source grid.")
        affected = _convolve_nonperiodic(
            mask.astype(self.kernel_gy_per_bq_s.dtype),
            self.footprint.astype(self.kernel_gy_per_bq_s.dtype),
        )
        return affected > 0.0


@dataclass(frozen=True, slots=True)
class SpatialDoseResult:
    asset: MeasurementAsset
    kernel_id: str
    evidence: InternalDosimetryEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MeasurementAsset):
            raise TypeError("asset must be MeasurementAsset.")
        kernel = _text(self.kernel_id, "kernel_id")
        if not isinstance(self.evidence, InternalDosimetryEvidence):
            raise TypeError("evidence must be InternalDosimetryEvidence.")
        if self.asset.intended_use != "research":
            raise ValueError("Spatial dose results are research-only.")
        if self.asset.field.uncertainty is not None:
            raise ValueError("Spatial dose uncertainty must not be fabricated.")
        object.__setattr__(self, "kernel_id", kernel)

    @property
    def dose_gy(self) -> np.ndarray:
        return self.asset.field.values

    @property
    def valid_mask(self) -> np.ndarray:
        return self.asset.field.valid_mask


@dataclass(frozen=True, slots=True)
class SpatialSValueConvolutionPlan:
    kernel: SpatialSValueKernel
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, SpatialSValueKernel):
            raise TypeError("kernel must be SpatialSValueKernel.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "spatial-s-value-convolution-plan",
                    "kernel": self.kernel.kernel_id,
                    "boundary": "zero-outside-nonperiodic",
                    "grid_policy": "exact-no-resampling",
                    "activity_policy": "concentration-times-voxel-volume",
                }
            ),
        )

    def prepare(self) -> PreparedSpatialSValueConvolution:
        return PreparedSpatialSValueConvolution(
            jnp.asarray(self.kernel.values),
            jnp.asarray(self.kernel.values != 0.0),
            jnp.asarray(self.kernel.voxel_volume_m3),
            self.kernel.source_support.spatial_shape,
            self.plan_id,
        )

    def apply(self, source: TimeActivityIntegrationResult, /) -> SpatialDoseResult:
        if not isinstance(source, TimeActivityIntegrationResult):
            raise TypeError("source must be TimeActivityIntegrationResult.")
        asset = source.asset
        field_ = asset.field
        if (
            field_.quantity.quantity_kind
            != RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION.value
        ):
            raise ValueError(
                "Spatial S-value convolution requires time-integrated activity concentration."
            )
        if source.transition.transition_id != self.kernel.transition.transition_id:
            raise ValueError(
                "Spatial activity and S-value kernel name different transitions."
            )
        if not isinstance(field_.support, MedicalImageSupport):
            raise TypeError("Spatial S-value convolution requires MedicalImageSupport.")
        if field_.support.support_id != self.kernel.source_support.support_id:
            raise ValueError(
                "Spatial activity does not use the exact kernel source grid; resampling is not performed."
            )
        if field_.sampling.spatial_kind is not SpatialSamplingKind.CELL_AVERAGE:
            raise ValueError(
                "Activity concentration must be a voxel-cell average before volume multiplication."
            )
        scale = float(
            conversion_factor(field_.quantity.unit, BECQUEREL_SECOND_PER_CUBIC_METER)
        )
        prepared = self.prepare()
        evaluation = prepared.evaluate(
            np.asarray(field_.values) * scale, field_.valid_mask
        )
        values = np.asarray(evaluation.dose_gy)
        valid = np.asarray(evaluation.valid)
        flags = tuple(
            QualityFlag(
                value.name,
                np.asarray(prepared.affected_targets(value.mask)),
                value.meaning,
            )
            for value in field_.quality_flags
        )
        quantity = resolve_radiation_quantity(
            "voxel-internal-dose",
            self.kernel.dose_kind,
            GRAY,
            axes=field_.quantity.axes,
            support_association="voxel-cell-average",
            reference_configuration=self.kernel.dose_reference_configuration,
        )
        sampling = SamplingSemantics(
            SpatialSamplingKind.CELL_AVERAGE,
            field_.sampling.temporal,
            self.kernel.kernel_id,
            "nonperiodic-s-value-convolution",
        )
        dose_field = QuantityField(
            f"{field_.field_id}:spatial-dose:{self.plan_id[:12]}",
            quantity,
            field_.layout,
            self.kernel.target_support,
            sampling,
            values,
            valid,
            None,
            flags,
        )
        result_asset = MeasurementAsset(
            f"{asset.asset_id}:spatial-dose:{self.plan_id[:12]}",
            dose_field,
            asset.acquisition,
            _references(asset, self.kernel.data),
            DerivationRecord(
                asset.derivation.origin,
                DataStage.DERIVED,
                (asset.content_id,),
                self.plan_id,
            ),
            "research",
            {
                "source_grid_id": self.kernel.source_grid_id,
                "target_grid_id": self.kernel.target_grid_id,
                "source_support_id": self.kernel.source_support.support_id,
                "target_support_id": self.kernel.target_support.support_id,
                "radionuclide_id": self.kernel.transition.parent.nuclide_id,
                "transition_id": self.kernel.transition.transition_id,
                "s_value_kernel_id": self.kernel.kernel_id,
                "kernel_data_id": self.kernel.data.provenance_id,
                "voxel_volume_m3": self.kernel.voxel_volume_m3,
                "boundary_policy": "zero-outside-nonperiodic",
                "uncertainty_propagation": "not-performed-covariance-unspecified",
            },
        )
        evidence = InternalDosimetryEvidence(
            "spatial-s-value",
            asset.content_id,
            self.kernel.target_support.support_id,
            self.kernel.transition.parent.nuclide_id,
            self.kernel.transition.transition_id,
            self.kernel.kernel_id,
            self.kernel.data.provenance_id,
            "activity-concentration-times-voxel-volume-then-convolution",
            "zero-outside-nonperiodic",
        )
        return SpatialDoseResult(result_asset, self.kernel.kernel_id, evidence)


__all__ = [
    "InternalDosimetryEvidence",
    "PreparedRegionalSValuePlan",
    "PreparedSpatialSValueConvolution",
    "RegionalDoseResult",
    "RegionalSValueEvaluation",
    "RegionalSValuePlan",
    "RegionalSValueTable",
    "S_VALUE_UNIT",
    "SpatialDoseResult",
    "SpatialSValueConvolutionPlan",
    "SpatialSValueEvaluation",
    "SpatialSValueKernel",
]

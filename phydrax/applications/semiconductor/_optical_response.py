#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen carrier optical response for reduced semiconductor laser models.

This module projects an already-computed active-region carrier state and then
freezes a constitutive gain/index/loss response. It does not solve, replace, or
claim coupling to the semiconductor drift-diffusion equations.
"""

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_VACUUM_WAVE_SPEED = 299_792_458.0


class SemiconductorOpticalResponseStatus(IntFlag):
    """Elementwise fail-closed constitutive disposition."""

    SUCCESS = 0
    CARRIER_OUTSIDE_SUPPORT = 1
    TEMPERATURE_OUTSIDE_SUPPORT = 2
    FREQUENCY_OUTSIDE_SUPPORT = 4
    INVALID_PROJECTION = 8
    NONFINITE = 16
    NEGATIVE_INTERNAL_LOSS = 32


class ActiveRegionOpticalProjection(StrictModule, NonTrainableState):
    """Explicit active-volume and normalized optical-mode projection.

    ``control_volumes`` are physical m3, ``active_fraction`` is the active share
    of each control volume, and ``modal_power_density`` is any nonnegative
    relative optical power density. The latter is normalized internally.
    Projected carrier density and temperature are mode-weighted *inside* the
    active region. ``confinement_factor`` is the fraction of total modal power
    within that active region.
    """

    control_volumes: Array
    active_fraction: Array
    modal_power_density: Array
    projection_weights: Array
    active_volume: Array
    confinement_factor: Array
    support_id: str = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        control_volumes: ArrayLike,
        active_fraction: ArrayLike,
        modal_power_density: ArrayLike,
        /,
        *,
        support_id: str,
        mode_id: str,
        provenance: str,
    ):
        volumes = np.asarray(control_volumes)
        active = np.asarray(active_fraction)
        mode = np.asarray(modal_power_density)
        if (
            volumes.ndim != 1
            or not volumes.size
            or active.shape != volumes.shape
            or mode.shape != volumes.shape
            or np.iscomplexobj(volumes)
            or np.iscomplexobj(active)
            or np.iscomplexobj(mode)
            or np.any(~np.isfinite(volumes))
            or np.any(~np.isfinite(active))
            or np.any(~np.isfinite(mode))
            or np.any(volumes <= 0.0)
            or np.any((active < 0.0) | (active > 1.0))
            or np.any(mode < 0.0)
        ):
            raise ValueError(
                "Projection arrays must be equal nonempty vectors with positive "
                "volumes, active fractions in [0, 1], and nonnegative mode density."
            )
        if not support_id or not mode_id or not provenance.strip():
            raise ValueError("Projection support, mode, and provenance must be explicit.")
        total_mode = float(np.sum(volumes * mode))
        active_mode = float(np.sum(volumes * active * mode))
        active_volume = float(np.sum(volumes * active))
        if total_mode <= 0.0 or active_mode <= 0.0 or active_volume <= 0.0:
            raise ValueError("Projection must overlap a nonzero active region and mode.")
        weights = volumes * active * mode / active_mode
        self.control_volumes = jnp.asarray(volumes)
        self.active_fraction = jnp.asarray(active)
        self.modal_power_density = jnp.asarray(mode / total_mode)
        self.projection_weights = jnp.asarray(weights)
        self.active_volume = jnp.asarray(active_volume)
        self.confinement_factor = jnp.asarray(active_mode / total_mode)
        self.support_id = str(support_id)
        self.mode_id = str(mode_id)
        self.provenance = provenance.strip()
        self.projection_id = canonical_fingerprint(
            {
                "kind": "active-region-optical-projection",
                "control_volumes": array_tree_fingerprint(volumes),
                "active_fraction": array_tree_fingerprint(active),
                "modal_power_density": array_tree_fingerprint(mode),
                "support_id": self.support_id,
                "mode_id": self.mode_id,
                "provenance": self.provenance,
            }
        )

    def project(
        self,
        carrier_pair_density: ArrayLike,
        lattice_temperature: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        density = jnp.asarray(carrier_pair_density)
        temperature = jnp.asarray(lattice_temperature)
        if density.shape != self.projection_weights.shape:
            raise ValueError("carrier_pair_density must match the projection support.")
        if temperature.shape == ():
            temperature = jnp.broadcast_to(temperature, density.shape)
        if temperature.shape != density.shape:
            raise ValueError("lattice_temperature must be scalar or match the support.")
        projected_density = contract("n,n->", self.projection_weights, density)
        projected_temperature = contract("n,n->", self.projection_weights, temperature)
        valid = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(density >= 0.0)
            & jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature > 0.0)
        )
        return projected_density, projected_temperature, valid


def _range(value: ArrayLike, name: str, /, *, nonnegative: bool, positive: bool) -> Array:
    host = np.asarray(value)
    shape_valid = host.shape == (2,)
    invalid_lower = False
    if shape_valid:
        invalid_lower = (
            host[0] < 0.0 if nonnegative else host[0] <= 0.0 if positive else False
        )
    if (
        not shape_valid
        or np.iscomplexobj(host)
        or np.any(~np.isfinite(host))
        or invalid_lower
        or host[1] <= host[0]
    ):
        qualifier = "nonnegative " if nonnegative else "positive " if positive else ""
        raise ValueError(f"{name} must contain two increasing finite {qualifier}bounds.")
    return jnp.asarray(host)


def _finite_scalar(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value)
    if host.shape != () or np.iscomplexobj(host) or not np.isfinite(host):
        raise ValueError(f"{name} must be one finite real scalar.")
    return jnp.asarray(host)


def _positive_scalar(
    value: ArrayLike, name: str, /, *, nonnegative: bool = False
) -> Array:
    resolved = _finite_scalar(value, name)
    host = float(resolved)
    invalid = host < 0.0 if nonnegative else host <= 0.0
    if invalid:
        qualifier = "nonnegative" if nonnegative else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return resolved


def _model_metadata(
    *,
    active_volume: ArrayLike,
    confinement_factor: ArrayLike,
    reference_wave_speed: ArrayLike,
    provenance: str,
) -> tuple[Array, Array, Array, str]:
    volume = _positive_scalar(active_volume, "active_volume")
    confinement = _positive_scalar(confinement_factor, "confinement_factor")
    if float(confinement) > 1.0:
        raise ValueError("confinement_factor must lie in (0, 1].")
    wave_speed = _positive_scalar(reference_wave_speed, "reference_wave_speed")
    provenance_ = provenance.strip()
    if not provenance_:
        raise ValueError("Optical-response provenance must be explicit nonempty text.")
    return volume, confinement, wave_speed, provenance_


class LinearizedCarrierOpticalResponsePlan(StrictModule, NonTrainableState):
    """First-order carrier/temperature/frequency gain, index, and loss law.

    Material power gain is linearized about transparency/reference conditions.
    ``differential_power_gain`` has SI m2. Differential refractive index has SI
    m3 per carrier-pair density, and is converted to propagation-constant shift
    with ``omega / reference_wave_speed``. Internal loss is modal power loss in
    m^-1 and is never halved; only the returned field gain is half power gain.
    """

    transparency_carrier_pair_density: Array
    differential_power_gain: Array
    reference_carrier_pair_density: Array
    reference_temperature: Array
    reference_angular_frequency: Array
    gain_temperature_coefficient: Array
    gain_angular_frequency_slope: Array
    differential_refractive_index: Array
    thermo_optic_coefficient: Array
    refractive_index_angular_frequency_slope: Array
    background_internal_loss: Array
    carrier_internal_loss_cross_section: Array
    density_range: Array
    temperature_range: Array
    angular_frequency_range: Array
    active_volume: Array
    confinement_factor: Array
    reference_wave_speed: Array
    provenance: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        transparency_carrier_pair_density: ArrayLike,
        differential_power_gain: ArrayLike,
        reference_angular_frequency: ArrayLike,
        /,
        *,
        reference_carrier_pair_density: ArrayLike | None = None,
        reference_temperature: ArrayLike = 300.0,
        gain_temperature_coefficient: ArrayLike = 0.0,
        gain_angular_frequency_slope: ArrayLike = 0.0,
        differential_refractive_index: ArrayLike = 0.0,
        thermo_optic_coefficient: ArrayLike = 0.0,
        refractive_index_angular_frequency_slope: ArrayLike = 0.0,
        background_internal_loss: ArrayLike = 0.0,
        carrier_internal_loss_cross_section: ArrayLike = 0.0,
        density_range: ArrayLike,
        temperature_range: ArrayLike,
        angular_frequency_range: ArrayLike,
        active_volume: ArrayLike = 1.0,
        confinement_factor: ArrayLike = 1.0,
        reference_wave_speed: ArrayLike = _VACUUM_WAVE_SPEED,
        provenance: str,
        model_id: str | None = None,
    ):
        transparency = _positive_scalar(
            transparency_carrier_pair_density,
            "transparency_carrier_pair_density",
            nonnegative=True,
        )
        differential_gain = _finite_scalar(
            differential_power_gain, "differential_power_gain"
        )
        if float(differential_gain) <= 0.0:
            raise ValueError("differential_power_gain must be positive.")
        frequency = _positive_scalar(
            reference_angular_frequency, "reference_angular_frequency"
        )
        reference_density = (
            transparency
            if reference_carrier_pair_density is None
            else _positive_scalar(
                reference_carrier_pair_density,
                "reference_carrier_pair_density",
                nonnegative=True,
            )
        )
        temperature = _positive_scalar(reference_temperature, "reference_temperature")
        linear = tuple(
            _finite_scalar(value, name)
            for value, name in (
                (gain_temperature_coefficient, "gain_temperature_coefficient"),
                (gain_angular_frequency_slope, "gain_angular_frequency_slope"),
                (differential_refractive_index, "differential_refractive_index"),
                (thermo_optic_coefficient, "thermo_optic_coefficient"),
                (
                    refractive_index_angular_frequency_slope,
                    "refractive_index_angular_frequency_slope",
                ),
            )
        )
        background_loss = _positive_scalar(
            background_internal_loss, "background_internal_loss", nonnegative=True
        )
        carrier_loss = _positive_scalar(
            carrier_internal_loss_cross_section,
            "carrier_internal_loss_cross_section",
            nonnegative=True,
        )
        density_bounds = _range(
            density_range, "density_range", nonnegative=True, positive=False
        )
        temperature_bounds = _range(
            temperature_range, "temperature_range", nonnegative=False, positive=True
        )
        frequency_bounds = _range(
            angular_frequency_range,
            "angular_frequency_range",
            nonnegative=False,
            positive=True,
        )
        if not bool(
            (transparency >= density_bounds[0])
            & (transparency <= density_bounds[1])
            & (reference_density >= density_bounds[0])
            & (reference_density <= density_bounds[1])
            & (temperature >= temperature_bounds[0])
            & (temperature <= temperature_bounds[1])
            & (frequency >= frequency_bounds[0])
            & (frequency <= frequency_bounds[1])
        ):
            raise ValueError("Linearization references must lie inside declared support.")
        volume, confinement, wave_speed, provenance_ = _model_metadata(
            active_volume=active_volume,
            confinement_factor=confinement_factor,
            reference_wave_speed=reference_wave_speed,
            provenance=provenance,
        )
        self.transparency_carrier_pair_density = transparency
        self.differential_power_gain = differential_gain
        self.reference_carrier_pair_density = reference_density
        self.reference_temperature = temperature
        self.reference_angular_frequency = frequency
        (
            self.gain_temperature_coefficient,
            self.gain_angular_frequency_slope,
            self.differential_refractive_index,
            self.thermo_optic_coefficient,
            self.refractive_index_angular_frequency_slope,
        ) = linear
        self.background_internal_loss = background_loss
        self.carrier_internal_loss_cross_section = carrier_loss
        self.density_range = density_bounds
        self.temperature_range = temperature_bounds
        self.angular_frequency_range = frequency_bounds
        self.active_volume = volume
        self.confinement_factor = confinement
        self.reference_wave_speed = wave_speed
        self.provenance = provenance_
        generated = canonical_fingerprint(
            {
                "kind": "linearized-carrier-optical-response",
                "transparency_density": float(transparency).hex(),
                "differential_power_gain": float(differential_gain).hex(),
                "reference_density": float(reference_density).hex(),
                "reference_temperature": float(temperature).hex(),
                "reference_angular_frequency": float(frequency).hex(),
                "gain_temperature_coefficient": float(linear[0]).hex(),
                "gain_angular_frequency_slope": float(linear[1]).hex(),
                "differential_refractive_index": float(linear[2]).hex(),
                "thermo_optic_coefficient": float(linear[3]).hex(),
                "refractive_index_frequency_slope": float(linear[4]).hex(),
                "background_internal_loss": float(background_loss).hex(),
                "carrier_internal_loss_cross_section": float(carrier_loss).hex(),
                "density_range": array_tree_fingerprint(np.asarray(density_bounds)),
                "temperature_range": array_tree_fingerprint(
                    np.asarray(temperature_bounds)
                ),
                "angular_frequency_range": array_tree_fingerprint(
                    np.asarray(frequency_bounds)
                ),
                "active_volume": float(volume).hex(),
                "confinement_factor": float(confinement).hex(),
                "reference_wave_speed": float(wave_speed).hex(),
                "provenance": provenance_,
            }
        )
        identifier = generated if model_id is None else str(model_id)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.model_id = identifier


class TabulatedCarrierOpticalResponsePlan(StrictModule, NonTrainableState):
    """Bounded trilinear carrier/temperature/angular-frequency response table."""

    carrier_pair_density_grid: Array
    lattice_temperature_grid: Array
    angular_frequency_grid: Array
    material_power_gain_table: Array
    refractive_index_change_table: Array
    internal_loss_table: Array
    density_range: Array
    temperature_range: Array
    angular_frequency_range: Array
    active_volume: Array
    confinement_factor: Array
    reference_wave_speed: Array
    provenance: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        carrier_pair_density_grid: ArrayLike,
        lattice_temperature_grid: ArrayLike,
        angular_frequency_grid: ArrayLike,
        material_power_gain_table: ArrayLike,
        refractive_index_change_table: ArrayLike,
        internal_loss_table: ArrayLike,
        /,
        *,
        active_volume: ArrayLike = 1.0,
        confinement_factor: ArrayLike = 1.0,
        reference_wave_speed: ArrayLike = _VACUUM_WAVE_SPEED,
        provenance: str,
        model_id: str | None = None,
    ):
        axes = tuple(
            np.asarray(value)
            for value in (
                carrier_pair_density_grid,
                lattice_temperature_grid,
                angular_frequency_grid,
            )
        )
        for axis, name, allow_zero in zip(
            axes,
            (
                "carrier_pair_density_grid",
                "lattice_temperature_grid",
                "angular_frequency_grid",
            ),
            (True, False, False),
            strict=True,
        ):
            if (
                axis.ndim != 1
                or axis.size < 2
                or np.iscomplexobj(axis)
                or np.any(~np.isfinite(axis))
                or np.any(np.diff(axis) <= 0.0)
                or (axis[0] < 0.0 if allow_zero else axis[0] <= 0.0)
            ):
                raise ValueError(f"{name} must be finite, physical, and increasing.")
        expected = tuple(axis.size for axis in axes)
        tables = tuple(
            np.asarray(value)
            for value in (
                material_power_gain_table,
                refractive_index_change_table,
                internal_loss_table,
            )
        )
        if any(
            table.shape != expected
            or np.iscomplexobj(table)
            or np.any(~np.isfinite(table))
            for table in tables
        ):
            raise ValueError(
                "Every optical response table must be finite and match the axes."
            )
        if np.any(tables[2] < 0.0):
            raise ValueError("internal_loss_table must be nonnegative.")
        volume, confinement, wave_speed, provenance_ = _model_metadata(
            active_volume=active_volume,
            confinement_factor=confinement_factor,
            reference_wave_speed=reference_wave_speed,
            provenance=provenance,
        )
        self.carrier_pair_density_grid = jnp.asarray(axes[0])
        self.lattice_temperature_grid = jnp.asarray(axes[1])
        self.angular_frequency_grid = jnp.asarray(axes[2])
        self.material_power_gain_table = jnp.asarray(tables[0])
        self.refractive_index_change_table = jnp.asarray(tables[1])
        self.internal_loss_table = jnp.asarray(tables[2])
        self.density_range = jnp.asarray((axes[0][0], axes[0][-1]))
        self.temperature_range = jnp.asarray((axes[1][0], axes[1][-1]))
        self.angular_frequency_range = jnp.asarray((axes[2][0], axes[2][-1]))
        self.active_volume = volume
        self.confinement_factor = confinement
        self.reference_wave_speed = wave_speed
        self.provenance = provenance_
        generated = canonical_fingerprint(
            {
                "kind": "tabulated-carrier-optical-response",
                "density_grid": array_tree_fingerprint(axes[0]),
                "temperature_grid": array_tree_fingerprint(axes[1]),
                "angular_frequency_grid": array_tree_fingerprint(axes[2]),
                "material_power_gain": array_tree_fingerprint(tables[0]),
                "refractive_index_change": array_tree_fingerprint(tables[1]),
                "internal_loss": array_tree_fingerprint(tables[2]),
                "active_volume": float(volume).hex(),
                "confinement_factor": float(confinement).hex(),
                "reference_wave_speed": float(wave_speed).hex(),
                "provenance": provenance_,
            }
        )
        identifier = generated if model_id is None else str(model_id)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.model_id = identifier


CarrierOpticalResponsePlan = (
    LinearizedCarrierOpticalResponsePlan | TabulatedCarrierOpticalResponsePlan
)


class SemiconductorOpticalResponseEvidence(StrictModule):
    """Constitutive support, projection, and immutable provenance evidence."""

    carrier_within_support: Array
    temperature_within_support: Array
    angular_frequency_within_support: Array
    projection_valid: Array
    finite: Array
    nonnegative_internal_loss: Array
    successful: Array
    status: Array
    model_id: str = eqx.field(static=True)
    model_provenance: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    projection_provenance: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class SemiconductorOpticalResponseResult(StrictModule):
    """Frozen modal power/field gain, index shift, loss, and support evidence."""

    carrier_pair_density: Array
    lattice_temperature: Array
    angular_frequency: Array
    material_power_gain: Array
    modal_power_gain: Array
    field_gain: Array
    refractive_index_change: Array
    propagation_constant_shift: Array
    internal_loss: Array
    active_volume: Array
    confinement_factor: Array
    evidence: SemiconductorOpticalResponseEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def status(self) -> Array:
        return self.evidence.status


def _axis_coordinate(axis: Array, value: Array, /) -> tuple[Array, Array]:
    upper = jnp.searchsorted(axis, value, side="right")
    lower = jnp.clip(upper - 1, 0, axis.size - 2)
    x0 = axis[lower]
    x1 = axis[lower + 1]
    return lower, (value - x0) / (x1 - x0)


def _trilinear(
    table: Array,
    density_axis: Array,
    temperature_axis: Array,
    frequency_axis: Array,
    density: Array,
    temperature: Array,
    frequency: Array,
    /,
) -> Array:
    i, u = _axis_coordinate(density_axis, density)
    j, v = _axis_coordinate(temperature_axis, temperature)
    k, w = _axis_coordinate(frequency_axis, frequency)
    c000 = table[i, j, k]
    c100 = table[i + 1, j, k]
    c010 = table[i, j + 1, k]
    c110 = table[i + 1, j + 1, k]
    c001 = table[i, j, k + 1]
    c101 = table[i + 1, j, k + 1]
    c011 = table[i, j + 1, k + 1]
    c111 = table[i + 1, j + 1, k + 1]
    low = (1.0 - v) * ((1.0 - u) * c000 + u * c100) + v * ((1.0 - u) * c010 + u * c110)
    high = (1.0 - v) * ((1.0 - u) * c001 + u * c101) + v * ((1.0 - u) * c011 + u * c111)
    return (1.0 - w) * low + w * high


def _material_response(
    plan: CarrierOpticalResponsePlan,
    density: Array,
    temperature: Array,
    frequency: Array,
    /,
) -> tuple[Array, Array, Array]:
    if isinstance(plan, LinearizedCarrierOpticalResponsePlan):
        material_gain = (
            plan.differential_power_gain
            * (density - plan.transparency_carrier_pair_density)
            + plan.gain_temperature_coefficient
            * (temperature - plan.reference_temperature)
            + plan.gain_angular_frequency_slope
            * (frequency - plan.reference_angular_frequency)
        )
        index_change = (
            plan.differential_refractive_index
            * (density - plan.reference_carrier_pair_density)
            + plan.thermo_optic_coefficient * (temperature - plan.reference_temperature)
            + plan.refractive_index_angular_frequency_slope
            * (frequency - plan.reference_angular_frequency)
        )
        internal_loss = (
            plan.background_internal_loss
            + plan.carrier_internal_loss_cross_section * density
        )
        return material_gain, index_change, internal_loss
    return (
        _trilinear(
            plan.material_power_gain_table,
            plan.carrier_pair_density_grid,
            plan.lattice_temperature_grid,
            plan.angular_frequency_grid,
            density,
            temperature,
            frequency,
        ),
        _trilinear(
            plan.refractive_index_change_table,
            plan.carrier_pair_density_grid,
            plan.lattice_temperature_grid,
            plan.angular_frequency_grid,
            density,
            temperature,
            frequency,
        ),
        _trilinear(
            plan.internal_loss_table,
            plan.carrier_pair_density_grid,
            plan.lattice_temperature_grid,
            plan.angular_frequency_grid,
            density,
            temperature,
            frequency,
        ),
    )


def evaluate_semiconductor_optical_response(
    plan: CarrierOpticalResponsePlan,
    carrier_pair_density: ArrayLike,
    lattice_temperature: ArrayLike,
    angular_frequency: ArrayLike,
    /,
    *,
    projection: ActiveRegionOpticalProjection | None = None,
) -> SemiconductorOpticalResponseResult:
    """Evaluate within declared support; rejected entries are returned as NaN."""

    if not isinstance(
        plan, (LinearizedCarrierOpticalResponsePlan, TabulatedCarrierOpticalResponsePlan)
    ):
        raise TypeError("plan must be a supported carrier optical-response plan.")
    frequency_input = jnp.asarray(angular_frequency)
    if projection is None:
        density, temperature, frequency = jnp.broadcast_arrays(
            jnp.asarray(carrier_pair_density),
            jnp.asarray(lattice_temperature),
            frequency_input,
        )
        projection_valid = jnp.ones(density.shape, dtype=jnp.bool_)
        active_volume = plan.active_volume
        confinement = plan.confinement_factor
        projection_id = "unprojected-active-region-state"
        projection_provenance = "carrier state already expressed on the active region"
    else:
        if not isinstance(projection, ActiveRegionOpticalProjection):
            raise TypeError(
                "projection must be an ActiveRegionOpticalProjection or None."
            )
        density, temperature, projection_valid = projection.project(
            carrier_pair_density, lattice_temperature
        )
        if frequency_input.shape != ():
            raise ValueError(
                "Projected optical response requires scalar angular_frequency."
            )
        frequency = frequency_input
        active_volume = projection.active_volume
        confinement = projection.confinement_factor
        projection_id = projection.projection_id
        projection_provenance = projection.provenance

    carrier_supported = (
        jnp.isfinite(density)
        & (density >= plan.density_range[0])
        & (density <= plan.density_range[1])
    )
    temperature_supported = (
        jnp.isfinite(temperature)
        & (temperature >= plan.temperature_range[0])
        & (temperature <= plan.temperature_range[1])
    )
    frequency_supported = (
        jnp.isfinite(frequency)
        & (frequency >= plan.angular_frequency_range[0])
        & (frequency <= plan.angular_frequency_range[1])
    )
    material_gain, index_change, internal_loss = _material_response(
        plan, density, temperature, frequency
    )
    modal_gain = confinement * material_gain
    field_gain = 0.5 * modal_gain
    propagation_shift = confinement * frequency * index_change / plan.reference_wave_speed
    finite = (
        jnp.isfinite(material_gain)
        & jnp.isfinite(index_change)
        & jnp.isfinite(internal_loss)
        & jnp.isfinite(modal_gain)
        & jnp.isfinite(propagation_shift)
    )
    nonnegative_loss = internal_loss >= 0.0
    successful = (
        carrier_supported
        & temperature_supported
        & frequency_supported
        & projection_valid
        & finite
        & nonnegative_loss
    )
    status = (
        jnp.where(
            carrier_supported,
            0,
            int(SemiconductorOpticalResponseStatus.CARRIER_OUTSIDE_SUPPORT),
        )
        | jnp.where(
            temperature_supported,
            0,
            int(SemiconductorOpticalResponseStatus.TEMPERATURE_OUTSIDE_SUPPORT),
        )
        | jnp.where(
            frequency_supported,
            0,
            int(SemiconductorOpticalResponseStatus.FREQUENCY_OUTSIDE_SUPPORT),
        )
        | jnp.where(
            projection_valid,
            0,
            int(SemiconductorOpticalResponseStatus.INVALID_PROJECTION),
        )
        | jnp.where(finite, 0, int(SemiconductorOpticalResponseStatus.NONFINITE))
        | jnp.where(
            nonnegative_loss,
            0,
            int(SemiconductorOpticalResponseStatus.NEGATIVE_INTERNAL_LOSS),
        )
    ).astype(jnp.int32)
    admitted = lambda value: jnp.where(successful, value, jnp.nan)
    evidence_id = canonical_fingerprint(
        {
            "kind": "semiconductor-optical-response-evidence",
            "model": plan.model_id,
            "projection": projection_id,
        }
    )
    evidence = SemiconductorOpticalResponseEvidence(
        carrier_supported,
        temperature_supported,
        frequency_supported,
        projection_valid,
        finite,
        nonnegative_loss,
        successful,
        status,
        plan.model_id,
        plan.provenance,
        projection_id,
        projection_provenance,
        evidence_id,
    )
    return SemiconductorOpticalResponseResult(
        density,
        temperature,
        frequency,
        admitted(material_gain),
        admitted(modal_gain),
        admitted(field_gain),
        admitted(index_change),
        admitted(propagation_shift),
        admitted(internal_loss),
        active_volume,
        confinement,
        evidence,
    )


__all__ = [
    "ActiveRegionOpticalProjection",
    "CarrierOpticalResponsePlan",
    "LinearizedCarrierOpticalResponsePlan",
    "SemiconductorOpticalResponseEvidence",
    "SemiconductorOpticalResponseResult",
    "SemiconductorOpticalResponseStatus",
    "TabulatedCarrierOpticalResponsePlan",
    "evaluate_semiconductor_optical_response",
]

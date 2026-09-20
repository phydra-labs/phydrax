#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import apply_gather_stencil, rectilinear_stencil
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest


class SuperconductingMaterialEvaluation(StrictModule):
    critical_current_density: Array
    superconducting_current_density: Array
    stabilizer_current_density: Array
    electric_field: Array
    joule_power_density: Array
    current_sharing_margin: Array
    temperature_margin: Array
    supported: Array
    finite: Array
    successful: Array
    material_id: str = eqx.field(static=True)


class SuperconductingMaterialLawPlan(StrictModule, NonTrainableState):
    temperature_axis: Array
    magnetic_field_axis: Array
    angle_axis: Array
    critical_current_density_table: Array
    critical_temperature: float = eqx.field(static=True)
    stabilizer_resistivity: float = eqx.field(static=True)
    criterion_electric_field: float = eqx.field(static=True)
    power_law_exponent: float = eqx.field(static=True)
    manifest: ReferenceArtifactManifest = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperature_axis: ArrayLike,
        magnetic_field_axis: ArrayLike,
        angle_axis: ArrayLike,
        critical_current_density: ArrayLike,
        manifest: ReferenceArtifactManifest,
        /,
        *,
        critical_temperature: float,
        stabilizer_resistivity: float,
        criterion_electric_field: float = 1.0e-4,
        power_law_exponent: float = 20.0,
        commercial_use: bool = False,
        export: bool = False,
    ):
        temperature = np.asarray(temperature_axis, dtype=np.float64)
        field = np.asarray(magnetic_field_axis, dtype=np.float64)
        angle = np.asarray(angle_axis, dtype=np.float64)
        current = np.asarray(critical_current_density, dtype=np.float64)
        values = tuple(
            float(value)
            for value in (
                critical_temperature,
                stabilizer_resistivity,
                criterion_electric_field,
                power_law_exponent,
            )
        )
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("Superconducting material law requires a manifest.")
        manifest.require_rights(commercial_use=commercial_use, export=export)
        if (
            any(axis.ndim != 1 or axis.size < 2 for axis in (temperature, field, angle))
            or any(
                np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0)
                for axis in (temperature, field, angle)
            )
            or current.shape != (temperature.size, field.size, angle.size)
            or np.any(~np.isfinite(current))
            or np.any(current < 0.0)
            or any(not np.isfinite(value) for value in values)
            or values[0] <= temperature[0]
            or values[1] <= 0.0
            or values[2] <= 0.0
            or values[3] <= 1.0
        ):
            raise ValueError("Superconducting material table or parameters are invalid.")
        self.temperature_axis = jnp.asarray(temperature)
        self.magnetic_field_axis = jnp.asarray(field)
        self.angle_axis = jnp.asarray(angle)
        self.critical_current_density_table = jnp.asarray(current)
        self.critical_temperature = values[0]
        self.stabilizer_resistivity = values[1]
        self.criterion_electric_field = values[2]
        self.power_law_exponent = values[3]
        self.manifest = manifest
        self.material_id = canonical_fingerprint(
            {
                "kind": "tabulated-superconducting-material-law",
                "temperature": array_tree_fingerprint(temperature),
                "field": array_tree_fingerprint(field),
                "angle": array_tree_fingerprint(angle),
                "critical_current_density": array_tree_fingerprint(current),
                "critical_temperature": values[0],
                "stabilizer_resistivity": values[1],
                "criterion_electric_field": values[2],
                "power_law_exponent": values[3],
                "manifest": manifest.manifest_id,
            }
        )

    def evaluate(
        self,
        temperature: ArrayLike,
        magnetic_field: ArrayLike,
        field_angle: ArrayLike,
        total_current_density: ArrayLike,
        /,
    ) -> SuperconductingMaterialEvaluation:
        temperature_, field, angle, total = jnp.broadcast_arrays(
            jnp.asarray(temperature),
            jnp.asarray(magnetic_field),
            jnp.asarray(field_angle),
            jnp.asarray(total_current_density),
        )
        query = jnp.stack((temperature_, field, angle), axis=-1)
        stencil = rectilinear_stencil(
            (self.temperature_axis, self.magnetic_field_axis, self.angle_axis),
            query,
            boundary=("constant", "constant", "constant"),
        )
        critical = apply_gather_stencil(
            self.critical_current_density_table.reshape((-1,)), stencil
        )
        critical_density = jnp.where(
            temperature_ < self.critical_temperature,
            critical.values,
            0.0,
        )
        absolute_total = jnp.abs(total)
        superconducting = jnp.minimum(absolute_total, critical_density)
        stabilizer = jnp.maximum(absolute_total - superconducting, 0.0)
        safe_critical = jnp.maximum(critical_density, jnp.finfo(temperature_.dtype).tiny)
        superconducting_field = (
            self.criterion_electric_field
            * (superconducting / safe_critical) ** self.power_law_exponent
        )
        stabilizer_field = self.stabilizer_resistivity * stabilizer
        electric = jnp.sign(total) * (superconducting_field + stabilizer_field)
        joule = total * electric
        sharing_margin = critical_density - absolute_total
        temperature_margin = self.critical_temperature - temperature_
        supported = stencil.support & jnp.isfinite(total)
        finite = (
            jnp.isfinite(critical_density) & jnp.isfinite(electric) & jnp.isfinite(joule)
        )
        successful = supported & finite & (critical_density >= 0.0) & (joule >= 0.0)
        return SuperconductingMaterialEvaluation(
            critical_density,
            jnp.sign(total) * superconducting,
            jnp.sign(total) * stabilizer,
            electric,
            joule,
            sharing_margin,
            temperature_margin,
            supported,
            finite,
            successful,
            self.material_id,
        )


__all__ = ["SuperconductingMaterialEvaluation", "SuperconductingMaterialLawPlan"]

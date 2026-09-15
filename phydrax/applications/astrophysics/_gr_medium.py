#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import apply_gather_stencil, GatherStencil, rectilinear_stencil
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._spacetime_conventions import RelativityConvention
from ...units import derived_unit, TESLA, UnitDefinition


if TYPE_CHECKING:
    from ._gr_transfer import PolarizedRayPath


class GRMediumFieldUnits(StrictModule, NonTrainableState):
    """Units of one GR plasma field set bound to a relativity scale."""

    scale: RelativityScaleContract
    coordinate_unit: UnitDefinition = eqx.field(static=True)
    coordinate_time_unit: UnitDefinition = eqx.field(static=True)
    rest_mass_density_unit: UnitDefinition = eqx.field(static=True)
    electron_number_density_unit: UnitDefinition = eqx.field(static=True)
    electron_temperature_unit: UnitDefinition = eqx.field(static=True)
    magnetic_field_unit: UnitDefinition = eqx.field(static=True)
    units_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        /,
        *,
        magnetic_field_unit: UnitDefinition = TESLA,
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not isinstance(magnetic_field_unit, UnitDefinition):
            raise TypeError("magnetic_field_unit must be a UnitDefinition.")
        length = scale.dimensional_scale.length_unit
        number_density = derived_unit(f"1/{length.symbol}^3", ((length, -3),))
        if (
            magnetic_field_unit.dimension != TESLA.dimension
            or magnetic_field_unit.reference_system_id != length.reference_system_id
        ):
            raise ValueError(
                "Magnetic-field units must have magnetic-flux-density dimension and "
                "share the relativity scale's reference system."
            )
        self.scale = scale
        self.coordinate_unit = length
        self.coordinate_time_unit = length
        self.rest_mass_density_unit = scale.mass_density_unit
        self.electron_number_density_unit = number_density
        self.electron_temperature_unit = scale.temperature_unit
        self.magnetic_field_unit = magnetic_field_unit
        self.units_id = canonical_fingerprint(
            {
                "kind": "gr-medium-field-units",
                "scale": scale.scale_id,
                "coordinate_unit": length.unit_id,
                "coordinate_time_unit": length.unit_id,
                "magnetic_field": magnetic_field_unit.unit_id,
            }
        )


class GRMediumSampleEvidence(StrictModule):
    finite: Array
    in_support: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRMediumSample(StrictModule):
    rest_mass_density: Array
    electron_number_density: Array
    electron_temperature: Array
    fluid_four_velocity: Array
    magnetic_four_vector: Array
    evidence: GRMediumSampleEvidence
    source_id: str = eqx.field(static=True)
    units_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)


class FixedGRFieldSamplingPlan(StrictModule, NonTrainableState):
    """Prepared eight-corner spatial gather with fixed query capacity."""

    stencil: GatherStencil
    interpolation_smooth: Array
    query_shape: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stencil: GatherStencil,
        interpolation_smooth: ArrayLike,
        /,
        *,
        query_shape: Sequence[int],
        source_id: str,
        query_fingerprint: str,
    ):
        shape = tuple(int(value) for value in query_shape)
        smooth = jax.lax.stop_gradient(jnp.asarray(interpolation_smooth, dtype=bool))
        if not isinstance(stencil, GatherStencil):
            raise TypeError("stencil must be a GatherStencil.")
        if smooth.shape != shape or stencil.support.shape != shape:
            raise ValueError("Sampling smoothness and stencil query shapes must match.")
        identifier = str(source_id).strip()
        query_id = str(query_fingerprint).strip()
        if not identifier or not query_id:
            raise ValueError("Sampling source and query identities must be non-empty.")
        self.stencil = stencil
        self.interpolation_smooth = smooth
        self.query_shape = shape
        self.source_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-gr-field-sampling",
                "source": identifier,
                "query": query_id,
                "query_shape": shape,
                "stencil_capacity": int(stencil.indices.shape[-1]),
            }
        )


def _runtime_sampling_stencil(
    axis_values: tuple[Array, ...], coordinates: Array, /
) -> tuple[GatherStencil, Array]:
    finite = jnp.all(jnp.isfinite(coordinates), axis=-1)
    fallback = jnp.stack(tuple(axis[0] for axis in axis_values))
    safe_coordinates = jnp.where(finite[..., None], coordinates, fallback)
    base = rectilinear_stencil(
        axis_values,
        safe_coordinates,
        boundary=("constant",) * len(axis_values),
    )
    stencil = GatherStencil(
        indices=base.indices,
        weights=base.weights,
        source_size=base.source_size,
        valid=base.valid,
        support=base.support & finite,
    )
    smooth = finite & base.support
    for axis, nodes in enumerate(axis_values):
        coordinate = coordinates[..., axis]
        smooth = smooth & (coordinate > nodes[0]) & (coordinate < nodes[-1])
        if int(nodes.size) > 2:
            smooth = smooth & ~jnp.any(coordinate[..., None] == nodes[1:-1], axis=-1)
    return stencil, smooth


def _sample_medium_fields(
    stencil: GatherStencil,
    interpolation_smooth: Array,
    source_id: str,
    rest_mass_density: Array,
    electron_number_density: Array,
    electron_temperature: Array,
    fluid_four_velocity: Array,
    magnetic_four_vector: Array,
    source_mask: Array,
    /,
    *,
    units_id: str,
    convention_id: str,
    chart_id: str,
) -> GRMediumSample:
    spatial_shape = source_mask.shape
    source_size = int(np.prod(spatial_shape))
    if stencil.source_size != source_size:
        raise ValueError(
            "Sampling stencil source capacity does not match medium storage."
        )
    mask = source_mask.reshape((source_size,))

    def gather(values: Array):
        payload_shape = values.shape[len(spatial_shape) :]
        return apply_gather_stencil(
            values.reshape((source_size,) + payload_shape),
            stencil,
            source_mask=mask,
            mask_mode="strict",
        )

    rest = gather(rest_mass_density)
    number = gather(electron_number_density)
    temperature = gather(electron_temperature)
    velocity = gather(fluid_four_velocity)
    magnetic = gather(magnetic_four_vector)
    in_support = (
        rest.support
        & number.support
        & temperature.support
        & velocity.support
        & magnetic.support
    )
    finite = (
        in_support
        & jnp.isfinite(rest.values)
        & jnp.isfinite(number.values)
        & jnp.isfinite(temperature.values)
        & jnp.all(jnp.isfinite(velocity.values), axis=-1)
        & jnp.all(jnp.isfinite(magnetic.values), axis=-1)
    )
    physically_valid = (
        in_support
        & (rest.values >= 0.0)
        & (number.values >= 0.0)
        & (temperature.values > 0.0)
    )
    qualified = finite & physically_valid
    evidence = GRMediumSampleEvidence(
        finite,
        in_support,
        physically_valid,
        qualified,
        qualified & interpolation_smooth,
    )
    return GRMediumSample(
        rest.values,
        number.values,
        temperature.values,
        velocity.values,
        magnetic.values,
        evidence,
        source_id,
        units_id,
        convention_id,
        chart_id,
    )


class FastLightSnapshot(StrictModule, NonTrainableState):
    """One immutable rectilinear GR plasma snapshot for fast-light sampling.

    All four chart coordinates, including ``coordinate_time``, use the bound
    scale's length unit (the temporal chart coordinate is ``c t``).  Scalar
    plasma fields use :class:`GRMediumFieldUnits`; four-vectors are contravariant
    components in ``chart_id`` under ``convention``.  The snapshot does not
    assert a metric-dependent four-vector normalization.
    """

    coordinate_axes: tuple[Array, Array, Array]
    coordinate_time: Array
    rest_mass_density: Array
    electron_number_density: Array
    electron_temperature: Array
    fluid_four_velocity: Array
    magnetic_four_vector: Array
    source_mask: Array
    units: GRMediumFieldUnits
    convention: RelativityConvention
    spatial_shape: tuple[int, int, int] = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_axes: Sequence[ArrayLike],
        coordinate_time: ArrayLike,
        rest_mass_density: ArrayLike,
        electron_number_density: ArrayLike,
        electron_temperature: ArrayLike,
        fluid_four_velocity: ArrayLike,
        magnetic_four_vector: ArrayLike,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        source_mask: ArrayLike | None = None,
        magnetic_field_unit: UnitDefinition = TESLA,
        chart_id: str,
        source_id: str,
    ):
        axes_host = tuple(np.asarray(axis, dtype=float) for axis in coordinate_axes)
        if (
            len(axes_host) != 3
            or any(axis.ndim != 1 or axis.size < 2 for axis in axes_host)
            or any(
                np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0)
                for axis in axes_host
            )
        ):
            raise ValueError(
                "Fast-light coordinate axes must be three finite increasing vectors."
            )
        spatial_shape = tuple(int(axis.size) for axis in axes_host)
        rest_host = np.asarray(rest_mass_density, dtype=float)
        number_host = np.asarray(electron_number_density, dtype=float)
        temperature_host = np.asarray(electron_temperature, dtype=float)
        velocity_host = np.asarray(fluid_four_velocity, dtype=float)
        magnetic_host = np.asarray(magnetic_four_vector, dtype=float)
        mask_host = (
            np.ones(spatial_shape, dtype=bool)
            if source_mask is None
            else np.asarray(source_mask, dtype=bool)
        )
        time_host = np.asarray(coordinate_time, dtype=float)
        if (
            rest_host.shape != spatial_shape
            or number_host.shape != spatial_shape
            or temperature_host.shape != spatial_shape
            or velocity_host.shape != spatial_shape + (4,)
            or magnetic_host.shape != spatial_shape + (4,)
            or mask_host.shape != spatial_shape
            or time_host.shape != ()
        ):
            raise ValueError("Fast-light fields do not match the rectilinear grid shape.")
        if not np.any(mask_host):
            raise ValueError(
                "Fast-light source mask must contain at least one active node."
            )
        active_scalars = (
            rest_host[mask_host],
            number_host[mask_host],
            temperature_host[mask_host],
        )
        if (
            not np.isfinite(time_host)
            or any(np.any(~np.isfinite(value)) for value in active_scalars)
            or np.any(~np.isfinite(velocity_host[mask_host]))
            or np.any(~np.isfinite(magnetic_host[mask_host]))
            or np.any(active_scalars[0] < 0.0)
            or np.any(active_scalars[1] < 0.0)
            or np.any(active_scalars[2] <= 0.0)
        ):
            raise ValueError(
                "Active fast-light fields must be finite with physical scalar states."
            )
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be a RelativityConvention.")
        chart = str(chart_id).strip()
        provenance = str(source_id).strip()
        if not chart or not provenance:
            raise ValueError("Fast-light chart and source identities must be non-empty.")
        units = GRMediumFieldUnits(scale, magnetic_field_unit=magnetic_field_unit)
        arrays = tuple(jax.lax.stop_gradient(jnp.asarray(axis)) for axis in axes_host)
        self.coordinate_axes = (arrays[0], arrays[1], arrays[2])
        self.coordinate_time = jax.lax.stop_gradient(jnp.asarray(time_host))
        self.rest_mass_density = jax.lax.stop_gradient(jnp.asarray(rest_host))
        self.electron_number_density = jax.lax.stop_gradient(jnp.asarray(number_host))
        self.electron_temperature = jax.lax.stop_gradient(jnp.asarray(temperature_host))
        self.fluid_four_velocity = jax.lax.stop_gradient(jnp.asarray(velocity_host))
        self.magnetic_four_vector = jax.lax.stop_gradient(jnp.asarray(magnetic_host))
        self.source_mask = jax.lax.stop_gradient(jnp.asarray(mask_host))
        self.units = units
        self.convention = convention
        self.spatial_shape = (
            spatial_shape[0],
            spatial_shape[1],
            spatial_shape[2],
        )
        self.chart_id = chart
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "fast-light-gr-medium-snapshot",
                "source_id": provenance,
                "chart": chart,
                "units": units.units_id,
                "convention": convention.convention_id,
                "arrays": array_tree_fingerprint(
                    (
                        axes_host,
                        time_host,
                        rest_host,
                        number_host,
                        temperature_host,
                        velocity_host,
                        magnetic_host,
                        mask_host,
                    )
                ),
            }
        )

    def prepare_sampling(self, coordinates: ArrayLike, /) -> FixedGRFieldSamplingPlan:
        query_host = np.asarray(coordinates, dtype=float)
        if query_host.ndim < 1 or query_host.shape[-1] != 3:
            raise ValueError("Fast-light query coordinates must end in three components.")
        stencil, smooth = _runtime_sampling_stencil(
            self.coordinate_axes, jnp.asarray(query_host)
        )
        return FixedGRFieldSamplingPlan(
            stencil,
            smooth,
            query_shape=query_host.shape[:-1],
            source_id=self.snapshot_id,
            query_fingerprint=array_tree_fingerprint(query_host),
        )

    def prepare_path_sampling(
        self, path: PolarizedRayPath, /
    ) -> FixedGRFieldSamplingPlan:
        """Prepare active segment-midpoint sampling for one exact GR ray path."""

        from ._gr_transfer import PolarizedRayPath

        if not isinstance(path, PolarizedRayPath):
            raise TypeError("path must be a PolarizedRayPath.")
        if (
            self.chart_id != path.chart_id
            or self.convention.convention_id != path.convention_id
            or self.units.scale.scale_id != path.scale_id
            or self.units.coordinate_unit.unit_id != path.coordinate_unit_id
        ):
            raise ValueError(
                "Fast-light snapshot and GR ray path must share exact chart, "
                "relativity convention, scale, and coordinate unit identities."
            )
        midpoints = 0.5 * (path.coordinates[:-1, 1:] + path.coordinates[1:, 1:])
        prepared = self.prepare_sampling(midpoints)
        segment_support = (
            path.active[:-1] & path.active[1:] & path.valid[:-1] & path.valid[1:]
        )
        stencil = GatherStencil(
            indices=prepared.stencil.indices,
            weights=prepared.stencil.weights,
            source_size=prepared.stencil.source_size,
            valid=prepared.stencil.valid,
            support=prepared.stencil.support & segment_support,
        )
        return FixedGRFieldSamplingPlan(
            stencil,
            prepared.interpolation_smooth & segment_support,
            query_shape=midpoints.shape[:-1],
            source_id=self.snapshot_id,
            query_fingerprint=canonical_fingerprint(
                {
                    "kind": "fast-light-ray-path-midpoints",
                    "path": path.path_id,
                    "snapshot": self.snapshot_id,
                }
            ),
        )

    def sample(self, plan: FixedGRFieldSamplingPlan, /) -> GRMediumSample:
        if not isinstance(plan, FixedGRFieldSamplingPlan):
            raise TypeError("plan must be a FixedGRFieldSamplingPlan.")
        if plan.source_id != self.snapshot_id:
            raise ValueError("Sampling plan was prepared for a different snapshot.")
        return _sample_medium_fields(
            plan.stencil,
            plan.interpolation_smooth,
            plan.source_id,
            self.rest_mass_density,
            self.electron_number_density,
            self.electron_temperature,
            self.fluid_four_velocity,
            self.magnetic_four_vector,
            self.source_mask,
            units_id=self.units.units_id,
            convention_id=self.convention.convention_id,
            chart_id=self.chart_id,
        )

    def evaluate(self, coordinates: ArrayLike, /) -> GRMediumSample:
        """Sample dynamic coordinates through a fixed eight-corner JAX stencil."""

        query = jnp.asarray(coordinates)
        if query.ndim < 1 or query.shape[-1] != 3:
            raise ValueError("Fast-light query coordinates must end in three components.")
        stencil, smooth = _runtime_sampling_stencil(self.coordinate_axes, query)
        return _sample_medium_fields(
            stencil,
            smooth,
            self.snapshot_id,
            self.rest_mass_density,
            self.electron_number_density,
            self.electron_temperature,
            self.fluid_four_velocity,
            self.magnetic_four_vector,
            self.source_mask,
            units_id=self.units.units_id,
            convention_id=self.convention.convention_id,
            chart_id=self.chart_id,
        )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FiniteVolumeDiscretization
from ...equations._relativistic_hydrodynamics import ValenciaGeometrySource
from ...metrix import (
    adm_extrinsic_curvature,
    ChartTransition,
    CoordinateChart,
    decompose_adm_metric,
    ingoing_kerr_domain_evidence,
    ingoing_kerr_metric,
    pullback_lorentzian_metric,
    RelativityConvention,
)
from ...metrix._adm_exchange import ADMGridGeometry
from ...solver._grrmhd_runtime import FixedGridGRRMHDIMEXPlan, GRRMHDState
from ...solver._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry
from ..astrophysics._gr_medium import FastLightSnapshot
from ._accretion import FishboneMoncriefTorusPlan


class _IngoingKerrTimeMap(StrictModule, NonTrainableState):
    direction: int = eqx.field(static=True)

    def __init__(self, direction: int, /) -> None:
        direction_ = int(direction)
        if direction_ not in (-1, 1):
            raise ValueError("Ingoing Kerr time-map direction must be -1 or 1.")
        self.direction = direction_

    def __call__(self, coordinates: Array, /) -> Array:
        return coordinates.at[0].add(self.direction * coordinates[1])


class IngoingKerrGridPlan(StrictModule, NonTrainableState):
    """ADM stages on spacelike horizon-penetrating ingoing-Kerr time slices.

    The spatial chart is ``(r, theta, phi_tilde)`` and the time coordinate is
    ``t_ingoing = v - r``. Unlike constant advanced-time ``v`` slices, these
    hypersurfaces are spacelike and therefore admit a 3+1 decomposition.
    """

    discretization: FiniteVolumeDiscretization
    scale: RelativityScaleContract
    convention: RelativityConvention
    chart: CoordinateChart
    mass: float = eqx.field(static=True)
    spin: float = eqx.field(static=True)
    metric: object
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        mass: float,
        spin: float,
        /,
        *,
        chart: CoordinateChart | None = None,
    ) -> None:
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        if len(discretization.cell_shape) != 3:
            raise ValueError("Ingoing Kerr GRRMHD requires a three-dimensional grid.")
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        mass_ = float(mass)
        spin_ = float(spin)
        if (
            not np.isfinite(mass_)
            or mass_ <= 0.0
            or not np.isfinite(spin_)
            or abs(spin_) > mass_
        ):
            raise ValueError("Ingoing Kerr mass and spin are invalid.")
        chart_ = (
            CoordinateChart(
                "ingoing-kerr-spacelike",
                ("ingoing_time", "radius", "polar", "ingoing_azimuth"),
            )
            if chart is None
            else chart
        )
        if not isinstance(chart_, CoordinateChart) or chart_.dimension != 4:
            raise TypeError("chart must be a four-dimensional CoordinateChart.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("Ingoing Kerr GRRMHD requires mostly-plus convention.")
        advanced_chart = CoordinateChart(
            f"{chart_.name}:advanced",
            ("advanced_time", "radius", "polar", "ingoing_azimuth"),
        )
        advanced_metric = ingoing_kerr_metric(
            mass_, spin_, chart=advanced_chart, convention="mostly_plus"
        )
        transition = ChartTransition(
            chart_,
            advanced_chart,
            _IngoingKerrTimeMap(1),
            inverse=_IngoingKerrTimeMap(-1),
        )
        metric = pullback_lorentzian_metric(advanced_metric, transition)
        self.discretization = discretization
        self.scale = scale
        self.convention = convention
        self.chart = chart_
        self.mass = mass_
        self.spin = spin_
        self.metric = metric
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ingoing-kerr-grrmhd-grid",
                "discretization": discretization.prepared_id,
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "chart": (chart_.name, list(chart_.coordinates)),
                "mass": mass_,
                "spin": spin_,
            }
        )

    def _spatial_coordinates(self, *, face_axis: int | None = None) -> Array:
        axes = []
        for axis, structured in enumerate(self.discretization.grid.structured_axes):
            axes.append(
                structured.point_coordinates
                if face_axis == axis
                else structured.interval_centers
            )
        mesh = jnp.meshgrid(*axes, indexing="ij")
        return jnp.stack(mesh, axis=-1)

    def _geometry(
        self,
        spatial_coordinates: Array,
        time: Array,
        snapshot_token: Array,
        lineage_suffix: str,
        /,
    ) -> ADMGridGeometry:
        time_field = jnp.broadcast_to(time, spatial_coordinates.shape[:-1])
        coordinates = jnp.concatenate(
            (time_field[..., None], spatial_coordinates), axis=-1
        )
        decomposition = decompose_adm_metric(self.metric, coordinates)
        extrinsic = adm_extrinsic_curvature(self.metric, coordinates)
        domain = ingoing_kerr_domain_evidence(
            self.mass,
            self.spin,
            coordinates,
            chart=self.chart,
        )
        determinant = jnp.sqrt(jnp.linalg.det(decomposition.spatial_metric))
        active = domain.physically_valid
        valid = domain.qualified
        return ADMGridGeometry(
            decomposition.lapse,
            decomposition.shift,
            decomposition.spatial_metric,
            decomposition.spatial_inverse,
            determinant,
            extrinsic,
            active,
            valid,
            snapshot_token=snapshot_token,
            chart_id=self.chart.name,
            convention_id=self.convention.convention_id,
            scale_id=self.scale.scale_id,
            topology_id=self.discretization.grid.topology.topology_id,
            geometry_lineage_id=canonical_fingerprint(
                {
                    "kind": "ingoing-kerr-adm-grid",
                    "plan": self.plan_id,
                    "location": lineage_suffix,
                }
            ),
        )

    def _source_derivatives(
        self,
        spatial_coordinates: Array,
        time: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        flat = spatial_coordinates.reshape((-1, 3))

        def fields(point):
            coordinate = jnp.concatenate((time[None], point))
            decomposition = decompose_adm_metric(self.metric, coordinate)
            return (
                decomposition.lapse,
                decomposition.shift,
                decomposition.spatial_metric,
            )

        lapse, shift, spatial = jax.vmap(jax.jacfwd(fields))(flat)
        shape = spatial_coordinates.shape[:-1]
        lapse = lapse.reshape(shape + (3,))
        shift = jnp.swapaxes(shift.reshape(shape + (3, 3)), -1, -2)
        spatial = jnp.moveaxis(spatial.reshape(shape + (3, 3, 3)), -1, -3)
        return lapse, shift, spatial

    def stage(
        self,
        time: ArrayLike,
        snapshot_token: ArrayLike,
        /,
    ) -> ValenciaFiniteVolumeStageGeometry:
        time_ = jnp.asarray(time)
        token = jnp.asarray(snapshot_token, dtype=jnp.int32)
        if time_.shape != () or token.shape != ():
            raise ValueError("Kerr stage time and snapshot token must be scalar.")
        cells = self._spatial_coordinates()
        cell_geometry = self._geometry(cells, time_, token, "cells")
        lapse, shift, spatial = self._source_derivatives(cells, time_)
        source = ValenciaGeometrySource(cell_geometry, lapse, shift, spatial)
        faces = tuple(
            self._geometry(
                self._spatial_coordinates(face_axis=axis),
                time_,
                token,
                f"faces:{axis}",
            )
            for axis in range(3)
        )
        return ValenciaFiniteVolumeStageGeometry(source, faces, time_)


class GRRMHDTorusInitialData(StrictModule):
    state: GRRMHDState
    primitive: Array
    radiation_moments: Array
    vector_potential: Array
    geometry: ValenciaFiniteVolumeStageGeometry
    finite: Array
    physically_valid: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRRMHDTorusInitialDataPlan(StrictModule, NonTrainableState):
    """Lower Fishbone-Moncrief data into one ingoing-Kerr GRRMHD state."""

    runtime: FixedGridGRRMHDIMEXPlan
    geometry: IngoingKerrGridPlan
    torus: FishboneMoncriefTorusPlan
    radiation_constant: float = eqx.field(static=True)
    radiation_energy_scale: float = eqx.field(static=True)
    caloric_temperature_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: FixedGridGRRMHDIMEXPlan,
        geometry: IngoingKerrGridPlan,
        torus: FishboneMoncriefTorusPlan,
        /,
        *,
        radiation_constant: float = 1.0,
        radiation_energy_scale: float = 1.0,
        caloric_temperature_scale: float = 1.0,
    ) -> None:
        if not isinstance(runtime, FixedGridGRRMHDIMEXPlan):
            raise TypeError("runtime must be FixedGridGRRMHDIMEXPlan.")
        if not isinstance(geometry, IngoingKerrGridPlan):
            raise TypeError("geometry must be IngoingKerrGridPlan.")
        if not isinstance(torus, FishboneMoncriefTorusPlan):
            raise TypeError("torus must be FishboneMoncriefTorusPlan.")
        values = tuple(
            float(value)
            for value in (
                radiation_constant,
                radiation_energy_scale,
                caloric_temperature_scale,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("GRRMHD torus radiation controls must be positive.")
        if (
            runtime.material_transport.system.eos.eos_id != torus.eos.eos_id
            or geometry.mass != torus.mass_parameter
            or geometry.spin != torus.spin_parameter
        ):
            raise ValueError(
                "Torus, Kerr geometry, and GRRMHD runtime identities differ."
            )
        self.runtime = runtime
        self.geometry = geometry
        self.torus = torus
        self.radiation_constant = values[0]
        self.radiation_energy_scale = values[1]
        self.caloric_temperature_scale = values[2]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "grrmhd-fishbone-moncrief-initial-data",
                "runtime": runtime.plan_id,
                "geometry": geometry.plan_id,
                "torus": torus.plan_id,
                "radiation_constant": values[0],
                "radiation_energy_scale": values[1],
                "caloric_temperature_scale": values[2],
            }
        )

    def _edge_vector_potential(self, /) -> Array:
        bridge = self.runtime.material_transport.constrained_transport.bridge
        degree = 1
        coordinates = bridge.cochain.coordinates[degree]
        torus = self.torus.evaluate(coordinates)
        potential = jnp.zeros(
            (bridge.cochain.cell_counts[degree],), dtype=coordinates.dtype
        )
        measures = bridge.cochain.primal_measures[degree]
        for orientation, shape, offset in zip(
            bridge.orientations[degree],
            bridge.orientation_shapes[degree],
            bridge.orientation_offsets[degree],
            strict=True,
        ):
            count = int(np.prod(shape))
            if orientation == (2,):
                potential = potential.at[offset : offset + count].set(
                    torus.vector_potential_covector[offset : offset + count, 2]
                    * measures[offset : offset + count]
                )
        return potential

    def initialize(
        self,
        /,
        *,
        time: ArrayLike = 0.0,
        step_size: ArrayLike | None = None,
        snapshot_token: ArrayLike = 1,
    ) -> GRRMHDTorusInitialData:
        stage = self.geometry.stage(time, snapshot_token)
        coordinates = self.geometry._spatial_coordinates()
        torus = self.torus.evaluate(coordinates)
        coordinate_velocity = (
            jnp.zeros_like(torus.primitive[..., 1:4])
            .at[..., 2]
            .set(torus.angular_velocity)
        )
        target_velocity = (
            coordinate_velocity + stage.cell.beta_contravariant
        ) / stage.cell.alpha[..., None]
        primitive = torus.primitive.at[..., 1:4].set(target_velocity)
        potential = self._edge_vector_potential()
        constrained = self.runtime.material_transport.constrained_transport
        magnetic_flux = constrained.bridge.exterior_derivative(1, potential)
        magnetic = constrained.cell_physical_magnetic_field(
            magnetic_flux, stage.cell.sqrt_det_spatial_metric
        )
        primitive = primitive.at[..., 5:8].set(magnetic)
        conserved = self.runtime.material_transport.system.primitive_to_conserved(
            primitive, stage.cell
        )
        temperature = (
            self.caloric_temperature_scale * primitive[..., 4] / primitive[..., 0]
        )
        radiation_energy = (
            self.radiation_energy_scale * self.radiation_constant * temperature**4
        )
        radiation_moments = (
            jnp.zeros(primitive.shape[:-1] + (4,), dtype=primitive.dtype)
            .at[..., 0]
            .set(radiation_energy)
        )
        gauge = constrained.gauge
        state = self.runtime.initialize(
            conserved,
            radiation_moments,
            stage,
            magnetic_flux=magnetic_flux,
            vector_potential=potential if gauge.evolves_vector_potential else None,
            step_size=step_size,
            time=time,
        )
        finite = (
            torus.finite
            & jnp.all(jnp.isfinite(primitive), axis=-1)
            & jnp.all(jnp.isfinite(radiation_moments), axis=-1)
        )
        physical = (
            torus.physically_valid
            & stage.cell.physically_valid
            & (radiation_energy > 0.0)
        )
        qualified = torus.qualified & physical
        return GRRMHDTorusInitialData(
            state,
            primitive,
            radiation_moments,
            potential,
            stage,
            finite,
            physical,
            qualified,
            self.plan_id,
        )


def grrmhd_fast_light_snapshot(
    runtime: FixedGridGRRMHDIMEXPlan,
    state: GRRMHDState,
    geometry: ValenciaFiniteVolumeStageGeometry,
    coordinate_axes: tuple[ArrayLike, ArrayLike, ArrayLike],
    /,
    *,
    electron_mass_per_particle: float,
    caloric_temperature_scale: float,
    source_id: str,
) -> FastLightSnapshot:
    """Convert one accepted GRRMHD state to the native fast-light medium."""

    if not isinstance(runtime, FixedGridGRRMHDIMEXPlan):
        raise TypeError("runtime must be FixedGridGRRMHDIMEXPlan.")
    if not isinstance(state, GRRMHDState):
        raise TypeError("state must be GRRMHDState.")
    particle_mass = float(electron_mass_per_particle)
    temperature_scale = float(caloric_temperature_scale)
    if (
        not np.isfinite(particle_mass)
        or particle_mass <= 0.0
        or not np.isfinite(temperature_scale)
        or temperature_scale <= 0.0
    ):
        raise ValueError("Fast-light plasma conversion controls are invalid.")
    full = runtime.material_transport.constrained_transport.full_state(
        state.material_state, state.constrained_transport.magnetic_flux
    )
    recovery = runtime.material_transport.system.recover(full, geometry.cell)
    primitive = recovery.primitive
    velocity = primitive[..., 1:4]
    metric = geometry.cell.spatial_metric
    velocity_covector = ein.contract("...ij,...j->...i", metric, velocity)
    speed_squared = ein.contract("...i,...i->...", velocity_covector, velocity)
    lorentz = 1.0 / jnp.sqrt(1.0 - speed_squared)
    lapse = geometry.cell.alpha
    shift = geometry.cell.beta_contravariant
    four_velocity = jnp.concatenate(
        (
            (lorentz / lapse)[..., None],
            lorentz[..., None] * (velocity - shift / lapse[..., None]),
        ),
        axis=-1,
    )
    magnetic = primitive[..., 5:8]
    magnetic_covector = ein.contract("...ij,...j->...i", metric, magnetic)
    magnetic_velocity = ein.contract("...i,...i->...", magnetic_covector, velocity)
    magnetic_time = lorentz * magnetic_velocity / lapse
    magnetic_spatial = magnetic / lorentz[..., None] + magnetic_time[..., None] * (
        lapse[..., None] * velocity - shift
    )
    magnetic_four = jnp.concatenate((magnetic_time[..., None], magnetic_spatial), axis=-1)
    density = primitive[..., 0]
    number_density = density / particle_mass
    temperature = temperature_scale * primitive[..., 4] / density
    source_mask = geometry.cell.active & recovery.qualified & jnp.isfinite(temperature)
    return FastLightSnapshot(
        coordinate_axes,
        state.time,
        density,
        number_density,
        temperature,
        four_velocity,
        magnetic_four,
        runtime.material_transport.system.scale,
        runtime.material_transport.system.convention,
        source_mask=source_mask,
        chart_id=geometry.cell.chart_id,
        source_id=source_id,
    )


__all__ = [
    "GRRMHDTorusInitialData",
    "GRRMHDTorusInitialDataPlan",
    "IngoingKerrGridPlan",
    "grrmhd_fast_light_snapshot",
]

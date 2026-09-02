#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._hydrostatic_grid import (
    HydrostaticMetricEpoch,
    PreparedHydrostaticGrid,
)
from ...linalg._tridiagonal_lines import solve_tridiagonal_lines
from ...solver._hydrostatic_free_surface import LinearImplicitFreeSurfacePlan
from ._external_mode import ExternalModeSubcyclePolicy


BoundaryKind = Literal[
    "closed", "prescribed-elevation", "prescribed-transport", "flather", "radiation"
]
MixingKind = Literal["prescribed", "ri", "kpp", "tke", "redi-gm"]
ExternalMode = Literal["implicit", "split-explicit"]


def _safe_divide(numerator: Array, denominator: Array, /) -> Array:
    valid = denominator > 0.0
    safe = jnp.where(valid, denominator, 1.0)
    return jnp.where(valid, numerator / safe, 0.0)


def _cell_from_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        centered = 0.5 * (moved + jnp.roll(moved, -1, axis=0))
    else:
        centered = 0.5 * (moved[:-1] + moved[1:])
    return jnp.moveaxis(centered, 0, axis)


def _faces_from_cell(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (jnp.roll(moved, 1, axis=0) + moved)
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _face_upwind(value: Array, flux: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    moved_flux = jnp.moveaxis(flux, axis, 0)
    if periodic:
        left = jnp.roll(moved, 1, axis=0)
        right = moved
    else:
        left = jnp.concatenate((moved[:1], moved), axis=0)
        right = jnp.concatenate((moved, moved[-1:]), axis=0)
    selected = jnp.where(moved_flux >= 0.0, left, right)
    return jnp.moveaxis(selected, 0, axis)


def _vertical_upwind(value: Array, flux: Array, /) -> Array:
    lower = jnp.concatenate((value[..., :1], value), axis=-1)
    upper = jnp.concatenate((value, value[..., -1:]), axis=-1)
    return jnp.where(flux >= 0.0, lower, upper)


class HydrostaticEOSResult(StrictModule):
    density: Array
    alpha: Array
    beta: Array
    density_pressure_derivative: Array
    valid: Array
    finite: Array
    successful: Array
    eos_id: str = eqx.field(static=True)


class LinearHydrostaticEOS(StrictModule, NonTrainableState):
    """Linear SA/CT Boussinesq equation of state."""

    reference_density: float = eqx.field(static=True)
    reference_salinity: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_density: float = 1027.0,
        reference_salinity: float = 35.0,
        reference_temperature: float = 10.0,
        alpha: float = 2.0e-4,
        beta: float = 7.6e-4,
    ):
        values = tuple(
            float(v)
            for v in (
                reference_density,
                reference_salinity,
                reference_temperature,
                alpha,
                beta,
            )
        )
        if any(not np.isfinite(v) for v in values) or values[0] <= 0.0:
            raise ValueError("Linear hydrostatic EOS parameters are invalid.")
        self.reference_density = values[0]
        self.reference_salinity = values[1]
        self.reference_temperature = values[2]
        self.alpha = values[3]
        self.beta = values[4]
        self.eos_id = canonical_fingerprint(
            {"kind": "linear-hydrostatic-eos", "values": list(values)}
        )

    def evaluate(
        self, salinity: ArrayLike, temperature: ArrayLike, pressure_dbar: ArrayLike, /
    ) -> HydrostaticEOSResult:
        salinity_ = jnp.asarray(salinity)
        temperature_ = jnp.asarray(temperature, dtype=salinity_.dtype)
        pressure = jnp.asarray(pressure_dbar, dtype=salinity_.dtype)
        if salinity_.shape != temperature_.shape or pressure.shape != salinity_.shape:
            raise ValueError("EOS SA, CT, and pressure arrays must share one shape.")
        anomaly = -self.alpha * (
            temperature_ - self.reference_temperature
        ) + self.beta * (salinity_ - self.reference_salinity)
        density = self.reference_density * (1.0 + anomaly)
        alpha = jnp.full_like(density, self.alpha)
        beta = jnp.full_like(density, self.beta)
        pressure_derivative = jnp.zeros_like(density)
        finite = jnp.all(jnp.isfinite(density))
        valid = jnp.all(salinity_ >= 0.0)
        return HydrostaticEOSResult(
            density=density,
            alpha=alpha,
            beta=beta,
            density_pressure_derivative=pressure_derivative,
            valid=valid,
            finite=finite,
            successful=finite & valid,
            eos_id=self.eos_id,
        )


class NonlinearSeawaterPolynomialEOS(StrictModule, NonTrainableState):
    """JAX-native nonlinear SA/CT/pressure seawater polynomial."""

    reference_density: float = eqx.field(static=True)
    reference_salinity: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_density: float = 1027.0,
        reference_salinity: float = 35.0,
        reference_temperature: float = 10.0,
        coefficients: Sequence[float] = (
            -2.0e-4,
            7.6e-4,
            -4.5e-6,
            1.2e-6,
            8.0e-10,
            2.5e-8,
            -1.5e-8,
        ),
    ):
        values = tuple(float(v) for v in coefficients)
        if len(values) != 7 or any(not np.isfinite(v) for v in values):
            raise ValueError("Nonlinear seawater EOS requires seven finite coefficients.")
        density = float(reference_density)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("Nonlinear seawater reference density must be positive.")
        self.reference_density = density
        self.reference_salinity = float(reference_salinity)
        self.reference_temperature = float(reference_temperature)
        self.coefficients = values
        self.eos_id = canonical_fingerprint(
            {
                "kind": "nonlinear-seawater-polynomial-eos",
                "rho0": density,
                "s0": self.reference_salinity,
                "t0": self.reference_temperature,
                "coefficients": list(values),
                "pressure_units": "dbar",
            }
        )

    def evaluate(
        self, salinity: ArrayLike, temperature: ArrayLike, pressure_dbar: ArrayLike, /
    ) -> HydrostaticEOSResult:
        salinity_ = jnp.asarray(salinity)
        temperature_ = jnp.asarray(temperature, dtype=salinity_.dtype)
        pressure = jnp.asarray(pressure_dbar, dtype=salinity_.dtype)
        if salinity_.shape != temperature_.shape or pressure.shape != salinity_.shape:
            raise ValueError("EOS SA, CT, and pressure arrays must share one shape.")
        s = salinity_ - self.reference_salinity
        t = temperature_ - self.reference_temperature
        p = pressure
        a_t, a_s, a_tt, a_ts, a_pp, a_tp, a_sp = self.coefficients
        relative = (
            a_t * t
            + a_s * s
            + a_tt * t**2
            + a_ts * t * s
            + a_pp * p**2
            + a_tp * t * p
            + a_sp * s * p
        )
        density = self.reference_density * (1.0 + relative)
        rho_t = self.reference_density * (a_t + 2.0 * a_tt * t + a_ts * s + a_tp * p)
        rho_s = self.reference_density * (a_s + a_ts * t + a_sp * p)
        rho_p_dbar = self.reference_density * (2.0 * a_pp * p + a_tp * t + a_sp * s)
        alpha = -rho_t / density
        beta = rho_s / density
        rho_p_pa = rho_p_dbar / 1.0e4
        finite = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(alpha))
            & jnp.all(jnp.isfinite(beta))
            & jnp.all(jnp.isfinite(rho_p_pa))
        )
        valid = (
            jnp.all((salinity_ >= 0.0) & (salinity_ <= 50.0))
            & jnp.all((temperature_ >= -5.0) & (temperature_ <= 45.0))
            & jnp.all((pressure >= 0.0) & (pressure <= 12_000.0))
        )
        return HydrostaticEOSResult(
            density=density,
            alpha=alpha,
            beta=beta,
            density_pressure_derivative=rho_p_pa,
            valid=valid,
            finite=finite,
            successful=finite & valid,
            eos_id=self.eos_id,
        )


class FreshwaterVolumeFluxPlan(StrictModule, NonTrainableState):
    """Real free-surface volume source with incoming tracer composition."""

    rate: Array
    absolute_salinity: float = eqx.field(static=True)
    conservative_temperature: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate: ArrayLike,
        /,
        *,
        absolute_salinity: float = 0.0,
        conservative_temperature: float = 10.0,
    ):
        rate_ = jnp.asarray(rate)
        if bool(jnp.any(~jnp.isfinite(rate_))):
            raise ValueError("Freshwater volume flux must be finite.")
        salinity = float(absolute_salinity)
        temperature = float(conservative_temperature)
        if salinity < 0.0 or not np.isfinite(salinity) or not np.isfinite(temperature):
            raise ValueError("Freshwater composition is invalid.")
        self.rate = rate_
        self.absolute_salinity = salinity
        self.conservative_temperature = temperature
        self.plan_id = canonical_fingerprint(
            {
                "kind": "real-freshwater-volume-flux",
                "salinity": salinity,
                "temperature": temperature,
            }
        )

    def evaluate(
        self, time: ArrayLike, shape: tuple[int, int], args: Any = None, /
    ) -> Array:
        del time, args
        value = jnp.asarray(self.rate)
        if value.shape == ():
            value = jnp.broadcast_to(value, shape)
        if value.shape != shape:
            raise ValueError("Freshwater volume flux shape is invalid.")
        return value


class HydrostaticOpenBoundary(StrictModule, NonTrainableState):
    """One fully-wet barotropic/baroclinic boundary segment."""

    axis: int = eqx.field(static=True)
    side: Literal["lower", "upper"] = eqx.field(static=True)
    kind: BoundaryKind = eqx.field(static=True)
    target_eta: float = eqx.field(static=True)
    target_transport: float = eqx.field(static=True)
    absolute_salinity: float = eqx.field(static=True)
    conservative_temperature: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        side: Literal["lower", "upper"],
        kind: BoundaryKind,
        /,
        *,
        target_eta: float = 0.0,
        target_transport: float = 0.0,
        absolute_salinity: float = 35.0,
        conservative_temperature: float = 10.0,
    ):
        axis_ = int(axis)
        if axis_ not in (0, 1) or side not in ("lower", "upper"):
            raise ValueError("Hydrostatic boundaries require horizontal axis and side.")
        if kind not in (
            "closed",
            "prescribed-elevation",
            "prescribed-transport",
            "flather",
            "radiation",
        ):
            raise ValueError("Unknown hydrostatic open-boundary kind.")
        values = tuple(
            float(v)
            for v in (
                target_eta,
                target_transport,
                absolute_salinity,
                conservative_temperature,
            )
        )
        if any(not np.isfinite(v) for v in values) or values[2] < 0.0:
            raise ValueError("Hydrostatic boundary target values are invalid.")
        self.axis = axis_
        self.side = side
        self.kind = kind
        self.target_eta = values[0]
        self.target_transport = values[1]
        self.absolute_salinity = values[2]
        self.conservative_temperature = values[3]
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "hydrostatic-open-boundary",
                "axis": axis_,
                "side": side,
                "mode": kind,
                "targets": list(values),
            }
        )


class HydrostaticMixingPlan(StrictModule, NonTrainableState):
    """Prescribed, Ri, KPP-like, TKE, or Redi/GM hydrostatic mixing."""

    kind: MixingKind = eqx.field(static=True)
    background_viscosity: float = eqx.field(static=True)
    background_diffusivity: float = eqx.field(static=True)
    maximum_coefficient: float = eqx.field(static=True)
    critical_ri: float = eqx.field(static=True)
    ri_width: float = eqx.field(static=True)
    redi_coefficient: float = eqx.field(static=True)
    gm_coefficient: float = eqx.field(static=True)
    tke_coefficient: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MixingKind = "prescribed",
        /,
        *,
        background_viscosity: float = 1.0e-4,
        background_diffusivity: float = 1.0e-5,
        maximum_coefficient: float = 1.0,
        critical_ri: float = 0.25,
        ri_width: float = 0.1,
        redi_coefficient: float = 0.0,
        gm_coefficient: float = 0.0,
        tke_coefficient: float = 0.1,
    ):
        if kind not in ("prescribed", "ri", "kpp", "tke", "redi-gm"):
            raise ValueError("Unknown hydrostatic mixing kind.")
        values = tuple(
            float(v)
            for v in (
                background_viscosity,
                background_diffusivity,
                maximum_coefficient,
                critical_ri,
                ri_width,
                redi_coefficient,
                gm_coefficient,
                tke_coefficient,
            )
        )
        if any(not np.isfinite(v) or v < 0.0 for v in values) or values[4] == 0.0:
            raise ValueError(
                "Hydrostatic mixing parameters must be finite and nonnegative."
            )
        self.kind = kind
        (
            self.background_viscosity,
            self.background_diffusivity,
            self.maximum_coefficient,
            self.critical_ri,
            self.ri_width,
            self.redi_coefficient,
            self.gm_coefficient,
            self.tke_coefficient,
        ) = values
        self.plan_id = canonical_fingerprint(
            {"kind": "hydrostatic-mixing", "mode": kind, "values": list(values)}
        )


class HydrostaticOceanState(StrictModule):
    """Authoritative hydrostatic eta, integrated transports, and inventories."""

    eta: Array
    transports: tuple[Array, Array]
    tracer_inventory: dict[str, Array]
    tke_inventory: Array


class HydrostaticBoundaryTraces(StrictModule):
    """Oriented neighbor-cell traces supplied at one multiblock stage."""

    surface: Any
    hydrostatic_pressure: Any
    density: Any
    velocity: tuple[Any, Any]
    tracers: dict[str, Any]
    tke: Any


class HydrostaticOceanView(StrictModule):
    eta: Array
    velocity: tuple[Array, Array]
    tracers: dict[str, Array]
    density: Array
    hydrostatic_pressure: Array
    vertical_flux: Array
    wet_column: Array
    eos_valid: Array
    eos_finite: Array
    eos_successful: Array
    view_id: str = eqx.field(static=True)


class HydrostaticStageResult(StrictModule):
    state: HydrostaticOceanState
    epoch: HydrostaticMetricEpoch
    freshwater_rate: Array
    volume_residual: Array
    tracer_residual: dict[str, Array]
    free_surface_residual: Array
    vertical_mixing_residual: Array
    finite: Array
    successful: Array


class HydrostaticPrimitiveEquationPlan(StrictModule, NonTrainableState):
    """Hydrostatic primitive equations over prepared tensor-z/spherical metrics."""

    geometry: PreparedHydrostaticGrid
    eos: Any = eqx.field(static=True)
    mixing: HydrostaticMixingPlan
    freshwater: FreshwaterVolumeFluxPlan
    boundaries: tuple[HydrostaticOpenBoundary, ...]
    gravity: float = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    coriolis_f0: float = eqx.field(static=True)
    coriolis_beta: float = eqx.field(static=True)
    external_mode: ExternalMode = eqx.field(static=True)
    wetting_and_drying: bool = eqx.field(static=True)
    wet_depth: float = eqx.field(static=True)
    subcycle_policy: ExternalModeSubcyclePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: PreparedHydrostaticGrid,
        /,
        *,
        eos: Any | None = None,
        mixing: HydrostaticMixingPlan | None = None,
        freshwater: FreshwaterVolumeFluxPlan | None = None,
        boundaries: Sequence[HydrostaticOpenBoundary] = (),
        gravity: float = 9.81,
        reference_density: float = 1027.0,
        coriolis_f0: float = 0.0,
        coriolis_beta: float = 0.0,
        external_mode: ExternalMode = "implicit",
        wetting_and_drying: bool = False,
        wet_depth: float = 1.0e-6,
        subcycle_policy: ExternalModeSubcyclePolicy | None = None,
    ):
        if not isinstance(geometry, PreparedHydrostaticGrid):
            raise TypeError("geometry must be PreparedHydrostaticGrid.")
        from ._teos10 import TEOS10GSW75EOS

        eos_ = LinearHydrostaticEOS() if eos is None else eos
        if not isinstance(
            eos_,
            (LinearHydrostaticEOS, NonlinearSeawaterPolynomialEOS, TEOS10GSW75EOS),
        ):
            raise TypeError("Unsupported hydrostatic equation of state.")
        mixing_ = HydrostaticMixingPlan() if mixing is None else mixing
        freshwater_ = FreshwaterVolumeFluxPlan(0.0) if freshwater is None else freshwater
        boundary_tuple = tuple(boundaries)
        if any(
            not isinstance(boundary, HydrostaticOpenBoundary)
            for boundary in boundary_tuple
        ):
            raise TypeError("Hydrostatic boundaries must be HydrostaticOpenBoundary.")
        if external_mode not in ("implicit", "split-explicit"):
            raise ValueError("Unknown hydrostatic external mode.")
        values = tuple(
            float(v)
            for v in (gravity, reference_density, coriolis_f0, coriolis_beta, wet_depth)
        )
        if (
            any(not np.isfinite(v) for v in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
        ):
            raise ValueError("Hydrostatic physical constants are invalid.")
        subcycle = (
            ExternalModeSubcyclePolicy.fixed(20)
            if subcycle_policy is None
            else subcycle_policy
        )
        if not isinstance(subcycle, ExternalModeSubcyclePolicy):
            raise TypeError(
                "subcycle_policy must be an ExternalModeSubcyclePolicy or None."
            )
        self.geometry = geometry
        self.eos = eos_
        self.mixing = mixing_
        self.freshwater = freshwater_
        self.boundaries = boundary_tuple
        self.gravity = values[0]
        self.reference_density = values[1]
        self.coriolis_f0 = values[2]
        self.coriolis_beta = values[3]
        self.external_mode = external_mode
        self.wetting_and_drying = bool(wetting_and_drying)
        self.wet_depth = values[4]
        self.subcycle_policy = subcycle
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydrostatic-primitive-equation-plan",
                "geometry": geometry.geometry_id,
                "eos": eos_.eos_id,
                "mixing": mixing_.plan_id,
                "freshwater": freshwater_.plan_id,
                "boundaries": [b.boundary_id for b in boundary_tuple],
                "gravity": values[0],
                "reference_density": values[1],
                "f0": values[2],
                "beta": values[3],
                "external_mode": external_mode,
                "wetdry": bool(wetting_and_drying),
                "wet_depth": values[4],
                "subcycle_policy": subcycle.policy_id,
            }
        )

    def prepare(self) -> "PreparedHydrostaticOcean":
        free_surface = LinearImplicitFreeSurfacePlan(self.geometry, gravity=self.gravity)
        return PreparedHydrostaticOcean(self, free_surface)


class PreparedHydrostaticOcean(StrictModule):
    plan: HydrostaticPrimitiveEquationPlan
    free_surface: LinearImplicitFreeSurfacePlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HydrostaticPrimitiveEquationPlan,
        free_surface: LinearImplicitFreeSurfacePlan,
        /,
    ):
        self.plan = plan
        self.free_surface = free_surface
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hydrostatic-ocean",
                "plan": plan.plan_id,
                "free_surface": free_surface.plan_id,
            }
        )

    @property
    def geometry(self) -> PreparedHydrostaticGrid:
        return self.plan.geometry

    def initialize_state(
        self,
        eta: ArrayLike,
        /,
        *,
        transports: tuple[ArrayLike, ArrayLike] | None = None,
        tracers: Mapping[str, ArrayLike] | None = None,
    ) -> HydrostaticOceanState:
        eta_ = jnp.asarray(eta, dtype=self.geometry.cell_area.dtype)
        epoch = self.geometry.metric_epoch(eta_)
        if not bool(epoch.valid):
            raise ValueError("Initial hydrostatic geometry is invalid.")
        if transports is None:
            transport_ = (
                jnp.zeros(self.geometry.x_face_shape, dtype=eta_.dtype),
                jnp.zeros(self.geometry.y_face_shape, dtype=eta_.dtype),
            )
        else:
            transport_ = (
                jnp.asarray(transports[0], dtype=eta_.dtype),
                jnp.asarray(transports[1], dtype=eta_.dtype),
            )
        if (
            transport_[0].shape != self.geometry.x_face_shape
            or transport_[1].shape != self.geometry.y_face_shape
        ):
            raise ValueError("Initial hydrostatic transport shapes are invalid.")
        provided = {} if tracers is None else dict(tracers)
        concentrations = {
            "absolute_salinity": jnp.asarray(
                provided.get("absolute_salinity", 35.0), dtype=eta_.dtype
            ),
            "conservative_temperature": jnp.asarray(
                provided.get("conservative_temperature", 10.0), dtype=eta_.dtype
            ),
        }
        for name, value in tuple(provided.items()):
            if name not in concentrations:
                concentrations[name] = jnp.asarray(value, dtype=eta_.dtype)
        inventory = {}
        for name, value in concentrations.items():
            if value.shape == ():
                value = jnp.broadcast_to(value, self.geometry.cell_shape)
            if value.shape != self.geometry.cell_shape:
                raise ValueError(f"Tracer {name!r} has an invalid shape.")
            inventory[name] = epoch.cell_volume * value
        tke = jnp.zeros(self.geometry.cell_shape, dtype=eta_.dtype)
        return HydrostaticOceanState(eta_, transport_, inventory, tke)

    def view(self, state: HydrostaticOceanState, /) -> HydrostaticOceanView:
        epoch = self.geometry.metric_epoch(state.eta)
        velocity = (
            _safe_divide(state.transports[0], epoch.x_face_area),
            _safe_divide(state.transports[1], epoch.y_face_area),
        )
        tracers = {
            name: _safe_divide(inventory, epoch.cell_volume)
            for name, inventory in state.tracer_inventory.items()
        }
        pressure_dbar = self._sea_pressure(epoch)
        eos = self.plan.eos.evaluate(
            jnp.where(
                epoch.active_cell,
                tracers["absolute_salinity"],
                jnp.asarray(35.0, dtype=pressure_dbar.dtype),
            ),
            jnp.where(
                epoch.active_cell,
                tracers["conservative_temperature"],
                jnp.asarray(10.0, dtype=pressure_dbar.dtype),
            ),
            jnp.where(epoch.active_cell, pressure_dbar, 0.0),
        )
        hydrostatic = self._hydrostatic_pressure(eos.density, epoch)
        vertical_flux = self.geometry.diagnose_vertical_flux(state.transports)
        return HydrostaticOceanView(
            eta=state.eta,
            velocity=velocity,
            tracers=tracers,
            density=eos.density,
            hydrostatic_pressure=hydrostatic,
            vertical_flux=vertical_flux,
            eos_valid=eos.valid,
            eos_finite=eos.finite,
            eos_successful=eos.successful,
            wet_column=epoch.wet_column,
            view_id=self.prepared_id,
        )

    def _sea_pressure(self, epoch: HydrostaticMetricEpoch, /) -> Array:
        overlying = jnp.cumsum(epoch.layer_thickness[..., ::-1], axis=-1)[..., ::-1]
        centers = overlying - 0.5 * epoch.layer_thickness
        return self.plan.reference_density * self.plan.gravity * centers / 1.0e4

    def _hydrostatic_pressure(
        self, density: Array, epoch: HydrostaticMetricEpoch, /
    ) -> Array:
        anomaly = (density - self.plan.reference_density) / self.plan.reference_density
        increment = self.plan.gravity * anomaly * epoch.layer_thickness
        return jnp.cumsum(increment[..., ::-1], axis=-1)[..., ::-1]

    def _coriolis_force(
        self, state: HydrostaticOceanState, epoch: HydrostaticMetricEpoch, /
    ) -> tuple[Array, Array]:
        x, y = state.transports
        if self.geometry.horizontal_coordinate == "latitude-longitude":
            f_cell = self.geometry.coriolis
        else:
            y_coordinate = self.geometry.latitude
            f_cell = self.plan.coriolis_f0 + self.plan.coriolis_beta * y_coordinate
        normal_velocity = (
            _cell_from_faces(
                _safe_divide(x, epoch.x_face_area),
                0,
                self.geometry.periodic[0],
            ),
            _cell_from_faces(
                _safe_divide(y, epoch.y_face_area),
                1,
                self.geometry.periodic[1],
            ),
        )
        x_acceleration, y_acceleration = self.geometry.rotate_normal_velocity(
            normal_velocity, f_cell
        )
        return (
            epoch.x_face_area
            * _faces_from_cell(x_acceleration, 0, self.geometry.periodic[0]),
            epoch.y_face_area
            * _faces_from_cell(y_acceleration, 1, self.geometry.periodic[1]),
        )

    def _horizontal_tracer_fluxes(
        self,
        name: str,
        concentration: Array,
        transports: tuple[Array, Array],
        /,
        *,
        boundary_values=None,
    ) -> tuple[Array, Array]:
        face_values = [
            _face_upwind(concentration, transports[0], 0, self.geometry.periodic[0]),
            _face_upwind(concentration, transports[1], 1, self.geometry.periodic[1]),
        ]
        boundaries = (
            ((None, None), (None, None)) if boundary_values is None else boundary_values
        )
        for axis in range(2):
            if self.geometry.periodic[axis]:
                continue
            for side_index, exterior in enumerate(boundaries[axis]):
                if exterior is None:
                    continue
                index = 0 if side_index == 0 else -1
                location = [slice(None)] * face_values[axis].ndim
                location[axis] = index
                location_ = tuple(location)
                transport = transports[axis][location_]
                inflow = transport > 0.0 if side_index == 0 else transport < 0.0
                face_values[axis] = (
                    face_values[axis]
                    .at[location_]
                    .set(jnp.where(inflow, exterior, face_values[axis][location_]))
                )
        fluxes = [
            transports[0] * face_values[0],
            transports[1] * face_values[1],
        ]
        for boundary in self.plan.boundaries:
            axis = boundary.axis
            index = 0 if boundary.side == "lower" else -1
            location = [slice(None)] * fluxes[axis].ndim
            location[axis] = index
            transport = transports[axis][tuple(location)]
            inflow = transport > 0.0 if boundary.side == "lower" else transport < 0.0
            if name == "absolute_salinity":
                exterior = boundary.absolute_salinity
            elif name == "conservative_temperature":
                exterior = boundary.conservative_temperature
            else:
                exterior = 0.0
            selected = jnp.where(
                inflow,
                jnp.asarray(exterior, dtype=transport.dtype),
                face_values[axis][tuple(location)],
            )
            fluxes[axis] = fluxes[axis].at[tuple(location)].set(transport * selected)
        return fluxes[0], fluxes[1]

    def _vertical_tracer_flux(
        self, concentration: Array, vertical_flux: Array, /
    ) -> Array:
        flux = vertical_flux * _vertical_upwind(concentration, vertical_flux)
        return flux.at[..., 0].set(0.0).at[..., -1].set(0.0)

    def _redi_gm_fluxes(
        self,
        concentration: Array,
        state: HydrostaticOceanState,
        epoch: HydrostaticMetricEpoch,
        /,
        *,
        concentration_boundary=None,
        density_boundary=None,
    ) -> tuple[tuple[Array, Array], Array]:
        gradient_x, gradient_y = self.geometry.layer_gradient(
            concentration, boundary_values=concentration_boundary
        )
        redi = self.plan.mixing.redi_coefficient
        gm = self.plan.mixing.gm_coefficient
        redi_x = -redi * epoch.x_face_area * gradient_x
        redi_y = -redi * epoch.y_face_area * gradient_y
        if gm == 0.0:
            return (
                (redi_x, redi_y),
                jnp.zeros(
                    self.geometry.horizontal_shape + (self.geometry.cell_shape[-1] + 1,),
                    dtype=concentration.dtype,
                ),
            )
        view = self.view(state)
        density_x, density_y = self.geometry.layer_gradient(
            view.density, boundary_values=density_boundary
        )
        density_vertical = jnp.gradient(view.density, axis=-1)
        concentration_vertical = jnp.gradient(concentration, axis=-1)
        x_vertical = self.geometry.face_average(density_vertical, 0)
        y_vertical = self.geometry.face_average(density_vertical, 1)
        slope_x = -_safe_divide(density_x, x_vertical)
        slope_y = -_safe_divide(density_y, y_vertical)
        gm_x = (
            gm
            * epoch.x_face_area
            * slope_x
            * self.geometry.face_average(concentration_vertical, 0)
        )
        gm_y = (
            gm
            * epoch.y_face_area
            * slope_y
            * self.geometry.face_average(concentration_vertical, 1)
        )
        cell_gradient_x = _cell_from_faces(gradient_x, 0, self.geometry.periodic[0])
        cell_gradient_y = _cell_from_faces(gradient_y, 1, self.geometry.periodic[1])
        cell_slope_x = _cell_from_faces(slope_x, 0, self.geometry.periodic[0])
        cell_slope_y = _cell_from_faces(slope_y, 1, self.geometry.periodic[1])
        vertical_cell_flux = (
            -gm
            * self.geometry.cell_area[..., None]
            * self.geometry.normal_velocity_inner_product(
                (cell_slope_x, cell_slope_y),
                (cell_gradient_x, cell_gradient_y),
            )
        )
        vertical_flux = jnp.zeros(
            self.geometry.horizontal_shape + (self.geometry.cell_shape[-1] + 1,),
            dtype=concentration.dtype,
        )
        vertical_flux = vertical_flux.at[..., 1:-1].set(
            0.5 * (vertical_cell_flux[..., :-1] + vertical_cell_flux[..., 1:])
        )
        return (redi_x + gm_x, redi_y + gm_y), vertical_flux

    def _tracer_tendency(
        self,
        state: HydrostaticOceanState,
        epoch: HydrostaticMetricEpoch,
        vertical_flux: Array,
        freshwater: Array,
        /,
        *,
        boundary_traces: HydrostaticBoundaryTraces | None = None,
    ) -> dict[str, Array]:
        tendency = {}
        top = [slice(None)] * 3
        top[-1] = -1
        for name, inventory in state.tracer_inventory.items():
            concentration = _safe_divide(inventory, epoch.cell_volume)
            concentration_boundary = (
                None if boundary_traces is None else boundary_traces.tracers[name]
            )
            horizontal = self._horizontal_tracer_fluxes(
                name,
                concentration,
                state.transports,
                boundary_values=concentration_boundary,
            )
            gm_vertical = jnp.zeros_like(vertical_flux)
            if self.plan.mixing.kind == "redi-gm":
                redi_gm_horizontal, gm_vertical = self._redi_gm_fluxes(
                    concentration,
                    state,
                    epoch,
                    concentration_boundary=concentration_boundary,
                    density_boundary=(
                        None if boundary_traces is None else boundary_traces.density
                    ),
                )
                horizontal = (
                    horizontal[0] + redi_gm_horizontal[0],
                    horizontal[1] + redi_gm_horizontal[1],
                )
            net_horizontal = self.geometry.net_cell_flux(horizontal)
            vertical = (
                self._vertical_tracer_flux(concentration, vertical_flux) + gm_vertical
            )
            if self.plan.mixing.kind == "kpp":
                column_volume = jnp.sum(epoch.cell_volume, axis=-1)
                column_mean = _safe_divide(jnp.sum(inventory, axis=-1), column_volume)
                top_concentration = concentration[..., -1]
                vertical_fraction = jnp.linspace(
                    0.0, 1.0, self.geometry.cell_shape[-1] + 1
                )
                nonlocal_shape = 4.0 * vertical_fraction * (1.0 - vertical_fraction)
                nonlocal_flux = (
                    self.plan.mixing.maximum_coefficient
                    * self.geometry.cell_area[..., None]
                    * _safe_divide(
                        top_concentration - column_mean,
                        epoch.total_depth,
                    )[..., None]
                    * nonlocal_shape
                )
                vertical = vertical + nonlocal_flux
            net_vertical = vertical[..., 1:] - vertical[..., :-1]
            rate = -(net_horizontal + net_vertical)
            if name == "absolute_salinity":
                incoming = self.plan.freshwater.absolute_salinity
            elif name == "conservative_temperature":
                incoming = self.plan.freshwater.conservative_temperature
            else:
                incoming = 0.0
            rate = rate.at[tuple(top)].add(
                self.geometry.cell_area * freshwater * incoming
            )
            tendency[name] = rate
        return tendency

    def _mixing_coefficients(
        self,
        state: HydrostaticOceanState,
        epoch: HydrostaticMetricEpoch,
        /,
        *,
        boundary_traces: HydrostaticBoundaryTraces | None = None,
    ) -> tuple[Array, Array]:
        view = self.view(state)
        nz = self.geometry.cell_shape[-1]
        interface_shape = self.geometry.horizontal_shape + (nz + 1,)
        viscosity = jnp.full(
            interface_shape,
            self.plan.mixing.background_viscosity,
            dtype=epoch.cell_volume.dtype,
        )
        diffusivity = jnp.full(
            interface_shape,
            self.plan.mixing.background_diffusivity,
            dtype=epoch.cell_volume.dtype,
        )
        if self.plan.mixing.kind in ("ri", "kpp"):
            density = view.density
            dz = 0.5 * (epoch.layer_thickness[..., 1:] + epoch.layer_thickness[..., :-1])
            density_gradient = _safe_divide(density[..., 1:] - density[..., :-1], dz)
            u_cell = _cell_from_faces(view.velocity[0], 0, self.geometry.periodic[0])
            v_cell = _cell_from_faces(view.velocity[1], 1, self.geometry.periodic[1])
            vertical_shear = (
                _safe_divide(u_cell[..., 1:] - u_cell[..., :-1], dz),
                _safe_divide(v_cell[..., 1:] - v_cell[..., :-1], dz),
            )
            shear = self.geometry.normal_velocity_inner_product(
                vertical_shear, vertical_shear
            )
            n2 = -self.plan.gravity / self.plan.reference_density * density_gradient
            ri = _safe_divide(n2, shear + 1.0e-14)
            taper = 0.5 * (
                1.0
                - jnp.tanh(
                    (ri - self.plan.mixing.critical_ri) / self.plan.mixing.ri_width
                )
            )
            coefficient = (
                self.plan.mixing.background_diffusivity
                + (
                    self.plan.mixing.maximum_coefficient
                    - self.plan.mixing.background_diffusivity
                )
                * taper
            )
            diffusivity = diffusivity.at[..., 1:-1].set(coefficient)
            viscosity = viscosity.at[..., 1:-1].set(
                jnp.maximum(coefficient, self.plan.mixing.background_viscosity)
            )
            if self.plan.mixing.kind == "kpp":
                depth_fraction = jnp.linspace(0.0, 1.0, nz + 1)
                kpp_shape = 4.0 * depth_fraction * (1.0 - depth_fraction)
                diffusivity = jnp.maximum(
                    diffusivity,
                    self.plan.mixing.maximum_coefficient * kpp_shape,
                )
        elif self.plan.mixing.kind == "tke":
            tke = _safe_divide(state.tke_inventory, epoch.cell_volume)
            tke_interface = _faces_from_cell(tke, 2, False)
            coefficient = self.plan.mixing.tke_coefficient * jnp.sqrt(
                jnp.maximum(tke_interface, 0.0)
            )
            diffusivity = jnp.maximum(diffusivity, coefficient)
            viscosity = jnp.maximum(viscosity, coefficient)
        elif self.plan.mixing.kind == "redi-gm":
            density_x, density_y = self.geometry.layer_gradient(
                view.density,
                boundary_values=(
                    None if boundary_traces is None else boundary_traces.density
                ),
            )
            density_vertical = jnp.gradient(view.density, axis=-1)
            slope_x = -_safe_divide(
                density_x,
                self.geometry.face_average(density_vertical, 0),
            )
            slope_y = -_safe_divide(
                density_y,
                self.geometry.face_average(density_vertical, 1),
            )
            slope_square = _cell_from_faces(
                slope_x**2, 0, self.geometry.periodic[0]
            ) + _cell_from_faces(slope_y**2, 1, self.geometry.periodic[1])
            interface_slope = _faces_from_cell(slope_square, 2, False)
            diffusivity = jnp.maximum(
                diffusivity,
                self.plan.mixing.background_diffusivity
                + self.plan.mixing.redi_coefficient * interface_slope,
            )
        return viscosity, diffusivity

    def _implicit_vertical_scalar(
        self,
        inventory: Array,
        epoch: HydrostaticMetricEpoch,
        coefficient: Array,
        dt: Array,
        /,
    ) -> tuple[Array, Array]:
        volume = epoch.cell_volume
        center_distance = 0.5 * (
            epoch.layer_thickness[..., 1:] + epoch.layer_thickness[..., :-1]
        )
        conductance = _safe_divide(
            self.geometry.cell_area[..., None] * coefficient[..., 1:-1],
            center_distance,
        )
        lower = jnp.zeros_like(volume)
        upper = jnp.zeros_like(volume)
        lower = lower.at[..., 1:].set(-dt * conductance)
        upper = upper.at[..., :-1].set(-dt * conductance)
        diagonal = volume - lower - upper
        inactive = ~epoch.active_cell
        lower = jnp.where(inactive, 0.0, lower)
        upper = jnp.where(inactive, 0.0, upper)
        diagonal = jnp.where(inactive, 1.0, diagonal)
        rhs = jnp.where(inactive, 0.0, inventory)
        result = solve_tridiagonal_lines(lower, diagonal, upper, rhs, -1)
        concentration = result.value
        return volume * concentration, result.residual_norm

    def _implicit_vertical_transport(
        self,
        transport: Array,
        face_area: Array,
        coefficient: Array,
        horizontal_axis: int,
        dt: Array,
        /,
    ) -> tuple[Array, Array]:
        face_coefficient = self.geometry.face_average(
            coefficient[..., 1:-1], horizontal_axis
        )
        edge_length = (
            self.geometry.x_edge_length[..., None]
            if horizontal_axis == 0
            else self.geometry.y_edge_length[..., None]
        )
        face_thickness = _safe_divide(face_area, edge_length)
        center_distance = 0.5 * (face_thickness[..., 1:] + face_thickness[..., :-1])
        conductance = _safe_divide(
            edge_length * face_coefficient,
            center_distance,
        )
        lower = jnp.zeros_like(face_area).at[..., 1:].set(-dt * conductance)
        upper = jnp.zeros_like(face_area).at[..., :-1].set(-dt * conductance)
        diagonal = face_area - lower - upper
        inactive = face_area <= 0.0
        lower = jnp.where(inactive, 0.0, lower)
        upper = jnp.where(inactive, 0.0, upper)
        diagonal = jnp.where(inactive, 1.0, diagonal)
        rhs = jnp.where(inactive, 0.0, transport)
        result = solve_tridiagonal_lines(lower, diagonal, upper, rhs, -1)
        return face_area * result.value, result.residual_norm

    def apply_vertical_mixing(
        self,
        state: HydrostaticOceanState,
        epoch: HydrostaticMetricEpoch,
        step_size: ArrayLike,
        /,
        *,
        boundary_traces: HydrostaticBoundaryTraces | None = None,
    ) -> tuple[HydrostaticOceanState, Array]:
        dt = jnp.asarray(step_size, dtype=epoch.cell_volume.dtype)
        viscosity, diffusivity = self._mixing_coefficients(
            state, epoch, boundary_traces=boundary_traces
        )
        inventory = {}
        residual = jnp.asarray(0.0, dtype=dt.dtype)
        for name, value in state.tracer_inventory.items():
            mixed, norm = self._implicit_vertical_scalar(value, epoch, diffusivity, dt)
            inventory[name] = mixed
            residual = jnp.maximum(residual, norm)
        tke, norm = self._implicit_vertical_scalar(
            state.tke_inventory, epoch, diffusivity, dt
        )
        residual = jnp.maximum(residual, norm)
        x_transport, x_norm = self._implicit_vertical_transport(
            state.transports[0], epoch.x_face_area, viscosity, 0, dt
        )
        y_transport, y_norm = self._implicit_vertical_transport(
            state.transports[1], epoch.y_face_area, viscosity, 1, dt
        )
        residual = jnp.maximum(residual, jnp.maximum(x_norm, y_norm))
        return (
            HydrostaticOceanState(
                state.eta,
                (x_transport, y_transport),
                inventory,
                tke,
            ),
            residual,
        )


__all__ = [
    "FreshwaterVolumeFluxPlan",
    "HydrostaticEOSResult",
    "HydrostaticMixingPlan",
    "HydrostaticOceanState",
    "HydrostaticOceanView",
    "HydrostaticOpenBoundary",
    "HydrostaticPrimitiveEquationPlan",
    "HydrostaticStageResult",
    "LinearHydrostaticEOS",
    "NonlinearSeawaterPolynomialEOS",
    "PreparedHydrostaticOcean",
]

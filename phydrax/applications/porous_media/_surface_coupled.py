#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...nonlinear import (
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._richards import _finish_root, RichardsPlan
from ._state import PorousState
from ._surface_exchange import OrthogonalDiffusiveWaveSurfacePlan, SurfaceWaterState


@jax.custom_jvp
def _complementarity(a, b):
    return jnp.sqrt(a * a + b * b) - a - b


@_complementarity.defjvp
def _complementarity_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    norm = jnp.sqrt(a * a + b * b)
    denominator = jnp.where(norm > 0, norm, 1.0)
    derivative = jnp.where(norm > 0, (a * da + b * db) / denominator, 0.0) - da - db
    return _complementarity(a, b), derivative


class SurfacePorousResult(StrictModule):
    porous: PorousState
    surface: SurfaceWaterState
    candidate_porous: PorousState
    candidate_surface: SurfaceWaterState
    dry_pressure_head: Array
    exchange_mass_rate: Array
    lateral_volume_rate: Array
    mass_residual: Array
    complementarity_residual: Array
    root: NonlinearResult
    successful: Array
    derivative_available: Array


class SurfaceRichardsPlan(StrictModule):
    """Shared-pressure Richards/runoff root with atmospheric complementarity.

    At wet faces p = rho*g*h; at dry faces h=0 and p<=0 (gauge pressure).
    Surface depth and dry suction head are complementary nonnegative unknowns.
    The exact volume-boundary mass rate supplies surface storage. Fluid density
    is constant and thermal physics is isothermal; no energy feedback is hidden
    in this water-only model. Input surface energy must match that temperature.
    """

    water: RichardsPlan
    surface: OrthogonalDiffusiveWaveSurfacePlan
    pressure_per_head: Array
    termination: NonlinearTermination
    boundary_faces: Array
    mass_scale: float = eqx.field(static=True)
    head_scale: float = eqx.field(static=True)

    def __init__(
        self,
        water: RichardsPlan,
        surface: OrthogonalDiffusiveWaveSurfacePlan,
        *,
        gravity_m_s2: float = 9.80665,
        mass_scale_kg_s: float = 1.0,
        head_scale_m: float = 1.0,
        termination: NonlinearTermination | None = None,
    ):
        if not isinstance(water, RichardsPlan) or not isinstance(
            surface, OrthogonalDiffusiveWaveSurfacePlan
        ):
            raise TypeError("Surface coupling requires prepared water and surface plans.")
        surface.trace.require_geometry(water.discretization)
        if not all(
            np.isfinite(x) and x > 0
            for x in (gravity_m_s2, mass_scale_kg_s, head_scale_m)
        ):
            raise ValueError("Gravity and residual scales must be positive and finite.")
        self.water = water
        self.surface = surface
        self.pressure_per_head = surface.density * gravity_m_s2
        self.mass_scale = float(mass_scale_kg_s)
        self.head_scale = float(head_scale_m)
        self.termination = (
            NonlinearTermination(
                absolute_residual=1e-9, relative_residual=1e-9, maximum_steps=60
            )
            if termination is None
            else termination
        )
        self.boundary_faces = jnp.asarray(
            np.flatnonzero(np.asarray(water.discretization.neighbour_cells) < 0),
            dtype=jnp.int32,
        )

    def step(
        self,
        previous: PorousState,
        surface_state: SurfaceWaterState,
        dt_s: ArrayLike,
        *,
        rainfall_m_s: ArrayLike = 0.0,
        source_kg_s: ArrayLike = 0.0,
    ) -> SurfacePorousResult:
        dt = jnp.asarray(dt_s)
        if dt.ndim != 0:
            raise ValueError("Time step must be scalar.")
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0), "Time step must be positive."
        )
        trace = self.surface.trace
        nc = self.water.discretization.cell_volumes.size
        nf = self.water.discretization.face_measures.size
        ns = trace.parent_faces.size
        rain = self.surface._vector(rainfall_m_s, "Rainfall")
        rain = eqx.error_if(rain, jnp.any(rain < 0), "Rainfall must be nonnegative.")
        source = jnp.broadcast_to(jnp.asarray(source_kg_s), (nc,))
        source = eqx.error_if(
            source, jnp.any(~jnp.isfinite(source)), "Porous mass sources must be finite."
        )
        initial_head = surface_state.volume / self.surface.projected_areas
        initial_dry = jnp.maximum(
            -previous.face_pressure_Pa[trace.parent_faces] / self.pressure_per_head,
            0.0,
        )
        initial = jnp.concatenate(
            (
                previous.pressure_Pa,
                previous.face_pressure_Pa,
                initial_head,
                initial_dry,
            )
        )
        density = self.surface.density
        old_density = previous.water_mass_kg / previous.water_volume_m3
        initial = eqx.error_if(
            initial,
            jnp.any(~jnp.isfinite(initial))
            | jnp.any(initial_head < 0)
            | jnp.any(jnp.abs(old_density - density) > 1e-8 * density),
            "Surface Richards requires nonnegative depth and matching constant liquid density.",
        )
        temperature = previous.temperature_K[trace.parent_cells]
        expected_energy = (
            density
            * self.surface.heat_capacity
            * surface_state.volume
            * (temperature - self.surface.reference_temperature)
        )
        initial = eqx.error_if(
            initial,
            jnp.any(
                jnp.abs(surface_state.energy - expected_energy)
                > 1e-8 * jnp.maximum(jnp.abs(expected_energy), 1.0)
            ),
            "Water-only coupling requires isothermal surface energy consistent with the porous state.",
        )
        rain_rate = rain * self.surface.projected_areas
        unknown_scale = jnp.concatenate(
            (
                jnp.full(nc + nf, self.water.pressure_scale_Pa),
                jnp.full(2 * ns, self.head_scale),
            )
        )
        anchored_cells = jnp.zeros(nc, dtype=bool).at[trace.parent_cells].set(True)

        def residual(scaled_unknown, args):
            old, old_surface, time_step, rainfall_rate, mass_source = args
            unknown = scaled_unknown * unknown_scale
            pressure = unknown[:nc]
            face_pressure = unknown[nc : nc + nf]
            head = unknown[nc + nf : nc + nf + ns]
            dry = unknown[nc + nf + ns :]
            raw = self.water.residual(unknown[: nc + nf], old, time_step, mass_source)
            # All non-interface equations retain their original physical law;
            # the interface replaces the previous boundary law, not its flux.
            scaled = raw / self.water.residual_scales()
            interface = (
                face_pressure[trace.parent_faces] / self.pressure_per_head - head + dry
            ) / self.head_scale
            scaled = scaled.at[nc + trace.parent_faces].set(interface)
            flux = self.water.fluxes(
                pressure,
                face_pressure,
                old.temperature_K,
                old.face_temperature_K,
            )
            outward = flux.mass_face_rates[trace.parent_faces]
            volume = self.surface.projected_areas * head
            lateral = self.surface.lateral_rates(volume)
            balance = (
                density * (volume - old_surface.volume) / time_step
                + density * self.surface.divergence(lateral)
                - density * rainfall_rate
                - outward
            )
            return jnp.concatenate(
                (
                    scaled,
                    balance / self.mass_scale,
                    _complementarity(head, dry) / self.head_scale,
                )
            )

        def valid(scaled_unknown, residual_value, auxiliary, args):
            del residual_value, auxiliary, args
            unknown = scaled_unknown * unknown_scale
            pressure = unknown[:nc]
            head = unknown[nc + nf : nc + nf + ns]
            dry = unknown[nc + nf + ns :]
            return (
                self.water.admissible(pressure, previous.temperature_K)
                & self.water.well_posed(
                    pressure,
                    previous.temperature_K,
                    additional_anchored_cells=anchored_cells,
                )
                & jnp.all(jnp.isfinite(head) & (head >= 0))
                & jnp.all(jnp.isfinite(dry) & (dry >= 0))
            )

        args = (previous, surface_state, dt, rain_rate, source)
        problem = NonlinearSystemProblem(
            residual,
            validity=valid,
            problem_id="surface-richards-complementarity",
        )
        root = implicit_root_result(
            problem,
            initial / unknown_scale,
            method=self.water.method,
            termination=self.termination,
            derivative_policy=self.water.derivative_policy,
            args=args,
        )
        successful = root.successful & valid(root.state, root.residual, None, args)
        root = _finish_root(root, successful)
        unknown = root.state * unknown_scale
        candidate = self.water.state_from_unknown(
            unknown[: nc + nf],
            time_s=previous.time_s + dt,
            temperature_K=previous.temperature_K,
            face_temperature_K=previous.face_temperature_K,
        )
        head = unknown[nc + nf : nc + nf + ns]
        dry = unknown[nc + nf + ns :]
        volume = self.surface.projected_areas * head
        flux = self.water.fluxes(
            candidate.pressure_Pa,
            candidate.face_pressure_Pa,
            candidate.temperature_K,
            candidate.face_temperature_K,
        )
        exchange = flux.mass_face_rates[trace.parent_faces]
        lateral = self.surface.lateral_rates(volume)
        mass_residual = (
            jnp.sum(candidate.water_mass_kg - previous.water_mass_kg)
            + density * jnp.sum(volume - surface_state.volume)
            + dt
            * (
                jnp.sum(flux.mass_face_rates[self.boundary_faces])
                - jnp.sum(exchange)
                - density * jnp.sum(rain_rate)
                - jnp.sum(source)
            )
        )
        candidate_density = candidate.water_mass_kg / candidate.water_volume_m3
        successful = (
            successful
            & jnp.all(head >= -1e-9)
            & jnp.all(dry >= -1e-9)
            & jnp.all(jnp.abs(candidate_density - density) <= 1e-8 * density)
        )
        root = _finish_root(root, successful)
        energy = (
            density
            * self.surface.heat_capacity
            * volume
            * (temperature - self.surface.reference_temperature)
        )
        candidate_surface = SurfaceWaterState(volume, energy)
        porous = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate,
            previous,
        )
        surface = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate_surface,
            surface_state,
        )
        derivative_available = successful & jnp.all(head + dry > 1e-8)
        return SurfacePorousResult(
            porous,
            surface,
            candidate,
            candidate_surface,
            dry,
            exchange,
            lateral,
            mass_residual,
            _complementarity(head, dry),
            root,
            successful,
            derivative_available,
        )


__all__ = ["SurfacePorousResult", "SurfaceRichardsPlan"]

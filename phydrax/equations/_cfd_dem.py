#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unresolved drag with bulk and point-particle resistance kept distinct.

Re = rho*d*w/mu is interstitial; Di Felice, Wen--Yu and Beetstra fits use
Rs = epsilon*Re (superficial flux). Schiller & Naumann, VDI 77:318 (1933);
Di Felice, IJMF 20:153--159 (1994), doi:10.1016/0301-9322(94)90011-6;
Ergun, Chem. Eng. Prog. 48:89--94 (1952); Wen & Yu, CEP Symp. Ser.
62:100--111 (1966); Gidaspow, Multiphase Flow and Fluidization (1994);
Huilin & Gidaspow, CES 58:3777--3792 (2003),
doi:10.1016/S0009-2509(03)00238-0; Beetstra et al., AIChE J. 53:489--501
(2007), doi:10.1002/aic.11065, Eqs. 1, 2 and 17.

Derivation: beta = 18*mu*(1-epsilon)*B/d**2. Di Felice's same-flux
definition gives B = Cd(Rs)*Rs/24 * epsilon**(1-chi); Dallavalle uses
Cd(Rs)*Rs = (0.63*sqrt(Rs)+4.8)**2. Wen--Yu gives
B = Cd(Rs)*Rs/24 * epsilon**-2.65. Ergun gives
B = (150*(1-epsilon)/epsilon + 1.75*Re)/18. Beetstra gives
B = epsilon*(f0+fRe). The Huilin--Gidaspow Ergun weight is
s = 0.5 + atan(150*1.75*((1-epsilon)-0.2))/pi, not a tunable sigmoid.
The original closed-access Di Felice and Huilin tables were inaccessible;
their coefficient transcription is supported by the secondary sources
explicitly recorded in the CFD--DEM guide, not claimed primary-verified.

No beta clipping or domain repair is performed. Admission is evidence, not
permission to extrapolate. Fractional-power coefficient limits at zero slip
are value-defined but do not certify derivatives there.
"""

from __future__ import annotations

import abc
from typing import assert_never, final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import (
    DEMRuntimeState,
    PreparedCompliantDEMDynamics,
    PreparedRigidSphereSet,
)
from ..discretization.splatting import (
    MeshSplatRoutes,
    PreparedMeshParticleGridSplat,
)
from ..typing import checked, parse


def _schiller_naumann_resistance(reynolds: Array, /) -> Array:
    """Cd Re / 24, with the analytic value limit and no 24 / Re at rest."""
    positive = reynolds > 0.0
    argument = jnp.where(positive, reynolds, 1.0)
    return jnp.where(positive, 1.0 + 0.15 * argument**0.687, 1.0)


def _wen_yu_resistance(porosity: Array, reynolds: Array, /) -> Array:
    superficial_reynolds = porosity * reynolds
    isolated = jnp.where(
        superficial_reynolds < 1000.0,
        _schiller_naumann_resistance(superficial_reynolds),
        0.44 * superficial_reynolds / 24.0,
    )
    return isolated * porosity**-2.65


def _di_felice_resistance(porosity: Array, reynolds: Array, /) -> Array:
    superficial_reynolds = porosity * reynolds
    positive = superficial_reynolds > 0.0
    argument = jnp.where(positive, superficial_reynolds, 1.0)
    exponent = jnp.where(
        positive,
        3.7 - 0.65 * jnp.exp(-0.5 * (1.5 - jnp.log10(argument)) ** 2),
        3.7,
    )
    cd_reynolds = jnp.where(positive, (0.63 * jnp.sqrt(argument) + 4.8) ** 2, 4.8**2)
    return cd_reynolds / 24.0 * porosity ** (1.0 - exponent)


def _beetstra_resistance(porosity: Array, reynolds: Array, /) -> Array:
    """Beetstra et al. (2007), Eq. 17, with their superficial Reynolds number."""
    solid_fraction = 1.0 - porosity
    superficial_reynolds = porosity * reynolds
    positive = superficial_reynolds > 0.0
    argument = jnp.where(positive, superficial_reynolds, 1.0)
    power = (5.0 - 4.0 * porosity) / 2.0
    creeping = 10.0 * solid_fraction / porosity**2 + porosity**2 * (
        1.0 + 1.5 * jnp.sqrt(solid_fraction)
    )
    # Multiplication by Re**power removes both negative powers in Eq. 17.
    inertial = (
        0.413
        / (24.0 * porosity**2)
        * (
            (1.0 / porosity + 3.0 * solid_fraction * porosity) * argument ** (1.0 + power)
            + 8.4 * argument ** (power + 0.657)
        )
        / (argument**power + 10.0 ** (3.0 * solid_fraction))
    )
    return porosity * (creeping + jnp.where(positive, inertial, 0.0))


class FluidParticleSample(StrictModule):
    velocity: Array
    density: Array
    dynamic_viscosity: Array
    pressure_gradient: Array
    porosity: Array


type InterphaseDragCorrelation = Literal[
    "stokes",
    "schiller_naumann",
    "di_felice",
    "ergun",
    "wen_yu",
    "gidaspow",
    "huilin_gidaspow",
    "beetstra",
]


@final
class InterphaseDragEvaluation(StrictModule):
    beta: Array
    particle_drag_coefficient: Array
    reynolds_number: Array
    branch_code: Array
    branch_margin: Array
    derivative_valid: Array
    admissible: Array
    successful: Array
    law_id: str = eqx.field(static=True)


class AbstractInterphaseDragLaw(StrictModule, NonTrainableState):
    correlation: eqx.AbstractVar[InterphaseDragCorrelation]
    law_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(
        self,
        porosity: ArrayLike,
        relative_speed: ArrayLike,
        particle_diameter: ArrayLike,
        dynamic_viscosity: ArrayLike,
        fluid_density: ArrayLike,
        particle_volume: ArrayLike,
        /,
    ) -> InterphaseDragEvaluation:
        raise NotImplementedError


@final
class InterphaseDragLawPlan(AbstractInterphaseDragLaw):
    correlation: InterphaseDragCorrelation = eqx.field(static=True)
    maximum_reynolds: float = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        correlation: InterphaseDragCorrelation,
        /,
        *,
        maximum_reynolds: float | None = None,
    ) -> None:
        selected = parse(correlation, InterphaseDragCorrelation, "correlation")
        match selected:
            case "stokes":
                published_maximum = 1.0
            case "schiller_naumann" | "beetstra":
                published_maximum = 1000.0
            case "di_felice" | "ergun" | "wen_yu" | "gidaspow" | "huilin_gidaspow":
                published_maximum = np.inf
            case _:
                assert_never(selected)
        maximum = (
            published_maximum if maximum_reynolds is None else float(maximum_reynolds)
        )
        if maximum_reynolds is not None and (
            not np.isfinite(maximum) or not 0.0 < maximum <= published_maximum
        ):
            raise ValueError("maximum_reynolds must be positive and tighten the domain.")
        self.correlation = selected
        self.maximum_reynolds = maximum
        self.law_id = canonical_fingerprint(
            {
                "kind": "interphase-drag-law",
                "correlation": selected,
                "maximum_reynolds": maximum if np.isfinite(maximum) else None,
                "reynolds_convention": "published-fit",
            }
        )

    def evaluate(
        self,
        porosity: ArrayLike,
        relative_speed: ArrayLike,
        particle_diameter: ArrayLike,
        dynamic_viscosity: ArrayLike,
        fluid_density: ArrayLike,
        particle_volume: ArrayLike,
        /,
    ) -> InterphaseDragEvaluation:
        epsilon, speed, diameter, viscosity, density, volume = jnp.broadcast_arrays(
            *(
                jnp.asarray(value)
                for value in (
                    porosity,
                    relative_speed,
                    particle_diameter,
                    dynamic_viscosity,
                    fluid_density,
                    particle_volume,
                )
            )
        )
        reynolds = density * diameter * speed / viscosity
        factor, code, margin, smooth, supported, fit_reynolds = _drag_resistance(
            self.correlation, epsilon, reynolds
        )
        beta = 18.0 * viscosity * (1.0 - epsilon) / diameter**2 * factor
        match self.correlation:
            case "stokes" | "schiller_naumann" | "di_felice":
                coefficient = 3.0 * jnp.pi * viscosity * diameter * factor
            case "ergun" | "wen_yu" | "gidaspow" | "huilin_gidaspow" | "beetstra":
                coefficient = volume * 18.0 * viscosity / diameter**2 * factor
            case _:
                assert_never(self.correlation)
        physical = (
            jnp.isfinite(epsilon)
            & (epsilon > 0.0)
            & (epsilon <= 1.0)
            & jnp.isfinite(speed)
            & (speed >= 0.0)
            & jnp.isfinite(diameter)
            & (diameter > 0.0)
            & jnp.isfinite(viscosity)
            & (viscosity > 0.0)
            & jnp.isfinite(density)
            & (density > 0.0)
            & jnp.isfinite(volume)
            & (volume > 0.0)
        )
        admissible = (
            physical
            & supported
            & jnp.isfinite(reynolds)
            & (fit_reynolds <= self.maximum_reynolds)
            & jnp.isfinite(beta)
            & jnp.isfinite(coefficient)
        )
        return InterphaseDragEvaluation(
            beta,
            coefficient,
            reynolds,
            code,
            margin,
            admissible & smooth,
            admissible,
            jnp.all(admissible),
            self.law_id,
        )


def _drag_resistance(
    correlation: InterphaseDragCorrelation, epsilon: Array, reynolds: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Dimensionless beta resistance, source branches and admission evidence."""
    code = jnp.zeros_like(reynolds, dtype=jnp.int32)
    margin = jnp.full_like(reynolds, jnp.inf)
    supported = jnp.ones_like(reynolds, dtype=jnp.bool_)
    smooth = reynolds > 0.0
    fit_reynolds = reynolds
    match correlation:
        case "stokes":
            factor = jnp.ones_like(reynolds)
            smooth = supported
        case "schiller_naumann":
            factor = _schiller_naumann_resistance(reynolds)
            margin = 1000.0 - reynolds
        case "di_felice":
            factor = _di_felice_resistance(epsilon, reynolds)
            fit_reynolds = epsilon * reynolds
        case "ergun":
            factor = (150.0 * (1.0 - epsilon) / epsilon + 1.75 * reynolds) / 18.0
            smooth = supported
        case "wen_yu":
            fit_reynolds = epsilon * reynolds
            factor = _wen_yu_resistance(epsilon, reynolds)
            code = jnp.asarray(fit_reynolds >= 1000.0, dtype=jnp.int32)
            margin = fit_reynolds - 1000.0
            smooth = smooth & (fit_reynolds != 1000.0)
            supported = epsilon > 0.8
        case "gidaspow":
            ergun = (150.0 * (1.0 - epsilon) / epsilon + 1.75 * reynolds) / 18.0
            wen_yu = _wen_yu_resistance(epsilon, reynolds)
            margin = epsilon - 0.8
            dense = epsilon <= 0.8
            fit_reynolds = jnp.where(dense, reynolds, epsilon * reynolds)
            factor = jnp.where(dense, ergun, wen_yu)
            code = jnp.asarray(~dense, dtype=jnp.int32)
            smooth = (margin != 0.0) & (dense | (smooth & (fit_reynolds != 1000.0)))
        case "huilin_gidaspow":
            ergun = (150.0 * (1.0 - epsilon) / epsilon + 1.75 * reynolds) / 18.0
            wen_yu = _wen_yu_resistance(epsilon, reynolds)
            # Both branches have nonzero weight; admission must cover both fits.
            fit_reynolds = jnp.maximum(reynolds, epsilon * reynolds)
            margin = epsilon - 0.8
            ergun_weight = 0.5 + jnp.arctan(150.0 * 1.75 * (0.8 - epsilon)) / jnp.pi
            factor = ergun_weight * ergun + (1.0 - ergun_weight) * wen_yu
            code = jnp.full_like(code, 2)
            smooth = smooth & (epsilon * reynolds != 1000.0)
        case "beetstra":
            fit_reynolds = epsilon * reynolds
            factor = _beetstra_resistance(epsilon, reynolds)
            margin = 1000.0 - fit_reynolds
            supported = (epsilon >= 0.35) & (epsilon <= 0.9)
        case _:
            assert_never(correlation)
    return factor, code, margin, smooth, supported, fit_reynolds


class UnresolvedCFDEMCouplingPlan(StrictModule, NonTrainableState):
    dynamics: PreparedCompliantDEMDynamics
    transfer: PreparedMeshParticleGridSplat
    drag_law: AbstractInterphaseDragLaw
    include_pressure_gradient: bool = eqx.field(static=True)
    minimum_porosity: float = eqx.field(static=True)
    maximum_porosity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        dynamics: PreparedCompliantDEMDynamics,
        transfer: PreparedMeshParticleGridSplat,
        drag_law: AbstractInterphaseDragLaw,
        /,
        *,
        include_pressure_gradient: bool = True,
        minimum_porosity: float = 1.0e-3,
        maximum_porosity: float = 1.0,
    ) -> None:
        if not isinstance(dynamics.shapes, PreparedRigidSphereSet):
            raise ValueError(
                "Unresolved CFD–DEM coupling requires sphere contact shapes."
            )
        particle_ids = dynamics.bodies.particles.particle_ids
        if transfer.particle_capacity != dynamics.bodies.capacity or not bool(
            jnp.all(transfer.stable_source_ids == particle_ids)
        ):
            raise ValueError(
                "CFD-DEM transfer and dynamics use different particle identities."
            )
        minimum = float(minimum_porosity)
        maximum = float(maximum_porosity)
        if not 0.0 < minimum < maximum <= 1.0:
            raise ValueError("Porosity bounds must satisfy 0 < min < max <= 1.")
        self.dynamics = dynamics
        self.transfer = transfer
        self.drag_law = drag_law
        self.include_pressure_gradient = bool(include_pressure_gradient)
        self.minimum_porosity = minimum
        self.maximum_porosity = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unresolved-cfd-dem-coupling-plan",
                "dynamics": dynamics.prepared_id,
                "transfer": transfer.prepared_id,
                "drag_law": drag_law.law_id,
                "include_pressure_gradient": self.include_pressure_gradient,
                "porosity": [minimum, maximum],
            }
        )


class CFDEMCouplingEvaluation(StrictModule):
    relation: MeshSplatRoutes
    particle_force: Array
    particle_torque: Array
    fluid_momentum_source_rate: Array
    particle_impulse: Array
    fluid_impulse: Array
    porosity: Array
    solid_velocity: Array
    momentum_residual: Array
    drag: InterphaseDragEvaluation
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_unresolved_cfd_dem(
    plan: UnresolvedCFDEMCouplingPlan,
    dem_state: DEMRuntimeState,
    fluid_velocity: ArrayLike,
    fluid_density: ArrayLike,
    dynamic_viscosity: ArrayLike,
    pressure_gradient: ArrayLike,
    particle_volume: ArrayLike,
    step_size: ArrayLike,
    /,
) -> CFDEMCouplingEvaluation:
    if not isinstance(plan, UnresolvedCFDEMCouplingPlan):
        raise TypeError("plan must be an UnresolvedCFDEMCouplingPlan.")
    transfer = plan.transfer
    active = dem_state.body_properties.active
    relation = transfer.routes(dem_state.kinematics.position, active)
    velocity = jnp.asarray(fluid_velocity)
    density = jnp.asarray(fluid_density)
    viscosity = jnp.asarray(dynamic_viscosity)
    pressure = jnp.asarray(pressure_gradient)
    volume = jnp.asarray(particle_volume, dtype=velocity.dtype)
    cell_count = transfer.target.entity_count
    dimension = plan.dynamics.bodies.ambient_dimension
    if velocity.shape != (cell_count, dimension) or pressure.shape != velocity.shape:
        raise ValueError("Fluid velocity/pressure-gradient shape is invalid.")
    if density.shape != (cell_count,) or viscosity.shape != density.shape:
        raise ValueError("Fluid density/viscosity shape is invalid.")
    if volume.shape != (plan.dynamics.bodies.capacity,):
        raise ValueError("particle_volume shape is invalid.")
    solid_volume_result = transfer.deposit(dem_state.kinematics.position, active, volume)
    solid_volume = solid_volume_result.content
    porosity = 1.0 - solid_volume / transfer.target.measure.weights.astype(
        solid_volume.dtype
    )
    momentum_result = transfer.deposit(
        dem_state.kinematics.position,
        active,
        volume[:, None] * dem_state.kinematics.velocity,
    )
    momentum = momentum_result.content
    solid_velocity = momentum / jnp.where(solid_volume > 0.0, solid_volume, 1.0)[:, None]
    porosity_result = transfer.gather(dem_state.kinematics.position, active, porosity)
    velocity_result = transfer.gather(dem_state.kinematics.position, active, velocity)
    density_result = transfer.gather(dem_state.kinematics.position, active, density)
    viscosity_result = transfer.gather(dem_state.kinematics.position, active, viscosity)
    pressure_result = transfer.gather(dem_state.kinematics.position, active, pressure)
    sample = FluidParticleSample(
        velocity_result.values,
        density_result.values,
        viscosity_result.values,
        pressure_result.values,
        porosity_result.values,
    )
    shapes = plan.dynamics.shapes
    if not isinstance(shapes, PreparedRigidSphereSet):
        raise ValueError("Unresolved CFD–DEM coupling requires sphere contact shapes.")
    slip = sample.velocity - dem_state.kinematics.velocity
    drag = plan.drag_law.evaluate(
        # Inactive capacity slots have no physical phase; benign inputs only
        # avoid evaluating undefined denominators in the masked lanes.
        jnp.where(active, sample.porosity, 0.9),
        jnp.where(active, jnp.linalg.norm(slip, axis=-1), 0.0),
        jnp.where(
            active, 2.0 * shapes.radii * dem_state.body_properties.shape_scale, 1.0
        ),
        jnp.where(active, sample.dynamic_viscosity, 1.0),
        jnp.where(active, sample.density, 1.0),
        jnp.where(active, volume, 1.0),
    )
    drag_force = drag.particle_drag_coefficient[:, None] * slip
    pressure_force = (
        -volume[:, None] * sample.pressure_gradient
        if plan.include_pressure_gradient
        else jnp.zeros_like(drag_force)
    )
    particle_force = jnp.where(active[:, None], drag_force + pressure_force, 0.0)
    angular_dimension = 1 if dimension == 2 else 3
    particle_torque = jnp.zeros(
        (plan.dynamics.bodies.capacity, angular_dimension), dtype=velocity.dtype
    )
    fluid_source_result = transfer.deposit(
        dem_state.kinematics.position, active, -particle_force
    )
    fluid_source = fluid_source_result.content
    dt = jnp.asarray(step_size, dtype=velocity.dtype)
    particle_impulse = dt * particle_force
    fluid_impulse = dt * fluid_source
    momentum_residual = jnp.sum(particle_impulse, axis=0) + jnp.sum(fluid_impulse, axis=0)
    porosity_valid = jnp.all(
        jnp.isfinite(porosity)
        & (porosity >= plan.minimum_porosity)
        & (porosity <= plan.maximum_porosity)
    )
    successful = (
        jnp.all(relation.evidence.complete)
        & solid_volume_result.successful
        & momentum_result.successful
        & porosity_result.successful
        & velocity_result.successful
        & density_result.successful
        & viscosity_result.successful
        & pressure_result.successful
        & fluid_source_result.successful
        & jnp.all(jnp.where(active, drag.admissible, True))
        & jnp.all(jnp.isfinite(particle_force))
        & porosity_valid
        & jnp.all(jnp.isfinite(solid_velocity))
        & (jnp.linalg.norm(momentum_residual) <= 1.0e-10)
    )
    return CFDEMCouplingEvaluation(
        relation,
        particle_force,
        particle_torque,
        fluid_source,
        particle_impulse,
        fluid_impulse,
        porosity,
        solid_velocity,
        momentum_residual,
        drag,
        successful,
        plan.plan_id,
    )


__all__ = [
    "AbstractInterphaseDragLaw",
    "CFDEMCouplingEvaluation",
    "FluidParticleSample",
    "InterphaseDragCorrelation",
    "InterphaseDragEvaluation",
    "InterphaseDragLawPlan",
    "UnresolvedCFDEMCouplingPlan",
    "evaluate_unresolved_cfd_dem",
]

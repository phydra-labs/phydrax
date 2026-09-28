#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._lorentz import boost_fields, boost_proper_velocity
from ..._physical import ElectromagneticScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import ParticleSetPlan, PreparedTensorGrid
from ...discretization.splatting import ParticleGridSplatPlan, PreparedParticleGridSplat
from ...operators.integral._free_space_convolution import FreeSpaceConvolutionPlan
from ._beam import AcceleratorBunch


class SpaceChargeIGFResult(StrictModule):
    """One rest-frame integrated-Green-function space-charge kick.

    Grid fields live on the plan grid in centroid-relative rest-frame
    coordinates ``(x − x̄, y − ȳ, γ₀(z − z̄))``. ``electric_field`` and
    ``magnetic_field`` are the lab-frame fields on that same grid;
    ``momentum_kick`` is ``Δ(p c)/(p₀ c)`` per particle. ``support`` marks
    particles that were fully deposited and gathered inside the grid.
    ``reference_gamma`` is the Lorentz factor ``γ₀`` of the rest frame.
    ``maximum_rest_frame_speed`` is the largest rest-frame ``|β'|`` of an
    active particle: the electrostatic rest-frame solve is valid only when it
    is small. ``cells_per_sigma`` is the charge-weighted rest-frame rms size of
    the deposited bunch per grid axis in cells; ``resolved`` holds when every
    axis reaches the plan's ``minimum_cells_per_sigma``. The bunch is updated
    only when ``accepted`` holds.
    """

    bunch: AcceleratorBunch
    momentum_kick: Array
    rest_frame_potential: Array
    rest_frame_electric_field: Array
    electric_field: Array
    magnetic_field: Array
    charge_density: Array
    deposited_charge: Array
    total_charge: Array
    reference_gamma: Array
    maximum_rest_frame_speed: Array
    cells_per_sigma: Array
    resolved: Array
    support: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class SpaceChargeIGFKick(NamedTuple):
    """Traceable arrays of one kick: updated coordinates plus the result fields.

    ``coordinates`` are the kicked bunch coordinates (unchanged unless
    ``accepted``); the remaining fields match :class:`SpaceChargeIGFResult`.
    """

    coordinates: Array
    momentum_kick: Array
    rest_frame_potential: Array
    rest_frame_electric_field: Array
    electric_field: Array
    magnetic_field: Array
    charge_density: Array
    deposited_charge: Array
    total_charge: Array
    reference_gamma: Array
    maximum_rest_frame_speed: Array
    cells_per_sigma: Array
    resolved: Array
    support: Array
    finite: Array
    accepted: Array


def _longitudinal_sign(bunch: AcceleratorBunch) -> float:
    """Sign mapping the bunch longitudinal coordinate onto the direction of motion."""
    sign = bunch.convention.longitudinal_sign
    if sign == "positive-late":
        return -1.0
    if sign == "positive-early":
        return 1.0
    raise ValueError(
        "Space charge requires a 'positive-late' or 'positive-early' longitudinal sign."
    )


class SpaceChargeIGFPlan(StrictModule, NonTrainableState):
    """Native rest-frame space-charge kick with integrated Green functions.

    Each ``kick`` boosts the bunch into the frame of the reference particle,
    deposits the macro-charge on ``grid`` (centroid-relative rest-frame
    coordinates, multilinear assignment), solves the open-boundary Poisson
    equation with the cell-integrated Coulomb kernel (Qiang et al. 2006) and
    its face-difference field kernels (exact for the deposited density,
    second-order accurate at any cell aspect ratio), transforms the
    electrostatic field back to the lab frame through ``boost_fields``, gathers
    ``E + v × B`` on the particles, and applies the exact three-momentum
    increment over the lab path length ``step_length``.

    ``reference_rest_energy`` and ``reference_momentum`` (``p₀ c``) of the
    bunch are read in the energy unit of ``scale``; positions use its length
    unit and ``reference_charge`` counts elementary charges. The provider kick
    route (:class:`SpaceChargeKickPlan`) remains distinct: it applies an
    external field solve, this plan owns its own.

    Deposit and gather each smooth the bunch over one cell, so the field error
    is set by the rest-frame rms size per axis in cells (Gaussian bunch,
    longitudinal field at the particles: ≈ 4 % at 4 cells, 7 % at 3, 15 % at
    2, 45 % at 1, independent of the cell aspect ratio). A kick whose
    ``cells_per_sigma`` falls below ``minimum_cells_per_sigma`` on any axis is
    refused.
    """

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    grid: PreparedTensorGrid
    convolution: FreeSpaceConvolutionPlan
    splat: PreparedParticleGridSplat
    capacity: int = eqx.field(static=True)
    maximum_rest_frame_speed: float = eqx.field(static=True)
    minimum_cells_per_sigma: float = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    elementary_charge: float = eqx.field(static=True)
    vacuum_permittivity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        grid: PreparedTensorGrid,
        /,
        *,
        capacity: int,
        maximum_rest_frame_speed: float = 0.3,
        minimum_cells_per_sigma: float = 3.0,
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        if len(grid.shape) != 3:
            raise ValueError("Space-charge deposition requires a three-dimensional grid.")
        capacity_ = int(capacity)
        if capacity_ < 1:
            raise ValueError("capacity must be positive.")
        speed_limit = float(maximum_rest_frame_speed)
        if not np.isfinite(speed_limit) or not 0.0 < speed_limit < 1.0:
            raise ValueError("maximum_rest_frame_speed must lie in (0, 1).")
        resolution = float(minimum_cells_per_sigma)
        if not np.isfinite(resolution) or resolution < 0.0:
            raise ValueError("minimum_cells_per_sigma must be finite and nonnegative.")
        convolution = FreeSpaceConvolutionPlan("coulomb-igf", grid, gradient=True)
        particles = ParticleSetPlan(
            np.arange(capacity_, dtype=np.int64),
            np.ones((capacity_,), dtype=np.float64),
            ambient_dimension=3,
            name="space-charge-macroparticles",
        ).prepare()
        splat = ParticleGridSplatPlan(grid, boundary="drop").prepare(particles)
        self.scale = scale
        self.grid = grid
        self.convolution = convolution
        self.splat = splat
        self.capacity = capacity_
        self.maximum_rest_frame_speed = speed_limit
        self.minimum_cells_per_sigma = resolution
        self.speed_of_light = float(scale.speed_of_light)
        self.elementary_charge = float(scale.elementary_charge)
        self.vacuum_permittivity = float(scale.vacuum_permittivity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-space-charge-igf",
                "scale": scale.scale_id,
                "convolution": convolution.plan_id,
                "splat": splat.prepared_id,
                "capacity": capacity_,
                "maximum_rest_frame_speed": speed_limit,
                "minimum_cells_per_sigma": resolution,
            }
        )

    def kick(
        self,
        bunch: AcceleratorBunch,
        step_length: ArrayLike,
        /,
        *,
        frame_lorentz_factor: ArrayLike | None = None,
    ) -> SpaceChargeIGFResult:
        kick = self.evaluate(
            bunch, step_length, frame_lorentz_factor=frame_lorentz_factor
        )
        result_bunch = AcceleratorBunch(
            kick.coordinates,
            bunch.weights,
            bunch.particle_ids,
            active=bunch.active,
            reference_rest_energy=float(bunch.reference_rest_energy),
            reference_momentum=float(bunch.reference_momentum),
            reference_charge=float(bunch.reference_charge),
            convention=bunch.convention,
            bunch_id=f"{bunch.bunch_id}:{self.plan_id}",
        )
        return SpaceChargeIGFResult(
            result_bunch,
            kick.momentum_kick,
            kick.rest_frame_potential,
            kick.rest_frame_electric_field,
            kick.electric_field,
            kick.magnetic_field,
            kick.charge_density,
            kick.deposited_charge,
            kick.total_charge,
            kick.reference_gamma,
            kick.maximum_rest_frame_speed,
            kick.cells_per_sigma,
            kick.resolved,
            kick.support,
            kick.finite,
            kick.accepted,
            self.plan_id,
        )

    def evaluate(
        self,
        bunch: AcceleratorBunch,
        step_length: ArrayLike,
        /,
        *,
        frame_lorentz_factor: ArrayLike | None = None,
    ) -> SpaceChargeIGFKick:
        """Traceable kick arrays without rebuilding the (host-validated) bunch.

        ``frame_lorentz_factor`` selects the quasi-static frame boosted along
        the reference axis (default: the bunch reference ``γ₀``), for example
        the frame of the mean longitudinal motion ``γ_z = γ/√(1 + a_w²)`` in an
        undulator; the particle coordinates must then describe that mean
        motion. A nonfinite frame or one at ``γ ≤ 1`` is refused.
        """
        if not isinstance(bunch, AcceleratorBunch):
            raise TypeError("bunch must be an AcceleratorBunch.")
        if bunch.capacity != self.capacity:
            raise ValueError("bunch capacity must match the space-charge plan capacity.")
        longitudinal_sign = _longitudinal_sign(bunch)
        coordinates = bunch.coordinates
        step = jnp.asarray(step_length, dtype=coordinates.dtype).reshape(())
        light = self.speed_of_light
        reference_momentum = bunch.reference_momentum
        rest_energy = bunch.reference_rest_energy

        # Lab kinematics: p c in energy units, u = γβ = p c / (m c²).
        transverse = jnp.stack((coordinates[:, 1], coordinates[:, 3]), axis=-1)
        transverse = transverse * reference_momentum
        total = reference_momentum * (1.0 + coordinates[:, 5])
        longitudinal_squared = total * total - jnp.sum(transverse * transverse, axis=-1)
        kinematic_valid = longitudinal_squared > 0.0
        longitudinal = jnp.sqrt(jnp.where(kinematic_valid, longitudinal_squared, 1.0))
        momentum = jnp.concatenate((transverse, longitudinal[:, None]), axis=-1)
        proper = momentum / rest_energy
        gamma = jnp.sqrt(1.0 + jnp.sum(proper * proper, axis=-1))
        beta = proper / gamma[:, None]
        if frame_lorentz_factor is None:
            reference_proper = reference_momentum / rest_energy
            reference_gamma = jnp.sqrt(1.0 + reference_proper * reference_proper)
        else:
            reference_gamma = jnp.asarray(
                frame_lorentz_factor, dtype=coordinates.dtype
            ).reshape(())
        frame_valid = jnp.isfinite(reference_gamma) & (reference_gamma > 1.0)
        safe_gamma = jnp.where(frame_valid, reference_gamma, 2.0)
        reference_beta = jnp.sqrt(1.0 - 1.0 / (safe_gamma * safe_gamma))
        boost = jnp.stack(
            (
                jnp.zeros_like(reference_beta),
                jnp.zeros_like(reference_beta),
                reference_beta,
            )
        )

        active = bunch.active & bunch.valid & kinematic_valid
        rest_proper = boost_proper_velocity(boost, proper)
        rest_speed = jnp.sqrt(
            jnp.sum(rest_proper * rest_proper, axis=-1)
            / (1.0 + jnp.sum(rest_proper * rest_proper, axis=-1))
        )
        maximum_rest_speed = jnp.max(jnp.where(active, rest_speed, 0.0), initial=0.0)

        # Rest-frame positions relative to the charge centroid; the longitudinal
        # snapshot coordinate dilates by γ₀ (charge is invariant, density is not).
        weights = jnp.where(active, bunch.weights, 0.0)
        total_weight = jnp.sum(weights)
        lab_positions = jnp.stack(
            (coordinates[:, 0], coordinates[:, 2], longitudinal_sign * coordinates[:, 4]),
            axis=-1,
        )
        centroid = jnp.sum(weights[:, None] * lab_positions, axis=0) / jnp.maximum(
            total_weight, jnp.finfo(coordinates.dtype).tiny
        )
        relative = lab_positions - centroid
        rest_positions = relative.at[:, 2].multiply(safe_gamma)
        rest_positions = jnp.where(active[:, None], rest_positions, 0.0)
        # Charge-weighted rest-frame rms size per axis, in cells.
        variance = jnp.sum(weights[:, None] * rest_positions**2, axis=0) / jnp.maximum(
            total_weight, jnp.finfo(coordinates.dtype).tiny
        )
        cells_per_sigma = jnp.sqrt(variance) / self.convolution.spacing
        resolved = jnp.all(cells_per_sigma >= self.minimum_cells_per_sigma)

        charge_per_particle = bunch.reference_charge * self.elementary_charge
        state = self.splat.build(rest_positions, active_mask=active)
        deposit = self.splat.deposit_content(state, charge_per_particle * weights)
        convolved = self.convolution.convolve(deposit.density)
        if convolved.gradient is None:
            raise AssertionError("Space-charge convolution lost its gradient kernels.")
        potential = convolved.field / self.vacuum_permittivity
        rest_electric = -convolved.gradient / self.vacuum_permittivity
        electric, magnetic = boost_fields(
            -boost,
            rest_electric,
            jnp.zeros_like(rest_electric),
            speed_of_light=jnp.asarray(light, dtype=rest_electric.dtype),
        )
        gathered = self.splat.gather(
            state, jnp.concatenate((electric, magnetic), axis=-1)
        )
        sampled_electric = gathered.values[:, :3]
        sampled_magnetic = gathered.values[:, 3:]
        support = gathered.support & state.supported_mask & ~state.truncated_support_mask

        # Δ(p c) = q (E + v × B) Δs / β_z in the lab frame.
        velocity = light * beta
        force = sampled_electric + jnp.cross(velocity, sampled_magnetic, axis=-1)
        safe_beta_z = jnp.where(active, beta[:, 2], 1.0)
        increment = charge_per_particle * force * step / safe_beta_z[:, None]
        increment = jnp.where(active[:, None], increment, 0.0)
        updated = momentum + increment
        new_total = jnp.sqrt(jnp.sum(updated * updated, axis=-1))
        candidate = coordinates.at[:, 1].set(updated[:, 0] / reference_momentum)
        candidate = candidate.at[:, 3].set(updated[:, 1] / reference_momentum)
        candidate = candidate.at[:, 5].set(new_total / reference_momentum - 1.0)
        candidate = jnp.where(active[:, None], candidate, coordinates)

        finite = (
            deposit.successful
            & convolved.finite
            & jnp.all(jnp.isfinite(electric))
            & jnp.all(jnp.isfinite(magnetic))
            & jnp.all(jnp.where(active[:, None], jnp.isfinite(increment), True))
            & jnp.isfinite(step)
        )
        accepted = (
            finite
            & frame_valid
            & resolved
            & jnp.all(jnp.where(bunch.active & bunch.valid, kinematic_valid, True))
            & jnp.all(jnp.where(active, support, True))
            & (maximum_rest_speed <= self.maximum_rest_frame_speed)
        )
        return SpaceChargeIGFKick(
            jnp.where(accepted, candidate, coordinates),
            increment / reference_momentum,
            potential,
            rest_electric,
            electric,
            magnetic,
            deposit.density,
            deposit.balance.target_total,
            deposit.balance.active_source_total,
            reference_gamma,
            maximum_rest_speed,
            cells_per_sigma,
            resolved,
            support & active,
            finite,
            accepted,
        )


__all__ = ["SpaceChargeIGFKick", "SpaceChargeIGFPlan", "SpaceChargeIGFResult"]

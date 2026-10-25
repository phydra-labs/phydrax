#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical discrete generalized-Langevin owner and the overdamped entry point.

``GeneralizedLangevinPlan`` is the single GLE execution owner of atomistic
dynamics.  It is an in-schedule thermostat of ``PreparedAtomisticDynamics``
(``O(dt/2) K(dt/2) D(dt) [constraints] K(dt/2) O(dt/2)``) whose half-step
operator is a fixed discrete transition in thermal-reduced momentum
coordinates ``s = p / sqrt(m kT / kinetic_to_energy)``::

    [s, z]' = T [s, z] + S xi,      T T^T + S S^T = I  (discrete FDT).

The discrete fluctuation-dissipation identity is checked at construction and
reported at execution; the target temperature comes from the bound
thermodynamic row and the auxiliary memory is a typed dynamics continuation.
Noise is addressed per particle and per channel by stable particle identity,
committed step, half-step phase and channel.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from phydrax.ein import contract

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._thermal import stable_particle_normals


if TYPE_CHECKING:
    from ._dynamics import PreparedAtomisticDynamics
    from ._hydrodynamic_brownian import PreparedHydrodynamicBrownian


GLE_OPERATOR_BASE = 16


class OverdampedAtomisticPlan(StrictModule, NonTrainableState):
    """Overdamped (Brownian) entry point over the hydrodynamic Brownian owner.

    ``step_size`` and ``realization_id`` are static selectors; ``mobility``
    and ``temperature`` are dynamic ``float64`` scalars whose exact host
    values key ``plan_identity`` (host ``plan_id``, unchanged digest).
    """

    step_size: float = eqx.field(static=True)
    mobility: Array
    temperature: Array
    realization_id: int = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(
        self,
        step_size: float,
        mobility: float,
        temperature: float,
        /,
        *,
        realization_id: int = 0,
    ) -> None:
        step = float(step_size)
        mobility_ = float(mobility)
        thermal = float(temperature)
        realization = int(realization_id)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(mobility_)
            or mobility_ <= 0.0
            or not math.isfinite(thermal)
            or thermal <= 0.0
            or realization < 0
        ):
            raise ValueError("Overdamped atomistic controls are invalid.")
        self.step_size = step
        self.mobility = jnp.asarray(mobility_, dtype=jnp.float64)
        self.temperature = jnp.asarray(thermal, dtype=jnp.float64)
        self.realization_id = realization
        self.plan_identity = IdentityWords.of(
            {
                "kind": "overdamped-atomistic-plan",
                "step_size": step,
                "mobility": mobility_,
                "temperature": thermal,
                "realization_id": realization,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    def prepare(
        self, dynamics: PreparedAtomisticDynamics, /
    ) -> PreparedHydrodynamicBrownian:
        """Host preparation: the declared host scalars configure the owner."""
        from ._hydrodynamic_brownian import HydrodynamicBrownianPlan
        from ._hydrodynamic_mobility import ConstantIsotropicMobilityPlan

        return HydrodynamicBrownianPlan(
            self.step_size,
            float(self.temperature),
            realization_id=self.realization_id,
        ).prepare(
            dynamics,
            ConstantIsotropicMobilityPlan(
                float(self.mobility),
                maximum_particles=dynamics.system.capacity,
            ),
        )


class GeneralizedLangevinPlan(StrictModule, NonTrainableState):
    """Canonical NVT generalized-Langevin integrator over a discrete transition.

    ``transition_matrix`` and ``noise_factor`` are the exact half-step
    propagator of the momentum channel plus ``K`` auxiliary channels in
    thermal-reduced coordinates.  The method is canonical exactly because the
    discrete FDT holds within ``covariance_tolerance``.  The plan digest binds
    the dynamic matrices, so it is the dynamic ``plan_identity`` (host
    ``plan_id``, unchanged digest); step size, tolerance and realization
    label are static selectors.
    """

    step_size: float = eqx.field(static=True)
    transition_matrix: Array
    noise_factor: Array
    covariance_tolerance: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(
        self,
        step_size: float,
        transition_matrix: ArrayLike,
        noise_factor: ArrayLike,
        /,
        *,
        covariance_tolerance: float = 1.0e-10,
        realization_id: int = 0,
    ) -> None:
        step = float(step_size)
        transition = np.asarray(transition_matrix, dtype=np.float64)
        noise = np.asarray(noise_factor, dtype=np.float64)
        tolerance = float(covariance_tolerance)
        realization = int(realization_id)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or transition.ndim != 2
            or transition.shape[0] != transition.shape[1]
            or noise.shape != transition.shape
            or transition.shape[0] == 0
            or np.any(~np.isfinite(transition))
            or np.any(~np.isfinite(noise))
            or not math.isfinite(tolerance)
            or tolerance < 0.0
            or realization < 0
        ):
            raise ValueError("Generalized Langevin controls are invalid.")
        residual = (
            transition @ transition.T + noise @ noise.T - np.eye(transition.shape[0])
        )
        if np.max(np.abs(residual)) > tolerance:
            raise ValueError(
                "Generalized Langevin transition and noise violate the discrete FDT."
            )
        self.step_size = step
        self.transition_matrix = jnp.asarray(transition)
        self.noise_factor = jnp.asarray(noise)
        self.covariance_tolerance = tolerance
        self.realization_id = realization
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-generalized-langevin",
                "realization": "discrete-half-step-transition",
                "step_size": step,
                "transition_matrix": transition.tolist(),
                "noise_factor": noise.tolist(),
                "covariance_tolerance": tolerance,
                "realization_id": realization,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def auxiliary_count(self) -> int:
        return self.transition_matrix.shape[0] - 1

    @property
    def canonical(self) -> bool:
        return True

    @property
    def stochastic(self) -> bool:
        return True

    def covariance_residual(self) -> Array:
        channels = self.transition_matrix.shape[0]
        return jnp.max(
            jnp.abs(
                self.transition_matrix @ self.transition_matrix.T
                + self.noise_factor @ self.noise_factor.T
                - jnp.eye(channels, dtype=self.transition_matrix.dtype)
            )
        )


class GeneralizedLangevinMemoryState(StrictModule):
    """Typed GLE continuation: auxiliary momenta ``(N, 3, K)`` in reduced units."""

    memory: Array


def initial_generalized_langevin_memory(
    plan: GeneralizedLangevinPlan, particle_capacity: int, dtype: DTypeLike, /
) -> GeneralizedLangevinMemoryState:
    return GeneralizedLangevinMemoryState(
        jnp.zeros((int(particle_capacity), 3, plan.auxiliary_count), dtype=dtype)
    )


class GeneralizedLangevinEvaluation(StrictModule):
    momenta: Array
    memory: GeneralizedLangevinMemoryState
    heat: Array
    covariance_residual: Array
    successful: Array


def apply_generalized_langevin(
    plan: GeneralizedLangevinPlan,
    key_data: ArrayLike,
    particle_ids: ArrayLike,
    momenta: ArrayLike,
    memory: GeneralizedLangevinMemoryState,
    masses: ArrayLike,
    mobile_mask: ArrayLike,
    step_index: ArrayLike,
    temperature: ArrayLike,
    /,
    *,
    phase: int,
    boltzmann_constant: float,
    kinetic_to_energy: float,
) -> GeneralizedLangevinEvaluation:
    """Apply one addressed half-step GLE transition at the row temperature."""
    if not isinstance(plan, GeneralizedLangevinPlan):
        raise TypeError("plan must be a GeneralizedLangevinPlan.")
    if phase not in (0, 1):
        raise ValueError("phase must be 0 (first half) or 1 (second half).")
    momentum = jnp.asarray(momenta)
    mass = jnp.asarray(masses, dtype=momentum.dtype)
    mobile = jnp.asarray(mobile_mask, dtype=jnp.bool_)
    thermal = jnp.asarray(temperature, dtype=momentum.dtype)
    channels = plan.transition_matrix.shape[0]
    expected = momentum.shape + (plan.auxiliary_count,)
    if memory.memory.shape != expected:
        raise ValueError(f"GLE memory must have shape {expected}.")
    scale = jnp.sqrt(mass[:, None] * boltzmann_constant * thermal / kinetic_to_energy)
    safe = jnp.where(scale > 0.0, scale, 1.0)
    combined = jnp.concatenate(
        (jnp.where(mobile[:, None], momentum / safe, 0.0)[..., None], memory.memory),
        axis=-1,
    )
    noise = jnp.stack(
        [
            stable_particle_normals(
                key_data,
                particle_ids,
                step_index,
                operator_id=GLE_OPERATOR_BASE + phase * channels + channel,
                realization_id=plan.realization_id,
                dtype=momentum.dtype,
            )
            for channel in range(channels)
        ],
        axis=-1,
    )
    transition = plan.transition_matrix.astype(momentum.dtype)
    factor = plan.noise_factor.astype(momentum.dtype)
    updated = contract("...i,ji->...j", combined, transition) + contract(
        "...i,ji->...j", noise, factor
    )
    new_momentum = jnp.where(mobile[:, None], updated[..., 0] * scale, 0.0)
    new_memory = jnp.where(mobile[:, None, None], updated[..., 1:], 0.0)
    before = (
        0.5
        * kinetic_to_energy
        * jnp.sum(jnp.where(mobile[:, None], momentum * momentum / mass[:, None], 0.0))
    )
    after = (
        0.5
        * kinetic_to_energy
        * jnp.sum(
            jnp.where(mobile[:, None], new_momentum * new_momentum / mass[:, None], 0.0)
        )
    )
    residual = plan.covariance_residual()
    successful = (
        (thermal > 0.0)
        & jnp.isfinite(thermal)
        & jnp.all(jnp.isfinite(updated))
        & (residual <= plan.covariance_tolerance)
    )
    return GeneralizedLangevinEvaluation(
        new_momentum,
        GeneralizedLangevinMemoryState(new_memory),
        after - before,
        residual,
        successful,
    )


__all__ = [
    "GLE_OPERATOR_BASE",
    "GeneralizedLangevinEvaluation",
    "GeneralizedLangevinMemoryState",
    "GeneralizedLangevinPlan",
    "OverdampedAtomisticPlan",
    "apply_generalized_langevin",
    "initial_generalized_langevin_memory",
]

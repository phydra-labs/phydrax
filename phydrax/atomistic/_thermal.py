#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermostat methods and their addressed numerical kernels.

Every method here receives its target temperature from the bound
thermodynamic row and its degrees of freedom from the prepared phase-space
measure; no plan carries a hidden physical temperature.  Stochastic draws are
addressed by run key, realization, committed step, operator identity and
either stable particle identity (per-particle draws) or a scalar-draw index
(global draws), so placement and batching never redefine the noise.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState


# Stable operator identities of the addressed thermostat draws.  Particle and
# scalar draws never share an identity.
BAOAB_OPERATOR_ID = 2
CSVR_FIRST_OPERATOR_ID = 3
CSVR_SECOND_OPERATOR_ID = 4

# Suzuki fractal compositions used by the source NHC owner (orders 1, 3, 5).
_YS3_OUTER = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YS5_OUTER = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
_YOSHIDA_SUZUKI_WEIGHTS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    3: (_YS3_OUTER, 1.0 - 2.0 * _YS3_OUTER, _YS3_OUTER),
    5: (
        _YS5_OUTER,
        _YS5_OUTER,
        1.0 - 4.0 * _YS5_OUTER,
        _YS5_OUTER,
        _YS5_OUTER,
    ),
}
MAXIMUM_NOSE_HOOVER_CHAIN_LENGTH = 8


def _positive_step(step_size: float, name: str, /) -> float:
    step = float(step_size)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return step


def _realization(realization_id: int, /) -> int:
    value = int(realization_id)
    if value < 0:
        raise ValueError("realization_id must be non-negative.")
    return value


def _thermal_key(
    key_data: ArrayLike,
    step_index: ArrayLike,
    /,
    *,
    operator_id: int,
    realization_id: int,
) -> Array:
    data = jnp.asarray(key_data, dtype=jnp.uint32)
    if data.shape != (2,):
        raise ValueError("key_data must contain two uint32 words.")
    if type(operator_id) is not int or operator_id < 0:
        raise ValueError("operator_id must be a non-negative integer.")
    if type(realization_id) is not int or realization_id < 0:
        raise ValueError("realization_id must be a non-negative integer.")
    step = jnp.asarray(step_index)
    if step.shape != () or step.dtype.kind not in "iu":
        raise TypeError("step_index must be an integer scalar.")
    data = eqx.error_if(data, step < 0, "step_index must be non-negative.")
    key = jr.wrap_key_data(data)
    key = jr.fold_in(key, jnp.asarray(realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, step.astype(jnp.uint32))
    return jr.fold_in(key, jnp.asarray(operator_id, dtype=jnp.uint32))


class BAOABLangevinPlan(StrictModule, NonTrainableState):
    """Fixed-step BAOAB numerical kernel with thermodynamics supplied at execution.

    ``step_size`` and ``realization_id`` are static method selectors (time
    grid and declared random stream label).  ``friction`` is a dynamic
    ``float64`` scalar, so friction revisions share one executable; the exact
    host value keys ``plan_identity`` (host ``plan_id``, unchanged digest).
    """

    step_size: float = eqx.field(static=True)
    friction: Array
    realization_id: int = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(
        self,
        step_size: float,
        friction: float,
        /,
        *,
        realization_id: int = 0,
    ) -> None:
        step = float(step_size)
        damping = float(friction)
        realization = int(realization_id)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(damping)
            or damping <= 0.0
            or realization < 0
        ):
            raise ValueError(
                "BAOAB step and friction must be positive finite; realization_id must be non-negative."
            )
        self.step_size = step
        self.friction = jnp.asarray(damping, dtype=jnp.float64)
        self.realization_id = realization
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-baoab-langevin",
                "step_size": step,
                "friction": damping,
                "realization_id": realization,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def canonical(self) -> bool:
        return True

    @property
    def stochastic(self) -> bool:
        return True


class NoseHooverChainParameters(StrictModule, NonTrainableState):
    """Chain length, relaxation time and Suzuki-Yoshida composition of one chain.

    Masses follow the source mass policy ``Q_1 = N_f kT tau^2`` and
    ``Q_k = kT tau^2`` (``k > 1``), evaluated from the bound thermodynamic row
    and the admitted degrees of freedom at every application, so a warm
    temperature rebase rebinds the chain inertia explicitly.  Chain length,
    composition order and substeps are static selectors; ``time_constant`` is
    a dynamic ``float64`` scalar whose exact host value keys
    ``parameters_identity`` (host ``parameters_id``, unchanged digest).
    """

    chain_length: int = eqx.field(static=True)
    time_constant: Array
    yoshida_suzuki_order: int = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    parameters_identity: IdentityWords

    def __init__(
        self,
        time_constant: float,
        /,
        *,
        chain_length: int = 3,
        yoshida_suzuki_order: int = 1,
        substeps: int = 1,
    ) -> None:
        tau = _positive_step(time_constant, "time_constant")
        length = int(chain_length)
        order = int(yoshida_suzuki_order)
        count = int(substeps)
        if not 1 <= length <= MAXIMUM_NOSE_HOOVER_CHAIN_LENGTH:
            raise ValueError(
                f"chain_length must lie in [1, {MAXIMUM_NOSE_HOOVER_CHAIN_LENGTH}]."
            )
        if order not in _YOSHIDA_SUZUKI_WEIGHTS:
            raise ValueError("yoshida_suzuki_order must be 1, 3 or 5.")
        if count <= 0:
            raise ValueError("substeps must be positive.")
        self.chain_length = length
        self.time_constant = jnp.asarray(tau, dtype=jnp.float64)
        self.yoshida_suzuki_order = order
        self.substeps = count
        self.parameters_identity = IdentityWords.of(
            {
                "kind": "nose-hoover-chain-parameters",
                "mass_policy": "q1-dof-kt-tau2-qk-kt-tau2",
                "chain_length": length,
                "time_constant": tau,
                "yoshida_suzuki_order": order,
                "substeps": count,
            }
        )

    @property
    def parameters_id(self) -> str:
        """Host decoding of the canonical chain-parameter digest."""
        return self.parameters_identity.fingerprint()

    @property
    def weights(self) -> tuple[float, ...]:
        return _YOSHIDA_SUZUKI_WEIGHTS[self.yoshida_suzuki_order]


class NoseHooverChainState(StrictModule):
    """Typed continuation of one Nose-Hoover chain: positions and velocities."""

    positions: Array
    velocities: Array


def initial_nose_hoover_chain_state(
    parameters: NoseHooverChainParameters, dtype: DTypeLike, /
) -> NoseHooverChainState:
    zero = jnp.zeros((parameters.chain_length,), dtype=dtype)
    return NoseHooverChainState(zero, zero)


def nose_hoover_chain_masses(
    parameters: NoseHooverChainParameters,
    degrees_of_freedom: ArrayLike,
    thermal_energy: ArrayLike,
    /,
) -> Array:
    """Chain masses; ``degrees_of_freedom`` may be a traced integer scalar."""
    thermal = jnp.asarray(thermal_energy)
    tau_squared = jnp.asarray(
        parameters.time_constant * parameters.time_constant, dtype=thermal.dtype
    )
    masses = jnp.full((parameters.chain_length,), thermal * tau_squared)
    dof = jnp.asarray(degrees_of_freedom, dtype=thermal.dtype)
    return masses.at[0].set(dof * thermal * tau_squared)


def nose_hoover_chain_energy(
    parameters: NoseHooverChainParameters,
    state: NoseHooverChainState,
    degrees_of_freedom: ArrayLike,
    thermal_energy: ArrayLike,
    /,
) -> Array:
    """Extended chain energy ``sum Q v^2 / 2 + N_f kT xi_1 + kT sum_{k>1} xi_k``."""
    thermal = jnp.asarray(thermal_energy, dtype=state.velocities.dtype)
    masses = nose_hoover_chain_masses(parameters, degrees_of_freedom, thermal)
    kinetic = 0.5 * jnp.sum(masses * state.velocities * state.velocities)
    weights = (
        jnp.ones_like(state.positions)
        .at[0]
        .set(jnp.asarray(degrees_of_freedom, dtype=state.positions.dtype))
    )
    return kinetic + thermal * jnp.sum(weights * state.positions)


class NoseHooverChainEvaluation(StrictModule):
    state: NoseHooverChainState
    scale: Array
    successful: Array


def propagate_nose_hoover_chain(
    parameters: NoseHooverChainParameters,
    state: NoseHooverChainState,
    twice_kinetic: ArrayLike,
    degrees_of_freedom: ArrayLike,
    thermal_energy: ArrayLike,
    duration: ArrayLike,
    /,
) -> NoseHooverChainEvaluation:
    """Apply ``exp(iL_NHC * duration)`` to one chain and its coupled kinetic DOF.

    Martyna-Tuckerman-Tobias-Klein (Mol. Phys. 87, 1117, 1996) Trotter
    factorization: ``substeps`` repetitions of the configured Suzuki-Yoshida
    weights, each sub-duration ``d`` applying a backward chain sweep (thermostat
    velocities kicked by ``d/2`` between ``exp(-d v_{k+1}/4)`` scalings), the
    coupled momentum scaling ``exp(-d v_1)``, chain positions ``xi += d v`` and
    the mirrored forward sweep.  Returns the accumulated momentum scale; the
    caller scales its coupled momenta by it.
    """
    velocity = jnp.asarray(state.velocities)
    position = jnp.asarray(state.positions)
    dtype = velocity.dtype
    thermal = jnp.asarray(thermal_energy, dtype=dtype)
    kinetic = jnp.asarray(twice_kinetic, dtype=dtype)
    span = jnp.asarray(duration, dtype=dtype)
    masses = nose_hoover_chain_masses(parameters, degrees_of_freedom, thermal)
    dof = jnp.asarray(degrees_of_freedom, dtype=dtype)
    length = parameters.chain_length
    scale = jnp.ones((), dtype=dtype)
    values = [velocity[k] for k in range(length)]
    positions = [position[k] for k in range(length)]

    def force(index: int, kinetic_value: Array) -> Array:
        if index == 0:
            return (kinetic_value - dof * thermal) / masses[0]
        previous = values[index - 1]
        return (masses[index - 1] * previous * previous - thermal) / masses[index]

    for _ in range(parameters.substeps):
        for weight in parameters.weights:
            sub = span * (weight / parameters.substeps)
            values[length - 1] = values[length - 1] + 0.5 * sub * force(
                length - 1, kinetic
            )
            for index in range(length - 2, -1, -1):
                damping = jnp.exp(-0.25 * sub * values[index + 1])
                values[index] = values[index] * damping
                values[index] = values[index] + 0.5 * sub * force(index, kinetic)
                values[index] = values[index] * damping
            factor = jnp.exp(-sub * values[0])
            scale = scale * factor
            kinetic = kinetic * factor * factor
            positions = [
                positions[index] + sub * values[index] for index in range(length)
            ]
            for index in range(length - 1):
                damping = jnp.exp(-0.25 * sub * values[index + 1])
                values[index] = values[index] * damping
                values[index] = values[index] + 0.5 * sub * force(index, kinetic)
                values[index] = values[index] * damping
            values[length - 1] = values[length - 1] + 0.5 * sub * force(
                length - 1, kinetic
            )
    updated = NoseHooverChainState(jnp.stack(positions), jnp.stack(values))
    successful = (
        (dof > 0.0)
        & (thermal > 0.0)
        & jnp.isfinite(thermal)
        & jnp.isfinite(scale)
        & jnp.all(jnp.isfinite(updated.positions))
        & jnp.all(jnp.isfinite(updated.velocities))
    )
    return NoseHooverChainEvaluation(updated, scale, successful)


class NoseHooverChainPlan(StrictModule, NonTrainableState):
    """Canonical NVT Nose-Hoover chain integrator (MTK chain + velocity Verlet).

    Schedule ``T(dt/2) K(dt/2) D(dt) [constraints] K(dt/2) T(dt/2)``; the chain
    couples to all admitted particle kinetic degrees of freedom and its
    extended energy enters the conserved-energy ledger.  The plan digest is
    the dynamic ``plan_identity`` (host ``plan_id``) because it binds the
    dynamic chain time constant.
    """

    step_size: float = eqx.field(static=True)
    chain: NoseHooverChainParameters
    plan_identity: IdentityWords

    def __init__(self, step_size: float, chain: NoseHooverChainParameters, /) -> None:
        if not isinstance(chain, NoseHooverChainParameters):
            raise TypeError("chain must be NoseHooverChainParameters.")
        self.step_size = _positive_step(step_size, "step_size")
        self.chain = chain
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-nose-hoover-chain-nvt",
                "realization": "mttk-1996-trotter-velocity-verlet",
                "step_size": self.step_size,
                "chain": chain.parameters_id,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def canonical(self) -> bool:
        return True

    @property
    def stochastic(self) -> bool:
        return False


class StochasticVelocityRescalingPlan(StrictModule, NonTrainableState):
    """Canonical Bussi-Donadio-Parrinello stochastic velocity rescaling (CSVR).

    Schedule ``R(dt/2) K(dt/2) D(dt) [constraints] K(dt/2) R(dt/2)`` with the
    exact CSVR kinetic-energy propagator over each half step (J. Chem. Phys.
    126, 014101, 2007).  The two global draws of each application are scalar
    draws ``0`` (Gaussian) and ``1`` (chi-square with ``N_f - 1`` degrees).
    ``step_size`` and ``realization_id`` are static selectors;
    ``time_constant`` is a dynamic ``float64`` scalar whose exact host value
    keys ``plan_identity`` (host ``plan_id``, unchanged digest).
    """

    step_size: float = eqx.field(static=True)
    time_constant: Array
    realization_id: int = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(
        self,
        step_size: float,
        time_constant: float,
        /,
        *,
        realization_id: int = 0,
    ) -> None:
        self.step_size = _positive_step(step_size, "step_size")
        tau = _positive_step(time_constant, "time_constant")
        self.time_constant = jnp.asarray(tau, dtype=jnp.float64)
        self.realization_id = _realization(realization_id)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-stochastic-velocity-rescaling",
                "realization": "bussi-2007-exact-kinetic-propagator",
                "step_size": self.step_size,
                "time_constant": tau,
                "realization_id": self.realization_id,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def canonical(self) -> bool:
        return True

    @property
    def stochastic(self) -> bool:
        return True


class VelocityRescalingPlan(StrictModule, NonTrainableState):
    """Deterministic instantaneous velocity rescaling: a noncanonical control.

    After every ``interval``-th committed velocity Verlet step all admitted
    momenta are scaled by ``sqrt(T_target / T_instantaneous)``.  This is an
    equilibration/temperature-control operation, never a canonical sampler:
    ``canonical`` is false, the injected energy is booked as reservoir heat,
    and it refuses any thermodynamic use that requires a canonical measure.
    """

    step_size: float = eqx.field(static=True)
    interval: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, step_size: float, /, *, interval: int = 1) -> None:
        count = int(interval)
        if count <= 0:
            raise ValueError("interval must be positive.")
        self.step_size = _positive_step(step_size, "step_size")
        self.interval = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-deterministic-velocity-rescaling",
                "canonical": False,
                "step_size": self.step_size,
                "interval": count,
            }
        )

    @property
    def canonical(self) -> bool:
        return False

    @property
    def stochastic(self) -> bool:
        return False


def stable_scalar_normals(
    key_data: ArrayLike,
    step_index: ArrayLike,
    /,
    *,
    operator_id: int,
    realization_id: int,
    count: int,
    dtype: DTypeLike,
) -> Array:
    """Draw ``count`` global normals addressed by scalar-draw index."""
    key = _thermal_key(
        key_data, step_index, operator_id=operator_id, realization_id=realization_id
    )
    indices = jnp.arange(int(count), dtype=jnp.uint32)
    return jax.vmap(lambda index: jr.normal(jr.fold_in(key, index), (), dtype=dtype))(
        indices
    )


class VelocityScaleEvaluation(StrictModule):
    scale: Array
    successful: Array


def stochastic_velocity_rescaling_scale(
    plan: StochasticVelocityRescalingPlan,
    key_data: ArrayLike,
    step_index: ArrayLike,
    kinetic_energy: ArrayLike,
    degrees_of_freedom: ArrayLike,
    thermal_energy: ArrayLike,
    duration: ArrayLike,
    /,
    *,
    operator_id: int,
) -> VelocityScaleEvaluation:
    """Exact CSVR momentum scale over ``duration`` at target ``kT / 2`` per DOF.

    ``degrees_of_freedom`` may be a traced integer scalar; a nonpositive
    count is an unsuccessful evaluation.  The chi-square draw (scalar draw
    ``1``) uses ``N_f - 1`` degrees and is replaced by zero for ``N_f = 1``.
    """
    kinetic = jnp.asarray(kinetic_energy)
    dtype = kinetic.dtype
    dof = jnp.asarray(degrees_of_freedom, dtype=dtype)
    key = _thermal_key(
        key_data, step_index, operator_id=operator_id, realization_id=plan.realization_id
    )
    gaussian = jr.normal(jr.fold_in(key, jnp.uint32(0)), (), dtype=dtype)
    remainder = jnp.where(
        dof > 1.0,
        jr.chisquare(
            jr.fold_in(key, jnp.uint32(1)), jnp.maximum(dof - 1.0, 1.0), (), dtype=dtype
        ),
        0.0,
    )
    thermal = jnp.asarray(thermal_energy, dtype=dtype)
    target = 0.5 * dof * thermal
    decay = jnp.exp(
        -jnp.asarray(duration, dtype=dtype) / jnp.asarray(plan.time_constant, dtype=dtype)
    )
    ratio = target / (dof * jnp.maximum(kinetic, jnp.finfo(dtype).tiny))
    squared = (
        decay
        + (1.0 - decay) * ratio * (gaussian * gaussian + remainder)
        + 2.0 * gaussian * jnp.sqrt(decay * (1.0 - decay) * ratio)
    )
    scale = jnp.sqrt(jnp.maximum(squared, 0.0))
    successful = (
        (dof > 0.0)
        & (kinetic > 0.0)
        & jnp.isfinite(kinetic)
        & (thermal > 0.0)
        & jnp.isfinite(thermal)
        & jnp.isfinite(scale)
        & (squared >= 0.0)
    )
    return VelocityScaleEvaluation(scale, successful)


def deterministic_velocity_rescaling_scale(
    kinetic_energy: ArrayLike,
    degrees_of_freedom: ArrayLike,
    thermal_energy: ArrayLike,
    /,
) -> VelocityScaleEvaluation:
    """Noncanonical scale ``sqrt(N_f kT / (2 K))`` to the target temperature."""
    kinetic = jnp.asarray(kinetic_energy)
    thermal = jnp.asarray(thermal_energy, dtype=kinetic.dtype)
    target = 0.5 * jnp.asarray(degrees_of_freedom, dtype=kinetic.dtype) * thermal
    scale = jnp.sqrt(target / jnp.maximum(kinetic, jnp.finfo(kinetic.dtype).tiny))
    successful = (
        (kinetic > 0.0) & jnp.isfinite(kinetic) & (thermal > 0.0) & jnp.isfinite(scale)
    )
    return VelocityScaleEvaluation(scale, successful)


def stable_particle_normals(
    key_data: ArrayLike,
    particle_ids: ArrayLike,
    step_index: ArrayLike,
    /,
    *,
    operator_id: int,
    realization_id: int,
    dtype: DTypeLike,
) -> Array:
    """Draw Cartesian normals addressed by stable IDs, step, operator, and path."""

    data = jnp.asarray(key_data, dtype=jnp.uint32)
    if data.shape != (2,):
        raise ValueError("key_data must contain two uint32 words.")
    raw_ids = jnp.asarray(particle_ids)
    if raw_ids.ndim != 1:
        raise ValueError("particle_ids must be a vector.")
    if raw_ids.dtype.kind not in "iu":
        raise TypeError("particle_ids must contain integers.")
    if type(operator_id) is not int or operator_id < 0:
        raise ValueError("operator_id must be a non-negative integer.")
    if type(realization_id) is not int or realization_id < 0:
        raise ValueError("realization_id must be a non-negative integer.")
    step = jnp.asarray(step_index)
    if step.shape != () or step.dtype.kind not in "iu":
        raise TypeError("step_index must be an integer scalar.")
    ids = raw_ids.astype(jnp.int64)
    ids = eqx.error_if(
        ids,
        jnp.any(ids < 0) | jnp.any(jnp.diff(jnp.sort(ids)) == 0),
        "particle_ids must be unique and non-negative.",
    )
    data = eqx.error_if(data, step < 0, "step_index must be non-negative.")
    key = jr.wrap_key_data(data)
    key = jr.fold_in(key, jnp.asarray(realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, step.astype(jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(operator_id, dtype=jnp.uint32))
    unsigned = ids.astype(jnp.uint64)
    lower = (unsigned & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    upper = (unsigned >> jnp.uint64(32)).astype(jnp.uint32)

    def particle_key(low: Array, high: Array) -> Array:
        return jr.fold_in(jr.fold_in(key, high), low)

    keys = jax.vmap(particle_key)(lower, upper)
    return jax.vmap(lambda value: jr.normal(value, (3,), dtype=dtype))(keys)


class ThermostatEvaluation(StrictModule):
    momenta: Array
    heat: Array
    decay: Array
    successful: Array


def apply_baoab_ornstein_uhlenbeck(
    plan: BAOABLangevinPlan,
    key_data: ArrayLike,
    particle_ids: ArrayLike,
    momenta: ArrayLike,
    masses: ArrayLike,
    mobile_mask: ArrayLike,
    step_index: ArrayLike,
    temperature: ArrayLike,
    /,
    *,
    boltzmann_constant: float,
    kinetic_to_energy: float,
) -> ThermostatEvaluation:
    if not isinstance(plan, BAOABLangevinPlan):
        raise TypeError("plan must be a BAOABLangevinPlan.")
    momentum = jnp.asarray(momenta)
    mass = jnp.asarray(masses, dtype=momentum.dtype)
    mobile = jnp.asarray(mobile_mask, dtype=jnp.bool_)
    thermal = jnp.asarray(temperature, dtype=momentum.dtype)
    if momentum.shape != mass.shape + (3,) or mobile.shape != mass.shape:
        raise ValueError("Momentum, mass, and mobile masks have incompatible shapes.")
    if thermal.shape != ():
        raise ValueError("temperature must be scalar.")
    decay = jnp.exp(-jnp.asarray(plan.friction * plan.step_size, dtype=momentum.dtype))
    variance = (
        mass * (1.0 - decay * decay) * (boltzmann_constant * thermal) / kinetic_to_energy
    )
    normals = stable_particle_normals(
        key_data,
        particle_ids,
        step_index,
        operator_id=BAOAB_OPERATOR_ID,
        realization_id=plan.realization_id,
        dtype=momentum.dtype,
    )
    before = (
        0.5 * kinetic_to_energy * jnp.sum(momentum * momentum / mass[:, None], axis=-1)
    )
    proposed = decay * momentum + jnp.sqrt(jnp.maximum(variance, 0.0))[:, None] * normals
    proposed = jnp.where(mobile[:, None], proposed, 0.0)
    after = (
        0.5 * kinetic_to_energy * jnp.sum(proposed * proposed / mass[:, None], axis=-1)
    )
    heat = jnp.sum(jnp.where(mobile, after - before, 0.0))
    successful = (
        (thermal > 0.0)
        & jnp.isfinite(thermal)
        & jnp.all(jnp.isfinite(proposed))
        & jnp.isfinite(heat)
    )
    return ThermostatEvaluation(proposed, heat, decay, successful)


__all__ = [
    "BAOABLangevinPlan",
    "BAOAB_OPERATOR_ID",
    "CSVR_FIRST_OPERATOR_ID",
    "CSVR_SECOND_OPERATOR_ID",
    "MAXIMUM_NOSE_HOOVER_CHAIN_LENGTH",
    "NoseHooverChainEvaluation",
    "NoseHooverChainParameters",
    "NoseHooverChainPlan",
    "NoseHooverChainState",
    "StochasticVelocityRescalingPlan",
    "ThermostatEvaluation",
    "VelocityRescalingPlan",
    "VelocityScaleEvaluation",
    "apply_baoab_ornstein_uhlenbeck",
    "deterministic_velocity_rescaling_scale",
    "initial_nose_hoover_chain_state",
    "nose_hoover_chain_energy",
    "nose_hoover_chain_masses",
    "propagate_nose_hoover_chain",
    "stable_particle_normals",
    "stable_scalar_normals",
    "stochastic_velocity_rescaling_scale",
]

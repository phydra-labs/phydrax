#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    PreparedRigidBodySet,
    rigid_body_kick_drift_kick,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ..typing import PRNGKey
from ._sites import AtomisticInteractionSiteState
from ._units import AtomisticUnitSystem


class SplittingOperatorKind(StrEnum):
    DRIFT = "drift"
    FORCE_KICK = "force-kick"
    THERMOSTAT = "thermostat"
    POSITION_CONSTRAINT = "position-constraint"
    MOMENTUM_CONSTRAINT = "momentum-constraint"
    BAROSTAT = "barostat"
    CELL_UPDATE = "cell-update"
    CELL_THERMOSTAT = "cell-thermostat"


class AtomisticSplittingPlan(StrictModule, NonTrainableState):
    operators: tuple[SplittingOperatorKind, ...] = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: Iterable[SplittingOperatorKind],
        coefficients: Iterable[float],
        /,
    ) -> None:
        ops = tuple(operators)
        coeff = tuple(float(value) for value in coefficients)
        if (
            not ops
            or len(ops) != len(coeff)
            or any(not isinstance(value, SplittingOperatorKind) for value in ops)
            or any(not np.isfinite(value) for value in coeff)
        ):
            raise ValueError("Splitting operators and coefficients are invalid.")
        self.operators = ops
        self.coefficients = coeff
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-splitting",
                "operators": [value.value for value in ops],
                "coefficients": list(coeff),
            }
        )

    @classmethod
    def velocity_verlet(cls, /, *, constrained: bool = False) -> Self:
        operators = [SplittingOperatorKind.FORCE_KICK, SplittingOperatorKind.DRIFT]
        coefficients = [0.5, 1.0]
        if constrained:
            operators.append(SplittingOperatorKind.POSITION_CONSTRAINT)
            coefficients.append(1.0)
        operators.append(SplittingOperatorKind.FORCE_KICK)
        coefficients.append(0.5)
        if constrained:
            operators.append(SplittingOperatorKind.MOMENTUM_CONSTRAINT)
            coefficients.append(1.0)
        return cls(operators, coefficients)

    @classmethod
    def baoab(cls, /, *, constrained: bool = False) -> Self:
        operators = [
            SplittingOperatorKind.FORCE_KICK,
            SplittingOperatorKind.DRIFT,
            SplittingOperatorKind.THERMOSTAT,
            SplittingOperatorKind.DRIFT,
        ]
        coefficients = [0.5, 0.5, 1.0, 0.5]
        if constrained:
            operators.append(SplittingOperatorKind.POSITION_CONSTRAINT)
            coefficients.append(1.0)
        operators.append(SplittingOperatorKind.FORCE_KICK)
        coefficients.append(0.5)
        if constrained:
            operators.append(SplittingOperatorKind.MOMENTUM_CONSTRAINT)
            coefficients.append(1.0)
        return cls(operators, coefficients)

    @classmethod
    def symmetric_thermostat(cls, /, *, constrained: bool = False) -> Self:
        """``T(dt/2) VV T(dt/2)``: NHC, CSVR and GLE half-step thermostats.

        Both thermostat halves act on constraint-projected momenta.
        """
        inner = cls.velocity_verlet(constrained=constrained)
        return cls(
            (
                SplittingOperatorKind.THERMOSTAT,
                *inner.operators,
                SplittingOperatorKind.THERMOSTAT,
            ),
            (0.5, *inner.coefficients, 0.5),
        )

    @classmethod
    def terminal_thermostat(cls, /, *, constrained: bool = False) -> Self:
        """``VV T``: a control applied after the completed mechanical step."""
        inner = cls.velocity_verlet(constrained=constrained)
        return cls(
            (*inner.operators, SplittingOperatorKind.THERMOSTAT),
            (*inner.coefficients, 1.0),
        )

    @classmethod
    def mtk(cls, /, *, thermostatted: bool) -> Self:
        """Tuckerman et al. (2006) MTK factorization with an exact affine drift.

        NPT: ``Tc Tp B K C K B Tp Tc``; NPH: ``B K C K B`` (all halves except
        the full cell/position update ``C``).
        """
        core = (
            SplittingOperatorKind.BAROSTAT,
            SplittingOperatorKind.FORCE_KICK,
            SplittingOperatorKind.CELL_UPDATE,
            SplittingOperatorKind.FORCE_KICK,
            SplittingOperatorKind.BAROSTAT,
        )
        weights = (0.5, 0.5, 1.0, 0.5, 0.5)
        if not thermostatted:
            return cls(core, weights)
        head = (SplittingOperatorKind.CELL_THERMOSTAT, SplittingOperatorKind.THERMOSTAT)
        return cls(
            (*head, *core, *reversed(head)),
            (0.5, 0.5, *weights, 0.5, 0.5),
        )


class ThermostatResult(StrictModule):
    momenta: Array
    auxiliary: Array
    heat: Array
    successful: Array


class AbstractThermostatPlan(StrictModule, NonTrainableState):
    plan_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def apply(
        self,
        momenta: ArrayLike,
        masses: Array,
        mobile_mask: ArrayLike,
        key: PRNGKey,
        step: ArrayLike,
        dt: float | Array,
        units: AtomisticUnitSystem,
        /,
        *,
        auxiliary: ArrayLike | None = None,
    ) -> ThermostatResult:
        raise NotImplementedError


class EulerMaruyamaGeneralizedLangevinPlan(AbstractThermostatPlan):
    """Euler-Maruyama GLE kernel of continuous drift/diffusion matrices.

    Retained only for the ring-polymer normal-mode consumers (PIGLET) whose
    physical definition is this first-order discretization; atomistic dynamics
    executes GLE exclusively through the discrete FDT-checked
    ``GeneralizedLangevinPlan`` owner.  The discretization is not
    FDT-exact and is never presented as a canonical sampler.  ``temperature``
    is a dynamic ``float64`` scalar; the content digest (matrices and exact
    host temperature) is the dynamic ``plan_identity`` (host ``plan_id``).
    """

    drift_matrix: Array
    diffusion_factor: Array
    temperature: Array
    plan_identity: IdentityWords

    def __init__(
        self, drift_matrix: ArrayLike, diffusion_factor: ArrayLike, temperature: float, /
    ) -> None:
        drift_host = np.asarray(drift_matrix, dtype=np.float64)
        diffusion_host = np.asarray(diffusion_factor, dtype=np.float64)
        if (
            drift_host.ndim != 2
            or drift_host.shape[0] != drift_host.shape[1]
            or diffusion_host.shape != drift_host.shape
            or np.any(~np.isfinite(drift_host))
            or np.any(~np.isfinite(diffusion_host))
            or float(temperature) <= 0.0
        ):
            raise ValueError("GLE matrices or temperature are invalid.")
        drift, diffusion = jnp.asarray(drift_host), jnp.asarray(diffusion_host)
        thermal = float(temperature)
        self.drift_matrix, self.diffusion_factor = drift, diffusion
        self.temperature = jnp.asarray(thermal, dtype=jnp.float64)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "generalized-langevin",
                "drift": drift_host.tolist(),
                "diffusion": diffusion_host.tolist(),
                "temperature": thermal,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    def apply(
        self,
        momenta: ArrayLike,
        masses: Array,
        mobile_mask: ArrayLike,
        key: PRNGKey,
        step: ArrayLike,
        dt: float | Array,
        units: AtomisticUnitSystem,
        /,
        *,
        auxiliary: ArrayLike | None = None,
    ) -> ThermostatResult:
        p = jnp.asarray(momenta)
        auxiliary_count = self.drift_matrix.shape[0] - 1
        expected = p.shape + (auxiliary_count,)
        memory = (
            jnp.zeros(expected, dtype=p.dtype)
            if auxiliary is None
            else jnp.asarray(auxiliary, dtype=p.dtype)
        )
        if memory.shape != expected:
            raise ValueError("GLE auxiliary state has the wrong shape.")
        mass = jnp.asarray(masses, dtype=p.dtype)
        if mass.shape != p.shape[:1]:
            raise ValueError("GLE masses must align with particles.")
        momentum_scale = jnp.sqrt(
            mass[:, None]
            * units.boltzmann_constant
            * jnp.asarray(self.temperature, dtype=p.dtype)
            / units.kinetic_to_energy
        )
        normalized_momenta = p / momentum_scale
        state = jnp.concatenate((normalized_momenta[..., None], memory), axis=-1)
        noise = jr.normal(
            jr.fold_in(key, jnp.asarray(step, dtype=jnp.uint32)),
            state.shape,
        )
        update = (
            state
            - dt * contract("...i,ji->...j", state, self.drift_matrix)
            + jnp.sqrt(dt) * contract("...i,ji->...j", noise, self.diffusion_factor)
        )
        mobile = jnp.asarray(mobile_mask)
        result = jnp.where(mobile[:, None], update[..., 0] * momentum_scale, 0.0)
        memory = jnp.where(mobile[:, None, None], update[..., 1:], 0.0)
        heat = 0.5 * units.kinetic_to_energy * jnp.sum((result**2 - p**2) / mass[:, None])
        successful = (
            jnp.all(jnp.isfinite(update))
            & jnp.all(jnp.isfinite(momentum_scale))
            & jnp.all(momentum_scale > 0.0)
        )
        return ThermostatResult(result, memory, heat, successful)


class NoisyForceLangevinPlan(AbstractThermostatPlan):
    """Langevin thermostat with an additional force-noise variance.

    ``force_noise_variance`` is a dynamic ``float64`` scalar; the digest is
    the dynamic ``plan_identity`` (host ``plan_id``, unchanged payload).
    """

    base: EulerMaruyamaGeneralizedLangevinPlan
    force_noise_variance: Array
    plan_identity: IdentityWords

    def __init__(
        self, friction: float, force_noise_variance: float, temperature: float, /
    ) -> None:
        if min(friction, force_noise_variance, temperature) <= 0:
            raise ValueError("Noisy-force Langevin parameters must be positive.")
        base = EulerMaruyamaGeneralizedLangevinPlan(
            jnp.asarray([[friction]]),
            jnp.asarray([[jnp.sqrt(2.0 * friction + force_noise_variance)]]),
            temperature,
        )
        variance = float(force_noise_variance)
        self.base = base
        self.force_noise_variance = jnp.asarray(variance, dtype=jnp.float64)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "noisy-force-langevin",
                "base": base.plan_id,
                "force_noise_variance": variance,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    def apply(
        self,
        momenta: ArrayLike,
        masses: Array,
        mobile_mask: ArrayLike,
        key: PRNGKey,
        step: ArrayLike,
        dt: float | Array,
        units: AtomisticUnitSystem,
        /,
        *,
        auxiliary: ArrayLike | None = None,
    ) -> ThermostatResult:
        return self.base.apply(
            momenta,
            masses,
            mobile_mask,
            key,
            step,
            dt,
            units,
            auxiliary=auxiliary,
        )


class AnisotropicPressurePlan(StrictModule, NonTrainableState):
    """Deterministic pressure-feedback cell update: mechanical control only.

    A first-order weak-coupling strain ``dt * compressibility * (P - P_target)``
    used for mechanical relaxation/equilibration.  It has no barostat momentum,
    measure or work contract and never establishes an NPT ensemble.
    """

    target_pressure: Array
    compressibility: Array
    semi_isotropic: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_pressure: ArrayLike,
        compressibility: ArrayLike,
        /,
        *,
        semi_isotropic: bool = False,
    ) -> None:
        pressure_host = np.asarray(target_pressure, dtype=np.float64)
        compress_host = np.asarray(compressibility, dtype=np.float64)
        if (
            pressure_host.shape not in ((), (3,), (3, 3))
            or compress_host.shape not in ((), (3,), (3, 3))
            or np.any(~np.isfinite(pressure_host))
            or np.any(~np.isfinite(compress_host) | (compress_host <= 0.0))
            or (
                pressure_host.ndim == 2
                and not np.allclose(pressure_host, pressure_host.T)
            )
            or (
                compress_host.ndim == 2
                and not np.allclose(compress_host, compress_host.T)
            )
        ):
            raise ValueError("Pressure target or compressibility is invalid.")
        pressure = jnp.asarray(pressure_host)
        compress = jnp.asarray(compress_host)
        self.target_pressure, self.compressibility, self.semi_isotropic = (
            pressure,
            compress,
            bool(semi_isotropic),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "anisotropic-pressure",
                "pressure_shape": list(pressure.shape),
                "compressibility_shape": list(compress.shape),
                "semi_isotropic": self.semi_isotropic,
            }
        )

    def update_cell(
        self, cell_vectors: ArrayLike, observed_pressure: ArrayLike, dt: ArrayLike, /
    ) -> Array:
        cell = jnp.asarray(cell_vectors)
        observed_value = jnp.asarray(observed_pressure, dtype=cell.dtype)
        if cell.shape != (3, 3) or observed_value.shape not in ((), (3,), (3, 3)):
            raise ValueError(
                "Pressure update requires cell and scalar/vector/tensor pressure."
            )
        target = (
            jnp.eye(3, dtype=cell.dtype) * self.target_pressure
            if self.target_pressure.ndim == 0
            else jnp.diag(self.target_pressure)
            if self.target_pressure.ndim == 1
            else self.target_pressure
        )
        observed = (
            jnp.eye(3, dtype=cell.dtype) * observed_value
            if observed_value.ndim == 0
            else jnp.diag(observed_value)
            if observed_value.ndim == 1
            else observed_value
        )
        compress = (
            jnp.eye(3, dtype=cell.dtype) * self.compressibility
            if self.compressibility.ndim == 0
            else jnp.diag(self.compressibility)
            if self.compressibility.ndim == 1
            else self.compressibility
        )
        strain = jnp.asarray(dt) * compress * (observed - target)
        strain = 0.5 * (strain + strain.T)
        if self.semi_isotropic:
            lateral = 0.5 * (strain[0, 0] + strain[1, 1])
            strain = jnp.diag(jnp.asarray([lateral, lateral, strain[2, 2]]))
        candidate = cell @ (jnp.eye(3, dtype=cell.dtype) + strain).T
        determinant = jnp.sum(candidate[0] * jnp.cross(candidate[1], candidate[2]))
        successful = (
            (jnp.asarray(dt) > 0.0) & jnp.isfinite(determinant) & (determinant > 0.0)
        )
        return jnp.where(successful, candidate, jnp.nan)


class RigidAtomisticCoordinateMap(StrictModule, NonTrainableState):
    """Rigid-body site realization; ``map_identity`` (host ``map_id``) is dynamic.

    The digest binds the prepared bodies, so it is carried as dynamic words
    and becomes the realized site state's ``coordinate_map_identity``.
    """

    bodies: PreparedRigidBodySet
    body_indices: Array
    local_positions: Array
    site_ids: Array
    map_identity: IdentityWords

    def __init__(
        self,
        bodies: PreparedRigidBodySet,
        body_indices: ArrayLike,
        local_positions: ArrayLike,
        site_ids: ArrayLike,
        /,
    ) -> None:
        index, local, ids = (
            jnp.asarray(body_indices, dtype=jnp.int32),
            jnp.asarray(local_positions),
            jnp.asarray(site_ids, dtype=jnp.int64),
        )
        if index.ndim != 1 or local.shape != (index.size, 3) or ids.shape != index.shape:
            raise ValueError("Rigid atomistic site routes are invalid.")
        self.bodies, self.body_indices, self.local_positions, self.site_ids = (
            bodies,
            index,
            local,
            ids,
        )
        self.map_identity = IdentityWords.of(
            {
                "kind": "rigid-atomistic-map",
                "bodies": bodies.prepared_id,
                "count": index.size,
            }
        )

    @property
    def map_id(self) -> str:
        """Host decoding of the canonical map digest."""
        return self.map_identity.fingerprint()

    def realize(
        self, kinematics: RigidBodyKinematics, /
    ) -> AtomisticInteractionSiteState:
        quaternion = kinematics.orientation[self.body_indices]
        w, vector = quaternion[:, :1], quaternion[:, 1:]
        local = self.local_positions
        rotated = local + 2.0 * jnp.cross(vector, jnp.cross(vector, local) + w * local)
        positions = kinematics.position[self.body_indices] + rotated
        valid = jnp.all(jnp.isfinite(positions))
        mask = jnp.ones((positions.shape[0],), dtype=jnp.bool_)
        return AtomisticInteractionSiteState(
            positions, mask, mask, mask, jnp.asarray(jnp.inf), valid, self.map_identity
        )


class RotationalAtomisticState(StrictModule):
    kinematics: RigidBodyKinematics
    force: Array
    torque: Array
    successful: Array


def rotational_velocity_verlet(
    bodies: PreparedRigidBodySet,
    state: RotationalAtomisticState,
    step_size: ArrayLike,
    load_function: Callable[[Array, RigidBodyKinematics, Any], RigidBodyLoad],
    /,
) -> RotationalAtomisticState:
    result = rigid_body_kick_drift_kick(
        bodies,
        state.kinematics,
        RigidBodyLoad(state.force, state.torque),
        jnp.zeros((), dtype=jnp.asarray(step_size).dtype),
        jnp.asarray(step_size),
        load_function,
    )
    return RotationalAtomisticState(
        result.kinematics, result.load.force, result.load.torque, result.successful
    )


class BrownianDynamicsPlan(StrictModule, NonTrainableState):
    """Euler-Maruyama Brownian step with dynamic ``mobility``/``temperature``.

    Both are ``float64`` scalars; the digest of their exact host values is the
    dynamic ``plan_identity`` (host ``plan_id``, unchanged payload).
    """

    mobility: Array
    temperature: Array
    plan_identity: IdentityWords

    def __init__(self, mobility: float, temperature: float, /) -> None:
        mobility_, thermal = float(mobility), float(temperature)
        if min(mobility_, thermal) <= 0:
            raise ValueError("Brownian mobility and temperature must be positive.")
        self.mobility = jnp.asarray(mobility_, dtype=jnp.float64)
        self.temperature = jnp.asarray(thermal, dtype=jnp.float64)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "brownian-dynamics",
                "mobility": mobility_,
                "temperature": thermal,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical plan digest."""
        return self.plan_identity.fingerprint()

    def step(
        self,
        positions: Array,
        forces: Array,
        dt: float | Array,
        key: PRNGKey,
        units: AtomisticUnitSystem,
        /,
    ) -> Array:
        dtype = jnp.result_type(positions)
        noise = jr.normal(key, jnp.asarray(positions).shape)
        diffusion = jnp.asarray(
            self.mobility * units.boltzmann_constant * self.temperature, dtype=dtype
        )
        mobility = jnp.asarray(self.mobility, dtype=dtype)
        return positions + dt * mobility * forces + jnp.sqrt(2.0 * diffusion * dt) * noise


__all__ = [
    "AbstractThermostatPlan",
    "AnisotropicPressurePlan",
    "AtomisticSplittingPlan",
    "BrownianDynamicsPlan",
    "EulerMaruyamaGeneralizedLangevinPlan",
    "NoisyForceLangevinPlan",
    "RigidAtomisticCoordinateMap",
    "RotationalAtomisticState",
    "SplittingOperatorKind",
    "ThermostatResult",
    "rotational_velocity_verlet",
]

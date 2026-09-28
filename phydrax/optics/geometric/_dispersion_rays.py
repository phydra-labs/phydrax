#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hamiltonian rays of an arbitrary dispersion relation, including cold plasmas.

A ray of a stationary medium at fixed angular frequency ``ω`` is a trajectory of
the canonical system ``dx/dτ = ∂H/∂p``, ``dp/dτ = −∂H/∂x`` on the level set
``H(x, p) = 0``, where ``p = c k / ω`` is the refractive-index vector. Any
Hamiltonian ``g(x, p) D(x, p)`` with ``g ≠ 0`` on the dispersion surface
``D = 0`` traces the same rays with a different parameter ``τ``; the
Hamiltonian owns that normalization.

Two symplectic integrators are provided. ``"implicit-midpoint"`` applies to any
Hamiltonian: each step solves ``m = z + (h/2) J ∇H(m)`` with
`phydrax.nonlinear.VectorLocalRootPlan` and sets ``z' = 2m − z``, so the step map
is the Cayley transform of ``h J ∇²H(m)`` and exactly symplectic at the solved
midpoint. ``"kick-drift-kick"`` (Störmer–Verlet) is the explicit specialization
for separable Hamiltonians ``½|p|² + V(x)`` such as graded-index optics.

`ColdPlasmaHamiltonian` is the magnetized cold-plasma specialization: the Stix
biquadratic ``D = A n⁴ − B n² + C`` in Cartesian form, normalized by
``G = p·∂D/∂p − ω ∂D/∂ω|_p = −ω ∂D/∂ω|_k`` so that ``τ = c t`` is the group
light path and ``|dx/dτ| = v_g / c``. The mode is fixed at launch by the labeled
`ColdPlasmaWaveResult` root and preserved by continuity of the quartic's branch.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from enum import IntFlag
from math import isfinite
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import ElectromagneticScaleContract, SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...electromagnetics._cold_plasma import (
    _cold_plasma_waves,
    _dispersion_polynomial,
    _faraday_coefficients,
    _quartic_coefficients,
    _species_frequencies,
    _species_stix,
    ColdPlasmaDielectric,
    ColdPlasmaWaveStatus,
    PlasmaWaveMode,
    StixParameters,
)
from ...linalg import DenseLinearOperator, LinearSystem, solve
from ...nonlinear import VectorLocalRootPlan
from ...typing import (
    as_host_array,
    Bool,
    ConvertibleToArray,
    Dim,
    Float,
    Float64,
    HostFloat64,
    Identifier,
    Int32,
    parse,
    Scalar,
    Scope,
    Size,
)


DispersionRayMethod: TypeAlias = Literal["kick-drift-kick", "implicit-midpoint"]


class _RayDim(Dim, minimum=1):
    """Independent rays."""


class _NodeDim(Dim, minimum=2):
    """Ray-parameter nodes, the launch point included."""


class _SegmentDim(Dim, minimum=1):
    """Ray segments between consecutive nodes."""


class _SpeciesDim(Dim, minimum=1):
    """Charged species of a cold-plasma profile."""


class DispersionRayStatus(IntFlag):
    """Per-ray evidence of `DispersionRayEvidence.status`.

    Every flag except ``CUTOFF`` makes the ray unsuccessful. ``CUTOFF`` records
    that ``|p|`` fell below the plan's ``cutoff_index`` (a turning point near a
    cutoff, where the geometric-optics amplitude needs an Airy connection).
    """

    NONE = 0
    LAUNCH_INVALID = 1
    UNSUPPORTED = 2
    NONFINITE = 4
    ROOT_UNCONVERGED = 8
    HAMILTONIAN_DRIFT = 16
    SYMPLECTIC_RESIDUAL = 32
    RESONANCE = 64
    CUTOFF = 128


_RAY_FAILURE = (
    DispersionRayStatus.LAUNCH_INVALID
    | DispersionRayStatus.UNSUPPORTED
    | DispersionRayStatus.NONFINITE
    | DispersionRayStatus.ROOT_UNCONVERGED
    | DispersionRayStatus.HAMILTONIAN_DRIFT
    | DispersionRayStatus.SYMPLECTIC_RESIDUAL
    | DispersionRayStatus.RESONANCE
).value


class AbstractDispersionHamiltonian(StrictModule):
    """Ray Hamiltonian ``H(x, p)`` of one medium at one angular frequency.

    Methods act on one ray: ``position`` and ``momentum`` are three-vectors and
    ``momentum`` is the refractive-index vector ``p = c k / ω``.
    """

    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)

    @abstractmethod
    def value(self, position: Array, momentum: Array, /) -> Array:
        """Scalar ``H``; rays live on ``H = 0``."""
        raise NotImplementedError

    @abstractmethod
    def supported(self, position: Array, /) -> Array:
        """Whether the medium is defined and admissible at ``position``."""
        raise NotImplementedError

    @abstractmethod
    def launch(self, position: Array, direction: Array, /) -> tuple[Array, Array]:
        """Momentum on ``H = 0`` along the unit wave normal and its validity."""
        raise NotImplementedError


class AbstractSeparableDispersionHamiltonian(AbstractDispersionHamiltonian):
    """Separable Hamiltonian ``½|p|² + V(x)`` admitting kick–drift–kick."""

    @abstractmethod
    def potential(self, position: Array, /) -> tuple[Array, Array, Array]:
        """``V``, ``∇V`` and support validity at ``position``."""
        raise NotImplementedError

    def value(self, position: Array, momentum: Array, /) -> Array:
        potential, _, _ = self.potential(position)
        return 0.5 * jnp.sum(momentum * momentum) + potential


class DispersionRayState(StrictModule):
    """Final canonical state, tangent map and path lengths of each ray.

    ``geometric_lengths`` is the trapezoidal ``∫ |∂H/∂p| dτ`` and
    ``optical_lengths`` the phase path ``∫ p · ∂H/∂p dτ``.
    """

    __strict_contract__ = True

    positions: Float[_RayDim, Literal[3]]
    momenta: Float[_RayDim, Literal[3]]
    tangent_maps: Float[_RayDim, Literal[6], Literal[6]]
    geometric_lengths: Float[_RayDim]
    optical_lengths: Float[_RayDim]
    valid: Bool[_RayDim]


class DispersionRayEvidence(StrictModule, NonTrainableState):
    """Integration evidence; ``status`` holds `DispersionRayStatus` bits per ray."""

    __strict_contract__ = True

    finite: Bool[Scalar]
    medium_covered: Bool[Scalar]
    maximum_hamiltonian_drift: Float[Scalar]
    maximum_symplectic_residual: Float[Scalar]
    root_converged: Bool[Scalar]
    maximum_root_residual: Float[Scalar]
    minimum_refractive_index: Float[_RayDim]
    maximum_refractive_index: Float[_RayDim]
    status: Int32[_RayDim]
    successful: Bool[Scalar]
    plan_id: str = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)


class DispersionRayResult(StrictModule):
    """Ray histories on the ``step_count + 1`` parameter nodes.

    ``initial_directions`` and ``final_directions`` are unit wave normals
    ``p/|p|``; in an isotropic medium they are the ray tangents.
    """

    __strict_contract__ = True

    state: DispersionRayState
    position_history: Float[_NodeDim, _RayDim, Literal[3]]
    momentum_history: Float[_NodeDim, _RayDim, Literal[3]]
    tangent_history: Float[_NodeDim, _RayDim, Literal[6], Literal[6]]
    initial_directions: Float[_RayDim, Literal[3]]
    final_directions: Float[_RayDim, Literal[3]]
    evidence: DispersionRayEvidence


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


class DispersionRayPlan(StrictModule, NonTrainableState):
    """Fixed-step symplectic ray integration of a dispersion Hamiltonian.

    ``step_size`` is the increment of the Hamiltonian's ray parameter ``τ``.
    ``hamiltonian_tolerance`` bounds ``|H − H₀|``; ``None`` demands roundoff
    conservation ``2·10³ ε step_count``, which a separable schedule meets on
    piecewise-quadratic potentials and which a nonquadratic Hamiltonian
    generally violates at ``O(h²)``. ``root_tolerance`` (on the midpoint
    residual scaled componentwise by ``1 + |z| + |(h/2) J∇H(z)|``) and
    ``root_steps`` configure the implicit-midpoint Newton solve. Rays whose ``|p|`` exceeds
    ``resonance_index`` are refused as resonant; rays whose ``|p|`` falls below
    ``cutoff_index`` carry the qualifying ``CUTOFF`` flag.
    """

    hamiltonian: AbstractDispersionHamiltonian
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    method: DispersionRayMethod = eqx.field(static=True)
    hamiltonian_tolerance: float | None = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    root_steps: int = eqx.field(static=True)
    cutoff_index: float = eqx.field(static=True)
    resonance_index: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: AbstractDispersionHamiltonian,
        step_size: float,
        step_count: int,
        /,
        *,
        method: DispersionRayMethod = "implicit-midpoint",
        hamiltonian_tolerance: float | None = None,
        root_tolerance: float = 1.0e-11,
        root_steps: int = 12,
        cutoff_index: float = 1.0e-2,
        resonance_index: float = 1.0e2,
    ) -> None:
        if not isinstance(hamiltonian, AbstractDispersionHamiltonian):
            raise TypeError("hamiltonian must be an AbstractDispersionHamiltonian.")
        method_ = parse(method, DispersionRayMethod, "method")
        if method_ == "kick-drift-kick" and not isinstance(
            hamiltonian, AbstractSeparableDispersionHamiltonian
        ):
            raise ValueError(
                "kick-drift-kick requires an AbstractSeparableDispersionHamiltonian."
            )
        size = _positive(step_size, "step_size")
        if not isinstance(step_count, int) or step_count < 1:
            raise ValueError("step_count must be a positive integer.")
        tolerance = (
            None
            if hamiltonian_tolerance is None
            else _positive(hamiltonian_tolerance, "hamiltonian_tolerance")
        )
        root = _positive(root_tolerance, "root_tolerance")
        if not isinstance(root_steps, int) or root_steps < 1:
            raise ValueError("root_steps must be a positive integer.")
        cutoff = _positive(cutoff_index, "cutoff_index")
        resonance = _positive(resonance_index, "resonance_index")
        if resonance <= cutoff:
            raise ValueError("resonance_index must exceed cutoff_index.")
        self.hamiltonian = hamiltonian
        self.step_size = size
        self.step_count = step_count
        self.method = method_
        self.hamiltonian_tolerance = tolerance
        self.root_tolerance = root
        self.root_steps = root_steps
        self.cutoff_index = cutoff
        self.resonance_index = resonance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dispersion-ray-plan",
                "hamiltonian": hamiltonian.hamiltonian_id,
                "step_size": size,
                "step_count": step_count,
                "method": method_,
                "hamiltonian_tolerance": tolerance,
                "root_tolerance": root,
                "root_steps": root_steps,
                "cutoff_index": cutoff,
                "resonance_index": resonance,
            }
        )

    def prepare(self) -> PreparedDispersionRay:
        return PreparedDispersionRay(self)


type _RayCarry = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]


class PreparedDispersionRay(StrictModule):
    """Prepared integrator: `DispersionRayPlan` plus the midpoint root plan."""

    plan: DispersionRayPlan
    root: VectorLocalRootPlan

    def __init__(self, plan: DispersionRayPlan, /) -> None:
        if not isinstance(plan, DispersionRayPlan):
            raise TypeError("plan must be a DispersionRayPlan.")
        self.plan = plan
        self.root = VectorLocalRootPlan(
            6,
            maximum_steps=plan.root_steps,
            tolerance=plan.root_tolerance,
            plan_id=f"{plan.plan_id}:implicit-midpoint",
        )

    @property
    def hamiltonian(self) -> AbstractDispersionHamiltonian:
        return self.plan.hamiltonian

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id

    def _velocity(self, position: Array, momentum: Array, /) -> Array:
        match self.plan.method:
            case "kick-drift-kick":
                return momentum
            case "implicit-midpoint":
                return jax.grad(self.hamiltonian.value, argnums=1)(position, momentum)
            case _:
                assert_never(self.plan.method)

    def _flow(self, vector: Array, /) -> Array:
        gradient = jax.grad(lambda state: self.hamiltonian.value(state[:3], state[3:]))(
            vector
        )
        return jnp.concatenate((gradient[3:], -gradient[:3]))

    def _kick_drift_kick(self, vector: Array, /) -> Array:
        hamiltonian = self.hamiltonian
        if not isinstance(hamiltonian, AbstractSeparableDispersionHamiltonian):
            raise TypeError("kick-drift-kick requires a separable Hamiltonian.")
        step = jnp.asarray(self.plan.step_size, dtype=vector.dtype)
        x, p = vector[:3], vector[3:]
        _, gradient, _ = hamiltonian.potential(x)
        p_half = p - 0.5 * step * gradient
        x_new = x + step * p_half
        _, gradient_new, _ = hamiltonian.potential(x_new)
        p_new = p_half - 0.5 * step * gradient_new
        return jnp.concatenate((x_new, p_new))

    def _step(self, vector: Array, /) -> tuple[Array, Array, Array, Array]:
        """Next state, step Jacobian, root convergence and root residual norm."""
        match self.plan.method:
            case "kick-drift-kick":
                return (
                    self._kick_drift_kick(vector),
                    jax.jacfwd(self._kick_drift_kick)(vector),
                    jnp.asarray(True),
                    jnp.zeros((), dtype=vector.dtype),
                )
            case "implicit-midpoint":
                half = 0.5 * jnp.asarray(self.plan.step_size, dtype=vector.dtype)
                # The componentwise scale 1 + |z| + |(h/2) f(z)| makes the root
                # tolerance relative to the state and the step without moving the root.
                predictor = half * self._flow(vector)
                scale = 1.0 + jnp.abs(vector) + jnp.abs(predictor)

                def residual(midpoint: Array) -> Array:
                    return (midpoint - vector - half * self._flow(midpoint)) / scale

                midpoint, diagnostics = self.root.solve_with_diagnostics(
                    residual, vector + predictor
                )
                identity = jnp.eye(6, dtype=vector.dtype)
                # z' = 2m − z with (I − (h/2) J ∇²H(m)) dm = dz, so the step map is
                # the Cayley transform 2 (∂r/∂m)⁻¹ − I; the scaled Jacobian is
                # diag(1/s) ∂r/∂m.
                inverse = solve(
                    LinearSystem(DenseLinearOperator(diagnostics.jacobian)),
                    jnp.diag(1.0 / scale),
                )
                return (
                    2.0 * midpoint - vector,
                    2.0 * jnp.asarray(inverse.value) - identity,
                    diagnostics.converged,
                    diagnostics.residual_norm,
                )
            case _:
                assert_never(self.plan.method)

    def integrate(
        self, positions: ArrayLike, directions: ArrayLike, /
    ) -> DispersionRayResult:
        """Trace rays launched at ``positions`` along wave normals ``directions``."""
        positions_ = jnp.asarray(positions)
        directions_ = jnp.asarray(directions, dtype=positions_.dtype)
        if (
            positions_.ndim != 2
            or positions_.shape[1] != 3
            or directions_.shape != positions_.shape
        ):
            raise ValueError("positions and directions must share shape (ray_count, 3).")
        hamiltonian = self.hamiltonian
        direction_norm = jnp.sqrt(jnp.sum(directions_ * directions_, axis=-1))
        direction_valid = direction_norm > jnp.finfo(positions_.dtype).tiny
        unit = directions_ / jnp.where(
            direction_valid[:, None], direction_norm[:, None], 1.0
        )
        momenta, launch_valid = jax.vmap(hamiltonian.launch)(positions_, unit)
        momenta = momenta.astype(positions_.dtype)
        launched = direction_valid & launch_valid
        valid = launched & jax.vmap(hamiltonian.supported)(positions_)
        h0 = jax.vmap(hamiltonian.value)(positions_, momenta)
        velocity0 = jax.vmap(self._velocity)(positions_, momenta)
        tangent = jnp.broadcast_to(
            jnp.eye(6, dtype=positions_.dtype), (positions_.shape[0], 6, 6)
        )
        index0 = jnp.sqrt(jnp.sum(momenta * momenta, axis=-1))
        zeros = jnp.zeros_like(index0)
        step = jnp.asarray(self.plan.step_size, dtype=positions_.dtype)

        def advance(
            carry: _RayCarry, _x: None
        ) -> tuple[_RayCarry, tuple[Array, Array, Array]]:
            (
                x,
                p,
                velocity,
                mapping,
                length,
                optical_path,
                active,
                maximum_h,
                converged,
                root_residual,
                minimum_index,
                maximum_index,
            ) = carry
            vector = jnp.concatenate((x, p), axis=-1)
            next_vector, jacobian, step_converged, step_residual = jax.vmap(self._step)(
                vector
            )
            next_mapping = jacobian @ mapping
            x_new, p_new = next_vector[:, :3], next_vector[:, 3:]
            velocity_new = jax.vmap(self._velocity)(x_new, p_new)
            speed_old = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
            speed_new = jnp.sqrt(jnp.sum(velocity_new * velocity_new, axis=-1))
            next_length = length + 0.5 * step * (speed_old + speed_new)
            next_optical = optical_path + 0.5 * step * (
                jnp.sum(p * velocity, axis=-1) + jnp.sum(p_new * velocity_new, axis=-1)
            )
            hamiltonian_value = jax.vmap(hamiltonian.value)(x_new, p_new)
            index = jnp.sqrt(jnp.sum(p_new * p_new, axis=-1))
            next_active = active & jax.vmap(hamiltonian.supported)(x_new)
            return (
                x_new,
                p_new,
                velocity_new,
                next_mapping,
                next_length,
                next_optical,
                next_active,
                jnp.maximum(maximum_h, jnp.abs(hamiltonian_value - h0)),
                converged & step_converged,
                jnp.maximum(root_residual, step_residual),
                jnp.minimum(minimum_index, index),
                jnp.maximum(maximum_index, index),
            ), (x_new, p_new, next_mapping)

        initial = (
            positions_,
            momenta,
            velocity0,
            tangent,
            zeros,
            zeros,
            valid,
            zeros,
            jnp.ones_like(valid),
            zeros,
            index0,
            index0,
        )
        final, history = jax.lax.scan(
            advance, initial, xs=None, length=self.plan.step_count
        )
        (
            x_final,
            p_final,
            _,
            map_final,
            geometric,
            optical,
            active,
            max_h,
            converged,
            root_residual,
            minimum_index,
            maximum_index,
        ) = final
        return self._result(
            positions_,
            momenta,
            unit,
            tangent,
            history,
            (x_final, p_final, map_final, geometric, optical),
            (
                launched,
                active,
                max_h,
                converged,
                root_residual,
                minimum_index,
                maximum_index,
            ),
        )

    def _result(
        self,
        positions: Array,
        momenta: Array,
        unit: Array,
        tangent: Array,
        history: tuple[Array, Array, Array],
        final: tuple[Array, Array, Array, Array, Array],
        tracked: tuple[Array, Array, Array, Array, Array, Array, Array],
        /,
    ) -> DispersionRayResult:
        x_final, p_final, map_final, geometric, optical = final
        (
            launched,
            active,
            max_h,
            converged,
            root_residual,
            minimum_index,
            maximum_index,
        ) = tracked
        dtype = positions.dtype
        final_norm = jnp.sqrt(jnp.sum(p_final * p_final, axis=-1))
        final_direction = p_final / jnp.where(
            final_norm[:, None] > 0.0, final_norm[:, None], 1.0
        )
        symplectic = jnp.block(
            [[jnp.zeros((3, 3)), jnp.eye(3)], [-jnp.eye(3), jnp.zeros((3, 3))]]
        ).astype(dtype)
        residual = jnp.max(
            jnp.abs(
                jnp.swapaxes(map_final, -1, -2) @ symplectic @ map_final
                - symplectic[None]
            ),
            axis=(-2, -1),
        )
        map_scale = jnp.maximum(1.0, jnp.max(jnp.abs(map_final), axis=(-2, -1)))
        finite = (
            jnp.all(jnp.isfinite(x_final), axis=-1)
            & jnp.all(jnp.isfinite(p_final), axis=-1)
            & jnp.all(jnp.isfinite(map_final), axis=(-2, -1))
            & jnp.isfinite(geometric)
            & jnp.isfinite(optical)
        )
        roundoff = 2.0e3 * jnp.finfo(dtype).eps * self.plan.step_count
        drift_tolerance = (
            roundoff
            if self.plan.hamiltonian_tolerance is None
            else jnp.asarray(self.plan.hamiltonian_tolerance, dtype=dtype)
        )
        status = DispersionRayStatus
        root_ok = converged & (root_residual <= self.plan.root_tolerance)
        flags = (
            (~launched).astype(jnp.int32) * status.LAUNCH_INVALID.value
            | (launched & ~active).astype(jnp.int32) * status.UNSUPPORTED.value
            | (~finite).astype(jnp.int32) * status.NONFINITE.value
            | (~root_ok).astype(jnp.int32) * status.ROOT_UNCONVERGED.value
            | (~(max_h <= drift_tolerance)).astype(jnp.int32)
            * status.HAMILTONIAN_DRIFT.value
            | (~(residual <= 10.0 * roundoff * map_scale * map_scale)).astype(jnp.int32)
            * status.SYMPLECTIC_RESIDUAL.value
            | (~(maximum_index <= self.plan.resonance_index)).astype(jnp.int32)
            * status.RESONANCE.value
            | (minimum_index < self.plan.cutoff_index).astype(jnp.int32)
            * status.CUTOFF.value
        )
        evidence = DispersionRayEvidence(
            finite=jnp.all(finite),
            medium_covered=jnp.all(active),
            maximum_hamiltonian_drift=jnp.max(max_h),
            maximum_symplectic_residual=jnp.max(residual),
            root_converged=jnp.all(root_ok),
            maximum_root_residual=jnp.max(root_residual),
            minimum_refractive_index=minimum_index,
            maximum_refractive_index=maximum_index,
            status=flags,
            successful=jnp.all((flags & _RAY_FAILURE) == 0),
            plan_id=self.plan.plan_id,
            hamiltonian_id=self.hamiltonian.hamiltonian_id,
        )
        state = DispersionRayState(
            positions=x_final,
            momenta=p_final,
            tangent_maps=map_final,
            geometric_lengths=geometric,
            optical_lengths=optical,
            valid=active & finite,
        )
        return DispersionRayResult(
            state=state,
            position_history=jnp.concatenate((positions[None], history[0]), axis=0),
            momentum_history=jnp.concatenate((momenta[None], history[1]), axis=0),
            tangent_history=jnp.concatenate((tangent[None], history[2]), axis=0),
            initial_directions=unit,
            final_directions=final_direction,
            evidence=evidence,
        )


class RayFanResult(StrictModule, NonTrainableState):
    determinant_history: Array
    minimum_singular_value: Array
    caustic_crossings: Array
    caustic_detected: Array
    successful: Array


class RayFanPlan(StrictModule, NonTrainableState):
    """Transverse position–momentum determinants of ray tangent maps (caustics)."""

    transverse_basis: Array
    determinant_tolerance: float = eqx.field(static=True)

    def __init__(
        self, transverse_basis: ArrayLike, determinant_tolerance: float = 1.0e-8
    ) -> None:
        basis = np.array(transverse_basis, dtype=np.float64, copy=True)
        tolerance = float(determinant_tolerance)
        if basis.shape != (2, 3) or not np.allclose(
            basis @ basis.T, np.eye(2), atol=1.0e-10, rtol=0.0
        ):
            raise ValueError("transverse_basis must be an orthonormal (2, 3) basis.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("determinant_tolerance must be positive.")
        self.transverse_basis = jnp.asarray(basis)
        self.determinant_tolerance = tolerance

    def evaluate(self, rays: DispersionRayResult, /) -> RayFanResult:
        basis = jnp.asarray(self.transverse_basis, dtype=rays.tangent_history.dtype)
        position_momentum = rays.tangent_history[..., :3, 3:]
        transverse = basis @ position_momentum @ basis.T
        determinant = (
            transverse[..., 0, 0] * transverse[..., 1, 1]
            - transverse[..., 0, 1] * transverse[..., 1, 0]
        )
        frobenius_squared = jnp.sum(transverse * transverse, axis=(-2, -1))
        discriminant = jnp.maximum(
            frobenius_squared * frobenius_squared - 4.0 * determinant * determinant,
            0.0,
        )
        minimum_eigenvalue = 0.5 * (frobenius_squared - jnp.sqrt(discriminant))
        singular = jnp.sqrt(jnp.maximum(minimum_eigenvalue, 0.0))
        near = jnp.abs(determinant) <= self.determinant_tolerance
        crossings = jnp.sum((determinant[1:] * determinant[:-1] < 0.0) | near[1:], axis=0)
        detected = crossings > 0
        finite = jnp.all(jnp.isfinite(determinant)) & jnp.all(jnp.isfinite(singular))
        return RayFanResult(
            determinant,
            jnp.min(singular, axis=0),
            crossings,
            detected,
            finite & rays.evidence.finite,
        )


# ---------------------------------------------------------------------------
# Magnetized cold plasma
# ---------------------------------------------------------------------------


class ColdPlasmaProfile(StrictModule, NonTrainableState):
    """Collisionless multi-species cold plasma with analytic spatial profiles.

    ``density(x)`` returns the species number densities (per cubic length unit
    of ``scale``) and ``magnetic_field(x)`` the field ``B₀`` (scale field unit)
    at one point; both must be JAX-traceable, since rays differentiate them.
    Coordinates use ``coordinate_contract``, whose length unit must be the
    scale's. ``continuation_steps`` and ``polarization_tolerance`` configure the
    `ColdPlasmaDielectric` branch identification used for mode labels.
    Collisions are excluded: the ray Hamiltonian must be real; absorption enters
    transport through mode-resolved coefficients.
    """

    __strict_contract__ = True

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)
    charge_numbers: Float64[_SpeciesDim]
    mass_ratios: Float64[_SpeciesDim]
    density: Callable[[Array], Array] = eqx.field(static=True)
    magnetic_field: Callable[[Array], Array] = eqx.field(static=True)
    species_count: Size[_SpeciesDim] = eqx.field(static=True)
    continuation_steps: int = eqx.field(static=True)
    polarization_tolerance: float = eqx.field(static=True)
    profile_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        charge_numbers: ConvertibleToArray,
        mass_ratios: ConvertibleToArray,
        density: Callable[[Array], Array],
        magnetic_field: Callable[[Array], Array],
        profile_id: str,
        continuation_steps: int = 64,
        polarization_tolerance: float = 1.0e-12,
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if (
            coordinate_contract.length_unit.unit_id
            != scale.relativity.dimensional_scale.length_unit.unit_id
        ):
            raise ValueError(
                "coordinate_contract must use the electromagnetic scale's length unit."
            )
        if not callable(density) or not callable(magnetic_field):
            raise TypeError("density and magnetic_field must be callable.")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("profile_id must be nonempty.")
        scope = Scope()
        charge = as_host_array(
            charge_numbers, HostFloat64[_SpeciesDim], "charge_numbers", scope=scope
        )
        mass = as_host_array(
            mass_ratios, HostFloat64[_SpeciesDim], "mass_ratios", scope=scope
        )
        count = parse(charge.shape[0], Size[_SpeciesDim], "species_count", scope=scope)
        if not np.all(np.isfinite(charge)) or np.any(charge == 0.0):
            raise ValueError("charge_numbers must be finite and nonzero.")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("mass_ratios must be finite and strictly positive.")
        if not isinstance(continuation_steps, int) or continuation_steps < 1:
            raise ValueError("continuation_steps must be a positive integer.")
        tolerance = _positive(polarization_tolerance, "polarization_tolerance")
        self.scale = scale
        self.coordinate_contract = coordinate_contract
        self.charge_numbers = jnp.asarray(charge)
        self.mass_ratios = jnp.asarray(mass)
        self.density = density
        self.magnetic_field = magnetic_field
        self.species_count = count
        self.continuation_steps = continuation_steps
        self.polarization_tolerance = tolerance
        self.profile_id = canonical_fingerprint(
            {
                "kind": "cold-plasma-profile",
                "name": profile_id.strip(),
                "scale": scale.scale_id,
                "coordinates": coordinate_contract.spatial_id,
                "species": array_tree_fingerprint(
                    {"charge_numbers": charge, "mass_ratios": mass}
                ),
                "continuation_steps": continuation_steps,
                "polarization_tolerance": tolerance,
            }
        )

    def local_state(self, position: Array, /) -> tuple[Array, Array]:
        """Species densities ``[species]`` and ``B₀[3]`` at one point."""
        densities = jnp.asarray(self.density(position), dtype=jnp.float64)
        field = jnp.asarray(self.magnetic_field(position), dtype=jnp.float64)
        if densities.shape != (self.species_count,) or field.shape != (3,):
            raise ValueError(
                "density must return (species_count,) and magnetic_field (3,) values."
            )
        return densities, field

    def stix(self, position: Array, omega: Array, /) -> StixParameters:
        """Stix parameters at one point and real angular frequency ``ω``."""
        densities, field = self.local_state(position)
        plasma, gyro = _species_frequencies(
            self.scale,
            densities,
            self.charge_numbers,
            self.mass_ratios,
            jnp.sqrt(jnp.sum(field * field)),
        )
        return _species_stix(
            jnp.asarray(omega).astype(jnp.complex128),
            plasma,
            gyro,
            jnp.zeros_like(plasma),
        )

    def dielectric_at(self, position: ArrayLike, /) -> ColdPlasmaDielectric:
        """Host-prepared homogeneous `ColdPlasmaDielectric` of the plasma at a point."""
        point = as_host_array(position, HostFloat64[Literal[3]], "position")
        densities, field = self.local_state(jnp.asarray(point))
        return ColdPlasmaDielectric(
            self.scale,
            densities=np.asarray(densities),
            charge_numbers=np.asarray(self.charge_numbers),
            mass_ratios=np.asarray(self.mass_ratios),
            magnetic_field=np.asarray(field),
            continuation_steps=self.continuation_steps,
            polarization_tolerance=self.polarization_tolerance,
        )


_LAUNCH_FAILURE = (
    ColdPlasmaWaveStatus.EVANESCENT
    | ColdPlasmaWaveStatus.RESONANT
    | ColdPlasmaWaveStatus.ROOT_DEGENERATE
    | ColdPlasmaWaveStatus.NONFINITE
).value

_PATH_FAILURE = (
    ColdPlasmaWaveStatus.EVANESCENT
    | ColdPlasmaWaveStatus.RESONANT
    | ColdPlasmaWaveStatus.ROOT_DEGENERATE
    | ColdPlasmaWaveStatus.POLARIZATION_UNDEFINED
    | ColdPlasmaWaveStatus.NONFINITE
).value


def _field_angle(direction: Array, unit_field: Array, /) -> Array:
    """Angle in ``[0, π]`` between a unit wave normal and the unit field."""
    cosine = jnp.dot(direction, unit_field)
    sine = jnp.sqrt(jnp.sum(jnp.cross(direction, unit_field) ** 2))
    return jnp.arctan2(sine, cosine)


def _stix_at(stix: StixParameters, /) -> StixParameters:
    return StixParameters(
        sum_term=jnp.reshape(stix.sum_term, ()),
        difference_term=jnp.reshape(stix.difference_term, ()),
        plasma_term=jnp.reshape(stix.plasma_term, ()),
        right_term=jnp.reshape(stix.right_term, ()),
        left_term=jnp.reshape(stix.left_term, ()),
    )


def _branch_index_squared(
    stix: StixParameters, angle: Array, sign: Array, stable_quotient: Array, /
) -> Array:
    """``n²(θ)`` on the branch ``(B + sF)/(2A)`` in its cancellation-free form."""
    sine = jnp.sin(angle)
    cosine = jnp.cos(angle)
    a, b, c, discriminant_squared = _quartic_coefficients(
        stix, sine * sine, cosine * cosine
    )
    splitting = sign * jnp.sqrt(discriminant_squared)
    direct = (b + splitting) / (2.0 * a)
    vieta = 2.0 * c / (b - splitting)
    return jnp.where(stable_quotient, direct, vieta).real


def _ray_refraction(
    stix: StixParameters, angle: Array, index_squared: Array, /
) -> tuple[Array, Array]:
    """Ray refractive index squared ``n_r²`` and ``cos α`` of the branch through ``n²``.

    ``n_r² = n² |sin θ / (sin Θ dΘ/dθ)| / cos α`` (Bekefi 1966) with the group
    direction ``Θ = θ + α``, ``tan α = −(dn/dθ)/n``; along ``B₀`` the ratio
    ``sin θ / sin Θ`` is its limit ``1/Θ'``.
    """
    sine = jnp.sin(angle)
    cosine = jnp.cos(angle)
    a, b, _, discriminant_squared = _quartic_coefficients(
        stix, sine * sine, cosine * cosine
    )
    splitting = jnp.sqrt(discriminant_squared)
    plus = ((b + splitting) / (2.0 * a)).real
    minus = ((b - splitting) / (2.0 * a)).real
    sign = jnp.where(
        jnp.abs(plus - index_squared) <= jnp.abs(minus - index_squared), 1.0, -1.0
    ).astype(jnp.complex128)
    stable = jnp.abs(b + sign * splitting) >= jnp.abs(b - sign * splitting)

    def branch(theta: Array) -> Array:
        return _branch_index_squared(stix, theta, sign, stable)

    def deviation(theta: Array) -> Array:
        return jnp.arctan(-0.5 * jax.grad(branch)(theta) / branch(theta))

    alpha = deviation(angle)
    turn = 1.0 + jax.grad(deviation)(angle)
    group_angle = angle + alpha
    group_sine = jnp.sin(group_angle)
    axial = jnp.abs(sine) <= 1.0e-8
    safe_group_sine = jnp.where(axial, 1.0, group_sine)
    ratio = jnp.where(axial, 1.0 / (turn * turn), sine / (safe_group_sine * turn))
    cos_alpha = jnp.cos(alpha)
    return branch(angle) * jnp.abs(ratio) / cos_alpha, cos_alpha


class ColdPlasmaHamiltonian(AbstractDispersionHamiltonian):
    """Ray Hamiltonian of one cold-plasma mode at angular frequency ``ω``.

    ``H = D / G`` with the Stix quartic ``D`` and ``G = −ω ∂D/∂ω|_k``, so the ray
    parameter is the group light path ``c t``. ``mode`` selects the launch root
    from the `ColdPlasmaDielectric` labels (``RIGHT``/``LEFT`` from ``θ = 0``,
    ``ORDINARY``/``EXTRAORDINARY`` from ``θ = π/2``); launches whose root is
    evanescent, resonant, degenerate or whose label is ambiguous are refused.
    """

    __strict_contract__ = True

    profile: ColdPlasmaProfile
    angular_frequency: Float64[Scalar]
    mode: PlasmaWaveMode = eqx.field(static=True)

    def __init__(
        self,
        profile: ColdPlasmaProfile,
        /,
        *,
        angular_frequency: float,
        mode: PlasmaWaveMode,
    ) -> None:
        if not isinstance(profile, ColdPlasmaProfile):
            raise TypeError("profile must be a ColdPlasmaProfile.")
        omega = _positive(angular_frequency, "angular_frequency")
        mode_ = parse(mode, PlasmaWaveMode, "mode")
        self.profile = profile
        self.angular_frequency = jnp.asarray(omega, dtype=jnp.float64)
        self.mode = mode_
        self.coordinate_contract = profile.coordinate_contract
        self.hamiltonian_id = canonical_fingerprint(
            {
                "kind": "cold-plasma-ray-hamiltonian",
                "profile": profile.profile_id,
                "angular_frequency": omega,
                "mode": mode_.name,
            }
        )

    @property
    def wavenumber(self) -> Array:
        """Vacuum wavenumber ``ω/c`` per length unit."""
        return self.angular_frequency / float(self.profile.scale.speed_of_light)

    def _dispersion(self, position: Array, momentum: Array, omega: Array, /) -> Array:
        _, field = self.profile.local_state(position)
        unit = field / jnp.sqrt(jnp.sum(field * field))
        parallel = jnp.dot(momentum, unit)
        parallel_squared = parallel * parallel
        stix = _stix_at(self.profile.stix(position, omega))
        return _dispersion_polynomial(
            stix, parallel_squared, jnp.sum(momentum * momentum) - parallel_squared
        ).real

    def value(self, position: Array, momentum: Array, /) -> Array:
        omega = self.angular_frequency
        dispersion = self._dispersion(position, momentum, omega)
        momentum_gradient = jax.grad(self._dispersion, argnums=1)(
            position, momentum, omega
        )
        frequency_gradient = jax.grad(self._dispersion, argnums=2)(
            position, momentum, omega
        )
        normalization = jnp.dot(momentum, momentum_gradient) - omega * frequency_gradient
        return dispersion / normalization

    def supported(self, position: Array, /) -> Array:
        densities, field = self.profile.local_state(position)
        stix = self.profile.stix(position, self.angular_frequency)
        terms = jnp.stack(
            (stix.sum_term, stix.difference_term, stix.plasma_term)
        ).reshape(-1)
        return (
            jnp.all(jnp.isfinite(densities))
            & jnp.all(densities >= 0.0)
            & jnp.all(jnp.isfinite(field))
            & (jnp.sum(field * field) > 0.0)
            & jnp.all(jnp.isfinite(terms))
        )

    def launch(self, position: Array, direction: Array, /) -> tuple[Array, Array]:
        _, field = self.profile.local_state(position)
        unit = field / jnp.sqrt(jnp.sum(field * field))
        unit_direction = direction.astype(jnp.float64)
        angle = _field_angle(unit_direction, unit)
        wave = _cold_plasma_waves(
            _stix_at(self.profile.stix(position, self.angular_frequency)),
            self.angular_frequency,
            angle,
            self.profile.continuation_steps,
            self.profile.polarization_tolerance,
        )
        match self.mode:
            case PlasmaWaveMode.RIGHT | PlasmaWaveMode.LEFT:
                ambiguous = ColdPlasmaWaveStatus.PARALLEL_LABEL_AMBIGUOUS.value
            case PlasmaWaveMode.ORDINARY | PlasmaWaveMode.EXTRAORDINARY:
                ambiguous = ColdPlasmaWaveStatus.PERPENDICULAR_LABEL_AMBIGUOUS.value
            case _:
                assert_never(self.mode)
        status = wave.select(self.mode, wave.status)
        index_squared = wave.select(self.mode, wave.n_squared)
        valid = (
            ((status & (_LAUNCH_FAILURE | ambiguous)) == 0)
            & (index_squared.real > 0.0)
            & jnp.isfinite(index_squared.real)
        )
        index = jnp.sqrt(jnp.where(valid, index_squared.real, 1.0))
        return index * unit_direction, valid

    def sample_path(
        self, rays: DispersionRayResult, transverse_reference: ConvertibleToArray, /
    ) -> ColdPlasmaRayPath:
        """Mode, polarization and coupling data on every segment of traced rays.

        ``transverse_reference`` fixes the first polarization basis vector
        ``e₁ ∝ a − (a·k̂)k̂`` at each ray's first segment; the basis is
        parallel-transported along the wave normals thereafter.
        """
        if not isinstance(rays, DispersionRayResult):
            raise TypeError("rays must be a DispersionRayResult.")
        if rays.evidence.hamiltonian_id != self.hamiltonian_id:
            raise ValueError("rays were not traced with this ColdPlasmaHamiltonian.")
        reference = as_host_array(
            transverse_reference, HostFloat64[Literal[3]], "transverse_reference"
        )
        if not np.all(np.isfinite(reference)) or not np.any(reference != 0.0):
            raise ValueError("transverse_reference must be a finite nonzero vector.")
        positions = jnp.swapaxes(rays.position_history, 0, 1).astype(jnp.float64)
        momenta = jnp.swapaxes(rays.momentum_history, 0, 1).astype(jnp.float64)
        chords = positions[:, 1:] - positions[:, :-1]
        lengths = jnp.sqrt(jnp.sum(chords * chords, axis=-1))
        midpoints = 0.5 * (positions[:, 1:] + positions[:, :-1])
        mid_momenta = 0.5 * (momenta[:, 1:] + momenta[:, :-1])
        samples = jax.vmap(jax.vmap(self._segment))(midpoints, mid_momenta)
        return _assemble_path(
            self,
            rays,
            lengths,
            midpoints,
            samples,
            jnp.asarray(reference),
        )

    def _segment(self, position: Array, momentum: Array, /) -> _SegmentSample:
        _, field = self.profile.local_state(position)
        unit_field = field / jnp.sqrt(jnp.sum(field * field))
        index_squared = jnp.sum(momentum * momentum)
        normal = momentum / jnp.sqrt(index_squared)
        angle = _field_angle(normal, unit_field)
        stix = _stix_at(self.profile.stix(position, self.angular_frequency))
        wave = _cold_plasma_waves(
            stix,
            self.angular_frequency,
            angle,
            self.profile.continuation_steps,
            self.profile.polarization_tolerance,
        )
        faraday = _faraday_coefficients(wave, float(self.profile.scale.speed_of_light))
        roots = wave.n_squared.real
        slot = jnp.argmin(jnp.abs(roots - index_squared)).astype(jnp.int32)
        other = 1 - slot
        ray_index_squared, group_cosine = _ray_refraction(stix, angle, roots[slot])
        sine = jnp.sin(angle)
        # ê₁ = (cos θ k̂ − B̂)/sin θ lies in the k–B₀ plane (`FaradayCoefficients`).
        local_first = (jnp.cos(angle) * normal - unit_field) / jnp.where(
            sine > 0.0, sine, 1.0
        )
        separation = jnp.abs(roots[0] - roots[1]) / (
            jnp.abs(roots[0]) + jnp.abs(roots[1])
        )
        index = wave.refractive_index.real
        return _SegmentSample(
            normal=normal,
            angle=angle,
            local_first=local_first,
            axial=~(sine > 1.0e-12),
            index=jnp.stack((index[slot], index[other])),
            ray_index_squared=ray_index_squared,
            group_cosine=group_cosine,
            mode_stokes=jnp.stack(
                (faraday.mode_stokes[slot], faraday.mode_stokes[other])
            ),
            faraday=2.0 * faraday.coefficients.real,
            slot=slot,
            parallel_mode=wave.parallel_mode[slot],
            perpendicular_mode=wave.perpendicular_mode[slot],
            wave_status=wave.status[slot],
            mismatch=jnp.abs(roots[slot] - index_squared) / index_squared,
            separation=separation,
            quasi_transverse=wave.quasi_transverse_term >= wave.quasi_longitudinal_term,
        )


class _SegmentSample(StrictModule):
    normal: Array
    angle: Array
    local_first: Array
    axial: Array
    index: Array
    ray_index_squared: Array
    group_cosine: Array
    mode_stokes: Array
    faraday: Array
    slot: Array
    parallel_mode: Array
    perpendicular_mode: Array
    wave_status: Array
    mismatch: Array
    separation: Array
    quasi_transverse: Array


class ColdPlasmaRayPath(StrictModule):
    """Cold-plasma mode data on the segments of traced rays (ray, segment axes).

    Each segment is sampled at its chord midpoint with the midpoint momentum.
    ``segment_lengths`` are chord lengths ``ds`` and ``normal_lengths`` their
    projection ``cos α ds`` on the wave normal, the length along which the
    per-wave-normal coefficients of `ColdPlasmaDielectric` and
    `MagnetobremsstrahlungPlan` act. ``polarization_basis`` is the transported
    ``e₁`` (``e₂ = k̂ × e₁``); ``mode_stokes`` (ray mode, companion) and the Faraday
    rotation vector ``faraday = 2 Re ρ`` are expressed in it. ``refractive_index``
    holds (ray mode, companion) indices and ``ray_index_squared`` the ray
    refractive index ``n_r²`` of the ray mode. ``coupling_parameter`` is the
    mode-axis turning rate on the Poincaré sphere over the beat wavenumber
    ``k |n₀ − n₁|``: weak coupling (modes propagate independently) needs it
    small, strong coupling (the polarization does not follow the modes) large.
    ``quasi_transverse`` marks segments where the quasi-transverse part of the
    discriminant dominates. ``mode_mismatch`` is ``|n_σ² − |p|²| / |p|²`` of the
    ray's root and ``wave_status`` its `ColdPlasmaWaveStatus` bits.
    """

    __strict_contract__ = True

    angular_frequency: Float64[Scalar]
    wavenumber: Float64[Scalar]
    segment_lengths: Float64[_RayDim, _SegmentDim]
    normal_lengths: Float64[_RayDim, _SegmentDim]
    midpoints: Float64[_RayDim, _SegmentDim, Literal[3]]
    wave_normals: Float64[_RayDim, _SegmentDim, Literal[3]]
    polarization_basis: Float64[_RayDim, _SegmentDim, Literal[3]]
    angle: Float64[_RayDim, _SegmentDim]
    refractive_index: Float64[_RayDim, _SegmentDim, Literal[2]]
    ray_index_squared: Float64[_RayDim, _SegmentDim]
    mode_stokes: Float64[_RayDim, _SegmentDim, Literal[2], Literal[3]]
    faraday: Float64[_RayDim, _SegmentDim, Literal[3]]
    mode_slot: Int32[_RayDim, _SegmentDim]
    parallel_mode: Int32[_RayDim, _SegmentDim]
    perpendicular_mode: Int32[_RayDim, _SegmentDim]
    wave_status: Int32[_RayDim, _SegmentDim]
    coupling_parameter: Float64[_RayDim, _SegmentDim]
    index_splitting: Float64[_RayDim, _SegmentDim]
    mode_separation: Float64[_RayDim, _SegmentDim]
    quasi_transverse: Bool[_RayDim, _SegmentDim]
    mode_mismatch: Float64[_RayDim, _SegmentDim]
    valid: Bool[_RayDim, _SegmentDim]
    hamiltonian_id: str = eqx.field(static=True)
    ray_plan_id: str = eqx.field(static=True)


def _transport_basis(normals: Array, reference: Array, /) -> Array:
    """Parallel transport of ``e₁`` along the segment wave normals of one ray."""

    def project(vector: Array, normal: Array) -> Array:
        transverse = vector - jnp.dot(vector, normal) * normal
        return transverse / jnp.sqrt(jnp.sum(transverse * transverse))

    def step(previous: Array, normal: Array) -> tuple[Array, Array]:
        current = project(previous, normal)
        return current, current

    first = project(reference, normals[0])
    _, basis = jax.lax.scan(step, first, normals[1:])
    return jnp.concatenate((first[None], basis), axis=0)


def _rotate_stokes(values: Array, angle: Array, /) -> Array:
    """Rotate ``(Q, U, V)`` from the local to the transported basis by ``2χ``."""
    cosine = jnp.cos(2.0 * angle)
    sine = jnp.sin(2.0 * angle)
    q, u, v = values[..., 0], values[..., 1], values[..., 2]
    return jnp.stack((cosine * q - sine * u, sine * q + cosine * u, v), axis=-1)


def _assemble_path(
    hamiltonian: ColdPlasmaHamiltonian,
    rays: DispersionRayResult,
    lengths: Array,
    midpoints: Array,
    samples: _SegmentSample,
    reference: Array,
    /,
) -> ColdPlasmaRayPath:
    basis = jax.vmap(_transport_basis, in_axes=(0, None))(samples.normal, reference)
    second = jnp.cross(samples.normal, basis)
    chi = jnp.where(
        samples.axial,
        0.0,
        jnp.arctan2(
            jnp.sum(samples.local_first * second, axis=-1),
            jnp.sum(samples.local_first * basis, axis=-1),
        ),
    )
    mode_stokes = _rotate_stokes(samples.mode_stokes, chi[..., None])
    faraday = _rotate_stokes(samples.faraday, chi)
    wavenumber = hamiltonian.wavenumber
    beat = wavenumber * jnp.abs(samples.index[..., 0] - samples.index[..., 1])
    ray_stokes = mode_stokes[..., 0, :]
    previous, current = ray_stokes[:, :-1], ray_stokes[:, 1:]
    turn = 2.0 * jnp.arctan2(
        jnp.sqrt(jnp.sum((current - previous) ** 2, axis=-1)),
        jnp.sqrt(jnp.sum((current + previous) ** 2, axis=-1)),
    )
    spacing = 0.5 * (lengths[:, 1:] + lengths[:, :-1])
    mean_beat = 0.5 * (beat[:, 1:] + beat[:, :-1])
    interface = turn / (jnp.where(spacing > 0.0, spacing, 1.0) * mean_beat)
    padding = jnp.zeros((interface.shape[0], 1), dtype=interface.dtype)
    coupling = jnp.maximum(
        jnp.concatenate((padding, interface), axis=1),
        jnp.concatenate((interface, padding), axis=1),
    )
    mean_index = 0.5 * (samples.index[..., 0] + samples.index[..., 1])
    splitting = jnp.abs(samples.index[..., 0] - samples.index[..., 1]) / mean_index
    ray_valid = jnp.broadcast_to(rays.state.valid[:, None], lengths.shape)
    valid = (
        ray_valid
        & ((samples.wave_status & _PATH_FAILURE) == 0)
        & jnp.isfinite(samples.ray_index_squared)
        & (samples.ray_index_squared > 0.0)
        & jnp.all(jnp.isfinite(mode_stokes), axis=(-2, -1))
    )
    return ColdPlasmaRayPath(
        angular_frequency=hamiltonian.angular_frequency,
        wavenumber=wavenumber,
        segment_lengths=lengths,
        normal_lengths=samples.group_cosine * lengths,
        midpoints=midpoints,
        wave_normals=samples.normal,
        polarization_basis=basis,
        angle=samples.angle,
        refractive_index=samples.index,
        ray_index_squared=samples.ray_index_squared,
        mode_stokes=mode_stokes,
        faraday=faraday,
        mode_slot=samples.slot,
        parallel_mode=samples.parallel_mode,
        perpendicular_mode=samples.perpendicular_mode,
        wave_status=samples.wave_status,
        coupling_parameter=coupling,
        index_splitting=splitting,
        mode_separation=samples.separation,
        quasi_transverse=samples.quasi_transverse,
        mode_mismatch=samples.mismatch,
        valid=valid,
        hamiltonian_id=hamiltonian.hamiltonian_id,
        ray_plan_id=rays.evidence.plan_id,
    )


__all__ = [
    "AbstractDispersionHamiltonian",
    "AbstractSeparableDispersionHamiltonian",
    "ColdPlasmaHamiltonian",
    "ColdPlasmaProfile",
    "ColdPlasmaRayPath",
    "DispersionRayEvidence",
    "DispersionRayMethod",
    "DispersionRayPlan",
    "DispersionRayResult",
    "DispersionRayState",
    "DispersionRayStatus",
    "PreparedDispersionRay",
    "RayFanPlan",
    "RayFanResult",
]

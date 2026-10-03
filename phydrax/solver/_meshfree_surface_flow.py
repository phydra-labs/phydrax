# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Closed-surface Stokes/Brinkman flow in covariant deformation (strain) form.

The viscous term is the divergence of the surface rate-of-strain tensor
``E(U) = 1/2 P (nabla U + nabla U^T) P``, not a vector Laplacian. On a
two-dimensional sheet with Gauss curvature ``K`` the two differ by
``2 Div E(U) = Delta_B U + K U + grad div U``: rigid motions (Killing fields)
carry no strain and dissipate nothing, whereas a Bochner or Hodge Laplacian
damps them. All derivatives compose the owning ``SurfaceTangentCalculus``
operators; the pressure gradient, divergence/tangency constraints and the
measure-mean pressure gauge are the owner's closed-surface Stokes blocks.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._validation import finite_real_scalar
from ..discretization.meshfree._surface_geometry import (
    SurfaceAmbientDim,
    SurfaceCodimensionDim,
    SurfaceNodeDim,
)
from ..discretization.meshfree._surface_pde import SurfaceTangentCalculus
from ..ein import contract
from ..linalg import (
    AbstractLinearOperator,
    BlockLinearOperator,
    ComposedLinearOperator,
    FailurePolicy,
    FunctionLinearOperator,
    GMRES,
    IdentityLinearOperator,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    ScaledLinearOperator,
    solve,
    SumLinearOperator,
    TolerancePolicy,
)
from ..typing import Bool, Float, Scalar


# Unpreconditioned collocated surface Stokes needs a long Krylov memory: the
# planner's default GMRES(20) stalls on a 400-node sphere, GMRES(400) converges
# in about 530 steps.
_DEFAULT_RESTART = 400
_DEFAULT_MAXIMUM_STEPS = 4000


def _default_policy(unknowns: int) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(restart=min(_DEFAULT_RESTART, unknowns)),
        tolerance=TolerancePolicy(
            relative=1e-10, absolute=1e-14, max_steps=_DEFAULT_MAXIMUM_STEPS
        ),
        failure=FailurePolicy("status"),
    )


def _symmetric_part(tensor: Array) -> Array:
    return 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))


@final
class MeshfreeSurfaceStokesResult(StrictModule):
    """Velocity, pressure and constraint/dissipation evidence of one solve.

    ``normal_residual`` is ``max |N^T U|``, ``divergence_residual`` is
    ``max |div_G U|`` and ``gauge_residual`` the measure-weighted pressure
    mean selected by the gauge block (the discrete divergence-theorem defect).
    ``dissipation`` is ``2 nu int E(U):E(U)`` with the surface quadrature.
    """

    __strict_contract__ = True
    velocity: Float[SurfaceNodeDim, SurfaceAmbientDim]
    pressure: Float[SurfaceNodeDim]
    normal_multiplier: Float[SurfaceNodeDim, SurfaceCodimensionDim]
    normal_residual: Float[Scalar]
    divergence_residual: Float[Scalar]
    gauge_residual: Float[Scalar]
    dissipation: Float[Scalar]
    finite: Bool[Scalar]
    linear_result: LinearSolveResult

    @property
    def successful(self) -> Array:
        return self.linear_result.successful & self.finite

    @property
    def status(self) -> Array:
        return self.linear_result.status

    @property
    def diagnostics(self) -> LinearSolveDiagnostics:
        return self.linear_result.diagnostics


@final
class MeshfreeSurfaceStokesPlan(StrictModule):
    """Prepared strain-form Stokes/Brinkman solve on one closed surface cloud.

    Unknowns ``(U, (p, lambda))`` share the owner's native block layout:
    ``r U - 2 nu P Div E(U) + grad_G p + N lambda = f``,
    ``div_G U - m (m.p) = 0`` and ``N^T U = 0``. The block operator is prepared
    once with ``linear_policy`` and reused by every ``solve``; ``None`` selects
    restarted ``GMRES(min(400, unknowns))``, relative tolerance ``1e-10``, at
    most 4000 steps and status-reported failure. With ``reaction = 0`` every
    Killing field of the surface lies in the velocity kernel; the native solve
    status reports the resulting singular behavior and no gauge is imposed on
    rigid motions.
    """

    calculus: SurfaceTangentCalculus
    system: LinearSystem
    linear_solve: PreparedLinearSolve
    viscosity: float = eqx.field(static=True)
    reaction: float = eqx.field(static=True)

    def __init__(
        self,
        calculus: SurfaceTangentCalculus,
        /,
        *,
        viscosity: float,
        reaction: float = 0.0,
        linear_policy: LinearSolvePolicy | None = None,
    ) -> None:
        if not isinstance(calculus, SurfaceTangentCalculus):
            raise TypeError("calculus must be a SurfaceTangentCalculus.")
        if linear_policy is not None and not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if calculus.surface.plan.boundary is not None:
            raise ValueError(
                "Surface Stokes admits closed surfaces only; open surfaces need velocity boundary rows."
            )
        viscosity_ = finite_real_scalar(viscosity, "viscosity")
        reaction_ = finite_real_scalar(reaction, "reaction")
        if viscosity_ <= 0:
            raise ValueError("viscosity must be positive.")
        if reaction_ < 0:
            raise ValueError("reaction must be nonnegative.")
        vector = calculus.vector_space
        tensor = calculus.covariant_gradient.target
        symmetric = FunctionLinearOperator(
            _symmetric_part,
            source=tensor,
            target=tensor,
            transpose_action=_symmetric_part,
            operator_id=f"{vector.space_id}:symmetric-part",
        )
        # The covariant gradient is tangential in both indices, so its
        # symmetric part already equals 1/2 P (nabla U + nabla U^T) P.
        strain = ComposedLinearOperator(symmetric, calculus.covariant_gradient)
        primal = SumLinearOperator(
            ScaledLinearOperator(IdentityLinearOperator(vector), reaction_),
            ScaledLinearOperator(
                ComposedLinearOperator(calculus.tensor_divergence, strain),
                -2 * viscosity_,
            ),
        )
        system = LinearSystem(
            _strain_block_operator(calculus, primal, viscosity_, reaction_)
        )
        self.calculus = calculus
        self.system = system
        self.linear_solve = prepare(
            system,
            _default_policy(system.operator.source.size)
            if linear_policy is None
            else linear_policy,
        )
        self.viscosity = viscosity_
        self.reaction = reaction_

    def strain(self, velocity: ArrayLike) -> Array:
        """Surface rate of strain ``E(U)`` of a tangent field, shape ``(N, A, A)``."""
        return _symmetric_part(
            self.calculus.covariant_gradient.mv(self._velocity(velocity))
        )

    def stress(self, velocity: ArrayLike, pressure: ArrayLike) -> Array:
        """Cauchy surface stress ``2 nu E(U) - p P``."""
        pressure_ = jnp.asarray(pressure, dtype=self.calculus.surface.points.dtype)
        if pressure_.shape != self.calculus.surface.measures.shape:
            raise ValueError("pressure must have one value per surface node.")
        projectors = self.calculus.surface.geometry.projectors
        return (
            2 * self.viscosity * self.strain(velocity)
            - pressure_[:, None, None] * projectors
        )

    def dissipation(self, velocity: ArrayLike) -> Array:
        """Viscous dissipation ``2 nu int E(U):E(U)``."""
        strain = self.strain(velocity)
        density = contract("nab,nab->n", strain, strain)
        return 2 * self.viscosity * jnp.sum(self.calculus.surface.measures * density)

    def solve(self, forcing: ArrayLike) -> MeshfreeSurfaceStokesResult:
        calculus = self.calculus
        rhs = calculus.stokes_rhs(self._velocity(forcing))
        linear_result = solve(self.linear_solve, rhs)
        velocity, constraints = linear_result.value
        pressure = constraints[:, 0]
        normal_residual = jnp.max(jnp.abs(calculus.normal_constraint.mv(velocity)))
        divergence_residual = jnp.max(jnp.abs(calculus.divergence.mv(velocity)))
        gauge_residual = calculus.stokes_gauge_residual(linear_result.value)
        dissipation = self.dissipation(velocity)
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(constraints))
            & jnp.isfinite(dissipation)
        )
        return MeshfreeSurfaceStokesResult(
            velocity=velocity,
            pressure=pressure,
            normal_multiplier=constraints[:, 1:],
            normal_residual=normal_residual,
            divergence_residual=divergence_residual,
            gauge_residual=gauge_residual,
            dissipation=dissipation,
            finite=finite,
            linear_result=linear_result,
        )

    def _velocity(self, field: ArrayLike) -> Array:
        values = jnp.asarray(field, dtype=self.calculus.surface.points.dtype)
        if values.shape != self.calculus.vector_space.shape:
            raise ValueError("Surface vector fields must have shape (nodes, ambient).")
        return values


def _strain_block_operator(
    calculus: SurfaceTangentCalculus,
    primal: AbstractLinearOperator,
    viscosity: float,
    reaction: float,
) -> BlockLinearOperator:
    """Owner's closed-surface Stokes blocks with the strain-form momentum block."""
    owner = calculus.stokes_system(viscosity=viscosity, reaction=reaction).operator
    if not isinstance(owner, BlockLinearOperator):
        raise TypeError(
            "SurfaceTangentCalculus.stokes_system must return a block operator."
        )
    (_, coupling), constraints = owner.blocks
    return BlockLinearOperator(
        ((primal, coupling), constraints),
        source=owner.source,
        target=owner.target,
        operator_id=f"{calculus.vector_space.space_id}:strain-stokes",
    )


__all__ = [
    "MeshfreeSurfaceStokesPlan",
    "MeshfreeSurfaceStokesResult",
]

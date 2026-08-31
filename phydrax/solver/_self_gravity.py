#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..discretization.finite_difference import (
    diagonalize_fd_laplacian,
    FDLaplacianSolvePlan,
)
from ..discretization.finite_volume import (
    ConservativeDiffusionPlan,
    PreparedFiniteVolumeDynamics,
)
from ..equations import (
    CompressibleNavierStokesSystem,
    EulerSystem,
    IdealMHDSystem,
)
from ._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ._balance_law_transport import (
    AbstractPreparedBalanceLawTransport,
    BalanceLawSourceView,
)


class NewtonianGravityDiagnostics(eqx.Module):
    potential: Array
    face_acceleration: tuple[Array, ...]
    cell_acceleration: Array
    poisson_residual: Array
    compatibility_residual: Array
    gauge_defect: Array
    force_mean: Array
    internal_energy_defect: Array
    converged: Array


class NewtonianSelfGravityPlan(AbstractBalanceLawProcessPlan):
    gravitational_constant: float = eqx.field(static=True)
    gravity_argument: str | None = eqx.field(static=True)
    freefall_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float = 1.0,
        /,
        *,
        gravity_argument: str | None = None,
        freefall_fraction: float = 0.25,
    ):
        coupling = float(gravitational_constant)
        fraction = float(freefall_fraction)
        argument = None if gravity_argument is None else str(gravity_argument)
        if (
            not np.isfinite(coupling)
            or coupling <= 0.0
            or not np.isfinite(fraction)
            or fraction <= 0.0
            or (argument is not None and not argument)
        ):
            raise ValueError("Newtonian gravity parameters are invalid.")
        self.gravitational_constant = coupling
        self.gravity_argument = argument
        self.freefall_fraction = fraction
        self.process_id = canonical_fingerprint(
            {
                "kind": "newtonian-self-gravity",
                "gravitational_constant": coupling,
                "gravity_argument": argument,
                "freefall_fraction": fraction,
            }
        )

    def prepare(
        self, transport: AbstractPreparedBalanceLawTransport, /
    ) -> PreparedNewtonianSelfGravity:
        return PreparedNewtonianSelfGravity(self, transport)


class PreparedNewtonianSelfGravity(AbstractPreparedBalanceLawProcess):
    plan: NewtonianSelfGravityPlan
    transport: AbstractPreparedBalanceLawTransport
    dynamics: PreparedFiniteVolumeDynamics
    diffusion: Any
    poisson: FDLaplacianSolvePlan
    density_index: int = eqx.field(static=True)
    momentum_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        plan: NewtonianSelfGravityPlan,
        transport: AbstractPreparedBalanceLawTransport,
        /,
    ):
        if not isinstance(plan, NewtonianSelfGravityPlan):
            raise TypeError("plan must be NewtonianSelfGravityPlan.")
        if not isinstance(
            transport, AbstractPreparedBalanceLawTransport
        ) or not isinstance(transport.dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError(
                "Newtonian self-gravity requires stationary structured FV dynamics."
            )
        dynamics = transport.dynamics
        if not isinstance(
            dynamics.system,
            (EulerSystem, CompressibleNavierStokesSystem, IdealMHDSystem),
        ):
            raise TypeError("Newtonian self-gravity requires a compressible system.")
        grid = dynamics.discretization.grid
        if any(not axis.periodic for axis in grid.structured_axes):
            raise ValueError("Initial Newtonian self-gravity support is fully periodic.")
        names = tuple(dynamics.system.component_names)
        density_index = names.index("density")
        momentum_indices = tuple(
            index for index, name in enumerate(names) if name.startswith("momentum_")
        )
        if len(momentum_indices) != dynamics.system.dimension:
            raise RuntimeError("Gravity system momentum layout is inconsistent.")
        energy_index = names.index("total_energy")
        boundaries = {name: ("periodic", "periodic") for name in grid.axis_names}
        diagonalization = diagonalize_fd_laplacian(grid, boundaries)
        poisson = FDLaplacianSolvePlan(
            diagonalization,
            operator_scale=1.0,
            compatibility="project_rhs",
            gauge="zero_mean",
        )
        diffusion = ConservativeDiffusionPlan(grid).prepare(1.0)
        probe = jnp.arange(
            int(np.prod(grid.shape)), dtype=diffusion.source.dtype
        ).reshape(grid.shape)
        action_defect = float(
            np.asarray(
                jnp.max(jnp.abs(diagonalization.apply(probe) - diffusion.mv(probe)))
            )
        )
        if action_defect > 5.0e-10:
            raise RuntimeError(
                "Gravity transform and conservative Laplacian actions disagree."
            )
        self.plan = plan
        self.transport = transport
        self.dynamics = dynamics
        self.diffusion = diffusion
        self.poisson = poisson
        self.density_index = density_index
        self.momentum_indices = momentum_indices
        self.energy_index = energy_index
        self.cell_shape = tuple(grid.shape)
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-newtonian-self-gravity",
                "plan": plan.process_id,
                "transport": transport.transport_id,
                "poisson": poisson.plan_id,
                "diffusion": diffusion.operator_id,
            }
        )
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = "smooth_discrete"
        self.modified_components = tuple(names[index] for index in momentum_indices) + (
            "total_energy",
        )

    def initialize(
        self, source_view: BalanceLawSourceView, args: Any = None, /
    ) -> BalanceLawProcessState:
        del source_view, args
        return BalanceLawProcessState.empty(self.process_id)

    def _gravity(self, args: Any, dtype, /) -> Array:
        value = (
            self.plan.gravitational_constant
            if self.plan.gravity_argument is None
            else args[self.plan.gravity_argument]
        )
        coupling = jnp.asarray(value, dtype=dtype).reshape(())
        return eqx.error_if(
            coupling,
            ~jnp.isfinite(coupling) | (coupling <= 0.0),
            "Gravitational constant must be positive and finite.",
        )

    def _field(self, cell_average: Array, /) -> Array:
        expected = (
            int(np.prod(self.cell_shape)),
            len(self.dynamics.system.component_names),
        )
        value = jnp.asarray(cell_average)
        if value.shape != expected:
            raise ValueError(f"Gravity cell_average must have shape {expected}.")
        return value.reshape(self.cell_shape + (expected[-1],))

    def solve_density(
        self, density: Array, args: Any = None, /
    ) -> tuple[Array, tuple[Array, ...], Array, Any]:
        density_ = jnp.asarray(density)
        if density_.shape != self.cell_shape:
            raise ValueError(f"Gravity density must have shape {self.cell_shape}.")
        coupling = self._gravity(args, density_.dtype)
        measure = self.dynamics.discretization.cell_volumes.astype(density_.dtype)
        mean_density = jnp.sum(measure * density_) / jnp.sum(measure)
        source = 4.0 * jnp.pi * coupling * (density_ - mean_density)
        solved = self.poisson.solve(source)
        potential = solved.value
        gradients = self.diffusion.fluxes(potential, 1.0, None)
        face_acceleration = tuple(-gradient for gradient in gradients)
        cell_components = tuple(
            0.5 * (component + jnp.roll(component, -1, axis=axis))
            for axis, component in enumerate(face_acceleration)
        )
        cell_acceleration = jnp.stack(cell_components, axis=-1)
        return potential, face_acceleration, cell_acceleration, solved

    def _solve(
        self, cell_average: Array, args: Any, /
    ) -> tuple[Array, tuple[Array, ...], Array, Any]:
        field = self._field(cell_average)
        return self.solve_density(field[..., self.density_index], args)

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, process_state
        field = self._field(cell_average)
        density = field[..., self.density_index]
        coupling = self._gravity(args, density.dtype)
        maximum_density = jnp.max(density)
        return self.plan.freefall_fraction / jnp.sqrt(
            4.0 * jnp.pi * coupling * maximum_density
        )

    def advance(
        self,
        start_time: Array,
        end_time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        realization: Any = None,
        args: Any = None,
        /,
    ) -> BalanceLawProcessAdvance:
        del realization
        if process_state.process_id != self.process_id or process_state.values:
            raise ValueError("Newtonian gravity process state changed.")
        step = jnp.asarray(end_time - start_time)
        field = self._field(cell_average)
        potential, face_acceleration, acceleration, solved = self._solve(
            cell_average, args
        )
        density = field[..., self.density_index]
        momentum = field[..., self.momentum_indices]
        momentum_new = momentum + density[..., None] * acceleration * step
        kinetic_before = 0.5 * jnp.sum(momentum**2, axis=-1) / density
        kinetic_after = 0.5 * jnp.sum(momentum_new**2, axis=-1) / density
        energy_new = field[..., self.energy_index] + kinetic_after - kinetic_before
        candidate = field.at[..., self.momentum_indices].set(momentum_new)
        candidate = candidate.at[..., self.energy_index].set(energy_new)
        internal_before = field[..., self.energy_index] - kinetic_before
        internal_after = candidate[..., self.energy_index] - kinetic_after
        internal_defect = jnp.max(jnp.abs(internal_after - internal_before))
        successful = (
            solved.converged
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(self.dynamics.system.admissible(candidate))
        )
        accepted = jnp.where(successful, candidate, field)
        volume = self.dynamics.discretization.cell_volumes.astype(density.dtype)
        force_mean = jnp.sum(
            volume[..., None] * density[..., None] * acceleration,
            axis=tuple(range(len(self.cell_shape))),
        )
        diagnostics = NewtonianGravityDiagnostics(
            potential=potential,
            face_acceleration=face_acceleration,
            cell_acceleration=acceleration,
            poisson_residual=solved.residual_norm,
            compatibility_residual=solved.compatibility_residual,
            gauge_defect=jnp.abs(jnp.sum(volume * potential) / jnp.sum(volume)),
            force_mean=force_mean,
            internal_energy_defect=internal_defect,
            converged=solved.converged,
        )
        incoming_flat = field.reshape(cell_average.shape)
        accepted_flat = accepted.reshape(cell_average.shape)
        return BalanceLawProcessAdvance(
            cell_average=accepted_flat,
            process_state=process_state,
            successful=successful,
            source_change=accepted_flat - incoming_flat,
            diagnostics=diagnostics,
        )


__all__ = [
    "NewtonianGravityDiagnostics",
    "NewtonianSelfGravityPlan",
    "PreparedNewtonianSelfGravity",
]

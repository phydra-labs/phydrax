#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellMesh, TetrahedralConnectivity
from ....discretization.fem._de_rham import FiniteElementDeRhamComplex
from ....linalg._complexes import coordinate_space
from ....typing import checked
from ._frequency_domain import (
    _hx_preconditioner,
    _material_stiffness,
    ConductiveEMMaterial,
    FrequencyDomainEMSurvey,
)


@final
class TimeDomainEMState(StrictModule):
    electric: Array
    time_s: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)


@final
class TimeDomainEMResult(StrictModule):
    observations: Array
    final_state: TimeDomainEMState
    residual_norms: Array
    dissipated_energy_J: Array
    successful: Array


@final
class ImplicitTimeDomainEMPlan(StrictModule, NonTrainableState):
    """Backward-Euler quasistatic electric diffusion on tetrahedral H(curl).

    M_sigma dE/dt + K_mu E = source. The caller-supplied source history is the
    weak right-hand side of this diffusion equation; transmitter-current-to-RHS
    differentiation is a separate source model and is never inferred here.
    """

    complex: FiniteElementDeRhamComplex
    survey: FrequencyDomainEMSurvey
    free_edges: Array
    reduced_space: la.ArraySpace
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(self, mesh: CellMesh, survey: FrequencyDomainEMSurvey, /) -> None:
        complex = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
        if survey.electric_current_functionals.shape[1] != complex.cell_counts[1]:
            raise ValueError("Time-domain EM survey does not match H(curl) edges.")
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Time-domain EM requires tetrahedral connectivity.")
        boundary = np.asarray(connectivity.boundary_edges)
        if np.any(np.asarray(survey.electric_current_functionals)[:, boundary] != 0):
            raise ValueError("PEC boundary source rows must vanish.")
        free = np.flatnonzero(~boundary)
        if free.size == 0:
            raise ValueError("Time-domain EM mesh has no interior H(curl) edges.")
        self.complex, self.survey = complex, survey
        self.free_edges = jnp.asarray(free, dtype=jnp.int32)
        self.reduced_space = coordinate_space(
            complex.hilbert_complex(boundary="relative").space(1)
        )
        self.policy = la.LinearSolvePolicy(
            la.ConjugateGradient(),
            tolerance=la.TolerancePolicy(relative=1e-8, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-time-domain-em",
                "complex": complex.realization_id,
                "survey": survey.survey_id,
            }
        )

    def initial_state(self) -> TimeDomainEMState:
        return TimeDomainEMState(
            jnp.zeros(
                (
                    self.survey.electric_current_functionals.shape[0],
                    self.complex.cell_counts[1],
                )
            ),
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def _operators(
        self, material: ConductiveEMMaterial, dt: Array
    ) -> tuple[
        la.AbstractLinearOperator,
        Callable[[ArrayLike], Array],
        Callable[[ArrayLike], Array],
    ]:
        mass_operator = self.complex.constitutive_operator(1, material.conductivity_S_m)
        stiffness_operator = _material_stiffness(
            self.complex, material.inverse_permeability_m_H
        )

        def mass(field: ArrayLike) -> Array:
            return mass_operator.mv(field)

        def stiffness(field: ArrayLike) -> Array:
            return stiffness_operator.mv(field)

        relative_mass = self.complex.constitutive_operator(
            1, material.conductivity_S_m, boundary="relative"
        )
        relative_stiffness = _material_stiffness(
            self.complex, material.inverse_permeability_m_H, boundary="relative"
        )
        operator = (1.0 / dt) * relative_mass + relative_stiffness
        operator = eqx.tree_at(
            lambda value: value.properties,
            operator,
            la.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"positive_definite": "construction"},
            ),
        )
        operator = la.assemble_sparse(operator)
        return operator, mass, stiffness

    def step(
        self,
        state: TimeDomainEMState,
        material: ConductiveEMMaterial,
        dt_s: ArrayLike,
        source_amplitudes: ArrayLike,
        /,
    ) -> tuple[TimeDomainEMState, Array, Array, Array]:
        if state.plan_id != self.plan_id:
            raise ValueError("Time-domain EM state belongs to another plan.")
        dt = jnp.asarray(dt_s)
        amplitudes = jnp.asarray(source_amplitudes)
        source_count = self.survey.electric_current_functionals.shape[0]
        if dt.shape != () or amplitudes.shape != (source_count,):
            raise ValueError("Time-domain EM timestep/source amplitude shape is invalid.")
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0) | jnp.any(~jnp.isfinite(amplitudes)),
            "Time-domain EM timestep must be positive and sources finite.",
        )
        if material.conductivity_S_m.shape[0] != self.complex.cell_counts[3]:
            raise ValueError("Time-domain EM material does not match mesh cells.")
        operator, mass, _ = self._operators(material, dt)
        inverse = _hx_preconditioner(self.complex, operator)
        # The plan policy declares no preconditioner; its absent ``None`` slot
        # is the single leaf replaced by the per-step H(curl) preconditioner.
        policy = eqx.tree_at(
            lambda value: value.preconditioning,
            self.policy,
            la.PreconditioningPolicy(inverse),
            is_leaf=lambda node: node is None,
        )
        fields, residuals, energies, successes = [], [], [], []
        for source, previous, amplitude in zip(
            self.survey.electric_current_functionals,
            state.electric,
            amplitudes,
            strict=True,
        ):
            rhs_full = mass(previous) / dt + amplitude * source
            result = la.solve(
                la.LinearSystem(operator), rhs_full[self.free_edges], policy=policy
            )
            field = (
                jnp.zeros((self.complex.cell_counts[1],), dtype=result.value.dtype)
                .at[self.free_edges]
                .set(result.value)
            )
            residual = operator.mv(result.value) - rhs_full[self.free_edges]
            dissipation = dt * jnp.real(jnp.vdot(field, mass(field)))
            fields.append(field)
            residuals.append(jnp.sqrt(jnp.real(jnp.vdot(residual, residual))))
            energies.append(dissipation)
            successes.append(
                result.successful & jnp.isfinite(dissipation) & (dissipation >= 0)
            )
        next_state = TimeDomainEMState(
            jnp.stack(fields), state.time_s + dt, state.step_index + 1, self.plan_id
        )
        return (
            next_state,
            jnp.stack(residuals),
            jnp.stack(energies),
            jnp.all(jnp.stack(successes)),
        )

    def simulate(
        self,
        material: ConductiveEMMaterial,
        time_steps_s: ArrayLike,
        source_amplitudes: ArrayLike,
        /,
    ) -> TimeDomainEMResult:
        steps = jnp.asarray(time_steps_s)
        amplitudes = jnp.asarray(source_amplitudes)
        if steps.ndim != 1 or amplitudes.shape != (
            steps.size,
            self.survey.electric_current_functionals.shape[0],
        ):
            raise ValueError("Time-domain EM step/source history shapes disagree.")
        state = self.initial_state()
        observations, residuals, energies, successes = [], [], [], []
        for dt, source in zip(steps, amplitudes, strict=True):
            state, residual, energy, successful = self.step(state, material, dt, source)
            observations.append(
                ein.contract(
                    "me,me->m",
                    self.survey.receiver_functionals,
                    state.electric[self.survey.source_indices],
                )
            )
            residuals.append(residual)
            energies.append(energy)
            successes.append(successful)
        return TimeDomainEMResult(
            jnp.stack(observations),
            state,
            jnp.stack(residuals),
            jnp.sum(jnp.stack(energies), axis=0),
            jnp.all(jnp.stack(successes)),
        )


__all__ = ["ImplicitTimeDomainEMPlan", "TimeDomainEMResult", "TimeDomainEMState"]

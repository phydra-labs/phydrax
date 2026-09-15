#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ..._thermodynamics import (
    BinaryPhaseThermodynamicClosure,
    BinaryThermodynamicParameters,
)
from ..._trainable import NonTrainableState
from ...discretization import FiniteElementDiscretization
from ...equations import (
    CellResidualAction,
    compile_finite_element_problem,
    FiniteElementForm,
)
from ...nonlinear import NewtonKrylov, NonlinearTermination
from ...solver import (
    FiniteElementAcceptedStepSchedule,
    FiniteElementAttemptResult,
    FiniteElementStepPolicy,
)


class AllenCahnParameters(StrictModule, NonTrainableState):
    mobility: Array
    thermodynamics: BinaryThermodynamicParameters
    closure: BinaryPhaseThermodynamicClosure

    def __init__(
        self,
        mobility: ArrayLike,
        thermodynamics: BinaryThermodynamicParameters,
        /,
        *,
        closure: BinaryPhaseThermodynamicClosure | None = None,
    ):
        mobility_ = jnp.asarray(mobility)
        if mobility_.shape != () or not bool(jnp.isfinite(mobility_)) or mobility_ <= 0.0:
            raise ValueError("Allen-Cahn mobility must be a positive finite scalar.")
        if not isinstance(thermodynamics, BinaryThermodynamicParameters):
            raise TypeError("thermodynamics must be BinaryThermodynamicParameters.")
        selected = BinaryPhaseThermodynamicClosure() if closure is None else closure
        if not isinstance(selected, BinaryPhaseThermodynamicClosure):
            raise TypeError("closure must be BinaryPhaseThermodynamicClosure.")
        self.mobility = mobility_
        self.thermodynamics = thermodynamics
        self.closure = selected


class CahnHilliardParameters(StrictModule, NonTrainableState):
    mobility: Array
    thermodynamics: BinaryThermodynamicParameters
    closure: BinaryPhaseThermodynamicClosure

    def __init__(
        self,
        mobility: ArrayLike,
        thermodynamics: BinaryThermodynamicParameters,
        /,
        *,
        closure: BinaryPhaseThermodynamicClosure | None = None,
    ):
        mobility_ = jnp.asarray(mobility)
        if mobility_.shape != () or not bool(jnp.isfinite(mobility_)) or mobility_ <= 0.0:
            raise ValueError("Cahn-Hilliard mobility must be a positive finite scalar.")
        if not isinstance(thermodynamics, BinaryThermodynamicParameters):
            raise TypeError("thermodynamics must be BinaryThermodynamicParameters.")
        selected = BinaryPhaseThermodynamicClosure() if closure is None else closure
        if not isinstance(selected, BinaryPhaseThermodynamicClosure):
            raise TypeError("closure must be BinaryPhaseThermodynamicClosure.")
        self.mobility = mobility_
        self.thermodynamics = thermodynamics
        self.closure = selected


class PhaseFieldRejectionReason(IntFlag):
    NONLINEAR_SOLVE = 1
    NONFINITE = 2
    ENERGY_INCREASE = 4
    MASS_DRIFT = 8


class PhaseFieldAcceptancePolicy(StrictModule, NonTrainableState):
    """Acceptance and allocation limits for one passive phase-field transaction."""

    relative_energy_tolerance: float = eqx.field(static=True)
    absolute_mass_tolerance: float = eqx.field(static=True)
    maximum_degrees_of_freedom: int = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        relative_energy_tolerance: float = 1.0e-10,
        absolute_mass_tolerance: float = 1.0e-10,
        maximum_degrees_of_freedom: int = 1_000_000,
    ):
        energy = float(relative_energy_tolerance)
        mass = float(absolute_mass_tolerance)
        capacity = int(maximum_degrees_of_freedom)
        if (
            not np.isfinite(energy)
            or energy < 0.0
            or not np.isfinite(mass)
            or mass < 0.0
            or capacity <= 0
        ):
            raise ValueError(
                "Phase-field acceptance tolerances and capacity are invalid."
            )
        self.relative_energy_tolerance = energy
        self.absolute_mass_tolerance = mass
        self.maximum_degrees_of_freedom = capacity


class PhaseFieldStepResult(StrictModule):
    """Candidate and atomically accepted states with thermodynamic evidence."""

    candidate_state: object
    accepted_state: object
    successful: Array
    nonlinear_successful: Array
    finite: Array
    energy_accepted: Array
    mass_accepted: Array
    mass_before: Array
    candidate_mass: Array
    accepted_mass: Array
    energy_before: Array
    candidate_energy: Array
    accepted_energy: Array
    rejection_reasons: Array
    nonlinear_result: object


def _accept_phase_field_candidate(
    candidate_state,
    previous_state,
    nonlinear_successful: Array,
    mass_before: Array,
    candidate_mass: Array,
    energy_before: Array,
    candidate_energy: Array,
    policy: PhaseFieldAcceptancePolicy,
    /,
    *,
    conserve_mass: bool,
    nonlinear_result: object,
) -> PhaseFieldStepResult:
    leaves = candidate_state if isinstance(candidate_state, tuple) else (candidate_state,)
    finite = (
        jnp.isfinite(mass_before)
        & jnp.isfinite(candidate_mass)
        & jnp.isfinite(energy_before)
        & jnp.isfinite(candidate_energy)
        & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in leaves)))
    )
    energy_scale = jnp.maximum(jnp.abs(energy_before), 1.0)
    energy_accepted = candidate_energy <= (
        energy_before + policy.relative_energy_tolerance * energy_scale
    )
    mass_accepted = (
        jnp.abs(candidate_mass - mass_before) <= policy.absolute_mass_tolerance
        if conserve_mass
        else jnp.asarray(True)
    )
    successful = nonlinear_successful & finite & energy_accepted & mass_accepted
    if isinstance(candidate_state, tuple):
        accepted_state = tuple(
            jnp.where(successful, candidate, previous)
            for candidate, previous in zip(candidate_state, previous_state, strict=True)
        )
    else:
        accepted_state = jnp.where(successful, candidate_state, previous_state)
    reasons = jnp.zeros((), dtype=jnp.int32)
    reasons = reasons | jnp.where(
        nonlinear_successful,
        0,
        int(PhaseFieldRejectionReason.NONLINEAR_SOLVE),
    ).astype(jnp.int32)
    reasons = reasons | jnp.where(
        finite, 0, int(PhaseFieldRejectionReason.NONFINITE)
    ).astype(jnp.int32)
    reasons = reasons | jnp.where(
        energy_accepted, 0, int(PhaseFieldRejectionReason.ENERGY_INCREASE)
    ).astype(jnp.int32)
    reasons = reasons | jnp.where(
        mass_accepted, 0, int(PhaseFieldRejectionReason.MASS_DRIFT)
    ).astype(jnp.int32)
    return PhaseFieldStepResult(
        candidate_state,
        accepted_state,
        successful,
        nonlinear_successful,
        finite,
        energy_accepted,
        mass_accepted,
        mass_before,
        candidate_mass,
        jnp.where(successful, candidate_mass, mass_before),
        energy_before,
        candidate_energy,
        jnp.where(successful, candidate_energy, energy_before),
        reasons,
        nonlinear_result,
    )


def _guard_phase_field_resources(
    states: tuple[Array, ...], policy: PhaseFieldAcceptancePolicy, /
) -> None:
    degrees = sum(int(value.size) for value in states)
    if degrees > policy.maximum_degrees_of_freedom:
        raise ValueError(
            "Phase-field state exceeds maximum_degrees_of_freedom before compilation."
        )


def _field_local_data(
    discretization: FiniteElementDiscretization,
    field_name: str,
    state: Array,
    block_index: int,
):
    field_index = discretization._field_index(field_name)
    dofs = discretization.dof_maps[field_index].cell_dofs[block_index]
    orientation = discretization.dof_maps[field_index].orientations[block_index]
    return state[dofs] * orientation


def phase_field_mass(
    discretization: FiniteElementDiscretization,
    field_name: str,
    state: ArrayLike,
    /,
) -> Array:
    value = jnp.asarray(state)
    total = jnp.asarray(0.0, dtype=value.dtype)
    field_index = discretization._field_index(field_name)
    for block_index, geometry in enumerate(discretization.block_geometries[field_index]):
        local = _field_local_data(discretization, field_name, value, block_index)
        reconstructed = ein.contract("qi,ci->cq", geometry.basis_values, local)
        total = total + jnp.sum(reconstructed * geometry.physical_weights)
    return total


def phase_field_energy(
    discretization: FiniteElementDiscretization,
    field_name: str,
    state: ArrayLike,
    parameters: AllenCahnParameters | CahnHilliardParameters,
    /,
) -> Array:
    value = jnp.asarray(state)
    total = jnp.asarray(0.0, dtype=value.dtype)
    field_index = discretization._field_index(field_name)
    for block_index, geometry in enumerate(discretization.block_geometries[field_index]):
        local = _field_local_data(discretization, field_name, value, block_index)
        reconstructed = ein.contract("qi,ci->cq", geometry.basis_values, local)
        gradient = ein.contract("cqid,ci->cqd", geometry.physical_gradients, local)
        density = (
            parameters.thermodynamics.bulk_scale
            * parameters.closure.free_energy.density(reconstructed)
            + 0.5
            * parameters.thermodynamics.gradient_coefficient
            * jnp.sum(gradient**2, axis=-1)
        )
        total = total + jnp.sum(density * geometry.physical_weights)
    return total


def allen_cahn_form(
    discretization: FiniteElementDiscretization,
    field_name: str,
    previous: ArrayLike,
    step_size: float,
    parameters: AllenCahnParameters,
    /,
) -> FiniteElementForm:
    previous_ = jnp.asarray(previous)
    dt = float(step_size)
    if dt <= 0.0 or not isinstance(parameters, AllenCahnParameters):
        raise ValueError("Allen-Cahn step data are invalid.")
    previous_local = tuple(
        _field_local_data(discretization, field_name, previous_, block_index)
        for block_index in range(len(discretization.mesh.blocks))
    )
    if len(previous_local) != 1:
        raise ValueError("Allen-Cahn currently requires one homogeneous cell block.")

    def residual(values, gradients, points, weights, test_basis, test_gradients, context):
        value = values[0]
        gradient = gradients[0]
        previous_value = ein.contract("qi,ci->cq", test_basis, previous_local[0])
        local_drive = (value - previous_value) / dt + parameters.mobility * (
            parameters.thermodynamics.bulk_scale
            * parameters.closure.free_energy.derivative(value)
        )
        return ein.contract("cq,cq,qi->ci", weights, local_drive, test_basis) + (
            parameters.mobility
            * parameters.thermodynamics.gradient_coefficient
            * ein.contract("cq,cqid,cqd->ci", weights, test_gradients, gradient)
        )

    return FiniteElementForm(
        "allen-cahn-step",
        field_name,
        (
            CellResidualAction(
                field_name,
                (field_name,),
                residual,
                action_id="allen-cahn-residual",
            ),
        ),
    )


def cahn_hilliard_form(
    discretization: FiniteElementDiscretization,
    concentration_field: str,
    chemical_field: str,
    previous: ArrayLike,
    step_size: float,
    parameters: CahnHilliardParameters,
    /,
) -> FiniteElementForm:
    previous_ = jnp.asarray(previous)
    dt = float(step_size)
    if dt <= 0.0 or not isinstance(parameters, CahnHilliardParameters):
        raise ValueError("Cahn-Hilliard step data are invalid.")
    if len(discretization.mesh.blocks) != 1:
        raise ValueError("Cahn-Hilliard currently requires one homogeneous cell block.")
    previous_local = _field_local_data(discretization, concentration_field, previous_, 0)

    def conservation(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        concentration, _ = values
        _, chemical_gradient = gradients
        previous_value = ein.contract("qi,ci->cq", test_basis, previous_local)
        return ein.contract(
            "cq,cq,qi->ci",
            weights,
            (concentration - previous_value) / dt,
            test_basis,
        ) + parameters.mobility * ein.contract(
            "cq,cqid,cqd->ci",
            weights,
            test_gradients,
            chemical_gradient,
        )

    def chemical(values, gradients, points, weights, test_basis, test_gradients, context):
        concentration, potential = values
        concentration_gradient, _ = gradients
        return ein.contract(
            "cq,cq,qi->ci",
            weights,
            potential
            - parameters.thermodynamics.bulk_scale
            * parameters.closure.free_energy.derivative(concentration),
            test_basis,
        ) - parameters.thermodynamics.gradient_coefficient * ein.contract(
            "cq,cqid,cqd->ci",
            weights,
            test_gradients,
            concentration_gradient,
        )

    return FiniteElementForm(
        "cahn-hilliard-step",
        (concentration_field, chemical_field),
        (
            CellResidualAction(
                concentration_field,
                (concentration_field, chemical_field),
                conservation,
                action_id="cahn-hilliard-conservation",
            ),
            CellResidualAction(
                chemical_field,
                (concentration_field, chemical_field),
                chemical,
                action_id="cahn-hilliard-chemical-potential",
            ),
        ),
    )


def solve_allen_cahn_step(
    discretization: FiniteElementDiscretization,
    field_name: str,
    previous: ArrayLike,
    step_size: float,
    parameters: AllenCahnParameters,
    /,
    *,
    method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
    acceptance: PhaseFieldAcceptancePolicy | None = None,
) -> PhaseFieldStepResult:
    previous_ = jnp.asarray(previous)
    acceptance_ = PhaseFieldAcceptancePolicy() if acceptance is None else acceptance
    if not isinstance(acceptance_, PhaseFieldAcceptancePolicy):
        raise TypeError("acceptance must be PhaseFieldAcceptancePolicy or None.")
    _guard_phase_field_resources((previous_,), acceptance_)
    form = allen_cahn_form(discretization, field_name, previous_, step_size, parameters)
    compiled = compile_finite_element_problem(form, discretization)
    method_ = NewtonKrylov() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    result = method_.solve(
        compiled.as_nonlinear_problem(),
        previous_,
        termination=termination_,
    )
    candidate = result.state
    mass_before = phase_field_mass(discretization, field_name, previous_)
    candidate_mass = phase_field_mass(discretization, field_name, candidate)
    energy_before = phase_field_energy(discretization, field_name, previous_, parameters)
    candidate_energy = phase_field_energy(
        discretization, field_name, candidate, parameters
    )
    return _accept_phase_field_candidate(
        candidate,
        previous_,
        result.successful,
        mass_before,
        candidate_mass,
        energy_before,
        candidate_energy,
        acceptance_,
        conserve_mass=False,
        nonlinear_result=result,
    )


def solve_cahn_hilliard_step(
    discretization: FiniteElementDiscretization,
    concentration_field: str,
    chemical_field: str,
    previous: ArrayLike,
    chemical_initial: ArrayLike,
    step_size: float,
    parameters: CahnHilliardParameters,
    /,
    *,
    method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
    acceptance: PhaseFieldAcceptancePolicy | None = None,
) -> PhaseFieldStepResult:
    previous_ = jnp.asarray(previous)
    chemical_initial_ = jnp.asarray(chemical_initial)
    acceptance_ = PhaseFieldAcceptancePolicy() if acceptance is None else acceptance
    if not isinstance(acceptance_, PhaseFieldAcceptancePolicy):
        raise TypeError("acceptance must be PhaseFieldAcceptancePolicy or None.")
    _guard_phase_field_resources((previous_, chemical_initial_), acceptance_)
    initial = (previous_, chemical_initial_)
    form = cahn_hilliard_form(
        discretization,
        concentration_field,
        chemical_field,
        previous_,
        step_size,
        parameters,
    )
    compiled = compile_finite_element_problem(form, discretization)
    method_ = NewtonKrylov() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    result = method_.solve(
        compiled.as_nonlinear_problem(),
        initial,
        termination=termination_,
    )
    candidate = result.state
    concentration, _ = candidate
    mass_before = phase_field_mass(discretization, concentration_field, previous_)
    candidate_mass = phase_field_mass(discretization, concentration_field, concentration)
    energy_before = phase_field_energy(
        discretization, concentration_field, previous_, parameters
    )
    candidate_energy = phase_field_energy(
        discretization, concentration_field, concentration, parameters
    )
    return _accept_phase_field_candidate(
        candidate,
        initial,
        result.successful,
        mass_before,
        candidate_mass,
        energy_before,
        candidate_energy,
        acceptance_,
        conserve_mass=True,
        nonlinear_result=result,
    )


def allen_cahn_schedule(
    discretization: FiniteElementDiscretization,
    field_name: str,
    parameters: AllenCahnParameters,
    /,
    *,
    method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
    policy: FiniteElementStepPolicy | None = None,
    acceptance: PhaseFieldAcceptancePolicy | None = None,
) -> FiniteElementAcceptedStepSchedule:
    def attempt(accepted, start, end, time_law, args):
        result = solve_allen_cahn_step(
            discretization,
            field_name,
            accepted.fields[0],
            end - start,
            parameters,
            method=method,
            termination=termination,
            acceptance=acceptance,
        )
        return FiniteElementAttemptResult(
            (result.accepted_state,),
            result.successful,
            retry_requested=~result.successful,
            suggested_step=0.5 * (end - start),
            diagnostics=result,
        )

    return FiniteElementAcceptedStepSchedule(
        attempt,
        policy=policy,
        schedule_id="allen-cahn-accepted-step",
    )


def cahn_hilliard_schedule(
    discretization: FiniteElementDiscretization,
    concentration_field: str,
    chemical_field: str,
    parameters: CahnHilliardParameters,
    /,
    *,
    method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
    policy: FiniteElementStepPolicy | None = None,
    acceptance: PhaseFieldAcceptancePolicy | None = None,
) -> FiniteElementAcceptedStepSchedule:
    def attempt(accepted, start, end, time_law, args):
        result = solve_cahn_hilliard_step(
            discretization,
            concentration_field,
            chemical_field,
            accepted.fields[0],
            accepted.fields[1],
            end - start,
            parameters,
            method=method,
            termination=termination,
            acceptance=acceptance,
        )
        return FiniteElementAttemptResult(
            result.accepted_state,
            result.successful,
            retry_requested=~result.successful,
            suggested_step=0.5 * (end - start),
            diagnostics=result,
        )

    return FiniteElementAcceptedStepSchedule(
        attempt,
        policy=policy,
        schedule_id="cahn-hilliard-accepted-step",
    )


__all__ = [
    "AllenCahnParameters",
    "CahnHilliardParameters",
    "PhaseFieldAcceptancePolicy",
    "PhaseFieldRejectionReason",
    "PhaseFieldStepResult",
    "allen_cahn_schedule",
    "allen_cahn_form",
    "cahn_hilliard_form",
    "cahn_hilliard_schedule",
    "phase_field_energy",
    "phase_field_mass",
    "solve_allen_cahn_step",
    "solve_cahn_hilliard_step",
]

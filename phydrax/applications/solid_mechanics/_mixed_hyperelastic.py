#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem._mixed_constraint import (
    MixedFiniteElementConstraintPlan,
    PreparedMixedFiniteElementConstraint,
    PressureGaugePolicy,
)
from ...equations import CellResidualAction, FiniteElementForm
from ...nn.parameters import ParameterSubspace
from ...solver._field_equilibrium import (
    prepare_functional_stationarity,
    prepare_virtual_work_equilibrium,
    PreparedFieldEquilibrium,
)


MixedHyperelasticFormulation: TypeAlias = Literal["exact", "finite-bulk"]
VolumetricConstraint: TypeAlias = Callable[[Array], Array]
IsochoricEnergy: TypeAlias = Callable[[Array], Array]


def _deformation_gradient(value: ArrayLike, /) -> Array:
    deformation = jnp.asarray(value)
    if deformation.ndim != 2 or deformation.shape[0] != deformation.shape[1]:
        raise ValueError("Deformation gradient must be one square rank-2 array.")
    if deformation.shape[0] not in (2, 3):
        raise ValueError("Mixed hyperelasticity supports 2D and 3D gradients.")
    if not jnp.issubdtype(deformation.dtype, jnp.inexact):
        deformation = deformation.astype(float)
    if jnp.issubdtype(deformation.dtype, jnp.complexfloating):
        raise TypeError("Deformation gradient must be real.")
    return deformation


def _pressure(value: ArrayLike, dtype: Any, /) -> Array:
    pressure = jnp.asarray(value, dtype=dtype)
    if pressure.shape != ():
        raise ValueError("Mixed pressure must be one scalar per material point.")
    return pressure.reshape(())


def _scalar(value: Any, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{name} must return one scalar per material point.")
    if jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must return a real scalar.")
    return scalar.reshape(())


def isochoric_deformation_gradient(deformation_gradient: ArrayLike, /) -> Array:
    """Return F̄ = J⁻¹ᐟᵈ F without hiding invalid Jacobians."""

    deformation = _deformation_gradient(deformation_gradient)
    jacobian = jnp.linalg.det(deformation)
    dimension = deformation.shape[0]
    return jnp.power(jacobian, -1.0 / dimension) * deformation


def mixed_pressure_first_piola(
    deformation_gradient: ArrayLike,
    pressure: ArrayLike,
    volumetric_constraint: VolumetricConstraint,
    /,
) -> Array:
    """Return the exact pressure contribution p ∂g(F)/∂F."""

    if not callable(volumetric_constraint):
        raise TypeError("volumetric_constraint must be callable.")
    deformation = _deformation_gradient(deformation_gradient)
    pressure_ = _pressure(pressure, deformation.dtype)

    def constraint(value: Array, /) -> Array:
        return _scalar(volumetric_constraint(value), "volumetric_constraint")

    return pressure_ * jax.grad(constraint)(deformation)


class MixedHyperelasticEvidence(StrictModule):
    """Pointwise finite-domain and orientation evidence for a mixed state."""

    jacobian: Array
    minimum_jacobian: Array
    jacobian_valid: Array
    pressure_finite: Array
    energy_finite: Array
    stress_finite: Array
    constraint_finite: Array
    finite: Array
    valid: Array


class MixedHyperelasticBlockTangent(StrictModule):
    """Four exact derivative blocks of the pointwise u-p residual."""

    deformation_deformation: Array
    deformation_pressure: Array
    pressure_deformation: Array
    pressure_pressure: Array


class MixedHyperelasticResponse(StrictModule):
    deformation_gradient: Array
    isochoric_deformation_gradient: Array
    pressure: Array
    jacobian: Array
    isochoric_energy: Array
    mixed_energy: Array
    volumetric_constraint: Array
    constraint_residual: Array
    isochoric_first_piola: Array
    pressure_first_piola: Array
    first_piola: Array
    evidence: MixedHyperelasticEvidence


class MixedHyperelasticLaw(StrictModule, NonTrainableState):
    """One exact or finite-bulk mixed material-point potential.

    The potential is Ψ_iso(J⁻¹ᐟᵈF) + p g(F) for exact incompressibility and
    Ψ_iso(J⁻¹ᐟᵈF) + p g(F) - p²/(2K) for finite bulk modulus. Consequently the
    pressure equation is exactly g(F) or g(F) - p/K and the pressure stress is
    +p ∂g/∂F.
    """

    isochoric_energy: IsochoricEnergy = eqx.field(static=True)
    volumetric_constraint: VolumetricConstraint = eqx.field(static=True)
    bulk_modulus: float | None = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    formulation: MixedHyperelasticFormulation = eqx.field(static=True)

    def __init__(
        self,
        isochoric_energy: IsochoricEnergy,
        volumetric_constraint: VolumetricConstraint,
        /,
        *,
        bulk_modulus: float | None = None,
        minimum_jacobian: float = 0.0,
    ):
        if not callable(isochoric_energy):
            raise TypeError("isochoric_energy must be callable.")
        if not callable(volumetric_constraint):
            raise TypeError("volumetric_constraint must be callable.")
        minimum = float(minimum_jacobian)
        if not isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_jacobian must be finite and nonnegative.")
        if bulk_modulus is None:
            bulk = None
            formulation: MixedHyperelasticFormulation = "exact"
        else:
            bulk = float(bulk_modulus)
            if not isfinite(bulk) or bulk <= 0.0:
                raise ValueError("bulk_modulus must be positive and finite.")
            formulation = "finite-bulk"
        self.isochoric_energy = isochoric_energy
        self.volumetric_constraint = volumetric_constraint
        self.bulk_modulus = bulk
        self.minimum_jacobian = minimum
        self.formulation = formulation

    def isochoric_value(self, deformation_gradient: ArrayLike, /) -> Array:
        deformation_bar = isochoric_deformation_gradient(deformation_gradient)
        return _scalar(self.isochoric_energy(deformation_bar), "isochoric_energy")

    def volumetric_value(self, deformation_gradient: ArrayLike, /) -> Array:
        deformation = _deformation_gradient(deformation_gradient)
        return _scalar(self.volumetric_constraint(deformation), "volumetric_constraint")

    def potential(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> Array:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        energy = self.isochoric_value(deformation)
        constraint = self.volumetric_value(deformation)
        mixed = energy + pressure_ * constraint
        if self.bulk_modulus is not None:
            mixed = mixed - 0.5 * pressure_ * pressure_ / self.bulk_modulus
        return mixed

    def constraint(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> Array:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        residual = self.volumetric_value(deformation)
        if self.bulk_modulus is not None:
            residual = residual - pressure_ / self.bulk_modulus
        return residual

    def first_piola(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> Array:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        return jax.grad(lambda value: self.potential(value, pressure_))(deformation)

    def residual(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        return self.first_piola(deformation, pressure_), self.constraint(
            deformation, pressure_
        )

    def block_tangent(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> MixedHyperelasticBlockTangent:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        deformation_deformation = jax.jacfwd(
            lambda value: self.first_piola(value, pressure_)
        )(deformation)
        deformation_pressure = jax.jacfwd(
            lambda value: self.first_piola(deformation, value)
        )(pressure_)
        pressure_deformation = jax.jacfwd(
            lambda value: self.constraint(value, pressure_)
        )(deformation)
        pressure_pressure = jax.jacfwd(lambda value: self.constraint(deformation, value))(
            pressure_
        )
        return MixedHyperelasticBlockTangent(
            deformation_deformation,
            deformation_pressure,
            pressure_deformation,
            pressure_pressure,
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> MixedHyperelasticResponse:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        jacobian = jnp.linalg.det(deformation)
        deformation_bar = isochoric_deformation_gradient(deformation)
        isochoric_energy = _scalar(
            self.isochoric_energy(deformation_bar), "isochoric_energy"
        )
        constraint_value = self.volumetric_value(deformation)
        constraint_residual = constraint_value
        if self.bulk_modulus is not None:
            constraint_residual = constraint_residual - pressure_ / self.bulk_modulus
        mixed_energy, first_piola = jax.value_and_grad(
            lambda value: self.potential(value, pressure_)
        )(deformation)
        pressure_stress = mixed_pressure_first_piola(
            deformation,
            pressure_,
            self.volumetric_constraint,
        )
        isochoric_stress = first_piola - pressure_stress
        pressure_finite = jnp.isfinite(pressure_)
        energy_finite = jnp.isfinite(isochoric_energy) & jnp.isfinite(mixed_energy)
        stress_finite = jnp.all(jnp.isfinite(first_piola)) & jnp.all(
            jnp.isfinite(isochoric_stress)
        )
        constraint_finite = jnp.isfinite(constraint_value) & jnp.isfinite(
            constraint_residual
        )
        finite = (
            jnp.all(jnp.isfinite(deformation))
            & jnp.all(jnp.isfinite(deformation_bar))
            & jnp.isfinite(jacobian)
            & pressure_finite
            & energy_finite
            & stress_finite
            & constraint_finite
        )
        jacobian_valid = jnp.isfinite(jacobian) & (jacobian > self.minimum_jacobian)
        evidence = MixedHyperelasticEvidence(
            jacobian,
            jnp.asarray(self.minimum_jacobian, dtype=jacobian.dtype),
            jacobian_valid,
            pressure_finite,
            energy_finite,
            stress_finite,
            constraint_finite,
            finite,
            finite & jacobian_valid,
        )
        return MixedHyperelasticResponse(
            deformation,
            deformation_bar,
            pressure_,
            jacobian,
            isochoric_energy,
            mixed_energy,
            constraint_value,
            constraint_residual,
            isochoric_stress,
            pressure_stress,
            first_piola,
            evidence,
        )


class MixedHyperelasticModel(StrictModule, NonTrainableState):
    """FE/neural material model retaining one authoritative pointwise law."""

    law: MixedHyperelasticLaw

    def __init__(self, law: MixedHyperelasticLaw, /):
        if not isinstance(law, MixedHyperelasticLaw):
            raise TypeError("law must be MixedHyperelasticLaw.")
        self.law = law

    def first_piola_points(self, deformation: Array, pressure: Array, /) -> Array:
        if deformation.ndim < 2 or deformation.shape[-2] != deformation.shape[-1]:
            raise ValueError("Pointwise deformation gradients must end in square axes.")
        if pressure.shape != deformation.shape[:-2]:
            raise ValueError("Pointwise pressure must match deformation leading axes.")
        dimension = deformation.shape[-1]
        flat_deformation = deformation.reshape((-1, dimension, dimension))
        flat_pressure = pressure.reshape((-1,))
        stress = jax.vmap(self.law.first_piola)(flat_deformation, flat_pressure)
        return stress.reshape(deformation.shape)

    def constraint_points(self, deformation: Array, pressure: Array, /) -> Array:
        if pressure.shape != deformation.shape[:-2]:
            raise ValueError("Pointwise pressure must match deformation leading axes.")
        dimension = deformation.shape[-1]
        flat_deformation = deformation.reshape((-1, dimension, dimension))
        flat_pressure = pressure.reshape((-1,))
        residual = jax.vmap(self.law.constraint)(flat_deformation, flat_pressure)
        return residual.reshape(pressure.shape)


def mixed_hyperelastic_form(
    displacement_field: str,
    pressure_field: str,
    model: MixedHyperelasticModel,
    /,
    *,
    form_id: str = "mixed-hyperelastic-equilibrium",
) -> FiniteElementForm:
    """Build the exact coupled FE weak residual with no hidden penalty term."""

    displacement_name = str(displacement_field)
    pressure_name = str(pressure_field)
    if not displacement_name or not pressure_name or displacement_name == pressure_name:
        raise ValueError("Mixed displacement and pressure field names must be distinct.")
    if not isinstance(model, MixedHyperelasticModel):
        raise TypeError("model must be MixedHyperelasticModel.")

    def displacement_kernel(
        values, gradients, points, weights, basis_values, basis_gradients, context
    ):
        del points, basis_values, context
        displacement_gradient = jnp.swapaxes(jnp.asarray(gradients[0]), -1, -2)
        pressure = jnp.asarray(values[1])
        dimension = displacement_gradient.shape[-1]
        deformation = displacement_gradient + jnp.eye(
            dimension, dtype=displacement_gradient.dtype
        )
        first_piola = model.first_piola_points(deformation, pressure)
        return contract(
            "cq,cqad,cqid->cia",
            weights,
            first_piola,
            basis_gradients,
        )

    def pressure_kernel(
        values, gradients, points, weights, basis_values, basis_gradients, context
    ):
        del points, basis_gradients, context
        displacement_gradient = jnp.swapaxes(jnp.asarray(gradients[0]), -1, -2)
        dimension = displacement_gradient.shape[-1]
        deformation = displacement_gradient + jnp.eye(
            dimension, dtype=displacement_gradient.dtype
        )
        pressure = (
            jnp.zeros(deformation.shape[:-2], dtype=deformation.dtype)
            if model.law.bulk_modulus is None
            else jnp.asarray(values[1])
        )
        constraint_residual = model.constraint_points(deformation, pressure)
        return contract("cq,cq,qi->ci", weights, constraint_residual, basis_values)

    pressure_inputs = (
        (displacement_name,)
        if model.law.bulk_modulus is None
        else (displacement_name, pressure_name)
    )
    return FiniteElementForm(
        form_id,
        (displacement_name, pressure_name),
        (
            CellResidualAction(
                displacement_name,
                (displacement_name, pressure_name),
                displacement_kernel,
                action_id=f"{form_id}:displacement",
            ),
            CellResidualAction(
                pressure_name,
                pressure_inputs,
                pressure_kernel,
                action_id=f"{form_id}:pressure",
            ),
        ),
    )


def prepare_mixed_hyperelastic_problem(
    model: MixedHyperelasticModel,
    plan: MixedFiniteElementConstraintPlan,
    /,
    *,
    initial_state: tuple[ArrayLike, ArrayLike] | None = None,
    args: object = None,
    form_id: str = "mixed-hyperelastic-equilibrium",
) -> PreparedMixedFiniteElementConstraint:
    """Compile and certify one Taylor-Hood/Q2-Q1 mixed hyperelastic root."""

    if not isinstance(model, MixedHyperelasticModel):
        raise TypeError("model must be MixedHyperelasticModel.")
    if not isinstance(plan, MixedFiniteElementConstraintPlan):
        raise TypeError("plan must be MixedFiniteElementConstraintPlan.")
    if model.law.formulation != plan.formulation:
        raise ValueError("Mixed FE plan and material formulation must match.")
    if model.law.bulk_modulus != plan.bulk_modulus:
        raise ValueError("Mixed FE plan and material bulk modulus must match.")
    form = mixed_hyperelastic_form(
        plan.displacement_field,
        plan.pressure_field,
        model,
        form_id=form_id,
    )
    return plan.prepare(form, initial_state=initial_state, args=args)


def _mixed_neural_fields(
    functions: Mapping[str, Any],
    displacement_field: str,
    pressure_field: str,
    /,
) -> None:
    if not isinstance(functions, Mapping):
        raise TypeError("functions must be a mapping of named fields.")
    displacement = str(displacement_field)
    pressure = str(pressure_field)
    if not displacement or not pressure or displacement == pressure:
        raise ValueError("Mixed neural field names must be distinct and non-empty.")
    if displacement not in functions or pressure not in functions:
        raise ValueError(
            "Mixed neural roots must contain displacement and pressure fields."
        )


def _mixed_neural_subspace(
    functions: Mapping[str, Any],
    parameter_subspace: ParameterSubspace,
    displacement_field: str,
    pressure_field: str,
    /,
) -> None:
    if not isinstance(parameter_subspace, ParameterSubspace):
        raise TypeError("parameter_subspace must be ParameterSubspace.")
    parameter_subspace.validate_root(functions)
    for name in (str(displacement_field), str(pressure_field)):
        selected = jax.tree.leaves(
            parameter_subspace.initial[name],
            is_leaf=lambda value: value is None,
        )
        if not any(eqx.is_inexact_array(value) for value in selected):
            raise ValueError(
                f"Mixed neural parameter subspace must select field {name!r}."
            )


def _mixed_neural_gauge(
    law: MixedHyperelasticLaw,
    gauge: PressureGaugePolicy,
    gauge_enforced: bool,
    /,
) -> None:
    if not isinstance(law, MixedHyperelasticLaw):
        raise TypeError("law must be MixedHyperelasticLaw.")
    if not isinstance(gauge, PressureGaugePolicy):
        raise TypeError("gauge must be PressureGaugePolicy.")
    enforced = bool(gauge_enforced)
    if law.bulk_modulus is None:
        if gauge.mode == "none" or not enforced:
            raise ValueError(
                "Exact mixed neural pressure requires an explicitly enforced gauge."
            )
    elif gauge.mode != "none" or enforced:
        raise ValueError("Finite-bulk mixed pressure must not impose a pressure gauge.")


def prepare_mixed_neural_stationarity(
    functions: Mapping[str, Any],
    action: Callable[[Mapping[str, Any], Any, Any], Any],
    parameter_subspace: ParameterSubspace,
    law: MixedHyperelasticLaw,
    gauge: PressureGaugePolicy,
    /,
    *,
    gauge_enforced: bool,
    displacement_field: str = "u",
    pressure_field: str = "p",
    sign: float = 1.0,
    realization: Any = None,
    realization_id: str,
    provenance_id: str,
    problem_id: str = "mixed-neural-functional-stationarity",
) -> PreparedFieldEquilibrium:
    """Prepare mixed neural stationarity on the canonical field root substrate."""

    _mixed_neural_fields(functions, displacement_field, pressure_field)
    _mixed_neural_subspace(
        functions, parameter_subspace, displacement_field, pressure_field
    )
    _mixed_neural_gauge(law, gauge, gauge_enforced)
    return prepare_functional_stationarity(
        functions,
        action,
        parameter_subspace,
        sign=sign,
        realization=realization,
        realization_id=realization_id,
        provenance_id=provenance_id,
        problem_id=problem_id,
    )


def prepare_mixed_neural_virtual_work(
    functions: Mapping[str, Any],
    field_jet: Callable[[Mapping[str, Any], Any, Any], PyTree[Any]],
    virtual_work: Callable[[Mapping[str, Any], PyTree[Any], Any, Any], PyTree[Any]],
    parameter_subspace: ParameterSubspace,
    realization: Any,
    law: MixedHyperelasticLaw,
    gauge: PressureGaugePolicy,
    /,
    *,
    gauge_enforced: bool,
    displacement_field: str = "u",
    pressure_field: str = "p",
    realization_id: str,
    provenance_id: str,
    problem_id: str = "mixed-neural-virtual-work",
) -> PreparedFieldEquilibrium:
    """Prepare mixed virtual work while retaining its complete nonsymmetric tangent."""

    _mixed_neural_fields(functions, displacement_field, pressure_field)
    _mixed_neural_subspace(
        functions, parameter_subspace, displacement_field, pressure_field
    )
    _mixed_neural_gauge(law, gauge, gauge_enforced)
    return prepare_virtual_work_equilibrium(
        functions,
        field_jet,
        virtual_work,
        parameter_subspace,
        realization,
        realization_id=realization_id,
        provenance_id=provenance_id,
        problem_id=problem_id,
    )


class MixedAugmentedLagrangianState(StrictModule):
    deformation_gradient: Array
    pressure: Array
    penalty: Array
    constraint_norm: Array
    outer_iteration: Array


class MixedAugmentedLagrangianEvidence(StrictModule):
    previous_constraint_norm: Array
    candidate_constraint_norm: Array
    inner_successful: Array
    finite: Array
    jacobian_valid: Array
    constraint_reduced: Array
    penalty_increased: Array
    converged: Array
    accepted: Array
    rollback_applied: Array


class MixedAugmentedLagrangianResult(StrictModule):
    previous: MixedAugmentedLagrangianState
    candidate: MixedAugmentedLagrangianState
    accepted_state: MixedAugmentedLagrangianState
    candidate_response: MixedHyperelasticResponse
    evidence: MixedAugmentedLagrangianEvidence


class MixedAugmentedLagrangianPlan(StrictModule, NonTrainableState):
    """Transactional outer multiplier update for an exact mixed constraint."""

    law: MixedHyperelasticLaw
    initial_penalty: float = eqx.field(static=True)
    penalty_growth: float = eqx.field(static=True)
    maximum_penalty: float = eqx.field(static=True)
    constraint_reduction: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        law: MixedHyperelasticLaw,
        /,
        *,
        initial_penalty: float,
        penalty_growth: float = 10.0,
        maximum_penalty: float = 1.0e12,
        constraint_reduction: float = 0.25,
        constraint_tolerance: float = 1.0e-8,
    ):
        if not isinstance(law, MixedHyperelasticLaw):
            raise TypeError("law must be MixedHyperelasticLaw.")
        if law.bulk_modulus is not None:
            raise ValueError("Augmented-Lagrangian outer updates require an exact law.")
        initial = float(initial_penalty)
        growth = float(penalty_growth)
        maximum = float(maximum_penalty)
        reduction = float(constraint_reduction)
        tolerance = float(constraint_tolerance)
        if (
            not isfinite(initial)
            or initial <= 0.0
            or not isfinite(growth)
            or growth <= 1.0
            or not isfinite(maximum)
            or maximum < initial
            or not isfinite(reduction)
            or reduction <= 0.0
            or reduction >= 1.0
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Augmented-Lagrangian policy values are invalid.")
        self.law = law
        self.initial_penalty = initial
        self.penalty_growth = growth
        self.maximum_penalty = maximum
        self.constraint_reduction = reduction
        self.constraint_tolerance = tolerance

    def inner_response(
        self,
        state: MixedAugmentedLagrangianState,
        deformation_gradient: ArrayLike,
        /,
    ) -> MixedHyperelasticResponse:
        """Evaluate the primal AL stationarity with p + μg as pressure."""

        if not isinstance(state, MixedAugmentedLagrangianState):
            raise TypeError("state must be MixedAugmentedLagrangianState.")
        deformation = _deformation_gradient(deformation_gradient)
        if deformation.shape != state.deformation_gradient.shape:
            raise ValueError("AL inner and accepted deformation gradients must match.")
        effective_pressure = state.pressure + state.penalty * self.law.volumetric_value(
            deformation
        )
        return self.law.evaluate(deformation, effective_pressure)

    def initialize(
        self,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike = 0.0,
        /,
    ) -> MixedAugmentedLagrangianState:
        deformation = _deformation_gradient(deformation_gradient)
        pressure_ = _pressure(pressure, deformation.dtype)
        constraint_norm = jnp.abs(self.law.volumetric_value(deformation))
        return MixedAugmentedLagrangianState(
            deformation,
            pressure_,
            jnp.asarray(self.initial_penalty, dtype=deformation.dtype),
            constraint_norm,
            jnp.asarray(0, dtype=jnp.int32),
        )

    def advance(
        self,
        state: MixedAugmentedLagrangianState,
        candidate_deformation_gradient: ArrayLike,
        /,
        *,
        inner_successful: ArrayLike,
    ) -> MixedAugmentedLagrangianResult:
        if not isinstance(state, MixedAugmentedLagrangianState):
            raise TypeError("state must be MixedAugmentedLagrangianState.")
        deformation = _deformation_gradient(candidate_deformation_gradient)
        if deformation.shape != state.deformation_gradient.shape:
            raise ValueError("Candidate and accepted deformation gradients must match.")
        successful = jnp.asarray(inner_successful, dtype=bool)
        if successful.shape != ():
            raise ValueError("inner_successful must be scalar.")
        constraint = self.law.volumetric_value(deformation)
        candidate_pressure = state.pressure + state.penalty * constraint
        candidate_norm = jnp.abs(constraint)
        reduction_target = jnp.maximum(
            jnp.asarray(self.constraint_tolerance, dtype=candidate_norm.dtype),
            jnp.asarray(self.constraint_reduction, dtype=candidate_norm.dtype)
            * state.constraint_norm,
        )
        constraint_reduced = candidate_norm <= reduction_target
        increase = ~constraint_reduced
        candidate_penalty = jnp.where(
            increase,
            jnp.minimum(
                state.penalty * self.penalty_growth,
                jnp.asarray(self.maximum_penalty, dtype=state.penalty.dtype),
            ),
            state.penalty,
        )
        response = self.law.evaluate(deformation, candidate_pressure)
        finite = (
            response.evidence.finite
            & jnp.isfinite(candidate_norm)
            & jnp.isfinite(candidate_penalty)
        )
        accepted = successful & finite & response.evidence.jacobian_valid
        candidate = MixedAugmentedLagrangianState(
            deformation,
            candidate_pressure,
            candidate_penalty,
            candidate_norm,
            state.outer_iteration + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted_state = MixedAugmentedLagrangianState(
            jnp.where(
                accepted, candidate.deformation_gradient, state.deformation_gradient
            ),
            jnp.where(accepted, candidate.pressure, state.pressure),
            jnp.where(accepted, candidate.penalty, state.penalty),
            jnp.where(accepted, candidate.constraint_norm, state.constraint_norm),
            jnp.where(accepted, candidate.outer_iteration, state.outer_iteration),
        )
        evidence = MixedAugmentedLagrangianEvidence(
            state.constraint_norm,
            candidate_norm,
            successful,
            finite,
            response.evidence.jacobian_valid,
            constraint_reduced,
            accepted & increase,
            candidate_norm <= self.constraint_tolerance,
            accepted,
            ~accepted,
        )
        return MixedAugmentedLagrangianResult(
            state,
            candidate,
            accepted_state,
            response,
            evidence,
        )


__all__ = [
    "IsochoricEnergy",
    "MixedAugmentedLagrangianEvidence",
    "MixedAugmentedLagrangianPlan",
    "MixedAugmentedLagrangianResult",
    "MixedAugmentedLagrangianState",
    "MixedHyperelasticBlockTangent",
    "MixedHyperelasticEvidence",
    "MixedHyperelasticFormulation",
    "MixedHyperelasticLaw",
    "MixedHyperelasticModel",
    "MixedHyperelasticResponse",
    "VolumetricConstraint",
    "isochoric_deformation_gradient",
    "mixed_hyperelastic_form",
    "mixed_pressure_first_piola",
    "prepare_mixed_hyperelastic_problem",
    "prepare_mixed_neural_stationarity",
    "prepare_mixed_neural_virtual_work",
]

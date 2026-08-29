#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import FiniteElementDiscretization
from ...equations import CellResidualAction, ConstitutiveResponse, FiniteElementForm
from ...equations.fem import FiniteElementAuxiliaryEvaluation, LocalImplicitMaterial
from ...linalg import SmallLinearSolvePlan, solve_small_linear


class CrystalSlipSystem(StrictModule, NonTrainableState):
    direction: Array
    normal: Array
    schmid: Array

    def __init__(self, direction: ArrayLike, normal: ArrayLike, /):
        direction_ = jnp.asarray(direction)
        normal_ = jnp.asarray(normal)
        if direction_.shape != (3,) or normal_.shape != (3,):
            raise ValueError("Crystal slip direction and normal must be 3-vectors.")
        direction_ = direction_ / jnp.sqrt(jnp.sum(direction_**2))
        normal_ = normal_ / jnp.sqrt(jnp.sum(normal_**2))
        if float(jnp.abs(jnp.dot(direction_, normal_))) > 1.0e-10:
            raise ValueError("Crystal slip direction and normal must be orthogonal.")
        self.direction = direction_
        self.normal = normal_
        self.schmid = jnp.outer(direction_, normal_)


class CrystalPlasticityParameters(StrictModule, NonTrainableState):
    shear_modulus: Array
    bulk_modulus: Array
    reference_rate: Array
    rate_sensitivity: Array
    hardening_modulus: Array
    initial_strength: Array
    maximum_slip_increment: Array

    def __init__(
        self,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        reference_rate: ArrayLike,
        rate_sensitivity: ArrayLike,
        hardening_modulus: ArrayLike,
        initial_strength: ArrayLike,
        /,
        *,
        maximum_slip_increment: ArrayLike = 0.2,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                shear_modulus,
                bulk_modulus,
                reference_rate,
                rate_sensitivity,
                hardening_modulus,
                initial_strength,
                maximum_slip_increment,
            )
        )
        if any(value.shape != () or not bool(jnp.isfinite(value)) for value in values):
            raise ValueError("CPFEM parameters must be finite scalars.")
        if any(value <= 0.0 for value in values[:4] + values[5:]):
            raise ValueError(
                "CPFEM elastic, rate, strength, and bound data are positive."
            )
        if values[4] < 0.0:
            raise ValueError("CPFEM hardening modulus must be nonnegative.")
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.reference_rate,
            self.rate_sensitivity,
            self.hardening_modulus,
            self.initial_strength,
            self.maximum_slip_increment,
        ) = values


class CrystalPlasticityState(StrictModule):
    plastic_deformation: Array
    strengths: Array
    accumulated_slip: Array

    def __init__(
        self,
        plastic_deformation: ArrayLike,
        strengths: ArrayLike,
        accumulated_slip: ArrayLike,
        /,
    ):
        plastic = jnp.asarray(plastic_deformation)
        strengths_ = jnp.asarray(strengths)
        accumulated = jnp.asarray(accumulated_slip)
        if plastic.shape != (3, 3) or strengths_.ndim != 1 or accumulated.shape != ():
            raise ValueError("CPFEM state shapes are invalid.")
        self.plastic_deformation = plastic
        self.strengths = strengths_
        self.accumulated_slip = accumulated

    def pack(self, /) -> Array:
        return jnp.concatenate(
            (
                self.plastic_deformation.reshape((-1,)),
                self.strengths,
                self.accumulated_slip[None],
            )
        )

    @classmethod
    def unpack(cls, value: ArrayLike, slip_count: int, /) -> CrystalPlasticityState:
        value_ = jnp.asarray(value)
        expected = 10 + int(slip_count)
        if value_.shape != (expected,):
            raise ValueError("Packed CPFEM state has an invalid shape.")
        return cls(
            value_[:9].reshape((3, 3)),
            value_[9 : 9 + slip_count],
            value_[-1],
        )


class CrystalPlasticityUpdate(StrictModule):
    first_piola: Array
    state: CrystalPlasticityState
    slip_increment: Array
    converged: Array
    admissible: Array
    suggested_step_factor: Array


class CrystalPlasticityModel(StrictModule, NonTrainableState):
    slip_systems: tuple[CrystalSlipSystem, ...]
    parameters: CrystalPlasticityParameters
    orientation: Array
    inverse_plan: SmallLinearSolvePlan

    def __init__(
        self,
        slip_systems: Sequence[CrystalSlipSystem],
        parameters: CrystalPlasticityParameters,
        /,
        *,
        orientation: ArrayLike | None = None,
    ):
        systems = tuple(slip_systems)
        if not systems or not all(
            isinstance(value, CrystalSlipSystem) for value in systems
        ):
            raise ValueError("CPFEM requires one or more CrystalSlipSystem values.")
        if not isinstance(parameters, CrystalPlasticityParameters):
            raise TypeError("parameters must be CrystalPlasticityParameters.")
        orientation_ = jnp.eye(3) if orientation is None else jnp.asarray(orientation)
        if orientation_.shape != (3, 3):
            raise ValueError("Crystal orientation must be a 3x3 rotation.")
        orthogonality = orientation_.T @ orientation_ - jnp.eye(3)
        if float(jnp.max(jnp.abs(orthogonality))) > 1.0e-10:
            raise ValueError("Crystal orientation must be orthogonal.")
        self.slip_systems = systems
        self.parameters = parameters
        self.orientation = orientation_
        self.inverse_plan = SmallLinearSolvePlan(3)

    @property
    def slip_count(self) -> int:
        return len(self.slip_systems)

    def initial_state(self, /) -> CrystalPlasticityState:
        return CrystalPlasticityState(
            jnp.eye(3),
            jnp.full((self.slip_count,), self.parameters.initial_strength),
            jnp.asarray(0.0),
        )

    def _inverse(self, matrix: Array, /) -> tuple[Array, Array]:
        result = solve_small_linear(self.inverse_plan, matrix, jnp.eye(3))
        return result.value, result.successful

    def _stress(self, deformation: Array, plastic: Array, /):
        plastic_inverse, plastic_ok = self._inverse(plastic)
        elastic = deformation @ plastic_inverse
        elastic_inverse, elastic_ok = self._inverse(elastic)
        determinant = jnp.linalg.det(elastic)
        logarithm = jnp.log(jnp.maximum(determinant, jnp.finfo(elastic.dtype).tiny))
        elastic_piola = (
            self.parameters.shear_modulus * (elastic - elastic_inverse.T)
            + self.parameters.bulk_modulus * logarithm * elastic_inverse.T
        )
        first_piola = elastic_piola @ plastic_inverse.T
        kirchhoff = elastic_piola @ elastic.T
        return first_piola, kirchhoff, determinant, plastic_ok & elastic_ok

    def update(
        self,
        deformation_gradient: ArrayLike,
        committed_state: CrystalPlasticityState,
        step_size: ArrayLike,
        /,
    ) -> CrystalPlasticityUpdate:
        deformation = jnp.asarray(deformation_gradient)
        dt = jnp.asarray(step_size)
        if deformation.shape != (3, 3) or dt.shape != ():
            raise ValueError("CPFEM deformation and step-size shapes are invalid.")
        if committed_state.strengths.shape != (self.slip_count,):
            raise ValueError("CPFEM committed strengths do not match slip systems.")
        rotated_schmid = jnp.stack(
            tuple(
                self.orientation @ system.schmid @ self.orientation.T
                for system in self.slip_systems
            )
        )
        committed_pack = committed_state.pack()

        def state_from_increment(increment):
            plastic_generator = oe.contract("a,aij->ij", increment, rotated_schmid)
            plastic = (
                jsp_linalg.expm(plastic_generator) @ committed_state.plastic_deformation
            )
            strengths = (
                committed_state.strengths
                + self.parameters.hardening_modulus * jnp.sum(jnp.abs(increment))
            )
            accumulated = committed_state.accumulated_slip + jnp.sum(jnp.abs(increment))
            return CrystalPlasticityState(plastic, strengths, accumulated)

        def residual(increment, args):
            state = state_from_increment(increment)
            _, kirchhoff, _, _ = self._stress(deformation, state.plastic_deformation)
            resolved = oe.contract("aij,ij->a", rotated_schmid, kirchhoff)
            ratio = jnp.abs(resolved) / state.strengths
            rate = (
                self.parameters.reference_rate
                * jnp.sign(resolved)
                * ratio ** (1.0 / self.parameters.rate_sensitivity)
            )
            return increment - dt * rate

        def response(increment, args):
            state = state_from_increment(increment)
            first_piola, _, determinant, invertible = self._stress(
                deformation, state.plastic_deformation
            )
            admissible = (
                invertible
                & jnp.isfinite(determinant)
                & (determinant > 0.0)
                & (jnp.max(jnp.abs(increment)) <= self.parameters.maximum_slip_increment)
            )
            return ConstitutiveResponse(
                first_piola,
                state.pack(),
                diagnostics={
                    "admissible": admissible,
                    "elastic_determinant": determinant,
                    "maximum_slip_increment": jnp.max(jnp.abs(increment)),
                    "committed_state_norm": jnp.sqrt(jnp.sum(committed_pack**2)),
                },
            )

        material = LocalImplicitMaterial(
            residual,
            response,
            state_shape=(self.slip_count,),
            model_id="finite-strain-crystal-slip",
            max_steps=25,
            tolerance=1.0e-10,
        )
        initial = jnp.zeros((self.slip_count,), dtype=deformation.dtype)
        constitutive = material.evaluate(initial, None)
        state = CrystalPlasticityState.unpack(constitutive.trial_state, self.slip_count)
        maximum_increment = constitutive.diagnostics["maximum_slip_increment"]
        admissible = constitutive.diagnostics["admissible"]
        converged = constitutive.diagnostics["converged"] & admissible
        factor = jnp.minimum(
            1.0,
            self.parameters.maximum_slip_increment
            / jnp.maximum(maximum_increment, jnp.finfo(deformation.dtype).tiny),
        )
        return CrystalPlasticityUpdate(
            first_piola=constitutive.response,
            state=state,
            slip_increment=material.solve(initial, None),
            converged=converged,
            admissible=admissible,
            suggested_step_factor=factor,
        )


def cpfem_equilibrium_form(
    discretization: FiniteElementDiscretization,
    field_name: str,
    model: CrystalPlasticityModel,
    committed_states: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    form_id: str = "cpfem-equilibrium",
) -> FiniteElementForm:
    """Build a total-Lagrangian equilibrium residual with implicit local slips."""

    if not isinstance(model, CrystalPlasticityModel):
        raise TypeError("model must be CrystalPlasticityModel.")
    committed = jnp.asarray(committed_states)
    dt = jnp.asarray(step_size)
    if committed.ndim != 3 or committed.shape[-1] != 10 + model.slip_count:
        raise ValueError(
            "CPFEM committed states require (cells, quadrature, state_width)."
        )
    if dt.shape != () or not bool(jnp.isfinite(dt)) or dt <= 0.0:
        raise ValueError("CPFEM step_size must be one positive finite scalar.")
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    if len(discretization.mesh.blocks) != 1:
        raise ValueError("CPFEM equilibrium currently requires one cell block.")
    field_index = discretization._field_index(field_name)

    def point_update(deformation_, packed_state):
        state = CrystalPlasticityState.unpack(packed_state, model.slip_count)
        update = model.update(deformation_, state, dt)
        return (
            update.first_piola,
            update.state.pack(),
            update.converged,
            update.admissible,
            update.suggested_step_factor,
        )

    def residual(values, gradients, points, weights, test_basis, test_gradients, context):
        deformation = jnp.eye(3) + gradients[0]
        if deformation.shape[:2] != committed.shape[:2]:
            raise ValueError("CPFEM material state does not match workset quadrature.")
        flat_deformation = deformation.reshape((-1, 3, 3))
        flat_state = committed.reshape((-1, committed.shape[-1]))

        def point_stress(deformation_, packed_state):
            return point_update(deformation_, packed_state)[0]

        stress = jax.vmap(point_stress)(flat_deformation, flat_state).reshape(
            deformation.shape
        )
        return oe.contract(
            "cq,cqib,cqab->cia",
            weights,
            test_gradients,
            stress,
        )

    def auxiliary(state, context):
        displacement = jnp.asarray(state)
        geometry = discretization.block_geometries[field_index][0]
        dofs = discretization.dof_maps[field_index].cell_dofs[0]
        orientation = discretization.dof_maps[field_index].orientations[0]
        local = displacement[dofs] * orientation[..., None]
        gradient = oe.contract(
            "cqid,cia->cqad",
            geometry.physical_gradients,
            local,
        )
        deformation = jnp.eye(3) + gradient
        if deformation.shape[:2] != committed.shape[:2]:
            raise ValueError(
                "CPFEM auxiliary state must match prepared workset quadrature."
            )
        outputs = jax.vmap(point_update)(
            deformation.reshape((-1, 3, 3)),
            committed.reshape((-1, committed.shape[-1])),
        )
        trial = outputs[1].reshape(committed.shape)
        return FiniteElementAuxiliaryEvaluation(
            trial,
            successful=jnp.all(outputs[2]),
            admissible=jnp.all(outputs[3]),
            retry_requested=~jnp.all(outputs[2] & outputs[3]),
            suggested_step=dt * jnp.min(outputs[4]),
            diagnostics={
                "converged_points": jnp.sum(outputs[2]),
                "admissible_points": jnp.sum(outputs[3]),
            },
        )

    return FiniteElementForm(
        form_id,
        field_name,
        (
            CellResidualAction(
                field_name,
                (field_name,),
                residual,
                action_id="cpfem-internal-force",
            ),
        ),
        auxiliary_evaluator=auxiliary,
        auxiliary_id="cpfem-material-worksets",
    )


__all__ = [
    "CrystalPlasticityModel",
    "CrystalPlasticityParameters",
    "CrystalPlasticityState",
    "CrystalPlasticityUpdate",
    "CrystalSlipSystem",
    "cpfem_equilibrium_form",
]

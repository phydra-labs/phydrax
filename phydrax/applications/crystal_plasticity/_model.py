#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import FiniteElementDiscretization, IntegrationDomain
from ...equations import (
    CellResidualAction,
    ConstitutiveResponse,
    FiniteElementForm,
    MaterialCheckpointPayload,
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from ...equations.fem import FiniteElementAuxiliaryEvaluation, LocalImplicitMaterial
from ...integration import (
    GaussLegendreRule,
    reference_rule_data,
    ReferenceHexahedronRule,
    ReferencePrismRule,
    ReferencePyramidRule,
    ReferenceRule,
    ReferenceTetrahedronRule,
)
from ...linalg import inverse_small_linear, SmallLinearSolvePlan


def _real_array(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(result.dtype, jnp.floating):
        result = result.astype(float)
    return result


def _finite_scalar(value: ArrayLike, name: str, /) -> Array:
    result = _real_array(value, name)
    if result.shape != () or not bool(jnp.isfinite(result)):
        raise ValueError(f"{name} must be one finite scalar.")
    return result


def _host_rotation_tolerance(value: np.ndarray, /) -> float:
    dtype = (
        value.dtype if np.issubdtype(value.dtype, np.floating) else np.dtype(np.float64)
    )
    return max(1.0e-8, 256.0 * float(np.finfo(dtype).eps))


def _crystal_rotation(value: ArrayLike, /) -> Array:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError("crystal_to_sample must be real-valued.")
    rotation = np.asarray(raw, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("crystal_to_sample must be one finite 3x3 rotation.")
    orthogonality = rotation.T @ rotation - np.eye(3)
    determinant = float(np.linalg.det(rotation))
    tolerance = _host_rotation_tolerance(raw)
    if np.max(np.abs(orthogonality)) > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            "crystal_to_sample must belong to SO(3); reflections are invalid."
        )
    return jnp.asarray(rotation)


def _crystal_rotation_field(
    value: ArrayLike,
    site_shape: tuple[int, int],
    /,
) -> Array:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError("crystal_to_sample must be real-valued.")
    expected = site_shape + (3, 3)
    if raw.shape == (3, 3):
        _crystal_rotation(raw)
        rotation = np.broadcast_to(np.asarray(raw, dtype=float), expected).copy()
    elif raw.shape == expected:
        rotation = np.asarray(raw, dtype=float)
    else:
        raise ValueError(
            "Routed crystal_to_sample must be one 3x3 rotation or have "
            f"exact site shape {expected}."
        )
    tolerance = _host_rotation_tolerance(raw)
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    determinant = np.linalg.det(rotation)
    invalid = (
        ~np.all(np.isfinite(rotation), axis=(-2, -1))
        | (np.max(np.abs(gram - np.eye(3)), axis=(-2, -1)) > tolerance)
        | (np.abs(determinant - 1.0) > tolerance)
    )
    if np.any(invalid):
        raise ValueError(
            "Every routed crystal_to_sample value must belong to SO(3); "
            "reflections are invalid."
        )
    return jnp.asarray(rotation)


def _runtime_step(value: ArrayLike, dtype: Any, /) -> Array:
    step = _real_array(value, "step_size")
    if step.shape != ():
        raise ValueError("CPFEM step_size must be scalar.")
    step = step.astype(dtype)
    return eqx.error_if(
        step,
        ~jnp.isfinite(step) | (step <= 0.0),
        "CPFEM step_size must be one positive finite scalar.",
    )


def _runtime_crystal_rotation(
    value: ArrayLike,
    dtype: Any,
    inverse_plan: SmallLinearSolvePlan,
    /,
) -> Array:
    rotation = _real_array(value, "crystal_to_sample")
    if rotation.shape != (3, 3):
        raise ValueError("crystal_to_sample must be one 3x3 rotation.")
    rotation = rotation.astype(dtype)
    inverse = inverse_small_linear(inverse_plan, rotation)
    orthogonality = ein.contract("ki,kj->ij", rotation, rotation) - jnp.eye(
        3, dtype=rotation.dtype
    )
    tolerance = jnp.asarray(
        max(1.0e-8, 256.0 * float(jnp.finfo(rotation.dtype).eps)),
        dtype=rotation.dtype,
    )
    invalid = (
        ~jnp.all(jnp.isfinite(rotation))
        | ~inverse.successful
        | (jnp.max(jnp.abs(orthogonality)) > tolerance)
        | (jnp.abs(inverse.determinant - 1.0) > tolerance)
    )
    return eqx.error_if(
        rotation,
        invalid,
        "crystal_to_sample must belong to SO(3); reflections are invalid.",
    )


class CrystalSlipSystem(StrictModule, NonTrainableState):
    """One normalized crystallographic slip direction and plane normal."""

    direction: Array
    normal: Array
    schmid: Array
    system_id: str = eqx.field(static=True)

    def __init__(self, direction: ArrayLike, normal: ArrayLike, /):
        direction_raw = np.asarray(direction)
        normal_raw = np.asarray(normal)
        if np.iscomplexobj(direction_raw) or np.iscomplexobj(normal_raw):
            raise TypeError("Crystal slip direction and normal must be real-valued.")
        direction_host = np.asarray(direction_raw, dtype=float)
        normal_host = np.asarray(normal_raw, dtype=float)
        if (
            direction_host.shape != (3,)
            or normal_host.shape != (3,)
            or not np.all(np.isfinite(direction_host))
            or not np.all(np.isfinite(normal_host))
        ):
            raise ValueError(
                "Crystal slip direction and normal must be finite 3-vectors."
            )
        direction_norm = float(np.sqrt(np.sum(direction_host**2)))
        normal_norm = float(np.sqrt(np.sum(normal_host**2)))
        if direction_norm <= 0.0 or normal_norm <= 0.0:
            raise ValueError("Crystal slip direction and normal must be nonzero.")
        direction_host = direction_host / direction_norm
        normal_host = normal_host / normal_norm
        if abs(float(np.dot(direction_host, normal_host))) > 1.0e-10:
            raise ValueError("Crystal slip direction and normal must be orthogonal.")
        direction_ = jnp.asarray(direction_host)
        normal_ = jnp.asarray(normal_host)
        schmid = jnp.asarray(np.outer(direction_host, normal_host))
        self.direction = direction_
        self.normal = normal_
        self.schmid = schmid
        self.system_id = canonical_fingerprint(
            {
                "kind": "crystal-slip-system",
                "direction": array_tree_fingerprint(direction_host),
                "normal": array_tree_fingerprint(normal_host),
            }
        )


class CrystalPlasticityParameters(StrictModule, NonTrainableState):
    """Finite-strain elasticity, viscoplastic flow, and isotropic hardening data."""

    shear_modulus: Array
    bulk_modulus: Array
    reference_rate: Array
    rate_sensitivity: Array
    hardening_modulus: Array
    initial_strength: Array
    maximum_slip_increment: Array
    parameters_id: str = eqx.field(static=True)

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
            _finite_scalar(value, name)
            for value, name in zip(
                (
                    shear_modulus,
                    bulk_modulus,
                    reference_rate,
                    rate_sensitivity,
                    hardening_modulus,
                    initial_strength,
                    maximum_slip_increment,
                ),
                (
                    "shear_modulus",
                    "bulk_modulus",
                    "reference_rate",
                    "rate_sensitivity",
                    "hardening_modulus",
                    "initial_strength",
                    "maximum_slip_increment",
                ),
                strict=True,
            )
        )
        if any(bool(value <= 0.0) for value in values[:4] + values[5:]):
            raise ValueError(
                "CPFEM elastic, rate, strength, and increment-bound data "
                "must be positive."
            )
        if bool(values[4] < 0.0):
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
        self.parameters_id = canonical_fingerprint(
            {
                "kind": "crystal-plasticity-parameters",
                "values": array_tree_fingerprint(values),
            }
        )


class CrystalPlasticityState(StrictModule):
    """Plastic deformation, current slip strengths, and total accumulated slip."""

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
        plastic = _real_array(plastic_deformation, "plastic_deformation")
        strengths_ = _real_array(strengths, "strengths")
        accumulated = _real_array(accumulated_slip, "accumulated_slip")
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
    """Local candidate with separate convergence and admissibility evidence."""

    first_piola: Array
    state: CrystalPlasticityState
    slip_increment: Array
    converged: Array
    admissible: Array
    suggested_step_factor: Array
    elastic_deformation: Array
    elastic_determinant: Array
    plastic_determinant: Array
    elastic_energy: Array
    hardening_energy: Array
    free_energy: Array
    plastic_work: Array
    incremental_dissipation: Array
    thermodynamic_admissible: Array
    residual_norm: Array

    @property
    def accepted(self) -> Array:
        return self.converged & self.admissible


class CrystalPlasticityModel(StrictModule, NonTrainableState):
    """Finite-strain multiplicative crystal law independent of specimen orientation."""

    slip_systems: tuple[CrystalSlipSystem, ...]
    parameters: CrystalPlasticityParameters
    inverse_plan: SmallLinearSolvePlan
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        slip_systems: Sequence[CrystalSlipSystem],
        parameters: CrystalPlasticityParameters,
        /,
    ):
        systems = tuple(slip_systems)
        if not systems or not all(
            isinstance(value, CrystalSlipSystem) for value in systems
        ):
            raise ValueError("CPFEM requires one or more CrystalSlipSystem values.")
        if not isinstance(parameters, CrystalPlasticityParameters):
            raise TypeError("parameters must be CrystalPlasticityParameters.")
        for index, left in enumerate(systems):
            left_schmid = np.asarray(left.schmid)
            for right in systems[:index]:
                right_schmid = np.asarray(right.schmid)
                equivalent = np.allclose(
                    left_schmid, right_schmid, atol=1.0e-10, rtol=0.0
                ) or np.allclose(left_schmid, -right_schmid, atol=1.0e-10, rtol=0.0)
                if equivalent:
                    raise ValueError(
                        "CPFEM slip systems must be distinct up to sign convention."
                    )
        self.slip_systems = systems
        self.parameters = parameters
        self.inverse_plan = SmallLinearSolvePlan(3)
        self.model_id = canonical_fingerprint(
            {
                "kind": "finite-strain-crystal-plasticity-model",
                "slip_systems": [system.system_id for system in systems],
                "parameters": parameters.parameters_id,
            }
        )

    @property
    def slip_count(self) -> int:
        return len(self.slip_systems)

    @property
    def state_width(self) -> int:
        return 10 + self.slip_count

    def initial_state(self, /, *, dtype: Any | None = None) -> CrystalPlasticityState:
        parameter_dtype = self.parameters.initial_strength.dtype
        resolved_dtype = parameter_dtype if dtype is None else jnp.dtype(dtype)
        return CrystalPlasticityState(
            jnp.eye(3, dtype=resolved_dtype),
            jnp.full(
                (self.slip_count,),
                self.parameters.initial_strength,
                dtype=resolved_dtype,
            ),
            jnp.asarray(0.0, dtype=resolved_dtype),
        )

    def _validate_state_shape(self, state: CrystalPlasticityState, /) -> None:
        if not isinstance(state, CrystalPlasticityState):
            raise TypeError("committed_state must be CrystalPlasticityState.")
        if state.strengths.shape != (self.slip_count,):
            raise ValueError("CPFEM committed strengths do not match slip systems.")

    def _stress(self, deformation: Array, plastic: Array, /):
        plastic_solve = inverse_small_linear(self.inverse_plan, plastic)
        elastic = ein.contract("ij,jk->ik", deformation, plastic_solve.value)
        elastic_solve = inverse_small_linear(self.inverse_plan, elastic)
        elastic_determinant = elastic_solve.determinant
        safe_determinant = jnp.maximum(
            elastic_determinant, jnp.finfo(elastic.real.dtype).tiny
        )
        logarithm = jnp.log(safe_determinant)
        elastic_piola = self.parameters.shear_modulus * (
            elastic - jnp.swapaxes(elastic_solve.value, -1, -2)
        ) + self.parameters.bulk_modulus * logarithm * jnp.swapaxes(
            elastic_solve.value, -1, -2
        )
        first_piola = ein.contract("ij,kj->ik", elastic_piola, plastic_solve.value)
        mandel = ein.contract("ki,kj->ij", elastic, elastic_piola)
        successful = (
            plastic_solve.successful
            & elastic_solve.successful
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.isfinite(logarithm)
        )
        return (
            first_piola,
            mandel,
            elastic,
            elastic_determinant,
            plastic_solve.determinant,
            logarithm,
            successful,
        )

    def _elastic_energy(self, elastic: Array, logarithm: Array, /) -> Array:
        return (
            0.5
            * self.parameters.shear_modulus
            * (ein.contract("ij,ij->", elastic, elastic) - 3.0 - 2.0 * logarithm)
            + 0.5 * self.parameters.bulk_modulus * logarithm**2
        )

    def _hardening_energy(self, accumulated_slip: Array, /) -> Array:
        return 0.5 * self.parameters.hardening_modulus * accumulated_slip**2

    def free_energy(
        self,
        deformation_gradient: ArrayLike,
        state: CrystalPlasticityState,
        /,
    ) -> Array:
        """Stored elastic plus isotropic-hardening energy for a fixed material state."""

        deformation = _real_array(deformation_gradient, "deformation_gradient")
        self._validate_state_shape(state)
        if deformation.shape != (3, 3):
            raise ValueError("CPFEM deformation gradient must be 3x3.")
        stress = self._stress(deformation, state.plastic_deformation)
        return self._elastic_energy(stress[2], stress[5]) + self._hardening_energy(
            state.accumulated_slip
        )

    def first_piola(
        self,
        deformation_gradient: ArrayLike,
        state: CrystalPlasticityState,
        /,
    ) -> Array:
        """First-Piola stress at fixed internal state, conjugate to deformation."""

        deformation = _real_array(deformation_gradient, "deformation_gradient")
        self._validate_state_shape(state)
        if deformation.shape != (3, 3):
            raise ValueError("CPFEM deformation gradient must be 3x3.")
        return self._stress(deformation, state.plastic_deformation)[0]

    def update(
        self,
        deformation_gradient: ArrayLike,
        committed_state: CrystalPlasticityState,
        crystal_to_sample: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> CrystalPlasticityUpdate:
        """Integrate one point using one explicit crystal-to-sample SO(3) rotation."""

        deformation = _real_array(deformation_gradient, "deformation_gradient")
        dt = _runtime_step(step_size, deformation.dtype)
        self._validate_state_shape(committed_state)
        if deformation.shape != (3, 3):
            raise ValueError("CPFEM deformation gradient must be 3x3.")
        orientation = _runtime_crystal_rotation(
            crystal_to_sample, deformation.dtype, self.inverse_plan
        )
        rotated_schmid = jnp.stack(
            tuple(
                ein.contract(
                    "ij,jk,lk->il",
                    orientation,
                    system.schmid.astype(deformation.dtype),
                    orientation,
                )
                for system in self.slip_systems
            )
        )
        committed_hardening = self._hardening_energy(committed_state.accumulated_slip)

        def state_from_increment(increment):
            total_increment = jnp.sum(jnp.abs(increment))
            plastic_generator = ein.contract("a,aij->ij", increment, rotated_schmid)
            plastic = ein.contract(
                "ij,jk->ik",
                jsp_linalg.expm(plastic_generator),
                committed_state.plastic_deformation,
            )
            accumulated = committed_state.accumulated_slip + total_increment
            strengths = jnp.full_like(
                committed_state.strengths,
                self.parameters.initial_strength
                + self.parameters.hardening_modulus * accumulated,
            )
            return CrystalPlasticityState(plastic, strengths, accumulated)

        def resolved_shear(state):
            stress = self._stress(deformation, state.plastic_deformation)
            resolved = ein.contract("aij,ij->a", rotated_schmid, stress[1])
            return stress, resolved

        def residual(increment, args):
            del args
            state = state_from_increment(increment)
            _, resolved = resolved_shear(state)
            hardening_force = self.parameters.hardening_modulus * state.accumulated_slip
            effective = jnp.maximum(jnp.abs(resolved) - hardening_force, 0.0)
            rate = (
                self.parameters.reference_rate
                * jnp.sign(resolved)
                * (effective / self.parameters.initial_strength)
                ** (1.0 / self.parameters.rate_sensitivity)
            )
            return increment - dt * rate

        def response(increment, args):
            del args
            state = state_from_increment(increment)
            stress, resolved = resolved_shear(state)
            (
                first_piola,
                _,
                elastic,
                elastic_determinant,
                plastic_determinant,
                logarithm,
                invertible,
            ) = stress
            elastic_energy = self._elastic_energy(elastic, logarithm)
            hardening_energy = self._hardening_energy(state.accumulated_slip)
            free_energy = elastic_energy + hardening_energy
            hardening_increment = hardening_energy - committed_hardening
            plastic_work = jnp.sum(resolved * increment)
            incremental_dissipation = plastic_work - hardening_increment
            dtype = deformation.real.dtype
            scale = jnp.maximum(
                1.0,
                jnp.maximum(jnp.abs(plastic_work), jnp.abs(hardening_increment)),
            )
            thermodynamic_tolerance = 256.0 * jnp.finfo(dtype).eps * scale
            thermodynamic_admissible = jnp.isfinite(incremental_dissipation) & (
                incremental_dissipation >= -thermodynamic_tolerance
            )
            isochoric_tolerance = 1024.0 * jnp.finfo(dtype).eps
            expected_strength = (
                self.parameters.initial_strength
                + self.parameters.hardening_modulus * state.accumulated_slip
            )
            strength_tolerance = (
                512.0
                * jnp.finfo(dtype).eps
                * jnp.maximum(1.0, jnp.abs(expected_strength))
            )
            committed_expected_strength = (
                self.parameters.initial_strength
                + self.parameters.hardening_modulus * committed_state.accumulated_slip
            )
            committed_valid = (
                jnp.all(jnp.isfinite(committed_state.plastic_deformation))
                & jnp.all(jnp.isfinite(committed_state.strengths))
                & jnp.isfinite(committed_state.accumulated_slip)
                & (committed_state.accumulated_slip >= 0.0)
                & jnp.all(committed_state.strengths > 0.0)
                & jnp.all(
                    jnp.abs(committed_state.strengths - committed_expected_strength)
                    <= strength_tolerance
                )
            )
            state_finite = (
                jnp.all(jnp.isfinite(state.plastic_deformation))
                & jnp.all(jnp.isfinite(state.strengths))
                & committed_valid
                & jnp.isfinite(state.accumulated_slip)
                & (state.accumulated_slip >= committed_state.accumulated_slip)
                & (state.accumulated_slip >= 0.0)
                & jnp.all(state.strengths > 0.0)
                & jnp.all(
                    jnp.abs(state.strengths - expected_strength) <= strength_tolerance
                )
            )
            admissible = (
                invertible
                & state_finite
                & jnp.isfinite(elastic_determinant)
                & jnp.isfinite(plastic_determinant)
                & jnp.isfinite(free_energy)
                & jnp.isfinite(plastic_work)
                & (elastic_determinant > 0.0)
                & (plastic_determinant > 0.0)
                & (jnp.abs(plastic_determinant - 1.0) <= isochoric_tolerance)
                & (jnp.max(jnp.abs(increment)) <= self.parameters.maximum_slip_increment)
                & thermodynamic_admissible
            )
            return ConstitutiveResponse(
                first_piola,
                state.pack(),
                energy=free_energy,
                dissipation=jnp.maximum(incremental_dissipation, 0.0),
                valid=admissible,
                diagnostics={
                    "admissible": admissible,
                    "thermodynamic_admissible": thermodynamic_admissible,
                    "elastic_deformation": elastic,
                    "elastic_determinant": elastic_determinant,
                    "plastic_determinant": plastic_determinant,
                    "elastic_energy": elastic_energy,
                    "hardening_energy": hardening_energy,
                    "free_energy": free_energy,
                    "plastic_work": plastic_work,
                    "incremental_dissipation": incremental_dissipation,
                    "maximum_slip_increment": jnp.max(jnp.abs(increment)),
                },
            )

        tolerance = max(1.0e-10, 64.0 * float(jnp.finfo(deformation.real.dtype).eps))
        material = LocalImplicitMaterial(
            residual,
            response,
            state_shape=(self.slip_count,),
            model_id=f"crystal-slip-root:{self.model_id}",
            max_steps=25,
            tolerance=tolerance,
        )
        initial = jnp.zeros((self.slip_count,), dtype=deformation.dtype)
        slip_increment, root_diagnostics = material.solve_with_diagnostics(initial, None)
        constitutive = response(slip_increment, None)
        state = CrystalPlasticityState.unpack(constitutive.trial_state, self.slip_count)
        maximum_increment = constitutive.diagnostics["maximum_slip_increment"]
        admissible = constitutive.diagnostics["admissible"]
        converged = root_diagnostics.converged
        bound_factor = jnp.minimum(
            1.0,
            self.parameters.maximum_slip_increment
            / jnp.maximum(maximum_increment, jnp.finfo(deformation.real.dtype).tiny),
        )
        accepted = converged & admissible
        factor = jnp.where(accepted, bound_factor, jnp.minimum(0.5, bound_factor))
        return CrystalPlasticityUpdate(
            first_piola=constitutive.response,
            state=state,
            slip_increment=slip_increment,
            converged=converged,
            admissible=admissible,
            suggested_step_factor=factor,
            elastic_deformation=constitutive.diagnostics["elastic_deformation"],
            elastic_determinant=constitutive.diagnostics["elastic_determinant"],
            plastic_determinant=constitutive.diagnostics["plastic_determinant"],
            elastic_energy=constitutive.diagnostics["elastic_energy"],
            hardening_energy=constitutive.diagnostics["hardening_energy"],
            free_energy=constitutive.diagnostics["free_energy"],
            plastic_work=constitutive.diagnostics["plastic_work"],
            incremental_dissipation=constitutive.diagnostics["incremental_dissipation"],
            thermodynamic_admissible=constitutive.diagnostics["thermodynamic_admissible"],
            residual_norm=root_diagnostics.residual_norm,
        )


def _route_rule(cell_kind: str, polynomial_degree: int, /) -> tuple[ReferenceRule, int]:
    degree = int(polynomial_degree)
    order = max(2, degree + 1)
    if cell_kind == "tetrahedron":
        order = max(order, degree + 2)
    axis_rule = GaussLegendreRule(order)
    if cell_kind == "tetrahedron":
        return ReferenceTetrahedronRule(axis_rule), order
    if cell_kind == "hexahedron":
        return ReferenceHexahedronRule(axis_rule), order
    if cell_kind == "prism":
        return ReferencePrismRule(axis_rule), order
    if cell_kind == "pyramid":
        return ReferencePyramidRule(axis_rule), order
    raise ValueError("CPFEM routes require supported three-dimensional cell blocks.")


class CrystalPlasticityRoute(StrictModule, NonTrainableState):
    """Static block-exact routing and ragged material-state ownership for CPFEM."""

    models: tuple[CrystalPlasticityModel, ...]
    crystal_to_sample: tuple[Array, ...]
    domains: tuple[IntegrationDomain, ...]
    rules: tuple[ReferenceRule, ...]
    site_ids: tuple[MaterialSiteId, ...]
    block_names: tuple[str, ...] = eqx.field(static=True)
    orientation_ids: tuple[str, ...] = eqx.field(static=True)
    entry_ids: tuple[str, ...] = eqx.field(static=True)
    material_ids: tuple[str, ...] = eqx.field(static=True)
    state_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    block_indices: tuple[int, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        routes: Sequence[tuple[str, CrystalPlasticityModel, ArrayLike]],
        /,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        field = str(field_name)
        if not field:
            raise ValueError("CPFEM route field_name must be non-empty.")
        field_index = discretization._field_index(field)
        if (
            discretization.mesh.topological_dimension != 3
            or discretization.mesh.ambient_dimension != 3
            or discretization.dof_maps[field_index].component_shape != (3,)
        ):
            raise ValueError("CPFEM routes require a three-dimensional vector field.")
        entries = tuple(routes)
        if not entries or any(
            not isinstance(entry, Sequence) or len(entry) != 3 for entry in entries
        ):
            raise ValueError(
                "CPFEM routes require (block_name, model, crystal_to_sample) entries."
            )
        names = tuple(str(entry[0]) for entry in entries)
        if len(set(names)) != len(names):
            raise ValueError(
                "CPFEM block routes overlap; each block has one phase route."
            )
        expected_names = tuple(block.name for block in discretization.mesh.blocks)
        if set(names) != set(expected_names):
            missing = sorted(set(expected_names) - set(names))
            unknown = sorted(set(names) - set(expected_names))
            raise ValueError(
                "CPFEM routes must exactly cover every cell block without gaps; "
                f"missing={missing!r}, unknown={unknown!r}."
            )
        by_name = {str(entry[0]): (entry[1], entry[2]) for entry in entries}
        models = []
        orientations = []
        orientation_ids = []
        domains = []
        rules = []
        site_ids = []
        material_ids = []
        state_shapes = []
        entry_ids = []
        cell_offset = 0
        cell_entity_set = discretization.cell_domain.entity_set_id
        for block_index, block in enumerate(discretization.mesh.blocks):
            model, orientation_value = by_name[block.name]
            if not isinstance(model, CrystalPlasticityModel):
                raise TypeError(
                    "Every CPFEM block route requires CrystalPlasticityModel."
                )
            polynomial_degree = max(
                discretization.elements[field_index][block_index].degree,
                discretization.coordinate_elements[block_index].degree,
            )
            rule, axis_order = _route_rule(block.cell_kind, polynomial_degree)
            rule_data = reference_rule_data(rule)
            shape = (
                block.cell_count,
                int(rule_data.weights.shape[0]),
                model.state_width,
            )
            orientation = _crystal_rotation_field(orientation_value, shape[:2])
            orientation_id = canonical_fingerprint(
                {
                    "kind": "crystal-to-sample-orientation-field",
                    "rotation": array_tree_fingerprint(np.asarray(orientation)),
                }
            )
            entry_id = canonical_fingerprint(
                {
                    "kind": "crystal-plasticity-block-route",
                    "support": discretization.support.support_id,
                    "prepared": discretization.prepared_id,
                    "field": field,
                    "block": block.block_id,
                    "model": model.model_id,
                    "orientation": orientation_id,
                    "state_shape": list(shape),
                    "quadrature_axis_order": axis_order,
                }
            )
            site = MaterialSiteId(f"cpfem:{field}:{block.name}")
            material_id = canonical_fingerprint(
                {
                    "kind": "crystal-plasticity-routed-material",
                    "site": site.site_id,
                    "route": entry_id,
                }
            )
            cells = np.arange(cell_offset, cell_offset + block.cell_count, dtype=np.int32)
            cell_offset += block.cell_count
            domain = IntegrationDomain(
                "cell",
                cells,
                discretization.support.support_id,
                cell_entity_set,
                owner_cells=cells,
                selection_id=entry_id,
            )
            models.append(model)
            orientations.append(orientation)
            orientation_ids.append(orientation_id)
            rules.append(rule)
            domains.append(domain)
            site_ids.append(site)
            material_ids.append(material_id)
            state_shapes.append(shape)
            entry_ids.append(entry_id)
        route_id = canonical_fingerprint(
            {
                "kind": "crystal-plasticity-route",
                "support": discretization.support.support_id,
                "prepared": discretization.prepared_id,
                "field": field,
                "entries": entry_ids,
            }
        )
        self.models = tuple(models)
        self.crystal_to_sample = tuple(orientations)
        self.domains = tuple(domains)
        self.rules = tuple(rules)
        self.site_ids = tuple(site_ids)
        self.orientation_ids = tuple(orientation_ids)
        self.entry_ids = tuple(entry_ids)
        self.block_names = expected_names
        self.material_ids = tuple(material_ids)
        self.state_shapes = tuple(state_shapes)
        self.block_indices = tuple(range(len(expected_names)))
        self.field_name = field
        self.support_id = discretization.support.support_id
        self.prepared_id = discretization.prepared_id
        self.route_id = route_id

    def validate_discretization(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        /,
    ) -> None:
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        if (
            str(field_name) != self.field_name
            or discretization.support.support_id != self.support_id
            or discretization.prepared_id != self.prepared_id
            or tuple(block.name for block in discretization.mesh.blocks)
            != self.block_names
        ):
            raise ValueError("CPFEM route belongs to another field or prepared support.")

    def initialize(self, /, *, dtype: Any | None = None) -> MaterialTransaction:
        states = []
        for model, site, material_id, shape in zip(
            self.models,
            self.site_ids,
            self.material_ids,
            self.state_shapes,
            strict=True,
        ):
            resolved_dtype = (
                model.parameters.initial_strength.dtype
                if dtype is None
                else jnp.dtype(dtype)
            )
            packed = model.initial_state(dtype=resolved_dtype).pack()
            values = jnp.broadcast_to(packed, shape)
            states.append(MaterialState(site, material_id, values))
        transaction = MaterialTransaction(tuple(states))
        self.validate(transaction)
        return transaction

    def validate(self, transaction: MaterialTransaction, /) -> None:
        if not isinstance(transaction, MaterialTransaction):
            raise TypeError("CPFEM material state must be one MaterialTransaction.")
        expected_keys = {site.key for site in self.site_ids}
        if {state.site_id.key for state in transaction.states} != expected_keys:
            raise ValueError(
                "CPFEM material transaction does not match exact site routes."
            )
        if len({state.state_version for state in transaction.states}) != 1:
            raise ValueError(
                "CPFEM material routes must share one atomic state revision."
            )
        for site, material_id, shape in zip(
            self.site_ids, self.material_ids, self.state_shapes, strict=True
        ):
            state = transaction.state(site)
            if (
                state.site_id.site_id != site.site_id
                or state.model_id != material_id
                or state.committed.shape != shape
                or state.trial.shape != shape
            ):
                raise ValueError(
                    "CPFEM material transaction layout does not match route."
                )

    def with_trials(
        self,
        transaction: MaterialTransaction,
        trials: Sequence[ArrayLike],
        /,
    ) -> MaterialTransaction:
        self.validate(transaction)
        values = tuple(trials)
        if len(values) != len(self.site_ids):
            raise ValueError("CPFEM trial tuple must cover every material route.")
        candidate = transaction.with_trials(
            {site.key: value for site, value in zip(self.site_ids, values, strict=True)}
        )
        self.validate(candidate)
        return candidate

    def commit(self, transaction: MaterialTransaction, /) -> MaterialTransaction:
        self.validate(transaction)
        committed = transaction.commit()
        self.validate(committed)
        return committed

    def rollback(self, transaction: MaterialTransaction, /) -> MaterialTransaction:
        self.validate(transaction)
        rolled_back = transaction.rollback()
        self.validate(rolled_back)
        return rolled_back

    def checkpoint(
        self, transaction: MaterialTransaction, /
    ) -> MaterialCheckpointPayload:
        self.validate(transaction)
        return transaction.checkpoint_payload(plan_id=self.route_id)

    def restore(self, checkpoint: MaterialCheckpointPayload, /) -> MaterialTransaction:
        if not isinstance(checkpoint, MaterialCheckpointPayload):
            raise TypeError("checkpoint must be MaterialCheckpointPayload.")
        if checkpoint.plan_id != self.route_id:
            raise ValueError("CPFEM checkpoint belongs to another material route.")
        transaction = checkpoint.restore()
        self.validate(transaction)
        return transaction


def _point_updates(
    model: CrystalPlasticityModel,
    crystal_to_sample: Array,
    step_size: Array,
    deformation: Array,
    committed: Array,
    /,
):
    if deformation.shape[:2] != committed.shape[
        :2
    ] or crystal_to_sample.shape != deformation.shape[:2] + (3, 3):
        raise ValueError(
            "CPFEM material state and orientation must match routed quadrature."
        )
    flat_deformation = deformation.reshape((-1, 3, 3))
    flat_state = committed.reshape((-1, committed.shape[-1]))
    flat_orientation = crystal_to_sample.reshape((-1, 3, 3))

    def point_update(deformation_, packed_state, orientation_):
        state = CrystalPlasticityState.unpack(packed_state, model.slip_count)
        update = model.update(deformation_, state, orientation_, step_size)
        return (
            update.first_piola,
            update.state.pack(),
            update.converged,
            update.admissible,
            update.suggested_step_factor,
            update.incremental_dissipation,
        )

    return jax.vmap(point_update)(flat_deformation, flat_state, flat_orientation)


def cpfem_equilibrium_form(
    discretization: FiniteElementDiscretization,
    field_name: str,
    route: CrystalPlasticityRoute,
    transaction: MaterialTransaction,
    step_size: ArrayLike,
    /,
    *,
    form_id: str = "cpfem-equilibrium",
) -> FiniteElementForm:
    """Build routed equilibrium with one atomic material transaction."""

    if not isinstance(route, CrystalPlasticityRoute):
        raise TypeError("route must be CrystalPlasticityRoute.")
    route.validate_discretization(discretization, field_name)
    route.validate(transaction)
    dt = _finite_scalar(step_size, "CPFEM step_size")
    if bool(dt <= 0.0):
        raise ValueError("CPFEM step_size must be positive.")
    field_index = discretization._field_index(field_name)
    actions = []

    for model, orientation, domain, site, block_name, rule, material_id in zip(
        route.models,
        route.crystal_to_sample,
        route.domains,
        route.site_ids,
        route.block_names,
        route.rules,
        route.material_ids,
        strict=True,
    ):
        committed = transaction.state(site).committed

        def residual(
            values,
            gradients,
            points,
            weights,
            test_basis,
            test_gradients,
            context,
            *,
            model_=model,
            orientation_=orientation,
            committed_=committed,
        ):
            del values, points, test_basis, context
            deformation = jnp.eye(3, dtype=gradients[0].dtype) + gradients[0]
            outputs = _point_updates(model_, orientation_, dt, deformation, committed_)
            stress = outputs[0].reshape(deformation.shape)
            return ein.contract(
                "cq,cqib,cqab->cia",
                weights,
                test_gradients,
                stress,
            )

        actions.append(
            CellResidualAction(
                field_name,
                (field_name,),
                residual,
                domain=domain,
                rules=((block_name, rule),),
                action_id=f"cpfem-internal-force:{material_id}",
            )
        )

    def auxiliary(state, context):
        displacement = jnp.asarray(state)
        runtime = discretization.default_runtime if context is None else context.runtime
        trial_values = []
        convergence = []
        admissibility = []
        step_factors = []
        dissipations = []
        for model, orientation, rule, site, block_index in zip(
            route.models,
            route.crystal_to_sample,
            route.rules,
            route.site_ids,
            route.block_indices,
            strict=True,
        ):
            rule_data = reference_rule_data(rule)
            geometry = discretization.evaluate_block_geometry(
                field_name,
                block_index,
                runtime.coordinates,
                rule_data.points,
                rule_data.weights,
            )
            dofs = discretization.dof_maps[field_index].cell_dofs[block_index]
            dof_orientation = discretization.dof_maps[field_index].orientations[
                block_index
            ]
            local = displacement[dofs] * dof_orientation[..., None]
            gradient = ein.contract(
                "cqid,cia->cqad",
                geometry.physical_gradients,
                local,
            )
            deformation = jnp.eye(3, dtype=gradient.dtype) + gradient
            committed = transaction.state(site).committed
            outputs = _point_updates(model, orientation, dt, deformation, committed)
            trial_values.append(outputs[1].reshape(committed.shape))
            convergence.append(outputs[2])
            admissibility.append(outputs[3])
            step_factors.append(outputs[4])
            dissipations.append(outputs[5])
        trial_transaction = route.with_trials(transaction, tuple(trial_values))
        converged = jnp.all(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in convergence))
        )
        admissible = jnp.all(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in admissibility))
        )
        minimum_factor = jnp.min(
            jnp.concatenate(tuple(value.reshape((-1,)) for value in step_factors))
        )
        return FiniteElementAuxiliaryEvaluation(
            trial_transaction,
            successful=converged,
            admissible=admissible,
            retry_requested=~(converged & admissible),
            suggested_step=dt * minimum_factor,
            diagnostics={
                "route_id": route.route_id,
                "converged_points": tuple(jnp.sum(value) for value in convergence),
                "admissible_points": tuple(jnp.sum(value) for value in admissibility),
                "incremental_dissipation": tuple(
                    value.reshape((-1,)) for value in dissipations
                ),
            },
        )

    return FiniteElementForm(
        form_id,
        field_name,
        tuple(actions),
        auxiliary_evaluator=auxiliary,
        auxiliary_id=f"cpfem-material-worksets:{route.route_id}",
    )


__all__ = [
    "CrystalPlasticityModel",
    "CrystalPlasticityParameters",
    "CrystalPlasticityRoute",
    "CrystalPlasticityState",
    "CrystalPlasticityUpdate",
    "CrystalSlipSystem",
    "cpfem_equilibrium_form",
]

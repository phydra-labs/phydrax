#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ConjugateGradient,
    DenseCholesky,
    DenseLinearOperator,
    DifferentiationPolicy,
    FailurePolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)
from ._rod_loads import RodLoad, RodLoadLedger
from ._rod_materials import (
    PreparedKelvinVoigtRodMaterial,
    PreparedLinearElasticRodMaterial,
    RodConstitutiveControl,
    RodConstitutiveResult,
    RodConstitutiveTrial,
)
from ._rod_reduced_kinematics import (
    lift_effort_pullback_operator,
    lift_reduced_rod_velocity,
    lift_velocity_operator,
    target_native_strains,
)
from ._rod_reduction import PreparedReducedRod, ReducedRodState


ReducedRodMaterial: TypeAlias = (
    PreparedLinearElasticRodMaterial | PreparedKelvinVoigtRodMaterial
)
ReducedRodMassSolver: TypeAlias = Literal["dense_cholesky", "matrix_free_cg"]


def _positive_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


class ReducedRodDenseCholeskyPlan(StrictModule, NonTrainableState):
    """Fixed dense reduced-mass Cholesky policy and fail-closed tolerances."""

    symmetry_tolerance: float = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    roundtrip_tolerance: float = eqx.field(static=True)
    solver: ReducedRodMassSolver = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        symmetry_tolerance: float = 1.0e-6,
        pivot_tolerance: float = 1.0e-9,
        condition_limit: float = 1.0e8,
        roundtrip_tolerance: float = 1.0e-6,
    ):
        symmetry = _positive_finite(symmetry_tolerance, "symmetry_tolerance")
        pivot = _positive_finite(pivot_tolerance, "pivot_tolerance")
        condition = _positive_finite(condition_limit, "condition_limit")
        roundtrip = _positive_finite(roundtrip_tolerance, "roundtrip_tolerance")
        if condition <= 1.0:
            raise ValueError("condition_limit must be greater than one.")
        self.symmetry_tolerance = symmetry
        self.pivot_tolerance = pivot
        self.condition_limit = condition
        self.roundtrip_tolerance = roundtrip
        self.solver = "dense_cholesky"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-dense-cholesky-plan",
                "symmetry_tolerance": symmetry,
                "pivot_tolerance": pivot,
                "condition_limit": condition,
                "roundtrip_tolerance": roundtrip,
            }
        )


class ReducedRodMatrixFreeCGPlan(StrictModule, NonTrainableState):
    """Fixed matrix-free CG and Lanczos evidence work budget."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    spectral_iterations: int | None = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    roundtrip_tolerance: float = eqx.field(static=True)
    solver: ReducedRodMassSolver = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-8,
        maximum_iterations: int = 128,
        spectral_iterations: int | None = None,
        symmetry_tolerance: float = 1.0e-6,
        positivity_tolerance: float = 1.0e-9,
        condition_limit: float = 1.0e8,
        roundtrip_tolerance: float = 1.0e-5,
    ):
        relative = _positive_finite(relative_tolerance, "relative_tolerance")
        absolute = _positive_finite(absolute_tolerance, "absolute_tolerance")
        symmetry = _positive_finite(symmetry_tolerance, "symmetry_tolerance")
        positivity = _positive_finite(positivity_tolerance, "positivity_tolerance")
        condition = _positive_finite(condition_limit, "condition_limit")
        roundtrip = _positive_finite(roundtrip_tolerance, "roundtrip_tolerance")
        iterations = int(maximum_iterations)
        spectral = None if spectral_iterations is None else int(spectral_iterations)
        if iterations < 1 or (spectral is not None and spectral < 1):
            raise ValueError("CG and spectral iteration counts must be positive.")
        if condition <= 1.0:
            raise ValueError("condition_limit must be greater than one.")
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.maximum_iterations = iterations
        self.spectral_iterations = spectral
        self.symmetry_tolerance = symmetry
        self.positivity_tolerance = positivity
        self.condition_limit = condition
        self.roundtrip_tolerance = roundtrip
        self.solver = "matrix_free_cg"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-matrix-free-cg-plan",
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "maximum_iterations": iterations,
                "spectral_iterations": spectral,
                "symmetry_tolerance": symmetry,
                "positivity_tolerance": positivity,
                "condition_limit": condition,
                "roundtrip_tolerance": roundtrip,
            }
        )


ReducedRodDynamicsPlan: TypeAlias = (
    ReducedRodDenseCholeskyPlan | ReducedRodMatrixFreeCGPlan
)


class ReducedRodMaterialState(StrictModule):
    """Committed stretch/shear and bend/twist material histories."""

    stretch_shear_history: Array
    bend_twist_history: Array


class ReducedRodMaterialControl(StrictModule):
    """Sitewise stretch/shear and bend/twist material controls."""

    stretch_shear_control: RodConstitutiveControl
    bend_twist_control: RodConstitutiveControl


class ReducedRodDirectLoad(StrictModule):
    """One named effort already expressed in the reduced coordinate dual."""

    effort: Array
    source_id: str = eqx.field(static=True)
    power_channel: str = eqx.field(static=True)

    def __init__(self, effort: ArrayLike, /, *, source_id: str, power_channel: str):
        value = jnp.asarray(effort)
        if (
            value.ndim != 1
            or not jnp.issubdtype(value.dtype, jnp.inexact)
            or jnp.iscomplexobj(value)
        ):
            raise TypeError("A direct reduced effort must be a real rank-one array.")
        source = str(source_id).strip()
        channel = str(power_channel).strip()
        if not source or not channel:
            raise ValueError("Reduced load source_id and power_channel must be nonempty.")
        self.effort = value
        self.source_id = source
        self.power_channel = channel


class ReducedRodMassEvidence(StrictModule):
    symmetry_error: Array
    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    minimum_cholesky_pivot: Array
    condition_estimate: Array
    finite: Array
    symmetric: Array
    positive_definite: Array
    pivot_checked: Array
    pivot_valid: Array
    conditioned: Array
    valid: Array
    spectral_iterations: int = eqx.field(static=True)
    solver: ReducedRodMassSolver = eqx.field(static=True)


class ReducedRodMassResult(StrictModule):
    operator: AbstractLinearOperator
    evidence: ReducedRodMassEvidence
    dynamics_id: str = eqx.field(static=True)


class ReducedRodSolveEvidence(StrictModule):
    status: Array
    residual_norm: Array
    relative_residual: Array
    iterations: Array
    condition_estimate: Array
    minimum_cholesky_pivot: Array
    roundtrip_error: Array
    relative_roundtrip_error: Array
    finite: Array
    symmetric: Array
    positive_definite: Array
    conditioned: Array
    converged: Array
    roundtrip_valid: Array
    valid: Array
    solver: ReducedRodMassSolver = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class ReducedRodInverseMassResult(StrictModule):
    acceleration: Array
    inverse_mass_operator: AbstractLinearOperator
    mass: ReducedRodMassResult
    solve_evidence: ReducedRodSolveEvidence
    dynamics_id: str = eqx.field(static=True)


class ReducedRodBiasResult(StrictModule):
    effort: Array
    lift_acceleration: tuple[Array, Array]
    native_gyroscopic_effort: tuple[Array, Array]
    finite: Array
    dynamics_id: str = eqx.field(static=True)


class ReducedRodEnergyResult(StrictModule):
    kinetic_energy: Array
    stored_energy: Array
    viscous_dissipation: Array
    total_mechanical_energy: Array
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class ReducedRodForceResult(StrictModule):
    elastic_effort: Array
    kelvin_voigt_effort: Array
    gravity_effort: Array
    native_external_effort: Array
    direct_reduced_effort: Array
    total_effort: Array
    source_efforts: Array
    channel_efforts: Array
    source_power: Array
    channel_power: Array
    total_power: Array
    paired_power: Array
    power_residual: Array
    finite: Array
    power_valid: Array
    valid: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    source_channels: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def effort_for_source(self, source_id: str, /) -> Array:
        source = str(source_id).strip()
        try:
            index = self.source_ids.index(source)
        except ValueError as error:
            raise KeyError(f"Unknown reduced rod load source {source!r}.") from error
        return self.source_efforts[index]

    def power_for_source(self, source_id: str, /) -> Array:
        source = str(source_id).strip()
        try:
            index = self.source_ids.index(source)
        except ValueError as error:
            raise KeyError(f"Unknown reduced rod load source {source!r}.") from error
        return self.source_power[index]

    def effort_for_channel(self, channel: str, /) -> Array:
        name = str(channel).strip()
        try:
            index = self.channel_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown reduced rod power channel {name!r}.") from error
        return self.channel_efforts[index]

    def power_for_channel(self, channel: str, /) -> Array:
        name = str(channel).strip()
        try:
            index = self.channel_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown reduced rod power channel {name!r}.") from error
        return self.channel_power[index]


class ReducedRodDynamicsEvaluation(StrictModule):
    mass: ReducedRodMassResult
    bias: ReducedRodBiasResult
    forces: ReducedRodForceResult
    energy: ReducedRodEnergyResult
    candidate_material_state: ReducedRodMaterialState
    stretch_shear_material_result: RodConstitutiveResult
    bend_twist_material_result: RodConstitutiveResult
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class ReducedRodForwardDynamicsResult(StrictModule):
    acceleration: Array
    rhs_effort: Array
    evaluation: ReducedRodDynamicsEvaluation
    solve_evidence: ReducedRodSolveEvidence
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class ReducedRodInverseDynamicsResult(StrictModule):
    required_effort: Array
    dynamic_effort: Array
    residual: Array
    evaluation: ReducedRodDynamicsEvaluation
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class ReducedRodDenseReferenceResult(StrictModule):
    """Explicit AD authority; never used by the fused production kernels."""

    mass_matrix: Array
    bias_effort: Array
    elastic_effort: Array
    kelvin_voigt_effort: Array
    kinetic_energy: Array
    stored_energy: Array
    finite: Array
    dynamics_id: str = eqx.field(static=True)


class PreparedReducedRodDynamics(StrictModule, NonTrainableState):
    """Prepared SR2 mechanics with fused lift/inertia/pullback production actions."""

    reduction: PreparedReducedRod
    plan: ReducedRodDynamicsPlan
    stretch_shear_material: RodConstitutiveTrial
    bend_twist_material: RodConstitutiveTrial
    gravity_load: RodLoad | None
    solve_policy: LinearSolvePolicy
    spectral_iterations: int = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: PreparedReducedRod,
        plan: ReducedRodDynamicsPlan | None = None,
        /,
        *,
        stretch_shear_material: ReducedRodMaterial | None = None,
        bend_twist_material: ReducedRodMaterial | None = None,
        gravity: ArrayLike | None = None,
    ):
        if not isinstance(reduction, PreparedReducedRod):
            raise TypeError("reduction must be a PreparedReducedRod.")
        plan_ = ReducedRodDenseCholeskyPlan() if plan is None else plan
        if not isinstance(
            plan_, (ReducedRodDenseCholeskyPlan, ReducedRodMatrixFreeCGPlan)
        ):
            raise TypeError("plan must be a reduced rod mass solve plan.")
        stretch = (
            reduction.rod.stretch_shear_material
            if stretch_shear_material is None
            else stretch_shear_material
        )
        bend = (
            reduction.rod.bend_twist_material
            if bend_twist_material is None
            else bend_twist_material
        )
        self._validate_material(stretch, reduction.rod.stretch_shear_workset.workset_id)
        self._validate_material(bend, reduction.rod.bend_twist_workset.workset_id)
        gravity_load = None
        if gravity is not None:
            acceleration = np.asarray(gravity)
            dimension = reduction.rod.plan.dimension
            if (
                acceleration.shape != (dimension,)
                or not np.issubdtype(acceleration.dtype, np.inexact)
                or np.iscomplexobj(acceleration)
                or not np.all(np.isfinite(acceleration))
            ):
                raise ValueError(
                    "gravity must be one finite real ambient acceleration vector."
                )
            dtype = np.dtype(reduction.rod.plan.rest_positions.dtype)
            acceleration = acceleration.astype(dtype, copy=False)
            forces = (
                np.asarray(reduction.rod.node_masses)[:, None] * acceleration[None, :]
            )
            angular_shape = (
                (reduction.rod.plan.segment_count,)
                if dimension == 2
                else (reduction.rod.plan.segment_count, 3)
            )
            gravity_load = RodLoad(
                forces,
                np.zeros(angular_shape, dtype=dtype),
                source_id="gravity",
                power_channel="gravity",
            )
        if isinstance(plan_, ReducedRodDenseCholeskyPlan):
            policy = LinearSolvePolicy(
                DenseCholesky(),
                differentiation=DifferentiationPolicy("mathematical"),
                failure=FailurePolicy("status"),
            )
            spectral_iterations = reduction.plan.coordinate_count
        else:
            policy = LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=plan_.relative_tolerance,
                    absolute=plan_.absolute_tolerance,
                    max_steps=plan_.maximum_iterations,
                ),
                differentiation=DifferentiationPolicy("mathematical"),
                failure=FailurePolicy("status"),
            )
            spectral_iterations = min(
                reduction.plan.coordinate_count,
                reduction.plan.coordinate_count
                if plan_.spectral_iterations is None
                else plan_.spectral_iterations,
            )
        self.reduction = reduction
        self.plan = plan_
        self.stretch_shear_material = stretch
        self.bend_twist_material = bend
        self.gravity_load = gravity_load
        self.solve_policy = policy
        self.spectral_iterations = spectral_iterations
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-rod-dynamics",
                "reduction": reduction.prepared_id,
                "plan": plan_.plan_id,
                "stretch_material": stretch.material_id,
                "bend_material": bend.material_id,
                "gravity": None
                if gravity_load is None
                else array_tree_fingerprint(np.asarray(gravity_load.forces)),
            }
        )

    @staticmethod
    def _validate_material(material: object, workset_id: str, /) -> None:
        if not isinstance(
            material,
            (PreparedLinearElasticRodMaterial, PreparedKelvinVoigtRodMaterial),
        ):
            raise TypeError(
                "Reduced dynamics supports prepared linear or Kelvin-Voigt rod trials."
            )
        if material.workset.workset_id != workset_id:
            raise ValueError("Prepared reduced material must use the native rod workset.")

    def initialize_material_state(self, /) -> ReducedRodMaterialState:
        return ReducedRodMaterialState(
            self.stretch_shear_material.initialize_history(),
            self.bend_twist_material.initialize_history(),
        )

    def initialize_material_control(self, /) -> ReducedRodMaterialControl:
        return ReducedRodMaterialControl(
            self.stretch_shear_material.initialize_control(),
            self.bend_twist_material.initialize_control(),
        )

    def _native_inertia_action(
        self, velocity: tuple[Array, Array], /
    ) -> tuple[Array, Array]:
        linear, angular = self.reduction.native_velocity_space.validate(velocity)
        forces = self.reduction.rod.node_masses[:, None] * linear
        if self.reduction.rod.plan.dimension == 2:
            moments = self.reduction.rod.segment_inertias * angular
        else:
            moments = ein.contract(
                "sij,sj->si", self.reduction.rod.segment_inertias, angular
            )
        return self.reduction.native_effort_space.validate((forces, moments))

    def _matrix_free_mass_operator(
        self, coefficients: ArrayLike, /
    ) -> AbstractLinearOperator:
        point = self.reduction.coefficient_space.validate(jnp.asarray(coefficients))
        velocity_operator = lift_velocity_operator(self.reduction, point)
        effort_pullback = lift_effort_pullback_operator(self.reduction, point)

        def action(tangent):
            velocity = velocity_operator.mv(tangent)
            native_effort = self._native_inertia_action(velocity)
            return effort_pullback.mv(native_effort)

        return FunctionLinearOperator(
            action,
            source=self.reduction.coefficient_space,
            target=self.reduction.reduced_effort_space,
            operator_id=canonical_fingerprint(
                {"kind": "reduced-rod-fused-mass", "dynamics": self.dynamics_id}
            ),
        )

    def _dense_mass_matrix(self, operator: AbstractLinearOperator, /) -> Array:
        size = self.reduction.plan.coordinate_count
        dtype = self.reduction.rod.plan.rest_positions.dtype
        columns = jax.vmap(operator.mv)(jnp.eye(size, dtype=dtype))
        return jnp.swapaxes(columns, -1, -2)

    def _mass_endomorphism_action(
        self,
        operator: AbstractLinearOperator,
        tangent: Array,
        /,
    ) -> Array:
        """Apply ``R^-1 M`` without relabelling an effort as a tangent."""
        effort = self.reduction.reduced_effort_space.validate(operator.mv(tangent))
        return self.reduction.coefficient_space.inverse_riesz(effort)

    def _mass_endomorphism_operator(
        self, operator: AbstractLinearOperator, /
    ) -> AbstractLinearOperator:
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        if isinstance(self.plan, ReducedRodDenseCholeskyPlan):
            if not isinstance(operator, DenseLinearOperator):
                raise TypeError("Dense reduced-mass execution requires a dense operator.")
            columns = jnp.swapaxes(operator.matrix, -1, -2)
            primal_columns = jax.vmap(self.reduction.coefficient_space.inverse_riesz)(
                columns
            )
            matrix = jnp.swapaxes(primal_columns, -1, -2)
            return DenseLinearOperator(
                matrix,
                source=self.reduction.coefficient_space,
                target=self.reduction.coefficient_space,
                properties=properties,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "reduced-rod-riesz-mass-endomorphism",
                        "dynamics": self.dynamics_id,
                    }
                ),
            )

        def action(tangent):
            return self._mass_endomorphism_action(operator, tangent)

        return FunctionLinearOperator(
            action,
            source=self.reduction.coefficient_space,
            target=self.reduction.coefficient_space,
            transpose_action=action,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-riesz-mass-endomorphism",
                    "dynamics": self.dynamics_id,
                }
            ),
        )

    def _lanczos_spectrum(self, operator: AbstractLinearOperator, /) -> Array:
        size = self.reduction.plan.coordinate_count
        steps = self.spectral_iterations
        dtype = self.reduction.rod.plan.rest_positions.dtype
        seed = jnp.arange(1, size + 1, dtype=dtype)
        q = seed / jnp.sqrt(jnp.real(self.reduction.coefficient_space.inner(seed, seed)))
        previous = jnp.zeros_like(q)
        alpha = jnp.zeros((steps,), dtype=dtype)
        beta = jnp.zeros((steps,), dtype=dtype)

        def body(index, carry):
            prior, current, prior_beta, diagonal, off_diagonal = carry
            image = operator.mv(current) - prior_beta * prior
            coefficient = jnp.real(self.reduction.coefficient_space.inner(current, image))
            residual = image - coefficient * current
            next_beta = jnp.sqrt(
                jnp.real(self.reduction.coefficient_space.inner(residual, residual))
            )
            safe = jnp.where(
                next_beta > jnp.finfo(dtype).eps, next_beta, jnp.ones_like(next_beta)
            )
            next_vector = jnp.where(
                next_beta > jnp.finfo(dtype).eps, residual / safe, current
            )
            diagonal = diagonal.at[index].set(coefficient)
            off_diagonal = off_diagonal.at[index].set(next_beta)
            return current, next_vector, next_beta, diagonal, off_diagonal

        _, _, _, alpha, beta = jax.lax.fori_loop(
            0, steps, body, (previous, q, jnp.asarray(0.0, dtype=dtype), alpha, beta)
        )
        tridiagonal = jnp.diag(alpha)
        if steps > 1:
            tridiagonal = tridiagonal + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
        return jnp.linalg.eigvalsh(tridiagonal)

    def mass(self, coefficients: ArrayLike, /) -> ReducedRodMassResult:
        """Return the production ``J* H J`` tangent-to-dual operator and evidence."""
        fused = self._matrix_free_mass_operator(coefficients)
        dtype = self.reduction.rod.plan.rest_positions.dtype
        if isinstance(self.plan, ReducedRodDenseCholeskyPlan):
            dual_matrix = self._dense_mass_matrix(fused)
            operator = DenseLinearOperator(
                dual_matrix,
                source=self.reduction.coefficient_space,
                target=self.reduction.reduced_effort_space,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "reduced-rod-dense-mass",
                        "dynamics": self.dynamics_id,
                    }
                ),
            )
            endomorphism = self._mass_endomorphism_operator(operator)
            assert isinstance(endomorphism, DenseLinearOperator)
            matrix = endomorphism.matrix
            symmetric_part = 0.5 * (matrix + matrix.T)
            eigenvalues = jnp.linalg.eigvalsh(symmetric_part)
            factor = jnp.linalg.cholesky(symmetric_part)
            pivot = jnp.min(jnp.diag(factor))
            symmetry_error = jnp.max(jnp.abs(matrix - matrix.T))
            pivot_checked = jnp.asarray(True)
            pivot_tolerance = self.plan.pivot_tolerance
            symmetry_tolerance = self.plan.symmetry_tolerance
        else:
            operator = fused
            endomorphism = self._mass_endomorphism_operator(operator)
            eigenvalues = self._lanczos_spectrum(endomorphism)
            size = self.reduction.plan.coordinate_count
            first = jnp.arange(1, size + 1, dtype=dtype)
            second = jnp.flip(first)
            first = first / jnp.sqrt(self.reduction.coefficient_space.inner(first, first))
            second = second / jnp.sqrt(
                self.reduction.coefficient_space.inner(second, second)
            )
            left = self.reduction.coefficient_space.inner(first, endomorphism.mv(second))
            right = self.reduction.coefficient_space.inner(endomorphism.mv(first), second)
            symmetry_error = jnp.abs(left - right)
            pivot = jnp.asarray(jnp.nan, dtype=dtype)
            pivot_checked = jnp.asarray(False)
            pivot_tolerance = self.plan.positivity_tolerance
            symmetry_tolerance = self.plan.symmetry_tolerance
        minimum = jnp.min(eigenvalues)
        maximum = jnp.max(eigenvalues)
        condition = jnp.where(
            minimum > 0.0,
            maximum / minimum,
            jnp.asarray(jnp.inf, dtype=dtype),
        )
        finite = (
            jnp.all(jnp.isfinite(eigenvalues))
            & jnp.isfinite(symmetry_error)
            & ((~pivot_checked) | jnp.isfinite(pivot))
        )
        symmetric = symmetry_error <= symmetry_tolerance * jnp.maximum(
            1.0, jnp.abs(maximum)
        )
        positive = minimum > pivot_tolerance
        pivot_valid = (~pivot_checked) | (pivot > pivot_tolerance)
        conditioned = jnp.isfinite(condition) & (condition <= self.plan.condition_limit)
        evidence = ReducedRodMassEvidence(
            symmetry_error,
            minimum,
            maximum,
            pivot,
            condition,
            finite,
            symmetric,
            positive,
            pivot_checked,
            pivot_valid,
            conditioned,
            finite & symmetric & positive & pivot_valid & conditioned,
            self.spectral_iterations,
            self.plan.solver,
        )
        return ReducedRodMassResult(operator, evidence, self.dynamics_id)

    def _inverse_mass_from_mass(
        self,
        mass: ReducedRodMassResult,
        effort: ArrayLike,
        /,
    ) -> ReducedRodInverseMassResult:
        rhs = self.reduction.reduced_effort_space.validate(jnp.asarray(effort))
        solver_operator = self._mass_endomorphism_operator(mass.operator)
        solver_rhs = self.reduction.coefficient_space.inverse_riesz(rhs)
        problem = LinearSystem(
            solver_operator,
            problem_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-mass-system",
                    "dynamics": self.dynamics_id,
                }
            ),
        )
        result = solve(problem, solver_rhs, policy=self.solve_policy)
        acceleration = self.reduction.coefficient_space.validate(result.value)
        reconstructed = mass.operator.mv(acceleration)
        difference = reconstructed - rhs
        error = jnp.sqrt(
            jnp.real(self.reduction.reduced_effort_space.inner(difference, difference))
        )
        rhs_norm = jnp.sqrt(jnp.real(self.reduction.reduced_effort_space.inner(rhs, rhs)))
        scale = jnp.maximum(jnp.asarray(1.0, dtype=rhs_norm.dtype), rhs_norm)
        relative = error / scale
        diagnostics = result.diagnostics
        finite = (
            jnp.all(jnp.isfinite(acceleration))
            & jnp.isfinite(error)
            & diagnostics.finite
            & mass.evidence.finite
        )
        roundtrip_valid = relative <= self.plan.roundtrip_tolerance
        evidence = ReducedRodSolveEvidence(
            result.status,
            diagnostics.residual_norm,
            diagnostics.relative_residual,
            diagnostics.iterations,
            mass.evidence.condition_estimate,
            mass.evidence.minimum_cholesky_pivot,
            error,
            relative,
            finite,
            mass.evidence.symmetric,
            mass.evidence.positive_definite,
            mass.evidence.conditioned,
            result.successful & diagnostics.converged,
            roundtrip_valid,
            finite
            & mass.evidence.valid
            & result.successful
            & diagnostics.converged
            & roundtrip_valid,
            self.plan.solver,
            self.dynamics_id,
        )

        def inverse_action(value):
            tangent_rhs = self.reduction.coefficient_space.inverse_riesz(value)
            return solve(problem, tangent_rhs, policy=self.solve_policy).value

        inverse_operator = FunctionLinearOperator(
            inverse_action,
            source=self.reduction.reduced_effort_space,
            target=self.reduction.coefficient_space,
            operator_id=canonical_fingerprint(
                {"kind": "reduced-rod-inverse-mass", "dynamics": self.dynamics_id}
            ),
        )
        return ReducedRodInverseMassResult(
            acceleration, inverse_operator, mass, evidence, self.dynamics_id
        )

    def inverse_mass(
        self, coefficients: ArrayLike, effort: ArrayLike, /
    ) -> ReducedRodInverseMassResult:
        point = self.reduction.coefficient_space.validate(jnp.asarray(coefficients))
        return self._inverse_mass_from_mass(self.mass(point), effort)

    def bias(
        self, coefficients: ArrayLike, coefficient_velocities: ArrayLike, /
    ) -> ReducedRodBiasResult:
        point = self.reduction.coefficient_space.validate(jnp.asarray(coefficients))
        velocity = self.reduction.coefficient_space.validate(
            jnp.asarray(coefficient_velocities)
        )
        native_velocity = lift_reduced_rod_velocity(self.reduction, point, velocity)

        def lifted_at(values):
            return lift_reduced_rod_velocity(self.reduction, values, velocity)

        _, lift_acceleration = jax.jvp(lifted_at, (point,), (velocity,))
        inertial = self._native_inertia_action(lift_acceleration)
        if self.reduction.rod.plan.dimension == 2:
            gyroscopic_moments = jnp.zeros_like(native_velocity[1])
        else:
            angular_momentum = ein.contract(
                "sij,sj->si",
                self.reduction.rod.segment_inertias,
                native_velocity[1],
            )
            gyroscopic_moments = jnp.cross(native_velocity[1], angular_momentum)
        gyroscopic = (
            jnp.zeros_like(native_velocity[0]),
            gyroscopic_moments,
        )
        native_bias = (inertial[0], inertial[1] + gyroscopic[1])
        effort = lift_effort_pullback_operator(self.reduction, point).mv(native_bias)
        finite = (
            jnp.all(jnp.isfinite(effort))
            & jnp.all(jnp.isfinite(lift_acceleration[0]))
            & jnp.all(jnp.isfinite(lift_acceleration[1]))
            & jnp.all(jnp.isfinite(gyroscopic[0]))
            & jnp.all(jnp.isfinite(gyroscopic[1]))
        )
        return ReducedRodBiasResult(
            effort, lift_acceleration, gyroscopic, finite, self.dynamics_id
        )

    def _material_results(
        self,
        state: ReducedRodState,
        source_state: ReducedRodState | None,
        material_state: ReducedRodMaterialState | None,
        material_control: ReducedRodMaterialControl | None,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> tuple[RodConstitutiveResult, RodConstitutiveResult]:
        self.reduction.validate_state(state)
        if source_state is not None:
            self.reduction.validate_state(source_state)
        history = (
            self.initialize_material_state() if material_state is None else material_state
        )
        control = (
            self.initialize_material_control()
            if material_control is None
            else material_control
        )
        if not isinstance(history, ReducedRodMaterialState) or not isinstance(
            control, ReducedRodMaterialControl
        ):
            raise TypeError("material_state and material_control have invalid types.")
        step = jnp.asarray(step_size, dtype=state.coefficients.dtype)
        time_ = jnp.asarray(time, dtype=state.coefficients.dtype)
        stretch_increment, bend_increment = target_native_strains(
            self.reduction, state.coefficients
        )
        stretch_candidate = (
            self.reduction.rod.stretch_shear_reference_strains + stretch_increment
        )
        bend_candidate = self.reduction.rod.bend_twist_reference_strains + bend_increment
        if source_state is None:
            stretch_rate, bend_rate = target_native_strains(
                self.reduction, state.coefficient_velocities
            )
            stretch_source = stretch_candidate - step * stretch_rate
            bend_source = bend_candidate - step * bend_rate
        else:
            source_stretch, source_bend = target_native_strains(
                self.reduction, source_state.coefficients
            )
            stretch_source = (
                self.reduction.rod.stretch_shear_reference_strains + source_stretch
            )
            bend_source = self.reduction.rod.bend_twist_reference_strains + source_bend
            safe_step = jnp.where(step > 0.0, step, jnp.ones_like(step))
            stretch_rate = (stretch_candidate - stretch_source) / safe_step
            bend_rate = (bend_candidate - bend_source) / safe_step
        stretch_result = self.stretch_shear_material(
            stretch_source,
            stretch_candidate,
            stretch_rate,
            history.stretch_shear_history,
            control.stretch_shear_control,
            time_,
            step,
        )
        bend_result = self.bend_twist_material(
            bend_source,
            bend_candidate,
            bend_rate,
            history.bend_twist_history,
            control.bend_twist_control,
            time_,
            step,
        )
        return stretch_result, bend_result

    def _forces(
        self,
        state: ReducedRodState,
        stretch_result: RodConstitutiveResult,
        bend_result: RodConstitutiveResult,
        native_loads: RodLoadLedger | None,
        direct_reduced_loads: Sequence[ReducedRodDirectLoad],
        /,
    ) -> ReducedRodForceResult:
        # Constitutive trials own their controlled elastic/viscous split. This
        # kernel only applies fixed native quadrature and the strain-basis dual.
        stretch_elastic = stretch_result.elastic_resultants
        stretch_viscous = stretch_result.viscous_resultants
        bend_elastic = bend_result.elastic_resultants
        bend_viscous = bend_result.viscous_resultants

        def pull_resultants(stretch_values, bend_values):
            return -ein.contract(
                "sdk,sd,s->k",
                self.reduction.stretch_shear_basis,
                stretch_values,
                self.reduction.rod.stretch_shear_measures,
            ) - ein.contract(
                "sdk,sd,s->k",
                self.reduction.bend_twist_basis,
                bend_values,
                self.reduction.rod.bend_twist_measures,
            )

        elastic = self.reduction.reduced_effort_space.validate(
            pull_resultants(stretch_elastic, bend_elastic)
        )
        viscous = self.reduction.reduced_effort_space.validate(
            pull_resultants(stretch_viscous, bend_viscous)
        )
        zeros = jnp.zeros_like(state.coefficients)
        gravity = zeros
        external = zeros
        source_ids: list[str] = ["elastic", "kelvin_voigt"]
        source_channels: list[str] = ["elastic", "kelvin_voigt"]
        source_efforts: list[Array] = [elastic, viscous]
        if native_loads is not None and not isinstance(native_loads, RodLoadLedger):
            raise TypeError("native_loads must be a RodLoadLedger or None.")
        native_sequence: tuple[RodLoad, ...] = (
            () if self.gravity_load is None else (self.gravity_load,)
        ) + (() if native_loads is None else native_loads.loads)
        seen = set(source_ids)
        if native_sequence:
            ledger = RodLoadLedger(native_sequence)
            pullback = lift_effort_pullback_operator(self.reduction, state.coefficients)
            for load, effort in zip(
                ledger.loads, ledger.source_efforts(self.reduction.rod), strict=True
            ):
                if load.source_id in seen:
                    raise ValueError("Every reduced load source_id must be unique.")
                seen.add(load.source_id)
                reduced = pullback.mv(effort)
                source_ids.append(load.source_id)
                source_channels.append(load.power_channel)
                source_efforts.append(reduced)
                if load is self.gravity_load:
                    gravity = gravity + reduced
                else:
                    external = external + reduced
        direct = zeros
        for load in tuple(direct_reduced_loads):
            if not isinstance(load, ReducedRodDirectLoad):
                raise TypeError(
                    "direct_reduced_loads must contain ReducedRodDirectLoad values."
                )
            value = self.reduction.reduced_effort_space.validate(load.effort)
            if load.source_id in seen:
                raise ValueError("Every reduced load source_id must be unique.")
            seen.add(load.source_id)
            direct = direct + value
            source_ids.append(load.source_id)
            source_channels.append(load.power_channel)
            source_efforts.append(value)
        efforts = jnp.stack(source_efforts)
        velocity = self.reduction.coefficient_space.validate(state.coefficient_velocities)
        source_power = jax.vmap(
            lambda effort: self.reduction.reduced_effort_space.pair(effort, velocity)
        )(efforts)
        channel_names = tuple(dict.fromkeys(source_channels))
        channel_indices = tuple(
            jnp.asarray(
                [
                    index
                    for index, source_channel in enumerate(source_channels)
                    if source_channel == channel
                ],
                dtype=jnp.int32,
            )
            for channel in channel_names
        )
        channel_efforts = jnp.stack(
            tuple(jnp.sum(efforts[indices], axis=0) for indices in channel_indices)
        )
        channel_power = jnp.stack(
            tuple(jnp.sum(source_power[indices]) for indices in channel_indices)
        )
        total = self.reduction.reduced_effort_space.validate(jnp.sum(efforts, axis=0))
        total_power = jnp.sum(source_power)
        paired_power = self.reduction.reduced_effort_space.pair(total, velocity)
        power_residual = jnp.abs(total_power - paired_power)
        power_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=power_residual.dtype),
            jnp.maximum(jnp.abs(total_power), jnp.abs(paired_power)),
        )
        finite = (
            jnp.all(jnp.isfinite(efforts))
            & jnp.all(jnp.isfinite(channel_efforts))
            & jnp.all(jnp.isfinite(source_power))
            & jnp.all(jnp.isfinite(channel_power))
            & jnp.all(jnp.isfinite(total))
            & jnp.isfinite(total_power)
            & jnp.isfinite(paired_power)
            & jnp.isfinite(power_residual)
        )
        power_valid = power_residual <= (
            64.0 * jnp.finfo(power_residual.dtype).eps * power_scale
        )
        valid = finite & power_valid
        return ReducedRodForceResult(
            elastic,
            viscous,
            gravity,
            external,
            direct,
            total,
            efforts,
            channel_efforts,
            source_power,
            channel_power,
            total_power,
            paired_power,
            power_residual,
            finite,
            power_valid,
            valid,
            tuple(source_ids),
            tuple(source_channels),
            channel_names,
            self.dynamics_id,
        )

    def energy(
        self,
        state: ReducedRodState,
        /,
        *,
        source_state: ReducedRodState | None = None,
        material_state: ReducedRodMaterialState | None = None,
        material_control: ReducedRodMaterialControl | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike = 1.0,
    ) -> ReducedRodEnergyResult:
        stretch, bend = self._material_results(
            state,
            source_state,
            material_state,
            material_control,
            time,
            step_size,
        )
        native_velocity = lift_reduced_rod_velocity(
            self.reduction, state.coefficients, state.coefficient_velocities
        )
        native_momentum = self._native_inertia_action(native_velocity)
        kinetic = 0.5 * self.reduction.native_effort_space.pair(
            native_momentum, native_velocity
        )
        stored = stretch.stored_energy + bend.stored_energy
        dissipation = stretch.viscous_dissipation + bend.viscous_dissipation
        total = kinetic + stored
        finite = (
            jnp.isfinite(kinetic)
            & jnp.isfinite(stored)
            & jnp.isfinite(dissipation)
            & jnp.isfinite(total)
        )
        valid = finite & stretch.evidence.valid & bend.evidence.valid
        return ReducedRodEnergyResult(
            kinetic, stored, dissipation, total, finite, valid, self.dynamics_id
        )

    def evaluate(
        self,
        state: ReducedRodState,
        /,
        *,
        source_state: ReducedRodState | None = None,
        material_state: ReducedRodMaterialState | None = None,
        material_control: ReducedRodMaterialControl | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike = 1.0,
        native_loads: RodLoadLedger | None = None,
        direct_reduced_loads: Sequence[ReducedRodDirectLoad] = (),
    ) -> ReducedRodDynamicsEvaluation:
        stretch, bend = self._material_results(
            state,
            source_state,
            material_state,
            material_control,
            time,
            step_size,
        )
        mass = self.mass(state.coefficients)
        bias = self.bias(state.coefficients, state.coefficient_velocities)
        forces = self._forces(
            state,
            stretch,
            bend,
            native_loads,
            direct_reduced_loads,
        )
        native_velocity = lift_reduced_rod_velocity(
            self.reduction, state.coefficients, state.coefficient_velocities
        )
        kinetic = 0.5 * self.reduction.native_effort_space.pair(
            self._native_inertia_action(native_velocity), native_velocity
        )
        stored = stretch.stored_energy + bend.stored_energy
        dissipation = stretch.viscous_dissipation + bend.viscous_dissipation
        energy_finite = (
            jnp.isfinite(kinetic) & jnp.isfinite(stored) & jnp.isfinite(dissipation)
        )
        energy = ReducedRodEnergyResult(
            kinetic,
            stored,
            dissipation,
            kinetic + stored,
            energy_finite,
            energy_finite & stretch.evidence.valid & bend.evidence.valid,
            self.dynamics_id,
        )
        candidate = ReducedRodMaterialState(
            stretch.candidate_history, bend.candidate_history
        )
        finite = mass.evidence.finite & bias.finite & forces.finite & energy.finite
        valid = finite & mass.evidence.valid & forces.valid & energy.valid
        return ReducedRodDynamicsEvaluation(
            mass,
            bias,
            forces,
            energy,
            candidate,
            stretch,
            bend,
            finite,
            valid,
            self.dynamics_id,
        )

    def forward_dynamics(
        self, state: ReducedRodState, /, **kwargs
    ) -> ReducedRodForwardDynamicsResult:
        evaluation = self.evaluate(state, **kwargs)
        rhs = self.reduction.reduced_effort_space.validate(
            evaluation.forces.total_effort - evaluation.bias.effort
        )
        inverse = self._inverse_mass_from_mass(evaluation.mass, rhs)
        finite = evaluation.finite & inverse.solve_evidence.finite
        valid = evaluation.valid & inverse.solve_evidence.valid
        return ReducedRodForwardDynamicsResult(
            inverse.acceleration,
            rhs,
            evaluation,
            inverse.solve_evidence,
            finite,
            valid,
            self.dynamics_id,
        )

    def inverse_dynamics(
        self,
        state: ReducedRodState,
        acceleration: ArrayLike,
        /,
        **kwargs,
    ) -> ReducedRodInverseDynamicsResult:
        evaluation = self.evaluate(state, **kwargs)
        acceleration_ = self.reduction.coefficient_space.validate(
            jnp.asarray(acceleration)
        )
        dynamic = self.reduction.reduced_effort_space.validate(
            evaluation.mass.operator.mv(acceleration_) + evaluation.bias.effort
        )
        required = self.reduction.reduced_effort_space.validate(
            dynamic - evaluation.forces.total_effort
        )
        residual = required
        finite = (
            evaluation.finite
            & jnp.all(jnp.isfinite(dynamic))
            & jnp.all(jnp.isfinite(required))
        )
        return ReducedRodInverseDynamicsResult(
            required,
            dynamic,
            residual,
            evaluation,
            finite,
            evaluation.valid & finite,
            self.dynamics_id,
        )

    def dense_reference(
        self,
        state: ReducedRodState,
        /,
        *,
        source_state: ReducedRodState | None = None,
        material_state: ReducedRodMaterialState | None = None,
        material_control: ReducedRodMaterialControl | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike = 1.0,
    ) -> ReducedRodDenseReferenceResult:
        """Evaluate the explicitly labelled dense AD authority for parity tests."""
        q = state.coefficients
        v = state.coefficient_velocities

        def kinetic(configuration, velocity):
            native = lift_reduced_rod_velocity(self.reduction, configuration, velocity)
            momentum = self._native_inertia_action(native)
            return 0.5 * self.reduction.native_effort_space.pair(momentum, native)

        mass = jax.hessian(lambda velocity: kinetic(q, velocity))(v)
        momentum_derivative = jax.jvp(
            lambda configuration: jax.grad(kinetic, argnums=1)(configuration, v),
            (q,),
            (v,),
        )[1]
        bias = momentum_derivative - jax.grad(
            lambda configuration: kinetic(configuration, v)
        )(q)

        def stored(configuration):
            candidate = ReducedRodState(configuration, v)
            stretch, bend = self._material_results(
                candidate,
                source_state,
                material_state,
                material_control,
                time,
                step_size,
            )
            return stretch.stored_energy + bend.stored_energy

        elastic = -jax.grad(stored)(q)
        stretch, bend = self._material_results(
            state,
            source_state,
            material_state,
            material_control,
            time,
            step_size,
        )
        production = self._forces(state, stretch, bend, None, ())
        finite = (
            jnp.all(jnp.isfinite(mass))
            & jnp.all(jnp.isfinite(bias))
            & jnp.all(jnp.isfinite(elastic))
            & jnp.all(jnp.isfinite(production.kelvin_voigt_effort))
        )
        return ReducedRodDenseReferenceResult(
            mass,
            bias,
            elastic,
            production.kelvin_voigt_effort,
            kinetic(q, v),
            stored(q),
            finite,
            self.dynamics_id,
        )


def prepare_reduced_rod_dynamics(
    reduction: PreparedReducedRod,
    plan: ReducedRodDynamicsPlan | None = None,
    /,
    **kwargs,
) -> PreparedReducedRodDynamics:
    return PreparedReducedRodDynamics(reduction, plan, **kwargs)


def reduced_rod_mass(
    prepared: PreparedReducedRodDynamics, coefficients: ArrayLike, /
) -> ReducedRodMassResult:
    return prepared.mass(coefficients)


def reduced_rod_inverse_mass(
    prepared: PreparedReducedRodDynamics,
    coefficients: ArrayLike,
    effort: ArrayLike,
    /,
) -> ReducedRodInverseMassResult:
    return prepared.inverse_mass(coefficients, effort)


def reduced_rod_bias(
    prepared: PreparedReducedRodDynamics,
    coefficients: ArrayLike,
    coefficient_velocities: ArrayLike,
    /,
) -> ReducedRodBiasResult:
    return prepared.bias(coefficients, coefficient_velocities)


def reduced_rod_energy(
    prepared: PreparedReducedRodDynamics,
    state: ReducedRodState,
    /,
    **kwargs,
) -> ReducedRodEnergyResult:
    return prepared.energy(state, **kwargs)


def reduced_rod_dense_reference(
    prepared: PreparedReducedRodDynamics,
    state: ReducedRodState,
    /,
    **kwargs,
) -> ReducedRodDenseReferenceResult:
    return prepared.dense_reference(state, **kwargs)


def evaluate_reduced_rod_dynamics(
    prepared: PreparedReducedRodDynamics, state: ReducedRodState, /, **kwargs
) -> ReducedRodDynamicsEvaluation:
    return prepared.evaluate(state, **kwargs)


def reduced_rod_forward_dynamics(
    prepared: PreparedReducedRodDynamics, state: ReducedRodState, /, **kwargs
) -> ReducedRodForwardDynamicsResult:
    return prepared.forward_dynamics(state, **kwargs)


def reduced_rod_inverse_dynamics(
    prepared: PreparedReducedRodDynamics,
    state: ReducedRodState,
    acceleration: ArrayLike,
    /,
    **kwargs,
) -> ReducedRodInverseDynamicsResult:
    return prepared.inverse_dynamics(state, acceleration, **kwargs)


__all__ = [
    "PreparedReducedRodDynamics",
    "ReducedRodBiasResult",
    "ReducedRodDenseCholeskyPlan",
    "ReducedRodDenseReferenceResult",
    "ReducedRodDirectLoad",
    "ReducedRodDynamicsEvaluation",
    "ReducedRodDynamicsPlan",
    "ReducedRodEnergyResult",
    "ReducedRodForceResult",
    "ReducedRodForwardDynamicsResult",
    "ReducedRodInverseDynamicsResult",
    "ReducedRodInverseMassResult",
    "ReducedRodMassEvidence",
    "ReducedRodMassResult",
    "ReducedRodMaterialControl",
    "ReducedRodMaterialState",
    "ReducedRodMatrixFreeCGPlan",
    "ReducedRodSolveEvidence",
    "evaluate_reduced_rod_dynamics",
    "prepare_reduced_rod_dynamics",
    "reduced_rod_bias",
    "reduced_rod_dense_reference",
    "reduced_rod_energy",
    "reduced_rod_forward_dynamics",
    "reduced_rod_inverse_dynamics",
    "reduced_rod_inverse_mass",
    "reduced_rod_mass",
]

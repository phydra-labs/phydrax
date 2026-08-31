#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    ArraySpace,
    DenseInversePreconditionerBuilder,
    DenseLinearOperator,
    JacobiPreconditionerBuilder,
    OperatorProperties,
    PreconditioningPolicy,
)
from ....nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NonlinearPrecisionPolicy,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)
from ._blocks import MemberNetworkAssembly, MemberNetworkAssemblyState
from ._reference import (
    MemberKinematics,
    MemberNetworkDefinition,
)


class MemberNetworkStatus(IntEnum):
    SUCCESS = 0
    NONLINEAR_SOLVE_FAILED = 1
    NONFINITE_STATE = 2
    INVALID_GEOMETRY = 3
    EQUILIBRIUM_RESIDUAL_TOO_LARGE = 4
    AMBIGUOUS_CABLE_ACTIVE_SET = 5


_STATUS_MESSAGES = {
    MemberNetworkStatus.SUCCESS: "success",
    MemberNetworkStatus.NONLINEAR_SOLVE_FAILED: "nonlinear member equilibrium failed",
    MemberNetworkStatus.NONFINITE_STATE: "member-network state contains non-finite values",
    MemberNetworkStatus.INVALID_GEOMETRY: "member-network geometry or constitutive state is invalid",
    MemberNetworkStatus.EQUILIBRIUM_RESIDUAL_TOO_LARGE: "member-network equilibrium residual exceeds tolerance",
    MemberNetworkStatus.AMBIGUOUS_CABLE_ACTIVE_SET: "one or more cable members lie near the activation surface",
}


def member_network_status_message(status: int | MemberNetworkStatus, /) -> str:
    return _STATUS_MESSAGES[MemberNetworkStatus(int(status))]


class MemberNetworkTolerances(StrictModule, NonTrainableState):
    absolute_equilibrium: float = eqx.field(static=True)
    relative_equilibrium: float = eqx.field(static=True)
    minimum_length: float = eqx.field(static=True)
    maximum_rotation: float = eqx.field(static=True)
    strict_cable_margin: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_equilibrium: float = 1.0e-8,
        relative_equilibrium: float = 1.0e-8,
        minimum_length: float = 1.0e-12,
        maximum_rotation: float = 3.0,
        strict_cable_margin: float = 1.0e-7,
    ):
        values = tuple(
            float(value)
            for value in (
                absolute_equilibrium,
                relative_equilibrium,
                minimum_length,
                maximum_rotation,
                strict_cable_margin,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Member-network tolerances must be finite and positive.")
        (
            self.absolute_equilibrium,
            self.relative_equilibrium,
            self.minimum_length,
            self.maximum_rotation,
            self.strict_cable_margin,
        ) = values


class MemberNetworkInputs(StrictModule):
    prescribed_positions: Array
    prescribed_rotations: Array
    nodal_forces: Array
    nodal_moments: Array
    rest_lengths: Array
    initial_strain: Array
    initial_temperature: Array
    cable_active: Array

    def __init__(
        self,
        prescribed_positions: ArrayLike,
        prescribed_rotations: ArrayLike,
        nodal_forces: ArrayLike,
        nodal_moments: ArrayLike,
        rest_lengths: ArrayLike,
        /,
        *,
        initial_strain: ArrayLike | None = None,
        initial_temperature: ArrayLike | None = None,
        cable_active: ArrayLike | None = None,
    ):
        prescribed_positions_ = jnp.asarray(prescribed_positions)
        dtype = prescribed_positions_.dtype
        self.prescribed_positions = prescribed_positions_
        self.prescribed_rotations = jnp.asarray(prescribed_rotations, dtype=dtype)
        self.nodal_forces = jnp.asarray(nodal_forces, dtype=dtype)
        self.nodal_moments = jnp.asarray(nodal_moments, dtype=dtype)
        self.rest_lengths = jnp.asarray(rest_lengths, dtype=dtype)
        self.initial_strain = (
            jnp.zeros_like(self.rest_lengths)
            if initial_strain is None
            else jnp.asarray(initial_strain, dtype=dtype)
        )
        self.initial_temperature = (
            jnp.zeros_like(self.rest_lengths)
            if initial_temperature is None
            else jnp.asarray(initial_temperature, dtype=dtype)
        )
        self.cable_active = (
            jnp.ones(self.rest_lengths.shape, dtype=bool)
            if cable_active is None
            else jnp.asarray(cable_active, dtype=bool)
        )


class MemberNetworkProblem(StrictModule, NonTrainableState):
    definition: MemberNetworkDefinition
    assembly: MemberNetworkAssembly
    tolerances: MemberNetworkTolerances
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        definition: MemberNetworkDefinition,
        assembly: MemberNetworkAssembly,
        /,
        *,
        tolerances: MemberNetworkTolerances | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(definition, MemberNetworkDefinition):
            raise TypeError("definition must be a MemberNetworkDefinition.")
        if not isinstance(assembly, MemberNetworkAssembly):
            raise TypeError("assembly must be a MemberNetworkAssembly.")
        tolerances_ = MemberNetworkTolerances() if tolerances is None else tolerances
        self.definition = definition
        self.assembly = assembly
        self.tolerances = tolerances_
        self.problem_id = str(
            problem_id
            or canonical_fingerprint(
                {
                    "kind": "member-network-problem",
                    "definition": definition.definition_id,
                    "assembly": assembly.assembly_id,
                    "tolerances": values_from_tolerances(tolerances_),
                }
            )
        )


class MemberNetworkState(StrictModule):
    kinematics: MemberKinematics
    assembly: MemberNetworkAssemblyState
    internal_forces: Array
    internal_moments: Array
    applied_forces: Array
    applied_moments: Array
    force_residual: Array
    moment_residual: Array
    support_reactions: Array
    support_moments: Array


class CableActiveSetEvidence(StrictModule):
    active: Array
    slack: Array
    switching_margin: Array
    ambiguous: Array
    complementarity_residual: Array
    active_set_changes: Array
    cycle_detected: Array
    sensitivity_valid: Array


class MemberNetworkDiagnostics(StrictModule):
    residual_norm: Array
    relative_residual: Array
    finite: Array
    geometry_valid: Array
    active_set: CableActiveSetEvidence
    equilibrium_valid: Array


class MemberNetworkProvenance(StrictModule):
    problem_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    numeric_version: Array


class MemberNetworkResult(StrictModule):
    state: MemberNetworkState
    status: Array
    diagnostics: MemberNetworkDiagnostics
    provenance: MemberNetworkProvenance
    nonlinear_result: NonlinearResult | None

    @property
    def successful(self) -> Array:
        return self.status == int(MemberNetworkStatus.SUCCESS)

    @property
    def message(self) -> str:
        return member_network_status_message(int(self.status))


class MemberNetworkPlan(StrictModule, NonTrainableState):
    problem: MemberNetworkProblem
    nonlinear_template: PreparedNonlinearSolve | None
    nonlinear_method: AbstractNonlinearMethod
    termination: NonlinearTermination
    precision: NonlinearPrecisionPolicy
    derivative_policy: ImplicitRootDerivativePolicy | None
    input_signature: tuple[Any, ...] = eqx.field(static=True)
    uses_linear_setup: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedMemberNetworkSolve(StrictModule):
    plan: MemberNetworkPlan
    inputs: MemberNetworkInputs
    nonlinear_solve: PreparedNonlinearSolve | None
    initial_reduced: Array
    numeric_version: Array


def values_from_tolerances(value: MemberNetworkTolerances, /) -> dict[str, float]:
    return {
        "absolute_equilibrium": value.absolute_equilibrium,
        "relative_equilibrium": value.relative_equilibrium,
        "minimum_length": value.minimum_length,
        "maximum_rotation": value.maximum_rotation,
        "strict_cable_margin": value.strict_cable_margin,
    }


def _signature(inputs: MemberNetworkInputs, /) -> tuple[Any, ...]:
    leaves, tree = jax.tree.flatten(inputs)
    return (
        str(tree),
        tuple(tuple(int(size) for size in leaf.shape) for leaf in leaves),
        tuple(str(leaf.dtype) for leaf in leaves),
    )


def _validate_inputs(
    problem: MemberNetworkProblem, inputs: MemberNetworkInputs, /
) -> None:
    definition = problem.definition
    structure = definition.structure
    dofs = definition.dofs
    expected = (
        (
            "prescribed_positions",
            inputs.prescribed_positions,
            (structure.constrained_dof_count,),
        ),
        (
            "prescribed_rotations",
            inputs.prescribed_rotations,
            (dofs.constrained_rotation_indices.size,),
        ),
        (
            "nodal_forces",
            inputs.nodal_forces,
            (structure.node_count, structure.dimension),
        ),
        (
            "nodal_moments",
            inputs.nodal_moments,
            (structure.node_count, dofs.rotation_dimension),
        ),
        ("rest_lengths", inputs.rest_lengths, (structure.member_count,)),
        ("initial_strain", inputs.initial_strain, (structure.member_count,)),
        ("initial_temperature", inputs.initial_temperature, (structure.member_count,)),
    )
    for name, value, shape in expected:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError(f"{name} must use a real inexact dtype.")
    if inputs.cable_active.shape != (structure.member_count,):
        raise ValueError("cable_active must match the member count.")
    dtypes = {str(value.dtype) for _, value, _ in expected}
    if len(dtypes) != 1:
        raise TypeError("Member-network numerical input leaves must share one dtype.")


def _dynamic_definition(
    problem: MemberNetworkProblem, inputs: MemberNetworkInputs, /
) -> MemberNetworkDefinition:
    reference = eqx.tree_at(
        lambda selected: (
            selected.rest_lengths,
            selected.initial_strain,
            selected.initial_temperature,
            selected.cable_active,
        ),
        problem.definition.reference,
        (
            inputs.rest_lengths,
            inputs.initial_strain,
            inputs.initial_temperature,
            inputs.cable_active,
        ),
    )
    return eqx.tree_at(
        lambda selected: selected.reference,
        problem.definition,
        reference,
    )


def _generalized_load(
    definition: MemberNetworkDefinition, inputs: MemberNetworkInputs, /
) -> Array:
    translations = definition.structure.reduce(inputs.nodal_forces)
    rotations = inputs.nodal_moments.reshape((-1,))[definition.dofs.free_rotation_indices]
    return jnp.concatenate((translations, rotations))


def _energy(
    problem: MemberNetworkProblem,
    inputs: MemberNetworkInputs,
    reduced: Array,
    /,
) -> Array:
    definition = _dynamic_definition(problem, inputs)
    kinematics = definition.dofs.expand(
        reduced,
        inputs.prescribed_positions,
        inputs.prescribed_rotations,
    )
    return problem.assembly.evaluate(definition, kinematics).energy


def _residual(
    problem: MemberNetworkProblem,
    reduced: Array,
    inputs: MemberNetworkInputs,
    /,
) -> Array:
    internal = jax.grad(lambda value: _energy(problem, inputs, value))(reduced)
    return internal - _generalized_load(problem.definition, inputs)


def _tangent_setup_operator(
    problem: MemberNetworkProblem,
    inputs: MemberNetworkInputs,
    reduced: Array,
    /,
) -> DenseLinearOperator:
    tangent = jax.jacfwd(lambda value: _residual(problem, value, inputs))(reduced)
    tangent = 0.5 * (tangent + tangent.T)
    return DenseLinearOperator(
        tangent,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=f"{problem.problem_id}:tangent-setup",
    )


def _nonlinear_problem(
    problem: MemberNetworkProblem,
    dtype: Any,
    /,
    *,
    use_linear_setup: bool,
) -> NonlinearSystemProblem:
    space = ArraySpace((problem.definition.dofs.reduced_size,), dtype=dtype)

    def residual(reduced, inputs):
        return _residual(problem, reduced, inputs)

    def validity(reduced, residual_value, auxiliary, inputs):
        del residual_value, auxiliary
        definition = _dynamic_definition(problem, inputs)
        kinematics = definition.dofs.expand(
            reduced,
            inputs.prescribed_positions,
            inputs.prescribed_rotations,
        )
        vectors = (
            kinematics.positions[definition.structure.receivers]
            - kinematics.positions[definition.structure.senders]
        )
        lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        rotation_norm = jnp.sqrt(jnp.sum(kinematics.rotation_vectors**2, axis=-1))
        assembly = problem.assembly.evaluate(definition, kinematics)
        return (
            jnp.all(jnp.isfinite(kinematics.positions))
            & jnp.all(jnp.isfinite(kinematics.rotation_vectors))
            & jnp.all(
                (~definition.structure.member_valid)
                | (lengths > problem.tolerances.minimum_length)
            )
            & jnp.all(rotation_norm < problem.tolerances.maximum_rotation)
            & assembly.valid
        )

    def linear_setup(reduced, inputs):
        return _tangent_setup_operator(problem, inputs, reduced)

    return NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        validity=validity,
        linear_setup=linear_setup if use_linear_setup else None,
        problem_id=f"{problem.problem_id}:equilibrium-root",
    )


def plan_member_network(
    problem: MemberNetworkProblem,
    sample_inputs: MemberNetworkInputs,
    initial_kinematics: MemberKinematics,
    /,
    *,
    nonlinear_method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
) -> MemberNetworkPlan:
    _validate_inputs(problem, sample_inputs)
    initial = problem.definition.dofs.reduce(
        initial_kinematics.positions, initial_kinematics.rotation_vectors
    )
    if nonlinear_method is None:
        method = NewtonKrylov()
        initial_assembly = problem.assembly.evaluate(
            _dynamic_definition(problem, sample_inputs),
            initial_kinematics,
        )
        if bool(jnp.any(initial_assembly.unilateral)):
            uses_linear_setup = False
        else:
            setup = _tangent_setup_operator(problem, sample_inputs, initial)
            builder = (
                DenseInversePreconditionerBuilder()
                if problem.definition.dofs.reduced_size <= 512
                else JacobiPreconditionerBuilder()
            )
            preconditioning = PreconditioningPolicy(
                builder,
                setup_operator=setup,
                side="right",
                refresh="numeric",
            )
            linear_policy = eqx.tree_at(
                lambda selected: selected.preconditioning,
                method.linear_policy,
                preconditioning,
                is_leaf=lambda value: value is None,
            )
            method = eqx.tree_at(
                lambda selected: selected.linear_policy,
                method,
                linear_policy,
            )
            uses_linear_setup = True
    else:
        method = nonlinear_method
        uses_linear_setup = False
    termination_ = (
        NonlinearTermination(
            absolute_residual=problem.tolerances.absolute_equilibrium,
            relative_residual=problem.tolerances.relative_equilibrium,
        )
        if termination is None
        else termination
    )
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if derivative_policy is None and uses_linear_setup:
        derivative_linear_policy = eqx.tree_at(
            lambda selected: selected.preconditioning,
            method.linear_policy,
            None,
        )
        derivative_policy = ImplicitRootDerivativePolicy(
            tangent_linear_policy=derivative_linear_policy,
            adjoint_linear_policy=derivative_linear_policy,
        )
    template = None
    if problem.definition.dofs.reduced_size:
        template = prepare_nonlinear(
            _nonlinear_problem(
                problem,
                sample_inputs.rest_lengths.dtype,
                use_linear_setup=uses_linear_setup,
            ),
            initial,
            method=method,
            termination=termination_,
            args=sample_inputs,
            precision=precision_,
        )
    signature = _signature(sample_inputs)
    plan_id = canonical_fingerprint(
        {
            "kind": "member-network-plan",
            "problem": problem.problem_id,
            "method": method.method_id,
            "termination": values_from_tolerances(problem.tolerances),
            "input_signature": repr(signature),
            "linear_template": None if template is None else template.linear_template_id,
            "derivative": repr(derivative_policy),
        }
    )
    return MemberNetworkPlan(
        problem,
        template,
        method,
        termination_,
        precision_,
        derivative_policy,
        signature,
        uses_linear_setup,
        plan_id,
    )


def _prepare(
    plan: MemberNetworkPlan,
    inputs: MemberNetworkInputs,
    initial_kinematics: MemberKinematics,
    numeric_version: Any,
    seed: PreparedNonlinearSolve | None,
    /,
) -> PreparedMemberNetworkSolve:
    _validate_inputs(plan.problem, inputs)
    if _signature(inputs) != plan.input_signature:
        raise ValueError("Member-network numeric refresh changed input structure.")
    initial = plan.problem.definition.dofs.reduce(
        initial_kinematics.positions, initial_kinematics.rotation_vectors
    )
    nonlinear = None
    source = plan.nonlinear_template if seed is None else seed
    if source is not None:
        nonlinear = refresh_nonlinear(
            source,
            _nonlinear_problem(
                plan.problem,
                inputs.rest_lengths.dtype,
                use_linear_setup=plan.uses_linear_setup,
            ),
            initial,
            args=inputs,
        )
    return PreparedMemberNetworkSolve(
        plan,
        inputs,
        nonlinear,
        initial,
        jnp.asarray(numeric_version, dtype=jnp.int32),
    )


def prepare_member_network(
    plan: MemberNetworkPlan,
    inputs: MemberNetworkInputs,
    initial_kinematics: MemberKinematics,
    /,
) -> PreparedMemberNetworkSolve:
    return _prepare(plan, inputs, initial_kinematics, 0, None)


def refresh_member_network(
    prepared: PreparedMemberNetworkSolve,
    inputs: MemberNetworkInputs,
    initial_kinematics: MemberKinematics,
    /,
) -> PreparedMemberNetworkSolve:
    return _prepare(
        prepared.plan,
        inputs,
        initial_kinematics,
        prepared.numeric_version + 1,
        prepared.nonlinear_solve,
    )


def _full_internal(
    problem: MemberNetworkProblem,
    inputs: MemberNetworkInputs,
    kinematics: MemberKinematics,
    /,
) -> tuple[Array, Array]:
    definition = _dynamic_definition(problem, inputs)

    def energy(positions, rotations):
        return problem.assembly.evaluate(
            definition, MemberKinematics(positions, rotations)
        ).energy

    return jax.grad(energy, argnums=(0, 1))(
        kinematics.positions, kinematics.rotation_vectors
    )


def solve_member_network(
    prepared: PreparedMemberNetworkSolve,
    /,
) -> MemberNetworkResult:
    plan = prepared.plan
    problem = plan.problem
    definition = problem.definition
    nonlinear_result = None
    if definition.dofs.reduced_size:
        if prepared.nonlinear_solve is None:
            raise RuntimeError("Prepared nonlinear member-network state is unavailable.")
        nonlinear_result = implicit_root_result(
            prepared.nonlinear_solve,
            derivative_policy=plan.derivative_policy,
        )
        reduced = nonlinear_result.state
    else:
        reduced = jnp.empty((0,), dtype=prepared.inputs.rest_lengths.dtype)
    dynamic = _dynamic_definition(problem, prepared.inputs)
    kinematics = definition.dofs.expand(
        reduced,
        prepared.inputs.prescribed_positions,
        prepared.inputs.prescribed_rotations,
    )
    assembly = problem.assembly.evaluate(dynamic, kinematics)
    internal_force, internal_moment = _full_internal(problem, prepared.inputs, kinematics)
    force_residual = internal_force - prepared.inputs.nodal_forces
    moment_residual = internal_moment - prepared.inputs.nodal_moments
    structure = definition.structure
    if structure.affine_constraints:
        if structure.affine_prolongation is None:
            raise RuntimeError("Affine translation prolongation is unavailable.")
        flat = force_residual.reshape((-1,))
        free = structure.affine_prolongation @ (structure.affine_prolongation.T @ flat)
        support_force = (flat - free).reshape(force_residual.shape)
    else:
        support_force = jnp.where(
            structure.constrained_dofs,
            force_residual,
            0.0,
        )
    support_moment = jnp.where(
        definition.dofs.rotation_constrained,
        moment_residual,
        0.0,
    )
    generalized = _residual(problem, reduced, prepared.inputs)
    residual_norm = jnp.sqrt(jnp.sum(generalized**2))
    load_norm = jnp.sqrt(jnp.sum(_generalized_load(definition, prepared.inputs) ** 2))
    relative = residual_norm / jnp.maximum(load_norm, jnp.finfo(residual_norm.dtype).tiny)
    finite = (
        jnp.all(jnp.isfinite(kinematics.positions))
        & jnp.all(jnp.isfinite(kinematics.rotation_vectors))
        & jnp.all(jnp.isfinite(assembly.axial_force))
        & jnp.isfinite(assembly.energy)
    )
    ambiguous = (
        assembly.unilateral
        & (assembly.switching_margin < problem.tolerances.strict_cable_margin)
        & definition.structure.member_valid
    )
    complementarity = jnp.max(
        jnp.where(
            assembly.active,
            jnp.maximum(-assembly.axial_force, 0.0),
            jnp.abs(assembly.axial_force),
        ),
        initial=0.0,
    )
    active_set = CableActiveSetEvidence(
        assembly.active,
        ~assembly.active & structure.member_valid,
        assembly.switching_margin,
        ambiguous,
        complementarity,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        ~jnp.any(ambiguous),
    )
    threshold = problem.tolerances.absolute_equilibrium + (
        problem.tolerances.relative_equilibrium * load_norm
    )
    equilibrium_valid = residual_norm <= threshold
    nested_success = (
        jnp.asarray(True) if nonlinear_result is None else nonlinear_result.successful
    )
    status = jnp.where(
        ~nested_success,
        int(MemberNetworkStatus.NONLINEAR_SOLVE_FAILED),
        jnp.where(
            ~finite,
            int(MemberNetworkStatus.NONFINITE_STATE),
            jnp.where(
                ~assembly.valid,
                int(MemberNetworkStatus.INVALID_GEOMETRY),
                jnp.where(
                    ~equilibrium_valid,
                    int(MemberNetworkStatus.EQUILIBRIUM_RESIDUAL_TOO_LARGE),
                    jnp.where(
                        jnp.any(ambiguous),
                        int(MemberNetworkStatus.AMBIGUOUS_CABLE_ACTIVE_SET),
                        int(MemberNetworkStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    state = MemberNetworkState(
        kinematics,
        assembly,
        internal_force,
        internal_moment,
        prepared.inputs.nodal_forces,
        prepared.inputs.nodal_moments,
        force_residual,
        moment_residual,
        support_force,
        support_moment,
    )
    diagnostics = MemberNetworkDiagnostics(
        residual_norm,
        relative,
        finite,
        assembly.valid,
        active_set,
        equilibrium_valid,
    )
    provenance = MemberNetworkProvenance(
        problem.problem_id,
        definition.definition_id,
        problem.assembly.assembly_id,
        plan.plan_id,
        prepared.numeric_version,
    )
    return MemberNetworkResult(
        state,
        status,
        diagnostics,
        provenance,
        nonlinear_result,
    )


def member_network_equilibrium(
    problem: MemberNetworkProblem,
    inputs: MemberNetworkInputs,
    initial_kinematics: MemberKinematics,
    /,
    **plan_options,
) -> MemberNetworkResult:
    plan = plan_member_network(
        problem,
        inputs,
        initial_kinematics,
        **plan_options,
    )
    return solve_member_network(prepare_member_network(plan, inputs, initial_kinematics))


__all__ = [
    "CableActiveSetEvidence",
    "MemberNetworkDiagnostics",
    "MemberNetworkInputs",
    "MemberNetworkPlan",
    "MemberNetworkProblem",
    "MemberNetworkProvenance",
    "MemberNetworkResult",
    "MemberNetworkState",
    "MemberNetworkStatus",
    "MemberNetworkTolerances",
    "PreparedMemberNetworkSolve",
    "member_network_equilibrium",
    "member_network_status_message",
    "plan_member_network",
    "prepare_member_network",
    "refresh_member_network",
    "solve_member_network",
]

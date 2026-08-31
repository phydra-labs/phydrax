#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_signature, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    bind_numeric,
    DifferentiationPolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveTemplate,
    LinearSystem,
    OperatorProperties,
    prepare_template,
    PreparedLinearSolve,
    solve as solve_linear,
)
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ...sparse import SparseCoordinateOperator
from ._force_density_loads import (
    AbstractForceDensityLoadModel,
    FixedNodalLoadModel,
)
from ._force_density_topology import ForceDensityStructure


ForceDensitySignMode: TypeAlias = Literal[
    "tension", "compression", "fixed-mixed", "unrestricted"
]


class ForceDensityStatus(IntEnum):
    """Portable status for one force-density equilibrium result."""

    SUCCESS = 0
    LINEAR_SOLVE_FAILED = 1
    NONLINEAR_SOLVE_FAILED = 2
    NONFINITE_STATE = 3
    EQUILIBRIUM_RESIDUAL_TOO_LARGE = 4
    DEGENERATE_MEMBER = 5
    INVALID_LOAD_GEOMETRY = 6


class ForceDensityTolerances(StrictModule, NonTrainableState):
    """Physical input margins and final equilibrium tolerances."""

    absolute_equilibrium: float = eqx.field(static=True)
    relative_equilibrium: float = eqx.field(static=True)
    minimum_force_density: float = eqx.field(static=True)
    minimum_member_length: float = eqx.field(static=True)
    prescribed_position: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_equilibrium: float = 1.0e-8,
        relative_equilibrium: float = 1.0e-8,
        minimum_force_density: float = 1.0e-8,
        minimum_member_length: float = 1.0e-12,
        prescribed_position: float = 1.0e-12,
    ):
        values = tuple(
            float(value)
            for value in (
                absolute_equilibrium,
                relative_equilibrium,
                minimum_force_density,
                minimum_member_length,
                prescribed_position,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Force-density tolerances must be finite and nonnegative.")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("Force-density and member-length margins must be positive.")
        (
            self.absolute_equilibrium,
            self.relative_equilibrium,
            self.minimum_force_density,
            self.minimum_member_length,
            self.prescribed_position,
        ) = values


class ForceDensityInputs(StrictModule):
    """Dynamic force densities, prescribed coordinates, and load parameters."""

    force_densities: Array
    prescribed_values: Array
    load_parameters: Any

    def __init__(
        self,
        force_densities: ArrayLike,
        prescribed_values: ArrayLike,
        load_parameters: Any,
        /,
    ):
        densities = jnp.asarray(force_densities)
        prescribed = jnp.asarray(prescribed_values)
        for name, value in (
            ("force_densities", densities),
            ("prescribed_values", prescribed),
        ):
            if value.ndim != 1:
                raise ValueError(f"{name} must be rank-1.")
            if not jnp.issubdtype(value.dtype, jnp.inexact) or jnp.iscomplexobj(value):
                raise TypeError(f"{name} must be a real inexact array.")
        self.force_densities = densities
        self.prescribed_values = prescribed
        self.load_parameters = load_parameters


class ForceDensityProblem(StrictModule, NonTrainableState):
    """One fixed-topology force-density equilibrium problem."""

    structure: ForceDensityStructure
    load_model: AbstractForceDensityLoadModel
    sign_mode: ForceDensitySignMode = eqx.field(static=True)
    fixed_signs: Array | None
    tolerances: ForceDensityTolerances
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure: ForceDensityStructure,
        /,
        *,
        load_model: AbstractForceDensityLoadModel | None = None,
        sign_mode: ForceDensitySignMode = "unrestricted",
        fixed_signs: ArrayLike | None = None,
        tolerances: ForceDensityTolerances | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(structure, ForceDensityStructure):
            raise TypeError("structure must be a ForceDensityStructure.")
        model = FixedNodalLoadModel() if load_model is None else load_model
        if not isinstance(model, AbstractForceDensityLoadModel):
            raise TypeError("load_model must be an AbstractForceDensityLoadModel.")
        if sign_mode not in (
            "tension",
            "compression",
            "fixed-mixed",
            "unrestricted",
        ):
            raise ValueError("Unknown force-density sign mode.")
        signs = None
        if sign_mode == "fixed-mixed":
            if fixed_signs is None:
                raise ValueError("fixed-mixed mode requires fixed_signs.")
            sign_array = np.asarray(fixed_signs)
            if sign_array.shape != (structure.member_count,):
                raise ValueError("fixed_signs must contain one sign per member.")
            active = np.asarray(structure.member_valid, dtype=bool)
            if np.any(np.abs(sign_array[active]) != 1):
                raise ValueError("Active fixed_signs must be -1 or +1.")
            signs = jnp.asarray(sign_array, dtype=float)
        elif fixed_signs is not None:
            raise ValueError("fixed_signs is valid only in fixed-mixed mode.")
        tolerances_ = ForceDensityTolerances() if tolerances is None else tolerances
        if not isinstance(tolerances_, ForceDensityTolerances):
            raise TypeError("tolerances must be ForceDensityTolerances or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "force-density-problem",
                    "structure": structure.structure_id,
                    "load_model": model.load_model_id,
                    "sign_mode": sign_mode,
                    "fixed_signs": (
                        None if signs is None else np.asarray(signs).tolist()
                    ),
                    "tolerances": {
                        "absolute_equilibrium": tolerances_.absolute_equilibrium,
                        "relative_equilibrium": tolerances_.relative_equilibrium,
                        "minimum_force_density": tolerances_.minimum_force_density,
                        "minimum_member_length": tolerances_.minimum_member_length,
                        "prescribed_position": tolerances_.prescribed_position,
                    },
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.structure = structure
        self.load_model = model
        self.sign_mode = sign_mode
        self.fixed_signs = signs
        self.tolerances = tolerances_
        self.problem_id = identifier


class ForceDensityState(StrictModule):
    """Complete physical state of one pin-jointed force-density system."""

    positions: Array
    member_vectors: Array
    member_lengths: Array
    force_densities: Array
    axial_forces: Array
    applied_nodal_loads: Array
    internal_nodal_forces: Array
    equilibrium_residual: Array
    support_reactions: Array
    node_valid: Array
    member_valid: Array
    constrained_dofs: Array


class ForceDensityDiagnostics(StrictModule):
    """Physical equilibrium and geometric validity evidence."""

    free_residual_norm: Array
    relative_free_residual: Array
    global_balance_norm: Array
    minimum_member_length: Array
    degenerate_member_count: Array
    minimum_force_density: Array
    finite: Array
    load_geometry_valid: Array
    equilibrium_valid: Array


class ForceDensityProvenance(StrictModule):
    """Static problem, plan, route, and sign identities."""

    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    load_model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    route: Literal["linear", "nonlinear"] = eqx.field(static=True)
    sign_mode: ForceDensitySignMode = eqx.field(static=True)


class ForceDensityResult(StrictModule):
    """Physical state paired with solver and certification evidence."""

    state: ForceDensityState
    status: Array
    diagnostics: ForceDensityDiagnostics
    provenance: ForceDensityProvenance
    linear_result: LinearSolveResult | None
    nonlinear_result: NonlinearResult | None

    @property
    def successful(self) -> Array:
        return self.status == int(ForceDensityStatus.SUCCESS)


class ForceDensityPlan(StrictModule, NonTrainableState):
    """Symbolic force-density execution plan."""

    problem: ForceDensityProblem
    linear_policy: LinearSolvePolicy
    linear_template: LinearSolveTemplate | None
    nonlinear_method: AbstractNonlinearMethod
    nonlinear_termination: NonlinearTermination
    derivative_policy: ImplicitRootDerivativePolicy | None
    plan_id: str = eqx.field(static=True)


class PreparedForceDensitySolve(StrictModule):
    """Numerically bound force-density inputs and optional initial state."""

    plan: ForceDensityPlan
    inputs: ForceDensityInputs
    linear_solve: PreparedLinearSolve | None
    initial_reduced: Array | None
    numeric_version: Array


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(jnp.asarray(value) ** 2))


def _validated_force_densities(
    problem: ForceDensityProblem,
    inputs: ForceDensityInputs,
    /,
) -> Array:
    structure = problem.structure
    densities = inputs.force_densities
    if densities.shape != (structure.member_count,):
        raise ValueError(
            "force_densities must have shape "
            f"({structure.member_count},); got {densities.shape}."
        )
    if inputs.prescribed_values.shape != (structure.constrained_dof_count,):
        raise ValueError(
            "prescribed_values must have shape "
            f"({structure.constrained_dof_count},); "
            f"got {inputs.prescribed_values.shape}."
        )
    active = structure.member_valid
    margin = jnp.asarray(problem.tolerances.minimum_force_density, dtype=densities.dtype)
    invalid = active & (~jnp.isfinite(densities) | (jnp.abs(densities) < margin))
    if problem.sign_mode == "tension":
        invalid = invalid | (active & (densities <= margin))
    elif problem.sign_mode == "compression":
        invalid = invalid | (active & (densities >= -margin))
    elif problem.sign_mode == "fixed-mixed":
        if problem.fixed_signs is None:
            raise RuntimeError("Fixed-mixed signs are unavailable.")
        invalid = invalid | (
            active & (jnp.sign(densities) != problem.fixed_signs.astype(densities.dtype))
        )
    checked = eqx.error_if(
        densities,
        jnp.any(invalid),
        "Active force densities violate the finite magnitude or sign contract.",
    )
    prescribed = eqx.error_if(
        inputs.prescribed_values,
        jnp.any(~jnp.isfinite(inputs.prescribed_values)),
        "Prescribed force-density coordinates must be finite.",
    )
    if prescribed.dtype != checked.dtype:
        raise TypeError("force_densities and prescribed_values must share one dtype.")
    return jnp.where(active, checked, 0.0)


def _normalization(problem: ForceDensityProblem, /) -> float:
    return -1.0 if problem.sign_mode == "compression" else 1.0


def _operator(
    problem: ForceDensityProblem,
    force_densities: Array,
    /,
) -> SparseCoordinateOperator:
    structure = problem.structure
    scale = jnp.asarray(_normalization(problem), dtype=force_densities.dtype)
    coefficients = (
        structure.route_signs.astype(force_densities.dtype)
        * force_densities[structure.route_members]
        * scale
    )
    space = ArraySpace((structure.free_dof_count,), dtype=force_densities.dtype)
    sign_definite = problem.sign_mode in ("tension", "compression")
    properties = OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=sign_definite,
        positive_definite=sign_definite,
        rank=structure.free_dof_count if sign_definite else None,
        evidence={
            "self_adjoint": "construction",
            **(
                {
                    "positive_semidefinite": "construction",
                    "positive_definite": "construction",
                    "rank": "construction",
                }
                if sign_definite
                else {}
            ),
        },
    )
    return SparseCoordinateOperator(
        structure.equilibrium_relation,
        coefficients,
        source=space,
        target=space,
        properties=properties,
        operator_id=f"{problem.problem_id}:reduced-equilibrium",
    )


def _linear_problem(
    problem: ForceDensityProblem,
    force_densities: Array,
    /,
) -> LinearSystem:
    return LinearSystem(
        _operator(problem, force_densities),
        problem_id=f"{problem.problem_id}:linear-state",
    )


def _member_vectors(structure: ForceDensityStructure, positions: Array, /) -> Array:
    vectors = positions[structure.receivers] - positions[structure.senders]
    return jnp.where(structure.member_valid[:, None], vectors, 0.0)


def _internal_nodal_forces(
    structure: ForceDensityStructure,
    force_densities: Array,
    positions: Array,
    /,
) -> Array:
    vectors = _member_vectors(structure, positions)
    member_vectors = force_densities[:, None] * vectors
    internal = jnp.zeros(
        (structure.node_count, structure.dimension), dtype=member_vectors.dtype
    )
    internal = internal.at[structure.senders].add(-member_vectors)
    return internal.at[structure.receivers].add(member_vectors)


def _nodal_loads(
    problem: ForceDensityProblem,
    inputs: ForceDensityInputs,
    positions: Array,
    dtype: Any,
    /,
) -> Array:
    structure = problem.structure
    loads = jnp.asarray(
        problem.load_model.nodal_loads(structure, positions, inputs.load_parameters)
    )
    expected = (structure.node_count, structure.dimension)
    if loads.shape != expected:
        raise ValueError(f"Load model must return shape {expected}; got {loads.shape}.")
    if loads.dtype != jnp.dtype(dtype):
        raise TypeError(
            "Load-model output must share the force-density coordinate dtype."
        )
    return loads


def _linear_rhs(
    problem: ForceDensityProblem,
    inputs: ForceDensityInputs,
    force_densities: Array,
    /,
) -> Array:
    structure = problem.structure
    lift = structure.lift(inputs.prescribed_values)
    loads = _nodal_loads(problem, inputs, lift, force_densities.dtype)
    boundary = _internal_nodal_forces(structure, force_densities, lift)
    reduced = structure.reduce(loads - boundary)
    return jnp.asarray(_normalization(problem), dtype=reduced.dtype) * reduced


def _initial_reduced(
    problem: ForceDensityProblem,
    inputs: ForceDensityInputs,
    initial_positions: ArrayLike | None,
    /,
) -> Array | None:
    structure = problem.structure
    if not problem.load_model.depends_on_positions:
        if initial_positions is not None:
            raise ValueError(
                "initial_positions are valid only for position-dependent loads."
            )
        return None
    if initial_positions is None:
        raise ValueError("Position-dependent loads require initial_positions.")
    positions = jnp.asarray(initial_positions)
    expected = (structure.node_count, structure.dimension)
    if positions.shape != expected or positions.dtype != inputs.force_densities.dtype:
        raise ValueError(
            "initial_positions must match the structure shape and force-density dtype."
        )
    positions = eqx.error_if(
        positions,
        jnp.any(structure.node_valid[:, None] & ~jnp.isfinite(positions)),
        "Active initial positions must be finite.",
    )
    prescribed = structure.prescribed_values(positions)
    mismatch = jnp.max(jnp.abs(prescribed - inputs.prescribed_values), initial=0.0)
    positions = eqx.error_if(
        positions,
        mismatch > problem.tolerances.prescribed_position,
        "Initial positions do not satisfy the prescribed coordinate values.",
    )
    valid = problem.load_model.valid(structure, positions, inputs.load_parameters)
    vectors = _member_vectors(structure, positions)
    lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    valid = valid & jnp.all(
        (~structure.member_valid) | (lengths > problem.tolerances.minimum_member_length)
    )
    positions = eqx.error_if(
        positions,
        ~valid,
        "Initial force-density geometry is invalid for the declared load model.",
    )
    return structure.reduce(positions)


def plan_force_density(
    problem: ForceDensityProblem,
    sample_inputs: ForceDensityInputs,
    /,
    *,
    linear_policy: LinearSolvePolicy | None = None,
    nonlinear_method: AbstractNonlinearMethod | None = None,
    nonlinear_termination: NonlinearTermination | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
    initial_positions: ArrayLike | None = None,
) -> ForceDensityPlan:
    """Freeze symbolic equilibrium and solver structure for repeated solves."""
    if not isinstance(problem, ForceDensityProblem):
        raise TypeError("problem must be a ForceDensityProblem.")
    if not isinstance(sample_inputs, ForceDensityInputs):
        raise TypeError("sample_inputs must be ForceDensityInputs.")
    force_densities = _validated_force_densities(problem, sample_inputs)
    initial = _initial_reduced(problem, sample_inputs, initial_positions)
    policy = (
        LinearSolvePolicy(
            differentiation=DifferentiationPolicy("mathematical"),
            require_device_binding=True,
        )
        if linear_policy is None
        else linear_policy
    )
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")
    method = NewtonKrylov() if nonlinear_method is None else nonlinear_method
    termination = (
        NonlinearTermination(
            absolute_residual=problem.tolerances.absolute_equilibrium,
            relative_residual=problem.tolerances.relative_equilibrium,
        )
        if nonlinear_termination is None
        else nonlinear_termination
    )
    if not isinstance(method, AbstractNonlinearMethod):
        raise TypeError("nonlinear_method must be AbstractNonlinearMethod or None.")
    if not isinstance(termination, NonlinearTermination):
        raise TypeError("nonlinear_termination must be NonlinearTermination or None.")
    if derivative_policy is not None and not isinstance(
        derivative_policy, ImplicitRootDerivativePolicy
    ):
        raise TypeError("derivative_policy must be ImplicitRootDerivativePolicy or None.")

    template = None
    if problem.structure.free_dof_count and not problem.load_model.depends_on_positions:
        linear_problem = _linear_problem(problem, force_densities)
        template = prepare_template(linear_problem, policy)
    route = "nonlinear" if problem.load_model.depends_on_positions else "linear"
    identifier = canonical_fingerprint(
        {
            "kind": "force-density-plan",
            "problem": problem.problem_id,
            "route": route,
            "linear_method": policy.method.name,
            "linear_template": None if template is None else template.template_id,
            "nonlinear_method": method.method_id,
            "input_signature": array_tree_signature(
                (sample_inputs.force_densities, sample_inputs.prescribed_values)
            ),
            "initial_shape": None if initial is None else list(initial.shape),
        }
    )
    return ForceDensityPlan(
        problem,
        policy,
        template,
        method,
        termination,
        derivative_policy,
        identifier,
    )


def prepare_force_density(
    plan: ForceDensityPlan,
    inputs: ForceDensityInputs,
    /,
    *,
    initial_positions: ArrayLike | None = None,
    numeric_version: Any = 0,
) -> PreparedForceDensitySolve:
    """Bind numeric force densities, supports, loads, and an optional root seed."""
    if not isinstance(plan, ForceDensityPlan):
        raise TypeError("plan must be a ForceDensityPlan.")
    if not isinstance(inputs, ForceDensityInputs):
        raise TypeError("inputs must be ForceDensityInputs.")
    force_densities = _validated_force_densities(plan.problem, inputs)
    initial = _initial_reduced(plan.problem, inputs, initial_positions)
    linear = None
    if plan.linear_template is not None:
        linear = bind_numeric(
            plan.linear_template,
            _linear_problem(plan.problem, force_densities),
            numeric_version=numeric_version,
        )
    return PreparedForceDensitySolve(
        plan,
        inputs,
        linear,
        initial,
        jnp.asarray(numeric_version, dtype=jnp.int32),
    )


def refresh_force_density(
    prepared: PreparedForceDensitySolve,
    inputs: ForceDensityInputs,
    /,
    *,
    initial_positions: ArrayLike | None = None,
) -> PreparedForceDensitySolve:
    """Refresh only numeric inputs while preserving symbolic solve identity."""
    if not isinstance(prepared, PreparedForceDensitySolve):
        raise TypeError("prepared must be a PreparedForceDensitySolve.")
    return prepare_force_density(
        prepared.plan,
        inputs,
        initial_positions=initial_positions,
        numeric_version=prepared.numeric_version + 1,
    )


def _nonlinear_problem(plan: ForceDensityPlan, dtype: Any, /) -> NonlinearSystemProblem:
    problem = plan.problem
    structure = problem.structure
    space = ArraySpace((structure.free_dof_count,), dtype=dtype)

    def residual(reduced, inputs):
        force_densities = _validated_force_densities(problem, inputs)
        positions = structure.expand(reduced, inputs.prescribed_values)
        loads = _nodal_loads(problem, inputs, positions, force_densities.dtype)
        internal = _internal_nodal_forces(structure, force_densities, positions)
        return structure.reduce(internal - loads)

    def validity(reduced, residual_value, auxiliary, inputs):
        del residual_value, auxiliary
        positions = structure.expand(reduced, inputs.prescribed_values)
        vectors = _member_vectors(structure, positions)
        lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        return (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(
                (~structure.member_valid)
                | (lengths > problem.tolerances.minimum_member_length)
            )
            & problem.load_model.valid(structure, positions, inputs.load_parameters)
        )

    return NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        validity=validity,
        problem_id=f"{problem.problem_id}:nonlinear-state",
    )


def _physical_state(
    problem: ForceDensityProblem,
    inputs: ForceDensityInputs,
    force_densities: Array,
    positions: Array,
    /,
) -> tuple[ForceDensityState, ForceDensityDiagnostics]:
    structure = problem.structure
    vectors = _member_vectors(structure, positions)
    lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    lengths = jnp.where(structure.member_valid, lengths, 0.0)
    axial = jnp.where(structure.member_valid, force_densities * lengths, 0.0)
    loads = _nodal_loads(problem, inputs, positions, force_densities.dtype)
    internal = _internal_nodal_forces(structure, force_densities, positions)
    residual = loads - internal
    reactions = jnp.where(
        structure.constrained_dofs & structure.node_valid[:, None],
        internal - loads,
        0.0,
    )
    free_residual = structure.reduce(residual)
    free_load = structure.reduce(loads)
    free_internal = structure.reduce(internal)
    residual_norm = _norm(free_residual)
    scale = jnp.maximum(_norm(free_load), _norm(free_internal))
    tiny = jnp.finfo(force_densities.dtype).tiny
    relative_residual = residual_norm / jnp.maximum(scale, tiny)
    global_balance = _norm(jnp.sum(loads + reactions, axis=0))
    active_lengths = jnp.where(structure.member_valid, lengths, jnp.inf)
    minimum_length = jnp.min(active_lengths)
    degenerate = structure.member_valid & (
        lengths <= problem.tolerances.minimum_member_length
    )
    active_density = jnp.where(structure.member_valid, jnp.abs(force_densities), jnp.inf)
    minimum_density = jnp.min(active_density)
    finite = (
        jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(loads))
        & jnp.all(jnp.isfinite(internal))
        & jnp.all(jnp.isfinite(lengths))
        & jnp.all(jnp.isfinite(axial))
    )
    load_valid = problem.load_model.valid(structure, positions, inputs.load_parameters)
    threshold = problem.tolerances.absolute_equilibrium + (
        problem.tolerances.relative_equilibrium * scale
    )
    equilibrium_valid = (residual_norm <= threshold) & (global_balance <= threshold)
    state = ForceDensityState(
        positions,
        vectors,
        lengths,
        force_densities,
        axial,
        loads,
        internal,
        residual,
        reactions,
        structure.node_valid,
        structure.member_valid,
        structure.constrained_dofs,
    )
    diagnostics = ForceDensityDiagnostics(
        residual_norm,
        relative_residual,
        global_balance,
        minimum_length,
        jnp.sum(degenerate, dtype=jnp.int32),
        minimum_density,
        finite,
        load_valid,
        equilibrium_valid,
    )
    return state, diagnostics


def _status(
    diagnostics: ForceDensityDiagnostics,
    /,
    *,
    linear_result: LinearSolveResult | None,
    nonlinear_result: NonlinearResult | None,
) -> Array:
    nested_success = jnp.asarray(True)
    nested_failure = ForceDensityStatus.SUCCESS
    if linear_result is not None:
        nested_success = linear_result.successful
        nested_failure = ForceDensityStatus.LINEAR_SOLVE_FAILED
    if nonlinear_result is not None:
        nested_success = nonlinear_result.successful
        nested_failure = ForceDensityStatus.NONLINEAR_SOLVE_FAILED
    return jnp.where(
        ~nested_success,
        int(nested_failure),
        jnp.where(
            ~diagnostics.finite,
            int(ForceDensityStatus.NONFINITE_STATE),
            jnp.where(
                ~diagnostics.load_geometry_valid,
                int(ForceDensityStatus.INVALID_LOAD_GEOMETRY),
                jnp.where(
                    diagnostics.degenerate_member_count > 0,
                    int(ForceDensityStatus.DEGENERATE_MEMBER),
                    jnp.where(
                        ~diagnostics.equilibrium_valid,
                        int(ForceDensityStatus.EQUILIBRIUM_RESIDUAL_TOO_LARGE),
                        int(ForceDensityStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def solve_force_density(
    prepared: PreparedForceDensitySolve,
    /,
) -> ForceDensityResult:
    """Solve one prepared force-density equilibrium problem."""
    if not isinstance(prepared, PreparedForceDensitySolve):
        raise TypeError("prepared must be a PreparedForceDensitySolve.")
    plan = prepared.plan
    problem = plan.problem
    structure = problem.structure
    inputs = prepared.inputs
    force_densities = _validated_force_densities(problem, inputs)
    linear_result = None
    nonlinear_result = None

    if structure.free_dof_count == 0:
        reduced = jnp.empty((0,), dtype=force_densities.dtype)
    elif problem.load_model.depends_on_positions:
        if prepared.initial_reduced is None:
            raise RuntimeError("Prepared nonlinear force-density seed is unavailable.")
        nonlinear_result = implicit_root_result(
            _nonlinear_problem(plan, force_densities.dtype),
            prepared.initial_reduced,
            method=plan.nonlinear_method,
            termination=plan.nonlinear_termination,
            derivative_policy=plan.derivative_policy,
            args=inputs,
        )
        reduced = nonlinear_result.state
    else:
        if prepared.linear_solve is None:
            raise RuntimeError("Prepared linear force-density solve is unavailable.")
        rhs = _linear_rhs(problem, inputs, force_densities)
        linear_result = solve_linear(prepared.linear_solve, rhs)
        reduced = linear_result.value

    positions = structure.expand(reduced, inputs.prescribed_values)
    state, diagnostics = _physical_state(problem, inputs, force_densities, positions)
    status = _status(
        diagnostics,
        linear_result=linear_result,
        nonlinear_result=nonlinear_result,
    )
    route: Literal["linear", "nonlinear"] = (
        "nonlinear" if problem.load_model.depends_on_positions else "linear"
    )
    provenance = ForceDensityProvenance(
        problem.problem_id,
        structure.structure_id,
        problem.load_model.load_model_id,
        plan.plan_id,
        route,
        problem.sign_mode,
    )
    return ForceDensityResult(
        state,
        status,
        diagnostics,
        provenance,
        linear_result,
        nonlinear_result,
    )


def force_density_equilibrium(
    problem: ForceDensityProblem,
    inputs: ForceDensityInputs,
    /,
    *,
    linear_policy: LinearSolvePolicy | None = None,
    nonlinear_method: AbstractNonlinearMethod | None = None,
    nonlinear_termination: NonlinearTermination | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
    initial_positions: ArrayLike | None = None,
) -> ForceDensityResult:
    """Plan, bind, and solve one force-density equilibrium problem."""
    plan = plan_force_density(
        problem,
        inputs,
        linear_policy=linear_policy,
        nonlinear_method=nonlinear_method,
        nonlinear_termination=nonlinear_termination,
        derivative_policy=derivative_policy,
        initial_positions=initial_positions,
    )
    prepared = prepare_force_density(plan, inputs, initial_positions=initial_positions)
    return solve_force_density(prepared)


def force_density_load_path(
    state: ForceDensityState,
    member_mask: ArrayLike | None = None,
    /,
) -> Array:
    """Return total axial load path over selected active members."""
    if not isinstance(state, ForceDensityState):
        raise TypeError("state must be a ForceDensityState.")
    selected = (
        state.member_valid
        if member_mask is None
        else jnp.asarray(member_mask, dtype=bool)
    )
    if selected.shape != state.member_valid.shape:
        raise ValueError("member_mask must match the member axis.")
    selected = selected & state.member_valid
    return jnp.sum(
        jnp.where(selected, jnp.abs(state.axial_forces) * state.member_lengths, 0.0)
    )


__all__ = [
    "ForceDensityDiagnostics",
    "ForceDensityInputs",
    "ForceDensityPlan",
    "ForceDensityProblem",
    "ForceDensityProvenance",
    "ForceDensityResult",
    "ForceDensitySignMode",
    "ForceDensityState",
    "ForceDensityStatus",
    "ForceDensityTolerances",
    "PreparedForceDensitySolve",
    "force_density_equilibrium",
    "force_density_load_path",
    "plan_force_density",
    "prepare_force_density",
    "refresh_force_density",
    "solve_force_density",
]

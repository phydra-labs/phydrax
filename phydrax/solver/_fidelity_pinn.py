#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model import AbstractArrayModel
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..domain import DomainFunction
from ..enforcement import EnforcementProgram
from ..fidelity import FidelityPath, FidelityRelation
from ..nn._keys import split_eval_key
from ..terms import PreparedFidelityObservation, ResidualPenalty
from ._functional_correction import (
    freeze_domain_function,
    FunctionalCorrectionProblem,
    prepare_functional_correction,
)
from ._functional_solver import FunctionalSolver


class FidelityFieldTransfer(StrictModule, NonTrainableState):
    """Explicit continuous-field transfer between adjacent fidelity levels."""

    transform: Callable[[DomainFunction], DomainFunction] | None = eqx.field(static=True)
    source_level_id: str = eqx.field(static=True)
    target_level_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_level_id: str,
        target_level_id: str,
        /,
        *,
        transfer_id: str | None = None,
        transform: Callable[[DomainFunction], DomainFunction] | None = None,
    ):
        source = str(source_level_id)
        target = str(target_level_id)
        if not source or not target or source == target:
            raise ValueError("Fidelity transfer levels must be distinct and non-empty.")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable or None.")
        identifier = (
            f"identity:{source}:{target}" if transfer_id is None else str(transfer_id)
        )
        if not identifier:
            raise ValueError("transfer_id must be non-empty.")
        if transform is not None and transfer_id is None:
            raise ValueError("A non-identity field transfer requires an explicit ID.")
        self.transform = transform
        self.source_level_id = source
        self.target_level_id = target
        self.transfer_id = identifier

    def __call__(self, field: DomainFunction, /) -> DomainFunction:
        if not isinstance(field, DomainFunction):
            raise TypeError("Fidelity field transfer requires a DomainFunction.")
        transferred = field if self.transform is None else self.transform(field)
        if not isinstance(transferred, DomainFunction):
            raise TypeError("Fidelity field transfer must return a DomainFunction.")
        return transferred


class _ConditionedCorrectionEvaluator(StrictModule):
    parent: DomainFunction
    model: AbstractArrayModel
    correction_id: str = eqx.field(static=True)

    def __init__(
        self,
        parent: DomainFunction,
        model: AbstractArrayModel,
        correction_id: str,
        /,
    ):
        self.parent = freeze_domain_function(parent)
        self.model = model
        self.correction_id = correction_id

    def __call__(self, *coordinates: Any, key=None, **kwargs: Any):
        parent_key, model_key = split_eval_key(key, 2)
        parent = self.parent.func(*coordinates, key=parent_key, **kwargs)
        packed = jnp.concatenate(
            (
                *tuple(jnp.ravel(jnp.asarray(value)) for value in coordinates),
                jnp.ravel(parent),
            )
        )
        return self.model(packed, key=model_key)


def condition_fidelity_correction(
    parent: DomainFunction,
    model: AbstractArrayModel,
    /,
    *,
    correction_id: str,
) -> DomainFunction:
    """Create a correction model conditioned on coordinates and a frozen parent field."""

    if not isinstance(parent, DomainFunction):
        raise TypeError("parent must be a DomainFunction.")
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("model must implement AbstractArrayModel.")
    identifier = str(correction_id)
    if not identifier:
        raise ValueError("correction_id must be non-empty.")
    return DomainFunction(
        domain=parent.domain,
        deps=parent.deps,
        func=_ConditionedCorrectionEvaluator(parent, model, identifier),
        metadata={"fidelity_conditioned_correction_id": identifier},
    )


class FidelityPINNResult(StrictModule, NonTrainableState):
    """One trained functional field bound to a fidelity path level."""

    path: FidelityPath
    solver: FunctionalSolver
    level_id: str = eqx.field(static=True)
    source_result_id: str | None = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)
    training_observation_ids: tuple[str, ...] = eqx.field(static=True)
    validation_observation_ids: tuple[str, ...] = eqx.field(static=True)
    training_group_ids: tuple[str, ...] = eqx.field(static=True)
    validation_group_ids: tuple[str, ...] = eqx.field(static=True)
    training_run_id: str | None = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: FidelityPath,
        solver: FunctionalSolver,
        /,
        *,
        level_id: str,
        stage_id: str,
        training_observations: Sequence[PreparedFidelityObservation] = (),
        validation_observations: Sequence[PreparedFidelityObservation] = (),
        source_result_id: str | None = None,
    ):
        if not isinstance(path, FidelityPath):
            raise TypeError("path must be a FidelityPath.")
        if not isinstance(solver, FunctionalSolver):
            raise TypeError("solver must be a FunctionalSolver.")
        level = str(level_id)
        if level not in path.level_ids:
            raise ValueError("Fidelity PINN result level must lie on its path.")
        stage = str(stage_id)
        if not stage:
            raise ValueError("stage_id must be non-empty.")
        training = _validate_observations(
            training_observations,
            path,
            level,
            solver,
            owner="training",
        )
        validation = _validate_observations(
            validation_observations,
            path,
            level,
            solver,
            owner="validation",
        )
        _require_disjoint_observation_groups(training, validation)
        source = None if source_result_id is None else str(source_result_id)
        if source is not None and not source:
            raise ValueError("source_result_id must be non-empty when provided.")
        training_state = solver.training_state
        run_id = None if training_state is None else training_state.run_id
        self.path = path
        self.solver = solver
        self.level_id = level
        self.source_result_id = source
        self.stage_id = stage
        self.training_observation_ids = tuple(
            item.observation_set_id for item in training
        )
        self.validation_observation_ids = tuple(
            item.observation_set_id for item in validation
        )
        self.training_group_ids = tuple(
            sorted({group for item in training for group in item.split_group_ids})
        )
        self.validation_group_ids = tuple(
            sorted({group for item in validation for group in item.split_group_ids})
        )
        self.training_run_id = run_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "fidelity-pinn-result",
                "path": path.path_id,
                "level": level,
                "source_result": source,
                "stage": stage,
                "solver_bundle": solver.discretization_bundle.bundle_id,
                "parameters": array_tree_fingerprint(solver.functions),
                "training_observations": list(self.training_observation_ids),
                "validation_observations": list(self.validation_observation_ids),
                "training_run": run_id,
            }
        )

    @property
    def functions(self):
        return self.solver.ansatz_functions()


class FidelityPINNStage(StrictModule, NonTrainableState):
    """Prepared frozen-parent correction stage for one target fidelity."""

    path: FidelityPath
    relation: FidelityRelation
    correction_problem: FunctionalCorrectionProblem
    parent_result: FidelityPINNResult
    training_observations: tuple[PreparedFidelityObservation, ...]
    validation_observations: tuple[PreparedFidelityObservation, ...]
    term_roles: tuple[tuple[str, int], ...] = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    @property
    def training_solver(self) -> FunctionalSolver:
        return self.correction_problem.training_solver

    @property
    def physical_solver(self) -> FunctionalSolver:
        return self.correction_problem.physical_solver

    @property
    def source_level_id(self) -> str:
        return self.relation.source_level_id

    @property
    def target_level_id(self) -> str:
        return self.relation.target_level_id

    @property
    def selection_kind(self) -> str:
        return "target_data" if self.validation_observations else "none"

    def finalize(self, trained: FunctionalSolver, /) -> FidelityPINNResult:
        physical = self.correction_problem.finalize(trained)
        return FidelityPINNResult(
            self.path,
            physical,
            level_id=self.target_level_id,
            stage_id=self.stage_id,
            training_observations=self.training_observations,
            validation_observations=self.validation_observations,
            source_result_id=self.parent_result.result_id,
        )


def bind_fidelity_pinn_level(
    path: FidelityPath,
    level_id: str,
    solver: FunctionalSolver,
    /,
    *,
    training_observations: Sequence[PreparedFidelityObservation] = (),
    validation_observations: Sequence[PreparedFidelityObservation] = (),
    stage_id: str | None = None,
) -> FidelityPINNResult:
    """Bind an already trained ordinary FunctionalSolver to one fidelity level."""

    level = str(level_id)
    identifier = (
        canonical_fingerprint(
            {
                "kind": "fidelity-pinn-base-stage",
                "path": path.path_id,
                "level": level,
                "solver_bundle": solver.discretization_bundle.bundle_id,
            }
        )
        if stage_id is None
        else str(stage_id)
    )
    return FidelityPINNResult(
        path,
        solver,
        level_id=level,
        stage_id=identifier,
        training_observations=training_observations,
        validation_observations=validation_observations,
    )


def prepare_fidelity_pinn_stage(
    parent: FidelityPINNResult,
    target_level_id: str,
    correction_functions: Mapping[str, DomainFunction],
    target_terms: Sequence[ResidualPenalty],
    /,
    *,
    training_observations: Sequence[PreparedFidelityObservation] = (),
    validation_observations: Sequence[PreparedFidelityObservation] = (),
    epsilon: float,
    field_transfers: Mapping[str, FidelityFieldTransfer] | None = None,
    parent_scales: Mapping[str, float] | None = None,
    replacement_functions: Mapping[str, DomainFunction] | None = None,
    enforcement: EnforcementProgram | None = None,
) -> FidelityPINNStage:
    """Prepare one adjacent target-fidelity PINN correction stage."""

    if not isinstance(parent, FidelityPINNResult):
        raise TypeError("parent must be a FidelityPINNResult.")
    target = str(target_level_id)
    source_index = parent.path.level_ids.index(parent.level_id)
    if source_index + 1 >= parent.path.num_levels:
        raise ValueError("Parent fidelity has no successor on this path.")
    if parent.path.level_ids[source_index + 1] != target:
        raise ValueError("Target fidelity must immediately follow the parent level.")
    relation = parent.path.relations[source_index]
    if relation.target_level_id != target:
        raise ValueError("Fidelity relation and target level do not match.")
    training = _validate_observations(
        training_observations,
        parent.path,
        target,
        None,
        owner="training",
    )
    validation = _validate_observations(
        validation_observations,
        parent.path,
        target,
        None,
        owner="validation",
    )
    _require_disjoint_observation_groups(training, validation)
    terms = tuple(target_terms)
    if not all(isinstance(term, ResidualPenalty) for term in terms):
        raise TypeError("target_terms must contain only ResidualPenalty values.")
    corrections = dict(correction_functions)
    if not corrections:
        raise ValueError("At least one target correction field is required.")
    transfers = {} if field_transfers is None else dict(field_transfers)
    scales = {} if parent_scales is None else dict(parent_scales)
    unknown_transfers = tuple(name for name in transfers if name not in parent.functions)
    if unknown_transfers:
        raise KeyError(
            f"Field transfers reference unknown parent fields {unknown_transfers}."
        )
    unknown_scales = tuple(name for name in scales if name not in parent.functions)
    if unknown_scales:
        raise KeyError(f"Parent scales reference unknown fields {unknown_scales}.")
    base_functions: dict[str, DomainFunction] = {}
    transfer_ids: list[str] = []
    for name, field in parent.functions.items():
        transfer = transfers.get(
            name,
            FidelityFieldTransfer(parent.level_id, target),
        )
        if not isinstance(transfer, FidelityFieldTransfer):
            raise TypeError(
                "field_transfers values must be FidelityFieldTransfer objects."
            )
        if (
            transfer.source_level_id != parent.level_id
            or transfer.target_level_id != target
        ):
            raise ValueError("Field transfer endpoints do not match the fidelity stage.")
        if (
            relation.observable_transfer_id is not None
            and name in corrections
            and transfer.transfer_id != relation.observable_transfer_id
        ):
            raise ValueError(
                "Field transfer identity does not match the fidelity relation."
            )
        scale = float(scales.get(name, 1.0))
        if not isfinite(scale):
            raise ValueError("Parent field scales must be finite.")
        base_functions[name] = scale * transfer(field)
        transfer_ids.append(transfer.transfer_id)
    all_terms = (*terms, *(item.term for item in training))
    physical = FunctionalSolver(
        functions=base_functions,
        terms=all_terms,
        evaluation_terms=tuple(item.term for item in validation),
        enforcement=enforcement,
    )
    problem = prepare_functional_correction(
        physical,
        corrections,
        epsilon=epsilon,
        replacement_functions=replacement_functions,
    )
    stage_id = canonical_fingerprint(
        {
            "kind": "fidelity-pinn-stage",
            "path": parent.path.path_id,
            "relation": relation.relation_id,
            "source_result": parent.result_id,
            "target": target,
            "epsilon": float(epsilon),
            "transfers": transfer_ids,
            "training_observations": [item.observation_set_id for item in training],
            "validation_observations": [item.observation_set_id for item in validation],
            "training_bundle": problem.training_solver.discretization_bundle.bundle_id,
            "physical_bundle": problem.physical_solver.discretization_bundle.bundle_id,
        }
    )
    problem = _bind_fidelity_stage_record(problem, stage_id)
    roles = tuple(
        [("target_physics", index) for index in range(len(terms))]
        + [("target_data", len(terms) + index) for index in range(len(training))]
    )
    return FidelityPINNStage(
        path=parent.path,
        relation=relation,
        correction_problem=problem,
        parent_result=parent,
        training_observations=training,
        validation_observations=validation,
        term_roles=roles,
        stage_id=stage_id,
    )


class FidelityPINNEvaluation(StrictModule, NonTrainableState):
    """Held-out target-fidelity data and physics evidence."""

    target_data_rmse: Array
    target_data_relative_l2: Array
    target_data_accuracy: Array
    physics_losses: Array
    result_id: str = eqx.field(static=True)
    level_id: str = eqx.field(static=True)
    test_observation_ids: tuple[str, ...] = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


def evaluate_fidelity_pinn(
    result: FidelityPINNResult,
    test_observations: Sequence[PreparedFidelityObservation],
    /,
    *,
    physics_terms: Sequence[ResidualPenalty] = (),
) -> FidelityPINNEvaluation:
    """Evaluate one finalized PINN only against held-out target-fidelity evidence."""

    if not isinstance(result, FidelityPINNResult):
        raise TypeError("result must be a FidelityPINNResult.")
    observations = _validate_observations(
        test_observations,
        result.path,
        result.level_id,
        result.solver,
        owner="test",
    )
    if not observations:
        raise ValueError("Target PINN evaluation requires held-out target observations.")
    test_groups = {group for item in observations for group in item.split_group_ids}
    used_groups = set(result.training_group_ids) | set(result.validation_group_ids)
    overlap = tuple(sorted(test_groups & used_groups))
    if overlap:
        raise ValueError(
            f"Target test groups were used before final evaluation: {overlap}."
        )
    functions = result.functions
    counts = jnp.asarray([len(item.evaluation_ids) for item in observations], dtype=float)
    metrics = tuple(item.term.data_metrics(functions) for item in observations)
    total = jnp.sum(counts)
    rmse = jnp.sqrt(
        jnp.sum(counts * jnp.stack(tuple(metric["data_rmse"] ** 2 for metric in metrics)))
        / total
    )
    relative_l2 = (
        jnp.sum(
            counts
            * jnp.stack(tuple(metric["data_relative_l2_error"] for metric in metrics))
        )
        / total
    )
    accuracy = (
        jnp.sum(counts * jnp.stack(tuple(metric["data_accuracy"] for metric in metrics)))
        / total
    )
    physics = tuple(physics_terms)
    if not all(isinstance(term, ResidualPenalty) for term in physics):
        raise TypeError("physics_terms must contain only ResidualPenalty values.")
    physics_losses = (
        jnp.stack(tuple(term.loss(functions) for term in physics))
        if physics
        else jnp.zeros((0,), dtype=float)
    )
    observation_ids = tuple(item.observation_set_id for item in observations)
    evaluation_id = canonical_fingerprint(
        {
            "kind": "fidelity-pinn-evaluation",
            "result": result.result_id,
            "level": result.level_id,
            "observations": list(observation_ids),
            "rmse": float(rmse),
            "relative_l2": float(relative_l2),
            "physics_losses": [float(value) for value in physics_losses],
        }
    )
    return FidelityPINNEvaluation(
        target_data_rmse=rmse,
        target_data_relative_l2=relative_l2,
        target_data_accuracy=accuracy,
        physics_losses=physics_losses,
        result_id=result.result_id,
        level_id=result.level_id,
        test_observation_ids=observation_ids,
        evaluation_id=evaluation_id,
    )


def _validate_observations(
    observations: Sequence[PreparedFidelityObservation],
    path: FidelityPath,
    level_id: str,
    solver: FunctionalSolver | None,
    /,
    *,
    owner: str,
) -> tuple[PreparedFidelityObservation, ...]:
    values = tuple(observations)
    if not all(isinstance(item, PreparedFidelityObservation) for item in values):
        raise TypeError(
            f"{owner}_observations must contain prepared fidelity observations."
        )
    for item in values:
        if item.hierarchy_fingerprint != path.hierarchy_fingerprint:
            raise ValueError(
                f"{owner} observation hierarchy does not match the PINN path."
            )
        if item.level_id != level_id:
            raise ValueError(
                f"{owner} observations must belong to fidelity {level_id!r}."
            )
        if solver is not None and item.field_name not in solver.functions:
            raise KeyError(
                f"{owner} observation field {item.field_name!r} is absent from the solver."
            )
    return values


def _require_disjoint_observation_groups(
    training: Sequence[PreparedFidelityObservation],
    validation: Sequence[PreparedFidelityObservation],
    /,
) -> None:
    training_groups = {group for item in training for group in item.split_group_ids}
    validation_groups = {group for item in validation for group in item.split_group_ids}
    overlap = tuple(sorted(training_groups & validation_groups))
    if overlap:
        raise ValueError(
            f"Fidelity PINN training and validation groups overlap: {overlap}."
        )


def _bind_fidelity_stage_record(
    problem: FunctionalCorrectionProblem,
    stage_id: str,
    /,
) -> FunctionalCorrectionProblem:
    training = _append_stage_record(problem.training_solver, stage_id)
    physical = _append_stage_record(problem.physical_solver, stage_id)
    return FunctionalCorrectionProblem(
        training,
        physical,
        problem.base_functions,
        problem.correction_functions,
        problem.replacement_functions,
        problem.epsilon,
    )


def _append_stage_record(solver: FunctionalSolver, stage_id: str, /) -> FunctionalSolver:
    bundle = solver.discretization_bundle
    labels = tuple(
        dict.fromkeys(
            dependency
            for function in solver.functions.values()
            for dependency in function.deps
        )
    )
    record = DiscretizationRecord(
        DiscretizationKey(
            "fidelity-pinn-stage",
            DiscretizationRole.AUXILIARY,
            domain_labels=labels,
        ),
        "fidelity-pinn-stage",
        stage_id,
        dependency_key_ids=tuple(item.key.key_id for item in bundle.records),
    )
    enriched = DiscretizationBundle(
        (*bundle.records, record),
        transfers=bundle.transfers,
        stochastic_coupling_ids=bundle.stochastic_coupling_ids,
    )
    return eqx.tree_at(lambda item: item.discretization_bundle, solver, enriched)


__all__ = [
    "FidelityFieldTransfer",
    "FidelityPINNEvaluation",
    "FidelityPINNResult",
    "FidelityPINNStage",
    "bind_fidelity_pinn_level",
    "condition_fidelity_correction",
    "evaluate_fidelity_pinn",
    "prepare_fidelity_pinn_stage",
]

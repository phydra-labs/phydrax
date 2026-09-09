#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import DiscreteFieldSpace, FieldTransfer
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    AdjointLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    LinearSubspace,
    MaterializationPolicy,
    PreconditionerCostEstimate,
    PreconditionerProperties,
    PreconditionerSource,
    SubspaceCorrectionTerm,
)
from .data import FunctionSamples, OperatorBatch
from .eigen import operator_trial_subspace, OperatorTrialSubspace
from .training._execution import PreparedOperatorInput
from .training._trained_operator import TrainedOperator


def _resident_storage_bytes(value: object, /) -> int:
    arrays = {id(leaf): leaf for leaf in jax.tree.leaves(value) if eqx.is_array(leaf)}
    return sum(int(array.size * array.dtype.itemsize) for array in arrays.values())


def _source_identifier(source: PreconditionerSource, /) -> str:
    if isinstance(source, AbstractPreconditioner):
        return source.preconditioner_id
    if isinstance(source, AbstractPreconditionerBuilder):
        return source.builder_id
    raise TypeError("local_solver must be a preconditioner or builder.")


def _validate_setup_operator(
    operator: AbstractLinearOperator,
    space: DiscreteFieldSpace,
    /,
) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("setup_operator must be an AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError("Learned preconditioning requires an unbatched setup operator.")
    if not operator.source.compatible(operator.target):
        raise ValueError("Learned preconditioning requires an endomorphism.")
    if not operator.source.compatible(space.vector_space):
        raise ValueError("Setup operator and learned correction spaces must match.")


class OperatorSubspaceCorrection(StrictModule, NonTrainableState):
    """One transferred operator basis lowered to native subspace correction."""

    trial: OperatorTrialSubspace
    transferred_subspace: LinearSubspace
    term: SubspaceCorrectionTerm
    source_field_space_id: str = eqx.field(static=True)
    target_field_space_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        trial: OperatorTrialSubspace,
        transferred_subspace: LinearSubspace,
        term: SubspaceCorrectionTerm,
        /,
        *,
        source_field_space_id: str,
        target_field_space_id: str,
        transfer_id: str,
        preparation_id: str,
    ):
        if not isinstance(trial, OperatorTrialSubspace):
            raise TypeError("trial must be an OperatorTrialSubspace.")
        if not isinstance(transferred_subspace, LinearSubspace):
            raise TypeError("transferred_subspace must be a LinearSubspace.")
        if not isinstance(term, SubspaceCorrectionTerm):
            raise TypeError("term must be a SubspaceCorrectionTerm.")
        identifiers = tuple(
            str(value)
            for value in (
                source_field_space_id,
                target_field_space_id,
                transfer_id,
                preparation_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Operator subspace correction IDs must be non-empty.")
        self.trial = trial
        self.transferred_subspace = transferred_subspace
        self.term = term
        (
            self.source_field_space_id,
            self.target_field_space_id,
            self.transfer_id,
            self.preparation_id,
        ) = identifiers


def prepare_operator_subspace_correction(
    samples: FunctionSamples,
    model_space: DiscreteFieldSpace,
    solver_space: DiscreteFieldSpace,
    transfer: FieldTransfer,
    local_solver: PreconditionerSource,
    /,
) -> OperatorSubspaceCorrection:
    """Transfer physical basis fields into one native Galerkin correction term."""
    if not isinstance(model_space, DiscreteFieldSpace) or not isinstance(
        solver_space, DiscreteFieldSpace
    ):
        raise TypeError("model_space and solver_space must be DiscreteFieldSpace values.")
    if not isinstance(transfer, FieldTransfer):
        raise TypeError("transfer must be a FieldTransfer.")
    if samples.support_id != model_space.support_id:
        raise ValueError("Basis samples must use the model field-space support.")
    if transfer.source.field_space_id != model_space.field_space_id:
        raise ValueError("Basis transfer source must match model_space.")
    if transfer.target.field_space_id != solver_space.field_space_id:
        raise ValueError("Basis transfer target must match solver_space.")

    trial = operator_trial_subspace(samples, model_space.vector_space)
    columns = tuple(
        solver_space.vector_space.flatten(
            transfer.primal_operator.mv(
                model_space.vector_space.unflatten(trial.basis[:, index])
            )
        )
        for index in range(trial.capacity)
    )
    transferred_basis = jnp.stack(columns, axis=1)
    source_id = _source_identifier(local_solver)
    preparation_id = canonical_fingerprint(
        {
            "kind": "operator-subspace-correction-preparation",
            "source_field_space": model_space.field_space_id,
            "target_field_space": solver_space.field_space_id,
            "transfer": transfer.transfer_id,
            "support": samples.support_id,
            "measure": samples.measure_id,
            "sample_shape": list(trial.sample_shape),
            "value_shape": list(trial.value_shape),
            "capacity": trial.capacity,
            "basis": array_tree_fingerprint(transferred_basis),
            "local_solver": source_id,
        }
    )
    subspace = LinearSubspace(
        solver_space.vector_space,
        transferred_basis,
        subspace_id=canonical_fingerprint(
            {
                "kind": "operator-transferred-linear-subspace",
                "preparation": preparation_id,
            }
        ),
    )
    coarse_space = ArraySpace(
        (trial.capacity,),
        dtype=transferred_basis.dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "operator-subspace-coarse-space",
                "preparation": preparation_id,
            }
        ),
    )
    prolongation = DenseLinearOperator(
        transferred_basis,
        source=coarse_space,
        target=solver_space.vector_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "operator-subspace-prolongation",
                "preparation": preparation_id,
            }
        ),
    )
    restriction = AdjointLinearOperator(prolongation)
    term = SubspaceCorrectionTerm(restriction, prolongation, local_solver)
    return OperatorSubspaceCorrection(
        trial,
        subspace,
        term,
        source_field_space_id=model_space.field_space_id,
        target_field_space_id=solver_space.field_space_id,
        transfer_id=transfer.transfer_id,
        preparation_id=preparation_id,
    )


class OperatorCorrectionCost(StrictModule, NonTrainableState):
    """Declared architecture-neutral workspace for one operator correction."""

    preparation_workspace_bytes: int = eqx.field(static=True)
    inference_workspace_bytes_per_rhs: int = eqx.field(static=True)
    cost_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        preparation_workspace_bytes: int,
        inference_workspace_bytes_per_rhs: int,
    ):
        preparation = int(preparation_workspace_bytes)
        inference = int(inference_workspace_bytes_per_rhs)
        if preparation < 0 or inference < 0:
            raise ValueError("Operator correction workspace counts must be non-negative.")
        self.preparation_workspace_bytes = preparation
        self.inference_workspace_bytes_per_rhs = inference
        self.cost_id = canonical_fingerprint(
            {
                "kind": "operator-correction-cost",
                "preparation_workspace_bytes": preparation,
                "inference_workspace_bytes_per_rhs": inference,
            }
        )


class OperatorCorrectionBinding(StrictModule, NonTrainableState):
    """Task-bound physical residual and correction representation contract."""

    trained_operator: TrainedOperator
    template: OperatorBatch
    solver_space: DiscreteFieldSpace
    model_residual_space: DiscreteFieldSpace
    model_correction_space: DiscreteFieldSpace
    residual_transfer: FieldTransfer
    correction_transfer: FieldTransfer
    residual_source_name: str = eqx.field(static=True)
    correction_field_name: str = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        trained_operator: TrainedOperator,
        template: OperatorBatch,
        solver_space: DiscreteFieldSpace,
        model_residual_space: DiscreteFieldSpace,
        model_correction_space: DiscreteFieldSpace,
        residual_transfer: FieldTransfer,
        correction_transfer: FieldTransfer,
        /,
        *,
        residual_source_name: str,
        correction_field_name: str,
        condition_ids: Sequence[str],
    ):
        if not isinstance(trained_operator, TrainedOperator):
            raise TypeError("trained_operator must be a TrainedOperator.")
        if not trained_operator.artifact_id:
            raise ValueError("Operator correction requires a trained artifact identity.")
        if trained_operator.sharding_policy is not None:
            raise ValueError(
                "Operator correction does not yet support sharded inference."
            )
        if not isinstance(template, OperatorBatch):
            raise TypeError("template must be an OperatorBatch.")
        if template.case_shape:
            raise ValueError("Operator correction templates must have no case axes.")
        spaces = (solver_space, model_residual_space, model_correction_space)
        if any(not isinstance(space, DiscreteFieldSpace) for space in spaces):
            raise TypeError(
                "Operator correction spaces must be DiscreteFieldSpace values."
            )
        if not isinstance(
            model_residual_space.vector_space, ArraySpace
        ) or not isinstance(model_correction_space.vector_space, ArraySpace):
            raise TypeError("Model residual and correction spaces must use ArraySpace.")
        if not isinstance(residual_transfer, FieldTransfer) or not isinstance(
            correction_transfer, FieldTransfer
        ):
            raise TypeError("Operator correction transfers must be FieldTransfer values.")
        if residual_transfer.source.field_space_id != solver_space.field_space_id or (
            residual_transfer.target.field_space_id != model_residual_space.field_space_id
        ):
            raise ValueError(
                "Residual transfer must map solver_space to model_residual_space."
            )
        if (
            correction_transfer.source.field_space_id
            != model_correction_space.field_space_id
            or correction_transfer.target.field_space_id != solver_space.field_space_id
        ):
            raise ValueError(
                "Correction transfer must map model_correction_space to solver_space."
            )

        source_name = str(residual_source_name)
        field_name = str(correction_field_name)
        if not source_name or not field_name:
            raise ValueError("Correction source and field names must be non-empty.")
        source_fields = tuple(
            field
            for field in trained_operator.task.source_fields
            if field.source_name == source_name
        )
        if len(source_fields) != 1 or source_name not in template.inputs:
            raise ValueError("Residual source must name exactly one bound task input.")
        residual_samples = template.input(source_name)
        if residual_samples.values is None:
            raise ValueError("Residual source template requires values.")
        if residual_samples.support_id != model_residual_space.support_id:
            raise ValueError("Residual source support must match model_residual_space.")
        model_residual_space.vector_space.validate(residual_samples.values)

        if field_name not in trained_operator.task.field_by_name:
            raise KeyError(f"Unknown correction task field {field_name!r}.")
        correction_field = trained_operator.task.field_by_name[field_name]
        if not correction_field.is_target or correction_field.is_classification:
            raise ValueError("Correction field must be a regression target field.")
        assert correction_field.query_name is not None
        assert correction_field.output_spec is not None
        query = template.query(correction_field.query_name)
        if query.support_id != model_correction_space.support_id:
            raise ValueError(
                "Correction query support must match model_correction_space."
            )
        expected_output_shape = (
            query.sample_shape + correction_field.output_spec.channel_shape
        )
        if expected_output_shape != model_correction_space.vector_space.shape:
            raise ValueError(
                "Correction query and output channels must match the model correction space."
            )

        conditions = tuple(sorted(str(value) for value in condition_ids))
        if not conditions or any(not value for value in conditions):
            raise ValueError("Operator correction condition IDs must be non-empty.")
        if len(set(conditions)) != len(conditions):
            raise ValueError("Operator correction condition IDs must be unique.")
        template_payload = {
            "inputs": {
                name: {
                    "geometry": samples.geometry_fingerprint(),
                    "values": (
                        None
                        if name == source_name or samples.values is None
                        else array_tree_fingerprint(samples.values)
                    ),
                }
                for name, samples in template.inputs.items()
            },
            "queries": {
                name: samples.geometry_fingerprint()
                for name, samples in template.queries.items()
            },
        }
        self.trained_operator = trained_operator
        self.template = template
        self.solver_space = solver_space
        self.model_residual_space = model_residual_space
        self.model_correction_space = model_correction_space
        self.residual_transfer = residual_transfer
        self.correction_transfer = correction_transfer
        self.residual_source_name = source_name
        self.correction_field_name = field_name
        self.condition_ids = conditions
        self.binding_id = canonical_fingerprint(
            {
                "kind": "operator-correction-binding",
                "artifact": trained_operator.artifact_id,
                "execution_plan": trained_operator.execution_plan.fingerprint,
                "solver_space": solver_space.field_space_id,
                "model_residual_space": model_residual_space.field_space_id,
                "model_correction_space": model_correction_space.field_space_id,
                "residual_transfer": residual_transfer.transfer_id,
                "correction_transfer": correction_transfer.transfer_id,
                "residual_source": source_name,
                "correction_field": field_name,
                "conditions": list(conditions),
                "template": template_payload,
                "evaluation_key": None,
            }
        )

    def prepare(self, /) -> PreparedOperatorCorrection:
        return PreparedOperatorCorrection(
            self,
            self.trained_operator.prepare(self.template),
        )


class PreparedOperatorCorrection(StrictModule, NonTrainableState):
    """Prepared deterministic residual-to-correction operator application."""

    binding: OperatorCorrectionBinding
    prepared_input: PreparedOperatorInput
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: OperatorCorrectionBinding,
        prepared_input: PreparedOperatorInput,
        /,
    ):
        if not isinstance(binding, OperatorCorrectionBinding):
            raise TypeError("binding must be an OperatorCorrectionBinding.")
        if not isinstance(prepared_input, PreparedOperatorInput):
            raise TypeError("prepared_input must be a PreparedOperatorInput.")
        if (
            prepared_input.plan_fingerprint
            != binding.trained_operator.execution_plan.fingerprint
        ):
            raise ValueError(
                "Prepared operator input belongs to a different binding plan."
            )
        self.binding = binding
        self.prepared_input = prepared_input
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-operator-correction",
                "binding": binding.binding_id,
            }
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[jax.Array]:
        binding = self.binding
        checked = binding.solver_space.vector_space.validate(residual)
        encoded = binding.residual_transfer.primal_operator.mv(checked)
        encoded = binding.model_residual_space.vector_space.validate(encoded)
        prepared = binding.trained_operator.execution_plan.replace_prepared_source(
            self.prepared_input,
            binding.residual_source_name,
            encoded,
        )
        prediction = binding.trained_operator.predict_prepared(prepared, key=None)
        field = prediction.field(binding.correction_field_name)
        query = prediction.query_geometry(field.query_name)
        values = jnp.asarray(field.values)
        mask = query.mask_array(case_shape=prediction.case_shape)
        trailing = (1,) * (values.ndim - mask.ndim)
        values = jnp.where(
            mask.reshape(mask.shape + trailing),
            values,
            jnp.zeros((), dtype=values.dtype),
        )
        lowered = binding.model_correction_space.vector_space.validate(values)
        correction = binding.correction_transfer.primal_operator.mv(lowered)
        correction = binding.solver_space.vector_space.validate(correction)
        coordinates = binding.solver_space.vector_space.flatten(correction)
        coordinates = eqx.error_if(
            coordinates,
            jnp.any(~jnp.isfinite(coordinates)),
            "Operator correction produced non-finite solver coordinates.",
        )
        return binding.solver_space.vector_space.unflatten(coordinates)


class TrainedOperatorPreconditioner(AbstractPreconditioner):
    """Prepared deterministic nonlinear approximate inverse for FGMRES."""

    prepared: PreparedOperatorCorrection

    def __init__(
        self,
        prepared: PreparedOperatorCorrection,
        setup_operator: AbstractLinearOperator,
        /,
    ):
        if not isinstance(prepared, PreparedOperatorCorrection):
            raise TypeError("prepared must be a PreparedOperatorCorrection.")
        _validate_setup_operator(setup_operator, prepared.binding.solver_space)
        self.prepared = prepared
        self.space = prepared.binding.solver_space.vector_space
        self.properties = PreconditionerProperties(
            stationary=True,
            evidence={"stationary": "construction"},
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "trained-operator-preconditioner",
                "binding": prepared.binding.binding_id,
                "setup_operator": setup_operator.operator_id,
            }
        )

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[jax.Array]:
        del iteration
        return self.space.validate(self.prepared.apply(residual))


class TrainedOperatorPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare a task-bound operator correction as a conservative preconditioner."""

    binding: OperatorCorrectionBinding
    cost: OperatorCorrectionCost
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: OperatorCorrectionBinding,
        cost: OperatorCorrectionCost,
        /,
    ):
        if not isinstance(binding, OperatorCorrectionBinding):
            raise TypeError("binding must be an OperatorCorrectionBinding.")
        if not isinstance(cost, OperatorCorrectionCost):
            raise TypeError("cost must be an OperatorCorrectionCost.")
        self.binding = binding
        self.cost = cost
        self._builder_id = canonical_fingerprint(
            {
                "kind": "trained-operator-preconditioner-builder",
                "binding": binding.binding_id,
                "cost": cost.cost_id,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "rebuild"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        _validate_setup_operator(setup_operator, self.binding.solver_space)
        return PreconditionerProperties(
            stationary=True,
            evidence={"stationary": "construction"},
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        _validate_setup_operator(setup_operator, self.binding.solver_space)
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=_resident_storage_bytes(self.binding),
            preparation_workspace_bytes=self.cost.preparation_workspace_bytes,
            apply_workspace_bytes_per_rhs=self.cost.inference_workspace_bytes_per_rhs,
            setup_matvec_count=0,
            reason="trained operator storage and declared inference workspace",
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        del materialization
        _validate_setup_operator(setup_operator, self.binding.solver_space)
        return TrainedOperatorPreconditioner(self.binding.prepare(), setup_operator)

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, TrainedOperatorPreconditioner):
            raise TypeError(
                "Trained operator refresh requires a TrainedOperatorPreconditioner."
            )
        if preconditioner.prepared.binding.binding_id != self.binding.binding_id:
            raise ValueError("Trained operator refresh cannot change its binding.")
        return self.prepare(setup_operator, materialization=materialization)


__all__ = [
    "OperatorCorrectionBinding",
    "OperatorCorrectionCost",
    "OperatorSubspaceCorrection",
    "PreparedOperatorCorrection",
    "TrainedOperatorPreconditioner",
    "TrainedOperatorPreconditionerBuilder",
    "prepare_operator_subspace_correction",
]

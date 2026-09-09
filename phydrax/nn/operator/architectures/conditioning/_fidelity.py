#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from phydrax.fidelity import FidelityPath
from phydrax.nn._keys import EvalKey, split_eval_key
from phydrax.nn.operator.capabilities import ConfiguredOperatorContract
from phydrax.nn.operator.data import OperatorBatch, OperatorOutputSpec
from phydrax.nn.operator.engine import AbstractOperatorModel


def _fidelity_correction_contract(model):
    wrapped = model.correction_operator.operator_contract
    return ConfiguredOperatorContract(
        architecture="FidelityCorrectionOperator",
        configuration=wrapped.configuration
        + (
            (
                "baseline_architecture",
                model.baseline_operator.operator_contract.architecture,
            ),
            ("correction_architecture", wrapped.architecture),
            ("fidelity_path", model.path.path_id),
            ("source_level", model.source_level_id),
            ("target_level", model.target_level_id),
        ),
        capabilities=wrapped.capabilities,
        training=wrapped.training,
    )


def _output_spec_signature(spec: OperatorOutputSpec, /):
    classification = spec.classification
    return (
        spec.channels,
        spec.component_names,
        None if classification is None else classification.to_dict(),
    )


class FidelityCorrectionOperator(AbstractOperatorModel):
    """Compose a low-fidelity operator with a learned target-level correction."""

    operator_architecture = "FidelityCorrectionOperator"
    _operator_contract_builder = staticmethod(_fidelity_correction_contract)

    baseline_operator: AbstractOperatorModel
    correction_operator: AbstractOperatorModel
    path: FidelityPath
    source_level_id: str
    target_level_id: str
    in_size: int | tuple[int, ...] | str
    out_size: int | tuple[int, ...] | str

    def __init__(
        self,
        baseline_operator: AbstractOperatorModel,
        correction_operator: AbstractOperatorModel,
        path: FidelityPath,
        /,
        *,
        source_level_id: str | None = None,
        target_level_id: str | None = None,
    ):
        if not isinstance(baseline_operator, AbstractOperatorModel) or not isinstance(
            correction_operator, AbstractOperatorModel
        ):
            raise TypeError(
                "baseline_operator and correction_operator must be operator models."
            )
        if not isinstance(path, FidelityPath):
            raise TypeError("path must be a FidelityPath.")
        source = (
            path.levels[0].level_id if source_level_id is None else str(source_level_id)
        )
        target = path.target.level_id if target_level_id is None else str(target_level_id)
        if not source or not target or source == target:
            raise ValueError(
                "Source and target fidelity levels must be distinct and non-empty."
            )
        if source not in path.level_ids or target not in path.level_ids:
            raise ValueError("Source and target levels must lie on the fidelity path.")
        if path.level_ids.index(source) >= path.level_ids.index(target):
            raise ValueError("Source fidelity must precede target fidelity on the path.")
        if baseline_operator.in_size != correction_operator.in_size:
            raise ValueError("Baseline and correction operators must share input size.")
        if baseline_operator.out_size != correction_operator.out_size:
            raise ValueError("Baseline and correction operators must share output size.")
        baseline_specs = baseline_operator.operator_output_specs
        correction_specs = correction_operator.operator_output_specs
        if tuple(baseline_specs) != ("output",) or tuple(correction_specs) != ("output",):
            raise ValueError(
                "FidelityCorrectionOperator currently requires one named 'output' field."
            )
        if _output_spec_signature(baseline_specs["output"]) != _output_spec_signature(
            correction_specs["output"]
        ):
            raise ValueError("Baseline and correction output contracts must match.")
        self.baseline_operator = baseline_operator
        self.correction_operator = correction_operator
        self.path = path
        self.source_level_id = source
        self.target_level_id = target
        self.in_size = correction_operator.in_size
        self.out_size = correction_operator.out_size

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        return self.correction_operator.operator_output_specs

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("FidelityCorrectionOperator requires an OperatorBatch.")
        baseline_key, correction_key = split_eval_key(key, 2)
        baseline = jnp.asarray(
            self.baseline_operator.__call_operator_batch__(batch, key=baseline_key)
        )
        correction = jnp.asarray(
            self.correction_operator.__call_operator_batch__(batch, key=correction_key)
        )
        if baseline.shape != correction.shape:
            raise ValueError(
                "Baseline and correction operators must return identical target shapes."
            )
        return baseline + correction

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        return self.__call_operator_batch__(batch, key=key)


__all__ = ["FidelityCorrectionOperator"]

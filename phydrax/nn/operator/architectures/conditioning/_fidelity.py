#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import lcm

import jax.numpy as jnp
from jaxtyping import Array

from phydrax._differentiation import DerivativeRegularity
from phydrax.fidelity import FidelityPath
from phydrax.nn._contracts import model_regularity, sum_regularity
from phydrax.nn._keys import EvalKey, split_eval_key
from phydrax.nn.operator.capabilities import (
    ConfiguredOperatorContract,
    OperatorCapabilitySpec,
    OperatorTrainingRequirement,
)
from phydrax.nn.operator.data import OperatorBatch, OperatorOutputSpec
from phydrax.nn.operator.engine import AbstractOperatorModel


def _ordered_intersection(left, right, /, *, name: str):
    values = tuple(value for value in left if value in frozenset(right))
    if not values:
        raise ValueError(f"Fidelity child operators have no shared {name}.")
    return values


def _optional_intersection(left, right, /):
    if not left:
        return tuple(right)
    if not right:
        return tuple(left)
    return _ordered_intersection(left, right, name="spatial dimension")


def _combined_requirement(left, right, /, *, optional, name: str):
    if left == right:
        return left
    if left == optional:
        return right
    if right == optional:
        return left
    raise ValueError(f"Fidelity child operators have incompatible {name} requirements.")


def _intersect_capabilities(
    baseline: OperatorCapabilitySpec,
    correction: OperatorCapabilitySpec,
    /,
) -> OperatorCapabilitySpec:
    if baseline.global_condition_sources != correction.global_condition_sources:
        raise ValueError(
            "Fidelity child operators must declare identical global condition sources."
        )
    axis_requirement = _combined_requirement(
        baseline.axis_requirement,
        correction.axis_requirement,
        optional="none",
        name="axis",
    )
    topology = _combined_requirement(
        baseline.topology,
        correction.topology,
        optional="optional",
        name="topology",
    )
    cochains = _combined_requirement(
        baseline.cochains,
        correction.cochains,
        optional="optional",
        name="cochain",
    )
    maximum_candidates = tuple(
        value
        for value in (baseline.maximum_sources, correction.maximum_sources)
        if value is not None
    )
    maximum_sources = min(maximum_candidates) if maximum_candidates else None
    minimum_sources = max(baseline.minimum_sources, correction.minimum_sources)
    if maximum_sources is not None and maximum_sources < minimum_sources:
        raise ValueError(
            "Fidelity child operators have incompatible source-count bounds."
        )
    quadrature_order = {"unused": 0, "optional": 1, "physical_required": 2}
    mask_order = {"supported": 0, "all_valid_only": 1, "unsupported": 2}
    minimum_axis_candidates = tuple(
        value
        for value in (baseline.minimum_axis_size, correction.minimum_axis_size)
        if value is not None
    )
    divisor_candidates = tuple(
        value
        for value in (baseline.axis_size_divisor, correction.axis_size_divisor)
        if value is not None
    )
    cochain_sides = _ordered_intersection(
        baseline.cochain_sides,
        correction.cochain_sides,
        name="cochain side",
    )
    return OperatorCapabilitySpec(
        source_geometries=_ordered_intersection(
            baseline.source_geometries,
            correction.source_geometries,
            name="source geometry",
        ),
        query_geometries=_ordered_intersection(
            baseline.query_geometries,
            correction.query_geometries,
            name="query geometry",
        ),
        spatial_dimensions=_optional_intersection(
            baseline.spatial_dimensions,
            correction.spatial_dimensions,
        ),
        source_query_relations=_ordered_intersection(
            baseline.source_query_relations,
            correction.source_query_relations,
            name="source-query relation",
        ),
        requires_fixed_query=(
            baseline.requires_fixed_query or correction.requires_fixed_query
        ),
        axis_requirement=axis_requirement,
        quadrature=max(
            (baseline.quadrature, correction.quadrature),
            key=quadrature_order.__getitem__,
        ),
        masks=max(
            (baseline.masks, correction.masks),
            key=mask_order.__getitem__,
        ),
        topology=topology,
        cochains=cochains,
        cochain_sides=cochain_sides,
        input_representations=_ordered_intersection(
            baseline.input_representations,
            correction.input_representations,
            name="input representation",
        ),
        output_representations=_ordered_intersection(
            baseline.output_representations,
            correction.output_representations,
            name="output representation",
        ),
        symmetry_groups=tuple(
            value
            for value in baseline.symmetry_groups
            if value in frozenset(correction.symmetry_groups)
        ),
        global_condition_sources=baseline.global_condition_sources,
        minimum_sources=minimum_sources,
        maximum_sources=maximum_sources,
        minimum_axis_size=(
            max(minimum_axis_candidates) if minimum_axis_candidates else None
        ),
        axis_size_divisor=(lcm(*divisor_candidates) if divisor_candidates else None),
        resolution_transfer=(
            baseline.resolution_transfer and correction.resolution_transfer
        ),
        encode_once_decode_many=(
            baseline.encode_once_decode_many and correction.encode_once_decode_many
        ),
        multiple_queries=baseline.multiple_queries and correction.multiple_queries,
        autoregressive_rollout=(
            baseline.autoregressive_rollout and correction.autoregressive_rollout
        ),
        requires_structured_tensors=(
            baseline.requires_structured_tensors or correction.requires_structured_tensors
        ),
    )


def _intersect_training(
    baseline: OperatorTrainingRequirement,
    correction: OperatorTrainingRequirement,
    /,
) -> OperatorTrainingRequirement:
    if baseline.regime != correction.regime:
        raise ValueError(
            "Fidelity child operators must share one representable training requirement."
        )

    def combine_text(left: str, right: str, /) -> str:
        if left == right or not right:
            return left
        if not left:
            return right
        return f"{left}; {right}"

    return OperatorTrainingRequirement(
        regime=baseline.regime,
        pretrained_weights_required=(
            baseline.pretrained_weights_required or correction.pretrained_weights_required
        ),
        corpus_description=combine_text(
            baseline.corpus_description,
            correction.corpus_description,
        ),
        claim_scope=combine_text(
            baseline.claim_scope,
            correction.claim_scope,
        ),
    )


def _fidelity_correction_contract(model):
    baseline = model.baseline_operator.operator_contract
    correction = model.correction_operator.operator_contract
    return ConfiguredOperatorContract(
        architecture="FidelityCorrectionOperator",
        configuration=correction.configuration
        + (
            ("baseline_architecture", baseline.architecture),
            ("baseline_configuration", baseline.configuration),
            ("correction_architecture", correction.architecture),
            ("fidelity_path", model.path.path_id),
            ("source_level", model.source_level_id),
            ("target_level", model.target_level_id),
        ),
        capabilities=_intersect_capabilities(
            baseline.capabilities,
            correction.capabilities,
        ),
        training=_intersect_training(baseline.training, correction.training),
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
        baseline_contract = baseline_operator.operator_contract
        correction_contract = correction_operator.operator_contract
        _intersect_capabilities(
            baseline_contract.capabilities,
            correction_contract.capabilities,
        )
        _intersect_training(
            baseline_contract.training,
            correction_contract.training,
        )
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

    def _value_regularity(self) -> DerivativeRegularity | None:
        return sum_regularity(
            (
                model_regularity(self.baseline_operator),
                model_regularity(self.correction_operator),
            )
        )


__all__ = ["FidelityCorrectionOperator"]

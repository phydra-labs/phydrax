#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

from phydrax._trainable import partition_trainable
from phydrax.domain import ConcatenatedModelEvaluator, DomainFunction

from .._model import KFACAffineBlock, KFACLayoutProvider
from ..optim._kfac._blocks import initialize_block_state
from ..optim._kfac._config import KFAC
from ..optim._kfac._types import (
    AffineBlockSpec,
    KFACState,
    ParameterLayout,
    UncoveredBlockSpec,
)


@dataclass(frozen=True, slots=True)
class KFACPlan:
    """Solver-owned execution plan for one KFAC parameter layout."""

    config: KFAC
    layout: ParameterLayout
    num_terms: int
    dtype: object

    def initialize(self, parameters: PyTree[Any], /) -> KFACState:
        flat, _ = ravel_pytree(parameters)
        if int(flat.size) != self.layout.parameter_count:
            raise ValueError("KFAC parameter structure changed after plan construction.")
        return KFACState(
            step=jnp.asarray(0, dtype=jnp.int32),
            curvature=initialize_block_state(
                self.layout,
                num_terms=self.num_terms,
                dtype=self.dtype,
            ),
            factor_updates=jnp.asarray(0, dtype=jnp.int32),
        )


def build_kfac_plan(
    config: KFAC,
    functions: Mapping[str, DomainFunction],
    parameters: PyTree[Any],
    /,
    *,
    num_terms: int,
) -> KFACPlan:
    """Build a static KFAC plan from solver-owned model semantics."""

    if int(num_terms) <= 0:
        raise ValueError("KFAC requires at least one ResidualPenalty term.")
    flat, _ = ravel_pytree(parameters)
    layout = discover_parameter_layout(
        functions,
        parameters,
        exact_block_max_size=config.exact_block_max_size,
        uncovered=config.uncovered,
    )
    return KFACPlan(
        config=config,
        layout=layout,
        num_terms=int(num_terms),
        dtype=flat.dtype,
    )


def _validate_affine_block(block: KFACAffineBlock, /, *, name: str) -> None:
    if block.parameterization == "rwf":
        raise ValueError(
            f"KFAC requires direct affine parameters; disable rwf for {name}."
        )
    if block.parameterization != "direct":
        raise ValueError(
            "KFAC supports only direct affine parameters; "
            f"{name} uses {block.parameterization!r} parameterization."
        )
    if jnp.iscomplexobj(block.weight) or (
        block.bias is not None and jnp.iscomplexobj(block.bias)
    ):
        raise ValueError(f"KFAC requires real affine parameters; {name} is complex.")


def validate_model_coverage(
    functions: Mapping[str, DomainFunction],
    /,
) -> tuple[tuple[str, tuple[KFACAffineBlock, ...]], ...]:
    """Validate and return explicitly declared affine blocks covered by KFAC."""

    layouts: list[tuple[str, tuple[KFACAffineBlock, ...]]] = []
    seen_parameter_ids: set[int] = set()
    for field_name, function in functions.items():
        evaluator = function.func
        trainable_function, _ = partition_trainable(function)
        trainable_leaves = tuple(jax.tree_util.tree_leaves(trainable_function))
        if any(jnp.iscomplexobj(leaf) for leaf in trainable_leaves):
            raise ValueError(
                f"KFAC requires real trainable parameters; field {field_name!r} "
                "contains complex state."
            )
        if not isinstance(evaluator, ConcatenatedModelEvaluator):
            if trainable_leaves and any(
                jnp.asarray(leaf).ndim != 0 for leaf in trainable_leaves
            ):
                raise ValueError(
                    f"KFAC field {field_name!r} has unsupported non-scalar "
                    "trainable state outside a KFAC layout provider."
                )
            continue
        binding = evaluator.binding
        if binding.input_mode != "flat" or binding.batch_mode != "pointwise":
            raise ValueError(
                f"KFAC field {field_name!r} requires input_mode='flat' and "
                "batch_mode='pointwise'."
            )
        model = evaluator.raw_model
        if not isinstance(model, KFACLayoutProvider):
            if trainable_leaves:
                raise ValueError(
                    f"KFAC field {field_name!r} uses unsupported model type "
                    f"{type(model).__name__}; expected a KFACLayoutProvider."
                )
            continue
        validation_errors = model.kfac_validation_errors()
        if validation_errors:
            raise ValueError(
                f"KFAC rejects {validation_errors[0]} in field {field_name!r}."
            )
        affine_blocks = model.kfac_affine_blocks()
        if not affine_blocks:
            raise ValueError(
                f"KFAC field {field_name!r} declared no affine parameter blocks."
            )
        named_blocks = tuple(
            (f"{field_name}/{block.name}", block) for block in affine_blocks
        )
        for block_name, block in named_blocks:
            _validate_affine_block(block, name=block_name)
            parameter_arrays = (block.weight,) + (
                () if block.bias is None else (block.bias,)
            )
            for parameter in parameter_arrays:
                parameter_id = id(parameter)
                if parameter_id in seen_parameter_ids:
                    raise ValueError(
                        "KFAC does not support shared or reused affine parameters; "
                        f"duplicate ownership detected at {block_name}."
                    )
                seen_parameter_ids.add(parameter_id)
        layouts.append((field_name, affine_blocks))
    if not layouts:
        raise ValueError("KFAC found no trainable KFACLayoutProvider fields.")
    return tuple(layouts)


def _flat_leaf_slices(params: PyTree[Any], /) -> tuple[dict[int, tuple[int, ...]], int]:
    leaves = jax.tree_util.tree_leaves(params)
    offset = 0
    slices: dict[int, tuple[int, ...]] = {}
    for leaf in leaves:
        array = jnp.asarray(leaf)
        size = int(array.size)
        slices[id(leaf)] = tuple(range(offset, offset + size))
        offset += size
    return slices, offset


def discover_parameter_layout(
    functions: Mapping[str, DomainFunction],
    params: PyTree[Any],
    /,
    *,
    exact_block_max_size: int,
    uncovered: Literal["error", "diagonal"],
) -> ParameterLayout:
    """Map every covered affine block and classify all remaining trainable leaves."""

    if int(exact_block_max_size) < 0:
        raise ValueError("exact_block_max_size must be nonnegative.")
    if uncovered not in ("error", "diagonal"):
        raise ValueError("uncovered must be either 'error' or 'diagonal'.")

    layouts = validate_model_coverage(functions)
    leaf_slices, parameter_count = _flat_leaf_slices(params)
    covered: set[int] = set()
    blocks: list[AffineBlockSpec] = []
    for field_name, affine_blocks in layouts:
        for block in affine_blocks:
            block_name = f"{field_name}/{block.name}"
            weight_indices = leaf_slices[id(block.weight)]
            bias_indices = () if block.bias is None else leaf_slices[id(block.bias)]
            out_size, in_size = (int(value) for value in block.weight.shape)
            augmented_indices: list[int] = []
            for output_index in range(out_size):
                row_start = output_index * in_size
                augmented_indices.extend(weight_indices[row_start : row_start + in_size])
                if block.bias is not None:
                    augmented_indices.append(bias_indices[output_index])
            block_indices = tuple(augmented_indices)
            overlap = covered.intersection(block_indices)
            if overlap:
                raise ValueError(
                    f"KFAC affine block {block_name} overlaps another parameter block."
                )
            covered.update(block_indices)
            blocks.append(
                AffineBlockSpec(
                    name=block_name,
                    indices=block_indices,
                    output_size=out_size,
                    input_size=in_size + int(block.bias is not None),
                    has_bias=block.bias is not None,
                )
            )

    remaining = tuple(index for index in range(parameter_count) if index not in covered)
    uncovered_block: UncoveredBlockSpec | None = None
    if remaining:
        if len(remaining) <= int(exact_block_max_size):
            approximation: Literal["exact", "diagonal"] = "exact"
        elif uncovered == "diagonal":
            approximation = "diagonal"
        else:
            raise ValueError(
                "KFAC found unsupported trainable parameters outside declared affine blocks "
                f"({len(remaining)} scalars); set uncovered='diagonal' or increase "
                "exact_block_max_size."
            )
        uncovered_block = UncoveredBlockSpec(
            name="uncovered",
            indices=remaining,
            approximation=approximation,
        )
    return ParameterLayout(tuple(blocks), uncovered_block, parameter_count)


__all__ = [
    "AffineBlockSpec",
    "KFACPlan",
    "ParameterLayout",
    "UncoveredBlockSpec",
    "build_kfac_plan",
    "discover_parameter_layout",
    "validate_model_coverage",
]

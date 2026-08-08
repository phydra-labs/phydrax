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

from ..nn.models.architectures._mlp import MLP
from ..nn.models.layers._linear import Linear
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


def _validate_linear(layer: Linear, /, *, name: str) -> None:
    if layer.random_weight_factorization:
        raise ValueError(
            f"KFAC does not support random weight factorization; disable rwf for {name}."
        )
    if layer.enforce_positive_weights:
        raise ValueError(
            f"KFAC does not support positive-weight reparameterization in {name}."
        )
    if jnp.iscomplexobj(layer.weight) or (
        layer.bias is not None and jnp.iscomplexobj(layer.bias)
    ):
        raise ValueError(f"KFAC requires real affine parameters; {name} is complex.")


def validate_model_coverage(
    functions: Mapping[str, DomainFunction],
    /,
) -> tuple[tuple[str, MLP], ...]:
    """Validate and return the raw pointwise flat MLPs covered by KFAC."""

    models: list[tuple[str, MLP]] = []
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
                    "trainable state outside a Phydrax MLP."
                )
            continue
        binding = evaluator.binding
        if binding.input_mode != "flat" or binding.batch_mode != "pointwise":
            raise ValueError(
                f"KFAC field {field_name!r} requires input_mode='flat' and "
                "batch_mode='pointwise'."
            )
        model = evaluator.raw_model
        if not isinstance(model, MLP):
            if trainable_leaves:
                raise ValueError(
                    f"KFAC field {field_name!r} uses unsupported model type "
                    f"{type(model).__name__}; expected phydrax.nn.MLP."
                )
            continue
        for site, dropout in enumerate(model.dropouts):
            if dropout.p > 0.0 and not dropout.inference:
                raise ValueError(
                    f"KFAC rejects active dropout in field {field_name!r} at site {site}."
                )
        named_layers: list[tuple[str, Linear]] = [
            (f"{field_name}/layers/{index}", layer)
            for index, layer in enumerate(model.layers)
        ]
        if model._residual_proj is not None:
            named_layers.append(
                (f"{field_name}/residual_projection", model._residual_proj)
            )
        for layer_name, layer in named_layers:
            _validate_linear(layer, name=layer_name)
            parameter_arrays = (layer.weight,) + (
                () if layer.bias is None else (layer.bias,)
            )
            for parameter in parameter_arrays:
                parameter_id = id(parameter)
                if parameter_id in seen_parameter_ids:
                    raise ValueError(
                        "KFAC does not support shared or reused affine parameters; "
                        f"duplicate ownership detected at {layer_name}."
                    )
                seen_parameter_ids.add(parameter_id)
        models.append((field_name, model))
    if not models:
        raise ValueError("KFAC found no trainable phydrax.nn.MLP fields.")
    return tuple(models)


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

    models = validate_model_coverage(functions)
    leaf_slices, parameter_count = _flat_leaf_slices(params)
    covered: set[int] = set()
    blocks: list[AffineBlockSpec] = []
    for field_name, model in models:
        named_layers: list[tuple[str, Linear]] = [
            (f"{field_name}/layers/{index}", layer)
            for index, layer in enumerate(model.layers)
        ]
        if model._residual_proj is not None:
            named_layers.append(
                (f"{field_name}/residual_projection", model._residual_proj)
            )
        for layer_name, layer in named_layers:
            weight_indices = leaf_slices[id(layer.weight)]
            bias_indices = () if layer.bias is None else leaf_slices[id(layer.bias)]
            out_size, in_size = (int(value) for value in layer.weight.shape)
            augmented_indices: list[int] = []
            for output_index in range(out_size):
                row_start = output_index * in_size
                augmented_indices.extend(weight_indices[row_start : row_start + in_size])
                if layer.bias is not None:
                    augmented_indices.append(bias_indices[output_index])
            block_indices = tuple(augmented_indices)
            overlap = covered.intersection(block_indices)
            if overlap:
                raise ValueError(
                    f"KFAC affine block {layer_name} overlaps another parameter block."
                )
            covered.update(block_indices)
            blocks.append(
                AffineBlockSpec(
                    name=layer_name,
                    indices=block_indices,
                    output_size=out_size,
                    input_size=in_size + int(layer.bias is not None),
                    has_bias=layer.bias is not None,
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
                "KFAC found unsupported trainable parameters outside affine MLP blocks "
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

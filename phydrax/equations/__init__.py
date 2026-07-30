#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Serializable, validated equation representations for physics-aware models."""

from ._compile import (
    compile_pde_expression,
    compile_pde_functional_constraint,
    compile_pde_problem,
    CompiledPDECondition,
    CompiledPDEEquation,
    CompiledPDEProblem,
    DifferentialBackend,
    IntegralCompiler,
    make_pde_operator,
)
from ._ir import (
    as_expression,
    PDECondition,
    PDEConditionKind,
    PDECoordinate,
    PDECoordinateKind,
    PDEEquation,
    PDEExpression,
    PDEExpressionOp,
    PDEField,
    PDEParameter,
    PDEProblemIR,
    PDERegion,
    PDERegionKind,
    PDERepresentation,
)
from ._serialize import (
    pde_ir_from_dict,
    pde_ir_from_json,
    pde_ir_hash,
    pde_ir_to_dict,
    pde_ir_to_json,
)
from ._tokens import (
    pad_pde_tokens,
    PDE_OPERATOR_VOCABULARY,
    PDE_TOKEN_ATTRIBUTES,
    PDE_TOKEN_KINDS,
    PDETokenBatch,
    stack_pde_tokens,
    tokenize_pde_ir,
)
from ._validate import infer_expression_type, PDEValueType, validate_pde_ir


__all__ = [
    "CompiledPDECondition",
    "CompiledPDEEquation",
    "CompiledPDEProblem",
    "DifferentialBackend",
    "IntegralCompiler",
    "PDECondition",
    "PDEConditionKind",
    "PDECoordinate",
    "PDECoordinateKind",
    "PDEEquation",
    "PDEExpression",
    "PDEExpressionOp",
    "PDEField",
    "PDEParameter",
    "PDEProblemIR",
    "PDERegion",
    "PDERegionKind",
    "PDERepresentation",
    "PDEValueType",
    "PDE_OPERATOR_VOCABULARY",
    "PDE_TOKEN_ATTRIBUTES",
    "PDE_TOKEN_KINDS",
    "PDETokenBatch",
    "as_expression",
    "compile_pde_expression",
    "compile_pde_functional_constraint",
    "compile_pde_problem",
    "infer_expression_type",
    "make_pde_operator",
    "pad_pde_tokens",
    "pde_ir_from_dict",
    "pde_ir_from_json",
    "pde_ir_hash",
    "pde_ir_to_dict",
    "pde_ir_to_json",
    "stack_pde_tokens",
    "tokenize_pde_ir",
    "validate_pde_ir",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic fixed-shape execution worksets and checkpoints."""

from ._execution_pool import PoolExecutionSignature
from ._execution_workset import (
    evaluate_execution_worksets_serial,
    evaluate_execution_worksets_vmap,
    ExecutionWorksetCheckpoint,
    ExecutionWorksetEvaluation,
    ExecutionWorksetEvidence,
    ExecutionWorksetPlan,
    PreparedExecutionWorksets,
    restore_execution_workset_checkpoint,
)


__all__ = [
    "ExecutionWorksetCheckpoint",
    "ExecutionWorksetEvaluation",
    "ExecutionWorksetEvidence",
    "ExecutionWorksetPlan",
    "PoolExecutionSignature",
    "PreparedExecutionWorksets",
    "evaluate_execution_worksets_serial",
    "evaluate_execution_worksets_vmap",
    "restore_execution_workset_checkpoint",
]

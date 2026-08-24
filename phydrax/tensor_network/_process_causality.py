#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._process_tensor import ProcessTensorMPO


class ProcessCombCausalityReport(StrictModule):
    slot_residuals: Array
    maximum_residual: Array
    finite: Array
    valid: Array

    def __init__(self, slot_residuals: ArrayLike, /, *, tolerance: float):
        self.slot_residuals = jnp.asarray(slot_residuals)
        self.maximum_residual = jnp.max(self.slot_residuals)
        self.finite = jnp.all(jnp.isfinite(self.slot_residuals))
        self.valid = self.finite & (self.maximum_residual <= tolerance)


def validate_process_comb_causality(
    process: ProcessTensorMPO,
    /,
    *,
    tolerance: float = 1e-8,
) -> ProcessCombCausalityReport:
    """Validate local temporal transfer of the trace functional at every slot."""
    dimension = process.dimension
    trace_vector = jnp.eye(dimension).reshape(-1)
    norm = jnp.vdot(trace_vector, trace_vector)
    residuals = []
    for tensor in process.tensors:
        reduced = oe.contract("o,loir->lir", trace_vector, tensor)
        bond_transfer = oe.contract("lir,i->lr", reduced, trace_vector) / norm
        expected = oe.contract("lr,i->lir", bond_transfer, trace_vector)
        residuals.append(jnp.max(jnp.abs(reduced - expected)))
    return ProcessCombCausalityReport(jnp.stack(residuals), tolerance=tolerance)


class ProcessSequenceLikelihood(StrictModule):
    log_likelihood: Array
    probabilities: Array
    normalization_residual: Array
    valid: Array

    def __init__(self, probabilities: ArrayLike, counts: ArrayLike, /):
        values = jnp.asarray(probabilities)
        counts_ = jnp.asarray(counts)
        if values.shape != counts_.shape:
            raise ValueError("Process probabilities and counts must share shape.")
        safe = jnp.maximum(values, jnp.finfo(values.dtype).tiny)
        self.probabilities = values
        self.log_likelihood = jnp.sum(counts_ * jnp.log(safe))
        self.normalization_residual = jnp.abs(jnp.sum(values) - 1.0)
        self.valid = (
            jnp.all(jnp.isfinite(values) & (values >= 0.0))
            & (self.normalization_residual <= 1e-8)
            & jnp.isfinite(self.log_likelihood)
        )


__all__ = [
    "ProcessCombCausalityReport",
    "ProcessSequenceLikelihood",
    "validate_process_comb_causality",
]

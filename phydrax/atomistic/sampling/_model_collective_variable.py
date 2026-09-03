#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractArrayModel
from ._collective_variable import (
    AbstractCollectiveVariableProgram,
    CollectiveVariableMetric,
)


class ModelCollectiveVariableProgram(AbstractCollectiveVariableProgram):
    """Frozen array model composed after one differentiable CV feature program."""

    source: AbstractCollectiveVariableProgram
    model: AbstractArrayModel
    output_size: int = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    metrics: tuple[CollectiveVariableMetric, ...]
    model_id: str = eqx.field(static=True)
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AbstractCollectiveVariableProgram,
        model: AbstractArrayModel,
        /,
        *,
        model_id: str,
        names: tuple[str, ...],
        metrics: tuple[CollectiveVariableMetric, ...] | None = None,
    ):
        if not isinstance(source, AbstractCollectiveVariableProgram):
            raise TypeError("source must implement AbstractCollectiveVariableProgram.")
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must implement AbstractArrayModel.")
        if model.in_size != source.output_size or not isinstance(model.out_size, int):
            raise ValueError("Model sizes do not match the source CV program.")
        identifier = str(model_id).strip()
        resolved_names = tuple(str(name) for name in names)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        if (
            len(resolved_names) != model.out_size
            or any(not name for name in resolved_names)
            or len(set(resolved_names)) != len(resolved_names)
        ):
            raise ValueError("names must be non-empty, unique, and match model output.")
        resolved_metrics = (
            tuple(CollectiveVariableMetric() for _ in range(model.out_size))
            if metrics is None
            else tuple(metrics)
        )
        if len(resolved_metrics) != model.out_size or any(
            not isinstance(metric, CollectiveVariableMetric)
            for metric in resolved_metrics
        ):
            raise TypeError(
                "metrics must contain one CollectiveVariableMetric per output."
            )
        self.source = source
        self.model = model
        self.output_size = model.out_size
        self.names = resolved_names
        self.metrics = resolved_metrics
        self.model_id = identifier
        self.program_id = canonical_fingerprint(
            {
                "kind": "model-collective-variable-program",
                "source": source.program_id,
                "model": identifier,
                "names": list(resolved_names),
                "metrics": [metric.metric_id for metric in resolved_metrics],
            }
        )

    def evaluate(self, positions: ArrayLike, /, **kwargs):
        source, source_valid = self.source.evaluate(positions, **kwargs)
        value = jnp.asarray(self.model(source, key=None)).reshape((self.output_size,))
        valid = source_valid & jnp.all(jnp.isfinite(value))
        return jnp.where(valid, value, 0.0), valid


__all__ = ["ModelCollectiveVariableProgram"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..._identity import IdentityWords
from ..._model import AbstractArrayModel
from ...typing import checked
from .._neighborhood_bundle import AtomisticNeighborhoodRequirement
from ._collective_variable import (
    AbstractCollectiveVariableProgram,
    CollectiveVariableMetric,
    CollectiveVariableProgramEvaluation,
)
from ._cv_geometry import (
    CollectiveVariableGeometry,
    CollectiveVariableStatus,
    status_flag,
)


class ModelCollectiveVariableProgram(AbstractCollectiveVariableProgram):
    """Array model composed after one differentiable CV feature program.

    The source geometry, metric branches, support and neighborhood requests are
    forwarded unchanged; the model's array parameters stay dynamic leaves.
    The caller's ``model_id`` enters only the exact program digest, carried
    with the source system binding as dynamic words (``program_identity``,
    ``system_identity``); no model or content identifier is a static field.
    """

    source: AbstractCollectiveVariableProgram
    model: AbstractArrayModel
    output_size: int = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    metrics: tuple[CollectiveVariableMetric, ...]
    program_identity: IdentityWords
    system_identity: IdentityWords
    requirements: tuple[AtomisticNeighborhoodRequirement, ...]
    cell_dependent: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: AbstractCollectiveVariableProgram,
        model: AbstractArrayModel,
        /,
        *,
        model_id: str,
        names: tuple[str, ...],
        metrics: tuple[CollectiveVariableMetric, ...] | None = None,
    ) -> None:
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
        self.system_identity = source.system_identity
        self.requirements = source.requirements
        self.cell_dependent = source.cell_dependent
        self.program_identity = IdentityWords.of(
            {
                "kind": "model-collective-variable-program",
                "source": source.program_id,
                "model": identifier,
                "names": list(resolved_names),
                "metrics": [metric.metric_id for metric in resolved_metrics],
            }
        )

    @property
    def support(self) -> tuple[int, ...]:
        """Host: the source program's particle support."""
        return self.source.support

    def evaluate(
        self, geometry: CollectiveVariableGeometry, /
    ) -> CollectiveVariableProgramEvaluation:
        source = self.source.evaluate(geometry)
        value = jnp.asarray(self.model(source.values, key=None)).reshape(
            (self.output_size,)
        )
        finite = jnp.all(jnp.isfinite(value))
        successful = source.successful & finite
        status = source.status | status_flag(~finite, CollectiveVariableStatus.NONFINITE)
        return CollectiveVariableProgramEvaluation(
            values=jnp.where(successful, value, 0.0),
            successful=successful,
            regular=successful & source.regular,
            status=status,
            branch_margins=source.branch_margins,
            program_identity=self.program_identity,
        )


__all__ = ["ModelCollectiveVariableProgram"]

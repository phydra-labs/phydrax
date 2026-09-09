#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._core import nonempty_identifier, resolved_identifier
from ._hierarchy import FidelityLevelSpec


if TYPE_CHECKING:
    from ._dataset import FidelityCaseSpec


class FidelityEvaluation(StrictModule, NonTrainableState):
    """One immutable model evaluation in a canonical observable space."""

    observable: PyTree[Array]
    valid: Array
    cost: Array
    result: Any
    case_id: str = eqx.field(static=True)
    pair_id: str = eqx.field(static=True)
    level_id: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)
    cost_unit: str = eqx.field(static=True)
    artifact_id: str | None = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        observable: PyTree[Array],
        /,
        *,
        case_id: str,
        pair_id: str,
        level_id: str,
        evaluator_id: str,
        valid: bool | Array,
        cost: float | Array,
        cost_unit: str,
        result: Any = None,
        artifact_id: str | None = None,
        evidence_ids: tuple[str, ...] = (),
        evaluation_id: str | None = None,
    ):
        leaves = tuple(jnp.asarray(leaf) for leaf in jax_tree_leaves(observable))
        if not leaves:
            raise ValueError("observable must contain at least one array leaf.")
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        cost_ = jnp.asarray(cost)
        if valid_.shape != () or cost_.shape != ():
            raise ValueError("valid and cost must be scalar values.")
        cost_host = float(np.asarray(cost_))
        if not np.isfinite(cost_host) or cost_host < 0.0:
            raise ValueError("cost must be finite and non-negative.")
        if bool(np.asarray(valid_)):
            for leaf in leaves:
                host = np.asarray(leaf)
                if np.issubdtype(host.dtype, np.number) and not np.all(np.isfinite(host)):
                    raise ValueError("A valid fidelity observable must be finite.")
        artifact = (
            None
            if artifact_id is None
            else nonempty_identifier("artifact_id", artifact_id)
        )
        evidence = tuple(str(value) for value in evidence_ids)
        if any(not value for value in evidence) or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids must be unique non-empty strings.")
        self.observable = observable
        self.valid = valid_
        self.cost = cost_
        self.result = result
        self.case_id = nonempty_identifier("case_id", case_id)
        self.pair_id = nonempty_identifier("pair_id", pair_id)
        self.level_id = nonempty_identifier("level_id", level_id)
        self.evaluator_id = nonempty_identifier("evaluator_id", evaluator_id)
        self.cost_unit = nonempty_identifier("cost_unit", cost_unit)
        self.artifact_id = artifact
        self.evidence_ids = evidence
        self.evaluation_id = resolved_identifier(
            "evaluation_id",
            evaluation_id,
            {
                "kind": "fidelity-evaluation",
                "case": self.case_id,
                "pair": self.pair_id,
                "level": self.level_id,
                "evaluator": self.evaluator_id,
                "valid": bool(np.asarray(valid_)),
                "cost": cost_host,
                "cost_unit": self.cost_unit,
                "observable": array_tree_fingerprint(observable),
                "artifact": artifact,
                "evidence": list(evidence),
            },
        )


class FidelityEvaluator(Protocol):
    """Callable contract for one fidelity-level model."""

    level: FidelityLevelSpec
    evaluator_id: str

    def __call__(
        self,
        case: FidelityCaseSpec,
        key: Array | None = None,
        /,
    ) -> FidelityEvaluation: ...


def jax_tree_leaves(tree: Any, /) -> list[Any]:
    """Return array leaves without importing a second tree implementation."""

    import jax

    return jax.tree_util.tree_leaves(tree)


__all__ = ["FidelityEvaluation", "FidelityEvaluator"]

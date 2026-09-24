#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admission of the model components that define a nonlinear residual."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax

from .._differentiation import (
    admit_regularity,
    ComponentAuthority,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from .._model import AbstractArrayModel, ComponentBinding
from .._model._component import admit_randomness
from .._model._realization import FrozenRealization
from ._types import NonlinearSystemProblem


DETERMINISM_UNDECLARED = "determinism-undeclared"
_REGULARITY_UNDECLARED = "regularity-undeclared"
_RANDOMNESS_HINTS = {
    "randomness-undeclared": (
        "declare a RandomnessContract in the model's model_execution_contract()"
    ),
    "inference-state-unbound": (
        "evaluate the model in its inference state (phydrax.nn.layers.inference_mode)"
    ),
    "realization-unbound": (
        "bind one realization with phydrax.FrozenRealization(model, key, "
        "realization_id=...)"
    ),
    "realization-unidentified": "declare the realization_id of the fixed realization",
    "resampled-randomness-not-admitted": (
        "bind one realization with phydrax.FrozenRealization(model, key, "
        "realization_id=...)"
    ),
}


def _is_component(node: Any, /) -> bool:
    return isinstance(node, (AbstractArrayModel, ComponentBinding))


def _components(
    scope: str, tree: Any, /
) -> Iterator[tuple[str, AbstractArrayModel, ComponentAuthority | None]]:
    entries, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_component)
    for path, node in entries:
        location = scope + jax.tree_util.keystr(path)
        if isinstance(node, ComponentBinding):
            yield location, node.model, node.authority
        elif isinstance(node, AbstractArrayModel):
            yield location, node, None


def _is_opaque(residual: Any, /) -> bool:
    leaves = jax.tree_util.tree_leaves(residual)
    return len(leaves) == 1 and leaves[0] is residual


def admit_residual_components(
    problem: NonlinearSystemProblem,
    args: Any,
    /,
    *,
    implicit: bool,
) -> tuple[str, ...]:
    """Admit the components of a residual for a certified or implicit root map.

    Components are the models found in the structured residual callable and in
    `args` (a `ComponentBinding` supplies its bound authority). Each component's
    randomness is admitted for authoritative use, with a `FrozenRealization` as
    the owner's explicit realization binding; on the implicit route each
    component additionally needs classical `C^1` value regularity (or a
    branch-margin condition) under its authority, `MODEL` for bare models. An
    opaque residual closure hides its components and is recorded as
    `"determinism-undeclared"` (and, on the implicit route,
    `"regularity-undeclared"`). Returns the sorted evidence records and raises
    `ValueError` for a rejected component.
    """
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    evidence: set[str] = set()
    residual = problem.residual_function
    if _is_opaque(residual):
        evidence.add(f"residual:{DETERMINISM_UNDECLARED}")
        if implicit:
            evidence.add(f"residual:{_REGULARITY_UNDECLARED}")
    for scope, tree in (("residual", residual), ("args", args)):
        for location, model, authority in _components(scope, tree):
            contract = model.model_execution_contract()
            admitted, reason = admit_randomness(
                contract.randomness,
                implicit=implicit,
                authoritative=True,
                realization_bound=isinstance(model, FrozenRealization),
                inference_state_bound=False,
            )
            if not admitted:
                hint = _RANDOMNESS_HINTS.get(reason)
                raise ValueError(
                    f"Residual component {location} ({type(model).__name__}) is not "
                    f"admitted for a certified root map: {reason}"
                    + ("." if hint is None else f"; {hint}.")
                )
            evidence.add(f"{location}:{reason}")
            if not implicit:
                continue
            owner = ComponentAuthority.MODEL if authority is None else authority
            admission = admit_regularity(
                contract.regularity,
                DifferentiationRequest(
                    (DerivativeSurface.PRIMAL_STATE,), authority=owner
                ),
                route=DerivativeRoute.IMPLICIT,
                policy=RegularityPolicy(),
            )
            if not admission.supported:
                raise ValueError(
                    f"Residual component {location} ({type(model).__name__}) is not "
                    "admitted for implicit root differentiation, which needs "
                    "classical C1 regularity near the root or a branch-margin "
                    f"certificate: {', '.join(admission.reasons)}."
                )
            evidence.update(
                f"{location}:{condition}" for condition in admission.conditions
            )
    return tuple(sorted(evidence))


__all__ = ["admit_residual_components", "DETERMINISM_UNDECLARED"]

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod, sqrt
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array

from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..layers._measure_attention import MeasureAwareAttention
from .context import EncodedOperatorState
from .data import OperatorOutputSpec


BranchRole = Literal["conditioning", "prediction", "both"]
BranchGeometryKind = Literal[
    "abstract",
    "tensor_grid",
    "point_cloud",
    "geometry",
    "surface",
    "volume",
    "boundary",
    "interface",
    "particle",
]


class OperatorBranchSpec(StrictModule):
    """Static semantics for one conditioning or prediction branch."""

    name: str
    role: BranchRole
    geometry_kind: BranchGeometryKind
    source_name: str | None
    query_name: str | None
    output_spec: OperatorOutputSpec | None
    processor_group: str
    decoder_group: str

    def __init__(
        self,
        name: str,
        /,
        *,
        role: BranchRole,
        geometry_kind: BranchGeometryKind,
        source_name: str | None = None,
        query_name: str | None = None,
        output_spec: OperatorOutputSpec | None = None,
        processor_group: str = "default",
        decoder_group: str | None = None,
    ):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("Branch name must not be empty.")
        predicts = role in ("prediction", "both")
        conditions = role in ("conditioning", "both")
        if predicts and output_spec is None:
            raise ValueError("Prediction branches require an output specification.")
        if not predicts and output_spec is not None:
            raise ValueError("Conditioning-only branches cannot define an output spec.")
        resolved_source = (
            resolved_name if conditions and source_name is None else source_name
        )
        resolved_query = resolved_name if predicts and query_name is None else query_name
        if conditions and not resolved_source:
            raise ValueError("Conditioning branches require a source name.")
        if predicts and not resolved_query:
            raise ValueError("Prediction branches require a query name.")
        self.name = resolved_name
        self.role = role
        self.geometry_kind = geometry_kind
        self.source_name = None if resolved_source is None else str(resolved_source)
        self.query_name = None if resolved_query is None else str(resolved_query)
        self.output_spec = output_spec
        self.processor_group = str(processor_group)
        self.decoder_group = (
            resolved_name if decoder_group is None else str(decoder_group)
        )

    @property
    def conditions(self) -> bool:
        return self.role in ("conditioning", "both")

    @property
    def predicts(self) -> bool:
        return self.role in ("prediction", "both")


class BranchInteractionSpec(StrictModule):
    """One directed cross-branch update at a static processor stage."""

    source: str
    target: str
    stage: int
    parameter_group: str
    residual: bool
    scale: float

    def __init__(
        self,
        source: str,
        target: str,
        /,
        *,
        stage: int,
        parameter_group: str = "cross_branch",
        residual: bool = True,
        scale: float = 1.0,
    ):
        if str(source) == str(target):
            raise ValueError("Branch interactions must connect distinct branches.")
        if int(stage) < 0:
            raise ValueError("Branch interaction stage must be non-negative.")
        if not str(parameter_group):
            raise ValueError("Branch interaction parameter group must not be empty.")
        if not jnp.isfinite(float(scale)):
            raise ValueError("Branch interaction scale must be finite.")
        self.source = str(source)
        self.target = str(target)
        self.stage = int(stage)
        self.parameter_group = str(parameter_group)
        self.residual = bool(residual)
        self.scale = float(scale)


class OperatorBranchGraph(StrictModule):
    """Validated static branch schema and directed interaction schedule."""

    branches: frozendict[str, OperatorBranchSpec]
    interactions: tuple[BranchInteractionSpec, ...]

    def __init__(
        self,
        branches: Sequence[OperatorBranchSpec],
        /,
        *,
        interactions: Sequence[BranchInteractionSpec] = (),
    ):
        branch_tuple = tuple(branches)
        if not branch_tuple:
            raise ValueError("OperatorBranchGraph requires at least one branch.")
        names = tuple(branch.name for branch in branch_tuple)
        if len(set(names)) != len(names):
            raise ValueError("Operator branch names must be unique.")
        interaction_tuple = tuple(interactions)
        known = set(names)
        for interaction in interaction_tuple:
            if interaction.source not in known or interaction.target not in known:
                raise ValueError(
                    "Branch interaction endpoints must name declared branches."
                )
        identities = tuple(
            (item.stage, item.source, item.target) for item in interaction_tuple
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate branch interaction at one stage.")
        self.branches = frozendict({branch.name: branch for branch in branch_tuple})
        self.interactions = tuple(
            sorted(
                interaction_tuple,
                key=lambda item: (item.stage, item.source, item.target),
            )
        )

    def branch(self, name: str, /) -> OperatorBranchSpec:
        if name not in self.branches:
            raise KeyError(
                f"Unknown branch {name!r}; expected one of {tuple(self.branches)}."
            )
        return self.branches[name]

    @property
    def prediction_names(self) -> tuple[str, ...]:
        return tuple(name for name, branch in self.branches.items() if branch.predicts)

    @property
    def conditioning_names(self) -> tuple[str, ...]:
        return tuple(name for name, branch in self.branches.items() if branch.conditions)

    def interactions_at(self, stage: int, /) -> tuple[BranchInteractionSpec, ...]:
        return tuple(item for item in self.interactions if item.stage == int(stage))


class BranchedEncodedOperatorState(StrictModule):
    """Named encoded contexts that share one case shape."""

    branches: frozendict[str, EncodedOperatorState]
    case_shape: tuple[int, ...]

    def __init__(self, branches: Mapping[str, EncodedOperatorState], /):
        if not branches:
            raise ValueError("Branched operator state must not be empty.")
        first = next(iter(branches.values()))
        if any(state.case_shape != first.case_shape for state in branches.values()):
            raise ValueError("Every branch state must have the same case shape.")
        self.branches = frozendict(branches)
        self.case_shape = first.case_shape

    def branch(self, name: str, /) -> EncodedOperatorState:
        if name not in self.branches:
            raise KeyError(
                f"Unknown encoded branch {name!r}; expected {tuple(self.branches)}."
            )
        return self.branches[name]

    def replace(
        self,
        replacements: Mapping[str, EncodedOperatorState],
        /,
    ) -> "BranchedEncodedOperatorState":
        unknown = set(replacements) - set(self.branches)
        if unknown:
            raise KeyError(f"Cannot replace unknown encoded branches {sorted(unknown)}.")
        values = dict(self.branches)
        values.update(replacements)
        return BranchedEncodedOperatorState(values)


def _flatten_state(state: EncodedOperatorState, /) -> tuple[Array, Array, Array]:
    cases = prod(state.case_shape) if state.case_shape else 1
    values = state.values.reshape((cases, state.num_tokens, state.channels))
    weights = state.weights.reshape((cases, state.num_tokens))
    mask = state.mask.reshape((cases, state.num_tokens))
    return values, weights, mask


def apply_branch_interactions(
    state: BranchedEncodedOperatorState,
    graph: OperatorBranchGraph,
    attention: Mapping[str, MeasureAwareAttention],
    stage: int,
    /,
) -> BranchedEncodedOperatorState:
    """Apply one stage synchronously so opposite edges see the same old state."""
    scheduled = graph.interactions_at(stage)
    if not scheduled:
        return state
    required_groups = {item.parameter_group for item in scheduled}
    missing = required_groups - set(attention)
    if missing:
        raise KeyError(f"Missing cross-branch attention groups {sorted(missing)}.")
    updates: dict[str, list[tuple[Array, BranchInteractionSpec]]] = {}
    for interaction in scheduled:
        source = state.branch(interaction.source)
        target = state.branch(interaction.target)
        source_values, source_weights, source_mask = _flatten_state(source)
        target_values, _, target_mask = _flatten_state(target)
        module = attention[interaction.parameter_group]
        update = module(
            source_values,
            target_values,
            source_weights,
            source_mask=source_mask,
            query_mask=target_mask,
        )
        if int(update.shape[-1]) != target.channels:
            raise ValueError(
                "Cross-branch attention output channels must match its target branch."
            )
        shaped = update.reshape(target.values.shape) * interaction.scale
        updates.setdefault(interaction.target, []).append((shaped, interaction))

    replacements: dict[str, EncodedOperatorState] = {}
    for name, incoming in updates.items():
        target = state.branch(name)
        combined = incoming[0][0]
        for value, _ in incoming[1:]:
            combined = combined + value
        combined = combined / sqrt(float(len(incoming)))
        residual_flags = {item.residual for _, item in incoming}
        if len(residual_flags) != 1:
            raise ValueError(
                "Incoming interactions at one stage must agree on residual semantics."
            )
        updated = target.values + combined if residual_flags.pop() else combined
        updated = updated * target.mask[..., None]
        replacements[name] = target.replace_layers(target.layer_values + (updated,))
    return state.replace(replacements)


def bidirectional_branch_interactions(
    left: str,
    right: str,
    /,
    *,
    stage: int,
    parameter_group: str = "cross_branch",
    residual: bool = True,
    scale: float = 1.0,
) -> tuple[BranchInteractionSpec, BranchInteractionSpec]:
    """Construct a symmetric pair of directed branch interactions."""
    return (
        BranchInteractionSpec(
            left,
            right,
            stage=stage,
            parameter_group=parameter_group,
            residual=residual,
            scale=scale,
        ),
        BranchInteractionSpec(
            right,
            left,
            stage=stage,
            parameter_group=parameter_group,
            residual=residual,
            scale=scale,
        ),
    )


__all__ = [
    "BranchGeometryKind",
    "BranchInteractionSpec",
    "BranchRole",
    "BranchedEncodedOperatorState",
    "OperatorBranchGraph",
    "OperatorBranchSpec",
    "apply_branch_interactions",
    "bidirectional_branch_interactions",
]

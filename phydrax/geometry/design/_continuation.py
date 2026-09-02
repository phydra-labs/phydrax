#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from .._contracts import CompiledGeometry, GeometrySource, GeometryTolerance
from ..analytic._operations import BlendCSG, SharpCSG
from ._constraints import AbstractDesignConstraint, DesignConstraintSystem
from ._schema import DesignState, ParameterId, ParameterSchema


class CSGContinuationPolicy(StrictModule):
    """Finite positive nonincreasing blend-width epochs and terminal tolerance."""

    widths: tuple[float, ...] = eqx.field(static=True)
    terminal_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        widths: Sequence[float],
        /,
        *,
        terminal_tolerance: float = 1e-8,
    ):
        widths_ = tuple(float(value) for value in widths)
        if not widths_ or any(not isfinite(value) or value <= 0.0 for value in widths_):
            raise ValueError("CSG continuation widths must be finite and positive.")
        if any(right > left for left, right in zip(widths_, widths_[1:])):
            raise ValueError("CSG continuation widths must be nonincreasing.")
        tolerance = float(terminal_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("terminal_tolerance must be finite and nonnegative.")
        self.widths = widths_
        self.terminal_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {"kind": "csg-continuation", "widths": widths_, "terminal": tolerance}
        )


class PreparedCSGContinuation(StrictModule):
    """One fixed smooth topology, terminal sharp topology, and exact state maps."""

    policy: CSGContinuationPolicy
    smooth_geometry: CompiledGeometry
    sharp_geometry: CompiledGeometry
    smooth_constraints: tuple[AbstractDesignConstraint, ...]
    sharp_constraints: tuple[AbstractDesignConstraint, ...]
    width_parameter_ids: tuple[ParameterId, ...] = eqx.field(static=True)
    shared_parameter_ids: tuple[ParameterId, ...] = eqx.field(static=True)
    smooth_shared_indices: tuple[int, ...] = eqx.field(static=True)
    sharp_shared_indices: tuple[int, ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


class CSGContinuationResult(StrictModule):
    """Per-epoch optimization evidence and independent sharp acceptance."""

    state: DesignState
    terminal_state: DesignState
    epoch_results: tuple[Any, ...]
    terminal_residual: Array
    terminal_residual_norm: Array
    sharp_valid: Array
    accepted: Array
    widths: Array
    preparation_id: str = eqx.field(static=True)


class _SmoothBuild:
    def __init__(self):
        self.width_ids: list[ParameterId] = []

    def convert(self, source: GeometrySource, path: str = "root") -> GeometrySource:
        if isinstance(source, SharpCSG):
            feature_id = f"csg-continuation-{path}"
            width_id = ParameterId(feature_id, "width")
            self.width_ids.append(width_id)
            children = tuple(
                self.convert(child, f"{path}-{index}")
                for index, child in enumerate(source.children)
            )
            return BlendCSG(
                children,
                1.0,
                operation=source.operation,
                feature_id=feature_id,
            )
        if isinstance(source, BlendCSG):
            raise TypeError(
                "Continuation input must be sharp; nested BlendCSG is unsupported."
            )
        if not isinstance(source, GeometrySource):
            raise TypeError("CSG children must be GeometrySource values.")
        return source


def _freeze_widths(
    geometry: CompiledGeometry, width_ids: tuple[ParameterId, ...], /
) -> CompiledGeometry:
    width_set = frozenset(width_ids)
    specs = tuple(
        replace(spec, trainable=False) if spec.parameter_id in width_set else spec
        for spec in geometry.schema.specs
    )
    schema = ParameterSchema(specs)
    state = DesignState(schema, geometry.state.values)
    return CompiledGeometry(geometry.kernel, state, tolerance=geometry.tolerance)


def prepare_csg_continuation(
    source: SharpCSG,
    constraints: Sequence[AbstractDesignConstraint],
    policy: CSGContinuationPolicy,
    /,
    *,
    tolerance: GeometryTolerance = GeometryTolerance(),
) -> PreparedCSGContinuation:
    """Prepare a supported recursive sharp intersection/difference continuation."""
    if not isinstance(source, SharpCSG):
        raise TypeError("source must be a SharpCSG intersection or difference.")
    if not isinstance(policy, CSGContinuationPolicy):
        raise TypeError("policy must be a CSGContinuationPolicy.")
    constraints_ = tuple(constraints)
    if not constraints_ or any(
        not isinstance(value, AbstractDesignConstraint) for value in constraints_
    ):
        raise TypeError("constraints must contain design constraint objects.")
    builder = _SmoothBuild()
    smooth_source = builder.convert(source)
    width_ids = tuple(builder.width_ids)
    if not width_ids:
        raise ValueError("No supported sharp CSG node was found.")
    smooth = _freeze_widths(smooth_source.compile(tolerance=tolerance), width_ids)
    sharp = source.compile(tolerance=tolerance)
    # Construction validates capabilities and every parameter reference on both epochs.
    DesignConstraintSystem(smooth, constraints_)
    DesignConstraintSystem(sharp, constraints_)
    smooth_ids = smooth.schema.parameter_ids
    sharp_ids = sharp.schema.parameter_ids
    shared = tuple(
        parameter_id for parameter_id in sharp_ids if parameter_id in smooth_ids
    )
    if set(shared) != set(sharp_ids):
        missing = set(sharp_ids) - set(shared)
        raise ValueError(
            f"Smooth continuation omits sharp parameters: {sorted(map(str, missing))}."
        )
    smooth_indices = tuple(smooth.schema.index(parameter_id) for parameter_id in shared)
    sharp_indices = tuple(sharp.schema.index(parameter_id) for parameter_id in shared)
    for smooth_index, sharp_index in zip(smooth_indices, sharp_indices, strict=True):
        left, right = smooth.schema.specs[smooth_index], sharp.schema.specs[sharp_index]
        if (left.shape, left.dtype, left.role) != (right.shape, right.dtype, right.role):
            raise ValueError("Shared CSG parameter schemas are not exactly transferable.")
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-csg-continuation",
            "policy": policy.policy_id,
            "width_ids": [str(value) for value in width_ids],
            "shared": [str(value) for value in shared],
        }
    )
    return PreparedCSGContinuation(
        policy,
        smooth,
        sharp,
        constraints_,
        constraints_,
        width_ids,
        shared,
        smooth_indices,
        sharp_indices,
        identifier,
    )


def _with_width(
    prepared: PreparedCSGContinuation, state: DesignState, width: float, /
) -> DesignState:
    updates = {
        parameter_id: jnp.asarray(
            width, dtype=state.values[state.schema.index(parameter_id)].dtype
        )
        for parameter_id in prepared.width_parameter_ids
    }
    return state.updated(updates)


def _terminal_state(
    prepared: PreparedCSGContinuation, smooth_state: DesignState, /
) -> DesignState:
    values = list(prepared.sharp_geometry.state.values)
    for smooth_index, sharp_index in zip(
        prepared.smooth_shared_indices,
        prepared.sharp_shared_indices,
        strict=True,
    ):
        values[sharp_index] = smooth_state.values[smooth_index]
    return DesignState(prepared.sharp_geometry.schema, values)


def solve_csg_continuation(
    prepared: PreparedCSGContinuation,
    /,
    *,
    initial_state: DesignState | None = None,
    search: DifferentialEvolutionSearch | None = None,
    key: Key[Array, ""] | None = None,
    bounds: Mapping[ParameterId, tuple[ArrayLike, ArrayLike]] | None = None,
    solve_options: Mapping[str, Any] | None = None,
) -> CSGContinuationResult:
    """Run fixed-width epochs and audit terminal acceptance on sharp geometry."""
    if not isinstance(prepared, PreparedCSGContinuation):
        raise TypeError("prepared must be a PreparedCSGContinuation.")
    state = prepared.smooth_geometry.state if initial_state is None else initial_state
    if (
        not isinstance(state, DesignState)
        or state.schema != prepared.smooth_geometry.schema
    ):
        raise ValueError("initial_state must match the prepared smooth schema.")
    if search is not None and not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be DifferentialEvolutionSearch or None.")
    if search is not None and key is None:
        raise ValueError("Global continuation search requires an explicit key.")
    options = {} if solve_options is None else dict(solve_options)
    epoch_results: list[Any] = []
    system = DesignConstraintSystem(prepared.smooth_geometry, prepared.smooth_constraints)
    for epoch, width in enumerate(prepared.policy.widths):
        state = _with_width(prepared, state, width)
        if search is None:
            result = system.solve(initial_state=state, **options)
        else:
            result = system.search(
                search,
                key=jr.fold_in(key, epoch),
                bounds=bounds,
                initial_state=state,
            )
        epoch_results.append(result)
        state = result.state
    terminal_state = _terminal_state(prepared, state)
    sharp_system = DesignConstraintSystem(
        prepared.sharp_geometry, prepared.sharp_constraints
    )
    residual = sharp_system.residual(terminal_state)
    residual_norm = jnp.linalg.norm(residual)
    validity = prepared.sharp_geometry.validity(terminal_state)
    accepted = (
        validity.accepted
        & jnp.all(jnp.isfinite(residual))
        & (residual_norm <= prepared.policy.terminal_tolerance)
    )
    return CSGContinuationResult(
        state,
        terminal_state,
        tuple(epoch_results),
        residual,
        residual_norm,
        validity.accepted,
        accepted,
        jnp.asarray(prepared.policy.widths),
        prepared.preparation_id,
    )


__all__ = [
    "CSGContinuationPolicy",
    "CSGContinuationResult",
    "PreparedCSGContinuation",
    "prepare_csg_continuation",
    "solve_csg_continuation",
]

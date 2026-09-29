#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Block assembly of spatial coupled problems into native linear/nonlinear problems.

Owner coordinates are the native state and row blocks of every component and
the unknown/row blocks of every law. Solve coordinates remove the rows that a
law eliminates by an explicit relation. The coupled chart maps solve
coordinates to owner coordinates (``L z + h``); residual rows are pulled back
by its transpose, so eliminated rows are summed into the rows they are
expressed by. Component constraint maps are composed exactly once, where a
full-space contribution meets its component.

The linear operator is a native ``BlockLinearOperator`` over named solve
blocks. Owner operators and contribution operators keep their own
(sparse, structured, or matrix-free) representation; a solve block is only
wrapped when an elimination actually changes its coordinates. With a prepared
lane execution, grouped components and contribution blocks leave the block
grid and act through their bounded lane worksets, added to it exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from ..._strict import StrictModule
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    assemble_block_operator,
    BlockSpace,
    FunctionLinearOperator,
    LinearSystem,
    NullspacePolicy,
)
from ...nonlinear import NonlinearSystemProblem
from ._components import AbstractSpatialComponent, ComponentField
from ._contributions import (
    Contribution,
    ContributionEndpoint,
    EliminationContribution,
    LinearContribution,
    LoadContribution,
    ResidualContribution,
)
from ._execution import (
    component_action_program,
    component_fields_program,
    component_residual_program,
    evaluate_lanes,
    interface_action_program,
    InterfaceLaneSubject,
    lane_arguments,
    LaneAction,
    LaneProgram,
    LaneWorkset,
    PreparedCoupledExecution,
    transposed_inverse_riesz,
)
from ._laws import PreparedLaw


type OwnerPath = tuple[str, str]
type OwnerValues = dict[OwnerPath, Array]
type LinearMap = Callable[[Array], Array]


@final
class _Elimination(StrictModule):
    """One elimination resolved to owner coordinates of its component block."""

    component: str = eqx.field(static=True)
    state_block: str = eqx.field(static=True)
    row_block: str = eqx.field(static=True)
    positions: Array
    retained: str = eqx.field(static=True)
    retained_field: str = eqx.field(static=True)
    columns: Array
    relation: Array


@final
class CoupledChart(StrictModule):
    """Owner and solve layouts of a coupled problem and the map between them.

    ``owner_states`` evaluates ``L z + h`` block by block (eliminated rows are
    filled from their retained fields, including the retained lifts);
    ``pull_back_rows`` applies ``L^T`` to owner-coordinate residual rows.
    ``execution`` holds the prepared lane worksets, or ``None`` for
    per-owner execution.
    """

    components: tuple[AbstractSpatialComponent, ...]
    laws: tuple[PreparedLaw, ...]
    kept_states: tuple[tuple[OwnerPath, Array], ...]
    kept_rows: tuple[tuple[OwnerPath, Array], ...]
    eliminations: tuple[_Elimination, ...]
    order: tuple[str, ...] = eqx.field(static=True)
    state_space: BlockSpace
    row_space: BlockSpace
    execution: PreparedCoupledExecution | None = None

    def component(self, name: str, /) -> AbstractSpatialComponent:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"Unknown component {name!r}.")

    def law(self, name: str, /) -> PreparedLaw:
        for law in self.laws:
            if law.law_id == name:
                return law
        raise KeyError(f"Unknown law {name!r}.")

    def kept_positions(self, path: OwnerPath, /, *, rows: bool = False) -> Array | None:
        """Owner positions kept as solve coordinates (``None``: the whole block)."""
        for candidate, positions in self.kept_rows if rows else self.kept_states:
            if candidate == path:
                return positions
        return None

    def owner_state_paths(self) -> tuple[tuple[OwnerPath, AbstractVectorSpace], ...]:
        return _owner_paths(self.components, self.laws, rows=False)

    def owner_row_paths(self) -> tuple[tuple[OwnerPath, AbstractVectorSpace], ...]:
        return _owner_paths(self.components, self.laws, rows=True)

    def solve_index(self, path: OwnerPath, /, *, rows: bool = False) -> tuple[int, int]:
        space = self.row_space if rows else self.state_space
        owner = space.names.index(path[0])
        member = space.spaces[owner]
        if not isinstance(member, BlockSpace):
            raise TypeError("Solve owners are block spaces.")
        return owner, member.names.index(path[1])

    def _embed(self, path: OwnerPath, value: Array, size: int, /) -> Array:
        positions = self.kept_positions(path)
        if positions is None:
            return value
        return jnp.zeros((size,), dtype=value.dtype).at[positions].set(value)

    def owner_states(
        self, state: tuple[tuple[Array, ...], ...], args: Mapping[str, object], /
    ) -> OwnerValues:
        """Owner-coordinate blocks ``L z + h`` of one solve state."""
        values: OwnerValues = {}
        for path, space in self.owner_state_paths():
            owner, block = self.solve_index(path)
            values[path] = self._embed(path, state[owner][block], space.size)
        by_component = _eliminations_by_component(self.eliminations)
        for name in self.order:
            for elimination in by_component.get(name, ()):
                retained = self.component(elimination.retained)
                full = retained.expand(
                    elimination.retained_field,
                    _component_state(retained, values),
                    args[retained.name],
                )
                path = (name, elimination.state_block)
                values[path] = (
                    values[path]
                    .at[elimination.positions]
                    .set(elimination.relation @ full[elimination.columns])
                )
        return values

    def accumulate_rows(self, rows: OwnerValues, /) -> OwnerValues:
        """Owner rows with every eliminated row summed into the rows expressing it.

        The kept positions of the result are the coupled equations in owner
        coordinates (for example the conforming sum of two reactions on a
        matching interface); eliminated positions are left unchanged.
        """
        accumulated = dict(rows)
        by_component = _eliminations_by_component(self.eliminations)
        for name in reversed(self.order):
            for elimination in by_component.get(name, ()):
                retained = self.component(elimination.retained)
                record = retained.field(elimination.retained_field)
                eliminated = accumulated[(name, elimination.row_block)][
                    elimination.positions
                ]
                covector = (
                    jnp.zeros((record.full_space.size,), dtype=eliminated.dtype)
                    .at[elimination.columns]
                    .add(elimination.relation.T @ eliminated)
                )
                target = (retained.name, record.row_block)
                accumulated[target] = accumulated[target] + record.pull_back(covector)
        return accumulated

    def pull_back_rows(self, rows: OwnerValues, /) -> tuple[tuple[Array, ...], ...]:
        """Solve-coordinate rows ``L^T r`` of owner-coordinate residual rows."""
        accumulated = self.accumulate_rows(rows)
        solve: list[list[Array]] = [[] for _ in self.row_space.names]
        for path, _ in self.owner_row_paths():
            owner, _ = self.solve_index(path, rows=True)
            positions = self.kept_positions(path, rows=True)
            value = accumulated[path]
            solve[owner].append(value if positions is None else value[positions])
        return tuple(tuple(blocks) for blocks in solve)


def _eliminations_by_component(
    eliminations: tuple[_Elimination, ...], /
) -> dict[str, list[_Elimination]]:
    """Eliminations of each component in declaration order (one pass)."""
    grouped: dict[str, list[_Elimination]] = {}
    for elimination in eliminations:
        grouped.setdefault(elimination.component, []).append(elimination)
    return grouped


def _component_state(
    component: AbstractSpatialComponent, values: OwnerValues, /
) -> tuple[Array, ...]:
    return tuple(values[(component.name, block.name)] for block in component.state_blocks)


# --- Chart construction ------------------------------------------------------------------


def _field_positions(field: ComponentField, rows: np.ndarray, /) -> np.ndarray:
    """Owner-coordinate positions of full field rows in the field's state block."""
    if field.constraint is None:
        return rows
    if field.free_rows is None:
        raise ValueError(
            f"Field {field.name!r} has a constraint chart that is not a row "
            "selection; matching elimination cannot address its rows."
        )
    free = np.asarray(field.free_rows)
    positions = np.searchsorted(free, rows)
    if np.any(positions >= free.size) or np.any(
        free[np.minimum(positions, free.size - 1)] != rows
    ):
        raise ValueError(
            "Eliminated rows must be free rows of their owner; rows the owner "
            "imposes strongly are not eliminated."
        )
    return positions


def _resolve_elimination(
    contribution: EliminationContribution,
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> _Elimination:
    component = components[contribution.eliminated.owner]
    field = component.field(contribution.eliminated.block)
    retained = components[contribution.retained.owner]
    retained.field(contribution.retained.block)
    state = component.state_blocks[component.state_index(field.state_block)].space
    row = component.row_blocks[component.row_index(field.row_block)].space
    if len(state.structure().shape) != 1 or state.size != row.size:
        raise ValueError(
            "Elimination needs a one-dimensional state block whose row block has "
            "the same coordinates."
        )
    positions = _field_positions(field, np.asarray(contribution.rows))
    return _Elimination(
        component.name,
        field.state_block,
        field.row_block,
        jnp.asarray(positions.astype(np.int32)),
        retained.name,
        contribution.retained.block,
        contribution.columns,
        contribution.relation,
    )


def _dependency_order(
    names: tuple[str, ...], eliminations: tuple[_Elimination, ...], /
) -> tuple[str, ...]:
    """Components ordered so that every retained side precedes its eliminated side."""
    order: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in order:
            return
        if name in visiting:
            raise ValueError(
                "Eliminations form a cycle; every eliminated row must be expressed "
                "by rows that are not eliminated through it."
            )
        visiting.add(name)
        for elimination in eliminations:
            if elimination.component == name:
                visit(elimination.retained)
        visiting.discard(name)
        order.append(name)

    for name in names:
        visit(name)
    return tuple(order)


def _kept_positions(
    eliminations: tuple[_Elimination, ...],
    components: Mapping[str, AbstractSpatialComponent],
    /,
    *,
    rows: bool,
) -> tuple[tuple[OwnerPath, Array], ...]:
    """Owner positions that stay solve coordinates in every eliminated block."""
    removed: dict[OwnerPath, np.ndarray] = {}
    for elimination in eliminations:
        block = elimination.row_block if rows else elimination.state_block
        path = (elimination.component, block)
        previous = removed.get(path, np.zeros((0,), dtype=np.int32))
        positions = np.asarray(elimination.positions)
        if np.intersect1d(previous, positions).size:
            raise ValueError("Two eliminations remove the same rows.")
        removed[path] = np.union1d(previous, positions)
    kept: list[tuple[OwnerPath, Array]] = []
    for path, positions in sorted(removed.items()):
        component = components[path[0]]
        blocks = component.row_blocks if rows else component.state_blocks
        size = next(block.space.size for block in blocks if block.name == path[1])
        remaining = np.setdiff1d(np.arange(size, dtype=np.int32), positions)
        kept.append((path, jnp.asarray(remaining)))
    return tuple(kept)


def _solve_space(
    paths: tuple[tuple[OwnerPath, AbstractVectorSpace], ...],
    kept: tuple[tuple[OwnerPath, Array], ...],
    owners: tuple[str, ...],
    /,
) -> BlockSpace:
    lookup = dict(kept)
    members: list[AbstractVectorSpace] = []
    for owner in owners:
        spaces: list[AbstractVectorSpace] = []
        names: list[str] = []
        for (candidate, block), space in paths:
            if candidate != owner:
                continue
            positions = lookup.get((candidate, block))
            reduced = (
                space
                if positions is None
                else ArraySpace((positions.shape[0],), dtype=space.structure().dtype)
            )
            spaces.append(reduced)
            names.append(block)
        members.append(BlockSpace(tuple(spaces), names=tuple(names)))
    return BlockSpace(tuple(members), names=owners)


def prepare_coupled_chart(
    components: tuple[AbstractSpatialComponent, ...],
    laws: tuple[PreparedLaw, ...],
    /,
    *,
    execution: PreparedCoupledExecution | None = None,
) -> CoupledChart:
    """Resolve eliminations and build the owner/solve layouts."""
    lookup = {component.name: component for component in components}
    eliminations = tuple(
        _resolve_elimination(contribution, lookup)
        for law in laws
        for contribution in law.contributions
        if isinstance(contribution, EliminationContribution)
    )
    order = _dependency_order(tuple(lookup), eliminations)
    kept_states = _kept_positions(eliminations, lookup, rows=False)
    kept_rows = _kept_positions(eliminations, lookup, rows=True)
    owners = tuple(lookup) + tuple(law.law_id for law in laws if law.state_blocks)
    state_paths = _owner_paths(components, laws, rows=False)
    row_paths = _owner_paths(components, laws, rows=True)
    return CoupledChart(
        components=components,
        laws=laws,
        kept_states=kept_states,
        kept_rows=kept_rows,
        eliminations=eliminations,
        order=order,
        state_space=_solve_space(state_paths, kept_states, owners),
        row_space=_solve_space(row_paths, kept_rows, owners),
        execution=execution,
    )


def _owner_paths(
    components: tuple[AbstractSpatialComponent, ...],
    laws: tuple[PreparedLaw, ...],
    /,
    *,
    rows: bool,
) -> tuple[tuple[OwnerPath, AbstractVectorSpace], ...]:
    paths: list[tuple[OwnerPath, AbstractVectorSpace]] = []
    for component in components:
        blocks = component.row_blocks if rows else component.state_blocks
        paths += [((component.name, block.name), block.space) for block in blocks]
    for law in laws:
        blocks = law.row_blocks if rows else law.state_blocks
        paths += [((law.law_id, block.name), block.space) for block in blocks]
    return tuple(paths)


# --- Endpoint resolution -----------------------------------------------------------------


def _owner_block_space(
    chart: CoupledChart, owner: str, block: str, /, *, rows: bool
) -> AbstractVectorSpace:
    for (candidate, name), space in (
        chart.owner_row_paths() if rows else chart.owner_state_paths()
    ):
        if candidate == owner and name == block:
            return space
    kind = "row" if rows else "state"
    raise ValueError(f"Owner {owner!r} has no {kind} block {block!r}.")


def _endpoint_field(
    chart: CoupledChart, endpoint: ContributionEndpoint, /
) -> tuple[AbstractSpatialComponent, ComponentField]:
    names = tuple(component.name for component in chart.components)
    if endpoint.owner not in names:
        raise ValueError(
            f"Full-space endpoint {endpoint.owner!r} must name a component field."
        )
    component = chart.component(endpoint.owner)
    return component, component.field(endpoint.block)


def endpoint_source_space(
    chart: CoupledChart, endpoint: ContributionEndpoint, /
) -> AbstractVectorSpace:
    """Space a contribution reads at ``endpoint``."""
    match endpoint.space:
        case "full":
            return _endpoint_field(chart, endpoint)[1].full_space
        case "reduced":
            return _owner_block_space(chart, endpoint.owner, endpoint.block, rows=False)


def endpoint_target_space(
    chart: CoupledChart, endpoint: ContributionEndpoint, /
) -> AbstractVectorSpace:
    """Row space a contribution writes at ``endpoint``."""
    match endpoint.space:
        case "full":
            return _endpoint_field(chart, endpoint)[1].row_space
        case "reduced":
            return _owner_block_space(chart, endpoint.owner, endpoint.block, rows=True)


def _require_compatible(
    space: AbstractVectorSpace, expected: AbstractVectorSpace, what: str, /
) -> None:
    if not space.compatible(expected):
        raise ValueError(
            f"Row-space mismatch: the contribution's {what} space {space.space_id} is "
            f"not the endpoint's space {expected.space_id}; a contribution prepared on "
            "one chart (full or reduced) cannot be applied on another."
        )


def _require_structure(value: object, space: AbstractVectorSpace, what: str, /) -> None:
    if eqx.tree_equal(value, space.structure()) is not True:
        raise ValueError(f"Row-space mismatch: the {what} does not match its endpoint.")


def validate_contributions(chart: CoupledChart, /) -> None:
    """Resolve every endpoint and check each contribution against its spaces."""
    for law in chart.laws:
        for contribution in law.contributions:
            _validate_contribution(chart, contribution)


def _validate_contribution(chart: CoupledChart, contribution: Contribution, /) -> None:
    match contribution:
        case LinearContribution():
            _require_compatible(
                contribution.operator.source,
                endpoint_source_space(chart, contribution.source),
                "source",
            )
            _require_compatible(
                contribution.operator.target,
                endpoint_target_space(chart, contribution.target),
                "target",
            )
        case LoadContribution():
            _require_structure(
                jax.ShapeDtypeStruct(
                    contribution.values.shape, contribution.values.dtype
                ),
                endpoint_target_space(chart, contribution.target),
                "load",
            )
        case ResidualContribution():
            inputs = tuple(
                endpoint_source_space(chart, endpoint).structure()
                for endpoint in contribution.sources
            )
            expected = tuple(
                endpoint_target_space(chart, endpoint).structure()
                for endpoint in contribution.targets
            )
            if contribution.residual.runtime_inputs:
                # A residual reading runtime arguments has no argument-free
                # evaluation; its own evaluation checks its output structure.
                return
            outputs = jax.eval_shape(
                lambda values: contribution.residual.evaluate(values, None), inputs
            )
            if eqx.tree_equal(outputs, expected) is not True:
                raise ValueError(
                    "Row-space mismatch: the residual contribution's outputs do not "
                    "match its target endpoints."
                )
        case EliminationContribution():
            for endpoint in contribution.sources:
                endpoint_source_space(chart, endpoint)


# --- Owner-coordinate residual -----------------------------------------------------------


type FieldValues = dict[OwnerPath, Array]


def _component_lanes(
    chart: CoupledChart,
    workset: LaneWorkset,
    values: OwnerValues,
    args: Mapping[str, object],
    program: Callable[[object], LaneProgram],
    /,
) -> tuple[tuple[AbstractSpatialComponent, PyTree[Array]], ...]:
    """One lane program over a component workset at owner states ``values``."""
    members = tuple(chart.component(name) for name in workset.members)
    shared, dynamic = lane_arguments(workset, [args[name] for name in workset.members])
    outputs = evaluate_lanes(
        workset,
        program(shared),
        [
            (_component_state(component, values), lane)
            for component, lane in zip(members, dynamic, strict=True)
        ],
    )
    return tuple(zip(members, outputs, strict=True))


def _lane_worksets(chart: CoupledChart, /) -> tuple[LaneWorkset, ...]:
    return () if chart.execution is None else chart.execution.components


def _grouped_components(chart: CoupledChart, /) -> frozenset[str]:
    return frozenset() if chart.execution is None else chart.execution.grouped_components


def component_residuals(
    chart: CoupledChart, values: OwnerValues, args: Mapping[str, object], /
) -> OwnerValues:
    """Native residual rows of every component (lane groups as bounded lanes)."""
    rows: OwnerValues = {}
    for workset in _lane_worksets(chart):
        lanes = _component_lanes(chart, workset, values, args, component_residual_program)
        for component, residual in lanes:
            for block, value in zip(component.row_blocks, residual, strict=True):
                rows[(component.name, block.name)] = value
    grouped = _grouped_components(chart)
    for component in chart.components:
        if component.name in grouped:
            continue
        residual = component.residual(
            _component_state(component, values), args[component.name]
        )
        for block, value in zip(component.row_blocks, residual, strict=True):
            rows[(component.name, block.name)] = value
    return rows


def component_fields(
    chart: CoupledChart,
    values: OwnerValues,
    args: Mapping[str, object],
    /,
    *,
    requested: frozenset[OwnerPath] | None = None,
) -> FieldValues:
    """Full coefficients ``P z + g`` of component fields, each expanded once.

    ``requested`` restricts per-owner expansion to those ``(component, field)``
    pairs (``None``: every field); a lane group expands every field of its
    members once any of them is requested.
    """
    fields: FieldValues = {}
    for workset in _lane_worksets(chart):
        if requested is not None and not any(
            owner in workset.members for owner, _ in requested
        ):
            continue
        lanes = _component_lanes(chart, workset, values, args, component_fields_program)
        for component, expanded in lanes:
            for record, value in zip(component.fields, expanded, strict=True):
                fields[(component.name, record.name)] = value
    grouped = _grouped_components(chart)
    for component in chart.components:
        if component.name in grouped:
            continue
        for record in component.fields:
            if requested is None or (component.name, record.name) in requested:
                fields[(component.name, record.name)] = component.expand(
                    record.name,
                    _component_state(component, values),
                    args[component.name],
                )
    return fields


def _full_sources(chart: CoupledChart, /) -> frozenset[OwnerPath]:
    """Component fields that law contributions read in full coordinates."""
    requested: set[OwnerPath] = set()
    for law in chart.laws:
        for contribution in law.contributions:
            sources: tuple[ContributionEndpoint, ...]
            match contribution:
                case LinearContribution():
                    sources = (contribution.source,)
                case ResidualContribution():
                    sources = contribution.sources
                case LoadContribution() | EliminationContribution():
                    sources = ()
            requested |= {
                (endpoint.owner, endpoint.block)
                for endpoint in sources
                if endpoint.space == "full"
            }
    return frozenset(requested)


def _source_value(
    endpoint: ContributionEndpoint, values: OwnerValues, fields: FieldValues, /
) -> Array:
    match endpoint.space:
        case "full":
            return fields[(endpoint.owner, endpoint.block)]
        case "reduced":
            return values[(endpoint.owner, endpoint.block)]


def _add_rows(
    chart: CoupledChart,
    rows: OwnerValues,
    endpoint: ContributionEndpoint,
    covector: Array,
    /,
) -> None:
    match endpoint.space:
        case "full":
            component, field = _endpoint_field(chart, endpoint)
            path = (component.name, field.row_block)
            rows[path] = rows[path] + field.pull_back(covector)
        case "reduced":
            path = (endpoint.owner, endpoint.block)
            rows[path] = rows[path] + covector


def _contribution_rows(
    chart: CoupledChart,
    contribution: Contribution,
    rows: OwnerValues,
    sources: tuple[OwnerValues, FieldValues],
    args: Mapping[str, object],
    /,
) -> None:
    values, fields = sources
    match contribution:
        case LinearContribution():
            source = _source_value(contribution.source, values, fields)
            _add_rows(chart, rows, contribution.target, contribution.operator.mv(source))
        case LoadContribution():
            _add_rows(chart, rows, contribution.target, contribution.values)
        case ResidualContribution():
            inputs = tuple(
                _source_value(endpoint, values, fields)
                for endpoint in contribution.sources
            )
            outputs = contribution.residual.evaluate(inputs, args)
            for endpoint, output in zip(contribution.targets, outputs, strict=True):
                _add_rows(chart, rows, endpoint, output)
        case EliminationContribution():
            pass


def owner_residual(
    chart: CoupledChart,
    values: OwnerValues,
    args: Mapping[str, object],
    /,
    *,
    native: OwnerValues | None = None,
) -> OwnerValues:
    """Owner-coordinate rows: native component residuals plus law contributions.

    ``native`` supplies already evaluated ``component_residuals`` at ``values``;
    every full-coordinate law source is expanded once per evaluation.
    """
    rows = dict(component_residuals(chart, values, args) if native is None else native)
    for law in chart.laws:
        for block in law.row_blocks:
            rows[(law.law_id, block.name)] = block.space.zeros()
    fields = component_fields(chart, values, args, requested=_full_sources(chart))
    for law in chart.laws:
        for contribution in law.contributions:
            _contribution_rows(chart, contribution, rows, (values, fields), args)
    return rows


def coupled_residual(
    chart: CoupledChart,
    state: tuple[tuple[Array, ...], ...],
    args: Mapping[str, object],
    /,
) -> tuple[tuple[Array, ...], ...]:
    """Solve-coordinate residual ``L^T R(L z + h)`` of the coupled problem."""
    values = chart.owner_states(state, args)
    return chart.pull_back_rows(owner_residual(chart, values, args))


# --- Linear assembly -------------------------------------------------------------------------


type _Term = tuple[OwnerPath, LinearMap | None, LinearMap | None]


def _keep_maps(positions: Array, size: int, /) -> tuple[LinearMap, LinearMap]:
    def forward(value: Array) -> Array:
        return jnp.zeros((size,), dtype=value.dtype).at[positions].set(value)

    def transpose(value: Array) -> Array:
        return value[positions]

    return forward, transpose


def _cross_maps(
    elimination: _Elimination, field: ComponentField, size: int, /
) -> tuple[LinearMap, LinearMap]:
    """``x_r -> scatter(E (P_r x_r)[columns])`` and its exact transpose."""
    prolongation = field.prolongation()

    def forward(value: Array) -> Array:
        full = value if prolongation is None else prolongation.mv(value)
        local = elimination.relation @ full[elimination.columns]
        return jnp.zeros((size,), dtype=value.dtype).at[elimination.positions].set(local)

    def transpose(value: Array) -> Array:
        local = elimination.relation.T @ value[elimination.positions]
        full = (
            jnp.zeros((field.full_space.size,), dtype=value.dtype)
            .at[elimination.columns]
            .add(local)
        )
        return full if prolongation is None else prolongation.transpose_mv(full)

    return forward, transpose


def _compose(outer: tuple[LinearMap, LinearMap], inner: _Term, /) -> _Term:
    path, forward, transpose = inner
    outer_forward, outer_transpose = outer

    def composed(value: Array) -> Array:
        return outer_forward(value if forward is None else forward(value))

    def composed_transpose(value: Array) -> Array:
        result = outer_transpose(value)
        return result if transpose is None else transpose(result)

    return path, composed, composed_transpose


def _terms(chart: CoupledChart, path: OwnerPath, /, *, rows: bool) -> tuple[_Term, ...]:
    """Solve blocks feeding one owner block (``L`` column terms or ``M`` row terms)."""
    positions = chart.kept_positions(path, rows=rows)
    if positions is None:
        return ((path, None, None),)
    size = _owner_block_space(chart, path[0], path[1], rows=rows).size
    terms: list[_Term] = [(path, *_keep_maps(positions, size))]
    for elimination in chart.eliminations:
        block = elimination.row_block if rows else elimination.state_block
        if (elimination.component, block) != path:
            continue
        retained = chart.component(elimination.retained)
        field = retained.field(elimination.retained_field)
        inner_path = (retained.name, field.row_block if rows else field.state_block)
        maps = _cross_maps(elimination, field, size)
        terms += [_compose(maps, term) for term in _terms(chart, inner_path, rows=rows)]
    return tuple(terms)


def _solve_block_space(
    chart: CoupledChart, path: OwnerPath, /, *, rows: bool
) -> AbstractVectorSpace:
    owner, block = chart.solve_index(path, rows=rows)
    space = chart.row_space if rows else chart.state_space
    member = space.spaces[owner]
    if not isinstance(member, BlockSpace):
        raise TypeError("Solve owners are block spaces.")
    return member.spaces[block]


def _solve_entry(
    chart: CoupledChart,
    operator: AbstractLinearOperator,
    row: _Term,
    column: _Term,
    /,
) -> tuple[OwnerPath, OwnerPath, AbstractLinearOperator]:
    row_path, row_forward, row_transpose = row
    column_path, column_forward, column_transpose = column
    if row_forward is None and column_forward is None:
        return row_path, column_path, operator
    source = _solve_block_space(chart, column_path, rows=False)
    target = _solve_block_space(chart, row_path, rows=True)

    def action(value: Array) -> Array:
        state = value if column_forward is None else column_forward(value)
        result = operator.mv(state)
        return result if row_transpose is None else row_transpose(result)

    def transposed(value: Array) -> Array:
        rows = value if row_forward is None else row_forward(value)
        result = operator.transpose_mv(rows)
        return result if column_transpose is None else column_transpose(result)

    return (
        row_path,
        column_path,
        FunctionLinearOperator(
            action, source=source, target=target, transpose_action=transposed
        ),
    )


def _contribution_operator(
    chart: CoupledChart,
    target: ContributionEndpoint,
    source: ContributionEndpoint,
    operator: AbstractLinearOperator,
    /,
) -> tuple[OwnerPath, OwnerPath, AbstractLinearOperator]:
    """Compose a contribution with its components' constraint maps exactly once."""
    column = (source.owner, source.block)
    if source.space == "full":
        _, field = _endpoint_field(chart, source)
        prolongation = field.prolongation()
        column = (source.owner, field.state_block)
        if prolongation is not None:
            operator = operator @ prolongation
    row = (target.owner, target.block)
    if target.space == "full":
        _, field = _endpoint_field(chart, target)
        pullback = field.pullback_operator()
        row = (target.owner, field.row_block)
        if pullback is not None:
            operator = pullback @ operator
    return row, column, operator


def _affine_residual_operators(
    chart: CoupledChart,
    contribution: ResidualContribution,
    args: Mapping[str, object],
    /,
) -> list[tuple[OwnerPath, OwnerPath, AbstractLinearOperator]]:
    """Exact linear parts of an affine residual contribution, one per endpoint pair."""
    spaces = tuple(
        endpoint_source_space(chart, endpoint) for endpoint in contribution.sources
    )
    zeros = tuple(space.zeros() for space in spaces)
    entries: list[tuple[OwnerPath, OwnerPath, AbstractLinearOperator]] = []
    for source_index, (endpoint, space) in enumerate(
        zip(contribution.sources, spaces, strict=True)
    ):
        for target_index, target in enumerate(contribution.targets):

            def action(
                value: Array,
                source_index: int = source_index,
                target_index: int = target_index,
            ) -> Array:
                def local(entry: Array) -> Array:
                    inputs = zeros[:source_index] + (entry,) + zeros[source_index + 1 :]
                    return contribution.residual.evaluate(inputs, args)[target_index]

                return jax.jvp(local, (zeros[source_index],), (value,))[1]

            operator = FunctionLinearOperator(
                action, source=space, target=endpoint_target_space(chart, target)
            )
            entries.append(_contribution_operator(chart, target, endpoint, operator))
    return entries


type _Entry = tuple[OwnerPath, OwnerPath, AbstractLinearOperator]


def _component_entries(
    component: AbstractSpatialComponent, args: object, /
) -> list[_Entry]:
    """Non-zero blocks of a component's own native affine operator."""
    operator = component.linear_operator(args)
    if operator is None:
        raise ValueError(
            f"Component {component.name!r} publishes no affine operator; solve "
            "the coupled problem through its nonlinear residual."
        )
    return [
        ((component.name, row.name), (component.name, column.name), block)
        for i, row in enumerate(component.row_blocks)
        for j, column in enumerate(component.state_blocks)
        if (block := operator.blocks[i][j]) is not None
    ]


def contribution_key(law: PreparedLaw, index: int, /) -> str:
    """Canonical key of one law contribution: law ID and declaration position."""
    return f"{law.law_id}#{index}"


def _grouped_contributions(chart: CoupledChart, /) -> frozenset[str]:
    return (
        frozenset() if chart.execution is None else chart.execution.grouped_contributions
    )


def _law_entries(
    chart: CoupledChart, law: PreparedLaw, args: Mapping[str, object], /
) -> list[_Entry]:
    """Linear contribution operators of one law, composed with constraint maps.

    Contributions grouped into interface lane worksets act through them.
    """
    grouped = _grouped_contributions(chart)
    entries: list[_Entry] = []
    for index, contribution in enumerate(law.contributions):
        match contribution:
            case LinearContribution():
                if contribution_key(law, index) in grouped:
                    continue
                entries.append(
                    _contribution_operator(
                        chart,
                        contribution.target,
                        contribution.source,
                        contribution.operator,
                    )
                )
            case ResidualContribution():
                if not contribution.affine:
                    raise ValueError(
                        f"Law {law.law_id!r} publishes a nonlinear residual; solve "
                        "the coupled problem through its nonlinear residual."
                    )
                entries += _affine_residual_operators(chart, contribution, args)
            case LoadContribution() | EliminationContribution():
                pass
    return entries


def owner_linear_entries(
    chart: CoupledChart, args: Mapping[str, object], /
) -> list[_Entry]:
    """Owner-coordinate operator entries outside the lane worksets."""
    grouped = _grouped_components(chart)
    entries: list[_Entry] = []
    for component in chart.components:
        if component.name not in grouped:
            entries += _component_entries(component, args[component.name])
    for law in chart.laws:
        entries += _law_entries(chart, law, args)
    return entries


type InterfaceRoute = tuple[OwnerPath, OwnerPath, OwnerPath]


def interface_lane_candidates(
    chart: CoupledChart, /
) -> tuple[tuple[str, InterfaceRoute, InterfaceLaneSubject], ...]:
    """Linear contribution blocks whose solve paths are owner paths.

    Each candidate is ``(key, (row, identified state, column), subject)``;
    blocks on eliminated rows keep their composed per-law route.
    """
    candidates: list[tuple[str, InterfaceRoute, InterfaceLaneSubject]] = []
    for law in chart.laws:
        for index, contribution in enumerate(law.contributions):
            if not isinstance(contribution, LinearContribution):
                continue
            row, column, operator = _contribution_operator(
                chart, contribution.target, contribution.source, contribution.operator
            )
            if (
                chart.kept_positions(row, rows=True) is not None
                or chart.kept_positions(column) is not None
            ):
                continue
            state = _paired_state(chart, row)
            identified = _solve_block_space(chart, state, rows=False)
            candidates.append(
                (
                    contribution_key(law, index),
                    (row, state, column),
                    InterfaceLaneSubject(operator, identified),
                )
            )
    return tuple(candidates)


def _read_blocks(
    chart: CoupledChart,
    value: tuple[tuple[Array, ...], ...],
    paths: tuple[OwnerPath, ...],
    /,
    *,
    rows: bool,
) -> tuple[Array, ...]:
    indices = (chart.solve_index(path, rows=rows) for path in paths)
    return tuple(value[owner][block] for owner, block in indices)


def _placed(
    chart: CoupledChart,
    updates: list[tuple[OwnerPath, Array]],
    /,
    *,
    rows: bool,
) -> tuple[tuple[Array, ...], ...]:
    """Solve-coordinate tree of the summed ``updates`` and zeros elsewhere."""
    space = chart.row_space if rows else chart.state_space
    blocks = [list(member) for member in space.zeros()]
    for path, value in updates:
        owner, block = chart.solve_index(path, rows=rows)
        blocks[owner][block] = blocks[owner][block] + value
    return tuple(tuple(member) for member in blocks)


def _lane_actions(identified: bool, /) -> tuple[LaneAction, LaneAction]:
    return (
        ("identified", "identified-transpose")
        if identified
        else ("weak", "weak-transpose")
    )


def _component_lane_operator(
    chart: CoupledChart,
    workset: LaneWorkset,
    args: Mapping[str, object],
    /,
    *,
    identified: bool,
) -> AbstractLinearOperator:
    """Own affine blocks of a component workset as one bounded lane operator."""
    members = tuple(chart.component(name) for name in workset.members)
    shared, dynamic = lane_arguments(workset, [args[name] for name in workset.members])
    states = [tuple((c.name, b.name) for b in c.state_blocks) for c in members]
    targets = (
        states
        if identified
        else [tuple((c.name, b.name) for b in c.row_blocks) for c in members]
    )
    forward_action, transpose_action = _lane_actions(identified)

    def apply(
        value: tuple[tuple[Array, ...], ...],
        action: LaneAction,
        route: tuple[
            list[tuple[OwnerPath, ...]], bool, list[tuple[OwnerPath, ...]], bool
        ],
    ) -> tuple[tuple[Array, ...], ...]:
        sources, source_rows, destinations, destination_rows = route
        inputs = [
            (_read_blocks(chart, value, paths, rows=source_rows), lane)
            for paths, lane in zip(sources, dynamic, strict=True)
        ]
        outputs = evaluate_lanes(
            workset, component_action_program(shared, action), inputs
        )
        updates = [
            (path, block)
            for paths, output in zip(destinations, outputs, strict=True)
            for path, block in zip(paths, output, strict=True)
        ]
        return _placed(chart, updates, rows=destination_rows)

    return FunctionLinearOperator(
        lambda value: apply(
            value, forward_action, (states, False, targets, not identified)
        ),
        source=chart.state_space,
        target=chart.state_space if identified else chart.row_space,
        transpose_action=lambda value: apply(
            value, transpose_action, (targets, not identified, states, False)
        ),
    )


def _interface_lane_operator(
    chart: CoupledChart, workset: LaneWorkset, /, *, identified: bool
) -> AbstractLinearOperator:
    """Grouped law contribution blocks as one bounded lane operator."""
    targets = tuple(state if identified else row for row, state, _ in workset.routes)
    columns = tuple(column for _, _, column in workset.routes)
    forward_action, transpose_action = _lane_actions(identified)

    def apply(
        value: tuple[tuple[Array, ...], ...],
        action: LaneAction,
        route: tuple[tuple[OwnerPath, ...], bool, tuple[OwnerPath, ...], bool],
    ) -> tuple[tuple[Array, ...], ...]:
        sources, source_rows, destinations, destination_rows = route
        inputs = list(_read_blocks(chart, value, sources, rows=source_rows))
        outputs = evaluate_lanes(workset, interface_action_program(action), inputs)
        return _placed(
            chart, list(zip(destinations, outputs, strict=True)), rows=destination_rows
        )

    return FunctionLinearOperator(
        lambda value: apply(
            value, forward_action, (columns, False, targets, not identified)
        ),
        source=chart.state_space,
        target=chart.state_space if identified else chart.row_space,
        transpose_action=lambda value: apply(
            value, transpose_action, (targets, not identified, columns, False)
        ),
    )


def _with_lanes(
    chart: CoupledChart,
    operator: AbstractLinearOperator,
    args: Mapping[str, object],
    /,
    *,
    identified: bool,
) -> AbstractLinearOperator:
    """The block grid plus every lane workset's operator (exact sum)."""
    if chart.execution is None:
        return operator
    for workset in chart.execution.components:
        operator = operator + _component_lane_operator(
            chart, workset, args, identified=identified
        )
    for workset in chart.execution.interfaces:
        operator = operator + _interface_lane_operator(
            chart, workset, identified=identified
        )
    return operator


def _solve_entries(
    chart: CoupledChart,
    entries: list[tuple[OwnerPath, OwnerPath, AbstractLinearOperator]],
    /,
) -> list[tuple[tuple[str, str], tuple[str, str], AbstractLinearOperator]]:
    grouped: dict[tuple[OwnerPath, OwnerPath], AbstractLinearOperator] = {}
    order: list[tuple[OwnerPath, OwnerPath]] = []
    for row_path, column_path, operator in entries:
        for row in _terms(chart, row_path, rows=True):
            for column in _terms(chart, column_path, rows=False):
                key_row, key_column, term = _solve_entry(chart, operator, row, column)
                key = (key_row, key_column)
                if key in grouped:
                    grouped[key] = grouped[key] + term
                else:
                    grouped[key] = term
                    order.append(key)
    return [(row, column, grouped[(row, column)]) for row, column in order]


def coupled_weak_operator(
    chart: CoupledChart, args: Mapping[str, object], /
) -> AbstractLinearOperator:
    """Weak coupled Jacobian ``J`` from solve states to solve residual rows."""
    entries = _solve_entries(chart, owner_linear_entries(chart, args))
    operator = assemble_block_operator(
        entries, source=chart.state_space, target=chart.row_space
    )
    return _with_lanes(chart, operator, args, identified=False)


def _paired_state(chart: CoupledChart, path: OwnerPath, /) -> OwnerPath:
    """State block identified with a row block (same owner, same position)."""
    owner, block = chart.solve_index(path, rows=True)
    member = chart.state_space.spaces[owner]
    if not isinstance(member, BlockSpace):
        raise TypeError("Solve owners are block spaces.")
    return path[0], member.names[block]


def _riesz_identification(
    row: AbstractVectorSpace, state: AbstractVectorSpace, /
) -> AbstractLinearOperator:
    """Inverse Riesz map ``M^{-1}`` of the state block, identifying its residual rows.

    Its coordinate transpose is ``M^{-T}`` (``transposed_inverse_riesz``).
    """
    return FunctionLinearOperator(
        state.inverse_riesz,
        source=row,
        target=state,
        transpose_action=lambda value: transposed_inverse_riesz(state, value),
    )


def require_square_owners(chart: CoupledChart, /) -> None:
    """Every owner pairs its row blocks one-to-one with equally sized state blocks."""
    for owner, rows, states in zip(
        chart.row_space.names,
        chart.row_space.spaces,
        chart.state_space.spaces,
        strict=True,
    ):
        if not isinstance(rows, BlockSpace) or not isinstance(states, BlockSpace):
            raise TypeError("Solve owners are block spaces.")
        sizes = (
            [space.size for space in rows.spaces],
            [space.size for space in states.spaces],
        )
        if sizes[0] != sizes[1]:
            raise ValueError(
                f"Row-space mismatch: owner {owner!r} has row blocks of sizes "
                f"{sizes[0]} for state blocks of sizes {sizes[1]}."
            )


def coupled_linear_system(
    chart: CoupledChart,
    args: Mapping[str, object],
    /,
    *,
    nullspace_policy: NullspacePolicy | None,
) -> tuple[LinearSystem, tuple[tuple[Array, ...], ...]]:
    """Native block ``LinearSystem`` of an affine coupled problem.

    Residual rows are identified with the paired state blocks through their
    inverse Riesz maps (as the owners' own ``linear_system`` do), so the
    operator is a named block endomorphism of the solve state space that both
    direct and Krylov methods accept; the right-hand side is the identified
    ``-F(0)``.
    """
    entries: list[tuple[OwnerPath, OwnerPath, AbstractLinearOperator]] = []
    for row, column, operator in _solve_entries(chart, owner_linear_entries(chart, args)):
        state = _paired_state(chart, row)
        identify = _riesz_identification(
            _solve_block_space(chart, row, rows=True),
            _solve_block_space(chart, state, rows=False),
        )
        entries.append((state, column, identify @ operator))
    operator = _with_lanes(
        chart,
        assemble_block_operator(
            entries, source=chart.state_space, target=chart.state_space
        ),
        args,
        identified=True,
    )
    residual = coupled_residual(chart, chart.state_space.zeros(), args)
    rhs = chart.state_space.inverse_riesz(jax.tree.map(lambda value: -value, residual))
    return LinearSystem(operator, nullspace_policy=nullspace_policy), rhs


def coupled_nonlinear_problem(
    chart: CoupledChart, problem_id: str, /
) -> NonlinearSystemProblem:
    """Native nonlinear residual of the coupled problem on its solve state space.

    The residual rows are identified with the state blocks through their
    inverse Riesz maps, matching ``coupled_linear_system``.
    """

    def residual(
        state: tuple[tuple[Array, ...], ...], args: Mapping[str, object]
    ) -> tuple[tuple[Array, ...], ...]:
        return chart.state_space.inverse_riesz(coupled_residual(chart, state, args))

    return NonlinearSystemProblem(
        residual,
        state_space=chart.state_space,
        residual_space=chart.state_space,
        problem_id=problem_id,
    )


__all__ = ["CoupledChart"]

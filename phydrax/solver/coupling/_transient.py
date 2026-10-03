#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transient coupled problems lowered onto the native index-one DAE runtime.

A prepared spatial coupled problem already owns the canonical named solve
layout, the owner residuals, and every law contribution. A transient
declaration adds, per differential field, the owner's own capacity operator
``M`` and states which components are quasistatic. The coupled rows become

``F(t, z, z') = L^T [ P^T M (d/dt u(t, z)) + R(L z + h(t)) ] = 0``,

where ``u(t, z) = P (L z + h(t)) + g(t)`` is the full field of a differential
component. Its time derivative is evaluated exactly (one forward derivative in
``(z, t)`` with tangent ``(z', 1)``), so time-dependent lifts and eliminated
coordinates that follow a time-dependent retained field contribute their actual
rate terms. Law unknowns and quasistatic components are algebraic.

The declared block incidence (component rows on their own states, capacity rows
on their field rates, law contributions on their sources, eliminations on their
retained fields) is lowered through the existing structural DAE compiler with
no differentiation or tearing capacity: only systems that are structurally
index one are admitted; an unreduced higher-index multiplier system (for
example a mortar constraint on differential traces) is refused with the
structural analysis as evidence. Local regularity remains evidence of the
native DAE solve. The native runtime keeps its ``ArraySpace`` stage
coordinates; named block views are available through ``DAECoordinateAdapter``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._admissibility import guard_derivative_validity
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier, positive_finite_float
from ...dynamics import (
    AcausalDAESource,
    analyze_dae_structure,
    compile_acausal_dae,
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEJet,
    DAERole,
    DAEStructuralAnalysis,
    DAEStructuralPolicy,
    DAEVariableBlock,
    ReducedDAECompilation,
    TimeGrid,
)
from ...linalg import AbstractVectorSpace, BlockSpace
from .._dae_coordinate_adapter import DAECoordinateAdapter
from .._differential_algebraic import (
    DAEContinuation,
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    DifferentialAlgebraicSolution,
    solve_dae,
)
from ._assembly import CoupledChart, owner_residual, OwnerPath, OwnerValues
from ._components import AbstractCapacityComponent, AbstractPreparedCapacity
from ._contributions import (
    Contribution,
    ContributionEndpoint,
    EliminationContribution,
    LinearContribution,
    LoadContribution,
    ResidualContribution,
)
from ._prepared_problem import PreparedCoupledProblem


type TransientArguments = Callable[[Array, Any], Mapping[str, object]]
type NestedBlocks = tuple[tuple[Array, ...], ...]
type _Edge = tuple[OwnerPath, int]
type _Incidence = dict[OwnerPath, set[_Edge]]

# Only structurally index-one systems reach the native runtime: no index
# reduction by differentiation and no tearing.
_INDEX_ONE = DAEStructuralPolicy(0, 0)


def _scalar_or_shaped(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be real floating.")
    return array


@final
class TransientField(StrictModule):
    """One differential field of a coupled component and its capacity coefficient.

    The component must publish ``AbstractCapacityComponent.prepare_capacity``;
    ``capacity`` multiplies the owner's own mass operator (for heat conduction
    the volumetric heat capacity ``rho c``).
    """

    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    capacity: Array

    def __init__(
        self, component: str, field: str, /, *, capacity: ArrayLike = 1.0
    ) -> None:
        capacity_ = _scalar_or_shaped(capacity, "capacity")
        if capacity_.shape != ():
            raise ValueError("capacity must be one scalar.")
        self.component = canonical_identifier(component, "component")
        self.field = canonical_identifier(field, "field")
        self.capacity = capacity_


@final
class TransientScale(StrictModule):
    """Native DAE scales of one solve state block and its paired row block.

    ``state`` and ``rate`` scale the state block ``(owner, block)``;
    ``residual`` scales the row block that the owner pairs with it (same owner,
    same position). Scalars or block-sized arrays.
    """

    owner: str = eqx.field(static=True)
    block: str = eqx.field(static=True)
    state: Array
    rate: Array
    residual: Array

    def __init__(
        self,
        owner: str,
        block: str,
        /,
        *,
        state: ArrayLike = 1.0,
        rate: ArrayLike = 1.0,
        residual: ArrayLike = 1.0,
    ) -> None:
        self.owner = canonical_identifier(owner, "owner")
        self.block = canonical_identifier(block, "block")
        self.state = _scalar_or_shaped(state, "state")
        self.rate = _scalar_or_shaped(rate, "rate")
        self.residual = _scalar_or_shaped(residual, "residual")

    @property
    def path(self) -> OwnerPath:
        return (self.owner, self.block)


def _paths(space: BlockSpace, /) -> tuple[tuple[OwnerPath, AbstractVectorSpace], ...]:
    """Solve paths ``(owner, block)`` and leaf spaces in canonical order."""
    paths: list[tuple[OwnerPath, AbstractVectorSpace]] = []
    for owner, member in zip(space.names, space.spaces, strict=True):
        if not isinstance(member, BlockSpace):
            raise TypeError("Coupled solve owners are block spaces.")
        paths.extend(
            ((owner, block), leaf)
            for block, leaf in zip(member.names, member.spaces, strict=True)
        )
    return tuple(paths)


def _nest(space: BlockSpace, values: Mapping[OwnerPath, Array], /) -> NestedBlocks:
    nested: list[tuple[Array, ...]] = []
    for owner, member in zip(space.names, space.spaces, strict=True):
        if not isinstance(member, BlockSpace):
            raise TypeError("Coupled solve owners are block spaces.")
        nested.append(
            tuple(
                leaf.unflatten(values[(owner, block)].reshape((-1,)))
                for block, leaf in zip(member.names, member.spaces, strict=True)
            )
        )
    return tuple(nested)


@final
class _TransientCore(StrictModule):
    """Exact transient rows of one prepared coupled problem."""

    prepared: PreparedCoupledProblem
    fields: tuple[TransientField, ...]
    capacities: tuple[AbstractPreparedCapacity, ...]
    arguments: TransientArguments

    def owner_arguments(self, time: Array, parameters: object, /) -> dict[str, object]:
        return self.prepared.arguments(self.arguments(time, parameters))

    def _states_and_fields(
        self, state: NestedBlocks, time: Array, parameters: object, /
    ) -> tuple[OwnerValues, tuple[Array, ...]]:
        chart = self.prepared.chart
        args = self.owner_arguments(time, parameters)
        values = chart.owner_states(state, args)
        fulls = tuple(
            _component_expand(chart, field, values, args) for field in self.fields
        )
        return values, fulls

    def parts(
        self,
        time: Array,
        state: NestedBlocks,
        rate: NestedBlocks,
        parameters: object,
        /,
    ) -> tuple[NestedBlocks, NestedBlocks]:
        """Solve rows of the steady residual and of the capacity terms."""
        # Every equation block of the lowered DAE reads these rows; one cached
        # trace serves all of them (and the host incidence verification).
        return _cached_parts(self, time, state, rate, parameters)

    def evaluate_parts(
        self,
        time: Array,
        state: NestedBlocks,
        rate: NestedBlocks,
        parameters: object,
        /,
    ) -> tuple[NestedBlocks, NestedBlocks]:
        chart = self.prepared.chart
        time_ = jnp.asarray(time)
        (values, _), (_, full_rates) = jax.jvp(
            lambda z, t: self._states_and_fields(z, t, parameters),
            (state, time_),
            (rate, jnp.ones_like(time_)),
        )
        args = self.owner_arguments(time_, parameters)
        steady = chart.pull_back_rows(owner_residual(chart, values, args))
        capacity: OwnerValues = {
            path: jnp.zeros((space.size,), dtype=time_.dtype)
            for path, space in chart.owner_row_paths()
        }
        for field, prepared_capacity, full_rate in zip(
            self.fields, self.capacities, full_rates, strict=True
        ):
            record = chart.component(field.component).field(field.field)
            mass = prepared_capacity.operator(args[field.component])
            path = (field.component, record.row_block)
            capacity[path] = capacity[path] + record.pull_back(mass.mv(full_rate))
        return steady, chart.pull_back_rows(capacity)

    def rows(
        self,
        time: Array,
        state: NestedBlocks,
        rate: NestedBlocks,
        parameters: object,
        /,
    ) -> NestedBlocks:
        steady, capacity = self.parts(time, state, rate, parameters)
        return jax.tree.map(jnp.add, steady, capacity)

    def capacity_actions(
        self,
        time: Array,
        state: NestedBlocks,
        rate: NestedBlocks,
        parameters: object,
        /,
    ) -> dict[str, Array]:
        """Norm of each component's full-field capacity action ``M d/dt u``.

        Every row of the owner's field is included: the pulled-back capacity rows
        of a component can cancel (a retained rate balancing the rate of the
        coordinates an elimination makes follow another field), so their norm
        alone is not the size of the individual terms.
        """
        time_ = jnp.asarray(time)
        _, (_, full_rates) = jax.jvp(
            lambda z, t: self._states_and_fields(z, t, parameters),
            (state, time_),
            (rate, jnp.ones_like(time_)),
        )
        args = self.owner_arguments(time_, parameters)
        actions: dict[str, Array] = {}
        for field, prepared_capacity, full_rate in zip(
            self.fields, self.capacities, full_rates, strict=True
        ):
            action = prepared_capacity.operator(args[field.component]).mv(full_rate)
            norm = jnp.sqrt(
                sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree.leaves(action))
            )
            actions[field.component] = actions.get(field.component, 0.0) + norm
        return actions


@eqx.filter_jit
def _cached_parts(
    core: _TransientCore,
    time: Array,
    state: NestedBlocks,
    rate: NestedBlocks,
    parameters: object,
    /,
) -> tuple[NestedBlocks, NestedBlocks]:
    return core.evaluate_parts(time, state, rate, parameters)


def _component_expand(
    chart: CoupledChart,
    field: TransientField,
    values: OwnerValues,
    args: Mapping[str, object],
    /,
) -> Array:
    component = chart.component(field.component)
    blocks = tuple(
        values[(component.name, block.name)] for block in component.state_blocks
    )
    return component.expand(field.field, blocks, args[component.name])


# --- Declared structural incidence ------------------------------------------------------


def _endpoint_path(
    chart: CoupledChart, endpoint: ContributionEndpoint, /, *, rows: bool
) -> OwnerPath:
    if endpoint.space == "reduced":
        return (endpoint.owner, endpoint.block)
    record = chart.component(endpoint.owner).field(endpoint.block)
    return (endpoint.owner, record.row_block if rows else record.state_block)


def _contribution_edges(
    chart: CoupledChart, contribution: Contribution, /
) -> tuple[tuple[OwnerPath, tuple[OwnerPath, ...]], ...]:
    match contribution:
        case LinearContribution():
            return (
                (
                    _endpoint_path(chart, contribution.target, rows=True),
                    (_endpoint_path(chart, contribution.source, rows=False),),
                ),
            )
        case ResidualContribution():
            sources = tuple(
                _endpoint_path(chart, endpoint, rows=False)
                for endpoint in contribution.sources
            )
            return tuple(
                (_endpoint_path(chart, endpoint, rows=True), sources)
                for endpoint in contribution.targets
            )
        case LoadContribution() | EliminationContribution():
            return ()
        case _:
            raise TypeError(f"Unknown contribution {type(contribution).__name__}.")


def _owner_incidence(
    chart: CoupledChart, fields: tuple[TransientField, ...], /
) -> _Incidence:
    """Owner-row incidence: own states, capacity rates, and law sources."""
    incidence: _Incidence = {path: set() for path, _ in chart.owner_row_paths()}
    for component in chart.components:
        own = {((component.name, block.name), 0) for block in component.state_blocks}
        for row in component.row_blocks:
            incidence[(component.name, row.name)] |= own
    for field in fields:
        record = chart.component(field.component).field(field.field)
        incidence[(field.component, record.row_block)].add(
            ((field.component, record.state_block), 1)
        )
    for law in chart.laws:
        for contribution in law.contributions:
            for target, sources in _contribution_edges(chart, contribution):
                incidence[target] |= {(source, 0) for source in sources}
    return incidence


def _solve_incidence(chart: CoupledChart, incidence: _Incidence, /) -> _Incidence:
    """Close owner incidence under the chart's eliminations (``L`` and ``L^T``)."""
    substitutions = {
        (item.component, item.state_block): (
            item.retained,
            chart.component(item.retained).field(item.retained_field).state_block,
        )
        for item in chart.eliminations
    }
    changed = True
    while changed:
        changed = False
        for edges in incidence.values():
            extra = {
                (substitutions[path], order)
                for path, order in edges
                if path in substitutions
            } - edges
            if extra:
                edges |= extra
                changed = True
    for name in reversed(chart.order):
        for item in chart.eliminations:
            if item.component != name:
                continue
            record = chart.component(item.retained).field(item.retained_field)
            incidence[(item.retained, record.row_block)] |= incidence[
                (name, item.row_block)
            ]
    return incidence


# --- Declaration validation --------------------------------------------------------------


def _validated_declaration(
    prepared: PreparedCoupledProblem,
    fields: Sequence[TransientField],
    quasistatic: Sequence[str],
    /,
) -> tuple[tuple[TransientField, ...], tuple[str, ...]]:
    fields_ = tuple(fields)
    if any(not isinstance(field, TransientField) for field in fields_):
        raise TypeError("fields must be TransientField values.")
    quasistatic_ = tuple(
        sorted(canonical_identifier(name, "quasistatic") for name in quasistatic)
    )
    names = {component.name for component in prepared.components}
    declared = [field.component for field in fields_]
    unknown = (set(declared) | set(quasistatic_)) - names
    if unknown:
        raise ValueError(
            f"Transient declarations name unknown components {sorted(unknown)}."
        )
    if len({(field.component, field.field) for field in fields_}) != len(fields_):
        raise ValueError("Each differential field is declared once.")
    if len(set(quasistatic_)) != len(quasistatic_) or set(declared) & set(quasistatic_):
        raise ValueError("A component is either transient or quasistatic, once.")
    undeclared = names - set(declared) - set(quasistatic_)
    if undeclared:
        raise ValueError(
            f"Components {sorted(undeclared)} are neither transient nor declared "
            "quasistatic; a steady component in a transient problem must be declared."
        )
    for field in fields_:
        _capacity_component(prepared, field.component).field(field.field)
    ordered = tuple(sorted(fields_, key=lambda field: (field.component, field.field)))
    return ordered, quasistatic_


def _capacity_component(
    prepared: PreparedCoupledProblem, name: str, /
) -> AbstractCapacityComponent:
    component = prepared.chart.component(name)
    if not isinstance(component, AbstractCapacityComponent):
        raise TypeError(f"Component {name!r} publishes no capacity operator.")
    return component


def _refuse_rates_on_algebraic(
    incidence: _Incidence, differential: frozenset[OwnerPath], /
) -> None:
    for row, edges in sorted(incidence.items()):
        for path, order in sorted(edges):
            if order > 0 and path not in differential:
                raise ValueError(
                    f"Capacity rows {row!r} carry the rate of {path!r}, which is not "
                    "differential: an eliminated transient field is expressed by a "
                    "quasistatic field. Eliminate the quasistatic side instead."
                )


def _refuse_unadmitted(analysis: DAEStructuralAnalysis, /) -> None:
    if analysis.successful:
        return
    counts = dict(
        zip(analysis.equation_names, analysis.differentiation_counts, strict=True)
    )
    raise ValueError(
        "Coupled transient refused: structural DAE analysis status "
        f"{analysis.status!r} (index-one admission, no differentiation or tearing); "
        f"required differentiations {counts}, matching {dict(analysis.matching)}, "
        f"unmatched equations {analysis.unmatched_equations}, unmatched variables "
        f"{analysis.unmatched_variables}. An unreduced higher-index multiplier "
        "system is not integrated; declare a matching elimination or a qualified "
        "multiplier-free imposition."
    )


# --- Preparation ------------------------------------------------------------------------


def _scale_lookup(
    scales: Sequence[TransientScale], paths: tuple[OwnerPath, ...], /
) -> dict[OwnerPath, TransientScale]:
    lookup: dict[OwnerPath, TransientScale] = {}
    for scale in scales:
        if not isinstance(scale, TransientScale):
            raise TypeError("scales must be TransientScale values.")
        if scale.path not in paths:
            raise ValueError(f"Scale names unknown solve block {scale.path!r}.")
        if scale.path in lookup:
            raise ValueError(f"Solve block {scale.path!r} is scaled twice.")
        lookup[scale.path] = scale
    return lookup


def _variable_name(path: OwnerPath, /) -> str:
    return f"{path[0]}.{path[1]}"


def _variables(
    states: tuple[tuple[OwnerPath, AbstractVectorSpace], ...],
    differential: frozenset[OwnerPath],
    scales: Mapping[OwnerPath, TransientScale],
    /,
) -> tuple[DAEVariableBlock, ...]:
    blocks: list[DAEVariableBlock] = []
    for path, space in states:
        scale = scales.get(path)
        blocks.append(
            DAEVariableBlock(
                _variable_name(path),
                (space.size,),
                1 if path in differential else 0,
                state_scale=1.0 if scale is None else scale.state,
                rate_scale=1.0 if scale is None else scale.rate,
            )
        )
    return tuple(blocks)


def _equation(
    core: _TransientCore,
    state_space: BlockSpace,
    differential: frozenset[OwnerPath],
    location: tuple[int, int],
    /,
) -> Callable[[Array, DAEJet, Any], Array]:
    states = _paths(state_space)

    def residual(time: Array, jet: DAEJet, parameters: object) -> Array:
        values = {path: jet.value(_variable_name(path)) for path, _ in states}
        rates = {
            path: (
                jet.value(_variable_name(path), 1)
                if path in differential
                else jnp.zeros_like(values[path])
            )
            for path, _ in states
        }
        rows = core.rows(
            time,
            _nest(state_space, values),
            _nest(state_space, rates),
            parameters,
        )
        return jnp.reshape(rows[location[0]][location[1]], (-1,))

    return residual


def _equations(
    core: _TransientCore,
    incidence: _Incidence,
    differential: frozenset[OwnerPath],
    scales: Mapping[OwnerPath, TransientScale],
    identity: str,
    /,
) -> tuple[DAEEquationBlock, ...]:
    prepared = core.prepared
    state_space = prepared.state_space
    row_space = prepared.row_space
    blocks: list[DAEEquationBlock] = []
    for owner_index, member in enumerate(row_space.spaces):
        if not isinstance(member, BlockSpace):
            raise TypeError("Coupled row owners are block spaces.")
        owner = row_space.names[owner_index]
        state_member = state_space.spaces[state_space.names.index(owner)]
        if not isinstance(state_member, BlockSpace):
            raise TypeError("Coupled solve owners are block spaces.")
        for block_index, block in enumerate(member.names):
            path = (owner, block)
            paired = (owner, state_member.names[block_index])
            scale = scales.get(paired)
            edges = tuple(
                DAEDerivativeIncidence(_variable_name(source), order)
                for source, order in sorted(incidence[path])
            )
            blocks.append(
                DAEEquationBlock(
                    _variable_name(path),
                    _equation(
                        core, state_space, differential, (owner_index, block_index)
                    ),
                    edges,
                    residual_semantic_id=f"coupled-transient-rows:{prepared.plan_id}:{owner}.{block}",
                    residual_numeric_id=f"{identity}:{owner}.{block}",
                    residual_scale=1.0 if scale is None else scale.residual,
                )
            )
    return tuple(blocks)


@final
class PreparedCoupledTransient(StrictModule):
    """Coupled transient rows lowered onto one admitted native DAE.

    ``compilation`` is the structurally admitted index-one DAE whose native
    coordinates are the named solve blocks of ``prepared.state_space``;
    ``adapter`` exposes them as named variable/equation blocks with their
    differential/algebraic roles and native scales. ``state_view`` /
    ``native_state`` convert between native coordinates and the nested solve
    layout of the spatial problem.
    """

    prepared: PreparedCoupledProblem
    core: _TransientCore
    quasistatic: tuple[str, ...] = eqx.field(static=True)
    compilation: ReducedDAECompilation
    adapter: DAECoordinateAdapter
    paths: tuple[OwnerPath, ...] = eqx.field(static=True)
    roles: tuple[DAERole, ...] = eqx.field(static=True)
    transient_id: str = eqx.field(static=True)

    @property
    def fields(self) -> tuple[TransientField, ...]:
        return self.core.fields

    @property
    def state_space(self) -> BlockSpace:
        return self.prepared.state_space

    @property
    def row_space(self) -> BlockSpace:
        return self.prepared.row_space

    @property
    def analysis(self) -> DAEStructuralAnalysis:
        return self.compilation.analysis

    def state_view(self, native: ArrayLike, /) -> NestedBlocks:
        """Nested solve blocks of one native state, rate, or increment."""
        blocks = self.adapter.state_view(jnp.asarray(native))
        lookup = dict(zip(self.adapter.variable_space.names, blocks, strict=True))
        values = {
            path: jnp.asarray(lookup[f"{self.prepared.plan_id}.{_variable_name(path)}"])
            for path in self.paths
        }
        return _nest(self.state_space, values)

    def native_state(self, state: NestedBlocks, /) -> Array:
        """Native coordinates of one nested solve state (or rate)."""
        nested = self.state_space.validate(state)
        values: dict[OwnerPath, Array] = {}
        for path in self.paths:
            owner, block = self._location(path)
            values[path] = jnp.reshape(nested[owner][block], (-1,))
        ordered = tuple(
            values[self._path_of(name)] for name in self.adapter.variable_space.names
        )
        return self.adapter.native_state(ordered)

    def _location(self, path: OwnerPath, /) -> tuple[int, int]:
        owner = self.state_space.names.index(path[0])
        member = self.state_space.spaces[owner]
        if not isinstance(member, BlockSpace):
            raise TypeError("Coupled solve owners are block spaces.")
        return owner, member.names.index(path[1])

    def _path_of(self, name: str, /) -> OwnerPath:
        prefix = f"{self.prepared.plan_id}."
        for path in self.paths:
            if name == prefix + _variable_name(path):
                return path
        raise KeyError(f"No solve block for native variable {name!r}.")

    def residual(
        self,
        time: ArrayLike,
        state: NestedBlocks,
        rate: NestedBlocks,
        parameters: object = None,
        /,
    ) -> NestedBlocks:
        """Unscaled coupled transient rows ``F(t, z, z')`` in ``row_space``."""
        return self.core.rows(
            jnp.asarray(time),
            self.state_space.validate(state),
            self.state_space.validate(rate),
            parameters,
        )

    def field(
        self,
        component: str,
        field: str,
        time: ArrayLike,
        state: NestedBlocks,
        parameters: object = None,
        /,
    ) -> Array:
        """Full coefficients of one component field at ``time`` (lifts included)."""
        args = self.core.owner_arguments(jnp.asarray(time), parameters)
        return self.prepared.field(
            component, field, self.state_space.validate(state), args
        )

    def problem(
        self,
        initial_state: NestedBlocks,
        /,
        *,
        parameters: object = None,
        problem_id: str | None = None,
    ) -> DifferentialAlgebraicProblem:
        """Native DAE initial-value problem with structural consistent initialization.

        Differential blocks of ``initial_state`` are fixed; algebraic blocks are
        guesses solved together with the rates at the initial time.
        """
        return DifferentialAlgebraicProblem(
            self.compilation,
            self.native_state(initial_state),
            args=parameters,
            initialization="structural",
            problem_id=problem_id,
        )


def prepare_coupled_transient(
    prepared: PreparedCoupledProblem,
    /,
    *,
    fields: Sequence[TransientField],
    arguments: TransientArguments,
    arguments_id: str,
    quasistatic: Sequence[str] = (),
    scales: Sequence[TransientScale] = (),
    parameters: object = None,
) -> PreparedCoupledTransient:
    """Admit and lower one transient coupled declaration onto the native DAE.

    ``arguments(time, parameters)`` returns the per-component owner arguments
    at ``time`` (lifts, sources, coefficients); it must be differentiable in
    ``time`` so that lift rates are exact. ``arguments_id`` names that map.
    ``parameters`` is a sample of the runtime DAE arguments used to verify the
    declared incidence once on the host.
    """
    if not isinstance(prepared, PreparedCoupledProblem):
        raise TypeError("prepared must be a PreparedCoupledProblem.")
    if not callable(arguments):
        raise TypeError("arguments must be callable.")
    arguments_id_ = canonical_identifier(arguments_id, "arguments_id")
    fields_, quasistatic_ = _validated_declaration(prepared, fields, quasistatic)
    chart = prepared.chart
    records = {
        (field.component, chart.component(field.component).field(field.field).state_block)
        for field in fields_
    }
    differential = frozenset(records)
    states = _paths(prepared.state_space)
    paths = tuple(path for path, _ in states)
    scale_lookup = _scale_lookup(scales, paths)
    incidence = _solve_incidence(chart, _owner_incidence(chart, fields_))
    _refuse_rates_on_algebraic(incidence, differential)
    identity = canonical_fingerprint(
        {
            "kind": "coupled-transient",
            "problem": prepared.problem_id,
            "fields": [[field.component, field.field] for field in fields_],
            "quasistatic": list(quasistatic_),
            "arguments": arguments_id_,
        }
    )
    capacities = tuple(
        _capacity_component(prepared, field.component).prepare_capacity(
            field.field, coefficient=field.capacity
        )
        for field in fields_
    )
    core = _TransientCore(prepared, fields_, capacities, arguments)
    source = AcausalDAESource(
        (
            DAEComponent(
                prepared.plan_id,
                _variables(states, differential, scale_lookup),
                _equations(core, incidence, differential, scale_lookup, identity),
            ),
        )
    )
    _refuse_unadmitted(analyze_dae_structure(source, _INDEX_ONE))
    compilation = compile_acausal_dae(source, _INDEX_ONE, args=parameters)
    return PreparedCoupledTransient(
        prepared=prepared,
        core=core,
        quasistatic=quasistatic_,
        compilation=compilation,
        adapter=DAECoordinateAdapter(compilation),
        paths=paths,
        roles=tuple(
            "differential" if path in differential else "algebraic" for path in paths
        ),
        transient_id=identity,
    )


# --- Execution and acceptance -----------------------------------------------------------


@final
class TransientCertificate(StrictModule):
    """Original coupled transient rows at every sample, per solve owner.

    ``residual_norms[i, k]`` is the norm of ``F(t_i, z_i, z'_i)`` on owner
    ``owners[k]``. ``scales[i, k]`` sums the norms of its individual terms, as
    the steady component certificates do: the state-dependent steady rows
    ``R(t, z) - R(t, 0)``, the steady offset ``R(t, 0)`` (sources and lifts), the
    rate-dependent capacity rows, the capacity offset (lift rates), and the
    component's full-field capacity action. A sample is accepted when every
    owner norm is finite and at most ``tolerance`` times its scale.
    """

    owners: tuple[str, ...] = eqx.field(static=True)
    residual_norms: Array
    scales: Array
    accepted: Array


def _owner_norms(rows: NestedBlocks, /) -> Array:
    return jnp.stack(
        [jnp.sqrt(sum(jnp.sum(jnp.abs(block) ** 2) for block in owner)) for owner in rows]
    )


def _certificate(
    transient: PreparedCoupledTransient,
    dae: DifferentialAlgebraicSolution,
    parameters: object,
    tolerance: float,
    /,
) -> TransientCertificate:
    core = transient.core

    def sample(time: Array, native: Array, rate: Array) -> tuple[Array, Array]:
        state = transient.state_view(native)
        state_rate = transient.state_view(rate)
        steady, capacity = core.parts(time, state, state_rate, parameters)
        zero = transient.state_space.zeros()
        steady_offset, capacity_offset = core.parts(time, zero, zero, parameters)
        total = jax.tree.map(jnp.add, steady, capacity)
        actions = core.capacity_actions(time, state, state_rate, parameters)
        scale = (
            _owner_norms(jax.tree.map(jnp.subtract, steady, steady_offset))
            + _owner_norms(steady_offset)
            + _owner_norms(jax.tree.map(jnp.subtract, capacity, capacity_offset))
            + _owner_norms(capacity_offset)
            + jnp.stack(
                [
                    jnp.asarray(actions.get(owner, 0.0), dtype=time.dtype)
                    for owner in transient.row_space.names
                ]
            )
        )
        return _owner_norms(total), scale

    # One compiled evaluation certifies every sample.
    norms, scales = eqx.filter_jit(jax.vmap(sample))(
        dae.times, dae.states, dae.state_rates
    )
    accepted = jnp.all(jnp.isfinite(norms) & (norms <= tolerance * scales), axis=-1)
    return TransientCertificate(
        owners=transient.row_space.names,
        residual_norms=norms,
        scales=scales,
        accepted=accepted & dae.valid,
    )


@final
class CoupledTransientSolution(StrictModule):
    """Native DAE solution of a coupled transient plus its original-row certificate.

    ``dae`` carries every native status, step, attempt, regularity, and
    continuation record; ``certificate`` evaluates the original coupled rows at
    the samples. ``accepted`` requires native success and every certified
    sample. Derivatives of the trajectory through the runtime ``parameters``
    are the native DAE's discrete implicit derivatives on its (fixed or frozen
    accepted) grid, ``dae.differentiation_mode``; ``derivative_valid`` admits
    them only at an accepted solution, and derivatives of a refused solution are
    NaN rather than a plausible sensitivity of an unaccepted trajectory.
    """

    dae: DifferentialAlgebraicSolution
    certificate: TransientCertificate
    tolerance: float = eqx.field(static=True)
    native_successful: Array
    accepted: Array
    derivative_valid: Array

    @property
    def times(self) -> Array:
        return self.dae.times

    @property
    def continuation(self) -> DAEContinuation:
        return self.dae.continuation


def solve_coupled_transient(
    transient: PreparedCoupledTransient,
    initial_state: NestedBlocks,
    time_grid: TimeGrid,
    /,
    *,
    parameters: object = None,
    policy: DAESolvePolicy | None = None,
    continuation: DAEContinuation | None = None,
    tolerance: float = 1.0e-6,
) -> CoupledTransientSolution:
    """Integrate a prepared coupled transient natively and certify its rows.

    ``continuation`` resumes from the accepted history of a previous adaptive
    segment (``initial_state`` then only fixes the problem signature).
    """
    if not isinstance(transient, PreparedCoupledTransient):
        raise TypeError("transient must be a PreparedCoupledTransient.")
    tolerance_ = positive_finite_float(tolerance, "tolerance")
    problem = transient.problem(initial_state, parameters=parameters)
    dae = solve_dae(problem, time_grid, policy=policy, continuation=continuation)
    certificate = _certificate(transient, dae, parameters, tolerance_)
    native = jnp.asarray(dae.successful)
    accepted = native & jnp.all(certificate.accepted)
    return CoupledTransientSolution(
        dae=guard_derivative_validity(
            dae,
            accepted,
            dependencies=parameters,
            failure="status",
            message="Coupled transient derivatives require an accepted solution.",
        ),
        certificate=certificate,
        tolerance=tolerance_,
        native_successful=native,
        accepted=accepted,
        derivative_valid=accepted,
    )


__all__ = [
    "CoupledTransientSolution",
    "PreparedCoupledTransient",
    "prepare_coupled_transient",
    "solve_coupled_transient",
    "TransientArguments",
    "TransientCertificate",
    "TransientField",
    "TransientScale",
]

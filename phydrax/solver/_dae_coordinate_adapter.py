#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Named block coordinates for structurally reduced differential-algebraic systems.

The native DAE runtime owns scaled ``ArraySpace`` coordinates for every implicit
stage, consistency root, and event root. This module identifies those coordinates
with named variable and equation blocks of a ``ReducedDAECompilation`` without
replacing the native spaces, scales, roles, or runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, assert_never, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..dynamics import (
    AbstractInputPolicy,
    DAERole,
    DifferentialAlgebraicSystem,
    ReducedDAECompilation,
)
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    assemble_block_operator,
    BlockLinearOperator,
    BlockPath,
    BlockProlongationLinearOperator,
    BlockRestrictionLinearOperator,
    BlockSelection,
    BlockSpace,
    CoordinateBlock,
    DenseLinearOperator,
    DiagonalLinearOperator,
    MappedBlockLinearOperator,
    ScaledLinearOperator,
    select_block_operator,
    SumLinearOperator,
)
from ..linalg._named_blocks import _coordinate_identity
from ..typing import parse
from ._dae_events import _DAEEventRootArguments
from ._dae_initialization import (
    _DAEInitializationArguments,
    _fixed_masks,
    DAEInitializationSpec,
)
from ._implicit_stage import ImplicitStageArguments


DAERootKind: TypeAlias = Literal["stage", "initialization", "event"]
DAESetupHook: TypeAlias = Literal[
    "stage", "initialization", "event", "tangent", "adjoint"
]
DAEScaleKind: TypeAlias = Literal["state", "rate", "residual"]

_ALL_HOOKS: tuple[DAESetupHook, ...] = (
    "stage",
    "initialization",
    "event",
    "tangent",
    "adjoint",
)


class DAEBlockJacobian(StrictModule):
    """Physical named-block partial derivatives of the reduced residual rows.

    ``state`` is ``dF/dstate`` and ``rate`` is ``dF/dstate_rate``; both act from the
    adapter ``variable_space`` to its ``equation_space`` and are unscaled. The
    adapter applies the native residual scale exactly once.
    """

    state: AbstractLinearOperator
    rate: AbstractLinearOperator

    def __init__(
        self, state: AbstractLinearOperator, rate: AbstractLinearOperator, /
    ) -> None:
        if not isinstance(state, AbstractLinearOperator) or not isinstance(
            rate, AbstractLinearOperator
        ):
            raise TypeError("DAE block Jacobians must be AbstractLinearOperator values.")
        if not state.source.compatible(rate.source) or not state.target.compatible(
            rate.target
        ):
            raise ValueError("State and rate Jacobians must share their block spaces.")
        self.state = state
        self.rate = rate


type AutonomousDAEBlockLinearization = Callable[
    [Array, PyTree[Array], PyTree[Array], Any], DAEBlockJacobian
]
type InputDAEBlockLinearization = Callable[
    [Array, PyTree[Array], PyTree[Array], Array, Any], DAEBlockJacobian
]
type DAEBlockLinearization = AutonomousDAEBlockLinearization | InputDAEBlockLinearization


class DAEBlockCoordinate(StrictModule):
    """One named native coordinate block with its differential-algebraic role."""

    path: BlockPath = eqx.field(static=True)
    role: DAERole = eqx.field(static=True)
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)


class DAERootCoordinates(StrictModule):
    """Exact named identification of one native DAE root's coordinates.

    ``column_map`` identifies the native unknown space and ``row_map`` the native
    residual space with named root blocks. Both are bijective and identity-ordered.
    """

    kind: DAERootKind = eqx.field(static=True)
    column_map: BlockSelection
    row_map: BlockSelection


def _identification(native: ArraySpace, view: BlockSpace, /) -> BlockSelection:
    if not isinstance(native, ArraySpace):
        raise TypeError("Native DAE root spaces must be ArraySpace values.")
    if native.size != view.size:
        raise ValueError(
            f"Native DAE root size {native.size} does not match the named view "
            f"size {view.size}."
        )
    members = []
    offset = 0
    for name, space in zip(view.names, view.spaces, strict=True):
        members.append((name, CoordinateBlock(space, ((offset, offset + space.size),))))
        offset += space.size
    return BlockSelection(native, members)


def _root_kind(arguments: object, /) -> DAERootKind:
    if isinstance(arguments, ImplicitStageArguments):
        return "stage"
    if isinstance(arguments, _DAEInitializationArguments):
        return "initialization"
    if isinstance(arguments, _DAEEventRootArguments):
        return "event"
    raise TypeError("DAE setup arguments do not belong to a native DAE root.")


def _combined(
    state: AbstractLinearOperator | None,
    rate: AbstractLinearOperator | None,
    shift: Array,
    /,
) -> AbstractLinearOperator | None:
    """Return ``state + shift * rate`` preserving explicit block grids."""
    grids = tuple(
        value for value in (state, rate) if isinstance(value, BlockLinearOperator)
    )
    others = tuple(
        value
        for value in (state, rate)
        if value is not None and not isinstance(value, BlockLinearOperator)
    )
    if grids and not others:
        grid = grids[0]
        blocks = tuple(
            tuple(
                _combined(
                    None
                    if state is None
                    else cast(BlockLinearOperator, state).blocks[i][j],
                    None
                    if rate is None
                    else cast(BlockLinearOperator, rate).blocks[i][j],
                    shift,
                )
                for j in range(len(grid.source.spaces))
            )
            for i in range(len(grid.target.spaces))
        )
        if all(block is None for row in blocks for block in row):
            return None
        return BlockLinearOperator(blocks, source=grid.source, target=grid.target)
    terms = [value for value in (state,) if value is not None]
    if rate is not None:
        terms.append(ScaledLinearOperator(rate, shift))
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else SumLinearOperator(terms[0], terms[1])


def _row_scaled(
    operator: AbstractLinearOperator | None, reciprocal: Array, /
) -> AbstractLinearOperator | None:
    """Apply ``diag(1 / residual_scale)`` to rows, preserving explicit grids."""
    if operator is None:
        return None
    if isinstance(operator, BlockLinearOperator):
        offsets = [0]
        for space in operator.target.spaces:
            offsets.append(offsets[-1] + space.size)
        blocks = tuple(
            tuple(
                _row_scaled(block, reciprocal[offsets[i] : offsets[i + 1]])
                for block in row
            )
            for i, row in enumerate(operator.blocks)
        )
        return BlockLinearOperator(blocks, source=operator.source, target=operator.target)
    return DiagonalLinearOperator(reciprocal, space=operator.target) @ operator


def _leaf_space(shape: tuple[int, ...], dtype: np.dtype, space_id: str, /) -> ArraySpace:
    return ArraySpace(shape, dtype=dtype, space_id=space_id)


class DAECoordinateAdapter(StrictModule):
    """Named block view of the native coordinates of one reduced DAE.

    The adapter admits only a ``ReducedDAECompilation``: its structural analysis
    succeeded within the declared differentiation and tearing capacities, and its
    lowered system is the native index-one runtime system. An unreduced higher-index
    declaration therefore never reaches this adapter; ``compile_acausal_dae`` refuses
    it with its analysis status. Structural admission is a hypothesis about the
    declared graph; local regularity evidence remains with the DAE solver.

    ``variable_space`` names each variable (a nested block of derivative segments for
    variables of order two or more). ``equation_space`` names each reduced equation
    and ``row_space`` groups ``"equations"`` and, when present, the reduction's
    ``"kinematics"`` rows. Native state, rate, and residual scales are exposed as
    named views and are never rescaled.
    """

    compilation: ReducedDAECompilation
    input_policy: AbstractInputPolicy | None
    initialization: DAEInitializationSpec
    variable_space: BlockSpace
    equation_space: BlockSpace
    kinematics_space: BlockSpace | None
    row_space: BlockSpace
    variables: tuple[DAEBlockCoordinate, ...]
    rows: tuple[DAEBlockCoordinate, ...]
    free_state_indices: Array
    free_rate_indices: Array
    free_state_paths: tuple[BlockPath, ...] = eqx.field(static=True)
    free_rate_paths: tuple[BlockPath, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        compilation: ReducedDAECompilation,
        /,
        *,
        input_policy: AbstractInputPolicy | None = None,
        initialization: DAEInitializationSpec | Literal["structural"] = "structural",
        dtype: DTypeLike = np.float64,
    ) -> None:
        system = _admitted_system(compilation)
        policy = _validated_input_policy(system, input_policy)
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not np.issubdtype(dtype_, np.inexact):
            raise TypeError("DAE coordinate dtype must be real or complex inexact.")
        spec = (
            DAEInitializationSpec.from_masks(
                compilation.fixed_state_mask, compilation.fixed_rate_mask
            )
            if initialization == "structural"
            else initialization
        )
        if not isinstance(spec, DAEInitializationSpec):
            raise TypeError(
                "initialization must be a DAEInitializationSpec or 'structural'."
            )
        base = canonical_fingerprint(
            {
                "kind": "dae-block-coordinates",
                "compilation": compilation.compilation_id,
                "dtype": dtype_.str,
            }
        )
        variable_space, variables = _variable_layout(compilation, dtype_, base)
        equation_space, kinematics_space, row_space, rows = _row_layout(
            compilation, variables, dtype_, base
        )
        _verify_roles(system, variables, rows)
        free = _free_coordinates(system, spec, variables)
        self.compilation = compilation
        self.input_policy = policy
        self.initialization = spec
        self.variable_space = variable_space
        self.equation_space = equation_space
        self.kinematics_space = kinematics_space
        self.row_space = row_space
        self.variables = variables
        self.rows = rows
        self.free_state_paths = free[0]
        self.free_rate_paths = free[1]
        self.free_state_indices = jnp.asarray(free[2], dtype=jnp.int32)
        self.free_rate_indices = jnp.asarray(free[3], dtype=jnp.int32)
        self.dtype = dtype_
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "dae-coordinate-adapter",
                "coordinates": base,
                "initialization": spec.initialization_id,
                "input_policy": None if policy is None else policy.policy_id,
            }
        )

    @property
    def system(self) -> DifferentialAlgebraicSystem:
        return self.compilation.system

    def state_view(self, value: Array, /) -> tuple[PyTree[Array], ...]:
        """Return the named variable blocks of one native state, rate, or increment."""
        return self.variable_space.unflatten(jnp.asarray(value))

    def native_state(self, blocks: PyTree[Array], /) -> Array:
        """Return the native coordinates of one named variable vector."""
        return self.variable_space.flatten(blocks)

    def row_view(self, value: Array, /) -> tuple[PyTree[Array], ...]:
        """Return the named row blocks of one native residual."""
        return self.row_space.unflatten(jnp.asarray(value))

    def native_rows(self, blocks: PyTree[Array], /) -> Array:
        """Return the native coordinates of one named row vector."""
        return self.row_space.flatten(blocks)

    def scale_view(self, kind: DAEScaleKind, /) -> tuple[PyTree[Array], ...]:
        """Return the native state, rate, or residual scale as named blocks."""
        kind_ = parse(kind, DAEScaleKind, "kind")
        system = self.system
        match kind_:
            case "state":
                return self.state_view(system.state_scale.astype(self.dtype))
            case "rate":
                return self.state_view(system.state_rate_scale.astype(self.dtype))
            case "residual":
                return self.row_view(system.residual_scale.astype(self.dtype))
            case _:
                assert_never(kind_)

    def root_columns(self, kind: DAERootKind, /) -> BlockSpace:
        """Return the named view of one root kind's native unknowns."""
        kind_ = parse(kind, DAERootKind, "kind")
        match kind_:
            case "stage":
                return BlockSpace((self.variable_space,), names=("increment",))
            case "initialization":
                return self._initialization_columns()
            case "event":
                return BlockSpace(
                    (self.variable_space, self._event_space("time")),
                    names=("increment", "time"),
                )
            case _:
                assert_never(kind_)

    def root_rows(self, kind: DAERootKind, /) -> BlockSpace:
        """Return the named view of one root kind's native residual rows."""
        kind_ = parse(kind, DAERootKind, "kind")
        match kind_:
            case "stage" | "initialization":
                return self.row_space
            case "event":
                return BlockSpace(
                    self.row_space.spaces + (self._event_space("guard"),),
                    names=self.row_space.names + ("guard",),
                )
            case _:
                assert_never(kind_)

    def root_coordinates(
        self,
        kind: DAERootKind,
        unknown_space: ArraySpace,
        residual_space: ArraySpace,
        /,
    ) -> DAERootCoordinates:
        """Identify supplied native root spaces with the named root views."""
        kind_ = parse(kind, DAERootKind, "kind")
        return DAERootCoordinates(
            kind_,
            _identification(unknown_space, self.root_columns(kind_)),
            _identification(residual_space, self.root_rows(kind_)),
        )

    def bind_linearization(
        self,
        linearization: DAEBlockLinearization,
        /,
        *,
        hooks: Sequence[DAESetupHook] = _ALL_HOOKS,
    ) -> ReducedDAECompilation:
        """Return the compilation whose system carries named-block setup hooks.

        ``linearization(time, state, rate, args)`` (with ``inputs`` before ``args``
        for input-aware systems) receives named physical blocks evaluated at the
        root's ``physical_state`` and ``state_rate`` and returns a
        ``DAEBlockJacobian``. Each selected hook maps it exactly to its native root:
        stage ``F_x + shift F_xdot``, free state/rate columns for initialization, the
        bordered time column and guard row for events, the same forward maps for
        tangent solves, and their coordinate transposes for adjoint solves.
        """
        if not callable(linearization):
            raise TypeError("linearization must be callable.")
        selected = tuple(parse(hook, DAESetupHook, "hooks") for hook in hooks)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("hooks must name one or more distinct setup hooks.")
        setups = {hook: _DAEBlockSetup(self, linearization, hook) for hook in selected}
        system = self.system
        bound = DifferentialAlgebraicSystem(
            system.residual,
            state_shape=system.state_shape,
            structure=system.structure,
            input_layout=system.input_layout,
            state_scale=system.state_scale,
            state_rate_scale=system.state_rate_scale,
            residual_scale=system.residual_scale,
            state_geometry=system.state_geometry,
            trial_validity=system.trial_validity,
            trial_validity_id=system.trial_validity_id,
            stage_linear_setup=setups.get("stage"),
            initialization_linear_setup=setups.get("initialization"),
            event_linear_setup=setups.get("event"),
            tangent_linear_setup=setups.get("tangent"),
            adjoint_linear_setup=setups.get("adjoint"),
            system_id=system.system_id,
        )
        compilation = self.compilation
        return ReducedDAECompilation(
            bound,
            compilation.reconstruction,
            compilation.residual_audit,
            compilation.structure,
            compilation.fixed_state_mask,
            compilation.fixed_rate_mask,
            compilation.analysis,
            compilation.compilation_id,
        )

    def role_groups(
        self, kind: DAERootKind, /
    ) -> tuple[tuple[str, tuple[BlockPath, ...], tuple[BlockPath, ...]], ...]:
        """Pair differential and algebraic rows with their root columns.

        Stage and event roots pair rows with increment blocks of the same role; the
        event root adds its guard row and time column. Initialization pairs
        differential rows with free rates and algebraic rows with free states.
        """
        kind_ = parse(kind, DAERootKind, "kind")
        rows = {
            role: tuple(row.path for row in self.rows if row.role == role)
            for role in ("differential", "algebraic")
        }
        match kind_:
            case "stage" | "event":
                columns = {
                    role: tuple(
                        ("increment",) + variable.path
                        for variable in self.variables
                        if variable.role == role
                    )
                    for role in ("differential", "algebraic")
                }
            case "initialization":
                columns = {
                    "differential": tuple(
                        ("rate", "/".join(path)) for path in self.free_rate_paths
                    ),
                    "algebraic": tuple(
                        ("state", "/".join(path)) for path in self.free_state_paths
                    ),
                }
            case _:
                assert_never(kind_)
        groups = tuple(
            (role, rows[role], columns[role])
            for role in ("differential", "algebraic")
            if rows[role] or columns[role]
        )
        if kind_ == "event":
            groups = groups + (("event", (("guard",),), (("time",),)),)
        return groups

    def correction_transfers(
        self,
        kind: DAERootKind,
        groups: Sequence[tuple[str, Sequence[BlockPath], Sequence[BlockPath]]],
        /,
    ) -> tuple[BlockRestrictionLinearOperator, BlockProlongationLinearOperator]:
        """Return exact subspace-correction transfers in canonical root coordinates.

        Each group gathers its named residual rows into one local coordinate block
        and scatters that block's correction into its named root columns. The
        transfers act on the canonical Euclidean coordinates that the native Newton
        solve gives its preconditioners, so a single ``SubspaceCorrectionTerm`` with
        a block-factorization local solver preconditions the native root with the
        named block structure.
        """
        kind_ = parse(kind, DAERootKind, "kind")
        entries = tuple(groups)
        if not entries:
            raise ValueError("Correction transfers require at least one group.")
        columns = self.root_columns(kind_)
        rows = self.root_rows(kind_)
        canonical = ArraySpace((columns.size,), dtype=self.dtype)
        row_members = []
        column_members = []
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise TypeError("Correction groups must be (name, rows, columns).")
            name, row_paths, column_paths = entry
            row_ranges = BlockSelection(rows, ((name, tuple(row_paths)),)).member_ranges[
                0
            ]
            column_ranges = BlockSelection(
                columns, ((name, tuple(column_paths)),)
            ).member_ranges[0]
            size = sum(stop - start for start, stop in row_ranges)
            if size != sum(stop - start for start, stop in column_ranges):
                raise ValueError(
                    f"Correction group {name!r} pairs {size} rows with a different "
                    "number of columns."
                )
            local = ArraySpace(
                (size,),
                dtype=self.dtype,
                space_id=f"{self.adapter_id}:{kind_}:correction:{name}",
            )
            row_members.append((name, CoordinateBlock(local, row_ranges)))
            column_members.append((name, CoordinateBlock(local, column_ranges)))
        return (
            BlockRestrictionLinearOperator(BlockSelection(canonical, row_members)),
            BlockProlongationLinearOperator(BlockSelection(canonical, column_members)),
        )

    def _event_space(self, name: Literal["time", "guard"], /) -> ArraySpace:
        return ArraySpace(
            (1,), dtype=self.dtype, space_id=f"{self.adapter_id}:event-{name}"
        )

    def _free_groups(
        self,
    ) -> tuple[tuple[Literal["state", "rate"], tuple[BlockPath, ...]], ...]:
        groups = tuple(
            (name, paths)
            for name, paths in (
                ("state", self.free_state_paths),
                ("rate", self.free_rate_paths),
            )
            if paths
        )
        if not groups:
            raise ValueError("The initialization contract has no free unknowns.")
        return groups

    def _initialization_columns(self) -> BlockSpace:
        # State and rate columns are distinct unknowns even when one variable
        # frees both, so each group selects from its own copy of variable_space.
        groups = self._free_groups()
        return BlockSpace(
            tuple(
                BlockSelection(self.variable_space, (group,)).target.spaces[0]
                for group in groups
            ),
            names=tuple(name for name, _ in groups),
        )

    def _physical_jacobian(
        self,
        linearization: DAEBlockLinearization,
        time: Array,
        state: Array,
        rate: Array,
        model_args: Any,
        /,
    ) -> DAEBlockJacobian:
        state_blocks = self.state_view(state.reshape((-1,)))
        rate_blocks = self.state_view(rate.reshape((-1,)))
        if self.input_policy is None:
            autonomous = cast(AutonomousDAEBlockLinearization, linearization)
            jacobian = autonomous(time, state_blocks, rate_blocks, model_args)
        else:
            inputs = self.input_policy.evaluate(time, state, model_args)
            forced = cast(InputDAEBlockLinearization, linearization)
            jacobian = forced(time, state_blocks, rate_blocks, inputs, model_args)
        if not isinstance(jacobian, DAEBlockJacobian):
            raise TypeError("A DAE block linearization must return DAEBlockJacobian.")
        if not jacobian.state.source.compatible(self.variable_space):
            raise ValueError("DAE block Jacobians must act on the variable_space.")
        if not jacobian.state.target.compatible(self.equation_space):
            raise ValueError("DAE block Jacobians must map into the equation_space.")
        return jacobian

    def _kinematic_jacobian(
        self,
    ) -> tuple[AbstractLinearOperator, AbstractLinearOperator] | None:
        """Return exact ``(d/dstate, d/drate)`` of ``rate[k] - state[k + 1]`` rows."""
        space = self.kinematics_space
        if space is None:
            return None
        negative = jnp.asarray(-1.0, dtype=np.finfo(self.dtype).dtype)
        state_entries = []
        rate_entries = []
        for name, segments in zip(space.names, space.spaces, strict=True):
            if not isinstance(segments, BlockSpace):
                raise RuntimeError("Kinematic rows must be nested per derivative order.")
            variable = self.variable_space.spaces[self.variable_space.names.index(name)]
            if not isinstance(variable, BlockSpace):
                raise RuntimeError(
                    "Kinematic variables must be nested per derivative order."
                )
            for order, row in zip(segments.names, segments.spaces, strict=True):
                current = variable.spaces[variable.names.index(order)]
                following = variable.spaces[variable.names.index(str(int(order) + 1))]
                rate_entries.append(
                    ((name, order), (name, order), _coordinate_identity(current, row))
                )
                state_entries.append(
                    (
                        (name, order),
                        (name, str(int(order) + 1)),
                        ScaledLinearOperator(
                            _coordinate_identity(following, row), negative
                        ),
                    )
                )
        return (
            assemble_block_operator(
                state_entries, source=self.variable_space, target=space
            ),
            assemble_block_operator(
                rate_entries, source=self.variable_space, target=space
            ),
        )

    def _reciprocal_rows(self) -> tuple[Array, Array | None]:
        reciprocal = 1.0 / self.system.residual_scale.astype(self.dtype)
        count = self.equation_space.size
        if self.kinematics_space is None:
            return reciprocal, None
        return reciprocal[:count], reciprocal[count:]

    def _increment_rows(
        self, jacobian: DAEBlockJacobian, shift: Array, /
    ) -> list[list[AbstractLinearOperator | None]]:
        equations, kinematics = self._reciprocal_rows()
        rows = [[_row_scaled(_combined(jacobian.state, jacobian.rate, shift), equations)]]
        kinematic = self._kinematic_jacobian()
        if kinematic is not None and kinematics is not None:
            rows.append(
                [_row_scaled(_combined(kinematic[0], kinematic[1], shift), kinematics)]
            )
        return rows

    def _stage_block(
        self,
        linearization: DAEBlockLinearization,
        unknown: Array,
        arguments: ImplicitStageArguments,
        /,
    ) -> BlockLinearOperator:
        jacobian = self._physical_jacobian(
            linearization,
            arguments.time,
            arguments.physical_state(unknown),
            arguments.state_rate(unknown),
            arguments.model_args,
        )
        return BlockLinearOperator(
            self._increment_rows(jacobian, arguments.shift.astype(self.dtype)),
            source=self.root_columns("stage"),
            target=self.row_space,
        )

    def _checked_initialization_unknown(
        self, unknown: Array, arguments: _DAEInitializationArguments, /
    ) -> Array:
        if (
            arguments.state_indices.shape != self.free_state_indices.shape
            or arguments.rate_indices.shape != self.free_rate_indices.shape
        ):
            raise ValueError(
                "The native initialization free coordinates differ from the adapter "
                "DAEInitializationSpec."
            )
        mismatch = jnp.any(arguments.state_indices != self.free_state_indices) | jnp.any(
            arguments.rate_indices != self.free_rate_indices
        )
        return eqx.error_if(
            unknown,
            mismatch,
            "The native initialization free coordinates differ from the adapter "
            "DAEInitializationSpec.",
        )

    def _initialization_block(
        self,
        linearization: DAEBlockLinearization,
        unknown: Array,
        arguments: _DAEInitializationArguments,
        /,
    ) -> BlockLinearOperator:
        checked = self._checked_initialization_unknown(unknown, arguments)
        jacobian = self._physical_jacobian(
            linearization,
            arguments.time,
            arguments.physical_state(checked),
            arguments.state_rate(checked),
            arguments.model_args,
        )
        columns = self._initialization_columns()
        equations, kinematics = self._reciprocal_rows()
        sources = [(jacobian.state, jacobian.rate, equations, self.equation_space)]
        kinematic = self._kinematic_jacobian()
        if (
            kinematic is not None
            and kinematics is not None
            and self.kinematics_space is not None
        ):
            sources.append(
                (kinematic[0], kinematic[1], kinematics, self.kinematics_space)
            )
        rows = []
        for state_operator, rate_operator, reciprocal, space in sources:
            everything = BlockSelection(
                space, (("rows", tuple((name,) for name in space.names)),)
            )
            operators = {"state": state_operator, "rate": rate_operator}
            row = []
            for name, paths in self._free_groups():
                group = BlockSelection(self.variable_space, ((name, paths),))
                block = select_block_operator(operators[name], everything, group)
                row.append(_row_scaled(block.blocks[0][0], reciprocal))
            rows.append(row)
        return BlockLinearOperator(rows, source=columns, target=self.row_space)

    def _event_block(
        self,
        linearization: DAEBlockLinearization,
        unknown: Array,
        arguments: _DAEEventRootArguments,
        /,
    ) -> BlockLinearOperator:
        time = unknown[-1]
        jacobian = self._physical_jacobian(
            linearization,
            time,
            arguments.physical_state(unknown),
            arguments.state_rate(unknown),
            arguments.model_args,
        )
        rows = self._increment_rows(
            jacobian, arguments.rate_shift(unknown).astype(self.dtype)
        )
        time_column = self._event_time_column(unknown, arguments)
        guard_gradient = self._event_guard_gradient(unknown, arguments)
        time_space = self._event_space("time")
        guard_space = self._event_space("guard")
        offset = 0
        for row, space in zip(rows, self.row_space.spaces, strict=True):
            row.append(
                DenseLinearOperator(
                    time_column[offset : offset + space.size, None],
                    source=time_space,
                    target=space,
                )
            )
            offset += space.size
        rows.append(
            [
                DenseLinearOperator(
                    guard_gradient[None, :-1],
                    source=self.variable_space,
                    target=guard_space,
                ),
                DenseLinearOperator(
                    guard_gradient[None, -1:], source=time_space, target=guard_space
                ),
            ]
        )
        return BlockLinearOperator(
            rows, source=self.root_columns("event"), target=self.root_rows("event")
        )

    def _event_time_column(
        self, unknown: Array, arguments: _DAEEventRootArguments, /
    ) -> Array:
        """Return the exact time derivative of the scaled event residual rows."""
        system = self.system
        policy = self.input_policy

        def rows(augmented: Array) -> Array:
            time = augmented[-1]
            state = arguments.physical_state(augmented)
            inputs = (
                None
                if policy is None
                else policy.evaluate(time, state, arguments.model_args)
            )
            return system.scaled_residual(
                time,
                state,
                arguments.state_rate(augmented),
                arguments.model_args,
                inputs=inputs,
            ).reshape((-1,))

        direction = jnp.zeros_like(unknown).at[-1].set(1.0)
        return jax.jvp(rows, (unknown,), (direction,))[1].astype(self.dtype)

    def _event_guard_gradient(
        self, unknown: Array, arguments: _DAEEventRootArguments, /
    ) -> Array:
        """Return the exact gradient of the scaled guard row in root coordinates."""

        def guard(augmented: Array) -> Array:
            value = arguments.guard(
                augmented[-1],
                arguments.physical_state(augmented),
                arguments.model_args,
            )
            scaled = jnp.asarray(value, dtype=augmented.dtype).reshape(())
            return scaled / arguments.guard_scale

        return jax.grad(guard)(unknown).astype(self.dtype)

    def _root_setup(
        self,
        linearization: DAEBlockLinearization,
        kind: DAERootKind,
        unknown: Array,
        arguments: object,
        unknown_space: ArraySpace,
        residual_space: ArraySpace,
        /,
    ) -> MappedBlockLinearOperator:
        coordinates = self.root_coordinates(kind, unknown_space, residual_space)
        match kind:
            case "stage":
                if not isinstance(arguments, ImplicitStageArguments):
                    raise TypeError("Stage setup requires implicit-stage arguments.")
                block = self._stage_block(linearization, unknown, arguments)
            case "initialization":
                if not isinstance(arguments, _DAEInitializationArguments):
                    raise TypeError(
                        "Initialization setup requires consistency arguments."
                    )
                block = self._initialization_block(linearization, unknown, arguments)
            case "event":
                if not isinstance(arguments, _DAEEventRootArguments):
                    raise TypeError("Event setup requires event-root arguments.")
                block = self._event_block(linearization, unknown, arguments)
            case _:
                assert_never(kind)
        return MappedBlockLinearOperator(
            block, row_map=coordinates.row_map, column_map=coordinates.column_map
        )


class _DAEBlockSetup(StrictModule):
    """Native DAE setup hook backed by one named physical block linearization."""

    adapter: DAECoordinateAdapter
    linearization: DAEBlockLinearization
    hook: DAESetupHook = eqx.field(static=True)

    def __call__(
        self,
        unknown: Array,
        arguments: object,
        source: ArraySpace,
        target: ArraySpace,
        /,
    ) -> MappedBlockLinearOperator:
        kind = _root_kind(arguments)
        match self.hook:
            case "stage" | "initialization" | "event":
                if kind != self.hook:
                    raise TypeError(
                        f"The {self.hook} setup hook received {kind} root arguments."
                    )
                return self.adapter._root_setup(
                    self.linearization, kind, unknown, arguments, source, target
                )
            case "tangent":
                return self.adapter._root_setup(
                    self.linearization, kind, unknown, arguments, source, target
                )
            case "adjoint":
                # Adjoint factories receive (residual, unknown) spaces and return the
                # coordinate transpose of the forward root setup.
                return self.adapter._root_setup(
                    self.linearization, kind, unknown, arguments, target, source
                ).coordinate_transpose()
            case _:
                assert_never(self.hook)


def _admitted_system(compilation: object, /) -> DifferentialAlgebraicSystem:
    if not isinstance(compilation, ReducedDAECompilation):
        raise TypeError(
            "Named DAE coordinates admit only a ReducedDAECompilation from the "
            "structural reduction owner."
        )
    analysis = compilation.analysis
    if not analysis.successful or analysis.selected_tears:
        raise ValueError(
            "The DAE compilation does not carry a successful untorn structural "
            f"analysis; status {analysis.status!r}."
        )
    system = compilation.system
    if system.system_id != f"reduced-dae:{analysis.analysis_id}":
        raise ValueError("The DAE compilation system does not match its analysis.")
    return system


def _validated_input_policy(
    system: DifferentialAlgebraicSystem, policy: AbstractInputPolicy | None, /
) -> AbstractInputPolicy | None:
    if system.input_layout is None:
        if policy is not None:
            raise ValueError("An autonomous DAE does not accept an input_policy.")
        return None
    if not isinstance(policy, AbstractInputPolicy):
        raise TypeError("An input-aware DAE requires an AbstractInputPolicy.")
    if policy.input_layout.layout_id != system.input_layout.layout_id:
        raise ValueError("input_policy must match the DAE input layout exactly.")
    return policy


def _variable_layout(
    compilation: ReducedDAECompilation, dtype: np.dtype, base: str, /
) -> tuple[BlockSpace, tuple[DAEBlockCoordinate, ...]]:
    reconstruction = compilation.reconstruction
    members = []
    coordinates = []
    for variable, segments in zip(
        reconstruction.variables, reconstruction.offsets, strict=True
    ):
        role: DAERole = (
            "differential" if variable.maximum_derivative_order > 0 else "algebraic"
        )
        shape = tuple(variable.shape)
        prefix = f"{base}:variable:{variable.name}"
        if len(segments) == 1:
            members.append(_leaf_space(shape, dtype, prefix))
            coordinates.append(DAEBlockCoordinate((variable.name,), role, *segments[0]))
            continue
        names = tuple(str(order) for order in range(len(segments)))
        members.append(
            BlockSpace(
                tuple(_leaf_space(shape, dtype, f"{prefix}:{name}") for name in names),
                names=names,
            )
        )
        coordinates.extend(
            DAEBlockCoordinate((variable.name, name), role, start, stop)
            for name, (start, stop) in zip(names, segments, strict=True)
        )
    space = BlockSpace(
        tuple(members),
        names=tuple(variable.name for variable in reconstruction.variables),
    )
    return space, tuple(coordinates)


def _row_layout(
    compilation: ReducedDAECompilation,
    variables: tuple[DAEBlockCoordinate, ...],
    dtype: np.dtype,
    base: str,
    /,
) -> tuple[BlockSpace, BlockSpace | None, BlockSpace, tuple[DAEBlockCoordinate, ...]]:
    analysis = compilation.analysis
    matching = dict(analysis.matching)
    execution = {value.name: value for value in compilation.reconstruction.variables}
    equations = []
    coordinates = []
    cursor = 0
    for name in analysis.equation_names:
        matched = execution[matching[name]]
        shape = tuple(matched.shape)
        role: DAERole = (
            "differential" if matched.maximum_derivative_order > 0 else "algebraic"
        )
        equations.append(_leaf_space(shape, dtype, f"{base}:equation:{name}"))
        size = matched.size
        coordinates.append(
            DAEBlockCoordinate(("equations", name), role, cursor, cursor + size)
        )
        cursor += size
    equation_space = BlockSpace(tuple(equations), names=analysis.equation_names)
    kinematic_members = []
    kinematic_names = []
    for variable in compilation.reconstruction.variables:
        count = max(variable.maximum_derivative_order - 1, 0)
        if not count:
            continue
        names = tuple(str(order) for order in range(count))
        kinematic_members.append(
            BlockSpace(
                tuple(
                    _leaf_space(
                        tuple(variable.shape),
                        dtype,
                        f"{base}:kinematics:{variable.name}:{name}",
                    )
                    for name in names
                ),
                names=names,
            )
        )
        kinematic_names.append(variable.name)
        for name in names:
            coordinates.append(
                DAEBlockCoordinate(
                    ("kinematics", variable.name, name),
                    "differential",
                    cursor,
                    cursor + variable.size,
                )
            )
            cursor += variable.size
    if cursor != sum(value.stop - value.start for value in variables):
        raise ValueError("Reduced DAE rows and variables must have equal size.")
    kinematics_space = (
        BlockSpace(tuple(kinematic_members), names=tuple(kinematic_names))
        if kinematic_members
        else None
    )
    row_space = BlockSpace(
        (equation_space,) + (() if kinematics_space is None else (kinematics_space,)),
        names=("equations",) + (() if kinematics_space is None else ("kinematics",)),
    )
    return equation_space, kinematics_space, row_space, tuple(coordinates)


def _expanded_roles(
    coordinates: tuple[DAEBlockCoordinate, ...], /
) -> tuple[DAERole, ...]:
    return tuple(
        role
        for value in coordinates
        for role in (value.role,) * (value.stop - value.start)
    )


def _verify_roles(
    system: DifferentialAlgebraicSystem,
    variables: tuple[DAEBlockCoordinate, ...],
    rows: tuple[DAEBlockCoordinate, ...],
    /,
) -> None:
    size = system.state_size
    if system.state_shape != (size,):
        raise ValueError("A reduced DAE must use one flat native state axis.")
    structure = system.structure
    if (
        len(structure.variable_roles) != size
        or _expanded_roles(variables) != structure.variable_roles
        or _expanded_roles(rows) != structure.equation_roles
    ):
        raise ValueError(
            "Named DAE block roles do not match the native variable and equation roles."
        )


def _free_coordinates(
    system: DifferentialAlgebraicSystem,
    spec: DAEInitializationSpec,
    variables: tuple[DAEBlockCoordinate, ...],
    /,
) -> tuple[tuple[BlockPath, ...], tuple[BlockPath, ...], np.ndarray, np.ndarray]:
    fixed_state, fixed_rate = _fixed_masks(system, spec)
    flat_state = np.asarray(fixed_state, dtype=np.bool_).reshape((-1,))
    flat_rate = np.asarray(fixed_rate, dtype=np.bool_).reshape((-1,))
    free_states = []
    free_rates = []
    for variable in variables:
        for mask, output in ((flat_state, free_states), (flat_rate, free_rates)):
            segment = mask[variable.start : variable.stop]
            if segment.any() and not segment.all():
                raise ValueError(
                    f"Initialization masks split named block {variable.path!r}; named "
                    "initialization coordinates require block-aligned masks."
                )
            if not segment.any():
                output.append(variable.path)
    return (
        tuple(free_states),
        tuple(free_rates),
        np.flatnonzero(~flat_state).astype(np.int32),
        np.flatnonzero(~flat_rate).astype(np.int32),
    )


__all__ = [
    "AutonomousDAEBlockLinearization",
    "DAEBlockCoordinate",
    "DAEBlockJacobian",
    "DAEBlockLinearization",
    "DAECoordinateAdapter",
    "DAERootCoordinates",
    "DAERootKind",
    "DAEScaleKind",
    "DAESetupHook",
    "InputDAEBlockLinearization",
]

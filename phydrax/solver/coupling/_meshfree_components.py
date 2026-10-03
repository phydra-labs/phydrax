#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native meshfree equations published as components of a coupled problem.

``MeshfreeComponent`` publishes a raw point-cloud, intrinsic-surface, or
conservative-exterior equation (one scalar field or a coupled block of fields)
without any facet claim. ``MeshfreeTraceComponent`` publishes a prepared
point-cloud elliptic or block system whose coupling boundary is authorized by
oriented geometry charts (``PointBoundaryCharts``): it owns side traces and
residual-reaction conormal (traction) fluxes through the canonical trace
contract. Both use the owner's native rows unchanged except for the explicit,
reported scaling of coupling flux rows by their physical boundary measure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...discretization import (
    BoundaryImposition,
    FacetTraceRule,
    IntegrationDomain,
    PreparedFieldReconstruction,
    PreparedFluxAction,
    PreparedTraceAction,
    SideTraceQuantity,
    TraceInverseEvidence,
)
from ...discretization._point_cloud import PreparedPointCloudDiscretization
from ...discretization._point_cloud_pde import (
    PointBoundaryCondition,
    PreparedPointCloudPoisson,
)
from ...discretization._point_cloud_view import PointCloudFieldReconstructionKernel
from ...discretization._side_actions import (
    AbstractSideFluxEvaluator,
    SideActionDescriptor,
)
from ...discretization._views import FieldTraceSide
from ...discretization.meshfree._boundary import (
    PointBoundaryCharts,
    PreparedPointGhostLayer,
)
from ...discretization.meshfree._exterior import PreparedMeshfreeExteriorCalculus
from ...discretization.meshfree._surface import (
    PreparedSurfacePointCloud,
    SurfaceFieldReconstructionKernel,
)
from ...discretization.meshfree._systems import PreparedPointBlockSystem
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    ConstraintMap,
    DiagonalLinearOperator,
    DualSpace,
    FunctionLinearOperator,
)
from ...sparse import SparseCoordinateOperator
from ...typing import checked, Dim, Float
from ._components import (
    AbstractCapacityComponent,
    AbstractPreparedCapacity,
    AbstractReconstructionComponent,
    AbstractTraceComponent,
    ComponentBlock,
    ComponentField,
    ComponentSpace,
)


type MeshfreeOwner = (
    PreparedPointCloudDiscretization
    | PreparedSurfacePointCloud
    | PreparedMeshfreeExteriorCalculus
)
type MeshfreeSystemOwner = PreparedPointCloudPoisson | PreparedPointBlockSystem
type MeshfreeReactionFunction = Callable[[Array, Array, object], Array]


class _MeshfreePointDim(Dim):
    """Full point-value coordinates."""


class _MeshfreeStateDim(Dim):
    """Native, possibly constrained state coordinates."""


class _MeshfreeNullityDim(Dim):
    """Declared homogeneous kernel columns."""


@final
class MeshfreeCapacity(AbstractPreparedCapacity, NonTrainableState):
    """Diagonal capacity ``coefficient * c_i`` of one meshfree field.

    ``c_i`` is the owner's capacity row measure: the physical quadrature measure
    of a raw equation, or the row measure of a system owner's bulk rows (zero on
    rows that its boundary conditions own, which stay algebraic).
    """

    __strict_contract__ = True
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    diagonal: Float[_MeshfreePointDim]
    full_space: ArraySpace

    def operator(self, args: object, /) -> AbstractLinearOperator:
        del args
        return FunctionLinearOperator(
            lambda value: self.diagonal * value,
            source=self.full_space,
            target=DualSpace(self.full_space),
            transpose_action=lambda value: self.diagonal * value,
        )


@final
class MeshfreeReaction(StrictModule):
    """Pointwise nonlinear term ``g(u, x, args)`` added to an owner's rows.

    ``function(values, points, args)`` maps the ``(points, fields)`` full field
    values and the owner's point coordinates to ``(points, fields)`` values. The
    component adds it with explicit row weights. A component with a reaction
    publishes no affine operator, so the coupled problem is solved by native
    Newton through the prepared linearization of its residual.
    """

    function: MeshfreeReactionFunction = eqx.field(static=True)
    reaction_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, function: MeshfreeReactionFunction, /, *, reaction_id: str
    ) -> None:
        self.function = function
        self.reaction_id = canonical_identifier(reaction_id, "reaction_id")

    def evaluate(self, values: Array, points: Array, args: object, /) -> Array:
        result = jnp.asarray(self.function(values, points, args))
        if result.shape != values.shape or not jnp.issubdtype(result.dtype, jnp.floating):
            raise ValueError(
                "A meshfree reaction must return real (points, fields) values."
            )
        return result


def _field_names(field: str | Sequence[str], /) -> tuple[str, ...]:
    names = (
        (canonical_identifier(field, "field"),)
        if isinstance(field, str)
        else tuple(canonical_identifier(name, "field") for name in field)
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("Component fields must be unique identifiers.")
    return names


def _field_spaces(
    operator: AbstractLinearOperator, count: int, /
) -> tuple[ArraySpace, ...]:
    """Full coefficient spaces of the operator's fields, in order."""
    source = operator.source
    members = source.spaces if isinstance(source, BlockSpace) else (source,)
    if len(members) != count:
        raise ValueError("Declare exactly one field per native operator block.")
    spaces: list[ArraySpace] = []
    for member in members:
        if not isinstance(member, ArraySpace) or len(member.shape) != 1:
            raise TypeError("Meshfree fields are one-dimensional nodal ArraySpaces.")
        spaces.append(member)
    if len({space.shape for space in spaces}) != 1:
        raise ValueError("Meshfree block fields share one nodal enumeration.")
    return tuple(spaces)


def _field_columns(
    value: ArrayLike | None, count: int, fields: int, name: str, /
) -> tuple[np.ndarray, ...]:
    """Host ``(points,)`` columns of a scalar or ``(points, fields)`` array."""
    array = np.zeros((count, fields)) if value is None else np.asarray(value)
    if fields == 1 and array.shape == (count,):
        array = array[:, None]
    if array.shape != (count, fields) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with one value per point and field.")
    return tuple(np.asarray(array[:, index], dtype=np.float64) for index in range(fields))


def _probe_block(
    operator: AbstractLinearOperator, row: int, column: int, /
) -> AbstractLinearOperator | None:
    """Block ``(row, column)`` of a block-space operator.

    A native ``BlockLinearOperator`` exposes its blocks; any other operator on a
    block space is restricted exactly through its own action and transpose.
    """
    if isinstance(operator, BlockLinearOperator):
        return operator.blocks[row][column]
    source, target = operator.source, operator.target
    if not isinstance(source, BlockSpace) or not isinstance(target, BlockSpace):
        raise TypeError("Block fields require a native block-space operator.")

    def apply(value: Array) -> Array:
        inputs = tuple(
            value if index == column else space.zeros()
            for index, space in enumerate(source.spaces)
        )
        return operator.mv(inputs)[row]

    def transpose(value: Array) -> Array:
        inputs = tuple(
            value if index == row else space.zeros()
            for index, space in enumerate(target.spaces)
        )
        return operator.transpose_mv(inputs)[column]

    return FunctionLinearOperator(
        apply,
        source=source.spaces[column],
        target=target.spaces[row],
        transpose_action=transpose,
        operator_id=f"{operator.operator_id}:block:{row}:{column}",
    )


def _as_field_rows(
    operator: AbstractLinearOperator, rows: DualSpace, /
) -> AbstractLinearOperator:
    """Identify native residual rows with the field's coordinate-dual rows."""
    if operator.target.compatible(rows):
        return operator
    if eqx.tree_equal(operator.target.structure(), rows.structure()) is not True:
        raise ValueError("Native residual rows do not match the field coordinates.")
    if isinstance(operator, SparseCoordinateOperator):
        return SparseCoordinateOperator(
            operator.relation,
            operator.coefficients,
            source=operator.source,
            target=rows,
            accumulation_dtype=operator.accumulation_dtype,
            block_shape=operator.block_shape,
            storage_plan=operator._storage_plan,
            operator_id=f"{operator.operator_id}:field-rows",
        )
    return (
        FunctionLinearOperator(
            lambda value: value,
            source=operator.target,
            target=rows,
            transpose_action=lambda value: value,
        )
        @ operator
    )


class _AbstractMeshfreeComponent(
    AbstractCapacityComponent, AbstractReconstructionComponent
):
    """Shared publication of native meshfree rows ``S (A u - b + W g(u))``.

    ``A`` is the owner's native row operator, ``b`` its load, ``S`` the declared
    row scaling (identity except measure-scaled coupling flux rows), ``W`` the
    reaction row weights, and ``u = P z + lift`` per field.
    """

    owner: eqx.AbstractVar[MeshfreeOwner]
    native_operator: eqx.AbstractVar[AbstractLinearOperator]
    reconstruction: eqx.AbstractVar[PreparedFieldReconstruction]
    mass_diagonal: eqx.AbstractVar[Array]
    offsets: eqx.AbstractVar[tuple[Array, ...]]
    loads: eqx.AbstractVar[tuple[Array, ...]]
    row_scales: eqx.AbstractVar[tuple[Array | None, ...]]
    capacity_weights: eqx.AbstractVar[tuple[Array, ...]]
    reaction: eqx.AbstractVar[MeshfreeReaction | None]
    reaction_weights: eqx.AbstractVar[tuple[Array, ...]]
    kernel: eqx.AbstractVar[Array | None]
    impositions: eqx.AbstractVar[tuple[BoundaryImposition, ...]]
    field_ids: eqx.AbstractVar[tuple[str, ...]]

    @property
    def coupled(self) -> bool:
        """Whether the native rows act on a block of several fields."""
        return isinstance(self.native_operator.source, BlockSpace)

    def field_index(self, field: str, /) -> int:
        for index, record in enumerate(self.fields):
            if record.name == field:
                return index
        raise KeyError(f"Component {self.name!r} publishes no field {field!r}.")

    def full_rows(self, fields: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        """Native full rows of every field at full field coefficients."""
        operator = self.native_operator
        image = tuple(operator.mv(fields)) if self.coupled else (operator.mv(fields[0]),)
        reaction = (
            None
            if self.reaction is None
            else self.reaction.evaluate(
                jnp.stack(fields, axis=1), jnp.asarray(self.owner.points), args
            )
        )
        rows: list[Array] = []
        for index, (value, load) in enumerate(zip(image, self.loads, strict=True)):
            row = value - load
            if reaction is not None:
                row = row + self.reaction_weights[index] * reaction[:, index]
            scale = self.row_scales[index]
            rows.append(row if scale is None else scale * row)
        return tuple(rows)

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        fields = tuple(self.expand(record.name, state, args) for record in self.fields)
        return tuple(
            record.pull_back(row)
            for record, row in zip(self.fields, self.full_rows(fields, args), strict=True)
        )

    def linear_operator(self, args: object, /) -> BlockLinearOperator | None:
        del args
        if self.reaction is not None:
            return None
        operator = self.native_operator
        blocks: list[list[AbstractLinearOperator | None]] = []
        for row, row_record in enumerate(self.fields):
            line: list[AbstractLinearOperator | None] = []
            for column, column_record in enumerate(self.fields):
                native = _probe_block(operator, row, column) if self.coupled else operator
                if native is None:
                    line.append(None)
                    continue
                if column_record.constraint is not None:
                    native = native @ column_record.constraint.prolongation
                scale = self.row_scales[row]
                if scale is not None:
                    native = DiagonalLinearOperator(scale, space=native.target) @ native
                native = _as_field_rows(native, row_record.row_space)
                if row_record.constraint is not None:
                    native = row_record.constraint.dual_pullback @ native
                line.append(native)
            blocks.append(line)
        return BlockLinearOperator(blocks, source=self.state_space, target=self.row_space)

    def lift(self, field: str, args: object, /) -> Array:
        del args
        return self.offsets[self.field_index(field)]

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        return None if self.kernel is None else (self.kernel,)

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return self.impositions

    def field_space_id(self, field: str, /) -> str:
        return self.field_ids[self.field_index(field)]

    def prepare_field_reconstruction(self, field: str, /) -> PreparedFieldReconstruction:
        self.field(field)
        return self.reconstruction

    def prepare_capacity(
        self, field: str, /, *, coefficient: ArrayLike = 1.0
    ) -> MeshfreeCapacity:
        index = self.field_index(field)
        record = self.fields[index]
        value = np.asarray(coefficient, dtype=np.float64)
        if value.shape not in ((), self.mass_diagonal.shape) or not np.all(
            np.isfinite(value) & (value > 0)
        ):
            raise ValueError(
                "Capacity coefficient must be finite and positive, scalar or per point."
            )
        return MeshfreeCapacity(
            self.name,
            field,
            self.capacity_weights[index] * jnp.asarray(value),
            record.full_space,
        )


def _owner_measures(
    owner: MeshfreeOwner,
    reconstruction: PreparedFieldReconstruction,
    /,
) -> tuple[Array, AbstractVectorSpace]:
    """Native quadrature measures and nominal field space; refuse foreign revisions."""
    if isinstance(owner, PreparedPointCloudDiscretization):
        if (
            not isinstance(reconstruction.kernel, PointCloudFieldReconstructionKernel)
            or reconstruction.kernel.source_owner_id != owner.prepared_id
        ):
            raise ValueError(
                "Reconstruction belongs to another native point-cloud owner revision."
            )
        return owner.quadrature_weights, owner.field_spaces[0].vector_space
    if isinstance(owner, PreparedSurfacePointCloud):
        if (
            not isinstance(reconstruction.kernel, SurfaceFieldReconstructionKernel)
            or reconstruction.kernel.source_owner_id != owner.prepared_id
        ):
            raise ValueError(
                "Reconstruction belongs to another native intrinsic-surface owner revision."
            )
        return owner.measures, owner.laplace_beltrami.source
    kernel_ = reconstruction.kernel
    if isinstance(kernel_, PointCloudFieldReconstructionKernel):
        reconstruction_points = kernel_.points
    elif isinstance(kernel_, SurfaceFieldReconstructionKernel):
        reconstruction_points = kernel_.surface.points
    else:
        raise TypeError(
            "Exterior fields require a canonical point-cloud or intrinsic reconstruction."
        )
    if not np.array_equal(np.asarray(reconstruction_points), np.asarray(owner.points)):
        raise ValueError(
            "Exterior reconstruction must explicitly share the compact native point enumeration."
        )
    return owner.node_volumes, owner.incidence.source


def _verified_kernel(
    nullspace: ArrayLike | None,
    record: ComponentField,
    operator: AbstractLinearOperator,
    state: ArraySpace,
    /,
) -> np.ndarray | None:
    kernel = None if nullspace is None else np.asarray(nullspace)
    if kernel is None:
        return None
    if (
        kernel.ndim != 2
        or kernel.shape[0] != state.size
        or not np.all(np.isfinite(kernel))
    ):
        raise ValueError("Nullspace columns must use the declared state coordinates.")
    for column in kernel.T:
        value = jnp.asarray(column, dtype=record.full_space.dtype)
        expanded = (
            value
            if record.constraint is None
            else record.constraint.homogeneous_correction(value)
        )
        residual = record.pull_back(operator.mv(expanded))
        if np.linalg.norm(np.asarray(residual)) > 1e-9 * max(1.0, np.linalg.norm(column)):
            raise ValueError("Declared nullspace is not a kernel of the native operator.")
    return kernel


def _verified_weak_trace_kernel(
    owner: PreparedPointCloudPoisson,
    record: ComponentField,
    operator: AbstractLinearOperator,
    /,
) -> Array | None:
    """Publish owner-declared floating constants only with both-side evidence."""
    gauges = owner.plan.gauges
    if not gauges:
        return None
    labels = np.asarray(owner.plan.component_labels)
    full = jnp.asarray(
        labels[:, None] == labels[np.asarray(gauges)][None, :],
        dtype=record.full_space.dtype,
    )
    kernel = full if record.free_rows is None else full[record.free_rows]
    for column in np.asarray(kernel).T:
        value = jnp.asarray(column, dtype=record.full_space.dtype)
        expanded = (
            value
            if record.constraint is None
            else record.constraint.homogeneous_correction(value)
        )
        threshold = 1e-9 * max(1.0, np.linalg.norm(column))
        for action in (operator.mv, operator.transpose_mv):
            residual = np.asarray(record.pull_back(action(expanded)))
            if not np.all(np.isfinite(residual)) or np.linalg.norm(residual) > threshold:
                raise ValueError(
                    "Floating weak trace constants require verified right and left "
                    "kernels of the native reduced operator."
                )
    return kernel


@final
class MeshfreeComponent(_AbstractMeshfreeComponent):
    """Publish a prepared point-cloud, intrinsic, or conservative exterior equation.

    ``operator`` is the native full-row operator, not a reconstructed solver:
    on one nodal ``ArraySpace`` for a scalar field, or on a ``BlockSpace`` of
    nodal spaces for a coupled block of fields (one name per block in
    ``field``). ``mass_diagonal`` is the owner's physical quadrature measure and
    the capacity row measure of every field. Reconstruction and equation
    coordinates must agree. A scalar field may carry a ``constraint`` that
    parameterizes ``u = P z + lift`` and pulls residual rows back exactly once.
    ``reaction`` adds ``reaction_weights * g(u, x, args)`` to the rows and makes
    the publication nonlinear. No facet capability is advertised. Nullspace
    columns are explicit declarations, verified against the homogeneous
    published operator at host preparation.
    """

    __strict_contract__ = True
    name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    owner: MeshfreeOwner
    native_operator: AbstractLinearOperator
    reconstruction: PreparedFieldReconstruction
    mass_diagonal: Float[_MeshfreePointDim]
    offsets: tuple[Array, ...]
    loads: tuple[Array, ...]
    row_scales: tuple[Array | None, ...]
    capacity_weights: tuple[Array, ...]
    reaction: MeshfreeReaction | None
    reaction_weights: tuple[Array, ...]
    kernel: Float[_MeshfreeStateDim, _MeshfreeNullityDim] | None
    impositions: tuple[BoundaryImposition, ...]
    field_ids: tuple[str, ...] = eqx.field(static=True)
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    @checked
    def __init__(
        self,
        owner: MeshfreeOwner,
        operator: AbstractLinearOperator,
        reconstruction: PreparedFieldReconstruction,
        mass_diagonal: ArrayLike,
        /,
        *,
        name: str,
        owner_id: str,
        field: str | Sequence[str] = "concentration",
        constraint: ConstraintMap | None = None,
        lift: ArrayLike | None = None,
        load: ArrayLike | None = None,
        free_rows: Array | None = None,
        nullspace: ArrayLike | None = None,
        boundary_impositions: tuple[BoundaryImposition, ...] = (),
        reaction: MeshfreeReaction | None = None,
        reaction_weights: ArrayLike | None = None,
    ) -> None:
        if operator.batch_shape:
            raise ValueError("operator must be an unbatched native linear operator.")
        if not operator.capabilities.transpose:
            raise ValueError(
                "The native equation must publish its exact coordinate transpose."
            )
        names = _field_names(field)
        coupled = isinstance(operator.source, BlockSpace)
        if not coupled and not isinstance(operator.source, ArraySpace):
            raise TypeError("Meshfree equations require an ArraySpace or BlockSpace.")
        spaces = _field_spaces(operator, len(names))
        full = spaces[0]
        if reconstruction.coefficient_shape != full.shape:
            raise ValueError(
                "Equation and reconstruction coefficient coordinates differ."
            )
        if reconstruction.value_shape or not reconstruction.coefficient_linear:
            raise ValueError("MeshfreeComponent requires a scalar linear reconstruction.")
        targets = (
            operator.target.spaces
            if isinstance(operator.target, BlockSpace)
            else (operator.target,)
        )
        if len(targets) != len(spaces) or not all(
            target.compatible(space) or target.compatible(DualSpace(space))
            for target, space in zip(targets, spaces, strict=True)
        ):
            raise ValueError(
                "Native residual rows must use the field's nominal coordinates or coordinate dual."
            )
        if coupled and (
            constraint is not None or free_rows is not None or nullspace is not None
        ):
            raise ValueError(
                "Constraints and nullspace declarations apply to a scalar field."
            )
        diagonal = np.asarray(mass_diagonal, dtype=np.float64)
        if diagonal.shape != full.shape or not np.all(
            np.isfinite(diagonal) & (diagonal > 0)
        ):
            raise ValueError("Native mass measures must be finite and strictly positive.")
        native_measures, native_space = _owner_measures(owner, reconstruction)
        if not coupled and not full.compatible(native_space):
            raise ValueError(
                "The operator source must be the native owner's nominal field space."
            )
        if not np.array_equal(diagonal, np.asarray(native_measures)):
            raise ValueError(
                "Mass diagonal must be the declared native owner's quadrature measures."
            )
        offsets = _field_columns(lift, full.size, len(names), "Lift")
        loads = _field_columns(load, full.size, len(names), "Load")
        if (reaction is None) != (reaction_weights is None):
            raise ValueError("A reaction and its row weights are declared together.")
        weights = _field_columns(
            reaction_weights, full.size, len(names), "Reaction weights"
        )
        if not all(
            isinstance(value, BoundaryImposition) for value in boundary_impositions
        ):
            raise TypeError(
                "Boundary declarations must be native BoundaryImposition values."
            )
        records = tuple(
            ComponentField(
                field_name,
                state_block=field_name,
                row_block=field_name,
                full_space=space,
                constraint=constraint,
                free_rows=free_rows,
            )
            for field_name, space in zip(names, spaces, strict=True)
        )
        state = full if constraint is None else constraint.reduced_space
        if not isinstance(state, ArraySpace) or len(state.shape) != 1:
            raise ValueError("Meshfree constraints must retain scalar array coordinates.")
        kernel = _verified_kernel(nullspace, records[0], operator, state)
        dtype = full.dtype
        self.name = canonical_identifier(name, "name")
        self.owner_id = canonical_identifier(owner_id, "owner_id")
        self.space = "full" if constraint is None else "reduced"
        self.owner = owner
        self.native_operator = operator
        self.reconstruction = reconstruction
        self.mass_diagonal = jnp.asarray(diagonal, dtype=dtype)
        self.offsets = tuple(jnp.asarray(value, dtype=dtype) for value in offsets)
        self.loads = tuple(jnp.asarray(value, dtype=dtype) for value in loads)
        self.row_scales = tuple(None for _ in names)
        self.capacity_weights = tuple(self.mass_diagonal for _ in names)
        self.reaction = reaction
        self.reaction_weights = tuple(
            jnp.asarray(value, dtype=dtype) for value in weights
        )
        self.kernel = None if kernel is None else jnp.asarray(kernel, dtype=dtype)
        self.impositions = boundary_impositions
        self.field_ids = (
            tuple(space.space_id for space in spaces)
            if coupled
            else (reconstruction.field_space_id,)
        )
        self.fields = records
        self.state_blocks = tuple(
            ComponentBlock(record.name, state if not coupled else record.full_space)
            for record in records
        )
        self.row_blocks = tuple(
            ComponentBlock(block.name, DualSpace(block.space))
            for block in self.state_blocks
        )


# --- Boundary-authorized meshfree traces -------------------------------------------------


@final
class MeshfreeBoundaryTrace(StrictModule, NonTrainableState):
    """One coupling boundary of a system owner's field, authorized by geometry charts.

    ``label`` names the owner's homogeneous ``neumann`` ``PointBoundaryCondition``
    of the field's boundary component: its rows are the pointwise outward
    conormal (traction) rows on which a coupling law acts. ``charts`` supplies the
    oriented facets; the condition's rows, normals, and physical measure must be
    exactly the charts' points, normals, and lumped measure.
    """

    field: str = eqx.field(static=True)
    label: str = eqx.field(static=True)
    charts: PointBoundaryCharts

    @checked
    def __init__(self, field: str, label: str, charts: PointBoundaryCharts, /) -> None:
        self.field = canonical_identifier(field, "field")
        self.label = canonical_identifier(label, "label")
        self.charts = charts


def _selection_constraint(full: ArraySpace, free: np.ndarray, /) -> ConstraintMap:
    """Row-selection chart ``u = P z`` that keeps the rows ``free``."""
    rows = jnp.asarray(free.astype(np.int32))
    reduced = ArraySpace(
        (free.size,), dtype=full.dtype, space_id=f"{full.space_id}:free-rows"
    )
    prolongation = FunctionLinearOperator(
        lambda value: jnp.zeros(full.shape, dtype=full.dtype).at[rows].set(value),
        source=reduced,
        target=full,
        transpose_action=lambda value: value[rows],
        operator_id=canonical_fingerprint(
            {
                "kind": "meshfree-free-row-selection",
                "space": full.space_id,
                "rows": array_tree_fingerprint(free.astype(np.int32)),
            }
        ),
    )
    return ConstraintMap(full, reduced, prolongation)


def _ghost_rows(
    operator: AbstractLinearOperator,
    rhs: Array,
    ghosts: PreparedPointGhostLayer,
    clouds: tuple[ArraySpace, ...],
    fields: tuple[str, ...],
    flux_rows: tuple[np.ndarray, ...],
    /,
) -> tuple[AbstractLinearOperator, tuple[ArraySpace, ...], tuple[Array, ...]]:
    """Pack owner unknowns and exchange only each component's natural rows.

    Native components use cloud-then-ghost coordinates. Published named blocks
    put all cloud fields first, then their ghost fields. At a flux-owned point
    the condition moves to its cloud row and the PDE to its ghost row. A point
    that another component owns with Dirichlet data keeps both native rows.
    The permutation is involutive, including in the coordinate transpose.
    """
    count = ghosts.cloud_count
    components = len(fields)
    _field_spaces(operator, components)
    extensions = tuple(
        ArraySpace(
            (ghosts.ghost_count,),
            dtype=cloud.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "meshfree-ghost-values",
                    "ghosts": ghosts.prepared_id,
                    "field": cloud.space_id,
                }
            ),
        )
        for cloud in clouds
    )
    names = fields + tuple(f"{field}-ghost" for field in fields)
    space = BlockSpace(clouds + extensions, names=names)
    ghost_points = np.asarray(ghosts.plan.rows)
    selections = tuple(
        (
            jnp.asarray(ghost_points[np.isin(ghost_points, rows)], dtype=jnp.int32),
            jnp.asarray(np.flatnonzero(np.isin(ghost_points, rows)), dtype=jnp.int32),
        )
        for rows in flux_rows
    )
    coupled = isinstance(operator.source, BlockSpace)

    def pack(value: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple(
            jnp.concatenate((value[index], value[components + index]))
            for index in range(components)
        )

    def unpack(value: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple(column[:count] for column in value) + tuple(
            column[count:] for column in value
        )

    def exchange(value: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple(
            column.at[rows]
            .set(column[count + indices])
            .at[count + indices]
            .set(column[rows])
            for column, (rows, indices) in zip(value, selections, strict=True)
        )

    def apply(value: tuple[Array, ...]) -> tuple[Array, ...]:
        packed = pack(value)
        image = tuple(operator.mv(packed)) if coupled else (operator.mv(packed[0]),)
        return unpack(exchange(image))

    def transpose(value: tuple[Array, ...]) -> tuple[Array, ...]:
        packed = exchange(pack(value))
        image = (
            tuple(operator.transpose_mv(packed))
            if coupled
            else (operator.transpose_mv(packed[0]),)
        )
        return unpack(image)

    exchanged = FunctionLinearOperator(
        apply,
        source=space,
        target=space,
        transpose_action=transpose,
        operator_id=canonical_fingerprint(
            {
                "kind": "meshfree-ghost-row-exchange",
                "operator": operator.operator_id,
                "ghosts": ghosts.prepared_id,
                "fields": fields,
                "flux_rows": array_tree_fingerprint(flux_rows),
            }
        ),
    )
    columns = rhs if rhs.ndim == 2 else rhs[:, None]
    loads = unpack(exchange(tuple(columns[:, index] for index in range(components))))
    return exchanged, extensions, loads


def _require_trace_condition(
    trace: MeshfreeBoundaryTrace,
    condition: PointBoundaryCondition,
    component: int,
    dirichlet: np.ndarray,
    /,
) -> np.ndarray:
    """Rows of the coupling condition after exact chart identity is verified."""
    if condition.kind != "neumann" or condition.component != component:
        raise ValueError(
            f"Coupling boundary {trace.label!r} must be a neumann condition of the "
            f"field's boundary component {component}."
        )
    if np.any(np.asarray(condition.values) != 0.0):
        raise ValueError(
            f"Coupling boundary {trace.label!r} carries native boundary data; the "
            "coupling law owns the interface flux, so the condition is homogeneous."
        )
    if condition.measure is None or condition.normals is None:
        raise ValueError(
            f"Coupling boundary {trace.label!r} must declare normals and measure."
        )
    rows = np.asarray(condition.rows)
    charts = trace.charts
    nodes = np.asarray(charts.nodes)
    if not np.all(np.isin(rows, nodes)):
        raise ValueError(
            f"Rows of coupling boundary {trace.label!r} lie off its authoritative charts."
        )
    owned = np.isin(nodes, rows) | np.isin(nodes, dirichlet)
    if not np.all(owned):
        raise ValueError(
            f"Every chart point of {trace.label!r} must be a coupling flux row or a "
            "strongly imposed row of the field."
        )
    measure = np.asarray(charts.measure)[rows]
    declared = np.asarray(condition.measure)
    if not np.allclose(declared, measure, rtol=1e-10, atol=0.0):
        raise ValueError(
            f"The physical measure of {trace.label!r} is not the charts' lumped "
            "boundary measure."
        )
    lookup = np.full(charts.measure.shape, -1, dtype=np.int64)
    lookup[rows] = np.arange(rows.size)
    on_rows = lookup[nodes] >= 0
    normals = np.asarray(charts.normals)
    expected = np.asarray(condition.normals)[np.maximum(lookup[nodes], 0)]
    if not np.allclose(normals[on_rows], expected[on_rows], rtol=0.0, atol=1e-10):
        raise ValueError(
            f"The outward normals of {trace.label!r} disagree with its oriented charts."
        )
    return rows


@final
class MeshfreeTraceComponent(_AbstractMeshfreeComponent, AbstractTraceComponent):
    """A prepared point-cloud system owner whose coupling boundaries are chart-authorized.

    ``owner`` is a ``PreparedPointCloudPoisson`` (one scalar field) or a
    ``PreparedPointBlockSystem`` (one field per block component). Its physical
    rows are published unchanged: scalar dissipative natural rows are integrated
    volume test equations; collocated and block owners carry pointwise
    conormal/traction rows. Dirichlet rows become a row-selection constraint
    with the owner's boundary values as lift. Pointwise rows of every ``traces``
    condition are multiplied by their physical boundary measure; integrated
    scalar weak rows need no further scaling. Value traces are the charts'
    nodal interpolants. ``reaction`` and capacities retain the owner's volume
    row measure on weak natural rows and bulk rows, excluding Dirichlet rows;
    pointwise condition rows stay algebraic.

    A boundary-ghost owner (``PointGhostLayerPlan``) adds one named block
    ``"<field>-ghost"`` per field, after all cloud field blocks. Each field's
    flux condition row (divided by the ghost offset) is exchanged with its
    boundary PDE row, which becomes that field's ghost row, and rescaled by
    ``measure * offset``. Dirichlet rows remain in their native coordinates.
    ``ghost_extension_defect`` reports the maximum extension defect over all
    fields. Ghost blocks publish zero capacity; cloud transient capacities and
    pointwise reactions are refused rather than dropping boundary PDE rates.

    Exact pointwise fluxes and trace-inverse stability are refused: a collocated
    or quadrature-adjoint point owner certifies no facet energy bound, so the
    meshfree side of a Nitsche law must carry zero flux weight.
    """

    __strict_contract__ = True
    name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    system: MeshfreeSystemOwner
    owner: PreparedPointCloudDiscretization
    native_operator: AbstractLinearOperator
    reconstruction: PreparedFieldReconstruction
    mass_diagonal: Float[_MeshfreePointDim]
    offsets: tuple[Array, ...]
    loads: tuple[Array, ...]
    row_scales: tuple[Array | None, ...]
    capacity_weights: tuple[Array, ...]
    reaction: MeshfreeReaction | None
    reaction_weights: tuple[Array, ...]
    kernel: Array | None
    impositions: tuple[BoundaryImposition, ...]
    field_ids: tuple[str, ...] = eqx.field(static=True)
    traces: tuple[MeshfreeBoundaryTrace, ...]
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    @checked
    def __init__(
        self,
        owner: MeshfreeSystemOwner,
        reconstruction: PreparedFieldReconstruction,
        traces: Sequence[MeshfreeBoundaryTrace],
        /,
        *,
        name: str,
        source: ArrayLike = 0.0,
        boundary_values: Mapping[str, ArrayLike] | None = None,
        fields: Sequence[str] | None = None,
        reaction: MeshfreeReaction | None = None,
    ) -> None:
        match owner:
            case PreparedPointCloudPoisson():
                plan = owner.plan
                if plan.route == "oversampled-least-squares" or plan.sides is not None:
                    raise ValueError(
                        "Trace publication requires single-sided square point equations."
                    )
                discretization, boundary = plan.discretization, plan.boundary
                operator: AbstractLinearOperator = owner.physical_assembly.operator
                count = discretization.state_shape[0]
                full_rhs = owner.physical_rhs(
                    jnp.broadcast_to(jnp.asarray(source, dtype=jnp.float64), (count,)),
                    boundary_values=boundary_values,
                )
                ghosts = plan.ghosts
                default: tuple[str, ...] = ("u",)
                form = plan.form
            case PreparedPointBlockSystem():
                plan = owner.plan
                discretization, boundary = plan.discretization, plan.boundary
                ghosts = plan.ghosts
                operator = (
                    owner.physical_assembly
                    if owner.equation_assembly is None
                    else owner.equation_assembly
                ).operator
                count = discretization.state_shape[0]
                full_rhs = owner.equation_rhs(
                    jnp.broadcast_to(
                        jnp.asarray(source, dtype=jnp.float64),
                        (count, len(plan.components)),
                    ),
                    boundary_values=boundary_values,
                )
                default = plan.components
                form = plan.form
            case _:
                assert_never(owner)
        _owner_measures(discretization, reconstruction)
        integrated_natural_rows = (
            isinstance(owner, PreparedPointCloudPoisson)
            and form == "dissipative"
            and ghosts is None
        )
        if (
            reconstruction.coefficient_shape != discretization.state_shape
            or reconstruction.value_shape
            or not reconstruction.coefficient_linear
        ):
            raise ValueError(
                "A trace component requires a scalar linear nodal reconstruction."
            )
        names = _field_names(default if fields is None else tuple(fields))
        # Boundary ghosts are extra owner unknowns: their rows exchange with the
        # flux rows so that each point's conormal condition stays at its own row.
        delta = np.ones(count)
        ghost_blocks: tuple[ArraySpace, ...] = ()
        ghost_loads: tuple[Array, ...] = ()
        if ghosts is None:
            rhs = full_rhs if full_rhs.ndim == 2 else full_rhs[:, None]
            spaces = _field_spaces(operator, len(names))
        else:
            if reaction is not None:
                raise ValueError(
                    "A ghost-route owner carries its boundary PDE rows in the ghost "
                    "block; a pointwise reaction on those rows is not published."
                )
            members = (
                owner.plan.space.spaces
                if isinstance(owner, PreparedPointBlockSystem)
                else (discretization.field_spaces[0].vector_space,)
            )
            cloud_spaces: list[ArraySpace] = []
            for member in members:
                if not isinstance(member, ArraySpace):
                    raise TypeError(
                        "A point cloud's nominal field space is an ArraySpace."
                    )
                cloud_spaces.append(member)
            spaces = tuple(cloud_spaces)
            flux_rows = tuple(
                np.concatenate(
                    [
                        np.asarray(condition.rows)
                        for condition in boundary.conditions
                        if condition.component == index
                        and condition.kind in ("neumann", "robin")
                    ]
                    or [np.zeros((0,), dtype=np.int64)]
                )
                for index in range(len(names))
            )
            operator, ghost_blocks, loads_ = _ghost_rows(
                operator, full_rhs, ghosts, spaces, names, flux_rows
            )
            rhs = jnp.stack(loads_[: len(names)], axis=1)
            ghost_loads = loads_[len(names) :]
            delta[np.asarray(ghosts.plan.rows)] = np.asarray(ghosts.offsets)
        traces_ = tuple(traces)
        if not traces_ or not all(isinstance(t, MeshfreeBoundaryTrace) for t in traces_):
            raise TypeError("traces must be nonempty MeshfreeBoundaryTrace values.")
        for trace in traces_:
            if trace.field not in names:
                raise ValueError(f"Trace {trace.label!r} names an unknown field.")
            if trace.charts.support_id != discretization.prepared_id:
                raise ValueError(
                    f"Charts of {trace.label!r} were prepared on another cloud revision."
                )
        labels = [(trace.field, trace.label) for trace in traces_]
        if len(set(labels)) != len(labels):
            raise ValueError("Each coupling boundary of a field is declared once.")
        measure = (
            np.asarray(discretization.quadrature_weights)
            if form == "dissipative"
            else np.ones(count)
        )
        dtype = spaces[0].dtype
        records: list[ComponentField] = []
        offsets: list[Array] = []
        scales: list[Array | None] = []
        capacities: list[Array] = []
        impositions: list[BoundaryImposition] = []
        identities = (
            tuple(space.space_id for space in spaces)
            if len(names) > 1
            else (reconstruction.field_space_id,)
        )
        for index, (field_name, space, identity) in enumerate(
            zip(names, spaces, identities, strict=True)
        ):
            conditions = tuple(
                condition
                for condition in boundary.conditions
                if condition.component == index
            )
            dirichlet = np.unique(
                np.concatenate(
                    [np.asarray(c.rows) for c in conditions if c.kind == "dirichlet"]
                    or [np.zeros((0,), dtype=np.int64)]
                )
            ).astype(np.int64)
            owned = np.zeros(count, dtype=np.bool_)
            for condition in conditions:
                owned[condition.owned_rows] = True
            scale = np.ones(count)
            coupling: set[str] = set()
            for trace in traces_:
                if trace.field != field_name:
                    continue
                rows = _require_trace_condition(
                    trace, boundary.condition(trace.label), index, dirichlet
                )
                if not integrated_natural_rows:
                    # Ghost conormal rows are divided by their offset; scalar
                    # weak rows already carry their integrated boundary load.
                    scale[rows] = (
                        np.asarray(boundary.condition(trace.label).measure) * delta[rows]
                    )
                coupling.add(trace.label)
            for condition in conditions:
                if condition.kind == "dirichlet" or condition.label in coupling:
                    continue
                impositions.append(
                    BoundaryImposition(
                        "robin" if condition.kind == "robin" else "natural",
                        field_space_id=identity,
                        source_id=condition.condition_id,
                        rows=condition.owned_rows,
                    )
                )
            lift = np.zeros(count)
            constraint: ConstraintMap | None = None
            free_rows: Array | None = None
            if dirichlet.size:
                lift[dirichlet] = np.asarray(rhs[:, index])[dirichlet]
                free = np.setdiff1d(np.arange(count), dirichlet)
                constraint = _selection_constraint(space, free)
                free_rows = jnp.asarray(free.astype(np.int32))
                impositions.append(
                    BoundaryImposition(
                        "strong",
                        field_space_id=identity,
                        source_id=canonical_fingerprint(
                            {"kind": "meshfree-dirichlet-rows", "field": identity}
                        ),
                        rows=dirichlet,
                    )
                )
            records.append(
                ComponentField(
                    field_name,
                    state_block=field_name,
                    row_block=field_name,
                    full_space=space,
                    constraint=constraint,
                    free_rows=free_rows,
                )
            )
            offsets.append(jnp.asarray(lift, dtype=dtype))
            scales.append(
                None
                if not coupling or integrated_natural_rows
                else jnp.asarray(scale, dtype=dtype)
            )
            algebraic = (
                np.isin(np.arange(count), dirichlet) if integrated_natural_rows else owned
            )
            capacities.append(jnp.asarray(np.where(algebraic, 0.0, measure), dtype=dtype))
        loads = [jnp.asarray(rhs[:, index], dtype=dtype) for index in range(len(names))]
        for field_name, extension, ghost_load in zip(
            names[: len(ghost_blocks)], ghost_blocks, ghost_loads, strict=True
        ):
            ghost_name = f"{field_name}-ghost"
            records.append(
                ComponentField(
                    ghost_name,
                    state_block=ghost_name,
                    row_block=ghost_name,
                    full_space=extension,
                    constraint=None,
                    free_rows=None,
                )
            )
            offsets.append(jnp.zeros(extension.shape, dtype=dtype))
            loads.append(jnp.asarray(ghost_load, dtype=dtype))
            scales.append(None)
            capacities.append(jnp.zeros(extension.shape, dtype=dtype))
            identities = identities + (extension.space_id,)
        system_id = owner.plan.plan_id
        self.owner_id = canonical_fingerprint(
            {
                "kind": "meshfree-trace-component",
                "system": system_id,
                "operator": operator.operator_id,
                "load": array_tree_fingerprint(np.asarray(full_rhs)),
                "traces": [
                    [trace.field, trace.label, trace.charts.revision_id]
                    for trace in traces_
                ],
                "reaction": None if reaction is None else reaction.reaction_id,
            }
        )
        self.name = canonical_identifier(name, "name")
        self.space = (
            "full" if all(record.constraint is None for record in records) else "reduced"
        )
        self.system = owner
        self.owner = discretization
        self.native_operator = operator
        self.reconstruction = reconstruction
        self.mass_diagonal = jnp.asarray(discretization.quadrature_weights, dtype=dtype)
        self.offsets = tuple(offsets)
        self.loads = tuple(loads)
        self.row_scales = tuple(scales)
        self.capacity_weights = tuple(capacities)
        self.reaction = reaction
        self.reaction_weights = tuple(capacities)
        self.kernel = (
            _verified_weak_trace_kernel(owner, records[0], operator)
            if integrated_natural_rows
            and isinstance(owner, PreparedPointCloudPoisson)
            and reaction is None
            else None
        )
        self.impositions = tuple(impositions)
        self.field_ids = identities
        self.traces = traces_
        self.fields = tuple(records)
        self.state_blocks = tuple(
            ComponentBlock(
                record.name,
                record.full_space
                if record.constraint is None
                else record.constraint.reduced_space,
            )
            for record in records
        )
        self.row_blocks = tuple(
            ComponentBlock(block.name, DualSpace(block.space))
            for block in self.state_blocks
        )

    @property
    def ghost_layer(self) -> PreparedPointGhostLayer | None:
        """Shared boundary ghost geometry; each cloud field has its own ghost block."""
        return self.system.plan.ghosts

    def ghost_extension_defect(self, fields: Mapping[str, Array], /) -> Array:
        """Maximum extension defect over full named cloud and ghost fields."""
        layer = self.ghost_layer
        if layer is None:
            raise ValueError(f"Component {self.name!r} declares no boundary ghosts.")
        count = len(self.fields) // 2
        defects = tuple(
            layer.extension_defect(
                jnp.concatenate((fields[record.name], fields[f"{record.name}-ghost"]))
            )
            for record in self.fields[:count]
        )
        return jnp.max(jnp.stack(defects))

    def prepare_capacity(
        self, field: str, /, *, coefficient: ArrayLike = 1.0
    ) -> MeshfreeCapacity:
        record = self.field(field)
        ghost_fields = (
            self.fields[len(self.fields) // 2 :] if self.ghost_layer is not None else ()
        )
        if any(ghost.name == record.name for ghost in ghost_fields):
            value = np.asarray(coefficient, dtype=np.float64)
            if value.shape not in ((), record.full_space.shape) or not np.all(
                np.isfinite(value) & (value > 0)
            ):
                raise ValueError(
                    "Capacity coefficient must be finite and positive, scalar or per ghost."
                )
            return MeshfreeCapacity(
                self.name,
                field,
                jnp.zeros(record.full_space.shape, dtype=record.full_space.dtype),
                record.full_space,
            )
        if self.ghost_layer is not None:
            # The PDE rows of ghost-extended boundary points live in the ghost row
            # block, whose diagonal capacity would drop their state rate.
            raise ValueError(
                f"Component {self.name!r} is a ghost-route owner: its boundary PDE rows "
                "carry no diagonal capacity, so it publishes no transient capacity."
            )
        return super().prepare_capacity(field, coefficient=coefficient)

    def boundary_trace(self, field: str, label: str, /) -> MeshfreeBoundaryTrace:
        for trace in self.traces:
            if trace.field == field and trace.label == label:
                return trace
        raise KeyError(f"Field {field!r} has no coupling boundary {label!r}.")

    def boundary_domain(self, field: str, label: str, /) -> IntegrationDomain:
        """Exterior-facet domain of one chart-authorized coupling boundary."""
        return self.boundary_trace(field, label).charts.domain()

    def _charts_for(self, field: str, entity_set_id: str, /) -> PointBoundaryCharts:
        for trace in self.traces:
            if trace.field == field and trace.charts.entity_set_id == entity_set_id:
                return trace.charts
        raise ValueError(
            f"Entity set {entity_set_id!r} is not a chart-authorized coupling boundary "
            f"of field {field!r}; a point cloud publishes no other facets."
        )

    def prepare_side_trace(
        self,
        field: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
    ) -> PreparedTraceAction:
        record = self.field(field)
        if quantity != "value" or side != "owner":
            raise ValueError(
                "Meshfree coupling boundaries publish owner-side value traces; the "
                "conormal flux is the owner's residual reaction."
            )
        charts = self._charts_for(field, domain.entity_set_id)
        return charts.prepare_trace(
            record.full_space,
            domain,
            rule,
            owner_id=self.owner_id,
            field_space_id=self.field_space_id(field),
        )

    @checked
    def prepare_conormal_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        descriptor = trace.descriptor
        index = next(
            (
                position
                for position, identity in enumerate(self.field_ids)
                if identity == descriptor.field_space_id
            ),
            None,
        )
        if descriptor.owner_id != self.owner_id or index is None:
            raise ValueError("The trace was prepared by another owner or field.")
        record = self.fields[index]
        charts = self._charts_for(record.name, descriptor.entity_set_id)
        trace.require_revision(charts.revision_id)
        domain = charts.domain(
            tuple(int(facet) for facet in np.asarray(descriptor.facets))
        )
        flux = SideActionDescriptor(
            owner_id=self.owner_id,
            field_space_id=descriptor.field_space_id,
            quantity="conormal-flux",
            representation="residual-reaction",
            orientation="outward",
            approximation="variational-reaction",
            side="owner",
            domain=domain,
            revision_id=descriptor.revision_id,
            rule=None,
            trace_degree=None,
            quadrature_exact_degree=None,
        )
        return PreparedFluxAction(
            flux,
            trace,
            _MeshfreeReactionEvaluator(self, index, trace.support_rows),
        )

    def prepare_pointwise_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        del trace
        raise ValueError(
            f"Component {self.name!r} is a point-collocation owner: it publishes its "
            "flux only as the measure-scaled residual reaction of its conormal rows, "
            "not as an exact pointwise facet flux; give its Nitsche side zero flux "
            "weight or use a mortar or conservative-flux law."
        )

    def certify_flux_stability(self, flux: PreparedFluxAction, /) -> TraceInverseEvidence:
        del flux
        raise ValueError(
            f"Component {self.name!r} certifies no facet trace-inverse energy bound."
        )


@final
class _MeshfreeReactionEvaluator(AbstractSideFluxEvaluator):
    """Measure-scaled native rows of one field on a coupling boundary.

    A coupled block field's reaction depends on every field of the block, so
    its state is the tuple of all full fields; a scalar field's is its own.
    """

    component: MeshfreeTraceComponent
    index: int = eqx.field(static=True)
    support_rows: Array

    @property
    def state_space(self) -> AbstractVectorSpace:
        fields = self.component.fields
        if not self.component.coupled:
            return fields[self.index].full_space
        return BlockSpace(
            tuple(record.full_space for record in fields),
            names=tuple(record.name for record in fields),
        )

    def evaluate(self, state: PyTree[Array], args: object, /) -> Array:
        if not self.component.coupled:
            values: tuple[Array, ...] = (jnp.asarray(state),)
        elif isinstance(state, tuple):
            values = tuple(jnp.asarray(value) for value in state)
        else:
            raise TypeError("A coupled block reaction reads every field of the block.")
        rows = self.component.full_rows(values, args)
        return rows[self.index][self.support_rows]


__all__ = [
    "MeshfreeBoundaryTrace",
    "MeshfreeCapacity",
    "MeshfreeComponent",
    "MeshfreeReaction",
    "MeshfreeTraceComponent",
]

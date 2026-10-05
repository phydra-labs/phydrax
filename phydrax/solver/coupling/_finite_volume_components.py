#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cell-centered finite-volume scalar diffusion published as a coupled component.

The component publishes the one scalar field of a prepared conservative
diffusion problem on a structured grid. Its state is the full set of cell
averages on the grid's cell coordinates; its rows are the cell balances
``V_c (-(D u)_c - f_c)``, where ``D`` is the native diffusion action including
the owner's own affine boundary data. Dirichlet, Robin, and Neumann faces are
imposed through that boundary data, never through a constraint map, so a face
coupled by a law must be declared homogeneous Neumann in the owner: its cell
rows then miss exactly the coupled face's flux, which is the residual reaction
the laws balance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never, final, NoReturn

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...discretization import (
    AbstractSideFluxEvaluator,
    BoundaryImposition,
    FacetTraceRule,
    FiniteVolumeDiscretization,
    IntegrationDomain,
    MUSCLReconstruction,
    PiecewiseConstantReconstruction,
    prepare_finite_volume_field_reconstruction,
    PreparedConservativeDiffusion,
    PreparedFieldReconstruction,
    PreparedFluxAction,
    PreparedTraceAction,
    SideActionDescriptor,
    SideTraceQuantity,
)
from ...discretization._views import FieldTraceSide
from ...discretization.finite_volume._side_trace import (
    finite_volume_field_space_id,
    finite_volume_scalar_space,
)
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    BlockLinearOperator,
    DualSpace,
    FunctionLinearOperator,
)
from ...typing import checked, Float, VariadicDim
from ._components import (
    AbstractCapacityComponent,
    AbstractPreparedCapacity,
    AbstractReconstructionComponent,
    AbstractTraceComponent,
    ComponentBlock,
    ComponentField,
    ComponentSpace,
)


_SIDES = ("lower", "upper")


class _CellDims(VariadicDim):
    """Row-major structured cell axes of the scalar field."""


@final
class FiniteVolumeCapacity(AbstractPreparedCapacity, NonTrainableState):
    """``coefficient * V`` of one finite-volume field.

    Cell averages are tested by cell indicators, so the capacity is exactly the
    diagonal of cell volumes; it is not a lumped approximation.
    """

    __strict_contract__ = True
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    diagonal: Float[_CellDims]
    full_space: ArraySpace

    def operator(self, args: object, /) -> AbstractLinearOperator:
        del args
        return FunctionLinearOperator(
            lambda value: self.diagonal * value,
            source=self.full_space,
            target=DualSpace(self.full_space),
            transpose_action=lambda value: self.diagonal * value,
        )


def _cell_values(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    """Finite host values broadcast from a scalar or given on every entity of ``shape``."""
    host = np.asarray(value, dtype=np.float64)
    if host.shape not in ((), shape):
        raise ValueError(f"{name} must be a scalar or have shape {shape}.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    return np.broadcast_to(host, shape).copy()


def _boundary_targets(
    diffusion: PreparedConservativeDiffusion,
    boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None,
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Owner boundary targets per axis and side on the transverse cell layout."""
    grid = diffusion.plan.grid
    supplied = {} if boundary_values is None else dict(boundary_values)
    unknown = set(supplied).difference(grid.axis_names)
    if unknown:
        raise ValueError(f"boundary_values names unknown axes {sorted(unknown)!r}.")
    targets = []
    for axis, (name, structured) in enumerate(
        zip(grid.axis_names, grid.structured_axes, strict=True)
    ):
        raw = supplied.get(name, (0.0, 0.0))
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError(f"boundary_values[{name!r}] must be a (lower, upper) pair.")
        if structured.periodic and name in supplied:
            raise ValueError(f"Periodic axis {name!r} carries no boundary data.")
        transverse = grid.shape[:axis] + grid.shape[axis + 1 :]
        lower, upper = (
            _cell_values(value, transverse, f"boundary_values[{name!r}]") for value in raw
        )
        targets.append((lower, upper))
    return tuple(targets)


def _exterior_faces(
    discretization: FiniteVolumeDiscretization, /
) -> tuple[IntegrationDomain, np.ndarray, np.ndarray, np.ndarray]:
    """Exterior facets and each facet's normal axis, side (0 lower), and owner cell."""
    exterior = discretization.integration_domain("exterior_facet")
    local = np.asarray(exterior.owner_local_entities, dtype=np.int64)
    return exterior, local // 2, local % 2, np.asarray(exterior.owner_cells, np.int64)


def _transverse(
    cells: np.ndarray, shape: tuple[int, ...], axis: int, /
) -> tuple[np.ndarray, ...]:
    """Index of the owner cells into the transverse layout of ``axis``."""
    index = np.unravel_index(cells, shape)
    return index[:axis] + index[axis + 1 :]


@final
class _FiniteVolumeReactionEvaluator(AbstractSideFluxEvaluator, NonTrainableState):
    """Cell balance rows of one finite-volume component on the rows of one side."""

    component: FiniteVolumeComponent
    rows: Array

    @property
    def state_space(self) -> AbstractVectorSpace:
        return self.component.fields[0].full_space

    def evaluate(self, state: PyTree[Array], args: object, /) -> Array:
        return self.component.residual((state,), args)[0].reshape((-1,))[self.rows]


@final
class FiniteVolumeComponent(
    AbstractTraceComponent,
    AbstractCapacityComponent,
    AbstractReconstructionComponent,
    NonTrainableState,
):
    """One scalar field of a prepared cell-centered finite-volume diffusion problem.

    ``discretization`` is the one-component structured finite-volume owner of
    the field and ``diffusion`` the conservative diffusion operator prepared
    on the same grid; both must act on the same cell coordinates. ``source``
    is the cell-average source density ``f`` of ``-div(K grad u) = f`` and
    ``boundary_values`` the per-axis ``(lower, upper)`` targets of the
    diffusion plan's boundary conditions (scalars or values on the transverse
    cells; homogeneous when omitted). The state block is the full set of cell
    averages and the row block of the same name its cell balances
    ``V (-(D u) - f)``, with ``D`` the native action including the boundary
    targets. ``reconstruction`` selects the published face state of side
    traces: the side cell average (default) or a coefficient-linear MUSCL face
    state.
    """

    __strict_contract__ = True
    name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    discretization: FiniteVolumeDiscretization
    diffusion: PreparedConservativeDiffusion
    reconstruction: PiecewiseConstantReconstruction | MUSCLReconstruction | None
    volumes: Float[_CellDims]
    source: Float[_CellDims]
    boundary_targets: tuple[tuple[Array, Array], ...]
    impositions: tuple[BoundaryImposition, ...]
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    @checked
    def __init__(
        self,
        name: str,
        discretization: FiniteVolumeDiscretization,
        diffusion: PreparedConservativeDiffusion,
        /,
        *,
        source: ArrayLike = 0.0,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
        reconstruction: (
            PiecewiseConstantReconstruction | MUSCLReconstruction | None
        ) = None,
    ) -> None:
        name_ = canonical_identifier(name, "name")
        if diffusion.plan.grid.prepared_id != discretization.grid.prepared_id:
            raise ValueError(
                "The diffusion operator and the finite-volume discretization are "
                "prepared on different grids; one component publishes one owner's cells."
            )
        full = finite_volume_scalar_space(discretization)
        if not (diffusion.source.compatible(full) and diffusion.target.compatible(full)):
            raise ValueError(
                "The diffusion operator does not act on the discretization's scalar "
                "cell coordinates (cell layout or storage dtype differ)."
            )
        cells = _cell_values(source, full.shape, "source")
        targets = _boundary_targets(diffusion, boundary_values)
        field_ = canonical_identifier(discretization.field_name, "field")
        owner_id = canonical_fingerprint(
            {
                "kind": "finite-volume-diffusion-component",
                "discretization": discretization.prepared_id,
                "diffusion": diffusion.operator_id,
                "coefficient": array_tree_fingerprint(np.asarray(diffusion.coefficient)),
                "source": array_tree_fingerprint(cells),
                "boundary_values": [
                    [array_tree_fingerprint(value) for value in pair] for pair in targets
                ],
            }
        )
        self.name = name_
        self.owner_id = owner_id
        self.space = "full"
        self.discretization = discretization
        self.diffusion = diffusion
        self.reconstruction = reconstruction
        self.volumes = jnp.asarray(discretization.cell_volumes, dtype=full.dtype)
        self.source = jnp.asarray(cells, dtype=full.dtype)
        self.boundary_targets = tuple(
            (jnp.asarray(lower, dtype=full.dtype), jnp.asarray(upper, dtype=full.dtype))
            for lower, upper in targets
        )
        self.impositions = _impositions(discretization, diffusion, targets, owner_id)
        self.state_blocks = (ComponentBlock(field_, full),)
        self.row_blocks = (ComponentBlock(field_, DualSpace(full)),)
        self.fields = (
            ComponentField(
                field_,
                state_block=field_,
                row_block=field_,
                full_space=full,
                constraint=None,
                free_rows=None,
            ),
        )

    def _balance(self, cells: Array, /) -> Array:
        action = self.diffusion.apply(
            cells,
            boundary_values=dict(
                zip(
                    self.discretization.grid.axis_names,
                    self.boundary_targets,
                    strict=True,
                )
            ),
        )
        return self.volumes * (-action - self.source)

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        del args
        return (self._balance(state[0]),)

    def linear_operator(self, args: object, /) -> BlockLinearOperator:
        del args
        diffusion = self.diffusion
        block = FunctionLinearOperator(
            lambda value: -self.volumes * diffusion.mv(value),
            source=self.state_blocks[0].space,
            target=self.row_blocks[0].space,
            transpose_action=lambda value: -diffusion.transpose_mv(self.volumes * value),
        )
        return BlockLinearOperator(
            ((block,),), source=self.state_space, target=self.row_space
        )

    def lift(self, field: str, args: object, /) -> Array:
        del args
        return self.field(field).full_space.zeros()

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        # The owner's conservation evidence: its homogeneous action annihilates
        # constants exactly when no Dirichlet or Robin face anchors the field.
        report = self.diffusion.conservation_report
        if report.constant_state_residual > report.tolerance:
            return None
        full = self.fields[0].full_space
        return (jnp.ones((full.size, 1), dtype=full.dtype),)

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return self.impositions

    def field_space_id(self, field: str, /) -> str:
        self.field(field)
        return finite_volume_field_space_id(self.discretization)

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
        self.field(field)
        return self.discretization.prepare_side_trace(
            self.discretization.field_name,
            domain,
            rule=rule,
            quantity=quantity,
            side=side,
            reconstruction=self.reconstruction,
            layout="scalar",
        )

    @checked
    def prepare_conormal_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        """Publish the cell-balance reaction on the rows of one exterior trace.

        At a state satisfying the remaining rows, the balance rows of the cells
        behind homogeneous-Neumann faces equal the outward conormal flux
        through those faces. Faces carrying a native Dirichlet, Robin, nonzero
        Neumann, or cross-diffusion flux are refused.
        """
        descriptor = trace.descriptor
        record = self.fields[0]
        if (
            descriptor.owner_id != self.discretization.prepared_id
            or descriptor.field_space_id != self.field_space_id(record.name)
        ):
            raise ValueError(
                "The trace was prepared by another finite-volume owner than this "
                "component's."
            )
        if not trace.coefficient_space.compatible(record.full_space):
            raise ValueError(
                "The trace acts on the owner's cell-average state layout; prepare it "
                "through this component's scalar cell layout."
            )
        if descriptor.quantity != "value" or descriptor.domain_kind != "exterior_facet":
            raise ValueError(
                "Residual reactions are published on exterior-facet value traces; "
                "interior facets carry two-sided fluxes that no row set represents."
            )
        facets = np.asarray(descriptor.facets, dtype=np.int64)
        exterior, axes, sides, cells = _exterior_faces(self.discretization)
        rows = np.asarray(exterior.entity_indices, dtype=np.int64)
        positions = np.minimum(np.searchsorted(rows, facets), rows.size - 1)
        if not np.array_equal(rows[positions], facets):
            raise ValueError("The trace names facets that are not exterior facets.")
        self._require_reaction_faces(axes[positions], sides[positions], cells[positions])
        flux = SideActionDescriptor(
            owner_id=self.owner_id,
            field_space_id=descriptor.field_space_id,
            quantity="conormal-flux",
            representation="residual-reaction",
            orientation="outward",
            approximation="variational-reaction",
            side=descriptor.side,
            domain=IntegrationDomain(
                "exterior_facet",
                facets,
                exterior.support_id,
                descriptor.entity_set_id,
                owner_cells=cells[positions],
                owner_local_entities=2 * axes[positions] + sides[positions],
            ),
            revision_id=descriptor.revision_id,
            rule=None,
            trace_degree=None,
            quadrature_exact_degree=None,
        )
        evaluator = _FiniteVolumeReactionEvaluator(self, trace.support_rows)
        return PreparedFluxAction(flux, trace, evaluator)

    def _require_reaction_faces(
        self, axes: np.ndarray, sides: np.ndarray, cells: np.ndarray, /
    ) -> None:
        """Refuse faces whose native boundary law contributes a flux of its own."""
        grid = self.discretization.grid
        coefficient = np.asarray(self.diffusion.coefficient, dtype=np.float64)
        dimension = len(grid.shape)
        tensors = coefficient.reshape((-1, dimension, dimension))
        faces = sorted(
            {(int(axis), int(side)) for axis, side in zip(axes, sides, strict=True)}
        )
        for axis, side in faces:
            selected = cells[(axes == axis) & (sides == side)]
            label = f"{grid.axis_names[axis]}:{_SIDES[side]}"
            condition = self.diffusion.plan.boundaries[axis][side]
            match condition.kind:
                case "neumann":
                    target = np.asarray(self.boundary_targets[axis][side])
                    if np.any(target[_transverse(selected, grid.shape, axis)] != 0.0):
                        raise ValueError(
                            f"Boundary {label} carries a nonzero native Neumann flux on "
                            "the coupled facets; declare it homogeneous so the cell "
                            "rows there are the coupling reaction."
                        )
                case "dirichlet" | "robin" | "periodic":
                    raise ValueError(
                        f"Boundary {label} carries a native {condition.kind} condition "
                        "on the coupled facets; a coupled face must be declared "
                        "homogeneous Neumann in the owner so its cell rows are the "
                        "coupling reaction."
                    )
                case _:
                    assert_never(condition.kind)
            cross = np.delete(tensors[selected, axis, :], axis, axis=1)
            if np.any(cross != 0.0):
                raise ValueError(
                    f"Boundary {label} has cross-diffusion coefficients in its cells; "
                    "the native Neumann face still carries the tangential flux, so the "
                    "cell rows are not the coupling reaction."
                )

    def prepare_pointwise_flux(self, trace: PreparedTraceAction, /) -> NoReturn:
        del trace
        self._refuse_pointwise()

    def certify_flux_stability(self, flux: PreparedFluxAction, /) -> NoReturn:
        del flux
        self._refuse_pointwise()

    def _refuse_pointwise(self, /) -> NoReturn:
        raise ValueError(
            f"Component {self.name!r} is a cell-centered finite-volume owner: its cell "
            "averages carry no pointwise gradient, so it publishes no exact pointwise "
            "conormal flux (only its cell-balance residual reaction) and no "
            "trace-inverse stability evidence. A one-sided Nitsche law with zero "
            "weight on this side remains admissible."
        )

    def prepare_field_reconstruction(self, field: str, /) -> PreparedFieldReconstruction:
        """Piecewise-constant cell averages of the scalar field (``C^-1``, degree 0)."""
        self.field(field)
        return prepare_finite_volume_field_reconstruction(
            self.discretization, PiecewiseConstantReconstruction(), layout="scalar"
        )

    def prepare_capacity(
        self, field: str, /, *, coefficient: ArrayLike = 1.0
    ) -> FiniteVolumeCapacity:
        record = self.field(field)
        value = np.asarray(coefficient, dtype=np.float64)
        if value.shape != () or not (np.isfinite(value) and value > 0.0):
            raise ValueError(
                "A finite-volume capacity coefficient is one finite positive scalar."
            )
        return FiniteVolumeCapacity(
            component=self.name,
            field=record.name,
            diagonal=self.volumes * jnp.asarray(value, dtype=record.full_space.dtype),
            full_space=record.full_space,
        )


def _impositions(
    discretization: FiniteVolumeDiscretization,
    diffusion: PreparedConservativeDiffusion,
    targets: tuple[tuple[np.ndarray, np.ndarray], ...],
    owner_id: str,
    /,
) -> tuple[BoundaryImposition, ...]:
    """Provenance of the owner's boundary laws on its exterior facets.

    Dirichlet and Robin faces act through the boundary flux (cell-centered
    finite volumes eliminate no rows), so they are weak and Robin impositions;
    Neumann faces impose a natural load only where their target is nonzero.
    Homogeneous Neumann faces impose nothing and remain free for laws.
    """
    exterior, axes, sides, cells = _exterior_faces(discretization)
    facets = np.asarray(exterior.entity_indices, dtype=np.int64)
    shape = discretization.grid.shape
    space_id = finite_volume_field_space_id(discretization)
    impositions = []
    for axis, name in enumerate(discretization.grid.axis_names):
        for side, condition in enumerate(diffusion.plan.boundaries[axis]):
            on_side = (axes == axis) & (sides == side)
            match condition.kind:
                case "periodic":
                    continue
                case "dirichlet":
                    kind = "weak"
                    selected = on_side
                case "robin":
                    kind = "robin"
                    selected = on_side
                case "neumann":
                    kind = "natural"
                    target = targets[axis][side]
                    selected = on_side.copy()
                    selected[on_side] = (
                        target[_transverse(cells[on_side], shape, axis)] != 0.0
                    )
                case _:
                    assert_never(condition.kind)
            if not np.any(selected):
                continue
            impositions.append(
                BoundaryImposition(
                    kind,
                    field_space_id=space_id,
                    source_id=f"{owner_id}:{condition.kind}:{name}:{_SIDES[side]}",
                    entity_set_id=exterior.entity_set_id,
                    facets=facets[selected],
                )
            )
    return tuple(impositions)


__all__ = ["FiniteVolumeCapacity", "FiniteVolumeComponent"]

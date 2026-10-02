#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed publication of native owners as components of a spatial coupled problem.

A component publishes, over an existing native owner, the named state blocks
it is solved for, the residual row blocks of its original equations (which may
differ from the state blocks), whether those blocks are the owner's full or
constraint-reduced coordinates, the full coefficient fields that laws may
couple (each with the owner's constraint map), the owner's boundary-imposition
provenance, and its side-trace and conormal-flux capabilities. The assembler
consumes only this publication; a new numerical method publishes its own
component and needs no assembler change.
"""

from __future__ import annotations

import abc
from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_finite_float
from ...discretization import (
    BoundaryImposition,
    ExplicitPolygonH1Discretization,
    FacetTraceRule,
    FiniteElementDiscretization,
    IntegrationDomain,
    prepare_explicit_polygon_h1_field_reconstruction,
    PreparedFieldReconstruction,
    PreparedFluxAction,
    PreparedTraceAction,
    SideTraceQuantity,
    SimplicialLocationPolicy,
    TraceInverseEvidence,
    VirtualElementDiscretization,
)
from ...discretization._views import FieldTraceSide
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    ConstraintMap,
    DualSpace,
    FunctionLinearOperator,
)
from ...typing import checked, parse
from .._fem_bem_scalar import PreparedScalarLaplaceFEMBEM3D
from .._fem_bem_vector import PreparedElasticityFEMBEM3D


if TYPE_CHECKING:
    from ...discretization.iga import PreparedIsogeometricDiscretization
    from ...equations import (
        CompiledFiniteElementProblem,
        CompiledVirtualElementProblem,
        PreparedFiniteElementMass,
        VariationalCoefficient,
    )
    from ...operators.integral.layer_potential import (
        ExteriorFarField2D,
        ScalarLaplaceGalerkin2D,
    )


ComponentSpace: TypeAlias = Literal["full", "reduced"]


@final
class ComponentBlock(StrictModule, NonTrainableState):
    """One named state (column) or residual-row block of a component."""

    name: str = eqx.field(static=True)
    space: AbstractVectorSpace

    @checked
    def __init__(self, name: str, space: AbstractVectorSpace, /) -> None:
        name_ = canonical_identifier(name, "name")
        self.name = name_
        self.space = space


@final
class ComponentField(StrictModule, NonTrainableState):
    """One full coefficient field of a component that coupling laws may address.

    ``state_block`` parameterizes the field and ``row_block`` holds the rows
    tested by the field's basis. With a ``constraint`` the state block holds
    reduced coordinates ``z`` and the field is ``P z + g``; its rows are
    pulled back by ``P^T``. ``free_rows`` lists the full rows the reduced
    coordinates select when the chart is a plain row selection (``None``
    otherwise, which refuses row elimination on this field).
    """

    name: str = eqx.field(static=True)
    state_block: str = eqx.field(static=True)
    row_block: str = eqx.field(static=True)
    full_space: ArraySpace
    constraint: ConstraintMap | None
    free_rows: Array | None

    @checked
    def __init__(
        self,
        name: str,
        /,
        *,
        state_block: str,
        row_block: str,
        full_space: ArraySpace,
        constraint: ConstraintMap | None,
        free_rows: Array | None,
    ) -> None:
        if constraint is not None and not isinstance(constraint, ConstraintMap):
            raise TypeError("constraint must be a ConstraintMap or None.")
        if constraint is not None and not constraint.full_space.compatible(full_space):
            raise ValueError("The constraint map acts on another full space.")
        rows = None
        if free_rows is not None:
            rows = np.asarray(free_rows)
            if (
                constraint is None
                or rows.ndim != 1
                or rows.size != constraint.reduced_space.size
                or np.any(np.diff(rows) <= 0)
            ):
                raise ValueError(
                    "free_rows must be the increasing full rows of the reduced chart."
                )
        self.name = canonical_identifier(name, "name")
        self.state_block = canonical_identifier(state_block, "state_block")
        self.row_block = canonical_identifier(row_block, "row_block")
        self.full_space = full_space
        self.constraint = constraint
        self.free_rows = None if rows is None else jnp.asarray(rows.astype(np.int32))

    @property
    def row_space(self) -> DualSpace:
        """Coordinate dual of the full field: the space of full-row covectors."""
        return DualSpace(self.full_space)

    def prolongation(self) -> AbstractLinearOperator | None:
        """Homogeneous chart ``P`` from the state block to the full field."""
        return None if self.constraint is None else self.constraint.prolongation

    def pullback_operator(self) -> AbstractLinearOperator | None:
        """Row pullback ``P^T`` from full-row covectors to the row block."""
        return None if self.constraint is None else self.constraint.dual_pullback

    def pull_back(self, covector: Array, /) -> Array:
        return (
            covector
            if self.constraint is None
            else self.constraint.pullback_dual(covector)
        )


class AbstractSpatialComponent(StrictModule):
    """Publication of one native owner to the spatial coupled assembler.

    ``residual`` evaluates the owner's original equations on its row blocks;
    ``linear_operator`` returns the owner's native affine operator (row blocks
    by state blocks) or ``None`` when the owner publishes only a residual.
    ``lift`` is the full-field value at zero state (the owner's Dirichlet
    lift or zero). ``nullspace`` returns a kernel basis of the owner's
    homogeneous operator per state block (columns), or ``None`` when the
    owner declares no kernel.
    """

    name: eqx.AbstractVar[str]
    owner_id: eqx.AbstractVar[str]
    space: eqx.AbstractVar[ComponentSpace]
    state_blocks: eqx.AbstractVar[tuple[ComponentBlock, ...]]
    row_blocks: eqx.AbstractVar[tuple[ComponentBlock, ...]]
    fields: eqx.AbstractVar[tuple[ComponentField, ...]]

    @abc.abstractmethod
    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def linear_operator(self, args: object, /) -> BlockLinearOperator | None:
        raise NotImplementedError

    @abc.abstractmethod
    def lift(self, field: str, args: object, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        raise NotImplementedError

    @abc.abstractmethod
    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        raise NotImplementedError

    @property
    def state_space(self) -> BlockSpace:
        return BlockSpace(
            tuple(block.space for block in self.state_blocks),
            names=tuple(block.name for block in self.state_blocks),
        )

    @property
    def row_space(self) -> BlockSpace:
        return BlockSpace(
            tuple(block.space for block in self.row_blocks),
            names=tuple(block.name for block in self.row_blocks),
        )

    def state_index(self, block: str, /) -> int:
        for index, candidate in enumerate(self.state_blocks):
            if candidate.name == block:
                return index
        raise KeyError(f"Component {self.name!r} has no state block {block!r}.")

    def row_index(self, block: str, /) -> int:
        for index, candidate in enumerate(self.row_blocks):
            if candidate.name == block:
                return index
        raise KeyError(f"Component {self.name!r} has no row block {block!r}.")

    def field(self, name: str, /) -> ComponentField:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(f"Component {self.name!r} publishes no field {name!r}.")

    def expand(self, field: str, state: tuple[Array, ...], args: object, /) -> Array:
        """Full coefficients ``P z + g`` of one field at one component state."""
        record = self.field(field)
        block = state[self.state_index(record.state_block)]
        homogeneous = (
            block
            if record.constraint is None
            else record.constraint.homogeneous_correction(block)
        )
        return homogeneous + self.lift(field, args)

    def strong_rows(self, field: str, /) -> np.ndarray:
        """Full rows of one field that the owner imposes strongly."""
        space_id = self.field_space_id(field)
        rows = [
            np.asarray(imposition.rows)
            for imposition in self.boundary_impositions()
            if imposition.kind == "strong"
            and imposition.rows is not None
            and imposition.field_space_id == space_id
        ]
        if not rows:
            return np.zeros((0,), dtype=np.int32)
        return np.unique(np.concatenate(rows)).astype(np.int32)

    @abc.abstractmethod
    def field_space_id(self, field: str, /) -> str:
        """Owner identity of one field's discrete coefficient space."""
        raise NotImplementedError


class AbstractTraceComponent(AbstractSpatialComponent):
    """Component that also publishes side traces and conormal fluxes of its fields.

    Laws that act through facet traces (transmission, flux, transfer laws)
    require this capability; owners without facet traces (for example
    boundary-integral products) publish plain ``AbstractSpatialComponent``
    values and are coupled through laws that use their own trace spaces.
    """

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_conormal_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_pointwise_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        """Exact pointwise conormal flux densities at the sites of a value trace.

        Owners without an exact pointwise flux of their physical operator refuse.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def certify_flux_stability(self, flux: PreparedFluxAction, /) -> TraceInverseEvidence:
        """Certified trace-inverse constants of one pointwise flux of this owner."""
        raise NotImplementedError


class AbstractReconstructionComponent(AbstractSpatialComponent):
    """Component that also publishes a pointwise reconstruction of its fields.

    Point observations evaluate a field through this reconstruction of the
    component's full coefficients; its ``approximation`` label states whether it
    is exact within cells or a projection.
    """

    @abc.abstractmethod
    def prepare_field_reconstruction(self, field: str, /) -> PreparedFieldReconstruction:
        raise NotImplementedError


class AbstractPreparedCapacity(StrictModule):
    """Host-prepared capacity (mass) operator of one component field.

    ``operator(args)`` evaluates the owner's own ``coefficient * M`` on the full
    field, from full coefficients to full-row covectors
    (``full_space -> DualSpace(full_space)``), with the owner's runtime
    arguments and without host preparation, so it may run inside traced
    residuals.
    """

    component: eqx.AbstractVar[str]
    field: eqx.AbstractVar[str]

    @abc.abstractmethod
    def operator(self, args: object, /) -> AbstractLinearOperator:
        raise NotImplementedError


class AbstractCapacityComponent(AbstractSpatialComponent):
    """Component that also publishes the capacity (mass) operator of its fields.

    Transient coupled problems require this capability for every field they
    declare differential. The capacity is always the owner's own operator; it
    is never synthesized by the coupled layer.
    """

    @abc.abstractmethod
    def prepare_capacity(
        self, field: str, /, *, coefficient: ArrayLike = 1.0
    ) -> AbstractPreparedCapacity:
        raise NotImplementedError


@final
class VariationalCapacity(AbstractPreparedCapacity, NonTrainableState):
    """Capacity of one variational field: FE prepared unit mass or VEM mass action."""

    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    record: ComponentField
    problem: CompiledFiniteElementProblem | CompiledVirtualElementProblem
    finite_element_mass: PreparedFiniteElementMass | None
    virtual_element_coefficient: VariationalCoefficient | None
    coefficient: Array

    def operator(self, args: object, /) -> AbstractLinearOperator:
        from ...equations import CompiledVirtualElementProblem

        prepared_mass = self.finite_element_mass
        if prepared_mass is not None:
            operator = prepared_mass.operator(
                args, coefficient=self.coefficient, return_full=True
            )
        elif isinstance(self.problem, CompiledVirtualElementProblem) and (
            self.virtual_element_coefficient is not None
        ):
            # The owner fingerprints a coefficient when binding it, which is a
            # host step; the capacity was bound once at preparation.
            operator = self.problem.mass_operator(
                args, coefficient=self.virtual_element_coefficient, return_full=True
            )
        else:
            raise ValueError("A finite-element capacity requires its prepared mass.")
        record = self.record
        if not operator.source.compatible(record.full_space):
            raise ValueError("The owner's mass operator does not act on its field.")
        if operator.target.compatible(record.row_space):
            return operator
        # Owners label full-row covectors with their own coefficient layout; the
        # identification with the field's row space is the coordinate identity.
        if (
            eqx.tree_equal(operator.target.structure(), record.row_space.structure())
            is not True
        ):
            raise ValueError("The owner's mass operator does not map to its field rows.")
        return (
            FunctionLinearOperator(
                lambda value: value,
                source=operator.target,
                target=record.row_space,
                transpose_action=lambda value: value,
            )
            @ operator
        )


def _variational_owner_types() -> tuple[type, type]:
    # The equation compilers import solver owners; resolve them at use time.
    from ...equations import (
        CompiledFiniteElementProblem,
        CompiledVirtualElementProblem,
    )

    return CompiledFiniteElementProblem, CompiledVirtualElementProblem


type _TraceDiscretization = (
    FiniteElementDiscretization
    | VirtualElementDiscretization
    | ExplicitPolygonH1Discretization
    | PreparedIsogeometricDiscretization
)


def _trace_discretization(discretization: object, /) -> _TraceDiscretization:
    """The owner's discretization when it publishes side traces of its fields."""
    from ...discretization.iga import PreparedIsogeometricDiscretization

    if not isinstance(
        discretization,
        (
            FiniteElementDiscretization,
            VirtualElementDiscretization,
            ExplicitPolygonH1Discretization,
            PreparedIsogeometricDiscretization,
        ),
    ):
        raise TypeError("The owner's discretization publishes no side traces.")
    return discretization


@final
class VariationalComponent(
    AbstractTraceComponent,
    AbstractCapacityComponent,
    AbstractReconstructionComponent,
    NonTrainableState,
):
    """One scalar field of a compiled variational owner.

    Finite-element (including spectral-element), explicit-polygon H1, and
    isogeometric owners compile through the finite-element compiler;
    virtual-element owners through the virtual-element compiler.

    The state block (named after the field) holds the owner's solve
    coordinates: reduced by its Dirichlet constraint map when it has one,
    otherwise the full coefficients. The row block of the same name holds the
    owner's weak residual rows. The owner's residual, affine operator,
    constraint map, lift, kernel (the right nullspace its native linear system
    declares), imposition provenance, side traces, and reaction fluxes are
    used unchanged; nothing is reassembled here. ``affine`` declares that the
    owner's weak residual is affine in the field (always true for the
    supported VEM actions); a nonaffine finite-element form must declare
    ``affine=False``, which publishes no linear operator, so the coupled
    problem is solved through Newton on the owner's residual instead of a
    zero-state linearization. ``location_policy`` sets the simplicial
    point-location capacities of the owner's field reconstruction; only
    finite-element discretizations locate points through one.
    """

    name: str = eqx.field(static=True)
    problem: CompiledFiniteElementProblem | CompiledVirtualElementProblem
    field_name: str = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    location_policy: SimplicialLocationPolicy | None
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    def __init__(
        self,
        name: str,
        problem: CompiledFiniteElementProblem | CompiledVirtualElementProblem,
        /,
        *,
        field: str,
        affine: bool = True,
        location_policy: SimplicialLocationPolicy | None = None,
    ) -> None:
        name_ = canonical_identifier(name, "name")
        field_ = canonical_identifier(field, "field")
        if not isinstance(problem, _variational_owner_types()):
            raise TypeError(
                "problem must be a CompiledFiniteElementProblem or "
                "CompiledVirtualElementProblem."
            )
        full_space = problem.full_space
        if not isinstance(full_space, ArraySpace):
            raise ValueError(
                "A variational component publishes one field; the owner solves a "
                "mixed form whose full space is not one coefficient array."
            )
        _trace_discretization(problem.discretization)
        if not isinstance(affine, bool):
            raise TypeError("affine must be a bool.")
        if location_policy is not None and not (
            isinstance(location_policy, SimplicialLocationPolicy)
            and isinstance(problem.discretization, FiniteElementDiscretization)
        ):
            raise ValueError(
                "location_policy is a SimplicialLocationPolicy of a finite-element "
                "owner; other owners locate points without one."
            )
        constraint = problem.constraint_map
        record = ComponentField(
            field_,
            state_block=field_,
            row_block=field_,
            full_space=full_space,
            constraint=constraint,
            free_rows=_selection_rows(problem),
        )
        self.name = name_
        self.problem = problem
        self.field_name = field_
        self.affine = affine
        self.location_policy = location_policy
        self.owner_id = problem.compilation_id
        self.space = "full" if constraint is None else "reduced"
        self.state_blocks = (ComponentBlock(field_, problem.state_space),)
        self.row_blocks = (ComponentBlock(field_, problem.residual_space),)
        self.fields = (record,)

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        return (self.problem.residual(state[0], args),)

    def linear_operator(self, args: object, /) -> BlockLinearOperator | None:
        if not self.affine:
            return None
        operator = self.problem.affine_operator(args)
        rows = self.row_blocks[0].space
        if not operator.target.compatible(rows):
            # The owner's weak operator yields residual-row coordinates labeled with
            # the coefficient space; identify them with its declared residual space.
            if eqx.tree_equal(operator.target.structure(), rows.structure()) is not True:
                raise ValueError("The owner's affine operator does not map to its rows.")
            operator = (
                FunctionLinearOperator(
                    lambda value: value,
                    source=operator.target,
                    target=rows,
                    transpose_action=lambda value: value,
                )
                @ operator
            )
        return BlockLinearOperator(
            ((operator,),), source=self.state_space, target=self.row_space
        )

    def lift(self, field: str, args: object, /) -> Array:
        self.field(field)
        zero = self.state_blocks[0].space.zeros()
        return self.problem.expand(zero, args)

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        # The owner publishes its kernel with its native linear system.
        system, _ = self.problem.linear_system(args)
        policy = system.nullspace_policy
        if policy is None or policy.right is None:
            return None
        return (policy.right.basis,)

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return self.problem.boundary_impositions()

    def _discretization(self, /) -> _TraceDiscretization:
        return _trace_discretization(self.problem.discretization)

    def field_space_id(self, field: str, /) -> str:
        self.field(field)
        discretization = self._discretization()
        if isinstance(discretization, VirtualElementDiscretization):
            return discretization.field_space.field_space_id
        for space in discretization.field_spaces:
            if space.name == self.field_name:
                return space.field_space_id
        raise ValueError(f"The owner discretizes no field {self.field_name!r}.")

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
        return self._discretization().prepare_side_trace(
            self.field_name, domain, rule=rule, quantity=quantity, side=side
        )

    def prepare_conormal_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        return self.problem.prepare_conormal_flux(trace)

    def prepare_pointwise_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        return self._finite_element_owner().prepare_pointwise_flux(trace)

    def certify_flux_stability(self, flux: PreparedFluxAction, /) -> TraceInverseEvidence:
        return self._finite_element_owner().certify_flux_stability(flux)

    def _finite_element_owner(self, /) -> CompiledFiniteElementProblem:
        from ...equations import CompiledVirtualElementProblem

        problem = self.problem
        if isinstance(problem, CompiledVirtualElementProblem):
            raise ValueError(
                f"Component {self.name!r} is a virtual-element owner: its virtual "
                "interior gradient is not computable, so it publishes no exact "
                "pointwise conormal flux (only a residual reaction and an "
                "h1-projected flux) and no trace-inverse stability evidence."
            )
        return problem

    def prepare_field_reconstruction(self, field: str, /) -> PreparedFieldReconstruction:
        """The owner's pointwise reconstruction of one field.

        Finite-element fields are reconstructed exactly within cells (located
        with the component's ``location_policy``). Virtual elements publish no
        pointwise interior basis; their reconstruction is the H1 polynomial
        projection, labeled ``"h1-projection"`` (exact edge traces are the
        separate ``prepare_side_trace`` capability).
        """
        self.field(field)
        discretization = self._discretization()
        match discretization:
            case VirtualElementDiscretization():
                from ...equations.vem import (
                    prepare_virtual_element_field_reconstruction,
                )

                return prepare_virtual_element_field_reconstruction(
                    discretization, channel="h1-projection"
                )
            case FiniteElementDiscretization():
                from ...discretization.fem import (
                    prepare_finite_element_field_reconstruction,
                )

                return prepare_finite_element_field_reconstruction(
                    discretization,
                    self.field_name,
                    location_policy=self.location_policy,
                )
            case ExplicitPolygonH1Discretization():
                return prepare_explicit_polygon_h1_field_reconstruction(discretization)
            case _:
                from ...discretization.iga import (
                    prepare_isogeometric_field_reconstruction,
                )

                return prepare_isogeometric_field_reconstruction(
                    discretization, self.field_name
                )

    def prepare_capacity(
        self, field: str, /, *, coefficient: ArrayLike = 1.0
    ) -> VariationalCapacity:
        from ...equations import coefficient as bind, CompiledFiniteElementProblem

        coefficient_ = jnp.asarray(coefficient)
        if coefficient_.shape != () or not jnp.issubdtype(
            coefficient_.dtype, jnp.floating
        ):
            raise ValueError("A capacity coefficient is one real scalar.")
        problem = self.problem
        return VariationalCapacity(
            component=self.name,
            field=self.field(field).name,
            record=self.field(field),
            problem=problem,
            finite_element_mass=(
                problem.prepare_mass()
                if isinstance(problem, CompiledFiniteElementProblem)
                else None
            ),
            virtual_element_coefficient=(
                None
                if isinstance(problem, CompiledFiniteElementProblem)
                else bind(coefficient_)
            ),
            coefficient=coefficient_,
        )


def _selection_rows(
    problem: CompiledFiniteElementProblem | CompiledVirtualElementProblem, /
) -> Array | None:
    """Free rows of a Dirichlet row-selection chart, or ``None`` for other charts."""
    constraint = problem.constraint_map
    if constraint is None:
        return None
    rows = [
        np.asarray(imposition.rows)
        for imposition in problem.boundary_impositions()
        if imposition.kind == "strong" and imposition.rows is not None
    ]
    if len(rows) != 1:
        return None
    full = problem.full_space.size
    free = np.setdiff1d(np.arange(full, dtype=np.int64), rows[0])
    if free.size != constraint.reduced_space.size:
        return None
    probe = np.arange(1, free.size + 1, dtype=np.float64)
    embedded = np.asarray(
        constraint.prolongation.mv(
            jnp.asarray(probe, dtype=constraint.reduced_space.structure().dtype)
        )
    )
    expected = np.zeros((full,), dtype=np.float64)
    expected[free] = probe
    # Only a chart that places each reduced coordinate on its own free row is a
    # row selection; any other chart (hanging nodes, periodicity) refuses elimination.
    if not np.array_equal(embedded, expected):
        return None
    return jnp.asarray(free.astype(np.int32))


def _galerkin_type() -> type:
    from ...operators.integral.layer_potential import ScalarLaplaceGalerkin2D

    return ScalarLaplaceGalerkin2D


def _far_field_declaration(
    far_field: ExteriorFarField2D, tolerance: float | None, /
) -> tuple[ExteriorFarField2D, float | None]:
    """Parsed far-field mode and its tolerance, with the 2-D Galerkin owner's rules."""
    from ...operators.integral.layer_potential import ExteriorFarField2D

    mode = parse(far_field, ExteriorFarField2D, "far_field")
    match mode:
        case "bounded":
            if tolerance is not None:
                raise ValueError("A bounded far field takes no far_field_tolerance.")
            return mode, None
        case "decaying":
            if tolerance is None:
                raise ValueError("A decaying far field requires far_field_tolerance.")
            return mode, positive_finite_float(tolerance, "far_field_tolerance")
        case _:
            assert_never(mode)


@final
class GalerkinBoundaryComponent(AbstractSpatialComponent, NonTrainableState):
    """Exterior 2-D scalar Laplace Galerkin product as a coupled component.

    The unknowns are the DP0 conormal ``q`` (block ``"conormal"``) and the
    far-field constant ``c`` (block ``"far_field_constant"``); the rows are the
    weak exterior boundary equation (DP0 dual) and the total-conormal
    compatibility row of the product's ``exterior_relation``
    ``((M/2 - K) phi + V q - m c, m^T q)``. The component's own operator is
    the ``(q, c)`` part of that relation; the Dirichlet-trace column
    ``dirichlet_operator = M/2 - K`` (P1 coefficients into the DP0 dual) is
    applied by the coupling law that supplies ``phi`` through a declared trace
    projection. Nothing of the product is reassembled. Both unknowns are
    published as fields (``"far_field_constant"`` pairs with the
    total-conormal row) so laws and observations can certify the exterior
    relation and the far-field behavior.

    ``far_field`` declares the exterior's behavior at infinity as the 2-D
    Galerkin owner does: ``"bounded"`` (``u -> c`` with a free far-field
    constant, the default) or ``"decaying"`` (``u -> 0``), which requires
    ``far_field_tolerance`` and makes the solved ``|c|`` an accepting defect
    of every coupling law acting on this owner: a nonzero constant beyond the
    tolerance is refused, never silently reported.
    """

    name: str = eqx.field(static=True)
    galerkin: ScalarLaplaceGalerkin2D
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]
    far_field: ExteriorFarField2D = eqx.field(static=True)
    far_field_tolerance: float | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        galerkin: ScalarLaplaceGalerkin2D,
        /,
        *,
        far_field: ExteriorFarField2D = "bounded",
        far_field_tolerance: float | None = None,
    ) -> None:
        name_ = canonical_identifier(name, "name")
        if not isinstance(galerkin, _galerkin_type()):
            raise TypeError("galerkin must be a prepared ScalarLaplaceGalerkin2D.")
        mode, tolerance = _far_field_declaration(far_field, far_field_tolerance)
        relation = galerkin.exterior_relation
        source, target = relation.source, relation.target
        if source.names != ("dirichlet_trace", "conormal", "far_field_constant") or (
            target.names != ("exterior_boundary_equation", "total_conormal")
        ):
            raise ValueError("The exterior relation has an unexpected block layout.")
        conormal, constant = source.spaces[1], source.spaces[2]
        if not isinstance(conormal, ArraySpace) or not isinstance(constant, ArraySpace):
            raise TypeError("The conormal and far-field spaces must be ArraySpaces.")
        self.name = name_
        self.galerkin = galerkin
        self.owner_id = galerkin.prepared_id
        self.space = "full"
        self.state_blocks = (
            ComponentBlock("conormal", conormal),
            ComponentBlock("far_field_constant", constant),
        )
        self.row_blocks = (
            ComponentBlock("exterior_boundary_equation", target.spaces[0]),
            ComponentBlock("total_conormal", target.spaces[1]),
        )
        self.fields = (
            ComponentField(
                "conormal",
                state_block="conormal",
                row_block="exterior_boundary_equation",
                full_space=conormal,
                constraint=None,
                free_rows=None,
            ),
            ComponentField(
                "far_field_constant",
                state_block="far_field_constant",
                row_block="total_conormal",
                full_space=constant,
                constraint=None,
                free_rows=None,
            ),
        )
        self.far_field = mode
        self.far_field_tolerance = tolerance

    def dirichlet_operator(self) -> AbstractLinearOperator:
        """``M/2 - K`` from P1 Dirichlet coefficients into the boundary-equation rows."""
        block = self.galerkin.exterior_relation.blocks[0][0]
        if block is None:
            raise ValueError("The exterior relation has no Dirichlet-trace column.")
        return block

    def linear_operator(self, args: object, /) -> BlockLinearOperator:
        del args
        blocks = self.galerkin.exterior_relation.blocks
        return BlockLinearOperator(
            ((blocks[0][1], blocks[0][2]), (blocks[1][1], blocks[1][2])),
            source=self.state_space,
            target=self.row_space,
        )

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        return self.linear_operator(args).mv(state)

    def lift(self, field: str, args: object, /) -> Array:
        del args
        return self.field(field).full_space.zeros()

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        return None

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return ()

    def field_space_id(self, field: str, /) -> str:
        return self.field(field).full_space.space_id


def _product_publication(
    operator: BlockLinearOperator, /
) -> tuple[
    tuple[ComponentBlock, ...], tuple[ComponentBlock, ...], tuple[ComponentField, ...]
]:
    """State blocks, row blocks, and fields of a prepared block product, verbatim.

    Every block keeps the product's name and space; each unknown block pairs
    with the row block of the same name, which the product tests with that
    unknown's basis.
    """
    source, target = operator.source, operator.target
    if source.names != target.names:
        raise ValueError("The product's row blocks do not pair with its unknown blocks.")
    fields: list[ComponentField] = []
    for name, space in zip(source.names, source.spaces, strict=True):
        if not isinstance(space, ArraySpace):
            raise ValueError(
                f"The product's block {name!r} is not one coefficient array field."
            )
        fields.append(
            ComponentField(
                name,
                state_block=name,
                row_block=name,
                full_space=space,
                constraint=None,
                free_rows=None,
            )
        )
    return (
        tuple(
            ComponentBlock(name, space)
            for name, space in zip(source.names, source.spaces, strict=True)
        ),
        tuple(
            ComponentBlock(name, space)
            for name, space in zip(target.names, target.spaces, strict=True)
        ),
        tuple(fields),
    )


@final
class ScalarLaplaceFEMBEMArguments(StrictModule):
    """Runtime data of one ``ScalarLaplaceFEMBEMComponent`` evaluation.

    The members are the runtime arguments of
    ``solve_scalar_laplace_fem_bem_3d``: P1 volume-source coefficients, DP0
    Dirichlet and conormal transmission jumps (``None`` is zero), and one
    interior conductivity per tetrahedron (``None`` is unit conductivity).
    The product validates their spaces when it evaluates them.
    """

    volume_source_coefficients: Array
    dirichlet_jump: Array | None
    conormal_jump: Array | None
    conductivity: Array | None

    def __init__(
        self,
        volume_source_coefficients: ArrayLike,
        /,
        *,
        dirichlet_jump: ArrayLike | None = None,
        conormal_jump: ArrayLike | None = None,
        conductivity: ArrayLike | None = None,
    ) -> None:
        self.volume_source_coefficients = jnp.asarray(volume_source_coefficients)
        self.dirichlet_jump = (
            None if dirichlet_jump is None else jnp.asarray(dirichlet_jump)
        )
        self.conormal_jump = None if conormal_jump is None else jnp.asarray(conormal_jump)
        self.conductivity = None if conductivity is None else jnp.asarray(conductivity)


@final
class ScalarLaplaceFEMBEMComponent(AbstractSpatialComponent, NonTrainableState):
    """Prepared 3-D scalar Johnson--Nédélec FEM--BEM product as a coupled component.

    The state and row blocks are the product's own ``"interior_field"`` (P1
    coefficients; interior weak rows) and ``"exterior_conormal"`` (DP0
    conormal; Calderón rows), each published as a field. ``linear_operator``
    is the product's block operator; a runtime conductivity replaces only its
    interior stiffness through ``product.stiffness``, exactly as the product's
    own solve rebinds it. ``residual`` is ``A x - b`` with ``b`` from
    ``product.right_hand_side``; its rows are NaN for a conductivity the
    product does not accept (nonpositive or nonfinite), so the coupled
    certificate is withheld as the product withholds ``valid``. Arguments are
    ``ScalarLaplaceFEMBEMArguments``; ``None`` evaluates the prepared operator
    only. Nothing is reassembled, the lift is zero, no impositions are
    published, and no kernel is declared (the product publishes none).
    """

    name: str = eqx.field(static=True)
    product: PreparedScalarLaplaceFEMBEM3D
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    @checked
    def __init__(self, name: str, product: PreparedScalarLaplaceFEMBEM3D, /) -> None:
        name_ = canonical_identifier(name, "name")
        state_blocks, row_blocks, fields = _product_publication(product.operator)
        self.name = name_
        self.product = product
        self.owner_id = product.prepared_id
        self.space = "full"
        self.state_blocks = state_blocks
        self.row_blocks = row_blocks
        self.fields = fields

    def _conductivity(self, args: object, /) -> Array | None:
        match args:
            case None:
                return None
            case ScalarLaplaceFEMBEMArguments():
                if args.conductivity is None:
                    return None
                return self.product.conductivity_space.validate(args.conductivity)
            case _:
                raise TypeError("args must be ScalarLaplaceFEMBEMArguments or None.")

    def linear_operator(self, args: object, /) -> BlockLinearOperator:
        conductivity = self._conductivity(args)
        operator = self.product.operator
        if conductivity is None:
            return operator
        blocks = operator.blocks
        return BlockLinearOperator(
            ((self.product.stiffness(conductivity), blocks[0][1]), blocks[1]),
            source=operator.source,
            target=operator.target,
            operator_id=operator.operator_id,
        )

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        if not isinstance(args, ScalarLaplaceFEMBEMArguments):
            raise TypeError("args must be ScalarLaplaceFEMBEMArguments.")
        image = self.linear_operator(args).mv(state)
        load = self.product.right_hand_side(
            args.volume_source_coefficients,
            dirichlet_jump=args.dirichlet_jump,
            conormal_jump=args.conormal_jump,
        )
        rows = tuple(value - rhs for value, rhs in zip(image, load, strict=True))
        if args.conductivity is None:
            return rows
        kappa = self.product.conductivity_space.validate(args.conductivity)
        admissible = jnp.all(jnp.isfinite(kappa) & (kappa > 0.0))
        return tuple(jnp.where(admissible, row, jnp.nan) for row in rows)

    def lift(self, field: str, args: object, /) -> Array:
        del args
        return self.field(field).full_space.zeros()

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        return None

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return ()

    def field_space_id(self, field: str, /) -> str:
        return self.field(field).full_space.space_id


@final
class ElasticityFEMBEMArguments(StrictModule):
    """Runtime loads of one ``ElasticityFEMBEMComponent`` evaluation.

    The members are the two block loads of ``solve_elasticity_fem_bem_3d``;
    the product validates their spaces when it evaluates them.
    """

    interior_load: Array
    boundary_load: Array

    def __init__(self, interior_load: ArrayLike, boundary_load: ArrayLike, /) -> None:
        self.interior_load = jnp.asarray(interior_load)
        self.boundary_load = jnp.asarray(boundary_load)


@final
class ElasticityFEMBEMComponent(AbstractSpatialComponent, NonTrainableState):
    """Prepared 3-D Costabel symmetric elasticity FEM--BEM product as a component.

    The state and row blocks are the product's own ``"interior_displacement"``
    and ``"boundary_traction"``, each published as a field.
    ``linear_operator`` is the product's qualified symmetric block operator
    and ``residual`` is ``A x - b`` with ``b`` from
    ``product.right_hand_side``. Arguments are ``ElasticityFEMBEMArguments``;
    ``None`` evaluates the prepared operator only. Nothing is reassembled, the
    lift is zero, no impositions are published, and no kernel is declared (the
    product publishes none).
    """

    name: str = eqx.field(static=True)
    product: PreparedElasticityFEMBEM3D
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    @checked
    def __init__(self, name: str, product: PreparedElasticityFEMBEM3D, /) -> None:
        name_ = canonical_identifier(name, "name")
        state_blocks, row_blocks, fields = _product_publication(product.operator)
        self.name = name_
        self.product = product
        self.owner_id = product.prepared_id
        self.space = "full"
        self.state_blocks = state_blocks
        self.row_blocks = row_blocks
        self.fields = fields

    def linear_operator(self, args: object, /) -> BlockLinearOperator:
        if args is not None and not isinstance(args, ElasticityFEMBEMArguments):
            raise TypeError("args must be ElasticityFEMBEMArguments or None.")
        return self.product.operator

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        if not isinstance(args, ElasticityFEMBEMArguments):
            raise TypeError("args must be ElasticityFEMBEMArguments.")
        image = self.product.operator.mv(state)
        load = self.product.right_hand_side(args.interior_load, args.boundary_load)
        return tuple(value - rhs for value, rhs in zip(image, load, strict=True))

    def lift(self, field: str, args: object, /) -> Array:
        del args
        return self.field(field).full_space.zeros()

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        return None

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return ()

    def field_space_id(self, field: str, /) -> str:
        return self.field(field).full_space.space_id


__all__ = [
    "AbstractCapacityComponent",
    "AbstractPreparedCapacity",
    "AbstractReconstructionComponent",
    "AbstractSpatialComponent",
    "AbstractTraceComponent",
    "ComponentBlock",
    "ComponentField",
    "ComponentSpace",
    "ElasticityFEMBEMArguments",
    "ElasticityFEMBEMComponent",
    "GalerkinBoundaryComponent",
    "ScalarLaplaceFEMBEMArguments",
    "ScalarLaplaceFEMBEMComponent",
    "VariationalCapacity",
    "VariationalComponent",
]

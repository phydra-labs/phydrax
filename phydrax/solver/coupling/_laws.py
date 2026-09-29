#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical coupling laws and their numerical impositions.

A law states physics (for example continuity of a scalar potential and balance
of its conormal flux across a two-sided interface); an imposition states how
that physics enters the discrete problem. Lowering a law produces typed
contributions, the facet/row impositions it claims on every side, host
evidence (coverage, rank, stability, relation residuals), and a certificate
that measures the law's interface defects at a solution.

Laws never reassemble an owner's operator. Values and loads flow only through
the owners' prepared side traces and their exact dual pullbacks; fluxes come
from the owners' physical conormal-flux publications. Nothing here depends on
which numerical methods sit on the two sides.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import uncancelled_direction
from ..._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
from ...discretization import (
    FacetTraceRule,
    FieldTransfer,
    IntegrationDomain,
    PreparedFluxAction,
    PreparedTraceAction,
    TraceInverseEvidence,
)
from ...geometry.surface import InterfaceSide
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DualSpace,
    FunctionLinearOperator,
)
from ...system_modeling import AcausalSystem, compile_linear_acausal_system
from ...typing import parse
from ._components import (
    AbstractSpatialComponent,
    AbstractTraceComponent,
    GalerkinBoundaryComponent,
)
from ._contributions import (
    AbstractContributionResidual,
    Contribution,
    ContributionEndpoint,
    EliminationContribution,
    LawBlock,
    LawImposition,
    LinearContribution,
    LoadContribution,
    ResidualContribution,
)
from ._interface_quadrature import (
    InterfaceCoverageEvidence,
    InterfaceQuadrature,
    InterfaceQuadraturePolicy,
    InterfaceSideQuadrature,
    prepare_boundary_panel_resampling,
    prepare_interface_quadrature,
)
from ._interfaces import InterfaceBinding, InterfaceOwner
from ._parameters import RuntimeInput


if TYPE_CHECKING:
    from ...operators.integral.layer_potential import (
        BoundaryTraceProjection2D,
        PreparedExteriorLaplaceDirichlet2D,
        ScalarLaplaceGalerkin2D,
    )


MultiplierFamily: TypeAlias = Literal["side-trace", "discontinuous-polynomial"]


# --- Certificates -----------------------------------------------------------------


@final
class InterfaceDefectReport(StrictModule):
    """Measured interface defects of one law at one solution.

    ``values`` are absolute defects and ``scales`` their reference magnitudes
    in the same order as ``names``. ``gated`` marks the defects that decide
    acceptance (``value <= tolerance * scale``); the others (for example the
    trace mismatch of a nonconforming mortar, a discretization error) are
    evidence only.
    """

    law_id: str = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    gated: tuple[bool, ...] = eqx.field(static=True)
    values: Array
    scales: Array

    def __init__(
        self,
        law_id: str,
        names: tuple[str, ...],
        gated: tuple[bool, ...],
        values: Array,
        scales: Array,
        /,
    ) -> None:
        if len(names) != len(gated) or values.shape != (len(names),):
            raise ValueError("Every interface defect needs a name, gate, and scale.")
        if scales.shape != values.shape:
            raise ValueError("Defect scales must match the defects.")
        self.law_id = canonical_identifier(law_id, "law_id")
        self.names = names
        self.gated = gated
        self.values = values
        self.scales = scales

    def value(self, name: str, /) -> Array:
        return self.values[self.names.index(name)]

    def accepted(self, tolerance: float, /) -> Array:
        """Whether every gated defect is within ``tolerance`` relative to its scale."""
        mask = jnp.asarray(self.gated, dtype=jnp.bool_)
        passing = self.values <= tolerance * self.scales
        return jnp.all(jnp.where(mask, passing, True)) & jnp.all(
            jnp.isfinite(self.values)
        )


type FieldStates = Mapping[tuple[str, str], Array]


class AbstractLawCertificate(StrictModule):
    """Law-owned evaluation of interface defects at a candidate solution.

    ``fields`` maps ``(component, field)`` to full coefficients, ``law_state``
    holds the law's own unknowns, and ``args`` the per-component arguments.
    """

    @abc.abstractmethod
    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        raise NotImplementedError


@final
class PreparedLaw(StrictModule):
    """Lowered law: its blocks, contributions, claimed impositions, and evidence."""

    law_id: str = eqx.field(static=True)
    binding_id: str | None = eqx.field(static=True)
    state_blocks: tuple[LawBlock, ...]
    row_blocks: tuple[LawBlock, ...]
    contributions: tuple[Contribution, ...]
    impositions: tuple[LawImposition, ...]
    certificate: AbstractLawCertificate
    evidence: StrictModule

    def __init__(
        self,
        law_id: str,
        /,
        *,
        binding_id: str | None,
        state_blocks: tuple[LawBlock, ...],
        row_blocks: tuple[LawBlock, ...],
        contributions: tuple[Contribution, ...],
        impositions: tuple[LawImposition, ...],
        certificate: AbstractLawCertificate,
        evidence: StrictModule,
    ) -> None:
        if len(state_blocks) != len(row_blocks):
            raise ValueError(
                "A law owns as many row blocks as unknown blocks so the coupled "
                "problem stays square."
            )
        if not isinstance(certificate, AbstractLawCertificate):
            raise TypeError("certificate must be an AbstractLawCertificate.")
        self.law_id = canonical_identifier(law_id, "law_id")
        self.binding_id = binding_id
        self.state_blocks = state_blocks
        self.row_blocks = row_blocks
        self.contributions = contributions
        self.impositions = impositions
        self.certificate = certificate
        self.evidence = evidence


class AbstractCouplingLaw(StrictModule):
    """Declared physical law with its chosen numerical imposition."""

    law_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        """Interface bindings the law acts on (declared by the plan)."""
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        """Lower the law through its components; ``interface_owners`` are the
        current geometry/mesh owners its bindings' endpoints are audited against."""
        raise NotImplementedError


# --- Impositions ---------------------------------------------------------------------


@final
class MatchingElimination(StrictModule, NonTrainableState):
    """Conforming elimination of one side's interface rows by the other's.

    The eliminated side's free trace rows are expressed through an explicit
    basis relation ``E`` computed from both owners' trace routes at a common
    quadrature (never from node-coordinate equality). The facet partitions
    must coincide; ``relation_tolerance`` bounds the relative residual
    ``||T_e E - T_r||`` that proves the two trace spaces coincide.
    """

    eliminated: str = eqx.field(static=True)
    relation_tolerance: float = eqx.field(static=True)
    max_evidence_entries: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        eliminated: str,
        relation_tolerance: float = 1.0e-10,
        max_evidence_entries: int = 16_000_000,
    ) -> None:
        self.eliminated = canonical_identifier(eliminated, "eliminated")
        self.relation_tolerance = positive_finite_float(
            relation_tolerance, "relation_tolerance"
        )
        self.max_evidence_entries = positive_integer(
            max_evidence_entries, "max_evidence_entries"
        )


@final
class MortarMultiplier(StrictModule, NonTrainableState):
    """Declared multiplier space of a mortar imposition.

    ``"side-trace"`` uses the trace basis of the ``side`` endpoint on its rows
    that the owner does not impose strongly, with every strongly imposed end
    function merged equally into the free functions sharing its facets (the
    standard crosspoint-modified mortar space, which reproduces constants on
    the end facets).
    ``"discontinuous-polynomial"`` uses Legendre polynomials of ``degree`` on
    every facet of the ``side`` endpoint.
    """

    family: MultiplierFamily = eqx.field(static=True)
    side: str = eqx.field(static=True)
    degree: int | None = eqx.field(static=True)

    def __init__(
        self, family: MultiplierFamily, /, *, side: str, degree: int | None = None
    ) -> None:
        family_ = parse(family, MultiplierFamily, "family")
        match family_:
            case "side-trace":
                if degree is not None:
                    raise ValueError("A side-trace multiplier takes its side's degree.")
            case "discontinuous-polynomial":
                if degree is None:
                    raise ValueError("A discontinuous multiplier declares its degree.")
                degree = nonnegative_integer(degree, "degree")
        self.family = family_
        self.side = canonical_identifier(side, "side")
        self.degree = degree


@final
class MortarImposition(StrictModule, NonTrainableState):
    """Lagrange-multiplier imposition with rank and stability evidence.

    Preparation refuses a multiplier space whose constraint matrix on the free
    trace rows is numerically rank deficient (``sigma_min <= rank_tolerance *
    sigma_max``) or whose discrete L2 inf-sup constant is below
    ``minimum_inf_sup``. Dependent constraints are never dropped silently.
    ``max_evidence_entries`` bounds the dense interface evidence.
    """

    multiplier: MortarMultiplier
    quadrature: InterfaceQuadraturePolicy
    rank_tolerance: float = eqx.field(static=True)
    minimum_inf_sup: float = eqx.field(static=True)
    max_evidence_entries: int = eqx.field(static=True)

    def __init__(
        self,
        multiplier: MortarMultiplier,
        /,
        *,
        quadrature: InterfaceQuadraturePolicy | None = None,
        rank_tolerance: float = 1.0e-10,
        minimum_inf_sup: float = 1.0e-6,
        max_evidence_entries: int = 16_000_000,
    ) -> None:
        if not isinstance(multiplier, MortarMultiplier):
            raise TypeError("multiplier must be a MortarMultiplier.")
        policy = InterfaceQuadraturePolicy() if quadrature is None else quadrature
        if not isinstance(policy, InterfaceQuadraturePolicy):
            raise TypeError("quadrature must be an InterfaceQuadraturePolicy.")
        self.multiplier = multiplier
        self.quadrature = policy
        self.rank_tolerance = positive_finite_float(rank_tolerance, "rank_tolerance")
        self.minimum_inf_sup = positive_finite_float(minimum_inf_sup, "minimum_inf_sup")
        self.max_evidence_entries = positive_integer(
            max_evidence_entries, "max_evidence_entries"
        )


type TransmissionImposition = MatchingElimination | MortarImposition | NitscheImposition


@final
class TransmissionSide(StrictModule, NonTrainableState):
    """One endpoint of a two-sided interface bound to a component field.

    ``role`` is the endpoint role in the interface binding, ``domain`` the
    component-owned exterior-facet domain that discretizes the interface.
    """

    role: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    domain: IntegrationDomain

    def __init__(
        self, role: str, component: str, field: str, domain: IntegrationDomain, /
    ) -> None:
        if not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be an IntegrationDomain.")
        if domain.kind != "exterior_facet":
            raise ValueError(
                "Transmission sides act on exterior facets of their component's "
                "support (the interface is the boundary of each side's region)."
            )
        self.role = canonical_identifier(role, "role")
        self.component = canonical_identifier(component, "component")
        self.field = canonical_identifier(field, "field")
        self.domain = domain


# --- Evidence ------------------------------------------------------------------------


@final
class MortarEvidence(StrictModule, NonTrainableState):
    """Rank, stability, degree, and coverage evidence of one mortar."""

    multiplier_family: MultiplierFamily = eqx.field(static=True)
    multiplier_side: str = eqx.field(static=True)
    multiplier_degree: int = eqx.field(static=True)
    multiplier_dimension: int = eqx.field(static=True)
    free_trace_rows: tuple[int, int] = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    singular_value_min: float = eqx.field(static=True)
    singular_value_max: float = eqx.field(static=True)
    inf_sup: float = eqx.field(static=True)
    trace_degrees: tuple[int, int] = eqx.field(static=True)
    quadrature_exact_degree: int = eqx.field(static=True)
    coverage: InterfaceCoverageEvidence

    def __init__(
        self,
        *,
        multiplier: MortarMultiplier,
        multiplier_degree: int,
        multiplier_dimension: int,
        free_trace_rows: tuple[int, int],
        singular_values: np.ndarray,
        rank_tolerance: float,
        inf_sup: float,
        trace_degrees: tuple[int, int],
        quadrature: InterfaceQuadrature,
    ) -> None:
        largest = float(singular_values[0])
        self.multiplier_family = multiplier.family
        self.multiplier_side = multiplier.side
        self.multiplier_degree = multiplier_degree
        self.multiplier_dimension = multiplier_dimension
        self.free_trace_rows = free_trace_rows
        self.numerical_rank = int(
            np.count_nonzero(singular_values > rank_tolerance * largest)
        )
        self.singular_value_min = float(singular_values[-1])
        self.singular_value_max = largest
        self.inf_sup = float(inf_sup)
        self.trace_degrees = trace_degrees
        self.quadrature_exact_degree = quadrature.exact_degree
        self.coverage = quadrature.evidence


@final
class EliminationEvidence(StrictModule, NonTrainableState):
    """Relation residual, rank, and coverage evidence of a matching elimination."""

    eliminated_role: str = eqx.field(static=True)
    eliminated_rows: int = eqx.field(static=True)
    retained_columns: int = eqx.field(static=True)
    relation_residual: float = eqx.field(static=True)
    trace_rank: int = eqx.field(static=True)
    trace_degrees: tuple[int, int] = eqx.field(static=True)
    coverage: InterfaceCoverageEvidence

    def __init__(
        self,
        *,
        eliminated_role: str,
        eliminated_rows: int,
        retained_columns: int,
        relation_residual: float,
        trace_rank: int,
        trace_degrees: tuple[int, int],
        coverage: InterfaceCoverageEvidence,
    ) -> None:
        self.eliminated_role = eliminated_role
        self.eliminated_rows = eliminated_rows
        self.retained_columns = retained_columns
        self.relation_residual = float(relation_residual)
        self.trace_rank = trace_rank
        self.trace_degrees = trace_degrees
        self.coverage = coverage


# --- Shared side preparation -----------------------------------------------------


@final
class _SideData(StrictModule):
    """Prepared trace, reaction flux, and row facts of one interface side."""

    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    trace: PreparedTraceAction
    flux: PreparedFluxAction
    support_rows: Array
    free_rows: Array
    full_size: int = eqx.field(static=True)

    def __init__(
        self,
        component: str,
        field: str,
        trace: PreparedTraceAction,
        flux: PreparedFluxAction,
        free_rows: np.ndarray,
        /,
    ) -> None:
        self.component = component
        self.field = field
        self.trace = trace
        self.flux = flux
        self.support_rows = trace.support_rows
        self.free_rows = jnp.asarray(free_rows.astype(np.int32))
        self.full_size = trace.coefficient_space.size

    @property
    def endpoint(self) -> ContributionEndpoint:
        return ContributionEndpoint(self.component, self.field)


def _prepare_trace(
    component: AbstractTraceComponent, side: TransmissionSide, /
) -> PreparedTraceAction:
    """Trace on Gauss-Lobatto sites with enough points to carry its degree."""
    probe = component.prepare_side_trace(
        side.field,
        side.domain,
        rule=FacetTraceRule("gauss-lobatto-legendre", points=2),
    )
    degree = probe.descriptor.trace_degree
    if degree is None:
        raise ValueError(
            f"Side {side.role!r} publishes no polynomial trace degree; the "
            "interface quadrature cannot re-evaluate its trace exactly."
        )
    if degree <= 1:
        trace = probe
    else:
        trace = component.prepare_side_trace(
            side.field,
            side.domain,
            rule=FacetTraceRule("gauss-lobatto-legendre", points=degree + 1),
        )
    if trace.value_shape != () or trace.row_shape != trace.coefficient_space.shape:
        raise ValueError(f"Side {side.role!r} is not a scalar field trace.")
    return trace


def _require_binding(
    binding: InterfaceBinding,
    sides: tuple[TransmissionSide, TransmissionSide],
    components: Mapping[str, AbstractSpatialComponent],
    data: tuple[_SideData, _SideData],
    interface_owners: tuple[InterfaceOwner, ...],
    /,
) -> None:
    """Minus/plus order, explicit field identities, and attached geometry of a
    two-sided binding: each side's prepared trace must lie on its endpoint's
    attached support with its outward normal on the attached side."""
    if binding.incidence != "two-sided":
        raise ValueError("A transmission law needs a two-sided interface binding.")
    expected = (
        binding.side(InterfaceSide.MINUS).role,
        binding.side(InterfaceSide.PLUS).role,
    )
    if (sides[0].role, sides[1].role) != expected:
        raise ValueError(
            f"Transmission sides must follow the binding's (minus, plus) roles "
            f"{expected!r}; the flux multiplier is oriented along its normal."
        )
    for side, datum in zip(sides, data, strict=True):
        endpoint = binding.endpoint(side.role)
        declared = dict(endpoint.fields).get("value")
        identity = components[side.component].field_space_id(side.field)
        if declared != identity:
            raise ValueError(
                f"Endpoint {side.role!r} binds value field {declared!r}, not the "
                f"component field space {identity!r}."
            )
        trace = datum.trace
        endpoint.require_side(
            interface_owners,
            trace.sites,
            trace.normals,
            facets=(trace.descriptor.entity_set_id, trace.descriptor.facets),
        )


def _require_components(
    sides: tuple[TransmissionSide, TransmissionSide],
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> None:
    for side in sides:
        if side.component not in components:
            raise ValueError(f"Law side {side.role!r} names unknown component.")
        component = components[side.component]
        if not isinstance(component, AbstractTraceComponent):
            raise TypeError(
                f"Component {side.component!r} publishes no side traces; a "
                "transmission law acts through facet traces and conormal fluxes."
            )
        component.field(side.field)
    if sides[0].component == sides[1].component:
        raise ValueError("A transmission law couples two different components.")


def _prepare_sides(
    sides: tuple[TransmissionSide, TransmissionSide],
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> tuple[_SideData, _SideData]:
    prepared: list[_SideData] = []
    for side in sides:
        component = components[side.component]
        if not isinstance(component, AbstractTraceComponent):
            raise TypeError(f"Component {side.component!r} publishes no side traces.")
        trace = _prepare_trace(component, side)
        _refuse_owner_facet_laws(component, trace, side.role)
        strong = component.strong_rows(side.field)
        free = np.setdiff1d(np.asarray(trace.support_rows), strong)
        prepared.append(
            _SideData(
                side.component,
                side.field,
                trace,
                component.prepare_conormal_flux(trace),
                free,
            )
        )
    return prepared[0], prepared[1]


def _refuse_owner_facet_laws(
    component: AbstractSpatialComponent, trace: PreparedTraceAction, role: str, /
) -> None:
    """Refuse an owner natural/Robin/weak law already acting on interface facets."""
    for imposition in component.boundary_impositions():
        if imposition.kind != "strong" and imposition.overlaps(trace):
            raise ValueError(
                f"Component {component.name!r} already imposes a {imposition.kind} "
                f"boundary law ({imposition.source_id}) on facets of interface side "
                f"{role!r}; one boundary law is imposed once."
            )


def _dense_side_trace(
    side: InterfaceSideQuadrature, rows: Array, size: int, budget: int, /
) -> np.ndarray:
    """Host matrix ``(points, rows)`` of the side's trace basis at common points."""
    count = rows.shape[0]
    if count * size > budget or count * side.parameters.shape[0] > budget:
        raise ValueError(
            "Dense interface evidence exceeds max_evidence_entries; raise the "
            "declared budget or refine the interface in pieces."
        )
    dtype = side.trace.coefficient_space.dtype

    def column(row: Array) -> Array:
        unit = jnp.zeros((size,), dtype=dtype).at[row].set(1.0)
        return side.values(side.trace.unflatten_rows(unit))

    return np.asarray(jax.vmap(column)(rows)).T


def _refuse_strong_facets(
    matrix: np.ndarray,
    side: InterfaceSideQuadrature,
    data: _SideData,
    role: str,
    /,
) -> None:
    """Refuse an interface facet whose whole trace the owner imposes strongly."""
    free = np.isin(np.asarray(data.support_rows), np.asarray(data.free_rows))
    active = np.abs(matrix) > 1.0e-12 * max(float(np.max(np.abs(matrix))), 1.0)
    facets = np.asarray(side.facets)
    for facet in np.unique(facets):
        rows = np.any(active[facets == facet], axis=0)
        if not np.any(rows & free):
            raise ValueError(
                f"The owner of side {role!r} imposes its whole trace strongly on an "
                "interface facet; a Dirichlet law and a transmission law cannot "
                "both hold there."
            )


def _impositions(
    law_id: str,
    sides: tuple[TransmissionSide, TransmissionSide],
    data: tuple[_SideData, _SideData],
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> tuple[LawImposition, ...]:
    return tuple(
        LawImposition(
            side.component,
            side.field,
            field_space_id=components[side.component].field_space_id(side.field),
            entity_set_id=datum.trace.descriptor.entity_set_id,
            facets=np.asarray(datum.trace.descriptor.facets),
            rows=np.asarray(datum.support_rows),
            imposition_id=canonical_fingerprint(
                {"kind": "law-imposition", "law": law_id, "role": side.role}
            ),
        )
        for side, datum in zip(sides, data, strict=True)
    )


def _trace_degree(trace: PreparedTraceAction, /) -> int:
    degree = trace.descriptor.trace_degree
    if degree is None:
        raise ValueError("Transmission traces must publish a polynomial degree.")
    return degree


# --- Mortar lowering ------------------------------------------------------------------


class _AbstractMultiplierBasis(StrictModule):
    """Multiplier functions evaluated at the common interface points."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def values(self, coefficients: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def moments(self, point_values: Array, /) -> Array:
        """Exact transpose of ``values``."""
        raise NotImplementedError


@final
class _DiscontinuousBasis(_AbstractMultiplierBasis):
    facets: Array
    basis: Array
    facet_count: int = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.facet_count * self.basis.shape[1]

    def values(self, coefficients: Array, /) -> Array:
        local = coefficients.reshape((self.facet_count, self.basis.shape[1]))
        return jnp.sum(self.basis * local[self.facets], axis=1)

    def moments(self, point_values: Array, /) -> Array:
        local = jnp.zeros(
            (self.facet_count, self.basis.shape[1]), dtype=point_values.dtype
        )
        local = local.at[self.facets].add(self.basis * point_values[:, None])
        return local.reshape((-1,))


@final
class _SideTraceBasis(_AbstractMultiplierBasis):
    """Free trace functions with the strongly imposed end functions merged in.

    Each strongly imposed support row ``j`` is distributed over the free rows
    sharing a facet with it (``merged[j] = weights[j] @ lambda``), so the
    multiplier space still reproduces constants on the end facets (the
    standard crosspoint modification of mortar multipliers).
    """

    side: InterfaceSideQuadrature
    rows: Array
    merged_rows: Array
    merge_weights: Array
    full_size: int = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.rows.shape[0]

    def values(self, coefficients: Array, /) -> Array:
        full = jnp.zeros((self.full_size,), dtype=coefficients.dtype)
        full = full.at[self.rows].set(coefficients)
        full = full.at[self.merged_rows].set(self.merge_weights @ coefficients)
        return self.side.values(self.side.trace.unflatten_rows(full))

    def moments(self, point_values: Array, /) -> Array:
        pulled = self.side.trace.flatten_rows(self.side.pullback(point_values))
        return pulled[self.rows] + self.merge_weights.T @ pulled[self.merged_rows]


def _merge_weights(
    trace: np.ndarray, side: InterfaceSideQuadrature, data: _SideData, /
) -> tuple[np.ndarray, np.ndarray]:
    """Rows imposed strongly and their equal split over free rows on shared facets."""
    support = np.asarray(data.support_rows)
    free = np.asarray(data.free_rows)
    active = np.abs(trace) > 1.0e-12 * max(float(np.max(np.abs(trace))), 1.0)
    facets = np.asarray(side.facets)
    merged = support[~np.isin(support, free)]
    weights = np.zeros((merged.size, free.size), dtype=np.float64)
    for index, row in enumerate(merged):
        column = int(np.flatnonzero(support == row)[0])
        shared = np.unique(facets[active[:, column]])
        neighbors = np.any(active[np.isin(facets, shared)], axis=0)
        targets = np.flatnonzero(np.isin(free, support[neighbors]))
        if targets.size:
            weights[index, targets] = 1.0 / targets.size
    return merged, weights


def _legendre_basis(parameters: np.ndarray, degree: int, /) -> np.ndarray:
    """Legendre polynomials ``P_a(2 t - 1)`` for ``a <= degree`` at facet parameters."""
    abscissa = 2.0 * parameters - 1.0
    return np.stack(
        [
            np.polynomial.legendre.legval(abscissa, np.eye(degree + 1)[order])
            for order in range(degree + 1)
        ],
        axis=1,
    )


def _multiplier_basis(
    multiplier: MortarMultiplier,
    side: InterfaceSideQuadrature,
    data: _SideData,
    trace: np.ndarray,
    /,
) -> tuple[_AbstractMultiplierBasis, int]:
    match multiplier.family:
        case "side-trace":
            if data.free_rows.shape[0] == 0:
                raise ValueError("The side-trace multiplier side has no free trace rows.")
            merged, weights = _merge_weights(trace, side, data)
            basis: _AbstractMultiplierBasis = _SideTraceBasis(
                side,
                data.free_rows,
                jnp.asarray(merged.astype(np.int32)),
                jnp.asarray(weights),
                data.full_size,
            )
            return basis, _trace_degree(data.trace)
        case "discontinuous-polynomial":
            degree = multiplier.degree
            if degree is None:
                raise ValueError("A discontinuous multiplier declares its degree.")
            values = _legendre_basis(np.asarray(side.parameters), degree)
            basis = _DiscontinuousBasis(
                side.facets, jnp.asarray(values), data.trace.output_shape[0]
            )
            return basis, degree


def _dense_basis(basis: _AbstractMultiplierBasis, points: int, /) -> np.ndarray:
    identity = jnp.eye(basis.dimension, dtype=jnp.float64)
    return np.asarray(jax.vmap(basis.values)(identity)).T.reshape((points, -1))


def _whitened(matrix: np.ndarray, name: str, /) -> np.ndarray:
    """Cholesky factor of an SPD Gram matrix, refusing a singular one."""
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if eigenvalues[0] <= 1.0e-14 * max(eigenvalues[-1], 1.0e-300):
        raise ValueError(
            f"The {name} Gram matrix on the interface is singular; its basis is not "
            "linearly independent on the common quadrature."
        )
    return np.linalg.cholesky(matrix)


def _mortar_stability(
    blocks: tuple[np.ndarray, np.ndarray],
    traces: tuple[np.ndarray, np.ndarray],
    multiplier: np.ndarray,
    weights: np.ndarray,
    /,
) -> tuple[np.ndarray, float]:
    """Singular values of ``[B_m, -B_p]`` and its discrete L2 inf-sup constant."""
    constraint = np.concatenate((blocks[0], -blocks[1]), axis=1)
    singular = np.linalg.svd(constraint, compute_uv=False)
    if singular.size < constraint.shape[0]:
        singular = np.concatenate(
            (singular, np.zeros((constraint.shape[0] - singular.size,)))
        )
    multiplier_factor = _whitened(
        multiplier.T @ (weights[:, None] * multiplier), "multiplier"
    )
    scaled = np.linalg.solve(multiplier_factor, constraint)
    offset = 0
    columns: list[np.ndarray] = []
    for trace in traces:
        width = trace.shape[1]
        if width:
            factor = _whitened(trace.T @ (weights[:, None] * trace), "trace")
            columns.append(
                np.linalg.solve(factor, scaled[:, offset : offset + width].T).T
            )
        offset += width
    normalized = np.concatenate(columns, axis=1)
    inf_sup = np.linalg.svd(normalized, compute_uv=False)
    smallest = 0.0 if inf_sup.size < normalized.shape[0] else float(inf_sup[-1])
    return singular, smallest


@final
class _MortarActions(StrictModule):
    """``B_s u`` and ``B_s^T lambda`` of both sides through the common quadrature."""

    quadrature: InterfaceQuadrature
    basis: _AbstractMultiplierBasis

    def constraint(self, side: int, coefficients: Array, /) -> Array:
        values = self.quadrature.sides[side].values(coefficients)
        return self.basis.moments(self.quadrature.weights * values)

    def load(self, side: int, multiplier: Array, /) -> Array:
        density = self.quadrature.weights * self.basis.values(multiplier)
        return self.quadrature.sides[side].pullback(density)

    def trace_values(self, side: int, coefficients: Array, /) -> Array:
        return self.quadrature.sides[side].values(coefficients)


def _mortar_operators(
    actions: _MortarActions,
    data: tuple[_SideData, _SideData],
    multiplier_space: ArraySpace,
    law_id: str,
    /,
) -> tuple[Contribution, ...]:
    """Symmetric saddle blocks: rows ``R_m - B_m^T l``, ``R_p + B_p^T l``, ``-B_m u_m + B_p u_p``."""
    multiplier = ContributionEndpoint(law_id, "multiplier", space="reduced")
    constraint = ContributionEndpoint(law_id, "constraint", space="reduced")
    row_space = DualSpace(multiplier_space)
    contributions: list[Contribution] = []
    for index, datum in enumerate(data):
        sign = -1.0 if index == 0 else 1.0
        full = datum.trace.coefficient_space
        imposition = canonical_fingerprint(
            {"kind": "mortar-imposition", "law": law_id, "side": index}
        )

        def load(value: Array, index: int = index, sign: float = sign) -> Array:
            return sign * actions.load(index, value)

        def restrict(value: Array, index: int = index, sign: float = sign) -> Array:
            return sign * actions.constraint(index, value)

        contributions.append(
            LinearContribution(
                datum.endpoint,
                multiplier,
                FunctionLinearOperator(
                    load,
                    source=multiplier_space,
                    target=DualSpace(full),
                    transpose_action=restrict,
                ),
                law_id=law_id,
                imposition_id=imposition,
            )
        )
        contributions.append(
            LinearContribution(
                constraint,
                datum.endpoint,
                FunctionLinearOperator(
                    restrict, source=full, target=row_space, transpose_action=load
                ),
                law_id=law_id,
                imposition_id=imposition,
            )
        )
    return tuple(contributions)


def _l2_mismatch(
    quadrature: InterfaceQuadrature, first: Array, second: Array, /
) -> tuple[Array, Array]:
    left = quadrature.sides[0].values(first)
    right = quadrature.sides[1].values(second)
    mismatch = jnp.sqrt(jnp.sum(quadrature.weights * (left - right) ** 2))
    scale = jnp.sqrt(jnp.sum(quadrature.weights * (left**2 + right**2)))
    return mismatch, scale


def _free_reaction(
    datum: _SideData, full: Array, args: Mapping[str, object], /
) -> tuple[Array, Array]:
    """Owner reaction flux on the interface support rows and their free mask."""
    reaction = datum.flux.evaluate(full, args[datum.component])
    free = jnp.isin(datum.support_rows, datum.free_rows)
    return reaction, free


def _uncancelled_terms[T](action: Callable[..., T], *states: Array) -> T:
    """Linearized ``action`` at ``states`` along their uncancelled direction.

    For a linear or affine action this is ``A (s * v)``, whose norm is the
    root-mean-square magnitude of the individual terms ``A_ij v_j`` that the
    action sums (``uncancelled_direction``). Certificates add it to their
    scales so that an exact state whose terms cancel (a constant in a diffusion
    kernel, a balanced flux) is measured against those terms rather than
    against the cancelled sum. It is evidence scale only and carries no
    derivative.
    """
    _, terms = jax.jvp(action, states, uncancelled_direction(states))
    return jax.lax.stop_gradient(terms)


def _reaction_terms(
    datum: _SideData, full: Array, args: Mapping[str, object], /
) -> Array:
    """Uncancelled terms of the owner reaction on the interface support rows."""
    return _uncancelled_terms(
        lambda value: datum.flux.evaluate(value, args[datum.component]), full
    )


def _owner_row_law(
    data: tuple[_SideData, _SideData],
    states: tuple[Array, Array],
    injected: tuple[Array, ...],
    injected_terms: tuple[Array, ...],
    args: Mapping[str, object],
    /,
) -> tuple[Array, Array]:
    """Squared law residual of injected rows against the owner reactions, and its scale.

    At a solution the owner reaction balances the law rows on free rows. The
    squared scale sums the reactions, the law rows, and the uncancelled terms of
    both, so a balance whose terms cancel is measured against those terms.
    """
    law = jnp.zeros((), dtype=states[0].dtype)
    scale = jnp.zeros((), dtype=states[0].dtype)
    for datum, full, rows, row_terms in zip(
        data, states, injected, injected_terms, strict=True
    ):
        reaction, free = _free_reaction(datum, full, args)
        local = datum.trace.flatten_rows(rows)[datum.support_rows]
        local_terms = datum.trace.flatten_rows(row_terms)[datum.support_rows]
        law = law + jnp.sum(jnp.where(free, reaction + local, 0.0) ** 2)
        scale = (
            scale
            + jnp.sum(jnp.where(free, reaction, 0.0) ** 2)
            + jnp.sum(jnp.where(free, local, 0.0) ** 2)
            + jnp.sum(jnp.where(free, _reaction_terms(datum, full, args), 0.0) ** 2)
            + jnp.sum(jnp.where(free, local_terms, 0.0) ** 2)
        )
    return law, scale


@final
class _MortarCertificate(AbstractLawCertificate):
    """Weak continuity, owner-flux balance, and trace mismatch of a mortar."""

    law_id: str = eqx.field(static=True)
    actions: _MortarActions
    data: tuple[_SideData, _SideData]

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        (multiplier,) = law_state
        first = fields[(self.data[0].component, self.data[0].field)]
        second = fields[(self.data[1].component, self.data[1].field)]
        left = self.actions.constraint(0, first)
        right = self.actions.constraint(1, second)
        weak = jnp.linalg.norm(left - right)
        weak_scale = jnp.linalg.norm(left) + jnp.linalg.norm(right)
        balance = jnp.zeros((), dtype=first.dtype)
        balance_scale = jnp.zeros((), dtype=first.dtype)
        for index, (datum, full) in enumerate(
            zip(self.data, (first, second), strict=True)
        ):
            reaction, free = _free_reaction(datum, full, args)
            terms = _reaction_terms(datum, full, args)
            sign = 1.0 if index == 0 else -1.0
            load = datum.trace.flatten_rows(self.actions.load(index, multiplier))
            injected = sign * load[datum.support_rows]
            balance = balance + jnp.sum(jnp.where(free, reaction - injected, 0.0) ** 2)
            balance_scale = (
                balance_scale
                + jnp.sum(jnp.where(free, reaction, 0.0) ** 2)
                + jnp.sum(jnp.where(free, terms, 0.0) ** 2)
                + jnp.sum(jnp.where(free, injected, 0.0) ** 2)
            )
        mismatch, mismatch_scale = _l2_mismatch(self.actions.quadrature, first, second)
        return InterfaceDefectReport(
            self.law_id,
            ("weak-continuity", "flux-balance", "trace-mismatch-l2"),
            (True, True, False),
            jnp.stack((weak, jnp.sqrt(balance), mismatch)),
            jnp.stack((weak_scale, jnp.sqrt(balance_scale), mismatch_scale)),
        )


def _lower_mortar(
    law_id: str,
    imposition: MortarImposition,
    sides: tuple[TransmissionSide, TransmissionSide],
    data: tuple[_SideData, _SideData],
    /,
) -> tuple[
    tuple[LawBlock, ...],
    tuple[LawBlock, ...],
    tuple[Contribution, ...],
    _MortarCertificate,
    MortarEvidence,
]:
    multiplier = imposition.multiplier
    roles = (sides[0].role, sides[1].role)
    if multiplier.side not in roles:
        raise ValueError(f"Multiplier side {multiplier.side!r} is not a law side.")
    owner = roles.index(multiplier.side)
    degrees = (_trace_degree(data[0].trace), _trace_degree(data[1].trace))
    multiplier_degree = degrees[owner] if multiplier.degree is None else multiplier.degree
    quadrature = prepare_interface_quadrature(
        data[0].trace,
        data[1].trace,
        exact_degree=max(2 * max(degrees), 2 * multiplier_degree, 1),
        policy=imposition.quadrature,
    )
    weights = np.asarray(quadrature.weights)
    traces = tuple(
        _dense_side_trace(
            quadrature.sides[index],
            data[index].support_rows,
            data[index].full_size,
            imposition.max_evidence_entries,
        )
        for index in range(2)
    )
    basis, _ = _multiplier_basis(
        multiplier, quadrature.sides[owner], data[owner], traces[owner]
    )
    for index, role in enumerate(roles):
        _refuse_strong_facets(traces[index], quadrature.sides[index], data[index], role)
    free = tuple(
        traces[index][
            :,
            np.isin(
                np.asarray(data[index].support_rows), np.asarray(data[index].free_rows)
            ),
        ]
        for index in range(2)
    )
    psi = _dense_basis(basis, quadrature.point_count)
    blocks = tuple(psi.T @ (weights[:, None] * matrix) for matrix in free)
    singular, inf_sup = _mortar_stability(
        (blocks[0], blocks[1]), (free[0], free[1]), psi, weights
    )
    evidence = MortarEvidence(
        multiplier=multiplier,
        multiplier_degree=multiplier_degree,
        multiplier_dimension=basis.dimension,
        free_trace_rows=(free[0].shape[1], free[1].shape[1]),
        singular_values=singular,
        rank_tolerance=imposition.rank_tolerance,
        inf_sup=inf_sup,
        trace_degrees=degrees,
        quadrature=quadrature,
    )
    if evidence.numerical_rank < basis.dimension:
        raise ValueError(
            f"The {multiplier.family} multiplier space on side {multiplier.side!r} "
            f"is rank deficient against the free interface traces: numerical rank "
            f"{evidence.numerical_rank} of {basis.dimension} (sigma_min/sigma_max = "
            f"{evidence.singular_value_min / evidence.singular_value_max:.3e}). "
            "Dependent constraints are not removed silently; declare a smaller "
            "multiplier space."
        )
    if inf_sup < imposition.minimum_inf_sup:
        raise ValueError(
            f"The mortar's discrete L2 inf-sup constant {inf_sup:.3e} is below the "
            f"declared minimum {imposition.minimum_inf_sup:.3e}."
        )
    space = ArraySpace((basis.dimension,), dtype=data[0].trace.coefficient_space.dtype)
    actions = _MortarActions(quadrature, basis)
    contributions = _mortar_operators(actions, data, space, law_id)
    certificate = _MortarCertificate(law_id, actions, data)
    return (
        (LawBlock("multiplier", space),),
        (LawBlock("constraint", DualSpace(space)),),
        contributions,
        certificate,
        evidence,
    )


# --- Matching elimination --------------------------------------------------------------


def _coincident_facets(quadrature: InterfaceQuadrature, /) -> None:
    coverage = quadrature.evidence
    segments = coverage.segment_count
    facets = (
        np.asarray(coverage.first_coverage).size,
        np.asarray(coverage.second_coverage).size,
    )
    if segments != facets[0] or segments != facets[1]:
        raise ValueError(
            "Matching elimination requires coincident facet partitions on the "
            f"interface ({facets[0]} and {facets[1]} facets meet in {segments} "
            "segments); declare a mortar for a nonmatching interface."
        )


@final
class _EliminationCertificate(AbstractLawCertificate):
    """Trace continuity and summed reaction balance of a matching elimination."""

    law_id: str = eqx.field(static=True)
    quadrature: InterfaceQuadrature
    data: tuple[_SideData, _SideData]
    eliminated: int = eqx.field(static=True)
    relation: Array

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        del law_state
        first = fields[(self.data[0].component, self.data[0].field)]
        second = fields[(self.data[1].component, self.data[1].field)]
        mismatch, mismatch_scale = _l2_mismatch(self.quadrature, first, second)
        states = (first, second)
        eliminated = self.data[self.eliminated]
        retained = self.data[1 - self.eliminated]
        reaction_e, free_e = _free_reaction(eliminated, states[self.eliminated], args)
        reaction_r, free_r = _free_reaction(retained, states[1 - self.eliminated], args)
        summed = reaction_r + jnp.where(free_e, reaction_e, 0.0) @ self.relation
        balance = jnp.linalg.norm(jnp.where(free_r, summed, 0.0))
        terms_e = _reaction_terms(eliminated, states[self.eliminated], args)
        terms_r = _reaction_terms(retained, states[1 - self.eliminated], args)
        scale = (
            jnp.linalg.norm(jnp.where(free_r, reaction_r, 0.0))
            + jnp.linalg.norm(jnp.where(free_e, reaction_e, 0.0))
            + jnp.linalg.norm(jnp.where(free_r, terms_r, 0.0))
            + jnp.linalg.norm(jnp.where(free_e, terms_e, 0.0))
        )
        return InterfaceDefectReport(
            self.law_id,
            ("trace-continuity-l2", "flux-balance"),
            (True, True),
            jnp.stack((mismatch, balance)),
            jnp.stack((mismatch_scale, scale)),
        )


def _lower_elimination(
    law_id: str,
    imposition: MatchingElimination,
    sides: tuple[TransmissionSide, TransmissionSide],
    data: tuple[_SideData, _SideData],
    /,
) -> tuple[
    tuple[LawBlock, ...],
    tuple[LawBlock, ...],
    tuple[Contribution, ...],
    _EliminationCertificate,
    EliminationEvidence,
]:
    roles = (sides[0].role, sides[1].role)
    if imposition.eliminated not in roles:
        raise ValueError(f"Eliminated side {imposition.eliminated!r} is not a law side.")
    eliminated = roles.index(imposition.eliminated)
    if any(
        datum.trace.row_shape != datum.trace.coefficient_space.shape[:1] for datum in data
    ):
        raise ValueError(
            "Matching elimination addresses one-axis coefficient rows; a side with a "
            "tensor row layout couples through a mortar imposition."
        )
    degrees = (_trace_degree(data[0].trace), _trace_degree(data[1].trace))
    quadrature = prepare_interface_quadrature(
        data[0].trace, data[1].trace, exact_degree=max(2 * max(degrees), 1)
    )
    _coincident_facets(quadrature)
    traces = tuple(
        _dense_side_trace(
            quadrature.sides[index],
            data[index].support_rows,
            data[index].full_size,
            imposition.max_evidence_entries,
        )
        for index in range(2)
    )
    for index, role in enumerate(roles):
        _refuse_strong_facets(traces[index], quadrature.sides[index], data[index], role)
    weights = np.sqrt(np.asarray(quadrature.weights))[:, None]
    source, target = traces[eliminated] * weights, traces[1 - eliminated] * weights
    singular = np.linalg.svd(source, compute_uv=False)
    rank = int(np.count_nonzero(singular > 1.0e-12 * singular[0]))
    if rank < source.shape[1]:
        raise ValueError(
            "The eliminated side's trace basis is not unisolvent on the interface "
            f"(rank {rank} of {source.shape[1]}); no explicit relation exists."
        )
    relation, *_ = np.linalg.lstsq(source, target, rcond=None)
    residual = float(np.linalg.norm(source @ relation - target)) / max(
        float(np.linalg.norm(target)), 1.0e-300
    )
    if residual > imposition.relation_tolerance:
        raise ValueError(
            "The two sides' interface trace spaces differ (relative relation "
            f"residual {residual:.3e}); matching elimination needs an exact basis "
            "relation. Declare a mortar instead."
        )
    relation[np.abs(relation) <= 1.0e-13 * np.max(np.abs(relation))] = 0.0
    support = np.asarray(data[eliminated].support_rows)
    free = np.isin(support, np.asarray(data[eliminated].free_rows))
    retained = data[1 - eliminated]
    retained_free = np.isin(
        np.asarray(retained.support_rows), np.asarray(retained.free_rows)
    )
    # Only free eliminated rows are replaced by the relation. A strongly
    # constrained eliminated row (a crosspoint carrying the owner's Dirichlet
    # value) keeps its native constraint, so continuity at that row holds only
    # if every retained row it relates to is constrained as well; a free
    # retained row there would never see the eliminated side's value.
    if np.any(relation[~free][:, retained_free] != 0.0):
        raise ValueError(
            f"Eliminated side {imposition.eliminated!r} strongly constrains "
            "interface rows (for example a crosspoint Dirichlet value) whose "
            f"retained side {roles[1 - eliminated]!r} rows are free; eliminating "
            "it would drop continuity there. Eliminate the other side or declare "
            "a mortar or Nitsche imposition."
        )
    contribution = EliminationContribution(
        data[eliminated].endpoint,
        data[1 - eliminated].endpoint,
        support[free],
        np.asarray(data[1 - eliminated].support_rows),
        relation[free],
        law_id=law_id,
        imposition_id=canonical_fingerprint(
            {"kind": "matching-elimination", "law": law_id}
        ),
    )
    certificate = _EliminationCertificate(
        law_id, quadrature, data, eliminated, jnp.asarray(relation)
    )
    evidence = EliminationEvidence(
        eliminated_role=imposition.eliminated,
        eliminated_rows=int(np.count_nonzero(free)),
        retained_columns=int(target.shape[1]),
        relation_residual=residual,
        trace_rank=rank,
        trace_degrees=degrees,
        coverage=quadrature.evidence,
    )
    return (), (), (contribution,), certificate, evidence


# --- Nitsche imposition -----------------------------------------------------------------


NitscheVariant: TypeAlias = Literal["symmetric", "nonsymmetric"]


@final
class NitscheImposition(StrictModule, NonTrainableState):
    """Nitsche imposition of scalar transmission with a certified penalty.

    With the minus outward normal, the owners' exact pointwise outward fluxes
    ``q_minus``/``q_plus``, the averaged flux leaving the minus side
    ``{q} = w_minus q_minus - w_plus q_plus`` (``weights`` sum to one) and the
    jump ``[u] = u_minus - u_plus``, the law adds
    ``-int {q(u)} [v] - theta int {q(v)} [u] + int gamma [u] [v]`` to the
    owners' weak residuals, with ``theta = 1`` (``"symmetric"``, adjoint
    consistent) or ``theta = -1`` (``"nonsymmetric"``). At every common
    interface point the penalty density is ``gamma = penalty_factor * C`` with
    ``C = sum_s w_s^2 m_s C_s``: the owners' certified trace-inverse constants
    ``C_s`` of the facets containing the point and the counts ``m_s`` of the
    law's interface facets sharing their side cells. The symmetric form is
    coercive for ``penalty_factor > 1`` with constant ``1 - penalty_factor**-0.5``
    in the energy-plus-penalty norm, and preparation refuses a smaller factor;
    the nonsymmetric form is coercive for every positive factor. A side with
    zero weight contributes no flux, so it needs neither a pointwise flux nor
    stability evidence (one-sided Nitsche). The certificate covers the law's
    own terms against the owners' diffusion energies; cells shared with other
    weak interface laws are not accounted.
    """

    variant: NitscheVariant = eqx.field(static=True)
    penalty_factor: float = eqx.field(static=True)
    weights: tuple[float, float] = eqx.field(static=True)
    quadrature: InterfaceQuadraturePolicy
    max_evidence_entries: int = eqx.field(static=True)

    def __init__(
        self,
        variant: NitscheVariant = "symmetric",
        /,
        *,
        penalty_factor: float,
        weights: tuple[float, float] = (0.5, 0.5),
        quadrature: InterfaceQuadraturePolicy | None = None,
        max_evidence_entries: int = 16_000_000,
    ) -> None:
        variant_ = parse(variant, NitscheVariant, "variant")
        factor = positive_finite_float(penalty_factor, "penalty_factor")
        if not isinstance(weights, tuple) or len(weights) != 2:
            raise TypeError("weights must be a (minus, plus) pair of floats.")
        pair = (float(weights[0]), float(weights[1]))
        if not all(np.isfinite(value) and value >= 0.0 for value in pair) or not (
            abs(pair[0] + pair[1] - 1.0) <= 1.0e-12
        ):
            raise ValueError("Nitsche flux weights must be nonnegative and sum to one.")
        policy = InterfaceQuadraturePolicy() if quadrature is None else quadrature
        if not isinstance(policy, InterfaceQuadraturePolicy):
            raise TypeError("quadrature must be an InterfaceQuadraturePolicy.")
        self.variant = variant_
        self.penalty_factor = factor
        self.weights = pair
        self.quadrature = policy
        self.max_evidence_entries = positive_integer(
            max_evidence_entries, "max_evidence_entries"
        )

    @property
    def theta(self) -> float:
        match self.variant:
            case "symmetric":
                return 1.0
            case "nonsymmetric":
                return -1.0
            case _:
                assert_never(self.variant)

    @property
    def coercivity_constant(self) -> float:
        """Certified coercivity constant in the energy-plus-penalty norm (0 if none)."""
        match self.variant:
            case "symmetric":
                return max(1.0 - self.penalty_factor**-0.5, 0.0)
            case "nonsymmetric":
                return 1.0
            case _:
                assert_never(self.variant)


@final
class NitscheEvidence(StrictModule, NonTrainableState):
    """Stability, degree, and coverage evidence of one Nitsche imposition.

    ``penalty_range`` is the smallest and largest penalty density at the
    common points, ``stability`` the owners' certified trace-inverse evidence
    (``None`` for a side with zero flux weight), and ``flux_degrees`` the
    published facet degrees of the fluxes that enter.
    """

    variant: NitscheVariant = eqx.field(static=True)
    penalty_factor: float = eqx.field(static=True)
    weights: tuple[float, float] = eqx.field(static=True)
    coercivity_constant: float = eqx.field(static=True)
    penalty_range: tuple[float, float] = eqx.field(static=True)
    trace_degrees: tuple[int, int] = eqx.field(static=True)
    flux_degrees: tuple[int | None, int | None] = eqx.field(static=True)
    quadrature_exact_degree: int = eqx.field(static=True)
    stability: tuple[TraceInverseEvidence | None, TraceInverseEvidence | None]
    coverage: InterfaceCoverageEvidence

    def __init__(
        self,
        imposition: NitscheImposition,
        /,
        *,
        penalty: np.ndarray,
        trace_degrees: tuple[int, int],
        flux_degrees: tuple[int | None, int | None],
        stability: tuple[TraceInverseEvidence | None, TraceInverseEvidence | None],
        quadrature: InterfaceQuadrature,
    ) -> None:
        self.variant = imposition.variant
        self.penalty_factor = imposition.penalty_factor
        self.weights = imposition.weights
        self.coercivity_constant = imposition.coercivity_constant
        self.penalty_range = (float(np.min(penalty)), float(np.max(penalty)))
        self.trace_degrees = trace_degrees
        self.flux_degrees = flux_degrees
        self.quadrature_exact_degree = quadrature.exact_degree
        self.stability = stability
        self.coverage = quadrature.evidence


def _component_args(args: object, component: str, /) -> object:
    """One owner's arguments from the per-component mapping (None when unbound)."""
    if args is None:
        return None
    if not isinstance(args, Mapping):
        raise TypeError("Law rows take per-component arguments.")
    return args[component]


@final
class _NitscheResidual(AbstractContributionResidual):
    """Both owners' Nitsche rows through the common interface quadrature."""

    quadrature: InterfaceQuadrature
    fluxes: tuple[PreparedFluxAction | None, PreparedFluxAction | None]
    penalty: Array
    components: tuple[str, str] = eqx.field(static=True)
    weights: tuple[float, float] = eqx.field(static=True)
    theta: float = eqx.field(static=True)

    def flux(self, index: int, full: Array, args: object, /) -> Array:
        """Outward flux of one side at the common points (zero without weight)."""
        flux = self.fluxes[index]
        if flux is None:
            return jnp.zeros((self.quadrature.point_count,), dtype=full.dtype)
        side_args = _component_args(args, self.components[index])
        return self.quadrature.sides[index].resampling.apply(
            flux.evaluate(full, side_args)
        )

    def flux_pullback(
        self, index: int, full: Array, density: Array, args: object, /
    ) -> Array:
        """Exact transpose of ``flux`` onto the owner's full rows."""
        flux = self.fluxes[index]
        if flux is None:
            return jnp.zeros_like(full)
        side_args = _component_args(args, self.components[index])
        sites = self.quadrature.sides[index].resampling.transpose(density)
        (rows,) = jax.linear_transpose(
            lambda value: flux.evaluate(value, side_args),
            jax.ShapeDtypeStruct(full.shape, full.dtype),
        )(sites)
        return rows

    def average(self, minus: Array, plus: Array, args: object, /) -> Array:
        """``{q}``: the averaged flux leaving the minus side at the common points."""
        minus_flux = self.flux(0, minus, args)
        return self.weights[0] * minus_flux - self.weights[1] * self.flux(1, plus, args)

    def jump(self, minus: Array, plus: Array, /) -> Array:
        sides = self.quadrature.sides
        return sides[0].values(minus) - sides[1].values(plus)

    def evaluate(self, inputs: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        minus, plus = inputs
        sides = self.quadrature.sides
        weights = self.quadrature.weights
        jump = self.jump(minus, plus)
        # The numerical flux {q} - gamma [u] leaves the minus side and enters the plus.
        exchanged = weights * (self.average(minus, plus, args) - self.penalty * jump)
        adjoint = weights * jump
        return (
            -sides[0].pullback(exchanged)
            - self.theta * self.weights[0] * self.flux_pullback(0, minus, adjoint, args),
            sides[1].pullback(exchanged)
            + self.theta * self.weights[1] * self.flux_pullback(1, plus, adjoint, args),
        )


@final
class _NitscheCertificate(AbstractLawCertificate):
    """Owner-row law residual, conservation of the rows, and the trace jump."""

    law_id: str = eqx.field(static=True)
    residual: _NitscheResidual
    data: tuple[_SideData, _SideData]

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        del law_state
        first = fields[(self.data[0].component, self.data[0].field)]
        second = fields[(self.data[1].component, self.data[1].field)]
        injected = self.residual.evaluate((first, second), args)
        injected_terms = _uncancelled_terms(
            lambda minus, plus: self.residual.evaluate((minus, plus), args),
            first,
            second,
        )
        law, law_scale = _owner_row_law(
            self.data, (first, second), injected, injected_terms, args
        )
        # Constants carry no flux and the partition of unity sums every row, so the
        # rows of the two owners cancel exactly: the numerical flux is conservative.
        conservation = jnp.abs(jnp.sum(injected[0]) + jnp.sum(injected[1]))
        quadrature = self.residual.quadrature
        exchanged = self.residual.average(
            first, second, args
        ) - self.residual.penalty * self.residual.jump(first, second)
        conservation_scale = jnp.sum(quadrature.weights * jnp.abs(exchanged)) + sum(
            (jnp.sum(jnp.abs(terms)) for terms in injected_terms),
            jnp.zeros((), dtype=first.dtype),
        )
        jump, jump_scale = _l2_mismatch(quadrature, first, second)
        return InterfaceDefectReport(
            self.law_id,
            ("law-residual", "flux-conservation", "trace-jump-l2"),
            (True, True, False),
            jnp.stack((jnp.sqrt(law), conservation, jump)),
            jnp.stack((jnp.sqrt(law_scale), conservation_scale, jump_scale)),
        )


def _pointwise_side(
    component: AbstractTraceComponent,
    side: TransmissionSide,
    datum: _SideData,
    /,
) -> tuple[_SideData, PreparedFluxAction, TraceInverseEvidence]:
    """Exact pointwise flux and its certified stability on the side's GLL trace.

    The trace is re-prepared with more sites when the flux's facet degree
    exceeds the value trace's, so that both resample exactly.
    """
    flux = component.prepare_pointwise_flux(datum.trace)
    descriptor = flux.descriptor
    if (
        descriptor.representation != "quadrature-values"
        or descriptor.approximation != "exact"
        or descriptor.orientation != "outward"
    ):
        raise ValueError(
            f"Side {side.role!r} publishes a {descriptor.approximation} "
            f"{descriptor.representation} flux; Nitsche consistency needs the exact "
            "pointwise outward flux of the owner's physical operator."
        )
    degree = descriptor.trace_degree
    if degree is None:
        raise ValueError(
            f"The pointwise flux of side {side.role!r} publishes no polynomial facet "
            "degree; the interface quadrature cannot re-evaluate it exactly."
        )
    if degree > datum.trace.output_shape[1] - 1:
        trace = component.prepare_side_trace(
            side.field,
            side.domain,
            rule=FacetTraceRule("gauss-lobatto-legendre", points=degree + 1),
        )
        datum = _SideData(
            datum.component,
            datum.field,
            trace,
            component.prepare_conormal_flux(trace),
            np.asarray(datum.free_rows),
        )
        flux = component.prepare_pointwise_flux(trace)
    stability = component.certify_flux_stability(flux)
    if stability.flux_action_id != flux.action_id or not np.array_equal(
        np.asarray(stability.facets), np.asarray(flux.descriptor.facets)
    ):
        raise ValueError(
            f"The stability evidence of side {side.role!r} certifies another flux."
        )
    return datum, flux, stability


def _nitsche_penalty(
    imposition: NitscheImposition,
    quadrature: InterfaceQuadrature,
    stability: tuple[TraceInverseEvidence | None, TraceInverseEvidence | None],
    /,
) -> np.ndarray:
    """``penalty_factor * sum_s w_s^2 m_s C_s`` at every common point."""
    certified = np.zeros((quadrature.point_count,), dtype=np.float64)
    for index, evidence in enumerate(stability):
        if evidence is None:
            continue
        facet = np.asarray(evidence.constants) * np.asarray(evidence.cell_multiplicity)
        points = np.asarray(quadrature.sides[index].facets)
        certified += imposition.weights[index] ** 2 * facet[points]
    return imposition.penalty_factor * certified


def _require_linear_residual(
    residual: _NitscheResidual, data: tuple[_SideData, _SideData], /
) -> None:
    """Refuse fluxes that are not additive: the rows are assembled linearly."""
    spaces = (data[0].trace.coefficient_space, data[1].trace.coefficient_space)
    args = {residual.components[0]: None, residual.components[1]: None}
    probes = tuple(
        tuple(
            jnp.asarray(
                np.cos(frequency * np.arange(space.size, dtype=np.float64)),
                dtype=space.dtype,
            ).reshape(space.shape)
            for space in spaces
        )
        for frequency in (1.0, 2.0)
    )
    first = residual.evaluate(probes[0], args)
    second = residual.evaluate(probes[1], args)
    both = residual.evaluate(
        (probes[0][0] + probes[1][0], probes[0][1] + probes[1][1]), args
    )
    for side in range(2):
        defect = float(jnp.max(jnp.abs(both[side] - first[side] - second[side])))
        scale = float(jnp.max(jnp.abs(first[side])) + jnp.max(jnp.abs(second[side])))
        if defect > _AFFINE_TOLERANCE * max(scale, 1.0e-300):
            raise ValueError(
                "The owners' pointwise fluxes are not linear in their states; the "
                "Nitsche rows of a nonlinear flux are not supported."
            )


def _lower_nitsche(
    law_id: str,
    imposition: NitscheImposition,
    sides: tuple[TransmissionSide, TransmissionSide],
    data: tuple[_SideData, _SideData],
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> tuple[
    tuple[LawBlock, ...],
    tuple[LawBlock, ...],
    tuple[Contribution, ...],
    _NitscheCertificate,
    NitscheEvidence,
]:
    prepared = list(data)
    fluxes: list[PreparedFluxAction | None] = [None, None]
    stability: list[TraceInverseEvidence | None] = [None, None]
    for index, side in enumerate(sides):
        component = components[side.component]
        if not isinstance(component, AbstractTraceComponent):
            raise TypeError(f"Component {side.component!r} publishes no side traces.")
        if imposition.weights[index] == 0.0:
            continue
        prepared[index], fluxes[index], stability[index] = _pointwise_side(
            component, side, data[index]
        )
    first, second = prepared
    degrees = (_trace_degree(first.trace), _trace_degree(second.trace))
    flux_degrees = tuple(
        None if flux is None else flux.descriptor.trace_degree for flux in fluxes
    )
    exact = max(
        2 * max(degrees),
        *(degree + max(degrees) for degree in flux_degrees if degree is not None),
        1,
    )
    quadrature = prepare_interface_quadrature(
        first.trace, second.trace, exact_degree=exact, policy=imposition.quadrature
    )
    for index, side in enumerate(sides):
        matrix = _dense_side_trace(
            quadrature.sides[index],
            prepared[index].support_rows,
            prepared[index].full_size,
            imposition.max_evidence_entries,
        )
        _refuse_strong_facets(matrix, quadrature.sides[index], prepared[index], side.role)
    penalty = _nitsche_penalty(imposition, quadrature, (stability[0], stability[1]))
    evidence = NitscheEvidence(
        imposition,
        penalty=penalty,
        trace_degrees=degrees,
        flux_degrees=(flux_degrees[0], flux_degrees[1]),
        stability=(stability[0], stability[1]),
        quadrature=quadrature,
    )
    if evidence.coercivity_constant <= 0.0:
        raise ValueError(
            f"Symmetric Nitsche coercivity is certified only for penalty_factor > 1; "
            f"penalty_factor {imposition.penalty_factor:.3g} times the certified "
            f"trace-inverse constants (penalty densities {evidence.penalty_range[0]:.3e}"
            f" to {evidence.penalty_range[1]:.3e}) does not dominate the flux terms. "
            "Raise the factor or declare the nonsymmetric variant."
        )
    residual = _NitscheResidual(
        quadrature,
        (fluxes[0], fluxes[1]),
        jnp.asarray(penalty, dtype=quadrature.weights.dtype),
        components=(sides[0].component, sides[1].component),
        weights=imposition.weights,
        theta=imposition.theta,
    )
    pair = (first, second)
    _require_linear_residual(residual, pair)
    endpoints = (first.endpoint, second.endpoint)
    contribution = ResidualContribution(
        endpoints,
        endpoints,
        residual,
        affine=True,
        law_id=law_id,
        imposition_id=canonical_fingerprint(
            {"kind": "nitsche-imposition", "law": law_id, "variant": imposition.variant}
        ),
    )
    certificate = _NitscheCertificate(law_id, residual, pair)
    return (), (), (contribution,), certificate, evidence


# --- Scalar transmission law -------------------------------------------------------------


@final
class ScalarTransmissionLaw(AbstractCouplingLaw, NonTrainableState):
    """Continuity of a scalar potential and balance of its conormal flux.

    On a two-sided interface with minus/plus sides, ``u_minus = u_plus`` and
    ``q_minus + q_plus = 0`` for the owners' outward conormal fluxes. The same
    law lowers through a ``MatchingElimination`` on coincident interfaces, a
    ``MortarImposition`` (whose multiplier is the outward flux of the minus
    side) on nonmatching ones, or a ``NitscheImposition`` through the owners'
    exact pointwise fluxes with a certified penalty; no lowering depends on the
    side methods.
    """

    law_id: str = eqx.field(static=True)
    binding: InterfaceBinding
    sides: tuple[TransmissionSide, TransmissionSide]
    imposition: TransmissionImposition

    def __init__(
        self,
        law_id: str,
        binding: InterfaceBinding,
        sides: tuple[TransmissionSide, TransmissionSide],
        imposition: TransmissionImposition,
        /,
    ) -> None:
        if not isinstance(binding, InterfaceBinding):
            raise TypeError("binding must be an InterfaceBinding.")
        if (
            not isinstance(sides, tuple)
            or len(sides) != 2
            or not all(isinstance(side, TransmissionSide) for side in sides)
        ):
            raise TypeError("sides must be a (minus, plus) pair of TransmissionSide.")
        if not isinstance(
            imposition, (MatchingElimination, MortarImposition, NitscheImposition)
        ):
            raise TypeError(
                "imposition must be a MatchingElimination, MortarImposition, or "
                "NitscheImposition."
            )
        self.law_id = canonical_identifier(law_id, "law_id")
        self.binding = binding
        self.sides = sides
        self.imposition = imposition

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        return (self.binding,)

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        _require_components(self.sides, components)
        data = _prepare_sides(self.sides, components)
        _require_binding(self.binding, self.sides, components, data, interface_owners)
        match self.imposition:
            case MortarImposition():
                lowered = _lower_mortar(self.law_id, self.imposition, self.sides, data)
            case MatchingElimination():
                lowered = _lower_elimination(
                    self.law_id, self.imposition, self.sides, data
                )
            case NitscheImposition():
                lowered = _lower_nitsche(
                    self.law_id, self.imposition, self.sides, data, components
                )
        states, rows, contributions, certificate, evidence = lowered
        return PreparedLaw(
            self.law_id,
            binding_id=self.binding.binding_id,
            state_blocks=states,
            row_blocks=rows,
            contributions=contributions,
            impositions=_impositions(self.law_id, self.sides, data, components),
            certificate=certificate,
            evidence=evidence,
        )


# --- Conservative numerical flux -------------------------------------------------------------


# Relative roundoff bound of the additivity check of a flux declared affine.
_AFFINE_TOLERANCE = 1.0e-10


class AbstractInterfaceFlux(StrictModule):
    """One shared flux density of a two-sided interface law.

    ``evaluate(minus, plus, points, normals, args)`` returns, at every common
    interface point, the outward conormal flux density of the minus side
    (``kappa grad u_minus . n`` with ``n`` the binding normal out of the minus
    side, the owners' reaction convention) from both sides' value traces at
    those points. ``affine`` declares the density affine in the two traces, so
    it enters linear assembly through its exact linearization; ``trace_degree``
    is its polynomial degree in the traces and sets the exactness of the
    common quadrature. ``runtime_inputs`` names the component runtime inputs
    the density reads (a learned response bound at every solve); such a flux is
    never evaluated without arguments, cannot be certified affine, and every
    named input must be the target of a refresh ``ParameterBinding``.
    """

    @property
    def runtime_inputs(self) -> tuple[RuntimeInput, ...]:
        return ()

    @property
    @abc.abstractmethod
    def affine(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def trace_degree(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        minus: Array,
        plus: Array,
        points: Array,
        normals: Array,
        args: object,
        /,
    ) -> Array:
        raise NotImplementedError


@final
class InterfaceConductance(AbstractInterfaceFlux):
    """Imperfect contact through a finite interface conductance ``h``.

    The flux leaving the minus side, ``-kappa grad u_minus . n``, is
    ``h (u_minus - u_plus)`` (a thermal contact resistance ``1 / h`` or a
    membrane permeability); the minus conormal flux is therefore
    ``h (u_plus - u_minus)``.
    """

    conductance: Array

    def __init__(self, conductance: float, /) -> None:
        value = positive_finite_float(conductance, "conductance")
        self.conductance = jnp.asarray(value, dtype=jnp.float64)

    @property
    def affine(self) -> bool:
        return True

    @property
    def trace_degree(self) -> int:
        return 1

    def evaluate(
        self,
        minus: Array,
        plus: Array,
        points: Array,
        normals: Array,
        args: object,
        /,
    ) -> Array:
        del points, normals, args
        return self.conductance * (plus - minus)


@final
class GapRadiation(AbstractInterfaceFlux):
    """Net gray-body radiation across a thin nonparticipating gap.

    Two diffuse gray surfaces with ``emissivities`` ``(e_minus, e_plus)`` at
    absolute temperatures ``u`` exchange ``sigma (u_minus^4 - u_plus^4) /
    (1 / e_minus + 1 / e_plus - 1)`` from the minus to the plus side; the
    minus conormal flux is its negative. ``stefan_boltzmann`` is ``sigma`` in
    the problem's units.
    """

    emissivities: Array
    stefan_boltzmann: Array

    def __init__(
        self,
        emissivities: tuple[float, float],
        /,
        *,
        stefan_boltzmann: float = 5.670374419e-8,
    ) -> None:
        if not isinstance(emissivities, tuple) or len(emissivities) != 2:
            raise TypeError("emissivities must be a (minus, plus) pair.")
        values = tuple(
            positive_finite_float(value, "emissivities") for value in emissivities
        )
        if max(values) > 1.0:
            raise ValueError("Gray-body emissivities lie in (0, 1].")
        sigma = positive_finite_float(stefan_boltzmann, "stefan_boltzmann")
        self.emissivities = jnp.asarray(values, dtype=jnp.float64)
        self.stefan_boltzmann = jnp.asarray(sigma, dtype=jnp.float64)

    @property
    def affine(self) -> bool:
        return False

    @property
    def trace_degree(self) -> int:
        return 4

    def evaluate(
        self,
        minus: Array,
        plus: Array,
        points: Array,
        normals: Array,
        args: object,
        /,
    ) -> Array:
        del points, normals, args
        exchange = self.stefan_boltzmann / (jnp.sum(1.0 / self.emissivities) - 1.0)
        return exchange * (plus**4 - minus**4)


@final
class ConservativeFluxEvidence(StrictModule, NonTrainableState):
    """Flux declaration, trace degrees, and coverage of one numerical-flux law."""

    affine: bool = eqx.field(static=True)
    flux_degree: int = eqx.field(static=True)
    trace_degrees: tuple[int, int] = eqx.field(static=True)
    quadrature_exact_degree: int = eqx.field(static=True)
    coverage: InterfaceCoverageEvidence

    def __init__(
        self,
        flux: AbstractInterfaceFlux,
        trace_degrees: tuple[int, int],
        quadrature: InterfaceQuadrature,
        /,
    ) -> None:
        self.affine = flux.affine
        self.flux_degree = flux.trace_degree
        self.trace_degrees = trace_degrees
        self.quadrature_exact_degree = quadrature.exact_degree
        self.coverage = quadrature.evidence


@final
class _FluxResidual(AbstractContributionResidual):
    """Both owners' injections of the one shared flux at the common quadrature."""

    quadrature: InterfaceQuadrature
    flux: AbstractInterfaceFlux

    @property
    def runtime_inputs(self) -> tuple[RuntimeInput, ...]:
        return self.flux.runtime_inputs

    def density(self, minus: Array, plus: Array, args: object, /) -> Array:
        quadrature = self.quadrature
        density = self.flux.evaluate(
            quadrature.sides[0].values(minus),
            quadrature.sides[1].values(plus),
            quadrature.points,
            quadrature.normals,
            args,
        )
        if density.shape != (quadrature.point_count,) or not jnp.issubdtype(
            density.dtype, jnp.floating
        ):
            raise ValueError(
                "The interface flux must return one real density per common point."
            )
        return density

    def evaluate(self, inputs: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        minus, plus = inputs
        weighted = self.quadrature.weights * self.density(minus, plus, args)
        # The plus outward normal is -n: what the minus side loses, the plus gains.
        return (
            -self.quadrature.sides[0].pullback(weighted),
            self.quadrature.sides[1].pullback(weighted),
        )


def _require_flux_density(
    residual: _FluxResidual, data: tuple[_SideData, _SideData], /
) -> None:
    """Refuse a flux of the wrong shape or one whose affine declaration fails."""
    if residual.flux.runtime_inputs:
        if residual.flux.affine:
            raise ValueError(
                "A flux that reads runtime arguments cannot be certified affine at "
                "preparation; declare it nonaffine."
            )
        return
    spaces = (data[0].trace.coefficient_space, data[1].trace.coefficient_space)
    output = jax.eval_shape(
        lambda minus, plus: residual.density(minus, plus, None),
        spaces[0].structure(),
        spaces[1].structure(),
    )
    if output.shape != (residual.quadrature.point_count,) or not jnp.issubdtype(
        output.dtype, jnp.floating
    ):
        raise ValueError(
            "The interface flux must return one real density per common point."
        )
    if not residual.flux.affine:
        return
    probes = tuple(
        tuple(
            jnp.asarray(
                np.cos(frequency * np.arange(space.size, dtype=np.float64)),
                dtype=space.dtype,
            ).reshape(space.shape)
            for space in spaces
        )
        for frequency in (1.0, 2.0)
    )
    zero = residual.density(spaces[0].zeros(), spaces[1].zeros(), None)
    first = residual.density(probes[0][0], probes[0][1], None)
    second = residual.density(probes[1][0], probes[1][1], None)
    both = residual.density(
        probes[0][0] + probes[1][0], probes[0][1] + probes[1][1], None
    )
    defect = float(jnp.max(jnp.abs(both - first - second + zero)))
    scale = float(
        jnp.max(jnp.abs(first)) + jnp.max(jnp.abs(second)) + jnp.max(jnp.abs(zero))
    )
    if defect > _AFFINE_TOLERANCE * max(scale, 1.0e-300):
        raise ValueError(
            f"The interface flux is declared affine but is not additive in the "
            f"traces (relative defect {defect / max(scale, 1.0e-300):.3e})."
        )


@final
class _FluxCertificate(AbstractLawCertificate):
    """Owner-flux law residual, conservation of the injections, and trace jump."""

    law_id: str = eqx.field(static=True)
    residual: _FluxResidual
    data: tuple[_SideData, _SideData]

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        del law_state
        first = fields[(self.data[0].component, self.data[0].field)]
        second = fields[(self.data[1].component, self.data[1].field)]
        injected = self.residual.evaluate((first, second), args)
        injected_terms = _uncancelled_terms(
            lambda minus, plus: self.residual.evaluate((minus, plus), args),
            first,
            second,
        )
        law, law_scale = _owner_row_law(
            self.data, (first, second), injected, injected_terms, args
        )
        quadrature = self.residual.quadrature
        density = self.residual.density(first, second, args)
        conservation = jnp.abs(jnp.sum(injected[0]) + jnp.sum(injected[1]))
        conservation_scale = jnp.sum(quadrature.weights * jnp.abs(density)) + sum(
            (jnp.sum(jnp.abs(terms)) for terms in injected_terms),
            jnp.zeros((), dtype=first.dtype),
        )
        jump, jump_scale = _l2_mismatch(quadrature, first, second)
        return InterfaceDefectReport(
            self.law_id,
            ("law-residual", "flux-conservation", "trace-jump-l2"),
            (True, True, False),
            jnp.stack((jnp.sqrt(law), conservation, jump)),
            jnp.stack((jnp.sqrt(law_scale), conservation_scale, jump_scale)),
        )


@final
class ConservativeFluxLaw(AbstractCouplingLaw, NonTrainableState):
    """Two-sided interface law imposed through one shared numerical flux.

    ``flux`` evaluates, at the common interface quadrature of the two sides'
    value traces, the outward conormal flux ``F(u_minus, u_plus, x, n)`` of the
    minus side. The one density is injected into both owners with consistent
    orientation: minus rows receive ``-int F v_minus`` and plus rows
    ``+int F v_plus`` (the plus outward normal is ``-n``), so what one side
    loses the other gains. The law owns no unknowns; an affine flux lowers to
    linear assembly, any other flux to the coupled Newton residual. Neither
    lowering depends on the side methods.
    """

    law_id: str = eqx.field(static=True)
    binding: InterfaceBinding
    sides: tuple[TransmissionSide, TransmissionSide]
    flux: AbstractInterfaceFlux
    quadrature: InterfaceQuadraturePolicy
    max_evidence_entries: int = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        binding: InterfaceBinding,
        sides: tuple[TransmissionSide, TransmissionSide],
        flux: AbstractInterfaceFlux,
        /,
        *,
        quadrature: InterfaceQuadraturePolicy | None = None,
        max_evidence_entries: int = 16_000_000,
    ) -> None:
        if not isinstance(binding, InterfaceBinding):
            raise TypeError("binding must be an InterfaceBinding.")
        if (
            not isinstance(sides, tuple)
            or len(sides) != 2
            or not all(isinstance(side, TransmissionSide) for side in sides)
        ):
            raise TypeError("sides must be a (minus, plus) pair of TransmissionSide.")
        if not isinstance(flux, AbstractInterfaceFlux):
            raise TypeError("flux must be an AbstractInterfaceFlux.")
        policy = InterfaceQuadraturePolicy() if quadrature is None else quadrature
        if not isinstance(policy, InterfaceQuadraturePolicy):
            raise TypeError("quadrature must be an InterfaceQuadraturePolicy.")
        self.law_id = canonical_identifier(law_id, "law_id")
        self.binding = binding
        self.sides = sides
        self.flux = flux
        self.quadrature = policy
        self.max_evidence_entries = positive_integer(
            max_evidence_entries, "max_evidence_entries"
        )

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        return (self.binding,)

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        _require_components(self.sides, components)
        data = _prepare_sides(self.sides, components)
        _require_binding(self.binding, self.sides, components, data, interface_owners)
        degrees = (_trace_degree(data[0].trace), _trace_degree(data[1].trace))
        quadrature = prepare_interface_quadrature(
            data[0].trace,
            data[1].trace,
            exact_degree=(self.flux.trace_degree + 1) * max(degrees),
            policy=self.quadrature,
        )
        for index, side in enumerate(self.sides):
            matrix = _dense_side_trace(
                quadrature.sides[index],
                data[index].support_rows,
                data[index].full_size,
                self.max_evidence_entries,
            )
            _refuse_strong_facets(matrix, quadrature.sides[index], data[index], side.role)
        residual = _FluxResidual(quadrature, self.flux)
        _require_flux_density(residual, data)
        endpoints = (data[0].endpoint, data[1].endpoint)
        contribution = ResidualContribution(
            endpoints,
            endpoints,
            residual,
            affine=self.flux.affine,
            law_id=self.law_id,
            imposition_id=canonical_fingerprint(
                {"kind": "numerical-flux-imposition", "law": self.law_id}
            ),
        )
        return PreparedLaw(
            self.law_id,
            binding_id=self.binding.binding_id,
            state_blocks=(),
            row_blocks=(),
            contributions=(contribution,),
            impositions=_impositions(self.law_id, self.sides, data, components),
            certificate=_FluxCertificate(self.law_id, residual, data),
            evidence=ConservativeFluxEvidence(self.flux, degrees, quadrature),
        )


# --- Integral port -----------------------------------------------------------------------------


@final
class PortSide(StrictModule, NonTrainableState):
    """Boundary port of one component field that realizes a lumped connector.

    ``domain`` is the component-owned exterior-facet domain of the port (an
    electrode or terminal surface) and ``connector`` the ID of the connector,
    in the lumped network, whose across and through variables the port carries.
    """

    connector: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    domain: IntegrationDomain

    def __init__(
        self, connector: str, component: str, field: str, domain: IntegrationDomain, /
    ) -> None:
        if not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be an IntegrationDomain.")
        if domain.kind != "exterior_facet":
            raise ValueError(
                "A field port is a set of exterior facets of its component's support."
            )
        self.connector = canonical_identifier(connector, "connector")
        self.component = canonical_identifier(component, "component")
        self.field = canonical_identifier(field, "field")
        self.domain = domain


@final
class IntegralPortEvidence(StrictModule, NonTrainableState):
    """Connector identity, port measure, and network closure of one port law.

    ``port_relation`` ``(a, b, c)`` is the one-port relation ``a V + b I = c``
    that the lumped network imposes on the port's across ``V`` and through
    ``I`` variables, scaled so that the larger of ``a`` and ``b`` is one;
    ``network_rank`` and the extreme singular values certify that the
    network's equations are independent.
    """

    connector: str = eqx.field(static=True)
    across_variable: str = eqx.field(static=True)
    through_variable: str = eqx.field(static=True)
    port_measure: float = eqx.field(static=True)
    free_trace_rows: int = eqx.field(static=True)
    trace_degree: int = eqx.field(static=True)
    network_rank: int = eqx.field(static=True)
    singular_value_min: float = eqx.field(static=True)
    singular_value_max: float = eqx.field(static=True)
    port_relation: tuple[float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        *,
        connector: str,
        variables: tuple[str, str],
        port_measure: float,
        free_trace_rows: int,
        trace_degree: int,
        network_rank: int,
        singular_values: np.ndarray,
        port_relation: tuple[float, float, float],
    ) -> None:
        self.connector = connector
        self.across_variable, self.through_variable = variables
        self.port_measure = float(port_measure)
        self.free_trace_rows = free_trace_rows
        self.trace_degree = trace_degree
        self.network_rank = network_rank
        self.singular_value_min = float(singular_values[-1])
        self.singular_value_max = float(singular_values[0])
        self.port_relation = port_relation


def _port_variables(
    system: AcausalSystem,
    connector: str,
    potential_unit: str,
    flux_unit: str,
    /,
) -> tuple[str, str]:
    """Across and through variable names of the port connector, by declared kind."""
    lookup = {item.connector_id: item for item in system.connectors}
    if connector not in lookup:
        raise ValueError(f"The lumped system has no connector {connector!r}.")
    declared = lookup[connector].connector_type
    across = tuple(item for item in declared.variables if item.kind == "across")
    through = tuple(item for item in declared.variables if item.kind == "through")
    if len(declared.variables) != 2 or len(across) != 1 or len(through) != 1:
        kinds = [item.kind for item in declared.variables]
        raise ValueError(
            f"Connector type {declared.connector_type_id!r} must declare exactly one "
            f"across and one through variable to realize a field port; it declares "
            f"{kinds}."
        )
    for variable, unit, role in (
        (across[0], potential_unit, "potential"),
        (through[0], flux_unit, "flux"),
    ):
        if variable.unit != unit:
            raise ValueError(
                f"Connector variable {variable.name!r} ({variable.kind}) has unit "
                f"{variable.unit!r}, but the field's {role} is in {unit!r}."
            )
    return across[0].name, through[0].name


def _port_relation(
    matrix: np.ndarray,
    vector: np.ndarray,
    indices: tuple[int, int],
    tolerance: float,
    /,
) -> tuple[tuple[float, float, float], int, np.ndarray]:
    """One-port relation ``a V + b I = c`` left open by independent network rows."""
    across, through = indices
    _, singular, right = np.linalg.svd(matrix)
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank < matrix.shape[0]:
        raise ValueError(
            f"The lumped network's equations are dependent (numerical rank {rank} of "
            f"{matrix.shape[0]}); a port closes exactly one missing equation."
        )
    direction = right[-1]
    coefficients = np.asarray((direction[through], -direction[across]))
    if np.max(np.abs(coefficients)) <= tolerance:
        raise ValueError(
            "The lumped network fixes both the port potential and the port flow; a "
            "field cannot close it."
        )
    pivot = coefficients[np.argmax(np.abs(coefficients))]
    particular, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    first, second = coefficients / pivot
    constant = first * particular[across] + second * particular[through]
    return (float(first), float(second), float(constant)), rank, singular


def _prepare_port(
    side: TransmissionSide, components: Mapping[str, AbstractSpatialComponent], /
) -> _SideData:
    if side.component not in components:
        raise ValueError(
            f"Port {side.role!r} names unknown component {side.component!r}."
        )
    component = components[side.component]
    if not isinstance(component, AbstractTraceComponent):
        raise TypeError(
            f"Component {side.component!r} publishes no side traces; a port law acts "
            "through its boundary trace and conormal flux."
        )
    component.field(side.field)
    trace = _prepare_trace(component, side)
    _refuse_owner_facet_laws(component, trace, side.role)
    free = np.setdiff1d(np.asarray(trace.support_rows), component.strong_rows(side.field))
    flux = component.prepare_conormal_flux(trace)
    return _SideData(side.component, side.field, trace, flux, free)


@final
class _PortActions(StrictModule):
    """Mean port trace, uniform-flux load, and network rows of one port law.

    ``weights`` are the port integrals ``int_port v`` of the owner's full rows
    and ``measure`` the port length. The law's unknowns are the network's
    connector variables; its rows are the port equation followed by the
    network rows.
    """

    weights: Array
    measure: Array
    matrix: Array
    vector: Array
    across: int = eqx.field(static=True)
    through: int = eqx.field(static=True)

    def _zeros(self, dtype: jnp.dtype, /) -> Array:
        return jnp.zeros((self.matrix.shape[1],), dtype=dtype)

    def inject(self, state: Array, /) -> Array:
        """Field rows ``-int (I / |port|) v`` of the uniform port flux density."""
        return -(state[self.through] / self.measure) * self.weights

    def pair(self, values: Array, /) -> Array:
        """``sum_i w_i x_i`` over the owner's full coefficient layout."""
        return jnp.tensordot(self.weights, values, axes=self.weights.ndim)

    def inject_transpose(self, rows: Array, /) -> Array:
        value = -self.pair(rows) / self.measure
        return self._zeros(rows.dtype).at[self.through].set(value)

    def mean(self, values: Array, /) -> Array:
        """Port equation row ``mean_port(u)`` (the trace tested by constants)."""
        value = self.pair(values) / self.measure
        return self._zeros(values.dtype).at[0].set(value)

    def mean_transpose(self, rows: Array, /) -> Array:
        return (rows[0] / self.measure) * self.weights

    def network(self, state: Array, /) -> Array:
        """Rows ``(-V, A x)``: the port equation's potential and the network rows."""
        return jnp.concatenate((-state[self.across][None], self.matrix @ state))

    def network_transpose(self, rows: Array, /) -> Array:
        return (self.matrix.T @ rows[1:]).at[self.across].add(-rows[0])


def _port_contributions(
    law_id: str, datum: _SideData, actions: _PortActions, space: ArraySpace, /
) -> tuple[Contribution, ...]:
    state = ContributionEndpoint(law_id, "connector-variables", space="reduced")
    rows = ContributionEndpoint(law_id, "port-equations", space="reduced")
    full = datum.trace.coefficient_space
    row_space = DualSpace(space)
    imposition = canonical_fingerprint({"kind": "integral-port", "law": law_id})
    load = jnp.concatenate((jnp.zeros((1,), dtype=actions.vector.dtype), -actions.vector))
    return (
        LinearContribution(
            datum.endpoint,
            state,
            FunctionLinearOperator(
                actions.inject,
                source=space,
                target=DualSpace(full),
                transpose_action=actions.inject_transpose,
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
        LinearContribution(
            rows,
            datum.endpoint,
            FunctionLinearOperator(
                actions.mean,
                source=full,
                target=row_space,
                transpose_action=actions.mean_transpose,
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
        LinearContribution(
            rows,
            state,
            FunctionLinearOperator(
                actions.network,
                source=space,
                target=row_space,
                transpose_action=actions.network_transpose,
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
        LoadContribution(rows, load, law_id=law_id, imposition_id=imposition),
    )


@final
class _PortCertificate(AbstractLawCertificate):
    """Port potential, owner-flux balance, network residual, and equipotentiality."""

    law_id: str = eqx.field(static=True)
    actions: _PortActions
    datum: _SideData

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        (state,) = law_state
        actions = self.actions
        trace = self.datum.trace
        full = fields[(self.datum.component, self.datum.field)]
        mean = actions.pair(full) / actions.measure
        mean_terms = _uncancelled_terms(actions.pair, full) / actions.measure
        potential = state[actions.across]
        reaction, free = _free_reaction(self.datum, full, args)
        injected = -trace.flatten_rows(actions.inject(state))[self.datum.support_rows]
        balance = jnp.linalg.norm(jnp.where(free, reaction - injected, 0.0))
        terms = _reaction_terms(self.datum, full, args)
        balance_scale = (
            jnp.linalg.norm(jnp.where(free, reaction, 0.0))
            + jnp.linalg.norm(jnp.where(free, injected, 0.0))
            + jnp.linalg.norm(jnp.where(free, terms, 0.0))
        )
        network = jnp.linalg.norm(actions.matrix @ state - actions.vector)
        network_scale = jnp.linalg.norm(
            jnp.abs(actions.matrix) @ jnp.abs(state)
        ) + jnp.linalg.norm(actions.vector)
        deviation = jnp.sqrt(
            jnp.sum(trace.weights * (trace.apply(full) - potential) ** 2)
            / actions.measure
        )
        return InterfaceDefectReport(
            self.law_id,
            (
                "port-potential",
                "port-flux",
                "network-residual",
                "port-potential-deviation",
            ),
            (True, True, True, False),
            jnp.stack((jnp.abs(mean - potential), balance, network, deviation)),
            jnp.stack(
                (
                    jnp.abs(mean) + jnp.abs(mean_terms) + jnp.abs(potential),
                    balance_scale,
                    network_scale,
                    jnp.abs(potential),
                )
            ),
        )


@final
class IntegralPortLaw(AbstractCouplingLaw, NonTrainableState):
    """Integral port joining a field boundary to a lumped acausal network.

    The port realizes connector ``port.connector`` of ``system``, whose
    across/through semantics and connection equations are owned by
    ``phydrax.system_modeling``. The connector's type must declare exactly one
    across and one through variable (identified by their declared kinds, never
    by name) in ``potential_unit`` and ``flux_unit``. The across variable is the
    port potential ``V``; the through variable ``I`` is the flow entering the
    field through the port, ``int_port kappa grad u . n`` (the owners' outward
    conormal flux; the physical flux ``-kappa grad u`` then flows into the
    region, the connector's sign convention).

    The field and the connector couple by a mortar whose multiplier space is
    the constants: the multiplier is the uniform flux density ``I / |port|``,
    so field rows receive ``-(I / |port|) int v``, and one law row imposes
    ``mean_port(u) - V = 0`` (trace equality tested by constants). Where the
    field is equipotential on the port the two agree pointwise; the
    ``port-potential-deviation`` evidence measures the difference otherwise.
    ``equations``/``rhs`` are the network's constitutive rows over the
    system's connector variables; ``compile_linear_acausal_system`` adds its
    connection equations. The compiled network must supply every equation but
    the port's own. The law's unknowns are all connector variables; its rows
    are the port equation and the network rows.
    """

    law_id: str = eqx.field(static=True)
    port: PortSide
    system: AcausalSystem = eqx.field(static=True)
    variable_keys: tuple[tuple[str, str], ...] = eqx.field(static=True)
    matrix: Array
    vector: Array
    across: int = eqx.field(static=True)
    through: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        port: PortSide,
        system: AcausalSystem,
        equations: ArrayLike,
        rhs: ArrayLike,
        /,
        *,
        potential_unit: str,
        flux_unit: str,
        rank_tolerance: float = 1.0e-12,
    ) -> None:
        if not isinstance(port, PortSide):
            raise TypeError("port must be a PortSide.")
        if not isinstance(system, AcausalSystem):
            raise TypeError("system must be an AcausalSystem.")
        compiled = compile_linear_acausal_system(
            system,
            np.asarray(equations, dtype=np.float64),
            np.asarray(rhs, dtype=np.float64),
        )
        across, through = _port_variables(
            system,
            port.connector,
            canonical_identifier(potential_unit, "potential_unit"),
            canonical_identifier(flux_unit, "flux_unit"),
        )
        keys = compiled.variable_keys
        rows = compiled.system_matrix.shape[0]
        if rows != len(keys) - 1:
            raise ValueError(
                f"The lumped network supplies {rows} equations for {len(keys)} "
                f"connector variables; the port closes exactly one, so it must supply "
                f"{len(keys) - 1}."
            )
        self.law_id = canonical_identifier(law_id, "law_id")
        self.port = port
        self.system = system
        self.variable_keys = keys
        self.matrix = jnp.asarray(compiled.system_matrix, dtype=jnp.float64)
        self.vector = jnp.asarray(compiled.right_hand_side, dtype=jnp.float64)
        self.across = keys.index((port.connector, across))
        self.through = keys.index((port.connector, through))
        self.rank_tolerance = positive_finite_float(rank_tolerance, "rank_tolerance")

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        # A port is a boundary of one component closed by a lumped network, not a
        # physical interface between two spatial supports.
        return ()

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        del interface_owners  # A port acts on no interface binding.
        port = self.port
        side = TransmissionSide(port.connector, port.component, port.field, port.domain)
        datum = _prepare_port(side, components)
        trace = datum.trace
        weights = trace.inject_load(
            jnp.ones(trace.output_shape, dtype=trace.weights.dtype)
        )
        free = np.asarray(datum.free_rows)
        if not np.any(np.abs(np.asarray(trace.flatten_rows(weights))[free]) > 0.0):
            raise ValueError(
                f"The owner imposes the whole trace of port {port.connector!r} "
                "strongly; a Dirichlet law and a port law cannot both hold there."
            )
        relation, rank, singular = _port_relation(
            np.asarray(self.matrix),
            np.asarray(self.vector),
            (self.across, self.through),
            self.rank_tolerance,
        )
        measure = jnp.sum(trace.weights)
        actions = _PortActions(
            weights, measure, self.matrix, self.vector, self.across, self.through
        )
        space = ArraySpace((self.matrix.shape[1],), dtype=jnp.float64)
        imposition = LawImposition(
            port.component,
            port.field,
            field_space_id=components[port.component].field_space_id(port.field),
            entity_set_id=trace.descriptor.entity_set_id,
            facets=np.asarray(trace.descriptor.facets),
            rows=np.asarray(datum.support_rows),
            imposition_id=canonical_fingerprint(
                {"kind": "law-imposition", "law": self.law_id, "role": port.connector}
            ),
        )
        evidence = IntegralPortEvidence(
            connector=port.connector,
            variables=(
                self.variable_keys[self.across][1],
                self.variable_keys[self.through][1],
            ),
            port_measure=float(measure),
            free_trace_rows=free.size,
            trace_degree=_trace_degree(trace),
            network_rank=rank,
            singular_values=singular,
            port_relation=relation,
        )
        return PreparedLaw(
            self.law_id,
            binding_id=None,
            state_blocks=(LawBlock("connector-variables", space),),
            row_blocks=(LawBlock("port-equations", DualSpace(space)),),
            contributions=_port_contributions(self.law_id, datum, actions, space),
            impositions=(imposition,),
            certificate=_PortCertificate(self.law_id, actions, datum),
            evidence=evidence,
        )


# --- Volume field transfer ---------------------------------------------------------------------


@final
class FieldTransferEvidence(StrictModule, NonTrainableState):
    """Transfer identity and its claimed and measured exchange properties.

    ``constant_preserving`` and ``conservative`` are the transfer's published
    claims; ``constant_defect`` is the measured ``max |T 1 - 1|`` on which the
    exchange's conservation rests, ``dual_pairing_residual`` the measured
    relative defect of ``<T a, b> = <a, T^T b>`` for the published load route,
    and ``target_measure`` the total measure ``1^T M 1`` of the exchange
    region.
    """

    transfer_id: str = eqx.field(static=True)
    constant_preserving: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    constant_defect: float = eqx.field(static=True)
    dual_pairing_residual: float = eqx.field(static=True)
    target_measure: float = eqx.field(static=True)

    def __init__(
        self,
        transfer: FieldTransfer,
        /,
        *,
        constant_defect: float,
        dual_pairing_residual: float,
        target_measure: float,
    ) -> None:
        self.transfer_id = transfer.transfer_id
        self.constant_preserving = transfer.properties.constant_preserving
        self.conservative = transfer.properties.conservative
        self.constant_defect = float(constant_defect)
        self.dual_pairing_residual = float(dual_pairing_residual)
        self.target_measure = float(target_measure)


@final
class _TransferActions(StrictModule):
    """Exchange blocks of ``alpha [T, -I]^T M [T, -I]`` through the transfer routes."""

    primal: AbstractLinearOperator
    pullback: AbstractLinearOperator
    measure: AbstractLinearOperator
    exchange: Array

    def gained(self, source: Array, target: Array, /) -> Array:
        """Target covector ``alpha M (T u_s - u_t)`` of the exchanged rate."""
        return self.exchange * self.measure.mv(self.primal.mv(source) - target)

    def coupling(self, source: Array, /) -> Array:
        """Target rows ``-alpha M T u_s`` of the source field."""
        return -self.exchange * self.measure.mv(self.primal.mv(source))

    def coupling_transpose(self, rows: Array, /) -> Array:
        """Source rows ``-alpha T^T M u_t``: the load route of the coupling."""
        return -self.exchange * self.pullback.mv(self.measure.mv(rows))

    def target_block(self, target: Array, /) -> Array:
        return self.exchange * self.measure.mv(target)

    def source_block(self, source: Array, /) -> Array:
        return self.exchange * self.pullback.mv(self.measure.mv(self.primal.mv(source)))


def _transfer_contributions(
    law_id: str,
    endpoints: tuple[ContributionEndpoint, ContributionEndpoint],
    spaces: tuple[ArraySpace, ArraySpace],
    actions: _TransferActions,
    /,
) -> tuple[Contribution, ...]:
    source, target = endpoints
    source_space, target_space = spaces
    imposition = canonical_fingerprint({"kind": "field-transfer", "law": law_id})
    blocks = (
        (target, source, actions.coupling, actions.coupling_transpose),
        (source, target, actions.coupling_transpose, actions.coupling),
        (target, target, actions.target_block, actions.target_block),
        (source, source, actions.source_block, actions.source_block),
    )
    lookup = {source.owner: source_space, target.owner: target_space}
    return tuple(
        LinearContribution(
            row,
            column,
            FunctionLinearOperator(
                action,
                source=lookup[column.owner],
                target=DualSpace(lookup[row.owner]),
                transpose_action=transpose,
            ),
            law_id=law_id,
            imposition_id=imposition,
        )
        for row, column, action, transpose in blocks
    )


def _transfer_evidence(
    transfer: FieldTransfer,
    actions: _TransferActions,
    spaces: tuple[ArraySpace, ArraySpace],
    tolerance: float,
    /,
) -> FieldTransferEvidence:
    """Measure constant preservation and the dual route; refuse when they fail."""
    source_space, target_space = spaces
    ones = jnp.ones(source_space.shape, dtype=source_space.dtype)
    constant = float(jnp.max(jnp.abs(actions.primal.mv(ones) - 1.0)))
    if constant > tolerance:
        raise ValueError(
            f"The transfer claims constant preservation but moves constants by "
            f"{constant:.3e}; the exchange would not conserve the exchanged quantity."
        )
    first = jnp.asarray(
        np.cos(np.arange(source_space.size, dtype=np.float64)), dtype=source_space.dtype
    ).reshape(source_space.shape)
    second = jnp.asarray(
        np.sin(1.0 + np.arange(target_space.size, dtype=np.float64)),
        dtype=target_space.dtype,
    ).reshape(target_space.shape)
    left = float(jnp.vdot(actions.primal.mv(first), second))
    right = float(jnp.vdot(first, actions.pullback.mv(second)))
    dual = abs(left - right) / max(abs(left), abs(right), 1.0)
    if dual > tolerance:
        raise ValueError(
            f"The transfer's dual pullback is not the coordinate transpose of its "
            f"primal map (relative defect {dual:.3e}); it is no work-consistent load "
            "route."
        )
    constants = jnp.ones(target_space.shape, dtype=target_space.dtype)
    measure = float(jnp.vdot(constants, actions.measure.mv(constants)))
    return FieldTransferEvidence(
        transfer,
        constant_defect=constant,
        dual_pairing_residual=dual,
        target_measure=measure,
    )


@final
class _TransferCertificate(AbstractLawCertificate):
    """Exchange balance, exchange dissipation, and transferred mismatch."""

    law_id: str = eqx.field(static=True)
    actions: _TransferActions
    source: ContributionEndpoint
    target: ContributionEndpoint

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        del law_state, args
        actions = self.actions
        source = fields[(self.source.owner, self.source.block)]
        target = fields[(self.target.owner, self.target.block)]
        transferred = actions.primal.mv(source)
        gained = actions.gained(source, target)
        lost = actions.pullback.mv(gained)
        # Target rows receive -gained and source rows +lost; their totals cancel.
        balance = jnp.abs(jnp.sum(lost) - jnp.sum(gained))
        balance_scale = jnp.sum(jnp.abs(gained)) + jnp.sum(jnp.abs(lost))
        difference = transferred - target
        dissipation = jnp.vdot(difference, gained)
        mismatch = jnp.sqrt(jnp.abs(jnp.vdot(difference, actions.measure.mv(difference))))
        mismatch_scale = jnp.sqrt(
            jnp.abs(jnp.vdot(transferred, actions.measure.mv(transferred)))
        ) + jnp.sqrt(jnp.abs(jnp.vdot(target, actions.measure.mv(target))))
        return InterfaceDefectReport(
            self.law_id,
            ("exchange-balance", "exchange-dissipation", "transferred-mismatch-l2"),
            (True, False, False),
            jnp.stack((balance, dissipation, mismatch)),
            jnp.stack((balance_scale, jnp.abs(dissipation), mismatch_scale)),
        )


@final
class FieldTransferLaw(AbstractCouplingLaw, NonTrainableState):
    """Conservative volumetric exchange between two co-located component fields.

    Two continua share a region (bidomain, double-porosity, two-temperature
    models): the target gains the volumetric rate ``q = alpha (u_source -
    u_target)`` that the source loses. ``transfer`` is the native prepared
    relation between the two fields; its source and target field spaces must
    be the components' field spaces (the explicit identity binding of the two
    fields). Its primal map ``T`` carries source values into target
    coefficients and its published dual pullback ``T^T`` is the load route.
    ``measure`` is the target field's measure (mass) ``M`` on the exchange
    region. Target rows receive ``-alpha M (T u_s - u_t)`` and source rows, by
    work duality, ``+alpha T^T M (T u_s - u_t)``; the exchange operator
    ``alpha [T, -I]^T M [T, -I]`` is symmetric positive semidefinite. The
    exchange conserves the exchanged quantity exactly when ``T`` preserves
    constants, which the transfer must claim and preparation measures to
    ``tolerance``.
    """

    law_id: str = eqx.field(static=True)
    source: ContributionEndpoint
    target: ContributionEndpoint
    transfer: FieldTransfer
    measure: AbstractLinearOperator
    exchange: Array
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        source: ContributionEndpoint,
        target: ContributionEndpoint,
        transfer: FieldTransfer,
        measure: AbstractLinearOperator,
        /,
        *,
        exchange: float,
        tolerance: float = 1.0e-10,
    ) -> None:
        for endpoint, name in ((source, "source"), (target, "target")):
            if not isinstance(endpoint, ContributionEndpoint):
                raise TypeError(f"{name} must be a ContributionEndpoint.")
            if endpoint.space != "full":
                raise ValueError(f"The {name} endpoint must name a component field.")
        if source.owner == target.owner:
            raise ValueError("A field transfer law couples two different components.")
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a FieldTransfer.")
        if not isinstance(measure, AbstractLinearOperator):
            raise TypeError("measure must be an AbstractLinearOperator.")
        if transfer.dual_pullback_operator is None:
            raise ValueError(
                "The transfer publishes no dual pullback; the exchange needs its "
                "load route T^T."
            )
        if not transfer.properties.constant_preserving:
            raise ValueError(
                "The transfer does not claim constant preservation; the exchange "
                "conserves the exchanged quantity only if T 1 = 1."
            )
        if not (
            measure.properties.certifies("self_adjoint")
            and measure.properties.certifies("positive_definite")
        ):
            raise ValueError(
                "The target measure must certify self-adjoint positive definiteness."
            )
        self.law_id = canonical_identifier(law_id, "law_id")
        self.source = source
        self.target = target
        self.transfer = transfer
        self.measure = measure
        self.exchange = jnp.asarray(
            positive_finite_float(exchange, "exchange"), dtype=jnp.float64
        )
        self.tolerance = positive_finite_float(tolerance, "tolerance")

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        # The transfer's own source and target field spaces bind the two fields.
        return ()

    def _spaces(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> tuple[ArraySpace, ArraySpace]:
        spaces: list[ArraySpace] = []
        for endpoint, declared in (
            (self.source, self.transfer.source),
            (self.target, self.transfer.target),
        ):
            if endpoint.owner not in components:
                raise ValueError(
                    f"Law endpoint names unknown component {endpoint.owner!r}."
                )
            component = components[endpoint.owner]
            field = component.field(endpoint.block)
            identity = component.field_space_id(endpoint.block)
            if declared.field_space_id != identity:
                raise ValueError(
                    f"The transfer maps field space {declared.field_space_id!r}, not "
                    f"the field space {identity!r} of {endpoint.owner!r}."
                )
            spaces.append(field.full_space)
        source, target = spaces
        if (
            self.transfer.primal_operator.source.size != source.size
            or self.transfer.primal_operator.target.size != target.size
            or self.measure.source.size != target.size
            or self.measure.target.size != target.size
        ):
            raise ValueError(
                "The transfer and measure sizes do not match the two field spaces."
            )
        return source, target

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        del interface_owners  # A field transfer acts on no interface binding.
        spaces = self._spaces(components)
        pullback = self.transfer.dual_pullback_operator
        if pullback is None:
            raise ValueError("The transfer publishes no dual pullback.")
        actions = _TransferActions(
            self.transfer.primal_operator, pullback, self.measure, self.exchange
        )
        evidence = _transfer_evidence(self.transfer, actions, spaces, self.tolerance)
        return PreparedLaw(
            self.law_id,
            binding_id=None,
            state_blocks=(),
            row_blocks=(),
            contributions=_transfer_contributions(
                self.law_id, (self.source, self.target), spaces, actions
            ),
            impositions=(),
            certificate=_TransferCertificate(
                self.law_id, actions, self.source, self.target
            ),
            evidence=evidence,
        )


# --- Boundary-integral transmission --------------------------------------------------------


@final
class BoundaryIntegralSide(StrictModule, NonTrainableState):
    """Exterior endpoint of a boundary-integral transmission law.

    ``role`` is the plus endpoint of the interface binding and ``component``
    the name of a :class:`GalerkinBoundaryComponent` whose closed curve is the
    interface; its unbounded side is the exterior domain.
    """

    role: str = eqx.field(static=True)
    component: str = eqx.field(static=True)

    def __init__(self, role: str, component: str, /) -> None:
        self.role = canonical_identifier(role, "role")
        self.component = canonical_identifier(component, "component")


@final
class BoundaryIntegralEvidence(StrictModule, NonTrainableState):
    """Formulation, declared trace projection, and coverage of one boundary law.

    ``projection_order`` Gauss points per panel sample the volume trace;
    ``projection_exact_degree`` is the largest per-panel trace degree whose
    projection load the rule integrates exactly (at least ``trace_degree``,
    so the load and the conormal work pairing are exact). The projection
    itself is an L2 projection onto continuous P1 and is never exact for a
    higher-degree trace; its defect is measured by the certificate.
    """

    formulation: str = eqx.field(static=True)
    trace_degree: int = eqx.field(static=True)
    projection_order: int = eqx.field(static=True)
    projection_exact_degree: int = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    galerkin_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    panel_count: int = eqx.field(static=True)
    interface_source: bool = eqx.field(static=True)
    coverage: InterfaceCoverageEvidence

    def __init__(
        self,
        *,
        trace_degree: int,
        projection: BoundaryTraceProjection2D,
        galerkin: ScalarLaplaceGalerkin2D,
        interface_source: bool,
        coverage: InterfaceCoverageEvidence,
    ) -> None:
        self.formulation = "johnson-nedelec-bordered-exterior-laplace-2d"
        self.trace_degree = trace_degree
        self.projection_order = projection.order
        self.projection_exact_degree = projection.exact_polynomial_degree
        self.projection_id = projection.projection_id
        self.galerkin_id = galerkin.prepared_id
        self.convention_id = galerkin.convention.convention_id
        self.panel_count = galerkin.curve.panel_count
        self.interface_source = interface_source
        self.coverage = coverage


@final
class _BoundaryIntegralActions(StrictModule):
    """Volume trace at the declared projection's panel samples and its pairings."""

    side: InterfaceSideQuadrature
    projection: BoundaryTraceProjection2D

    def samples(self, coefficients: Array, /) -> Array:
        """Volume trace values at the ``(panel, sample)`` projection points."""
        values = self.side.values(coefficients)
        return values.reshape(self.projection.sample_weights.shape)

    def density_load(self, density: Array, /) -> Array:
        """``int_Gamma g gamma v ds`` of per-sample density values ``g``."""
        return self.side.pullback((self.projection.sample_weights * density).reshape(-1))

    def conormal_load(self, conormal: Array, /) -> Array:
        """``int_Gamma q gamma v ds`` of a DP0 conormal, on full volume rows."""
        return self.density_load(
            jnp.broadcast_to(conormal[:, None], self.projection.sample_weights.shape)
        )

    def conormal_moments(self, coefficients: Array, /) -> Array:
        """Panel integrals ``int_panel gamma u ds``: the transpose of ``conormal_load``."""
        return jnp.sum(
            self.projection.sample_weights * self.samples(coefficients), axis=1
        )

    def projection_load(self, coefficients: Array, /) -> Array:
        """``int_Gamma gamma u phi_v ds`` in the P1 dual (the projection's load)."""
        return self.projection.load.mv(self.samples(coefficients))

    def projection_pullback(self, dirichlet: Array, /) -> Array:
        """Exact transpose of ``projection_load`` onto full volume rows."""
        return self.side.pullback(
            self.projection.load.transpose_mv(dirichlet).reshape(-1)
        )

    def reconstruct(self, dirichlet: Array, /) -> Array:
        """P1 coefficients evaluated at the projection points."""
        vertices = self.projection.spaces.curve.panel_vertices
        hats = self.projection.hat_values
        return (
            dirichlet[vertices[:, 0]][:, None] * hats[None, :, 0]
            + dirichlet[vertices[:, 1]][:, None] * hats[None, :, 1]
        )


def _dp0_dual_norm(rows: Array, lengths: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(rows * rows / lengths))


@final
class _BoundaryIntegralCertificate(AbstractLawCertificate):
    """Exterior Cauchy relation, compatibility, projection, and flux balance.

    ``decay_tolerance`` is the boundary owner's declared decaying far-field
    tolerance (``None`` for a bounded exterior): its excess
    ``max(|c| - tolerance, 0)`` is gated relative to the tolerance, so a
    decaying exterior accepts ``|c| <= tolerance * (1 + solve tolerance)``.
    ``exterior`` is the boundary owner's bordered Dirichlet-to-Neumann solve;
    its response to the uncancelled Dirichlet terms scales the Cauchy and
    compatibility defects, so an exact datum whose exterior response cancels
    (a constant trace) is measured against the terms that cancel.
    """

    law_id: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    volume: _SideData
    actions: _BoundaryIntegralActions
    galerkin: ScalarLaplaceGalerkin2D
    exterior: PreparedExteriorLaplaceDirichlet2D
    source: Array
    decay_tolerance: float | None = eqx.field(static=True)

    def _cauchy(
        self,
        dirichlet: tuple[Array, Array],
        conormal: tuple[Array, Array],
        constant: tuple[Array, Array],
        /,
    ) -> tuple[Array, Array]:
        """Cauchy residual at the solution; term norms at it and its uncancelled response."""
        lengths = self.galerkin.spaces.conormal_trace.integral_weights
        double = self.galerkin.exterior_relation.blocks[0][0]
        if double is None:
            raise ValueError("The exterior relation has no Dirichlet-trace column.")
        solution, uncancelled = (
            (
                double.mv(dirichlet[index]),
                self.galerkin.single_layer.mv(conormal[index]),
                -lengths * constant[index][0],
            )
            for index in range(2)
        )
        residual = _dp0_dual_norm(solution[0] + solution[1] + solution[2], lengths)
        scale = sum(
            (_dp0_dual_norm(term, lengths) for term in (*solution, *uncancelled)),
            jnp.zeros(()),
        )
        return residual, scale

    def _balance(
        self, volume: Array, conormal: Array, args: Mapping[str, object], /
    ) -> tuple[Array, Array]:
        reaction, free = _free_reaction(self.volume, volume, args)
        injected = self.volume.trace.flatten_rows(
            self.actions.conormal_load(conormal) + self.source
        )[self.volume.support_rows]
        terms = _reaction_terms(self.volume, volume, args)
        balance = jnp.sqrt(jnp.sum(jnp.where(free, reaction - injected, 0.0) ** 2))
        scale = (
            jnp.sqrt(jnp.sum(jnp.where(free, reaction, 0.0) ** 2))
            + jnp.sqrt(jnp.sum(jnp.where(free, terms, 0.0) ** 2))
            + jnp.sqrt(jnp.sum(jnp.where(free, injected, 0.0) ** 2))
        )
        return balance, scale

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        from ...operators.integral.layer_potential._scalar_galerkin2d import (
            uncancelled_exterior_response,
        )

        (dirichlet,) = law_state
        volume = fields[(self.volume.component, self.volume.field)]
        conormal = fields[(self.boundary, "conormal")]
        constant = fields[(self.boundary, "far_field_constant")]
        spaces = self.galerkin.spaces
        lengths = spaces.conormal_trace.integral_weights
        weights = self.actions.projection.sample_weights
        direction, conormal_terms, constant_terms, terms_solved = (
            uncancelled_exterior_response(self.exterior, dirichlet)
        )
        cauchy, cauchy_scale = self._cauchy(
            (dirichlet, direction), (conormal, conormal_terms), (constant, constant_terms)
        )
        # A failed response solve leaves no reference magnitude: fail closed.
        missing = jnp.where(terms_solved, 0.0, jnp.nan)
        mass_rows = spaces.dirichlet_trace.mass.mv(dirichlet)
        load_rows = self.actions.projection_load(volume)
        samples = self.actions.samples(volume)
        difference = samples - self.actions.reconstruct(dirichlet)
        trace_norm = jnp.sqrt(jnp.sum(weights * samples * samples))
        exact_work = jnp.sum(weights * conormal[:, None] * samples)
        projected_work = jnp.dot(conormal, spaces.mixed_mass.mv(dirichlet))
        balance, balance_scale = self._balance(volume, conormal, args)
        values = (
            cauchy,
            jnp.abs(jnp.dot(lengths, conormal)),
            jnp.linalg.norm(mass_rows - load_rows),
            balance,
            jnp.sqrt(jnp.sum(weights * difference * difference)),
            jnp.abs(exact_work - projected_work),
            jnp.abs(constant[0]),
        )
        scales = (
            cauchy_scale + missing,
            jnp.dot(lengths, jnp.abs(conormal) + jnp.abs(conormal_terms)) + missing,
            jnp.linalg.norm(mass_rows) + jnp.linalg.norm(load_rows),
            balance_scale,
            trace_norm,
            jnp.abs(exact_work) + jnp.abs(projected_work),
            jnp.max(jnp.abs(samples)),
        )
        names = (
            "exterior-cauchy-residual",
            "logarithmic-compatibility",
            "trace-projection",
            "flux-balance",
            "projection-defect-l2",
            "work-pairing-defect",
            "far-field-constant",
        )
        gated = (True, True, True, True, False, False, False)
        if self.decay_tolerance is not None:
            tolerance = jnp.asarray(self.decay_tolerance, dtype=constant.dtype)
            names += ("far-field-decay",)
            gated += (True,)
            values += (jnp.maximum(jnp.abs(constant[0]) - tolerance, 0.0),)
            scales += (tolerance,)
        return InterfaceDefectReport(
            self.law_id, names, gated, jnp.stack(values), jnp.stack(scales)
        )


def _projection_order(degree: int, declared: int | None, /) -> int:
    """Gauss points per panel whose rule makes the projection load exact.

    The P1 load of a degree-``p`` trace integrates ``p + 1`` polynomials per
    panel, which ``n`` Gauss points integrate exactly when ``p <= 2 n - 2``;
    the conormal pairing (degree ``p``) is then exact as well.
    """
    minimum = (degree + 3) // 2
    if declared is None:
        return minimum
    if declared < minimum:
        raise ValueError(
            f"projection_order {declared} integrates the P1 load of the degree-"
            f"{degree} volume trace inexactly; at least {minimum} points per panel "
            "are required."
        )
    return declared


def _boundary_components(
    volume: TransmissionSide,
    boundary: BoundaryIntegralSide,
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> tuple[AbstractTraceComponent, GalerkinBoundaryComponent]:
    for name, role in (
        (volume.component, volume.role),
        (boundary.component, boundary.role),
    ):
        if name not in components:
            raise ValueError(f"Law side {role!r} names unknown component {name!r}.")
    interior = components[volume.component]
    if not isinstance(interior, AbstractTraceComponent):
        raise TypeError(
            f"Component {volume.component!r} publishes no side traces; the volume "
            "side of a boundary-integral law acts through its facet trace."
        )
    exterior = components[boundary.component]
    if not isinstance(exterior, GalerkinBoundaryComponent):
        raise TypeError(
            f"Component {boundary.component!r} is not a 2-D Galerkin boundary "
            "component; the exterior side needs its weak boundary operators."
        )
    interior.field(volume.field)
    return interior, exterior


def _require_boundary_binding(
    binding: InterfaceBinding,
    volume: TransmissionSide,
    boundary: BoundaryIntegralSide,
    components: tuple[AbstractTraceComponent, GalerkinBoundaryComponent],
    trace: PreparedTraceAction,
    panel_sites: np.ndarray,
    interface_owners: tuple[InterfaceOwner, ...],
    /,
) -> None:
    """Volume on the minus (bounded) side, exterior owner on the plus side.

    The volume trace and the boundary curve's per-panel ``panel_sites`` must lie
    on their endpoints' attached supports: the volume's outward normal and the
    exterior's (the negated curve normal) on their attached sides.
    """
    if binding.incidence != "two-sided":
        raise ValueError("A boundary-integral law needs a two-sided interface binding.")
    expected = (
        binding.side(InterfaceSide.MINUS).role,
        binding.side(InterfaceSide.PLUS).role,
    )
    if (volume.role, boundary.role) != expected:
        raise ValueError(
            f"The volume side must be the binding's minus role and the exterior "
            f"owner its plus role {expected!r}; the interface normal points into "
            "the exterior."
        )
    for role, quantity, identity in (
        (volume.role, "value", components[0].field_space_id(volume.field)),
        (boundary.role, "conormal", components[1].field_space_id("conormal")),
    ):
        declared = dict(binding.endpoint(role).fields).get(quantity)
        if declared != identity:
            raise ValueError(
                f"Endpoint {role!r} binds {quantity} field {declared!r}, not the "
                f"component field space {identity!r}."
            )
    binding.endpoint(volume.role).require_side(
        interface_owners,
        trace.sites,
        trace.normals,
        facets=(trace.descriptor.entity_set_id, trace.descriptor.facets),
    )
    exterior = -np.asarray(components[1].galerkin.curve.normals, dtype=np.float64)
    binding.endpoint(boundary.role).require_side(
        interface_owners,
        panel_sites,
        np.broadcast_to(exterior[:, None, :], panel_sites.shape),
    )


def _boundary_side_data(
    component: AbstractTraceComponent, side: TransmissionSide, /
) -> _SideData:
    """Volume trace and reaction flux; refuse owner laws on the boundary facets."""
    trace = _prepare_trace(component, side)
    _refuse_owner_facet_laws(component, trace, side.role)
    rows = np.asarray(trace.support_rows)
    if np.intersect1d(rows, component.strong_rows(side.field)).size:
        raise ValueError(
            f"The owner of side {side.role!r} imposes trace rows strongly on the "
            "boundary curve; a Dirichlet law and a transmission law cannot both "
            "hold there."
        )
    return _SideData(
        side.component, side.field, trace, component.prepare_conormal_flux(trace), rows
    )


def _boundary_contributions(
    law_id: str,
    volume: _SideData,
    exterior: GalerkinBoundaryComponent,
    actions: _BoundaryIntegralActions,
    source: Array | None,
    /,
) -> tuple[Contribution, ...]:
    """Volume rows ``R - int q v``, projection rows ``M phi - L gamma u``, and BEM rows."""
    spaces = exterior.galerkin.spaces
    full = volume.trace.coefficient_space
    conormal_space = spaces.conormal_trace.vector_space
    dirichlet_space = spaces.dirichlet_trace.vector_space
    dirichlet = ContributionEndpoint(law_id, "dirichlet-trace", space="reduced")
    projection = ContributionEndpoint(law_id, "trace-projection", space="reduced")
    imposition = canonical_fingerprint({"kind": "boundary-integral", "law": law_id})
    contributions: list[Contribution] = [
        LinearContribution(
            volume.endpoint,
            ContributionEndpoint(exterior.name, "conormal", space="reduced"),
            FunctionLinearOperator(
                lambda value: -actions.conormal_load(value),
                source=conormal_space,
                target=DualSpace(full),
                transpose_action=lambda value: -actions.conormal_moments(value),
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
        LinearContribution(
            projection,
            volume.endpoint,
            FunctionLinearOperator(
                lambda value: -actions.projection_load(value),
                source=full,
                target=DualSpace(dirichlet_space),
                transpose_action=lambda value: -actions.projection_pullback(value),
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
        LinearContribution(
            projection,
            dirichlet,
            spaces.dirichlet_trace.mass,
            law_id=law_id,
            imposition_id=imposition,
        ),
        LinearContribution(
            ContributionEndpoint(
                exterior.name, "exterior_boundary_equation", space="reduced"
            ),
            dirichlet,
            exterior.dirichlet_operator(),
            law_id=law_id,
            imposition_id=imposition,
        ),
    ]
    if source is not None:
        contributions.append(
            LoadContribution(
                volume.endpoint, -source, law_id=law_id, imposition_id=imposition
            )
        )
    return tuple(contributions)


def _boundary_impositions(
    law_id: str,
    volume: _SideData,
    components: tuple[AbstractTraceComponent, GalerkinBoundaryComponent],
    /,
) -> tuple[LawImposition, ...]:
    interior, exterior = components
    panels = np.arange(exterior.galerkin.curve.panel_count)
    return (
        LawImposition(
            volume.component,
            volume.field,
            field_space_id=interior.field_space_id(volume.field),
            entity_set_id=volume.trace.descriptor.entity_set_id,
            facets=np.asarray(volume.trace.descriptor.facets),
            rows=np.asarray(volume.support_rows),
            imposition_id=canonical_fingerprint(
                {"kind": "law-imposition", "law": law_id, "side": "volume"}
            ),
        ),
        LawImposition(
            exterior.name,
            "conormal",
            field_space_id=exterior.field_space_id("conormal"),
            entity_set_id=exterior.galerkin.curve.curve_id,
            facets=panels,
            rows=panels,
            imposition_id=canonical_fingerprint(
                {"kind": "law-imposition", "law": law_id, "side": "exterior"}
            ),
        ),
    )


@final
class BoundaryIntegralTransmissionLaw(AbstractCouplingLaw, NonTrainableState):
    """Transmission of a volume field to the exterior Laplace field of a closed curve.

    The volume owner (minus side, bounded interior) and the 2-D Galerkin
    boundary owner (plus side, unbounded exterior with ``kappa = 1``) share
    the value ``u`` and the conormal ``q = kappa d_n u`` on the curve, with
    ``n`` pointing into the exterior. The law lowers the Johnson--Nedelec
    coupling with the bounded exterior's far-field constant ``c``:

    - volume rows ``R(u) - int q gamma v ds - int g gamma v ds`` (``g`` the
      optional declared ``interface_source``, a prescribed conormal jump
      ``kappa d_n u_minus - d_n u_plus``);
    - law rows ``M phi - L gamma u`` for the law's continuous-P1 Dirichlet
      trace ``phi``: the declared L2 projection ``phi = Pi gamma u`` of the
      boundary owner, with ``L`` its load on per-panel Gauss samples of the
      volume trace;
    - boundary rows ``(M/2 - K) phi + V q - m c`` and ``m^T q`` of the
      boundary component.

    The projection is never the exact volume trace; its defect and the
    difference of the exact and projected work pairings are measured.
    Boundary panels must refine the volume facets so every per-panel load is
    integrated exactly. The volume's own coefficient enters only through its
    residual, so any conforming or nonconforming volume owner that publishes
    a polynomial side trace and a reaction flux substitutes unchanged.

    The far-field constant ``c`` is evidence (``"far-field-constant"``) for
    the default bounded exterior. When the boundary component declares a
    decaying exterior (``far_field="decaying"`` with a tolerance), the gated
    ``"far-field-decay"`` defect refuses a solved ``|c|`` beyond that
    tolerance.
    """

    law_id: str = eqx.field(static=True)
    binding: InterfaceBinding
    volume: TransmissionSide
    boundary: BoundaryIntegralSide
    projection_order: int | None = eqx.field(static=True)
    interface_source: Callable[[Array], Array] | None = eqx.field(static=True)
    quadrature: InterfaceQuadraturePolicy

    def __init__(
        self,
        law_id: str,
        binding: InterfaceBinding,
        volume: TransmissionSide,
        boundary: BoundaryIntegralSide,
        /,
        *,
        projection_order: int | None = None,
        interface_source: Callable[[Array], Array] | None = None,
        quadrature: InterfaceQuadraturePolicy | None = None,
    ) -> None:
        if not isinstance(binding, InterfaceBinding):
            raise TypeError("binding must be an InterfaceBinding.")
        if not isinstance(volume, TransmissionSide):
            raise TypeError("volume must be a TransmissionSide.")
        if not isinstance(boundary, BoundaryIntegralSide):
            raise TypeError("boundary must be a BoundaryIntegralSide.")
        if interface_source is not None and not callable(interface_source):
            raise TypeError("interface_source must be callable or None.")
        policy = InterfaceQuadraturePolicy() if quadrature is None else quadrature
        if not isinstance(policy, InterfaceQuadraturePolicy):
            raise TypeError("quadrature must be an InterfaceQuadraturePolicy.")
        self.law_id = canonical_identifier(law_id, "law_id")
        self.binding = binding
        self.volume = volume
        self.boundary = boundary
        self.projection_order = (
            None
            if projection_order is None
            else positive_integer(projection_order, "projection_order")
        )
        self.interface_source = interface_source
        self.quadrature = policy

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        return (self.binding,)

    def _source_load(self, actions: _BoundaryIntegralActions, /) -> Array | None:
        if self.interface_source is None:
            return None
        points = actions.projection.sample_points
        density = jnp.asarray(self.interface_source(points))
        if density.shape != points.shape[:2] or not jnp.issubdtype(
            density.dtype, jnp.floating
        ):
            raise ValueError(
                "interface_source must return one real value per projection point, "
                f"shape {points.shape[:2]}."
            )
        return actions.density_load(density.astype(points.dtype))

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        from ...operators.integral.layer_potential import (
            prepare_boundary_trace_projection_2d,
            prepare_exterior_laplace_dirichlet_2d,
        )

        owners = _boundary_components(self.volume, self.boundary, components)
        volume = _boundary_side_data(owners[0], self.volume)
        degree = _trace_degree(volume.trace)
        galerkin = owners[1].galerkin
        projection = prepare_boundary_trace_projection_2d(
            galerkin.spaces, order=_projection_order(degree, self.projection_order)
        )
        curve = galerkin.curve
        vertices = np.asarray(curve.vertices)
        ends = np.asarray(curve.panel_vertices)
        side, coverage = prepare_boundary_panel_resampling(
            volume.trace,
            (
                vertices[ends[:, 0]],
                vertices[ends[:, 1]] - vertices[ends[:, 0]],
                -np.asarray(curve.normals),
            ),
            np.asarray(projection.sample_points),
            policy=self.quadrature,
        )
        _require_boundary_binding(
            self.binding,
            self.volume,
            self.boundary,
            owners,
            volume.trace,
            np.asarray(projection.sample_points),
            interface_owners,
        )
        actions = _BoundaryIntegralActions(side, projection)
        source = self._source_load(actions)
        dirichlet_space = galerkin.spaces.dirichlet_trace.vector_space
        return PreparedLaw(
            self.law_id,
            binding_id=self.binding.binding_id,
            state_blocks=(LawBlock("dirichlet-trace", dirichlet_space),),
            row_blocks=(LawBlock("trace-projection", DualSpace(dirichlet_space)),),
            contributions=_boundary_contributions(
                self.law_id, volume, owners[1], actions, source
            ),
            impositions=_boundary_impositions(self.law_id, volume, owners),
            certificate=_BoundaryIntegralCertificate(
                self.law_id,
                owners[1].name,
                volume,
                actions,
                galerkin,
                prepare_exterior_laplace_dirichlet_2d(galerkin),
                jnp.zeros(
                    volume.trace.coefficient_space.shape,
                    dtype=projection.sample_points.dtype,
                )
                if source is None
                else source,
                owners[1].far_field_tolerance,
            ),
            evidence=BoundaryIntegralEvidence(
                trace_degree=degree,
                projection=projection,
                galerkin=galerkin,
                interface_source=source is not None,
                coverage=coverage,
            ),
        )


__all__ = [
    "AbstractCouplingLaw",
    "AbstractInterfaceFlux",
    "AbstractLawCertificate",
    "BoundaryIntegralEvidence",
    "BoundaryIntegralSide",
    "BoundaryIntegralTransmissionLaw",
    "ConservativeFluxEvidence",
    "ConservativeFluxLaw",
    "EliminationEvidence",
    "FieldTransferEvidence",
    "FieldTransferLaw",
    "GapRadiation",
    "IntegralPortEvidence",
    "IntegralPortLaw",
    "InterfaceConductance",
    "InterfaceDefectReport",
    "MatchingElimination",
    "MortarEvidence",
    "MortarImposition",
    "MortarMultiplier",
    "MultiplierFamily",
    "PortSide",
    "PreparedLaw",
    "ScalarTransmissionLaw",
    "TransmissionImposition",
    "TransmissionSide",
]

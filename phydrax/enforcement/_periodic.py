#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared enforcement of `phydrax.conditions.Periodic` seam relations.

Three routes realize the same declarations:

- ``"analytic"`` lifts the seam residual with the exact centered Bernoulli
  endpoint basis of each identified coordinate. For declarations ``J_i`` on one
  coordinate the lift is ``u + sum_n q_n(x) [R (g - J u)]_n`` where ``R`` is the
  native minimum-norm right inverse of the exact endpoint matrix
  ``M_in = T_i q_n(b) - Gamma_i S_i q_n(a)``. Several identified coordinates are
  composed sequentially; the composition is exact for every declaration when the
  transported seam targets are compatible at seam intersections.
- ``"coefficient"`` eliminates the conditions in an explicit finite linear
  representation through `CoefficientElimination`.
- ``"construction"`` admits fields whose model carries a matching
  `PeriodicInputCertificate` and leaves them unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._model import PERIODIC_INPUT_CERTIFICATE_KEY, PeriodicInputCertificate
from .._model._protocols import TRIAL_SPACE_CERTIFICATE_KEY
from .._polynomial._endpoint import EndpointJetBasis
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..conditions._evidence import ConditionRealizationStamp
from ..conditions._functional import EventLinearMap
from ..conditions._ir import (
    codomains_compatible,
    Condition,
    FieldCodomain,
    FieldSpec,
    ProductCodomain,
    ProductFieldSpec,
)
from ..conditions._lowering import BoundCondition
from ..conditions._periodic import _field_value_issue, Periodic, PeriodicTraceAction
from ..conditions._relations import Equality
from ..domain import (
    Domain,
    DomainFunction,
    PointwiseEvaluator,
)
from ..domain._model_function import ConcatenatedModelEvaluator
from ..domain.decomposition._periodic import PeriodicIdentification
from ..linalg import ConstraintOperatorPlan, DenseLinearOperator
from ..typing import checked, parse, PRNGKey
from ._fiber import AnalyticFiberProjectionUnit, FiberProjectionState
from ._lifecycle import (
    commit_refresh,
    propose_refresh,
    RealizationLifecycleState,
    record_realization_stamp,
    RefreshValidation,
    validate_refresh,
)
from ._linear_representation import AbstractLinearRepresentation, CoefficientElimination
from ._realization import (
    AbstractFieldRealization,
    ConditionEvaluationContext,
    FieldRealizationResult,
    RealizationAdmission,
    RealizationStatus,
)


PeriodicProjectionRoute: TypeAlias = Literal["analytic", "coefficient", "construction"]
PeriodicEqualityScope: TypeAlias = Literal[
    "continuum", "finite-representation", "structural"
]
PeriodicPreservationStatus: TypeAlias = Literal["certified", "probed"]
PeriodicDataCompatibility: TypeAlias = Literal["certified", "probed"]


class PeriodicResourcePolicy(StrictModule):
    """Explicit capacities of one prepared periodic projection.

    ``maximum_axes`` bounds the identified coordinates per field. Each axis
    requests the source and target jets declared by its actions. Sequential
    composition bounds base-field work by ``product(1 + jets_axis)``, including
    the query, before compiler reuse. ``maximum_order`` bounds the matched jet
    order and ``maximum_rows`` the endpoint right-inverse rows (declarations times
    event size) per coordinate.
    """

    maximum_axes: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    maximum_rows: int = eqx.field(static=True)
    maximum_endpoint_evaluations: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_axes: int = 3,
        maximum_order: int = 8,
        maximum_rows: int = 256,
        maximum_endpoint_evaluations: int = 125,
    ) -> None:
        values = (maximum_axes, maximum_order, maximum_rows, maximum_endpoint_evaluations)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("Periodic resource limits must be integers.")
        if (
            any(
                value <= 0
                for value in (maximum_axes, maximum_rows, maximum_endpoint_evaluations)
            )
            or maximum_order < 0
        ):
            raise ValueError(
                "Periodic resource limits must be positive; maximum_order may be zero."
            )
        self.maximum_axes = maximum_axes
        self.maximum_order = maximum_order
        self.maximum_rows = maximum_rows
        self.maximum_endpoint_evaluations = maximum_endpoint_evaluations


class PeriodicAxisEvidence(StrictModule, NonTrainableState):
    """Endpoint right-inverse evidence of one identified coordinate of one field."""

    field: str = eqx.field(static=True)
    identification_id: str = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    basis_size: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    rows: int = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    max_order: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        field: str,
        identification_id: str,
        condition_ids: Sequence[str],
        basis_size: int,
        rank: int,
        rows: int,
        condition_number: float,
        max_order: int,
    ) -> None:
        self.field = field
        self.identification_id = identification_id
        self.condition_ids = tuple(condition_ids)
        self.basis_size = basis_size
        self.rank = rank
        self.rows = rows
        self.condition_number = condition_number
        self.max_order = max_order


class PeriodicPreservationRecord(StrictModule, NonTrainableState):
    """One earlier field contract kept satisfied by the periodic projection.

    ``"certified"`` contracts are preserved structurally (their data is constant or
    independent of every identified coordinate). ``"probed"`` contracts showed no
    seam incompatibility on ``probes`` sampled points; their exactness depends on
    the supplied data being seam-compatible, which finite probing cannot prove.
    """

    field: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    status: PeriodicPreservationStatus = eqx.field(static=True)
    probes: int = eqx.field(static=True)
    maximum_defect: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        field: str,
        contract_id: str,
        status: PeriodicPreservationStatus,
        probes: int,
        maximum_defect: float,
    ) -> None:
        self.field = field
        self.contract_id = contract_id
        self.status = parse(status, PeriodicPreservationStatus, "status")
        self.probes = probes
        self.maximum_defect = maximum_defect


class PeriodicProjectionEvidence(StrictModule, NonTrainableState):
    """Route, scope, numerical, regularity, and composition evidence."""

    route: PeriodicProjectionRoute = eqx.field(static=True)
    scope: PeriodicEqualityScope = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    axes: tuple[PeriodicAxisEvidence, ...]
    required_regularity: tuple[tuple[str, str, int], ...] = eqx.field(static=True)
    endpoint_evaluations: int = eqx.field(static=True)
    preserved: tuple[PeriodicPreservationRecord, ...]
    construction_certificates: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        route: PeriodicProjectionRoute,
        scope: PeriodicEqualityScope,
        condition_ids: Sequence[str],
        field_names: Sequence[str],
        axes: Sequence[PeriodicAxisEvidence] = (),
        required_regularity: Sequence[tuple[str, str, int]] = (),
        endpoint_evaluations: int = 0,
        preserved: Sequence[PeriodicPreservationRecord] = (),
        construction_certificates: Sequence[str] = (),
    ) -> None:
        self.route = parse(route, PeriodicProjectionRoute, "route")
        self.scope = parse(scope, PeriodicEqualityScope, "scope")
        self.condition_ids = tuple(condition_ids)
        self.field_names = tuple(field_names)
        self.axes = tuple(axes)
        self.required_regularity = tuple(required_regularity)
        self.endpoint_evaluations = endpoint_evaluations
        self.preserved = tuple(preserved)
        self.construction_certificates = tuple(construction_certificates)

    @property
    def preserved_ids(self) -> tuple[str, ...]:
        """Contracts certified preserved; probed contracts are excluded."""
        return tuple(
            record.contract_id
            for record in self.preserved
            if record.status == "certified"
        )


def _flat_event_size(shape: tuple[int, ...], /) -> int:
    return math.prod(shape) if shape else 1


def _jet_row(
    jets: np.ndarray, orders: tuple[int, ...], coefficients: np.ndarray, side: int, /
) -> np.ndarray:
    row = np.zeros((jets.shape[-1],), dtype=np.result_type(coefficients, np.float64))
    for order, coefficient in zip(orders, coefficients, strict=True):
        row = row + coefficient * jets[side, order]
    return row


def _endpoint_matrix(
    conditions: tuple[Periodic, ...],
    jets: np.ndarray,
    event: tuple[int, ...],
    /,
) -> np.ndarray:
    size = _flat_event_size(event)
    identity = np.eye(size)
    blocks = []
    for condition in conditions:
        target = _jet_row(
            jets,
            condition.target_action.orders,
            np.asarray(condition.target_action.coefficients),
            1,
        )
        source = _jet_row(
            jets,
            condition.source_action.orders,
            np.asarray(condition.source_action.coefficients),
            0,
        )
        transport = (
            np.asarray(condition.transport.matrix)
            if isinstance(condition.transport, EventLinearMap)
            else np.asarray(condition.transport) * identity
        )
        blocks.append(
            np.kron(target[None, :], identity) - np.kron(source[None, :], transport)
        )
    return np.concatenate(blocks, axis=0)


class _SeamLiftEvaluator(StrictModule):
    """Evaluate ``sum_n q_n(x) [R z(y)]_n`` at one point."""

    basis: EndpointJetBasis
    right_inverse: Array
    residuals: tuple[DomainFunction, ...]
    positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coordinate_position: int = eqx.field(static=True)
    component: int | None = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        basis: EndpointJetBasis,
        right_inverse: Array,
        residuals: tuple[DomainFunction, ...],
        positions: tuple[tuple[int, ...], ...],
        coordinate_position: int,
        component: int | None,
        lower: float,
        length: float,
        event_shape: tuple[int, ...],
        /,
    ) -> None:
        self.basis = basis
        self.right_inverse = right_inverse
        self.residuals = residuals
        self.positions = positions
        self.coordinate_position = coordinate_position
        self.component = component
        self.lower = lower
        self.length = length
        self.event_shape = event_shape

    def __call__(self, *args: Any, key: PRNGKey | None = None, **kwargs: Any) -> Array:
        values = tuple(
            jnp.broadcast_to(
                jnp.asarray(
                    residual.func(*(args[index] for index in position), key=key, **kwargs)
                ),
                self.event_shape,
            ).reshape((-1,))
            for residual, position in zip(self.residuals, self.positions, strict=True)
        )
        stacked = jnp.concatenate(values)
        size = _flat_event_size(self.event_shape)
        # The lift adopts the residual precision: a real right inverse acting on a
        # complex residual stays complex, and nothing widens a single-precision field.
        dtype = (
            stacked.dtype
            if jnp.issubdtype(stacked.dtype, jnp.complexfloating)
            or not jnp.issubdtype(self.right_inverse.dtype, jnp.complexfloating)
            else np.result_type(stacked.dtype, np.complex64)
        )
        inverse = self.right_inverse.astype(dtype)
        inverse = eqx.error_if(
            inverse,
            ~jnp.all(jnp.isfinite(inverse))
            | jnp.any((self.right_inverse != 0) & (inverse == 0)),
            "Periodic right inverse is not representable at the field precision.",
        )
        coefficients = (inverse @ stacked.astype(dtype)).reshape(
            (self.basis.basis_size, size)
        )
        coordinate = jnp.asarray(args[self.coordinate_position])
        value = coordinate if self.component is None else coordinate[self.component]
        normalized = (value - self.lower) / self.length
        basis = self.basis.values(normalized.astype(jnp.finfo(coefficients.dtype).dtype))
        value = (basis.astype(coefficients.dtype) @ coefficients).reshape(
            self.event_shape
        )
        return eqx.error_if(
            value,
            ~jnp.all(jnp.isfinite(value)),
            "Periodic lift produced nonfinite values.",
        )


class _PreparedSeamAxis(StrictModule):
    """One identified coordinate of one field with its exact endpoint right inverse."""

    identification: PeriodicIdentification
    conditions: tuple[Periodic, ...]
    basis: EndpointJetBasis
    right_inverse: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        identification: PeriodicIdentification,
        conditions: tuple[Periodic, ...],
        basis: EndpointJetBasis,
        right_inverse: Array,
        event_shape: tuple[int, ...],
        /,
    ) -> None:
        self.identification = identification
        self.conditions = conditions
        self.basis = basis
        self.right_inverse = right_inverse
        self.event_shape = event_shape

    # These methods feed source-addressed seam callbacks. Keep their code identity
    # unwrapped; the prepared constructor owns their nominal state.
    def actions(self, u: DomainFunction, /) -> tuple[DomainFunction, ...]:
        """Homogeneous seam actions ``J_i u`` (constant along the coordinate)."""
        return tuple(condition.action(u, u) for condition in self.conditions)

    def lift(
        self, residuals: tuple[DomainFunction, ...], domain: Domain, /
    ) -> DomainFunction:
        """Right-inverse lift of seam residual fields into the fundamental domain."""
        label = self.identification.label
        deps = tuple(
            name
            for name in domain.labels
            if name == label or any(name in residual.deps for residual in residuals)
        )
        index = {name: position for position, name in enumerate(deps)}
        promoted = tuple(residual.promote(domain) for residual in residuals)
        return DomainFunction(
            domain=domain,
            deps=deps,
            func=PointwiseEvaluator(
                _SeamLiftEvaluator(
                    self.basis,
                    self.right_inverse,
                    promoted,
                    tuple(
                        tuple(index[name] for name in residual.deps)
                        for residual in promoted
                    ),
                    index[label],
                    self.identification.component
                    if self.identification.vector_coordinate
                    else None,
                    self.identification.lower,
                    self.identification.period,
                    self.event_shape,
                )
            ),
            metadata={},
        )


class _FieldSeamProgram(StrictModule):
    """Sequential seam projection of one field over its identified coordinates."""

    field: str = eqx.field(static=True)
    domain: Domain
    axes: tuple[_PreparedSeamAxis, ...]

    @checked
    def __init__(
        self, field: str, domain: Domain, axes: tuple[_PreparedSeamAxis, ...], /
    ) -> None:
        self.field = field
        self.domain = domain
        self.axes = axes

    @property
    def conditions(self) -> tuple[Periodic, ...]:
        return tuple(condition for axis in self.axes for condition in axis.conditions)

    # Bound callbacks are source-addressed by compiler identity and retain their
    # original functions. Their array-bearing owners remain dynamic PyTree leaves.
    def action(
        self, fields: Mapping[str, Any], context: Any, /
    ) -> tuple[DomainFunction, ...]:
        del context
        u = fields[self.field]
        return tuple(value for axis in self.axes for value in axis.actions(u))

    def target(
        self, fields: Mapping[str, Any], context: Any, /
    ) -> tuple[DomainFunction, ...]:
        del fields, context
        return tuple(condition.target for condition in self.conditions)

    def lift(self, residual: Any, context: Any, /) -> frozendict[str, DomainFunction]:
        del context
        values = tuple(residual)
        correction: DomainFunction | None = None
        offset = 0
        for axis in self.axes:
            count = len(axis.conditions)
            block = values[offset : offset + count]
            offset += count
            if correction is not None:
                block = tuple(
                    value - action
                    for value, action in zip(block, axis.actions(correction), strict=True)
                )
            step = axis.lift(block, self.domain)
            correction = step if correction is None else correction + step
        if correction is None:
            raise RuntimeError("A prepared seam program lost its coordinates.")
        return frozendict({self.field: correction})

    def residual_codomain(self) -> ProductCodomain:
        return ProductCodomain(
            tuple(
                FieldCodomain(condition.on, condition.value)
                for condition in self.conditions
            )
        )


def _joint_condition(
    conditions: tuple[Periodic, ...], condition_id: str | None, /
) -> Condition:
    sources = tuple(
        dict.fromkeys(name for condition in conditions for name in condition.fields)
    )
    supports: dict[str, FieldCodomain] = {}
    for condition in conditions:
        codomain = FieldCodomain(
            condition.identification.domain.component(), condition.value
        )
        for name in condition.fields:
            if name in supports and not codomains_compatible(supports[name], codomain):
                raise ValueError(
                    f"Periodic declarations of field {name!r} disagree on its "
                    "fundamental support or value codomain."
                )
            supports[name] = codomain
    return Condition(
        condition_id
        or canonical_fingerprint(
            {
                "kind": "periodic-projection-condition",
                "conditions": [condition.condition_id for condition in conditions],
            }
        ),
        ProductFieldSpec(tuple(FieldSpec(name, supports[name]) for name in sources)),
        PeriodicTraceAction(conditions),
        ProductCodomain(
            tuple(
                FieldCodomain(condition.on, condition.value) for condition in conditions
            )
        ),
        Equality(tuple(condition.target for condition in conditions)),
    )


def _validate_corner_compatibility(axes: tuple[_PreparedSeamAxis, ...], /) -> None:
    """Certify sequential exactness: ``J_j g_k = J_k g_j`` at seam intersections."""
    if len(axes) < 2:
        return
    for first_index, first in enumerate(axes):
        for second in axes[first_index + 1 :]:
            for left in first.conditions:
                for right in second.conditions:
                    _require_commuting_transports(left, right)
                    if left.homogeneous and right.homogeneous:
                        continue
                    if left.target_constant is None or right.target_constant is None:
                        raise ValueError(
                            "Seam intersections of several identified coordinates require "
                            "constant affine targets so their compatibility can be "
                            "certified; supply constant targets or enforce the "
                            "non-constant jump on one coordinate only."
                        )
                    lhs = _constant_action(left, right.target_constant)
                    rhs = _constant_action(right, left.target_constant)
                    if not bool(
                        np.allclose(
                            np.asarray(lhs), np.asarray(rhs), rtol=0.0, atol=1e-12
                        )
                    ):
                        raise ValueError(
                            "Incompatible periodic seam targets at an intersection of "
                            f"{left.identification.identification_id!r} and "
                            f"{right.identification.identification_id!r}: J g differ "
                            f"({np.asarray(lhs)} != {np.asarray(rhs)})."
                        )


@checked
def _transport_matrix(condition: Periodic, /) -> np.ndarray:
    transport = condition.transport
    if isinstance(transport, EventLinearMap):
        return np.asarray(transport.matrix)
    return np.asarray(transport) * np.eye(_flat_event_size(condition.value.shape))


@checked
def _require_commuting_transports(left: Periodic, right: Periodic, /) -> None:
    """Refuse sequential seam projections whose event transports do not commute.

    The axis lift of one coordinate preserves the seam of another exactly when
    its transport commutes with the other transport and its adjoint (then the
    endpoint right inverse commutes too). Scalar transports always commute.
    """
    first = _transport_matrix(left)
    second = _transport_matrix(right)
    if first.shape != second.shape:
        raise ValueError(
            "Periodic declarations on one field must share one event size across "
            "identified coordinates."
        )
    adjoint = second.conj().T
    if not (
        np.array_equal(first @ second, second @ first)
        and np.array_equal(first @ adjoint, adjoint @ first)
    ):
        raise ValueError(
            "The event transports of identified coordinates "
            f"{left.identification.identification_id!r} and "
            f"{right.identification.identification_id!r} do not commute, so their "
            "seam projections cannot be composed exactly; enforce them jointly in a "
            "linear representation (route='coefficient')."
        )


@checked
def _constant_action(condition: Periodic, value: Array, /) -> np.ndarray:
    """Host preparation of the seam action on a constant, with its event layout."""
    target = sum(
        coefficient
        for order, coefficient in zip(
            condition.target_action.orders,
            np.asarray(condition.target_action.coefficients),
            strict=True,
        )
        if order == 0
    )
    source = sum(
        coefficient
        for order, coefficient in zip(
            condition.source_action.orders,
            np.asarray(condition.source_action.coefficients),
            strict=True,
        )
        if order == 0
    )
    event = np.broadcast_to(np.asarray(value), condition.value.shape)
    transported = (
        np.asarray(condition.transport.matrix) @ event
        if isinstance(condition.transport, EventLinearMap)
        else np.asarray(condition.transport) * event
    )
    return np.asarray(target * event - source * transported)


@checked
def _prepare_axis(
    field: str,
    identification: PeriodicIdentification,
    conditions: tuple[Periodic, ...],
    resources: PeriodicResourcePolicy,
    /,
) -> tuple[_PreparedSeamAxis, PeriodicAxisEvidence]:
    value = conditions[0].value
    if any(not codomains_compatible(condition.value, value) for condition in conditions):
        raise ValueError(
            f"Periodic declarations of field {field!r} on one coordinate must share one "
            "value codomain."
        )
    event = value.shape
    max_order = max(condition.max_order for condition in conditions)
    if max_order > resources.maximum_order:
        raise ValueError(
            f"Periodic jet order {max_order} exceeds the resource limit "
            f"{resources.maximum_order}."
        )
    rows = len(conditions) * _flat_event_size(event)
    if rows > resources.maximum_rows:
        raise ValueError(
            f"Periodic endpoint system with {rows} rows exceeds the resource limit "
            f"{resources.maximum_rows}."
        )
    basis = EndpointJetBasis(
        max_order, identification.period, maximum_order=resources.maximum_order
    )
    jets = basis.endpoint_jets(identification.period)
    matrix = _endpoint_matrix(conditions, jets, event)
    dtype = jnp.complex128 if np.iscomplexobj(matrix) else jnp.float64
    prepared = ConstraintOperatorPlan(
        DenseLinearOperator(jnp.asarray(matrix, dtype=dtype)),
        require_full_row_rank=False,
    ).prepare()
    if not prepared.evidence.full_row_rank:
        raise ValueError(
            f"Periodic declarations of field {field!r} on coordinate "
            f"{identification.label!r} are linearly dependent (rank "
            f"{prepared.evidence.rank} of {rows} endpoint rows); remove the redundant "
            "declaration."
        )
    singular = np.asarray(prepared.evidence.singular_values)
    condition_number = float(singular[0] / singular[-1])
    axis = _PreparedSeamAxis(
        identification, conditions, basis, prepared.right_inverse, event
    )
    evidence = PeriodicAxisEvidence(
        field=field,
        identification_id=identification.identification_id,
        condition_ids=tuple(condition.condition_id for condition in conditions),
        basis_size=basis.basis_size,
        rank=prepared.evidence.rank,
        rows=rows,
        condition_number=condition_number,
        max_order=max_order,
    )
    return axis, evidence


def _group_conditions(
    conditions: tuple[Periodic, ...], /
) -> dict[str, dict[tuple[str, int | None], list[Periodic]]]:
    grouped: dict[str, dict[tuple[str, int | None], list[Periodic]]] = {}
    for condition in conditions:
        if not condition.same_field:
            raise ValueError(
                "Hard periodic enforcement corrects one field across its own seam; "
                f"the declaration coupling {condition.source_field!r} to "
                f"{condition.target_field!r} is soft-only (use ResidualPenalty)."
            )
        if (
            not condition.pairing.self_seam
            or not condition.identification.domain.same_support(condition.on.domain)
        ):
            raise ValueError(
                "Hard periodic enforcement requires the self-seam of a fundamental "
                "domain from PeriodicIdentification.pairing(); decomposed patch seams "
                "are coupled by their decomposition solver."
            )
        canonical = condition.identification.pairing()
        witness = eqx.tree_equal(
            (
                condition.pairing.component,
                condition.pairing.left_coordinates,
                condition.pairing.right_coordinates,
            ),
            (
                canonical.component,
                canonical.left_coordinates,
                canonical.right_coordinates,
            ),
        )
        if not bool(witness):
            raise ValueError(
                "Hard periodic enforcement prepares its endpoint right inverse from the "
                "identification, so the pairing must carry exactly the identification's "
                "face maps; use PeriodicIdentification.pairing()."
            )
        by_axis = grouped.setdefault(condition.source_field, {})
        coordinate = (condition.identification.label, condition.identification.component)
        by_axis.setdefault(coordinate, []).append(condition)
    return grouped


def _certificate_issue(field: Any, condition: Periodic, name: str, /) -> str | None:
    """Why ``field`` is not certified periodic for ``condition`` by construction."""
    if not isinstance(field, DomainFunction):
        return f"Periodic field {name!r} must be a DomainFunction."
    certificate = field.metadata.get(PERIODIC_INPUT_CERTIFICATE_KEY)
    if not isinstance(certificate, PeriodicInputCertificate):
        return (
            f"Field {name!r} carries no PeriodicInputCertificate; the construction "
            "route requires an explicitly certified periodic model, and local "
            "overlays applied before it do not carry that certificate."
        )
    if isinstance(field.func, ConcatenatedModelEvaluator):
        live_certificate = field.func.model_metadata().get(PERIODIC_INPUT_CERTIFICATE_KEY)
        if (
            not isinstance(live_certificate, PeriodicInputCertificate)
            or live_certificate.certificate_id != certificate.certificate_id
            or field.func.deps != field.deps
            or field.func.domain_labels != field.domain.labels
        ):
            return f"The periodic construction evidence of {name!r} is stale."
    offsets: dict[str, int] = {}
    position = 0
    for label in field.deps:
        offsets[label] = position
        position += field.domain.coordinate(label).event_size
    if certificate.input_size != position:
        return (
            f"The periodic certificate of {name!r} describes {certificate.input_size} "
            f"inputs, but the field binding supplies {position}."
        )
    identification = condition.identification
    if identification.label not in offsets:
        return (
            f"Field {name!r} does not depend on the identified coordinate "
            f"{identification.label!r}; a constant direction needs no seam."
        )
    index = offsets[identification.label] + (
        identification.component
        if identification.vector_coordinate and identification.component is not None
        else 0
    )
    period = certificate.period_of(index)
    if period is None or period != identification.period:
        return (
            f"The periodic certificate of {name!r} does not certify period "
            f"{identification.period} for input entry {index}."
        )
    if not condition.homogeneous or not condition.identity_transport:
        return (
            "A periodic construction certifies homogeneous identity-transport "
            "relations only; transported or affine seams need the analytic route "
            "or an explicit phase/affine lift."
        )
    if condition.source_action.action_id != condition.target_action.action_id:
        return "A periodic construction certifies equal source and target jets only."
    if not certificate.supports_order(condition.max_order):
        return (
            f"The periodic certificate of {name!r} declares too little "
            f"regularity for jet order {condition.max_order}."
        )
    issue = _field_value_issue(field, condition.value)
    if issue is not None:
        return f"Construction field {name!r} {issue}."
    return None


def _construction_issue(
    fields: Mapping[str, Any], declarations: tuple[Periodic, ...], /
) -> tuple[tuple[str, ...], str | None]:
    """Certificate identities of the current fields, or the first refusal reason.

    The construction route publishes fields unchanged, so it checks the fields it
    is given, not the fields seen at preparation.
    """
    certificates: dict[str, str] = {}
    for condition in declarations:
        name = condition.source_field
        issue = _certificate_issue(fields[name], condition, name)
        if issue is not None:
            return (), issue
        certificates[name] = (
            fields[name].metadata[PERIODIC_INPUT_CERTIFICATE_KEY].certificate_id
        )
    return tuple(certificates.values()), None


@checked
def _validate_traced_support(
    result: FieldRealizationResult, checks: Sequence[Array], /
) -> FieldRealizationResult:
    """Bind runtime support checks to every successful route's complete result."""
    if not checks or not result.successful:
        return result
    return eqx.error_if(
        result,
        ~jnp.all(jnp.stack(checks)),
        "Periodic field has a different fundamental support.",
    )


class PreparedPeriodicProjection(AbstractFieldRealization):
    """Jointly prepared realization of `Periodic` declarations.

    Pass it with its joint ``condition`` to
    ``EnforcementSpec(prepared.condition, realization=prepared)``. Realization is
    trace-safe: it only composes lazy `DomainFunction` corrections, so it can run
    inside compiled training steps.
    """

    condition: Condition
    route: PeriodicProjectionRoute = eqx.field(static=True)
    data_compatibility: PeriodicDataCompatibility = eqx.field(static=True)
    state: FiberProjectionState | None
    elimination: CoefficientElimination | None
    declarations: tuple[Periodic, ...]
    evidence: PeriodicProjectionEvidence
    realization_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        condition: Condition,
        route: PeriodicProjectionRoute,
        declarations: tuple[Periodic, ...],
        evidence: PeriodicProjectionEvidence,
        /,
        *,
        state: FiberProjectionState | None = None,
        elimination: CoefficientElimination | None = None,
        data_compatibility: PeriodicDataCompatibility = "certified",
    ) -> None:
        route_ = parse(route, PeriodicProjectionRoute, "route")
        match route_:
            case "analytic":
                if state is None or elimination is not None:
                    raise TypeError(
                        "The analytic route requires one fiber projection state."
                    )
            case "coefficient":
                if elimination is None or state is not None:
                    raise TypeError(
                        "The coefficient route requires one coefficient elimination."
                    )
            case "construction":
                if state is not None or elimination is not None:
                    raise TypeError("The construction route carries no correction.")
            case _:
                raise ValueError(f"Unknown periodic projection route {route_!r}.")
        self.condition = condition
        self.route = route_
        self.data_compatibility = parse(
            data_compatibility, PeriodicDataCompatibility, "data_compatibility"
        )
        self.state = state
        self.elimination = elimination
        self.declarations = declarations
        self.evidence = evidence
        self.provider_id = f"phydrax.enforcement.periodic/{route_}"
        self.realization_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-projection",
                "route": route_,
                "data_compatibility": self.data_compatibility,
                "condition": condition.condition_id,
                "declarations": [value.condition_id for value in declarations],
                "state": None if state is None else state.prepared_id,
                "elimination": None
                if elimination is None
                else elimination.realization_id,
                "preserved": [
                    [record.contract_id, record.status] for record in evidence.preserved
                ],
            }
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.evidence.field_names

    def with_preservation(
        self, records: Sequence[PeriodicPreservationRecord], /
    ) -> PreparedPeriodicProjection:
        """Return this projection bound to the earlier contracts it preserves."""
        evidence = self.evidence
        bound = PeriodicProjectionEvidence(
            route=evidence.route,
            scope=evidence.scope,
            condition_ids=evidence.condition_ids,
            field_names=evidence.field_names,
            axes=evidence.axes,
            required_regularity=evidence.required_regularity,
            endpoint_evaluations=evidence.endpoint_evaluations,
            preserved=(*evidence.preserved, *records),
            construction_certificates=evidence.construction_certificates,
        )
        return PreparedPeriodicProjection(
            self.condition,
            self.route,
            self.declarations,
            bound,
            state=self.state,
            elimination=self.elimination,
            data_compatibility=self.data_compatibility,
        )

    @checked
    def admission(self, condition: Condition, /) -> RealizationAdmission:
        del condition
        writes = () if self.route == "construction" else self.field_names
        return RealizationAdmission(
            reads=self.field_names,
            writes=writes,
            establishes=(self.condition.condition_id, *self.evidence.condition_ids),
            preserves=self.evidence.preserved_ids,
        )

    @checked
    def _failed(
        self,
        state: RealizationLifecycleState,
        context: ConditionEvaluationContext,
        status: RealizationStatus,
        message: str,
        /,
    ) -> FieldRealizationResult:
        proposal = propose_refresh((), state, context=context)
        failed = commit_refresh(
            state,
            proposal,
            RefreshValidation.reject(status, message=message, evidence=self.evidence),
        )
        return FieldRealizationResult.failure(
            status, state=failed, message=message, evidence=self.evidence
        )

    @checked
    def realize(
        self,
        fields: Mapping[str, Any],
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        current = RealizationLifecycleState.initial() if state is None else state
        if context.condition_id != self.condition.condition_id:
            return self._failed(
                current,
                context,
                RealizationStatus.INVALID_INPUT,
                "The evaluation context names a different condition than this projection.",
            )
        missing = tuple(name for name in self.field_names if name not in fields)
        if missing:
            return self._failed(
                current,
                context,
                RealizationStatus.INVALID_INPUT,
                f"Periodic projection fields {missing!r} are missing.",
            )
        if any(not isinstance(fields[name], DomainFunction) for name in self.field_names):
            return self._failed(
                current,
                context,
                RealizationStatus.INVALID_INPUT,
                "Periodic projection fields must be DomainFunction values.",
            )
        support_checks: list[Array] = []
        for declaration in self.declarations:
            for name in declaration.fields:
                expected = declaration.identification.domain
                actual = fields[name].domain
                equality = eqx.tree_equal(expected, actual)
                if isinstance(equality, jax_core.Tracer):
                    support_checks.append(equality)
                elif not expected.same_support(actual):
                    return self._failed(
                        current,
                        context,
                        RealizationStatus.INVALID_INPUT,
                        f"Periodic field {name!r} has a different fundamental support.",
                    )
        if self.route == "construction":
            certificates, issue = _construction_issue(fields, self.declarations)
            if (
                issue is not None
                or certificates != self.evidence.construction_certificates
            ):
                return self._failed(
                    current,
                    context,
                    RealizationStatus.UNSUPPORTED,
                    issue
                    or "The fields carry different periodic construction certificates "
                    "than the prepared projection; prepare the projection again.",
                )
        certified = tuple(
            name
            for name in self.field_names
            if TRIAL_SPACE_CERTIFICATE_KEY in fields[name].metadata
        )
        if certified and self.route == "analytic":
            return self._failed(
                current,
                context,
                RealizationStatus.UNSUPPORTED,
                f"Fields {certified!r} are certified exact PDE trial fields; an analytic "
                "seam lift does not preserve their trial space. Use the coefficient route.",
            )
        if self.route == "coefficient":
            if self.elimination is None:
                raise RuntimeError("The coefficient route lost its elimination.")
            return _validate_traced_support(
                self.elimination.realize(fields, current, context=context), support_checks
            )
        proposal = propose_refresh((), current, context=context)
        validation = validate_refresh(proposal)
        committed = commit_refresh(current, proposal, validation)
        if not validation.accepted:
            return FieldRealizationResult.failure(
                validation.status,
                state=committed,
                message=validation.message,
                evidence=validation.evidence,
            )
        stamp = ConditionRealizationStamp(
            context.condition_id,
            canonical_fingerprint(
                {
                    "kind": "periodic-projection-source",
                    "generation": committed.generation,
                    "parameter_revision": context.parameter_revision,
                }
            ),
            self.realization_id,
            self.provider_id,
            quantifier=context.quantifier,
            exact=True,
        )
        if self.route == "construction":
            result = FieldRealizationResult.success(
                fields,
                state=record_realization_stamp(committed, stamp),
                stamp=stamp,
                evidence=self.evidence,
                unchanged=True,
            )
            return _validate_traced_support(result, support_checks)
        if self.state is None:
            raise RuntimeError("The analytic route lost its fiber state.")
        projected = self.state.project_analytic(fields, context)
        result = FieldRealizationResult.success(
            projected,
            state=record_realization_stamp(committed, stamp),
            stamp=stamp,
            evidence=self.evidence,
        )
        return _validate_traced_support(result, support_checks)

    def seam_defect(
        self,
        fields: Mapping[str, DomainFunction],
        sampling: Any,
        /,
        *,
        key: PRNGKey,
    ) -> Array:
        """Maximum absolute seam residual of ``fields`` on sampled seam points.

        This is sampled observation evidence for diagnostics, not a proof.
        """
        defects = tuple(
            jnp.max(
                jnp.abs(
                    declaration.residual(fields)(
                        declaration.on.sample(sampling, key=key)
                    ).data
                )
            )
            for declaration in self.declarations
        )
        return jnp.max(jnp.stack(defects))


@checked
def prepare_periodic_projection(
    functions: Mapping[str, Any],
    conditions: Sequence[Periodic],
    /,
    *,
    route: PeriodicProjectionRoute,
    representation: AbstractLinearRepresentation | None = None,
    resources: PeriodicResourcePolicy | None = None,
    condition_id: str | None = None,
    data_compatibility: PeriodicDataCompatibility = "certified",
) -> PreparedPeriodicProjection:
    """Prepare one joint realization of periodic seam declarations.

    **Arguments:**

    - `functions`: Solver fields; every declared field must be a `DomainFunction`
      on the identification's fundamental domain.
    - `conditions`: `Periodic` declarations, one per matched trace. Declarations on
      one field and coordinate are solved jointly; several identified coordinates
      are composed sequentially.
    - `route`: ``"analytic"`` (continuum endpoint-jet lift), ``"coefficient"``
      (finite representation elimination; requires `representation`), or
      ``"construction"`` (fields certified periodic by construction).
    - `representation`: Explicit linear representation for the coefficient route.
    - `resources`: Explicit capacities; see `PeriodicResourcePolicy`.
    - `condition_id`: Optional identity of the joint typed condition.
    - `data_compatibility`: How earlier hard wall and initial contracts on the
      field must be shown seam-compatible before the compiler composes them with
      this projection. ``"certified"`` (default) admits only data whose
      compatibility is proven exactly: constants, data independent of every
      identified coordinate on a homogeneous annihilating seam, or data carrying a
      matching `PeriodicInputCertificate`. ``"probed"`` additionally admits
      function data that shows no seam defect on sampled points; such contracts
      are reported as ``"probed"`` evidence and are never claimed preserved.

    **Returns:** a `PreparedPeriodicProjection`; use
    ``EnforcementSpec(prepared.condition, realization=prepared)``.
    """
    route_ = parse(route, PeriodicProjectionRoute, "route")
    compatibility = parse(
        data_compatibility, PeriodicDataCompatibility, "data_compatibility"
    )
    declarations = tuple(conditions)
    if not declarations or any(not isinstance(value, Periodic) for value in declarations):
        raise TypeError(
            "conditions must be a nonempty sequence of Periodic declarations."
        )
    identities = tuple(value.condition_id for value in declarations)
    if len(set(identities)) != len(identities):
        raise ValueError("Periodic declarations must be unique.")
    resources_ = PeriodicResourcePolicy() if resources is None else resources
    if route_ != "coefficient" and representation is not None:
        raise ValueError("Only the coefficient route consumes a representation.")
    joint = _joint_condition(declarations, condition_id)
    field_names = joint.fields.sources
    missing = tuple(name for name in field_names if name not in functions)
    if missing:
        raise KeyError(f"Unknown periodic fields {missing!r}.")
    for name in field_names:
        field = functions[name]
        if not isinstance(field, DomainFunction):
            raise TypeError(f"Periodic field {name!r} must be a DomainFunction.")
    for declaration in declarations:
        for name in declaration.fields:
            if not declaration.identification.domain.same_support(functions[name].domain):
                raise ValueError(
                    f"Periodic field {name!r} does not live on the identified "
                    "fundamental domain."
                )
    match route_:
        case "coefficient":
            if representation is None:
                raise TypeError(
                    "The coefficient route requires an AbstractLinearRepresentation."
                )
            bound = BoundCondition(joint, {name: functions[name] for name in field_names})
            elimination = CoefficientElimination(
                representation, representation.assemble(bound)
            )
            evidence = PeriodicProjectionEvidence(
                route=route_,
                scope="finite-representation",
                condition_ids=identities,
                field_names=field_names,
            )
            return PreparedPeriodicProjection(
                joint,
                route_,
                declarations,
                evidence,
                elimination=elimination,
                data_compatibility=compatibility,
            )
        case "construction":
            _group_conditions(declarations)
            certificates, issue = _construction_issue(functions, declarations)
            if issue is not None:
                raise ValueError(issue)
            evidence = PeriodicProjectionEvidence(
                route=route_,
                scope="structural",
                condition_ids=identities,
                field_names=field_names,
                construction_certificates=certificates,
            )
            return PreparedPeriodicProjection(
                joint,
                route_,
                declarations,
                evidence,
                data_compatibility=compatibility,
            )
        case "analytic":
            return _prepare_analytic(
                joint, declarations, functions, resources_, compatibility
            )
        case _:
            raise ValueError(f"Unknown periodic projection route {route_!r}.")


@checked
def _prepare_analytic(
    joint: Condition,
    declarations: tuple[Periodic, ...],
    functions: Mapping[str, Any],
    resources: PeriodicResourcePolicy,
    data_compatibility: PeriodicDataCompatibility,
    /,
) -> PreparedPeriodicProjection:
    grouped = _group_conditions(declarations)
    units = []
    axis_evidence = []
    regularity = []
    evaluations = 0
    for name, by_axis in grouped.items():
        field = functions[name]
        if TRIAL_SPACE_CERTIFICATE_KEY in field.metadata:
            raise ValueError(
                f"Field {name!r} is a certified exact PDE trial field; an analytic seam "
                "lift does not preserve its trial space. Use the coefficient route."
            )
        if len(by_axis) > resources.maximum_axes:
            raise ValueError(
                f"Field {name!r} has {len(by_axis)} identified coordinates, above the "
                f"resource limit {resources.maximum_axes}."
            )
        evaluations_field = math.prod(
            1
            + sum(
                len(condition.source_action.orders) + len(condition.target_action.orders)
                for condition in conditions
            )
            for conditions in by_axis.values()
        )
        if evaluations_field > resources.maximum_endpoint_evaluations:
            raise ValueError(
                f"Field {name!r} needs {evaluations_field} endpoint evaluations per "
                f"query, above the resource limit {resources.maximum_endpoint_evaluations}."
            )
        evaluations = max(evaluations, evaluations_field)
        axes = []
        for conditions in by_axis.values():
            identification = conditions[0].identification
            if not identification.domain.same_support(field.domain):
                raise ValueError(
                    f"Field {name!r} does not live on the identified fundamental domain."
                )
            axis, evidence = _prepare_axis(
                name, identification, tuple(conditions), resources
            )
            axes.append(axis)
            axis_evidence.append(evidence)
            regularity.append((name, identification.label, evidence.max_order))
        prepared_axes = tuple(axes)
        _validate_corner_compatibility(prepared_axes)
        program = _FieldSeamProgram(name, field.domain, prepared_axes)
        units.append(
            AnalyticFiberProjectionUnit(
                program.action,
                program.target,
                program.lift,
                program.residual_codomain(),
                field_names=(name,),
                condition_ids=tuple(
                    condition.condition_id for condition in program.conditions
                ),
                evidence=tuple(axis_evidence),
                exactness_scope="continuum",
            )
        )
    evidence = PeriodicProjectionEvidence(
        route="analytic",
        scope="continuum",
        condition_ids=tuple(value.condition_id for value in declarations),
        field_names=joint.fields.sources,
        axes=tuple(axis_evidence),
        required_regularity=tuple(regularity),
        endpoint_evaluations=evaluations,
    )
    return PreparedPeriodicProjection(
        joint,
        "analytic",
        declarations,
        evidence,
        state=FiberProjectionState(tuple(units)),
        data_compatibility=data_compatibility,
    )


__all__ = [
    "PeriodicAxisEvidence",
    "PeriodicDataCompatibility",
    "PeriodicEqualityScope",
    "PeriodicPreservationRecord",
    "PeriodicPreservationStatus",
    "PeriodicProjectionEvidence",
    "PeriodicProjectionRoute",
    "PeriodicResourcePolicy",
    "PreparedPeriodicProjection",
    "prepare_periodic_projection",
]

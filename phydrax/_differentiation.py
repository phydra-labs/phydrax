#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical derivative vocabulary, regularity algebra, admission, and authority."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


RegularityPieces: TypeAlias = Literal["none", "polynomial", "smooth"]

DERIVATIVE_SUPPORTED = "derivative-supported"
DERIVATIVE_UNSUPPORTED = "derivative-unsupported"

_SINGULAR_PART_IGNORED = "singular-part-ignored"
_ROUTE_STOPPED = "route-stopped"
_BRANCH_MARGIN = "branch-margin"
_REGULARITY_UNDECLARED = "regularity-undeclared"


class DerivativeSurface(StrEnum):
    """Quantity with respect to which a derivative is requested.

    Capability surfaces (every surface not listed as owned) describe derivatives
    that must propagate through every participant of a combined map; a participant
    that does not declare a capability surface does not support it. Owned surfaces
    (`MODEL_PARAMETER`, `MODEL_STATE`, `EVENT`, `STOCHASTIC_REALIZATION`) belong to
    the components that hold them; a participant that does not declare an owned
    surface does not own it. `STORED_VALUES` is a capability surface, matching the
    all-participant semantics of stored-value derivatives in artifact contracts.
    """

    INPUT = "input"
    PRIMAL_STATE = "primal-state"
    MODEL_PARAMETER = "model-parameter"
    PHYSICAL_PARAMETER = "physical-parameter"
    SOLVER_ARGUMENT = "solver-argument"
    MODEL_STATE = "model-state"
    STORED_VALUES = "stored-values"
    FIT_FEATURES = "fit-features"
    FIT_TARGETS = "fit-targets"
    FIT_WEIGHTS = "fit-weights"
    FIT_HYPERPARAMETERS = "fit-hyperparameters"
    EVENT = "event"
    STOCHASTIC_REALIZATION = "stochastic-realization"


class GradientLevel(StrEnum):
    """Mathematical support level of one derivative claim.

    Levels are not compared blindly: a claim carrying conditions that are not
    known to hold resolves to at most `CONDITIONAL` before comparison. Use
    `resolve_gradient_level` and `gradient_level_at_least`.
    """

    SMOOTH = "smooth"
    ALMOST_EVERYWHERE = "almost-everywhere"
    CONDITIONAL = "conditional"
    NONE = "none"


class DerivativeRoute(StrEnum):
    """Mechanism that forms the derivatives a contract declares.

    `STOPPED` records that no derivative mechanism is claimed for the combined
    map, so every differentiation request against it is unsupported with the
    reason `"route-stopped"`, whatever levels its surfaces declare.
    `authority_admits` never admits `STOPPED`.
    """

    DIRECT = "direct"
    IMPLICIT = "implicit"
    UNROLLED = "unrolled"
    SPECTRAL = "spectral"
    RELAXED = "relaxed"
    EXTERNAL_ADJOINT = "external-adjoint"
    STOPPED = "stopped"


class BranchDifferentiationPolicy(StrEnum):
    """Canonical differentiation semantics of discrete branch decisions."""

    SMOOTH = "smooth"
    BRANCHWISE = "branchwise"
    FROZEN_DECISION = "frozen-decision"
    SMOOTH_SURROGATE = "smooth-surrogate"
    EVENT_AWARE = "event-aware"
    UNSUPPORTED = "unsupported"


class ObjectiveKind(StrEnum):
    """Scientific meaning of a training objective."""

    PHYSICAL_RESIDUAL = "physical-residual"
    DATA_FIT = "data-fit"
    SOLUTION_MAP = "solution-map"
    ROLLOUT = "rollout"
    ALGORITHMIC_WORK = "algorithmic-work"
    SUPERVISED_PROXY = "supervised-proxy"


class ComponentAuthority(StrEnum):
    """What a component is trusted to decide inside its owner."""

    ACCELERATOR = "accelerator"
    DISCRETIZATION = "discretization"
    MODEL = "model"
    SURROGATE = "surrogate"
    DECISION = "decision"


class CapabilityEvidenceKind(StrEnum):
    """Provenance kind of evidence supporting one declared capability."""

    DECLARED = "declared"
    CONSTRUCTED = "constructed"
    RUNTIME_CHECKED = "runtime-checked"


_VALUE_SURFACES = frozenset({DerivativeSurface.INPUT, DerivativeSurface.PRIMAL_STATE})
_SURFACE_ORDER = {surface: index for index, surface in enumerate(DerivativeSurface)}
_EVIDENCE_ORDER = {kind: index for index, kind in enumerate(CapabilityEvidenceKind)}


def _identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string without surrounding space.")
    return value


def _identifier_set(values: Iterable[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a collection of strings, not one string.")
    return tuple(sorted({_identifier(value, name) for value in values}))


def _derivative_order(order: Any, /) -> int:
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("Derivative order must be an int.")
    if order < 1:
        raise ValueError("Derivative order must be at least one.")
    return order


def _require_level(level: Any, /) -> GradientLevel:
    if not isinstance(level, GradientLevel):
        raise TypeError("Gradient levels must be GradientLevel members.")
    return level


def _require_surface(surface: Any, /) -> DerivativeSurface:
    if not isinstance(surface, DerivativeSurface):
        raise TypeError("Derivative surfaces must be DerivativeSurface members.")
    return surface


def _require_route(route: Any, /) -> DerivativeRoute:
    if not isinstance(route, DerivativeRoute):
        raise TypeError("Derivative routes must be DerivativeRoute members.")
    return route


def _surface_set(
    surfaces: Iterable[DerivativeSurface], /
) -> tuple[DerivativeSurface, ...]:
    values = tuple(_require_surface(surface) for surface in surfaces)
    if not values:
        raise ValueError("At least one derivative surface is required.")
    if len(set(values)) != len(values):
        raise ValueError("Derivative surfaces must be unique.")
    return tuple(sorted(values, key=_SURFACE_ORDER.__getitem__))


def _level_rank(level: GradientLevel, /) -> int:
    match _require_level(level):
        case GradientLevel.NONE:
            return 0
        case GradientLevel.CONDITIONAL:
            return 1
        case GradientLevel.ALMOST_EVERYWHERE:
            return 2
        case GradientLevel.SMOOTH:
            return 3
        case _:
            raise ValueError(f"Unknown gradient level {level!r}.")


def is_owned_surface(surface: DerivativeSurface, /) -> bool:
    """Return whether `surface` is owned by components rather than a capability."""
    match _require_surface(surface):
        case (
            DerivativeSurface.MODEL_PARAMETER
            | DerivativeSurface.MODEL_STATE
            | DerivativeSurface.EVENT
            | DerivativeSurface.STOCHASTIC_REALIZATION
        ):
            return True
        case (
            DerivativeSurface.INPUT
            | DerivativeSurface.PRIMAL_STATE
            | DerivativeSurface.PHYSICAL_PARAMETER
            | DerivativeSurface.SOLVER_ARGUMENT
            | DerivativeSurface.STORED_VALUES
            | DerivativeSurface.FIT_FEATURES
            | DerivativeSurface.FIT_TARGETS
            | DerivativeSurface.FIT_WEIGHTS
            | DerivativeSurface.FIT_HYPERPARAMETERS
        ):
            return False
        case _:
            raise ValueError(f"Unknown derivative surface {surface!r}.")


def weakest_level(levels: Iterable[GradientLevel], /) -> GradientLevel:
    """Return the weakest of one or more gradient levels."""
    values = tuple(_require_level(level) for level in levels)
    if not values:
        raise ValueError("weakest_level requires at least one level.")
    return min(values, key=_level_rank)


def resolve_gradient_level(
    level: GradientLevel,
    /,
    *,
    conditions: Iterable[str] = (),
    satisfied: Iterable[str] = (),
) -> GradientLevel:
    """Return the level a consumer may rely on once conditions are resolved.

    A claim holds at its declared level only when every attached condition is in
    `satisfied`; otherwise it is at most `CONDITIONAL`.
    """
    level_ = _require_level(level)
    required = _identifier_set(conditions, "conditions")
    accepted = frozenset(_identifier_set(satisfied, "satisfied"))
    if accepted.issuperset(required):
        return level_
    return weakest_level((level_, GradientLevel.CONDITIONAL))


def gradient_level_at_least(
    level: GradientLevel,
    required: GradientLevel,
    /,
    *,
    conditions: Iterable[str] = (),
    satisfied: Iterable[str] = (),
) -> bool:
    """Return whether a conditioned claim meets `required` after resolution."""
    resolved = resolve_gradient_level(level, conditions=conditions, satisfied=satisfied)
    return _level_rank(resolved) >= _level_rank(_require_level(required))


def _continuity(value: Any, /) -> int | Literal["smooth"]:
    if isinstance(value, str):
        if value != "smooth":
            raise ValueError("continuity must be an int >= -1 or 'smooth'.")
        return "smooth"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("continuity must be an int or 'smooth'.")
    if value < -1:
        raise ValueError("continuity must be at least -1.")
    return value


def _min_continuity(
    left: int | Literal["smooth"], right: int | Literal["smooth"], /
) -> int | Literal["smooth"]:
    if left == "smooth":
        return right
    if right == "smooth":
        return left
    return min(left, right)


def _piece_rank(pieces: RegularityPieces, /) -> int:
    if not isinstance(pieces, str):
        raise TypeError("pieces must be a string.")
    match pieces:
        case "polynomial":
            return 0
        case "smooth":
            return 1
        case "none":
            return 2
        case _:
            raise ValueError("pieces must be 'none', 'polynomial', or 'smooth'.")


def _combined_support(left: str | None, right: str | None, /) -> str | None:
    if left is None:
        return right
    if right is None or left == right:
        return left
    raise ValueError(f"Regularity supports {left!r} and {right!r} are incompatible.")


class DerivativeRegularity(StrictModule, NonTrainableState):
    """Declared smoothness of a component map in its value arguments.

    The value arguments are the `INPUT` and `PRIMAL_STATE` surfaces. `continuity`
    is the classical continuity order (`-1` for discontinuous maps, `k >= 0` for
    `C^k`, or `"smooth"`). `pieces` names the structure between non-smooth loci:
    `"polynomial"` pieces carry `degree_bound`, an upper bound on the piece degree;
    `"smooth"` pieces are smooth but not polynomial; `"none"` declares no piece
    decomposition. `support` identifies the region on which the claim holds
    (`None` for the whole domain). Declarations are canonical: a `C^k` piecewise
    polynomial of degree at most `k` is one global polynomial and is stored as
    smooth, and a smooth map always has smooth or polynomial pieces.
    """

    continuity: int | Literal["smooth"] = eqx.field(static=True)
    pieces: RegularityPieces = eqx.field(static=True)
    degree_bound: int | None = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)
    support: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        continuity: int | Literal["smooth"],
        pieces: RegularityPieces,
        degree_bound: int | None = None,
        conditions: Iterable[str] = (),
        support: str | None = None,
    ):
        continuity_ = _continuity(continuity)
        _piece_rank(pieces)
        if pieces == "polynomial":
            if isinstance(degree_bound, bool) or not isinstance(degree_bound, int):
                raise TypeError("Polynomial pieces require an int degree_bound.")
            if degree_bound < 0:
                raise ValueError("degree_bound must be non-negative.")
            if continuity_ != "smooth" and continuity_ >= degree_bound:
                # Pieces of degree <= k that agree to order k at an interface
                # agree identically, so the map is one global polynomial.
                continuity_ = "smooth"
        elif degree_bound is not None:
            raise ValueError("degree_bound is declared only for polynomial pieces.")
        conditions_ = _identifier_set(conditions, "conditions")
        support_ = None if support is None else _identifier(support, "support")
        self.continuity = continuity_
        self.pieces = "smooth" if continuity_ == "smooth" and pieces == "none" else pieces
        self.degree_bound = degree_bound
        self.conditions = conditions_
        self.support = support_

    @classmethod
    def smooth(
        cls,
        *,
        degree_bound: int | None = None,
        conditions: Iterable[str] = (),
        support: str | None = None,
    ) -> DerivativeRegularity:
        """Smooth map; `degree_bound` declares a global polynomial."""
        return cls(
            continuity="smooth",
            pieces="smooth" if degree_bound is None else "polynomial",
            degree_bound=degree_bound,
            conditions=conditions,
            support=support,
        )

    @classmethod
    def piecewise_polynomial(
        cls,
        *,
        continuity: int,
        degree_bound: int,
        conditions: Iterable[str] = (),
        support: str | None = None,
    ) -> DerivativeRegularity:
        """`C^continuity` map with polynomial pieces of degree at most `degree_bound`."""
        return cls(
            continuity=continuity,
            pieces="polynomial",
            degree_bound=degree_bound,
            conditions=conditions,
            support=support,
        )

    @classmethod
    def piecewise_smooth(
        cls,
        *,
        continuity: int,
        conditions: Iterable[str] = (),
        support: str | None = None,
    ) -> DerivativeRegularity:
        """`C^continuity` map with smooth non-polynomial pieces."""
        return cls(
            continuity=continuity,
            pieces="smooth",
            conditions=conditions,
            support=support,
        )

    @classmethod
    def discontinuous(
        cls,
        *,
        conditions: Iterable[str] = (),
        support: str | None = None,
    ) -> DerivativeRegularity:
        """Discontinuous map without a declared piece decomposition."""
        return cls(continuity=-1, pieces="none", conditions=conditions, support=support)

    def admits_order(self, order: int, /) -> tuple[GradientLevel, tuple[str, ...]]:
        """Return the level and conditions of value derivatives of `order`.

        Orders within the continuity class are smooth. Proven degeneracy is `NONE`:
        polynomial pieces whose degree bound is below `order`, or a discontinuous
        map without pieces. Every other order is almost-everywhere; from order
        `continuity + 2` the distributional derivative has a singular part on the
        non-smooth locus, recorded as the `"singular-part-ignored"` condition.
        """
        order_ = _derivative_order(order)
        continuity = self.continuity
        if continuity == "smooth" or order_ <= continuity:
            return GradientLevel.SMOOTH, self.conditions
        match self.pieces:
            case "polynomial":
                if self.degree_bound < order_:
                    return GradientLevel.NONE, ()
            case "none":
                if continuity == -1:
                    return GradientLevel.NONE, ()
            case "smooth":
                pass
            case _:
                raise ValueError(f"Unknown regularity pieces {self.pieces!r}.")
        conditions = self.conditions
        if order_ >= continuity + 2:
            conditions = _identifier_set(
                (*conditions, _SINGULAR_PART_IGNORED), "conditions"
            )
        return GradientLevel.ALMOST_EVERYWHERE, conditions

    def differentiate(self, order: int, /) -> DerivativeRegularity:
        """Regularity of the order-`order` value derivative of this map.

        The derivative is the classical derivative inside pieces; singular parts on
        the non-smooth locus are not represented, so the result carries the
        conditions `admits_order` attaches to `order` (including
        `"singular-part-ignored"`). Continuity drops by `order` (to at most `-1`),
        polynomial degree bounds drop by `order`, and a proven-degenerate order
        raises `ValueError`.
        """
        order_ = _derivative_order(order)
        level, conditions = self.admits_order(order_)
        if level is GradientLevel.NONE:
            raise ValueError(
                f"An order-{order_} derivative of this regularity is degenerate."
            )
        continuity = (
            "smooth" if self.continuity == "smooth" else max(self.continuity - order_, -1)
        )
        degree = (
            None if self.pieces != "polynomial" else max(self.degree_bound - order_, 0)
        )
        return DerivativeRegularity(
            continuity=continuity,
            pieces=self.pieces,
            degree_bound=degree,
            conditions=conditions,
            support=self.support,
        )

    def add(self, other: DerivativeRegularity, /) -> DerivativeRegularity:
        """Regularity of a sum or juxtaposition: the maximum degree bound."""
        return self._combine(other, max)

    def multiply(self, other: DerivativeRegularity, /) -> DerivativeRegularity:
        """Regularity of a product: degree bounds add."""
        return self._combine(other, lambda left, right: left + right)

    def compose(self, outer: DerivativeRegularity, /) -> DerivativeRegularity:
        """Regularity of `outer` applied after `self`: degree bounds multiply."""
        return self._combine(outer, lambda inner, outer_: inner * outer_)

    def _combine(
        self,
        other: DerivativeRegularity,
        degree_rule: Callable[[int, int], int],
        /,
    ) -> DerivativeRegularity:
        other_ = _require_regularity(other)
        # Pieces stay polynomial only when both stages are polynomial; a smooth
        # non-polynomial stage yields smooth pieces, and undeclared piece
        # structure is the most general.
        pieces = max((self.pieces, other_.pieces), key=_piece_rank)
        degree = (
            degree_rule(self.degree_bound, other_.degree_bound)
            if pieces == "polynomial"
            else None
        )
        return DerivativeRegularity(
            continuity=_min_continuity(self.continuity, other_.continuity),
            pieces=pieces,
            degree_bound=degree,
            conditions=(*self.conditions, *other_.conditions),
            support=_combined_support(self.support, other_.support),
        )


def _require_regularity(value: Any, /) -> DerivativeRegularity:
    if not isinstance(value, DerivativeRegularity):
        raise TypeError("Expected a DerivativeRegularity.")
    return value


def _regularity_payload(regularity: DerivativeRegularity | None, /) -> Any:
    if regularity is None:
        return None
    return {
        "continuity": regularity.continuity,
        "pieces": regularity.pieces,
        "degree_bound": regularity.degree_bound,
        "conditions": list(regularity.conditions),
        "support": regularity.support,
    }


class SurfaceDerivative(StrictModule, NonTrainableState):
    """Declared gradient level and conditions for one derivative surface."""

    surface: DerivativeSurface = eqx.field(static=True)
    level: GradientLevel = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        surface: DerivativeSurface,
        level: GradientLevel,
        /,
        *,
        conditions: Iterable[str] = (),
    ):
        surface_ = _require_surface(surface)
        level_ = _require_level(level)
        conditions_ = _identifier_set(conditions, "conditions")
        self.surface = surface_
        self.level = level_
        self.conditions = conditions_


class DifferentiationRequest(StrictModule, NonTrainableState):
    """Explicit request for derivatives of one order on a set of surfaces.

    Surfaces are a set stored in canonical `DerivativeSurface` order. `authority`
    is the authority of the requesting owner; `None` denotes direct eager
    differentiation outside scientific preparation.
    """

    surfaces: tuple[DerivativeSurface, ...] = eqx.field(static=True)
    order: int = eqx.field(static=True)
    authority: ComponentAuthority | None = eqx.field(static=True)

    def __init__(
        self,
        surfaces: Iterable[DerivativeSurface],
        /,
        *,
        order: int = 1,
        authority: ComponentAuthority | None = None,
    ):
        surfaces_ = _surface_set(surfaces)
        order_ = _derivative_order(order)
        if authority is not None and not isinstance(authority, ComponentAuthority):
            raise TypeError("authority must be a ComponentAuthority or None.")
        self.surfaces = surfaces_
        self.order = order_
        self.authority = authority


class DerivativeAdmission(StrictModule, NonTrainableState):
    """Audited answer to one differentiation request.

    `levels` align with `request.surfaces`. The admission is supported exactly
    when no requested level is `NONE`; a `STOPPED` route admits no level. An
    unsupported admission names its rejection `reasons`, and a supported
    admission carries none. `conditions` qualify every admitted level (resolve
    them with `gradient_level_at_least`).
    """

    request: DifferentiationRequest
    levels: tuple[GradientLevel, ...] = eqx.field(static=True)
    route: DerivativeRoute = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_outputs: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        request: DifferentiationRequest,
        levels: Iterable[GradientLevel],
        /,
        *,
        route: DerivativeRoute,
        conditions: Iterable[str] = (),
        reasons: Iterable[str] = (),
        nondifferentiable_outputs: Iterable[str] = (),
    ):
        if not isinstance(request, DifferentiationRequest):
            raise TypeError("request must be a DifferentiationRequest.")
        levels_ = tuple(_require_level(level) for level in levels)
        if len(levels_) != len(request.surfaces):
            raise ValueError("Admission levels must align with the requested surfaces.")
        route_ = _require_route(route)
        if route_ is DerivativeRoute.STOPPED and any(
            level is not GradientLevel.NONE for level in levels_
        ):
            raise ValueError("A stopped route admits no derivative level.")
        reasons_ = _identifier_set(reasons, "reasons")
        supported = all(level is not GradientLevel.NONE for level in levels_)
        if supported and reasons_:
            raise ValueError("A supported admission carries no rejection reasons.")
        if not supported and not reasons_:
            raise ValueError("An unsupported admission must name its reasons.")
        conditions_ = _identifier_set(conditions, "conditions")
        outputs = _identifier_set(nondifferentiable_outputs, "nondifferentiable_outputs")
        self.request = request
        self.levels = levels_
        self.route = route_
        self.supported = supported
        self.status = DERIVATIVE_SUPPORTED if supported else DERIVATIVE_UNSUPPORTED
        self.conditions = conditions_
        self.reasons = reasons_
        self.nondifferentiable_outputs = outputs

    def level(self, surface: DerivativeSurface, /) -> GradientLevel:
        """Return the admitted level of one requested surface."""
        surface_ = _require_surface(surface)
        if surface_ not in self.request.surfaces:
            raise ValueError(f"Surface {surface_.value!r} was not requested.")
        return self.levels[self.request.surfaces.index(surface_)]


class RegularityPolicy(StrictModule, NonTrainableState):
    """Owner permissions for regularity claims that are not classical.

    `allow_almost_everywhere` admits almost-everywhere value derivatives.
    `allow_undeclared` admits undeclared regularity for exploratory surrogate,
    accelerator, and decision components on non-implicit routes. Both default to
    refusal.
    """

    allow_almost_everywhere: bool = eqx.field(static=True)
    allow_undeclared: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        allow_almost_everywhere: bool = False,
        allow_undeclared: bool = False,
    ):
        if not isinstance(allow_almost_everywhere, bool) or not isinstance(
            allow_undeclared, bool
        ):
            raise TypeError("RegularityPolicy permissions must be bool.")
        self.allow_almost_everywhere = allow_almost_everywhere
        self.allow_undeclared = allow_undeclared


def _undeclared_regularity_admitted(
    authority: ComponentAuthority | None,
    policy: RegularityPolicy,
    implicit: bool,
    /,
) -> bool:
    match authority:
        case None:
            return True
        case ComponentAuthority.MODEL | ComponentAuthority.DISCRETIZATION:
            return False
        case (
            ComponentAuthority.SURROGATE
            | ComponentAuthority.ACCELERATOR
            | ComponentAuthority.DECISION
        ):
            return policy.allow_undeclared and not implicit
        case _:
            raise ValueError(f"Unknown component authority {authority!r}.")


def _declared_regularity_decision(
    regularity: DerivativeRegularity,
    order: int,
    policy: RegularityPolicy,
    implicit: bool,
    /,
) -> tuple[GradientLevel, tuple[str, ...], tuple[str, ...]]:
    level, conditions = regularity.admits_order(order)
    reasons: list[str] = []
    if level is GradientLevel.NONE:
        reasons.append("regularity-degenerate")
    if level is GradientLevel.ALMOST_EVERYWHERE and not policy.allow_almost_everywhere:
        reasons.append("almost-everywhere-not-allowed")
    classical_c1 = regularity.continuity == "smooth" or regularity.continuity >= 1
    if implicit and not classical_c1 and _BRANCH_MARGIN not in regularity.conditions:
        reasons.append("implicit-requires-c1")
    if reasons:
        return GradientLevel.NONE, (), tuple(reasons)
    return level, conditions, ()


def admit_regularity(
    regularity: DerivativeRegularity | None,
    request: DifferentiationRequest,
    /,
    *,
    route: DerivativeRoute,
    policy: RegularityPolicy,
) -> DerivativeAdmission:
    """Admit the regularity a derivative request requires for its owner.

    A `STOPPED` route claims no derivative mechanism: every requested level is
    `NONE` with the reason `"route-stopped"`. Otherwise, regularity is required
    when the request touches a value surface (`INPUT`,
    `PRIMAL_STATE`) or `route` is `IMPLICIT`; otherwise every level is `SMOOTH`,
    the identity of `weakest_level`. When required:

    - proven degeneracy (`admits_order` is `NONE`) is always rejected;
    - almost-everywhere levels need `policy.allow_almost_everywhere`;
    - implicit routes need classical `C^1` or a `"branch-margin"` condition;
    - undeclared regularity is admitted for direct eager requests
      (`authority=None`); rejected for `MODEL` and `DISCRETIZATION` authorities;
      and admitted for `SURROGATE`, `ACCELERATOR`, and `DECISION` only with
      `policy.allow_undeclared` on a non-implicit route. Admitted undeclared
      regularity carries the `"regularity-undeclared"` condition, and a rejection
      names it as the reason.

    The resulting level bounds the value surfaces, or every surface on an
    implicit route; other surfaces report `SMOOTH`.
    """
    if regularity is not None:
        _require_regularity(regularity)
    if not isinstance(request, DifferentiationRequest):
        raise TypeError("request must be a DifferentiationRequest.")
    route_ = _require_route(route)
    if not isinstance(policy, RegularityPolicy):
        raise TypeError("policy must be a RegularityPolicy.")
    implicit = route_ is DerivativeRoute.IMPLICIT
    if route_ is DerivativeRoute.STOPPED:
        return DerivativeAdmission(
            request,
            (GradientLevel.NONE,) * len(request.surfaces),
            route=route_,
            reasons=(_ROUTE_STOPPED,),
        )
    touches_values = any(surface in _VALUE_SURFACES for surface in request.surfaces)
    if not (touches_values or implicit):
        return DerivativeAdmission(
            request, (GradientLevel.SMOOTH,) * len(request.surfaces), route=route_
        )
    if regularity is None:
        admitted = _undeclared_regularity_admitted(request.authority, policy, implicit)
        level = GradientLevel.SMOOTH if admitted else GradientLevel.NONE
        conditions = (_REGULARITY_UNDECLARED,) if admitted else ()
        reasons = () if admitted else (_REGULARITY_UNDECLARED,)
    else:
        level, conditions, reasons = _declared_regularity_decision(
            regularity, request.order, policy, implicit
        )
    levels = tuple(
        level if implicit or surface in _VALUE_SURFACES else GradientLevel.SMOOTH
        for surface in request.surfaces
    )
    return DerivativeAdmission(
        request, levels, route=route_, conditions=conditions, reasons=reasons
    )


def _require_policy(policy: RegularityPolicy | None, /) -> RegularityPolicy:
    if policy is None:
        return RegularityPolicy()
    if not isinstance(policy, RegularityPolicy):
        raise TypeError("policy must be a RegularityPolicy or None.")
    return policy


def _combined_route(
    contracts: Sequence[DerivativeContract],
    composition_route: DerivativeRoute | None,
    /,
) -> tuple[DerivativeRoute, tuple[str, ...]]:
    if composition_route is not None:
        return _require_route(composition_route), ()
    routes = sorted({contract.route for contract in contracts})
    if len(routes) == 1:
        return routes[0], ()
    return DerivativeRoute.STOPPED, (
        "mixed-derivative-routes:" + ",".join(route.value for route in routes),
    )


def _surface_entry(
    table: dict[DerivativeSurface, SurfaceDerivative],
    surface: DerivativeSurface,
    /,
) -> SurfaceDerivative:
    return table.get(surface, SurfaceDerivative(surface, GradientLevel.NONE))


def _combined_surface(
    surface: DerivativeSurface, entries: Sequence[SurfaceDerivative], /
) -> SurfaceDerivative:
    return SurfaceDerivative(
        surface,
        weakest_level(entry.level for entry in entries),
        conditions=(condition for entry in entries for condition in entry.conditions),
    )


def _met_surfaces(
    contracts: Sequence[DerivativeContract], /
) -> tuple[SurfaceDerivative, ...]:
    tables = tuple(
        {entry.surface: entry for entry in contract.surfaces} for contract in contracts
    )
    result = []
    for surface in DerivativeSurface:
        if not any(surface in table for table in tables):
            continue
        if is_owned_surface(surface):
            entries = tuple(table[surface] for table in tables if surface in table)
        else:
            entries = tuple(_surface_entry(table, surface) for table in tables)
        result.append(_combined_surface(surface, entries))
    return tuple(result)


def _composed_surfaces(
    upstream: DerivativeContract, downstream: DerivativeContract, /
) -> tuple[SurfaceDerivative, ...]:
    first = {entry.surface: entry for entry in upstream.surfaces}
    second = {entry.surface: entry for entry in downstream.surfaces}
    passage = _surface_entry(second, DerivativeSurface.INPUT)
    result = []
    for surface in DerivativeSurface:
        if surface not in first and surface not in second:
            continue
        if surface is DerivativeSurface.INPUT:
            entries: tuple[SurfaceDerivative, ...] = (
                _surface_entry(first, surface),
                passage,
            )
        elif is_owned_surface(surface):
            entries = ()
            if surface in first:
                entries += (first[surface], passage)
            if surface in second:
                entries += (second[surface],)
        else:
            entries = (
                _surface_entry(first, surface),
                passage,
                _surface_entry(second, surface),
            )
        result.append(_combined_surface(surface, entries))
    return tuple(result)


class DerivativeContract(StrictModule, NonTrainableState):
    """Canonical static declaration of the derivatives a component supports.

    Surfaces are stored in canonical `DerivativeSurface` order. A capability
    surface that is not declared has level `NONE`, so explicit `NONE` capability
    entries (and their conditions) are canonicalized away; owned surfaces keep
    explicit `NONE` entries because they record ownership. `regularity` describes
    the value surfaces (`INPUT`, `PRIMAL_STATE`); `None` means undeclared.
    `contract_id` content-addresses the canonical declaration.
    """

    surfaces: tuple[SurfaceDerivative, ...]
    route: DerivativeRoute = eqx.field(static=True)
    regularity: DerivativeRegularity | None
    conditions: tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_outputs: tuple[str, ...] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        surfaces: Iterable[SurfaceDerivative] = (),
        /,
        *,
        route: DerivativeRoute,
        regularity: DerivativeRegularity | None = None,
        conditions: Iterable[str] = (),
        nondifferentiable_outputs: Iterable[str] = (),
    ):
        entries = tuple(surfaces)
        if any(not isinstance(entry, SurfaceDerivative) for entry in entries):
            raise TypeError("surfaces must contain SurfaceDerivative values.")
        declared = tuple(entry.surface for entry in entries)
        if len(set(declared)) != len(declared):
            raise ValueError("A derivative contract declares each surface once.")
        canonical = tuple(
            sorted(
                (
                    entry
                    for entry in entries
                    if entry.level is not GradientLevel.NONE
                    or is_owned_surface(entry.surface)
                ),
                key=lambda entry: _SURFACE_ORDER[entry.surface],
            )
        )
        route_ = _require_route(route)
        if regularity is not None:
            _require_regularity(regularity)
        conditions_ = _identifier_set(conditions, "conditions")
        outputs = _identifier_set(nondifferentiable_outputs, "nondifferentiable_outputs")
        self.surfaces = canonical
        self.route = route_
        self.regularity = regularity
        self.conditions = conditions_
        self.nondifferentiable_outputs = outputs
        self.contract_id = canonical_fingerprint(_contract_content(self))

    @classmethod
    def smooth(
        cls,
        surfaces: Iterable[DerivativeSurface],
        /,
        *,
        route: DerivativeRoute = DerivativeRoute.DIRECT,
        conditions: Iterable[str] = (),
    ) -> DerivativeContract:
        """Contract of a smooth map: every surface in `surfaces` is `SMOOTH`.

        The regularity is `DerivativeRegularity.smooth()`, so derivatives of every
        order are claimed.
        """
        return cls(
            (SurfaceDerivative(surface, GradientLevel.SMOOTH) for surface in surfaces),
            route=route,
            regularity=DerivativeRegularity.smooth(),
            conditions=conditions,
        )

    @property
    def supported_surfaces(self) -> tuple[DerivativeSurface, ...]:
        """Declared surfaces whose level is not `NONE`, in canonical order."""
        return tuple(
            entry.surface
            for entry in self.surfaces
            if entry.level is not GradientLevel.NONE
        )

    def level(self, surface: DerivativeSurface, /) -> GradientLevel:
        """Return the declared level of `surface` (`NONE` when undeclared)."""
        surface_ = _require_surface(surface)
        for entry in self.surfaces:
            if entry.surface is surface_:
                return entry.level
        return GradientLevel.NONE

    def admit(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        """Admit `request` against the declared levels and regularity.

        Each requested level is the weakest of its declared level and the
        `admit_regularity` bound under `policy` (default `RegularityPolicy()`,
        which refuses almost-everywhere and undeclared claims where the authority
        requires permission). An undeclared surface is rejected with the reason
        `"surface-unsupported:<surface>"`, and a `STOPPED` contract rejects every
        request with the reason `"route-stopped"`. Conditions collect the
        contract, requested-surface, and regularity conditions.
        """
        if not isinstance(request, DifferentiationRequest):
            raise TypeError("request must be a DifferentiationRequest.")
        bound = admit_regularity(
            self.regularity, request, route=self.route, policy=_require_policy(policy)
        )
        table = {entry.surface: entry for entry in self.surfaces}
        conditions = [*self.conditions, *bound.conditions]
        reasons = list(bound.reasons)
        levels = []
        for surface, limit in zip(request.surfaces, bound.levels, strict=True):
            entry = _surface_entry(table, surface)
            conditions.extend(entry.conditions)
            if entry.level is GradientLevel.NONE:
                reasons.append(f"surface-unsupported:{surface.value}")
            levels.append(weakest_level((entry.level, limit)))
        return DerivativeAdmission(
            request,
            levels,
            route=self.route,
            conditions=conditions,
            reasons=reasons,
            nondifferentiable_outputs=self.nondifferentiable_outputs,
        )

    def require(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        """Return the admission of `request`, raising `ValueError` if unsupported.

        The message starts with `DERIVATIVE_UNSUPPORTED` and names the
        unsupported surfaces and rejection reasons.
        """
        admission = self.admit(request, policy=policy)
        if not admission.supported:
            unsupported = tuple(
                surface.value
                for surface, level in zip(
                    admission.request.surfaces, admission.levels, strict=True
                )
                if level is GradientLevel.NONE
            )
            raise ValueError(
                f"{DERIVATIVE_UNSUPPORTED}: order-{request.order} derivatives with "
                f"respect to {unsupported!r} are unsupported "
                f"(reasons: {', '.join(admission.reasons)}); inspect "
                "DerivativeContract.admit before transforming."
            )
        return admission

    def meet(
        self,
        *others: DerivativeContract,
        composition_route: DerivativeRoute | None = None,
    ) -> DerivativeContract:
        """Combine the contracts of parallel parts of one map.

        Capability surfaces take the weakest level over every participant (an
        undeclared capability counts as `NONE`); owned surfaces take the weakest
        level over the participants that own them and stay absent when none does.
        Differing routes combine to `STOPPED` with a `"mixed-derivative-routes:"`
        condition, so the combination admits no derivative request, unless
        `composition_route` is supplied, which then becomes the combined route.
        Regularity combines by `DerivativeRegularity.add` and is
        undeclared if any participant's is. Conditions and nondifferentiable
        outputs are unions.
        """
        contracts = (self, *others)
        if any(not isinstance(contract, DerivativeContract) for contract in contracts):
            raise TypeError("meet requires DerivativeContract values.")
        regularity = None
        if all(contract.regularity is not None for contract in contracts):
            regularity = self.regularity
            for contract in others:
                regularity = regularity.add(contract.regularity)
        return _combined_contract(
            contracts, _met_surfaces(contracts), regularity, composition_route
        )

    def compose(
        self,
        downstream: DerivativeContract,
        /,
        *,
        composition_route: DerivativeRoute | None = None,
    ) -> DerivativeContract:
        """Contract of the sequential pipeline `downstream(self(...))`.

        The downstream `INPUT` level is the passage through which every upstream
        derivative reaches the pipeline output:

        - `INPUT` is the weakest of both stages' `INPUT` levels;
        - a capability surface is the weakest of the upstream level, the
          downstream `INPUT` level, and the downstream level (undeclared is `NONE`);
        - an owned surface is the weakest of the upstream level together with the
          downstream `INPUT` level (when upstream owns it) and the downstream level
          (when downstream owns it), and stays absent when neither owns it.

        Regularity composes by `self.regularity.compose(downstream.regularity)` and
        is undeclared if either stage's is. Routes, conditions, and
        nondifferentiable outputs combine as in `meet`.
        """
        if not isinstance(downstream, DerivativeContract):
            raise TypeError("compose requires a DerivativeContract.")
        regularity = (
            None
            if self.regularity is None or downstream.regularity is None
            else self.regularity.compose(downstream.regularity)
        )
        return _combined_contract(
            (self, downstream),
            _composed_surfaces(self, downstream),
            regularity,
            composition_route,
        )


def _combined_contract(
    contracts: Sequence[DerivativeContract],
    surfaces: tuple[SurfaceDerivative, ...],
    regularity: DerivativeRegularity | None,
    composition_route: DerivativeRoute | None,
    /,
) -> DerivativeContract:
    route, route_conditions = _combined_route(contracts, composition_route)
    return DerivativeContract(
        surfaces,
        route=route,
        regularity=regularity,
        conditions=(
            *route_conditions,
            *(condition for contract in contracts for condition in contract.conditions),
        ),
        nondifferentiable_outputs=(
            output
            for contract in contracts
            for output in contract.nondifferentiable_outputs
        ),
    )


def _contract_content(contract: DerivativeContract, /) -> dict[str, Any]:
    return {
        "kind": "derivative-contract",
        "surfaces": [
            [entry.surface.value, entry.level.value, list(entry.conditions)]
            for entry in contract.surfaces
        ],
        "route": contract.route.value,
        "regularity": _regularity_payload(contract.regularity),
        "conditions": list(contract.conditions),
        "nondifferentiable_outputs": list(contract.nondifferentiable_outputs),
    }


_CONTRACT_PAYLOAD_KEYS = frozenset(
    {
        "kind",
        "surfaces",
        "route",
        "regularity",
        "conditions",
        "nondifferentiable_outputs",
        "contract_id",
    }
)
_REGULARITY_PAYLOAD_KEYS = frozenset(
    {"continuity", "pieces", "degree_bound", "conditions", "support"}
)


def derivative_contract_payload(contract: DerivativeContract, /) -> dict[str, Any]:
    """Return the JSON-compatible payload of `contract`.

    The payload is the fingerprinted content of the contract plus its
    `contract_id`; `derivative_contract_from_payload` reconstructs the identical
    contract.
    """
    if not isinstance(contract, DerivativeContract):
        raise TypeError("contract must be a DerivativeContract.")
    return {**_contract_content(contract), "contract_id": contract.contract_id}


def _payload_mapping(value: Any, keys: frozenset[str], name: str, /) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} payload must be a mapping.")
    if set(value) != keys:
        raise ValueError(f"{name} payload must have exactly the keys {sorted(keys)}.")
    return value


def _payload_list(value: Any, name: str, /) -> list | tuple:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a list.")
    return value


def _payload_enum(enum: type[StrEnum], value: Any, name: str, /) -> Any:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return enum(value)


def _surface_from_payload(value: Any, /) -> SurfaceDerivative:
    entry = _payload_list(value, "Surface derivative payload")
    if len(entry) != 3:
        raise ValueError("Surface derivative payload is [surface, level, conditions].")
    surface, level, conditions = entry
    return SurfaceDerivative(
        _payload_enum(DerivativeSurface, surface, "surface"),
        _payload_enum(GradientLevel, level, "level"),
        conditions=_payload_list(conditions, "Surface conditions"),
    )


def _regularity_from_payload(value: Any, /) -> DerivativeRegularity | None:
    if value is None:
        return None
    payload = _payload_mapping(value, _REGULARITY_PAYLOAD_KEYS, "Regularity")
    return DerivativeRegularity(
        continuity=payload["continuity"],
        pieces=payload["pieces"],
        degree_bound=payload["degree_bound"],
        conditions=_payload_list(payload["conditions"], "Regularity conditions"),
        support=payload["support"],
    )


def derivative_contract_from_payload(payload: Any, /) -> DerivativeContract:
    """Reconstruct a contract from `derivative_contract_payload` output.

    The contract is rebuilt through the canonicalizing constructor. Raises
    `TypeError` for malformed values and `ValueError` for unknown keys, unknown
    enum values, or a `contract_id` that does not match the content.
    """
    value = _payload_mapping(payload, _CONTRACT_PAYLOAD_KEYS, "Derivative contract")
    if value["kind"] != "derivative-contract":
        raise ValueError("Payload is not a derivative contract.")
    contract = DerivativeContract(
        (
            _surface_from_payload(entry)
            for entry in _payload_list(value["surfaces"], "surfaces")
        ),
        route=_payload_enum(DerivativeRoute, value["route"], "route"),
        regularity=_regularity_from_payload(value["regularity"]),
        conditions=_payload_list(value["conditions"], "conditions"),
        nondifferentiable_outputs=_payload_list(
            value["nondifferentiable_outputs"], "nondifferentiable_outputs"
        ),
    )
    if value["contract_id"] != contract.contract_id:
        raise ValueError("Derivative contract identity does not match its payload.")
    return contract


def branch_policy_contract(
    policy: BranchDifferentiationPolicy,
    /,
    *,
    surfaces: Iterable[DerivativeSurface],
) -> DerivativeContract:
    """Return the canonical derivative contract of a branch policy on `surfaces`.

    | Policy | Level | Route | Regularity | Condition |
    |---|---|---|---|---|
    | `SMOOTH` | smooth | direct | smooth | none |
    | `BRANCHWISE` | almost-everywhere | direct | piecewise smooth, `C^-1` | `executed-branch` |
    | `FROZEN_DECISION` | smooth | direct | smooth | `decisions-frozen` |
    | `SMOOTH_SURROGATE` | smooth | relaxed | smooth | `smooth-surrogate` |
    | `EVENT_AWARE` | almost-everywhere | direct | piecewise smooth, `C^-1` | `transversal-events` |
    | `UNSUPPORTED` | none | stopped | undeclared | none |

    Conditions name what is differentiated: the executed branch, the map with
    decisions held fixed, a smooth surrogate of the sharp map, or the event-aware
    flow away from grazing events.
    """
    if not isinstance(policy, BranchDifferentiationPolicy):
        raise TypeError("policy must be a BranchDifferentiationPolicy.")
    surfaces_ = _surface_set(surfaces)
    match policy:
        case BranchDifferentiationPolicy.SMOOTH:
            level, route = GradientLevel.SMOOTH, DerivativeRoute.DIRECT
            regularity, conditions = DerivativeRegularity.smooth(), ()
        case BranchDifferentiationPolicy.BRANCHWISE:
            level, route = GradientLevel.ALMOST_EVERYWHERE, DerivativeRoute.DIRECT
            regularity = DerivativeRegularity.piecewise_smooth(continuity=-1)
            conditions = ("executed-branch",)
        case BranchDifferentiationPolicy.FROZEN_DECISION:
            level, route = GradientLevel.SMOOTH, DerivativeRoute.DIRECT
            regularity, conditions = DerivativeRegularity.smooth(), ("decisions-frozen",)
        case BranchDifferentiationPolicy.SMOOTH_SURROGATE:
            level, route = GradientLevel.SMOOTH, DerivativeRoute.RELAXED
            regularity, conditions = DerivativeRegularity.smooth(), ("smooth-surrogate",)
        case BranchDifferentiationPolicy.EVENT_AWARE:
            level, route = GradientLevel.ALMOST_EVERYWHERE, DerivativeRoute.DIRECT
            regularity = DerivativeRegularity.piecewise_smooth(continuity=-1)
            conditions = ("transversal-events",)
        case BranchDifferentiationPolicy.UNSUPPORTED:
            level, route = GradientLevel.NONE, DerivativeRoute.STOPPED
            regularity, conditions = None, ()
        case _:
            raise ValueError(f"Unknown branch differentiation policy {policy!r}.")
    return DerivativeContract(
        (SurfaceDerivative(surface, level) for surface in surfaces_),
        route=route,
        regularity=regularity,
        conditions=conditions,
    )


def authority_admits(
    authority: ComponentAuthority,
    route: DerivativeRoute,
    kind: ObjectiveKind,
    /,
) -> bool:
    """Return whether a component of `authority` may train on (`route`, `kind`)."""
    if not isinstance(authority, ComponentAuthority):
        raise TypeError("authority must be a ComponentAuthority.")
    _require_route(route)
    if not isinstance(kind, ObjectiveKind):
        raise TypeError("kind must be an ObjectiveKind.")
    match authority:
        case ComponentAuthority.ACCELERATOR:
            admitted = {
                (DerivativeRoute.UNROLLED, ObjectiveKind.ALGORITHMIC_WORK),
                (DerivativeRoute.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
            }
        case ComponentAuthority.DISCRETIZATION:
            admitted = {
                (DerivativeRoute.UNROLLED, ObjectiveKind.ROLLOUT),
                (DerivativeRoute.DIRECT, ObjectiveKind.ROLLOUT),
                (DerivativeRoute.IMPLICIT, ObjectiveKind.SOLUTION_MAP),
                (DerivativeRoute.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
                (DerivativeRoute.DIRECT, ObjectiveKind.PHYSICAL_RESIDUAL),
            }
        case ComponentAuthority.MODEL:
            admitted = {
                (DerivativeRoute.DIRECT, ObjectiveKind.DATA_FIT),
                (DerivativeRoute.DIRECT, ObjectiveKind.PHYSICAL_RESIDUAL),
                (DerivativeRoute.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
                (DerivativeRoute.IMPLICIT, ObjectiveKind.SOLUTION_MAP),
                (DerivativeRoute.UNROLLED, ObjectiveKind.ROLLOUT),
                (DerivativeRoute.EXTERNAL_ADJOINT, ObjectiveKind.SOLUTION_MAP),
                (DerivativeRoute.EXTERNAL_ADJOINT, ObjectiveKind.ROLLOUT),
            }
        case ComponentAuthority.SURROGATE:
            admitted = {
                (DerivativeRoute.DIRECT, ObjectiveKind.PHYSICAL_RESIDUAL),
                (DerivativeRoute.DIRECT, ObjectiveKind.DATA_FIT),
                (DerivativeRoute.UNROLLED, ObjectiveKind.ROLLOUT),
            }
        case ComponentAuthority.DECISION:
            admitted = {
                (DerivativeRoute.DIRECT, ObjectiveKind.ROLLOUT),
                (DerivativeRoute.UNROLLED, ObjectiveKind.ROLLOUT),
                (DerivativeRoute.RELAXED, ObjectiveKind.ROLLOUT),
                (DerivativeRoute.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
            }
        case _:
            raise ValueError(f"Unknown component authority {authority!r}.")
    return (route, kind) in admitted


def _evidence_kinds(
    kinds: Iterable[CapabilityEvidenceKind], /
) -> frozenset[CapabilityEvidenceKind]:
    if isinstance(kinds, str):
        raise TypeError("Evidence kinds must be a collection, not one string.")
    values = frozenset(kinds)
    if any(not isinstance(kind, CapabilityEvidenceKind) for kind in values):
        raise TypeError("Evidence kinds must be CapabilityEvidenceKind members.")
    return values


class CapabilityRequirement(StrictModule, NonTrainableState):
    """Evidence a consumer requires before relying on one capability.

    The requirement is satisfied when the provided evidence kinds include every
    kind of at least one alternative. A safety-critical requirement never accepts
    declaration alone, so an alternative consisting only of `DECLARED` is
    rejected at construction.
    """

    capability_id: str = eqx.field(static=True)
    alternatives: tuple[tuple[CapabilityEvidenceKind, ...], ...] = eqx.field(static=True)
    safety_critical: bool = eqx.field(static=True)

    def __init__(
        self,
        capability_id: str,
        alternatives: Iterable[Iterable[CapabilityEvidenceKind]],
        /,
        *,
        safety_critical: bool = False,
    ):
        capability_id_ = _identifier(capability_id, "capability_id")
        if not isinstance(safety_critical, bool):
            raise TypeError("safety_critical must be bool.")
        options = set()
        for alternative in alternatives:
            kinds = _evidence_kinds(alternative)
            if not kinds:
                raise ValueError("Evidence alternatives must be non-empty.")
            options.add(tuple(sorted(kinds, key=_EVIDENCE_ORDER.__getitem__)))
        if not options:
            raise ValueError("A capability requirement needs at least one alternative.")
        if safety_critical and (CapabilityEvidenceKind.DECLARED,) in options:
            raise ValueError(
                "A safety-critical requirement cannot be satisfied by declaration alone."
            )
        self.capability_id = capability_id_
        self.alternatives = tuple(
            sorted(
                options,
                key=lambda option: tuple(_EVIDENCE_ORDER[kind] for kind in option),
            )
        )
        self.safety_critical = safety_critical

    def is_satisfied_by(self, kinds: Iterable[CapabilityEvidenceKind], /) -> bool:
        """Return whether the provided evidence kinds satisfy an alternative."""
        provided = _evidence_kinds(kinds)
        return any(provided.issuperset(option) for option in self.alternatives)


class AbstractConstructionCertificate(StrictModule, NonTrainableState):
    """Construction evidence establishing one declared capability.

    Implementations provide `capability_id` (the capability the construction
    establishes) and `certificate_id` (the content identity of this certificate)
    as fields or properties.
    """

    capability_id: eqx.AbstractVar[str]
    certificate_id: eqx.AbstractVar[str]


__all__ = [
    "AbstractConstructionCertificate",
    "BranchDifferentiationPolicy",
    "CapabilityEvidenceKind",
    "CapabilityRequirement",
    "ComponentAuthority",
    "DERIVATIVE_SUPPORTED",
    "DERIVATIVE_UNSUPPORTED",
    "DerivativeAdmission",
    "DerivativeContract",
    "DerivativeRegularity",
    "DerivativeRoute",
    "DerivativeSurface",
    "DifferentiationRequest",
    "GradientLevel",
    "ObjectiveKind",
    "RegularityPieces",
    "RegularityPolicy",
    "SurfaceDerivative",
    "admit_regularity",
    "authority_admits",
    "branch_policy_contract",
    "derivative_contract_from_payload",
    "derivative_contract_payload",
    "gradient_level_at_least",
    "is_owned_surface",
    "resolve_gradient_level",
    "weakest_level",
]

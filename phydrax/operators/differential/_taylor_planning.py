# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._differentiation import (
    admit_regularity,
    DerivativeAdmission,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from ..._fingerprint import canonical_fingerprint
from ..._sampling._addressing import derive_key, SampleAddress
from ..._sampling._designs import seed_from_key
from ..._strict import StrictModule
from ...typing import parse, PRNGKey
from ._taylor_certification import (
    _bounded_candidates,
    taylor_prime_candidates,
    TaylorCurveCertificate,
    TaylorLinearCertificate,
)
from ._taylor_contracts import (
    TaylorContractionExtraction,
    TaylorContractionPolicy,
    TaylorContractionRequest,
    TaylorContractionResources,
    TaylorContractionSchedule,
    TaylorContractionStrategy,
)


type _CurveCoefficients = tuple[tuple[str, int, int], ...]
type _RecipeTerm = tuple[_CurveCoefficients, int, int, int]


@final
class TaylorContractionRecipe(StrictModule):
    request: TaylorContractionRequest = eqx.field(static=True)
    strategy: TaylorContractionStrategy = eqx.field(static=True)
    certificate: TaylorCurveCertificate | TaylorLinearCertificate = eqx.field(static=True)

    def __init__(
        self,
        request: TaylorContractionRequest,
        strategy: TaylorContractionStrategy,
        *,
        resources: TaylorContractionResources,
        degrees: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(request, TaylorContractionRequest):
            raise TypeError("request must be TaylorContractionRequest.")
        selected = parse(strategy, TaylorContractionStrategy, "strategy")
        match selected:
            case "linear":
                if degrees is not None:
                    raise ValueError("Linear recipes do not accept curve degrees.")
                certificate = TaylorLinearCertificate(
                    request.multiplicities, resources=resources
                )
            case "single" | "prime":
                if degrees is None:
                    raise ValueError("Single-curve recipes require explicit degrees.")
                certificate = TaylorCurveCertificate(
                    degrees, request.multiplicities, resources=resources
                )
                if not certificate.valid:
                    raise ValueError(
                        "Curve has colliding partitions; its target is not isolated."
                    )
                if selected == "prime" and any(
                    degree < 2
                    or any(
                        degree % divisor == 0
                        for divisor in range(2, math.isqrt(degree) + 1)
                    )
                    for degree in degrees
                ):
                    raise ValueError("Prime recipes require prime degrees.")
            case "auto":
                raise ValueError("An executed recipe must specify its actual strategy.")
            case _:
                assert_never(selected)
        self.request = request
        self.strategy = selected
        self.certificate = certificate


def _admission(
    regularity: DerivativeRegularity | None,
    policy: RegularityPolicy,
    order: int,
) -> DerivativeAdmission:
    return admit_regularity(
        regularity,
        DifferentiationRequest((DerivativeSurface.INPUT,), order=order),
        route=DerivativeRoute.DIRECT,
        policy=policy,
    )


def _curve_admission(
    regularity: DerivativeRegularity | None,
    policy: RegularityPolicy,
    max_slot: int,
    order: int,
) -> DerivativeAdmission:
    # Jet differentiates f(curve(t)), not the source map at the executed scalar
    # order. Composition multiplies polynomial bounds but cannot strengthen C^k.
    composed = (
        None
        if regularity is None
        else DerivativeRegularity.smooth(
            degree_bound=max_slot,
        ).compose(regularity)
    )
    return _admission(composed, policy, order)


def _recipe_terms(
    recipe: TaylorContractionRecipe,
) -> tuple[_RecipeTerm, ...]:
    certificate = recipe.certificate
    if isinstance(certificate, TaylorCurveCertificate):
        if not certificate.valid:
            raise ValueError("Taylor execution refuses an invalid curve certificate.")
        curve = tuple(
            (identifier, degree, 1)
            for identifier, degree in zip(
                recipe.request.direction_ids, certificate.degrees, strict=True
            )
        )
        numerator, denominator = certificate.extraction_weight
        return ((curve, certificate.coefficient_order, numerator, denominator),)
    return tuple(
        (
            tuple(
                (identifier, 1, scale)
                for identifier, scale in zip(
                    recipe.request.direction_ids, scales, strict=True
                )
                if scale
            ),
            certificate.coefficient_order,
            weight,
            1,
        )
        for scales, weight in certificate.terms
    )


def _coalesce_schedules(
    terms: tuple[tuple[_RecipeTerm, ...], ...],
    policy: TaylorContractionPolicy,
) -> tuple[
    tuple[TaylorContractionSchedule, ...],
    tuple[tuple[TaylorContractionExtraction, ...], ...],
    dict[_CurveCoefficients, int],
]:
    orders: dict[_CurveCoefficients, int] = {}
    for request_terms in terms:
        for curve, order, _, _ in request_terms:
            orders[curve] = max(order, orders.get(curve, 0))
    curves = tuple(sorted(orders))
    if len(curves) > policy.resources.max_linear_terms:
        raise ValueError("Shared Taylor schedules exceed bounded plan resources.")
    schedules = tuple(TaylorContractionSchedule(curve, orders[curve]) for curve in curves)
    positions = {curve: index for index, curve in enumerate(curves)}
    extractions = tuple(
        tuple(
            TaylorContractionExtraction(positions[curve], order, numerator, denominator)
            for curve, order, numerator, denominator in request_terms
        )
        for request_terms in terms
    )
    return schedules, extractions, orders


def _plan_identity(
    requests: tuple[TaylorContractionRequest, ...],
    checked: tuple[TaylorContractionRecipe, ...],
    schedules: tuple[TaylorContractionSchedule, ...],
    policy: TaylorContractionPolicy,
    regularity: DerivativeRegularity | None,
    permissions: RegularityPolicy,
) -> str:
    canonical_requests = sorted(
        (request.request_id, request.direction_ids, request.multiplicities)
        for request in requests
    )
    return canonical_fingerprint(
        {
            "requests": canonical_requests,
            "output_layout": [
                (request.request_id, request.direction_ids, request.multiplicities)
                for request in requests
            ],
            "policy": (
                policy.strategy,
                policy.resources.max_order,
                policy.resources.max_certificate_states,
                policy.resources.max_linear_terms,
                policy.resources.max_candidates,
                policy.resources.workset_size,
                policy.resources.max_logical_buffer_elements,
            ),
            "recipes": sorted(
                (
                    recipe.request.request_id,
                    recipe.strategy,
                    recipe.certificate.degrees
                    if isinstance(recipe.certificate, TaylorCurveCertificate)
                    else (),
                )
                for recipe in checked
            ),
            "schedules": [
                (schedule.coefficients, schedule.order) for schedule in schedules
            ],
            "regularity": None
            if regularity is None
            else (
                regularity.continuity,
                regularity.pieces,
                regularity.degree_bound,
                regularity.conditions,
                regularity.support,
            ),
            "permissions": (
                permissions.allow_almost_everywhere,
                permissions.allow_undeclared,
            ),
        }
    )


@final
class TaylorContractionPlan(StrictModule):
    """Immutable mathematical metadata; no callable, point, or direction leaves."""

    requests: tuple[TaylorContractionRequest, ...] = eqx.field(static=True)
    direction_ids: tuple[str, ...] = eqx.field(static=True)
    recipes: tuple[TaylorContractionRecipe, ...] = eqx.field(static=True)
    schedules: tuple[TaylorContractionSchedule, ...] = eqx.field(static=True)
    extractions: tuple[tuple[TaylorContractionExtraction, ...], ...] = eqx.field(
        static=True
    )
    policy: TaylorContractionPolicy = eqx.field(static=True)
    requested_admissions: tuple[DerivativeAdmission, ...] = eqx.field(static=True)
    executed_admissions: tuple[DerivativeAdmission, ...] = eqx.field(static=True)
    required_regularity_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        recipes: Sequence[TaylorContractionRecipe],
        *,
        policy: TaylorContractionPolicy,
        regularity: DerivativeRegularity | None = None,
        regularity_policy: RegularityPolicy | None = None,
    ) -> None:
        selected = tuple(recipes)
        if not selected or any(
            not isinstance(item, TaylorContractionRecipe) for item in selected
        ):
            raise ValueError("A Taylor plan requires nonempty certified recipes.")
        if not isinstance(policy, TaylorContractionPolicy):
            raise TypeError("policy must be TaylorContractionPolicy.")
        permissions = (
            RegularityPolicy() if regularity_policy is None else regularity_policy
        )
        requests = tuple(recipe.request for recipe in selected)
        identities: dict[str, tuple[tuple[str, int], ...]] = {}
        for request in requests:
            identity = tuple(
                zip(request.direction_ids, request.multiplicities, strict=True)
            )
            if (
                request.request_id in identities
                and identities[request.request_id] != identity
            ):
                raise ValueError("One request ID cannot denote different contractions.")
            identities[request.request_id] = identity
        # Reconstruct certificates through their owning constructors. User-claimed
        # validity and malformed restoration cannot enter a mathematical plan.
        checked = tuple(
            TaylorContractionRecipe(
                recipe.request,
                recipe.strategy,
                resources=policy.resources,
                degrees=recipe.certificate.degrees
                if isinstance(recipe.certificate, TaylorCurveCertificate)
                else None,
            )
            for recipe in selected
        )
        terms = tuple(_recipe_terms(recipe) for recipe in checked)
        schedules, extractions, orders = _coalesce_schedules(terms, policy)
        requested = tuple(
            _admission(regularity, permissions, request.order) for request in requests
        )
        executed = tuple(
            _curve_admission(
                regularity,
                permissions,
                max(degree for curve, _, _, _ in request_terms for _, degree, _ in curve),
                max(orders[curve] for curve, _, _, _ in request_terms),
            )
            for request_terms in terms
        )
        if any(not item.supported for item in (*requested, *executed)):
            reasons = sorted(
                {reason for item in (*requested, *executed) for reason in item.reasons}
            )
            raise ValueError(
                f"Taylor derivative regularity is not admitted: {', '.join(reasons)}."
            )
        identity = _plan_identity(
            requests, checked, schedules, policy, regularity, permissions
        )
        self.requests = requests
        self.direction_ids = tuple(
            sorted(
                {
                    identifier
                    for request in requests
                    for identifier in request.direction_ids
                }
            )
        )
        self.recipes = checked
        self.schedules = schedules
        self.extractions = extractions
        self.policy = policy
        self.requested_admissions = requested
        self.executed_admissions = executed
        self.required_regularity_order = max(schedule.order for schedule in schedules)
        self.plan_id = identity


def _recipe_options(
    request: TaylorContractionRequest,
    policy: TaylorContractionPolicy,
    regularity: DerivativeRegularity | None,
    permissions: RegularityPolicy,
    seed: int | None,
) -> tuple[TaylorContractionRecipe, ...]:
    resources = policy.resources
    if request.order > resources.max_order:
        raise ValueError("Requested derivative order exceeds Taylor resources.")
    if not _admission(regularity, permissions, request.order).supported:
        raise ValueError("Requested derivative regularity is not admitted.")
    if len(request.direction_ids) == 1:
        return (
            TaylorContractionRecipe(request, "single", resources=resources, degrees=(1,)),
        )
    options: list[TaylorContractionRecipe] = []
    counts = request.multiplicities
    linear_admitted = (
        math.prod(m + 1 for m in counts) <= resources.max_linear_terms
        and sum((m + 1) ** 2 for m in counts) <= resources.max_certificate_states
        and _curve_admission(regularity, permissions, 1, request.order).supported
    )
    match policy.strategy:
        case "linear":
            return (TaylorContractionRecipe(request, "linear", resources=resources),)
        case "auto" | "single" | "prime":
            if policy.strategy == "auto" and linear_admitted:
                options.append(
                    TaylorContractionRecipe(request, "linear", resources=resources)
                )
            max_degree = min(
                resources.max_order,
                resources.max_certificate_states // len(request.direction_ids) - 1,
            )
            candidates = (
                taylor_prime_candidates(counts, resources=resources, seed=seed)
                if policy.strategy == "prime"
                else _bounded_candidates(
                    tuple(range(1, max_degree + 1)), counts, resources, seed
                )
            )
            for degrees in candidates:
                order = sum(a * m for a, m in zip(degrees, counts, strict=True))
                if (order + 1) * len(degrees) > resources.max_certificate_states:
                    continue
                if not _curve_admission(
                    regularity, permissions, max(degrees), order
                ).supported:
                    continue
                certificate = TaylorCurveCertificate(degrees, counts, resources=resources)
                if certificate.valid:
                    options.append(
                        TaylorContractionRecipe(
                            request,
                            "prime" if policy.strategy == "prime" else "single",
                            resources=resources,
                            degrees=degrees,
                        )
                    )
        case _:
            assert_never(policy.strategy)
    if not options:
        raise ValueError(
            "No certified admitted recipe within bounded candidate resources."
        )
    return tuple(options)


def _recipe_rank(
    recipes: Sequence[TaylorContractionRecipe],
) -> tuple[int, int, int, int, Fraction, tuple[tuple[str, tuple[int, ...]], ...]]:
    """Rank actual shared schedule work, not the first isolated certificate."""
    orders: dict[tuple[tuple[str, int, int], ...], int] = {}
    cancellation = Fraction(0)
    identities: list[tuple[str, tuple[int, ...]]] = []
    for recipe in recipes:
        for curve, order, numerator, denominator in _recipe_terms(recipe):
            orders[curve] = max(order, orders.get(curve, 0))
            cancellation += abs(Fraction(numerator, denominator))
        certificate = recipe.certificate
        identities.append(
            (
                recipe.strategy,
                certificate.degrees
                if isinstance(certificate, TaylorCurveCertificate)
                else (),
            )
        )
    return (
        sum(order**2 for order in orders.values()),
        len(orders),
        max(orders.values()),
        sum(orders.values()),
        cancellation,
        tuple(identities),
    )


def _ranked_recipes(
    requests: tuple[TaylorContractionRequest, ...],
    policy: TaylorContractionPolicy,
    regularity: DerivativeRegularity | None,
    permissions: RegularityPolicy,
    seed: int | None,
) -> tuple[TaylorContractionRecipe, ...]:
    # Canonical greedy admission followed by one bounded coordinate sweep accounts
    # for sharing without an exponential Cartesian search over request recipes.
    canonical = tuple(
        sorted(
            {request.request_id: request for request in requests}.values(),
            key=lambda request: (
                request.direction_ids,
                request.multiplicities,
                request.request_id,
            ),
        )
    )
    options = tuple(
        _recipe_options(request, policy, regularity, permissions, seed)
        for request in canonical
    )
    chosen: list[TaylorContractionRecipe] = []
    for candidates in options:
        admissible = tuple(
            candidate
            for candidate in candidates
            if _recipe_rank((*chosen, candidate))[1] <= policy.resources.max_linear_terms
        )
        if not admissible:
            raise ValueError("Shared Taylor schedules exceed bounded plan resources.")
        chosen.append(
            min(admissible, key=lambda candidate: _recipe_rank((*chosen, candidate)))
        )
    for index, candidates in enumerate(options):
        others = (*chosen[:index], *chosen[index + 1 :])
        admissible = tuple(
            candidate
            for candidate in candidates
            if _recipe_rank((*others, candidate))[1] <= policy.resources.max_linear_terms
        )
        chosen[index] = min(
            admissible, key=lambda candidate: _recipe_rank((*others, candidate))
        )
    by_identity = {recipe.request.request_id: recipe for recipe in chosen}
    return tuple(by_identity[request.request_id] for request in requests)


def plan_taylor_contractions(
    requests: Sequence[TaylorContractionRequest],
    *,
    policy: TaylorContractionPolicy | None = None,
    regularity: DerivativeRegularity | None = None,
    regularity_policy: RegularityPolicy | None = None,
    key: PRNGKey | None = None,
) -> TaylorContractionPlan:
    """Prepare bounded exact contractions, preserving caller request order."""
    selected = TaylorContractionPolicy() if policy is None else policy
    if not isinstance(selected, TaylorContractionPolicy):
        raise TypeError("policy must be TaylorContractionPolicy.")
    permissions = RegularityPolicy() if regularity_policy is None else regularity_policy
    if not isinstance(permissions, RegularityPolicy):
        raise TypeError("regularity_policy must be RegularityPolicy.")
    items = tuple(requests)
    if not items or any(not isinstance(item, TaylorContractionRequest) for item in items):
        raise ValueError("requests must be nonempty TaylorContractionRequest values.")
    seed = None
    if key is not None:
        root = jnp.asarray(key)
        if root.shape != () or not jnp.issubdtype(root.dtype, jax.dtypes.prng_key):
            raise TypeError("Taylor planning requires a scalar typed JAX key.")
        # One intentional host boundary, semantically addressed independently of
        # execution layout. Runtime Taylor evaluation never consumes randomness.
        seed = seed_from_key(
            derive_key(root, SampleAddress("differential", "taylor", role="planning"))
        )
    identities: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
    for request in items:
        identity = (request.direction_ids, request.multiplicities)
        if (
            request.request_id in identities
            and identities[request.request_id] != identity
        ):
            raise ValueError("One request ID cannot denote different contractions.")
        identities[request.request_id] = identity
    recipes = _ranked_recipes(items, selected, regularity, permissions, seed)
    return TaylorContractionPlan(
        recipes, policy=selected, regularity=regularity, regularity_policy=permissions
    )

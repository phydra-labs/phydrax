# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import final

import equinox as eqx

from ..._strict import StrictModule
from ..._validation import positive_integer
from ._taylor_contracts import TaylorContractionResources


def _counts(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(positive_integer(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


@final
class TaylorCurveCertificate(StrictModule):
    """Exact unrestricted integer-partition certificate for a scalar curve.

    A curve sum(v_i t**a_i) isolates the target iff the weighted degree has
    exactly one nonnegative partition. Counts are saturated at two: only
    uniqueness is claimed, not a count of all collisions.
    """

    degrees: tuple[int, ...] = eqx.field(static=True)
    multiplicities: tuple[int, ...] = eqx.field(static=True)
    coefficient_order: int = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    target_coefficient: tuple[int, int] = eqx.field(static=True)
    extraction_weight: tuple[int, int] = eqx.field(static=True)
    state_count: int = eqx.field(static=True)

    def __init__(
        self,
        degrees: Sequence[int],
        multiplicities: Sequence[int],
        *,
        resources: TaylorContractionResources | None = None,
    ) -> None:
        degree = _counts(degrees, "degree")
        counts = _counts(multiplicities, "multiplicity")
        if len(degree) != len(counts):
            raise ValueError("Degrees and multiplicities must align.")
        limits = TaylorContractionResources() if resources is None else resources
        if not isinstance(limits, TaylorContractionResources):
            raise TypeError("resources must be TaylorContractionResources.")
        order = sum(a * m for a, m in zip(degree, counts, strict=True))
        states = (order + 1) * len(degree)
        if order > limits.max_order or states > limits.max_certificate_states:
            raise ValueError("Curve certificate exceeds bounded preparation resources.")
        # Coin-change DP counts all n_i >= 0, including lower/higher derivative
        # orders. Restricting total order to the target would miss KdV collisions.
        partitions = [0] * (order + 1)
        partitions[0] = 1
        for a in degree:
            for total in range(a, order + 1):
                partitions[total] = min(2, partitions[total] + partitions[total - a])
        factor = math.prod(math.factorial(m) for m in counts)
        rational = Fraction(1, factor)
        self.degrees = degree
        self.multiplicities = counts
        self.coefficient_order = order
        self.partition_count = partitions[order]
        self.target_coefficient = (rational.numerator, rational.denominator)
        self.extraction_weight = (factor, 1)
        self.state_count = states

    @property
    def valid(self) -> bool:
        return self.partition_count == 1


@final
class TaylorLinearCertificate(StrictModule):
    """Signed-binomial finite-difference identity at normalized jet order k.

    The kth coefficient of f(x+t sum(s_i v_i)) is combined with
    (-1)**(k-sum(s_i))*prod(binomial(m_i,s_i)). Lower monomials cancel;
    the surviving multinomial and finite-difference factors cancel k! exactly.
    """

    multiplicities: tuple[int, ...] = eqx.field(static=True)
    coefficient_order: int = eqx.field(static=True)
    terms: tuple[tuple[tuple[int, ...], int], ...] = eqx.field(static=True)
    target_coefficient: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        multiplicities: Sequence[int],
        *,
        resources: TaylorContractionResources | None = None,
    ) -> None:
        counts = _counts(multiplicities, "multiplicity")
        limits = TaylorContractionResources() if resources is None else resources
        if not isinstance(limits, TaylorContractionResources):
            raise TypeError("resources must be TaylorContractionResources.")
        order = sum(counts)
        term_count = math.prod(m + 1 for m in counts)
        moment_work = sum((m + 1) ** 2 for m in counts)
        if (
            order > limits.max_order
            or term_count > limits.max_linear_terms
            or moment_work > limits.max_certificate_states
        ):
            raise ValueError("Linear certificate exceeds bounded preparation resources.")
        for m in counts:
            for power in range(m + 1):
                moment = sum(
                    (-1) ** (m - s) * math.comb(m, s) * s**power for s in range(m + 1)
                )
                expected = math.factorial(m) if power == m else 0
                if moment != expected:
                    raise ValueError("Signed binomial identity is invalid.")
        terms = []
        for scales in itertools.product(*(range(m + 1) for m in counts)):
            if not any(scales):
                continue  # The positive-order coefficient of a constant curve is zero.
            weight = (-1) ** (order - sum(scales)) * math.prod(
                math.comb(m, s) for m, s in zip(counts, scales, strict=True)
            )
            terms.append((scales, weight))
        self.multiplicities = counts
        self.coefficient_order = order
        self.terms = tuple(terms)
        self.target_coefficient = (1, 1)

    @property
    def valid(self) -> bool:
        return True


def taylor_prime_candidates(
    multiplicities: Sequence[int],
    *,
    resources: TaylorContractionResources | None = None,
    seed: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """Deterministic bounded prime schedules; each still requires certification.

    No polynomial-time completeness claim: exhausting candidates is a refusal,
    never evidence that an unchecked schedule is valid.
    """
    counts = _counts(multiplicities, "multiplicity")
    limits = TaylorContractionResources() if resources is None else resources
    if not isinstance(limits, TaylorContractionResources):
        raise TypeError("resources must be TaylorContractionResources.")
    primes: list[int] = []
    max_degree = min(limits.max_order, limits.max_certificate_states // len(counts) - 1)
    for candidate in range(2, max_degree + 1):
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
    yield from _bounded_candidates(tuple(primes), counts, limits, seed)


def _bounded_candidates(
    degrees: tuple[int, ...],
    counts: tuple[int, ...],
    resources: TaylorContractionResources,
    seed: int | None,
) -> Iterator[tuple[int, ...]]:
    # Bound candidate traversal, not merely the number of accepted candidates.
    # Preserve the low-degree frontier. A semantic seed changes assignment of
    # degrees to directions, not whether small candidate subsets are visited.
    offset = 0 if seed is None else seed % len(counts)
    candidates = (
        schedule
        for last in range(len(counts) - 1, len(degrees))
        for prefix in itertools.combinations(degrees[:last], len(counts) - 1)
        for subset in ((*prefix, degrees[last]),)
        for schedule in itertools.permutations(subset[offset:] + subset[:offset])
    )
    for schedule in itertools.islice(candidates, resources.max_candidates):
        if (
            sum(a * m for a, m in zip(schedule, counts, strict=True))
            <= resources.max_order
        ):
            yield schedule

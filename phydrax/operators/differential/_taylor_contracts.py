# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Sequence
from typing import final, Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier, nonnegative_integer, positive_integer
from ...typing import Dim, Identifier, Identifiers, parse, Scope, Size, VariadicDim


TaylorContractionStrategy: TypeAlias = Literal["auto", "linear", "single", "prime"]


class TaylorDirectionDim(Dim, minimum=1):
    """Number of explicitly identified directions in one request."""


class TaylorRequestDim(Dim, minimum=1):
    """Number of ordered contraction output rows."""


class TaylorEventDims(VariadicDim):
    """Output event layout shared by primal and contraction values."""


@final
class TaylorContractionRequest(StrictModule):
    """An unscaled derivative contraction with explicitly identified directions."""

    __strict_contract__ = True

    direction_ids: Identifiers[TaylorDirectionDim] = eqx.field(static=True)
    multiplicities: tuple[int, ...] = eqx.field(static=True)
    request_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        direction_ids: Sequence[str],
        multiplicities: Sequence[int],
        *,
        request_id: str | None = None,
    ) -> None:
        scope = Scope()
        ids = tuple(canonical_identifier(item, "direction ID") for item in direction_ids)
        counts = tuple(positive_integer(item, "multiplicity") for item in multiplicities)
        if not ids or len(ids) != len(counts):
            raise ValueError(
                "Directions and multiplicities must be nonempty and aligned."
            )
        if len(set(ids)) != len(ids):
            raise ValueError("Each direction ID must occur once; use multiplicities.")
        parse(ids, Identifiers[TaylorDirectionDim], "direction_ids", scope=scope)
        parse(len(counts), Size[TaylorDirectionDim], "multiplicity count", scope=scope)
        canonical = tuple(sorted(zip(ids, counts, strict=True)))
        identity = canonical_fingerprint({"directions": canonical})
        supplied_id = (
            None if request_id is None else canonical_identifier(request_id, "request_id")
        )
        self.direction_ids = tuple(item[0] for item in canonical)
        self.multiplicities = tuple(item[1] for item in canonical)
        self.request_id = identity if supplied_id is None else supplied_id

    @property
    def order(self) -> int:
        return sum(self.multiplicities)


@final
class TaylorContractionResources(StrictModule):
    """Hard preparation and logical-buffer limits, not compiler-temporary claims."""

    max_order: int = eqx.field(static=True)
    max_certificate_states: int = eqx.field(static=True)
    max_linear_terms: int = eqx.field(static=True)
    max_candidates: int = eqx.field(static=True)
    workset_size: int = eqx.field(static=True)
    max_logical_buffer_elements: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_order: int = 128,
        max_certificate_states: int = 65536,
        max_linear_terms: int = 4096,
        max_candidates: int = 256,
        workset_size: int = 16,
        max_logical_buffer_elements: int = 1048576,
    ) -> None:
        self.max_order = positive_integer(max_order, "max_order")
        self.max_certificate_states = positive_integer(
            max_certificate_states, "max_certificate_states"
        )
        self.max_linear_terms = positive_integer(max_linear_terms, "max_linear_terms")
        self.max_candidates = positive_integer(max_candidates, "max_candidates")
        self.workset_size = positive_integer(workset_size, "workset_size")
        self.max_logical_buffer_elements = positive_integer(
            max_logical_buffer_elements, "max_logical_buffer_elements"
        )


@final
class TaylorContractionPolicy(StrictModule):
    strategy: TaylorContractionStrategy = eqx.field(static=True)
    resources: TaylorContractionResources = eqx.field(static=True)

    def __init__(
        self,
        *,
        strategy: TaylorContractionStrategy = "auto",
        resources: TaylorContractionResources | None = None,
    ) -> None:
        selected = parse(strategy, TaylorContractionStrategy, "strategy")
        if resources is not None and not isinstance(
            resources, TaylorContractionResources
        ):
            raise TypeError("resources must be TaylorContractionResources.")
        self.strategy = selected
        self.resources = TaylorContractionResources() if resources is None else resources


@final
class TaylorContractionSchedule(StrictModule):
    """One shared scalar polynomial curve, coefficients indexed by direction ID."""

    coefficients: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __init__(
        self, coefficients: tuple[tuple[str, int, int], ...], order: int
    ) -> None:
        if not coefficients or tuple(sorted(coefficients)) != coefficients:
            raise ValueError("Curve coefficients must be nonempty and canonical.")
        order = positive_integer(order, "order")
        for identifier, degree, scale in coefficients:
            canonical_identifier(identifier, "curve direction")
            positive_integer(degree, "degree")
            positive_integer(scale, "scale")
            if degree > order:
                raise ValueError("Curve degree exceeds executed order.")
        if len({item[0] for item in coefficients}) != len(coefficients):
            raise ValueError("Curve direction IDs must be unique.")
        self.coefficients = coefficients
        self.order = order


@final
class TaylorContractionExtraction(StrictModule):
    schedule_index: int = eqx.field(static=True)
    coefficient_order: int = eqx.field(static=True)
    numerator: int = eqx.field(static=True)
    denominator: int = eqx.field(static=True)

    def __init__(
        self,
        schedule_index: int,
        coefficient_order: int,
        numerator: int,
        denominator: int = 1,
    ) -> None:
        schedule_index = nonnegative_integer(schedule_index, "schedule_index")
        coefficient_order = positive_integer(coefficient_order, "coefficient_order")
        denominator = positive_integer(denominator, "denominator")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator == 0
        ):
            raise ValueError("Extraction numerator must be a nonzero integer.")
        self.schedule_index = schedule_index
        self.coefficient_order = coefficient_order
        self.numerator = numerator
        self.denominator = denominator

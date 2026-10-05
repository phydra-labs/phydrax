#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse


AdditiveIMEXScheme: TypeAlias = Literal[
    "forward-backward-euler", "ars-222", "ssp2-222", "ars-443"
]
"""Named additive IMEX Runge--Kutta schemes with published coefficients.

``forward-backward-euler`` is first order. ``ars-222`` (Ascher, Ruuth and
Spiteri, Appl. Numer. Math. 25, 1997, Sec. 2.6) and ``ssp2-222`` (Pareschi and
Russo, J. Sci. Comput. 25, 2005) are second order; ``ars-443`` (ARS Sec. 2.8) is
third order. Every ARS scheme and forward--backward Euler is stiffly accurate
with an explicit-only first stage.
"""


ImplicitCallbacks: TypeAlias = Callable[..., Any] | tuple[Callable[..., Any], ...]
"""One implicit callback per implicit part; a single part accepts a bare callable."""


def _part_callables(
    callbacks: ImplicitCallbacks, count: int, name: str, /
) -> tuple[Callable[..., Any], ...]:
    """Return exactly one callable per implicit part."""
    if isinstance(callbacks, tuple):
        values = callbacks
    elif callable(callbacks) and count == 1:
        values = (callbacks,)
    else:
        raise TypeError(f"{name} must hold one callable per implicit part.")
    if len(values) != count or not all(callable(value) for value in values):
        raise TypeError(f"{name} must hold one callable per implicit part.")
    return values


def _implicit_parts(
    implicit: np.ndarray,
    weights: np.ndarray,
    declared: tuple[int | None, ...] | None,
    /,
) -> tuple[int | None, ...]:
    count = weights.size
    parts = (0,) * count if declared is None else tuple(declared)
    if len(parts) != count or any(
        part is not None and (type(part) is not int or part < 0) for part in parts
    ):
        raise ValueError("implicit_parts must give every stage a part index or None.")
    used = sorted({part for part in parts if part is not None})
    if used != list(range(len(used))):
        raise ValueError("implicit_parts must number parts consecutively from zero.")
    for stage, part in enumerate(parts):
        if part is None and (np.any(implicit[:, stage] != 0.0) or weights[stage] != 0.0):
            raise ValueError(
                "An explicit-only stage requires a zero implicit column and weight."
            )
    for part in used:
        selected = [stage for stage, value in enumerate(parts) if value == part]
        if not np.isclose(np.sum(weights[selected]), 1.0):
            raise ValueError("Implicit weights must sum to one for every implicit part.")
    return parts


class AdditiveIMEXTableau(StrictModule, NonTrainableState):
    """Additive Runge--Kutta tableau of one explicit part and ordered implicit parts.

    Stage ``s`` evaluates the explicit rate and the implicit rate of part
    ``implicit_parts[s]``. Distinct parts give block-sequential stages: each
    stage solves one implicit part with every earlier stage increment fixed
    (the N-additive form of Kennedy and Carpenter, Appl. Numer. Math. 44,
    2003). ``None`` declares an explicit-only stage; its implicit diagonal,
    implicit column and implicit weight must vanish and no implicit callback
    runs for it. ``weights`` are the implicit weights and sum to one per part;
    ``explicit_weights`` default to ``weights``. Separate weights express
    schemes such as forward--backward Euler whose explicit and implicit
    quadratures differ.

    Two algebraic identities of the rows are detected once on the host and
    used exactly: ``stage_reuse[s]`` when stage ``s`` starts from the value of
    stage ``s - 1`` (identical earlier coefficients and no new explicit
    term), and ``stiffly_accurate`` when the last rows equal the weights, so
    the step result is the last stage value. Both avoid re-summing stage
    rates, which keeps committed components such as constrained values exact.
    """

    explicit_matrix: Array
    implicit_matrix: Array
    weights: Array
    nodes: Array
    explicit_weights: Array
    implicit_parts: tuple[int | None, ...] = eqx.field(static=True)
    stage_reuse: tuple[bool, ...] = eqx.field(static=True)
    stiffly_accurate: bool = eqx.field(static=True)
    tableau_id: str = eqx.field(static=True)

    def __init__(
        self,
        explicit_matrix: ArrayLike,
        implicit_matrix: ArrayLike,
        weights: ArrayLike,
        nodes: ArrayLike,
        /,
        *,
        explicit_weights: ArrayLike | None = None,
        implicit_parts: tuple[int | None, ...] | None = None,
    ) -> None:
        explicit = np.asarray(explicit_matrix, dtype=np.float64)
        implicit = np.asarray(implicit_matrix, dtype=np.float64)
        weights_ = np.asarray(weights, dtype=np.float64)
        nodes_ = np.asarray(nodes, dtype=np.float64)
        explicit_weights_ = (
            weights_
            if explicit_weights is None
            else np.asarray(explicit_weights, dtype=np.float64)
        )
        if (
            explicit.ndim != 2
            or explicit.shape[0] != explicit.shape[1]
            or implicit.shape != explicit.shape
            or weights_.shape != (explicit.shape[0],)
            or explicit_weights_.shape != weights_.shape
            or explicit.shape[0] == 0
            or not np.all(np.isfinite(explicit))
            or not np.all(np.isfinite(implicit))
            or not np.all(np.isfinite(weights_))
            or not np.all(np.isfinite(explicit_weights_))
            or not np.all(np.isfinite(nodes_))
            or nodes_.shape != weights_.shape
            or np.any(np.triu(explicit) != 0.0)
            or np.any(np.triu(implicit, k=1) != 0.0)
            or not np.isclose(np.sum(explicit_weights_), 1.0)
        ):
            raise ValueError("Additive IMEX tableau is invalid.")
        parts = _implicit_parts(implicit, weights_, implicit_parts)
        count = weights_.size
        self.explicit_matrix = jnp.asarray(explicit)
        self.implicit_matrix = jnp.asarray(implicit)
        self.weights = jnp.asarray(weights_)
        self.nodes = jnp.asarray(nodes_)
        self.explicit_weights = jnp.asarray(explicit_weights_)
        self.implicit_parts = parts
        self.stage_reuse = tuple(
            bool(
                stage > 0
                and np.array_equal(
                    explicit[stage, : stage - 1], explicit[stage - 1, : stage - 1]
                )
                and explicit[stage, stage - 1] == 0.0
                and np.array_equal(implicit[stage, :stage], implicit[stage - 1, :stage])
            )
            for stage in range(count)
        )
        self.stiffly_accurate = bool(
            np.array_equal(explicit[-1], explicit_weights_)
            and np.array_equal(implicit[-1], weights_)
        )
        identity: dict[str, Any] = {
            "kind": "additive-imex-tableau",
            "explicit": explicit.tolist(),
            "implicit": implicit.tolist(),
            "weights": weights_.tolist(),
            "nodes": nodes_.tolist(),
        }
        # Shared weights and one implicit part are the classical ARK form and
        # keep their established identity.
        if not np.array_equal(explicit_weights_, weights_):
            identity["explicit_weights"] = explicit_weights_.tolist()
        if parts != (0,) * count:
            identity["implicit_parts"] = list(parts)
        self.tableau_id = canonical_fingerprint(identity)

    @property
    def stage_count(self) -> int:
        return self.weights.size

    @property
    def part_count(self) -> int:
        """Number of implicit parts, each with its own rate and stage solver."""
        return len({part for part in self.implicit_parts if part is not None})

    def explicit_rate_used(self, stage: int, /) -> bool:
        """Whether the explicit rate of ``stage`` enters a later stage or the result."""
        return not (self.stiffly_accurate and stage == self.stage_count - 1)

    def provisional(
        self,
        stage: int,
        state: Array,
        step_size: Array,
        explicit_rates: list[Array | None],
        implicit_rates: list[Array],
        stage_values: list[Array],
        /,
    ) -> Array:
        """Return the known part ``y + dt sum_j (a_E[s, j] E_j + a_I[s, j] I_j)``."""
        if self.stage_reuse[stage]:
            return stage_values[stage - 1]
        provisional = state
        for previous in range(stage):
            explicit_rate = explicit_rates[previous]
            if explicit_rate is None:
                raise ValueError("A later stage requires an omitted explicit rate.")
            provisional = provisional + step_size * (
                self.explicit_matrix[stage, previous] * explicit_rate
                + self.implicit_matrix[stage, previous] * implicit_rates[previous]
            )
        return provisional

    def combine(
        self,
        state: Array,
        step_size: Array,
        explicit_rates: list[Array | None],
        implicit_rates: list[Array],
        stage_values: list[Array],
        /,
    ) -> Array:
        """Return the step result from the stage rates or the last stage value."""
        if self.stiffly_accurate:
            return stage_values[-1]
        result = state
        for stage in range(self.stage_count):
            explicit_rate = explicit_rates[stage]
            if explicit_rate is None:
                raise ValueError("The step result requires an omitted explicit rate.")
            result = result + step_size * (
                self.explicit_weights[stage] * explicit_rate
                + self.weights[stage] * implicit_rates[stage]
            )
        return result

    def step(
        self,
        state: Array,
        time: Array,
        step_size: Array,
        explicit_rhs: Callable[[Array, Array, Any], ArrayLike],
        implicit_solve: ImplicitCallbacks,
        args: Any = None,
        /,
        *,
        implicit_rhs: ImplicitCallbacks,
    ) -> Array:
        """Apply the tableau; RHS callbacks take ``(state, time, args)``.

        ``implicit_solve`` and ``implicit_rhs`` hold one callback per implicit
        part; a single part also accepts a bare callable. The implicit RHS is
        required even when a solve callback is supplied: a zero diagonal or
        zero step does not determine it from a state change.
        """
        solvers = _part_callables(implicit_solve, self.part_count, "implicit_solve")
        rates = _part_callables(implicit_rhs, self.part_count, "implicit_rhs")
        state = jnp.asarray(state)
        structures = eqx.filter_eval_shape(
            lambda candidate: (
                jnp.asarray(explicit_rhs(candidate, time, args)),
                *(jnp.asarray(rate(candidate, time, args)) for rate in rates),
            ),
            state,
        )
        state = state.astype(
            jnp.result_type(
                state,
                step_size,
                self.weights,
                *(structure.dtype for structure in structures),
            )
        )
        solved_structures = eqx.filter_eval_shape(
            lambda candidate: tuple(
                jnp.asarray(solve(candidate, time, step_size, args)) for solve in solvers
            ),
            state,
        )
        state = state.astype(
            jnp.result_type(state, *(structure.dtype for structure in solved_structures))
        )
        explicit_stages: list[Array | None] = []
        implicit_stages: list[Array] = []
        stage_values: list[Array] = []
        for stage in range(self.stage_count):
            provisional = self.provisional(
                stage, state, step_size, explicit_stages, implicit_stages, stage_values
            )
            stage_time = time + self.nodes[stage] * step_size
            part = self.implicit_parts[stage]
            if part is None:
                solved = provisional
                implicit_stages.append(jnp.zeros_like(provisional))
            else:
                coefficient = step_size * self.implicit_matrix[stage, stage]
                solve = solvers[part]
                solved = jax.lax.cond(
                    coefficient != 0.0,
                    lambda provisional: jnp.asarray(
                        solve(provisional, stage_time, coefficient, args),
                        dtype=state.dtype,
                    ),
                    lambda provisional: provisional,
                    provisional,
                )
                safe_coefficient = jnp.where(coefficient != 0.0, coefficient, 1.0)
                implicit_stages.append(
                    jnp.where(
                        coefficient != 0.0,
                        (solved - provisional) / safe_coefficient,
                        rates[part](solved, stage_time, args),
                    )
                )
            stage_values.append(solved)
            explicit_stages.append(
                jnp.asarray(explicit_rhs(solved, stage_time, args))
                if self.explicit_rate_used(stage)
                else None
            )
        return self.combine(
            state, step_size, explicit_stages, implicit_stages, stage_values
        )


def additive_imex_tableau(scheme: AdditiveIMEXScheme, /) -> AdditiveIMEXTableau:
    """Return the published tableau of one named additive IMEX scheme."""
    selected = parse(scheme, AdditiveIMEXScheme, "scheme")
    match selected:
        case "forward-backward-euler":
            return AdditiveIMEXTableau(
                np.asarray(((0.0, 0.0), (1.0, 0.0))),
                np.asarray(((0.0, 0.0), (0.0, 1.0))),
                np.asarray((0.0, 1.0)),
                np.asarray((0.0, 1.0)),
                explicit_weights=np.asarray((1.0, 0.0)),
                implicit_parts=(None, 0),
            )
        case "ars-222":
            gamma = 1.0 - 1.0 / np.sqrt(2.0)
            delta = 1.0 - 1.0 / (2.0 * gamma)
            return AdditiveIMEXTableau(
                np.asarray(
                    ((0.0, 0.0, 0.0), (gamma, 0.0, 0.0), (delta, 1.0 - delta, 0.0))
                ),
                np.asarray(
                    ((0.0, 0.0, 0.0), (0.0, gamma, 0.0), (0.0, 1.0 - gamma, gamma))
                ),
                np.asarray((0.0, 1.0 - gamma, gamma)),
                np.asarray((0.0, gamma, 1.0)),
                explicit_weights=np.asarray((delta, 1.0 - delta, 0.0)),
                implicit_parts=(None, 0, 0),
            )
        case "ssp2-222":
            gamma = 1.0 - 1.0 / np.sqrt(2.0)
            return AdditiveIMEXTableau(
                np.asarray(((0.0, 0.0), (1.0, 0.0))),
                np.asarray(((gamma, 0.0), (1.0 - 2.0 * gamma, gamma))),
                np.asarray((0.5, 0.5)),
                np.asarray((gamma, 1.0)),
            )
        case "ars-443":
            return AdditiveIMEXTableau(
                np.asarray(
                    (
                        (0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.0, 0.0, 0.0),
                        (11.0 / 18.0, 1.0 / 18.0, 0.0, 0.0, 0.0),
                        (5.0 / 6.0, -5.0 / 6.0, 0.5, 0.0, 0.0),
                        (0.25, 1.75, 0.75, -1.75, 0.0),
                    )
                ),
                np.asarray(
                    (
                        (0.0, 0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.5, 0.0, 0.0, 0.0),
                        (0.0, 1.0 / 6.0, 0.5, 0.0, 0.0),
                        (0.0, -0.5, 0.5, 0.5, 0.0),
                        (0.0, 1.5, -1.5, 0.5, 0.5),
                    )
                ),
                np.asarray((0.0, 1.5, -1.5, 0.5, 0.5)),
                np.asarray((0.0, 0.5, 2.0 / 3.0, 0.5, 1.0)),
                explicit_weights=np.asarray((0.25, 1.75, 0.75, -1.75, 0.0)),
                implicit_parts=(None, 0, 0, 0, 0),
            )
        case _:
            assert_never(selected)


class BalanceLawCompositionPlan(StrictModule, NonTrainableState):
    """Symmetric process subcycles; each process owns its finite-update method."""

    process_subcycles: tuple[int, ...] = eqx.field(static=True)
    composition_id: str = eqx.field(static=True)

    def __init__(
        self,
        process_subcycles: tuple[int, ...],
        /,
    ) -> None:
        subcycles = tuple(process_subcycles)
        if not subcycles or any(value <= 0 for value in subcycles):
            raise ValueError("Balance-law multirate composition is invalid.")
        self.process_subcycles = subcycles
        self.composition_id = canonical_fingerprint(
            {
                "kind": "balance-law-symmetric-multirate-composition",
                "process_subcycles": list(subcycles),
            }
        )


__all__ = [
    "AdditiveIMEXScheme",
    "AdditiveIMEXTableau",
    "BalanceLawCompositionPlan",
    "additive_imex_tableau",
]

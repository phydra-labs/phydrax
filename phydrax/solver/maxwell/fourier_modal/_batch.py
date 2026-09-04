#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.spectral import PreparedBrillouinZone
from ._contracts import FourierModalMaxwellProblem
from ._runtime import (
    FourierModalRefreshSpec,
    FourierModalSolvePolicy,
    FourierModalSolveResult,
    prepare_fourier_modal_maxwell,
    PreparedFourierModalMaxwell,
    refresh_fourier_modal_maxwell,
    solve_fourier_modal_maxwell,
)
from ._sources import FourierModalExcitation


class PreparedFourierModalCaseBatch(StrictModule):
    """Static collection of independently prepared frequency/Bloch cases."""

    cases: tuple[PreparedFourierModalMaxwell, ...]
    case_shape: tuple[int, ...] = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)


class FourierModalCaseBatchResult(StrictModule):
    """Case-shaped directional port powers plus complete per-case results."""

    results: tuple[FourierModalSolveResult, ...]
    right_outgoing: Array
    left_outgoing: Array
    left_incoming_power: Array
    right_incoming_power: Array
    left_outgoing_power: Array
    right_outgoing_power: Array
    net_port_power_into_stack: Array
    power_audit_residual: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)


def _prepared_case_batch(
    cases: tuple[PreparedFourierModalMaxwell, ...],
    case_shape: tuple[int, ...],
    /,
) -> PreparedFourierModalCaseBatch:
    batch_id = canonical_fingerprint(
        {
            "kind": "prepared-fourier-modal-case-batch",
            "case_shape": list(case_shape),
            "cases": [value.preparation_id for value in cases],
        }
    )
    return PreparedFourierModalCaseBatch(cases, case_shape, batch_id=batch_id)


def prepare_fourier_modal_case_batch(
    problems: Sequence[FourierModalMaxwellProblem],
    case_shape: tuple[int, ...],
    /,
    *,
    policy: FourierModalSolvePolicy | None = None,
) -> PreparedFourierModalCaseBatch:
    shape = tuple(int(value) for value in case_shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError("case_shape must contain positive dimensions.")
    problem_tuple = tuple(problems)
    if len(problem_tuple) != prod(shape):
        raise ValueError("The problem count must equal the product of case_shape.")
    first = problem_tuple[0]
    for problem in problem_tuple[1:]:
        if (
            problem.harmonics.plan.layout.layout_id
            != first.harmonics.plan.layout.layout_id
        ):
            raise ValueError("All cases must share one harmonic layout.")
        if tuple(type(value) for value in problem.elements) != tuple(
            type(value) for value in first.elements
        ):
            raise ValueError("All cases must share one stack topology.")
    prepared = tuple(
        prepare_fourier_modal_maxwell(problem, policy) for problem in problem_tuple
    )
    return _prepared_case_batch(prepared, shape)


def prepare_brillouin_zone_maxwell(
    problem: FourierModalMaxwellProblem,
    rule: PreparedBrillouinZone,
    /,
    *,
    policy: FourierModalSolvePolicy | None = None,
) -> PreparedFourierModalCaseBatch:
    """Prepare one static stack at every wavevector of a Brillouin-zone rule."""
    if rule.lattice_preparation_id != problem.harmonics.preparation_id:
        raise ValueError("The Brillouin rule belongs to a different lattice layout.")
    expected_rule = rule.plan.prepare(problem.harmonics)
    wavevectors = eqx.error_if(
        rule.wavevectors,
        ~jnp.all(rule.wavevectors == expected_rule.wavevectors),
        "The Brillouin rule primitive-vector values do not match the problem lattice.",
    ).reshape((-1, 2))
    problems = tuple(
        FourierModalMaxwellProblem(
            problem.harmonics,
            problem.angular_frequency,
            wavevectors[index],
            problem.superstrate,
            problem.elements,
            problem.substrate,
            numeric_version=f"{problem.numeric_version}:bz:{index}",
        )
        for index in range(wavevectors.shape[0])
    )
    first = prepare_fourier_modal_maxwell(problems[0], policy)
    refresh_spec = FourierModalRefreshSpec(
        tuple("unchanged" for _ in range(problem.layer_count)),
        bloch_wavevector_changed=True,
    )
    prepared = (first,) + tuple(
        refresh_fourier_modal_maxwell(first, value, refresh_spec)
        for value in problems[1:]
    )
    return _prepared_case_batch(prepared, rule.plan.grid_shape)


def solve_fourier_modal_case_batch(
    prepared: PreparedFourierModalCaseBatch,
    excitations: Sequence[FourierModalExcitation],
    /,
) -> FourierModalCaseBatchResult:
    excitation_tuple = tuple(excitations)
    if len(excitation_tuple) != len(prepared.cases):
        raise ValueError("One excitation is required for every prepared case.")
    results = tuple(
        solve_fourier_modal_maxwell(case, excitation)
        for case, excitation in zip(prepared.cases, excitation_tuple, strict=True)
    )

    def stacked(values: tuple[Array, ...]) -> Array:
        value = jnp.stack(values, axis=0)
        return value.reshape(prepared.case_shape + value.shape[1:])

    return FourierModalCaseBatchResult(
        results,
        stacked(tuple(result.right_outgoing for result in results)),
        stacked(tuple(result.left_outgoing for result in results)),
        stacked(tuple(result.left_incoming_power for result in results)),
        stacked(tuple(result.right_incoming_power for result in results)),
        stacked(tuple(result.left_outgoing_power for result in results)),
        stacked(tuple(result.right_outgoing_power for result in results)),
        stacked(tuple(result.net_port_power_into_stack for result in results)),
        stacked(tuple(result.power_audit_residual for result in results)),
        stacked(tuple(result.status for result in results)),
        prepared.case_shape,
        batch_id=prepared.batch_id,
    )


__all__ = [
    "FourierModalCaseBatchResult",
    "PreparedFourierModalCaseBatch",
    "prepare_brillouin_zone_maxwell",
    "prepare_fourier_modal_case_batch",
    "solve_fourier_modal_case_batch",
]

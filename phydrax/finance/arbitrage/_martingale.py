#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.axes as cx

from ..._strict import StrictModule
from ...integration import DiscreteMeasureTarget
from ...optim import ConvexSolvePolicy
from ...stochastic._state_space import AbstractTransitionKernel, StateSpaceStepContext
from ...transport import discrete_problem, DiscreteTransportProblem
from ...transport._costs import GroundCost
from ...transport._martingale import (
    ConvexOrderEvidence,
    MartingaleTransportProblem,
    MartingaleTransportResult,
    solve_martingale_transport,
)
from ...transport.dynamic import SchrodingerBridgeProblem
from ...transport.dynamic._martingale import (
    MartingaleBridgeRefinementEvidence,
    MartingaleSchrodingerBridgeProblem,
    MartingaleSchrodingerBridgeResult,
    MartingaleSchrodingerBridgeSolver,
)
from ..core import FinanceEvidenceBinding, PricingLaw


class NumeraireOptionMarginal(StrictModule):
    """Finite option-implied asset marginal bound to one Q-law and numeraire."""

    asset_values: Array
    probabilities: Array
    numeraire_values: Array
    discount_factor: Array
    pricing_law: PricingLaw
    maturity: float = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)
    currency_code: str = eqx.field(static=True)
    market_snapshot_id: str = eqx.field(static=True)
    option_evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        asset_values: ArrayLike,
        probabilities: ArrayLike,
        numeraire_values: ArrayLike,
        discount_factor: ArrayLike,
        pricing_law: PricingLaw,
        /,
        *,
        maturity: float,
        asset_id: str,
        currency_code: str,
        market_snapshot_id: str,
        option_evidence_id: str,
    ):
        if not isinstance(pricing_law, PricingLaw):
            raise TypeError(
                "pricing_law must be a PricingLaw, not a physical/stress law."
            )
        values = jnp.asarray(asset_values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        probabilities_ = jnp.asarray(probabilities, dtype=float)
        numeraires = jnp.asarray(numeraire_values, dtype=float)
        discount = jnp.asarray(discount_factor, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("asset_values must have shape (atom, asset_coordinate).")
        if (
            probabilities_.shape != (values.shape[0],)
            or numeraires.shape != probabilities_.shape
        ):
            raise ValueError(
                "Probabilities and numeraire values need one value per atom."
            )
        if discount.shape != ():
            raise ValueError("discount_factor must be scalar.")
        if bool(
            jnp.any(~jnp.isfinite(values))
            | jnp.any(~jnp.isfinite(probabilities_))
            | jnp.any(~jnp.isfinite(numeraires))
            | ~jnp.isfinite(discount)
            | jnp.any(probabilities_ < 0.0)
            | jnp.any(numeraires <= 0.0)
            | (discount <= 0.0)
            | ~jnp.isclose(jnp.sum(probabilities_), 1.0, rtol=1e-6, atol=1e-7)
        ):
            raise ValueError(
                "Option marginal values must be finite; probabilities must be "
                "nonnegative and normalized; numeraires/discount must be positive."
            )
        maturity_ = float(maturity)
        if not isfinite(maturity_) or maturity_ < 0.0:
            raise ValueError("maturity must be finite and nonnegative.")
        identifiers = tuple(
            str(value)
            for value in (
                asset_id,
                currency_code,
                market_snapshot_id,
                option_evidence_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Option marginal identities must be nonempty.")
        self.asset_values = values
        self.probabilities = probabilities_
        self.numeraire_values = numeraires
        self.discount_factor = discount
        self.pricing_law = pricing_law
        self.maturity = maturity_
        (
            self.asset_id,
            self.currency_code,
            self.market_snapshot_id,
            self.option_evidence_id,
        ) = identifiers

    @property
    def martingale_coordinates(self) -> Array:
        """Return asset values expressed in units of the declared numeraire."""
        return self.asset_values / self.numeraire_values[:, None]

    @property
    def num_atoms(self) -> int:
        return int(self.probabilities.shape[0])


class FinanceMartingaleTransportProblem(StrictModule):
    """Finance binding for a generic finite martingale transport problem."""

    initial: NumeraireOptionMarginal
    terminal: NumeraireOptionMarginal
    transport: MartingaleTransportProblem
    bound_kind: str = eqx.field(static=True)


class SemiStaticHedgeDual(StrictModule):
    """Static endpoint claims plus predictable holdings from the MOT dual."""

    initial_static_claim: Array
    terminal_static_claim: Array
    dynamic_numeraire_holding: Array
    dual_value: Array
    inequality_violation: Array
    law_id: str = eqx.field(static=True)
    numeraire_id: str = eqx.field(static=True)
    bound_kind: str = eqx.field(static=True)
    valid: Array


class FinanceMartingaleTransportResult(StrictModule):
    """Candidate-only financial MOT result with exact evidence binding."""

    problem: FinanceMartingaleTransportProblem
    transport_result: MartingaleTransportResult
    semi_static_hedge: SemiStaticHedgeDual
    evidence_binding: FinanceEvidenceBinding
    accepted: Array
    candidate_only: bool = eqx.field(static=True, default=True)


class FinanceMartingaleBridgeProblem(StrictModule):
    """Q-law/numeraire binding of option marginals to a martingale bridge."""

    marginals: tuple[NumeraireOptionMarginal, ...]
    bridge: MartingaleSchrodingerBridgeProblem
    law_id: str = eqx.field(static=True)
    numeraire_id: str = eqx.field(static=True)


class IndependentBridgeValidation(StrictModule):
    """Endpoint and conditional-mean evidence from independently sampled paths."""

    initial_probabilities: Array
    terminal_probabilities: Array
    conditional_means: Array
    endpoint_residual: Array
    martingale_defect: Array
    path_count: int = eqx.field(static=True)
    validation_id: str = eqx.field(static=True)
    independence_id: str = eqx.field(static=True)
    training_independence_id: str = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    valid: Array


class FinanceMartingaleBridgeResult(StrictModule):
    """Candidate bridge which cannot pass without independent path validation."""

    problem: FinanceMartingaleBridgeProblem
    bridge_result: MartingaleSchrodingerBridgeResult
    independent_validation: IndependentBridgeValidation | None
    refinement: MartingaleBridgeRefinementEvidence | None
    evidence_binding: FinanceEvidenceBinding
    accepted: Array
    candidate_only: bool = eqx.field(static=True, default=True)


def _same_pricing_semantics(
    initial: NumeraireOptionMarginal, terminal: NumeraireOptionMarginal, /
) -> None:
    if (
        initial.asset_id != terminal.asset_id
        or initial.currency_code != terminal.currency_code
    ):
        raise ValueError("Option marginals must share asset and currency semantics.")
    if initial.maturity >= terminal.maturity:
        raise ValueError("Option marginal maturities must be strictly increasing.")
    left = initial.pricing_law
    right = terminal.pricing_law
    if (
        left.law_id != right.law_id
        or left.measure_id != right.measure_id
        or left.numeraire_id != right.numeraire_id
        or left.collateral_convention_id != right.collateral_convention_id
        or left.factor_layout_id != right.factor_layout_id
        or left.filtration_id != right.filtration_id
    ):
        raise ValueError("Option marginals must share one exact pricing law/numeraire.")


def _target(
    marginal: NumeraireOptionMarginal, atom_axis: str, /
) -> DiscreteMeasureTarget:
    return DiscreteMeasureTarget(
        marginal.asset_values,
        cx.AxisArray(marginal.probabilities, dims=(atom_axis,)),
        axes=atom_axis,
        normalized=True,
        provenance=marginal.option_evidence_id,
    )


def bind_option_marginals_to_martingale_transport(
    initial: NumeraireOptionMarginal,
    terminal: NumeraireOptionMarginal,
    cost: GroundCost,
    /,
    *,
    bound_kind: str,
    mass_tolerance: float = 1e-8,
    constraint_tolerance: float = 1e-7,
) -> FinanceMartingaleTransportProblem:
    """Lower Q-law option marginals to exact martingale constraints."""
    if not isinstance(initial, NumeraireOptionMarginal) or not isinstance(
        terminal, NumeraireOptionMarginal
    ):
        raise TypeError("initial and terminal must be NumeraireOptionMarginal objects.")
    _same_pricing_semantics(initial, terminal)
    kind = str(bound_kind)
    if kind not in ("lower", "upper"):
        raise ValueError("bound_kind must be 'lower' or 'upper'.")
    base: DiscreteTransportProblem = discrete_problem(
        _target(initial, "source_atom"),
        _target(terminal, "target_atom"),
        cost=cost,
        mass_tolerance=mass_tolerance,
    )
    problem = MartingaleTransportProblem(
        base,
        source_coordinates=initial.martingale_coordinates,
        target_coordinates=terminal.martingale_coordinates,
        constraint_tolerance=constraint_tolerance,
    )
    return FinanceMartingaleTransportProblem(initial, terminal, problem, kind)


def semi_static_hedge_dual(
    problem: FinanceMartingaleTransportProblem,
    result: MartingaleTransportResult,
    /,
) -> SemiStaticHedgeDual:
    """Bind an audited generic martingale dual to its finance semantics."""
    if not isinstance(problem, FinanceMartingaleTransportProblem):
        raise TypeError("problem must be a FinanceMartingaleTransportProblem.")
    if not isinstance(result, MartingaleTransportResult):
        raise TypeError(
            "result must be a MartingaleTransportResult; classical OT duals are invalid."
        )
    if result.problem is not problem.transport:
        raise ValueError(
            "Martingale result was produced for a different finance problem."
        )
    dual = result.dual
    return SemiStaticHedgeDual(
        dual.source_potential,
        dual.target_potential,
        dual.dynamic_holding,
        dual.dual_objective,
        dual.maximum_inequality_violation,
        problem.initial.pricing_law.law_id,
        problem.initial.pricing_law.numeraire_id,
        problem.bound_kind,
        result.successful & dual.valid,
    )


def _complete_evidence(binding: FinanceEvidenceBinding, /) -> bool:
    return bool(
        binding.data_evidence_ids
        and binding.model_evidence_ids
        and binding.numerical_evidence_ids
        and binding.use_evidence_ids
    )


def solve_finance_martingale_transport(
    problem: FinanceMartingaleTransportProblem,
    evidence_binding: FinanceEvidenceBinding,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
) -> FinanceMartingaleTransportResult:
    """Solve financial MOT while retaining its candidate-only qualification boundary."""
    if not isinstance(problem, FinanceMartingaleTransportProblem):
        raise TypeError("problem must be a FinanceMartingaleTransportProblem.")
    if not isinstance(evidence_binding, FinanceEvidenceBinding):
        raise TypeError("evidence_binding must be a FinanceEvidenceBinding.")
    transport_result = solve_martingale_transport(problem.transport, policy=policy)
    hedge = semi_static_hedge_dual(problem, transport_result)
    accepted = (
        transport_result.successful & hedge.valid & _complete_evidence(evidence_binding)
    )
    return FinanceMartingaleTransportResult(
        problem,
        transport_result,
        hedge,
        evidence_binding,
        accepted,
        True,
    )


def bind_option_marginals_to_martingale_bridge(
    marginals: Sequence[NumeraireOptionMarginal],
    times: ArrayLike,
    reference: AbstractTransitionKernel,
    context: StateSpaceStepContext,
    /,
    *,
    transition_tolerance: float = 1e-7,
    constraint_tolerance: float = 1e-7,
) -> FinanceMartingaleBridgeProblem:
    """Bind common-support option marginals to a finite martingale bridge."""
    values = tuple(marginals)
    if len(values) != 2 or not all(
        isinstance(value, NumeraireOptionMarginal) for value in values
    ):
        raise TypeError("marginals must contain exactly two endpoint option marginals.")
    for left, right in zip(values, values[1:], strict=True):
        _same_pricing_semantics(left, right)
    grid = jnp.asarray(times, dtype=float)
    if grid.shape != (len(values),):
        raise ValueError("times must contain one node per option marginal.")
    first_support = values[0].asset_values
    if any(
        value.asset_values.shape != first_support.shape
        or not bool(jnp.allclose(value.asset_values, first_support, rtol=0.0, atol=0.0))
        for value in values[1:]
    ):
        raise ValueError(
            "Finite bridge option marginals require one common ordered asset support."
        )
    base = SchrodingerBridgeProblem(
        _target(values[0], "state"),
        _target(values[-1], "state"),
        grid,
        reference,
        context,
        transition_tolerance=transition_tolerance,
    )
    coordinates = jnp.stack(tuple(value.martingale_coordinates for value in values))
    bridge = MartingaleSchrodingerBridgeProblem(
        base,
        martingale_coordinates=coordinates,
        constraint_tolerance=constraint_tolerance,
    )
    law = values[0].pricing_law
    return FinanceMartingaleBridgeProblem(
        values,
        bridge,
        law.law_id,
        law.numeraire_id,
    )


def validate_martingale_bridge_paths(
    problem: FinanceMartingaleBridgeProblem,
    path_state_indices: ArrayLike,
    /,
    *,
    validation_id: str,
    independence_id: str,
    training_independence_id: str,
    tolerance: float,
) -> IndependentBridgeValidation:
    """Audit independently supplied state-index paths against endpoints and martingality."""
    if not isinstance(problem, FinanceMartingaleBridgeProblem):
        raise TypeError("problem must be a FinanceMartingaleBridgeProblem.")
    paths = jnp.asarray(path_state_indices, dtype=jnp.int32)
    states = problem.bridge.bridge.num_states
    nodes = problem.bridge.bridge.num_steps + 1
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] != nodes:
        raise ValueError("Validation paths must have nonempty (path, time) shape.")
    if bool(jnp.any((paths < 0) | (paths >= states))):
        raise ValueError("Validation path state index is outside the bridge support.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    identifiers = tuple(
        str(value) for value in (validation_id, independence_id, training_independence_id)
    )
    if any(not value for value in identifiers):
        raise ValueError("Validation and independence IDs must be nonempty.")
    initial = jnp.bincount(paths[:, 0], length=states) / paths.shape[0]
    terminal = jnp.bincount(paths[:, -1], length=states) / paths.shape[0]
    expected_initial = problem.bridge.bridge.initial_probabilities
    expected_terminal = problem.bridge.bridge.terminal_probabilities
    endpoint_residual = jnp.maximum(
        jnp.sum(jnp.abs(initial - expected_initial)),
        jnp.sum(jnp.abs(terminal - expected_terminal)),
    )
    coordinates = problem.bridge.martingale_coordinates
    conditional_rows = []
    defect = jnp.asarray(0.0)
    for step in range(nodes - 1):
        step_rows = []
        for state in range(states):
            selected = paths[:, step] == state
            count = jnp.sum(selected)
            following = coordinates[step + 1, paths[:, step + 1]]
            mean = jnp.sum(
                jnp.where(selected[:, None], following, 0.0), axis=0
            ) / jnp.maximum(count, 1)
            mean = jnp.where(count > 0, mean, coordinates[step, state])
            step_rows.append(mean)
            defect = jnp.maximum(
                defect,
                jnp.where(
                    count > 0,
                    jnp.max(jnp.abs(mean - coordinates[step, state])),
                    0.0,
                ),
            )
        conditional_rows.append(jnp.stack(tuple(step_rows)))
    conditional = jnp.stack(tuple(conditional_rows))
    independent = identifiers[1] != identifiers[2]
    valid = independent & (endpoint_residual <= tolerance_) & (defect <= tolerance_)
    return IndependentBridgeValidation(
        initial,
        terminal,
        conditional,
        endpoint_residual,
        defect,
        int(paths.shape[0]),
        identifiers[0],
        identifiers[1],
        identifiers[2],
        independent,
        valid,
    )


def solve_finance_martingale_bridge(
    problem: FinanceMartingaleBridgeProblem,
    solver: MartingaleSchrodingerBridgeSolver,
    evidence_binding: FinanceEvidenceBinding,
    /,
    *,
    independent_validation: IndependentBridgeValidation | None,
    refinement: MartingaleBridgeRefinementEvidence | None = None,
) -> FinanceMartingaleBridgeResult:
    """Solve a candidate bridge and fail closed without independent validation."""
    if not isinstance(problem, FinanceMartingaleBridgeProblem):
        raise TypeError("problem must be a FinanceMartingaleBridgeProblem.")
    if not isinstance(solver, MartingaleSchrodingerBridgeSolver):
        raise TypeError("solver must be a MartingaleSchrodingerBridgeSolver.")
    if not isinstance(evidence_binding, FinanceEvidenceBinding):
        raise TypeError("evidence_binding must be a FinanceEvidenceBinding.")
    if independent_validation is not None and not isinstance(
        independent_validation, IndependentBridgeValidation
    ):
        raise TypeError("independent_validation has the wrong evidence type.")
    if refinement is not None and not isinstance(
        refinement, MartingaleBridgeRefinementEvidence
    ):
        raise TypeError("refinement has the wrong evidence type.")
    result = solver(problem.bridge)
    accepted = (
        result.successful
        & (False if independent_validation is None else independent_validation.valid)
        & (True if refinement is None else refinement.accepted)
        & _complete_evidence(evidence_binding)
    )
    return FinanceMartingaleBridgeResult(
        problem,
        result,
        independent_validation,
        refinement,
        evidence_binding,
        accepted,
        True,
    )


def option_marginal_convex_order(
    initial: NumeraireOptionMarginal,
    terminal: NumeraireOptionMarginal,
    /,
    *,
    tolerance: float = 1e-7,
) -> ConvexOrderEvidence:
    """Expose the exact scalar convex-order diagnostic in numeraire units."""
    _same_pricing_semantics(initial, terminal)
    from ...transport._martingale import convex_order_evidence

    return convex_order_evidence(
        initial.probabilities,
        initial.martingale_coordinates,
        terminal.probabilities,
        terminal.martingale_coordinates,
        tolerance=tolerance,
    )


__all__ = [
    "FinanceMartingaleBridgeProblem",
    "FinanceMartingaleBridgeResult",
    "FinanceMartingaleTransportProblem",
    "FinanceMartingaleTransportResult",
    "IndependentBridgeValidation",
    "NumeraireOptionMarginal",
    "SemiStaticHedgeDual",
    "bind_option_marginals_to_martingale_bridge",
    "bind_option_marginals_to_martingale_transport",
    "option_marginal_convex_order",
    "semi_static_hedge_dual",
    "solve_finance_martingale_bridge",
    "solve_finance_martingale_transport",
    "validate_martingale_bridge_paths",
]

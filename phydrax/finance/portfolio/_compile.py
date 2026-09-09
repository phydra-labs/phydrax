#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...optim import (
    Bounds,
    ConicProgram,
    ConvexProgramResult,
    ExponentialCone,
    LinearProgram,
    MixedIntegerProgram,
    MixedIntegerResult,
    NonnegativeCone,
    ProductCone,
    QuadraticProgram,
    SecondOrderCone,
    ZeroCone,
)
from ._objectives import (
    BlackLittermanObjective,
    CVaRObjective,
    DrawdownRiskObjective,
    EVaRObjective,
    FiniteScenarioKellyObjective,
    KLDivergenceRobustObjective,
    MeanVarianceObjective,
    SpectralRiskObjective,
    TrackingErrorObjective,
)
from ._problem import PortfolioProblem


CanonicalPortfolioProgram: TypeAlias = (
    LinearProgram | QuadraticProgram | ConicProgram | MixedIntegerProgram
)
PortfolioProgramKind: TypeAlias = Literal["lp", "qp", "conic", "mip"]


class PortfolioPlan(StrictModule):
    """Deterministic variable layout and native canonical route."""

    variable_slices: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    weight_shape: tuple[int, ...] = eqx.field(static=True)
    canonical_kind: PortfolioProgramKind = eqx.field(static=True)
    objective_kind: str = eqx.field(static=True)
    integer_indices: tuple[int, ...] = eqx.field(static=True)
    binary_indices: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        variable_slices: tuple[tuple[str, int, int], ...],
        weight_shape: tuple[int, ...],
        canonical_kind: PortfolioProgramKind,
        objective_kind: str,
        integer_indices: tuple[int, ...],
        binary_indices: tuple[int, ...],
        structure_id: str,
    ):
        if canonical_kind not in ("lp", "qp", "conic", "mip"):
            raise ValueError("canonical_kind is invalid.")
        slices = tuple(
            (str(name), int(start), int(stop)) for name, start, stop in variable_slices
        )
        if (
            not slices
            or slices[0][0] != "weights"
            or slices[0][1] != 0
            or any(not name or start < 0 or stop <= start for name, start, stop in slices)
        ):
            raise ValueError(
                "variable_slices must begin with one non-empty weight block."
            )
        if any(
            left[2] != right[1] for left, right in zip(slices, slices[1:], strict=False)
        ):
            raise ValueError("variable_slices must be contiguous.")
        shape = tuple(int(size) for size in weight_shape)
        if (
            not shape
            or any(size <= 0 for size in shape)
            or int(np.prod(shape)) != slices[0][2]
        ):
            raise ValueError("weight_shape does not match the weight variable slice.")
        identifier, objective = str(structure_id), str(objective_kind)
        if not identifier or not objective:
            raise ValueError("structure_id and objective_kind must be non-empty.")
        integers = tuple(int(index) for index in integer_indices)
        binaries = tuple(int(index) for index in binary_indices)
        variable_count = slices[-1][2]
        if (
            len(set(integers)) != len(integers)
            or len(set(binaries)) != len(binaries)
            or set(integers) & set(binaries)
            or any(
                index < 0 or index >= variable_count for index in (*integers, *binaries)
            )
        ):
            raise ValueError("Integral variable indices are invalid.")
        self.variable_slices = slices
        self.weight_shape = shape
        self.canonical_kind = canonical_kind
        self.objective_kind = objective
        self.integer_indices, self.binary_indices = integers, binaries
        self.structure_id = identifier

    def slice(self, name: str, /) -> slice:
        matches = tuple(
            (start, stop) for label, start, stop in self.variable_slices if label == name
        )
        if len(matches) != 1:
            raise KeyError(f"Unknown or ambiguous portfolio variable block {name!r}.")
        return slice(*matches[0])

    @property
    def variable_count(self) -> int:
        return self.variable_slices[-1][2]


class PortfolioCompiled(StrictModule):
    """Native canonical program plus immutable financial decode information."""

    program: CanonicalPortfolioProgram
    plan: PortfolioPlan
    weight_scale: Array
    reference_weights: Array
    numeric_version: Array
    problem_id: str = eqx.field(static=True)
    forecast_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        program: CanonicalPortfolioProgram,
        plan: PortfolioPlan,
        weight_scale: ArrayLike,
        reference_weights: ArrayLike,
        /,
        *,
        problem_id: str,
        forecast_law_id: str,
        numeric_version: int = 0,
    ):
        if not isinstance(
            program, (LinearProgram, QuadraticProgram, ConicProgram, MixedIntegerProgram)
        ):
            raise TypeError("program must be a native canonical optimization program.")
        if not isinstance(plan, PortfolioPlan):
            raise TypeError("plan must be a PortfolioPlan.")
        route = (
            "mip"
            if isinstance(program, MixedIntegerProgram)
            else "conic"
            if isinstance(program, ConicProgram)
            else "qp"
            if isinstance(program, QuadraticProgram)
            else "lp"
        )
        if plan.canonical_kind != route:
            raise ValueError("The native program kind does not match the portfolio plan.")
        scale = jnp.asarray(weight_scale)
        reference = jnp.asarray(reference_weights, dtype=scale.dtype)
        expected = (plan.variable_slices[0][2],)
        program_variables = (
            program.relaxation.num_variables
            if isinstance(program, MixedIntegerProgram)
            else program.num_variables
        )
        if program_variables != plan.variable_count:
            raise ValueError(
                "The native program and portfolio plan variable counts differ."
            )
        if scale.shape != expected or reference.shape != expected:
            raise ValueError(
                "weight_scale and reference_weights must match the flat weight block."
            )
        if (
            not np.all(np.isfinite(np.asarray(scale)))
            or not np.all(np.isfinite(np.asarray(reference)))
            or np.any(np.asarray(scale) <= 0.0)
        ):
            raise ValueError(
                "weight_scale must be positive and decode arrays must be finite."
            )
        version = int(numeric_version)
        problem_identifier, law_identifier = str(problem_id), str(forecast_law_id)
        if version < 0 or not problem_identifier or not law_identifier:
            raise ValueError("Compiled provenance is invalid.")
        self.program, self.plan = program, plan
        self.weight_scale, self.reference_weights = scale, reference
        self.numeric_version = jnp.asarray(version, dtype=jnp.int32)
        self.problem_id, self.forecast_law_id = problem_identifier, law_identifier


class PortfolioDecision(StrictModule):
    """Decoded financial decision, independent of the optimizer certificate."""

    weights: Array
    trades: Array
    lot_counts: Array
    active: Array
    fees_activated: Array
    canonical_primal: Array
    objective: Array
    structure_id: str = eqx.field(static=True)


class PortfolioOptimizerCertificate(StrictModule):
    """Independent solver/canonical feasibility evidence for a decoded decision."""

    primal_residual: Array
    dual_residual: Array
    complementarity_gap: Array
    certified: Array
    status: Array
    backend: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


class PortfolioResult(StrictModule):
    """Optimization output only; realized portfolio performance is a separate type."""

    decision: PortfolioDecision
    certificate: PortfolioOptimizerCertificate
    numeric_version: Array
    problem_id: str = eqx.field(static=True)
    forecast_law_id: str = eqx.field(static=True)


class _Layout:
    def __init__(self, weight_count: int):
        self.cursor = 0
        self.slices: dict[str, slice] = {}
        self.add("weights", weight_count)

    def add(self, name: str, size: int) -> slice:
        if name in self.slices or size <= 0:
            raise ValueError("Invalid canonical variable block.")
        selected = slice(self.cursor, self.cursor + int(size))
        self.slices[name] = selected
        self.cursor = selected.stop
        return selected

    def records(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (name, block.start, block.stop) for name, block in self.slices.items()
        )


def _objective_auxiliaries(problem: PortfolioProblem, layout: _Layout) -> None:
    objective = problem.objective
    scenarios = problem.forecast.scenario_count
    if isinstance(objective, FiniteScenarioKellyObjective):
        layout.add("log_growth", scenarios)
    elif isinstance(objective, CVaRObjective):
        layout.add("cvar_threshold", 1)
        layout.add("cvar_excess", scenarios)
    elif isinstance(objective, (EVaRObjective, KLDivergenceRobustObjective)):
        layout.add("entropy_location", 1)
        layout.add("entropy_scale", 1)
        layout.add("entropy_perspective", scenarios)
    elif isinstance(objective, SpectralRiskObjective):
        atoms = int(objective.confidences.shape[0])
        layout.add("spectral_thresholds", atoms)
        layout.add("spectral_excess", atoms * scenarios)
    elif isinstance(objective, DrawdownRiskObjective):
        returns = problem.forecast.scenario_returns
        entries = int(returns.shape[0] * returns.shape[1])
        layout.add("running_peak", entries)
        layout.add("drawdown", entries)
        layout.add("maximum_drawdown", int(returns.shape[0]))


def _weight_topology(problem: PortfolioProblem) -> tuple[tuple[int, ...], int, int]:
    tree = problem.constraints.scenario_tree
    assets = problem.forecast.asset_count
    if tree is None:
        return (assets,), 1, assets
    shape = (tree.scenario_count, tree.stage_count, assets)
    return shape, tree.scenario_count * tree.stage_count, int(np.prod(shape))


def _path_return_map(problem: PortfolioProblem, scale: np.ndarray, /) -> np.ndarray:
    returns = np.asarray(problem.forecast.scenario_returns)
    assets = problem.forecast.asset_count
    tree = problem.constraints.scenario_tree
    if tree is None:
        return returns * scale[None, :]
    scenarios, stages, _ = returns.shape
    result = np.zeros((scenarios, scenarios * stages * assets), dtype=returns.dtype)
    asset_scale = np.asarray(problem.scaling.weight_scale)
    for scenario in range(scenarios):
        for stage in range(stages):
            start = (scenario * stages + stage) * assets
            result[scenario, start : start + assets] = (
                returns[scenario, stage] * asset_scale
            )
    return result


def compile_portfolio_problem(problem: PortfolioProblem, /) -> PortfolioCompiled:
    """Compile one portfolio definition to a deterministic native LP/QP/conic/MIP."""

    if not isinstance(problem, PortfolioProblem):
        raise TypeError("problem must be a PortfolioProblem.")
    forecast, constraints, objective = (
        problem.forecast,
        problem.constraints,
        problem.objective,
    )
    assets = forecast.asset_count
    weight_shape, copies, weight_count = _weight_topology(problem)
    layout = _Layout(weight_count)
    gross = (
        layout.add("gross", weight_count) if constraints.gross_limit is not None else None
    )
    needs_trade = (
        constraints.turnover_limit is not None or constraints.fixed_fees is not None
    )
    trade_abs = layout.add("trade_absolute", weight_count) if needs_trade else None
    lot_counts = (
        layout.add("lot_counts", weight_count)
        if constraints.lot_sizes is not None
        else None
    )
    active = (
        layout.add("active", weight_count)
        if constraints.maximum_cardinality is not None
        else None
    )
    fee_active = (
        layout.add("fee_active", weight_count)
        if constraints.fixed_fees is not None
        else None
    )
    _objective_auxiliaries(problem, layout)
    variables = layout.cursor
    dtype = np.asarray(forecast.expected_returns).dtype
    linear = np.zeros((variables,), dtype=dtype)
    quadratic = np.zeros((variables, variables), dtype=dtype)
    equalities: list[np.ndarray] = []
    equality_rhs: list[float] = []
    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    cone_blocks: list[tuple[np.ndarray, np.ndarray, Any]] = []

    def eq(row: np.ndarray, rhs: float) -> None:
        equalities.append(np.asarray(row, dtype=dtype))
        equality_rhs.append(float(rhs))

    def le(row: np.ndarray, rhs: float) -> None:
        inequalities.append(np.asarray(row, dtype=dtype))
        inequality_rhs.append(float(rhs))

    asset_scale = np.asarray(problem.scaling.weight_scale, dtype=dtype)
    flat_scale = np.tile(asset_scale, copies)
    lower_asset = (
        np.full((assets,), -np.inf, dtype=dtype)
        if constraints.lower_weights is None
        else np.asarray(constraints.lower_weights, dtype=dtype)
    )
    upper_asset = (
        np.full((assets,), np.inf, dtype=dtype)
        if constraints.upper_weights is None
        else np.asarray(constraints.upper_weights, dtype=dtype)
    )
    lower_physical = np.tile(lower_asset, copies)
    upper_physical = np.tile(upper_asset, copies)
    lower = np.full((variables,), -np.inf, dtype=dtype)
    upper = np.full((variables,), np.inf, dtype=dtype)
    lower[layout.slices["weights"]] = lower_physical / flat_scale
    upper[layout.slices["weights"]] = upper_physical / flat_scale
    reference_asset = (
        np.zeros((assets,), dtype=dtype)
        if problem.current_weights is None
        else np.asarray(problem.current_weights, dtype=dtype)
    )
    reference = np.tile(reference_asset, copies)

    for copy in range(copies):
        start = copy * assets
        row = np.zeros((variables,), dtype=dtype)
        row[start : start + assets] = asset_scale
        eq(row, constraints.budget)
        if constraints.linear_matrix is not None:
            matrix = np.asarray(constraints.linear_matrix, dtype=dtype)
            lows = np.asarray(constraints.linear_lower)
            highs = np.asarray(constraints.linear_upper)
            for index in range(matrix.shape[0]):
                row = np.zeros((variables,), dtype=dtype)
                row[start : start + assets] = matrix[index] * asset_scale
                if (
                    np.isfinite(lows[index])
                    and np.isfinite(highs[index])
                    and lows[index] == highs[index]
                ):
                    eq(row, float(lows[index]))
                else:
                    if np.isfinite(highs[index]):
                        le(row, float(highs[index]))
                    if np.isfinite(lows[index]):
                        le(-row, float(-lows[index]))

    tree = constraints.scenario_tree
    if tree is not None:
        history = np.asarray(tree.history_labels)
        stages = tree.stage_count
        for stage in range(stages):
            for label in np.unique(history[:, stage]):
                members = np.flatnonzero(history[:, stage] == label)
                representative = int(members[0])
                for scenario in members[1:]:
                    for asset in range(assets):
                        row = np.zeros((variables,), dtype=dtype)
                        left = (int(scenario) * stages + stage) * assets + asset
                        right = (representative * stages + stage) * assets + asset
                        row[left], row[right] = asset_scale[asset], -asset_scale[asset]
                        eq(row, 0.0)

    if gross is not None:
        lower[gross] = 0.0
        for index in range(weight_count):
            for sign in (-1.0, 1.0):
                row = np.zeros((variables,), dtype=dtype)
                row[index] = sign * flat_scale[index]
                row[gross.start + index] = -1.0
                le(row, 0.0)
        for copy in range(copies):
            row = np.zeros((variables,), dtype=dtype)
            begin = gross.start + copy * assets
            row[begin : begin + assets] = 1.0
            le(row, constraints.gross_limit)

    if trade_abs is not None:
        lower[trade_abs] = 0.0
        stages = 1 if tree is None else tree.stage_count
        for index in range(weight_count):
            copy = index // assets
            stage = 0 if tree is None else copy % stages
            row = np.zeros((variables,), dtype=dtype)
            row[index] = flat_scale[index]
            rhs = reference[index]
            if stage > 0:
                previous = index - assets
                row[previous] = -flat_scale[previous]
                rhs = 0.0
            row[trade_abs.start + index] = -1.0
            le(row, rhs)
            le(_opposite_trade_row(row, trade_abs.start + index), -rhs)
        if constraints.turnover_limit is not None:
            for copy in range(copies):
                row = np.zeros((variables,), dtype=dtype)
                begin = trade_abs.start + copy * assets
                row[begin : begin + assets] = 1.0
                le(row, constraints.turnover_limit)

    integer_indices: list[int] = []
    binary_indices: list[int] = []
    if lot_counts is not None:
        if not np.all(np.isfinite(lower_physical)) or not np.all(
            np.isfinite(upper_physical)
        ):
            raise ValueError("Lot constraints require finite weight bounds.")
        lots = np.tile(np.asarray(constraints.lot_sizes, dtype=dtype), copies)
        lower[lot_counts] = np.ceil(lower_physical / lots)
        upper[lot_counts] = np.floor(upper_physical / lots)
        if np.any(lower[lot_counts] > upper[lot_counts]):
            raise ValueError("Weight bounds contain no feasible lot multiple.")
        for index in range(weight_count):
            row = np.zeros((variables,), dtype=dtype)
            row[index] = flat_scale[index]
            row[lot_counts.start + index] = -lots[index]
            eq(row, 0.0)
        integer_indices.extend(range(lot_counts.start, lot_counts.stop))

    if active is not None:
        if not np.all(np.isfinite(lower_physical)) or not np.all(
            np.isfinite(upper_physical)
        ):
            raise ValueError("Cardinality constraints require finite weight bounds.")
        lower[active], upper[active] = 0.0, 1.0
        magnitude = np.maximum(np.abs(lower_physical), np.abs(upper_physical))
        for index in range(weight_count):
            for sign in (-1.0, 1.0):
                row = np.zeros((variables,), dtype=dtype)
                row[index] = sign * flat_scale[index]
                row[active.start + index] = -magnitude[index]
                le(row, 0.0)
        for copy in range(copies):
            row = np.zeros((variables,), dtype=dtype)
            begin = active.start + copy * assets
            row[begin : begin + assets] = 1.0
            le(row, constraints.maximum_cardinality)
        binary_indices.extend(range(active.start, active.stop))

    if fee_active is not None:
        if not np.all(np.isfinite(lower_physical)) or not np.all(
            np.isfinite(upper_physical)
        ):
            raise ValueError("Fixed fees require finite weight bounds.")
        lower[fee_active], upper[fee_active] = 0.0, 1.0
        stages = 1 if tree is None else tree.stage_count
        probabilities = (
            np.ones((1,), dtype=dtype)
            if tree is None
            else np.asarray(forecast.scenario_probabilities, dtype=dtype)
        )
        fees = np.tile(np.asarray(constraints.fixed_fees, dtype=dtype), copies)
        for index in range(weight_count):
            copy = index // assets
            stage = 0 if tree is None else copy % stages
            scenario = 0 if tree is None else copy // stages
            movement = (
                max(
                    abs(lower_physical[index] - reference[index]),
                    abs(upper_physical[index] - reference[index]),
                )
                if stage == 0
                else upper_physical[index] - lower_physical[index]
            )
            row = np.zeros((variables,), dtype=dtype)
            row[trade_abs.start + index] = 1.0
            row[fee_active.start + index] = -movement
            le(row, 0.0)
            linear[fee_active.start + index] += fees[index] * probabilities[scenario]
        binary_indices.extend(range(fee_active.start, fee_active.stop))

    scale_diagonal = np.diag(asset_scale)
    if tree is None:
        covariance_blocks = ((0, 1.0),)
    else:
        probabilities = np.asarray(forecast.scenario_probabilities, dtype=dtype)
        covariance_blocks = tuple(
            (scenario * tree.stage_count + stage, probabilities[scenario])
            for scenario in range(tree.scenario_count)
            for stage in range(tree.stage_count)
        )
    mean_map = None
    path_map = None
    if forecast.scenario_returns is not None:
        path_map = _path_return_map(problem, flat_scale)
        mean_map = np.asarray(forecast.scenario_probabilities) @ path_map
    else:
        mean_map = np.tile(np.asarray(forecast.expected_returns) * asset_scale, copies)
        if copies > 1:
            mean_map = mean_map / copies

    if isinstance(
        objective,
        (MeanVarianceObjective, TrackingErrorObjective, BlackLittermanObjective),
    ):
        covariance = np.asarray(forecast.covariance, dtype=dtype)
        expected = np.asarray(forecast.expected_returns, dtype=dtype)
        if isinstance(objective, BlackLittermanObjective):
            posterior_mean, posterior_covariance = objective.posterior(
                forecast.covariance
            )
            covariance = np.asarray(posterior_covariance, dtype=dtype)
            expected = np.asarray(posterior_mean, dtype=dtype)
            aversion, reward = objective.risk_aversion, 1.0
            benchmark = None
        elif isinstance(objective, TrackingErrorObjective):
            aversion, reward = objective.tracking_aversion, objective.return_weight
            benchmark = np.asarray(objective.benchmark_weights, dtype=dtype)
            if benchmark.shape != (assets,):
                raise ValueError("benchmark_weights must match the asset universe.")
        else:
            aversion, reward = objective.risk_aversion, objective.return_weight
            benchmark = None
        transformed_covariance = scale_diagonal @ covariance @ scale_diagonal
        for copy, mass in covariance_blocks:
            block = slice(copy * assets, (copy + 1) * assets)
            quadratic[block, block] += 2.0 * aversion * mass * transformed_covariance
            local_mean = expected * asset_scale
            if tree is not None:
                scenario = copy // tree.stage_count
                local_mean = (
                    np.asarray(forecast.scenario_returns)[
                        scenario, copy % tree.stage_count
                    ]
                    * asset_scale
                    * np.asarray(forecast.scenario_probabilities)[scenario]
                )
            linear[block] -= reward * local_mean
            if benchmark is not None:
                linear[block] -= (
                    2.0 * aversion * mass * (scale_diagonal @ covariance @ benchmark)
                )
    elif isinstance(objective, FiniteScenarioKellyObjective):
        logs = layout.slices["log_growth"]
        probabilities = np.asarray(forecast.scenario_probabilities, dtype=dtype)
        linear[logs] = -probabilities
        for scenario in range(forecast.scenario_count):
            rows = np.zeros((3, variables), dtype=dtype)
            rhs = np.asarray((0.0, 1.0, objective.initial_wealth), dtype=dtype)
            rows[0, logs.start + scenario] = -1.0
            rows[2, :weight_count] = -path_map[scenario]
            cone_blocks.append((rows, rhs, ExponentialCone()))
            row = np.zeros((variables,), dtype=dtype)
            row[:weight_count] = -path_map[scenario]
            le(row, objective.initial_wealth - objective.bankruptcy_floor)
    elif isinstance(objective, CVaRObjective):
        eta = layout.slices["cvar_threshold"].start
        excess = layout.slices["cvar_excess"]
        probabilities = np.asarray(forecast.scenario_probabilities, dtype=dtype)
        lower[excess] = 0.0
        linear[:weight_count] -= objective.return_weight * mean_map
        linear[eta] = objective.risk_weight
        linear[excess] = (
            objective.risk_weight * probabilities / (1.0 - objective.confidence)
        )
        for scenario in range(forecast.scenario_count):
            row = np.zeros((variables,), dtype=dtype)
            row[:weight_count] = -path_map[scenario]
            row[eta] = -1.0
            row[excess.start + scenario] = -1.0
            le(row, 0.0)
    elif isinstance(objective, (EVaRObjective, KLDivergenceRobustObjective)):
        eta = layout.slices["entropy_location"].start
        tau = layout.slices["entropy_scale"].start
        perspective = layout.slices["entropy_perspective"]
        probabilities = np.asarray(forecast.scenario_probabilities, dtype=dtype)
        radius = (
            objective.relative_entropy_radius
            if isinstance(objective, EVaRObjective)
            else objective.radius
        )
        lower[tau] = 0.0
        lower[perspective] = 0.0
        linear[:weight_count] -= objective.return_weight * mean_map
        linear[eta] = objective.risk_weight
        linear[tau] = objective.risk_weight * radius
        for scenario in range(forecast.scenario_count):
            rows = np.zeros((3, variables), dtype=dtype)
            rows[0, :weight_count] = path_map[scenario]
            rows[0, eta] = 1.0
            rows[1, tau] = -1.0
            rows[2, perspective.start + scenario] = -1.0
            cone_blocks.append((rows, np.zeros((3,), dtype=dtype), ExponentialCone()))
        row = np.zeros((variables,), dtype=dtype)
        row[perspective] = probabilities
        row[tau] = -1.0
        le(row, 0.0)
    elif isinstance(objective, SpectralRiskObjective):
        thresholds = layout.slices["spectral_thresholds"]
        excess = layout.slices["spectral_excess"]
        lower[excess] = 0.0
        probabilities = np.asarray(forecast.scenario_probabilities, dtype=dtype)
        linear[:weight_count] -= objective.return_weight * mean_map
        scenarios = forecast.scenario_count
        for atom, (confidence, mass) in enumerate(
            zip(
                np.asarray(objective.confidences),
                np.asarray(objective.weights),
                strict=True,
            )
        ):
            linear[thresholds.start + atom] = objective.risk_weight * mass
            atom_excess = slice(
                excess.start + atom * scenarios, excess.start + (atom + 1) * scenarios
            )
            linear[atom_excess] = (
                objective.risk_weight * mass * probabilities / (1.0 - confidence)
            )
            for scenario in range(scenarios):
                row = np.zeros((variables,), dtype=dtype)
                row[:weight_count] = -path_map[scenario]
                row[thresholds.start + atom] = -1.0
                row[atom_excess.start + scenario] = -1.0
                le(row, 0.0)
    elif isinstance(objective, DrawdownRiskObjective):
        peaks = layout.slices["running_peak"]
        drawdowns = layout.slices["drawdown"]
        maxima = layout.slices["maximum_drawdown"]
        lower[drawdowns], lower[maxima] = 0.0, 0.0
        probabilities = np.asarray(forecast.scenario_probabilities, dtype=dtype)
        linear[:weight_count] -= objective.return_weight * mean_map
        linear[maxima] = objective.risk_weight * probabilities
        returns = np.asarray(forecast.scenario_returns, dtype=dtype)
        stages = returns.shape[1]
        for scenario in range(returns.shape[0]):
            cumulative = np.zeros((variables,), dtype=dtype)
            for stage in range(stages):
                start = (scenario * stages + stage) * assets
                cumulative[start : start + assets] += (
                    returns[scenario, stage] * asset_scale
                )
                slot = scenario * stages + stage
                peak = peaks.start + slot
                draw = drawdowns.start + slot
                row = cumulative.copy()
                row[peak] -= 1.0
                le(row, 0.0)
                row = np.zeros((variables,), dtype=dtype)
                row[peak] = -1.0
                le(row, 0.0)
                if stage > 0:
                    row = np.zeros((variables,), dtype=dtype)
                    row[peaks.start + slot - 1] = 1.0
                    row[peak] = -1.0
                    le(row, 0.0)
                row = -cumulative.copy()
                row[peak] += 1.0
                row[draw] -= 1.0
                le(row, 0.0)
                row = np.zeros((variables,), dtype=dtype)
                row[draw] = 1.0
                row[maxima.start + scenario] = -1.0
                le(row, 0.0)

    for copy in range(copies):
        start = copy * assets
        for robust in constraints.robust:
            factors = int(robust.factor_loading.shape[1])
            rows = np.zeros((factors + 1, variables), dtype=dtype)
            rhs = np.zeros((factors + 1,), dtype=dtype)
            rows[0, start : start + assets] = np.asarray(robust.nominal) * asset_scale
            rhs[0] = robust.bound
            rows[1:, start : start + assets] = (
                -robust.radius
                * np.asarray(robust.factor_loading).T
                * asset_scale[None, :]
            )
            cone_blocks.append((rows, rhs, SecondOrderCone(factors + 1)))

    linear *= problem.scaling.objective_scale
    quadratic *= problem.scaling.objective_scale
    equality_matrix = (
        np.stack(equalities) if equalities else np.empty((0, variables), dtype=dtype)
    )
    equality_values = np.asarray(equality_rhs, dtype=dtype)
    inequality_matrix = (
        np.stack(inequalities) if inequalities else np.empty((0, variables), dtype=dtype)
    )
    inequality_values = np.asarray(inequality_rhs, dtype=dtype)
    equality_matrix *= problem.scaling.constraint_scale
    equality_values *= problem.scaling.constraint_scale
    inequality_matrix *= problem.scaling.constraint_scale
    inequality_values *= problem.scaling.constraint_scale
    bounds = Bounds(jnp.asarray(lower), jnp.asarray(upper))
    uses_cones = bool(cone_blocks)
    has_quadratic = bool(np.any(quadratic != 0.0))
    if uses_cones:
        matrices = [equality_matrix, inequality_matrix]
        right_sides = [equality_values, inequality_values]
        cones: list[Any] = [
            ZeroCone(equality_matrix.shape[0]),
            NonnegativeCone(inequality_matrix.shape[0]),
        ]
        for matrix, rhs, cone in cone_blocks:
            matrices.append(matrix)
            right_sides.append(rhs)
            cones.append(cone)
        relaxation: LinearProgram | QuadraticProgram | ConicProgram = ConicProgram(
            jnp.asarray(quadratic) if has_quadratic else None,
            jnp.asarray(linear),
            jnp.asarray(np.concatenate(matrices, axis=0)),
            jnp.asarray(np.concatenate(right_sides, axis=0)),
            ProductCone(tuple(cones)),
            bounds=bounds,
            problem_id=problem.problem_id,
            convexity_evidence="construction",
        )
        base_kind: PortfolioProgramKind = "conic"
    elif has_quadratic:
        relaxation = QuadraticProgram(
            jnp.asarray(quadratic),
            jnp.asarray(linear),
            equality_matrix=jnp.asarray(equality_matrix),
            equality_rhs=jnp.asarray(equality_values),
            inequality_matrix=jnp.asarray(inequality_matrix),
            inequality_rhs=jnp.asarray(inequality_values),
            bounds=bounds,
            problem_id=problem.problem_id,
            convexity_evidence="construction",
        )
        base_kind = "qp"
    else:
        relaxation = LinearProgram(
            jnp.asarray(linear),
            equality_matrix=jnp.asarray(equality_matrix),
            equality_rhs=jnp.asarray(equality_values),
            inequality_matrix=jnp.asarray(inequality_matrix),
            inequality_rhs=jnp.asarray(inequality_values),
            bounds=bounds,
            problem_id=problem.problem_id,
        )
        base_kind = "lp"
    if integer_indices or binary_indices:
        program: CanonicalPortfolioProgram = MixedIntegerProgram(
            relaxation,
            integer_indices=tuple(integer_indices),
            binary_indices=tuple(binary_indices),
            program_id=problem.problem_id,
        )
        kind: PortfolioProgramKind = "mip"
    else:
        program = relaxation
        kind = base_kind
    structure = canonical_fingerprint(
        {
            "kind": "portfolio-program",
            "problem_id": problem.problem_id,
            "asset_ids": list(forecast.asset_ids),
            "physical_law": {
                "law_id": forecast.law.law_id,
                "provenance": forecast.law.provenance,
                "factor_layout_id": forecast.law.factor_layout_id,
                "filtration_id": forecast.law.filtration_id,
            },
            "objective": type(objective).__name__,
            "weight_shape": list(weight_shape),
            "layout": [list(record) for record in layout.records()],
            "canonical": kind,
            "lower_roles": np.isfinite(lower).tolist(),
            "upper_roles": np.isfinite(upper).tolist(),
            "linear_pattern": (np.asarray(constraints.linear_matrix) != 0.0).tolist()
            if constraints.linear_matrix is not None
            else None,
            "robust_patterns": [
                (np.asarray(item.factor_loading) != 0.0).tolist()
                for item in constraints.robust
            ],
            "robust_ids": [item.constraint_id for item in constraints.robust],
            "tree_id": tree.tree_id if tree is not None else None,
            "tree_histories": np.asarray(tree.history_labels).tolist()
            if tree is not None
            else None,
            "integers": integer_indices,
            "binaries": binary_indices,
        }
    )
    plan = PortfolioPlan(
        variable_slices=layout.records(),
        weight_shape=weight_shape,
        canonical_kind=kind,
        objective_kind=type(objective).__name__,
        integer_indices=tuple(integer_indices),
        binary_indices=tuple(binary_indices),
        structure_id=structure,
    )
    return PortfolioCompiled(
        program,
        plan,
        jnp.asarray(flat_scale),
        jnp.asarray(reference),
        problem_id=problem.problem_id,
        forecast_law_id=forecast.law_id,
    )


def _opposite_trade_row(row: np.ndarray, auxiliary_index: int, /) -> np.ndarray:
    result = -row.copy()
    result[auxiliary_index] = -1.0
    return result


def refresh_portfolio_compilation(
    compiled: PortfolioCompiled,
    problem: PortfolioProblem,
    /,
) -> PortfolioCompiled:
    """Refresh numerical data while refusing every structural portfolio change."""

    if not isinstance(compiled, PortfolioCompiled) or not isinstance(
        problem, PortfolioProblem
    ):
        raise TypeError("compiled and problem have incorrect types.")
    candidate = compile_portfolio_problem(problem)
    if candidate.plan.structure_id != compiled.plan.structure_id:
        raise ValueError("Numeric refresh cannot change portfolio structure.")
    return PortfolioCompiled(
        candidate.program,
        compiled.plan,
        candidate.weight_scale,
        candidate.reference_weights,
        problem_id=problem.problem_id,
        forecast_law_id=problem.forecast.law_id,
        numeric_version=int(np.asarray(compiled.numeric_version)) + 1,
    )


def _relaxation(program: CanonicalPortfolioProgram, /):
    return program.relaxation if isinstance(program, MixedIntegerProgram) else program


def _primal_feasibility(compiled: PortfolioCompiled, primal: np.ndarray, /) -> float:
    program = _relaxation(compiled.program)
    violations: list[float] = []
    lower, upper = np.asarray(program.lower_bounds), np.asarray(program.upper_bounds)
    violations.append(float(np.max(np.maximum(lower - primal, 0.0))))
    violations.append(float(np.max(np.maximum(primal - upper, 0.0))))
    if isinstance(program, (LinearProgram, QuadraticProgram)):
        if program.equality_matrix.shape[0]:
            violations.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(program.equality_matrix) @ primal
                            - np.asarray(program.equality_rhs)
                        )
                    )
                )
            )
        if program.inequality_matrix.shape[0]:
            violations.append(
                float(
                    np.max(
                        np.maximum(
                            np.asarray(program.inequality_matrix) @ primal
                            - np.asarray(program.inequality_rhs),
                            0.0,
                        )
                    )
                )
            )
    else:
        slack = jnp.asarray(program.constraint_rhs) - jnp.asarray(
            program.constraint_matrix
        ) @ jnp.asarray(primal)
        projected = program.cone.project(slack)
        violations.append(float(np.max(np.abs(np.asarray(slack - projected)))))
    if compiled.plan.integer_indices:
        values = primal[np.asarray(compiled.plan.integer_indices)]
        violations.append(float(np.max(np.abs(values - np.rint(values)))))
    if compiled.plan.binary_indices:
        values = primal[np.asarray(compiled.plan.binary_indices)]
        violations.append(float(np.max(np.minimum(np.abs(values), np.abs(values - 1.0)))))
    return max(violations, default=0.0)


def decode_portfolio_decision(
    compiled: PortfolioCompiled,
    primal: ArrayLike,
    /,
    *,
    structure_id: str,
    feasibility_tolerance: float = 1e-6,
) -> PortfolioDecision:
    """Decode only a finite, structurally matched, independently feasible primal."""

    if not isinstance(compiled, PortfolioCompiled):
        raise TypeError("compiled must be a PortfolioCompiled.")
    if str(structure_id) != compiled.plan.structure_id:
        raise ValueError("The primal decode structure does not match the portfolio plan.")
    value = jnp.asarray(primal)
    if value.shape != (compiled.plan.variable_count,) or not bool(
        np.all(np.isfinite(np.asarray(value)))
    ):
        raise ValueError(
            "primal must be one finite vector matching the compiled variable count."
        )
    tolerance = float(feasibility_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("feasibility_tolerance must be finite and positive.")
    residual = _primal_feasibility(compiled, np.asarray(value))
    if residual > tolerance:
        raise ValueError(
            f"Decoded primal violates the portfolio program by {residual:g}."
        )
    weights_flat = value[compiled.plan.slice("weights")] * compiled.weight_scale
    weights = jnp.reshape(weights_flat, compiled.plan.weight_shape)
    if len(compiled.plan.weight_shape) == 1:
        trades = jnp.reshape(
            weights_flat - compiled.reference_weights,
            compiled.plan.weight_shape,
        )
    else:
        reference = jnp.reshape(
            compiled.reference_weights,
            compiled.plan.weight_shape,
        )
        trades = jnp.concatenate(
            (
                weights[:, :1, :] - reference[:, :1, :],
                weights[:, 1:, :] - weights[:, :-1, :],
            ),
            axis=1,
        )
    empty = jnp.empty((0,), dtype=value.dtype)
    lots = (
        value[compiled.plan.slice("lot_counts")]
        if any(name == "lot_counts" for name, _, _ in compiled.plan.variable_slices)
        else empty
    )
    active = (
        value[compiled.plan.slice("active")]
        if any(name == "active" for name, _, _ in compiled.plan.variable_slices)
        else empty
    )
    fees = (
        value[compiled.plan.slice("fee_active")]
        if any(name == "fee_active" for name, _, _ in compiled.plan.variable_slices)
        else empty
    )
    relaxation = _relaxation(compiled.program)
    linear = jnp.asarray(relaxation.linear)
    if isinstance(relaxation, LinearProgram) or (
        isinstance(relaxation, ConicProgram) and relaxation.quadratic is None
    ):
        objective = linear @ value
    else:
        objective = (
            0.5 * value @ jnp.asarray(relaxation.quadratic) @ value + linear @ value
        )
    return PortfolioDecision(
        weights=weights,
        trades=trades,
        lot_counts=jnp.rint(lots).astype(jnp.int32),
        active=jnp.rint(active).astype(jnp.int32),
        fees_activated=jnp.rint(fees).astype(jnp.int32),
        canonical_primal=value,
        objective=objective,
        structure_id=compiled.plan.structure_id,
    )


def portfolio_result_from_native(
    compiled: PortfolioCompiled,
    result: ConvexProgramResult | MixedIntegerResult,
    /,
    *,
    feasibility_tolerance: float = 1e-6,
) -> PortfolioResult:
    """Pair a decoded decision with separate native and independent certificate evidence."""

    if isinstance(result, ConvexProgramResult):
        primal = result.primal
        status = result.status
        dual = result.dual_residual_norm
        gap = result.complementarity_gap
        backend = result.backend
        native_success = result.successful
    elif isinstance(result, MixedIntegerResult):
        primal = result.primal
        status = result.status
        dual = (
            jnp.asarray(jnp.inf)
            if result.relaxation_result is None
            else result.relaxation_result.dual_residual_norm
        )
        gap = result.absolute_gap
        backend = "native-branch-and-bound"
        native_success = result.successful
    else:
        raise TypeError("result must be a native convex or mixed-integer result.")
    decision = decode_portfolio_decision(
        compiled,
        primal,
        structure_id=compiled.plan.structure_id,
        feasibility_tolerance=feasibility_tolerance,
    )
    residual = jnp.asarray(_primal_feasibility(compiled, np.asarray(primal)))
    certified = jnp.asarray(native_success) & (residual <= feasibility_tolerance)
    certificate = PortfolioOptimizerCertificate(
        primal_residual=residual,
        dual_residual=jnp.asarray(dual),
        complementarity_gap=jnp.asarray(gap),
        certified=certified,
        status=jnp.asarray(status, dtype=jnp.int32),
        backend=backend,
        structure_id=compiled.plan.structure_id,
    )
    return PortfolioResult(
        decision=decision,
        certificate=certificate,
        numeric_version=compiled.numeric_version,
        problem_id=compiled.problem_id,
        forecast_law_id=compiled.forecast_law_id,
    )


__all__ = [
    "CanonicalPortfolioProgram",
    "PortfolioCompiled",
    "PortfolioDecision",
    "PortfolioOptimizerCertificate",
    "PortfolioPlan",
    "PortfolioResult",
    "compile_portfolio_problem",
    "decode_portfolio_decision",
    "portfolio_result_from_native",
    "refresh_portfolio_compilation",
]

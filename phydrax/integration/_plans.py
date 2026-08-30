#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx

from .._frozendict import frozendict
from .._numerics import normalize_anisotropy, normalize_axis_rules, SmolyakAxisRule
from .._sampling import (
    AntitheticDesign,
    IIDDesign,
    LatinHypercubeDesign,
    RandomizedQMCDesign,
)
from .._strict import StrictModule
from ._bayesian_quadrature import BayesianQuadraturePlan
from ._rules import (
    CubatureRule,
    GaussKronrodRule,
    GaussLegendreRule,
    IntervalRule,
    ProbabilityRule,
    ReferenceRule,
)


class StratifiedDesign(StrictModule):
    """Sampling design over an explicit measurable partition."""

    partition: Any
    allocation: Literal["proportional", "equal", "explicit"] = eqx.field(static=True)
    allocation_weights: tuple[float, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        partition: Any,
        /,
        *,
        allocation: Literal["proportional", "equal", "explicit"] = "proportional",
        allocation_weights: Sequence[float] | None = None,
    ):
        if allocation not in ("proportional", "equal", "explicit"):
            raise ValueError("Unknown stratified allocation policy.")
        weights = (
            None
            if allocation_weights is None
            else tuple(float(value) for value in allocation_weights)
        )
        if allocation == "explicit" and weights is None:
            raise ValueError("Explicit allocation requires allocation_weights.")
        if allocation != "explicit" and weights is not None:
            raise ValueError("allocation_weights are only valid for explicit allocation.")
        self.partition = partition
        self.allocation = allocation
        self.allocation_weights = weights


class SampleMeanEstimator(StrictModule):
    """Ordinary sample-mean estimator."""

    def __init__(self):
        pass


class SelfNormalizedEstimator(StrictModule):
    """Ratio estimator using the observed sum of raw weights."""

    def __init__(self):
        pass


class ControlVariateEstimator(StrictModule):
    """One or more controls with supplied or independent-pilot coefficients."""

    controls: tuple[Any, ...]
    expectations: tuple[Any, ...]
    coefficients: Any
    pilot_samples: int = eqx.field(static=True)
    same_sample_asymptotic: bool = eqx.field(static=True)
    regularization: float = eqx.field(static=True)

    def __init__(
        self,
        controls: Sequence[Callable],
        expectations: Sequence[Any],
        /,
        *,
        coefficients: Any = None,
        pilot_samples: int = 64,
        same_sample_asymptotic: bool = False,
        regularization: float = 1e-8,
    ):
        controls_ = tuple(controls)
        expectations_ = tuple(expectations)
        if not controls_ or len(controls_) != len(expectations_):
            raise ValueError("controls and expectations must have equal nonzero length.")
        pilot = int(pilot_samples)
        if coefficients is None and pilot < 2:
            raise ValueError(
                "pilot_samples must be at least two when fitting coefficients."
            )
        regularization_ = float(regularization)
        if not math.isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and nonnegative.")
        self.controls = controls_
        self.expectations = expectations_
        self.coefficients = coefficients
        self.pilot_samples = pilot
        self.same_sample_asymptotic = bool(same_sample_asymptotic)
        self.regularization = regularization_


class FixedQuadraturePlan(StrictModule):
    """Materialize a reusable deterministic quadrature batch."""

    rule: Any

    def __init__(
        self, rule: IntervalRule | ProbabilityRule | ReferenceRule | None = None
    ):
        self.rule = GaussLegendreRule() if rule is None else rule


class AdaptiveQuadraturePlan(StrictModule):
    """Bounded globally adaptive one-dimensional quadrature."""

    rule: IntervalRule
    absolute_tolerance: float | None = eqx.field(static=True)
    relative_tolerance: float | None = eqx.field(static=True)
    max_intervals: int = eqx.field(static=True)
    max_evaluations: int | None = eqx.field(static=True)
    breakpoints: tuple[float, ...] = eqx.field(static=True)
    collect_partition: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        rule: IntervalRule | None = None,
        /,
        *,
        absolute_tolerance: float | None = None,
        relative_tolerance: float | None = None,
        max_intervals: int = 50,
        max_evaluations: int | None = None,
        breakpoints: Sequence[float] = (),
        collect_partition: bool = False,
        throw: bool = True,
    ):
        rule_ = GaussKronrodRule() if rule is None else rule
        absolute = _validate_tolerance(absolute_tolerance, "absolute_tolerance")
        relative = _validate_tolerance(relative_tolerance, "relative_tolerance")
        intervals = int(max_intervals)
        if intervals < 1:
            raise ValueError("max_intervals must be positive.")
        evaluations = None if max_evaluations is None else int(max_evaluations)
        if evaluations is not None and evaluations < 1:
            raise ValueError("max_evaluations must be positive.")
        points = tuple(float(point) for point in breakpoints)
        if any(not math.isfinite(point) for point in points):
            raise ValueError("breakpoints must be finite.")
        if any(
            right <= left for left, right in zip(points[:-1], points[1:], strict=True)
        ):
            raise ValueError("breakpoints must be strictly increasing and unique.")
        if intervals < len(points) + 1:
            raise ValueError(
                "max_intervals must cover every initial breakpoint interval."
            )
        self.rule = rule_
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.max_intervals = intervals
        self.max_evaluations = evaluations
        self.breakpoints = points
        self.collect_partition = bool(collect_partition)
        self.throw = bool(throw)


def _validate_tolerance(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    value_ = float(value)
    if not math.isfinite(value_) or value_ < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value_


class AdaptiveTrianglePlan(StrictModule):
    """Static-capacity adaptive quadrature over affine triangle charts."""

    low_rule: CubatureRule
    high_rule: CubatureRule
    absolute_tolerance: float | None = eqx.field(static=True)
    relative_tolerance: float | None = eqx.field(static=True)
    max_cells: int = eqx.field(static=True)
    max_evaluations: int | None = eqx.field(static=True)
    collect_partition: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        low_rule: CubatureRule | None = None,
        high_rule: CubatureRule | None = None,
        /,
        *,
        absolute_tolerance: float | None = None,
        relative_tolerance: float | None = None,
        max_cells: int = 256,
        max_evaluations: int | None = None,
        collect_partition: bool = False,
        throw: bool = True,
    ):
        low = CubatureRule("triangle", 5) if low_rule is None else low_rule
        high = CubatureRule("triangle", 10) if high_rule is None else high_rule
        if low.reference_domain != "triangle" or high.reference_domain != "triangle":
            raise ValueError("Adaptive triangle rules must use the triangle reference.")
        if high.exact_degree <= low.exact_degree:
            raise ValueError("Adaptive high_rule must have greater exact degree.")
        cells = int(max_cells)
        if cells < 1:
            raise ValueError("max_cells must be positive.")
        evaluations = None if max_evaluations is None else int(max_evaluations)
        if evaluations is not None and evaluations < 1:
            raise ValueError("max_evaluations must be positive.")
        self.low_rule = low
        self.high_rule = high
        self.absolute_tolerance = _validate_tolerance(
            absolute_tolerance, "absolute_tolerance"
        )
        self.relative_tolerance = _validate_tolerance(
            relative_tolerance, "relative_tolerance"
        )
        self.max_cells = cells
        self.max_evaluations = evaluations
        self.collect_partition = bool(collect_partition)
        self.throw = bool(throw)


class MonteCarloPlan(StrictModule):
    """Direct sampling from a target with an explicit sample design."""

    num_samples: int = eqx.field(static=True)
    design: Any
    control_variate: ControlVariateEstimator | None

    def __init__(
        self,
        num_samples: int,
        /,
        *,
        design: Any | None = None,
        control_variate: ControlVariateEstimator | None = None,
    ):
        count = int(num_samples)
        if count < 2:
            raise ValueError("Monte Carlo num_samples must be at least two.")
        design_ = IIDDesign() if design is None else design
        if not isinstance(design_, (IIDDesign, LatinHypercubeDesign, AntitheticDesign)):
            raise TypeError("Unsupported Monte Carlo sample design.")
        if control_variate is not None and not isinstance(
            control_variate, ControlVariateEstimator
        ):
            raise TypeError("control_variate must be a ControlVariateEstimator.")
        self.num_samples = count
        self.design = design_
        self.control_variate = control_variate


class StratifiedMonteCarloPlan(StrictModule):
    """Monte Carlo integration over an explicit target partition."""

    num_samples: int = eqx.field(static=True)
    design: StratifiedDesign

    def __init__(
        self,
        num_samples: int,
        design: StratifiedDesign,
        /,
    ):
        count = int(num_samples)
        if count < 2:
            raise ValueError("Stratified num_samples must be at least two.")
        if not isinstance(design, StratifiedDesign):
            raise TypeError("design must be a StratifiedDesign.")
        self.num_samples = count
        self.design = design


class QuasiMonteCarloPlan(StrictModule):
    """Deterministic or independently scrambled QMC integration."""

    num_samples: int = eqx.field(static=True)
    design: RandomizedQMCDesign
    control_variate: ControlVariateEstimator | None

    def __init__(
        self,
        num_samples: int,
        /,
        *,
        sequence: Literal["sobol", "halton"] = "sobol",
        scrambled: bool = True,
        num_replicates: int = 8,
        allow_arbitrary_count: bool = False,
        control_variate: ControlVariateEstimator | None = None,
    ):
        count = int(num_samples)
        if count < 2:
            raise ValueError("QMC num_samples must be at least two.")
        if sequence == "sobol" and not allow_arbitrary_count and count & (count - 1):
            raise ValueError("Sobol num_samples must be a power of two by default.")
        if control_variate is not None and not isinstance(
            control_variate, ControlVariateEstimator
        ):
            raise TypeError("control_variate must be a ControlVariateEstimator.")
        self.num_samples = count
        self.design = RandomizedQMCDesign(
            sequence=sequence,
            scrambled=scrambled,
            num_replicates=num_replicates,
            allow_arbitrary_count=allow_arbitrary_count,
        )
        self.control_variate = control_variate


class ImportanceSamplingPlan(StrictModule):
    """Proposal-corrected integration under a target measure."""

    num_samples: int = eqx.field(static=True)
    proposal: Any
    estimator: Any
    support_policy: Literal["strict"] = eqx.field(static=True)

    def __init__(
        self,
        num_samples: int,
        proposal: Any,
        /,
        *,
        self_normalized: bool = False,
        support_policy: Literal["strict"] = "strict",
    ):
        count = int(num_samples)
        if count < 2:
            raise ValueError("Importance num_samples must be at least two.")
        if support_policy != "strict":
            raise ValueError("Only strict proposal-support handling is supported.")
        self.num_samples = count
        self.proposal = proposal
        self.estimator = (
            SelfNormalizedEstimator() if self_normalized else SampleMeanEstimator()
        )
        self.support_policy = support_policy


class MultilevelMonteCarloPlan(StrictModule):
    """Fixed or variance-cost-adaptive multilevel Monte Carlo allocation."""

    initial_samples: int | tuple[int, ...] = eqx.field(static=True)
    samples_per_level: tuple[int, ...] | None = eqx.field(static=True)
    target_rmse: float | None = eqx.field(static=True)
    max_samples_per_level: int | tuple[int, ...] = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    variance_fraction: float = eqx.field(static=True)
    max_rounds: int = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        initial_samples: int | Sequence[int] = 16,
        samples_per_level: Sequence[int] | None = None,
        target_rmse: float | None = None,
        max_samples_per_level: int | Sequence[int] = 1_000_000,
        batch_size: int = 1024,
        variance_fraction: float = 0.5,
        max_rounds: int = 20,
    ):
        def counts(
            value: int | Sequence[int],
            name: str,
            minimum: int,
        ) -> int | tuple[int, ...]:
            if isinstance(value, int):
                resolved: int | tuple[int, ...] = int(value)
                values = (resolved,)
            else:
                resolved = tuple(int(item) for item in value)
                values = resolved
            if not values or any(item < minimum for item in values):
                raise ValueError(f"{name} values must be at least {minimum}.")
            return resolved

        initial = counts(initial_samples, "initial_samples", 2)
        fixed = (
            None
            if samples_per_level is None
            else tuple(int(value) for value in samples_per_level)
        )
        if fixed is not None and (not fixed or any(value < 2 for value in fixed)):
            raise ValueError("samples_per_level values must be at least two.")
        rmse = None if target_rmse is None else float(target_rmse)
        if rmse is not None and (not math.isfinite(rmse) or rmse <= 0.0):
            raise ValueError("target_rmse must be finite and positive.")
        if fixed is None and rmse is None:
            raise ValueError("Adaptive MLMC requires target_rmse.")
        maximum = counts(max_samples_per_level, "max_samples_per_level", 2)
        chunk = int(batch_size)
        if chunk < 1:
            raise ValueError("batch_size must be positive.")
        fraction = float(variance_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("variance_fraction must lie strictly between zero and one.")
        rounds = int(max_rounds)
        if rounds < 1:
            raise ValueError("max_rounds must be positive.")
        self.initial_samples = initial
        self.samples_per_level = fixed
        self.target_rmse = rmse
        self.max_samples_per_level = maximum
        self.batch_size = chunk
        self.variance_fraction = fraction
        self.max_rounds = rounds


class SparseGridPlan(StrictModule):
    """Smolyak sparse-grid integration with explicit per-axis rule families."""

    dimension: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    anisotropy: tuple[float, ...] = eqx.field(static=True)
    axis_rules: tuple[SmolyakAxisRule, ...] = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        level: int,
        /,
        *,
        anisotropy: Sequence[float] | None = None,
        axis_rules: (
            SmolyakAxisRule | Sequence[SmolyakAxisRule] | None
        ) = "clenshaw-curtis",
    ):
        dimension_ = int(dimension)
        level_ = int(level)
        if dimension_ < 1 or level_ < 1:
            raise ValueError("Sparse-grid dimension and level must be positive.")
        self.dimension = dimension_
        self.level = level_
        self.anisotropy = normalize_anisotropy(dimension_, anisotropy)
        self.axis_rules = normalize_axis_rules(
            dimension_,
            axis_rules,
            default="clenshaw-curtis",
            allowed=("clenshaw-curtis", "gauss-hermite"),
        )


class CellQuadraturePlan(StrictModule):
    """Apply a reference-cell rule through a supplied mapped target."""

    rule: ReferenceRule

    def __init__(self, rule: ReferenceRule):
        self.rule = rule


class ProductIntegrationPlan(StrictModule):
    """Map named axes or axis groups to independently composable plans."""

    plans: frozendict[tuple[str, ...], Any]

    def __init__(self, plans: Mapping[str | tuple[str, ...], Any]):
        if not plans:
            raise ValueError("ProductIntegrationPlan requires at least one axis plan.")
        normalized = {
            (key,) if isinstance(key, str) else tuple(key): factor_plan
            for key, factor_plan in plans.items()
        }
        if len(normalized) != len(plans):
            raise ValueError("ProductIntegrationPlan contains duplicate axis groups.")
        if any(
            not all(isinstance(label, str) and label for label in key)
            for key in normalized
        ):
            raise ValueError(
                "ProductIntegrationPlan axis groups must contain nonempty labels."
            )
        if any(not key for key in normalized):
            raise ValueError("ProductIntegrationPlan axis groups cannot be empty.")
        self.plans = frozendict(normalized)


IntegrationPlan: TypeAlias = (
    FixedQuadraturePlan
    | AdaptiveQuadraturePlan
    | AdaptiveTrianglePlan
    | BayesianQuadraturePlan
    | MonteCarloPlan
    | StratifiedMonteCarloPlan
    | QuasiMonteCarloPlan
    | ImportanceSamplingPlan
    | MultilevelMonteCarloPlan
    | SparseGridPlan
    | CellQuadraturePlan
    | ProductIntegrationPlan
)


__all__ = [
    "AdaptiveQuadraturePlan",
    "AdaptiveTrianglePlan",
    "AntitheticDesign",
    "BayesianQuadraturePlan",
    "CellQuadraturePlan",
    "ControlVariateEstimator",
    "FixedQuadraturePlan",
    "IIDDesign",
    "ImportanceSamplingPlan",
    "IntegrationPlan",
    "LatinHypercubeDesign",
    "MultilevelMonteCarloPlan",
    "MonteCarloPlan",
    "ProductIntegrationPlan",
    "QuasiMonteCarloPlan",
    "RandomizedQMCDesign",
    "SampleMeanEstimator",
    "SelfNormalizedEstimator",
    "SparseGridPlan",
    "StratifiedDesign",
    "StratifiedMonteCarloPlan",
]

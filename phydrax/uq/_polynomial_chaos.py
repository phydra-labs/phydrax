#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import ProbabilityDomain, ProductDomain
from phydrax.integration import (
    FixedQuadraturePlan,
    GaussianCubatureRule,
    IntegrationPrecisionPolicy,
    materialize as materialize_integration,
    MonteCarloPlan,
    over,
    ProductIntegrationPlan,
    ProductIntegrationRealization,
    QuasiMonteCarloPlan,
    SparseGridPlan,
)
from phydrax.integration._lowering import _fixed_rule_node_count
from phydrax.linalg import (
    DenseLinearOperator,
    DenseLU,
    DenseQR,
    LeastSquaresProblem,
    linear_status_message,
    LinearSolvePolicy,
    LinearSystem,
    RankPolicy,
    solve as solve_linear,
)

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._frozendict import frozendict
from .._polynomial import evaluate_tensor_basis, PolynomialMultiIndexSet
from .._sampling import RandomizedQMCDesign
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._distributions import Normal, Uniform


_DEFAULT_MAXIMUM_MODEL_EVALUATIONS = 1_000_000
_DEFAULT_MAXIMUM_SAMPLES = 1_000_000
_DEFAULT_MAXIMUM_BASIS_BYTES = 256 * 1024**2
_DEFAULT_MAXIMUM_DESIGN_BYTES = 256 * 1024**2
_POLYNOMIAL_MODE_DIM = "__phydra_uq_polynomial_mode"


@dataclass(frozen=True, slots=True)
class _OutputLeafSpec:
    shape: tuple[int, ...]
    field_dims: tuple[str | None, ...] | None


class PolynomialChaosBasis(StrictModule, NonTrainableState):
    """Labeled orthonormal tensor basis for independent scalar probability factors."""

    factors: tuple[ProbabilityDomain, ...]
    multiindices: PolynomialMultiIndexSet
    labels: tuple[str, ...] = eqx.field(static=True)
    reference_measures: tuple[Literal["uniform", "standard-normal"], ...] = eqx.field(
        static=True
    )
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        factors: ProbabilityDomain | Sequence[ProbabilityDomain],
        degree: int | PolynomialMultiIndexSet,
        /,
        *,
        maximum_features: int = 4096,
        maximum_storage_bytes: int = 64 * 1024**2,
    ):
        factor_values = (
            (factors,) if isinstance(factors, ProbabilityDomain) else tuple(factors)
        )
        if not factor_values:
            raise ValueError("PolynomialChaosBasis requires at least one factor.")
        if any(not isinstance(factor, ProbabilityDomain) for factor in factor_values):
            raise TypeError(
                "Polynomial chaos factors must be independent scalar "
                "ProbabilityDomain objects."
            )
        labels = tuple(factor.label for factor in factor_values)
        if len(set(labels)) != len(labels):
            raise ValueError("Polynomial chaos factor labels must be unique.")

        measures: list[Literal["uniform", "standard-normal"]] = []
        factor_identity = []
        for factor in factor_values:
            distribution = factor.distribution
            if isinstance(distribution, Uniform):
                measures.append("uniform")
                factor_identity.append(
                    {
                        "label": factor.label,
                        "law": "uniform",
                        "parameters": array_tree_fingerprint(
                            (distribution.low, distribution.high)
                        ),
                    }
                )
            elif isinstance(distribution, Normal):
                measures.append("standard-normal")
                factor_identity.append(
                    {
                        "label": factor.label,
                        "law": "normal",
                        "parameters": array_tree_fingerprint(
                            (distribution.location, distribution.scale)
                        ),
                    }
                )
            else:
                raise TypeError(
                    "Polynomial chaos supports only phydrax.uq.Uniform and "
                    "phydrax.uq.Normal scalar factors."
                )

        if isinstance(degree, PolynomialMultiIndexSet):
            multiindices = degree
            if multiindices.dimension != len(factor_values):
                raise ValueError(
                    "Polynomial multiindex dimension must match the factor count."
                )
        else:
            multiindices = PolynomialMultiIndexSet(
                len(factor_values),
                degree,
                maximum_features=maximum_features,
                maximum_storage_bytes=maximum_storage_bytes,
            )

        self.factors = factor_values
        self.multiindices = multiindices
        self.labels = labels
        self.reference_measures = tuple(measures)
        self.basis_id = canonical_fingerprint(
            {
                "kind": "labeled-polynomial-chaos-basis-v1",
                "factors": factor_identity,
                "multiindices": multiindices.content_id,
            }
        )

    @property
    def degree(self) -> int:
        return self.multiindices.degree

    @property
    def feature_count(self) -> int:
        return self.multiindices.feature_count

    def evaluate(
        self,
        points: ArrayLike | Mapping[str, Any] | None = None,
        /,
        **coordinates: Any,
    ) -> Array:
        """Evaluate every basis mode while preserving declared factor order."""
        point_array = _ordered_points(points, coordinates, self.labels)
        canonical = jnp.stack(
            tuple(
                factor.reference_transport.to_reference(point_array[..., index])
                for index, factor in enumerate(self.factors)
            ),
            axis=-1,
        )
        return evaluate_tensor_basis(
            canonical,
            self.reference_measures,
            self.multiindices,
        )


class PolynomialChaosExpansion(StrictModule):
    """Immutable callable orthonormal polynomial-chaos expansion."""

    basis: PolynomialChaosBasis
    coefficient_leaves: tuple[Array, ...]
    output_tree: Any = eqx.field(static=True)
    output_specs: tuple[_OutputLeafSpec, ...] = eqx.field(static=True)
    expansion_id: str = eqx.field(static=True)

    def __init__(self, basis: PolynomialChaosBasis, coefficients: Any, /):
        if not isinstance(basis, PolynomialChaosBasis):
            raise TypeError("basis must be a PolynomialChaosBasis.")
        leaves, tree = jax.tree_util.tree_flatten(
            coefficients, is_leaf=lambda value: isinstance(value, cx.Field)
        )
        if not leaves:
            raise ValueError("Polynomial-chaos coefficients must have array leaves.")
        arrays: list[Array] = []
        specs: list[_OutputLeafSpec] = []
        for leaf in leaves:
            if isinstance(leaf, cx.Field):
                array = jnp.asarray(leaf.data)
                if len(leaf.dims) != array.ndim or not leaf.dims:
                    raise ValueError(
                        "Coefficient Field dimensions must include a leading basis axis."
                    )
                field_dims = tuple(leaf.dims[1:])
            else:
                array = jnp.asarray(leaf)
                field_dims = None
            if array.ndim < 1 or array.shape[0] != basis.feature_count:
                raise ValueError(
                    "Every coefficient leaf must have the basis feature count on "
                    "its leading axis."
                )
            if not jnp.issubdtype(array.dtype, jnp.number):
                raise TypeError("Polynomial-chaos coefficients must be numeric.")
            if bool(jnp.any(~jnp.isfinite(array))):
                raise ValueError("Polynomial-chaos coefficients must be finite.")
            arrays.append(array)
            specs.append(_OutputLeafSpec(tuple(array.shape[1:]), field_dims))

        self.basis = basis
        self.coefficient_leaves = tuple(arrays)
        self.output_tree = tree
        self.output_specs = tuple(specs)
        self.expansion_id = canonical_fingerprint(
            {
                "kind": "polynomial-chaos-expansion-v1",
                "basis_id": basis.basis_id,
                "coefficients": array_tree_fingerprint(tuple(arrays)),
                "output": tuple(
                    {
                        "shape": spec.shape,
                        "field_dims": spec.field_dims,
                    }
                    for spec in specs
                ),
            }
        )

    @property
    def coefficients(self) -> Any:
        """Coefficient PyTree with the orthonormal-mode axis leading every leaf."""
        return _restore_coefficients(
            self.coefficient_leaves, self.output_tree, self.output_specs
        )

    def __call__(
        self,
        points: ArrayLike | Mapping[str, Any] | None = None,
        /,
        **coordinates: Any,
    ) -> Any:
        basis_values = self.basis.evaluate(points, **coordinates)
        leading_dims = _point_leading_dims(
            points,
            coordinates,
            self.basis.labels,
            basis_values.ndim - 1,
        )
        values = tuple(
            _contract_basis(basis_values, coefficient)
            for coefficient in self.coefficient_leaves
        )
        return _restore_outputs(
            values,
            self.output_tree,
            self.output_specs,
            leading_dims=leading_dims,
        )

    @property
    def mean(self) -> Any:
        """Exact expansion mean under the declared product probability measure."""
        return _restore_outputs(
            tuple(coefficient[0] for coefficient in self.coefficient_leaves),
            self.output_tree,
            self.output_specs,
            leading_dims=(),
        )

    @property
    def variance(self) -> Any:
        """Pointwise output variance without flattening physical output axes."""
        return _restore_outputs(
            tuple(
                _coefficient_energy(coefficient[1:])
                for coefficient in self.coefficient_leaves
            ),
            self.output_tree,
            self.output_specs,
            leading_dims=(),
        )

    @property
    def first_order_sobol(self) -> frozendict[str, Any]:
        """First-order Sobol effects derived from orthonormal coefficient energy."""
        indices = self.basis.multiindices.indices
        only = jnp.count_nonzero(indices, axis=1) == 1
        effects = {}
        for coordinate, label in enumerate(self.basis.labels):
            active = indices[:, coordinate] > 0
            effects[label] = self._sobol_effect(active & only)
        return frozendict(effects)

    @property
    def total_order_sobol(self) -> frozendict[str, Any]:
        """Total-order Sobol effects derived from orthonormal coefficient energy."""
        indices = self.basis.multiindices.indices
        return frozendict(
            {
                label: self._sobol_effect(indices[:, coordinate] > 0)
                for coordinate, label in enumerate(self.basis.labels)
            }
        )

    def _sobol_effect(self, mask: Array, /) -> Any:
        leaves = []
        for coefficient in self.coefficient_leaves:
            total = _coefficient_energy(coefficient[1:])
            selected = coefficient * mask.reshape(
                (mask.shape[0],) + (1,) * (coefficient.ndim - 1)
            )
            contribution = _coefficient_energy(selected)
            denominator = jnp.where(total > 0.0, total, 1.0)
            leaves.append(jnp.where(total > 0.0, contribution / denominator, 0.0))
        return _restore_outputs(
            tuple(leaves),
            self.output_tree,
            self.output_specs,
            leading_dims=(),
        )


class PolynomialChaosFitResult(StrictModule):
    """A fitted expansion with method, residual, solver, and provenance evidence."""

    expansion: PolynomialChaosExpansion
    residual_norm: Any
    relative_residual_norm: Any
    solver_statuses: tuple[Array, ...]
    solver_diagnostics: tuple[Any, ...]
    method: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    model_evaluations: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence: frozendict[str, Any] = eqx.field(static=True)
    provenance: frozendict[str, Any] = eqx.field(static=True)

    def __init__(
        self,
        expansion: PolynomialChaosExpansion,
        /,
        *,
        method: str,
        sample_count: int,
        model_evaluations: int,
        rank: int,
        residual_norm: Any = None,
        relative_residual_norm: Any = None,
        solver_statuses: Sequence[Array] = (),
        solver_diagnostics: Sequence[Any] = (),
        evidence: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ):
        if not isinstance(expansion, PolynomialChaosExpansion):
            raise TypeError("expansion must be a PolynomialChaosExpansion.")
        statuses = tuple(
            jnp.asarray(status, dtype=jnp.int32) for status in solver_statuses
        )
        diagnostics = tuple(solver_diagnostics)
        if len(statuses) != len(diagnostics):
            raise ValueError("Solver statuses and diagnostics must align.")
        self.expansion = expansion
        self.residual_norm = residual_norm
        self.relative_residual_norm = relative_residual_norm
        self.solver_statuses = statuses
        self.solver_diagnostics = diagnostics
        self.method = str(method)
        self.sample_count = int(sample_count)
        self.model_evaluations = int(model_evaluations)
        self.rank = int(rank)
        self.successful = True
        self.evidence = frozendict(evidence)
        self.provenance = frozendict(provenance)


class PolynomialChaosProjectionPlan(StrictModule, NonTrainableState):
    """Project a pointwise model through one existing product integration plan."""

    basis: PolynomialChaosBasis
    integration_plan: ProductIntegrationPlan
    precision: IntegrationPrecisionPolicy
    maximum_model_evaluations: int = eqx.field(static=True)
    maximum_basis_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PolynomialChaosBasis,
        integration_plan: ProductIntegrationPlan,
        /,
        *,
        precision: IntegrationPrecisionPolicy | None = None,
        maximum_model_evaluations: int = _DEFAULT_MAXIMUM_MODEL_EVALUATIONS,
        maximum_basis_bytes: int = _DEFAULT_MAXIMUM_BASIS_BYTES,
    ):
        if not isinstance(basis, PolynomialChaosBasis):
            raise TypeError("basis must be a PolynomialChaosBasis.")
        if not isinstance(integration_plan, ProductIntegrationPlan):
            raise TypeError("integration_plan must be a ProductIntegrationPlan.")
        precision_ = IntegrationPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, IntegrationPrecisionPolicy):
            raise TypeError("precision must be an IntegrationPrecisionPolicy.")
        maximum = _positive_integer(
            maximum_model_evaluations, "maximum_model_evaluations"
        )
        maximum_basis = _positive_integer(maximum_basis_bytes, "maximum_basis_bytes")
        covered = tuple(label for group in integration_plan.plans for label in group)
        if len(covered) != len(set(covered)) or set(covered) != set(basis.labels):
            raise ValueError(
                "The product integration plan must cover every polynomial-chaos "
                "factor label exactly once."
            )
        self.basis = basis
        self.integration_plan = integration_plan
        self.precision = precision_
        self.maximum_model_evaluations = maximum
        self.maximum_basis_bytes = maximum_basis
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polynomial-chaos-projection-plan-v1",
                "basis_id": basis.basis_id,
                "factor_plans": tuple(
                    {
                        "labels": labels,
                        "plan": _content_identity(factor_plan),
                    }
                    for labels, factor_plan in integration_plan.plans.items()
                ),
                "precision": precision_.policy_id,
                "maximum_model_evaluations": maximum,
                "maximum_basis_bytes": maximum_basis,
            }
        )

    def fit(
        self,
        model: Callable[..., Any],
        /,
        *,
        key: Key[Array, ""] | None = None,
    ) -> PolynomialChaosFitResult:
        """Evaluate and orthogonally project a deterministic pointwise model."""
        if not callable(model):
            raise TypeError("model must be callable.")
        basis_bytes_per_sample = _basis_storage_bytes(
            1,
            self.basis.feature_count,
            self.precision,
        )
        basis_sample_limit = self.maximum_basis_bytes // basis_bytes_per_sample
        preflight_limit = min(
            self.maximum_model_evaluations,
            basis_sample_limit,
        )
        preflight_samples, preflight_replicates = _preflight_product_counts(
            self.integration_plan,
            limit=preflight_limit,
        )
        preflight_evaluations = preflight_samples * preflight_replicates
        if preflight_evaluations > self.maximum_model_evaluations:
            raise ValueError(
                f"Projection requires at most {preflight_evaluations} model "
                "evaluations before materialization, exceeding "
                f"maximum_model_evaluations={self.maximum_model_evaluations}."
            )
        preflight_basis_bytes = _basis_storage_bytes(
            preflight_samples,
            self.basis.feature_count,
            self.precision,
        )
        if preflight_basis_bytes > self.maximum_basis_bytes:
            raise ValueError(
                f"Projection basis evaluation requires at most "
                f"{preflight_basis_bytes} bytes before materialization, exceeding "
                f"maximum_basis_bytes={self.maximum_basis_bytes}."
            )
        domain = ProductDomain(*self.basis.factors)
        target = over(domain.component())
        realization = (
            materialize_integration(
                target,
                self.integration_plan,
                precision=self.precision,
            )
            if key is None
            else materialize_integration(
                target,
                self.integration_plan,
                key=key,
                precision=self.precision,
            )
        )
        product = realization.batch
        if not isinstance(product, ProductIntegrationRealization):
            raise TypeError("Projection requires a product integration realization.")
        sample_counts = tuple(
            math.prod(int(batch.weights.named_shape[axis]) for axis in batch.axes)
            for batch in product.batches
        )
        if len(product.batches) != preflight_replicates:
            raise RuntimeError(
                "Product integration replicate count changed after preflight."
            )
        if any(count > preflight_samples for count in sample_counts):
            raise RuntimeError(
                "Product integration sample count exceeded its preflight bound."
            )
        total_evaluations = sum(sample_counts)
        if total_evaluations > self.maximum_model_evaluations:
            raise ValueError(
                f"Projection requires {total_evaluations} model evaluations, exceeding "
                f"maximum_model_evaluations={self.maximum_model_evaluations}."
            )

        projected_batches: list[tuple[Array, ...]] = []
        output_tree = None
        output_specs = None
        weight_masses: list[float] = []
        for batch in product.batches:
            points, weights = _flatten_product_batch(batch, self.basis.labels)
            values, tree, specs = _evaluate_pointwise_model(model, points)
            if output_tree is None:
                output_tree = tree
                output_specs = specs
            elif tree != output_tree or specs != output_specs:
                raise ValueError(
                    "Projection model output structure changed between "
                    "integration batches."
                )
            if any(bool(jnp.any(~jnp.isfinite(value))) for value in values):
                raise ValueError("Projection model outputs must be finite.")
            evaluated_values = tuple(self.precision.evaluation(value) for value in values)
            if any(bool(jnp.any(~jnp.isfinite(value))) for value in evaluated_values):
                raise ValueError(
                    "Projection model outputs are nonfinite at evaluation precision."
                )
            basis_values = self.precision.evaluation(self.basis.evaluate(points))
            if bool(jnp.any(~jnp.isfinite(basis_values))):
                raise ValueError(
                    "Polynomial basis values are nonfinite at evaluation precision."
                )
            accumulation_weights = self.precision.accumulation(weights)
            accumulation_basis = self.precision.accumulation(basis_values)
            accumulation_values = tuple(
                self.precision.accumulation(value) for value in evaluated_values
            )
            if (
                bool(jnp.any(~jnp.isfinite(accumulation_weights)))
                or bool(jnp.any(~jnp.isfinite(accumulation_basis)))
                or any(
                    bool(jnp.any(~jnp.isfinite(value))) for value in accumulation_values
                )
            ):
                raise ValueError(
                    "Projection inputs are nonfinite at accumulation precision."
                )
            projected = tuple(
                _project_leaf(
                    accumulation_weights,
                    accumulation_basis,
                    value,
                )
                for value in accumulation_values
            )
            if any(bool(jnp.any(~jnp.isfinite(value))) for value in projected):
                raise ValueError(
                    "Polynomial projection contraction produced nonfinite coefficients."
                )
            projected_batches.append(projected)
            weight_masses.append(float(np.asarray(jnp.sum(accumulation_weights))))

        if output_tree is None or output_specs is None:
            raise RuntimeError("Product integration produced no projection batches.")
        accumulated_coefficients = tuple(
            self.precision.accumulation(
                jnp.mean(
                    jnp.stack(tuple(batch[index] for batch in projected_batches), axis=0),
                    axis=0,
                )
            )
            for index in range(len(projected_batches[0]))
        )
        if any(bool(jnp.any(~jnp.isfinite(value))) for value in accumulated_coefficients):
            raise ValueError(
                "Projection replicate reduction produced nonfinite coefficients."
            )
        coefficient_leaves = tuple(
            self.precision.output(value) for value in accumulated_coefficients
        )
        if any(bool(jnp.any(~jnp.isfinite(value))) for value in coefficient_leaves):
            raise ValueError(
                "Projection output precision produced nonfinite coefficients."
            )
        coefficient_tree = _restore_coefficients(
            coefficient_leaves, output_tree, output_specs
        )
        expansion = PolynomialChaosExpansion(self.basis, coefficient_tree)
        return PolynomialChaosFitResult(
            expansion,
            method="projection",
            sample_count=max(sample_counts),
            model_evaluations=total_evaluations,
            rank=self.basis.feature_count,
            evidence={
                "basis_id": self.basis.basis_id,
                "plan_id": self.plan_id,
                "replicate_count": len(product.batches),
                "weight_masses": tuple(weight_masses),
                "deterministic_axes": product.deterministic_axes,
                "stochastic_axes": product.stochastic_axes,
                "precision_policy_id": self.precision.policy_id,
                "preflight_sample_upper_bound": preflight_samples,
                "preflight_evaluation_upper_bound": preflight_evaluations,
                "preflight_basis_bytes": preflight_basis_bytes,
            },
            provenance={
                "method": "product-integration-orthogonal-projection",
                "integration_plan": type(self.integration_plan).__name__,
                "basis": "normalized-legendre-hermite-tensor",
                "approximation": "nonintrusive-projection",
            },
        )


class PolynomialChaosRegressionPlan(StrictModule, NonTrainableState):
    """Diagnosed exact or overdetermined native linear polynomial-chaos fit."""

    basis: PolynomialChaosBasis
    exact_policy: LinearSolvePolicy
    least_squares_policy: LinearSolvePolicy
    maximum_samples: int = eqx.field(static=True)
    maximum_design_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PolynomialChaosBasis,
        /,
        *,
        exact_policy: LinearSolvePolicy | None = None,
        least_squares_policy: LinearSolvePolicy | None = None,
        maximum_samples: int = _DEFAULT_MAXIMUM_SAMPLES,
        maximum_design_bytes: int = _DEFAULT_MAXIMUM_DESIGN_BYTES,
    ):
        if not isinstance(basis, PolynomialChaosBasis):
            raise TypeError("basis must be a PolynomialChaosBasis.")
        exact = LinearSolvePolicy(DenseLU()) if exact_policy is None else exact_policy
        least_squares = (
            LinearSolvePolicy(DenseQR(), rank=RankPolicy(require_full_rank=True))
            if least_squares_policy is None
            else least_squares_policy
        )
        if not isinstance(exact, LinearSolvePolicy):
            raise TypeError("exact_policy must be a LinearSolvePolicy.")
        if not isinstance(least_squares, LinearSolvePolicy):
            raise TypeError("least_squares_policy must be a LinearSolvePolicy.")
        maximum_samples_ = _positive_integer(maximum_samples, "maximum_samples")
        maximum_bytes = _positive_integer(maximum_design_bytes, "maximum_design_bytes")
        self.basis = basis
        self.exact_policy = exact
        self.least_squares_policy = least_squares
        self.maximum_samples = maximum_samples_
        self.maximum_design_bytes = maximum_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polynomial-chaos-regression-plan-v1",
                "basis_id": basis.basis_id,
                "exact_policy": _content_identity(exact),
                "least_squares_policy": _content_identity(least_squares),
                "maximum_samples": maximum_samples_,
                "maximum_design_bytes": maximum_bytes,
            }
        )

    def fit(
        self,
        points: ArrayLike | Mapping[str, Any],
        values: Any,
        /,
        *,
        weights: ArrayLike | None = None,
    ) -> PolynomialChaosFitResult:
        """Fit supplied observations without rank repair or sample truncation."""
        point_array = _ordered_points(points, {}, self.basis.labels)
        if point_array.ndim != 2:
            raise ValueError("Regression points must describe one leading sample axis.")
        sample_count = int(point_array.shape[0])
        feature_count = self.basis.feature_count
        if sample_count < feature_count:
            raise ValueError(
                f"Regression requires at least {feature_count} samples for this basis; "
                f"received {sample_count}."
            )
        if sample_count > self.maximum_samples:
            raise ValueError(
                f"Regression received {sample_count} samples, exceeding "
                f"maximum_samples={self.maximum_samples}."
            )
        if bool(jnp.any(~jnp.isfinite(point_array))):
            raise ValueError("Regression samples must be finite.")
        design_dtype = jnp.asarray(point_array, dtype=float).dtype
        design_bytes = int(sample_count * feature_count * np.dtype(design_dtype).itemsize)
        if design_bytes > self.maximum_design_bytes:
            raise ValueError(
                f"Regression design requires {design_bytes} bytes, exceeding "
                f"maximum_design_bytes={self.maximum_design_bytes}."
            )
        design = self.basis.evaluate(point_array)
        if bool(jnp.any(~jnp.isfinite(design))):
            raise ValueError(
                "Regression transformed coordinates and basis values must be finite."
            )

        output_leaves, output_tree, output_specs = _sampled_output_leaves(
            values, sample_count
        )
        if any(bool(jnp.any(~jnp.isfinite(value))) for value in output_leaves):
            raise ValueError("Regression outputs must be finite.")
        weights_ = None if weights is None else jnp.asarray(weights).reshape((-1,))
        if weights_ is not None:
            if weights_.shape != (sample_count,):
                raise ValueError("Regression weights must contain one value per sample.")
            if bool(jnp.any(~jnp.isfinite(weights_))) or bool(jnp.any(weights_ < 0.0)):
                raise ValueError("Regression weights must be finite and non-negative.")
            if not bool(jnp.any(weights_ > 0.0)):
                raise ValueError("Regression weights must have positive total weight.")

        exact = sample_count == feature_count and weights_ is None
        operator = DenseLinearOperator(design)
        problem = (
            LinearSystem(operator)
            if exact
            else LeastSquaresProblem(operator, weights=weights_)
        )
        policy = self.exact_policy if exact else self.least_squares_policy
        coefficient_leaves: list[Array] = []
        residual_leaves: list[Array] = []
        relative_leaves: list[Array] = []
        statuses: list[Array] = []
        diagnostics = []
        ranks: list[int] = []
        backends: list[str] = []
        methods: list[str] = []
        for output, spec in zip(output_leaves, output_specs, strict=True):
            right_hand_side = output.reshape((sample_count, -1))
            result = solve_linear(problem, right_hand_side, policy=policy)
            if not bool(jnp.all(result.successful)):
                status_value = int(np.asarray(result.status).reshape((-1,))[0])
                raise ValueError(
                    "Polynomial-chaos regression solve failed: "
                    f"{linear_status_message(status_value)}."
                )
            coefficients = jnp.asarray(result.value).reshape(
                (feature_count,) + spec.shape
            )
            if bool(jnp.any(~jnp.isfinite(coefficients))):
                raise ValueError(
                    "Polynomial-chaos regression coefficients are nonfinite."
                )
            prediction = _contract_design(design, coefficients)
            residual = prediction - output
            residual_norm = jnp.sqrt(
                jnp.sum(jnp.real(residual * jnp.conj(residual)), axis=0)
            )
            reference_norm = jnp.sqrt(
                jnp.sum(jnp.real(output * jnp.conj(output)), axis=0)
            )
            safe_reference = jnp.where(reference_norm > 0.0, reference_norm, 1.0)
            relative = jnp.where(
                reference_norm > 0.0,
                residual_norm / safe_reference,
                jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
            )
            coefficient_leaves.append(coefficients)
            residual_leaves.append(residual_norm)
            relative_leaves.append(relative)
            statuses.append(result.status)
            diagnostics.append(result.diagnostics)
            diagnosed_rank = int(np.asarray(result.diagnostics.rank).min())
            ranks.append(feature_count if diagnosed_rank < 0 else diagnosed_rank)
            backends.append(result.provenance.backend)
            methods.append(result.provenance.method)

        coefficient_tree = _restore_coefficients(
            tuple(coefficient_leaves), output_tree, output_specs
        )
        expansion = PolynomialChaosExpansion(self.basis, coefficient_tree)
        residual_tree = _restore_outputs(
            tuple(residual_leaves),
            output_tree,
            output_specs,
            leading_dims=(),
        )
        relative_tree = _restore_outputs(
            tuple(relative_leaves),
            output_tree,
            output_specs,
            leading_dims=(),
        )
        contract = "exact" if exact else "least-squares"
        return PolynomialChaosFitResult(
            expansion,
            method=f"regression-{contract}",
            sample_count=sample_count,
            model_evaluations=0,
            rank=min(ranks),
            residual_norm=residual_tree,
            relative_residual_norm=relative_tree,
            solver_statuses=statuses,
            solver_diagnostics=diagnostics,
            evidence={
                "basis_id": self.basis.basis_id,
                "plan_id": self.plan_id,
                "linear_problem": contract,
                "design_shape": tuple(design.shape),
                "design_bytes": design_bytes,
                "weighted": weights_ is not None,
                "diagnosed_rank": min(ranks),
            },
            provenance={
                "method": f"native-{contract}-polynomial-regression",
                "backends": tuple(backends),
                "linear_methods": tuple(methods),
                "basis": "normalized-legendre-hermite-tensor",
                "approximation": "nonintrusive-regression",
            },
        )


def _content_identity(value: Any, /) -> dict[str, Any]:
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "representation": repr(value),
        "arrays": array_tree_fingerprint(value),
    }


def _preflight_product_counts(
    plan: ProductIntegrationPlan,
    /,
    *,
    limit: int,
) -> tuple[int, int]:
    replicate_counts = {
        factor_plan.design.num_replicates
        for factor_plan in plan.plans.values()
        if (
            isinstance(factor_plan, QuasiMonteCarloPlan)
            and isinstance(factor_plan.design, RandomizedQMCDesign)
            and factor_plan.design.num_replicates > 1
        )
    }
    if len(replicate_counts) > 1:
        raise ValueError("Randomized-QMC product factors must use one replicate count.")
    replicates = 1 if not replicate_counts else replicate_counts.pop()
    samples_per_replicate = 1
    for labels, factor_plan in plan.plans.items():
        factor_count = _factor_preflight_count(
            labels,
            factor_plan,
            limit=limit,
        )
        samples_per_replicate = _saturating_product(
            samples_per_replicate,
            factor_count,
            limit,
        )
        if samples_per_replicate > limit:
            break
    return samples_per_replicate, replicates


def _factor_preflight_count(
    labels: tuple[str, ...],
    plan: Any,
    /,
    *,
    limit: int,
) -> int:
    if isinstance(plan, FixedQuadraturePlan):
        count = _fixed_rule_node_count(plan.rule)
        if isinstance(plan.rule, GaussianCubatureRule):
            if plan.rule.dimension != len(labels):
                raise ValueError(
                    "Gaussian cubature dimension must match its product labels."
                )
            return min(count, limit + 1)
        result = 1
        for _ in labels:
            result = _saturating_product(result, count, limit)
        return result
    if isinstance(plan, SparseGridPlan):
        if plan.dimension != len(labels):
            raise ValueError("Sparse-grid factor dimension must match its label group.")
        return _sparse_grid_node_upper_bound(plan, limit=limit)
    if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)):
        return min(plan.num_samples, limit + 1)
    raise TypeError(f"Unsupported product factor plan {type(plan).__name__!r}.")


def _sparse_grid_node_upper_bound(
    plan: SparseGridPlan,
    /,
    *,
    limit: int,
) -> int:
    maximum_levels = []
    for weight in plan.anisotropy:
        numerator, denominator = weight.as_integer_ratio()
        maximum_level = (plan.level * denominator) // numerator
        if maximum_level > limit:
            return limit + 1
        maximum_levels.append(maximum_level)
    axis_counts = tuple(
        _sparse_axis_node_upper_bound(rule, level, limit=limit)
        for rule, level in zip(
            plan.axis_rules,
            maximum_levels,
            strict=True,
        )
    )
    if plan.dimension == 1:
        return axis_counts[0]
    term_count = 1
    tensor_count = 1
    for maximum_level, axis_count in zip(
        maximum_levels,
        axis_counts,
        strict=True,
    ):
        term_count = _saturating_product(
            term_count,
            maximum_level + 1,
            limit,
        )
        tensor_count = _saturating_product(
            tensor_count,
            axis_count,
            limit,
        )
    return _saturating_product(term_count, tensor_count, limit)


def _sparse_axis_node_upper_bound(
    rule: str,
    level: int,
    /,
    *,
    limit: int,
) -> int:
    if rule == "gauss-hermite":
        return min(level + 1, limit + 1)
    if rule == "clenshaw-curtis":
        if level == 0:
            return 1
        if level >= max(1, (limit + 1).bit_length()):
            return limit + 1
        return min((1 << level) + 1, limit + 1)
    raise ValueError(f"Unsupported sparse-grid axis rule {rule!r}.")


def _saturating_product(left: int, right: int, limit: int, /) -> int:
    if left > limit or right > limit:
        return limit + 1
    if right != 0 and left > limit // right:
        return limit + 1
    return left * right


def _basis_storage_bytes(
    sample_count: int,
    feature_count: int,
    precision: IntegrationPrecisionPolicy,
    /,
) -> int:
    source_dtype = jnp.asarray(0.0).dtype
    evaluation_dtype = (
        source_dtype
        if precision.evaluation_dtype is None
        else jnp.dtype(precision.evaluation_dtype)
    )
    accumulation_dtype = (
        evaluation_dtype
        if precision.accumulation_dtype is None
        else jnp.dtype(precision.accumulation_dtype)
    )
    itemsize = source_dtype.itemsize
    if evaluation_dtype != source_dtype:
        itemsize += evaluation_dtype.itemsize
    if accumulation_dtype != evaluation_dtype:
        itemsize += accumulation_dtype.itemsize
    return int(sample_count * feature_count * itemsize)


def _ordered_points(
    points: ArrayLike | Mapping[str, Any] | None,
    coordinates: Mapping[str, Any],
    labels: tuple[str, ...],
    /,
) -> Array:
    if points is not None and coordinates:
        raise TypeError("Supply points or labeled coordinate keywords, not both.")
    selected: Any = coordinates if points is None else points
    if selected is None:
        raise TypeError("Polynomial basis evaluation requires points.")
    if isinstance(selected, Mapping):
        missing = tuple(label for label in labels if label not in selected)
        extra = tuple(label for label in selected if label not in labels)
        if missing or extra:
            raise ValueError(
                f"Labeled points must match basis labels; missing={missing!r}, "
                f"extra={extra!r}."
            )
        arrays = tuple(
            jnp.asarray(
                selected[label].data
                if isinstance(selected[label], cx.Field)
                else selected[label]
            )
            for label in labels
        )
        broadcast = jnp.broadcast_arrays(*arrays)
        return jnp.stack(broadcast, axis=-1)
    array = jnp.asarray(selected)
    if len(labels) == 1 and array.ndim == 0:
        return array.reshape((1,))
    if array.ndim < 1 or array.shape[-1] != len(labels):
        raise ValueError(
            f"Point arrays must have final dimension {len(labels)} in factor order "
            f"{labels!r}."
        )
    return array


def _point_leading_dims(
    points: ArrayLike | Mapping[str, Any] | None,
    coordinates: Mapping[str, Any],
    labels: tuple[str, ...],
    rank: int,
    /,
) -> tuple[str | None, ...]:
    selected: Any = coordinates if points is None else points
    if not isinstance(selected, Mapping):
        return (None,) * rank
    field_dims = tuple(
        tuple(selected[label].dims)
        for label in labels
        if isinstance(selected[label], cx.Field)
    )
    if not field_dims:
        return (None,) * rank
    expected = field_dims[0]
    if any(dims != expected for dims in field_dims[1:]):
        raise ValueError(
            "Labeled Field coordinates must use identical aligned dimensions."
        )
    if len(expected) != rank:
        raise ValueError(
            "Labeled Field coordinate dimensions do not match their broadcast rank."
        )
    return expected


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _restore_coefficients(
    leaves: tuple[Array, ...],
    tree: Any,
    specs: tuple[_OutputLeafSpec, ...],
    /,
) -> Any:
    restored = tuple(
        cx.Field(
            value,
            dims=(_POLYNOMIAL_MODE_DIM,) + spec.field_dims,
        )
        if spec.field_dims is not None
        else value
        for value, spec in zip(leaves, specs, strict=True)
    )
    return jax.tree_util.tree_unflatten(tree, restored)


def _restore_outputs(
    leaves: tuple[Array, ...],
    tree: Any,
    specs: tuple[_OutputLeafSpec, ...],
    /,
    *,
    leading_dims: tuple[str | None, ...],
) -> Any:
    restored = tuple(
        cx.Field(
            value,
            dims=leading_dims + spec.field_dims,
        )
        if spec.field_dims is not None
        else value
        for value, spec in zip(leaves, specs, strict=True)
    )
    return jax.tree_util.tree_unflatten(tree, restored)


def _contract_basis(basis_values: Array, coefficients: Array, /) -> Array:
    input_rank = basis_values.ndim - 1
    output_rank = coefficients.ndim - 1
    mode = input_rank + output_rank
    input_axes = list(range(input_rank))
    output_axes = list(range(input_rank, input_rank + output_rank))
    return oe.contract(
        basis_values,
        input_axes + [mode],
        coefficients,
        [mode] + output_axes,
        input_axes + output_axes,
    )


def _coefficient_energy(coefficients: Array, /) -> Array:
    if coefficients.shape[0] == 0:
        return jnp.zeros(coefficients.shape[1:], dtype=jnp.real(coefficients).dtype)
    mode = coefficients.ndim - 1
    output_axes = list(range(mode))
    return jnp.real(
        oe.contract(
            jnp.conj(coefficients),
            [mode] + output_axes,
            coefficients,
            [mode] + output_axes,
            output_axes,
        )
    )


def _project_leaf(weights: Array, basis: Array, values: Array, /) -> Array:
    output_rank = values.ndim - 1
    mode = output_rank + 1
    output_axes = list(range(1, output_rank + 1))
    return oe.contract(
        weights,
        [0],
        basis,
        [0, mode],
        values,
        [0] + output_axes,
        [mode] + output_axes,
    )


def _contract_design(design: Array, coefficients: Array, /) -> Array:
    output_rank = coefficients.ndim - 1
    output_axes = list(range(1, output_rank + 1))
    mode = output_rank + 1
    return oe.contract(
        design,
        [0, mode],
        coefficients,
        [mode] + output_axes,
        [0] + output_axes,
    )


def _broadcast_field_on_axes(
    field: cx.Field,
    axes: tuple[str, ...],
    sizes: tuple[int, ...],
    /,
) -> Array:
    if any(dim is None for dim in field.dims):
        raise ValueError(
            "Polynomial-chaos probability coordinates must be scalar fields."
        )
    unknown = tuple(dim for dim in field.dims if dim not in axes)
    if unknown:
        raise ValueError(f"Integration field contains unknown axes {unknown!r}.")
    present = tuple(axis for axis in axes if axis in field.dims)
    permutation = tuple(field.dims.index(axis) for axis in present)
    data = jnp.asarray(field.data)
    if permutation != tuple(range(len(permutation))):
        data = jnp.transpose(data, permutation)
    shape = tuple(
        sizes[index] if axis in present else 1 for index, axis in enumerate(axes)
    )
    return jnp.broadcast_to(data.reshape(shape), sizes)


def _flatten_product_batch(batch: Any, labels: tuple[str, ...], /) -> tuple[Array, Array]:
    axes = tuple(batch.axes)
    sizes = tuple(int(batch.weights.named_shape[axis]) for axis in axes)
    point_columns = []
    for label in labels:
        point = batch.points[label]
        if not isinstance(point, cx.Field):
            raise TypeError("Product probability points must be coordax.Field values.")
        point_columns.append(_broadcast_field_on_axes(point, axes, sizes).reshape((-1,)))
    points = jnp.stack(tuple(point_columns), axis=-1)
    weights = _broadcast_field_on_axes(batch.weights, axes, sizes).reshape((-1,))
    if bool(jnp.any(~jnp.isfinite(weights))):
        raise ValueError("Projection integration weights must be finite.")
    return points, weights


def _output_parts(
    value: Any, /
) -> tuple[tuple[Array, ...], Any, tuple[_OutputLeafSpec, ...]]:
    leaves, tree = jax.tree_util.tree_flatten(
        value, is_leaf=lambda leaf: isinstance(leaf, cx.Field)
    )
    if not leaves:
        raise ValueError("Polynomial-chaos model outputs must have array leaves.")
    arrays = []
    specs = []
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            array = jnp.asarray(leaf.data)
            if len(leaf.dims) != array.ndim:
                raise ValueError("Output Field dimensions must match its data rank.")
            field_dims = tuple(leaf.dims)
        else:
            array = jnp.asarray(leaf)
            field_dims = None
        if not jnp.issubdtype(array.dtype, jnp.number):
            raise TypeError("Polynomial-chaos model outputs must be numeric.")
        arrays.append(array)
        specs.append(_OutputLeafSpec(tuple(array.shape), field_dims))
    return tuple(arrays), tree, tuple(specs)


def _evaluate_pointwise_model(
    model: Callable[..., Any], points: Array, /
) -> tuple[tuple[Array, ...], Any, tuple[_OutputLeafSpec, ...]]:
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError("Projection requires a nonempty matrix of factor points.")
    first_coordinates = tuple(points[0, index] for index in range(points.shape[1]))
    first, tree, specs = _output_parts(model(*first_coordinates))

    def evaluate_arrays(*coordinates: Array) -> tuple[Array, ...]:
        leaves, candidate_tree, candidate_specs = _output_parts(model(*coordinates))
        if candidate_tree != tree or candidate_specs != specs:
            raise ValueError("Projection model output structure changed across points.")
        return leaves

    if points.shape[0] == 1:
        return tuple(value[None, ...] for value in first), tree, specs
    remaining = jax.vmap(evaluate_arrays)(
        *tuple(points[1:, index] for index in range(points.shape[1]))
    )
    values = tuple(
        jnp.concatenate((initial[None, ...], rest), axis=0)
        for initial, rest in zip(first, remaining, strict=True)
    )
    return values, tree, specs


def _sampled_output_leaves(
    values: Any, sample_count: int, /
) -> tuple[tuple[Array, ...], Any, tuple[_OutputLeafSpec, ...]]:
    leaves, tree = jax.tree_util.tree_flatten(
        values, is_leaf=lambda leaf: isinstance(leaf, cx.Field)
    )
    if not leaves:
        raise ValueError("Regression outputs must have array leaves.")
    arrays = []
    specs = []
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            array = jnp.asarray(leaf.data)
            if len(leaf.dims) != array.ndim or not leaf.dims:
                raise ValueError(
                    "Regression output Fields must include a leading sample dimension."
                )
            field_dims = tuple(leaf.dims[1:])
        else:
            array = jnp.asarray(leaf)
            field_dims = None
        if array.ndim < 1 or array.shape[0] != sample_count:
            raise ValueError(
                "Every regression output leaf must have one leading value per sample."
            )
        if not jnp.issubdtype(array.dtype, jnp.number):
            raise TypeError("Regression outputs must be numeric.")
        arrays.append(array)
        specs.append(_OutputLeafSpec(tuple(array.shape[1:]), field_dims))
    return tuple(arrays), tree, tuple(specs)


__all__ = [
    "PolynomialChaosBasis",
    "PolynomialChaosExpansion",
    "PolynomialChaosFitResult",
    "PolynomialChaosProjectionPlan",
    "PolynomialChaosRegressionPlan",
    "PolynomialMultiIndexSet",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...backends.clarabel import ClarabelPlan
from ...backends.mpax import MPAXPlan, MPAXRepresentation
from ...linalg import (
    FailurePolicy,
    MaterializationPolicy,
    SolveResourcePolicy,
    SparseFactorizationPolicy,
)
from ...typing import parse
from ._types import ConvexProgramCapabilities


ConvexDifferentiationMode: TypeAlias = Literal[
    "active-set-kkt", "barrier-kkt", "algorithmic", "none"
]
NativeConicNewtonRoute: TypeAlias = Literal["factorized", "matrix-free"]


class ConvexTermination(StrictModule):
    """Scale-aware optimality and ray-certificate termination thresholds."""

    absolute: float = eqx.field(static=True)
    relative: float = eqx.field(static=True)
    primal_infeasible: float = eqx.field(static=True)
    dual_infeasible: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute: float = 1e-7,
        relative: float = 0.0,
        primal_infeasible: float = 1e-8,
        dual_infeasible: float = 1e-8,
        maximum_steps: int = 100,
    ) -> None:
        values = tuple(
            float(value)
            for value in (absolute, relative, primal_infeasible, dual_infeasible)
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Convex-program tolerances must be finite and non-negative.")
        if values[0] == 0.0 and values[1] == 0.0:
            raise ValueError("At least one optimality tolerance must be positive.")
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        (
            self.absolute,
            self.relative,
            self.primal_infeasible,
            self.dual_infeasible,
        ) = values
        self.maximum_steps = steps


class AbstractConvexProgramMethod(StrictModule):
    """Algorithm identity and static capabilities for one program method."""

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ConvexProgramCapabilities:
        raise NotImplementedError

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return ()


class NativeHomogeneousConic(AbstractConvexProgramMethod):
    """JAX-native fixed-capacity primal-dual execution on built-in cones.

    ``newton`` selects how sparse homogeneous-embedding Newton systems are
    solved. ``"factorized"`` eliminates slacks and non-zero-cone multipliers
    into a quasi-definite reduced KKT system that is symbolically analyzed once
    per program structure under ``factorization`` and numerically refreshed each
    iteration; preparation refuses when the reduced pattern exceeds the declared
    symbolic budget. ``"matrix-free"`` uses unpreconditioned native GMRES on
    the full embedding. Dense programs always use the dense LU Newton solve.
    """

    primal_step: float = eqx.field(static=True)
    dual_step: float = eqx.field(static=True)
    extrapolation: float = eqx.field(static=True)
    newton: NativeConicNewtonRoute = eqx.field(static=True)
    factorization: SparseFactorizationPolicy

    def __init__(
        self,
        *,
        primal_step: float = 1e-2,
        dual_step: float = 1e-2,
        extrapolation: float = 1.0,
        newton: NativeConicNewtonRoute = "factorized",
        factorization: SparseFactorizationPolicy | None = None,
    ) -> None:
        primal = float(primal_step)
        dual = float(dual_step)
        extrapolation_ = float(extrapolation)
        if not isfinite(primal) or not isfinite(dual) or primal <= 0.0 or dual <= 0.0:
            raise ValueError("Native conic steps must be finite and positive.")
        if not isfinite(extrapolation_) or not 0.0 <= extrapolation_ <= 1.0:
            raise ValueError("extrapolation must lie in [0, 1].")
        route = parse(newton, NativeConicNewtonRoute, "newton")
        factorization_ = (
            SparseFactorizationPolicy("lu", ordering="approximate-minimum-degree")
            if factorization is None
            else factorization
        )
        if not isinstance(factorization_, SparseFactorizationPolicy):
            raise TypeError("factorization must be a SparseFactorizationPolicy or None.")
        if factorization_.kind != "lu":
            raise ValueError(
                "The reduced homogeneous KKT system is symmetric indefinite; "
                "select SparseFactorizationPolicy('lu')."
            )
        if (
            factorization_.fill_level is not None
            or factorization_.drop_tolerance != 0.0
            or factorization_.maximum_fill_per_row is not None
            or factorization_.allow_pivot_replacement
            or factorization_.diagonal_shift != 0.0
        ):
            raise ValueError(
                "Native conic Newton factorizations must be complete and unmodified."
            )
        self.primal_step = primal
        self.dual_step = dual
        self.extrapolation = extrapolation_
        self.newton = route
        self.factorization = factorization_

    @property
    def method_id(self) -> str:
        return "native-homogeneous-conic"

    @property
    def backend(self) -> str:
        return "phydrax"

    @property
    def capabilities(self) -> ConvexProgramCapabilities:
        return ConvexProgramCapabilities(
            linear_program=True,
            quadratic_program=True,
            conic_program=True,
            dense=True,
            sparse=True,
            matrix_free=True,
            warm_start=False,
            prepared_refresh=True,
            infeasibility_certificates=True,
            # Fixed-active projection-KKT sensitivities of the audited witness
            # through prepare_conic_sensitivity (dense and matrix-free routes).
            implicit_differentiation=True,
            algorithmic_differentiation=False,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("primal_step", str(self.primal_step)),
            ("dual_step", str(self.dual_step)),
            ("extrapolation", str(self.extrapolation)),
            ("newton", self.newton),
            ("factorization_kind", self.factorization.kind),
            ("factorization_ordering", self.factorization.ordering),
            ("factorization_pivot_tolerance", repr(self.factorization.pivot_tolerance)),
            ("factorization_max_factor_nnz", str(self.factorization.max_factor_nnz)),
            ("factorization_max_factor_bytes", str(self.factorization.max_factor_bytes)),
            (
                "factorization_max_symbolic_work",
                str(self.factorization.max_symbolic_work),
            ),
        )


class DensePrimalDualQP(AbstractConvexProgramMethod):
    """Native dense predictor-corrector QP method."""

    step_fraction: float = eqx.field(static=True)
    max_kkt_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        step_fraction: float = 0.995,
        max_kkt_dimension: int = 512,
    ) -> None:
        fraction = float(step_fraction)
        dimension = int(max_kkt_dimension)
        if not isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("step_fraction must lie strictly between zero and one.")
        if dimension < 1:
            raise ValueError("max_kkt_dimension must be positive.")
        self.step_fraction = fraction
        self.max_kkt_dimension = dimension

    @property
    def method_id(self) -> str:
        return "dense-primal-dual"

    @property
    def backend(self) -> str:
        return "phydrax"

    @property
    def capabilities(self) -> ConvexProgramCapabilities:
        return ConvexProgramCapabilities(
            linear_program=True,
            quadratic_program=True,
            conic_program=False,
            dense=True,
            sparse=False,
            matrix_free=False,
            warm_start=True,
            prepared_refresh=True,
            infeasibility_certificates=True,
            implicit_differentiation=True,
            algorithmic_differentiation=False,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("step_fraction", repr(self.step_fraction)),
            ("max_kkt_dimension", str(self.max_kkt_dimension)),
        )


class MPAXraPDHG(AbstractConvexProgramMethod):
    """MPAX restarted-average PDHG for assembled LPs and convex QPs."""

    plan: MPAXPlan

    def __init__(
        self,
        *,
        representation: MPAXRepresentation = "dense",
        warm_start: bool = False,
        feasibility_polishing: bool = False,
        unroll: bool = False,
        iteration_limit: int = 10_000,
    ) -> None:
        self.plan = MPAXPlan(
            "rapdhg",
            representation=representation,
            warm_start=warm_start,
            feasibility_polishing=feasibility_polishing,
            unroll=unroll,
            iteration_limit=iteration_limit,
        )

    @property
    def method_id(self) -> str:
        return "mpax-rapdhg"

    @property
    def backend(self) -> str:
        return "mpax"

    @property
    def capabilities(self) -> ConvexProgramCapabilities:
        return ConvexProgramCapabilities(
            linear_program=True,
            quadratic_program=True,
            conic_program=self.plan.representation == "sparse",
            dense=self.plan.representation == "dense",
            sparse=self.plan.representation == "sparse",
            matrix_free=False,
            warm_start=self.plan.warm_start,
            prepared_refresh=True,
            infeasibility_certificates=True,
            implicit_differentiation=False,
            algorithmic_differentiation=self.plan.unroll,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (("plan_id", self.plan.plan_id),)


class MPAXr2HPDHG(AbstractConvexProgramMethod):
    """MPAX reflected restarted Halpern PDHG for assembled LPs."""

    plan: MPAXPlan

    def __init__(
        self,
        *,
        representation: MPAXRepresentation = "dense",
        warm_start: bool = False,
        feasibility_polishing: bool = False,
        unroll: bool = False,
        iteration_limit: int = 10_000,
    ) -> None:
        self.plan = MPAXPlan(
            "r2hpdhg",
            representation=representation,
            warm_start=warm_start,
            feasibility_polishing=feasibility_polishing,
            unroll=unroll,
            iteration_limit=iteration_limit,
        )

    @property
    def method_id(self) -> str:
        return "mpax-r2hpdhg"

    @property
    def backend(self) -> str:
        return "mpax"

    @property
    def capabilities(self) -> ConvexProgramCapabilities:
        return ConvexProgramCapabilities(
            linear_program=True,
            quadratic_program=False,
            conic_program=self.plan.representation == "sparse",
            dense=self.plan.representation == "dense",
            sparse=self.plan.representation == "sparse",
            matrix_free=False,
            warm_start=self.plan.warm_start,
            prepared_refresh=True,
            infeasibility_certificates=True,
            implicit_differentiation=False,
            algorithmic_differentiation=self.plan.unroll,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (("plan_id", self.plan.plan_id),)


class ClarabelInteriorPoint(AbstractConvexProgramMethod):
    """Clarabel 0.11.1 host interior-point method for quadratic-conic programs."""

    plan: ClarabelPlan

    def __init__(self, *, presolve: bool = True, verbose: bool = False) -> None:
        self.plan = ClarabelPlan(presolve=presolve, verbose=verbose)

    @property
    def method_id(self) -> str:
        return "clarabel-interior-point"

    @property
    def backend(self) -> str:
        return "clarabel"

    @property
    def capabilities(self) -> ConvexProgramCapabilities:
        return ConvexProgramCapabilities(
            linear_program=True,
            quadratic_program=True,
            conic_program=True,
            dense=True,
            sparse=True,
            matrix_free=False,
            warm_start=False,
            prepared_refresh=True,
            infeasibility_certificates=True,
            implicit_differentiation=False,
            algorithmic_differentiation=False,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (("plan_id", self.plan.plan_id),)


class ConvexSolvePolicy(StrictModule):
    """Composable method, termination, regularization, and resource contract."""

    method: AbstractConvexProgramMethod
    termination: ConvexTermination
    materialization: MaterializationPolicy
    resources: SolveResourcePolicy
    failure: FailurePolicy
    regularization: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractConvexProgramMethod | None = None,
        /,
        *,
        termination: ConvexTermination | None = None,
        regularization: float = 0.0,
        materialization: MaterializationPolicy | None = None,
        resources: SolveResourcePolicy | None = None,
        failure: FailurePolicy | None = None,
    ) -> None:
        method_ = DensePrimalDualQP() if method is None else method
        termination_ = ConvexTermination() if termination is None else termination
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        resources_ = SolveResourcePolicy() if resources is None else resources
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(method_, AbstractConvexProgramMethod):
            raise TypeError("method must be an AbstractConvexProgramMethod or None.")
        if not isinstance(termination_, ConvexTermination):
            raise TypeError("termination must be a ConvexTermination or None.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        if not isinstance(resources_, SolveResourcePolicy):
            raise TypeError("resources must be a SolveResourcePolicy or None.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        regularization_ = float(regularization)
        if not isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        self.method = method_
        self.termination = termination_
        self.materialization = materialization_
        self.resources = resources_
        self.failure = failure_
        self.regularization = regularization_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "convex-solve-policy",
                "method": method_.method_id,
                "method_configuration": dict(method_.configuration),
                "termination": {
                    "absolute": termination_.absolute,
                    "relative": termination_.relative,
                    "primal_infeasible": termination_.primal_infeasible,
                    "dual_infeasible": termination_.dual_infeasible,
                    "maximum_steps": termination_.maximum_steps,
                },
                "regularization": regularization_,
                "materialization": {
                    "max_entries": materialization_.max_entries,
                    "max_bytes": materialization_.max_bytes,
                },
                "resources": {
                    "factorization_bytes": resources_.factorization_bytes,
                    "workspace_bytes": resources_.workspace_bytes,
                    "krylov_basis_bytes": resources_.krylov_basis_bytes,
                    "preconditioner_bytes": resources_.preconditioner_bytes,
                    "recycling_state_bytes": resources_.recycling_state_bytes,
                },
                "failure": failure_.mode,
            }
        )


class ConicGeneralizedDerivativePolicy(StrictModule):
    """One fixed selected cone-projection derivative at classified strata."""

    orthant_zero_value: float = eqx.field(static=True)
    approach_direction: tuple[float, ...] = eqx.field(static=True)
    approach_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        orthant_zero_value: float = 0.5,
        approach_direction: tuple[float, ...] = (),
        approach_scale: float = 1e-6,
    ) -> None:
        zero = float(orthant_zero_value)
        scale = float(approach_scale)
        direction = tuple(float(value) for value in approach_direction)
        if not 0.0 <= zero <= 1.0:
            raise ValueError("orthant_zero_value must lie in [0, 1].")
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("approach_scale must be finite and positive.")
        if any(not isfinite(value) for value in direction):
            raise ValueError("approach_direction must be finite.")
        self.orthant_zero_value = zero
        self.approach_direction = direction
        self.approach_scale = scale


class ConvexDifferentiationPolicy(StrictModule):
    """Explicit derivative of a regular mathematical or executed solution map."""

    mode: ConvexDifferentiationMode = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)
    strict_complementarity_tolerance: float = eqx.field(static=True)
    barrier: float | None = eqx.field(static=True)
    centering_tolerance: float = eqx.field(static=True)
    maximum_centering_steps: int = eqx.field(static=True)

    def __init__(
        self,
        mode: ConvexDifferentiationMode = "active-set-kkt",
        /,
        *,
        active_tolerance: float = 1e-5,
        strict_complementarity_tolerance: float = 1e-4,
        barrier: float | None = None,
        centering_tolerance: float = 1e-8,
        maximum_centering_steps: int = 32,
    ) -> None:
        mode = parse(mode, ConvexDifferentiationMode, "mode")
        active = float(active_tolerance)
        strict = float(strict_complementarity_tolerance)
        centering = float(centering_tolerance)
        steps = int(maximum_centering_steps)
        barrier_ = None if barrier is None else float(barrier)
        if (
            not isfinite(active)
            or active <= 0.0
            or not isfinite(strict)
            or strict <= active
        ):
            raise ValueError(
                "strict_complementarity_tolerance must exceed a finite positive "
                "active_tolerance."
            )
        if not isfinite(centering) or centering <= 0.0:
            raise ValueError("centering_tolerance must be finite and positive.")
        if steps < 1:
            raise ValueError("maximum_centering_steps must be positive.")
        if mode == "barrier-kkt":
            if barrier_ is None or not isfinite(barrier_) or barrier_ <= 0.0:
                raise ValueError(
                    "barrier-kkt differentiation requires a finite positive barrier."
                )
        elif barrier_ is not None:
            raise ValueError("barrier is only valid for barrier-kkt differentiation.")
        self.mode = mode
        self.active_tolerance = active
        self.strict_complementarity_tolerance = strict
        self.barrier = barrier_
        self.centering_tolerance = centering
        self.maximum_centering_steps = steps


__all__ = [
    "AbstractConvexProgramMethod",
    "ClarabelInteriorPoint",
    "ConicGeneralizedDerivativePolicy",
    "ConvexDifferentiationMode",
    "ConvexDifferentiationPolicy",
    "ConvexSolvePolicy",
    "ConvexTermination",
    "DensePrimalDualQP",
    "MPAXr2HPDHG",
    "MPAXraPDHG",
    "NativeConicNewtonRoute",
    "NativeHomogeneousConic",
]

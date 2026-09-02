#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-free permanent-multipole and induced-dipole polarization."""

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._sites import AtomisticInteractionSiteState


def _enum_value(value, enum_type, name, /):
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be {enum_type.__name__} or str.")
    matches = tuple(member for member in enum_type if member.value == value)
    if not matches:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {choices}.")
    return matches[0]


def _finite_positive(value, name, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


class PermanentMultipoleSiteData(StrictModule, NonTrainableState):
    """Fixed-capacity Cartesian multipoles and isotropic polarizabilities."""

    charges: Array
    dipoles: Array
    quadrupoles: Array
    polarizabilities: Array
    damping: Array
    multipole_id: str = eqx.field(static=True)

    def __init__(
        self,
        charges: ArrayLike,
        dipoles: ArrayLike,
        quadrupoles: ArrayLike,
        polarizabilities: ArrayLike,
        damping: ArrayLike,
        /,
    ):
        charge = np.asarray(charges, dtype=float)
        dipole = np.asarray(dipoles, dtype=float)
        quadrupole = np.asarray(quadrupoles, dtype=float)
        polar = np.asarray(polarizabilities, dtype=float)
        damp = np.asarray(damping, dtype=float)
        count = charge.size
        if (
            count == 0
            or charge.shape != (count,)
            or dipole.shape != (count, 3)
            or quadrupole.shape != (count, 3, 3)
            or polar.shape != (count,)
            or damp.shape != (count,)
        ):
            raise ValueError("Multipole arrays have incompatible fixed-capacity shapes.")
        if (
            np.any(~np.isfinite(charge))
            or np.any(~np.isfinite(dipole))
            or np.any(~np.isfinite(quadrupole))
            or np.any(~np.isfinite(polar))
            or np.any(~np.isfinite(damp))
            or np.any(polar < 0.0)
            or np.any(damp <= 0.0)
            or not np.allclose(quadrupole, np.swapaxes(quadrupole, -1, -2))
        ):
            raise ValueError(
                "Multipoles must be finite, quadrupoles symmetric, "
                "polarizabilities nonnegative, and damping positive."
            )
        (
            self.charges,
            self.dipoles,
            self.quadrupoles,
            self.polarizabilities,
            self.damping,
        ) = (jnp.asarray(value) for value in (charge, dipole, quadrupole, polar, damp))
        self.multipole_id = canonical_fingerprint(
            {
                "kind": "permanent-multipoles",
                "arrays": array_tree_fingerprint(
                    {
                        "q": charge,
                        "mu": dipole,
                        "Q": quadrupole,
                        "alpha": polar,
                        "damping": damp,
                    }
                ),
            }
        )

    @property
    def site_capacity(self) -> int:
        return self.charges.shape[0]


class PolarizationScaleData(StrictModule, NonTrainableState):
    """Independent d-, p-, and u-field pair scaling.

    ``direct`` scales the permanent field used by a direct-response predictor,
    ``polarization`` scales the permanent right-hand-side field, and ``mutual``
    scales the induced-dipole field in the matrix-free operator.
    """

    direct: Array
    polarization: Array
    mutual: Array
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        direct: ArrayLike,
        polarization: ArrayLike,
        mutual: ArrayLike,
        /,
    ):
        values = tuple(
            np.asarray(value, dtype=float) for value in (direct, polarization, mutual)
        )
        shape = values[0].shape
        if (
            len(shape) != 2
            or shape[0] == 0
            or shape[0] != shape[1]
            or any(value.shape != shape for value in values)
        ):
            raise ValueError("Polarization scaling matrices must share shape (N,N).")
        if any(
            np.any(~np.isfinite(value))
            or np.any(value < 0.0)
            or np.any(value > 1.0)
            or not np.allclose(value, value.T)
            or not np.allclose(np.diag(value), 0.0)
            for value in values
        ):
            raise ValueError(
                "Polarization scaling must be finite, symmetric, in [0,1], "
                "and zero on the diagonal."
            )
        self.direct, self.polarization, self.mutual = (
            jnp.asarray(value) for value in values
        )
        self.scale_id = canonical_fingerprint(
            {
                "kind": "polarization-scaling",
                "arrays": array_tree_fingerprint(
                    {"d": values[0], "p": values[1], "u": values[2]}
                ),
            }
        )

    @classmethod
    def unscaled(cls, site_capacity: int, /) -> PolarizationScaleData:
        capacity = int(site_capacity)
        if capacity <= 0:
            raise ValueError("site_capacity must be positive.")
        off_diagonal = np.ones((capacity, capacity), dtype=float) - np.eye(capacity)
        return cls(off_diagonal, off_diagonal, off_diagonal)


class PolarizationSolverKind(StrEnum):
    """Supported fixed-capacity induced-dipole algorithms."""

    PCG = "pcg"
    TCG = "tcg"


class PolarizationPreconditionerKind(StrEnum):
    """Local matrix-free preconditioner choices."""

    IDENTITY = "identity"
    POLARIZABILITY = "polarizability"


class PolarizationOperatorPlan(StrictModule, NonTrainableState):
    """Plan for a matrix-free ``alpha^-1 - T`` polarization operator."""

    minimum_distance: float = eqx.field(static=True)
    periodic_plan: MultipolePMEPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        minimum_distance: float = 1.0e-8,
        periodic_plan: MultipolePMEPlan | None = None,
    ):
        distance = _finite_positive(minimum_distance, "minimum_distance")
        if periodic_plan is not None and not isinstance(periodic_plan, MultipolePMEPlan):
            raise TypeError("periodic_plan must be MultipolePMEPlan or None.")
        self.minimum_distance, self.periodic_plan = distance, periodic_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-operator-plan",
                "minimum_distance": distance.hex(),
                "periodic": None if periodic_plan is None else periodic_plan.plan_id,
            }
        )

    def prepare(
        self,
        multipoles: PermanentMultipoleSiteData,
        /,
        *,
        scaling: PolarizationScaleData | None = None,
        active_mask: ArrayLike | None = None,
    ) -> PreparedPolarizationOperator:
        if not isinstance(multipoles, PermanentMultipoleSiteData):
            raise TypeError("multipoles must be PermanentMultipoleSiteData.")
        capacity = multipoles.site_capacity
        scale = PolarizationScaleData.unscaled(capacity) if scaling is None else scaling
        if not isinstance(scale, PolarizationScaleData):
            raise TypeError("scaling must be PolarizationScaleData or None.")
        if scale.direct.shape != (capacity, capacity):
            raise ValueError("Scaling and multipole capacities differ.")
        if active_mask is None:
            active = jnp.ones((capacity,), dtype=bool)
            active_id = "all-active"
        else:
            host_active = np.asarray(active_mask)
            if host_active.shape != (capacity,) or host_active.dtype != np.bool_:
                raise ValueError("active_mask must be a boolean array with shape (N,).")
            active = jnp.asarray(host_active)
            active_id = array_tree_fingerprint(host_active)
        return PreparedPolarizationOperator(self, multipoles, scale, active, active_id)


class PreparedPolarizationOperator(StrictModule, NonTrainableState):
    """Prepared fixed-capacity matrix-free polarization operator."""

    plan: PolarizationOperatorPlan
    multipoles: PermanentMultipoleSiteData
    scaling: PolarizationScaleData
    active_mask: Array
    site_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, multipoles, scaling, active_mask, active_id, /):
        self.plan, self.multipoles, self.scaling, self.active_mask = (
            plan,
            multipoles,
            scaling,
            active_mask,
        )
        self.site_capacity = multipoles.site_capacity
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-polarization-operator",
                "plan": plan.plan_id,
                "multipoles": multipoles.multipole_id,
                "scaling": scaling.scale_id,
                "active": active_id,
            }
        )

    def apply(
        self,
        positions: ArrayLike,
        induced_dipoles: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
    ) -> PolarizationOperatorResult:
        coordinate = _positions(positions, self.site_capacity)
        induced = jnp.asarray(induced_dipoles, dtype=coordinate.dtype)
        if induced.shape != (self.site_capacity, 3):
            raise ValueError("induced_dipoles must have shape (N,3).")
        cell = _periodic_cell(self, cell_vectors)
        return _operator_result(self, coordinate, induced, cell)


class PolarizationOperatorResult(StrictModule):
    """Fields, operator action, and geometry evidence for one application."""

    action: Array
    d_field: Array
    p_field: Array
    u_field: Array
    minimum_pair_distance: Array
    periodic_contract_valid: Array
    finite: Array
    successful: Array


class PolarizationPreconditionerPlan(StrictModule, NonTrainableState):
    """Plan for an identity or local polarizability preconditioner."""

    kind: PolarizationPreconditionerKind = eqx.field(static=True)
    diagonal_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PolarizationPreconditionerKind | str = (
            PolarizationPreconditionerKind.POLARIZABILITY
        ),
        /,
        *,
        diagonal_floor: float = 1.0e-12,
    ):
        kind_ = _enum_value(kind, PolarizationPreconditionerKind, "kind")
        floor = _finite_positive(diagonal_floor, "diagonal_floor")
        self.kind, self.diagonal_floor = kind_, floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-preconditioner-plan",
                "preconditioner": kind_.value,
                "diagonal_floor": floor.hex(),
            }
        )

    def prepare(
        self, operator: PreparedPolarizationOperator, /
    ) -> PreparedPolarizationPreconditioner:
        if not isinstance(operator, PreparedPolarizationOperator):
            raise TypeError("operator must be PreparedPolarizationOperator.")
        return PreparedPolarizationPreconditioner(self, operator)


class PreparedPolarizationPreconditioner(StrictModule, NonTrainableState):
    """Prepared local preconditioner sharing an operator's fixed capacity."""

    plan: PolarizationPreconditionerPlan
    operator: PreparedPolarizationOperator
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, operator, /):
        self.plan, self.operator = plan, operator
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-polarization-preconditioner",
                "plan": plan.plan_id,
                "operator": operator.prepared_id,
            }
        )

    def apply(self, residual: ArrayLike, /) -> PolarizationPreconditionerResult:
        value = jnp.asarray(residual)
        if value.shape != (self.operator.site_capacity, 3):
            raise ValueError("residual must have shape (N,3).")
        result = _apply_preconditioner(self, value)
        finite = jnp.all(jnp.isfinite(result))
        return PolarizationPreconditionerResult(
            jnp.where(finite, result, jnp.nan), finite, finite
        )


class PolarizationPreconditionerResult(StrictModule):
    """One preconditioner application and fail-closed evidence."""

    value: Array
    finite: Array
    successful: Array


class PolarizationPredictorPlan(StrictModule, NonTrainableState):
    """Two-frame induced-dipole history predictor."""

    history_coefficient: float = eqx.field(static=True)
    direct_fallback: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        history_coefficient: float = 1.0,
        direct_fallback: bool = True,
    ):
        coefficient = float(history_coefficient)
        if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 2.0:
            raise ValueError("history_coefficient must be finite and in [0,2].")
        self.history_coefficient = coefficient
        self.direct_fallback = bool(direct_fallback)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-predictor-plan",
                "history_coefficient": coefficient.hex(),
                "direct_fallback": self.direct_fallback,
            }
        )


class PolarizationPredictorState(StrictModule):
    """Fixed-shape two-frame warm-start history."""

    history: Array
    valid_count: Array
    prepared_id: str = eqx.field(static=True)


class PolarizationSolverPlan(StrictModule, NonTrainableState):
    """PCG or fixed-cost TCG solver plan."""

    kind: PolarizationSolverKind = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tcg_order: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    breakdown_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PolarizationSolverKind | str = PolarizationSolverKind.PCG,
        /,
        *,
        maximum_iterations: int = 100,
        tcg_order: int = 3,
        tolerance: float = 1.0e-8,
        force_tolerance: float | None = None,
        breakdown_tolerance: float = 1.0e-12,
    ):
        kind_ = _enum_value(kind, PolarizationSolverKind, "kind")
        iterations, order = int(maximum_iterations), int(tcg_order)
        tolerance_ = _finite_positive(tolerance, "tolerance")
        force = (
            tolerance_
            if force_tolerance is None
            else _finite_positive(force_tolerance, "force_tolerance")
        )
        breakdown = _finite_positive(breakdown_tolerance, "breakdown_tolerance")
        if iterations <= 0 or order <= 0:
            raise ValueError("maximum_iterations and tcg_order must both be positive.")
        (
            self.kind,
            self.maximum_iterations,
            self.tcg_order,
            self.tolerance,
            self.force_tolerance,
            self.breakdown_tolerance,
        ) = (kind_, iterations, order, tolerance_, force, breakdown)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-solver-plan",
                "solver": kind_.value,
                "maximum_iterations": iterations,
                "tcg_order": order,
                "tolerance": tolerance_.hex(),
                "force_tolerance": force.hex(),
                "breakdown_tolerance": breakdown.hex(),
            }
        )

    def prepare(
        self,
        operator: PreparedPolarizationOperator,
        preconditioner: PreparedPolarizationPreconditioner,
        /,
        *,
        predictor: PolarizationPredictorPlan | None = None,
    ) -> PreparedPolarizationSolver:
        if not isinstance(operator, PreparedPolarizationOperator):
            raise TypeError("operator must be PreparedPolarizationOperator.")
        if not isinstance(preconditioner, PreparedPolarizationPreconditioner):
            raise TypeError("preconditioner must be PreparedPolarizationPreconditioner.")
        if preconditioner.operator.prepared_id != operator.prepared_id:
            raise ValueError("Preconditioner and operator preparations differ.")
        predictor_ = PolarizationPredictorPlan() if predictor is None else predictor
        if not isinstance(predictor_, PolarizationPredictorPlan):
            raise TypeError("predictor must be PolarizationPredictorPlan or None.")
        return PreparedPolarizationSolver(
            self,
            operator,
            preconditioner,
            predictor_,
            self.plan_id,
        )


class PolarizationPlan(StrictModule, NonTrainableState):
    """Compatibility facade composing operator, preconditioner, and solver plans."""

    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    operator: PolarizationOperatorPlan
    preconditioner: PolarizationPreconditionerPlan
    solver: PolarizationSolverPlan
    predictor: PolarizationPredictorPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 100,
        tolerance: float = 1e-8,
        relaxation: float = 0.7,
        solver_kind: PolarizationSolverKind | str = PolarizationSolverKind.PCG,
        tcg_order: int = 3,
        force_tolerance: float | None = None,
        preconditioner_kind: PolarizationPreconditionerKind | str = (
            PolarizationPreconditionerKind.POLARIZABILITY
        ),
        predictor: PolarizationPredictorPlan | None = None,
        periodic_plan: MultipolePMEPlan | None = None,
        minimum_distance: float = 1.0e-8,
    ):
        relaxation_ = float(relaxation)
        if not np.isfinite(relaxation_) or not 0.0 < relaxation_ <= 1.0:
            raise ValueError("relaxation must be finite and in (0,1].")
        operator = PolarizationOperatorPlan(
            minimum_distance=minimum_distance, periodic_plan=periodic_plan
        )
        preconditioner = PolarizationPreconditionerPlan(preconditioner_kind)
        solver = PolarizationSolverPlan(
            solver_kind,
            maximum_iterations=maximum_iterations,
            tcg_order=tcg_order,
            tolerance=tolerance,
            force_tolerance=force_tolerance,
        )
        predictor_ = (
            PolarizationPredictorPlan(history_coefficient=relaxation_)
            if predictor is None
            else predictor
        )
        if not isinstance(predictor_, PolarizationPredictorPlan):
            raise TypeError("predictor must be PolarizationPredictorPlan or None.")
        (
            self.maximum_iterations,
            self.tolerance,
            self.relaxation,
            self.operator,
            self.preconditioner,
            self.solver,
            self.predictor,
        ) = (
            solver.maximum_iterations,
            solver.tolerance,
            relaxation_,
            operator,
            preconditioner,
            solver,
            predictor_,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarization-plan",
                "operator": operator.plan_id,
                "preconditioner": preconditioner.plan_id,
                "solver": solver.plan_id,
                "predictor": predictor_.plan_id,
                "relaxation": relaxation_.hex(),
            }
        )

    def prepare(
        self,
        multipoles: PermanentMultipoleSiteData,
        /,
        *,
        scaling: PolarizationScaleData | None = None,
        active_mask: ArrayLike | None = None,
    ) -> PreparedPolarizationSolver:
        operator = self.operator.prepare(
            multipoles, scaling=scaling, active_mask=active_mask
        )
        preconditioner = self.preconditioner.prepare(operator)
        return PreparedPolarizationSolver(
            self.solver,
            operator,
            preconditioner,
            self.predictor,
            self.plan_id,
        )


class PolarizationState(StrictModule):
    """Induced dipoles and explicit convergence/force-validity evidence."""

    induced_dipoles: Array
    residual: Array
    iterations: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    relative_residual: Array
    force_valid: Array
    finite: Array
    solver_kind: str = eqx.field(static=True)

    def __init__(
        self,
        induced_dipoles,
        residual,
        iterations,
        converged,
        successful,
        plan_id,
        *,
        relative_residual=None,
        force_valid=None,
        finite=None,
        solver_kind="legacy",
    ):
        residual_ = jnp.asarray(residual)
        finite_ = (
            jnp.all(jnp.isfinite(induced_dipoles)) & jnp.isfinite(residual_)
            if finite is None
            else jnp.asarray(finite)
        )
        self.induced_dipoles = jnp.asarray(induced_dipoles)
        self.residual = residual_
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.converged = jnp.asarray(converged)
        self.successful = jnp.asarray(successful)
        self.plan_id = str(plan_id)
        self.relative_residual = (
            residual_ if relative_residual is None else jnp.asarray(relative_residual)
        )
        self.force_valid = (
            jnp.asarray(converged) if force_valid is None else jnp.asarray(force_valid)
        )
        self.finite = finite_
        self.solver_kind = str(solver_kind)


class PolarizationSolveResult(StrictModule):
    """Solver state, next warm start, fields, and breakdown evidence."""

    state: PolarizationState
    predictor_state: PolarizationPredictorState
    operator: PolarizationOperatorResult
    initial_dipoles: Array
    breakdown: Array
    successful: Array


class PreparedPolarizationSolver(StrictModule, NonTrainableState):
    """Prepared fixed-capacity polarization solve runtime."""

    plan: PolarizationSolverPlan
    operator: PreparedPolarizationOperator
    preconditioner: PreparedPolarizationPreconditioner
    predictor: PolarizationPredictorPlan
    result_plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, operator, preconditioner, predictor, result_plan_id, /):
        self.plan, self.operator, self.preconditioner, self.predictor = (
            plan,
            operator,
            preconditioner,
            predictor,
        )
        self.result_plan_id = str(result_plan_id)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-polarization-solver",
                "plan": plan.plan_id,
                "operator": operator.prepared_id,
                "preconditioner": preconditioner.prepared_id,
                "predictor": predictor.plan_id,
                "result_plan": self.result_plan_id,
            }
        )

    def initial_predictor_state(self, /) -> PolarizationPredictorState:
        capacity = self.operator.site_capacity
        return PolarizationPredictorState(
            jnp.zeros((2, capacity, 3), dtype=self.operator.multipoles.dipoles.dtype),
            jnp.zeros((), dtype=jnp.int32),
            self.prepared_id,
        )

    def solve(
        self,
        positions: ArrayLike,
        /,
        *,
        predictor_state: PolarizationPredictorState | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> PolarizationSolveResult:
        coordinate = _positions(positions, self.operator.site_capacity)
        cell = _periodic_cell(self.operator, cell_vectors)
        state = (
            self.initial_predictor_state() if predictor_state is None else predictor_state
        )
        if not isinstance(state, PolarizationPredictorState):
            raise TypeError("predictor_state must be PolarizationPredictorState or None.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Predictor state belongs to a different prepared solver.")
        if state.history.shape != (2, self.operator.site_capacity, 3):
            raise ValueError("Predictor history has an incompatible fixed shape.")
        return _solve_prepared(self, coordinate, state, cell)


class PolarizationDifferentiationEvidence(StrictModule):
    """Qualification evidence for envelope-theorem force differentiation."""

    mode: str = eqx.field(static=True)
    residual: Array
    force_tolerance: Array
    fixed_topology: Array
    periodic_contract_valid: Array
    finite: Array
    successful: Array


class PolarizationEvaluation(StrictModule):
    """Fail-closed energy, force, solver state, and derivative evidence."""

    energy: Array
    forces: Array
    state: PolarizationState
    successful: Array
    evidence: PolarizationDifferentiationEvidence | None

    def __init__(self, energy, forces, state, successful, *, evidence=None):
        self.energy = jnp.asarray(energy)
        self.forces = jnp.asarray(forces)
        self.state = state
        self.successful = jnp.asarray(successful)
        self.evidence = evidence


def _positions(value, capacity, /):
    coordinate = jnp.asarray(value)
    if coordinate.shape != (capacity, 3):
        raise ValueError("positions must have the prepared fixed shape (N,3).")
    if not jnp.issubdtype(coordinate.dtype, jnp.floating):
        coordinate = coordinate.astype(jnp.float32)
    return coordinate


def _periodic_cell(operator, cell_vectors, /):
    periodic = operator.plan.periodic_plan is not None
    if periodic and cell_vectors is None:
        raise ValueError("Periodic polarization requires cell_vectors.")
    if not periodic and cell_vectors is not None:
        raise ValueError(
            "cell_vectors require a PolarizationOperatorPlan with periodic_plan."
        )
    if cell_vectors is None:
        return None
    cell = jnp.asarray(cell_vectors)
    if cell.shape != (3, 3):
        raise ValueError("cell_vectors must have shape (3,3).")
    if not jnp.issubdtype(cell.dtype, jnp.floating):
        cell = cell.astype(jnp.float32)
    return cell


def _pair_geometry(operator, positions, cell, /):
    displacement = positions[:, None, :] - positions[None, :, :]
    periodic_valid = jnp.asarray(True)
    if cell is not None:
        determinant = jnp.sum(cell[0] * jnp.cross(cell[1], cell[2]))
        inverse = jnp.stack(
            (
                jnp.cross(cell[1], cell[2]),
                jnp.cross(cell[2], cell[0]),
                jnp.cross(cell[0], cell[1]),
            ),
            axis=1,
        ) / jnp.where(jnp.abs(determinant) > 0.0, determinant, 1.0)
        fractional = contract("ijd,dk->ijk", displacement, inverse)
        fractional = fractional - jax.lax.stop_gradient(jnp.round(fractional))
        displacement = contract("ijk,kd->ijd", fractional, cell)
        gram = contract("ik,jk->ij", cell, cell)
        row_squared_norm = jnp.diag(gram)
        row_norm_product = jnp.sqrt(
            jnp.maximum(
                row_squared_norm[:, None] * row_squared_norm[None, :],
                jnp.finfo(cell.dtype).tiny,
            )
        )
        off_diagonal = gram - jnp.diag(row_squared_norm)
        normalized_off_diagonal = jnp.abs(off_diagonal) / row_norm_product
        orthogonal = jnp.all(row_squared_norm > operator.plan.minimum_distance**2) & (
            jnp.max(normalized_off_diagonal) <= 64.0 * jnp.finfo(cell.dtype).eps
        )
        periodic_valid = (
            jnp.all(jnp.isfinite(cell))
            & jnp.isfinite(determinant)
            & (jnp.abs(determinant) > operator.plan.minimum_distance**3)
            & orthogonal
        )
    identity = jnp.eye(operator.site_capacity, dtype=bool)
    active_pair = (
        operator.active_mask[:, None] & operator.active_mask[None, :] & ~identity
    )
    scaled_pair = (
        (operator.scaling.direct > 0.0)
        | (operator.scaling.polarization > 0.0)
        | (operator.scaling.mutual > 0.0)
    )
    participating = active_pair & scaled_pair
    pair = active_pair if cell is not None else participating
    distance2 = jnp.sum(displacement * displacement, axis=-1)
    positive = distance2 > operator.plan.minimum_distance**2
    safe2 = jnp.where(pair & positive, distance2, 1.0)
    distance = jnp.sqrt(safe2)
    direction = displacement / distance[..., None]
    minimum = jnp.min(jnp.where(pair, jnp.sqrt(jnp.maximum(distance2, 0.0)), jnp.inf))
    distinct = jnp.all(jnp.where(pair, positive, True))
    return (
        displacement,
        direction,
        distance,
        safe2,
        pair,
        minimum,
        distinct,
        periodic_valid,
    )


def _damping(multipoles, distance, /):
    exponent = (
        jnp.sqrt(multipoles.damping[:, None] * multipoles.damping[None, :]) * distance**3
    )
    return -jnp.expm1(-exponent)


def _unscreened_permanent_fields(displacement, distance, multipoles, /):
    direction = displacement / distance[..., None]
    safe2 = distance**2
    charge = multipoles.charges[None, :, None] * direction / safe2[..., None]
    dipole_dot = contract("ijd,jd->ij", direction, multipoles.dipoles)
    dipole = (
        3.0 * direction * dipole_dot[..., None] - multipoles.dipoles[None, :, :]
    ) / (distance**3)[..., None]
    qr = contract("jab,ijb->ija", multipoles.quadrupoles, displacement)
    rqr = contract("ija,ija->ij", displacement, qr)
    trace = jnp.trace(multipoles.quadrupoles, axis1=-2, axis2=-1)
    quadrupole = (
        7.5 * displacement * rqr[..., None] / (distance**7)[..., None]
        - 3.0 * qr / (distance**5)[..., None]
        - 1.5 * trace[None, :, None] * displacement / (distance**5)[..., None]
    )
    return charge, dipole, quadrupole


def _screened_radial_derivatives(alpha, distance, /):
    argument = alpha * distance
    complementary = jsp.erfc(argument)
    gaussian = (
        2.0
        * alpha
        / jnp.sqrt(jnp.asarray(jnp.pi, dtype=distance.dtype))
        * jnp.exp(-(argument**2))
    )
    first = -complementary / distance**2 - gaussian / distance
    second = (
        2.0 * complementary / distance**3
        + 2.0 * gaussian / distance**2
        + 2.0 * alpha**2 * gaussian
    )
    third = (
        -6.0 * complementary / distance**4
        - 6.0 * gaussian / distance**3
        - 4.0 * alpha**2 * gaussian / distance
        - 4.0 * alpha**4 * distance * gaussian
    )
    return first, second, third


def _screened_permanent_fields(displacement, distance, multipoles, alpha, /):
    first, second, third = _screened_radial_derivatives(alpha, distance)
    charge = (
        -first[..., None]
        / distance[..., None]
        * multipoles.charges[None, :, None]
        * displacement
    )
    radial_a = second / distance**2 - first / distance**3
    radial_b = first / distance
    dipole_dot = contract("ijd,jd->ij", displacement, multipoles.dipoles)
    dipole = (
        radial_a[..., None] * displacement * dipole_dot[..., None]
        + radial_b[..., None] * multipoles.dipoles[None, :, :]
    )
    qr = contract("jab,ijb->ija", multipoles.quadrupoles, displacement)
    rqr = contract("ija,ija->ij", displacement, qr)
    trace = jnp.trace(multipoles.quadrupoles, axis1=-2, axis2=-1)
    radial_a_prime_over_r = (
        third / distance**3 - 3.0 * second / distance**4 + 3.0 * first / distance**5
    )
    quadrupole = -0.5 * (
        radial_a_prime_over_r[..., None] * displacement * rqr[..., None]
        + 2.0 * radial_a[..., None] * qr
        + radial_a[..., None] * displacement * trace[None, :, None]
    )
    return charge, dipole, quadrupole


def _permanent_field(operator, geometry, scale, /):
    displacement, _, distance, _, pair, _, _, _ = geometry
    multipoles = operator.multipoles
    unscreened = _unscreened_permanent_fields(displacement, distance, multipoles)
    damping_scale = _damping(multipoles, distance) * scale
    if operator.plan.periodic_plan is None:
        field = damping_scale[..., None] * sum(unscreened)
    else:
        screened = _screened_permanent_fields(
            displacement,
            distance,
            multipoles,
            operator.plan.periodic_plan.alpha,
        )
        field = sum(screened) + (damping_scale - 1.0)[..., None] * sum(unscreened)
    return jnp.sum(
        jnp.where(pair[..., None], field, 0.0),
        axis=1,
    )


def _induced_field(operator, geometry, induced, /):
    displacement, _, distance, _, pair, _, _, _ = geometry
    polarizable = operator.active_mask & (operator.multipoles.polarizabilities > 0.0)
    source = jnp.where(polarizable[:, None], induced, 0.0)
    direction = displacement / distance[..., None]
    dot = contract("ijd,jd->ij", direction, source)
    unscreened = (3.0 * direction * dot[..., None] - source[None, :, :]) / (distance**3)[
        ..., None
    ]
    damping_scale = _damping(operator.multipoles, distance) * operator.scaling.mutual
    if operator.plan.periodic_plan is None:
        field = damping_scale[..., None] * unscreened
    else:
        first, second, _ = _screened_radial_derivatives(
            operator.plan.periodic_plan.alpha, distance
        )
        radial_a = second / distance**2 - first / distance**3
        radial_b = first / distance
        displacement_dot = contract("ijd,jd->ij", displacement, source)
        screened = (
            radial_a[..., None] * displacement * displacement_dot[..., None]
            + radial_b[..., None] * source[None, :, :]
        )
        field = screened + (damping_scale - 1.0)[..., None] * unscreened
    return jnp.sum(
        jnp.where(pair[..., None], field, 0.0),
        axis=1,
    )


def _reciprocal_field(plan, positions, charges, dipoles, quadrupoles, active, cell, /):
    determinant = jnp.sum(cell[0] * jnp.cross(cell[1], cell[2]))
    inverse = jnp.stack(
        (
            jnp.cross(cell[1], cell[2]),
            jnp.cross(cell[2], cell[0]),
            jnp.cross(cell[0], cell[1]),
        ),
        axis=1,
    ) / jnp.where(jnp.abs(determinant) > 0.0, determinant, 1.0)
    integer_axes = tuple(jnp.fft.fftfreq(size) * size for size in plan.grid_shape)
    modes = jnp.stack(jnp.meshgrid(*integer_axes, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    wave = 2.0 * jnp.pi * contract("gi,ji->gj", modes, inverse)
    squared = jnp.sum(wave * wave, axis=-1)
    kernel = jnp.where(
        squared > 0.0,
        jnp.exp(-squared / (4.0 * plan.alpha**2))
        / jnp.where(squared > 0.0, squared, 1.0),
        0.0,
    )
    phase = contract("nd,gd->ng", positions, wave)
    source_phase = jnp.exp(-1.0j * phase)
    dipole_projection = contract("gd,nd->ng", wave, dipoles)
    quadrupole_projection = contract("gd,nde,ge->ng", wave, quadrupoles, wave)
    source = (
        (charges[:, None] - 1.0j * dipole_projection - 0.5 * quadrupole_projection)
        * source_phase
        * active[:, None]
    )
    structure = jnp.sum(source, axis=0)
    mode_field = (
        -1.0j
        * jnp.exp(1.0j * phase)[:, :, None]
        * structure[None, :, None]
        * wave[None, :, :]
    )
    prefactor = (
        4.0 * jnp.pi / jnp.abs(jnp.where(jnp.abs(determinant) > 0.0, determinant, 1.0))
    )
    return prefactor * jnp.sum(kernel[None, :, None] * jnp.real(mode_field), axis=1)


def _dipole_self_field(plan, dipoles, /):
    coefficient = (
        4.0 * plan.alpha**3 / (3.0 * jnp.sqrt(jnp.asarray(jnp.pi, dtype=dipoles.dtype)))
    )
    return coefficient * dipoles


def _operator_result(operator, positions, induced, cell, /):
    geometry = _pair_geometry(operator, positions, cell)
    d_field = _permanent_field(operator, geometry, operator.scaling.direct)
    p_field = _permanent_field(operator, geometry, operator.scaling.polarization)
    u_field = _induced_field(operator, geometry, induced)
    polarizable = operator.active_mask & (operator.multipoles.polarizabilities > 0.0)
    induced_source = jnp.where(polarizable[:, None], induced, 0.0)
    if cell is not None:
        zeros_charge = jnp.zeros_like(operator.multipoles.charges)
        zeros_quadrupole = jnp.zeros_like(operator.multipoles.quadrupoles)
        permanent_reciprocal = _reciprocal_field(
            operator.plan.periodic_plan,
            positions,
            operator.multipoles.charges,
            operator.multipoles.dipoles,
            operator.multipoles.quadrupoles,
            operator.active_mask,
            cell,
        )
        permanent_reciprocal = permanent_reciprocal + _dipole_self_field(
            operator.plan.periodic_plan,
            jnp.where(
                operator.active_mask[:, None],
                operator.multipoles.dipoles,
                0.0,
            ),
        )
        induced_reciprocal = _reciprocal_field(
            operator.plan.periodic_plan,
            positions,
            zeros_charge,
            induced_source,
            zeros_quadrupole,
            polarizable,
            cell,
        )
        induced_reciprocal = induced_reciprocal + _dipole_self_field(
            operator.plan.periodic_plan, induced_source
        )
        d_field = d_field + permanent_reciprocal
        p_field = p_field + permanent_reciprocal
        u_field = u_field + induced_reciprocal
    inverse_alpha = jnp.where(
        polarizable,
        1.0
        / jnp.where(
            operator.multipoles.polarizabilities > 0.0,
            operator.multipoles.polarizabilities,
            1.0,
        ),
        1.0,
    )
    action = inverse_alpha[:, None] * induced - u_field
    action = jnp.where(polarizable[:, None], action, induced)
    d_field = jnp.where(polarizable[:, None], d_field, 0.0)
    p_field = jnp.where(polarizable[:, None], p_field, 0.0)
    u_field = jnp.where(polarizable[:, None], u_field, 0.0)
    finite = (
        jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(induced))
        & jnp.all(jnp.isfinite(action))
        & jnp.all(jnp.isfinite(d_field))
        & jnp.all(jnp.isfinite(p_field))
        & jnp.all(jnp.isfinite(u_field))
    )
    successful = finite & geometry[6] & geometry[7]
    return PolarizationOperatorResult(
        action,
        d_field,
        p_field,
        u_field,
        geometry[5],
        geometry[7],
        finite,
        successful,
    )


def _action(operator, positions, induced, cell, geometry, /):
    u_field = _induced_field(operator, geometry, induced)
    polarizable = operator.active_mask & (operator.multipoles.polarizabilities > 0.0)
    induced_source = jnp.where(polarizable[:, None], induced, 0.0)
    if cell is not None:
        u_field = u_field + _reciprocal_field(
            operator.plan.periodic_plan,
            positions,
            jnp.zeros_like(operator.multipoles.charges),
            induced_source,
            jnp.zeros_like(operator.multipoles.quadrupoles),
            polarizable,
            cell,
        )
        u_field = u_field + _dipole_self_field(
            operator.plan.periodic_plan, induced_source
        )
    inverse_alpha = jnp.where(
        polarizable,
        1.0
        / jnp.where(
            operator.multipoles.polarizabilities > 0.0,
            operator.multipoles.polarizabilities,
            1.0,
        ),
        1.0,
    )
    action = inverse_alpha[:, None] * induced - u_field
    action = jnp.where(polarizable[:, None], action, induced)
    successful = (
        geometry[6]
        & geometry[7]
        & jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(induced))
        & jnp.all(jnp.isfinite(action))
    )
    return action, successful


def _apply_preconditioner(preconditioner, residual, /):
    polarizable = preconditioner.operator.active_mask & (
        preconditioner.operator.multipoles.polarizabilities > 0.0
    )
    if preconditioner.plan.kind is PolarizationPreconditionerKind.IDENTITY:
        scale = jnp.ones_like(preconditioner.operator.multipoles.polarizabilities)
    else:
        scale = jnp.where(
            polarizable,
            jnp.maximum(
                preconditioner.operator.multipoles.polarizabilities,
                preconditioner.plan.diagonal_floor,
            ),
            1.0,
        )
    return scale[:, None] * residual


def _norm(value, /):
    return jnp.sqrt(jnp.maximum(contract("nd,nd->", value, value), 0.0))


def _linear_solve(prepared, positions, right, initial, cell, /):
    geometry = _pair_geometry(prepared.operator, positions, cell)
    initial_action, action_successful = _action(
        prepared.operator, positions, initial, cell, geometry
    )
    residual = right - initial_action
    preconditioned = _apply_preconditioner(prepared.preconditioner, residual)
    rz = contract("nd,nd->", residual, preconditioned)
    initial_norm = _norm(residual)
    right_norm = _norm(right)
    relative = initial_norm / jnp.maximum(right_norm, jnp.finfo(right.dtype).eps)
    initial_finite = (
        action_successful
        & jnp.all(jnp.isfinite(right))
        & jnp.all(jnp.isfinite(preconditioned))
        & jnp.isfinite(rz)
        & (rz >= 0.0)
    )
    carry = (
        initial,
        residual,
        preconditioned,
        preconditioned,
        rz,
        initial_norm,
        relative,
        jnp.zeros((), dtype=jnp.int32),
        ~initial_finite,
    )
    loop_count = (
        prepared.plan.maximum_iterations
        if prepared.plan.kind is PolarizationSolverKind.PCG
        else prepared.plan.tcg_order
    )

    def iteration(_, values):
        value, residual_, z, direction, rz_, norm_, relative_, count, breakdown = values
        action, action_successful = _action(
            prepared.operator, positions, direction, cell, geometry
        )
        denominator = contract("nd,nd->", direction, action)
        denominator_scale = jnp.maximum(
            _norm(direction) * _norm(action), jnp.finfo(action.dtype).tiny
        )
        needs_step = norm_ > prepared.plan.tolerance
        denominator_valid = (
            jnp.isfinite(denominator)
            & (denominator > prepared.plan.breakdown_tolerance * denominator_scale)
            & action_successful
        )
        active = needs_step & ~breakdown & denominator_valid
        alpha = jnp.where(active, rz_ / denominator, 0.0)
        candidate_value = value + alpha * direction
        candidate_residual = residual_ - alpha * action
        candidate_z = _apply_preconditioner(prepared.preconditioner, candidate_residual)
        candidate_rz = contract("nd,nd->", candidate_residual, candidate_z)
        rz_scale = jnp.maximum(
            _norm(residual_) * _norm(z), jnp.finfo(residual_.dtype).tiny
        )
        beta = jnp.where(
            active & (rz_ > prepared.plan.breakdown_tolerance * rz_scale),
            candidate_rz / rz_,
            0.0,
        )
        candidate_direction = candidate_z + beta * direction
        candidate_norm = _norm(candidate_residual)
        candidate_relative = candidate_norm / jnp.maximum(
            right_norm, jnp.finfo(right.dtype).eps
        )
        next_breakdown = breakdown | (needs_step & ~breakdown & ~denominator_valid)
        return (
            jnp.where(active, candidate_value, value),
            jnp.where(active, candidate_residual, residual_),
            jnp.where(active, candidate_z, z),
            jnp.where(active, candidate_direction, direction),
            jnp.where(active, candidate_rz, rz_),
            jnp.where(active, candidate_norm, norm_),
            jnp.where(active, candidate_relative, relative_),
            count + active.astype(jnp.int32),
            next_breakdown,
        )

    return jax.lax.fori_loop(0, loop_count, iteration, carry)


def _predict(prepared, operator_result, predictor_state, /):
    alpha = prepared.operator.multipoles.polarizabilities[:, None]
    direct = alpha * operator_result.d_field
    fallback = direct if prepared.predictor.direct_fallback else jnp.zeros_like(direct)
    previous = predictor_state.history[0]
    extrapolated = previous + prepared.predictor.history_coefficient * (
        previous - predictor_state.history[1]
    )
    return jnp.where(
        predictor_state.valid_count >= 2,
        extrapolated,
        jnp.where(predictor_state.valid_count >= 1, previous, fallback),
    )


def _solve_prepared(prepared, positions, predictor_state, cell, /):
    zero = jnp.zeros_like(prepared.operator.multipoles.dipoles)
    fields = _operator_result(prepared.operator, positions, zero, cell)
    initial = _predict(prepared, fields, predictor_state)
    (
        induced,
        _,
        final_z,
        _,
        _,
        residual,
        relative,
        iterations,
        breakdown,
    ) = _linear_solve(prepared, positions, fields.p_field, initial, cell)
    final_operator = _operator_result(prepared.operator, positions, induced, cell)
    true_residual_vector = fields.p_field - final_operator.action
    residual = _norm(true_residual_vector)
    relative = residual / jnp.maximum(
        _norm(fields.p_field), jnp.finfo(fields.p_field.dtype).eps
    )
    final_z = _apply_preconditioner(prepared.preconditioner, true_residual_vector)
    if prepared.plan.kind is PolarizationSolverKind.TCG:
        iterations = jnp.asarray(prepared.plan.tcg_order, dtype=jnp.int32)
    finite = (
        final_operator.successful
        & jnp.all(jnp.isfinite(induced))
        & jnp.all(jnp.isfinite(final_z))
        & jnp.isfinite(residual)
        & jnp.isfinite(relative)
    )
    converged = finite & ~breakdown & (residual <= prepared.plan.tolerance)
    force_valid = finite & ~breakdown & (residual <= prepared.plan.force_tolerance)
    successful = converged
    state = PolarizationState(
        induced,
        residual,
        iterations,
        converged,
        successful,
        prepared.result_plan_id,
        relative_residual=relative,
        force_valid=force_valid,
        finite=finite,
        solver_kind=prepared.plan.kind.value,
    )
    candidate_history = jnp.stack((induced, predictor_state.history[0]), axis=0)
    next_predictor = PolarizationPredictorState(
        jnp.where(successful, candidate_history, predictor_state.history),
        jnp.where(
            successful,
            jnp.minimum(predictor_state.valid_count + 1, 2),
            predictor_state.valid_count,
        ),
        prepared.prepared_id,
    )
    return PolarizationSolveResult(
        state, next_predictor, final_operator, initial, breakdown, successful
    )


def solve_induced_dipoles(
    plan: PolarizationPlan,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
    *,
    scaling: PolarizationScaleData | None = None,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
) -> PolarizationState:
    """Solve induced dipoles while preserving the original state-returning API."""
    if not isinstance(plan, PolarizationPlan):
        raise TypeError("plan must be PolarizationPlan.")
    prepared = plan.prepare(multipoles, scaling=scaling)
    return prepared.solve(
        positions, predictor_state=predictor_state, cell_vectors=cell_vectors
    ).state


def prepared_polarization_energy(
    prepared: PreparedPolarizationSolver,
    positions: ArrayLike,
    /,
    *,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
):
    """Return the variational polarization energy and solve result."""
    if not isinstance(prepared, PreparedPolarizationSolver):
        raise TypeError("prepared must be PreparedPolarizationSolver.")
    coordinate = _positions(positions, prepared.operator.site_capacity)
    result = prepared.solve(
        coordinate, predictor_state=predictor_state, cell_vectors=cell_vectors
    )
    induced = jax.lax.stop_gradient(result.state.induced_dipoles)
    action = prepared.operator.apply(coordinate, induced, cell_vectors=cell_vectors)
    energy = 0.5 * contract("nd,nd->", induced, action.action) - contract(
        "nd,nd->", induced, action.p_field
    )
    return energy, result


def polarization_energy(
    plan: PolarizationPlan,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
    *,
    scaling: PolarizationScaleData | None = None,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
):
    """Compatibility energy entry point using the variational functional."""
    if not isinstance(plan, PolarizationPlan):
        raise TypeError("plan must be PolarizationPlan.")
    prepared = plan.prepare(multipoles, scaling=scaling)
    energy, result = prepared_polarization_energy(
        prepared,
        positions,
        predictor_state=predictor_state,
        cell_vectors=cell_vectors,
    )
    return energy, (result.state, result.operator.minimum_pair_distance)


def evaluate_prepared_polarization(
    prepared: PreparedPolarizationSolver,
    positions: ArrayLike,
    /,
    *,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
) -> PolarizationEvaluation:
    """Evaluate envelope-theorem forces with an explicit residual gate."""
    coordinate = _positions(positions, prepared.operator.site_capacity)
    (energy, result), gradient = jax.value_and_grad(
        lambda value: prepared_polarization_energy(
            prepared,
            value,
            predictor_state=predictor_state,
            cell_vectors=cell_vectors,
        ),
        has_aux=True,
    )(coordinate)
    finite = jnp.isfinite(energy) & jnp.all(jnp.isfinite(gradient))
    successful = (
        result.state.successful
        & result.state.force_valid
        & result.operator.successful
        & finite
    )
    evidence = PolarizationDifferentiationEvidence(
        "envelope",
        result.state.residual,
        jnp.asarray(prepared.plan.force_tolerance, dtype=energy.dtype),
        jnp.asarray(True),
        result.operator.periodic_contract_valid,
        finite,
        successful,
    )
    return PolarizationEvaluation(
        jnp.where(successful, energy, jnp.nan),
        jnp.where(successful, -gradient, jnp.nan),
        result.state,
        successful,
        evidence=evidence,
    )


def evaluate_polarization(
    plan: PolarizationPlan,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
    *,
    scaling: PolarizationScaleData | None = None,
    predictor_state: PolarizationPredictorState | None = None,
    cell_vectors: ArrayLike | None = None,
) -> PolarizationEvaluation:
    """Compatibility evaluation entry point with fail-closed forces."""
    if not isinstance(plan, PolarizationPlan):
        raise TypeError("plan must be PolarizationPlan.")
    return evaluate_prepared_polarization(
        plan.prepare(multipoles, scaling=scaling),
        positions,
        predictor_state=predictor_state,
        cell_vectors=cell_vectors,
    )


class PolarizationJVPResult(StrictModule):
    """Implicit induced-dipole derivative and qualification evidence."""

    primal: Array
    tangent: Array
    state: PolarizationState
    evidence: PolarizationDifferentiationEvidence
    successful: Array


def evaluate_implicit_polarization_jvp(
    plan: PolarizationPlan,
    positions: ArrayLike,
    tangent: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
    *,
    scaling: PolarizationScaleData | None = None,
    cell_vectors: ArrayLike | None = None,
) -> PolarizationJVPResult:
    """Return an implicit induced-dipole JVP with explicit evidence."""
    if not isinstance(plan, PolarizationPlan):
        raise TypeError("plan must be PolarizationPlan.")
    prepared = plan.prepare(multipoles, scaling=scaling)
    coordinate = _positions(positions, multipoles.site_capacity)
    direction = jnp.asarray(tangent, dtype=coordinate.dtype)
    if direction.shape != coordinate.shape:
        raise ValueError("tangent must have the same shape as positions.")
    cell = _periodic_cell(prepared.operator, cell_vectors)
    predictor_state = prepared.initial_predictor_state()
    primal = _solve_prepared(prepared, coordinate, predictor_state, cell)
    induced = primal.state.induced_dipoles

    def stationarity(position):
        result = _operator_result(prepared.operator, position, induced, cell)
        return result.action - result.p_field

    _, forcing = jax.jvp(stationarity, (coordinate,), (direction,))
    zeros = jnp.zeros_like(induced)
    (
        derivative,
        _,
        _,
        _,
        _,
        residual,
        _,
        _,
        breakdown,
    ) = _linear_solve(prepared, coordinate, -forcing, zeros, cell)
    geometry = _pair_geometry(prepared.operator, coordinate, cell)
    derivative_action, derivative_action_valid = _action(
        prepared.operator, coordinate, derivative, cell, geometry
    )
    residual = _norm(-forcing - derivative_action)
    finite = (
        jnp.all(jnp.isfinite(induced))
        & jnp.all(jnp.isfinite(derivative))
        & jnp.isfinite(residual)
        & derivative_action_valid
    )
    successful = (
        primal.state.successful
        & ~breakdown
        & (residual <= prepared.plan.tolerance)
        & finite
    )
    evidence = PolarizationDifferentiationEvidence(
        "implicit",
        residual,
        jnp.asarray(prepared.plan.tolerance, dtype=coordinate.dtype),
        jnp.asarray(True),
        primal.operator.periodic_contract_valid,
        finite,
        successful,
    )
    return PolarizationJVPResult(
        induced,
        jnp.where(successful, derivative, jnp.nan),
        primal.state,
        evidence,
        successful,
    )


def implicit_polarization_jvp(
    plan: PolarizationPlan,
    positions: ArrayLike,
    tangent: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
    *,
    scaling: PolarizationScaleData | None = None,
    cell_vectors: ArrayLike | None = None,
):
    """Preserve the original ``(primal, tangent)`` implicit-JVP API."""
    result = evaluate_implicit_polarization_jvp(
        plan,
        positions,
        tangent,
        multipoles,
        scaling=scaling,
        cell_vectors=cell_vectors,
    )
    return result.primal, result.tangent


class MultipolePMEPlan(StrictModule, NonTrainableState):
    """Reciprocal multipole modes and Ewald splitting contract."""

    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, grid_shape: tuple[int, int, int], alpha: float, /):
        shape = tuple(int(value) for value in grid_shape)
        alpha_ = float(alpha)
        if (
            len(shape) != 3
            or any(value < 4 for value in shape)
            or not np.isfinite(alpha_)
            or alpha_ <= 0.0
        ):
            raise ValueError("Multipole PME grid and splitting parameter are invalid.")
        self.grid_shape, self.alpha = shape, alpha_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multipole-pme",
                "grid_shape": list(shape),
                "alpha": self.alpha,
            }
        )

    def energy(
        self,
        site_state: AtomisticInteractionSiteState,
        multipoles: PermanentMultipoleSiteData,
        cell_vectors: ArrayLike,
        coulomb_constant: float,
        /,
    ):
        if multipoles.charges.shape != site_state.active_mask.shape:
            raise ValueError("Multipoles and interaction sites must have equal capacity.")
        vectors = jnp.asarray(cell_vectors)
        determinant = jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2]))
        inverse = (
            jnp.stack(
                (
                    jnp.cross(vectors[1], vectors[2]),
                    jnp.cross(vectors[2], vectors[0]),
                    jnp.cross(vectors[0], vectors[1]),
                ),
                axis=1,
            )
            / determinant
        )
        shape = jnp.asarray(self.grid_shape, dtype=site_state.positions.dtype)
        fractional = contract("nd,di->ni", site_state.positions, inverse)
        anchor_index = jnp.argmax(site_state.active_mask.astype(jnp.int32))
        relative_fractional = fractional - fractional[anchor_index]
        scaled = jnp.mod(relative_fractional + 0.5, 1.0) * shape
        base = jax.lax.stop_gradient(jnp.floor(scaled).astype(jnp.int32))
        remainder = scaled - base
        charge_grid = jnp.zeros(self.grid_shape, dtype=site_state.positions.dtype)
        dipole_grid = jnp.zeros(self.grid_shape + (3,), dtype=site_state.positions.dtype)
        quadrupole_grid = jnp.zeros(
            self.grid_shape + (3, 3), dtype=site_state.positions.dtype
        )
        active = site_state.active_mask.astype(site_state.positions.dtype)
        for x_offset in (0, 1):
            for y_offset in (0, 1):
                for z_offset in (0, 1):
                    corner = jnp.asarray((x_offset, y_offset, z_offset))
                    weight_axis = jnp.where(
                        corner[None, :] == 1, remainder, 1.0 - remainder
                    )
                    weight = jnp.prod(weight_axis, axis=-1) * active
                    index = (base + corner[None, :]) % jnp.asarray(
                        self.grid_shape, dtype=jnp.int32
                    )
                    route = (index[:, 0], index[:, 1], index[:, 2])
                    charge_grid = charge_grid.at[route].add(weight * multipoles.charges)
                    dipole_grid = dipole_grid.at[route].add(
                        weight[:, None] * multipoles.dipoles
                    )
                    quadrupole_grid = quadrupole_grid.at[route].add(
                        weight[:, None, None] * multipoles.quadrupoles
                    )
        charge_modes = jnp.fft.fftn(charge_grid)
        dipole_modes = jnp.fft.fftn(dipole_grid, axes=(0, 1, 2))
        quadrupole_modes = jnp.fft.fftn(quadrupole_grid, axes=(0, 1, 2))
        integer_axes = tuple(jnp.fft.fftfreq(size) * size for size in self.grid_shape)
        mode_components = jnp.meshgrid(*integer_axes, indexing="ij")
        modes = jnp.stack(mode_components, axis=-1)
        wave = 2.0 * jnp.pi * contract("...i,ji->...j", modes, inverse)
        squared = jnp.sum(wave * wave, axis=-1)
        dipole_structure = -1.0j * jnp.sum(wave * dipole_modes, axis=-1)
        quadrupole_structure = -0.5 * contract(
            "...i,...ij,...j->...", wave, quadrupole_modes, wave
        )
        window = jnp.prod(
            jnp.stack(
                tuple(
                    jnp.sinc(mode_components[axis] / self.grid_shape[axis]) ** 2
                    for axis in range(3)
                ),
                axis=-1,
            ),
            axis=-1,
        )
        structure = (charge_modes + dipole_structure + quadrupole_structure) / jnp.where(
            jnp.abs(window) > 0.0, window, 1.0
        )
        safe_squared = jnp.where(squared > 0.0, squared, 1.0)
        kernel = jnp.where(
            squared > 0.0,
            jnp.exp(-safe_squared / (4.0 * self.alpha**2)) / safe_squared,
            0.0,
        )
        volume = jnp.abs(determinant)
        energy = (
            2.0
            * jnp.pi
            * coulomb_constant
            / volume
            * jnp.sum(kernel * jnp.real(structure * jnp.conj(structure)))
        )
        successful = (
            site_state.successful
            & jnp.isfinite(volume)
            & (volume > 0.0)
            & jnp.isfinite(energy)
        )
        return jnp.where(successful, energy, jnp.nan)


class ImplicitSolventPlan(StrictModule, NonTrainableState):
    model: str = eqx.field(static=True)
    solvent_dielectric: float = eqx.field(static=True)
    solute_dielectric: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    kirkwood_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: str = "gb",
        /,
        *,
        solvent_dielectric: float = 78.5,
        solute_dielectric: float = 1.0,
        surface_tension: float = 0.005,
        kirkwood_factor: float = 2.455,
    ):
        if (
            model not in ("gb", "gk")
            or min(solvent_dielectric, solute_dielectric, kirkwood_factor) <= 0
            or not np.isfinite(
                [solvent_dielectric, solute_dielectric, surface_tension, kirkwood_factor]
            ).all()
            or surface_tension < 0
        ):
            raise ValueError("Implicit-solvent parameters are invalid.")
        (
            self.model,
            self.solvent_dielectric,
            self.solute_dielectric,
            self.surface_tension,
            self.kirkwood_factor,
        ) = (
            model,
            float(solvent_dielectric),
            float(solute_dielectric),
            float(surface_tension),
            float(kirkwood_factor),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-solvent",
                "model": model,
                "solvent_dielectric": self.solvent_dielectric,
                "solute_dielectric": self.solute_dielectric,
                "surface_tension": self.surface_tension,
                "kirkwood_factor": self.kirkwood_factor,
            }
        )

    def energy(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        radii: ArrayLike,
        coulomb_constant: float,
        /,
    ):
        coordinate, charge, radius = (
            jnp.asarray(positions),
            jnp.asarray(charges),
            jnp.asarray(radii),
        )
        count = coordinate.shape[0]
        if (
            coordinate.shape != (count, 3)
            or charge.shape != (count,)
            or radius.shape != (count,)
            or not np.isfinite(coulomb_constant)
            or float(coulomb_constant) <= 0.0
        ):
            raise ValueError("Implicit-solvent arrays or Coulomb constant are invalid.")
        valid = (
            jnp.all(jnp.isfinite(coordinate))
            & jnp.all(jnp.isfinite(charge))
            & jnp.all(jnp.isfinite(radius) & (radius > 0.0))
        )
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        displacement = coordinate[:, None, :] - coordinate[None, :, :]
        distance2 = jnp.sum(displacement**2, axis=-1)
        radius_product = safe_radius[:, None] * safe_radius[None, :]
        denominator = 4.0 if self.model == "gb" else self.kirkwood_factor
        effective_distance = jnp.sqrt(
            distance2
            + radius_product * jnp.exp(-distance2 / (denominator * radius_product))
        )
        dielectric_factor = 1.0 / self.solvent_dielectric - 1.0 / self.solute_dielectric
        polar = (
            0.5
            * coulomb_constant
            * dielectric_factor
            * jnp.sum(charge[:, None] * charge[None, :] / effective_distance)
        )
        area = 4.0 * jnp.pi * jnp.sum(safe_radius**2)
        energy = polar + self.surface_tension * area
        return jnp.where(valid & jnp.isfinite(energy), energy, jnp.nan)


__all__ = [
    "ImplicitSolventPlan",
    "MultipolePMEPlan",
    "PermanentMultipoleSiteData",
    "PolarizationDifferentiationEvidence",
    "PolarizationEvaluation",
    "PolarizationJVPResult",
    "PolarizationOperatorPlan",
    "PolarizationOperatorResult",
    "PolarizationPlan",
    "PolarizationPreconditionerKind",
    "PolarizationPreconditionerPlan",
    "PolarizationPreconditionerResult",
    "PolarizationPredictorPlan",
    "PolarizationPredictorState",
    "PolarizationScaleData",
    "PolarizationSolveResult",
    "PolarizationSolverKind",
    "PolarizationSolverPlan",
    "PolarizationState",
    "PreparedPolarizationOperator",
    "PreparedPolarizationPreconditioner",
    "PreparedPolarizationSolver",
    "evaluate_implicit_polarization_jvp",
    "evaluate_polarization",
    "evaluate_prepared_polarization",
    "implicit_polarization_jvp",
    "polarization_energy",
    "prepared_polarization_energy",
    "solve_induced_dipoles",
]

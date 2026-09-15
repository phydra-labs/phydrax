#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._lattice_boundary import LatticeBoundaryPhasePlan
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class LatticeRegulator(StrictModule, NonTrainableState):
    """Exact finite tensor regulator; no continuum or volume-limit claim is implied."""

    boundary: LatticeBoundaryPhasePlan
    lattice_spacing: Array
    physical_extent: Array
    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    gauge_topology_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    regulator_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary: LatticeBoundaryPhasePlan,
        lattice_spacing: ArrayLike,
        /,
        *,
        gauge_topology_id: str,
    ):
        if not isinstance(boundary, LatticeBoundaryPhasePlan):
            raise TypeError("boundary must be LatticeBoundaryPhasePlan.")
        spacing = np.asarray(lattice_spacing, dtype=float)
        dimension = boundary.dimension
        if spacing.shape == ():
            spacing = np.full((dimension,), float(spacing))
        if spacing.shape != (dimension,):
            raise ValueError("lattice_spacing must be scalar or have one value per axis.")
        if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
            raise ValueError("Lattice spacings must be finite and positive.")
        gauge_topology = _identifier(gauge_topology_id, "Gauge topology ID")
        shape = boundary.topology.axis_sizes
        extent = spacing * np.asarray(shape)
        self.boundary = boundary
        self.lattice_spacing = jnp.asarray(spacing)
        self.physical_extent = jnp.asarray(extent)
        self.lattice_shape = shape
        self.topology_id = boundary.topology.topology_id
        self.gauge_topology_id = gauge_topology
        self.dimension = dimension
        self.site_count = prod(shape)
        self.regulator_id = canonical_fingerprint(
            {
                "kind": "finite-lattice-regulator",
                "boundary": boundary.plan_id,
                "tensor_topology": boundary.topology.topology_id,
                "gauge_topology": gauge_topology,
                "lattice_shape": list(shape),
                "lattice_spacing": array_tree_fingerprint(spacing),
                "physical_extent": array_tree_fingerprint(extent),
                "site_order": "c-order",
            }
        )


class LatticeTheoryPoint(StrictModule, NonTrainableState):
    """Bare theory parameters at one exact regulator on a named trajectory."""

    regulator: LatticeRegulator
    bare_parameter_values: Array
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    scale_setting_id: str = eqx.field(static=True)
    theory_point_id: str = eqx.field(static=True)

    def __init__(
        self,
        regulator: LatticeRegulator,
        /,
        *,
        action_id: str,
        bare_parameters: Mapping[str, float],
        trajectory_id: str,
        scale_setting_id: str,
    ):
        if not isinstance(regulator, LatticeRegulator):
            raise TypeError("regulator must be LatticeRegulator.")
        if not isinstance(bare_parameters, Mapping) or not bare_parameters:
            raise TypeError("bare_parameters must be a non-empty mapping.")
        parameters = sorted(
            (str(name).strip(), value) for name, value in bare_parameters.items()
        )
        names = tuple(name for name, _ in parameters)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Bare-parameter names must be distinct and non-empty.")
        values = np.asarray([value for _, value in parameters], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("Bare-parameter values must be finite.")
        self.regulator = regulator
        self.bare_parameter_values = jnp.asarray(values)
        self.parameter_names = names
        self.action_id = _identifier(action_id, "Action ID")
        self.trajectory_id = _identifier(trajectory_id, "Trajectory ID")
        self.scale_setting_id = _identifier(scale_setting_id, "Scale-setting ID")
        self.theory_point_id = canonical_fingerprint(
            {
                "kind": "finite-lattice-theory-point",
                "regulator": regulator.regulator_id,
                "action": self.action_id,
                "bare_parameters": dict(zip(names, values.tolist(), strict=True)),
                "trajectory": self.trajectory_id,
                "scale_setting": self.scale_setting_id,
            }
        )


class LatticeEnsembleStatus(IntEnum):
    SUCCESS = 0
    INCOMPLETE = 1
    NONFINITE_OBSERVABLES = 2
    MISSING_QUALIFICATION = 3


class LatticeEnsemblePlan(StrictModule, NonTrainableState):
    """Fixed-capacity admission plan for externally generated chain observables."""

    theory_point: LatticeTheoryPoint
    observable_names: tuple[str, ...] = eqx.field(static=True)
    qualification_ids: tuple[str, ...] = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    maximum_observables: int = eqx.field(static=True)
    maximum_storage_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        theory_point: LatticeTheoryPoint,
        observable_names: Sequence[str],
        /,
        *,
        sample_count: int,
        qualification_ids: Sequence[str],
        maximum_samples: int = 1_000_000,
        maximum_observables: int = 256,
        maximum_storage_bytes: int = 1 << 30,
    ):
        if not isinstance(theory_point, LatticeTheoryPoint):
            raise TypeError("theory_point must be LatticeTheoryPoint.")
        names = tuple(str(value).strip() for value in observable_names)
        qualifications = tuple(str(value).strip() for value in qualification_ids)
        count = int(sample_count)
        sample_limit = int(maximum_samples)
        observable_limit = int(maximum_observables)
        byte_limit = int(maximum_storage_bytes)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Observable names must be distinct and non-empty.")
        if any(not value for value in qualifications) or len(set(qualifications)) != len(
            qualifications
        ):
            raise ValueError("Qualification IDs must be distinct and non-empty.")
        if count < 2 or sample_limit < 2 or count > sample_limit:
            raise ValueError("sample_count must lie in [2, maximum_samples].")
        if observable_limit < 1 or len(names) > observable_limit:
            raise ValueError("Observable count exceeds maximum_observables.")
        required_bytes = count * len(names) * np.dtype(np.float64).itemsize
        if byte_limit < 1 or required_bytes > byte_limit:
            raise ValueError("Declared ensemble observables exceed the storage guard.")
        self.theory_point = theory_point
        self.observable_names = names
        self.qualification_ids = qualifications
        self.sample_count = count
        self.maximum_samples = sample_limit
        self.maximum_observables = observable_limit
        self.maximum_storage_bytes = byte_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-lattice-ensemble-plan",
                "theory_point": theory_point.theory_point_id,
                "observables": list(names),
                "sample_count": count,
                "required_qualifications": list(qualifications),
                "resource_guards": {
                    "maximum_samples": sample_limit,
                    "maximum_observables": observable_limit,
                    "maximum_storage_bytes": byte_limit,
                },
                "generation": "caller-owned-no-implicit-sampler",
            }
        )

    def prepare(self, /) -> PreparedLatticeEnsemble:
        return PreparedLatticeEnsemble(self)


class LatticeEnsembleEvidence(StrictModule, NonTrainableState):
    status: Array
    finite: Array
    complete: Array
    valid_sample_count: Array
    means: Array
    covariance: Array
    plan_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LatticeEnsembleStatus.SUCCESS)


class PreparedLatticeEnsemble(StrictModule, NonTrainableState):
    """Immutable ensemble schema whose runtime only admits fixed-shape observations."""

    plan: LatticeEnsemblePlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: LatticeEnsemblePlan, /):
        if not isinstance(plan, LatticeEnsemblePlan):
            raise TypeError("plan must be LatticeEnsemblePlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-lattice-ensemble",
                "plan": plan.plan_id,
                "shape": [plan.sample_count, len(plan.observable_names)],
                "dtype": np.dtype(np.float64).str,
            }
        )

    def admit(
        self,
        observables: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        evidence_ids: Sequence[str],
    ) -> LatticeEnsembleEvidence:
        values = jnp.asarray(observables)
        expected = (self.plan.sample_count, len(self.plan.observable_names))
        if values.shape != expected:
            raise ValueError(f"observables must have shape {expected}.")
        if jnp.iscomplexobj(values):
            raise TypeError("Admitted production observables must be real-valued.")
        mask = (
            jnp.ones((self.plan.sample_count,), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if mask.shape != (self.plan.sample_count,):
            raise ValueError("valid must contain one boolean per sample.")
        supplied = tuple(str(value).strip() for value in evidence_ids)
        if any(not value for value in supplied) or len(set(supplied)) != len(supplied):
            raise ValueError("Evidence IDs must be distinct and non-empty.")
        qualifications_present = set(self.plan.qualification_ids).issubset(supplied)
        finite_rows = jnp.all(jnp.isfinite(values), axis=1)
        active = mask & finite_rows
        active_count = jnp.sum(active.astype(jnp.int32))
        safe_count = jnp.maximum(active_count, 1).astype(values.dtype)
        safe = jnp.where(active[:, None], values, 0.0)
        means = jnp.sum(safe, axis=0) / safe_count
        centered = jnp.where(active[:, None], values - means, 0.0)
        covariance = (centered.T @ centered) / jnp.maximum(active_count - 1, 1).astype(
            values.dtype
        )
        finite = jnp.all(finite_rows | ~mask) & jnp.all(jnp.isfinite(covariance))
        complete = jnp.all(mask) & (active_count == self.plan.sample_count)
        status = jnp.where(
            not qualifications_present,
            int(LatticeEnsembleStatus.MISSING_QUALIFICATION),
            jnp.where(
                ~finite,
                int(LatticeEnsembleStatus.NONFINITE_OBSERVABLES),
                jnp.where(
                    ~complete,
                    int(LatticeEnsembleStatus.INCOMPLETE),
                    int(LatticeEnsembleStatus.SUCCESS),
                ),
            ),
        )
        ensemble_id = canonical_fingerprint(
            {
                "kind": "finite-lattice-ensemble-evidence",
                "prepared": self.prepared_id,
                "evidence_ids": list(supplied),
            }
        )
        return LatticeEnsembleEvidence(
            status,
            finite,
            complete,
            active_count,
            means,
            covariance,
            self.plan.plan_id,
            supplied,
            ensemble_id,
        )


class ContinuumExtrapolationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    LINEAR_SOLVE_FAILED = 2
    INSUFFICIENT_DEGREES_OF_FREEDOM = 3


class ContinuumExtrapolationPlan(StrictModule, NonTrainableState):
    """Finite fixed-volume lattice-spacing trajectory and extrapolation basis."""

    theory_points: tuple[LatticeTheoryPoint, ...]
    powers: tuple[int, ...] = eqx.field(static=True)
    observable_name: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    maximum_points: int = eqx.field(static=True)
    extent_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        theory_points: Sequence[LatticeTheoryPoint],
        /,
        *,
        observable_name: str,
        powers: Sequence[int] = (0, 2),
        maximum_points: int = 64,
        extent_tolerance: float = 1.0e-10,
    ):
        points = tuple(theory_points)
        powers_ = tuple(int(value) for value in powers)
        point_limit = int(maximum_points)
        tolerance = float(extent_tolerance)
        if not points or not all(
            isinstance(point, LatticeTheoryPoint) for point in points
        ):
            raise TypeError("theory_points must contain LatticeTheoryPoint values.")
        if len({point.theory_point_id for point in points}) != len(points):
            raise ValueError("Continuum theory points must be distinct.")
        if point_limit < 2 or point_limit > 4096 or len(points) > point_limit:
            raise ValueError("Theory point count exceeds the fixed maximum_points guard.")
        if not powers_ or powers_[0] != 0 or any(value < 0 for value in powers_):
            raise ValueError("powers must begin with zero and be non-negative.")
        if tuple(sorted(set(powers_))) != powers_:
            raise ValueError("powers must be strictly increasing.")
        if len(points) < len(powers_):
            raise ValueError("Continuum fit is underdetermined for the declared powers.")
        trajectories = {point.trajectory_id for point in points}
        scale_settings = {point.scale_setting_id for point in points}
        dimensions = {point.regulator.dimension for point in points}
        actions = {point.action_id for point in points}
        if len(trajectories) != 1 or len(scale_settings) != 1 or len(dimensions) != 1:
            raise ValueError(
                "Continuum points must share trajectory, scale setting, and dimension."
            )
        if len(actions) != 1:
            raise ValueError("Continuum points must share one lattice action family.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("extent_tolerance must be finite and non-negative.")
        extents = np.stack(
            tuple(np.asarray(point.regulator.physical_extent) for point in points)
        )
        extent_scale = np.maximum(np.max(np.abs(extents), axis=0), 1.0)
        if np.any(
            np.max(np.abs(extents - extents[0]), axis=0) > tolerance * extent_scale
        ):
            raise ValueError(
                "Continuum extrapolation requires fixed physical extent; vary volume separately."
            )
        spacings = np.asarray(
            [
                float(np.prod(np.asarray(point.regulator.lattice_spacing)))
                ** (1.0 / point.regulator.dimension)
                for point in points
            ]
        )
        if np.unique(spacings).size != spacings.size:
            raise ValueError(
                "Continuum points require distinct effective lattice spacings."
            )
        self.theory_points = points
        self.powers = powers_
        self.observable_name = _identifier(observable_name, "Observable name")
        self.trajectory_id = next(iter(trajectories))
        self.point_count = len(points)
        self.coefficient_count = len(powers_)
        self.maximum_points = point_limit
        self.extent_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-volume-continuum-extrapolation-plan",
                "theory_points": [point.theory_point_id for point in points],
                "observable": self.observable_name,
                "powers": list(powers_),
                "trajectory": self.trajectory_id,
                "maximum_points": point_limit,
                "extent_tolerance": tolerance,
            }
        )

    def prepare(self, /) -> PreparedContinuumExtrapolation:
        return PreparedContinuumExtrapolation(self)


class ContinuumExtrapolationEvidence(StrictModule, NonTrainableState):
    status: Array
    finite: Array
    solve_successful: Array
    residual_norm: Array
    chi_square: Array
    degrees_of_freedom: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(ContinuumExtrapolationStatus.SUCCESS)


class ContinuumExtrapolationResult(StrictModule):
    continuum_value: Array
    continuum_standard_error: Array
    coefficients: Array
    fitted_values: Array
    residuals: Array
    evidence: ContinuumExtrapolationEvidence
    plan_id: str = eqx.field(static=True)


class PreparedContinuumExtrapolation(StrictModule, NonTrainableState):
    """Prepared fixed design; runtime solves only the supplied weighted fit."""

    plan: ContinuumExtrapolationPlan
    lattice_spacings: Array
    design: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ContinuumExtrapolationPlan, /):
        if not isinstance(plan, ContinuumExtrapolationPlan):
            raise TypeError("plan must be ContinuumExtrapolationPlan.")
        spacings = np.asarray(
            [
                float(np.prod(np.asarray(point.regulator.lattice_spacing)))
                ** (1.0 / point.regulator.dimension)
                for point in plan.theory_points
            ]
        )
        reference = float(np.max(spacings))
        scaled = spacings / reference
        design = np.stack(tuple(scaled**power for power in plan.powers), axis=1)
        if np.linalg.matrix_rank(design) != plan.coefficient_count:
            raise ValueError("Continuum design matrix is rank deficient.")
        self.plan = plan
        self.lattice_spacings = jnp.asarray(spacings)
        self.design = jnp.asarray(design)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-continuum-extrapolation",
                "plan": plan.plan_id,
                "lattice_spacings": array_tree_fingerprint(spacings),
                "scaled_design": array_tree_fingerprint(design),
            }
        )

    def fit(
        self,
        estimates: ArrayLike,
        standard_errors: ArrayLike,
        /,
    ) -> ContinuumExtrapolationResult:
        values = jnp.asarray(estimates)
        errors = jnp.asarray(standard_errors)
        expected = (self.plan.point_count,)
        if values.shape != expected or errors.shape != expected:
            raise ValueError(f"estimates and standard_errors must have shape {expected}.")
        if jnp.iscomplexobj(values) or jnp.iscomplexobj(errors):
            raise TypeError("Continuum fits require real estimates and errors.")
        safe_errors = jnp.where(
            jnp.isfinite(errors) & (errors > 0.0), errors, jnp.ones_like(errors)
        )
        weighted_design = self.design / safe_errors[:, None]
        weighted_values = values / safe_errors
        normal = weighted_design.T @ weighted_design
        right = weighted_design.T @ weighted_values
        policy = LinearSolvePolicy(DenseLU())
        coefficient_solve = solve(
            LinearSystem(
                DenseLinearOperator(normal),
                problem_id=f"{self.prepared_id}:weighted-normal-equations",
            ),
            right,
            policy=policy,
        )
        covariance_column_solve = solve(
            LinearSystem(
                DenseLinearOperator(normal),
                problem_id=f"{self.prepared_id}:continuum-covariance-column",
            ),
            jnp.eye(self.plan.coefficient_count, dtype=normal.dtype)[:, 0],
            policy=policy,
        )
        coefficients = coefficient_solve.value
        fitted = self.design @ coefficients
        residuals = values - fitted
        weighted_residuals = residuals / safe_errors
        residual_norm = jnp.linalg.norm(weighted_design @ coefficients - weighted_values)
        chi_square = jnp.sum(weighted_residuals * weighted_residuals)
        finite_input = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(errors))
            & jnp.all(errors > 0.0)
        )
        solve_successful = jnp.all(coefficient_solve.status == 0) & jnp.all(
            covariance_column_solve.status == 0
        )
        finite = (
            finite_input
            & jnp.all(jnp.isfinite(coefficients))
            & jnp.isfinite(chi_square)
            & jnp.isfinite(residual_norm)
        )
        degrees_of_freedom = self.plan.point_count - self.plan.coefficient_count
        status = jnp.where(
            ~finite_input,
            int(ContinuumExtrapolationStatus.NONFINITE_INPUT),
            jnp.where(
                ~solve_successful,
                int(ContinuumExtrapolationStatus.LINEAR_SOLVE_FAILED),
                jnp.where(
                    degrees_of_freedom < 1,
                    int(ContinuumExtrapolationStatus.INSUFFICIENT_DEGREES_OF_FREEDOM),
                    int(ContinuumExtrapolationStatus.SUCCESS),
                ),
            ),
        )
        evidence = ContinuumExtrapolationEvidence(
            status,
            finite,
            solve_successful,
            residual_norm,
            chi_square,
            degrees_of_freedom,
            self.plan.plan_id,
        )
        standard_error = jnp.sqrt(jnp.maximum(covariance_column_solve.value[0], 0.0))
        return ContinuumExtrapolationResult(
            coefficients[0],
            standard_error,
            coefficients,
            fitted,
            residuals,
            evidence,
            self.plan.plan_id,
        )


__all__ = [
    "ContinuumExtrapolationEvidence",
    "ContinuumExtrapolationPlan",
    "ContinuumExtrapolationResult",
    "ContinuumExtrapolationStatus",
    "LatticeEnsembleEvidence",
    "LatticeEnsemblePlan",
    "LatticeEnsembleStatus",
    "LatticeRegulator",
    "LatticeTheoryPoint",
    "PreparedContinuumExtrapolation",
    "PreparedLatticeEnsemble",
]

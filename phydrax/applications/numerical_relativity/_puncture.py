#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.collocation import ChebyshevCollocation
from ...linalg import ArraySpace, GMRES, LinearSolvePolicy, TolerancePolicy
from ...nonlinear import (
    JacobianRefreshPolicy,
    NewtonForcingPolicy,
    NewtonKrylov,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    RootLineSearch,
)
from ._initial_data import ADMInitialData, _coordinates, _package_initial_data
from ._status import NumericalRelativityStatus, ScientificStatus


class Puncture(StrictModule, NonTrainableState):
    """One Bowen--York puncture with bare mass and asymptotic charge parameters."""

    bare_mass: Array
    position: Array
    linear_momentum: Array
    spin: Array
    puncture_id: str = eqx.field(static=True)

    def __init__(
        self,
        bare_mass: ArrayLike,
        position: ArrayLike,
        /,
        *,
        linear_momentum: ArrayLike = (0.0, 0.0, 0.0),
        spin: ArrayLike = (0.0, 0.0, 0.0),
        puncture_id: str | None = None,
    ):
        mass = float(np.asarray(bare_mass))
        location = np.asarray(position, dtype=float)
        momentum = np.asarray(linear_momentum, dtype=float)
        angular = np.asarray(spin, dtype=float)
        if not isfinite(mass) or mass <= 0.0:
            raise ValueError("Puncture bare mass must be finite and positive.")
        if any(
            value.shape != (3,) or not np.all(np.isfinite(value))
            for value in (location, momentum, angular)
        ):
            raise ValueError("Puncture position, momentum, and spin must be finite three-vectors.")
        identity = (
            canonical_fingerprint(
                {
                    "kind": "bowen-york-puncture",
                    "bare_mass": mass,
                    "position": location.tolist(),
                    "linear_momentum": momentum.tolist(),
                    "spin": angular.tolist(),
                }
            )
            if puncture_id is None
            else str(puncture_id)
        )
        if not identity:
            raise ValueError("puncture_id must be non-empty.")
        self.bare_mass = jnp.asarray(mass)
        self.position = jnp.asarray(location)
        self.linear_momentum = jnp.asarray(momentum)
        self.spin = jnp.asarray(angular)
        self.puncture_id = identity


class BrillLindquistInitialData(StrictModule, NonTrainableState):
    """Time-symmetric conformally flat vacuum data for one or more punctures."""

    punctures: tuple[Puncture, ...]
    data_id: str = eqx.field(static=True)

    def __init__(self, punctures: tuple[Puncture, ...], /):
        self.punctures = _puncture_tuple(punctures, minimum=1)
        self.data_id = canonical_fingerprint(
            {
                "kind": "brill-lindquist-initial-data",
                "punctures": [item.puncture_id for item in self.punctures],
            }
        )

    def __call__(self, coordinates: ArrayLike, /) -> ADMInitialData:
        points = _coordinates(coordinates)
        zeros = jnp.zeros(points.shape[:-1], dtype=points.dtype)
        return _puncture_initial_data(
            self.punctures,
            points,
            zeros,
            include_bowen_york=False,
            data_id=self.data_id,
        )


class BowenYorkInitialData(StrictModule, NonTrainableState):
    """Conformally flat maximal Bowen--York data before Hamiltonian correction."""

    punctures: tuple[Puncture, ...]
    data_id: str = eqx.field(static=True)

    def __init__(self, punctures: tuple[Puncture, ...], /):
        self.punctures = _puncture_tuple(punctures, minimum=1)
        self.data_id = canonical_fingerprint(
            {
                "kind": "uncorrected-bowen-york-initial-data",
                "punctures": [item.puncture_id for item in self.punctures],
            }
        )

    def __call__(self, coordinates: ArrayLike, /) -> ADMInitialData:
        points = _coordinates(coordinates)
        zeros = jnp.zeros(points.shape[:-1], dtype=points.dtype)
        return _puncture_initial_data(
            self.punctures,
            points,
            zeros,
            include_bowen_york=True,
            data_id=self.data_id,
        )


class PunctureTuningEvidence(StrictModule):
    """Targets, achieved ADM estimates, and separate qualification predicates."""

    target_adm_mass: Array
    achieved_adm_mass: Array
    mass_error: Array
    target_linear_momentum: Array
    achieved_linear_momentum: Array
    momentum_error: Array
    target_spin: Array
    achieved_spin: Array
    spin_error: Array
    target_angular_momentum: Array
    achieved_angular_momentum: Array
    angular_momentum_error: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    evidence_id: str = eqx.field(static=True)


class TwoPunctureRestart(StrictModule, NonTrainableState):
    """Shape-fixed correction field bound to one exact solver plan."""

    correction: Array
    plan_id: str = eqx.field(static=True)
    source_content_id: str = eqx.field(static=True)
    restart_id: str = eqx.field(static=True)

    def __init__(self, correction, /, *, plan_id: str, source_content_id: str):
        value = jnp.asarray(correction)
        plan = str(plan_id)
        source = str(source_content_id)
        if not plan or not source:
            raise ValueError("Restart plan and source content IDs must be non-empty.")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("Puncture restart correction must have floating-point dtype.")
        self.correction = value
        self.plan_id = plan
        self.source_content_id = source
        self.restart_id = canonical_fingerprint(
            {
                "kind": "two-puncture-restart",
                "plan": plan,
                "source_content": source,
                "correction": array_tree_fingerprint(np.asarray(value)),
            }
        )


class TwoPunctureHamiltonianResult(StrictModule):
    """Solved correction and complete nonlinear/scientific evidence."""

    coordinates: Array
    correction: Array
    conformal_factor: Array
    residual: Array
    initial_residual_norm: Array
    final_residual_norm: Array
    nonlinear_iterations: Array
    linear_iterations: Array
    nonlinear_status: Array
    initial_data: ADMInitialData
    tuning: PunctureTuningEvidence
    status: ScientificStatus
    restart: TwoPunctureRestart
    plan_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    matrix_free: bool = eqx.field(static=True)
    operator_storage_bytes: int = eqx.field(static=True)


class TwoPunctureHamiltonianPlan(StrictModule, NonTrainableState):
    """Fixed-resolution tensor-Chebyshev, matrix-free Newton--Krylov solve.

    Only one-dimensional collocation matrices are stored.  The
    ``nx*ny*nz`` Jacobian is represented by JVP actions inside native GMRES and
    is never assembled as a dense global matrix.
    """

    punctures: tuple[Puncture, Puncture]
    x: ChebyshevCollocation
    y: ChebyshevCollocation
    z: ChebyshevCollocation
    coordinates: Array
    boundary_mask: Array
    volume_weights: Array
    target_adm_mass: Array
    target_linear_momentum: Array
    target_spin: Array
    target_angular_momentum: Array
    nonlinear_tolerance: float = eqx.field(static=True)
    maximum_newton_steps: int = eqx.field(static=True)
    linear_tolerance: float = eqx.field(static=True)
    maximum_linear_steps: int = eqx.field(static=True)
    mass_tolerance: float = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
    operator_storage_bytes: int = eqx.field(static=True)
    matrix_free: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        punctures: tuple[Puncture, Puncture],
        /,
        *,
        resolution: int | tuple[int, int, int] = 9,
        half_extent: float | tuple[float, float, float] = 12.0,
        nonlinear_tolerance: float = 1.0e-9,
        maximum_newton_steps: int = 20,
        linear_tolerance: float = 1.0e-7,
        maximum_linear_steps: int = 128,
        target_adm_mass: float | None = None,
        target_linear_momentum: ArrayLike | None = None,
        target_spin: ArrayLike | None = None,
        target_angular_momentum: ArrayLike | None = None,
        mass_tolerance: float = 5.0e-2,
        charge_tolerance: float = 1.0e-10,
    ):
        pair = _puncture_tuple(punctures, minimum=2)
        if len(pair) != 2:
            raise ValueError("TwoPunctureHamiltonianPlan requires exactly two punctures.")
        shape = _triple_int(resolution, "resolution")
        if any(count < 4 for count in shape):
            raise ValueError("Each two-puncture spectral resolution must be at least four.")
        extents = _triple_float(half_extent, "half_extent")
        if any(not isfinite(value) or value <= 0.0 for value in extents):
            raise ValueError("Each spectral half-extent must be finite and positive.")
        nonlinear_tol = float(nonlinear_tolerance)
        linear_tol = float(linear_tolerance)
        mass_tol = float(mass_tolerance)
        charge_tol = float(charge_tolerance)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (nonlinear_tol, linear_tol, mass_tol, charge_tol)
        ):
            raise ValueError("Puncture solve and tuning tolerances must be finite and positive.")
        newton_steps = int(maximum_newton_steps)
        linear_steps = int(maximum_linear_steps)
        if newton_steps < 1 or linear_steps < 1:
            raise ValueError("Puncture nonlinear and linear step limits must be positive.")
        axes = tuple(
            ChebyshevCollocation(
                count,
                lower=-extent,
                upper=extent,
                maximum_dimension=max(shape),
            )
            for count, extent in zip(shape, extents, strict=True)
        )
        mesh = jnp.meshgrid(*(axis.nodes for axis in axes), indexing="ij")
        coordinates = jnp.stack(mesh, axis=-1)
        boundary = jnp.zeros(shape, dtype=bool)
        for axis in range(3):
            lower_index = [slice(None)] * 3
            upper_index = [slice(None)] * 3
            lower_index[axis] = 0
            upper_index[axis] = -1
            boundary = boundary.at[tuple(lower_index)].set(True)
            boundary = boundary.at[tuple(upper_index)].set(True)
        for puncture in pair:
            distances = np.linalg.norm(
                np.asarray(coordinates) - np.asarray(puncture.position), axis=-1
            )
            scale = max(extents)
            if float(np.min(distances)) <= 128.0 * np.finfo(float).eps * scale:
                raise ValueError(
                    "A puncture coincides with a collocation node; use another fixed resolution."
                )
        axis_weights = tuple(
            _clenshaw_curtis_weights(np.asarray(axis.nodes)) for axis in axes
        )
        volume_weights = (
            jnp.asarray(axis_weights[0])[:, None, None]
            * jnp.asarray(axis_weights[1])[None, :, None]
            * jnp.asarray(axis_weights[2])[None, None, :]
        )
        default_mass = sum(float(item.bare_mass) for item in pair)
        default_momentum = sum(
            (np.asarray(item.linear_momentum) for item in pair), start=np.zeros(3)
        )
        default_spin = sum(
            (np.asarray(item.spin) for item in pair), start=np.zeros(3)
        )
        default_angular = sum(
            (
                np.asarray(item.spin)
                + np.cross(np.asarray(item.position), np.asarray(item.linear_momentum))
                for item in pair
            ),
            start=np.zeros(3),
        )
        mass_target = default_mass if target_adm_mass is None else float(target_adm_mass)
        momentum_target = (
            default_momentum
            if target_linear_momentum is None
            else np.asarray(target_linear_momentum, dtype=float)
        )
        spin_target = (
            default_spin
            if target_spin is None
            else np.asarray(target_spin, dtype=float)
        )
        angular_target = (
            default_angular
            if target_angular_momentum is None
            else np.asarray(target_angular_momentum, dtype=float)
        )
        if (
            not isfinite(mass_target)
            or mass_target <= 0.0
            or momentum_target.shape != (3,)
            or spin_target.shape != (3,)
            or angular_target.shape != (3,)
            or not np.all(np.isfinite(momentum_target))
            or not np.all(np.isfinite(spin_target))
            or not np.all(np.isfinite(angular_target))
        ):
            raise ValueError("Puncture ADM charge targets are invalid.")
        storage = sum(
            int(
                (axis.first_derivative.size + axis.second_derivative.size)
                * axis.second_derivative.dtype.itemsize
            )
            for axis in axes
        )
        self.punctures = (pair[0], pair[1])
        self.x, self.y, self.z = axes
        self.coordinates = coordinates
        self.boundary_mask = boundary
        self.volume_weights = volume_weights
        self.target_adm_mass = jnp.asarray(mass_target)
        self.target_linear_momentum = jnp.asarray(momentum_target)
        self.target_spin = jnp.asarray(spin_target)
        self.target_angular_momentum = jnp.asarray(angular_target)
        self.nonlinear_tolerance = nonlinear_tol
        self.maximum_newton_steps = newton_steps
        self.linear_tolerance = linear_tol
        self.maximum_linear_steps = linear_steps
        self.mass_tolerance = mass_tol
        self.charge_tolerance = charge_tol
        self.operator_storage_bytes = storage
        self.matrix_free = True
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-resolution-two-puncture-hamiltonian",
                "punctures": [item.puncture_id for item in pair],
                "shape": list(shape),
                "half_extents": list(extents),
                "axis_discretizations": [axis.discretization_id for axis in axes],
                "nonlinear_tolerance": nonlinear_tol,
                "maximum_newton_steps": newton_steps,
                "linear_tolerance": linear_tol,
                "maximum_linear_steps": linear_steps,
                "targets": {
                    "mass": mass_target,
                    "momentum": momentum_target.tolist(),
                    "spin": spin_target.tolist(),
                    "angular_momentum": angular_target.tolist(),
                },
            }
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.x.count, self.y.count, self.z.count)

    @property
    def degrees_of_freedom(self) -> int:
        return int(np.prod(self.shape))

    def singular_conformal_factor(self) -> Array:
        conformal, _, _ = _conformal_fields(self.punctures, self.coordinates)
        return conformal

    def conformal_extrinsic_curvature(self) -> Array:
        _, conformal_extrinsic, _ = _conformal_fields(
            self.punctures, self.coordinates
        )
        return conformal_extrinsic

    def hamiltonian_residual(self, correction: ArrayLike, /) -> Array:
        value = jnp.asarray(correction)
        if value.shape != self.shape:
            raise ValueError("Hamiltonian correction must match the fixed spectral shape.")
        singular, conformal_extrinsic, _ = _conformal_fields(
            self.punctures, self.coordinates
        )
        conformal = singular + value
        extrinsic_squared = ein.contract(
            "...ij,...ij->...", conformal_extrinsic, conformal_extrinsic
        )
        laplacian = (
            ein.contract("ia,ajk->ijk", self.x.second_derivative, value)
            + ein.contract("ja,iak->ijk", self.y.second_derivative, value)
            + ein.contract("ka,ija->ijk", self.z.second_derivative, value)
        )
        source = 0.125 * conformal ** (-7) * extrinsic_squared
        interior = laplacian + source
        return jnp.where(self.boundary_mask, value, interior)

    def solve(
        self,
        initial_correction: ArrayLike | TwoPunctureRestart | None = None,
        /,
    ) -> TwoPunctureHamiltonianResult:
        if isinstance(initial_correction, TwoPunctureRestart):
            if initial_correction.plan_id != self.plan_id:
                raise ValueError("Puncture restart belongs to another solver plan.")
            initial = initial_correction.correction
        elif initial_correction is None:
            initial = jnp.zeros(self.shape, dtype=self.coordinates.dtype)
        else:
            initial = jnp.asarray(initial_correction, dtype=self.coordinates.dtype)
        if initial.shape != self.shape:
            raise ValueError("Initial Hamiltonian correction must match the plan shape.")
        if not bool(jnp.all(jnp.isfinite(initial))):
            raise ValueError("Initial Hamiltonian correction must be finite.")
        singular = self.singular_conformal_factor()
        space = ArraySpace(self.shape, dtype=initial.dtype)
        problem = NonlinearSystemProblem(
            lambda correction, _: self.hamiltonian_residual(correction),
            state_space=space,
            residual_space=space,
            trial_validity=lambda correction, _: jnp.all(singular + correction > 0.0),
            trial_validity_id=f"positive-conformal-factor:{self.plan_id}",
            problem_id=f"two-puncture-hamiltonian:{self.plan_id}",
        )
        restart_size = min(32, self.degrees_of_freedom)
        method = NewtonKrylov(
            linear_policy=LinearSolvePolicy(
                GMRES(restart=restart_size, stagnation_iterations=restart_size),
                tolerance=TolerancePolicy(
                    relative=self.linear_tolerance,
                    absolute=self.linear_tolerance * self.nonlinear_tolerance,
                    max_steps=self.maximum_linear_steps,
                ),
            ),
            forcing_policy=NewtonForcingPolicy(
                "eisenstat-walker",
                initial=min(0.1, max(self.linear_tolerance, 1.0e-6)),
                minimum=min(self.linear_tolerance, 1.0e-4),
                maximum=0.5,
            ),
            jacobian_refresh=JacobianRefreshPolicy("every-step"),
            line_search=RootLineSearch(maximum_steps=20),
        )
        nonlinear = method.solve(
            problem,
            initial,
            termination=NonlinearTermination(
                absolute_residual=self.nonlinear_tolerance,
                relative_residual=self.nonlinear_tolerance,
                maximum_steps=self.maximum_newton_steps,
                maximum_linear_iterations=(
                    self.maximum_newton_steps * self.maximum_linear_steps
                ),
            ),
        )
        correction = nonlinear.state
        residual = self.hamiltonian_residual(correction)
        final_norm = jnp.sqrt(ein.contract("ijk,ijk->", residual, residual))
        initial_residual = self.hamiltonian_residual(initial)
        initial_norm = jnp.sqrt(
            ein.contract("ijk,ijk->", initial_residual, initial_residual)
        )
        conformal = singular + correction
        data_identity = canonical_fingerprint(
            {
                "kind": "solved-two-puncture-initial-data",
                "plan": self.plan_id,
                "correction": array_tree_fingerprint(np.asarray(correction)),
            }
        )
        initial_data = _puncture_initial_data(
            self.punctures,
            self.coordinates,
            correction,
            include_bowen_york=True,
            data_id=data_identity,
        )
        finite = (
            jnp.all(jnp.isfinite(correction))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(conformal))
        )
        converged = (
            nonlinear.status == int(NonlinearStatus.SUCCESS)
        ) & (final_norm <= self.nonlinear_tolerance * (1.0 + initial_norm))
        physical = finite & jnp.all(conformal > 0.0) & initial_data.status.physically_valid
        tuning = self.tuning_evidence(
            correction,
            converged=converged,
            physically_valid=physical,
        )
        qualified = converged & physical & tuning.qualified
        status_value = (
            jnp.where(finite, 0, int(NumericalRelativityStatus.NONFINITE_STATE))
            | jnp.where(
                jnp.all(conformal > 0.0),
                0,
                int(NumericalRelativityStatus.NONPOSITIVE_CONFORMAL_FACTOR),
            )
            | jnp.where(
                converged,
                0,
                int(NumericalRelativityStatus.CONSTRAINT_TOLERANCE_EXCEEDED),
            )
            | jnp.where(
                False, 0, int(NumericalRelativityStatus.DERIVATIVE_INVALID)
            )
        ).astype(jnp.int32)
        content_id = canonical_fingerprint(
            {
                "kind": "two-puncture-hamiltonian-solution",
                "plan": self.plan_id,
                "correction": array_tree_fingerprint(np.asarray(correction)),
                "residual": array_tree_fingerprint(np.asarray(residual)),
            }
        )
        restart = TwoPunctureRestart(
            correction,
            plan_id=self.plan_id,
            source_content_id=content_id,
        )
        return TwoPunctureHamiltonianResult(
            self.coordinates,
            correction,
            conformal,
            residual,
            initial_norm,
            final_norm,
            nonlinear.diagnostics.iterations,
            nonlinear.diagnostics.linear_iterations,
            nonlinear.status,
            initial_data,
            tuning,
            ScientificStatus(
                status_value,
                finite,
                converged,
                physical,
                qualified,
                False,
            ),
            restart,
            self.plan_id,
            content_id,
            True,
            self.operator_storage_bytes,
        )

    def tuning_evidence(
        self,
        correction: ArrayLike,
        /,
        *,
        converged: ArrayLike = True,
        physically_valid: ArrayLike = True,
    ) -> PunctureTuningEvidence:
        value = jnp.asarray(correction)
        if value.shape != self.shape:
            raise ValueError("Tuning evidence correction must match the plan shape.")
        singular, conformal_extrinsic, _ = _conformal_fields(
            self.punctures, self.coordinates
        )
        conformal = singular + value
        extrinsic_squared = ein.contract(
            "...ij,...ij->...", conformal_extrinsic, conformal_extrinsic
        )
        energy_density = conformal ** (-7) * extrinsic_squared
        mass_correction = ein.contract(
            "ijk,ijk->", self.volume_weights, energy_density
        ) / (16.0 * pi)
        bare_mass = sum(item.bare_mass for item in self.punctures)
        achieved_mass = bare_mass + mass_correction
        achieved_momentum = sum(
            (item.linear_momentum for item in self.punctures),
            start=jnp.zeros((3,), dtype=value.dtype),
        )
        achieved_spin = sum(
            (item.spin for item in self.punctures),
            start=jnp.zeros((3,), dtype=value.dtype),
        )
        achieved_angular = sum(
            (
                item.spin + jnp.cross(item.position, item.linear_momentum)
                for item in self.punctures
            ),
            start=jnp.zeros((3,), dtype=value.dtype),
        )
        mass_error = achieved_mass - self.target_adm_mass
        momentum_error = achieved_momentum - self.target_linear_momentum
        spin_error = achieved_spin - self.target_spin
        angular_error = achieved_angular - self.target_angular_momentum
        finite = (
            jnp.isfinite(achieved_mass)
            & jnp.all(jnp.isfinite(achieved_momentum))
            & jnp.all(jnp.isfinite(achieved_spin))
            & jnp.all(jnp.isfinite(achieved_angular))
        )
        converged_ = jnp.asarray(converged, dtype=bool).reshape(())
        physical_ = jnp.asarray(physically_valid, dtype=bool).reshape(())
        qualified = (
            finite
            & converged_
            & physical_
            & (jnp.abs(mass_error) <= self.mass_tolerance)
            & (jnp.max(jnp.abs(spin_error)) <= self.charge_tolerance)
            & (jnp.max(jnp.abs(momentum_error)) <= self.charge_tolerance)
            & (jnp.max(jnp.abs(angular_error)) <= self.charge_tolerance)
        )
        return PunctureTuningEvidence(
            self.target_adm_mass,
            achieved_mass,
            mass_error,
            self.target_linear_momentum,
            achieved_momentum,
            momentum_error,
            self.target_spin,
            achieved_spin,
            spin_error,
            self.target_angular_momentum,
            achieved_angular,
            angular_error,
            finite,
            converged_,
            physical_,
            qualified,
            False,
            canonical_fingerprint(
                {
                    "kind": "two-puncture-charge-tuning-evidence",
                    "plan": self.plan_id,
                    "mass_tolerance": self.mass_tolerance,
                    "charge_tolerance": self.charge_tolerance,
                }
            ),
        )


def brill_lindquist_initial_data(
    coordinates: ArrayLike, punctures: tuple[Puncture, ...], /
) -> ADMInitialData:
    return BrillLindquistInitialData(punctures)(coordinates)


def bowen_york_initial_data(
    coordinates: ArrayLike, punctures: tuple[Puncture, ...], /
) -> ADMInitialData:
    return BowenYorkInitialData(punctures)(coordinates)


def _puncture_tuple(value, /, *, minimum):
    punctures = tuple(value)
    if len(punctures) < minimum or any(not isinstance(item, Puncture) for item in punctures):
        raise TypeError(f"punctures must contain at least {minimum} Puncture objects.")
    identifiers = tuple(item.puncture_id for item in punctures)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Puncture identities must be unique.")
    return tuple(sorted(punctures, key=lambda item: item.puncture_id))


def _triple_int(value, name, /):
    if isinstance(value, int):
        return (int(value),) * 3
    values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or three values.")
    return values


def _triple_float(value, name, /):
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or three values.")
    return values


def _clenshaw_curtis_weights(nodes):
    """Polynomially exact Lobatto quadrature weights on one bounded axis."""

    values = np.asarray(nodes, dtype=float)
    intervals = values.size - 1
    theta = np.pi * np.arange(values.size, dtype=float) / intervals
    weights = np.empty_like(values)
    interior = np.ones((intervals - 1,), dtype=float)
    if intervals % 2 == 0:
        endpoint = 1.0 / (intervals**2 - 1.0)
        for mode in range(1, intervals // 2):
            interior -= (
                2.0
                * np.cos(2.0 * mode * theta[1:-1])
                / (4.0 * mode**2 - 1.0)
            )
        interior -= np.cos(intervals * theta[1:-1]) / (intervals**2 - 1.0)
    else:
        endpoint = 1.0 / intervals**2
        for mode in range(1, (intervals + 1) // 2):
            interior -= (
                2.0
                * np.cos(2.0 * mode * theta[1:-1])
                / (4.0 * mode**2 - 1.0)
            )
    weights[[0, -1]] = endpoint
    weights[1:-1] = 2.0 * interior / intervals
    return 0.5 * (values[-1] - values[0]) * weights


def _conformal_fields(punctures, coordinates):
    points = jnp.asarray(coordinates)
    leading = points.shape[:-1]
    dtype = points.dtype
    conformal = jnp.ones(leading, dtype=dtype)
    conformal_extrinsic = jnp.zeros(leading + (3, 3), dtype=dtype)
    domain = jnp.ones(leading, dtype=bool)
    identity = jnp.eye(3, dtype=dtype)
    for puncture in punctures:
        offset = points - puncture.position.astype(dtype)
        radius = jnp.sqrt(ein.contract("...i,...i->...", offset, offset))
        point_valid = radius > 0.0
        safe_radius = jnp.where(point_valid, radius, jnp.asarray(1.0, dtype=dtype))
        normal = offset / safe_radius[..., None]
        mass = puncture.bare_mass.astype(dtype)
        momentum = puncture.linear_momentum.astype(dtype)
        spin = puncture.spin.astype(dtype)
        conformal = conformal + mass / (2.0 * safe_radius)
        momentum_dot_normal = ein.contract("i,...i->...", momentum, normal)
        momentum_term = 1.5 / safe_radius[..., None, None] ** 2 * (
            ein.contract("i,...j->...ij", momentum, normal)
            + ein.contract("...i,j->...ij", normal, momentum)
            - (
                identity
                - ein.contract("...i,...j->...ij", normal, normal)
            )
            * momentum_dot_normal[..., None, None]
        )
        spin_cross_normal = jnp.cross(spin, normal)
        spin_term = 3.0 / safe_radius[..., None, None] ** 3 * (
            ein.contract("...i,...j->...ij", normal, spin_cross_normal)
            + ein.contract("...i,...j->...ij", spin_cross_normal, normal)
        )
        conformal_extrinsic = conformal_extrinsic + jnp.where(
            point_valid[..., None, None], momentum_term + spin_term, 0.0
        )
        domain = domain & point_valid
    return conformal, conformal_extrinsic, domain


def _puncture_initial_data(
    punctures,
    coordinates,
    correction,
    /,
    *,
    include_bowen_york,
    data_id,
):
    points = jnp.asarray(coordinates)
    value = jnp.asarray(correction, dtype=points.dtype)
    if value.shape == ():
        value = jnp.broadcast_to(value, points.shape[:-1])
    if value.shape != points.shape[:-1]:
        raise ValueError("Hamiltonian correction must match coordinate leading shape.")
    singular, conformal_extrinsic, domain = _conformal_fields(punctures, points)
    conformal = singular + value
    leading = points.shape[:-1]
    identity = jnp.broadcast_to(jnp.eye(3, dtype=points.dtype), leading + (3, 3))
    metric = conformal[..., None, None] ** 4 * identity
    extrinsic = (
        conformal[..., None, None] ** (-2) * conformal_extrinsic
        if include_bowen_york
        else jnp.zeros_like(metric)
    )
    lapse = conformal ** (-2)
    return _package_initial_data(
        points,
        lapse,
        jnp.zeros(leading + (3,), dtype=points.dtype),
        metric,
        extrinsic,
        conformal,
        domain & (conformal > 0.0),
        data_id,
    )

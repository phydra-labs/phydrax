#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, ClassVar, Literal

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from diffrax._term import WrapTerm
from jaxtyping import Array

from .._strict import StrictModule
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry


class GeometricLocalInterpolation(dfx.AbstractLocalInterpolation):
    """One-step interpolation that evaluates through a state retraction."""

    t0: Array  # ty: ignore[invalid-attribute-override]
    t1: Array  # ty: ignore[invalid-attribute-override]
    y0: Array
    y1: Array
    local_increment: Array
    geometry: AbstractStateGeometry

    def evaluate(self, t0, t1=None, left: bool = True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        weight = (t0 - self.t0) / (self.t1 - self.t0)
        interior = self.geometry.retract(self.y0, weight * self.local_increment)
        return jnp.where(
            weight <= 0.0,
            self.y0,
            jnp.where(weight >= 1.0, self.y1, interior),
        )

    def derivative(self, t, left: bool = True):
        del left
        weight = (t - self.t0) / (self.t1 - self.t0)
        local = weight * self.local_increment
        local_velocity = self.local_increment / (self.t1 - self.t0)
        return self.geometry.retraction_jvp(
            self.y0,
            local,
            local_velocity,
        )


class AbstractGeometricSolver(dfx.AbstractSolver):
    """Diffrax solver contract for explicit retraction stages.

    ``stage_abscissae`` statically enumerates every time coefficient used by
    ``step``. ``causal_stage_extent`` is a finite positive static bound on those
    coefficients; delay backends use it to certify fixed steps before tracing
    the numerical loop.
    """

    geometry: AbstractStateGeometry
    solver_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)
    stage_abscissae: tuple[float, ...] = eqx.field(static=True)
    causal_stage_extent: float = eqx.field(static=True)


def _require_exact_differential(
    geometry: AbstractStateGeometry,
    owner: str,
    /,
) -> None:
    if not geometry.supports_exact_differential:
        raise ValueError(
            f"{owner} requires geometry with exact retraction differential capability."
        )


def _term_vector_field(
    term: dfx.AbstractTerm,
    time: Array,
    state: Array,
    args: Any,
    /,
) -> Array:
    """Evaluate an ODE field without coercing a physical tangent to point shape."""
    if isinstance(term, WrapTerm):
        return _term_vector_field(
            term.term,
            time * term.direction,
            state,
            args,
        )
    if isinstance(term, dfx.ODETerm):
        return term.vector_field(time, state, args)
    return term.vf(time, state, args)


class GeometricODETerm(dfx.ODETerm):
    """ODE term whose vector field retains physical tangent coordinates."""

    def vf(self, t, y, args):
        return self.vector_field(t, y, args)


def _term_vf_prod(
    term: dfx.AbstractTerm,
    time: Array,
    state: Array,
    args: Any,
    control: Any,
    /,
) -> Array:
    return term.prod(_term_vector_field(term, time, state, args), control)


def _physical_tangent(
    geometry: AbstractStateGeometry,
    state: Array,
    value: Array,
    owner: str,
    /,
) -> Array:
    point = jnp.asarray(state)
    candidate = jnp.asarray(value)
    tangent_zero = jnp.asarray(geometry.project_tangent(point, jnp.zeros_like(point)))
    if candidate.shape != tangent_zero.shape:
        raise ValueError(
            f"{owner} must have physical tangent shape {tangent_zero.shape}; "
            f"got {candidate.shape}."
        )
    if candidate.shape != point.shape:
        return candidate
    projected = jnp.asarray(geometry.project_tangent(point, candidate))
    if projected.shape != tangent_zero.shape:
        raise ValueError(
            f"{owner} projection must preserve physical tangent shape "
            f"{tangent_zero.shape}; got {projected.shape}."
        )
    return projected


def _local_zero(geometry: AbstractStateGeometry, state: Array, /) -> Array:
    point = jnp.asarray(state)
    tangent_zero = jnp.asarray(geometry.project_tangent(point, jnp.zeros_like(point)))
    return jnp.asarray(geometry.retraction_inverse_jvp(point, point, tangent_zero))


def _local_velocity(
    geometry: AbstractStateGeometry,
    state: Array,
    point: Array,
    tangent: Array,
    /,
) -> Array:
    return jnp.asarray(geometry.retraction_inverse_jvp(state, point, tangent))


class GeometricEuler(AbstractGeometricSolver):
    """First-order retraction Euler method for deterministic dynamics."""

    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )

    def __init__(self, geometry: AbstractStateGeometry, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("GeometricEuler geometry must be an AbstractStateGeometry.")
        _require_exact_differential(geometry, "GeometricEuler")
        self.geometry = geometry
        self.solver_id = f"solver:geometric-euler:{geometry.geometry_id}"
        self.resolved_method = f"euler:{geometry.retraction_method}"
        self.stage_abscissae = (0.0,)
        self.causal_stage_extent = 1.0

    def order(self, terms):
        del terms
        return 1

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        value = _term_vf_prod(terms, t0, y0, args, terms.contr(t0, t1))
        tangent = _physical_tangent(
            self.geometry,
            y0,
            value,
            "GeometricEuler vector field",
        )
        local = _local_velocity(self.geometry, y0, y0, tangent)
        y1 = self.geometry.retract(y0, local)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return _term_vector_field(terms, t0, y0, args)


class SeparableHamiltonianVectorField(StrictModule):
    """Canonical vector field for ``H(q, p) = V(q) + T(p)``."""

    potential_gradient: Callable[[Array, Array, Any], Array]
    kinetic_gradient: Callable[[Array, Array, Any], Array]
    configuration_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        potential_gradient: Callable[[Array, Array, Any], Array],
        kinetic_gradient: Callable[[Array, Array, Any], Array],
        configuration_dimension: int,
        /,
    ):
        if not callable(potential_gradient) or not callable(kinetic_gradient):
            raise TypeError("Hamiltonian gradients must be callable.")
        if int(configuration_dimension) <= 0:
            raise ValueError("configuration_dimension must be positive.")
        self.potential_gradient = potential_gradient
        self.kinetic_gradient = kinetic_gradient
        self.configuration_dimension = int(configuration_dimension)

    def split(self, state: Array, /) -> tuple[Array, Array]:
        state_array = jnp.asarray(state)
        expected = 2 * self.configuration_dimension
        if state_array.shape[-1] != expected:
            raise ValueError(
                f"Canonical phase state must have trailing dimension {expected}; "
                f"got {state_array.shape[-1]}."
            )
        return (
            state_array[..., : self.configuration_dimension],
            state_array[..., self.configuration_dimension :],
        )

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        configuration, momentum = self.split(state)
        velocity = jnp.asarray(self.kinetic_gradient(time, momentum, args))
        force = -jnp.asarray(self.potential_gradient(time, configuration, args))
        if velocity.shape != configuration.shape or force.shape != momentum.shape:
            raise ValueError(
                "Hamiltonian gradients must have the corresponding phase-state shape."
            )
        return jnp.concatenate((velocity, force), axis=-1)


def _separable_hamiltonian_vector_field(
    terms: dfx.ODETerm | WrapTerm,
    /,
) -> tuple[SeparableHamiltonianVectorField, Array]:
    if isinstance(terms, WrapTerm):
        term = terms.term
        direction = jnp.asarray(terms.direction)
    else:
        term = terms
        direction = jnp.asarray(1)
    vector_field = term.vector_field if isinstance(term, dfx.ODETerm) else None
    if not isinstance(vector_field, SeparableHamiltonianVectorField):
        raise TypeError(
            "StormerVerlet requires an ODETerm containing "
            "SeparableHamiltonianVectorField."
        )
    return vector_field, direction


class StormerVerlet(AbstractGeometricSolver):
    """Second-order symplectic integrator for separable canonical Hamiltonians."""

    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )
    configuration_dimension: int = eqx.field(static=True)

    def __init__(self, configuration_dimension: int, /):
        if int(configuration_dimension) <= 0:
            raise ValueError("configuration_dimension must be positive.")
        self.configuration_dimension = int(configuration_dimension)
        self.geometry = EuclideanStateGeometry(
            geometry_id="state-geometry:canonical-phase"
        )
        self.solver_id = "solver:stormer-verlet:canonical"
        self.resolved_method = "kick-drift-kick"
        self.stage_abscissae = (0.0, 0.5, 1.0)
        self.causal_stage_extent = 1.0

    def order(self, terms):
        del terms
        return 2

    def init(self, terms, t0, t1, y0, args):
        del t0, t1, args
        vector_field, _ = _separable_hamiltonian_vector_field(terms)
        if vector_field.configuration_dimension != self.configuration_dimension:
            raise ValueError(
                "Solver and Hamiltonian vector field configuration dimensions differ."
            )
        vector_field.split(y0)
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        vector_field, direction = _separable_hamiltonian_vector_field(terms)
        configuration, momentum = vector_field.split(y0)
        dt = terms.contr(t0, t1)
        half_momentum = momentum - 0.5 * dt * vector_field.potential_gradient(
            t0 * direction,
            configuration,
            args,
        )
        next_configuration = configuration + dt * vector_field.kinetic_gradient(
            (t0 + 0.5 * (t1 - t0)) * direction,
            half_momentum,
            args,
        )
        next_momentum = half_momentum - 0.5 * dt * vector_field.potential_gradient(
            t1 * direction,
            next_configuration,
            args,
        )
        y1 = jnp.concatenate((next_configuration, next_momentum), axis=-1)
        local_increment = y1 - y0
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local_increment,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


RKMKMethod = Literal["midpoint", "rk4"]


class RKMK(AbstractGeometricSolver):
    """Explicit Runge--Kutta--Munthe-Kaas method in retraction coordinates."""

    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )
    method: RKMKMethod = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractStateGeometry,
        /,
        *,
        method: RKMKMethod = "rk4",
    ):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("RKMK geometry must be an AbstractStateGeometry.")
        _require_exact_differential(geometry, "RKMK")
        if method not in ("midpoint", "rk4"):
            raise ValueError("RKMK method must be 'midpoint' or 'rk4'.")
        self.geometry = geometry
        self.method = method
        self.solver_id = f"solver:rkmk:{method}:{geometry.geometry_id}"
        self.resolved_method = f"{method}:{geometry.retraction_method}"
        self.stage_abscissae = (
            (0.0, 0.5) if method == "midpoint" else (0.0, 0.5, 0.5, 1.0)
        )
        self.causal_stage_extent = max(self.stage_abscissae)

    def order(self, terms):
        del terms
        return 2 if self.method == "midpoint" else 4

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def _tableau(self):
        if self.method == "midpoint":
            return (
                self.stage_abscissae,
                ((), (0.5,)),
                (0.0, 1.0),
            )
        return (
            self.stage_abscissae,
            ((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
            (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        )

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0
        abscissae, coefficients, weights = self._tableau()
        local_zero = _local_zero(self.geometry, y0)
        stages: list[Array] = []
        for abscissa, row in zip(abscissae, coefficients, strict=True):
            local = jnp.zeros_like(local_zero)
            for coefficient, stage in zip(row, stages, strict=True):
                local = local + coefficient * dt * stage
            point = self.geometry.retract(y0, local)
            value = _term_vector_field(terms, t0 + abscissa * dt, point, args)
            tangent = _physical_tangent(
                self.geometry,
                point,
                value,
                "RKMK vector field",
            )
            stages.append(_local_velocity(self.geometry, y0, point, tangent))
        local_increment = jnp.zeros_like(local_zero)
        for weight, stage in zip(weights, stages, strict=True):
            local_increment = local_increment + weight * dt * stage
        y1 = self.geometry.retract(y0, local_increment)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local_increment,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return _term_vector_field(terms, t0, y0, args)


class CommutatorFreeTableau(StrictModule):
    """Explicit stages and exponential compositions for a CF Lie integrator."""

    abscissae: tuple[float, ...] = eqx.field(static=True)
    stage_coefficients: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    composition_coefficients: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    order: int = eqx.field(static=True)
    tableau_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        abscissae: Sequence[float],
        stage_coefficients: Sequence[Sequence[float]],
        composition_coefficients: Sequence[Sequence[float]],
        order: int,
        tableau_id: str,
    ):
        nodes = tuple(float(value) for value in abscissae)
        stages = tuple(tuple(float(value) for value in row) for row in stage_coefficients)
        compositions = tuple(
            tuple(float(value) for value in row) for row in composition_coefficients
        )
        if not nodes or len(stages) != len(nodes):
            raise ValueError("A commutator-free tableau needs one row per stage.")
        if any(not isfinite(node) or node < 0.0 for node in nodes):
            raise ValueError(
                "Commutator-free stage abscissae must be finite and nonnegative."
            )
        if any(len(row) != index for index, row in enumerate(stages)):
            raise ValueError("Stage coefficient rows must be strictly lower triangular.")
        if not compositions or any(len(row) != len(nodes) for row in compositions):
            raise ValueError(
                "Every composition row must provide one coefficient per stage."
            )
        if int(order) <= 0:
            raise ValueError("Commutator-free tableau order must be positive.")
        if not isinstance(tableau_id, str) or not tableau_id:
            raise ValueError("tableau_id must be a non-empty string.")
        self.abscissae = nodes
        self.stage_coefficients = stages
        self.composition_coefficients = compositions
        self.order = int(order)
        self.tableau_id = tableau_id


def commutator_free_midpoint_tableau() -> CommutatorFreeTableau:
    """Return the explicit two-stage, second-order CF midpoint tableau."""
    return CommutatorFreeTableau(
        abscissae=(0.0, 1.0),
        stage_coefficients=((), (1.0,)),
        composition_coefficients=((0.5, 0.5),),
        order=2,
        tableau_id="tableau:commutator-free-midpoint",
    )


class CommutatorFreeSolver(AbstractGeometricSolver):
    """Tableau-driven composition method requiring no commutator evaluation."""

    term_structure: ClassVar = dfx.AbstractTerm
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )
    tableau: CommutatorFreeTableau

    def __init__(
        self,
        geometry: AbstractStateGeometry,
        /,
        *,
        tableau: CommutatorFreeTableau | None = None,
    ):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError(
                "CommutatorFreeSolver geometry must be an AbstractStateGeometry."
            )
        _require_exact_differential(geometry, "CommutatorFreeSolver")
        if not geometry.supports_exact_inverse:
            raise ValueError(
                "CommutatorFreeSolver requires exact inverse-retraction capability."
            )
        if not geometry.supports_commutator_free:
            raise ValueError(
                "CommutatorFreeSolver requires shared-trivialization capability."
            )
        resolved = commutator_free_midpoint_tableau() if tableau is None else tableau
        if not isinstance(resolved, CommutatorFreeTableau):
            raise TypeError("tableau must be a CommutatorFreeTableau or None.")
        self.geometry = geometry
        self.tableau = resolved
        self.solver_id = (
            f"solver:commutator-free:{resolved.tableau_id}:{geometry.geometry_id}"
        )
        self.resolved_method = f"{resolved.tableau_id}:{geometry.retraction_method}"
        self.stage_abscissae = resolved.abscissae
        self.causal_stage_extent = max(resolved.abscissae) or 1.0

    def order(self, terms):
        del terms
        return self.tableau.order

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0
        local_zero = _local_zero(self.geometry, y0)
        stages: list[Array] = []
        for abscissa, row in zip(
            self.tableau.abscissae,
            self.tableau.stage_coefficients,
            strict=True,
        ):
            local = jnp.zeros_like(local_zero)
            for coefficient, stage in zip(row, stages, strict=True):
                local = local + coefficient * dt * stage
            point = self.geometry.retract(y0, local)
            value = _term_vector_field(terms, t0 + abscissa * dt, point, args)
            tangent = _physical_tangent(
                self.geometry,
                point,
                value,
                "Commutator-free vector field",
            )
            stages.append(_local_velocity(self.geometry, point, point, tangent))
        y1 = y0
        for row in self.tableau.composition_coefficients:
            local = jnp.zeros_like(local_zero)
            for coefficient, stage in zip(row, stages, strict=True):
                local = local + coefficient * dt * stage
            y1 = self.geometry.retract(y1, local)
        local_increment = self.geometry.inverse_retract(y0, y1)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local_increment,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return _term_vector_field(terms, t0, y0, args)


class SRKMK(AbstractGeometricSolver, dfx.AbstractStratonovichSolver):
    """Explicit first-order Stratonovich RKMK method for drift plus diffusion."""

    term_structure: ClassVar = dfx.MultiTerm[tuple[dfx.AbstractTerm, dfx.AbstractTerm]]
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )

    def __init__(self, geometry: AbstractStateGeometry, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("SRKMK geometry must be an AbstractStateGeometry.")
        _require_exact_differential(geometry, "SRKMK")
        self.geometry = geometry
        self.solver_id = f"solver:srkmk:{geometry.geometry_id}"
        self.resolved_method = f"stratonovich-heun:{geometry.retraction_method}"
        self.stage_abscissae = (0.0, 1.0)
        self.causal_stage_extent = 1.0

    def order(self, terms):
        del terms
        return 1

    def strong_order(self, terms):
        del terms
        return 0.5

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        dw = diffusion.contr(t0, t1)
        drift_value = _term_vf_prod(drift, t0, y0, args, dt)
        diffusion_value = _term_vf_prod(diffusion, t0, y0, args, dw)
        drift_tangent = _physical_tangent(
            self.geometry,
            y0,
            drift_value,
            "SRKMK drift",
        )
        diffusion_tangent = _physical_tangent(
            self.geometry,
            y0,
            diffusion_value,
            "SRKMK diffusion",
        )
        drift_local = _local_velocity(
            self.geometry,
            y0,
            y0,
            drift_tangent,
        )
        diffusion_local = _local_velocity(
            self.geometry,
            y0,
            y0,
            diffusion_tangent,
        )
        predictor = self.geometry.retract(y0, diffusion_local)
        corrected_value = _term_vf_prod(diffusion, t1, predictor, args, dw)
        corrected_tangent = _physical_tangent(
            self.geometry,
            predictor,
            corrected_value,
            "SRKMK corrected diffusion",
        )
        corrected_local = _local_velocity(
            self.geometry,
            y0,
            predictor,
            corrected_tangent,
        )
        local_increment = drift_local + 0.5 * (diffusion_local + corrected_local)
        y1 = self.geometry.retract(y0, local_increment)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local_increment,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        drift, diffusion = terms.terms
        return (
            _term_vector_field(drift, t0, y0, args),
            _term_vector_field(diffusion, t0, y0, args),
        )


def solver_state_geometry(
    solver: AbstractGeometricSolver,
    /,
) -> AbstractStateGeometry:
    """Resolve the geometry through the explicit geometric-solver contract."""
    if not isinstance(solver, AbstractGeometricSolver):
        raise TypeError("solver does not implement the geometric-solver contract.")
    return solver.geometry


__all__ = [
    "AbstractGeometricSolver",
    "CommutatorFreeSolver",
    "CommutatorFreeTableau",
    "GeometricEuler",
    "GeometricLocalInterpolation",
    "RKMK",
    "SeparableHamiltonianVectorField",
    "SRKMK",
    "StormerVerlet",
    "commutator_free_midpoint_tableau",
    "solver_state_geometry",
]

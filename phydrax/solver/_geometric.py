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
from jaxtyping import Array

from .._strict import StrictModule
from ..metrix import AbstractStateGeometry


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


class GeometricEuler(AbstractGeometricSolver):
    """First-order retraction Euler method for deterministic dynamics."""

    term_structure: ClassVar = dfx.ODETerm
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )

    def __init__(self, geometry: AbstractStateGeometry, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("GeometricEuler geometry must be an AbstractStateGeometry.")
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
        ambient = terms.vf_prod(t0, y0, args, terms.contr(t0, t1))
        tangent = self.geometry.project_tangent(y0, ambient)
        local = self.geometry.to_local(y0, tangent)
        y1 = self.geometry.retract(y0, local)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


RKMKMethod = Literal["midpoint", "rk4"]


class RKMK(AbstractGeometricSolver):
    """Explicit Runge--Kutta--Munthe-Kaas method in retraction coordinates."""

    term_structure: ClassVar = dfx.ODETerm
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
        if not geometry.supports_exact_pullback:
            raise ValueError(
                "RKMK requires geometry with explicit exact pullback capability."
            )
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
        retraction = self.geometry.local_retraction(y0)
        stages: list[Array] = []
        for abscissa, row in zip(abscissae, coefficients, strict=True):
            local = jnp.zeros_like(y0)
            for coefficient, stage in zip(row, stages, strict=True):
                local = local + coefficient * dt * stage
            point = retraction.evaluate(local)
            ambient = terms.vf(t0 + abscissa * dt, point, args)
            tangent = self.geometry.project_tangent(point, ambient)
            stages.append(retraction.pullback(local, tangent))
        local_increment = jnp.zeros_like(y0)
        for weight, stage in zip(weights, stages, strict=True):
            local_increment = local_increment + weight * dt * stage
        y1 = retraction.evaluate(local_increment)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local_increment,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


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

    term_structure: ClassVar = dfx.ODETerm
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
        stages: list[Array] = []
        for abscissa, row in zip(
            self.tableau.abscissae,
            self.tableau.stage_coefficients,
            strict=True,
        ):
            local = jnp.zeros_like(y0)
            for coefficient, stage in zip(row, stages, strict=True):
                local = local + coefficient * dt * stage
            point = self.geometry.retract(y0, local)
            ambient = terms.vf(t0 + abscissa * dt, point, args)
            tangent = self.geometry.project_tangent(point, ambient)
            stages.append(self.geometry.to_local(point, tangent))
        y1 = y0
        for row in self.tableau.composition_coefficients:
            local = jnp.zeros_like(y0)
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
        return terms.vf(t0, y0, args)


class SRKMK(AbstractGeometricSolver, dfx.AbstractStratonovichSolver):
    """Explicit first-order Stratonovich RKMK method for drift plus diffusion."""

    term_structure: ClassVar = dfx.MultiTerm[
        tuple[dfx.AbstractTerm[Any, Any], dfx.AbstractTerm]
    ]
    interpolation_cls: ClassVar[Callable[..., GeometricLocalInterpolation]] = (
        GeometricLocalInterpolation
    )

    def __init__(self, geometry: AbstractStateGeometry, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("SRKMK geometry must be an AbstractStateGeometry.")
        if not geometry.supports_exact_pullback:
            raise ValueError(
                "SRKMK requires geometry with explicit exact pullback capability."
            )
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
        drift_ambient = drift.vf_prod(t0, y0, args, dt)
        diffusion_ambient = diffusion.vf_prod(t0, y0, args, dw)
        retraction = self.geometry.local_retraction(y0)
        zero = jnp.zeros_like(y0)
        drift_local = retraction.pullback(
            zero,
            self.geometry.project_tangent(y0, drift_ambient),
        )
        diffusion_local = retraction.pullback(
            zero,
            self.geometry.project_tangent(y0, diffusion_ambient),
        )
        predictor = retraction.evaluate(diffusion_local)
        corrected_ambient = diffusion.vf_prod(t1, predictor, args, dw)
        corrected_local = retraction.pullback(
            diffusion_local,
            self.geometry.project_tangent(predictor, corrected_ambient),
        )
        local_increment = drift_local + 0.5 * (diffusion_local + corrected_local)
        y1 = retraction.evaluate(local_increment)
        dense_info = dict(
            y0=y0,
            y1=y1,
            local_increment=local_increment,
            geometry=self.geometry,
        )
        return y1, None, dense_info, None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)


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
    "SRKMK",
    "commutator_free_midpoint_tableau",
    "solver_state_geometry",
]

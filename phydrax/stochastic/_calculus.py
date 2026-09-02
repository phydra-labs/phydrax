#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import AbstractStateGeometry


class StratonovichCorrectionResult(StrictModule):
    """Intrinsic correction and the geometry evidence gating its use."""

    correction: Array
    tangent_residual: Array
    connection_residual: Array
    rank_margin: Array
    precision_evidence: Any
    valid: Array
    status: Array
    geometry_id: str = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)


class _CombinedDiffusion(eqx.Module):
    terms: tuple[Any, ...]
    time: Array
    args: Any

    def __call__(self, state: Array, /) -> Array:
        matrices = tuple(
            term.coefficient_matrix(self.time, state, self.args) for term in self.terms
        )
        flattened = tuple(matrix.reshape((state.size, -1)) for matrix in matrices)
        return jnp.concatenate(flattened, axis=-1).reshape(state.shape + (-1,))


class _ItoDrift(eqx.Module):
    drift: Any
    terms: tuple[Any, ...]
    geometry: AbstractStateGeometry | None
    tangent_evidence: Any
    precision: Any

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        diffusion = _CombinedDiffusion(self.terms, time, args)
        evidence = stratonovich_correction(
            diffusion,
            state,
            geometry=self.geometry,
            tangent_evidence=self.tangent_evidence,
            args=args,
            precision=self.precision,
        )
        correction = eqx.error_if(
            evidence.correction,
            ~evidence.valid,
            "Stratonovich correction geometry validation failed.",
        )
        return jnp.asarray(self.drift(time, state, args)) + correction


def _euclidean_correction(vector_fields: Any, state: Array, /) -> tuple[Array, Array]:
    sigma = jnp.asarray(vector_fields(state))
    if sigma.shape[:-1] != state.shape or sigma.shape[-1] == 0:
        raise ValueError("vector_fields must return state.shape + (driver_dimension,).")
    directions = jnp.moveaxis(sigma, -1, 0)
    indices = jnp.arange(sigma.shape[-1])

    def one(index, direction):
        column = lambda value: jnp.asarray(vector_fields(value))[..., index]
        _, derivative = jax.jvp(column, (state,), (direction,))
        return derivative

    return 0.5 * jnp.sum(jax.vmap(one)(indices, directions), axis=0), sigma


def stratonovich_correction(
    vector_fields: Any,
    state: ArrayLike,
    /,
    *,
    geometry: AbstractStateGeometry | None = None,
    tangent_evidence: Any = None,
    args: Any = None,
    precision: Any = None,
) -> StratonovichCorrectionResult:
    """Compute ``1/2 Σ_a ∇_{V_a}V_a`` on a declared represented geometry.

    The Euclidean route is the ordinary directional-JVP formula.  The manifold route
    projects both vector fields and directional derivatives with the supplied state
    geometry.  Optional GTA tangent evidence gates rank and chart validity without
    attempting to infer geometry from covariance samples.
    """
    del args
    if not callable(vector_fields):
        raise TypeError("vector_fields must be callable; covariance is insufficient.")
    point = jnp.asarray(state)
    if point.ndim < 1 or not jnp.issubdtype(point.dtype, jnp.inexact):
        raise ValueError("state must be a real inexact array with at least one axis.")
    ambient, sigma = _euclidean_correction(vector_fields, point)
    finite = jnp.all(jnp.isfinite(point)) & jnp.all(jnp.isfinite(sigma))
    if geometry is None:
        if tangent_evidence is not None:
            raise ValueError("tangent_evidence requires an explicit geometry.")
        correction = ambient
        tangent_residual = jnp.asarray(0.0, dtype=point.dtype)
        connection_residual = jnp.asarray(0.0, dtype=point.dtype)
        rank_margin = jnp.asarray(jnp.inf, dtype=point.dtype)
        evidence_valid = jnp.asarray(True)
        geometry_id = "geometry:euclidean"
        approximation = "euclidean-directional-jvp"
    else:
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("geometry must implement AbstractStateGeometry.")
        directions = jnp.moveaxis(sigma, -1, 0)
        projected_directions = jax.vmap(
            lambda direction: geometry.project_tangent(point, direction)
        )(directions)
        tangent_residual = jnp.max(jnp.abs(directions - projected_directions))
        correction = geometry.project_tangent(point, ambient)
        connection_residual = jnp.max(
            jnp.abs(correction - geometry.project_tangent(point, correction))
        )
        if tangent_evidence is None:
            rank_margin = jnp.asarray(jnp.inf, dtype=point.dtype)
            evidence_valid = jnp.asarray(geometry.contains(point), dtype=bool)
        else:
            projector = jnp.asarray(tangent_evidence.tangent_projector)
            size = int(point.size)
            if projector.shape != (size, size):
                raise ValueError("tangent_evidence projector must match flattened state.")
            evidence_projected = (projector @ sigma.reshape((size, -1))).reshape(
                sigma.shape
            )
            tangent_residual = jnp.maximum(
                tangent_residual, jnp.max(jnp.abs(sigma - evidence_projected))
            )
            rank_margin = jnp.asarray(tangent_evidence.rank_margin)
            evidence_valid = jnp.asarray(tangent_evidence.valid, dtype=bool)
        geometry_id = geometry.geometry_id
        approximation = "projected-covariant-directional-jvp"
    valid = (
        finite
        & evidence_valid
        & jnp.isfinite(tangent_residual)
        & jnp.isfinite(connection_residual)
        & (rank_margin > 0.0)
    )
    status = jnp.where(valid, 0, jnp.where(~finite, 1, 2)).astype(jnp.int32)
    return StratonovichCorrectionResult(
        correction=jnp.where(valid, correction, jnp.zeros_like(correction)),
        tangent_residual=tangent_residual,
        connection_residual=connection_residual,
        rank_margin=rank_margin,
        precision_evidence=precision,
        valid=valid,
        status=status,
        geometry_id=geometry_id,
        approximation_kind=approximation,
    )


def stratonovich_to_ito_problem(
    problem: Any,
    /,
    *,
    tangent_evidence: Any = None,
    precision: Any = None,
):
    """Convert a declared Stratonovich problem and return initial-point evidence."""
    from ..solver._differential import DifferentialProblem

    if not isinstance(problem, DifferentialProblem):
        raise TypeError("problem must be a DifferentialProblem.")
    if problem.interpretation != "stratonovich" or not problem.stochastic:
        raise ValueError("problem must be a stochastic Stratonovich DifferentialProblem.")
    combined = _CombinedDiffusion(problem.wiener_terms, problem.t0, problem.args)
    evidence = stratonovich_correction(
        combined,
        problem.initial_state,
        geometry=problem.state_geometry,
        tangent_evidence=tangent_evidence,
        args=problem.args,
        precision=precision,
    )
    if not bool(evidence.valid):
        raise ValueError("Initial Stratonovich correction evidence is invalid.")
    converted = DifferentialProblem(
        _ItoDrift(
            problem.drift,
            problem.wiener_terms,
            problem.state_geometry,
            tangent_evidence,
            precision,
        ),
        problem.initial_state,
        t0=problem.t0,
        t1=problem.t1,
        args=problem.args,
        wiener_terms=problem.wiener_terms,
        interpretation="ito",
        state_geometry=problem.state_geometry,
        discretization_bundle=problem.discretization_bundle,
        problem_id=f"{problem.problem_id}:ito-covariant",
    )
    return converted, evidence


__all__ = [
    "StratonovichCorrectionResult",
    "stratonovich_correction",
    "stratonovich_to_ito_problem",
]

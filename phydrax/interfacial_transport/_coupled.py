#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative bulk--surface species transport on sparse fixed topology.

Surface control areas exchange species through symmetric nonnegative edge
conductances (``A_s dGamma_s/dt = sum_t c_st (Gamma_t - Gamma_s)``), and each
surface area exchanges with bulk cells through a sparse partition of unity
(``weights`` over each surface area sum to one). One backward-Euler step solves

``A (Gamma' - Gamma) + dt K Gamma' - dt A j(c_loc', Gamma') = 0``,
``V (c' - c) + dt P (A j) = 0``, ``c_loc = P^T c``,

with Langmuir kinetics ``j`` through the native prepared Newton solve. The
committed amounts are recomputed in flux form from the converged
concentrations, so the total amount is conserved to roundoff; a candidate
with any negative amount or over-capacity coverage is rejected, never
clipped.
"""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field
from .._validation import nonnegative_integer, positive_integer
from ..linalg import PreparedSparseFactorization
from ..nonlinear import NonlinearStatus
from ..sparse import EdgeRelation
from ._core import AdsorptionKinetics
from ._film_evidence import FilmStepStatus, resolve_film_status
from ._film_solve import film_termination, PreparedFilmNewton


class BulkSurfaceTransportStep(StrictModule):
    """Accepted and candidate amounts with conservation and solver evidence.

    ``transferred_to_surface_mol`` is the adsorbed amount per surface cell over
    the step (negative for net desorption).
    """

    bulk_amount_mol: Array
    surface_amount_mol: Array
    candidate_bulk_amount_mol: Array
    candidate_surface_amount_mol: Array
    transferred_to_surface_mol: Array
    total_amount_residual_mol: Array
    minimum_amount_mol: Array
    maximum_coverage: Array
    status: Array
    nonlinear_status: Array
    nonlinear_iterations: Array
    nonlinear_residual_norm: Array

    def __init__(
        self,
        *,
        bulk_amount_mol: Array,
        surface_amount_mol: Array,
        candidate_bulk_amount_mol: Array,
        candidate_surface_amount_mol: Array,
        transferred_to_surface_mol: Array,
        total_amount_residual_mol: Array,
        minimum_amount_mol: Array,
        maximum_coverage: Array,
        status: Array,
        nonlinear_status: Array,
        nonlinear_iterations: Array,
        nonlinear_residual_norm: Array,
    ) -> None:
        self.bulk_amount_mol = jnp.asarray(bulk_amount_mol)
        self.surface_amount_mol = jnp.asarray(surface_amount_mol)
        self.candidate_bulk_amount_mol = jnp.asarray(candidate_bulk_amount_mol)
        self.candidate_surface_amount_mol = jnp.asarray(candidate_surface_amount_mol)
        self.transferred_to_surface_mol = jnp.asarray(transferred_to_surface_mol)
        self.total_amount_residual_mol = jnp.asarray(total_amount_residual_mol)
        self.minimum_amount_mol = jnp.asarray(minimum_amount_mol)
        self.maximum_coverage = jnp.asarray(maximum_coverage)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.nonlinear_residual_norm = jnp.asarray(nonlinear_residual_norm)

    @property
    def successful(self) -> Array:
        return self.status == FilmStepStatus.ACCEPTED


class _ExchangeStructure(StrictModule):
    """Sparse surface conductance topology and surface-to-bulk partition."""

    surface_edges: Array
    exchange_relation: EdgeRelation
    exchange_weights: Array
    kinetics: AdsorptionKinetics | None
    bulk_size: int = eqx.field(static=True)
    surface_size: int = eqx.field(static=True)

    def __init__(
        self,
        surface_edges: np.ndarray,
        bulk_index: np.ndarray,
        surface_index: np.ndarray,
        weights: np.ndarray,
        kinetics: AdsorptionKinetics | None,
        bulk_size: int,
        surface_size: int,
        /,
    ) -> None:
        self.surface_edges = jnp.asarray(surface_edges, dtype=jnp.int32)
        self.exchange_relation = EdgeRelation(
            surface_index.astype(np.int32),
            bulk_index.astype(np.int32),
            source_size=surface_size,
            target_size=bulk_size,
        )
        self.exchange_weights = jnp.asarray(weights)
        self.kinetics = kinetics
        self.bulk_size = bulk_size
        self.surface_size = surface_size

    def capacity(self) -> Array:
        if self.kinetics is None:
            return jnp.asarray(jnp.inf)
        return self.kinetics.maximum_surface_concentration_mol_m2

    def local_bulk_concentration(self, bulk_concentration: Array, /) -> Array:
        """Return ``c_loc = P^T c`` seen by each surface cell."""
        relation = self.exchange_relation
        contributions = (
            self.exchange_weights * bulk_concentration[relation.target_indices]
        )
        local = jnp.zeros((self.surface_size,), dtype=contributions.dtype)
        return local.at[relation.source_indices].add(contributions)

    def distribute_to_bulk(self, surface_values: Array, /) -> Array:
        """Return ``P s``: per-bulk totals of surface-cell values."""
        relation = self.exchange_relation
        contributions = self.exchange_weights * surface_values[relation.source_indices]
        bulk = jnp.zeros((self.bulk_size,), dtype=contributions.dtype)
        return bulk.at[relation.target_indices].add(contributions)

    def surface_divergence(self, conductance: Array, concentration: Array, /) -> Array:
        """Return ``K Gamma``: net diffusive outflow per surface cell (mol/s)."""
        first, second = self.surface_edges[:, 0], self.surface_edges[:, 1]
        flux = conductance * (concentration[first] - concentration[second])
        result = jnp.zeros((self.surface_size,), dtype=flux.dtype)
        return result.at[first].add(flux).at[second].add(-flux)

    def exchange_flux(
        self, bulk_concentration: Array, surface_concentration: Array, /
    ) -> Array:
        """Return the adsorption flux per unit area of each surface cell."""
        if self.kinetics is None:
            return jnp.zeros_like(surface_concentration)
        return self.kinetics.flux(
            self.local_bulk_concentration(bulk_concentration), surface_concentration
        )


class CoupledBulkSurfaceTransport(StrictModule):
    """Prepared implicit surface transport with conservative Langmuir exchange.

    ``kinetics=None`` declares a surface-only transport (no bulk cells).
    Exchange routes are ``(bulk_index, surface_index, weight)`` triples; the
    weights of each surface cell must sum to one.
    """

    structure: _ExchangeStructure
    solver: PreparedFilmNewton = fixed_field()
    tolerance: float = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_edges: ArrayLike,
        exchange_bulk_indices: ArrayLike,
        exchange_surface_indices: ArrayLike,
        exchange_weights: ArrayLike,
        kinetics: AdsorptionKinetics | None,
        /,
        *,
        bulk_size: int,
        surface_size: int,
        tolerance: float = 1e-12,
        maximum_iterations: int = 30,
    ) -> None:
        if kinetics is not None and not isinstance(kinetics, AdsorptionKinetics):
            raise TypeError("kinetics must be AdsorptionKinetics or None.")
        bulk_count = nonnegative_integer(bulk_size, "bulk_size")
        surface_count = positive_integer(surface_size, "surface_size")
        edges = np.asarray(surface_edges, dtype=np.int64).reshape((-1, 2))
        bulk_index = np.asarray(exchange_bulk_indices, dtype=np.int64).reshape((-1,))
        surface_index = np.asarray(exchange_surface_indices, dtype=np.int64).reshape(
            (-1,)
        )
        weights = np.asarray(exchange_weights, dtype=np.float64).reshape((-1,))
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if (
            np.any(edges < 0)
            or np.any(edges >= surface_count)
            or np.any(edges[:, 0] == edges[:, 1])
        ):
            raise ValueError("surface_edges must join distinct surface cells.")
        if not (bulk_index.shape == surface_index.shape == weights.shape):
            raise ValueError("Exchange routes must have aligned indices and weights.")
        if kinetics is None:
            if bulk_count != 0 or bulk_index.size != 0:
                raise ValueError("Surface-only transport cannot declare bulk exchange.")
        else:
            if bulk_count == 0:
                raise ValueError("Bulk exchange requires at least one bulk cell.")
            if (
                np.any(bulk_index < 0)
                | np.any(bulk_index >= bulk_count)
                | np.any(surface_index < 0)
                | np.any(surface_index >= surface_count)
            ):
                raise ValueError("Exchange routes reference cells outside the topology.")
            if not np.all(np.isfinite(weights) & (weights >= 0.0)):
                raise ValueError("Exchange weights must be finite and nonnegative.")
            totals = np.bincount(surface_index, weights=weights, minlength=surface_count)
            if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
                raise ValueError("Every surface cell must map completely to bulk.")
        structure = _ExchangeStructure(
            edges, bulk_index, surface_index, weights, kinetics, bulk_count, surface_count
        )
        self.structure = structure
        self.tolerance = tolerance_
        self.transport_id = canonical_fingerprint(
            {
                "kind": "coupled-bulk-surface-transport",
                "surface_edges": edges,
                "exchange": (bulk_index, surface_index, weights),
                "bulk_size": bulk_count,
                "surface_size": surface_count,
                "kinetics": "none" if kinetics is None else "langmuir",
                "time_integration": "backward-euler-flux-form",
            }
        )
        self.solver = PreparedFilmNewton(
            _transport_residual,
            _transport_pattern(
                edges, bulk_index, surface_index, bulk_count, surface_count
            ),
            jnp.zeros((surface_count + bulk_count,)),
            _TransportArguments.sample(structure),
            termination=film_termination(
                tolerance=tolerance_,
                maximum_iterations=positive_integer(
                    maximum_iterations, "maximum_iterations"
                ),
            ),
            solver_id=f"bulk-surface-transport/{self.transport_id}",
        )

    def rates(
        self,
        bulk_amount_mol: Array,
        surface_amount_mol: Array,
        bulk_volume_m3: Array,
        surface_area_m2: Array,
        surface_conductance_m2_s: Array,
        /,
    ) -> tuple[Array, Array]:
        """Return the bulk and surface amount rates (mol/s) of the implicit operator.

        These are the rates whose backward-Euler step ``advance`` solves:
        surface diffusion ``-K Gamma`` plus adsorption, and the matching bulk
        depletion.
        """
        structure = self.structure
        surface = surface_amount_mol / surface_area_m2
        bulk = bulk_amount_mol / bulk_volume_m3
        adsorbed = surface_area_m2 * structure.exchange_flux(bulk, surface)
        return (
            -structure.distribute_to_bulk(adsorbed),
            adsorbed - structure.surface_divergence(surface_conductance_m2_s, surface),
        )

    def _factorization(
        self,
        surface_area_m2: Array,
        surface_conductance_m2_s: Array,
        step_size_s: Array,
        /,
    ) -> PreparedSparseFactorization:
        """Factor the fixed surface-only backward-Euler diffusion operator."""
        structure = self.structure
        if structure.kinetics is not None:
            raise ValueError("Only surface-only diffusion has a fixed linear operator.")
        empty = jnp.zeros((0,), dtype=surface_area_m2.dtype)
        args = _TransportArguments(
            structure,
            empty,
            jnp.zeros_like(surface_area_m2),
            empty,
            surface_area_m2,
            surface_conductance_m2_s,
            step_size_s,
            jnp.ones((), dtype=surface_area_m2.dtype),
        )
        initial = jnp.zeros_like(surface_area_m2)
        return self.solver.factorize(initial, args)

    def advance(
        self,
        bulk_amount_mol: ArrayLike,
        surface_amount_mol: ArrayLike,
        bulk_volume_m3: ArrayLike,
        surface_area_m2: ArrayLike,
        surface_conductance_m2_s: ArrayLike,
        step_size_s: ArrayLike,
        /,
        *,
        factorization: PreparedSparseFactorization | None = None,
    ) -> BulkSurfaceTransportStep:
        """Advance extensive amounts by one conservative backward-Euler step."""
        bulk_amount = jnp.asarray(bulk_amount_mol, dtype=jnp.float64)
        surface_amount = jnp.asarray(surface_amount_mol, dtype=jnp.float64)
        volume = jnp.asarray(bulk_volume_m3, dtype=jnp.float64)
        area = jnp.asarray(surface_area_m2, dtype=jnp.float64)
        conductance = jnp.asarray(surface_conductance_m2_s, dtype=jnp.float64)
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        structure = self.structure
        if bulk_amount.shape != (structure.bulk_size,) or volume.shape != (
            structure.bulk_size,
        ):
            raise ValueError("Bulk amounts and volumes must match the bulk cells.")
        if surface_amount.shape != (structure.surface_size,) or area.shape != (
            structure.surface_size,
        ):
            raise ValueError("Surface amounts and areas must match the surface cells.")
        if conductance.shape != (structure.surface_edges.shape[0],):
            raise ValueError("Surface conductances must match the surface edges.")
        capacity = structure.capacity()
        valid = (
            jnp.all(jnp.isfinite(bulk_amount) & (bulk_amount >= 0.0))
            & jnp.all(jnp.isfinite(surface_amount) & (surface_amount >= 0.0))
            & jnp.all(jnp.isfinite(volume) & (volume > 0.0))
            & jnp.all(jnp.isfinite(area) & (area > 0.0))
            & jnp.all(jnp.isfinite(conductance) & (conductance >= 0.0))
            & jnp.isfinite(step_size)
            & (step_size > 0.0)
            & jnp.all(surface_amount < capacity * area)
        )
        safe_volume = jnp.where(valid, volume, 1.0)
        safe_area = jnp.where(valid, area, 1.0)
        safe_bulk = jnp.where(valid, bulk_amount, 0.0)
        safe_surface = jnp.where(valid, surface_amount, 0.0)
        # Residual rows are amounts over the capacity of a mean surface cell, or
        # over the mean amount per surface cell when no capacity is declared.
        mean_amount = (
            jnp.sum(safe_surface) + jnp.sum(safe_bulk)
        ) / structure.surface_size
        scale = (
            jnp.ones((), dtype=safe_surface.dtype)
            if structure.kinetics is None
            else jnp.where(
                jnp.isfinite(capacity),
                capacity * jnp.mean(safe_area),
                mean_amount,
            )
        )
        scale = jnp.where(scale > 0.0, scale, 1.0)
        args = _TransportArguments(
            structure,
            safe_bulk,
            safe_surface,
            safe_volume,
            safe_area,
            conductance,
            step_size,
            scale,
        )
        initial = jnp.concatenate((safe_surface / safe_area, safe_bulk / safe_volume))
        if factorization is not None and (
            factorization.plan.plan_id != self.solver.factorization.plan_id
        ):
            raise ValueError(
                "Reusable surface-transport factorization belongs to another solver."
            )
        if structure.kinetics is None:
            factor = (
                self.solver.factorize(initial, args)
                if factorization is None
                else factorization
            )
            direct = factor.solve(safe_surface)
            surface_concentration = direct.value
            bulk_concentration = safe_bulk
            converged = direct.success
            nonlinear_status = jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                int(NonlinearStatus.LINEAR_SOLVE_FAILED),
            )
            nonlinear_iterations = jnp.where(converged, 1, 0)
            nonlinear_residual_norm = jnp.linalg.norm(
                _transport_residual(surface_concentration, args)
            )
        else:
            solution = self.solver.solve(initial, args, factorization=factorization)
            surface_concentration = solution.state[: structure.surface_size]
            bulk_concentration = solution.state[structure.surface_size :]
            converged = solution.status == NonlinearStatus.SUCCESS
            nonlinear_status = solution.status
            nonlinear_iterations = solution.diagnostics.iterations
            nonlinear_residual_norm = solution.diagnostics.final_residual_norm
        transferred = (
            step_size
            * safe_area
            * structure.exchange_flux(bulk_concentration, surface_concentration)
        )
        candidate_surface = (
            safe_surface
            - step_size * structure.surface_divergence(conductance, surface_concentration)
            + transferred
        )
        candidate_bulk = safe_bulk - structure.distribute_to_bulk(transferred)
        residual = (
            jnp.sum(candidate_surface)
            + jnp.sum(candidate_bulk)
            - jnp.sum(safe_surface)
            - jnp.sum(safe_bulk)
        )
        finite = jnp.all(jnp.isfinite(candidate_surface)) & jnp.all(
            jnp.isfinite(candidate_bulk)
        )
        minimum = jnp.minimum(
            jnp.min(candidate_surface),
            jnp.min(candidate_bulk) if structure.bulk_size else jnp.inf,
        )
        coverage = jnp.max(candidate_surface / (safe_area * capacity))
        status = resolve_film_status(
            (FilmStepStatus.INADMISSIBLE_INPUT, ~valid),
            (FilmStepStatus.SOLVE_FAILED, ~converged),
            (FilmStepStatus.NONFINITE, ~finite),
            (FilmStepStatus.POSITIVITY_VIOLATED, minimum < 0.0),
            (FilmStepStatus.CAPACITY_EXCEEDED, coverage >= 1.0),
        )
        accepted = status == FilmStepStatus.ACCEPTED
        return BulkSurfaceTransportStep(
            bulk_amount_mol=jnp.where(accepted, candidate_bulk, bulk_amount),
            surface_amount_mol=jnp.where(accepted, candidate_surface, surface_amount),
            candidate_bulk_amount_mol=candidate_bulk,
            candidate_surface_amount_mol=candidate_surface,
            transferred_to_surface_mol=transferred,
            total_amount_residual_mol=residual,
            minimum_amount_mol=minimum,
            maximum_coverage=coverage,
            status=status,
            nonlinear_status=nonlinear_status,
            nonlinear_iterations=nonlinear_iterations,
            nonlinear_residual_norm=nonlinear_residual_norm,
        )


class _TransportArguments(StrictModule):
    structure: _ExchangeStructure
    bulk_amount: Array
    surface_amount: Array
    bulk_volume: Array
    surface_area: Array
    conductance: Array
    step_size: Array
    amount_scale: Array

    def __init__(
        self,
        structure: _ExchangeStructure,
        bulk_amount: Array,
        surface_amount: Array,
        bulk_volume: Array,
        surface_area: Array,
        conductance: Array,
        step_size: Array,
        amount_scale: Array,
        /,
    ) -> None:
        self.structure = structure
        self.bulk_amount = bulk_amount
        self.surface_amount = surface_amount
        self.bulk_volume = bulk_volume
        self.surface_area = surface_area
        self.conductance = conductance
        self.step_size = step_size
        self.amount_scale = amount_scale

    @staticmethod
    def sample(structure: _ExchangeStructure, /) -> _TransportArguments:
        return _TransportArguments(
            structure,
            jnp.zeros((structure.bulk_size,)),
            jnp.zeros((structure.surface_size,)),
            jnp.ones((structure.bulk_size,)),
            jnp.ones((structure.surface_size,)),
            jnp.ones((structure.surface_edges.shape[0],)),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
        )


def _transport_residual(unknowns: Array, args: _TransportArguments) -> Array:
    transport = args.structure
    surface = unknowns[: transport.surface_size]
    bulk = unknowns[transport.surface_size :]
    adsorbed = args.surface_area * transport.exchange_flux(bulk, surface)
    surface_residual = (
        args.surface_area * surface
        - args.surface_amount
        + args.step_size * transport.surface_divergence(args.conductance, surface)
        - args.step_size * adsorbed
    )
    bulk_residual = (
        args.bulk_volume * bulk
        - args.bulk_amount
        + args.step_size * transport.distribute_to_bulk(adsorbed)
    )
    return jnp.concatenate((surface_residual, bulk_residual)) / args.amount_scale


def _transport_pattern(
    edges: np.ndarray,
    bulk_index: np.ndarray,
    surface_index: np.ndarray,
    bulk_count: int,
    surface_count: int,
    /,
) -> EdgeRelation:
    """Return the host sparsity of the coupled surface/bulk residual."""
    size = surface_count + bulk_count
    surface = np.arange(surface_count)
    rows = [surface, edges[:, 0], edges[:, 1]]
    columns = [surface, edges[:, 1], edges[:, 0]]
    if bulk_count:
        routes = sp.csr_matrix(
            (np.ones_like(bulk_index, dtype=np.float64), (bulk_index, surface_index)),
            shape=(bulk_count, surface_count),
        )
        bulk_coupling = (routes @ routes.T).tocoo()
        # A surface row sees bulk cells only through its own partition routes,
        # while a bulk row sees every bulk cell sharing one of its surfaces.
        rows += [
            surface_index,
            surface_count + bulk_index,
            surface_count + np.arange(bulk_count),
            surface_count + bulk_coupling.row,
        ]
        columns += [
            surface_count + bulk_index,
            surface_index,
            surface_count + np.arange(bulk_count),
            surface_count + bulk_coupling.col,
        ]
    return EdgeRelation(
        np.concatenate(columns).astype(np.int32),
        np.concatenate(rows).astype(np.int32),
        source_size=size,
        target_size=size,
    )


__all__ = ["BulkSurfaceTransportStep", "CoupledBulkSurfaceTransport"]

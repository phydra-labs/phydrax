#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..stochastic._path_ensemble import (
    prepare_stochastic_path_ensemble,
    PreparedStochasticPathEnsemble,
    solve_stochastic_path_ensemble,
    StochasticPathEnsemblePlan,
    StochasticPathEnsembleResult,
)
from ._spde import SemidiscreteSPDE


class SPDEApproximationLevel(StrictModule):
    """One executable finite spatial/noise/time truncation level."""

    spde: SemidiscreteSPDE
    temporal_mesh: Any
    transfer_to_reference: Any
    resolution: tuple[int, ...] = eqx.field(static=True)
    work: float = eqx.field(static=True)
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        spde: SemidiscreteSPDE,
        temporal_mesh: Any,
        transfer_to_reference: Any,
        resolution: tuple[int, ...],
        work: float,
        /,
        *,
        level_id: str,
    ):
        if not isinstance(spde, SemidiscreteSPDE):
            raise TypeError("spde must be a SemidiscreteSPDE.")
        if not callable(transfer_to_reference):
            raise TypeError("transfer_to_reference must be callable.")
        resolved = tuple(int(value) for value in resolution)
        effort = float(work)
        if not resolved or any(value <= 0 for value in resolved):
            raise ValueError("resolution must contain positive finite capacities.")
        if not isfinite(effort) or effort <= 0.0:
            raise ValueError("work must be finite and positive.")
        if not isinstance(level_id, str) or not level_id:
            raise ValueError("level_id must be non-empty.")
        self.spde = spde
        self.temporal_mesh = temporal_mesh
        self.transfer_to_reference = transfer_to_reference
        self.resolution = resolved
        self.work = effort
        self.level_id = level_id


class SPDEApproximationFamily(StrictModule):
    """Finite refinement family; it never represents an infinite array."""

    levels: tuple[SPDEApproximationLevel, ...]
    tail_envelope: Any
    refined_axis: Literal["space", "time", "noise-rank"] = eqx.field(static=True)
    reference_level: int = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: tuple[SPDEApproximationLevel, ...],
        refined_axis: Literal["space", "time", "noise-rank"],
        reference_level: int,
        coupling_id: str,
        /,
        *,
        tail_envelope: Any = None,
    ):
        selected = tuple(levels)
        if not selected or any(
            not isinstance(item, SPDEApproximationLevel) for item in selected
        ):
            raise TypeError("levels must contain SPDEApproximationLevel values.")
        if refined_axis not in ("space", "time", "noise-rank"):
            raise ValueError("Unknown refined_axis.")
        reference = int(reference_level)
        if not 0 <= reference < len(selected):
            raise ValueError("reference_level is outside levels.")
        if not isinstance(coupling_id, str) or not coupling_id:
            raise ValueError("coupling_id must be non-empty.")
        if len({item.level_id for item in selected}) != len(selected):
            raise ValueError("SPDE level identities must be unique.")
        common_interval = {
            (float(item.spde.problem.t0), float(item.spde.problem.t1))
            for item in selected
        }
        if len(common_interval) != 1:
            raise ValueError("All SPDE levels must share one physical time interval.")
        self.levels = selected
        self.refined_axis = refined_axis
        self.reference_level = reference
        self.coupling_id = coupling_id
        self.tail_envelope = tail_envelope
        self.family_id = canonical_fingerprint(
            {
                "kind": "finite-spde-approximation-family-v1",
                "levels": tuple(item.level_id for item in selected),
                "refined_axis": refined_axis,
                "reference_level": reference,
                "coupling_id": coupling_id,
                "tail_envelope": tail_envelope is not None,
            }
        )


class PreparedSPDEApproximation(StrictModule):
    family: SPDEApproximationFamily
    ensembles: tuple[PreparedStochasticPathEnsemble, ...]
    prepared_id: str = eqx.field(static=True)


class SPDEApproximationResult(StrictModule):
    """Coupled finite solutions with decomposed empirical/tail evidence."""

    solutions: tuple[StochasticPathEnsembleResult, ...]
    transferred_states: tuple[Array, ...]
    cauchy_differences: Array
    tail_bounds: Array | None
    valid: Array
    status: Array
    family_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    approximation_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def prepare_spde_approximation(
    family: SPDEApproximationFamily,
    /,
    *,
    ensemble_plan: StochasticPathEnsemblePlan,
    key: Key[Array, ""],
) -> PreparedSPDEApproximation:
    """Prepare coupled finite levels using common prefix-stable Wiener keys."""
    if not isinstance(family, SPDEApproximationFamily):
        raise TypeError("family must be an SPDEApproximationFamily.")
    if not isinstance(ensemble_plan, StochasticPathEnsemblePlan):
        raise TypeError("ensemble_plan must be a StochasticPathEnsemblePlan.")
    ensembles = []
    for level in family.levels:
        realization = level.spde.wiener_realization(
            key,
            sample_shape=(ensemble_plan.path_count,),
            tolerance=ensemble_plan.wiener_tolerance,
            levy_area=ensemble_plan.levy_area,
            label=f"spde-level:{level.level_id}",
            coupling_id=family.coupling_id,
        )
        ensembles.append(
            prepare_stochastic_path_ensemble(
                level.spde.problem,
                ensemble_plan,
                realization=realization,
            )
        )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-finite-spde-family-v1",
            "family": family.family_id,
            "ensemble_plan": ensemble_plan.plan_id,
            "realizations": tuple(item.realization.realization_id for item in ensembles),
        }
    )
    return PreparedSPDEApproximation(
        family=family,
        ensembles=tuple(ensembles),
        prepared_id=prepared_id,
    )


def solve_spde_approximation(
    prepared: PreparedSPDEApproximation,
    /,
    *,
    observables: tuple[Any, ...] = (),
) -> SPDEApproximationResult:
    """Execute all finite levels and expose Cauchy and optional tail evidence."""
    del observables
    if not isinstance(prepared, PreparedSPDEApproximation):
        raise TypeError("prepared must be a PreparedSPDEApproximation.")
    solutions = tuple(solve_stochastic_path_ensemble(item) for item in prepared.ensembles)
    transferred = tuple(
        level.transfer_to_reference(result.states)
        for level, result in zip(prepared.family.levels, solutions, strict=True)
    )
    differences = []
    for left, right in zip(transferred[:-1], transferred[1:], strict=True):
        if left.shape != right.shape:
            raise ValueError("Transferred SPDE levels must share one reference shape.")
        numerator = jnp.sqrt(jnp.sum(jnp.abs(right - left) ** 2))
        denominator = 1.0 + jnp.sqrt(jnp.sum(jnp.abs(right) ** 2))
        differences.append(numerator / denominator)
    cauchy = jnp.stack(differences) if differences else jnp.zeros((0,))
    envelope = prepared.family.tail_envelope
    if envelope is None:
        tails = None
    elif callable(envelope):
        tails = jnp.stack(
            tuple(jnp.asarray(envelope(level)) for level in prepared.family.levels)
        )
    else:
        tails = jnp.asarray(envelope)
        if tails.shape != (len(prepared.family.levels),):
            raise ValueError("tail_envelope values must align with levels.")
    valid = jnp.asarray(tuple(result.valid for result in solutions), dtype=bool)
    if tails is not None:
        valid = valid & jnp.all(jnp.isfinite(tails) & (tails >= 0.0))
    status = jnp.where(valid, 0, 1).astype(jnp.int32)
    return SPDEApproximationResult(
        solutions=solutions,
        transferred_states=transferred,
        cauchy_differences=cauchy,
        tail_bounds=tails,
        valid=valid,
        status=status,
        family_id=prepared.family.family_id,
        prepared_id=prepared.prepared_id,
        approximation_kind=f"finite-{prepared.family.refined_axis}-refinement",
        bounded_non_claim=(
            "Empirical Cauchy differences are not continuum tail bounds; only an "
            "explicit caller-supplied tail envelope is reported as such."
        ),
    )


__all__ = [
    "PreparedSPDEApproximation",
    "SPDEApproximationFamily",
    "SPDEApproximationLevel",
    "SPDEApproximationResult",
    "prepare_spde_approximation",
    "solve_spde_approximation",
]

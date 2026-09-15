#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact active-space solver boundary for selected-CI, DMRG, and stochastic providers."""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import AbstractAttribute, StrictModule
from ...._trainable import NonTrainableState
from ._ci import CASCIPlan
from ._orbital import MolecularOrbitalIntegralStore


class ActiveSpaceSolverResult(StrictModule, NonTrainableState):
    energies: Array
    one_particle_density: Array
    two_particle_density: Array
    residuals: Array
    discarded_weights: Array
    successful: Array
    solver_kind: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies: ArrayLike,
        one_particle_density: ArrayLike,
        two_particle_density: ArrayLike,
        residuals: ArrayLike,
        discarded_weights: ArrayLike,
        successful: ArrayLike,
        solver_kind: str,
        provider_id: str,
        plan_id: str,
        store_id: str,
        /,
    ):
        energy = jnp.asarray(energies)
        one = jnp.asarray(one_particle_density, dtype=energy.dtype)
        two = jnp.asarray(two_particle_density, dtype=energy.dtype)
        residual = jnp.asarray(residuals, dtype=energy.real.dtype)
        discarded = jnp.asarray(discarded_weights, dtype=energy.real.dtype)
        kind = str(solver_kind).strip()
        provider = str(provider_id).strip()
        if (
            energy.ndim != 1
            or one.ndim != 3
            or two.ndim != 5
            or one.shape[0] != energy.size
            or two.shape[0] != energy.size
        ):
            raise ValueError(
                "Active-space energies and density matrices must align by root."
            )
        if (
            residual.shape != energy.shape
            or discarded.shape != energy.shape
            or not kind
            or not provider
        ):
            raise ValueError(
                "Active-space residual, truncation, or identities are invalid."
            )
        self.energies = energy
        self.one_particle_density = one
        self.two_particle_density = two
        self.residuals = residual
        self.discarded_weights = discarded
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.solver_kind = kind
        self.provider_id = provider
        self.plan_id = str(plan_id)
        self.store_id = str(store_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "active-space-solver-result",
                "solver_kind": kind,
                "provider": provider,
                "plan": self.plan_id,
                "store": self.store_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(energy),
                        "one_particle_density": np.asarray(one),
                        "two_particle_density": np.asarray(two),
                        "residuals": np.asarray(residual),
                        "discarded_weights": np.asarray(discarded),
                    }
                ),
            }
        )


class AbstractActiveSpaceSolver(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]
    solver_kind: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        plan: CASCIPlan,
        integrals: MolecularOrbitalIntegralStore,
        /,
    ) -> ActiveSpaceSolverResult:
        raise NotImplementedError


ActiveSpaceEvaluator = Callable[
    [CASCIPlan, MolecularOrbitalIntegralStore], ActiveSpaceSolverResult
]


class CallableActiveSpaceSolver(AbstractActiveSpaceSolver):
    evaluator: ActiveSpaceEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    solver_kind: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: ActiveSpaceEvaluator,
        provider_id: str,
        solver_kind: str,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        kind = str(solver_kind).strip()
        if not provider or kind not in ("selected-ci", "dmrg", "fciqmc"):
            raise ValueError("Active-space provider or solver kind is invalid.")
        self.evaluator = evaluator
        self.provider_id = provider
        self.solver_kind = kind

    def evaluate(
        self,
        plan: CASCIPlan,
        integrals: MolecularOrbitalIntegralStore,
        /,
    ) -> ActiveSpaceSolverResult:
        result = self.evaluator(plan, integrals)
        if not isinstance(result, ActiveSpaceSolverResult):
            raise TypeError("Active-space evaluator returned the wrong result type.")
        if (
            result.provider_id != self.provider_id
            or result.solver_kind != self.solver_kind
            or result.plan_id != plan.plan_id
            or result.store_id != integrals.store_id
        ):
            raise ValueError("Active-space provider changed bound identities.")
        return result


__all__ = [
    "AbstractActiveSpaceSolver",
    "ActiveSpaceSolverResult",
    "CallableActiveSpaceSolver",
]

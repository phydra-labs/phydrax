#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuum-solvent and relativistic one-electron model boundaries."""

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import ImplicitSolventPlan
from ...ein import contract


class ContinuumSolvationKind(StrEnum):
    GENERALIZED_BORN = "generalized-born"
    GENERALIZED_KIRKWOOD = "generalized-kirkwood"
    PCM = "pcm"
    COSMO = "cosmo"


class ContinuumSolvationResult(StrictModule):
    energy: Array
    forces: Array
    surface_charges: Array | None
    residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ContinuumSolvationPlan(StrictModule, NonTrainableState):
    kind: ContinuumSolvationKind = eqx.field(static=True)
    solvent_dielectric: float = eqx.field(static=True)
    solute_dielectric: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    provider_definition_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ContinuumSolvationKind,
        /,
        *,
        solvent_dielectric: float = 78.5,
        solute_dielectric: float = 1.0,
        surface_tension: float = 0.005,
        provider_definition_id: str | None = None,
    ):
        if not isinstance(kind, ContinuumSolvationKind):
            raise TypeError("kind must be ContinuumSolvationKind.")
        values = (
            float(solvent_dielectric),
            float(solute_dielectric),
            float(surface_tension),
        )
        provider = (
            None
            if provider_definition_id is None
            else str(provider_definition_id).strip()
        )
        if (
            not np.all(np.isfinite(values))
            or min(values[:2]) <= 0.0
            or values[2] < 0.0
            or provider_definition_id is not None
            and not provider
        ):
            raise ValueError("Continuum-solvation parameters are invalid.")
        if (
            kind in (ContinuumSolvationKind.PCM, ContinuumSolvationKind.COSMO)
            and not provider
        ):
            raise ValueError("PCM/COSMO require an exact surface provider definition.")
        self.kind = kind
        self.solvent_dielectric = values[0]
        self.solute_dielectric = values[1]
        self.surface_tension = values[2]
        self.provider_definition_id = provider
        self.plan_id = canonical_fingerprint(
            {
                "kind": "continuum-solvation-plan",
                "model": kind.value,
                "solvent_dielectric": values[0],
                "solute_dielectric": values[1],
                "surface_tension": values[2],
                "provider_definition": provider,
            }
        )

    def evaluate_fixed_charges(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        radii: ArrayLike,
        coulomb_constant: float,
        /,
    ) -> ContinuumSolvationResult:
        if self.kind not in (
            ContinuumSolvationKind.GENERALIZED_BORN,
            ContinuumSolvationKind.GENERALIZED_KIRKWOOD,
        ):
            raise ValueError("PCM/COSMO evaluation requires its declared provider.")
        model = "gb" if self.kind is ContinuumSolvationKind.GENERALIZED_BORN else "gk"
        substrate = ImplicitSolventPlan(
            model,
            solvent_dielectric=self.solvent_dielectric,
            solute_dielectric=self.solute_dielectric,
            surface_tension=self.surface_tension,
        )
        coordinate = jnp.asarray(positions)
        charges_ = jnp.asarray(charges, dtype=coordinate.dtype)
        radii_ = jnp.asarray(radii, dtype=coordinate.dtype)

        def energy(value):
            return substrate.energy(value, charges_, radii_, float(coulomb_constant))

        energy_, gradient = jax.value_and_grad(energy)(coordinate)
        successful = jnp.isfinite(energy_) & jnp.all(jnp.isfinite(gradient))
        return ContinuumSolvationResult(
            energy_,
            -gradient,
            None,
            jnp.asarray(0.0, dtype=coordinate.dtype),
            successful,
            self.plan_id,
        )


ContinuumEvaluator = Callable[[ArrayLike, ArrayLike], ContinuumSolvationResult]


class AbstractContinuumSolventProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    plan_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(
        self, positions: ArrayLike, electronic_density: ArrayLike, /
    ) -> ContinuumSolvationResult:
        raise NotImplementedError


class CallableContinuumSolventProvider(AbstractContinuumSolventProvider):
    evaluator: ContinuumEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: ContinuumEvaluator,
        provider_id: str,
        plan_id: str,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        plan = str(plan_id).strip()
        if not provider or not plan:
            raise ValueError("Continuum provider and plan identities must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider
        self.plan_id = plan

    def evaluate(
        self, positions: ArrayLike, electronic_density: ArrayLike, /
    ) -> ContinuumSolvationResult:
        result = self.evaluator(positions, electronic_density)
        if (
            not isinstance(result, ContinuumSolvationResult)
            or result.plan_id != self.plan_id
        ):
            raise ValueError("Continuum provider changed result type or plan identity.")
        return result


class RelativisticHamiltonianKind(StrEnum):
    SCALAR_X2C = "scalar-x2c"
    SPIN_FREE_DKH = "spin-free-dkh"
    TWO_COMPONENT_X2C = "two-component-x2c"
    FOUR_COMPONENT_NO_PAIR = "four-component-no-pair"


class RelativisticOneElectronResult(StrictModule):
    hamiltonian: Array
    metric: Array
    metric_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class RelativisticOneElectronPlan(StrictModule, NonTrainableState):
    kind: RelativisticHamiltonianKind = eqx.field(static=True)
    order: int = eqx.field(static=True)
    positive_energy_projector_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: RelativisticHamiltonianKind,
        /,
        *,
        order: int = 2,
        positive_energy_projector_id: str | None = None,
    ):
        if not isinstance(kind, RelativisticHamiltonianKind):
            raise TypeError("kind must be RelativisticHamiltonianKind.")
        order_ = int(order)
        projector = (
            None
            if positive_energy_projector_id is None
            else str(positive_energy_projector_id).strip()
        )
        if order_ < 1 or order_ > 8:
            raise ValueError("Relativistic transformation order must lie in [1, 8].")
        if kind is RelativisticHamiltonianKind.FOUR_COMPONENT_NO_PAIR and not projector:
            raise ValueError(
                "Four-component no-pair plans require a positive-energy projector."
            )
        self.kind = kind
        self.order = order_
        self.positive_energy_projector_id = projector
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-one-electron-plan",
                "hamiltonian": kind.value,
                "order": order_,
                "positive_energy_projector": projector,
            }
        )

    def transform(
        self,
        one_body: ArrayLike,
        overlap: ArrayLike,
        decoupling: ArrayLike,
        /,
        *,
        metric_tolerance: float = 1.0e-8,
    ) -> RelativisticOneElectronResult:
        hamiltonian = jnp.asarray(one_body)
        metric = jnp.asarray(overlap, dtype=hamiltonian.dtype)
        transform = jnp.asarray(decoupling, dtype=hamiltonian.dtype)
        if (
            hamiltonian.ndim != 2
            or hamiltonian.shape[0] != hamiltonian.shape[1]
            or metric.shape != hamiltonian.shape
            or transform.ndim != 2
            or transform.shape[0] != hamiltonian.shape[0]
        ):
            raise ValueError("Relativistic one-electron arrays have incompatible shapes.")
        transformed_metric = contract(
            "pi,pq,qj->ij", jnp.conj(transform), metric, transform
        )
        transformed_hamiltonian = contract(
            "pi,pq,qj->ij", jnp.conj(transform), hamiltonian, transform
        )
        residual = jnp.max(
            jnp.abs(
                transformed_metric
                - jnp.eye(transformed_metric.shape[0], dtype=metric.dtype)
            ),
            initial=0.0,
        )
        successful = (
            jnp.all(jnp.isfinite(transformed_hamiltonian))
            & jnp.all(jnp.isfinite(transformed_metric))
            & (residual <= float(metric_tolerance))
        )
        return RelativisticOneElectronResult(
            transformed_hamiltonian,
            transformed_metric,
            residual,
            successful,
            self.plan_id,
        )


__all__ = [
    "AbstractContinuumSolventProvider",
    "CallableContinuumSolventProvider",
    "ContinuumSolvationKind",
    "ContinuumSolvationPlan",
    "ContinuumSolvationResult",
    "RelativisticHamiltonianKind",
    "RelativisticOneElectronPlan",
    "RelativisticOneElectronResult",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coupled-cluster task, result, and exact provider boundary."""

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._orbital import MolecularOrbitalIntegralStore


class CoupledClusterCheckpoint(StrictModule, NonTrainableState):
    singles_amplitudes: Array
    doubles_amplitudes: Array
    lambda_singles: Array | None
    lambda_doubles: Array | None
    completed_iterations: int = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        singles_amplitudes,
        doubles_amplitudes,
        completed_iterations: int,
        provider_id: str,
        plan_id: str,
        store_id: str,
        /,
        *,
        lambda_singles=None,
        lambda_doubles=None,
    ):
        singles = jnp.asarray(singles_amplitudes)
        doubles = jnp.asarray(doubles_amplitudes, dtype=singles.dtype)
        lambda_s = (
            None
            if lambda_singles is None
            else jnp.asarray(lambda_singles, dtype=singles.dtype)
        )
        lambda_d = (
            None
            if lambda_doubles is None
            else jnp.asarray(lambda_doubles, dtype=singles.dtype)
        )
        provider = str(provider_id).strip()
        plan = str(plan_id).strip()
        store = str(store_id).strip()
        iterations = int(completed_iterations)
        if (
            singles.ndim != 2
            or doubles.ndim != 4
            or (lambda_s is None) != (lambda_d is None)
            or iterations < 0
            or not provider
            or not plan
            or not store
        ):
            raise ValueError("Coupled-cluster checkpoint contents are invalid.")
        if lambda_s is not None:
            if lambda_d is None:
                raise RuntimeError("Lambda amplitude presence invariant failed.")
            if lambda_s.shape != singles.shape or lambda_d.shape != doubles.shape:
                raise ValueError("Checkpoint Lambda amplitudes do not align.")
        self.singles_amplitudes = singles
        self.doubles_amplitudes = doubles
        self.lambda_singles = lambda_s
        self.lambda_doubles = lambda_d
        self.completed_iterations = iterations
        self.provider_id = provider
        self.plan_id = plan
        self.store_id = store
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "coupled-cluster-checkpoint",
                "provider": provider,
                "plan": plan,
                "store": store,
                "completed_iterations": iterations,
                "arrays": array_tree_fingerprint(
                    {
                        "singles": np.asarray(singles),
                        "doubles": np.asarray(doubles),
                        "lambda_singles": None
                        if lambda_s is None
                        else np.asarray(lambda_s),
                        "lambda_doubles": None
                        if lambda_d is None
                        else np.asarray(lambda_d),
                    }
                ),
            }
        )


class CoupledClusterResult(StrictModule, NonTrainableState):
    correlation_energy: Array
    triples_correction: Array
    total_energy: Array
    singles_amplitudes: Array
    doubles_amplitudes: Array
    lambda_singles: Array | None
    lambda_doubles: Array | None
    amplitude_residual: Array
    lambda_residual: Array
    iterations: Array
    successful: Array
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        correlation_energy,
        triples_correction,
        total_energy,
        singles_amplitudes,
        doubles_amplitudes,
        amplitude_residual,
        lambda_residual,
        iterations: int,
        successful,
        provider_id: str,
        plan_id: str,
        store_id: str,
        /,
        *,
        lambda_singles=None,
        lambda_doubles=None,
    ):
        singles = jnp.asarray(singles_amplitudes)
        doubles = jnp.asarray(doubles_amplitudes, dtype=singles.dtype)
        dtype = singles.real.dtype
        provider = str(provider_id).strip()
        if singles.ndim != 2 or doubles.ndim != 4 or not provider:
            raise ValueError(
                "Coupled-cluster amplitudes or provider identity is invalid."
            )
        lambda_s = (
            None
            if lambda_singles is None
            else jnp.asarray(lambda_singles, dtype=singles.dtype)
        )
        lambda_d = (
            None
            if lambda_doubles is None
            else jnp.asarray(lambda_doubles, dtype=singles.dtype)
        )
        if (lambda_s is None) != (lambda_d is None):
            raise ValueError(
                "Coupled-cluster lambda amplitudes must be supplied together."
            )
        if lambda_s is not None:
            if lambda_d is None:
                raise RuntimeError("Lambda amplitude presence invariant failed.")
            if lambda_s.shape != singles.shape or lambda_d.shape != doubles.shape:
                raise ValueError("Lambda amplitudes must align with right amplitudes.")
        self.correlation_energy = jnp.asarray(correlation_energy, dtype=dtype).reshape(())
        self.triples_correction = jnp.asarray(triples_correction, dtype=dtype).reshape(())
        self.total_energy = jnp.asarray(total_energy, dtype=dtype).reshape(())
        self.singles_amplitudes = singles
        self.doubles_amplitudes = doubles
        self.lambda_singles = lambda_s
        self.lambda_doubles = lambda_d
        self.amplitude_residual = jnp.asarray(amplitude_residual, dtype=dtype).reshape(())
        self.lambda_residual = jnp.asarray(lambda_residual, dtype=dtype).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.provider_id = provider
        self.plan_id = str(plan_id)
        self.store_id = str(store_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "coupled-cluster-result",
                "provider": provider,
                "plan": self.plan_id,
                "store": self.store_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "correlation_energy": np.asarray(self.correlation_energy),
                        "triples_correction": np.asarray(self.triples_correction),
                        "total_energy": np.asarray(self.total_energy),
                        "singles": np.asarray(singles),
                        "doubles": np.asarray(doubles),
                        "lambda_singles": None
                        if lambda_s is None
                        else np.asarray(lambda_s),
                        "lambda_doubles": None
                        if lambda_d is None
                        else np.asarray(lambda_d),
                        "amplitude_residual": np.asarray(self.amplitude_residual),
                        "lambda_residual": np.asarray(self.lambda_residual),
                        "iterations": np.asarray(self.iterations),
                    }
                ),
            }
        )

    def checkpoint(self, /) -> CoupledClusterCheckpoint:
        return CoupledClusterCheckpoint(
            self.singles_amplitudes,
            self.doubles_amplitudes,
            int(self.iterations),
            self.provider_id,
            self.plan_id,
            self.store_id,
            lambda_singles=self.lambda_singles,
            lambda_doubles=self.lambda_doubles,
        )


class CoupledClusterPlan(StrictModule, NonTrainableState):
    level: str = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    solve_lambda: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: str = "ccsd",
        /,
        *,
        convergence_tolerance: float = 1.0e-8,
        maximum_iterations: int = 100,
        solve_lambda: bool = True,
    ):
        level_ = str(level).strip().lower()
        tolerance = float(convergence_tolerance)
        iterations = int(maximum_iterations)
        if (
            level_ not in ("ccsd", "ccsd(t)")
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or iterations <= 0
        ):
            raise ValueError(
                "Coupled-cluster level, tolerance, or iteration limit is invalid."
            )
        self.level = level_
        self.convergence_tolerance = tolerance
        self.maximum_iterations = iterations
        self.solve_lambda = bool(solve_lambda)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-cluster-plan",
                "level": level_,
                "convergence_tolerance": tolerance,
                "maximum_iterations": iterations,
                "solve_lambda": self.solve_lambda,
            }
        )


class AbstractCoupledClusterProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(
        self,
        plan: CoupledClusterPlan,
        integrals: MolecularOrbitalIntegralStore,
        /,
        *,
        checkpoint: CoupledClusterCheckpoint | None = None,
    ) -> CoupledClusterResult:
        raise NotImplementedError


CoupledClusterEvaluator = Callable[
    [
        CoupledClusterPlan,
        MolecularOrbitalIntegralStore,
        CoupledClusterCheckpoint | None,
    ],
    CoupledClusterResult,
]


class CallableCoupledClusterProvider(AbstractCoupledClusterProvider):
    evaluator: CoupledClusterEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: CoupledClusterEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(
        self,
        plan: CoupledClusterPlan,
        integrals: MolecularOrbitalIntegralStore,
        /,
        *,
        checkpoint: CoupledClusterCheckpoint | None = None,
    ) -> CoupledClusterResult:
        if checkpoint is not None and (
            checkpoint.provider_id != self.provider_id
            or checkpoint.plan_id != plan.plan_id
            or checkpoint.store_id != integrals.store_id
        ):
            raise ValueError("Coupled-cluster checkpoint belongs to another calculation.")
        result = self.evaluator(plan, integrals, checkpoint)
        if not isinstance(result, CoupledClusterResult):
            raise TypeError("Coupled-cluster evaluator returned the wrong result type.")
        if (
            result.provider_id != self.provider_id
            or result.plan_id != plan.plan_id
            or result.store_id != integrals.store_id
        ):
            raise ValueError("Coupled-cluster provider changed bound identities.")
        return result


class MolecularCoupledClusterGradientResult(StrictModule, NonTrainableState):
    reference_energy: Array
    correlation_energy: Array
    triples_correction: Array
    total_energy: Array
    gradient: Array
    iterations: Array
    scf_converged: Array
    amplitudes_converged: Array
    lambda_converged: Array
    successful: Array
    reference: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_energy,
        correlation_energy,
        triples_correction,
        gradient,
        iterations,
        scf_converged,
        amplitudes_converged,
        lambda_converged,
        reference: str,
        provider_id: str,
        plan_id: str,
        /,
    ):
        gradient_ = jnp.asarray(gradient)
        dtype = gradient_.real.dtype
        reference_ = str(reference).strip().lower()
        provider = str(provider_id).strip()
        plan = str(plan_id).strip()
        if (
            gradient_.ndim != 2
            or gradient_.shape[1] != 3
            or reference_ not in ("rhf", "uhf", "rohf")
            or not provider
            or not plan
        ):
            raise ValueError("Molecular coupled-cluster gradient data are invalid.")
        self.reference_energy = jnp.asarray(reference_energy, dtype=dtype).reshape(())
        self.correlation_energy = jnp.asarray(correlation_energy, dtype=dtype).reshape(())
        self.triples_correction = jnp.asarray(triples_correction, dtype=dtype).reshape(())
        self.total_energy = (
            self.reference_energy + self.correlation_energy + self.triples_correction
        )
        self.gradient = gradient_
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.scf_converged = jnp.asarray(scf_converged, dtype=jnp.bool_).reshape(())
        self.amplitudes_converged = jnp.asarray(
            amplitudes_converged, dtype=jnp.bool_
        ).reshape(())
        self.lambda_converged = jnp.asarray(lambda_converged, dtype=jnp.bool_).reshape(())
        self.successful = (
            self.scf_converged
            & self.amplitudes_converged
            & self.lambda_converged
            & jnp.isfinite(self.total_energy)
            & jnp.all(jnp.isfinite(gradient_))
        )
        self.reference = reference_
        self.provider_id = provider
        self.plan_id = plan
        self.result_id = canonical_fingerprint(
            {
                "kind": "molecular-coupled-cluster-gradient-result",
                "reference": reference_,
                "provider": provider,
                "plan": plan,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "reference_energy": np.asarray(self.reference_energy),
                        "correlation_energy": np.asarray(self.correlation_energy),
                        "triples_correction": np.asarray(self.triples_correction),
                        "gradient": np.asarray(gradient_),
                    }
                ),
            }
        )


class AbstractMolecularCoupledClusterGradientProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    plan_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(self, positions: ArrayLike, /) -> MolecularCoupledClusterGradientResult:
        raise NotImplementedError


__all__ = [
    "AbstractCoupledClusterProvider",
    "AbstractMolecularCoupledClusterGradientProvider",
    "CallableCoupledClusterProvider",
    "CoupledClusterCheckpoint",
    "MolecularCoupledClusterGradientResult",
    "CoupledClusterPlan",
    "CoupledClusterResult",
]

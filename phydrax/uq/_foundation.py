#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._frozendict import frozendict
from .._probability import AbstractProbabilityLaw
from .._strict import StrictModule
from ._mcmc import MCMCResult
from ._posterior import PosteriorProblem


UQProduct = Literal["forward", "calibration", "robust"]
UncertainVariableRole = Literal["aleatoric", "epistemic", "calibration", "nuisance"]


class UncertainVariable(StrictModule):
    """Stable uncertain-variable identity, law, and semantic role."""

    variable_id: str = eqx.field(static=True)
    law: AbstractProbabilityLaw
    role: UncertainVariableRole = eqx.field(static=True)
    unit: str | None = eqx.field(static=True)

    def __init__(
        self,
        variable_id: str,
        law: AbstractProbabilityLaw,
        /,
        *,
        role: UncertainVariableRole = "aleatoric",
        unit: str | None = None,
    ):
        identifier = _identifier(variable_id, "variable_id")
        if not isinstance(law, AbstractProbabilityLaw):
            raise TypeError("law must implement AbstractProbabilityLaw.")
        if role not in ("aleatoric", "epistemic", "calibration", "nuisance"):
            raise ValueError("Unknown uncertain-variable role.")
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise ValueError("unit must be a non-empty string or None.")
        self.variable_id = identifier
        self.law = law
        self.role = role
        self.unit = unit


class Experiment(StrictModule):
    """Immutable experiment data and conditions bound to one likelihood identity."""

    experiment_id: str = eqx.field(static=True)
    observations: Any
    conditions: frozendict[str, Array]
    likelihood_id: str = eqx.field(static=True)
    diagnostic_ids: tuple[str, ...] = eqx.field(static=True)
    data_digest: str = eqx.field(static=True)

    def __init__(
        self,
        experiment_id: str,
        observations: Any,
        /,
        *,
        conditions: dict[str, Any] | frozendict[str, Any] | None = None,
        likelihood_id: str,
        diagnostic_ids: tuple[str, ...] = (),
    ):
        identifier = _identifier(experiment_id, "experiment_id")
        likelihood = _identifier(likelihood_id, "likelihood_id")
        observation_leaves = jax.tree_util.tree_leaves(observations)
        if not observation_leaves:
            raise ValueError("Experiment observations must contain array leaves.")
        observations_ = jax.tree_util.tree_map(jnp.asarray, observations)
        if any(
            bool(jnp.any(~jnp.isfinite(jnp.asarray(value))))
            for value in jax.tree_util.tree_leaves(observations_)
        ):
            raise ValueError("Experiment observations must be finite.")
        condition_values = {} if conditions is None else dict(conditions)
        if any(not isinstance(name, str) or not name for name in condition_values):
            raise ValueError("Experiment condition names must be non-empty strings.")
        conditions_ = frozendict(
            {name: jnp.asarray(value) for name, value in condition_values.items()}
        )
        diagnostics = _identifiers(diagnostic_ids, "diagnostic_ids")
        fingerprint = array_tree_fingerprint(
            {"observations": observations_, "conditions": conditions_}
        )
        self.experiment_id = identifier
        self.observations = observations_
        self.conditions = conditions_
        self.likelihood_id = likelihood
        self.diagnostic_ids = diagnostics
        self.data_digest = fingerprint["sha256"]


@dataclass(frozen=True, slots=True)
class UQPlan:
    """Replay-stable product plan layered on common analysis lifecycle IDs."""

    product: UQProduct
    method: str
    analysis_plan_id: str
    numeric_revision_id: str
    execution_plan_id: str
    num_samples: int
    root_seed: int
    profile: str | None = None
    num_replicates: int = 1
    checkpoint_every: int | None = None
    plan_id: str = field(init=False)

    def __post_init__(self):
        if self.product not in ("forward", "calibration", "robust"):
            raise ValueError("Unknown UQ product.")
        method = _identifier(self.method, "method")
        analysis = _identifier(self.analysis_plan_id, "analysis_plan_id")
        revision = _identifier(self.numeric_revision_id, "numeric_revision_id")
        execution = _identifier(self.execution_plan_id, "execution_plan_id")
        count = int(self.num_samples)
        seed = int(self.root_seed)
        replicas = int(self.num_replicates)
        if count < 1:
            raise ValueError("num_samples must be positive.")
        if seed < 0:
            raise ValueError("root_seed must be nonnegative.")
        if replicas < 1:
            raise ValueError("num_replicates must be positive.")
        profile = self.profile
        if profile is not None and (not isinstance(profile, str) or not profile):
            raise ValueError("profile must be a non-empty string or None.")
        if self.product == "calibration" and profile is not None:
            raise ValueError("Calibration plans do not use a forward sampling profile.")
        interval = self.checkpoint_every
        if interval is not None:
            interval = int(interval)
            if interval < 1:
                raise ValueError("checkpoint_every must be positive or None.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "analysis_plan_id", analysis)
        object.__setattr__(self, "numeric_revision_id", revision)
        object.__setattr__(self, "execution_plan_id", execution)
        object.__setattr__(self, "num_samples", count)
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(self, "num_replicates", replicas)
        object.__setattr__(self, "checkpoint_every", interval)
        payload = {
            "kind": "uq-plan",
            "product": self.product,
            "method": method,
            "analysis_plan_id": analysis,
            "numeric_revision_id": revision,
            "execution_plan_id": execution,
            "num_samples": count,
            "root_seed": seed,
            "profile": profile,
            "num_replicates": replicas,
            "checkpoint_every": interval,
        }
        object.__setattr__(self, "plan_id", canonical_fingerprint(payload))


class PosteriorRecord(StrictModule):
    """Lifecycle binding over the authoritative HMC/NUTS posterior result."""

    plan: UQPlan = eqx.field(static=True)
    result: MCMCResult
    checkpoint_id: str | None = eqx.field(static=True)
    diagnostic_ids: tuple[str, ...] = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UQPlan,
        result: MCMCResult,
        /,
        *,
        checkpoint_id: str | None = None,
        diagnostic_ids: tuple[str, ...] = (),
    ):
        if not isinstance(plan, UQPlan) or plan.product != "calibration":
            raise TypeError("plan must be a calibration UQPlan.")
        if not isinstance(result, MCMCResult):
            raise TypeError("result must be an MCMCResult.")
        if result.algorithm not in ("hmc", "nuts"):
            raise ValueError(
                "PosteriorRecord supports authoritative HMC or NUTS results."
            )
        if plan.method != result.algorithm:
            raise ValueError("UQ plan method and MCMC algorithm do not match.")
        if plan.num_samples != result.num_draws:
            raise ValueError("UQ plan sample count and MCMC draw count do not match.")
        checkpoint = (
            None if checkpoint_id is None else _identifier(checkpoint_id, "checkpoint_id")
        )
        diagnostics = _identifiers(diagnostic_ids, "diagnostic_ids")
        key_digest = array_tree_fingerprint(
            {"root_key": result.root_key, "chain_keys": result.chain_keys}
        )["sha256"]
        record_id = canonical_fingerprint(
            {
                "kind": "posterior-record",
                "plan_id": plan.plan_id,
                "algorithm": result.algorithm,
                "num_chains": result.num_chains,
                "num_draws": result.num_draws,
                "sample_lineage_digest": key_digest,
                "checkpoint_id": checkpoint,
                "diagnostic_ids": list(diagnostics),
            }
        )
        self.plan = plan
        self.result = result
        self.checkpoint_id = checkpoint
        self.diagnostic_ids = diagnostics
        self.record_id = record_id

    @property
    def problem(self) -> PosteriorProblem:
        return self.result.problem

    @property
    def diagnostics(self):
        return self.result.diagnostics

    @property
    def samples(self):
        return self.result.samples

    def predict(self, *args: Any, **kwargs: Any):
        """Delegate prediction to the authoritative chain-preserving result."""
        return self.result.predict(*args, **kwargs)

    def predict_observations(self, key: Array, /, *args: Any, **kwargs: Any):
        """Delegate observation prediction without changing sample axes."""
        return self.result.predict_observations(key, *args, **kwargs)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _identifiers(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    identifiers = tuple(values)
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must not contain duplicates.")
    return identifiers


__all__ = [
    "Experiment",
    "PosteriorRecord",
    "UQPlan",
    "UQProduct",
    "UncertainVariable",
    "UncertainVariableRole",
]

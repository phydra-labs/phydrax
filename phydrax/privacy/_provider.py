#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from .._fingerprint import canonical_fingerprint
from ._accounting import (
    account_mechanism_trace,
    AccountingMethod,
    MechanismTrace,
    PrivacyGuarantee,
)
from ._definition import RandomnessAssurance
from ._release import certify_private_release, PrivacyCertificate
from ._training import PrivateTrainingPlan


def _require_module(name: str, extra: str, /) -> Any:
    if importlib.util.find_spec(name) is None:
        raise ImportError(f"{name} requires phydrax[{extra}].")
    return importlib.import_module(name)


def _upstream_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    dp = _require_module("dp_accounting", "privacy-jax")
    batch_selection = _require_module("jax_privacy.batch_selection", "privacy-jax")
    clipping = _require_module("jax_privacy.clipping", "privacy-jax")
    noise_addition = _require_module("jax_privacy.noise_addition", "privacy-jax")
    accounting = _require_module("jax_privacy.experimental.accounting", "privacy-jax")
    execution_plan = _require_module(
        "jax_privacy.experimental.execution_plan", "privacy-jax"
    )
    return dp, batch_selection, clipping, noise_addition, accounting, execution_plan


def _fresh_accountant(method: AccountingMethod, dp: Any, /) -> object:
    neighboring = dp.NeighboringRelation.ADD_OR_REMOVE_ONE
    if method is AccountingMethod.PLD:
        return dp.pld.PLDAccountant(neighboring)
    if method is AccountingMethod.RDP:
        return dp.rdp.RdpAccountant(neighboring_relation=neighboring)
    raise ValueError(f"Unsupported accounting method {method!r}.")


@dataclass(frozen=True, slots=True)
class _PreparedPrivateGradient:
    """One inseparable sampler, clipping query, noiser, and accountant event."""

    training_plan: PrivateTrainingPlan
    noise_multiplier: float
    per_step_trace: MechanismTrace
    planned_guarantee: PrivacyGuarantee
    upstream_plan: Any = field(repr=False, compare=False)
    sampler_seed: int = field(repr=False, compare=False)
    prepared_id: str = field(init=False)
    provider_id: str = "jax-privacy"
    provider_version: str = "2.0.0"
    mechanism_id: str = "poisson-gaussian-dp-sgd"
    randomness: RandomnessAssurance = RandomnessAssurance.RESEARCH_PRNG

    def __post_init__(self) -> None:
        if not isinstance(self.training_plan, PrivateTrainingPlan):
            raise TypeError("training_plan must be a PrivateTrainingPlan.")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be finite and positive.")
        if not isinstance(self.per_step_trace, MechanismTrace):
            raise TypeError("per_step_trace must be a MechanismTrace.")
        if not isinstance(self.planned_guarantee, PrivacyGuarantee):
            raise TypeError("planned_guarantee must be a PrivacyGuarantee.")
        if int(self.sampler_seed) < 0:
            raise ValueError("sampler_seed must be non-negative.")
        object.__setattr__(self, "sampler_seed", int(self.sampler_seed))
        object.__setattr__(
            self,
            "prepared_id",
            canonical_fingerprint(
                {
                    "kind": "prepared-private-gradient",
                    "training_plan_id": self.training_plan.plan_id,
                    "noise_multiplier": self.noise_multiplier,
                    "per_step_trace_id": self.per_step_trace.trace_id,
                    "planned_guarantee_id": self.planned_guarantee.guarantee_id,
                    "provider_id": self.provider_id,
                    "provider_version": self.provider_version,
                    "mechanism_id": self.mechanism_id,
                    "randomness": self.randomness.value,
                }
            ),
        )

    def batch_iterator(self, num_examples: int, /, *, start_step: int = 0) -> Any:
        num_examples = int(num_examples)
        start_step = int(start_step)
        if num_examples < 1:
            raise ValueError("num_examples must be positive.")
        if not 0 <= start_step <= self.training_plan.mechanism.iterations:
            raise ValueError("start_step lies outside the prepared training schedule.")
        batches = self.upstream_plan.batch_selection_strategy.batch_iterator(
            num_examples, rng=self.sampler_seed
        )
        return itertools.islice(batches, start_step, None)

    def clipped_grad(
        self,
        loss_function: Callable[..., Any],
        /,
        *,
        argnums: int = 0,
        batch_argnums: int | tuple[int, ...] = 1,
        keep_batch_dim: bool = True,
        prng_argnum: int | None = None,
    ) -> Any:
        return self.upstream_plan.clipped_grad(
            loss_function,
            argnums=argnums,
            batch_argnums=batch_argnums,
            keep_batch_dim=keep_batch_dim,
            prng_argnum=prng_argnum,
        )

    def init_noise(self, parameters: Any, /) -> Any:
        return self.upstream_plan.noise_addition_transform.init(parameters)

    def checkpoint_noise_state(self, noise_state: Any, /) -> tuple[Any, Any]:
        key, inner_state = noise_state
        return jax.random.key_data(key), inner_state

    def restore_noise_state(self, checkpoint_state: tuple[Any, Any], /) -> Any:
        key_data, inner_state = checkpoint_state
        return jax.random.wrap_key_data(key_data), inner_state

    def privatize(self, clipped_gradient: Any, noise_state: Any, /) -> tuple[Any, Any]:
        return self.upstream_plan.noise_addition_transform.update(
            clipped_gradient, noise_state
        )

    def trace(self, completed_steps: int, /) -> MechanismTrace:
        completed_steps = int(completed_steps)
        if not 1 <= completed_steps <= self.training_plan.mechanism.iterations:
            raise ValueError("completed_steps lies outside the prepared mechanism.")
        return MechanismTrace(self.per_step_trace.event_json, completed_steps)

    def qualification_profile_id(self) -> str:
        from ..qualification import builtin_capability_catalog

        mechanism = self.training_plan.mechanism
        support = {
            "unit": "operator-case",
            "adjacency": "add-or-remove-one",
            "trust-model": "central",
            "sampling": "poisson",
            "mechanism": "gaussian-dp-sgd",
            "accounting-provider-version": "0.6.0",
            "accountant": mechanism.accounting_method.value,
            "accountant-configuration": (
                "pld-discretization-1e-4"
                if mechanism.accounting_method is AccountingMethod.PLD
                else "rdp-default-orders"
            ),
            "microbatch-size": (
                "none" if mechanism.microbatch_size is None else mechanism.microbatch_size
            ),
            "dtype": mechanism.dtype,
            "process-count": 1,
            "device-count": 1,
            "randomness": "research-prng",
            "prng-implementation": "threefry2x32",
            "provider-version": self.provider_version,
            "static-case-schema": "fixed",
            "data-source": "in-memory-case-source",
        }
        declaration = builtin_capability_catalog().declaration("privacy.control-plane")
        matches = tuple(
            profile
            for profile in declaration.profiles
            if any(dict(item.attributes) == support for item in profile.support_tuples)
        )
        if len(matches) != 1:
            raise RuntimeError(
                "Prepared private mechanism has no unique qualification profile."
            )
        return matches[0].profile_id

    def certificate(self, completed_steps: int, /) -> PrivacyCertificate:
        return certify_private_release(
            self.training_plan.scope,
            self.trace(completed_steps),
            self.training_plan.budget,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            mechanism_id=self.mechanism_id,
            mechanism_plan_id=self.prepared_id,
            query_l2_sensitivity=(
                self.training_plan.mechanism.clipping_norm
                / self.training_plan.mechanism.normalize_by
            ),
            planned_iterations=self.training_plan.mechanism.iterations,
            randomness=self.randomness,
            accounting_method=self.training_plan.mechanism.accounting_method,
            qualification_profile_id=self.qualification_profile_id(),
        )


def _prepare_private_gradient(
    training_plan: PrivateTrainingPlan,
    /,
    *,
    noise_key: jax.Array,
    sampler_seed: int,
) -> _PreparedPrivateGradient:
    """Prepare standard Poisson DP-SGD from one reviewed upstream component set."""
    if not isinstance(training_plan, PrivateTrainingPlan):
        raise TypeError("training_plan must be a PrivateTrainingPlan.")
    mechanism = training_plan.mechanism
    if mechanism.budget.epsilon <= 0.0:
        raise ValueError("Finite-noise DP-SGD requires a positive epsilon budget.")
    key_data = jax.random.key_data(noise_key)
    if key_data.ndim != 1 or key_data.size == 0:
        raise TypeError("noise_key must be one JAX PRNG key.")
    if str(jax.random.key_impl(noise_key)) != "threefry2x32":
        raise ValueError(
            "The initial private provider supports only JAX threefry2x32 keys."
        )
    sampler_seed = int(sampler_seed)
    if sampler_seed < 0:
        raise ValueError("sampler_seed must be non-negative.")

    dp, batch_selection, clipping, noise_addition, accounting, execution_plan = (
        _upstream_modules()
    )

    def event_for_noise(noise_multiplier: float) -> object:
        return accounting.dpsgd_event(
            noise_multiplier,
            mechanism.iterations,
            sampling_prob=mechanism.sampling_probability,
        )

    def make_accountant() -> object:
        return _fresh_accountant(mechanism.accounting_method, dp)

    noise_multiplier = float(
        dp.calibrate_dp_mechanism(
            make_fresh_accountant=make_accountant,
            make_event_from_param=event_for_noise,
            target_epsilon=mechanism.budget.epsilon,
            target_delta=mechanism.budget.delta,
        )
    )
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError(
            "DP accountant did not produce a finite positive noise multiplier."
        )

    dtype = np.float32 if mechanism.dtype == "float32" else np.float64

    def clipped_grad_transform(*args: Any, **kwargs: Any) -> Any:
        return clipping.clipped_grad(
            *args,
            **kwargs,
            l2_clip_norm=mechanism.clipping_norm,
            rescale_to_unit_norm=False,
            normalize_by=mechanism.normalize_by,
            microbatch_size=mechanism.microbatch_size,
            nan_safe=True,
            dtype=dtype,
        )

    sensitivity = mechanism.clipping_norm / mechanism.normalize_by
    privatizer = noise_addition.gaussian_privatizer(
        stddev=noise_multiplier * sensitivity,
        prng_key=noise_key,
        dtype=dtype,
    )
    sampler = batch_selection.CyclicPoissonSampling(
        sampling_prob=mechanism.sampling_probability,
        iterations=mechanism.iterations,
        partition_type=batch_selection.PartitionType.INDEPENDENT,
    )
    per_step_event = dp.PoissonSampledDpEvent(
        mechanism.sampling_probability,
        dp.GaussianDpEvent(noise_multiplier),
    )
    per_step_trace = MechanismTrace.from_event(per_step_event)
    full_trace = MechanismTrace(per_step_trace.event_json, mechanism.iterations)
    planned_guarantee = account_mechanism_trace(
        full_trace,
        mechanism.scope.definition,
        mechanism.budget,
        method=mechanism.accounting_method,
    )
    upstream = execution_plan.DPExecutionPlan(
        clipped_grad=clipped_grad_transform,
        batch_selection_strategy=sampler,
        noise_addition_transform=privatizer,
        dp_event=event_for_noise(noise_multiplier),
    )
    return _PreparedPrivateGradient(
        training_plan,
        noise_multiplier,
        per_step_trace,
        planned_guarantee,
        upstream,
        sampler_seed,
    )

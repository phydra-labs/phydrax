#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field

from .._fingerprint import canonical_fingerprint
from .._validation import canonical_identifier, finite_real_scalar
from ._accounting import AccountingMethod, PrivacyBudget
from ._definition import NeighboringRelation, PrivateDataScope, TrustModel


@dataclass(frozen=True, slots=True)
class DPSGDPlan:
    """One exact case-level Poisson-sampled Gaussian DP-SGD mechanism."""

    scope: PrivateDataScope
    budget: PrivacyBudget
    sampling_probability: float
    iterations: int
    clipping_norm: float
    normalize_by: float = 1.0
    microbatch_size: int | None = None
    dtype: str = "float32"
    accounting_method: AccountingMethod = AccountingMethod.PLD
    provider_id: str = "jax-privacy"
    provider_version: str = "2.0.0"
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PrivateDataScope):
            raise TypeError("scope must be a PrivateDataScope.")
        if not isinstance(self.budget, PrivacyBudget):
            raise TypeError("budget must be a PrivacyBudget.")
        if self.scope.definition.trust_model is not TrustModel.CENTRAL:
            raise ValueError("The initial DP-SGD profile supports only central DP.")
        if (
            self.scope.definition.neighboring_relation
            is not NeighboringRelation.ADD_OR_REMOVE_ONE
        ):
            raise ValueError(
                "The initial DP-SGD profile supports only add/remove-one adjacency."
            )
        probability = finite_real_scalar(
            self.sampling_probability, "sampling_probability"
        )
        clipping_norm = finite_real_scalar(self.clipping_norm, "clipping_norm")
        normalize_by = finite_real_scalar(self.normalize_by, "normalize_by")
        if type(self.iterations) is not int:
            raise TypeError("iterations must be an integer.")
        iterations = self.iterations
        if not 0.0 < probability <= 1.0:
            raise ValueError("sampling_probability must lie in (0, 1].")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if clipping_norm <= 0.0 or normalize_by <= 0.0:
            raise ValueError("clipping_norm and normalize_by must be positive.")
        if self.microbatch_size is not None and type(self.microbatch_size) is not int:
            raise TypeError("microbatch_size must be an integer or None.")
        if self.microbatch_size not in (None, 1):
            raise ValueError(
                "Variable Poisson batches currently support only microbatch_size=1."
            )
        if self.dtype not in ("float32", "float64"):
            raise ValueError("Private DP-SGD supports only float32 or float64.")
        if not isinstance(self.accounting_method, AccountingMethod):
            raise TypeError("accounting_method must be an AccountingMethod.")
        provider_id = canonical_identifier(self.provider_id, "provider ID")
        provider_version = canonical_identifier(self.provider_version, "provider version")
        if provider_id != "jax-privacy" or provider_version != "2.0.0":
            raise ValueError("The initial DP-SGD profile is pinned to jax-privacy 2.0.0.")
        object.__setattr__(self, "sampling_probability", probability)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "clipping_norm", clipping_norm)
        object.__setattr__(self, "normalize_by", normalize_by)
        if self.microbatch_size is not None:
            object.__setattr__(self, "microbatch_size", self.microbatch_size)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "provider_version", provider_version)
        object.__setattr__(self, "plan_id", canonical_fingerprint(self._content_record()))

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "dp-sgd-plan",
            "scope_contract_id": self.scope.scope_contract_id,
            "budget": self.budget.to_record(),
            "sampling_probability": self.sampling_probability,
            "iterations": self.iterations,
            "clipping_norm": self.clipping_norm,
            "normalize_by": self.normalize_by,
            "microbatch_size": self.microbatch_size,
            "dtype": self.dtype,
            "accounting_method": self.accounting_method.value,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "plan_id": self.plan_id}


@dataclass(frozen=True, slots=True)
class PrivateTrainingPlan:
    """Privacy mechanism plus fail-closed preprocessing and reporting policy."""

    mechanism: DPSGDPlan
    normalization_is_public: bool = True
    validation_is_public: bool = False
    release_private_metrics: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism, DPSGDPlan):
            raise TypeError("mechanism must be a DPSGDPlan.")
        if any(
            type(value) is not bool
            for value in (
                self.normalization_is_public,
                self.validation_is_public,
                self.release_private_metrics,
            )
        ):
            raise TypeError("Private training policy flags must be booleans.")
        if not self.normalization_is_public:
            raise ValueError(
                "Private normalization is unsupported until a preprocessing event "
                "can be composed into the mechanism trace."
            )
        if self.release_private_metrics:
            raise ValueError(
                "Private metric release is unsupported until a metric mechanism "
                "can be composed into the mechanism trace."
            )
        object.__setattr__(self, "plan_id", canonical_fingerprint(self._content_record()))

    @property
    def scope(self) -> PrivateDataScope:
        return self.mechanism.scope

    @property
    def budget(self) -> PrivacyBudget:
        return self.mechanism.budget

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "private-training-plan",
            "mechanism": self.mechanism.to_record(),
            "normalization_is_public": self.normalization_is_public,
            "validation_is_public": self.validation_is_public,
            "release_private_metrics": self.release_private_metrics,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "plan_id": self.plan_id}


__all__ = ["DPSGDPlan", "PrivateTrainingPlan"]

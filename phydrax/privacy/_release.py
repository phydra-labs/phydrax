#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._fingerprint import canonical_fingerprint
from .._validation import canonical_identifier, finite_real_scalar
from ._accounting import (
    account_mechanism_traces,
    AccountingMethod,
    MechanismTrace,
    PrivacyBudget,
    PrivacyGuarantee,
)
from ._definition import _require_record, PrivateDataScope, RandomnessAssurance


@dataclass(frozen=True, slots=True)
class PrivacyCertificate:
    """Public-safe evidence binding one DP mechanism trace to one release root."""

    scope: PrivateDataScope
    trace: MechanismTrace
    guarantee: PrivacyGuarantee
    provider_id: str
    provider_version: str
    mechanism_id: str
    mechanism_plan_id: str
    query_l2_sensitivity: float
    planned_iterations: int
    randomness: RandomnessAssurance
    qualification_profile_id: str | None = None
    qualification_released: bool = False
    certificate_id: str = field(init=False)
    public_release_allowed: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PrivateDataScope):
            raise TypeError("scope must be a PrivateDataScope.")
        if not isinstance(self.trace, MechanismTrace):
            raise TypeError("trace must be a MechanismTrace.")
        if not isinstance(self.guarantee, PrivacyGuarantee):
            raise TypeError("guarantee must be a PrivacyGuarantee.")
        if self.guarantee.definition_id != self.scope.definition.definition_id:
            raise ValueError("Privacy guarantee and data scope definitions differ.")
        if self.guarantee.trace_id != self.trace.trace_id:
            raise ValueError("Privacy guarantee and mechanism trace identities differ.")
        provider_id = canonical_identifier(self.provider_id, "provider ID")
        provider_version = canonical_identifier(self.provider_version, "provider version")
        mechanism_id = canonical_identifier(self.mechanism_id, "mechanism ID")
        mechanism_plan_id = canonical_identifier(
            self.mechanism_plan_id, "mechanism plan ID"
        )
        sensitivity = finite_real_scalar(
            self.query_l2_sensitivity, "query_l2_sensitivity"
        )
        if type(self.planned_iterations) is not int:
            raise TypeError("planned_iterations must be an integer.")
        planned_iterations = self.planned_iterations
        if sensitivity <= 0.0:
            raise ValueError("query_l2_sensitivity must be positive.")
        if planned_iterations < self.trace.repetitions:
            raise ValueError(
                "planned_iterations cannot be smaller than the mechanism trace."
            )
        if type(self.qualification_released) is not bool:
            raise TypeError("qualification_released must be a boolean.")
        profile_id = self.qualification_profile_id
        if profile_id is not None:
            profile_id = canonical_identifier(profile_id, "qualification profile ID")
        if self.qualification_released:
            raise ValueError(
                "No current privacy capability profile is public-release authorized."
            )
        if not isinstance(self.randomness, RandomnessAssurance):
            raise TypeError("randomness must be a RandomnessAssurance.")
        expected_guarantee = account_mechanism_traces(
            (self.trace,),
            self.scope.definition,
            PrivacyBudget(1.7976931348623157e308, self.guarantee.delta),
            method=self.guarantee.accounting_method,
        )
        if expected_guarantee != self.guarantee:
            raise ValueError(
                "Privacy guarantee is not the exact native accounting result for its trace."
            )
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "provider_version", provider_version)
        object.__setattr__(self, "mechanism_id", mechanism_id)
        object.__setattr__(self, "mechanism_plan_id", mechanism_plan_id)
        object.__setattr__(self, "query_l2_sensitivity", sensitivity)
        object.__setattr__(self, "planned_iterations", planned_iterations)
        object.__setattr__(self, "qualification_profile_id", profile_id)
        object.__setattr__(self, "public_release_allowed", False)
        object.__setattr__(
            self, "certificate_id", canonical_fingerprint(self._content_record())
        )

    def require_public_release(self) -> None:
        if not self.public_release_allowed:
            raise PermissionError(
                "This privacy certificate is not qualified for public release."
            )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "privacy-certificate",
            "scope": self.scope.to_record(),
            "trace": self.trace.to_record(),
            "guarantee": self.guarantee.to_record(),
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "mechanism_id": self.mechanism_id,
            "mechanism_plan_id": self.mechanism_plan_id,
            "query_l2_sensitivity": self.query_l2_sensitivity,
            "planned_iterations": self.planned_iterations,
            "randomness": self.randomness.value,
            "qualification_profile_id": self.qualification_profile_id,
            "qualification_released": self.qualification_released,
            "public_release_allowed": self.public_release_allowed,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "certificate_id": self.certificate_id}

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> PrivacyCertificate:
        _require_record(
            record,
            "privacy-certificate",
            frozenset(
                (
                    "scope",
                    "trace",
                    "guarantee",
                    "provider_id",
                    "provider_version",
                    "mechanism_id",
                    "mechanism_plan_id",
                    "query_l2_sensitivity",
                    "planned_iterations",
                    "randomness",
                    "qualification_profile_id",
                    "qualification_released",
                    "public_release_allowed",
                )
            ),
            optional=frozenset(("certificate_id",)),
        )
        scope = record["scope"]
        trace = record["trace"]
        guarantee = record["guarantee"]
        if not isinstance(scope, dict):
            raise TypeError("Serialized certificate scope must be a mapping.")
        if not isinstance(trace, Mapping):
            raise TypeError("Serialized certificate trace must be a mapping.")
        if not isinstance(guarantee, Mapping):
            raise TypeError("Serialized certificate guarantee must be a mapping.")
        profile_id = record["qualification_profile_id"]
        value = cls(
            PrivateDataScope.from_record(scope),
            MechanismTrace.from_record(trace),
            PrivacyGuarantee.from_record(guarantee),
            str(record["provider_id"]),
            str(record["provider_version"]),
            str(record["mechanism_id"]),
            str(record["mechanism_plan_id"]),
            float(record["query_l2_sensitivity"]),
            _exact_integer(record["planned_iterations"], "planned_iterations"),
            RandomnessAssurance(str(record["randomness"])),
            None if profile_id is None else str(profile_id),
            _exact_bool(record["qualification_released"], "qualification_released"),
        )
        if (
            _exact_bool(record["public_release_allowed"], "public_release_allowed")
            != value.public_release_allowed
        ):
            raise ValueError("Serialized certificate has an invalid release disposition.")
        recorded_id = record.get("certificate_id")
        if recorded_id is not None and str(recorded_id) != value.certificate_id:
            raise ValueError(
                "Serialized privacy certificate has an invalid content address."
            )
        return value


@dataclass(frozen=True, slots=True)
class PrivacyReleaseReceipt:
    """One unique randomized release root charged to a privacy ledger."""

    release_root_id: str
    certificate: PrivacyCertificate
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        root_id = canonical_identifier(self.release_root_id, "release root ID")
        if not isinstance(self.certificate, PrivacyCertificate):
            raise TypeError("certificate must be a PrivacyCertificate.")
        object.__setattr__(self, "release_root_id", root_id)
        object.__setattr__(
            self, "receipt_id", canonical_fingerprint(self._content_record())
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "privacy-release-receipt",
            "release_root_id": self.release_root_id,
            "certificate_id": self.certificate.certificate_id,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._content_record(),
            "certificate": self.certificate.to_record(),
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True, slots=True)
class PrivacyReleaseLedger:
    """Immutable budget ledger over distinct release roots in one data scope."""

    scope: PrivateDataScope
    budget: PrivacyBudget
    accounting_method: AccountingMethod = AccountingMethod.PLD
    receipts: tuple[PrivacyReleaseReceipt, ...] = ()
    parent_ledger_id: str | None = None
    ledger_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PrivateDataScope):
            raise TypeError("scope must be a PrivateDataScope.")
        if not isinstance(self.budget, PrivacyBudget):
            raise TypeError("budget must be a PrivacyBudget.")
        if not isinstance(self.accounting_method, AccountingMethod):
            raise TypeError("accounting_method must be an AccountingMethod.")
        if any(not isinstance(value, PrivacyReleaseReceipt) for value in self.receipts):
            raise TypeError("receipts must contain PrivacyReleaseReceipt values.")
        root_ids = tuple(value.release_root_id for value in self.receipts)
        if len(set(root_ids)) != len(root_ids):
            raise ValueError("Release-root identities must be unique within one ledger.")
        for receipt in self.receipts:
            if (
                receipt.certificate.scope.scope_contract_id
                != self.scope.scope_contract_id
            ):
                raise ValueError("Every release receipt must use the ledger data scope.")
        if self.receipts and self.parent_ledger_id is None:
            raise ValueError(
                "A nonempty privacy ledger requires its parent ledger identity."
            )
        if not self.receipts and self.parent_ledger_id is not None:
            raise ValueError("An empty privacy ledger cannot declare a parent.")
        if self.parent_ledger_id is not None:
            canonical_identifier(self.parent_ledger_id, "parent ledger ID")
        if self.receipts:
            account_mechanism_traces(
                tuple(value.certificate.trace for value in self.receipts),
                self.scope.definition,
                self.budget,
                method=self.accounting_method,
            )
        object.__setattr__(
            self, "ledger_id", canonical_fingerprint(self._content_record())
        )

    @property
    def spent(self) -> PrivacyGuarantee | None:
        if not self.receipts:
            return None
        return account_mechanism_traces(
            tuple(value.certificate.trace for value in self.receipts),
            self.scope.definition,
            self.budget,
            method=self.accounting_method,
        )

    def register(
        self, release_root_id: str, certificate: PrivacyCertificate, /
    ) -> PrivacyReleaseLedger:
        if not isinstance(certificate, PrivacyCertificate):
            raise TypeError("certificate must be a PrivacyCertificate.")
        root_id = canonical_identifier(release_root_id, "release root ID")
        if certificate.scope.scope_contract_id != self.scope.scope_contract_id:
            raise ValueError("Certificate and release ledger data scopes differ.")
        for receipt in self.receipts:
            if receipt.release_root_id == root_id:
                if receipt.certificate.certificate_id != certificate.certificate_id:
                    raise ValueError(
                        "One release root cannot identify two privacy certificates."
                    )
                return self
        receipt = PrivacyReleaseReceipt(root_id, certificate)
        return PrivacyReleaseLedger(
            self.scope,
            self.budget,
            self.accounting_method,
            self.receipts + (receipt,),
            self.ledger_id,
        )

    def require_successor(self, previous: PrivacyReleaseLedger, /) -> None:
        """Reject rollback, fork, or accounting regression from a persisted ledger head."""

        if not isinstance(previous, PrivacyReleaseLedger):
            raise TypeError("previous must be a PrivacyReleaseLedger.")
        if (
            self.scope.scope_contract_id != previous.scope.scope_contract_id
            or self.budget != previous.budget
            or self.accounting_method is not previous.accounting_method
            or self.parent_ledger_id != previous.ledger_id
            or self.receipts[:-1] != previous.receipts
        ):
            raise ValueError("Privacy ledger does not monotonically extend its parent.")
        current_spent = self.spent
        previous_spent = previous.spent
        if previous_spent is not None and (
            current_spent is None
            or current_spent.delta != previous_spent.delta
            or current_spent.epsilon + 1e-12 < previous_spent.epsilon
        ):
            raise ValueError("Privacy accounting cannot regress across ledger commits.")

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "privacy-release-ledger",
            "scope_contract_id": self.scope.scope_contract_id,
            "budget": self.budget.to_record(),
            "accounting_method": self.accounting_method.value,
            "parent_ledger_id": self.parent_ledger_id,
            "receipts": [value.to_record() for value in self.receipts],
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._content_record(),
            "scope": self.scope.to_record(),
            "ledger_id": self.ledger_id,
        }

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        /,
        *,
        previous: PrivacyReleaseLedger | None = None,
    ) -> PrivacyReleaseLedger:
        _require_record(
            record,
            "privacy-release-ledger",
            frozenset(
                (
                    "scope_contract_id",
                    "budget",
                    "accounting_method",
                    "receipts",
                    "scope",
                    "parent_ledger_id",
                )
            ),
            optional=frozenset(("ledger_id",)),
        )
        scope = record["scope"]
        budget = record["budget"]
        receipts = record["receipts"]
        parent_ledger_id = record["parent_ledger_id"]
        if parent_ledger_id is not None and type(parent_ledger_id) is not str:
            raise TypeError("Serialized parent ledger identity must be a string or null.")
        if not isinstance(scope, dict) or not isinstance(budget, Mapping):
            raise TypeError("Serialized ledger scope and budget must be mappings.")
        if not isinstance(receipts, list):
            raise TypeError("Serialized ledger receipts must be a list.")
        receipt_values: list[PrivacyReleaseReceipt] = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise TypeError("Every serialized release receipt must be a mapping.")
            _require_record(
                receipt,
                "privacy-release-receipt",
                frozenset(("release_root_id", "certificate_id", "certificate")),
                optional=frozenset(("receipt_id",)),
            )
            certificate = receipt["certificate"]
            if not isinstance(certificate, Mapping):
                raise TypeError("Serialized receipt certificate must be a mapping.")
            receipt_value = PrivacyReleaseReceipt(
                str(receipt["release_root_id"]),
                PrivacyCertificate.from_record(certificate),
            )
            if str(receipt["certificate_id"]) != receipt_value.certificate.certificate_id:
                raise ValueError(
                    "Serialized release receipt certificate identity changed."
                )
            recorded_receipt_id = receipt.get("receipt_id")
            if (
                recorded_receipt_id is not None
                and str(recorded_receipt_id) != receipt_value.receipt_id
            ):
                raise ValueError(
                    "Serialized release receipt has an invalid content address."
                )
            receipt_values.append(receipt_value)
        value = cls(
            PrivateDataScope.from_record(scope),
            PrivacyBudget.from_record(budget),
            AccountingMethod(str(record["accounting_method"])),
            tuple(receipt_values),
            parent_ledger_id,
        )
        if record["scope_contract_id"] != value.scope.scope_contract_id:
            raise ValueError("Serialized privacy ledger data-scope identity changed.")
        recorded_id = record.get("ledger_id")
        if recorded_id is not None and str(recorded_id) != value.ledger_id:
            raise ValueError("Serialized privacy ledger has an invalid content address.")
        if value.receipts:
            if previous is None:
                raise ValueError(
                    "Nonempty privacy ledger loading requires the exact previous head."
                )
            value.require_successor(previous)
        elif previous is not None:
            raise ValueError("Privacy ledger does not monotonically extend its parent.")
        return value


def certify_private_release(
    scope: PrivateDataScope,
    trace: MechanismTrace,
    budget: PrivacyBudget,
    /,
    *,
    provider_id: str,
    provider_version: str,
    mechanism_id: str,
    mechanism_plan_id: str,
    query_l2_sensitivity: float,
    planned_iterations: int,
    randomness: RandomnessAssurance,
    accounting_method: AccountingMethod = AccountingMethod.PLD,
    qualification_profile_id: str | None = None,
) -> PrivacyCertificate:
    guarantee = account_mechanism_traces(
        (trace,), scope.definition, budget, method=accounting_method
    )
    return PrivacyCertificate(
        scope,
        trace,
        guarantee,
        provider_id,
        provider_version,
        mechanism_id,
        mechanism_plan_id,
        query_l2_sensitivity,
        planned_iterations,
        randomness,
        qualification_profile_id,
    )


def _exact_bool(value: object, name: str, /) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean.")
    return value


def _exact_integer(value: object, name: str, /) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


__all__ = [
    "PrivacyCertificate",
    "PrivacyReleaseLedger",
    "PrivacyReleaseReceipt",
    "certify_private_release",
]

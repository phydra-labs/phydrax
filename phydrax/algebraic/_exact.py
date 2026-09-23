#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact sparse polynomials and closed symbolic-provider lifecycle contracts."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, StrEnum
from fractions import Fraction
from math import isqrt
from numbers import Integral
from typing import Any, Literal, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._system import SparsePolynomialSupport


_MAX_PRIME = 2_147_483_647


def _is_prime(value: int, /) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    candidate = 5
    step = 2
    limit = isqrt(value)
    while candidate <= limit:
        if value % candidate == 0:
            return False
        candidate += step
        step = 6 - step
    return True


def _fraction(value: object, /) -> Fraction:
    if isinstance(value, bool) or isinstance(value, (float, complex, np.inexact)):
        raise TypeError("Exact polynomial coefficients cannot be floating-point values.")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if not isinstance(value, str):
        raise TypeError("Exact coefficients must be integers, rationals, or strings.")
    text = value.strip()
    if not text or any(character.isspace() for character in text):
        raise ValueError("Exact coefficient strings must be nonempty without whitespace.")
    if text.count("/") > 1:
        raise ValueError("Exact rational strings must have at most one slash.")
    parts = text.split("/")
    if any(not part or part in ("+", "-") for part in parts):
        raise ValueError("Malformed exact rational coefficient string.")
    for part in parts:
        digits = part[1:] if part[0] in "+-" else part
        if not digits.isascii() or not digits.isdigit():
            raise ValueError("Exact coefficient strings must contain decimal integers.")
        if len(digits) > 1 and digits[0] == "0":
            raise ValueError("Exact coefficient strings cannot contain leading zeroes.")
    numerator = int(parts[0])
    denominator = 1 if len(parts) == 1 else int(parts[1])
    if denominator == 0:
        raise ValueError("Exact rational denominators must be nonzero.")
    return Fraction(numerator, denominator)


def _fraction_string(value: Fraction, /) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


class ExactCoefficientDomain(StrictModule):
    """One exact host coefficient ring; coefficient values are never JAX leaves."""

    kind: Literal["ZZ", "QQ", "GF"] = eqx.field(static=True)
    modulus: int | None = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(self, kind: Literal["ZZ", "QQ", "GF"], modulus: int | None = None, /):
        if kind not in ("ZZ", "QQ", "GF"):
            raise ValueError("Exact coefficient domain must be ZZ, QQ, or GF.")
        if kind == "GF":
            if isinstance(modulus, bool) or not isinstance(modulus, Integral):
                raise TypeError("GF modulus must be an integer.")
            modulus_ = int(modulus)
            if modulus_ > _MAX_PRIME:
                raise ValueError(f"GF modulus must not exceed {_MAX_PRIME}.")
            if not _is_prime(modulus_):
                raise ValueError("GF modulus must be prime.")
        else:
            if modulus is not None:
                raise ValueError("Only GF domains have a modulus.")
            modulus_ = None
        self.kind = kind
        self.modulus = modulus_
        self.domain_id = canonical_fingerprint(
            {"kind": "exact-coefficient-domain", "ring": kind, "modulus": modulus_}
        )

    @property
    def label(self) -> str:
        return self.kind if self.modulus is None else f"GF({self.modulus})"

    def normalize(self, value: object, /) -> str:
        """Return the unique transport string for one admitted exact coefficient."""
        rational = _fraction(value)
        if self.kind == "ZZ":
            if rational.denominator != 1:
                raise ValueError("ZZ coefficients must be integers.")
            return str(rational.numerator)
        if self.kind == "QQ":
            return _fraction_string(rational)
        assert self.modulus is not None
        denominator = rational.denominator % self.modulus
        if denominator == 0:
            raise ValueError("GF coefficient denominator is zero modulo the prime.")
        residue = (
            (rational.numerator % self.modulus)
            * pow(denominator, self.modulus - 2, self.modulus)
        ) % self.modulus
        return str(residue)

    def parse(self, value: object, /) -> int | Fraction:
        canonical = self.normalize(value)
        return Fraction(canonical) if self.kind == "QQ" else int(canonical)

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            **({} if self.modulus is None else {"modulus": self.modulus}),
        }


ZZ = ExactCoefficientDomain("ZZ")
QQ = ExactCoefficientDomain("QQ")


def GF(modulus: int, /) -> ExactCoefficientDomain:
    """Construct the exact prime field with canonical least-nonnegative residues."""
    return ExactCoefficientDomain("GF", modulus)


class ExactSparsePolynomialSystem(StrictModule):
    """Canonical exact coefficients attached to shared sparse monomial support."""

    support: SparsePolynomialSupport
    coefficients: tuple[str, ...] = eqx.field(static=True)
    domain: ExactCoefficientDomain = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SparsePolynomialSupport,
        coefficients: Sequence[object],
        domain: ExactCoefficientDomain = QQ,
    ):
        if not isinstance(support, SparsePolynomialSupport):
            raise TypeError("support must be SparsePolynomialSupport.")
        if not isinstance(domain, ExactCoefficientDomain):
            raise TypeError("domain must be an ExactCoefficientDomain.")
        values = tuple(domain.normalize(value) for value in coefficients)
        if len(values) != support.term_count:
            raise ValueError("Exact coefficient count must equal support term count.")
        self.support = support
        self.coefficients = values
        self.domain = domain
        self.system_id = canonical_fingerprint(
            {
                "kind": "exact-sparse-polynomial-system",
                "support": support.support_id,
                "domain": domain.domain_id,
                "coefficients": list(values),
            }
        )

    @classmethod
    def from_coo(
        cls,
        variable_labels: Sequence[str],
        equation_labels: Sequence[str],
        equation_indices: object,
        exponents: object,
        coefficients: Sequence[object],
        domain: ExactCoefficientDomain = QQ,
        *,
        groups: Sequence[object] = (),
    ) -> ExactSparsePolynomialSystem:
        """Jointly canonicalize COO terms and their exact host coefficients."""
        support = SparsePolynomialSupport(
            variable_labels,
            equation_labels,
            equation_indices,
            exponents,
            groups=groups,
        )
        raw = tuple(coefficients)
        if len(raw) != support.term_count:
            raise ValueError("Exact coefficient count must equal support term count.")
        permutation = tuple(support.canonical_term_permutation)
        return cls(support, tuple(raw[index] for index in permutation), domain)

    @property
    def canonical_coefficients(self) -> tuple[str, ...]:
        return self.coefficients

    @property
    def variable_count(self) -> int:
        return self.support.variable_count

    @property
    def equation_count(self) -> int:
        return self.support.equation_count

    @property
    def term_count(self) -> int:
        return self.support.term_count

    def coefficient_values(self) -> tuple[int | Fraction, ...]:
        return tuple(self.domain.parse(value) for value in self.coefficients)


class ExactSymbolicOperation(StrEnum):
    GROEBNER_BASIS = "groebner_basis"
    NORMAL_FORM = "normal_form"
    ELIMINATE = "eliminate"
    RESULTANT_UNIVARIATE = "resultant_univariate"
    DISCRIMINANT_UNIVARIATE = "discriminant_univariate"


class ExactSymbolicStatus(IntEnum):
    SUCCESS = 0
    PROVIDER_FAILED = 1
    INVALID_OUTPUT = 2
    IDENTITY_MISMATCH = 3


MonomialOrder: TypeAlias = Literal["grevlex", "lex"]


def _order(value: str, /) -> MonomialOrder:
    if value not in ("grevlex", "lex"):
        raise ValueError("Exact monomial order must be 'grevlex' or 'lex'.")
    return value


class GroebnerBasisArguments(StrictModule):
    monomial_order: MonomialOrder = eqx.field(static=True)

    def __init__(self, monomial_order: MonomialOrder = "grevlex", /):
        self.monomial_order = _order(monomial_order)

    def to_record(self) -> dict[str, object]:
        return {"monomial_order": self.monomial_order}


class NormalFormArguments(StrictModule):
    polynomial: ExactSparsePolynomialSystem
    monomial_order: MonomialOrder = eqx.field(static=True)

    def __init__(
        self,
        polynomial: ExactSparsePolynomialSystem,
        monomial_order: MonomialOrder = "grevlex",
        /,
    ):
        if not isinstance(polynomial, ExactSparsePolynomialSystem):
            raise TypeError("normal-form polynomial must be exact.")
        if polynomial.equation_count != 1:
            raise ValueError("Normal form accepts exactly one polynomial dividend.")
        self.polynomial = polynomial
        self.monomial_order = _order(monomial_order)

    def to_record(self) -> dict[str, object]:
        return {
            "monomial_order": self.monomial_order,
            "polynomial": _system_record(self.polynomial),
        }


class EliminateArguments(StrictModule):
    variable_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, variable_indices: Sequence[int], /):
        values = tuple(
            _index(value, "elimination variable") for value in variable_indices
        )
        if (
            not values
            or len(values) != len(set(values))
            or values != tuple(sorted(values))
        ):
            raise ValueError(
                "Elimination variables must be nonempty, unique, and ordered."
            )
        self.variable_indices = values

    def to_record(self) -> dict[str, object]:
        return {"variable_indices": list(self.variable_indices)}


class UnivariateResultantArguments(StrictModule):
    equation_indices: tuple[int, int] = eqx.field(static=True)
    variable_index: int = eqx.field(static=True)

    def __init__(
        self, equation_indices: Sequence[int] = (0, 1), variable_index: int = 0, /
    ):
        equations = tuple(
            _index(value, "resultant equation") for value in equation_indices
        )
        if len(equations) != 2 or equations[0] == equations[1]:
            raise ValueError("Univariate resultant requires two distinct equations.")
        self.equation_indices = equations
        self.variable_index = _index(variable_index, "resultant variable")

    def to_record(self) -> dict[str, object]:
        return {
            "equation_indices": list(self.equation_indices),
            "variable_index": self.variable_index,
        }


class UnivariateDiscriminantArguments(StrictModule):
    equation_index: int = eqx.field(static=True)
    variable_index: int = eqx.field(static=True)

    def __init__(self, equation_index: int = 0, variable_index: int = 0, /):
        self.equation_index = _index(equation_index, "discriminant equation")
        self.variable_index = _index(variable_index, "discriminant variable")

    def to_record(self) -> dict[str, object]:
        return {
            "equation_index": self.equation_index,
            "variable_index": self.variable_index,
        }


ExactSymbolicArguments: TypeAlias = (
    GroebnerBasisArguments
    | NormalFormArguments
    | EliminateArguments
    | UnivariateResultantArguments
    | UnivariateDiscriminantArguments
)


class ExactSymbolicPlan(StrictModule):
    """Bounded, data-only request for one closed exact symbolic operation."""

    system: ExactSparsePolynomialSystem
    operation: ExactSymbolicOperation = eqx.field(static=True)
    arguments: ExactSymbolicArguments
    maximum_input_bytes: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    timeout_seconds: float = eqx.field(static=True)
    maximum_variable_count: int = eqx.field(static=True)
    maximum_equation_count: int = eqx.field(static=True)
    maximum_term_count: int = eqx.field(static=True)
    maximum_exponent_entries: int = eqx.field(static=True)
    maximum_storage_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: ExactSparsePolynomialSystem,
        operation: ExactSymbolicOperation,
        arguments: ExactSymbolicArguments,
        /,
        *,
        maximum_input_bytes: int = 1 << 20,
        maximum_output_bytes: int = 8 << 20,
        timeout_seconds: float = 120.0,
        maximum_variable_count: int = 4_096,
        maximum_equation_count: int = 100_000,
        maximum_term_count: int = 1_000_000,
        maximum_exponent_entries: int = 10_000_000,
        maximum_storage_bytes: int = 256 * 1024 * 1024,
    ):
        if not isinstance(system, ExactSparsePolynomialSystem):
            raise TypeError("system must be ExactSparsePolynomialSystem.")
        operation_ = ExactSymbolicOperation(operation)
        _validate_arguments(system, operation_, arguments)
        input_limit = _positive_integer(maximum_input_bytes, "maximum_input_bytes")
        output_limit = _positive_integer(maximum_output_bytes, "maximum_output_bytes")
        resource_limits = (
            _positive_integer(maximum_variable_count, "maximum_variable_count"),
            _positive_integer(maximum_equation_count, "maximum_equation_count"),
            _positive_integer(maximum_term_count, "maximum_term_count"),
            _positive_integer(maximum_exponent_entries, "maximum_exponent_entries"),
            _positive_integer(maximum_storage_bytes, "maximum_storage_bytes"),
        )
        hard_limits = (4_096, 100_000, 1_000_000, 10_000_000, 256 * 1024 * 1024)
        if any(
            value > hard for value, hard in zip(resource_limits, hard_limits, strict=True)
        ):
            raise ValueError("Exact symbolic resource limits exceed worker bounds.")
        timeout = float(timeout_seconds)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite.")
        systems = [system]
        if isinstance(arguments, NormalFormArguments):
            systems.append(arguments.polynomial)
        term_count = sum(item.term_count for item in systems)
        exponent_entries = sum(item.term_count * item.variable_count for item in systems)
        estimated_storage = (
            sum(item.variable_count * 64 + item.equation_count * 64 for item in systems)
            + term_count * 40
            + exponent_entries * 8
            + sum(
                sum(len(value.encode("ascii")) for value in item.coefficients)
                for item in systems
            )
        )
        checks = (
            (system.variable_count, resource_limits[0], "variable_count"),
            (system.equation_count, resource_limits[1], "equation_count"),
            (term_count, resource_limits[2], "term_count"),
            (exponent_entries, resource_limits[3], "exponent_entries"),
            (estimated_storage, resource_limits[4], "estimated_storage_bytes"),
        )
        for observed, maximum, name in checks:
            if observed > maximum:
                raise ValueError(f"Exact symbolic {name}={observed} exceeds {maximum}.")
        self.system = system
        self.operation = operation_
        self.arguments = arguments
        self.maximum_input_bytes = input_limit
        self.maximum_output_bytes = output_limit
        self.timeout_seconds = timeout
        (
            self.maximum_variable_count,
            self.maximum_equation_count,
            self.maximum_term_count,
            self.maximum_exponent_entries,
            self.maximum_storage_bytes,
        ) = resource_limits
        self.plan_id = canonical_fingerprint(
            {
                "kind": "exact-symbolic-plan",
                "system": system.system_id,
                "operation": operation_.value,
                "arguments": arguments.to_record(),
                "maximum_input_bytes": input_limit,
                "maximum_output_bytes": output_limit,
                "timeout_seconds": timeout,
                "maximum_variable_count": resource_limits[0],
                "maximum_equation_count": resource_limits[1],
                "maximum_term_count": resource_limits[2],
                "maximum_exponent_entries": resource_limits[3],
                "maximum_storage_bytes": resource_limits[4],
            }
        )


class PreparedExactSymbolic(StrictModule):
    """One exact plan bound to one explicit pinned provider environment."""

    plan: ExactSymbolicPlan
    provider: Any = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(self, plan: ExactSymbolicPlan, provider: Any, request_id: str, /):
        if not isinstance(plan, ExactSymbolicPlan):
            raise TypeError("plan must be ExactSymbolicPlan.")
        identifier = str(request_id)
        if not identifier:
            raise ValueError("request_id must be nonempty.")
        self.plan = plan
        self.provider = provider
        self.request_id = identifier


class ExactSymbolicEvidence(StrictModule):
    """External claim identity and independently checked transport evidence."""

    provider_id: str = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    executable_sha256: str = eqx.field(static=True)
    worker_sha256: str = eqx.field(static=True)
    run_artifact_id: str = eqx.field(static=True)
    independently_checked: tuple[str, ...] = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider_id: str,
        provider_version: str,
        executable_sha256: str,
        worker_sha256: str,
        run_artifact_id: str,
        independently_checked: Sequence[str],
        claim: str = "exact_claimed_by_external_provider",
    ):
        fields = tuple(
            str(value)
            for value in (
                provider_id,
                provider_version,
                executable_sha256,
                worker_sha256,
                run_artifact_id,
                claim,
            )
        )
        checks = tuple(str(value) for value in independently_checked)
        if any(not value for value in (*fields, *checks)):
            raise ValueError("Exact symbolic evidence fields must be nonempty.")
        (
            self.provider_id,
            self.provider_version,
            self.executable_sha256,
            self.worker_sha256,
            self.run_artifact_id,
            self.claim,
        ) = fields
        self.independently_checked = checks
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "exact-symbolic-evidence",
                "provider": fields[0],
                "provider_version": fields[1],
                "executable": fields[2],
                "worker": fields[3],
                "run": fields[4],
                "claim": fields[5],
                "independently_checked": list(checks),
            }
        )


class ExactSymbolicResult(StrictModule):
    """Exact provider payload with external claim kept separate from local checks."""

    status: ExactSymbolicStatus = eqx.field(static=True)
    output: ExactSparsePolynomialSystem | None
    evidence: ExactSymbolicEvidence | None
    plan_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    diagnostic: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: ExactSymbolicStatus,
        output: ExactSparsePolynomialSystem | None,
        evidence: ExactSymbolicEvidence | None,
        /,
        *,
        plan_id: str,
        request_id: str,
        diagnostic: str = "",
    ):
        status_ = ExactSymbolicStatus(status)
        if status_ is ExactSymbolicStatus.SUCCESS:
            if output is None or evidence is None:
                raise ValueError(
                    "Successful exact symbolic results require output and evidence."
                )
        elif output is not None or evidence is not None:
            raise ValueError(
                "Failed exact symbolic results cannot carry output or exactness evidence."
            )
        self.status = status_
        self.output = output
        self.evidence = evidence
        self.plan_id = str(plan_id)
        self.request_id = str(request_id)
        self.diagnostic = str(diagnostic)
        self.result_id = canonical_fingerprint(
            {
                "kind": "exact-symbolic-result",
                "status": int(status_),
                "output": None if output is None else output.system_id,
                "evidence": None if evidence is None else evidence.evidence_id,
                "plan": self.plan_id,
                "request": self.request_id,
                "diagnostic": self.diagnostic,
            }
        )

    @property
    def externally_claimed_exact(self) -> bool:
        return (
            self.status is ExactSymbolicStatus.SUCCESS
            and self.output is not None
            and self.evidence is not None
        )


ExactPolynomialResult = ExactSymbolicResult


def plan_exact_symbolic(
    system: ExactSparsePolynomialSystem,
    operation: ExactSymbolicOperation,
    arguments: ExactSymbolicArguments | None = None,
    /,
    *,
    maximum_input_bytes: int = 1 << 20,
    maximum_output_bytes: int = 8 << 20,
    timeout_seconds: float = 120.0,
    maximum_variable_count: int = 4_096,
    maximum_equation_count: int = 100_000,
    maximum_term_count: int = 1_000_000,
    maximum_exponent_entries: int = 10_000_000,
    maximum_storage_bytes: int = 256 * 1024 * 1024,
) -> ExactSymbolicPlan:
    operation_ = ExactSymbolicOperation(operation)
    selected = _default_arguments(operation_) if arguments is None else arguments
    return ExactSymbolicPlan(
        system,
        operation_,
        selected,
        maximum_input_bytes=maximum_input_bytes,
        maximum_output_bytes=maximum_output_bytes,
        timeout_seconds=timeout_seconds,
        maximum_variable_count=maximum_variable_count,
        maximum_equation_count=maximum_equation_count,
        maximum_term_count=maximum_term_count,
        maximum_exponent_entries=maximum_exponent_entries,
        maximum_storage_bytes=maximum_storage_bytes,
    )


def prepare_exact_symbolic(
    plan: ExactSymbolicPlan, provider: Any, /
) -> PreparedExactSymbolic:
    from ..backends.macaulay2 import prepare_macaulay2_symbolic

    return prepare_macaulay2_symbolic(plan, provider)


def execute_exact_symbolic(prepared: PreparedExactSymbolic, /) -> ExactSymbolicResult:
    from ..backends.macaulay2 import execute_macaulay2_symbolic

    return execute_macaulay2_symbolic(prepared)


def _system_record(system: ExactSparsePolynomialSystem, /) -> dict[str, object]:
    return {
        "system_id": system.system_id,
        "support_id": system.support.support_id,
        "domain": system.domain.to_record(),
        "variable_count": system.variable_count,
        "equation_count": system.equation_count,
        "equation_indices": np.asarray(system.support.equation_indices).tolist(),
        "exponents": np.asarray(system.support.exponents).tolist(),
        "coefficients": list(system.coefficients),
    }


def _index(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} index must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} index must be nonnegative.")
    return result


def _positive_integer(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _default_arguments(operation: ExactSymbolicOperation, /) -> ExactSymbolicArguments:
    if operation is ExactSymbolicOperation.GROEBNER_BASIS:
        return GroebnerBasisArguments()
    if operation is ExactSymbolicOperation.ELIMINATE:
        raise ValueError("Elimination requires explicit variable indices.")
    if operation is ExactSymbolicOperation.NORMAL_FORM:
        raise ValueError("Normal form requires an explicit exact polynomial dividend.")
    if operation is ExactSymbolicOperation.RESULTANT_UNIVARIATE:
        return UnivariateResultantArguments()
    return UnivariateDiscriminantArguments()


def _validate_arguments(
    system: ExactSparsePolynomialSystem,
    operation: ExactSymbolicOperation,
    arguments: ExactSymbolicArguments,
    /,
) -> None:
    expected: dict[ExactSymbolicOperation, type[ExactSymbolicArguments]] = {
        ExactSymbolicOperation.GROEBNER_BASIS: GroebnerBasisArguments,
        ExactSymbolicOperation.NORMAL_FORM: NormalFormArguments,
        ExactSymbolicOperation.ELIMINATE: EliminateArguments,
        ExactSymbolicOperation.RESULTANT_UNIVARIATE: UnivariateResultantArguments,
        ExactSymbolicOperation.DISCRIMINANT_UNIVARIATE: UnivariateDiscriminantArguments,
    }
    if not isinstance(arguments, expected[operation]):
        raise TypeError(f"Arguments do not match operation {operation.value!r}.")
    if isinstance(arguments, NormalFormArguments):
        polynomial = arguments.polynomial
        if (
            polynomial.domain.domain_id != system.domain.domain_id
            or polynomial.support.variable_labels != system.support.variable_labels
        ):
            raise ValueError(
                "Normal-form dividend must use the ideal's domain and variables."
            )
    elif isinstance(arguments, EliminateArguments):
        if arguments.variable_indices[-1] >= system.variable_count:
            raise ValueError("Elimination variable index is outside the system.")
        if len(arguments.variable_indices) >= system.variable_count:
            raise ValueError("Initial elimination must retain at least one variable.")
    elif isinstance(arguments, UnivariateResultantArguments):
        if system.variable_count != 1 or arguments.variable_index != 0:
            raise ValueError("Univariate resultant requires exactly one system variable.")
        if max(arguments.equation_indices) >= system.equation_count:
            raise ValueError("Resultant equation index is outside the system.")
    elif isinstance(arguments, UnivariateDiscriminantArguments):
        if system.variable_count != 1 or arguments.variable_index != 0:
            raise ValueError(
                "Univariate discriminant requires exactly one system variable."
            )
        if arguments.equation_index >= system.equation_count:
            raise ValueError("Discriminant equation index is outside the system.")


__all__ = [
    "EliminateArguments",
    "ExactCoefficientDomain",
    "ExactPolynomialResult",
    "ExactSparsePolynomialSystem",
    "ExactSymbolicArguments",
    "ExactSymbolicEvidence",
    "ExactSymbolicOperation",
    "ExactSymbolicPlan",
    "ExactSymbolicResult",
    "ExactSymbolicStatus",
    "GF",
    "GroebnerBasisArguments",
    "NormalFormArguments",
    "PreparedExactSymbolic",
    "QQ",
    "UnivariateDiscriminantArguments",
    "UnivariateResultantArguments",
    "ZZ",
    "execute_exact_symbolic",
    "plan_exact_symbolic",
    "prepare_exact_symbolic",
]

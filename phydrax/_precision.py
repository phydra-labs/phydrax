#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from ._fingerprint import canonical_fingerprint


RealPrecisionDType: TypeAlias = Literal[
    "float16",
    "bfloat16",
    "float32",
    "float64",
]
ComplexPrecisionDType: TypeAlias = Literal["complex64", "complex128"]
ScalarPrecisionDType: TypeAlias = RealPrecisionDType | ComplexPrecisionDType
PrecisionRole: TypeAlias = Literal[
    "storage",
    "coefficient",
    "compute",
    "factorization",
    "preconditioner",
    "basis",
    "accumulation",
    "residual",
    "certification",
    "communication",
    "checkpoint",
    "output",
]

_PRECISION_DTYPES = frozenset(
    ("float16", "bfloat16", "float32", "float64", "complex64", "complex128")
)
_PRECISION_ROLES = frozenset(
    (
        "storage",
        "coefficient",
        "compute",
        "factorization",
        "preconditioner",
        "basis",
        "accumulation",
        "residual",
        "certification",
        "communication",
        "checkpoint",
        "output",
    )
)
_CONTRACT_VERSION = 1


def precision_dtype_name(value: Any, /) -> ScalarPrecisionDType:
    """Return one canonical supported JAX scalar dtype name."""
    dtype = jnp.dtype(jax.dtypes.canonicalize_dtype(jnp.dtype(value)))
    name = dtype.name
    if name not in _PRECISION_DTYPES:
        raise ValueError(f"Unsupported precision dtype {name!r}.")
    return name


def real_precision_dtype_name(value: Any, /) -> RealPrecisionDType:
    """Return one canonical supported real floating dtype name."""
    name = precision_dtype_name(value)
    if name not in ("float16", "bfloat16", "float32", "float64"):
        raise ValueError(f"Precision dtype {name!r} is not real floating-point.")
    return name


def complex_precision_dtype(value: RealPrecisionDType | Any, /) -> ComplexPrecisionDType:
    """Return the complex companion used for one real precision dtype."""
    name = real_precision_dtype_name(value)
    return "complex128" if name == "float64" else "complex64"


def precision_itemsize(value: Any, /) -> int:
    return int(jnp.dtype(precision_dtype_name(value)).itemsize)


def _identifier(name: str, value: Any, /) -> str:
    resolved = str(value)
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


def _canonical_entries(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
    /,
) -> tuple[tuple[str, ScalarPrecisionDType | None], ...]:
    items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    names = tuple(str(name) for name, _ in items)
    if any(name not in _PRECISION_ROLES for name in names):
        invalid = tuple(sorted(name for name in names if name not in _PRECISION_ROLES))
        raise ValueError(f"Unknown precision roles {invalid!r}.")
    if len(set(names)) != len(names):
        raise ValueError("Precision roles must be unique.")
    return tuple(
        sorted(
            (
                (
                    str(name),
                    None if value is None else precision_dtype_name(value),
                )
                for name, value in items
            ),
            key=lambda item: item[0],
        )
    )


def _mapping(value: Any, name: str, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _strict_fields(value: Mapping[str, Any], expected: set[str], owner: str, /) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{owner} must use the current canonical fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )


@dataclass(frozen=True, slots=True)
class PrecisionRequest:
    """Versioned domain request without execution or resource claims."""

    domain: str
    requested: tuple[tuple[str, ScalarPrecisionDType | None], ...]
    request_id: str
    version: int = _CONTRACT_VERSION

    def __init__(
        self,
        domain: str,
        requested: Mapping[str, Any] | Sequence[tuple[str, Any]],
        /,
    ):
        domain_ = _identifier("domain", domain)
        requested_ = _canonical_entries(requested)
        object.__setattr__(self, "domain", domain_)
        object.__setattr__(self, "requested", requested_)
        object.__setattr__(self, "version", _CONTRACT_VERSION)
        object.__setattr__(
            self,
            "request_id",
            canonical_fingerprint(
                {
                    "kind": "precision-request",
                    "version": _CONTRACT_VERSION,
                    "domain": domain_,
                    "requested": dict(requested_),
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "requested": dict(self.requested),
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> PrecisionRequest:
        expected = {"version", "domain", "requested", "request_id"}
        _strict_fields(value, expected, "PrecisionRequest")
        if int(value["version"]) != _CONTRACT_VERSION:
            raise ValueError("Unsupported precision request version.")
        result = cls(
            str(value["domain"]),
            _mapping(value["requested"], "requested"),
        )
        if result.request_id != value["request_id"]:
            raise ValueError("Precision request identity mismatch.")
        return result


@dataclass(frozen=True, slots=True)
class PrecisionResolution:
    """Provider-accepted effective stage dtypes for one request."""

    request_id: str
    domain: str
    provider: str
    effective: tuple[tuple[str, ScalarPrecisionDType | None], ...]
    resolution_id: str
    version: int = _CONTRACT_VERSION

    def __init__(
        self,
        request: PrecisionRequest,
        provider: str,
        effective: Mapping[str, Any] | Sequence[tuple[str, Any]],
        /,
    ):
        if not isinstance(request, PrecisionRequest):
            raise TypeError("request must be a PrecisionRequest.")
        provider_ = _identifier("provider", provider)
        effective_ = _canonical_entries(effective)
        requested_roles = {name for name, _ in request.requested}
        effective_roles = {name for name, _ in effective_}
        if effective_roles != requested_roles:
            raise ValueError(
                "Resolved precision roles must exactly match requested roles."
            )
        object.__setattr__(self, "request_id", request.request_id)
        object.__setattr__(self, "domain", request.domain)
        object.__setattr__(self, "provider", provider_)
        object.__setattr__(self, "effective", effective_)
        object.__setattr__(self, "version", _CONTRACT_VERSION)
        object.__setattr__(
            self,
            "resolution_id",
            canonical_fingerprint(
                {
                    "kind": "precision-resolution",
                    "version": _CONTRACT_VERSION,
                    "request": request.request_id,
                    "domain": request.domain,
                    "provider": provider_,
                    "effective": dict(effective_),
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "domain": self.domain,
            "provider": self.provider,
            "effective": dict(self.effective),
            "resolution_id": self.resolution_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> PrecisionResolution:
        expected = {
            "version",
            "request_id",
            "domain",
            "provider",
            "effective",
            "resolution_id",
        }
        _strict_fields(value, expected, "PrecisionResolution")
        if int(value["version"]) != _CONTRACT_VERSION:
            raise ValueError("Unsupported precision resolution version.")
        effective = _canonical_entries(_mapping(value["effective"], "effective"))
        payload = {
            "kind": "precision-resolution",
            "version": _CONTRACT_VERSION,
            "request": str(value["request_id"]),
            "domain": str(value["domain"]),
            "provider": str(value["provider"]),
            "effective": dict(effective),
        }
        resolution_id = canonical_fingerprint(payload)
        if resolution_id != value["resolution_id"]:
            raise ValueError("Precision resolution identity mismatch.")
        result = object.__new__(cls)
        object.__setattr__(result, "request_id", str(value["request_id"]))
        object.__setattr__(result, "domain", _identifier("domain", value["domain"]))
        object.__setattr__(result, "provider", _identifier("provider", value["provider"]))
        object.__setattr__(result, "effective", effective)
        object.__setattr__(result, "resolution_id", resolution_id)
        object.__setattr__(result, "version", _CONTRACT_VERSION)
        return result


@dataclass(frozen=True, slots=True)
class PrecisionEvidenceEnvelope:
    """Observed execution precision with nested child-domain evidence."""

    resolution_id: str
    domain: str
    provider: str
    observed: tuple[tuple[str, ScalarPrecisionDType | None], ...]
    children: tuple[tuple[str, PrecisionEvidenceEnvelope], ...]
    evidence_id: str
    version: int = _CONTRACT_VERSION

    def __init__(
        self,
        resolution: PrecisionResolution,
        observed: Mapping[str, Any] | Sequence[tuple[str, Any]],
        /,
        *,
        children: Mapping[str, PrecisionEvidenceEnvelope]
        | Sequence[tuple[str, PrecisionEvidenceEnvelope]] = (),
    ):
        if not isinstance(resolution, PrecisionResolution):
            raise TypeError("resolution must be a PrecisionResolution.")
        observed_ = _canonical_entries(observed)
        effective_roles = {name for name, _ in resolution.effective}
        if {name for name, _ in observed_} != effective_roles:
            raise ValueError("Observed precision roles must match resolved roles.")
        child_items = (
            tuple(children.items()) if isinstance(children, Mapping) else tuple(children)
        )
        child_names = tuple(str(name) for name, _ in child_items)
        if any(not name for name in child_names) or len(set(child_names)) != len(
            child_names
        ):
            raise ValueError(
                "Precision evidence child names must be unique and non-empty."
            )
        if not all(
            isinstance(child, PrecisionEvidenceEnvelope) for _, child in child_items
        ):
            raise TypeError("Precision evidence children must be evidence envelopes.")
        children_ = tuple(sorted(child_items, key=lambda item: str(item[0])))
        payload = {
            "kind": "precision-evidence",
            "version": _CONTRACT_VERSION,
            "resolution": resolution.resolution_id,
            "domain": resolution.domain,
            "provider": resolution.provider,
            "observed": dict(observed_),
            "children": [[str(name), child.evidence_id] for name, child in children_],
        }
        object.__setattr__(self, "resolution_id", resolution.resolution_id)
        object.__setattr__(self, "domain", resolution.domain)
        object.__setattr__(self, "provider", resolution.provider)
        object.__setattr__(self, "observed", observed_)
        object.__setattr__(
            self,
            "children",
            tuple((str(name), child) for name, child in children_),
        )
        object.__setattr__(self, "version", _CONTRACT_VERSION)
        object.__setattr__(self, "evidence_id", canonical_fingerprint(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "resolution_id": self.resolution_id,
            "domain": self.domain,
            "provider": self.provider,
            "observed": dict(self.observed),
            "children": {name: child.to_dict() for name, child in self.children},
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> PrecisionEvidenceEnvelope:
        expected = {
            "version",
            "resolution_id",
            "domain",
            "provider",
            "observed",
            "children",
            "evidence_id",
        }
        _strict_fields(value, expected, "PrecisionEvidenceEnvelope")
        if int(value["version"]) != _CONTRACT_VERSION:
            raise ValueError("Unsupported precision evidence version.")
        observed = _canonical_entries(_mapping(value["observed"], "observed"))
        child_mapping = _mapping(value["children"], "children")
        children = tuple(
            sorted(
                (
                    (str(name), cls.from_dict(_mapping(child, str(name))))
                    for name, child in child_mapping.items()
                ),
                key=lambda item: item[0],
            )
        )
        payload = {
            "kind": "precision-evidence",
            "version": _CONTRACT_VERSION,
            "resolution": str(value["resolution_id"]),
            "domain": str(value["domain"]),
            "provider": str(value["provider"]),
            "observed": dict(observed),
            "children": [[name, child.evidence_id] for name, child in children],
        }
        evidence_id = canonical_fingerprint(payload)
        if evidence_id != value["evidence_id"]:
            raise ValueError("Precision evidence identity mismatch.")
        result = object.__new__(cls)
        object.__setattr__(result, "resolution_id", str(value["resolution_id"]))
        object.__setattr__(result, "domain", _identifier("domain", value["domain"]))
        object.__setattr__(result, "provider", _identifier("provider", value["provider"]))
        object.__setattr__(result, "observed", observed)
        object.__setattr__(result, "children", children)
        object.__setattr__(result, "evidence_id", evidence_id)
        object.__setattr__(result, "version", _CONTRACT_VERSION)
        return result


@dataclass(frozen=True, slots=True)
class PrecisionResourceAssumptions:
    """Dtype item sizes used only for static resource accounting."""

    domain: str
    dtypes: tuple[tuple[str, ScalarPrecisionDType | None], ...]
    item_sizes: tuple[tuple[str, int | None], ...]
    assumptions_id: str
    version: int = _CONTRACT_VERSION

    def __init__(
        self,
        domain: str,
        dtypes: Mapping[str, Any] | Sequence[tuple[str, Any]],
        /,
    ):
        domain_ = _identifier("domain", domain)
        dtypes_ = _canonical_entries(dtypes)
        item_sizes = tuple(
            (name, None if dtype is None else int(np.dtype(dtype).itemsize))
            for name, dtype in dtypes_
        )
        object.__setattr__(self, "domain", domain_)
        object.__setattr__(self, "dtypes", dtypes_)
        object.__setattr__(self, "item_sizes", item_sizes)
        object.__setattr__(self, "version", _CONTRACT_VERSION)
        object.__setattr__(
            self,
            "assumptions_id",
            canonical_fingerprint(
                {
                    "kind": "precision-resource-assumptions",
                    "version": _CONTRACT_VERSION,
                    "domain": domain_,
                    "dtypes": dict(dtypes_),
                    "item_sizes": dict(item_sizes),
                }
            ),
        )

    def itemsize(self, role: PrecisionRole | str, /) -> int:
        role_ = str(role)
        for name, value in self.item_sizes:
            if name == role_:
                if value is None:
                    raise ValueError(f"Precision role {role_!r} has no dtype assumption.")
                return value
        raise KeyError(f"Unknown precision resource role {role_!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "dtypes": dict(self.dtypes),
            "item_sizes": dict(self.item_sizes),
            "assumptions_id": self.assumptions_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> PrecisionResourceAssumptions:
        expected = {
            "version",
            "domain",
            "dtypes",
            "item_sizes",
            "assumptions_id",
        }
        _strict_fields(value, expected, "PrecisionResourceAssumptions")
        if int(value["version"]) != _CONTRACT_VERSION:
            raise ValueError("Unsupported precision resource-assumptions version.")
        result = cls(str(value["domain"]), _mapping(value["dtypes"], "dtypes"))
        if (
            dict(result.item_sizes) != dict(_mapping(value["item_sizes"], "item_sizes"))
            or result.assumptions_id != value["assumptions_id"]
        ):
            raise ValueError("Precision resource-assumptions identity mismatch.")
        return result


__all__ = [
    "ComplexPrecisionDType",
    "PrecisionEvidenceEnvelope",
    "PrecisionRequest",
    "PrecisionResolution",
    "PrecisionResourceAssumptions",
    "PrecisionRole",
    "RealPrecisionDType",
    "ScalarPrecisionDType",
    "complex_precision_dtype",
    "precision_dtype_name",
    "precision_itemsize",
    "real_precision_dtype_name",
]

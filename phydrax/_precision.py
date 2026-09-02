#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule


RealPrecisionDType: TypeAlias = Literal[
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]
ComplexPrecisionDType: TypeAlias = Literal["complex64", "complex128"]
ScalarPrecisionDType: TypeAlias = RealPrecisionDType | ComplexPrecisionDType
MicroscalingElementFormat: TypeAlias = Literal[
    "float8_e4m3fn",
    "float8_e5m2",
    "float6_e2m3",
    "float6_e3m2",
    "float4_e2m1",
]
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
    (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
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

_MX_ALIASES = {
    "mxfp8-e4m3": "float8_e4m3fn",
    "mxfp8-e5m2": "float8_e5m2",
    "mxfp6-e2m3": "float6_e2m3",
    "mxfp6-e3m2": "float6_e3m2",
    "mxfp4-e2m1": "float4_e2m1",
}
_MX_BITS = {
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float6_e2m3": 6,
    "float6_e3m2": 6,
    "float4_e2m1": 4,
}


@dataclass(frozen=True, slots=True)
class MicroscalingFormat:
    """Portable OCP-style block-scaled storage format."""

    element_format: MicroscalingElementFormat
    block_size: int = 32
    scale_format: str = "float8_e8m0fnu"
    axis: int = -1
    packing: Literal["packed", "byte"] = "packed"

    def __init__(
        self,
        element_format: str,
        block_size: int = 32,
        scale_format: str = "float8_e8m0fnu",
        axis: int = -1,
        packing: Literal["packed", "byte"] | None = None,
    ):
        element = _MX_ALIASES.get(str(element_format), str(element_format))
        if element not in _MX_BITS:
            raise ValueError(f"Unsupported microscaling element format {element!r}.")
        block = int(block_size)
        if block < 1:
            raise ValueError("Microscaling block_size must be positive.")
        if str(scale_format) != "float8_e8m0fnu":
            raise ValueError("Only the OCP E8M0 scale format is supported.")
        packing_ = (
            ("byte" if _MX_BITS[element] == 8 else "packed")
            if packing is None
            else packing
        )
        if packing_ not in ("packed", "byte"):
            raise ValueError("Microscaling packing must be 'packed' or 'byte'.")
        if packing_ == "byte" and _MX_BITS[element] != 8:
            raise ValueError("Sub-byte microscaling formats require packed storage.")
        if block * _MX_BITS[element] % 8:
            raise ValueError(
                "Microscaling block_size must produce a whole-byte packed block."
            )
        object.__setattr__(self, "element_format", element)
        object.__setattr__(self, "block_size", block)
        object.__setattr__(self, "scale_format", "float8_e8m0fnu")
        object.__setattr__(self, "axis", int(axis))
        object.__setattr__(self, "packing", packing_)

    @property
    def bits_per_element(self) -> int:
        return _MX_BITS[self.element_format]

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_format": self.element_format,
            "block_size": self.block_size,
            "scale_format": self.scale_format,
            "axis": self.axis,
            "packing": self.packing,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> MicroscalingFormat:
        expected = {
            "element_format",
            "block_size",
            "scale_format",
            "axis",
            "packing",
        }
        _strict_fields(value, expected, "MicroscalingFormat")
        return cls(
            str(value["element_format"]),
            int(value["block_size"]),
            str(value["scale_format"]),
            int(value["axis"]),
            str(value["packing"]),
        )


PrecisionFormat: TypeAlias = (
    ScalarPrecisionDType | ComplexPrecisionDType | MicroscalingFormat
)


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
    if name not in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ):
        raise ValueError(f"Precision dtype {name!r} is not real floating-point.")
    return name


def complex_precision_dtype(value: RealPrecisionDType | Any, /) -> ComplexPrecisionDType:
    """Return the complex companion used for one real precision dtype."""
    name = real_precision_dtype_name(value)
    return "complex128" if name == "float64" else "complex64"


def precision_itemsize(value: Any, /) -> int:
    if isinstance(value, MicroscalingFormat):
        raise ValueError(
            "Microscaling formats have fractional payload widths; use storage_bytes."
        )
    return int(jnp.dtype(precision_dtype_name(value)).itemsize)


def _identifier(name: str, value: Any, /) -> str:
    resolved = str(value)
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


def _canonical_format(value: Any, /) -> PrecisionFormat:
    if isinstance(value, MicroscalingFormat):
        return value
    if isinstance(value, Mapping):
        return MicroscalingFormat.from_dict(value)
    return precision_dtype_name(value)


def _format_payload(value: PrecisionFormat | None, /) -> Any:
    if value is None or isinstance(value, str):
        return value
    return value.to_dict()


def _entries_payload(
    entries: Sequence[tuple[str, PrecisionFormat | None]],
    /,
) -> dict[str, Any]:
    return {name: _format_payload(value) for name, value in entries}


def _canonical_entries(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
    /,
) -> tuple[tuple[str, PrecisionFormat | None], ...]:
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
                    None if value is None else _canonical_format(value),
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
    requested: tuple[tuple[str, PrecisionFormat | None], ...]
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
        for role, format_ in requested_:
            if role in ("accumulation", "residual", "certification") and (
                isinstance(format_, MicroscalingFormat)
                or (isinstance(format_, str) and format_.startswith("float8_"))
            ):
                raise ValueError(
                    f"Precision role {role!r} requires float32 or wider precision."
                )
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
                    "requested": _entries_payload(requested_),
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "requested": _entries_payload(self.requested),
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
    effective: tuple[tuple[str, PrecisionFormat | None], ...]
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
                    "effective": _entries_payload(effective_),
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "domain": self.domain,
            "provider": self.provider,
            "effective": _entries_payload(self.effective),
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
            "effective": _entries_payload(effective),
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
    observed: tuple[tuple[str, PrecisionFormat | None], ...]
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
            "observed": _entries_payload(observed_),
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
            "observed": _entries_payload(self.observed),
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
            "observed": _entries_payload(observed),
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
    dtypes: tuple[tuple[str, PrecisionFormat | None], ...]
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
            (
                name,
                None
                if dtype is None or isinstance(dtype, MicroscalingFormat)
                else precision_itemsize(dtype),
            )
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
                    "dtypes": _entries_payload(dtypes_),
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

    def storage_bytes(
        self,
        role: PrecisionRole | str,
        shape: Sequence[int],
        /,
    ) -> int:
        role_ = str(role)
        shape_ = tuple(int(size) for size in shape)
        if any(size < 0 for size in shape_):
            raise ValueError("Storage shape dimensions must be non-negative.")
        format_: PrecisionFormat | None = None
        for name, value in self.dtypes:
            if name == role_:
                format_ = value
                break
        else:
            raise KeyError(f"Unknown precision resource role {role_!r}.")
        if format_ is None:
            raise ValueError(f"Precision role {role_!r} has no format assumption.")
        if not isinstance(format_, MicroscalingFormat):
            return prod(shape_) * precision_itemsize(format_)
        if not shape_:
            axis_size = 1
            outer = 1
        else:
            axis = format_.axis
            if axis < 0:
                axis += len(shape_)
            if axis < 0 or axis >= len(shape_):
                raise ValueError("Microscaling axis lies outside the storage shape.")
            axis_size = shape_[axis]
            outer = prod(shape_[:axis] + shape_[axis + 1 :])
        blocks = ceil(axis_size / format_.block_size) if axis_size else 0
        padded = blocks * format_.block_size
        value_bytes = ceil(outer * padded * format_.bits_per_element / 8)
        scale_bytes = outer * blocks
        return value_bytes + scale_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "dtypes": _entries_payload(self.dtypes),
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


class MicroscaledArray(StrictModule):
    """Immutable fixed-shape block-scaled payload and numerical diagnostics."""

    packed_values: jax.Array
    scales: jax.Array
    finite: jax.Array
    saturation_count: jax.Array
    original_shape: tuple[int, ...] = eqx.field(static=True)
    padded_size: int = eqx.field(static=True)
    format: MicroscalingFormat = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        packed_values: Any,
        scales: Any,
        original_shape: Sequence[int],
        padded_size: int,
        format: MicroscalingFormat,
        finite: Any,
        saturation_count: Any,
        /,
    ):
        if not isinstance(format, MicroscalingFormat):
            raise TypeError("format must be a MicroscalingFormat.")
        packed = jnp.asarray(packed_values, dtype=jnp.uint8)
        scale_codes = jnp.asarray(scales, dtype=jnp.uint8)
        shape = tuple(int(size) for size in original_shape)
        padded = int(padded_size)
        if any(size < 0 for size in shape) or padded < 0:
            raise ValueError("Microscaled array shapes must be non-negative.")
        self.packed_values = packed
        self.scales = scale_codes
        self.finite = jnp.asarray(finite, dtype=bool)
        self.saturation_count = jnp.asarray(saturation_count, dtype=jnp.int32)
        self.original_shape = shape
        self.padded_size = padded
        self.format = format
        self.payload_id = canonical_fingerprint(
            {
                "kind": "microscaled-array",
                "format": format.to_dict(),
                "original_shape": list(shape),
                "padded_size": padded,
                "packed_shape": list(packed.shape),
                "scale_shape": list(scale_codes.shape),
            }
        )

    @property
    def payload_bytes(self) -> int:
        return int(self.packed_values.size + self.scales.size)


def _low_bit_parameters(element_format: str, /) -> tuple[int, int]:
    if element_format == "float4_e2m1":
        return 2, 1
    if element_format == "float6_e2m3":
        return 2, 3
    if element_format == "float6_e3m2":
        return 3, 2
    raise ValueError(f"{element_format!r} is not a low-bit finite format.")


def _decode_low_bit_codes(
    codes: jax.Array,
    exponent_bits: int,
    mantissa_bits: int,
    /,
) -> jax.Array:
    bits = 1 + exponent_bits + mantissa_bits
    sign = (codes >> (bits - 1)) & 1
    exponent_mask = (1 << exponent_bits) - 1
    exponent = (codes >> mantissa_bits) & exponent_mask
    mantissa = codes & ((1 << mantissa_bits) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    normal = (1.0 + mantissa.astype(jnp.float32) / float(1 << mantissa_bits)) * jnp.exp2(
        exponent.astype(jnp.float32) - float(bias)
    )
    subnormal = (
        mantissa.astype(jnp.float32)
        / float(1 << mantissa_bits)
        * jnp.exp2(jnp.asarray(1 - bias, dtype=jnp.float32))
    )
    magnitude = jnp.where(exponent == 0, subnormal, normal)
    return jnp.where(sign == 0, magnitude, -magnitude)


def _encode_low_bit_values(
    values: jax.Array,
    exponent_bits: int,
    mantissa_bits: int,
    /,
) -> jax.Array:
    bits = 1 + exponent_bits + mantissa_bits
    codebook = jnp.arange(1 << bits, dtype=jnp.uint8)
    levels = _decode_low_bit_codes(codebook, exponent_bits, mantissa_bits)
    distances = jnp.abs(values[..., None].astype(jnp.float32) - levels)
    minimum = jnp.min(distances, axis=-1, keepdims=True)
    tied = distances == minimum
    even = tied & ((codebook & 1) == 0)
    candidates = jnp.where(jnp.any(even, axis=-1, keepdims=True), even, tied)
    selected = jnp.argmax(candidates, axis=-1).astype(jnp.uint8)
    sign_code = jnp.asarray(1 << (bits - 1), dtype=jnp.uint8)
    return jnp.where(
        values == 0,
        jnp.where(jnp.signbit(values), sign_code, jnp.uint8(0)),
        selected,
    )


def _element_maximum(element_format: str, /) -> float:
    if element_format in ("float8_e4m3fn", "float8_e5m2"):
        return float(jnp.finfo(jnp.dtype(element_format)).max)
    exponent_bits, mantissa_bits = _low_bit_parameters(element_format)
    maximum_code = (1 << (exponent_bits + mantissa_bits)) - 1
    return float(
        _decode_low_bit_codes(
            jnp.asarray(maximum_code, dtype=jnp.uint8),
            exponent_bits,
            mantissa_bits,
        )
    )


def _pack_codes(codes: jax.Array, bits: int, /) -> jax.Array:
    if bits == 8:
        return codes
    if bits == 4:
        return codes[..., 0::2] | (codes[..., 1::2] << 4)
    if bits == 6:
        groups = codes.reshape(codes.shape[:-1] + (-1, 4))
        first = groups[..., 0] | ((groups[..., 1] & 0x03) << 6)
        second = (groups[..., 1] >> 2) | ((groups[..., 2] & 0x0F) << 4)
        third = (groups[..., 2] >> 4) | (groups[..., 3] << 2)
        return jnp.stack((first, second, third), axis=-1).reshape(
            codes.shape[:-1] + (-1,)
        )
    raise ValueError("Unsupported microscaling payload width.")


def _unpack_codes(packed: jax.Array, bits: int, block_size: int, /) -> jax.Array:
    if bits == 8:
        return packed
    if bits == 4:
        values = jnp.stack((packed & 0x0F, packed >> 4), axis=-1)
        return values.reshape(packed.shape[:-1] + (block_size,))
    if bits == 6:
        groups = packed.reshape(packed.shape[:-1] + (-1, 3))
        first = groups[..., 0] & 0x3F
        second = (groups[..., 0] >> 6) | ((groups[..., 1] & 0x0F) << 2)
        third = (groups[..., 1] >> 4) | ((groups[..., 2] & 0x03) << 4)
        fourth = groups[..., 2] >> 2
        values = jnp.stack((first, second, third, fourth), axis=-1)
        return values.reshape(packed.shape[:-1] + (block_size,))
    raise ValueError("Unsupported microscaling payload width.")


def quantize_mx(
    value: Any,
    format: MicroscalingFormat,
    /,
    *,
    overflow: Literal["error", "saturate"] = "error",
) -> MicroscaledArray:
    """Quantize with deterministic RNE, E8M0 scales, and explicit overflow."""
    if not isinstance(format, MicroscalingFormat):
        raise TypeError("format must be a MicroscalingFormat.")
    if overflow not in ("error", "saturate"):
        raise ValueError("overflow must be 'error' or 'saturate'.")
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError("Microscaling quantization requires a real floating array.")
    original_shape = tuple(int(size) for size in array.shape)
    scalar_input = array.ndim == 0
    working = array.reshape((1,)) if scalar_input else array
    axis = 0 if scalar_input else format.axis
    if axis < 0:
        axis += working.ndim
    if axis < 0 or axis >= working.ndim:
        raise ValueError("Microscaling axis lies outside the input rank.")
    moved = jnp.moveaxis(working, axis, -1)
    axis_size = moved.shape[-1]
    blocks = ceil(axis_size / format.block_size) if axis_size else 0
    padded_size = blocks * format.block_size
    padding = padded_size - axis_size
    padded = jnp.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(0, padding)])
    blocked = padded.reshape(moved.shape[:-1] + (blocks, format.block_size))
    maximum = jnp.asarray(_element_maximum(format.element_format), dtype=array.dtype)
    maximum_absolute = jnp.max(jnp.abs(blocked), axis=-1)
    ratio = maximum_absolute / maximum
    exponent = jnp.where(
        maximum_absolute == 0,
        0,
        jnp.ceil(jnp.log2(ratio)),
    )
    exponent = jnp.clip(exponent, -127, 127).astype(jnp.int32)
    scales = jnp.exp2(exponent.astype(array.dtype))
    normalized = blocked / scales[..., None]
    nonfinite = ~jnp.isfinite(blocked)
    overflowed = jnp.abs(normalized) > maximum
    saturation_count = jnp.sum(nonfinite | overflowed, dtype=jnp.int32)
    if overflow == "error":
        normalized = eqx.error_if(
            normalized,
            saturation_count != 0,
            "Microscaling input is nonfinite or outside the representable range.",
        )
    normalized = jnp.nan_to_num(
        normalized,
        nan=0.0,
        posinf=maximum,
        neginf=-maximum,
    )
    normalized = jnp.clip(normalized, -maximum, maximum)
    if format.element_format in ("float8_e4m3fn", "float8_e5m2"):
        element_dtype = jnp.dtype(format.element_format)
        quantized = normalized.astype(element_dtype)
        codes = jax.lax.bitcast_convert_type(quantized, jnp.uint8)
    else:
        exponent_bits, mantissa_bits = _low_bit_parameters(format.element_format)
        codes = _encode_low_bit_values(
            normalized,
            exponent_bits,
            mantissa_bits,
        )
    packed = _pack_codes(codes, format.bits_per_element)
    scale_codes = (exponent + 127).astype(jnp.uint8)
    return MicroscaledArray(
        packed,
        scale_codes,
        original_shape,
        padded_size,
        format,
        jnp.all(jnp.isfinite(array)),
        saturation_count,
    )


def dequantize_mx(
    value: MicroscaledArray,
    /,
    *,
    dtype: Any = jnp.float32,
) -> jax.Array:
    """Decode a portable microscaling payload into explicit compute precision."""
    if not isinstance(value, MicroscaledArray):
        raise TypeError("value must be a MicroscaledArray.")
    output_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(output_dtype, jnp.floating):
        raise TypeError("Microscaling output dtype must be real floating-point.")
    format = value.format
    codes = _unpack_codes(
        value.packed_values,
        format.bits_per_element,
        format.block_size,
    )
    if format.element_format in ("float8_e4m3fn", "float8_e5m2"):
        element_dtype = jnp.dtype(format.element_format)
        decoded = jax.lax.bitcast_convert_type(codes, element_dtype).astype(output_dtype)
    else:
        exponent_bits, mantissa_bits = _low_bit_parameters(format.element_format)
        decoded = _decode_low_bit_codes(
            codes,
            exponent_bits,
            mantissa_bits,
        ).astype(output_dtype)
    scales = jnp.exp2(
        value.scales.astype(jnp.int32).astype(output_dtype)
        - jnp.asarray(127, dtype=output_dtype)
    )
    blocked = decoded * scales[..., None]
    moved = blocked.reshape(blocked.shape[:-2] + (value.padded_size,))
    original_shape = value.original_shape
    scalar_output = not original_shape
    working_shape = (1,) if scalar_output else original_shape
    axis = 0 if scalar_output else format.axis
    if axis < 0:
        axis += len(working_shape)
    axis_size = working_shape[axis]
    moved = moved[..., :axis_size]
    restored = jnp.moveaxis(moved, -1, axis).reshape(working_shape)
    return restored.reshape(()) if scalar_output else restored


__all__ = [
    "ComplexPrecisionDType",
    "MicroscaledArray",
    "MicroscalingElementFormat",
    "MicroscalingFormat",
    "PrecisionEvidenceEnvelope",
    "PrecisionFormat",
    "PrecisionRequest",
    "PrecisionResolution",
    "PrecisionResourceAssumptions",
    "PrecisionRole",
    "RealPrecisionDType",
    "ScalarPrecisionDType",
    "complex_precision_dtype",
    "dequantize_mx",
    "precision_dtype_name",
    "precision_itemsize",
    "quantize_mx",
    "real_precision_dtype_name",
]

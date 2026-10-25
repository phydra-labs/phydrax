#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scientific conditioning and observable requests for native atomistic models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntFlag, StrEnum
from typing import Any, assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from typing_extensions import TypeForm

from .._differentiation import (
    ComponentAuthority,
    DerivativeSurface,
    DifferentiationRequest,
)
from .._fingerprint import canonical_fingerprint
from .._model import register_artifact_value
from .._model._ports import ValuePort
from .._strict import StrictModule
from .._trainable import ExplicitFreeze, NonTrainableState
from .._validation import (
    canonical_identifier,
    finite_real_scalar,
    integer,
    positive_integer,
)
from ..typing import (
    AnyShape,
    as_array,
    as_host_array,
    Bool,
    Casting,
    checked,
    ConvertibleToArray,
    Float,
    Float32,
    Float64,
    HostShaped,
    Identifier,
    Int32,
    Int64,
    parse,
    Scope,
    Size,
)
from ..units import CHARGE, DIMENSIONLESS, UnitDefinition
from ._dimensions import AtomisticAtomDim, AtomisticCaseDim, AtomisticNodeDim


TOTAL_CHARGE_PORT = ValuePort(
    "atomistic.total-charge",
    event_shape=(),
    component_ids=("charge",),
    representation="case-total-charge",
    dimensions=(CHARGE,),
    normalization_id="identity",
)
SPIN_MULTIPLICITY_PORT = ValuePort(
    "atomistic.spin-multiplicity",
    event_shape=(),
    component_ids=("multiplicity",),
    representation="physical-spin-multiplicity-2s-plus-1",
    dimensions=(DIMENSIONLESS,),
    normalization_id="identity",
)

_FIELD_RECORD_FIELDS = frozenset(
    {
        "name",
        "port",
        "unit",
        "level",
        "kind",
        "dtype",
        "required",
        "minimum",
        "maximum",
        "categories",
        "field_id",
    }
)
_CONDITIONING_RECORD_FIELDS = frozenset(
    {"spec", "case_count", "atom_capacity", "bindings"}
)
# How a recorded field enters the conditioning constructor: complete values,
# one explicit default event, or an absent optional field.
_CONDITION_BINDINGS = ("value", "default", "absent")


def _validate_port_unit(port: ValuePort, unit: UnitDefinition | None, /) -> None:
    dimensions = port.dimensions
    if dimensions is None:
        return
    if unit is None:
        if any(dimension != DIMENSIONLESS for dimension in dimensions):
            raise ValueError("A dimensioned atomistic port requires an explicit unit.")
    elif any(dimension != unit.dimension for dimension in dimensions):
        raise ValueError("The atomistic port and its unit have incompatible dimensions.")


class AtomisticConditionLevel(StrEnum):
    CASE = "case"
    ATOM = "atom"


class AtomisticConditionKind(StrEnum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class AtomisticConditionDType(StrEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


class AtomisticConditionStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    OUT_OF_RANGE = 2
    INVALID_CATEGORY = 4


def _condition_form(dtype: AtomisticConditionDType, /) -> TypeForm[Array]:
    match dtype:
        case AtomisticConditionDType.FLOAT32:
            return Float32[AnyShape]
        case AtomisticConditionDType.FLOAT64:
            return Float64[AnyShape]
        case AtomisticConditionDType.INT32:
            return Int32[AnyShape]
        case AtomisticConditionDType.INT64:
            return Int64[AnyShape]
        case AtomisticConditionDType.BOOL:
            return Bool[AnyShape]
        case _ as unreachable:
            assert_never(unreachable)


def _condition_bound(
    value: int | float | None,
    kind: AtomisticConditionKind,
    name: str,
    /,
) -> int | float | None:
    if value is None:
        return None
    match kind:
        case AtomisticConditionKind.CONTINUOUS:
            return finite_real_scalar(value, name)
        case AtomisticConditionKind.CATEGORICAL:
            return integer(value, name)
        case AtomisticConditionKind.BOOLEAN:
            raise ValueError("Boolean conditioning has no numerical bounds.")
        case _ as unreachable:
            assert_never(unreachable)


@final
class AtomisticConditionField(StrictModule, NonTrainableState):
    """One explicitly identified case- or atom-level condition.

    ``name`` is the input binding key; ``port`` owns its scientific meaning.
    Categories are admissible integer codes, not an inferred label ordering.
    Atom-level values use the same flat case-major rows as the model graph.
    """

    __strict_contract__ = True

    name: Identifier = eqx.field(static=True)
    port: ValuePort
    unit: UnitDefinition | None
    level: AtomisticConditionLevel = eqx.field(static=True)
    kind: AtomisticConditionKind = eqx.field(static=True)
    dtype: AtomisticConditionDType = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    minimum: int | float | None = eqx.field(static=True)
    maximum: int | float | None = eqx.field(static=True)
    categories: tuple[int, ...] = eqx.field(static=True)
    field_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        name: str,
        port: ValuePort,
        /,
        *,
        unit: UnitDefinition | None = None,
        level: AtomisticConditionLevel | str = AtomisticConditionLevel.CASE,
        kind: AtomisticConditionKind | str = AtomisticConditionKind.CONTINUOUS,
        dtype: DTypeLike = jnp.float64,
        required: bool = True,
        minimum: int | float | None = None,
        maximum: int | float | None = None,
        categories: Sequence[int] = (),
    ) -> None:
        name_ = canonical_identifier(name, "name")
        _validate_port_unit(port, unit)
        level_ = parse(
            AtomisticConditionLevel(level) if isinstance(level, str) else level,
            AtomisticConditionLevel,
            "level",
        )
        kind_ = parse(
            AtomisticConditionKind(kind) if isinstance(kind, str) else kind,
            AtomisticConditionKind,
            "kind",
        )
        dtype_ = parse(
            AtomisticConditionDType(np.dtype(dtype).name),
            AtomisticConditionDType,
            "dtype",
        )
        if not isinstance(required, bool):
            raise TypeError("required must be a boolean.")
        dtype_kind = np.dtype(dtype_.value).kind
        match kind_:
            case AtomisticConditionKind.CONTINUOUS:
                compatible = dtype_kind == "f"
            case AtomisticConditionKind.CATEGORICAL:
                compatible = dtype_kind in "iu"
            case AtomisticConditionKind.BOOLEAN:
                compatible = dtype_kind == "b"
            case _ as unreachable:
                assert_never(unreachable)
        if not compatible:
            raise ValueError("Condition kind and declared dtype are incompatible.")
        lower = _condition_bound(minimum, kind_, "minimum")
        upper = _condition_bound(maximum, kind_, "maximum")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Condition minimum must not exceed its maximum.")
        codes = tuple(integer(value, "category") for value in categories)
        if len(set(codes)) != len(codes):
            raise ValueError("Condition categories must be unique.")
        if codes and kind_ is not AtomisticConditionKind.CATEGORICAL:
            raise ValueError("Only categorical conditions declare category codes.")
        codes = tuple(sorted(codes))
        if kind_ is AtomisticConditionKind.CATEGORICAL:
            limits = np.iinfo(np.dtype(dtype_.value))
            bounds = (
                *codes,
                *((lower,) if lower is not None else ()),
                *((upper,) if upper is not None else ()),
            )
            if any(value < limits.min or value > limits.max for value in bounds):
                raise ValueError(
                    "Condition codes and bounds must fit the declared dtype."
                )
        if any(
            (lower is not None and code < lower) or (upper is not None and code > upper)
            for code in codes
        ):
            raise ValueError("Condition category lies outside its declared bounds.")
        identity = canonical_fingerprint(
            {
                "kind": "atomistic-condition-field",
                "name": name_,
                "port": port.port_id,
                "unit": None if unit is None else unit.unit_id,
                "level": level_.value,
                "condition_kind": kind_.value,
                "dtype": dtype_.value,
                "required": required,
                "minimum": lower,
                "maximum": upper,
                "categories": codes,
            }
        )
        self.name = name_
        self.port = port
        self.unit = unit
        self.level = level_
        self.kind = kind_
        self.dtype = dtype_
        self.required = required
        self.minimum = lower
        self.maximum = upper
        self.categories = codes
        self.field_id = identity

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON record of this field, with its port, unit and identity."""
        return {
            "name": self.name,
            "port": self.port.to_dict(),
            "unit": None if self.unit is None else self.unit.to_dict(),
            "level": self.level.value,
            "kind": self.kind.value,
            "dtype": self.dtype.value,
            "required": self.required,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "categories": list(self.categories),
            "field_id": self.field_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], /) -> AtomisticConditionField:
        """Rebuild a field through its constructor; any field or identity change refuses."""
        if not isinstance(payload, Mapping) or set(payload) != _FIELD_RECORD_FIELDS:
            raise ValueError("Condition field record must use the canonical fields.")
        categories = payload["categories"]
        unit = payload["unit"]
        if not isinstance(categories, list):
            raise TypeError("Condition field categories must be a list.")
        if unit is not None and not isinstance(unit, Mapping):
            raise TypeError("Condition field unit must be a unit record or None.")
        dtype = payload["dtype"]
        if not isinstance(dtype, str):
            raise TypeError("Condition field dtype must be text.")
        field = cls(
            payload["name"],
            ValuePort.from_dict(payload["port"]),
            unit=None if unit is None else UnitDefinition.from_dict(unit),
            level=payload["level"],
            kind=payload["kind"],
            dtype=AtomisticConditionDType(dtype).value,
            required=payload["required"],
            minimum=payload["minimum"],
            maximum=payload["maximum"],
            categories=tuple(categories),
        )
        if field.field_id != payload["field_id"] or field.to_dict() != dict(payload):
            raise ValueError("Condition field record does not match its content.")
        return field


@final
class AtomisticConditioningSpec(StrictModule, NonTrainableState):
    """Canonical named bindings to explicit scientific ports."""

    __strict_contract__ = True

    fields: tuple[AtomisticConditionField, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    spec_id: Identifier = eqx.field(static=True)

    def __init__(self, fields: Sequence[AtomisticConditionField] = (), /) -> None:
        items = tuple(fields)
        if any(not isinstance(field, AtomisticConditionField) for field in items):
            raise TypeError("fields must contain AtomisticConditionField values.")
        ordered = tuple(sorted(items, key=lambda field: field.name))
        names = tuple(field.name for field in ordered)
        ports = tuple(field.port.port_id for field in ordered)
        if len(set(names)) != len(names) or len(set(ports)) != len(ports):
            raise ValueError("Condition names and scientific ports must each be unique.")
        identity = canonical_fingerprint(
            {
                "kind": "atomistic-conditioning-spec",
                "fields": tuple(field.field_id for field in ordered),
            }
        )
        self.fields = ordered
        self.names = names
        self.spec_id = identity

    def index(self, name: str, /) -> int:
        name_ = canonical_identifier(name, "name")
        if name_ not in self.names:
            raise KeyError(f"Condition {name_!r} is not declared by this specification.")
        return self.names.index(name_)

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON record of every field binding and the specification identity."""
        return {
            "fields": [field.to_dict() for field in self.fields],
            "spec_id": self.spec_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], /) -> AtomisticConditioningSpec:
        """Rebuild a specification through its constructors in canonical order."""
        if not isinstance(payload, Mapping) or set(payload) != {"fields", "spec_id"}:
            raise ValueError("Conditioning specification record is not canonical.")
        records = payload["fields"]
        if not isinstance(records, list):
            raise TypeError("Conditioning specification fields must be a list.")
        fields = tuple(AtomisticConditionField.from_dict(record) for record in records)
        spec = cls(fields)
        if (
            spec.spec_id != payload["spec_id"]
            or tuple(field.name for field in fields) != spec.names
        ):
            raise ValueError(
                "Conditioning specification record does not match its content."
            )
        return spec


def _convert_condition(
    value: ConvertibleToArray, field: AtomisticConditionField, /
) -> Array:
    if isinstance(value, Array):
        source = value
    else:
        source = as_host_array(value, HostShaped[AnyShape], field.name)
    casting: Casting
    match field.kind:
        case AtomisticConditionKind.CONTINUOUS:
            allowed = source.dtype.kind in "iuf"
            casting = "same_kind"
        case AtomisticConditionKind.CATEGORICAL:
            allowed = source.dtype.kind in "iu"
            casting = "safe"
        case AtomisticConditionKind.BOOLEAN:
            allowed = source.dtype.kind == "b"
            casting = "safe"
        case _ as unreachable:
            assert_never(unreachable)
    if not allowed:
        raise TypeError(f"Condition {field.name!r} has the wrong numerical kind.")
    return as_array(source, _condition_form(field.dtype), field.name, casting=casting)


def _case_condition_validity(
    valid: Array,
    field: AtomisticConditionField,
    atom_mask: Array,
    case_count: int,
    atom_capacity: int,
    /,
) -> Array:
    match field.level:
        case AtomisticConditionLevel.CASE:
            return jnp.all(valid.reshape((case_count, -1)), axis=1)
        case AtomisticConditionLevel.ATOM:
            node_valid = jnp.all(valid.reshape((case_count * atom_capacity, -1)), axis=1)
            return jnp.all(
                jnp.where(atom_mask, node_valid, True).reshape(
                    (case_count, atom_capacity)
                ),
                axis=1,
            )
        case _ as unreachable:
            assert_never(unreachable)


@final
class AtomisticConditioning(StrictModule, NonTrainableState):
    """Dynamic condition values, with explicit default-use and case validity.

    Construction checks kinds, shapes, bindings and safe integer conversion.
    Numerical domain failures remain device-visible through ``case_status``.
    Defaults are supplied explicitly by the caller and broadcast from one event;
    an absent optional field stays absent, never a plausible numerical zero.
    """

    __strict_contract__ = True

    spec: AtomisticConditioningSpec
    values: tuple[Array | None, ...]
    defaults_used: Bool[AnyShape]
    case_count: Size[AtomisticCaseDim] = eqx.field(static=True)
    atom_capacity: Size[AtomisticAtomDim] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        spec: AtomisticConditioningSpec,
        values: Mapping[str, ConvertibleToArray],
        /,
        *,
        case_count: int,
        atom_capacity: int,
        defaults: Mapping[str, ConvertibleToArray] | None = None,
    ) -> None:
        cases = positive_integer(case_count, "case_count")
        atoms = positive_integer(atom_capacity, "atom_capacity")
        default_values = {} if defaults is None else defaults
        unknown = (set(values) | set(default_values)).difference(spec.names)
        if unknown:
            raise ValueError(f"Undeclared conditioning fields: {sorted(unknown)!r}.")
        arrays: list[Array | None] = []
        default_flags: list[bool] = []
        for field in spec.fields:
            count = (
                cases if field.level is AtomisticConditionLevel.CASE else cases * atoms
            )
            shape = (count, *field.port.event_shape)
            used_default = field.name not in values and field.name in default_values
            if field.name in values:
                value = _convert_condition(values[field.name], field)
                if value.shape != shape:
                    raise ValueError(f"Condition {field.name!r} must have shape {shape}.")
            elif used_default:
                default = _convert_condition(default_values[field.name], field)
                if default.shape not in ((), field.port.event_shape):
                    raise ValueError(
                        f"Default for {field.name!r} must describe one port event."
                    )
                value = jnp.broadcast_to(default, shape)
            elif field.required:
                raise ValueError(f"Required condition {field.name!r} is missing.")
            else:
                value = None
            arrays.append(value)
            default_flags.append(used_default)
        flags = jnp.broadcast_to(
            jnp.asarray(default_flags, dtype=jnp.bool_)[None, :],
            (cases, len(spec.fields)),
        )
        self.spec = spec
        self.values = tuple(arrays)
        self.defaults_used = flags
        self.case_count = cases
        self.atom_capacity = atoms

    def value(self, name: str, /) -> Array:
        """Read an admitted field; absence is never interpreted as a default."""
        value = self.values[self.spec.index(name)]
        if value is None:
            raise ValueError(f"Optional condition {name!r} is unavailable.")
        return value

    def present(self, name: str, /) -> bool:
        return self.values[self.spec.index(name)] is not None

    @checked
    def case_status(
        self, atom_mask: Bool[AtomisticNodeDim], /
    ) -> Int32[AtomisticCaseDim]:
        if atom_mask.shape != (self.case_count * self.atom_capacity,):
            raise ValueError(
                "Conditioning atom mask does not match its case/atom layout."
            )
        status = jnp.zeros((self.case_count,), dtype=jnp.int32)
        for field, value in zip(self.spec.fields, self.values, strict=True):
            if value is None:
                continue
            finite = _case_condition_validity(
                jnp.isfinite(value), field, atom_mask, self.case_count, self.atom_capacity
            )
            status = status | jnp.where(
                finite, 0, int(AtomisticConditionStatus.NONFINITE)
            ).astype(jnp.int32)
            bounded = jnp.ones(value.shape, dtype=jnp.bool_)
            if field.minimum is not None:
                bounded = bounded & (
                    value >= jnp.asarray(field.minimum, dtype=value.dtype)
                )
            if field.maximum is not None:
                bounded = bounded & (
                    value <= jnp.asarray(field.maximum, dtype=value.dtype)
                )
            in_range = _case_condition_validity(
                bounded, field, atom_mask, self.case_count, self.atom_capacity
            )
            status = status | jnp.where(
                in_range, 0, int(AtomisticConditionStatus.OUT_OF_RANGE)
            ).astype(jnp.int32)
            if field.categories:
                categories = jnp.asarray(field.categories, dtype=value.dtype)
                index = jnp.searchsorted(categories, value)
                safe = jnp.minimum(index, categories.size - 1)
                member = (index < categories.size) & (categories[safe] == value)
                admitted = _case_condition_validity(
                    member, field, atom_mask, self.case_count, self.atom_capacity
                )
                status = status | jnp.where(
                    admitted, 0, int(AtomisticConditionStatus.INVALID_CATEGORY)
                ).astype(jnp.int32)
        return status

    def to_record(self) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
        """Host record and bound-field arrays that rebuild exactly this conditioning.

        Each field is bound as an explicit ``value`` (complete rows), a
        ``default`` (its one broadcast event) or ``absent``; the arrays follow
        the non-absent fields in specification order. ``from_record`` replays
        the constructor, so default-use evidence and optional absence are
        reproduced rather than stored as unchecked flags. A value whose layout,
        dtype or default evidence its constructor could not have produced
        refuses.
        """
        flags = np.asarray(self.defaults_used)
        if (
            flags.dtype != np.bool_
            or flags.shape != (self.case_count, len(self.spec.fields))
            or not np.array_equal(flags, np.broadcast_to(flags[:1], flags.shape))
        ):
            raise ValueError("Conditioning default evidence is not one per-field record.")
        bindings: list[str] = []
        arrays: list[np.ndarray] = []
        for index, (field, value) in enumerate(
            zip(self.spec.fields, self.values, strict=True)
        ):
            used_default = bool(flags[0, index])
            if value is None:
                if used_default or field.required:
                    raise ValueError(
                        f"Condition {field.name!r} absence is not canonical."
                    )
                bindings.append("absent")
                continue
            host = np.asarray(value)
            count = (
                self.case_count
                if field.level is AtomisticConditionLevel.CASE
                else self.case_count * self.atom_capacity
            )
            if host.dtype != np.dtype(field.dtype.value) or host.shape != (
                count,
                *field.port.event_shape,
            ):
                raise ValueError(f"Condition {field.name!r} has a non-canonical layout.")
            if used_default:
                event = host[0]
                if not np.array_equal(
                    host,
                    np.broadcast_to(event, host.shape),
                    equal_nan=host.dtype.kind == "f",
                ):
                    raise ValueError(
                        f"Default condition {field.name!r} is not one broadcast event."
                    )
                arrays.append(np.array(event, copy=True))
                bindings.append("default")
            else:
                arrays.append(np.array(host, copy=True))
                bindings.append("value")
        record = {
            "spec": self.spec.to_dict(),
            "case_count": self.case_count,
            "atom_capacity": self.atom_capacity,
            "bindings": bindings,
        }
        return record, tuple(arrays)

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any], arrays: Sequence[np.ndarray], /
    ) -> AtomisticConditioning:
        """Rebuild conditioning from ``to_record`` through its constructors.

        The exact array count, shapes and dtypes are checked against the
        rebuilt specification before any value is converted.
        """
        if not isinstance(record, Mapping) or set(record) != _CONDITIONING_RECORD_FIELDS:
            raise ValueError("Conditioning record must use the canonical fields.")
        spec = AtomisticConditioningSpec.from_dict(record["spec"])
        cases = record["case_count"]
        atoms = record["atom_capacity"]
        bindings = record["bindings"]
        if (
            type(cases) is not int
            or type(atoms) is not int
            or cases < 1
            or atoms < 1
            or not isinstance(bindings, list)
            or len(bindings) != len(spec.fields)
            or any(binding not in _CONDITION_BINDINGS for binding in bindings)
        ):
            raise ValueError("Conditioning record layout or bindings are not canonical.")
        bound = tuple(
            (field, binding)
            for field, binding in zip(spec.fields, bindings, strict=True)
            if binding != "absent"
        )
        if not isinstance(arrays, Sequence) or len(arrays) != len(bound):
            raise ValueError("Conditioning arrays do not match the record inventory.")
        values: dict[str, np.ndarray] = {}
        defaults: dict[str, np.ndarray] = {}
        for (field, binding), array in zip(bound, arrays, strict=True):
            count = (
                cases if field.level is AtomisticConditionLevel.CASE else cases * atoms
            )
            shape = (
                (count, *field.port.event_shape)
                if binding == "value"
                else field.port.event_shape
            )
            if (
                not isinstance(array, np.ndarray)
                or array.shape != shape
                or array.dtype != np.dtype(field.dtype.value)
            ):
                raise ValueError(
                    f"Condition {field.name!r} array differs from its declared layout."
                )
            (values if binding == "value" else defaults)[field.name] = array
        return cls(spec, values, case_count=cases, atom_capacity=atoms, defaults=defaults)


class AtomisticObservable(StrEnum):
    ENERGY = "energy"
    REPORTED_ENERGY = "reported-energy"
    GLOBAL_ENERGY_COMPONENT = "global-energy-component"
    ATOM_ENERGY = "atom-energy"
    FORCES = "forces"
    STRESS = "stress"
    DIRECT_FORCES = "direct-forces"
    DIRECT_STRESS = "direct-stress"
    CHARGES = "charges"
    LATENT_CHARGES = "latent-charges"
    SPIN_CHARGES = "spin-charges"
    CHARGE_RESIDUAL = "charge-residual"
    ATOMIC_C6 = "atomic-c6"
    ATOMIC_POLARIZABILITY = "atomic-polarizability"
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"
    POLARIZABILITY = "polarizability"
    NODE_EMBEDDINGS = "node-embeddings"
    GRAPH_EMBEDDINGS = "graph-embeddings"
    BORN_CHARGES = "born-charges"
    HESSIAN_ACTION = "hessian-action"
    HESSIAN_BLOCKS = "hessian-blocks"


@final
class AtomisticFieldReference(StrictModule, NonTrainableState):
    """A typed dependency on an explicitly named program producer.

    The producer is a program binding key. Quantity identity comes from the
    complete port; the enclosing program binds that key to the actual model
    revision. ``unit`` is the consumer's required numerical unit.
    """

    __strict_contract__ = True

    producer: Identifier = eqx.field(static=True)
    observable: AtomisticObservable = eqx.field(static=True)
    port: ValuePort
    level: AtomisticConditionLevel = eqx.field(static=True)
    unit: UnitDefinition | None
    reference_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        producer: str,
        observable: AtomisticObservable,
        port: ValuePort,
        /,
        *,
        level: AtomisticConditionLevel,
        unit: UnitDefinition | None = None,
    ) -> None:
        producer_ = canonical_identifier(producer, "producer")
        observable_ = parse(observable, AtomisticObservable, "observable")
        level_ = parse(level, AtomisticConditionLevel, "level")
        _validate_port_unit(port, unit)
        identity = canonical_fingerprint(
            {
                "kind": "atomistic-field-reference",
                "producer": producer_,
                "observable": observable_.value,
                "port": port.port_id,
                "level": level_.value,
                "unit": None if unit is None else unit.unit_id,
            }
        )
        self.producer = producer_
        self.observable = observable_
        self.port = port
        self.level = level_
        self.unit = unit
        self.reference_id = identity


@final
class AtomisticObservableRequest(StrictModule, NonTrainableState):
    """Canonical observable set and existing native derivative requests."""

    __strict_contract__ = True

    observables: tuple[AtomisticObservable, ...] = eqx.field(static=True)
    derivatives: tuple[DifferentiationRequest, ...]
    request_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        observables: Sequence[AtomisticObservable | str],
        /,
        *,
        derivatives: Sequence[DifferentiationRequest] = (),
    ) -> None:
        selected = {
            parse(
                AtomisticObservable(value) if isinstance(value, str) else value,
                AtomisticObservable,
                "observable",
            )
            for value in observables
        }
        if not selected:
            raise ValueError("An observable request must not be empty.")
        ordered = tuple(value for value in AtomisticObservable if value in selected)
        requests = tuple(derivatives)
        if any(not isinstance(value, DifferentiationRequest) for value in requests):
            raise TypeError("derivatives must contain DifferentiationRequest values.")
        records = tuple(
            (
                tuple(surface.value for surface in value.surfaces),
                value.order,
                None if value.authority is None else value.authority.value,
            )
            for value in requests
        )
        if len(set(records)) != len(records):
            raise ValueError("Derivative requests must be unique.")
        ordered_pairs = sorted(
            zip(records, requests, strict=True),
            key=lambda pair: (pair[0][0], pair[0][1], pair[0][2] or ""),
        )
        identity = canonical_fingerprint(
            {
                "kind": "atomistic-observable-request",
                "observables": tuple(value.value for value in ordered),
                "derivatives": tuple(record for record, _ in ordered_pairs),
            }
        )
        self.observables = ordered
        self.derivatives = tuple(value for _, value in ordered_pairs)
        self.request_id = identity

    def require_supported(
        self, supported: Sequence[AtomisticObservable], owner: str, /
    ) -> None:
        missing = set(self.observables).difference(supported)
        if missing:
            names = tuple(value.value for value in self.observables if value in missing)
            raise ValueError(f"{owner} does not provide requested observables {names!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON record of the observables, derivative requests and identity."""
        return {
            "observables": [value.value for value in self.observables],
            "derivatives": [
                {
                    "surfaces": [surface.value for surface in value.surfaces],
                    "order": value.order,
                    "authority": None
                    if value.authority is None
                    else value.authority.value,
                }
                for value in self.derivatives
            ],
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], /) -> AtomisticObservableRequest:
        """Rebuild a request through its constructors; any change refuses."""
        if not isinstance(payload, Mapping) or set(payload) != {
            "observables",
            "derivatives",
            "request_id",
        }:
            raise ValueError("Observable request record must use the canonical fields.")
        observables = payload["observables"]
        derivatives = payload["derivatives"]
        if (
            not isinstance(observables, list)
            or any(not isinstance(value, str) for value in observables)
            or not isinstance(derivatives, list)
            or any(
                not isinstance(value, Mapping)
                or set(value) != {"surfaces", "order", "authority"}
                or not isinstance(value["surfaces"], list)
                or any(not isinstance(surface, str) for surface in value["surfaces"])
                or type(value["order"]) is not int
                or not (value["authority"] is None or isinstance(value["authority"], str))
                for value in derivatives
            )
        ):
            raise ValueError("Observable request record is not canonical.")
        request = cls(
            tuple(AtomisticObservable(value) for value in observables),
            derivatives=tuple(
                DifferentiationRequest(
                    tuple(DerivativeSurface(surface) for surface in value["surfaces"]),
                    order=value["order"],
                    authority=(
                        None
                        if value["authority"] is None
                        else ComponentAuthority(value["authority"])
                    ),
                )
                for value in derivatives
            ),
        )
        if request.request_id != payload["request_id"] or request.to_dict() != dict(
            payload
        ):
            raise ValueError("Observable request record does not match its content.")
        return request


@final
class AtomisticObservableValue(StrictModule, NonTrainableState):
    """One numerical property with explicit port and availability evidence."""

    __strict_contract__ = True

    observable: AtomisticObservable = eqx.field(static=True)
    port: ValuePort
    unit: UnitDefinition | None
    level: AtomisticConditionLevel = eqx.field(static=True)
    values: Float[AnyShape]
    available: Bool[AtomisticCaseDim]
    successful: Bool[AtomisticCaseDim]
    case_count: Size[AtomisticCaseDim] = eqx.field(static=True)
    atom_capacity: Size[AtomisticAtomDim] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        observable: AtomisticObservable | str,
        port: ValuePort,
        values: ConvertibleToArray,
        available: ConvertibleToArray,
        /,
        *,
        level: AtomisticConditionLevel | str,
        case_count: int,
        atom_capacity: int,
        unit: UnitDefinition | None = None,
    ) -> None:
        observable_ = parse(
            AtomisticObservable(observable)
            if isinstance(observable, str)
            else observable,
            AtomisticObservable,
            "observable",
        )
        _validate_port_unit(port, unit)
        level_ = parse(
            AtomisticConditionLevel(level) if isinstance(level, str) else level,
            AtomisticConditionLevel,
            "level",
        )
        cases = positive_integer(case_count, "case_count")
        atoms = positive_integer(atom_capacity, "atom_capacity")
        scope = Scope()
        parse(cases, Size[AtomisticCaseDim], "case_count", scope=scope)
        array = as_array(values, Float[AnyShape], "values", scope=scope)
        mask = as_array(available, Bool[AtomisticCaseDim], "available", scope=scope)
        count = cases if level_ is AtomisticConditionLevel.CASE else cases * atoms
        if array.shape != (count, *port.event_shape):
            raise ValueError(
                "Observable values do not match their port and level extents."
            )
        self.observable = observable_
        self.port = port
        self.unit = unit
        self.level = level_
        self.values = array
        self.available = mask
        self.successful = mask & jnp.all(jnp.isfinite(array.reshape((cases, -1))), axis=1)
        self.case_count = cases
        self.atom_capacity = atoms


@final
class AtomisticModelOutputs(StrictModule, ExplicitFreeze):
    """Native scalar-energy model outputs with case-isolated availability.

    Additional fields retain their own scientific ports. ``atom_energy_complete``
    distinguishes a complete local decomposition from a model whose global term
    is included only in ``energy``; no invented atomic shares are substituted.
    Outputs and retained evidence are evaluation snapshots, not parameter
    owners. Embedded trained components are intentionally fixed for parameter
    discovery; numerical leaves remain dynamic for JAX transformations.
    """

    __strict_contract__ = True

    _energy: Float[AtomisticCaseDim] | None
    _atom_energy: Float[AtomisticCaseDim, AtomisticAtomDim] | None
    successful: Bool[AtomisticCaseDim]
    additional: tuple[AtomisticObservableValue, ...]
    atom_energy_complete: bool = eqx.field(static=True)
    evidence: tuple[StrictModule, ...]

    def __init__(
        self,
        energy: ConvertibleToArray | None,
        atom_energy: ConvertibleToArray | None,
        successful: ConvertibleToArray,
        /,
        *,
        additional: Sequence[AtomisticObservableValue] = (),
        atom_energy_complete: bool = True,
        evidence: Sequence[StrictModule] = (),
    ) -> None:
        scope = Scope()
        energies = (
            None
            if energy is None
            else as_array(energy, Float[AtomisticCaseDim], "energy", scope=scope)
        )
        atomic = (
            None
            if atom_energy is None
            else as_array(
                atom_energy,
                Float[AtomisticCaseDim, AtomisticAtomDim],
                "atom_energy",
                scope=scope,
            )
        )
        if (energies is None) != (atomic is None):
            raise ValueError(
                "Energy and scoped atomic contributions must be supplied together."
            )
        success = as_array(successful, Bool[AtomisticCaseDim], "successful", scope=scope)
        fields = tuple(additional)
        if any(not isinstance(value, AtomisticObservableValue) for value in fields):
            raise TypeError("additional must contain AtomisticObservableValue records.")
        if len({value.port.port_id for value in fields}) != len(fields):
            raise ValueError("Additional observable ports must be unique.")
        if any(value.case_count != success.shape[0] for value in fields):
            raise ValueError("Additional observables belong to another case layout.")
        capacities = {value.atom_capacity for value in fields}
        if atomic is not None:
            capacities.add(atomic.shape[1])
        if len(capacities) > 1:
            raise ValueError("Additional observables belong to another atom layout.")
        if energies is None and not fields:
            raise ValueError("A property-only result must contain observable values.")
        if not isinstance(atom_energy_complete, bool):
            raise TypeError("atom_energy_complete must be a boolean.")
        if energies is None and atom_energy_complete:
            raise ValueError(
                "A property-only result has no complete atomic energy decomposition."
            )
        records = tuple(evidence)
        if any(not isinstance(record, StrictModule) for record in records):
            raise TypeError("Model evidence must use native typed records.")
        for field in fields:
            success = success & (~field.available | field.successful)
        if energies is not None and atomic is not None:
            success = (
                success & jnp.isfinite(energies) & jnp.all(jnp.isfinite(atomic), axis=1)
            )
        self._energy = energies
        self._atom_energy = atomic
        self.successful = success
        self.additional = tuple(
            sorted(
                fields,
                key=lambda value: (
                    tuple(AtomisticObservable).index(value.observable),
                    value.port.port_id,
                ),
            )
        )
        self.atom_energy_complete = atom_energy_complete
        self.evidence = records

    @property
    def energy_available(self) -> bool:
        return self._energy is not None

    @property
    def energy(self) -> Array:
        if self._energy is None:
            raise ValueError("This model result has no scalar potential energy.")
        return self._energy

    @property
    def atom_energy(self) -> Array:
        if self._atom_energy is None:
            raise ValueError("This model result has no atomic energy contributions.")
        return self._atom_energy

    @checked
    def observable(
        self,
        observable: AtomisticObservable,
        /,
        *,
        port: ValuePort | None = None,
    ) -> AtomisticObservableValue:
        kind = parse(observable, AtomisticObservable, "observable")
        selected: AtomisticObservableValue | None = None
        for field in self.additional:
            if field.observable is kind and (
                port is None or field.port.port_id == port.port_id
            ):
                if selected is not None:
                    raise ValueError(
                        "Multiple observable fields match; select their ValuePort."
                    )
                selected = field
        if selected is None:
            raise ValueError(f"Observable {kind.value!r} is unavailable.")
        return selected


__all__ = [
    "AtomisticFieldReference",
    "TOTAL_CHARGE_PORT",
    "SPIN_MULTIPLICITY_PORT",
    "AtomisticConditionDType",
    "AtomisticConditionField",
    "AtomisticConditionKind",
    "AtomisticConditionLevel",
    "AtomisticConditionStatus",
    "AtomisticConditioning",
    "AtomisticConditioningSpec",
    "AtomisticModelOutputs",
    "AtomisticObservable",
    "AtomisticObservableRequest",
    "AtomisticObservableValue",
]


for _artifact_type in (
    AtomisticConditionDType,
    AtomisticConditionField,
    AtomisticConditionKind,
    AtomisticConditionLevel,
    AtomisticConditionStatus,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticFieldReference,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
):
    register_artifact_value(
        f"phydrax.atomistic:{_artifact_type.__name__}", _artifact_type
    )
del _artifact_type

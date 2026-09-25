#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scientific value ports and explicit model-to-owner port bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import prod
from typing import Any, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..axes._core import AxisKey
from ..units._dimension import DimensionSignature


PortVariance: TypeAlias = Literal[
    "neutral", "primal", "dual", "covariant", "contravariant"
]

_VARIANCES = frozenset({"neutral", "primal", "dual", "covariant", "contravariant"})
_DIRECTIONS = ("input", "output")
_ASPECTS = ("axes", "dimensions", "frame", "normalization", "space")


def _identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string without surrounding space.")
    return value


def _optional_identifier(value: Any, name: str, /) -> str | None:
    return None if value is None else _identifier(value, name)


def _event_shape(value: Iterable[int], /) -> tuple[int, ...]:
    if isinstance(value, str):
        raise TypeError("event_shape must be a sequence of ints.")
    shape = tuple(value)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        raise TypeError("event_shape entries must be ints.")
    if any(size < 1 for size in shape):
        raise ValueError("event_shape entries must be positive.")
    return shape


def _component_ids(
    values: Iterable[str], shape: tuple[int, ...], semantic_id: str, /
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("component_ids must be a sequence of strings.")
    components = tuple(_identifier(value, "component_ids") for value in values)
    if len(set(components)) != len(components):
        raise ValueError("component_ids must be unique.")
    if len(components) != prod(shape):
        raise ValueError(
            f"Port {semantic_id!r} declares {len(components)} component IDs for "
            f"event shape {shape}."
        )
    return components


def _optional_aligned(
    values: Iterable[Any] | None,
    kind: type,
    count: int,
    name: str,
    unit: str,
    /,
    *,
    distinct: bool,
) -> tuple[Any, ...] | None:
    if values is None:
        return None
    aligned = tuple(values)
    if any(not isinstance(value, kind) for value in aligned):
        raise TypeError(f"{name} must contain {kind.__name__} values.")
    if len(aligned) != count:
        raise ValueError(f"{name} must give one value per {unit}.")
    if distinct and len(set(aligned)) != len(aligned):
        raise ValueError(f"{name} must be distinct.")
    return aligned


class ValuePort(StrictModule, NonTrainableState):
    """Explicit scientific identity of one value a model consumes or produces.

    `component_ids` name every scalar component of one event in row-major order,
    so their count equals the event size. `dimensions`, when declared, give one
    physical dimension per component; `axis_keys`, when declared, give one
    semantic axis per event axis. `space_id`, `frame_id`, and `normalization_id`
    are `None` when undeclared; declare an explicit identifier (for example an
    identity normalization) to state that no transformation applies. `variance`
    records whether the value is a neutral quantity, a primal or dual space
    element, or covariant or contravariant tensor components. `port_id`
    content-addresses every declared field.
    """

    semantic_id: str = eqx.field(static=True)
    space_id: str | None = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    component_ids: tuple[str, ...] = eqx.field(static=True)
    dimensions: tuple[DimensionSignature, ...] | None = eqx.field(static=True)
    representation: str = eqx.field(static=True)
    frame_id: str | None = eqx.field(static=True)
    normalization_id: str | None = eqx.field(static=True)
    axis_keys: tuple[AxisKey, ...] | None = eqx.field(static=True)
    variance: PortVariance = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic_id: str,
        /,
        *,
        event_shape: Iterable[int],
        component_ids: Iterable[str],
        representation: str,
        space_id: str | None = None,
        dimensions: Iterable[DimensionSignature] | None = None,
        frame_id: str | None = None,
        normalization_id: str | None = None,
        axis_keys: Iterable[AxisKey] | None = None,
        variance: PortVariance = "neutral",
    ):
        semantic_id_ = _identifier(semantic_id, "semantic_id")
        shape = _event_shape(event_shape)
        components = _component_ids(component_ids, shape, semantic_id_)
        dimensions_ = _optional_aligned(
            dimensions,
            DimensionSignature,
            len(components),
            "dimensions",
            "component",
            distinct=False,
        )
        axes = _optional_aligned(
            axis_keys, AxisKey, len(shape), "axis_keys", "event axis", distinct=True
        )
        if not isinstance(variance, str):
            raise TypeError("variance must be a string.")
        if variance not in _VARIANCES:
            raise ValueError(f"Unknown port variance {variance!r}.")
        representation_ = _identifier(representation, "representation")
        space = _optional_identifier(space_id, "space_id")
        frame = _optional_identifier(frame_id, "frame_id")
        normalization = _optional_identifier(normalization_id, "normalization_id")
        self.semantic_id = semantic_id_
        self.space_id = space
        self.event_shape = shape
        self.component_ids = components
        self.dimensions = dimensions_
        self.representation = representation_
        self.frame_id = frame
        self.normalization_id = normalization
        self.axis_keys = axes
        self.variance = variance
        self.port_id = canonical_fingerprint(
            {
                "kind": "value-port",
                "semantic_id": semantic_id_,
                "space_id": space,
                "event_shape": list(shape),
                "component_ids": list(components),
                "dimensions": (
                    None
                    if dimensions_ is None
                    else [value.dimension_id for value in dimensions_]
                ),
                "representation": representation_,
                "frame_id": frame,
                "normalization_id": normalization,
                "axis_keys": None if axes is None else [[k.scope, k.name] for k in axes],
                "variance": variance,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible record of this port, with `port_id`."""
        return {
            "semantic_id": self.semantic_id,
            "space_id": self.space_id,
            "event_shape": list(self.event_shape),
            "component_ids": list(self.component_ids),
            "dimensions": (
                None
                if self.dimensions is None
                else [value.to_dict() for value in self.dimensions]
            ),
            "representation": self.representation,
            "frame_id": self.frame_id,
            "normalization_id": self.normalization_id,
            "axis_keys": (
                None
                if self.axis_keys is None
                else [[key.scope, key.name] for key in self.axis_keys]
            ),
            "variance": self.variance,
            "port_id": self.port_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], /) -> ValuePort:
        """Restore a port from `to_dict`, failing closed on any field or ID change."""
        if not isinstance(payload, Mapping):
            raise TypeError("ValuePort payload must be a mapping.")
        expected = {
            "semantic_id",
            "space_id",
            "event_shape",
            "component_ids",
            "dimensions",
            "representation",
            "frame_id",
            "normalization_id",
            "axis_keys",
            "variance",
            "port_id",
        }
        if set(payload) != expected:
            raise ValueError(
                "ValuePort payload must use the canonical fields; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}."
            )
        for name in ("event_shape", "component_ids"):
            if not isinstance(payload[name], list):
                raise TypeError(f"ValuePort payload {name} must be a list.")
        dimensions = payload["dimensions"]
        axis_keys = payload["axis_keys"]
        if dimensions is not None and not isinstance(dimensions, list):
            raise TypeError("ValuePort payload dimensions must be a list or None.")
        if axis_keys is not None and (
            not isinstance(axis_keys, list)
            or any(not isinstance(key, list) or len(key) != 2 for key in axis_keys)
        ):
            raise TypeError(
                "ValuePort payload axis_keys must be [scope, name] pairs or None."
            )
        port = cls(
            payload["semantic_id"],
            event_shape=payload["event_shape"],
            component_ids=payload["component_ids"],
            representation=payload["representation"],
            space_id=payload["space_id"],
            dimensions=(
                None
                if dimensions is None
                else [DimensionSignature.from_dict(value) for value in dimensions]
            ),
            frame_id=payload["frame_id"],
            normalization_id=payload["normalization_id"],
            axis_keys=(
                None
                if axis_keys is None
                else [AxisKey(scope, name) for scope, name in axis_keys]
            ),
            variance=payload["variance"],
        )
        if payload["port_id"] != port.port_id:
            raise ValueError("ValuePort payload port_id does not match its content.")
        return port


def _port_tuple(ports: Iterable[ValuePort], name: str, /) -> tuple[ValuePort, ...]:
    values = tuple(ports)
    if any(not isinstance(port, ValuePort) for port in values):
        raise TypeError(f"{name} must contain ValuePort values.")
    identifiers = tuple(port.port_id for port in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must not repeat a port.")
    return values


class ModelPorts(StrictModule, NonTrainableState):
    """Ordered input and output ports of one model or owner slot.

    Port order is semantic: it is the order in which values are supplied and
    produced. `ports_id` content-addresses both ordered port lists.
    """

    inputs: tuple[ValuePort, ...]
    outputs: tuple[ValuePort, ...]
    ports_id: str = eqx.field(static=True)

    def __init__(self, *, inputs: Iterable[ValuePort], outputs: Iterable[ValuePort]):
        inputs_ = _port_tuple(inputs, "inputs")
        outputs_ = _port_tuple(outputs, "outputs")
        self.inputs = inputs_
        self.outputs = outputs_
        self.ports_id = canonical_fingerprint(
            {
                "kind": "model-ports",
                "inputs": [port.port_id for port in inputs_],
                "outputs": [port.port_id for port in outputs_],
            }
        )


def _pairs(
    values: Iterable[tuple[str, str]], name: str, /
) -> tuple[tuple[str, str], ...]:
    pairs = []
    for pair in values:
        if not isinstance(pair, tuple):
            raise TypeError(f"{name} entries must be (model_port_id, owner_port_id).")
        if len(pair) != 2:
            raise ValueError(f"{name} entries must be (model_port_id, owner_port_id).")
        pairs.append(
            (
                _identifier(pair[0], f"{name} model port ID"),
                _identifier(pair[1], f"{name} owner port ID"),
            )
        )
    for position, side in ((0, "model"), (1, "owner")):
        identifiers = tuple(pair[position] for pair in pairs)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name} binds a {side} port more than once.")
    return tuple(pairs)


class PortMapping(StrictModule, NonTrainableState):
    """Explicit `(model_port_id, owner_port_id)` bindings for inputs and outputs.

    Ports are never matched by name, shape, or order. Each model port and each
    owner port appears at most once per direction; entries are stored sorted.
    """

    inputs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    outputs: tuple[tuple[str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        inputs: Iterable[tuple[str, str]] = (),
        outputs: Iterable[tuple[str, str]] = (),
    ):
        inputs_ = tuple(sorted(_pairs(inputs, "inputs")))
        outputs_ = tuple(sorted(_pairs(outputs, "outputs")))
        self.inputs = inputs_
        self.outputs = outputs_


class PortBindingEvidence(StrictModule, NonTrainableState):
    """Audited result of binding model ports to owner ports.

    `inputs` and `outputs` hold `(model_port_id, owner_port_id)` pairs in model
    port order. `unverified` holds sorted `(direction, model_port_id, aspect)`
    records for every aspect (`"axes"`, `"dimensions"`, `"frame"`,
    `"normalization"`, `"space"`) left unverified because a side did not declare
    it; declared mismatches never produce evidence.
    """

    inputs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    outputs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    unverified: tuple[tuple[str, str, str], ...] = eqx.field(static=True)
    binding_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        inputs: Iterable[tuple[str, str]],
        outputs: Iterable[tuple[str, str]],
        unverified: Iterable[tuple[str, str, str]],
    ):
        inputs_ = _pairs(inputs, "inputs")
        outputs_ = _pairs(outputs, "outputs")
        bound = {
            "input": {model_id for model_id, _ in inputs_},
            "output": {model_id for model_id, _ in outputs_},
        }
        checked = set()
        for record in unverified:
            if not isinstance(record, tuple):
                raise TypeError("unverified records must be tuples.")
            if (
                len(record) != 3
                or record[0] not in _DIRECTIONS
                or record[2] not in _ASPECTS
            ):
                raise ValueError(
                    "unverified records must be (direction, model_port_id, aspect)."
                )
            if record[1] not in bound[record[0]]:
                raise ValueError("unverified records must name a bound model port.")
            checked.add(record)
        records = tuple(sorted(checked))
        self.inputs = inputs_
        self.outputs = outputs_
        self.unverified = records
        self.binding_fingerprint = canonical_fingerprint(
            {
                "kind": "port-binding",
                "inputs": [list(pair) for pair in inputs_],
                "outputs": [list(pair) for pair in outputs_],
                "unverified": [list(record) for record in records],
            }
        )

    def _verified(self, aspect: str, /) -> bool:
        return all(record[2] != aspect for record in self.unverified)

    @property
    def dimensions_verified(self) -> bool:
        """Whether both sides declared, and agreed on, every component dimension."""
        return self._verified("dimensions")

    @property
    def axes_verified(self) -> bool:
        """Whether both sides declared, and agreed on, every semantic event axis."""
        return self._verified("axes")

    @property
    def normalizations_verified(self) -> bool:
        """Whether both sides declared, and agreed on, every normalization."""
        return self._verified("normalization")

    @property
    def frames_verified(self) -> bool:
        """Whether both sides declared, and agreed on, every frame."""
        return self._verified("frame")

    @property
    def spaces_verified(self) -> bool:
        """Whether both sides declared, and agreed on, every support space."""
        return self._verified("space")


def _check_pair(
    direction: str, model: ValuePort, owner: ValuePort, /
) -> tuple[tuple[str, str, str], ...]:
    label = (
        f"{direction} port pair {model.semantic_id!r} "
        f"({model.port_id} -> {owner.port_id})"
    )
    for aspect, left, right in (
        ("semantic_id", model.semantic_id, owner.semantic_id),
        ("component_ids", model.component_ids, owner.component_ids),
        ("event_shape", model.event_shape, owner.event_shape),
        ("representation", model.representation, owner.representation),
        ("variance", model.variance, owner.variance),
    ):
        if left != right:
            raise ValueError(f"{label} {aspect} mismatch: {left!r} != {right!r}.")
    unverified = []
    for aspect, left, right in (
        ("axes", model.axis_keys, owner.axis_keys),
        ("dimensions", model.dimensions, owner.dimensions),
        ("frame", model.frame_id, owner.frame_id),
        ("normalization", model.normalization_id, owner.normalization_id),
        ("space", model.space_id, owner.space_id),
    ):
        if left is None or right is None:
            unverified.append((direction, model.port_id, aspect))
        elif left != right:
            raise ValueError(f"{label} {aspect} mismatch: {left!r} != {right!r}.")
    return tuple(unverified)


def _bind_direction(
    direction: str,
    model_ports: tuple[ValuePort, ...],
    owner_ports: tuple[ValuePort, ...],
    pairs: tuple[tuple[str, str], ...],
    /,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, str], ...]]:
    models = {port.port_id: port for port in model_ports}
    owners = {port.port_id: port for port in owner_ports}
    mapped = dict(pairs)
    unknown_models = sorted(set(mapped) - set(models))
    if unknown_models:
        raise ValueError(
            f"Port mapping names unknown model {direction} ports {unknown_models}."
        )
    unknown_owners = sorted(set(mapped.values()) - set(owners))
    if unknown_owners:
        raise ValueError(
            f"Port mapping names unknown owner {direction} ports {unknown_owners}."
        )
    missing = [port.semantic_id for port in model_ports if port.port_id not in mapped]
    if missing:
        raise ValueError(
            f"Port mapping leaves model {direction} ports {missing} unbound."
        )
    bound = tuple((port.port_id, mapped[port.port_id]) for port in model_ports)
    unverified = tuple(
        record
        for model_id, owner_id in bound
        for record in _check_pair(direction, models[model_id], owners[owner_id])
    )
    return bound, unverified


def resolve_port_mapping(
    model_ports: ModelPorts,
    owner_ports: ModelPorts,
    mapping: PortMapping,
    /,
) -> PortBindingEvidence:
    """Bind every model port to the owner port named by an explicit mapping.

    Every model input and output must be mapped; owner ports may stay unused.
    Semantic ID, component IDs, event shape, representation, and variance must
    match exactly. Dimensions, semantic axes, frame, normalization, and space
    must match when both sides declare them and are recorded as unverified
    otherwise. Violations raise `ValueError` naming the port pair.
    """
    if not isinstance(model_ports, ModelPorts) or not isinstance(owner_ports, ModelPorts):
        raise TypeError("model_ports and owner_ports must be ModelPorts.")
    if not isinstance(mapping, PortMapping):
        raise TypeError("mapping must be a PortMapping.")
    inputs, input_records = _bind_direction(
        "input", model_ports.inputs, owner_ports.inputs, mapping.inputs
    )
    outputs, output_records = _bind_direction(
        "output", model_ports.outputs, owner_ports.outputs, mapping.outputs
    )
    return PortBindingEvidence(
        inputs=inputs, outputs=outputs, unverified=input_records + output_records
    )


@runtime_checkable
class PortProvider(Protocol):
    """Model declaring intrinsic scientific ports."""

    def model_ports(self) -> ModelPorts:
        """Return the model's ordered input and output ports."""
        ...


def intrinsic_model_ports(model: Any, /) -> ModelPorts | None:
    """Return a model's intrinsic ports, or `None` when it declares none.

    A `FrozenModel` declares exactly the ports of the model it freezes; any
    other `PortProvider` returns `model_ports()`.
    """
    from ._frozen import FrozenModel

    while isinstance(model, FrozenModel):
        model = model.as_trainable()
    if not isinstance(model, PortProvider):
        return None
    ports = model.model_ports()
    if not isinstance(ports, ModelPorts):
        raise TypeError(f"{type(model).__name__}.model_ports() must return ModelPorts.")
    return ports


def bind_model_ports(
    model: Any,
    owner_ports: ModelPorts,
    mapping: PortMapping | None,
    /,
    *,
    site: str,
) -> PortBindingEvidence | None:
    """Bind a model's intrinsic ports to an owner's ports at one binding site.

    A model with intrinsic ports (`intrinsic_model_ports`) requires an explicit
    `PortMapping` and returns the `PortBindingEvidence` of
    `resolve_port_mapping`; a model without intrinsic ports takes no mapping and
    returns `None`. Violations raise `ValueError` naming `site`.
    """
    if mapping is not None and not isinstance(mapping, PortMapping):
        raise TypeError(f"{site} port_mapping must be a PortMapping or None.")
    model_ports = intrinsic_model_ports(model)
    if model_ports is None:
        if mapping is not None:
            raise ValueError(
                f"{site} received a port_mapping, but {type(model).__name__} "
                "declares no model ports."
            )
        return None
    if mapping is None:
        raise ValueError(
            f"{site} requires an explicit port_mapping: {type(model).__name__} "
            "declares model ports."
        )
    return resolve_port_mapping(model_ports, owner_ports, mapping)


def require_mapped_order(
    evidence: PortBindingEvidence,
    direction: Literal["input", "output"],
    expected: tuple[str, ...],
    /,
    *,
    site: str,
) -> None:
    """Require owner ports bound in model port order to equal `expected` owner port IDs.

    Binding sites that pass owner values positionally never repack them, so the
    owner ports mapped from the model's ordered ports must be exactly the owner's
    positional order. Raises `ValueError` otherwise.
    """
    pairs = evidence.inputs if direction == "input" else evidence.outputs
    mapped = tuple(owner for _, owner in pairs)
    if mapped != tuple(expected):
        raise ValueError(
            f"{site} passes owner {direction} ports in order {tuple(expected)}, but "
            f"the port mapping binds the model's ordered {direction} ports to "
            f"{mapped}; owner values are never repacked."
        )


def require_port_shapes(
    ports: ModelPorts | None,
    /,
    *,
    inputs: tuple[tuple[int, ...], ...],
    outputs: tuple[tuple[int, ...], ...],
    site: str,
) -> ModelPorts | None:
    """Require owner `ports` to declare the owner's value layout; return them.

    An owner that declares its ports explicitly must declare exactly the event
    shapes of the values it passes to (`inputs`) and reads from (`outputs`) its
    model, in owner order. `None` (no owner ports) is returned unchanged.
    Raises `TypeError` for a non-`ModelPorts` value and `ValueError` naming
    `site` for any other layout.
    """
    if ports is None:
        return None
    if not isinstance(ports, ModelPorts):
        raise TypeError(f"{site} ports must be ModelPorts or None.")
    for direction, values, expected in (
        ("input", ports.inputs, inputs),
        ("output", ports.outputs, outputs),
    ):
        declared = tuple(port.event_shape for port in values)
        if declared != tuple(expected):
            raise ValueError(
                f"{site} {direction} ports must declare the event shapes "
                f"{tuple(expected)} of its {direction} values in owner order; got "
                f"{declared}."
            )
    return ports


__all__ = [
    "bind_model_ports",
    "intrinsic_model_ports",
    "ModelPorts",
    "PortBindingEvidence",
    "PortMapping",
    "PortProvider",
    "PortVariance",
    "require_mapped_order",
    "require_port_shapes",
    "ValuePort",
    "resolve_port_mapping",
]

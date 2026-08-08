#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from typing import Any, cast, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._frozendict import frozendict
from ..._strict import StrictModule
from ...graph._operator_topology import (
    broadcast_operator_topology,
    operator_topology_fingerprint,
    OperatorTopology,
    pad_operator_topology,
    slice_operator_topology,
    stack_operator_topologies,
)


@dataclass(frozen=True)
class OperatorCaseProvenance:
    """Leakage-relevant identities and ordering coordinates for one case."""

    case_id: str
    identities: Mapping[str, str] = dataclass_field(default_factory=dict)
    order: Mapping[str, float] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        case_id = str(self.case_id)
        if not case_id:
            raise ValueError("Operator case IDs must be non-empty.")
        identities = frozendict(
            {str(name): str(value) for name, value in dict(self.identities).items()}
        )
        if any(not name or not value for name, value in identities.items()):
            raise ValueError("Provenance identity names and values must be non-empty.")
        order = frozendict(
            {str(name): float(value) for name, value in dict(self.order).items()}
        )
        if any(not name or not np.isfinite(value) for name, value in order.items()):
            raise ValueError("Provenance order coordinates must be named and finite.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "order", order)


OperatorBasis = Literal[
    "uniform",
    "fourier",
    "sine",
    "cosine",
    "legendre",
    "nested",
    "point_cloud",
    "sphere",
]


class OperatorAxis(StrictModule):
    """One named coordinate axis with its integration and basis metadata."""

    name: str
    nodes: Array
    quadrature_weights: Array | None
    basis: OperatorBasis
    periodic: bool

    def __init__(
        self,
        name: str,
        nodes: Array,
        /,
        *,
        quadrature_weights: Array | None = None,
        basis: OperatorBasis = "uniform",
        periodic: bool = False,
    ):
        nodes_ = jnp.asarray(nodes, dtype=float).reshape((-1,))
        if int(nodes_.size) == 0:
            raise ValueError("OperatorAxis nodes must be non-empty.")
        if quadrature_weights is None:
            weights = None
        else:
            weights = jnp.asarray(quadrature_weights, dtype=float).reshape((-1,))
            if weights.shape != nodes_.shape:
                raise ValueError(
                    "OperatorAxis quadrature weights must match the node shape."
                )
        self.name = str(name)
        self.nodes = nodes_
        self.quadrature_weights = weights
        self.basis = basis
        self.periodic = bool(periodic)

    @property
    def size(self) -> int:
        return int(self.nodes.shape[0])

    @classmethod
    def from_discretization(
        cls,
        name: str,
        discretization: Any,
        /,
    ) -> "OperatorAxis":
        return cls(
            name,
            discretization.nodes,
            quadrature_weights=discretization.quad_weights,
            basis=discretization.basis,
            periodic=discretization.periodic,
        )


class FunctionSamples(StrictModule):
    """A discretized function or query set used by a neural operator.

    Tensor-product grids use shared ``axes`` and leave ``coordinates`` as
    ``None``. Point clouds use coordinates with shape
    ``case_shape + (num_points, coord_dim)``; a rank-2 coordinate array is
    shared by every case. Quadrature weights and masks may likewise have either
    ``sample_shape`` or ``case_shape + sample_shape``. Optional native topology
    aligns these sample sites with nodes of a canonical graph.
    """

    values: Array | None
    axes: tuple[OperatorAxis, ...]
    coordinates: Array | None
    quadrature_weights: Array | None
    mask: Array | None
    topology: OperatorTopology | None

    def __init__(
        self,
        *,
        values: Array | None,
        axes: Sequence[OperatorAxis] = (),
        coordinates: Array | None = None,
        quadrature_weights: Array | None = None,
        mask: Array | None = None,
        topology: OperatorTopology | None = None,
    ):
        axes_ = tuple(axes)
        if len({axis.name for axis in axes_}) != len(axes_):
            raise ValueError("FunctionSamples axis names must be unique.")
        if axes_ and coordinates is not None:
            raise ValueError(
                "FunctionSamples accepts tensor-product axes or point coordinates, not both."
            )
        if not axes_ and coordinates is None and values is None:
            raise ValueError("FunctionSamples requires values or coordinates.")
        if isinstance(values, Mapping):
            raise TypeError(
                "FunctionSamples values must be one array or None; "
                "use named OperatorBatch inputs for multiple fields."
            )
        values_ = None if values is None else jnp.asarray(values)

        coordinates_: Array | None
        sample_shape: tuple[int, ...]
        geometry_case_shapes: list[tuple[int, ...]] = []
        if coordinates is None:
            coordinates_ = None
            sample_shape = tuple(axis.size for axis in axes_)
        else:
            coordinates_ = jnp.asarray(coordinates, dtype=float)
            if coordinates_.ndim < 2:
                raise ValueError(
                    "Point-cloud coordinates must have shape "
                    "case_shape + (num_points, coord_dim)."
                )
            if int(coordinates_.shape[-2]) <= 0:
                raise ValueError("Point clouds must contain at least one padded point.")
            if int(coordinates_.shape[-1]) <= 0:
                raise ValueError("Point-cloud coordinate dimension must be positive.")
            sample_shape = (int(coordinates_.shape[-2]),)
            geometry_case_shapes.append(
                tuple(int(size) for size in coordinates_.shape[:-2])
            )

        def prepare_geometry_array(
            value: Array | None,
            *,
            dtype: Any,
            name: str,
        ) -> Array | None:
            if value is None:
                return None
            if not sample_shape:
                raise ValueError(f"{name} requires a non-empty sample geometry.")
            array = jnp.asarray(value, dtype=dtype)
            sample_ndim = len(sample_shape)
            if (
                array.ndim < sample_ndim
                or tuple(int(size) for size in array.shape[-sample_ndim:]) != sample_shape
            ):
                raise ValueError(
                    f"FunctionSamples {name} must end in sample shape {sample_shape}; "
                    f"got {array.shape}."
                )
            geometry_case_shapes.append(
                tuple(int(size) for size in array.shape[:-sample_ndim])
            )
            return array

        weights_ = prepare_geometry_array(
            quadrature_weights,
            dtype=float,
            name="quadrature weights",
        )
        mask_ = prepare_geometry_array(mask, dtype=bool, name="mask")
        if topology is not None:
            if not isinstance(topology, OperatorTopology):
                raise TypeError("FunctionSamples topology must be an OperatorTopology.")
            if topology.sample_shape != sample_shape:
                raise ValueError(
                    "FunctionSamples topology sample shape must match its geometry; "
                    f"got {topology.sample_shape} and {sample_shape}."
                )
            geometry_case_shapes.append(topology.case_shape)
        nonshared = [shape for shape in geometry_case_shapes if shape]
        if nonshared and any(shape != nonshared[0] for shape in nonshared[1:]):
            raise ValueError(
                "Per-case coordinates, quadrature weights, masks, and topology must "
                f"have one case shape; got {tuple(nonshared)}."
            )

        self.values = values_
        self.axes = axes_
        self.coordinates = coordinates_
        self.quadrature_weights = weights_
        self.mask = mask_
        self.topology = topology

    @property
    def sample_shape(self) -> tuple[int, ...]:
        if self.axes:
            return tuple(axis.size for axis in self.axes)
        if self.coordinates is not None:
            return (int(self.coordinates.shape[-2]),)
        return ()

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    @property
    def geometry_case_shape(self) -> tuple[int, ...]:
        """Return the explicit geometry case shape, or ``()`` when shared."""
        shapes: list[tuple[int, ...]] = []
        if self.coordinates is not None:
            shapes.append(tuple(int(size) for size in self.coordinates.shape[:-2]))
        sample_ndim = len(self.sample_shape)
        for array in (self.quadrature_weights, self.mask):
            if array is not None:
                shapes.append(tuple(int(size) for size in array.shape[:-sample_ndim]))
        if self.topology is not None:
            shapes.append(self.topology.case_shape)
        return next((shape for shape in shapes if shape), ())

    def _target_case_shape(
        self,
        case_shape: Sequence[int] | None,
        /,
    ) -> tuple[int, ...]:
        explicit = self.geometry_case_shape
        target = (
            explicit if case_shape is None else tuple(int(size) for size in case_shape)
        )
        if explicit and explicit != target:
            raise ValueError(
                f"Geometry case shape {explicit} cannot broadcast to requested {target}."
            )
        return target

    def coordinates_array(
        self,
        /,
        *,
        case_shape: Sequence[int] | None = None,
        flatten: bool = False,
    ) -> Array:
        """Return coordinates broadcast over cases, optionally flattening samples."""
        target_cases = self._target_case_shape(case_shape)
        if self.axes:
            grid = jnp.meshgrid(*(axis.nodes for axis in self.axes), indexing="ij")
            coordinates = jnp.stack(grid, axis=-1)
        elif self.coordinates is not None:
            coordinates = self.coordinates
        else:
            raise ValueError("FunctionSamples has no coordinate geometry.")
        target = target_cases + self.sample_shape + (int(coordinates.shape[-1]),)
        coordinates = jnp.broadcast_to(coordinates, target)
        if flatten:
            count = 1
            for size in self.sample_shape:
                count *= int(size)
            return coordinates.reshape(target_cases + (count, target[-1]))
        return coordinates

    @property
    def has_physical_quadrature(self) -> bool:
        """Whether every sample dimension carries an explicit physical measure."""
        if self.axes:
            return all(axis.quadrature_weights is not None for axis in self.axes)
        return self.quadrature_weights is not None

    def geometry_fingerprint(self) -> str:
        """Return a host-side digest of physical geometry and measure metadata."""
        digest = hashlib.sha256()

        def update_array(label: str, value: Array | None) -> None:
            digest.update(label.encode("utf-8"))
            if value is None:
                digest.update(b"none")
                return
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))

        digest.update(repr(self.sample_shape).encode("ascii"))
        for axis in self.axes:
            digest.update(repr((axis.name, axis.basis, axis.periodic)).encode("utf-8"))
            update_array(f"axis:{axis.name}:nodes", axis.nodes)
            update_array(
                f"axis:{axis.name}:quadrature",
                axis.quadrature_weights,
            )
        update_array("coordinates", self.coordinates)
        update_array("quadrature", self.quadrature_weights)
        update_array("mask", self.mask)
        if self.topology is None:
            digest.update(b"topology:none")
        else:
            digest.update(operator_topology_fingerprint(self.topology).encode("ascii"))
        return digest.hexdigest()

    def quadrature(
        self,
        /,
        *,
        case_shape: Sequence[int] | None = None,
    ) -> Array:
        """Return unmasked quadrature weights broadcast over cases."""
        target_cases = self._target_case_shape(case_shape)
        if self.quadrature_weights is not None:
            weights = self.quadrature_weights
        elif self.axes:
            factors = []
            for axis in self.axes:
                factor = (
                    jnp.ones_like(axis.nodes, dtype=float)
                    if axis.quadrature_weights is None
                    else axis.quadrature_weights
                )
                factors.append(factor)
            weights = tensor_product(factors)
        elif self.coordinates is not None:
            weights = jnp.ones(self.sample_shape, dtype=float)
        else:
            weights = jnp.asarray(1.0, dtype=float)
        return jnp.broadcast_to(weights, target_cases + self.sample_shape)

    def mask_array(
        self,
        /,
        *,
        case_shape: Sequence[int] | None = None,
    ) -> Array:
        """Return a Boolean mask broadcast over cases."""
        target_cases = self._target_case_shape(case_shape)
        if self.mask is None:
            mask = jnp.ones(self.sample_shape, dtype=bool)
        else:
            mask = self.mask
        mask = jnp.broadcast_to(mask, target_cases + self.sample_shape)
        if self.topology is not None:
            mapped = jnp.broadcast_to(
                self.topology.sample_entities >= 0,
                target_cases + self.sample_shape,
            )
            mask = mask & mapped
        return mask

    def weights(
        self,
        /,
        *,
        normalized: bool = False,
        case_shape: Sequence[int] | None = None,
    ) -> Array:
        """Return quadrature times mask, normalized independently per case."""
        target_cases = self._target_case_shape(case_shape)
        weights = self.quadrature(case_shape=target_cases)
        weights = weights * self.mask_array(case_shape=target_cases).astype(weights.dtype)
        if normalized:
            sample_ndim = len(self.sample_shape)
            axes = tuple(range(len(target_cases), len(target_cases) + sample_ndim))
            total = jnp.sum(weights, axis=axes, keepdims=True)
            weights = jnp.where(
                total > 0.0,
                weights / total,
                jnp.zeros_like(weights),
            )
        return weights


def _validate_sample_values(
    samples: FunctionSamples,
    /,
    *,
    case_ndim: int,
) -> tuple[tuple[int, ...], ...]:
    if samples.values is None:
        return ()
    sample_shape = samples.sample_shape
    sample_ndim = len(sample_shape)
    array = samples.values
    if array.ndim < case_ndim + sample_ndim:
        raise ValueError(
            "FunctionSamples value rank is smaller than its case and sample rank."
        )
    if (
        sample_shape
        and tuple(int(size) for size in array.shape[case_ndim : case_ndim + sample_ndim])
        != sample_shape
    ):
        raise ValueError(
            "FunctionSamples values do not contain sample shape "
            f"{sample_shape} after {case_ndim} case axes; got {array.shape}."
        )
    return (tuple(int(size) for size in array.shape[:case_ndim]),)


class OperatorBatch(StrictModule):
    """Canonical named source/query representation for neural-operator evaluation."""

    inputs: frozendict[str, FunctionSamples]
    queries: frozendict[str, FunctionSamples]
    case_axes: tuple[str, ...]
    case_shape: tuple[int, ...]

    def __init__(
        self,
        *,
        inputs: Mapping[str, FunctionSamples],
        queries: Mapping[str, FunctionSamples],
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] | None = None,
    ):
        if not inputs:
            raise ValueError("OperatorBatch requires at least one input function.")
        if not queries:
            raise ValueError("OperatorBatch requires at least one query branch.")
        for category, samples_by_name in (
            ("input", inputs),
            ("query", queries),
        ):
            for name, value in samples_by_name.items():
                if not str(name):
                    raise ValueError(f"OperatorBatch {category} names must not be empty.")
                if not isinstance(value, FunctionSamples):
                    raise TypeError(
                        f"OperatorBatch {category} {name!r} must be a FunctionSamples value."
                    )

        axes = tuple(str(axis) for axis in case_axes)
        if len(set(axes)) != len(axes):
            raise ValueError("OperatorBatch case axis names must be unique.")
        case_ndim = len(axes)
        candidates: list[tuple[int, ...]] = []
        if case_shape is not None:
            candidates.append(tuple(int(size) for size in case_shape))
        for samples in (*inputs.values(), *queries.values()):
            geometry_shape = samples.geometry_case_shape
            if geometry_shape:
                candidates.append(geometry_shape)
            candidates.extend(_validate_sample_values(samples, case_ndim=case_ndim))

        if case_ndim == 0:
            if any(candidate for candidate in candidates):
                raise ValueError("Per-case arrays require named OperatorBatch case_axes.")
            resolved_shape: tuple[int, ...] = ()
        else:
            if not candidates:
                raise ValueError(
                    "OperatorBatch case shape cannot be inferred; supply case_shape."
                )
            resolved_shape = candidates[0]
            if len(resolved_shape) != case_ndim:
                raise ValueError(
                    f"case_axes has rank {case_ndim}, but case shape is {resolved_shape}."
                )
            if any(candidate != resolved_shape for candidate in candidates[1:]):
                raise ValueError(
                    "OperatorBatch inputs and queries have inconsistent case shapes: "
                    f"{tuple(candidates)}."
                )
            if any(size <= 0 for size in resolved_shape):
                raise ValueError("OperatorBatch case dimensions must be positive.")

        self.inputs = frozendict({str(name): value for name, value in inputs.items()})
        self.queries = frozendict({str(name): value for name, value in queries.items()})
        self.case_axes = axes
        self.case_shape = resolved_shape

    def input(self, name: str, /) -> FunctionSamples:
        if name not in self.inputs:
            raise KeyError(
                f"Unknown operator input {name!r}; expected one of {tuple(self.inputs)}."
            )
        return self.inputs[name]

    def query(self, name: str, /) -> FunctionSamples:
        if name not in self.queries:
            raise KeyError(
                f"Unknown operator query {name!r}; expected one of {tuple(self.queries)}."
            )
        return self.queries[name]

    def single_query_name(self, /) -> str:
        """Return the only query name or fail instead of silently selecting one."""
        if len(self.queries) != 1:
            raise ValueError(
                "This operator requires exactly one query branch; "
                f"got {tuple(self.queries)}."
            )
        return next(iter(self.queries))

    def require_single_query(self, /) -> FunctionSamples:
        """Return the only query or fail instead of silently selecting one."""
        if len(self.queries) != 1:
            raise ValueError(
                "This operator requires exactly one query branch; "
                f"got {tuple(self.queries)}."
            )
        return next(iter(self.queries.values()))

    def take(self, index: Any, /, *, axis: int | str = 0) -> "OperatorBatch":
        """Index one named case axis while preserving all sample metadata."""
        return slice_operator_batch(self, index, axis=axis)


class OperatorOutputSpec(StrictModule):
    """Explicit scalar or channel-last neural-operator output contract."""

    channels: int | Literal["scalar"]
    component_names: tuple[str, ...]

    def __init__(
        self,
        channels: int | Literal["scalar"] = "scalar",
        /,
        *,
        component_names: Sequence[str] = (),
    ):
        if channels == "scalar":
            count = 1
        else:
            count = int(channels)
            if count <= 0:
                raise ValueError("Operator output channels must be positive.")
        names = tuple(str(name) for name in component_names)
        if names and (channels == "scalar" or len(names) != count):
            raise ValueError(
                "component_names must be empty for scalar output or match channel count."
            )
        if len(set(names)) != len(names):
            raise ValueError("Operator output component names must be unique.")
        self.channels = channels
        self.component_names = names

    @property
    def channel_shape(self) -> tuple[int, ...]:
        return () if self.channels == "scalar" else (int(self.channels),)

    def expected_shape(
        self,
        batch: OperatorBatch,
        /,
        *,
        query_name: str | None = None,
    ) -> tuple[int, ...]:
        query = (
            batch.require_single_query()
            if query_name is None
            else batch.query(query_name)
        )
        return batch.case_shape + query.sample_shape + self.channel_shape

    def validate(
        self,
        values: Array,
        batch: OperatorBatch,
        /,
        *,
        query_name: str | None = None,
    ) -> Array:
        array = jnp.asarray(values)
        expected = self.expected_shape(batch, query_name=query_name)
        if tuple(int(size) for size in array.shape) != expected:
            raise ValueError(
                f"Operator output shape must be {expected}; got {array.shape}."
            )
        return array


class OperatorFieldBatch(StrictModule):
    """Values for one named output field bound to one named query branch."""

    values: Array
    query_name: str
    spec: OperatorOutputSpec

    def __init__(
        self,
        values: Array,
        /,
        *,
        query_name: str,
        spec: OperatorOutputSpec,
    ):
        resolved_query = str(query_name)
        if not resolved_query:
            raise ValueError("Operator output fields require a query name.")
        if not isinstance(spec, OperatorOutputSpec):
            raise TypeError("Operator output field spec must be an OperatorOutputSpec.")
        self.values = jnp.asarray(values)
        self.query_name = resolved_query
        self.spec = spec


class OperatorTargetBatch(StrictModule):
    """Named supervised fields bound to query branches and case axes."""

    fields: frozendict[str, OperatorFieldBatch]
    case_axes: tuple[str, ...]
    case_shape: tuple[int, ...]

    def __init__(
        self,
        fields: Mapping[str, OperatorFieldBatch],
        /,
        *,
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] = (),
    ):
        field_map = frozendict({str(name): field for name, field in fields.items()})
        for name, field in field_map.items():
            if not name:
                raise ValueError("Operator target field names must be non-empty.")
            if not isinstance(field, OperatorFieldBatch):
                raise TypeError(
                    f"Operator target field {name!r} must be an OperatorFieldBatch."
                )
        axes = tuple(str(axis) for axis in case_axes)
        shape = tuple(int(size) for size in case_shape)
        if len(axes) != len(shape):
            raise ValueError("case_axes and case_shape must have equal lengths.")
        if len(set(axes)) != len(axes) or any(not axis for axis in axes):
            raise ValueError("case axis names must be unique and non-empty.")
        if any(size <= 0 for size in shape):
            raise ValueError("case_shape entries must be positive.")
        self.fields = field_map
        self.case_axes = axes
        self.case_shape = shape

    @classmethod
    def from_arrays(
        cls,
        values: Mapping[str, Array],
        batch: OperatorBatch,
        /,
        *,
        query_names: Mapping[str, str] | None = None,
        specs: Mapping[str, OperatorOutputSpec] | None = None,
    ) -> "OperatorTargetBatch":
        """Build and validate named target arrays against an operator batch."""
        if not values:
            return cls(
                {},
                case_axes=batch.case_axes,
                case_shape=batch.case_shape,
            )
        names = tuple(str(name) for name in values)
        if query_names is None:
            query = next(iter(batch.queries)) if len(batch.queries) == 1 else None
            if query is None:
                raise ValueError(
                    "query_names is required when a batch has multiple query branches."
                )
            resolved_queries = {name: query for name in names}
        else:
            if set(query_names) != set(names):
                raise ValueError("query_names must define every target field.")
            resolved_queries = {name: str(query_names[name]) for name in names}
        if specs is not None and set(specs) != set(names):
            raise ValueError("specs must define every target field.")
        fields: dict[str, OperatorFieldBatch] = {}
        for name, value in values.items():
            query_name = resolved_queries[name]
            if query_name not in batch.queries:
                raise KeyError(
                    f"Unknown target query {query_name!r}; "
                    f"expected one of {tuple(batch.queries)}."
                )
            array = jnp.asarray(value)
            if specs is None:
                prefix = batch.case_shape + batch.query(query_name).sample_shape
                if tuple(int(size) for size in array.shape[: len(prefix)]) != prefix:
                    raise ValueError(
                        f"Target field {name!r} must start with shape {prefix}; "
                        f"got {array.shape}."
                    )
                trailing = tuple(int(size) for size in array.shape[len(prefix) :])
                if not trailing:
                    spec = OperatorOutputSpec("scalar")
                elif len(trailing) == 1:
                    spec = OperatorOutputSpec(trailing[0])
                else:
                    raise ValueError(
                        f"Target field {name!r} may have at most one channel axis."
                    )
            else:
                spec = specs[name]
            fields[name] = OperatorFieldBatch(
                array,
                query_name=query_name,
                spec=spec,
            )
        target = cls(
            fields,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )
        target.validate(batch)
        return target

    def field(self, name: str, /) -> OperatorFieldBatch:
        if name not in self.fields:
            raise KeyError(
                f"Unknown operator target field {name!r}; "
                f"expected one of {tuple(self.fields)}."
            )
        return self.fields[name]

    def validate(self, batch: OperatorBatch, /) -> None:
        if self.case_axes != batch.case_axes or self.case_shape != batch.case_shape:
            raise ValueError(
                "Operator target case axes and shape must match the input batch."
            )
        for name, field in self.fields.items():
            if field.query_name not in batch.queries:
                raise KeyError(
                    f"Target field {name!r} references unknown query "
                    f"{field.query_name!r}."
                )
            field.spec.validate(
                field.values,
                batch,
                query_name=field.query_name,
            )

    def take(self, index: Any, /, *, axis: int | str = 0) -> "OperatorTargetBatch":
        if isinstance(axis, str):
            if axis not in self.case_axes:
                raise KeyError(f"Unknown case axis {axis!r}; expected {self.case_axes}.")
            position = self.case_axes.index(axis)
        else:
            position = int(axis)
            if position < 0:
                position += len(self.case_axes)
            if position < 0 or position >= len(self.case_axes):
                raise ValueError("Case axis index is out of range.")
        if isinstance(index, int):
            axes = self.case_axes[:position] + self.case_axes[position + 1 :]
            shape = self.case_shape[:position] + self.case_shape[position + 1 :]
        else:
            if isinstance(index, slice):
                size = len(range(*index.indices(self.case_shape[position])))
            else:
                size = int(jnp.asarray(index).size)
            axes = self.case_axes
            shape_list = list(self.case_shape)
            shape_list[position] = size
            shape = tuple(shape_list)
        return OperatorTargetBatch(
            {
                name: OperatorFieldBatch(
                    _take_array(field.values, index, position),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in self.fields.items()
            },
            case_axes=axes,
            case_shape=shape,
        )

    def map_values(
        self,
        function: Callable[[Array], Array],
        /,
    ) -> "OperatorTargetBatch":
        """Transform every named target array while preserving its contract."""
        return OperatorTargetBatch(
            {
                name: OperatorFieldBatch(
                    function(field.values),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in self.fields.items()
            },
            case_axes=self.case_axes,
            case_shape=self.case_shape,
        )


class OperatorPrediction(StrictModule):
    """Named output fields retaining query and case-axis semantics."""

    fields: frozendict[str, OperatorFieldBatch]
    queries: frozendict[str, FunctionSamples]
    case_axes: tuple[str, ...]
    case_shape: tuple[int, ...]

    def __init__(
        self,
        fields: Mapping[str, OperatorFieldBatch],
        queries: Mapping[str, FunctionSamples],
        /,
        *,
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] | None = None,
    ):
        if not fields:
            raise ValueError("OperatorPrediction requires at least one named field.")
        if not queries:
            raise ValueError("OperatorPrediction requires at least one named query.")
        axes = tuple(str(axis) for axis in case_axes)
        shape = () if case_shape is None else tuple(int(size) for size in case_shape)
        if len(axes) != len(shape):
            raise ValueError("OperatorPrediction case_axes and case_shape ranks differ.")
        query_map = frozendict({str(name): value for name, value in queries.items()})
        for name, query in query_map.items():
            if not name or not isinstance(query, FunctionSamples):
                raise TypeError(
                    "OperatorPrediction queries must map non-empty names to FunctionSamples."
                )
            geometry_shape = query.geometry_case_shape
            if geometry_shape and geometry_shape != shape:
                raise ValueError(
                    f"Prediction query {name!r} has case shape {geometry_shape}; "
                    f"expected {shape}."
                )
        field_map = frozendict({str(name): value for name, value in fields.items()})
        for name, field in field_map.items():
            if not name or not isinstance(field, OperatorFieldBatch):
                raise TypeError(
                    "OperatorPrediction fields must map non-empty names to "
                    "OperatorFieldBatch values."
                )
            if field.query_name not in query_map:
                raise ValueError(
                    f"Output field {name!r} references unknown query "
                    f"{field.query_name!r}."
                )
            expected = (
                shape
                + query_map[field.query_name].sample_shape
                + field.spec.channel_shape
            )
            if tuple(int(size) for size in field.values.shape) != expected:
                raise ValueError(
                    f"Operator output field {name!r} must have shape {expected}; "
                    f"got {field.values.shape}."
                )
        self.fields = field_map
        self.queries = query_map
        self.case_axes = axes
        self.case_shape = shape

    @classmethod
    def from_field(
        cls,
        name: str,
        values: Array,
        query_name: str,
        query: FunctionSamples,
        /,
        *,
        spec: OperatorOutputSpec,
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] | None = None,
    ) -> "OperatorPrediction":
        """Construct the common one-field prediction without weakening its schema."""
        return cls(
            {
                str(name): OperatorFieldBatch(
                    values,
                    query_name=query_name,
                    spec=spec,
                )
            },
            {str(query_name): query},
            case_axes=case_axes,
            case_shape=case_shape,
        )

    def field(self, name: str, /) -> OperatorFieldBatch:
        if name not in self.fields:
            raise KeyError(
                f"Unknown operator output field {name!r}; "
                f"expected one of {tuple(self.fields)}."
            )
        return self.fields[name]

    def query_geometry(self, name: str, /) -> FunctionSamples:
        if name not in self.queries:
            raise KeyError(
                f"Unknown prediction query {name!r}; expected one of {tuple(self.queries)}."
            )
        return self.queries[name]


def tensor_product(factors: Sequence[Array], /) -> Array:
    """Return the outer product of one-dimensional factors."""
    factors_ = tuple(jnp.asarray(factor) for factor in factors)
    if not factors_:
        return jnp.asarray(1.0, dtype=float)
    result = factors_[0]
    for factor in factors_[1:]:
        result = jnp.multiply.outer(result, factor)
    return result


def _pad_axis(array: Array, size: int, axis: int, /, *, value: Any) -> Array:
    width = int(size) - int(array.shape[axis])
    if width < 0:
        raise ValueError(
            f"Cannot pad axis of size {array.shape[axis]} to smaller size {size}."
        )
    padding = [(0, 0)] * array.ndim
    padding[axis] = (0, width)
    return jnp.pad(array, padding, constant_values=value)


def pad_function_samples(
    samples: FunctionSamples,
    size: int,
    /,
    *,
    case_shape: Sequence[int] = (),
) -> FunctionSamples:
    """Pad one point-cloud sample axis and materialize a validity mask."""
    if samples.axes or samples.coordinates is None or len(samples.sample_shape) != 1:
        raise ValueError("Only point-cloud FunctionSamples can be padded.")
    target = int(size)
    current = samples.sample_shape[0]
    if target < current:
        raise ValueError(f"Cannot pad {current} points to {target}.")
    cases = tuple(int(value) for value in case_shape)
    case_ndim = len(cases)
    coordinates = _pad_axis(
        samples.coordinates_array(case_shape=cases),
        target,
        case_ndim,
        value=0.0,
    )
    quadrature = _pad_axis(
        samples.quadrature(case_shape=cases),
        target,
        case_ndim,
        value=0.0,
    )
    mask = _pad_axis(
        samples.mask_array(case_shape=cases),
        target,
        case_ndim,
        value=False,
    )
    if samples.values is None:
        values = None
    else:
        values = _pad_axis(samples.values, target, case_ndim, value=0)
    return FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=quadrature,
        mask=mask,
        topology=(
            None
            if samples.topology is None
            else pad_operator_topology(samples.topology, target)
        ),
    )


def _take_array(array: Array, index: Any, axis: int, /) -> Array:
    if isinstance(index, slice):
        selection = [slice(None)] * array.ndim
        selection[axis] = index
        return array[tuple(selection)]
    if isinstance(index, int):
        return jnp.take(array, index, axis=axis)
    indices = jnp.asarray(index, dtype=jnp.int32)
    if indices.ndim != 1:
        raise ValueError("Case index arrays must be one-dimensional.")
    return jnp.take(array, indices, axis=axis)


def _slice_samples(
    samples: FunctionSamples,
    index: Any,
    axis: int,
    /,
    *,
    case_ndim: int,
) -> FunctionSamples:
    if samples.values is None:
        values = None
    else:
        values = _take_array(samples.values, index, axis)

    def slice_geometry(array: Array | None, sample_ndim: int) -> Array | None:
        if array is None:
            return None
        if array.ndim == sample_ndim:
            return array
        return _take_array(array, index, axis)

    coordinates = slice_geometry(samples.coordinates, 2)
    quadrature = slice_geometry(
        samples.quadrature_weights,
        len(samples.sample_shape),
    )
    mask = slice_geometry(samples.mask, len(samples.sample_shape))
    del case_ndim
    topology = samples.topology
    if topology is not None and topology.case_shape:
        topology = slice_operator_topology(topology, index, axis)
    return FunctionSamples(
        values=values,
        axes=samples.axes,
        coordinates=coordinates,
        quadrature_weights=quadrature,
        mask=mask,
        topology=topology,
    )


def slice_operator_batch(
    batch: OperatorBatch,
    index: Any,
    /,
    *,
    axis: int | str = 0,
) -> OperatorBatch:
    """Index an operator batch along one case axis."""
    if isinstance(axis, str):
        if axis not in batch.case_axes:
            raise KeyError(f"Unknown case axis {axis!r}; expected {batch.case_axes}.")
        position = batch.case_axes.index(axis)
    else:
        position = int(axis)
        if position < 0:
            position += len(batch.case_axes)
        if position < 0 or position >= len(batch.case_axes):
            raise ValueError("Case axis index is out of range.")
    inputs = {
        name: _slice_samples(
            samples,
            index,
            position,
            case_ndim=len(batch.case_axes),
        )
        for name, samples in batch.inputs.items()
    }
    queries = {
        name: _slice_samples(
            samples,
            index,
            position,
            case_ndim=len(batch.case_axes),
        )
        for name, samples in batch.queries.items()
    }
    if isinstance(index, int):
        axes = batch.case_axes[:position] + batch.case_axes[position + 1 :]
        shape = batch.case_shape[:position] + batch.case_shape[position + 1 :]
    else:
        if isinstance(index, slice):
            size = len(range(*index.indices(batch.case_shape[position])))
        else:
            size = int(jnp.asarray(index).size)
        axes = batch.case_axes
        shape_list = list(batch.case_shape)
        shape_list[position] = size
        shape = tuple(shape_list)
    return OperatorBatch(
        inputs=inputs,
        queries=queries,
        case_axes=axes,
        case_shape=shape,
    )


def _axes_match(left: tuple[OperatorAxis, ...], right: tuple[OperatorAxis, ...]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        a.name == b.name
        and a.basis == b.basis
        and a.periodic == b.periodic
        and bool(jnp.array_equal(a.nodes, b.nodes))
        and (
            (a.quadrature_weights is None and b.quadrature_weights is None)
            or (
                a.quadrature_weights is not None
                and b.quadrature_weights is not None
                and bool(jnp.array_equal(a.quadrature_weights, b.quadrature_weights))
            )
        )
        for a, b in zip(left, right, strict=True)
    )


def _stack_samples(
    samples: Sequence[FunctionSamples],
    case_shapes: Sequence[tuple[int, ...]],
    /,
) -> FunctionSamples:
    first = samples[0]
    if any(not _axes_match(first.axes, item.axes) for item in samples[1:]):
        raise ValueError("Tensor-grid axes must match when stacking operator batches.")
    if first.coordinates is None and any(
        item.coordinates is not None for item in samples[1:]
    ):
        raise ValueError("FunctionSamples geometry kinds must match when stacking.")
    if first.coordinates is not None:
        target = max(item.sample_shape[0] for item in samples)
        prepared = tuple(
            pad_function_samples(item, target, case_shape=shape)
            for item, shape in zip(samples, case_shapes, strict=True)
        )
        coordinates = jnp.stack(
            tuple(cast(Array, item.coordinates) for item in prepared),
            axis=0,
        )
        quadrature = jnp.stack(
            tuple(cast(Array, item.quadrature_weights) for item in prepared),
            axis=0,
        )
        mask = jnp.stack(
            tuple(cast(Array, item.mask) for item in prepared),
            axis=0,
        )
    elif first.axes:
        prepared = tuple(samples)
        coordinates = None
        quadrature = jnp.stack(
            tuple(
                item.quadrature(case_shape=shape)
                for item, shape in zip(samples, case_shapes, strict=True)
            ),
            axis=0,
        )
        mask = jnp.stack(
            tuple(
                item.mask_array(case_shape=shape)
                for item, shape in zip(samples, case_shapes, strict=True)
            ),
            axis=0,
        )
    else:
        prepared = tuple(samples)
        coordinates = None
        quadrature = None
        mask = None

    topology_presence = tuple(item.topology is not None for item in prepared)
    if any(topology_presence) and not all(topology_presence):
        raise ValueError("FunctionSamples topology must all be present or all be absent.")
    if all(topology_presence):
        topologies = tuple(
            broadcast_operator_topology(
                cast(OperatorTopology, item.topology),
                case_shape,
            )
            for item, case_shape in zip(prepared, case_shapes, strict=True)
        )
        topology = stack_operator_topologies(topologies)
    else:
        topology = None
    if first.values is None:
        if any(item.values is not None for item in prepared[1:]):
            raise ValueError("FunctionSamples values must all be present or all be None.")
        values = None
    else:
        if any(item.values is None for item in prepared[1:]):
            raise ValueError("FunctionSamples values must all be present or all be None.")
        values = jnp.stack(
            tuple(cast(Array, item.values) for item in prepared),
            axis=0,
        )
    return FunctionSamples(
        values=values,
        axes=first.axes,
        coordinates=coordinates,
        quadrature_weights=quadrature,
        mask=mask,
        topology=topology,
    )


def stack_operator_batches(
    batches: Sequence[OperatorBatch],
    /,
    *,
    case_axis: str = "batch",
) -> OperatorBatch:
    """Stack compatible batches, padding point clouds to one compiled shape."""
    batches_ = tuple(batches)
    if not batches_:
        raise ValueError("stack_operator_batches requires at least one batch.")
    first = batches_[0]
    if case_axis in first.case_axes:
        raise ValueError(f"New case axis {case_axis!r} already exists.")
    if any(batch.case_axes != first.case_axes for batch in batches_[1:]):
        raise ValueError("Existing case axes must match when stacking batches.")
    if any(batch.case_shape != first.case_shape for batch in batches_[1:]):
        raise ValueError("Existing case shapes must match when stacking batches.")
    names = tuple(first.inputs)
    if any(set(batch.inputs) != set(names) for batch in batches_[1:]):
        raise ValueError("Operator input names must match when stacking.")
    query_names = tuple(first.queries)
    if any(set(batch.queries) != set(query_names) for batch in batches_[1:]):
        raise ValueError("Operator query names must match when stacking.")
    case_shapes = tuple(batch.case_shape for batch in batches_)
    inputs = {
        name: _stack_samples(
            tuple(batch.input(name) for batch in batches_),
            case_shapes,
        )
        for name in names
    }
    queries = {
        name: _stack_samples(
            tuple(batch.query(name) for batch in batches_),
            case_shapes,
        )
        for name in query_names
    }
    return OperatorBatch(
        inputs=inputs,
        queries=queries,
        case_axes=(str(case_axis),) + first.case_axes,
        case_shape=(len(batches_),) + first.case_shape,
    )


__all__ = [
    "FunctionSamples",
    "OperatorAxis",
    "OperatorBasis",
    "OperatorBatch",
    "OperatorFieldBatch",
    "OperatorCaseProvenance",
    "OperatorOutputSpec",
    "OperatorPrediction",
    "OperatorTargetBatch",
    "pad_function_samples",
    "slice_operator_batch",
    "stack_operator_batches",
    "tensor_product",
]

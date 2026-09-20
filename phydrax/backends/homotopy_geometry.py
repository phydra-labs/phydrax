#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded data-only positive-dimensional HomotopyContinuation.jl adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from math import isfinite
from operator import index
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._external_runtime import (
    EnergyRunResult,
    EnergyRuntimeError,
    run_energy_command,
)
from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..algebraic._positive_dimensional import (
    AffineSlice,
    MonodromyEvidence,
    PathInventory,
    PathRecord,
    PathStatus,
    PseudoWitnessSet,
    RegenerationEdge,
    RegenerationPlan,
    TraceTestEvidence,
    WitnessSet,
)
from ..algebraic._system import SparsePolynomialSystem
from .homotopy_continuation import HomotopyContinuationProvider


HOMOTOPY_GEOMETRY_PROTOCOL = "phydrax.homotopy-geometry"
_WORKER_PATH = Path(__file__).with_name("_homotopy_geometry_worker.jl")
_WORKER_BYTES = _WORKER_PATH.read_bytes()
HOMOTOPY_GEOMETRY_WORKER_SHA256 = hashlib.sha256(_WORKER_BYTES).hexdigest()
_MAX_REQUEST_BYTES = 32 * 1024 * 1024


class HomotopyGeometryOperation(str, Enum):
    GENERIC_SLICE = "generic-slice"
    WITNESS_TRANSPORT = "witness-transport"
    TRACE_TEST = "trace-test"
    MONODROMY = "monodromy"
    REGENERATION_STAGE = "regeneration-stage"
    IMAGE_DEGREE = "image-degree"
    MEMBERSHIP = "membership"


class HomotopyGeometryStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_PATH_FAILURE = "partial-path-failure"
    TRACE_TEST_FAILED = "trace-test-failed"
    BUDGET_EXHAUSTED = "budget-exhausted"
    PROVIDER_FAILED = "provider-failed"
    INVALID_OUTPUT = "invalid-output"
    IDENTITY_MISMATCH = "identity-mismatch"


def _identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _nonnegative_integer(value: Any, name: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        result = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_integer(value: Any, name: str, /) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _strict_fields(value: Mapping[str, Any], expected: set[str], owner: str, /) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{owner} must use the exact protocol fields; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )


def _mapping(value: Any, name: str, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, name: str, /) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _complex_payload(value: ArrayLike, name: str, ndim: int, /) -> dict[str, Any]:
    array = np.asarray(value)
    if (
        array.ndim != ndim
        or array.dtype.kind not in "fciu"
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite rank-{ndim} real or complex array.")
    stacked = np.stack((array.real, array.imag), axis=-1)
    return {
        "shape": list(array.shape),
        "values": stacked.reshape((-1, 2)).tolist(),
    }


def _real_payload(value: ArrayLike, name: str, ndim: int, /) -> list[Any]:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "fiu" or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rank-{ndim} real array.")
    return array.tolist()


def _complex_array(value: Any, name: str, ndim: int, /) -> np.ndarray:
    record = _mapping(value, name)
    _strict_fields(record, {"shape", "values"}, name)
    shape_values = _sequence(record["shape"], f"{name}.shape")
    shape = tuple(_nonnegative_integer(entry, f"{name} shape") for entry in shape_values)
    pairs = np.asarray(record["values"])
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.float64)
    if (
        len(shape) != ndim
        or pairs.shape != (int(np.prod(shape, dtype=np.int64)), 2)
        or pairs.dtype.kind not in "fiu"
        or np.any(~np.isfinite(pairs))
    ):
        raise ValueError(
            f"{name} must be a finite rank-{ndim} array of [real, imaginary] pairs."
        )
    values = pairs[:, 0].astype("float64") + 1j * pairs[:, 1].astype("float64")
    return values.reshape(shape)


def _real_array(value: Any, name: str, ndim: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "fiu" or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real rank-{ndim} array.")
    return array.astype("float64", copy=False)


def _system_record(system: SparsePolynomialSystem, /) -> dict[str, Any]:
    support = system.support
    coefficients = np.asarray(system.coefficients)
    return {
        "variable_count": support.variable_count,
        "equation_count": support.equation_count,
        "equation_indices": np.asarray(support.equation_indices).tolist(),
        "exponents": np.asarray(support.exponents).tolist(),
        "coefficients": _complex_payload(coefficients, "coefficients", 1),
    }


class HomotopyGeometryPolicy(StrictModule, NonTrainableState):
    """Hard request, loop, stage, process-time, and output-byte bounds."""

    path_capacity: int = eqx.field(static=True)
    loop_capacity: int = eqx.field(static=True)
    stage_capacity: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    timeout_seconds: float = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        path_capacity: int = 10_000,
        loop_capacity: int = 64,
        stage_capacity: int = 64,
        seed: int = 0,
        timeout_seconds: float = 3_600.0,
        maximum_output_bytes: int = 64 * 1024 * 1024,
    ):
        paths = _positive_integer(path_capacity, "path_capacity")
        loops = _positive_integer(loop_capacity, "loop_capacity")
        stages = _positive_integer(stage_capacity, "stage_capacity")
        seed_ = _nonnegative_integer(seed, "seed")
        timeout = float(timeout_seconds)
        output_bytes = _positive_integer(maximum_output_bytes, "maximum_output_bytes")
        if seed_ > 2**32 - 1:
            raise ValueError("seed must fit an unsigned 32-bit integer.")
        if not isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive.")
        self.path_capacity = paths
        self.loop_capacity = loops
        self.stage_capacity = stages
        self.seed = seed_
        self.timeout_seconds = timeout
        self.maximum_output_bytes = output_bytes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "homotopy-geometry-policy",
                "paths": paths,
                "loops": loops,
                "stages": stages,
                "seed": seed_,
                "timeout_seconds": timeout,
                "maximum_output_bytes": output_bytes,
            }
        )


class HomotopyGeometryPathRequest(StrictModule, NonTrainableState):
    """Stable path identity, batch identity, and source endpoint index."""

    path_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)
    source_index: int = eqx.field(static=True)

    def __init__(self, path_id: str, batch_id: str, source_index: int, /):
        self.path_id = _identifier(path_id, "path_id")
        self.batch_id = _identifier(batch_id, "batch_id")
        self.source_index = _nonnegative_integer(source_index, "source_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "batch_id": self.batch_id,
            "source_index": self.source_index,
        }


def _payload_fields(operation: HomotopyGeometryOperation) -> set[str]:
    if operation is HomotopyGeometryOperation.GENERIC_SLICE:
        return {"dimension", "slice_matrix", "slice_offset"}
    if operation is HomotopyGeometryOperation.WITNESS_TRANSPORT:
        return {
            "witness_set_id",
            "dimension",
            "source_slice_matrix",
            "source_slice_offset",
            "slice_matrix",
            "slice_offset",
            "source_points",
        }
    if operation is HomotopyGeometryOperation.TRACE_TEST:
        return {
            "witness_set_id",
            "point_indices",
            "source_slice_matrix",
            "source_slice_offset",
            "source_points",
            "sample_parameters",
            "sample_offsets",
            "tolerance",
        }
    if operation is HomotopyGeometryOperation.MONODROMY:
        return {
            "witness_set_id",
            "slice_matrix",
            "slice_offset",
            "source_points",
            "loops",
        }
    if operation is HomotopyGeometryOperation.REGENERATION_STAGE:
        return {
            "plan_id",
            "edge_id",
            "stage_count",
            "dimension",
            "equation_indices",
            "slice_matrix",
            "slice_offset",
        }
    if operation is HomotopyGeometryOperation.IMAGE_DEGREE:
        return {
            "source_system_id",
            "map_id",
            "source_dimension",
            "image_dimension",
            "map_equation_count",
            "map_equation_indices",
            "map_exponents",
            "map_coefficients",
            "source_slice_matrix",
            "source_slice_offset",
            "image_slice_matrix",
            "image_slice_offset",
        }
    return {"witness_sets", "query_points", "tolerance"}


def _validate_slice_payload(
    payload: Mapping[str, Any],
    matrix_name: str,
    offset_name: str,
    ambient_dimension: int,
    codimension: int | None,
    /,
) -> AffineSlice:
    slice_ = AffineSlice(
        _complex_array(payload[matrix_name], matrix_name, 2),
        _complex_array(payload[offset_name], offset_name, 1),
    )
    if slice_.ambient_dimension != ambient_dimension:
        raise ValueError(f"{matrix_name} has the wrong ambient dimension.")
    if codimension is not None and slice_.codimension != codimension:
        raise ValueError(f"{matrix_name} has the wrong codimension.")
    return slice_


def _validate_payload(
    operation: HomotopyGeometryOperation,
    payload: Mapping[str, Any],
    system: SparsePolynomialSystem,
    /,
) -> None:
    _strict_fields(payload, _payload_fields(operation), f"{operation.value} payload")
    variable_count = system.support.variable_count
    if operation is HomotopyGeometryOperation.GENERIC_SLICE:
        dimension = _nonnegative_integer(payload["dimension"], "dimension")
        _validate_slice_payload(
            payload, "slice_matrix", "slice_offset", variable_count, dimension
        )
        return
    if operation is HomotopyGeometryOperation.WITNESS_TRANSPORT:
        _identifier(payload["witness_set_id"], "witness_set_id")
        dimension = _nonnegative_integer(payload["dimension"], "dimension")
        _validate_slice_payload(
            payload,
            "source_slice_matrix",
            "source_slice_offset",
            variable_count,
            dimension,
        )
        _validate_slice_payload(
            payload, "slice_matrix", "slice_offset", variable_count, dimension
        )
        points = _complex_array(payload["source_points"], "source_points", 2)
        if points.shape[1] != variable_count or not len(points):
            raise ValueError("source_points must be nonempty ambient points.")
        return
    if operation is HomotopyGeometryOperation.TRACE_TEST:
        _identifier(payload["witness_set_id"], "witness_set_id")
        source_slice = _validate_slice_payload(
            payload,
            "source_slice_matrix",
            "source_slice_offset",
            variable_count,
            None,
        )
        points = _complex_array(payload["source_points"], "source_points", 2)
        parameters = _real_array(payload["sample_parameters"], "sample_parameters", 1)
        offsets = _complex_array(payload["sample_offsets"], "sample_offsets", 2)
        point_indices = tuple(
            _nonnegative_integer(value, "point index")
            for value in _sequence(payload["point_indices"], "point_indices")
        )
        tolerance = float(payload["tolerance"])
        if (
            points.shape[1] != variable_count
            or not point_indices
            or len(set(point_indices)) != len(point_indices)
            or any(value >= len(points) for value in point_indices)
            or len(parameters) < 3
            or offsets.shape != (len(parameters), source_slice.codimension)
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Trace-test points, samples, or tolerance are invalid.")
        return
    if operation is HomotopyGeometryOperation.MONODROMY:
        _identifier(payload["witness_set_id"], "witness_set_id")
        source_slice = _validate_slice_payload(
            payload, "slice_matrix", "slice_offset", variable_count, None
        )
        points = _complex_array(payload["source_points"], "source_points", 2)
        loops = _sequence(payload["loops"], "loops")
        loop_ids: list[str] = []
        for loop_value in loops:
            loop = _mapping(loop_value, "loop")
            _strict_fields(
                loop, {"loop_id", "midpoint_matrix", "midpoint_offset"}, "loop"
            )
            loop_ids.append(_identifier(loop["loop_id"], "loop_id"))
            midpoint = AffineSlice(
                _complex_array(loop["midpoint_matrix"], "midpoint_matrix", 2),
                _complex_array(loop["midpoint_offset"], "midpoint_offset", 1),
            )
            if (
                midpoint.ambient_dimension != variable_count
                or midpoint.codimension != source_slice.codimension
            ):
                raise ValueError("Monodromy loop slices must match the witness slice.")
        if (
            points.shape[1] != variable_count
            or not len(points)
            or not loop_ids
            or len(set(loop_ids)) != len(loop_ids)
        ):
            raise ValueError("Monodromy points and loop IDs must be nonempty and unique.")
        return
    if operation is HomotopyGeometryOperation.REGENERATION_STAGE:
        _identifier(payload["plan_id"], "plan_id")
        _identifier(payload["edge_id"], "edge_id")
        _positive_integer(payload["stage_count"], "stage_count")
        dimension = _nonnegative_integer(payload["dimension"], "dimension")
        equations = tuple(
            _nonnegative_integer(value, "equation index")
            for value in _sequence(payload["equation_indices"], "equation_indices")
        )
        if len(set(equations)) != len(equations) or any(
            value >= system.support.equation_count for value in equations
        ):
            raise ValueError("Regeneration equation indices are invalid.")
        _validate_slice_payload(
            payload, "slice_matrix", "slice_offset", variable_count, dimension
        )
        return
    if operation is HomotopyGeometryOperation.IMAGE_DEGREE:
        _identifier(payload["source_system_id"], "source_system_id")
        _identifier(payload["map_id"], "map_id")
        source_dimension = _nonnegative_integer(
            payload["source_dimension"], "source_dimension"
        )
        image_dimension = _nonnegative_integer(
            payload["image_dimension"], "image_dimension"
        )
        map_equations = _positive_integer(
            payload["map_equation_count"], "map_equation_count"
        )
        if image_dimension > source_dimension:
            raise ValueError("image_dimension cannot exceed source_dimension.")
        equation_indices = np.asarray(payload["map_equation_indices"])
        exponents = np.asarray(payload["map_exponents"])
        coefficients = _complex_array(payload["map_coefficients"], "map_coefficients", 1)
        if (
            equation_indices.ndim != 1
            or equation_indices.dtype.kind not in "iu"
            or exponents.ndim != 2
            or exponents.dtype.kind not in "iu"
            or exponents.shape != (len(equation_indices), variable_count)
            or coefficients.shape != (len(equation_indices),)
            or np.any(equation_indices < 0)
            or np.any(equation_indices >= map_equations)
            or np.any(exponents < 0)
        ):
            raise ValueError("Sparse polynomial map term arrays are invalid.")
        _validate_slice_payload(
            payload,
            "source_slice_matrix",
            "source_slice_offset",
            variable_count,
            source_dimension - image_dimension,
        )
        _validate_slice_payload(
            payload,
            "image_slice_matrix",
            "image_slice_offset",
            map_equations,
            image_dimension,
        )
        return
    witness_sets = _sequence(payload["witness_sets"], "witness_sets")
    query_points = _complex_array(payload["query_points"], "query_points", 2)
    tolerance = float(payload["tolerance"])
    witness_ids: list[str] = []
    for value in witness_sets:
        witness = _mapping(value, "membership witness set")
        _strict_fields(
            witness,
            {
                "witness_set_id",
                "dimension",
                "slice_matrix",
                "slice_offset",
                "points",
            },
            "membership witness set",
        )
        witness_ids.append(_identifier(witness["witness_set_id"], "witness_set_id"))
        dimension = _nonnegative_integer(witness["dimension"], "dimension")
        _validate_slice_payload(
            witness, "slice_matrix", "slice_offset", variable_count, dimension
        )
        points = _complex_array(witness["points"], "points", 2)
        if points.shape[1] != variable_count or not len(points):
            raise ValueError("Membership witness points have the wrong shape.")
    if (
        not witness_sets
        or len(set(witness_ids)) != len(witness_ids)
        or query_points.shape[1] != variable_count
        or not len(query_points)
        or not isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("Membership witnesses, queries, or tolerance are invalid.")


class HomotopyGeometryRequest(StrictModule, NonTrainableState):
    """Canonical provider-neutral operation request with an exact path inventory."""

    system: SparsePolynomialSystem
    operation: HomotopyGeometryOperation = eqx.field(static=True)
    payload_json: str = eqx.field(static=True)
    paths: tuple[HomotopyGeometryPathRequest, ...]
    support_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SparsePolynomialSystem,
        operation: HomotopyGeometryOperation | str,
        payload: Mapping[str, Any],
        paths: Sequence[HomotopyGeometryPathRequest],
        /,
    ):
        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be a SparsePolynomialSystem.")
        try:
            operation_ = HomotopyGeometryOperation(operation)
        except ValueError as error:
            raise ValueError(
                f"Unknown homotopy-geometry operation {operation!r}."
            ) from error
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a data-only mapping.")
        payload_json = canonical_json(dict(payload))
        if len(payload_json.encode("utf-8")) > _MAX_REQUEST_BYTES:
            raise ValueError("Homotopy-geometry payload exceeds its byte bound.")
        payload_ = json.loads(payload_json)
        _validate_payload(operation_, payload_, system)
        paths_ = tuple(paths)
        if not paths_ or any(
            not isinstance(path, HomotopyGeometryPathRequest) for path in paths_
        ):
            raise TypeError(
                "paths must contain at least one HomotopyGeometryPathRequest."
            )
        path_ids = tuple(path.path_id for path in paths_)
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("Homotopy-geometry path IDs must be unique.")
        self.system = system
        self.operation = operation_
        self.payload_json = payload_json
        self.paths = paths_
        self.support_id = system.support.support_id
        self.system_id = system.system_id
        self.request_id = canonical_fingerprint(
            {
                "kind": "homotopy-geometry-request",
                "protocol": HOMOTOPY_GEOMETRY_PROTOCOL,
                "operation": operation_.value,
                "support": self.support_id,
                "system": self.system_id,
                "payload": payload_,
                "paths": [path.to_dict() for path in paths_],
            }
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @classmethod
    def generic_slice(
        cls,
        system: SparsePolynomialSystem,
        dimension: int,
        slice_matrix: ArrayLike,
        slice_offset: ArrayLike,
        /,
        *,
        path_count: int,
    ) -> HomotopyGeometryRequest:
        dimension_ = _nonnegative_integer(dimension, "dimension")
        count = _positive_integer(path_count, "path_count")
        payload = {
            "dimension": dimension_,
            "slice_matrix": _complex_payload(slice_matrix, "slice_matrix", 2),
            "slice_offset": _complex_payload(slice_offset, "slice_offset", 1),
        }
        paths = tuple(
            HomotopyGeometryPathRequest(f"generic:{value}", "generic-slice", value)
            for value in range(count)
        )
        return cls(system, HomotopyGeometryOperation.GENERIC_SLICE, payload, paths)

    @classmethod
    def witness_transport(
        cls,
        system: SparsePolynomialSystem,
        witness_set: WitnessSet,
        target_slice_matrix: ArrayLike,
        target_slice_offset: ArrayLike,
        /,
    ) -> HomotopyGeometryRequest:
        if not isinstance(witness_set, WitnessSet):
            raise TypeError("witness_set must be a WitnessSet.")
        if witness_set.system_id != system.system_id:
            raise ValueError("Witness set belongs to a different polynomial system.")
        payload = {
            "witness_set_id": witness_set.witness_id,
            "dimension": witness_set.dimension,
            "source_slice_matrix": _complex_payload(
                witness_set.slice.linear, "source_slice_matrix", 2
            ),
            "source_slice_offset": _complex_payload(
                witness_set.slice.offset, "source_slice_offset", 1
            ),
            "slice_matrix": _complex_payload(
                target_slice_matrix, "target_slice_matrix", 2
            ),
            "slice_offset": _complex_payload(
                target_slice_offset, "target_slice_offset", 1
            ),
            "source_points": _complex_payload(witness_set.points, "source_points", 2),
        }
        paths = tuple(
            HomotopyGeometryPathRequest(
                f"transport:{witness_set.witness_id}:{value}", "transport", value
            )
            for value in range(witness_set.degree)
        )
        return cls(system, HomotopyGeometryOperation.WITNESS_TRANSPORT, payload, paths)

    @classmethod
    def trace_test(
        cls,
        system: SparsePolynomialSystem,
        witness_set: WitnessSet,
        point_indices: Sequence[int],
        sample_parameters: ArrayLike,
        sample_offsets: ArrayLike,
        /,
        *,
        tolerance: float,
    ) -> HomotopyGeometryRequest:
        if not isinstance(witness_set, WitnessSet):
            raise TypeError("witness_set must be a WitnessSet.")
        if witness_set.system_id != system.system_id:
            raise ValueError("Witness set belongs to a different polynomial system.")
        parameters = np.asarray(sample_parameters)
        point_indices_ = tuple(index(value) for value in point_indices)
        payload = {
            "witness_set_id": witness_set.witness_id,
            "point_indices": list(point_indices_),
            "source_slice_matrix": _complex_payload(
                witness_set.slice.linear, "source_slice_matrix", 2
            ),
            "source_slice_offset": _complex_payload(
                witness_set.slice.offset, "source_slice_offset", 1
            ),
            "source_points": _complex_payload(witness_set.points, "source_points", 2),
            "sample_parameters": _real_payload(parameters, "sample_parameters", 1),
            "sample_offsets": _complex_payload(sample_offsets, "sample_offsets", 2),
            "tolerance": float(tolerance),
        }
        paths = tuple(
            HomotopyGeometryPathRequest(
                f"trace:{sample}:{source}", f"trace:{sample}", source
            )
            for sample in range(len(parameters))
            for source in point_indices_
        )
        return cls(system, HomotopyGeometryOperation.TRACE_TEST, payload, paths)

    @classmethod
    def monodromy(
        cls,
        system: SparsePolynomialSystem,
        witness_set: WitnessSet,
        loops: Sequence[tuple[str, ArrayLike, ArrayLike]],
        /,
    ) -> HomotopyGeometryRequest:
        if not isinstance(witness_set, WitnessSet):
            raise TypeError("witness_set must be a WitnessSet.")
        if witness_set.system_id != system.system_id:
            raise ValueError("Witness set belongs to a different polynomial system.")
        loops_ = tuple(loops)
        payload_loops = [
            {
                "loop_id": _identifier(loop_id, "loop_id"),
                "midpoint_matrix": _complex_payload(matrix, "midpoint_matrix", 2),
                "midpoint_offset": _complex_payload(offset, "midpoint_offset", 1),
            }
            for loop_id, matrix, offset in loops_
        ]
        payload = {
            "witness_set_id": witness_set.witness_id,
            "slice_matrix": _complex_payload(witness_set.slice.linear, "slice_matrix", 2),
            "slice_offset": _complex_payload(witness_set.slice.offset, "slice_offset", 1),
            "source_points": _complex_payload(witness_set.points, "source_points", 2),
            "loops": payload_loops,
        }
        paths = tuple(
            HomotopyGeometryPathRequest(f"monodromy:{loop_id}:{source}", loop_id, source)
            for loop_id, _, _ in loops_
            for source in range(witness_set.degree)
        )
        return cls(system, HomotopyGeometryOperation.MONODROMY, payload, paths)

    @classmethod
    def regeneration_stage(
        cls,
        system: SparsePolynomialSystem,
        plan: RegenerationPlan,
        edge: RegenerationEdge,
        target_slice_matrix: ArrayLike,
        target_slice_offset: ArrayLike,
        /,
    ) -> HomotopyGeometryRequest:
        if not isinstance(plan, RegenerationPlan) or not isinstance(
            edge, RegenerationEdge
        ):
            raise TypeError("plan and edge must be regeneration contracts.")
        if plan.system_id != system.system_id or edge not in plan.edges:
            raise ValueError(
                "Regeneration plan/edge belongs to a different system or plan."
            )
        target = next(
            stage for stage in plan.stages if stage.stage_id == edge.target_stage_id
        )
        payload = {
            "plan_id": plan.plan_id,
            "edge_id": edge.edge_id,
            "stage_count": len(plan.stages),
            "dimension": target.dimension,
            "equation_indices": list(target.equation_indices),
            "slice_matrix": _complex_payload(
                target_slice_matrix, "target_slice_matrix", 2
            ),
            "slice_offset": _complex_payload(
                target_slice_offset, "target_slice_offset", 1
            ),
        }
        paths = tuple(
            HomotopyGeometryPathRequest(path_id, edge.edge_id, source)
            for source, path_id in enumerate(edge.expected_path_ids)
        )
        return cls(system, HomotopyGeometryOperation.REGENERATION_STAGE, payload, paths)

    @classmethod
    def image_degree(
        cls,
        source_system: SparsePolynomialSystem,
        map_system: SparsePolynomialSystem,
        map_id: str,
        source_dimension: int,
        image_dimension: int,
        source_slice_matrix: ArrayLike,
        source_slice_offset: ArrayLike,
        image_slice_matrix: ArrayLike,
        image_slice_offset: ArrayLike,
        /,
        *,
        path_count: int,
    ) -> HomotopyGeometryRequest:
        if not isinstance(map_system, SparsePolynomialSystem):
            raise TypeError("map_system must be a SparsePolynomialSystem.")
        if map_system.support.variable_labels != source_system.support.variable_labels:
            raise ValueError(
                "Polynomial map source variables do not match the source system."
            )
        map_record = _system_record(map_system)
        payload = {
            "source_system_id": source_system.system_id,
            "map_id": _identifier(map_id, "map_id"),
            "source_dimension": _nonnegative_integer(
                source_dimension, "source_dimension"
            ),
            "image_dimension": _nonnegative_integer(image_dimension, "image_dimension"),
            "map_equation_count": map_record["equation_count"],
            "map_equation_indices": map_record["equation_indices"],
            "map_exponents": map_record["exponents"],
            "map_coefficients": map_record["coefficients"],
            "source_slice_matrix": _complex_payload(
                source_slice_matrix, "source_slice_matrix", 2
            ),
            "source_slice_offset": _complex_payload(
                source_slice_offset, "source_slice_offset", 1
            ),
            "image_slice_matrix": _complex_payload(
                image_slice_matrix, "image_slice_matrix", 2
            ),
            "image_slice_offset": _complex_payload(
                image_slice_offset, "image_slice_offset", 1
            ),
        }
        count = _positive_integer(path_count, "path_count")
        paths = tuple(
            HomotopyGeometryPathRequest(f"image:{value}", "image-degree", value)
            for value in range(count)
        )
        return cls(source_system, HomotopyGeometryOperation.IMAGE_DEGREE, payload, paths)

    @classmethod
    def membership(
        cls,
        system: SparsePolynomialSystem,
        witness_sets: Sequence[WitnessSet],
        query_points: ArrayLike,
        /,
        *,
        tolerance: float,
    ) -> HomotopyGeometryRequest:
        witnesses = tuple(witness_sets)
        if not witnesses or any(
            not isinstance(witness, WitnessSet) for witness in witnesses
        ):
            raise TypeError("witness_sets must contain at least one WitnessSet.")
        if any(witness.system_id != system.system_id for witness in witnesses):
            raise ValueError(
                "All membership witnesses must belong to the requested system."
            )
        queries = np.asarray(query_points)
        payload = {
            "witness_sets": [
                {
                    "witness_set_id": witness.witness_id,
                    "dimension": witness.dimension,
                    "slice_matrix": _complex_payload(
                        witness.slice.linear, "slice_matrix", 2
                    ),
                    "slice_offset": _complex_payload(
                        witness.slice.offset, "slice_offset", 1
                    ),
                    "points": _complex_payload(witness.points, "points", 2),
                }
                for witness in witnesses
            ],
            "query_points": _complex_payload(queries, "query_points", 2),
            "tolerance": float(tolerance),
        }
        paths = tuple(
            HomotopyGeometryPathRequest(
                f"membership:{query}:{witness.witness_id}:{source}",
                f"membership:{query}:{witness.witness_id}",
                source,
            )
            for query in range(len(queries))
            for witness in witnesses
            for source in range(witness.degree)
        )
        return cls(system, HomotopyGeometryOperation.MEMBERSHIP, payload, paths)

    def to_dict(
        self,
        provider: HomotopyContinuationProvider,
        policy: HomotopyGeometryPolicy,
        /,
    ) -> dict[str, Any]:
        if not isinstance(provider, HomotopyContinuationProvider):
            raise TypeError("provider must be a HomotopyContinuationProvider.")
        if not isinstance(policy, HomotopyGeometryPolicy):
            raise TypeError("policy must be a HomotopyGeometryPolicy.")
        if len(self.paths) > policy.path_capacity:
            raise ValueError("Request path inventory exceeds policy.path_capacity.")
        payload = self.payload
        if (
            self.operation is HomotopyGeometryOperation.MONODROMY
            and len(payload["loops"]) > policy.loop_capacity
        ):
            raise ValueError("Monodromy loop inventory exceeds policy.loop_capacity.")
        if (
            self.operation is HomotopyGeometryOperation.REGENERATION_STAGE
            and payload["stage_count"] > policy.stage_capacity
        ):
            raise ValueError(
                "Regeneration stage inventory exceeds policy.stage_capacity."
            )
        return {
            "protocol": HOMOTOPY_GEOMETRY_PROTOCOL,
            "request_id": self.request_id,
            "provider_id": provider.provider_id,
            "environment_id": provider.environment.environment_id,
            "worker_sha256": HOMOTOPY_GEOMETRY_WORKER_SHA256,
            "runtime": {
                "project_sha256": provider.environment.project_sha256,
                "manifest_sha256": provider.environment.manifest_sha256,
                "homotopy_continuation_uuid": (
                    provider.environment.homotopy_continuation_uuid
                ),
                "homotopy_continuation_version": (
                    provider.environment.homotopy_continuation_version
                ),
                "worker_sha256": HOMOTOPY_GEOMETRY_WORKER_SHA256,
            },
            "operation": self.operation.value,
            "system_id": self.system_id,
            "support_id": self.support_id,
            "policy": {
                "path_capacity": policy.path_capacity,
                "loop_capacity": policy.loop_capacity,
                "stage_capacity": policy.stage_capacity,
                "seed": policy.seed,
            },
            "system": _system_record(self.system),
            "payload": payload,
            "paths": [path.to_dict() for path in self.paths],
        }


class MembershipEvidence(StrictModule, NonTrainableState):
    """Witness-transport membership observations, not exact ideal membership proof."""

    query_points: Array
    member_witness_set_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    residual_norms: Array
    tolerance: float = eqx.field(static=True)
    paths: PathInventory
    evidence_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        query_points: ArrayLike,
        member_witness_set_ids: Sequence[Sequence[str]],
        residual_norms: ArrayLike,
        tolerance: float,
        paths: PathInventory,
        /,
    ):
        points = np.asarray(query_points)
        residuals = np.asarray(residual_norms)
        members = tuple(
            tuple(_identifier(value, "member witness-set ID") for value in values)
            for values in member_witness_set_ids
        )
        tolerance_ = float(tolerance)
        if (
            points.ndim != 2
            or points.dtype.kind not in "fciu"
            or np.any(~np.isfinite(points))
            or residuals.ndim != 1
            or residuals.shape != (len(points),)
            or residuals.dtype.kind not in "fiu"
            or np.any(~np.isfinite(residuals))
            or np.any(residuals < 0)
            or len(members) != len(points)
            or any(len(set(values)) != len(values) for values in members)
            or not isfinite(tolerance_)
            or tolerance_ < 0.0
        ):
            raise ValueError("Membership observations have invalid shapes or values.")
        if points.dtype.kind in "iu":
            points = points.astype("float64")
        if not isinstance(paths, PathInventory):
            raise TypeError("paths must be a PathInventory.")
        self.query_points = jnp.asarray(points)
        self.member_witness_set_ids = members
        self.residual_norms = jnp.asarray(residuals)
        self.tolerance = tolerance_
        self.paths = paths
        self.claim = "numerical-witness-transport-membership-not-exact-ideal-membership"
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-membership-evidence",
                "query_points": points,
                "member_witness_set_ids": members,
                "residual_norms": residuals,
                "tolerance": tolerance_,
                "paths": paths.inventory_id,
                "claim": self.claim,
            }
        )

    @property
    def evidence_complete(self) -> bool:
        return self.paths.successful


GeometryOutput = (
    WitnessSet
    | PseudoWitnessSet
    | MonodromyEvidence
    | TraceTestEvidence
    | MembershipEvidence
)


class HomotopyGeometryResult(StrictModule, NonTrainableState):
    """Qualified external result retaining path, process, and identity evidence."""

    operation: HomotopyGeometryOperation = eqx.field(static=True)
    status: HomotopyGeometryStatus = eqx.field(static=True)
    output: GeometryOutput | None
    paths: PathInventory | None
    run: EnergyRunResult | None
    request_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    worker_sha256: str = eqx.field(static=True)
    error: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        operation: HomotopyGeometryOperation,
        status: HomotopyGeometryStatus,
        output: GeometryOutput | None,
        paths: PathInventory | None,
        run: EnergyRunResult | None,
        request_id: str,
        provider_id: str,
        environment_id: str,
        worker_sha256: str,
        /,
        *,
        error: str = "",
    ):
        self.operation = operation
        self.status = status
        self.output = output
        self.paths = paths
        self.run = run
        self.request_id = _identifier(request_id, "request_id")
        self.provider_id = _identifier(provider_id, "provider_id")
        self.environment_id = _identifier(environment_id, "environment_id")
        self.worker_sha256 = _identifier(worker_sha256, "worker_sha256")
        self.error = str(error)
        output_id: str | None
        if isinstance(output, WitnessSet):
            output_id = output.witness_id
        elif isinstance(output, PseudoWitnessSet):
            output_id = output.pseudo_witness_id
        elif isinstance(output, MonodromyEvidence):
            output_id = output.evidence_id
        elif isinstance(output, TraceTestEvidence):
            output_id = output.evidence_id
        elif isinstance(output, MembershipEvidence):
            output_id = output.evidence_id
        elif output is None:
            output_id = None
        else:
            raise TypeError("Unsupported homotopy-geometry output contract.")
        self.result_id = canonical_fingerprint(
            {
                "kind": "homotopy-geometry-result",
                "operation": operation.value,
                "status": status.value,
                "output": output_id,
                "paths": None if paths is None else paths.inventory_id,
                "run": None if run is None else run.artifact.artifact_id,
                "request": self.request_id,
                "provider": self.provider_id,
                "environment": self.environment_id,
                "worker": self.worker_sha256,
                "error": self.error,
            }
        )

    @property
    def successful(self) -> bool:
        return self.status is HomotopyGeometryStatus.SUCCESS


_RESPONSE_FIELDS = {
    "protocol",
    "request_id",
    "provider_id",
    "environment_id",
    "worker_sha256",
    "operation",
    "system_id",
    "support_id",
    "status",
    "budget_exhausted",
    "paths",
    "result",
}
_PATH_FIELDS = {
    "path_id",
    "batch_id",
    "source_index",
    "target_index",
    "status",
    "residual_norm",
    "diagnostic",
}


def _identity_failure(
    request: HomotopyGeometryRequest,
    provider_id: str,
    environment_id: str,
    run: EnergyRunResult | None,
    error: str,
    /,
) -> HomotopyGeometryResult:
    return HomotopyGeometryResult(
        request.operation,
        HomotopyGeometryStatus.IDENTITY_MISMATCH,
        None,
        None,
        run,
        request.request_id,
        provider_id,
        environment_id,
        HOMOTOPY_GEOMETRY_WORKER_SHA256,
        error=error,
    )


def _decode_paths(
    request: HomotopyGeometryRequest,
    policy: HomotopyGeometryPolicy,
    records_value: Any,
    budget_exhausted: Any,
    /,
) -> PathInventory:
    records_sequence = _sequence(records_value, "paths")
    records: list[PathRecord] = []
    for value in records_sequence:
        record = _mapping(value, "path record")
        _strict_fields(record, _PATH_FIELDS, "path record")
        records.append(
            PathRecord(
                str(record["path_id"]),
                str(record["batch_id"]),
                record["source_index"],
                str(record["status"]),
                target_index=record["target_index"],
                residual_norm=record["residual_norm"],
                diagnostic=str(record["diagnostic"]),
            )
        )
    inventory = PathInventory(
        tuple(path.path_id for path in request.paths),
        records,
        path_capacity=policy.path_capacity,
        budget_exhausted=bool(budget_exhausted),
    )
    expected_specs = {
        path.path_id: (path.batch_id, path.source_index) for path in request.paths
    }
    if any(
        expected_specs[record.path_id] != (record.batch_id, record.source_index)
        for record in inventory.records
    ):
        raise ValueError("Path result batch/source inventory differs from the request.")
    return inventory


def _maximum_residuals(
    system: SparsePolynomialSystem,
    points: np.ndarray,
    equation_indices: Sequence[int] | None = None,
    /,
) -> np.ndarray:
    values = np.asarray(system.evaluate(points))
    if equation_indices is not None:
        values = values[:, np.asarray(tuple(equation_indices), dtype=np.intp)]
    return np.max(np.abs(values), axis=1, initial=0.0)


def _require_residual_match(
    reported: np.ndarray,
    observed: np.ndarray,
    owner: str,
    /,
) -> None:
    tolerance = 1.0e-10 + 1.0e-7 * np.maximum(1.0, observed)
    if reported.shape != observed.shape or np.any(
        np.abs(reported - observed) > tolerance
    ):
        raise ValueError(
            f"{owner} residual evidence disagrees with native polynomial evaluation."
        )


def _evaluate_map_payload(
    payload: Mapping[str, Any],
    points: np.ndarray,
    /,
) -> np.ndarray:
    equation_indices = np.asarray(payload["map_equation_indices"], dtype=np.intp)
    exponents = np.asarray(payload["map_exponents"], dtype=np.int64)
    coefficients = _complex_array(payload["map_coefficients"], "map_coefficients", 1)
    monomials = np.prod(
        points[:, None, :] ** exponents[None, :, :],
        axis=-1,
    )
    terms = monomials * coefficients[None, :]
    values = np.zeros(
        (points.shape[0], int(payload["map_equation_count"])),
        dtype=np.result_type(points.dtype, coefficients.dtype),
    )
    for term, equation in enumerate(equation_indices):
        values[:, equation] += terms[:, term]
    return values


def _decode_witness(
    request: HomotopyGeometryRequest,
    value: Mapping[str, Any],
    /,
) -> WitnessSet:
    _strict_fields(
        value,
        {
            "dimension",
            "slice_matrix",
            "slice_offset",
            "points",
            "residual_norms",
        },
        "witness result",
    )
    points = _complex_array(value["points"], "points", 2)
    residuals = _real_array(value["residual_norms"], "residual_norms", 1)
    payload = request.payload
    equation_indices = (
        payload["equation_indices"]
        if request.operation is HomotopyGeometryOperation.REGENERATION_STAGE
        else None
    )
    _require_residual_match(
        residuals,
        _maximum_residuals(request.system, points, equation_indices),
        "Witness",
    )
    return WitnessSet(
        request.system_id,
        value["dimension"],
        _complex_array(value["slice_matrix"], "slice_matrix", 2),
        _complex_array(value["slice_offset"], "slice_offset", 1),
        points,
        residuals,
    )


def _decode_output(
    request: HomotopyGeometryRequest,
    result_value: Any,
    paths: PathInventory,
    /,
) -> GeometryOutput:
    value = _mapping(result_value, "result")
    operation = request.operation
    if operation in (
        HomotopyGeometryOperation.GENERIC_SLICE,
        HomotopyGeometryOperation.WITNESS_TRANSPORT,
        HomotopyGeometryOperation.REGENERATION_STAGE,
    ):
        return _decode_witness(request, value)
    if operation is HomotopyGeometryOperation.TRACE_TEST:
        _strict_fields(
            value,
            {
                "witness_set_id",
                "point_indices",
                "sample_parameters",
                "trace_values",
                "affine_fit_residual",
                "tolerance",
                "passed",
            },
            "trace-test result",
        )
        evidence = TraceTestEvidence(
            str(value["witness_set_id"]),
            tuple(value["point_indices"]),
            _real_array(value["sample_parameters"], "sample_parameters", 1),
            _complex_array(value["trace_values"], "trace_values", 2),
            float(value["affine_fit_residual"]),
            float(value["tolerance"]),
            paths,
        )
        numerical_pass = (
            evidence.affine_fit_residual <= evidence.tolerance and evidence.finite
        )
        if bool(value["passed"]) != numerical_pass:
            raise ValueError("Trace-test pass flag contradicts its residual evidence.")
        payload = request.payload
        if (
            evidence.witness_set_id != payload["witness_set_id"]
            or evidence.point_indices != tuple(sorted(payload["point_indices"]))
            or not np.array_equal(
                np.asarray(evidence.sample_parameters),
                np.asarray(payload["sample_parameters"]),
            )
            or evidence.tolerance != float(payload["tolerance"])
        ):
            raise ValueError("Trace-test result identity or sample schedule mismatch.")
        return evidence
    if operation is HomotopyGeometryOperation.MONODROMY:
        _strict_fields(
            value,
            {
                "witness_set_id",
                "point_count",
                "attempted_loop_ids",
                "completed",
            },
            "monodromy result",
        )
        completed_values = _sequence(value["completed"], "completed")
        completed: list[tuple[str, Sequence[int]]] = []
        for completed_value in completed_values:
            loop = _mapping(completed_value, "completed loop")
            _strict_fields(loop, {"loop_id", "permutation"}, "completed loop")
            completed.append((str(loop["loop_id"]), tuple(loop["permutation"])))
        evidence = MonodromyEvidence(
            str(value["witness_set_id"]),
            value["point_count"],
            tuple(value["attempted_loop_ids"]),
            completed,
            paths,
        )
        payload = request.payload
        expected_loops = tuple(loop["loop_id"] for loop in payload["loops"])
        if (
            evidence.witness_set_id != payload["witness_set_id"]
            or evidence.attempted_loop_ids != expected_loops
            or evidence.point_count
            != _complex_array(payload["source_points"], "source_points", 2).shape[0]
        ):
            raise ValueError("Monodromy result identity or loop inventory mismatch.")
        return evidence
    if operation is HomotopyGeometryOperation.IMAGE_DEGREE:
        _strict_fields(
            value,
            {
                "source_system_id",
                "map_id",
                "source_dimension",
                "image_dimension",
                "source_slice_matrix",
                "source_slice_offset",
                "image_slice_matrix",
                "image_slice_offset",
                "source_points",
                "image_points",
                "residual_norms",
                "image_degree",
            },
            "image-degree result",
        )
        payload = request.payload
        source_slice_matrix = _complex_array(
            value["source_slice_matrix"], "source_slice_matrix", 2
        )
        source_slice_offset = _complex_array(
            value["source_slice_offset"], "source_slice_offset", 1
        )
        image_slice_matrix = _complex_array(
            value["image_slice_matrix"], "image_slice_matrix", 2
        )
        image_slice_offset = _complex_array(
            value["image_slice_offset"], "image_slice_offset", 1
        )
        source_points = _complex_array(value["source_points"], "source_points", 2)
        image_points = _complex_array(value["image_points"], "image_points", 2)
        residuals = _real_array(value["residual_norms"], "residual_norms", 1)
        map_values = _evaluate_map_payload(payload, source_points)
        source_residuals = _maximum_residuals(request.system, source_points)
        graph_residuals = np.max(np.abs(image_points - map_values), axis=1, initial=0.0)
        source_slice_residuals = np.max(
            np.abs(source_points @ source_slice_matrix.T + source_slice_offset),
            axis=1,
            initial=0.0,
        )
        image_slice_residuals = np.max(
            np.abs(image_points @ image_slice_matrix.T + image_slice_offset),
            axis=1,
            initial=0.0,
        )
        observed_residuals = np.maximum.reduce(
            (
                source_residuals,
                graph_residuals,
                source_slice_residuals,
                image_slice_residuals,
            )
        )
        _require_residual_match(residuals, observed_residuals, "Pseudo-witness")
        pseudo = PseudoWitnessSet(
            str(value["source_system_id"]),
            str(value["map_id"]),
            value["source_dimension"],
            value["image_dimension"],
            source_slice_matrix,
            source_slice_offset,
            image_slice_matrix,
            image_slice_offset,
            source_points,
            image_points,
            residuals,
            image_degree=value["image_degree"],
        )
        if (
            pseudo.source_system_id != request.system_id
            or pseudo.map_id != payload["map_id"]
            or pseudo.source_dimension != payload["source_dimension"]
            or pseudo.image_dimension != payload["image_dimension"]
        ):
            raise ValueError("Pseudo-witness result identity or dimensions mismatch.")
        return pseudo
    _strict_fields(
        value,
        {
            "query_points",
            "member_witness_set_ids",
            "residual_norms",
            "tolerance",
            "claim",
        },
        "membership result",
    )
    if (
        value["claim"]
        != "numerical-witness-transport-membership-not-exact-ideal-membership"
    ):
        raise ValueError("Membership result asserted an unsupported claim.")
    query_points = _complex_array(value["query_points"], "query_points", 2)
    residuals = _real_array(value["residual_norms"], "residual_norms", 1)
    _require_residual_match(
        residuals,
        _maximum_residuals(request.system, query_points),
        "Membership",
    )
    evidence = MembershipEvidence(
        query_points,
        tuple(tuple(values) for values in value["member_witness_set_ids"]),
        residuals,
        float(value["tolerance"]),
        paths,
    )
    payload = request.payload
    allowed = {witness["witness_set_id"] for witness in payload["witness_sets"]}
    if (
        not np.array_equal(
            np.asarray(evidence.query_points),
            _complex_array(payload["query_points"], "query_points", 2),
        )
        or evidence.tolerance != float(payload["tolerance"])
        or any(set(values) - allowed for values in evidence.member_witness_set_ids)
    ):
        raise ValueError(
            "Membership result query, tolerance, or witness identity mismatch."
        )
    return evidence


def _validate_path_targets(
    request: HomotopyGeometryRequest,
    output: GeometryOutput,
    paths: PathInventory,
    /,
) -> None:
    successful = tuple(
        record for record in paths.records if record.status is PathStatus.SUCCESS
    )
    if request.operation in (
        HomotopyGeometryOperation.GENERIC_SLICE,
        HomotopyGeometryOperation.REGENERATION_STAGE,
        HomotopyGeometryOperation.IMAGE_DEGREE,
    ):
        if request.operation is HomotopyGeometryOperation.IMAGE_DEGREE:
            if not isinstance(output, PseudoWitnessSet):
                raise TypeError("Image-degree output must be a PseudoWitnessSet.")
            upper = output.source_points.shape[0]
        else:
            if not isinstance(output, WitnessSet):
                raise TypeError("Slice output must be a WitnessSet.")
            upper = output.degree
        if any(
            record.target_index is None or record.target_index >= upper
            for record in successful
        ):
            raise ValueError("Successful path target index exceeds returned endpoints.")
        return
    if request.operation is HomotopyGeometryOperation.MONODROMY:
        return
    allowed_by_batch: dict[str, set[int]] = {}
    for path in request.paths:
        allowed_by_batch.setdefault(path.batch_id, set()).add(path.source_index)
    if any(
        record.target_index is None
        or record.target_index not in allowed_by_batch[record.batch_id]
        for record in successful
    ):
        raise ValueError(
            "Successful transport path target index is absent from its batch inventory."
        )


def decode_homotopy_geometry_result(
    request: HomotopyGeometryRequest,
    policy: HomotopyGeometryPolicy,
    value: Mapping[str, Any],
    /,
    *,
    provider_id: str,
    environment_id: str,
    run: EnergyRunResult | None = None,
) -> HomotopyGeometryResult:
    """Validate one detached worker record without executing a provider."""

    if not isinstance(request, HomotopyGeometryRequest):
        raise TypeError("request must be a HomotopyGeometryRequest.")
    if not isinstance(policy, HomotopyGeometryPolicy):
        raise TypeError("policy must be a HomotopyGeometryPolicy.")
    record = _mapping(value, "homotopy-geometry response")
    _strict_fields(record, _RESPONSE_FIELDS, "homotopy-geometry response")
    provider = _identifier(provider_id, "provider_id")
    environment = _identifier(environment_id, "environment_id")
    identity_errors: list[str] = []
    expected_echoes = {
        "protocol": HOMOTOPY_GEOMETRY_PROTOCOL,
        "request_id": request.request_id,
        "provider_id": provider,
        "environment_id": environment,
        "worker_sha256": HOMOTOPY_GEOMETRY_WORKER_SHA256,
        "operation": request.operation.value,
        "system_id": request.system_id,
        "support_id": request.support_id,
    }
    for field, expected in expected_echoes.items():
        if record[field] != expected:
            identity_errors.append(field)
    if identity_errors:
        return _identity_failure(
            request,
            provider,
            environment,
            run,
            f"Provider response identity mismatch in {tuple(identity_errors)!r}.",
        )
    try:
        claimed_status = HomotopyGeometryStatus(record["status"])
    except ValueError as error:
        raise ValueError("Worker returned an unknown geometry status.") from error
    if claimed_status not in (
        HomotopyGeometryStatus.SUCCESS,
        HomotopyGeometryStatus.PARTIAL_PATH_FAILURE,
        HomotopyGeometryStatus.TRACE_TEST_FAILED,
        HomotopyGeometryStatus.BUDGET_EXHAUSTED,
    ):
        raise ValueError("Worker returned a host-only failure status.")
    paths = _decode_paths(request, policy, record["paths"], record["budget_exhausted"])
    output = _decode_output(request, record["result"], paths)
    _validate_path_targets(request, output, paths)
    if paths.budget_exhausted:
        derived_status = HomotopyGeometryStatus.BUDGET_EXHAUSTED
    elif not paths.successful:
        derived_status = HomotopyGeometryStatus.PARTIAL_PATH_FAILURE
    elif isinstance(output, TraceTestEvidence) and not output.passed:
        derived_status = HomotopyGeometryStatus.TRACE_TEST_FAILED
    else:
        derived_status = HomotopyGeometryStatus.SUCCESS
    if claimed_status is not derived_status:
        raise ValueError("Worker status contradicts path or trace-test evidence.")
    return HomotopyGeometryResult(
        request.operation,
        derived_status,
        output,
        paths,
        run,
        request.request_id,
        provider,
        environment,
        HOMOTOPY_GEOMETRY_WORKER_SHA256,
    )


def _host_failure(
    request: HomotopyGeometryRequest,
    provider: HomotopyContinuationProvider,
    status: HomotopyGeometryStatus,
    run: EnergyRunResult | None,
    error: str,
    /,
) -> HomotopyGeometryResult:
    return HomotopyGeometryResult(
        request.operation,
        status,
        None,
        None,
        run,
        request.request_id,
        provider.provider_id,
        provider.environment.environment_id,
        HOMOTOPY_GEOMETRY_WORKER_SHA256,
        error=error,
    )


def execute_homotopy_geometry(
    provider: HomotopyContinuationProvider,
    policy: HomotopyGeometryPolicy,
    request: HomotopyGeometryRequest,
    /,
) -> HomotopyGeometryResult:
    """Execute the fixed worker in a fresh bounded process with pinned inputs."""

    if not isinstance(provider, HomotopyContinuationProvider):
        raise TypeError("provider must be a HomotopyContinuationProvider.")
    if not isinstance(policy, HomotopyGeometryPolicy):
        raise TypeError("policy must be a HomotopyGeometryPolicy.")
    if not isinstance(request, HomotopyGeometryRequest):
        raise TypeError("request must be a HomotopyGeometryRequest.")
    project_toml, manifest_toml = provider.environment.verify()
    if _WORKER_PATH.read_bytes() != _WORKER_BYTES:
        raise ValueError("The fixed homotopy-geometry worker changed after import.")
    request_bytes = canonical_json(request.to_dict(provider, policy)).encode("utf-8")
    if len(request_bytes) > min(_MAX_REQUEST_BYTES, policy.maximum_output_bytes):
        raise ValueError("Serialized homotopy-geometry request exceeds its byte bound.")
    environment = {"JULIA_HISTORY": "/dev/null", "LC_ALL": "C"}
    if provider.environment.depot_path:
        environment["JULIA_DEPOT_PATH"] = provider.environment.depot_path
    try:
        run = run_energy_command(
            provider.executable,
            (
                "--startup-file=no",
                "--history-file=no",
                "--project=julia-project",
                "homotopy-geometry-worker.jl",
                "homotopy-geometry-request.json",
                "homotopy-geometry-result.json",
            ),
            inputs={
                "julia-project/Project.toml": project_toml,
                "julia-project/Manifest.toml": manifest_toml,
                "homotopy-geometry-worker.jl": _WORKER_BYTES,
                "homotopy-geometry-request.json": request_bytes,
            },
            outputs=("homotopy-geometry-result.json",),
            timeout=policy.timeout_seconds,
            max_output_bytes=policy.maximum_output_bytes,
            environment=environment,
        )
    except EnergyRuntimeError as failure:
        provider.environment.verify()
        if _WORKER_PATH.read_bytes() != _WORKER_BYTES:
            raise ValueError(
                "The fixed homotopy-geometry worker changed during execution."
            )
        return _host_failure(
            request,
            provider,
            HomotopyGeometryStatus.PROVIDER_FAILED,
            failure.result,
            str(failure),
        )
    provider.environment.verify()
    if _WORKER_PATH.read_bytes() != _WORKER_BYTES:
        raise ValueError("The fixed homotopy-geometry worker changed during execution.")
    try:
        response = json.loads(run.output("homotopy-geometry-result.json"))
        return decode_homotopy_geometry_result(
            request,
            policy,
            response,
            provider_id=provider.provider_id,
            environment_id=provider.environment.environment_id,
            run=run,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ) as failure:
        return _host_failure(
            request,
            provider,
            HomotopyGeometryStatus.INVALID_OUTPUT,
            run,
            f"{type(failure).__name__}: {failure}",
        )


__all__ = [
    "HOMOTOPY_GEOMETRY_PROTOCOL",
    "HOMOTOPY_GEOMETRY_WORKER_SHA256",
    "HomotopyGeometryOperation",
    "HomotopyGeometryPathRequest",
    "HomotopyGeometryPolicy",
    "HomotopyGeometryRequest",
    "HomotopyGeometryResult",
    "HomotopyGeometryStatus",
    "MembershipEvidence",
    "decode_homotopy_geometry_result",
    "execute_homotopy_geometry",
]

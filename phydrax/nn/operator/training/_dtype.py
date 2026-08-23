#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._precision import (
    complex_precision_dtype,
    ComplexPrecisionDType,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
    RealPrecisionDType,
)
from ...._trainable import combine_trainable, partition_trainable
from ..data import FunctionSamples, OperatorBatch, OperatorTargetBatch


DTypeName: TypeAlias = RealPrecisionDType
ComplexDTypeName: TypeAlias = ComplexPrecisionDType
MatmulPrecisionName = Literal[
    "default",
    "high",
    "highest",
    "F16_F16_F32",
    "BF16_BF16_F32",
    "TF32_TF32_F32",
    "F32_F32_F32",
]

_MATMUL_PRECISIONS = (
    "default",
    "high",
    "highest",
    "F16_F16_F32",
    "BF16_BF16_F32",
    "TF32_TF32_F32",
    "F32_F32_F32",
)
_ALGORITHM_COMPUTE_DTYPES = {
    "F16_F16_F32": "float16",
    "BF16_BF16_F32": "bfloat16",
    "TF32_TF32_F32": "float32",
    "F32_F32_F32": "float32",
}


def _dtype(name: DTypeName):
    return jnp.dtype(real_precision_dtype_name(name))


def _complex_dtype_name(name: DTypeName, /) -> ComplexDTypeName:
    return complex_precision_dtype(name)


def _cast_inexact(value: Any, dtype, /):
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return array
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        target = jnp.complex128 if jnp.dtype(dtype).itemsize >= 8 else jnp.complex64
        return array.astype(target)
    return array.astype(dtype)


def _cast_parameter_tree(tree: Any, dtype, /) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: None if leaf is None else _cast_inexact(leaf, dtype),
        tree,
    )


@dataclass(frozen=True, slots=True)
class OperatorPrecisionEvidence:
    """Effective real/complex arithmetic and preserved-geometry semantics."""

    parameter_dtype: DTypeName
    parameter_complex_dtype: ComplexDTypeName
    compute_dtype: DTypeName
    compute_complex_dtype: ComplexDTypeName
    reduction_dtype: DTypeName
    matmul_precision: MatmulPrecisionName | None
    geometry_mode: Literal["preserve"] = "preserve"

    def __post_init__(self):
        parameter = real_precision_dtype_name(self.parameter_dtype)
        compute = real_precision_dtype_name(self.compute_dtype)
        real_precision_dtype_name(self.reduction_dtype)
        if self.parameter_complex_dtype != complex_precision_dtype(
            parameter
        ) or self.compute_complex_dtype != complex_precision_dtype(compute):
            raise ValueError(
                "Operator complex precision must match its effective real companion."
            )
        if (
            self.matmul_precision is not None
            and self.matmul_precision not in _MATMUL_PRECISIONS
        ):
            raise ValueError(f"Unsupported matmul precision {self.matmul_precision!r}.")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "parameter_dtype": self.parameter_dtype,
            "parameter_complex_dtype": self.parameter_complex_dtype,
            "compute_dtype": self.compute_dtype,
            "compute_complex_dtype": self.compute_complex_dtype,
            "reduction_dtype": self.reduction_dtype,
            "matmul_precision": self.matmul_precision,
            "geometry_mode": self.geometry_mode,
        }

    def to_envelope(self) -> PrecisionEvidenceEnvelope:
        request = PrecisionRequest(
            "neural-operator",
            {
                "storage": self.parameter_dtype,
                "compute": self.compute_dtype,
                "accumulation": self.reduction_dtype,
                "output": self.compute_dtype,
            },
        )
        resolution = PrecisionResolution(
            request,
            "phydrax-operator",
            dict(request.requested),
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        /,
    ) -> OperatorPrecisionEvidence:
        expected = {
            "parameter_dtype",
            "parameter_complex_dtype",
            "compute_dtype",
            "compute_complex_dtype",
            "reduction_dtype",
            "matmul_precision",
            "geometry_mode",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "Operator precision evidence must use the current canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if value["geometry_mode"] != "preserve":
            raise ValueError("Operator precision evidence must preserve geometry.")
        return cls(
            parameter_dtype=cast(DTypeName, value["parameter_dtype"]),
            parameter_complex_dtype=cast(
                ComplexDTypeName,
                value["parameter_complex_dtype"],
            ),
            compute_dtype=cast(DTypeName, value["compute_dtype"]),
            compute_complex_dtype=cast(
                ComplexDTypeName,
                value["compute_complex_dtype"],
            ),
            reduction_dtype=cast(DTypeName, value["reduction_dtype"]),
            matmul_precision=cast(
                MatmulPrecisionName | None,
                value["matmul_precision"],
            ),
        )


@dataclass(frozen=True, slots=True)
class OperatorDTypePolicy:
    """Persistent parameters, transient compute, and reduction precision."""

    parameter_dtype: DTypeName = "float32"
    compute_dtype: DTypeName = "float32"
    reduction_dtype: DTypeName = "float32"
    matmul_precision: MatmulPrecisionName | None = None

    def __post_init__(self):
        for value in (
            self.parameter_dtype,
            self.compute_dtype,
            self.reduction_dtype,
        ):
            _dtype(value)
        precision = self.matmul_precision
        if precision is not None and precision not in _MATMUL_PRECISIONS:
            raise ValueError(f"Unsupported matmul precision {precision!r}.")
        required = _ALGORITHM_COMPUTE_DTYPES.get(precision)
        if required is not None and self.compute_dtype != required:
            raise ValueError(
                f"matmul_precision={precision!r} requires compute_dtype={required!r}."
            )

    @property
    def precision_evidence(self) -> OperatorPrecisionEvidence:
        return OperatorPrecisionEvidence(
            parameter_dtype=self.parameter_dtype,
            parameter_complex_dtype=_complex_dtype_name(self.parameter_dtype),
            compute_dtype=self.compute_dtype,
            compute_complex_dtype=_complex_dtype_name(self.compute_dtype),
            reduction_dtype=self.reduction_dtype,
            matmul_precision=self.matmul_precision,
        )

    def cast_parameters(self, parameters: Any, /) -> Any:
        """Cast persistent trainable parameters without touching fixed state."""
        return _cast_parameter_tree(parameters, _dtype(self.parameter_dtype))

    def cast_compute_parameters(self, parameters: Any, /) -> Any:
        """Create a transient compute-precision view of persistent parameters."""
        return _cast_parameter_tree(parameters, _dtype(self.compute_dtype))

    def cast_model(self, model: Any, /) -> Any:
        parameters, fixed = partition_trainable(model)
        return combine_trainable(self.cast_parameters(parameters), fixed)

    def compute_model(self, model: Any, /) -> Any:
        parameters, fixed = partition_trainable(model)
        return combine_trainable(self.cast_compute_parameters(parameters), fixed)

    def _samples(self, samples: FunctionSamples, /) -> FunctionSamples:
        dtype = _dtype(self.compute_dtype)
        values = None if samples.values is None else _cast_inexact(samples.values, dtype)
        return FunctionSamples(
            values=values,
            axes=samples.axes,
            coordinates=samples.coordinates,
            quadrature_weights=samples.quadrature_weights,
            mask=samples.mask,
            topology=samples.topology,
        )

    def cast_batch(self, batch: OperatorBatch, /) -> OperatorBatch:
        return OperatorBatch(
            inputs={
                name: self._samples(samples) for name, samples in batch.inputs.items()
            },
            queries={
                name: self._samples(samples) for name, samples in batch.queries.items()
            },
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    def cast_targets(
        self,
        targets: OperatorTargetBatch,
        /,
    ) -> OperatorTargetBatch:
        return targets.map_values(
            lambda value: _cast_inexact(value, _dtype(self.compute_dtype))
        )

    def reduction(self, value: Any, /) -> Array:
        return _cast_inexact(value, _dtype(self.reduction_dtype))

    def to_dict(self) -> dict[str, str | None]:
        return {
            "parameter_dtype": self.parameter_dtype,
            "compute_dtype": self.compute_dtype,
            "reduction_dtype": self.reduction_dtype,
            "matmul_precision": self.matmul_precision,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        /,
    ) -> OperatorDTypePolicy:
        expected = {
            "parameter_dtype",
            "compute_dtype",
            "reduction_dtype",
            "matmul_precision",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "Operator dtype policy must use the current canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return cls(
            parameter_dtype=cast(DTypeName, value["parameter_dtype"]),
            compute_dtype=cast(DTypeName, value["compute_dtype"]),
            reduction_dtype=cast(DTypeName, value["reduction_dtype"]),
            matmul_precision=cast(
                MatmulPrecisionName | None,
                value["matmul_precision"],
            ),
        )


__all__ = [
    "ComplexDTypeName",
    "DTypeName",
    "MatmulPrecisionName",
    "OperatorDTypePolicy",
    "OperatorPrecisionEvidence",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._precision import TensorNetworkPrecisionPolicy


def _chain_structure_id(
    kind: str,
    tensors: tuple[Array, ...],
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": f"{kind}-structure",
            "boundary": "open",
            "shapes": tuple(
                tuple(int(size) for size in tensor.shape) for tensor in tensors
            ),
            "dtype": str(tensors[0].dtype),
            "precision": precision.policy_id,
        }
    )


class MatrixProductState(StrictModule):
    tensors: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    site_count: int
    physical_dimensions: tuple[int, ...]
    bond_dimensions: tuple[int, ...]
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        values = tuple(precision_.storage(jnp.asarray(tensor)) for tensor in tensors)
        precision_.validate_storage(values)
        if not values or any(tensor.ndim != 3 for tensor in values):
            raise ValueError("MPS tensors must have shape (left, physical, right).")
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Open-boundary MPS edge bonds must be one.")
        for left, right in pairwise(values):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent MPS bond dimensions must match.")
        self.tensors = values
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(values)
        self.site_count = len(values)
        self.physical_dimensions = tuple(int(tensor.shape[1]) for tensor in values)
        self.bond_dimensions = tuple(int(tensor.shape[-1]) for tensor in values[:-1])
        self.structure_id = _chain_structure_id(
            "matrix-product-state", values, precision_
        )

    def _contract(self) -> Array:
        tensors = self.precision.contraction(self.tensors)
        state = tensors[0][0]
        for tensor in tensors[1:]:
            state = ein.contract("...l,lpr->...pr", state, tensor)
        return state[..., 0].reshape(-1)

    def to_dense(self, /, *, maximum_elements: int = 1_000_000) -> Array:
        count = 1
        for dimension in self.physical_dimensions:
            count *= dimension
        if int(maximum_elements) <= 0 or count > int(maximum_elements):
            raise ValueError(
                f"Dense MPS materialization requires {count} elements; "
                f"capacity is {int(maximum_elements)}."
            )
        return self.precision.output(self._contract())

    def inner(self, other: MatrixProductState, /) -> Array:
        from ._environments import mps_inner

        return mps_inner(self, other)

    def norm(self) -> Array:
        from ._environments import mps_norm_squared

        return self.precision.decision(jnp.sqrt(mps_norm_squared(self)))

    def normalized(self) -> MatrixProductState:
        norm = self.norm()
        norm = eqx.error_if(
            norm,
            ~jnp.isfinite(norm) | (norm <= 0.0),
            "MPS norm must be finite and positive.",
        )
        tensors = (self.tensors[0] / norm,) + self.tensors[1:]
        return MatrixProductState(tensors, precision=self.precision)


class MatrixProductOperator(StrictModule):
    tensors: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    site_count: int
    output_dimensions: tuple[int, ...]
    input_dimensions: tuple[int, ...]
    bond_dimensions: tuple[int, ...]
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        values = tuple(precision_.storage(jnp.asarray(tensor)) for tensor in tensors)
        precision_.validate_storage(values)
        if not values or any(tensor.ndim != 4 for tensor in values):
            raise ValueError("MPO tensors require (left, output, input, right).")
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Open-boundary MPO edge bonds must be one.")
        for left, right in pairwise(values):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent MPO bonds must match.")
        self.tensors = values
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(values)
        self.site_count = len(values)
        self.output_dimensions = tuple(int(tensor.shape[1]) for tensor in values)
        self.input_dimensions = tuple(int(tensor.shape[2]) for tensor in values)
        self.bond_dimensions = tuple(int(tensor.shape[-1]) for tensor in values[:-1])
        self.structure_id = _chain_structure_id(
            "matrix-product-operator", values, precision_
        )

    def to_dense(self, /, *, maximum_elements: int = 1_000_000) -> Array:
        count = prod(self.output_dimensions) * prod(self.input_dimensions)
        if int(maximum_elements) <= 0 or count > int(maximum_elements):
            raise ValueError(
                f"Dense MPO materialization requires {count} elements; "
                f"capacity is {int(maximum_elements)}."
            )
        tensors = self.precision.contraction(self.tensors)
        operator = tensors[0][0]
        for tensor in tensors[1:]:
            operator = ein.contract("...l,labr->...abr", operator, tensor)
        operator = operator[..., 0]
        output_axes = tuple(range(0, 2 * self.site_count, 2))
        input_axes = tuple(range(1, 2 * self.site_count, 2))
        operator = jnp.transpose(operator, output_axes + input_axes)
        return self.precision.output(
            operator.reshape((prod(self.output_dimensions), prod(self.input_dimensions)))
        )


class LocallyPurifiedDensity(StrictModule):
    tensors: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    site_count: int
    physical_dimensions: tuple[int, ...]
    purification_dimensions: tuple[int, ...]
    bond_dimensions: tuple[int, ...]
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        values = tuple(precision_.storage(jnp.asarray(tensor)) for tensor in tensors)
        precision_.validate_storage(values)
        if not values or any(tensor.ndim != 4 for tensor in values):
            raise ValueError(
                "Purification tensors require (left, physical, kraus, right)."
            )
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Purification edge bonds must be one.")
        for left, right in pairwise(values):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent purification bonds must match.")
        self.tensors = values
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(values)
        self.site_count = len(values)
        self.physical_dimensions = tuple(int(tensor.shape[1]) for tensor in values)
        self.purification_dimensions = tuple(int(tensor.shape[2]) for tensor in values)
        self.bond_dimensions = tuple(int(tensor.shape[-1]) for tensor in values[:-1])
        self.structure_id = _chain_structure_id(
            "locally-purified-density", values, precision_
        )

    def _amplitude(self) -> Array:
        tensors = self.precision.contraction(self.tensors)
        amplitude = tensors[0][0]
        for tensor in tensors[1:]:
            amplitude = ein.contract("...l,lpkr->...pkr", amplitude, tensor)
        amplitude = amplitude[..., 0]
        physical_axes = tuple(range(0, 2 * self.site_count, 2))
        kraus_axes = tuple(range(1, 2 * self.site_count, 2))
        amplitude = jnp.transpose(amplitude, physical_axes + kraus_axes)
        return amplitude.reshape(
            (prod(self.physical_dimensions), prod(self.purification_dimensions))
        )

    def raw_trace(self) -> Array:
        from ._environments import lpdo_raw_trace

        return self.precision.decision(lpdo_raw_trace(self))

    def to_dense_density(
        self,
        /,
        *,
        normalize: bool = False,
        maximum_elements: int = 1_000_000,
    ) -> Array:
        physical = prod(self.physical_dimensions)
        purification = prod(self.purification_dimensions)
        required = max(physical * purification, physical * physical)
        if int(maximum_elements) <= 0 or required > int(maximum_elements):
            raise ValueError(
                f"Dense LPDO materialization requires {required} workspace/output "
                f"elements; capacity is {int(maximum_elements)}."
            )
        amplitude = self.precision.contraction(self._amplitude())
        density = self.precision.accumulation(amplitude @ jnp.conj(amplitude.T))
        if not normalize:
            return self.precision.output(density)
        trace = self.precision.decision(jnp.real(self.precision.sum(jnp.diag(density))))
        trace = eqx.error_if(
            trace,
            ~jnp.isfinite(trace) | (trace <= 0.0),
            "LPDO trace must be finite and positive.",
        )
        return self.precision.output(density / trace)

    def normalized(self) -> LocallyPurifiedDensity:
        trace = self.raw_trace()
        trace = eqx.error_if(
            trace,
            ~jnp.isfinite(trace) | (trace <= 0.0),
            "LPDO trace must be finite and positive.",
        )
        tensors = (self.tensors[0] / jnp.sqrt(trace),) + self.tensors[1:]
        return LocallyPurifiedDensity(tensors, precision=self.precision)


__all__ = ["LocallyPurifiedDensity", "MatrixProductOperator", "MatrixProductState"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt
from typing import Literal, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.typing import DTypeLike
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.ein import contract
from phydrax.nn.operator.representations import O3Representation
from phydrax.nn.operator.representations._o3 import _tensor_basis


O3TensorProductConnectionMode = Literal["uvw"]
O3TensorProductNormalization = Literal["component"]


class _O3Block(NamedTuple):
    name: str
    degree: int
    parity: int
    dimension: int


class _O3TensorProductInstruction(NamedTuple):
    left_block: int
    right_block: int
    output_block: int
    left_degree: int
    right_degree: int
    output_degree: int
    left_parity: int
    right_parity: int
    output_parity: int
    left_multiplicity: int
    right_multiplicity: int
    output_multiplicity: int
    weight_offset: int
    weight_count: int
    multiply_add_count: int


_BLOCKS = (
    _O3Block("scalars", 0, 1, 1),
    _O3Block("pseudoscalars", 0, -1, 1),
    _O3Block("vectors", 1, -1, 3),
    _O3Block("pseudovectors", 1, 1, 3),
    _O3Block("tensors", 2, 1, 5),
    _O3Block("pseudotensors", 2, -1, 5),
)


def _multiplicities(representation: O3Representation, /) -> tuple[int, ...]:
    return (
        representation.scalars,
        representation.pseudoscalars,
        representation.vectors,
        representation.pseudovectors,
        representation.tensors,
        representation.pseudotensors,
    )


class O3TensorProductPlan(StrictModule, NonTrainableState):
    """Resource-bounded low-degree Cartesian O(3) tensor-product plan.

    The sole supported ``uvw`` connection mode gives every legal input/output
    multiplicity triple its own weight. ``component`` normalization makes each
    Clebsch--Gordan output component have unit coefficient norm. Degrees greater
    than two are outside this finite plan rather than being silently discarded.
    """

    left_representation: O3Representation
    right_representation: O3Representation
    output_representation: O3Representation
    instructions: tuple[_O3TensorProductInstruction, ...] = eqx.field(static=True)
    connection_mode: O3TensorProductConnectionMode = eqx.field(static=True)
    normalization: O3TensorProductNormalization = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    multiply_add_count: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    maximum_paths: int = eqx.field(static=True)
    maximum_parameters: int = eqx.field(static=True)
    maximum_multiply_adds: int = eqx.field(static=True)
    maximum_coefficients: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_representation: O3Representation,
        right_representation: O3Representation,
        output_representation: O3Representation,
        /,
        *,
        connection_mode: O3TensorProductConnectionMode = "uvw",
        normalization: O3TensorProductNormalization = "component",
        maximum_paths: int = 1_024,
        maximum_parameters: int = 10_000_000,
        maximum_multiply_adds: int = 2_000_000_000,
        maximum_coefficients: int = 1_000_000,
    ):
        if not isinstance(left_representation, O3Representation):
            raise TypeError("left_representation must be an O3Representation.")
        if not isinstance(right_representation, O3Representation):
            raise TypeError("right_representation must be an O3Representation.")
        if not isinstance(output_representation, O3Representation):
            raise TypeError("output_representation must be an O3Representation.")
        if connection_mode != "uvw":
            raise ValueError("Low-degree O(3) tensor products support only 'uvw'.")
        if normalization != "component":
            raise ValueError(
                "Low-degree O(3) tensor products support only component normalization."
            )
        limits = (
            int(maximum_paths),
            int(maximum_parameters),
            int(maximum_multiply_adds),
            int(maximum_coefficients),
        )
        if any(value < 0 for value in limits):
            raise ValueError("O(3) tensor-product resource limits must be non-negative.")

        left_counts = _multiplicities(left_representation)
        right_counts = _multiplicities(right_representation)
        output_counts = _multiplicities(output_representation)
        instructions: list[_O3TensorProductInstruction] = []
        weight_offset = 0
        multiply_add_count = 0
        coefficient_count = 0
        for left_index, (left_block, left_count) in enumerate(
            zip(_BLOCKS, left_counts, strict=True)
        ):
            if left_count == 0:
                continue
            for right_index, (right_block, right_count) in enumerate(
                zip(_BLOCKS, right_counts, strict=True)
            ):
                if right_count == 0:
                    continue
                for output_index, (output_block, output_count) in enumerate(
                    zip(_BLOCKS, output_counts, strict=True)
                ):
                    if output_count == 0:
                        continue
                    degree_is_legal = (
                        abs(left_block.degree - right_block.degree)
                        <= output_block.degree
                        <= min(left_block.degree + right_block.degree, 2)
                    )
                    parity_is_legal = (
                        output_block.parity == left_block.parity * right_block.parity
                    )
                    if not degree_is_legal or not parity_is_legal:
                        continue
                    weight_count = left_count * right_count * output_count
                    instruction_cost = (
                        weight_count
                        * left_block.dimension
                        * right_block.dimension
                        * output_block.dimension
                    )
                    instructions.append(
                        _O3TensorProductInstruction(
                            left_block=left_index,
                            right_block=right_index,
                            output_block=output_index,
                            left_degree=left_block.degree,
                            right_degree=right_block.degree,
                            output_degree=output_block.degree,
                            left_parity=left_block.parity,
                            right_parity=right_block.parity,
                            output_parity=output_block.parity,
                            left_multiplicity=left_count,
                            right_multiplicity=right_count,
                            output_multiplicity=output_count,
                            weight_offset=weight_offset,
                            weight_count=weight_count,
                            multiply_add_count=instruction_cost,
                        )
                    )
                    weight_offset += weight_count
                    multiply_add_count += instruction_cost
                    coefficient_count += (
                        left_block.dimension
                        * right_block.dimension
                        * output_block.dimension
                    )
        if not instructions:
            raise ValueError(
                "The requested O(3) layouts have no legal degree/parity tensor-product path."
            )
        evidence = (
            len(instructions),
            weight_offset,
            multiply_add_count,
            coefficient_count,
        )
        labels = ("paths", "parameters", "multiply-adds", "coefficients")
        for observed, allowed, label in zip(evidence, limits, labels, strict=True):
            if observed > allowed:
                raise ValueError(
                    f"O(3) tensor-product plan requires {observed} {label}, "
                    f"exceeding the declared limit {allowed}."
                )

        instruction_data = [instruction._asdict() for instruction in instructions]
        plan_id = canonical_fingerprint(
            {
                "kind": "low-degree-o3-tensor-product-plan",
                "left": left_counts,
                "right": right_counts,
                "output": output_counts,
                "connection_mode": connection_mode,
                "normalization": normalization,
                "instructions": instruction_data,
                "path_count": evidence[0],
                "parameter_count": evidence[1],
                "multiply_add_count": evidence[2],
                "coefficient_count": evidence[3],
                "maximum_paths": limits[0],
                "maximum_parameters": limits[1],
                "maximum_multiply_adds": limits[2],
                "maximum_coefficients": limits[3],
            }
        )
        self.left_representation = left_representation
        self.right_representation = right_representation
        self.output_representation = output_representation
        self.instructions = tuple(instructions)
        self.connection_mode = connection_mode
        self.normalization = normalization
        self.path_count = evidence[0]
        self.parameter_count = evidence[1]
        self.multiply_add_count = evidence[2]
        self.coefficient_count = evidence[3]
        self.maximum_paths = limits[0]
        self.maximum_parameters = limits[1]
        self.maximum_multiply_adds = limits[2]
        self.maximum_coefficients = limits[3]
        self.plan_id = plan_id

    @property
    def content_id(self) -> str:
        """Canonical identity of layouts, instructions, normalization, and budgets."""

        return self.plan_id

    @property
    def resource_evidence(self) -> dict[str, int]:
        """Exact scalar-work and allocation evidence resolved before preparation."""

        return {
            "path_count": self.path_count,
            "parameter_count": self.parameter_count,
            "multiply_add_count": self.multiply_add_count,
            "coefficient_count": self.coefficient_count,
        }


def _levi_civita(dtype: jnp.dtype, /) -> Array:
    values = jnp.zeros((3, 3, 3), dtype=dtype)
    values = values.at[0, 1, 2].set(1.0)
    values = values.at[1, 2, 0].set(1.0)
    values = values.at[2, 0, 1].set(1.0)
    values = values.at[0, 2, 1].set(-1.0)
    values = values.at[2, 1, 0].set(-1.0)
    return values.at[1, 0, 2].set(-1.0)


def _component_normalize(coefficients: Array, /) -> Array:
    norm = jnp.sqrt(jnp.sum(coefficients[0] * coefficients[0]))
    return coefficients / norm


def _clebsch_gordan(
    left_degree: int,
    right_degree: int,
    output_degree: int,
    dtype: jnp.dtype,
    /,
) -> Array:
    dimensions = (1, 3, 5)
    left_dimension = dimensions[left_degree]
    right_dimension = dimensions[right_degree]
    output_dimension = dimensions[output_degree]
    if left_degree == 0 and output_degree == right_degree:
        return jnp.eye(output_dimension, dtype=dtype)[:, None, :]
    if right_degree == 0 and output_degree == left_degree:
        return jnp.eye(output_dimension, dtype=dtype)[:, :, None]

    identity = jnp.eye(3, dtype=dtype)
    epsilon = _levi_civita(dtype)
    tensor_basis = _tensor_basis(dtype)
    if (left_degree, right_degree, output_degree) == (1, 1, 0):
        return identity[None, :, :] / sqrt(3.0)
    if (left_degree, right_degree, output_degree) == (1, 1, 1):
        return epsilon / sqrt(2.0)
    if (left_degree, right_degree, output_degree) == (1, 1, 2):
        return tensor_basis
    if (left_degree, right_degree, output_degree) == (1, 2, 1):
        coefficients = contract("qij->ijq", tensor_basis)
        return _component_normalize(coefficients)
    if (left_degree, right_degree, output_degree) == (2, 1, 1):
        coefficients = contract("qij->iqj", tensor_basis)
        return _component_normalize(coefficients)
    if (left_degree, right_degree, output_degree) == (1, 2, 2):
        coefficients = contract("kij,iab,qbj->kaq", tensor_basis, epsilon, tensor_basis)
        return _component_normalize(coefficients)
    if (left_degree, right_degree, output_degree) == (2, 1, 2):
        coefficients = -contract("kij,iab,qbj->kqa", tensor_basis, epsilon, tensor_basis)
        return _component_normalize(coefficients)
    if (left_degree, right_degree, output_degree) == (2, 2, 0):
        return jnp.eye(5, dtype=dtype)[None, :, :] / sqrt(5.0)
    if (left_degree, right_degree, output_degree) == (2, 2, 1):
        coefficients = contract("iab,pac,qcb->ipq", epsilon, tensor_basis, tensor_basis)
        return _component_normalize(coefficients)
    if (left_degree, right_degree, output_degree) == (2, 2, 2):
        coefficients = 0.5 * (
            contract("kij,pic,qcj->kpq", tensor_basis, tensor_basis, tensor_basis)
            + contract("kij,qic,pcj->kpq", tensor_basis, tensor_basis, tensor_basis)
        )
        return _component_normalize(coefficients)
    raise ValueError(
        "Illegal low-degree Clebsch--Gordan path "
        f"({left_degree}, {right_degree}) -> {output_degree}."
    )


class _PreparedClebschGordan(StrictModule, NonTrainableState):
    coefficients: tuple[Array, ...]


class O3TensorProduct(StrictModule):
    """Prepared weighted low-degree O(3) tensor product in Cartesian components."""

    plan: O3TensorProductPlan
    prepared: _PreparedClebschGordan
    weight: Array | None
    internal_weights: bool = eqx.field(static=True)

    def __init__(
        self,
        plan: O3TensorProductPlan,
        /,
        *,
        internal_weights: bool = True,
        dtype: DTypeLike = jnp.float64,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(plan, O3TensorProductPlan):
            raise TypeError("plan must be an O3TensorProductPlan.")
        dtype_ = jnp.dtype(dtype)
        prepared = tuple(
            _clebsch_gordan(
                instruction.left_degree,
                instruction.right_degree,
                instruction.output_degree,
                dtype_,
            )
            for instruction in plan.instructions
        )
        if internal_weights:
            pieces = []
            keys = jr.split(key, plan.path_count)
            for instruction, path_key in zip(plan.instructions, keys, strict=True):
                scale = 1.0 / sqrt(
                    float(instruction.left_multiplicity * instruction.right_multiplicity)
                )
                pieces.append(
                    scale * jr.normal(path_key, (instruction.weight_count,), dtype=dtype_)
                )
            weight = jnp.concatenate(pieces)
        else:
            weight = None
        self.plan = plan
        self.prepared = _PreparedClebschGordan(coefficients=prepared)
        self.weight = weight
        self.internal_weights = bool(internal_weights)

    @staticmethod
    def _blocks(representation: O3Representation, values: Array, /) -> tuple[Array, ...]:
        counts = _multiplicities(representation)
        offset = 0
        blocks = []
        for count, block in zip(counts, _BLOCKS, strict=True):
            size = count * block.dimension
            blocks.append(
                values[..., offset : offset + size].reshape(
                    values.shape[:-1] + (count, block.dimension)
                )
            )
            offset += size
        return tuple(blocks)

    def __call__(
        self,
        left: Array,
        right: Array,
        path_weights: Array | None = None,
        /,
    ) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if int(left_.shape[-1]) != self.plan.left_representation.packed_size:
            raise ValueError("Left values do not match the planned O(3) layout.")
        if int(right_.shape[-1]) != self.plan.right_representation.packed_size:
            raise ValueError("Right values do not match the planned O(3) layout.")
        if left_.shape[:-1] != right_.shape[:-1]:
            raise ValueError(
                "O(3) tensor-product inputs must have identical leading axes."
            )
        if path_weights is None:
            if self.weight is None:
                raise ValueError(
                    "Externally weighted O(3) tensor product needs path weights."
                )
            weights = self.weight
        else:
            if self.weight is not None:
                raise ValueError(
                    "Path weights cannot override an internally weighted O(3) tensor product."
                )
            weights = jnp.asarray(path_weights)
        leading = left_.shape[:-1]
        if weights.shape == (self.plan.parameter_count,):
            weights = jnp.broadcast_to(weights, leading + weights.shape)
        elif weights.shape != leading + (self.plan.parameter_count,):
            raise ValueError(
                "O(3) path weights must have the input leading axes and planned "
                "parameter count."
            )
        left_blocks = self._blocks(self.plan.left_representation, left_)
        right_blocks = self._blocks(self.plan.right_representation, right_)
        output_counts = _multiplicities(self.plan.output_representation)
        output_blocks = [
            jnp.zeros(
                leading + (count, block.dimension),
                dtype=jnp.result_type(left_, right_, weights),
            )
            for count, block in zip(output_counts, _BLOCKS, strict=True)
        ]
        for instruction, coefficients in zip(
            self.plan.instructions, self.prepared.coefficients, strict=True
        ):
            start = instruction.weight_offset
            stop = start + instruction.weight_count
            instruction_weights = weights[..., start:stop].reshape(
                leading
                + (
                    instruction.output_multiplicity,
                    instruction.left_multiplicity,
                    instruction.right_multiplicity,
                )
            )
            contribution = contract(
                "...ui,...vj,oij,...wuv->...wo",
                left_blocks[instruction.left_block],
                right_blocks[instruction.right_block],
                coefficients,
                instruction_weights,
            )
            output_blocks[instruction.output_block] = (
                output_blocks[instruction.output_block] + contribution
            )
        return jnp.concatenate(
            [
                block.reshape(leading + (count * block_type.dimension,))
                for block, count, block_type in zip(
                    output_blocks,
                    output_counts,
                    _BLOCKS,
                    strict=True,
                )
            ],
            axis=-1,
        )


__all__ = ["O3TensorProduct", "O3TensorProductPlan"]

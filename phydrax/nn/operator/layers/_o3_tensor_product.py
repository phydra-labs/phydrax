#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike, DTypeLike

from phydrax._array_archive import array_collection_digest
from phydrax._doc import DOC_KEY0
from phydrax._fingerprint import canonical_fingerprint
from phydrax._model import register_artifact_value
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax._validation import (
    canonical_identifier,
    finite_real_scalar,
    nonnegative_integer,
)
from phydrax.ein import contract
from phydrax.nn.operator.representations import (
    o3_real_coupling,
    O3IrrepBlock,
    O3IrrepLayout,
    O3Representation,
)
from phydrax.nn.operator.representations._o3 import _tensor_basis

from ....typing import AnyShape, as_host_array, checked, HostFloat64, parse, PRNGKey


O3TensorProductConnectionMode: TypeAlias = Literal["uvw", "uvu"]
O3TensorProductNormalization: TypeAlias = Literal["component"]
type O3TensorProductLayout = O3Representation | O3IrrepLayout


@final
class O3TensorProductPath(StrictModule):
    """One declared coupling path between named blocks of three layouts.

    ``"uvw"`` gives every ``(output w, left u, right v)`` multiplicity triple its
    own weight, stored ``(w, u, v)``. ``"uvu"`` keeps the left channel,
    ``out[u] = sum_v weight[u, v] C(left[u], right[v])``, stores weights
    ``(u, v)``, and requires equal left and output multiplicities. Unweighted
    paths exist only for ``"uvu"``, where they sum over ``v``. ``path_scale``
    multiplies the component-normalized coupling, for example an imported
    source path normalization and basis sign.
    """

    left: str = eqx.field(static=True)
    right: str = eqx.field(static=True)
    output: str = eqx.field(static=True)
    connection_mode: O3TensorProductConnectionMode = eqx.field(static=True)
    weighted: bool = eqx.field(static=True)
    path_scale: float = eqx.field(static=True)

    def __init__(
        self,
        left: str,
        right: str,
        output: str,
        /,
        *,
        connection_mode: O3TensorProductConnectionMode = "uvw",
        weighted: bool = True,
        path_scale: float = 1.0,
    ) -> None:
        left_ = canonical_identifier(left, "left block")
        right_ = canonical_identifier(right, "right block")
        output_ = canonical_identifier(output, "output block")
        mode = parse(connection_mode, O3TensorProductConnectionMode, "connection_mode")
        if not isinstance(weighted, bool):
            raise TypeError("weighted must be a bool.")
        scale = finite_real_scalar(path_scale, "path_scale")
        if scale == 0.0:
            raise ValueError("path_scale must be nonzero; omit the path instead.")
        if mode == "uvw" and not weighted:
            raise ValueError("'uvw' paths map multiplicities through their weights.")
        self.left = left_
        self.right = right_
        self.output = output_
        self.connection_mode = mode
        self.weighted = weighted
        self.path_scale = scale


class _O3Block(NamedTuple):
    name: str
    degree: int
    parity: int
    multiplicity: int
    dimension: int
    offset: int


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
    connection_mode: str
    weighted: bool
    path_scale: float
    weight_offset: int
    weight_count: int
    multiply_add_count: int
    working_set: int


def _layout_blocks(layout: O3TensorProductLayout, /) -> tuple[_O3Block, ...]:
    declared: tuple[O3IrrepBlock, ...] = (
        layout.irrep_blocks if isinstance(layout, O3Representation) else layout.blocks
    )
    blocks: list[_O3Block] = []
    offset = 0
    for block in declared:
        blocks.append(
            _O3Block(
                block.name,
                block.degree,
                block.parity,
                block.multiplicity,
                block.dimension,
                offset,
            )
        )
        offset += block.size
    return tuple(blocks)


def _is_legal(left: _O3Block, right: _O3Block, output: _O3Block, /) -> bool:
    return (
        abs(left.degree - right.degree) <= output.degree <= left.degree + right.degree
        and output.parity == left.parity * right.parity
    )


def _enumerated_paths(
    left: tuple[_O3Block, ...],
    right: tuple[_O3Block, ...],
    output: tuple[_O3Block, ...],
    connection_mode: O3TensorProductConnectionMode,
    maximum_paths: int,
    /,
) -> tuple[O3TensorProductPath, ...]:
    paths: list[O3TensorProductPath] = []
    for left_block in left:
        for right_block in right:
            for output_block in output:
                if not _is_legal(left_block, right_block, output_block):
                    continue
                if len(paths) == maximum_paths:
                    raise ValueError(
                        f"O(3) tensor-product plan requires more than {maximum_paths} paths, exceeding the declared limit."
                    )
                paths.append(
                    O3TensorProductPath(
                        left_block.name,
                        right_block.name,
                        output_block.name,
                        connection_mode=connection_mode,
                    )
                )
    return tuple(paths)


def _declared_paths(
    paths: Sequence[O3TensorProductPath], /
) -> tuple[O3TensorProductPath, ...]:
    resolved = tuple(paths)
    if any(not isinstance(path, O3TensorProductPath) for path in resolved):
        raise TypeError("paths must contain O3TensorProductPath values.")
    keys = [
        (path.left, path.right, path.output, path.connection_mode) for path in resolved
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Repeated O(3) tensor-product paths declare a redundant parameterization."
        )
    return resolved


def _block_index(blocks: tuple[_O3Block, ...], name: str, role: str, /) -> int:
    for index, block in enumerate(blocks):
        if block.name == name:
            return index
    raise ValueError(f"The {role} layout has no block named {name!r}.")


def _instruction(
    path: O3TensorProductPath,
    left: tuple[_O3Block, ...],
    right: tuple[_O3Block, ...],
    output: tuple[_O3Block, ...],
    weight_offset: int,
    /,
) -> _O3TensorProductInstruction:
    left_index = _block_index(left, path.left, "left")
    right_index = _block_index(right, path.right, "right")
    output_index = _block_index(output, path.output, "output")
    a, b, c = left[left_index], right[right_index], output[output_index]
    if not _is_legal(a, b, c):
        raise ValueError(
            f"Path {path.left!r} x {path.right!r} -> {path.output!r} violates the "
            "O(3) degree triangle or parity product."
        )
    # Staged execution: (left, coupling) -> (right) -> (weights).
    staged = a.multiplicity * c.dimension * b.dimension * (a.dimension + b.multiplicity)
    match path.connection_mode:
        case "uvw":
            weight_count = c.multiplicity * a.multiplicity * b.multiplicity
            final_work = weight_count * c.dimension
        case "uvu":
            if c.multiplicity != a.multiplicity:
                raise ValueError(
                    f"'uvu' path {path.left!r} x {path.right!r} -> {path.output!r} "
                    "requires equal left and output multiplicities."
                )
            weight_count = a.multiplicity * b.multiplicity if path.weighted else 0
            final_work = a.multiplicity * b.multiplicity * c.dimension
        case _:
            assert_never(path.connection_mode)
    working_set = max(
        a.multiplicity * c.dimension * b.dimension,
        a.multiplicity * b.multiplicity * c.dimension,
        c.multiplicity * c.dimension,
    )
    return _O3TensorProductInstruction(
        left_block=left_index,
        right_block=right_index,
        output_block=output_index,
        left_degree=a.degree,
        right_degree=b.degree,
        output_degree=c.degree,
        left_parity=a.parity,
        right_parity=b.parity,
        output_parity=c.parity,
        left_multiplicity=a.multiplicity,
        right_multiplicity=b.multiplicity,
        output_multiplicity=c.multiplicity,
        connection_mode=path.connection_mode,
        weighted=path.weighted,
        path_scale=path.path_scale,
        weight_offset=weight_offset,
        weight_count=weight_count,
        multiply_add_count=staged + final_work,
        working_set=working_set,
    )


def _component_basis(
    left: O3TensorProductLayout,
    right: O3TensorProductLayout,
    output: O3TensorProductLayout,
    /,
) -> Literal["cartesian", "real-spherical"]:
    kinds = {isinstance(layout, O3Representation) for layout in (left, right, output)}
    if len(kinds) != 1:
        raise ValueError(
            "O(3) tensor-product layouts must all be Cartesian O3Representation or all "
            "general O3IrrepLayout; their component bases are not interchangeable."
        )
    return "cartesian" if kinds == {True} else "real-spherical"


@final
class O3TensorProductPlan(StrictModule, NonTrainableState):
    """Resource-bounded O(3) tensor-product plan over one native layout kind.

    Layouts are either all Cartesian `O3Representation` physical fields (degrees
    zero to two, with their independently derived analytic couplings) or all
    general `O3IrrepLayout` real-harmonic layouts of arbitrary degree (exact
    real couplings from `o3_real_coupling`). Without explicit ``paths``, every
    legal degree/parity block triple is enumerated left-major with one
    ``connection_mode`` (``"uvw"`` by default). ``component`` normalization
    gives every coupling output component unit coefficient norm; each path then
    multiplies by its ``path_scale``.

    Path count, weights, dense coefficient storage, staged multiply-adds, and the
    largest per-leading-element staged intermediate (``working_set``) are
    resolved and checked against the declared limits before any coefficient or
    weight allocation.
    """

    left_representation: O3Representation | O3IrrepLayout
    right_representation: O3Representation | O3IrrepLayout
    output_representation: O3Representation | O3IrrepLayout
    paths: tuple[O3TensorProductPath, ...]
    instructions: tuple[_O3TensorProductInstruction, ...] = eqx.field(static=True)
    component_basis: Literal["cartesian", "real-spherical"] = eqx.field(static=True)
    normalization: O3TensorProductNormalization = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    multiply_add_count: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    working_set: int = eqx.field(static=True)
    maximum_paths: int = eqx.field(static=True)
    maximum_parameters: int = eqx.field(static=True)
    maximum_multiply_adds: int = eqx.field(static=True)
    maximum_coefficients: int = eqx.field(static=True)
    maximum_working_set: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        left_representation: O3Representation | O3IrrepLayout,
        right_representation: O3Representation | O3IrrepLayout,
        output_representation: O3Representation | O3IrrepLayout,
        /,
        *,
        paths: Sequence[O3TensorProductPath] | None = None,
        connection_mode: O3TensorProductConnectionMode | None = None,
        normalization: O3TensorProductNormalization = "component",
        maximum_paths: int = 1_024,
        maximum_parameters: int = 10_000_000,
        maximum_multiply_adds: int = 2_000_000_000,
        maximum_coefficients: int = 1_000_000,
        maximum_working_set: int = 100_000_000,
    ) -> None:
        normalization_ = parse(
            normalization, O3TensorProductNormalization, "normalization"
        )
        limits = (
            nonnegative_integer(maximum_paths, "maximum_paths"),
            nonnegative_integer(maximum_parameters, "maximum_parameters"),
            nonnegative_integer(maximum_multiply_adds, "maximum_multiply_adds"),
            nonnegative_integer(maximum_coefficients, "maximum_coefficients"),
            nonnegative_integer(maximum_working_set, "maximum_working_set"),
        )
        basis = _component_basis(
            left_representation, right_representation, output_representation
        )
        left = _layout_blocks(left_representation)
        right = _layout_blocks(right_representation)
        output = _layout_blocks(output_representation)
        if paths is None:
            mode = parse(
                "uvw" if connection_mode is None else connection_mode,
                O3TensorProductConnectionMode,
                "connection_mode",
            )
            resolved_paths = _enumerated_paths(left, right, output, mode, limits[0])
        elif connection_mode is not None:
            raise ValueError(
                "Explicit O(3) tensor-product paths declare their own connection modes."
            )
        else:
            resolved_paths = _declared_paths(paths)
        if not resolved_paths:
            raise ValueError(
                "The requested O(3) layouts have no legal degree/parity tensor-product path."
            )
        if len(resolved_paths) > limits[0]:
            raise ValueError(
                f"O(3) tensor-product plan requires {len(resolved_paths)} paths, exceeding the declared limit {limits[0]}."
            )
        instructions: list[_O3TensorProductInstruction] = []
        weight_offset = 0
        for path in resolved_paths:
            instruction = _instruction(path, left, right, output, weight_offset)
            instructions.append(instruction)
            weight_offset += instruction.weight_count
        evidence = (
            len(instructions),
            weight_offset,
            sum(instruction.multiply_add_count for instruction in instructions),
            sum(
                (2 * instruction.left_degree + 1)
                * (2 * instruction.right_degree + 1)
                * (2 * instruction.output_degree + 1)
                for instruction in instructions
            ),
            max(instruction.working_set for instruction in instructions),
        )
        labels = (
            "paths",
            "parameters",
            "multiply-adds",
            "coefficients",
            "working-set elements",
        )
        for observed, allowed, label in zip(evidence, limits, labels, strict=True):
            if observed > allowed:
                raise ValueError(
                    f"O(3) tensor-product plan requires {observed} {label}, exceeding the declared limit {allowed}."
                )

        plan_id = canonical_fingerprint(
            {
                "kind": "o3-tensor-product-plan",
                "component_basis": basis,
                "left": left_representation.layout_id,
                "right": right_representation.layout_id,
                "output": output_representation.layout_id,
                "normalization": normalization_,
                "instructions": [instruction._asdict() for instruction in instructions],
                "path_count": evidence[0],
                "parameter_count": evidence[1],
                "multiply_add_count": evidence[2],
                "coefficient_count": evidence[3],
                "working_set": evidence[4],
                "maximum_paths": limits[0],
                "maximum_parameters": limits[1],
                "maximum_multiply_adds": limits[2],
                "maximum_coefficients": limits[3],
                "maximum_working_set": limits[4],
            }
        )
        self.left_representation = left_representation
        self.right_representation = right_representation
        self.output_representation = output_representation
        self.paths = resolved_paths
        self.instructions = tuple(instructions)
        self.component_basis = basis
        self.normalization = normalization_
        self.path_count = evidence[0]
        self.parameter_count = evidence[1]
        self.multiply_add_count = evidence[2]
        self.coefficient_count = evidence[3]
        self.working_set = evidence[4]
        self.maximum_paths = limits[0]
        self.maximum_parameters = limits[1]
        self.maximum_multiply_adds = limits[2]
        self.maximum_coefficients = limits[3]
        self.maximum_working_set = limits[4]
        self.plan_id = plan_id

    @property
    def content_id(self) -> str:
        """Canonical identity of layouts, paths, normalization, and budgets."""

        return self.plan_id

    @property
    def resource_evidence(self) -> dict[str, int]:
        """Exact scalar-work and allocation evidence resolved before preparation."""

        return {
            "path_count": self.path_count,
            "parameter_count": self.parameter_count,
            "multiply_add_count": self.multiply_add_count,
            "coefficient_count": self.coefficient_count,
            "working_set": self.working_set,
        }

    @property
    def path_weight_layout(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Per-path ``(offset, shape)`` inside the flat path-weight axis.

        Shapes are ``(w, u, v)`` for ``"uvw"``, ``(u, v)`` for weighted ``"uvu"``,
        and ``()`` with no entries for unweighted paths; flat weights are
        path-major and row-major within each path.
        """

        return tuple(
            (instruction.weight_offset, _weight_shape(instruction))
            for instruction in self.instructions
        )


def _weight_shape(instruction: _O3TensorProductInstruction, /) -> tuple[int, ...]:
    if not instruction.weighted:
        return ()
    if instruction.connection_mode == "uvw":
        return (
            instruction.output_multiplicity,
            instruction.left_multiplicity,
            instruction.right_multiplicity,
        )
    return (instruction.left_multiplicity, instruction.right_multiplicity)


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
    dimensions[left_degree]
    dimensions[right_degree]
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
        f"Illegal low-degree Clebsch--Gordan path ({left_degree}, {right_degree}) -> {output_degree}."
    )


def _path_coefficients(
    plan: O3TensorProductPlan,
    instruction: _O3TensorProductInstruction,
    dtype: jnp.dtype,
    /,
) -> Array:
    degrees = (
        instruction.left_degree,
        instruction.right_degree,
        instruction.output_degree,
    )
    match plan.component_basis:
        case "cartesian":
            coefficients = _clebsch_gordan(*degrees, dtype)
        case "real-spherical":
            coefficients = jnp.asarray(o3_real_coupling(*degrees).dense(), dtype=dtype)
        case _:
            assert_never(plan.component_basis)
    if instruction.path_scale == 1.0:
        return coefficients
    return instruction.path_scale * coefficients


class _PreparedClebschGordan(StrictModule, NonTrainableState):
    coefficients: tuple[Array, ...]


O3TensorProductCoefficientOrigin: TypeAlias = Literal["native", "imported"]


def _imported_coefficients(
    plan: O3TensorProductPlan,
    coefficients: Sequence[ArrayLike],
    tolerance: float | None,
    dtype: jnp.dtype,
    /,
) -> tuple[tuple[Array, ...], float]:
    """Bind source coefficient tables after checking them against the native coupling.

    Imported tables carry source rounding (for example float32-rounded coupling
    buffers). They are admitted only within an explicit absolute tolerance of
    ``path_scale`` times the native component-normalized coupling, entry by
    entry, including entries where the native coupling is exactly zero.
    """
    if tolerance is None:
        raise ValueError(
            "Imported O(3) coefficients require an explicit coefficient_tolerance."
        )
    tolerance_ = finite_real_scalar(tolerance, "coefficient_tolerance")
    if tolerance_ < 0.0:
        raise ValueError("coefficient_tolerance must be nonnegative.")
    tables = tuple(coefficients)
    if len(tables) != plan.path_count:
        raise ValueError(
            f"Imported O(3) coefficients need one table per path ({plan.path_count}); got {len(tables)}."
        )
    bound: list[Array] = []
    for index, (instruction, table) in enumerate(
        zip(plan.instructions, tables, strict=True)
    ):
        if isinstance(table, jax_core.Tracer):
            raise TypeError("Imported O(3) coefficients must be concrete host data.")
        # One owner conversion: real kinds cast to float64, complex kinds are
        # refused instead of silently discarding imaginary parts.
        values = as_host_array(
            table, HostFloat64[AnyShape], f"Imported coefficients of path {index}"
        )
        native = np.asarray(_path_coefficients(plan, instruction, jnp.dtype(jnp.float64)))
        if values.shape != native.shape:
            raise ValueError(
                f"Imported coefficients of path {index} must have output-major shape "
                f"{native.shape}; got {values.shape}."
            )
        deviation = float(np.max(np.abs(values - native)))
        if not deviation <= tolerance_:
            raise ValueError(
                f"Imported coefficients of path {index} deviate from the native coupling "
                f"by {deviation}, exceeding coefficient_tolerance {tolerance_}."
            )
        bound.append(jnp.asarray(values, dtype=dtype))
    return tuple(bound), tolerance_


_CoefficientSupport: TypeAlias = tuple[tuple[tuple[int, int, int], ...], ...]


def _coefficient_support(coefficients: tuple[Array, ...], /) -> _CoefficientSupport:
    """Output-major ``(o, i, j)`` nonzeros of each executed table, row-major order.

    This is a host boundary over the concrete tables in their executed dtype,
    so underflow of a cast coefficient is reflected honestly.
    """
    support = []
    for table in coefficients:
        if isinstance(table, jax_core.Tracer):
            raise TypeError("O(3) coefficient support is read from concrete tables only.")
        host = np.asarray(jax.device_get(table))
        support.append(
            tuple((int(o), int(i), int(j)) for o, i, j in np.argwhere(host != 0).tolist())
        )
    return tuple(support)


def _coefficient_id(
    plan: O3TensorProductPlan,
    coefficients: tuple[Array, ...],
    support: _CoefficientSupport,
    origin: O3TensorProductCoefficientOrigin,
    tolerance: float | None,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "o3-tensor-product-coefficients",
            "plan_id": plan.plan_id,
            "origin": origin,
            "tolerance": tolerance,
            "support": [[list(entry) for entry in path] for path in support],
            "values": array_collection_digest(
                {
                    f"path-{index:06d}": np.asarray(table)
                    for index, table in enumerate(coefficients)
                }
            ),
        }
    )


@final
class O3TensorProduct(StrictModule):
    """Prepared weighted O(3) tensor product over one planned layout kind.

    Coefficients are dense output-major tables ``C[o, i, j]`` (``path_scale``
    folded in). Each path executes in a fixed staged order whose largest
    intermediate is the plan ``working_set`` per leading element.

    By default the tables are the native couplings. ``coefficients`` binds
    explicit source tables instead, one per plan path in the same shape and
    with ``path_scale`` included; each must be real and lie within the declared
    ``coefficient_tolerance`` of the native table. ``coefficient_support`` lists,
    per plan path, the output-major ``(o, i, j)`` positions where the executed
    (dtype-cast) table is nonzero, in row-major order; support-routed consumers
    execute exactly these entries. ``coefficient_id`` identifies the executed
    tables, their support, their origin, and the admitted tolerance. `validate`
    re-admits restored or transformed values.
    """

    plan: O3TensorProductPlan
    prepared: _PreparedClebschGordan
    weight: Array | None
    internal_weights: bool = eqx.field(static=True)
    coefficient_origin: O3TensorProductCoefficientOrigin = eqx.field(static=True)
    coefficient_tolerance: float | None = eqx.field(static=True)
    coefficient_support: _CoefficientSupport = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: O3TensorProductPlan,
        /,
        *,
        internal_weights: bool = True,
        dtype: DTypeLike = jnp.float64,
        coefficients: Sequence[ArrayLike] | None = None,
        coefficient_tolerance: float | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        if not isinstance(plan, O3TensorProductPlan):
            raise TypeError("plan must be an O3TensorProductPlan.")
        dtype_ = jnp.dtype(dtype)
        if coefficients is None:
            if coefficient_tolerance is not None:
                raise ValueError(
                    "coefficient_tolerance applies only to imported coefficients."
                )
            origin: O3TensorProductCoefficientOrigin = "native"
            tolerance = None
            prepared = tuple(
                _path_coefficients(plan, instruction, dtype_)
                for instruction in plan.instructions
            )
        else:
            origin = "imported"
            prepared, tolerance = _imported_coefficients(
                plan, coefficients, coefficient_tolerance, dtype_
            )
        if internal_weights:
            pieces = [jnp.zeros((0,), dtype=dtype_)]
            keys = jr.split(key, plan.path_count)
            for instruction, path_key in zip(plan.instructions, keys, strict=True):
                if not instruction.weighted:
                    continue
                fan_in = instruction.right_multiplicity * (
                    instruction.left_multiplicity
                    if instruction.connection_mode == "uvw"
                    else 1
                )
                scale = 1.0 / sqrt(float(fan_in))
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
        self.coefficient_origin = origin
        self.coefficient_tolerance = tolerance
        self.coefficient_support = _coefficient_support(prepared)
        self.coefficient_id = _coefficient_id(
            plan, prepared, self.coefficient_support, origin, tolerance
        )

    def validate(self) -> None:
        """Re-admit the executed coefficient tables, their support and identity.

        This is a host boundary over concrete tables. Raw restoration and
        transformations such as `equinox.tree_at` bypass the constructor, so the
        tables are re-bound through it: native tables must equal the native
        coupling in their dtype, imported tables must pass their recorded
        tolerance, and the support and ``coefficient_id`` recomputed from the
        current tables must match the recorded ones. A table entry that became
        nonzero outside `coefficient_support` is refused rather than dropped;
        prepare the tensor product again.
        """
        if not isinstance(self.plan, O3TensorProductPlan):
            raise TypeError("plan must be an O3TensorProductPlan.")
        if not isinstance(self.prepared, _PreparedClebschGordan):
            raise TypeError("prepared must hold the bound O(3) coefficient tables.")
        tables = tuple(self.prepared.coefficients)
        if len(tables) != self.plan.path_count:
            raise ValueError(
                f"O(3) tensor product holds {len(tables)} coefficient tables for "
                f"{self.plan.path_count} plan paths."
            )
        if any(isinstance(table, jax_core.Tracer) for table in tables):
            raise TypeError(
                "O(3) coefficient tables are validated as concrete data only."
            )
        dtypes = {jnp.dtype(table.dtype) for table in tables}
        if len(dtypes) != 1:
            raise ValueError("O(3) coefficient tables must share one executed dtype.")
        support = _coefficient_support(tables)
        if support != self.coefficient_support:
            raise ValueError(
                "O(3) coefficient tables do not match their recorded "
                "coefficient_support; prepare the tensor product again."
            )
        imported = self.coefficient_origin == "imported"
        readmitted = O3TensorProduct(
            self.plan,
            internal_weights=False,
            dtype=dtypes.pop(),
            coefficients=(
                tuple(np.asarray(jax.device_get(table)) for table in tables)
                if imported
                else None
            ),
            coefficient_tolerance=self.coefficient_tolerance if imported else None,
        )
        current = _coefficient_id(
            self.plan,
            tables,
            support,
            self.coefficient_origin,
            self.coefficient_tolerance,
        )
        if (
            readmitted.coefficient_support != support
            or readmitted.coefficient_id != self.coefficient_id
            or current != self.coefficient_id
        ):
            raise ValueError(
                "O(3) coefficient tables do not match their recorded identity; "
                "prepare the tensor product again."
            )

    @staticmethod
    def _blocks(layout: O3TensorProductLayout, values: Array, /) -> tuple[Array, ...]:
        leading = values.shape[:-1]
        return tuple(
            values[
                ..., block.offset : block.offset + block.multiplicity * block.dimension
            ].reshape(leading + (block.multiplicity, block.dimension))
            for block in _layout_blocks(layout)
        )

    @staticmethod
    def _contribution(
        instruction: _O3TensorProductInstruction,
        left: Array,
        right: Array,
        coefficients: Array,
        weights: Array,
        /,
    ) -> Array:
        leading = left.shape[:-2]
        coupled = contract("...ui,oij->...uoj", left, coefficients)
        paired = contract("...uoj,...vj->...uvo", coupled, right)
        path_weights = weights[
            ...,
            instruction.weight_offset : instruction.weight_offset
            + instruction.weight_count,
        ]
        if instruction.connection_mode == "uvw":
            return contract(
                "...uvo,...wuv->...wo",
                paired,
                path_weights.reshape(leading + _weight_shape(instruction)),
            )
        if instruction.weighted:
            return contract(
                "...uvo,...uv->...uo",
                paired,
                path_weights.reshape(leading + _weight_shape(instruction)),
            )
        return jnp.sum(paired, axis=-2)

    def __call__(
        self,
        left: Array,
        right: Array,
        path_weights: Array | None = None,
        /,
    ) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.ndim < 1 or left_.shape[-1] != self.plan.left_representation.packed_size:
            raise ValueError("Left values do not match the planned O(3) layout.")
        if (
            right_.ndim < 1
            or right_.shape[-1] != self.plan.right_representation.packed_size
        ):
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
                "O(3) path weights must have the input leading axes and planned parameter count."
            )
        left_blocks = self._blocks(self.plan.left_representation, left_)
        right_blocks = self._blocks(self.plan.right_representation, right_)
        output_layout = _layout_blocks(self.plan.output_representation)
        dtype = jnp.result_type(left_, right_, weights)
        output_blocks = [
            jnp.zeros(leading + (block.multiplicity, block.dimension), dtype=dtype)
            for block in output_layout
        ]
        for instruction, coefficients in zip(
            self.plan.instructions, self.prepared.coefficients, strict=True
        ):
            output_blocks[instruction.output_block] = output_blocks[
                instruction.output_block
            ] + self._contribution(
                instruction,
                left_blocks[instruction.left_block],
                right_blocks[instruction.right_block],
                coefficients,
                weights,
            )
        return jnp.concatenate(
            [
                block.reshape(leading + (layout.multiplicity * layout.dimension,))
                for block, layout in zip(output_blocks, output_layout, strict=True)
            ],
            axis=-1,
        )


register_artifact_value("phydrax.nn.operator:O3TensorProduct", O3TensorProduct)
register_artifact_value("phydrax.nn.operator:O3TensorProductPath", O3TensorProductPath)
register_artifact_value("phydrax.nn.operator:O3TensorProductPlan", O3TensorProductPlan)
register_artifact_value(
    "phydrax.nn.operator.internal:O3TensorProductInstruction",
    _O3TensorProductInstruction,
)
register_artifact_value(
    "phydrax.nn.operator.internal:PreparedClebschGordan", _PreparedClebschGordan
)


__all__ = ["O3TensorProduct", "O3TensorProductPath", "O3TensorProductPlan"]

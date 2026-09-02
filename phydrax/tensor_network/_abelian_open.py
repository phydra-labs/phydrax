#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._array_archive import array_collection_digest
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._abelian import AbelianCharge, AbelianLeg


class AbelianKrausOperator(StrictModule):
    """One explicitly charge-covariant Kraus operator in sector blocks."""

    input_leg: AbelianLeg
    output_leg: AbelianLeg
    charge_shift: AbelianCharge = eqx.field(static=True)
    routes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    blocks: tuple[Array, ...]
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_leg: AbelianLeg,
        output_leg: AbelianLeg,
        charge_shift: Sequence[int],
        routes: Sequence[tuple[int, int]],
        blocks: Sequence[ArrayLike],
        /,
    ):
        if not isinstance(input_leg, AbelianLeg) or not isinstance(
            output_leg, AbelianLeg
        ):
            raise TypeError("Kraus input and output legs must be AbelianLeg values.")
        if input_leg.orientation != 1 or output_leg.orientation != 1:
            raise ValueError("Kraus physical legs must be outward oriented.")
        if input_leg.group.group_id != output_leg.group.group_id:
            raise ValueError("Kraus input and output groups must match.")
        shift = input_leg.group.normalize(charge_shift)
        routes_ = tuple((int(output), int(input_)) for output, input_ in routes)
        blocks_ = tuple(jnp.asarray(block) for block in blocks)
        if len(routes_) != len(blocks_) or len(set(routes_)) != len(routes_):
            raise ValueError("Kraus routes and blocks must align uniquely.")
        for (output, input_), block in zip(routes_, blocks_, strict=True):
            if not 0 <= output < len(output_leg.charges) or not 0 <= input_ < len(
                input_leg.charges
            ):
                raise ValueError("Kraus sector route is outside a physical leg.")
            if output_leg.charges[output] != input_leg.group.add(
                input_leg.charges[input_], shift
            ):
                raise ValueError("Kraus route violates its declared charge shift.")
            expected = (output_leg.capacities[output], input_leg.capacities[input_])
            if block.shape != expected:
                raise ValueError("Kraus block shape differs from sector capacities.")
        self.input_leg = input_leg
        self.output_leg = output_leg
        self.charge_shift = shift
        self.routes = routes_
        self.blocks = blocks_
        arrays = {f"block/{index:06d}": block for index, block in enumerate(blocks_)}
        self.operator_id = canonical_fingerprint(
            {
                "kind": "abelian-kraus-operator",
                "input": input_leg.allocation_id,
                "output": output_leg.allocation_id,
                "shift": shift,
                "routes": routes_,
                "dtypes": tuple(str(block.dtype) for block in blocks_),
                "values": array_collection_digest(arrays),
            }
        )


class ChargeCovariantKrausMap(StrictModule):
    operators: tuple[AbelianKrausOperator, ...]
    completeness_residual: Array
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: Sequence[AbelianKrausOperator],
        /,
        *,
        completeness_tolerance: float = 1e-10,
        require_trace_preserving: bool = True,
    ):
        values = tuple(operators)
        if not values or any(
            not isinstance(value, AbelianKrausOperator) for value in values
        ):
            raise TypeError("A Kraus map requires AbelianKrausOperator values.")
        input_leg, output_leg = values[0].input_leg, values[0].output_leg
        if any(
            value.input_leg.allocation_id != input_leg.allocation_id
            or value.output_leg.allocation_id != output_leg.allocation_id
            for value in values
        ):
            raise ValueError("Every Kraus operator must use the same allocations.")
        dtype = jnp.result_type(*(block for value in values for block in value.blocks))
        residual = jnp.asarray(0.0, dtype=jnp.empty((), dtype=dtype).real.dtype)
        for input_sector, capacity in enumerate(input_leg.capacities):
            gram = jnp.zeros((capacity, capacity), dtype=dtype)
            for value in values:
                for route, block in zip(value.routes, value.blocks, strict=True):
                    if route[1] == input_sector:
                        gram = gram + jnp.conj(block.T) @ block
            residual = jnp.maximum(
                residual,
                jnp.linalg.norm(gram - jnp.eye(capacity, dtype=dtype)),
            )
        if require_trace_preserving:
            first = values[0]
            checked = eqx.error_if(
                first.blocks[0],
                residual > float(completeness_tolerance),
                "Charge-covariant Kraus map is not trace preserving.",
            )
            values = (
                AbelianKrausOperator(
                    first.input_leg,
                    first.output_leg,
                    first.charge_shift,
                    first.routes,
                    (checked,) + first.blocks[1:],
                ),
            ) + values[1:]
        self.operators = values
        self.completeness_residual = residual
        self.map_id = canonical_fingerprint(
            {
                "kind": "charge-covariant-kraus-map",
                "operators": tuple(value.operator_id for value in values),
                "trace_preserving": bool(require_trace_preserving),
            }
        )


class AbelianLPDO(StrictModule):
    """Positive block state stored as finite sector purification factors."""

    physical_leg: AbelianLeg
    purification_capacities: tuple[int, ...] = eqx.field(static=True)
    factors: tuple[Array, ...]
    lpdo_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_leg: AbelianLeg,
        purification_capacities: Sequence[int],
        factors: Sequence[ArrayLike],
        /,
        *,
        normalize: bool = False,
    ):
        if not isinstance(physical_leg, AbelianLeg) or physical_leg.orientation != 1:
            raise TypeError("LPDO physical_leg must be an outward AbelianLeg.")
        capacities = tuple(int(value) for value in purification_capacities)
        values = tuple(jnp.asarray(value) for value in factors)
        if len(capacities) != len(physical_leg.charges) or len(values) != len(capacities):
            raise ValueError("LPDO requires one purification factor per charge sector.")
        if any(value < 1 for value in capacities):
            raise ValueError("LPDO purification capacities must be positive.")
        for sector, (capacity, value) in enumerate(zip(capacities, values, strict=True)):
            if value.shape != (physical_leg.capacities[sector], capacity):
                raise ValueError("LPDO factor shape differs from static capacities.")
        trace = sum(jnp.real(jnp.vdot(value, value)) for value in values)
        if normalize:
            scale = jnp.sqrt(
                eqx.error_if(
                    trace,
                    (~jnp.isfinite(trace)) | (trace <= 0),
                    "LPDO trace must be finite and positive.",
                )
            )
            values = tuple(value / scale for value in values)
        self.physical_leg = physical_leg
        self.purification_capacities = capacities
        self.factors = values
        self.lpdo_id = canonical_fingerprint(
            {
                "kind": "abelian-lpdo",
                "physical_leg": physical_leg.allocation_id,
                "purification_capacities": capacities,
                "dtypes": tuple(str(value.dtype) for value in values),
            }
        )

    def trace(self, /) -> Array:
        return sum(jnp.real(jnp.vdot(value, value)) for value in self.factors)

    def density_blocks(self, /) -> tuple[Array, ...]:
        return tuple(value @ jnp.conj(value.T) for value in self.factors)


class AbelianOpenEvolutionEvidence(StrictModule):
    input_trace: Array
    output_trace: Array
    trace_residual: Array
    discarded_weight: Array
    per_sector_retained_ranks: Array
    undeclared_charge_residual: Array
    valid: Array
    route_id: str = eqx.field(static=True)


def apply_charge_covariant_kraus(
    channel: ChargeCovariantKrausMap,
    state: AbelianLPDO,
    /,
    *,
    maximum_purification_dimension: int,
    normalize: bool = False,
) -> tuple[AbelianLPDO, AbelianOpenEvolutionEvidence]:
    """Apply every Kraus route and compress factors sector by sector."""

    if not isinstance(channel, ChargeCovariantKrausMap) or not isinstance(
        state, AbelianLPDO
    ):
        raise TypeError("channel and state have invalid open-system types.")
    input_leg = channel.operators[0].input_leg
    output_leg = channel.operators[0].output_leg
    if state.physical_leg.allocation_id != input_leg.allocation_id:
        raise ValueError("Kraus input allocation differs from LPDO state.")
    maximum = int(maximum_purification_dimension)
    if maximum < 1:
        raise ValueError("maximum_purification_dimension must be positive.")
    dtype = jnp.result_type(
        *(
            state.factors
            + tuple(block for value in channel.operators for block in value.blocks)
        )
    )
    output_factors = []
    output_capacities = []
    discarded = jnp.asarray(0.0, dtype=jnp.empty((), dtype=dtype).real.dtype)
    retained_ranks = []
    for output_sector, output_capacity in enumerate(output_leg.capacities):
        columns = []
        for value in channel.operators:
            for (output, input_), block in zip(value.routes, value.blocks, strict=True):
                if output == output_sector:
                    columns.append(block @ state.factors[input_])
        combined = (
            jnp.concatenate(columns, axis=1)
            if columns
            else jnp.zeros((output_capacity, 1), dtype=dtype)
        )
        u, singular_values, _ = jnp.linalg.svd(combined, full_matrices=False)
        retained = min(maximum, int(singular_values.shape[0]))
        factor = u[:, :retained] * singular_values[:retained]
        discarded = discarded + jnp.sum(jnp.abs(singular_values[retained:]) ** 2)
        output_factors.append(factor)
        output_capacities.append(retained)
        retained_ranks.append(retained)
    result = AbelianLPDO(
        output_leg,
        tuple(output_capacities),
        tuple(output_factors),
        normalize=normalize,
    )
    input_trace, output_trace = state.trace(), result.trace()
    trace_residual = jnp.abs(output_trace - (1.0 if normalize else input_trace))
    charge_residual = jnp.asarray(0.0, dtype=trace_residual.dtype)
    valid = (
        jnp.isfinite(input_trace)
        & jnp.isfinite(output_trace)
        & jnp.isfinite(discarded)
        & (discarded >= 0)
        & (channel.completeness_residual >= 0)
    )
    return result, AbelianOpenEvolutionEvidence(
        input_trace,
        output_trace,
        trace_residual,
        discarded,
        jnp.asarray(retained_ranks, dtype=jnp.int32),
        charge_residual,
        valid,
        channel.map_id,
    )


class AbelianLindbladian(StrictModule):
    physical_leg: AbelianLeg
    hamiltonian_blocks: tuple[Array, ...]
    jumps: tuple[AbelianKrausOperator, ...]
    lindbladian_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_leg: AbelianLeg,
        hamiltonian_blocks: Sequence[ArrayLike],
        jumps: Sequence[AbelianKrausOperator],
        /,
    ):
        if not isinstance(physical_leg, AbelianLeg):
            raise TypeError("physical_leg must be AbelianLeg.")
        blocks = tuple(jnp.asarray(value) for value in hamiltonian_blocks)
        jumps_ = tuple(jumps)
        if len(blocks) != len(physical_leg.charges):
            raise ValueError("One Hamiltonian block is required per charge sector.")
        checked = []
        for capacity, block in zip(physical_leg.capacities, blocks, strict=True):
            if block.shape != (capacity, capacity):
                raise ValueError("Lindbladian Hamiltonian block shape is invalid.")
            checked.append(
                eqx.error_if(
                    block,
                    ~jnp.allclose(block, jnp.conj(block.T)),
                    "Lindbladian Hamiltonian blocks must be Hermitian.",
                )
            )
        if any(
            not isinstance(jump, AbelianKrausOperator)
            or jump.input_leg.allocation_id != physical_leg.allocation_id
            or jump.output_leg.allocation_id != physical_leg.allocation_id
            for jump in jumps_
        ):
            raise ValueError("Lindbladian jumps must act on the physical allocation.")
        self.physical_leg = physical_leg
        self.hamiltonian_blocks = tuple(checked)
        self.jumps = jumps_
        arrays = {
            f"hamiltonian/{index:06d}": block for index, block in enumerate(checked)
        }
        self.lindbladian_id = canonical_fingerprint(
            {
                "kind": "abelian-lindbladian",
                "physical_leg": physical_leg.allocation_id,
                "hamiltonian_shapes": tuple(block.shape for block in blocks),
                "jumps": tuple(jump.operator_id for jump in jumps_),
                "hamiltonian_values": array_collection_digest(arrays),
            }
        )

    def kraus_step(self, step_size: ArrayLike, /) -> ChargeCovariantKrausMap:
        step = jnp.asarray(step_size, dtype=self.hamiltonian_blocks[0].real.dtype)
        step = eqx.error_if(
            step,
            (~jnp.isfinite(step)) | (step <= 0),
            "Lindbladian step must be finite and positive.",
        )
        neutral_routes = tuple(
            (sector, sector) for sector in range(len(self.physical_leg.charges))
        )
        neutral_blocks = []
        for sector, hamiltonian in enumerate(self.hamiltonian_blocks):
            gram = jnp.zeros_like(hamiltonian)
            for jump in self.jumps:
                for route, block in zip(jump.routes, jump.blocks, strict=True):
                    if route[1] == sector:
                        gram = gram + jnp.conj(block.T) @ block
            neutral_blocks.append(
                jnp.eye(hamiltonian.shape[0], dtype=hamiltonian.dtype)
                - step * (1.0j * hamiltonian + 0.5 * gram)
            )
        no_jump = AbelianKrausOperator(
            self.physical_leg,
            self.physical_leg,
            self.physical_leg.group.zero,
            neutral_routes,
            tuple(neutral_blocks),
        )
        scaled_jumps = tuple(
            AbelianKrausOperator(
                jump.input_leg,
                jump.output_leg,
                jump.charge_shift,
                jump.routes,
                tuple(jnp.sqrt(step) * block for block in jump.blocks),
            )
            for jump in self.jumps
        )
        return ChargeCovariantKrausMap(
            (no_jump,) + scaled_jumps,
            require_trace_preserving=False,
        )


class AbelianOpenProcess(StrictModule):
    routes: tuple[ChargeCovariantKrausMap, ...]
    process_id: str = eqx.field(static=True)

    def __init__(self, routes: Sequence[ChargeCovariantKrausMap], /):
        values = tuple(routes)
        if not values or any(
            not isinstance(value, ChargeCovariantKrausMap) for value in values
        ):
            raise TypeError("Open process routes must be charge-covariant Kraus maps.")
        for first, second in pairwise(values):
            if (
                first.operators[0].output_leg.allocation_id
                != second.operators[0].input_leg.allocation_id
            ):
                raise ValueError(
                    "Adjacent open-process route allocations do not compose."
                )
        self.routes = values
        self.process_id = canonical_fingerprint(
            {
                "kind": "abelian-open-process",
                "routes": tuple(value.map_id for value in values),
            }
        )


class AbelianOpenProcessEvidence(StrictModule):
    trace_residuals: Array
    discarded_weights: Array
    charge_residuals: Array
    valid: Array
    process_id: str = eqx.field(static=True)


def execute_abelian_open_process(
    process: AbelianOpenProcess,
    state: AbelianLPDO,
    /,
    *,
    maximum_purification_dimension: int,
    normalize_each_route: bool = False,
) -> tuple[AbelianLPDO, AbelianOpenProcessEvidence]:
    if not isinstance(process, AbelianOpenProcess):
        raise TypeError("process must be AbelianOpenProcess.")
    current = state
    records = []
    for route in process.routes:
        current, evidence = apply_charge_covariant_kraus(
            route,
            current,
            maximum_purification_dimension=maximum_purification_dimension,
            normalize=normalize_each_route,
        )
        records.append(evidence)
    traces = jnp.stack([record.trace_residual for record in records])
    discarded = jnp.stack([record.discarded_weight for record in records])
    charge = jnp.stack([record.undeclared_charge_residual for record in records])
    valid = jnp.all(jnp.stack([record.valid for record in records])) & jnp.all(
        jnp.isfinite(traces)
    )
    return current, AbelianOpenProcessEvidence(
        traces, discarded, charge, valid, process.process_id
    )


__all__ = [
    "AbelianKrausOperator",
    "AbelianLPDO",
    "AbelianLindbladian",
    "AbelianOpenEvolutionEvidence",
    "AbelianOpenProcess",
    "AbelianOpenProcessEvidence",
    "ChargeCovariantKrausMap",
    "apply_charge_covariant_kraus",
    "execute_abelian_open_process",
]

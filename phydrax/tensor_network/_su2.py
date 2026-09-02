#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._array_archive import array_collection_digest
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


def _spin(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("Doubled SU2 spins must be nonnegative integers.")
    return result


def su2_fusion(left: int, right: int, /) -> tuple[int, ...]:
    left_, right_ = _spin(left), _spin(right)
    return tuple(range(abs(left_ - right_), left_ + right_ + 1, 2))


def _triangle(left: int, right: int, output: int, /) -> bool:
    return output in su2_fusion(left, right)


def _fact(value: int) -> int:
    return math.factorial(value) if value >= 0 else 0


def su2_clebsch_gordan(left: int, right: int, output: int, /) -> Array:
    """Deterministic Condon--Shortley CG table for doubled spins."""

    a, b, c = _spin(left), _spin(right), _spin(output)
    if not _triangle(a, b, c):
        raise ValueError("Requested SU2 fusion channel is forbidden.")
    table = jnp.zeros((a + 1, b + 1, c + 1), dtype=jnp.float64)
    for ia, ma in enumerate(range(-a, a + 1, 2)):
        for ib, mb in enumerate(range(-b, b + 1, 2)):
            total_m = ma + mb
            if abs(total_m) > c or (total_m + c) % 2:
                continue
            ic = (total_m + c) // 2
            x1 = (a + b - c) // 2
            x2 = (c + a - b) // 2
            x3 = (c - a + b) // 2
            prefactor = math.sqrt(
                (c + 1) * _fact(x1) * _fact(x2) * _fact(x3) / _fact((a + b + c) // 2 + 1)
            )
            prefactor *= math.sqrt(
                _fact((c + total_m) // 2)
                * _fact((c - total_m) // 2)
                * _fact((a - ma) // 2)
                * _fact((a + ma) // 2)
                * _fact((b - mb) // 2)
                * _fact((b + mb) // 2)
            )
            terms = []
            for k in range(x1 + 1):
                arguments = (
                    k,
                    x1 - k,
                    (a - ma) // 2 - k,
                    (b + mb) // 2 - k,
                    (c - b + ma) // 2 + k,
                    (c - a - mb) // 2 + k,
                )
                if any(value < 0 for value in arguments):
                    continue
                terms.append(
                    ((-1.0) ** k) / math.prod(_fact(value) for value in arguments)
                )
            table = table.at[ia, ib, ic].set(prefactor * math.fsum(terms))
    return table


def _delta(first: int, second: int, third: int) -> float:
    if not _triangle(first, second, third):
        return 0.0
    return math.sqrt(
        _fact((first + second - third) // 2)
        * _fact((first - second + third) // 2)
        * _fact((-first + second + third) // 2)
        / _fact((first + second + third) // 2 + 1)
    )


def su2_wigner_6j(
    first: int,
    second: int,
    third: int,
    fourth: int,
    fifth: int,
    sixth: int,
    /,
) -> float:
    """Racah finite-sum 6j symbol, all arguments doubled."""

    a, b, c, d, e, f = map(_spin, (first, second, third, fourth, fifth, sixth))
    prefactor = _delta(a, b, c) * _delta(a, e, f) * _delta(d, b, f) * _delta(d, e, c)
    if prefactor == 0.0:
        return 0.0
    x = ((a + b + c) // 2, (a + e + f) // 2, (d + b + f) // 2, (d + e + c) // 2)
    y = ((a + b + d + e) // 2, (a + c + d + f) // 2, (b + c + e + f) // 2)
    terms = []
    for z in range(max(x), min(y) + 1):
        denominator = math.prod(_fact(z - value) for value in x) * math.prod(
            _fact(value - z) for value in y
        )
        terms.append(((-1.0) ** z) * _fact(z + 1) / denominator)
    return prefactor * math.fsum(terms)


def su2_recoupling_matrix(
    first: int, second: int, third: int, total: int, /
) -> tuple[tuple[int, ...], tuple[int, ...], Array]:
    """F move from ((a b)e c)J to (a (b c)f)J via CG overlaps."""

    a, b, c, target = map(_spin, (first, second, third, total))
    left_channels = tuple(e for e in su2_fusion(a, b) if target in su2_fusion(e, c))
    right_channels = tuple(f for f in su2_fusion(b, c) if target in su2_fusion(a, f))
    matrix = jnp.zeros((len(left_channels), len(right_channels)), dtype=jnp.float64)
    for ie, e in enumerate(left_channels):
        cg_ab = su2_clebsch_gordan(a, b, e)
        cg_ec = su2_clebsch_gordan(e, c, target)
        for jf, f in enumerate(right_channels):
            cg_bc = su2_clebsch_gordan(b, c, f)
            cg_af = su2_clebsch_gordan(a, f, target)
            overlap = jnp.asarray(0.0, dtype=jnp.float64)
            for ia, ma in enumerate(range(-a, a + 1, 2)):
                for ib, mb in enumerate(range(-b, b + 1, 2)):
                    for ic, mc in enumerate(range(-c, c + 1, 2)):
                        mt = ma + mb + mc
                        if abs(mt) > target or (mt + target) % 2:
                            continue
                        me, mf = ma + mb, mb + mc
                        overlap = overlap + (
                            cg_ab[ia, ib, (me + e) // 2]
                            * cg_ec[(me + e) // 2, ic, (mt + target) // 2]
                            * cg_bc[ib, ic, (mf + f) // 2]
                            * cg_af[ia, (mf + f) // 2, (mt + target) // 2]
                        )
            matrix = matrix.at[ie, jf].set(overlap / (target + 1))
    return left_channels, right_channels, matrix


def su2_f_symbol(
    first: int,
    second: int,
    third: int,
    total: int,
    left_channel: int,
    right_channel: int,
    /,
) -> Array:
    """Return one unitary F-move coefficient in the deterministic CG gauge."""

    left, right, matrix = su2_recoupling_matrix(first, second, third, total)
    if left_channel not in left or right_channel not in right:
        raise ValueError("Requested SU2 F-symbol channel is forbidden.")
    return matrix[left.index(left_channel), right.index(right_channel)]


def _su2_f_value(
    first: int,
    second: int,
    third: int,
    total: int,
    left_channel: int,
    right_channel: int,
    /,
) -> Array:
    left, right, _ = su2_recoupling_matrix(first, second, third, total)
    if left_channel not in left or right_channel not in right:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return su2_f_symbol(first, second, third, total, left_channel, right_channel)


def su2_pentagon_residual(
    first: int, second: int, third: int, fourth: int, total: int, /
) -> Array:
    """Compute the pentagon from five finite fusion-path bases and local F moves."""

    a, b, c, d, target = map(_spin, (first, second, third, fourth, total))
    tree_zero = tuple(
        (e, f)
        for e in su2_fusion(a, b)
        for f in su2_fusion(e, c)
        if target in su2_fusion(f, d)
    )
    tree_one = tuple(
        (g, f)
        for g in su2_fusion(b, c)
        for f in su2_fusion(a, g)
        if target in su2_fusion(f, d)
    )
    tree_two = tuple(
        (g, h)
        for g in su2_fusion(b, c)
        for h in su2_fusion(g, d)
        if target in su2_fusion(a, h)
    )
    tree_three = tuple(
        (i, h)
        for i in su2_fusion(c, d)
        for h in su2_fusion(b, i)
        if target in su2_fusion(a, h)
    )
    tree_four = tuple(
        (e, i)
        for e in su2_fusion(a, b)
        for i in su2_fusion(c, d)
        if target in su2_fusion(e, i)
    )
    zero_to_one = jnp.asarray(
        tuple(
            tuple(
                _su2_f_value(a, b, c, f, e, g) if f == source_f else 0.0
                for e, source_f in tree_zero
            )
            for g, f in tree_one
        )
    )
    one_to_two = jnp.asarray(
        tuple(
            tuple(
                _su2_f_value(a, g, d, target, f, h) if g == source_g else 0.0
                for source_g, f in tree_one
            )
            for g, h in tree_two
        )
    )
    two_to_three = jnp.asarray(
        tuple(
            tuple(
                _su2_f_value(b, c, d, h, g, i) if h == source_h else 0.0
                for g, source_h in tree_two
            )
            for i, h in tree_three
        )
    )
    zero_to_four = jnp.asarray(
        tuple(
            tuple(
                _su2_f_value(e, c, d, target, f, i) if e == source_e else 0.0
                for source_e, f in tree_zero
            )
            for e, i in tree_four
        )
    )
    four_to_three = jnp.asarray(
        tuple(
            tuple(
                _su2_f_value(a, b, i, target, e, h) if i == source_i else 0.0
                for e, source_i in tree_four
            )
            for i, h in tree_three
        )
    )
    first_path = two_to_three @ one_to_two @ zero_to_one
    second_path = four_to_three @ zero_to_four
    return jnp.linalg.norm(first_path - second_path)


class SU2ReducedLeg(StrictModule):
    twice_spins: tuple[int, ...] = eqx.field(static=True)
    capacities: tuple[int, ...] = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    active_multiplicities: Array
    allocation_id: str = eqx.field(static=True)

    def __init__(
        self,
        twice_spins: Sequence[int],
        capacities: Sequence[int],
        /,
        *,
        orientation: int,
        active_multiplicities: ArrayLike | None = None,
    ):
        spins = tuple(_spin(value) for value in twice_spins)
        allocated = tuple(int(value) for value in capacities)
        if not spins or len(spins) != len(allocated) or len(set(spins)) != len(spins):
            raise ValueError("SU2 spins and capacities must align uniquely.")
        if any(value < 1 for value in allocated):
            raise ValueError("SU2 capacities must be positive.")
        direction = int(orientation)
        if direction not in (-1, 1):
            raise ValueError("SU2 leg orientation must be +1 or -1.")
        active = jnp.asarray(
            allocated if active_multiplicities is None else active_multiplicities,
            dtype=jnp.int32,
        )
        if active.shape != (len(spins),):
            raise ValueError("One active multiplicity is required per SU2 spin.")
        active = eqx.error_if(
            active,
            jnp.any((active < 0) | (active > jnp.asarray(allocated))),
            "Active SU2 multiplicities exceed capacity.",
        )
        self.twice_spins, self.capacities, self.orientation = spins, allocated, direction
        self.active_multiplicities = active
        self.allocation_id = canonical_fingerprint(
            {
                "kind": "su2-reduced-leg",
                "spins": spins,
                "capacities": allocated,
                "orientation": direction,
            }
        )

    def dual(self) -> SU2ReducedLeg:
        return SU2ReducedLeg(
            self.twice_spins,
            self.capacities,
            orientation=-self.orientation,
            active_multiplicities=self.active_multiplicities,
        )


class SU2ReducedTensor(StrictModule):
    legs: tuple[SU2ReducedLeg, ...]
    sectors: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    blocks: tuple[Array, ...]
    total_twice_spin: int = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)

    def __init__(
        self,
        legs: Sequence[SU2ReducedLeg],
        sectors: Sequence[Sequence[int]],
        blocks: Sequence[ArrayLike],
        /,
        *,
        total_twice_spin: int = 0,
    ):
        legs_ = tuple(legs)
        sectors_ = tuple(tuple(int(value) for value in sector) for sector in sectors)
        blocks_ = tuple(jnp.asarray(block) for block in blocks)
        target = _spin(total_twice_spin)
        if not legs_ or any(not isinstance(leg, SU2ReducedLeg) for leg in legs_):
            raise TypeError("SU2 reduced tensor legs are invalid.")
        if len(sectors_) != len(blocks_) or len(set(sectors_)) != len(sectors_):
            raise ValueError("SU2 sectors and blocks must align uniquely.")
        for sector, block in zip(sectors_, blocks_, strict=True):
            if len(sector) != len(legs_) or any(
                not 0 <= ordinal < len(leg.twice_spins)
                for ordinal, leg in zip(sector, legs_, strict=True)
            ):
                raise ValueError("SU2 sector ordinal is invalid.")
            expected = tuple(
                leg.capacities[ordinal]
                for leg, ordinal in zip(legs_, sector, strict=True)
            )
            if block.shape != expected:
                raise ValueError("SU2 reduced block shape differs from capacity.")
            reachable = {legs_[0].twice_spins[sector[0]]}
            for leg, ordinal in zip(legs_[1:], sector[1:], strict=True):
                reachable = {
                    result
                    for current in reachable
                    for result in su2_fusion(current, leg.twice_spins[ordinal])
                }
            if target not in reachable:
                raise ValueError("SU2 sector cannot fuse to the declared total spin.")
        self.legs, self.sectors, self.blocks, self.total_twice_spin = (
            legs_,
            sectors_,
            blocks_,
            target,
        )
        self.tensor_id = canonical_fingerprint(
            {
                "kind": "su2-reduced-tensor",
                "legs": tuple(leg.allocation_id for leg in legs_),
                "sectors": sectors_,
                "total": target,
                "dtypes": tuple(str(block.dtype) for block in blocks_),
            }
        )


def contract_su2_reduced(
    left: SU2ReducedTensor, right: SU2ReducedTensor, left_axis: int, right_axis: int, /
) -> SU2ReducedTensor | Array:
    """Contract invariant reduced tensors with exact multiplet dimensions."""

    if not isinstance(left, SU2ReducedTensor) or not isinstance(right, SU2ReducedTensor):
        raise TypeError("left and right must be SU2ReducedTensor values.")
    la, ra = int(left_axis) % len(left.legs), int(right_axis) % len(right.legs)
    if left.legs[la].allocation_id != right.legs[ra].dual().allocation_id:
        raise ValueError("Contracted SU2 legs must be dual compatible.")
    left_free = tuple(index for index in range(len(left.legs)) if index != la)
    right_free = tuple(index for index in range(len(right.legs)) if index != ra)
    output_legs = tuple(left.legs[index] for index in left_free) + tuple(
        right.legs[index] for index in right_free
    )
    output_sectors: list[tuple[int, ...]] = []
    output_blocks: list[Array] = []
    scalar = jnp.asarray(0.0, dtype=jnp.result_type(left.blocks[0], right.blocks[0]))
    for ls, lb in zip(left.sectors, left.blocks, strict=True):
        for rs, rb in zip(right.sectors, right.blocks, strict=True):
            if ls[la] != rs[ra]:
                continue
            spin = left.legs[la].twice_spins[ls[la]]
            left_labels = list(range(lb.ndim))
            right_labels = [
                la if axis == ra else lb.ndim + axis for axis in range(rb.ndim)
            ]
            output_labels = [axis for axis in left_labels if axis != la] + [
                lb.ndim + axis for axis in range(rb.ndim) if axis != ra
            ]
            contribution = (spin + 1) * ein.contract(
                lb, left_labels, rb, right_labels, output_labels
            )
            if not output_legs:
                scalar = scalar + contribution
                continue
            sector = tuple(ls[index] for index in left_free) + tuple(
                rs[index] for index in right_free
            )
            if sector in output_sectors:
                index = output_sectors.index(sector)
                output_blocks[index] = output_blocks[index] + contribution
            else:
                output_sectors.append(sector)
                output_blocks.append(contribution)
    if not output_legs:
        return scalar
    return SU2ReducedTensor(output_legs, output_sectors, output_blocks)


class SU2MultipletTruncationEvidence(StrictModule):
    retained_multiplet_dimension: Array
    available_multiplet_dimension: int = eqx.field(static=True)
    retained_multiplicities: Array
    discarded_weight: Array
    protected_multiplets_satisfied: Array
    valid: Array


def truncate_su2_multiplets(
    twice_spins: Sequence[int],
    singular_values: Sequence[ArrayLike],
    /,
    *,
    maximum_dimension: int,
    protected_twice_spins: Sequence[int] = (),
) -> tuple[tuple[Array, ...], SU2MultipletTruncationEvidence]:
    """Globally truncate only complete (2j+1)-dimensional multiplets."""

    spins = tuple(_spin(value) for value in twice_spins)
    spectra = tuple(jnp.asarray(value) for value in singular_values)
    if len(spins) != len(spectra) or any(value.ndim != 1 for value in spectra):
        raise ValueError("One singular-value vector is required per SU2 spin.")
    maximum = int(maximum_dimension)
    if maximum < 1:
        raise ValueError("maximum_dimension must be positive.")
    protected = tuple(_spin(value) for value in protected_twice_spins)
    if len(set(protected)) != len(protected):
        raise ValueError("Protected SU2 spins must be unique.")
    flat = jnp.concatenate(spectra)
    costs = jnp.concatenate(
        tuple(
            jnp.full(value.shape, spin + 1, dtype=jnp.int32)
            for spin, value in zip(spins, spectra, strict=True)
        )
    )
    selected = jnp.zeros(flat.shape, dtype=bool)
    used = jnp.asarray(0, dtype=jnp.int32)
    cursor = 0
    for spin, values in zip(spins, spectra, strict=True):
        if spin in protected and values.shape[0] > 0:
            index = cursor + jnp.argmax(jnp.abs(values))
            selected = selected.at[index].set(True)
            used = used + spin + 1
        cursor += values.shape[0]
    order = jnp.argsort(-jnp.abs(flat), stable=True)
    for index in order:
        take = (~selected[index]) & (used + costs[index] <= maximum)
        selected = selected.at[index].set(selected[index] | take)
        used = jnp.where(take, used + costs[index], used)
    masks, counts = [], []
    cursor = 0
    for values in spectra:
        mask = selected[cursor : cursor + values.shape[0]]
        masks.append(mask)
        counts.append(jnp.sum(mask.astype(jnp.int32)))
        cursor += values.shape[0]
    checks = tuple(
        jnp.any(masks[spins.index(spin)]) if spin in spins else jnp.asarray(False)
        for spin in protected
    )
    protected_ok = (
        jnp.all(jnp.stack(checks)) & (used <= maximum) if checks else jnp.asarray(True)
    )
    discarded = jnp.sum(jnp.where(selected, 0.0, costs * jnp.abs(flat) ** 2))
    evidence = SU2MultipletTruncationEvidence(
        used,
        sum(
            (spin + 1) * value.shape[0]
            for spin, value in zip(spins, spectra, strict=True)
        ),
        jnp.stack(counts),
        discarded,
        protected_ok,
        jnp.isfinite(discarded) & (discarded >= 0) & protected_ok,
    )
    return tuple(masks), evidence


class SU2SectorState(StrictModule):
    twice_spin: int = eqx.field(static=True)
    amplitudes: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self, twice_spin: int, amplitudes: ArrayLike, /, *, normalize: bool = True
    ):
        spin, values = _spin(twice_spin), jnp.asarray(amplitudes)
        if values.ndim != 1 or values.shape[0] < 1:
            raise ValueError("SU2 amplitudes must be a nonempty vector.")
        norm = jnp.linalg.norm(values)
        if normalize:
            values = values / eqx.error_if(
                norm,
                (~jnp.isfinite(norm)) | (norm <= 0),
                "SU2 state norm must be finite and positive.",
            )
        self.twice_spin, self.amplitudes = spin, values
        self.state_id = canonical_fingerprint(
            {
                "kind": "su2-sector-state",
                "spin": spin,
                "capacity": values.shape[0],
                "dtype": str(values.dtype),
            }
        )


class SU2InvariantOperator(StrictModule):
    twice_spins: tuple[int, ...] = eqx.field(static=True)
    blocks: tuple[Array, ...]
    operator_id: str = eqx.field(static=True)

    def __init__(self, twice_spins: Sequence[int], blocks: Sequence[ArrayLike], /):
        spins, values = (
            tuple(_spin(value) for value in twice_spins),
            tuple(jnp.asarray(value) for value in blocks),
        )
        if not spins or len(spins) != len(values) or len(set(spins)) != len(spins):
            raise ValueError("SU2 sectors must align uniquely with blocks.")
        checked = []
        for value in values:
            if value.ndim != 2 or value.shape[0] != value.shape[1]:
                raise ValueError("SU2 operator blocks must be square.")
            checked.append(
                eqx.error_if(
                    value,
                    ~jnp.allclose(value, jnp.conj(value.T)),
                    "SU2 operator blocks must be Hermitian.",
                )
            )
        self.twice_spins, self.blocks = spins, tuple(checked)
        arrays = {f"block/{index:06d}": block for index, block in enumerate(checked)}
        self.operator_id = canonical_fingerprint(
            {
                "kind": "su2-invariant-operator",
                "spins": spins,
                "shapes": tuple(value.shape for value in values),
                "dtypes": tuple(str(value.dtype) for value in values),
                "values": array_collection_digest(arrays),
            }
        )

    def block(self, twice_spin: int, /) -> Array:
        spin = _spin(twice_spin)
        if spin not in self.twice_spins:
            raise ValueError("SU2 operator lacks the requested sector.")
        return self.blocks[self.twice_spins.index(spin)]


class SU2DMRGEvidence(StrictModule):
    energy: Array
    residual_norm: Array
    spectral_gap: Array
    protected_twice_spin: int = eqx.field(static=True)
    sweep_count: int = eqx.field(static=True)
    converged: Array
    valid: Array


def su2_finite_dmrg(
    operator: SU2InvariantOperator,
    /,
    *,
    protected_twice_spin: int,
    maximum_sweeps: int,
    residual_tolerance: float = 1e-10,
) -> tuple[SU2SectorState, SU2DMRGEvidence]:
    sweeps = int(maximum_sweeps)
    if sweeps < 1:
        raise ValueError("maximum_sweeps must be positive.")
    spin = _spin(protected_twice_spin)
    block = operator.block(spin)
    eigenvalues, eigenvectors = jnp.linalg.eigh(block)
    state = SU2SectorState(spin, eigenvectors[:, 0])
    residual_norm = jnp.linalg.norm(
        block @ state.amplitudes - eigenvalues[0] * state.amplitudes
    )
    gap = (
        eigenvalues[1] - eigenvalues[0]
        if block.shape[0] > 1
        else jnp.asarray(jnp.inf, dtype=eigenvalues.dtype)
    )
    converged = residual_norm <= float(residual_tolerance)
    return state, SU2DMRGEvidence(
        eigenvalues[0],
        residual_norm,
        gap,
        spin,
        sweeps,
        converged,
        jnp.isfinite(eigenvalues[0]) & jnp.isfinite(residual_norm),
    )


class SU2TDVPEvidence(StrictModule):
    times: Array
    norm_residuals: Array
    energy_values: Array
    valid: Array
    step_count: int = eqx.field(static=True)
    twice_spin: int = eqx.field(static=True)


def su2_finite_tdvp(
    initial_state: SU2SectorState,
    operator: SU2InvariantOperator,
    step_size: ArrayLike,
    /,
    *,
    step_count: int,
    imaginary_time: bool = False,
) -> tuple[SU2SectorState, SU2TDVPEvidence]:
    count = int(step_count)
    if count < 1:
        raise ValueError("step_count must be positive.")
    block = operator.block(initial_state.twice_spin)
    if block.shape[0] != initial_state.amplitudes.shape[0]:
        raise ValueError("SU2 state and operator capacities differ.")
    step = jnp.asarray(step_size, dtype=initial_state.amplitudes.real.dtype)
    step = eqx.error_if(
        step,
        (~jnp.isfinite(step)) | (step <= 0),
        "SU2 TDVP step must be finite and positive.",
    )
    propagator = jsp.linalg.expm(step * (-1.0 if imaginary_time else -1.0j) * block)
    values, times, norms, energies = initial_state.amplitudes, [], [], []
    for index in range(count):
        values = propagator @ values
        if imaginary_time:
            values = values / jnp.linalg.norm(values)
        norm = jnp.linalg.norm(values)
        times.append((index + 1) * step)
        norms.append(jnp.abs(norm - 1.0))
        energies.append(
            jnp.real(jnp.vdot(values, block @ values) / jnp.vdot(values, values))
        )
    result = SU2SectorState(initial_state.twice_spin, values, normalize=False)
    time_values, norm_values, energy_values = (
        jnp.stack(times),
        jnp.stack(norms),
        jnp.stack(energies),
    )
    valid = (
        jnp.all(jnp.isfinite(time_values))
        & jnp.all(jnp.isfinite(norm_values))
        & jnp.all(jnp.isfinite(energy_values))
    )
    return result, SU2TDVPEvidence(
        time_values, norm_values, energy_values, valid, count, initial_state.twice_spin
    )


def _su2_fusion_paths(
    site_twice_spins: tuple[int, ...], total_twice_spin: int, /
) -> tuple[tuple[int, ...], ...]:
    partial = ((site_twice_spins[0],),)
    for spin in site_twice_spins[1:]:
        partial = tuple(
            path + (output,) for path in partial for output in su2_fusion(path[-1], spin)
        )
    return tuple(path for path in partial if path[-1] == total_twice_spin)


class SU2MatrixProductState(StrictModule):
    """Finite SU2 MPS represented in its exact left fusion-path basis."""

    site_twice_spins: tuple[int, ...] = eqx.field(static=True)
    total_twice_spin: int = eqx.field(static=True)
    fusion_paths: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    amplitudes: Array
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_twice_spins: Sequence[int],
        total_twice_spin: int,
        amplitudes: ArrayLike,
        /,
        *,
        normalize: bool = True,
    ):
        sites = tuple(_spin(value) for value in site_twice_spins)
        total = _spin(total_twice_spin)
        if len(sites) < 2:
            raise ValueError("An SU2 MPS requires at least two sites.")
        paths = _su2_fusion_paths(sites, total)
        values = jnp.asarray(amplitudes)
        if not paths or values.shape != (len(paths),):
            raise ValueError("SU2 MPS amplitudes must match all fusion paths.")
        norm = jnp.linalg.norm(values)
        if normalize:
            values = values / eqx.error_if(
                norm,
                (~jnp.isfinite(norm)) | (norm <= 0),
                "SU2 MPS norm must be finite and positive.",
            )
        self.site_twice_spins = sites
        self.total_twice_spin = total
        self.fusion_paths = paths
        self.amplitudes = values
        self.structure_id = canonical_fingerprint(
            {
                "kind": "su2-matrix-product-state",
                "sites": sites,
                "total": total,
                "fusion_paths": paths,
                "dtype": str(values.dtype),
            }
        )

    def norm(self, /) -> Array:
        return jnp.linalg.norm(self.amplitudes)


class SU2MatrixProductOperator(StrictModule):
    """Invariant finite-chain MPO as blocks on fusion-path multiplicities."""

    site_twice_spins: tuple[int, ...] = eqx.field(static=True)
    operator: SU2InvariantOperator
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_twice_spins: Sequence[int],
        operator: SU2InvariantOperator,
        /,
    ):
        sites = tuple(_spin(value) for value in site_twice_spins)
        if len(sites) < 2 or not isinstance(operator, SU2InvariantOperator):
            raise TypeError("SU2 MPO sites or invariant operator are invalid.")
        for spin, block in zip(operator.twice_spins, operator.blocks, strict=True):
            paths = _su2_fusion_paths(sites, spin)
            if block.shape != (len(paths), len(paths)):
                raise ValueError("SU2 MPO block must span its fusion-path multiplicity.")
        self.site_twice_spins = sites
        self.operator = operator
        self.structure_id = canonical_fingerprint(
            {
                "kind": "su2-matrix-product-operator",
                "sites": sites,
                "operator": operator.operator_id,
            }
        )


def su2_mps_dmrg(
    initial_state: SU2MatrixProductState,
    operator: SU2MatrixProductOperator,
    /,
    *,
    maximum_sweeps: int,
    residual_tolerance: float = 1e-10,
) -> tuple[SU2MatrixProductState, SU2DMRGEvidence]:
    if not isinstance(initial_state, SU2MatrixProductState) or not isinstance(
        operator, SU2MatrixProductOperator
    ):
        raise TypeError("SU2 DMRG requires SU2 MPS and MPO values.")
    if initial_state.site_twice_spins != operator.site_twice_spins:
        raise ValueError("SU2 MPS and MPO site representations differ.")
    sector, evidence = su2_finite_dmrg(
        operator.operator,
        protected_twice_spin=initial_state.total_twice_spin,
        maximum_sweeps=maximum_sweeps,
        residual_tolerance=residual_tolerance,
    )
    return (
        SU2MatrixProductState(
            initial_state.site_twice_spins,
            initial_state.total_twice_spin,
            sector.amplitudes,
        ),
        evidence,
    )


def su2_mps_tdvp(
    initial_state: SU2MatrixProductState,
    operator: SU2MatrixProductOperator,
    step_size: ArrayLike,
    /,
    *,
    step_count: int,
    imaginary_time: bool = False,
) -> tuple[SU2MatrixProductState, SU2TDVPEvidence]:
    if not isinstance(initial_state, SU2MatrixProductState) or not isinstance(
        operator, SU2MatrixProductOperator
    ):
        raise TypeError("SU2 TDVP requires SU2 MPS and MPO values.")
    if initial_state.site_twice_spins != operator.site_twice_spins:
        raise ValueError("SU2 MPS and MPO site representations differ.")
    sector = SU2SectorState(
        initial_state.total_twice_spin,
        initial_state.amplitudes,
        normalize=False,
    )
    result, evidence = su2_finite_tdvp(
        sector,
        operator.operator,
        step_size,
        step_count=step_count,
        imaginary_time=imaginary_time,
    )
    return (
        SU2MatrixProductState(
            initial_state.site_twice_spins,
            initial_state.total_twice_spin,
            result.amplitudes,
            normalize=imaginary_time,
        ),
        evidence,
    )


__all__ = [
    "SU2DMRGEvidence",
    "SU2InvariantOperator",
    "SU2MatrixProductOperator",
    "SU2MatrixProductState",
    "SU2MultipletTruncationEvidence",
    "SU2ReducedLeg",
    "SU2ReducedTensor",
    "SU2SectorState",
    "SU2TDVPEvidence",
    "contract_su2_reduced",
    "su2_clebsch_gordan",
    "su2_finite_dmrg",
    "su2_finite_tdvp",
    "su2_fusion",
    "su2_mps_dmrg",
    "su2_mps_tdvp",
    "su2_pentagon_residual",
    "su2_recoupling_matrix",
    "su2_wigner_6j",
    "su2_f_symbol",
    "truncate_su2_multiplets",
]

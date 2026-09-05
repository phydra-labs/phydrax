#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....solver._jump import finite_state_generator, FiniteStateGenerator
from ....stochastic import AbstractJumpProcess
from .._construct import NucleicAcidConstruct, NucleotideKey
from ._model import AssociationConvention, SecondaryEnergyModel, SecondaryRateLaw
from ._state import _noncrossing, SecondaryMove, SecondaryStructureState


class SecondaryJumpProcess(AbstractJumpProcess):
    """Bounded pair-toggle channels over a completely enumerated legal support.

    Runtime state is a one-element integer *state index*, not a nucleotide or
    atom ID. All structural preparation is host-only. Array lookup and rate
    evaluation are JIT-compatible; event choices are not pathwise differentiable.
    Every illegal toggle has zero intensity and executes as an identity map.
    """

    destinations: Array
    legal: Array
    rates: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(self, destinations, legal, rates, *, process_id):
        raw_destinations = jnp.asarray(destinations)
        if raw_destinations.ndim != 2 or min(raw_destinations.shape) <= 0:
            raise ValueError(
                "A secondary jump process requires nonempty state/channel tables."
            )
        if not jnp.issubdtype(raw_destinations.dtype, jnp.integer):
            raise ValueError("Secondary destination indices must be exact integers.")
        self.destinations = jnp.asarray(destinations, dtype=jnp.int32)
        self.legal = jnp.asarray(legal, dtype=bool)
        self.rates = jnp.asarray(rates, dtype=float)
        if (
            self.destinations.shape != self.rates.shape
            or self.legal.shape != self.rates.shape
        ):
            raise ValueError(
                "Destination, legality and rate tables must have matching shape."
            )
        if bool(jnp.any(~jnp.isfinite(self.rates) | (self.rates < 0))) or bool(
            jnp.any(~self.legal & (self.rates != 0))
        ):
            raise ValueError(
                "Rates must be finite, nonnegative and zero for illegal moves."
            )
        if bool(
            jnp.any((self.destinations < 0) | (self.destinations >= self.rates.shape[0]))
        ):
            raise ValueError(
                "Secondary destination indices must belong to the enumerated support."
            )
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("A secondary jump process requires an explicit identity.")
        self.num_channels = int(self.rates.shape[1])
        self.process_id = process_id
        self.state_shape = (1,)
        self.mark_shape = ()

    def _index(self, state):
        value = jnp.asarray(state)[0]
        valid = (
            jnp.isfinite(value)
            & (value == jnp.floor(value))
            & (value >= 0)
            & (value < self.rates.shape[0])
        )
        return jnp.clip(value, 0, self.rates.shape[0] - 1).astype(jnp.int32), valid

    def intensities(self, t: ArrayLike, state: ArrayLike, args=None, /) -> Array:
        del t, args
        index, valid = self._index(state)
        return jnp.where(valid, self.rates[index], jnp.nan)

    def jump(
        self, state: ArrayLike, channel: ArrayLike, mark: ArrayLike, args=None, /
    ) -> Array:
        del mark, args
        index, valid = self._index(state)
        channel_value = jnp.asarray(channel)
        channel_valid = (
            (channel_value >= 0)
            & (channel_value < self.num_channels)
            & (channel_value == jnp.floor(channel_value))
        )
        safe_channel = jnp.clip(channel_value, 0, self.num_channels - 1).astype(jnp.int32)
        destination = self.destinations[index, safe_channel]
        allowed = valid & channel_valid & self.legal[index, safe_channel]
        return jnp.where(
            allowed,
            jnp.asarray([destination], dtype=jnp.asarray(state).dtype),
            jnp.asarray(state),
        )

    def sample_mark(self, key, t, state, channel, args=None, /) -> Array:
        del key, t, channel, args
        return jnp.asarray(0, dtype=jnp.asarray(state).dtype)


class CompiledSecondaryTarget(StrictModule):
    mask: Array
    name: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __call__(self, state: ArrayLike) -> Array:
        value = jnp.asarray(state)[0]
        valid = (value >= 0) & (value < self.mask.shape[0]) & (value == jnp.floor(value))
        index = jnp.clip(value, 0, self.mask.shape[0] - 1).astype(jnp.int32)
        return valid & self.mask[index]


@dataclass(frozen=True, slots=True)
class PreparedSecondaryKinetics:
    construct: NucleicAcidConstruct
    model: SecondaryEnergyModel
    association: AssociationConvention
    rate_law: SecondaryRateLaw
    states: tuple[SecondaryStructureState, ...]
    channel_pairs: tuple[tuple[NucleotideKey, NucleotideKey], ...]
    standard_free_energies: tuple[float, ...]
    free_energies: tuple[float, ...]
    process: SecondaryJumpProcess

    def encode(self, state: SecondaryStructureState) -> Array:
        if state.construct.fingerprint() != self.construct.fingerprint():
            raise ValueError("Secondary state belongs to a different construct.")
        for i, candidate in enumerate(self.states):
            if state.numeric_pairs == candidate.numeric_pairs:
                return jnp.asarray([i], dtype=jnp.int32)
        raise ValueError("State is outside this model's legal pairing support.")

    def decode(self, state: ArrayLike) -> SecondaryStructureState:
        """Host-only conversion from an exact scalar-index runtime state."""
        values = jax.device_get(jnp.asarray(state))
        if (
            values.shape != (1,)
            or not math.isfinite(float(values[0]))
            or int(values[0]) != values[0]
            or not 0 <= int(values[0]) < len(self.states)
        ):
            raise ValueError("Invalid compiled secondary state index.")
        return self.states[int(values[0])]

    def moves(self, state: SecondaryStructureState) -> tuple[SecondaryMove, ...]:
        index = int(self.encode(state)[0])
        legal = jax.device_get(self.process.legal[index])
        destinations = jax.device_get(self.process.destinations[index])
        result = []
        for channel, enabled in enumerate(legal):
            if not enabled:
                continue
            after = self.states[int(destinations[channel])]
            delta = after.partition.complex_count - state.partition.complex_count
            kind = (
                "join"
                if delta < 0
                else "split"
                if delta > 0
                else "formation"
                if after.pair_count > state.pair_count
                else "removal"
            )
            result.append(SecondaryMove(kind, self.channel_pairs[channel], state, after))
        return tuple(result)

    def target(
        self, name: str, predicate: Callable[[SecondaryStructureState], bool]
    ) -> CompiledSecondaryTarget:
        """Compile a named host macrostate predicate against this support."""
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Target predicates require a canonical name.")
        values = tuple(bool(predicate(state)) for state in self.states)
        return CompiledSecondaryTarget(
            mask=jnp.asarray(values),
            name=name,
            process_id=self.process.process_id,
            target_id=canonical_fingerprint((self.process.process_id, name, values)),
        )

    def pair_count_target(self, minimum: int) -> CompiledSecondaryTarget:
        if (
            type(minimum) is not int
            or not 0 <= minimum <= self.construct.nucleotide_count // 2
        ):
            raise ValueError("Pair-count target is outside construct capacity.")
        return self.target(
            f"pair-count-at-least-{minimum}", lambda state: state.pair_count >= minimum
        )

    def exact_state_target(
        self, state: SecondaryStructureState
    ) -> CompiledSecondaryTarget:
        self.encode(state)
        return self.target(
            f"exact-state-{state.fingerprint()}",
            lambda candidate: candidate.numeric_pairs == state.numeric_pairs,
        )

    def joined_target(self, strand_ids: Sequence[str]) -> CompiledSecondaryTarget:
        strands = tuple(strand_ids)
        if (
            len(strands) < 2
            or len(set(strands)) != len(strands)
            or not set(strands) <= set(self.construct.strand_ids)
        ):
            raise ValueError(
                "A joined target needs at least two distinct declared strands."
            )
        selected = frozenset(strands)
        return self.target(
            "joined-" + canonical_fingerprint(tuple(sorted(strands))),
            lambda state: any(
                selected <= set(block) for block in state.partition.complexes
            ),
        )

    def generator(
        self,
        states: Sequence[SecondaryStructureState] | None = None,
        *,
        boundary_policy="error",
    ) -> FiniteStateGenerator:
        selected = self.states if states is None else tuple(states)
        if not selected:
            raise ValueError("Finite generator state selection cannot be empty.")
        values = jnp.stack(tuple(self.encode(state) for state in selected))
        return finite_state_generator(
            self.process, values, boundary_policy=boundary_policy
        )

    def equilibrium_probabilities(self) -> Array:
        return jax.nn.softmax(-jnp.asarray(self.free_energies))

    def elementary_association_rate_constant(self, move: SecondaryMove) -> float:
        """Return one labelled elementary join's dilute coefficient in m³/(mol·time_unit).

        This is not a macroscopic first-passage-derived rate or a pseudo-first-
        order excess-bath approximation. The admitted finite-volume dilute
        well-mixed elementary association law is required explicitly.
        """
        if (
            self.association.mode != "fixed_volume"
            or self.rate_law.name != "association_metropolis"
            or move.kind != "join"
        ):
            raise ValueError(
                "Bimolecular conversion requires a fixed-volume association_metropolis join."
            )
        source = int(self.encode(move.before)[0])
        destination = int(self.encode(move.after)[0])
        if move not in self.moves(move.before):
            raise ValueError("The proposed join is not an elementary move in this model.")
        delta = (
            self.standard_free_energies[destination] - self.standard_free_energies[source]
        )
        return (
            self.rate_law.association_prefactor
            * math.exp(-max(delta, 0.0))
            / self.association.standard_concentration_mol_per_m3
        )


def prepare_secondary_kinetics(
    construct: NucleicAcidConstruct,
    model: SecondaryEnergyModel,
    association: AssociationConvention,
    rate_law: SecondaryRateLaw,
    /,
    *,
    temperature: float,
    max_states: int = 10000,
    max_channels: int = 4096,
) -> PreparedSecondaryKinetics:
    """Exhaustively compile a bounded ordered-planar labelled-strand CTMC.

    Capacity is a preparation refusal, never a reflecting state truncation.
    Linear DNA/RNA/hybrid constructs are supported only with an independently
    supplied matching chemistry artifact. ``temperature`` is Kelvin and must
    equal the source model temperature; no temperature extrapolation is implied.
    """
    if (
        type(max_states) is not int
        or max_states <= 0
        or type(max_channels) is not int
        or max_channels <= 0
    ):
        raise ValueError("State and channel capacities must be positive integers.")
    if temperature != model.temperature_kelvin:
        raise ValueError(
            "Execution temperature must match the parameter artifact calibration."
        )
    if any(construct.circular):
        raise ValueError(
            "The ordered-planar nicked-exterior profile requires linear strands."
        )
    chemistry = set(construct.polymer_types)
    expected = {"DNA", "RNA"} if model.chemistry == "DNA-RNA" else {model.chemistry}
    if chemistry != expected:
        raise ValueError(
            "Construct chemistry requires its own explicitly matching parameter profile."
        )
    keys, bases = construct.nucleotide_keys, construct.bases
    strand_type = dict(zip(construct.strand_ids, construct.polymer_types, strict=True))
    allowed = []
    for i in range(construct.nucleotide_count):
        for j in range(i + 1, construct.nucleotide_count):
            pair = bases[i] + bases[j]
            canonical = pair in ("AT", "TA", "CG", "GC", "AU", "UA")
            wobble = (
                model.pairing_rule == "watson_crick_wobble"
                and pair in ("GU", "UG")
                and strand_type[keys[i].strand_id]
                == strand_type[keys[j].strand_id]
                == "RNA"
            )
            short_loop = (
                keys[i].strand_id == keys[j].strand_id
                and j - i - 1 < model.minimum_hairpin_unpaired
            )
            if (canonical or wobble) and not short_loop:
                allowed.append((i, j))
                if len(allowed) > max_channels:
                    raise ValueError(
                        "Secondary channel capacity exceeded; no channels were truncated."
                    )
    # A dummy never-enabled channel represents a genuinely inert support, because
    # native Poisson clocks require positive channel capacity.
    channel_pairs = tuple((keys[i], keys[j]) for i, j in allowed)
    supports: list[tuple[tuple[int, int], ...]] = [()]
    lookup = {(): 0}
    cursor = 0
    while cursor < len(supports):
        current = supports[cursor]
        occupied = frozenset(i for pair in current for i in pair)
        for pair in allowed:
            if pair[0] in occupied or pair[1] in occupied:
                continue
            candidate = tuple(sorted((*current, pair)))
            if not _noncrossing(candidate) or candidate in lookup:
                continue
            if len(supports) >= max_states:
                raise ValueError(
                    "Secondary state capacity exceeded; exhaustive closure is required."
                )
            lookup[candidate] = len(supports)
            supports.append(candidate)
        cursor += 1
    states = tuple(
        SecondaryStructureState(construct, tuple((keys[i], keys[j]) for i, j in support))
        for support in supports
    )
    standard = tuple(model.standard_free_energy(state) for state in states)
    counts = tuple(
        len(construct.strand_ids) - state.partition.complex_count for state in states
    )
    energies = tuple(
        value + count * association.log_standard_volume
        for value, count in zip(standard, counts, strict=True)
    )
    destinations, legal, rates = [], [], []
    for source, support in enumerate(supports):
        row_destination, row_legal, row_rates = [], [], []
        for pair in allowed:
            next_support = (
                tuple(value for value in support if value != pair)
                if pair in support
                else tuple(sorted((*support, pair)))
            )
            destination = lookup.get(next_support, source)
            enabled = destination != source
            rate = 0.0
            if enabled:
                delta = energies[destination] - energies[source]
                associations = counts[destination] - counts[source]
                attempt = (
                    rate_law.association_prefactor
                    if associations
                    else rate_law.unimolecular_prefactor
                )
                if rate_law.name == "association_metropolis" and associations:
                    standard_delta = standard[destination] - standard[source]
                    log_rate = (
                        math.log(attempt)
                        - max(standard_delta, 0.0)
                        - (association.log_standard_volume if associations > 0 else 0.0)
                    )
                elif rate_law.name == "symmetric_barrier":
                    log_rate = math.log(attempt) - 0.5 * delta
                else:
                    log_rate = math.log(attempt) - max(delta, 0.0)
                if not math.isfinite(log_rate) or not -700 <= log_rate <= 700:
                    raise ValueError(
                        "Rate dynamic range is unsupported; rescale physical time or qualify another model."
                    )
                rate = math.exp(log_rate)
            row_destination.append(destination)
            row_legal.append(enabled)
            row_rates.append(rate)
        destinations.append(row_destination or [source])
        legal.append(row_legal or [False])
        rates.append(row_rates or [0.0])
    process_id = canonical_fingerprint(
        (
            "secondary-pair-toggle-ctmc",
            construct.fingerprint(),
            model.model_id,
            association.fingerprint(),
            rate_law.fingerprint(),
            supports,
        )
    )
    process = SecondaryJumpProcess(destinations, legal, rates, process_id=process_id)
    if not bool(jnp.all(jnp.isfinite(process.rates))) or bool(
        jnp.any(jnp.asarray(legal) & (process.rates <= 0))
    ):
        raise ValueError(
            "Rate precision loses a legal transition; enable an adequate numeric precision."
        )
    return PreparedSecondaryKinetics(
        construct,
        model,
        association,
        rate_law,
        states,
        channel_pairs,
        standard,
        energies,
        process,
    )


__all__ = [
    "CompiledSecondaryTarget",
    "PreparedSecondaryKinetics",
    "SecondaryJumpProcess",
    "prepare_secondary_kinetics",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import TypeAlias, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ...typing import checked, PRNGKey
from .._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
)
from .._dynamics import (
    _underlying_program,
    AtomisticDynamicsState,
    PreparedAtomisticDynamics,
)
from .._term_state import AtomisticTermStateBindings
from .._thermodynamic import AtomisticBiasRowBinding, PreparedThermodynamicStateTable
from .._units import AtomisticUnitSystem
from ._bias import (
    AtomisticBiasTermState,
    authenticate_frozen_bias_histories,
    HillHistoryBiasForm,
    PreparedAtomisticBiasTerm,
    with_bias_controls,
    with_bias_history,
)
from ._bias_history import (
    AtomisticBiasHistoryPlan,
    bias_history_digest,
    BiasHistoryOwnership,
    deposit_bias_history,
)
from ._cv_geometry import CollectiveVariableGeometry


class BiasHistoryLabel(StrEnum):
    """Owner tag of a state-owned kernel deposited after a label move.

    ``ORIGIN`` tags the kernel with the state whose Hamiltonian sampled the
    configuration (the pre-move label); ``TARGET`` with the post-move label.
    """

    ORIGIN = "origin"
    TARGET = "target"


_EXCHANGE_STREAM = 0x45584348
_SAMS_STREAM = 0x53414D53
_UINT32_MAX = np.iinfo(np.uint32).max

_ExchangeOutcome: TypeAlias = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array
]


def _identity_token(identifier: str, /) -> Array:
    """Continuation token: the ``uint32[8]`` words of one canonical digest."""
    return IdentityWords.from_fingerprint(str(identifier)).words


def _stack_identities(values: tuple[IdentityWords, ...], /) -> IdentityWords:
    """Stack unbatched identities into one ``(n, 8)`` word table."""
    if not values:
        return IdentityWords(jnp.zeros((0, 8), dtype=jnp.uint32))
    return IdentityWords(jnp.stack(tuple(value.words for value in values)))


def _decode_identities(identities: IdentityWords, /) -> tuple[str, ...]:
    """Host decoding of a stacked ``(n, 8)`` identity table."""
    return tuple(
        IdentityWords(identities.words[row]).fingerprint()
        for row in range(identities.words.shape[0])
    )


def _history_declarations(
    kinds: tuple[str, ...], identities: IdentityWords, /
) -> tuple[str, ...]:
    """Host: exact ``"adaptive"``/``"frozen:<digest>"`` row declarations."""
    digests = _decode_identities(identities)
    return tuple(
        "adaptive" if kind == "adaptive" else f"frozen:{digest}"
        for kind, digest in zip(kinds, digests, strict=True)
    )


def _bias_term_state(
    bindings: AtomisticTermStateBindings, slot: str, /
) -> AtomisticBiasTermState:
    """The (possibly lane-batched) bias state bound to one declared bias slot."""
    state = bindings[slot]
    if not isinstance(state, AtomisticBiasTermState):
        raise TypeError(f"Bias slot {slot!r} does not bind an AtomisticBiasTermState.")
    return state


class AtomisticCanonicalSamplingQualification(StrictModule, NonTrainableState):
    """Authenticated qualification bound to one dynamics and target table.

    A canonical target exists only for a complete scalar Hamiltonian:
    a runtime applying nonconservative force ports (force-only ABF) is
    refused for every table, including a single state.  The qualification
    records whether every table row is one fixed Hamiltonian
    (``fixed_row_hamiltonians``: no row declares an ``"adaptive"`` history);
    consumers of equilibrium distributions require it through
    :meth:`require_fixed_row_hamiltonians` instead of issuing their own
    certificate.

    PyTree: ``sampling_exact`` and ``fixed_row_hamiltonians`` are static
    declaration flags; the bias bound is a dynamic float64 scalar and the
    bound dynamics, table, measure and qualification digests are dynamic
    words (``dynamics_id``, ``thermodynamic_table_id``, ``measure_id``,
    ``qualification_id`` decode them on the host).  The caller's
    ``evidence_id`` enters only the exact qualification digest.
    """

    dynamics_identity: IdentityWords
    thermodynamic_table_identity: IdentityWords
    measure_identity: IdentityWords
    sampling_exact: bool = eqx.field(static=True)
    fixed_row_hamiltonians: bool = eqx.field(static=True)
    sampling_bias_bound: Array
    qualification_identity: IdentityWords

    @checked
    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        thermodynamic: PreparedThermodynamicStateTable,
        evidence_id: str,
        /,
        *,
        sampling_exact: bool,
        sampling_bias_bound: float,
    ) -> None:
        thermodynamic.validate_dynamics(dynamics)
        dynamics.require_conservative_force_definition(
            "Canonical-sampling qualification (every multistate analysis runtime)"
        )
        evidence = str(evidence_id)
        bound = float(sampling_bias_bound)
        if (
            not evidence
            or evidence != evidence.strip()
            or not isinstance(sampling_exact, bool)
            or not math.isfinite(bound)
            or bound < 0.0
            or (sampling_exact and bound != 0.0)
        ):
            raise ValueError(
                "Canonical-sampling evidence, exactness, or bias bound is invalid."
            )
        fixed = not any(thermodynamic.bias_adaptive)
        self.dynamics_identity = dynamics.prepared_identity
        self.thermodynamic_table_identity = thermodynamic.table_identity
        self.measure_identity = thermodynamic.phase_space_measure_identity
        self.sampling_exact = sampling_exact
        self.fixed_row_hamiltonians = fixed
        self.sampling_bias_bound = jnp.asarray(bound, dtype=jnp.float64)
        self.qualification_identity = IdentityWords.of(
            {
                "kind": "atomistic-canonical-sampling-qualification",
                "dynamics": dynamics.prepared_id,
                "integrator": dynamics.integrator.plan_id,
                "thermodynamic_table": thermodynamic.table_id,
                "measure": thermodynamic.phase_space_measure_id,
                "program": thermodynamic.program_id,
                "conservative_force_definition": True,
                "fixed_row_hamiltonians": fixed,
                "evidence": evidence,
                "sampling_exact": sampling_exact,
                "sampling_bias_bound": bound,
            }
        )

    def require_fixed_row_hamiltonians(self, consumer: str, /) -> None:
        """Host refusal unless every qualified row is one fixed Hamiltonian."""
        if not self.fixed_row_hamiltonians:
            raise ValueError(
                f"{consumer} needs fixed-Hamiltonian rows; this qualification's "
                "table declares an advancing ('adaptive') bias history."
            )

    @property
    def dynamics_id(self) -> str:
        """Host decoding of the qualified dynamics-runtime digest."""
        return self.dynamics_identity.fingerprint()

    @property
    def thermodynamic_table_id(self) -> str:
        """Host decoding of the qualified thermodynamic-table digest."""
        return self.thermodynamic_table_identity.fingerprint()

    @property
    def measure_id(self) -> str:
        """Host decoding of the qualified phase-space measure digest."""
        return self.measure_identity.fingerprint()

    @property
    def qualification_id(self) -> str:
        """Host decoding of the canonical qualification digest."""
        return self.qualification_identity.fingerprint()


class AtomisticReplicaExchangePlan(StrictModule, NonTrainableState):
    """Deterministic odd-even adjacent exchange schedule.

    ``exchange_interval`` is a static cadence; ``realization_id`` is a dynamic
    ``uint32`` random-stream tag and ``plan_identity`` the exact plan digest
    (``plan_id`` on the host).
    """

    exchange_interval: int = eqx.field(static=True)
    realization_id: Array
    plan_identity: IdentityWords

    def __init__(self, exchange_interval: int = 1, /, *, realization_id: int = 0) -> None:
        interval = int(exchange_interval)
        realization = int(realization_id)
        if interval <= 0 or realization < 0 or realization > _UINT32_MAX:
            raise ValueError("Exchange interval and realization ID are invalid.")
        self.exchange_interval = interval
        self.realization_id = jnp.asarray(realization, dtype=jnp.uint32)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-replica-exchange",
                "interval": interval,
                "realization_id": realization,
                "proposal": "adjacent-odd-even",
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical exchange-plan digest."""
        return self.plan_identity.fingerprint()

    def scheduled(self, next_iteration: ArrayLike, /) -> Array:
        iteration = jnp.asarray(next_iteration, dtype=jnp.int64)
        return (iteration > 0) & (iteration % self.exchange_interval == 0)


class AtomisticSAMSPlan(StrictModule, NonTrainableState):
    """Finite-adaptation stochastic approximation over thermodynamic labels.

    ``move_interval`` is a static cadence; the adaptation length, gain
    schedule and realization tag are dynamic scalars and ``plan_identity``
    is the exact plan digest (``plan_id`` on the host).
    """

    target_probabilities: Array
    move_interval: int = eqx.field(static=True)
    adaptation_steps: Array
    gain_exponent: Array
    initial_gain: Array
    realization_id: Array
    plan_identity: IdentityWords

    def __init__(
        self,
        target_probabilities: ArrayLike,
        /,
        *,
        move_interval: int = 1,
        adaptation_steps: int = 1_000,
        gain_exponent: float = 0.6,
        initial_gain: float = 1.0,
        realization_id: int = 0,
    ) -> None:
        target = np.asarray(target_probabilities, dtype=np.float64).reshape((-1,))
        interval = int(move_interval)
        steps = int(adaptation_steps)
        exponent = float(gain_exponent)
        gain = float(initial_gain)
        realization = int(realization_id)
        if (
            target.size < 2
            or np.any(~np.isfinite(target))
            or np.any(target <= 0.0)
            or not np.isclose(np.sum(target), 1.0)
            or interval <= 0
            or steps < 0
            or not math.isfinite(exponent)
            or exponent <= 0.5
            or exponent > 1.0
            or not math.isfinite(gain)
            or gain <= 0.0
            or realization < 0
            or realization > _UINT32_MAX
        ):
            raise ValueError("SAMS target, cadence, adaptation, or gain is invalid.")
        normalized = target / np.sum(target)
        self.target_probabilities = jnp.asarray(normalized)
        self.move_interval = interval
        self.adaptation_steps = jnp.asarray(steps, dtype=jnp.int64)
        self.gain_exponent = jnp.asarray(exponent, dtype=jnp.float64)
        self.initial_gain = jnp.asarray(gain, dtype=jnp.float64)
        self.realization_id = jnp.asarray(realization, dtype=jnp.uint32)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-sams",
                "target": array_tree_fingerprint(normalized),
                "move_interval": interval,
                "adaptation_steps": steps,
                "gain_exponent": exponent,
                "initial_gain": gain,
                "realization_id": realization,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical SAMS-plan digest."""
        return self.plan_identity.fingerprint()

    def scheduled(self, next_iteration: ArrayLike, /) -> Array:
        iteration = jnp.asarray(next_iteration, dtype=jnp.int64)
        return (iteration > 0) & (iteration % self.move_interval == 0)


class AtomisticSAMSState(StrictModule):
    log_weights: Array
    visit_counts: Array
    adaptation_index: Array
    frozen: Array


class AtomisticReducedPotentialEvaluation(StrictModule):
    """Dimensionless dense K-by-R reduced potentials and explicit coverage.

    ``table_identity`` is the evaluated table's exact digest (``table_id``).
    """

    values: Array
    successful: Array
    state_indices: Array
    table_identity: IdentityWords

    @property
    def table_id(self) -> str:
        """Host decoding of the evaluated thermodynamic-table digest."""
        return self.table_identity.fingerprint()


class AtomisticMultistateState(StrictModule):
    """Batched multistate state; ``runtime_identity`` binds it to its runtime.

    ``runtime_identity`` holds the owning runtime's exact ``prepared_id`` as
    dynamic words (``plan_id`` decodes it on the host).
    """

    dynamics: AtomisticDynamicsState
    state_at_replica: Array
    reduced_potential_cache: Array
    reduced_potential_valid: Array
    cache_iteration: Array
    iteration_index: Array
    draw_index: Array
    exchange_action_counter: Array
    exchange_parity: Array
    sams_action_counter: Array
    sams: AtomisticSAMSState
    root_key: Array
    segment_index: Array
    continuation_token: Array
    runtime_identity: IdentityWords

    @property
    def plan_id(self) -> str:
        """Host decoding of the owning multistate runtime digest."""
        return self.runtime_identity.fingerprint()


class AtomisticMultistateIteration(StrictModule):
    """One atomic iteration and its complete bias-event evidence.

    ``bias_owner_keys`` (``(R, B)``) are the history owner keys every replica's
    bias terms evaluate after the assignment move (``bias_term_slots`` order);
    ``bias_history_revision`` (``(H,)``) is each history's ``written`` counter
    after the iteration (``bias_history_term_slots`` order) and
    ``bias_deposited`` whether that history received this iteration's event.
    """

    candidate_state: AtomisticMultistateState
    state: AtomisticMultistateState
    reduced_potentials: Array
    coverage: Array
    origin_state: Array
    state_at_replica_after: Array
    pair_indices: Array
    exchange_attempted: Array
    exchange_accepted: Array
    exchange_log_acceptance: Array
    sams_attempted: Array
    sams_changed: Array
    sams_adapting: Array
    dynamics_accepted: Array
    barostat_attempted: Array
    barostat_accepted: Array
    equilibrium_sample: Array
    bias_adapting: Array
    bias_deposited: Array
    bias_exhausted: Array
    bias_owner_keys: Array
    bias_history_revision: Array
    successful: Array
    runtime_identity: IdentityWords

    @property
    def plan_id(self) -> str:
        """Host decoding of the producing multistate runtime digest."""
        return self.runtime_identity.fingerprint()


def _admit_transition_kernels(
    thermodynamic: PreparedThermodynamicStateTable,
    replica_count: int,
    exchange: AtomisticReplicaExchangePlan | None,
    sams: AtomisticSAMSPlan | None,
    /,
) -> None:
    """At most one label kernel: a dense R=K exchange ladder or a K-state SAMS."""
    if exchange is not None and not isinstance(exchange, AtomisticReplicaExchangePlan):
        raise TypeError("exchange must be AtomisticReplicaExchangePlan or None.")
    if sams is not None and not isinstance(sams, AtomisticSAMSPlan):
        raise TypeError("sams must be AtomisticSAMSPlan or None.")
    if exchange is not None and sams is not None:
        raise ValueError("Replica exchange and SAMS are distinct transition kernels.")
    if exchange is not None and (
        replica_count != thermodynamic.state_count or thermodynamic.state_count < 2
    ):
        raise ValueError("Dense replica exchange requires R equal to K and K >= 2.")
    if sams is not None and sams.target_probabilities.shape != (
        thermodynamic.state_count,
    ):
        raise ValueError("SAMS target probabilities must have one entry per state.")


def _admit_barostat_schedule(
    thermodynamic: PreparedThermodynamicStateTable,
    barostat: IsotropicMonteCarloBarostatPlan | None,
    barostat_interval: int | None,
    /,
) -> int | None:
    """The positive barostat cadence, scheduled exactly when every state is NPT."""
    if barostat is not None and not isinstance(barostat, IsotropicMonteCarloBarostatPlan):
        raise TypeError("barostat must be IsotropicMonteCarloBarostatPlan or None.")
    interval = None if barostat_interval is None else int(barostat_interval)
    if (barostat is None) != (interval is None) or (
        interval is not None and interval <= 0
    ):
        raise ValueError(
            "Barostat and positive barostat_interval must be supplied together."
        )
    npt = np.asarray(thermodynamic.pressure_mask)
    if (np.any(npt) and (barostat is None or not np.all(npt))) or (
        not np.any(npt) and barostat is not None
    ):
        raise ValueError(
            "A scheduled barostat is required exactly when every state is NPT."
        )
    return interval


def _admit_replica_layout(
    replica_count: int,
    exchange: AtomisticReplicaExchangePlan | None,
    chain_indices: ArrayLike | None,
    dependence_group_indices: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-replica chain and dependence-group indices.

    Defaults: one chain per replica; an exchange-connected ladder shares one
    dependence group, otherwise each chain is its own group.
    """
    chains = (
        np.arange(replica_count, dtype=np.int32)
        if chain_indices is None
        else np.asarray(chain_indices, dtype=np.int32).reshape((-1,))
    )
    groups = (
        np.zeros((replica_count,), dtype=np.int32)
        if dependence_group_indices is None and exchange is not None
        else chains.copy()
        if dependence_group_indices is None
        else np.asarray(dependence_group_indices, dtype=np.int32).reshape((-1,))
    )
    if (
        chains.shape != (replica_count,)
        or groups.shape != (replica_count,)
        or np.any(chains < 0)
        or np.any(groups < 0)
    ):
        raise ValueError(
            "Chain and dependence-group indices must be non-negative vectors of length R."
        )
    if exchange is not None and len(set(groups.tolist())) != 1:
        raise ValueError(
            "An exchange-connected replica ladder must share one dependence group."
        )
    return chains, groups


def _admit_run_declaration(repeat_index: int, run_id: str, /) -> tuple[int, str]:
    """A ``uint32`` repeat index and a canonical nonempty run label."""
    repeat = int(repeat_index)
    identifier = str(run_id)
    if (
        repeat < 0
        or repeat > _UINT32_MAX
        or not identifier
        or identifier != identifier.strip()
    ):
        raise ValueError("repeat_index or run_id is invalid.")
    return repeat, identifier


class AtomisticMultistatePlan(StrictModule, NonTrainableState):
    """Replica layout, transition kernels and run declaration of one ensemble.

    PyTree: replica/state counts (shapes), barostat cadence, the declared
    ``run_id`` label and ``history_label`` are static; ``repeat_index`` is a
    dynamic ``uint32`` scalar and the exact plan digest is the dynamic
    ``plan_identity`` (``plan_id`` decodes it on the host).
    """

    thermodynamic: PreparedThermodynamicStateTable
    qualification: AtomisticCanonicalSamplingQualification
    replica_ids: Array
    chain_indices: Array
    dependence_group_indices: Array
    exchange: AtomisticReplicaExchangePlan | None
    sams: AtomisticSAMSPlan | None
    barostat: IsotropicMonteCarloBarostatPlan | None
    replica_count: int = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    barostat_interval: int | None = eqx.field(static=True)
    repeat_index: Array
    run_id: str = eqx.field(static=True)
    history_label: BiasHistoryLabel = eqx.field(static=True)
    plan_identity: IdentityWords

    @checked
    def __init__(
        self,
        thermodynamic: PreparedThermodynamicStateTable,
        replica_ids: ArrayLike,
        /,
        *,
        qualification: AtomisticCanonicalSamplingQualification,
        exchange: AtomisticReplicaExchangePlan | None = None,
        sams: AtomisticSAMSPlan | None = None,
        barostat: IsotropicMonteCarloBarostatPlan | None = None,
        barostat_interval: int | None = None,
        chain_indices: ArrayLike | None = None,
        dependence_group_indices: ArrayLike | None = None,
        repeat_index: int = 0,
        run_id: str,
        history_label: BiasHistoryLabel = BiasHistoryLabel.ORIGIN,
    ) -> None:
        if (
            qualification.dynamics_id != thermodynamic.dynamics_id
            or qualification.thermodynamic_table_id != thermodynamic.table_id
            or qualification.measure_id != thermodynamic.phase_space_measure_id
        ):
            raise ValueError(
                "Canonical-sampling qualification belongs to another dynamics/table target."
            )
        replicas = np.asarray(replica_ids, dtype=np.int64).reshape((-1,))
        replica_count = replicas.size
        if (
            replica_count <= 0
            or len(set(replicas.tolist())) != replica_count
            or np.any(replicas < 0)
        ):
            raise ValueError("Replica IDs must be a nonempty unique non-negative vector.")
        _admit_transition_kernels(thermodynamic, replica_count, exchange, sams)
        if not bool(np.all(np.asarray(thermodynamic.temperature_mask))):
            raise ValueError(
                "Multistate reduced potentials require temperature for every state."
            )
        interval = _admit_barostat_schedule(thermodynamic, barostat, barostat_interval)
        chains, groups = _admit_replica_layout(
            replica_count, exchange, chain_indices, dependence_group_indices
        )
        repeat, identifier = _admit_run_declaration(repeat_index, run_id)
        self.thermodynamic = thermodynamic
        self.qualification = qualification
        self.replica_ids = jnp.asarray(replicas)
        self.chain_indices = jnp.asarray(chains)
        self.dependence_group_indices = jnp.asarray(groups)
        self.exchange = exchange
        self.sams = sams
        self.barostat = barostat
        self.replica_count = replica_count
        self.state_count = thermodynamic.state_count
        self.barostat_interval = interval
        self.repeat_index = jnp.asarray(repeat, dtype=jnp.uint32)
        self.run_id = identifier
        if not isinstance(history_label, BiasHistoryLabel):
            raise TypeError("history_label must be a BiasHistoryLabel.")
        self.history_label = history_label
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-multistate-plan",
                "qualification": qualification.qualification_id,
                "thermodynamic": thermodynamic.table_id,
                "replica_ids": array_tree_fingerprint(replicas),
                "chain_indices": array_tree_fingerprint(chains),
                "dependence_groups": array_tree_fingerprint(groups),
                "exchange": None if exchange is None else exchange.plan_id,
                "sams": None if sams is None else sams.plan_id,
                "barostat": None if barostat is None else barostat.plan_id,
                "barostat_interval": interval,
                "repeat_index": repeat,
                "run_id": identifier,
                "history_label": history_label.value,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical multistate-plan digest."""
        return self.plan_identity.fingerprint()

    def prepare(
        self, dynamics: PreparedAtomisticDynamics, /
    ) -> "PreparedAtomisticMultistate":
        return PreparedAtomisticMultistate(self, dynamics)


def _hamiltonian_bias_terms(
    dynamics: PreparedAtomisticDynamics, layout: tuple[str, ...], /
) -> tuple[dict[str, PreparedAtomisticBiasTerm], tuple[PreparedAtomisticBiasTerm, ...]]:
    """The Hamiltonian's bias terms by slot, and the row layout's terms in order."""
    terms = {
        term.name: term
        for term in _underlying_program(dynamics.potential).terms
        if isinstance(term, PreparedAtomisticBiasTerm)
    }
    if any(slot not in terms for slot in layout):
        raise ValueError("Every bias row term must be a bias term of the Hamiltonian.")
    return terms, tuple(terms[slot] for slot in layout)


def _stack_row_controls(
    term: PreparedAtomisticBiasTerm,
    position: int,
    rows: tuple[AtomisticBiasRowBinding, ...],
    /,
) -> Array | None:
    """``(K, control_size)`` controls of one term: declared in every row or none."""
    values = [row.controls[position] for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("A bias term needs controls in every row or in none.")
    stacked = jnp.stack(present)
    if stacked.shape[1] != term.plan.form.control_size:
        raise ValueError("Row controls do not match the bias form.")
    return stacked


def _stack_bias_row_parameters(
    table: PreparedThermodynamicStateTable,
    rows: tuple[AtomisticBiasRowBinding, ...],
    bias_terms: tuple[PreparedAtomisticBiasTerm, ...],
    /,
) -> tuple[tuple[Array | None, ...], tuple[Array, ...], tuple[Array, ...]]:
    """Per-term row controls, owner keys and couplings (slot order).

    A well-tempered history shared across rows admits only rows at its owner
    temperature.
    """
    row_controls: list[Array | None] = []
    row_owners: list[Array] = []
    row_couplings: list[Array] = []
    for position, term in enumerate(bias_terms):
        row_controls.append(_stack_row_controls(term, position, rows))
        row_owners.append(
            jnp.asarray([row.owner_keys[position] for row in rows], dtype=jnp.int32)
        )
        row_couplings.append(
            jnp.asarray([row.couplings[position] for row in rows], dtype=jnp.float64)
        )
        history = term.plan.form.history
        if history is not None and history.bias_factor is not None:
            temperatures = np.asarray(table.temperature)
            temperature = history.temperature
            if temperature is None:
                raise ValueError(
                    "Well-tempered deposition requires the history-owner temperature."
                )
            if not np.allclose(temperatures, temperature, rtol=1e-12, atol=0.0):
                raise ValueError(
                    "A well-tempered history shared across rows requires every "
                    "row at its owner temperature; mixed-temperature damping is "
                    "not a derived well-tempered estimator."
                )
    return tuple(row_controls), tuple(row_owners), tuple(row_couplings)


def _history_owner_resolution(
    term: PreparedAtomisticBiasTerm,
    history: AtomisticBiasHistoryPlan,
    declaration: str,
    keys: list[int],
    replica_count: int,
    /,
) -> str:
    """Which history owner a row evaluates, with its owner-key contract.

    A shared history has one owner; a state-owned or *frozen* walker-owned
    history evaluates the row's declared owner key (one fixed Hamiltonian per
    row, the owner its potential identity binds); an adaptive walker-owned
    history evaluates each replica lane's own kernels (never analysis rows).
    """
    if history.ownership is BiasHistoryOwnership.SHARED:
        resolution = "shared"
    elif history.ownership is BiasHistoryOwnership.WALKER and declaration == "adaptive":
        resolution = "lane"
    else:
        resolution = "row"
    if resolution == "row" and max(keys) >= history.owner_count:
        raise ValueError(
            f"History {term.name!r} owner keys exceed its declared owner count."
        )
    if resolution == "lane" and replica_count > history.owner_count:
        raise ValueError(
            f"History {term.name!r} declares fewer walker owners than replicas."
        )
    if resolution == "lane" and any(keys):
        raise ValueError(
            f"Adaptive walker-owned history {term.name!r} evaluates each "
            "replica's own kernels; its rows cannot select an owner key."
        )
    return resolution


def _resolve_bias_histories(
    rows: tuple[AtomisticBiasRowBinding, ...],
    bias_terms: tuple[PreparedAtomisticBiasTerm, ...],
    replica_count: int,
    /,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """History slots, their common row declarations, and every owner resolution.

    A history term needs one common ``"adaptive"``/``"frozen:<digest>"``
    declaration in every row; a term without history declares none.
    """
    history_slots: list[str] = []
    declarations: list[str] = []
    resolutions: list[str] = []
    for position, term in enumerate(bias_terms):
        declared = {row.histories[position] for row in rows}
        history = term.plan.form.history
        if history is None:
            if declared != {None}:
                raise ValueError(
                    "Only adaptive-history bias terms declare 'adaptive' or a "
                    "frozen digest."
                )
            resolutions.append("none")
            continue
        if len(declared) != 1 or None in declared:
            raise ValueError(
                f"History term {term.name!r} needs one common declaration in "
                "every row: 'adaptive' or one 'frozen:<digest>'."
            )
        declaration = str(next(iter(declared)))
        keys = [int(np.asarray(row.owner_keys[position])) for row in rows]
        resolutions.append(
            _history_owner_resolution(term, history, declaration, keys, replica_count)
        )
        history_slots.append(term.name)
        declarations.append(declaration)
    return tuple(history_slots), tuple(declarations), tuple(resolutions)


def _require_bound_bias_terms(
    terms: dict[str, PreparedAtomisticBiasTerm],
    layout: tuple[str, ...],
    state_count: int,
    /,
) -> None:
    """Rows bind every history term, and every stateful term when K > 1."""
    unbound = {
        term.name for term in terms.values() if term.plan.form.history is not None
    } - set(layout)
    if unbound:
        raise ValueError(
            "Every adaptive-history bias term must be bound by the table rows "
            f"with an 'adaptive' or frozen declaration; unbound: {sorted(unbound)}."
        )
    stateful = {term.name for term in terms.values() if term.plan.stateful}
    if stateful - set(layout) and state_count > 1:
        raise ValueError(
            "A multi-state table must bind every stateful bias term in its rows."
        )


class PreparedAtomisticMultistate(StrictModule):
    """Multistate runtime over one dynamics runtime and thermodynamic table.

    Every bias term carrying an adaptive history must be bound by the rows
    with one common declaration: ``"adaptive"`` (deposits at committed
    iterations; its draws are not equilibrium samples) or one
    ``"frozen:<digest>"`` (a fixed Hamiltonian whose bound history content is
    authenticated against the digest at initialization, checkpoint and
    analysis).

    ``bias_owner_resolutions`` (``bias_term_slots`` order) records which
    history owner each row evaluates: ``"none"`` (no history), ``"shared"``,
    ``"row"`` (state-owned, or a frozen walker-owned history: the row's
    declared owner key, so every row is one fixed Hamiltonian under complete
    cross-evaluation) or ``"lane"`` (an adaptive walker-owned history: each
    replica lane evaluates and deposits its own kernels).  Lane-resolved
    rows depend on the sample column and are never analysis rows.

    PyTree: bias rows are keyed by the bias terms' static slot names
    (``bias_term_slots``/``bias_history_term_slots``) and the declaration
    kinds (``bias_history_kinds``: ``"adaptive"``/``"frozen"``) are static
    selectors; frozen digests are the dynamic ``bias_history_identities``
    words (zero words for adaptive histories) and the runtime and initial
    continuation digests are dynamic words.  Host properties
    ``bias_term_ids``, ``bias_history_term_ids``,
    ``bias_history_declarations``, ``prepared_id`` and
    ``initial_continuation_id`` keep the exact former strings.
    """

    plan: AtomisticMultistatePlan
    qualification: AtomisticCanonicalSamplingQualification
    dynamics: PreparedAtomisticDynamics
    thermodynamic: PreparedThermodynamicStateTable
    bias_terms: tuple[PreparedAtomisticBiasTerm, ...]
    bias_row_controls: tuple[Array | None, ...]
    bias_row_owners: tuple[Array, ...]
    bias_row_couplings: tuple[Array, ...]
    bias_term_slots: tuple[str, ...] = eqx.field(static=True)
    bias_owner_resolutions: tuple[str, ...] = eqx.field(static=True)
    bias_history_term_slots: tuple[str, ...] = eqx.field(static=True)
    bias_history_kinds: tuple[str, ...] = eqx.field(static=True)
    bias_history_identities: IdentityWords
    history_adaptive: bool = eqx.field(static=True)
    prepared_identity: IdentityWords
    initial_continuation_identity: IdentityWords

    @checked
    def __init__(
        self,
        plan: AtomisticMultistatePlan,
        dynamics: PreparedAtomisticDynamics,
        /,
    ) -> None:
        plan.thermodynamic.validate_dynamics(dynamics)
        if (
            plan.qualification.dynamics_id != dynamics.prepared_id
            or plan.qualification.thermodynamic_table_id != plan.thermodynamic.table_id
        ):
            raise ValueError(
                "Canonical-sampling qualification is not bound to this prepared runtime."
            )
        table = plan.thermodynamic
        # The table binds the same bias layout in every row, or none in all.
        rows = tuple(row for row in table.bias_rows if row is not None)
        layout = rows[0].term_slots if rows else ()
        terms, bias_terms = _hamiltonian_bias_terms(dynamics, layout)
        row_controls, row_owners, row_couplings = _stack_bias_row_parameters(
            table, rows, bias_terms
        )
        history_slots, declarations, resolutions = _resolve_bias_histories(
            rows, bias_terms, plan.replica_count
        )
        _require_bound_bias_terms(terms, layout, table.state_count)
        kinds = tuple(
            "adaptive" if declaration == "adaptive" else "frozen"
            for declaration in declarations
        )
        self.plan = plan
        self.qualification = plan.qualification
        self.dynamics = dynamics
        self.thermodynamic = table
        self.bias_terms = bias_terms
        self.bias_row_controls = row_controls
        self.bias_row_owners = row_owners
        self.bias_row_couplings = row_couplings
        self.bias_term_slots = tuple(layout)
        self.bias_owner_resolutions = resolutions
        self.bias_history_term_slots = history_slots
        self.bias_history_kinds = kinds
        self.bias_history_identities = _stack_identities(
            tuple(
                IdentityWords(jnp.zeros((8,), dtype=jnp.uint32))
                if kind == "adaptive"
                else IdentityWords.from_fingerprint(declaration.removeprefix("frozen:"))
                for kind, declaration in zip(kinds, declarations, strict=True)
            )
        )
        self.history_adaptive = "adaptive" in kinds
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-multistate",
                "plan": plan.plan_id,
                "qualification": plan.qualification.qualification_id,
                "dynamics": dynamics.prepared_id,
                "thermodynamic": plan.thermodynamic.table_id,
            }
        )
        self.prepared_identity = IdentityWords.from_fingerprint(prepared_id)
        self.initial_continuation_identity = IdentityWords.of(
            {"kind": "atomistic-multistate-initial", "runtime": prepared_id}
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical multistate-runtime digest."""
        return self.prepared_identity.fingerprint()

    @property
    def initial_continuation_id(self) -> str:
        """Host decoding of the runtime's initial continuation digest."""
        return self.initial_continuation_identity.fingerprint()

    @property
    def bias_term_ids(self) -> tuple[str, ...]:
        """Host: exact scientific digests of the row bias terms (slot order)."""
        return tuple(term.term_id for term in self.bias_terms)

    @property
    def bias_history_term_ids(self) -> tuple[str, ...]:
        """Host: exact scientific digests of the history terms (slot order)."""
        by_slot = {term.name: term for term in self.bias_terms}
        return tuple(by_slot[slot].term_id for slot in self.bias_history_term_slots)

    @property
    def bias_history_declarations(self) -> tuple[str, ...]:
        """Host: exact row declarations ``"adaptive"``/``"frozen:<digest>"``."""
        return _history_declarations(
            self.bias_history_kinds, self.bias_history_identities
        )

    @property
    def bias_row_owner_keys(self) -> Array:
        """Declared ``(K, B)`` owner key of every row and bias term (slot order)."""
        if not self.bias_row_owners:
            return jnp.zeros((self.plan.state_count, 0), dtype=jnp.int32)
        return jnp.stack(self.bias_row_owners, axis=-1)

    @staticmethod
    def _semantic_key(
        root_key: Array, stream: ArrayLike, counter: Array, identity: Array
    ) -> PRNGKey:
        key = jr.wrap_key_data(root_key)
        key = jr.fold_in(key, jnp.asarray(stream, dtype=jnp.uint32))
        key = jr.fold_in(key, jnp.asarray(counter, dtype=jnp.uint32))
        unsigned = jnp.asarray(identity, dtype=jnp.uint64)
        key = jr.fold_in(key, (unsigned >> jnp.uint64(32)).astype(jnp.uint32))
        return jr.fold_in(key, (unsigned & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32))

    def initialize(
        self,
        states: Iterable[AtomisticDynamicsState],
        state_at_replica: ArrayLike,
        key: PRNGKey,
        /,
    ) -> AtomisticMultistateState:
        lanes = tuple(states)
        if len(lanes) != self.plan.replica_count or any(
            not isinstance(state, AtomisticDynamicsState) for state in lanes
        ):
            raise ValueError("Initial states must contain exactly R dynamics states.")
        if any(
            state.prepared_dynamics_id != self.dynamics.prepared_id
            or state.thermodynamic_table_id != self.thermodynamic.table_id
            for state in lanes
        ):
            raise ValueError("Initial dynamics states belong to another runtime.")
        labels = np.asarray(state_at_replica, dtype=np.int32).reshape((-1,))
        if (
            labels.shape != (self.plan.replica_count,)
            or np.any(labels < 0)
            or np.any(labels >= self.plan.state_count)
        ):
            raise ValueError("state_at_replica must contain R valid state indices.")
        if self.plan.exchange is not None and sorted(labels.tolist()) != list(
            range(self.plan.state_count)
        ):
            raise ValueError("Dense replica exchange requires a state permutation.")
        # Frozen declarations bind content, not labels: every lane's bound
        # history must reproduce the declared digest (common to every row).
        # A history is one shared array state replicated on every lane.
        for lane in lanes:
            for term in self.bias_terms:
                if not term.plan.stateful:
                    continue
                bound = lane.term_states[term.name]
                if (
                    not isinstance(bound, AtomisticBiasTermState)
                    or bound.owner_identity.fingerprint() != term.term_id
                ):
                    raise ValueError(
                        f"Bias slot {term.name!r} binds another term's state."
                    )
            authenticate_frozen_bias_histories(
                self.thermodynamic.bias_rows[0], lane.term_states
            )
        for slot in self.bias_history_term_slots:
            contents = set()
            for lane in lanes:
                bound = lane.term_states[slot]
                if not isinstance(bound, AtomisticBiasTermState) or bound.history is None:
                    raise ValueError(f"History term {slot!r} binds no history state.")
                contents.add(bias_history_digest(bound.history))
            if len(contents) != 1:
                raise ValueError(
                    f"Every replica must bind the same content of history {slot!r}."
                )
        batched = jax.tree.map(lambda *values: jnp.stack(values), *lanes)
        root = key
        for identity in (
            canonical_fingerprint(
                {"kind": "atomistic-multistate-run", "run_id": self.plan.run_id}
            ),
            self.prepared_id,
        ):
            token = _identity_token(identity)
            for index in range(token.shape[0]):
                root = jr.fold_in(root, token[index])
        root = jr.fold_in(root, jnp.asarray(self.plan.repeat_index, dtype=jnp.uint32))
        root_data = jr.key_data(root).astype(jnp.uint32)
        physical_keys = jax.vmap(
            lambda replica_id: jr.key_data(
                self._semantic_key(root_data, 0, jnp.uint32(0), replica_id)
            )
        )(self.plan.replica_ids)
        batched = eqx.tree_at(
            lambda value: value.random_key,
            batched,
            physical_keys,
        )
        replica_lanes = jnp.arange(self.plan.replica_count, dtype=jnp.int32)
        batched = eqx.tree_at(
            lambda value: value.term_states,
            batched,
            jax.vmap(self._walker_bindings)(batched.term_states, replica_lanes),
        )
        primed = jax.vmap(
            lambda lane: self.dynamics.prime_state(lane, self.thermodynamic)
        )(batched)
        label_array = jnp.asarray(labels)
        rebased = jax.vmap(
            lambda lane, label: self.dynamics.rebase_thermodynamic_state(
                lane,
                self.thermodynamic,
                label,
                term_states=self._row_bindings(lane.term_states, label),
            )
        )(primed.accepted_state, label_array)
        checked_positions = eqx.error_if(
            rebased.accepted_state.kinematics.positions,
            ~(jnp.all(primed.successful) & jnp.all(rebased.successful)),
            "Initial multistate force and ledger rebasing failed.",
        )
        dynamics = eqx.tree_at(
            lambda value: value.kinematics.positions,
            rebased.accepted_state,
            checked_positions,
        )
        state_count = self.plan.state_count
        replica_count = self.plan.replica_count
        dtype = dynamics.kinematics.positions.dtype
        sams = AtomisticSAMSState(
            log_weights=jnp.zeros((state_count,), dtype=dtype),
            visit_counts=jnp.zeros((state_count,), dtype=jnp.int64),
            adaptation_index=jnp.zeros((), dtype=jnp.int64),
            frozen=jnp.asarray(
                self.plan.sams is None or self.plan.sams.adaptation_steps == 0
            ),
        )
        return AtomisticMultistateState(
            dynamics=dynamics,
            state_at_replica=label_array,
            reduced_potential_cache=jnp.zeros((state_count, replica_count), dtype=dtype),
            reduced_potential_valid=jnp.zeros(
                (state_count, replica_count), dtype=jnp.bool_
            ),
            cache_iteration=jnp.asarray(-1, dtype=jnp.int64),
            iteration_index=jnp.zeros((), dtype=jnp.int64),
            draw_index=jnp.zeros((replica_count,), dtype=jnp.int64),
            exchange_action_counter=jnp.zeros((), dtype=jnp.uint32),
            exchange_parity=jnp.zeros((), dtype=jnp.int32),
            sams_action_counter=jnp.zeros((replica_count,), dtype=jnp.uint32),
            sams=sams,
            root_key=root_data,
            segment_index=jnp.zeros((), dtype=jnp.int64),
            continuation_token=self.initial_continuation_identity.words,
            runtime_identity=self.prepared_identity,
        )

    def _walker_bindings(
        self, bindings: AtomisticTermStateBindings, lane: Array, /
    ) -> AtomisticTermStateBindings:
        """Lane-resolved (adaptive walker-owned) histories evaluate the lane's kernels."""
        values = dict(zip(bindings.slots, bindings.values, strict=True))
        for term, resolution in zip(
            self.bias_terms, self.bias_owner_resolutions, strict=True
        ):
            if resolution == "lane":
                current = term.require_state(values[term.name])
                values[term.name] = with_bias_controls(
                    current, current.controls, owner_key=lane
                )
        return AtomisticTermStateBindings(values)

    def _row_bindings(
        self, bindings: AtomisticTermStateBindings, row: Array, /
    ) -> AtomisticTermStateBindings:
        """Substitute one row's bias controls, declared owner and coupling."""
        if not self.bias_terms:
            return bindings
        values = dict(zip(bindings.slots, bindings.values, strict=True))
        index = jnp.asarray(row, dtype=jnp.int32)
        for position, term in enumerate(self.bias_terms):
            current = term.require_state(values[term.name])
            controls = self.bias_row_controls[position]
            row_owned = self.bias_owner_resolutions[position] == "row"
            values[term.name] = with_bias_controls(
                current,
                current.controls
                if controls is None
                else jnp.take(controls, index, axis=0).astype(current.controls.dtype),
                owner_key=jnp.take(self.bias_row_owners[position], index)
                if row_owned
                else None,
                coupling=jnp.take(self.bias_row_couplings[position], index),
            )
        return AtomisticTermStateBindings(values)

    def _deposit(
        self,
        dynamics_state: AtomisticDynamicsState,
        origin: Array,
        target: Array,
        iteration: Array,
        /,
    ) -> tuple[AtomisticDynamicsState, Array, Array, Array]:
        """Simultaneous all-or-nothing kernel deposition of every walker.

        Only ``"adaptive"``-declared histories deposit; ``deposited`` has one
        entry per history term in ``bias_history_term_slots`` order (frozen
        histories never change).  ``valid`` is false when a re-prime fails or
        a bound history belongs to another plan (nothing is deposited then).
        """
        deposited: list[Array] = []
        exhausted = jnp.asarray(False)
        changed = jnp.asarray(False)
        bound = jnp.asarray(True)
        bindings = dynamics_state.term_states
        lanes = jnp.arange(self.plan.replica_count, dtype=jnp.int32)
        labels = origin if self.plan.history_label is BiasHistoryLabel.ORIGIN else target
        kinds = dict(
            zip(self.bias_history_term_slots, self.bias_history_kinds, strict=True)
        )
        for position, term in enumerate(self.bias_terms):
            history_plan = term.plan.form.history
            if history_plan is None:
                continue
            if kinds[term.name] != "adaptive":
                deposited.append(jnp.asarray(False))
                continue
            if not isinstance(term.plan.form, HillHistoryBiasForm):
                raise TypeError("An adaptive history term must be a hill-history form.")

            def values_of(
                lane: AtomisticDynamicsState, term: PreparedAtomisticBiasTerm = term
            ) -> tuple[Array, Array]:
                context = self.dynamics.potential.context(
                    lane.kinematics.positions,
                    lane.neighborhoods,
                    term=term,
                    **self.dynamics._potential_kwargs(
                        self.dynamics._unwrapped(lane.kinematics, lane.cell_vectors),
                        lane.kinematics.image_counts,
                        lane.species,
                        lane.cell_vectors,
                        self.thermodynamic.state_at_replica(
                            lane.thermodynamic_state_index
                        ).controls,
                        lane.term_states,
                    ),
                )
                evaluation = term.plan.variables.evaluate(
                    CollectiveVariableGeometry.from_context(context)
                )
                return evaluation.values, evaluation.successful

            values, valid = jax.vmap(values_of)(dynamics_state)
            ownership = history_plan.ownership
            # A state-owned kernel carries the declared owner key of the row
            # whose label (origin or target) receives it, the same key that
            # row evaluates in ``_row_bindings``.
            owners = (
                jnp.zeros_like(lanes)
                if ownership is BiasHistoryOwnership.SHARED
                else lanes
                if ownership is BiasHistoryOwnership.WALKER
                else jnp.take(self.bias_row_owners[position], labels.astype(jnp.int32))
            )
            first = jax.tree.map(lambda leaf: leaf[0], bindings[term.name])
            if not isinstance(first, AtomisticBiasTermState) or first.history is None:
                raise TypeError("A hill term lost its history state.")
            due = history_plan.due(iteration)
            event = deposit_bias_history(
                history_plan,
                first.history,
                values,
                owners,
                valid & due,
                term.plan.variables.metrics,
                boltzmann_constant=float(
                    self.dynamics.system.plan.units.boltzmann_constant
                ),
            )
            bound = bound & event.successful
            exhausted = exhausted | (
                due & event.successful & ~event.inserted & jnp.any(valid)
            )
            deposited.append(event.inserted)
            changed = changed | event.inserted
            replaced = jax.vmap(
                lambda current, history=event.state: with_bias_history(current, history)
            )(bindings[term.name])
            slot = bindings.slots.index(term.name)
            # Stacked lanes: replace the leaf in place (no scalar re-parse).
            bindings = eqx.tree_at(
                lambda value, slot=slot: value.values[slot],
                bindings,
                tree_where(event.inserted, replaced, bindings[term.name]),
            )
        updated = eqx.tree_at(lambda value: value.term_states, dynamics_state, bindings)
        primed = jax.vmap(
            lambda lane: self.dynamics.prime_state(lane, self.thermodynamic)
        )(updated)
        refreshed = tree_where(changed, primed.accepted_state, dynamics_state)
        valid = bound & (~changed | jnp.all(primed.successful))
        return (
            refreshed,
            jnp.stack(deposited) if deposited else jnp.zeros((0,), dtype=jnp.bool_),
            exhausted,
            valid,
        )

    def reduced_potentials(
        self, dynamics_state: AtomisticDynamicsState, /
    ) -> AtomisticReducedPotentialEvaluation:
        indices = jnp.arange(self.plan.state_count, dtype=jnp.int32)
        unwrapped = jax.vmap(self.dynamics._unwrapped)(
            dynamics_state.kinematics, dynamics_state.cell_vectors
        )

        def evaluate_state(state_index: Array) -> tuple[Array, Array]:
            row = self.thermodynamic.state_at_replica(state_index)
            evaluation = jax.vmap(
                lambda lane, whole: self.dynamics.energy_configuration(
                    lane.kinematics.positions,
                    lane.neighborhoods,
                    unwrapped_positions=whole,
                    image_counts=lane.kinematics.image_counts,
                    species=lane.species,
                    cell_vectors=lane.cell_vectors,
                    controls=row.controls,
                    term_states=self._row_bindings(lane.term_states, state_index),
                )
            )(dynamics_state, unwrapped)
            return evaluation.energy, evaluation.successful & row.valid

        energies, successful = jax.vmap(evaluate_state)(indices)
        if self.dynamics.system.cell is None:
            volumes = jnp.zeros((self.plan.replica_count,), dtype=energies.dtype)
        else:
            vectors = dynamics_state.cell_vectors
            volumes = jnp.abs(
                jnp.sum(vectors[:, 0] * jnp.cross(vectors[:, 1], vectors[:, 2]), axis=-1)
            )
        reduced = self.thermodynamic.beta[:, None] * (
            energies + self.thermodynamic.pressure[:, None] * volumes[None, :]
        )
        coverage = (
            successful
            & self.thermodynamic.temperature_mask[:, None]
            & jnp.isfinite(reduced)
        )
        return AtomisticReducedPotentialEvaluation(
            jnp.where(coverage, reduced, 0.0),
            coverage,
            indices,
            self.thermodynamic.table_identity,
        )

    def _exchange(
        self,
        state: AtomisticMultistateState,
        reduced: AtomisticReducedPotentialEvaluation,
        scheduled: Array,
    ) -> _ExchangeOutcome:
        pair_count = max(self.plan.replica_count - 1, 0)
        starts = jnp.arange(pair_count, dtype=jnp.int32)
        pair_indices = jnp.stack((starts, starts + 1), axis=-1)
        parity = state.exchange_parity & 1
        attempted = scheduled & (starts % 2 == parity)
        labels = state.state_at_replica
        left_label = labels[:-1]
        right_label = labels[1:]
        left_slot = starts
        right_slot = starts + 1
        current = (
            reduced.values[left_label, left_slot]
            + reduced.values[right_label, right_slot]
        )
        proposed = (
            reduced.values[right_label, left_slot]
            + reduced.values[left_label, right_slot]
        )
        log_acceptance = jnp.where(attempted, current - proposed, 0.0)
        exchange = self.plan.exchange
        realization = (
            jnp.zeros((), dtype=jnp.uint32)
            if exchange is None
            else exchange.realization_id
        )

        def propose(_: None) -> Array:
            keys = jax.vmap(
                lambda start: self._semantic_key(
                    state.root_key,
                    jnp.uint32(_EXCHANGE_STREAM) ^ realization,
                    state.exchange_action_counter,
                    start,
                )
            )(starts)
            uniforms = jax.vmap(lambda value: jr.uniform(value, ()))(keys)
            return attempted & (jnp.log(uniforms) < jnp.minimum(log_acceptance, 0.0))

        accepted = jax.lax.cond(
            scheduled,
            propose,
            lambda _: jnp.zeros((pair_count,), dtype=jnp.bool_),
            operand=None,
        )
        next_labels = labels
        for index in range(pair_count):
            accept = accepted[index]
            left_value = next_labels[index]
            right_value = next_labels[index + 1]
            next_labels = next_labels.at[index].set(
                jnp.where(accept, right_value, left_value)
            )
            next_labels = next_labels.at[index + 1].set(
                jnp.where(accept, left_value, right_value)
            )
        counter_valid = (~scheduled) | (
            state.exchange_action_counter < jnp.uint32(_UINT32_MAX)
        )
        valid = counter_valid & jnp.all((~attempted) | jnp.isfinite(log_acceptance))
        next_counter = state.exchange_action_counter + scheduled.astype(jnp.uint32)
        next_parity = jnp.where(scheduled, 1 - parity, parity)
        return (
            next_labels,
            pair_indices,
            attempted,
            accepted,
            log_acceptance,
            next_counter,
            next_parity,
            valid,
        )

    def _sams(
        self,
        state: AtomisticMultistateState,
        reduced: AtomisticReducedPotentialEvaluation,
        scheduled: Array,
    ) -> tuple[Array, Array, Array, Array, AtomisticSAMSState, Array]:
        sams_plan = self.plan.sams
        replica_count = self.plan.replica_count
        if sams_plan is None:
            return (
                state.state_at_replica,
                jnp.zeros((replica_count,), dtype=jnp.bool_),
                jnp.zeros((replica_count,), dtype=jnp.bool_),
                state.sams_action_counter,
                state.sams,
                jnp.asarray(True),
            )
        attempted = jnp.full((replica_count,), scheduled, dtype=jnp.bool_)
        logits = (
            jnp.log(sams_plan.target_probabilities)[:, None]
            + state.sams.log_weights[:, None]
            - reduced.values
        )

        def propose_labels(_: None) -> Array:
            keys = jax.vmap(
                lambda replica_id, counter: self._semantic_key(
                    state.root_key,
                    jnp.uint32(_SAMS_STREAM) ^ sams_plan.realization_id,
                    counter,
                    replica_id,
                )
            )(self.plan.replica_ids, state.sams_action_counter)
            return jax.vmap(
                lambda action_key, lane_logits: jr.categorical(action_key, lane_logits)
            )(keys, logits.T).astype(jnp.int32)

        labels = jax.lax.cond(
            scheduled,
            propose_labels,
            lambda _: state.state_at_replica,
            operand=None,
        )
        changed = attempted & (labels != state.state_at_replica)
        counter_valid = (~scheduled) | jnp.all(
            state.sams_action_counter < jnp.uint32(_UINT32_MAX)
        )
        finite = (~scheduled) | jnp.all(jnp.isfinite(logits))
        next_counters = state.sams_action_counter + attempted.astype(jnp.uint32)
        visits = jnp.bincount(labels, length=self.plan.state_count)
        adapting = scheduled & (state.sams.adaptation_index < sams_plan.adaptation_steps)
        dtype = reduced.values.dtype
        gain = sams_plan.initial_gain.astype(dtype) * (
            state.sams.adaptation_index.astype(dtype) + 1.0
        ) ** (-sams_plan.gain_exponent.astype(dtype))
        observed = visits.astype(reduced.values.dtype) / replica_count
        increment = gain * (sams_plan.target_probabilities - observed)
        weights = jnp.where(
            adapting, state.sams.log_weights + increment, state.sams.log_weights
        )
        weights = weights - jnp.sum(weights * sams_plan.target_probabilities)
        adaptation_index = state.sams.adaptation_index + adapting.astype(jnp.int64)
        sams = AtomisticSAMSState(
            weights,
            state.sams.visit_counts + jnp.where(scheduled, visits, 0),
            adaptation_index,
            adaptation_index >= sams_plan.adaptation_steps,
        )
        return labels, attempted, changed, next_counters, sams, counter_valid & finite

    def _bias_evidence(
        self, dynamics_state: AtomisticDynamicsState, /
    ) -> tuple[Array, Array]:
        """``(owner keys (R, B), history written counters (H,))`` of a batched state."""
        replicas = self.plan.replica_count
        bindings = dynamics_state.term_states
        states = {slot: _bias_term_state(bindings, slot) for slot in self.bias_term_slots}
        owners = tuple(
            jnp.asarray(states[slot].owner_key, dtype=jnp.int32)
            for slot in self.bias_term_slots
        )
        revisions = []
        for slot in self.bias_history_term_slots:
            history = states[slot].history
            if history is None:
                raise TypeError("A history term lost its history state.")
            # Every lane carries the same (replicated) history content.
            revisions.append(jnp.asarray(history.written, dtype=jnp.int32)[0])
        return (
            jnp.stack(owners, axis=-1)
            if owners
            else jnp.zeros((replicas, 0), dtype=jnp.int32),
            jnp.stack(revisions) if revisions else jnp.zeros((0,), dtype=jnp.int32),
        )

    @checked
    def iterate(self, state: AtomisticMultistateState, /) -> AtomisticMultistateIteration:
        runtime_bound = state.runtime_identity.equals(self.prepared_identity)
        propagated = jax.vmap(
            lambda lane: self.dynamics.step_detailed(lane, self.thermodynamic)
        )(state.dynamics)
        dynamics_state = propagated.accepted_state
        propagation_valid = jnp.all(propagated.successful)
        next_iteration = state.iteration_index + 1
        barostat_plan = self.plan.barostat
        barostat_interval = self.plan.barostat_interval
        if barostat_plan is None or barostat_interval is None:
            barostat_scheduled = jnp.asarray(False)
            barostat_attempted = jnp.zeros((self.plan.replica_count,), dtype=jnp.bool_)
            barostat_accepted = jnp.zeros_like(barostat_attempted)
            barostat_valid = jnp.asarray(True)
        else:
            barostat_scheduled = next_iteration % barostat_interval == 0
            barostat_attempted = jnp.full(
                (self.plan.replica_count,), barostat_scheduled, dtype=jnp.bool_
            )

            def apply_moves(_: None) -> tuple[AtomisticDynamicsState, Array, Array]:
                evaluations = jax.vmap(
                    lambda lane: apply_isotropic_monte_carlo_barostat(
                        self.dynamics,
                        lane,
                        self.thermodynamic,
                        barostat_plan,
                    )
                )(dynamics_state)
                return (
                    evaluations.accepted_state,
                    evaluations.accepted,
                    jnp.all(evaluations.successful),
                )

            def skip_moves(_: None) -> tuple[AtomisticDynamicsState, Array, Array]:
                return (
                    dynamics_state,
                    jnp.zeros((self.plan.replica_count,), dtype=jnp.bool_),
                    jnp.asarray(True),
                )

            dynamics_state, moved, moves_valid = jax.lax.cond(
                barostat_scheduled, apply_moves, skip_moves, operand=None
            )
            barostat_accepted = barostat_attempted & moved
            barostat_valid = (~barostat_scheduled) | moves_valid
        reduced = self.reduced_potentials(dynamics_state)
        cross_valid = jnp.all(reduced.successful)
        origin = state.state_at_replica
        exchange_plan = self.plan.exchange
        exchange_scheduled = (
            jnp.asarray(False)
            if exchange_plan is None
            else exchange_plan.scheduled(next_iteration)
        )
        (
            exchange_labels,
            pair_indices,
            exchange_attempted,
            exchange_accepted,
            log_acceptance,
            exchange_counter,
            exchange_parity,
            exchange_valid,
        ) = self._exchange(state, reduced, exchange_scheduled)
        sams_plan = self.plan.sams
        sams_adapting = jnp.asarray(False) if sams_plan is None else ~state.sams.frozen
        equilibrium_sample = ~sams_adapting & ~jnp.asarray(self.history_adaptive)
        sams_scheduled = (
            jnp.asarray(False)
            if sams_plan is None
            else sams_plan.scheduled(next_iteration)
        )
        (
            next_labels,
            sams_attempted,
            sams_changed,
            sams_counter,
            sams_state,
            sams_valid,
        ) = self._sams(
            eqx.tree_at(lambda value: value.state_at_replica, state, exchange_labels),
            reduced,
            sams_scheduled,
        )
        rebased = jax.vmap(
            lambda lane, label: self.dynamics.rebase_thermodynamic_state(
                lane,
                self.thermodynamic,
                label,
                term_states=self._row_bindings(lane.term_states, label),
            )
        )(dynamics_state, next_labels)
        rebase_valid = jnp.all(rebased.successful)
        if self.history_adaptive:
            deposited_state, bias_deposited, bias_exhausted, deposit_valid = (
                self._deposit(rebased.accepted_state, origin, next_labels, next_iteration)
            )
        else:
            deposited_state = rebased.accepted_state
            bias_deposited = jnp.zeros(
                (len(self.bias_history_term_slots),), dtype=jnp.bool_
            )
            bias_exhausted = jnp.asarray(False)
            deposit_valid = jnp.asarray(True)
        owner_keys, history_revision = self._bias_evidence(deposited_state)
        successful = (
            runtime_bound
            & propagation_valid
            & barostat_valid
            & cross_valid
            & exchange_valid
            & sams_valid
            & rebase_valid
            & deposit_valid
            & ~bias_exhausted
        )
        candidate = AtomisticMultistateState(
            dynamics=deposited_state,
            state_at_replica=next_labels,
            reduced_potential_cache=reduced.values,
            reduced_potential_valid=reduced.successful,
            cache_iteration=next_iteration,
            iteration_index=next_iteration,
            draw_index=state.draw_index + 1,
            exchange_action_counter=exchange_counter,
            exchange_parity=exchange_parity,
            sams_action_counter=sams_counter,
            sams=sams_state,
            root_key=state.root_key,
            segment_index=state.segment_index,
            continuation_token=state.continuation_token,
            runtime_identity=self.prepared_identity,
        )
        committed = tree_where(successful, candidate, state)
        return AtomisticMultistateIteration(
            candidate_state=candidate,
            state=committed,
            reduced_potentials=jnp.where(successful, reduced.values, 0.0),
            coverage=successful & reduced.successful,
            origin_state=jnp.where(successful, origin, -1),
            state_at_replica_after=jnp.where(successful, next_labels, -1),
            pair_indices=pair_indices,
            exchange_attempted=successful & exchange_attempted,
            exchange_accepted=successful & exchange_accepted,
            exchange_log_acceptance=jnp.where(
                successful & exchange_attempted, log_acceptance, 0.0
            ),
            sams_attempted=successful & sams_attempted,
            sams_changed=successful & sams_changed,
            sams_adapting=successful & sams_adapting,
            dynamics_accepted=successful & propagated.successful,
            barostat_attempted=successful & barostat_attempted,
            barostat_accepted=successful & barostat_accepted,
            equilibrium_sample=successful & equilibrium_sample,
            bias_adapting=successful & jnp.asarray(self.history_adaptive),
            bias_deposited=successful & bias_deposited,
            bias_exhausted=bias_exhausted,
            bias_owner_keys=jnp.where(successful, owner_keys, -1),
            bias_history_revision=jnp.where(successful, history_revision, -1),
            successful=successful,
            runtime_identity=self.prepared_identity,
        )


class AtomisticDrawConfigurationCapture(StrictModule, NonTrainableState):
    """Explicit bounded capture of every committed draw's configuration.

    A segment run with this policy records, inside its compiled scan, the
    exact configuration at which each committed draw's reduced potentials
    were evaluated (the propagated configuration after any cell move; the
    assignment rebase, deposition and force refresh do not move it):
    wrapped coordinates, integer image counts, species and, for a system with
    a cell, the cell vectors.  ``required_bytes`` is the captured footprint
    of one segment; a segment plan whose footprint exceeds ``max_bytes`` is
    refused.  Without a policy segments record no configurations.

    PyTree: ``max_bytes`` is static; ``plan_identity`` (``plan_id``) is the
    exact policy digest bound into every capturing segment's identity.
    """

    max_bytes: int = eqx.field(static=True)
    plan_identity: IdentityWords

    def __init__(self, *, max_bytes: int) -> None:
        limit = int(max_bytes)
        if limit <= 0:
            raise ValueError("max_bytes must be positive.")
        self.max_bytes = limit
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-multistate-draw-configuration-capture",
                "max_bytes": limit,
                "fields": ["positions", "image_counts", "species", "cell_vectors"],
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical capture-policy digest."""
        return self.plan_identity.fingerprint()

    def required_bytes(
        self, runtime: PreparedAtomisticMultistate, capacity: int, /
    ) -> int:
        """Captured bytes of one segment of ``capacity`` iterations (host)."""
        system = runtime.dynamics.system
        atoms = int(system.mobile_mask.shape[0])
        coordinate = np.dtype(system.plan.coordinate_dtype).itemsize
        integer = np.dtype(np.int32).itemsize
        per_draw = atoms * 3 * (coordinate + integer) + atoms * integer
        if system.cell is not None:
            per_draw += int(np.prod(system.cell.vectors.shape)) * coordinate
        return int(capacity) * runtime.plan.replica_count * per_draw


def _segment_identity_payload(
    runtime: PreparedAtomisticMultistate,
    capacity: int,
    start: int,
    index: int,
    predecessor: str,
    configurations: AtomisticDrawConfigurationCapture | None,
    /,
) -> dict[str, object]:
    """The canonical segment-identity payload (segment plan and checkpoint)."""
    return {
        "kind": "atomistic-multistate-segment",
        "runtime": runtime.prepared_id,
        "capacity": capacity,
        "start_iteration": start,
        "segment_index": index,
        "predecessor": predecessor,
        "configurations": None if configurations is None else configurations.plan_id,
    }


class AtomisticMultistateSegmentPlan(StrictModule, NonTrainableState):
    """One fixed-capacity segment of a multistate run.

    ``configurations`` optionally captures every committed draw's
    configuration (:class:`AtomisticDrawConfigurationCapture`); its policy
    enters the segment identity.

    PyTree: only ``capacity`` (the scan length) is static.  The watermarks
    ``expected_start_iteration``/``segment_index`` are dynamic ``int64``
    scalars and the predecessor and segment digests are dynamic words
    (``predecessor_id``/``segment_id`` decode them on the host), so every
    segment of one runtime layout shares one executable.
    """

    runtime: PreparedAtomisticMultistate
    configurations: AtomisticDrawConfigurationCapture | None
    capacity: int = eqx.field(static=True)
    expected_start_iteration: Array
    segment_index: Array
    predecessor_identity: IdentityWords
    segment_identity: IdentityWords

    @checked
    def __init__(
        self,
        runtime: PreparedAtomisticMultistate,
        capacity: int,
        expected_start_iteration: int,
        segment_index: int,
        predecessor_id: str,
        /,
        *,
        configurations: AtomisticDrawConfigurationCapture | None = None,
    ) -> None:
        capacity_ = int(capacity)
        start = int(expected_start_iteration)
        index = int(segment_index)
        predecessor = str(predecessor_id)
        if capacity_ <= 0 or start < 0 or index < 0 or not predecessor:
            raise ValueError("Segment capacity, watermarks, or predecessor are invalid.")
        if (
            configurations is not None
            and configurations.required_bytes(runtime, capacity_)
            > configurations.max_bytes
        ):
            raise ValueError(
                "Draw-configuration capture of this segment needs "
                f"{configurations.required_bytes(runtime, capacity_)} bytes, above "
                f"its declared max_bytes={configurations.max_bytes}."
            )
        predecessor_identity = IdentityWords.from_fingerprint(predecessor)
        self.runtime = runtime
        self.configurations = configurations
        self.capacity = capacity_
        self.expected_start_iteration = jnp.asarray(start, dtype=jnp.int64)
        self.segment_index = jnp.asarray(index, dtype=jnp.int64)
        self.predecessor_identity = predecessor_identity
        self.segment_identity = IdentityWords.of(
            _segment_identity_payload(
                runtime, capacity_, start, index, predecessor, configurations
            )
        )

    @property
    def predecessor_id(self) -> str:
        """Host decoding of the predecessor continuation digest."""
        return self.predecessor_identity.fingerprint()

    @property
    def segment_id(self) -> str:
        """Host decoding of the canonical segment digest."""
        return self.segment_identity.fingerprint()

    @checked
    def run(
        self, state: AtomisticMultistateState, /
    ) -> "AtomisticMultistateSegmentResult":
        valid_predecessor = (
            state.runtime_identity.equals(self.runtime.prepared_identity)
            & jnp.all(state.continuation_token == self.predecessor_identity.words)
            & (state.iteration_index == self.expected_start_iteration)
            & (state.segment_index == self.segment_index)
        )
        checked_iteration = eqx.error_if(
            state.iteration_index,
            ~valid_predecessor,
            "Segment predecessor identity or watermark does not match state.",
        )
        state = eqx.tree_at(
            lambda value: value.iteration_index,
            state,
            checked_iteration,
        )

        def advance(
            carry: tuple[AtomisticMultistateState, Array], _: None
        ) -> tuple[
            tuple[AtomisticMultistateState, Array],
            tuple[AtomisticMultistateIteration, Array, Array, Array],
        ]:
            current, active = carry
            iteration = self.runtime.iterate(current)
            committed = active & iteration.successful
            successor = tree_where(active, iteration.state, current)
            return (successor, committed), (
                iteration,
                committed,
                active,
                current.draw_index,
            )

        (successor, successful), (iterations, valid, attempted, draw_index) = (
            jax.lax.scan(
                advance,
                (state, jnp.asarray(True)),
                xs=None,
                length=self.capacity,
            )
        )
        replicas = self.runtime.plan.replica_count
        retained = valid & iterations.equilibrium_sample
        sample_active = retained[:, None] & jnp.ones(
            (self.capacity, self.runtime.plan.replica_count), dtype=jnp.bool_
        )
        reduced = jnp.where(retained[:, None, None], iterations.reduced_potentials, 0.0)
        coverage = retained[:, None, None] & iterations.coverage
        origin = jnp.where(sample_active, iterations.origin_state, -1)
        assignments = jnp.where(sample_active, iterations.state_at_replica_after, -1)
        chain = jnp.where(
            sample_active,
            self.runtime.plan.chain_indices[None, :],
            -1,
        )
        repeat = jnp.where(
            sample_active,
            self.runtime.plan.repeat_index.astype(jnp.int32),
            jnp.int32(-1),
        )
        dependence = jnp.where(
            sample_active,
            self.runtime.plan.dependence_group_indices[None, :],
            -1,
        )
        draw = jnp.where(sample_active, draw_index, -1)
        count = jnp.sum(valid, dtype=jnp.int64)
        stop = state.iteration_index + count
        # Committed iterations never move the evaluated configuration after
        # the reduced-potential evaluation (rebase, deposit and force refresh
        # keep coordinates, images, species and cell), so the committed
        # state's configuration is the evaluated one.
        configurations = (
            _absent_draw_configurations()
            if self.configurations is None
            else _draw_configurations(
                self.runtime,
                valid,
                iterations.state.dynamics.kinematics.positions,
                iterations.state.dynamics.kinematics.image_counts,
                iterations.state.dynamics.species,
                iterations.state.dynamics.cell_vectors,
            )
        )
        successor = AtomisticMultistateState(
            dynamics=successor.dynamics,
            state_at_replica=successor.state_at_replica,
            reduced_potential_cache=successor.reduced_potential_cache,
            reduced_potential_valid=successor.reduced_potential_valid,
            cache_iteration=successor.cache_iteration,
            iteration_index=successor.iteration_index,
            draw_index=successor.draw_index,
            exchange_action_counter=successor.exchange_action_counter,
            exchange_parity=successor.exchange_parity,
            sams_action_counter=successor.sams_action_counter,
            sams=successor.sams,
            root_key=successor.root_key,
            segment_index=successor.segment_index + 1,
            continuation_token=self.segment_identity.words,
            runtime_identity=successor.runtime_identity,
        )
        return AtomisticMultistateSegmentResult(
            successor_state=successor,
            reduced_potentials=reduced,
            coverage=coverage,
            sample_active=sample_active,
            origin_state=origin,
            state_at_replica=assignments,
            chain_index=chain,
            draw_index=draw,
            repeat_index=repeat,
            dependence_group_index=dependence,
            pair_indices=jnp.where(valid[:, None, None], iterations.pair_indices, -1),
            exchange_attempted=valid[:, None] & iterations.exchange_attempted,
            exchange_accepted=valid[:, None] & iterations.exchange_accepted,
            exchange_log_acceptance=jnp.where(
                valid[:, None], iterations.exchange_log_acceptance, 0.0
            ),
            sams_attempted=valid[:, None] & iterations.sams_attempted,
            sams_changed=valid[:, None] & iterations.sams_changed,
            sams_adapting=valid[:, None]
            & iterations.sams_adapting[:, None]
            & jnp.ones((1, replicas), dtype=jnp.bool_),
            dynamics_accepted=valid[:, None] & iterations.dynamics_accepted,
            barostat_attempted=valid[:, None] & iterations.barostat_attempted,
            barostat_accepted=valid[:, None] & iterations.barostat_accepted,
            bias_adapting=valid[:, None]
            & iterations.bias_adapting[:, None]
            & jnp.ones((1, replicas), dtype=jnp.bool_),
            bias_deposited=valid[:, None] & iterations.bias_deposited,
            bias_exhausted=attempted & iterations.bias_exhausted,
            bias_owner_keys=jnp.where(
                valid[:, None, None], iterations.bias_owner_keys, -1
            ),
            bias_history_revision=jnp.where(
                valid[:, None], iterations.bias_history_revision, -1
            ),
            iteration_valid=valid,
            count=count,
            start_watermark=state.iteration_index,
            stop_watermark=stop,
            successful=successful,
            inverse_temperatures=self.runtime.thermodynamic.beta,
            units=self.runtime.dynamics.system.plan.units,
            **configurations,
            **_segment_provenance(
                self.runtime,
                segment_identity=self.segment_identity,
                predecessor_identity=self.predecessor_identity,
            ),
        )


class _DrawConfigurations(TypedDict):
    """Captured per-draw configuration fields of a segment result."""

    draw_positions: Array | None
    draw_image_counts: Array | None
    draw_species: Array | None
    draw_cell_vectors: Array | None


def _absent_draw_configurations() -> _DrawConfigurations:
    return {
        "draw_positions": None,
        "draw_image_counts": None,
        "draw_species": None,
        "draw_cell_vectors": None,
    }


def _draw_configurations(
    runtime: PreparedAtomisticMultistate,
    valid: Array,
    positions: Array,
    image_counts: Array,
    species: Array,
    cell_vectors: Array,
    /,
) -> _DrawConfigurations:
    """Canonically padded ``(capacity, R, ...)`` configurations of committed draws.

    Uncommitted iterations carry zero coordinates, images and cells and
    species ``-1``.
    """
    lanes = valid[:, None, None, None]
    return {
        "draw_positions": jnp.where(lanes, positions, 0.0).astype(positions.dtype),
        "draw_image_counts": jnp.where(lanes, image_counts, 0).astype(jnp.int32),
        "draw_species": jnp.where(valid[:, None, None], species, -1).astype(jnp.int32),
        "draw_cell_vectors": None
        if runtime.dynamics.system.cell is None
        else jnp.where(lanes, cell_vectors, 0.0).astype(cell_vectors.dtype),
    }


class _SegmentProvenance(TypedDict):
    """Runtime/table provenance fields of a segment result (words + layout)."""

    run_id: str
    measure_identity: IdentityWords
    state_labels: tuple[str | None, ...]
    state_identities: IdentityWords
    potential_identities: IdentityWords
    bias_identities: IdentityWords | None
    unit_id: str
    reduced_convention_identity: IdentityWords
    qualification_identity: IdentityWords
    sampling_exact: bool
    sampling_bias_bound: Array
    segment_identity: IdentityWords
    predecessor_identity: IdentityWords
    runtime_identity: IdentityWords
    bias_term_slots: tuple[str, ...]
    bias_term_identities: IdentityWords
    bias_owner_resolutions: tuple[str, ...]
    bias_row_owner_keys: Array
    bias_history_term_slots: tuple[str, ...]
    bias_history_kinds: tuple[str, ...]
    bias_history_identities: IdentityWords


def _segment_provenance(
    runtime: PreparedAtomisticMultistate,
    /,
    *,
    segment_identity: IdentityWords,
    predecessor_identity: IdentityWords,
) -> _SegmentProvenance:
    """Runtime/table provenance fields of a segment result (words + layout)."""
    table = runtime.thermodynamic
    qualification = runtime.plan.qualification
    return {
        "run_id": runtime.plan.run_id,
        "measure_identity": table.phase_space_measure_identity,
        "state_labels": table.state_labels,
        "state_identities": table.state_identities,
        "potential_identities": table.potential_identities,
        "bias_identities": table.bias_identities,
        "unit_id": "1",
        "reduced_convention_identity": table.reduced_convention_identity,
        "qualification_identity": qualification.qualification_identity,
        "sampling_exact": qualification.sampling_exact,
        "sampling_bias_bound": qualification.sampling_bias_bound,
        "segment_identity": segment_identity,
        "predecessor_identity": predecessor_identity,
        "runtime_identity": runtime.prepared_identity,
        "bias_term_slots": runtime.bias_term_slots,
        "bias_term_identities": _stack_identities(
            tuple(term.term_identity for term in runtime.bias_terms)
        ),
        "bias_owner_resolutions": runtime.bias_owner_resolutions,
        "bias_row_owner_keys": runtime.bias_row_owner_keys,
        "bias_history_term_slots": runtime.bias_history_term_slots,
        "bias_history_kinds": runtime.bias_history_kinds,
        "bias_history_identities": runtime.bias_history_identities,
    }


class AtomisticMultistateSegmentResult(StrictModule):
    """Committed multistate segment with complete sample and bias-event evidence.

    Per-draw bias evidence (padding ``False``/``-1`` beyond ``count``):
    ``bias_adapting`` ``(capacity, R)`` marks draws sampled under an
    advancing history (never equilibrium samples); ``bias_deposited``
    ``(capacity, H)`` the histories that received the iteration's event;
    ``bias_exhausted`` ``(capacity,)`` the attempted iteration refused by
    terminal history capacity (it ends the segment); ``bias_owner_keys``
    ``(capacity, R, B)`` the history owner each replica's bias terms
    evaluated; ``bias_history_revision`` ``(capacity, H)`` each history's
    ``written`` counter after the draw.  ``bias_history_declarations`` are the
    rows' ``"adaptive"``/``"frozen:<digest>"`` declarations; the content they
    authenticate is checked by :meth:`bias_history_digests`.
    ``bias_owner_resolutions`` (``B``) and the rows' declared owner keys
    ``bias_row_owner_keys`` ``(K, B)`` define the owner every analysis row
    evaluates; :meth:`authenticate_bias_owners` checks the per-draw owner
    evidence against them.

    Draw configurations (``None`` unless the segment plan captured them with
    an :class:`AtomisticDrawConfigurationCapture`): ``draw_positions``
    ``(capacity, R, N, 3)`` wrapped coordinates in the dynamics coordinate
    dtype and ``draw_image_counts`` ``(capacity, R, N, 3)`` ``int32``, so the
    unwrapped coordinates are ``positions + image_counts @ cell_vectors``;
    ``draw_species`` ``(capacity, R, N)`` ``int32``; ``draw_cell_vectors``
    ``(capacity, R, *cell)`` for a system with a cell, else ``None``.  Each is
    the configuration at which that draw's ``reduced_potentials[d, :, r]``
    were evaluated; uncommitted iterations are zero (species ``-1``).

    PyTree: provenance digests (measure, states, potentials, biases, reduced
    convention, qualification, segment, predecessor, runtime, bias terms and
    frozen histories) and the row owner keys are dynamic; only declared
    labels (``run_id``, ``state_labels``, term slots), declaration kinds,
    owner resolutions, ``unit_id`` and ``sampling_exact`` are static.  Host
    properties keep every former string
    attribute (``measure_id``, ``state_ids``, ``potential_ids``, ``bias_ids``,
    ``producer_id``/``runtime_id``, ``reduced_convention_id``,
    ``qualification_id``, ``segment_id``, ``predecessor_id``,
    ``bias_term_ids``, ``bias_history_term_ids``,
    ``bias_history_declarations``) with exact values.
    """

    successor_state: AtomisticMultistateState
    reduced_potentials: Array
    coverage: Array
    sample_active: Array
    origin_state: Array
    state_at_replica: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    pair_indices: Array
    exchange_attempted: Array
    exchange_accepted: Array
    exchange_log_acceptance: Array
    sams_attempted: Array
    sams_changed: Array
    sams_adapting: Array
    dynamics_accepted: Array
    barostat_attempted: Array
    barostat_accepted: Array
    bias_adapting: Array
    bias_deposited: Array
    bias_exhausted: Array
    bias_owner_keys: Array
    bias_history_revision: Array
    iteration_valid: Array
    count: Array
    start_watermark: Array
    stop_watermark: Array
    successful: Array
    inverse_temperatures: Array
    units: AtomisticUnitSystem
    draw_positions: Array | None
    draw_image_counts: Array | None
    draw_species: Array | None
    draw_cell_vectors: Array | None
    measure_identity: IdentityWords
    state_identities: IdentityWords
    potential_identities: IdentityWords
    bias_identities: IdentityWords | None
    reduced_convention_identity: IdentityWords
    qualification_identity: IdentityWords
    sampling_bias_bound: Array
    segment_identity: IdentityWords
    predecessor_identity: IdentityWords
    runtime_identity: IdentityWords
    bias_term_identities: IdentityWords
    bias_history_identities: IdentityWords
    bias_row_owner_keys: Array
    run_id: str = eqx.field(static=True)
    state_labels: tuple[str | None, ...] = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    bias_term_slots: tuple[str, ...] = eqx.field(static=True)
    bias_owner_resolutions: tuple[str, ...] = eqx.field(static=True)
    bias_history_term_slots: tuple[str, ...] = eqx.field(static=True)
    bias_history_kinds: tuple[str, ...] = eqx.field(static=True)

    @property
    def measure_id(self) -> str:
        return self.measure_identity.fingerprint()

    @property
    def state_ids(self) -> tuple[str, ...]:
        """Declared state labels, else the exact state digests."""
        digests = _decode_identities(self.state_identities)
        return tuple(
            digest if label is None else label
            for label, digest in zip(self.state_labels, digests, strict=True)
        )

    @property
    def potential_ids(self) -> tuple[str, ...]:
        return _decode_identities(self.potential_identities)

    @property
    def bias_ids(self) -> tuple[str | None, ...]:
        if self.bias_identities is None:
            return (None,) * len(self.state_labels)
        return _decode_identities(self.bias_identities)

    @property
    def reduced_convention_id(self) -> str:
        return self.reduced_convention_identity.fingerprint()

    @property
    def qualification_id(self) -> str:
        return self.qualification_identity.fingerprint()

    @property
    def segment_id(self) -> str:
        return self.segment_identity.fingerprint()

    @property
    def predecessor_id(self) -> str:
        return self.predecessor_identity.fingerprint()

    @property
    def runtime_id(self) -> str:
        return self.runtime_identity.fingerprint()

    @property
    def producer_id(self) -> str:
        """The producing runtime (identical to ``runtime_id``)."""
        return self.runtime_identity.fingerprint()

    @property
    def bias_term_ids(self) -> tuple[str, ...]:
        return _decode_identities(self.bias_term_identities)

    @property
    def bias_history_term_ids(self) -> tuple[str, ...]:
        by_slot = dict(zip(self.bias_term_slots, self.bias_term_ids, strict=True))
        return tuple(by_slot[slot] for slot in self.bias_history_term_slots)

    @property
    def bias_history_declarations(self) -> tuple[str, ...]:
        return _history_declarations(
            self.bias_history_kinds, self.bias_history_identities
        )

    def bias_history_digests(self) -> tuple[str, ...]:
        """Authenticated content digest of every history at the segment end.

        Host transaction.  Every replica lane must carry the same history
        content; a frozen declaration must equal ``"frozen:<digest>"`` of that
        content, and its history may not have received any event or changed
        revision in a committed draw.  An adaptive history's last committed
        revision must equal the bound content's ``written`` counter.
        """
        if not self.bias_history_term_slots:
            return ()
        valid = np.asarray(self.iteration_valid, dtype=np.bool_)
        revisions = np.asarray(self.bias_history_revision)
        deposited = np.asarray(self.bias_deposited, dtype=np.bool_)
        count = int(np.asarray(self.count))
        bindings = self.successor_state.dynamics.term_states
        replicas = int(np.asarray(self.sample_active).shape[1])
        digests: list[str] = []
        for position, (term_id, declaration) in enumerate(
            zip(self.bias_history_term_slots, self.bias_history_declarations, strict=True)
        ):
            state = bindings[term_id]
            if not isinstance(state, AtomisticBiasTermState) or state.history is None:
                raise ValueError(f"History term {term_id!r} binds no history state.")
            history = state.history
            lane_digests = {
                bias_history_digest(
                    jax.tree.map(lambda leaf, lane=lane: leaf[lane], history)
                )
                for lane in range(replicas)
            }
            if len(lane_digests) != 1:
                raise ValueError(
                    f"Replica lanes disagree on the content of history {term_id!r}."
                )
            digest = next(iter(lane_digests))
            written = int(np.asarray(history.written)[0])
            committed = revisions[valid, position]
            if declaration == "adaptive":
                if count and int(committed[-1]) != written:
                    raise ValueError(
                        f"History {term_id!r} content does not match its last revision."
                    )
            elif (
                declaration != f"frozen:{digest}"
                or np.any(committed != written)
                or np.any(deposited[:, position])
            ):
                raise ValueError(
                    f"History {term_id!r}: bound content does not authenticate its "
                    "frozen declaration."
                )
            digests.append(digest)
        return tuple(digests)

    def authenticate_bias_owners(self) -> None:
        """Host check that every active draw evaluated its row's declared owner.

        An active draw is a column of every analysis row, so each row must be
        one fixed Hamiltonian at every column.  A lane-resolved (adaptive
        walker-owned) history makes a row's bias depend on the sample column
        and is refused; owner-resolved analysis rows are frozen histories
        whose rows declare owner keys.  A row-resolved history's evaluated
        owner (after the assignment move) must equal the declared owner key
        of the replica's row.
        """
        resolutions = self.bias_owner_resolutions
        keys = np.asarray(self.bias_owner_keys)
        declared = np.asarray(self.bias_row_owner_keys)
        sample = np.asarray(self.sample_active, dtype=np.bool_)
        labels = np.asarray(self.state_at_replica)
        if (
            len(resolutions) != len(self.bias_term_slots)
            or declared.shape != (len(self.state_labels), len(resolutions))
            or keys.shape != (*sample.shape, len(resolutions))
        ):
            raise ValueError("Segment owner evidence does not match its bias layout.")
        if not np.any(sample):
            return
        for slot, resolution in zip(self.bias_term_slots, resolutions, strict=True):
            if resolution == "lane":
                raise ValueError(
                    f"History {slot!r} is walker-owned and adaptive: each sample "
                    "column evaluates its own walker's kernels, so no row is one "
                    "fixed Hamiltonian. Freeze it and declare owner keys per row."
                )
        rows = labels[sample]
        if np.any((rows < 0) | (rows >= declared.shape[0])):
            raise ValueError("Every active draw must name one valid row label.")
        for position, resolution in enumerate(resolutions):
            if resolution == "row" and np.any(
                keys[..., position][sample] != declared[rows, position]
            ):
                raise ValueError(
                    f"History {self.bias_term_slots[position]!r}: a sample "
                    "evaluated another owner than its row's declared owner key."
                )


class AtomisticMultistateEquilibriumSamples(StrictModule):
    """Admitted fixed-Hamiltonian draws of one committed multistate segment.

    The single canonical admission of multistate draws for every consumer of
    their reduced potentials (MBAR datasets, single-generator distribution
    objectives), for any state count ``K >= 1``.  Construction is a host
    transaction refusing an unsuccessful segment, active SAMS-adapting or
    bias-adapting draws, unauthenticated frozen histories
    (:meth:`AtomisticMultistateSegmentResult.bias_history_digests`), owner
    evidence that is not one fixed Hamiltonian per row
    (:meth:`AtomisticMultistateSegmentResult.authenticate_bias_owners`) and
    inconsistent identities.  Nonconservative force-port runtimes never
    produce a segment: their canonical qualification is refused.

    Columns are draw-major ``n = draw * R + replica``.  ``reduced_potentials``
    ``(K, N)`` are ``beta_k (U_k + p_k V)`` including every row's bias at its
    declared owner; ``origin_state`` is the row whose Hamiltonian generated
    the column's configuration.  When the segment captured its draw
    configurations, ``draw_positions`` ``(N, atoms, 3)`` (wrapped, dynamics
    coordinate dtype), ``draw_image_counts`` ``(N, atoms, 3)`` ``int32``,
    ``draw_species`` ``(N, atoms)`` ``int32`` and ``draw_cell_vectors``
    ``(N, *cell)`` (``None`` without a cell) are the exact configuration of
    each column; otherwise all four are ``None``.  Identifiers are the
    segment's exact host strings.
    """

    reduced_potentials: Array
    coverage: Array
    sample_active: Array
    origin_state: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    inverse_temperatures: Array
    draw_positions: Array | None
    draw_image_counts: Array | None
    draw_species: Array | None
    draw_cell_vectors: Array | None
    state_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    bias_history_digests: tuple[str, ...] = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    segment_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)

    @checked
    def __init__(self, result: AtomisticMultistateSegmentResult, /) -> None:
        if not bool(np.asarray(result.successful)):
            raise ValueError("Multistate segment must be successful before analysis.")
        values = jnp.asarray(result.reduced_potentials)
        coverage = jnp.asarray(result.coverage, dtype=jnp.bool_)
        active = jnp.asarray(result.sample_active, dtype=jnp.bool_)
        if values.ndim != 3:
            raise ValueError(
                "Segment reduced_potentials must have shape (capacity,states,replicas)."
            )
        capacity, state_count, replica_count = values.shape
        expected_samples = (capacity, replica_count)
        if coverage.shape != values.shape:
            raise ValueError("Segment coverage must match reduced_potentials.")
        for name, array in (
            ("sample_active", active),
            ("origin_state", result.origin_state),
            ("state_at_replica", result.state_at_replica),
            ("chain_index", result.chain_index),
            ("draw_index", result.draw_index),
            ("repeat_index", result.repeat_index),
            ("dependence_group_index", result.dependence_group_index),
            ("sams_adapting", result.sams_adapting),
            ("bias_adapting", result.bias_adapting),
        ):
            if jnp.asarray(array).shape != expected_samples:
                raise ValueError(f"Segment {name} must have shape (capacity,replicas).")
        if bool(jnp.any(active & jnp.asarray(result.sams_adapting, dtype=jnp.bool_))):
            raise ValueError(
                "SAMS adaptation draws must be inactive for free-energy analysis."
            )
        if bool(jnp.any(active & jnp.asarray(result.bias_adapting, dtype=jnp.bool_))):
            raise ValueError(
                "Draws under an advancing bias history are not equilibrium samples; "
                "freeze the history before free-energy analysis."
            )
        digests = result.bias_history_digests()
        result.authenticate_bias_owners()
        if (
            len(result.state_ids) != state_count
            or len(result.potential_ids) != state_count
            or len(result.bias_ids) != state_count
            or result.inverse_temperatures.shape != (state_count,)
        ):
            raise ValueError(
                "Segment state/potential/bias identities do not match the state axis."
            )
        if result.unit_id != "1":
            raise ValueError("Multistate reduced potentials must use unit_id='1'.")
        if result.producer_id != result.runtime_id:
            raise ValueError("Segment producer_id must identify its prepared runtime.")
        configurations = _admitted_draw_configurations(result, active)
        self.reduced_potentials = values.transpose((1, 0, 2)).reshape((state_count, -1))
        self.coverage = coverage.transpose((1, 0, 2)).reshape((state_count, -1))
        self.sample_active = active.reshape((-1,))
        self.origin_state = jnp.asarray(result.origin_state).reshape((-1,))
        self.chain_index = jnp.asarray(result.chain_index).reshape((-1,))
        self.draw_index = jnp.asarray(result.draw_index).reshape((-1,))
        self.repeat_index = jnp.asarray(result.repeat_index).reshape((-1,))
        self.dependence_group_index = jnp.asarray(result.dependence_group_index).reshape(
            (-1,)
        )
        self.inverse_temperatures = jnp.asarray(result.inverse_temperatures)
        self.draw_positions = configurations["draw_positions"]
        self.draw_image_counts = configurations["draw_image_counts"]
        self.draw_species = configurations["draw_species"]
        self.draw_cell_vectors = configurations["draw_cell_vectors"]
        self.state_ids = result.state_ids
        self.potential_ids = result.potential_ids
        self.bias_ids = result.bias_ids
        self.bias_history_digests = digests
        self.measure_id = result.measure_id
        self.reduced_convention_id = result.reduced_convention_id
        self.qualification_id = result.qualification_id
        self.sampling_exact = result.sampling_exact
        self.sampling_bias_bound = float(np.asarray(result.sampling_bias_bound))
        self.producer_id = result.producer_id
        self.run_id = result.run_id
        self.segment_id = result.segment_id
        self.unit_system_id = result.units.unit_system_id
        self.unit_id = result.unit_id


def _admitted_draw_configurations(
    result: AtomisticMultistateSegmentResult, active: Array, /
) -> _DrawConfigurations:
    """Validated, draw-major flattened configurations of a segment (host)."""
    positions = result.draw_positions
    images = result.draw_image_counts
    species = result.draw_species
    cells = result.draw_cell_vectors
    if positions is None:
        if images is not None or species is not None or cells is not None:
            raise ValueError("Segment draw configurations are incomplete.")
        return _absent_draw_configurations()
    if images is None or species is None:
        raise ValueError("Segment draw configurations are incomplete.")
    capacity, replicas = active.shape
    atoms = positions.shape[2] if positions.ndim == 4 else -1
    if (
        positions.shape != (capacity, replicas, atoms, 3)
        or images.shape != positions.shape
        or images.dtype != jnp.int32
        or species.shape != (capacity, replicas, atoms)
        or species.dtype != jnp.int32
        or (cells is not None and cells.shape[:2] != (capacity, replicas))
    ):
        raise ValueError("Segment draw configurations do not match its draws.")
    host_active = np.asarray(active, dtype=np.bool_)
    if np.any(~np.isfinite(np.asarray(positions)[host_active])) or np.any(
        np.asarray(species)[host_active] < 0
    ):
        raise ValueError("An active draw carries no captured configuration.")
    columns = capacity * replicas
    return {
        "draw_positions": positions.reshape((columns, atoms, 3)),
        "draw_image_counts": images.reshape((columns, atoms, 3)),
        "draw_species": species.reshape((columns, atoms)),
        "draw_cell_vectors": None
        if cells is None
        else cells.reshape((columns, *cells.shape[2:])),
    }


__all__ = [
    "AtomisticCanonicalSamplingQualification",
    "AtomisticDrawConfigurationCapture",
    "AtomisticMultistateEquilibriumSamples",
    "AtomisticMultistateIteration",
    "AtomisticMultistatePlan",
    "AtomisticMultistateSegmentPlan",
    "AtomisticMultistateSegmentResult",
    "AtomisticMultistateState",
    "AtomisticReducedPotentialEvaluation",
    "AtomisticReplicaExchangePlan",
    "AtomisticSAMSPlan",
    "AtomisticSAMSState",
    "PreparedAtomisticMultistate",
]

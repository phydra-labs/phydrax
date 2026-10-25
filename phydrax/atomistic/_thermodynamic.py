#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, parse
from ._system import PreparedAtomisticSystem


AtomisticEnsembleKind = Literal["nve", "nvt", "npt", "nph"]
AtomisticVolumeCoordinateConvention = Literal["molecular-center", "atomic"]
AtomisticMomentumRestriction = Literal["none", "center-of-mass"]
_ENSEMBLE_CODES = {"nve": 0, "nvt": 1, "npt": 2, "nph": 3}
_FROZEN_PREFIX = "frozen:"


def _stack_identities(values: tuple[IdentityWords, ...], /) -> IdentityWords:
    """Stack unbatched identities into one ``(N, 8)`` dynamic word table."""
    return IdentityWords(jnp.stack(tuple(value.words for value in values)))


def _identity_rows(identities: IdentityWords, /) -> tuple[str, ...]:
    """Host decoding of every row of a stacked ``(N, 8)`` identity table."""
    return tuple(
        IdentityWords(identities.words[index]).fingerprint()
        for index in range(identities.words.shape[0])
    )


class AtomisticBiasRowBinding(StrictModule, NonTrainableState):
    """Bias part of one thermodynamic row: per-term controls, owner and coupling.

    Keys are the declared slot names (prepared term ``name``) of stateful bias
    terms.  ``controls`` maps a slot to the row's control vector (umbrella
    centers of a window); unnamed terms keep their state's controls.
    ``owner_keys`` select the history owner a state-owned or walker-owned
    history evaluates for this row; ``couplings`` scale a term's energy
    (``0`` is the unbiased target row of the same term).  ``histories``
    declare each history term as ``"adaptive"`` (a live, advancing history:
    rows sampled with it are not equilibrium samples) or
    ``"frozen:<digest>"``, which makes the row a fixed Hamiltonian.  ``bias_id``
    fingerprints all of it and enters the row's potential identity.

    PyTree: ``term_slots`` and ``history_kinds`` (``"adaptive"``, ``"frozen"``
    or ``None``) are static declared labels/selectors; ``owner_keys``
    (``int32[B]``), ``couplings`` (``float64[B]``), the frozen history digests
    ``history_identities`` (``(B, 8)`` words, zero for non-frozen slots) and
    ``bias_identity`` are dynamic, so rows differing only in content share one
    executable.  ``bias_id`` and ``histories`` decode them on the host.
    """

    term_slots: tuple[str, ...] = eqx.field(static=True)
    controls: tuple[Array | None, ...]
    owner_keys: Array
    couplings: Array
    history_kinds: tuple[str | None, ...] = eqx.field(static=True)
    history_identities: IdentityWords
    bias_identity: IdentityWords

    def __init__(
        self,
        controls: Mapping[str, ArrayLike],
        /,
        *,
        owner_keys: Mapping[str, int] | None = None,
        couplings: Mapping[str, float] | None = None,
        histories: Mapping[str, str] | None = None,
    ) -> None:
        owners = {} if owner_keys is None else dict(owner_keys)
        scales = {} if couplings is None else dict(couplings)
        declared = {} if histories is None else dict(histories)
        slots = tuple(sorted(set(controls) | set(owners) | set(scales) | set(declared)))
        if not slots or any(
            not isinstance(slot, str) or not slot or slot != slot.strip()
            for slot in slots
        ):
            raise ValueError("A bias row binds canonical nonempty bias term slots.")
        values = tuple(
            None
            if slot not in controls
            else jnp.asarray(controls[slot], dtype=jnp.float64).reshape((-1,))
            for slot in slots
        )
        if any(
            value is not None and bool(jnp.any(~jnp.isfinite(value))) for value in values
        ):
            raise ValueError("Bias row controls must be finite.")
        keys = tuple(int(owners.get(slot, 0)) for slot in slots)
        if any(key < 0 for key in keys):
            raise ValueError("History owner keys must be nonnegative.")
        coupling = tuple(float(scales.get(slot, 1.0)) for slot in slots)
        if any(not np.isfinite(value) or value < 0.0 for value in coupling):
            raise ValueError("Bias couplings must be finite and nonnegative.")
        history = tuple(declared.get(slot) for slot in slots)
        if any(
            value is not None
            and (value != "adaptive" and not value.startswith(_FROZEN_PREFIX))
            for value in history
        ):
            raise ValueError("Histories are 'adaptive' or a 'frozen:<digest>' identity.")
        kinds = tuple(
            None if value is None else "adaptive" if value == "adaptive" else "frozen"
            for value in history
        )
        words = np.zeros((len(slots), 8), dtype=np.uint32)
        for index, value in enumerate(history):
            if value is not None and value.startswith(_FROZEN_PREFIX):
                words[index] = np.asarray(
                    IdentityWords.from_fingerprint(value[len(_FROZEN_PREFIX) :]).words
                )
        self.term_slots = slots
        self.controls = values
        self.owner_keys = jnp.asarray(keys, dtype=jnp.int32)
        self.couplings = jnp.asarray(coupling, dtype=jnp.float64)
        self.history_kinds = kinds
        self.history_identities = IdentityWords(jnp.asarray(words))
        self.bias_identity = IdentityWords.of(
            {
                "kind": "atomistic-bias-row",
                "terms": list(slots),
                "controls": [
                    None if value is None else array_tree_fingerprint(np.asarray(value))
                    for value in values
                ],
                "owner_keys": list(keys),
                "couplings": [value.hex() for value in coupling],
                "histories": list(history),
            }
        )

    @property
    def bias_id(self) -> str:
        """Host decoding of the canonical bias-row digest."""
        return self.bias_identity.fingerprint()

    @property
    def histories(self) -> tuple[str | None, ...]:
        """Host decoding of the declared ``"adaptive"``/``"frozen:<digest>"`` labels."""
        digests = _identity_rows(self.history_identities)
        return tuple(
            None
            if kind is None
            else "adaptive"
            if kind == "adaptive"
            else f"{_FROZEN_PREFIX}{digest}"
            for kind, digest in zip(self.history_kinds, digests, strict=True)
        )

    @property
    def adaptive(self) -> bool:
        return any(kind == "adaptive" for kind in self.history_kinds)


class AtomisticPhaseSpaceMeasurePlan(StrictModule, NonTrainableState):
    """Single authority of the coordinate, momentum, cell and unit measure.

    ``volume_coordinate_convention`` names the scaled entities of a volume
    change: ``"molecular-center"`` scales molecule centers (rigid molecular
    frames, ``scaled_entity_count`` molecules) and ``"atomic"`` scales every
    active atom (``scaled_entity_count`` equal to the active atom count), the
    convention of integrator-owned MTK cells.  ``momentum_restriction``
    declares whether total linear momentum is a removed (conserved, zero)
    degree of freedom; the prepared table subtracts it, together with the
    independent constraint rank, from the thermostatted degrees of freedom.

    PyTree: capacity, unit system, convention and restriction are static
    layout/selectors.  Material-derived counts (``scaled_entity_count``,
    ``mobile_count``, ``active_count``) are dynamic ``int32`` scalars and the
    system, topology, coordinate-map and measure digests are dynamic words
    (``*_identity``) decoded on the host by the former ``*_id`` names.
    """

    particle_capacity: int = eqx.field(static=True)
    scaled_entity_count: Array
    volume_coordinate_convention: AtomisticVolumeCoordinateConvention = eqx.field(
        static=True
    )
    momentum_restriction: AtomisticMomentumRestriction = eqx.field(static=True)
    mobile_count: Array
    active_count: Array
    system_identity: IdentityWords
    topology_identity: IdentityWords
    unit_system_id: str = eqx.field(static=True)
    coordinate_map_identity: IdentityWords
    measure_identity: IdentityWords

    @checked
    def __init__(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        scaled_entity_count: int | None = None,
        volume_coordinate_convention: AtomisticVolumeCoordinateConvention = (
            "molecular-center"
        ),
        momentum_restriction: AtomisticMomentumRestriction = "none",
    ) -> None:
        convention = str(volume_coordinate_convention)
        active = np.asarray(system.active_mask, dtype=np.bool_)
        mobile = np.asarray(system.mobile_mask, dtype=np.bool_)
        active_count = int(np.count_nonzero(active))
        mobile_count = int(np.count_nonzero(mobile))
        if convention == "molecular-center":
            entities = (
                int(system.molecule_count)
                if scaled_entity_count is None
                else int(scaled_entity_count)
            )
        elif convention == "atomic":
            entities = (
                active_count if scaled_entity_count is None else int(scaled_entity_count)
            )
            if entities != active_count:
                raise ValueError(
                    "Atomic volume scaling scales every active atom; "
                    "scaled_entity_count must equal the active atom count."
                )
        else:
            raise ValueError(
                "volume_coordinate_convention must be 'molecular-center' or 'atomic'."
            )
        if entities <= 0:
            raise ValueError(
                "Phase-space volume scaling requires a positive entity count."
            )
        restriction = parse(
            momentum_restriction, AtomisticMomentumRestriction, "momentum_restriction"
        )
        if restriction == "center-of-mass" and not np.array_equal(active, mobile):
            raise ValueError(
                "A center-of-mass momentum restriction requires every active atom "
                "to be mobile; fixed atoms exchange momentum with the system."
            )
        system_id = system.prepared_id
        topology_id = system.topology.topology_id
        unit_system_id = system.plan.units.unit_system_id
        coordinate_map_id = system.coordinate_map.prepared_id
        self.particle_capacity = system.capacity
        self.scaled_entity_count = jnp.asarray(entities, dtype=jnp.int32)
        self.volume_coordinate_convention = convention
        self.momentum_restriction = restriction
        self.mobile_count = jnp.asarray(mobile_count, dtype=jnp.int32)
        self.active_count = jnp.asarray(active_count, dtype=jnp.int32)
        self.system_identity = IdentityWords.from_fingerprint(system_id)
        self.topology_identity = IdentityWords.from_fingerprint(topology_id)
        self.unit_system_id = unit_system_id
        self.coordinate_map_identity = IdentityWords.from_fingerprint(coordinate_map_id)
        self.measure_identity = IdentityWords.of(
            {
                "kind": "atomistic-phase-space-measure",
                "system": system_id,
                "topology": topology_id,
                "units": unit_system_id,
                "coordinate_map": coordinate_map_id,
                "particle_capacity": system.capacity,
                "scaled_entity_count": entities,
                "volume_coordinate_convention": convention,
                "momentum_restriction": restriction,
                "particle_ids": array_tree_fingerprint(system.plan.particle_ids),
                "masses": array_tree_fingerprint(system.plan.masses),
                "active": array_tree_fingerprint(system.active_mask),
                "mobile": array_tree_fingerprint(system.mobile_mask),
            }
        )

    @property
    def system_id(self) -> str:
        return self.system_identity.fingerprint()

    @property
    def topology_id(self) -> str:
        return self.topology_identity.fingerprint()

    @property
    def coordinate_map_id(self) -> str:
        return self.coordinate_map_identity.fingerprint()

    @property
    def measure_id(self) -> str:
        """Host decoding of the canonical phase-space measure digest."""
        return self.measure_identity.fingerprint()


class AtomisticThermodynamicStatePlan(StrictModule, NonTrainableState):
    """One physical ensemble state on a declared common phase-space measure.

    PyTree: ``ensemble`` and ``control_ids`` are static selectors/labels and
    ``state_label`` is the declared ``state_id`` (``None`` when generated).
    ``temperature``/``pressure`` are dynamic ``float64`` scalars (``None``
    exactly when the ensemble has no such intensive variable).  The generated
    content digest is always the dynamic ``state_identity``; ``state_id``
    returns the declared label or, without one, that digest (unchanged
    values).  ``bias_id`` decodes the bias row's dynamic identity.
    """

    phase_space: AtomisticPhaseSpaceMeasurePlan
    controls: Array
    ensemble: AtomisticEnsembleKind = eqx.field(static=True)
    temperature: Array | None
    pressure: Array | None
    control_ids: tuple[str, ...] = eqx.field(static=True)
    bias: AtomisticBiasRowBinding | None
    state_label: str | None = eqx.field(static=True)
    state_identity: IdentityWords

    @checked
    def __init__(
        self,
        phase_space: AtomisticPhaseSpaceMeasurePlan,
        /,
        *,
        ensemble: AtomisticEnsembleKind,
        temperature: float | None = None,
        pressure: float | None = None,
        controls: ArrayLike | None = None,
        control_ids: tuple[str, ...] = (),
        bias: AtomisticBiasRowBinding | None = None,
        state_id: str | None = None,
    ) -> None:
        if ensemble not in _ENSEMBLE_CODES:
            raise ValueError("ensemble must be 'nve', 'nvt', 'npt', or 'nph'.")
        temperature_ = None if temperature is None else float(temperature)
        pressure_ = None if pressure is None else float(pressure)
        ids = tuple(control_ids)
        values = (
            jnp.zeros((0,), dtype=jnp.float64)
            if controls is None
            else jnp.asarray(controls, dtype=jnp.float64).reshape((-1,))
        )
        if (
            len(ids) != values.size
            or len(set(ids)) != len(ids)
            or any(
                not isinstance(name, str) or not name or name != name.strip()
                for name in ids
            )
            or bool(jnp.any(~jnp.isfinite(values)))
        ):
            raise ValueError("Thermodynamic control IDs and values are invalid.")
        if ensemble == "nve":
            valid_intensives = temperature_ is None and pressure_ is None
        elif ensemble == "nvt":
            valid_intensives = (
                temperature_ is not None
                and np.isfinite(temperature_)
                and temperature_ > 0.0
                and pressure_ is None
            )
        elif ensemble == "npt":
            valid_intensives = (
                temperature_ is not None
                and np.isfinite(temperature_)
                and temperature_ > 0.0
                and pressure_ is not None
                and np.isfinite(pressure_)
            )
        else:
            valid_intensives = (
                temperature_ is None and pressure_ is not None and np.isfinite(pressure_)
            )
        if not valid_intensives:
            raise ValueError(
                "Thermodynamic intensive variables do not match the ensemble."
            )
        if bias is not None and not isinstance(bias, AtomisticBiasRowBinding):
            raise TypeError("bias must be an AtomisticBiasRowBinding or None.")
        label = None if state_id is None else str(state_id)
        if label is not None and (not label or label != label.strip()):
            raise ValueError("state_id must be a canonical nonempty string.")
        self.phase_space = phase_space
        self.controls = values
        self.ensemble = ensemble
        self.temperature = (
            None if temperature_ is None else jnp.asarray(temperature_, dtype=jnp.float64)
        )
        self.pressure = (
            None if pressure_ is None else jnp.asarray(pressure_, dtype=jnp.float64)
        )
        self.control_ids = ids
        self.bias = bias
        self.state_label = label
        self.state_identity = IdentityWords.of(
            {
                "kind": "atomistic-thermodynamic-state",
                "phase_space": phase_space.measure_id,
                "ensemble": ensemble,
                "temperature": temperature_,
                "pressure": pressure_,
                "control_ids": list(ids),
                "controls": array_tree_fingerprint(values),
                "bias": None if bias is None else bias.bias_id,
            }
        )

    @property
    def bias_id(self) -> str | None:
        return None if self.bias is None else self.bias.bias_id

    @property
    def state_id(self) -> str:
        """Declared label, or the generated canonical state digest (host)."""
        return (
            self.state_identity.fingerprint()
            if self.state_label is None
            else self.state_label
        )

    def prepare(self, dynamics: Any, /) -> "PreparedThermodynamicStateTable":
        return PreparedThermodynamicStateTable(dynamics, (self,))


class _ThermodynamicStateRows(StrictModule):
    state_index: Array
    beta: Array
    temperature: Array
    temperature_mask: Array
    pressure: Array
    hamiltonian_index: Array
    pressure_mask: Array
    controls: Array
    ensemble_code: Array
    valid: Array


class PreparedThermodynamicStateTable(StrictModule, NonTrainableState):
    """Fixed-shape numerical thermodynamic states bound to one dynamics runtime.

    PyTree (content-free; every material and parameter revision of one
    runtime layout shares one executable):

    - static layout/selectors: ``state_count``, ``control_count``,
      ``control_ids``, ``control_layout_id``, ``state_labels`` (declared
      ``state_id`` labels, ``None`` when generated), ``bias_adaptive``,
      ``volume_coordinate_convention``, ``unit_system_id``,
      ``momentum_restriction`` and ``dynamics_layout_id`` (the bound
      runtime's ``layout_id``);
    - dynamic arrays: the row tables, ``requires_cross_evaluation`` (bool),
      ``degrees_of_freedom``, ``constraint_rank`` and ``scaled_entity_count``
      (``int32`` scalars);
    - dynamic identity words (host ``*_id`` properties decode the unchanged
      digests): ``table_identity``, ``state_identities`` (generated row
      digests), ``potential_identities``, ``bias_identities`` (``None``
      without bias rows), ``phase_space_measure_identity``,
      ``declared_phase_space_measure_identity``, ``reduced_convention_identity``,
      ``dynamics_identity``, ``system_identity``, ``program_identity`` and
      ``constraint_manifold_identity`` (``None`` without constraints).

    ``validate_dynamics`` is the host exact refusal; compiled code uses the
    static ``require_dynamics_layout`` and the traced ``bound_to_dynamics``.
    """

    beta: Array
    temperature: Array
    temperature_mask: Array
    pressure: Array
    pressure_mask: Array
    controls: Array
    ensemble_code: Array
    state_labels: tuple[str | None, ...] = eqx.field(static=True)
    state_identities: IdentityWords
    control_ids: tuple[str, ...] = eqx.field(static=True)
    bias_identities: IdentityWords | None
    bias_rows: tuple[AtomisticBiasRowBinding | None, ...]
    bias_adaptive: tuple[bool, ...] = eqx.field(static=True)
    potential_identities: IdentityWords
    hamiltonian_index: Array
    state_count: int = eqx.field(static=True)
    control_count: int = eqx.field(static=True)
    phase_space_measure_identity: IdentityWords
    declared_phase_space_measure_identity: IdentityWords
    scaled_entity_count: Array
    volume_coordinate_convention: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    system_identity: IdentityWords
    program_identity: IdentityWords
    control_layout_id: str = eqx.field(static=True)
    dynamics_identity: IdentityWords
    dynamics_layout_id: str = eqx.field(static=True)
    constraint_manifold_identity: IdentityWords | None
    reduced_convention_identity: IdentityWords
    requires_cross_evaluation: Array
    degrees_of_freedom: Array
    constraint_rank: Array
    momentum_restriction: str = eqx.field(static=True)
    table_identity: IdentityWords

    def __init__(
        self,
        dynamics: Any,
        states: Iterable[AtomisticThermodynamicStatePlan],
        /,
    ) -> None:
        from ._dynamics import PreparedAtomisticDynamics

        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        plans = tuple(states)
        measure_id, constraint_manifold_id, executed_measure_id = _admit_shared_measure(
            dynamics, plans
        )
        measure = plans[0].phase_space
        control_ids = plans[0].control_ids
        state_ids = _admit_row_layout(dynamics, plans)
        if any(state.ensemble in ("npt", "nph") for state in plans):
            if dynamics.system.cell is None or not dynamics.system.cell.fully_periodic:
                raise ValueError(
                    "NPT/NPH states require a fully periodic atomistic system."
                )
        _admit_method(dynamics, plans, measure)
        constraint_rank, degrees_of_freedom = _kinetic_degrees_of_freedom(
            dynamics, measure
        )
        scaled_entity_count = int(measure.scaled_entity_count)
        dtype = np.dtype(dynamics.system.plan.coordinate_dtype)
        boltzmann = dynamics.system.plan.units.boltzmann_constant
        temperature_mask, pressure_mask, temperature, pressure, beta = (
            _state_condition_rows(plans, dtype, boltzmann)
        )
        controls = np.stack([np.asarray(state.controls, dtype=dtype) for state in plans])
        ensemble_code = np.asarray(
            [_ENSEMBLE_CODES[state.ensemble] for state in plans], dtype=np.int32
        )
        bias_rows, bias_ids, layout = _admit_bias_layout(dynamics, plans)
        if len(plans) > 1:
            dynamics.require_conservative_force_definition(
                "A multi-state thermodynamic table (cross-evaluated reduced potentials)"
            )
        requires_cross = bool(np.any(controls != controls[0])) or len(set(bias_ids)) > 1
        hamiltonian_index = _hamiltonian_index(controls, bias_ids)
        layout_id = dynamics.potential.control_layout_id
        program_id = dynamics.potential.prepared_id
        system_id = dynamics.system.prepared_id
        dynamics_id = dynamics.prepared_id
        potential_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "atomistic-state-hamiltonian",
                    "program": program_id,
                    "control_layout": layout_id,
                    "controls": array_tree_fingerprint(control_row),
                    "bias": bias_id,
                }
            )
            for bias_id, control_row in zip(bias_ids, controls, strict=True)
        )
        reduced_convention_id = canonical_fingerprint(
            {
                "kind": "atomistic-configurational-reduced-potential",
                "expression": "beta*(potential_energy+pressure*volume)",
                "unit_id": "1",
                "unit_system": dynamics.system.plan.units.unit_system_id,
                "volume_coordinate_convention": measure.volume_coordinate_convention,
                "measure": executed_measure_id,
            }
        )
        table_id = canonical_fingerprint(
            {
                "kind": "prepared-thermodynamic-state-table",
                "dynamics": dynamics_id,
                "system": system_id,
                "program": program_id,
                "measure": executed_measure_id,
                "declared_measure": measure_id,
                "scaled_entity_count": scaled_entity_count,
                "volume_coordinate_convention": measure.volume_coordinate_convention,
                "constraint_manifold": constraint_manifold_id,
                "reduced_convention": reduced_convention_id,
                "units": dynamics.system.plan.units.unit_system_id,
                "layout": layout_id,
                "state_ids": list(state_ids),
                "bias_ids": list(bias_ids),
                "potential_ids": list(potential_ids),
                "beta": array_tree_fingerprint(beta),
                "temperature": array_tree_fingerprint(temperature),
                "temperature_mask": array_tree_fingerprint(temperature_mask),
                "pressure": array_tree_fingerprint(pressure),
                "pressure_mask": array_tree_fingerprint(pressure_mask),
                "controls": array_tree_fingerprint(controls),
                "ensemble_code": array_tree_fingerprint(ensemble_code),
                "hamiltonian_index": array_tree_fingerprint(hamiltonian_index),
                "degrees_of_freedom": degrees_of_freedom,
                "constraint_rank": constraint_rank,
                "momentum_restriction": measure.momentum_restriction,
            }
        )
        self.beta = jnp.asarray(beta)
        self.temperature = jnp.asarray(temperature)
        self.temperature_mask = jnp.asarray(temperature_mask)
        self.pressure = jnp.asarray(pressure)
        self.pressure_mask = jnp.asarray(pressure_mask)
        self.controls = jnp.asarray(controls)
        self.ensemble_code = jnp.asarray(ensemble_code)
        self.state_labels = tuple(state.state_label for state in plans)
        self.state_identities = _stack_identities(
            tuple(state.state_identity for state in plans)
        )
        self.control_ids = control_ids
        self.hamiltonian_index = jnp.asarray(hamiltonian_index)
        self.bias_identities = (
            None
            if layout is None
            else _stack_identities(
                tuple(row.bias_identity for row in bias_rows if row is not None)
            )
        )
        self.bias_rows = bias_rows
        self.bias_adaptive = tuple(row is not None and row.adaptive for row in bias_rows)
        self.potential_identities = _stack_identities(
            tuple(IdentityWords.from_fingerprint(value) for value in potential_ids)
        )
        self.state_count = len(plans)
        self.control_count = len(control_ids)
        self.phase_space_measure_identity = IdentityWords.from_fingerprint(
            executed_measure_id
        )
        self.declared_phase_space_measure_identity = measure.measure_identity
        self.scaled_entity_count = jnp.asarray(scaled_entity_count, dtype=jnp.int32)
        self.volume_coordinate_convention = measure.volume_coordinate_convention
        self.unit_system_id = dynamics.system.plan.units.unit_system_id
        self.system_identity = dynamics.system.prepared_identity
        self.program_identity = dynamics.potential.prepared_identity
        self.control_layout_id = layout_id
        self.dynamics_identity = dynamics.prepared_identity
        self.dynamics_layout_id = dynamics.layout_id
        self.constraint_manifold_identity = (
            None
            if dynamics.constraints is None
            else dynamics.constraints.prepared_identity
        )
        self.reduced_convention_identity = IdentityWords.from_fingerprint(
            reduced_convention_id
        )
        self.requires_cross_evaluation = jnp.asarray(requires_cross)
        self.degrees_of_freedom = jnp.asarray(degrees_of_freedom, dtype=jnp.int32)
        self.constraint_rank = jnp.asarray(constraint_rank, dtype=jnp.int32)
        self.momentum_restriction = measure.momentum_restriction
        self.table_identity = IdentityWords.from_fingerprint(table_id)

    # ------------------------------------------------------- host identities

    @property
    def table_id(self) -> str:
        """Host decoding of the canonical table digest."""
        return self.table_identity.fingerprint()

    @property
    def state_ids(self) -> tuple[str, ...]:
        """Declared labels, or the generated canonical row digests (host)."""
        return tuple(
            digest if label is None else label
            for label, digest in zip(
                self.state_labels, _identity_rows(self.state_identities), strict=True
            )
        )

    @property
    def bias_ids(self) -> tuple[str | None, ...]:
        if self.bias_identities is None:
            return (None,) * self.state_count
        return _identity_rows(self.bias_identities)

    @property
    def potential_ids(self) -> tuple[str, ...]:
        return _identity_rows(self.potential_identities)

    @property
    def phase_space_measure_id(self) -> str:
        return self.phase_space_measure_identity.fingerprint()

    @property
    def declared_phase_space_measure_id(self) -> str:
        return self.declared_phase_space_measure_identity.fingerprint()

    @property
    def system_id(self) -> str:
        return self.system_identity.fingerprint()

    @property
    def program_id(self) -> str:
        return self.program_identity.fingerprint()

    @property
    def dynamics_id(self) -> str:
        return self.dynamics_identity.fingerprint()

    @property
    def constraint_manifold_id(self) -> str | None:
        if self.constraint_manifold_identity is None:
            return None
        return self.constraint_manifold_identity.fingerprint()

    @property
    def reduced_convention_id(self) -> str:
        return self.reduced_convention_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free executable layout of this table."""
        first = self.bias_rows[0]
        return canonical_fingerprint(
            {
                "kind": "prepared-thermodynamic-state-table-layout",
                "dynamics": self.dynamics_layout_id,
                "state_count": self.state_count,
                "state_labels": list(self.state_labels),
                "control_layout": self.control_layout_id,
                "bias_slots": None if first is None else list(first.term_slots),
                "bias_history_kinds": None
                if first is None
                else list(first.history_kinds),
                "bias_controls": None
                if first is None
                else [
                    None if value is None else list(value.shape)
                    for value in first.controls
                ],
                "bias_adaptive": list(self.bias_adaptive),
                "volume_coordinate_convention": self.volume_coordinate_convention,
                "units": self.unit_system_id,
                "constrained": self.constraint_manifold_identity is not None,
                "momentum_restriction": self.momentum_restriction,
            }
        )

    # ---------------------------------------------------------- validation

    def require_dynamics_layout(self, dynamics: Any, /) -> None:
        """Static (trace-safe) refusal of a runtime of another layout."""
        if (
            dynamics.layout_id != self.dynamics_layout_id
            or dynamics.system.plan.units.unit_system_id != self.unit_system_id
            or (dynamics.constraints is None)
            != (self.constraint_manifold_identity is None)
        ):
            raise ValueError(
                "Thermodynamic state table belongs to another runtime layout."
            )

    def bound_to_dynamics(self, dynamics: Any, /) -> Array:
        """Traced: whether this table was prepared for exactly ``dynamics``."""
        self.require_dynamics_layout(dynamics)
        bound = (
            self.dynamics_identity.equals(dynamics.prepared_identity)
            & self.system_identity.equals(dynamics.system.prepared_identity)
            & self.program_identity.equals(dynamics.potential.prepared_identity)
        )
        if self.constraint_manifold_identity is not None:
            bound = bound & self.constraint_manifold_identity.equals(
                dynamics.constraints.prepared_identity
            )
        return bound

    def validate_dynamics(self, dynamics: Any, /) -> None:
        """Host exact refusal of a table prepared for another runtime."""
        self.require_dynamics_layout(dynamics)
        constraint_manifold_id = (
            None if dynamics.constraints is None else dynamics.constraints.prepared_id
        )
        if (
            dynamics.prepared_id != self.dynamics_id
            or dynamics.system.prepared_id != self.system_id
            or dynamics.potential.prepared_id != self.program_id
            or constraint_manifold_id != self.constraint_manifold_id
        ):
            raise ValueError("Thermodynamic state table belongs to another runtime.")

    def state_at_replica(self, state_at_replica: ArrayLike, /) -> _ThermodynamicStateRows:
        indices = jnp.asarray(state_at_replica, dtype=jnp.int32)
        valid = (indices >= 0) & (indices < self.state_count)
        safe = jnp.clip(indices, 0, self.state_count - 1)
        return _ThermodynamicStateRows(
            state_index=indices,
            beta=jnp.take(self.beta, safe, axis=0),
            temperature=jnp.take(self.temperature, safe, axis=0),
            temperature_mask=jnp.take(self.temperature_mask, safe, axis=0),
            pressure=jnp.take(self.pressure, safe, axis=0),
            hamiltonian_index=jnp.take(self.hamiltonian_index, safe, axis=0),
            pressure_mask=jnp.take(self.pressure_mask, safe, axis=0),
            controls=jnp.take(self.controls, safe, axis=0),
            ensemble_code=jnp.take(self.ensemble_code, safe, axis=0),
            valid=valid,
        )


class _AdmittedEnsemble(Protocol):
    """The ensemble fact `_admit_method` reads from a thermodynamic state."""

    @property
    def ensemble(self) -> AtomisticEnsembleKind: ...


class _AdmittedMeasure(Protocol):
    """The measure facts `_admit_method` reads from a phase-space measure."""

    @property
    def volume_coordinate_convention(self) -> AtomisticVolumeCoordinateConvention: ...

    @property
    def momentum_restriction(self) -> AtomisticMomentumRestriction: ...


def _admit_method(
    dynamics: Any,
    plans: Sequence[_AdmittedEnsemble],
    measure: _AdmittedMeasure,
    /,
) -> None:
    """Admit each row ensemble for the integrator method and measure.

    The integrator alone fixes which ensembles are executable: velocity Verlet
    is NVE; Langevin, NHC, CSVR and GLE are canonical NVT (and NPT only when a
    scheduled Monte Carlo volume move owns the cell); deterministic rescaling is
    a noncanonical NVT control; MTK NPT/NPH own the cell and admit only their
    own ensemble under atomic volume scaling without holonomic constraints.
    Per-particle stochastic thermostats break total-momentum conservation and
    refuse a center-of-mass momentum restriction.
    """
    from ._cell_dynamics import MTKNPHPlan, MTKNPTPlan
    from ._dynamics import VelocityVerletPlan
    from ._stochastic_runtime import GeneralizedLangevinPlan
    from ._thermal import (
        BAOABLangevinPlan,
        NoseHooverChainPlan,
        StochasticVelocityRescalingPlan,
        VelocityRescalingPlan,
    )

    method = dynamics.integrator
    if isinstance(method, VelocityVerletPlan):
        admitted = {"nve"}
    elif isinstance(
        method,
        (
            BAOABLangevinPlan,
            NoseHooverChainPlan,
            StochasticVelocityRescalingPlan,
            GeneralizedLangevinPlan,
        ),
    ):
        admitted = {"nvt", "npt"}
    elif isinstance(method, VelocityRescalingPlan):
        admitted = {"nvt"}
    elif isinstance(method, MTKNPTPlan):
        admitted = {"npt"}
    elif isinstance(method, MTKNPHPlan):
        admitted = {"nph"}
    else:
        raise TypeError(f"Unsupported dynamics method {type(method).__name__}.")
    refused = sorted({state.ensemble for state in plans} - admitted)
    if refused:
        raise ValueError(f"{type(method).__name__} cannot execute ensembles {refused!r}.")
    if isinstance(method, (MTKNPTPlan, MTKNPHPlan)):
        if measure.volume_coordinate_convention != "atomic":
            raise ValueError(
                "MTK cell dynamics scales every atom; the phase-space measure must "
                "declare atomic volume scaling."
            )
        if dynamics.constraints is not None:
            raise ValueError(
                "Constrained MTK cell dynamics lacks a derived constrained measure "
                "and projection; refusing."
            )
    if measure.momentum_restriction == "center-of-mass" and isinstance(
        method, (BAOABLangevinPlan, GeneralizedLangevinPlan)
    ):
        raise ValueError(
            "Per-particle stochastic thermostats do not conserve total momentum; "
            "a center-of-mass momentum restriction is inadmissible."
        )


def _admit_shared_measure(
    dynamics: Any, plans: tuple[AtomisticThermodynamicStatePlan, ...], /
) -> tuple[str, str | None, str]:
    """Declared measure shared by every row and the measure the runtime executes.

    All rows declare one phase-space measure of the runtime's system; topology
    constraints need the runtime's prepared constraint executor, whose manifold
    identity binds the executed measure.  Returns the declared measure,
    constraint-manifold (``None`` without constraints) and executed measure IDs.
    """
    if not plans or any(
        not isinstance(state, AtomisticThermodynamicStatePlan) for state in plans
    ):
        raise TypeError("states must contain AtomisticThermodynamicStatePlan values.")
    measure_id = plans[0].phase_space.measure_id
    if any(state.phase_space.measure_id != measure_id for state in plans):
        raise ValueError("All thermodynamic states must share one phase-space measure.")
    if plans[0].phase_space.system_id != dynamics.system.prepared_id:
        raise ValueError("Thermodynamic states belong to another atomistic system.")
    constraint_manifold_id = (
        None if dynamics.constraints is None else dynamics.constraints.prepared_id
    )
    if dynamics.system.topology.constraint_count > 0 and constraint_manifold_id is None:
        raise ValueError(
            "Thermodynamic binding requires a prepared executor for topology constraints."
        )
    executed_measure_id = canonical_fingerprint(
        {
            "kind": "prepared-atomistic-phase-space-measure",
            "declared_measure": measure_id,
            "constraint_manifold": constraint_manifold_id,
        }
    )
    return measure_id, constraint_manifold_id, executed_measure_id


def _admit_row_layout(
    dynamics: Any, plans: tuple[AtomisticThermodynamicStatePlan, ...], /
) -> tuple[str, ...]:
    """Shared Hamiltonian control layout and the unique ordered row state IDs.

    Every row orders the runtime Hamiltonian's control IDs identically, and
    their canonical layout digest is the Hamiltonian's.
    """
    control_ids = plans[0].control_ids
    if any(state.control_ids != control_ids for state in plans):
        raise ValueError("Thermodynamic states must share one ordered control layout.")
    if (
        control_ids != dynamics.potential.control_ids
        or canonical_fingerprint(
            {
                "kind": "atomistic-control-layout",
                "control_ids": list(control_ids),
            }
        )
        != dynamics.potential.control_layout_id
    ):
        raise ValueError(
            "Thermodynamic controls do not match the Hamiltonian control layout."
        )
    state_ids = tuple(state.state_id for state in plans)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("Thermodynamic state IDs must be unique and ordered.")
    return state_ids


def _kinetic_degrees_of_freedom(
    dynamics: Any, measure: AtomisticPhaseSpaceMeasurePlan, /
) -> tuple[int, int]:
    """Independent constraint rank and the thermostatted kinetic degrees of freedom.

    Three per mobile atom, less the independent constraint rank and, under a
    center-of-mass momentum restriction, the three removed total momenta.
    """
    constraint_rank = (
        0
        if dynamics.constraints is None
        else int(dynamics.constraints.independent_constraint_count)
    )
    degrees_of_freedom = (
        3 * int(measure.mobile_count)
        - constraint_rank
        - (3 if measure.momentum_restriction == "center-of-mass" else 0)
    )
    if degrees_of_freedom <= 0:
        raise ValueError("The admitted phase space has no kinetic degrees of freedom.")
    return constraint_rank, degrees_of_freedom


def _state_condition_rows(
    plans: tuple[AtomisticThermodynamicStatePlan, ...],
    dtype: np.dtype,
    boltzmann: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked temperature/pressure columns and inverse temperature in ``dtype``.

    Undeclared conditions are zero under a false mask; ``beta`` is
    ``1 / (k_B T)`` on temperature rows and zero elsewhere.
    """
    temperature_mask = np.asarray(
        [state.temperature is not None for state in plans], dtype=np.bool_
    )
    pressure_mask = np.asarray(
        [state.pressure is not None for state in plans], dtype=np.bool_
    )
    temperature = np.asarray(
        [
            0.0 if state.temperature is None else float(state.temperature)
            for state in plans
        ],
        dtype=dtype,
    )
    pressure = np.asarray(
        [0.0 if state.pressure is None else float(state.pressure) for state in plans],
        dtype=dtype,
    )
    beta = np.zeros_like(temperature)
    beta[temperature_mask] = 1.0 / (boltzmann * temperature[temperature_mask])
    return temperature_mask, pressure_mask, temperature, pressure, beta


def _admit_bias_layout(
    dynamics: Any, plans: tuple[AtomisticThermodynamicStatePlan, ...], /
) -> tuple[
    tuple[AtomisticBiasRowBinding | None, ...],
    tuple[str | None, ...],
    tuple[str, ...] | None,
]:
    """Row bias bindings, their digests and the one term layout every row binds.

    All rows bind the same bias term slots (or none), and those slots name
    stateful terms of the runtime's Hamiltonian.
    """
    bias_rows = tuple(state.bias for state in plans)
    bias_ids = tuple(state.bias_id for state in plans)
    stateful = set(dynamics.program_term_state_slots)
    layouts = {None if row is None else row.term_slots for row in bias_rows}
    if len(layouts) != 1:
        raise ValueError(
            "Every thermodynamic row must bind the same bias terms (or none)."
        )
    layout = next(iter(layouts))
    if layout is not None and not set(layout) <= stateful:
        raise ValueError(
            "Bias rows must name stateful terms of this runtime's Hamiltonian."
        )
    return bias_rows, bias_ids, layout


def _hamiltonian_index(
    controls: np.ndarray, bias_ids: tuple[str | None, ...], /
) -> np.ndarray:
    """Labels of the distinct ``(control row, bias)`` Hamiltonians in first-row order."""
    hamiltonian_labels: dict[tuple[bytes, str | None], int] = {}
    hamiltonian_indices = []
    for control_row, bias_id in zip(controls, bias_ids, strict=True):
        key = (np.ascontiguousarray(control_row).tobytes(), bias_id)
        if key not in hamiltonian_labels:
            hamiltonian_labels[key] = len(hamiltonian_labels)
        hamiltonian_indices.append(hamiltonian_labels[key])
    return np.asarray(hamiltonian_indices, dtype=np.int32)


__all__ = [
    "AtomisticBiasRowBinding",
    "AtomisticEnsembleKind",
    "AtomisticMomentumRestriction",
    "AtomisticVolumeCoordinateConvention",
    "AtomisticPhaseSpaceMeasurePlan",
    "AtomisticThermodynamicStatePlan",
    "PreparedThermodynamicStateTable",
]

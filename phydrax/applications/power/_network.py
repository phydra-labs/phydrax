#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Balanced RMS phasors: exp(+i wt), inward terminal currents, generation-positive S."""

from __future__ import annotations

from math import isfinite, sqrt

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import ArraySpace
from ...sparse import EdgeRelation, SparseCoordinateOperator


class PowerBase(StrictModule):
    """Total three-phase MVA and line-line RMS kV bases; no extra factor of three."""

    base_mva: float = 100.0

    def __check_init__(self):
        if not isfinite(self.base_mva) or self.base_mva <= 0:
            raise ValueError("base_mva must be finite and positive.")

    def impedance_ohm(self, base_kv: float) -> float:
        if not isfinite(base_kv) or base_kv <= 0:
            raise ValueError("base_kv must be finite and positive.")
        return base_kv**2 / self.base_mva

    def current_ampere(self, base_kv: float) -> float:
        if not isfinite(base_kv) or base_kv <= 0:
            raise ValueError("base_kv must be finite and positive.")
        return self.base_mva * 1000 / (sqrt(3) * base_kv)

    def phase_voltage_volt(self, base_kv: float) -> float:
        if not isfinite(base_kv) or base_kv <= 0:
            raise ValueError("base_kv must be finite and positive.")
        return base_kv * 1000 / sqrt(3)

    def rebase_impedance(
        self, value: ArrayLike, *, old_mva: float, old_kv: float, new_kv: float
    ) -> Array:
        """Convert a per-unit impedance from a declared old base to this base."""
        if any(not isfinite(v) or v <= 0 for v in (old_mva, old_kv, new_kv)):
            raise ValueError("Impedance rebasing requires positive finite bases.")
        return jnp.asarray(value) * (self.base_mva / old_mva) * (old_kv / new_kv) ** 2


class Bus(StrictModule):
    """Physical bus identity, voltage base and admissible voltage bounds."""

    id: str = eqx.field(static=True)
    base_kv: float = 1.0
    v_min: float = 0.9
    v_max: float = 1.1


class BusControl(StrictModule):
    """One study's bus equations and initial/reference voltage, not equipment."""

    bus: str = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="pq")
    voltage: float = 1.0
    angle: float = 0.0


class PowerStudy(StrictModule):
    """Explicit controls for every bus; reference selection is never inferred."""

    controls: tuple[BusControl, ...]


class Branch(StrictModule):
    """Pi line/two-winding transformer; complex tap at the from terminal.

    b is total charging susceptance. phase is radians; tap is relative to bus
    voltage bases, not the physical winding ratio. rate is either-end MVA pu.
    """

    id: str = eqx.field(static=True)
    from_bus: str = eqx.field(static=True)
    to_bus: str = eqx.field(static=True)
    r: float
    x: float
    b: float = 0.0
    tap: float = 1.0
    phase: float = 0.0
    rate: float = float("inf")
    in_service: bool = eqx.field(static=True, default=True)


class Shunt(StrictModule):
    """Bus admittance pu: positive g consumes P, positive b supplies Q."""

    bus: str = eqx.field(static=True)
    g: float = 0.0
    b: float = 0.0


class Generator(StrictModule):
    """Generation-positive dispatch; cost is (quadratic, linear, constant) in pu P."""

    id: str = eqx.field(static=True)
    bus: str = eqx.field(static=True)
    p: float = 0.0
    q: float = 0.0
    p_min: float = -float("inf")
    p_max: float = float("inf")
    q_min: float = -float("inf")
    q_max: float = float("inf")
    cost: tuple[float, float, float] = (0.0, 0.0, 0.0)
    in_service: bool = eqx.field(static=True, default=True)


class Load(StrictModule):
    """Constant total three-phase complex power demand in pu; consumption-positive."""

    id: str = eqx.field(static=True)
    bus: str = eqx.field(static=True)
    p: float
    q: float = 0.0
    in_service: bool = eqx.field(static=True, default=True)


class PowerNetwork(StrictModule):
    buses: tuple[Bus, ...]
    branches: tuple[Branch, ...]
    generators: tuple[Generator, ...] = ()
    loads: tuple[Load, ...] = ()
    shunts: tuple[Shunt, ...] = ()
    base_mva: float = 100.0
    frequency: float = 60.0

    @property
    def base(self) -> PowerBase:
        return PowerBase(self.base_mva)


class CompiledNetwork(StrictModule):
    """Physical network and explicit study bound to a native sparse operator."""

    network: PowerNetwork
    study: PowerStudy
    admittance: SparseCoordinateOperator
    branch_admittance: Array
    shunt_admittance: Array
    from_indices: Array
    to_indices: Array
    generator_indices: Array
    load_power: Array
    specified_power: Array
    voltage_setpoints: Array
    initial_voltage: Array
    initial_angles: Array
    control_modes: tuple[str, ...] = eqx.field(static=True)
    references: tuple[int, ...] = eqx.field(static=True)
    islands: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    generators_at_bus: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    @property
    def ybus(self) -> Array:
        """Explicit dense materialization for consumers requiring a matrix."""
        return self.admittance.as_dense()

    def bus_currents(self, voltage: ArrayLike) -> Array:
        """Current from each bus inward into the passive network."""
        return self.admittance.mv(jnp.asarray(voltage))

    def branch_currents(self, voltage: ArrayLike) -> tuple[Array, Array]:
        value = jnp.asarray(voltage)
        vf, vt = value[self.from_indices], value[self.to_indices]
        y = self.branch_admittance
        return y[:, 0] * vf + y[:, 1] * vt, y[:, 2] * vf + y[:, 3] * vt

    def branch_powers(self, voltage: ArrayLike) -> tuple[Array, Array]:
        value = jnp.asarray(voltage)
        current_from, current_to = self.branch_currents(value)
        return (
            value[self.from_indices] * jnp.conj(current_from),
            value[self.to_indices] * jnp.conj(current_to),
        )


def _unique_ids(values, owner):
    ids = tuple(value.id for value in values)
    if any(not isinstance(value, str) or not value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError(f"{owner} IDs must be unique nonempty strings.")


def _finite(values, owner):
    if any(not isfinite(value) for value in values):
        raise ValueError(f"{owner} must be finite.")


def compile_network(network: PowerNetwork, study: PowerStudy) -> CompiledNetwork:
    """Bind physical topology to explicit study controls and compile pi stamps.

    Reference buses are ideal voltage boundaries when they have no generator.
    Such an external source is reported by power flow but OPF requires generators.
    No unreferenced or multiply referenced island is silently repaired.
    """
    if not isinstance(network, PowerNetwork) or not network.buses:
        raise ValueError("A PowerNetwork requires at least one bus.")
    if not isinstance(study, PowerStudy):
        raise TypeError("compile_network requires an explicit PowerStudy.")
    PowerBase(network.base_mva)
    if not isfinite(network.frequency) or network.frequency <= 0:
        raise ValueError("frequency must be finite and positive.")
    for values, owner in (
        (network.buses, "Bus"),
        (network.branches, "Branch"),
        (network.generators, "Generator"),
        (network.loads, "Load"),
    ):
        _unique_ids(values, owner)
    indices = {bus.id: index for index, bus in enumerate(network.buses)}
    n = len(indices)
    if any(not isinstance(control, BusControl) for control in study.controls):
        raise TypeError("PowerStudy controls must be BusControl values.")
    control_ids = tuple(control.bus for control in study.controls)
    if len(set(control_ids)) != len(control_ids) or set(control_ids) != set(indices):
        raise ValueError(
            "PowerStudy requires exactly one explicit BusControl for every physical bus."
        )
    by_bus = {control.bus: control for control in study.controls}
    controls = tuple(by_bus[bus.id] for bus in network.buses)
    for control in controls:
        if control.kind not in ("pq", "pv", "reference"):
            raise ValueError(f"Unsupported study bus kind {control.kind!r}.")
        _finite((control.voltage, control.angle), "Study voltage data")
        if control.voltage <= 0:
            raise ValueError("Study voltage magnitudes must be positive.")
    neighbours = [set() for _ in range(n)]
    from_indices, to_indices, stamps = [], [], []
    for bus in network.buses:
        _finite((bus.base_kv, bus.v_min, bus.v_max), "Bus data")
        if bus.base_kv <= 0 or not 0 < bus.v_min <= bus.v_max:
            raise ValueError("Bus bases and voltage limits must be positive and ordered.")
    for branch in network.branches:
        if branch.from_bus not in indices or branch.to_bus not in indices:
            raise ValueError(f"Unknown terminal bus on branch {branch.id!r}.")
        f, t = indices[branch.from_bus], indices[branch.to_bus]
        if f == t:
            raise ValueError("A branch cannot connect a bus to itself; use a Shunt.")
        _finite((branch.r, branch.x, branch.b, branch.tap, branch.phase), "Branch data")
        if (
            branch.r < 0
            or branch.tap <= 0
            or branch.rate != branch.rate
            or branch.rate <= 0
        ):
            raise ValueError(
                "Branch r must be nonnegative, tap/rate positive and not NaN."
            )
        if branch.in_service and branch.r == 0 and branch.x == 0:
            raise ValueError("Zero-impedance branches require explicit bus reduction.")
        from_indices.append(f)
        to_indices.append(t)
        if branch.in_service:
            neighbours[f].add(t)
            neighbours[t].add(f)
            series = 1 / complex(branch.r, branch.x)
            total = series + 0.5j * branch.b
            tap = branch.tap * jnp.exp(1j * branch.phase)
            stamps.append(
                jnp.asarray(
                    (total / branch.tap**2, -series / jnp.conj(tap), -series / tap, total)
                )
            )
        else:
            stamps.append(jnp.zeros(4, dtype=complex))
    islands, references = [], []
    unseen = set(range(n))
    while unseen:
        queue, reached = [min(unseen)], set()
        while queue:
            bus_index = queue.pop()
            if bus_index not in reached:
                reached.add(bus_index)
                queue.extend(neighbours[bus_index] - reached)
        unseen -= reached
        island = tuple(sorted(reached))
        reference = tuple(i for i in island if controls[i].kind == "reference")
        if len(reference) != 1:
            labels = tuple(network.buses[i].id for i in island)
            raise ValueError(
                f"Island {labels} requires exactly one reference bus; found {len(reference)}."
            )
        islands.append(island)
        references.append(reference[0])
    shunts = [0j] * n
    for shunt in network.shunts:
        if shunt.bus not in indices:
            raise ValueError(f"Unknown shunt bus {shunt.bus!r}.")
        _finite((shunt.g, shunt.b), "Shunt data")
        if shunt.g < 0:
            raise ValueError("Passive shunt conductance must be nonnegative.")
        shunts[indices[shunt.bus]] += complex(shunt.g, shunt.b)
    loads = [0j] * n
    for load in network.loads:
        if load.bus not in indices:
            raise ValueError(f"Unknown load bus {load.bus!r}.")
        _finite((load.p, load.q), "Load data")
        if load.in_service:
            loads[indices[load.bus]] += complex(load.p, load.q)
    specified = [-value for value in loads]
    generators_at_bus = [[] for _ in range(n)]
    generator_indices = []
    setpoints = [control.voltage for control in controls]
    for index, gen in enumerate(network.generators):
        if gen.bus not in indices:
            raise ValueError(f"Unknown generator bus {gen.bus!r}.")
        i = indices[gen.bus]
        generator_indices.append(i)
        _finite((gen.p, gen.q, *gen.cost), "Generator data")
        if len(gen.cost) != 3:
            raise ValueError("Generator cost must have three coefficients.")
        limits = (gen.p_min, gen.p_max, gen.q_min, gen.q_max)
        if any(v != v for v in limits) or gen.p_min > gen.p_max or gen.q_min > gen.q_max:
            raise ValueError("Generator limits must be ordered and not NaN.")
        if (
            gen.p_min == float("inf")
            or gen.q_min == float("inf")
            or gen.p_max == -float("inf")
            or gen.q_max == -float("inf")
        ):
            raise ValueError("Generator limits must admit a finite dispatch.")
        if gen.in_service:
            generators_at_bus[i].append(index)
            specified[i] += complex(gen.p, gen.q)
    for i, generators in enumerate(generators_at_bus):
        if controls[i].kind == "pv" and not generators:
            raise ValueError("A PV bus requires at least one in-service generator.")
    f = jnp.asarray(from_indices, dtype=jnp.int32)
    t = jnp.asarray(to_indices, dtype=jnp.int32)
    branch_y = jnp.stack(stamps) if stamps else jnp.zeros((0, 4), dtype=complex)
    diagonal = jnp.arange(n, dtype=jnp.int32)
    relation = EdgeRelation(
        jnp.concatenate((f, t, f, t, diagonal)),
        jnp.concatenate((f, f, t, t, diagonal)),
        source_size=n,
        target_size=n,
    )
    coefficients = jnp.concatenate((branch_y.T.reshape(-1), jnp.asarray(shunts)))
    space = ArraySpace(
        (n,), dtype=coefficients.dtype, space_id="power-positive-sequence-bus"
    )
    admittance = SparseCoordinateOperator(
        relation, coefficients, source=space, target=space
    )
    angles = jnp.asarray([control.angle for control in controls])
    initial = jnp.asarray(setpoints) * jnp.exp(1j * angles)
    return CompiledNetwork(
        network,
        study,
        admittance,
        branch_y,
        jnp.asarray(shunts),
        f,
        t,
        jnp.asarray(generator_indices, dtype=jnp.int32),
        jnp.asarray(loads),
        jnp.asarray(specified),
        jnp.asarray(setpoints),
        initial,
        angles,
        tuple(control.kind for control in controls),
        tuple(references),
        tuple(islands),
        tuple(tuple(g) for g in generators_at_bus),
    )


__all__ = [
    "PowerBase",
    "Bus",
    "BusControl",
    "PowerStudy",
    "Branch",
    "Shunt",
    "Generator",
    "Load",
    "PowerNetwork",
    "CompiledNetwork",
    "compile_network",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamic single-level traps and prepared elastic nonlocal WKB paths.

Kinetics return physical source/storage/transport ledgers, not DD coordinates.
Bulk trap sources are per volume; surface trap sources are per area and MUST
be scattered using physical interface area, never treated as volume density.
All supplied energies share the explicitly named electronic energy reference.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...units import derived_unit, JOULE, KELVIN, KILOGRAM, METER, SECOND
from ._high_field import (
    _admitted,
    _finite_scalar,
    _temperature_bounds,
    _temperature_valid,
)
from ._quantities import (
    _positive_scalar,
    _si,
    _text,
    ELEMENTARY_CHARGE_SI,
    PER_CUBIC_METER,
)


PER_SQUARE_METER = derived_unit("1/m2", ((METER, -2),))
CAPTURE_COEFFICIENT_UNIT = derived_unit("m3/s", ((METER, 3), (SECOND, -1)))
RATE_UNIT = derived_unit("1/s", ((SECOND, -1),))
HBAR_SI = 6.62607015e-34 / (2 * np.pi)


class TrapStorage(StrictModule):
    """Occupied count, charge C, electronic energy J, per declared measure."""

    occupied_density: Array
    charge_density: Array
    energy_density: Array
    successful: Array


class TrapExchangeEvaluation(StrictModule):
    """Rates per m³ (bulk) or m² (surface); energies positive into each owner.

    electron/hole_energy_source are TOTAL electronic energies, not kinetic
    energies. For a kinetic carrier equation subtract its band energy times
    its number source exactly once. ``trap_energy_source`` is Et*d(Nt*f)/dt.
    Carrier + trap + lattice energies sum to zero, as do their charge rates.
    """

    occupancy_rate: Array
    electron_source: Array
    hole_source: Array
    trap_population_source: Array
    electron_charge_source: Array
    hole_charge_source: Array
    trap_charge_source: Array
    electron_energy_source: Array
    hole_energy_source: Array
    trap_energy_source: Array
    lattice_energy_source: Array
    successful: Array


class TrapStepEvaluation(StrictModule):
    """Exact frozen-reservoir occupancy and time-averaged conserving sources."""

    occupancy: Array
    average_exchange: TrapExchangeEvaluation
    successful: Array


class DynamicTrap(StrictModule):
    """One spin-resolved occupancy f in [0,1], bulk or surface, at fixed level.

    df/dt=(cn*n+ep)*(1-f)-(en+cp*p)*f. Capture coefficients are m³/s
    for adjacent BULK carrier densities; en,ep are s^-1. Nt is m^-3 or
    m^-2 as declared by ``population_kind``. All coefficients are explicit
    constant-property approximations on ``temperature_range``. For thermal
    detailed balance they must be supplied consistently with the material,
    trap degeneracy and level (en=cn*n1, ep=cp*p1); no ni/DOS is invented.

    Empty-state charge number is 0 for an acceptor, +1 for a donor. Filling
    always adds one electron and charge -q. The fixed empty-state energy is
    the datum; filling adds Et in the named electronic reference. Emission
    and capture exchange the explicitly supplied total electron/hole energies
    with the lattice; no defect creation, annealing or reliability claim is
    implied. A moving trap level additionally needs external parametric work,
    not included in this fixed-level closure.
    """

    density: Array
    electron_capture_coefficient: Array
    hole_capture_coefficient: Array
    electron_emission_rate: Array
    hole_emission_rate: Array
    trap_energy: Array
    temperature_range: Array
    empty_charge_number: int = eqx.field(static=True)
    population_kind: str = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        density,
        electron_capture_coefficient,
        hole_capture_coefficient,
        electron_emission_rate,
        hole_emission_rate,
        trap_energy,
        /,
        *,
        empty_charge_number,
        population_kind,
        temperature_range,
        energy_reference,
        provenance,
        density_unit=None,
        capture_unit=CAPTURE_COEFFICIENT_UNIT,
        rate_unit=RATE_UNIT,
        energy_unit=JOULE,
        temperature_unit=KELVIN,
    ):
        if population_kind not in ("bulk", "surface"):
            raise ValueError("population_kind must be 'bulk' or 'surface'.")
        if isinstance(empty_charge_number, bool) or empty_charge_number not in (0, 1):
            raise ValueError("empty_charge_number must be 0 (acceptor) or 1 (donor).")
        density_reference = (
            PER_CUBIC_METER if population_kind == "bulk" else PER_SQUARE_METER
        )
        self.density = _positive_scalar(
            density,
            density_reference if density_unit is None else density_unit,
            density_reference,
            "trap density",
        )
        self.electron_capture_coefficient = _positive_scalar(
            electron_capture_coefficient,
            capture_unit,
            CAPTURE_COEFFICIENT_UNIT,
            "electron capture coefficient",
            nonnegative=True,
        )
        self.hole_capture_coefficient = _positive_scalar(
            hole_capture_coefficient,
            capture_unit,
            CAPTURE_COEFFICIENT_UNIT,
            "hole capture coefficient",
            nonnegative=True,
        )
        self.electron_emission_rate = _positive_scalar(
            electron_emission_rate,
            rate_unit,
            RATE_UNIT,
            "electron emission rate",
            nonnegative=True,
        )
        self.hole_emission_rate = _positive_scalar(
            hole_emission_rate,
            rate_unit,
            RATE_UNIT,
            "hole emission rate",
            nonnegative=True,
        )
        self.trap_energy = _finite_scalar(
            _si(trap_energy, energy_unit, JOULE), "trap energy"
        )
        self.temperature_range = _temperature_bounds(temperature_range, temperature_unit)
        self.empty_charge_number = int(empty_charge_number)
        self.population_kind = population_kind
        self.energy_reference = _text(energy_reference, "trap energy reference")
        self.provenance = _text(provenance, "trap kinetics provenance")

    @staticmethod
    def occupancy_from_logit(logit):
        """Optional solver chart only; stored inventory remains Nt*f."""
        return jax.nn.sigmoid(jnp.asarray(logit))

    def storage(self, occupancy):
        occupancy = jnp.asarray(occupancy)
        valid = jnp.isfinite(occupancy) & (occupancy >= 0) & (occupancy <= 1)
        population = self.density * occupancy
        charge = (
            ELEMENTARY_CHARGE_SI * self.density * (self.empty_charge_number - occupancy)
        )
        return TrapStorage(
            _admitted(population, valid),
            _admitted(charge, valid),
            _admitted(self.trap_energy * population, valid),
            valid,
        )

    def evaluate(
        self,
        occupancy,
        electron_density,
        hole_density,
        temperature,
        electron_exchange_energy,
        hole_exchange_energy,
    ):
        """Evaluate SI densities, T(K), and exchanged TOTAL carrier energies J.

        Electron exchange energy is Ec plus its selected kinetic energy;
        hole exchange energy is -Ev plus its selected kinetic energy. These
        are the energy-resolved capture/emission moment closure, not inferred
        from trap density or an unrelated nondegenerate statistics formula.
        """
        f, n, p, temperature, en, ep = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (
                    occupancy,
                    electron_density,
                    hole_density,
                    temperature,
                    electron_exchange_energy,
                    hole_exchange_energy,
                ),
            )
        )
        electron_capture = self.density * (
            self.electron_capture_coefficient * n * (1 - f)
            - self.electron_emission_rate * f
        )
        hole_capture = self.density * (
            self.hole_capture_coefficient * p * f - self.hole_emission_rate * (1 - f)
        )
        population_rate = electron_capture - hole_capture
        trap_energy = self.trap_energy * population_rate
        electron_energy = -en * electron_capture
        hole_energy = -ep * hole_capture
        lattice_energy = electron_capture * (en - self.trap_energy) + hole_capture * (
            ep + self.trap_energy
        )
        valid = (
            jnp.isfinite(f)
            & (f >= 0)
            & (f <= 1)
            & jnp.isfinite(n)
            & (n >= 0)
            & jnp.isfinite(p)
            & (p >= 0)
            & _temperature_valid(temperature, self.temperature_range)
            & jnp.isfinite(en)
            & jnp.isfinite(ep)
            & jnp.isfinite(population_rate)
            & jnp.isfinite(lattice_energy)
        )
        q = ELEMENTARY_CHARGE_SI
        return TrapExchangeEvaluation(
            _admitted(population_rate / self.density, valid),
            _admitted(-electron_capture, valid),
            _admitted(-hole_capture, valid),
            _admitted(population_rate, valid),
            _admitted(q * electron_capture, valid),
            _admitted(-q * hole_capture, valid),
            _admitted(-q * population_rate, valid),
            _admitted(electron_energy, valid),
            _admitted(hole_energy, valid),
            _admitted(trap_energy, valid),
            _admitted(lattice_energy, valid),
            valid,
        )

    def advance(
        self,
        occupancy,
        electron_density,
        hole_density,
        temperature,
        electron_exchange_energy,
        hole_exchange_energy,
        time_step,
    ):
        """Exact bounded step for FROZEN reservoirs, not an explicit Euler clip.

        The time-averaged capture/emission ledger exactly matches the change
        in occupied trap inventory. For coupled evolution use the continuous
        evaluate() law in the native DAE, or a qualified partitioned exchange.
        """
        f, n, p, dt = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (occupancy, electron_density, hole_density, time_step),
            )
        )
        filling = self.electron_capture_coefficient * n + self.hole_emission_rate
        emptying = self.electron_emission_rate + self.hole_capture_coefficient * p
        rate = filling + emptying
        safe_rate = jnp.where(rate > 0, rate, 1.0)
        equilibrium = jnp.where(rate > 0, filling / safe_rate, f)
        x = rate * dt
        decay_increment = -jnp.expm1(-x)
        next_f = f + (equilibrium - f) * decay_increment
        # 1-phi1(x), evaluated without small-step cancellation; phi1=(1-e^-x)/x.
        safe_x = jnp.where(jnp.abs(x) > 1e-4, x, 1.0)
        average_fraction = jnp.where(
            jnp.abs(x) <= 1e-4,
            x * (0.5 + x * (-1 / 6 + x * (1 / 24 - x / 120))),
            1 - decay_increment / safe_x,
        )
        average_f = f + (equilibrium - f) * average_fraction
        # Admission of the INITIAL state matters even if relaxation would put
        # an out-of-bounds input back into the interval after a long step.
        valid_step = (
            jnp.isfinite(dt)
            & (dt >= 0)
            & jnp.isfinite(x)
            & jnp.isfinite(f)
            & (f >= 0)
            & (f <= 1)
        )
        exchange = self.evaluate(
            _admitted(average_f, valid_step),
            n,
            p,
            temperature,
            electron_exchange_energy,
            hole_exchange_energy,
        )
        valid = valid_step & exchange.successful
        return TrapStepEvaluation(_admitted(next_f, valid), exchange, valid)


class WKBTransmissionEvaluation(StrictModule):
    """Leading WKB probability, dimensionless action, forbidden path length m."""

    transmission: Array
    log_transmission: Array
    action: Array
    forbidden_length: Array
    successful: Array


class WKBBarrierPath(StrictModule):
    """Prepared 1D scalar effective-mass barrier with exact segment actions.

    Potential energy is piecewise linear on increasing arc length; mass is
    positive and constant on each segment. The forbidden sqrt(U-E) integral
    includes every true linear-segment turning point analytically. Leading
    transmission exp(-2*S/hbar) omits interface/reflection prefactors and
    resonant interference: a path must have at most ONE connected forbidden
    interval. Over-barrier propagation has T=1 (no reflection in this limit).
    ``minimum_action`` explicitly bounds the user's admitted WKB regime;
    under-barrier evaluations below it are unsuccessful, not silently fitted.
    No effective mass, complex band or barrier data are invented.
    """

    positions: Array
    barrier_energies: Array
    effective_masses: Array
    minimum_action: Array
    energy_reference: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        positions,
        barrier_energies,
        effective_masses,
        /,
        *,
        minimum_action,
        energy_reference,
        provenance,
        length_unit=METER,
        energy_unit=JOULE,
        mass_unit=KILOGRAM,
    ):
        x = _si(positions, length_unit, METER)
        u = _si(barrier_energies, energy_unit, JOULE)
        mass = _si(effective_masses, mass_unit, KILOGRAM)
        xh, uh, mh = np.asarray(x), np.asarray(u), np.asarray(mass)
        if (
            xh.ndim != 1
            or xh.size < 2
            or not np.all(np.isfinite(xh))
            or np.any(np.diff(xh) <= 0)
        ):
            raise ValueError(
                "WKB arc-length nodes must be finite and strictly increasing."
            )
        if uh.shape != xh.shape or not np.all(np.isfinite(uh)):
            raise ValueError(
                "WKB barrier energies must have one finite value per path node."
            )
        if mh.shape != (xh.size - 1,) or not np.all(np.isfinite(mh)) or np.any(mh <= 0):
            raise ValueError(
                "WKB masses must have one positive finite SI mass per segment."
            )
        self.positions, self.barrier_energies, self.effective_masses = (
            jnp.asarray(x),
            jnp.asarray(u),
            jnp.asarray(mass),
        )
        self.minimum_action = _finite_scalar(minimum_action, "minimum WKB action")
        if float(self.minimum_action) < 0:
            raise ValueError("minimum WKB action must be nonnegative.")
        self.energy_reference = _text(energy_reference, "WKB energy reference")
        self.provenance = _text(provenance, "WKB path provenance")

    def evaluate(self, energy):
        """One energy J or a batch; geometry and mass are fixed during AD."""
        energy = jnp.asarray(energy)
        excess = self.barrier_energies - energy[..., None]
        a, b = excess[..., :-1], excess[..., 1:]
        ap, bp = jnp.maximum(a, 0.0), jnp.maximum(b, 0.0)
        ra = jnp.where(a > 0, jnp.sqrt(jnp.where(a > 0, a, 1.0)), 0.0)
        rb = jnp.where(b > 0, jnp.sqrt(jnp.where(b > 0, b, 1.0)), 0.0)
        both_positive = (a > 0) & (b > 0)
        one_positive = (a > 0) | (b > 0)
        denominator = jnp.where(one_positive, ra + rb, 1.0)
        # Stable divided difference for two positive endpoints, including a=b.
        both_integral = (2 / 3) * (ap + ra * rb + bp) / denominator
        slope = jnp.abs(a - b)
        safe_slope = jnp.where(slope > 0, slope, 1.0)
        crossing_integral = (2 / 3) * (ap * ra + bp * rb) / safe_slope
        average_root = jnp.where(
            both_positive, both_integral, jnp.where(one_positive, crossing_integral, 0.0)
        )
        lengths = jnp.diff(self.positions)
        action = (
            jnp.sum(lengths * jnp.sqrt(2 * self.effective_masses) * average_root, axis=-1)
            / HBAR_SI
        )
        forbidden_fraction = jnp.where(
            both_positive, 1.0, jnp.where(one_positive, (ap + bp) / safe_slope, 0.0)
        )
        forbidden_length = jnp.sum(lengths * forbidden_fraction, axis=-1)
        # More than one positive-node run means separated barriers or a
        # zero-width allowed turning point, outside this nonresonant model.
        positive = excess > 0
        starts = positive[..., 0].astype(jnp.int32) + jnp.sum(
            positive[..., 1:] & ~positive[..., :-1], axis=-1
        )
        valid = (
            jnp.isfinite(energy)
            & jnp.isfinite(action)
            & (starts <= 1)
            & ((action >= self.minimum_action) | (forbidden_length == 0))
        )
        log_transmission = -2 * action
        return WKBTransmissionEvaluation(
            _admitted(jnp.exp(log_transmission), valid),
            _admitted(log_transmission, valid),
            _admitted(action, valid),
            _admitted(forbidden_length, valid),
            valid,
        )


class NonlocalTunnelingEvaluation(StrictModule):
    """Extensive endpoint sources (#/s, A, W) and edge transfers (A, W).

    Path edges point from valence source to conduction destination. Their
    conventional current is -q*pair_rate. Scattering that current's inward
    incidence yields ``charge_source`` at EVERY node, not just global zero.
    This is a transport decomposition of endpoint generation: do not add
    the same incidence a second time as an independent carrier source.

    Total electronic energy source is the incidence of ``path_energy_power``.
    Kinetic endpoint sources plus band_storage_source give that same source.
    Elastic transfer has no lattice heat. A negative rate is pair annihilation.
    """

    pair_rate: Array
    electron_source: Array
    hole_source: Array
    charge_source: Array
    electron_kinetic_energy_source: Array
    hole_kinetic_energy_source: Array
    band_storage_source: Array
    total_energy_source: Array
    path_charge_current: Array
    path_energy_power: Array
    lattice_energy_source: Array
    transmission: Array
    successful: Array


class NonlocalTunnelingPath(StrictModule):
    """One elastic spectral valence-to-conduction channel with a real barrier.

    ``attempt_rate`` in s^-1 includes the caller's incident mode/energy-bin
    state counting. Net transfer is attempt*T*(f_source-f_destination), the
    algebraically factored Pauli forward-minus-reverse rate. Endpoint electron
    energies are the SAME energy E: destination electron carries E and source
    hole carries -E. Admit only the overlap Ec_destination <= E <= Ev_source.
    This is an explicitly bounded scalar WKB barrier approximation; it is NOT
    a multiband complex-band prediction or an invented local leakage law.
    Independent spectral channels must be integrated by their prepared native
    integration owner, not counted again as a local pair generator.
    """

    barrier: WKBBarrierPath
    path_nodes: Array
    node_count: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(self, barrier, node_count, path_nodes, /, *, provenance):
        if not isinstance(barrier, WKBBarrierPath):
            raise TypeError("barrier must be a WKBBarrierPath.")
        if (
            not isinstance(node_count, int)
            or isinstance(node_count, bool)
            or node_count < 2
        ):
            raise ValueError("node_count must be an integer at least two.")
        nodes = np.asarray(path_nodes)
        if (
            nodes.ndim != 1
            or nodes.size != barrier.positions.size
            or not np.issubdtype(nodes.dtype, np.integer)
        ):
            raise ValueError(
                "path_nodes must provide one integer device node per barrier node."
            )
        if (
            np.any(nodes < 0)
            or np.any(nodes >= node_count)
            or np.unique(nodes).size != nodes.size
        ):
            raise ValueError(
                "path_nodes must be distinct in-range nodes ordered source to destination."
            )
        self.barrier, self.node_count, self.path_nodes = (
            barrier,
            node_count,
            jnp.asarray(nodes),
        )
        self.provenance = _text(provenance, "nonlocal tunneling provenance")

    def evaluate(
        self,
        energy,
        source_occupation,
        destination_occupation,
        attempt_rate,
        source_valence_edge,
        destination_conduction_edge,
    ):
        """One scalar spectral channel, all energies J in barrier's reference."""
        energy, fs, fd, attempt, ev, ec = map(
            jnp.asarray,
            (
                energy,
                source_occupation,
                destination_occupation,
                attempt_rate,
                source_valence_edge,
                destination_conduction_edge,
            ),
        )
        if any(value.shape != () for value in (energy, fs, fd, attempt, ev, ec)):
            raise ValueError(
                "NonlocalTunnelingPath.evaluate accepts one scalar spectral channel; batch with vmap."
            )
        transmission = self.barrier.evaluate(energy)
        rate = attempt * transmission.transmission * (fs - fd)
        valid = (
            transmission.successful
            & jnp.isfinite(fs)
            & (fs >= 0)
            & (fs <= 1)
            & jnp.isfinite(fd)
            & (fd >= 0)
            & (fd <= 1)
            & jnp.isfinite(attempt)
            & (attempt >= 0)
            & jnp.isfinite(ev)
            & jnp.isfinite(ec)
            & (ec <= energy)
            & (energy <= ev)
            & jnp.isfinite(rate)
        )
        rate = _admitted(rate, valid)
        zeros = jnp.zeros(self.node_count, dtype=rate.dtype)
        source, destination = self.path_nodes[0], self.path_nodes[-1]
        electron_source = zeros.at[destination].set(rate)
        hole_source = zeros.at[source].set(rate)
        charge = ELEMENTARY_CHARGE_SI * (hole_source - electron_source)
        electron_kinetic = electron_source * (energy - ec)
        hole_kinetic = hole_source * (ev - energy)
        band_storage = ec * electron_source - ev * hole_source
        total_energy = energy * (electron_source - hole_source)
        path_current = jnp.full((self.path_nodes.size - 1,), -ELEMENTARY_CHARGE_SI * rate)
        path_power = jnp.full((self.path_nodes.size - 1,), energy * rate)
        return NonlocalTunnelingEvaluation(
            rate,
            electron_source,
            hole_source,
            charge,
            electron_kinetic,
            hole_kinetic,
            band_storage,
            total_energy,
            path_current,
            path_power,
            _admitted(zeros, valid),
            transmission.transmission,
            valid,
        )


__all__ = [
    "DynamicTrap",
    "NonlocalTunnelingEvaluation",
    "NonlocalTunnelingPath",
    "TrapExchangeEvaluation",
    "TrapStepEvaluation",
    "TrapStorage",
    "WKBBarrierPath",
    "WKBTransmissionEvaluation",
]

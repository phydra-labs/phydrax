# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Conservative classical assembly; no independent solver or time integrator."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._strict import StrictModule
from ...discretization import scharfetter_gummel_flux
from ._interfaces import interface_electrostatics, ThermionicInterface
from ._kinetics import DynamicTrap, NonlocalTunnelingPath
from ._materials import SemiconductorMaterial
from ._quantities import BOLTZMANN_CONSTANT_SI as K, ELEMENTARY_CHARGE_SI as Q
from ._thermal import ThermalBoundaryExchange


class MaterialInterface(StrictModule):
    """An ordered support edge cut at an explicit physical position.

    Side distances sum to the edge length. ``None`` laws mean transparent
    electrochemical traces; thermionic laws require four algebraic Fermi traces.
    Sheet charge is integrated C and a dipole jump is V, right minus left.
    """

    edge: int = eqx.field(static=True)
    fraction: float = eqx.field(static=True)
    sheet_charge: jax.Array
    potential_jump: jax.Array
    electron_law: ThermionicInterface | None
    hole_law: ThermionicInterface | None

    def __init__(
        self,
        edge,
        *,
        fraction,
        sheet_charge=0.0,
        potential_jump=0.0,
        electron_law=None,
        hole_law=None,
    ):
        if not isinstance(edge, int) or edge < 0 or not 0 < fraction < 1:
            raise ValueError(
                "An interface needs an edge and a strictly interior fraction."
            )
        if (electron_law is None) != (hole_law is None):
            raise ValueError(
                "Specify both reciprocal carrier interface laws, or neither."
            )
        if any(
            law is not None and not isinstance(law, ThermionicInterface)
            for law in (electron_law, hole_law)
        ):
            raise TypeError("Interface emission requires ThermionicInterface laws.")
        if not np.all(np.isfinite([sheet_charge, potential_jump])):
            raise ValueError("Interface charge and dipole jump must be finite SI values.")
        self.edge, self.fraction = edge, float(fraction)
        self.sheet_charge, self.potential_jump = (
            jnp.asarray(sheet_charge),
            jnp.asarray(potential_jump),
        )
        self.electron_law, self.hole_law = electron_law, hole_law


class TrapBinding(StrictModule):
    """Bulk node or material-interface surface trap, never a fictitious volume.

    A surface trap exchanges carriers with the declared left material; its
    charge enters the exact sheet displacement jump. Its energy reference is
    a material reference (electrostatic field energy is owned separately).
    """

    trap: DynamicTrap
    location: int = eqx.field(static=True)
    initial_occupancy: jax.Array

    def __init__(self, trap, location, *, initial_occupancy=0.5):
        if (
            not isinstance(trap, DynamicTrap)
            or not isinstance(location, int)
            or location < 0
        ):
            raise ValueError(
                "Trap binding requires a DynamicTrap and a nonnegative location."
            )
        if not np.isfinite(initial_occupancy) or not 0 < initial_occupancy < 1:
            raise ValueError(
                "Initial logit-chart occupancy must lie strictly between zero and one."
            )
        self.trap, self.location = trap, location
        self.initial_occupancy = jnp.asarray(initial_occupancy)


class ThermalPort(StrictModule):
    name: str = eqx.field(static=True)
    node: int = eqx.field(static=True)
    exchange: ThermalBoundaryExchange

    def __init__(self, name, node, exchange):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(node, int)
            or node < 0
            or not isinstance(exchange, ThermalBoundaryExchange)
        ):
            raise ValueError(
                "ThermalPort requires a name, node, and ThermalBoundaryExchange."
            )
        self.name, self.node, self.exchange = name, node, exchange


class TunnelingChannel(StrictModule):
    """Prepared elastic spectral channel with explicit incident state counting."""

    path: NonlocalTunnelingPath
    energy: jax.Array
    attempt_rate: jax.Array

    def __init__(self, path, energy, attempt_rate):
        energy_host, attempt_host = np.asarray(energy), np.asarray(attempt_rate)
        if (
            not isinstance(path, NonlocalTunnelingPath)
            or energy_host.shape != ()
            or attempt_host.shape != ()
            or not np.isfinite(energy_host)
            or not np.isfinite(attempt_host)
            or attempt_host < 0
        ):
            raise ValueError(
                "A tunneling channel needs a physical path, scalar finite energy "
                "and scalar nonnegative attempt rate."
            )
        self.path = path
        self.energy = jnp.asarray(energy)
        self.attempt_rate = jnp.asarray(attempt_rate)


def _harmonic(a, b):
    return 2 * a * b / jnp.where(a + b > 0, a + b, 1)


def _flux(
    nl,
    nr,
    chemical_left,
    chemical_right,
    temperature_left,
    temperature_right,
    einstein_left,
    einstein_right,
    mobility,
    metric,
    transported_entropy=0.0,
):
    """Exponential fitting with thermodynamic chemical/thermoelectric affinity.

    Arithmetic D/mu is a positive degenerate diffusion enhancement. The explicit
    entropy per particle yields Kelvin thermopower; zero selects isothermal SG.
    """
    ratio = 0.5 * (einstein_left + einstein_right)
    affinity = (
        chemical_right
        - chemical_left
        + transported_entropy * (temperature_right - temperature_left)
    ) / (Q * ratio)
    drift = affinity - (jnp.log(nr) - jnp.log(nl))
    diffusion = mobility * ratio
    forward = scharfetter_gummel_flux(nl, 0.0, drift, diffusion)
    reverse = -scharfetter_gummel_flux(0.0, nr, drift, diffusion)
    value = jnp.where(
        affinity >= 0,
        reverse * jnp.expm1(jnp.where(affinity >= 0, -affinity, 0)),
        -forward * jnp.expm1(jnp.where(affinity < 0, affinity, 0)),
    )
    return metric * value


class ClassicalPhysics(StrictModule):
    __strict_abstract__ = True

    @abstractmethod
    def _coordinates(self, coordinates):
        raise NotImplementedError

    @property
    def layout(self):
        return self.plan.layout

    def field(self, u, name):
        return self.layout.field(u, name)

    def temperatures(self, u):
        plan = self.plan
        lattice = jnp.broadcast_to(plan.temperature, (self.num_nodes,))
        if plan.electrothermal:
            lattice = plan.temperature * jnp.exp(self.field(u, "lattice_energy"))
        electron, hole = lattice, lattice
        if plan.carrier_energy:
            electron = plan.temperature * jnp.exp(self.field(u, "electron_energy"))
            hole = plan.temperature * jnp.exp(self.field(u, "hole_energy"))
        return lattice, electron, hole

    def _state_valid(self, u):
        plan = self.plan
        psi = plan.thermal_voltage * self.field(u, "potential")
        fn = K * plan.temperature * self.field(u, "electron")
        fp = K * plan.temperature * self.field(u, "hole")
        tl, tn, tp = self.temperatures(u)
        valid = jnp.all(jnp.isfinite(u))
        for index, model in enumerate(plan.material_models):
            nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
            if plan.electrothermal:
                capacity_bounds = model.lattice_heat_capacity.temperature_range
                valid &= jnp.all(
                    (tl[nodes] >= capacity_bounds[0]) & (tl[nodes] <= capacity_bounds[1])
                )
            if not isinstance(model, SemiconductorMaterial):
                continue
            valid &= jnp.all(
                (tl[nodes] >= model.temperature_range[0])
                & (tl[nodes] <= model.temperature_range[1])
            )
            if model.thermodynamics is not None:
                bands = model.thermodynamics
                lower, upper = bands.temperature_range
                valid &= jnp.all(
                    (tn[nodes] >= lower)
                    & (tn[nodes] <= upper)
                    & (tp[nodes] >= lower)
                    & (tp[nodes] <= upper)
                )
                # Bounded values are used only to form an admission predicate;
                # the physical residual receives the original temperatures.
                safe_tn = jnp.clip(tn[nodes], lower, upper)
                safe_tp = jnp.clip(tp[nodes], lower, upper)
                ec, _ = bands.band_edges(psi[nodes], safe_tn)
                _, ev = bands.band_edges(psi[nodes], safe_tp)
                eta_n = (fn[nodes] - ec) / (K * safe_tn)
                eta_p = (ev - fp[nodes]) / (K * safe_tp)
                upper_eta = 80 if bands.statistics == "fermi-dirac" else 600
                valid &= jnp.all(
                    (eta_n <= upper_eta)
                    & (eta_p <= upper_eta)
                    & (eta_n > -650)
                    & (eta_p > -650)
                )
            else:
                eta_n = (
                    self.field(u, "potential")[nodes] + self.field(u, "electron")[nodes]
                )
                eta_p = -self.field(u, "potential")[nodes] - self.field(u, "hole")[nodes]
                valid &= jnp.all((jnp.abs(eta_n) < 600) & (jnp.abs(eta_p) < 600))
        return valid

    def densities(self, u):
        u = self._coordinates(u)
        plan = self.plan
        psi = plan.thermal_voltage * self.field(u, "potential")
        fn, fp = (
            K * plan.temperature * self.field(u, "electron"),
            K * plan.temperature * self.field(u, "hole"),
        )
        _, tn, tp = self.temperatures(u)
        n, p = jnp.zeros_like(psi), jnp.zeros_like(psi)
        for index, model in enumerate(plan.material_models):
            if not isinstance(model, SemiconductorMaterial):
                continue
            nodes = jnp.asarray(plan.material_nodes[index])
            if model.thermodynamics is None:
                ni = model.intrinsic_density_at(tn[nodes])
                nn = ni * jnp.exp(
                    self.field(u, "potential")[nodes] + self.field(u, "electron")[nodes]
                )
                pp = ni * jnp.exp(
                    -self.field(u, "potential")[nodes] - self.field(u, "hole")[nodes]
                )
            else:
                nn = model.thermodynamics.electron_density(
                    psi[nodes], fn[nodes], tn[nodes]
                )
                pp = model.thermodynamics.hole_density(psi[nodes], fp[nodes], tp[nodes])
            n, p = n.at[nodes].set(nn), p.at[nodes].set(pp)
        return n, p

    def coordinates_from_densities(self, u, n, p):
        plan = self.plan
        psi = plan.thermal_voltage * self.field(u, "potential")
        _, tn, tp = self.temperatures(u)
        fn, fp = self.field(u, "electron"), self.field(u, "hole")
        for index, model in enumerate(plan.material_models):
            if not isinstance(model, SemiconductorMaterial):
                continue
            nodes = jnp.asarray(plan.material_nodes[index])
            if model.thermodynamics is None:
                ni = model.intrinsic_density_at(tn[nodes])
                en = jnp.log(n[nodes] / ni) - self.field(u, "potential")[nodes]
                ep = -jnp.log(p[nodes] / ni) - self.field(u, "potential")[nodes]
            else:
                en = model.thermodynamics.electron_fermi_energy(
                    psi[nodes], n[nodes], tn[nodes]
                ) / (K * plan.temperature)
                ep = model.thermodynamics.hole_fermi_energy(
                    psi[nodes], p[nodes], tp[nodes]
                ) / (K * plan.temperature)
            fn, fp = fn.at[nodes].set(en), fp.at[nodes].set(ep)
        return self.layout.set(self.layout.set(u, "electron", fn), "hole", fp)

    def _thermodynamic_fields(self, u, n, p):
        plan = self.plan
        psi = plan.thermal_voltage * self.field(u, "potential")
        fn, fp = (
            K * plan.temperature * self.field(u, "electron"),
            K * plan.temperature * self.field(u, "hole"),
        )
        tl, tn, tp = self.temperatures(u)
        en, ep = K * tn / Q, K * tp / Q
        ec, ev = -Q * psi, -Q * psi
        un, up = 1.5 * K * tn * n, 1.5 * K * tp * p
        bn, bp = jnp.zeros_like(n), jnp.zeros_like(p)
        for index, model in enumerate(plan.material_models):
            if not isinstance(model, SemiconductorMaterial):
                continue
            nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
            if model.thermodynamics is None:
                conduction_material = -model.electron_affinity
                valence_material = conduction_material - model.band_gap
                ec = ec.at[nodes].set(conduction_material - Q * psi[nodes])
                ev = ev.at[nodes].set(valence_material - Q * psi[nodes])
                bn = bn.at[nodes].set(conduction_material)
                bp = bp.at[nodes].set(-valence_material)
                continue
            bands = model.thermodynamics
            cn, vp = bands.band_edges(psi[nodes], tl[nodes])
            cm, vm = bands.band_edges(0.0, tl[nodes])
            dc, dv = bands.material_band_temperature_derivatives(tl[nodes])
            en = en.at[nodes].set(
                n[nodes]
                / (Q * bands.electron_compressibility(psi[nodes], fn[nodes], tn[nodes]))
            )
            ep = ep.at[nodes].set(
                p[nodes]
                / (Q * bands.hole_compressibility(psi[nodes], fp[nodes], tp[nodes]))
            )
            ec, ev = ec.at[nodes].set(cn), ev.at[nodes].set(vp)
            un = un.at[nodes].set(bands.electron_energy_density(n[nodes], tn[nodes]))
            up = up.at[nodes].set(bands.hole_energy_density(p[nodes], tp[nodes]))
            bn, bp = (
                bn.at[nodes].set(cm - tl[nodes] * dc),
                bp.at[nodes].set(-vm + tl[nodes] * dv),
            )
        return en, ep, ec, ev, un, up, bn, bp

    def ionized_dopants(self, u):
        """Return local-equilibrium ionized donors/acceptors for steady models."""
        plan = self.plan
        donors, acceptors = plan.donor_density, plan.acceptor_density
        psi = plan.thermal_voltage * self.field(u, "potential")
        electron_fermi = K * plan.temperature * self.field(u, "electron")
        hole_fermi = K * plan.temperature * self.field(u, "hole")
        lattice, _, _ = self.temperatures(u)
        ionized_donors, ionized_acceptors = donors, acceptors
        for index, model in enumerate(plan.material_models):
            if (
                not isinstance(model, SemiconductorMaterial)
                or model.incomplete_ionization is None
            ):
                continue
            nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
            nd, na = model.incomplete_ionization.ionized_densities_split(
                model.thermodynamics,
                psi[nodes],
                electron_fermi[nodes],
                hole_fermi[nodes],
                lattice[nodes],
                donors[nodes],
                acceptors[nodes],
            )
            ionized_donors = ionized_donors.at[nodes].set(nd)
            ionized_acceptors = ionized_acceptors.at[nodes].set(na)
        return ionized_donors, ionized_acceptors

    def _trap_geometry(self, binding):
        if binding.trap.population_kind == "bulk":
            return binding.location, self.plan.support.volumes[binding.location]
        interface = self.plan.interfaces[binding.location]
        edge = interface.edge
        return int(
            self.plan.interface_nodes[binding.location][0]
        ), self.plan.support.transmissibility[edge] * self.plan.edge_lengths[edge]

    def sheet_charge(self, u):
        charges = jnp.asarray(
            [interface.sheet_charge for interface in self.plan.interfaces]
        )
        for index, binding in enumerate(self.plan.traps):
            if binding.trap.population_kind == "surface":
                _, area = self._trap_geometry(binding)
                occupancy = jax.nn.sigmoid(self.field(u, f"trap_{index}")[0])
                charges = charges.at[binding.location].add(
                    area * binding.trap.storage(occupancy).charge_density
                )
        return charges

    def interface_states(self, u):
        plan, support = self.plan, self.plan.support
        psi = plan.thermal_voltage * self.field(u, "potential")
        charges = self.sheet_charge(u)
        return tuple(
            interface_electrostatics(
                psi[left],
                psi[right],
                plan.permittivity[left],
                plan.permittivity[right],
                interface.fraction * plan.edge_lengths[interface.edge],
                (1 - interface.fraction) * plan.edge_lengths[interface.edge],
                support.transmissibility[interface.edge]
                * plan.edge_lengths[interface.edge],
                sheet_charge=charges[index],
                potential_jump=interface.potential_jump,
            )
            for index, (interface, (left, right)) in enumerate(
                zip(plan.interfaces, plan.interface_nodes, strict=True)
            )
        )

    def charge_density(self, u):
        n, p = self.densities(u)
        donors, acceptors = self.ionized_dopants(u)
        charge = Q * (p - n + donors - acceptors)
        for index, binding in enumerate(self.plan.traps):
            if binding.trap.population_kind == "bulk":
                occupancy = jax.nn.sigmoid(self.field(u, f"trap_{index}")[0])
                charge = charge.at[binding.location].add(
                    binding.trap.storage(occupancy).charge_density
                )
        return charge

    def poisson_reaction(self, u):
        plan, support = self.plan, self.plan.support
        psi = plan.thermal_voltage * self.field(u, "potential")
        flux = self.edge_capacitance * (psi[support.tail] - psi[support.head])
        result = -self._incoming(flux) - support.volumes * self.charge_density(u)
        for interface, (left, right), state in zip(
            plan.interfaces, plan.interface_nodes, self.interface_states(u), strict=True
        ):
            result = result.at[left].add(state.displacement_left - flux[interface.edge])
            result = result.at[right].add(
                -state.displacement_right + flux[interface.edge]
            )
        return result

    def recombination(self, u):
        n, p = self.densities(u)
        plan = self.plan
        lattice, _, _ = self.temperatures(u)
        result = jnp.zeros_like(n)
        for index, model in enumerate(plan.material_models):
            if not isinstance(model, SemiconductorMaterial):
                continue
            nodes = jnp.asarray(plan.material_nodes[index])
            # A local detailed-balanced SRH activity closure. The explicit DOS
            # branch uses the actual product and chemical affinity, not ni^2.
            affinity = self.field(u, "electron")[nodes] - self.field(u, "hole")[nodes]
            affinity = affinity * plan.temperature / lattice[nodes]
            product = n[nodes] * p[nodes]
            numerator = product * (-jnp.expm1(-affinity))
            ni = model.intrinsic_density_at(lattice[nodes])
            rate = numerator / (
                plan.hole_lifetime[nodes] * (n[nodes] + ni)
                + plan.electron_lifetime[nodes] * (p[nodes] + ni)
            )
            result = result.at[nodes].set(rate)
        return result

    def _incoming(self, flux):
        support = self.plan.support
        return (
            jnp.zeros((self.num_nodes,), dtype=flux.dtype)
            .at[support.tail]
            .add(-flux)
            .at[support.head]
            .add(flux)
        )

    def _terminal_sum(self, values, mask):
        return (
            jnp.zeros((self.num_terminals,), dtype=values.dtype)
            .at[jnp.maximum(self.plan.terminal_index, 0)]
            .add(jnp.where(mask, values, 0))
        )

    def terminal_charge(self, u):
        return self._terminal_sum(self.poisson_reaction(u), self.plan.potential_mask)

    def _transport(self, u):
        plan, support = self.plan, self.plan.support
        left, right = support.tail, support.head
        n, p = self.densities(u)
        en, ep, ec, ev, un, up, bn, bp = self._thermodynamic_fields(u, n, p)
        tl, tn, tp = self.temperatures(u)
        psi = plan.thermal_voltage * self.field(u, "potential")
        fn, fp = (
            K * plan.temperature * self.field(u, "electron"),
            K * plan.temperature * self.field(u, "hole"),
        )
        active = plan.semiconductor_mask[left] & plan.semiconductor_mask[right]
        safe_n = jnp.where(plan.semiconductor_mask, n, 1)
        safe_p = jnp.where(plan.semiconductor_mask, p, 1)
        if any(
            isinstance(model, SemiconductorMaterial)
            and model.incomplete_ionization is not None
            for model in plan.material_models
        ):
            ionized_donors, ionized_acceptors = self.ionized_dopants(u)
            mn, mp = plan._mobilities(ionized_donors + ionized_acceptors)
        else:
            mn, mp = plan.electron_mobility, plan.hole_mobility
        face_n = _harmonic(mn[left], mn[right])
        face_p = _harmonic(mp[left], mp[right])
        field = (psi[left] - psi[right]) / plan.edge_lengths
        for index, model in enumerate(plan.material_models):
            if not isinstance(model, SemiconductorMaterial):
                continue
            edges = jnp.asarray(plan.material_edges[index], dtype=jnp.int32)
            if model.electron_saturation is not None:
                law = model.electron_saturation
                force = law.driving_force.from_fields(
                    field[edges],
                    (fn[right[edges]] - fn[left[edges]]) / plan.edge_lengths[edges],
                )
                face_n = face_n.at[edges].set(
                    law.evaluate(
                        face_n[edges], force, 0.5 * (tl[left[edges]] + tl[right[edges]])
                    ).mobility
                )
            if model.hole_saturation is not None:
                law = model.hole_saturation
                force = law.driving_force.from_fields(
                    field[edges],
                    (fp[right[edges]] - fp[left[edges]]) / plan.edge_lengths[edges],
                )
                face_p = face_p.at[edges].set(
                    law.evaluate(
                        face_p[edges], force, 0.5 * (tl[left[edges]] + tl[right[edges]])
                    ).mobility
                )
        entropy_n = (5 * un / (3 * safe_n) + ec - fn) / tn
        entropy_p = (5 * up / (3 * safe_p) - ev + fp) / tp
        sn = 0.5 * (entropy_n[left] + entropy_n[right]) if plan.electrothermal else 0.0
        sp = 0.5 * (entropy_p[left] + entropy_p[right]) if plan.electrothermal else 0.0
        gn = _flux(
            safe_n[left],
            safe_n[right],
            fn[left],
            fn[right],
            tn[left],
            tn[right],
            en[left],
            en[right],
            face_n,
            support.transmissibility,
            sn,
        )
        gp = _flux(
            safe_p[left],
            safe_p[right],
            -fp[left],
            -fp[right],
            tp[left],
            tp[right],
            ep[left],
            ep[right],
            face_p,
            support.transmissibility,
            sp,
        )
        gn, gp = jnp.where(active, gn, 0), jnp.where(active, gp, 0)
        nl, nr, pl, pr = gn, gn, gp, gp
        traces = {}
        interface_exchanges = []
        for index, (interface, (a, b), electrostatic) in enumerate(
            zip(
                plan.interfaces,
                plan.interface_nodes,
                self.interface_states(u),
                strict=True,
            )
        ):
            if interface.electron_law is None:
                continue
            edge = interface.edge
            area = support.transmissibility[edge] * plan.edge_lengths[edge]
            dl, dr = (
                interface.fraction * plan.edge_lengths[edge],
                (1 - interface.fraction) * plan.edge_lengths[edge],
            )
            bands_l = plan.material_models[
                plan.interface_materials[index][0]
            ].thermodynamics
            bands_r = plan.material_models[
                plan.interface_materials[index][1]
            ].thermodynamics
            fns = tuple(
                K * plan.temperature * self.field(u, f"interface_{index}_{name}")[0]
                for name in ("electron_left", "electron_right", "hole_left", "hole_right")
            )
            fnl, fnr, fpl, fpr = fns
            n_l = bands_l.electron_density(electrostatic.potential_left, fnl, tn[a])
            n_r = bands_r.electron_density(electrostatic.potential_right, fnr, tn[b])
            p_l = bands_l.hole_density(electrostatic.potential_left, fpl, tp[a])
            p_r = bands_r.hole_density(electrostatic.potential_right, fpr, tp[b])
            ecl, evl = bands_l.band_edges(electrostatic.potential_left, tl[a])
            ecr, evr = bands_r.band_edges(electrostatic.potential_right, tl[b])
            exchange_n = interface.electron_law.evaluate(
                fnl, fnr, tn[a], tn[b], jnp.maximum(ecl, ecr), area
            )
            exchange_p = interface.hole_law.evaluate(
                -fpl, -fpr, tp[a], tp[b], jnp.maximum(-evl, -evr), area
            )
            gnl = _flux(
                n[a], n_l, fn[a], fnl, tn[a], tn[a], en[a], en[a], mn[a], area / dl
            )
            gnr = _flux(
                n_r, n[b], fnr, fn[b], tn[b], tn[b], en[b], en[b], mn[b], area / dr
            )
            gpl = _flux(
                p[a], p_l, -fp[a], -fpl, tp[a], tp[a], ep[a], ep[a], mp[a], area / dl
            )
            gpr = _flux(
                p_r, p[b], -fpr, -fp[b], tp[b], tp[b], ep[b], ep[b], mp[b], area / dr
            )
            valid = exchange_n.successful & exchange_p.successful
            interface_exchanges.append((index, exchange_n, exchange_p))
            for name, rate in zip(
                ("electron_left", "electron_right", "hole_left", "hole_right"),
                (
                    gnl - exchange_n.number_flux,
                    gnr - exchange_n.number_flux,
                    gpl - exchange_p.number_flux,
                    gpr - exchange_p.number_flux,
                ),
                strict=True,
            ):
                traces[f"interface_{index}_{name}"] = jnp.where(
                    valid,
                    rate / jnp.sqrt(self.count_scale[a] * self.count_scale[b]),
                    jnp.nan,
                )
            nl, nr = nl.at[edge].set(gnl), nr.at[edge].set(gnr)
            pl, pr = pl.at[edge].set(gpl), pr.at[edge].set(gpr)
            gn, gp = (
                gn.at[edge].set(exchange_n.number_flux),
                gp.at[edge].set(exchange_p.number_flux),
            )
        return (
            gn,
            gp,
            nl,
            nr,
            pl,
            pr,
            traces,
            tuple(interface_exchanges),
            (en, ep, ec, ev, un, up, bn, bp),
        )

    def edge_fluxes(self, u):
        result = self._transport(u)
        return result[0], result[1]

    def _transport_sources(self, left_flux, right_flux):
        support = self.plan.support
        return (
            jnp.zeros((self.num_nodes,), dtype=left_flux.dtype)
            .at[support.tail]
            .add(-left_flux)
            .at[support.head]
            .add(right_flux)
        )

    def _sources(self, u):
        plan, support = self.plan, self.plan.support
        n, p = self.densities(u)
        tl, tn, tp = self.temperatures(u)
        gn, gp, nl, nr, pl, pr, traces, exchanges, thermo = self._transport(u)
        _, _, ec, ev, un, up, bn, bp = thermo
        recombination = support.volumes * self.recombination(u)
        ns = self._transport_sources(nl, nr) - recombination
        ps = self._transport_sources(pl, pr) - recombination
        kn, kp, heat = jnp.zeros_like(n), jnp.zeros_like(n), jnp.zeros_like(n)
        safe_n, safe_p = jnp.where(n > 0, n, 1), jnp.where(p > 0, p, 1)
        # Pair annihilation transfers kinetic plus material gap energy once.
        kn -= recombination * un / safe_n
        kp -= recombination * up / safe_p
        heat += recombination * (un / safe_n + up / safe_p + bn + bp)
        left, right = support.tail, support.head
        if plan.electrothermal:
            thermionic_edges = jnp.zeros(gn.shape, dtype=bool)
            for index, _, _ in exchanges:
                thermionic_edges = thermionic_edges.at[plan.interfaces[index].edge].set(
                    True
                )
            for number, density, kinetic, band, carrier in (
                (gn, safe_n, un, ec, "electron"),
                (gp, safe_p, up, -ev, "hole"),
            ):
                number = jnp.where(thermionic_edges, 0.0, number)
                carried = jnp.where(
                    number >= 0,
                    5 * kinetic[left] / (3 * density[left]),
                    5 * kinetic[right] / (3 * density[right]),
                )
                flux = number * carried
                work = 0.5 * (band[left] - band[right]) * number
                source = self._incoming(flux).at[left].add(work).at[right].add(work)
                if carrier == "electron":
                    kn += source
                else:
                    kp += source
            for index, electron_exchange, hole_exchange in exchanges:
                a, b = plan.interface_nodes[index]
                electron_flux = electron_exchange.number_flux
                hole_flux = hole_exchange.number_flux
                kn = kn.at[a].add(-electron_exchange.energy_flux + bn[a] * electron_flux)
                kn = kn.at[b].add(electron_exchange.energy_flux - bn[b] * electron_flux)
                kp = kp.at[a].add(-hole_exchange.energy_flux + bp[a] * hole_flux)
                kp = kp.at[b].add(hole_exchange.energy_flux - bp[b] * hole_flux)
            conductivity = jnp.zeros_like(n)
            for index, model in enumerate(plan.material_models):
                nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
                conductivity = conductivity.at[nodes].set(
                    model.lattice_thermal_conductivity
                )
            conductive = (
                support.transmissibility
                * _harmonic(conductivity[left], conductivity[right])
                * (tl[left] - tl[right])
            )
            heat += self._incoming(conductive)
            for port in plan.thermal_ports:
                heat = heat.at[port.node].add(
                    port.exchange.evaluate(tl[port.node]).left_energy_source
                )
        trap_rates = {}
        for index, binding in enumerate(plan.traps):
            node, measure = self._trap_geometry(binding)
            occupancy = jax.nn.sigmoid(self.field(u, f"trap_{index}")[0])
            ledger = binding.trap.evaluate(
                occupancy,
                n[node],
                p[node],
                tl[node],
                bn[node] + un[node] / safe_n[node],
                bp[node] + up[node] / safe_p[node],
            )
            ns = ns.at[node].add(measure * ledger.electron_source)
            ps = ps.at[node].add(measure * ledger.hole_source)
            kn = kn.at[node].add(
                measure
                * (ledger.electron_energy_source - bn[node] * ledger.electron_source)
            )
            kp = kp.at[node].add(
                measure * (ledger.hole_energy_source - bp[node] * ledger.hole_source)
            )
            heat = heat.at[node].add(measure * ledger.lattice_energy_source)
            trap_rates[f"trap_{index}"] = ledger.occupancy_rate
        fn = K * plan.temperature * self.field(u, "electron")
        fp = K * plan.temperature * self.field(u, "hole")
        for channel in plan.tunneling:
            a, b = channel.path.path_nodes[0], channel.path.path_nodes[-1]
            fs = jax.nn.sigmoid((fp[a] - channel.energy) / (K * tp[a]))
            fd = jax.nn.sigmoid((fn[b] - channel.energy) / (K * tn[b]))
            ledger = channel.path.evaluate(
                channel.energy, fs, fd, channel.attempt_rate, ev[a], ec[b]
            )
            ns += ledger.electron_source
            ps += ledger.hole_source
            kn += ledger.electron_kinetic_energy_source
            kp += ledger.hole_kinetic_energy_source
        if plan.ionization is not None:
            psi = plan.thermal_voltage * self.field(u, "potential")
            area = support.transmissibility * plan.edge_lengths
            gap = 0.5 * (ec[left] - ev[left] + ec[right] - ev[right])
            ledger = plan.ionization.evaluate(
                (psi[left] - psi[right]) / plan.edge_lengths,
                gn / area,
                gp / area,
                0.5 * (tl[left] + tl[right]),
                gap,
            )
            measure = 0.5 * area * plan.edge_lengths
            for values, target in (
                (ledger.electron_source, "n"),
                (ledger.hole_source, "p"),
                (ledger.electron_energy_source, "kn"),
                (ledger.hole_energy_source, "kp"),
            ):
                source = (
                    jnp.zeros_like(n)
                    .at[left]
                    .add(measure * values)
                    .at[right]
                    .add(measure * values)
                )
                if target == "n":
                    ns += source
                elif target == "p":
                    ps += source
                elif target == "kn":
                    kn += source
                else:
                    kp += source
            material_pair_energy = 0.5 * (bn[left] + bp[left] + bn[right] + bp[right])
            band_entropy_heat = (
                ledger.band_energy_source - material_pair_energy * ledger.electron_source
            )
            heat += (
                jnp.zeros_like(n)
                .at[left]
                .add(measure * band_entropy_heat)
                .at[right]
                .add(measure * band_entropy_heat)
            )
        if plan.carrier_energy:
            for index, model in enumerate(plan.material_models):
                if not isinstance(model, SemiconductorMaterial):
                    continue
                nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
                edges = jnp.asarray(plan.material_edges[index], dtype=jnp.int32)
                carriers = (
                    (
                        n,
                        tn,
                        model.electron_energy_relaxation,
                        model.electron_energy_transport,
                        True,
                    ),
                    (
                        p,
                        tp,
                        model.hole_energy_relaxation,
                        model.hole_energy_transport,
                        False,
                    ),
                )
                for (
                    density,
                    temperature,
                    relaxation_law,
                    transport_law,
                    electron,
                ) in carriers:
                    relaxation = relaxation_law.evaluate(
                        density[nodes], temperature[nodes], tl[nodes]
                    )
                    heat = heat.at[nodes].add(
                        support.volumes[nodes] * relaxation.lattice_energy_source
                    )
                    conductance = (
                        transport_law.thermal_conductivity
                        * support.transmissibility[edges]
                    )
                    conductive = conductance * (
                        temperature[left[edges]] - temperature[right[edges]]
                    )
                    exchange = (
                        jnp.zeros_like(n)
                        .at[left[edges]]
                        .add(-conductive)
                        .at[right[edges]]
                        .add(conductive)
                    )
                    exchange = exchange.at[nodes].add(
                        support.volumes[nodes] * relaxation.carrier_energy_source
                    )
                    if electron:
                        kn += exchange
                    else:
                        kp += exchange
        return ns, ps, kn, kp, heat, traces, trap_rates, thermo

    def physical_storage(self, u):
        """Extensive inventories; potential and interface traces store nothing."""
        plan, volumes = self.plan, self.plan.support.volumes
        n, p = self.densities(u)
        result = self.layout.pack(electron=volumes * n, hole=volumes * p)
        tl, _, _ = self.temperatures(u)
        if plan.electrothermal:
            _, _, _, _, un, up, bn, bp = self._thermodynamic_fields(u, n, p)
            lattice = jnp.zeros_like(n)
            for index, model in enumerate(plan.material_models):
                nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
                lattice = lattice.at[nodes].set(
                    model.lattice_heat_capacity.internal_energy(tl[nodes])
                )
            if plan.carrier_energy:
                result = self.layout.set(
                    result, "electron_energy", volumes * (un + bn * n)
                )
                result = self.layout.set(result, "hole_energy", volumes * (up + bp * p))
            else:
                lattice += un + up + bn * n + bp * p
            result = self.layout.set(result, "lattice_energy", volumes * lattice)
        for index, binding in enumerate(plan.traps):
            _, measure = self._trap_geometry(binding)
            result = self.layout.set(
                result,
                f"trap_{index}",
                measure
                * binding.trap.density
                * jax.nn.sigmoid(self.field(u, f"trap_{index}")[0]),
            )
        return result

    @property
    def storage_scale(self):
        scale = self.layout.pack(
            potential=1.0, electron=self.count_scale, hole=self.count_scale
        )
        for name in self.layout.names:
            if name.endswith("_energy"):
                scale = self.layout.set(scale, name, self.plan.energy_scale)
            elif name.startswith("trap_"):
                binding = self.plan.traps[int(name.split("_")[1])]
                _, measure = self._trap_geometry(binding)
                scale = self.layout.set(scale, name, measure * binding.trap.density)
            elif name.startswith("interface_"):
                scale = self.layout.set(scale, name, 1.0)
        return scale

    def storage(self, u):
        return jnp.where(
            self.differential_mask, self.physical_storage(u) / self.storage_scale, 0
        )

    def storage_coordinates(self, u):
        physical = self.physical_storage(u) / self.storage_scale
        mask = self.differential_mask
        # Algebraic contact rows still use invertible physical storage charts;
        # their time derivatives are recovered from the contact constraints.
        mask = self.layout.set(mask, "electron", self.plan.semiconductor_mask)
        mask = self.layout.set(mask, "hole", self.plan.semiconductor_mask)
        if self.plan.electrothermal:
            mask = self.layout.set(mask, "lattice_energy", True)
        if self.plan.carrier_energy:
            mask = self.layout.set(mask, "electron_energy", self.plan.semiconductor_mask)
            mask = self.layout.set(mask, "hole_energy", self.plan.semiconductor_mask)
        return jnp.where(mask, physical, u)

    def coordinates_from_storage(self, z):
        plan = self.plan
        stored = z * self.storage_scale
        n = self.field(stored, "electron") / plan.support.volumes
        p = self.field(stored, "hole") / plan.support.volumes
        result = z
        if plan.electrothermal:
            # The local energy inversion solves the actual extensive material
            # storage. Native implicit differentiation avoids differentiating
            # scalar iteration histories. No temperature interpolation/floor.
            from ._thermodynamics import _implicit_scalar_root

            lattice = jnp.broadcast_to(plan.temperature, (self.num_nodes,))
            for index, model in enumerate(plan.material_models):
                nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
                target = (
                    self.field(stored, "lattice_energy")[nodes]
                    / plan.support.volumes[nodes]
                )
                capacity = model.lattice_heat_capacity
                lower, upper = capacity.temperature_range
                if isinstance(model, SemiconductorMaterial):
                    lower = jnp.maximum(lower, model.temperature_range[0])
                    upper = jnp.minimum(upper, model.temperature_range[1])
                    bands = model.thermodynamics

                    def solve_one(
                        nn,
                        pp,
                        energy,
                        capacity=capacity,
                        bands=bands,
                        lower=lower,
                        upper=upper,
                    ):
                        def balance(temperature):
                            value = capacity.internal_energy(temperature)
                            if not plan.carrier_energy:
                                value += bands.electron_material_internal_energy_density(
                                    nn, temperature
                                )
                                value += bands.hole_material_internal_energy_density(
                                    pp, temperature
                                )
                            return (value - energy) / (
                                capacity.volumetric_heat_capacity * plan.temperature
                            )

                        return _implicit_scalar_root(balance, lower, upper)

                    values = jax.vmap(solve_one)(n[nodes], p[nodes], target)
                else:
                    values = capacity.temperature(target)
                lattice = lattice.at[nodes].set(values)
            result = self.layout.set(
                result, "lattice_energy", jnp.log(lattice / plan.temperature)
            )
            if plan.carrier_energy:
                carriers = (
                    (
                        "electron",
                        n,
                        lambda bands, density, temperature: bands.electron_energy_density(
                            density, temperature
                        ),
                        True,
                    ),
                    (
                        "hole",
                        p,
                        lambda bands, density, temperature: bands.hole_energy_density(
                            density, temperature
                        ),
                        False,
                    ),
                )
                for name, density, kinetic_energy, electron in carriers:
                    temperatures = lattice
                    for index, model in enumerate(plan.material_models):
                        if not isinstance(model, SemiconductorMaterial):
                            continue
                        nodes = jnp.asarray(plan.material_nodes[index], dtype=jnp.int32)
                        bands = model.thermodynamics
                        target = (
                            self.field(stored, f"{name}_energy")[nodes]
                            / plan.support.volumes[nodes]
                        )
                        ec, ev = bands.band_edges(0.0, lattice[nodes])
                        dec, dev = bands.material_band_temperature_derivatives(
                            lattice[nodes]
                        )
                        coefficient = (
                            ec - lattice[nodes] * dec
                            if electron
                            else -ev + lattice[nodes] * dev
                        )
                        kinetic_target = target - coefficient * density[nodes]
                        lower, upper = model.temperature_range

                        def solve_one(
                            nn,
                            energy,
                            kinetic_energy=kinetic_energy,
                            bands=bands,
                            lower=lower,
                            upper=upper,
                        ):
                            return _implicit_scalar_root(
                                lambda temperature: (
                                    (kinetic_energy(bands, nn, temperature) - energy)
                                    / (K * nn * plan.temperature)
                                ),
                                jnp.asarray(lower),
                                jnp.asarray(upper),
                            )

                        temperatures = temperatures.at[nodes].set(
                            jax.vmap(solve_one)(density[nodes], kinetic_target)
                        )
                    result = self.layout.set(
                        result,
                        f"{name}_energy",
                        jnp.log(temperatures / plan.temperature),
                    )
        for index, binding in enumerate(plan.traps):
            occupancy = self.field(z, f"trap_{index}")
            result = self.layout.set(
                result, f"trap_{index}", jnp.log(occupancy) - jnp.log1p(-occupancy)
            )
        return self.coordinates_from_densities(result, n, p)

    def _physical_residual(self, u, voltages):
        plan = self.plan
        voltage = voltages[jnp.maximum(plan.terminal_index, 0)]
        contact_potential = (
            voltage + plan.neutrality_potential(self.temperatures(u)[0])
        ) / plan.thermal_voltage
        contact_potential = jnp.where(
            plan.ohmic_mask,
            contact_potential,
            (voltage + plan.contact_potential_offset) / plan.thermal_voltage,
        )
        contact_fermi = -voltage / plan.thermal_voltage
        poisson = self.poisson_reaction(u) / self.charge_scale
        poisson = jnp.where(
            plan.potential_mask, self.field(u, "potential") - contact_potential, poisson
        )
        ns, ps, kn, kp, heat, traces, trap_rates, thermo = self._sources(u)
        result = self.layout.pack(potential=poisson / self.time_scale)
        for name, source in (("electron", ns), ("hole", ps)):
            constraint = jnp.where(
                plan.ohmic_mask, self.field(u, name) - contact_fermi, self.field(u, name)
            )
            result = self.layout.set(
                result,
                name,
                jnp.where(
                    plan.semiconductor_mask & ~plan.ohmic_mask,
                    -source / self.count_scale,
                    constraint / self.time_scale,
                ),
            )
        if plan.electrothermal:
            bn, bp = thermo[-2:]
            if plan.carrier_energy:
                result = self.layout.set(
                    result, "electron_energy", -(kn + bn * ns) / plan.energy_scale
                )
                result = self.layout.set(
                    result, "hole_energy", -(kp + bp * ps) / plan.energy_scale
                )
            else:
                heat += kn + kp + bn * ns + bp * ps
            result = self.layout.set(result, "lattice_energy", -heat / plan.energy_scale)
            for name in self.layout.bulk_names[3:]:
                values = self.field(result, name)
                constrained = (
                    plan.ohmic_mask
                    if name == "lattice_energy"
                    else plan.ohmic_mask | ~plan.semiconductor_mask
                )
                # Electrical reservoirs accommodate carrier and lattice energy;
                # dielectric lattice storage remains dynamic unless separately ported.
                values = jnp.where(
                    constrained,
                    self.field(u, name) / self.time_scale,
                    values,
                )
                result = self.layout.set(result, name, values)
        for name, rate in traces.items():
            result = self.layout.set(result, name, rate)
        for name, rate in trap_rates.items():
            result = self.layout.set(result, name, -rate)
        return result

    def residual(self, u, voltages):
        u, voltages = self._coordinates(u), self._voltages(voltages)
        # Proposal admission is separate from constitutive evaluation. Invalid
        # Newton states produce rejected nonfinite proposals, not error_if aborts.
        return jax.lax.cond(
            self._state_valid(u),
            lambda _: self._physical_residual(u, voltages),
            lambda _: jnp.full_like(u, jnp.nan),
            operand=None,
        )

    def residual_function(self, flat, voltages):
        return (
            self.time_scale
            * self.residual(jnp.asarray(flat).reshape(self.layout.shape), voltages)
        ).reshape(-1)

    def terminal_current(self, u, udot=None):
        ns, ps, *_ = self._sources(u)
        result = Q * self._terminal_sum(ns - ps, self.plan.ohmic_mask)
        if udot is not None:
            result += jax.jvp(self.terminal_charge, (u,), (udot,))[1]
        return result

    def thermal_power(self, u, udot=None):
        """Heat W into the device, electrical-contact accommodation then ports."""
        ns, ps, kn, kp, heat, _, _, thermo = self._sources(u)
        total = heat + kn + kp + thermo[-2] * ns + thermo[-1] * ps
        reaction = -total
        if udot is not None:
            rate = jax.jvp(self.physical_storage, (u,), (udot,))[1]
            for name in self.layout.bulk_names:
                if name.endswith("_energy"):
                    reaction += self.field(rate, name)
        contacts = self._terminal_sum(reaction, self.plan.ohmic_mask)
        tl, _, _ = self.temperatures(u)
        ports = jnp.asarray(
            [
                port.exchange.evaluate(tl[port.node]).left_energy_source
                for port in self.plan.thermal_ports
            ]
        )
        return jnp.concatenate((contacts, ports))

    def total_energy(self, u):
        """Material, lattice, trap and Poisson energy, each stored exactly once."""
        stored = self.physical_storage(u)
        energy = jnp.asarray(0.0)
        for name in self.layout.bulk_names:
            if name.endswith("_energy"):
                energy += jnp.sum(self.field(stored, name))
        psi = self.plan.thermal_voltage * self.field(u, "potential")
        field = (
            0.5
            * self.edge_capacitance
            * (psi[self.plan.support.tail] - psi[self.plan.support.head]) ** 2
        )
        for interface, state in zip(
            self.plan.interfaces, self.interface_states(u), strict=True
        ):
            field = field.at[interface.edge].set(state.field_energy)
        for index, binding in enumerate(self.plan.traps):
            energy += binding.trap.trap_energy * jnp.sum(
                self.field(stored, f"trap_{index}")
            )
        return energy + jnp.sum(field)

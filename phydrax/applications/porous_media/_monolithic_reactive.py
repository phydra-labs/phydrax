#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...nonlinear import (
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._reactions import MineralKinetics
from ._speciation import MassActionSystem
from ._transport import ComponentTransport, TransportBoundary


class MonolithicReactiveTransportResult(StrictModule):
    species_concentrations_mol_m3: Array
    component_inventory_mol: Array
    mineral_inventory_mol: Array | None
    face_component_flux_mol_s: Array
    component_balance_mol: Array
    root: NonlinearResult
    successful: Array


class MonolithicReactiveTransportPlan(StrictModule):
    """One global transport/speciation/mineral root on fixed water fluxes."""

    transport: ComponentTransport
    chemistry: MassActionSystem
    minerals: MineralKinetics | None

    def __init__(
        self,
        transport: ComponentTransport,
        chemistry: MassActionSystem,
        /,
        *,
        minerals: MineralKinetics | None = None,
    ):
        if not isinstance(transport, ComponentTransport) or not isinstance(
            chemistry, MassActionSystem
        ):
            raise TypeError(
                "Monolithic reactive transport requires transport and mass action."
            )
        if transport.dispersion is not None:
            raise ValueError(
                "Monolithic chemistry currently requires cell-only transport; "
                "hybrid trace unknowns need an explicit coupled layout."
            )
        if len(transport.component_names) != chemistry.component_count:
            raise ValueError("Transport component and chemistry basis counts disagree.")
        if minerals is not None and (
            not isinstance(minerals, MineralKinetics)
            or minerals.chemistry is not chemistry
        ):
            raise ValueError("Mineral kinetics must use the identical chemistry object.")
        self.transport, self.chemistry, self.minerals = transport, chemistry, minerals

    def step(
        self,
        previous_component_inventory_mol: ArrayLike,
        water_volumes_m3: ArrayLike,
        face_water_rates_m3_s: ArrayLike,
        dt_s: ArrayLike,
        boundary: TransportBoundary,
        initial_species_concentrations_mol_m3: ArrayLike,
        /,
        *,
        previous_mineral_inventory_mol: ArrayLike | None = None,
        reactive_area_m2: ArrayLike | None = None,
        component_source_mol_s: ArrayLike = 0.0,
        termination: NonlinearTermination | None = None,
    ) -> MonolithicReactiveTransportResult:
        cells = self.transport.discretization.cell_count
        species = self.chemistry.species_count
        components = self.chemistry.component_count
        old = jnp.asarray(previous_component_inventory_mol)
        volume = jnp.asarray(water_volumes_m3)
        water_rate = jnp.asarray(face_water_rates_m3_s)
        initial = jnp.asarray(initial_species_concentrations_mol_m3)
        source = jnp.broadcast_to(
            jnp.asarray(component_source_mol_s), (cells, components)
        )
        dt = jnp.asarray(dt_s)
        if (
            old.shape != (cells, components)
            or volume.shape != (cells,)
            or water_rate.shape != (self.transport.discretization.owner_cells.size,)
            or initial.shape != (cells, species)
            or dt.shape != ()
        ):
            raise ValueError(
                "Monolithic reactive transport state/flux/timestep shapes are invalid."
            )
        old = eqx.error_if(
            old,
            jnp.any(~jnp.isfinite(old))
            | jnp.any(~jnp.isfinite(volume))
            | jnp.any(volume <= 0)
            | jnp.any(~jnp.isfinite(water_rate))
            | jnp.any(~jnp.isfinite(initial))
            | jnp.any(initial <= 0)
            | jnp.any(~jnp.isfinite(source))
            | ~jnp.isfinite(dt)
            | (dt <= 0),
            "Monolithic reactive transport inputs must be finite and physical.",
        )
        logs_initial = jnp.log(initial / self.chemistry.reference_concentration)
        mineral_count = 0 if self.minerals is None else len(self.minerals.mineral_names)
        if self.minerals is None:
            mineral_old = area = extent_initial = jnp.zeros((cells, 0))
        else:
            if previous_mineral_inventory_mol is None or reactive_area_m2 is None:
                raise ValueError(
                    "Monolithic mineral chemistry needs inventory and reactive area."
                )
            mineral_old = jnp.asarray(previous_mineral_inventory_mol)
            area = jnp.asarray(reactive_area_m2)
            if (
                mineral_old.shape != (cells, mineral_count)
                or area.shape != mineral_old.shape
            ):
                raise ValueError("Monolithic mineral inventory/area shapes are invalid.")
            mineral_old = eqx.error_if(
                mineral_old,
                jnp.any(~jnp.isfinite(mineral_old))
                | jnp.any(mineral_old < 0)
                | jnp.any(~jnp.isfinite(area))
                | jnp.any(area < 0),
                "Monolithic mineral inventories and areas must be finite nonnegative.",
            )
            extent_initial = jnp.zeros_like(mineral_old)
        scale = jnp.maximum(jnp.max(jnp.abs(old), axis=0), 1.0)

        def residual(state, args):
            logs, extent = state
            old_, volume_, water_, time_, source_, mineral_, area_ = args
            concentrations = self.chemistry.reference_concentration * jnp.exp(logs)
            totals = jax.vmap(self.chemistry.component_totals)(concentrations)
            transport_balance = self.transport.residual(
                totals,
                old_,
                volume_,
                water_,
                time_,
                boundary,
                source=source_,
            )
            equilibrium = jax.vmap(
                lambda local_logs, local_totals: self.chemistry.residual(
                    local_logs, local_totals
                )[components:]
            )(logs, totals)
            if self.minerals is None:
                mineral_residual = extent
                mineral_contribution = jnp.zeros_like(transport_balance)
            else:
                mineral_contribution = contract(
                    "cm,mb->cb", extent, self.minerals.stoichiometry
                )
                rates = jax.vmap(self.minerals.rates)(logs, mineral_, area_)
                unconstrained = time_ * rates
                bounded = jnp.minimum(unconstrained, mineral_)
                mineral_scale = jnp.maximum(mineral_ + time_ * jnp.abs(rates), 1e-30)
                mineral_residual = (extent - bounded) / mineral_scale
            return (
                (transport_balance - mineral_contribution) / scale,
                equilibrium,
                mineral_residual,
            )

        def flat_residual(flat, args):
            logs = flat[: cells * species].reshape((cells, species))
            extent = flat[cells * species :].reshape((cells, mineral_count))
            transport_residual, equilibrium, mineral_residual = residual(
                (logs, extent), args
            )
            return jnp.concatenate(
                (
                    transport_residual.reshape(-1),
                    equilibrium.reshape(-1),
                    mineral_residual.reshape(-1),
                )
            )

        def valid(flat, residual_value, auxiliary, args):
            del residual_value, auxiliary, args
            logs = flat[: cells * species].reshape((cells, species))
            extent = flat[cells * species :].reshape((cells, mineral_count))
            return (
                jnp.all(jnp.isfinite(logs))
                & jnp.all(jnp.isfinite(extent))
                & jnp.all(extent >= 0)
                & jnp.all(extent <= mineral_old)
            )

        initial_flat = jnp.concatenate(
            (logs_initial.reshape(-1), extent_initial.reshape(-1))
        )
        arguments = (old, volume, water_rate, dt, source, mineral_old, area)
        root = implicit_root_result(
            NonlinearSystemProblem(
                flat_residual,
                validity=valid,
                problem_id="monolithic-reactive-transport",
            ),
            initial_flat,
            args=arguments,
            termination=termination,
        )
        logs = root.state[: cells * species].reshape((cells, species))
        extent = root.state[cells * species :].reshape((cells, mineral_count))
        concentrations = self.chemistry.reference_concentration * jnp.exp(logs)
        totals = jax.vmap(self.chemistry.component_totals)(concentrations)
        inventory = volume[:, None] * totals
        face_flux = self.transport.advective_fluxes(totals, water_rate, boundary)
        boundary_flux = jnp.sum(
            jnp.where(
                (self.transport.discretization.neighbour_cells < 0)[:, None],
                face_flux,
                0.0,
            ),
            axis=0,
        )
        if self.minerals is None:
            remaining = None
            solid_change = jnp.zeros_like(boundary_flux)
        else:
            remaining = mineral_old - extent
            solid_change = jnp.sum(
                contract(
                    "cm,mb->cb",
                    remaining - mineral_old,
                    self.minerals.stoichiometry,
                ),
                axis=0,
            )
        balance = (
            jnp.sum(inventory - old - dt * source, axis=0)
            + solid_change
            + dt * boundary_flux
        )
        successful = root.successful & valid(root.state, root.residual, None, arguments)
        return MonolithicReactiveTransportResult(
            concentrations,
            inventory,
            remaining,
            face_flux,
            balance,
            root,
            successful,
        )


__all__ = ["MonolithicReactiveTransportPlan", "MonolithicReactiveTransportResult"]

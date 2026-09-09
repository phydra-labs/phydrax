#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conserved aqueous-component transport independent of a water-flow model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.finite_volume._hybrid_diffusion import HybridMimeticDiffusion
from ...discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _face_components(value, faces, components, name):
    array = jnp.asarray(value, dtype=float)
    if array.shape not in ((), (components,), (faces, components)):
        raise ValueError(
            f"{name} must be scalar, component vector or face-by-component array."
        )
    return jnp.broadcast_to(array, (faces, components))


def _integrated_divergence(discretization, face_rates):
    """Scatter one owner-oriented integrated rate, without multiplying by area."""
    neighbour = discretization.neighbour_cells
    shape = (discretization.cell_count,) + face_rates.shape[1:]
    result = (
        jnp.zeros(shape, dtype=face_rates.dtype)
        .at[discretization.owner_cells]
        .add(face_rates)
    )
    mask = (neighbour >= 0).reshape((neighbour.size,) + (1,) * (face_rates.ndim - 1))
    return result.at[jnp.maximum(neighbour, 0)].add(jnp.where(mask, -face_rates, 0.0))


class TransportBoundary(StrictModule):
    """Separate upstream advective data from dispersive boundary conditions.

    Concentrations are mol/m³ of water; ``dispersion_flux`` is outward,
    area-integrated mol/s and defaults to zero (impermeable to dispersion).
    ``dirichlet_mask`` selects dispersive concentration traces on boundary faces.
    Inflow concentrations are mandatory on any face whose accepted water rate is
    inward. Outflow always uses the owner concentration, never an inlet value.
    """

    inflow_concentration: Array | None
    dirichlet_mask: Array
    dispersion_concentration: Array
    dispersion_flux: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        component_count: int,
        *,
        inflow_concentration: ArrayLike | None = None,
        dirichlet_mask: ArrayLike | None = None,
        dispersion_concentration: ArrayLike = 0.0,
        dispersion_flux: ArrayLike = 0.0,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("TransportBoundary requires native prepared FV geometry.")
        if not isinstance(component_count, int) or component_count < 1:
            raise ValueError("component_count must be positive.")
        nf = discretization.owner_cells.size
        mask = (
            jnp.zeros((nf,), dtype=bool)
            if dirichlet_mask is None
            else jnp.asarray(dirichlet_mask, dtype=bool)
        )
        if mask.shape != (nf,):
            raise ValueError("dirichlet_mask must have one entry per face.")
        mask = eqx.error_if(
            mask,
            jnp.any(mask & (discretization.neighbour_cells >= 0)),
            "Only boundary faces admit prescribed dispersive traces.",
        )
        concentration = _face_components(
            dispersion_concentration, nf, component_count, "dispersion_concentration"
        )
        flux = _face_components(dispersion_flux, nf, component_count, "dispersion_flux")
        concentration = eqx.error_if(
            concentration,
            jnp.any(mask[:, None] & ~jnp.isfinite(concentration)),
            "Prescribed dispersion concentrations must be finite.",
        )
        flux = eqx.error_if(
            flux,
            jnp.any(~jnp.isfinite(flux))
            | jnp.any((discretization.neighbour_cells >= 0)[:, None] & (flux != 0))
            | jnp.any(mask[:, None] & (flux != 0)),
            "Dispersive fluxes must be finite and prescribed only on Neumann boundary faces.",
        )
        self.inflow_concentration = (
            None
            if inflow_concentration is None
            else _face_components(
                inflow_concentration, nf, component_count, "inflow_concentration"
            )
        )
        self.dirichlet_mask, self.dispersion_concentration, self.dispersion_flux = (
            mask,
            concentration,
            flux,
        )
        self.prepared_id = discretization.prepared_id


class TransportStep(StrictModule):
    concentrations: Array
    component_inventory: Array
    face_concentrations: Array | None
    face_component_rates: Array
    component_balance: Array
    root: NonlinearResult


class ComponentTransport(StrictModule):
    """Backward-Euler upstream transport with optional native hybrid dispersion.

    Inventory is water_volume * total_component_concentration (mol); face water
    rates are ALREADY area-integrated m³/s, positive out of owner. Water volumes
    are accepted from any conservative hydraulic model, not recomputed here.
    All numerical geometry is in metres. Dry cells are outside this aqueous-only
    profile and rejected rather than assigned a fictitious concentration.

    ``dispersion_tensor`` is the effective bulk coefficient theta*D (m²/s) on
    3D cells, shared by all components. It must be SPD as checked by the native
    hybrid operator; off-diagonal tensor entries are retained. The hybrid solve
    includes shared face traces and flux-continuity equations: there is no tensor
    TPFA approximation or cell-only substitution. Two-dimensional FV geometry can
    transport explicitly supplied water volumes/face rates (e.g. aperture-weighted
    fractures), but this 3D hybrid dispersion operator is not applied to it.

    Transporting analytical totals with a common velocity and dispersion tensor
    assumes every included aqueous species shares that mobility; ion-specific
    diffusion/electromigration requires a species-resolved electrochemical model.
    """

    discretization: UnstructuredFiniteVolumeDiscretization
    component_names: tuple[str, ...] = eqx.field(static=True)
    nonnegative_components: tuple[bool, ...] = eqx.field(static=True)
    dispersion: HybridMimeticDiffusion | None
    dispersion_tensor: Array | None

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        component_names: tuple[str, ...],
        *,
        nonnegative_components: tuple[bool, ...] | None = None,
        dispersion: HybridMimeticDiffusion | None = None,
        dispersion_tensor: ArrayLike | None = None,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("ComponentTransport requires native prepared FV geometry.")
        names = tuple(component_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n.strip() for n in names)
        ):
            raise ValueError("Components require unique nonempty names.")
        signs = (
            (True,) * len(names)
            if nonnegative_components is None
            else tuple(nonnegative_components)
        )
        if len(signs) != len(names) or any(not isinstance(v, bool) for v in signs):
            raise ValueError(
                "nonnegative_components must declare every component; signed proton bases may use False."
            )
        if (dispersion is None) != (dispersion_tensor is None):
            raise ValueError(
                "A dispersion tensor and its native hybrid operator must be supplied together."
            )
        if dispersion is not None:
            if (
                not isinstance(dispersion, HybridMimeticDiffusion)
                or dispersion.discretization.prepared_id != discretization.prepared_id
            ):
                raise ValueError(
                    "Dispersion must be the native hybrid operator for this exact prepared geometry."
                )
            if discretization.cell_dimension != 3:
                raise ValueError("Hybrid tensor dispersion requires 3D geometry.")
        self.discretization, self.component_names, self.nonnegative_components = (
            discretization,
            names,
            signs,
        )
        self.dispersion = dispersion
        self.dispersion_tensor = (
            None
            if dispersion_tensor is None
            else jnp.asarray(dispersion_tensor, dtype=float)
        )

    def _fields(
        self, previous_inventory, water_volumes, face_water_rates, boundary, source
    ):
        d, nb = self.discretization, len(self.component_names)
        inventory, volumes, rates = (
            jnp.asarray(previous_inventory, dtype=float),
            jnp.asarray(water_volumes, dtype=float),
            jnp.asarray(face_water_rates, dtype=float),
        )
        if (
            inventory.shape != (d.cell_count, nb)
            or volumes.shape != (d.cell_count,)
            or rates.shape != (d.owner_cells.size,)
        ):
            raise ValueError(
                "Inventory, water volumes and integrated water rates must cover native cells/faces."
            )
        if (
            not isinstance(boundary, TransportBoundary)
            or boundary.prepared_id != d.prepared_id
            or boundary.dispersion_flux.shape[1] != nb
        ):
            raise ValueError(
                "Transport boundary must match this geometry and component basis."
            )
        volumes = eqx.error_if(
            volumes,
            jnp.any(~jnp.isfinite(volumes) | (volumes <= 0)),
            "Aqueous transport requires positive finite water volumes; dry cells need a separate phase model.",
        )
        inventory = eqx.error_if(
            inventory,
            jnp.any(~jnp.isfinite(inventory))
            | jnp.any((inventory < 0) & jnp.asarray(self.nonnegative_components)),
            "Component inventories violate their declared domain.",
        )
        rates = eqx.error_if(
            rates,
            jnp.any(~jnp.isfinite(rates)),
            "Accepted integrated water rates must be finite.",
        )
        inflow = (d.neighbour_cells < 0) & (rates < 0)
        if boundary.inflow_concentration is None:
            rates = eqx.error_if(
                rates,
                jnp.any(inflow),
                "Inward water flux requires prescribed upstream component concentrations.",
            )
        else:
            bad = ~jnp.isfinite(boundary.inflow_concentration) | (
                (boundary.inflow_concentration < 0)
                & jnp.asarray(self.nonnegative_components)
            )
            rates = eqx.error_if(
                rates,
                jnp.any(inflow[:, None] & bad),
                "Incoming concentrations violate the component domain.",
            )
        sources = jnp.broadcast_to(
            jnp.asarray(source, dtype=inventory.dtype), inventory.shape
        )
        sources = eqx.error_if(
            sources,
            jnp.any(~jnp.isfinite(sources)),
            "Cell component sources must be finite integrated mol/s.",
        )
        sources = eqx.error_if(
            sources,
            jnp.any(
                boundary.dirichlet_mask[:, None]
                & (boundary.dispersion_concentration < 0)
                & jnp.asarray(self.nonnegative_components)
            ),
            "Prescribed dispersion concentrations violate the component domain.",
        )
        if self.dispersion is None:
            sources = eqx.error_if(
                sources,
                jnp.any(boundary.dirichlet_mask) | jnp.any(boundary.dispersion_flux != 0),
                "Dispersive boundary data require a dispersion operator.",
            )
        return inventory, volumes, rates, sources

    def advective_fluxes(
        self,
        concentrations: ArrayLike,
        face_water_rates: ArrayLike,
        boundary: TransportBoundary,
    ) -> Array:
        """Return owner-oriented mol/s with exactly one upstream concentration."""
        d = self.discretization
        c, q = jnp.asarray(concentrations), jnp.asarray(face_water_rates)
        if c.shape != (d.cell_count, len(self.component_names)) or q.shape != (
            d.owner_cells.size,
        ):
            raise ValueError(
                "Concentrations and water rates must cover native cells/faces."
            )
        outside = (
            jnp.zeros_like(c[d.owner_cells])
            if boundary.inflow_concentration is None
            else boundary.inflow_concentration
        )
        downstream = jnp.where(
            (d.neighbour_cells >= 0)[:, None],
            c[jnp.maximum(d.neighbour_cells, 0)],
            outside,
        )
        upstream = jnp.where((q >= 0)[:, None], c[d.owner_cells], downstream)
        return q[:, None] * upstream

    def _dispersion_fields(self, concentrations, face_concentrations):
        dispersion = self.dispersion
        if dispersion is None:
            raise ValueError(
                "Dispersion fields require a prepared hybrid diffusion operator."
            )
        local = jax.vmap(
            lambda c, f: dispersion.local_fluxes(c, f, self.dispersion_tensor),
            in_axes=(1, 1),
            out_axes=2,
        )(concentrations, face_concentrations)
        continuity = jax.vmap(dispersion.continuity_residual, in_axes=2, out_axes=1)(
            local
        )
        faces = jax.vmap(dispersion.owner_rates, in_axes=2, out_axes=1)(local)
        return jnp.sum(local, axis=1), continuity, faces

    def residual(
        self,
        state,
        previous_inventory: ArrayLike,
        water_volumes: ArrayLike,
        face_water_rates: ArrayLike,
        dt: ArrayLike,
        boundary: TransportBoundary,
        *,
        source: ArrayLike = 0.0,
    ):
        """Unscaled inventory (mol) and hybrid trace equations for monolithic use.

        Without dispersion state/residual is (cells,components). With dispersion
        state is ``(cell_concentrations, face_concentrations)``; residual is
        ``(inventory_residual, trace_residual)``. Interior/Neumann trace equations
        are mol/s; Dirichlet trace equations are mol/m³. No chemistry is applied
        inside this function; append local mass-action/mineral equations directly.
        """
        c = state if self.dispersion is None else state[0]
        advective = self.advective_fluxes(c, face_water_rates, boundary)
        outflow = _integrated_divergence(self.discretization, advective)
        if self.dispersion is not None:
            diffusive, continuity, flux = self._dispersion_fields(c, state[1])
            outflow = outflow + diffusive
            interior = self.discretization.neighbour_cells >= 0
            trace = jnp.where(
                interior[:, None], continuity, flux - boundary.dispersion_flux
            )
            trace = jnp.where(
                boundary.dirichlet_mask[:, None],
                state[1] - boundary.dispersion_concentration,
                trace,
            )
        balance = (
            jnp.asarray(water_volumes)[:, None] * c
            - jnp.asarray(previous_inventory)
            + jnp.asarray(dt) * (outflow - jnp.asarray(source))
        )
        return balance if self.dispersion is None else (balance, trace)

    def step(
        self,
        previous_inventory: ArrayLike,
        water_volumes: ArrayLike,
        face_water_rates: ArrayLike,
        dt: ArrayLike,
        boundary: TransportBoundary,
        *,
        source: ArrayLike = 0.0,
        initial_concentrations: ArrayLike | None = None,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
    ) -> TransportStep:
        """One conservative implicit step; unsuccessful/unphysical roots raise.

        ``water_volumes`` are END-of-step volumes. ``source`` is an integrated
        component rate per cell, positive injection; sinks must be physically
        supplied, not hidden concentration clipping. This step preserves the
        discrete global inventory balance including all boundary contributions.
        """
        inventory, volumes, q, sources = self._fields(
            previous_inventory, water_volumes, face_water_rates, boundary, source
        )
        time = jnp.asarray(dt)
        if time.shape != ():
            raise ValueError("dt must be scalar seconds.")
        time = eqx.error_if(
            time, ~jnp.isfinite(time) | (time <= 0), "dt must be positive finite seconds."
        )
        initial = (
            inventory / volumes[:, None]
            if initial_concentrations is None
            else jnp.asarray(initial_concentrations)
        )
        if initial.shape != inventory.shape:
            raise ValueError(
                "Initial concentrations must match cell component inventories."
            )
        signs = jnp.asarray(self.nonnegative_components)
        initial = eqx.error_if(
            initial,
            jnp.any(~jnp.isfinite(initial) | ((initial < 0) & signs)),
            "Initial concentrations violate their domain.",
        )
        concentration_scale = jnp.maximum(jnp.max(jnp.abs(initial), axis=0), 1.0)
        scale = jnp.maximum(
            jnp.max(jnp.abs(inventory) + time * jnp.abs(sources), axis=0),
            jnp.max(volumes) * concentration_scale,
        )
        initial_state = initial
        if self.dispersion is not None:
            traces = initial[self.discretization.owner_cells]
            traces = jnp.where(
                boundary.dirichlet_mask[:, None],
                boundary.dispersion_concentration,
                traces,
            )
            initial_state = (initial, traces)

        def scaled(state, args):
            old, volume, water, step_time, forcing = args
            physical = self.residual(
                state, old, volume, water, step_time, boundary, source=forcing
            )
            if self.dispersion is None:
                return physical / scale
            trace_scale = jnp.where(
                boundary.dirichlet_mask[:, None], concentration_scale, scale / step_time
            )
            return physical[0] / scale, physical[1] / trace_scale

        def valid(state, residual, auxiliary, args):
            c = state if self.dispersion is None else state[0]
            return jnp.all(jnp.isfinite(c) & (~signs | (c >= 0)))

        problem = NonlinearSystemProblem(
            scaled,
            validity=valid,
            problem_id="conserved-component-upstream-hybrid-transport",
        )
        root = implicit_root_result(
            problem,
            initial_state,
            args=(inventory, volumes, q, time, sources),
            method=method,
            termination=termination,
        )
        c = root.state if self.dispersion is None else root.state[0]
        c = eqx.error_if(
            c,
            ~root.successful | ~jnp.all(jnp.isfinite(c) & (~signs | (c >= 0))),
            "Transport requires a successful physical root; no inventory clipping "
            "or failed-root derivatives are admitted.",
        )
        traces = None if self.dispersion is None else root.state[1]
        flux = self.advective_fluxes(c, q, boundary)
        if self.dispersion is not None:
            flux = flux + self._dispersion_fields(c, traces)[2]
        new_inventory = volumes[:, None] * c
        boundary_flux = jnp.sum(
            jnp.where((self.discretization.neighbour_cells < 0)[:, None], flux, 0.0),
            axis=0,
        )
        balance = (
            jnp.sum(new_inventory - inventory - time * sources, axis=0)
            + time * boundary_flux
        )
        return TransportStep(c, new_inventory, traces, flux, balance, root)


__all__ = ["ComponentTransport", "TransportBoundary", "TransportStep"]

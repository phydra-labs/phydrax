#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Stoichiometric mineral kinetics and conservative transport/chemistry splitting."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._speciation import (
    _chemical_method,
    _chemical_termination,
    MassActionSystem,
    SpeciationResult,
)
from ._transport import ComponentTransport, TransportBoundary, TransportStep


class MineralReactionResult(StrictModule):
    """Moles of aqueous components, solid minerals and dissolution extents.

    Positive extent dissolves mineral; negative extent precipitates it. The
    returned component balance includes aqueous AND solid component inventory.
    The kinetic law is implicit Euler with a mineral-availability constraint,
    not an unconstrained solve followed by mass-destroying concentration clipping.
    """

    concentrations: Array
    activities: Array
    component_inventory: Array
    mineral_inventory: Array
    extents: Array
    rates: Array
    exhausted: Array
    component_balance: Array
    mass_action_residual: Array
    kinetic_residual: Array
    root: NonlinearResult


class MineralKinetics(StrictModule):
    """Mass-action saturation kinetics in an explicit neutral-mineral basis.

    ``stoichiometry[mineral, component]`` is the aqueous component yield per mole
    dissolved. Negative entries consume that component (e.g. proton-consuming
    dissolution). ``log_k`` is the natural log of dissolution equilibrium K, so
    Q=∏a_primary**ν and raw rate=k*area*(1-Q/K) in mol/s. ``rate_constants`` are
    mol/(m² s), and caller-provided reactive areas are m², frozen over this step.
    Mineral stoichiometry must conserve charge in the aqueous basis.

    For each mineral, extent=min(dt*rate_at_new_activities, old_mineral_moles).
    This is a physical complementarity condition: an exhausted mineral may have
    positive dissolution affinity but cannot release unavailable matter. The
    SAME bounded extent updates all aqueous and solid inventories. Precipitation
    is permitted on an initially present solid; on an absent phase it requires
    ``allow_nucleation=True`` and a caller-specified nonzero nucleation area.
    Surface evolution is supplied between steps, not invented from bulk moles.

    This is equilibrium aqueous speciation with kinetic solids, not equilibrium
    phase selection. It admits no gas, sorption, redox-electron reservoir, or
    undocumented mineral database. Native implicit derivatives are valid only
    on successful nonsingular roots within a fixed exhaustion/nucleation branch,
    not at a phase switching equality.
    """

    chemistry: MassActionSystem
    mineral_names: tuple[str, ...] = eqx.field(static=True)
    stoichiometry: Array
    log_k: Array
    rate_constants: Array
    allow_nucleation: tuple[bool, ...] = eqx.field(static=True)

    def __init__(
        self,
        chemistry: MassActionSystem,
        mineral_names: tuple[str, ...],
        stoichiometry: ArrayLike,
        log_k: ArrayLike,
        rate_constants: ArrayLike,
        *,
        allow_nucleation: tuple[bool, ...] | None = None,
    ):
        if not isinstance(chemistry, MassActionSystem):
            raise TypeError("MineralKinetics requires a declared MassActionSystem.")
        if chemistry.charge_balance_component is not None:
            raise ValueError(
                "Conservative mineral kinetics requires every component balance; "
                "charge replacement opens a component reservoir."
            )
        names = tuple(mineral_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n.strip() for n in names)
        ):
            raise ValueError("Minerals require unique nonempty names.")
        nu, constants, rates = (
            jnp.asarray(stoichiometry, dtype=float),
            jnp.asarray(log_k, dtype=float),
            jnp.asarray(rate_constants, dtype=float),
        )
        count = len(names)
        if (
            nu.shape != (count, chemistry.component_count)
            or constants.shape != (count,)
            or rates.shape != (count,)
        ):
            raise ValueError(
                "Mineral coefficients must match the declared mineral/component basis."
            )
        nu = eqx.error_if(
            nu,
            jnp.any(~jnp.isfinite(nu)) | jnp.any(jnp.all(nu == 0, axis=1)),
            "Mineral stoichiometry must be finite and nonzero.",
        )
        nu = eqx.error_if(
            nu,
            jnp.any(
                jnp.abs(
                    contract(
                        "mb,b->m", nu, chemistry.charges[: chemistry.component_count]
                    )
                )
                > 1e-10
            ),
            "Neutral mineral dissolution must conserve charge.",
        )
        constants = eqx.error_if(
            constants,
            jnp.any(~jnp.isfinite(constants)),
            "Mineral log equilibrium constants must be finite.",
        )
        rates = eqx.error_if(
            rates,
            jnp.any(~jnp.isfinite(rates) | (rates < 0)),
            "Mineral rate constants must be finite and nonnegative.",
        )
        nucleation = (
            (False,) * count if allow_nucleation is None else tuple(allow_nucleation)
        )
        if len(nucleation) != count or any(not isinstance(v, bool) for v in nucleation):
            raise ValueError("allow_nucleation must explicitly describe each mineral.")
        self.chemistry, self.mineral_names = chemistry, names
        self.stoichiometry, self.log_k, self.rate_constants = nu, constants, rates
        self.allow_nucleation = nucleation

    def saturation_log_ratio(self, log_concentrations: ArrayLike) -> Array:
        activities = self.chemistry.log_activities(log_concentrations)
        return (
            contract(
                "...b,mb->...m",
                activities[..., : self.chemistry.component_count],
                self.stoichiometry,
            )
            - self.log_k
        )

    def rates(
        self,
        log_concentrations: ArrayLike,
        previous_mineral_inventory: ArrayLike,
        reactive_area: ArrayLike,
    ) -> Array:
        """Unconstrained affinity rates; availability is imposed on extents."""
        ratio = self.saturation_log_ratio(log_concentrations)
        raw = -jnp.asarray(reactive_area) * self.rate_constants * jnp.expm1(ratio)
        seeded = (jnp.asarray(previous_mineral_inventory) > 0) | jnp.asarray(
            self.allow_nucleation
        )
        return jnp.where(seeded, raw, 0.0)

    def residual(
        self,
        state,
        previous_component_inventory: ArrayLike,
        previous_mineral_inventory: ArrayLike,
        water_volume: ArrayLike,
        dt: ArrayLike,
        reactive_area: ArrayLike,
    ):
        """Native-root-ready (aqueous equations, bounded kinetic equations).

        ``state=(log(c/reference_concentration), dissolution_extents)``. This
        residual is directly composable with a monolithic global balance: use
        ``chemistry.residual`` with globally transported totals and the same
        mineral extent contribution, and retain the kinetic residual unchanged.
        """
        logs, extent = state
        target = (
            jnp.asarray(previous_component_inventory)
            + contract("...m,mb->...b", extent, self.stoichiometry)
        ) / jnp.asarray(water_volume)[..., None]
        aqueous = self.chemistry.residual(logs, target)
        unconstrained = jnp.asarray(dt)[..., None] * self.rates(
            logs, previous_mineral_inventory, reactive_area
        )
        bounded = jnp.minimum(unconstrained, jnp.asarray(previous_mineral_inventory))
        # A numerical residual scale only; it does not alter physical inventories.
        scale = jnp.maximum(
            jnp.abs(jnp.asarray(previous_mineral_inventory))
            + jnp.asarray(dt)[..., None]
            * jnp.asarray(reactive_area)
            * self.rate_constants,
            1e-30,
        )
        return aqueous, (extent - bounded) / scale

    def step(
        self,
        previous_component_inventory: ArrayLike,
        previous_mineral_inventory: ArrayLike,
        water_volume: ArrayLike,
        dt: ArrayLike,
        reactive_area: ArrayLike,
        *,
        initial_concentrations: ArrayLike,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
    ) -> MineralReactionResult:
        """One coupled aqueous-equilibrium/kinetic-mineral cell step."""
        old = jnp.asarray(previous_component_inventory, dtype=float)
        minerals = jnp.asarray(previous_mineral_inventory, dtype=float)
        volume, time = (
            jnp.asarray(water_volume, dtype=float),
            jnp.asarray(dt, dtype=float),
        )
        area, initial = (
            jnp.asarray(reactive_area, dtype=float),
            jnp.asarray(initial_concentrations, dtype=float),
        )
        if (
            old.shape != (self.chemistry.component_count,)
            or minerals.shape != (len(self.mineral_names),)
            or area.shape != minerals.shape
            or initial.shape != (self.chemistry.species_count,)
            or volume.shape != ()
            or time.shape != ()
        ):
            raise ValueError(
                "Mineral cell solve requires matching local component, mineral, "
                "area and species vectors and scalar volume/time."
            )
        old = eqx.error_if(
            old,
            jnp.any(~jnp.isfinite(old)),
            "Aqueous component inventories must be finite.",
        )
        minerals = eqx.error_if(
            minerals,
            jnp.any(~jnp.isfinite(minerals) | (minerals < 0)),
            "Mineral inventories must be nonnegative finite moles.",
        )
        area = eqx.error_if(
            area,
            jnp.any(~jnp.isfinite(area) | (area < 0)),
            "Reactive areas must be nonnegative finite m².",
        )
        volume = eqx.error_if(
            volume,
            ~jnp.isfinite(volume) | (volume <= 0),
            "Aqueous chemistry requires positive finite water volume.",
        )
        time = eqx.error_if(
            time,
            ~jnp.isfinite(time) | (time <= 0),
            "Reaction dt must be positive finite seconds.",
        )
        initial = eqx.error_if(
            initial,
            jnp.any(~jnp.isfinite(initial) | (initial <= 0)),
            "Initial aqueous concentrations must be positive finite values.",
        )
        arguments = (old, minerals, volume, time, area)
        problem = NonlinearSystemProblem(
            lambda state, args: self.residual(state, *args),
            validity=lambda state, residual, auxiliary, args: (
                self.chemistry.valid(state[0]) & jnp.all(args[1] - state[1] >= 0)
            ),
            problem_id="aqueous-equilibrium-bounded-mineral-kinetics",
        )
        root = implicit_root_result(
            problem,
            (
                jnp.log(initial / self.chemistry.reference_concentration),
                jnp.zeros_like(minerals),
            ),
            args=arguments,
            method=_chemical_method(method),
            termination=_chemical_termination(termination),
        )
        logs, extent = root.state
        extent = eqx.error_if(
            extent,
            ~root.successful
            | ~self.chemistry.valid(logs)
            | jnp.any(minerals - extent < 0),
            "Mineral kinetics requires a successful physical root; unavailable "
            "minerals are never clipped after reaction.",
        )
        c = self.chemistry.reference_concentration * jnp.exp(logs)
        computed = volume * self.chemistry.component_totals(c)
        remaining = minerals - extent
        total_balance = (
            computed - old + contract("m,mb->b", remaining - minerals, self.stoichiometry)
        )
        aqueous, kinetic = self.residual((logs, extent), *arguments)
        return MineralReactionResult(
            c,
            jnp.exp(self.chemistry.log_activities(logs)),
            computed,
            remaining,
            extent,
            self.rates(logs, minerals, area),
            remaining == 0,
            total_balance,
            aqueous[self.chemistry.component_count :],
            kinetic,
            root,
        )


class ReactiveTransportStep(StrictModule):
    """First-order Lie transport→chemistry composition, not a monolithic solve.

    Both substeps conserve analytical totals; global balance includes solids and
    the transport substep's boundary/source exchanges. Chemistry uses the same
    end-of-step water volumes, so no concentration/volume mismatch is introduced.
    """

    transport: TransportStep
    concentrations: Array
    component_inventory: Array
    mineral_inventory: Array | None
    chemistry: SpeciationResult | MineralReactionResult
    component_balance: Array


def reactive_transport_step(
    transport: ComponentTransport,
    chemistry: MassActionSystem | MineralKinetics,
    previous_component_inventory: ArrayLike,
    water_volumes: ArrayLike,
    face_water_rates: ArrayLike,
    dt: ArrayLike,
    boundary: TransportBoundary,
    *,
    initial_species_concentrations: ArrayLike,
    previous_mineral_inventory: ArrayLike | None = None,
    reactive_area: ArrayLike | None = None,
    source: ArrayLike = 0.0,
    transport_method: AbstractNonlinearMethod | None = None,
    chemistry_method: AbstractNonlinearMethod | None = None,
    transport_termination: NonlinearTermination | None = None,
    chemistry_termination: NonlinearTermination | None = None,
) -> ReactiveTransportStep:
    """Conservative first-order operator splitting with actual local root solves.

    Hydraulic rates/volumes must be accepted inputs. A common analytical basis is
    mandatory. Local charge replacement is forbidden because it would turn an
    untracked reservoir into an apparent component source.
    """
    aqueous = chemistry.chemistry if isinstance(chemistry, MineralKinetics) else chemistry
    if (
        not isinstance(aqueous, MassActionSystem)
        or transport.component_names != aqueous.primary_names
    ):
        raise ValueError(
            "Transport and chemistry must use the identical ordered primary-component basis."
        )
    if aqueous.charge_balance_component is not None:
        raise ValueError(
            "Closed reactive transport cannot replace a conserved component with charge balance."
        )
    species = jnp.asarray(initial_species_concentrations)
    nc = transport.discretization.cell_count
    if species.shape != (nc, aqueous.species_count):
        raise ValueError(
            "Initial species concentrations must cover every transported cell."
        )
    moved = transport.step(
        previous_component_inventory,
        water_volumes,
        face_water_rates,
        dt,
        boundary,
        source=source,
        method=transport_method,
        termination=transport_termination,
    )
    volumes = jnp.asarray(water_volumes)
    if isinstance(chemistry, MineralKinetics):
        if previous_mineral_inventory is None or reactive_area is None:
            raise ValueError(
                "Kinetic reactive transport requires mineral inventories and reactive areas."
            )
        solids, areas = (
            jnp.asarray(previous_mineral_inventory),
            jnp.asarray(reactive_area),
        )
        if (
            solids.shape != (nc, len(chemistry.mineral_names))
            or areas.shape != solids.shape
        ):
            raise ValueError(
                "Mineral inventories and reactive areas must cover each cell and mineral."
            )
        reacted = jax.vmap(
            lambda inventory, mineral, volume, area, initial: chemistry.step(
                inventory,
                mineral,
                volume,
                dt,
                area,
                initial_concentrations=initial,
                method=chemistry_method,
                termination=chemistry_termination,
            )
        )(moved.component_inventory, solids, volumes, areas, species)
        inventory, remaining = reacted.component_inventory, reacted.mineral_inventory
        balance = moved.component_balance + jnp.sum(reacted.component_balance, axis=0)
    else:
        if previous_mineral_inventory is not None or reactive_area is not None:
            raise ValueError("Mineral data require a MineralKinetics model.")
        reacted = jax.vmap(
            lambda totals, initial: chemistry.solve(
                totals,
                initial_concentrations=initial,
                method=chemistry_method,
                termination=chemistry_termination,
            )
        )(moved.concentrations, species)
        inventory = volumes[:, None] * reacted.component_totals
        remaining = None
        balance = moved.component_balance + jnp.sum(
            inventory - moved.component_inventory, axis=0
        )
    return ReactiveTransportStep(
        moved, reacted.concentrations, inventory, remaining, reacted, balance
    )


__all__ = [
    "MineralKinetics",
    "MineralReactionResult",
    "ReactiveTransportStep",
    "reactive_transport_step",
]

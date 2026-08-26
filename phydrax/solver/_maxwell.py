#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CochainDiscretization,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    StructuredCochainBridge,
)
from ._maxwell_boundaries import MaxwellBoundaryPlan, PreparedMaxwellBoundary
from ._maxwell_observers import (
    AbstractMaxwellObserverPlan,
    AbstractPreparedMaxwellObserver,
)
from ._maxwell_pml import MaxwellCPMLPlan, MaxwellCPMLState, PreparedMaxwellCPML


class MaxwellCapabilities(StrictModule, NonTrainableState):
    """Fail-closed material, execution, and differentiation capabilities."""

    lossless: bool = eqx.field(static=True)
    passive: bool = eqx.field(static=True)
    active: bool = eqx.field(static=True)
    dispersive: bool = eqx.field(static=True)
    nonlinear: bool = eqx.field(static=True)
    reversible: bool = eqx.field(static=True)
    complex_required: bool = eqx.field(static=True)
    structured_only: bool = eqx.field(static=True)
    pml: bool = eqx.field(static=True)
    observers: bool = eqx.field(static=True)
    frequency_domain: bool = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        lossless: bool,
        passive: bool,
        active: bool = False,
        dispersive: bool = False,
        nonlinear: bool = False,
        reversible: bool = False,
        complex_required: bool = False,
        structured_only: bool = True,
        pml: bool = False,
        observers: bool = False,
        frequency_domain: bool = False,
        distributed: bool = False,
    ):
        if active and passive:
            raise ValueError("A Maxwell capability set cannot be passive and active.")
        if lossless and (active or not passive):
            raise ValueError(
                "Lossless Maxwell capabilities must be passive and inactive."
            )
        if reversible and (not lossless or dispersive or nonlinear or pml):
            raise ValueError(
                "Reversible Maxwell capabilities require lossless instantaneous dynamics."
            )
        values = {
            "lossless": bool(lossless),
            "passive": bool(passive),
            "active": bool(active),
            "dispersive": bool(dispersive),
            "nonlinear": bool(nonlinear),
            "reversible": bool(reversible),
            "complex_required": bool(complex_required),
            "structured_only": bool(structured_only),
            "pml": bool(pml),
            "observers": bool(observers),
            "frequency_domain": bool(frequency_domain),
            "distributed": bool(distributed),
        }
        for name, value in values.items():
            setattr(self, name, value)
        self.capability_id = canonical_fingerprint(
            {"kind": "maxwell-capabilities", **values}
        )


class MaxwellPrimaryState(StrictModule):
    """Conservative electromagnetic fluxes and charge."""

    electric_displacement: Array
    magnetic_flux: Array
    charge: Array


class MaxwellAuxiliaryState(StrictModule):
    """Material and boundary/PML state kept outside the primary fluxes."""

    material: Any
    boundary: Any


class CompatibleMaxwellState(StrictModule):
    """Complete Maxwell carry with primary, auxiliary, and observer state."""

    primary: MaxwellPrimaryState
    auxiliary: MaxwellAuxiliaryState
    observations: Any


class CompatibleMaxwellDiagnostics(StrictModule):
    """Dynamic conservation, energy, source-power, and CFL evidence."""

    energy: Array
    electric_constraint_linf: Array
    pml_dissipation: Array
    magnetic_constraint_linf: Array
    gauss_rate_linf: Array
    source_power: Array
    boundary_dissipation: Array
    power_balance_residual: Array
    stable_step: Array
    step_fraction: Array | None


class AbstractPreparedMaxwellConstitutive(StrictModule):
    """Prepared D↔E and B↔H maps with energy and auxiliary-state semantics."""

    capabilities: MaxwellCapabilities
    prepared_id: str

    @abc.abstractmethod
    def initialize_state(self, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_state(self, state: Any, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def wave_speed_bound(self, /) -> Array:
        """Return a conservative material wave-speed bound."""
        raise NotImplementedError


class AbstractMaxwellConstitutivePlan(StrictModule):
    """Plan that binds a material law to exact cochain spaces."""

    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> AbstractPreparedMaxwellConstitutive:
        raise NotImplementedError


def _apply_hodge_metric(metric: Array, values: Array, /) -> Array:
    return metric * values if metric.ndim == 1 else metric @ values


def _positive_material(name: str, value: ArrayLike, count: int, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    if array.shape not in ((), (1,), (count,)):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    array = jnp.broadcast_to(array, (count,))
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | jnp.any(array <= 0.0),
        f"{name} must be finite and strictly positive.",
    )


class DiagonalMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Positive degree-aligned instantaneous permittivity/permeability."""

    permittivity: Array
    permeability: Array

    def __init__(
        self,
        *,
        permittivity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
        plan_id: str | None = None,
    ):
        epsilon = jnp.asarray(permittivity)
        mu = jnp.asarray(permeability)
        if jnp.iscomplexobj(epsilon) or jnp.iscomplexobj(mu):
            raise TypeError("Diagonal lossless Maxwell materials must be real.")
        if not jnp.issubdtype(epsilon.dtype, jnp.inexact):
            epsilon = epsilon.astype(float)
        if not jnp.issubdtype(mu.dtype, jnp.inexact):
            mu = mu.astype(float)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "diagonal-maxwell-constitutive-plan",
                    "permittivity_shape": list(epsilon.shape),
                    "permittivity_dtype": str(epsilon.dtype),
                    "permeability_shape": list(mu.shape),
                    "permeability_dtype": str(mu.dtype),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.permittivity = epsilon
        self.permeability = mu
        self.plan_id = identifier

    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> PreparedDiagonalMaxwellConstitutive:
        return PreparedDiagonalMaxwellConstitutive(self, cochain)


class PreparedDiagonalMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    """Exact diagonal D↔E and B↔H material maps."""

    permittivity: Array
    permeability: Array
    capabilities: MaxwellCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DiagonalMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        /,
    ):
        if cochain.max_degree != 3:
            raise ValueError("Maxwell constitutive preparation requires dimension three.")
        epsilon = _positive_material(
            "permittivity",
            plan.permittivity,
            cochain.cell_counts[1],
        )
        mu = _positive_material(
            "permeability",
            plan.permeability,
            cochain.cell_counts[2],
        )
        self.permittivity = epsilon
        self.permeability = mu
        self.capabilities = MaxwellCapabilities(
            lossless=True,
            passive=True,
            reversible=True,
            structured_only=False,
            frequency_domain=True,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-diagonal-maxwell-constitutive",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Diagonal instantaneous material state must be None.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return displacement / self.permittivity

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permittivity * electric

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(electric)

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del electric, magnetic, electric_star, magnetic_star
        self.validate_state(state)
        return jnp.asarray(0.0)

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement_rate))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_rate))
        )

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(jnp.max(1.0 / self.permeability) / jnp.min(self.permittivity))


class CompatibleMaxwellPlan(StrictModule):
    """Compatible Maxwell plan over conservative D/B primary fluxes."""

    bridge: StructuredCochainBridge
    constitutive: AbstractMaxwellConstitutivePlan
    boundaries: tuple[MaxwellBoundaryPlan, ...]
    observers: tuple[AbstractMaxwellObserverPlan, ...]
    pml: MaxwellCPMLPlan | None
    current_source: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(
        static=True
    )
    courant_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        constitutive: AbstractMaxwellConstitutivePlan | None = None,
        boundaries: Sequence[MaxwellBoundaryPlan] = (),
        observers: Sequence[AbstractMaxwellObserverPlan] = (),
        pml: MaxwellCPMLPlan | None = None,
        current_source: Callable[[Array, Array, Any], ArrayLike] | None = None,
        courant_factor: float = 0.95,
        plan_id: str | None = None,
    ):
        if not isinstance(bridge, StructuredCochainBridge) or bridge.dimension != 3:
            raise ValueError(
                "Compatible Maxwell dynamics requires a three-dimensional bridge."
            )
        material = (
            DiagonalMaxwellConstitutivePlan() if constitutive is None else constitutive
        )
        if not isinstance(material, AbstractMaxwellConstitutivePlan):
            raise TypeError("constitutive must be an AbstractMaxwellConstitutivePlan.")
        boundary_plans = tuple(boundaries)
        if any(not isinstance(value, MaxwellBoundaryPlan) for value in boundary_plans):
            raise TypeError("boundaries must contain MaxwellBoundaryPlan values.")
        observer_plans = tuple(observers)
        if any(
            not isinstance(value, AbstractMaxwellObserverPlan) for value in observer_plans
        ):
            raise TypeError("observers must contain AbstractMaxwellObserverPlan values.")
        if pml is not None and not isinstance(pml, MaxwellCPMLPlan):
            raise TypeError("pml must be MaxwellCPMLPlan or None.")
        if current_source is not None and not callable(current_source):
            raise TypeError("current_source must be callable or None.")
        factor = float(courant_factor)
        if not np.isfinite(factor) or factor <= 0.0 or factor > 1.0:
            raise ValueError("courant_factor must be finite and lie in (0, 1].")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "compatible-maxwell-plan",
                    "bridge": bridge.bridge_id,
                    "constitutive": material.plan_id,
                    "boundaries": [value.plan_id for value in boundary_plans],
                    "observers": [value.plan_id for value in observer_plans],
                    "pml": None if pml is None else pml.plan_id,
                    "current_source": (
                        None if current_source is None else repr(current_source)
                    ),
                    "courant_factor": factor,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.bridge = bridge
        self.constitutive = material
        self.boundaries = boundary_plans
        self.observers = observer_plans
        self.pml = pml
        self.current_source = current_source
        self.courant_factor = factor
        self.plan_id = identifier

    def prepare(self, /) -> PreparedCompatibleMaxwell:
        return PreparedCompatibleMaxwell(self)


class PreparedCompatibleMaxwell(StrictModule):
    """Prepared heterogeneous compatible Maxwell evolution."""

    plan: CompatibleMaxwellPlan
    constitutive: AbstractPreparedMaxwellConstitutive
    boundaries: tuple[PreparedMaxwellBoundary, ...]
    observers: tuple[AbstractPreparedMaxwellObserver, ...]
    capabilities: MaxwellCapabilities
    pml: PreparedMaxwellCPML | None
    cfl_limit: Array
    stable_dt: Array
    discretization_bundle: DiscretizationBundle
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: CompatibleMaxwellPlan, /):
        if not isinstance(plan, CompatibleMaxwellPlan):
            raise TypeError("plan must be a CompatibleMaxwellPlan.")
        constitutive = plan.constitutive.prepare(plan.bridge.cochain)
        boundaries = tuple(value.prepare(plan.bridge) for value in plan.boundaries)
        counts = plan.bridge.cochain.cell_counts
        observers = tuple(value.prepare(counts[1], counts[2]) for value in plan.observers)
        pml = None if plan.pml is None else plan.pml.prepare(plan.bridge)
        spacings = tuple(
            jnp.min(axis.interval_widths) for axis in plan.bridge.grid.structured_axes
        )
        inverse_spacing_norm = jnp.sqrt(jnp.sum(1.0 / jnp.asarray(spacings) ** 2))
        wave_speed_bound = constitutive.wave_speed_bound()
        cfl_limit = 1.0 / (wave_speed_bound * inverse_spacing_norm)
        stable_dt = plan.courant_factor * cfl_limit
        cochain = plan.bridge.cochain
        system_key = DiscretizationKey(
            "compatible_maxwell",
            DiscretizationRole.RESIDUAL,
            domain_labels=plan.bridge.grid.axis_names,
        )
        bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    cochain.key,
                    type(cochain).__name__,
                    cochain.prepared_id,
                    numeric_version=cochain.numeric_version,
                    precision_evidence_id=cochain.precision_evidence_id,
                    resource_evidence_id=cochain.resource_evidence_id,
                ),
                DiscretizationRecord(
                    system_key,
                    "compatible-maxwell-system",
                    plan.plan_id,
                    dependency_key_ids=(cochain.key.key_id,),
                ),
            )
        )
        self.plan = plan
        self.constitutive = constitutive
        self.capabilities = MaxwellCapabilities(
            lossless=(
                constitutive.capabilities.lossless
                and all(value.kind != "impedance" for value in boundaries)
            ),
            passive=constitutive.capabilities.passive,
            active=constitutive.capabilities.active,
            dispersive=constitutive.capabilities.dispersive,
            nonlinear=constitutive.capabilities.nonlinear,
            reversible=(
                constitutive.capabilities.reversible
                and all(value.kind != "impedance" for value in boundaries)
                and pml is None
            ),
            observers=bool(observers),
            structured_only=True,
            pml=pml is not None,
            frequency_domain=constitutive.capabilities.frequency_domain,
            distributed=False,
        )
        self.boundaries = boundaries
        self.observers = observers
        self.cfl_limit = cfl_limit
        self.pml = pml
        self.stable_dt = stable_dt
        self.discretization_bundle = bundle
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-compatible-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "constitutive": constitutive.prepared_id,
            }
        )

    @property
    def primary_counts(self) -> tuple[int, int, int]:
        counts = self.plan.bridge.cochain.cell_counts
        return counts[1], counts[2], counts[0]

    def _constrain_primary(
        self,
        displacement: Array,
        magnetic_flux: Array,
        /,
    ) -> tuple[Array, Array]:
        for boundary in self.boundaries:
            displacement, magnetic_flux = boundary.constrain_primary(
                displacement,
                magnetic_flux,
            )
        return displacement, magnetic_flux

    def pack(
        self,
        electric_displacement: ArrayLike,
        magnetic_flux: ArrayLike,
        charge: ArrayLike | None = None,
        /,
        *,
        material_state: Any = None,
        boundary_state: Any = None,
        observations: Any = None,
    ) -> CompatibleMaxwellState:
        displacement = jnp.asarray(electric_displacement)
        flux = jnp.asarray(magnetic_flux)
        electric_count, magnetic_count, charge_count = self.primary_counts
        charge_ = (
            jnp.zeros((charge_count,), dtype=displacement.dtype)
            if charge is None
            else jnp.asarray(charge)
        )
        if boundary_state is None:
            boundary_state_ = None if self.pml is None else self.pml.initialize()
        else:
            boundary_state_ = boundary_state
            if self.pml is None or not isinstance(boundary_state_, MaxwellCPMLState):
                raise ValueError("Maxwell boundary state is incompatible with CPML.")
        displacement, flux = self._constrain_primary(displacement, flux)
        if observations is None:
            observation_state = tuple(value.initialize() for value in self.observers)
        else:
            observation_state = tuple(observations)
            if len(observation_state) != len(self.observers):
                raise ValueError(
                    "Maxwell observation state count does not match observers."
                )
        if (
            displacement.shape != (electric_count,)
            or flux.shape != (magnetic_count,)
            or charge_.shape != (charge_count,)
        ):
            raise ValueError("Maxwell D, B, and charge cochains have wrong sizes.")
        self.constitutive.validate_state(material_state)
        return CompatibleMaxwellState(
            primary=MaxwellPrimaryState(
                electric_displacement=displacement,
                magnetic_flux=flux,
                charge=charge_,
            ),
            auxiliary=MaxwellAuxiliaryState(
                material=material_state,
                boundary=boundary_state_,
            ),
            observations=observation_state,
        )

    def initialize(
        self,
        *,
        electric_displacement: ArrayLike | None = None,
        magnetic_flux: ArrayLike | None = None,
        charge: ArrayLike | None = None,
    ) -> CompatibleMaxwellState:
        electric_count, magnetic_count, _ = self.primary_counts
        displacement = (
            jnp.zeros((electric_count,))
            if electric_displacement is None
            else electric_displacement
        )
        flux = jnp.zeros((magnetic_count,)) if magnetic_flux is None else magnetic_flux
        return self.pack(
            displacement,
            flux,
            charge,
            material_state=self.constitutive.initialize_state(),
        )

    def _state(self, state: CompatibleMaxwellState, /) -> CompatibleMaxwellState:
        if not isinstance(state, CompatibleMaxwellState):
            raise TypeError("state must be CompatibleMaxwellState.")
        self.pack(
            state.primary.electric_displacement,
            state.primary.magnetic_flux,
            state.primary.charge,
            material_state=state.auxiliary.material,
            boundary_state=state.auxiliary.boundary,
            observations=state.observations,
        )
        return state

    def electric_field(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        electric = self.constitutive.electric_field(
            state_.primary.electric_displacement,
            state_.auxiliary.material,
        )
        magnetic = self.constitutive.magnetic_field(
            state_.primary.magnetic_flux,
            state_.auxiliary.material,
        )
        for boundary in self.boundaries:
            electric, magnetic = boundary.constrain_fields(electric, magnetic)
        return electric

    def magnetic_field(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        electric = self.constitutive.electric_field(
            state_.primary.electric_displacement,
            state_.auxiliary.material,
        )
        magnetic = self.constitutive.magnetic_field(
            state_.primary.magnetic_flux,
            state_.auxiliary.material,
        )
        for boundary in self.boundaries:
            electric, magnetic = boundary.constrain_fields(electric, magnetic)
        return magnetic

    def _current(self, time: Array, state: CompatibleMaxwellState, args: Any, /) -> Array:
        if self.plan.current_source is None:
            return jnp.zeros_like(state.primary.electric_displacement)
        coordinates = self.plan.bridge.cochain.coordinates[1]
        if coordinates is None:
            raise RuntimeError("Compatible Maxwell edge coordinates are unavailable.")
        current = jnp.asarray(
            self.plan.current_source(jnp.asarray(time), coordinates, args)
        )
        if current.shape != state.primary.electric_displacement.shape:
            raise ValueError("Maxwell current_source must return one value per edge.")
        return eqx.error_if(
            current,
            jnp.any(~jnp.isfinite(current)),
            "Maxwell current_source must return finite values.",
        )

    def _boundary_current(self, electric: Array, /) -> Array:
        current = jnp.zeros_like(electric)
        for boundary in self.boundaries:
            current = current + boundary.impedance_current(electric)
        return current

    def boundary_dissipation(self, state: CompatibleMaxwellState, /) -> Array:
        electric = self.electric_field(state)
        total = jnp.asarray(0.0, dtype=electric.real.dtype)
        for boundary in self.boundaries:
            total = total + boundary.dissipated_power(electric)
        return total

    def pml_dissipation(self, state: CompatibleMaxwellState, /) -> Array:
        if self.pml is None:
            return jnp.asarray(0.0)
        return self.pml.diagnostics(
            self.electric_field(state),
            self.magnetic_field(state),
        ).absorbed_power

    def material_dissipation(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        return self.constitutive.dissipated_power(
            self.electric_field(state_),
            self.magnetic_field(state_),
            state_.auxiliary.material,
            self.plan.bridge.cochain.hodge_metric(1),
            self.plan.bridge.cochain.hodge_metric(2),
        )

    def _rates_with_current(
        self,
        state: CompatibleMaxwellState,
        current: Array,
        /,
    ) -> MaxwellPrimaryState:
        electric = self.electric_field(state)
        magnetic = self.magnetic_field(state)
        bridge = self.plan.bridge
        total_current = (
            current
            + self._boundary_current(electric)
            + self.constitutive.electric_conduction(
                electric,
                state.auxiliary.material,
            )
        )
        magnetic_loss = self.constitutive.magnetic_conduction(
            magnetic,
            state.auxiliary.material,
        )
        return MaxwellPrimaryState(
            electric_displacement=bridge.codifferential(2, magnetic) - total_current,
            magnetic_flux=-bridge.exterior_derivative(1, electric) - magnetic_loss,
            charge=-bridge.codifferential(1, total_current),
        )

    def drift(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> CompatibleMaxwellState:
        state_ = self._state(state)
        current = self._current(jnp.asarray(time), state_, args)
        rates = self._rates_with_current(state_, current)
        return self.pack(
            rates.electric_displacement,
            rates.magnetic_flux,
            rates.charge,
            material_state=state_.auxiliary.material,
            boundary_state=state_.auxiliary.boundary,
            observations=state_.observations,
        )

    def _step_size(self, step_size: ArrayLike, /) -> Array:
        dt = jnp.asarray(step_size)
        if dt.shape != ():
            raise ValueError("Maxwell step_size must be scalar.")
        return eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0) | (dt > self.stable_dt),
            "Maxwell step_size must be finite, positive, and no larger than stable_dt.",
        )

    def leapfrog_step(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> CompatibleMaxwellState:
        state_ = self._state(state)
        dt = self._step_size(step_size)
        half_step = 0.5 * dt
        electric = self.electric_field(state_)
        magnetic_forcing = -self.plan.bridge.exterior_derivative(
            1, electric
        ) - self.constitutive.magnetic_conduction(
            self.magnetic_field(state_),
            state_.auxiliary.material,
        )
        boundary_half = state_.auxiliary.boundary
        if self.pml is not None:
            _, boundary_half = self.pml.apply_magnetic(
                magnetic_forcing,
                boundary_half,
                half_step,
            )
        magnetic_half_flux = state_.primary.magnetic_flux + half_step * magnetic_forcing
        material_half = self.constitutive.advance_state(
            jnp.asarray(time),
            state_.auxiliary.material,
            state_.primary.electric_displacement,
            magnetic_half_flux,
            half_step,
            args,
        )
        magnetic_half = self.constitutive.magnetic_field(
            magnetic_half_flux,
            material_half,
        )
        current = self._current(jnp.asarray(time) + half_step, state_, args)
        electric_half = self.constitutive.electric_field(
            state_.primary.electric_displacement,
            material_half,
        )
        total_current = (
            current
            + self._boundary_current(electric_half)
            + self.constitutive.electric_conduction(electric_half, material_half)
        )
        electric_forcing = (
            self.plan.bridge.codifferential(2, magnetic_half) - total_current
        )
        if self.pml is not None:
            electric_forcing, boundary_half = self.pml.apply_electric(
                electric_forcing,
                boundary_half,
                dt,
            )
        displacement_new = state_.primary.electric_displacement + dt * electric_forcing
        charge_new = state_.primary.charge + dt * self.plan.bridge.codifferential(
            1, electric_forcing
        )
        material_new = self.constitutive.advance_state(
            jnp.asarray(time) + half_step,
            material_half,
            displacement_new,
            magnetic_half_flux,
            half_step,
            args,
        )
        electric_new = self.constitutive.electric_field(
            displacement_new,
            material_new,
        )
        magnetic_forcing_new = -self.plan.bridge.exterior_derivative(
            1, electric_new
        ) - self.constitutive.magnetic_conduction(
            self.constitutive.magnetic_field(magnetic_half_flux, material_new),
            material_new,
        )
        boundary_new = boundary_half
        if self.pml is not None:
            _, boundary_new = self.pml.apply_magnetic(
                magnetic_forcing_new,
                boundary_half,
                half_step,
            )
        magnetic_new = magnetic_half_flux + half_step * magnetic_forcing_new
        displacement_new, magnetic_new = self._constrain_primary(
            displacement_new,
            magnetic_new,
        )
        charge_new = self.plan.bridge.codifferential(
            1, displacement_new
        ) - self.electric_constraint(state_)
        provisional = self.pack(
            displacement_new,
            magnetic_new,
            charge_new,
            material_state=material_new,
            boundary_state=boundary_new,
            observations=state_.observations,
        )
        electric_observed = self.electric_field(provisional)
        magnetic_observed = self.magnetic_field(provisional)
        observation_state = tuple(
            observer.update(
                jnp.asarray(time) + dt,
                electric_observed,
                magnetic_observed,
                value,
            )
            for observer, value in zip(
                self.observers,
                tuple(state_.observations),
                strict=True,
            )
        )
        return self.pack(
            displacement_new,
            magnetic_new,
            charge_new,
            material_state=material_new,
            boundary_state=boundary_new,
            observations=observation_state,
        )

    def energy(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        return self.constitutive.energy(
            state_.primary.electric_displacement,
            state_.primary.magnetic_flux,
            state_.auxiliary.material,
            self.plan.bridge.cochain.hodge_metric(1),
            self.plan.bridge.cochain.hodge_metric(2),
        )

    def electric_constraint(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        return (
            self.plan.bridge.codifferential(
                1,
                state_.primary.electric_displacement,
            )
            - state_.primary.charge
        )

    def observe(self, state: CompatibleMaxwellState, /) -> tuple[Array, ...]:
        state_ = self._state(state)
        return tuple(
            observer.value(value)
            for observer, value in zip(
                self.observers,
                tuple(state_.observations),
                strict=True,
            )
        )

    def magnetic_constraint(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        return self.plan.bridge.exterior_derivative(2, state_.primary.magnetic_flux)

    def source_power(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self._state(state)
        current = self._current(jnp.asarray(time), state_, args)
        electric = self.electric_field(state_)
        return jnp.real(
            jnp.vdot(
                electric,
                self.plan.bridge.cochain.apply_hodge(1, current),
            )
        )

    def power_balance_residual(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self._state(state)
        current = self._current(jnp.asarray(time), state_, args)
        rates = self._rates_with_current(state_, current)
        energy_rate = self.constitutive.energy_rate(
            state_.primary.electric_displacement,
            state_.primary.magnetic_flux,
            rates.electric_displacement,
            rates.magnetic_flux,
            state_.auxiliary.material,
            self.plan.bridge.cochain.hodge_metric(1),
            self.plan.bridge.cochain.hodge_metric(2),
        )
        return (
            energy_rate
            + self.source_power(time, state_, args)
            + self.boundary_dissipation(state_)
            + self.pml_dissipation(state_)
            + self.material_dissipation(state_)
        )

    def gauss_rate_residual(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self._state(state)
        current = self._current(jnp.asarray(time), state_, args)
        rates = self._rates_with_current(state_, current)
        return (
            self.plan.bridge.codifferential(1, rates.electric_displacement) - rates.charge
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
        *,
        step_size: ArrayLike | None = None,
    ) -> CompatibleMaxwellDiagnostics:
        electric = self.electric_constraint(state)
        magnetic = self.magnetic_constraint(state)
        gauss_rate = self.gauss_rate_residual(time, state, args)
        fraction = None if step_size is None else jnp.asarray(step_size) / self.stable_dt
        return CompatibleMaxwellDiagnostics(
            energy=self.energy(state),
            electric_constraint_linf=jnp.max(jnp.abs(electric)),
            magnetic_constraint_linf=jnp.max(jnp.abs(magnetic)),
            gauss_rate_linf=jnp.max(jnp.abs(gauss_rate)),
            source_power=self.source_power(time, state, args),
            boundary_dissipation=self.boundary_dissipation(state),
            pml_dissipation=self.pml_dissipation(state),
            power_balance_residual=self.power_balance_residual(time, state, args),
            stable_step=self.stable_dt,
            step_fraction=fraction,
        )


__all__ = [
    "AbstractMaxwellConstitutivePlan",
    "AbstractPreparedMaxwellConstitutive",
    "CompatibleMaxwellDiagnostics",
    "CompatibleMaxwellPlan",
    "CompatibleMaxwellState",
    "DiagonalMaxwellConstitutivePlan",
    "MaxwellAuxiliaryState",
    "MaxwellCapabilities",
    "MaxwellPrimaryState",
    "PreparedCompatibleMaxwell",
    "PreparedDiagonalMaxwellConstitutive",
]

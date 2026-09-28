#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Full 3-D periodic compatible-cochain Maxwell as a PIC field solver."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._dtype_names import RealPrecisionDType
from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization.pic import (
    ChargeConservingCurrentPlan,
    PICMaxwellCurrentArguments,
    PICSpeciesPlan,
    PreparedPICParticleCochainTransfer,
)
from ._cochain_electrostatic import CochainElectrostaticPlan
from ._maxwell import (
    CompatibleMaxwellState,
    MaxwellPrimaryState,
    PreparedCompatibleMaxwell,
    PreparedDiagonalMaxwellConstitutive,
)
from ._maxwell_far_field import HuygensSurfacePhasors, PreparedMaxwellHuygensBox
from ._pic_current_source import PreparedPICMaxwellCurrentSource
from ._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    PICFieldAdvance,
    PICFieldDeposit,
    PICGaussProjectionResult,
    PICRestartComponent,
    PICTensorKind,
    PICTensorMap,
    restart_component,
    restore_component,
)


def _shift_without_wrap(value: Array, axis: int, cells: int, /) -> Array:
    shifted = jnp.roll(value, -cells, axis=axis)
    index: list[slice] = [slice(None)] * value.ndim
    index[axis] = slice(value.shape[axis] - cells, value.shape[axis])
    return shifted.at[tuple(index)].set(0.0)


class CochainMaxwellPICFieldSolver(AbstractPreparedPICFieldSolver, NonTrainableState):
    """Periodic 3-D compatible Maxwell with Whitney charge-conserving transfer.

    One prepared cochain transfer and charge-conserving current plan per
    species. Maxwell remains the sole owner of D, B, charge, material, observer,
    and CFL updates; the Gauss charge is the degree-zero cochain.
    """

    maxwell: PreparedCompatibleMaxwell
    electrostatic: CochainElectrostaticPlan
    transfers: tuple[PreparedPICParticleCochainTransfer, ...]
    currents: tuple[ChargeConservingCurrentPlan, ...]
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)

    def __init__(
        self,
        maxwell: PreparedCompatibleMaxwell,
        electrostatic: CochainElectrostaticPlan,
        transfers: Sequence[PreparedPICParticleCochainTransfer],
        currents: Sequence[ChargeConservingCurrentPlan],
        /,
    ) -> None:
        if not isinstance(maxwell, PreparedCompatibleMaxwell):
            raise TypeError("maxwell must be PreparedCompatibleMaxwell.")
        if not isinstance(electrostatic, CochainElectrostaticPlan):
            raise TypeError("electrostatic must be CochainElectrostaticPlan.")
        transfer_values = tuple(transfers)
        current_values = tuple(currents)
        if not transfer_values or len(transfer_values) != len(current_values):
            raise ValueError("One transfer and current plan is required per PIC species.")
        if any(
            not isinstance(value, PreparedPICParticleCochainTransfer)
            for value in transfer_values
        ) or any(
            not isinstance(value, ChargeConservingCurrentPlan) for value in current_values
        ):
            raise TypeError("PIC transfers and current plans have incompatible types.")
        bridge = maxwell.plan.bridge
        if bridge.dimension != 3 or any(
            not axis.periodic for axis in bridge.grid.structured_axes
        ):
            raise ValueError("Cochain PIC field solver requires a periodic 3-D grid.")
        if electrostatic.bridge.bridge_id != bridge.bridge_id or any(
            value.bridge.bridge_id != bridge.bridge_id for value in transfer_values
        ):
            raise ValueError(
                "PIC electrostatic, transfer, and Maxwell plans must share one bridge."
            )
        if any(
            current.transfer.prepared_id != transfer.prepared_id
            for current, transfer in zip(current_values, transfer_values, strict=True)
        ):
            raise ValueError("Every current plan must use its matching PIC transfer.")
        if not any(
            isinstance(source, PreparedPICMaxwellCurrentSource)
            for source in maxwell.sources
        ):
            raise ValueError(
                "Maxwell PIC requires CompatibleMaxwellPlan(sources=(PICMaxwellCurrentSourcePlan(),))."
            )
        if maxwell.boundaries:
            raise ValueError(
                "The periodic 3-D cochain PIC field solver admits no Maxwell boundary."
            )
        if (
            maxwell.capabilities.dispersive
            or maxwell.capabilities.nonlinear
            or not maxwell.capabilities.passive
        ):
            raise ValueError(
                "Electromagnetic PIC requires passive instantaneous Maxwell material."
            )
        self.maxwell = maxwell
        self.electrostatic = electrostatic
        self.transfers = transfer_values
        self.currents = current_values
        self.spatial_dimension = 3
        self.field_dtype = "float64"
        self.solver_id = canonical_fingerprint(
            {
                "kind": "cochain-maxwell-pic-field-solver",
                "maxwell": maxwell.prepared_id,
                "electrostatic": electrostatic.plan_id,
                "transfers": [value.prepared_id for value in transfer_values],
                "currents": [value.plan_id for value in current_values],
            }
        )

    @property
    def stable_step(self) -> Array:
        return self.maxwell.stable_dt

    @property
    def displacement_widths(self) -> Array:
        return jnp.stack(
            tuple(
                jnp.min(axis.interval_widths)
                for axis in self.maxwell.plan.bridge.grid.structured_axes
            )
        )

    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        if len(species) != len(self.transfers):
            raise ValueError("Cochain PIC requires one prepared transfer per species.")
        for value, transfer in zip(species, self.transfers, strict=True):
            if (
                value.population.particles.prepared_id
                != transfer.species.particles.prepared_id
            ):
                raise ValueError(
                    "Species population and cochain transfer use different particles."
                )

    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        del species
        axes = self.maxwell.plan.bridge.grid.structured_axes
        slot = np.arange(capacity)
        start = np.stack(
            tuple(
                np.asarray(axis.bounds[0])
                + (slot % axis.interval_centers.size + 0.7)
                * np.asarray(axis.interval_widths[0])
                for axis in axes
            ),
            axis=-1,
        )
        step = np.stack(
            tuple(
                fraction * np.asarray(axis.interval_widths[0])
                for fraction, axis in zip((0.5, 0.2, -0.1), axes, strict=True)
            )
        )
        return jnp.asarray(start), jnp.asarray(start + step)

    def field_with_charge(self, charge: Array, /) -> CompatibleMaxwellState:
        counts = self.maxwell.primary_counts
        return self.maxwell.pack(
            jnp.zeros((counts[0],), dtype=charge.dtype),
            jnp.zeros((counts[1],), dtype=charge.dtype),
            charge,
            material_state=self.maxwell.constitutive.initialize_state(),
        )

    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[CompatibleMaxwellState, Array]:
        electrostatic = self.electrostatic.solve(charge)
        displacement = self.maxwell.constitutive.electric_displacement(
            electrostatic.electric,
            self.maxwell.constitutive.initialize_state(),
        )
        flux = (
            jnp.zeros((self.maxwell.primary_counts[1],), dtype=displacement.dtype)
            if magnetic is None
            else jnp.asarray(magnetic, dtype=displacement.dtype)
        )
        field = self.maxwell.pack(
            displacement,
            flux,
            charge,
            material_state=self.maxwell.constitutive.initialize_state(),
        )
        return field, electrostatic.successful

    def field_charge(self, field: CompatibleMaxwellState, /) -> Array:
        return field.primary.charge

    def field_energy(self, field: CompatibleMaxwellState, /) -> Array:
        return self.maxwell.energy(field)

    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        transfer = self.transfers[species]
        deposit = transfer.deposit_macrocharge(
            transfer.build(position, active_mask=active), macrocharge
        )
        return deposit.cochain, deposit.successful

    def deposit(
        self,
        species: int,
        start: Array,
        end: Array,
        velocity: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        del velocity
        result = self.currents[species].deposit(
            start, end, step_size, macrocharge=macrocharge, active_mask=active
        )
        return PICFieldDeposit(
            result.current,
            result.start_charge.cochain,
            result.end_charge.cochain,
            result.maximum_continuity_defect,
            result.successful,
        )

    def advance(
        self,
        time: Array,
        field: CompatibleMaxwellState,
        current: Array,
        step_size: Array,
        /,
    ) -> PICFieldAdvance:
        args = PICMaxwellCurrentArguments(current, None)
        candidate = self.maxwell.leapfrog_step(time, field, step_size, args)
        diagnostics = self.maxwell.diagnostics(time + step_size, candidate, args)
        finite = jnp.all(jnp.isfinite(candidate.primary.electric_displacement)) & jnp.all(
            jnp.isfinite(candidate.primary.magnetic_flux)
        )
        return PICFieldAdvance(
            candidate,
            candidate.primary.charge,
            diagnostics.electric_constraint_linf,
            diagnostics.magnetic_constraint_linf,
            self.maxwell.energy(candidate),
            diagnostics,
            finite & (step_size <= self.maxwell.stable_dt),
        )

    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: CompatibleMaxwellState,
        /,
    ) -> tuple[Array, Array, Array]:
        transfer = self.transfers[species]
        routes = transfer.build(position, active_mask=active)
        electric = transfer.gather_electric(routes, self.maxwell.electric_field(field))
        magnetic = transfer.gather_magnetic(routes, self.maxwell.magnetic_field(field))
        return electric.values, magnetic.values, electric.support & magnetic.support

    @property
    def tensor_periodic(self) -> tuple[bool, ...]:
        return tuple(
            bool(axis.periodic) for axis in self.maxwell.plan.bridge.grid.structured_axes
        )

    def _map_cochain(self, degree: int, value: Array, function: PICTensorMap, /) -> Array:
        # Oriented components of degree k are intervals along their orientation
        # axes; those axes carry the odd mirror parity (polar edges, axial faces).
        bridge = self.maxwell.plan.bridge
        mapped = []
        for orientation, component in zip(
            bridge.orientations[degree], bridge.unpack(degree, value), strict=True
        ):
            location = tuple(
                "interval" if axis in orientation else "point"
                for axis in range(bridge.dimension)
            )
            parity = tuple(
                -1 if axis in orientation else 1 for axis in range(bridge.dimension)
            )
            mapped.append(function(component, location, parity))
        return bridge.pack(degree, tuple(mapped))

    def tensor_template(self, kind: PICTensorKind, /) -> Any:
        counts = self.maxwell.plan.bridge.cochain.cell_counts
        match kind:
            case "charge":
                return jnp.zeros((counts[0],), dtype=jnp.float64)
            case "current":
                return jnp.zeros((counts[1],), dtype=jnp.float64)
            case "field":
                return self.field_with_charge(jnp.zeros((counts[0],), dtype=jnp.float64))
            case _:
                raise ValueError("PIC tensor kind is invalid.")

    def map_tensors(
        self, kind: PICTensorKind, value: Any, function: PICTensorMap, /
    ) -> Any:
        """Map charge, current, or the gathered ``E``/``H`` cochains of a field.

        Field maps act on the physical ``E`` and ``H`` the gather interpolates and
        return ``D``/``B`` through the instantaneous constitutive law; charge,
        auxiliary, and observer state are unchanged.
        """
        match kind:
            case "charge":
                return self._map_cochain(0, value, function)
            case "current":
                return self._map_cochain(1, value, function)
            case "field":
                constitutive = self.maxwell.constitutive
                material = value.auxiliary.material
                electric = self._map_cochain(
                    1,
                    constitutive.electric_field(
                        value.primary.electric_displacement, material
                    ),
                    function,
                )
                magnetic = self._map_cochain(
                    2,
                    constitutive.magnetic_field(value.primary.magnetic_flux, material),
                    function,
                )
                return CompatibleMaxwellState(
                    MaxwellPrimaryState(
                        constitutive.electric_displacement(electric, material),
                        constitutive.magnetic_flux(magnetic, material),
                        value.primary.charge,
                    ),
                    value.auxiliary,
                    value.observations,
                )
            case _:
                raise ValueError("PIC tensor kind is invalid.")

    def project_gauss(
        self, field: CompatibleMaxwellState, charge: Array, /
    ) -> PICGaussProjectionResult:
        """Cochain Poisson projection onto the degree-zero Gauss charge ``charge``.

        The residual ``ρ + δD`` is solved by the prepared electrostatic plan
        (``-δ(ε E_c) = ρ + δD``, ``E_c = -dφ``); adding ``D_c = ε E_c`` leaves
        ``B``, material memory, and observers untouched.
        """
        maxwell = self.maxwell
        target = CompatibleMaxwellState(
            MaxwellPrimaryState(
                field.primary.electric_displacement, field.primary.magnetic_flux, charge
            ),
            field.auxiliary,
            field.observations,
        )
        residual = maxwell.electric_constraint(target)
        correction = self.electrostatic.solve(-residual)
        displacement = maxwell.constitutive.electric_displacement(
            correction.electric, field.auxiliary.material
        )
        projected = CompatibleMaxwellState(
            MaxwellPrimaryState(
                field.primary.electric_displacement + displacement,
                field.primary.magnetic_flux,
                charge,
            ),
            field.auxiliary,
            field.observations,
        )
        after = maxwell.electric_constraint(projected)
        return PICGaussProjectionResult(
            projected,
            jnp.max(jnp.abs(residual), initial=0.0),
            jnp.max(jnp.abs(after), initial=0.0),
            maxwell.energy(projected) - maxwell.energy(field),
            correction.successful & jnp.all(jnp.isfinite(after)),
            "cochain-poisson",
        )

    def dispersion_frequency(
        self, wavevector: ArrayLike, step_size: ArrayLike, /
    ) -> Array:
        """Yee vacuum dispersion ``sin(ωΔt/2) = cΔt |Σ sin²(k_iΔ_i/2)/Δ_i²|^{1/2}``.

        Defined for homogeneous diagonal material on the uniform periodic grid;
        heterogeneous material has no single symbol and is refused.
        """
        constitutive = self.maxwell.constitutive
        if not isinstance(constitutive, PreparedDiagonalMaxwellConstitutive):
            raise ValueError("The cochain spectral symbol requires diagonal material.")
        epsilon = np.asarray(constitutive.permittivity)
        mu = np.asarray(constitutive.permeability)
        if np.ptp(epsilon) != 0.0 or np.ptp(mu) != 0.0:
            raise ValueError("The cochain spectral symbol requires homogeneous material.")
        dt = float(jnp.asarray(step_size))
        if not 0.0 < dt <= float(self.maxwell.stable_dt):
            raise ValueError("step_size must be positive and within the stable step.")
        k = jnp.asarray(wavevector, dtype=jnp.float64)
        if k.ndim != 2 or k.shape[1] != 3:
            raise ValueError("wavevector must have shape (K, 3).")
        spacing = jnp.asarray(
            [
                np.asarray(axis.interval_widths)[0]
                for axis in self.maxwell.plan.bridge.grid.structured_axes
            ]
        )
        speed = 1.0 / np.sqrt(float(epsilon.flat[0]) * float(mu.flat[0]))
        spatial = jnp.sqrt(jnp.sum((jnp.sin(0.5 * k * spacing) / spacing) ** 2, axis=-1))
        return 2.0 / dt * jnp.arcsin(speed * dt * spatial)

    def huygens_phasors(
        self, field: CompatibleMaxwellState, /
    ) -> tuple[HuygensSurfacePhasors, ...]:
        return tuple(
            observer.surface_phasors(state)
            for observer, state in zip(
                self.maxwell.observers, tuple(field.observations), strict=True
            )
            if isinstance(observer, PreparedMaxwellHuygensBox)
        )

    def window_interval(self, axis: int, /) -> float:
        widths = np.asarray(
            self.maxwell.plan.bridge.grid.structured_axes[axis].interval_widths
        )
        if not np.allclose(widths, widths[0]):
            raise ValueError("Moving windows require a uniform window axis.")
        return float(widths[0])

    def window_bounds(self, axis: int, /) -> tuple[float, float]:
        bounds = self.maxwell.plan.bridge.grid.structured_axes[axis].bounds
        return float(bounds[0]), float(bounds[1])

    def _shift_cochain(
        self, degree: int, value: Array, axis: int, cells: int, /
    ) -> Array:
        bridge = self.maxwell.plan.bridge
        return bridge.pack(
            degree,
            tuple(
                _shift_without_wrap(component, axis, cells)
                for component in bridge.unpack(degree, value)
            ),
        )

    def shift_window(
        self, field: CompatibleMaxwellState, axis: int, cells: int, /
    ) -> CompatibleMaxwellState:
        counts = self.maxwell.plan.bridge.cochain.cell_counts

        def shift_leaf(value: Any) -> Any:
            # Auxiliary and observer arrays are shifted when they are full
            # cochains of one degree; other leaves are translation invariant.
            if not eqx.is_array(value):
                return value
            for degree in range(3):
                if value.shape == (counts[degree],):
                    return self._shift_cochain(degree, value, axis, cells)
            return value

        primary = MaxwellPrimaryState(
            self._shift_cochain(1, field.primary.electric_displacement, axis, cells),
            self._shift_cochain(2, field.primary.magnetic_flux, axis, cells),
            self._shift_cochain(0, field.primary.charge, axis, cells),
        )
        return CompatibleMaxwellState(
            primary,
            jax.tree.map(shift_leaf, field.auxiliary),
            jax.tree.map(shift_leaf, field.observations),
        )

    def restart_component(self, field: CompatibleMaxwellState, /) -> PICRestartComponent:
        return restart_component("field", self.solver_id, field)

    def restore_component(
        self, component: PICRestartComponent, /
    ) -> CompatibleMaxwellState:
        template = self.field_with_charge(
            jnp.zeros((self.maxwell.primary_counts[2],), dtype=jnp.float64)
        )
        return restore_component(component, "field", self.solver_id, template)


__all__ = ["CochainMaxwellPICFieldSolver"]

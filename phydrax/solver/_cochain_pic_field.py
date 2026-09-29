#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Full 3-D compatible-cochain Maxwell as a PIC field solver.

Periodic and wall-bounded axes, Maxwell boundaries (PEC/PMC/impedance), CPML
absorbers, and every linear passive medium of the compatible Maxwell runtime —
lossy conductors, Lorentz–Drude electric and magnetic poles (negative index),
and magnetized cold plasma — drive self-consistent PIC. The solver reports the
field/material energy split and the source-free loss power
(`PICEnergyAccounting`) the PIC energy ledger integrates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, assert_never

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
from ..linalg import LinearSolvePolicy, TolerancePolicy
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
    PICCapabilityRecord,
    PICFieldAdvance,
    PICFieldDeposit,
    PICFieldEnergy,
    PICFieldSolverCapability,
    PICGaussProjectionResult,
    PICRelativisticFieldResult,
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
    """3-D compatible Maxwell with spline-Whitney charge-conserving transfer.

    One prepared cochain transfer and charge-conserving current plan per
    species. Maxwell remains the sole owner of D, B, charge, material, CPML,
    boundary, observer, and CFL updates; the Gauss charge is the degree-zero
    cochain. Its charge follows ``−δD`` and so also carries the induced charge
    of conducting boundaries, conduction or plasma charge of the medium, and
    the CPML bookkeeping divergence; `advance` reports the current-driven part.

    The electrostatic plan initializes Gauss-consistent fields: periodic grids
    use its periodic boundary, bounded grids a Dirichlet (grounded) or mixed
    boundary whose fixed vertices carry the induced charge ``−δD``. Its
    permittivity must equal the medium's instantaneous electric response.
    """

    maxwell: PreparedCompatibleMaxwell
    electrostatic: CochainElectrostaticPlan
    transfers: tuple[PreparedPICParticleCochainTransfer, ...]
    currents: tuple[ChargeConservingCurrentPlan, ...]
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    wall_widths: tuple[float, ...] = eqx.field(static=True)
    grounded: bool = eqx.field(static=True)

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
        if bridge.dimension != 3:
            raise ValueError("Cochain PIC field solver requires a 3-D grid.")
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
        capabilities = maxwell.capabilities
        if capabilities.nonlinear or capabilities.active or not capabilities.passive:
            raise ValueError(
                "Self-consistent electromagnetic PIC requires a linear passive "
                "Maxwell medium."
            )
        constitutive = maxwell.constitutive
        instantaneous = constitutive.electric_displacement(
            jnp.ones((maxwell.layout.electric_count,), dtype=jnp.float64),
            constitutive.initialize_state(),
        )
        if not np.allclose(
            np.asarray(instantaneous),
            np.asarray(electrostatic.permittivity),
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "The electrostatic permittivity must equal the Maxwell medium's "
                "instantaneous electric response, or the initial field would not "
                "carry the particle charge."
            )
        self.maxwell = maxwell
        self.electrostatic = electrostatic
        self.transfers = transfer_values
        self.currents = current_values
        self.spatial_dimension = 3
        self.field_dtype = "float64"
        axes = bridge.grid.structured_axes
        self.periodic = tuple(bool(axis.periodic) for axis in axes)
        self.lower = tuple(float(axis.bounds[0]) for axis in axes)
        self.upper = tuple(float(axis.bounds[1]) for axis in axes)
        self.wall_widths = tuple(
            float(np.max(np.asarray(axis.interval_widths)[[0, -1]])) for axis in axes
        )
        self.grounded = bool(np.any(np.asarray(electrostatic.boundary.dirichlet_mask)))
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
    def pic_configuration(self) -> str:
        return "cochain-3d"

    def _homogeneous_diagonal(self) -> bool:
        constitutive = self.maxwell.constitutive
        return (
            isinstance(constitutive, PreparedDiagonalMaxwellConstitutive)
            and np.ptp(np.asarray(constitutive.permittivity)) == 0.0
            and np.ptp(np.asarray(constitutive.permeability)) == 0.0
        )

    def _zero_grounded(self) -> bool:
        boundary = self.electrostatic.boundary
        return not (
            boundary.gauge_required
            or bool(np.any(np.asarray(boundary.dirichlet_values) != 0.0))
            or bool(np.any(np.asarray(boundary.neumann_source) != 0.0))
        )

    def pic_capability(
        self, capability: PICFieldSolverCapability, /
    ) -> PICCapabilityRecord:
        route, refusal = PICCapabilityRecord.route, PICCapabilityRecord.refusal
        match capability:
            case "tensor-layout":
                return route(
                    capability,
                    "Oriented cochain components with mirror parities (PICFilterPlan).",
                )
            case "spectral-symbol":
                if self._homogeneous_diagonal():
                    return route(
                        capability, "Yee vacuum dispersion within the stable step."
                    )
                return refusal(
                    capability,
                    "Heterogeneous or non-diagonal material has no single symbol.",
                    published=True,
                )
            case "huygens-sampling":
                if any(
                    isinstance(value, PreparedMaxwellHuygensBox)
                    for value in self.maxwell.observers
                ):
                    return route(capability, "Phasors of the Maxwell Huygens boxes.")
                # Maxwell refuses Huygens boxes beside the dynamic PIC current.
                return refusal(
                    capability,
                    "Maxwell refuses Huygens boxes with dynamic PIC currents (J = 0 "
                    "cannot be certified on the surface), so no phasors exist.",
                    published=True,
                )
            case "multi-deposit":
                return refusal(
                    capability, "Species deposit through their own cochain transfers."
                )
            case "window-shift":
                return route(
                    capability,
                    "Integer-cell cochain, auxiliary, and observer translation along "
                    "a uniform axis.",
                )
            case "galilean-grid":
                return refusal(capability, "The Yee leapfrog grid is lab-fixed.")
            case "energy-accounting":
                return route(
                    capability,
                    "Leapfrog-corrected field and medium energy with the source-free "
                    "loss power.",
                )
            case "open-domain":
                return route(
                    capability,
                    f"Periodic axes {self.periodic}; bounded axes with the spline "
                    "stencil wall inset.",
                )
            case "restart-state":
                return route(
                    capability,
                    "Field cochains with medium, CPML, and observer memory, admitted "
                    "by solver identity.",
                )
            case "gauss-projection":
                return route(capability, "Cochain Poisson projection (cochain-poisson).")
            case "relativistic-self-fields":
                if self._zero_grounded():
                    return route(
                        capability,
                        "Anisotropic Poisson solve per species drifting along one axis.",
                    )
                return refusal(
                    capability,
                    "Superposed Coulomb fields require a zero-valued grounded "
                    "electrostatic boundary; this boundary is gauged or sourced.",
                    published=True,
                )
            case _:
                assert_never(capability)

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
        """Probe paths through every periodic cell and two central bounded cells.

        Bounded axes keep the paths central, away from wall traces, absorbers,
        and the stencil limits of higher shape orders.
        """
        del species
        axes = self.maxwell.plan.bridge.grid.structured_axes
        slot = np.arange(capacity)

        def cell(axis_count: int, periodic: bool) -> np.ndarray:
            return slot % axis_count if periodic else axis_count // 2 - 1 + slot % 2

        start = np.stack(
            tuple(
                np.asarray(axis.bounds[0])
                + (cell(axis.interval_centers.size, bool(axis.periodic)) + 0.7)
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
        return self._with_induced_charge(field, charge), electrostatic.successful

    def _axis_edges(self, axis: int, value: ArrayLike, /) -> Array:
        """Edge cochain equal to ``value`` (broadcast) on ``axis`` edges, else zero."""
        bridge = self.maxwell.plan.bridge
        zeros = bridge.unpack(1, jnp.zeros((bridge.cochain.cell_counts[1],)))
        return bridge.pack(
            1,
            tuple(
                jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), component.shape)
                if index == axis
                else component
                for index, component in enumerate(zeros)
            ),
        )

    def initialize_relativistic_field(
        self,
        charges: tuple[Array, ...],
        drifts: tuple[tuple[float, ...], ...],
        speed_of_light: float,
        /,
    ) -> PICRelativisticFieldResult:
        """Superposed boosted-Coulomb fields of species drifting along grid axes.

        The anisotropic Poisson operator of a drift along axis ``∥`` is the
        electrostatic one with ``ε/γ²`` on the ``∥`` edges, so each species'
        ``E_s`` carries its charge to the certified solve tolerance.
        ``B = d(A)`` with the edge vector potential ``A = (β/c)φ`` (trapezoid
        of ``φ`` along each ``∥`` edge) is divergence-free by ``dd = 0`` and
        equals ``β × E/c`` to second order. Drifts are host values; each
        drifting species prepares its own Poisson solve with the electrostatic
        plan's tolerance and linear policy.
        """
        if len(charges) != len(drifts):
            raise ValueError("One charge and one drift are required per species.")
        boundary = self.electrostatic.boundary
        if (
            boundary.gauge_required
            or bool(np.any(np.asarray(boundary.dirichlet_values) != 0.0))
            or bool(np.any(np.asarray(boundary.neumann_source) != 0.0))
        ):
            raise ValueError(
                "Relativistic self-fields superpose per-species Coulomb fields and "
                "require a grounded (zero-valued Dirichlet or mixed) electrostatic "
                "boundary without Neumann sources."
            )
        light = float(speed_of_light)
        if not np.isfinite(light) or light <= 0.0:
            raise ValueError("speed_of_light must be positive and finite.")
        maxwell = self.maxwell
        bridge = maxwell.plan.bridge
        measures = bridge.unpack(1, bridge.cochain.primal_measures[1])
        electric = jnp.zeros((maxwell.layout.electric_count,), dtype=jnp.float64)
        potential = jnp.zeros_like(electric)
        total = jnp.zeros((bridge.cochain.cell_counts[0],), dtype=jnp.float64)
        successful = jnp.asarray(True)
        for charge, drift in zip(charges, drifts, strict=True):
            beta = np.asarray(drift, dtype=np.float64)
            if beta.shape != (3,) or not np.all(np.isfinite(beta)):
                raise ValueError("Each drift must be a finite 3-vector β = v/c.")
            speed = float(np.linalg.norm(beta))
            if speed >= 1.0:
                raise ValueError("Drifts must be subluminal.")
            total = total + charge
            if speed == 0.0:
                result = self.electrostatic.solve(charge)
                electric = electric + result.electric
                successful = successful & result.successful
                continue
            axis = int(np.argmax(np.abs(beta)))
            if np.any(np.abs(np.delete(beta, axis)) > 1.0e-12 * speed):
                raise ValueError(
                    "Relativistic self-fields require each drift along one grid axis."
                )
            # 1/γ² = 1 − β² on the edges along the drift. The operator's
            # condition number grows by γ², so Krylov iterations grow by γ.
            contraction = 1.0 - speed**2
            scale = jnp.where(self._axis_edges(axis, 1.0) > 0.0, contraction, 1.0)
            policy = self.electrostatic.linear_policy
            steps = policy.tolerance.max_steps
            plan = CochainElectrostaticPlan(
                bridge,
                boundary,
                permittivity=self.electrostatic.permittivity * scale,
                tolerance=self.electrostatic.tolerance,
                compatibility_tolerance=self.electrostatic.compatibility_tolerance,
                linear_policy=LinearSolvePolicy(
                    policy.method,
                    tolerance=TolerancePolicy(
                        relative=policy.tolerance.relative,
                        absolute=policy.tolerance.absolute,
                        max_steps=None
                        if steps is None
                        else steps * math.ceil(contraction**-0.5),
                    ),
                    rank=policy.rank,
                    materialization=policy.materialization,
                    preconditioning=policy.preconditioning,
                    recycling=policy.recycling,
                    differentiation=policy.differentiation,
                    derivative_solve=policy.derivative_solve,
                    failure=policy.failure,
                    resources=policy.resources,
                    precision=policy.precision,
                    require_device_binding=policy.require_device_binding,
                ),
            )
            result = plan.solve(charge)
            electric = electric + scale * result.electric
            vertex = bridge.unpack(0, result.potential)[0]
            count = vertex.shape[axis]
            following = (
                jnp.roll(vertex, -1, axis=axis)
                if self.periodic[axis]
                else jnp.take(vertex, jnp.arange(1, count), axis=axis)
            )
            leading = (
                vertex
                if self.periodic[axis]
                else jnp.take(vertex, jnp.arange(count - 1), axis=axis)
            )
            potential = potential + self._axis_edges(
                axis, (beta[axis] / light) * 0.5 * (leading + following) * measures[axis]
            )
            successful = successful & result.successful
        constitutive = maxwell.constitutive
        material = constitutive.initialize_state()
        flux = bridge.exterior_derivative(1, potential)
        field = self._with_induced_charge(
            maxwell.pack(
                constitutive.electric_displacement(electric, material),
                flux,
                total,
                material_state=material,
            ),
            total,
        )
        gauss = jnp.max(jnp.abs(maxwell.electric_constraint(field)), initial=0.0)
        divergence = jnp.max(jnp.abs(bridge.exterior_derivative(2, flux)), initial=0.0)
        finite = (
            jnp.all(jnp.isfinite(electric))
            & jnp.all(jnp.isfinite(flux))
            & jnp.isfinite(gauss)
        )
        return PICRelativisticFieldResult(field, gauss, divergence, successful & finite)

    def _with_induced_charge(
        self, field: CompatibleMaxwellState, charge: Array, /
    ) -> CompatibleMaxwellState:
        """Charge ``charge`` on free vertices and ``−δD`` on fixed-potential ones.

        Grounded (Dirichlet) vertices of a bounded grid hold the induced charge
        of the grounded wall, so Gauss's law holds on every vertex.
        """
        if not self.grounded:
            return field
        mask = self.electrostatic.boundary.dirichlet_mask
        gauss = -self.maxwell.plan.bridge.codifferential(
            1, field.primary.electric_displacement
        )
        return CompatibleMaxwellState(
            MaxwellPrimaryState(
                field.primary.electric_displacement,
                field.primary.magnetic_flux,
                jnp.where(mask, gauss, charge),
            ),
            field.auxiliary,
            field.observations,
        )

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
            result.continuity_scale,
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
        # Continuity of the deposited current alone: ρ̇ = δJ (δ = −div).
        charge = (
            field.primary.charge
            + step_size
            * self.maxwell.plan.bridge.codifferential(
                self.maxwell.layout.electric_degree, current
            )
        )
        return PICFieldAdvance(
            candidate,
            charge,
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
        projected = self._with_induced_charge(
            CompatibleMaxwellState(
                MaxwellPrimaryState(
                    field.primary.electric_displacement + displacement,
                    field.primary.magnetic_flux,
                    charge,
                ),
                field.auxiliary,
                field.observations,
            ),
            charge,
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

    @property
    def domain_periodic(self) -> tuple[bool, ...]:
        return self.periodic

    @property
    def domain_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self.lower, self.upper

    def boundary_inset(self, species: int, /) -> tuple[float, ...]:
        """Wall distance keeping the degree-``p`` spline stencil on the grid.

        Order one deposits in-box paths exactly up to the wall; orders two and
        three need ``(p + 1)/2`` wall cells for their vertex stencil.
        """
        order = self.transfers[species].plan.shape_order
        cells = 0.0 if order == 1 else 0.5 * (order + 1)
        return tuple(cells * width for width in self.wall_widths)

    def energy_components(
        self, field: CompatibleMaxwellState, step_size: Array, /
    ) -> PICFieldEnergy:
        """Instantaneous-response field energy, leapfrog-corrected, and medium energy.

        ``½⟨E, ⋆(∂D/∂E)E⟩`` and ``½⟨H, ⋆(∂B/∂H)H⟩`` use the constitutive
        law's instantaneous response at the held medium state; the medium
        energy is the rest of the constitutive energy. The magnetic term
        subtracts the leapfrog half-kick energy (`leapfrog_energy`).
        """
        maxwell = self.maxwell
        constitutive = maxwell.constitutive
        material = field.auxiliary.material
        layout = maxwell.layout
        cochain = maxwell.plan.bridge.cochain
        electric = maxwell.electric_field(field)
        magnetic = maxwell.magnetic_field(field)
        _, displacement = jax.jvp(
            lambda value: constitutive.electric_displacement(value, material),
            (electric,),
            (electric,),
        )
        _, flux = jax.jvp(
            lambda value: constitutive.magnetic_flux(value, material),
            (magnetic,),
            (magnetic,),
        )
        electric_energy = 0.5 * jnp.real(
            jnp.vdot(electric, cochain.apply_hodge(layout.electric_degree, displacement))
        )
        magnetic_energy = 0.5 * jnp.real(
            jnp.vdot(magnetic, cochain.apply_hodge(layout.magnetic_degree, flux))
        )
        total = maxwell.energy(field)
        leapfrog = maxwell.leapfrog_energy(field, step_size)
        return PICFieldEnergy(
            electric_energy,
            magnetic_energy - (total - leapfrog),
            total - electric_energy - magnetic_energy,
        )

    def loss_power(self, field: CompatibleMaxwellState, /) -> Array:
        return self.maxwell.loss_power(field)

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
        record = self.pic_capability("huygens-sampling")
        if not record.admitted:
            raise ValueError(record.basis)
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Dry, compressible column and vertical-slice reference dynamics in SI units."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import TensorGridPlan, UniformCellAxisSpec
from ...discretization._fv_precision import FiniteVolumePrecisionPolicy
from ...discretization.finite_volume._atmospheric_balance import (
    PreparedAtmosphericBalance,
)
from ...discretization.finite_volume._structured import FiniteVolumePlan
from ...equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from ...equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
)
from ...equations._gas_dynamics import HomogeneousMixtureEulerSystem
from ...equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    ZeroResidualHelmholtzTerm,
)
from ...solver._finite_volume_content import FiniteVolumeConservativeContentState
from ...solver._fixed_step import AbstractFixedStepMethod, FixedStepResult


class DryAir(StrictModule, NonTrainableState):
    """Explicit calorically perfect N2/O2/Ar dry surrogate, not moist air.

    Default mole amounts 0.78084/0.20946/0.00934 are normalized after excluding
    trace gases. Molar masses are in kg/mol; diatomics have cv=5R/2 and argon
    cv=3R/2. Internal energies use the cv*T zero, without formation chemistry.
    The declared operating interval is 150--400 K. Condensed phases belong in
    a multiphase thermodynamic model, not in this all-gas Euler schema.
    """

    thermodynamics: HomogeneousHelmholtzPlan
    mole_fractions: Array
    mass_fractions: Array
    gas_constant: float = eqx.field(static=True)
    heat_capacity_pressure: float = eqx.field(static=True)
    air_id: str = eqx.field(static=True)

    def __init__(self, mole_fractions: Sequence[float] = (0.78084, 0.20946, 0.00934)):
        fraction = np.asarray(mole_fractions, dtype=float)
        if (
            fraction.shape != (3,)
            or np.any(~np.isfinite(fraction))
            or np.any(fraction <= 0.0)
        ):
            raise ValueError("Dry-air mole amounts must be three finite positive values.")
        fraction = fraction / np.sum(fraction)
        masses = np.asarray((0.0280134, 0.0319988, 0.039948))
        cv = UNIVERSAL_GAS_CONSTANT * np.asarray((2.5, 2.5, 1.5))
        schema = ChemicalSpeciesSchema.from_unique_species(
            ("N2", "O2", "Ar"),
            (ChemicalPhaseKind.GAS,) * 3,
            masses,
            ("N", "O", "Ar"),
            np.diag((2, 2, 1)).astype(np.int32),
            np.zeros(3, dtype=np.int32),
            gas_standard_pressure=100000.0,
        )
        calorics = PolynomialSpeciesThermodynamicsPlan(
            schema,
            cv,
            cv * 298.15,
            reference_temperature=298.15,
            minimum_temperature=150.0,
            maximum_temperature=400.0,
        )
        self.thermodynamics = HomogeneousHelmholtzPlan(
            IdealGasReferenceHelmholtzTerm(schema, calorics),
            ZeroResidualHelmholtzTerm(schema),
            minimum_molar_density=1.0e-8,
        )
        molar_mass = float(np.sum(fraction * masses))
        self.mole_fractions = jnp.asarray(fraction)
        self.mass_fractions = jnp.asarray(fraction * masses / molar_mass)
        self.gas_constant = UNIVERSAL_GAS_CONSTANT / molar_mass
        self.heat_capacity_pressure = float(
            np.sum(fraction * (cv + UNIVERSAL_GAS_CONSTANT)) / molar_mass
        )
        self.air_id = canonical_fingerprint(
            {
                "kind": "dry-nitrogen-oxygen-argon",
                "model": self.thermodynamics.model_id,
                "composition": fraction.tolist(),
            }
        )

    def system(self, dimension: int = 1) -> HomogeneousMixtureEulerSystem:
        return HomogeneousMixtureEulerSystem(
            self.thermodynamics, dimension, density_floor=1.0e-8, pressure_floor=1.0e-3
        )

    def conserved(
        self,
        system: HomogeneousMixtureEulerSystem,
        pressure: ArrayLike,
        temperature: ArrayLike,
        velocity: ArrayLike | None = None,
    ) -> Array:
        """Prepare ideal-mixture data through the native Helmholtz pressure/EOS.

        No caloric or conserved/primitive conversion is reimplemented here.
        Pressure is linear in molar density for the declared zero-residual EOS.
        """
        if system.thermodynamics.model_id != self.thermodynamics.model_id:
            raise ValueError("Dry-air initial data and Euler thermodynamics must agree.")
        pressure_, temperature_ = jnp.broadcast_arrays(
            jnp.asarray(pressure), jnp.asarray(temperature)
        )
        evaluation = self.thermodynamics.evaluate(
            temperature_, jnp.ones_like(temperature_), self.mole_fractions
        )
        density = (
            (pressure_ / evaluation.pressure)[..., None]
            * self.mole_fractions
            * self.thermodynamics.schema.molar_masses
        )
        motion = (
            jnp.zeros(temperature_.shape + (system.dimension,), dtype=temperature_.dtype)
            if velocity is None
            else jnp.broadcast_to(
                jnp.asarray(velocity), temperature_.shape + (system.dimension,)
            )
        )
        primitive = jnp.concatenate((density, motion, temperature_[..., None]), axis=-1)
        state = system.primitive_to_conserved(primitive)
        return eqx.error_if(
            state,
            jnp.any(~system.admissible(state)),
            "Dry atmospheric initial state is outside the declared thermodynamic domain.",
        )


class DryHydrostaticReference(StrictModule, NonTrainableState):
    family: str = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    pressure: float = eqx.field(static=True)
    height: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: str = "isothermal",
        *,
        temperature: float = 300.0,
        pressure: float = 100000.0,
        height: float = 0.0,
        gravity: float = 9.80665,
    ):
        if family not in ("isothermal", "isentropic"):
            raise ValueError("Hydrostatic family must be isothermal or isentropic.")
        values = (float(temperature), float(pressure), float(height), float(gravity))
        if (
            not all(np.isfinite(value) for value in values)
            or temperature <= 0.0
            or pressure <= 0.0
            or gravity <= 0.0
        ):
            raise ValueError(
                "Hydrostatic reference requires finite positive temperature, pressure, and gravity."
            )
        self.family = family
        self.temperature, self.pressure, self.height, self.gravity = values
        self.reference_id = canonical_fingerprint(
            {"kind": "dry-hydrostatic-reference", "family": family, "values": values}
        )

    def pressure_temperature(self, height: ArrayLike, air: DryAir) -> tuple[Array, Array]:
        z = jnp.asarray(height) - self.height
        if self.family == "isothermal":
            temperature = jnp.full_like(z, self.temperature)
            pressure = self.pressure * jnp.exp(
                -self.gravity * z / (air.gas_constant * self.temperature)
            )
        else:
            temperature = self.temperature - self.gravity * z / air.heat_capacity_pressure
            pressure = self.pressure * (temperature / self.temperature) ** (
                air.heat_capacity_pressure / air.gas_constant
            )
        return pressure, temperature

    def conserved(
        self, height: ArrayLike, air: DryAir, system: HomogeneousMixtureEulerSystem
    ) -> Array:
        pressure, temperature = self.pressure_temperature(height, air)
        return air.conserved(system, pressure, temperature)


class DryAtmosphereState(StrictModule):
    """Native FV content authority plus restart-complete accepted budget ledger."""

    content: FiniteVolumeConservativeContentState
    initial_integral: Array
    boundary_integral: Array
    source_integral: Array
    accepted_steps: Array
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def conserved(self) -> Array:
        return self.content.cell_average().reshape(
            self.cell_shape + self.content.component_shape
        )

    @property
    def time(self) -> Array:
        return self.content.time


class DryAtmosphereBudget(StrictModule):
    species_mass: Array
    momentum: Array
    gas_energy: Array
    potential_energy: Array
    total_energy: Array
    boundary_integral: Array
    source_integral: Array
    closure: Array


class DryAtmosphereStepResult(StrictModule):
    state: DryAtmosphereState
    accepted: Array
    stable_step: Array
    stage_valid: Array
    budget: DryAtmosphereBudget


class DryAtmosphereRolloutResult(StrictModule):
    state: DryAtmosphereState
    times: Array
    states: Array
    accepted: Array
    successful: Array
    budget: DryAtmosphereBudget


class DryAtmosphereRestart(StrictModule):
    state: DryAtmosphereState
    prepared_id: str = eqx.field(static=True)
    restart_id: str = eqx.field(static=True)


class DryAtmospherePlan(StrictModule, NonTrainableState):
    """Uniform Cartesian column (z) or vertical slice (x,z), SI coordinates.

    Boundary pairs are positive-axis lower/upper. Periodicity is horizontal
    only; closed means inviscid reflecting wall. Prescribed data are stationary
    exterior conserved states, one array per boundary face batch; they are not
    a normal flux. Omitted prescribed arrays use the hydrostatic reference.
    """

    air: DryAir
    reference: DryHydrostaticReference
    shape: tuple[int, ...] = eqx.field(static=True)
    bounds: tuple[tuple[float, ...], tuple[float, ...]] = eqx.field(static=True)
    boundaries: tuple[tuple[str, str], ...] = eqx.field(static=True)
    prescribed: tuple[tuple[Array | None, Array | None], ...]
    order: int = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        bounds: ArrayLike,
        *,
        air: DryAir | None = None,
        reference: DryHydrostaticReference | None = None,
        boundaries: Sequence[tuple[str, str]] | None = None,
        prescribed: Sequence[tuple[ArrayLike | None, ArrayLike | None]] | None = None,
        order: int = 2,
        cfl: float = 0.35,
    ):
        shape_ = tuple(int(n) for n in shape)
        bounds_ = np.asarray(bounds, dtype=float)
        if (
            len(shape_) not in (1, 2)
            or any(n < 2 for n in shape_)
            or bounds_.shape != (2, len(shape_))
            or np.any(~np.isfinite(bounds_))
            or np.any(bounds_[1] <= bounds_[0])
        ):
            raise ValueError(
                "Dry columns/slices need positive finite extents and at least two cells per axis."
            )
        pairs = (
            tuple(tuple(pair) for pair in boundaries)
            if boundaries is not None
            else (("periodic", "periodic"),) * (len(shape_) - 1) + (("closed", "closed"),)
        )
        if (
            len(pairs) != len(shape_)
            or any(
                len(pair) != 2
                or any(side not in ("periodic", "closed", "prescribed") for side in pair)
                or (("periodic" in pair) and pair != ("periodic", "periodic"))
                for pair in pairs
            )
            or "periodic" in pairs[-1]
        ):
            raise ValueError(
                "Boundary pairs must match axes; vertical gravity is not periodic."
            )
        if order not in (1, 2) or not np.isfinite(cfl) or not 0.0 < cfl <= 0.5:
            raise ValueError("Dry atmosphere requires order one/two and 0 < CFL <= 0.5.")
        data = (
            tuple((None, None) for _ in shape_)
            if prescribed is None
            else tuple(
                tuple(None if side is None else jnp.asarray(side) for side in pair)
                for pair in prescribed
            )
        )
        if len(data) != len(shape_) or any(len(pair) != 2 for pair in data):
            raise ValueError(
                "Prescribed boundary data must have one lower/upper pair per axis."
            )
        if any(
            data[axis][side] is not None and pairs[axis][side] != "prescribed"
            for axis in range(len(shape_))
            for side in range(2)
        ):
            raise ValueError(
                "Exterior data may only be attached to prescribed boundaries."
            )
        self.air = DryAir() if air is None else air
        self.reference = DryHydrostaticReference() if reference is None else reference
        if not isinstance(self.air, DryAir) or not isinstance(
            self.reference, DryHydrostaticReference
        ):
            raise TypeError(
                "Dry atmosphere requires DryAir and DryHydrostaticReference descriptors."
            )
        self.shape, self.bounds, self.boundaries = (
            shape_,
            tuple(tuple(float(x) for x in row) for row in bounds_),
            pairs,
        )
        self.prescribed, self.order, self.cfl = data, int(order), float(cfl)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dry-atmosphere",
                "air": self.air.air_id,
                "reference": self.reference.reference_id,
                "shape": shape_,
                "bounds": self.bounds,
                "boundaries": pairs,
                "prescribed": array_tree_fingerprint(data),
                "order": self.order,
                "cfl": self.cfl,
            }
        )

    def prepare(self) -> PreparedDryAtmosphere:
        system = self.air.system(len(self.shape))
        grid = TensorGridPlan(
            tuple(
                UniformCellAxisSpec(n, periodic=pair[0] == "periodic")
                for n, pair in zip(self.shape, self.boundaries, strict=True)
            ),
            axis_names=("z",) if len(self.shape) == 1 else ("x", "z"),
        ).prepare(jnp.asarray(self.bounds))
        geometry = FiniteVolumePlan(
            grid, field_name="dry_atmosphere", component_names=system.component_names
        ).prepare()
        # Eight-point vertical Gauss integration of the analytic reference gives
        # actual FV averages, rather than pretending point samples are averages.
        nodes, weights = np.polynomial.legendre.leggauss(8)
        dz = (self.bounds[1][-1] - self.bounds[0][-1]) / self.shape[-1]
        z = geometry.cell_centers[..., -1, None] + 0.5 * dz * jnp.asarray(nodes)
        samples = self.reference.conserved(z, self.air, system)
        reference = jnp.sum(samples * jnp.asarray(weights)[..., None], axis=-2) * 0.5
        # Match the content-to-average round trip used by the runtime exactly.
        reference = (
            reference
            * geometry.cell_volumes[..., None]
            / geometry.cell_volumes[..., None]
        )
        faces, data = [], []
        for axis, points in enumerate(geometry.face_centers):
            face = self.reference.conserved(points[..., -1], self.air, system)
            if self.boundaries[axis][0] == "periodic":
                face = jnp.concatenate(
                    (face, jnp.take(face, jnp.asarray([0]), axis=axis)), axis=axis
                )
            faces.append(face)
            pair = []
            for side, index in enumerate((0, -1)):
                supplied = self.prescribed[axis][side]
                if self.boundaries[axis][side] == "prescribed":
                    template = jnp.take(face, index, axis=axis)
                    exterior = (
                        template
                        if supplied is None
                        else jnp.broadcast_to(supplied, template.shape)
                    )
                    exterior = eqx.error_if(
                        exterior,
                        jnp.any(~system.admissible(exterior)),
                        "Prescribed atmosphere is thermodynamically inadmissible.",
                    )
                    pair.append(exterior)
                else:
                    pair.append(None)
            data.append(tuple(pair))
        balance = PreparedAtmosphericBalance(
            system,
            geometry,
            reference,
            tuple(faces),
            gravity=self.reference.gravity,
            boundaries=self.boundaries,
            prescribed=tuple(data),
            order=self.order,
        )
        return PreparedDryAtmosphere(self, balance)


class PreparedDryAtmosphere(AbstractFixedStepMethod):
    """Atomic SSPRK(3,3) on native FV content with stage-CFL/admissibility veto.

    This is a native fixed-step method for production continuation. It does not
    misrepresent generic FV fallback dynamics as equilibrium-preserving.
    No atmosphere-specific positivity clipping or silent step reduction occurs.
    """

    plan: DryAtmospherePlan
    balance: PreparedAtmosphericBalance
    precision: FiniteVolumePrecisionPolicy
    method_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DryAtmospherePlan, balance: PreparedAtmosphericBalance):
        self.plan, self.balance = plan, balance
        self.precision = FiniteVolumePrecisionPolicy(
            jnp.dtype(balance.reference.dtype).name
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dry-atmosphere",
                "plan": plan.plan_id,
                "balance": balance.balance_id,
                "precision": self.precision.policy_id,
            }
        )
        self.method_id = canonical_fingerprint(
            {"kind": "dry-atmosphere-ssprk33", "prepared": self.prepared_id}
        )

    @property
    def system(self) -> HomogeneousMixtureEulerSystem:
        return self.balance.system

    def _check(self, state: DryAtmosphereState) -> None:
        if (
            not isinstance(state, DryAtmosphereState)
            or state.prepared_id != self.prepared_id
        ):
            raise ValueError(
                "Atmospheric state/restart belongs to another prepared plan."
            )

    def _integral(self, conserved: Array) -> Array:
        values = conserved.at[..., -1].add(
            self.balance.potential * self.system.density(conserved)
        )
        return jnp.sum(
            (values * self.balance.discretization.cell_volumes[..., None]).reshape(
                (-1, self.system.component_count)
            ),
            axis=0,
        )

    def initial_state(
        self, conserved: ArrayLike | None = None, *, time: float = 0.0
    ) -> DryAtmosphereState:
        values = (
            self.balance.reference
            if conserved is None
            else jnp.asarray(conserved, dtype=self.balance.reference.dtype)
        )
        if values.shape != self.balance.reference.shape:
            raise ValueError(
                "Initial atmosphere must match the prepared cell/component shape."
            )
        values = eqx.error_if(
            values,
            jnp.any(~self.system.admissible(values)),
            "Initial dry atmosphere is inadmissible (including near vacuum).",
        )
        geometry = self.balance.discretization
        volumes = geometry.cell_volumes.reshape((-1,))
        content = FiniteVolumeConservativeContentState.from_cell_average(
            values.reshape((-1, self.system.component_count)),
            volumes,
            jnp.ones(volumes.shape, dtype=bool),
            jnp.asarray(time),
            topology_epoch_id=geometry.prepared_id,
            geometry_family_id=geometry.plan_id,
            geometry_layout_id=geometry.cell_layout.layout_id,
            geometry_version=jnp.asarray(0, dtype=jnp.int32),
            evidence_policy_id=self.balance.balance_id,
            evidence_version=jnp.asarray(0, dtype=jnp.int32),
            precision=self.precision,
        )
        initial = self._integral(content.cell_average().reshape(values.shape))
        return DryAtmosphereState(
            content,
            initial,
            jnp.zeros_like(initial),
            jnp.zeros_like(initial),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan.shape,
            self.prepared_id,
        )

    def thermal_state(
        self, temperature_perturbation: ArrayLike, *, velocity: ArrayLike | None = None
    ) -> DryAtmosphereState:
        """Pressure-balanced temperature perturbation of the cell-average rest state."""
        recovered = self.system.recover_thermodynamics(self.balance.reference).state
        values = self.plan.air.conserved(
            self.system,
            recovered.pressure,
            recovered.temperature + jnp.asarray(temperature_perturbation),
            velocity,
        )
        return self.initial_state(values)

    def budget(self, state: DryAtmosphereState) -> DryAtmosphereBudget:
        self._check(state)
        values = state.conserved
        integral = self._integral(values)
        potential = jnp.sum(
            self.system.density(values)
            * self.balance.potential
            * self.balance.discretization.cell_volumes
        )
        return DryAtmosphereBudget(
            integral[: self.system.species_count],
            integral[self.system.momentum_slice],
            integral[-1] - potential,
            potential,
            integral[-1],
            state.boundary_integral,
            state.source_integral,
            integral
            - state.initial_integral
            + state.boundary_integral
            - state.source_integral,
        )

    def stable_step(self, state: DryAtmosphereState) -> Array:
        self._check(state)
        evaluated = self.balance.evaluate(state.conserved)
        acoustic = self.plan.cfl / jnp.maximum(
            evaluated.maximum_rate, jnp.finfo(state.conserved.dtype).tiny
        )
        gravity = jnp.sqrt(
            self.plan.cfl * self.balance.widths[-1] / self.plan.reference.gravity
        )
        return jnp.where(evaluated.successful, jnp.minimum(acoustic, gravity), 0.0)

    def advance(
        self, state: DryAtmosphereState, step_size: ArrayLike
    ) -> DryAtmosphereStepResult:
        self._check(state)
        step = jnp.asarray(step_size, dtype=state.time.dtype)
        if step.shape != ():
            raise ValueError("Atmospheric step size must be scalar.")
        u0 = state.conserved
        evaluations, valid_stages, limits = [], [], []
        current = u0
        for stage in range(3):
            evaluated = self.balance.evaluate(current)
            limit = jnp.minimum(
                self.plan.cfl
                / jnp.maximum(evaluated.maximum_rate, jnp.finfo(u0.dtype).tiny),
                jnp.sqrt(
                    self.plan.cfl * self.balance.widths[-1] / self.plan.reference.gravity
                ),
            )
            trial = current + step * evaluated.residual
            if stage == 1:
                trial = u0 + 0.25 * ((current - u0) + step * evaluated.residual)
            elif stage == 2:
                trial = u0 + (2.0 / 3.0) * ((current - u0) + step * evaluated.residual)
            valid = (
                evaluated.successful
                & jnp.all(self.system.admissible(trial))
                & jnp.isfinite(step)
                & (step > 0.0)
                & (step <= limit)
            )
            evaluations.append(evaluated)
            valid_stages.append(valid)
            limits.append(limit)
            # Failed stages are never passed into a later thermodynamic solve.
            current = jnp.where(valid, trial, u0)
        accepted = jnp.all(jnp.stack(valid_stages))
        weights = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
        boundary = step * sum(
            weight * evaluation.boundary_outward_flux
            for weight, evaluation in zip(weights, evaluations, strict=True)
        )
        source = step * sum(
            weight * evaluation.source_integral
            for weight, evaluation in zip(weights, evaluations, strict=True)
        )
        increment = (
            (current - u0) * self.balance.discretization.cell_volumes[..., None]
        ).reshape(state.content.conservative_content.shape)
        content = state.content.with_content(
            state.content.conservative_content + jnp.where(accepted, increment, 0.0),
            time=state.time + jnp.where(accepted, step, 0.0),
            evidence_version=state.content.evidence_version + accepted.astype(jnp.int32),
        )
        result = DryAtmosphereState(
            content,
            state.initial_integral,
            state.boundary_integral + jnp.where(accepted, boundary, 0.0),
            state.source_integral + jnp.where(accepted, source, 0.0),
            state.accepted_steps + accepted.astype(jnp.int32),
            state.cell_shape,
            state.prepared_id,
        )
        return DryAtmosphereStepResult(
            result,
            accepted,
            jnp.min(jnp.stack(limits)),
            jnp.stack(valid_stages),
            self.budget(result),
        )

    def step(self, step_index, time, state, step_size, args, /) -> FixedStepResult:
        del args
        size = eqx.error_if(
            jnp.asarray(step_size),
            (jnp.asarray(time) != state.time)
            | (jnp.asarray(step_index) != state.accepted_steps),
            "Atmospheric continuation schedule and accepted state disagree.",
        )
        result = self.advance(state, size)
        return FixedStepResult(
            result.state,
            result.state,
            result.accepted,
            jnp.maximum(size - result.stable_step, 0.0),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=size.dtype),
        )

    def rollout(
        self, state: DryAtmosphereState, step_sizes: ArrayLike
    ) -> DryAtmosphereRolloutResult:
        """A prescribed schedule; the first rejection freezes all later steps."""
        self._check(state)
        steps = jnp.asarray(step_sizes, dtype=state.time.dtype)
        if steps.ndim != 1 or steps.size == 0:
            raise ValueError(
                "Atmospheric rollout requires a nonempty vector of step sizes."
            )

        def body(carry, size):
            previous, running = carry
            result = self.advance(previous, jnp.where(running, size, 0.0))
            running = running & result.accepted
            return (result.state, running), (
                result.state.time,
                result.state.conserved,
                running,
            )

        (final, successful), (times, states, accepted) = jax.lax.scan(
            body, (state, jnp.asarray(True)), steps
        )
        return DryAtmosphereRolloutResult(
            final, times, states, accepted, successful, self.budget(final)
        )

    def checkpoint(self, state: DryAtmosphereState) -> DryAtmosphereRestart:
        self._check(state)
        identifier = canonical_fingerprint(
            {
                "kind": "dry-atmosphere-restart",
                "prepared": self.prepared_id,
                "state": array_tree_fingerprint(state),
            }
        )
        return DryAtmosphereRestart(state, self.prepared_id, identifier)

    def restore(self, restart: DryAtmosphereRestart) -> DryAtmosphereState:
        if (
            not isinstance(restart, DryAtmosphereRestart)
            or restart.prepared_id != self.prepared_id
        ):
            raise ValueError("Dry atmospheric restart identity does not match this plan.")
        self._check(restart.state)
        if self.checkpoint(restart.state).restart_id != restart.restart_id:
            raise ValueError(
                "Dry atmospheric restart content fingerprint is inconsistent."
            )
        return restart.state


__all__ = [
    "DryAir",
    "DryHydrostaticReference",
    "DryAtmospherePlan",
    "PreparedDryAtmosphere",
    "DryAtmosphereState",
    "DryAtmosphereBudget",
    "DryAtmosphereStepResult",
    "DryAtmosphereRolloutResult",
    "DryAtmosphereRestart",
]

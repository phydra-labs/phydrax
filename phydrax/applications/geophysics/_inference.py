# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Local inverse experiments for the actual conservative interactive moist column.

This is an observation/parameter binding, not a second dynamics or UQ engine.
Bounds constrain native least squares; native fixed-step replay advances physics;
UQ information actions and native SVD expose local confounding. No state analysis
increment is ever inserted into an atmospheric or slab physical-flux ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ... import linalg, optim, uq
from ..._array_archive import read_array_archive, write_array_archive
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...metrix import EuclideanStateGeometry
from ...solver import FixedStepProblem, FixedStepRolloutPlan
from ...solver._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ..atmosphere._interactive_column import (
    InteractiveMoistColumnPlan,
    InteractiveMoistColumnState,
)
from ._observations import GeophysicalObservationOperator, PreparedGeophysicalObservations
from ._time import GeophysicalTimeSpec


# Each selector identifies an existing numerical leaf; no surrogate parameters.
_PARAMETERS = {
    "shortwave_absorption_scale": lambda p: p.radiation.shortwave_absorption_scale,
    "shortwave_scattering_scale": lambda p: p.radiation.shortwave_scattering_scale,
    "longwave_absorption_scale": lambda p: p.radiation.longwave_absorption_scale,
    "heat_transfer_coefficient": lambda p: p.surface_exchange.heat_transfer_coefficient,
    "moisture_transfer_coefficient": lambda p: (
        p.surface_exchange.moisture_transfer_coefficient
    ),
    "background_diffusivity": lambda p: p.background_diffusivity,
    "mixing_length": lambda p: p.mixing_length,
    "critical_richardson": lambda p: p.critical_richardson,
    "condensation_timescale": lambda p: p.condensation_timescale,
    "autoconversion_timescale": lambda p: p.autoconversion_timescale,
    "rain_evaporation_timescale": lambda p: p.rain_evaporation_timescale,
    "phase_conversion_timescale": lambda p: p.phase_conversion_timescale,
    "cloud_threshold": lambda p: p.cloud_threshold,
    "rain_fall_speed": lambda p: p.rain_fall_speed,
    "snow_fall_speed": lambda p: p.snow_fall_speed,
}
# kind, sign, reference. SI conversion is made through the observation quantity.
_SIGNALS = {
    "toa_net_upward_flux": ("radiative_flux", "upward", "absolute"),
    "surface_net_downward_flux": ("radiative_flux", "downward", "absolute"),
    "temperature": ("temperature", "positive", "absolute"),
    "specific_humidity": ("specific_humidity", "positive", "absolute"),
    "surface_temperature": ("temperature", "positive", "absolute"),
    "surface_water_mass": ("water_mass_per_area", "positive", "absolute"),
    "precipitated_water": (
        "precipitation_amount",
        "positive",
        "experiment-initial-state",
    ),
}


def _nonempty(value, role):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be an explicit nonempty provenance label.")
    return value


class ColumnParameterSpace(StrictModule):
    """Physical theta = scale * z; scales are units, not a hidden prior.

    Optical scalar parameters set all four species' scale leaves equally. Fixed
    species contrast lives in ColumnOpticalProperties. Bounds and scales refer to
    physical units (seconds, m²/s, m/s, or dimensionless native coefficients).
    """

    scales: Array
    lower: Array
    upper: Array
    names: tuple[str, ...] = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(self, names, scales, lower, upper):
        names = tuple(names)
        if not names or len(set(names)) != len(names) or set(names) - set(_PARAMETERS):
            raise ValueError("Parameters must be unique supported native column leaves.")
        arrays = tuple(np.asarray(x, dtype=float) for x in (scales, lower, upper))
        if any(x.shape != (len(names),) or not np.all(np.isfinite(x)) for x in arrays):
            raise ValueError("Scales and bounds require finite vectors matching names.")
        scale, lo, hi = arrays
        if np.any(scale <= 0) or np.any(lo < 0) or np.any(hi <= lo):
            raise ValueError("Scales must be positive; bounds nonnegative and ordered.")
        for i, name in enumerate(names):
            if (name.endswith("timescale") or name == "critical_richardson") and lo[
                i
            ] <= 0:
                raise ValueError(
                    "Timescales and critical Richardson number require positive bounds."
                )
        self.names = names
        self.scales, self.lower, self.upper = map(jnp.asarray, arrays)
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "interactive-column-parameters",
                "names": names,
                "values": array_tree_fingerprint(arrays),
            }
        )

    def apply(self, plan, physical):
        values = jnp.asarray(physical)
        if values.shape != self.scales.shape:
            raise ValueError("Physical parameter vector has the wrong shape.")
        for i, name in enumerate(self.names):
            if "wave_" in name and plan.radiation is None:
                raise ValueError("Optical calibration requires active radiation.")
            if "transfer_coefficient" in name and plan.surface_exchange is None:
                raise ValueError("Surface calibration requires active exchange.")
            selector = _PARAMETERS[name]
            old = selector(plan)
            plan = eqx.tree_at(
                selector, plan, jnp.broadcast_to(values[i], jnp.shape(old))
            )
        return plan

    def check(self, physical):
        values = np.asarray(physical)
        if values.shape != self.scales.shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "Physical parameters must be a finite vector matching the space."
            )
        if np.any(values < np.asarray(self.lower)) or np.any(
            values > np.asarray(self.upper)
        ):
            raise ValueError(
                "Physical parameters lie outside declared bounds; no clipping occurs."
            )


@dataclass(frozen=True)
class ColumnIntervention:
    """Known forcing changes applied BEFORE an experiment, not fitted parameters.

    Capacity changes prepare the slab at the same initial temperature and water
    inventory. The associated preparation energy is reported separately, never
    inserted as a model flux. Opacity multiplies all native LW absorption scales.
    """

    label: str
    solar_multiplier: float = 1.0
    longwave_multiplier: float = 1.0
    slab_capacity_multiplier: float = 1.0

    def __post_init__(self):
        _nonempty(self.label, "Intervention label")
        values = (
            self.solar_multiplier,
            self.longwave_multiplier,
            self.slab_capacity_multiplier,
        )
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("Intervention multipliers must be finite and positive.")

    @property
    def intervention_id(self):
        return canonical_fingerprint({"kind": "column-intervention", **self.__dict__})


@dataclass(frozen=True)
class ColumnObservationBinding:
    """One native observation operator sampling one explicitly named physical field."""

    signal: str
    operator: GeophysicalObservationOperator
    times: tuple[float, ...]

    def __post_init__(self):
        if self.signal not in _SIGNALS:
            raise ValueError("Unknown interactive-column observation signal.")
        if not isinstance(self.operator, GeophysicalObservationOperator):
            raise TypeError("operator must be a native GeophysicalObservationOperator.")
        q = self.operator.quantity
        if (q.quantity_kind, q.sign_convention, q.reference_configuration) != _SIGNALS[
            self.signal
        ]:
            raise ValueError(
                "Observation quantity kind, sign or reference differs from physical signal."
            )
        times = np.asarray(self.times, dtype=float)
        if (
            times.ndim != 1
            or not times.size
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Observation times must be finite and strictly increasing.")
        object.__setattr__(self, "times", tuple(float(t) for t in times))

    @property
    def binding_id(self):
        return canonical_fingerprint(
            {
                "signal": self.signal,
                "operator": self.operator.operator_id,
                "times": self.times,
            }
        )


class _DifferentiableColumnMethod(AbstractFixedStepMethod):
    """Native fixed-step adapter retaining admission/branch validity as a Boolean state."""

    plan: InteractiveMoistColumnPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan):
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {"kind": "column-inference-fixed-step", "plan": plan.plan_id}
        )

    def step(self, step_index, time, state, step_size, args, /):
        del step_index
        physical, regular = state
        result = self.plan.step(physical, step_size, **args)
        eps = jnp.finfo(physical.time.dtype).eps
        clock_valid = jnp.abs(time - physical.time) <= 64 * eps * jnp.maximum(
            jnp.abs(time), 1
        )
        valid = result.successful & clock_valid
        candidate = (result.state, regular & result.derivative_valid)
        accepted = jax.tree.map(
            lambda new, old: jnp.where(valid, new, old), candidate, state
        )
        return FixedStepResult(
            candidate,
            accepted,
            valid,
            jnp.maximum(step_size - result.stable_step, 0),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(1, jnp.int32),
            jnp.asarray(False),
            jnp.zeros_like(physical.time),
        )


class ColumnPrediction(StrictModule):
    values: Array
    final_state: InteractiveMoistColumnState
    successful: Array
    derivative_valid: Array
    preparation_energy: Array
    regime_signature: Array


@dataclass(frozen=True, init=False)
class ColumnExperiment:
    """Fixed physical run and targeted observations, with no observed values.

    Binding times use the named calendar/epoch; numerical column time is seconds
    from that epoch. Only grid-aligned instantaneous samples are admitted: there
    is no nearest-time snapping or interval-as-point reinterpretation. The
    precipitation signal samples the cumulative native fallout ledger relative
    to the initial state, not an instantaneous precipitation rate.
    """

    initial_state: InteractiveMoistColumnState
    bindings: tuple[ColumnObservationBinding, ...]
    time: GeophysicalTimeSpec
    intervention: ColumnIntervention
    dt: float
    steps: int
    forcing: MappingProxyType
    start: float
    source_id: str
    experiment_id: str

    def __init__(
        self,
        initial_state,
        bindings,
        *,
        time,
        dt,
        steps,
        source_id,
        intervention=None,
        solar_down=340.0,
        wind_speed=5.0,
        measurement_height=10.0,
        ventilation=0.0,
        shear=0.0,
        heating_rate=0.0,
    ):
        bindings = tuple(bindings)
        if not bindings or not all(
            isinstance(x, ColumnObservationBinding) for x in bindings
        ):
            raise ValueError("Experiment requires native physical observation bindings.")
        if not isinstance(time, GeophysicalTimeSpec):
            raise TypeError("time must be GeophysicalTimeSpec.")
        if not np.isfinite(dt) or dt <= 0 or int(steps) != steps or steps < 1:
            raise ValueError("dt and integer steps must be positive.")
        initial_time = float(initial_state.time)
        intervention = (
            ColumnIntervention("control") if intervention is None else intervention
        )
        forcing = dict(
            solar_down=solar_down,
            wind_speed=wind_speed,
            measurement_height=measurement_height,
            ventilation=ventilation,
            shear=shear,
            heating_rate=heating_rate,
        )
        if any(not np.all(np.isfinite(x)) for x in forcing.values()):
            raise ValueError(
                "Forcing must be finite; physical admission remains the column's responsibility."
            )
        n = initial_state.dry_mass.size
        for binding in bindings:
            if binding.operator.time.time_id != time.time_id:
                raise ValueError(
                    "Observation calendar/epoch differs from experiment clock."
                )
            ticks = (
                np.asarray(binding.times) * time.seconds_per_unit - initial_time
            ) / dt
            if (
                np.any(ticks < 0)
                or np.any(ticks > steps)
                or not np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-10)
            ):
                raise ValueError(
                    "Observation times must lie exactly on the physical integration grid."
                )
            expected = (
                (n,) if binding.signal in ("temperature", "specific_humidity") else (1,)
            )
            if binding.operator.transfer.source.vector_space.shape != expected:
                raise ValueError(
                    "Observation source shape differs from the column field."
                )
        forcing = {name: jnp.asarray(value) for name, value in forcing.items()}
        values = dict(
            initial_state=initial_state,
            bindings=bindings,
            time=time,
            dt=float(dt),
            steps=int(steps),
            forcing=MappingProxyType(forcing),
            start=initial_time,
            intervention=intervention,
            source_id=_nonempty(source_id, "Source"),
        )
        for key, value in values.items():
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "experiment_id",
            canonical_fingerprint(
                {
                    "kind": "interactive-column-experiment",
                    "initial": array_tree_fingerprint(initial_state),
                    "plan": initial_state.plan_id,
                    "bindings": [b.binding_id for b in bindings],
                    "clock": time.time_id,
                    "dt": float(dt),
                    "steps": int(steps),
                    "forcing": array_tree_fingerprint(forcing),
                    "source": source_id,
                    "intervention": intervention.intervention_id,
                }
            ),
        )

    def refined(self, factor=2):
        if int(factor) != factor or factor < 2:
            raise ValueError("Refinement factor must be an integer at least two.")
        return ColumnExperiment(
            self.initial_state,
            self.bindings,
            time=self.time,
            dt=self.dt / factor,
            steps=self.steps * factor,
            source_id=self.source_id,
            intervention=self.intervention,
            **self.forcing,
        )

    def prepared_plan_state(self, plan):
        state = self.initial_state
        if plan.plan_id != state.plan_id:
            raise ValueError(
                "Experiment initial state and physical plan differ structurally."
            )
        if self.intervention.longwave_multiplier != 1:
            if plan.radiation is None:
                raise ValueError("Opacity intervention requires radiation.")
            plan = eqx.tree_at(
                lambda p: p.radiation.longwave_absorption_scale,
                plan,
                plan.radiation.longwave_absorption_scale
                * self.intervention.longwave_multiplier,
            )
        temperature = plan.slab.temperature(state.slab, plan.thermodynamics)
        if self.intervention.slab_capacity_multiplier != 1:
            plan = eqx.tree_at(
                lambda p: p.slab.dry_heat_capacity,
                plan,
                plan.slab.dry_heat_capacity * self.intervention.slab_capacity_multiplier,
            )
        slab = plan.slab.initialize(temperature, state.slab.water_mass)
        preparation_energy = slab.energy - state.slab.energy
        state = eqx.tree_at(lambda s: s.slab, state, slab)
        return plan, state, preparation_energy

    def predict(self, plan):
        plan, initial, preparation = self.prepared_plan_state(plan)
        forcing = {
            **self.forcing,
            "solar_down": self.forcing["solar_down"] * self.intervention.solar_multiplier,
        }
        method = _DifferentiableColumnMethod(plan)
        initial_valid = jnp.all(plan.diagnose(initial).derivative_valid)
        problem = FixedStepProblem(
            method,
            (initial, initial_valid),
            t0=self.start,
            t1=self.start + self.steps * self.dt,
            step_size=self.dt,
            args=forcing,
            state_geometry=EuclideanStateGeometry(),
            problem_id=self.experiment_id,
        )
        trajectory = FixedStepRolloutPlan(retention="trajectory").rollout(problem)
        states, regular = trajectory.states
        output = []
        valid = trajectory.successful
        for binding in self.bindings:
            ticks = np.rint(
                (np.asarray(binding.times) * self.time.seconds_per_unit - self.start)
                / self.dt
            ).astype(int)
            selected = jax.tree.map(lambda x, ticks=ticks: x[jnp.asarray(ticks)], states)

            def sample(state, binding=binding):
                diagnosed = plan.diagnose(state)
                signal = binding.signal
                radiation_valid = jnp.asarray(True)
                if signal in ("toa_net_upward_flux", "surface_net_downward_flux"):
                    if plan.radiation is None:
                        raise ValueError(
                            "Radiative observations require active physical radiation."
                        )
                    r = plan.radiation.evaluate(
                        diagnosed.temperature,
                        state.layer_mass,
                        state.vapor_mass,
                        state.cloud_liquid_mass,
                        state.cloud_ice_mass,
                        diagnosed.surface_temperature,
                        forcing["solar_down"],
                        rain_mass=state.rain_mass,
                        snow_mass=state.snow_mass,
                    )
                    field = jnp.atleast_1d(
                        r.space_heating
                        if signal == "toa_net_upward_flux"
                        else r.surface_heating
                    )
                    radiation_valid = r.successful
                elif signal == "temperature":
                    field = diagnosed.temperature
                elif signal == "specific_humidity":
                    field = state.vapor_mass / state.layer_mass
                elif signal == "surface_temperature":
                    field = jnp.atleast_1d(diagnosed.surface_temperature)
                elif signal == "surface_water_mass":
                    field = jnp.atleast_1d(state.slab.water_mass)
                else:
                    field = jnp.atleast_1d(
                        state.precipitated_water - initial.precipitated_water
                    )
                return binding.operator(
                    binding.operator.quantity.from_si(field)
                ), jnp.all(diagnosed.successful) & radiation_valid

            values, samples_valid = jax.vmap(sample)(selected)
            output.append(values.reshape(-1))
            valid = valid & jnp.all(samples_valid)
        values = jnp.concatenate(output)
        valid = valid & jnp.all(jnp.isfinite(values))
        diagnosed_states = jax.vmap(plan.diagnose)(states)
        regime = jnp.concatenate(
            (
                diagnosed_states.temperature < plan.thermodynamics.reference_temperature,
                diagnosed_states.relative_humidity > 1,
                states.cloud_liquid_mass > 0,
                states.cloud_ice_mass > 0,
                states.rain_mass > 0,
                states.snow_mass > 0,
                states.cloud_liquid_mass + states.cloud_ice_mass
                > plan.cloud_threshold * states.dry_mass,
                (
                    diagnosed_states.surface_temperature
                    > diagnosed_states.temperature[:, -1]
                )[:, None],
            ),
            axis=1,
        )
        return ColumnPrediction(
            values,
            trajectory.final_state[0],
            valid,
            valid & jnp.all(regular),
            preparation,
            regime,
        )


@dataclass(frozen=True, init=False)
class ColumnObservationData:
    """Accepted native product values and an explicit joint error covariance.

    Flattening order is binding, then time, then native target-field order.
    Cross-signal correlations may be supplied; their diagonal MUST agree with
    native prepared total error variances. Missing/QC observations are removed
    with a principal covariance submatrix, not replaced by zero-weight evidence.
    """

    experiment: ColumnExperiment
    products: tuple[PreparedGeophysicalObservations, ...]
    values: Array
    indices: Array
    covariance: uq.DenseCovariance
    whitener: Array
    provenance: str
    role: str
    data_id: str

    def __init__(self, experiment, products, *, provenance, role, covariance=None):
        products = tuple(products)
        if role not in ("calibration", "holdout"):
            raise ValueError("Observation role must be calibration or holdout.")
        if len(products) != len(experiment.bindings):
            raise ValueError(
                "Exactly one native observation product is required per binding."
            )
        values, variances, masks = [], [], []
        for binding, product in zip(experiment.bindings, products):
            if not isinstance(product, PreparedGeophysicalObservations):
                raise TypeError("Products must be PreparedGeophysicalObservations.")
            if (
                product.operator.operator_id != binding.operator.operator_id
                or not np.array_equal(np.asarray(product.sequence.times), binding.times)
            ):
                raise ValueError(
                    "Product operator/time support differs from experiment binding."
                )
            values.append(np.asarray(product.sequence.values).reshape(-1))
            variances.append(np.asarray(product.error_variance).reshape(-1))
            masks.append(np.asarray(product.sequence.observation_mask).reshape(-1))
        values, variances, mask = (
            np.concatenate(values),
            np.concatenate(variances),
            np.concatenate(masks),
        )
        indices = np.flatnonzero(mask)
        if not indices.size:
            raise ValueError("No accepted observations remain after native QC.")
        full = (
            np.diag(variances)
            if covariance is None
            else np.asarray(covariance, dtype=float)
        )
        if full.shape != (values.size, values.size) or not np.all(np.isfinite(full)):
            raise ValueError(
                "Joint covariance must be a finite square matrix in declared flattening order."
            )
        if not np.allclose(full, full.T, rtol=1e-10, atol=1e-12) or not np.allclose(
            np.diag(full)[indices], variances[indices], rtol=1e-8, atol=0
        ):
            raise ValueError(
                "Covariance must be symmetric and retain native marginal error variances."
            )
        accepted = full[np.ix_(indices, indices)]
        # Host preparation certifies strict positive definiteness; no jitter.
        factor = np.linalg.cholesky(accepted)
        whitening = linalg.inverse(jnp.asarray(factor), linalg.FactorizationPolicy("lu"))
        if not bool(whitening.successful):
            raise ValueError("Observation covariance whitening failed.")
        fields = dict(
            experiment=experiment,
            products=products,
            values=jnp.asarray(values[indices]),
            indices=jnp.asarray(indices),
            covariance=uq.DenseCovariance(jnp.asarray(accepted)),
            whitener=whitening.value,
            provenance=_nonempty(provenance, "Observation provenance"),
            role=role,
        )
        for key, value in fields.items():
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "data_id",
            canonical_fingerprint(
                {
                    "kind": "column-observation-data",
                    "experiment": experiment.experiment_id,
                    "products": [p.preparation_id for p in products],
                    "covariance": array_tree_fingerprint(accepted),
                    "provenance": provenance,
                    "role": role,
                }
            ),
        )


class ColumnLocalInformation(StrictModule):
    jacobian: Array
    fisher: Array
    singular_values: Array
    combinations: Array
    identifiable: Array
    rank: Array
    covariance_on_identifiable_subspace: Array
    unidentifiable_projector: Array
    derivative_valid: Array
    active_bounds: Array
    approximation: str = eqx.field(
        static=True, default="local-Gaussian-linearization-not-global-posterior"
    )


def column_local_information(
    output_fn, center, *, derivative_valid=True, active_bounds=None, rank_rtol=1e-7
):
    """Native SVD of a whitened JVP map in dimensionless parameter coordinates.

    Rows of combinations are orthonormal z directions. The covariance field is
    conditional on the nullspace being fixed: zero nullspace variance is NEVER
    asserted. unidentifiable_projector explicitly carries these unconstrained
    directions. Bounds invalidate an unconstrained Gaussian interpretation.
    """
    center = jnp.asarray(center)
    if (
        center.ndim != 1
        or not center.size
        or not np.isfinite(rank_rtol)
        or rank_rtol <= 0
    ):
        raise ValueError("A nonempty vector and positive rank tolerance are required.")
    _, push = jax.linearize(output_fn, center)
    jacobian = jax.vmap(push)(jnp.eye(center.size, dtype=center.dtype)).T
    if jacobian.ndim != 2:
        raise ValueError("Output must be one flattened vector.")
    padded = jnp.pad(jacobian, ((0, max(0, center.size - jacobian.shape[0])), (0, 0)))
    decomposition = linalg.svd.svd(
        linalg.svd.SVDProblem(linalg.DenseLinearOperator(padded)),
        policy=linalg.svd.SVDSolvePolicy(
            count=center.size, rank=linalg.RankPolicy(relative_cutoff=rank_rtol)
        ),
    )
    singular = decomposition.singular_values
    # Native vectors use columns; reported identifiable combinations use rows.
    combinations = decomposition.right_vectors.T
    identified = singular > rank_rtol * jnp.max(singular)
    inverse_variance = jnp.where(identified, 1 / jnp.where(identified, singular**2, 1), 0)
    covariance = (combinations.T * inverse_variance) @ combinations
    null = (combinations.T * (~identified)) @ combinations
    active = (
        jnp.zeros(center.shape, bool)
        if active_bounds is None
        else jnp.asarray(active_bounds)
    )
    valid = (
        jnp.asarray(derivative_valid)
        & decomposition.successful
        & jnp.all(jnp.isfinite(jacobian))
    )
    return ColumnLocalInformation(
        jacobian,
        jacobian.T @ jacobian,
        singular,
        combinations,
        identified,
        jnp.sum(identified),
        covariance,
        null,
        valid,
        active,
    )


@dataclass(frozen=True)
class ColumnCalibrationResult:
    parameters: Array
    optimizer: optim.LeastSquaresResult
    information: ColumnLocalInformation
    successful: Array
    inference_id: str
    problem_id: str


@dataclass(frozen=True, init=False)
class ColumnCalibrationProblem:
    plan: InteractiveMoistColumnPlan
    space: ColumnParameterSpace
    data: tuple[ColumnObservationData, ...]
    problem_id: str

    def __init__(self, plan, space, data):
        data = tuple(data)
        if not isinstance(plan, InteractiveMoistColumnPlan) or not isinstance(
            space, ColumnParameterSpace
        ):
            raise TypeError(
                "Calibration requires an actual interactive column and its numeric parameter space."
            )
        if not data or any(x.role != "calibration" for x in data):
            raise ValueError(
                "Only explicitly calibration-role observations may enter the objective."
            )
        product_ids = [p.preparation_id for d in data for p in d.products]
        if len(set(product_ids)) != len(product_ids):
            raise ValueError(
                "Duplicate observation products would double-count evidence."
            )
        space.apply(plan, space.lower)  # Structural parameter checks, no simulation.
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "data", data)
        object.__setattr__(
            self,
            "problem_id",
            canonical_fingerprint(
                {
                    "kind": "column-calibration",
                    "plan": plan.plan_id,
                    "numerics": array_tree_fingerprint(plan),
                    "space": space.parameter_id,
                    "observations": [d.data_id for d in data],
                }
            ),
        )

    def residual(self, dimensionless):
        plan = self.space.apply(self.plan, dimensionless * self.space.scales)
        residuals = []
        for data in self.data:
            predicted = data.experiment.predict(plan)
            residual = data.whitener @ (predicted.values[data.indices] - data.values)
            # Failed admission is not a flat zero residual. Nonsmooth but admitted
            # values remain real; their derivative validity is reported separately.
            residuals.append(jnp.where(predicted.successful, residual, jnp.nan))
        return jnp.concatenate(residuals)

    def information(self, physical, *, rank_rtol=1e-7):
        plan = self.space.apply(self.plan, physical)
        valid = jnp.all(
            jnp.stack([d.experiment.predict(plan).derivative_valid for d in self.data])
        )
        z = jnp.asarray(physical) / self.space.scales
        tolerance = 64 * jnp.finfo(z.dtype).eps * jnp.maximum(jnp.abs(z), 1)
        active = (z - self.space.lower / self.space.scales <= tolerance) | (
            self.space.upper / self.space.scales - z <= tolerance
        )
        return column_local_information(
            self.residual,
            z,
            derivative_valid=valid,
            active_bounds=active,
            rank_rtol=rank_rtol,
        )

    def fisher_action(self, physical, direction):
        """Native UQ JVP/VJP action with physical branch/admission validity."""
        result = uq.gauss_newton_action(
            self.residual,
            jnp.asarray(physical) / self.space.scales,
            jnp.asarray(direction),
        )
        plan = self.space.apply(self.plan, physical)
        regular = jnp.all(
            jnp.stack([d.experiment.predict(plan).derivative_valid for d in self.data])
        )
        return eqx.tree_at(
            lambda r: (r.valid, r.status),
            result,
            (
                result.valid & regular,
                jnp.where(regular, result.status, uq.SENSITIVITY_INVALID_INFORMATION),
            ),
        )

    def calibrate(self, initial, *, termination=None, rank_rtol=1e-7):
        """Fit with a caller-declared native stopping policy in whitened z coordinates."""
        self.space.check(initial)
        termination = (
            optim.OptimizationTermination(maximum_steps=32)
            if termination is None
            else termination
        )
        if not isinstance(termination, optim.OptimizationTermination):
            raise TypeError("termination must be a native OptimizationTermination.")
        problem = optim.NonlinearLeastSquaresProblem(
            lambda z, _: self.residual(z),
            bounds=optim.Bounds(
                self.space.lower / self.space.scales, self.space.upper / self.space.scales
            ),
            problem_id=self.problem_id,
        )
        result = optim.least_squares(
            problem,
            jnp.asarray(initial) / self.space.scales,
            method=optim.BoundedLevenbergMarquardt(),
            termination=termination,
        )
        physical = result.parameters * self.space.scales
        info = self.information(physical, rank_rtol=rank_rtol)
        success = (
            result.successful
            & info.derivative_valid
            & jnp.all(jnp.isfinite(result.residual))
        )
        identity = canonical_fingerprint(
            {
                "kind": "column-calibration-result",
                "problem": self.problem_id,
                "parameters": array_tree_fingerprint(physical),
                "rank_rtol": rank_rtol,
                "approximation": info.approximation,
                "termination": {
                    "absolute_optimality": termination.absolute_optimality,
                    "relative_optimality": termination.relative_optimality,
                    "absolute_step": termination.absolute_step,
                    "relative_step": termination.relative_step,
                    "maximum_steps": termination.maximum_steps,
                    "maximum_evaluations": termination.maximum_evaluations,
                },
            }
        )
        return ColumnCalibrationResult(
            physical, result, info, success, identity, self.problem_id
        )


def column_gradient_audit(problem, physical, direction, *, epsilons=(1e-2, 1e-3, 1e-4)):
    """Real central model perturbations versus JVP, plus dt/2 and dt/4 JVPs.

    Every stencil checks bounds and physical derivative validity. Invalid
    stencils retain their errors for diagnosis but must not certify a gradient.
    No smoothing, phase projection, or hidden time-step repair is performed.
    """
    problem.space.check(physical)
    z = jnp.asarray(physical) / problem.space.scales
    direction = jnp.asarray(direction, dtype=z.dtype)
    if (
        direction.shape != z.shape
        or not np.all(np.isfinite(direction))
        or not np.any(np.asarray(direction))
    ):
        raise ValueError("Audit direction must be a nonzero finite dimensionless vector.")
    epsilons = tuple(float(e) for e in epsilons)
    if not epsilons or min(epsilons) <= 0 or not np.all(np.isfinite(epsilons)):
        raise ValueError("Finite positive perturbation epsilons are required.")
    _, exact = jax.jvp(problem.residual, (z,), (direction,))
    errors, valid = [], []
    centers = [
        d.experiment.predict(problem.space.apply(problem.plan, physical))
        for d in problem.data
    ]
    for epsilon in epsilons:
        sides = [
            problem.space.scales * (z + sign * epsilon * direction) for sign in (-1, 1)
        ]
        for side in sides:
            problem.space.check(side)
        minus, plus = [problem.residual(side / problem.space.scales) for side in sides]
        fd = (plus - minus) / (2 * epsilon)
        errors.append(
            jnp.sqrt(jnp.sum((fd - exact) ** 2))
            / jnp.maximum(jnp.sqrt(jnp.sum(exact**2)), 1e-12)
        )
        flags = []
        for side in sides:
            for data, center_prediction in zip(problem.data, centers):
                prediction = data.experiment.predict(
                    problem.space.apply(problem.plan, side)
                )
                flags.append(
                    prediction.derivative_valid
                    & center_prediction.derivative_valid
                    & jnp.all(
                        prediction.regime_signature == center_prediction.regime_signature
                    )
                )
        valid.append(jnp.all(jnp.stack(flags)))
    tangents, refinement_valid = [], []
    for factor in (2, 4):

        def refined_residual(point, factor=factor):
            plan = problem.space.apply(problem.plan, point * problem.space.scales)
            return jnp.concatenate(
                [
                    d.whitener
                    @ (
                        d.experiment.refined(factor).predict(plan).values[d.indices]
                        - d.values
                    )
                    for d in problem.data
                ]
            )

        _, tangent = jax.jvp(refined_residual, (z,), (direction,))
        tangents.append(tangent)
        refinement_valid.append(
            jnp.all(
                jnp.stack(
                    [
                        d.experiment.refined(factor)
                        .predict(problem.space.apply(problem.plan, physical))
                        .derivative_valid
                        for d in problem.data
                    ]
                )
            )
        )
    norm = lambda x: jnp.sqrt(jnp.sum(x * x))
    return {
        "epsilons": jnp.asarray(epsilons),
        "relative_errors": jnp.stack(errors),
        "stencil_valid": jnp.stack(valid),
        "center_valid": problem.information(physical).derivative_valid,
        "dt_relative_changes": jnp.stack(
            (norm(tangents[0] - exact), norm(tangents[1] - tangents[0]))
        )
        / jnp.maximum(norm(tangents[1]), 1e-12),
        "refinement_valid": jnp.stack(refinement_valid),
    }


@dataclass(frozen=True)
class ColumnDesignCandidate:
    """A prospective run with declared observation noise, but NO response values."""

    experiment: ColumnExperiment
    covariance: uq.DenseCovariance
    provenance: str

    def __post_init__(self):
        _nonempty(self.provenance, "Candidate error provenance")
        size = sum(
            len(b.times) * b.operator.transfer.target.vector_space.size
            for b in self.experiment.bindings
        )
        if not isinstance(
            self.covariance, uq.DenseCovariance
        ) or self.covariance.matrix.shape != (size, size):
            raise ValueError(
                "Candidate covariance must match every prospective observation."
            )
        np.linalg.cholesky(np.asarray(self.covariance.matrix))


@dataclass(frozen=True)
class ColumnDesignResult:
    chosen: int
    expected_information_gain: Array
    valid: Array
    design_id: str
    approximation: str = (
        "local-linear-Gaussian-mutual-information-with-declared-reference-covariance"
    )


def design_column_intervention(problem, result, candidates, *, reference_covariance):
    """Choose maximum conditional local EIG using actual forward sensitivities.

    reference_covariance is an explicitly declared SPD uncertainty in z before
    calibration (not an inferred global posterior). Conditional local covariance
    is (C_ref^-1 + J_cal^T R^-1 J_cal)^-1. Candidate outcomes never enter design.
    Active bounds or invalid derivatives reject the Gaussian design approximation.
    """
    _matching_result(problem, result)
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("At least one prospective intervention is required.")
    covariance = np.asarray(reference_covariance, dtype=float)
    n = problem.space.scales.size
    if covariance.shape != (n, n) or not np.allclose(covariance, covariance.T):
        raise ValueError(
            "Reference covariance must be a symmetric matrix in dimensionless coordinates."
        )
    np.linalg.cholesky(covariance)
    precision = linalg.inverse(jnp.asarray(covariance), linalg.FactorizationPolicy("lu"))
    conditional = linalg.inverse(
        precision.value + result.information.fisher, linalg.FactorizationPolicy("lu")
    )
    if not bool(precision.successful & conditional.successful):
        raise ValueError("Local conditional covariance factorization failed.")
    factor = jnp.asarray(np.linalg.cholesky(np.asarray(conditional.value)))
    values, flags = [], []
    z = result.parameters / problem.space.scales
    for candidate in candidates:
        whitening = linalg.inverse(
            jnp.asarray(np.linalg.cholesky(np.asarray(candidate.covariance.matrix))),
            linalg.FactorizationPolicy("lu"),
        )

        def response(point, whitening=whitening, candidate=candidate):
            return (
                whitening.value
                @ candidate.experiment.predict(
                    problem.space.apply(problem.plan, point * problem.space.scales)
                ).values
            )

        _, push = jax.linearize(response, z)
        jacobian = jax.vmap(push)(factor.T).T
        objective = uq.experiment_design_objective(
            jacobian.T @ jacobian, criterion="mutual_information"
        )
        physical = candidate.experiment.predict(
            problem.space.apply(problem.plan, result.parameters)
        )
        valid = (
            objective.valid
            & whitening.successful
            & physical.derivative_valid
            & result.information.derivative_valid
            & ~jnp.any(result.information.active_bounds)
        )
        values.append(jnp.where(valid, objective.value, jnp.nan))
        flags.append(valid)
    flags = jnp.stack(flags)
    values = jnp.stack(values)
    chosen = (
        int(jnp.argmax(jnp.where(flags, values, -jnp.inf)))
        if bool(jnp.any(flags))
        else -1
    )
    identity = canonical_fingerprint(
        {
            "kind": "column-local-design",
            "inference": result.inference_id,
            "candidates": [c.experiment.experiment_id for c in candidates],
            "noise": [array_tree_fingerprint(c.covariance) for c in candidates],
            "provenance": [c.provenance for c in candidates],
            "reference_covariance": array_tree_fingerprint(covariance),
        }
    )
    return ColumnDesignResult(chosen, values, flags, identity)


def _matching_result(problem, result):
    if result.problem_id != problem.problem_id:
        raise ValueError(
            "Inference artifact belongs to a different model/parameter/observation lineage."
        )


def score_column_holdout(problem, result, data):
    """Score untouched physical responses and conditional local predictive errors.

    Unidentified parameter directions visible to a prediction make its marginal
    standard deviation infinite. At an active bound/nonsmooth trajectory the
    local Gaussian score is invalid, not silently treated as calibrated coverage.
    """
    _matching_result(problem, result)
    if data.role != "holdout":
        raise ValueError("Prediction scoring requires explicit holdout-role data.")
    trained_ids = {p.preparation_id for d in problem.data for p in d.products}
    if trained_ids.intersection(p.preparation_id for p in data.products):
        raise ValueError("Holdout observation leakage: a product entered calibration.")
    if data.experiment.experiment_id in {
        d.experiment.experiment_id for d in problem.data
    }:
        raise ValueError("Holdout must be a distinct physical/observation experiment.")
    plan = problem.space.apply(problem.plan, result.parameters)
    prediction = data.experiment.predict(plan)

    def response(z):
        return data.experiment.predict(
            problem.space.apply(problem.plan, z * problem.space.scales)
        ).values[data.indices]

    _, push = jax.linearize(response, result.parameters / problem.space.scales)
    jacobian = jax.vmap(push)(
        jnp.eye(problem.space.scales.size, dtype=result.parameters.dtype)
    ).T
    covariance = (
        data.covariance.matrix
        + jacobian @ result.information.covariance_on_identifiable_subspace @ jacobian.T
    )
    null_sensitivity = jnp.sum(
        (jacobian @ result.information.unidentifiable_projector) ** 2, axis=1
    )
    identifiable = null_sensitivity <= 1e-12 * jnp.maximum(
        jnp.sum(jacobian**2, axis=1), 1e-30
    )
    error = prediction.values[data.indices] - data.values
    standard = jnp.where(identifiable, jnp.sqrt(jnp.diag(covariance)), jnp.inf)
    valid = (
        prediction.derivative_valid
        & result.information.derivative_valid
        & ~jnp.any(result.information.active_bounds)
        & jnp.all(identifiable)
    )
    standardized = jnp.where(identifiable, error / standard, jnp.nan)
    return {
        "prediction": prediction.values[data.indices],
        "error": error,
        "noise_normalized_rmse": jnp.sqrt(jnp.mean((data.whitener @ error) ** 2)),
        "standard_deviation": standard,
        "standardized_error": standardized,
        "coverage_95": jnp.where(
            valid, jnp.mean(jnp.abs(standardized) <= 1.959963984540054), jnp.nan
        ),
        "identifiable_predictions": identifiable,
        "uncertainty_valid": valid,
        "physical_successful": prediction.successful,
        "derivative_valid": prediction.derivative_valid,
        "preparation_energy": prediction.preparation_energy,
        "score_id": canonical_fingerprint(
            {"inference": result.inference_id, "holdout": data.data_id}
        ),
        "approximation": result.information.approximation,
    }


def save_column_inference(path, problem, result, *, continuation_data_index=0):
    """Bind inference evidence to the native column checkpoint and its exact physics.

    This is a physical continuation checkpoint, not a warm optimizer-state resume.
    An optimizer may explicitly restart from the archived bounded parameter point.
    """
    _matching_result(problem, result)
    if not bool(result.successful):
        raise ValueError(
            "Cannot archive an unsuccessful calibration as accepted inference."
        )
    index = continuation_data_index
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or not 0 <= index < len(problem.data)
    ):
        raise ValueError(
            "Continuation experiment index must be a nonnegative in-range integer, not bool."
        )
    index = int(index)
    experiment = problem.data[index].experiment
    fitted = problem.space.apply(problem.plan, result.parameters)
    prediction = experiment.predict(fitted)
    effective, _, _ = experiment.prepared_plan_state(fitted)
    path = Path(path)
    checkpoint = path.with_name(path.name + ".column.npz")
    effective.save_checkpoint(checkpoint, prediction.final_state)
    arrays = {
        "parameters": result.parameters,
        "singular_values": result.information.singular_values,
        "combinations": result.information.combinations,
        "identifiable": result.information.identifiable,
        "covariance_on_identifiable_subspace": result.information.covariance_on_identifiable_subspace,
        "unidentifiable_projector": result.information.unidentifiable_projector,
    }
    return write_array_archive(
        path,
        manifest={
            "kind": "interactive-column-inference",
            "problem_id": problem.problem_id,
            "inference_id": result.inference_id,
            "experiment_index": index,
            "checkpoint": checkpoint.name,
            "state_id": array_tree_fingerprint(prediction.final_state),
            "evidence_id": array_tree_fingerprint(arrays),
            "approximation": result.information.approximation,
        },
        arrays=arrays,
    )


def load_column_inference(path, problem):
    """Validate lineage and load the actual native physical continuation unchanged."""
    path = Path(path)
    manifest, arrays = read_array_archive(path)
    if (
        manifest["kind"] != "interactive-column-inference"
        or manifest["problem_id"] != problem.problem_id
    ):
        raise ValueError("Inference model, parameter, or observation lineage mismatch.")
    if manifest["checkpoint"] != path.name + ".column.npz" or manifest[
        "evidence_id"
    ] != array_tree_fingerprint(arrays):
        raise ValueError("Inference evidence or native checkpoint binding differs.")
    index = manifest["experiment_index"]
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(problem.data)
    ):
        raise ValueError(
            "Continuation experiment index must be a nonnegative in-range integer, not bool."
        )
    problem.space.check(arrays["parameters"])
    plan = problem.space.apply(problem.plan, jnp.asarray(arrays["parameters"]))
    effective, _, _ = problem.data[index].experiment.prepared_plan_state(plan)
    state = effective.load_checkpoint(path.with_name(manifest["checkpoint"]))
    if array_tree_fingerprint(state) != manifest["state_id"]:
        raise ValueError(
            "Native continuation inventories differ from inference artifact."
        )
    return {
        "parameters": jnp.asarray(arrays["parameters"]),
        "plan": effective,
        "state": state,
        "evidence": arrays,
        "manifest": manifest,
    }


__all__ = [
    "ColumnParameterSpace",
    "ColumnIntervention",
    "ColumnObservationBinding",
    "ColumnExperiment",
    "ColumnPrediction",
    "ColumnObservationData",
    "ColumnLocalInformation",
    "ColumnCalibrationProblem",
    "ColumnCalibrationResult",
    "ColumnDesignCandidate",
    "ColumnDesignResult",
    "column_local_information",
    "column_gradient_audit",
    "design_column_intervention",
    "score_column_holdout",
    "save_column_inference",
    "load_column_inference",
]

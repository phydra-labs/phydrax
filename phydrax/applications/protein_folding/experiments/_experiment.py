# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-validated conditions, named sharing, and prepared observation operators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....ein import contract
from ....observation import CholeskyCovarianceAction, CoordinateLayout
from ....qualification import ReferenceArtifactManifest
from ....units import conversion_factor, derived_unit, KELVIN, ONE, SECOND, UnitDefinition
from ._models import (
    ChevronKinetics,
    DimerThreeStateUnfolding,
    DimerTwoStateUnfolding,
    EquilibriumModel,
    KineticModel,
    ParallelPathKinetics,
    RepeatTransferUnfolding,
    ThermodynamicConvention,
    ThreeStateUnfolding,
    TwoStateUnfolding,
)


def _identifier(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty trimmed identifier.")
    return value


class ExperimentConditions(StrictModule):
    """Paired condition rows, not a Cartesian-product or a physical trajectory."""

    temperature: Array
    denaturant: Array
    concentration: Array
    convention: ThermodynamicConvention = eqx.field(static=True)

    def __init__(
        self,
        temperature,
        denaturant,
        /,
        *,
        convention=ThermodynamicConvention(),
        temperature_unit=KELVIN,
        denaturant_unit=None,
        concentration=None,
        concentration_unit=None,
    ):
        t = np.asarray(temperature, dtype=float) * float(
            conversion_factor(temperature_unit, KELVIN)
        )
        d_unit = (
            convention.concentration_unit if denaturant_unit is None else denaturant_unit
        )
        c_unit = (
            convention.concentration_unit
            if concentration_unit is None
            else concentration_unit
        )
        d = np.asarray(denaturant, dtype=float) * float(
            conversion_factor(d_unit, convention.concentration_unit)
        )
        t, d = np.broadcast_arrays(t, d)
        if (
            t.ndim != 1
            or not t.size
            or np.any(~np.isfinite(t))
            or np.any(t <= 0)
            or np.any(~np.isfinite(d))
            or np.any(d < 0)
        ):
            raise ValueError(
                "Conditions need nonempty paired finite vectors, T>0 Kelvin and denaturant>=0."
            )
        c = (
            np.zeros_like(t)
            if concentration is None
            else np.broadcast_to(np.asarray(concentration, dtype=float), t.shape)
            * float(conversion_factor(c_unit, convention.concentration_unit))
        )
        if np.any(~np.isfinite(c)) or np.any(c < 0):
            raise ValueError("Protein concentration must be finite and nonnegative.")
        self.temperature, self.denaturant, self.concentration = map(
            jnp.asarray, (t, d, c)
        )
        self.convention = convention


@dataclass(frozen=True)
class ExperimentParameter:
    """One named physical parameter; scale conditions its free fit coordinate.

    Free coordinates z start at zero and decode to initial + scale*z. Fixed
    parameters are excluded from sensitivity rank, covariance and posterior.
    """

    name: str
    initial: float
    unit: UnitDefinition
    scale: float
    free: bool = True

    def __post_init__(self):
        _identifier(self.name, "Parameter name")
        if not isinstance(self.unit, UnitDefinition):
            raise TypeError("Parameter unit must be UnitDefinition.")
        if not isfinite(self.initial) or not isfinite(self.scale) or self.scale <= 0:
            raise ValueError(
                "Parameter initial must be finite and scale finite positive."
            )
        if not isinstance(self.free, bool):
            raise TypeError("Parameter free must be boolean.")


class NamedParameterMap(StrictModule):
    initial: Array
    scale: Array
    free_indices: Array
    names: tuple[str, ...] = eqx.field(static=True)
    free_names: tuple[str, ...] = eqx.field(static=True)
    units: tuple[UnitDefinition, ...] = eqx.field(static=True)

    def __init__(self, parameters):
        parameters = tuple(parameters)
        if not parameters or any(
            not isinstance(p, ExperimentParameter) for p in parameters
        ):
            raise TypeError("parameters must be nonempty ExperimentParameter values.")
        names = tuple(p.name for p in parameters)
        if len(set(names)) != len(names):
            raise ValueError("Shared parameters are declared once, with unique names.")
        self.initial = jnp.asarray([p.initial for p in parameters], dtype=float)
        self.scale = jnp.asarray([p.scale for p in parameters], dtype=float)
        self.free_indices = jnp.asarray(
            [i for i, p in enumerate(parameters) if p.free], dtype=jnp.int32
        )
        self.names, self.units = names, tuple(p.unit for p in parameters)
        self.free_names = tuple(p.name for p in parameters if p.free)

    def decode(self, coordinates):
        z = jnp.asarray(coordinates)
        if z.shape != (len(self.free_names),):
            raise ValueError("Fit coordinate vector must match the free parameter map.")
        return self.initial.at[self.free_indices].add(self.scale[self.free_indices] * z)

    def named_values(self, coordinates):
        values = self.decode(coordinates)
        return dict(zip(self.names, values, strict=True))


_BASELINE_TERMS = ("intercept", "temperature", "denaturant", "temperature_denaturant")


def _baseline_units(signal_unit, convention, terms):
    choices = (
        signal_unit,
        derived_unit("signal/K", ((signal_unit, 1), (KELVIN, -1))),
        derived_unit(
            "signal/concentration",
            ((signal_unit, 1), (convention.concentration_unit, -1)),
        ),
        derived_unit(
            "signal/(K concentration)",
            ((signal_unit, 1), (KELVIN, -1), (convention.concentration_unit, -1)),
        ),
    )
    return tuple(choices[_BASELINE_TERMS.index(term)] for term in terms)


@dataclass(frozen=True)
class FluorescenceExperiment:
    """Equilibrium fluorescence with group/state-specific nuisance baselines.

    Every row names a channel/replicate group explicitly. All groups see the
    same state populations at a condition. Noise is calibrated, not estimated
    from residual scatter. Correlation, when supplied, is the Cholesky factor
    over active observations in mask order, not a precision or full covariance.
    """

    name: str
    model: EquilibriumModel
    conditions: ExperimentConditions
    groups: tuple[str, ...]
    observed: ArrayLike
    standard_errors: ArrayLike
    source_id: str
    equilibrium_evidence: str
    reversible: bool
    source_kind: str = "synthetic"
    reference: ReferenceArtifactManifest | None = None
    signal_unit: UnitDefinition = ONE
    baseline_terms: tuple[str, ...] = ("intercept", "temperature", "denaturant")
    mask: ArrayLike | None = None
    covariance_cholesky: ArrayLike | None = None
    bindings: Mapping[str, str] | None = None
    commercial_use: bool = False

    def parameter_slots(self):
        groups = tuple(dict.fromkeys(self.groups))
        units = _baseline_units(
            self.signal_unit, self.model.convention, self.baseline_terms
        )
        baselines = tuple(
            (f"{self.name}.{group}.{state}.{term}", unit)
            for group in groups
            for state in self.model.state_names
            for term, unit in zip(self.baseline_terms, units, strict=True)
        )
        return self.model.parameter_slots() + baselines


@dataclass(frozen=True)
class KineticRateExperiment:
    """Isothermal relaxation-rate observations with calibrated Gaussian log noise.

    observed_log_rates are log(k * time_unit), e.g. log numerical rates in s^-1
    for time_unit=SECOND. The prepared target is log numerical rates in s^-1.
    A sum of pathways describes one relaxation mode, not resolved pathway data.
    """

    name: str
    model: KineticModel
    conditions: ExperimentConditions
    observed_log_rates: ArrayLike
    log_standard_errors: ArrayLike
    source_id: str
    source_kind: str = "synthetic"
    reference: ReferenceArtifactManifest | None = None
    time_unit: UnitDefinition = SECOND
    mask: ArrayLike | None = None
    covariance_cholesky: ArrayLike | None = None
    bindings: Mapping[str, str] | None = None
    commercial_use: bool = False

    def parameter_slots(self):
        return self.model.parameter_slots()


class PreparedProteinObservation(StrictModule):
    conditions: ExperimentConditions
    parameter_indices: Array
    parameter_factors: Array
    group_indices: Array
    baseline_features: Array
    active_indices: Array
    observed: Array
    standard_errors: Array
    covariance: CholeskyCovarianceAction | None
    layout: CoordinateLayout
    model: EquilibriumModel | KineticModel = eqx.field(static=True)
    name: str = eqx.field(static=True)
    group_count: int = eqx.field(static=True)
    model_parameter_count: int = eqx.field(static=True)
    kinetic: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    reference_id: str | None = eqx.field(static=True)
    equilibrium_evidence: str | None = eqx.field(static=True)
    group_names: tuple[str, ...] = eqx.field(static=True)
    baseline_terms: tuple[str, ...] = eqx.field(static=True)

    def _predict(self, conditions, group_indices, baseline_features, physical_parameters):
        p = physical_parameters[self.parameter_indices] * self.parameter_factors
        t, d, c = conditions.temperature, conditions.denaturant, conditions.concentration
        if self.kinetic:
            return self.model.predict_log_rate(p, t, d)
        n = self.model_parameter_count
        populations = self.model.populations(p[:n], t, d, c)
        baselines = p[n:].reshape(
            (self.group_count, len(self.model.state_names), baseline_features.shape[1])
        )
        state_signal = contract("nst,nt->ns", baselines[group_indices], baseline_features)
        return jnp.sum(populations * state_signal, axis=-1)

    def predict(self, physical_parameters):
        return self._predict(
            self.conditions,
            self.group_indices,
            self.baseline_features,
            physical_parameters,
        )

    def prepare_prediction(self, conditions, /, *, groups=None):
        """Prepare a JIT/grad-safe callable on new conditions, without fake data.

        The callable consumes the joint map's physical parameter vector. New
        channel/replicate groups are refused because their baselines were not
        fitted. This boundary is host-only; the returned callable is numeric.
        """
        if conditions.convention != self.model.convention:
            raise ValueError("Prediction and fitted model conventions disagree.")
        count = conditions.temperature.size
        if self.kinetic:
            if groups is not None:
                raise ValueError("Rate predictions do not have fluorescence groups.")
            if not np.allclose(
                np.asarray(conditions.temperature),
                self.model.convention.reference_temperature,
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(
                    "Kinetic prediction is restricted to the fitted isotherm."
                )
            indices, features = jnp.zeros(count, dtype=jnp.int32), jnp.empty((count, 0))
        else:
            if (
                groups is None
                or len(groups) != count
                or any(group not in self.group_names for group in groups)
            ):
                raise ValueError(
                    "Prediction rows must use previously fitted channel/replicate groups."
                )
            if isinstance(
                self.model, (DimerTwoStateUnfolding, DimerThreeStateUnfolding)
            ) and np.any(np.asarray(conditions.concentration) <= 0):
                raise ValueError(
                    "Normalized dimer prediction requires positive concentration."
                )
            indices = jnp.asarray(
                [self.group_names.index(group) for group in groups], dtype=jnp.int32
            )
            t = conditions.temperature - self.model.convention.reference_temperature
            d = conditions.denaturant
            choices = (jnp.ones(count), t, d, t * d)
            features = jnp.stack(
                tuple(
                    choices[_BASELINE_TERMS.index(term)] for term in self.baseline_terms
                ),
                axis=-1,
            )
        return eqx.Partial(self._predict, conditions, indices, features)

    def residual(self, physical_parameters):
        residual = self.predict(physical_parameters)[self.active_indices] - self.observed
        if self.covariance is not None:
            return self.covariance.whiten(residual)
        return residual / self.standard_errors

    def log_likelihood(self, physical_parameters):
        residual = self.residual(physical_parameters)
        logdet = (
            self.covariance.logdet_covariance
            if self.covariance is not None
            else 2 * jnp.sum(jnp.log(self.standard_errors))
        )
        logp = -0.5 * (
            jnp.sum(residual * residual)
            + logdet
            + self.observed.size * jnp.log(2 * jnp.pi)
        )
        return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)


def _prepare_observation(plan, parameters):
    _identifier(plan.name, "Experiment name")
    _identifier(plan.source_id, "Source identity")
    if plan.source_kind not in ("synthetic", "experimental"):
        raise ValueError("source_kind must be synthetic or experimental.")
    if plan.source_kind == "experimental" and plan.reference is None:
        raise ValueError(
            "Experimental data need a rights-cleared reference and measured uncertainty."
        )
    reference_id = None
    if plan.reference is not None:
        reference_id = plan.reference.require_rights(commercial_use=plan.commercial_use)
        if plan.source_kind == "experimental":
            plan.reference.require_uncertainty()
    kinetic = isinstance(plan, KineticRateExperiment)
    model = plan.model
    if model.convention != plan.conditions.convention:
        raise ValueError(
            "Model and prepared condition unit/standard-state conventions disagree."
        )
    count = plan.conditions.temperature.size
    if kinetic:
        if not isinstance(model, (ChevronKinetics, ParallelPathKinetics)):
            raise TypeError(
                "Kinetic data require a named chevron or parallel-path model."
            )
        if not np.allclose(
            np.asarray(plan.conditions.temperature),
            model.convention.reference_temperature,
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                "Chevron/parallel models are isothermal; no activation-enthalpy law is supplied."
            )
        observed = np.asarray(plan.observed_log_rates, dtype=float) - np.log(
            float(conversion_factor(plan.time_unit, SECOND))
        )
        errors = np.asarray(plan.log_standard_errors, dtype=float)
        group_indices, features, group_count = (
            np.zeros(count, dtype=np.int32),
            np.empty((count, 0)),
            0,
        )
        group_names, baseline_terms = (), ()
        evidence = None
    else:
        if not isinstance(
            model, (TwoStateUnfolding, ThreeStateUnfolding, RepeatTransferUnfolding)
        ):
            raise TypeError("Fluorescence data require a named equilibrium model.")
        if plan.reversible is not True:
            raise ValueError(
                "Equilibrium fitting refuses irreversible/aggregation/ramp-dependent data."
            )
        evidence = _identifier(
            plan.equilibrium_evidence, "Equilibrium applicability evidence"
        )
        if isinstance(
            model, (DimerTwoStateUnfolding, DimerThreeStateUnfolding)
        ) and np.any(np.asarray(plan.conditions.concentration) <= 0):
            raise ValueError(
                "Normalized dimer fluorescence requires positive total monomer concentration."
            )
        if len(plan.groups) != count:
            raise ValueError("Each condition row must name its channel/replicate group.")
        group_names = tuple(dict.fromkeys(plan.groups))
        baseline_terms = tuple(plan.baseline_terms)
        for group in group_names:
            _identifier(group, "Channel/replicate group")
        if (
            not plan.baseline_terms
            or len(set(plan.baseline_terms)) != len(plan.baseline_terms)
            or any(term not in _BASELINE_TERMS for term in plan.baseline_terms)
        ):
            raise ValueError(
                "Baseline terms must be distinct named thermal/denaturant features."
            )
        t = (
            np.asarray(plan.conditions.temperature)
            - model.convention.reference_temperature
        )
        d = np.asarray(plan.conditions.denaturant)
        all_features = (np.ones(count), t, d, t * d)
        features = np.stack(
            [all_features[_BASELINE_TERMS.index(term)] for term in plan.baseline_terms],
            axis=-1,
        )
        group_indices = np.asarray(
            [group_names.index(group) for group in plan.groups], dtype=np.int32
        )
        group_count = len(group_names)
        observed, errors = (
            np.asarray(plan.observed, dtype=float),
            np.asarray(plan.standard_errors, dtype=float),
        )
    observed, errors = np.broadcast_arrays(observed, errors)
    if observed.shape != (count,):
        raise ValueError("Measurements and calibrated errors must match condition rows.")
    mask = np.ones(count, dtype=bool) if plan.mask is None else np.asarray(plan.mask)
    if mask.dtype != bool or mask.shape != (count,) or not np.any(mask):
        raise ValueError("Observation mask must be a nonempty boolean selection of rows.")
    active = np.flatnonzero(mask)
    if (
        np.any(~np.isfinite(observed[active]))
        or np.any(~np.isfinite(errors[active]))
        or np.any(errors[active] <= 0)
    ):
        raise ValueError(
            "Active measurements need finite values and calibrated positive errors."
        )
    slots = plan.parameter_slots()
    binding = {} if plan.bindings is None else dict(plan.bindings)
    if set(binding) - {name for name, _ in slots}:
        raise ValueError("Parameter bindings contain unknown local slots.")
    indices, factors = [], []
    for local, unit in slots:
        name = binding.get(local, local)
        if name not in parameters.names:
            raise ValueError(f"Missing named parameter {name!r} for {local!r}.")
        index = parameters.names.index(name)
        indices.append(index)
        factors.append(float(conversion_factor(parameters.units[index], unit)))
    layout = CoordinateLayout(tuple(f"{plan.name}:{i}" for i in active))
    covariance = None
    if plan.covariance_cholesky is not None:
        root = np.asarray(plan.covariance_cholesky, dtype=float)
        if (
            root.shape != (active.size, active.size)
            or np.any(~np.isfinite(root))
            or np.any(np.triu(root, 1) != 0)
            or np.any(np.diag(root) <= 0)
        ):
            raise ValueError(
                "Covariance root must be lower triangular and positive over active rows."
            )
        if not np.allclose(
            np.sum(root * root, axis=1), errors[active] ** 2, rtol=1e-6, atol=0
        ):
            raise ValueError(
                "Covariance marginal variances disagree with calibrated standard errors."
            )
        covariance = CholeskyCovarianceAction(root, layout)
    return PreparedProteinObservation(
        plan.conditions,
        jnp.asarray(indices, dtype=jnp.int32),
        jnp.asarray(factors),
        jnp.asarray(group_indices),
        jnp.asarray(features),
        jnp.asarray(active, dtype=jnp.int32),
        jnp.asarray(observed[active]),
        jnp.asarray(errors[active]),
        covariance,
        layout,
        model,
        plan.name,
        group_count,
        len(model.parameter_slots()),
        kinetic,
        plan.source_id,
        reference_id,
        evidence,
        group_names,
        baseline_terms,
    )


class PreparedProteinExperiments(StrictModule):
    """Joint fixed-shape fit/predict target, sharing parameters by explicit name."""

    observations: tuple[PreparedProteinObservation, ...]
    parameters: NamedParameterMap
    problem_id: str = eqx.field(static=True)

    def predict(self, coordinates):
        values = self.parameters.decode(coordinates)
        return tuple(observation.predict(values) for observation in self.observations)

    def residual(self, coordinates):
        values = self.parameters.decode(coordinates)
        return jnp.concatenate(
            tuple(observation.residual(values) for observation in self.observations)
        )

    def log_likelihood(self, coordinates):
        values = self.parameters.decode(coordinates)
        return sum(
            observation.log_likelihood(values) for observation in self.observations
        )

    @property
    def initial_coordinates(self):
        return jnp.zeros(
            (len(self.parameters.free_names),), dtype=self.parameters.initial.dtype
        )


def prepare_protein_experiments(experiments, parameters):
    """Host preparation for the finite named equilibrium/kinetic model family."""
    plans = tuple(experiments)
    if not plans or any(
        not isinstance(plan, (FluorescenceExperiment, KineticRateExperiment))
        for plan in plans
    ):
        raise TypeError(
            "experiments must contain named fluorescence/kinetic observation plans."
        )
    if len({plan.name for plan in plans}) != len(plans):
        raise ValueError("Experiment names must be unique in a joint problem.")
    mapping = NamedParameterMap(parameters)
    observations = tuple(_prepare_observation(plan, mapping) for plan in plans)
    used = set(
        int(index)
        for observation in observations
        for index in np.asarray(observation.parameter_indices)
    )
    if used != set(range(len(mapping.names))):
        raise ValueError("Unused named parameters must not enter the inference problem.")
    problem_id = canonical_fingerprint(
        {
            "kind": "protein-experimental-inference",
            "names": mapping.names,
            "units": tuple(unit.unit_id for unit in mapping.units),
            "free": mapping.free_names,
            "numeric": array_tree_fingerprint(
                eqx.filter((observations, mapping), eqx.is_array)
            ),
            "models": tuple(
                (
                    type(o.model).__name__,
                    repr(o.model),
                    o.name,
                    o.source_id,
                    o.reference_id,
                    o.equilibrium_evidence,
                )
                for o in observations
            ),
        }
    )
    return PreparedProteinExperiments(observations, mapping, problem_id)

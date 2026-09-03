#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._model import AbstractArrayModel
from phydrax._strict import StrictModule
from phydrax.equations import (
    ChemicalConditionalAffineDrivers,
    ChemicalConditionalAffineResult,
    ChemicalRateRuntime,
    PreparedChemicalConditionalAffine,
)
from phydrax.linalg import MatrixFunctionPolicy
from phydrax.nn._keys import EvalKey, split_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.operator.data import OperatorBatch, OperatorOutputSpec
from phydrax.nn.operator.engine import AbstractOperatorModel


DriverOutputTransform = Literal["direct", "softplus"]


class ChemicalConditionalAffineScaling(StrictModule):
    state_scale: Array
    driver_scale: Array
    duration_scale: Array
    driver_output_transform: DriverOutputTransform = eqx.field(static=True)
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_scale: ArrayLike,
        driver_scale: ArrayLike,
        duration_scale: ArrayLike,
        /,
        *,
        driver_output_transform: DriverOutputTransform = "direct",
        scaling_id: str | None = None,
    ):
        state = jnp.asarray(state_scale)
        if state.ndim != 1:
            raise ValueError("state_scale must be one-dimensional.")
        if not jnp.issubdtype(state.dtype, jnp.inexact):
            state = state.astype(float)
        drivers = jnp.asarray(driver_scale, dtype=state.dtype)
        duration = jnp.asarray(duration_scale, dtype=state.dtype)
        if drivers.ndim != 1:
            raise ValueError("driver_scale must be one-dimensional.")
        if duration.shape != ():
            raise ValueError("duration_scale must be scalar.")
        if (
            np.any(~np.isfinite(np.asarray(state)))
            or np.any(np.asarray(state) <= 0.0)
            or np.any(~np.isfinite(np.asarray(drivers)))
            or np.any(np.asarray(drivers) <= 0.0)
            or not np.isfinite(float(np.asarray(duration)))
            or float(np.asarray(duration)) <= 0.0
        ):
            raise ValueError("All conditional-affine scales must be finite and positive.")
        if driver_output_transform not in ("direct", "softplus"):
            raise ValueError("driver_output_transform must be 'direct' or 'softplus'.")
        generated = canonical_fingerprint(
            {
                "kind": "chemical-conditional-affine-scaling",
                "state": array_tree_fingerprint(state),
                "drivers": array_tree_fingerprint(drivers),
                "duration": array_tree_fingerprint(duration),
                "driver_output_transform": driver_output_transform,
            }
        )
        identifier = generated if scaling_id is None else str(scaling_id)
        if not identifier:
            raise ValueError("scaling_id must be non-empty.")
        self.state_scale = state
        self.driver_scale = drivers
        self.duration_scale = duration
        self.driver_output_transform = driver_output_transform
        self.scaling_id = identifier

    def scale_state(self, state: Array, /) -> Array:
        return state / self.state_scale

    def scale_duration(self, duration: Array, /) -> Array:
        return jnp.log1p(duration / self.duration_scale)

    def physical_drivers(self, raw: Array, /) -> Array:
        normalized = (
            jax.nn.softplus(raw) if self.driver_output_transform == "softplus" else raw
        )
        return normalized * self.driver_scale


class StoichiometricRateCorrection(StrictModule):
    context_model: AbstractArrayModel
    species_model: AbstractArrayModel
    species_features: Array
    net_stoichiometry: Array
    strength: Array
    log_multiplier_bound: float | None = eqx.field(static=True)
    correction_id: str = eqx.field(static=True)

    def __init__(
        self,
        context_model: AbstractArrayModel,
        species_model: AbstractArrayModel,
        species_features: ArrayLike,
        net_stoichiometry: ArrayLike,
        /,
        *,
        log_multiplier_bound: float | None = None,
        correction_id: str | None = None,
    ):
        if not isinstance(context_model, AbstractArrayModel) or not isinstance(
            species_model, AbstractArrayModel
        ):
            raise TypeError(
                "context_model and species_model must be AbstractArrayModel values."
            )
        features = jnp.asarray(species_features)
        stoichiometry = jnp.asarray(net_stoichiometry, dtype=features.dtype)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("species_features must have shape (species, features).")
        if stoichiometry.ndim != 2 or stoichiometry.shape[1] != features.shape[0]:
            raise ValueError(
                "net_stoichiometry must have shape (reactions, species_features rows)."
            )
        latent_size = _get_size(context_model.out_size)
        if _get_size(species_model.in_size) != features.shape[1]:
            raise ValueError("species_model input size must match species features.")
        if _get_size(species_model.out_size) != latent_size + 1:
            raise ValueError(
                "species_model output size must equal context latent size plus one bias."
            )
        bound = None if log_multiplier_bound is None else float(log_multiplier_bound)
        if bound is not None and (not np.isfinite(bound) or bound <= 0.0):
            raise ValueError("log_multiplier_bound must be finite and positive or None.")
        generated = canonical_fingerprint(
            {
                "kind": "stoichiometric-rate-correction",
                "species_features": array_tree_fingerprint(features),
                "net_stoichiometry": array_tree_fingerprint(stoichiometry),
                "latent_size": latent_size,
                "log_multiplier_bound": bound,
            }
        )
        identifier = generated if correction_id is None else str(correction_id)
        if not identifier:
            raise ValueError("correction_id must be non-empty.")
        self.context_model = context_model
        self.species_model = species_model
        self.species_features = features
        self.net_stoichiometry = stoichiometry
        self.strength = jnp.asarray(0.0, dtype=features.dtype)
        self.log_multiplier_bound = bound
        self.correction_id = identifier

    @property
    def reaction_count(self) -> int:
        return int(self.net_stoichiometry.shape[0])

    @property
    def species_count(self) -> int:
        return int(self.net_stoichiometry.shape[1])

    def __call__(
        self,
        scaled_state: Array,
        scaled_duration: Array,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        if scaled_state.shape[:-1] != scaled_duration.shape:
            raise ValueError(
                "scaled_state and scaled_duration leading shapes must match."
            )
        context_features = jnp.concatenate(
            (scaled_state, scaled_duration[..., None]), axis=-1
        )
        if context_features.shape[-1] != _get_size(self.context_model.in_size):
            raise ValueError(
                "Rate-correction context feature size does not match its model."
            )
        context_key, species_key = split_eval_key(key, 2)
        flat_context = context_features.reshape((-1, context_features.shape[-1]))
        dynamic = jax.vmap(lambda value: self.context_model(value, key=context_key))(
            flat_context
        )
        latent_size = _get_size(self.context_model.out_size)
        dynamic = jnp.asarray(dynamic).reshape(
            context_features.shape[:-1] + (latent_size,)
        )
        encoded_species = jax.vmap(
            lambda value: self.species_model(value, key=species_key)
        )(self.species_features)
        static_latent = encoded_species[..., :latent_size]
        static_bias = encoded_species[..., latent_size]
        potentials = contract("...p,sp->...s", dynamic, static_latent) + static_bias
        raw_log_multiplier = contract("rs,...s->...r", self.net_stoichiometry, potentials)
        log_multiplier = self.strength * raw_log_multiplier
        if self.log_multiplier_bound is not None:
            log_multiplier = self.log_multiplier_bound * jnp.tanh(
                log_multiplier / self.log_multiplier_bound
            )
        return jnp.exp(log_multiplier)


class ChemicalConditionalAffineOperator(AbstractOperatorModel):
    operator_architecture = "ChemicalConditionalAffineOperator"

    chemistry: PreparedChemicalConditionalAffine
    driver_model: AbstractOperatorModel
    scaling: ChemicalConditionalAffineScaling
    rate_correction: StoichiometricRateCorrection | None
    matrix_function_policy: MatrixFunctionPolicy
    runtime: ChemicalRateRuntime
    state_name: str = eqx.field(static=True)
    temperature_name: str = eqx.field(static=True)
    pressure_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        chemistry: PreparedChemicalConditionalAffine,
        driver_model: AbstractOperatorModel,
        scaling: ChemicalConditionalAffineScaling,
        /,
        *,
        rate_correction: StoichiometricRateCorrection | None = None,
        matrix_function_policy: MatrixFunctionPolicy | None = None,
        runtime: ChemicalRateRuntime | None = None,
        state_name: str = "state",
        temperature_name: str = "temperature",
        pressure_name: str = "pressure",
        query_name: str = "time",
    ):
        if not isinstance(chemistry, PreparedChemicalConditionalAffine):
            raise TypeError("chemistry must be PreparedChemicalConditionalAffine.")
        if not isinstance(driver_model, AbstractOperatorModel):
            raise TypeError("driver_model must be AbstractOperatorModel.")
        if not isinstance(scaling, ChemicalConditionalAffineScaling):
            raise TypeError("scaling must be ChemicalConditionalAffineScaling.")
        if _get_size(driver_model.out_size) != chemistry.driver_size:
            raise ValueError("driver_model output size must match chemistry driver size.")
        species_count = chemistry.mechanism.schema.species_count
        if scaling.state_scale.shape != (species_count,):
            raise ValueError("state scaling must match the complete species axis.")
        if scaling.driver_scale.shape != (chemistry.driver_size,):
            raise ValueError("driver scaling must match the driver species axis.")
        if rate_correction is not None:
            if not isinstance(rate_correction, StoichiometricRateCorrection):
                raise TypeError(
                    "rate_correction must be StoichiometricRateCorrection or None."
                )
            if rate_correction.reaction_count != chemistry.mechanism.reaction_count:
                raise ValueError("rate_correction reaction count must match chemistry.")
            if rate_correction.species_count != species_count:
                raise ValueError("rate_correction species count must match chemistry.")
        policy = (
            MatrixFunctionPolicy()
            if matrix_function_policy is None
            else matrix_function_policy
        )
        if not isinstance(policy, MatrixFunctionPolicy):
            raise TypeError("matrix_function_policy must be MatrixFunctionPolicy.")
        runtime_ = ChemicalRateRuntime() if runtime is None else runtime
        if not isinstance(runtime_, ChemicalRateRuntime):
            raise TypeError("runtime must be ChemicalRateRuntime.")
        names = tuple(
            str(value)
            for value in (state_name, temperature_name, pressure_name, query_name)
        )
        if any(not value for value in names) or len(set(names)) != len(names):
            raise ValueError("Conditional-affine source and query names must be unique.")
        self.chemistry = chemistry
        self.driver_model = driver_model
        self.scaling = scaling
        self.rate_correction = rate_correction
        self.matrix_function_policy = policy
        self.runtime = runtime_
        self.state_name, self.temperature_name, self.pressure_name, self.query_name = (
            names
        )
        self.in_size = species_count
        self.out_size = species_count

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        return {
            "output": OperatorOutputSpec(
                self.out_size,
                component_names=self.chemistry.mechanism.schema.species_names,
            )
        }

    def _source_state(self, batch: OperatorBatch, /) -> Array:
        source = batch.input(self.state_name)
        if source.sample_shape or source.values is None:
            raise ValueError("Conditional-affine state input must be one abstract value.")
        values = jnp.asarray(source.values)
        expected = batch.case_shape + (self.out_size,)
        if values.shape != expected:
            raise ValueError(
                f"State input must have shape {expected}; got {values.shape}."
            )
        return values

    def _scalar_source(self, batch: OperatorBatch, name: str, /) -> Array:
        source = batch.input(name)
        if source.sample_shape or source.values is None:
            raise ValueError(
                f"Conditional-affine input {name!r} must be abstract scalar data."
            )
        values = jnp.asarray(source.values)
        expected = batch.case_shape
        if values.shape != expected:
            raise ValueError(
                f"Input {name!r} must have shape {expected}; got {values.shape}."
            )
        return values

    def _query_duration(self, batch: OperatorBatch, /) -> tuple[Array, Array]:
        if batch.single_query_name() != self.query_name:
            raise ValueError(f"Expected query {self.query_name!r}.")
        query = batch.query(self.query_name)
        if query.coordinates is None or int(query.coordinates.shape[-1]) != 1:
            raise ValueError(
                "Conditional-affine query requires one-dimensional point coordinates."
            )
        coordinates = query.coordinates_array(case_shape=batch.case_shape)
        duration = coordinates[..., 0]
        mask = query.mask_array(case_shape=batch.case_shape)
        return jnp.where(mask, duration, 0.0), mask

    def _driver_batch(
        self,
        batch: OperatorBatch,
        state: Array,
        duration: Array,
        /,
    ) -> OperatorBatch:
        scaled_state = self.scaling.scale_state(state)
        midpoint_coordinates = self.scaling.scale_duration(0.5 * duration)[..., None]
        scaled = eqx.tree_at(
            lambda value: value.inputs[self.state_name].values,
            batch,
            scaled_state,
        )
        return eqx.tree_at(
            lambda value: value.queries[self.query_name].coordinates,
            scaled,
            midpoint_coordinates,
        )

    def predict_drivers(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        state = self._source_state(batch)
        duration, _ = self._query_duration(batch)
        driver_batch = self._driver_batch(batch, state, duration)
        raw = jnp.asarray(
            self.driver_model.__call_operator_batch__(driver_batch, key=key)
        )
        expected = (
            batch.case_shape
            + batch.query(self.query_name).sample_shape
            + (self.chemistry.driver_size,)
        )
        if raw.shape != expected:
            raise ValueError(
                f"driver_model must return shape {expected}; got {raw.shape}."
            )
        return self.scaling.physical_drivers(raw)

    def transition_with_drivers(
        self,
        batch: OperatorBatch,
        drivers: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> ChemicalConditionalAffineResult:
        state = self._source_state(batch)
        temperature = self._scalar_source(batch, self.temperature_name)
        pressure = self._scalar_source(batch, self.pressure_name)
        duration, _ = self._query_duration(batch)
        query_shape = batch.query(self.query_name).sample_shape
        expected_drivers = batch.case_shape + query_shape + (self.chemistry.driver_size,)
        driver_values = jnp.asarray(drivers, dtype=state.dtype)
        if driver_values.shape != expected_drivers:
            raise ValueError(
                f"drivers must have shape {expected_drivers}; got {driver_values.shape}."
            )
        expanded_state = jnp.broadcast_to(
            state.reshape(batch.case_shape + (1,) * len(query_shape) + (self.out_size,)),
            batch.case_shape + query_shape + (self.out_size,),
        )
        expanded_temperature = jnp.broadcast_to(
            temperature.reshape(batch.case_shape + (1,) * len(query_shape)),
            batch.case_shape + query_shape,
        )
        expanded_pressure = jnp.broadcast_to(
            pressure.reshape(batch.case_shape + (1,) * len(query_shape)),
            batch.case_shape + query_shape,
        )
        physical_drivers = ChemicalConditionalAffineDrivers(
            driver_values,
            expanded_temperature,
            expanded_pressure,
            runtime=self.runtime,
        )
        reaction_multiplier = None
        if self.rate_correction is not None:
            scaled_state = self.scaling.scale_state(expanded_state)
            scaled_duration = self.scaling.scale_duration(duration)
            reaction_multiplier = self.rate_correction(
                scaled_state,
                scaled_duration,
                key=key,
            )
        return self.chemistry.advance(
            expanded_state,
            physical_drivers,
            duration,
            reaction_multiplier=reaction_multiplier,
            policy=self.matrix_function_policy,
        )

    def evaluate_transition(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> ChemicalConditionalAffineResult:
        driver_key, transition_key = split_eval_key(key, 2)
        drivers = self.predict_drivers(batch, key=driver_key)
        return self.transition_with_drivers(batch, drivers, key=transition_key)

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        result = self.evaluate_transition(batch, key=key)
        _, mask = self._query_duration(batch)
        value = eqx.error_if(
            result.candidate_state,
            jnp.any(mask & ~result.successful),
            "Chemical conditional-affine transition failed.",
        )
        return jnp.where(mask[..., None], value, jnp.zeros((), dtype=value.dtype))

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError(
                "ChemicalConditionalAffineOperator requires OperatorBatch input."
            )
        return self.__call_operator_batch__(x, key=key)


__all__ = [
    "ChemicalConditionalAffineOperator",
    "ChemicalConditionalAffineScaling",
    "StoichiometricRateCorrection",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._probability import AbstractProbabilityLaw
from ...._strict import StrictModule
from ....uq import (
    AbstractBijector,
    NestedPriorPlan,
    ParameterSpace,
    PeriodicNestedCoordinate,
)


def component_masses_from_chirp_mass_mass_ratio(
    chirp_mass: ArrayLike, mass_ratio: ArrayLike, /
) -> tuple[Array, Array]:
    chirp = jnp.asarray(chirp_mass)
    ratio = jnp.asarray(mass_ratio)
    valid = (
        jnp.isfinite(chirp)
        & jnp.isfinite(ratio)
        & (chirp > 0.0)
        & (ratio > 0.0)
        & (ratio <= 1.0)
    )
    safe_chirp = jnp.where(chirp > 0.0, chirp, 1.0)
    safe_ratio = jnp.where((ratio > 0.0) & (ratio <= 1.0), ratio, 1.0)
    primary = safe_chirp * (1.0 + safe_ratio) ** (1.0 / 5.0) / safe_ratio ** (3.0 / 5.0)
    secondary = safe_ratio * primary
    primary = eqx.error_if(
        primary, ~valid, "Chirp mass and mass ratio are outside physical support."
    )
    return primary, secondary


def chirp_mass_from_component_masses(
    primary_mass: ArrayLike, secondary_mass: ArrayLike, /
) -> Array:
    primary = jnp.asarray(primary_mass)
    secondary = jnp.asarray(secondary_mass)
    valid = (
        jnp.isfinite(primary)
        & jnp.isfinite(secondary)
        & (primary > 0.0)
        & (secondary > 0.0)
        & (primary >= secondary)
    )
    total = jnp.where(valid, primary + secondary, 1.0)
    value = (primary * secondary) ** (3.0 / 5.0) / total ** (1.0 / 5.0)
    return eqx.error_if(
        value, ~valid, "Component masses must satisfy primary >= secondary > 0."
    )


def mass_ratio_from_component_masses(
    primary_mass: ArrayLike, secondary_mass: ArrayLike, /
) -> Array:
    primary = jnp.asarray(primary_mass)
    secondary = jnp.asarray(secondary_mass)
    valid = (
        jnp.isfinite(primary)
        & jnp.isfinite(secondary)
        & (primary > 0.0)
        & (secondary > 0.0)
        & (primary >= secondary)
    )
    value = secondary / jnp.where(primary > 0.0, primary, 1.0)
    return eqx.error_if(
        value, ~valid, "Component masses must satisfy primary >= secondary > 0."
    )


def symmetric_mass_ratio(primary_mass: ArrayLike, secondary_mass: ArrayLike, /) -> Array:
    primary = jnp.asarray(primary_mass)
    secondary = jnp.asarray(secondary_mass)
    total = primary + secondary
    valid = jnp.isfinite(total) & (primary > 0.0) & (secondary > 0.0)
    value = primary * secondary / jnp.where(valid, total * total, 1.0)
    return eqx.error_if(
        value, ~valid, "Symmetric mass ratio requires positive finite masses."
    )


def detector_frame_mass(source_frame_mass: ArrayLike, redshift: ArrayLike, /) -> Array:
    mass = jnp.asarray(source_frame_mass)
    z = jnp.asarray(redshift)
    valid = jnp.isfinite(mass) & jnp.isfinite(z) & (mass > 0.0) & (z >= 0.0)
    value = mass * (1.0 + z)
    return eqx.error_if(
        value, ~valid, "Source mass and redshift are outside physical support."
    )


def source_frame_mass(detector_mass: ArrayLike, redshift: ArrayLike, /) -> Array:
    mass = jnp.asarray(detector_mass)
    z = jnp.asarray(redshift)
    valid = jnp.isfinite(mass) & jnp.isfinite(z) & (mass > 0.0) & (z >= 0.0)
    value = mass / jnp.where(1.0 + z > 0.0, 1.0 + z, 1.0)
    return eqx.error_if(
        value, ~valid, "Detector mass and redshift are outside physical support."
    )


def default_extrinsic_parameters(
    parameters: PyTree[Any], /
) -> tuple[Array, Array, Array, Array]:
    if not isinstance(parameters, Mapping):
        raise TypeError("Default gravitational-wave parameters must be a mapping.")
    required = ("right_ascension", "declination", "polarization", "geocent_time")
    if any(name not in parameters for name in required):
        raise KeyError(f"Extrinsic parameters require {required}.")
    return tuple(jnp.asarray(parameters[name]).reshape(()) for name in required)  # type: ignore[return-value]


class GravitationalWaveParameterPlan(StrictModule):
    """One source of truth for posterior coordinates and nested topology."""

    parameter_space: ParameterSpace
    nested_prior: NestedPriorPlan
    waveform_parameters_fn: Callable[[PyTree[Any]], PyTree[Any]] = eqx.field(static=True)
    extrinsic_parameters_fn: Callable[
        [PyTree[Any]], tuple[Array, Array, Array, Array]
    ] = eqx.field(static=True)
    derived_parameters_fn: Callable[[PyTree[Any]], Mapping[str, ArrayLike]] = eqx.field(
        static=True
    )
    parameterization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial: PyTree[Any],
        priors: PyTree[AbstractProbabilityLaw],
        /,
        *,
        continuous_paths: Sequence[str],
        waveform_parameters: Callable[[PyTree[Any]], PyTree[Any]],
        parameterization_id: str,
        periodic: Sequence[tuple[str, float, float]] = (),
        finite_supports: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
        bijectors: PyTree[AbstractBijector] | None = None,
        extrinsic_parameters: Callable[
            [PyTree[Any]], tuple[Array, Array, Array, Array]
        ] = default_extrinsic_parameters,
        derived_parameters: Callable[[PyTree[Any]], Mapping[str, ArrayLike]]
        | None = None,
    ):
        if not callable(waveform_parameters) or not callable(extrinsic_parameters):
            raise TypeError("Parameter extraction functions must be callable.")
        identity = str(parameterization_id).strip()
        if not identity:
            raise ValueError("parameterization_id must be non-empty.")
        derived = (lambda _: {}) if derived_parameters is None else derived_parameters
        if not callable(derived):
            raise TypeError("derived_parameters must be callable.")
        space = ParameterSpace(initial, priors=priors, bijectors=bijectors)
        topology = NestedPriorPlan(
            continuous_paths=tuple(continuous_paths),
            finite_supports=finite_supports,
            periodic=tuple(PeriodicNestedCoordinate(*value) for value in periodic),
        )
        self.parameter_space = space
        self.nested_prior = topology
        self.waveform_parameters_fn = waveform_parameters
        self.extrinsic_parameters_fn = extrinsic_parameters
        self.derived_parameters_fn = derived
        self.parameterization_id = identity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-parameter-plan",
                "parameterization": identity,
                "initial": array_tree_fingerprint(initial),
                "prior_types": [
                    type(value).__qualname__
                    for value in jax.tree_util.tree_leaves(
                        priors,
                        is_leaf=lambda value: isinstance(value, AbstractProbabilityLaw),
                    )
                ],
                "prior_values": array_tree_fingerprint(priors),
                "nested_prior": topology.plan_id,
            }
        )

    def waveform_parameters(self, physical: PyTree[Any], /) -> PyTree[Any]:
        return self.waveform_parameters_fn(physical)

    def extrinsic_parameters(
        self, physical: PyTree[Any], /
    ) -> tuple[Array, Array, Array, Array]:
        return self.extrinsic_parameters_fn(physical)

    def derived_parameters(self, physical: PyTree[Any], /) -> Mapping[str, ArrayLike]:
        return self.derived_parameters_fn(physical)


__all__ = [
    "GravitationalWaveParameterPlan",
    "chirp_mass_from_component_masses",
    "component_masses_from_chirp_mass_mass_ratio",
    "default_extrinsic_parameters",
    "detector_frame_mass",
    "mass_ratio_from_component_masses",
    "source_frame_mass",
    "symmetric_mass_ratio",
]

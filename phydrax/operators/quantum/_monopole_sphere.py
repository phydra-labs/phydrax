#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gauge-covariant continuum electrons on a monopole sphere."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, prod, sqrt

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._fingerprint import canonical_fingerprint
from ..._precision import real_precision_dtype_name
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._amplitude import LogAmplitude
from ._local import (
    AbstractLocalQuantumOperator,
    LocalOperatorEstimate,
    LocalOperatorStatus,
)


class MonopoleKineticPolicy(StrictModule, NonTrainableState):
    compute_dtype: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, *, compute_dtype: object = "float64"):
        dtype = real_precision_dtype_name(compute_dtype)
        self.compute_dtype = dtype
        self.method_id = f"monopole-sphere-selected-hessian-diagonal:dtype={dtype}"

    def local_kinetic(
        self,
        model: Callable[[Array], LogAmplitude],
        configuration: Array,
        twice_monopole_flux: int,
        radius: float,
        /,
    ) -> tuple[Array, Array]:
        shape = tuple(configuration.shape)
        flat = jnp.asarray(configuration, dtype=self.compute_dtype).reshape((-1,))
        dimension = flat.shape[0]

        def log_components(coordinates):
            amplitude = model(coordinates.reshape(shape))
            if not isinstance(amplitude, LogAmplitude):
                raise TypeError("The sphere amplitude model must return LogAmplitude.")
            if amplitude.log_abs.shape != ():
                raise ValueError(
                    "The sphere amplitude model must return one scalar amplitude."
                )
            return jnp.stack((amplitude.log_abs, jnp.angle(amplitude.phase)))

        jacobian = jax.jacrev(log_components)
        gradient_components = jacobian(flat)

        def diagonal_component(direction):
            _, directional = jax.jvp(jacobian, (flat,), (direction,))
            return jnp.sum(directional * direction[None, :], axis=1)

        diagonal = jax.vmap(diagonal_component)(jnp.eye(dimension, dtype=flat.dtype))
        gradient = (gradient_components[0] + 1.0j * gradient_components[1]).reshape(shape)
        hessian_diagonal = (diagonal[:, 0] + 1.0j * diagonal[:, 1]).reshape(shape)
        theta = flat.reshape(shape)[:, 0]
        sine = jnp.sin(theta)
        cosine = jnp.cos(theta)
        safe_sine = jnp.where(
            jnp.abs(sine) > 64.0 * jnp.finfo(flat.dtype).eps,
            sine,
            1.0,
        )
        grad_theta = gradient[:, 0]
        grad_phi = gradient[:, 1]
        laplacian_log = jnp.sum(
            hessian_diagonal[:, 0]
            + cosine / safe_sine * grad_theta
            + hessian_diagonal[:, 1] / safe_sine**2
        )
        gradient_square = jnp.sum(grad_theta**2 + grad_phi**2 / safe_sine**2)
        monopole = 0.5 * twice_monopole_flux
        magnetic = jnp.sum(
            (monopole * cosine / safe_sine) ** 2
            + 2.0j * monopole * cosine / safe_sine**2 * grad_phi
        )
        kinetic = (-laplacian_log - gradient_square + magnetic) / (2.0 * radius**2)
        amplitude = model(flat.reshape(shape))
        valid = (
            amplitude.valid
            & amplitude.nonzero
            & jnp.all(jnp.abs(sine) > 64.0 * jnp.finfo(flat.dtype).eps)
        )
        return kinetic, valid


class MonopoleSphereCoulombHamiltonian(AbstractLocalQuantumOperator):
    electron_count: int = eqx.field(static=True)
    twice_monopole_flux: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    kinetic_strength: float = eqx.field(static=True)
    interaction_strength: float = eqx.field(static=True)
    kinetic: MonopoleKineticPolicy
    configuration_shape: tuple[int, int] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        electron_count: int,
        twice_monopole_flux: int,
        /,
        *,
        radius: float | None = None,
        kinetic_strength: float = 1.0,
        interaction_strength: float = 1.0,
        kinetic: MonopoleKineticPolicy | None = None,
    ):
        count = int(electron_count)
        flux = int(twice_monopole_flux)
        radius_ = sqrt(flux / 2.0) if radius is None else float(radius)
        kinetic_strength_ = float(kinetic_strength)
        interaction_strength_ = float(interaction_strength)
        policy = MonopoleKineticPolicy() if kinetic is None else kinetic
        if not isinstance(policy, MonopoleKineticPolicy):
            raise TypeError("kinetic must be MonopoleKineticPolicy or None.")
        if (
            count < 2
            or flux < 1
            or any(
                not isfinite(value) or value <= 0.0
                for value in (radius_, kinetic_strength_, interaction_strength_)
            )
        ):
            raise ValueError("Monopole sphere Hamiltonian inputs are invalid.")
        self.electron_count = count
        self.twice_monopole_flux = flux
        self.radius = radius_
        self.kinetic_strength = kinetic_strength_
        self.interaction_strength = interaction_strength_
        self.kinetic = policy
        self.configuration_shape = (count, 2)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "monopole-sphere-coulomb-hamiltonian",
                "electron_count": count,
                "twice_monopole_flux": flux,
                "radius": radius_,
                "kinetic_strength": kinetic_strength_,
                "interaction_strength": interaction_strength_,
                "kinetic": policy.method_id,
            }
        )

    def _potential(self, configuration: Array, /) -> tuple[Array, Array]:
        theta = configuration[:, 0]
        phi = configuration[:, 1]
        sine = jnp.sin(theta)
        unit = jnp.stack(
            (sine * jnp.cos(phi), sine * jnp.sin(phi), jnp.cos(theta)),
            axis=-1,
        )
        cosine = unit @ unit.T
        pair = jnp.triu(
            jnp.ones((self.electron_count, self.electron_count), dtype=jnp.bool_),
            k=1,
        )
        chord_squared = jnp.where(pair, 2.0 - 2.0 * cosine, 1.0)
        singular = jnp.any(pair & (chord_squared <= 0.0))
        distance = jnp.sqrt(jnp.where(chord_squared > 0.0, chord_squared, 1.0))
        potential = jnp.sum(jnp.where(pair, 1.0 / distance, 0.0)) / self.radius
        return self.interaction_strength * potential, singular

    def _estimate_one(
        self,
        model: Callable[[Array], LogAmplitude],
        configuration: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        coordinate = jnp.asarray(configuration, dtype=self.kinetic.compute_dtype)
        potential, singular = self._potential(coordinate)
        kinetic, amplitude_valid = self.kinetic.local_kinetic(
            model,
            coordinate,
            self.twice_monopole_flux,
            self.radius,
        )
        raw = self.kinetic_strength * kinetic + potential
        finite = jnp.isfinite(raw)
        status = jnp.where(
            singular,
            int(LocalOperatorStatus.SINGULAR_CONFIGURATION),
            jnp.where(
                ~amplitude_valid,
                int(LocalOperatorStatus.INVALID_AMPLITUDE),
                jnp.where(
                    ~finite,
                    int(LocalOperatorStatus.NONFINITE),
                    int(LocalOperatorStatus.SUCCESS),
                ),
            ),
        ).astype(jnp.int32)
        valid = status == int(LocalOperatorStatus.SUCCESS)
        return jnp.where(valid, raw, jnp.nan + 0.0j), valid, status

    def estimate(
        self,
        model: Callable[[Array], LogAmplitude],
        configurations: Array,
        /,
    ) -> LocalOperatorEstimate:
        configs = jnp.asarray(configurations)
        batch_shape = tuple(configs.shape[:-2])
        count = prod(batch_shape) if batch_shape else 1
        flat = configs.reshape((count,) + self.configuration_shape)
        value, valid, status = jax.vmap(lambda x: self._estimate_one(model, x))(flat)
        work = jnp.full(
            (count,),
            2 * self.electron_count,
            dtype=jnp.int32,
        )
        return LocalOperatorEstimate(
            value.reshape(batch_shape),
            valid.reshape(batch_shape),
            status.reshape(batch_shape),
            work.reshape(batch_shape),
            configuration_shape=self.configuration_shape,
            operator_id=self.operator_id,
            method_id=self.kinetic.method_id,
            compute_dtype=self.kinetic.compute_dtype,
        )


def uniform_sphere_electron_walkers(
    key: Key[Array, ""],
    chain_count: int,
    electron_count: int,
    /,
    *,
    dtype=jnp.float64,
) -> Array:
    chains = int(chain_count)
    electrons = int(electron_count)
    if chains < 1 or electrons < 1:
        raise ValueError("Sphere walker counts must be positive.")
    cosine_key, phi_key = jr.split(key)
    cosine = jr.uniform(
        cosine_key,
        (chains, electrons),
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
    )
    phi = jr.uniform(
        phi_key,
        (chains, electrons),
        minval=-jnp.pi,
        maxval=jnp.pi,
        dtype=dtype,
    )
    return jnp.stack((jnp.arccos(cosine), phi), axis=-1)


__all__ = [
    "MonopoleKineticPolicy",
    "MonopoleSphereCoulombHamiltonian",
    "uniform_sphere_electron_walkers",
]

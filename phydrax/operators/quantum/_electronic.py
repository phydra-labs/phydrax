#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._precision import real_precision_dtype_name
from ..._sampling import AbstractProposal
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure
from ._amplitude import LogAmplitude
from ._local import (
    AbstractLocalQuantumOperator,
    LocalOperatorEstimate,
    LocalOperatorStatus,
)


ElectronicTraceMethod: TypeAlias = Literal["exact", "chunked-exact"]


class ElectronicKineticPolicy(StrictModule, NonTrainableState):
    """Exact coordinate-Laplacian policy for continuum electronic kinetics.

    Both methods evaluate every coordinate second derivative. ``chunked-exact``
    limits the simultaneous Hessian-vector products; it does not have
    dimension-independent cost.
    """

    trace_method: ElectronicTraceMethod = eqx.field(static=True)
    coordinate_chunk_size: int | None = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        trace_method: ElectronicTraceMethod = "exact",
        coordinate_chunk_size: int | None = None,
        compute_dtype: object = "float64",
    ):
        if trace_method not in ("exact", "chunked-exact"):
            raise ValueError("trace_method must be 'exact' or 'chunked-exact'.")
        if trace_method == "exact":
            if coordinate_chunk_size is not None:
                raise ValueError("coordinate_chunk_size is only valid for chunked-exact.")
            chunk = None
        else:
            if coordinate_chunk_size is None or int(coordinate_chunk_size) <= 0:
                raise ValueError(
                    "chunked-exact requires a positive coordinate_chunk_size."
                )
            chunk = int(coordinate_chunk_size)
        dtype = real_precision_dtype_name(compute_dtype)
        self.trace_method = trace_method
        self.coordinate_chunk_size = chunk
        self.compute_dtype = dtype
        self.method_id = (
            f"electronic-kinetic:{trace_method}:"
            f"chunk={chunk if chunk is not None else 'all'}:dtype={dtype}"
        )

    def _trace_and_gradient(
        self,
        model: Callable[[Array], LogAmplitude],
        configuration: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        shape = tuple(int(size) for size in configuration.shape)
        flat = jnp.asarray(configuration, dtype=self.compute_dtype).reshape((-1,))
        dimension = int(flat.shape[0])

        def log_components(coordinates):
            amplitude = model(coordinates.reshape(shape))
            if not isinstance(amplitude, LogAmplitude):
                raise TypeError("The electronic amplitude model must return LogAmplitude.")
            if amplitude.log_abs.shape != ():
                raise ValueError(
                    "The electronic amplitude model must return one scalar amplitude."
                )
            return jnp.stack((amplitude.log_abs, jnp.angle(amplitude.phase)))

        jacobian = jax.jacrev(log_components)
        component_gradient = jacobian(flat)

        def diagonal_component(direction):
            _, directional_jacobian = jax.jvp(jacobian, (flat,), (direction,))
            return contract("ad,d->a", directional_jacobian, direction)

        if self.trace_method == "exact":
            basis = jnp.eye(dimension, dtype=flat.dtype)
            diagonal = jax.vmap(diagonal_component)(basis)
        else:
            chunk = self.coordinate_chunk_size
            if chunk is None:
                raise RuntimeError("chunked-exact policy lost its static chunk size.")
            blocks = tuple(
                jax.vmap(diagonal_component)(
                    jax.nn.one_hot(
                        jnp.arange(start, min(start + chunk, dimension)),
                        dimension,
                        dtype=flat.dtype,
                    )
                )
                for start in range(0, dimension, chunk)
            )
            diagonal = jnp.concatenate(blocks, axis=0)
        trace = jnp.sum(diagonal, axis=0)
        complex_gradient = component_gradient[0] + 1j * component_gradient[1]
        complex_trace = trace[0] + 1j * trace[1]
        amplitude = model(flat.reshape(shape))
        if not isinstance(amplitude, LogAmplitude):
            raise TypeError("The electronic amplitude model must return LogAmplitude.")
        return complex_trace, complex_gradient, amplitude.valid & amplitude.nonzero

    def local_kinetic(
        self,
        model: Callable[[Array], LogAmplitude],
        configuration: Array,
        /,
    ) -> tuple[Array, Array]:
        trace, gradient, amplitude_valid = self._trace_and_gradient(
            model, configuration
        )
        laplacian_ratio = trace + contract("d,d->", gradient, gradient)
        return -0.5 * laplacian_ratio, amplitude_valid


class ElectronicCoulombHamiltonian(AbstractLocalQuantumOperator):
    """Nonrelativistic Born–Oppenheimer molecular Coulomb Hamiltonian.

    Electron configurations have shape ``(electron_count, 3)``. Coordinates use
    the nuclei's length unit, and returned local energies use its energy unit.
    Periodic metadata is rejected. Exact coincident electron/electron,
    electron/nucleus, or active nucleus/nucleus pairs are reported as singular.
    """

    nuclei: AtomicStructure
    kinetic: ElectronicKineticPolicy
    configuration_shape: tuple[int, int] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    electron_count: int = eqx.field(static=True)

    def __init__(
        self,
        nuclei: AtomicStructure,
        electron_count: int,
        /,
        *,
        kinetic: ElectronicKineticPolicy | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(nuclei, AtomicStructure):
            raise TypeError("nuclei must be an AtomicStructure.")
        if nuclei.has_periodic_metadata:
            raise ValueError(
                "ElectronicCoulombHamiltonian supports finite nonperiodic molecules "
                "and rejects cell or periodic metadata."
            )
        count = int(electron_count)
        if count <= 0:
            raise ValueError("electron_count must be positive.")
        policy = ElectronicKineticPolicy() if kinetic is None else kinetic
        if not isinstance(policy, ElectronicKineticPolicy):
            raise TypeError("kinetic must be an ElectronicKineticPolicy or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "electronic-coulomb-hamiltonian",
                    "structure": nuclei.structure_id,
                    "scale": nuclei.scale.scale_id,
                    "electron_count": count,
                    "kinetic": policy.method_id,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.nuclei = nuclei
        self.kinetic = policy
        self.configuration_shape = (count, 3)
        self.operator_id = identifier
        self.electron_count = count

    def _coulomb(self, electrons: Array, /) -> tuple[Array, Array]:
        dtype = jnp.dtype(self.kinetic.compute_dtype)
        coordinate = jnp.asarray(electrons, dtype=dtype)
        nuclei_mask = self.nuclei.active_mask
        nuclei_positions = jnp.where(
            nuclei_mask[:, None],
            self.nuclei.positions.astype(dtype),
            jnp.zeros((), dtype=dtype),
        )
        charges = jnp.where(
            nuclei_mask,
            self.nuclei.atomic_numbers.astype(dtype),
            jnp.zeros((), dtype=dtype),
        )

        electron_pair = jnp.triu(
            jnp.ones((self.electron_count, self.electron_count), dtype=bool), k=1
        )
        electron_delta = coordinate[:, None, :] - coordinate[None, :, :]
        electron_squared_distance = jnp.sum(electron_delta**2, axis=-1)
        electron_distance = jnp.sqrt(
            jnp.where(electron_pair, electron_squared_distance, 1.0)
        )
        electron_singular = jnp.any(electron_pair & (electron_distance == 0.0))
        safe_electron_distance = jnp.where(
            electron_pair & (electron_distance != 0.0), electron_distance, 1.0
        )
        electron_electron = jnp.sum(
            jnp.where(electron_pair, 1.0 / safe_electron_distance, 0.0)
        )

        electron_nuclear_delta = coordinate[:, None, :] - nuclei_positions[None, :, :]
        electron_nuclear_mask = jnp.broadcast_to(
            nuclei_mask[None, :],
            (self.electron_count, int(nuclei_mask.shape[0])),
        )
        electron_nuclear_squared_distance = jnp.sum(
            electron_nuclear_delta**2, axis=-1
        )
        electron_nuclear_distance = jnp.sqrt(
            jnp.where(
                electron_nuclear_mask, electron_nuclear_squared_distance, 1.0
            )
        )
        electron_nuclear_singular = jnp.any(
            electron_nuclear_mask & (electron_nuclear_distance == 0.0)
        )
        safe_electron_nuclear_distance = jnp.where(
            electron_nuclear_mask & (electron_nuclear_distance != 0.0),
            electron_nuclear_distance,
            1.0,
        )
        electron_nuclear = -jnp.sum(
            jnp.where(
                electron_nuclear_mask,
                charges[None, :] / safe_electron_nuclear_distance,
                0.0,
            )
        )

        nuclear_pair = (
            nuclei_mask[:, None]
            & nuclei_mask[None, :]
            & jnp.triu(
                jnp.ones(
                    (int(nuclei_mask.shape[0]), int(nuclei_mask.shape[0])),
                    dtype=bool,
                ),
                k=1,
            )
        )
        nuclear_delta = nuclei_positions[:, None, :] - nuclei_positions[None, :, :]
        nuclear_squared_distance = jnp.sum(nuclear_delta**2, axis=-1)
        nuclear_distance = jnp.sqrt(
            jnp.where(nuclear_pair, nuclear_squared_distance, 1.0)
        )
        nuclear_singular = jnp.any(nuclear_pair & (nuclear_distance == 0.0))
        safe_nuclear_distance = jnp.where(
            nuclear_pair & (nuclear_distance != 0.0), nuclear_distance, 1.0
        )
        nuclear_nuclear = jnp.sum(
            jnp.where(
                nuclear_pair,
                charges[:, None] * charges[None, :] / safe_nuclear_distance,
                0.0,
            )
        )

        length_factor = jnp.asarray(
            self.nuclei.scale.length_to_reference, dtype=dtype
        )
        energy_factor = jnp.asarray(
            self.nuclei.scale.energy_to_reference, dtype=dtype
        )
        potential = (
            electron_electron + electron_nuclear + nuclear_nuclear
        ) / (length_factor * energy_factor)
        singular = (
            electron_singular | electron_nuclear_singular | nuclear_singular
        )
        return potential, singular

    def _estimate_one(
        self,
        model: Callable[[Array], LogAmplitude],
        configuration: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        coordinate = jnp.asarray(configuration, dtype=self.kinetic.compute_dtype)
        potential, singular = self._coulomb(coordinate)
        kinetic, amplitude_valid = self.kinetic.local_kinetic(model, coordinate)
        length_factor = jnp.asarray(
            self.nuclei.scale.length_to_reference, dtype=coordinate.dtype
        )
        energy_factor = jnp.asarray(
            self.nuclei.scale.energy_to_reference, dtype=coordinate.dtype
        )
        kinetic = kinetic / (length_factor**2 * energy_factor)
        raw_value = kinetic + potential
        finite = jnp.isfinite(raw_value)
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
        value = jnp.where(valid, raw_value, jnp.asarray(jnp.nan, dtype=raw_value.dtype))
        return value, valid, status

    def estimate(
        self,
        model: Callable[[Array], LogAmplitude],
        configurations: Array,
        /,
    ) -> LocalOperatorEstimate:
        configs = jnp.asarray(configurations)
        batch_shape = tuple(int(size) for size in configs.shape[:-2])
        count = math.prod(batch_shape) if batch_shape else 1
        flat = configs.reshape((count,) + self.configuration_shape)
        value, valid, status = jax.vmap(lambda x: self._estimate_one(model, x))(flat)
        coordinate_work = 3 * self.electron_count
        work = jnp.full((count,), coordinate_work, dtype=jnp.int32)
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


class _HarmonicMeanElectronProposal(AbstractProposal):
    nuclei: AtomicStructure
    electron_count: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        nuclei: AtomicStructure,
        electron_count: int,
        step_size: float,
        /,
    ):
        self.nuclei = nuclei
        self.electron_count = int(electron_count)
        self.step_size = float(step_size)
        self.proposal_id = canonical_fingerprint(
            {
                "kind": "harmonic-mean-electron-proposal",
                "structure": nuclei.structure_id,
                "electron_count": self.electron_count,
                "step_size": self.step_size,
            }
        )

    def _standard_deviation(self, current: Array, /) -> tuple[Array, Array]:
        coordinate = jnp.asarray(current)
        active = self.nuclei.active_mask
        positions = jnp.where(
            active[:, None], self.nuclei.positions.astype(coordinate.dtype), 0.0
        )
        included = jnp.broadcast_to(
            active[None, :], (self.electron_count, int(active.shape[0]))
        )
        squared_distance = jnp.sum(
            (coordinate[:, None, :] - positions[None, :, :]) ** 2, axis=-1
        )
        distance = jnp.sqrt(jnp.where(included, squared_distance, 1.0))
        singular = jnp.any(included & (distance == 0.0), axis=-1)
        safe_distance = jnp.where(included & (distance != 0.0), distance, 1.0)
        inverse_sum = jnp.sum(jnp.where(included, 1.0 / safe_distance, 0.0), axis=-1)
        nucleus_count = jnp.sum(active, dtype=coordinate.dtype)
        harmonic_mean = nucleus_count / inverse_sum
        standard_deviation = self.step_size * harmonic_mean
        return standard_deviation, ~singular & jnp.isfinite(standard_deviation)

    def sample(self, key: Key[Array, ""], current: Array, /) -> Array:
        coordinate = jnp.asarray(current)
        if coordinate.shape != (self.electron_count, 3):
            raise ValueError(
                "Electron proposal positions must have shape "
                f"({self.electron_count}, 3)."
            )
        standard_deviation, valid = self._standard_deviation(coordinate)
        noise = jr.normal(key, coordinate.shape, dtype=coordinate.dtype)
        proposed = coordinate + standard_deviation[:, None] * noise
        return jnp.where(valid[:, None], proposed, jnp.nan)

    def log_prob(self, proposed: Array, current: Array, /) -> Array:
        proposed_coordinate = jnp.asarray(proposed)
        current_coordinate = jnp.asarray(current)
        if (
            proposed_coordinate.shape != (self.electron_count, 3)
            or current_coordinate.shape != (self.electron_count, 3)
        ):
            raise ValueError(
                "Electron proposal positions must have shape "
                f"({self.electron_count}, 3)."
            )
        standard_deviation, valid = self._standard_deviation(current_coordinate)
        safe_standard_deviation = jnp.where(valid, standard_deviation, 1.0)
        standardized = (
            proposed_coordinate - current_coordinate
        ) / safe_standard_deviation[:, None]
        log_probability = -0.5 * (
            jnp.sum(standardized**2)
            + 3.0
            * self.electron_count
            * jnp.log(jnp.asarray(2.0 * jnp.pi))
            + 6.0 * jnp.sum(jnp.log(safe_standard_deviation))
        )
        return jnp.where(jnp.all(valid), log_probability, -jnp.inf)


def harmonic_mean_electron_proposal(
    nuclei: AtomicStructure,
    electron_count: int,
    /,
    *,
    step_size: float = 0.2,
) -> AbstractProposal:
    """Build a state-dependent Gaussian proposal with exact MH correction.

    The standard deviation of each electron is ``step_size`` times the harmonic
    mean of its active electron–nucleus distances. The existing Metropolis-
    Hastings kernel evaluates both proposal directions through ``log_prob``.
    """
    if not isinstance(nuclei, AtomicStructure):
        raise TypeError("nuclei must be an AtomicStructure.")
    if nuclei.has_periodic_metadata:
        raise ValueError("Electronic proposals support finite nonperiodic nuclei only.")
    count = int(electron_count)
    step = float(step_size)
    if count <= 0:
        raise ValueError("electron_count must be positive.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    return _HarmonicMeanElectronProposal(nuclei, count, step)


def electronic_initial_walkers(
    key: Key[Array, ""],
    nuclei: AtomicStructure,
    electron_count: int,
    walker_count: int,
    /,
    *,
    spread: float = 1.0,
) -> Array:
    """Draw finite molecular electron walkers around charge-weighted nuclei.

    ``spread`` is measured in reference length units. Electron centers are
    assigned deterministically across nuclear charges; only displacements are
    random, making replay depend solely on ``key``.
    """
    if not isinstance(nuclei, AtomicStructure):
        raise TypeError("nuclei must be an AtomicStructure.")
    if nuclei.has_periodic_metadata:
        raise ValueError("Electronic walkers support finite nonperiodic nuclei only.")
    electrons = int(electron_count)
    walkers = int(walker_count)
    spread_value = float(spread)
    if electrons <= 0 or walkers <= 0:
        raise ValueError("electron_count and walker_count must be positive.")
    if not np.isfinite(spread_value) or spread_value <= 0.0:
        raise ValueError("spread must be finite and positive.")
    active = np.asarray(nuclei.active_mask, dtype=bool)
    active_indices = np.flatnonzero(active)
    charges = np.asarray(nuclei.atomic_numbers, dtype=np.int32)[active_indices]
    charge_centers = np.repeat(active_indices, charges)
    assignments = charge_centers[np.arange(electrons) % charge_centers.size]
    centers = nuclei.positions[jnp.asarray(assignments, dtype=jnp.int32)]
    assigned_charges = nuclei.atomic_numbers[jnp.asarray(assignments, dtype=jnp.int32)]
    dtype = nuclei.positions.dtype
    physical_spread = (
        jnp.asarray(spread_value, dtype=dtype)
        / jnp.asarray(nuclei.scale.length_to_reference, dtype=dtype)
        / jnp.sqrt(assigned_charges.astype(dtype))
    )
    noise = jr.normal(key, (walkers, electrons, 3), dtype=dtype)
    return centers[None, :, :] + physical_spread[None, :, None] * noise


__all__ = [
    "ElectronicCoulombHamiltonian",
    "ElectronicKineticPolicy",
    "ElectronicTraceMethod",
    "electronic_initial_walkers",
    "harmonic_mean_electron_proposal",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resource-admitted finite electronic VMC and finite-basis Hamiltonians."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ._amplitude import LogAmplitude


_TRACE_ADDRESS = SampleAddress(
    "quantum", "electronic-kinetic", target="trace-probe", role="hutchinson"
)


class ElectronicVMCResourcePlan(StrictModule):
    """Caller-visible finite admission replacing a global electron ceiling."""

    electron_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    coordinate_dimension: int = eqx.field(static=True)
    pair_stream_elements: int = eqx.field(static=True)
    determinant_work: int = eqx.field(static=True)
    admitted_pair_elements: int = eqx.field(static=True)
    admitted_determinant_work: int = eqx.field(static=True)
    valid: Array
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        electron_count: int,
        /,
        *,
        determinant_count: int = 1,
        spatial_dimension: int = 3,
        maximum_pair_elements: int = 1_000_000,
        maximum_determinant_work: int = 100_000_000,
    ):
        electrons, determinants, dimension = (
            int(electron_count),
            int(determinant_count),
            int(spatial_dimension),
        )
        pair_limit, work_limit = int(maximum_pair_elements), int(maximum_determinant_work)
        if min(electrons, determinants, dimension, pair_limit, work_limit) <= 0:
            raise ValueError("Electronic resource counts and limits must be positive.")
        pair_elements = electrons * electrons * dimension
        work = determinants * electrons**3
        if pair_elements > pair_limit or work > work_limit:
            raise ValueError(
                "Electronic resource plan exceeds caller limits: "
                f"pair_elements={pair_elements}/{pair_limit}, determinant_work={work}/{work_limit}."
            )
        self.electron_count = electrons
        self.determinant_count = determinants
        self.coordinate_dimension = electrons * dimension
        self.pair_stream_elements = pair_elements
        self.determinant_work = work
        self.admitted_pair_elements = pair_limit
        self.admitted_determinant_work = work_limit
        self.valid = jnp.asarray(True)
        self.claim = "finite-resource-admission-not-unrestricted-scaling"


class StochasticKineticEstimate(StrictModule):
    value: Array
    estimator_variance: Array
    estimator_count: Array
    exhausted: Array
    valid: Array
    method: str = eqx.field(static=True)


class StochasticElectronicKineticPolicy(StrictModule):
    """Fixed-capacity Hutchinson Hessian trace with replayable semantic probes."""

    maximum_probes: int = eqx.field(static=True)
    minimum_probes: int = eqx.field(static=True)
    standard_error_tolerance: float = eqx.field(static=True)
    method: Literal["hutchinson", "orthogonal-hutchinson"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_probes: int,
        minimum_probes: int = 1,
        standard_error_tolerance: float = 0.0,
        method: Literal["hutchinson", "orthogonal-hutchinson"] = "hutchinson",
    ):
        maximum, minimum = int(maximum_probes), int(minimum_probes)
        tolerance = float(standard_error_tolerance)
        if maximum <= 0 or minimum <= 0 or minimum > maximum or tolerance < 0.0:
            raise ValueError("Stochastic trace probe capacities/tolerance are invalid.")
        if method not in ("hutchinson", "orthogonal-hutchinson"):
            raise ValueError("Unknown stochastic electronic trace method.")
        self.maximum_probes = maximum
        self.minimum_probes = minimum
        self.standard_error_tolerance = tolerance
        self.method = method

    def local_kinetic(
        self,
        model: Callable[[Array], LogAmplitude],
        configuration: ArrayLike,
        /,
        *,
        key: Key[Array, ""],
    ) -> StochasticKineticEstimate:
        coordinates = jnp.asarray(configuration)
        shape = coordinates.shape
        flat = coordinates.reshape(-1)

        def components(value):
            amplitude = model(value.reshape(shape))
            if not isinstance(amplitude, LogAmplitude):
                raise TypeError("Electronic amplitude model must return LogAmplitude.")
            return jnp.stack((amplitude.log_abs, jnp.angle(amplitude.phase)))

        jacobian = jax.jacrev(components)
        gradient_components = jacobian(flat)
        probes = []
        for probe in range(self.maximum_probes):
            probe_key = derive_key(key, _TRACE_ADDRESS, 0, probe)
            direction = (
                2.0 * jr.bernoulli(probe_key, 0.5, flat.shape).astype(flat.dtype) - 1.0
            )
            if self.method == "orthogonal-hutchinson":
                hadamard_index = jnp.arange(flat.size, dtype=jnp.uint32)
                direction = direction * jnp.where(
                    jax.lax.population_count(
                        hadamard_index & jnp.asarray(probe, dtype=jnp.uint32)
                    )
                    % 2
                    == 0,
                    1.0,
                    -1.0,
                )
            _, directional_jacobian = jax.jvp(jacobian, (flat,), (direction,))
            probes.append(contract("ad,d->a", directional_jacobian, direction))
        samples = jnp.stack(probes)
        complex_samples = samples[:, 0] + 1j * samples[:, 1]
        cumulative_mean = jnp.cumsum(complex_samples) / jnp.arange(
            1, self.maximum_probes + 1
        )
        centered = complex_samples[:, None] - cumulative_mean[None, :]
        # The final fixed population remains the reported unbiased estimate.
        trace = cumulative_mean[-1]
        variance = jnp.sum(jnp.abs(complex_samples - trace) ** 2) / max(
            self.maximum_probes - 1, 1
        )
        standard_error = jnp.sqrt(variance / self.maximum_probes)
        complex_gradient = gradient_components[0] + 1j * gradient_components[1]
        laplacian_ratio = trace + contract("d,d->", complex_gradient, complex_gradient)
        exhausted = (self.standard_error_tolerance > 0.0) & (
            standard_error > self.standard_error_tolerance
        )
        return StochasticKineticEstimate(
            value=-0.5 * laplacian_ratio,
            estimator_variance=0.25 * variance,
            estimator_count=jnp.asarray(self.maximum_probes, dtype=jnp.int32),
            exhausted=jnp.asarray(exhausted),
            valid=jnp.isfinite(laplacian_ratio) & jnp.isfinite(variance),
            method=self.method,
        )


class ElectronicConnections(StrictModule):
    configurations: Array
    matrix_elements: Array
    active: Array
    capacity: int = eqx.field(static=True)


class ElectronicIntegralHamiltonian(StrictModule):
    """Finite spin-orbital integral Hamiltonian with explicit no-pair metadata."""

    one_body: Array
    two_body: Array
    representation: Literal["spin-free", "two-component", "four-component-no-pair"] = (
        eqx.field(static=True)
    )
    projector_id: str | None = eqx.field(static=True)
    orbital_count: int = eqx.field(static=True)
    hermiticity_residual: Array
    antisymmetry_residual: Array
    valid: Array
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        one_body: ArrayLike,
        two_body: ArrayLike,
        /,
        *,
        representation: Literal[
            "spin-free", "two-component", "four-component-no-pair"
        ] = "spin-free",
        projector_id: str | None = None,
        tolerance: float = 1e-8,
    ):
        one, two = jnp.asarray(one_body), jnp.asarray(two_body)
        if one.ndim != 2 or one.shape[0] != one.shape[1] or two.shape != one.shape * 2:
            raise ValueError("one_body/two_body require shapes (n,n) and (n,n,n,n).")
        if representation not in ("spin-free", "two-component", "four-component-no-pair"):
            raise ValueError("Unknown finite electronic integral representation.")
        if representation == "four-component-no-pair" and (
            projector_id is None or not str(projector_id)
        ):
            raise ValueError(
                "four-component-no-pair requires an explicit positive-energy projector_id."
            )
        hermiticity = jnp.maximum(
            jnp.max(jnp.abs(one - jnp.conj(one.T))),
            jnp.max(jnp.abs(two - jnp.conj(jnp.transpose(two, axes=(2, 3, 0, 1))))),
        )
        antisymmetry = jnp.maximum(
            jnp.max(jnp.abs(two + jnp.swapaxes(two, 2, 3))),
            jnp.max(jnp.abs(two + jnp.swapaxes(two, 0, 1))),
        )
        self.one_body = one
        self.two_body = two
        self.representation = representation
        self.projector_id = None if projector_id is None else str(projector_id)
        self.orbital_count = int(one.shape[0])
        self.hermiticity_residual = hermiticity
        self.antisymmetry_residual = antisymmetry
        self.valid = (
            jnp.all(jnp.isfinite(one))
            & jnp.all(jnp.isfinite(two))
            & (hermiticity <= tolerance)
            & (antisymmetry <= tolerance)
        )
        self.claim = (
            "finite-basis-no-pair-only"
            if representation == "four-component-no-pair"
            else "finite-basis-electronic-integrals"
        )

    def connected(self, occupation: ArrayLike, /) -> ElectronicConnections:
        occupied_mask = np.asarray(occupation, dtype=bool)
        if occupied_mask.shape != (self.orbital_count,):
            raise ValueError("occupation must select the finite spin-orbital basis.")
        occupied = np.flatnonzero(occupied_mask).tolist()
        virtual = np.flatnonzero(~occupied_mask).tolist()
        configurations = [occupied_mask.copy()]
        elements = [self.diagonal(occupied_mask)]
        for source in occupied:
            for target in virtual:
                candidate = occupied_mask.copy()
                candidate[source], candidate[target] = False, True
                configurations.append(candidate)
                phase = (-1) ** int(
                    np.sum(occupied_mask[min(source, target) + 1 : max(source, target)])
                )
                element = self.one_body[target, source]
                for other in occupied:
                    if other != source:
                        element = element + self.two_body[target, other, source, other]
                elements.append(phase * element)
        for first, second in itertools.combinations(occupied, 2):
            for target_first, target_second in itertools.combinations(virtual, 2):
                candidate = occupied_mask.copy()
                candidate[first] = candidate[second] = False
                candidate[target_first] = candidate[target_second] = True
                configurations.append(candidate)
                phase = 1
                intermediate = occupied_mask.copy()
                for orbital, occupied_after in (
                    (first, False),
                    (second, False),
                    (target_second, True),
                    (target_first, True),
                ):
                    phase *= (-1) ** int(np.sum(intermediate[:orbital]))
                    intermediate[orbital] = occupied_after
                elements.append(
                    phase * self.two_body[target_first, target_second, first, second]
                )
        capacity = len(configurations)
        return ElectronicConnections(
            configurations=jnp.asarray(np.stack(configurations)),
            matrix_elements=jnp.asarray(elements),
            active=jnp.ones((capacity,), dtype=bool),
            capacity=capacity,
        )

    def diagonal(self, occupation: ArrayLike, /) -> Array:
        mask = jnp.asarray(occupation, dtype=bool)
        weights = mask.astype(self.one_body.real.dtype)
        return contract("p,p->", weights, jnp.diag(self.one_body)) + 0.5 * contract(
            "p,q,pqpq->", weights, weights, self.two_body
        )


class PeriodicElectronicEvidence(StrictModule):
    real_cutoff: Array
    reciprocal_cutoff: Array
    neutral: Array
    background_used: Array
    valid: Array
    claim: str = eqx.field(static=True)


def periodic_coulomb_energy(
    fractional_positions: ArrayLike,
    charges: ArrayLike,
    cell: ArrayLike,
    /,
    *,
    real_image_radius: int,
    reciprocal_radius: int,
    screening: float,
    uniform_background: bool = False,
) -> tuple[Array, PeriodicElectronicEvidence]:
    """Finite Ewald sum with declared real/reciprocal resolution and neutrality."""
    positions, charge, lattice = map(jnp.asarray, (fractional_positions, charges, cell))
    if (
        positions.ndim != 2
        or charge.shape != (positions.shape[0],)
        or lattice.shape != (positions.shape[1], positions.shape[1])
    ):
        raise ValueError("periodic positions/charges/cell shapes are inconsistent.")
    dimension = int(positions.shape[1])
    if dimension != 3:
        raise ValueError("The finite Ewald electronic route currently supports 3D cells.")
    real_radius, reciprocal = int(real_image_radius), int(reciprocal_radius)
    alpha = float(screening)
    if real_radius < 0 or reciprocal < 0 or alpha <= 0.0:
        raise ValueError("Ewald radii/screening are invalid.")
    fractional_displacement = positions[:, None, :] - positions[None, :, :]
    integer_displacement = jnp.all(
        fractional_displacement == jnp.rint(fractional_displacement), axis=-1
    )
    distinct_particles = ~jnp.eye(positions.shape[0], dtype=bool)
    periodic_coincidence = jnp.any(integer_displacement & distinct_particles)
    positions = jnp.mod(positions, jnp.asarray(1, dtype=positions.dtype))
    net_charge = jnp.sum(charge)
    neutral = jnp.isclose(net_charge, 0.0)
    if not uniform_background and not bool(np.asarray(neutral)):
        raise ValueError(
            "Periodic Coulomb energy requires neutrality or declared uniform background."
        )
    shifts = jnp.asarray(
        list(itertools.product(range(-real_radius, real_radius + 1), repeat=3))
    )
    cartesian = positions @ lattice
    displacement = (
        cartesian[:, None, None, :]
        - cartesian[None, :, None, :]
        + shifts[None, None, :, :] @ lattice
    )
    squared = jnp.sum(displacement * displacement, axis=-1)
    particle_identity = jnp.eye(positions.shape[0], dtype=bool)[:, :, None]
    zero_image = jnp.all(shifts == 0, axis=-1)[None, None, :]
    self_zero = particle_identity & zero_image
    coincident = (squared == 0.0) & ~self_zero
    distance = jnp.sqrt(jnp.where(squared == 0.0, 1.0, squared))
    real_energy = 0.5 * jnp.sum(
        charge[:, None, None]
        * charge[None, :, None]
        * jnp.where(self_zero, 0.0, jax.scipy.special.erfc(alpha * distance) / distance)
    )
    real_energy = jnp.where(
        jnp.any(coincident) | periodic_coincidence, jnp.inf, real_energy
    )
    cell_factor = factorize(
        DenseLinearOperator(lattice),
        FactorizationPolicy("lu"),
    )
    volume = jnp.exp(cell_factor.log_abs_determinant())
    reciprocal_lattice = (
        2.0
        * jnp.pi
        * jnp.asarray(cell_factor.solve(jnp.eye(dimension, dtype=lattice.dtype)).value).T
    )
    modes = jnp.asarray(
        list(itertools.product(range(-reciprocal, reciprocal + 1), repeat=3))
    )
    nonzero = jnp.any(modes != 0, axis=-1)
    wavevectors = modes @ reciprocal_lattice
    k2 = jnp.sum(wavevectors * wavevectors, axis=-1)
    phase = cartesian @ wavevectors.T
    structure = contract("n,nk->k", charge, jnp.exp(1j * phase))
    reciprocal_energy = (2.0 * jnp.pi / volume) * jnp.sum(
        jnp.where(
            nonzero,
            jnp.exp(-k2 / (4.0 * alpha**2))
            * jnp.abs(structure) ** 2
            / jnp.where(nonzero, k2, 1.0),
            0.0,
        )
    )
    self_energy = -alpha / jnp.sqrt(jnp.pi) * jnp.sum(charge**2)
    background = (
        -jnp.pi * net_charge**2 / (2.0 * alpha**2 * volume) if uniform_background else 0.0
    )
    energy = real_energy + reciprocal_energy + self_energy + background
    evidence = PeriodicElectronicEvidence(
        real_cutoff=jnp.asarray(real_radius),
        reciprocal_cutoff=jnp.asarray(reciprocal),
        neutral=neutral,
        background_used=jnp.asarray(uniform_background),
        valid=jnp.isfinite(energy) & (neutral | uniform_background),
        claim="finite-ewald-resolution-no-continuum-exactness-claim",
    )
    return energy, evidence


__all__ = [
    "ElectronicConnections",
    "ElectronicIntegralHamiltonian",
    "ElectronicVMCResourcePlan",
    "PeriodicElectronicEvidence",
    "StochasticElectronicKineticPolicy",
    "StochasticKineticEstimate",
    "periodic_coulomb_energy",
]

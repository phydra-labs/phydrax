#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Causal bath fitting and bounded all-sector exact impurity solving."""

from __future__ import annotations

import abc
from math import isfinite
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.dlr import matsubara_frequencies
from ..linalg import HermitianSpectrum
from ..operators.quantum._fermionic_fock import FermionModeOrder
from ..operators.quantum._impurity import (
    anderson_bath_to_matsubara,
    AndersonBath,
    ImpurityEnvironment,
    MatsubaraHybridization,
)
from ..operators.quantum._thermal_green import (
    evaluate_fermionic_thermal_channel,
    extract_self_energy,
    fermionic_thermal_sector_channel,
    FermionicThermalSectorChannel,
    GreenFunctionMoments,
    MatsubaraGreenFunction,
    MatsubaraSelfEnergy,
    SelfEnergyMoments,
)
from ..operators.quantum.lattice._sector import (
    FixedCardinalityFermionBasis,
    SectorBasisResourcePolicy,
)


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


class AndersonBathFitPlan(StrictModule, NonTrainableState):
    """Fixed causal NNLS profile on a declared bath-energy support."""

    site_count: int = eqx.field(static=True)
    lower_energy: float = eqx.field(static=True)
    upper_energy: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    moment_tolerance: float = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_count: int,
        lower_energy: float,
        upper_energy: float,
        /,
        *,
        maximum_iterations: int = 2000,
        residual_tolerance: float = 2e-2,
        moment_tolerance: float = 2e-2,
        maximum_bytes: int = 64 * 1024**2,
    ):
        sites = _positive_int(site_count, "site_count")
        iterations = _positive_int(maximum_iterations, "maximum_iterations")
        lower = float(lower_energy)
        upper = float(upper_energy)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("Bath energy bounds must be finite and strictly ordered.")
        residual = _positive(residual_tolerance, "residual_tolerance")
        moment = _positive(moment_tolerance, "moment_tolerance")
        budget = _positive_int(maximum_bytes, "maximum_bytes")
        self.site_count = sites
        self.lower_energy = lower
        self.upper_energy = upper
        self.maximum_iterations = iterations
        self.residual_tolerance = residual
        self.moment_tolerance = moment
        self.maximum_bytes = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "causal-anderson-bath-fit",
                "sites": sites,
                "support": (lower, upper),
                "iterations": iterations,
                "residual_tolerance": residual,
                "moment_tolerance": moment,
                "maximum_bytes": budget,
            }
        )


class AndersonBathFitEvidence(StrictModule):
    """Fit, finite-bath, causality, and moment errors without conflation."""

    fit_residual: Array
    relative_fit_residual: Array
    finite_bath_error: Array
    causality_residual: Array
    moment_residual: Array
    projected_gradient_norm: Array
    finite: Array
    converged: Array
    valid: Array
    iteration_count: Array


class AndersonBathFitResult(StrictModule):
    bath: AndersonBath
    fitted: MatsubaraHybridization
    evidence: AndersonBathFitEvidence
    fit_id: str = eqx.field(static=True)


def fit_causal_anderson_bath(
    plan: AndersonBathFitPlan,
    target: MatsubaraHybridization,
    /,
) -> AndersonBathFitResult:
    """Fit non-negative bath strengths; causality is structural, never repaired."""

    if not isinstance(plan, AndersonBathFitPlan):
        raise TypeError("plan must be AndersonBathFitPlan.")
    if not isinstance(target, MatsubaraHybridization):
        raise TypeError("target must be MatsubaraHybridization.")
    samples = target.indices.shape[0]
    required = np.dtype(np.complex128).itemsize * (
        2 * samples * plan.site_count + 6 * plan.site_count + 4 * samples
    )
    if required > plan.maximum_bytes:
        raise ValueError("Bath fit exceeds maximum_bytes before allocation.")
    energies = (
        jnp.asarray([(plan.lower_energy + plan.upper_energy) / 2.0])
        if plan.site_count == 1
        else jnp.linspace(plan.lower_energy, plan.upper_energy, plan.site_count)
    )
    kernel = jnp.reciprocal(1j * target.frequencies[:, None] - energies[None, :])
    mask = target.sample_active.astype(kernel.real.dtype)
    design = jnp.concatenate((kernel.real * mask[:, None], kernel.imag * mask[:, None]))
    observations = jnp.concatenate((target.values.real * mask, target.values.imag * mask))
    lipschitz = jnp.sum(design**2)
    step = jnp.reciprocal(jnp.maximum(lipschitz, jnp.finfo(design.dtype).tiny))
    if target.moments is None:
        initial_total = jnp.maximum(
            jnp.max(jnp.abs(target.frequencies * target.values)), 1e-8
        )
    else:
        initial_total = jnp.maximum(target.moments.zeroth, 1e-8)
    strengths = jnp.full((plan.site_count,), initial_total / plan.site_count)

    def iteration(_: int, value: Array) -> Array:
        gradient = design.T @ (design @ value - observations)
        return jnp.maximum(value - step * gradient, 0.0)

    strengths = jax.lax.fori_loop(0, plan.maximum_iterations, iteration, strengths)
    residual_vector = design @ strengths - observations
    residual = jnp.sqrt(jnp.sum(residual_vector**2))
    target_norm = jnp.sqrt(jnp.sum(observations**2))
    relative = residual / jnp.maximum(target_norm, jnp.finfo(target_norm.dtype).tiny)
    gradient = design.T @ residual_vector
    projected = jnp.where((strengths > 0.0) | (gradient < 0.0), gradient, 0.0)
    projected_norm = jnp.max(jnp.abs(projected), initial=0.0)
    bath = AndersonBath(
        energies,
        jnp.sqrt(strengths),
        frequency_unit=target.frequency_unit,
    )
    fitted = anderson_bath_to_matsubara(
        bath,
        target.beta,
        target.indices,
        sample_active=target.sample_active,
    )
    difference = jnp.where(target.sample_active, fitted.values - target.values, 0.0)
    finite_bath_error = jnp.max(jnp.abs(difference), initial=0.0)
    moment_residual = (
        jnp.asarray(0.0, dtype=relative.dtype)
        if target.moments is None
        else jnp.maximum(
            jnp.abs(bath.moments.zeroth - target.moments.zeroth),
            jnp.abs(bath.moments.first - target.moments.first),
        )
    )
    finite = (
        target.evidence.finite
        & jnp.isfinite(relative)
        & jnp.isfinite(finite_bath_error)
        & jnp.isfinite(moment_residual)
        & jnp.isfinite(projected_norm)
    )
    converged = relative <= plan.residual_tolerance
    valid = (
        finite
        & target.evidence.causal
        & fitted.evidence.causal
        & converged
        & (moment_residual <= plan.moment_tolerance)
    )
    evidence = AndersonBathFitEvidence(
        residual,
        relative,
        finite_bath_error,
        fitted.evidence.causality_residual,
        moment_residual,
        projected_norm,
        finite,
        converged,
        valid,
        jnp.asarray(plan.maximum_iterations, dtype=jnp.int32),
    )
    fit_id = canonical_fingerprint(
        {
            "kind": "causal-anderson-bath-fit-result",
            "plan": plan.plan_id,
            "target": target.environment_id,
            "bath": bath.bath_id,
        }
    )
    return AndersonBathFitResult(bath, fitted, evidence, fit_id)


class EDImpurityPolicy(StrictModule, NonTrainableState):
    """Conservative Hilbert-space and numerical limits for all-sector ED."""

    maximum_modes: int = eqx.field(static=True)
    maximum_sector_dimension: int = eqx.field(static=True)
    maximum_total_states: int = eqx.field(static=True)
    maximum_transitions: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    dyson_tolerance: float = eqx.field(static=True)
    moment_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_modes: int = 12,
        maximum_sector_dimension: int = 1024,
        maximum_total_states: int = 4096,
        maximum_transitions: int = 1 << 20,
        maximum_bytes: int = 512 * 1024**2,
        hermiticity_tolerance: float = 1e-10,
        dyson_tolerance: float = 1e-8,
        moment_tolerance: float = 2e-1,
    ):
        self.maximum_modes = _positive_int(maximum_modes, "maximum_modes")
        self.maximum_sector_dimension = _positive_int(
            maximum_sector_dimension, "maximum_sector_dimension"
        )
        self.maximum_total_states = _positive_int(
            maximum_total_states, "maximum_total_states"
        )
        self.maximum_transitions = _positive_int(
            maximum_transitions, "maximum_transitions"
        )
        self.maximum_bytes = _positive_int(maximum_bytes, "maximum_bytes")
        self.hermiticity_tolerance = _positive(
            hermiticity_tolerance, "hermiticity_tolerance"
        )
        self.dyson_tolerance = _positive(dyson_tolerance, "dyson_tolerance")
        self.moment_tolerance = _positive(moment_tolerance, "moment_tolerance")


class ImpuritySolveRequest(StrictModule, NonTrainableState):
    """Normalized normal-state single-orbital impurity request."""

    indices: Array
    environment: ImpurityEnvironment
    onsite_energy: float = eqx.field(static=True)
    interaction: float = eqx.field(static=True)
    chemical_potential: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        onsite_energy: float,
        interaction: float,
        chemical_potential: float,
        beta: float,
        indices: ArrayLike,
        environment: ImpurityEnvironment,
        /,
    ):
        onsite = float(onsite_energy)
        interaction_ = float(interaction)
        chemical = float(chemical_potential)
        beta_ = _positive(beta, "beta")
        if not all(isfinite(value) for value in (onsite, interaction_, chemical)):
            raise ValueError("Impurity energies must be finite.")
        if interaction_ < 0.0:
            raise ValueError("The normal ED profile requires non-negative interaction.")
        labels = jnp.asarray(indices)
        if (
            labels.ndim != 1
            or labels.size == 0
            or not jnp.issubdtype(labels.dtype, jnp.integer)
        ):
            raise TypeError("indices must be one nonempty rank-one integer array.")
        if not isinstance(environment, ImpurityEnvironment):
            raise TypeError("environment must be ImpurityEnvironment.")
        self.onsite_energy = onsite
        self.interaction = interaction_
        self.chemical_potential = chemical
        self.beta = beta_
        self.indices = labels.astype(jnp.int32)
        self.environment = environment
        self.request_id = canonical_fingerprint(
            {
                "kind": "single-orbital-impurity-request",
                "energies": (onsite, interaction_, chemical),
                "beta": beta_,
                "indices": array_tree_fingerprint(labels),
                "environment": environment.environment_id,
            }
        )


class ImpuritySolveEvidence(StrictModule):
    """All-sector, causality, moment, Dyson, and density errors kept separate."""

    log_partition_function: Array
    spectral_sum_residual: Array
    causality_residual: Array
    moment_residual: Array
    dyson_residual: Array
    density_residual: Array
    hamiltonian_residual: Array
    finite: Array
    valid: Array
    sector_count: int = eqx.field(static=True)
    total_state_count: int = eqx.field(static=True)


class ImpuritySolveResult(StrictModule):
    green: MatsubaraGreenFunction
    self_energy: MatsubaraSelfEnergy
    density: Array
    double_occupancy: Array
    evidence: ImpuritySolveEvidence
    request_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class AbstractImpurityProvider(StrictModule, NonTrainableState):
    """Host-selected provider returning the normalized impurity result contract."""

    provider_id: eqx.AbstractVar[str]
    differentiable: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def solve(self, request: ImpuritySolveRequest, /) -> ImpuritySolveResult:
        raise NotImplementedError


def _sector_states(basis: FixedCardinalityFermionBasis, /) -> tuple[int, ...]:
    return tuple(
        sum(
            int(occupation) << mode
            for mode, occupation in enumerate(np.asarray(basis.coordinate(index)))
        )
        for index in range(basis.dimension)
    )


def _ladder_sign(state: int, mode: int) -> int:
    return -1 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1


def _hop(state: int, destination: int, source: int) -> tuple[int, int] | None:
    if not (state & (1 << source)) or state & (1 << destination):
        return None
    sign = _ladder_sign(state, source)
    removed = state ^ (1 << source)
    sign *= _ladder_sign(removed, destination)
    return removed | (1 << destination), sign


def _sector_hamiltonian(
    states: tuple[int, ...], request: ImpuritySolveRequest, bath: AndersonBath
) -> Array:
    index = {state: position for position, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=np.complex128)
    impurity_level = request.onsite_energy - request.chemical_potential
    for column, state in enumerate(states):
        n_up = (state >> 0) & 1
        n_down = (state >> 1) & 1
        diagonal = impurity_level * (n_up + n_down)
        diagonal += request.interaction * n_up * n_down
        for bath_site, energy in enumerate(np.asarray(bath.site_energies)):
            diagonal += float(energy) * (
                ((state >> (2 + 2 * bath_site)) & 1)
                + ((state >> (3 + 2 * bath_site)) & 1)
            )
        matrix[column, column] = diagonal
        for bath_site, coupling in enumerate(np.asarray(bath.couplings)):
            for spin in range(2):
                impurity_mode = spin
                bath_mode = 2 + 2 * bath_site + spin
                forward = _hop(state, impurity_mode, bath_mode)
                if forward is not None:
                    destination, sign = forward
                    matrix[index[destination], column] += coupling * sign
                reverse = _hop(state, bath_mode, impurity_mode)
                if reverse is not None:
                    destination, sign = reverse
                    matrix[index[destination], column] += np.conj(coupling) * sign
    return jnp.asarray(matrix)


def _annihilation_between(
    source_states: tuple[int, ...], target_states: tuple[int, ...], mode: int
) -> Array:
    target_index = {state: index for index, state in enumerate(target_states)}
    matrix = np.zeros((len(target_states), len(source_states)), dtype=np.complex128)
    for column, state in enumerate(source_states):
        if state & (1 << mode):
            destination = state ^ (1 << mode)
            matrix[target_index[destination], column] = _ladder_sign(state, mode)
    return jnp.asarray(matrix)


def solve_all_sector_ed_impurity(
    request: ImpuritySolveRequest,
    /,
    *,
    policy: EDImpurityPolicy | None = None,
    provider_id: str = "phydrax.all-sector-ed.single-orbital",
) -> ImpuritySolveResult:
    """Solve every particle-number sector and assemble the grand-canonical Green function."""

    if not isinstance(request, ImpuritySolveRequest):
        raise TypeError("request must be ImpuritySolveRequest.")
    policy_ = EDImpurityPolicy() if policy is None else policy
    if not isinstance(policy_, EDImpurityPolicy):
        raise TypeError("policy must be EDImpurityPolicy or None.")
    bath = request.environment.bath
    if bath is None:
        raise ValueError("All-sector ED requires an AndersonBath environment.")
    mode_count = 2 * (1 + bath.site_count)
    total_states = 1 << mode_count
    if mode_count > policy_.maximum_modes or total_states > policy_.maximum_total_states:
        raise ValueError("ED Hilbert space exceeds the declared mode/state capacity.")
    mode_order = FermionModeOrder(
        ("impurity-up", "impurity-down")
        + tuple(
            f"bath-{site}-{spin}"
            for site in range(bath.site_count)
            for spin in ("up", "down")
        )
    )
    basis_resources = SectorBasisResourcePolicy(
        maximum_dimension=policy_.maximum_sector_dimension,
        maximum_table_bytes=policy_.maximum_bytes,
    )
    sector_bases = tuple(
        FixedCardinalityFermionBasis(mode_order, particles, resources=basis_resources)
        for particles in range(mode_count + 1)
    )
    sector_states = tuple(_sector_states(basis) for basis in sector_bases)
    largest_sector = max(len(states) for states in sector_states)
    if largest_sector > policy_.maximum_sector_dimension:
        raise ValueError("ED sector dimension exceeds maximum_sector_dimension.")
    dense_elements = sum(len(states) ** 2 for states in sector_states)
    required = np.dtype(np.complex128).itemsize * (4 * dense_elements + total_states)
    if required > policy_.maximum_bytes:
        raise ValueError("ED solve exceeds maximum_bytes before Hamiltonian allocation.")
    spectra = tuple(
        HermitianSpectrum(
            _sector_hamiltonian(states, request, bath),
            tolerance=policy_.hermiticity_tolerance,
        )
        for states in sector_states
    )
    all_energies = jnp.concatenate(tuple(spectrum.eigenvalues for spectrum in spectra))
    ground = jnp.min(all_energies)
    partition_scaled = jnp.sum(jnp.exp(-request.beta * (all_energies - ground)))
    log_partition = jnp.log(partition_scaled) - request.beta * ground
    frequencies = matsubara_frequencies(
        request.indices, beta=request.beta, statistics="fermionic"
    )
    spin_greens = []
    spectral_sums = []
    channels: list[FermionicThermalSectorChannel] = []
    for spin in range(2):
        values = jnp.zeros(request.indices.shape, dtype=jnp.complex128)
        spin_sum = jnp.asarray(0.0)
        for particles in range(1, mode_count + 1):
            source_spectrum = spectra[particles]
            target_spectrum = spectra[particles - 1]
            operator_basis = _annihilation_between(
                sector_states[particles], sector_states[particles - 1], spin
            )
            operator_eigen = (
                jnp.conj(target_spectrum.eigenvectors.T)
                @ operator_basis
                @ source_spectrum.eigenvectors
            )
            channel = fermionic_thermal_sector_channel(
                source_spectrum.eigenvalues,
                target_spectrum.eigenvalues,
                operator_eigen,
                request.beta,
                log_partition,
                source_sector=f"N={particles}",
                target_sector=f"N={particles - 1}",
                maximum_transitions=policy_.maximum_transitions,
            )
            channels.append(channel)
            values = values + evaluate_fermionic_thermal_channel(
                channel, 1j * frequencies
            )
            spin_sum = spin_sum + channel.evidence.spectral_sum
        spin_greens.append(values)
        spectral_sums.append(spin_sum)
    green_values = 0.5 * (spin_greens[0] + spin_greens[1])
    green = MatsubaraGreenFunction(
        request.beta,
        request.indices,
        green_values,
        moments=GreenFunctionMoments(
            jnp.asarray([1.0]), jnp.asarray([True]), "fermionic", "inverse-frequency"
        ),
    )
    probabilities = tuple(
        jnp.exp(-request.beta * spectrum.eigenvalues - log_partition)
        for spectrum in spectra
    )
    spin_density = []
    double_occupancy = jnp.asarray(0.0)
    for spin in range(2):
        density = jnp.asarray(0.0)
        for states, spectrum, probability in zip(
            sector_states, spectra, probabilities, strict=True
        ):
            occupation = jnp.asarray([(state >> spin) & 1 for state in states])
            expectation = jnp.sum(
                jnp.abs(spectrum.eigenvectors) ** 2 * occupation[:, None], axis=0
            )
            density = density + jnp.sum(probability * expectation)
        spin_density.append(density)
    for states, spectrum, probability in zip(
        sector_states, spectra, probabilities, strict=True
    ):
        occupation = jnp.asarray(
            [((state >> 0) & 1) * ((state >> 1) & 1) for state in states]
        )
        expectation = jnp.sum(
            jnp.abs(spectrum.eigenvectors) ** 2 * occupation[:, None], axis=0
        )
        double_occupancy = double_occupancy + jnp.sum(probability * expectation)
    density = spin_density[0] + spin_density[1]
    density_residual = jnp.abs(spin_density[0] - spin_density[1])
    hybridization = anderson_bath_to_matsubara(bath, request.beta, request.indices)
    noninteracting_values = jnp.reciprocal(
        1j * frequencies
        + request.chemical_potential
        - request.onsite_energy
        - hybridization.values
    )
    noninteracting = MatsubaraGreenFunction(
        request.beta, request.indices, noninteracting_values
    )
    extracted = extract_self_energy(noninteracting, green)
    hartree = request.interaction * density / 2.0
    tail = request.interaction**2 * (density / 2.0) * (1.0 - density / 2.0)
    largest = int(jnp.argmax(jnp.abs(frequencies)))
    asymptotic = hartree + tail / (1j * frequencies[largest])
    moment_residual = jnp.abs(extracted.self_energy.values[largest] - asymptotic)
    self_energy = MatsubaraSelfEnergy(
        request.beta,
        request.indices,
        extracted.self_energy.values,
        moments=SelfEnergyMoments(hartree, jnp.asarray([tail])),
        moment_residual=moment_residual,
        moment_tolerance=policy_.moment_tolerance,
    )
    spectral_sum_residual = jnp.max(jnp.abs(jnp.stack(spectral_sums) - 1.0))
    dyson_residual = jnp.max(extracted.evidence.relative_residual)
    hamiltonian_residual = jnp.max(
        jnp.stack(tuple(spectrum.hermiticity_residual for spectrum in spectra))
    )
    finite = (
        jnp.isfinite(log_partition)
        & jnp.all(jnp.isfinite(green.values))
        & jnp.isfinite(density)
        & jnp.isfinite(double_occupancy)
        & jnp.isfinite(moment_residual)
    )
    valid = (
        finite
        & (spectral_sum_residual <= 1e-8)
        & self_energy.evidence.causal
        & (moment_residual <= policy_.moment_tolerance)
        & (dyson_residual <= policy_.dyson_tolerance)
        & (density_residual <= 1e-8)
        & (hamiltonian_residual <= policy_.hermiticity_tolerance)
    )
    evidence = ImpuritySolveEvidence(
        log_partition,
        spectral_sum_residual,
        self_energy.evidence.causality_residual,
        moment_residual,
        dyson_residual,
        density_residual,
        hamiltonian_residual,
        finite,
        valid,
        mode_count + 1,
        total_states,
    )
    result_id = canonical_fingerprint(
        {
            "kind": "normalized-all-sector-ed-impurity-result",
            "request": request.request_id,
            "provider": provider_id,
            "green": green.representation_id,
            "self_energy": self_energy.representation_id,
        }
    )
    return ImpuritySolveResult(
        green,
        self_energy,
        density,
        double_occupancy,
        evidence,
        request.request_id,
        provider_id,
        result_id,
    )


class ExactDiagonalizationImpurityProvider(AbstractImpurityProvider):
    """Native normalized provider for the bounded finite-bath ED profile."""

    policy: EDImpurityPolicy
    provider_id: str = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self,
        policy: EDImpurityPolicy | None = None,
        /,
        *,
        provider_id: str = "phydrax.all-sector-ed.single-orbital",
    ):
        policy_ = EDImpurityPolicy() if policy is None else policy
        if not isinstance(policy_, EDImpurityPolicy):
            raise TypeError("policy must be EDImpurityPolicy or None.")
        identifier = str(provider_id)
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        self.policy = policy_
        self.provider_id = identifier
        self.differentiable = False

    def solve(self, request: ImpuritySolveRequest, /) -> ImpuritySolveResult:
        return solve_all_sector_ed_impurity(
            request, policy=self.policy, provider_id=self.provider_id
        )


__all__ = [
    "AbstractImpurityProvider",
    "AndersonBathFitEvidence",
    "AndersonBathFitPlan",
    "AndersonBathFitResult",
    "EDImpurityPolicy",
    "ExactDiagonalizationImpurityProvider",
    "ImpuritySolveEvidence",
    "ImpuritySolveRequest",
    "ImpuritySolveResult",
    "fit_causal_anderson_bath",
    "solve_all_sector_ed_impurity",
]

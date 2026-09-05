#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite electronic-sector preparation and native open-quantum execution.

Preparation/evolution wrappers are host operations. Prepared native problem
operators, density reductions and observables retain their numeric JAX paths.
Quantum-jump trajectories inherit the native fixed-step approximation and are
not advertised as exact event-time trajectories or pathwise differentiable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....artifacts import ScientificArtifactEnvelope
from ....atomistic import AtomisticUnitSystem
from ....discretization import TemporalMesh
from ....qualification import ReferenceArtifactManifest
from ....series import SampledSeries, SeriesSupport
from ....solver import (
    FiniteCPTPIntegrationResult,
    FiniteLindbladChannelPlan,
    integrate_finite_cptp,
    LindbladProblem,
    LindbladSolution,
    QuantumJumpProblem,
    QuantumTrajectoryEnsemble,
    solve_lindblad,
    solve_quantum_jump_ensemble,
    StateVectorOperator,
)
from ....units import conversion_factor, UnitDefinition
from ._model import _admit, BasisKey, ElectronicParameterArtifact, ElectronicSiteGraph


def _bound(
    dimension: int,
    channels: int,
    maximum_dimension: int,
    maximum_liouville_elements: int,
    maximum_channels: int,
) -> None:
    for value in (maximum_dimension, maximum_liouville_elements, maximum_channels):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Electronic resource bounds must be positive integers.")
    if (
        dimension > maximum_dimension
        or dimension**4 > maximum_liouville_elements
        or max(channels, 1) > maximum_channels
    ):
        raise ValueError("Electronic model exceeds declared dense resource bounds.")


def _envelope(
    kind: str,
    digest: str,
    parents: tuple[str, ...],
    license_id: str,
    *,
    successful: bool = True,
) -> ScientificArtifactEnvelope:
    return ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=digest,
        producer="phydrax.nucleic_acid_biophysics.electronics",
        producer_version="native",
        build_id="declared-finite-electronic-model",
        license_id=license_id,
        resource_id=digest,
        status="complete" if successful else "failed",
        failure_reason="none" if successful else "native-electronic-execution-invalid",
        parent_artifact_ids=parents,
    )


@dataclass(frozen=True, slots=True)
class PreparedElectronicModel:
    """One-carrier or electron-major/hole-minor sector, optionally plus vacuum.

    hamiltonian is already H/hbar in units.time_unit^-1. jumps are dimensionless;
    rates follow the finite-channel ABI. The dense/jump ABI is available through
    collapse_operators, which multiplies by sqrt(rate) exactly once. All basis
    entries are material electronic states; only inactive jump slots are padding.
    """

    hamiltonian: Array
    jumps: Array
    rates: Array
    active_jumps: Array
    basis_keys: tuple[BasisKey, ...]
    channel_ids: tuple[str, ...]
    graphs: tuple[ElectronicSiteGraph, ...]
    parameters: tuple[ElectronicParameterArtifact, ...]
    units: AtomisticUnitSystem
    artifact: ScientificArtifactEnvelope
    include_vacuum: bool

    @property
    def dimension(self) -> int:
        return len(self.basis_keys)

    @property
    def rights(self) -> tuple[ReferenceArtifactManifest, ...]:
        return tuple(parameter.source for parameter in self.parameters) + tuple(
            manifest for graph in self.graphs for manifest in graph.structure_rights
        )

    def require_rights(self, requested_use: Mapping[str, bool]) -> None:
        """Re-admit all inherited restrictions before a new derivative/use/export."""
        _admit(self.rights, requested_use)

    @property
    def collapse_operators(self) -> Array:
        return (
            self.jumps
            * jnp.sqrt(jnp.where(self.active_jumps, self.rates, 0.0))[:, None, None]
        )

    def density_problem(self, initial_density: ArrayLike) -> LindbladProblem:
        """Host-validate a native dense problem without changing density/trace."""
        return LindbladProblem(
            self.hamiltonian,
            self.collapse_operators,
            initial_density,
            problem_id=self.artifact.artifact_id,
        )

    def finite_plan(
        self, slicing: TemporalMesh, *, time_unit: UnitDefinition, tolerance: float = 1e-8
    ) -> FiniteLindbladChannelPlan:
        """Prepare native finite channels; mesh values are converted once."""
        if not np.all(np.asarray(slicing.active_intervals)):
            raise ValueError(
                "Electronic evolution requires a fully active temporal partition."
            )
        factor = float(conversion_factor(time_unit, self.units.time_unit))
        mesh = TemporalMesh(
            np.asarray(slicing.nodes) * factor,
            role="internal",
            source_plan_id=slicing.mesh_id,
        )
        return FiniteLindbladChannelPlan(
            self.hamiltonian,
            self.jumps,
            self.rates,
            mesh,
            active_jumps=self.active_jumps,
            evaluation="left",
            tolerance=tolerance,
            plan_id=self.artifact.artifact_id,
        )

    def jump_problem(self, initial_state: ArrayLike) -> QuantumJumpProblem:
        """Host-prepare actual native quantum jumps, retaining channel identity."""
        state = np.asarray(initial_state, dtype=complex)
        if (
            state.shape != (self.dimension,)
            or not np.all(np.isfinite(state))
            or not np.isclose(np.sum(np.abs(state) ** 2), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError(
                "Electronic state vectors must already have finite unit norm."
            )
        hamiltonian = StateVectorOperator.from_matrix(
            self.hamiltonian, operator_id=f"{self.artifact.artifact_id}:H/hbar"
        )
        collapse = tuple(
            StateVectorOperator.from_matrix(
                operator, operator_id=f"{self.artifact.artifact_id}:{channel}"
            )
            for operator, channel in zip(
                self.collapse_operators, self.channel_ids, strict=True
            )
        )
        return QuantumJumpProblem(
            hamiltonian, collapse, state, problem_id=self.artifact.artifact_id
        )

    def basis_state(self, key: BasisKey) -> Array:
        """Create a normalized state by stable site key; () selects declared vacuum."""
        if key not in self.basis_keys:
            raise ValueError("Initial electronic basis key is outside prepared support.")
        return jax.nn.one_hot(self.basis_keys.index(key), self.dimension, dtype=complex)


def _parameter_arrays(
    parameters: ElectronicParameterArtifact,
    keys: tuple[BasisKey, ...],
    units: AtomisticUnitSystem,
    include_vacuum: bool,
):
    if set(parameters.basis_keys) != set(keys):
        raise ValueError(
            "Parameter basis must exactly cover the declared electronic sector."
        )
    size = len(keys) + int(include_vacuum)
    indices = {key: index for index, key in enumerate(keys)}
    energy_factor = (
        float(conversion_factor(parameters.energy_unit, units.scale.energy_unit))
        / units.reduced_planck_constant
    )
    hamiltonian = np.zeros((size, size), dtype=complex)
    for key, value in zip(parameters.basis_keys, parameters.site_energies, strict=True):
        hamiltonian[indices[key], indices[key]] = value * energy_factor
    for row, column, value in parameters.couplings:
        value = complex(value) * energy_factor
        hamiltonian[indices[row], indices[column]] = value
        hamiltonian[indices[column], indices[row]] = value.conjugate()
    jumps = np.zeros((len(parameters.channels), size, size), dtype=complex)
    rates = np.zeros(len(parameters.channels))
    for index, channel in enumerate(parameters.channels):
        source = indices[channel.source]
        if channel.kind == "recombination":
            if not include_vacuum:
                raise ValueError(
                    "Recombination requires an explicitly included vacuum state."
                )
            target = size - 1
        elif channel.kind == "dephasing":
            target = source
        else:
            target = indices[channel.target]
        jumps[index, target, source] = 1.0
        rates[index] = channel.rate * float(
            conversion_factor(channel.rate_unit, units.frequency_unit)
        )
    if not np.all(np.isfinite(hamiltonian)) or not np.all(np.isfinite(rates)):
        raise ValueError("Unit conversion produced nonfinite electronic coefficients.")
    return hamiltonian, jumps, rates


def _prepared(
    hamiltonian,
    jumps,
    rates,
    keys,
    channel_ids,
    graphs,
    parameters,
    units,
    include_vacuum,
    parents=(),
) -> PreparedElectronicModel:
    active = np.ones(len(rates), dtype=bool)
    if not len(rates):
        # The native finite-channel and trajectory ABIs have positive capacity.
        # This explicitly inactive slot contributes no physical channel.
        jumps = np.zeros((1, len(keys), len(keys)), dtype=complex)
        rates = np.zeros(1)
        active = np.zeros(1, dtype=bool)
        channel_ids = ("inactive-capacity",)
    if any(
        not np.all(np.isfinite(value)) for value in (hamiltonian, jumps, rates)
    ) or np.any(rates < 0):
        raise ValueError(
            "Compiled electronic coefficients must be finite with nonnegative rates."
        )
    digest = canonical_fingerprint(
        {
            "graphs": [graph.fingerprint() for graph in graphs],
            "parameters": [parameter.fingerprint() for parameter in parameters],
            "basis": keys,
            "units": units.unit_system_id,
            "numeric": array_tree_fingerprint((hamiltonian, jumps, rates, active)),
            "include_vacuum": include_vacuum,
        }
    )
    parents = (
        tuple(parents)
        + tuple(parameter.source.manifest_id for parameter in parameters)
        + tuple(
            envelope.artifact_id
            for graph in graphs
            for envelope in graph.structure_artifacts
        )
    )
    licenses = {parameter.source.license_id for parameter in parameters} | {
        manifest.license_id for graph in graphs for manifest in graph.structure_rights
    }
    artifact = _envelope(
        "prepared-electronic-model", digest, parents, " AND ".join(sorted(licenses))
    )
    return PreparedElectronicModel(
        jnp.asarray(hamiltonian),
        jnp.asarray(jumps),
        jnp.asarray(rates),
        jnp.asarray(active),
        keys,
        channel_ids,
        graphs,
        parameters,
        units,
        artifact,
        include_vacuum,
    )


def prepare_electronics(
    graph: ElectronicSiteGraph,
    parameters: ElectronicParameterArtifact,
    /,
    *,
    units: AtomisticUnitSystem,
    requested_use: Mapping[str, bool],
    include_vacuum: bool = False,
    maximum_dimension: int = 32,
    maximum_liouville_elements: int = 1_048_576,
    maximum_channels: int = 256,
) -> PreparedElectronicModel:
    """Compile one carrier, with no inferred edges, parameters, bath or sinks."""
    if not isinstance(include_vacuum, bool):
        raise TypeError("include_vacuum must be a boolean.")
    _admit((parameters.source,) + graph.structure_rights, requested_use)
    if parameters.structure_derived and not graph.structure_artifacts:
        raise ValueError(
            "Structure-derived parameters require retained structure artifacts and rights."
        )
    keys = tuple((site,) for site in graph.site_ids)
    _bound(
        len(keys) + int(include_vacuum),
        len(parameters.channels),
        maximum_dimension,
        maximum_liouville_elements,
        maximum_channels,
    )
    edges = {tuple(sorted(edge)) for edge in graph.edges}
    if any(
        tuple(sorted((row[0], column[0]))) not in edges
        for row, column, _ in parameters.couplings
    ):
        raise ValueError("Parameter coupling is outside the declared electronic graph.")
    hamiltonian, jumps, rates = _parameter_arrays(parameters, keys, units, include_vacuum)
    return _prepared(
        hamiltonian,
        jumps,
        rates,
        keys + (((),) if include_vacuum else ()),
        tuple(channel.channel_id for channel in parameters.channels),
        (graph,),
        (parameters,),
        units,
        include_vacuum,
    )


def prepare_electron_hole(
    electron: PreparedElectronicModel,
    hole: PreparedElectronicModel,
    interaction: ElectronicParameterArtifact,
    /,
    *,
    requested_use: Mapping[str, bool],
    include_vacuum: bool = False,
    maximum_dimension: int = 32,
    maximum_liouville_elements: int = 1_048_576,
    maximum_channels: int = 256,
) -> PreparedElectronicModel:
    """Compile H_e tensor I + I tensor H_h + declared interaction.

    Carrier channels lift as L_e tensor I and I tensor L_h. Interaction channels
    can dephase, transfer or recombine explicit electron/hole pairs into vacuum.
    No single-carrier sink is lifted into an undeclared partial-loss sector.
    """
    if not isinstance(include_vacuum, bool):
        raise TypeError("include_vacuum must be a boolean.")
    if (
        len(electron.graphs) != 1
        or len(hole.graphs) != 1
        or electron.include_vacuum
        or hole.include_vacuum
    ):
        raise ValueError(
            "Electron/hole factors must be one-carrier models without vacuum."
        )
    if (
        electron.graphs[0].construct.fingerprint()
        != hole.graphs[0].construct.fingerprint()
    ):
        raise ValueError("Electron and hole sites must bind the same nucleic construct.")
    if electron.units.constant_set_id != hole.units.constant_set_id:
        raise ValueError("Electron/hole models must use the same physical constant set.")
    electron.require_rights(requested_use)
    hole.require_rights(requested_use)
    _admit((interaction.source,), requested_use)
    if interaction.structure_derived and not (
        electron.graphs[0].structure_artifacts and hole.graphs[0].structure_artifacts
    ):
        raise ValueError(
            "Structure-derived interactions require both factor structure sources."
        )
    ne, nh = electron.dimension, hole.dimension
    keys = tuple((e[0], h[0]) for e in electron.basis_keys for h in hole.basis_keys)
    e_active = np.flatnonzero(np.asarray(electron.active_jumps))
    h_active = np.flatnonzero(np.asarray(hole.active_jumps))
    channel_count = len(e_active) + len(h_active) + len(interaction.channels)
    _bound(
        ne * nh + int(include_vacuum),
        channel_count,
        maximum_dimension,
        maximum_liouville_elements,
        maximum_channels,
    )
    hamiltonian, jumps, rates = _parameter_arrays(
        interaction, keys, electron.units, include_vacuum
    )
    hole_factor = float(
        conversion_factor(hole.units.frequency_unit, electron.units.frequency_unit)
    )
    hamiltonian[: ne * nh, : ne * nh] += np.kron(
        np.asarray(electron.hamiltonian), np.eye(nh)
    ) + np.kron(np.eye(ne), np.asarray(hole.hamiltonian) * hole_factor)
    lifted = np.zeros(
        (len(e_active) + len(h_active), len(hamiltonian), len(hamiltonian)), dtype=complex
    )
    lifted_rates = []
    channel_ids = []
    for index, channel in enumerate(e_active):
        lifted[index, : ne * nh, : ne * nh] = np.kron(
            np.asarray(electron.jumps[channel]), np.eye(nh)
        )
        lifted_rates.append(float(electron.rates[channel]))
        channel_ids.append(f"electron:{electron.channel_ids[channel]}")
    for index, channel in enumerate(h_active):
        lifted[len(e_active) + index, : ne * nh, : ne * nh] = np.kron(
            np.eye(ne), np.asarray(hole.jumps[channel])
        )
        lifted_rates.append(float(hole.rates[channel]) * hole_factor)
        channel_ids.append(f"hole:{hole.channel_ids[channel]}")
    channel_ids.extend(f"pair:{channel.channel_id}" for channel in interaction.channels)
    return _prepared(
        hamiltonian,
        np.concatenate((lifted, jumps)),
        np.concatenate((np.asarray(lifted_rates), rates)),
        keys + (((),) if include_vacuum else ()),
        tuple(channel_ids),
        electron.graphs + hole.graphs,
        electron.parameters + hole.parameters + (interaction,),
        electron.units,
        include_vacuum,
        (electron.artifact.artifact_id, hole.artifact.artifact_id),
    )


def electronic_reduced_density(
    model: PreparedElectronicModel, density: ArrayLike, /, *, carrier: int = 0
) -> Array:
    """Unnormalized surviving-carrier density in graph.site_ids order.

    Partial trace removes the other carrier, not loss probability. Vacuum is
    excluded; no renormalization hides recombination. Fixed-support numeric path.
    """
    rho = jnp.asarray(density)
    if rho.shape[-2:] != (model.dimension, model.dimension):
        raise ValueError("Density dimensions do not match the prepared electronic basis.")
    if carrier not in range(len(model.graphs)):
        raise ValueError("Requested carrier is outside the electronic sector.")
    size = model.dimension - int(model.include_vacuum)
    sector = rho[..., :size, :size]
    if len(model.graphs) == 1:
        return sector
    ne, nh = (len(graph.site_ids) for graph in model.graphs)
    tensor = sector.reshape(sector.shape[:-2] + (ne, nh, ne, nh))
    return (
        jnp.trace(tensor, axis1=-3, axis2=-1)
        if carrier == 0
        else jnp.trace(tensor, axis1=-4, axis2=-2)
    )


def electronic_populations(
    model: PreparedElectronicModel, density: ArrayLike, /, *, carrier: int = 0
) -> Array:
    """Electronic site populations, not atom charges or damage probabilities."""
    return jnp.real(
        jnp.diagonal(
            electronic_reduced_density(model, density, carrier=carrier),
            axis1=-2,
            axis2=-1,
        )
    )


def electronic_coherences(
    model: PreparedElectronicModel,
    density: ArrayLike,
    pairs: tuple[tuple[int, int], ...],
    /,
    *,
    carrier: int = 0,
) -> Array:
    """Complex coherences rho[row,column] in the declared orbital gauge."""
    reduced = electronic_reduced_density(model, density, carrier=carrier)
    sites = model.graphs[carrier].site_ids
    if any(left not in sites or right not in sites for left, right in pairs):
        raise ValueError("Coherence pair contains an unknown electronic site ID.")
    rows = jnp.asarray([sites.index(left) for left, _ in pairs], dtype=jnp.int32)
    columns = jnp.asarray([sites.index(right) for _, right in pairs], dtype=jnp.int32)
    return reduced[..., rows, columns]


def nucleotide_electronic_populations(
    model: PreparedElectronicModel, density: ArrayLike, /, *, carrier: int = 0
) -> Array:
    """Sum explicitly mapped orbitals in construct.nucleotide_keys order."""
    populations = electronic_populations(model, density, carrier=carrier)
    graph = model.graphs[carrier]
    binding = jnp.asarray(
        [
            [key == site_key for key in graph.construct.nucleotide_keys]
            for site_key in graph.nucleotide_keys
        ],
        dtype=populations.dtype,
    )
    return populations @ binding


@dataclass(frozen=True, slots=True)
class ElectronicEvolution:
    model: PreparedElectronicModel
    densities: SampledSeries
    native_result: LindbladSolution | FiniteCPTPIntegrationResult
    artifact: ScientificArtifactEnvelope


@dataclass(frozen=True, slots=True)
class ElectronicJumpEvolution:
    model: PreparedElectronicModel
    mean_densities: SampledSeries
    native_result: QuantumTrajectoryEnsemble
    artifact: ScientificArtifactEnvelope


def _step(model, step_size, time_unit, steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Electronic evolution requires a positive integer step count.")
    step = float(step_size) * float(conversion_factor(time_unit, model.units.time_unit))
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Electronic evolution step size must be finite and positive.")
    return step


def _history(model, densities, times, valid, method, extra):
    digest = canonical_fingerprint(
        {
            "model": model.artifact.artifact_id,
            "method": method,
            "result": array_tree_fingerprint((densities, times, valid)),
            "execution": extra,
        }
    )
    support = SeriesSupport(
        times,
        node_valid=jnp.broadcast_to(valid, times.shape),
        coordinate_name="electronic-evolution-time",
        coordinate_id=model.units.time_unit.unit_id,
    )
    history = SampledSeries(support, densities, series_id=digest)
    artifact = _envelope(
        "electronic-density-evolution",
        digest,
        (model.artifact.artifact_id,),
        model.artifact.license_id,
        successful=bool(valid),
    )
    return history, artifact


def evolve_electronics(
    model: PreparedElectronicModel,
    initial_density: ArrayLike,
    /,
    *,
    step_size: float,
    time_unit: UnitDefinition,
    steps: int,
    requested_use: Mapping[str, bool],
    method: str = "lindblad",
    tolerance: float = 1e-8,
) -> ElectronicEvolution:
    """Execute the native dense Lindblad or certified finite-CPTP solver."""
    model.require_rights(requested_use)
    step = _step(model, step_size, time_unit, steps)
    problem = model.density_problem(initial_density)
    if method == "lindblad":
        result = solve_lindblad(problem, step_size=step, steps=steps)
        densities, times = result.states, result.times
    elif method == "cptp":
        slicing = TemporalMesh.uniform(0.0, step * steps, steps)
        plan = model.finite_plan(
            slicing, time_unit=model.units.time_unit, tolerance=tolerance
        )
        result = integrate_finite_cptp(plan, problem.initial_density)
        densities, times = result.densities, plan.slicing.nodes
    else:
        raise ValueError("Electronic density method must be lindblad or cptp.")
    history, artifact = _history(
        model, densities, times, result.valid, method, {"tolerance": tolerance}
    )
    return ElectronicEvolution(model, history, result, artifact)


def evolve_electronic_jumps(
    model: PreparedElectronicModel,
    initial_state: ArrayLike,
    key: Array,
    /,
    *,
    step_size: float,
    time_unit: UnitDefinition,
    steps: int,
    trajectory_count: int,
    requested_use: Mapping[str, bool],
    maximum_state_elements: int = 16_777_216,
) -> ElectronicJumpEvolution:
    """Native fixed-step unraveling, with explicit trajectory/time-step evidence."""
    model.require_rights(requested_use)
    step = _step(model, step_size, time_unit, steps)
    if (
        isinstance(trajectory_count, bool)
        or not isinstance(trajectory_count, int)
        or trajectory_count < 1
    ):
        raise ValueError("trajectory_count must be a positive integer.")
    if (
        isinstance(maximum_state_elements, bool)
        or not isinstance(maximum_state_elements, int)
        or maximum_state_elements < 1
    ):
        raise ValueError("maximum_state_elements must be a positive integer.")
    if trajectory_count * (steps + 1) * model.dimension > maximum_state_elements:
        raise ValueError(
            "Electronic trajectory history exceeds the declared resource bound."
        )
    # Sufficient all-state rate bound, rather than testing only the initial state.
    collapse = np.asarray(model.collapse_operators)
    rate_bound = np.sum(np.abs(collapse) ** 2)
    if step * rate_bound > 0.1:
        raise ValueError("Declared step exceeds the native 0.1 jump-probability bound.")
    result = solve_quantum_jump_ensemble(
        model.jump_problem(initial_state),
        key,
        step_size=step,
        steps=steps,
        trajectory_count=trajectory_count,
    )
    densities = (
        ein.contract("kti,ktj->tij", result.states, jnp.conj(result.states))
        / trajectory_count
    )
    history, artifact = _history(
        model,
        densities,
        result.times,
        result.valid,
        "fixed-step-quantum-jumps",
        {
            "key": array_tree_fingerprint(jax.random.key_data(key)),
            "trajectory_count": trajectory_count,
        },
    )
    return ElectronicJumpEvolution(model, history, result, artifact)

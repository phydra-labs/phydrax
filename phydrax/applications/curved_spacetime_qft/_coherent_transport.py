#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-capacity coherent dark-sector quantum transport."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ..cosmology._dark_sector_species import DarkSectorSpeciesPlan
from ..cosmology._quantum_dark_kinetics import QuantumKineticState
from ..relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)


_COHERENT_SUCCESS = 0
_COHERENT_INVALID_NUMERICS = 1


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _adjoint(value: Array, /) -> Array:
    return jnp.conj(jnp.swapaxes(value, -1, -2))


class CoherentTransportProfile(StrictModule, NonTrainableState):
    support: tuple[str, ...] = eqx.field(static=True)
    refusals: tuple[str, ...] = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    provenance_ids: tuple[str, ...] = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    production_ready: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(self, production_evidence_ids: Sequence[str] = (), /):
        evidence = tuple(
            _identifier(value, "production_evidence_id")
            for value in production_evidence_ids
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("Production evidence identities must be unique.")
        support = (
            "on-shell-density-matrix",
            "finite-cell-momentum-species-internal-support",
            "cayley-unitary-plus-local-kraus-collision",
        )
        refusals = (
            "internal-dimension-above-four",
            "non-hermitian-hamiltonian",
            "non-trace-preserving-kraus-map",
            "post-hoc-positivity-projection",
        )
        differentiation = "algorithmic-through-fixed-cayley-and-kraus-map"
        rights_id = "phydra-native-no-external-provider"
        provenance_ids = ("cayley-kraus-density-matrix-kinetics",)
        self.support = support
        self.refusals = refusals
        self.differentiation = differentiation
        self.rights_id = rights_id
        self.provenance_ids = provenance_ids
        self.production_evidence_ids = evidence
        self.production_ready = bool(evidence)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "coherent-dark-transport-profile",
                "support": support,
                "refusals": refusals,
                "differentiation": differentiation,
                "rights_id": rights_id,
                "provenance_ids": provenance_ids,
                "production_evidence_ids": evidence,
            }
        )


class CoherentTransportPlan(StrictModule, NonTrainableState):
    quantum_support: QuantumKineticState
    species: tuple[DarkSectorSpeciesPlan, ...]
    momentum_weights: Array
    cell_volumes: Array
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    profile: CoherentTransportProfile
    linear_solve: SmallLinearSolvePlan
    species_plan_ids: tuple[str, ...] = eqx.field(static=True)
    charge_names: tuple[str, ...] = eqx.field(static=True)
    momentum_quadrature_id: str = eqx.field(static=True)
    internal_dimension: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    maximum_matrix_bytes: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantum_support: QuantumKineticState,
        momentum_weights: ArrayLike,
        cell_volumes: ArrayLike,
        /,
        *,
        momentum_quadrature_id: str,
        internal_dimension: int,
        hermiticity_tolerance: float = 1.0e-10,
        positivity_tolerance: float = 1.0e-10,
        trace_tolerance: float = 1.0e-10,
        maximum_matrix_bytes: int = 256_000_000,
        production_evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(quantum_support, QuantumKineticState):
            raise TypeError("quantum_support must be a QuantumKineticState.")
        if not isinstance(quantum_support.units, RelativisticUnitContract):
            raise TypeError("Quantum support must own a RelativisticUnitContract.")
        if not isinstance(quantum_support.frame, LocalRelativisticFramePlan):
            raise TypeError("Quantum support must own a LocalRelativisticFramePlan.")
        species = tuple(quantum_support.species)
        if not species or any(
            not isinstance(value, DarkSectorSpeciesPlan) for value in species
        ):
            raise TypeError("Quantum support species must use DarkSectorSpeciesPlan.")
        occupancy_shape = tuple(quantum_support.occupancy.shape)
        if len(occupancy_shape) != 3 or occupancy_shape[0] != len(species):
            raise ValueError(
                "Quantum occupancy must have shape (species, cell, momentum)."
            )
        momentum = np.asarray(momentum_weights, dtype=float)
        cells = np.asarray(cell_volumes, dtype=float)
        if (
            momentum.shape != (occupancy_shape[2],)
            or cells.shape != (occupancy_shape[1],)
            or np.any(~np.isfinite(momentum))
            or np.any(momentum <= 0.0)
            or np.any(~np.isfinite(cells))
            or np.any(cells <= 0.0)
        ):
            raise ValueError(
                "Momentum weights and cell volumes must be finite, positive, and support-aligned."
            )
        dimension = int(internal_dimension)
        if dimension < 1 or dimension > 4 or dimension != internal_dimension:
            raise ValueError(
                "Coherent internal_dimension must be an integer from one to four."
            )
        hermitian_tolerance = float(hermiticity_tolerance)
        psd_tolerance = float(positivity_tolerance)
        conservation_tolerance = float(trace_tolerance)
        maximum = int(maximum_matrix_bytes)
        if (
            not np.isfinite(hermitian_tolerance)
            or hermitian_tolerance < 0.0
            or not np.isfinite(psd_tolerance)
            or psd_tolerance < 0.0
            or not np.isfinite(conservation_tolerance)
            or conservation_tolerance < 0.0
            or maximum < 1
        ):
            raise ValueError("Coherent tolerances and resource capacity are invalid.")
        complex_bytes = np.dtype(np.complex128).itemsize
        required = int(np.prod(occupancy_shape)) * dimension * dimension * complex_bytes
        if required > maximum:
            raise ValueError(
                f"Coherent density matrices require {required} bytes; capacity is {maximum}."
            )
        charge_names = species[0].charge_names
        if any(value.charge_names != charge_names for value in species):
            raise ValueError(
                "All coherent species must use the same ordered charge names."
            )
        quadrature = _identifier(momentum_quadrature_id, "momentum_quadrature_id")
        units = quantum_support.units
        frame = quantum_support.frame
        if frame.units.contract_id != units.contract_id:
            raise ValueError(
                "Quantum support frame and unit contract identities disagree."
            )
        expected_mass_unit = units.scale.dimensional_scale.mass_unit.unit_id
        expected_energy_unit = units.energy_unit.unit_id
        if any(
            value.mass_unit != expected_mass_unit
            or value.energy_unit != expected_energy_unit
            for value in species
        ):
            raise ValueError(
                "Coherent species must bind the exact relativistic mass and energy units."
            )
        frame_realization = frame.realization_id()
        profile = CoherentTransportProfile(production_evidence_ids)
        self.quantum_support = quantum_support
        self.species = species
        self.momentum_weights = jnp.asarray(momentum)
        self.cell_volumes = jnp.asarray(cells)
        self.units = units
        self.frame = frame
        self.profile = profile
        self.linear_solve = SmallLinearSolvePlan(
            dimension,
            singular_tolerance=max(1.0e-14, conservation_tolerance * 0.01),
            maximum_condition=1.0e12,
            refinement_iterations=1,
        )
        self.species_plan_ids = tuple(value.species_plan_id for value in species)
        self.charge_names = charge_names
        self.momentum_quadrature_id = quadrature
        self.internal_dimension = dimension
        self.hermiticity_tolerance = hermitian_tolerance
        self.positivity_tolerance = psd_tolerance
        self.trace_tolerance = conservation_tolerance
        self.maximum_matrix_bytes = maximum
        self.support_id = quantum_support.support_id
        self.frame_id = frame.frame_id
        self.frame_realization_id = frame_realization
        self.unit_contract_id = units.contract_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-coherent-dark-transport",
                "support": self.support_id,
                "species": self.species_plan_ids,
                "frame": self.frame_id,
                "frame_realization": self.frame_realization_id,
                "units": self.unit_contract_id,
                "momentum_quadrature": quadrature,
                "momentum_weights": array_tree_fingerprint(momentum),
                "cell_volumes": array_tree_fingerprint(cells),
                "internal_dimension": dimension,
                "tolerances": (
                    hermitian_tolerance,
                    psd_tolerance,
                    conservation_tolerance,
                ),
                "maximum_matrix_bytes": maximum,
                "profile": profile.profile_id,
            }
        )

    @property
    def density_shape(self) -> tuple[int, ...]:
        species, cells, momenta = self.quantum_support.occupancy.shape
        return (cells, momenta, species, self.internal_dimension, self.internal_dimension)


class CoherentStateEvidence(StrictModule):
    hermiticity_residual: Array
    minimum_eigenvalue: Array
    trace_density: Array
    charge_by_cell: Array
    finite: Array
    hermitian: Array
    positive_semidefinite: Array
    support_respected: Array
    valid: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


class CoherentDensityMatrixState(StrictModule):
    density_matrix: Array
    time: Array
    evidence: CoherentStateEvidence
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def frame_token(self) -> Array:
        return self.evidence.frame_token

    @property
    def frame_time(self) -> Array:
        return self.evidence.frame_time

    @property
    def frame_scale_factor(self) -> Array:
        return self.evidence.frame_scale_factor

    @property
    def frame_realization_id(self) -> str:
        return self.evidence.frame_realization_id


class LocalKrausCollisionMap(StrictModule):
    operators: Array
    active: Array
    completeness_residual: Array
    trace_preserving: Array
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: ArrayLike,
        active: ArrayLike,
        /,
        *,
        internal_dimension: int,
        trace_tolerance: float = 1.0e-10,
        map_id: str,
    ):
        values = jnp.asarray(operators)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        mask = jnp.asarray(active, dtype=bool)
        dimension = int(internal_dimension)
        tolerance = float(trace_tolerance)
        if dimension < 1 or dimension > 4 or dimension != internal_dimension:
            raise ValueError(
                "Kraus internal_dimension must be an integer from one to four."
            )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Kraus trace_tolerance must be finite and nonnegative.")
        if values.ndim < 3 or values.shape[-2:] != (dimension, dimension):
            raise ValueError(
                "Kraus operators must end in the declared square internal shape."
            )
        if mask.shape != values.shape[:-2]:
            raise ValueError(
                "Kraus active mask must match every operator batch/channel axis."
            )
        selected = jnp.where(mask[..., None, None], values, 0.0)
        completeness = contract("...kai,...kaj->...ij", jnp.conj(selected), selected)
        identity = jnp.eye(dimension, dtype=values.dtype)
        residual = jnp.max(jnp.abs(completeness - identity), axis=(-2, -1))
        finite = jnp.all(jnp.isfinite(values), axis=(-2, -1))
        valid = jnp.all((~mask) | finite) & jnp.all(residual <= tolerance)
        values = eqx.error_if(
            values,
            ~valid,
            "Active Kraus operators must be finite and trace preserving.",
        )
        self.operators = values
        self.active = mask
        self.completeness_residual = residual
        self.trace_preserving = valid
        self.map_id = _identifier(map_id, "map_id")


class CoherentTransportObservables(StrictModule):
    number_density: Array
    charge_density: Array
    total_number: Array
    total_charge: Array
    internal_density_matrix: Array
    stress_energy: Array
    finite: Array
    state_valid: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


class CoherentTransportStep(StrictModule):
    state: CoherentDensityMatrixState
    candidate_state: CoherentDensityMatrixState
    unitary_residual: Array
    kraus_completeness_residual: Array
    trace_residual: Array
    charge_residual: Array
    accepted: Array
    rolled_back: Array
    status: Array
    differentiation: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)

    @property
    def frame_token(self) -> Array:
        return self.state.frame_token

    @property
    def frame_time(self) -> Array:
        return self.state.frame_time

    @property
    def frame_scale_factor(self) -> Array:
        return self.state.frame_scale_factor

    @property
    def frame_realization_id(self) -> str:
        return self.state.frame_realization_id


def coherent_state_evidence(
    plan: CoherentTransportPlan,
    density_matrix: ArrayLike,
    /,
) -> CoherentStateEvidence:
    if not isinstance(plan, CoherentTransportPlan):
        raise TypeError("plan must be CoherentTransportPlan.")
    density = jnp.asarray(density_matrix)
    if density.shape != plan.density_shape:
        raise ValueError(f"density_matrix must have shape {plan.density_shape}.")
    if not jnp.issubdtype(density.dtype, jnp.complexfloating):
        density = density.astype(jnp.complex128)
    hermiticity = jnp.max(jnp.abs(density - _adjoint(density)), axis=(-2, -1))
    eigenvalues = jnp.linalg.eigvalsh(density)
    minimum = jnp.min(eigenvalues, axis=-1)
    traces = jnp.real(jnp.trace(density, axis1=-2, axis2=-1))
    active = (
        jnp.asarray(plan.quantum_support.spatial_active, dtype=bool)[:, None, None]
        & jnp.asarray(plan.quantum_support.momentum_active, dtype=bool)[None, :, None]
    )
    active = jnp.broadcast_to(active, traces.shape)
    inactive_residual = jnp.max(jnp.abs(jnp.where(active[..., None, None], 0.0, density)))
    finite = jnp.all(jnp.isfinite(density))
    hermitian = jnp.all(hermiticity <= plan.hermiticity_tolerance)
    psd = jnp.all(minimum >= -plan.positivity_tolerance)
    support_respected = inactive_residual <= plan.trace_tolerance
    charges = jnp.stack(tuple(value.charges for value in plan.species), axis=0)
    weighted_trace = traces * plan.momentum_weights[None, :, None]
    charge = contract("xps,sc->xc", weighted_trace, charges)
    valid = finite & hermitian & psd & support_respected
    return CoherentStateEvidence(
        hermiticity,
        minimum,
        traces,
        charge,
        finite,
        hermitian,
        psd,
        support_respected,
        valid,
        plan.frame.frame_token,
        plan.frame.time,
        plan.frame.scale_factor,
        plan.frame_realization_id,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
    )


def coherent_density_matrix_state(
    plan: CoherentTransportPlan,
    density_matrix: ArrayLike,
    /,
    *,
    time: ArrayLike,
) -> CoherentDensityMatrixState:
    evidence = coherent_state_evidence(plan, density_matrix)
    density = eqx.error_if(
        jnp.asarray(density_matrix),
        ~evidence.valid,
        "Coherent state must be finite, Hermitian, positive semidefinite, and mask-respecting.",
    )
    time_ = jnp.asarray(time, dtype=density.real.dtype)
    time_ = eqx.error_if(
        time_, ~jnp.isfinite(time_), "Coherent state time must be finite."
    )
    return CoherentDensityMatrixState(
        density,
        time_,
        evidence,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
        plan.plan_id,
    )


def initialize_coherent_state(
    plan: CoherentTransportPlan,
    /,
    *,
    time: ArrayLike | None = None,
) -> CoherentDensityMatrixState:
    if not isinstance(plan, CoherentTransportPlan):
        raise TypeError("plan must be CoherentTransportPlan.")
    occupancy = jnp.transpose(plan.quantum_support.occupancy, (1, 2, 0))
    identity = jnp.eye(plan.internal_dimension, dtype=jnp.complex128)
    density = occupancy[..., None, None] * identity / plan.internal_dimension
    active = (
        jnp.asarray(plan.quantum_support.spatial_active, dtype=bool)[:, None, None]
        & jnp.asarray(plan.quantum_support.momentum_active, dtype=bool)[None, :, None]
    )
    density = jnp.where(active[..., None, None], density, 0.0)
    initial_time = plan.quantum_support.time if time is None else time
    return coherent_density_matrix_state(plan, density, time=initial_time)


def _checked_hamiltonian(
    plan: CoherentTransportPlan,
    value: ArrayLike,
    name: str,
    /,
) -> Array:
    matrix = jnp.asarray(value)
    if jnp.broadcast_shapes(matrix.shape, plan.density_shape) != plan.density_shape:
        raise ValueError(
            f"{name} Hamiltonian is not broadcastable to {plan.density_shape}."
        )
    matrix = jnp.broadcast_to(matrix, plan.density_shape)
    residual = jnp.max(jnp.abs(matrix - _adjoint(matrix)))
    return eqx.error_if(
        matrix,
        (~jnp.all(jnp.isfinite(matrix))) | (residual > plan.hermiticity_tolerance),
        f"{name} Hamiltonian must be finite and Hermitian.",
    )


def advance_coherent_transport(
    plan: CoherentTransportPlan,
    state: CoherentDensityMatrixState,
    /,
    *,
    time_step: ArrayLike,
    vacuum_hamiltonian: ArrayLike,
    mean_field_hamiltonian: ArrayLike,
    gravity_hamiltonian: ArrayLike,
    gauge_hamiltonian: ArrayLike,
    collision: LocalKrausCollisionMap | None = None,
) -> CoherentTransportStep:
    """Advance by one Cayley commutator and an optional local Kraus channel."""

    if not isinstance(plan, CoherentTransportPlan):
        raise TypeError("plan must be CoherentTransportPlan.")
    if not isinstance(state, CoherentDensityMatrixState) or state.plan_id != plan.plan_id:
        raise ValueError("Coherent state belongs to a different transport plan.")
    width = jnp.asarray(time_step, dtype=state.density_matrix.real.dtype)
    width = eqx.error_if(
        width,
        (~jnp.isfinite(width)) | (width <= 0.0),
        "Coherent time_step must be finite and positive.",
    )
    hamiltonian = (
        _checked_hamiltonian(plan, vacuum_hamiltonian, "vacuum")
        + _checked_hamiltonian(plan, mean_field_hamiltonian, "mean-field")
        + _checked_hamiltonian(plan, gravity_hamiltonian, "gravity")
        + _checked_hamiltonian(plan, gauge_hamiltonian, "gauge")
    )
    identity = jnp.eye(plan.internal_dimension, dtype=hamiltonian.dtype)
    left = identity + 0.5j * width * hamiltonian
    right = identity - 0.5j * width * hamiltonian
    solved = solve_small_linear(plan.linear_solve, left, right)
    unitary = eqx.error_if(
        solved.value,
        ~jnp.all(solved.successful),
        "Cayley transport solve failed its conditioning or residual policy.",
    )
    unitary_residual = jnp.max(
        jnp.abs(contract("...ji,...jk->...ik", jnp.conj(unitary), unitary) - identity)
    )
    evolved = contract(
        "...ai,...ij,...bj->...ab",
        unitary,
        state.density_matrix,
        jnp.conj(unitary),
    )
    kraus_residual = jnp.asarray(0.0, dtype=evolved.real.dtype)
    if collision is not None:
        if not isinstance(collision, LocalKrausCollisionMap):
            raise TypeError("collision must be LocalKrausCollisionMap or None.")
        if collision.operators.shape[-2:] != (
            plan.internal_dimension,
            plan.internal_dimension,
        ):
            raise ValueError("Collision internal dimension does not match the plan.")
        operators = jnp.where(collision.active[..., None, None], collision.operators, 0.0)
        batch_shape = evolved.shape[:-2]
        target_shape = batch_shape + (operators.shape[-3],) + evolved.shape[-2:]
        if jnp.broadcast_shapes(operators.shape, target_shape) != target_shape:
            raise ValueError("Collision batches do not broadcast over coherent support.")
        operators = jnp.broadcast_to(operators, target_shape)
        evolved = contract(
            "...kai,...ij,...kbj->...ab",
            operators,
            evolved,
            jnp.conj(operators),
        )
        kraus_residual = jnp.max(collision.completeness_residual)
    candidate_evidence = coherent_state_evidence(plan, evolved)
    candidate_state = CoherentDensityMatrixState(
        evolved,
        state.time + width,
        candidate_evidence,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
        plan.plan_id,
    )
    trace_residual = jnp.max(
        jnp.abs(candidate_evidence.trace_density - state.evidence.trace_density)
    )
    charge_residual = jnp.max(
        jnp.abs(candidate_evidence.charge_by_cell - state.evidence.charge_by_cell),
        initial=0.0,
    )
    finite = (
        candidate_evidence.valid
        & jnp.isfinite(unitary_residual)
        & jnp.isfinite(kraus_residual)
        & jnp.isfinite(trace_residual)
        & jnp.isfinite(charge_residual)
    )
    accepted = (
        finite
        & (unitary_residual <= 10.0 * plan.trace_tolerance)
        & (kraus_residual <= plan.trace_tolerance)
        & (trace_residual <= 10.0 * plan.trace_tolerance)
        & (charge_residual <= 10.0 * plan.trace_tolerance)
    )
    accepted_density = jnp.where(accepted, evolved, state.density_matrix)
    accepted_time = jnp.where(accepted, state.time + width, state.time)
    accepted_evidence = coherent_state_evidence(plan, accepted_density)
    accepted_state = CoherentDensityMatrixState(
        accepted_density,
        accepted_time,
        accepted_evidence,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
        plan.plan_id,
    )
    rolled_back = ~accepted
    status = jnp.where(
        accepted,
        jnp.asarray(_COHERENT_SUCCESS, dtype=jnp.int32),
        jnp.asarray(_COHERENT_INVALID_NUMERICS, dtype=jnp.int32),
    )
    return CoherentTransportStep(
        accepted_state,
        candidate_state,
        unitary_residual,
        kraus_residual,
        trace_residual,
        charge_residual,
        accepted,
        rolled_back,
        status,
        plan.profile.differentiation,
        plan.plan_id,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
    )


def coherent_transport_observables(
    plan: CoherentTransportPlan,
    state: CoherentDensityMatrixState,
    local_four_momenta: ArrayLike,
    /,
) -> CoherentTransportObservables:
    """Integrate trace, charges, internal coherence, and local-frame stress energy."""

    if not isinstance(plan, CoherentTransportPlan):
        raise TypeError("plan must be CoherentTransportPlan.")
    if not isinstance(state, CoherentDensityMatrixState) or state.plan_id != plan.plan_id:
        raise ValueError("Coherent state belongs to a different transport plan.")
    momentum = jnp.asarray(local_four_momenta)
    expected = (
        plan.quantum_support.occupancy.shape[2],
        len(plan.species),
        4,
    )
    if momentum.shape != expected:
        raise ValueError(f"local_four_momenta must have shape {expected}.")
    momentum = eqx.error_if(
        momentum,
        ~jnp.all(jnp.isfinite(momentum)) | jnp.any(momentum[..., 0] <= 0.0),
        "Local four-momenta must be finite with positive energy components.",
    )
    traces = state.evidence.trace_density
    weighted = traces * plan.momentum_weights[None, :, None]
    number = jnp.sum(weighted, axis=(1, 2))
    charges = jnp.stack(tuple(value.charges for value in plan.species), axis=0)
    charge = contract("xps,sc->xc", weighted, charges)
    total_number = contract("x,x->", plan.cell_volumes, number)
    total_charge = contract("x,xc->c", plan.cell_volumes, charge)
    internal = contract("p,xpsij->xsij", plan.momentum_weights, state.density_matrix)
    inverse_energy = 1.0 / momentum[..., 0]
    stress = contract(
        "xps,p,ps,psa,psb->xab",
        traces,
        plan.momentum_weights,
        inverse_energy,
        momentum,
        momentum,
    )
    finite = (
        state.evidence.valid
        & jnp.all(jnp.isfinite(number))
        & jnp.all(jnp.isfinite(charge))
        & jnp.all(jnp.isfinite(internal))
        & jnp.all(jnp.isfinite(stress))
    )
    return CoherentTransportObservables(
        number,
        charge,
        total_number,
        total_charge,
        internal,
        stress,
        finite,
        state.evidence.valid,
        plan.frame.frame_token,
        plan.frame.time,
        plan.frame.scale_factor,
        plan.frame_realization_id,
        plan.plan_id,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
    )


__all__ = [
    "CoherentDensityMatrixState",
    "CoherentStateEvidence",
    "CoherentTransportObservables",
    "CoherentTransportPlan",
    "CoherentTransportProfile",
    "CoherentTransportStep",
    "LocalKrausCollisionMap",
    "advance_coherent_transport",
    "coherent_density_matrix_state",
    "coherent_state_evidence",
    "coherent_transport_observables",
    "initialize_coherent_state",
]

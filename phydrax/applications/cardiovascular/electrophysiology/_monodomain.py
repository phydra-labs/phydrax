#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared affine-P1 phenomenological monodomain propagation.

This module couples a dimensional-time Aliev--Panfilov reaction to anisotropic
phenomenological diffusion.  It makes no physical ionic-current claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag
from math import isfinite
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import FiniteElementDiscretization
from ....lifecycle import (
    CheckpointManifest,
    CheckpointShard,
    create as create_lifecycle_archive,
    LifecycleArchive,
    open as open_lifecycle_archive,
    payload_byte_count,
    payload_digest,
)
from ....linalg import HermitianSpectrum, OperatorProperties
from ....metrix import (
    EuclideanStateGeometry,
    ProductStateGeometry,
    ProductStateGeometryBlock,
)
from ....sparse import SparseLinearMap
from ._aliev_panfilov import (
    AlievPanfilovEvidence,
    AlievPanfilovParameters,
    AlievPanfilovState,
    evaluate_aliev_panfilov,
)


_SSPRK33_NEGATIVE_REAL_STABILITY_RADIUS = 2.5127453266183286


class CellwiseDiffusivity(StrictModule, NonTrainableState):
    """Cellwise symmetric positive-semidefinite tensors in ``mm^2/ms``."""

    tensor_mm2_per_ms: Array
    cell_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    diffusivity_id: str = eqx.field(static=True)

    def __init__(self, tensor_mm2_per_ms: ArrayLike, /):
        tensor_host = np.asarray(tensor_mm2_per_ms, dtype=float)
        if (
            tensor_host.ndim != 3
            or tensor_host.shape[1] != tensor_host.shape[2]
            or tensor_host.shape[0] <= 0
            or tensor_host.shape[1] not in (2, 3)
        ):
            raise ValueError(
                "tensor_mm2_per_ms must have shape (cell_count, dimension, dimension) "
                "for dimension two or three."
            )
        if not np.all(np.isfinite(tensor_host)):
            raise ValueError("Diffusivity tensors must be finite.")
        scale = max(float(np.max(np.abs(tensor_host))), 1.0)
        tolerance = 128.0 * np.finfo(tensor_host.dtype).eps * scale
        if (
            float(np.max(np.abs(tensor_host - np.swapaxes(tensor_host, -1, -2))))
            > tolerance
        ):
            raise ValueError("Every diffusivity tensor must be symmetric.")
        tensor = jnp.asarray(tensor_host)
        spectrum = HermitianSpectrum(tensor, tolerance=tolerance)
        if not bool(np.all(np.asarray(spectrum.valid))):
            raise ValueError("Diffusivity spectral validation failed.")
        if float(np.min(np.asarray(spectrum.minimum_eigenvalue))) < -tolerance:
            raise ValueError("Every diffusivity tensor must be positive semidefinite.")
        self.tensor_mm2_per_ms = tensor
        self.cell_count = tensor_host.shape[0]
        self.dimension = tensor_host.shape[1]
        self.diffusivity_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-cellwise-diffusivity",
                "tensor_mm2_per_ms": array_tree_fingerprint(tensor_host),
                "cell_count": self.cell_count,
                "dimension": self.dimension,
                "symmetry": "K=K^T",
                "sign_convention": "positive K produces smoothing under -M_lumped^-1*S",
            }
        )

    @classmethod
    def from_fibers(
        cls,
        fiber_directions: ArrayLike,
        /,
        longitudinal_mm2_per_ms: ArrayLike,
        transverse_mm2_per_ms: ArrayLike,
    ) -> CellwiseDiffusivity:
        """Build sign-invariant ``K = d_t I + (d_l-d_t) f f^T`` tensors."""

        fibers = np.asarray(fiber_directions, dtype=float)
        if fibers.ndim != 2 or fibers.shape[0] <= 0 or fibers.shape[1] not in (2, 3):
            raise ValueError("fiber_directions must have shape (cell_count, 2 or 3).")
        if not np.all(np.isfinite(fibers)):
            raise ValueError("fiber_directions must be finite.")
        norms = np.sqrt(np.sum(fibers * fibers, axis=1))
        if np.any(norms <= 0.0):
            raise ValueError("Every fiber direction must be nonzero.")
        unit = fibers / norms[:, None]
        pivot = np.argmax(np.abs(unit), axis=1)
        pivot_values = unit[np.arange(unit.shape[0]), pivot]
        unit = unit * np.where(pivot_values < 0.0, -1.0, 1.0)[:, None]
        unit[unit == 0.0] = 0.0
        longitudinal = np.broadcast_to(
            np.asarray(longitudinal_mm2_per_ms, dtype=float), (fibers.shape[0],)
        )
        transverse = np.broadcast_to(
            np.asarray(transverse_mm2_per_ms, dtype=float), (fibers.shape[0],)
        )
        if (
            not np.all(np.isfinite(longitudinal))
            or not np.all(np.isfinite(transverse))
            or np.any(longitudinal < 0.0)
            or np.any(transverse < 0.0)
        ):
            raise ValueError(
                "Longitudinal and transverse diffusivities must be finite and nonnegative."
            )
        dyads = contract("ci,cj->cij", unit, unit)
        identity = np.eye(fibers.shape[1], dtype=float)[None, :, :]
        tensor = (
            transverse[:, None, None] * identity
            + (longitudinal - transverse)[:, None, None] * dyads
        )
        return cls(tensor)


class CellStimulusPulse(StrictModule, NonTrainableState):
    """A cell-selected, grid-aligned half-open activation-rate pulse."""

    cell_ids: Array
    cell_id_tuple: tuple[int, ...] = eqx.field(static=True)
    start_ms: float = eqx.field(static=True)
    stop_ms: float = eqx.field(static=True)
    amplitude_per_ms: float = eqx.field(static=True)
    pulse_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_ids: Sequence[int],
        start_ms: float,
        stop_ms: float,
        amplitude_per_ms: float,
        /,
    ):
        ids = tuple(int(value) for value in cell_ids)
        if not ids or len(set(ids)) != len(ids) or any(value < 0 for value in ids):
            raise ValueError("cell_ids must be nonempty, unique, and nonnegative.")
        start = float(start_ms)
        stop = float(stop_ms)
        amplitude = float(amplitude_per_ms)
        if not all(isfinite(value) for value in (start, stop, amplitude)):
            raise ValueError("Pulse times and amplitude must be finite.")
        if start < 0.0 or stop <= start:
            raise ValueError("Pulse interval must satisfy 0 <= start_ms < stop_ms.")
        self.cell_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.cell_id_tuple = ids
        self.start_ms = start
        self.stop_ms = stop
        self.amplitude_per_ms = amplitude
        self.pulse_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-cell-l2-stimulus-pulse",
                "cell_ids": list(ids),
                "interval_ms": [start, stop],
                "interval_convention": "half-open",
                "amplitude_per_ms": amplitude,
            }
        )


class PhenomenologicalMonodomainPlan(StrictModule, NonTrainableState):
    """Affine H1-P1 simplex FEM plan for phenomenological monodomain dynamics."""

    discretization: FiniteElementDiscretization
    diffusivity: CellwiseDiffusivity
    reaction: AlievPanfilovParameters
    pulses: tuple[CellStimulusPulse, ...]
    field_name: str = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    field_index: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        diffusivity: CellwiseDiffusivity,
        reaction: AlievPanfilovParameters,
        /,
        *,
        pulses: Sequence[CellStimulusPulse] = (),
        field_name: str = "activation",
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError(
                "discretization must be a prepared FiniteElementDiscretization."
            )
        if not isinstance(diffusivity, CellwiseDiffusivity):
            raise TypeError("diffusivity must be CellwiseDiffusivity.")
        if not isinstance(reaction, AlievPanfilovParameters):
            raise TypeError("reaction must be AlievPanfilovParameters.")
        pulse_tuple = tuple(pulses)
        if not all(isinstance(pulse, CellStimulusPulse) for pulse in pulse_tuple):
            raise TypeError("pulses must contain CellStimulusPulse values.")
        name = str(field_name)
        names = tuple(space.name for space in discretization.field_spaces)
        if name not in names:
            raise ValueError(f"Finite-element field {name!r} is not present.")
        field_index = names.index(name)
        dof_map = discretization.dof_maps[field_index]
        if dof_map.component_shape or dof_map.association != "vertex":
            raise ValueError(
                "Monodomain activation requires one scalar vertex-associated H1 field."
            )
        dimension = discretization.mesh.topological_dimension
        if dimension not in (2, 3) or discretization.mesh.ambient_dimension != dimension:
            raise ValueError(
                "Monodomain FEM requires a full-dimensional triangle or tetrahedron mesh."
            )
        cell_count = sum(block.cell_count for block in discretization.mesh.blocks)
        if diffusivity.cell_count != cell_count or diffusivity.dimension != dimension:
            raise ValueError("Diffusivity cells and dimension must match the FEM mesh.")
        expected_kind = "triangle" if dimension == 2 else "tetrahedron"
        elements = discretization.elements[field_index]
        for block, element, coordinate_element, coordinate_dofs in zip(
            discretization.mesh.blocks,
            elements,
            discretization.coordinate_elements,
            discretization.coordinate_dofs,
            strict=True,
        ):
            if (
                block.cell_kind != expected_kind
                or element.cell_kind != expected_kind
                or element.degree != 1
                or element.conformity != "H1"
                or element.mapping != "identity"
            ):
                raise ValueError(
                    "Monodomain FEM supports only scalar affine H1 P1 simplices."
                )
            if (
                coordinate_element.cell_kind != expected_kind
                or coordinate_element.degree != 1
            ):
                raise ValueError("Monodomain coordinate geometry must be affine P1.")
            if not np.array_equal(
                np.asarray(coordinate_dofs), np.asarray(block.vertices)
            ):
                raise ValueError(
                    "Affine P1 coordinate DOFs must coincide with simplex vertices."
                )
        if any(max(pulse.cell_id_tuple) >= cell_count for pulse in pulse_tuple):
            raise ValueError("A stimulus pulse indexes a cell outside the FEM mesh.")
        self.discretization = discretization
        self.diffusivity = diffusivity
        self.reaction = reaction
        self.pulses = pulse_tuple
        self.field_name = name
        self.node_count = dof_map.global_dof_count
        self.cell_count = cell_count
        self.dimension = dimension
        self.field_index = field_index
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-phenomenological-monodomain-plan",
                "fidelity_route": "affine-p1-aliev-panfilov-ssprk33",
                "discretization": discretization.prepared_id,
                "field_name": name,
                "diffusivity": diffusivity.diffusivity_id,
                "reaction": reaction.parameter_id,
                "pulses": [pulse.pulse_id for pulse in pulse_tuple],
                "mass": "row-sum-lumped-consistent-p1",
                "stimulus": "selected-cell-l2-load-projection",
                "coordinates_unit": "mm",
                "time_unit": "ms",
            }
        )

    def prepare(self, dt_ms: float, /) -> PreparedPhenomenologicalMonodomain:
        return prepare_phenomenological_monodomain(self, dt_ms)


class PreparedPhenomenologicalMonodomain(StrictModule, NonTrainableState):
    """Fully assembled fixed-shape SSPRK33 runtime; stepping performs no assembly."""

    plan: PhenomenologicalMonodomainPlan
    stiffness: SparseLinearMap
    lumped_mass: Array
    stimulus_projection_per_ms: Array
    pulse_start_steps: Array
    pulse_stop_steps: Array
    geometry: ProductStateGeometry
    dt_ms: float = eqx.field(static=True)
    diffusion_step_limit_ms: float = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PhenomenologicalMonodomainPlan,
        stiffness: SparseLinearMap,
        lumped_mass: Array,
        stimulus_projection_per_ms: Array,
        pulse_start_steps: Array,
        pulse_stop_steps: Array,
        geometry: ProductStateGeometry,
        dt_ms: float,
        diffusion_step_limit_ms: float,
        /,
    ):
        self.plan = plan
        self.stiffness = stiffness
        self.lumped_mass = lumped_mass
        self.stimulus_projection_per_ms = stimulus_projection_per_ms
        self.pulse_start_steps = pulse_start_steps
        self.pulse_stop_steps = pulse_stop_steps
        self.geometry = geometry
        self.dt_ms = dt_ms
        self.diffusion_step_limit_ms = diffusion_step_limit_ms
        self.state_layout_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-monodomain-flat-product-state",
                "geometry": geometry.geometry_id,
                "blocks": ["activation", "recovery"],
                "node_count": plan.node_count,
            }
        )
        self.execution_plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-monodomain-execution",
                "method": "SSPRK33",
                "dt_ms": dt_ms,
                "diffusion_step_limit_ms": (
                    diffusion_step_limit_ms if isfinite(diffusion_step_limit_ms) else None
                ),
                "diffusion_unbounded": not isfinite(diffusion_step_limit_ms),
                "pulse_interval": "grid-indexed-half-open",
            }
        )
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-phenomenological-monodomain",
                "plan": plan.plan_id,
                "state_layout": self.state_layout_id,
                "execution": self.execution_plan_id,
                "stiffness": stiffness.operator_id,
            }
        )

    def initialize(
        self,
        activation: ArrayLike,
        recovery: ArrayLike,
        /,
        *,
        time_ms: float = 0.0,
        step_index: int = 0,
    ) -> MonodomainState:
        return initialize_monodomain_state(
            self,
            activation,
            recovery,
            time_ms=time_ms,
            step_index=step_index,
        )

    def split(self, state: MonodomainState, /) -> tuple[Array, Array]:
        if not isinstance(state, MonodomainState):
            raise TypeError("state must be a MonodomainState.")
        if state.runtime_id != self.runtime_id:
            raise ValueError("Monodomain state does not match this prepared runtime.")
        return self.geometry.split(state.values)

    def evaluate(self, state: MonodomainState, /) -> MonodomainCandidate:
        return evaluate_monodomain_step(self, state)

    def commit(
        self, candidate: MonodomainCandidate, current: MonodomainState, /
    ) -> MonodomainState:
        if not isinstance(candidate, MonodomainCandidate):
            raise TypeError("candidate must be a MonodomainCandidate.")
        if not isinstance(current, MonodomainState):
            raise TypeError("current must be a MonodomainState.")
        if (
            candidate.source.runtime_id != self.runtime_id
            or candidate.proposed.runtime_id != self.runtime_id
            or current.runtime_id != self.runtime_id
        ):
            raise ValueError(
                "Monodomain candidate and state must match this prepared runtime."
            )
        return commit_monodomain_step(candidate, current)

    def advance(self, state: MonodomainState, /) -> MonodomainState:
        return self.commit(self.evaluate(state), state)


class MonodomainState(StrictModule):
    """Flat product-state coordinates and serial logical time."""

    values: Array
    time_ms: Array
    step_index: Array
    runtime_id: str = eqx.field(static=True)


class MonodomainStatus(IntFlag):
    """Fail-closed status for current state and all SSPRK33 stages."""

    SUCCESS = 0
    INVALID_CURRENT_STATE = 1
    TIME_STEP_MISMATCH = 2
    REACTION_STAGE_1_FAILURE = 4
    REACTION_STAGE_2_FAILURE = 8
    REACTION_STAGE_3_FAILURE = 16
    NONFINITE_STAGE_1 = 32
    NONFINITE_STAGE_2 = 64
    NONFINITE_STAGE_3 = 128


class MonodomainEvidence(StrictModule):
    """Current-state, reaction, and finite-stage evidence for one time step."""

    current_state_valid: Array
    time_step_consistent: Array
    stage_finite: Array
    reaction_successful: Array
    reaction_singularity_counts: Array
    status: Array
    successful: Array


class MonodomainCandidate(StrictModule):
    """Uncommitted SSPRK33 result bound to its exact source state."""

    source: MonodomainState
    proposed: MonodomainState
    evidence: MonodomainEvidence


class MonodomainReplayResult(StrictModule):
    """Serial replay result and deterministic content identity."""

    state: MonodomainState
    accepted_steps: Array
    successful: Array
    state_id: str = eqx.field(static=True)


def _local_mass(geometry) -> Array:
    if geometry.basis_values.ndim != 2:
        raise ValueError("Monodomain scalar P1 basis must have rank two.")
    return contract(
        "cq,qi,qj->cij",
        geometry.physical_weights,
        geometry.basis_values,
        geometry.basis_values,
        backend="jax",
    )


def _local_anisotropic_stiffness(geometry, tensor: Array) -> Array:
    return contract(
        "cq,cqid,cde,cqje->cij",
        geometry.physical_weights,
        geometry.physical_gradients,
        tensor,
        geometry.physical_gradients,
        backend="jax",
    )


def _local_cell_load(geometry) -> Array:
    return contract(
        "cq,qi->ci",
        geometry.physical_weights,
        geometry.basis_values,
        backend="jax",
    )


def _aligned_step(time_ms: float, dt_ms: float, name: str, /) -> int:
    ratio = time_ms / dt_ms
    nearest = int(round(ratio))
    tolerance = 64.0 * np.finfo(float).eps * max(abs(ratio), 1.0)
    if abs(ratio - nearest) > tolerance:
        raise ValueError(f"{name} must be aligned to the dt_ms grid.")
    return nearest


def prepare_phenomenological_monodomain(
    plan: PhenomenologicalMonodomainPlan, dt_ms: float, /
) -> PreparedPhenomenologicalMonodomain:
    """Assemble P1 mass, anisotropic stiffness, and selected-cell L2 loads once."""

    if not isinstance(plan, PhenomenologicalMonodomainPlan):
        raise TypeError("plan must be a PhenomenologicalMonodomainPlan.")
    dt = float(dt_ms)
    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_ms must be finite and positive.")
    start_steps = tuple(
        _aligned_step(pulse.start_ms, dt, "pulse start_ms") for pulse in plan.pulses
    )
    stop_steps = tuple(
        _aligned_step(pulse.stop_ms, dt, "pulse stop_ms") for pulse in plan.pulses
    )
    if any(stop <= start for start, stop in zip(start_steps, stop_steps, strict=True)):
        raise ValueError("Every grid-aligned pulse must occupy at least one time step.")

    mass_locals: list[Array] = []
    stiffness_locals: list[Array] = []
    cell_loads: list[Array] = []
    cell_offset = 0
    tensor = plan.diffusivity.tensor_mm2_per_ms
    for geometry, block in zip(
        plan.discretization.block_geometries[plan.field_index],
        plan.discretization.mesh.blocks,
        strict=True,
    ):
        block_slice = slice(cell_offset, cell_offset + block.cell_count)
        mass_locals.append(_local_mass(geometry))
        stiffness_locals.append(
            _local_anisotropic_stiffness(geometry, tensor[block_slice])
        )
        cell_loads.append(_local_cell_load(geometry))
        cell_offset += block.cell_count
    stiffness_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-monodomain-anisotropic-stiffness",
            "plan": plan.plan_id,
            "diffusivity": plan.diffusivity.diffusivity_id,
        }
    )
    properties = OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    stiffness = plan.discretization.assemble_cell_operator(
        plan.field_name,
        stiffness_locals,
        operator_id=stiffness_id,
        properties=properties,
    )
    mass = plan.discretization.assemble_cell_operator(
        plan.field_name,
        mass_locals,
        operator_id=canonical_fingerprint(
            {"kind": "cardiovascular-monodomain-consistent-mass", "plan": plan.plan_id}
        ),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    ones = jnp.ones((plan.node_count,), dtype=plan.discretization.mesh.coordinates.dtype)
    lumped_mass = mass.mv(ones)
    if not bool(np.all(np.asarray(lumped_mass) > 0.0)):
        raise ValueError("Row-sum lumping produced a nonpositive nodal mass.")

    all_cell_load = jnp.zeros((plan.cell_count, plan.node_count), dtype=lumped_mass.dtype)
    cell_offset = 0
    dof_map = plan.discretization.dof_maps[plan.field_index]
    for block, routes, local_load in zip(
        plan.discretization.mesh.blocks,
        dof_map.cell_dofs,
        cell_loads,
        strict=True,
    ):
        cell_ids = cell_offset + jnp.arange(block.cell_count, dtype=jnp.int32)
        rows = jnp.broadcast_to(cell_ids[:, None], routes.shape)
        all_cell_load = all_cell_load.at[rows, routes].add(local_load)
        cell_offset += block.cell_count
    if plan.pulses:
        projected = jnp.stack(
            tuple(
                pulse.amplitude_per_ms
                * jnp.sum(all_cell_load[pulse.cell_ids], axis=0)
                / lumped_mass
                for pulse in plan.pulses
            )
        )
    else:
        projected = jnp.zeros((0, plan.node_count), dtype=lumped_mass.dtype)

    dense_stiffness = stiffness.as_dense()
    inverse_root_mass = jax.lax.rsqrt(lumped_mass)
    symmetric_generator = (
        inverse_root_mass[:, None] * dense_stiffness * inverse_root_mass[None, :]
    )
    spectrum = HermitianSpectrum(symmetric_generator, tolerance=1.0e-6)
    if not bool(np.asarray(spectrum.valid)):
        raise ValueError("Diffusion stability spectrum is invalid.")
    largest = float(np.max(np.asarray(spectrum.eigenvalues)))
    diffusion_limit = (
        float("inf")
        if largest <= np.finfo(float).eps
        else _SSPRK33_NEGATIVE_REAL_STABILITY_RADIUS / largest
    )
    tolerance = 64.0 * np.finfo(float).eps * max(diffusion_limit, dt, 1.0)
    if dt > diffusion_limit + tolerance:
        raise ValueError(
            f"dt_ms={dt} exceeds the SSPRK33 diffusion-only bound {diffusion_limit} ms."
        )

    euclidean_activation = EuclideanStateGeometry(
        geometry_id=canonical_fingerprint(
            {
                "kind": "cardiovascular-monodomain-euclidean",
                "block": "activation",
                "plan": plan.plan_id,
            }
        )
    )
    euclidean_recovery = EuclideanStateGeometry(
        geometry_id=canonical_fingerprint(
            {
                "kind": "cardiovascular-monodomain-euclidean",
                "block": "recovery",
                "plan": plan.plan_id,
            }
        )
    )
    geometry = ProductStateGeometry(
        (
            ProductStateGeometryBlock(
                euclidean_activation, (plan.node_count,), block_id="activation"
            ),
            ProductStateGeometryBlock(
                euclidean_recovery, (plan.node_count,), block_id="recovery"
            ),
        ),
        geometry_id=canonical_fingerprint(
            {
                "kind": "cardiovascular-monodomain-product-geometry",
                "plan": plan.plan_id,
                "representation": "flat",
            }
        ),
    )
    return PreparedPhenomenologicalMonodomain(
        plan,
        stiffness,
        lumped_mass,
        projected,
        jnp.asarray(start_steps, dtype=jnp.int32),
        jnp.asarray(stop_steps, dtype=jnp.int32),
        geometry,
        dt,
        diffusion_limit,
    )


def initialize_monodomain_state(
    runtime: PreparedPhenomenologicalMonodomain,
    activation: ArrayLike,
    recovery: ArrayLike,
    /,
    *,
    time_ms: float = 0.0,
    step_index: int = 0,
) -> MonodomainState:
    """Validate and combine activation/recovery into the runtime's flat geometry."""

    if not isinstance(runtime, PreparedPhenomenologicalMonodomain):
        raise TypeError("runtime must be PreparedPhenomenologicalMonodomain.")
    activation_ = jnp.asarray(activation, dtype=runtime.lumped_mass.dtype)
    recovery_ = jnp.asarray(recovery, dtype=runtime.lumped_mass.dtype)
    expected = (runtime.plan.node_count,)
    if activation_.shape != expected or recovery_.shape != expected:
        raise ValueError(f"activation and recovery must each have shape {expected}.")
    if not bool(np.all(np.isfinite(np.asarray(activation_)))) or not bool(
        np.all(np.isfinite(np.asarray(recovery_)))
    ):
        raise ValueError("Initial monodomain state must be finite.")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a nonnegative integer.")
    time = float(time_ms)
    if not isfinite(time) or time < 0.0:
        raise ValueError("time_ms must be finite and nonnegative.")
    expected_time = step_index * runtime.dt_ms
    tolerance = 64.0 * np.finfo(float).eps * max(time, expected_time, 1.0)
    if abs(time - expected_time) > tolerance:
        raise ValueError("time_ms must equal step_index * dt_ms.")
    return MonodomainState(
        runtime.geometry.combine((activation_, recovery_)),
        jnp.asarray(time, dtype=activation_.dtype),
        jnp.asarray(step_index, dtype=jnp.int32),
        runtime.runtime_id,
    )


def _stimulus_for_step(
    runtime: PreparedPhenomenologicalMonodomain, step_index: Array, /
) -> Array:
    active = (step_index >= runtime.pulse_start_steps) & (
        step_index < runtime.pulse_stop_steps
    )
    return contract(
        "p,pn->n",
        active.astype(runtime.stimulus_projection_per_ms.dtype),
        runtime.stimulus_projection_per_ms,
        backend="jax",
    )


def _right_hand_side(
    runtime: PreparedPhenomenologicalMonodomain,
    values: Array,
    stimulus_per_ms: Array,
    /,
) -> tuple[Array, AlievPanfilovEvidence]:
    activation, recovery = runtime.geometry.split(values)
    reaction = evaluate_aliev_panfilov(
        runtime.plan.reaction,
        AlievPanfilovState(activation, recovery),
        activation_source_per_ms=stimulus_per_ms,
    )
    diffusion = -runtime.stiffness.mv(activation) / runtime.lumped_mass
    rates = runtime.geometry.combine(
        (
            diffusion + reaction.rates.activation_per_ms,
            reaction.rates.recovery_per_ms,
        )
    )
    return rates, reaction.evidence


def evaluate_monodomain_step(
    runtime: PreparedPhenomenologicalMonodomain,
    state: MonodomainState,
    /,
) -> MonodomainCandidate:
    """Evaluate one assembled SSPRK33 candidate with constant interval stimulus."""

    if not isinstance(runtime, PreparedPhenomenologicalMonodomain):
        raise TypeError("runtime must be PreparedPhenomenologicalMonodomain.")
    if not isinstance(state, MonodomainState):
        raise TypeError("state must be a MonodomainState.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Monodomain state does not match the prepared runtime.")
    if state.values.shape != (runtime.geometry.total_size,):
        raise ValueError("Monodomain state does not match the runtime product geometry.")
    stimulus = _stimulus_for_step(runtime, state.step_index)
    dt = jnp.asarray(runtime.dt_ms, dtype=state.values.dtype)
    first_rate, first_reaction = _right_hand_side(runtime, state.values, stimulus)
    first = state.values + dt * first_rate
    second_rate, second_reaction = _right_hand_side(runtime, first, stimulus)
    second = 0.75 * state.values + 0.25 * (first + dt * second_rate)
    third_rate, third_reaction = _right_hand_side(runtime, second, stimulus)
    third = (1.0 / 3.0) * state.values + (2.0 / 3.0) * (second + dt * third_rate)
    current_valid = (
        runtime.geometry.contains(state.values)
        & (state.step_index >= 0)
        & jnp.isfinite(state.time_ms)
    )
    expected_time = state.step_index.astype(state.time_ms.dtype) * runtime.dt_ms
    time_consistent = jnp.abs(state.time_ms - expected_time) <= (
        64.0
        * jnp.finfo(state.time_ms.dtype).eps
        * jnp.maximum(jnp.abs(expected_time), 1.0)
    )
    stage_finite = jnp.stack(
        (
            jnp.all(jnp.isfinite(first)),
            jnp.all(jnp.isfinite(second)),
            jnp.all(jnp.isfinite(third)),
        )
    )
    reaction_successful = jnp.stack(
        (
            first_reaction.successful,
            second_reaction.successful,
            third_reaction.successful,
        )
    )
    singularity_counts = jnp.stack(
        (
            first_reaction.singular_count,
            second_reaction.singular_count,
            third_reaction.singular_count,
        )
    )
    status = jnp.asarray(int(MonodomainStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        current_valid,
        status,
        jnp.bitwise_or(status, int(MonodomainStatus.INVALID_CURRENT_STATE)),
    )
    status = jnp.where(
        time_consistent,
        status,
        jnp.bitwise_or(status, int(MonodomainStatus.TIME_STEP_MISMATCH)),
    )
    reaction_flags = (
        MonodomainStatus.REACTION_STAGE_1_FAILURE,
        MonodomainStatus.REACTION_STAGE_2_FAILURE,
        MonodomainStatus.REACTION_STAGE_3_FAILURE,
    )
    finite_flags = (
        MonodomainStatus.NONFINITE_STAGE_1,
        MonodomainStatus.NONFINITE_STAGE_2,
        MonodomainStatus.NONFINITE_STAGE_3,
    )
    for index, flag in enumerate(reaction_flags):
        status = jnp.where(
            reaction_successful[index],
            status,
            jnp.bitwise_or(status, int(flag)),
        )
    for index, flag in enumerate(finite_flags):
        status = jnp.where(
            stage_finite[index],
            status,
            jnp.bitwise_or(status, int(flag)),
        )
    successful = status == int(MonodomainStatus.SUCCESS)
    proposed = MonodomainState(
        third,
        state.time_ms + dt,
        state.step_index + jnp.asarray(1, dtype=jnp.int32),
        runtime.runtime_id,
    )
    return MonodomainCandidate(
        state,
        proposed,
        MonodomainEvidence(
            current_valid,
            time_consistent,
            stage_finite,
            reaction_successful,
            singularity_counts,
            status,
            successful,
        ),
    )


def commit_monodomain_step(
    candidate: MonodomainCandidate, current: MonodomainState, /
) -> MonodomainState:
    """Commit only a successful candidate evaluated from exactly ``current``."""

    if not isinstance(candidate, MonodomainCandidate):
        raise TypeError("candidate must be a MonodomainCandidate.")
    if not isinstance(current, MonodomainState):
        raise TypeError("current must be a MonodomainState.")
    if candidate.proposed.runtime_id != candidate.source.runtime_id:
        raise ValueError(
            "Monodomain candidate runtime identity is internally inconsistent."
        )
    source_matches = (
        (candidate.source.runtime_id == current.runtime_id)
        & jnp.array_equal(candidate.source.values, current.values, equal_nan=True)
        & (candidate.source.time_ms == current.time_ms)
        & (candidate.source.step_index == current.step_index)
    )
    return jax.lax.cond(
        candidate.evidence.successful & source_matches,
        lambda _: candidate.proposed,
        lambda _: current,
        operand=None,
    )


def run_monodomain_steps(
    runtime: PreparedPhenomenologicalMonodomain,
    state: MonodomainState,
    step_count: int,
    /,
) -> MonodomainReplayResult:
    """Run a serial deterministic lifecycle, stopping at the first rejected step."""

    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
        raise ValueError("step_count must be a nonnegative integer.")
    current = state
    accepted = 0
    successful = True
    for _ in range(step_count):
        candidate = runtime.evaluate(current)
        if not bool(np.asarray(candidate.evidence.successful)):
            successful = False
            break
        current = runtime.commit(candidate, current)
        accepted += 1
    return MonodomainReplayResult(
        current,
        jnp.asarray(accepted, dtype=jnp.int32),
        jnp.asarray(successful),
        monodomain_state_identity(runtime, current),
    )


def monodomain_state_identity(
    runtime: PreparedPhenomenologicalMonodomain, state: MonodomainState, /
) -> str:
    """Return deterministic content identity bound to the prepared runtime."""

    if not isinstance(runtime, PreparedPhenomenologicalMonodomain):
        raise TypeError("runtime must be PreparedPhenomenologicalMonodomain.")
    if not isinstance(state, MonodomainState):
        raise TypeError("state must be a MonodomainState.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Monodomain state does not match the prepared runtime.")
    return canonical_fingerprint(
        {
            "kind": "cardiovascular-monodomain-state",
            "runtime": runtime.runtime_id,
            "state_runtime": state.runtime_id,
            "values": array_tree_fingerprint(state.values),
            "time_ms": array_tree_fingerprint(state.time_ms),
            "step_index": array_tree_fingerprint(state.step_index),
        }
    )


def write_monodomain_checkpoint(
    runtime: PreparedPhenomenologicalMonodomain,
    state: MonodomainState,
    path: str | Path,
    /,
    *,
    parent_checkpoint_id: str | None = None,
) -> LifecycleArchive:
    """Create one complete checksum-bound serial checkpoint archive."""

    if not isinstance(runtime, PreparedPhenomenologicalMonodomain):
        raise TypeError("runtime must be PreparedPhenomenologicalMonodomain.")
    if not isinstance(state, MonodomainState):
        raise TypeError("state must be a MonodomainState.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("Cannot checkpoint state from another monodomain runtime.")
    if not bool(np.asarray(runtime.geometry.contains(state.values))):
        raise ValueError("Cannot checkpoint a nonfinite monodomain state.")
    activation, recovery = runtime.split(state)
    time = np.asarray(state.time_ms)
    step = np.asarray(state.step_index)
    if (
        time.shape != ()
        or step.shape != ()
        or not np.issubdtype(step.dtype, np.integer)
        or not np.isfinite(time)
        or int(step) < 0
    ):
        raise ValueError("Cannot checkpoint invalid serial time or step state.")
    expected_time = int(step) * runtime.dt_ms
    tolerance = 64.0 * np.finfo(float).eps * max(float(time), expected_time, 1.0)
    if abs(float(time) - expected_time) > tolerance:
        raise ValueError("Cannot checkpoint time inconsistent with step_index * dt_ms.")
    arrays = {
        "activation": np.asarray(activation),
        "recovery": np.asarray(recovery),
        "time_ms": time,
        "step_index": step,
    }
    shards = tuple(
        CheckpointShard(
            name,
            payload_digest(value),
            payload_byte_count(value),
            (runtime.state_layout_id,),
        )
        for name, value in arrays.items()
    )
    checkpoint_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-monodomain-checkpoint",
            "runtime": runtime.runtime_id,
            "state": monodomain_state_identity(runtime, state),
            "parent_checkpoint_id": parent_checkpoint_id,
        }
    )
    manifest = CheckpointManifest(
        checkpoint_id,
        runtime.plan.plan_id,
        runtime.runtime_id,
        runtime.execution_plan_id,
        shards,
        complete=True,
        parent_checkpoint_id=parent_checkpoint_id,
    )
    return create_lifecycle_archive(path, manifest=manifest, arrays=arrays)


def read_monodomain_checkpoint(
    runtime: PreparedPhenomenologicalMonodomain,
    path: str | Path,
    /,
) -> MonodomainState:
    """Open and validate a serial checkpoint, failing closed on any mismatch."""

    if not isinstance(runtime, PreparedPhenomenologicalMonodomain):
        raise TypeError("runtime must be PreparedPhenomenologicalMonodomain.")
    archive = open_lifecycle_archive(path)
    manifest = archive.manifest
    if not isinstance(manifest, CheckpointManifest):
        raise ValueError("Archive does not contain a checkpoint manifest.")
    if (
        manifest.analysis_plan_id != runtime.plan.plan_id
        or manifest.numeric_revision_id != runtime.runtime_id
        or manifest.execution_plan_id != runtime.execution_plan_id
    ):
        raise ValueError(
            "Checkpoint identity does not match the prepared monodomain runtime."
        )
    activation = np.asarray(archive.arrays["activation"])
    recovery = np.asarray(archive.arrays["recovery"])
    time = np.asarray(archive.arrays["time_ms"])
    step = np.asarray(archive.arrays["step_index"])
    expected = (runtime.plan.node_count,)
    if (
        activation.shape != expected
        or recovery.shape != expected
        or time.shape != ()
        or step.shape != ()
    ):
        raise ValueError(
            "Checkpoint state shape does not match the prepared monodomain runtime."
        )
    expected_dtype = np.dtype(runtime.lumped_mass.dtype)
    if (
        activation.dtype != expected_dtype
        or recovery.dtype != expected_dtype
        or time.dtype != expected_dtype
        or step.dtype != np.dtype(np.int32)
    ):
        raise ValueError(
            "Checkpoint state dtype does not match the prepared monodomain runtime."
        )
    step_value = int(step)
    time_value = float(time)
    state = initialize_monodomain_state(
        runtime,
        activation,
        recovery,
        time_ms=time_value,
        step_index=step_value,
    )
    expected_checkpoint_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-monodomain-checkpoint",
            "runtime": runtime.runtime_id,
            "state": monodomain_state_identity(runtime, state),
            "parent_checkpoint_id": manifest.parent_checkpoint_id,
        }
    )
    if manifest.checkpoint_id != expected_checkpoint_id:
        raise ValueError("Checkpoint content identity does not match its restored state.")
    return state


__all__ = [
    "CellStimulusPulse",
    "CellwiseDiffusivity",
    "MonodomainCandidate",
    "MonodomainEvidence",
    "MonodomainReplayResult",
    "MonodomainState",
    "MonodomainStatus",
    "PhenomenologicalMonodomainPlan",
    "PreparedPhenomenologicalMonodomain",
    "commit_monodomain_step",
    "evaluate_monodomain_step",
    "initialize_monodomain_state",
    "monodomain_state_identity",
    "prepare_phenomenological_monodomain",
    "read_monodomain_checkpoint",
    "run_monodomain_steps",
    "write_monodomain_checkpoint",
]

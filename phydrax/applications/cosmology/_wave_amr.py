#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology and accepted-boundary adaptive AMR for complex wave matter."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, pi, prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._execution_plan import ExecutionPlan
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.amr._complex_field import (
    complex_amr_fill_patch,
    ComplexCompositeAMRCellLayout,
    complexify_composite_amr_operator,
)
from ...discretization.amr._core import (
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelState,
)
from ...discretization.amr._distributed import (
    BlockAMRPartitionPlan,
    PreparedDistributedBlockAMRHierarchy,
)
from ...discretization.amr._fd_runtime import PreparedFDAMRHierarchy
from ...discretization.amr._topology_compiler import BlockTopologyCompileResult
from ...discretization.amr._topology_transfer import BlockFieldTopologyTransition
from ...discretization.finite_volume._amr_diffusion import (
    CompositeAMRDiffusionPlan,
    PreparedCompositeAMRDiffusion,
)
from ...execution import ExecutionGroup
from ...linalg import (
    FunctionLinearOperator,
    GMRES,
    IdentityLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
    TolerancePolicy,
)
from ...solver._distributed_wave_amr import (
    DistributedWaveAMRResult,
    DistributedWaveAMRState,
    DistributedWaveAMRTopologyTransferResult,
    PreparedDistributedWaveAMR,
    PreparedDistributedWaveAMRTopologyTransition,
)
from ._background import FLRWBackground
from ._cosmological_amr import (
    BlockAMRGravityPlan,
    BlockAMRGravityResult,
    BlockAMRParticleRoutingPlan,
)
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract
from ._wave_boundaries import (
    IsolatedWaveBoundaryDescriptor,
    PeriodicWaveBoundaryDescriptor,
    WaveBoundaryDescriptor,
)


class WaveAMRPhysicsPlan(StrictModule, NonTrainableState):
    """Schrödinger--Poisson coefficients in one cosmological code-unit contract."""

    boson_mass: float = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    scale: CosmologyScaleContract
    dtype: np.dtype = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boson_mass: float,
        /,
        *,
        gravitational_constant: float = 1.0,
        reduced_planck_constant: float = 1.0,
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
        dtype: Any = np.complex128,
    ):
        mass = float(boson_mass)
        gravity = float(gravitational_constant)
        hbar = float(reduced_planck_constant)
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not all(isfinite(value) and value > 0.0 for value in (mass, gravity, hbar)):
            raise ValueError(
                "Wave AMR physical coefficients must be finite and positive."
            )
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be CosmologyScaleContract.")
        if scale.length_coordinate_kind != "comoving":
            raise ValueError("Wave AMR requires comoving Cartesian length coordinates.")
        if not np.issubdtype(dtype_, np.complexfloating):
            raise TypeError("Wave AMR requires a native complex dtype.")
        self.boson_mass = mass
        self.gravitational_constant = gravity
        self.reduced_planck_constant = hbar
        self.scale = scale
        self.dtype = dtype_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-amr-physics",
                "boson_mass": mass,
                "gravitational_constant": gravity,
                "reduced_planck_constant": hbar,
                "scale": scale.scale_id,
                "dtype": dtype_.str,
                "state": "pure-complex-psi",
            }
        )


class WaveAMRAdaptivityPlan(StrictModule, NonTrainableState):
    """Host topology proposal thresholds with explicit hysteresis and gates."""

    minimum_de_broglie_cells: float = eqx.field(static=True)
    maximum_phase_change: float = eqx.field(static=True)
    maximum_density_contrast: float = eqx.field(static=True)
    maximum_quantum_potential_indicator: float = eqx.field(static=True)
    vortex_threshold: float = eqx.field(static=True)
    coarsening_hysteresis: float = eqx.field(static=True)
    relative_node_floor: float = eqx.field(static=True)
    probability_relative_tolerance: float = eqx.field(static=True)
    current_relative_tolerance: float = eqx.field(static=True)
    current_absolute_tolerance: float = eqx.field(static=True)
    phase_defect_tolerance: float = eqx.field(static=True)
    winding_absolute_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        minimum_de_broglie_cells: float = 4.0,
        maximum_phase_change: float = 0.5,
        maximum_density_contrast: float = 2.0,
        maximum_quantum_potential_indicator: float = 1.0,
        vortex_threshold: float = pi,
        coarsening_hysteresis: float = 0.6,
        relative_node_floor: float = 1.0e-10,
        probability_relative_tolerance: float = 1.0e-10,
        current_relative_tolerance: float = 0.2,
        current_absolute_tolerance: float = 1.0e-10,
        phase_defect_tolerance: float = 0.75,
        winding_absolute_tolerance: float = 1.0e-8,
    ):
        values = tuple(
            float(value)
            for value in (
                minimum_de_broglie_cells,
                maximum_phase_change,
                maximum_density_contrast,
                maximum_quantum_potential_indicator,
                vortex_threshold,
                coarsening_hysteresis,
                relative_node_floor,
                probability_relative_tolerance,
                current_relative_tolerance,
                current_absolute_tolerance,
                phase_defect_tolerance,
                winding_absolute_tolerance,
            )
        )
        if (
            not all(isfinite(value) for value in values)
            or values[0] < 2.0
            or any(value <= 0.0 for value in values[1:5])
            or not 0.0 < values[5] < 1.0
            or not 0.0 < values[6] < 1.0
            or any(value < 0.0 for value in values[7:])
        ):
            raise ValueError("Wave AMR adaptivity values are invalid.")
        (
            self.minimum_de_broglie_cells,
            self.maximum_phase_change,
            self.maximum_density_contrast,
            self.maximum_quantum_potential_indicator,
            self.vortex_threshold,
            self.coarsening_hysteresis,
            self.relative_node_floor,
            self.probability_relative_tolerance,
            self.current_relative_tolerance,
            self.current_absolute_tolerance,
            self.phase_defect_tolerance,
            self.winding_absolute_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-amr-adaptivity",
                "minimum_de_broglie_cells": values[0],
                "maximum_phase_change": values[1],
                "maximum_density_contrast": values[2],
                "maximum_quantum_potential_indicator": values[3],
                "vortex_threshold": values[4],
                "coarsening_hysteresis": values[5],
                "relative_node_floor": values[6],
                "probability_relative_tolerance": values[7],
                "current_relative_tolerance": values[8],
                "current_absolute_tolerance": values[9],
                "phase_defect_tolerance": values[10],
                "winding_absolute_tolerance": values[11],
            }
        )


class WaveAMRDiscretizationPlan(StrictModule, NonTrainableState):
    """AMR geometry, boundary, global-solve, and adaptive topology policy."""

    hierarchy: PreparedFDAMRHierarchy
    boundary: WaveBoundaryDescriptor
    adaptivity: WaveAMRAdaptivityPlan | None
    solve_relative_tolerance: float = eqx.field(static=True)
    solve_absolute_tolerance: float = eqx.field(static=True)
    maximum_solve_steps: int = eqx.field(static=True)
    norm_relative_tolerance: float = eqx.field(static=True)
    self_adjoint_tolerance: float = eqx.field(static=True)
    maximum_phase_radians: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: PreparedFDAMRHierarchy,
        /,
        *,
        boundary: WaveBoundaryDescriptor | None = None,
        adaptivity: WaveAMRAdaptivityPlan | None = None,
        solve_relative_tolerance: float = 1.0e-9,
        solve_absolute_tolerance: float = 1.0e-11,
        maximum_solve_steps: int = 600,
        norm_relative_tolerance: float = 1.0e-8,
        self_adjoint_tolerance: float = 1.0e-10,
        maximum_phase_radians: float = 0.75,
    ):
        if not isinstance(hierarchy, PreparedFDAMRHierarchy):
            raise TypeError("hierarchy must be PreparedFDAMRHierarchy.")
        boundary_ = PeriodicWaveBoundaryDescriptor() if boundary is None else boundary
        if not isinstance(
            boundary_, (PeriodicWaveBoundaryDescriptor, IsolatedWaveBoundaryDescriptor)
        ):
            raise TypeError("boundary must be a wave boundary descriptor.")
        if adaptivity is not None and not isinstance(adaptivity, WaveAMRAdaptivityPlan):
            raise TypeError("adaptivity must be WaveAMRAdaptivityPlan or None.")
        values = tuple(
            float(value)
            for value in (
                solve_relative_tolerance,
                solve_absolute_tolerance,
                norm_relative_tolerance,
                self_adjoint_tolerance,
                maximum_phase_radians,
            )
        )
        steps = int(maximum_solve_steps)
        if (
            not all(isfinite(value) and value >= 0.0 for value in values[:-1])
            or not isfinite(values[-1])
            or values[-1] <= 0.0
            or steps < 1
        ):
            raise ValueError("Wave AMR solve policy values are invalid.")
        self.hierarchy = hierarchy
        self.boundary = boundary_
        self.adaptivity = adaptivity
        self.solve_relative_tolerance = values[0]
        self.solve_absolute_tolerance = values[1]
        self.maximum_solve_steps = steps
        self.norm_relative_tolerance = values[2]
        self.self_adjoint_tolerance = values[3]
        self.maximum_phase_radians = values[4]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-amr-discretization",
                "hierarchy": hierarchy.prepared_id,
                "boundary": boundary_.descriptor_id,
                "adaptivity": None if adaptivity is None else adaptivity.plan_id,
                "solve_relative_tolerance": values[0],
                "solve_absolute_tolerance": values[1],
                "maximum_solve_steps": steps,
                "norm_relative_tolerance": values[2],
                "self_adjoint_tolerance": values[3],
                "maximum_phase_radians": values[4],
                "profile": "fixed-topology-synchronized-global-composite",
            }
        )

    def prepare(
        self,
        physics: WaveAMRPhysicsPlan,
        topology: BlockHierarchyTopology,
        background: FLRWBackground,
        /,
    ) -> "PreparedWaveAMR":
        if isinstance(self.boundary, IsolatedWaveBoundaryDescriptor):
            raise ValueError(
                "Isolated wave AMR requires a finite-domain multipole Poisson owner; "
                "the periodic composite Poisson profile cannot alias that boundary."
            )
        return PreparedWaveAMR(self, physics, self.hierarchy, topology, background)


class WaveAMRState(StrictModule):
    """Pure complex wave hierarchy at one synchronized scale-factor time level."""

    psi: BlockHierarchyState
    scale_factor: Array
    accepted_boundary: Array


class WaveAMRDiagnostics(StrictModule):
    initial_probability: Array
    final_probability: Array
    probability_relative_error: Array
    initial_current: Array
    final_current: Array
    current_relative_defect: Array
    current_absolute_defect: Array
    cayley_relative_residual: Array
    self_adjoint_residual: Array
    maximum_kinetic_phase: Array
    maximum_potential_phase: Array
    kinetic_energy: Array
    potential_energy: Array
    finite: Array
    initial_poisson_closed: Array
    final_poisson_closed: Array
    poisson_closed: Array
    kinetic_closed: Array
    accepted: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "non_finite",
            "poisson_failed",
            "kinetic_solve_failed",
            "cayley_residual",
            "self_adjoint_residual",
            "probability_drift",
            "phase_unresolved",
        ),
    )


class WaveAMRResult(StrictModule):
    state: WaveAMRState
    candidate_state: WaveAMRState
    diagnostics: WaveAMRDiagnostics
    gravity: BlockAMRGravityResult
    second_gravity: BlockAMRGravityResult
    kinetic_solve: LinearSolveResult
    successful: Array
    prepared_id: str = eqx.field(static=True)


class WaveAMRTopologyIndicators(StrictModule):
    de_broglie: tuple[Array, ...]
    phase_change: tuple[Array, ...]
    density_contrast: tuple[Array, ...]
    quantum_potential: tuple[Array, ...]
    vortex: tuple[Array, ...]
    node: tuple[Array, ...]
    maximum_values: Array
    finite: Array


class WaveAMRTopologyProposal(StrictModule, NonTrainableState):
    """Host-compiled candidate topology and complete fixed-capacity tag evidence."""

    tags: tuple[Array, ...]
    indicators: WaveAMRTopologyIndicators
    compilation: BlockTopologyCompileResult
    source_prepared_id: str = eqx.field(static=True)
    source_epoch_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    accepted_boundary: bool = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)


class WaveAMRTransferEvidence(StrictModule):
    source_probability: Array
    target_probability: Array
    probability_relative_defect: Array
    source_current: Array
    target_current: Array
    current_relative_defect: Array
    current_absolute_defect: Array
    source_winding: Array
    target_winding: Array
    winding_absolute_defect: Array
    phase_defect: Array
    source_node_count: Array
    target_node_count: Array
    source_vortex_count: Array
    target_vortex_count: Array
    finite: Array
    probability_preserved: Array
    current_preserved: Array
    winding_preserved: Array
    successful: Array
    transition_id: str = eqx.field(static=True)


class WaveAMRTransitionResult(StrictModule):
    state: WaveAMRState
    candidate_state: WaveAMRState
    prepared: "PreparedWaveAMR"
    candidate_prepared: "PreparedWaveAMR"
    evidence: WaveAMRTransferEvidence
    successful: Array
    rolled_back: Array
    proposal_id: str = eqx.field(static=True)


class WaveAMRDistributedTransitionResult(StrictModule):
    state: DistributedWaveAMRState
    candidate_state: DistributedWaveAMRState
    prepared: "PreparedWaveAMR"
    distributed_preparation: "WaveAMRDistributedPreparation"
    candidate_prepared: "PreparedWaveAMR"
    candidate_distributed_preparation: "WaveAMRDistributedPreparation"
    evidence: WaveAMRTransferEvidence
    transfer: DistributedWaveAMRTopologyTransferResult
    successful: Array
    rolled_back: Array
    proposal_id: str = eqx.field(static=True)


class WaveAMRDistributedPreparation(StrictModule, NonTrainableState):
    hierarchy: PreparedDistributedBlockAMRHierarchy | None
    execution: PreparedDistributedWaveAMR | None
    admitted: bool = eqx.field(static=True)
    executable: bool = eqx.field(static=True)
    preflight_required_bytes: int = eqx.field(static=True)
    required_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_plan_id: str = eqx.field(static=True)
    execution_plan_id: str | None = eqx.field(static=True)
    mesh_id: str | None = eqx.field(static=True)
    operator_id: str | None = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


class PreparedWaveAMR(StrictModule, NonTrainableState):
    """One immutable AMR epoch with global composite Poisson and Cayley solves."""

    plan: WaveAMRDiscretizationPlan
    physics: WaveAMRPhysicsPlan
    fd_hierarchy: PreparedFDAMRHierarchy
    topology: BlockHierarchyTopology
    background: FLRWBackground
    layout: ComplexCompositeAMRCellLayout
    diffusion: PreparedCompositeAMRDiffusion
    kinetic_operator: FunctionLinearOperator
    gravity: BlockAMRGravityPlan
    solve_policy: LinearSolvePolicy
    kinetic_spectral_upper_bound: float = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    potential_convention: str = eqx.field(static=True)
    time_level_convention: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: WaveAMRDiscretizationPlan,
        physics: WaveAMRPhysicsPlan,
        fd_hierarchy: PreparedFDAMRHierarchy,
        topology: BlockHierarchyTopology,
        background: FLRWBackground,
        /,
    ):
        if not isinstance(plan, WaveAMRDiscretizationPlan):
            raise TypeError("plan must be WaveAMRDiscretizationPlan.")
        if not isinstance(physics, WaveAMRPhysicsPlan):
            raise TypeError("physics must be WaveAMRPhysicsPlan.")
        if not isinstance(fd_hierarchy, PreparedFDAMRHierarchy):
            raise TypeError("fd_hierarchy must be PreparedFDAMRHierarchy.")
        if fd_hierarchy.prepared_id != plan.hierarchy.prepared_id:
            raise ValueError(
                "Prepared Wave AMR hierarchy does not match its discretization plan."
            )
        if not isinstance(topology, BlockHierarchyTopology):
            raise TypeError("topology must be BlockHierarchyTopology.")
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if topology.plan.plan_id != plan.hierarchy.plan.hierarchy.plan_id:
            raise ValueError("Wave AMR topology does not match its hierarchy plan.")
        if not all(topology.plan.periodic_axes):
            raise ValueError("The first Wave AMR profile is periodic on every axis.")
        if not isinstance(plan.boundary, PeriodicWaveBoundaryDescriptor):
            raise ValueError(
                "Prepared periodic Wave AMR cannot alias an isolated boundary."
            )
        if background.scale.scale_id != physics.scale.scale_id:
            raise ValueError("Wave AMR background and physics scale contracts disagree.")
        if float(np.asarray(background.curvature_density)) != 0.0:
            raise ValueError("Periodic Wave AMR requires flat FLRW spatial geometry.")
        if plan.adaptivity is not None and any(
            width < 1 for level in topology.plan.levels for width in level.halo_width
        ):
            raise ValueError(
                "Adaptive Wave AMR requires at least one FillPatch halo cell per axis."
            )
        layout = ComplexCompositeAMRCellLayout(topology, dtype=physics.dtype)
        diffusion = CompositeAMRDiffusionPlan(layout.real_layout).prepare(1.0)
        kinetic = complexify_composite_amr_operator(diffusion, layout)
        routing = BlockAMRParticleRoutingPlan(topology)
        gravity = BlockAMRGravityPlan(
            diffusion,
            routing,
            gravitational_constant=physics.gravitational_constant,
        )
        solve_policy = LinearSolvePolicy(
            GMRES(restart=min(50, max(8, layout.space.size))),
            tolerance=TolerancePolicy(
                relative=plan.solve_relative_tolerance,
                absolute=plan.solve_absolute_tolerance,
                max_steps=plan.maximum_solve_steps,
            ),
        )
        kinetic_spectral_upper_bound = 4.0 * sum(
            1.0 / spacing**2 for spacing in topology.plan.level_spacings[-1]
        )
        self.plan = plan
        self.physics = physics
        self.fd_hierarchy = fd_hierarchy
        self.topology = topology
        self.background = background
        self.layout = layout
        self.diffusion = diffusion
        self.kinetic_operator = kinetic
        self.gravity = gravity
        self.solve_policy = solve_policy
        self.kinetic_spectral_upper_bound = kinetic_spectral_upper_bound
        self.coordinate_convention = "flat-periodic-comoving-cartesian"
        self.potential_convention = (
            "phi=a*Phi; composite laplacian(phi)=4*pi*G*(rho_c-volume-mean)"
        )
        self.time_level_convention = "all-levels-synchronized-scale-factor-endpoints"
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-wave-amr",
                "plan": plan.plan_id,
                "physics": physics.plan_id,
                "epoch": topology.epoch.epoch_id,
                "topology": topology.topology_id,
                "partition": topology.partition_id,
                "layout": layout.layout_id,
                "kinetic": kinetic.operator_id,
                "gravity": gravity.plan_id,
                "coordinate_convention": self.coordinate_convention,
                "potential_convention": self.potential_convention,
                "time_level_convention": self.time_level_convention,
                "kinetic_spectral_upper_bound": kinetic_spectral_upper_bound,
                "profile": "synchronized-levels-no-subcycling",
            }
        )

    @property
    def real_dtype(self) -> np.dtype:
        return self.layout.real_dtype

    def _hierarchy(self, values: Sequence[ArrayLike], /) -> BlockHierarchyState:
        arrays = self.layout.zero_masked(
            tuple(jnp.asarray(value, dtype=self.physics.dtype) for value in values)
        )
        return BlockHierarchyState(
            self.topology,
            tuple(
                BlockLevelState(plan, metadata, value)
                for plan, metadata, value in zip(
                    self.topology.plan.levels,
                    self.topology.levels,
                    arrays,
                    strict=True,
                )
            ),
        )

    def _validate_state(self, state: WaveAMRState, /) -> WaveAMRState:
        if not isinstance(state, WaveAMRState):
            raise TypeError("state must be WaveAMRState.")
        values = self.layout.bind_state(state.psi)
        scale = jnp.asarray(state.scale_factor, dtype=self.real_dtype)
        accepted_boundary = jnp.asarray(state.accepted_boundary, dtype=bool)
        if scale.shape != () or accepted_boundary.shape != ():
            raise ValueError("Wave AMR scale and accepted-boundary flag must be scalar.")
        matrix = self.layout.flatten_cells(values)
        masked_nonzero = jnp.any(
            jnp.where(self.layout.flat_leaf_mask[:, None], 0.0, matrix) != 0.0
        )
        probability = self.layout.probability(values)
        checked = list(values)
        checked[0] = eqx.error_if(
            checked[0],
            ~jnp.all(jnp.isfinite(matrix))
            | ~jnp.isfinite(scale)
            | (scale <= 0.0)
            | (probability <= 0.0)
            | masked_nonzero,
            "Wave AMR state must be finite, positive in norm/time, and zero on "
            "inactive or covered composite storage.",
        )
        return WaveAMRState(self._hierarchy(tuple(checked)), scale, accepted_boundary)

    def initialize(
        self,
        psi: BlockHierarchyState | Sequence[ArrayLike],
        scale_factor: ArrayLike,
        /,
    ) -> WaveAMRState:
        hierarchy = psi if isinstance(psi, BlockHierarchyState) else self._hierarchy(psi)
        return self._validate_state(
            WaveAMRState(hierarchy, jnp.asarray(scale_factor), jnp.asarray(True))
        )

    def density(self, state: WaveAMRState, /) -> tuple[Array, ...]:
        checked = self._validate_state(state)
        values = self.layout.bind_state(checked.psi)
        return tuple(self.physics.boson_mass * jnp.abs(value) ** 2 for value in values)

    def probability(self, state: WaveAMRState, /) -> Array:
        checked = self._validate_state(state)
        return self.layout.probability(self.layout.bind_state(checked.psi))

    def _integrated_current(self, values: tuple[Array, ...], /) -> Array:
        matrix = self.layout.flatten_cells(values)[:, 0]
        routes = self.diffusion.plan.routes
        distance = routes.edge_left_distance + routes.edge_right_distance
        face = (
            self.physics.reduced_planck_constant
            / self.physics.boson_mass
            * jnp.imag(jnp.conj(matrix[routes.edge_left]) * matrix[routes.edge_right])
            / distance
        )
        dimension = len(self.topology.plan.grid.shape)
        return jnp.stack(
            tuple(
                jnp.sum(
                    jnp.where(
                        routes.edge_axis == axis,
                        face * routes.edge_area * distance,
                        0.0,
                    )
                )
                for axis in range(dimension)
            )
        )

    def _phase_winding(self, values: tuple[Array, ...], /) -> Array:
        matrix = self.layout.flatten_cells(values)[:, 0]
        routes = self.diffusion.plan.routes
        phase_jump = jnp.angle(
            jnp.conj(matrix[routes.edge_left]) * matrix[routes.edge_right]
        )
        lengths = tuple(
            float(np.asarray(axis.bounds[1] - axis.bounds[0]))
            for axis in self.topology.plan.grid.structured_axes
        )
        volume = prod(lengths)
        return jnp.stack(
            tuple(
                jnp.sum(
                    jnp.where(
                        routes.edge_axis == axis,
                        routes.edge_area * phase_jump,
                        0.0,
                    )
                )
                / (2.0 * pi * volume / lengths[axis])
                for axis in range(len(lengths))
            )
        )

    def _self_adjoint_residual(self, values: tuple[Array, ...], /) -> Array:
        probe = tuple(
            value
            * (
                1.0
                + jnp.arange(value.size, dtype=self.real_dtype).reshape(value.shape)
                / max(1, value.size)
            )
            for value in values
        )
        left = self.layout.space.inner(values, self.kinetic_operator.mv(probe))
        right = self.layout.space.inner(self.kinetic_operator.mv(values), probe)
        scale = jnp.maximum(jnp.maximum(jnp.abs(left), jnp.abs(right)), 1.0)
        return jnp.abs(left - right) / scale

    def _potential_kick(
        self,
        values: tuple[Array, ...],
        potential: tuple[Array, ...],
        kick_factor: Array,
        fraction: float,
        /,
    ) -> tuple[tuple[Array, ...], Array]:
        coefficient = (
            fraction
            * self.physics.boson_mass
            * kick_factor
            / self.physics.reduced_planck_constant
        )
        phases = tuple(coefficient * value for value in potential)
        updated = tuple(
            jnp.where(mask, psi * jnp.exp(-1j * phase), 0.0)
            for psi, phase, mask in zip(
                values, phases, self.layout.leaf_mask, strict=True
            )
        )
        maximum = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.where(mask, jnp.abs(phase), 0.0))
                    for phase, mask in zip(phases, self.layout.leaf_mask, strict=True)
                )
            )
        )
        return updated, maximum

    def _kinetic_drift(
        self,
        values: tuple[Array, ...],
        drift_factor: Array,
        /,
    ) -> tuple[tuple[Array, ...], LinearSolveResult, Array, Array]:
        action = (
            self.physics.reduced_planck_constant
            * drift_factor
            / (2.0 * self.physics.boson_mass)
        )
        alpha = 0.5 * action
        identity = IdentityLinearOperator(self.layout.space)
        left = identity + (1j * alpha) * self.kinetic_operator
        right = identity + (-1j * alpha) * self.kinetic_operator
        right_hand_side = right.mv(values)
        solved = solve(LinearSystem(left), right_hand_side, policy=self.solve_policy)
        candidate = self.layout.zero_masked(solved.value)
        residual = tuple(
            image - source
            for image, source in zip(left.mv(candidate), right_hand_side, strict=True)
        )
        residual_norm = jnp.sqrt(jnp.real(self.layout.space.inner(residual, residual)))
        rhs_norm = jnp.sqrt(
            jnp.real(self.layout.space.inner(right_hand_side, right_hand_side))
        )
        relative = residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
        converged = residual_norm <= jnp.maximum(
            self.plan.solve_absolute_tolerance,
            self.plan.solve_relative_tolerance * rhs_norm,
        )
        return candidate, solved, relative, converged

    def step(
        self,
        state: WaveAMRState,
        end_scale_factor: ArrayLike,
        /,
    ) -> WaveAMRResult:
        """Commit one synchronized fixed-topology interval transactionally."""
        initial = self._validate_state(state)
        end = jnp.asarray(end_scale_factor, dtype=self.real_dtype)
        if end.shape != ():
            raise ValueError("End scale factor must be scalar.")
        end = eqx.error_if(
            end,
            ~jnp.isfinite(end) | (end <= initial.scale_factor),
            "Wave AMR end scale factor must be finite and greater than the current level.",
        )
        values = self.layout.bind_state(initial.psi)
        initial_probability = self.layout.probability(values)
        initial_current = self._integrated_current(values)
        first_gravity = self.gravity.solve_density(self.density(initial))
        kick = self.background.kick_factor(initial.scale_factor, end).astype(
            self.real_dtype
        )
        drift = self.background.drift_factor(initial.scale_factor, end).astype(
            self.real_dtype
        )
        first, first_phase = self._potential_kick(
            values,
            first_gravity.potential,
            kick,
            0.5,
        )
        kinetic, solved, cayley_residual, cayley_closed = self._kinetic_drift(
            first, drift
        )
        drifted_density = tuple(
            self.physics.boson_mass * jnp.abs(value) ** 2 for value in kinetic
        )
        second_gravity = self.gravity.solve_density(drifted_density)
        candidate_values, second_phase = self._potential_kick(
            kinetic,
            second_gravity.potential,
            kick,
            0.5,
        )
        candidate_hierarchy = self._hierarchy(candidate_values)
        candidate = WaveAMRState(candidate_hierarchy, end, jnp.asarray(True))
        final_probability = self.layout.probability(candidate_values)
        probability_error = (
            jnp.abs(final_probability - initial_probability) / initial_probability
        )
        final_current = self._integrated_current(candidate_values)
        current_delta = final_current - initial_current
        current_absolute_defect = jnp.sqrt(
            ein.contract("i,i->", current_delta, current_delta)
        )
        initial_current_norm = jnp.sqrt(
            ein.contract("i,i->", initial_current, initial_current)
        )
        current_defect = jnp.where(
            initial_current_norm > 32.0 * jnp.finfo(self.real_dtype).eps,
            current_absolute_defect / initial_current_norm,
            0.0,
        )
        self_adjoint_residual = self._self_adjoint_residual(values)
        action = (
            self.physics.reduced_planck_constant * drift / (2.0 * self.physics.boson_mass)
        )
        kinetic_phase = 2.0 * jnp.arctan(
            0.5 * jnp.abs(action) * self.kinetic_spectral_upper_bound
        )
        potential_phase = jnp.maximum(first_phase, second_phase)
        maximum_phase = jnp.maximum(kinetic_phase, potential_phase)
        kinetic_energy = (
            self.physics.reduced_planck_constant**2
            / (2.0 * self.physics.boson_mass * end**2)
            * jnp.real(
                self.layout.space.inner(
                    candidate_values,
                    self.kinetic_operator.mv(candidate_values),
                )
            )
        )
        potential_energy = (
            0.5
            / end
            * jnp.real(
                sum(
                    jnp.sum(
                        mask
                        * self.layout.real_layout.pairing_weights[index]
                        * second_gravity.potential[index]
                        * self.physics.boson_mass
                        * jnp.abs(candidate_values[index]) ** 2
                    )
                    for index, mask in enumerate(self.layout.leaf_mask)
                )
            )
        )
        finite = (
            jnp.all(jnp.isfinite(final_probability))
            & jnp.all(jnp.isfinite(final_current))
            & jnp.isfinite(current_absolute_defect)
            & jnp.isfinite(current_defect)
            & jnp.isfinite(cayley_residual)
            & jnp.isfinite(self_adjoint_residual)
            & jnp.isfinite(kinetic_phase)
            & jnp.isfinite(kinetic_energy)
            & jnp.isfinite(potential_energy)
        )
        initial_poisson_closed = first_gravity.successful
        final_poisson_closed = second_gravity.successful
        poisson_closed = initial_poisson_closed & final_poisson_closed
        kinetic_closed = solved.successful & jnp.all(solved.diagnostics.converged)
        self_adjoint = self_adjoint_residual <= self.plan.self_adjoint_tolerance
        norm_closed = probability_error <= self.plan.norm_relative_tolerance
        phase_resolved = maximum_phase <= self.plan.maximum_phase_radians
        accepted = (
            finite
            & poisson_closed
            & kinetic_closed
            & cayley_closed
            & self_adjoint
            & norm_closed
            & phase_resolved
        )
        status = jnp.where(
            ~finite,
            1,
            jnp.where(
                ~poisson_closed,
                2,
                jnp.where(
                    ~kinetic_closed,
                    3,
                    jnp.where(
                        ~cayley_closed,
                        4,
                        jnp.where(
                            ~self_adjoint,
                            5,
                            jnp.where(
                                ~norm_closed,
                                6,
                                jnp.where(~phase_resolved, 7, 0),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        committed = WaveAMRState(
            self._hierarchy(
                tuple(
                    jnp.where(accepted, candidate_value, initial_value)
                    for candidate_value, initial_value in zip(
                        candidate_values,
                        values,
                        strict=True,
                    )
                )
            ),
            jnp.where(accepted, end, initial.scale_factor),
            jnp.where(accepted, jnp.asarray(True), initial.accepted_boundary),
        )
        diagnostics = WaveAMRDiagnostics(
            initial_probability=initial_probability,
            final_probability=final_probability,
            probability_relative_error=probability_error,
            initial_current=initial_current,
            final_current=final_current,
            current_relative_defect=current_defect,
            current_absolute_defect=current_absolute_defect,
            cayley_relative_residual=cayley_residual,
            self_adjoint_residual=self_adjoint_residual,
            maximum_kinetic_phase=kinetic_phase,
            maximum_potential_phase=potential_phase,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            finite=finite,
            initial_poisson_closed=initial_poisson_closed,
            final_poisson_closed=final_poisson_closed,
            poisson_closed=poisson_closed,
            kinetic_closed=kinetic_closed,
            accepted=accepted,
            status=status,
        )
        return WaveAMRResult(
            state=committed,
            candidate_state=candidate,
            diagnostics=diagnostics,
            gravity=first_gravity,
            second_gravity=second_gravity,
            kinetic_solve=solved,
            successful=accepted,
            prepared_id=self.prepared_id,
        )

    def _indicator_levels(self, state: WaveAMRState, /) -> WaveAMRTopologyIndicators:
        adaptivity = self.plan.adaptivity
        if adaptivity is None:
            raise ValueError("This Wave AMR plan has no adaptive topology policy.")
        checked = self._validate_state(state)
        filled = complex_amr_fill_patch(self.fd_hierarchy, checked.psi)
        workspaces = filled.require_complete()
        probability = self.layout.probability(self.layout.bind_state(checked.psi))
        volume = jnp.sum(
            jnp.where(self.layout.flat_leaf_mask, self.layout.cell_measures, 0.0)
        )
        mean_density = self.physics.boson_mass * probability / volume
        de_broglie_levels = []
        phase_levels = []
        contrast_levels = []
        quantum_levels = []
        vortex_levels = []
        node_levels = []
        maxima = []
        finite = jnp.asarray(True)
        for level_index, (workspace, level_plan, spacing, metadata) in enumerate(
            zip(
                workspaces,
                self.topology.plan.levels,
                self.topology.plan.level_spacings,
                self.topology.levels,
                strict=True,
            )
        ):
            halo = level_plan.halo_width
            interior = tuple(
                slice(width, width + size)
                for width, size in zip(halo, level_plan.block_shape, strict=True)
            )
            psi = workspace.values[(slice(None),) + interior]
            amplitude_squared = jnp.abs(psi) ** 2
            maximum = jnp.max(amplitude_squared)
            node = amplitude_squared <= adaptivity.relative_node_floor * maximum
            safe = jnp.where(node, 1.0, amplitude_squared)
            wave_squared = jnp.zeros_like(amplitude_squared)
            maximum_phase_change = jnp.zeros_like(amplitude_squared)
            laplacian_amplitude = jnp.zeros_like(amplitude_squared)
            amplitude = jnp.sqrt(amplitude_squared)
            for axis, dx in enumerate(spacing):
                padded_axis = axis + 1
                lower_index = [slice(None)] + list(interior)
                upper_index = [slice(None)] + list(interior)
                lower_index[padded_axis] = slice(
                    halo[axis] - 1, halo[axis] - 1 + level_plan.block_shape[axis]
                )
                upper_index[padded_axis] = slice(
                    halo[axis] + 1, halo[axis] + 1 + level_plan.block_shape[axis]
                )
                lower = workspace.values[tuple(lower_index)]
                upper = workspace.values[tuple(upper_index)]
                derivative = (upper - lower) / (2.0 * dx)
                component = jnp.imag(jnp.conj(psi) * derivative) / safe
                wave_squared = wave_squared + jnp.where(node, 0.0, component**2)
                lower_phase = jnp.angle(jnp.conj(psi) * lower)
                upper_phase = jnp.angle(jnp.conj(psi) * upper)
                maximum_phase_change = jnp.maximum(
                    maximum_phase_change,
                    jnp.maximum(jnp.abs(lower_phase), jnp.abs(upper_phase)),
                )
                laplacian_amplitude = (
                    laplacian_amplitude
                    + (jnp.abs(upper) - 2.0 * amplitude + jnp.abs(lower)) / dx**2
                )
            minimum_dx = min(spacing)
            de_broglie = jnp.sqrt(wave_squared) * minimum_dx
            contrast = jnp.abs(
                self.physics.boson_mass * amplitude_squared / mean_density - 1.0
            )
            quantum = (
                minimum_dx**2
                * jnp.abs(laplacian_amplitude)
                / jnp.where(amplitude > 0.0, amplitude, 1.0)
            )
            vortex = jnp.zeros_like(amplitude_squared)

            def offset_value(first: int, second: int, /) -> Array:
                slices = [slice(None)]
                for axis, (width, size) in enumerate(
                    zip(halo, level_plan.block_shape, strict=True)
                ):
                    offset = int(axis == first) + int(axis == second)
                    slices.append(slice(width + offset, width + offset + size))
                return workspace.values[tuple(slices)]

            for first_axis in range(len(spacing)):
                for second_axis in range(first_axis + 1, len(spacing)):
                    first = offset_value(first_axis, -1)
                    second = offset_value(second_axis, -1)
                    diagonal = offset_value(first_axis, second_axis)
                    circulation = (
                        jnp.angle(jnp.conj(psi) * first)
                        + jnp.angle(jnp.conj(first) * diagonal)
                        + jnp.angle(jnp.conj(diagonal) * second)
                        + jnp.angle(jnp.conj(second) * psi)
                    )
                    vortex = jnp.maximum(vortex, jnp.abs(circulation))
            active = metadata.active.reshape(
                (level_plan.maximum_blocks,) + (1,) * len(level_plan.block_shape)
            )
            de_broglie = jnp.where(active, de_broglie, 0.0)
            maximum_phase_change = jnp.where(active, maximum_phase_change, 0.0)
            contrast = jnp.where(active, contrast, 0.0)
            quantum = jnp.where(active, quantum, 0.0)
            vortex = jnp.where(active, vortex, 0.0)
            node = jnp.where(active, node, False)
            values = (de_broglie, maximum_phase_change, contrast, quantum, vortex)
            finite = finite & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
            )
            maxima.append(jnp.stack(tuple(jnp.max(value) for value in values)))
            de_broglie_levels.append(de_broglie)
            phase_levels.append(maximum_phase_change)
            contrast_levels.append(contrast)
            quantum_levels.append(quantum)
            vortex_levels.append(vortex)
            node_levels.append(node)
        return WaveAMRTopologyIndicators(
            tuple(de_broglie_levels),
            tuple(phase_levels),
            tuple(contrast_levels),
            tuple(quantum_levels),
            tuple(vortex_levels),
            tuple(node_levels),
            jnp.stack(tuple(maxima)),
            finite & filled.complete,
        )

    def propose_topology(self, state: WaveAMRState, /) -> WaveAMRTopologyProposal:
        """Compile tags only at an accepted step boundary; never mutate the epoch."""
        adaptivity = self.plan.adaptivity
        if adaptivity is None:
            raise ValueError("This Wave AMR plan has no adaptive topology policy.")
        checked = self._validate_state(state)
        accepted_boundary = bool(np.asarray(checked.accepted_boundary))
        if not accepted_boundary:
            raise ValueError(
                "Wave AMR topology changes require an accepted step boundary."
            )
        indicators = self._indicator_levels(checked)
        tags = []
        for level in range(len(self.topology.plan.levels) - 1):
            covered = np.asarray(self.topology.covered_cells[level], dtype=bool)
            hysteresis = np.where(
                covered,
                adaptivity.coarsening_hysteresis,
                1.0,
            )
            selected = (
                (
                    np.asarray(indicators.de_broglie[level])
                    > hysteresis / adaptivity.minimum_de_broglie_cells
                )
                | (
                    np.asarray(indicators.phase_change[level])
                    > hysteresis * adaptivity.maximum_phase_change
                )
                | (
                    np.asarray(indicators.density_contrast[level])
                    > hysteresis * adaptivity.maximum_density_contrast
                )
                | (
                    np.asarray(indicators.quantum_potential[level])
                    > hysteresis * adaptivity.maximum_quantum_potential_indicator
                )
                | (
                    np.asarray(indicators.vortex[level])
                    > hysteresis * adaptivity.vortex_threshold
                )
            )
            active = np.asarray(self.topology.levels[level].active, dtype=bool)
            selected &= active.reshape(
                (active.size,) + (1,) * len(self.topology.plan.levels[level].block_shape)
            )
            tags.append(jnp.asarray(selected))
        compilation = self.fd_hierarchy.compile_topology(self.topology, tuple(tags))
        successful = bool(compilation.status.successful) and bool(
            np.asarray(indicators.finite)
        )
        return WaveAMRTopologyProposal(
            tuple(tags),
            indicators,
            compilation,
            self.prepared_id,
            self.topology.epoch.epoch_id,
            self.topology.topology_id,
            accepted_boundary,
            successful,
            canonical_fingerprint(
                {
                    "kind": "wave-amr-topology-proposal",
                    "prepared": self.prepared_id,
                    "compilation": compilation.result_id,
                    "accepted_boundary": accepted_boundary,
                    "indicators_finite": bool(np.asarray(indicators.finite)),
                }
            ),
        )

    def _leaf_vortex_count(self, values: tuple[Array, ...], /) -> Array:
        dimension = len(self.topology.plan.grid.shape)
        if dimension < 2:
            return jnp.asarray(0, dtype=jnp.int32)
        hierarchy = self._hierarchy(values)
        workspaces = complex_amr_fill_patch(
            self.fd_hierarchy,
            hierarchy,
        ).require_complete()
        count = jnp.asarray(0, dtype=jnp.int32)
        for workspace, level_plan, mask in zip(
            workspaces,
            self.topology.plan.levels,
            self.layout.leaf_mask,
            strict=True,
        ):
            halo = level_plan.halo_width

            def point(first: int, second: int, /) -> Array:
                slices = [slice(None)]
                for axis, (width, size) in enumerate(
                    zip(halo, level_plan.block_shape, strict=True)
                ):
                    offset = int(axis == first) + int(axis == second)
                    slices.append(slice(width + offset, width + offset + size))
                return workspace.values[tuple(slices)]

            center = point(-1, -1)
            for first_axis in range(dimension):
                for second_axis in range(first_axis + 1, dimension):
                    first = point(first_axis, -1)
                    second = point(second_axis, -1)
                    diagonal = point(first_axis, second_axis)
                    circulation = (
                        jnp.angle(jnp.conj(center) * first)
                        + jnp.angle(jnp.conj(first) * diagonal)
                        + jnp.angle(jnp.conj(diagonal) * second)
                        + jnp.angle(jnp.conj(second) * center)
                    )
                    count = count + jnp.sum(mask & (jnp.abs(circulation) >= pi)).astype(
                        jnp.int32
                    )
        return count

    def _leaf_node_count(self, values: tuple[Array, ...], /) -> Array:
        power = jnp.abs(self.layout.flatten_cells(values)[:, 0]) ** 2
        leaf_power = jnp.where(self.layout.flat_leaf_mask, power, 0.0)
        maximum = jnp.max(leaf_power)
        return jnp.sum(
            self.layout.flat_leaf_mask
            & (power <= self.plan.adaptivity.relative_node_floor * maximum)
        ).astype(jnp.int32)

    def _cell_centers(
        self,
        transition: BlockFieldTopologyTransition,
        /,
        *,
        source: bool,
    ) -> np.ndarray:
        topology = transition.source_topology if source else transition.target_topology
        levels = np.asarray(
            transition.source_levels if source else transition.target_levels
        )
        slots = np.asarray(transition.source_slots if source else transition.target_slots)
        local = np.asarray(transition.source_local if source else transition.target_local)
        bounds = np.asarray(
            [np.asarray(axis.bounds) for axis in topology.plan.grid.structured_axes]
        )
        result = []
        for level_value, slot, flat in zip(levels, slots, local, strict=True):
            level = int(level_value)
            plan = topology.plan.levels[level]
            logical = np.asarray(topology.levels[level].logical_indices)[int(slot)]
            local_index = np.unravel_index(int(flat), plan.block_shape)
            global_index = logical * np.asarray(plan.block_shape) + np.asarray(
                local_index
            )
            spacing = np.asarray(topology.plan.level_spacings[level])
            result.append(bounds[:, 0] + (global_index + 0.5) * spacing)
        return np.asarray(result)

    def transition(
        self,
        state: WaveAMRState,
        proposal: WaveAMRTopologyProposal,
        /,
    ) -> WaveAMRTransitionResult:
        """Phase-aware probability transfer with current/winding rejection."""
        checked = self._validate_state(state)
        if not isinstance(proposal, WaveAMRTopologyProposal):
            raise TypeError("proposal must be WaveAMRTopologyProposal.")
        if not bool(np.asarray(checked.accepted_boundary)):
            raise ValueError("Wave AMR regrid requires an accepted-boundary state.")
        if (
            not proposal.accepted_boundary
            or proposal.source_prepared_id != self.prepared_id
            or proposal.source_epoch_id != self.topology.epoch.epoch_id
            or proposal.source_topology_id != self.topology.topology_id
        ):
            raise ValueError(
                "Wave AMR proposal does not bind this accepted source prepared epoch."
            )
        if not proposal.successful or not proposal.compilation.status.successful:
            raise ValueError("Cannot transition an unsuccessful Wave AMR proposal.")
        if not proposal.compilation.status.changed:
            values = self.layout.bind_state(checked.psi)
            probability = self.layout.probability(values)
            current = self._integrated_current(values)
            winding = self._phase_winding(values)
            nodes = self._leaf_node_count(values)
            vortices = self._leaf_vortex_count(values)
            evidence = WaveAMRTransferEvidence(
                probability,
                probability,
                jnp.asarray(0.0),
                current,
                current,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                winding,
                winding,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                nodes,
                nodes,
                vortices,
                vortices,
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(True),
                canonical_fingerprint(
                    {
                        "kind": "wave-amr-unchanged-transition",
                        "proposal": proposal.proposal_id,
                    }
                ),
            )
            return WaveAMRTransitionResult(
                checked,
                checked,
                self,
                self,
                evidence,
                jnp.asarray(True),
                jnp.asarray(False),
                proposal.proposal_id,
            )
        target = proposal.compilation.topology
        target_prepared = self.plan.prepare(self.physics, target, self.background)
        source_values = self.layout.bind_state(checked.psi)
        density_state = BlockHierarchyState(
            self.topology,
            tuple(
                BlockLevelState(level.plan, level.metadata, jnp.abs(level.values) ** 2)
                for level in checked.psi.levels
            ),
        )
        density_transfer = self.fd_hierarchy.field_transition(
            self.topology,
            target,
            "wave-probability-density",
            dtype=self.real_dtype,
        )
        phase_routes = density_transfer
        density_result = density_transfer.apply(density_state)
        source_centers = self._cell_centers(phase_routes, source=True)
        target_centers = self._cell_centers(phase_routes, source=False)
        bounds = np.asarray(
            [np.asarray(axis.bounds) for axis in self.topology.plan.grid.structured_axes],
            dtype=float,
        )
        lengths = bounds[:, 1] - bounds[:, 0]
        dimension = source_centers.shape[1]
        source_levels_host = np.asarray(phase_routes.source_levels)
        source_slots_host = np.asarray(phase_routes.source_slots)
        source_local_host = np.asarray(phase_routes.source_local)
        source_storage = tuple(
            self.layout.real_layout.cell_offsets[int(level)]
            + int(slot) * prod(self.topology.plan.levels[int(level)].block_shape)
            + int(local)
            for level, slot, local in zip(
                source_levels_host,
                source_slots_host,
                source_local_host,
                strict=True,
            )
        )
        source_by_storage = {
            storage: ordinal for ordinal, storage in enumerate(source_storage)
        }
        neighbor_plus = np.broadcast_to(
            np.arange(source_centers.shape[0], dtype=np.int32)[:, None],
            (source_centers.shape[0], dimension),
        ).copy()
        neighbor_minus = neighbor_plus.copy()
        plus_area = np.full(neighbor_plus.shape, -np.inf)
        minus_area = np.full(neighbor_minus.shape, -np.inf)
        routes = self.diffusion.plan.routes
        for left_storage, right_storage, axis, area in zip(
            np.asarray(routes.edge_left),
            np.asarray(routes.edge_right),
            np.asarray(routes.edge_axis),
            np.asarray(routes.edge_area),
            strict=True,
        ):
            left = source_by_storage.get(int(left_storage))
            right = source_by_storage.get(int(right_storage))
            axis_index = int(axis)
            area_value = float(area)
            if left is not None and right is not None:
                if area_value > plus_area[left, axis_index]:
                    neighbor_plus[left, axis_index] = right
                    plus_area[left, axis_index] = area_value
                if area_value > minus_area[right, axis_index]:
                    neighbor_minus[right, axis_index] = left
                    minus_area[right, axis_index] = area_value
        if np.any(~np.isfinite(plus_area)) or np.any(~np.isfinite(minus_area)):
            raise RuntimeError(
                "Periodic composite AMR phase transfer lacks a complete face neighbor."
            )
        target_levels_host = np.asarray(phase_routes.target_levels)
        target_slots_host = np.asarray(phase_routes.target_slots)
        target_local_host = np.asarray(phase_routes.target_local)
        target_row_leaf = tuple(
            bool(
                np.asarray(target_prepared.layout.leaf_mask[int(level)])[
                    int(slot)
                ].reshape((-1,))[int(local)]
            )
            for level, slot, local in zip(
                target_levels_host,
                target_slots_host,
                target_local_host,
                strict=True,
            )
        )
        leaf_storage_rows = np.flatnonzero(np.asarray(target_row_leaf, dtype=bool))
        nearest = np.zeros((target_centers.shape[0],), dtype=np.int32)
        best_weight = np.full((leaf_storage_rows.size,), -np.inf)
        leaf_relation = phase_routes.leaf_routes.relation
        for source_index, target_index, weight, valid in zip(
            np.asarray(leaf_relation.source_indices),
            np.asarray(leaf_relation.target_indices),
            np.asarray(phase_routes.leaf_routes.weights),
            np.asarray(leaf_relation.valid),
            strict=True,
        ):
            target_index_ = int(target_index)
            if valid and float(weight) > best_weight[target_index_]:
                nearest[int(leaf_storage_rows[target_index_])] = int(source_index)
                best_weight[target_index_] = float(weight)
        if np.any(~np.isfinite(best_weight)):
            raise RuntimeError(
                "AMR phase transfer lacks a source overlap for a target leaf cell."
            )
        target_displacement = target_centers - source_centers[nearest]
        target_displacement -= np.round(target_displacement / lengths) * lengths
        source_flat = []
        for level, slot, local in zip(
            source_levels_host,
            source_slots_host,
            source_local_host,
            strict=True,
        ):
            block = source_values[int(level)][int(slot)]
            source_flat.append(block.reshape((-1,))[int(local)])
        source_flat_array = jnp.stack(source_flat)
        overlap_weights = jnp.where(
            leaf_relation.valid,
            phase_routes.leaf_routes.weights.astype(self.real_dtype),
            0.0,
        )
        overlap_source = source_flat_array[leaf_relation.source_indices]
        reference_leaf_real = (
            jnp.zeros(
                (leaf_storage_rows.size,),
                dtype=self.real_dtype,
            )
            .at[leaf_relation.target_indices]
            .add(overlap_weights * jnp.real(overlap_source))
        )
        reference_leaf_imaginary = (
            jnp.zeros(
                (leaf_storage_rows.size,),
                dtype=self.real_dtype,
            )
            .at[leaf_relation.target_indices]
            .add(overlap_weights * jnp.imag(overlap_source))
        )
        reference_leaf = jax.lax.complex(
            reference_leaf_real,
            reference_leaf_imaginary,
        )
        reference = (
            jnp.zeros(
                (target_centers.shape[0],),
                dtype=self.physics.dtype,
            )
            .at[jnp.asarray(leaf_storage_rows, dtype=jnp.int32)]
            .set(reference_leaf)
        )
        reference = jax.lax.stop_gradient(reference)
        source_phase = jnp.angle(source_flat_array)
        plus_indices = jnp.asarray(neighbor_plus, dtype=jnp.int32)
        minus_indices = jnp.asarray(neighbor_minus, dtype=jnp.int32)
        plus_phase = source_phase[plus_indices]
        minus_phase = source_phase[minus_indices]
        phase_difference = jnp.angle(jnp.exp(1j * (plus_phase - minus_phase)))
        coordinate_difference = np.stack(
            tuple(
                (
                    source_centers[neighbor_plus[:, axis], axis]
                    - source_centers[:, axis]
                    - np.round(
                        (
                            source_centers[neighbor_plus[:, axis], axis]
                            - source_centers[:, axis]
                        )
                        / lengths[axis]
                    )
                    * lengths[axis]
                )
                - (
                    source_centers[neighbor_minus[:, axis], axis]
                    - source_centers[:, axis]
                    - np.round(
                        (
                            source_centers[neighbor_minus[:, axis], axis]
                            - source_centers[:, axis]
                        )
                        / lengths[axis]
                    )
                    * lengths[axis]
                )
                for axis in range(dimension)
            ),
            axis=1,
        )
        coordinate_difference_array = jnp.asarray(
            coordinate_difference,
            dtype=self.real_dtype,
        )
        safe_coordinate = jnp.where(
            coordinate_difference_array != 0.0,
            coordinate_difference_array,
            1.0,
        )
        phase_gradient = phase_difference / safe_coordinate
        nearest_indices = jnp.asarray(nearest, dtype=jnp.int32)
        target_phase = source_phase[nearest_indices] + jnp.sum(
            phase_gradient[nearest_indices]
            * jnp.asarray(target_displacement, dtype=self.real_dtype),
            axis=1,
        )
        target_density_flat = []
        for level, slot, local in zip(
            target_levels_host,
            target_slots_host,
            target_local_host,
            strict=True,
        ):
            level_index = int(level)
            slot_index = int(slot)
            local_index = int(local)
            density_block = density_result.state.levels[level_index].values[slot_index]
            target_density_flat.append(density_block.reshape((-1,))[local_index])
        target_density = jnp.stack(target_density_flat)
        target_leaf = jnp.asarray(target_row_leaf)
        negative_density = jnp.any(target_leaf & (target_density < 0.0))
        physical_density = jnp.where(target_leaf, target_density, 0.0)
        maximum_density = jnp.max(physical_density)
        occupied = target_leaf & (
            target_density > self.plan.adaptivity.relative_node_floor * maximum_density
        )
        density_phase_candidate = jnp.sqrt(
            jnp.where(physical_density >= 0.0, physical_density, jnp.nan)
        ).astype(self.physics.dtype) * jnp.exp(1j * target_phase)
        reference_phase = jnp.angle(reference)
        phase_difference_target = jnp.abs(
            jnp.angle(jnp.exp(1j * (target_phase - reference_phase)))
        )
        complex_overlap_branch = (~occupied) | (phase_difference_target >= 0.5 * pi)
        target_flat = jnp.where(
            target_leaf,
            jnp.where(
                complex_overlap_branch,
                reference,
                density_phase_candidate,
            ),
            0.0,
        )
        phase_defect = jnp.max(jnp.where(occupied, phase_difference_target / pi, 0.0))
        target_arrays = [
            jnp.zeros(shape, dtype=self.physics.dtype)
            for shape in target_prepared.layout.level_shapes
        ]
        for row, (level, slot, local) in enumerate(
            zip(
                target_levels_host,
                target_slots_host,
                target_local_host,
                strict=True,
            )
        ):
            level_index = int(level)
            block_shape = target.plan.levels[level_index].block_shape
            local_index = np.unravel_index(int(local), block_shape)
            target_arrays[level_index] = (
                target_arrays[level_index]
                .at[(int(slot),) + local_index]
                .set(target_flat[row])
            )
        target_arrays_tuple = target_prepared.layout.zero_masked(tuple(target_arrays))
        candidate = WaveAMRState(
            target_prepared._hierarchy(target_arrays_tuple),
            checked.scale_factor,
            jnp.asarray(True),
        )
        source_probability = self.layout.probability(source_values)
        target_probability = target_prepared.layout.probability(target_arrays_tuple)
        probability_defect = (
            jnp.abs(target_probability - source_probability) / source_probability
        )
        source_current = self._integrated_current(source_values)
        target_current = target_prepared._integrated_current(target_arrays_tuple)
        source_current_norm = jnp.sqrt(
            ein.contract("i,i->", source_current, source_current)
        )
        current_delta = target_current - source_current
        current_absolute_defect = jnp.sqrt(
            ein.contract("i,i->", current_delta, current_delta)
        )
        current_defect = jnp.where(
            source_current_norm > self.plan.adaptivity.current_absolute_tolerance,
            current_absolute_defect / source_current_norm,
            0.0,
        )
        source_winding = self._phase_winding(source_values)
        target_winding = target_prepared._phase_winding(target_arrays_tuple)
        winding_defect = jnp.max(jnp.abs(target_winding - source_winding))
        source_nodes = self._leaf_node_count(source_values)
        target_nodes = jnp.sum(target_leaf & ~occupied).astype(jnp.int32)
        source_vortices = self._leaf_vortex_count(source_values)
        target_vortices = target_prepared._leaf_vortex_count(target_arrays_tuple)
        finite = (
            ~negative_density
            & density_result.successful
            & jnp.isfinite(probability_defect)
            & jnp.isfinite(current_defect)
            & jnp.isfinite(current_absolute_defect)
            & jnp.all(jnp.isfinite(source_winding))
            & jnp.all(jnp.isfinite(target_winding))
            & jnp.isfinite(winding_defect)
            & jnp.isfinite(phase_defect)
            & jnp.all(jnp.isfinite(target_flat))
        )
        probability_preserved = (
            probability_defect <= self.plan.adaptivity.probability_relative_tolerance
        )
        current_preserved = jnp.where(
            source_current_norm > self.plan.adaptivity.current_absolute_tolerance,
            current_defect <= self.plan.adaptivity.current_relative_tolerance,
            current_absolute_defect <= self.plan.adaptivity.current_absolute_tolerance,
        )
        winding_preserved = (
            winding_defect <= self.plan.adaptivity.winding_absolute_tolerance
        ) & (source_vortices == target_vortices)
        successful = (
            finite
            & probability_preserved
            & current_preserved
            & winding_preserved
            & (phase_defect <= self.plan.adaptivity.phase_defect_tolerance)
        )
        evidence = WaveAMRTransferEvidence(
            source_probability,
            target_probability,
            probability_defect,
            source_current,
            target_current,
            current_defect,
            current_absolute_defect,
            source_winding,
            target_winding,
            winding_defect,
            phase_defect,
            source_nodes,
            target_nodes,
            source_vortices,
            target_vortices,
            finite,
            probability_preserved,
            current_preserved,
            winding_preserved,
            successful,
            canonical_fingerprint(
                {
                    "kind": "wave-amr-phase-aware-transition",
                    "proposal": proposal.proposal_id,
                    "density_transition": density_transfer.transition_id,
                    "phase_transition": phase_routes.transition_id,
                    "derivative": "nondifferentiable-topology-event",
                }
            ),
        )
        committed = candidate if bool(np.asarray(successful)) else checked
        committed_prepared = target_prepared if bool(np.asarray(successful)) else self
        return WaveAMRTransitionResult(
            state=committed,
            candidate_state=candidate,
            prepared=committed_prepared,
            candidate_prepared=target_prepared,
            evidence=evidence,
            successful=successful,
            rolled_back=~successful,
            proposal_id=proposal.proposal_id,
        )

    def transition_distributed(
        self,
        source: WaveAMRDistributedPreparation,
        state: DistributedWaveAMRState,
        proposal: WaveAMRTopologyProposal,
        target: WaveAMRDistributedPreparation,
        /,
        *,
        maximum_bytes: int | None = None,
    ) -> WaveAMRDistributedTransitionResult:
        """Commit one phase-aware topology successor collectively or roll back."""
        adaptivity = self.plan.adaptivity
        if adaptivity is None:
            raise ValueError("This Wave AMR plan has no adaptive topology policy.")
        if (
            not isinstance(source, WaveAMRDistributedPreparation)
            or not isinstance(target, WaveAMRDistributedPreparation)
            or source.execution is None
            or target.execution is None
            or source.hierarchy is None
            or target.hierarchy is None
        ):
            raise TypeError(
                "Distributed topology transition requires source and target executions."
            )
        checked = source.execution.validate_state(state)
        if not bool(np.asarray(checked.accepted_boundary)):
            raise ValueError(
                "Distributed Wave AMR topology transition requires an accepted boundary."
            )
        if (
            not isinstance(proposal, WaveAMRTopologyProposal)
            or not proposal.successful
            or not proposal.compilation.status.successful
            or not proposal.compilation.status.changed
            or proposal.source_prepared_id != self.prepared_id
            or proposal.source_epoch_id != self.topology.epoch.epoch_id
            or proposal.source_topology_id != self.topology.topology_id
            or source.source_prepared_id != self.prepared_id
        ):
            raise ValueError(
                "Distributed topology proposal does not bind the accepted source epoch."
            )
        target_prepared = self.plan.prepare(
            self.physics,
            proposal.compilation.topology,
            self.background,
        )
        if (
            target.source_prepared_id != target_prepared.prepared_id
            or target.topology_id != target_prepared.topology.topology_id
            or not source.executable
            or not target.executable
            or not source.admitted
            or not target.admitted
        ):
            raise ValueError(
                "Distributed topology successor preparation is unavailable or inconsistent."
            )
        budget = (
            min(source.maximum_bytes, target.maximum_bytes)
            if maximum_bytes is None
            else int(maximum_bytes)
        )
        if budget < 0:
            raise ValueError("maximum_bytes must be non-negative.")
        if source.hierarchy.partition.part_count == 1:
            authority_hierarchy = source.hierarchy.unpack(checked.psi)
            authority_state = WaveAMRState(
                authority_hierarchy,
                checked.scale_factor,
                checked.accepted_boundary,
            )
            required = source.required_bytes + target.required_bytes
            if required > budget:
                authority_values = self.layout.bind_state(authority_hierarchy)
                probability = self.layout.probability(authority_values)
                current = self._integrated_current(authority_values)
                winding = self._phase_winding(authority_values)
                node_count = self._leaf_node_count(authority_values)
                vortex_count = self._leaf_vortex_count(authority_values)
                finite = (
                    jnp.isfinite(probability)
                    & jnp.all(jnp.isfinite(current))
                    & jnp.all(jnp.isfinite(winding))
                )
                transition_id = canonical_fingerprint(
                    {
                        "kind": "single-part-wave-amr-topology-capacity-rejection",
                        "source": source.preparation_id,
                        "target": target.preparation_id,
                        "required_bytes": required,
                        "maximum_bytes": budget,
                    }
                )
                evidence = WaveAMRTransferEvidence(
                    probability,
                    probability,
                    jnp.asarray(0.0, dtype=self.real_dtype),
                    current,
                    current,
                    jnp.asarray(0.0, dtype=self.real_dtype),
                    jnp.asarray(0.0, dtype=self.real_dtype),
                    winding,
                    winding,
                    jnp.asarray(0.0, dtype=self.real_dtype),
                    jnp.asarray(jnp.inf, dtype=self.real_dtype),
                    node_count,
                    node_count,
                    vortex_count,
                    vortex_count,
                    finite,
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(False),
                    transition_id,
                )
                transfer = DistributedWaveAMRTopologyTransferResult(
                    checked,
                    jnp.asarray(jnp.inf, dtype=self.real_dtype),
                    jnp.asarray(jnp.inf, dtype=self.real_dtype),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    finite,
                    jnp.asarray(False),
                    transition_id,
                )
                return WaveAMRDistributedTransitionResult(
                    checked,
                    checked,
                    self,
                    source,
                    self,
                    source,
                    evidence,
                    transfer,
                    jnp.asarray(False),
                    jnp.asarray(True),
                    proposal.proposal_id,
                )
            authority = self.transition(authority_state, proposal)
            candidate_values = target.execution.pack_canonical_values(
                target_prepared.layout.bind_state(authority.candidate_state.psi)
            )
            candidate_state = target.execution._bind_unchecked(
                candidate_values,
                target.execution._replicate_scalar(
                    authority.candidate_state.scale_factor,
                    target_prepared.real_dtype,
                ),
                target.execution._replicate_scalar(
                    authority.candidate_state.accepted_boundary,
                    bool,
                ),
            )
            successful = authority.successful
            accepted = bool(np.asarray(successful))
            transfer = DistributedWaveAMRTopologyTransferResult(
                candidate_state,
                authority.evidence.probability_relative_defect,
                authority.evidence.phase_defect,
                jnp.asarray(False),
                jnp.asarray(True),
                authority.evidence.finite,
                successful,
                authority.evidence.transition_id,
            )
            return WaveAMRDistributedTransitionResult(
                candidate_state if accepted else checked,
                candidate_state,
                authority.prepared,
                target if accepted else source,
                target_prepared,
                target,
                authority.evidence,
                transfer,
                successful,
                ~successful,
                proposal.proposal_id,
            )
        density_transition = self.fd_hierarchy.field_transition(
            self.topology,
            target_prepared.topology,
            "wave-probability-density",
            dtype=self.real_dtype,
        )
        owner_estimate = (
            PreparedDistributedWaveAMRTopologyTransition.estimate_required_bytes(
                source.execution,
                target.execution,
                density_transition,
            )
        )
        required = source.required_bytes + target.required_bytes + owner_estimate
        capacity = required <= budget
        owner_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-topology-preflight",
                "source": source.preparation_id,
                "target": target.preparation_id,
                "density_transition": density_transition.transition_id,
                "required_bytes": required,
                "maximum_bytes": budget,
                "admitted": capacity,
            }
        )
        source_observables = source.execution.observables(
            checked,
            relative_node_floor=adaptivity.relative_node_floor,
        )
        if capacity:
            owner = PreparedDistributedWaveAMRTopologyTransition(
                source.execution,
                target.execution,
                density_transition,
            )
            required = (
                source.required_bytes + target.required_bytes + owner.required_bytes
            )
            capacity = required <= budget
            owner_id = owner.transition_id
        if capacity:
            transfer = owner.execute(
                checked,
                relative_node_floor=adaptivity.relative_node_floor,
            )
        else:
            transfer = DistributedWaveAMRTopologyTransferResult(
                checked,
                jnp.asarray(jnp.inf, dtype=self.real_dtype),
                jnp.asarray(jnp.inf, dtype=self.real_dtype),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                canonical_fingerprint(
                    {
                        "kind": "distributed-wave-amr-topology-capacity-rejection",
                        "source": source.preparation_id,
                        "target": target.preparation_id,
                        "preflight": owner_id,
                        "required_bytes": required,
                        "maximum_bytes": budget,
                    }
                ),
            )
        if bool(np.asarray(transfer.successful)):
            target_observables = target.execution.observables(
                transfer.candidate,
                relative_node_floor=adaptivity.relative_node_floor,
            )
        else:
            target_observables = source_observables
        source_probability = source_observables.probability
        target_probability = target_observables.probability
        probability_defect = jnp.abs(target_probability - source_probability) / jnp.where(
            source_probability > 0.0, source_probability, 1.0
        )
        current_delta = target_observables.current - source_observables.current
        current_absolute_defect = jnp.sqrt(
            ein.contract("i,i->", current_delta, current_delta)
        )
        source_current_norm = jnp.sqrt(
            ein.contract(
                "i,i->",
                source_observables.current,
                source_observables.current,
            )
        )
        current_defect = jnp.where(
            source_current_norm > adaptivity.current_absolute_tolerance,
            current_absolute_defect / source_current_norm,
            0.0,
        )
        winding_defect = jnp.max(
            jnp.abs(target_observables.winding - source_observables.winding)
        )
        finite = (
            transfer.finite
            & transfer.route_complete
            & ~transfer.negative_density
            & source_observables.finite
            & target_observables.finite
            & jnp.isfinite(probability_defect)
            & jnp.isfinite(current_defect)
            & jnp.isfinite(current_absolute_defect)
            & jnp.isfinite(winding_defect)
            & jnp.isfinite(transfer.phase_defect)
        )
        probability_preserved = (
            probability_defect <= adaptivity.probability_relative_tolerance
        )
        current_preserved = jnp.where(
            source_current_norm > adaptivity.current_absolute_tolerance,
            current_defect <= adaptivity.current_relative_tolerance,
            current_absolute_defect <= adaptivity.current_absolute_tolerance,
        )
        winding_preserved = (winding_defect <= adaptivity.winding_absolute_tolerance) & (
            source_observables.vortex_count == target_observables.vortex_count
        )
        successful = (
            transfer.successful
            & capacity
            & finite
            & probability_preserved
            & current_preserved
            & winding_preserved
            & (transfer.phase_defect <= adaptivity.phase_defect_tolerance)
        )
        transition_id = canonical_fingerprint(
            {
                "kind": "distributed-wave-amr-phase-aware-transition",
                "proposal": proposal.proposal_id,
                "owner": owner_id,
                "density_transition": density_transition.transition_id,
                "source_preparation": source.preparation_id,
                "target_preparation": target.preparation_id,
                "required_bytes": required,
                "capacity": capacity,
            }
        )
        evidence = WaveAMRTransferEvidence(
            source_probability,
            target_probability,
            probability_defect,
            source_observables.current,
            target_observables.current,
            current_defect,
            current_absolute_defect,
            source_observables.winding,
            target_observables.winding,
            winding_defect,
            transfer.phase_defect,
            source_observables.node_count,
            target_observables.node_count,
            source_observables.vortex_count,
            target_observables.vortex_count,
            finite,
            probability_preserved,
            current_preserved,
            winding_preserved,
            successful,
            transition_id,
        )
        accepted = bool(np.asarray(successful))
        candidate_state = transfer.candidate if capacity else checked
        candidate_prepared = target_prepared if capacity else self
        candidate_distribution = target if capacity else source
        return WaveAMRDistributedTransitionResult(
            transfer.candidate if accepted else checked,
            candidate_state,
            target_prepared if accepted else self,
            target if accepted else source,
            candidate_prepared,
            candidate_distribution,
            evidence,
            transfer,
            successful,
            ~successful,
            proposal.proposal_id,
        )

    def prepare_distributed(
        self,
        partition: BlockAMRPartitionPlan,
        /,
        *,
        maximum_bytes: int,
        execution_group: ExecutionGroup | None = None,
        execution_plan: ExecutionPlan | None = None,
        costs: Sequence[ArrayLike | None] | None = None,
    ) -> WaveAMRDistributedPreparation:
        """Bind one owner-computes execution and reject unsupported resources early."""
        if not isinstance(partition, BlockAMRPartitionPlan):
            raise TypeError("partition must be BlockAMRPartitionPlan.")
        if partition.hierarchy.plan_id != self.topology.plan.plan_id:
            raise ValueError("Distributed partition and Wave AMR topology differ.")
        if execution_group is not None and not isinstance(
            execution_group,
            ExecutionGroup,
        ):
            raise TypeError("execution_group must be ExecutionGroup or None.")
        if execution_plan is not None and not isinstance(execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be ExecutionPlan or None.")
        if (
            execution_group is not None
            and len(execution_group.devices) != partition.part_count
        ):
            raise ValueError(
                "Distributed Wave AMR requires one execution-group device per part."
            )
        if execution_plan is not None and (
            execution_group is None
            or execution_plan.group is None
            or execution_plan.group.group_id != execution_group.spec.group_id
        ):
            raise ValueError(
                "Distributed Wave AMR ExecutionPlan and live ExecutionGroup differ."
            )
        budget = int(maximum_bytes)
        if budget < 0:
            raise ValueError("maximum_bytes must be non-negative.")
        wave_bytes = sum(
            prod(shape) * self.physics.dtype.itemsize
            for shape in self.layout.level_shapes
        )
        cell_entries = sum(
            level.maximum_blocks * prod(level.block_shape)
            for level in self.topology.plan.levels
        )
        halo_entries = sum(
            level.maximum_blocks
            * prod(
                size + 2 * width
                for size, width in zip(
                    level.block_shape,
                    level.halo_width,
                    strict=True,
                )
            )
            for level in self.topology.plan.levels
        )
        metadata_entries = sum(
            level.maximum_blocks * (4 + 2 * len(level.block_shape))
            for level in self.topology.plan.levels
        )
        route_envelope_bytes = (
            128 * partition.part_count * (cell_entries + halo_entries + metadata_entries)
        )
        preflight_required = wave_bytes * 6 + route_envelope_bytes
        execution_plan_id = (
            None if execution_plan is None else execution_plan.plan_fingerprint
        )
        if preflight_required > budget:
            reason = (
                "metadata-only distributed Wave AMR resource preflight exceeded "
                "maximum_bytes before layout or route construction"
            )
            return WaveAMRDistributedPreparation(
                None,
                None,
                False,
                False,
                preflight_required,
                preflight_required,
                budget,
                reason,
                self.prepared_id,
                self.topology.topology_id,
                partition.plan_id,
                execution_plan_id,
                None,
                None,
                self.physics.plan_id,
                canonical_fingerprint(
                    {
                        "kind": "wave-amr-distributed-preflight-rejection",
                        "prepared": self.prepared_id,
                        "partition": partition.plan_id,
                        "execution_plan": execution_plan_id,
                        "wave_bytes": wave_bytes,
                        "route_envelope_bytes": route_envelope_bytes,
                        "required_bytes": preflight_required,
                        "maximum_bytes": budget,
                    }
                ),
            )
        distributed = partition.prepare(
            self.topology,
            self.fd_hierarchy,
            costs=costs,
            execution_group=execution_group,
        )
        execution = PreparedDistributedWaveAMR(
            distributed,
            self.layout,
            self.diffusion,
            execution_plan=execution_plan,
            boson_mass=self.physics.boson_mass,
            gravitational_constant=self.physics.gravitational_constant,
            reduced_planck_constant=self.physics.reduced_planck_constant,
            solve_relative_tolerance=self.plan.solve_relative_tolerance,
            solve_absolute_tolerance=self.plan.solve_absolute_tolerance,
            maximum_solve_steps=self.plan.maximum_solve_steps,
            norm_relative_tolerance=self.plan.norm_relative_tolerance,
            self_adjoint_tolerance=self.plan.self_adjoint_tolerance,
            maximum_phase_radians=self.plan.maximum_phase_radians,
            kinetic_spectral_upper_bound=self.kinetic_spectral_upper_bound,
            source_prepared_id=self.prepared_id,
            physics_id=self.physics.plan_id,
        )
        required = execution.required_bytes
        admitted = required <= budget
        executable = admitted and (partition.part_count == 1 or execution.executable)
        reason = (
            "single-part owner-computes execution uses the local authority"
            if executable and partition.part_count == 1
            else (
                "multi-part owner-computes composite execution is prepared"
                if executable
                else (
                    "exact distributed route and Krylov resources exceeded maximum_bytes"
                    if not admitted
                    else "multi-part execution requires a real JAX ExecutionGroup mesh"
                )
            )
        )
        preparation_id = canonical_fingerprint(
            {
                "kind": "wave-amr-distributed-preparation",
                "prepared": self.prepared_id,
                "distributed": distributed.prepared_id,
                "execution": execution.execution_id,
                "execution_plan": execution_plan_id,
                "mesh": execution.mesh_id,
                "operator": execution.operator_id,
                "physics": execution.physics_id,
                "resources": distributed.resource_evidence_id,
                "preflight_required_bytes": preflight_required,
                "required_bytes": required,
                "maximum_bytes": budget,
                "admitted": admitted,
                "executable": executable,
            }
        )
        return WaveAMRDistributedPreparation(
            distributed,
            execution,
            admitted,
            executable,
            preflight_required,
            required,
            budget,
            reason,
            self.prepared_id,
            self.topology.topology_id,
            partition.plan_id,
            execution_plan_id,
            execution.mesh_id,
            execution.operator_id,
            self.physics.plan_id,
            preparation_id,
        )

    def distributed_step(
        self,
        prepared: WaveAMRDistributedPreparation,
        state: WaveAMRState | DistributedWaveAMRState,
        end_scale_factor: ArrayLike,
        /,
    ) -> WaveAMRResult | DistributedWaveAMRResult:
        if not isinstance(prepared, WaveAMRDistributedPreparation):
            raise TypeError("prepared must be WaveAMRDistributedPreparation.")
        if (
            prepared.source_prepared_id != self.prepared_id
            or prepared.topology_id != self.topology.topology_id
            or prepared.physics_id != self.physics.plan_id
        ):
            raise ValueError(
                "Distributed Wave AMR preparation does not bind this prepared topology."
            )
        if (
            prepared.hierarchy is not None
            and prepared.partition_plan_id != prepared.hierarchy.partition.plan_id
        ):
            raise ValueError("Distributed Wave AMR partition identity is inconsistent.")
        if not prepared.admitted:
            raise ValueError("Distributed Wave AMR resources were not admitted.")
        if not prepared.executable:
            raise ValueError(prepared.reason)
        if prepared.hierarchy is None or prepared.execution is None:
            raise RuntimeError("Distributed Wave AMR preparation has no execution owner.")
        if prepared.hierarchy.partition.part_count == 1:
            if isinstance(state, DistributedWaveAMRState):
                checked_distributed = prepared.execution.validate_state(state)
                hierarchy = prepared.hierarchy.unpack(checked_distributed.psi)
                authority_state = WaveAMRState(
                    hierarchy,
                    checked_distributed.scale_factor,
                    checked_distributed.accepted_boundary,
                )
            else:
                authority_state = state
            return self.step(authority_state, end_scale_factor)
        if isinstance(state, WaveAMRState):
            checked = self._validate_state(state)
            packed_state = prepared.execution.bind_packed_state(
                prepared.execution.pack_canonical_values(
                    self.layout.bind_state(checked.psi)
                ),
                checked.scale_factor,
                accepted_boundary=checked.accepted_boundary,
            )
        elif isinstance(state, DistributedWaveAMRState):
            packed_state = prepared.execution.validate_state(state)
        else:
            raise TypeError("state must be WaveAMRState or DistributedWaveAMRState.")
        end_host = np.asarray(end_scale_factor)
        start_host = np.asarray(packed_state.scale_factor)
        if (
            end_host.shape != ()
            or start_host.shape != ()
            or not np.isfinite(end_host).item()
            or not np.isfinite(start_host).item()
            or float(end_host) <= float(start_host)
        ):
            raise ValueError(
                "Distributed Wave AMR end scale factor must be finite and greater "
                "than the current accepted level."
            )
        end = prepared.execution._replicate_scalar(end_scale_factor, self.real_dtype)
        kick = self.background.kick_factor(packed_state.scale_factor, end).astype(
            self.real_dtype
        )
        drift = self.background.drift_factor(packed_state.scale_factor, end).astype(
            self.real_dtype
        )
        return prepared.execution.step(packed_state, end, kick, drift)

    def migrate_distributed_state(
        self,
        source: WaveAMRDistributedPreparation,
        state: DistributedWaveAMRState,
        target: WaveAMRDistributedPreparation,
        /,
    ) -> DistributedWaveAMRState:
        """Repartition an accepted fixed-topology boundary by stable block ID."""
        if (
            not isinstance(source, WaveAMRDistributedPreparation)
            or not isinstance(target, WaveAMRDistributedPreparation)
            or source.execution is None
            or target.execution is None
        ):
            raise TypeError("Distributed migration requires two prepared executions.")
        if (
            source.source_prepared_id != self.prepared_id
            or target.source_prepared_id != self.prepared_id
        ):
            raise ValueError("Distributed migration preparations bind another Wave AMR.")
        return source.execution.migrate_accepted_state(state, target.execution)

    def fixed_topology_jvp(
        self,
        state: WaveAMRState,
        tangent: Sequence[ArrayLike],
        end_scale_factor: ArrayLike,
        /,
    ) -> tuple[Array, ...]:
        """Differentiate only the smooth fixed-epoch step map, never regridding."""
        checked = self._validate_state(state)
        tangent_values = self.layout.validate(
            tuple(jnp.asarray(value, dtype=self.physics.dtype) for value in tangent)
        )

        def action(values):
            hierarchy = self._hierarchy(values)
            result = self.step(
                WaveAMRState(hierarchy, checked.scale_factor, jnp.asarray(True)),
                end_scale_factor,
            )
            return self.layout.bind_state(result.candidate_state.psi)

        primal = self.step(checked, end_scale_factor)
        _, result = jax.jvp(
            action, (self.layout.bind_state(checked.psi),), (tangent_values,)
        )
        leaves = list(result)
        leaves[0] = eqx.error_if(
            leaves[0],
            ~primal.successful,
            "Wave AMR fixed-topology derivative requires an accepted primal step.",
        )
        return tuple(leaves)


__all__ = [
    "PreparedWaveAMR",
    "WaveAMRAdaptivityPlan",
    "WaveAMRDiagnostics",
    "WaveAMRDiscretizationPlan",
    "WaveAMRDistributedPreparation",
    "WaveAMRPhysicsPlan",
    "WaveAMRResult",
    "WaveAMRState",
    "WaveAMRTopologyIndicators",
    "WaveAMRTopologyProposal",
    "WaveAMRTransferEvidence",
    "WaveAMRTransitionResult",
]

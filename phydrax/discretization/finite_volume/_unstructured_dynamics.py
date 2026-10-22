#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

import phydrax.ein as ein
import phydrax.linalg as la

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum, compensated_sum_chunks
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import fixed_field, NonTrainableState
from ..._validation import positive_finite_float
from ...ein import contract
from ...equations import (
    _multiphase as multiphase_owner,
    _vof_phase_change as vof_phase_change_owner,
)
from ...equations._hyperbolic_systems import AbstractConservationSystem
from ...solver import _finite_volume_content as finite_volume_content_owner
from ...typing import checked
from .._conservation_boundary import (
    AbstractConservationBoundary,
    ALEBoundaryContext,
    ConstantStateBoundary,
    ExtrapolationBoundary,
    PrescribedStateBoundary,
    SourceFunction,
)
from .._conservation_ledger import (
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from .._distributed_field import DistributedHaloPlan, make_owner_local_field_array
from .._distributed_preparation import owner_local_host_array, replicated_host_array
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._closure import AbstractFaceClosurePlan, FaceFluxContext
from ._contact_angle import reconstruct_wall_interface_normal
from ._coupling import (
    PreparedUnstructuredFiniteVolumeCoupling,
    UnstructuredFiniteVolumeCouplingPlan,
)
from ._dyadic import DyadicFiniteVolumeDiscretization
from ._embedded_dynamics import (
    lower_embedded_stage_metrics,
    UnstructuredEmbeddedBoundarySet,
)
from ._geometry_protocol import (
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageFaceBlock,
    FiniteVolumeStageFaceLayout,
    FiniteVolumeStageMetrics,
    lower_static_unstructured_stage_metrics,
)
from ._physical_boundaries import (
    MovingSlipWallBoundary,
    SlipWallBoundary,
    SupersonicInflowBoundary,
    SupersonicOutflowBoundary,
)
from ._precision import FiniteVolumePrecisionPolicy
from ._reconstruction import PiecewiseConstantReconstruction
from ._riemann import (
    AbstractArbitraryNormalALENumericalFluxPlan,
    AbstractArbitraryNormalNumericalFluxPlan,
    AbstractNumericalFluxPlan,
    NumericalFluxResult,
)
from ._small_cell import ConservativeSmallCellRedistributionPlan
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_overset import PeriodicSlidingCoupling, UnstructuredOversetPlan
from ._unstructured_vof import JAXPLICStageReconstruction, UnstructuredVOFPlan
from ._unstructured_weno import PreparedUnstructuredWENOZReconstruction


type PreparedUnstructuredReconstruction = (
    PreparedCellPolynomialReconstruction | PreparedUnstructuredWENOZReconstruction
)


class UnstructuredFiniteVolumeBoundarySet(StrictModule, NonTrainableState):
    """Bind mesh patches to named physical-boundary policies."""

    patch_names: tuple[str, ...] = eqx.field(static=True)
    boundaries: tuple[AbstractConservationBoundary, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_names: tuple[str, ...],
        boundaries: Mapping[str, AbstractConservationBoundary],
        /,
    ) -> None:
        names = tuple(patch_names)
        if set(boundaries) != set(names):
            raise ValueError(
                "dynamics.boundaries must cover every mesh patch; got mismatched patch sets."
            )
        values = tuple(boundaries[name] for name in names)
        for value in values:
            match value:
                case (
                    ConstantStateBoundary()
                    | ExtrapolationBoundary()
                    | PrescribedStateBoundary()
                    | MovingSlipWallBoundary()
                    | SlipWallBoundary()
                    | SupersonicInflowBoundary()
                    | SupersonicOutflowBoundary()
                ):
                    pass
                case _:
                    raise TypeError(
                        "dynamics.boundaries must declare normal-oriented exterior-state policies; got an unsupported policy."
                    )
        self.patch_names = names
        self.boundaries = values
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "unstructured-fv-boundary-set",
                "patches": [
                    {"name": name, "boundary": value.boundary_id}
                    for name, value in zip(names, values, strict=True)
                ],
            }
        )


class UnstructuredFiniteVolumeMethodPlan(StrictModule):
    """Reconstruction, normal numerical flux, and optional face closure."""

    reconstruction: PiecewiseConstantReconstruction | PreparedUnstructuredReconstruction
    interface_solver: AbstractArbitraryNormalNumericalFluxPlan
    closure: AbstractFaceClosurePlan | None
    method_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reconstruction: PiecewiseConstantReconstruction
        | PreparedUnstructuredReconstruction,
        interface_solver: AbstractArbitraryNormalNumericalFluxPlan,
        /,
        *,
        closure: AbstractFaceClosurePlan | None = None,
    ) -> None:
        match reconstruction:
            case PiecewiseConstantReconstruction():
                reconstruction_id = reconstruction.plan_id
            case (
                PreparedCellPolynomialReconstruction()
                | PreparedUnstructuredWENOZReconstruction()
            ):
                reconstruction_id = reconstruction.prepared_id
            case _:
                raise TypeError(
                    "dynamics.reconstruction must be a supported reconstruction; got an unsupported kind."
                )
        self.reconstruction = reconstruction
        self.interface_solver = interface_solver
        self.closure = closure
        self.method_id = canonical_fingerprint(
            {
                "kind": "unstructured-fv-method",
                "reconstruction": reconstruction_id,
                "flux": interface_solver.flux_id,
                **({} if closure is None else {"closure": closure.closure_id}),
            }
        )


class UnstructuredFiniteVolumeDiagnostics(StrictModule):
    normal_flux: Array
    signal_speed: Array
    boundary_outward_flux: Array
    source_integral: Array
    conservation_defect: Array
    maximum_rate: Array
    precision_evidence: PrecisionEvidenceEnvelope


class UnstructuredFiniteVolumeStageEvaluation(StrictModule):
    """One certified ALE stage rate and its relative-wave CFL accounting."""

    ledger: ConservationStageLedger
    relative_signal_speeds: tuple[Array, ...]
    cell_relative_rate: Array
    maximum_relative_rate: Array
    relative_cfl_step: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)


class _CouplingBinding(StrictModule):
    coupling: PreparedUnstructuredFiniteVolumeCoupling


class _DynamicsBinding(StrictModule):
    neighbor_state_action: Array | None
    state_action_id: str = eqx.field(static=True)
    coupling: PreparedUnstructuredFiniteVolumeCoupling
    source_id: str | None = eqx.field(static=True)
    precision: FiniteVolumePrecisionPolicy


class _OversetMasks(StrictModule):
    donor_active: np.ndarray
    donor_hole: np.ndarray
    donor_eligible: np.ndarray
    receptor_active: np.ndarray
    receptor_hole: np.ndarray
    receptor_fringe: np.ndarray
    active_cell_mask: np.ndarray


class _OversetPreparation(StrictModule):
    template: ConservationStageFluxRateBlock | None
    active_cell_mask: np.ndarray
    effective_cell_volumes: Array
    policy_id: str | None = eqx.field(static=True)
    mapping_id: str | None = eqx.field(static=True)
    epoch_id: str | None = eqx.field(static=True)


class _StageRoutePreparation(StrictModule):
    boundary_face_indices: tuple[Array, ...]
    rate_templates: tuple[ConservationStageFluxRateBlock, ...]
    boundary_routes: tuple[tuple[Array, ...], ...]
    source_cell_indices: Array


def _prepare_coupling_binding(
    discretization: UnstructuredFiniteVolumeDiscretization
    | DyadicFiniteVolumeDiscretization,
    method: UnstructuredFiniteVolumeMethodPlan,
    reconstruction: PiecewiseConstantReconstruction | PreparedUnstructuredReconstruction,
    coupling: PreparedUnstructuredFiniteVolumeCoupling | None,
    /,
) -> _CouplingBinding:
    coupling_ = (
        UnstructuredFiniteVolumeCouplingPlan().prepare(discretization)
        if coupling is None
        else coupling
    )
    if (
        coupling_.topology_id != discretization.topology_id
        or coupling_.geometry_id != discretization.geometry_id
        or coupling_.discretization_id != discretization.prepared_id
    ):
        raise ValueError(
            "dynamics.coupling must bind the prepared topology and geometry; got mismatched IDs."
        )
    if coupling_.motion is not None:
        match reconstruction:
            case PiecewiseConstantReconstruction():
                moving_supported = True
            case PreparedCellPolynomialReconstruction():
                moving_supported = reconstruction.basis.degree in (1, 2)
            case PreparedUnstructuredWENOZReconstruction():
                moving_supported = reconstruction.optimal.basis.degree in (1, 2)
            case _:
                raise TypeError(
                    "dynamics.reconstruction must be a supported reconstruction; got an unsupported kind."
                )
        if not moving_supported:
            raise ValueError(
                f"dynamics.reconstruction must be piecewise constant or stage-refreshable degree-one/two cell-polynomial or WENO-Z for motion; got {type(reconstruction).__name__} (coupling={coupling_.prepared_id}, method={method.method_id})."
            )
    if coupling_.motion is not None or coupling_.overset is not None:
        match method.interface_solver:
            case AbstractArbitraryNormalALENumericalFluxPlan():
                pass
            case AbstractArbitraryNormalNumericalFluxPlan():
                raise ValueError(
                    f"dynamics.interface_solver must declare an arbitrary-normal ALE numerical flux for moving or overset routes; got {type(method.interface_solver).__name__} (coupling={coupling_.prepared_id}, method={method.method_id})."
                )
            case _:
                raise TypeError(
                    "dynamics.interface_solver must declare an arbitrary-normal flux; got an unsupported solver."
                )
    return _CouplingBinding(coupling_)


def _prepare_dynamics_binding(
    system: AbstractConservationSystem,
    discretization: UnstructuredFiniteVolumeDiscretization
    | DyadicFiniteVolumeDiscretization,
    method: UnstructuredFiniteVolumeMethodPlan,
    boundaries: UnstructuredFiniteVolumeBoundarySet,
    source: SourceFunction | None,
    source_id: str | None,
    precision: FiniteVolumePrecisionPolicy | None,
    coupling: PreparedUnstructuredFiniteVolumeCoupling | None,
) -> _DynamicsBinding:
    if boundaries.patch_names != discretization.boundary_patch_names:
        raise ValueError(
            "dynamics.boundaries.patch_names must match the prepared mesh patches; got mismatched names."
        )
    if (
        system.dimension != discretization.cell_dimension
        or system.component_count != discretization.component_count
    ):
        raise ValueError(
            "dynamics.system must match the geometry dimension and component count; got incompatible dimensions or components."
        )
    state_action = None
    state_action_id = "identity"
    match discretization:
        case UnstructuredFiniteVolumeDiscretization():
            if discretization.mesh.periodic_topology is not None:
                dimension = discretization.cell_dimension
                state_action = system.proper_isometry_state_matrix(
                    discretization.neighbor_frame_maps[:, :dimension, :dimension]
                )
                state_action_id = canonical_fingerprint(
                    {
                        "system": system.system_id,
                        "periodic": discretization.mesh.periodic_topology.periodic_topology_id,
                    }
                )
            if discretization.partition_count > 1:
                if type(method.reconstruction) is not PiecewiseConstantReconstruction:
                    raise ValueError(
                        "dynamics.reconstruction must be piecewise constant for owner-local neighborhoods; got a higher-order reconstruction."
                    )
                if coupling is not None:
                    raise ValueError(
                        "dynamics.coupling must be absent for owner-local geometry; got a supplied coupling."
                    )
        case DyadicFiniteVolumeDiscretization():
            pass
        case _:
            raise TypeError(
                "dynamics.discretization must declare unstructured or dyadic geometry; got an unsupported kind."
            )
    if method.closure is not None:
        method.closure.admit_system(system)
    reconstruction = method.reconstruction
    match reconstruction:
        case PiecewiseConstantReconstruction():
            pass
        case (
            PreparedCellPolynomialReconstruction()
            | PreparedUnstructuredWENOZReconstruction()
        ):
            if reconstruction.discretization.prepared_id != discretization.prepared_id:
                raise ValueError(
                    "dynamics.reconstruction must bind the prepared geometry; got a different discretization ID."
                )
        case _:
            raise TypeError(
                "dynamics.reconstruction must be a supported reconstruction; got an unsupported kind."
            )
    coupling_binding = _prepare_coupling_binding(
        discretization, method, reconstruction, coupling
    )
    coupling_ = coupling_binding.coupling
    source_identifier = None if source_id is None else str(source_id)
    if (source is None) != (source_identifier is None) or source_identifier == "":
        raise ValueError(
            "dynamics.source_id must identify exactly one source callable; got missing, empty, or unpaired source metadata."
        )
    precision_ = (
        FiniteVolumePrecisionPolicy(jnp.dtype(discretization.cell_volumes.dtype).name)
        if precision is None
        else precision
    )
    return _DynamicsBinding(
        state_action, state_action_id, coupling_, source_identifier, precision_
    )


def _prepare_overset_masks(
    discretization: UnstructuredFiniteVolumeDiscretization
    | DyadicFiniteVolumeDiscretization,
    overset: UnstructuredOversetPlan,
    /,
) -> _OversetMasks:
    donor_count = np.asarray(overset.donor_covered_measures).size
    if donor_count != discretization.cell_count:
        raise ValueError(
            "dynamics.overset must share one donor and receptor content axis; got different cell counts."
        )
    donor_active = np.asarray(overset.donor_active_mask)
    donor_hole = np.asarray(overset.donor_hole_mask)
    donor_eligible = np.asarray(overset.donor_eligible_mask)
    receptor_active = np.asarray(overset.receptor_active_mask)
    receptor_hole = np.asarray(overset.receptor_hole_mask)
    masks = (
        donor_active,
        donor_hole,
        donor_eligible,
        receptor_active,
        receptor_hole,
    )
    if any(value.dtype.kind != "b" for value in masks):
        raise TypeError(
            "dynamics.overset.masks must have Boolean dtype; got a non-Boolean activity or hole mask."
        )
    donor_active = donor_active & ~donor_hole
    donor_eligible = donor_eligible & ~donor_hole
    receptor_active = receptor_active & ~receptor_hole
    receptor_fringe = np.asarray(overset.receptor_fringe_mask)
    if receptor_fringe.dtype.kind != "b":
        raise TypeError(
            "dynamics.overset.receptor_fringe_mask must have Boolean dtype; got a non-Boolean mask."
        )
    if (
        donor_active.shape != (donor_count,)
        or donor_hole.shape != (donor_count,)
        or donor_eligible.shape != (donor_count,)
        or receptor_active.shape != (discretization.cell_count,)
        or receptor_hole.shape != (discretization.cell_count,)
        or receptor_fringe.shape != (discretization.cell_count,)
        or np.any(receptor_fringe & receptor_hole)
        or np.any(receptor_fringe & ~receptor_active)
    ):
        raise ValueError(
            "dynamics.overset.masks must agree on active, hole, and fringe ownership; got incompatible masks."
        )
    overset_active_cell_mask = receptor_active & ~receptor_hole
    return _OversetMasks(
        donor_active,
        donor_hole,
        donor_eligible,
        receptor_active,
        receptor_hole,
        receptor_fringe,
        overset_active_cell_mask,
    )


def _prepare_overset_routes(
    discretization: UnstructuredFiniteVolumeDiscretization
    | DyadicFiniteVolumeDiscretization,
    coupling_: PreparedUnstructuredFiniteVolumeCoupling,
    precision_: FiniteVolumePrecisionPolicy,
) -> _OversetPreparation:
    overset = coupling_.overset
    overset_rate_block_template = None
    overset_policy_id = coupling_.overset_policy_id
    overset_mapping_id = coupling_.overset_mapping_id
    overset_epoch_id = coupling_.overset_epoch_id
    if overset is None:
        overset_active_cell_mask = np.ones((discretization.cell_count,), dtype=np.bool_)
        overset_effective_cell_volumes = jnp.asarray(discretization.cell_volumes)
    else:
        masks = _prepare_overset_masks(discretization, overset)
        donor_active, donor_hole, donor_eligible = (
            masks.donor_active,
            masks.donor_hole,
            masks.donor_eligible,
        )
        receptor_fringe = masks.receptor_fringe
        overset_active_cell_mask = masks.active_cell_mask
        donor_indices = np.asarray(overset.donor_indices, dtype=np.int32)
        receptor_cells = np.asarray(overset.receptor_cells, dtype=np.int32)
        receptor_routes = np.asarray(overset.receptor_routes, dtype=np.int32)
        sliding_coupling = coupling_.sliding_coupling
        if sliding_coupling is None:
            template_owner_cells = donor_indices
            template_neighbor_cells = receptor_cells[receptor_routes]
            if overset.receptor_face_cells is not None:
                certified_face_cells = np.asarray(
                    overset.receptor_face_cells,
                    dtype=np.int32,
                )
                if (
                    certified_face_cells.shape != template_neighbor_cells.shape
                    or not np.array_equal(
                        certified_face_cells,
                        template_neighbor_cells,
                    )
                ):
                    raise ValueError(
                        "dynamics.overset.receptor_face_cells must match the mapped receptor routes in order; got mismatched cells."
                    )
            sliding_coupling_id = None
        else:
            if overset.receptor_face_cells is None:
                raise ValueError(
                    "dynamics.overset.receptor_face_cells must certify sliding receptor faces; got None."
                )
            left_routes = np.asarray(sliding_coupling.left_routes, dtype=np.int32)
            right_routes = np.asarray(sliding_coupling.right_routes, dtype=np.int32)
            face_cells = np.asarray(overset.receptor_face_cells, dtype=np.int32)
            template_owner_cells = donor_indices[left_routes]
            template_neighbor_cells = face_cells[right_routes]
            sliding_coupling_id = sliding_coupling.coupling_id
        route_active = (
            donor_active[template_owner_cells]
            & donor_eligible[template_owner_cells]
            & ~donor_hole[template_owner_cells]
            & overset_active_cell_mask[template_neighbor_cells]
            & receptor_fringe[template_neighbor_cells]
            & (template_owner_cells != template_neighbor_cells)
        )
        overset_rate_block_template = ConservationStageFluxRateBlock(
            jnp.zeros(
                (template_owner_cells.size, discretization.component_count),
                dtype=jnp.dtype(precision_.reduction_dtype),
            ),
            template_owner_cells,
            template_neighbor_cells,
            route_active,
            canonical_fingerprint(
                {
                    "kind": "unstructured-overset-correction-route",
                    "mapping": coupling_.overset_mapping_id,
                    "epoch": coupling_.overset_epoch_id,
                    "sliding_coupling": sliding_coupling_id,
                }
            ),
            "overset-correction",
        )
        overset_effective_cell_volumes = jnp.where(
            jnp.asarray(overset_active_cell_mask),
            jnp.asarray(discretization.cell_volumes),
            jnp.zeros_like(discretization.cell_volumes),
        )
    return _OversetPreparation(
        overset_rate_block_template,
        overset_active_cell_mask,
        overset_effective_cell_volumes,
        overset_policy_id,
        overset_mapping_id,
        overset_epoch_id,
    )


def _prepare_stage_routes(
    discretization: UnstructuredFiniteVolumeDiscretization
    | DyadicFiniteVolumeDiscretization,
    boundaries: UnstructuredFiniteVolumeBoundarySet,
    coupling_: PreparedUnstructuredFiniteVolumeCoupling,
    precision_: FiniteVolumePrecisionPolicy,
    overset_active_cell_mask: np.ndarray,
    state_action: Array | None,
    state_action_id: str,
) -> _StageRoutePreparation:
    if coupling_.motion is not None:
        stage_layouts = (coupling_.motion.face_layout,)
    elif coupling_.embedded_metrics is not None:
        embedded_boundaries = coupling_.embedded_boundaries
        match embedded_boundaries:
            case UnstructuredEmbeddedBoundarySet():
                pass
            case _:
                raise TypeError(
                    "dynamics.coupling.embedded_boundaries must contain an embedded boundary set; got an unsupported owner."
                )
        embedded_stage = lower_embedded_stage_metrics(
            discretization,
            coupling_.embedded_metrics,
            embedded_boundaries,
            discretization.topology_id,
            0,
            0,
            time=0.0,
        )
        stage_layouts = tuple(block.layout for block in embedded_stage.face_blocks)
    else:
        match discretization:
            case UnstructuredFiniteVolumeDiscretization() if (
                discretization.partition_count > 1
            ):
                # Owner-local execution consumes the certified face routes directly;
                # it never lowers ghost closure measures into serial stage metrics.
                stage_layouts = tuple(
                    FiniteVolumeStageFaceLayout(
                        face_ids=block.face_ids,
                        owner_cells=block.owner_cells,
                        neighbor_cells=block.neighbor_cells,
                        boundary_policy_ids=block.boundary_patch_ids,
                        boundary_policy_count=len(discretization.boundary_patch_names),
                        active_mask=block.active_mask,
                        block_kind="physical",
                        spatial_shape=tuple(block.face_centers.shape),
                        quadrature_shape=tuple(
                            discretization.face_quadrature_weights[block.face_ids].shape
                        ),
                        block_id=block.block_id,
                    )
                    for block in discretization.face_blocks
                )
            case (
                UnstructuredFiniteVolumeDiscretization()
                | DyadicFiniteVolumeDiscretization()
            ):
                static_stage = lower_static_unstructured_stage_metrics(discretization)
                stage_layouts = tuple(block.layout for block in static_stage.face_blocks)
            case _:
                raise TypeError(
                    "dynamics.discretization must declare unstructured or dyadic geometry; got an unsupported kind."
                )
    embedded_policy_count = (
        0
        if coupling_.embedded_boundaries is None
        else len(coupling_.embedded_boundaries.boundaries)
    )
    for layout in stage_layouts:
        policy_count = (
            len(boundaries.boundaries)
            if layout.block_kind == "physical"
            else embedded_policy_count
        )
        layout.validate_boundary_policy_count(policy_count)
    stage_rate_block_templates = tuple(
        ConservationStageFluxRateBlock(
            jnp.zeros(
                (layout.face_count, discretization.component_count),
                dtype=jnp.dtype(precision_.reduction_dtype),
            ),
            layout.owner_cells,
            layout.neighbor_cells,
            np.asarray(layout.active_mask)
            & np.asarray(overset_active_cell_mask)[np.asarray(layout.owner_cells)]
            & (
                (np.asarray(layout.neighbor_cells) < 0)
                | np.asarray(overset_active_cell_mask)[
                    np.maximum(np.asarray(layout.neighbor_cells), 0)
                ]
            ),
            layout.block_id,
            layout.block_kind,
            neighbor_state_action=None
            if state_action is None
            else state_action[layout.face_ids],
            state_action_id=state_action_id,
        )
        for layout in stage_layouts
    )
    stage_boundary_face_indices = tuple(
        tuple(
            jnp.asarray(
                np.flatnonzero(np.asarray(layout.boundary_policy_ids) == policy_id),
                dtype=jnp.int32,
            )
            for policy_id in range(layout.boundary_policy_count)
        )
        for layout in stage_layouts
    )
    boundary_face_indices = tuple(
        jnp.asarray(
            np.flatnonzero(np.asarray(discretization.boundary_patch_ids) == patch_id),
            dtype=jnp.int32,
        )
        for patch_id in range(len(boundaries.patch_names))
    )
    source_active_cell_mask = np.asarray(overset_active_cell_mask, dtype=np.bool_)
    if coupling_.embedded_metrics is not None:
        source_active_cell_mask = source_active_cell_mask & np.asarray(
            coupling_.embedded_metrics.active_fluid_cells,
            dtype=np.bool_,
        )
    source_cell_indices = jnp.asarray(
        np.flatnonzero(source_active_cell_mask),
        dtype=jnp.int32,
    )
    return _StageRoutePreparation(
        boundary_face_indices,
        stage_rate_block_templates,
        stage_boundary_face_indices,
        source_cell_indices,
    )


class _ConstantStageReconstruction(StrictModule):
    state: Array


class _FixedPolynomialStageReconstruction(StrictModule):
    state: Array
    polynomial: PreparedCellPolynomialReconstruction
    coefficients: Array
    limiter: PreparedUnstructuredWENOZReconstruction | None


class _MovingPolynomialStageReconstruction(StrictModule):
    state: Array
    polynomial: PreparedCellPolynomialReconstruction
    coefficients: Array
    lengths: Array
    metrics: FiniteVolumeStageMetrics
    limiter: PreparedUnstructuredWENOZReconstruction | None


type _StageReconstruction = (
    _ConstantStageReconstruction
    | _FixedPolynomialStageReconstruction
    | _MovingPolynomialStageReconstruction
)


class _TracePair(StrictModule):
    left: Array
    right: Array


class _BoundaryStageState(StrictModule):
    right: Array
    normal: Array


class _OversetFaceGeometry(StrictModule):
    cells: Array
    points: Array
    normals: Array
    weights: Array
    grid_velocity: Array
    face_measures: Array


class _OversetCorrection(StrictModule):
    block: ConservationStageFluxRateBlock | None
    signal_speed: Array
    route_measures: Array


class _OversetPhaseFlux(StrictModule):
    flux_rate: Array


class _VOFStage(StrictModule):
    system: multiphase_owner.TwoMaterialVOFSystem
    plan: UnstructuredVOFPlan
    plic: JAXPLICStageReconstruction
    primitive: Array


class _VOFFaceFlux(StrictModule):
    flux_rate: Array
    volume_divergence: Array


class _StageFaceRate(StrictModule):
    block: ConservationStageFluxRateBlock
    signal_speed: Array
    relative_rate: Array
    volume_divergence: Array


class _StageSourceRate(StrictModule):
    source_rate: Array


class _PhaseChangeSourceRate(StrictModule):
    source_rate: Array
    phase_change_step: Array
    thermal_step: Array


class _CapillarySourceRate(StrictModule):
    source_rate: Array
    capillary_step: Array


class _FaceStageState(StrictModule):
    left: Array
    right: Array
    normal: Array


class PreparedUnstructuredFiniteVolumeDynamics(StrictModule):
    """Single-device conservative dynamics over explicit unstructured faces."""

    system: AbstractConservationSystem
    discretization: (
        UnstructuredFiniteVolumeDiscretization | DyadicFiniteVolumeDiscretization
    )
    method: UnstructuredFiniteVolumeMethodPlan
    boundaries: UnstructuredFiniteVolumeBoundarySet
    coupling: PreparedUnstructuredFiniteVolumeCoupling
    precision: FiniteVolumePrecisionPolicy
    boundary_face_indices: tuple[Array, ...] = fixed_field()
    stage_rate_block_templates: tuple[ConservationStageFluxRateBlock, ...] = fixed_field()
    stage_boundary_face_indices: tuple[tuple[Array, ...], ...] = fixed_field()
    neighbor_state_action: Array | None = fixed_field()
    source_cell_indices: Array = fixed_field()
    overset_rate_block_template: ConservationStageFluxRateBlock | None = fixed_field()
    overset_active_cell_mask: Array = fixed_field()
    overset_effective_cell_volumes: Array = fixed_field()
    overset_policy_id: str | None = eqx.field(static=True)
    overset_mapping_id: str | None = eqx.field(static=True)
    overset_epoch_id: str | None = eqx.field(static=True)
    source: SourceFunction | None = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        system: AbstractConservationSystem,
        discretization: (
            UnstructuredFiniteVolumeDiscretization | DyadicFiniteVolumeDiscretization
        ),
        method: UnstructuredFiniteVolumeMethodPlan,
        boundaries: UnstructuredFiniteVolumeBoundarySet,
        /,
        *,
        source: SourceFunction | None = None,
        source_id: str | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
        coupling: PreparedUnstructuredFiniteVolumeCoupling | None = None,
    ) -> None:
        binding = _prepare_dynamics_binding(
            system,
            discretization,
            method,
            boundaries,
            source,
            source_id,
            precision,
            coupling,
        )
        self.neighbor_state_action = binding.neighbor_state_action
        coupling_ = binding.coupling
        precision_ = binding.precision
        source_identifier = binding.source_id
        overset_routes = _prepare_overset_routes(discretization, coupling_, precision_)
        stage_routes = _prepare_stage_routes(
            discretization,
            boundaries,
            coupling_,
            precision_,
            overset_routes.active_cell_mask,
            binding.neighbor_state_action,
            binding.state_action_id,
        )
        self.system = system
        self.discretization = discretization
        self.method = method
        self.boundaries = boundaries
        self.coupling = coupling_
        self.precision = precision_
        self.source = source
        self.source_id = source_identifier
        self.boundary_face_indices = stage_routes.boundary_face_indices
        self.stage_rate_block_templates = stage_routes.rate_templates
        self.stage_boundary_face_indices = stage_routes.boundary_routes
        self.source_cell_indices = stage_routes.source_cell_indices
        self.overset_rate_block_template = overset_routes.template
        self.overset_active_cell_mask = jnp.asarray(overset_routes.active_cell_mask)

        self.overset_effective_cell_volumes = jnp.asarray(
            overset_routes.effective_cell_volumes
        )
        self.overset_policy_id = overset_routes.policy_id
        self.overset_mapping_id = overset_routes.mapping_id
        self.overset_epoch_id = overset_routes.epoch_id
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-fv-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "coupling": coupling_.prepared_id,
                "boundaries": boundaries.boundary_set_id,
                "precision": precision_.policy_id,
                "overset_policy": overset_routes.policy_id,
                "overset_mapping": overset_routes.mapping_id,
                "overset_epoch": overset_routes.epoch_id,
                "source": source_identifier,
            }
        )

    @checked
    def with_sliding_coupling(
        self,
        sliding_coupling: PeriodicSlidingCoupling,
        /,
    ) -> "PreparedUnstructuredFiniteVolumeDynamics":
        """Reprepare dynamics with an accepted immutable sliding overlap map."""

        coupling = self.coupling.with_sliding_coupling(
            self.discretization,
            sliding_coupling,
        )
        return PreparedUnstructuredFiniteVolumeDynamics(
            self.system,
            self.discretization,
            self.method,
            self.boundaries,
            source=self.source,
            source_id=self.source_id,
            precision=self.precision,
            coupling=coupling,
        )

    @checked
    def make_fallback_dynamics(
        self, fallback_flux: AbstractNumericalFluxPlan, /
    ) -> "PreparedUnstructuredFiniteVolumeDynamics":
        match fallback_flux:
            case AbstractArbitraryNormalNumericalFluxPlan():
                pass
            case _:
                raise TypeError(
                    "dynamics.fallback_flux must declare an arbitrary-normal numerical flux; got an unsupported flux."
                )
        return PreparedUnstructuredFiniteVolumeDynamics(
            self.system,
            self.discretization,
            UnstructuredFiniteVolumeMethodPlan(
                PiecewiseConstantReconstruction(), fallback_flux
            ),
            self.boundaries,
            source=self.source,
            source_id=self.source_id,
            precision=self.precision,
            coupling=self.coupling,
        )

    def _stage_normal_flux(
        self,
        left: Array,
        right: Array,
        normal: Array,
        grid_normal_velocity: Array,
        face_measure: Array,
        active: Array,
        geometry_id: str,
        args: Any,
        /,
    ) -> NumericalFluxResult:
        """Stage normal flux with the closure applied at every active site.

        `face_measure` and `active` match the face-site batch; inactive routes
        carry placeholder geometry in the closure context and no correction.
        """
        solver = self.method.interface_solver
        match solver:
            case AbstractArbitraryNormalALENumericalFluxPlan():
                result = solver.normal_ale_face_flux(
                    self.system, left, right, normal, grid_normal_velocity, args
                )
            case AbstractArbitraryNormalNumericalFluxPlan():
                # Preparation admits stationary-only fluxes only without motion or
                # overset coupling, where every stage grid-normal velocity is zero.
                result = solver.normal_face_flux(self.system, left, right, normal, args)
            case _:
                raise TypeError(
                    "dynamics.interface_solver must declare an arbitrary-normal flux; got an unsupported solver."
                )
        if self.method.closure is None:
            return result
        placeholder = jnp.zeros((normal.shape[-1],), dtype=normal.dtype).at[0].set(1.0)
        context = FaceFluxContext(
            jnp.where(active[..., None], normal, placeholder),
            jnp.where(active, face_measure, jnp.ones((), dtype=face_measure.dtype)),
            grid_normal_velocity,
            geometry_id=geometry_id,
            active=active,
        )
        return NumericalFluxResult(
            self.method.closure.apply(
                self.system, left, right, result.normal_flux, context, args
            ),
            result.max_speed,
            fallback_activated=result.fallback_activated,
        )

    def _stage_route_active(
        self,
        layout: FiniteVolumeStageFaceLayout,
        active_cell_mask: Array,
        /,
    ) -> Array:
        safe_neighbor = jnp.maximum(layout.neighbor_cells, 0)
        return (
            layout.active_mask
            & self.overset_active_cell_mask[layout.owner_cells]
            & ((layout.neighbor_cells < 0) | self.overset_active_cell_mask[safe_neighbor])
            & active_cell_mask[layout.owner_cells]
            & ((layout.neighbor_cells < 0) | active_cell_mask[safe_neighbor])
        )

    def _stage_rate_templates(
        self,
        metrics: FiniteVolumeStageMetrics,
        /,
    ) -> tuple[ConservationStageFluxRateBlock, ...]:
        if len(metrics.face_blocks) != len(self.stage_rate_block_templates):
            raise ValueError(
                "stage.metrics.face_blocks must match the prepared block count; got a different count."
            )
        templates: list[ConservationStageFluxRateBlock] = []
        for geometry_block, prepared in zip(
            metrics.face_blocks,
            self.stage_rate_block_templates,
            strict=True,
        ):
            layout = geometry_block.layout
            if (
                layout.block_kind != prepared.block_kind
                or layout.face_count != prepared.flux_rate.shape[0]
            ):
                raise ValueError(
                    "stage.metrics.face_blocks must match prepared block kinds and capacities; got a different kind or shape."
                )
            active = self._stage_route_active(layout, metrics.active_cell_mask)
            zero_rate = eqx.error_if(
                jnp.zeros_like(prepared.flux_rate),
                jnp.any(layout.owner_cells != prepared.owner_cells)
                | jnp.any(layout.neighbor_cells != prepared.neighbor_cells)
                | jnp.any(active & ~prepared.active_mask),
                "Stage geometry face routes do not match the dynamics.",
            )
            template = prepared.with_flux_rate(zero_rate)
            templates.append(template)
        return tuple(templates)

    @checked
    def _validated_stage_average(
        self,
        content_state: finite_volume_content_owner.FiniteVolumeConservativeContentState,
        metrics: FiniteVolumeStageMetrics,
        /,
    ) -> Array:

        if content_state.precision.policy_id != self.precision.policy_id:
            raise ValueError(
                "stage.content.precision must match the dynamics precision policy; got a different policy ID."
            )
        if content_state.topology_epoch_id != metrics.topology_epoch_id:
            raise ValueError(
                "stage.content.topology_epoch_id must match the stage geometry; got a different epoch."
            )
        if content_state.geometry_family_id != metrics.geometry_family_id:
            raise ValueError(
                "stage.content.geometry_family_id must match the stage geometry; got a different family."
            )
        if content_state.geometry_layout_id != metrics.geometry_layout_id:
            raise ValueError(
                "stage.content.geometry_layout_id must match the stage geometry; got a different layout."
            )
        if content_state.evidence_policy_id != metrics.evidence.policy_id:
            raise ValueError(
                "stage.content.evidence_policy_id must match the stage geometry; got a different policy."
            )
        if (
            content_state.cell_count != metrics.cell_count
            or content_state.component_shape != (self.discretization.component_count,)
        ):
            raise ValueError(
                "stage.content must match the cell count and component shape; got an incompatible shape."
            )
        self._stage_rate_templates(metrics)

        average = content_state.cell_average()
        average = eqx.error_if(
            average,
            ~metrics.evidence.passed
            | (metrics.evidence.status != int(FiniteVolumeGeometryStatus.SUCCESS)),
            "Physical flux evaluation requires SUCCESS stage geometry evidence.",
        )
        average = eqx.error_if(
            average,
            (content_state.geometry_version != metrics.geometry_version)
            | (content_state.evidence_version != metrics.evidence.evidence_version)
            | (content_state.time != metrics.time),
            "Stage content version/time does not match its certified geometry.",
        )
        average = eqx.error_if(
            average,
            jnp.any(
                content_state.effective_cell_volumes
                != self.precision.reduction(metrics.effective_cell_volumes)
            )
            | jnp.any(content_state.active_cell_mask != metrics.active_cell_mask),
            "Stage content volumes/activity do not match its certified geometry.",
        )
        self.precision.validate_state(average)
        return average

    def _prepare_stage_reconstruction(
        self, state: Array, metrics: FiniteVolumeStageMetrics, /
    ) -> _StageReconstruction:
        match self.method.reconstruction:
            case PiecewiseConstantReconstruction():
                return _ConstantStageReconstruction(state)
            case PreparedCellPolynomialReconstruction() as polynomial:
                if self.coupling.motion is None:
                    return _FixedPolynomialStageReconstruction(
                        state, polynomial, polynomial.coefficients(state), None
                    )
                coefficients, lengths = polynomial.stage_coefficients(state, metrics)
                return _MovingPolynomialStageReconstruction(
                    state, polynomial, coefficients, lengths, metrics, None
                )
            case PreparedUnstructuredWENOZReconstruction() as weno:
                coefficients = weno.coefficients(state)
                if self.coupling.motion is None:
                    return _FixedPolynomialStageReconstruction(
                        state, weno.optimal, coefficients, weno
                    )
                _, lengths = weno.optimal.stage_coefficients(state, metrics)
                return _MovingPolynomialStageReconstruction(
                    state, weno.optimal, coefficients, lengths, metrics, weno
                )
            case _:
                raise TypeError(
                    "dynamics.reconstruction must contain a supported prepared reconstruction; got an unsupported kind."
                )

    def _evaluate_stage_trace(
        self, prepared: _StageReconstruction, cells: Array, points: Array, /
    ) -> Array:
        match prepared:
            case _ConstantStageReconstruction():
                return jnp.broadcast_to(
                    prepared.state[cells, None, :],
                    (*points.shape[:-1], prepared.state.shape[-1]),
                )
            case _FixedPolynomialStageReconstruction():
                value = prepared.polynomial.evaluate_coefficients(
                    prepared.state, prepared.coefficients, cells, points
                )
            case _MovingPolynomialStageReconstruction():
                value = prepared.polynomial.evaluate_stage_coefficients(
                    prepared.state,
                    prepared.coefficients,
                    prepared.lengths,
                    prepared.metrics,
                    cells,
                    points,
                )
            case _:
                raise TypeError(
                    "dynamics.reconstruction must contain a supported prepared reconstruction; got an unsupported kind."
                )
        return value

    def _evaluate_stage_trace_pair(
        self,
        prepared: _StageReconstruction,
        left_cells: Array,
        left_points: Array,
        right_cells: Array,
        right_points: Array,
        /,
    ) -> _TracePair:
        left = self._evaluate_stage_trace(prepared, left_cells, left_points)
        right = self._evaluate_stage_trace(prepared, right_cells, right_points)
        match prepared:
            case _ConstantStageReconstruction():
                pass
            case (
                _FixedPolynomialStageReconstruction()
                | _MovingPolynomialStageReconstruction()
            ):
                if prepared.limiter is not None:
                    left = prepared.limiter.limit_traces(prepared.state, left, left_cells)
                    right = prepared.limiter.limit_traces(
                        prepared.state, right, right_cells
                    )
            case _:
                raise TypeError(
                    "dynamics.reconstruction must contain a supported prepared reconstruction; got an unsupported kind."
                )
        return _TracePair(left, right)

    def _apply_stage_boundaries(
        self,
        geometry_block: FiniteVolumeStageFaceBlock,
        policy_indices: tuple[Array, ...],
        metrics: FiniteVolumeStageMetrics,
        points: Array,
        left: Array,
        right: Array,
        route_active: Array,
        args: Any,
        /,
    ) -> _BoundaryStageState:
        layout = geometry_block.layout
        safe_measures = jnp.where(
            route_active,
            geometry_block.face_measures,
            jnp.ones_like(geometry_block.face_measures),
        )
        normal = geometry_block.area_vectors / safe_measures[:, None]
        normal = jnp.where(
            route_active[:, None],
            normal,
            jnp.zeros_like(normal),
        )
        normal = jnp.broadcast_to(normal[:, None, :], points.shape)
        grid_velocity = geometry_block.quadrature_grid_normal_velocity
        if grid_velocity is None:
            if self.coupling.motion is not None:
                raise ValueError(
                    "stage.metrics.grid_normal_velocity must be present for moving geometry; got None."
                )
            grid_velocity = jnp.zeros_like(geometry_block.quadrature_weights)
        grid_normal = self.precision.reconstruction(grid_velocity)
        match layout.block_kind:
            case "physical":
                policies = self.boundaries.boundaries
            case "cut":
                match self.coupling.embedded_boundaries:
                    case UnstructuredEmbeddedBoundarySet() as embedded_boundaries:
                        policies = embedded_boundaries.boundaries
                    case _:
                        raise TypeError(
                            "stage.cut.embedded_boundaries must contain an embedded boundary set; got an unsupported owner."
                        )
            case _:
                raise ValueError(
                    f"stage.face_block.kind must declare physical or cut geometry; got {layout.block_kind!r}."
                )
        if len(policies) != len(policy_indices):
            raise ValueError(
                "stage.boundary_routes must match the prepared policy count; got a different count."
            )

        motion_plan_id = (
            self.coupling.motion.plan_id
            if self.coupling.motion is not None
            else self.coupling.prepared_id
        )
        for policy, indices in zip(policies, policy_indices, strict=True):
            if indices.shape[0] == 0:
                continue
            patch_points = points[indices]
            patch_normal = normal[indices]
            patch_grid_velocity = grid_normal[indices, :, None] * patch_normal
            match policy:
                case MovingSlipWallBoundary():
                    context = policy.make_context(
                        metrics.time,
                        patch_points,
                        patch_normal,
                        patch_grid_velocity,
                        args,
                        topology_epoch_id=metrics.topology_epoch_id,
                        geometry_layout_id=metrics.geometry_layout_id,
                        geometry_version=metrics.geometry_version,
                        face_block_id=layout.block_id,
                        motion_plan_id=motion_plan_id,
                    )
                case AbstractConservationBoundary():
                    context_dtype = jnp.result_type(
                        patch_points.dtype,
                        patch_normal.dtype,
                        patch_grid_velocity.dtype,
                    )
                    tolerance = (
                        64.0 * np.finfo(np.dtype(context_dtype)).eps
                        if jnp.issubdtype(context_dtype, jnp.inexact)
                        else 0.0
                    )
                    context = ALEBoundaryContext(
                        face_point=patch_points,
                        outward_normal=patch_normal,
                        quadrature_grid_velocity=patch_grid_velocity,
                        wall_velocity=jnp.zeros_like(patch_grid_velocity),
                        time=metrics.time,
                        args=args,
                        absolute_tolerance=float(tolerance),
                        relative_tolerance=float(tolerance),
                        topology_epoch_id=metrics.topology_epoch_id,
                        geometry_layout_id=metrics.geometry_layout_id,
                        geometry_version=metrics.geometry_version,
                        face_block_id=layout.block_id,
                        motion_plan_id=motion_plan_id,
                    )
                case _:
                    raise TypeError(
                        "stage.boundary must declare an exterior-state policy; got unsupported boundary."
                    )
            patch_interior = context.validate_consumer_identity(
                left[indices],
                topology_epoch_id=metrics.topology_epoch_id,
                geometry_layout_id=metrics.geometry_layout_id,
                geometry_version=metrics.geometry_version,
                face_block_id=layout.block_id,
                motion_plan_id=motion_plan_id,
            )
            exterior = policy.ale_exterior_state(
                self.system,
                patch_interior,
                context,
                0,
            )
            right = right.at[indices].set(exterior)
        return _BoundaryStageState(right, normal)

    def _stage_face_states(
        self,
        state: Array,
        metrics: FiniteVolumeStageMetrics,
        args: Any,
        /,
    ) -> tuple[_FaceStageState, ...]:
        active_mask = metrics.active_cell_mask
        safe_seed = state[jnp.argmax(active_mask)]
        safe_state = jnp.where(active_mask[:, None], state, safe_seed[None, :])
        reconstruction_state = self.precision.reconstruction(safe_state)
        prepared_reconstruction: _StageReconstruction | None = None
        outputs: list[_FaceStageState] = []
        for block_index, geometry_block in enumerate(metrics.face_blocks):
            layout = geometry_block.layout
            policy_indices = self.stage_boundary_face_indices[block_index]
            owner = layout.owner_cells
            neighbor = layout.neighbor_cells
            safe_neighbor = jnp.maximum(neighbor, 0)
            route_active = self._stage_route_active(layout, metrics.active_cell_mask)
            points = self.precision.reconstruction(geometry_block.quadrature_points)
            neighbor_points = points
            if self.neighbor_state_action is not None:
                match self.discretization:
                    case UnstructuredFiniteVolumeDiscretization() as geometry:
                        maps = geometry.neighbor_frame_maps[layout.face_ids]
                    case _:
                        raise TypeError(
                            "stage.periodic_frame_action must bind unstructured finite-volume geometry; got another geometry kind."
                        )
                dimension = self.discretization.cell_dimension
                neighbor_points = (
                    contract(
                        "fij,fqj->fqi",
                        maps[:, :dimension, :dimension],
                        points,
                    )
                    + maps[:, None, :dimension, dimension]
                )
            if prepared_reconstruction is None:
                prepared_reconstruction = self._prepare_stage_reconstruction(
                    reconstruction_state, metrics
                )
            traces = self._evaluate_stage_trace_pair(
                prepared_reconstruction, owner, points, safe_neighbor, neighbor_points
            )
            left, right = traces.left, traces.right
            if self.neighbor_state_action is not None:
                right = contract(
                    "fji,fqj->fqi", self.neighbor_state_action[layout.face_ids], right
                )

            boundary_state = self._apply_stage_boundaries(
                geometry_block,
                policy_indices,
                metrics,
                points,
                left,
                right,
                route_active,
                args,
            )
            right, normal = boundary_state.right, boundary_state.normal
            outputs.append(_FaceStageState(left, right, normal))
        return tuple(outputs)

    def _overset_face_geometry(
        self, overset: UnstructuredOversetPlan, metrics: FiniteVolumeStageMetrics, /
    ) -> _OversetFaceGeometry:
        face_ids = overset.receptor_face_ids
        face_cells = overset.receptor_face_cells
        if face_ids is None or face_cells is None or overset.face_artifact_id is None:
            raise ValueError(
                "stage.overset.receptor_faces must retain certified face IDs, cells, and artifact identity; got missing face metadata."
            )
        physical_blocks = tuple(
            block
            for block in metrics.face_blocks
            if block.layout.block_kind == "physical"
        )
        if len(physical_blocks) != 1:
            raise ValueError(
                "stage.overset.face_blocks must contain one physical block; got a different physical block count."
            )
        physical = physical_blocks[0]
        layout = physical.layout
        ids = jnp.asarray(face_ids, dtype=jnp.int32)
        cells = jnp.asarray(face_cells, dtype=jnp.int32)
        matches = layout.face_ids[:, None] == ids[None, :]
        positions = jnp.argmax(matches, axis=0).astype(jnp.int32)
        points_array = self.precision.reconstruction(
            physical.quadrature_points[positions]
        )
        points_array = eqx.error_if(
            points_array,
            jnp.any(jnp.sum(matches, axis=0) != 1),
            "Certified overset receptor face IDs are stale for this stage layout.",
        )
        physical_owners = layout.owner_cells[positions]
        physical_neighbors = layout.neighbor_cells[positions]
        incident = (physical_owners == cells) | (physical_neighbors == cells)
        points_array = eqx.error_if(
            points_array,
            jnp.any(~incident) | jnp.any(~layout.active_mask[positions]),
            "Certified overset receptor face routes are stale or inactive.",
        )
        face_measures = self.precision.reduction(physical.face_measures[positions])
        unit_normals = self.precision.reconstruction(
            physical.area_vectors[positions]
            / face_measures.astype(physical.area_vectors.dtype)[:, None]
        )
        orientation = jnp.where(physical_owners == cells, 1.0, -1.0)
        normals_array = orientation[:, None, None] * unit_normals[:, None, :]
        normals_array = jnp.broadcast_to(normals_array, points_array.shape)
        measures_array = self.precision.reduction(physical.quadrature_weights[positions])
        grid_velocity = self.precision.flux(
            physical.quadrature_grid_normal_velocity[positions] * orientation[:, None]
        )
        return _OversetFaceGeometry(
            cells,
            points_array,
            normals_array,
            measures_array,
            grid_velocity,
            face_measures,
        )

    def _overset_phase_flux(
        self,
        system: multiphase_owner.TwoMaterialVOFSystem,
        vof_plan: UnstructuredVOFPlan,
        stage_plic: JAXPLICStageReconstruction | None,
        route_flux: Array,
        route_donor_trace: Array,
        route_donor_cells: Array,
        route_points: Array,
        normals_array: Array,
        grid_velocity: Array,
        measures_array: Array,
        route_right: Array,
        route_fraction: Array,
        /,
    ) -> _OversetPhaseFlux:
        if stage_plic is None or stage_plic.plan_id != vof_plan.plan_id:
            raise ValueError(
                "stage.overset.plic must bind the current VOF plan; got missing or mismatched PLIC."
            )
        donor_primitive = system.conserved_to_primitive(route_donor_trace)
        donor_velocity = donor_primitive[..., 2 : 2 + system.dimension]
        receptor_normals = normals_array[route_right]
        relative_normal_velocity = (
            ein.contract(
                "fqd,fqd->fq",
                donor_velocity,
                receptor_normals,
            )
            - grid_velocity[route_right]
        )
        canonical_volume_flux = -relative_normal_velocity
        donor_normals = stage_plic.normals[route_donor_cells]
        donor_offsets = stage_plic.offsets[route_donor_cells]
        signed_phase0 = donor_offsets[:, None] - ein.contract(
            "fd,fqd->fq",
            donor_normals,
            route_points,
        )
        donor_phase0_aperture = (signed_phase0 >= 0.0).astype(canonical_volume_flux.dtype)
        route_quadrature_measures = measures_array[route_right] * route_fraction[:, None]
        phase0_mass_flux = jnp.sum(
            route_quadrature_measures
            * canonical_volume_flux
            * donor_primitive[..., 0]
            * donor_phase0_aperture,
            axis=1,
        )
        phase1_mass_flux = jnp.sum(
            route_quadrature_measures
            * canonical_volume_flux
            * donor_primitive[..., 1]
            * (1.0 - donor_phase0_aperture),
            axis=1,
        )
        alpha_flux = jnp.sum(
            route_quadrature_measures * canonical_volume_flux * donor_phase0_aperture,
            axis=1,
        )
        route_flux = route_flux.at[:, 0].set(phase0_mass_flux)
        route_flux = route_flux.at[:, 1].set(phase1_mass_flux)
        route_flux = route_flux.at[:, system.alpha_index].set(alpha_flux)
        return _OversetPhaseFlux(route_flux)

    def _overset_correction(
        self,
        state: Array,
        metrics: FiniteVolumeStageMetrics,
        args: Any,
        stage_plic: JAXPLICStageReconstruction | None = None,
        /,
    ) -> _OversetCorrection:
        """Evaluate one moved, stage-bound conservative sliding correction."""

        template = self.overset_rate_block_template
        overset = self.coupling.overset
        reduction_dtype = jnp.dtype(self.precision.reduction_dtype)
        if template is None or overset is None:
            return _OversetCorrection(
                None,
                jnp.zeros((), dtype=reduction_dtype),
                jnp.zeros((0,), dtype=reduction_dtype),
            )
        face_geometry = self._overset_face_geometry(overset, metrics)
        cells, points_array = face_geometry.cells, face_geometry.points
        normals_array, measures_array = face_geometry.normals, face_geometry.weights
        grid_velocity, face_measures = (
            face_geometry.grid_velocity,
            face_geometry.face_measures,
        )
        reconstructed = self.precision.reconstruction(state)
        donor_indices = jnp.asarray(overset.donor_indices, dtype=jnp.int32)
        sliding = self.coupling.sliding_coupling
        if sliding is None:
            route_left = jnp.arange(donor_indices.size, dtype=jnp.int32)
            route_right = jnp.arange(cells.size, dtype=jnp.int32)
            route_fraction = jnp.ones((cells.size,), dtype=reduction_dtype)
            if donor_indices.shape[0] != cells.shape[0]:
                raise ValueError(
                    "stage.overset.face_routes must pair donors and receptors without sliding; got different route counts."
                )
        else:
            route_left = jnp.asarray(sliding.left_routes, dtype=jnp.int32)
            route_right = jnp.asarray(sliding.right_routes, dtype=jnp.int32)
            route_fraction = (
                self.precision.reduction(sliding.overlap_measures)
                / (self.precision.reduction(sliding.right_measures)[route_right])
            )
        route_donor_cells = donor_indices[route_left]
        route_points = points_array[route_right]
        prepared_reconstruction = self._prepare_stage_reconstruction(
            reconstructed, metrics
        )
        traces = self._evaluate_stage_trace_pair(
            prepared_reconstruction, route_donor_cells, route_points, cells, points_array
        )
        route_donor_trace, receptor_interior = traces.left, traces.right
        if sliding is None:
            donor_trace = route_donor_trace
        else:
            weighted_donor_trace = (
                route_donor_trace
                * self.precision.reconstruction(sliding.overlap_measures)[:, None, None]
            )
            donor_trace = jnp.zeros(
                (
                    cells.shape[0],
                    points_array.shape[1],
                    reconstructed.shape[-1],
                ),
                dtype=weighted_donor_trace.dtype,
            )
            donor_trace = donor_trace.at[route_right].add(weighted_donor_trace)
            donor_trace = (
                donor_trace
                / self.precision.reconstruction(sliding.right_measures)[:, None, None]
            )

        site_shape = points_array.shape[:-1]
        normal_flux = self._stage_normal_flux(
            self.precision.flux(receptor_interior),
            self.precision.flux(donor_trace),
            self.precision.flux(normals_array),
            grid_velocity,
            jnp.broadcast_to(self.precision.flux(face_measures)[:, None], site_shape),
            jnp.ones(site_shape, dtype=jnp.bool_),
            metrics.geometry_layout_id,
            args,
        )
        integrated_face_flux = ein.contract(
            "fq,fqc->fc",
            measures_array,
            self.precision.reduction(normal_flux.normal_flux),
        )
        # The Riemann solve is receptor-left with a receptor-outward normal,
        # whereas the conservative ledger route is donor-owner to
        # receptor-neighbor. Canonicalize to that opposite positive direction.
        route_flux = -integrated_face_flux[route_right] * route_fraction[:, None]
        if self.coupling.vof is not None:
            match self.system:
                case multiphase_owner.TwoMaterialVOFSystem() as system:
                    phase_flux = self._overset_phase_flux(
                        system,
                        self.coupling.vof,
                        stage_plic,
                        route_flux,
                        route_donor_trace,
                        route_donor_cells,
                        route_points,
                        normals_array,
                        grid_velocity,
                        measures_array,
                        route_right,
                        route_fraction,
                    )
                    route_flux = phase_flux.flux_rate
                case _:
                    raise TypeError(
                        "stage.overset.system must declare TwoMaterialVOFSystem for VOF transport; got another system kind."
                    )
        route_flux = jnp.where(
            template.active_mask[:, None],
            route_flux,
            jnp.zeros((), dtype=reduction_dtype),
        )
        face_speed = jnp.max(
            self.precision.reduction(normal_flux.max_speed),
            axis=1,
        )
        route_speed = self.precision.decision(
            jnp.where(template.active_mask, face_speed[route_right], 0.0)
        )
        route_measures = face_measures[route_right] * route_fraction
        return _OversetCorrection(
            template.with_flux_rate(route_flux), route_speed, route_measures
        )

    def _append_overset_block(
        self,
        ledger: ConservationStageLedger,
        block: ConservationStageFluxRateBlock | None,
        /,
    ) -> ConservationStageLedger:
        if block is None:
            return ledger
        return ConservationStageLedger(
            (*ledger.blocks, block),
            ledger.source_rate,
            ledger.active_cell_mask,
            geometry_family_id=ledger.geometry_family_id,
            geometry_layout_id=ledger.geometry_layout_id,
            geometry_version=ledger.geometry_version,
            evidence_policy_id=ledger.evidence_policy_id,
            evidence_version=ledger.evidence_version,
            topology_epoch_id=ledger.topology_epoch_id,
        )

    def _append_redistribution_block(
        self,
        ledger: ConservationStageLedger,
        redistribution: ConservativeSmallCellRedistributionPlan | None,
        /,
    ) -> ConservationStageLedger:
        embedded = self.coupling.embedded_metrics is not None
        if embedded != (redistribution is not None):
            raise ValueError(
                f"stage.redistribution must be present exactly for embedded geometry; got embedded={embedded}, redistribution={type(redistribution).__name__}."
            )
        if redistribution is None:
            return ledger
        if redistribution.evidence.prepared_geometry_id != (
            self.discretization.prepared_id
        ):
            raise ValueError(
                "stage.redistribution must bind the prepared geometry; got a different discretization ID."
            )
        block = redistribution.redistribution_flux_rate_block(
            ledger.scatter_content_rate()
        )
        if block is None:
            return ledger
        return ConservationStageLedger(
            (*ledger.blocks, block),
            ledger.source_rate,
            ledger.active_cell_mask,
            geometry_family_id=ledger.geometry_family_id,
            geometry_layout_id=ledger.geometry_layout_id,
            geometry_version=ledger.geometry_version,
            evidence_policy_id=ledger.evidence_policy_id,
            evidence_version=ledger.evidence_version,
            topology_epoch_id=ledger.topology_epoch_id,
        )

    @checked
    def zero_stage_ledger(
        self,
        metrics: FiniteVolumeStageMetrics,
        /,
        *,
        redistribution: ConservativeSmallCellRedistributionPlan | None = None,
    ) -> ConservationStageLedger:
        """Create a routed zero rate without invoking reconstruction or physics."""

        blocks = self._stage_rate_templates(metrics)
        ledger = ConservationStageLedger(
            blocks,
            jnp.zeros(
                (metrics.cell_count, self.discretization.component_count),
                dtype=jnp.dtype(self.precision.reduction_dtype),
            ),
            metrics.active_cell_mask,
            geometry_family_id=metrics.geometry_family_id,
            geometry_layout_id=metrics.geometry_layout_id,
            geometry_version=metrics.geometry_version,
            evidence_policy_id=metrics.evidence.policy_id,
            evidence_version=metrics.evidence.evidence_version,
            topology_epoch_id=metrics.topology_epoch_id,
        )
        zero_overset = (
            None
            if self.overset_rate_block_template is None
            else self.overset_rate_block_template.with_flux_rate(
                jnp.zeros_like(self.overset_rate_block_template.flux_rate)
            )
        )
        ledger = self._append_overset_block(ledger, zero_overset)
        return self._append_redistribution_block(ledger, redistribution)

    def _prepare_vof_stage(
        self, average: Array, metrics: FiniteVolumeStageMetrics, /
    ) -> _VOFStage | None:
        vof_plan = self.coupling.vof
        if vof_plan is not None:
            match self.system:
                case multiphase_owner.TwoMaterialVOFSystem() as system:
                    pass
                case _:
                    raise TypeError(
                        "stage.vof.system must declare TwoMaterialVOFSystem; got another system kind."
                    )
            alpha = average[:, system.layout.alpha_index]
            normal_override = None
            override_mask = None
            contact_angles = self.coupling.contact_angles
            embedded_metrics = self.coupling.embedded_metrics
            if contact_angles is not None:
                if embedded_metrics is None:
                    raise ValueError(
                        "stage.contact_angles.embedded_metrics must be present; got None."
                    )
                contact_angles.validate_bindings(
                    embedded_metrics.geometry_id,
                    vof_plan.plan_id,
                )
                adjusted = vof_plan.interface_normals(alpha)
                override_mask = embedded_metrics.cut_face_active
                for body_tag, condition in zip(
                    contact_angles.body_tags,
                    contact_angles.conditions,
                    strict=True,
                ):
                    mask = override_mask & (embedded_metrics.body_tags == body_tag)
                    safe_plic = jnp.where(
                        mask[:, None],
                        adjusted,
                        jnp.asarray((0.0, 1.0), dtype=adjusted.dtype),
                    )
                    safe_wall = jnp.where(
                        mask[:, None],
                        embedded_metrics.cut_face_normals,
                        jnp.asarray((1.0, 0.0), dtype=adjusted.dtype),
                    )
                    result = reconstruct_wall_interface_normal(
                        safe_plic,
                        safe_wall,
                        condition,
                        geometry_id=embedded_metrics.geometry_id,
                        plic_id=vof_plan.plan_id,
                    )
                    failed = mask & ~result.evidence.passed
                    checked_normal = eqx.error_if(
                        result.normal,
                        jnp.any(failed),
                        "Active contact-angle reconstruction evidence failed.",
                    )
                    adjusted = jnp.where(mask[:, None], checked_normal, adjusted)
                normal_override = adjusted
            stage_plic = vof_plan.reconstruct_stage(
                alpha,
                effective_geometry=embedded_metrics,
                stage_metrics=(metrics if self.coupling.motion is not None else None),
                geometry_layout_id=metrics.geometry_layout_id,
                geometry_version=metrics.geometry_version,
                normal_override=normal_override,
                override_mask=override_mask,
            )
            primitive = system.conserved_to_primitive(average)
            return _VOFStage(system, vof_plan, stage_plic, primitive)
        return None

    def _physical_vof_face_flux(
        self,
        material: _VOFStage,
        layout: FiniteVolumeStageFaceLayout,
        flux_rate: Array,
        normal_flux: Array,
        right: Array,
        weights: Array,
        active_quadrature: Array,
        cell_volume_divergence: Array,
        /,
    ) -> _VOFFaceFlux:
        system, vof_plan = material.system, material.plan
        stage_plic, primitive = material.plic, material.primitive
        face_ids = layout.face_ids
        total_mass_flux = flux_rate[:, 0] + flux_rate[:, 1]
        full_total_mass = (
            jnp.zeros(
                (self.discretization.face_measures.size,),
                dtype=total_mass_flux.dtype,
            )
            .at[face_ids]
            .set(total_mass_flux)
        )
        donor_apertures = vof_plan.donor_phase_apertures(full_total_mass, stage_plic)[
            face_ids
        ]
        owner = layout.owner_cells
        neighbor = layout.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)
        donor_left = total_mass_flux >= 0.0
        boundary_inflow = (neighbor < 0) & ~donor_left
        rho0 = jnp.where(
            donor_left,
            primitive[owner, 0],
            primitive[safe_neighbor, 0],
        )
        rho1 = jnp.where(
            donor_left,
            primitive[owner, 1],
            primitive[safe_neighbor, 1],
        )
        mixture_density = donor_apertures[:, 0] * rho0 + donor_apertures[:, 1] * rho1
        checked_mixture_density = jnp.where(
            boundary_inflow,
            jnp.ones_like(mixture_density),
            mixture_density,
        )
        checked_mixture_density = eqx.error_if(
            checked_mixture_density,
            jnp.any(
                ~boundary_inflow
                & (
                    ~jnp.isfinite(checked_mixture_density)
                    | (checked_mixture_density <= system.eos.density_floor)
                )
            ),
            "PLIC donor face mixture density is invalid.",
        )
        volume_flux = total_mass_flux / checked_mixture_density
        mass0_flux = donor_apertures[:, 0] * rho0 * volume_flux
        alpha_flux = donor_apertures[:, 0] * volume_flux

        right_primitive = system.conserved_to_primitive(right)
        exterior_alpha = right_primitive[..., system.layout.alpha_index]
        exterior_rho0 = right_primitive[..., 0]
        exterior_rho1 = right_primitive[..., 1]
        exterior_mixture_density = (
            exterior_alpha * exterior_rho0 + (1.0 - exterior_alpha) * exterior_rho1
        )
        active_boundary_inflow = boundary_inflow[:, None] & active_quadrature
        exterior_mixture_density = jnp.where(
            active_boundary_inflow,
            exterior_mixture_density,
            jnp.ones_like(exterior_mixture_density),
        )
        exterior_mixture_density = eqx.error_if(
            exterior_mixture_density,
            jnp.any(
                active_boundary_inflow
                & (
                    ~jnp.isfinite(exterior_mixture_density)
                    | (exterior_mixture_density <= system.eos.density_floor)
                )
            ),
            "Exterior VOF inflow mixture density is invalid.",
        )
        exterior_total_mass_flux = normal_flux[..., 0] + normal_flux[..., 1]
        exterior_volume_flux_density = exterior_total_mass_flux / exterior_mixture_density
        exterior_volume_flux = ein.contract(
            "fq,fq->f",
            weights,
            exterior_volume_flux_density,
        )
        exterior_mass0_flux = ein.contract(
            "fq,fq->f",
            weights,
            exterior_alpha * exterior_rho0 * exterior_volume_flux_density,
        )
        exterior_alpha_flux = ein.contract(
            "fq,fq->f",
            weights,
            exterior_alpha * exterior_volume_flux_density,
        )
        volume_flux = jnp.where(
            boundary_inflow,
            exterior_volume_flux,
            volume_flux,
        )
        mass0_flux = jnp.where(
            boundary_inflow,
            exterior_mass0_flux,
            mass0_flux,
        )
        alpha_flux = jnp.where(
            boundary_inflow,
            exterior_alpha_flux,
            alpha_flux,
        )
        mass1_flux = total_mass_flux - mass0_flux
        flux_rate = flux_rate.at[:, 0].set(mass0_flux)
        flux_rate = flux_rate.at[:, 1].set(mass1_flux)
        flux_rate = flux_rate.at[:, system.layout.alpha_index].set(alpha_flux)
        cell_volume_divergence = cell_volume_divergence.at[owner].add(volume_flux)
        cell_volume_divergence = cell_volume_divergence.at[safe_neighbor].add(
            jnp.where(neighbor >= 0, -volume_flux, 0.0)
        )
        return _VOFFaceFlux(flux_rate, cell_volume_divergence)

    def _evaluate_stage_face_rate(
        self,
        geometry_block: FiniteVolumeStageFaceBlock,
        template: ConservationStageFluxRateBlock,
        trace: _FaceStageState,
        material: _VOFStage | None,
        metrics: FiniteVolumeStageMetrics,
        cell_volume_divergence: Array,
        args: Any,
        /,
    ) -> _StageFaceRate:
        left, right, normal = trace.left, trace.right, trace.normal
        grid_velocity = geometry_block.quadrature_grid_normal_velocity
        if grid_velocity is None:
            if self.coupling.motion is not None:
                raise ValueError(
                    "stage.metrics.grid_normal_velocity must be present for moving geometry; got None."
                )
            grid_velocity = jnp.zeros_like(geometry_block.quadrature_weights)
        site_shape = geometry_block.quadrature_weights.shape
        route_active = self._stage_route_active(
            geometry_block.layout, metrics.active_cell_mask
        )
        result = self._stage_normal_flux(
            self.precision.flux(left),
            self.precision.flux(right),
            self.precision.flux(normal),
            self.precision.flux(grid_velocity),
            jnp.broadcast_to(
                self.precision.flux(geometry_block.face_measures)[:, None],
                site_shape,
            ),
            jnp.broadcast_to(route_active[:, None], site_shape),
            metrics.geometry_layout_id,
            args,
        )
        layout = geometry_block.layout
        active_quadrature = template.active_mask[:, None]
        normal_flux = jnp.where(
            active_quadrature[..., None],
            self.precision.reduction(result.normal_flux),
            jnp.zeros((), dtype=jnp.dtype(self.precision.reduction_dtype)),
        )
        signal_speed = jnp.where(
            active_quadrature,
            self.precision.reduction(result.max_speed),
            jnp.zeros((), dtype=jnp.dtype(self.precision.reduction_dtype)),
        )
        weights = self.precision.reduction(geometry_block.quadrature_weights)
        flux_rate = ein.contract(
            "fq,fqc->fc",
            weights,
            normal_flux,
        )
        if material is not None and layout.block_kind == "physical":
            phase_flux = self._physical_vof_face_flux(
                material,
                layout,
                flux_rate,
                normal_flux,
                right,
                weights,
                active_quadrature,
                cell_volume_divergence,
            )
            flux_rate = phase_flux.flux_rate
            cell_volume_divergence = phase_flux.volume_divergence
        face_rate = ein.contract(
            "fq,fq->f",
            weights,
            signal_speed,
        )
        return _StageFaceRate(
            template.with_flux_rate(flux_rate),
            self.precision.decision(signal_speed),
            face_rate,
            cell_volume_divergence,
        )

    def _integrate_stage_source(
        self,
        average: Array,
        metrics: FiniteVolumeStageMetrics,
        material: _VOFStage | None,
        cell_volume_divergence: Array,
        args: Any,
        /,
    ) -> _StageSourceRate:
        source_rate = jnp.zeros_like(
            self.precision.reduction(average),
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        if self.source is not None and self.source_cell_indices.shape[0] > 0:
            source_average = average[self.source_cell_indices]
            source_centers = metrics.cell_centers[self.source_cell_indices]
            source_value = self.precision.flux(
                self.source(
                    self.precision.decision(metrics.time),
                    self.precision.flux(source_average),
                    self.precision.flux(source_centers),
                    args,
                )
            )
            if source_value.shape != source_average.shape:
                raise ValueError(
                    "stage.source must match the active state shape; got an incompatible shape."
                )
            active_source_rate = self.precision.reduction(
                source_value
            ) * self.precision.reduction(
                metrics.effective_cell_volumes[self.source_cell_indices, None]
            )
            source_rate = source_rate.at[self.source_cell_indices].set(active_source_rate)
        if material is not None:
            system = material.system
            alpha = average[:, system.layout.alpha_index]
            volumes = self.precision.reduction(metrics.effective_cell_volumes)
            divergence = cell_volume_divergence / volumes
            alpha_source = system.volume_fraction_source(
                alpha,
                divergence,
                average,
            )
            source_rate = source_rate.at[:, system.layout.alpha_index].add(
                self.precision.reduction(alpha_source) * volumes
            )
        return _StageSourceRate(source_rate)

    def _integrate_phase_change_source(
        self,
        average: Array,
        metrics: FiniteVolumeStageMetrics,
        material: _VOFStage | None,
        source_rate: Array,
        /,
    ) -> _PhaseChangeSourceRate:
        phase_change_step = jnp.asarray(jnp.inf, dtype=average.dtype)
        thermal_step = jnp.asarray(jnp.inf, dtype=average.dtype)
        phase_change = self.coupling.phase_change
        if phase_change is not None:
            match material:
                case _VOFStage():
                    stage_plic, system = material.plic, material.system
                case None:
                    raise ValueError(
                        "stage.phase_change.material must contain PLIC reconstruction; got no VOF stage."
                    )
                case _:
                    raise TypeError(
                        "stage.material must contain VOF reconstruction; got unsupported material."
                    )
            volumes = self.precision.reduction(metrics.effective_cell_volumes)
            area_density = self.precision.flux(stage_plic.interface_measures / volumes)
            heat_flux0 = jnp.zeros_like(area_density)
            heat_flux1 = jnp.zeros_like(area_density)
            thermal_diffusion = phase_change.thermal_diffusion
            if thermal_diffusion is not None:
                thermal = thermal_diffusion.evaluate(
                    system, average, stage_metrics=metrics
                )
                source_rate = source_rate.at[:, system.layout.energy_index].add(
                    self.precision.reduction(thermal.cell_energy_rate)
                )
                thermal_step = self.precision.decision(thermal.explicit_step_restriction)
                source_rate = eqx.error_if(
                    source_rate,
                    ~thermal.successful,
                    "Two-material thermal diffusion evaluation failed.",
                )

                match phase_change.phase_change.rate_law:
                    case vof_phase_change_owner.StefanHeatFluxPhaseChangePlan():
                        stefan_flux = phase_change.stefan_heat_fluxes(
                            thermal.temperature,
                            stage_plic,
                            cell_centers=metrics.cell_centers,
                        )
                        heat_flux0 = stefan_flux.phase0_normal_heat_flux
                        heat_flux1 = stefan_flux.phase1_normal_heat_flux
                        source_rate = eqx.error_if(
                            source_rate,
                            ~stefan_flux.successful,
                            "Stefan one-sided heat-flux reconstruction failed.",
                        )
                    case vof_phase_change_owner.AbstractVOFMassTransferPlan():
                        pass
                    case _:
                        raise TypeError(
                            "phase_change.rate_law must declare a mass-transfer law; got unsupported law."
                        )
            phase_source = phase_change.phase_change.differential_source(
                self.precision.flux(average),
                interface_area_density=area_density,
                heat_flux0=heat_flux0,
                heat_flux1=heat_flux1,
            )
            source_rate = (
                source_rate
                + self.precision.reduction(phase_source.state_rate) * volumes[:, None]
            )
            phase_change_step = self.precision.decision(
                jnp.min(phase_source.explicit_step_restriction)
            )
            source_rate = eqx.error_if(
                source_rate,
                ~jnp.all(phase_source.successful),
                "VOF phase-change source evaluation failed.",
            )
        return _PhaseChangeSourceRate(source_rate, phase_change_step, thermal_step)

    def _integrate_capillary_source(
        self,
        average: Array,
        metrics: FiniteVolumeStageMetrics,
        material: _VOFStage | None,
        source_rate: Array,
        /,
    ) -> _CapillarySourceRate:
        capillary_step = jnp.asarray(jnp.inf, dtype=average.dtype)
        capillarity = self.coupling.capillarity
        if capillarity is not None:
            match material:
                case _VOFStage():
                    stage_plic, primitive, system = (
                        material.plic,
                        material.primitive,
                        material.system,
                    )
                case None:
                    raise ValueError(
                        "stage.capillarity.material must contain PLIC reconstruction; got no VOF stage."
                    )
                case _:
                    raise TypeError(
                        "stage.material must contain VOF reconstruction; got unsupported material."
                    )
            density = average[:, 0] + average[:, 1]
            velocity = primitive[:, 2 : 2 + system.dimension]
            alpha = average[:, system.layout.alpha_index]
            capillary_block = capillarity.face_rate_block(stage_plic, alpha, velocity)
            momentum_rate = capillary_block.cell_momentum_rate(metrics.cell_count)
            energy_rate = capillary_block.cell_energy_rate(metrics.cell_count)
            source_rate = source_rate.at[
                :, system.layout.momentum_start : system.layout.momentum_stop
            ].add(self.precision.reduction(momentum_rate))
            source_rate = source_rate.at[:, system.layout.energy_index].add(
                self.precision.reduction(energy_rate)
            )
            cell_size = jnp.sqrt(metrics.effective_cell_volumes)
            capillary_step = capillarity.capillary_step(
                cell_size,
                density,
                interface_active=stage_plic.interface_active,
            )
        return _CapillarySourceRate(source_rate, capillary_step)

    def _finish_stage_evaluation(
        self,
        blocks: tuple[ConservationStageFluxRateBlock, ...],
        source_rate: Array,
        metrics: FiniteVolumeStageMetrics,
        speeds: tuple[Array, ...],
        cell_rate: Array,
        overset_block: ConservationStageFluxRateBlock | None,
        redistribution: ConservativeSmallCellRedistributionPlan | None,
        cfl_: float,
        phase_change: _PhaseChangeSourceRate,
        capillary: _CapillarySourceRate,
        /,
    ) -> UnstructuredFiniteVolumeStageEvaluation:
        phase_change_step, thermal_step = (
            phase_change.phase_change_step,
            phase_change.thermal_step,
        )
        capillary_step = capillary.capillary_step
        ledger = ConservationStageLedger(
            tuple(blocks),
            source_rate,
            metrics.active_cell_mask,
            geometry_family_id=metrics.geometry_family_id,
            geometry_layout_id=metrics.geometry_layout_id,
            geometry_version=metrics.geometry_version,
            evidence_policy_id=metrics.evidence.policy_id,
            evidence_version=metrics.evidence.evidence_version,
            topology_epoch_id=metrics.topology_epoch_id,
        )
        ledger = self._append_overset_block(ledger, overset_block)
        ledger = self._append_redistribution_block(ledger, redistribution)

        cfl_volumes = self.precision.reduction(metrics.effective_cell_volumes)
        if self.coupling.embedded_metrics is not None:
            stabilization = self.coupling.embedded_stabilization_policy
            if stabilization is None:
                raise ValueError(
                    "stage.cfl.embedded_stabilization_policy must be present; got None."
                )
            cfl_volumes = jnp.maximum(
                cfl_volumes,
                self.precision.reduction(self.discretization.cell_volumes)
                * stabilization.minimum_volume_fraction,
            )
        safe_cfl_volumes = jnp.where(
            metrics.active_cell_mask,
            cfl_volumes,
            jnp.ones_like(cfl_volumes),
        )
        cell_relative_rate = self.precision.decision(
            jnp.where(
                metrics.active_cell_mask,
                cell_rate / safe_cfl_volumes,
                jnp.zeros_like(cell_rate),
            )
        )
        maximum_relative_rate = jnp.max(cell_relative_rate)
        hyperbolic_step = jnp.where(
            maximum_relative_rate > 0.0,
            cfl_ / maximum_relative_rate,
            jnp.inf,
        )
        relative_cfl_step = self.precision.decision(
            jnp.minimum(
                hyperbolic_step,
                jnp.minimum(
                    capillary_step,
                    jnp.minimum(phase_change_step, thermal_step),
                ),
            )
        )
        return UnstructuredFiniteVolumeStageEvaluation(
            ledger=ledger,
            relative_signal_speeds=tuple(speeds),
            cell_relative_rate=cell_relative_rate,
            maximum_relative_rate=maximum_relative_rate,
            relative_cfl_step=relative_cfl_step,
            precision_evidence=self.precision.evidence(),
        )

    @checked
    def evaluate_stage(
        self,
        content_state: finite_volume_content_owner.FiniteVolumeConservativeContentState,
        metrics: FiniteVolumeStageMetrics,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
        redistribution: ConservativeSmallCellRedistributionPlan | None = None,
    ) -> UnstructuredFiniteVolumeStageEvaluation:
        """Evaluate one certified quadrature-integrated ALE content rate."""
        self._require_serial_entrypoint()

        cfl_ = positive_finite_float(cfl, "stage.cfl")
        average = self._validated_stage_average(content_state, metrics)
        cell_volume_divergence = jnp.zeros(
            (metrics.cell_count,), dtype=jnp.dtype(self.precision.reduction_dtype)
        )
        material = self._prepare_vof_stage(average, metrics)
        stage_plic = None if material is None else material.plic
        traces = self._stage_face_states(average, metrics, args)
        templates = self._stage_rate_templates(metrics)
        blocks: list[ConservationStageFluxRateBlock] = []
        speeds: list[Array] = []
        cell_rate = jnp.zeros(
            (metrics.cell_count,),
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        for geometry_block, template, trace in zip(
            metrics.face_blocks,
            templates,
            traces,
            strict=True,
        ):
            face = self._evaluate_stage_face_rate(
                geometry_block,
                template,
                trace,
                material,
                metrics,
                cell_volume_divergence,
                args,
            )
            cell_volume_divergence = face.volume_divergence
            blocks.append(face.block)
            speeds.append(face.signal_speed)
            owner = geometry_block.layout.owner_cells
            neighbor = geometry_block.layout.neighbor_cells
            cell_rate = cell_rate.at[owner].add(face.relative_rate)
            cell_rate = cell_rate.at[jnp.maximum(neighbor, 0)].add(
                jnp.where(neighbor >= 0, face.relative_rate, 0.0)
            )
        overset_correction = self._overset_correction(average, metrics, args, stage_plic)
        overset_block = overset_correction.block
        overset_speed = overset_correction.signal_speed
        overset_route_measures = overset_correction.route_measures
        if overset_block is not None:
            speeds.append(overset_speed)
            overset_face_rate = overset_speed * overset_route_measures
            cell_rate = cell_rate.at[overset_block.owner_cells].add(overset_face_rate)
            cell_rate = cell_rate.at[jnp.maximum(overset_block.neighbor_cells, 0)].add(
                jnp.where(
                    overset_block.neighbor_cells >= 0,
                    overset_face_rate,
                    0.0,
                )
            )

        source = self._integrate_stage_source(
            average, metrics, material, cell_volume_divergence, args
        )
        phase_change = self._integrate_phase_change_source(
            average, metrics, material, source.source_rate
        )
        capillary = self._integrate_capillary_source(
            average, metrics, material, phase_change.source_rate
        )
        source_rate = capillary.source_rate
        return self._finish_stage_evaluation(
            tuple(blocks),
            source_rate,
            metrics,
            tuple(speeds),
            cell_rate,
            overset_block,
            redistribution,
            cfl_,
            phase_change,
            capillary,
        )

    def _centroid_face_states(
        self, time: Array, state: Array, args: Any, /
    ) -> _FaceStageState:
        owner = self.discretization.owner_cells
        neighbor = self.discretization.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)
        left = self.precision.reconstruction(state[owner])
        right = self.precision.reconstruction(state[safe_neighbor])
        if self.neighbor_state_action is not None:
            right = contract("fji,fj->fi", self.neighbor_state_action, right)
        normal = (
            self.discretization.area_vectors / self.discretization.face_measures[:, None]
        )
        boundary = neighbor < 0
        for patch_id, policy in enumerate(self.boundaries.boundaries):
            patch_mask = boundary & (self.discretization.boundary_patch_ids == patch_id)
            exterior = policy.exterior_state(
                self.system,
                self.precision.decision(time),
                left,
                self.precision.reconstruction(self.discretization.face_centers),
                self.precision.reconstruction(normal),
                0,
                args,
            )
            right = jnp.where(patch_mask[:, None], exterior, right)
        return _FaceStageState(left, right, normal)

    def _quadrature_face_states(
        self,
        reconstruction: PreparedUnstructuredReconstruction,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> _FaceStageState:
        points = self.precision.reconstruction(self.discretization.face_quadrature_points)
        left, right = reconstruction.reconstruct_at(
            self.precision.reconstruction(state), points
        )
        neighbor = self.discretization.neighbor_cells
        normal = (
            self.discretization.area_vectors / self.discretization.face_measures[:, None]
        )
        normal = jnp.broadcast_to(normal[:, None, :], points.shape)
        boundary = neighbor < 0
        for patch_id, policy in enumerate(self.boundaries.boundaries):
            patch_mask = boundary & (self.discretization.boundary_patch_ids == patch_id)
            exterior = policy.exterior_state(
                self.system,
                self.precision.decision(time),
                left,
                points,
                self.precision.reconstruction(normal),
                0,
                args,
            )
            right = jnp.where(patch_mask[:, None, None], exterior, right)
        return _FaceStageState(left, right, normal)

    def _static_normal_flux(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any,
        /,
        *,
        face_indices: Array | None = None,
    ) -> tuple[Array, Array]:
        """Stationary normal flux with the closure applied at every face site."""
        left_ = self.precision.flux(left)
        right_ = self.precision.flux(right)
        normal_ = self.precision.flux(normal)
        result = self.method.interface_solver.normal_face_flux(
            self.system, left_, right_, normal_, args
        )
        normal_flux = result.normal_flux
        if self.method.closure is not None:
            measures = self.discretization.face_measures
            if face_indices is not None:
                measures = measures[face_indices]
            context = FaceFluxContext(
                normal_,
                self.precision.flux(
                    jnp.broadcast_to(
                        measures.reshape(measures.shape + (1,) * (normal_.ndim - 2)),
                        normal_.shape[:-1],
                    )
                ),
                geometry_id=self.discretization.prepared_id,
            )
            normal_flux = self.method.closure.apply(
                self.system, left_, right_, normal_flux, context, args
            )
        return normal_flux, result.max_speed

    def face_fluxes(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, Array]:
        self._require_serial_entrypoint()
        value = jnp.asarray(state)
        if value.shape != self.discretization.state_shape:
            raise ValueError(
                f"dynamics.state must have shape {self.discretization.state_shape}; got {value.shape}."
            )
        self.precision.validate_state(value)
        match self.method.reconstruction:
            case (
                PreparedCellPolynomialReconstruction()
                | PreparedUnstructuredWENOZReconstruction() as reconstruction
            ):
                trace = self._quadrature_face_states(reconstruction, time, value, args)
                left, right, normal = trace.left, trace.right, trace.normal
                normal_flux, max_speed = self._static_normal_flux(
                    left, right, normal, args
                )
                integrated = jnp.sum(
                    self.precision.reduction(
                        self.discretization.face_quadrature_weights[..., None]
                    )
                    * self.precision.reduction(normal_flux),
                    axis=1,
                )
                average_flux = integrated / self.precision.reduction(
                    self.discretization.face_measures[:, None]
                )
                return self.precision.flux(average_flux), self.precision.decision(
                    jnp.max(max_speed, axis=1)
                )
            case PiecewiseConstantReconstruction():
                trace = self._centroid_face_states(time, value, args)
                left, right, normal = trace.left, trace.right, trace.normal
                normal_flux, max_speed = self._static_normal_flux(
                    left, right, normal, args
                )
                return self.precision.flux(normal_flux), self.precision.decision(
                    max_speed
                )
            case _:
                raise TypeError(
                    "dynamics.reconstruction must be a supported reconstruction; got an unsupported kind."
                )

    def _content_rate_from_fluxes(
        self, normal_flux: Array, /, *, active_mask: Array | None
    ) -> Array:
        integrated = self.precision.reduction(normal_flux) * self.precision.reduction(
            self.discretization.face_measures[:, None]
        )
        if active_mask is not None:
            integrated = jnp.where(active_mask[:, None], integrated, 0.0)
        owner = self.discretization.owner_cells
        neighbor = self.discretization.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)
        residual = jnp.zeros(
            self.discretization.state_shape,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        residual = residual.at[owner].add(-integrated)
        residual = residual.at[safe_neighbor].add(
            jnp.where(
                (neighbor >= 0)[:, None],
                integrated
                if self.neighbor_state_action is None
                else contract("fij,fj->fi", self.neighbor_state_action, integrated),
                0.0,
            )
        )
        return residual

    def residual_from_fluxes(self, normal_flux: Array, /) -> Array:
        self._require_serial_entrypoint()
        return self.precision.storage(
            self._content_rate_from_fluxes(
                normal_flux, active_mask=self.discretization.face_block.active_mask
            )
            / self.precision.reduction(self.discretization.cell_volumes[:, None])
        )

    def _require_serial_entrypoint(self) -> None:
        match self.discretization:
            case UnstructuredFiniteVolumeDiscretization() if (
                self.discretization.partition_count > 1
            ):
                raise ValueError(
                    "dynamics.entrypoint must use owner-local execution with certified named-axis halo communication; got a serial entrypoint for partitioned geometry."
                )
            case (
                UnstructuredFiniteVolumeDiscretization()
                | DyadicFiniteVolumeDiscretization()
            ):
                pass
            case _:
                raise TypeError(
                    "dynamics.discretization must declare unstructured or dyadic geometry; got an unsupported kind."
                )

    @checked
    def owner_local_residual(
        self,
        time: ArrayLike,
        state: ArrayLike,
        halo: DistributedHaloPlan,
        /,
        *,
        axis_name: str,
        args: Any = None,
    ) -> Array:
        """Exactly-once owned-facet solve and conservative owned-cell rate.

        Call inside a real named-axis map. Halo values are refreshed from
        their owners, and equal/opposite ghost contributions are returned
        through the halo transpose; no global field is materialized. Rows
        outside ``halo.local_valid`` are inactive execution padding: physics
        reads a real closure row there and their rate is zero. Halo/ID
        disagreement on any partition refuses every partition.
        """
        geometry = _owner_local_geometry(self)
        if halo.local_global_ids.ndim != 1:
            raise TypeError(
                "dynamics.halo.local_global_ids must have an owner-local rank-one layout; got another layout."
            )
        if (
            halo.entity_count != geometry.global_entity_counts[geometry.cell_dimension]
            or halo.part_count != geometry.partition_count
            or halo.local_global_ids.shape != geometry.cell_global_ids.shape
            or halo.evidence_id != geometry.closure_evidence_id
        ):
            raise ValueError(
                "dynamics.halo must agree with logical cell storage and closure evidence; got mismatched metadata."
            )
        value = jnp.asarray(state)
        if value.shape != geometry.state_shape:
            raise ValueError(
                "dynamics.state must match the local closure shape; got another shape."
            )
        self.precision.validate_state(value)
        route_valid = halo.partition_index == geometry.partition_index
        owners = geometry.cell_owner
        for phase, permutation in enumerate(halo.permutations):
            receive = halo.phase_receive_indices[phase]
            expected = jnp.asarray(-1, dtype=jnp.int32)
            for source, target in permutation:
                expected = jnp.where(geometry.partition_index == target, source, expected)
            route_valid = route_valid & jnp.all(
                ~halo.phase_receive_valid[phase] | (owners[receive] == expected)
            )
        local_valid = (
            ~jnp.any(halo.local_global_ids != geometry.cell_global_ids)
            & ~jnp.any(halo.local_owned != geometry.cell_owned)
            & route_valid
        )
        value = eqx.error_if(
            value,
            (jax.lax.pmin(local_valid.astype(jnp.int32), axis_name) == 0)
            | ~halo.collective_certificate(axis_name=axis_name),
            "Owner-local FV halo ownership, stable IDs, or collective coverage is invalid.",
        )
        part = jnp.asarray(geometry.partition_index, dtype=jnp.int32)
        exchanged = halo.exchange(value, part, axis_name=axis_name)
        # Valid rows are a nonempty prefix; masked padding lanes evaluate the
        # physics at row zero, so they stay finite in value and derivative.
        exchanged = jnp.where(halo.local_valid[:, None], exchanged, exchanged[0])
        time_ = jnp.asarray(time)
        trace = self._centroid_face_states(time_, exchanged, args)
        left, right, normal = trace.left, trace.right, trace.normal
        faces = geometry.owned_face_indices
        if geometry.owned_face_buffered:

            def solve_slot(slot: tuple[Array, Array]) -> Array:
                face, active = slot

                def solve(index: Array) -> Array:
                    selected = index[None]
                    flux_, _ = self._static_normal_flux(
                        left[selected],
                        right[selected],
                        normal[selected],
                        args,
                        face_indices=selected,
                    )
                    return self.precision.flux(flux_[0])

                return jax.lax.cond(
                    active,
                    solve,
                    lambda _: jnp.zeros(
                        (geometry.component_count,),
                        dtype=jnp.dtype(self.precision.flux_dtype),
                    ),
                    face,
                )

            flux = jax.lax.map(solve_slot, (faces, geometry.owned_face_valid))
        else:
            flux, _ = self._static_normal_flux(
                left[faces], right[faces], normal[faces], args, face_indices=faces
            )
        face_flux = (
            jnp.zeros(
                (geometry.face_measures.size, geometry.component_count),
                dtype=flux.dtype,
            )
            .at[faces]
            .add(
                jnp.where(
                    geometry.owned_face_valid[:, None],
                    flux,
                    jnp.zeros((), dtype=flux.dtype),
                )
            )
        )
        # The sparse buffer already contains only this owner's valid facets;
        # the unmapped face block can retain another owner's activity mask.
        content_rate = self._content_rate_from_fluxes(face_flux, active_mask=None)
        accumulated = halo.accumulate_halo(content_rate, part, axis_name=axis_name)
        residual = accumulated / self.precision.reduction(geometry.cell_volumes[:, None])
        residual = residual + self.precision.reduction(
            self.source_value(time_, exchanged, args)
        )
        return self.precision.storage(
            jnp.where(
                geometry.cell_owned[:, None],
                residual,
                jnp.zeros((), dtype=residual.dtype),
            )
        )

    @checked
    def owner_local_advance(
        self,
        time: ArrayLike,
        state: ArrayLike,
        step: ArrayLike,
        halo: DistributedHaloPlan,
        /,
        *,
        axis_name: str,
        args: Any = None,
    ) -> Array:
        """Advance owned cell averages; closure ghosts remain read-only."""
        value = jnp.asarray(state)
        rate = self.owner_local_residual(
            time, value, halo, axis_name=axis_name, args=args
        )
        return self.precision.storage(value + jnp.asarray(step, dtype=value.dtype) * rate)

    def source_value(self, time: Array, state: Array, args: Any, /) -> Array:
        if self.source is None:
            return jnp.zeros_like(state)
        value = self.precision.flux(
            self.source(
                self.precision.decision(time),
                self.precision.flux(state),
                self.precision.flux(self.discretization.cell_centers),
                args,
            )
        )
        if value.shape != state.shape:
            raise ValueError(
                "dynamics.source must match the state shape; got another shape."
            )
        return self.precision.storage(value)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        flux, _ = self.face_fluxes(time, state, args)
        return self.precision.storage(
            self.precision.reduction(self.residual_from_fluxes(flux))
            + self.precision.reduction(self.source_value(time, state, args))
        )

    def _cell_rate(self, speed: Array, /) -> Array:
        weighted = self.precision.reduction(speed) * self.precision.reduction(
            self.discretization.face_measures
        )
        rate = jnp.zeros(
            (self.discretization.cell_count,),
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        rate = rate.at[self.discretization.owner_cells].add(weighted)
        neighbor = self.discretization.neighbor_cells
        rate = rate.at[jnp.maximum(neighbor, 0)].add(
            jnp.where(neighbor >= 0, weighted, 0.0)
        )
        return self.precision.decision(
            rate / self.precision.reduction(self.discretization.cell_volumes)
        )

    def stable_step(
        self,
        state: Array,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        self.precision.validate_state(state)
        _, speed = self.face_fluxes(jnp.asarray(0.0), state, args)
        maximum = jnp.max(self._cell_rate(speed))
        return self.precision.decision(
            jnp.where(maximum > 0.0, float(cfl) / maximum, jnp.inf)
        )

    def linearize(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[
        Array,
        Callable[[Array], Array],
        Callable[[Array], tuple[Array]],
    ]:
        linearization = la.prepare_linearization(
            lambda value: self(time, value, args), state
        )
        return (
            linearization.primal,
            linearization.pushforward,
            lambda cotangent: (linearization.pullback(cotangent),),
        )

    def residual_with_diagnostics(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, UnstructuredFiniteVolumeDiagnostics]:
        flux, speed = self.face_fluxes(time, state, args)
        source = self.source_value(time, state, args)
        residual = self(time, state, args)
        boundary = self.discretization.neighbor_cells < 0
        integrated = self.precision.reduction(flux) * self.precision.reduction(
            self.discretization.face_measures[:, None]
        )
        boundary_terms = jnp.where(boundary[:, None], integrated, 0.0)
        source_terms = self.precision.reduction(
            self.discretization.cell_volumes[:, None] * source
        )
        change_terms = self.precision.reduction(
            self.discretization.cell_volumes[:, None] * residual
        )
        boundary_flux = compensated_sum(boundary_terms, axis=0)
        source_integral = compensated_sum(source_terms, axis=0)
        defect = compensated_sum_chunks(
            (change_terms, -source_terms, boundary_terms),
            output_ndim=1,
        )
        return self.precision.storage(residual), UnstructuredFiniteVolumeDiagnostics(
            normal_flux=flux,
            signal_speed=speed,
            boundary_outward_flux=boundary_flux,
            source_integral=source_integral,
            conservation_defect=defect,
            maximum_rate=jnp.max(self._cell_rate(speed)),
            precision_evidence=self.precision.evidence(),
        )


class OwnerLocalFiniteVolumeAdvance(StrictModule):
    """Retain an owner-local advance and globally reduced owned content.

    ``state`` keeps the execution layout of its
    `OwnerLocalFiniteVolumeExecution`: logical closure rows prefix every
    partition bucket and padding rows are zero. Byte counts describe arrays,
    not a measured allocator peak: ``logical_state_bytes`` counts the logical
    closure rows of every partition, ``execution_state_bytes`` the padded
    global state and ``process_execution_bytes`` this process's padded
    execution leaves. Admissibility is meaningful only when
    ``admissibility_checked`` is true.
    """

    state: Array
    inventory_before: Array
    inventory_after: Array
    inventory_change: Array
    finite: Array
    admissible: Array
    admissibility_checked: bool = eqx.field(static=True)
    process_execution_bytes: int = eqx.field(static=True)
    logical_state_bytes: int = eqx.field(static=True)
    execution_state_bytes: int = eqx.field(static=True)


_OWNER_GEOMETRY_FIELDS = (
    "cell_volumes",
    "cell_centers",
    "face_centers",
    "area_vectors",
    "face_measures",
    "owner_cells",
    "neighbor_cells",
    "boundary_patch_ids",
    "cell_owned",
    "cell_owner",
    "cell_global_ids",
    "owned_face_indices",
    "owned_face_valid",
    "partition_index",
)
# Descriptor layout: closure cells, faces, owned-facet slots, halo message
# entries and valid owned facets; one refusal flag per reason; family digest.
_OWNER_COUNTS = 5
_OWNER_REFUSALS = (
    "a local program admission failure",
    "halo and closure stable IDs, ownership, evidence, or routes that disagree",
    "dynamics outside the uncoupled piecewise-constant owner-local route",
    "an empty closure",
)
_OWNER_DIGEST = 10


class _OwnerExecutionData(StrictModule):
    system: AbstractConservationSystem
    method: UnstructuredFiniteVolumeMethodPlan
    boundaries: UnstructuredFiniteVolumeBoundarySet
    precision: FiniteVolumePrecisionPolicy
    cell_volumes: Array
    cell_centers: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    owner_cells: Array
    neighbor_cells: Array
    boundary_patch_ids: Array
    cell_owned: Array
    cell_owner: Array
    cell_global_ids: Array
    owned_face_indices: Array
    owned_face_valid: Array
    partition_index: Array
    neighbor_state_action: Array | None
    halo: DistributedHaloPlan


class _OwnerExecutionDescriptor(StrictModule):
    values: Array
    family: str = eqx.field(static=True)


def _owner_local_geometry(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics, /
) -> UnstructuredFiniteVolumeDiscretization:
    match dynamics.discretization:
        case UnstructuredFiniteVolumeDiscretization() as geometry:
            return geometry
        case DyadicFiniteVolumeDiscretization():
            raise TypeError(
                "owner_local.programs.geometry must contain CellMesh finite-volume geometry; got another geometry kind."
            )
        case _:
            raise TypeError(
                "owner_local.programs.geometry must contain CellMesh finite-volume geometry; got another geometry kind."
            )


def _prepare_owner_program_placement(
    programs: tuple[
        tuple[PreparedUnstructuredFiniteVolumeDynamics, DistributedHaloPlan], ...
    ],
    local_parts: tuple[int, ...],
    parts: int,
    /,
) -> tuple[tuple[PreparedUnstructuredFiniteVolumeDynamics, DistributedHaloPlan], ...]:
    if not programs:
        raise ValueError(
            "owner_local.programs must contain addressable owner programs; got an empty tuple."
        )
    by_part: dict[
        int, tuple[PreparedUnstructuredFiniteVolumeDynamics, DistributedHaloPlan]
    ] = {}
    for dynamics, halo in programs:
        match dynamics:
            case PreparedUnstructuredFiniteVolumeDynamics():
                pass
            case _:
                raise TypeError(
                    "owner_local.programs.dynamics must contain prepared finite-volume dynamics; got another kind."
                )
        geometry = _owner_local_geometry(dynamics)
        match halo:
            case DistributedHaloPlan() if halo.owner_local:
                pass
            case _:
                raise TypeError(
                    "owner_local.programs.halo must contain an owner-local sparse halo; got another kind or mode."
                )
        rank = int(owner_local_host_array(geometry.partition_index))
        if rank in by_part:
            raise ValueError(
                "owner_local.programs must have unique semantic partition indices; got a duplicate index."
            )
        by_part[rank] = (dynamics, halo)
    if set(by_part) != set(local_parts):
        raise ValueError(
            "owner_local.programs must cover the addressable group ranks; got a mismatched partition set."
        )
    ordered = tuple(by_part[rank] for rank in local_parts)
    if any(
        _owner_local_geometry(dynamics).partition_count != parts
        for dynamics, _ in ordered
    ):
        raise ValueError(
            "owner_local.execution_group must preserve the certified partition count; got a different count."
        )
    return ordered


def _owner_closure_identity(
    geometry: UnstructuredFiniteVolumeDiscretization, halo: DistributedHaloPlan, /
) -> bool:
    """Host check that a logical halo matches closure IDs, ownership and routes."""
    ids = owner_local_host_array(halo.local_global_ids, dtype=np.int64)
    cell_ids = owner_local_host_array(geometry.cell_global_ids, dtype=np.int64)
    index = int(owner_local_host_array(geometry.partition_index))
    if (
        halo.part_count != geometry.partition_count
        or halo.entity_count != geometry.global_entity_counts[geometry.cell_dimension]
        or halo.evidence_id != geometry.closure_evidence_id
        or int(owner_local_host_array(halo.partition_index)) != index
        or ids.shape != cell_ids.shape
        or not np.all(owner_local_host_array(halo.local_valid, dtype=np.bool_))
        or not np.array_equal(ids, cell_ids)
        or not np.array_equal(
            owner_local_host_array(halo.local_owned, dtype=np.bool_),
            owner_local_host_array(geometry.cell_owned, dtype=np.bool_),
        )
    ):
        return False
    owners = owner_local_host_array(geometry.cell_owner, dtype=np.int64)
    receive = owner_local_host_array(halo.phase_receive_indices)
    receive_valid = owner_local_host_array(halo.phase_receive_valid, dtype=np.bool_)
    return all(
        np.all(
            owners[receive[phase][receive_valid[phase]]]
            == next((source for source, target in permutation if target == index), -1)
        )
        for phase, permutation in enumerate(halo.permutations)
    )


def _owner_execution_descriptor(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    halo: DistributedHaloPlan,
    controls: tuple[str | int, ...],
    /,
) -> _OwnerExecutionDescriptor:
    geometry = _owner_local_geometry(dynamics)
    physics = jax.tree_util.tree_leaves(
        eqx.filter(
            (dynamics.system, dynamics.method, dynamics.boundaries, dynamics.precision),
            eqx.is_array,
        )
    )
    geometric = tuple(getattr(geometry, name) for name in _OWNER_GEOMETRY_FIELDS) + (
        ()
        if dynamics.neighbor_state_action is None
        else (dynamics.neighbor_state_action,)
    )
    family = canonical_fingerprint(
        {
            "kind": "owner-local-finite-volume-execution-family",
            "controls": controls,
            "system": (type(dynamics.system).__qualname__, dynamics.system.system_id),
            "method": dynamics.method.method_id,
            "boundaries": dynamics.boundaries.boundary_set_id,
            "precision": dynamics.precision.policy_id,
            "source": dynamics.source_id,
            "geometry": (
                geometry.cell_dimension,
                geometry.component_names,
                geometry.boundary_patch_names,
                geometry.owned_face_buffered,
                geometry.partition_count,
                geometry.global_entity_counts,
                geometry.closure_evidence_id,
            ),
            "halo": (
                halo.permutations,
                halo.entity_count,
                halo.part_count,
                halo.evidence_id,
            ),
            "physics_leaves": [(tuple(leaf.shape), str(leaf.dtype)) for leaf in physics],
            "geometry_leaves": [
                (tuple(leaf.shape[1:]), str(leaf.dtype)) for leaf in geometric
            ],
        }
    )
    counts = (
        geometry.cell_count,
        geometry.face_measures.size,
        geometry.owned_face_indices.size,
        halo.message_capacity,
        int(np.count_nonzero(owner_local_host_array(geometry.owned_face_valid))),
    )
    # Owner-local advances solve centroid piecewise-constant traces only.
    supported = (
        type(dynamics.method.reconstruction) is PiecewiseConstantReconstruction
        and dynamics.coupling.plan_id == UnstructuredFiniteVolumeCouplingPlan().plan_id
    )
    refusals = (
        0,
        int(not _owner_closure_identity(geometry, halo)),
        int(not supported),
        int(geometry.cell_count == 0),
    )
    digest = tuple(int(family[offset : offset + 7], 16) for offset in range(0, 64, 7))
    return _OwnerExecutionDescriptor(
        jnp.asarray(counts + refusals + digest, dtype=jnp.int64), family
    )


def _agree_owner_execution(
    descriptors: tuple[_OwnerExecutionDescriptor, ...],
    local_parts: tuple[int, ...],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> np.ndarray:
    """The single preparation sync: maxima, totals, refusals and agreement."""
    packed = make_owner_local_field_array(
        tuple(descriptor.values for descriptor in descriptors),
        local_parts,
        execution_group.devices,
        axis_name=axis_name,
    )
    flags = _OWNER_COUNTS + len(_OWNER_REFUSALS)

    def admit(value: Array) -> Array:
        local = value[0]
        counts, digest = local[:_OWNER_COUNTS], local[flags:]
        agreed = jnp.all(
            jax.lax.pmin(digest, axis_name) == jax.lax.pmax(digest, axis_name)
        )
        return jnp.concatenate(
            (
                jax.lax.pmax(counts, axis_name),
                jax.lax.psum(counts, axis_name),
                jax.lax.pmax(local[_OWNER_COUNTS:flags], axis_name),
                agreed.astype(local.dtype)[None],
            )
        )

    summary = jax.shard_map(
        admit,
        mesh=execution_group.mesh,
        in_specs=PartitionSpec(axis_name),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(packed)
    return replicated_host_array(summary).astype(np.int64)


def _padded_execution_data(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    halo: DistributedHaloPlan,
    capacities: tuple[int, int, int, int],
    /,
) -> _OwnerExecutionData:
    """Pad one logical closure to the agreed buckets with inactive, safe lanes.

    Padding faces join the first closure cell to no neighbor or patch with a
    unit normal and measure and are never owned-facet slots; padding cells have
    unit volume, the first cell's center, no ownership, and negative IDs equal
    to the halo bucket's. Nothing is gathered from other partitions.
    """
    cells, faces, slots, messages = capacities
    geometry = _owner_local_geometry(dynamics)
    anchor = geometry.cell_centers[0]

    def pad(value: Array, size: int, fill: ArrayLike) -> Array:
        filler = jnp.broadcast_to(
            jnp.asarray(fill, dtype=value.dtype),
            (size - value.shape[0], *value.shape[1:]),
        )
        return jnp.concatenate((value, filler))

    action = dynamics.neighbor_state_action
    return _OwnerExecutionData(
        dynamics.system,
        dynamics.method,
        dynamics.boundaries,
        dynamics.precision,
        pad(geometry.cell_volumes, cells, 1),
        pad(geometry.cell_centers, cells, anchor),
        pad(geometry.face_centers, faces, anchor),
        pad(geometry.area_vectors, faces, jnp.eye(geometry.area_vectors.shape[1])[0]),
        pad(geometry.face_measures, faces, 1),
        pad(geometry.owner_cells, faces, 0),
        pad(geometry.neighbor_cells, faces, -1),
        pad(geometry.boundary_patch_ids, faces, -1),
        pad(geometry.cell_owned, cells, False),
        pad(geometry.cell_owner, cells, -1),
        jnp.concatenate(
            (
                geometry.cell_global_ids,
                jnp.arange(
                    geometry.cell_count - cells,
                    0,
                    dtype=geometry.cell_global_ids.dtype,
                ),
            )
        ),
        pad(geometry.owned_face_indices, slots, 0),
        pad(geometry.owned_face_valid, slots, False),
        geometry.partition_index,
        None if action is None else pad(action, faces, jnp.eye(action.shape[-1])),
        halo.execution_bucket(cells, messages),
    )


def _is_none(value: object, /) -> bool:
    return value is None


def _bind_owner_execution(
    skeleton: PreparedUnstructuredFiniteVolumeDynamics, data: _OwnerExecutionData, /
) -> PreparedUnstructuredFiniteVolumeDynamics:
    """Rebind the array-free prototype to one partition's padded leaves."""
    geometry = eqx.tree_at(
        lambda value: tuple(getattr(value, name) for name in _OWNER_GEOMETRY_FIELDS),
        skeleton.discretization,
        tuple(getattr(data, name) for name in _OWNER_GEOMETRY_FIELDS),
        is_leaf=_is_none,
    )
    return eqx.tree_at(
        lambda value: (
            value.system,
            value.method,
            value.boundaries,
            value.precision,
            value.discretization,
            value.neighbor_state_action,
        ),
        skeleton,
        (
            data.system,
            data.method,
            data.boundaries,
            data.precision,
            geometry,
            data.neighbor_state_action,
        ),
        is_leaf=_is_none,
    )


def _owner_execution_program(
    skeleton: PreparedUnstructuredFiniteVolumeDynamics,
    static: _OwnerExecutionData,
    structure: jax.tree_util.PyTreeDef,
    specifications: tuple[PartitionSpec, ...],
    mesh: Mesh,
    axis_name: str,
    /,
) -> Callable[..., tuple[Array, ...]]:
    state_spec = PartitionSpec(axis_name, None, None)

    def run(
        numeric: tuple[Array, ...],
        state: Array,
        time: Array,
        step: Array,
        arguments: tuple[Array, ...],
        argument_tree: jax.tree_util.PyTreeDef,
        constant: Any,
    ) -> tuple[Array, ...]:
        def advance(
            local_numeric: tuple[Array, ...],
            local_state: Array,
            local_time: Array,
            local_step: Array,
            local_arguments: tuple[Array, ...],
        ) -> tuple[Array, ...]:
            data = eqx.combine(
                jax.tree_util.tree_unflatten(
                    structure, tuple(value[0] for value in local_numeric)
                ),
                static,
            )
            dynamics = _bind_owner_execution(skeleton, data)
            halo, precision = data.halo, data.precision
            args = eqx.combine(
                jax.tree_util.tree_unflatten(argument_tree, local_arguments), constant
            )
            valid = halo.local_valid[:, None]
            before = local_state[0]
            after = jnp.where(
                valid,
                dynamics.owner_local_advance(
                    local_time, before, local_step, halo, axis_name=axis_name, args=args
                ),
                jnp.zeros((), dtype=before.dtype),
            )
            owned = data.cell_owned[:, None]
            volumes = precision.reduction(data.cell_volumes[:, None])

            def inventory(value: Array) -> Array:
                local_content = compensated_sum(
                    jnp.where(owned, volumes * precision.reduction(value), 0.0), axis=0
                )
                return jax.lax.psum(local_content, axis_name)

            def collective(flag: Array) -> Array:
                return jax.lax.pmin(flag.astype(jnp.int32), axis_name).astype(jnp.bool_)

            # Padding is judged at the residual's real anchor row, never as an
            # inactive zero state whose invalid predicate is masked away.
            judged = jnp.where(valid, after, after[0])
            return (
                after[None, ...],
                inventory(before),
                inventory(after),
                collective(jnp.all(jnp.where(valid, jnp.isfinite(after), True))),
                collective(
                    jnp.all(
                        jnp.where(data.cell_owned, data.system.admissible(judged), True)
                    )
                ),
            )

        return jax.shard_map(
            advance,
            mesh=mesh,
            in_specs=(
                specifications,
                state_spec,
                PartitionSpec(),
                PartitionSpec(),
                tuple(PartitionSpec() for _ in arguments),
            ),
            out_specs=(
                state_spec,
                PartitionSpec(None),
                PartitionSpec(None),
                PartitionSpec(),
                PartitionSpec(),
            ),
            check_vma=False,
        )(numeric, state, time, step, arguments)

    return eqx.filter_jit(run)


def _replicated(value: ArrayLike, execution_group: ExecutionGroup, /) -> Array:
    """Replicate one control from local device copies, without a host round trip."""
    sharding = execution_group.named_sharding(PartitionSpec())
    array = jnp.asarray(value)
    if array.sharding.is_equivalent_to(sharding, array.ndim):
        return array
    return jax.make_array_from_single_device_arrays(
        array.shape,
        sharding,
        [jax.device_put(array, device) for device in sharding.addressable_devices],
    )


class OwnerLocalFiniteVolumeExecution(StrictModule):
    """Collectively agreed fixed-capacity buckets for owner-local FV advances.

    Logical closure cells, faces, owned-facet slots and halo message slots are
    prefixes of buckets whose capacities are the collective maxima. Padding is
    inactive execution storage, never scientific content: negative halo IDs,
    no ownership, no owned-facet solve or message, unit placeholder measures
    and zero state; physics on padding lanes reads a real closure row, so
    masked lanes stay finite in value and derivative. ``local_cell_counts``,
    ``logical_cell_count`` and ``logical_owned_face_count`` are logical sizes;
    capacities and ``process_execution_bytes`` (this process's padded
    execution leaves) describe the execution layout. Advances reuse one
    compiled program and never synchronize with the host.
    """

    numeric: tuple[Array, ...]
    program: Callable[..., tuple[Array, ...]] = eqx.field(static=True)
    execution_group: ExecutionGroup = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    local_parts: tuple[int, ...] = eqx.field(static=True)
    local_cell_counts: tuple[int, ...] = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    owned_face_capacity: int = eqx.field(static=True)
    message_capacity: int = eqx.field(static=True)
    logical_cell_count: int = eqx.field(static=True)
    logical_owned_face_count: int = eqx.field(static=True)
    process_execution_bytes: int = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)

    def require_state(self, state: Array, /) -> None:
        if state.shape != (
            self.partition_count,
            self.cell_capacity,
            self.component_count,
        ) or jnp.dtype(state.dtype) != jnp.dtype(self.state_dtype):
            raise ValueError(
                "owner_local.state must use the padded execution layout and storage dtype; got another shape or dtype."
            )

    def pack_state(self, local_states: Sequence[ArrayLike], /) -> Array:
        """Place logical closure states, in ``local_parts`` order, into zero-padded buckets."""
        values = tuple(jnp.asarray(value) for value in local_states)
        if len(values) != len(self.local_parts) or any(
            value.shape != (count, self.component_count)
            or jnp.dtype(value.dtype) != jnp.dtype(self.state_dtype)
            for value, count in zip(values, self.local_cell_counts, strict=True)
        ):
            raise ValueError(
                "owner_local.state must contain one logical closure state per addressable partition with the storage dtype; got another count, shape, or dtype."
            )
        return make_owner_local_field_array(
            tuple(
                jnp.pad(value, ((0, self.cell_capacity - count), (0, 0)))
                for value, count in zip(values, self.local_cell_counts, strict=True)
            ),
            self.local_parts,
            self.execution_group.devices,
            axis_name=self.axis_name,
        )

    def unpack_state(self, state: Array, /) -> tuple[Array, ...]:
        """Return this process's logical closure rows in ``local_parts`` order."""
        self.require_state(state)
        placed = jax.device_put(
            state,
            self.execution_group.named_sharding(
                PartitionSpec(self.axis_name, None, None)
            ),
        )
        shards = {
            0 if shard.index[0].start is None else int(shard.index[0].start): shard.data
            for shard in placed.addressable_shards
        }
        return tuple(
            shards[part][0, :count]
            for part, count in zip(self.local_parts, self.local_cell_counts, strict=True)
        )


@checked
def prepare_owner_local_finite_volume_execution(
    programs: tuple[
        tuple[PreparedUnstructuredFiniteVolumeDynamics, DistributedHaloPlan], ...
    ],
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
    maximum_cells: int,
    maximum_faces: int,
    maximum_message_entries: int,
    maximum_storage_bytes: int,
) -> OwnerLocalFiniteVolumeExecution:
    """Agree fixed-capacity owner-local buckets with one collective host sync.

    ``programs`` are this process's logical closures with their halos, one per
    addressable execution-group partition. Every partition contributes closure
    counts, local refusal flags, and an execution-family digest to one
    collective; capacities are the agreed maxima. Disagreement, a refusal on
    any partition, or a bucket beyond the declared maxima refuses on every
    partition. ``maximum_faces`` bounds closure faces and owned-facet slots;
    ``maximum_storage_bytes`` bounds every partition's padded execution leaves
    plus state input and output, with the FE coexistence reserve.
    """
    bounds = (
        maximum_cells,
        maximum_faces,
        maximum_message_entries,
        maximum_storage_bytes,
    )
    mesh = execution_group.mesh
    if any(type(value) is not int or value < 1 for value in bounds) or tuple(
        mesh.axis_names
    ) != (axis_name,):
        raise ValueError(
            "owner_local.execution must declare positive integer bounds and one explicit partition axis; got invalid bounds or a different mesh axis layout."
        )
    parts = mesh.shape[axis_name]
    local_parts = tuple(
        index
        for index, device in enumerate(execution_group.devices)
        if device.process_index == jax.process_index()
    )
    # A local refusal still joins the collective, so every partition refuses.
    failure: Exception | None = None
    try:
        ordered = _prepare_owner_program_placement(programs, local_parts, parts)
        descriptors = tuple(
            _owner_execution_descriptor(dynamics, halo, (axis_name, *bounds))
            for dynamics, halo in ordered
        )
    except (TypeError, ValueError) as error:
        failure, ordered = error, ()
        refused = jnp.zeros(
            (_OWNER_COUNTS + len(_OWNER_REFUSALS) + _OWNER_DIGEST,), dtype=jnp.int64
        )
        descriptors = tuple(
            _OwnerExecutionDescriptor(refused.at[_OWNER_COUNTS].set(1), "")
            for _ in local_parts
        )
    summary = _agree_owner_execution(descriptors, local_parts, execution_group, axis_name)
    flags = summary[2 * _OWNER_COUNTS : 2 * _OWNER_COUNTS + len(_OWNER_REFUSALS)]
    if np.any(flags):
        reasons = ", ".join(
            reason for reason, flag in zip(_OWNER_REFUSALS, flags, strict=True) if flag
        )
        raise ValueError(
            f"owner_local.programs must be admitted on every partition; got {reasons}."
        ) from failure
    if not summary[-1]:
        raise ValueError(
            "owner_local.programs must share one execution family on every partition; got incompatible metadata, system types, or declared bounds."
        )
    cells, faces, slots, messages = (int(value) for value in summary[:4])
    if (
        cells > maximum_cells
        or max(faces, slots) > maximum_faces
        or messages > maximum_message_entries
    ):
        raise ValueError(
            "owner_local.capacities must fit the declared execution bounds; got overflowing cell, face, or halo buckets."
        )
    padded = tuple(
        _padded_execution_data(dynamics, halo, (cells, faces, slots, messages))
        for dynamics, halo in ordered
    )
    prototype = ordered[0][0]
    precision = prototype.precision
    components = _owner_local_geometry(prototype).component_count
    first, structure = jax.tree_util.tree_flatten(eqx.filter(padded[0], eqx.is_array))
    leaf_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in first)
    state_bytes = cells * components * jnp.dtype(precision.storage_dtype).itemsize
    # The FE execution-bucket host-staging/execution/launch coexistence
    # reserve: an admission bound, not a measured allocation.
    if 3 * parts * (leaf_bytes + 2 * state_bytes) > maximum_storage_bytes:
        raise ValueError(
            "owner_local.storage_bytes must fit the declared coexistence bound; got excess padded storage."
        )
    local_leaves = tuple(
        jax.tree_util.tree_leaves(eqx.filter(data, eqx.is_array)) for data in padded
    )
    numeric = tuple(
        make_owner_local_field_array(
            tuple(leaves[index] for leaves in local_leaves),
            local_parts,
            execution_group.devices,
            axis_name=axis_name,
        )
        for index in range(len(first))
    )
    specifications = tuple(
        PartitionSpec(axis_name, *(None for _ in leaf.shape)) for leaf in first
    )
    return OwnerLocalFiniteVolumeExecution(
        numeric=numeric,
        program=_owner_execution_program(
            eqx.filter(prototype, lambda value: not eqx.is_array(value)),
            eqx.filter(padded[0], lambda value: not eqx.is_array(value)),
            structure,
            specifications,
            mesh,
            axis_name,
        ),
        execution_group=execution_group,
        axis_name=axis_name,
        local_parts=local_parts,
        local_cell_counts=tuple(
            _owner_local_geometry(dynamics).cell_count for dynamics, _ in ordered
        ),
        component_count=components,
        state_dtype=precision.storage_dtype,
        partition_count=parts,
        cell_capacity=cells,
        face_capacity=faces,
        owned_face_capacity=slots,
        message_capacity=messages,
        logical_cell_count=int(summary[_OWNER_COUNTS]),
        logical_owned_face_count=int(summary[_OWNER_COUNTS + 4]),
        process_execution_bytes=len(local_parts) * leaf_bytes,
        execution_id=canonical_fingerprint(
            {
                "kind": "owner-local-finite-volume-execution",
                "family": descriptors[0].family,
                "capacities": (cells, faces, slots, messages),
                "partitions": parts,
            }
        ),
    )


@checked
def execute_owner_local_finite_volume_advance(
    execution: OwnerLocalFiniteVolumeExecution,
    state: Array,
    time: ArrayLike,
    step: ArrayLike,
    /,
    *,
    args: Any = None,
) -> OwnerLocalFiniteVolumeAdvance:
    """Advance prepared owner programs once, without host synchronization.

    ``state`` is a global JAX array in the execution layout ``(partitions,
    cell_capacity, components)`` (see `OwnerLocalFiniteVolumeExecution`)
    whose first axis is the persisted semantic partition index. This boundary
    does not infer stable IDs as global slots or repartition a changed
    partition count. Only padded local execution leaves enter the cached
    program; mesh topology and logical checkpoint arrays never become map
    arguments. ``time``, ``step`` and array leaves of ``args`` are replicated
    controls shared by every partition.
    """
    execution.require_state(state)
    group = execution.execution_group
    placed = jax.device_put(
        state, group.named_sharding(PartitionSpec(execution.axis_name, None, None))
    )
    dynamic, constant = eqx.partition(args, eqx.is_array)
    leaves, tree = jax.tree_util.tree_flatten(dynamic)
    advanced, before, after, finite, admissible = execution.program(
        execution.numeric,
        placed,
        _replicated(time, group),
        _replicated(step, group),
        tuple(_replicated(leaf, group) for leaf in leaves),
        tree,
        constant,
    )
    itemsize = jnp.dtype(execution.state_dtype).itemsize
    return OwnerLocalFiniteVolumeAdvance(
        state=advanced,
        inventory_before=before,
        inventory_after=after,
        inventory_change=after - before,
        finite=finite,
        admissible=admissible,
        admissibility_checked=True,
        process_execution_bytes=execution.process_execution_bytes,
        logical_state_bytes=execution.logical_cell_count
        * execution.component_count
        * itemsize,
        execution_state_bytes=state.size * itemsize,
    )


__all__ = [
    "PreparedUnstructuredFiniteVolumeDynamics",
    "OwnerLocalFiniteVolumeAdvance",
    "OwnerLocalFiniteVolumeExecution",
    "execute_owner_local_finite_volume_advance",
    "prepare_owner_local_finite_volume_execution",
    "UnstructuredFiniteVolumeBoundarySet",
    "UnstructuredFiniteVolumeDiagnostics",
    "UnstructuredFiniteVolumeMethodPlan",
    "UnstructuredFiniteVolumeStageEvaluation",
]

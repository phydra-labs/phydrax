#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum, compensated_sum_chunks
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._conservation_boundary import (
    AbstractConservationBoundary,
    ALEBoundaryContext,
    ConstantStateBoundary,
    ExtrapolationBoundary,
    PrescribedStateBoundary,
)
from .._conservation_ledger import (
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from ._cell_polynomial import PreparedCellPolynomialReconstruction
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
    FiniteVolumeStageMetrics,
    lower_static_unstructured_stage_metrics,
)
from ._physical_boundaries import (
    MovingSlipWallBoundary,
    SlipWallBoundary,
    SupersonicInflowBoundary,
    SupersonicOutflowBoundary,
)
from ._positivity import EinfeldtHLLFluxPlan
from ._precision import FiniteVolumePrecisionPolicy
from ._reconstruction import PiecewiseConstantReconstruction
from ._riemann import (
    AbstractNumericalFluxPlan,
    HLLCFluxPlan,
    HLLFluxPlan,
    RusanovFluxPlan,
)
from ._small_cell import ConservativeSmallCellRedistributionPlan
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_overset import PeriodicSlidingCoupling
from ._unstructured_weno import PreparedUnstructuredWENOZReconstruction


SourceFunction = Callable[[Array, Array, Array, Any], ArrayLike]
PreparedUnstructuredReconstruction = (
    PreparedCellPolynomialReconstruction | PreparedUnstructuredWENOZReconstruction
)


class UnstructuredFiniteVolumeBoundarySet(StrictModule, NonTrainableState):
    """Complete named physical-boundary ownership for an unstructured mesh."""

    patch_names: tuple[str, ...] = eqx.field(static=True)
    boundaries: tuple[AbstractConservationBoundary, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_names: tuple[str, ...],
        boundaries: Mapping[str, AbstractConservationBoundary],
        /,
    ):
        names = tuple(patch_names)
        if set(boundaries) != set(names):
            raise ValueError(
                "Unstructured FV boundaries must cover every mesh patch exactly."
            )
        values = tuple(boundaries[name] for name in names)
        allowed = (
            ConstantStateBoundary,
            ExtrapolationBoundary,
            PrescribedStateBoundary,
            MovingSlipWallBoundary,
            SlipWallBoundary,
            SupersonicInflowBoundary,
            SupersonicOutflowBoundary,
        )
        if any(not isinstance(value, allowed) for value in values):
            raise TypeError(
                "Unstructured FV boundaries require normal-oriented exterior-state policies."
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


class UnstructuredFiniteVolumeMethodPlan(StrictModule, NonTrainableState):
    """Reconstruction and normal numerical flux for unstructured cells."""

    reconstruction: PiecewiseConstantReconstruction | PreparedUnstructuredReconstruction
    interface_solver: RusanovFluxPlan | HLLFluxPlan | HLLCFluxPlan | EinfeldtHLLFluxPlan
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PiecewiseConstantReconstruction
        | PreparedUnstructuredReconstruction,
        interface_solver: RusanovFluxPlan
        | HLLFluxPlan
        | HLLCFluxPlan
        | EinfeldtHLLFluxPlan,
        /,
    ):
        if not isinstance(
            reconstruction,
            (
                PiecewiseConstantReconstruction,
                PreparedCellPolynomialReconstruction,
                PreparedUnstructuredWENOZReconstruction,
            ),
        ):
            raise TypeError(
                "Unstructured FV reconstruction must be piecewise constant or prepared cell-polynomial."
            )
        if not isinstance(
            interface_solver,
            (RusanovFluxPlan, HLLFluxPlan, HLLCFluxPlan, EinfeldtHLLFluxPlan),
        ):
            raise TypeError("Unstructured FV supports Rusanov, HLL, or HLLC flux.")
        reconstruction_id = (
            reconstruction.plan_id
            if isinstance(reconstruction, PiecewiseConstantReconstruction)
            else reconstruction.prepared_id
        )
        self.reconstruction = reconstruction
        self.interface_solver = interface_solver
        self.method_id = canonical_fingerprint(
            {
                "kind": "unstructured-fv-method",
                "reconstruction": reconstruction_id,
                "flux": interface_solver.flux_id,
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


class PreparedUnstructuredFiniteVolumeDynamics(StrictModule):
    """Single-device conservative dynamics over explicit unstructured faces."""

    system: Any
    discretization: (
        UnstructuredFiniteVolumeDiscretization | DyadicFiniteVolumeDiscretization
    )
    method: UnstructuredFiniteVolumeMethodPlan
    boundaries: UnstructuredFiniteVolumeBoundarySet
    coupling: PreparedUnstructuredFiniteVolumeCoupling
    precision: FiniteVolumePrecisionPolicy
    boundary_face_indices: tuple[Array, ...]
    stage_rate_block_templates: tuple[ConservationStageFluxRateBlock, ...]
    stage_boundary_face_indices: tuple[tuple[Array, ...], ...]
    source_cell_indices: Array
    overset_rate_block_template: ConservationStageFluxRateBlock | None
    overset_active_cell_mask: Array
    overset_effective_cell_volumes: Array
    overset_policy_id: str | None = eqx.field(static=True)
    overset_mapping_id: str | None = eqx.field(static=True)
    overset_epoch_id: str | None = eqx.field(static=True)
    source: SourceFunction | None = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
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
    ):
        if not isinstance(
            discretization,
            (UnstructuredFiniteVolumeDiscretization, DyadicFiniteVolumeDiscretization),
        ):
            raise TypeError(
                "discretization must be explicit-face finite-volume geometry."
            )
        if not isinstance(method, UnstructuredFiniteVolumeMethodPlan):
            raise TypeError("method must be UnstructuredFiniteVolumeMethodPlan.")
        if not isinstance(boundaries, UnstructuredFiniteVolumeBoundarySet):
            raise TypeError("boundaries must be UnstructuredFiniteVolumeBoundarySet.")
        if boundaries.patch_names != discretization.boundary_patch_names:
            raise ValueError("Boundary patch names must match prepared mesh patches.")
        if (
            system.dimension != discretization.cell_dimension
            or system.component_count != discretization.component_count
        ):
            raise ValueError(
                "Unstructured FV system dimension/components do not match geometry."
            )
        reconstruction = method.reconstruction
        if (
            isinstance(
                reconstruction,
                (
                    PreparedCellPolynomialReconstruction,
                    PreparedUnstructuredWENOZReconstruction,
                ),
            )
            and reconstruction.discretization.prepared_id != discretization.prepared_id
        ):
            raise ValueError("Reconstruction belongs to a different geometry.")
        coupling_ = (
            UnstructuredFiniteVolumeCouplingPlan().prepare(discretization)
            if coupling is None
            else coupling
        )
        if not isinstance(coupling_, PreparedUnstructuredFiniteVolumeCoupling):
            raise TypeError(
                "coupling must be PreparedUnstructuredFiniteVolumeCoupling or None."
            )
        if (
            coupling_.topology_id != discretization.topology_id
            or coupling_.geometry_id != discretization.geometry_id
            or coupling_.discretization_id != discretization.prepared_id
        ):
            raise ValueError("Coupling belongs to a different prepared geometry.")
        if coupling_.motion is not None:
            moving_polynomial = isinstance(
                reconstruction, PreparedCellPolynomialReconstruction
            ) and reconstruction.basis.degree in (1, 2)
            moving_weno = isinstance(
                reconstruction, PreparedUnstructuredWENOZReconstruction
            ) and reconstruction.optimal.basis.degree in (1, 2)
            if (
                type(reconstruction) is not PiecewiseConstantReconstruction
                and not moving_polynomial
                and not moving_weno
            ):
                raise ValueError(
                    "Moving unstructured finite-volume reconstruction requires "
                    "piecewise constant or stage-refreshable degree-one WLSQ "
                    f"(coupling={coupling_.prepared_id}, method={method.method_id})."
                )
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        source_identifier = None if source_id is None else str(source_id)
        if (source is None) != (source_identifier is None) or source_identifier == "":
            raise ValueError(
                "A source callable requires exactly one non-empty source_id."
            )
        precision_ = (
            FiniteVolumePrecisionPolicy(jnp.dtype(discretization.cell_volumes.dtype).name)
            if precision is None
            else precision
        )
        overset = coupling_.overset
        overset_rate_block_template = None
        overset_policy_id = coupling_.overset_policy_id
        overset_mapping_id = coupling_.overset_mapping_id
        overset_epoch_id = coupling_.overset_epoch_id
        if overset is None:
            overset_active_cell_mask = np.ones((discretization.cell_count,), dtype=bool)
            overset_effective_cell_volumes = jnp.asarray(discretization.cell_volumes)
        else:
            donor_count = int(np.asarray(overset.donor_covered_measures).size)
            if donor_count != discretization.cell_count:
                raise ValueError(
                    "Overset dynamics require donor and receptor cell layouts "
                    "to share one single-device content axis."
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
                raise TypeError("Overset activity and hole masks must be Boolean.")
            donor_active = donor_active & ~donor_hole
            donor_eligible = donor_eligible & ~donor_hole
            receptor_active = receptor_active & ~receptor_hole
            receptor_fringe = np.asarray(overset.receptor_fringe_mask)
            if receptor_fringe.dtype.kind != "b":
                raise TypeError("Overset receptor fringe mask must be Boolean.")
            if (
                donor_active.shape != (donor_count,)
                or donor_hole.shape != (donor_count,)
                or donor_eligible.shape != (donor_count,)
                or receptor_active.shape != (discretization.cell_count,)
                or receptor_hole.shape != (discretization.cell_count,)
                or receptor_fringe.shape != (discretization.cell_count,)
                or np.any(donor_active & donor_hole)
                or np.any(donor_eligible & donor_hole)
                or np.any(receptor_active & receptor_hole)
                or np.any(receptor_fringe & receptor_hole)
                or np.any(receptor_fringe & ~receptor_active)
            ):
                raise ValueError("Overset active/hole/fringe ownership is invalid.")
            overset_active_cell_mask = receptor_active & ~receptor_hole
            donor_indices = np.asarray(overset.donor_indices, dtype=np.int32)
            receptor_cells = np.asarray(overset.receptor_cells, dtype=np.int32)
            receptor_routes = np.asarray(overset.receptor_routes, dtype=np.int32)
            sliding_coupling = coupling_.sliding_coupling
            if sliding_coupling is None:
                template_owner_cells = donor_indices
                template_neighbour_cells = receptor_cells[receptor_routes]
                if overset.receptor_face_cells is not None:
                    certified_face_cells = np.asarray(
                        overset.receptor_face_cells,
                        dtype=np.int32,
                    )
                    if (
                        certified_face_cells.shape != template_neighbour_cells.shape
                        or not np.array_equal(
                            certified_face_cells,
                            template_neighbour_cells,
                        )
                    ):
                        raise ValueError(
                            "Non-sliding overset receptor faces must match mapped "
                            "receptor routes in order."
                        )
                sliding_coupling_id = None
            else:
                if overset.receptor_face_cells is None:
                    raise ValueError(
                        "Sliding overset correction requires certified receptor faces."
                    )
                left_routes = np.asarray(sliding_coupling.left_routes, dtype=np.int32)
                right_routes = np.asarray(sliding_coupling.right_routes, dtype=np.int32)
                face_cells = np.asarray(overset.receptor_face_cells, dtype=np.int32)
                template_owner_cells = donor_indices[left_routes]
                template_neighbour_cells = face_cells[right_routes]
                sliding_coupling_id = sliding_coupling.coupling_id
            route_active = (
                donor_active[template_owner_cells]
                & donor_eligible[template_owner_cells]
                & ~donor_hole[template_owner_cells]
                & overset_active_cell_mask[template_neighbour_cells]
                & receptor_fringe[template_neighbour_cells]
                & (template_owner_cells != template_neighbour_cells)
            )
            overset_rate_block_template = ConservationStageFluxRateBlock(
                jnp.zeros(
                    (template_owner_cells.size, discretization.component_count),
                    dtype=jnp.dtype(precision_.reduction_dtype),
                ),
                template_owner_cells,
                template_neighbour_cells,
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
        if overset is not None and not isinstance(
            reconstruction, PiecewiseConstantReconstruction
        ):
            for name in (
                "operator_geometry_id",
                "trace_operator_geometry_id",
                "geometry_bound_operator_id",
            ):
                operator_geometry_id = getattr(overset, name, None)
                if operator_geometry_id is not None and operator_geometry_id not in (
                    discretization.prepared_id,
                    discretization.geometry_id,
                ):
                    raise ValueError(
                        "High-order mapped overset traces require current "
                        "geometry-bound operators."
                    )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy.")
        if coupling_.motion is not None:
            stage_layouts = (coupling_.motion.face_layout,)
        elif coupling_.embedded_metrics is not None:
            embedded_boundaries = coupling_.embedded_boundaries
            if not isinstance(embedded_boundaries, UnstructuredEmbeddedBoundarySet):
                raise TypeError(
                    "Prepared embedded coupling requires embedded boundary ownership."
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
            static_stage = lower_static_unstructured_stage_metrics(discretization)
            stage_layouts = tuple(block.layout for block in static_stage.face_blocks)
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
                layout.neighbour_cells,
                np.asarray(layout.active_mask)
                & np.asarray(overset_active_cell_mask)[np.asarray(layout.owner_cells)]
                & (
                    (np.asarray(layout.neighbour_cells) < 0)
                    | np.asarray(overset_active_cell_mask)[
                        np.maximum(np.asarray(layout.neighbour_cells), 0)
                    ]
                ),
                layout.block_id,
                layout.block_kind,
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
        source_active_cell_mask = np.asarray(overset_active_cell_mask, dtype=bool)
        if coupling_.embedded_metrics is not None:
            source_active_cell_mask = source_active_cell_mask & np.asarray(
                coupling_.embedded_metrics.active_fluid_cells,
                dtype=bool,
            )
        source_cell_indices = jnp.asarray(
            np.flatnonzero(source_active_cell_mask),
            dtype=jnp.int32,
        )
        self.system = system
        self.discretization = discretization
        self.method = method
        self.boundaries = boundaries
        self.coupling = coupling_
        self.precision = precision_
        self.source = source
        self.source_id = source_identifier
        self.boundary_face_indices = boundary_face_indices
        self.stage_rate_block_templates = stage_rate_block_templates
        self.stage_boundary_face_indices = stage_boundary_face_indices
        self.source_cell_indices = source_cell_indices
        self.overset_rate_block_template = overset_rate_block_template
        self.overset_active_cell_mask = jnp.asarray(overset_active_cell_mask)

        self.overset_effective_cell_volumes = jnp.asarray(overset_effective_cell_volumes)
        self.overset_policy_id = overset_policy_id
        self.overset_mapping_id = overset_mapping_id
        self.overset_epoch_id = overset_epoch_id
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-fv-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "coupling": coupling_.prepared_id,
                "boundaries": boundaries.boundary_set_id,
                "precision": precision_.policy_id,
                "overset_policy": overset_policy_id,
                "overset_mapping": overset_mapping_id,
                "overset_epoch": overset_epoch_id,
                "source": source_identifier,
            }
        )

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

    def make_fallback_dynamics(
        self, fallback_flux: AbstractNumericalFluxPlan, /
    ) -> "PreparedUnstructuredFiniteVolumeDynamics":
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

    def _stage_route_active(
        self,
        layout: Any,
        active_cell_mask: Array,
        /,
    ) -> Array:
        safe_neighbour = jnp.maximum(layout.neighbour_cells, 0)
        return (
            layout.active_mask
            & self.overset_active_cell_mask[layout.owner_cells]
            & (
                (layout.neighbour_cells < 0)
                | self.overset_active_cell_mask[safe_neighbour]
            )
            & active_cell_mask[layout.owner_cells]
            & ((layout.neighbour_cells < 0) | active_cell_mask[safe_neighbour])
        )

    def _stage_rate_templates(
        self,
        metrics: FiniteVolumeStageMetrics,
        /,
    ) -> tuple[ConservationStageFluxRateBlock, ...]:
        if len(metrics.face_blocks) != len(self.stage_rate_block_templates):
            raise ValueError(
                "Stage geometry does not match the prepared face-block layout."
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
                    "Stage geometry face-block kind/shape does not match the dynamics."
                )
            active = self._stage_route_active(layout, metrics.active_cell_mask)
            zero_rate = eqx.error_if(
                jnp.zeros_like(prepared.flux_rate),
                jnp.any(layout.owner_cells != prepared.owner_cells)
                | jnp.any(layout.neighbour_cells != prepared.neighbour_cells)
                | jnp.any(active & ~prepared.active_mask),
                "Stage geometry face routes do not match the dynamics.",
            )
            template = prepared.with_flux_rate(zero_rate)
            templates.append(template)
        return tuple(templates)

    def _validated_stage_average(
        self,
        content_state: Any,
        metrics: FiniteVolumeStageMetrics,
        /,
    ) -> Array:
        from ...solver._finite_volume_content import (
            FiniteVolumeConservativeContentState,
        )

        if not isinstance(content_state, FiniteVolumeConservativeContentState):
            raise TypeError("content_state must be FiniteVolumeConservativeContentState.")
        if not isinstance(metrics, FiniteVolumeStageMetrics):
            raise TypeError("metrics must be FiniteVolumeStageMetrics.")
        if content_state.precision.policy_id != self.precision.policy_id:
            raise ValueError("Stage content precision does not match the dynamics.")
        if content_state.topology_epoch_id != metrics.topology_epoch_id:
            raise ValueError("Stage content and geometry must share one topology epoch.")
        if content_state.geometry_family_id != metrics.geometry_family_id:
            raise ValueError("Stage content and geometry must share one geometry family.")
        if content_state.geometry_layout_id != metrics.geometry_layout_id:
            raise ValueError("Stage content and geometry must share one geometry layout.")
        if content_state.evidence_policy_id != metrics.evidence.policy_id:
            raise ValueError("Stage content and geometry must share one evidence policy.")
        if (
            content_state.cell_count != metrics.cell_count
            or content_state.component_shape != (self.discretization.component_count,)
        ):
            raise ValueError("Stage content has incompatible cell/component shape.")
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

    def _stage_face_states(
        self,
        state: Array,
        metrics: FiniteVolumeStageMetrics,
        args: Any,
        /,
    ) -> tuple[tuple[Array, Array, Array], ...]:
        reconstruction = self.method.reconstruction
        active_mask = metrics.active_cell_mask
        safe_seed = state[jnp.argmax(active_mask)]
        safe_state = jnp.where(active_mask[:, None], state, safe_seed[None, :])
        reconstruction_state = self.precision.reconstruction(safe_state)
        coefficients: Array | None = None
        stage_lengths: Array | None = None
        outputs: list[tuple[Array, Array, Array]] = []
        for block_index, geometry_block in enumerate(metrics.face_blocks):
            layout = geometry_block.layout
            policy_indices = self.stage_boundary_face_indices[block_index]
            owner = layout.owner_cells
            neighbour = layout.neighbour_cells
            safe_neighbour = jnp.maximum(neighbour, 0)
            route_active = self._stage_route_active(layout, metrics.active_cell_mask)
            points = self.precision.reconstruction(geometry_block.quadrature_points)
            if isinstance(reconstruction, PiecewiseConstantReconstruction):
                quadrature_count = layout.quadrature_count
                left = jnp.broadcast_to(
                    reconstruction_state[owner, None, :],
                    (layout.face_count, quadrature_count, state.shape[-1]),
                )
                right = jnp.broadcast_to(
                    reconstruction_state[safe_neighbour, None, :],
                    left.shape,
                )
            elif isinstance(reconstruction, PreparedCellPolynomialReconstruction):
                if coefficients is None:
                    if self.coupling.motion is None:
                        coefficients = reconstruction.coefficients(reconstruction_state)
                    else:
                        coefficients, stage_lengths = reconstruction.stage_coefficients(
                            reconstruction_state, metrics
                        )
                if stage_lengths is None:
                    left = reconstruction.evaluate_coefficients(
                        reconstruction_state,
                        coefficients,
                        owner,
                        points,
                    )
                    right = reconstruction.evaluate_coefficients(
                        reconstruction_state,
                        coefficients,
                        safe_neighbour,
                        points,
                    )
                else:
                    left = reconstruction.evaluate_stage_coefficients(
                        reconstruction_state,
                        coefficients,
                        stage_lengths,
                        metrics,
                        owner,
                        points,
                    )
                    right = reconstruction.evaluate_stage_coefficients(
                        reconstruction_state,
                        coefficients,
                        stage_lengths,
                        metrics,
                        safe_neighbour,
                        points,
                    )
            elif isinstance(reconstruction, PreparedUnstructuredWENOZReconstruction):
                if coefficients is None:
                    coefficients = reconstruction.coefficients(reconstruction_state)
                    if self.coupling.motion is not None:
                        _, stage_lengths = reconstruction.optimal.stage_coefficients(
                            reconstruction_state, metrics
                        )
                if stage_lengths is None:
                    left = reconstruction.optimal.evaluate_coefficients(
                        reconstruction_state,
                        coefficients,
                        owner,
                        points,
                    )
                    right = reconstruction.optimal.evaluate_coefficients(
                        reconstruction_state,
                        coefficients,
                        safe_neighbour,
                        points,
                    )
                else:
                    left = reconstruction.optimal.evaluate_stage_coefficients(
                        reconstruction_state,
                        coefficients,
                        stage_lengths,
                        metrics,
                        owner,
                        points,
                    )
                    right = reconstruction.optimal.evaluate_stage_coefficients(
                        reconstruction_state,
                        coefficients,
                        stage_lengths,
                        metrics,
                        safe_neighbour,
                        points,
                    )
                left = reconstruction._limit(reconstruction_state, left, owner)
                right = reconstruction._limit(
                    reconstruction_state,
                    right,
                    safe_neighbour,
                )
            else:
                raise TypeError("Unsupported prepared unstructured reconstruction.")

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
                        "Moving stage geometry must provide grid-normal velocity."
                    )
                grid_velocity = jnp.zeros_like(geometry_block.quadrature_weights)
            grid_normal = self.precision.reconstruction(grid_velocity)
            if layout.block_kind == "physical":
                policies = self.boundaries.boundaries
            elif layout.block_kind == "cut":
                embedded_boundaries = self.coupling.embedded_boundaries
                if not isinstance(
                    embedded_boundaries,
                    UnstructuredEmbeddedBoundarySet,
                ):
                    raise TypeError(
                        "Cut stage blocks require embedded boundary ownership."
                    )
                policies = embedded_boundaries.boundaries
            else:
                raise ValueError(f"Unsupported stage block kind {layout.block_kind!r}.")
            if len(policies) != len(policy_indices):
                raise ValueError(
                    "Stage boundary policy routes do not match prepared ownership."
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
                if isinstance(policy, MovingSlipWallBoundary):
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
                else:
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
            outputs.append((left, right, normal))
        return tuple(outputs)

    def _overset_correction(
        self,
        state: Array,
        metrics: FiniteVolumeStageMetrics,
        args: Any,
        /,
    ) -> tuple[ConservationStageFluxRateBlock | None, Array, Array]:
        """Evaluate one moved, stage-bound conservative sliding correction."""

        template = self.overset_rate_block_template
        overset = self.coupling.overset
        reduction_dtype = jnp.dtype(self.precision.reduction_dtype)
        if template is None or overset is None:
            return (
                None,
                jnp.zeros((), dtype=reduction_dtype),
                jnp.zeros((0,), dtype=reduction_dtype),
            )
        face_ids = overset.receptor_face_ids
        face_cells = overset.receptor_face_cells
        if face_ids is None or face_cells is None or overset.face_artifact_id is None:
            raise ValueError(
                "Overset correction requires a certified receptor-face artifact "
                "with explicit physical face IDs."
            )
        physical_blocks = tuple(
            block
            for block in metrics.face_blocks
            if block.layout.block_kind == "physical"
        )
        if len(physical_blocks) != 1:
            raise ValueError(
                "Overset correction requires exactly one physical stage face block."
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
        physical_neighbours = layout.neighbour_cells[positions]
        incident = (physical_owners == cells) | (physical_neighbours == cells)
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

        reconstruction = self.method.reconstruction
        reconstructed = self.precision.reconstruction(state)
        donor_indices = jnp.asarray(overset.donor_indices, dtype=jnp.int32)
        sliding = self.coupling.sliding_coupling
        if sliding is None:
            route_left = jnp.arange(donor_indices.size, dtype=jnp.int32)
            route_right = jnp.arange(cells.size, dtype=jnp.int32)
            route_fraction = jnp.ones((cells.size,), dtype=reduction_dtype)
            if donor_indices.shape[0] != cells.shape[0]:
                raise ValueError(
                    "Non-sliding overset face routes must pair donors and receptors."
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
        if isinstance(reconstruction, PiecewiseConstantReconstruction):
            route_donor_trace = jnp.broadcast_to(
                reconstructed[route_donor_cells, None, :],
                (
                    route_donor_cells.shape[0],
                    points_array.shape[1],
                    reconstructed.shape[-1],
                ),
            )
            receptor_interior = jnp.broadcast_to(
                reconstructed[cells, None, :],
                (
                    cells.shape[0],
                    points_array.shape[1],
                    reconstructed.shape[-1],
                ),
            )
        elif isinstance(reconstruction, PreparedCellPolynomialReconstruction):
            if self.coupling.motion is None:
                coefficients = reconstruction.coefficients(reconstructed)
                route_donor_trace = reconstruction.evaluate_coefficients(
                    reconstructed,
                    coefficients,
                    route_donor_cells,
                    route_points,
                )
                receptor_interior = reconstruction.evaluate_coefficients(
                    reconstructed,
                    coefficients,
                    cells,
                    points_array,
                )
            else:
                coefficients, stage_lengths = reconstruction.stage_coefficients(
                    reconstructed, metrics
                )
                route_donor_trace = reconstruction.evaluate_stage_coefficients(
                    reconstructed,
                    coefficients,
                    stage_lengths,
                    metrics,
                    route_donor_cells,
                    route_points,
                )
                receptor_interior = reconstruction.evaluate_stage_coefficients(
                    reconstructed,
                    coefficients,
                    stage_lengths,
                    metrics,
                    cells,
                    points_array,
                )
        elif isinstance(reconstruction, PreparedUnstructuredWENOZReconstruction):
            coefficients = reconstruction.coefficients(reconstructed)
            if self.coupling.motion is None:
                route_donor_trace = reconstruction.optimal.evaluate_coefficients(
                    reconstructed,
                    coefficients,
                    route_donor_cells,
                    route_points,
                )
                receptor_interior = reconstruction.optimal.evaluate_coefficients(
                    reconstructed,
                    coefficients,
                    cells,
                    points_array,
                )
            else:
                _, stage_lengths = reconstruction.optimal.stage_coefficients(
                    reconstructed, metrics
                )
                route_donor_trace = reconstruction.optimal.evaluate_stage_coefficients(
                    reconstructed,
                    coefficients,
                    stage_lengths,
                    metrics,
                    route_donor_cells,
                    route_points,
                )
                receptor_interior = reconstruction.optimal.evaluate_stage_coefficients(
                    reconstructed,
                    coefficients,
                    stage_lengths,
                    metrics,
                    cells,
                    points_array,
                )
            route_donor_trace = reconstruction._limit(
                reconstructed,
                route_donor_trace,
                route_donor_cells,
            )
            receptor_interior = reconstruction._limit(
                reconstructed,
                receptor_interior,
                cells,
            )
        else:
            raise TypeError("Unsupported prepared unstructured reconstruction.")
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

        normal_flux = self.method.interface_solver.normal_ale_face_flux(
            self.system,
            self.precision.flux(receptor_interior),
            self.precision.flux(donor_trace),
            self.precision.flux(normals_array),
            grid_velocity,
            args,
        )
        integrated_face_flux = oe.contract(
            "fq,fqc->fc",
            measures_array,
            self.precision.reduction(normal_flux.normal_flux),
        )
        # The Riemann solve is receptor-left with a receptor-outward normal,
        # whereas the conservative ledger route is donor-owner to
        # receptor-neighbour. Canonicalize to that opposite positive direction.
        route_flux = -integrated_face_flux[route_right] * route_fraction[:, None]
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
        return template.with_flux_rate(route_flux), route_speed, route_measures

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
                "Embedded stages require exactly one prepared small-cell "
                "redistribution plan; non-embedded stages require none."
            )
        if redistribution is None:
            return ledger
        if redistribution.evidence.prepared_geometry_id != (
            self.discretization.prepared_id
        ):
            raise ValueError("Small-cell redistribution belongs to another geometry.")
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

    def zero_stage_ledger(
        self,
        metrics: FiniteVolumeStageMetrics,
        /,
        *,
        redistribution: ConservativeSmallCellRedistributionPlan | None = None,
    ) -> ConservationStageLedger:
        """Create a routed zero rate without invoking reconstruction or physics."""

        if not isinstance(metrics, FiniteVolumeStageMetrics):
            raise TypeError("metrics must be FiniteVolumeStageMetrics.")
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

    def evaluate_stage(
        self,
        content_state: Any,
        metrics: FiniteVolumeStageMetrics,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
        redistribution: ConservativeSmallCellRedistributionPlan | None = None,
    ) -> UnstructuredFiniteVolumeStageEvaluation:
        """Evaluate one certified quadrature-integrated ALE content rate."""

        cfl_ = float(cfl)
        if not np.isfinite(cfl_) or cfl_ <= 0.0:
            raise ValueError("cfl must be positive and finite.")
        average = self._validated_stage_average(content_state, metrics)
        vof_plan = self.coupling.vof
        stage_plic = None
        primitive = None
        cell_volume_divergence = jnp.zeros(
            (metrics.cell_count,),
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        if vof_plan is not None:
            from ...equations._multiphase import TwoMaterialVOFSystem

            if not isinstance(self.system, TwoMaterialVOFSystem):
                raise TypeError("VOF stage coupling requires TwoMaterialVOFSystem.")
            alpha = average[:, self.system.layout.alpha_index]
            normal_override = None
            override_mask = None
            contact_angles = self.coupling.contact_angles
            embedded_metrics = self.coupling.embedded_metrics
            if contact_angles is not None:
                if embedded_metrics is None:
                    raise ValueError("Contact-angle coupling requires embedded metrics.")
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
                geometry_layout_id=metrics.geometry_layout_id,
                geometry_version=metrics.geometry_version,
                normal_override=normal_override,
                override_mask=override_mask,
            )
            primitive = self.system.conserved_to_primitive(average)
        traces = self._stage_face_states(average, metrics, args)
        templates = self._stage_rate_templates(metrics)
        blocks: list[ConservationStageFluxRateBlock] = []
        speeds: list[Array] = []
        cell_rate = jnp.zeros(
            (metrics.cell_count,),
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        for geometry_block, template, (left, right, normal) in zip(
            metrics.face_blocks,
            templates,
            traces,
            strict=True,
        ):
            grid_velocity = geometry_block.quadrature_grid_normal_velocity
            if grid_velocity is None:
                if self.coupling.motion is not None:
                    raise ValueError(
                        "Moving stage geometry must provide grid-normal velocity."
                    )
                grid_velocity = jnp.zeros_like(geometry_block.quadrature_weights)
            result = self.method.interface_solver.normal_ale_face_flux(
                self.system,
                self.precision.flux(left),
                self.precision.flux(right),
                self.precision.flux(normal),
                self.precision.flux(grid_velocity),
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
            flux_rate = oe.contract(
                "fq,fqc->fc",
                weights,
                normal_flux,
            )
            if vof_plan is not None and layout.block_kind == "physical":
                if stage_plic is None or primitive is None:
                    raise RuntimeError("VOF stage reconstruction was not prepared.")
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
                donor_apertures = vof_plan.donor_phase_apertures(
                    full_total_mass, stage_plic
                )[face_ids]
                owner = layout.owner_cells
                neighbour = layout.neighbour_cells
                safe_neighbour = jnp.maximum(neighbour, 0)
                donor_left = total_mass_flux >= 0.0
                boundary_inflow = (neighbour < 0) & ~donor_left
                rho0 = jnp.where(
                    donor_left,
                    primitive[owner, 0],
                    primitive[safe_neighbour, 0],
                )
                rho1 = jnp.where(
                    donor_left,
                    primitive[owner, 1],
                    primitive[safe_neighbour, 1],
                )
                mixture_density = (
                    donor_apertures[:, 0] * rho0 + donor_apertures[:, 1] * rho1
                )
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
                            | (checked_mixture_density <= self.system.eos.density_floor)
                        )
                    ),
                    "PLIC donor face mixture density is invalid.",
                )
                volume_flux = total_mass_flux / checked_mixture_density
                mass0_flux = donor_apertures[:, 0] * rho0 * volume_flux
                alpha_flux = donor_apertures[:, 0] * volume_flux

                right_primitive = self.system.conserved_to_primitive(right)
                exterior_alpha = right_primitive[..., self.system.layout.alpha_index]
                exterior_rho0 = right_primitive[..., 0]
                exterior_rho1 = right_primitive[..., 1]
                exterior_mixture_density = (
                    exterior_alpha * exterior_rho0
                    + (1.0 - exterior_alpha) * exterior_rho1
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
                            | (exterior_mixture_density <= self.system.eos.density_floor)
                        )
                    ),
                    "Exterior VOF inflow mixture density is invalid.",
                )
                exterior_total_mass_flux = normal_flux[..., 0] + normal_flux[..., 1]
                exterior_volume_flux_density = (
                    exterior_total_mass_flux / exterior_mixture_density
                )
                exterior_volume_flux = oe.contract(
                    "fq,fq->f",
                    weights,
                    exterior_volume_flux_density,
                )
                exterior_mass0_flux = oe.contract(
                    "fq,fq->f",
                    weights,
                    exterior_alpha * exterior_rho0 * exterior_volume_flux_density,
                )
                exterior_alpha_flux = oe.contract(
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
                flux_rate = flux_rate.at[:, self.system.layout.alpha_index].set(
                    alpha_flux
                )
                cell_volume_divergence = cell_volume_divergence.at[owner].add(volume_flux)
                cell_volume_divergence = cell_volume_divergence.at[safe_neighbour].add(
                    jnp.where(neighbour >= 0, -volume_flux, 0.0)
                )
            face_rate = oe.contract(
                "fq,fq->f",
                weights,
                signal_speed,
            )
            blocks.append(template.with_flux_rate(flux_rate))
            speeds.append(self.precision.decision(signal_speed))
            owner = layout.owner_cells
            neighbour = layout.neighbour_cells
            cell_rate = cell_rate.at[owner].add(face_rate)
            cell_rate = cell_rate.at[jnp.maximum(neighbour, 0)].add(
                jnp.where(neighbour >= 0, face_rate, 0.0)
            )
        (
            overset_block,
            overset_speed,
            overset_route_measures,
        ) = self._overset_correction(average, metrics, args)
        if overset_block is not None:
            speeds.append(overset_speed)
            overset_face_rate = overset_speed * overset_route_measures
            cell_rate = cell_rate.at[overset_block.owner_cells].add(overset_face_rate)
            cell_rate = cell_rate.at[jnp.maximum(overset_block.neighbour_cells, 0)].add(
                jnp.where(
                    overset_block.neighbour_cells >= 0,
                    overset_face_rate,
                    0.0,
                )
            )

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
                    "Unstructured FV source must match the active state shape."
                )
            active_source_rate = self.precision.reduction(
                source_value
            ) * self.precision.reduction(
                metrics.effective_cell_volumes[self.source_cell_indices, None]
            )
            source_rate = source_rate.at[self.source_cell_indices].set(active_source_rate)
        if vof_plan is not None:
            from ...equations._multiphase import TwoMaterialVOFSystem

            if not isinstance(self.system, TwoMaterialVOFSystem):
                raise TypeError("VOF source requires TwoMaterialVOFSystem.")
            alpha = average[:, self.system.layout.alpha_index]
            volumes = self.precision.reduction(metrics.effective_cell_volumes)
            divergence = cell_volume_divergence / volumes
            alpha_source = self.system.volume_fraction_source(
                alpha,
                divergence,
                average,
            )
            source_rate = source_rate.at[:, self.system.layout.alpha_index].add(
                self.precision.reduction(alpha_source) * volumes
            )
        capillary_step = jnp.asarray(jnp.inf, dtype=average.dtype)
        capillarity = self.coupling.capillarity
        if capillarity is not None:
            if stage_plic is None or primitive is None:
                raise ValueError("Capillarity requires a stage PLIC reconstruction.")
            if not isinstance(self.system, TwoMaterialVOFSystem):
                raise TypeError("Capillarity requires TwoMaterialVOFSystem.")
            density = average[:, 0] + average[:, 1]
            velocity = primitive[:, 2 : 2 + self.system.dimension]
            alpha = average[:, self.system.layout.alpha_index]
            capillary_block = capillarity.face_rate_block(
                stage_plic,
                density,
                alpha,
                velocity,
            )
            momentum_rate = capillary_block.cell_momentum_rate(metrics.cell_count)
            energy_rate = capillary_block.cell_energy_rate(metrics.cell_count)
            source_rate = source_rate.at[
                :, self.system.layout.momentum_start : self.system.layout.momentum_stop
            ].add(self.precision.reduction(momentum_rate))
            source_rate = source_rate.at[:, self.system.layout.energy_index].add(
                self.precision.reduction(energy_rate)
            )
            cell_size = jnp.sqrt(metrics.effective_cell_volumes)
            capillary_step = capillarity.capillary_step(
                cell_size,
                density,
                interface_active=stage_plic.interface_active,
            )
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
                    "Embedded dynamics require a stabilization policy for CFL."
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
            jnp.minimum(hyperbolic_step, capillary_step)
        )
        return UnstructuredFiniteVolumeStageEvaluation(
            ledger=ledger,
            relative_signal_speeds=tuple(speeds),
            cell_relative_rate=cell_relative_rate,
            maximum_relative_rate=maximum_relative_rate,
            relative_cfl_step=relative_cfl_step,
            precision_evidence=self.precision.evidence(),
        )

    def _centroid_face_states(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[Array, Array, Array]:
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        left = self.precision.reconstruction(state[owner])
        right = self.precision.reconstruction(state[safe_neighbour])
        normal = (
            self.discretization.area_vectors / self.discretization.face_measures[:, None]
        )
        boundary = neighbour < 0
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
        return left, right, normal

    def _quadrature_face_states(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[Array, Array, Array]:
        reconstruction = self.method.reconstruction
        if not isinstance(
            reconstruction,
            (
                PreparedCellPolynomialReconstruction,
                PreparedUnstructuredWENOZReconstruction,
            ),
        ):
            raise TypeError("Quadrature traces require a prepared cell polynomial.")
        points = self.precision.reconstruction(self.discretization.face_quadrature_points)
        left, right = reconstruction.reconstruct_at(
            self.precision.reconstruction(state), points
        )
        neighbour = self.discretization.neighbour_cells
        normal = (
            self.discretization.area_vectors / self.discretization.face_measures[:, None]
        )
        normal = jnp.broadcast_to(normal[:, None, :], points.shape)
        boundary = neighbour < 0
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
        return left, right, normal

    def face_fluxes(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, Array]:
        value = jnp.asarray(state)
        if value.shape != self.discretization.state_shape:
            raise ValueError(
                f"Unstructured FV state must have shape {self.discretization.state_shape}."
            )
        self.precision.validate_state(value)
        if isinstance(
            self.method.reconstruction,
            (
                PreparedCellPolynomialReconstruction,
                PreparedUnstructuredWENOZReconstruction,
            ),
        ):
            left, right, normal = self._quadrature_face_states(time, value, args)
            result = self.method.interface_solver.normal_face_flux(
                self.system,
                self.precision.flux(left),
                self.precision.flux(right),
                self.precision.flux(normal),
                args,
            )
            integrated = jnp.sum(
                self.precision.reduction(
                    self.discretization.face_quadrature_weights[..., None]
                )
                * self.precision.reduction(result.normal_flux),
                axis=1,
            )
            average_flux = integrated / self.precision.reduction(
                self.discretization.face_measures[:, None]
            )
            return self.precision.flux(average_flux), self.precision.decision(
                jnp.max(result.max_speed, axis=1)
            )
        left, right, normal = self._centroid_face_states(time, value, args)
        result = self.method.interface_solver.normal_face_flux(
            self.system,
            self.precision.flux(left),
            self.precision.flux(right),
            self.precision.flux(normal),
            args,
        )
        return self.precision.flux(result.normal_flux), self.precision.decision(
            result.max_speed
        )

    def residual_from_fluxes(self, normal_flux: Array, /) -> Array:
        integrated = self.precision.reduction(normal_flux) * self.precision.reduction(
            self.discretization.face_measures[:, None]
        )
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        residual = jnp.zeros(
            self.discretization.state_shape,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        residual = residual.at[owner].add(-integrated)
        residual = residual.at[safe_neighbour].add(
            jnp.where((neighbour >= 0)[:, None], integrated, 0.0)
        )
        return self.precision.storage(
            residual / self.precision.reduction(self.discretization.cell_volumes[:, None])
        )

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
            raise ValueError("Unstructured FV source must match the state shape.")
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
        neighbour = self.discretization.neighbour_cells
        rate = rate.at[jnp.maximum(neighbour, 0)].add(
            jnp.where(neighbour >= 0, weighted, 0.0)
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

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        residual, jvp = jax.linearize(lambda value: self(time, value, args), state)
        _, vjp = jax.vjp(lambda value: self(time, value, args), state)
        return residual, jvp, vjp

    def residual_with_diagnostics(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, UnstructuredFiniteVolumeDiagnostics]:
        flux, speed = self.face_fluxes(time, state, args)
        source = self.source_value(time, state, args)
        residual = self(time, state, args)
        boundary = self.discretization.neighbour_cells < 0
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


__all__ = [
    "PreparedUnstructuredFiniteVolumeDynamics",
    "UnstructuredFiniteVolumeBoundarySet",
    "UnstructuredFiniteVolumeDiagnostics",
    "UnstructuredFiniteVolumeMethodPlan",
    "UnstructuredFiniteVolumeStageEvaluation",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._conservation_boundary import evaluate_conservation_boundary
from ...discretization._conservation_policy import (
    DifferentiabilityPolicy,
    validate_differentiability_policy,
)
from ...discretization.fem._boundary import (
    FiniteElementBoundarySet,
    tensor_local_face,
)
from ...discretization.fem._generic import (
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
    IntegrationDomain,
)
from ...discretization.fem._mortar import (
    FiniteElementMortarMetricData,
    FiniteElementMortarPlan,
)
from ...discretization.fem._sbp import (
    ElementLocalSBPData,
    MappedTensorMetricPlan,
    MappedTensorMetrics,
    MetricFacePair,
    TensorGLLSBPPlan,
)
from ...discretization.finite_volume._riemann import (
    AbstractArbitraryNormalNumericalFluxPlan,
    AbstractSymmetricTwoPointFluxPlan,
)
from ...linalg import DiagonalLinearOperator, OperatorProperties
from .._entropy_pair import ConvexEntropyPair
from .._finite_element_variational import (
    CellResidualAction,
    CompiledFiniteElementProblem,
    ExteriorFacetAction,
    FiniteElementExecutionContext,
    FiniteElementExecutionPolicy,
    FiniteElementForm,
    InteriorFacetAction,
    PairwiseVolumeFluxAction,
)
from ._viscous_conservation import (
    PreparedViscousDGOperator,
    ViscousDGPlan,
)


class DGSEMSampledFluxCompatibilityEvidence(StrictModule, NonTrainableState):
    """Sampled compatibility evidence for concrete flux and entropy identities.

    The entropy pair supplies thermodynamic functions; this object records only
    finite-state numerical checks for concrete volume/interface flux plans.
    """

    system_id: str = eqx.field(static=True)
    entropy_pair_id: str = eqx.field(static=True)
    volume_flux_id: str = eqx.field(static=True)
    interface_flux_id: str = eqx.field(static=True)
    symmetry_defect: Array
    consistency_defect: Array
    entropy_potential_defect: Array
    interface_entropy_residual: Array
    tolerance: float = eqx.field(static=True)
    volume_entropy_conservative: bool = eqx.field(static=True)
    interface_entropy_stable: bool = eqx.field(static=True)
    boundary_evidence: str = eqx.field(static=True)
    source_evidence: str = eqx.field(static=True)
    viscous_evidence: str = eqx.field(static=True)
    sampled_periodic_entropy_compatibility: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        system_id: str,
        entropy_pair_id: str,
        volume_flux_id: str,
        interface_flux_id: str,
        symmetry_defect: ArrayLike,
        consistency_defect: ArrayLike,
        entropy_potential_defect: ArrayLike,
        interface_entropy_residual: ArrayLike,
        /,
        *,
        tolerance: float,
        boundary_evidence: str,
        source_evidence: str,
        viscous_evidence: str,
    ):
        identifiers = tuple(
            str(value)
            for value in (
                system_id,
                entropy_pair_id,
                volume_flux_id,
                interface_flux_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Flux compatibility identities must be non-empty.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Flux compatibility tolerance must be positive and finite.")
        boundary = str(boundary_evidence)
        source = str(source_evidence)
        viscous = str(viscous_evidence)
        if boundary not in ("periodic_pair_cancellation", "uncertified"):
            raise ValueError("Unknown DGSEM boundary entropy evidence.")
        if source not in ("absent", "uncertified"):
            raise ValueError("Unknown DGSEM source entropy evidence.")
        if viscous not in ("absent", "uncertified"):
            raise ValueError("Unknown DGSEM viscous entropy evidence.")
        symmetry = jnp.asarray(symmetry_defect)
        consistency = jnp.asarray(consistency_defect)
        potential = jnp.asarray(entropy_potential_defect)
        interface = jnp.asarray(interface_entropy_residual)
        if any(
            np.asarray(value).size == 0
            for value in (symmetry, consistency, potential, interface)
        ):
            raise ValueError("Flux compatibility evidence arrays must be non-empty.")
        symmetry_max = float(np.max(np.abs(np.asarray(symmetry)), initial=0.0))
        consistency_max = float(np.max(np.abs(np.asarray(consistency)), initial=0.0))
        potential_max = float(np.max(np.abs(np.asarray(potential)), initial=0.0))
        interface_upper = float(np.max(np.asarray(interface), initial=-np.inf))
        volume_compatible = bool(
            symmetry_max <= tolerance_
            and consistency_max <= tolerance_
            and potential_max <= tolerance_
        )
        interface_compatible = bool(interface_upper <= tolerance_)
        sampled_periodic_compatible = bool(
            volume_compatible
            and interface_compatible
            and boundary == "periodic_pair_cancellation"
            and source == "absent"
            and viscous == "absent"
        )
        self.system_id = identifiers[0]
        self.entropy_pair_id = identifiers[1]
        self.volume_flux_id = identifiers[2]
        self.interface_flux_id = identifiers[3]
        self.symmetry_defect = symmetry
        self.consistency_defect = consistency
        self.entropy_potential_defect = potential
        self.interface_entropy_residual = interface
        self.tolerance = tolerance_
        self.volume_entropy_conservative = volume_compatible
        self.interface_entropy_stable = interface_compatible
        self.boundary_evidence = boundary
        self.source_evidence = source
        self.viscous_evidence = viscous
        self.sampled_periodic_entropy_compatibility = sampled_periodic_compatible
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "dgsem-sampled-flux-compatibility-evidence",
                "system": identifiers[0],
                "entropy_pair": identifiers[1],
                "volume_flux": identifiers[2],
                "interface_flux": identifiers[3],
                "tolerance": tolerance_,
                "symmetry_defect": array_tree_fingerprint(np.asarray(symmetry)),
                "consistency_defect": array_tree_fingerprint(np.asarray(consistency)),
                "entropy_potential_defect": array_tree_fingerprint(np.asarray(potential)),
                "interface_entropy_residual": array_tree_fingerprint(
                    np.asarray(interface)
                ),
                "sampled_volume_entropy_compatible": volume_compatible,
                "sampled_interface_entropy_compatible": interface_compatible,
                "boundary_evidence": boundary,
                "source_evidence": source,
                "viscous_evidence": viscous,
            }
        )


def sample_dgsem_flux_compatibility(
    system: Any,
    volume_flux: AbstractSymmetricTwoPointFluxPlan,
    interface_flux: AbstractArbitraryNormalNumericalFluxPlan,
    entropy_pair: ConvexEntropyPair,
    left_states: ArrayLike,
    right_states: ArrayLike,
    /,
    *,
    args: Any = None,
    normals: ArrayLike | None = None,
    tolerance: float = 5.0e-11,
    boundary_evidence: str = "periodic_pair_cancellation",
    source_evidence: str = "absent",
    viscous_evidence: str = "absent",
) -> DGSEMSampledFluxCompatibilityEvidence:
    """Evaluate symmetry, consistency, Tadmor identity, and interface inequality."""

    if not isinstance(volume_flux, AbstractSymmetricTwoPointFluxPlan):
        raise TypeError("volume_flux must be a symmetric two-point flux plan.")
    if not isinstance(interface_flux, AbstractArbitraryNormalNumericalFluxPlan):
        raise TypeError("interface_flux must have typed arbitrary-normal capability.")
    if not isinstance(entropy_pair, ConvexEntropyPair):
        raise TypeError("entropy_pair must be ConvexEntropyPair.")
    if entropy_pair.system.system_id != system.system_id:
        raise ValueError("Entropy pair and sampled flux evidence system must match.")
    left = jnp.asarray(left_states)
    right = jnp.asarray(right_states)
    if (
        left.shape != right.shape
        or left.ndim < 2
        or int(np.prod(left.shape[:-1])) == 0
        or left.shape[-1] != system.component_count
    ):
        raise ValueError(
            "Sampled-evidence left/right states need equal batched component shape."
        )
    if not bool(jnp.all(entropy_pair.admissible(left))) or not bool(
        jnp.all(entropy_pair.admissible(right))
    ):
        raise ValueError("Sampled states must lie in the entropy admissible set.")
    symmetry = []
    consistency = []
    potential = []
    interface_residual = []
    for axis in range(system.dimension):
        forward = volume_flux.two_point_flux(system, left, right, axis, args)
        reverse = volume_flux.two_point_flux(system, right, left, axis, args)
        left_diagonal = volume_flux.two_point_flux(system, left, left, axis, args)
        right_diagonal = volume_flux.two_point_flux(system, right, right, axis, args)
        symmetry.append(forward - reverse)
        consistency.append(
            jnp.stack(
                (
                    left_diagonal - system.physical_flux(left, axis, args),
                    right_diagonal - system.physical_flux(right, axis, args),
                ),
                axis=0,
            )
        )
        potential.append(
            entropy_pair.interface_entropy_residual(left, right, forward, axis, args)
        )
        interface_result = interface_flux.face_flux(system, left, right, axis, args)
        interface_residual.append(
            entropy_pair.interface_entropy_residual(
                left,
                right,
                interface_result.normal_flux,
                axis,
                args,
            )
        )
    normal_values = (
        np.concatenate(
            (
                np.eye(system.dimension, dtype=np.asarray(left).dtype),
                np.ones((1, system.dimension), dtype=np.asarray(left).dtype),
            ),
            axis=0,
        )
        if normals is None
        else np.asarray(normals)
    )
    if (
        normal_values.ndim != 2
        or normal_values.shape[1] != system.dimension
        or not np.all(np.isfinite(normal_values))
    ):
        raise ValueError(
            "Sampled normals must have shape (normal_count, system.dimension)."
        )
    normal_lengths = np.linalg.norm(normal_values, axis=-1)
    if np.any(normal_lengths <= 0.0):
        raise ValueError("Sampled normals must be nonzero.")
    normal_values = normal_values / normal_lengths[:, None]
    variables_jump = entropy_pair.entropy_variables(
        right
    ) - entropy_pair.entropy_variables(left)
    for normal_value in normal_values:
        normal = jnp.broadcast_to(
            jnp.asarray(normal_value), left.shape[:-1] + (system.dimension,)
        )
        result = interface_flux.normal_face_flux(system, left, right, normal, args)
        potential_left = sum(
            normal[..., axis] * entropy_pair.entropy_potential(left, axis, args)
            for axis in range(system.dimension)
        )
        potential_right = sum(
            normal[..., axis] * entropy_pair.entropy_potential(right, axis, args)
            for axis in range(system.dimension)
        )
        interface_residual.append(
            ein.contract(
                "...i,...i->...",
                variables_jump,
                result.normal_flux,
                backend="jax",
            )
            - (potential_right - potential_left)
        )
    return DGSEMSampledFluxCompatibilityEvidence(
        system.system_id,
        entropy_pair.pair_id,
        volume_flux.flux_id,
        interface_flux.flux_id,
        jnp.stack(tuple(symmetry), axis=0),
        jnp.stack(tuple(consistency), axis=0),
        jnp.stack(tuple(potential), axis=0),
        jnp.stack(tuple(interface_residual), axis=0),
        tolerance=tolerance,
        boundary_evidence=boundary_evidence,
        source_evidence=source_evidence,
        viscous_evidence=viscous_evidence,
    )


class DGSEMMortarCompatibilityCertificate(StrictModule, NonTrainableState):
    """Mass, constant, geometry, and entropy evidence for one hp mortar."""

    left_mass_error: Array
    right_mass_error: Array
    constant_error: Array
    opposite_normal_error: Array
    entropy_error: Array
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        mortar: FiniteElementMortarPlan,
        metric: FiniteElementMortarMetricData,
        entropy_error: ArrayLike,
        tolerance: float,
        /,
    ):
        if not isinstance(mortar, FiniteElementMortarPlan) or not isinstance(
            metric, FiniteElementMortarMetricData
        ):
            raise TypeError("DGSEM mortar certification requires mortar and metric data.")
        tolerance_ = float(tolerance)
        left = jnp.max(
            jnp.abs(
                mortar.left_mass @ mortar.left_pairing_adjoint
                - mortar.left_weighted_pairing_pullback
            )
        )
        right = jnp.max(
            jnp.abs(
                mortar.right_mass @ mortar.right_pairing_adjoint
                - mortar.right_weighted_pairing_pullback
            )
        )
        constant = jnp.maximum(
            jnp.max(
                jnp.abs(
                    mortar.left_interpolation
                    @ jnp.ones((mortar.left_interpolation.shape[1],))
                    - 1.0
                )
            ),
            jnp.max(
                jnp.abs(
                    mortar.right_interpolation
                    @ jnp.ones((mortar.right_interpolation.shape[1],))
                    - 1.0
                )
            ),
        )
        normals = metric.opposite_normal_error
        entropy = jnp.max(jnp.abs(jnp.asarray(entropy_error)))
        passed = bool(
            max(
                float(left),
                float(right),
                float(constant),
                float(normals),
                float(entropy),
            )
            <= tolerance_
        )
        self.left_mass_error = left
        self.right_mass_error = right
        self.constant_error = constant
        self.opposite_normal_error = normals
        self.entropy_error = entropy
        self.tolerance = tolerance_
        self.passed = passed
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "dgsem-mortar-compatibility",
                "mortar": mortar.plan_id,
                "metric": metric.metric_id,
                "left_mass_error": float(left),
                "right_mass_error": float(right),
                "constant_error": float(constant),
                "normal_error": float(normals),
                "entropy_error": float(entropy),
                "tolerance": tolerance_,
            }
        )


def certify_dgsem_mortar_compatibility(
    mortar: FiniteElementMortarPlan,
    metric: FiniteElementMortarMetricData,
    /,
    *,
    entropy_error: ArrayLike = 0.0,
    tolerance: float = 1.0e-10,
) -> DGSEMMortarCompatibilityCertificate:
    return DGSEMMortarCompatibilityCertificate(
        mortar,
        metric,
        entropy_error,
        tolerance,
    )


class DGSEMMortarFluxLedger(StrictModule):
    left_residual: Array
    right_residual: Array
    conservation_residual: Array
    entropy_flux: Array
    certificate_id: str = eqx.field(static=True)


def dgsem_mortar_flux_ledger(
    mortar: FiniteElementMortarPlan,
    metric: FiniteElementMortarMetricData,
    certificate: DGSEMMortarCompatibilityCertificate,
    flux: ArrayLike,
    /,
    *,
    entropy_flux: ArrayLike | None = None,
) -> DGSEMMortarFluxLedger:
    """Execute one certified mortar flux and accumulate conservation/entropy ledgers."""

    if (
        not isinstance(certificate, DGSEMMortarCompatibilityCertificate)
        or not certificate.passed
    ):
        raise ValueError("Nonconforming DGSEM requires a passing mortar certificate.")
    left, right = mortar.conservative_flux_contributions(flux, metric)
    conservation = jnp.sum(left, axis=0) + jnp.sum(right, axis=0)
    entropy = (
        jnp.zeros_like(conservation)
        if entropy_flux is None
        else mortar.integrated_flux(entropy_flux, metric)
    )
    return DGSEMMortarFluxLedger(
        left,
        right,
        conservation,
        entropy,
        certificate.certificate_id,
    )


class DGSEMNonconformingMortarPlan(StrictModule, NonTrainableState):
    """A set of hp mortars accepted only with explicit entropy certificates."""

    mortars: tuple[FiniteElementMortarPlan, ...]
    metrics: tuple[FiniteElementMortarMetricData, ...]
    certificates: tuple[DGSEMMortarCompatibilityCertificate, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mortars: Sequence[FiniteElementMortarPlan],
        metrics: Sequence[FiniteElementMortarMetricData],
        certificates: Sequence[DGSEMMortarCompatibilityCertificate],
        /,
    ):
        mortars_ = tuple(mortars)
        metrics_ = tuple(metrics)
        certificates_ = tuple(certificates)
        if (
            not mortars_
            or len(mortars_) != len(metrics_)
            or len(mortars_) != len(certificates_)
            or any(not value.passed for value in certificates_)
        ):
            raise ValueError(
                "Every nonconforming DGSEM mortar requires passing evidence."
            )
        self.mortars = mortars_
        self.metrics = metrics_
        self.certificates = certificates_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dgsem-nonconforming-mortars",
                "mortars": [value.plan_id for value in mortars_],
                "metrics": [value.metric_id for value in metrics_],
                "certificates": [value.certificate_id for value in certificates_],
            }
        )

    def ledgers(
        self,
        fluxes: Sequence[ArrayLike],
        /,
        *,
        entropy_fluxes: Sequence[ArrayLike] | None = None,
    ) -> tuple[DGSEMMortarFluxLedger, ...]:
        fluxes_ = tuple(fluxes)
        entropy_ = (
            (None,) * len(fluxes_) if entropy_fluxes is None else tuple(entropy_fluxes)
        )
        if len(fluxes_) != len(self.mortars) or len(entropy_) != len(self.mortars):
            raise ValueError("DGSEM mortar flux counts do not match the plan.")
        return tuple(
            dgsem_mortar_flux_ledger(
                mortar,
                metric,
                certificate,
                flux,
                entropy_flux=entropy_flux,
            )
            for mortar, metric, certificate, flux, entropy_flux in zip(
                self.mortars,
                self.metrics,
                self.certificates,
                fluxes_,
                entropy_,
                strict=True,
            )
        )


class DGSEMConservationMethodPlan(StrictModule, NonTrainableState):
    """Collocated tensor GLL flux-differencing and surface-flux policy."""

    volume_flux: AbstractSymmetricTwoPointFluxPlan
    interface_flux: AbstractArbitraryNormalNumericalFluxPlan
    compatibility: DGSEMSampledFluxCompatibilityEvidence | None
    viscous: ViscousDGPlan | None
    explicit_mass: str = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    differentiability: DifferentiabilityPolicy = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_flux: AbstractSymmetricTwoPointFluxPlan,
        interface_flux: AbstractArbitraryNormalNumericalFluxPlan,
        /,
        *,
        compatibility: DGSEMSampledFluxCompatibilityEvidence | None = None,
        viscous: ViscousDGPlan | None = None,
        explicit_mass: str = "diagonal_gll",
        accumulation: str = "deterministic",
        differentiability: DifferentiabilityPolicy = "branchwise",
    ):
        if not isinstance(volume_flux, AbstractSymmetricTwoPointFluxPlan):
            raise TypeError("DGSEM volume_flux must be a symmetric two-point flux plan.")
        if not volume_flux.symmetric or not volume_flux.consistent:
            raise ValueError("DGSEM volume flux must declare symmetry and consistency.")
        if not isinstance(interface_flux, AbstractArbitraryNormalNumericalFluxPlan):
            raise TypeError(
                "DGSEM interface_flux requires typed arbitrary-normal capability."
            )
        if compatibility is not None and not isinstance(
            compatibility, DGSEMSampledFluxCompatibilityEvidence
        ):
            raise TypeError("compatibility must be sampled DGSEM flux evidence or None.")
        if viscous is not None and not isinstance(viscous, ViscousDGPlan):
            raise TypeError("viscous must be ViscousDGPlan or None.")
        mass = str(explicit_mass)
        if mass != "diagonal_gll":
            raise ValueError(
                "DGSEM explicit dynamics requires the diagonal GLL mass solve."
            )
        accumulation_ = str(accumulation)
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown DGSEM finite-element accumulation policy.")
        differentiability_ = validate_differentiability_policy(differentiability)
        self.volume_flux = volume_flux
        self.interface_flux = interface_flux
        self.compatibility = compatibility
        self.viscous = viscous
        self.explicit_mass = mass
        self.accumulation = accumulation_
        self.differentiability = differentiability_
        self.method_id = canonical_fingerprint(
            {
                "kind": "dgsem-conservation-method",
                "volume_flux": volume_flux.flux_id,
                "interface_flux": interface_flux.flux_id,
                "compatibility": (
                    None if compatibility is None else compatibility.evidence_id
                ),
                "viscous": None if viscous is None else viscous.plan_id,
                "explicit_mass": mass,
                "accumulation": accumulation_,
                "differentiability": differentiability_,
                "shock_capturing": None,
                "positivity": None,
                "motion": None,
            }
        )

    @property
    def entropy_diagnostics(self) -> bool:
        return self.compatibility is not None


class DGSEMPreparationReport(StrictModule, NonTrainableState):
    """Compiler, SBP, metric, mass, and executable-facet provenance."""

    sbp_report_id: str = eqx.field(static=True)
    metric_report_id: str = eqx.field(static=True)
    finite_element_compilation_id: str = eqx.field(static=True)
    action_ir_id: str = eqx.field(static=True)
    workset_program_id: str = eqx.field(static=True)
    kernel_table_id: str = eqx.field(static=True)
    facet_route_count: int = eqx.field(static=True)
    minimum_mass: Array
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        sbp: ElementLocalSBPData,
        metrics: MappedTensorMetrics,
        compiled: CompiledFiniteElementProblem,
        facet_route_count: int,
        minimum_mass: ArrayLike,
        /,
    ):
        minimum = jnp.asarray(minimum_mass)
        passed = bool(
            sbp.report.passed
            and metrics.report.passed
            and int(facet_route_count) > 0
            and float(np.asarray(minimum)) > 0.0
        )
        self.sbp_report_id = sbp.report.report_id
        self.metric_report_id = metrics.report.report_id
        self.finite_element_compilation_id = compiled.compilation_id
        self.action_ir_id = compiled._action_ir.ir_id
        self.workset_program_id = compiled._workset_program.program_id
        self.kernel_table_id = compiled._kernel_table.table_id
        self.facet_route_count = int(facet_route_count)
        self.minimum_mass = minimum
        self.passed = passed
        self.report_id = canonical_fingerprint(
            {
                "kind": "dgsem-preparation-report",
                "sbp": self.sbp_report_id,
                "metrics": self.metric_report_id,
                "finite_element_compilation": self.finite_element_compilation_id,
                "action_ir": self.action_ir_id,
                "worksets": self.workset_program_id,
                "kernels": self.kernel_table_id,
                "facet_routes": self.facet_route_count,
                "passed": passed,
            }
        )


class DGSEMFaceFluxes(StrictModule):
    """Owner-oriented periodic face flux density and integrated ledger."""

    normal_flux: Array
    signal_speed: Array
    surface_jacobian: Array
    integrated_flux: Array
    is_boundary: Array
    boundary_patch_indices: Array
    owner_cells: Array
    neighbour_cells: Array


class DGSEMStableStepEvidence(StrictModule, NonTrainableState):
    step: Array
    maximum_nodal_rate: Array
    maximum_diffusive_rate: Array
    cfl: float = eqx.field(static=True)
    positive: Array
    method_id: str = eqx.field(static=True)


class DGSEMConservationDiagnostics(StrictModule, NonTrainableState):
    boundary_flux_rate: Array
    source_integral: Array
    conservation_balance_defect: Array
    total_integral: Array
    conservation_rate: Array
    total_entropy: Array | None
    semidiscrete_entropy_rate: Array | None
    convective_entropy_rate: Array | None
    boundary_entropy_rate: Array | None
    source_entropy_rate: Array | None
    interface_entropy_production: Array | None
    entropy_inequality_defect: Array | None
    admissible: Array | None
    free_stream_residual: Array
    sampled_evidence_id: str | None = eqx.field(static=True)
    sampled_entropy_inequality: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


def _coordinate_values(
    discretization: FiniteElementDiscretization,
    field_index: int,
    runtime: FiniteElementRuntimeData,
    /,
) -> Array:
    field_element = discretization.elements[field_index][0]
    coordinate_element = discretization.coordinate_elements[0]
    coordinate_routes = discretization.coordinate_dofs[0]
    values = coordinate_element.tabulate(field_element.reference_nodes)[0]
    return ein.contract(
        "qi,cid->cqd",
        values,
        runtime.coordinates[coordinate_routes],
        backend="jax",
    )


def _interior_metric_pairs(
    discretization: FiniteElementDiscretization,
    cell_kind: str,
    /,
) -> tuple[MetricFacePair, ...]:
    domain = discretization.interior_facet_domain
    return tuple(
        MetricFacePair(
            int(owner),
            *tensor_local_face(cell_kind, int(owner_local)),
            int(neighbour),
            *tensor_local_face(cell_kind, int(neighbour_local)),
        )
        for owner, owner_local, neighbour, neighbour_local in zip(
            np.asarray(domain.owner_cells),
            np.asarray(domain.owner_local_entities),
            np.asarray(domain.neighbour_cells),
            np.asarray(domain.neighbour_local_entities),
            strict=True,
        )
    )


def _periodic_metric_pairs(
    discretization: FiniteElementDiscretization,
    cell_kind: str,
    provisional: MappedTensorMetrics,
    tolerance: float,
    /,
) -> tuple[tuple[MetricFacePair, ...], tuple[tuple[int, int], ...]]:
    exterior = discretization.exterior_facet_domain
    facets = np.asarray(exterior.entity_indices, dtype=np.int32)
    owners = np.asarray(exterior.owner_cells, dtype=np.int32)
    local_entities = np.asarray(exterior.owner_local_entities, dtype=np.int32)
    if facets.size % 2:
        raise ValueError("Periodic DGSEM requires an even number of exterior faces.")
    available = list(range(facets.size))
    pairs = []
    facet_pairs = []
    while available:
        first = available.pop(0)
        owner_axis, owner_side = tensor_local_face(cell_kind, int(local_entities[first]))
        best = None
        for candidate in available:
            neighbour_axis, neighbour_side = tensor_local_face(
                cell_kind, int(local_entities[candidate])
            )
            pair = MetricFacePair(
                int(owners[first]),
                owner_axis,
                owner_side,
                int(owners[candidate]),
                neighbour_axis,
                neighbour_side,
                periodic_translation=True,
            )
            _permutation, point_defect, normal_defect = provisional.face_pair_evidence(
                pair
            )
            score = max(
                float(np.asarray(point_defect)),
                float(np.asarray(normal_defect)),
            )
            if best is None or score < best[0]:
                best = score, candidate, pair
        if best is None or best[0] > tolerance:
            raise ValueError(
                "Exterior tensor faces cannot be paired by periodic translation with "
                "opposite compatible scaled normals."
            )
        available.remove(best[1])
        pairs.append(best[2])
        facet_pairs.append((int(facets[first]), int(facets[best[1]])))
    return tuple(pairs), tuple(facet_pairs)


def _explicit_periodic_metric_pairs(
    discretization: FiniteElementDiscretization,
    cell_kind: str,
    provisional: MappedTensorMetrics,
    boundaries: FiniteElementBoundarySet,
    tolerance: float,
    /,
) -> tuple[tuple[MetricFacePair, ...], tuple[tuple[int, int], ...]]:
    exterior = discretization.exterior_facet_domain
    facets = np.asarray(exterior.entity_indices, dtype=np.int32)
    owners = np.asarray(exterior.owner_cells, dtype=np.int32)
    local_entities = np.asarray(exterior.owner_local_entities, dtype=np.int32)
    positions = {int(facet): index for index, facet in enumerate(facets)}
    metric_pairs = []
    facet_pairs = []
    for boundary_pair in boundaries.periodic_pairs:
        owner_position = positions[boundary_pair.owner_facet]
        neighbour_position = positions[boundary_pair.neighbour_facet]
        owner_axis, owner_side = tensor_local_face(
            cell_kind, int(local_entities[owner_position])
        )
        neighbour_axis, neighbour_side = tensor_local_face(
            cell_kind, int(local_entities[neighbour_position])
        )
        pair = MetricFacePair(
            int(owners[owner_position]),
            owner_axis,
            owner_side,
            int(owners[neighbour_position]),
            neighbour_axis,
            neighbour_side,
            periodic_translation=True,
        )
        _permutation, point_defect, normal_defect = provisional.face_pair_evidence(pair)
        if (
            max(
                float(np.asarray(point_defect)),
                float(np.asarray(normal_defect)),
            )
            > tolerance
        ):
            raise ValueError(
                "Explicit periodic facets have incompatible points or scaled normals."
            )
        metric_pairs.append(pair)
        facet_pairs.append((boundary_pair.owner_facet, boundary_pair.neighbour_facet))
    return tuple(metric_pairs), tuple(facet_pairs)


def _exterior_local_domain(
    domain: IntegrationDomain,
    local_facet: int,
    /,
    *,
    selection_id: str,
) -> IntegrationDomain:
    local_entities = np.asarray(domain.owner_local_entities, dtype=np.int32)
    positions = np.flatnonzero(local_entities == int(local_facet))
    if positions.size == 0:
        raise ValueError("Physical boundary subdomain contains no selected facets.")
    return IntegrationDomain(
        "exterior_facet",
        np.asarray(domain.entity_indices)[positions],
        domain.support_id,
        domain.entity_set_id,
        owner_cells=np.asarray(domain.owner_cells)[positions],
        neighbour_cells=np.asarray(domain.neighbour_cells)[positions],
        owner_local_entities=local_entities[positions],
        neighbour_local_entities=np.asarray(domain.neighbour_local_entities)[positions],
        selection_id=selection_id,
    )


def _periodic_facet_domain(
    discretization: FiniteElementDiscretization,
    cell_kind: str,
    periodic_pairs: Sequence[MetricFacePair],
    periodic_facet_ids: Sequence[tuple[int, int]],
    face_permutations: Sequence[ArrayLike],
    /,
    *,
    selection_id: str,
) -> IntegrationDomain:
    interior = discretization.interior_facet_domain
    entity_indices = list(np.asarray(interior.entity_indices, dtype=np.int32))
    owners = list(np.asarray(interior.owner_cells, dtype=np.int32))
    neighbours = list(np.asarray(interior.neighbour_cells, dtype=np.int32))
    owner_local = list(np.asarray(interior.owner_local_entities, dtype=np.int32))
    neighbour_local = list(np.asarray(interior.neighbour_local_entities, dtype=np.int32))
    inverse_local = {
        face: local
        for local, face in enumerate(
            {
                "quadrilateral": ((1, 0), (0, 1), (1, 1), (0, 0)),
                "hexahedron": ((2, 0), (2, 1), (1, 0), (0, 1), (1, 1), (0, 0)),
            }[cell_kind]
        )
    }
    for pair, facet_pair in zip(periodic_pairs, periodic_facet_ids, strict=True):
        entity_indices.append(facet_pair[0])
        owners.append(pair.owner_cell)
        neighbours.append(pair.neighbour_cell)
        owner_local.append(inverse_local[(pair.owner_axis, pair.owner_side)])
        neighbour_local.append(inverse_local[(pair.neighbour_axis, pair.neighbour_side)])
    periodic_mask = np.concatenate(
        (
            np.zeros((int(interior.entity_indices.shape[0]),), dtype=bool),
            np.ones((len(periodic_pairs),), dtype=bool),
        )
    )
    return IntegrationDomain(
        "interior_facet",
        np.asarray(entity_indices, dtype=np.int32),
        interior.support_id,
        interior.entity_set_id,
        owner_cells=np.asarray(owners, dtype=np.int32),
        neighbour_cells=np.asarray(neighbours, dtype=np.int32),
        owner_local_entities=np.asarray(owner_local, dtype=np.int32),
        neighbour_local_entities=np.asarray(neighbour_local, dtype=np.int32),
        neighbour_trace_permutations=np.stack(
            tuple(np.asarray(value) for value in face_permutations), axis=0
        ),
        periodic_face_mask=periodic_mask,
        selection_id=selection_id,
    )


def _tensor_mass_weights(sbp: ElementLocalSBPData, dimension: int, /) -> Array:
    result = sbp.norm_weights
    for _axis in range(1, dimension):
        result = ein.contract("...i,j->...ij", result, sbp.norm_weights, backend="jax")
    return result


def _rules(cell_kind: str, node_count: int, /):
    from ...integration._rules import (
        GaussLobattoLegendreRule,
        ReferenceHexahedronRule,
        ReferenceIntervalRule,
        ReferenceQuadrilateralRule,
    )

    axis = GaussLobattoLegendreRule(node_count)
    if cell_kind == "quadrilateral":
        return ReferenceQuadrilateralRule(axis), ReferenceIntervalRule(axis)
    if cell_kind == "hexahedron":
        return ReferenceHexahedronRule(axis), ReferenceQuadrilateralRule(axis)
    raise ValueError("DGSEM rules require quadrilateral or hexahedron cells.")


class PreparedDGSEMConservationDynamics(StrictModule):
    """Mapped quad/hex DGSEM with explicit periodic and physical facets."""

    system: Any
    discretization: FiniteElementDiscretization
    method: DGSEMConservationMethodPlan
    entropy_pair: ConvexEntropyPair | None
    boundaries: FiniteElementBoundarySet | None
    source: Callable | None = eqx.field(static=True)
    runtime: FiniteElementRuntimeData
    sbp: ElementLocalSBPData
    metrics: MappedTensorMetrics
    face_pairs: tuple[MetricFacePair, ...]
    face_permutations: tuple[Array, ...]
    compiled_finite_element_problem: CompiledFiniteElementProblem
    mass_operator: DiagonalLinearOperator
    inverse_mass_operator: DiagonalLinearOperator
    scalar_mass_weights: Array
    report: DGSEMPreparationReport
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: FiniteElementDiscretization,
        method: DGSEMConservationMethodPlan,
        /,
        *,
        source: Callable | None = None,
        boundaries: FiniteElementBoundarySet | None = None,
        entropy_pair: ConvexEntropyPair | None = None,
        runtime: FiniteElementRuntimeData | None = None,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("DGSEM requires FiniteElementDiscretization.")
        if not isinstance(method, DGSEMConservationMethodPlan):
            raise TypeError("method must be DGSEMConservationMethodPlan.")
        if source is not None and not callable(source):
            raise TypeError("DGSEM source must be callable or None.")
        if entropy_pair is not None and not isinstance(entropy_pair, ConvexEntropyPair):
            raise TypeError("entropy_pair must be ConvexEntropyPair or None.")
        if boundaries is not None:
            if not isinstance(boundaries, FiniteElementBoundarySet):
                raise TypeError(
                    "DGSEM boundaries must be FiniteElementBoundarySet or None."
                )
            if (
                boundaries.support_id != discretization.support.support_id
                or boundaries.entity_set_id
                != discretization.exterior_facet_domain.entity_set_id
            ):
                raise ValueError(
                    "DGSEM boundary ownership does not match the FE discretization."
                )
        realized = discretization.default_runtime if runtime is None else runtime
        if not isinstance(realized, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData or None.")
        if (
            realized.topology_id != discretization.mesh.topology_id
            or realized.geometry_layout_id
            != discretization.default_runtime.geometry_layout_id
        ):
            raise ValueError("DGSEM runtime does not match the prepared FE layout.")
        if len(discretization.mesh.blocks) != 1:
            raise ValueError("Initial DGSEM requires one homogeneous tensor cell block.")
        block = discretization.mesh.blocks[0]
        if block.cell_kind not in ("quadrilateral", "hexahedron"):
            raise ValueError("DGSEM supports mapped quadrilateral and hexahedron cells.")
        if discretization.mesh.topological_dimension != system.dimension:
            raise ValueError("DGSEM mesh and conservation-system dimensions must match.")
        if len(discretization.field_spaces) != 1:
            raise ValueError("Initial DGSEM requires exactly one conserved FE field.")
        field_index = discretization._field_index(discretization.field_spaces[0].name)
        if field_index != 0:
            raise ValueError("DGSEM conserved field ordering is inconsistent.")
        dof_map = discretization.dof_maps[field_index]
        element = discretization.elements[field_index][0]
        if dof_map.association != "cell" or element.conformity != "L2":
            raise ValueError(
                "DGSEM conserved state requires discontinuous cell-local DOFs."
            )
        if (
            element.representation != "point_value"
            or discretization.field_spaces[0].representation != "point_value"
        ):
            raise ValueError("DGSEM requires point-value finite-element representation.")
        if dof_map.component_shape != (system.component_count,):
            raise ValueError(
                "DGSEM field component shape must match the conservation system."
            )
        if element.degree < 1:
            raise ValueError("DGSEM requires polynomial degree >= 1.")
        sbp = TensorGLLSBPPlan(element.degree).prepare()
        expected_nodes = np.stack(
            np.meshgrid(
                *(np.asarray(sbp.nodes),) * system.dimension,
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, system.dimension))
        if not np.allclose(
            np.asarray(element.reference_nodes),
            expected_nodes,
            rtol=0.0,
            atol=sbp.report.tolerance,
        ):
            raise ValueError("DGSEM element nodes must be the tensor GLL nodes.")
        coordinate_values = _coordinate_values(discretization, field_index, realized)
        metric_plan = MappedTensorMetricPlan(sbp, system.dimension)
        provisional = metric_plan.prepare(coordinate_values)
        interior_pairs = _interior_metric_pairs(discretization, block.cell_kind)
        if boundaries is None:
            periodic_pairs, periodic_facet_ids = _periodic_metric_pairs(
                discretization,
                block.cell_kind,
                provisional,
                metric_plan.tolerance,
            )
            facet_selection_id = "dgsem-periodic-all-faces"
        else:
            periodic_pairs, periodic_facet_ids = _explicit_periodic_metric_pairs(
                discretization,
                block.cell_kind,
                provisional,
                boundaries,
                metric_plan.tolerance,
            )
            facet_selection_id = boundaries.boundary_set_id
        all_pairs = interior_pairs + periodic_pairs
        metrics = metric_plan.prepare(coordinate_values, face_pairs=all_pairs)
        face_permutations = tuple(
            metrics.face_pair_evidence(pair)[0] for pair in all_pairs
        )
        facet_domain = (
            _periodic_facet_domain(
                discretization,
                block.cell_kind,
                periodic_pairs,
                periodic_facet_ids,
                face_permutations,
                selection_id=facet_selection_id,
            )
            if all_pairs
            else None
        )
        volume_rule, facet_rule = _rules(block.cell_kind, sbp.node_count)

        def volume_kernel(left, right, x_left, x_right, context):
            del x_left, x_right
            return jnp.stack(
                tuple(
                    method.volume_flux.two_point_flux(
                        system,
                        left,
                        right,
                        axis,
                        context.user_args,
                    )
                    for axis in range(system.dimension)
                ),
                axis=-1,
            )

        def interface_kernel(plus_values, minus_values, points, weights, normal, context):
            del points, weights
            plus = plus_values[0]
            minus = minus_values[0]
            numerical = method.interface_flux.normal_face_flux(
                system,
                plus,
                minus,
                normal,
                context.user_args,
            ).normal_flux
            plus_physical = system.physical_normal_flux(plus, normal, context.user_args)
            minus_physical = system.physical_normal_flux(minus, normal, context.user_args)
            return numerical - plus_physical, -numerical + minus_physical

        actions = [
            PairwiseVolumeFluxAction(
                discretization.field_spaces[0].name,
                volume_kernel,
                domain=discretization.cell_domain,
                rules=((block.name, volume_rule),),
                action_id="dgsem-pairwise-volume-flux",
            )
        ]
        if facet_domain is not None:
            actions.append(
                InteriorFacetAction(
                    discretization.field_spaces[0].name,
                    (discretization.field_spaces[0].name,),
                    interface_kernel,
                    domain=facet_domain,
                    rules=((block.name, facet_rule),),
                    action_id="dgsem-periodic-interface-correction",
                )
            )
        if boundaries is not None:

            def physical_boundary_kernel(boundary, axis):
                def kernel(plus_values, points, weights, normal, context):
                    del weights
                    plus = plus_values[0]
                    plus_physical = system.physical_normal_flux(
                        plus, normal, context.user_args
                    )
                    trace = evaluate_conservation_boundary(
                        boundary,
                        system,
                        context.time,
                        plus,
                        points,
                        normal,
                        axis,
                        context.user_args,
                    )
                    if trace.direct_normal_flux is not None:
                        numerical = trace.direct_normal_flux
                    else:
                        if trace.exterior_state is None:
                            raise RuntimeError(
                                "Boundary trace supplied neither state nor normal flux."
                            )
                        numerical = method.interface_flux.normal_face_flux(
                            system,
                            plus,
                            trace.exterior_state,
                            normal,
                            context.user_args,
                        ).normal_flux
                    return numerical - plus_physical

                return kernel

            for patch in boundaries.patches:
                local_facets = tuple(
                    sorted(
                        {
                            int(value)
                            for value in np.asarray(
                                patch.domain.owner_local_entities, dtype=np.int32
                            )
                        }
                    )
                )
                for local_facet in local_facets:
                    axis, _side = tensor_local_face(block.cell_kind, local_facet)
                    domain = _exterior_local_domain(
                        patch.domain,
                        local_facet,
                        selection_id=canonical_fingerprint(
                            {
                                "kind": "dgsem-physical-boundary-subdomain",
                                "patch": patch.patch_id,
                                "local_facet": local_facet,
                            }
                        ),
                    )
                    actions.append(
                        ExteriorFacetAction(
                            discretization.field_spaces[0].name,
                            (discretization.field_spaces[0].name,),
                            physical_boundary_kernel(patch.boundary, axis),
                            domain=domain,
                            rules=((block.name, facet_rule),),
                            action_id=canonical_fingerprint(
                                {
                                    "kind": "dgsem-physical-boundary-action",
                                    "patch": patch.patch_id,
                                    "local_facet": local_facet,
                                }
                            ),
                        )
                    )
        if source is not None:

            def source_kernel(
                values,
                gradients,
                points,
                physical_weights,
                test_basis,
                test_gradients,
                context,
            ):
                del gradients, test_gradients
                source_values = jnp.asarray(
                    source(
                        context.time,
                        values[0],
                        points,
                        context.user_args,
                    )
                )
                if source_values.shape != values[0].shape:
                    raise ValueError(
                        "DGSEM source must return the collocated conserved-state shape."
                    )
                return -ein.contract(
                    "cq,cq...,qi->ci...",
                    physical_weights,
                    source_values,
                    test_basis,
                    backend="jax",
                )

            actions.append(
                CellResidualAction(
                    discretization.field_spaces[0].name,
                    (discretization.field_spaces[0].name,),
                    source_kernel,
                    domain=discretization.cell_domain,
                    rules=((block.name, volume_rule),),
                    action_id="dgsem-source",
                )
            )
        form = FiniteElementForm(
            "dgsem-conservation",
            discretization.field_spaces[0].name,
            tuple(actions),
        )
        execution_policy = FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="collocated",
            accumulation=method.accumulation,
        )
        compiled = CompiledFiniteElementProblem(
            form,
            discretization,
            execution_policy=execution_policy,
        )
        tensor_weights = _tensor_mass_weights(sbp, system.dimension)
        local_mass = metrics.determinant * tensor_weights[None, ...]
        scalar_mass = jnp.zeros((dof_map.global_dof_count,), dtype=local_mass.dtype)
        scalar_mass = scalar_mass.at[dof_map.cell_dofs[0]].set(
            local_mass.reshape((block.cell_count, -1))
        )
        state_space = discretization.field_spaces[0].vector_space
        state_shape = state_space.structure().shape
        full_mass = jnp.broadcast_to(scalar_mass[:, None], state_shape).reshape((-1,))
        properties = OperatorProperties(
            diagonal=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "diagonal": "construction",
                "self_adjoint": "construction",
                "positive_definite": "verified",
            },
        )
        mass_operator = DiagonalLinearOperator(
            full_mass,
            space=state_space,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "dgsem-diagonal-mass",
                    "discretization": discretization.prepared_id,
                    "metrics": metrics.metrics_id,
                }
            ),
        )
        inverse_mass_operator = DiagonalLinearOperator(
            jnp.reciprocal(full_mass),
            space=state_space,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "dgsem-explicit-inverse-mass",
                    "mass": mass_operator.operator_id,
                }
            ),
        )
        certificate = method.compatibility
        if certificate is not None:
            if (
                certificate.system_id != system.system_id
                or certificate.volume_flux_id != method.volume_flux.flux_id
                or certificate.interface_flux_id != method.interface_flux.flux_id
                or entropy_pair is None
                or certificate.entropy_pair_id != entropy_pair.pair_id
            ):
                raise ValueError(
                    "DGSEM flux certificate does not match system, fluxes, and entropy pair."
                )
            if source is not None and certificate.source_evidence == "absent":
                raise ValueError(
                    "An entropy certificate advertising absent source cannot compile a source."
                )
            if method.viscous is not None and certificate.viscous_evidence == "absent":
                raise ValueError(
                    "An entropy certificate advertising absent viscosity cannot "
                    "compile a viscous DG operator."
                )
        elif entropy_pair is not None:
            raise ValueError(
                "DGSEM entropy diagnostics require an explicit flux compatibility certificate."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-dgsem-conservation-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "finite_element_compilation": compiled.compilation_id,
                "sbp": sbp.data_id,
                "metrics": metrics.metrics_id,
                "source": "none" if source is None else repr(source),
                "boundaries": (
                    None if boundaries is None else boundaries.boundary_set_id
                ),
                "viscous": (None if method.viscous is None else method.viscous.plan_id),
                "entropy_pair": None if entropy_pair is None else entropy_pair.pair_id,
            }
        )
        physical_facet_count = (
            0
            if boundaries is None
            else sum(
                int(patch.domain.entity_indices.shape[0]) for patch in boundaries.patches
            )
        )
        report = DGSEMPreparationReport(
            sbp,
            metrics,
            compiled,
            len(all_pairs) + physical_facet_count,
            jnp.min(scalar_mass),
        )
        if not report.passed:
            raise RuntimeError("DGSEM preparation evidence failed.")
        self.system = system
        self.discretization = discretization
        self.method = method
        self.entropy_pair = entropy_pair
        self.boundaries = boundaries
        self.source = source
        self.runtime = realized
        self.sbp = sbp
        self.metrics = metrics
        self.face_pairs = all_pairs
        self.face_permutations = face_permutations
        self.compiled_finite_element_problem = compiled
        self.mass_operator = mass_operator
        self.inverse_mass_operator = inverse_mass_operator
        self.scalar_mass_weights = scalar_mass
        self.report = report
        self.dynamics_id = identifier

    @property
    def state_space(self):
        return self.compiled_finite_element_problem.state_space

    @property
    def residual_space(self):
        return self.compiled_finite_element_problem.residual_space

    @property
    def viscous_operator(self) -> PreparedViscousDGOperator | None:
        return (
            None
            if self.method.viscous is None
            else PreparedViscousDGOperator(self.method.viscous, self)
        )

    def _state(self, state: ArrayLike, /) -> Array:
        return self.state_space.validate(jnp.asarray(state))

    def _context(self, time: Array, args: Any, /) -> FiniteElementExecutionContext:
        if isinstance(args, FiniteElementExecutionContext):
            if args.runtime.runtime_id != self.runtime.runtime_id:
                raise ValueError(
                    "Stationary DGSEM dynamics cannot change mapped runtime coordinates."
                )
            return FiniteElementExecutionContext(
                args.runtime,
                time=time,
                lift=args.lift,
                lift_rate=args.lift_rate,
                lift_acceleration=args.lift_acceleration,
                metric_data=self.metrics,
                user_args=args.user_args,
            )
        return FiniteElementExecutionContext(
            self.runtime,
            time=time,
            metric_data=self.metrics,
            user_args=args,
        )

    def weak_residual(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        """Return the FE dual residual before any mass inversion."""

        value = self._state(state)
        context = self._context(jnp.asarray(time), args)
        residual = self.compiled_finite_element_problem.residual(value, context)
        if self.viscous_operator is not None:
            residual = residual + self.viscous_operator.weak_residual(
                jnp.asarray(time), value, args
            )
        return residual

    def mass_inverted_rate(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> Array:
        """Apply the explicitly selected diagonal GLL mass inverse to ``-R``."""

        residual = self.weak_residual(time, state, args)
        return self.inverse_mass_operator.mv(-residual)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        return self.mass_inverted_rate(time, state, args)

    def _local_state(self, state: Array, /) -> Array:
        routes = self.discretization.dof_maps[0].cell_dofs[0]
        node_count = self.sbp.node_count
        return state[routes].reshape(
            (routes.shape[0],) + (node_count,) * self.metrics.dimension + state.shape[1:]
        )

    def _face_value(
        self, local_state: Array, cell: int, axis: int, side: int, /
    ) -> Array:
        return jnp.take(local_state[cell], 0 if side == 0 else -1, axis=axis).reshape(
            (-1,) + local_state.shape[-1:]
        )

    def face_fluxes(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> DGSEMFaceFluxes:
        time_ = jnp.asarray(time)
        value = self._state(state)
        local = self._local_state(value)
        user_args = (
            args.user_args if isinstance(args, FiniteElementExecutionContext) else args
        )
        fluxes = []
        speeds = []
        measures = []
        integrated = []
        owner_cells = []
        neighbour_cells = []
        is_boundary = []
        boundary_patch_indices = []
        face_weight = _tensor_mass_weights(self.sbp, self.metrics.dimension - 1).reshape(
            (-1,)
        )
        for pair, permutation in zip(
            self.face_pairs, self.face_permutations, strict=True
        ):
            plus = self._face_value(
                local,
                pair.owner_cell,
                pair.owner_axis,
                pair.owner_side,
            )
            minus = self._face_value(
                local,
                pair.neighbour_cell,
                pair.neighbour_axis,
                pair.neighbour_side,
            )[permutation]
            scaled_normal = self.metrics.face_scaled_normals[pair.owner_axis][
                pair.owner_cell, pair.owner_side
            ].reshape((-1, self.metrics.dimension))
            surface_jacobian = jnp.sqrt(
                ein.contract("qd,qd->q", scaled_normal, scaled_normal, backend="jax")
            )
            normal = scaled_normal / surface_jacobian[:, None]
            result = self.method.interface_flux.normal_face_flux(
                self.system,
                plus,
                minus,
                normal,
                user_args,
            )
            fluxes.append(result.normal_flux)
            speeds.append(result.max_speed)
            measures.append(surface_jacobian)
            integrated.append(
                ein.contract(
                    "q,q,qi->i",
                    face_weight,
                    surface_jacobian,
                    result.normal_flux,
                    backend="jax",
                )
            )
            owner_cells.append(pair.owner_cell)
            neighbour_cells.append(pair.neighbour_cell)
            is_boundary.append(False)
            boundary_patch_indices.append(-1)
        if self.boundaries is not None:
            for patch_index, patch in enumerate(self.boundaries.patches):
                owners = np.asarray(patch.domain.owner_cells, dtype=np.int32)
                local_facets = np.asarray(
                    patch.domain.owner_local_entities, dtype=np.int32
                )
                for owner_cell, local_facet in zip(owners, local_facets, strict=True):
                    axis, side = tensor_local_face(
                        self.discretization.mesh.blocks[0].cell_kind,
                        int(local_facet),
                    )
                    plus = self._face_value(local, int(owner_cell), axis, side)
                    points = self.metrics.face_coordinates[axis][
                        int(owner_cell), side
                    ].reshape((-1, self.metrics.dimension))
                    scaled_normal = self.metrics.face_scaled_normals[axis][
                        int(owner_cell), side
                    ].reshape((-1, self.metrics.dimension))
                    surface_jacobian = jnp.sqrt(
                        ein.contract(
                            "qd,qd->q",
                            scaled_normal,
                            scaled_normal,
                            backend="jax",
                        )
                    )
                    normal = scaled_normal / surface_jacobian[:, None]
                    trace = evaluate_conservation_boundary(
                        patch.boundary,
                        self.system,
                        time_,
                        plus,
                        points,
                        normal,
                        axis,
                        user_args,
                    )
                    if trace.direct_normal_flux is not None:
                        normal_flux = trace.direct_normal_flux
                        speed = self.system.max_normal_wave_speed(
                            plus, plus, normal, user_args
                        )
                    else:
                        if trace.exterior_state is None:
                            raise RuntimeError(
                                "Boundary trace supplied neither state nor normal flux."
                            )
                        result = self.method.interface_flux.normal_face_flux(
                            self.system,
                            plus,
                            trace.exterior_state,
                            normal,
                            user_args,
                        )
                        normal_flux = result.normal_flux
                        speed = result.max_speed
                    fluxes.append(normal_flux)
                    speeds.append(speed)
                    measures.append(surface_jacobian)
                    integrated.append(
                        ein.contract(
                            "q,q,qi->i",
                            face_weight,
                            surface_jacobian,
                            normal_flux,
                            backend="jax",
                        )
                    )
                    owner_cells.append(int(owner_cell))
                    neighbour_cells.append(-1)
                    is_boundary.append(True)
                    boundary_patch_indices.append(patch_index)
        return DGSEMFaceFluxes(
            normal_flux=jnp.stack(tuple(fluxes), axis=0),
            signal_speed=jnp.stack(tuple(speeds), axis=0),
            surface_jacobian=jnp.stack(tuple(measures), axis=0),
            integrated_flux=jnp.stack(tuple(integrated), axis=0),
            owner_cells=jnp.asarray(owner_cells, dtype=jnp.int32),
            neighbour_cells=jnp.asarray(neighbour_cells, dtype=jnp.int32),
            is_boundary=jnp.asarray(is_boundary, dtype=bool),
            boundary_patch_indices=jnp.asarray(boundary_patch_indices, dtype=jnp.int32),
        )

    def _source_values(self, time: Array, state: Array, args: Any, /) -> Array:
        if self.source is None:
            return jnp.zeros_like(state)
        local = self._local_state(state)
        source = jnp.asarray(self.source(time, local, self.metrics.coordinates, args))
        if source.shape != local.shape:
            raise ValueError("DGSEM source must match the local collocated state shape.")
        routes = self.discretization.dof_maps[0].cell_dofs[0]
        result = jnp.zeros_like(state)
        return result.at[routes].set(source.reshape(state[routes].shape))

    def residual_with_diagnostics(
        self,
        time: Array,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, DGSEMConservationDiagnostics]:
        value = self._state(state)
        rate = self.mass_inverted_rate(time, value, args)
        scalar_mass = self.scalar_mass_weights
        total_integral = ein.contract("n,ni->i", scalar_mass, value, backend="jax")
        conservation_rate = ein.contract("n,ni->i", scalar_mass, rate, backend="jax")
        user_args = (
            args.user_args if isinstance(args, FiniteElementExecutionContext) else args
        )
        source_values = self._source_values(jnp.asarray(time), value, user_args)
        source_integral = ein.contract(
            "n,ni->i", scalar_mass, source_values, backend="jax"
        )
        faces = self.face_fluxes(time, value, args)
        boundary_flux_rate = jnp.sum(
            jnp.where(
                faces.is_boundary[:, None],
                faces.integrated_flux,
                jnp.zeros_like(faces.integrated_flux),
            ),
            axis=0,
        )
        balance_defect = conservation_rate + boundary_flux_rate - source_integral
        total_entropy = None
        entropy_rate = None
        convective_rate = None
        boundary_entropy_rate = None
        source_rate = None
        interface_production = None
        inequality_defect = None
        admissible = None
        certificate = self.method.compatibility
        if self.entropy_pair is not None:
            entropy_variables = self.entropy_pair.entropy_variables(value)
            total_entropy = jnp.sum(scalar_mass * self.entropy_pair.entropy(value))
            entropy_rate = jnp.sum(
                scalar_mass
                * ein.contract("ni,ni->n", entropy_variables, rate, backend="jax")
            )
            source_rate = jnp.sum(
                scalar_mass
                * ein.contract(
                    "ni,ni->n",
                    entropy_variables,
                    source_values,
                    backend="jax",
                )
            )
            convective_rate = entropy_rate - source_rate
            local = self._local_state(value)
            face_weight = _tensor_mass_weights(
                self.sbp, self.metrics.dimension - 1
            ).reshape((-1,))
            productions = []
            for face_index, (pair, permutation) in enumerate(
                zip(self.face_pairs, self.face_permutations, strict=True)
            ):
                plus = self._face_value(
                    local, pair.owner_cell, pair.owner_axis, pair.owner_side
                )
                minus = self._face_value(
                    local,
                    pair.neighbour_cell,
                    pair.neighbour_axis,
                    pair.neighbour_side,
                )[permutation]
                scaled_normal = self.metrics.face_scaled_normals[pair.owner_axis][
                    pair.owner_cell, pair.owner_side
                ].reshape((-1, self.metrics.dimension))
                surface_jacobian = faces.surface_jacobian[face_index]
                normal = scaled_normal / surface_jacobian[:, None]
                variables_jump = self.entropy_pair.entropy_variables(
                    minus
                ) - self.entropy_pair.entropy_variables(plus)
                potential_plus = sum(
                    normal[:, axis]
                    * self.entropy_pair.entropy_potential(plus, axis, user_args)
                    for axis in range(self.metrics.dimension)
                )
                potential_minus = sum(
                    normal[:, axis]
                    * self.entropy_pair.entropy_potential(minus, axis, user_args)
                    for axis in range(self.metrics.dimension)
                )
                density = ein.contract(
                    "qi,qi->q",
                    variables_jump,
                    faces.normal_flux[face_index],
                    backend="jax",
                ) - (potential_minus - potential_plus)
                productions.append(jnp.sum(face_weight * surface_jacobian * density))
            interface_production = sum(
                productions, jnp.asarray(0.0, dtype=entropy_rate.dtype)
            )
            boundary_terms = []
            if self.boundaries is not None:
                face_index = len(self.face_pairs)
                for patch in self.boundaries.patches:
                    owners = np.asarray(patch.domain.owner_cells, dtype=np.int32)
                    local_facets = np.asarray(
                        patch.domain.owner_local_entities, dtype=np.int32
                    )
                    for owner_cell, local_facet in zip(owners, local_facets, strict=True):
                        axis, side = tensor_local_face(
                            self.discretization.mesh.blocks[0].cell_kind,
                            int(local_facet),
                        )
                        plus = self._face_value(local, int(owner_cell), axis, side)
                        scaled_normal = self.metrics.face_scaled_normals[axis][
                            int(owner_cell), side
                        ].reshape((-1, self.metrics.dimension))
                        surface_jacobian = faces.surface_jacobian[face_index]
                        normal = scaled_normal / surface_jacobian[:, None]
                        potential = sum(
                            normal[:, direction]
                            * self.entropy_pair.entropy_potential(
                                plus, direction, user_args
                            )
                            for direction in range(self.metrics.dimension)
                        )
                        entropy_flux = (
                            ein.contract(
                                "qi,qi->q",
                                self.entropy_pair.entropy_variables(plus),
                                faces.normal_flux[face_index],
                                backend="jax",
                            )
                            - potential
                        )
                        boundary_terms.append(
                            jnp.sum(face_weight * surface_jacobian * entropy_flux)
                        )
                        face_index += 1
            boundary_entropy_rate = sum(
                boundary_terms, jnp.asarray(0.0, dtype=entropy_rate.dtype)
            )
            inequality_defect = jnp.maximum(convective_rate + boundary_entropy_rate, 0.0)
            admissible = jnp.all(self.entropy_pair.admissible(value))
        diagnostics = DGSEMConservationDiagnostics(
            total_integral=total_integral,
            conservation_rate=conservation_rate,
            boundary_flux_rate=boundary_flux_rate,
            source_integral=source_integral,
            conservation_balance_defect=balance_defect,
            total_entropy=total_entropy,
            semidiscrete_entropy_rate=entropy_rate,
            convective_entropy_rate=convective_rate,
            boundary_entropy_rate=boundary_entropy_rate,
            source_entropy_rate=source_rate,
            interface_entropy_production=interface_production,
            entropy_inequality_defect=inequality_defect,
            admissible=admissible,
            free_stream_residual=self.metrics.report.free_stream_residual,
            sampled_evidence_id=(
                None if certificate is None else certificate.evidence_id
            ),
            sampled_entropy_inequality=bool(
                certificate is not None
                and certificate.sampled_periodic_entropy_compatibility
                and (self.boundaries is None or not self.boundaries.patches)
            ),
            method_id=self.method.method_id,
        )
        return rate, diagnostics

    def stable_step_evidence(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> DGSEMStableStepEvidence:
        value = self._state(state)
        local = self._local_state(value)
        user_args = (
            args.user_args if isinstance(args, FiniteElementExecutionContext) else args
        )
        row_bound = jnp.max(jnp.sum(jnp.abs(self.sbp.derivative_matrix), axis=1))
        nodal_rate = jnp.zeros_like(self.metrics.determinant)
        for axis in range(self.metrics.dimension):
            cofactor = self.metrics.contravariant_cofactors[..., axis, :]
            scale = jnp.sqrt(
                ein.contract("c...d,c...d->c...", cofactor, cofactor, backend="jax")
            )
            normal = cofactor / scale[..., None]
            speed = self.system.max_normal_wave_speed(local, local, normal, user_args)
            nodal_rate = nodal_rate + row_bound * scale * speed
        nodal_rate = nodal_rate / self.metrics.determinant
        maximum = jnp.max(nodal_rate)
        if self.boundaries is not None and self.boundaries.patches:
            faces = self.face_fluxes(jnp.asarray(0.0, dtype=value.dtype), value, args)
            face_index = len(self.face_pairs)
            for patch in self.boundaries.patches:
                owners = np.asarray(patch.domain.owner_cells, dtype=np.int32)
                local_facets = np.asarray(
                    patch.domain.owner_local_entities, dtype=np.int32
                )
                for owner_cell, local_facet in zip(owners, local_facets, strict=True):
                    axis, side = tensor_local_face(
                        self.discretization.mesh.blocks[0].cell_kind,
                        int(local_facet),
                    )
                    determinant = jnp.take(
                        self.metrics.determinant[int(owner_cell)],
                        0 if side == 0 else -1,
                        axis=axis,
                    ).reshape((-1,))
                    boundary_rate = (
                        row_bound
                        * faces.surface_jacobian[face_index]
                        * faces.signal_speed[face_index]
                        / determinant
                    )
                    maximum = jnp.maximum(maximum, jnp.max(boundary_rate))
                    face_index += 1
        cfl_ = float(cfl)
        if not np.isfinite(cfl_) or cfl_ <= 0.0:
            raise ValueError("DGSEM CFL number must be positive and finite.")
        step = jnp.asarray(cfl_, dtype=maximum.dtype) / jnp.where(
            maximum > 0.0, maximum, jnp.inf
        )
        diffusive_rate = jnp.zeros((), dtype=maximum.dtype)
        if self.viscous_operator is not None:
            viscous = self.viscous_operator.stability_evidence(value, args, cfl=cfl_)
            diffusive_rate = viscous.maximum_diffusive_rate
            step = jnp.minimum(step, viscous.step)
        return DGSEMStableStepEvidence(
            step=step,
            maximum_nodal_rate=maximum,
            maximum_diffusive_rate=diffusive_rate,
            cfl=cfl_,
            positive=jnp.isfinite(step) & (step > 0.0),
            method_id=self.method.method_id,
        )

    def stable_step(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        return self.stable_step_evidence(state, args, cfl=cfl).step

    def linearize(self, time: Array, state: ArrayLike, args: Any = None, /):
        value = self._state(state)
        residual, pushforward = jax.linearize(
            lambda candidate: self(time, candidate, args), value
        )
        _, pullback = jax.vjp(lambda candidate: self(time, candidate, args), value)
        return residual, pushforward, pullback

    def linearize_weak_residual(self, time: Array, state: ArrayLike, args: Any = None, /):
        value = self._state(state)
        residual, pushforward = jax.linearize(
            lambda candidate: self.weak_residual(time, candidate, args), value
        )
        _, pullback = jax.vjp(
            lambda candidate: self.weak_residual(time, candidate, args), value
        )
        return residual, pushforward, pullback


__all__ = [
    "DGSEMConservationDiagnostics",
    "DGSEMMortarCompatibilityCertificate",
    "DGSEMMortarFluxLedger",
    "DGSEMNonconformingMortarPlan",
    "DGSEMConservationMethodPlan",
    "DGSEMFaceFluxes",
    "DGSEMSampledFluxCompatibilityEvidence",
    "DGSEMPreparationReport",
    "DGSEMStableStepEvidence",
    "PreparedDGSEMConservationDynamics",
    "certify_dgsem_mortar_compatibility",
    "sample_dgsem_flux_compatibility",
    "dgsem_mortar_flux_ledger",
]

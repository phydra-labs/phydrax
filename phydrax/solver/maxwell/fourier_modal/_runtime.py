#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._boundary_cascade import (
    BoundaryCascadePolicy,
    BoundaryRelation,
    compose_boundary_relations,
    identity_boundary_relation,
    prepare_layer_boundary,
)
from ._continuous import (
    prepare_continuous_fourier_modal_layer,
    PreparedContinuousFourierModalLayer,
)
from ._contracts import (
    ContinuousFourierModalLayer,
    FourierModalLayer,
    FourierModalMaxwellProblem,
    FourierModalSourcePlane,
    FrequencyMaxwellMaterial,
    PeriodicMaxwellPort,
)
from ._factorization import (
    _dense_solve,
    _tensor_samples,
    AnalyticInterfaceFramePlan,
    prepare_fourier_material,
    PreparedFourierMaterial,
    translate_prepared_fourier_material,
    VectorFourierFactorizationPlan,
)
from ._layer import prepare_layer_operator, PreparedLayerOperator
from ._scattering import (
    boundary_to_scattering,
    MaxwellPortScatteringOperator,
    prepare_fourier_modal_port_modes,
    PreparedFourierModalPortModes,
    shift_scattering_reference_planes,
)
from ._sources import (
    AffineBoundaryRelation,
    compose_affine_boundary_relations,
    emitted_port_amplitudes,
    FourierModalExcitation,
    homogeneous_affine_relation,
    source_plane_affine_relation,
)


LayerRefreshKind: TypeAlias = Literal[
    "unchanged",
    "thickness",
    "translation",
    "material",
]


class FourierModalSolveStatus(IntEnum):
    SUCCESS = 0
    PROPAGATION_TOLERANCE_NOT_MET = 1
    NONFINITE_RESULT = 2


class FourierModalResourcePolicy(StrictModule, NonTrainableState):
    max_harmonics: int = eqx.field(static=True)
    max_layers: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_harmonics: int = 4096,
        max_layers: int = 256,
        preparation_bytes: int = 8 * 2**30,
        workspace_bytes: int = 4 * 2**30,
    ):
        values = tuple(
            int(value)
            for value in (
                max_harmonics,
                max_layers,
                preparation_bytes,
                workspace_bytes,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Fourier-modal resource limits must be positive.")
        (
            self.max_harmonics,
            self.max_layers,
            self.preparation_bytes,
            self.workspace_bytes,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-resources",
                "max_harmonics": self.max_harmonics,
                "max_layers": self.max_layers,
                "preparation_bytes": self.preparation_bytes,
                "workspace_bytes": self.workspace_bytes,
            }
        )


class FourierModalSolvePolicy(StrictModule, NonTrainableState):
    boundary: BoundaryCascadePolicy
    resources: FourierModalResourcePolicy
    retain_boundary_fields: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        boundary: BoundaryCascadePolicy | None = None,
        resources: FourierModalResourcePolicy | None = None,
        retain_boundary_fields: bool = True,
    ):
        boundary_ = BoundaryCascadePolicy() if boundary is None else boundary
        resources_ = FourierModalResourcePolicy() if resources is None else resources
        self.boundary = boundary_
        self.resources = resources_
        self.retain_boundary_fields = bool(retain_boundary_fields)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-solve-policy",
                "boundary": boundary_.policy_id,
                "resources": resources_.policy_id,
                "retain_boundary_fields": self.retain_boundary_fields,
            }
        )


class FourierModalCostEstimate(StrictModule, NonTrainableState):
    harmonic_count: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    dense_solve_size: int = eqx.field(static=True)


class FourierModalCapabilities(StrictModule, NonTrainableState):
    full_tensor_layers: bool = eqx.field(static=True)
    bianisotropic_layers: bool = eqx.field(static=True)
    patterned_ports: bool = eqx.field(static=True)
    continuous_z_layers: bool = eqx.field(static=True)
    lateral_transformation_optics_pml: bool = eqx.field(static=True)
    finite_aperture_far_fields: bool = eqx.field(static=True)
    harmonic_epochs: bool = eqx.field(static=True)
    boundary_differentiation: bool = eqx.field(static=True)
    modal_subspace_differentiation: bool = eqx.field(static=True)
    internal_sources: bool = eqx.field(static=True)
    brillouin_zone: bool = eqx.field(static=True)

    def __init__(self):
        self.full_tensor_layers = True
        self.bianisotropic_layers = True
        self.patterned_ports = True
        self.continuous_z_layers = True
        self.lateral_transformation_optics_pml = True
        self.finite_aperture_far_fields = True
        self.harmonic_epochs = True
        self.boundary_differentiation = True
        self.modal_subspace_differentiation = True
        self.internal_sources = True
        self.brillouin_zone = True


class FourierModalSolvePlan(StrictModule, NonTrainableState):
    policy: FourierModalSolvePolicy
    cost: FourierModalCostEstimate
    capabilities: FourierModalCapabilities
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedFourierModalLayer(StrictModule):
    layer: FourierModalLayer
    base_material: PreparedFourierMaterial
    material: PreparedFourierMaterial
    operator: PreparedLayerOperator
    boundary: BoundaryRelation


PreparedElement: TypeAlias = (
    PreparedFourierModalLayer
    | PreparedContinuousFourierModalLayer
    | FourierModalSourcePlane
)


class PreparedFourierModalMaxwell(StrictModule):
    problem: FourierModalMaxwellProblem
    plan: FourierModalSolvePlan
    elements: tuple[PreparedElement, ...]
    source_host_indices: tuple[int, ...] = eqx.field(static=True)
    global_boundary: BoundaryRelation
    left_modes: PreparedFourierModalPortModes
    right_modes: PreparedFourierModalPortModes
    interface_scattering: MaxwellPortScatteringOperator
    scattering: MaxwellPortScatteringOperator
    total_thickness: Array
    refresh_count: int = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


class FourierModalRefreshSpec(StrictModule, NonTrainableState):
    layer_updates: tuple[LayerRefreshKind, ...] = eqx.field(static=True)
    angular_frequency_changed: bool = eqx.field(static=True)
    bloch_wavevector_changed: bool = eqx.field(static=True)
    ports_changed: bool = eqx.field(static=True)

    def __init__(
        self,
        layer_updates: tuple[LayerRefreshKind, ...],
        /,
        *,
        angular_frequency_changed: bool = False,
        bloch_wavevector_changed: bool = False,
        ports_changed: bool = False,
    ):
        updates = tuple(layer_updates)
        if any(
            value not in ("unchanged", "thickness", "translation", "material")
            for value in updates
        ):
            raise ValueError("Unknown Fourier-modal layer refresh kind.")
        self.layer_updates = updates
        self.angular_frequency_changed = bool(angular_frequency_changed)
        self.bloch_wavevector_changed = bool(bloch_wavevector_changed)
        self.ports_changed = bool(ports_changed)


class FourierModalDiagnostics(StrictModule):
    maximum_constitutive_residual: Array
    maximum_boundary_solve_residual: Array
    maximum_boundary_paired_error: Array
    scattering_conversion_residual: Array
    finite: Array
    propagation_converged: Array
    refresh_count: Array


class FourierModalProvenance(StrictModule, NonTrainableState):
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    harmonic_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)


class FourierModalSolveResult(StrictModule):
    scattering: MaxwellPortScatteringOperator
    right_outgoing: Array
    left_outgoing: Array
    left_incoming_power: Array
    right_incoming_power: Array
    left_outgoing_power: Array
    right_outgoing_power: Array
    net_port_power_into_stack: Array
    weighted_left_incoming_power: Array
    weighted_right_incoming_power: Array
    weighted_left_outgoing_power: Array
    weighted_right_outgoing_power: Array
    weighted_net_port_power_into_stack: Array
    internal_source_excitation: Array
    boundary_electric_fields: tuple[Array, ...]
    boundary_magnetic_fields: tuple[Array, ...]
    status: Array
    diagnostics: FourierModalDiagnostics
    provenance: FourierModalProvenance


class FourierModalConvergenceReport(StrictModule):
    harmonic_counts: Array
    scattering_differences: Array
    port_power_differences: Array
    converged: Array


def _cost_estimate(problem: FourierModalMaxwellProblem) -> FourierModalCostEstimate:
    count = problem.harmonics.harmonic_count
    layers = problem.layer_count
    itemsize = np.dtype(problem.harmonics.plan.precision.coefficient_dtype).itemsize
    sample_points = int(np.prod(problem.harmonics.plan.sample_shape))
    # Per layer: base and translated materials (36), the complete layer operator
    # including retained constitutive references (42), and its boundary relation
    # (16). Global storage covers the composed boundary (16), both port bases
    # (16), and the interface and reference-shifted scattering operators (32).
    matrix_elements = (94 * layers + 64) * count**2
    tangent_field_elements = 4 * layers * sample_points
    port_vector_elements = 18 * count
    preparation = (
        matrix_elements + tangent_field_elements + port_vector_elements
    ) * itemsize
    # The largest conversion forms two 4N-by-4N systems while retaining solve
    # factors and products; three copies of each system are budgeted.
    workspace = 6 * (4 * count) ** 2 * itemsize
    return FourierModalCostEstimate(count, layers, preparation, workspace, 4 * count)


def plan_fourier_modal_maxwell(
    problem: FourierModalMaxwellProblem,
    policy: FourierModalSolvePolicy | None = None,
    /,
) -> FourierModalSolvePlan:
    if not isinstance(problem, FourierModalMaxwellProblem):
        raise TypeError("problem must be a FourierModalMaxwellProblem.")
    policy_ = FourierModalSolvePolicy() if policy is None else policy
    cost = _cost_estimate(problem)
    resources = policy_.resources
    if cost.harmonic_count > resources.max_harmonics:
        raise ValueError("The problem exceeds max_harmonics.")
    if cost.layer_count > resources.max_layers:
        raise ValueError("The problem exceeds max_layers.")
    if cost.preparation_bytes > resources.preparation_bytes:
        raise ValueError("The problem exceeds the preparation byte budget.")
    if cost.workspace_bytes > resources.workspace_bytes:
        raise ValueError("The problem exceeds the workspace byte budget.")
    plan_id = canonical_fingerprint(
        {
            "kind": "fourier-modal-solve-plan",
            "problem": problem.problem_id,
            "policy": policy_.policy_id,
        }
    )
    return FourierModalSolvePlan(
        policy_,
        cost,
        FourierModalCapabilities(),
        problem_id=problem.problem_id,
        plan_id=plan_id,
    )


def _tree_has_tracer(value: object, /) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _canonical_material_samples(
    material: FrequencyMaxwellMaterial,
    problem: FourierModalMaxwellProblem,
    /,
) -> tuple[Array, Array, Array, Array]:
    return tuple(
        _tensor_samples(value, problem.harmonics)[0]
        for value in (
            material.permittivity,
            material.permeability,
            material.magnetoelectric_xi,
            material.magnetoelectric_zeta,
        )
    )


def _concrete_arrays_equal(left: tuple[Array, ...], right: tuple[Array, ...], /) -> bool:
    if _tree_has_tracer((left, right)):
        return False
    return all(
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.array_equal(np.asarray(first), np.asarray(second), equal_nan=False)
        for first, second in zip(left, right, strict=True)
    )


def _checked_material_slot(
    material: FrequencyMaxwellMaterial,
    problem: FourierModalMaxwellProblem,
    records: dict[
        str, tuple[FrequencyMaxwellMaterial, tuple[Array, Array, Array, Array]]
    ],
    /,
) -> FrequencyMaxwellMaterial:
    samples = _canonical_material_samples(material, problem)
    previous = records.get(material.material_id)
    if previous is None:
        records[material.material_id] = (material, samples)
        return material
    previous_material, previous_samples = previous
    if (
        material.material_role != previous_material.material_role
        or material.origin_evidence_id != previous_material.origin_evidence_id
    ):
        raise ValueError(
            "Occurrences of one material_id must share role and origin evidence."
        )
    if _tree_has_tracer((samples, previous_samples)):
        mismatch = jnp.asarray(False)
        for current, reference in zip(samples, previous_samples, strict=True):
            if current.shape != reference.shape or current.dtype != reference.dtype:
                raise ValueError(
                    "Occurrences of one material_id have incompatible canonical samples."
                )
            mismatch = mismatch | ~jnp.all(current == reference)
        checked = eqx.error_if(
            material.permittivity,
            mismatch,
            "Occurrences of one material_id must have equal canonical samples.",
        )
        material = eqx.tree_at(lambda value: value.permittivity, material, checked)
        return material
    if not _concrete_arrays_equal(samples, previous_samples):
        raise ValueError(
            "Occurrences of one material_id must have equal canonical samples."
        )
    return material


def _checked_factorization_frame(
    factorization,
    records: dict[str, Array],
    /,
):
    if not isinstance(factorization, VectorFourierFactorizationPlan) or not isinstance(
        factorization.frame, AnalyticInterfaceFramePlan
    ):
        return factorization
    frame = factorization.frame
    previous = records.get(frame.frame_id)
    if previous is None:
        records[frame.frame_id] = frame.tangent_field
        return factorization
    if (
        frame.tangent_field.shape != previous.shape
        or frame.tangent_field.dtype != previous.dtype
    ):
        raise ValueError("Occurrences of one frame_id have incompatible tangent fields.")
    if _tree_has_tracer((frame.tangent_field, previous)):
        checked = eqx.error_if(
            frame.tangent_field,
            ~jnp.all(frame.tangent_field == previous),
            "Occurrences of one frame_id must have equal tangent-field values.",
        )
        return eqx.tree_at(
            lambda value: value.frame.tangent_field,
            factorization,
            checked,
        )
    if not np.array_equal(
        np.asarray(frame.tangent_field), np.asarray(previous), equal_nan=False
    ):
        raise ValueError(
            "Occurrences of one frame_id must have equal tangent-field values."
        )
    return factorization


def _checked_problem_slots(
    problem: FourierModalMaxwellProblem, /
) -> FourierModalMaxwellProblem:
    material_records: dict[
        str, tuple[FrequencyMaxwellMaterial, tuple[Array, Array, Array, Array]]
    ] = {}
    frame_records: dict[str, Array] = {}

    def checked_port(port):
        material = _checked_material_slot(port.material, problem, material_records)
        port = eqx.tree_at(lambda value: value.material, port, material)
        if isinstance(port, PeriodicMaxwellPort):
            factorization = _checked_factorization_frame(
                port.factorization, frame_records
            )
            port = eqx.tree_at(lambda value: value.factorization, port, factorization)
        return port

    superstrate = checked_port(problem.superstrate)
    elements = []
    for element in problem.elements:
        if isinstance(element, FourierModalLayer):
            material = _checked_material_slot(element.material, problem, material_records)
            factorization = _checked_factorization_frame(
                element.factorization, frame_records
            )
            element = eqx.tree_at(
                lambda value: (value.material, value.factorization),
                element,
                (material, factorization),
            )
        elif isinstance(element, ContinuousFourierModalLayer):
            factorization = _checked_factorization_frame(
                element.factorization, frame_records
            )
            element = eqx.tree_at(
                lambda value: value.factorization, element, factorization
            )
        elements.append(element)
    substrate = checked_port(problem.substrate)
    return eqx.tree_at(
        lambda value: (value.superstrate, value.elements, value.substrate),
        problem,
        (superstrate, tuple(elements), substrate),
    )


def _source_host_indices(problem: FourierModalMaxwellProblem) -> tuple[int, ...]:
    hosts: list[int] = []
    for index, element in enumerate(problem.elements):
        if not isinstance(element, FourierModalSourcePlane):
            continue
        left = index - 1
        right = index + 1
        if left < 0 or right >= len(problem.elements):
            raise ValueError("A source plane must lie between two finite layers.")
        left_layer = problem.elements[left]
        right_layer = problem.elements[right]
        if not isinstance(left_layer, FourierModalLayer) or not isinstance(
            right_layer, FourierModalLayer
        ):
            raise TypeError("A source plane must lie directly between two layers.")
        if left_layer.material.material_id != right_layer.material.material_id:
            raise ValueError(
                "A source plane must split one host material into adjacent layers."
            )
        hosts.append(left)
    return tuple(hosts)


def _prepare_layer(
    problem: FourierModalMaxwellProblem,
    layer: FourierModalLayer,
    policy: FourierModalSolvePolicy,
    /,
    *,
    base_material: PreparedFourierMaterial | None = None,
) -> PreparedFourierModalLayer:
    thickness = eqx.error_if(
        layer.thickness,
        (~jnp.isfinite(layer.thickness)) | (jnp.real(layer.thickness) < 0.0),
        "Layer thickness must be finite and non-negative.",
    )
    base = (
        prepare_fourier_material(
            layer.material,
            problem.harmonics,
            layer.factorization,
        )
        if base_material is None
        else base_material
    )
    material = translate_prepared_fourier_material(
        base,
        problem.harmonics,
        layer.translation,
    )
    operator = prepare_layer_operator(
        material,
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
    )
    boundary = prepare_layer_boundary(operator, thickness, policy.boundary)
    return PreparedFourierModalLayer(layer, base, material, operator, boundary)


def _compose_elements(
    elements: tuple[PreparedElement, ...],
    tangential_size: int,
    dtype: jnp.dtype,
    /,
) -> BoundaryRelation:
    relation = identity_boundary_relation(tangential_size, dtype)
    for element in elements:
        component = (
            element.boundary
            if isinstance(
                element,
                PreparedFourierModalLayer | PreparedContinuousFourierModalLayer,
            )
            else identity_boundary_relation(tangential_size, dtype)
        )
        relation = compose_boundary_relations(relation, component)
    return relation


def _finalize_prepared(
    problem: FourierModalMaxwellProblem,
    plan: FourierModalSolvePlan,
    elements: tuple[PreparedElement, ...],
    source_hosts: tuple[int, ...],
    refresh_count: int,
    /,
) -> PreparedFourierModalMaxwell:
    count = problem.harmonics.harmonic_count
    dtype = jnp.dtype(problem.harmonics.plan.precision.coefficient_dtype)
    global_boundary = _compose_elements(elements, 2 * count, dtype)
    left_modes = prepare_fourier_modal_port_modes(
        problem.superstrate,
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
        outward_sign=-1,
    )
    right_modes = prepare_fourier_modal_port_modes(
        problem.substrate,
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
        outward_sign=1,
    )
    interface_scattering = boundary_to_scattering(
        global_boundary, left_modes, right_modes
    )
    total_thickness = sum(
        (
            element.layer.thickness
            for element in elements
            if isinstance(
                element,
                PreparedFourierModalLayer | PreparedContinuousFourierModalLayer,
            )
        ),
        start=jnp.asarray(0.0, dtype=dtype),
    )
    scattering = shift_scattering_reference_planes(
        interface_scattering,
        problem.superstrate.reference_plane,
        problem.substrate.reference_plane,
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-fourier-modal-maxwell",
            "plan": plan.plan_id,
            "numeric_version": problem.numeric_version,
            "refresh_count": int(refresh_count),
        }
    )
    return PreparedFourierModalMaxwell(
        problem,
        plan,
        elements,
        source_hosts,
        global_boundary,
        left_modes,
        right_modes,
        interface_scattering,
        scattering,
        total_thickness,
        int(refresh_count),
        preparation_id=preparation_id,
    )


def prepare_fourier_modal_maxwell(
    problem: FourierModalMaxwellProblem,
    plan_or_policy: FourierModalSolvePlan | FourierModalSolvePolicy | None = None,
    /,
) -> PreparedFourierModalMaxwell:
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, FourierModalSolvePlan)
        else plan_fourier_modal_maxwell(problem, plan_or_policy)
    )
    if plan.problem_id != problem.problem_id:
        raise ValueError("The solve plan belongs to a different problem identity.")
    problem = _checked_problem_slots(problem)
    prepared_elements: list[PreparedElement] = []
    material_cache: dict[tuple[str, str, str, str], PreparedFourierMaterial] = {}
    for element in problem.elements:
        if isinstance(element, FourierModalSourcePlane):
            prepared_elements.append(element)
            continue
        if isinstance(element, ContinuousFourierModalLayer):
            prepared_elements.append(
                prepare_continuous_fourier_modal_layer(
                    problem, element, plan.policy.boundary
                )
            )
            continue
        key = (
            element.material.material_id,
            element.material.material_role,
            element.material.origin_evidence_id,
            element.factorization.plan_id,
        )
        reuse_is_proven = not _tree_has_tracer(
            (
                element.material,
                element.factorization,
                problem.harmonics.primitive_vectors,
            )
        )
        base_material = material_cache.get(key) if reuse_is_proven else None
        prepared_layer = _prepare_layer(
            problem,
            element,
            plan.policy,
            base_material=base_material,
        )
        if reuse_is_proven:
            material_cache[key] = prepared_layer.base_material
        prepared_elements.append(prepared_layer)
    return _finalize_prepared(
        problem,
        plan,
        tuple(prepared_elements),
        _source_host_indices(problem),
        0,
    )


def _values_proven_equal(left: object, right: object, /) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves) or _tree_has_tracer(
        (left_leaves, right_leaves)
    ):
        return False
    return all(
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.array_equal(np.asarray(first), np.asarray(second), equal_nan=False)
        for first, second in zip(left_leaves, right_leaves, strict=True)
    )


def _port_identity(port, /) -> tuple[object, ...]:
    return (
        type(port),
        port.port_id,
        port.material.material_id,
        port.material.material_role,
        port.material.origin_evidence_id,
        port.factorization.plan_id if isinstance(port, PeriodicMaxwellPort) else None,
    )


def _stack_identity(problem: FourierModalMaxwellProblem, /) -> tuple[object, ...]:
    return (
        problem.harmonics.plan.plan_id,
        problem.harmonics.plan.layout.layout_id,
        _port_identity(problem.superstrate),
        tuple(
            (
                type(element),
                (
                    element.source_id
                    if isinstance(element, FourierModalSourcePlane)
                    else element.layer_id
                ),
                (
                    None
                    if isinstance(element, FourierModalSourcePlane)
                    else element.factorization.plan_id
                ),
                (
                    None
                    if isinstance(
                        element,
                        FourierModalSourcePlane | ContinuousFourierModalLayer,
                    )
                    else (
                        element.material.material_id,
                        element.material.material_role,
                        element.material.origin_evidence_id,
                    )
                ),
            )
            for element in problem.elements
        ),
        _port_identity(problem.substrate),
    )


def _refreshed_layer(
    problem: FourierModalMaxwellProblem,
    element: FourierModalLayer,
    old: PreparedFourierModalLayer,
    policy: FourierModalSolvePolicy,
    /,
    *,
    lattice_same: bool,
    frequency_same: bool,
    bloch_same: bool,
) -> PreparedFourierModalLayer:
    old_problem = old.layer
    if (
        element.material.material_id != old_problem.material.material_id
        or element.material.material_role != old_problem.material.material_role
        or element.material.origin_evidence_id != old_problem.material.origin_evidence_id
        or element.factorization.plan_id != old_problem.factorization.plan_id
    ):
        raise ValueError(
            "Refresh cannot change material slots, origins, or factorization."
        )
    material_same = _concrete_arrays_equal(
        _canonical_material_samples(element.material, problem),
        _canonical_material_samples(old_problem.material, problem),
    )
    frame_same = _values_proven_equal(element.factorization, old_problem.factorization)
    translation_same = _values_proven_equal(element.translation, old_problem.translation)
    thickness_same = _values_proven_equal(element.thickness, old_problem.thickness)
    if not lattice_same or not material_same or not frame_same:
        return _prepare_layer(problem, element, policy)
    if not translation_same:
        material = translate_prepared_fourier_material(
            old.base_material,
            problem.harmonics,
            element.translation,
        )
        operator = prepare_layer_operator(
            material,
            problem.harmonics,
            problem.angular_frequency,
            problem.bloch_wavevector,
        )
        boundary = prepare_layer_boundary(operator, element.thickness, policy.boundary)
        return PreparedFourierModalLayer(
            element, old.base_material, material, operator, boundary
        )
    if not frequency_same or not bloch_same:
        operator = prepare_layer_operator(
            old.material,
            problem.harmonics,
            problem.angular_frequency,
            problem.bloch_wavevector,
        )
        boundary = prepare_layer_boundary(operator, element.thickness, policy.boundary)
        return PreparedFourierModalLayer(
            element, old.base_material, old.material, operator, boundary
        )
    if not thickness_same:
        boundary = prepare_layer_boundary(
            old.operator, element.thickness, policy.boundary
        )
        return PreparedFourierModalLayer(
            element, old.base_material, old.material, old.operator, boundary
        )
    return PreparedFourierModalLayer(
        element, old.base_material, old.material, old.operator, old.boundary
    )


def refresh_fourier_modal_maxwell(
    prepared: PreparedFourierModalMaxwell,
    problem: FourierModalMaxwellProblem,
    spec: FourierModalRefreshSpec | None = None,
    /,
) -> PreparedFourierModalMaxwell:
    if _stack_identity(problem) != _stack_identity(prepared.problem):
        raise ValueError("Refresh cannot change harmonic layout or stack identity.")
    layer_count = problem.layer_count
    if spec is not None and len(spec.layer_updates) != layer_count:
        raise ValueError("layer_updates must contain one entry per finite layer.")
    problem = _checked_problem_slots(problem)
    lattice_same = _values_proven_equal(
        problem.harmonics.primitive_vectors,
        prepared.problem.harmonics.primitive_vectors,
    )
    frequency_same = _values_proven_equal(
        problem.angular_frequency, prepared.problem.angular_frequency
    )
    bloch_same = _values_proven_equal(
        problem.bloch_wavevector, prepared.problem.bloch_wavevector
    )
    if any(
        isinstance(element, ContinuousFourierModalLayer) for element in problem.elements
    ):
        refreshed = prepare_fourier_modal_maxwell(problem, prepared.plan.policy)
        refresh_count = prepared.refresh_count + 1
        preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-fourier-modal-maxwell",
                "plan": refreshed.plan.plan_id,
                "numeric_version": problem.numeric_version,
                "refresh_count": refresh_count,
            }
        )
        return eqx.tree_at(
            lambda value: (value.refresh_count, value.preparation_id),
            refreshed,
            (refresh_count, preparation_id),
        )
    old_layers = tuple(
        value
        for value in prepared.elements
        if isinstance(value, PreparedFourierModalLayer)
    )
    new_elements: list[PreparedElement] = []
    layer_index = 0
    for element in problem.elements:
        if isinstance(element, FourierModalSourcePlane):
            new_elements.append(element)
            continue
        old = old_layers[layer_index]
        new_elements.append(
            _refreshed_layer(
                problem,
                element,
                old,
                prepared.plan.policy,
                lattice_same=lattice_same,
                frequency_same=frequency_same,
                bloch_same=bloch_same,
            )
        )
        layer_index += 1
    new_plan = plan_fourier_modal_maxwell(problem, prepared.plan.policy)
    return _finalize_prepared(
        problem,
        new_plan,
        tuple(new_elements),
        _source_host_indices(problem),
        prepared.refresh_count + 1,
    )


def _source_lookup(
    excitation: FourierModalExcitation,
    source_id: str,
    count: int,
    rhs_count: int,
    dtype: jnp.dtype,
    /,
) -> tuple[Array, Array]:
    if source_id not in excitation.source_ids:
        zero = jnp.zeros((3, count, rhs_count), dtype=dtype)
        return zero, zero
    index = excitation.source_ids.index(source_id)
    return excitation.electric_currents[index], excitation.magnetic_currents[index]


def _affine_stack(
    prepared: PreparedFourierModalMaxwell,
    excitation: FourierModalExcitation,
    /,
) -> tuple[AffineBoundaryRelation, tuple[AffineBoundaryRelation, ...]]:
    count = prepared.problem.harmonics.harmonic_count
    dtype = prepared.interface_scattering.s11.matrix.dtype
    affine_elements: list[AffineBoundaryRelation] = []
    source_index = 0
    for index, element in enumerate(prepared.elements):
        if isinstance(
            element,
            PreparedFourierModalLayer | PreparedContinuousFourierModalLayer,
        ):
            affine_elements.append(
                homogeneous_affine_relation(element.boundary, excitation.rhs_count)
            )
            continue
        electric, magnetic = _source_lookup(
            excitation,
            element.source_id,
            count,
            excitation.rhs_count,
            dtype,
        )
        host_element_index = prepared.source_host_indices[source_index]
        host = prepared.elements[host_element_index]
        if not isinstance(host, PreparedFourierModalLayer):
            raise TypeError("Prepared source host is not a finite layer.")
        affine_elements.append(
            source_plane_affine_relation(host.operator, electric, magnetic)
        )
        source_index += 1
    total = homogeneous_affine_relation(
        identity_boundary_relation(2 * count, dtype),
        excitation.rhs_count,
    )
    for element in affine_elements:
        total = compose_affine_boundary_relations(total, element)
    return total, tuple(affine_elements)


def _interface_boundary_fields(
    prepared: PreparedFourierModalMaxwell,
    excitation: FourierModalExcitation,
    left_outgoing: Array,
    right_outgoing: Array,
    affine_elements: tuple[AffineBoundaryRelation, ...],
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    left_phase = jnp.exp(
        1j
        * jnp.repeat(prepared.left_modes.longitudinal_wavevector, 2)[:, None]
        * prepared.problem.superstrate.reference_plane
    )
    right_phase = jnp.exp(
        1j
        * jnp.repeat(prepared.right_modes.longitudinal_wavevector, 2)[:, None]
        * prepared.problem.substrate.reference_plane
    )
    forward_left = left_phase * excitation.left_incident
    backward_left = left_outgoing / left_phase
    electric = prepared.left_modes.electric_matrix @ (forward_left + backward_left)
    magnetic = prepared.left_modes.magnetic_matrix @ (forward_left - backward_left)
    electric_fields = [electric]
    magnetic_fields = [magnetic]
    for affine in affine_elements:
        relation = affine.relation
        magnetic_right = _dense_solve(
            relation.d,
            magnetic - relation.c @ electric - affine.magnetic_source,
        )
        electric_right = (
            relation.a @ electric + relation.b @ magnetic_right + affine.electric_source
        )
        electric = electric_right
        magnetic = magnetic_right
        electric_fields.append(electric)
        magnetic_fields.append(magnetic)
    expected_forward = right_outgoing / right_phase
    expected_backward = right_phase * excitation.right_incident
    expected_electric = prepared.right_modes.electric_matrix @ (
        expected_forward + expected_backward
    )
    expected_magnetic = prepared.right_modes.magnetic_matrix @ (
        expected_forward - expected_backward
    )
    electric_fields[-1] = 0.5 * (electric_fields[-1] + expected_electric)
    magnetic_fields[-1] = 0.5 * (magnetic_fields[-1] + expected_magnetic)
    return tuple(electric_fields), tuple(magnetic_fields)


def solve_fourier_modal_maxwell(
    prepared: PreparedFourierModalMaxwell,
    excitation: FourierModalExcitation,
    /,
) -> FourierModalSolveResult:
    if excitation.left_incident.shape[0] != prepared.interface_scattering.block_size:
        raise ValueError("Excitation port size does not match the prepared stack.")
    if any(
        source_id not in prepared.problem.source_ids
        for source_id in excitation.source_ids
    ):
        raise KeyError("Excitation references an unknown source plane.")
    interface = prepared.interface_scattering
    external_right = (
        interface.s11.matrix @ excitation.left_incident
        + interface.s12.matrix @ excitation.right_incident
    )
    external_left = (
        interface.s21.matrix @ excitation.left_incident
        + interface.s22.matrix @ excitation.right_incident
    )
    affine, affine_elements = _affine_stack(prepared, excitation)
    emitted_right, emitted_left = emitted_port_amplitudes(
        affine,
        prepared.left_modes,
        prepared.right_modes,
    )
    interface_right = external_right + emitted_right
    interface_left = external_left + emitted_left
    left_phase = jnp.exp(
        1j
        * jnp.repeat(prepared.left_modes.longitudinal_wavevector, 2)[:, None]
        * prepared.problem.superstrate.reference_plane
    )
    right_phase = jnp.exp(
        1j
        * jnp.repeat(prepared.right_modes.longitudinal_wavevector, 2)[:, None]
        * prepared.problem.substrate.reference_plane
    )
    right_outgoing = right_phase * interface_right
    left_outgoing = left_phase * interface_left
    left_weights = jnp.abs(prepared.left_modes.flux_weights)[:, None]
    right_weights = jnp.abs(prepared.right_modes.flux_weights)[:, None]
    left_incoming_power = jnp.sum(
        left_weights * jnp.abs(excitation.left_incident) ** 2, axis=0
    )
    right_incoming_power = jnp.sum(
        right_weights * jnp.abs(excitation.right_incident) ** 2, axis=0
    )
    left_outgoing_power = jnp.sum(left_weights * jnp.abs(left_outgoing) ** 2, axis=0)
    right_outgoing_power = jnp.sum(right_weights * jnp.abs(right_outgoing) ** 2, axis=0)
    net_port_power = (
        left_incoming_power
        + right_incoming_power
        - left_outgoing_power
        - right_outgoing_power
    )
    weights = excitation.channel_weights
    weighted_left_incoming = jnp.sum(weights * left_incoming_power)
    weighted_right_incoming = jnp.sum(weights * right_incoming_power)
    weighted_left_outgoing = jnp.sum(weights * left_outgoing_power)
    weighted_right_outgoing = jnp.sum(weights * right_outgoing_power)
    weighted_net_port_power = jnp.sum(weights * net_port_power)
    internal_source_excitation = jnp.any(
        jnp.stack(
            tuple(
                jnp.any(jnp.abs(value) > 0.0)
                for value in excitation.electric_currents + excitation.magnetic_currents
            )
            or (jnp.asarray(False),)
        )
    )
    boundary_electric, boundary_magnetic = _interface_boundary_fields(
        prepared,
        excitation,
        left_outgoing,
        right_outgoing,
        affine_elements,
    )
    layers = tuple(
        element
        for element in prepared.elements
        if isinstance(
            element,
            PreparedFourierModalLayer | PreparedContinuousFourierModalLayer,
        )
    )
    maximum_constitutive = jnp.max(
        jnp.stack(
            tuple(layer.operator.diagnostics.constitutive_residual for layer in layers)
            or (jnp.asarray(0.0),)
        )
    )
    maximum_boundary_solve = jnp.max(
        jnp.stack(
            tuple(layer.boundary.diagnostics.solve_residual for layer in layers)
            or (jnp.asarray(0.0),)
        )
    )
    maximum_paired = jnp.max(
        jnp.stack(
            tuple(layer.boundary.diagnostics.paired_error for layer in layers)
            or (jnp.asarray(0.0),)
        )
    )
    propagation_converged = jnp.all(
        jnp.stack(
            tuple(layer.boundary.diagnostics.converged for layer in layers)
            or (jnp.asarray(True),)
        )
    )
    propagation_converged = propagation_converged & jnp.all(
        jnp.stack(
            tuple(
                element.successful
                for element in prepared.elements
                if isinstance(element, PreparedContinuousFourierModalLayer)
            )
            or (jnp.asarray(True),)
        )
    )
    finite = (
        jnp.all(jnp.isfinite(right_outgoing))
        & jnp.all(jnp.isfinite(left_outgoing))
        & jnp.all(jnp.isfinite(left_incoming_power))
        & jnp.all(jnp.isfinite(right_incoming_power))
        & jnp.all(jnp.isfinite(left_outgoing_power))
        & jnp.all(jnp.isfinite(right_outgoing_power))
        & jnp.all(jnp.isfinite(net_port_power))
    )
    status = jnp.where(
        ~finite,
        int(FourierModalSolveStatus.NONFINITE_RESULT),
        jnp.where(
            ~propagation_converged,
            int(FourierModalSolveStatus.PROPAGATION_TOLERANCE_NOT_MET),
            int(FourierModalSolveStatus.SUCCESS),
        ),
    )
    diagnostics = FourierModalDiagnostics(
        maximum_constitutive,
        maximum_boundary_solve,
        maximum_paired,
        interface.diagnostics.conversion_residual,
        finite,
        propagation_converged,
        jnp.asarray(prepared.refresh_count, dtype=jnp.int32),
    )
    provenance = FourierModalProvenance(
        problem_id=prepared.problem.problem_id,
        plan_id=prepared.plan.plan_id,
        preparation_id=prepared.preparation_id,
        harmonic_layout_id=prepared.problem.harmonics.plan.layout.layout_id,
        numeric_version=prepared.problem.numeric_version,
        backend="boundary-cascade",
    )
    return FourierModalSolveResult(
        prepared.scattering,
        right_outgoing,
        left_outgoing,
        left_incoming_power,
        right_incoming_power,
        left_outgoing_power,
        right_outgoing_power,
        net_port_power,
        weighted_left_incoming,
        weighted_right_incoming,
        weighted_left_outgoing,
        weighted_right_outgoing,
        weighted_net_port_power,
        internal_source_excitation,
        boundary_electric if prepared.plan.policy.retain_boundary_fields else (),
        boundary_magnetic if prepared.plan.policy.retain_boundary_fields else (),
        status,
        diagnostics,
        provenance,
    )


def fourier_modal_convergence_report(
    harmonic_counts: Array,
    scattering_matrices: tuple[Array, ...],
    port_power_values: tuple[Array, ...],
    /,
    *,
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-8,
) -> FourierModalConvergenceReport:
    counts = jnp.asarray(harmonic_counts, dtype=jnp.int32)
    if counts.ndim != 1 or counts.size != len(scattering_matrices):
        raise ValueError("harmonic_counts must match the supplied result sequences.")
    if len(scattering_matrices) < 2 or len(port_power_values) != len(scattering_matrices):
        raise ValueError(
            "At least two matching scattering and directional-power results are required."
        )
    scattering_differences = []
    port_power_differences = []
    for previous, current, previous_power, current_power in zip(
        scattering_matrices[:-1],
        scattering_matrices[1:],
        port_power_values[:-1],
        port_power_values[1:],
        strict=True,
    ):
        scattering_scale = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(current) ** 2)), 1.0)
        power_scale = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(current_power) ** 2)), 1.0)
        scattering_differences.append(
            jnp.sqrt(jnp.sum(jnp.abs(current - previous) ** 2)) / scattering_scale
        )
        port_power_differences.append(
            jnp.sqrt(jnp.sum(jnp.abs(current_power - previous_power) ** 2)) / power_scale
        )
    scattering_array = jnp.stack(scattering_differences)
    port_power_array = jnp.stack(port_power_differences)
    converged = (scattering_array[-1] <= relative_tolerance + absolute_tolerance) & (
        port_power_array[-1] <= relative_tolerance + absolute_tolerance
    )
    return FourierModalConvergenceReport(
        counts,
        scattering_array,
        port_power_array,
        converged,
    )


__all__ = [
    "FourierModalCapabilities",
    "FourierModalConvergenceReport",
    "FourierModalCostEstimate",
    "FourierModalDiagnostics",
    "FourierModalProvenance",
    "FourierModalRefreshSpec",
    "FourierModalResourcePolicy",
    "FourierModalSolvePlan",
    "FourierModalSolvePolicy",
    "FourierModalSolveResult",
    "FourierModalSolveStatus",
    "LayerRefreshKind",
    "PreparedFourierModalLayer",
    "PreparedFourierModalMaxwell",
    "fourier_modal_convergence_report",
    "plan_fourier_modal_maxwell",
    "prepare_fourier_modal_maxwell",
    "refresh_fourier_modal_maxwell",
    "solve_fourier_modal_maxwell",
]

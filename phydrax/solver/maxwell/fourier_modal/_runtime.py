#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
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
from ._contracts import (
    FourierModalLayer,
    FourierModalMaxwellProblem,
    FourierModalSourcePlane,
)
from ._factorization import (
    _dense_solve,
    prepare_fourier_material,
    PreparedFourierMaterial,
    translate_prepared_fourier_material,
)
from ._layer import prepare_layer_operator, PreparedLayerOperator
from ._scattering import (
    boundary_to_scattering,
    HomogeneousPortModes,
    MaxwellPortScatteringOperator,
    prepare_homogeneous_port_modes,
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
    POWER_BALANCE_NOT_MET = 3


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
    power_tolerance: float = eqx.field(static=True)
    retain_boundary_fields: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        boundary: BoundaryCascadePolicy | None = None,
        resources: FourierModalResourcePolicy | None = None,
        power_tolerance: float = 1e-7,
        retain_boundary_fields: bool = True,
    ):
        boundary_ = BoundaryCascadePolicy() if boundary is None else boundary
        resources_ = FourierModalResourcePolicy() if resources is None else resources
        tolerance = float(power_tolerance)
        if tolerance < 0.0:
            raise ValueError("power_tolerance must be non-negative.")
        self.boundary = boundary_
        self.resources = resources_
        self.power_tolerance = tolerance
        self.retain_boundary_fields = bool(retain_boundary_fields)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-solve-policy",
                "boundary": boundary_.policy_id,
                "resources": resources_.policy_id,
                "power_tolerance": tolerance,
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
    patterned_ports: bool = eqx.field(static=True)
    boundary_differentiation: bool = eqx.field(static=True)
    modal_differentiation: bool = eqx.field(static=True)
    internal_sources: bool = eqx.field(static=True)
    brillouin_zone: bool = eqx.field(static=True)

    def __init__(self):
        self.full_tensor_layers = True
        self.patterned_ports = False
        self.boundary_differentiation = True
        self.modal_differentiation = False
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


PreparedElement: TypeAlias = PreparedFourierModalLayer | FourierModalSourcePlane


class PreparedFourierModalMaxwell(StrictModule):
    problem: FourierModalMaxwellProblem
    plan: FourierModalSolvePlan
    elements: tuple[PreparedElement, ...]
    source_host_indices: tuple[int, ...] = eqx.field(static=True)
    global_boundary: BoundaryRelation
    left_modes: HomogeneousPortModes
    right_modes: HomogeneousPortModes
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
    power_balance_residual: Array
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
    incident_power: Array
    reflected_power: Array
    transmitted_power: Array
    absorbed_power: Array
    weighted_incident_power: Array
    weighted_reflected_power: Array
    weighted_transmitted_power: Array
    weighted_absorbed_power: Array
    boundary_electric_fields: tuple[Array, ...]
    boundary_magnetic_fields: tuple[Array, ...]
    status: Array
    diagnostics: FourierModalDiagnostics
    provenance: FourierModalProvenance


class FourierModalConvergenceReport(StrictModule):
    harmonic_counts: Array
    scattering_differences: Array
    power_differences: Array
    converged: Array


def _cost_estimate(problem: FourierModalMaxwellProblem) -> FourierModalCostEstimate:
    count = problem.harmonics.harmonic_count
    layers = problem.layer_count
    itemsize = np.dtype(problem.harmonics.plan.precision.coefficient_dtype).itemsize
    matrices_per_layer = 18 + 16 + 16
    preparation = layers * matrices_per_layer * count**2 * itemsize
    workspace = (8 * count) ** 2 * itemsize
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
            if isinstance(element, PreparedFourierModalLayer)
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
    left_modes = prepare_homogeneous_port_modes(
        problem.superstrate,
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
    )
    right_modes = prepare_homogeneous_port_modes(
        problem.substrate,
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
    )
    interface_scattering = boundary_to_scattering(
        global_boundary, left_modes, right_modes
    )
    total_thickness = sum(
        (
            element.layer.thickness
            for element in elements
            if isinstance(element, PreparedFourierModalLayer)
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
    prepared_elements: list[PreparedElement] = []
    material_cache: dict[tuple[str, str], PreparedFourierMaterial] = {}
    for element in problem.elements:
        if isinstance(element, FourierModalSourcePlane):
            prepared_elements.append(element)
            continue
        key = (element.material.material_id, element.factorization.plan_id)
        base_material = material_cache.get(key)
        prepared_layer = _prepare_layer(
            problem,
            element,
            plan.policy,
            base_material=base_material,
        )
        material_cache[key] = prepared_layer.base_material
        prepared_elements.append(prepared_layer)
    return _finalize_prepared(
        problem,
        plan,
        tuple(prepared_elements),
        _source_host_indices(problem),
        0,
    )


def refresh_fourier_modal_maxwell(
    prepared: PreparedFourierModalMaxwell,
    problem: FourierModalMaxwellProblem,
    spec: FourierModalRefreshSpec | None = None,
    /,
) -> PreparedFourierModalMaxwell:
    if len(problem.elements) != len(prepared.problem.elements):
        raise ValueError("Refresh cannot change stack topology.")
    if tuple(type(value) for value in problem.elements) != tuple(
        type(value) for value in prepared.problem.elements
    ):
        raise ValueError("Refresh cannot change stack element kinds.")
    layer_count = problem.layer_count
    spec_ = (
        FourierModalRefreshSpec(
            tuple("material" for _ in range(layer_count)),
            angular_frequency_changed=True,
            bloch_wavevector_changed=True,
            ports_changed=True,
        )
        if spec is None
        else spec
    )
    if len(spec_.layer_updates) != layer_count:
        raise ValueError("layer_updates must contain one entry per finite layer.")
    force_operator_refresh = (
        spec_.angular_frequency_changed or spec_.bloch_wavevector_changed
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
        update = spec_.layer_updates[layer_index]
        if update == "material":
            new_layer = _prepare_layer(problem, element, prepared.plan.policy)
        elif update == "translation":
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
            boundary = prepare_layer_boundary(
                operator,
                element.thickness,
                prepared.plan.policy.boundary,
            )
            new_layer = PreparedFourierModalLayer(
                element,
                old.base_material,
                material,
                operator,
                boundary,
            )
        elif force_operator_refresh:
            operator = prepare_layer_operator(
                old.material,
                problem.harmonics,
                problem.angular_frequency,
                problem.bloch_wavevector,
            )
            boundary = prepare_layer_boundary(
                operator,
                element.thickness,
                prepared.plan.policy.boundary,
            )
            new_layer = PreparedFourierModalLayer(
                element,
                old.base_material,
                old.material,
                operator,
                boundary,
            )
        elif update == "thickness":
            boundary = prepare_layer_boundary(
                old.operator,
                element.thickness,
                prepared.plan.policy.boundary,
            )
            new_layer = PreparedFourierModalLayer(
                element,
                old.base_material,
                old.material,
                old.operator,
                boundary,
            )
        else:
            new_layer = PreparedFourierModalLayer(
                element,
                old.base_material,
                old.material,
                old.operator,
                old.boundary,
            )
        new_elements.append(new_layer)
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
        if isinstance(element, PreparedFourierModalLayer):
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
    incident_power = jnp.sum(
        left_weights * jnp.abs(excitation.left_incident) ** 2
        + right_weights * jnp.abs(excitation.right_incident) ** 2,
        axis=0,
    )
    reflected_power = jnp.sum(left_weights * jnp.abs(left_outgoing) ** 2, axis=0)
    transmitted_power = jnp.sum(right_weights * jnp.abs(right_outgoing) ** 2, axis=0)
    absorbed_power = incident_power - reflected_power - transmitted_power
    weights = excitation.channel_weights
    weighted_incident = jnp.sum(weights * incident_power)
    weighted_reflected = jnp.sum(weights * reflected_power)
    weighted_transmitted = jnp.sum(weights * transmitted_power)
    weighted_absorbed = jnp.sum(weights * absorbed_power)
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
        if isinstance(element, PreparedFourierModalLayer)
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
    finite = (
        jnp.all(jnp.isfinite(right_outgoing))
        & jnp.all(jnp.isfinite(left_outgoing))
        & jnp.all(jnp.isfinite(absorbed_power))
    )
    power_scale = jnp.maximum(jnp.max(jnp.abs(incident_power)), 1.0)
    power_balance = (
        jnp.max(
            jnp.abs(incident_power - reflected_power - transmitted_power - absorbed_power)
        )
        / power_scale
    )
    status = jnp.where(
        ~finite,
        int(FourierModalSolveStatus.NONFINITE_RESULT),
        jnp.where(
            ~propagation_converged,
            int(FourierModalSolveStatus.PROPAGATION_TOLERANCE_NOT_MET),
            jnp.where(
                power_balance > prepared.plan.policy.power_tolerance,
                int(FourierModalSolveStatus.POWER_BALANCE_NOT_MET),
                int(FourierModalSolveStatus.SUCCESS),
            ),
        ),
    )
    diagnostics = FourierModalDiagnostics(
        maximum_constitutive,
        maximum_boundary_solve,
        maximum_paired,
        interface.diagnostics.conversion_residual,
        power_balance,
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
        incident_power,
        reflected_power,
        transmitted_power,
        absorbed_power,
        weighted_incident,
        weighted_reflected,
        weighted_transmitted,
        weighted_absorbed,
        boundary_electric if prepared.plan.policy.retain_boundary_fields else (),
        boundary_magnetic if prepared.plan.policy.retain_boundary_fields else (),
        status,
        diagnostics,
        provenance,
    )


def fourier_modal_convergence_report(
    harmonic_counts: Array,
    scattering_matrices: tuple[Array, ...],
    power_values: tuple[Array, ...],
    /,
    *,
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-8,
) -> FourierModalConvergenceReport:
    counts = jnp.asarray(harmonic_counts, dtype=jnp.int32)
    if counts.ndim != 1 or counts.size != len(scattering_matrices):
        raise ValueError("harmonic_counts must match the supplied result sequences.")
    if len(scattering_matrices) < 2 or len(power_values) != len(scattering_matrices):
        raise ValueError(
            "At least two matching scattering and power results are required."
        )
    scattering_differences = []
    power_differences = []
    for previous, current, previous_power, current_power in zip(
        scattering_matrices[:-1],
        scattering_matrices[1:],
        power_values[:-1],
        power_values[1:],
        strict=True,
    ):
        scattering_scale = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(current) ** 2)), 1.0)
        power_scale = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(current_power) ** 2)), 1.0)
        scattering_differences.append(
            jnp.sqrt(jnp.sum(jnp.abs(current - previous) ** 2)) / scattering_scale
        )
        power_differences.append(
            jnp.sqrt(jnp.sum(jnp.abs(current_power - previous_power) ** 2)) / power_scale
        )
    scattering_array = jnp.stack(scattering_differences)
    power_array = jnp.stack(power_differences)
    converged = (scattering_array[-1] <= relative_tolerance + absolute_tolerance) & (
        power_array[-1] <= relative_tolerance + absolute_tolerance
    )
    return FourierModalConvergenceReport(
        counts,
        scattering_array,
        power_array,
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

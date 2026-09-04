#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._local_variational import (
    AbstractPreparedLocalDiscretization,
)
from ...discretization.fem import FiniteElementDiscretization
from ...discretization.fem._high_order import ReferenceNodalFamily
from ...discretization.fem._mortar import (
    FiniteElementMortarMetricData,
    FiniteElementMortarPlan,
)
from ...discretization.fem._reference_operator import PreparedFiniteElementReference
from ._ir import FieldSlot, FiniteElementActionIR, LocalActionIR, RegionIR
from ._kernels import KernelBinding, KernelTable
from ._selection import select_prepared_local_execution
from ._worksets import CompiledWorkset, WorksetProgram, WorksetSignature


def _descriptor(action):
    try:
        descriptor = action.descriptor
    except AttributeError as error:
        raise TypeError(
            "Finite-element lowering requires an action with an explicit descriptor."
        ) from error
    return descriptor


def _domain_for_action(action, discretization: AbstractPreparedLocalDiscretization):
    if action.domain is not None:
        return action.domain
    return discretization.integration_domain(_descriptor(action).default_domain_kind)


def _action_kind(action) -> str:
    return _descriptor(action).action_kind


def _output_fields(action) -> tuple[str, ...]:
    return _descriptor(action).output_fields


def _input_fields(action) -> tuple[str, ...]:
    return _descriptor(action).input_fields


def _operators(action) -> tuple[tuple[str, str], ...]:
    return _descriptor(action).operators


def _region_kind(kind: str) -> str:
    return {
        "cell": "cell",
        "exterior_facet": "exterior-facet",
        "interior_facet": "interior-facet",
    }[kind]


def _default_facet_rule(cell_kind: str):
    from .._finite_element_variational import (
        _interval_rule,
        _quadrilateral_rule,
        _triangle_rule,
    )

    if cell_kind in ("triangle", "quadrilateral"):
        return _interval_rule()
    if cell_kind == "tetrahedron":
        return _triangle_rule()
    if cell_kind == "hexahedron":
        return _quadrilateral_rule()
    raise ValueError(f"No facet rule exists for cell kind {cell_kind!r}.")


def _rule_ids(action, discretization: AbstractPreparedLocalDiscretization):
    from .._finite_element_variational import _action_rule, _rule_id

    domain = _domain_for_action(action, discretization)
    if isinstance(discretization, FiniteElementDiscretization):
        explicit = dict(action.rules)
        result = []
        for block in discretization.mesh.blocks:
            if domain.kind == "cell":
                rule = _action_rule(action, block.name, block.cell_kind)
            else:
                rule = explicit.get(block.name, _default_facet_rule(block.cell_kind))
            result.append((block.name, _rule_id(rule)))
        return tuple(result)
    return tuple((str(name), _rule_id(rule)) for name, rule in action.rules)


def _action_coefficients(action) -> tuple[Any, ...]:
    return _descriptor(action).coefficient_values


def _coefficient_layout_ids(action) -> tuple[str, ...]:
    return tuple(sorted(value.layout_id for value in _action_coefficients(action)))


def _coefficient_fields(
    action, discretization: AbstractPreparedLocalDiscretization
) -> tuple[str, ...]:
    requested = {
        value.field_space_id
        for value in _action_coefficients(action)
        if value.location == "dof"
    }
    fields = []
    for field_space in discretization.field_spaces:
        if field_space.field_space_id in requested:
            fields.append(field_space.name)
    if len(fields) != len(requested):
        raise ValueError("A DOF coefficient field space is absent from the form support.")
    return tuple(fields)


def _tensor_family(element) -> ReferenceNodalFamily | None:
    if element.cell_kind not in ("quadrilateral", "hexahedron"):
        return None
    if (
        element.family
        not in (
            "TensorProductLagrange",
            "DiscontinuousLagrange",
        )
        or element.representation != "point_value"
    ):
        return None
    nodes = np.asarray(element.reference_nodes)
    orders = tuple(np.unique(nodes[:, axis]).size - 1 for axis in range(nodes.shape[1]))
    for node_set in ("gauss-lobatto", "equispaced"):
        candidate = ReferenceNodalFamily(element.cell_kind, orders, node_set=node_set)
        candidate_element = candidate.finite_element()
        if candidate_element.element_id == element.element_id:
            return candidate
        if (
            element.family == "DiscontinuousLagrange"
            and element.tabulator_id == f"discontinuous:{candidate_element.element_id}"
        ):
            return candidate
    return None


def _prepared_reference(action, block, element, precision, domain_kind: str):
    if (
        block.cell_kind not in ("triangle", "quadrilateral", "tetrahedron", "hexahedron")
        or element.representation != "point_value"
        or element.value_shape
        or (
            block.cell_kind in ("triangle", "tetrahedron")
            and _input_fields(action) != _output_fields(action)
        )
    ):
        identifier = canonical_fingerprint(
            {
                "kind": "dense-finite-element-reference-binding",
                "element": element.element_id,
                "rules": _rule_ids_for_block(action, block),
                "precision": precision.policy_id,
                "operators": _operators(action),
            }
        )
        return None, identifier
    from .._finite_element_variational import (
        _action_rule,
        _default_rule,
        _interval_rule,
        _quadrilateral_rule,
        _triangle_rule,
    )

    explicit = dict(action.rules)
    if domain_kind == "cell":
        volume_rule = _action_rule(action, block.name, block.cell_kind)
        facet_rule = {
            "triangle": _interval_rule(),
            "quadrilateral": _interval_rule(),
            "tetrahedron": _triangle_rule(),
            "hexahedron": _quadrilateral_rule(),
        }[block.cell_kind]
    else:
        from ...integration import (
            ReferenceHexahedronRule,
            ReferenceIntervalRule,
            ReferenceQuadrilateralRule,
            ReferenceTetrahedronRule,
            ReferenceTriangleRule,
        )

        default_facet = {
            "triangle": _interval_rule(),
            "quadrilateral": _interval_rule(),
            "tetrahedron": _triangle_rule(),
            "hexahedron": _quadrilateral_rule(),
        }[block.cell_kind]
        facet_rule = explicit.get(block.name, default_facet)
        if block.cell_kind == "triangle" and isinstance(
            facet_rule, ReferenceIntervalRule
        ):
            volume_rule = ReferenceTriangleRule(facet_rule.rule)
        elif block.cell_kind == "quadrilateral" and isinstance(
            facet_rule, ReferenceIntervalRule
        ):
            volume_rule = ReferenceQuadrilateralRule(facet_rule.rule)
        elif block.cell_kind == "tetrahedron" and isinstance(
            facet_rule, ReferenceTriangleRule
        ):
            volume_rule = ReferenceTetrahedronRule(facet_rule.rule)
        elif block.cell_kind == "hexahedron" and isinstance(
            facet_rule, ReferenceQuadrilateralRule
        ):
            volume_rule = ReferenceHexahedronRule(facet_rule.rule)
        else:
            volume_rule = _default_rule(block.cell_kind)
    facet_count = {
        "triangle": 3,
        "quadrilateral": 4,
        "tetrahedron": 4,
        "hexahedron": 6,
    }[block.cell_kind]
    actions = {operation for _, operation in _operators(action)}
    prepared_actions = {"interpolate", "interpolate_transpose"}
    if "grad" in actions or "normal-trace" in actions:
        prepared_actions.update(("gradient", "gradient_transpose"))
    if domain_kind != "cell":
        prepared_actions.update(("trace", "trace_transpose"))
    reference = PreparedFiniteElementReference(
        element,
        volume_rule,
        (facet_rule,) * facet_count,
        tuple(sorted(prepared_actions)),
        precision,
        tensor_family=_tensor_family(element),
    )
    return reference, reference.prepared_id


def _rule_ids_for_block(action, block) -> tuple[str, ...]:
    from .._finite_element_variational import _action_rule, _rule_id

    domain_kind = (
        _descriptor(action).default_domain_kind
        if action.domain is None
        else action.domain.kind
    )
    if domain_kind == "cell":
        rule = _action_rule(action, block.name, block.cell_kind)
    else:
        rule = dict(action.rules).get(block.name, _default_facet_rule(block.cell_kind))
    return (_rule_id(rule),)


def _collocated(reference: PreparedFiniteElementReference | None) -> bool:
    if reference is None or reference.tensor_tabulation is None:
        return False
    return all(
        factor.shape[0] == factor.shape[1]
        and np.allclose(np.asarray(factor), np.eye(factor.shape[0]))
        for factor in reference.tensor_tabulation.basis_factors
    )


def _select_local_kernel(
    requested: str,
    realization: str,
    action,
    domain_kind: str,
    reference: PreparedFiniteElementReference | None,
) -> str:
    action_kind = _action_kind(action)
    tensor = reference is not None and reference.tensor_tabulation is not None
    collocated = _collocated(reference)
    if action_kind == "functional":
        if requested not in ("auto", "dense"):
            raise ValueError(
                "Local functional actions currently require local_kernel='auto' "
                "or 'dense'."
            )
        return "dense"
    if action_kind == "pairwise-volume-flux":
        if domain_kind != "cell" or not collocated:
            raise ValueError(
                "Pairwise volume flux requires a collocated tensor-cell reference."
            )
        if requested not in ("auto", "collocated"):
            raise ValueError("Pairwise volume flux requires local_kernel='collocated'.")
        return "collocated"
    if requested == "auto":
        if realization == "sparse":
            return "dense"
        if collocated:
            return "collocated"
        if tensor:
            return "sum_factorized"
        return "dense"
    if requested == "dense":
        return "dense"
    if requested == "partial":
        if domain_kind != "cell":
            raise ValueError("Partial local kernels are only defined on cells.")
        return "partial"
    if requested == "sum_factorized":
        if not tensor:
            raise ValueError("Sum factorization requires prepared tensor factors.")
        return "sum_factorized"
    if requested == "collocated":
        if not collocated:
            raise ValueError("Collocated kernels require nodal quadrature collocation.")
        return "collocated"
    raise ValueError("Unknown finite-element local-kernel strategy.")


def lower_finite_element_form(
    form: Any,
    discretization: AbstractPreparedLocalDiscretization,
    /,
) -> LocalActionIR:
    if type(form).__name__ != "FiniteElementForm" or not isinstance(
        discretization, AbstractPreparedLocalDiscretization
    ):
        raise TypeError(
            "Expected FiniteElementForm and AbstractPreparedLocalDiscretization."
        )
    slots = []
    for field_name in form.field_names:
        binding = discretization.local_field_binding(field_name)
        slots.append(
            FieldSlot(
                field_name,
                "unknown",
                binding.field_space.vector_space.space_id,
                value_shape=binding.execution_shape,
            )
        )
    actions = []
    for action in form.actions:
        domain = _domain_for_action(action, discretization)
        region = RegionIR(
            _region_kind(domain.kind),
            domain.domain_id,
            _rule_ids(action, discretization),
        )
        actions.append(
            FiniteElementActionIR(
                _action_kind(action),
                _output_fields(action),
                _input_fields(action),
                _operators(action),
                region,
                action.action_id,
            )
        )
    return LocalActionIR(slots, actions)


def _facet_permutations(mesh, owner_cells, neighbour_cells, owner_local, neighbour_local):
    connectivity = mesh.connectivity
    count = owner_cells.size
    if isinstance(connectivity, PolygonalConnectivity):
        signs = np.asarray(connectivity.cell_edge_signs)
        owner = signs[owner_cells, owner_local].astype(np.int32)
        safe_neighbours = np.maximum(neighbour_cells, 0)
        safe_local = np.maximum(neighbour_local, 0)
        neighbour = np.where(
            neighbour_cells >= 0,
            signs[safe_neighbours, safe_local],
            1,
        ).astype(np.int32)
        return owner, neighbour
    if isinstance(connectivity, HexahedralConnectivity):
        permutations = np.asarray(
            connectivity.cell_face_vertex_permutations, dtype=np.int32
        )
        owner = permutations[owner_cells, owner_local]
        safe_neighbours = np.maximum(neighbour_cells, 0)
        safe_local = np.maximum(neighbour_local, 0)
        neighbour = permutations[safe_neighbours, safe_local]
        neighbour = np.where((neighbour_cells >= 0)[:, None], neighbour, np.arange(4))
        return owner, neighbour
    if isinstance(connectivity, TetrahedralConnectivity):
        signs = np.asarray(connectivity.cell_face_signs)
        owner = signs[owner_cells, owner_local].astype(np.int32)
        safe_neighbours = np.maximum(neighbour_cells, 0)
        safe_local = np.maximum(neighbour_local, 0)
        neighbour = np.where(
            neighbour_cells >= 0,
            signs[safe_neighbours, safe_local],
            1,
        ).astype(np.int32)
        return owner, neighbour
    return np.ones((count,), dtype=np.int32), np.ones((count,), dtype=np.int32)


def compile_workset_program(
    ir: LocalActionIR,
    form: Any,
    discretization: AbstractPreparedLocalDiscretization,
    /,
    *,
    local_kernel: str = "auto",
    realization: str = "sparse",
) -> WorksetProgram:
    worksets = []
    common_action_indices = set()
    for action_index, action in enumerate(form.actions):
        descriptor = _descriptor(action)
        domain = _domain_for_action(action, discretization)
        common_action = "prepared-local" in descriptor.provider_offers and (
            (
                not isinstance(discretization, FiniteElementDiscretization)
                and domain.kind in ("cell", "exterior_facet")
                and descriptor.action_kind
                in ("residual", "linear", "energy", "functional")
            )
            or (
                isinstance(discretization, FiniteElementDiscretization)
                and domain.kind == "cell"
                and descriptor.evaluator is None
                and descriptor.action_kind in ("residual", "linear")
            )
        )
        if not common_action:
            continue
        fields = tuple(
            dict.fromkeys(
                _output_fields(action)
                + _input_fields(action)
                + _coefficient_fields(action, discretization)
            )
        )
        bindings = tuple(discretization.local_field_binding(name) for name in fields)
        use_common = not action.rules and (
            not isinstance(discretization, FiniteElementDiscretization)
            or (
                all(binding.conformity == "H1" for binding in bindings)
                and all(
                    len(
                        {
                            element.local_dof_count
                            for element in discretization.elements[
                                discretization._field_index(field)
                            ]
                        }
                    )
                    == 1
                    for field in fields
                )
            )
        )
        if not use_common:
            continue
        operators = _operators(action)
        derivative_order = (
            1
            if any(
                operator in ("grad", "sym-grad", "div", "curl", "normal-trace")
                for _, operator in operators
            )
            else 0
        )
        selection = select_prepared_local_execution(
            action,
            discretization,
            domain,
            requested_kernel_mode=str(local_kernel),
            requested_operator_realization=str(realization),
        )
        mode = selection.kernel_mode
        regions = discretization.prepare_local_regions(
            domain,
            field_names=fields,
            maximum_derivative_order=derivative_order,
            kernel_mode=mode,
        )
        if not regions:
            raise ValueError(
                "Local provider returned no regions for the requested domain."
            )
        for region in regions:
            if (
                region.field_names != fields
                or region.domain.kind != domain.kind
                or region.domain.support_id != domain.support_id
                or region.domain.entity_set_id != domain.entity_set_id
                or any(
                    reference.maximum_derivative_order < derivative_order
                    or mode not in reference.kernel_modes
                    or reference.local_width != binding.local_width
                    or reference.realization_id != selection.reference_realization_id
                    for reference, binding in zip(
                        region.reference_actions, bindings, strict=True
                    )
                )
            ):
                raise ValueError(
                    "Local provider region does not satisfy compiled kernel capabilities."
                )
            gathers = dict(zip(fields, region.field_gathers, strict=True))
            neighbour_gathers = (
                None
                if not region.neighbour_gathers
                else dict(zip(fields, region.neighbour_gathers, strict=True))
            )
            references = tuple(
                reference.action_id for reference in region.reference_actions
            )
            rule_id = dict(ir.actions[action_index].region.rule_ids).get(
                region.block_name,
                canonical_fingerprint(
                    {
                        "kind": "prepared-local-rule",
                        "reference_actions": references,
                    }
                ),
            )
            signature = WorksetSignature(
                _region_kind(region.domain.kind),
                region.block_name,
                region.cell_kind,
                rule_id,
                {
                    name: reference.local_width
                    for name, reference in zip(
                        fields, region.reference_actions, strict=True
                    )
                },
                support_id=region.domain.support_id,
                entity_set_id=region.domain.entity_set_id,
                reference_action_ids=references,
                field_layout_ids=tuple(binding.layout_id for binding in bindings),
                geometry_action_id=region.geometry_actions.action_id,
                coefficient_layout_ids=_coefficient_layout_ids(action),
                precision_id=cast(Any, discretization.precision_policy).policy_id,
                ir_semantics_id=ir.actions[action_index].action_id,
                local_kernel=mode,
                provider_selection_id=selection.selection_id,
                execution_kind=selection.execution_kind,
                operator_realization=selection.operator_realization,
                reference_realization_id=selection.reference_realization_id,
            )
            worksets.append(
                CompiledWorkset(
                    signature,
                    jnp.asarray([action_index], dtype=jnp.int32),
                    region.entity_indices,
                    region.owner_cells,
                    region.neighbour_cells,
                    gathers,
                    local_region=region,
                    neighbour_gathers=neighbour_gathers,
                    owner_local_entities=region.owner_local_entities,
                    neighbour_local_entities=region.neighbour_local_entities,
                    neighbour_trace_permutations=region.trace_permutations,
                    valid=region.valid,
                )
            )
        common_action_indices.add(action_index)
    if len(common_action_indices) == len(form.actions):
        return WorksetProgram(ir, worksets)
    if not isinstance(discretization, FiniteElementDiscretization):
        raise ValueError(
            "Non-cell, pairwise, and prepared-operator actions require finite elements."
        )
    mesh = discretization.mesh
    cell_blocks = np.concatenate(
        tuple(
            np.full((block.cell_count,), index, dtype=np.int32)
            for index, block in enumerate(mesh.blocks)
        )
    )
    cell_locals = np.concatenate(
        tuple(np.arange(block.cell_count, dtype=np.int32) for block in mesh.blocks)
    )
    cell_offset = 0
    for block_index, block in enumerate(mesh.blocks):
        block_cells = np.arange(
            cell_offset,
            cell_offset + block.cell_count,
            dtype=np.int32,
        )
        cell_offset += block.cell_count
        for action_index, action in enumerate(form.actions):
            if action_index in common_action_indices:
                continue
            domain = _domain_for_action(action, discretization)
            if domain.kind == "cell":
                selected = np.flatnonzero(
                    np.isin(block_cells, np.asarray(domain.entity_indices))
                )
                if selected.size == 0:
                    continue
                entity_indices = block_cells[selected]
                owner_cells = entity_indices
                neighbour_cells = np.full_like(entity_indices, -1)
                owner_local_entities = np.full_like(entity_indices, -1)
                neighbour_local_entities = np.full_like(entity_indices, -1)
                owner_permutations = np.ones_like(entity_indices, dtype=np.int32)
                neighbour_permutations = np.ones_like(entity_indices, dtype=np.int32)
                neighbour_trace_permutations = None
            else:
                domain_owners = np.asarray(domain.owner_cells, dtype=np.int32)
                selected = np.flatnonzero(
                    (domain_owners >= block_cells[0]) & (domain_owners <= block_cells[-1])
                )
                if selected.size == 0:
                    continue
                entity_indices = np.asarray(domain.entity_indices)[selected]
                owner_cells = domain_owners[selected]
                neighbour_cells = np.asarray(domain.neighbour_cells)[selected]
                owner_local_entities = np.asarray(
                    domain.owner_local_entities, dtype=np.int32
                )[selected]
                neighbour_local_entities = np.asarray(
                    domain.neighbour_local_entities, dtype=np.int32
                )[selected]
                neighbour_trace_permutations = np.asarray(
                    domain.neighbour_trace_permutations, dtype=np.int32
                )[selected]
                owner_permutations, neighbour_permutations = _facet_permutations(
                    mesh,
                    owner_cells,
                    neighbour_cells,
                    owner_local_entities,
                    neighbour_local_entities,
                )
            fields = tuple(
                dict.fromkeys(
                    _output_fields(action)
                    + _input_fields(action)
                    + _coefficient_fields(action, discretization)
                )
            )
            gathers = {}
            neighbour_gathers = {}
            widths = {}
            for field in fields:
                field_index = discretization._field_index(field)
                dof_map = discretization.dof_maps[field_index]
                if domain.kind == "cell":
                    route = np.asarray(dof_map.cell_dofs[block_index])[selected]
                    neighbour_route = np.full_like(route, -1)
                else:
                    local_owners = owner_cells - block_cells[0]
                    route = np.asarray(dof_map.cell_dofs[block_index])[local_owners]
                    neighbour_rows = []
                    for neighbour in neighbour_cells:
                        if neighbour < 0:
                            neighbour_rows.append(
                                np.full((route.shape[1],), -1, dtype=np.int32)
                            )
                            continue
                        neighbour_block = int(cell_blocks[neighbour])
                        neighbour_local = int(cell_locals[neighbour])
                        neighbour_dofs = np.asarray(
                            dof_map.cell_dofs[neighbour_block][neighbour_local],
                            dtype=np.int32,
                        )
                        if neighbour_dofs.shape != (route.shape[1],):
                            raise ValueError(
                                "Facet neighbours require compatible local widths."
                            )
                        neighbour_rows.append(neighbour_dofs)
                    neighbour_route = np.asarray(neighbour_rows, dtype=np.int32)
                gathers[field] = route
                neighbour_gathers[field] = neighbour_route
                widths[field] = route.shape[1]
            representative_output = _output_fields(action)[0]
            output_field_index = discretization._field_index(representative_output)
            element = discretization.elements[output_field_index][block_index]
            coordinate_element = discretization.coordinate_elements[block_index]
            reference, reference_id = _prepared_reference(
                action,
                block,
                element,
                discretization.precision_policy,
                domain.kind,
            )
            strategy = _select_local_kernel(
                str(local_kernel), str(realization), action, domain.kind, reference
            )
            signature = WorksetSignature(
                _region_kind(domain.kind),
                block.name,
                block.cell_kind,
                dict(ir.actions[action_index].region.rule_ids)[block.name],
                widths,
                support_id=domain.support_id,
                entity_set_id=domain.entity_set_id,
                reference_action_ids=(reference_id,),
                field_layout_ids=tuple(
                    discretization.local_field_binding(field).layout_id
                    for field in fields
                ),
                geometry_action_id=canonical_fingerprint(
                    {
                        "kind": "finite-element-geometry-actions",
                        "coordinate_element": coordinate_element.element_id,
                        "block": block.name,
                    }
                ),
                coefficient_layout_ids=_coefficient_layout_ids(action),
                precision_id=discretization.precision_policy.policy_id,
                ir_semantics_id=ir.actions[action_index].action_id,
                local_kernel=strategy,
            )
            worksets.append(
                CompiledWorkset(
                    signature,
                    jnp.asarray([action_index], dtype=jnp.int32),
                    entity_indices,
                    owner_cells,
                    neighbour_cells,
                    gathers,
                    reference=reference,
                    neighbour_gathers=neighbour_gathers,
                    owner_local_entities=owner_local_entities,
                    neighbour_local_entities=neighbour_local_entities,
                    owner_permutations=owner_permutations,
                    neighbour_permutations=neighbour_permutations,
                    neighbour_trace_permutations=neighbour_trace_permutations,
                )
            )
    return WorksetProgram(ir, worksets)


def compile_finite_element_hp_mortar_workset(
    discretization: FiniteElementDiscretization,
    action_index: int,
    action_ir: FiniteElementActionIR,
    field_name: str,
    owner_block_index: int,
    neighbour_block_index: int,
    owner_cell: int,
    neighbour_cell: int,
    owner_local_facet: int,
    neighbour_local_facet: int,
    owner_reference: PreparedFiniteElementReference,
    neighbour_reference: PreparedFiniteElementReference,
    mortar: FiniteElementMortarPlan,
    metric: FiniteElementMortarMetricData,
    /,
    *,
    entity_index: int,
    entity_set_id: str,
) -> CompiledWorkset:
    """Lower one asymmetric hp mortar patch into the generic workset contract."""

    field_index = discretization._field_index(field_name)
    owner_element = discretization.elements[field_index][owner_block_index]
    neighbour_element = discretization.elements[field_index][neighbour_block_index]

    def trace_indices(element, local_facet):
        nodes = np.asarray(element.reference_nodes)
        if element.cell_kind == "quadrilateral":
            axis_side = ((1, 0), (0, 1), (1, 1), (0, 0))
        elif element.cell_kind == "hexahedron":
            axis_side = ((2, 0), (0, 1), (2, 1), (0, 0), (1, 0), (1, 1))
        else:
            raise ValueError("hp mortar lowering requires quad/hex tensor elements.")
        axis, side = axis_side[int(local_facet)]
        return np.flatnonzero(np.isclose(nodes[:, axis], float(side))).astype(np.int32)

    owner_trace = trace_indices(owner_element, owner_local_facet)
    neighbour_trace = trace_indices(neighbour_element, neighbour_local_facet)
    if (
        owner_trace.size != mortar.left_interpolation.shape[1]
        or neighbour_trace.size != mortar.right_interpolation.shape[1]
    ):
        raise ValueError("Mortar trace widths and finite-element traces disagree.")
    owner_routes = np.asarray(
        discretization.dof_maps[field_index].cell_dofs[owner_block_index][owner_cell]
    )[owner_trace]
    neighbour_routes = np.asarray(
        discretization.dof_maps[field_index].cell_dofs[neighbour_block_index][
            neighbour_cell
        ]
    )[neighbour_trace]
    cell_offsets = np.cumsum(
        np.asarray(
            (0,) + tuple(block.cell_count for block in discretization.mesh.blocks),
            dtype=np.int32,
        )
    )
    owner_global = int(cell_offsets[owner_block_index]) + int(owner_cell)
    neighbour_global = int(cell_offsets[neighbour_block_index]) + int(neighbour_cell)
    coordinate_element = discretization.coordinate_elements[owner_block_index]
    signature = WorksetSignature(
        "interior_facet",
        discretization.mesh.blocks[owner_block_index].name,
        owner_element.cell_kind,
        mortar.plan_id,
        {field_name: owner_trace.size},
        support_id=discretization.support.support_id,
        entity_set_id=entity_set_id,
        reference_action_ids=(
            owner_reference.prepared_id,
            neighbour_reference.prepared_id,
        ),
        field_layout_ids=(discretization.local_field_binding(field_name).layout_id,),
        geometry_action_id=canonical_fingerprint(
            {
                "kind": "finite-element-mortar-geometry-actions",
                "coordinate_element": coordinate_element.element_id,
                "mortar_metric": metric.metric_id,
            }
        ),
        precision_id=discretization.precision_policy.policy_id,
        ir_semantics_id=action_ir.action_id,
        local_kernel="mortar",
        neighbour_local_widths={field_name: neighbour_trace.size},
    )
    return CompiledWorkset(
        signature,
        np.asarray((int(action_index),), dtype=np.int32),
        np.asarray((int(entity_index),), dtype=np.int32),
        np.asarray((owner_global,), dtype=np.int32),
        np.asarray((neighbour_global,), dtype=np.int32),
        {field_name: owner_routes[None, :]},
        reference=owner_reference,
        neighbour_reference=neighbour_reference,
        mortar=mortar,
        mortar_metric=metric,
        neighbour_gathers={field_name: neighbour_routes[None, :]},
        owner_local_entities=np.asarray((owner_local_facet,), dtype=np.int32),
        neighbour_local_entities=np.asarray((neighbour_local_facet,), dtype=np.int32),
    )


def kernel_table_from_form(
    form: Any,
    ir: LocalActionIR,
    workset_program: WorksetProgram,
    discretization: AbstractPreparedLocalDiscretization,
    /,
) -> KernelTable:
    if type(form).__name__ != "FiniteElementForm":
        raise TypeError("Expected FiniteElementForm.")
    bindings = []
    for action_index, action in enumerate(form.actions):
        descriptor = _descriptor(action)
        evaluator = descriptor.evaluator
        signatures = tuple(
            workset.signature
            for workset in workset_program.worksets
            if action_index in workset.action_index_values
        )
        strategies = tuple(sorted(set(value.local_kernel for value in signatures)))
        strategy = (
            strategies[0] if len(strategies) == 1 else f"mixed[{','.join(strategies)}]"
        )
        bindings.append(
            KernelBinding(
                action.action_id,
                descriptor.action_kind,
                evaluator,
                local_kernel=strategy,
                reference_action_ids=tuple(
                    reference
                    for value in signatures
                    for reference in value.reference_action_ids
                ),
                field_layout_ids=tuple(
                    layout for value in signatures for layout in value.field_layout_ids
                ),
                geometry_action_ids=tuple(
                    value.geometry_action_id for value in signatures
                ),
                coefficient_layout_ids=_coefficient_layout_ids(action),
                precision_id=cast(Any, discretization.precision_policy).policy_id,
                ir_semantics_id=ir.actions[action_index].action_id,
            )
        )
    return KernelTable(bindings)


__all__ = [
    "compile_workset_program",
    "kernel_table_from_form",
    "lower_finite_element_form",
]

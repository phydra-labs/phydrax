"""Local clock admission binds actual conserved owners without collectives."""

from copy import copy
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils

import phydrax as phx
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from phydrax._frozendict import frozendict
from phydrax._model._structure import prepare_model_recipe_catalog
from phydrax.lifecycle import (
    _meshing_field_records as fields,
    _meshing_sources as sources,
)
from phydrax.lifecycle._meshing_sources import _validate_meshing_accepted_roles
from phydrax.linalg._spaces import ArraySpace
from tests._support.coordinate_references import native_meshing_request
from tests._support.distributed_meshing.field_history_proof import _prepare_history
from tools.meshing_qualification.scenarios._distributed_euler import (
    COMPONENTS,
    inventory,
    state_units,
)


pytestmark = pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU devices")


@pytest.fixture(scope="module")
def actual_clock_records() -> dict[str, Any]:
    target = _prepare_history().target
    element = phx.discretization.lagrange_element("triangle", 1)
    specifications = (phx.discretization.FiniteElementFieldSpec("temperature", element),)
    precision = phx.discretization.FiniteElementPrecisionPolicy()
    limits = phx.meshing.MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        with budget.bind_ticket(scope.ticket_view()):
            physical = phx.discretization.FiniteElementPlan(
                target.mesh, specifications, coordinate_spec=target.geometry,
                precision_policy=precision, budget=budget,
            ).prepare(numeric_version="clock-state", budget=budget)
            space = physical.field_spaces[0]
            vector_space = space.vector_space
            assert isinstance(vector_space, ArraySpace)
            coefficients = np.zeros(vector_space.shape, dtype=np.float64)
            coefficient_role = fields.MeshingFieldStateRole(
                "temperature", "temperature", "coefficients", coefficients.shape,
                coefficients.dtype, 1,
                index_binding=fields.prepare_meshing_field_index_binding(
                    target, "field/global_coefficient_ids", field_space=space,
                ),
            )
            binding = fields.prepare_meshing_field_index_binding(target, "mesh/cell_ids")
            system = phx.equations.EulerSystem(2)
            primitive = jnp.ones((binding.index_values.size, 4), dtype=jnp.float64)
            conserved = system.primitive_to_conserved(primitive)
            role = fields.MeshingFieldStateRole(
                "euler-conserved", "temperature", "stage-history", conserved.shape,
                np.dtype(conserved.dtype), 1, value_units=state_units(), index_binding=binding,
            )
            epoch_role = fields.MeshingFieldStateRole(
                "accepted-epoch", "temperature", "state-epoch", (), np.dtype(np.int64), 1,
            )
            declaration = fields.MeshingFiniteElementFieldDeclaration(
                part_name="domain", topology_id=target.mesh.topology_id,
                geometry_layout_id=target.geometry.geometry_layout_id,
                field_space_ids=frozendict({"temperature": space.field_space_id}),
                value_units=frozendict({"temperature": (phx.units.KELVIN,)}),
                maximum_derivative_orders=frozendict({"temperature": 1}),
                branch_policy=phx.BranchDifferentiationPolicy.SMOOTH,
                trace_policy=phx.discretization.FieldTracePolicy("cell-sided"),
                state_roles=(coefficient_role, role, epoch_role), history_policy="accepted-state",
                numeric_version="clock-state", finite_element_fields=specifications,
                precision_policy=precision,
            )
    model = {
        "system": system,
        "reconstruction": phx.discretization.PiecewiseConstantReconstruction(),
        "flux": phx.discretization.RusanovFluxPlan(),
        "boundary": phx.discretization.SlipWallBoundary(),
        "time_step": jnp.asarray(0.01, dtype=jnp.float64),
    }
    clock = {
        "time": np.asarray(0.02, dtype=np.float64),
        "steps": np.asarray(2, dtype=np.int64), "time_units": phx.units.SECOND,
        "inventory": inventory(target, conserved, system=system),
        "state_entry": role.entry_name, "component_names": COMPONENTS,
        "model": model, "model_id": canonical_fingerprint(array_tree_fingerprint(model)),
    }
    return {
        "collective_target": target, "field_declarations": {"domain": declaration},
        "accepted_data": {"fields": {"temperature": coefficients, role.entry_name: conserved,
                                    "accepted-epoch": np.asarray(1, dtype=np.int64)},
                          "euler_clock": clock},
    }


def test_actual_owned_euler_clock_admission_uses_no_collectives(
    actual_clock_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def collective_forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("local clock admission must not call a collective scientific validator")

    monkeypatch.setattr(multihost_utils, "process_allgather", collective_forbidden)
    _validate_meshing_accepted_roles(actual_clock_records)


@pytest.mark.parametrize("defect", ("foreign-state", "foreign-target", "components", "model-components", "wrong-role", "model", "missing-clock-role", "undeclared-role"))
def test_clock_admission_refuses_foreign_payload_and_ownership(
    actual_clock_records: dict[str, Any], defect: str
) -> None:
    records = dict(actual_clock_records)
    accepted = dict(records["accepted_data"])
    clock = dict(accepted["euler_clock"])
    records["accepted_data"] = accepted
    accepted["euler_clock"] = clock
    if defect == "foreign-state":
        clock["state_entry"] = "temperature"
    elif defect == "foreign-target":
        declaration = records["field_declarations"]["domain"]
        coefficient, role, epoch_role = declaration.state_roles
        changed = replace(role, index_binding=replace(role.index_binding, source_result_id="foreign-result"))
        records["field_declarations"] = {"domain": replace(declaration, state_roles=(coefficient, changed, epoch_role))}
    elif defect == "components":
        clock["component_names"] = tuple(reversed(clock["component_names"]))
    elif defect == "model-components":
        clock["component_names"] = clock["model"]["system"].component_names
    elif defect == "wrong-role":
        declaration = records["field_declarations"]["domain"]
        coefficient, role, epoch_role = declaration.state_roles
        records["field_declarations"] = {"domain": replace(
            declaration, state_roles=(coefficient, replace(role, role="physical-values"), epoch_role)
        )}
    elif defect == "model":
        clock["model"] = {**clock["model"], "time_step": np.asarray(0.25, dtype=np.float64)}
    elif defect == "missing-clock-role":
        clock.pop("steps")
    else:
        accepted["not-a-declared-role"] = np.asarray(1.0)
    with pytest.raises(ValueError, match="accepted_data"):
        _validate_meshing_accepted_roles(records)


def _field_source_validation(
    target: phx.meshing.CellMeshingResult,
) -> sources.MeshingSourceValidation:
    sources.register_meshing_source_artifacts()
    _, _, arrays = prepare_model_recipe_catalog(
        target, prefix="field-test", limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    banks = {name: value for name, value in arrays.items() if isinstance(value, jax.Array)}
    return sources.MeshingSourceValidation(
        ({"collective_target": target},), banks,
        logical_array_value_collection_digest(banks),
    )


def _replace_clock_bank(
    target: phx.meshing.CellMeshingResult, clock: jax.Array | None
) -> phx.meshing.CellMeshingResult:
    changed = copy(target)
    assert target.collective_evidence is not None
    evidence = copy(target.collective_evidence)
    banks = dict(evidence.logical_arrays)
    if clock is None:
        banks.pop("epoch/clocks")
    else:
        banks["epoch/clocks"] = clock
    object.__setattr__(evidence, "logical_arrays", tuple(sorted(banks.items())))
    object.__setattr__(changed, "collective_evidence", evidence)
    return changed


def test_field_index_clock_identity_matches_actual_digest(
    actual_clock_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    target = actual_clock_records["collective_target"]
    binding = fields.prepare_meshing_field_index_binding(target, "epoch/cell_ids")
    proof = _field_source_validation(target)
    clocks = dict(target.collective_evidence.logical_arrays)["epoch/clocks"]
    expected = logical_array_value_collection_digest({"epoch/clocks": clocks})

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("admitted field index validation entered a collective")

    monkeypatch.setattr(multihost_utils, "process_allgather", forbidden)
    monkeypatch.setattr(fields, "logical_array_value_collection_digest", forbidden)
    admitted = fields.prepare_meshing_field_index_binding(
        target, "epoch/cell_ids", source_validation=proof
    )
    assert admitted.source_epoch_id == binding.source_epoch_id == expected
    assert admitted.index_values is binding.index_values


@pytest.mark.parametrize("defect", ("changed", "missing", "unbound", "deleted"))
def test_field_index_refuses_changed_clock_binding(
    actual_clock_records: dict[str, Any], defect: str
) -> None:
    target = actual_clock_records["collective_target"]
    clocks = dict(target.collective_evidence.logical_arrays)["epoch/clocks"]
    # Keep deletion isolated from the module-scoped scientific fixture.
    if defect == "deleted":
        target = _prepare_history().target
        assert target.collective_evidence is not None
        clocks = dict(target.collective_evidence.logical_arrays)["epoch/clocks"]
    proof = _field_source_validation(target)
    if defect == "deleted":
        clocks.delete()
        changed = target
    elif defect == "missing":
        changed = _replace_clock_bank(target, None)
    else:
        changed = _replace_clock_bank(
            target, clocks + 1 if defect == "changed" else jnp.array(clocks, copy=True)
        )
    with pytest.raises(ValueError):
        fields.prepare_meshing_field_index_binding(
            changed, "epoch/cell_ids", source_validation=proof
        )


@pytest.mark.parametrize("defect", ("rows", "field-space", "source"))
def test_field_index_refuses_altered_identity(
    actual_clock_records: dict[str, Any], defect: str
) -> None:
    target = actual_clock_records["collective_target"]
    declaration = actual_clock_records["field_declarations"]["domain"]
    role = declaration.state_roles[1]
    binding = role.index_binding
    proof = _field_source_validation(target)
    if defect == "rows":
        values = np.asarray(binding.index_values).copy()
        values.flat[0] += 1
        values.flags.writeable = False
        changed = replace(binding, index_values=values)
    elif defect == "field-space":
        coefficient = declaration.state_roles[0].index_binding
        changed = replace(
            binding,
            field_space_id=coefficient.field_space_id,
            coefficient_bank=coefficient.coefficient_bank,
        )
    else:
        changed = replace(binding, source_result_id="foreign-result")
    with pytest.raises(ValueError):
        fields.validate_meshing_field_index_binding(
            replace(role, index_binding=changed), target, source_validation=proof
        )


def test_field_index_normal_and_empty_roles(
    actual_clock_records: dict[str, Any],
) -> None:
    target = actual_clock_records["collective_target"]
    declaration = actual_clock_records["field_declarations"]["domain"]
    proof = _field_source_validation(target)
    for role in declaration.state_roles[1:]:
        fields.validate_meshing_field_index_binding(role, target, source_validation=proof)
    fields.validate_meshing_field_declarations({}, source_validation=proof)


def test_field_index_local_comparison_preserves_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = sources.MeshingSourceValidation(
        (), {}, logical_array_value_collection_digest({})
    )
    first = jnp.asarray([3, 8, 12], dtype=jnp.int64)
    same = jnp.array(first, copy=True)
    changed = first.at[1].set(9)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("admitted row comparison must compare actual local bytes")

    monkeypatch.setattr(fields, "logical_array_value_collection_digest", forbidden)
    assert fields._same_index_bank(first, same, source_validation=proof)
    assert not fields._same_index_bank(first, changed, source_validation=proof)


@pytest.mark.parametrize("subclass", (False, True))
def test_field_helpers_refuse_foreign_proof_callbacks(
    actual_clock_records: dict[str, Any], subclass: bool
) -> None:
    calls: list[str] = []

    class ForeignCallbacks:
        def field_epoch_clock_digest(self, carrier: Any) -> str:
            calls.append("clock")
            return "caller-expected-digest"

        def array_equal(self, first: Any, second: Any) -> bool:
            calls.append("array")
            return True

        def equivalent(self, first: Any, second: Any, **kwargs: Any) -> bool:
            calls.append("equivalent")
            return True

    class ForeignProof(ForeignCallbacks, sources.MeshingSourceValidation):
        pass

    # No scientific proof is constructed: every helper must reject its type
    # before accessing even one caller-provided attribute or callback.
    proof = object.__new__(ForeignProof if subclass else ForeignCallbacks)
    target = actual_clock_records["collective_target"]
    declaration = actual_clock_records["field_declarations"]["domain"]
    values = np.asarray([1], dtype=np.int64)
    operations = (
        lambda: fields.prepare_meshing_field_index_binding(
            target, "epoch/cell_ids", source_validation=proof
        ),
        lambda: fields.validate_meshing_field_index_binding(
            declaration.state_roles[-1], target, source_validation=proof
        ),
        lambda: fields._same_index_bank(values, values, source_validation=proof),
        lambda: fields.prepare_meshing_field_owner(
            declaration, target, source_validation=proof
        ),
        lambda: fields.prepare_meshing_polyhedral_owner_binding(
            target, source_validation=proof
        ),
        lambda: fields.validate_meshing_field_declarations({}, source_validation=proof),
        lambda: fields.validate_meshing_accepted_epochs({}, source_validation=proof),
        lambda: fields._validate_field_owner_banks((), {}, source_validation=proof),
    )
    for operation in operations:
        with pytest.raises(TypeError, match="exact MeshingSourceValidation"):
            operation()
    assert calls == []

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import FiniteElementDiscretization
from phydrax.discretization.fem._point_interpolation import (
    prepare_finite_element_field_reconstruction,
)
from phydrax.discretization.finite_volume import (
    CellPolynomialReconstructionPlan,
    PiecewiseConstantReconstruction,
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    MeshingFieldDeclaration,
    MeshingSourceClosure,
    prepare_meshing_field_dynamics,
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    recertify_restored_meshing_source,
    validate_meshing_field_declarations,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_sources import validate_meshing_source_closure
from phydrax.meshing import CellMeshingResult
from tests._support.meshing_field_declarations import (
    NativeFieldScenario,
    prepare_native_field_scenario,
    prepare_single_native_field_scenario,
)


FINITE_ELEMENT_KINDS = (
    "mapped-scalar",
    "mapped-vector",
    "rt",
    "bdm2",
    "nedelec",
    "nedelec1",
)
FINITE_VOLUME_KINDS = ("fv-scalar", "fv-euler-history")
FIELD_KINDS = FINITE_ELEMENT_KINDS + FINITE_VOLUME_KINDS
# Locked native physical archive acceptance budgets from the cold-field
# scenario in test_meshing_restart.py. Reconstruction and reduction order are
# unchanged; these bounds are retained rather than widened during the split.
FV_ARCHIVE_ATOL = 1e-13
FE_RECONSTRUCTION_ATOL = 1e-12
FE_CONTINUATION_ATOL = 1e-10


@dataclass(frozen=True, slots=True)
class PublishedFieldState:
    native: NativeFieldScenario
    restored: MeshingSourceClosure
    carrier: CellMeshingResult
    declaration: MeshingFieldDeclaration
    prepared: FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization
    reconstruction: Any
    coefficients: np.ndarray
    report: Any
    accepted_data: Mapping[str, Any]


@pytest.fixture(scope="module")
def declared_field(request: pytest.FixtureRequest) -> NativeFieldScenario:
    return prepare_native_field_scenario(request.param)


@pytest.fixture(scope="module")
def published_field(
    declared_field: NativeFieldScenario,
    tmp_path_factory: pytest.TempPathFactory,
) -> PublishedFieldState:
    path = tmp_path_factory.mktemp("physical-field") / "source.zip"
    receipt = write_meshing_source_closure(
        path, declared_field.closure, limits=declared_field.limits
    )
    restored = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
        limits=declared_field.limits,
    )
    assert isinstance(restored, MeshingSourceClosure)
    registration = restored.registration
    assert registration is not None
    connectivity = registration.prepare()
    connectivity.require_complete()
    carrier = connectivity.assembly.part(declared_field.part.name).carrier
    assert isinstance(carrier, CellMeshingResult)
    inputs = restored.certification_inputs
    assert inputs is not None
    report = recertify_restored_meshing_source(
        inputs,
        carrier.mesh,
        carrier.geometry,
        carrier.audit,
        archive_limits=declared_field.limits,
    )
    declarations = restored.field_declarations
    assert declarations is not None
    declaration = declarations[declared_field.part.name]
    prepared, reconstruction = prepare_meshing_field_owner(declaration, carrier)
    accepted_data = restored.accepted_data
    assert accepted_data is not None
    coefficients = accepted_data["fields"]["physical-coefficients"]
    return PublishedFieldState(
        declared_field,
        restored,
        carrier,
        declaration,
        prepared,
        reconstruction,
        coefficients,
        report,
        accepted_data,
    )


@pytest.mark.parametrize("declared_field", FIELD_KINDS, indirect=True, ids=FIELD_KINDS)
def test_cold_field_report_identity(published_field: PublishedFieldState) -> None:
    assert published_field.report.report_id == published_field.native.report.report_id


@pytest.mark.parametrize("declared_field", FIELD_KINDS, indirect=True, ids=FIELD_KINDS)
def test_cold_field_coefficients_and_units(published_field: PublishedFieldState) -> None:
    state = published_field
    np.testing.assert_array_equal(state.coefficients, state.native.coefficients)
    assert not state.coefficients.flags.writeable
    assert (
        state.declaration.value_units[state.native.name][0].unit_id
        == state.native.declaration.value_units[state.native.name][0].unit_id
    )


@pytest.mark.parametrize(
    "declared_field", FINITE_ELEMENT_KINDS, indirect=True, ids=FINITE_ELEMENT_KINDS
)
def test_cold_finite_element_values_and_derivatives(
    published_field: PublishedFieldState,
) -> None:
    state = published_field
    original, prepared = state.native.original, state.prepared
    assert isinstance(original, FiniteElementDiscretization)
    assert isinstance(prepared, FiniteElementDiscretization)
    for block in state.native.carrier.mesh.blocks:
        sites = np.mean(
            np.asarray(state.native.carrier.mesh.coordinates)[np.asarray(block.vertices)],
            axis=1,
        )
        before = prepare_finite_element_field_reconstruction(
            original, state.native.name, block_name=block.name
        )
        after = prepare_finite_element_field_reconstruction(
            prepared, state.native.name, block_name=block.name
        )
        np.testing.assert_allclose(
            after.evaluate(state.coefficients, sites).values,
            before.evaluate(state.native.coefficients, sites).values,
            atol=FE_RECONSTRUCTION_ATOL,
        )
        derivative = (1, 0, 0)
        np.testing.assert_allclose(
            after.derivative(state.coefficients, sites, derivative).values,
            before.derivative(state.native.coefficients, sites, derivative).values,
            atol=FE_RECONSTRUCTION_ATOL,
        )


@pytest.mark.parametrize(
    "declared_field", FINITE_ELEMENT_KINDS, indirect=True, ids=FINITE_ELEMENT_KINDS
)
def test_cold_finite_element_refuses_foreign_family(
    published_field: PublishedFieldState,
) -> None:
    state = published_field
    wrong = phx.discretization.FiniteElementFieldSpec(
        state.native.name,
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
            for block in state.carrier.mesh.blocks
        },
    )
    with pytest.raises(ValueError, match="field_space_ids"):
        prepare_meshing_field_owner(
            replace(state.declaration, finite_element_fields=(wrong,)),
            state.carrier,
        )


@pytest.mark.parametrize(
    "declared_field", FINITE_VOLUME_KINDS, indirect=True, ids=FINITE_VOLUME_KINDS
)
def test_cold_finite_volume_identities_and_extensive_state(
    published_field: PublishedFieldState,
) -> None:
    state = published_field
    original, prepared = state.native.original, state.prepared
    assert isinstance(original, UnstructuredFiniteVolumeDiscretization)
    assert isinstance(prepared, UnstructuredFiniteVolumeDiscretization)
    np.testing.assert_array_equal(prepared.cell_global_ids, original.cell_global_ids)
    np.testing.assert_allclose(
        np.sum(np.asarray(prepared.cell_volumes)[:, None] * state.coefficients, axis=0),
        np.sum(
            np.asarray(original.cell_volumes)[:, None] * state.native.coefficients, axis=0
        ),
        atol=FV_ARCHIVE_ATOL,
    )
    assert type(state.reconstruction) is PiecewiseConstantReconstruction


@pytest.mark.parametrize(
    "declared_field", FINITE_VOLUME_KINDS, indirect=True, ids=FINITE_VOLUME_KINDS
)
def test_cold_finite_volume_refuses_foreign_reconstruction(
    published_field: PublishedFieldState,
) -> None:
    with pytest.raises(ValueError, match="reconstruction_policy_id"):
        replace(
            published_field.declaration,
            reconstruction_policy=CellPolynomialReconstructionPlan(1),
        )


@pytest.mark.parametrize(
    "declared_field", ("fv-euler-history",), indirect=True, ids=("fv-euler-history",)
)
def test_authored_field_refuses_foreign_epoch(
    declared_field: NativeFieldScenario,
) -> None:
    corrupted = dict(declared_field.fields)
    corrupted["state-epoch"] = np.asarray(8, dtype=np.int64)
    with pytest.raises(ValueError, match="MeshingFieldStateRole.state_epoch"):
        validate_meshing_source_closure(
            {**declared_field.closure, "accepted_data": {"fields": corrupted}},
            limits=declared_field.limits,
        )


@pytest.mark.parametrize(
    "declared_field", ("fv-euler-history",), indirect=True, ids=("fv-euler-history",)
)
def test_cold_material_history_and_epoch(published_field: PublishedFieldState) -> None:
    state = published_field
    np.testing.assert_array_equal(
        state.accepted_data["fields"]["material-history"],
        state.native.fields["material-history"],
    )
    assert int(state.accepted_data["fields"]["state-epoch"]) == 7


@pytest.mark.parametrize(
    "declared_field", ("fv-euler-history",), indirect=True, ids=("fv-euler-history",)
)
def test_cold_material_history_refuses_reordered_identity_bank(
    published_field: PublishedFieldState,
) -> None:
    state = published_field
    history_role = next(
        role for role in state.declaration.state_roles if role.role == "material-history"
    )
    history_binding = history_role.index_binding
    assert history_binding is not None
    reordered_bank = replace(
        history_binding,
        index_values=np.asarray(history_binding.index_values)[::-1].copy(),
    )
    wrong_roles = tuple(
        replace(role, index_binding=reordered_bank) if role is history_role else role
        for role in state.declaration.state_roles
    )
    with pytest.raises(ValueError, match="index_binding.index_values"):
        prepare_meshing_field_owner(
            replace(state.declaration, state_roles=wrong_roles), state.carrier
        )


@pytest.mark.parametrize(
    "declared_field", ("fv-euler-history",), indirect=True, ids=("fv-euler-history",)
)
def test_cold_euler_continuation_preserves_rates_and_conserved_state(
    published_field: PublishedFieldState,
) -> None:
    state = published_field
    declaration = state.declaration
    assert isinstance(declaration, phx.lifecycle.MeshingFiniteVolumeFieldDeclaration)
    native_declaration = state.native.declaration
    assert isinstance(
        native_declaration, phx.lifecycle.MeshingFiniteVolumeFieldDeclaration
    )
    dynamics = prepare_meshing_field_dynamics(declaration, state.carrier)
    original_dynamics = prepare_meshing_field_dynamics(
        native_declaration, state.native.carrier
    )
    time = jnp.asarray(0.0, dtype=jnp.float64)
    cold_values = jnp.asarray(state.coefficients)
    native_values = jnp.asarray(state.native.coefficients)
    cold_rate = np.asarray(dynamics(time, cold_values))
    original_rate = np.asarray(original_dynamics(time, native_values))
    np.testing.assert_allclose(cold_rate, original_rate, atol=FV_ARCHIVE_ATOL)
    prepared = state.prepared
    assert isinstance(prepared, UnstructuredFiniteVolumeDiscretization)
    volumes = np.asarray(prepared.cell_volumes)
    np.testing.assert_allclose(
        np.sum(volumes * cold_rate[:, 0]), 0.0, atol=FV_ARCHIVE_ATOL
    )
    np.testing.assert_allclose(
        np.sum(volumes * cold_rate[:, -1]), 0.0, atol=FV_ARCHIVE_ATOL
    )
    next_state = state.coefficients + 1e-3 * cold_rate
    restored_system = declaration.finite_volume_system
    assert isinstance(restored_system, EulerSystem)
    assert bool(np.all(restored_system.admissible(next_state)))
    np.testing.assert_allclose(
        np.sum(volumes[:, None] * next_state[:, (0, -1)], axis=0),
        np.sum(volumes[:, None] * state.coefficients[:, (0, -1)], axis=0),
        atol=FV_ARCHIVE_ATOL,
    )


@pytest.mark.parametrize(
    "declared_field", ("mapped-scalar",), indirect=True, ids=("mapped-scalar",)
)
def test_cold_finite_element_essential_boundary_continuation(
    published_field: PublishedFieldState,
) -> None:
    state = published_field
    prepared = state.prepared
    assert isinstance(prepared, FiniteElementDiscretization)

    def harmonic(points: jax.Array, /) -> jax.Array:
        return points @ jnp.asarray((1.0, 2.0, 3.0))

    form = phx.equations.FiniteElementForm(
        "cold-physical-diffusion",
        state.native.name,
        (phx.equations.DiffusionAction(state.native.name, 1.0),),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=phx.discretization.dirichlet_constraint(
            prepared,
            state.native.name,
            boundary_mask=state.accepted_data["fields"]["essential-mask"],
        ),
        dirichlet_values=state.accepted_data["fields"]["essential-values"],
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(
        operator,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=1e-13)
        ),
    )
    assert bool(jnp.all(solved.successful))
    np.testing.assert_allclose(
        problem.expand(solved.value),
        harmonic(prepared.dof_maps[0].dof_coordinates),
        atol=FE_CONTINUATION_ATOL,
    )


@pytest.fixture(scope="module")
def declared_single_field() -> NativeFieldScenario:
    return prepare_single_native_field_scenario()


@pytest.fixture(scope="module")
def published_single_field(
    declared_single_field: NativeFieldScenario,
    tmp_path_factory: pytest.TempPathFactory,
) -> PublishedFieldState:
    native = declared_single_field
    receipt = write_meshing_source_closure(
        tmp_path_factory.mktemp("single-field") / "source.zip",
        native.closure,
        limits=native.limits,
    )
    restored = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
        limits=native.limits,
    )
    assert isinstance(restored, MeshingSourceClosure)
    generation = restored.generation_part
    assert generation is not None
    carrier = generation.carrier
    assert isinstance(carrier, CellMeshingResult)
    declarations = restored.field_declarations
    assert declarations is not None
    declaration = declarations[native.part.name]
    prepared, reconstruction = prepare_meshing_field_owner(declaration, carrier)
    accepted_data = restored.accepted_data
    assert accepted_data is not None
    return PublishedFieldState(
        native,
        restored,
        carrier,
        declaration,
        prepared,
        reconstruction,
        accepted_data["fields"]["temperature"],
        restored.report,
        accepted_data,
    )


def test_single_native_field_archive_preserves_layout_and_banks(
    published_single_field: PublishedFieldState,
) -> None:
    state = published_single_field
    assert "registration" not in state.restored
    assert (
        state.prepared.field_spaces[0].field_space_id
        == state.native.original.field_spaces[0].field_space_id
    )
    np.testing.assert_array_equal(
        state.accepted_data["fields"]["temperature"], state.native.coefficients
    )
    np.testing.assert_array_equal(
        state.accepted_data["fields"]["material-temperature"],
        state.native.fields["material-temperature"],
    )
    assert int(np.asarray(state.accepted_data["fields"]["accepted-epoch"])) == 7


def test_single_native_field_archive_refuses_foreign_topology(
    published_single_field: PublishedFieldState,
) -> None:
    state = published_single_field
    stale = dict(state.restored)
    stale["field_declarations"] = {
        state.native.part.name: replace(
            state.declaration, topology_id="other-accepted-topology"
        ),
    }
    with pytest.raises(ValueError, match="topology_id"):
        validate_meshing_field_declarations(stale)


def test_single_native_field_archive_refuses_foreign_epoch(
    published_single_field: PublishedFieldState,
) -> None:
    state = published_single_field
    wrong_epoch = dict(state.restored)
    wrong_epoch["accepted_data"] = {
        "fields": {
            **state.accepted_data["fields"],
            "accepted-epoch": np.asarray(8, dtype=np.int64),
        }
    }
    with pytest.raises(ValueError, match="state_epoch"):
        validate_meshing_field_declarations(wrong_epoch)


@dataclass(frozen=True, slots=True)
class PublishedFieldEpoch:
    native: NativeFieldScenario
    epoch: MeshingAcceptedEpoch
    closure: dict[str, Any]
    restored: MeshingSourceClosure
    accepted_data: Mapping[str, Any]


@pytest.fixture(scope="module")
def published_field_epoch(
    declared_single_field: NativeFieldScenario,
    tmp_path_factory: pytest.TempPathFactory,
) -> PublishedFieldEpoch:
    native = declared_single_field
    epoch = MeshingAcceptedEpoch(native.carrier, (native.declaration,), native.fields)
    closure = {
        name: value
        for name, value in native.closure.items()
        if name not in ("generation_part", "field_declarations", "accepted_data")
    }
    closure["accepted_target"] = native.carrier
    closure["accepted_data"] = {"accepted_epochs": (epoch,)}
    receipt = write_meshing_source_closure(
        tmp_path_factory.mktemp("field-epoch") / "source.zip",
        closure,
        limits=native.limits,
    )
    restored = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
        limits=native.limits,
    )
    assert isinstance(restored, MeshingSourceClosure)
    accepted_data = restored.accepted_data
    assert accepted_data is not None
    return PublishedFieldEpoch(native, epoch, closure, restored, accepted_data)


def test_accepted_field_epoch_archive_preserves_identity(
    published_field_epoch: PublishedFieldEpoch,
) -> None:
    state = published_field_epoch
    assert "generation_part" not in state.restored
    assert state.accepted_data["accepted_epochs"][0].epoch_id == state.epoch.epoch_id
    target = state.restored.accepted_target
    assert target is not None
    assert target.result_id == state.native.carrier.result_id


@pytest.mark.parametrize("duplicate_role", ["generation-part", "adaptation-event"])
def test_accepted_field_epoch_refuses_duplicate_history_owners(
    published_field_epoch: PublishedFieldEpoch,
    duplicate_role: str,
) -> None:
    state = published_field_epoch
    duplicates = {
        "generation-part": {**state.closure, "generation_part": state.native.part},
        "adaptation-event": {
            **state.closure,
            "accepted_data": {"accepted_epochs": (state.epoch,), "adaptation_event": {}},
        },
    }
    expected_refusal = (
        r"source_closure\.roles must match its retained source binding; got .*mislabeled as generation"
        if duplicate_role == "generation-part"
        else "accepted_epochs|adaptation_event|accepted_data"
    )
    with pytest.raises(ValueError, match=expected_refusal):
        validate_meshing_source_closure(
            duplicates[duplicate_role], limits=state.native.limits
        )

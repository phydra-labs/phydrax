"""Canonical physical bank ownership; these tests do not claim a PDE solve."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle import _meshing_field_records as fields
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.meshing.qualification._native_lifecycle import execute_native_request
from phydrax.meshing.qualification._polyhedral import PolyhedralArchiveOwners
from tests._support.polyhedral_meshes import voxel_plc


def _fail(stage: str, message: str, data: dict[str, object]) -> Exception:
    return ValueError(f"{stage}: {message}; {data}")


@pytest.fixture(scope="module")
def actual_epoch() -> tuple[fields.MeshingAcceptedEpoch, dict[str, Any]]:
    source = phx.meshing.NativePlcSource(voxel_plc([(0, 0, 0)]), "owned-cube", "r1")
    scope = phx.meshing.MeshingScope(
        source.source_id, source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY, 2, "owned-cube-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("polyhedron",))
        ),
        scope, phx.meshing.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(phx.meshing.UniformSizeControl(
            scope, 10.0, strength=phx.meshing.SizeControlStrength.SOFT
        ),),
    )
    options = phx.meshing.NativeMeshingOptions("plc_restricted_power")
    carrier = phx.meshing.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()

    def prepare(budget: CoordinateEnclosureBudget) -> fields.MeshingAcceptedEpoch:
        binding, _ = fields.prepare_meshing_polyhedral_owner_binding(carrier, budget=budget)
        roles = []
        arrays = {}

        def add(
            name: str,
            owner: fields.MeshingPolyhedralOwner,
            bank: fields.MeshingPolyhedralIndexBank,
            meaning: fields.MeshingPolyhedralAlgebraicRole,
            role: fields.MeshingFieldStateKind,
            axes: tuple[int, ...],
            shape: tuple[int, ...],
            units: phx.units.UnitDefinition,
            components: tuple[str, ...] = (),
        ) -> None:
            arrays[name] = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 16.0
            roles.append(fields.MeshingPolyhedralFieldStateRole(
                entry_name=name, field_name=owner, role=role, shape=shape,
                dtype=np.dtype("float64"), state_epoch=2,
                value_units=(units,), algebraic_role=meaning,
                component_names=components,
                index_binding=fields.MeshingPolyhedralFieldIndexBinding(
                    owner_binding=binding, owner=owner, bank_name=bank, index_axes=axes,
                ),
            ))

        nv = binding.coefficient_banks["vem"].row_count
        nf = binding.coefficient_banks["source_star"].row_count
        nc = binding.index_banks["cell_ids"].size
        nt = binding.index_banks["active_face_ids"].size
        add("vem", "vem", "coefficients", "primal", "coefficients", (0,), (nv,), phx.units.ONE)
        add("fem", "source_star", "coefficients", "primal", "coefficients", (0,), (nf,), phx.units.ONE)
        add("hybrid", "hybrid", "cell-trace", "primal", "physical-values", (0,), (nc + nt,), phx.units.ONE)
        add("local-flux", "hybrid", "local-facet", "physical", "physical-values", (0, 1), tuple(binding.index_banks["cell_faces"].shape), phx.units.METER)
        add("dual-history", "vem", "coefficients", "dual", "stage-history", (1,), (3, nv), phx.units.METER)
        add("conserved", "hybrid", "cell", "physical", "physical-values", (0,), (nc, 7), phx.units.ONE,
            ("density", "momentum_x", "momentum_y", "momentum_z", "total_energy", "species_0", "species_1"))
        declaration = fields.prepare_meshing_polyhedral_field_declaration(
            carrier, binding, tuple(roles), part_name="owned-cube",
            normalization_reference_length=phx.units.METER,
        )
        epoch = fields.MeshingAcceptedEpoch(carrier, (declaration,), arrays)
        first, _ = fields.prepare_meshing_field_owner(declaration, carrier, budget=budget)
        second, _ = fields.prepare_meshing_field_owner(declaration, carrier, budget=budget)
        assert first is second
        return epoch

    request = execute_native_request(
        carrier, specification.limits, prepare,
        target=lambda epoch: epoch.carrier, failure_factory=_fail,
    )
    # Settling the same request changes only its receipt, never its earned layout
    # proof or bank identity. Authenticate that congruence before shallow binding.
    epoch = fields.rebind_meshing_accepted_epoch_execution(
        request.value, request.result, transition=None
    )
    assert epoch.epoch_id == request.value.epoch_id
    assert carrier.certification is not None
    return epoch, {
        "generation_source": source, "generation_specification": specification,
        "generation_options": options, "certification_inputs": carrier.certification.request,
        "report": carrier.certification, "associations": carrier.associations,
        "accepted_target": epoch.carrier, "accepted_data": {"accepted_epochs": (epoch,)},
    }


def test_actual_polyhedral_banks_restore_on_one_current_request(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]], tmp_path: Path,
) -> None:
    epoch, records = actual_epoch
    receipt = write_meshing_source_closure(tmp_path / "physical.phx", records)
    controls = read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id
    )

    def restore(budget: CoordinateEnclosureBudget) -> fields.MeshingAcceptedEpoch:
        restored = read_meshing_source_closure(
            receipt.path, expected_content_id=receipt.content_id, budget=budget
        )
        (current,) = fields.validate_meshing_accepted_epochs(restored)
        owner, _ = fields.prepare_meshing_field_owner(
            current.declarations[0], current.carrier, budget=budget
        )
        assert isinstance(owner, PolyhedralArchiveOwners)
        assert owner.source_star.source_mesh_id == current.carrier.mesh.mesh_id
        for name, value in epoch.fields.items():
            np.testing.assert_array_equal(current.fields[name], value)
        return current

    result = execute_native_request(
        controls.execution_evidence, controls.limits, restore,
        target=lambda current: current.carrier, failure_factory=_fail,
    )
    assert result.value.epoch_id == epoch.epoch_id
    assert result.result.execution_evidence is not None
    assert result.result.execution_evidence.history_evidence


def test_polyhedral_components_and_algebraic_meaning_are_explicit(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]],
) -> None:
    epoch, _ = actual_epoch
    declaration = epoch.declarations[0]
    assert isinstance(declaration, fields.MeshingPolyhedralFieldDeclaration)
    role = next(role for role in declaration.state_roles if role.entry_name == "conserved")
    assert isinstance(role, fields.MeshingPolyhedralFieldStateRole)
    with pytest.raises(ValueError, match="component_names"):
        replace(role, component_names=())
    coefficient = next(role for role in declaration.state_roles if role.entry_name == "vem")
    with pytest.raises(ValueError, match="algebraic_role"):
        replace(coefficient, algebraic_role="dual")
    assert declaration.normalization_reference_length == phx.units.METER


def test_polyhedral_hybrid_has_no_fictional_coefficient_space(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]],
) -> None:
    epoch, _ = actual_epoch
    declaration = epoch.declarations[0]
    assert isinstance(declaration, fields.MeshingPolyhedralFieldDeclaration)
    binding = declaration.owner_binding
    assert set(binding.field_space_ids) == {"vem", "source_star"}
    with pytest.raises(ValueError, match="invented coefficient"):
        fields.MeshingPolyhedralFieldIndexBinding(
            owner_binding=binding, owner="hybrid", bank_name="coefficients", index_axes=(0,)
        )


def test_polyhedral_owner_binding_rejects_changed_source_maps(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]],
) -> None:
    epoch, _ = actual_epoch
    declaration = epoch.declarations[0]
    assert isinstance(declaration, fields.MeshingPolyhedralFieldDeclaration)
    # A changed map gets an honest new binding ID, but cannot borrow source proofs.
    changed = replace(declaration.owner_binding,
        index_banks=frozendict({**declaration.owner_binding.index_banks,
            "source_face_indices": declaration.owner_binding.index_banks["source_face_indices"] + 1}),
    )
    roles = tuple(
        replace(
            cast(fields.MeshingPolyhedralFieldStateRole, role),
            index_binding=replace(
                cast(fields.MeshingPolyhedralFieldIndexBinding, role.index_binding),
                owner_binding=changed,
            ),
        )
        for role in declaration.state_roles
    )
    altered = fields.prepare_meshing_polyhedral_field_declaration(
        epoch.carrier, changed, roles, part_name=declaration.part_name,
        normalization_reference_length=phx.units.METER,
    )
    with pytest.raises(ValueError, match="changed scientific ownership"):
        fields.prepare_meshing_field_owner(altered, epoch.carrier)


def test_epoch_receipt_sealing_never_reprepares_its_admitted_owners(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch, _ = actual_epoch

    def reproof_forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("receipt sealing must not reprepare geometry or field owners")

    monkeypatch.setattr(fields, "_validate_field_owner_banks", reproof_forbidden)
    sealed = fields.rebind_meshing_accepted_epoch_execution(
        epoch, epoch.carrier, transition=None
    )
    assert sealed.carrier is epoch.carrier
    assert sealed.declarations is epoch.declarations
    assert sealed.fields is epoch.fields
    assert sealed.epoch_id == epoch.epoch_id


def test_epoch_receipt_sealing_authenticates_current_field_bytes(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]],
) -> None:
    epoch, _ = actual_epoch
    bank = epoch.fields["vem"]
    assert isinstance(bank, np.ndarray)
    original = bank.flat[0]
    try:
        bank.flat[0] = np.nextafter(original, np.inf)
        with pytest.raises(ValueError, match="changed epoch payload"):
            fields.rebind_meshing_accepted_epoch_execution(
                epoch, epoch.carrier, transition=None
            )
    finally:
        bank.flat[0] = original

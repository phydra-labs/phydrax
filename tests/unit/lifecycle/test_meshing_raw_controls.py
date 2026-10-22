# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit raw-owner dispatch; these fixtures do not assert scientific acceptance."""
from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._meshcore import (
    MeshcoreStatus,
    NativeExecutionEvidence,
    NativeExecutionMemoryCounters,
    NativeExecutionWorkCounters,
)
from phydrax._model._structure import model_array_catalog, model_recipe_array_values, model_structure_recipe
from phydrax.lifecycle import _meshing_sources as owner
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.geometry._source_fidelity import ParametricCurveBoundarySource
from tests._support.structured_sources import square
from tests.unit.lifecycle.test_meshing_polyhedral_fields import actual_epoch


def _receipt(plan: str, peak: int | None, history: tuple = ()):
    evidence = NativeExecutionEvidence(
        NativeExecutionWorkCounters(11, 7, 5, 3, 2, 1),
        NativeExecutionMemoryCounters(1024, 16, 32, 64, 4, 1),
        0.25, MeshcoreStatus.CAPACITY_EXCEEDED, 8, 6, 9, 10, 20, peak, 0.125,
    )
    return owner.NativeExecutionRecord(evidence, owner_id=plan, history_evidence=history)


class _Controls:
    """Real numerical controls with minimal, explicitly named owner-field tables."""

    def __init__(self, monkeypatch, *, peak=None, history=False):
        owner.register_meshing_source_artifacts()
        previous = _receipt("previous-plan", None)
        self.generation = _receipt("generation-plan", peak, (previous,) if history else ())
        self.accepted = _receipt("adaptation-plan", peak, (self.generation,))
        values = (self.generation, self.accepted, owner._contracts.MeshingLimits())
        recipe = model_structure_recipe(values)
        self.table, self.catalog = model_array_catalog(
            recipe, prefix="source-record", limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )
        self.nodes = self.table.nodes
        self.receipt, self.accepted_receipt, self.limits = self.nodes[-1]["items"]
        self.arrays = {
            name: np.asarray(value)
            for name, value in model_recipe_array_values(values, recipe, prefix="source-record").items()
        }
        monkeypatch.setattr(owner, "model_array_catalog", lambda *args, **kwargs: (self.table, self.catalog))
        self.none = self.literal(None)
        self.source_id = self.literal("source")
        self.revision = self.literal("revision")
        self.specification_id = self.literal("specification")
        self.source = self.dataclass(
            owner.NativePlanarSource, source_id=self.source_id, source_revision=self.revision,
            binding_id=self.literal("declared-binding"),
        )
        scope = self.dataclass(owner.MeshingScope, source_id=self.source_id, source_revision=self.revision)
        self.specification = self.dataclass(
            owner._contracts.SurfaceMeshingSpec, specification_id=self.specification_id,
            scope=scope, limits=self.limits,
        )
        self.options = self.dataclass(owner.NativeMeshingOptions)
        self.request = self.dataclass(owner.MeshCertificationInputs)
        self.report = self.dataclass(owner.MeshCertificationReport, request=self.request)
        self.provenance = self.literal(json.dumps({"kind": "mapping", "items": [
            ["plan", "generation-plan"], ["specification", "specification"],
        ]}))
        compliance = self.dataclass(owner._result.MeshingComplianceReport, specification_id=self.specification_id)
        binding = self.dataclass(owner._trace.MeshingEvidenceBinding, source_id=self.source_id, source_revision=self.revision)
        trace = self.dataclass(owner._trace.MeshingTrace, binding=binding)
        provenance = self.dataclass(owner.SemanticProvenance, content_json=self.provenance)
        self.carrier_fields = dict(
            execution_evidence=self.receipt, compliance=compliance, trace=trace,
            provenance=provenance, certification=self.report,
        )
        self.carrier = self.dataclass(owner._result.CellMeshingResult, **self.carrier_fields)
        self.part = self.dataclass(owner.MeshPart, name=self.literal("background"), carrier=self.carrier)
        self.roles = dict(
            certification_inputs=self.request, report=self.report, generation_source=self.source,
            generation_specification=self.specification, generation_options=self.options,
            generation_part=self.part,
        )

    def add(self, node):
        self.nodes.append(node)
        return len(self.nodes) - 1

    def literal(self, value):
        return self.add({"kind": "literal", "value": value})

    def dataclass(self, cls, **members):
        identifier = owner.registered_artifact_value_id(cls)
        assert identifier is not None
        return self.add({"kind": "dataclass", "type": identifier, "items": [
            members.get(field.name, self.none) for field in fields(cls)
        ]})

    def mapping(self, members):
        return self.add({"kind": "mapping", "items": [
            [self.literal(name), value] for name, value in members.items()
        ]})

    def select(self):
        # Negative tests may append a mutated child after the previously built root.
        self.root = self.add(self.nodes[self.root])
        return owner._select_execution_controls({}, DEFAULT_ARRAY_ARCHIVE_LIMITS)

    def restore(self):
        return owner._restore_execution_controls(self.select(), self.arrays)

    def singular(self):
        self.root = self.mapping(self.roles)

    def epochs(self):
        self.roles.pop("generation_part")
        final = self.dataclass(owner._result.CellMeshingResult,
            **(self.carrier_fields | {"execution_evidence": self.accepted_receipt,
                "provenance": self.dataclass(owner.SemanticProvenance,
                    content_json=self.literal("adapted-owner"))}))
        initial_epoch = self.dataclass(
            owner.MeshingAcceptedEpoch, carrier=self.carrier, transition=self.none,
        )
        final_epoch = self.dataclass(owner.MeshingAcceptedEpoch, carrier=final)
        epochs = self.add({"kind": "tuple", "items": [initial_epoch, final_epoch]})
        self.roles["accepted_data"] = self.mapping({"accepted_epochs": epochs})
        self.roles["accepted_target"] = final
        self.singular()

    def plural(self):
        foreign_carrier = self.dataclass(owner._result.CellMeshingResult)
        foreign_part = self.dataclass(owner.MeshPart, name=self.literal("alpha"), carrier=foreign_carrier)
        parts = self.add({"kind": "tuple", "items": [foreign_part, self.part]})
        assembly = self.dataclass(owner._assembly.MeshAssembly, parts=parts)
        registration = self.dataclass(owner._overset.OversetRegistration, assembly=assembly)
        self.plural_roles = dict(
            certification_inputs=self.request, report=self.report, registration=registration,
            primary_generation_part=self.literal("background"),
            generation_sources=self.mapping({"alpha": self.source, "background": self.source}),
            generation_specifications=self.mapping({"alpha": self.specification, "background": self.specification}),
            generation_options=self.mapping({"alpha": self.options, "background": self.options}),
        )
        self.root = self.mapping(self.plural_roles)
        return foreign_part

    def design(self):
        self.source = self.dataclass(owner.NativeImplicitSource, source_id=self.source_id, source_revision=self.revision)
        self.plan = self.dataclass(owner.NativeMeshingPlan, source=self.source,
            specification=self.specification, options=self.options, plan_id=self.literal("generation-plan"))
        self.target = self.dataclass(owner._result.CellMeshingResult,
            **(self.carrier_fields | {"execution_evidence": self.accepted_receipt}))
        adaptation = self.dataclass(owner._adaptation.MeshAdaptationResult, source=self.carrier, target=self.target)
        self.design_fields = dict(source=self.source, specification=self.specification,
            options=self.options, plan=self.plan, initial=self.carrier, accepted=self.target, adaptation=adaptation)
        self.root = self.dataclass(owner._decision.NativeDesignSourceState, **self.design_fields)


def _replace_field(graph, index, cls, name, value):
    position = tuple(field.name for field in fields(cls)).index(name)
    graph.nodes[index]["items"][position] = value


@pytest.mark.parametrize("peak", (None, 80), ids=("unobserved", "observed"))
@pytest.mark.parametrize("history", (False, True), ids=("empty-history", "receipt-history"))
def test_existing_raw_controls_preserve_managed_observation_and_history(monkeypatch, peak, history):
    graph = _Controls(monkeypatch, peak=peak, history=history)
    graph.singular()
    controls = graph.restore()
    owner._require_execution_control_binding(controls)
    assert controls.record.to_record() == graph.generation.to_record()
    assert controls.identity["source_binding_id"] == "declared-binding"


def test_plural_raw_controls_select_declared_background_not_first_part(monkeypatch):
    graph = _Controls(monkeypatch)
    graph.plural()
    controls = graph.restore()
    owner._require_execution_control_binding(controls)
    assert controls.record.receipt_hex() == graph.generation.receipt_hex()


@pytest.mark.parametrize("defect", ("missing-name", "duplicate-part", "missing-options", "mixed", "foreign-request", "duplicate-map"))
def test_plural_raw_controls_refuse_ambiguous_or_foreign_occurrences(monkeypatch, defect):
    graph = _Controls(monkeypatch)
    first = graph.plural()
    if defect == "missing-name":
        graph.nodes[graph.plural_roles["primary_generation_part"]]["value"] = "foreign"
    elif defect == "duplicate-part":
        _replace_field(graph, first, owner.MeshPart, "name", graph.literal("background"))
    elif defect == "missing-options":
        graph.nodes[graph.plural_roles["generation_options"]]["items"].pop()
    elif defect == "mixed":
        graph.plural_roles["generation_source"] = graph.source
        graph.root = graph.mapping(graph.plural_roles)
    elif defect == "foreign-request":
        graph.plural_roles["certification_inputs"] = graph.dataclass(
            owner.MeshCertificationInputs, source_id=graph.literal("foreign")
        )
        graph.root = graph.mapping(graph.plural_roles)
    else:
        entries = graph.nodes[graph.plural_roles["generation_sources"]]["items"]
        entries.append(entries[-1])
    with pytest.raises(ArrayArchiveCorruptionError):
        graph.restore()


def test_compartment_raw_binding_fingerprints_whole_retained_source(monkeypatch):
    graph = _Controls(monkeypatch)
    interpretation = graph.literal("occupied-voxel-cells")
    graph.source = graph.dataclass(owner.CompartmentMeshingSource, source_id=graph.source_id,
        source_revision=graph.revision, interpretation=interpretation)
    graph.roles["generation_source"] = graph.source
    graph.singular()
    first = graph.restore()
    owner._require_execution_control_binding(first)
    graph.nodes[interpretation]["value"] = "changed-retained-container"
    second = graph.restore()
    assert first.identity["source_binding_id"] != second.identity["source_binding_id"]
    assert first.identity["source_id"] == second.identity["source_id"]


def test_design_standalone_returns_actual_accepted_receipt_binds_original_generation(monkeypatch):
    graph = _Controls(monkeypatch)
    graph.design()
    controls = graph.restore()
    owner._require_execution_control_binding(controls)
    assert controls.record.to_record() == graph.accepted.to_record()
    assert controls.generation_record.to_record() == graph.generation.to_record()
    assert controls.record.owner_id != controls.generation_record.owner_id
    assert controls.identity["source_binding_id"]


def test_design_raw_controls_allow_genuine_unchanged_accepted_generation(monkeypatch):
    graph = _Controls(monkeypatch)
    graph.design()
    graph.design_fields["accepted"] = graph.carrier
    graph.root = graph.dataclass(owner._decision.NativeDesignSourceState, **graph.design_fields)
    controls = graph.restore()
    owner._require_execution_control_binding(controls)
    assert controls.record.receipt_hex() == graph.generation.receipt_hex()


@pytest.fixture(scope="module")
def structured_generation_records():
    """Publish a real transfinite owner, not a hand-authored control table."""
    block = square(2)
    controls = block.boundaries
    revision = canonical_fingerprint(
        {"curves": tuple(array_tree_fingerprint(control.curve) for control in controls)}
    )
    query = ParametricCurveBoundarySource(
        tuple(control.curve for control in controls),
        tuple(control.parameter_range for control in controls),
        source_id="square", source_revision=revision,
    )
    source = phx.meshing.NativeStructuredSource(
        (block,), (), "square", revision, fidelity_source=query, maximum_deviation=0.08,
    )
    scope = phx.meshing.MeshingScope(
        source.source_id, revision, phx.meshing.MeshingEntityKind.GEOMETRY,
        2, "square-domain", np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("quadrilateral",)),
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        ),
        size_controls=(phx.meshing.UniformSizeControl(
            scope, 0.5, strength=phx.meshing.SizeControlStrength.SOFT,
        ),),
    )
    options = phx.meshing.NativeMeshingOptions("structured_transfinite")
    provider = phx.meshing.NativeMeshingProvider(options)
    plan = provider.plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    carrier = plan.execute()
    assert carrier.certification is not None
    return plan, {
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "generation_part": phx.meshing.MeshPart("square", carrier),
        "certification_inputs": carrier.certification.request,
        "report": carrier.certification,
    }


def test_actual_structured_generation_raw_controls_bind_published_plan_and_specification(
    structured_generation_records, tmp_path,
):
    plan, records = structured_generation_records
    carrier = records["generation_part"].carrier
    provenance = dict(json.loads(carrier.provenance.content_json)["items"])
    assert provenance["plan"] == plan.plan_id == carrier.execution_evidence.owner_id
    assert provenance["specification"] == plan.specification.specification_id
    receipt = owner.write_meshing_source_closure(tmp_path / "structured.phx", records)
    controls = owner.read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id,
    )
    assert controls.execution_evidence.to_record() == carrier.execution_evidence.to_record()
    assert controls.specification_id == plan.specification.specification_id
    assert controls.limits.limits_id == plan.specification.limits.limits_id


@pytest.mark.parametrize("defect", ("foreign-plan", "foreign-specification"))
def test_actual_structured_raw_controls_refuse_foreign_generation_join(
    structured_generation_records, tmp_path, defect,
):
    plan, original = structured_generation_records
    records = dict(original)
    foreign_specification = phx.meshing.SurfaceMeshingSpec(
        plan.specification.target, plan.specification.scope,
        planar_embedding=plan.specification.planar_embedding,
        limits=plan.specification.limits,
        size_controls=(phx.meshing.UniformSizeControl(
            plan.specification.scope, 0.25,
            strength=phx.meshing.SizeControlStrength.SOFT,
        ),),
    )
    foreign_plan = phx.meshing.NativeMeshingProvider(plan.options).plan(
        plan.source, foreign_specification, coordinate_contract=plan.coordinate_contract,
    )
    assert foreign_plan.plan_id != plan.plan_id
    if defect == "foreign-specification":
        records["generation_specification"] = foreign_specification
    else:
        carrier = records["generation_part"].carrier
        foreign_carrier = foreign_plan.execute()
        assert foreign_carrier.execution_evidence is not None
        foreign_carrier = carrier.with_execution_evidence(foreign_carrier.execution_evidence)
        records["generation_part"] = phx.meshing.MeshPart("square", foreign_carrier)
    receipt = owner.write_meshing_source_closure(tmp_path / "foreign.phx", records)
    with pytest.raises(ArrayArchiveCorruptionError, match="another plan/specification"):
        owner.read_meshing_source_execution_controls(
            receipt.path, expected_content_id=receipt.content_id,
        )


def test_actual_epoch_controls_preserve_original_receipt_and_accepted_history(
    actual_epoch, tmp_path,
):
    epoch, records = actual_epoch
    assert isinstance(epoch, owner.MeshingAcceptedEpoch)
    generation = epoch.carrier.execution_evidence
    assert isinstance(generation, owner.NativeExecutionRecord)
    assert generation.history_evidence
    receipt = owner.write_meshing_source_closure(tmp_path / "epochs.phx", records)
    controls = owner.read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id,
    )
    assert controls.execution_evidence.to_record() == generation.to_record()
    assert tuple(record.receipt_hex() for record in controls.execution_evidence.history_evidence) == (
        tuple(record.receipt_hex() for record in generation.history_evidence)
    )


def test_accepted_epoch_controls_use_original_generation_not_final_adapted_provenance(monkeypatch):
    graph = _Controls(monkeypatch)
    graph.epochs()
    controls = graph.restore()
    owner._require_execution_control_binding(controls)
    assert controls.generation_record.receipt_hex() == graph.generation.receipt_hex()
    assert controls.record.receipt_hex() == graph.generation.receipt_hex()
    assert controls.record.receipt_hex() != graph.accepted.receipt_hex()


@pytest.mark.parametrize("defect", ("foreign-plan", "foreign-target", "foreign-history"))
def test_design_raw_controls_refuse_foreign_lineage(monkeypatch, defect):
    graph = _Controls(monkeypatch)
    graph.design()
    if defect == "foreign-plan":
        _replace_field(graph, graph.plan, owner.NativeMeshingPlan, "options", graph.none)
    elif defect == "foreign-target":
        graph.design_fields["accepted"] = graph.dataclass(owner._result.CellMeshingResult)
        graph.root = graph.dataclass(owner._decision.NativeDesignSourceState, **graph.design_fields)
    else:
        members = tuple(field.name for field in fields(owner.NativeExecutionRecord))
        history = graph.nodes[graph.accepted_receipt]["items"][members.index("history_evidence")]
        graph.nodes[history] = {"kind": "literal_tuple", "values": []}
    with pytest.raises((ArrayArchiveCorruptionError, ValueError)):
        graph.restore()


@pytest.mark.parametrize("defect", ("nonoptional-none", "nonempty-literal-history", "foreign-history-owner"))
def test_raw_receipt_nullable_and_history_guards_remain_strict(monkeypatch, defect):
    graph = _Controls(monkeypatch)
    graph.singular()
    if defect == "nonoptional-none":
        _replace_field(graph, graph.receipt, owner.NativeExecutionRecord, "elapsed_seconds", graph.none)
    else:
        bad = graph.add({"kind": "literal_tuple", "values": [1]}) if defect == "nonempty-literal-history" else graph.add({"kind": "tuple", "items": [graph.options]})
        _replace_field(graph, graph.receipt, owner.NativeExecutionRecord, "history_evidence", bad)
    with pytest.raises(ArrayArchiveCorruptionError):
        graph.restore()


@pytest.mark.parametrize("defect", ("foreign-plan", "foreign-specification", "missing-specification", "duplicate-plan"))
@pytest.mark.parametrize("schema", ("singular", "epochs"))
def test_generation_provenance_join_remains_exact(monkeypatch, defect, schema):
    graph = _Controls(monkeypatch)
    getattr(graph, schema)()
    items = [["plan", "generation-plan"], ["specification", "specification"]]
    if defect == "foreign-plan":
        items[0][1] = "unrelated-plan"
    elif defect == "foreign-specification":
        items[1][1] = "unrelated-specification"
    elif defect == "missing-specification":
        items.pop()
    else:
        items.append(["plan", "generation-plan"])
    graph.nodes[graph.provenance]["value"] = json.dumps({"kind": "mapping", "items": items})
    controls = graph.restore()
    with pytest.raises(ArrayArchiveCorruptionError):
        owner._require_execution_control_binding(controls)


@pytest.mark.parametrize("schema", ("compartment", "design"))
def test_raw_container_identity_does_not_construct_scientific_owners(monkeypatch, schema):
    graph = _Controls(monkeypatch)
    if schema == "compartment":
        graph.roles["generation_source"] = graph.dataclass(
            owner.CompartmentMeshingSource, source_id=graph.source_id, source_revision=graph.revision,
        )
        graph.singular()
    else:
        graph.design()

    def forbidden(*args, **kwargs):
        raise AssertionError("raw controls must not construct scientific owners")

    for cls in (owner.CompartmentMeshingSource, owner.NativeImplicitSource,
                owner.NativeMeshingPlan, owner._decision.NativeDesignSourceState):
        monkeypatch.setattr(cls, "__init__", forbidden)
    controls = graph.restore()
    owner._require_execution_control_binding(controls)
    assert controls.identity["source_binding_id"]


def test_raw_container_fingerprint_includes_actual_array_bank_content(monkeypatch):
    graph = _Controls(monkeypatch)
    entry, index = graph.catalog[0]
    graph.roles["generation_source"] = graph.dataclass(
        owner.CompartmentMeshingSource, source_id=graph.source_id,
        source_revision=graph.revision, labels=index,
    )
    graph.singular()
    selection = graph.select()
    source_index = selection.source_container_index
    assert source_index is not None
    before = owner._execution_control_fingerprints(selection, graph.arrays, (source_index,))
    graph.arrays[entry.name] = graph.arrays[entry.name] + np.ones_like(graph.arrays[entry.name])
    after = owner._execution_control_fingerprints(selection, graph.arrays, (source_index,))
    assert before[source_index] != after[source_index]

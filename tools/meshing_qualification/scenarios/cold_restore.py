from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


_BLOCK_ENGINE_IMPORTS = r"""
import importlib.abc, sys
blocked = ("OCP", "gmsh", "manifold3d", "open3d", "pyvista", "pymeshlab", "vtk")
attempted = []
class _RejectEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            attempted.append(fullname)
            raise ModuleNotFoundError("optional geometry engine import rejected: " + fullname)
        return None
sys.meta_path.insert(0, _RejectEngine())

def _assert_engine_free():
    loaded = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in blocked))
    if attempted or loaded:
        raise RuntimeError("optional geometry engine import was attempted or loaded: " + repr({"attempted": attempted, "loaded": loaded}))
"""


def _run_child(code: str, /, *, environment: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cold-process qualification child failed:\n" + result.stdout + result.stderr
        )
    try:
        record = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Cold-process child returned no canonical JSON record."
        ) from error
    if not isinstance(record, dict):
        raise RuntimeError("Cold-process child record must be an object.")
    return record


def native_cold_restore() -> dict[str, object]:
    environment = dict(os.environ)
    with TemporaryDirectory(prefix="phydrax-native-cold-composition-") as temporary:
        root = Path(temporary)
        source_archive = root / "native-brep-source.phx"
        composition_archive = root / "accepted-state-composition.phx"
        writer = (
            _BLOCK_ENGINE_IMPORTS
            + f"""
import json, os, numpy as np, jax.numpy as jnp
import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.geometry.brep._constructors import brep_box
from phydrax.interchange._cad_archive import save_brep_archive
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._contracts import CellFamilyPolicy, CellMeshingTarget, VolumeFillStrategy, VolumeMeshingSpec
from phydrax.meshing._model_codec import write_model_artifact
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.discretization.fem import FiniteElementFieldSpec, lagrange_element
from phydrax.solver._finite_element_schedule import FiniteElementAcceptedFieldLayout, FiniteElementAcceptedState, FiniteElementFieldTransferSemantics
contract = phx.SpatialCoordinateContract.si()
model = brep_box(np.asarray((0.0, 0.0, 0.0)), np.asarray((1.0, 1.0, 1.0)), coordinate_contract=contract, source_id="qualification:cold-brep-source")
save_brep_archive(model, {str(source_archive)!r})
boundary = MeshingScope.whole(model, 2)
specification = VolumeMeshingSpec(CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))), boundary, VolumeFillStrategy.SIMPLEX, size_controls=(UniformSizeControl(boundary, 0.5),), maximum_deviation=0.02)
plan = NativeMeshingProvider().plan(model, specification, coordinate_contract=contract)
result = plan.execute()
result.audit.require_passed()
if result.certification is None:
    raise RuntimeError("Cold producer omitted computed mesh certification.")
result.certification.require_passed()
if not result.compliance.passed or not result.region_boundary_evidence:
    raise RuntimeError("Cold producer omitted accepted compliance/source-boundary evidence.")
volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
if not np.isclose(volume, 1.0, rtol=0.0, atol=1.0e-10):
    raise RuntimeError("Produced tetrahedra do not integrate to exact unit-box volume.")
field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
space = phx.discretization.FiniteElementPlan(result.mesh, field, coordinate_spec=result.geometry).prepare()
constraint = phx.discretization.dirichlet_constraint(space, "u")
form = phx.equations.FiniteElementForm("native-cold-unit-box-harmonic", "u", (phx.equations.DiffusionAction("u", 1.0),))
problem = phx.equations.compile_finite_element_problem(form, space, constraint=constraint, dirichlet_values=lambda points: 1.0 + 2.0 * points[..., 0] - points[..., 1] + 0.5 * points[..., 2])
operator, rhs = problem.linear_system()
solved = phx.linalg.solve(operator, rhs)
field_values = jnp.asarray(problem.expand(solved.value))
expected_values = 1.0 + 2.0 * np.asarray(result.mesh.coordinates)[:, 0] - np.asarray(result.mesh.coordinates)[:, 1] + 0.5 * np.asarray(result.mesh.coordinates)[:, 2]
writer_fe_error = float(np.max(np.abs(np.asarray(field_values) - expected_values), initial=0.0))
writer_fe_residual = float(jnp.linalg.norm(problem.residual(solved.value)))
writer_fe_rhs_norm = float(jnp.linalg.norm(rhs))
if not bool(jnp.all(solved.successful)) or writer_fe_error > 2.0e-9 or writer_fe_residual > 2.0e-9 * max(1.0, writer_fe_rhs_norm):
    raise RuntimeError("Cold producer FE solve failed its independent affine/residual oracle.")
field_layout = FiniteElementAcceptedFieldLayout(FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1)), transfer_semantics=FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION)
accepted_state = FiniteElementAcceptedState((field_values,), np.asarray(0.0, dtype=np.float64), 0, result.mesh.topology_id, result.result_id, "qualification:linear-field-compilation", field_layouts=(field_layout,), state_version=1)
source_entry = CompositionEntry(model, entry_id="cad/source", role="topology", owner_id="phydrax.geometry.brep", structure_id=model.geometry.geometry_id, revision_id=model.source_revision, semantics_id=model.source_id)
mesh_structure = canonical_fingerprint({{"topology": result.mesh.topology_id, "geometry_layout": result.geometry.geometry_layout_id}})
mesh_entry = CompositionEntry(result, entry_id="mesh/accepted-result", role="discretization", owner_id="phydrax.meshing", structure_id=mesh_structure, revision_id=result.result_id, semantics_id=model.source_id, dependencies=(source_entry.binding("revision"),))
field_entry = CompositionEntry(accepted_state, entry_id="solver/state", role="physical-state", owner_id="phydrax.solver", structure_id=mesh_structure, revision_id="field-state-1", semantics_id="linear-scalar-u", dependencies=(mesh_entry.binding("structure"),))
composition = Composition((source_entry, mesh_entry, field_entry), boundary_id="native-cold-accepted-state")
write_model_artifact({str(composition_archive)!r}, composition, limits=None)
_assert_engine_free()
print(json.dumps({{"pid": os.getpid(), "source_id": model.source_id, "source_revision": model.source_revision, "geometry_id": model.geometry.geometry_id, "result_id": result.result_id, "topology_id": result.mesh.topology_id, "composition_id": composition.composition_id, "composition_structure_id": composition.structure_id, "field_size": int(field_values.shape[0]), "provider": result.provider.name, "actual_version": result.runtime.actual_version, "volume": volume, "certification_id": result.certification.report_id, "fe_nodal_error": writer_fe_error, "fe_residual": writer_fe_residual}}))
"""
        )
        writer_record = _run_child(writer, environment=environment)
        for path in (source_archive, composition_archive):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"Cold producer did not publish its canonical artifact: {path.name}."
                )
        artifact_bytes = {
            path.name: path.stat().st_size
            for path in (source_archive, composition_archive)
        }
        consumer = (
            _BLOCK_ENGINE_IMPORTS
            + f"""
import json, os, numpy as np, jax.numpy as jnp
import phydrax as phx
from phydrax.discretization import FiniteElementFieldSpec, FiniteElementPlan, dirichlet_constraint, lagrange_element
from phydrax.geometry.brep._model import BRepModel
from phydrax.interchange._cad_archive import load_brep_archive
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._adaptation import MarkedMeshAdaptation, MeshAdaptationPolicy, MeshAdaptationRoute, execute_mesh_adaptation, prepare_mesh_adaptation
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._model_codec import read_model_artifact
from phydrax.meshing._organization import RegionBoundaryEvidence
from phydrax.meshing._result import CellMeshingResult
from phydrax._model._structure import FiniteElementAcceptedStateArchiveRecipe
from phydrax.solver._finite_element_schedule import FiniteElementAcceptedFieldLayout, FiniteElementAcceptedState
def _rebuild_result(raw):
    if not isinstance(raw, CellMeshingResult) or raw.certification is None:
        raise TypeError("Cold composition lost its accepted CellMeshingResult.")
    mesh = raw.mesh
    region_boundaries = tuple(
        RegionBoundaryEvidence(
            mesh,
            item.source_scope,
            item.source_region_id,
            item.control_id,
            item.material_id,
            item.role,
            item.boundary_label,
            item.patch_sides,
            source_occurrence_path=item.source_occurrence_path,
        )
        for item in raw.region_boundary_evidence
    )
    recomputed = certify_cell_mesh(mesh, raw.coordinate_contract, geometry=raw.geometry, audit_policy=raw.certification.audit_policy, boundary=raw.boundary, certification_inputs=raw.certification.inputs, patches=raw.patches, zones=raw.zones, labels=raw.labels, attributes=raw.attributes, associations=raw.associations, surface_source=raw.surface_source, region_boundary_evidence=region_boundaries, planar_embedding=raw.planar_embedding)
    if recomputed.certification is None:
        raise RuntimeError("Cold result revalidation dropped computed certification.")
    recomputed.audit.require_passed()
    recomputed.certification.require_passed()
    recomputed.certification.revalidate(mesh, raw.geometry)
    if recomputed.audit.report_id != raw.audit.report_id or recomputed.certification.report_id != raw.certification.report_id:
        raise RuntimeError("Cold result reconstruction changed its computed audit/certification identity.")
    rebuilt = CellMeshingResult(mesh, raw.geometry, raw.coordinate_contract, recomputed.audit, raw.quality, raw.compliance, raw.trace, raw.provider, raw.runtime, raw.derivative_mode, raw.provenance, boundary=raw.boundary, patches=raw.patches, zones=raw.zones, labels=raw.labels, attributes=raw.attributes, associations=raw.associations, adapter_reports=raw.adapter_reports, surface_source=raw.surface_source, region_boundary_evidence=region_boundaries, planar_embedding=raw.planar_embedding, certification=recomputed.certification)
    if rebuilt.result_id != raw.result_id:
        raise RuntimeError("Cold result reconstruction changed accepted result identity.")
    return rebuilt
def _restore_composition(raw):
    source = load_brep_archive({str(source_archive)!r})
    entries = []
    for entry in raw.entries:
        value = entry.value
        if entry.entry_id == "cad/source":
            if not isinstance(value, BRepModel) or (value.source_id, value.source_revision, value.geometry.geometry_id) != (source.source_id, source.source_revision, source.geometry.geometry_id):
                raise RuntimeError("Cold source archive differs from the composition source root.")
            value = source
        elif entry.entry_id == "mesh/accepted-result":
            value = _rebuild_result(value)
        elif entry.entry_id == "solver/state":
            if not isinstance(value, FiniteElementAcceptedStateArchiveRecipe):
                raise TypeError("Cold composition did not store the registered FE state recipe.")
            layouts = tuple(FiniteElementAcceptedFieldLayout.from_recipe(item) for item in value.field_layouts)
            state = FiniteElementAcceptedState(value.fields, value.time, value.step, value.topology_id, value.prepared_id, value.compilation_id, field_layouts=layouts, materials=value.materials, schedule_cursor=value.schedule_cursor, state_version=value.state_version)
            if state.accepted_id != value.expected_accepted_id:
                raise RuntimeError("Cold FE state accepted identity changed during layout reconstruction.")
            value = state
        entries.append(CompositionEntry(value, entry_id=entry.entry_id, role=entry.role, owner_id=entry.owner_id, structure_id=entry.structure_id, revision_id=entry.revision_id, semantics_id=entry.semantics_id, dependencies=entry.dependencies))
    restored = Composition(entries, boundary_id=raw.boundary_id)
    if restored.composition_id != raw.composition_id or restored.structure_id != raw.structure_id:
        raise RuntimeError("Cold composition rebuild changed its accepted boundary identities.")
    return restored
source = load_brep_archive({str(source_archive)!r})
composition = read_model_artifact({str(composition_archive)!r}, expected_type=Composition, restore=_restore_composition, limits=None)
restored_model = composition.value("cad/source")
restored = composition.value("mesh/accepted-result")
accepted_state = composition.value("solver/state")
if not isinstance(accepted_state, FiniteElementAcceptedState) or accepted_state.topology_id != restored.mesh.topology_id:
    raise RuntimeError("Cold FE state is not bound to the restored mesh topology.")
source_values = np.asarray(accepted_state.fields[0], dtype=np.float64)
if (restored_model.source_id, restored_model.source_revision) != (source.source_id, source.source_revision):
    raise RuntimeError("Cold composition source identity differs from its BRep archive.")
restored.audit.require_passed()
if restored.certification is None:
    raise RuntimeError("Cold composition lost computed mesh certification.")
restored.certification.require_passed()
if source_values.shape != (restored.mesh.coordinates.shape[0],):
    raise RuntimeError("Cold physical field layout no longer matches the accepted mesh.")
def _linear(points):
    return 1.0 + 2.0 * points[:, 0] - points[:, 1] + 0.5 * points[:, 2]
if not np.allclose(source_values, _linear(np.asarray(restored.mesh.coordinates, dtype=np.float64)), rtol=0.0, atol=2.0e-9):
    raise RuntimeError("Cold physical field payload differs from its declared exact linear semantics.")
def _solve_affine(result):
    field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
    space = FiniteElementPlan(result.mesh, field, coordinate_spec=result.geometry).prepare()
    constraint = dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm("native-cold-unit-box-harmonic", "u", (phx.equations.DiffusionAction("u", 1.0),))
    problem = phx.equations.compile_finite_element_problem(form, space, constraint=constraint, dirichlet_values=lambda points: _linear(points))
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    expected = _linear(coordinates)
    error = float(np.max(np.abs(values - expected), initial=0.0))
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    rhs_norm = float(jnp.linalg.norm(rhs))
    if not bool(jnp.all(solved.successful)) or error > 2.0e-9 or residual > 2.0e-9 * max(1.0, rhs_norm):
        raise RuntimeError("Cold-process FE reanalysis failed its independent affine/residual oracle.")
    return values, error, residual
restored_solution, restored_fe_error, restored_fe_residual = _solve_affine(restored)
if not np.allclose(restored_solution, source_values, rtol=0.0, atol=2.0e-9):
    raise RuntimeError("Cold FE continuation changed the archived accepted solution.")
if restored_fe_error > 2.0e-9 or restored_fe_residual > 2.0e-9:
    raise RuntimeError("Cold restored FE solve failed its exact physical oracle.")
prepared = prepare_mesh_adaptation(restored, MarkedMeshAdaptation(np.asarray(restored.mesh.blocks[0].global_ids[:1], dtype=np.int64)), policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION))
adaptation = execute_mesh_adaptation(prepared)
target = adaptation.target
transition = adaptation.transition
if transition.vertex_stencil is None:
    raise RuntimeError("Cold-restored adaptation omitted the physical field transfer stencil.")
transferred = np.asarray(transition.vertex_stencil.apply(np.asarray(restored.mesh.vertex_global_ids, dtype=np.int64), source_values), dtype=np.float64)
target_values = _linear(np.asarray(target.mesh.coordinates, dtype=np.float64))
linear_error = float(np.max(np.abs(transferred - target_values)))
def _integral(mesh, measures, values):
    cells = np.concatenate([np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks], axis=0)
    return float(np.sum(np.asarray(measures, dtype=np.float64) * np.mean(values[cells], axis=1)))
source_integral = _integral(restored.mesh, restored.quality.evaluation.measures, source_values)
target_integral = _integral(target.mesh, target.quality.evaluation.measures, transferred)
integral_error = abs(target_integral - source_integral)
if linear_error > 1.0e-12 or integral_error > 1.0e-12:
    raise RuntimeError(f"Cold continuation failed exact linear/conservative transfer: {{linear_error=}}, {{integral_error=}}.")
if not target.audit.passed or target.certification is None or not target.certification.passed:
    raise RuntimeError("Cold-restored adaptation target did not remain accepted.")
continued_solution, continued_fe_error, continued_fe_residual = _solve_affine(target)
continued_solution_error = float(np.max(np.abs(continued_solution - transferred), initial=0.0))
if continued_fe_error > 2.0e-9 or continued_fe_residual > 2.0e-9 or continued_solution_error > 2.0e-9:
    raise RuntimeError("Cold adapted FE solve failed its independent transferred-state/affine oracle.")
_assert_engine_free()
print(json.dumps({{"pid": os.getpid(), "source_id": restored_model.source_id, "source_revision": restored_model.source_revision, "geometry_id": restored_model.geometry.geometry_id, "result_id": restored.result_id, "topology_id": restored.mesh.topology_id, "composition_id": composition.composition_id, "provider": restored.provider.name, "actual_version": restored.runtime.actual_version, "volume": float(np.sum(np.asarray(restored.quality.evaluation.measures))), "certification_id": restored.certification.report_id, "field_dofs": int(source_values.size), "adapted_result_id": target.result_id, "linear_transfer_error": linear_error, "conservative_integral_error": integral_error, "restored_fe_error": restored_fe_error, "restored_fe_residual": restored_fe_residual, "continued_fe_error": continued_fe_error, "continued_fe_residual": continued_fe_residual, "continued_solution_transfer_error": continued_solution_error}}))
"""
        )
        consumer_record = _run_child(consumer, environment=environment)
    if writer_record["pid"] == consumer_record["pid"]:
        raise RuntimeError(
            "Cold qualification reused the producer process instead of restoring in a fresh process."
        )
    if any(
        writer_record[key] != consumer_record[key]
        for key in (
            "source_id",
            "source_revision",
            "geometry_id",
            "result_id",
            "topology_id",
            "composition_id",
        )
    ):
        raise RuntimeError(
            "Cold composition reconstruction changed source, result, topology, or boundary identity."
        )
    if (
        writer_record["provider"] != "phydrax-native"
        or consumer_record["provider"] != "phydrax-native"
    ):
        raise RuntimeError("Cold workflow executed through a non-native provider.")
    fe_tolerance = 2.0e-9
    for record in (writer_record, consumer_record):
        for key in (
            "fe_nodal_error",
            "fe_residual",
            "restored_fe_error",
            "restored_fe_residual",
            "continued_fe_error",
            "continued_fe_residual",
            "continued_solution_transfer_error",
        ):
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"Cold FE error field {key!r} is not numeric.")
            if float(value) > fe_tolerance:
                raise RuntimeError(
                    "Cold FE producer/reader reanalysis exceeded its independent field/residual tolerance."
                )
    return {
        "status": "passed",
        "producer_process_exited_before_consumer": True,
        "fresh_process": writer_record["pid"] != consumer_record["pid"],
        "logical_archive_bytes": artifact_bytes,
        "fresh_fe_pde_solve": {
            "writer_nodal_error": writer_record["fe_nodal_error"],
            "writer_residual": writer_record["fe_residual"],
            "reader_restored_nodal_error": consumer_record["restored_fe_error"],
            "reader_restored_residual": consumer_record["restored_fe_residual"],
            "reader_continued_nodal_error": consumer_record["continued_fe_error"],
            "reader_continued_residual": consumer_record["continued_fe_residual"],
            "continued_solution_transfer_error": consumer_record[
                "continued_solution_transfer_error"
            ],
            "tolerance": fe_tolerance,
        },
        "writer": writer_record,
        "consumer": consumer_record,
    }

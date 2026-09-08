#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from dataclasses import replace

import h5py
import jax
import numpy as np
import pytest

from phydrax.applications import electrophysiology as ep
from phydrax.interchange import AdapterError, require_lossless, ResourceReadError


jax.config.update("jax_enable_x64", True)


def _fixture(root, *, delay=0.25):
    (root / "lif.json").write_text(
        json.dumps(
            {
                "capacitance_nF": 0.25,
                "leak_conductance_uS": 0.125,
                "resting_mV": -65.0,
                "threshold_mV": -50.0,
                "reset_mV": -60.0,
            }
        )
    )
    (root / "current.json").write_text(
        json.dumps({"time_constant_ms": 2.0, "current_scale_nA": -0.5})
    )
    (root / "node_types.csv").write_text(
        "node_type_id model_type model_template dynamics_params label\n"
        '2 point_neuron phydrax:LeakyIntegrateAndFire lif.json "type default"\n'
        "3 virtual NULL NULL input\n"
    )
    (root / "edge_types.csv").write_text(
        "edge_type_id model_template dynamics_params delay syn_weight\n"
        f"8 phydrax:CurrentSynapse current.json {delay} 1.0\n"
    )
    with h5py.File(root / "nodes.h5", "w") as handle:
        target = handle.create_group("nodes/cells")
        target["node_id"] = np.asarray([2**63 + 7, 7], dtype=np.uint64)
        target["node_type_id"] = [2, 2]
        target["node_group_id"] = [5, 5]
        target["node_group_index"] = [1, 0]
        group = target.create_group("5")
        group["label"] = np.asarray([b"first", b"second"])
        dynamics = group.create_group("dynamics_params")
        dynamics["threshold_mV"] = [-48.0, -45.0]
        source = handle.create_group("nodes/input")
        source["node_id"] = np.asarray([7], dtype=np.uint64)
        source["node_type_id"] = [3]
        source["node_group_id"] = [0]
        source["node_group_index"] = [0]
        source.create_group("0")
    with h5py.File(root / "edges.h5", "w") as handle:
        edges = handle.create_group("edges/input_to_cells")
        edges["edge_id"] = np.asarray([19, 20], dtype=np.uint64)
        edges["edge_type_id"] = [8, 8]
        edges["edge_group_id"] = [4, 4]
        edges["edge_group_index"] = [1, 0]
        edges["source_node_id"] = np.asarray([7, 7], dtype=np.uint64)
        edges["target_node_id"] = np.asarray([7, 7], dtype=np.uint64)
        edges["source_node_id"].attrs["node_population"] = "input"
        edges["target_node_id"].attrs["node_population"] = "cells"
        group = edges.create_group("4")
        group["syn_weight"] = [3.0, 2.0]
    with h5py.File(root / "spikes.h5", "w") as handle:
        group = handle.create_group("spikes/input")
        group.attrs.create(
            "sorting",
            1,
            dtype=h5py.enum_dtype({"none": 0, "by_id": 1, "by_time": 2}, basetype="u1"),
        )
        group["node_ids"] = np.asarray([7, 7], dtype=np.uint64)
        group["timestamps"] = [0.001, 0.003]
        group["timestamps"].attrs["units"] = "s"
    return dict(
        node_files=(ep.SONATAFilePair("nodes.h5", "node_types.csv"),),
        edge_files=(ep.SONATAFilePair("edges.h5", "edge_types.csv"),),
        spike_files=("spikes.h5",),
        trusted_root=root,
        components={"lif.json": "lif.json", "current.json": "current.json"},
    )


def test_group_type_precedence_population_keys_multiedges_and_semantic_roundtrip(
    tmp_path,
):
    circuit = ep.import_sonata(**_fixture(tmp_path))
    runtime = ep.prepare_sonata_network(
        circuit,
        queue_capacity=4,
        spike_capacity=16,
        recording_capacity=40,
        maximum_events_per_step=16,
        root_subdivisions=1,
    )
    execution = ep.run_neural_network(runtime, ep.initialize_neural_network(runtime), 40)
    np.testing.assert_array_equal(execution.status, np.zeros(40, dtype=np.int32))
    assert int(np.sum(np.asarray(execution.delivered_messages))) == 4
    lookup = {node.key: node for node in circuit.nodes}
    assert float(lookup["cells", 7].model.threshold_mV) == -48.0
    assert float(lookup["cells", 2**63 + 7].model.threshold_mV) == -45.0
    assert dict(lookup["cells", 7].properties)["label"] == "first"
    assert lookup["input", 7].model is None
    assert [edge.connection.weight for edge in circuit.edges] == [2.0, 3.0]
    assert circuit.edges[0].source == circuit.edges[1].source == ("input", 7)
    assert circuit.edges[0].target == circuit.edges[1].target == ("cells", 7)
    assert (
        circuit.edges[0].connection.relation_id != circuit.edges[1].connection.relation_id
    )
    np.testing.assert_array_equal(circuit.spikes[0].timestamps_ms, [1.0, 3.0])
    exported = ep.export_sonata(circuit, tmp_path / "export")
    reimported = ep.import_sonata(
        node_files=exported.node_files,
        edge_files=exported.edge_files,
        spike_files=exported.spike_files,
        components=dict(exported.components),
        trusted_root=tmp_path / "export",
    )
    require_lossless(ep.sonata_roundtrip_report(circuit, reimported))
    altered = replace(
        reimported,
        spikes=(
            ep.SONATASpikes("input", np.array([7], dtype=np.uint64), np.array([2.0])),
        ),
    )
    with pytest.raises(AdapterError):
        ep.sonata_roundtrip_report(circuit, altered)


def test_missing_component_and_foreign_mechanism_fail_without_execution(tmp_path):
    arguments = _fixture(tmp_path)
    with pytest.raises(AdapterError, match="component resource binding"):
        ep.import_sonata(**(arguments | {"components": {}}))
    (tmp_path / "node_types.csv").write_text(
        "node_type_id model_type model_template dynamics_params\n"
        "2 point_neuron hoc:dangerous.hoc lif.json\n3 virtual NULL NULL\n"
    )
    with pytest.raises(AdapterError, match="never executed"):
        ep.import_sonata(**arguments)
    assert not (tmp_path / "dangerous.hoc").exists()


def test_group_index_and_population_binding_fail_before_wrong_connectivity(tmp_path):
    arguments = _fixture(tmp_path)
    with h5py.File(tmp_path / "nodes.h5", "r+") as handle:
        handle["nodes/cells/node_group_index"][0] = 2
    with pytest.raises(ValueError, match="group index"):
        ep.import_sonata(**arguments)
    with h5py.File(tmp_path / "nodes.h5", "r+") as handle:
        handle["nodes/cells/node_group_index"][0] = 1
    with h5py.File(tmp_path / "edges.h5", "r+") as handle:
        handle["edges/input_to_cells/source_node_id"].attrs["node_population"] = "missing"
    with pytest.raises(ValueError, match="population-qualified"):
        ep.import_sonata(**arguments)


def test_clock_rejects_physical_offgrid_delay_and_spike_time(tmp_path):
    arguments = _fixture(tmp_path, delay=0.3)
    event = ep.import_sonata(**arguments, dt_ms=0.25, execution="event")
    assert event.edges[0].connection.delay_ms == 0.3
    with pytest.raises(AdapterError, match="off-grid"):
        ep.import_sonata(**arguments, dt_ms=0.25, execution="clock")
    (tmp_path / "edge_types.csv").write_text(
        "edge_type_id model_template dynamics_params delay syn_weight\n"
        "8 phydrax:CurrentSynapse current.json 0.25 1.0\n"
    )
    with h5py.File(tmp_path / "spikes.h5", "r+") as handle:
        handle["spikes/input/timestamps"][0] = 0.0011
    with pytest.raises(AdapterError, match="Recorded spike time"):
        ep.import_sonata(**arguments, dt_ms=0.25, execution="clock")


def test_spike_secondary_sorting_units_and_float_ids_are_not_silently_coerced(tmp_path):
    arguments = _fixture(tmp_path)
    with h5py.File(tmp_path / "spikes.h5", "r+") as handle:
        handle["spikes/input/timestamps"][:] = [0.003, 0.001]
    with pytest.raises(ValueError, match="secondary key"):
        ep.import_sonata(**arguments)
    with h5py.File(tmp_path / "spikes.h5", "r+") as handle:
        handle["spikes/input/timestamps"][:] = [0.001, 0.003]
        handle["spikes/input/timestamps"].attrs["units"] = "ticks"
    with pytest.raises(AdapterError, match="units"):
        ep.import_sonata(**arguments)
    with h5py.File(tmp_path / "nodes.h5", "r+") as handle:
        del handle["nodes/cells/node_id"]
        handle["nodes/cells/node_id"] = [1.0, 2.5]
    with pytest.raises(ValueError, match="integer dataset"):
        ep.import_sonata(**arguments)


@pytest.mark.parametrize("hazard", ["external", "vds", "reference", "compressed_size"])
def test_untrusted_hdf5_cannot_escape_resources_or_decode_unbounded_arrays(
    tmp_path, hazard
):
    arguments = _fixture(tmp_path)
    with h5py.File(tmp_path / "nodes.h5", "r+") as handle:
        if hazard == "external":
            handle["hidden"] = h5py.ExternalLink("/outside/not-read.h5", "/x")
        elif hazard == "vds":
            layout = h5py.VirtualLayout(shape=(1,), dtype="f8")
            layout[0] = h5py.VirtualSource("/outside/not-read.h5", "x", shape=(1,))[0]
            handle.create_virtual_dataset("hidden", layout)
        elif hazard == "reference":
            handle.create_dataset(
                "hidden", data=[handle["nodes/cells"].ref], dtype=h5py.ref_dtype
            )
        else:
            handle.create_dataset(
                "hidden",
                shape=(100_000_000,),
                dtype="f8",
                chunks=(1024,),
                compression="gzip",
                fillvalue=0,
            )
    with pytest.raises((AdapterError, ResourceReadError)):
        ep.import_sonata(**arguments)


def test_cable_sites_require_exact_unambiguous_supplied_mapping(tmp_path):
    arguments = _fixture(tmp_path)
    (tmp_path / "morphology.swc").write_text("1 1 0 0 0 10 -1\n2 3 10 0 0 1 1\n")
    (tmp_path / "node_types.csv").write_text(
        "node_type_id model_type model_template morphology\n"
        "2 biophysical hoc:caller-owned morphology.swc\n3 virtual NULL NULL\n"
    )
    with h5py.File(tmp_path / "nodes.h5", "r+") as handle:
        del handle["nodes/cells/5/dynamics_params"]
    with h5py.File(tmp_path / "edges.h5", "r+") as handle:
        handle["edges/input_to_cells/4/afferent_section_id"] = [9, 9]
        handle["edges/input_to_cells/4/afferent_section_pos"] = [0.5, 0.5]
    morphology = ep.CellMorphologyPlan(
        "cable",
        (
            ep.CompartmentSpec("soma", None, 20.0, 20.0),
            ep.CompartmentSpec("branch", "soma", 10.0, 2.0),
        ),
    ).prepare()
    cable = ep.CableSolverPlan(0.1).prepare(
        morphology, ep.MembraneProgram((ep.PassiveLeak(0.3, -65.0),))
    )
    with pytest.raises(ValueError, match="ambiguous"):
        ep.SONATACableBinding(cable, "morphology.swc", ((9, 0.5, 0), (9, 0.5, 1)))
    binding = ep.SONATACableBinding(cable, "morphology.swc", ((9, 0.5, 1),))
    arguments["components"]["morphology.swc"] = "morphology.swc"
    with pytest.raises(AdapterError, match="cable binding"):
        ep.import_sonata(**arguments)
    arguments["cable_bindings"] = {("cells", 7): binding, ("cells", 2**63 + 7): binding}
    circuit = ep.import_sonata(**arguments)
    assert [edge.connection.post_compartment for edge in circuit.edges] == [1, 1]
    exported = ep.export_sonata(circuit, tmp_path / "cable_export")
    restored = ep.import_sonata(
        node_files=exported.node_files,
        edge_files=exported.edge_files,
        spike_files=exported.spike_files,
        components=dict(exported.components),
        cable_bindings=arguments["cable_bindings"],
        trusted_root=tmp_path / "cable_export",
    )
    require_lossless(ep.sonata_roundtrip_report(circuit, restored))
    with h5py.File(tmp_path / "edges.h5", "r+") as handle:
        handle["edges/input_to_cells/4/afferent_section_pos"][0] = 0.500001
    with pytest.raises(AdapterError, match="exact native compartment"):
        ep.import_sonata(**arguments)

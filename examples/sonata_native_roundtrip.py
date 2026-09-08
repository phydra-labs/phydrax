#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Import grouped native SONATA, exercise a neuron, and export/reimport semantics."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications import electrophysiology as ep
from phydrax.interchange import require_lossless


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    with TemporaryDirectory(prefix="phydrax-sonata-") as directory:
        root = Path(directory)
        (root / "lif.json").write_text(
            json.dumps(
                {
                    "capacitance_nF": 0.25,
                    "leak_conductance_uS": 0.025,
                    "resting_mV": -65.0,
                    "threshold_mV": -50.0,
                    "reset_mV": -60.0,
                    "refractory_ms": 2.0,
                }
            )
        )
        (root / "synapse.json").write_text(
            json.dumps(
                {
                    "time_constant_ms": 2.0,
                    "conductance_scale_uS": 0.02,
                    "reversal_mV": 0.0,
                }
            )
        )
        (root / "node_types.csv").write_text(
            "node_type_id model_type model_template dynamics_params\n"
            "0 virtual NULL NULL\n"
            "1 point_neuron phydrax:LeakyIntegrateAndFire lif.json\n"
        )
        (root / "edge_types.csv").write_text(
            "edge_type_id model_template dynamics_params syn_weight delay\n"
            "0 phydrax:ConductanceSynapse synapse.json 1.0 0.5\n"
        )
        with h5py.File(root / "nodes.h5", "w") as handle:
            for population, type_id in (("input", 0), ("neurons", 1)):
                group = handle.create_group(f"nodes/{population}")
                group["node_id"] = np.asarray([17], dtype=np.uint64)
                group["node_type_id"] = [type_id]
                group["node_group_id"] = [3]
                group["node_group_index"] = [0]
                group.create_group("3")
            overrides = handle["nodes/neurons/3"].create_group("dynamics_params")
            overrides["threshold_mV"] = [-48.0]
        with h5py.File(root / "edges.h5", "w") as handle:
            group = handle.create_group("edges/input_to_neurons")
            group["edge_type_id"] = [0, 0]
            group["edge_group_id"] = [0, 0]
            group["edge_group_index"] = [1, 0]
            properties = group.create_group("0")
            properties["syn_weight"] = [0.5, 1.5]
            for name, population in (
                ("source_node_id", "input"),
                ("target_node_id", "neurons"),
            ):
                group[name] = np.asarray([17, 17], dtype=np.uint64)
                group[name].attrs["node_population"] = population
        with h5py.File(root / "spikes.h5", "w") as handle:
            group = handle.create_group("spikes/input")
            group.attrs["sorting"] = "by_time"
            group["node_ids"] = np.asarray([17, 17, 17], dtype=np.uint64)
            group["timestamps"] = [1.0, 4.0, 9.0]
            group["timestamps"].attrs["units"] = "ms"
        circuit = ep.import_sonata(
            node_files=(ep.SONATAFilePair("nodes.h5", "node_types.csv"),),
            edge_files=(ep.SONATAFilePair("edges.h5", "edge_types.csv"),),
            spike_files=("spikes.h5",),
            trusted_root=root,
            components={"lif.json": "lif.json", "synapse.json": "synapse.json"},
        )
        runtime = ep.prepare_sonata_network(
            circuit,
            queue_capacity=16,
            spike_capacity=16,
            recording_capacity=100,
            maximum_events_per_step=16,
            root_subdivisions=1,
        )
        run = ep.run_neural_network(runtime, ep.initialize_neural_network(runtime), 100)
        if bool(jnp.any(run.status != 0)):
            raise RuntimeError(f"Native SONATA execution rejected: {run.status}")
        if int(jnp.sum(run.delivered_messages)) != 6:
            raise RuntimeError("Parallel SONATA edges did not deliver every input spike.")
        exported = ep.export_sonata(circuit, root / "exported")
        restored = ep.import_sonata(
            node_files=exported.node_files,
            edge_files=exported.edge_files,
            spike_files=exported.spike_files,
            components=dict(exported.components),
            trusted_root=root / "exported",
        )
        report = ep.sonata_roundtrip_report(circuit, restored)
        require_lossless(report)
        print(
            "Native target voltage after 10 ms:",
            float(ep.neural_voltage(runtime, run.state)[1]),
            "mV",
        )
        print(f"Population-qualified nodes: {[node.key for node in circuit.nodes]}")
        print(
            f"Parallel-edge weights: {[edge.connection.weight for edge in circuit.edges]}"
        )
        print(f"Recorded spikes: {circuit.spikes[0].timestamps_ms.tolist()} ms")
        print(
            f"Semantic roundtrip: {report.status.name}; {len(circuit.resources)} bounded source manifests"
        )


if __name__ == "__main__":
    main()

# Native SONATA circuits and spikes

The electrophysiology SONATA adapter imports actual grouped HDF5 node and edge
populations, space-separated type tables, explicit JSON parameter resources, and
population spike files. It constructs native point-neuron objects and a
`SynapseNetworkPlan`; it does not run another simulator or interpret a SONATA
simulation configuration.

The layout follows the [official SONATA developer guide](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md).
The supported profile is deliberately strict: a model name is not evidence that
a foreign mechanism has the same equations, units, reset, or parameter meanings.

## Import explicit resources

```python
import jax
from phydrax.applications import electrophysiology as ep

jax.config.update("jax_enable_x64", True)

circuit = ep.import_sonata(
    node_files=(ep.SONATAFilePair("nodes.h5", "node_types.csv"),),
    edge_files=(ep.SONATAFilePair("edges.h5", "edge_types.csv"),),
    spike_files=("spikes.h5",),
    trusted_root="/data/circuit",
    components={
        "lif.json": "parameters/lif.json",
        "synapse.json": "parameters/synapse.json",
    },
    execution="event",
    dt_ms=0.1,
)
```

Paths are loaded through the canonical bounded resource reader beneath
`trusted_root`. Component names are opaque lookup keys: `dynamics_params=lif.json`
uses only the supplied `components["lif.json"]` path. There is no environment
expansion, network fetching, component-directory search, config recursion, HOC
execution, mechanism compilation, or automatic foreign model mapping. Enable
64-bit JAX before constructing models when the source parameters cannot be
represented exactly in native 32-bit arrays; the adapter rejects precision loss
rather than returning a lossless report for rounded parameters.

The result contains:

- `nodes`: `SONATANode` records, sorted by `(population, node_id)`. Each `.key` is
  population-qualified. `.model` is a native LIF, AdEx, supplied prepared cable, or
  `None` for a virtual source. `.properties` and `.dynamics` are resolved metadata.
- `edges`: `SONATAEdge` records, sorted by `(population, edge_id)`. Every edge has
  its own native `.connection`, including parallel edges with the same endpoints.
  Connection cell indices address `circuit.nodes`; compartment indices address
  the corresponding native cell. No endpoint pair is deduplicated.
- `synapse_plan`: the shared physical-delay synapse API, with heterogeneous
  compartment counts and one initial relation per edge. Empty connectivity uses
  one inactive spare slot, not a fabricated connection.
- `spikes`: `SONATASpikes` with a population name, read-only host `uint64`
  `.node_ids`, and read-only `float64` `.timestamps_ms`. These include both virtual
  input spikes and recorded spikes from real neurons. Import does not
  automatically replay recorded real-neuron spikes as external stimulation.
- `report`: the canonical `AdapterReport`.
- `resources`: canonical `ResourceManifest` entries with exact source byte
  identities and decoding counts. `components` retains explicitly supplied
  component bytes for semantic export.

Node IDs remain uint64 host IDs, not JAX array indices. Distinct populations can
both contain node 7. An edge dataset's `node_population` attribute determines
which node 7 it means. Native cell and relation indices must fit int32.

## Execute the supported native circuit

`prepare_sonata_network` preserves the imported node order and lowers virtual
population spike records into external source events. Spike records belonging
to physical populations remain observations and are not replayed as inputs.
Capacities are explicit:

```python
runtime = ep.prepare_sonata_network(
    circuit,
    queue_capacity=4096,
    spike_capacity=8192,
    recording_capacity=2000,
    maximum_events_per_step=512,
    root_subdivisions=4,
)
state = ep.initialize_neural_network(runtime)
run = ep.run_neural_network(runtime, state, 2000)
```

Point-neuron reset and threshold semantics come from their imported native
models. A supplied cable model additionally requires
`cable_detectors={(population, node_id): (compartment, threshold_mV,
rearm_mV)}` because morphology and synaptic section binding do not specify an
action-potential detector. `compartment` may be the exact native compartment ID
or index. Missing or surplus cable detector bindings fail rather than assuming
a soma threshold.

The prepared runtime uses the same physical delay and event/clock contracts
described in the electrophysiology guide. Importing valid data does not certify
that capacities are sufficient or that a numerical execution will be accepted;
inspect every `NeuralNetworkEvidence`.

## Group and type precedence

Each node population requires `node_type_id`, `node_group_id`, and
`node_group_index`. `node_id` is optional; omission means contiguous IDs from
zero. The same rule applies to `edge_id`; edge populations additionally require
`source_node_id` and `target_node_id`, each with a `node_population` attribute.

A record's group ID selects its numbered property group. Its group index selects
that group's row, **not** the population row. Arbitrary group IDs and permuted
indices are supported. Property arrays in a group have equal lengths; invalid
indices, missing groups, missing type bindings, duplicate record/type IDs, and
inconsistent lengths fail explicitly.

Precedence is:

1. Type-table property defaults.
2. Indexed HDF5 per-record properties overriding those defaults.
3. The resolved `dynamics_params` component's flat JSON parameters.
4. Indexed HDF5 `dynamics_params/<parameter>` values overriding component values.

The CSV dialect is ASCII, space-separated, with repeated spaces and standard
quoted fields supported. `NULL` means an absent type-table property. HDF5
per-record properties must be rank-one scalar arrays. Flat population-level
property columns are not substituted for SONATA group resolution.

## Native model templates and units

Supported point node types are `model_type=point_neuron` with exactly one of:

| `model_template` | Required dynamics parameters |
| --- | --- |
| `phydrax:LeakyIntegrateAndFire` | `capacitance_nF`, `leak_conductance_uS`, `resting_mV`, `threshold_mV`, `reset_mV` |
| `phydrax:AdaptiveExponentialIntegrateAndFire` | The LIF parameters plus `slope_mV`, `adaptation_conductance_uS`, `adaptation_time_constant_ms`, `adaptation_increment_nA` |

Both support `refractory_ms` (default 0). AdEx additionally supports
`exponential_threshold_mV` (default −50 mV), which is the exponential-current
onset, **not** the spike cutoff `threshold_mV`. Unknown or missing required
parameters fail, as do invalid physical parameters.

`model_type=virtual` has no model template, dynamics, or receiving synaptic
inputs. It is a spike source, not an approximated point neuron.

Supported edge templates are:

| `model_template` | Required dynamics parameters |
| --- | --- |
| `phydrax:CurrentSynapse` | `time_constant_ms`, `current_scale_nA` |
| `phydrax:ConductanceSynapse` | `time_constant_ms`, `conductance_scale_uS`, `reversal_mV` |

Edges require explicit `delay` and `syn_weight`. For these native templates,
`delay` is milliseconds and `syn_weight` is dimensionless; the physical scale is
in the dynamics parameters. Current scale is **outward-positive**, so an
excitatory current synapse uses negative `current_scale_nA`. This is not a unit
reinterpretation of an arbitrary NEST or NEURON `syn_weight`. `nsyns` may be 1;
aggregated multiplicities must instead be represented as individual edge rows.

`execution="event"` retains physical delays. `execution="clock"` requires every
delay and recorded spike timestamp to be representable on `dt_ms`, allowing only
a floating-point representational ULP allowance, not physical time rounding.
Negative/nonfinite times and clock indices outside int32 are rejected.

## Explicit cable bindings

A `model_type=biophysical` node requires an already prepared native cable and an
explicit morphology resource. The adapter never guesses a foreign template's
channel composition or a morphology's section numbering.

```python
binding = ep.SONATACableBinding(
    prepared_cable,
    "cell.swc",
    ((0, 0.5, 0), (9, 0.5, 3)),
)
# Pass these alongside node_files, edge_files, trusted_root, etc.
components = {"cell.swc": "morphologies/cell.swc"}
cable_bindings = {("cells", 42): binding}
```

Each site tuple is `(SONATA section ID, exact normalized section position,
native compartment index)`. The mapping may contain multiple exact sites for
one compartment, but no duplicate section/position pair. A cable endpoint must
supply both `afferent_section_id` and `afferent_section_pos` (target), or both
`efferent_section_id` and `efferent_section_pos` (source), and must match the
binding exactly. Missing mappings and ambiguous or nearly matching sites fail;
no nearest-compartment or soma fallback exists. Section-based sites on point or
virtual nodes, spatial center/surface site coordinates, and foreign cable
parameter overrides are unsupported.

The caller is responsible for the scientific correspondence between the supplied
cable, opaque morphology bytes, and exact section map. Export retains the
resource bytes and mapping semantics, but it does not serialize or execute cable
mechanism code. Re-import requires the explicit native bindings again.

## Spike ordering and units

Spike files use `/spikes/<population>/node_ids` and `timestamps`. The population
`sorting` attribute may be a SONATA HDF5 enum or a scalar string: `none`, `by_id`,
or `by_time`. Missing sorting means `none`. Declared sorting is checked before
normalization. `by_id` additionally requires nondecreasing timestamps among
spikes with the same ID; `by_time` permits any ID order at a time tie.

Timestamp units must explicitly be `ms` or `s`. Seconds are converted to
milliseconds; unknown units are rejected. All node references must exist. The
returned and exported representation is sorted by time with ID as a deterministic
tie-breaker. Duplicate spikes remain distinct records.

## Resource and decoding boundaries

`limits` is the canonical `ResourceLimits` object. The default per-resource byte
limit is 64 MiB, nesting depth 16, object/element bound 2,000,000, and aggregate
attribute bound 100,000. `max_decoded_bytes` defaults to 256 MiB and bounds the
aggregate resource images and decoded numeric arrays, not just compressed file
size. Limits are checked before reading HDF5 dataset payloads.

The adapter rejects external/soft links, hard-link aliases/cycles, virtual
datasets, external dataset storage, object/region references, compound datasets,
variable-length datasets, unknown filters, unknown attributes, and unsupported
nested property groups. Use fixed-width UTF-8 string datasets. Scalar string
attributes can be variable-length; decoding reserves a bounded source image per
scalar. Only built-in deflate, shuffle, and Fletcher32 filters are accepted.
Malformed or lossy integer narrowing is rejected rather than silently casting
floating IDs to integers. JSON dynamics resources must be flat scalar objects
with unique keys.

This profile does not import arbitrary reports, simulator configuration,
extracellular fields, plasticity recipes, foreign mechanisms, or aggregated
synapse semantics. Unsupported required semantics raise `AdapterError`; malformed
values raise `ValueError`; canonical resource-policy/limit failures raise
`ResourceReadError`.

## Semantic export and roundtrip

```python
from phydrax.interchange import require_lossless

exported = ep.export_sonata(circuit, "/data/new-export-directory")
restored = ep.import_sonata(
    node_files=exported.node_files,
    edge_files=exported.edge_files,
    spike_files=exported.spike_files,
    trusted_root="/data/new-export-directory",
    components=dict(exported.components),
    # cable_bindings=the_same_explicit_native_bindings,  # if cables are present
)
require_lossless(ep.sonata_roundtrip_report(circuit, restored))
```

Export requires a new directory and never overwrites source files. It emits real
HDF5 property groups and space-separated type CSV, one file pair per population,
plus sorted spike data and explicit component resources. IDs, parallel edges,
resolved model parameters, per-record metadata, physical delays, and spike
population/ID/time semantics survive. Group organization, redundant type
defaults, original spike ordering/units, parameter-file indirection, and auxiliary
edge indices need not be byte-identical. This normalization is declared in the
canonical report's assumptions; losslessness applies to the supported resolved
semantic profile, not arbitrary SONATA contents.

The import record is a coherent snapshot. Export rejects a changed semantic
snapshot rather than mixing altered model/connectivity objects with stale source
metadata. A roundtrip report actually compares semantic fingerprints and raises
on mismatch; it is not an unconditional success flag.

Run the self-contained native import/advance/export example with:

```sh
python examples/sonata_native_roundtrip.py
```

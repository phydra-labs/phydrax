# Planar layout and explicit process stacks

Layout decoding and three-dimensional process construction are separate host-side
boundaries. A layout supplies planar geometry, layer keys, occurrence paths, and
provenance. A process stack supplies every vertical interval, precedence, and void
target explicitly. Neither boundary infers a physical process from a GDSII or OASIS
layer number.

## Decode a bounded layout

Choose the coordinate contract before decoding. The decoder preserves it on the
resulting `LayoutModel`; it does not guess units from a file header. File reads are
bounded and rooted in a caller-selected trusted directory.

```python
from pathlib import Path

import phydrax as phx

layout_path = Path("chip.gds")
limits = phx.interchange.ResourceLimits(
    max_bytes=16 * 1024 * 1024,
    max_depth=4,
    max_nodes=100_000,
    max_attributes=10_000,
    max_losses=0,
)
policy = phx.interchange.LayoutImportPolicy(
    phx.interchange.LayoutFormat.GDSII,
    phx.SpatialCoordinateContract(phx.units.MICROMETER),
    limits,
    top_cell="TOP",
)
decoded = phx.interchange.read_layout(
    layout_path,
    policy,
    trusted_root=layout_path.parent,
)

core_layer = phx.interchange.LayoutLayerKey(1, 0)
core = next(region for region in decoded.model.regions if region.layer == core_layer)
assert decoded.source_digest == decoded.resource_manifest.content_sha256
assert decoded.model.coordinate_contract == policy.coordinate_contract
```

`LayoutRegion` retains the `LayoutLayerKey`, exact occurrence path, planar geometry,
and provenance identity. `LayoutImportResult.report` records semantic loss and
validity. Paths are accepted only under the policy's declared tolerance and loss
waivers; the decoder never silently polygonizes or discards them. Missing optional
GDSII/OASIS support is an `LayoutAdapterError`, not an empty layout.

## Lower an explicit stack

Pass decoded planar geometry into `geometry.process`, then declare physical
thickness and ordering. `ZInterval` is physical in the layout model's coordinate
contract. Add a `StackVoid` only for an explicitly named target-region set.

```python
from phydrax.geometry.process import (
    ProcessStack,
    StackRegion,
    ZInterval,
    lower_process_stack,
)

metal = StackRegion(
    "metal",
    core.geometry,
    ZInterval(0.0, 0.8),
    precedence=20,
)
stack = ProcessStack(decoded.model.coordinate_contract, (metal,))
stack_result = lower_process_stack(
    stack,
    Path("process-output.brep"),
    operand_directory=Path("process-operands"),
)

assert stack_result.model.source_revision
assert stack_result.revision.revision_id
assert stack_result.association_graph.graph_id
assert dict(stack_result.named_solid_entity_ids)["metal"]
```

A `StackVoid("opening", footprint, ZInterval(...), target_region_ids=("metal",))`
removes only the specified stack regions. There is no default target, inferred
material, inferred deposition order, or layout-process inference.

`ProcessStackResult` carries the persisted B-Rep model, named solid and face
identities, region/patch adjacency, `CADRevision`, and `AssociationGraph`. Use those
identities to construct downstream `RegionControl` and `PatchControl` scopes; do not
recover regions from geometry proximity or raw CAD importer tags.

## Hand off to meshing

Meshing retains the stack result's coordinate contract. Create scopes only through
the meshing provider's exact B-Rep entity-scope API, bind `RegionControl` to solid
identities and `PatchControl` to face identities, then place a `UniformSizeControl`
on the complete meshing scope. A subsequent provider preflight decides whether its
specific backend can honor the requested controls.

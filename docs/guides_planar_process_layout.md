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
    ZInterval(0.0, 0.8, unit=decoded.model.coordinate_contract.length_unit),
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

A `StackVoid("opening", footprint, ZInterval(lower, upper, unit=unit), target_region_ids=("metal",))`
removes only the specified stack regions. Each interval requires an explicit native
length unit, exactly matching the stack coordinate contract; mismatched scales or
dimensions refuse rather than silently converting. Use `phydrax.units.convert_value`
explicitly before declaring coordinates in another unit. There is no default target,
inferred material, inferred deposition order, or layout-process inference.

`ProcessStackResult` carries the persisted B-Rep model, named solid and face
identities, region/patch adjacency, `CADRevision`, and `AssociationGraph`. Use those
identities to construct downstream `RegionControl` and `PatchControl` scopes; do not
recover regions from geometry proximity or raw CAD importer tags.

## Bind stable recipe names

`geometry.design.RecipeEntityName(kind, region, source_entity, lineage_path)` is a
value identity authored by the recipe. Its region, semantic source name, and
lineage path contain no model/revision fingerprint, numerical parameter value,
or geometry-derived label. Keep these values fixed during parameter edits.

Pass a complete mapping from exact face/edge occurrences in
`stack_result.association_graph.source_revision` to authored names into
`stack_result.recipe_names(source_names)`. For example, the source occurrence
`"operand:0/solid:0/face:0"` may bind to
`RecipeEntityName("face", "metal", "bottom", ("extrude",))`; declaring all other
source face occurrences is required. This source binding is explicit, not a
claim that importer indices are themselves stable names.

The returned `RecipeEntityNames` carries names in the lowered model's canonical
order. `bindings.resolve(name)` returns the current `BRepEntityId`, and
`bindings.name(entity_id)` performs the reverse lookup. Retain the stable name
for boundary conditions, then resolve it in each new realization. The partition
step appends `"partition"` to the authored lineage path.

Names require exhaustive source and target lineage and refuse splits, merges,
creation/deletion, unresolved ancestry, incomplete authored bindings, and
ambiguous names on shared entities. Planar `name_partition_entities` supports
both face and edge ancestry. Current three-dimensional partition publications
certify faces but not edges: `bindings.edge_names is None`, and resolving an
edge name raises `RecipeNamingError`. No edge ancestry is inferred from face
adjacency, proximity, or geometry. `None` is distinct from an empty, supported
family.

`transport_recipe_names(source_bindings, target_model, sewing_lineage)` consumes
the native `BRepSewingLineage`: faces retain their indices and edge images must
be exhaustive and bijective. Vertex/coedge coverage must also be exhaustive and
topology-preserving. Sewing appends `"sewing"` to the path, refuses split/merged
edge identity, and cannot make an unsupported source family supported. The
caller supplies the actual source/target publication lineage, not an unrelated
record of matching dimensions.

## Bind selected boundary regions

Geometric predicates and STEP label lookup produce the same `BRepSelection`.
Use the selection's exact identities to bind a meshing patch through exhaustive
recipe names. The following is illustrative: `selection`, `bindings`, and
`domain` must already refer to the same native publication, and `"fluid"` must
be an adjacent region declared by that domain.

```python
outlet_control = selection.bind_region(
    "outlet", bindings, domain, ("fluid",), include_unresolved=False
)
assert outlet_control.recipe_binding is not None
```

Omitting the unresolved inclusion choice refuses when any membership is
unresolved; it never silently selects only convenient faces. Empty selections
also refuse. Definition selections expand to all explicitly declared domain
occurrences; qualified paths require equality rather than prefix/proximity
matching.

After a topology-preserving parameter edit, obtain the new exhaustive
`RecipeEntityNames` and native domain, then call
`outlet_control.rebind(edited_bindings, edited_domain)`. Rebinding verifies the
stable selected keys **and the complete family inventory**. Splits/merges,
missing ancestry, or a mismatched model/revision refuse; equal counts or
similar geometry cannot preserve a boundary-condition binding.
See [geometric selection and normal analysis](api/geometry.md#geometric-predicates-and-normal-ranges).


## Hand off to meshing

Meshing retains the stack result's coordinate contract. Create scopes only through
the meshing provider's exact B-Rep entity-scope API, bind `RegionControl` to solid
identities and `PatchControl` to face identities, then place a `UniformSizeControl`
on the complete meshing scope. A subsequent provider preflight decides whether its
specific backend can honor the requested controls.

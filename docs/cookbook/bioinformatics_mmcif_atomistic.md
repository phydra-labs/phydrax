# mmCIF to macromolecular and atomistic state

Run this recipe with a local PDBx/mmCIF path. It plans from the concrete host record and
performs all-or-nothing chemistry lowering.

```python
from pathlib import Path
import sys

from phydrax.bioinformatics.interchange import load_mmcif
from phydrax.bioinformatics.structure import (
    StructureLoweringPlan,
    lower_macromolecular_record,
)

path = Path(sys.argv[1])
record = load_mmcif(path)
plan = StructureLoweringPlan.for_record(
    record,
    strict_component_chemistry=True,
    coordinate_dtype="float64",
)
result = lower_macromolecular_record(record, plan)

if not bool(result.valid):
    evidence = {
        label: int(value)
        for label, value in zip(result.evidence_labels, result.evidence, strict=True)
    }
    raise RuntimeError(f"structure lowering failed: status={int(result.status)}, {evidence}")

structure = result.structure
atomistic = result.atomistic_structure
topology = result.atomistic_topology
assert structure is not None
assert atomistic is not None
assert topology is not None
print(atomistic.positions.shape, topology.bonds.shape)
```

`load_mmcif` is host-only. It preserves entities, label/auth identity, alternate
locations, coordinate models, occupancy, missingness, chemical components, connections,
and assemblies. `for_record` chooses sufficient capacities for this record; production
pipelines may instead use fixed bucket capacities and must reject overflow. Lowering
never guesses unresolved atomic numbers, component atoms, or bond references. The
atomistic result supplies geometry/topology only: it is not protonated, repaired,
force-field-parameterized, or assigned an energy model. Record the input byte digest in
your surrounding artifact; the parser does not download or identify structures.

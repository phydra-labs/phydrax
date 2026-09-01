# Fixed-tree phylogenetic likelihood

This recipe evaluates, but does not search, a rooted two-tip tree under JC69.

```python
from phydrax.bioinformatics.phylogenetics import (
    LikelihoodPartition,
    felsenstein_pruning,
    jc69,
    tip_partials_from_sequence,
    tree_topology,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, encode_sequences

# Nodes 0 and 1 are tips; node 2 is their root.
topology = tree_topology([2, 2, -1], child_capacity=2, tip_indices=[0, 1])
assert bool(topology.valid)

alignment = encode_sequences(
    ["ACGTAC", "ACGTTC"],
    DNA_IUPAC,
    record_ids=[0, 1],
)
partials = tip_partials_from_sequence(alignment)
assert bool(partials.valid)

partition = LikelihoodPartition(
    partials.site_mask,
    jc69(),
    ascertainment="none",
    partition_name="all-sites",
)
result = felsenstein_pruning(
    topology,
    partials.tip_partials,
    [0.10, 0.10, 0.0],  # one length per node; the root length must be zero
    (partition,),
)
assert bool(result.valid)
print(float(result.log_likelihood))
```

The record order must match `topology.tip_indices`. Ambiguous symbols become finite
state sets; gap, unknown, missing, mask, and padding carry no state information. Every
site must belong to exactly one `LikelihoodPartition`. The result is exact for the
fixed finite-state model, fixed topology, branches, rates, ascertainment choice, and
pattern weights, subject to floating-point transition evaluation. It is not evidence
that JC69 is adequate or that this topology is optimal; bounded NNI search is a
separate heuristic operation.

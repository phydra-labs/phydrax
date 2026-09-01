# Spatial neighbor graph and restricted autocorrelation

This example uses one calibrated frame, disconnects tissue sections, preflights graph
capacity, and restricts permutations by section.

```python
import jax

from phydrax.bioinformatics.spatial import (
    MICROMETRE,
    SpatialCoordinates,
    SpatialFrame,
    SpatialNeighborPlan,
    build_spatial_neighbor_graph,
    spatial_autocorrelation_test,
)

frame = SpatialFrame("assay-frame", ("x", "y"), MICROMETRE)
coordinates = SpatialCoordinates(
    [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    frame,
)
sections = [0, 0, 1, 1]
graph = build_spatial_neighbor_graph(
    coordinates.values,
    SpatialNeighborPlan(
        mode="knn",
        k=1,
        capacity=1,
        weight="binary",
        row_normalize=True,
    ),
    section_index=sections,
)
assert bool(graph.valid)

result = spatial_autocorrelation_test(
    [1.0, 1.5, 3.0, 3.5],
    graph,
    jax.random.key(42),
    statistic="moran",
    permutations=999,
    donor_index=[0, 0, 1, 1],
    section_index=sections,
    exchangeability_blocks=sections,
)
print(result.statistic, result.p_value, result.valid, result.status)
```

Coordinates must already be in commensurate axes and units; `PIXEL` has no physical
scale. kNN routing and tied-distance choices are nondifferentiable. Graph overflow
returns an invalid graph with required-capacity evidence rather than a truncated graph.
The $p$-value is a finite restricted-randomization estimate. Spots are observations,
but independent donors are the biological replication; a one-donor test is invalid.
For a full `SpatialAssay`, use `assay_autocorrelation_test` to derive donor, section, and
sampling weights from lineage-aware assay metadata.

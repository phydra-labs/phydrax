# Domains

Domains describe the geometry and coordinate systems for PDE problems. They are used to
sample points, build product structures, and define functions with explicit coordinate
labels.

For a guide to components and sampling modes (paired vs coord-separable), see
[Guides → Domains and sampling](../../guides_domain.md).

Use `HyperRectangle` for analytic axis-aligned boxes where a point is naturally a
single vector, such as `R^6 -> scalar` supervised learning inputs. Use product
domains when each coordinate factor should keep its own label and sampling
structure.

Use `TrajectoryDatasetDomain` when each dataset row has a vector-valued trajectory
with a shared `dt` but row-specific length. It keeps `data` and `t` paired so
physics residuals at `t` only sample valid times for the selected dataset row.

Use `RaggedSeriesDatasetDomain` when the ragged time series is an input attached
to an empirical row and the target is supervised once per row, such as
`(static, variable-length sensor series) -> summary parameters`.

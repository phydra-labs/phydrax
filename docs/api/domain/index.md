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

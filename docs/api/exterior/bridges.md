# Integration, chains and boundary bridges

## Smooth de Rham integration

`DeRhamBridge(complex, chart, parameterizations, /)` binds an explicit realization
and chart to one `CellParameterization` per represented degree. Cell coordinates
are not silently interpreted as maps or quadrature.

`CellParameterization(degree, cell_count, ambient_dimension, map_function,
jacobian_function, reference_points, quadrature_weights, orientation_signs, /,
*, coorientation_signs=None)` declares all integration data. Builders
`simplicial_parameterizations(topology, vertices, /, *, order)` and
`structured_parameterizations(bridge, /, *, order)` take polynomial exactness order,
not an inferred refinement-dependent rule.

Native unit-simplex/unit-cube rules retain explicit reference dimension, exact
degree and measure mass, with resource admission before tensor allocation.
Generic facet traces use canonical `ReferenceCellTopology` descriptors. This
prepared-rule support does not implicitly admit a generic public integration
reference lacking dimension.


`integrate_form(form, bridge, /)` accepts chart or domain forms and returns
`DiscreteForm`. It currently integrates scalar fibers; twist must match the
realization's primal twist. Embedded twisted integration additionally requires
per-cell coorientation. `validate_de_rham_commutation(form, bridge, /, *, tolerance)`
returns numerical R(dα)−d(Rα) residual evidence, with realization and degree identity.

`metric_dual_hodges(bridge, metric, /, *, dual)` receives explicit dual cell
parameterizations: dual[k] has degree n−k and count equal to primal degree k.
It returns metric-only diagonal Hodges, not a duplicate metric-assembly carrier.

::: phydrax.exterior.CellParameterization

::: phydrax.exterior.DeRhamBridge

::: phydrax.exterior.simplicial_parameterizations

::: phydrax.exterior.structured_parameterizations

::: phydrax.exterior.integrate_form

::: phydrax.exterior.validate_de_rham_commutation

::: phydrax.exterior.metric_dual_hodges

## Prepared chain integration

`AbstractChainIntegrationKernel` supplies degree counts, orientation-block offsets
and explicit `kernel_id`. It prepares point/segment integration and degreewise
point evaluation. `integrate_segments(start, end, /, *, weight="uniform",
phase_rate=None, maximum_segments=None)` uses exp(+i phase_rate t) for phase
weight, with dimensionless phase along the segment.

`PreparedChainQuery(indices, coefficients, valid, successful, overflow, /, *,
dof_count, degree, kernel_id)` holds reusable sparse routes. `gather` maps a
rank-one cochain to `(points, *value_shape)` and `deposit` is its conjugate adjoint.
A failed or overflowing path is evidence, not silently clamped physical data.

The affine simplicial locator walks exact facet intervals with fixed capacity,
deterministic ties, zero-segment support and explicit exit/overflow evidence.
Every interval is integrated; classifying endpoints alone is insufficient.
Curved exactness requires a different geometry contract and is explicitly refused
by affine routes. Cubical spline Whitney routes preserve shape order and canonical
component offsets.

::: phydrax.exterior.AbstractChainIntegrationKernel

::: phydrax.exterior.PreparedChainQuery

::: phydrax.discretization.CubicalSplineWhitneyKernel

::: phydrax.discretization.fem.SimplicialWhitneyKernel


## Traces and exact products

Cell boundary orientation is outward-normal first. Trace maps carry a degreewise
`ComplexMap`; the scientific invariant is d_boundary tr = tr d. Finite-element
circulation uses tangential traces and flux uses normal traces. Embedded RWG flux
requires the declared coorientation and reuses the generic Gram/J⁺ mapping owner.

`boundary_subcomplex(topology, boundary_mask=...)` binds selected boundary facets
and their incidence closure. `trace_map(realization, boundary_mask=None)` returns
a signed restriction `ComplexMap` with the true principal restricted pairing;
FE and spline owners provide their canonical DOF trace maps.
`trace_evidence(mapping, values, tolerance=1e-12)` measures the commuting residual
on one supplied vector per degree.

::: phydrax.discretization.boundary_subcomplex

::: phydrax.exterior.trace_map

::: phydrax.exterior.trace_evidence


Alexander–Whitney and Serre diagonals remain in `phydrax.topology`:
`alexander_whitney_diagonal(topology, support, /)` and `serre_diagonal(cubical, /)`
retain canonical oriented occurrence multiplicity. Exact cup-product arithmetic
reduces each multiplication before int64 accumulation and checks topology identity.
Whitney products and Cartan transport use reconstruction/integration rather than
pretending all finite cochain products are exact smooth wedge products.

`WhitneyProductPlan(complex, kernel, /, *, quadrature_order=4,
boundary="absolute")` binds genuine reconstruction/integration geometry.
Quadrature order is nodes per reference direction. `whitney_wedge`,
`interior_product` and `lie_derivative` accept that plan or a structured bridge;
direct bridge products retain the bridge's own realization identity.
Fixed wedge/interior execution reuses quadrature/blades and reconstruction/vector
queries cached at admission. JAX-built numeric cache leaves preserve admission-time
geometry differentiation; geometry binding is admission-only, with no partial
tree-update or geometry-refresh contract. Semi-Lagrangian backtracked-chain
geometry and source queries remain genuinely dynamic.

Primal placement is required; dual product geometry is not inferred from shape.
Semi-Lagrangian degree0/1 uses exact points/segments; higher degrees use declared
quadrature over backtracked chains. Matrix-fiber raw wedge retains operand order.

::: phydrax.exterior.WhitneyProductPlan

::: phydrax.exterior.cochain_cup_product

::: phydrax.exterior.whitney_wedge

::: phydrax.exterior.interior_product

::: phydrax.exterior.lie_derivative


## Coefficient systems and physics boundaries

A coefficient system declares scalar/matrix-fiber transport on incidence
occurrences. Sparse block routes are cell-major/fiber-minor and distinguish
transpose, conjugate reverse and Hilbert adjoint. Flat transport supports d²=0;
curvature evidence records holonomy defects. Orientation coefficients and Bloch
transport do not become lossy SPD metric Hodges.

`CoefficientSystem(topology, transports, /, *, spaces=None, key=None)` binds
unsigned scalar or block transports; its owner applies incidence signs.
`.refresh(transports)` retains the prepared binding. A curved connection's d²
operator is measured by `curvature_evidence`; it is not admitted as a flat complex.
`bloch_coefficient_system(cubical, wavevector, periods=..., spaces=..., key=...)`
uses actual periodic wraps exp(+ikL), not conjugation of ordinary periodic d.
Orientation systems support regular manifold stars and declared paired polygon
presentations; nonrecoverable attaching structure is refused.

::: phydrax.exterior.CoefficientSystem

::: phydrax.exterior.twisted_differential

::: phydrax.exterior.curvature_evidence

::: phydrax.exterior.bloch_coefficient_system

::: phydrax.exterior.orientation_coefficient_system

::: phydrax.exterior.sheaf_laplacian


PIC gathers physical E and raw constrained B, never H in the Lorentz force.
Current content is +∫W and the already-consistent end-to-end continuity sign is
preserved. Far-field and PSATD box geometry have different stencils and stay under
their respective owners.

Matching Maxwell FEM–BEM requires real trace, dual conormal, RWG/BC pairing and
boundary operators. A caller-supplied periodic block solve is a separate envelope
contract, not automatic physical coupling. Support reports and residual evidence
must identify the actual route; matching/nonmatching agreement does not justify
ignoring conormal data.

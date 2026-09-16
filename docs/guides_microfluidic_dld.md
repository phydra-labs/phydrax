# Deterministic lateral displacement

Phydrax exposes deterministic lateral displacement (DLD) as a composition of exact
circular-post geometry, an explicitly qualified velocity field, one-way finite-radius
particle transport, outlet classification, and separation metrics. It is not a universal
microfluidic solver and does not silently change fidelity.

## Geometry and flow identity

`DLDTopology` fixes rows, columns, shift period, outlets, and particle capacity.
`DLDDesign` supplies post radius, pitches, row shift, channel walls, depth, and inlet/outlet
coordinates. `DLDGeometryPlan` computes exact distance and inward normals for disjoint
circular posts and straight sidewalls. Closest-feature ties are rejected; they are not
resolved by an arbitrary normal.

`DLDLatticeBoltzmannFlowPlan` consumes an already prepared Phydrax LBM dynamics object
whose lattice, collision, forcing, and precision match one
`LatticeBoltzmannOperatingEnvelopePlan`. The flow result remains tied to the DLD geometry
identity and reports steady residual, mass drift, and envelope admission. A prescribed
field may be used for synthetic invariant studies, but its evidence ID must match the
particle field binding and it does not establish device accuracy.

## Finite particles and outlets

`FiniteParticleTransportPlan` provides explicit overdamped Stokes, inertial Stokes, and
Brownian branches. Radius erosion, bounded contact bisection, wall response, capacity,
random state, and time commit atomically. The initial DLD workflow is one-way: there is no
particle feedback, particle-particle collision, lubrication, aggregation, or point-particle
fallback.

`DLDOutletPlan` uses disjoint half-open transverse intervals, with the final upper edge
closed. Out-of-range exits are ambiguous (`-3`), not assigned to the nearest outlet.
`DLDMetricPlan` reports the class/outlet transfer matrix, purity, recovery, contamination,
residence time, terminal fraction, and ambiguity count.

## Screening and robustness

`DLDEmpiricalScreenPlan` is an independent empirical fidelity level. Coefficient, exponent,
support interval, and primary-equation source identity are mandatory. Its critical diameter
is not substituted for resolved transport.

`DLDRobustnessPlan` consumes an explicit fixed as-built sample ensemble and reports nominal,
mean, worst, conditional value at risk, chance satisfaction, and invalid sample count.
Invalid samples remain visible and make the robustness evidence ineligible.

Run the executable synthetic workflow with:

```console
python examples/dld_separation.py
```

The example proves geometry-to-transport-to-outlet wiring and conservation invariants. Its
scientific status is explicitly synthetic. Device validation requires a rights-bearing
reference manifest and remains outside the candidate claim.

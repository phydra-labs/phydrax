# Rarefied gas dynamics

Phydrax distinguishes physical molecular-velocity kinetics from low-order D2V/LBM reference-population methods.

`MolecularVelocityQuadrature` always stores three-dimensional molecular velocities and independent physical integration weights. Its spatial streaming projection may have one, two, or three dimensions. `PopulationUpwindFluxPlan` streams every population using its projected molecular velocity.

`PositiveDiscreteMaxwellianPlan` solves the five-moment entropy-dual problem for a strictly positive discrete equilibrium. It matches mass, three momentum components, and energy on the actual finite velocity quadrature rather than sampling a continuous Maxwellian and accepting quadrature drift.

`MonatomicBGKCollisionPlan` uses relaxation time equal to dynamic viscosity divided by pressure and advances homogeneous relaxation analytically. It reports moment defect, entropy change, positivity, and relaxation time. `ShakhovCollisionPlan` provides the selected non-unit-Prandtl extension; positivity and invariant correction are explicit evidence.

`MaxwellGasSurfaceBoundary` blends exact specular routing and diffuse wall emission. Construction requires a velocity set closed under reflection for the selected wall normal, and diffuse density enforces zero normal mass flux.

`KineticBreakdownPlan` combines a local Knudsen estimate and distribution-to-equilibrium defect. It is the physical eligibility seam for later continuum-kinetic routing; shock sensing alone is not a Knudsen model.

`KineticSyntheticAccelerationPlan` exposes the deterministic micro-macro correction used after a continuum synthetic solve. It preserves a zero-invariant micro distribution, limits only for positivity, and reports the resulting moment defect. It is not DSMC and performs no stochastic reconstruction or dynamic particle repartitioning.

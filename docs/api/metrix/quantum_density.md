# Bures and mixed-state quantum geometry

`BuresDensityManifold` represents faithful density matrices: Hermitian,
strictly positive, and trace one. Rank-deficient states belong to explicit
fixed-rank strata rather than the same smooth manifold.

::: phydrax.metrix.BuresDensityManifold

::: phydrax.metrix.SLDQuantumFisherGeometry

::: phydrax.metrix.FaithfulDensityReport

The package convention fixes `g_Bures = g_SLD / 4`. SLD actions solve the
Sylvester equation `(rho L + L rho) / 2 = tangent` through the shared Hermitian
matrix-equation substrate.

`GeometryPrecisionPolicy` controls density coordinates, metric reductions, and
reported decisions. `HermitianPrecisionPolicy` independently controls the
Hermitian spectra and Sylvester factorization used by Bures/SLD operations.
POVM, likelihood, tomography, and frozen artifacts retain both evidence
envelopes; output casting occurs only at public result boundaries.

::: phydrax.metrix.bures_squared_distance

::: phydrax.metrix.principal_purification

::: phydrax.metrix.uhlmann_alignment

## Tomography

`QuantumPOVM` validates positive effects and completeness. Tomography uses
trace- and positivity-preserving Bures natural-gradient retractions.

::: phydrax.uq.QuantumPOVM

::: phydrax.uq.QuantumTomographyData

::: phydrax.solver.QuantumTomographyProblem

::: phydrax.solver.solve_quantum_tomography

## Dissipative dynamics

`LindbladProblem` exponentiates the finite-dimensional Lindblad generator as a
linear channel. The result reports trace, Hermiticity, and minimum-eigenvalue
histories.

::: phydrax.solver.LindbladProblem

::: phydrax.solver.solve_lindblad

## Fixed-rank stratification

`FixedRankDensityManifold(n, r)` stores a full-column purification factor and
implements horizontal quotient geometry for exactly one rank. It never extends
SLD/Bures inversion across support loss. `DensityRankStratification` classifies
with explicit absolute/relative thresholds; eigenvalues in the ambiguity band
produce invalid evidence instead of a rank gradient. `RankTransitionProposal`
records source/target rank, trigger, deterministic truncation/embedding,
discarded mass, and optimizer-state transfer semantics for a fresh host epoch.
The union of ranks is not advertised as one smooth manifold.

::: phydrax.metrix.FixedRankDensityManifold

::: phydrax.metrix.DensityRankStratification

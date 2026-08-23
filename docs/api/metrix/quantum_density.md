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

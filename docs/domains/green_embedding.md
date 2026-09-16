# Green functions, impurity solving, and finite-bath DMFT

The Green-function core distinguishes `MatsubaraGreenFunction` from
`MatsubaraSelfEnergy`. Self-energy moments contain a static limit plus an
inverse-frequency tail; they are not Green-function moments. Scalar retarded data and
fermionic spectral densities retain frequency units, mode labels, causality,
positivity, zeroth-moment, and first-moment residuals separately. Source-to-target
fermionic thermal channels use one global partition function and never assume that an
annihilation operator is square within one particle-number sector.

`ImpurityEnvironment` contains exactly one `MatsubaraHybridization` or one
`AndersonBath`. The finite-bath fitter uses non-negative coupling strengths, so it
cannot hide an acausal target by fitting negative spectral weight. Fit residual,
finite-bath discrepancy, causality, and moment discrepancy remain distinct evidence.
The native exact-diagonalization provider solves every fixed-cardinality sector through
the direct quantum-lattice sector basis and reports spectral sum, causality, moments,
Dyson closure, density symmetry, and Hamiltonian residuals.

`SingleSiteDMFTPlan` is intentionally narrow: normal state, orthonormal basis, one
orbital, one local self-energy, finite temperature, and a finite Anderson bath. The
chemistry owner controls the lattice projection, filling, and chemical potential; the
solver owner controls bath fitting and impurity solution. The primal iteration and
`NonlinearSystemProblem` call the same `dmft_physical_residual`. Implicit derivatives
are rejected unless the result converged and independent smooth-bath-branch and
Jacobian evidence pass; the native ED provider declares itself non-differentiable, so
it fails closed.

Scalar fermionic maximum-entropy reconstruction has a separate typed profile with a
non-negative density and explicit zeroth/first moment checks. It does not claim matrix,
multiorbital, cluster, Nambu, continuous-bath, or general analytic-continuation
support. Finite-bath convergence is evidence for the chosen bath only and is not a
thermodynamic-limit claim.

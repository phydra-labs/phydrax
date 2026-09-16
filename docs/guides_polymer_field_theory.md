# Polymer field theory

`phydrax.applications.polymer_field_theory` implements periodic incompressible Gaussian-chain SCFT and bounded fluctuating-field execution on the existing Fourier, nonlinear, continuation, implicit-root, symmetry, and complex-Langevin substrates.

## Model and contour architecture

`PolymerContourArchitecturePlan` is a rooted acyclic contour tree. Blocks carry a monomer species, positive contour fraction, and fixed contour-step count. It is not a molecular topology, reaction graph, or PRISM site mixture. `IncompressibleGaussianMixturePlan` binds component fractions, polymerization indices, segment lengths, and one symmetric χN matrix.

Prepared propagation uses directed tree messages. Every block receives complementary propagators from both endpoints; junctions multiply all incoming messages except the outgoing edge. Linear chains are the one-path special case.

## Contour integration and gauge

Two named contour methods are available:

- second-order symmetric Strang splitting;
- fourth-order Richardson-extrapolated Strang splitting.

The latter is not mislabeled as ETDRK4. Diffusion uses the prepared Fourier Laplacian spectrum and exact block-boundary steps.

The field root contains M − 1 exchange equations plus incompressibility. The incompressibility zero mode is replaced by the mean common-field gauge residual, giving one square, gauge-fixed system. `solve_scft` uses native Newton–Krylov. `solve_scft_implicit` exposes JVP/VJP only for a successful isolated fixed branch.

Results retain species densities, component partition functions, free energy under the declared dimensionless convention, incompressibility and gauge residuals, minimum propagator magnitude, nonlinear evidence, and symmetry defects.

## Continuation, cells, and symmetry

`continue_scft_interactions` follows a scalar interaction multiplier with native rollback and endpoint enforcement. It does not perform automatic phase discovery.

`solve_isotropic_cell_scft` augments the field root with the derivative of free energy with respect to log isotropic cell scale. Spectral mode indices stay fixed while reciprocal diffusion factors vary differentiably. General shear cells are not claimed.

Declared `TensorSpectralSymmetry` actions may project initial fields and report fixed-subspace defects. Iterations are not silently projected, so symmetry-breaking branches remain representable.

## Fluctuating field theory

`PartialSaddleFTSPlan` evolves real exchange fields with addressed noise while relaxing the common incompressibility field conditionally. Results retain fixed-capacity samples, free energies, incompressibility norms, final replay state, and an exact claim string.

`ComplexFTSPlan` binds the holomorphic polymer-field action to the existing bounded `ComplexLangevinPlan`. Drift-tail, finiteness, state-norm, trajectory, and gauge residual evidence are retained. A finite successful trajectory is not a thermodynamic-limit or ergodicity claim.

All topology, grid, contour, symmetry, branch, and cell-chart choices remain frozen differentiation boundaries.

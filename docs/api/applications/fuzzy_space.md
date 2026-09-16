# Fuzzy-space quantum models

::: phydrax.applications.fuzzy_space

The initial fuzzy-space support tuple is deliberately narrow: two identical
particles in the lowest Landau level on a fuzzy two-sphere.

`FuzzySphereTwoParticlePlan` declares doubled monopole flux, bosonic or
fermionic exchange statistics, every allowed total-pair-spin pseudopotential,
and hard product/sector/matrix capacities. `SU2CouplingTreePlan` constructs the
left-associated Condon–Shortley basis from native Clebsch–Gordan tables. The
fuzzy-sphere preparation selects exactly the exchange-compatible total-spin
multiplets and constructs the rotationally invariant pseudopotential
Hamiltonian.

Evidence includes:

- coupled-basis orthonormality;
- physical-projector idempotence;
- complete symmetric or antisymmetric Hilbert-space dimension;
- commutation with total spin squared;
- finite spectrum with explicit doubled spin, magnetic projection and
  degeneracy labels.

The product-space dense matrix is an admitted finite reference, not an ambient
many-body fallback. No DMRG, thermodynamic, fuzzy-to-continuum, operator-state
correspondence, or CFT identification is inferred.

`tools/fuzzy_space_qualification.py` retains exact SU(2) transforms and small
boson/fermion spectra. `benchmarks/fuzzy_space.py` measures preparation and
spectrum costs over flux and statistics while recording symmetry residuals and
logical bytes.

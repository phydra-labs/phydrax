# Dark-matter production sources and rights ledger

This ledger records scientific-method references. It does not grant redistribution
rights to code, tables, trained weights, profiles or catalogs; runtime artifacts still
require `ReferenceArtifactManifest` admission.

## Wave dark matter

- Schive, Tsai and Chiueh, cosmological wave-dark-matter simulations and adaptive mesh
  refinement: https://arxiv.org/abs/1407.7762
- Mocz et al., wave versus fluid numerical limitations and interference:
  https://arxiv.org/abs/1810.01915
- AxioNyx, pseudospectral root and finite-difference AMR wave evolution:
  https://arxiv.org/abs/2007.08256
- GAMER-2 scaling and AMR runtime: https://arxiv.org/abs/1712.07070
- Modern wave/Hamilton--Jacobi hybrid method and its phase/interface limitations:
  https://arxiv.org/abs/2411.17288

Phydrax implements an independently designed global Fourier profile and a separate
volume-paired finite-difference/composite-AMR Cayley profile. It does not copy upstream
execution stacks or claim Hamilton--Jacobi/wave conversion.

## Rare and frequent SIDM

- Rocha et al., cosmological rare elastic SIDM implementation and halo benchmarks:
  https://arxiv.org/abs/1208.3025
- Koda and Shapiro, Monte Carlo SIDM and gravothermal comparison:
  https://arxiv.org/abs/1101.3097
- Robertson et al., anisotropic differential scattering in halo mergers:
  https://arxiv.org/abs/1612.03906
- Fischer et al., frequent pairwise drag--diffusion SIDM:
  https://arxiv.org/abs/2012.10277
- Fischer et al., anisotropic rare/frequent comparisons:
  https://arxiv.org/abs/2205.02243
- Balberg, Shapiro and Inagaki, gravothermal fluid closure:
  https://arxiv.org/abs/astro-ph/0110561

Phydrax keeps rare, frequent and spherical-fluid profiles separate and requires the full
differential kernel or declared transport moments.

## Inelastic and multistate SIDM

- Schutz and Slatyer, excited-state and exothermic SIDM phenomenology:
  https://arxiv.org/abs/1409.2867
- Vogelsberger et al., inelastic SIDM halo simulations:
  https://arxiv.org/abs/1805.03203
- Multilevel/dissipative dark-sector scattering examples:
  https://arxiv.org/abs/1709.06577

The admitted Phydrax reaction profile is bounded reversible nonrelativistic `2 <-> 2`
with explicit detailed balance, dynamic mass and a fixed-capacity radiation ledger.

## Runtime and external products

Existing external matter-power, cascade, stellar/terrestrial-profile and catalog
products retain their own upstream package, release, checksum and data-license manifests.
Scientific citation here is never substituted for an artifact-rights decision.

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral support tuples for bounded production chemistry profiles."""

from __future__ import annotations

from ..qualification import SupportTuple


def production_chemistry_support_tuples() -> tuple[SupportTuple, ...]:
    """Return the exact released support coordinates; this is not gate evidence."""

    return (
        SupportTuple(
            "chemistry.ao-integrals",
            {
                "angular_momentum": "s",
                "basis_capacity": 16,
                "contracted": True,
                "dense_eri": True,
                "periodic": False,
            },
        ),
        SupportTuple(
            "chemistry.scf.molecular-hf",
            {
                "reference": "rhf",
                "basis": "contracted-s-gaussian",
                "force_route": "central-finite-difference",
                "float64": True,
                "periodic": False,
            },
        ),
        SupportTuple(
            "chemistry.dft.response",
            {
                "functional": "slater-exchange-lda",
                "grid": "fixed-cartesian",
                "response": "static-finite-field",
                "reference": "restricted",
            },
        ),
        SupportTuple(
            "chemistry.vibration.constrained",
            {
                "constraint_hessian": "lagrangian",
                "constraint_types": "distance-cartesian-angle-dihedral",
                "periodic": False,
                "rigid_modes": "tangent-only",
            },
        ),
        SupportTuple(
            "chemistry.reaction-path",
            {
                "method": "improved-tangent-ci-neb",
                "optimizer": "fire",
                "irc": "mass-weighted-steepest-descent",
                "fixed_image_count": True,
            },
        ),
        SupportTuple(
            "chemistry.qmmm",
            {
                "region": "fixed",
                "link_atoms": "affine",
                "mechanical": True,
                "electrostatic": True,
                "polarizable": False,
                "periodic": False,
            },
        ),
        SupportTuple(
            "chemistry.spectroscopy.raman",
            {
                "theory": "nonresonant-placzek",
                "response": "static-polarizability",
                "harmonic": True,
                "relative_intensity": True,
            },
        ),
        SupportTuple(
            "chemistry.excited-manifold",
            {
                "method": "real-symmetric-tda",
                "spin_sector": "singlet",
                "spatial_symmetry": "unsupported",
                "tracking": "hungarian-subspace",
                "transition_property": "electric-dipole",
                "nac": "aligned-finite-difference",
            },
        ),
        SupportTuple(
            "chemistry.spectroscopy.uv-visible",
            {
                "profile": "vertical-electric-dipole",
                "axes": "energy-wavenumber-wavelength",
                "line_shapes": "gaussian-lorentzian",
                "vibronic": False,
            },
        ),
    )


def periodic_chemistry_support_tuples() -> tuple[SupportTuple, ...]:
    """Return maturity-neutral, physics-local periodic chemistry coordinates."""

    return (
        SupportTuple(
            "chemistry.periodic.pencil.orthonormal",
            {
                "basis": "ordered-localized-orbitals",
                "gauge": "explicit-lattice-or-atomic",
                "hamiltonian": "periodic-translation-family",
                "overlap": "identity",
                "units": "explicit",
            },
        ),
        SupportTuple(
            "chemistry.periodic.pencil.generalized",
            {
                "basis": "ordered-localized-orbitals",
                "cross_k_connection": "explicit",
                "gauge": "explicit-lattice-or-atomic",
                "hamiltonian": "periodic-translation-family",
                "overlap": "positive-definite-periodic-family",
                "units": "explicit",
            },
        ),
        SupportTuple(
            "chemistry.periodic.spectrum.bands",
            {
                "eigensystem": "dense-hermitian-pencil",
                "gauge": "basis-defined",
                "overlap": "orthonormal-or-positive-definite",
            },
        ),
        SupportTuple(
            "chemistry.periodic.spectrum.dos-pdos",
            {
                "broadening": "deterministic-named-kernel",
                "projector": "named-metric-projector",
                "weights": "reciprocal-mesh",
            },
        ),
        SupportTuple(
            "chemistry.periodic.spectrum.fermi-surface",
            {
                "connectivity": "fixed-regular-mesh",
                "failure": "unresolved-or-lifshitz-cell",
                "statistics": "fermion",
            },
        ),
        SupportTuple(
            "chemistry.periodic.topology.wilson-zak",
            {
                "bundle": "cross-k-overlap",
                "invariant": "wilson-loop-and-zak-phase",
                "qualification": "gap-link-refinement",
            },
        ),
        SupportTuple(
            "chemistry.periodic.topology.first-chern",
            {
                "bundle": "cross-k-overlap",
                "invariant": "first-chern",
                "qualification": "gap-link-mesh",
            },
        ),
        *(
            SupportTuple(
                "chemistry.periodic.finite.realization",
                {
                    "boundary": boundary,
                    "disorder": "prescribed-fixed-support",
                    "source": "periodic-one-particle-pencil",
                },
            )
            for boundary in ("open", "periodic", "twisted", "slab")
        ),
        SupportTuple(
            "chemistry.interchange.wannier90-hr",
            {
                "format": "hr",
                "lowering": "periodic-hamiltonian-family",
                "provenance": "required-source-context",
            },
        ),
        SupportTuple(
            "chemistry.interchange.wannier90-mmn",
            {
                "connectivity": "prepared-reciprocal-connectivity",
                "format": "mmn",
                "lowering": "raw-cross-k-overlap",
                "provenance": "required-source-context",
            },
        ),
        SupportTuple(
            "chemistry.periodic.electrostatics.ewald-neutral",
            {
                "boundary": "three-dimensional-periodic",
                "charge": "neutral",
                "route": "prepared-ewald",
            },
        ),
        SupportTuple(
            "chemistry.periodic.electrostatics.ewald-background",
            {
                "background": "homogeneous",
                "boundary": "three-dimensional-periodic",
                "route": "prepared-ewald",
            },
        ),
        SupportTuple(
            "chemistry.periodic.pseudopotential.gth-components",
            {
                "components": "local-and-separable-nonlocal",
                "source_manifest": "required",
                "units": "explicit",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.ao-hubbard-orthonormal-insulator-restricted",
            {
                "basis": "orthonormal",
                "occupation": "zero-smearing-gapped",
                "reference": "restricted",
                "self_consistency": "diagonal-hubbard",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.ao-hubbard-generalized-insulator-restricted",
            {
                "basis": "positive-definite-overlap",
                "occupation": "zero-smearing-gapped",
                "reference": "restricted",
                "self_consistency": "diagonal-hubbard",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.ao-hubbard-orthonormal-metal-restricted",
            {
                "basis": "orthonormal",
                "occupation": "finite-temperature-fermi-dirac",
                "reference": "restricted",
                "self_consistency": "diagonal-hubbard",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.ao-hubbard-orthonormal-metal-collinear",
            {
                "basis": "orthonormal",
                "occupation": "finite-temperature-fermi-dirac",
                "reference": "collinear",
                "self_consistency": "diagonal-hubbard",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.gamma-gdf-rhf",
            {
                "eri": "governed-supplied-density-factors",
                "k_sampling": "gamma",
                "reference": "restricted-hartree-fock",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.provider-scalar-relativistic",
            {
                "binding": "exact-method-basis-pseudopotential-spin-kmesh",
                "provenance": "provider-request-input-build-rights",
                "relativity": "scalar",
            },
        ),
        SupportTuple(
            "chemistry.periodic.derivatives.stationary-force-stress",
            {
                "ledger": "stationary-total-or-free-energy",
                "terms": "pulay-entropy-nonlocal-cell",
                "verification": "directional-closure",
            },
        ),
        SupportTuple(
            "chemistry.periodic.scf.gamma-local-gth-lda-x",
            {
                "density_fitting": "gamma-fftdf-local-only",
                "functional": "lda-exchange-only",
                "reference": "restricted",
            },
        ),
        SupportTuple(
            "chemistry.periodic.many-body.diagonal-self-energy",
            {
                "kernel": "provider-supplied",
                "postprocessing": "diagonal-quasiparticle-root",
                "provenance": "retained",
            },
        ),
        SupportTuple(
            "chemistry.periodic.many-body.supplied-bse",
            {
                "kernel": "provider-supplied-transition-kernel",
                "postprocessing": "bounded-bethe-salpeter-eigensystem",
                "provenance": "retained",
            },
        ),
    )


def candidate_complete_chemistry_support_tuples() -> tuple[SupportTuple, ...]:
    """Return candidate coordinates without asserting release or dependencies."""

    return (
        SupportTuple(
            "chemistry.candidate.ao-integrals",
            {
                "angular_momentum": "cartesian-general-real-spherical-lte-12",
                "one_electron": "overlap-kinetic-nuclear-multipole",
                "two_electron": "dense-hermite",
                "factorization": "density-fitting-pivoted-cholesky",
                "derivatives": "first-second-autodiff",
                "ecp": "provider-boundary",
            },
        ),
        SupportTuple(
            "chemistry.candidate.mean-field",
            {
                "molecular_references": "rhf-uhf-rohf-ghf",
                "dft": "spin-lda-pbe-hybrid-long-range-custom-meta",
                "response": "cphf-cpks",
                "derivatives": "analytic-gradient-implicit-hessian",
                "solvation": "gb-gk-native-pcm-cosmo-provider",
                "relativistic": "supplied-decoupling-transform",
            },
        ),
        SupportTuple(
            "chemistry.candidate.correlation",
            {
                "native": "rmp2-fci-casci-bounded-casscf",
                "coupled_cluster": "provider-ccsd-ccsd(t)-lambda-checkpoint",
                "open_shell_gradient": "pyscf-uhf-rohf-ccsd(t)",
                "active_space": "selected-ci-dmrg-fciqmc-provider",
            },
        ),
        SupportTuple(
            "chemistry.candidate.excited",
            {
                "response": "tda-tdhf-restricted-adiabatic-tddft",
                "correlated": "adc-eom-cas-provider",
                "tracking": "global-assignment-subspace-polar-alignment",
                "derivatives": "tda-rpa-eigenproblem",
                "dynamics": "fewest-switches-unitary-electronic",
            },
        ),
        SupportTuple(
            "chemistry.candidate.spectroscopy-nuclei",
            {
                "profiles": "gaussian-lorentzian-voigt-five-axes",
                "vibronic": "duschinsky-tensor-quadrature-condon-herzberg-teller",
                "raman": "placzek-khd-roa-provider",
                "anharmonic": "cubic-quartic-vpt2-gvpt2-vscf-vci",
                "nuclear": "hindered-rotor-conformer-ensemble",
            },
        ),
        SupportTuple(
            "chemistry.candidate.reaction-multiscale",
            {
                "coordinates": "redundant-internal-trust-retraction",
                "optimization": "bfgs-eigenvector-following-dimer",
                "paths": "ci-neb-predictor-corrector-irc",
                "kinetics": "tst-wigner-master-equation",
                "embedding": "multipole-mutual-adaptive-periodic-multilevel",
            },
        ),
        *periodic_chemistry_support_tuples(),
    )


__all__ = [
    "candidate_complete_chemistry_support_tuples",
    "periodic_chemistry_support_tuples",
    "production_chemistry_support_tuples",
]

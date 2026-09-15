#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral support tuples for bounded production chemistry profiles."""

from __future__ import annotations

from ..qualification import SupportTuple
from ._support import ChemistrySupportEdge, ChemistrySupportRegistry


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
        SupportTuple(
            "chemistry.scf.periodic",
            {
                "dimensions": 3,
                "charge": "neutral",
                "spin": "unpolarized",
                "model": "supplied-ao-hubbard",
                "zero_smearing": "insulator-only",
                "k_points": True,
                "smearing": "fermi-dirac",
            },
        ),
    )


def candidate_complete_chemistry_support_tuples() -> tuple[SupportTuple, ...]:
    """Return bounded candidate coordinates; these are not released profiles."""

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
        SupportTuple(
            "chemistry.candidate.periodic-electronic",
            {
                "electrostatics": "ewald-gth",
                "mean_field": "fftdf-gamma-gdf-hf-hybrid-kpoint-spin-metal",
                "derivatives": "autodiff-forces-stress",
                "properties": "bands-berry-wannier-defect",
                "provider": "task-bound-external-reference",
            },
        ),
        SupportTuple(
            "chemistry.candidate.periodic-lattice",
            {
                "force_constants": "central-supercell-asr",
                "phonons": "qpoint-lo-to",
                "thermodynamics": "harmonic-qha",
                "transport": "three-phonon-rta",
                "many_body": "diagonal-gw-bse",
            },
        ),
    )


def candidate_complete_chemistry_support_registry() -> ChemistrySupportRegistry:
    """Return dependency structure for candidate, not released, support."""

    support = candidate_complete_chemistry_support_tuples()
    by_capability = {value.capability: value for value in support}
    dependency_pairs = (
        (
            "chemistry.candidate.mean-field",
            "chemistry.candidate.ao-integrals",
        ),
        (
            "chemistry.candidate.correlation",
            "chemistry.candidate.mean-field",
        ),
        (
            "chemistry.candidate.excited",
            "chemistry.candidate.mean-field",
        ),
        (
            "chemistry.candidate.spectroscopy-nuclei",
            "chemistry.candidate.excited",
        ),
        (
            "chemistry.candidate.reaction-multiscale",
            "chemistry.candidate.mean-field",
        ),
        (
            "chemistry.candidate.periodic-lattice",
            "chemistry.candidate.periodic-electronic",
        ),
    )
    return ChemistrySupportRegistry(
        support,
        tuple(
            ChemistrySupportEdge(
                by_capability[source].support_tuple_id,
                by_capability[dependency].support_tuple_id,
            )
            for source, dependency in dependency_pairs
        ),
    )


def production_chemistry_support_registry() -> ChemistrySupportRegistry:
    """Return the exact dependency graph over released bounded profiles."""

    support = production_chemistry_support_tuples()
    by_capability = {value.capability: value for value in support}
    dependency_pairs = (
        ("chemistry.scf.molecular-hf", "chemistry.ao-integrals"),
        ("chemistry.dft.response", "chemistry.ao-integrals"),
        ("chemistry.excited-manifold", "chemistry.scf.molecular-hf"),
        ("chemistry.spectroscopy.raman", "chemistry.dft.response"),
        ("chemistry.spectroscopy.raman", "chemistry.vibration.constrained"),
        ("chemistry.spectroscopy.uv-visible", "chemistry.excited-manifold"),
    )
    return ChemistrySupportRegistry(
        support,
        tuple(
            ChemistrySupportEdge(
                by_capability[source].support_tuple_id,
                by_capability[dependency].support_tuple_id,
            )
            for source, dependency in dependency_pairs
        ),
    )


__all__ = [
    "candidate_complete_chemistry_support_registry",
    "candidate_complete_chemistry_support_tuples",
    "production_chemistry_support_registry",
    "production_chemistry_support_tuples",
]

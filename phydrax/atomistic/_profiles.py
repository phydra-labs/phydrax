#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact unreleased candidate envelopes for the native atomistic capabilities.

Every capability below is a closure exception under ``docs/CAPABILITY_LIFECYCLE.md``:
its declaration names the scientific owner, exact support tuples, capability
dependencies, evidence gates and nonclaims. A support tuple states what an
independent campaign must qualify. It is not measured success, source-checkpoint
fidelity, chemical accuracy, accelerator or multi-host evidence, or release
authority, and no profile here may be released by this module.

Standard MACE stays owned by :mod:`._mace_profiles`; the attention/nonlinear,
joint-embedding and latent-Ewald MACE families are separate capabilities that
depend on it. The crystal domain is owned by :mod:`phydrax.atomistic.crystal`.
:mod:`._capability_closure` binds these capabilities to the C01–C22 closure
obligations without copying their support tuples.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product

from ..qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ..qualification._registry import CapabilityProfile, SupportTuple, SupportValue


CLOSURE_EXCEPTION = "native-atomistic-capabilities"

_GEOMETRIES = ("finite", "orthorhombic", "triclinic", "partial-periodic")
_PERIODIC = ("orthorhombic", "triclinic", "partial-periodic")
_BULK = ("orthorhombic", "triclinic")
_PRECISIONS = ("float32", "float64")
_STRESS_CONVENTION = "tensile-positive-strain-gradient-per-cell-volume"

_BASE: dict[str, SupportValue] = {
    "closure-exception": CLOSURE_EXCEPTION,
    "device": "cpu",
    "provider": "phydrax-native",
    "release-claim": False,
}

_COMMON_GATES = (
    "compiler-resource-envelope",
    "implementation-complete",
    "independent-review",
    "numerical-reference",
    "operations-restart",
    "rights-and-security",
)
_SCIENTIFIC_GATES = (*_COMMON_GATES, "derivative-reference", "scientific-validation")
_MODEL_GATES = (
    *_SCIENTIFIC_GATES,
    "capacity-scaling",
    "source-checkpoint-fidelity",
    "weights-rights-admission",
)
_DISTRIBUTED_GATES = (
    *_COMMON_GATES,
    "actual-multi-process-execution",
    "derivative-reference",
    "exact-reverse-communication",
    "physical-multi-device",
    "physical-multi-host",
    "single-device-reference-parity",
)

_COMMON_NONCLAIMS = (
    "no-external-numerical-provider-execution",
    "not-release-authorized",
    "registered-profile-is-not-retained-evidence",
)


def _support(capability: str, attributes: Mapping[str, SupportValue], /) -> SupportTuple:
    return SupportTuple(capability, {**_BASE, **attributes})


def _profile(
    capability: str,
    route: str,
    supports: Iterable[Mapping[str, SupportValue]],
    gates: Iterable[str],
    /,
) -> CapabilityProfile:
    return CapabilityProfile(
        f"{capability}.{route}.profile",
        "phydrax",
        "candidate",
        tuple(_support(capability, attributes) for attributes in supports),
        required_gates=tuple(sorted(set(gates))),
        released=False,
    )


def _model_derivatives(geometry: str, /) -> tuple[str, ...]:
    derivatives = ("energy-force", "mixed-parameter", "coordinate-hvp")
    return derivatives if geometry == "finite" else (*derivatives, "energy-force-stress")


def _stress(geometry: str, /) -> str:
    return "energy-force" if geometry == "finite" else "energy-force-stress"


# P1/P2: canonical neighborhood resources, segmented reductions, splines.

NEIGHBORHOOD_CAPABILITY = "atomistic.neighborhood-resources"
_NEIGHBORHOOD_ROLES = (
    ("classical-minimum-image-pairs", "cutoff-radius"),
    ("directed-image-graph", "cutoff-radius"),
    ("directed-image-graph", "nearest-neighbor-band"),
    ("image-pair-sum", "cutoff-radius"),
    ("coordination", "cutoff-radius"),
    ("interaction-site-pairs", "cutoff-radius"),
)
_TILED_REFERENCE: dict[str, SupportValue] = {
    "work": "all-pairs-reference",
    "tile-workspace": "explicit-receiver-source-candidate-bound",
    "scalar-tiles": True,
}


def _neighborhood_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        _profile(
            NEIGHBORHOOD_CAPABILITY,
            search,
            (
                {
                    "role": role,
                    "selection": selection,
                    "search": search,
                    "geometry": geometry,
                    "precision": precision,
                    "layout": "fixed-capacity-image-edge-list",
                    "cutoff-tiers": "strictly-increasing-independent-capacity",
                    "derivative": "fixed-topology-piecewise-geometry",
                    "capacity-failure": "distinct-from-scientific-failure",
                    "rebuild": "batched-search-selected-case-merge",
                    "lifecycle": "candidate-accept-rebase-checkpoint",
                    "distribution": "single-device-batched",
                    **(_TILED_REFERENCE if search == "tiled-reference" else {}),
                }
                for (role, selection), geometry, precision in product(
                    _NEIGHBORHOOD_ROLES, _GEOMETRIES, _PRECISIONS
                )
            ),
            (
                *_COMMON_GATES,
                "capacity-scaling",
                "derivative-reference",
                "exact-route-multiset-reference",
                "source-sharing-compatibility",
                "stale-state-refusal",
            ),
        )
        for search in ("dense", "tiled-reference", "cell-list", "cluster-tile")
    )


ROUTE_REDUCTION_CAPABILITY = "atomistic.segmented-route-reduction"
_ROUTE_OPERATIONS = (
    ("sum", ("fast", "deterministic", "compensated"), "smooth-jvp-vjp"),
    ("mean", ("fast", "deterministic", "compensated"), "smooth-jvp-vjp"),
    ("max", ("fast", "deterministic"), "piecewise-selection"),
    ("min", ("fast", "deterministic"), "piecewise-selection"),
    ("dot", ("fast", "deterministic", "compensated"), "smooth-jvp-vjp"),
    ("multiply", ("fast", "deterministic", "compensated"), "smooth-jvp-vjp"),
    ("matvec", ("fast", "deterministic", "compensated"), "smooth-jvp-vjp"),
    ("rms", ("fast", "deterministic", "compensated"), "nan-at-available-zero-segment"),
)


def _route_reduction_profiles() -> tuple[CapabilityProfile, ...]:
    return (
        _profile(
            ROUTE_REDUCTION_CAPABILITY,
            "native",
            (
                {
                    "operation": operation,
                    "accumulation": accumulation,
                    "precision": precision,
                    "derivative": derivative,
                    "empty-segment": "unavailable-inert-zero",
                    "relations": "edge-and-row-relations",
                }
                for operation, accumulations, derivative in _ROUTE_OPERATIONS
                for accumulation, precision in product(accumulations, _PRECISIONS)
            ),
            (*_COMMON_GATES, "accumulation-determinism", "derivative-reference"),
        ),
    )


SPLINE_CAPABILITY = "atomistic.particle-grid-spline"


def _spline_profiles() -> tuple[CapabilityProfile, ...]:
    degrees: tuple[tuple[str, int], ...] = (
        *(("uniform", degree) for degree in range(1, 6)),
        ("mixed-per-axis-with-zero-axis", 5),
    )
    return (
        _profile(
            SPLINE_CAPABILITY,
            "tensor-bspline",
            (
                {
                    "degree-layout": layout,
                    "maximum-degree": degree,
                    "continuity": "c-degree-minus-one-across-knots",
                    "boundary": boundary,
                    "geometry-ad": geometry_ad,
                    "precision": precision,
                    "operations": "deposit-gather-weight-gradient-trailing-channels",
                    "explicit-derivative-order": 1,
                    "accumulations": "compensated,deterministic,fast",
                }
                for (layout, degree), boundary, geometry_ad, precision in product(
                    degrees,
                    ("periodic-wrap", "bounded-reject", "bounded-drop"),
                    ("piecewise", "frozen"),
                    _PRECISIONS,
                )
            ),
            (
                *_COMMON_GATES,
                "adjoint-identity",
                "derivative-reference",
                "polynomial-reproduction",
            ),
        ),
    )


# P3: long-range electrostatics, multipoles, dispersion and latent Ewald.

ELECTROSTATICS_CAPABILITY = "atomistic.electrostatics"
_EWALD_BOUNDARIES = (
    ("tin-foil", "require-neutral", _BULK, 3),
    ("tin-foil", "uniform-background", _BULK, 3),
    ("vacuum-sphere", "require-neutral", _BULK, 3),
    ("slab-yeh-berkowitz", "require-neutral", ("partial-periodic",), 2),
    ("slab-yeh-berkowitz", "uniform-background", ("partial-periodic",), 2),
)


def _electrostatics_profiles() -> tuple[CapabilityProfile, ...]:
    capability = ELECTROSTATICS_CAPABILITY
    gates = (*_SCIENTIFIC_GATES, "field-response-reference", "unique-image-admission")
    direct = _profile(
        capability,
        "direct-coulomb",
        (
            {
                "method": "direct-coulomb",
                "geometry": "finite",
                "precision": precision,
                "derivative": "energy-force",
                "pair-support": "complete-finite-pairs-no-cutoff",
                "field-response": "differentiated-predicted-charges",
            }
            for precision in _PRECISIONS
        ),
        gates,
    )
    pairwise = _profile(
        capability,
        "screened-pairwise",
        (
            {
                "method": method,
                "truncation": truncation,
                "geometry": geometry,
                "precision": precision,
                "derivative": _stress(geometry),
                "image-convention": "minimum-image-below-unique-image-radius",
                "field-response": "differentiated-predicted-charges",
                "stress-convention": _STRESS_CONVENTION,
            }
            for (method, truncation), geometry, precision in product(
                (
                    ("screened-coulomb", "hard"),
                    ("screened-coulomb", "shifted-potential"),
                    ("screened-coulomb", "shifted-force"),
                    ("damped-shifted-force", "fennell-gezelter-self-corrected"),
                ),
                ("finite", *_BULK),
                _PRECISIONS,
            )
        ),
        gates,
    )
    ewald_rows = tuple(
        (boundary, background, geometry, axes)
        for boundary, background, geometries, axes in _EWALD_BOUNDARIES
        for geometry in geometries
    )
    common_periodic: dict[str, SupportValue] = {
        "derivative": "energy-force-stress",
        "stress-convention": _STRESS_CONVENTION,
        "partial-periodic-volume": "explicit-three-dimensional-embedding",
        "field-response": "differentiated-predicted-charges",
        "splitting-estimate": "uncertified-gaussian-tail-heuristic",
    }
    ewald = _profile(
        capability,
        "ewald",
        (
            {
                **common_periodic,
                "method": "ewald",
                "boundary": boundary,
                "background": background,
                "geometry": geometry,
                "periodic-axes": axes,
                "precision": precision,
                "real-space-images": "explicit-image,minimum-image",
            }
            for (boundary, background, geometry, axes), precision in product(
                ewald_rows, _PRECISIONS
            )
        ),
        (*gates, "independent-converged-lattice-reference", "slab-correction-reference"),
    )
    pme = _profile(
        capability,
        "particle-mesh-ewald",
        (
            {
                **common_periodic,
                "method": "particle-mesh-ewald",
                "boundary": boundary,
                "background": background,
                "geometry": geometry,
                "periodic-axes": axes,
                "precision": precision,
                "influence": influence,
                "maximum-assignment-degree": 5,
                "nyquist": "excluded,symmetrized",
                "real-space-images": "explicit-image,minimum-image",
                "reciprocal-budget": "bounded-grid-modes-bytes",
            }
            for (boundary, background, geometry, axes), precision, influence in product(
                ewald_rows,
                _PRECISIONS,
                ("discrete-bspline-modulus", "continuous-sinc-window"),
            )
        ),
        (
            *gates,
            "independent-converged-lattice-reference",
            "mesh-convergence",
            "slab-correction-reference",
        ),
    )
    return direct, pairwise, ewald, pme


MULTIPOLE_CAPABILITY = "atomistic.multipole-electrostatics"


def _multipole_profiles() -> tuple[CapabilityProfile, ...]:
    routes = (
        ("finite", "finite"),
        ("ewald", "orthorhombic"),
        ("ewald", "triclinic"),
        ("particle-mesh", "orthorhombic"),
        ("particle-mesh", "triclinic"),
    )
    return (
        _profile(
            MULTIPOLE_CAPABILITY,
            "native",
            (
                {
                    "density": density,
                    "route": route,
                    "geometry": geometry,
                    "precision": precision,
                    "maximum-degree": 2,
                    "bases": (
                        "cartesian-primitive,native-real-orthonormal,"
                        "nvalchemi-ops-e3nn-component"
                    ),
                    "trace-policies": "project,retain",
                    "intrasite": "exclude" if density == "point" else "exclude,include",
                    "boundary": "none" if geometry == "finite" else "tin-foil",
                    "derivative": _stress(geometry),
                    "field-response": "differentiated-predicted-moments",
                }
                for density, (route, geometry), precision in product(
                    ("point", "gaussian"), routes, _PRECISIONS
                )
            ),
            (
                *_SCIENTIFIC_GATES,
                "basis-conversion-reference",
                "field-response-reference",
            ),
        ),
    )


DISPERSION_CAPABILITY = "atomistic.dispersion-d3"


def _dispersion_profiles() -> tuple[CapabilityProfile, ...]:
    tables: dict[str, SupportValue] = {
        "reference-tables": "caller-held-digest-authenticated",
        "three-body": False,
        "damping": "becke-johnson-c6-c8",
    }
    gates = (
        *_SCIENTIFIC_GATES,
        "d3-table-rights-admission",
        "regularity-locus-reference",
        "element-table-fidelity",
    )
    conventional = _profile(
        DISPERSION_CAPABILITY,
        "conventional",
        (
            {
                **tables,
                "method": "d3-bj-conventional-cn",
                "interpolation": interpolation,
                "dispersion-cutoff": switch,
                "geometry": geometry,
                "precision": precision,
                "derivative": _stress(geometry),
                "stress-convention": _STRESS_CONVENTION,
                "regularity": "smooth-away-from-cn-cutoff-threshold-coincidence",
            }
            for interpolation, switch, geometry, precision in product(
                ("normalized-gaussian", "nvalchemi-relative-threshold"),
                ("hard-cutoff", "s5-switch"),
                _GEOMETRIES,
                _PRECISIONS,
            )
        ),
        gates,
    )
    fourier = _profile(
        DISPERSION_CAPABILITY,
        "fourier-d3",
        (
            {
                **tables,
                "method": "fourier-d3-modified-cn",
                "c6-factorization": "signed-low-rank-maximum-relative-entry-error",
                "zero-mode": "included",
                "dispersion-cutoff": "none",
                "geometry": geometry,
                "periodic-axes": 3,
                "precision": precision,
                "derivative": "energy-force-stress",
                "stress-convention": _STRESS_CONVENTION,
                "regularity": "c1-transition-cutoff-step-spline-c-degree-minus-one",
            }
            for geometry, precision in product(_BULK, _PRECISIONS)
        ),
        (*gates, "mesh-convergence"),
    )
    return conventional, fourier


LATENT_CAPABILITY = "atomistic.latent-electrostatics"


def _latent_profiles() -> tuple[CapabilityProfile, ...]:
    rows = (
        *(
            ("finite", "finite", "not-applicable"),
            ("per-case", "finite", "not-applicable"),
        ),
        *(
            (branch, geometry, support)
            for branch, geometry, support in product(
                ("per-case", "periodic"),
                _PERIODIC,
                ("complete-sphere", "index-box-sphere"),
            )
        ),
    )
    return (
        _profile(
            LATENT_CAPABILITY,
            "native",
            (
                {
                    "method": "latent-ewald-gaussian",
                    "branch": branch,
                    "geometry": geometry,
                    "mode-support": support,
                    "precision": precision,
                    "zero-mode": "excluded-no-background",
                    "periodic-sum": "full-cell-three-dimensional",
                    "derivative": _stress(geometry),
                    "regularity": "almost-everywhere-at-mode-crossings",
                    "observables": "born-charges,latent-charges",
                }
                for (branch, geometry, support), precision in product(rows, _PRECISIONS)
            ),
            (
                *_SCIENTIFIC_GATES,
                "les-source-rights-admission",
                "source-reference-parity",
            ),
        ),
    )


# P4: native model families and source-artifact interchange.

AIMNET2_CAPABILITY = "atomistic.aimnet2-inference"


def _aimnet2_profiles() -> tuple[CapabilityProfile, ...]:
    channels = (
        (1, "single-channel", "continuous-total-charge"),
        (2, "neutral-spin-equilibrated", "continuous-total-charge-and-multiplicity"),
    )
    return (
        _profile(
            AIMNET2_CAPABILITY,
            "native-exact",
            (
                {
                    "architecture": "aimnet2",
                    "charge-channels": count,
                    "charge-mode": mode,
                    "conditioning": conditioning,
                    "geometry": geometry,
                    "precision": precision,
                    "derivative": derivative,
                    "maximum-atomic-number": 63,
                    "source-stages": (
                        "dipole,dispersion-parameters,quadrupole,short-range-coulomb,"
                        "short-range-repulsion,ts-dispersion"
                    ),
                    "embedded-long-range": "externalized-explicit-program-terms",
                    "periodic-charge-moments": "unavailable",
                    "higher-order-regularity": "explicit-almost-everywhere-policy",
                    "ts-derivative": "requires-positive-dispersion-parameter-dataflow",
                    "image-topology": "explicit-stable-directed-shifts",
                    "stress-convention": _STRESS_CONVENTION,
                }
                for (count, mode, conditioning), geometry, precision in product(
                    channels, _GEOMETRIES, _PRECISIONS
                )
                for derivative in _model_derivatives(geometry)
            ),
            (*_MODEL_GATES, "charge-equilibration-reference"),
        ),
    )


UMA_CAPABILITY = "atomistic.uma-inference"


def _uma_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "architecture": "uma-escn-md",
        "conditioning": "categorical-total-charge-spin-index-task",
        "selection": "nearest-neighbor-squared-distance-closed-band",
        "composition-rules": "mean,sum-over-count-plus-one",
        "preparation-x64": True,
        "image-topology": "explicit-stable-directed-shifts",
    }
    backbones = (("escnmd", "none"), ("escnmd-moe", "so2,so2m0"))
    potential = _profile(
        UMA_CAPABILITY,
        "potential",
        (
            {
                **common,
                "model-kind": "conservative-potential",
                "backbone": backbone,
                "mole-coefficients": mole,
                "geometry": geometry,
                "precision": precision,
                "derivative": derivative,
                "higher-order-regularity": "piecewise-almost-everywhere",
                "stress-convention": _STRESS_CONVENTION,
            }
            for (backbone, mole), geometry, precision in product(
                backbones, _GEOMETRIES, _PRECISIONS
            )
            for derivative in _model_derivatives(geometry)
        ),
        (*_MODEL_GATES, "nearest-neighbor-selection-reference"),
    )
    property_model = _profile(
        UMA_CAPABILITY,
        "direct-property",
        (
            {
                **common,
                "model-kind": "property-only",
                "backbone": backbone,
                "mole-coefficients": mole,
                "geometry": geometry,
                "precision": precision,
                "derivative": "none",
                "outputs": "direct-forces,direct-stress,reported-energy",
                "invented-energy": False,
            }
            for (backbone, mole), geometry, precision in product(
                backbones, _GEOMETRIES, _PRECISIONS
            )
        ),
        (
            *_COMMON_GATES,
            "nearest-neighbor-selection-reference",
            "scientific-validation",
            "source-checkpoint-fidelity",
            "weights-rights-admission",
        ),
    )
    merged = _profile(
        UMA_CAPABILITY,
        "merged-mixture",
        (
            {
                **common,
                "model-kind": "merged-conservative-potential",
                "backbone": "escnmd-moe",
                "mole-coefficients": "so2,so2m0",
                "geometry": geometry,
                "precision": precision,
                "derivative": derivative,
                "parameter-surface": "frozen-merge",
                "binding": "exact-composition-condition-revision",
                "stress-convention": _STRESS_CONVENTION,
            }
            for geometry, precision in product(_GEOMETRIES, _PRECISIONS)
            for derivative in _model_derivatives(geometry)
            if derivative in ("energy-force", "energy-force-stress")
        ),
        (*_MODEL_GATES, "merged-routing-parity", "stale-merge-refusal"),
    )
    return potential, property_model, merged


MACE_EXTENDED_CAPABILITY = "atomistic.mace-extended-inference"


def _mace_extended_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "route": "native-exact",
        "image-topology": "explicit-stable-directed-shifts",
        "basis-identity": "declared-real-o3-layout",
        "parameterization": "original-w-fixed-u",
        "stress-convention": _STRESS_CONVENTION,
        "topology-derivative": "frozen-candidates-and-images",
        "higher-order-regularity": "classical-off-singularities-mixed-third-almost-everywhere",
    }
    interactions = _profile(
        MACE_EXTENDED_CAPABILITY,
        "interaction-families",
        (
            {
                **common,
                "interaction-family": family,
                "interaction": interaction,
                "geometry": geometry,
                "precision": precision,
                "derivative": derivative,
            }
            for (family, interaction), geometry, precision in product(
                (
                    ("attention", "real-agnostic-attention-residual"),
                    ("nonlinear", "real-agnostic-residual-nonlinear"),
                ),
                _GEOMETRIES,
                _PRECISIONS,
            )
            for derivative in _model_derivatives(geometry)
        ),
        _MODEL_GATES,
    )
    embedding = _profile(
        MACE_EXTENDED_CAPABILITY,
        "joint-embedding",
        (
            {
                **common,
                "interaction-family": family,
                "embedding": "joint-categorical-continuous-case-atom",
                "physical-charge-spin": "continuous-only",
                "geometry": geometry,
                "precision": precision,
                "derivative": _stress(geometry) if stress else "energy-force",
            }
            for family, geometry, precision, stress in product(
                ("standard", "attention", "nonlinear"),
                _GEOMETRIES,
                _PRECISIONS,
                (False, True),
            )
            if not (geometry == "finite" and stress)
        ),
        (*_MODEL_GATES, "conditioning-admission-reference"),
    )
    return interactions, embedding


MACELES_CAPABILITY = "atomistic.maceles-inference"


def _maceles_profiles() -> tuple[CapabilityProfile, ...]:
    return (
        _profile(
            MACELES_CAPABILITY,
            "native-exact",
            (
                {
                    "architecture": "scale-shift-mace-latent-ewald",
                    "latent-channels": 1,
                    "latent-branch": branch,
                    "geometry": geometry,
                    "precision": precision,
                    "derivative": derivative,
                    "latent-periodic-sum": "full-cell-three-dimensional",
                    "atom-energy": "excludes-global-latent-component",
                    "observables": "born-charges,global-energy-component,latent-charges",
                    "higher-order-regularity": "almost-everywhere-at-mode-crossings",
                    "stress-convention": _STRESS_CONVENTION,
                }
                for geometry, precision in product(_GEOMETRIES, _PRECISIONS)
                for branch in (
                    "per-case",
                    "finite" if geometry == "finite" else "periodic",
                )
                for derivative in _model_derivatives(geometry)
            ),
            (*_MODEL_GATES, "les-source-rights-admission"),
        ),
    )


INTERCHANGE_CAPABILITY = "atomistic.model-artifact-interchange"


def _interchange_profiles() -> tuple[CapabilityProfile, ...]:
    rows = (
        ("mace", "torch-state-dict", "metadata-admitted-safe-load", "mace-torch-0.3.15"),
        ("mace", "torch-state-dict", "metadata-admitted-safe-load", "mace-torch-0.3.16"),
        ("mace", "torch-full-model", "digest-bound-trusted-pickle", "mace-torch-0.3.15"),
        ("mace", "torch-full-model", "digest-bound-trusted-pickle", "mace-torch-0.3.16"),
        ("aimnet2", "hf-safetensors", "pickle-free", "aimnetcentral-0.2.0"),
        ("aimnet2", "torch-v2-state", "pickle-free", "aimnetcentral-0.2.0"),
        (
            "aimnet2",
            "torch-legacy-jit",
            "digest-bound-trusted-pickle",
            "aimnetcentral-0.2.0",
        ),
        (
            "uma",
            "fairchem-inference-checkpoint",
            "symbolic-allowlist-no-unpickle",
            "fairchem-bb0a062a9a08d8b3ad3995aca5b9765e07804990",
        ),
    )
    return (
        _profile(
            INTERCHANGE_CAPABILITY,
            "source-extraction",
            (
                {
                    "family": family,
                    "source-format": source_format,
                    "load-policy": policy,
                    "source-release": release,
                    "extraction": "caller-pinned-isolated-interpreter",
                    "result": "bounded-pickle-free-array-archive",
                    "artifact-admission": "external-artifact-sha256-size-license",
                    "automatic-download": False,
                    "unsafe-load-fallback": False,
                    "native-runtime-source-import": False,
                }
                for family, source_format, policy, release in rows
            ),
            (
                *_COMMON_GATES,
                "fresh-environment-restore",
                "source-checkpoint-fidelity",
                "weights-rights-admission",
            ),
        ),
    )


# P5/P6: dynamics, relaxation, reaction paths and staged campaigns.

DYNAMICS_CAPABILITY = "atomistic.dynamics"
_FIXED_CELL_METHODS = (
    ("velocity-verlet", "nve"),
    ("baoab-langevin", "nvt"),
    ("nose-hoover-chain", "nvt"),
    ("stochastic-velocity-rescaling", "nvt"),
    ("generalized-langevin", "nvt"),
    ("velocity-rescaling", "nvt-noncanonical-control"),
)
_THERMOSTATS = (
    "baoab-langevin",
    "nose-hoover-chain",
    "stochastic-velocity-rescaling",
    "generalized-langevin",
)


def _dynamics_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "force-source": "conservative-scalar-program",
        "neighborhood-state": "single-bundle-state",
        "lifecycle": "authenticated-full-continuation",
    }
    gates = (
        *_SCIENTIFIC_GATES,
        "ensemble-measure-reference",
        "failure-class-separation",
        "restart-bitwise-continuation",
    )
    fixed = _profile(
        DYNAMICS_CAPABILITY,
        "fixed-cell",
        (
            {
                **common,
                "integrator": method,
                "ensemble": ensemble,
                "cell": "fixed",
                "geometry": geometry,
                "precision": precision,
                "derivative": "energy-force",
                **(
                    {"maximum-chain-length": 8, "yoshida-suzuki-orders": "1,3,5"}
                    if method == "nose-hoover-chain"
                    else {}
                ),
            }
            for (method, ensemble), geometry, precision in product(
                _FIXED_CELL_METHODS, _GEOMETRIES, _PRECISIONS
            )
        ),
        gates,
    )
    monte_carlo = _profile(
        DYNAMICS_CAPABILITY,
        "isotropic-monte-carlo-barostat",
        (
            {
                **common,
                "integrator": method,
                "ensemble": "npt",
                "cell": "isotropic-monte-carlo-log-volume",
                "geometry": geometry,
                "periodic-axes": 3,
                "precision": precision,
                "derivative": "energy-force",
                "constrained-volume-scaling": "molecular-center",
            }
            for method, geometry, precision in product(_THERMOSTATS, _BULK, _PRECISIONS)
        ),
        (*gates, "volume-jacobian-reference"),
    )
    mtk = _profile(
        DYNAMICS_CAPABILITY,
        "mtk-cell",
        (
            {
                **common,
                "integrator": f"mtk-{ensemble}",
                "ensemble": ensemble,
                "cell": "mttk-tuckerman-2006-exponential",
                "cell-chart": chart,
                "geometry": geometry,
                "periodic-axes": 3,
                "precision": precision,
                "derivative": "energy-force-stress",
                "stress-convention": _STRESS_CONVENTION,
                "holonomic-constraints": False,
                "volume-scaling": "atomic",
            }
            for ensemble, chart, geometry, precision in product(
                ("npt", "nph"),
                ("isotropic", "semi-isotropic-xy", "diagonal", "triclinic-symmetric"),
                _BULK,
                _PRECISIONS,
            )
        ),
        (*gates, "strain-pressure-reference"),
    )
    return fixed, monte_carlo, mtk


RELAXATION_CAPABILITY = "atomistic.relaxation"
_OPTIMIZERS = "fire,fire2,trust-scaled-lbfgs"


def _relaxation_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "optimizers": _OPTIMIZERS,
        "grouping": "grouped-stepwise-bounded-history",
        "residuals": "projected-and-full",
        "minimum-sensitivity": "implicit-pcg-converged-stationary-positive-curvature",
    }
    charts = (
        *(("cartesian", "fixed-cell", geometry) for geometry in _GEOMETRIES),
        *(
            (f"deformation-{parameterization}", subspace, geometry)
            for parameterization, subspaces in (
                ("affine", ("symmetric", "hydrostatic", "selected")),
                ("logarithmic", ("symmetric", "hydrostatic", "fixed-volume", "selected")),
            )
            for subspace in subspaces
            for geometry in _BULK
        ),
        *(("symmetry", "symmetry-adapted", geometry) for geometry in _BULK),
    )
    return (
        _profile(
            RELAXATION_CAPABILITY,
            "native",
            (
                {
                    **common,
                    "chart": chart,
                    "strain-subspace": subspace,
                    "geometry": geometry,
                    "precision": precision,
                    "derivative": "energy-force"
                    if chart == "cartesian"
                    else "energy-force-stress",
                    "holonomic-constraints": False,
                }
                for (chart, subspace, geometry), precision in product(charts, _PRECISIONS)
            ),
            (
                *_SCIENTIFIC_GATES,
                "known-minimum-reference",
                "symmetry-preservation",
                "failure-class-separation",
            ),
        ),
    )


REACTION_PATH_CAPABILITY = "atomistic.reaction-path"


def _reaction_path_profiles() -> tuple[CapabilityProfile, ...]:
    rows = (
        ("finite", "explicit"),
        *(
            (geometry, branch)
            for geometry in _BULK
            for branch in ("explicit", "shortest-image")
        ),
    )
    return (
        _profile(
            REACTION_PATH_CAPABILITY,
            "native-band",
            (
                {
                    "method": method,
                    "cell": "fixed",
                    "geometry": geometry,
                    "image-branch": branch,
                    "precision": precision,
                    "optimizers": _OPTIMIZERS,
                    "endpoints": "fixed",
                    "commit": "whole-band-one-iterate",
                    "holonomic-constraints": False,
                }
                for (geometry, branch), method, precision in product(
                    rows, ("nudged-elastic-band", "climbing-image"), _PRECISIONS
                )
            ),
            (*_SCIENTIFIC_GATES, "known-saddle-reference"),
        ),
    )


CAMPAIGN_CAPABILITY = "atomistic.campaign"


def _campaign_profiles() -> tuple[CapabilityProfile, ...]:
    return (
        _profile(
            CAMPAIGN_CAPABILITY,
            "staged",
            (
                {
                    "operation": operation,
                    "transport": transport,
                    "publication": "publish-before-commit-exactly-once",
                    "backpressure": "bounded-parked-queue",
                    "revision-policies": "drain,reprime",
                    "rank-failure": "fail-coupled-restart-from-durable-commit",
                }
                for operation, transport in product(
                    ("dynamics", "relaxation", "reaction-path", "generation", "label"),
                    (
                        "single-process",
                        "multi-process-replicated",
                        "multi-process-process-local",
                    ),
                )
            ),
            (
                *_COMMON_GATES,
                "actual-multi-process-execution",
                "crash-nonpublication",
                "heterogeneous-refill",
            ),
        ),
        _profile(
            CAMPAIGN_CAPABILITY,
            "domain-stages",
            (
                {
                    "operation": "domain-dynamics",
                    "strategy": strategy,
                    "stage-group": "explicit-process-subset-child-group",
                    "transport": "root-fixed-schedule-process-local-domain-packets",
                    "continuation-schema": "adjacent-stages-identical-or-refused",
                    "publication": "semantic-key-idempotent-sink",
                    "backpressure": "park-finished-resident-without-physics",
                    "checkpoint": "one-global-commit",
                    "restore": restore,
                    "rank-failure": "fail-coupled-restart-from-durable-commit",
                }
                for strategy, restore in product(
                    ("halo", "graph-partition"),
                    ("same-topology-bitwise", "changed-topology-migration"),
                )
            ),
            (*_DISTRIBUTED_GATES, "crash-nonpublication", "migration-restart"),
        ),
    )


# P7/P8: training and distillation.

TRAINING_CAPABILITY = "atomistic.training"


def _training_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "residuals": "absolute,euclidean-vector-only,huber,squared",
        "schedules": "constant,cosine,linear,piecewise",
        "composition": "literal,normalized",
        "accumulation": "merged-numerator-support",
        "clipping": "global-norm-after-merge",
        "ema": "parameter-only",
        "fine-tuning": "irrep-low-rank-adaptation,parameter-selection",
        "distribution": "single-placement",
        "lifecycle": "atomic-training-restart",
    }
    targets = (
        *(
            (target, geometry)
            for target in ("energy", "forces", "property")
            for geometry in _GEOMETRIES
        ),
        *(("stress", geometry) for geometry in _PERIODIC),
    )
    return (
        _profile(
            TRAINING_CAPABILITY,
            "streaming",
            (
                {
                    **common,
                    "target": target,
                    "geometry": geometry,
                    "loss-precision": precision,
                    "data-source": source,
                }
                for (target, geometry), precision, source in product(
                    targets,
                    _PRECISIONS,
                    ("in-memory-structure-source", "zarr-v3-structure-store"),
                )
            ),
            (
                *_SCIENTIFIC_GATES,
                "global-normalization-reference",
                "failed-update-rollback",
                "mid-epoch-continuation",
            ),
        ),
    )


DISTILLATION_CAPABILITY = "atomistic.distillation"


def _distillation_profiles() -> tuple[CapabilityProfile, ...]:
    labels = (
        ("energy-force-stress", "first-order"),
        ("atom-energy", "decomposition-bound"),
        ("embedding", "equivariant-multiplicity-projector"),
        ("coordinate-hvp", "mixed-order-three"),
        ("boltzmann-kl", "self-normalized-measure-admitted"),
    )
    return (
        _profile(
            DISTILLATION_CAPABILITY,
            "native-teacher",
            (
                {
                    "mode": mode,
                    "label": label,
                    "student-derivative": derivative,
                    "teacher": "frozen-native-parameters-geometry-differentiable",
                    "label-precision": "float64",
                    "replay": "bounded-fifo-idempotent",
                    "geometry-admission": "inherits-model-profile",
                }
                for mode, (label, derivative) in product(("offline", "on-policy"), labels)
            ),
            (
                *_SCIENTIFIC_GATES,
                "phase-interruption-continuation",
                "teacher-immutability",
            ),
        ),
    )


# P9: enhanced sampling.

SAMPLING_CAPABILITY = "atomistic.enhanced-sampling"
_CV_KINDS = (
    "aligned-msd,aligned-rmsd,angle,cell-volume,center-of-mass-distance,configuration,"
    "contact-similarity,coordination,density,distance,group-coordination,path-distance,"
    "path-progress,radius-of-gyration,torsion"
)


def _sampling_profiles() -> tuple[CapabilityProfile, ...]:
    geometries = ("finite", *_BULK)
    common: dict[str, SupportValue] = {
        "collective-variables": _CV_KINDS,
        "image-branches": "anchored,minimum-image,topology,unwrapped",
        "precision": "float64",
    }
    gates = (*_SCIENTIFIC_GATES, "ensemble-measure-reference")
    bias = _profile(
        SAMPLING_CAPABILITY,
        "conservative-bias",
        (
            {
                **common,
                "bias": bias,
                "geometry": geometry,
                "hamiltonian": "single-scalar-program-term",
                "derivative": _stress(geometry),
                "history": history,
            }
            for (bias, history), geometry in product(
                (
                    ("harmonic", "none"),
                    ("wall", "none"),
                    ("flat-bottom", "none"),
                    ("learned-free-energy", "none"),
                    ("integrated-profile", "frozen-profile"),
                    ("metadynamics", "bounded-hill-history"),
                    ("well-tempered-metadynamics", "bounded-hill-history"),
                ),
                geometries,
            )
        ),
        (*gates, "history-cadence-ownership", "restart-bitwise-continuation"),
    )
    abf = _profile(
        SAMPLING_CAPABILITY,
        "adaptive-biasing-force",
        (
            {
                **common,
                "estimator": estimator,
                "base-measure": measure,
                "geometry": geometry,
                "energy": False,
                "derivative": "none-force-only",
                "maximum-closure-atoms": 64,
            }
            for (estimator, measure), geometry in product(
                (
                    ("pair-distance", "configurational-lebesgue"),
                    ("generalized-explicit-index", "configurational-lebesgue"),
                    ("generalized-explicit-index", "rattle-constrained-fixman"),
                ),
                geometries,
            )
        ),
        (*gates, "abf-sign-measure-reference"),
    )
    exchange = _profile(
        SAMPLING_CAPABILITY,
        "multistate-exchange",
        (
            {
                **common,
                "exchange": exchange,
                "schedule": "synchronous-adjacent-odd-even",
                "cross-evaluation": "complete-bias-control-owner-reduced-potential",
                "geometry": geometry,
                "distribution": "batched-replicated-history",
            }
            for exchange, geometry in product(
                ("temperature", "equal-temperature-windows"), geometries
            )
        ),
        (*gates, "exchange-detailed-balance", "restart-bitwise-continuation"),
    )
    return bias, abf, exchange


# P10: generation streams and the crystal domain.

GENERATION_CAPABILITY = "atomistic.generation"


def _generation_profiles() -> tuple[CapabilityProfile, ...]:
    rows = (
        *(
            ("custom-transition", cell, chemistry)
            for cell in ("nonperiodic", "fixed-cell", "variable-cell")
            for chemistry in ("fixed", "variable-composition")
        ),
        *(
            (sampler, cell, chemistry)
            for sampler in ("periodic-reverse-sde", "periodic-probability-flow")
            for cell in ("fixed-cell", "variable-cell")
            for chemistry in ("fixed", "variable-composition")
        ),
    )
    stream = tuple(
        {
            "sampler": sampler,
            "cell": cell,
            "chemistry": chemistry,
            "representation": "cartesian" if cell == "nonperiodic" else "fractional",
            "precision": "float64",
            "stream": "bounded-cursor-semantic-ordinals",
            "filters": "pure-bounded",
            "likelihood": False,
            "lifecycle": "root-key-cursor-sampler-state-pending-bank",
        }
        for sampler, cell, chemistry in rows
    )
    gates = (*_COMMON_GATES, "sample-measure-reference", "stream-restart-exactly-once")
    return (
        _profile(GENERATION_CAPABILITY, "stream", stream, gates),
        _profile(
            GENERATION_CAPABILITY,
            "distributed-quota",
            (
                {
                    **attributes,
                    "placement": "execution-group-participants",
                    "ordinal-window": "one-global-pending-window",
                    "quota": "one-global-accepted-target-and-attempt-budget",
                    "publication": "participant-rank-zero-ordinal-order-receipt-agreement",
                    "transport": "fixed-shape-participant-all-gather-per-transaction",
                    "byte-bounds": "round-and-replicated",
                    "persistence": "distributed-commit-group-rank-writers",
                    "restore": "same-profile-and-placement-bitwise",
                    "changed-placement": "refused",
                }
                for attributes in stream
            ),
            (*gates, "actual-multi-process-execution", "crash-nonpublication"),
        ),
    )


CRYSTAL_CAPABILITY = "atomistic.crystal-structure-prediction"


def _crystal_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "geometry": "three-periodic-crystal",
        "precision": "float64",
        "arithmetic": "exact-rational-affine",
    }
    gates = (
        *_COMMON_GATES,
        "exact-group-closure-reference",
        "space-group-table-rights-admission",
    )
    settings = _profile(
        CRYSTAL_CAPABILITY,
        "settings-and-asymmetric-unit",
        (
            {
                **common,
                "stage": "space-group-setting",
                "hall-entries": 530,
                "standard-settings": 230,
                "metric": "operation-derived-constraints",
            },
            *(
                {
                    **common,
                    "stage": "asymmetric-unit",
                    "site-enumeration": "bounded-exact-stabilizer-orbits",
                    "wyckoff-letters": False,
                    "chirality": chirality,
                }
                for chirality in ("enantiopure", "racemic")
            ),
            {
                **common,
                "stage": "special-site-expansion",
                "site-symmetry": "special-position",
                "site-action": "admitted-molecular-automorphism-permutations",
                "expansion": "stabilizer-left-coset-representatives",
                "z-prime": "rational-per-component",
                "continuation": "symmetry-preserving-rigid-retraction",
            },
        ),
        (*gates, "symmetry-preservation"),
    )
    packing = _profile(
        CRYSTAL_CAPABILITY,
        "molecular-packing",
        (
            {
                **common,
                "stage": "packing",
                "conformer-source": source,
                "setting-policy": policy,
                "z-prime": "positive-integer",
                "site-symmetry": "general-position",
                "components": "multiple",
                "contact-rules": "donor-acceptor-reduced,radius-sum",
                "contact-relief": "canonical-half-image-rigid-body-bounded-cell",
                "volume-policy": "explicit-or-caller-admitted-atomic-data",
                "derivative": "none",
            }
            for source, policy in product(
                ("input", "native-distance-geometry"), ("fixed", "weighted")
            )
        ),
        (*gates, "contact-image-coverage", "space-group-prior-rights-admission"),
    )
    relaxation = _profile(
        CRYSTAL_CAPABILITY,
        "symmetry-relaxation",
        (
            {
                **common,
                "stage": "physical-relaxation",
                "tangent": "sparse-site-routes-and-strain-blocks",
                "rigid-chart": "quaternion-retraction",
                "residuals": "projected-and-full",
                "optimizers": _OPTIMIZERS,
                "derivative": "energy-force-stress",
            },
        ),
        (*gates, "derivative-reference", "symmetry-preservation"),
    )
    comparison = _profile(
        CRYSTAL_CAPABILITY,
        "comparison-and-ranking",
        (
            *(
                {
                    **common,
                    "stage": "equivalence-confirmation",
                    "supercell": supercell,
                    "enantiomorph": enantiomorph,
                    "undetermined-on-budget": True,
                }
                for supercell, enantiomorph in product(
                    ("identical-lattice", "primitive-reduction"),
                    ("proper-only", "allow-improper"),
                )
            ),
            *(
                {
                    **common,
                    "stage": "radial-screening",
                    "typing": typing,
                    "threshold": "inclusive",
                    "greedy-nontransitive": True,
                }
                for typing in ("untyped", "center-typed", "neighbor-typed")
            ),
            *(
                {**common, "stage": "certified-deduplication", "grouping": grouping}
                for grouping in ("greedy", "components")
            ),
            *(
                {**common, "stage": "ranking", "objective": objective}
                for objective in ("energy-per-formula-unit", "grand-energy-per-atom")
            ),
        ),
        (*gates, "bounded-equivalence-adequacy", "scientific-validation"),
    )
    recipe = _profile(
        CRYSTAL_CAPABILITY,
        "recipe-persistence",
        (
            {
                **common,
                "stage": "recipe-persistence",
                "archive": "bounded-pickle-free-array-archive",
                "content-identity": "canonical-recipe-and-array-digest",
                "verification": "constructor-replay-identity-reencode",
                "restore": "fresh-process-same-numerical-profile",
                "linkage": "plan-lanes-unfiltered-stream-identity",
                "proposal-filters": "caller-supplied-not-persisted",
            },
        ),
        (*gates, "fresh-process-continuation"),
    )
    return settings, packing, relaxation, comparison, recipe


# P11: domain execution, reciprocal distribution and distributed training.

DOMAIN_CAPABILITY = "atomistic.domain-execution"
_DOMAIN_MODELS = (
    ("mace", "none", _PERIODIC),
    ("aimnet2", "charge-equilibration", _PERIODIC),
    ("uma", "charge-spin-channel-balance,composition-routing", _PERIODIC),
    ("maceles", "latent-ewald-epilogue", _BULK),
)


def _domain_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "owner-count": 2,
        "collective": "jax-native-collectives",
        "capacity-failure": "refuse-without-truncation",
    }
    spatial = _profile(
        DOMAIN_CAPABILITY,
        "spatial-models",
        (
            {
                **common,
                "strategy": strategy,
                "search": "cell-list,cluster-tile"
                if strategy == "graph-partition"
                else "owner-halo",
                "model": model,
                "system-stages": stages,
                "geometry": geometry,
                "precision": precision,
                "derivative": (
                    "energy-force-parameter"
                    if geometry == "partial-periodic"
                    else "energy-force-stress-parameter"
                ),
            }
            for strategy in ("halo", "graph-partition")
            for model, stages, geometries in _DOMAIN_MODELS
            for geometry, precision in product(geometries, _PRECISIONS)
        ),
        (*_DISTRIBUTED_GATES, "migration-restart", "shared-cell-cotangent"),
    )
    dynamics = _profile(
        DOMAIN_CAPABILITY,
        "domain-dynamics",
        (
            {
                **common,
                "integrator": method,
                "ensemble": ensemble,
                "geometry": geometry,
                "precision": "float64",
                "constraints": "shake-rattle-independent-support-capacity",
                "migration": "complete-continuation-transaction",
            }
            for method, ensemble, geometries in (
                *(
                    (method, ensemble, _PERIODIC)
                    for method, ensemble in _FIXED_CELL_METHODS
                ),
                ("isotropic-monte-carlo-barostat", "npt", _BULK),
                ("mtk-npt", "npt", _BULK),
                ("mtk-nph", "nph", _BULK),
            )
            for geometry in geometries
        ),
        (*_DISTRIBUTED_GATES, "ensemble-measure-reference", "migration-restart"),
    )
    lifecycle = _profile(
        DOMAIN_CAPABILITY,
        "checkpoint-publication",
        (
            {
                **common,
                "publication": "authenticated-process-regions-global-commit",
                "restore": restore,
            }
            for restore in ("same-topology-bitwise", "changed-topology-migration")
        ),
        (*_DISTRIBUTED_GATES, "crash-nonpublication", "migration-restart"),
    )
    dispersion = _profile(
        DOMAIN_CAPABILITY,
        "dispersion-terms",
        (
            {
                **common,
                "owner-count": owners,
                "term": term,
                "real-space": real_space,
                "strategy": strategy,
                "geometry": geometry,
                "cell": "static-or-dynamic",
                "fft-layout": layout,
                "precision": "float64",
                "derivative": "energy-force-stress",
                "cotangents": "coordination-and-coefficient-exactly-once",
            }
            for (term, real_space), strategy, geometry, (layout, owners) in product(
                (
                    ("lj-pme", "identity-site-map-topology-exception-scales"),
                    ("fourier-d3", "modified-coordination-own-role"),
                ),
                ("halo", "graph-partition"),
                _BULK,
                (("x-slab", 2), ("xy-pencil-physical-z-slab-modal", 4)),
            )
        ),
        (*_DISTRIBUTED_GATES, "single-device-term-parity"),
    )
    return spatial, dynamics, lifecycle, dispersion


RECIPROCAL_CAPABILITY = "atomistic.reciprocal-distribution"


def _reciprocal_profiles() -> tuple[CapabilityProfile, ...]:
    common: dict[str, SupportValue] = {
        "owner-count": 2,
        "fft-layout": "x-slab",
        "precision": "float64",
        "derivative": "energy-force-stress",
    }
    meshes = tuple(
        {
            **common,
            "channel": channel,
            "strategy": strategy,
            "geometry": geometry,
            "periodic-axes": 3,
        }
        for channel, strategy, geometry in product(
            ("pme", "multipole-particle-mesh", "lj-pme", "fourier-d3"),
            ("distributed-fft", "replicated-mesh"),
            _BULK,
        )
    )
    pencil: dict[str, SupportValue] = {
        "owner-count": 4,
        "fft-layout": "xy-pencil-physical-z-slab-modal",
        "pencil-grid": "2x2-on-one-ownership-axis",
    }
    pencils = tuple(
        {
            **common,
            **pencil,
            "channel": channel,
            "strategy": "distributed-fft",
            "geometry": geometry,
            "periodic-axes": 3,
        }
        for channel, geometry in product(
            ("pme", "multipole-particle-mesh", "lj-pme", "fourier-d3"), _BULK
        )
    )
    modes = (
        *(
            {
                **common,
                "channel": channel,
                "strategy": "distributed-mode-chunks",
                "geometry": geometry,
                "periodic-axes": 3,
            }
            for channel, geometry in product(
                ("ewald", "multipole-ewald", "latent-ewald"), _BULK
            )
        ),
        {
            **common,
            "channel": "ewald-slab-yeh-berkowitz",
            "strategy": "distributed-mode-chunks",
            "geometry": "partial-periodic",
            "periodic-axes": 2,
        },
    )
    reciprocal = _profile(
        RECIPROCAL_CAPABILITY,
        "reciprocal",
        (*meshes, *pencils, *modes),
        (*_DISTRIBUTED_GATES, "patch-coverage-certificate", "single-device-term-parity"),
    )
    solvers = (("pcg", "implicit-stationary"), ("tcg", "algorithmic-fixed-order"))
    polarization = _profile(
        RECIPROCAL_CAPABILITY,
        "owner-polarization",
        (
            *(
                {
                    "owner-count": 2,
                    "precision": "float64",
                    "solver": solver,
                    "sensitivity": sensitivity,
                    "route": route,
                    "geometry": geometry,
                }
                for (solver, sensitivity), (route, geometry) in product(
                    solvers,
                    (
                        ("standalone-finite", "finite"),
                        *(
                            (f"halo-composed-{periodic}", geometry)
                            for periodic in ("ewald", "particle-mesh")
                            for geometry in _BULK
                        ),
                    ),
                )
            ),
            *(
                {
                    **pencil,
                    "precision": "float64",
                    "solver": solver,
                    "sensitivity": sensitivity,
                    "route": "halo-composed-particle-mesh",
                    "geometry": geometry,
                }
                for (solver, sensitivity), geometry in product(solvers, _BULK)
            ),
        ),
        (*_DISTRIBUTED_GATES, "global-convergence-residual"),
    )
    return reciprocal, polarization


TRAINING_DISTRIBUTION_CAPABILITY = "atomistic.training-distribution"


def _training_distribution_profiles() -> tuple[CapabilityProfile, ...]:
    return (
        _profile(
            TRAINING_DISTRIBUTION_CAPABILITY,
            "data-parallel",
            (
                {
                    "mesh": mesh,
                    "reduction": reduction,
                    "normalization": "global-numerator-support",
                    "gradient-reduction": "exactly-once",
                    "partition-bitwise-invariance": False,
                }
                for mesh, reduction in product(
                    ("data-parallel", "data-by-domain"),
                    ("fast", "deterministic", "compensated"),
                )
            ),
            (*_DISTRIBUTED_GATES, "global-normalization-reference"),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ClosureException:
    capability: str
    owner: str
    profiles: Callable[[], tuple[CapabilityProfile, ...]]
    public_symbols: tuple[str, ...]
    dependencies: tuple[str, ...]
    documentation: tuple[str, ...]
    examples: tuple[str, ...]
    intended_use: str
    nonclaims: tuple[str, ...]


def _symbols(module: str, *names: str) -> tuple[str, ...]:
    return tuple(f"{module}.{name}" for name in names)


_A = "phydrax.atomistic"
_NN = "phydrax.nn.atomistic"
_MODEL_NONCLAIMS = (
    "architecture-support-is-not-named-checkpoint-fidelity",
    "no-accelerated-route-profile",
    "no-pretrained-chemical-accuracy",
    "source-code-license-does-not-license-weights",
)

_EXCEPTIONS = (
    _ClosureException(
        NEIGHBORHOOD_CAPABILITY,
        _A,
        _neighborhood_profiles,
        _symbols(
            _A,
            "AtomisticCellListImageSearch",
            "AtomisticClusterTileImageSearch",
            "AtomisticDenseImageSearch",
            "AtomisticNeighborhoodBundlePlan",
            "AtomisticNeighborhoodBundleState",
            "AtomisticNeighborhoodRequirement",
            "AtomisticNeighborhoodRole",
            "AtomisticNeighborhoodSource",
            "AtomisticNeighborhoodView",
            "AtomisticRoleCapacityLadder",
            "AtomisticTiledImageSearch",
            "PreparedAtomisticNeighborhoodBundle",
        ),
        (),
        ("docs/api/atomistic.md", "docs/guides_atomistic.md"),
        (),
        "bounded-atomistic-resource-preparation",
        (
            "neighbor-discovery-is-not-differentiated",
            "no-accelerator-search-route",
            "selective-rebuild-executes-batched-search",
            "tiled-reference-is-all-pairs-work-not-a-scalable-route",
        ),
    ),
    _ClosureException(
        ROUTE_REDUCTION_CAPABILITY,
        "phydrax.sparse",
        _route_reduction_profiles,
        _symbols(
            "phydrax.sparse",
            "route_dot",
            "route_matvec",
            "route_multiply",
            "route_reduce",
            "route_rms",
            "route_support",
        ),
        (),
        ("docs/guides_sparse_spatial_hierarchies.md",),
        (),
        "bounded-segmented-reduction",
        ("fast-reductions-are-not-bitwise-deterministic",),
    ),
    _ClosureException(
        SPLINE_CAPABILITY,
        "phydrax.discretization.splatting",
        _spline_profiles,
        _symbols(
            "phydrax.discretization.splatting",
            "ParticleGridSplatPlan",
            "PreparedParticleGridSplat",
            "TensorBSplineSplatAssignment",
        ),
        (),
        ("docs/guides_particle_splatting.md",),
        ("examples/particle_grid_splat.py",),
        "bounded-particle-grid-transfer",
        ("no-materialized-derivative-beyond-first-order",),
    ),
    _ClosureException(
        ELECTROSTATICS_CAPABILITY,
        _A,
        _electrostatics_profiles,
        _symbols(
            _A,
            "CoulombSplittingEstimate",
            "DampedShiftedForceCoulombPotential",
            "DirectCoulombPotential",
            "EwaldReferencePotential",
            "ParticleMeshEwaldPotential",
            "ScreenedCoulombPotential",
            "estimate_ewald_splitting",
            "estimate_pme_splitting",
        ),
        (NEIGHBORHOOD_CAPABILITY, SPLINE_CAPABILITY),
        ("docs/api/atomistic_force_fields.md", "docs/guides_atomistic_force_fields.md"),
        (),
        "bounded-electrostatics-evaluation",
        (
            "splitting-estimates-are-not-force-error-certificates",
            "no-hessian-operator-admission-for-program-terms",
        ),
    ),
    _ClosureException(
        MULTIPOLE_CAPABILITY,
        _A,
        _multipole_profiles,
        _symbols(
            _A,
            "EwaldMultipoleRoute",
            "FiniteMultipoleRoute",
            "MultipoleElectrostaticPotential",
            "MultipoleElectrostaticsDefinition",
            "MultipoleElectrostaticsPlan",
            "ParticleMeshMultipoleRoute",
        ),
        (NEIGHBORHOOD_CAPABILITY, SPLINE_CAPABILITY),
        ("docs/guides_atomistic_polarization.md",),
        (),
        "bounded-multipole-electrostatics",
        ("no-multipole-beyond-quadrupole", "no-periodic-multipole-slab-correction"),
    ),
    _ClosureException(
        DISPERSION_CAPABILITY,
        _A,
        _dispersion_profiles,
        _symbols(
            _A,
            "D3BJDamping",
            "D3BJPotential",
            "D3CoordinationPolicy",
            "D3ReferenceTables",
            "D3TableProvenance",
            "FourierC6Decomposition",
            "FourierD3Potential",
        ),
        (NEIGHBORHOOD_CAPABILITY, SPLINE_CAPABILITY),
        ("docs/guides_atomistic_force_fields.md",),
        (),
        "bounded-dispersion-correction",
        (
            "no-atm-three-body-term",
            "no-bundled-d3-reference-tables",
            "synthetic-tables-do-not-qualify-element-fidelity",
        ),
    ),
    _ClosureException(
        LATENT_CAPABILITY,
        _A,
        _latent_profiles,
        _symbols(_A, "LatentEwaldDefinition", "LatentEwaldPotential"),
        (NEIGHBORHOOD_CAPABILITY,),
        ("docs/guides_atomistic_models.md",),
        (),
        "bounded-latent-electrostatics",
        (
            "latent-charges-are-not-physical-partial-charges",
            "no-slab-latent-electrostatics",
        ),
    ),
    _ClosureException(
        AIMNET2_CAPABILITY,
        _NN,
        _aimnet2_profiles,
        (
            *_symbols(
                _NN, "AIMNet2Architecture", "AIMNet2OutputKind", "AIMNet2Potential"
            ),
            f"{_A}.TSDispersionPotential",
        ),
        (
            DISPERSION_CAPABILITY,
            ELECTROSTATICS_CAPABILITY,
            NEIGHBORHOOD_CAPABILITY,
            ROUTE_REDUCTION_CAPABILITY,
        ),
        ("docs/guides_atomistic_models.md",),
        (),
        "bounded-learned-potential-evaluation",
        (*_MODEL_NONCLAIMS, "nse-charge-conservation-is-reported-not-asserted"),
    ),
    _ClosureException(
        UMA_CAPABILITY,
        _NN,
        _uma_profiles,
        _symbols(
            _NN,
            "PreparedUMAMergedModel",
            "UMAArchitecture",
            "UMAPotential",
            "UMAPropertyModel",
        ),
        (NEIGHBORHOOD_CAPABILITY,),
        ("docs/guides_atomistic_models.md",),
        (),
        "bounded-learned-potential-evaluation",
        (*_MODEL_NONCLAIMS, "direct-heads-are-not-conservative-responses"),
    ),
    _ClosureException(
        MACE_EXTENDED_CAPABILITY,
        _NN,
        _mace_extended_profiles,
        _symbols(_NN, "MACEJointEmbedding"),
        ("atomistic.mace-inference", NEIGHBORHOOD_CAPABILITY),
        ("docs/guides_atomistic_models.md", "docs/guides_mace_execution.md"),
        ("examples/mace_checkpoint_conversion.py",),
        "bounded-learned-potential-evaluation",
        (*_MODEL_NONCLAIMS, "no-named-mh-1-checkpoint-qualification"),
    ),
    _ClosureException(
        MACELES_CAPABILITY,
        _NN,
        _maceles_profiles,
        _symbols(_NN, "MACELESPotential"),
        ("atomistic.mace-inference", LATENT_CAPABILITY),
        ("docs/guides_atomistic_models.md",),
        (),
        "bounded-learned-potential-evaluation",
        (*_MODEL_NONCLAIMS, "les-noncommercial-source-rights-not-admitted"),
    ),
    _ClosureException(
        INTERCHANGE_CAPABILITY,
        "phydrax.atomistic.interchange",
        _interchange_profiles,
        _symbols(
            "phydrax.atomistic.interchange",
            "AIMNet2Source",
            "MACESource",
            "TrustedTorchPickleSource",
            "UMASource",
            "convert_aimnet2_checkpoint",
            "convert_mace_checkpoint",
            "convert_uma_checkpoint",
        ),
        (
            "atomistic.mace-inference",
            AIMNET2_CAPABILITY,
            MACE_EXTENDED_CAPABILITY,
            MACELES_CAPABILITY,
            UMA_CAPABILITY,
        ),
        ("docs/guides_atomistic_interop.md", "docs/guides_atomistic_models.md"),
        ("examples/mace_checkpoint_conversion.py",),
        "admitted-source-artifact-conversion",
        (
            "artifact-digest-does-not-grant-rights",
            "no-opaque-compiled-model-wrapping",
            "provider-subprocess-is-not-a-memory-sandbox",
        ),
    ),
    _ClosureException(
        DYNAMICS_CAPABILITY,
        _A,
        _dynamics_profiles,
        _symbols(
            _A,
            "AtomisticCellChart",
            "AtomisticDynamicsPlan",
            "BAOABLangevinPlan",
            "GeneralizedLangevinPlan",
            "IsotropicMonteCarloBarostatPlan",
            "MTKBarostatPlan",
            "MTKNPHPlan",
            "MTKNPTPlan",
            "NoseHooverChainParameters",
            "NoseHooverChainPlan",
            "StochasticVelocityRescalingPlan",
            "VelocityRescalingPlan",
            "VelocityVerletPlan",
        ),
        (NEIGHBORHOOD_CAPABILITY,),
        ("docs/cookbook/atomistic_dynamics.md", "docs/guides_atomistic_dynamics.md"),
        ("examples/atomistic_dynamics.py", "examples/periodic_mace_dynamics.py"),
        "bounded-molecular-dynamics",
        (
            "velocity-rescaling-is-not-canonical",
            "no-berendsen-or-c-rescale-barostat",
        ),
    ),
    _ClosureException(
        RELAXATION_CAPABILITY,
        _A,
        _relaxation_profiles,
        (
            *_symbols(
                _A,
                "AtomisticCartesianChart",
                "AtomisticDeformationChart",
                "AtomisticRelaxationPlan",
                "AtomisticSymmetryChart",
            ),
            *_symbols("phydrax.optim", "FIRE", "FIRE2", "TrustScaledLBFGS"),
        ),
        (NEIGHBORHOOD_CAPABILITY,),
        ("docs/api/optim.md", "docs/guides_atomistic_dynamics.md"),
        ("examples/atomistic_relaxation_campaign.py",),
        "bounded-structure-relaxation",
        ("trust-scaled-lbfgs-has-no-line-search-monotonicity",),
    ),
    _ClosureException(
        REACTION_PATH_CAPABILITY,
        "phydrax.chemistry.reaction",
        _reaction_path_profiles,
        _symbols(
            "phydrax.chemistry",
            "AtomisticNudgedElasticBandPlan",
            "PreparedAtomisticNudgedElasticBand",
            "ReactionPathBranch",
            "ReactionPathBranchCertificate",
        ),
        (NEIGHBORHOOD_CAPABILITY, RELAXATION_CAPABILITY),
        ("docs/guides_atomistic_dynamics.md",),
        (),
        "bounded-reaction-path-search",
        ("no-variable-cell-neb", "no-holonomic-native-neb"),
    ),
    _ClosureException(
        CAMPAIGN_CAPABILITY,
        _A,
        _campaign_profiles,
        _symbols(
            _A,
            "AtomisticCampaignPlan",
            "AtomisticDomainCampaignCheckpointPolicy",
            "AtomisticDomainCampaignPlan",
            "AtomisticDomainStage",
            "AtomisticDomainStageProgram",
            "AtomisticStagePlan",
            "AtomisticStageTransportPlan",
            "AtomisticTaskPacket",
            "PreparedAtomisticCampaign",
            "PreparedAtomisticDomainCampaign",
        ),
        (
            DOMAIN_CAPABILITY,
            DYNAMICS_CAPABILITY,
            GENERATION_CAPABILITY,
            REACTION_PATH_CAPABILITY,
            RELAXATION_CAPABILITY,
        ),
        (
            "docs/guides_atomistic_distributed_execution.md",
            "docs/guides_atomistic_dynamics.md",
        ),
        ("examples/atomistic_relaxation_campaign.py",),
        "bounded-staged-campaigns",
        (
            "lane-reference-is-not-distributed-hardware-evidence",
            "no-multi-host-evidence",
            "rank-loss-is-restart-not-in-job-rollback",
        ),
    ),
    _ClosureException(
        TRAINING_CAPABILITY,
        _A,
        _training_profiles,
        _symbols(
            _A,
            "AtomisticEnergyObjective",
            "AtomisticForceObjective",
            "AtomisticIrrepLowRankAdaptation",
            "AtomisticLoaderCursor",
            "AtomisticObjectiveSchedule",
            "AtomisticParameterSelection",
            "AtomisticPropertyObjective",
            "AtomisticResidualLoss",
            "AtomisticStressObjective",
            "AtomisticStructureStoreSchema",
            "AtomisticStructureStoreWriter",
            "AtomisticTrainingPolicy",
            "AtomisticTrainingPrecision",
            "AtomisticTrainingProblem",
            "AtomisticTrainingResult",
            "fit_atomistic_potential",
        ),
        (NEIGHBORHOOD_CAPABILITY,),
        ("docs/api/atomistic.md", "docs/guides_atomistic.md"),
        ("examples/atomistic_streaming_training.py",),
        "bounded-potential-training",
        ("no-mixed-precision-loss-scaling", "no-staging-overlap-claim"),
    ),
    _ClosureException(
        DISTILLATION_CAPABILITY,
        _A,
        _distillation_profiles,
        _symbols(
            _A,
            "AtomisticEmbeddingObjective",
            "AtomisticExperienceBuffer",
            "AtomisticHessianVectorObjective",
            "AtomisticLabelRequest",
            "AtomisticOnPolicyDistillationPlan",
            "AtomisticTeacher",
            "label_atomistic_structures",
            "run_atomistic_distillation_segment",
        ),
        (CAMPAIGN_CAPABILITY, TRAINING_CAPABILITY),
        ("docs/guides_atomistic_active_learning.md",),
        ("examples/atomistic_distillation.py",),
        "bounded-native-distillation",
        (
            "atom-energy-matching-is-not-physical-supervision",
            "no-strain-hvp-matching",
            "random-teacher-demonstrates-mechanics-only",
        ),
    ),
    _ClosureException(
        SAMPLING_CAPABILITY,
        "phydrax.atomistic.sampling",
        _sampling_profiles,
        _symbols(
            "phydrax.atomistic.sampling",
            "AdaptiveProfileResult",
            "AtomisticABFForcePort",
            "AtomisticABFPlan",
            "AtomisticBiasHistoryPlan",
            "AtomisticBiasTerm",
            "AtomisticMultistateCheckpointPlan",
            "AtomisticReplicaExchangePlan",
            "BiasHistoryOwnership",
            "BiasHistoryStorage",
            "CollectiveVariableGeometry",
            "CollectiveVariableKind",
            "CollectiveVariableMetric",
            "CollectiveVariablePlan",
            "CollectiveVariableProgram",
            "FlatBottomBiasForm",
            "FrozenBiasHistory",
            "HarmonicBiasForm",
            "HillHistoryBiasForm",
            "HillKernel",
            "IntegratedProfileBiasForm",
            "PreparedAtomisticMultistate",
            "PreparedBiasedDynamics",
            "WallBiasForm",
            "aligned_deviation",
        ),
        (DYNAMICS_CAPABILITY,),
        ("docs/api/atomistic_sampling.md", "docs/guides_atomistic_sampling.md"),
        ("examples/atomistic_sampling.py",),
        "bounded-enhanced-sampling",
        (
            "adaptive-profiles-are-diagnostics-not-equilibrium-free-energies",
            "no-asynchronous-or-walker-dependent-hamiltonian-exchange",
            "no-float32-sampling-envelope",
            "wrapped-hill-kernel-is-a-finite-image-sum",
        ),
    ),
    _ClosureException(
        GENERATION_CAPABILITY,
        _A,
        _generation_profiles,
        _symbols(
            _A,
            "AtomisticGenerationCapacityError",
            "AtomisticGenerationCheckpointError",
            "AtomisticGenerationDivergenceError",
            "AtomisticGenerationPlacement",
            "AtomisticGenerationStreamPlan",
            "AtomisticGenerationSupport",
            "AtomisticPeriodicCoordinateDiffusion",
            "PreparedAtomisticGenerationStream",
            "PreparedPeriodicDiffusionSampler",
            "PreparedPeriodicFlowSampler",
        ),
        (),
        ("docs/guides_atomistic_generation.md",),
        ("examples/atomistic_generation_stream.py",),
        "bounded-structure-generation",
        (
            "no-accelerator-or-multi-host-generation-evidence",
            "no-changed-placement-generation-migration",
            "no-generative-likelihood",
            "no-partial-periodic-generation",
        ),
    ),
    _ClosureException(
        CRYSTAL_CAPABILITY,
        "phydrax.atomistic.crystal",
        _crystal_profiles,
        _symbols(
            "phydrax.atomistic.crystal",
            "AsymmetricUnitPlan",
            "AsymmetricUnitSite",
            "ConformerGenerationPlan",
            "CrystalPackingPlan",
            "CrystalPackingRecipe",
            "CrystalPackingRecipeReceipt",
            "DedupBank",
            "MolecularComponent",
            "PreparedCrystalPacker",
            "RadialScreenPolicy",
            "RigidSymmetryLayout",
            "SettingMetricChart",
            "SpaceGroupSetting",
            "SymmetryTangentOperator",
            "certify_equivalence",
            "enumerate_site_orbits",
            "expand_asymmetric_unit",
            "rank_crystal_candidates",
            "read_crystal_packing_recipe",
            "write_crystal_packing_recipe",
        ),
        (GENERATION_CAPABILITY, RELAXATION_CAPABILITY),
        ("docs/guides_atomistic_generation.md",),
        ("examples/atomistic_crystal_prediction.py",),
        "bounded-molecular-crystal-search",
        (
            "no-authoritative-wyckoff-letters",
            "no-cross-backend-recipe-replay",
            "no-finite-temperature-crystal-stability",
            "no-randomized-special-position-packing",
            "packing-overlap-tensor-is-not-physical-stress",
            "radial-screening-is-not-equivalence-proof",
        ),
    ),
    _ClosureException(
        DOMAIN_CAPABILITY,
        _A,
        _domain_profiles,
        _symbols(
            _A,
            "AtomisticDomainConstraintPlan",
            "AtomisticDomainDynamicsPlan",
            "AtomisticDomainEwaldTerm",
            "AtomisticDomainFourierD3Term",
            "AtomisticDomainLennardJonesPMETerm",
            "AtomisticDomainPlan",
            "AtomisticDomainPolarizationTerm",
            "AtomisticDomainRolePlan",
            "AtomisticDomainState",
            "AtomisticDomainStrategy",
            "AtomisticGraphPartitionPlan",
            "PreparedAtomisticDomainRuntime",
            "migrate_domain_checkpoint",
            "prepare_atomistic_domain",
            "publish_domain_checkpoint",
            "restore_domain_checkpoint",
        ),
        (
            AIMNET2_CAPABILITY,
            DISPERSION_CAPABILITY,
            DYNAMICS_CAPABILITY,
            "atomistic.mace-inference",
            MACELES_CAPABILITY,
            NEIGHBORHOOD_CAPABILITY,
            UMA_CAPABILITY,
        ),
        ("docs/guides_atomistic_distributed_execution.md",),
        (),
        "bounded-distributed-atomistic-execution",
        (
            "lane-reference-is-not-distributed-hardware-evidence",
            "no-multi-host-evidence",
            "no-predicted-moment-or-polarizability-domain-coupling",
            "rank-loss-is-restart-not-in-job-rollback",
        ),
    ),
    _ClosureException(
        RECIPROCAL_CAPABILITY,
        _A,
        _reciprocal_profiles,
        _symbols(
            _A,
            "AtomisticReciprocalDistributionPlan",
            "AtomisticReciprocalSchedule",
            "OwnerPolarizationPlan",
            "solve_owner_polarization",
        ),
        (
            DISPERSION_CAPABILITY,
            DOMAIN_CAPABILITY,
            ELECTROSTATICS_CAPABILITY,
            LATENT_CAPABILITY,
            MULTIPOLE_CAPABILITY,
            "distributed-spectral-execution",
        ),
        ("docs/guides_atomistic_distributed_execution.md",),
        (),
        "bounded-distributed-reciprocal-work",
        (
            "lane-reference-is-not-distributed-hardware-evidence",
            "no-observed-multi-device-pencil-execution",
            "replicated-mesh-is-not-a-distributed-fft",
            "tcg-never-reports-convergence",
        ),
    ),
    _ClosureException(
        TRAINING_DISTRIBUTION_CAPABILITY,
        _A,
        _training_distribution_profiles,
        _symbols(_A, "AtomisticTrainingDistributionPlan"),
        (DOMAIN_CAPABILITY, TRAINING_CAPABILITY),
        ("docs/guides_atomistic_distributed_execution.md",),
        (),
        "bounded-distributed-training",
        ("lane-reference-is-not-distributed-hardware-evidence",),
    ),
)


def atomistic_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return every evidence-free native atomistic candidate profile."""
    return tuple(profile for value in _EXCEPTIONS for profile in value.profiles())


def atomistic_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    """Declare each closure exception at its scientific owner, never released."""
    return tuple(
        CapabilityDeclaration(
            value.capability,
            value.owner,
            CapabilityDisposition.CANDIDATE,
            domain_maturity="unreleased-candidate",
            public_symbols=value.public_symbols,
            profiles=value.profiles(),
            dependencies=value.dependencies,
            documentation=value.documentation,
            examples=value.examples,
            intended_uses=(value.intended_use,),
            nonclaims=(*_COMMON_NONCLAIMS, *value.nonclaims),
        )
        for value in _EXCEPTIONS
    )


__all__ = [
    "CLOSURE_EXCEPTION",
    "atomistic_candidate_declarations",
    "atomistic_candidate_profiles",
]

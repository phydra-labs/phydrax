#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact unreleased semiconductor capability and campaign declarations."""

from __future__ import annotations

from collections.abc import Mapping

from ...qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportDependency,
    SupportTuple,
)


_FIELD_PROFILE = "semiconductor.detector.fixed-linear-electrostatics.v1"
_RESPONSE_PROFILE = "semiconductor.detector.prescribed-shockley-ramo.v1"


_PROFILE_SPECS: dict[str, tuple[str, Mapping[str, str | int | bool], tuple[str, ...]]] = {
    "semiconductor.device.isothermal-homojunction-dd.v1": (
        "semiconductor.device.stationary",
        {
            "code_id": "phydrax.applications.semiconductor.PreparedSemiconductorDevice.solve",
            "physics": "isothermal-homojunction-poisson-scharfetter-gummel-srh",
            "statistics": "nondegenerate",
            "terminals": "ohmic-and-electrostatic-gate",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "caller-declared-nonlinear-and-mesh-bounds",
        },
        (),
    ),
    "semiconductor.device.heterointerface-dd.v1": (
        "semiconductor.device.stationary",
        {
            "code_id": "phydrax.applications.semiconductor.PreparedSemiconductorDevice.solve",
            "physics": "stationary-poisson-drift-diffusion-explicit-material-interface",
            "interface": "separate-material-traces-reciprocal-thermionic-spectrum",
            "statistics": "declared-boltzmann-or-fermi-dirac",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "caller-declared-nonlinear-and-mesh-bounds",
        },
        ("semiconductor.device.isothermal-homojunction-dd.v1",),
    ),
    "semiconductor.device.electrothermal-dd.v1": (
        "semiconductor.device.electrothermal",
        {
            "code_id": "phydrax.applications.semiconductor.PreparedSemiconductorDevice.solve",
            "physics": "stationary-electrothermal-drift-diffusion-extensive-energy",
            "thermal_boundary": "declared-thermal-ports",
            "statistics": "declared-boltzmann-or-fermi-dirac",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "caller-declared-nonlinear-and-mesh-bounds",
        },
        ("semiconductor.device.isothermal-homojunction-dd.v1",),
    ),
    "semiconductor.quantum.effective-mass-1d.v1": (
        "semiconductor.quantum.stationary",
        {
            "code_id": "phydrax.applications.semiconductor.quantum.solve_schrodinger_poisson",
            "physics": "one-dimensional-effective-mass-schrodinger-poisson",
            "basis": "finite-difference-open-boundary",
            "statistics": "declared-reservoir-occupations",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "quantum-resources-preallocation-refusal",
        },
        (),
    ),
    "semiconductor.device.stationary-quantum-classical-hybrid.v1": (
        "semiconductor.device.stationary-hybrid",
        {
            "code_id": "phydrax.applications.semiconductor.quantum.solve_quantum_classical_interface",
            "physics": "stationary-one-dimensional-quantum-classical-current-matching",
            "coupling": "explicit-interface-and-bracketed-voltage-root",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "classical-and-quantum-preallocation-refusal",
        },
        (
            "semiconductor.device.isothermal-homojunction-dd.v1",
            "semiconductor.quantum.effective-mass-1d.v1",
        ),
    ),
    _FIELD_PROFILE: (
        "semiconductor.detector.electrostatics",
        {
            "code_id": "phydrax.applications.semiconductor.DetectorWeightingFieldPlan.solve",
            "physics": "fixed-linear-dielectric-cochain-poisson",
            "outputs": "distinct-bias-and-zero-charge-one-hot-weighting",
            "electrodes": "explicit-disjoint-boundary-masks",
            "partition": "certified-only-for-complete-dirichlet-electrodes",
            "capacitance": "maxwell-matrix-with-reciprocity-and-charge-evidence",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "caller-declared-node-edge-electrode-and-iteration-bounds",
        },
        (),
    ),
    _RESPONSE_PROFILE: (
        "semiconductor.detector.response",
        {
            "code_id": "phydrax.applications.semiconductor.PrescribedShockleyRamoPlan.evaluate",
            "physics": "prescribed-point-charge-shockley-ramo",
            "trajectory": "phydrax-dynamics-trajectory-data-contiguous-routes",
            "interpolation": "bounded-multilinear-nodal-weighting-potential",
            "sign": "q-induced-equals-minus-q-phi-weighting-current-into-electrode-is-time-derivative",
            "current": "interval-secant-with-endpoint-integral-closure",
            "dynamics": "none-prescribed-trajectories-are-not-advanced",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "caller-declared-case-sample-route-bounds",
        },
        (_FIELD_PROFILE,),
    ),
    "semiconductor.quantum.chain-landauer.v1": (
        "semiconductor.quantum.chain-landauer",
        {
            "code_id": "phydrax.applications.semiconductor.quantum.integrate_coherent",
            "physics": "stationary-two-terminal-scalar-chain-landauer",
            "leads": "analytic-semi-infinite-single-orbital",
            "green_function": "selected-boundary-source-columns-no-dense-inverse",
            "bound_states": "explicit-preparation-required",
            "current": "conduction-only-positive-into-device",
            "nonclaim": "no-displacement-current-or-interacting-transport",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "quantum-resources-preallocation-refusal",
        },
        (),
    ),
    "semiconductor.quantum.chain-coherent-ac.v1": (
        "semiconductor.quantum.chain-coherent-ac",
        {
            "code_id": (
                "phydrax.applications.semiconductor.quantum."
                "finite_frequency_quantum_response"
            ),
            "physics": "connected-equilibrium-coherent-kubo-capacitive-hartree",
            "lead_model": "finite-spatial-dilation-with-lead-refinement",
            "frequency": "positive-adiabatic-switch-on-rate",
            "evidence": "gauge-kcl-ward-electrostatic-and-refinement",
            "nonclaim": "no-scba-vertex-or-ballistic-dc-from-adiabatic-rate",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "quantum-resources-dense-dilation-bound",
        },
        ("semiconductor.quantum.chain-landauer.v1",),
    ),
    "semiconductor.quantum.chain-finite-lead-transient.v1": (
        "semiconductor.quantum.chain-finite-lead-transient",
        {
            "code_id": (
                "phydrax.applications.semiconductor.quantum.solve_quantum_transient"
            ),
            "physics": "unitary-finite-spatial-lead-dilation",
            "drive": "explicit-piecewise-constant-energy-shifts",
            "state": "complete-one-body-correlation-with-all-coherences",
            "evidence": "return-window-lead-size-memory-and-unitarity-refinement",
            "nonclaim": "no-lindblad-or-general-interacting-transient-negf",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "bounded-quadratic-correlation-storage",
        },
        ("semiconductor.quantum.chain-landauer.v1",),
    ),
    "semiconductor.quantum.chain-optical-phonon-scba.v1": (
        "semiconductor.quantum.chain-optical-phonon-scba",
        {
            "code_id": (
                "phydrax.applications.semiconductor.quantum.solve_phonon_transport"
            ),
            "physics": "local-optical-phonon-fock-scba-scalar-chain",
            "energy_grid": "uniform-midpoint-integer-phonon-shift-no-wrap",
            "modes": "independent-declared-transverse-modes",
            "evidence": "causality-spectral-kms-particle-energy-and-refinement",
            "nonclaim": "no-hartree-tadpole-numerical-eta-or-vertex-corrections",
            "units": "si",
            "execution": "single-device-cpu-float64",
            "resources": "quantum-resources-energy-correlation-bound",
        },
        ("semiconductor.quantum.chain-landauer.v1",),
    ),
}


_REQUIRED_GATES = (
    "scientific-calibration",
    "locked-evaluation",
    "resource-envelope",
    "runtime-attestation",
    "lifecycle-restore",
    "documentation-and-nonclaims",
)


def _candidate(
    name: str,
    version: str,
    cache: dict[str, CapabilityProfile],
    /,
) -> CapabilityProfile:
    if name in cache:
        return cache[name]
    if name not in _PROFILE_SPECS:
        known = ", ".join(sorted(_PROFILE_SPECS))
        raise ValueError(
            f"Unknown semiconductor capability profile {name!r}; expected {known}."
        )
    capability, attributes, dependency_names = _PROFILE_SPECS[name]
    dependencies = []
    for dependency_name in dependency_names:
        profile = _candidate(dependency_name, version, cache)
        dependencies.append(
            SupportDependency(
                profile.profile_id,
                profile.support_tuples[0].support_tuple_id,
            )
        )
    profile = CapabilityProfile(
        name,
        "phydrax",
        version,
        (SupportTuple(capability, attributes),),
        dependencies=tuple(dependencies),
        required_gates=_REQUIRED_GATES,
        released=False,
    )
    cache[name] = profile
    return profile


def semiconductor_candidate_profile(
    name: str,
    /,
    *,
    version: str = "candidate",
) -> CapabilityProfile:
    """Return one evidence-free exact candidate; this never grants admission."""

    return _candidate(name, version, {})


def semiconductor_candidate_profiles(
    *,
    version: str = "candidate",
) -> tuple[CapabilityProfile, ...]:
    """Return all exact owner-local semiconductor candidates in stable order."""

    cache: dict[str, CapabilityProfile] = {}
    return tuple(_candidate(name, version, cache) for name in sorted(_PROFILE_SPECS))


def semiconductor_detector_campaign() -> ScientificCampaign:
    """Predeclare analytic calibration, segmented lock, and refusal cases."""

    definitions = (
        (
            "detector-parallel-plate-calibration",
            "parallel-plate-independent-unit",
            "parallel-plate",
            "uniform-linear-dielectric",
            "parallel-plate-preparation",
            "parallel-plate-batch",
        ),
        (
            "detector-coax-calibration",
            "coax-independent-unit",
            "coaxial-cylinder",
            "radial-linear-dielectric",
            "coax-preparation",
            "coax-batch",
        ),
        (
            "detector-segmented-locked",
            "segmented-independent-unit",
            "segmented-planar-detector",
            "complete-electrode-prescribed-trajectory",
            "segmented-preparation",
            "segmented-batch",
        ),
        (
            "detector-incomplete-electrodes-locked",
            "incomplete-independent-unit",
            "segmented-planar-detector",
            "incomplete-electrode-refusal",
            "incomplete-preparation",
            "incomplete-batch",
        ),
        (
            "detector-invalid-route-locked",
            "invalid-route-independent-unit",
            "parallel-plate",
            "out-of-support-route-refusal",
            "invalid-route-preparation",
            "invalid-route-batch",
        ),
        (
            "detector-resource-overflow-locked",
            "overflow-independent-unit",
            "parallel-plate",
            "preallocation-resource-refusal",
            "overflow-preparation",
            "overflow-batch",
        ),
    )
    cases = tuple(
        ScientificCase(
            case_id,
            independent_unit,
            construct,
            condition,
            preparation,
            batch,
            (f"semiconductor-detector-source:{case_id}",),
        )
        for (
            case_id,
            independent_unit,
            construct,
            condition,
            preparation,
            batch,
        ) in definitions
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole(
                "calibration",
                (
                    "detector-parallel-plate-calibration",
                    "detector-coax-calibration",
                ),
            ),
            CampaignRole(
                "locked_evaluation",
                (
                    "detector-segmented-locked",
                    "detector-incomplete-electrodes-locked",
                    "detector-invalid-route-locked",
                    "detector-resource-overflow-locked",
                ),
            ),
        ),
        criteria_ids=(
            "bias-weighting-type-separation",
            "parallel-plate-and-coax-capacitance",
            "complete-electrode-partition",
            "capacitance-reciprocity",
            "shockley-ramo-endpoint-integral-closure",
            "prescribed-route-no-carrier-advance",
            "route-and-resource-refusal",
        ),
    )


def semiconductor_quantum_transport_campaign() -> ScientificCampaign:
    """Predeclare disjoint chain-transport calibration, lock, and refusal cases."""

    definitions = (
        (
            "semiconductor-landauer-transparent-calibration",
            "landauer-transparent-independent-unit",
            "uniform-transparent-chain",
            "small-bias-common-temperature",
            "landauer-transparent-preparation",
            "landauer-transparent-batch",
        ),
        (
            "semiconductor-landauer-resonant-calibration",
            "landauer-resonant-independent-unit",
            "single-resonant-level",
            "energy-dependent-analytic-transmission",
            "landauer-resonant-preparation",
            "landauer-resonant-batch",
        ),
        (
            "semiconductor-coherent-ac-locked",
            "coherent-ac-independent-unit",
            "screened-finite-lead-chain",
            "equilibrium-positive-adiabatic-rate",
            "coherent-ac-preparation",
            "coherent-ac-batch",
        ),
        (
            "semiconductor-finite-lead-transient-locked",
            "finite-lead-transient-independent-unit",
            "held-pulse-finite-lead-chain",
            "pre-recurrence-time-window",
            "finite-lead-transient-preparation",
            "finite-lead-transient-batch",
        ),
        (
            "semiconductor-optical-phonon-scba-locked",
            "optical-phonon-scba-independent-unit",
            "scalar-chain-local-fock-scba",
            "integer-phonon-shift-energy-grid",
            "optical-phonon-scba-preparation",
            "optical-phonon-scba-batch",
        ),
        (
            "semiconductor-undetermined-bound-state-refusal",
            "bound-state-refusal-independent-unit",
            "gapped-lead-bound-state-chain",
            "missing-bound-state-preparation",
            "bound-state-refusal-preparation",
            "bound-state-refusal-batch",
        ),
        (
            "semiconductor-quantum-resource-refusal",
            "quantum-resource-refusal-independent-unit",
            "scalar-chain-resource-overflow",
            "workspace-or-refinement-capacity-exceeded",
            "quantum-resource-refusal-preparation",
            "quantum-resource-refusal-batch",
        ),
    )
    cases = tuple(
        ScientificCase(
            case_id,
            independent_unit,
            construct,
            condition,
            preparation,
            batch,
            (f"semiconductor-quantum-source:{case_id}",),
        )
        for (
            case_id,
            independent_unit,
            construct,
            condition,
            preparation,
            batch,
        ) in definitions
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole(
                "calibration",
                (
                    "semiconductor-landauer-transparent-calibration",
                    "semiconductor-landauer-resonant-calibration",
                ),
            ),
            CampaignRole(
                "locked_evaluation",
                (
                    "semiconductor-coherent-ac-locked",
                    "semiconductor-finite-lead-transient-locked",
                    "semiconductor-optical-phonon-scba-locked",
                    "semiconductor-undetermined-bound-state-refusal",
                    "semiconductor-quantum-resource-refusal",
                ),
            ),
        ),
        criteria_ids=(
            "landauer-analytic-current-and-gauge",
            "coherent-ac-gauge-kcl-ward-and-refinement",
            "finite-lead-transient-unitarity-energy-and-return-window",
            "optical-phonon-scba-causality-kms-and-conservation",
            "bound-state-preparation-refusal",
            "quantum-resource-preallocation-refusal",
        ),
    )


__all__ = [
    "semiconductor_candidate_profile",
    "semiconductor_candidate_profiles",
    "semiconductor_detector_campaign",
    "semiconductor_quantum_transport_campaign",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""PIC qualification gates and unreleased PIC capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

import jax.numpy as jnp
import numpy as np

from ..discretization import (
    ChargedParticlePlan,
    ParticlePopulationPlan,
    ParticleSetPlan,
    PreparedTensorGrid,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from ..discretization.pic import (
    ChargeConservingCurrentPlan,
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICParticleCochainTransferPlan,
    PICSpeciesPlan,
    RelativisticPusher,
    RelativisticPushPlan,
)
from ..discretization.spectral._qualification import (
    distributed_spectral_candidate_profiles,
)
from ..qualification import CapabilityProfile, SupportTuple
from ._cochain_electrostatic import (
    CochainElectrostaticBoundaryPlan,
    CochainElectrostaticPlan,
)
from ._cochain_pic_field import CochainMaxwellPICFieldSolver
from ._electromagnetic_pic import ElectromagneticPICPlan
from ._electrostatic_pic import ElectrostaticPICPlan
from ._maxwell import CompatibleMaxwellPlan
from ._pic_current_source import PICMaxwellCurrentSourcePlan


_RADIATION_REACTION_GATES = (
    "planar-landau-lifshitz-cooling-analytic",
    "first-order-step-convergence",
    "radiated-equals-kinetic-loss",
    "reduced-equals-full-in-uniform-fields",
    "quantum-to-classical-limit",
    "niel-quantum-corrections",
    "fokker-planck-moment-equations",
    "identity-addressed-wiener-increments",
    "support-and-ownership-refusals",
    "species-charge-to-mass-refusal",
    "documentation-nonclaims",
)


@dataclass(frozen=True, slots=True)
class PICQualificationReport:
    """Host qualification evidence; every gate must pass for ``successful``."""

    poisson_residual: float
    charge_balance_defect: float
    electrostatic_energy_defect: float
    continuity_defect: float
    electromagnetic_continuity_defect: float
    particle_field_charge_defect: float
    gauss_defect: float
    deposit_gauss_pairing_defect: float
    pusher_speed_defect: float
    successful: bool


def _periodic_grid(count: int, dimension: int, /) -> PreparedTensorGrid:
    names = ("x", "y", "z")[:dimension]
    return TensorGridPlan(
        tuple(UniformCellAxisSpec(count, periodic=True) for _ in range(dimension)),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))


def _electrostatic_gates(smoke: bool, /) -> tuple[float, float, float, float, bool]:
    count = 16 if smoke else 32
    particle_count = 8 if smoke else 32
    bridge = StructuredCochainBridge(_periodic_grid(count, 1))
    base = (jnp.arange(particle_count, dtype=jnp.float64)[:, None] + 0.5) / particle_count
    transfers = []
    for offset, sign, name in ((0, -1.0, "negative"), (1000, 1.0, "positive")):
        support = ParticleSetPlan(
            jnp.arange(offset, offset + particle_count),
            jnp.ones((particle_count,)),
            ambient_dimension=1,
        ).prepare()
        charged = ChargedParticlePlan(sign * jnp.ones((particle_count,)), name).prepare(
            support
        )
        transfers.append(PICParticleCochainTransferPlan(bridge).prepare(charged))
    plan = ElectrostaticPICPlan(
        CochainElectrostaticPlan(
            bridge, CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        tuple(transfers),
    )
    state = plan.initialize(
        (base + 0.002 * jnp.sin(2.0 * jnp.pi * base), base),
        (jnp.zeros((particle_count, 1)), jnp.zeros((particle_count, 1))),
    )
    step = plan.step_detailed(state, 5.0e-4)
    diagnostics = step.diagnostics
    values = (
        float(diagnostics.poisson_residual),
        float(diagnostics.charge_balance_defect),
        float(diagnostics.energy.defect),
        float(diagnostics.continuity_defect),
    )
    passed = (
        bool(step.successful)
        and values[0] < 1.0e-8
        and values[1] < 1.0e-10
        and abs(values[2]) < 1.0e-8
        and values[3] < 1.0e-10
    )
    return (*values, passed)


def _electromagnetic_gates(smoke: bool, /) -> tuple[float, float, float, float, bool]:
    count = 3 if smoke else 4
    particle_count = 2 if smoke else 4
    bridge = StructuredCochainBridge(_periodic_grid(count, 3))
    transfer_plan = PICParticleCochainTransferPlan(bridge)
    transfers = []
    species = []
    for offset, sign, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        support = ParticleSetPlan(
            jnp.arange(offset, offset + particle_count),
            jnp.ones((particle_count,)),
            ambient_dimension=3,
        ).prepare()
        transfers.append(
            transfer_plan.prepare(
                ChargedParticlePlan(sign * jnp.ones((particle_count,)), name).prepare(
                    support
                )
            )
        )
        species.append(
            PICSpeciesPlan(
                ParticlePopulationPlan(support),
                PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    maxwell = CompatibleMaxwellPlan(
        bridge,
        sources=(PICMaxwellCurrentSourcePlan(),),
        plan_id="pic-qualification-maxwell",
    ).prepare()
    solver = CochainMaxwellPICFieldSolver(
        maxwell,
        CochainElectrostaticPlan(
            bridge, CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        tuple(transfers),
        tuple(ChargeConservingCurrentPlan(value) for value in transfers),
    )
    plan = ElectromagneticPICPlan(solver, species=species)
    slot = (np.arange(particle_count)[:, None] + np.asarray([0.3, 0.5, 0.7])) / (
        particle_count + 1
    )
    position = jnp.asarray(slot)
    velocity = jnp.zeros((particle_count, 3)).at[:, 0].set(0.1)
    dt = 0.01 * maxwell.stable_dt
    state = plan.initialize(
        (position + jnp.asarray([0.002, 0.0, 0.0]), position),
        (jnp.zeros((particle_count, 3)), velocity),
        dt,
    )
    step = plan.step_detailed(state, dt)
    diagnostics = step.diagnostics
    values = (
        float(diagnostics.continuity_defect),
        float(diagnostics.particle_field_charge_defect),
        float(diagnostics.electric_constraint),
        plan.pairing_defect,
    )
    passed = bool(step.successful) and max(values) < 1.0e-10
    return (*values, passed)


def _pusher_speed_defect() -> float:
    """Worst ``| |u⁺|² − |u⁻|² |`` of every pusher in a pure magnetic field."""
    proper = jnp.asarray([[0.2, 0.1, 0.0]])
    worst = 0.0
    for method in get_args(RelativisticPusher):
        pushed = RelativisticPushPlan(PIC_CODE_RELATIVITY, method=method).push(
            proper,
            jnp.zeros_like(proper),
            jnp.asarray([[0.0, 0.0, 0.7]]),
            jnp.asarray([1.0]),
            jnp.asarray([True]),
            1.0e-3,
        )
        if not bool(pushed.successful):
            return float("inf")
        worst = max(
            worst,
            float(jnp.abs(jnp.sum(pushed.proper_velocity**2) - jnp.sum(proper**2))),
        )
    return worst


def run_pic_qualification(*, smoke: bool = False) -> PICQualificationReport:
    """Evaluate the electrostatic, electromagnetic, and pusher PIC gates."""
    poisson, balance, energy, continuity, electrostatic_passed = _electrostatic_gates(
        smoke
    )
    em_continuity, charge_defect, gauss, pairing, electromagnetic_passed = (
        _electromagnetic_gates(smoke)
    )
    speed = _pusher_speed_defect()
    return PICQualificationReport(
        poisson,
        balance,
        energy,
        continuity,
        em_continuity,
        charge_defect,
        gauss,
        pairing,
        speed,
        electrostatic_passed and electromagnetic_passed and speed < 1.0e-10,
    )


def pic_radiation_reaction_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.radiation-reaction",
            {
                "models": "landau-lifshitz-reduced-full-quantum-corrected-fokker-planck",
                "coupling": "pic-momentum-stage-process-and-detector-propagation",
                "time_integration": "first-order-explicit-split-after-push",
                "field_derivatives": "order-one-gather-and-staggered-field-history",
                "quantum_corrections": "baier-katkov-g-and-h-tables-from-synchrotron-kernels",
                "stochasticity": "euler-maruyama-identity-addressed-wiener-increments",
                "ownership": "subgrid-reaction-with-grid-cutoff-scale-separation",
                "validity": "lcfa-declared-maximum-chi-and-minimum-gamma",
                "precision": "float64",
            },
        ),
    )


def pic_radiation_reaction_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_RADIATION_REACTION_GATES,
            released=False,
        )
        for support in pic_radiation_reaction_support_tuples()
    )


_SPECTRAL_GATES = (
    "vacuum-dispersion-exact-infinite-order",
    "vacuum-dispersion-modified-wavenumber-finite-order",
    "gauss-law-per-charge-conservation-mode",
    "godfrey-vay-predicted-nci-growth",
    "psatd-pml-reflection",
    "psatd-pml-gauss-law-outside-layers",
    "spectral-sheet-antenna-one-way-work-doppler-gaussian-beam",
    "local-guarded-equals-global-within-truncation",
    "dipole-far-field-matches-cochain",
    "compatibility-refusals",
    "restart-round-trip",
)
_GALILEAN_GATES = (
    "gauss-law-per-charge-conservation-mode",
    "galilean-suppresses-nci-drifting-plasma",
    "compatibility-refusals",
    "restart-round-trip",
)
_QUASI_CYLINDRICAL_GATES = (
    "shared-grid-hankel-laplacian-eigenfunctions-with-rank-evidence",
    "vacuum-tm-modes-exact-dispersion",
    "m1-gaussian-antenna-beam-matches-paraxial-waist-and-gouy",
    "gauss-law-per-charge-conservation-mode-and-shape",
    "near-axis-regular-modal-densities",
    "axisymmetric-field-equals-cartesian-psatd",
    "hertzian-dipole-far-field-through-huygens-cylinder",
    "radial-damping-absorption-ledger",
    "window-shift-with-gauss-projection",
    "compatibility-refusals",
    "restart-round-trip",
    "lwfa-wake-matches-pinned-fbpic",
)


def pic_spectral_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.psatd",
            {
                "geometry": "periodic-uniform-cartesian-3d",
                "time_integration": "exact-phi-function-psatd-constant-linear-multi-j",
                "charge_conservation": "spectral-correction-vay-deposition-update-with-rho",
                "stencils": "infinite-order-and-even-finite-order",
                "grids": "collocated-and-staggered",
                "decomposition": "global-fft-and-local-guarded",
                "absorber": "two-step-split-field-psatd-pml-with-layer-absorber-charge",
                "sources": "band-limited-moving-plane-sheet-antennas",
                "observation": "huygens-box-far-field",
                "precision": "float64",
            },
        ),
        SupportTuple(
            "pic.galilean-psatd",
            {
                "geometry": "periodic-uniform-cartesian-3d",
                "variants": "galilean-and-averaged-galilean",
                "particle_frame": "grid-coordinates-drifting-at-galilean-velocity",
                "charge_conservation": "update-with-rho-and-galilean-form-correction",
                "stability": "nci-monitor-high-k-shell-growth-fit",
                "precision": "float64",
            },
        ),
        SupportTuple(
            "pic.quasi-cylindrical",
            {
                "geometry": "azimuthal-modes-radial-hankel-periodic-axial",
                "transform": "shared-grid-hankel-pseudoinverse-orders-m-and-m-plus-minus-1",
                "time_integration": "exact-phi-function-psatd-constant-j",
                "variants": "standard-galilean-and-averaged-galilean",
                "charge_conservation": "update-with-rho-and-spectral-correction",
                "deposition": "azimuthal-mode-splines-order-1-to-3-near-axis-volumes",
                "absorber": "divergence-preserving-radial-damping",
                "sources": "per-mode-one-way-sheet-antennas-with-declared-charges",
                "observation": "huygens-cylinder-far-field",
                "moving_window": "axial-shift-with-spectral-gauss-projection",
                "oracle": "pinned-fbpic",
                "precision": "float64",
            },
        ),
    )


def pic_spectral_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "pic.psatd": _SPECTRAL_GATES,
        "pic.galilean-psatd": _GALILEAN_GATES,
        "pic.quasi-cylindrical": _QUASI_CYLINDRICAL_GATES,
    }
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=gates[support.capability],
            released=False,
        )
        for support in pic_spectral_support_tuples()
    )


_QED_CASCADE_GATES = (
    "compton-rate-vs-bessel-quadrature",
    "breit-wheeler-rate-vs-quadrature-and-erber-asymptotics",
    "photon-spectrum-kolmogorov-smirnov",
    "pair-spectrum-kolmogorov-smirnov",
    "small-chi-power-equals-quantum-corrected-landau-lifshitz",
    "improved-lcfa-infrared-spectrum",
    "declared-conservation-with-field-exchange-defect",
    "rotating-field-cascade-growth-vs-grismayer",
    "ledger-closure-with-field-exchange",
    "storage-slot-order-invariance",
    "atomic-capacity-refusal",
    "restart-component",
    "documentation-nonclaims",
)


def pic_qed_cascade_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.qed-cascade",
            {
                "processes": "nonlinear-compton-and-nonlinear-breit-wheeler",
                "models": "lcfa-and-improved-lcfa-di-piazza-2019",
                "tables": "host-quadrature-of-synchrotron-kernels-fingerprinted-monotone-cdf",
                "monte_carlo": "optical-depth-with-bounded-adaptive-subcycling",
                "randomness": "identity-addressed-step-id-event-keys",
                "coupling": "pic-creation-stage-before-drift-and-deposit",
                "conservation": "declared-momentum-or-energy-with-field-exchange-ledger",
                "photons": "process-owned-ballistic-bank-with-escape-histograms",
                "capacity": "atomic-refusal-and-occupancy-merge-trigger",
                "validity": "per-event-formation-ratio-trident-and-splitting-flags",
                "ownership": "subgrid-reaction-with-grid-cutoff-scale-separation",
                "precision": "float64",
            },
        ),
    )


def pic_qed_cascade_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_QED_CASCADE_GATES,
            released=False,
        )
        for support in pic_qed_cascade_support_tuples()
    )


_POLARIZED_QED_GATES = (
    "polarized-tables-average-to-unpolarized-tables",
    "polarized-rates-vs-seipt-king-quadrature",
    "channel-spectra-vs-seipt-king",
    "sokolov-ternov-flip-rate-and-equilibrium",
    "monte-carlo-radiative-polarization-relaxation",
    "tbmt-anomaly-precession-all-pushers",
    "pic-spin-precession-with-run-pusher",
    "photon-polarization-degree-vs-lcfa",
    "polarized-pair-rates-and-pair-spins",
    "galilean-grid-qed-equals-lab-grid",
    "spin-polarized-restart-component",
    "documentation-nonclaims",
)


def pic_polarized_qed_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.polarized-qed",
            {
                "models": "photon-polarized-and-spin-and-photon-polarized",
                "rates": "seipt-king-2020-lcfa-spin-quantization-axis-channels",
                "tables": "polarized-compton-and-breit-wheeler-mixture-components",
                "spin": "identity-carried-polarization-vector-quantum-jump",
                "no_event_evolution": "mass-operator-and-vacuum-dichroism",
                "precession": "tbmt-consistent-with-boris-vay-higuera-cary",
                "photons": "linear-stokes-rotated-to-local-field-basis",
                "grids": "lab-and-galilean",
                "precision": "float64",
            },
        ),
    )


def pic_polarized_qed_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_POLARIZED_QED_GATES,
            released=False,
        )
        for support in pic_polarized_qed_support_tuples()
    )


_DISTRIBUTED_GATES = (
    "n-devices-equal-one-device-within-reduction-order",
    "halo-accumulation-equals-global-deposit",
    "migration-overflow-atomic-rejection",
    "same-topology-restart-bitwise",
    "repartition-restart-within-tolerance",
    "window-locality-refusal",
    "local-guarded-equals-global-fft-within-stencil-truncation",
    "distributed-processes-equal-one-device-identities-bitwise",
    "weak-and-strong-scaling-benchmark",
)


def pic_distributed_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.distributed",
            {
                "decomposition": "static-equal-blocks-1d-2d-3d-mesh-restart-repartition",
                "field_solvers": "cochain-reduced-and-psatd-consumer-configurations",
                "deposition": "window-local-axis-by-axis-halo-accumulation-plus-uniform",
                "migration": "fixed-capacity-ppermute-packets-diagonals-atomic-rejection",
                "processes": (
                    "per-device-creation-population-stateful-resampling-"
                    "tile-reserved-identities"
                ),
                "restart": "lifecycle-addressable-shards-bitwise-same-topology",
                "precision": "float64",
            },
        ),
    )


def pic_distributed_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    spectral_profile = distributed_spectral_candidate_profiles()[0]
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            dependencies=(spectral_profile.profile_id,),
            required_gates=_DISTRIBUTED_GATES,
            released=False,
        )
        for support in pic_distributed_support_tuples()
    )


_DISPERSIVE_SELF_CONSISTENT_GATES = (
    "open-boundary-gauss-and-charge-ledger-closure",
    "particle-exit-charge-mass-energy-ledger",
    "reduced-2d-nonperiodic-pairing-roundoff",
    "dispersive-magnetized-energy-ledger-second-order",
    "cpml-absorbed-energy-in-ledger",
    "weak-coupling-beam-equals-prescribed-charge-cherenkov",
    "filtered-gauss-initialization",
    "ownership-and-medium-refusals",
    "documentation-nonclaims",
)


def pic_dispersive_self_consistent_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.dispersive-self-consistent",
            {
                "geometry": "cartesian-3d-cochain-and-reduced-1d-2d-periodic-or-bounded",
                "boundaries": "cpml-pec-pmc-impedance-with-particle-exit-ledger",
                "media": "linear-passive-lossy-lorentz-drude-magnetic-poles-magnetized-plasma",
                "deposition": "charge-conserving-spline-whitney-clipped-at-walls",
                "gauss": "current-driven-pairing-with-induced-wall-and-medium-charge",
                "energy_ledger": "leapfrog-field-medium-split-trapezoidal-losses",
                "initialization": "filtered-charge-electrostatic-grounded-or-periodic",
                "ownership": "field-solver-radiation-claim-refused-on-overlap",
                "precision": "float64",
            },
        ),
    )


def pic_dispersive_self_consistent_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_DISPERSIVE_SELF_CONSISTENT_GATES,
            released=False,
        )
        for support in pic_dispersive_self_consistent_support_tuples()
    )


_BOOSTED_FRAME_GATES = (
    "particle-boost-contraction-and-velocity-addition",
    "undulator-gather-only-magnetic-type-subluminal",
    "boosted-tracks-through-trajectory-radiation-equal-lab",
    "back-transformed-vacuum-laser-equals-lab",
    "lwfa-energy-gain-boosted-equals-lab",
    "back-transformed-wake-and-particles-match-lab-run",
    "nci-guard-rejection",
    "moving-antenna-world-line",
    "refusals",
    "restart-round-trip",
)


def pic_boosted_frame_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "pic.boosted-frame",
            {
                "frame": "pure-lorentz-boost-along-one-grid-axis",
                "field_solver": "galilean-psatd-comoving-with-boosted-plasma",
                "particles": "proper-velocity-boost-and-ballistic-slice-loading",
                "external_fields": "gather-only-boosted-lab-sources",
                "antennas": "moving-sampled-plane-sheet",
                "stability": "mandatory-high-k-nci-guard",
                "diagnostics": "back-transformed-lab-fields-and-particles-bounded-ring",
                "tracks": "lab-frame-per-lane-times-for-trajectory-radiation",
                "far_field": "huygens-standard-grid-zero-surface-current-vacuum-only",
                "precision": "float64",
            },
        ),
    )


def pic_boosted_frame_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_BOOSTED_FRAME_GATES,
            released=False,
        )
        for support in pic_boosted_frame_support_tuples()
    )


__all__ = [
    "PICQualificationReport",
    "pic_boosted_frame_candidate_profiles",
    "pic_boosted_frame_support_tuples",
    "pic_dispersive_self_consistent_candidate_profiles",
    "pic_dispersive_self_consistent_support_tuples",
    "pic_distributed_candidate_profiles",
    "pic_distributed_support_tuples",
    "pic_polarized_qed_candidate_profiles",
    "pic_polarized_qed_support_tuples",
    "pic_qed_cascade_candidate_profiles",
    "pic_qed_cascade_support_tuples",
    "pic_radiation_reaction_candidate_profiles",
    "pic_radiation_reaction_support_tuples",
    "pic_spectral_candidate_profiles",
    "pic_spectral_support_tuples",
    "run_pic_qualification",
]

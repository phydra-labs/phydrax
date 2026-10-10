#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased accelerator capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_INSERTION_DEVICE_GATES = (
    "undulator-resonance-and-odd-harmonics",
    "harmonic-line-width",
    "matched-termination-exit-orbit",
    "exit-delay-convention",
    "bend-field-integral",
    "tabulated-map-interpolation-bound",
    "field-support-refusal",
    "resource-envelope",
    "documentation-nonclaims",
)

_RING_RADIATION_GATES = (
    "weak-focusing-closed-form-integrals",
    "isomagnetic-fodo-hill-equation-integrals",
    "published-radiation-constants-equilibrium",
    "photon-spectrum-mellin-moments",
    "zero-radiation-symplectic-identity",
    "classical-loss-per-turn",
    "partitioned-damping-rates",
    "stochastic-equilibrium-convergence",
    "identity-addressed-emission",
    "unsupported-lattice-refusal",
    "resource-envelope",
    "documentation-nonclaims",
)

_FEL_AVERAGED_GATES = (
    "one-dimensional-cold-growth-rate",
    "one-dimensional-energy-spread-dispersion",
    "ming-xie-gain-length",
    "saturation-power-scale",
    "quiet-start-and-fawley-shot-noise",
    "harmonic-bessel-coupling",
    "particle-field-energy-ledger",
    "taper-and-focusing-transfer",
    "per-slice-wake-loss",
    "identity-addressed-loading",
    "unsupported-configuration-refusal",
    "documentation-nonclaims",
)

_FEL_TIME_DEPENDENT_GATES = (
    "uniform-periodic-beam-reduces-to-time-independent",
    "commensurate-slippage-exact-roll",
    "spectral-slippage-exact-translation",
    "open-window-padding-and-exit-ledger",
    "seeded-spectrum-linear-theory-transfer",
    "sase-gamma-statistics-and-coherence-time",
    "hghg-eehg-bunching-bessel-formulas",
    "per-slice-wake-and-space-charge",
    "whole-window-energy-ledger",
    "genesis4-pinned-oracle",
    "unsupported-configuration-refusal",
    "documentation-nonclaims",
)

_CSR_1D_GATES = (
    "saldin-derbenev-steady-wake",
    "gaussian-steady-energy-loss-closed-form",
    "saldin-1997-entrance-transient",
    "murphy-krinsky-gluckstern-parallel-plate-shielding",
    "shielding-truncation-evidence",
    "history-completeness-refusal",
    "chicane-first-order-optics",
    "tracked-loss-integrates-wake",
    "ocelot-chicane-emittance-oracle",
    "resource-envelope",
    "documentation-nonclaims",
)

_CSR_3D_GATES = (
    "thin-beam-reduces-to-1d",
    "round-beam-residual-centripetal-coefficient-two",
    "retarded-mesh-agrees-with-steady-igf-longitudinal-and-horizontal",
    "retarded-mesh-particle-and-kernel-width-convergence",
    "kernel-quadrature-defect-evidence",
    "pycsr3d-wake-oracle",
    "resource-envelope",
    "documentation-nonclaims",
)

_WAKE_IMPEDANCE_GATES = (
    "resonator-point-bunch-loss-factor",
    "wake-causality-behind-source",
    "resonator-wake-impedance-fourier-pair",
    "transverse-resonator-impedance-closed-form",
    "resistive-wall-bane-sands-short-and-long-range",
    "dipolar-source-and-quadrupolar-witness-offsets",
    "coupled-bunch-growth-rate-impedance-sum",
    "unit-and-causality-refusals",
    "documentation-nonclaims",
)

_SPACE_CHARGE_IGF_GATES = (
    "single-cell-kernel-quadrature",
    "tent-density-exact-gradient-kernel",
    "uniform-ellipsoid-depolarization-field",
    "gaussian-charge-closed-form-field",
    "integrated-kernel-converges-faster-than-point-green-function",
    "elongated-gaussian-any-aspect-ratio",
    "relativistic-gaussian-bunch-boosted-coulomb-kick",
    "declared-frame-not-reference-momentum",
    "cells-per-sigma-and-admissibility-refusals",
    "documentation-nonclaims",
)

_FEL_FULL_WAVE_GATES = (
    "seeded-small-signal-gain-equals-averaged-fel",
    "seed-antenna-plane-wave-and-closed-ledger",
    "spontaneous-huygens-spectrum-equals-trajectory-radiation",
    "prebunched-steady-amplitude-equals-kmr",
    "prebunched-coherent-power-current-times-bunching-squared",
    "boosted-run-equals-lab-frame-pic",
    "unsupported-configuration-refusal",
    "documentation-nonclaims",
)


def _support_gates() -> tuple[tuple[SupportTuple, tuple[str, ...]], ...]:
    return (
        (
            SupportTuple(
                "accelerator.insertion-device-radiation",
                {
                    "fields": "planar-helical-undulator-wiggler-dipole-bend-tabulated-3d",
                    "field_model": "exact-vacuum-analytic-or-multilinear-table",
                    "frame": "straight-cartesian-map-frame",
                    "tracking": "lab-time-relativistic-push-per-lane-arrival",
                    "coordinates": "accelerator-convention-s-to-t",
                    "radiation": "vacuum-trajectory-radiation-far-field",
                    "precision": "float64-only",
                },
            ),
            _INSERTION_DEVICE_GATES,
        ),
        (
            SupportTuple(
                "accelerator.ring-radiation",
                {
                    "lattice": "periodic-cell-drift-quadrupole-sector-bend-thin-rf",
                    "bends": "horizontal-normal-entry-sector-optional-gradient",
                    "optics": "first-order-uncoupled-periodic",
                    "integrals": "sands-i1-i5-gauss-kronrod",
                    "radiation": "classical-ultrarelativistic-collinear-emission",
                    "spectrum": "synchrotron-function-photon-number-table",
                    "tracking": "sliced-bend-kicks-classical-or-poisson-photons",
                    "randomness": "particle-identity-addressed",
                    "precision": "float64-only",
                },
            ),
            _RING_RADIATION_GATES,
        ),
        (
            SupportTuple(
                "accelerator.fel-averaged",
                {
                    "model": "kmr-period-averaged-time-independent-slices",
                    "lattice": "planar-or-helical-modules-stepwise-taper-thin-quad-breaks",
                    "harmonics": "odd-planar-bessel-coupled-or-helical-fundamental",
                    "field": "one-dimensional-or-angular-spectrum-grid",
                    "integration": "strang-betatron-diffraction-rk4-source",
                    "loading": "quiet-start-fawley-shot-noise-identity-addressed",
                    "collective": "per-slice-longitudinal-wake-loss",
                    "precision": "float64-only",
                },
            ),
            _FEL_AVERAGED_GATES,
        ),
        (
            SupportTuple(
                "accelerator.fel-time-dependent",
                {
                    "model": "kmr-period-averaged-coupled-slices-with-slippage",
                    "slippage": "commensurate-roll-or-spectral-phase-ramp-in-strang",
                    "window": "periodic-or-open-with-head-padding-exit-ledger",
                    "field": "one-dimensional-or-angular-spectrum-grid",
                    "start": "sase-fawley-shot-noise-or-optics-pulse-envelope-seed",
                    "prebunching": "hghg-eehg-modulators-and-symplectic-maps",
                    "collective": "per-slice-wake-and-igf-space-charge-rates",
                    "diagnostics": "temporal-spectral-power-spikes-coherence-ledger",
                    "precision": "float64-only",
                },
            ),
            _FEL_TIME_DEPENDENT_GATES,
        ),
        (
            SupportTuple(
                "accelerator.csr-1d",
                {
                    "models": "1d-steady-saldin-derbenev-or-1d-transient-mayes-hoffstaetter",
                    "path": "planar-drift-and-arc-lattice-with-incoming-straight-line",
                    "retardation": "exact-line-charge-with-straight-space-charge-subtracted",
                    "history": "bounded-density-ring-with-incoming-drift-model",
                    "shielding": "free-space-or-parallel-plate-image-series",
                    "tracking": "first-order-split-step-with-telescoping-potential",
                    "precision": "float64-only",
                },
            ),
            _CSR_1D_GATES,
        ),
        (
            SupportTuple(
                "accelerator.csr-3d",
                {
                    "models": "3d-steady-igf-exact-lorentz-or-3d-retarded-mesh",
                    "kernel": "cell-integrated-green-function-hockney-doubled-grid",
                    "retardation": "smooth-deposited-density-over-recorded-history",
                    "sources": "reference-path-with-transverse-translation-invariance",
                    "shielding": "free-space-only",
                    "precision": "float64-only",
                },
            ),
            _CSR_3D_GATES,
        ),
        (
            SupportTuple(
                "accelerator.wake-impedance",
                {
                    "kinds": "longitudinal-dipolar-quadrupolar-x-y",
                    "wakes": "tabulated-resonator-resistive-wall-bane-sands",
                    "impedance": "type3-nonuniform-fourier-and-inverse",
                    "application": "binned-convolution-with-self-half-rule",
                    "memory": "multi-bunch-multi-turn-passage-history",
                    "precision": "float64-only",
                },
            ),
            _WAKE_IMPEDANCE_GATES,
        ),
        (
            SupportTuple(
                "accelerator.space-charge-igf",
                {
                    "frame": "reference-particle-rest-frame-boost-fields",
                    "kernel": "cell-integrated-coulomb-igf-face-difference-field",
                    "boundary": "open-free-space-doubled-grid-convolution",
                    "deposit": "multilinear-splat-and-gather",
                    "admissibility": "rest-frame-speed-and-cells-per-sigma",
                    "precision": "float64-only",
                },
            ),
            _SPACE_CHARGE_IGF_GATES,
        ),
        (
            SupportTuple(
                "accelerator.fel-full-wave",
                {
                    "frame": "pure-boost-near-undulator-resonance-with-frame-evidence",
                    "field": "standard-staggered-psatd-nci-guard-open-axis-psatd-pml",
                    "undulator": "boosted-insertion-device-gather-only-external-field",
                    "beam": "lab-macroparticles-injected-ballistically-image-pairs",
                    "seed": "boosted-one-way-plane-antenna-flat-top",
                    "radiation": "lab-track-trajectory-radiation-or-admissible-huygens",
                    "ledger": "lab-beam-field-escaped-injected-energy",
                    "precision": "float64-only",
                },
            ),
            _FEL_FULL_WAVE_GATES,
        ),
    )


def accelerator_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(support for support, _ in _support_gates())


def accelerator_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            (support,),
            required_gates=gates,
            released=False,
        )
        for support, gates in _support_gates()
    )


__all__ = ["accelerator_candidate_profiles", "accelerator_support_tuples"]

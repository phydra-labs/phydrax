#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dark-matter transport, source-yield, and detector-facing workflows."""
# ruff: noqa: F401

from ._indirect_detection import (
    annihilation_flux,
    BinnedIndirectDetectionEvidence,
    BinnedIndirectDetectionPlan,
    BinnedIndirectDetectionResult,
    decay_flux,
    DFactor,
    ExactLineFluxTable,
    FluxNormalizationEvidence,
    IndirectDetectionStatus,
    IndirectFluxResult,
    JFactor,
)
from ._profiles import (
    LayeredTerrestrialProfile,
    RadialBodyProfile,
    RadialProfileEvaluation,
    SmoothStellarRadialProfile,
)
from ._rates import (
    CrossingDirection,
    CrossingStatus,
    observation_flux,
    ObservationFlux,
    spherical_surface_crossings,
    SurfaceCrossingDiagnostics,
    SurfaceCrossingMeasure,
)
from ._scattering import (
    BoundedThermalMarkSamplerPlan,
    elastic_scatter_velocity,
    ElasticCollisionResult,
    ElasticScatteringTable,
)
from ._solar import (
    classify_solar_outcomes,
    ExteriorKeplerTransportResult,
    gravitational_focusing_speed,
    kepler_specific_energy,
    propagate_exterior_kepler,
    SolarOutcomeClassification,
    SolarTransportPlan,
    SolarTransportResult,
    stellar_specific_energy,
    stellar_specific_energy_evidence,
    StellarSpecificEnergyEvidence,
)
from ._terrestrial import (
    HazardIntegrationResult,
    HazardQuadraturePlan,
    layered_analytic_optical_depth,
    quadrature_optical_depth,
    sample_target_from_partial_rates,
    TerrestrialTransportPlan,
    TerrestrialTransportResult,
)
from ._transport import (
    BodyFrameTransportState,
    jump_differential_problem,
    ProfiledElasticJumpProcess,
    radial_guard_schedule,
    RadialGravityDrift,
    RadialSignedDistanceGuard,
    transparent_state,
    transport_collision_evidence,
    transport_path_evidence,
    TransportCollisionEvidence,
    TransportNumericalStatus,
    TransportOutcome,
    TransportPathEvidence,
)
from ._yields import (
    AnnihilationProcessDescriptor,
    DarkMatterProcessKind,
    DecayProcessDescriptor,
    ExactLineTable,
    ExternalYieldProviderResult,
    mix_particle_yields,
    ParticleYieldSpectrum,
    ProviderExecution,
    YieldIntegralEvidence,
    YieldProviderStatus,
    YieldUncertainty,
)


__all__ = [name for name in globals() if not name.startswith("_")]

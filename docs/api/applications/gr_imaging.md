# General-relativistic imaging

Observer screens, fixed-capacity ray trajectories and ordered events, transported ray
bundles, fast-/slow-light plasma fields, source-evidenced Stokes-$I$ thermal
synchrotron with explicitly unqualified polarization/Faraday approximations, typed
ray paths with chart/path/snapshot-bound active-segment midpoint sampling, invariant
transfer, Stokes images with canonical `phydrax.units.JANSKY`, interferometry, neutral
array payloads, and fixed-branch inference. See the
[GR imaging guide](../../guides_black_hole_imaging.md).

`gr_chart_identity(metric)` content-addresses only the four-dimensional coordinate
schema—chart name and ordered coordinate names. `gr_metric_identity(metric)` separately
content-addresses that chart ID, Lorentzian convention, and the metric matrix callable's
content. Therefore Kerr metrics with different mass/spin parameters may share one chart
ID but must have different metric IDs. Screen, ray-result, and `PolarizedRayPath`
composition checks both; same coordinates do not authorize substituting another
parameterized metric.

Inspectable callables are fingerprinted from their canonical callable payload. An opaque
metric callable must supply nonempty `semantic_id` and `numeric_id` together when its
identity is formed; one ID alone is rejected. These are content identities, not display
labels or a claim that equal chart names imply equal geometry.

::: phydrax.applications.astrophysics
    options:
      show_root_heading: true
      members:
        - gr_chart_identity
        - gr_metric_identity
        - GRTemporalDirection
        - GRObserverScreenPlan
        - GRObserverScreenResult
        - initialize_gr_observer_screen
        - GRRayEventCode
        - GRRayEventMargin
        - GRRayEventSurfaces
        - GRRayEventLedger
        - ordered_gr_ray_event_code
        - GRRayStatus
        - GRRayStatusEvidence
        - gr_ray_status_message
        - GRRayKind
        - GRRayState
        - GRRayPlan
        - GRRayResult
        - trace_gr_rays
        - AbstractGRConstantOfMotion
        - GRCallableConstantOfMotion
        - GRCoordinateMomentumConstant
        - GRJacobiEvidence
        - GRRayBundleEvidence
        - build_gr_ray_bundle_evidence
        - GRMediumFieldUnits
        - GRMediumSampleEvidence
        - GRMediumSample
        - FixedGRFieldSamplingPlan
        - FastLightSnapshot
        - FixedGRWorldtubeSamplingPlan
        - MonotoneSlowLightWorldtube
        - ThermalSynchrotronUnitContract
        - ThermalSynchrotronDomain
        - ThermalSynchrotronEvidence
        - ThermalSynchrotronCoefficients
        - InvariantSynchrotronCoefficients
        - ThermalSynchrotronModel
        - invariant_synchrotron_coefficients
        - InvariantTransferUnitContract
        - InvariantScalarTransferEvidence
        - InvariantScalarTransferResult
        - InvariantScalarTransferPlan
        - PolarizedRayPathEvidence
        - PolarizedRayPath
        - PolarizedTransferEvidence
        - PolarizedInvariantTransferResult
        - PolarizedInvariantTransferPlan
        - stokes_basis_rotation
        - rotate_stokes_coefficients
        - GRImageScreen
        - StokesImage
        - VisibilitySampling
        - StokesVisibilityData
        - direct_stokes_visibilities
        - apply_station_gains
        - PolarizationVisibilityProducts
        - polarization_visibility_products
        - ClosureTopology
        - ClosureProducts
        - closure_products
        - InterferometryStatus
        - interferometry_status_message
        - NeutralArrayPayload
        - stokes_image_to_fits_payload
        - stokes_image_from_fits_payload
        - visibility_data_to_uvfits_payload
        - visibility_data_from_uvfits_payload
        - GRInferenceEvaluation
        - FixedBranchRayInferencePlan
        - fixed_branch_ray_inverse_adapter
        - gr_ray_model_evaluation
        - GRPosteriorRealizationBinding
        - GRPosteriorPrediction

# Camera calibration and particle tracking velocimetry

PTV reconstructs individual particles and their identities. The native workflow is detection, geometric association, robust triangulation, temporal association, and optional frozen-association smoothing.

## Cameras and rigs

A native camera maps world points to `(row_down, column_right)` pixels and pixels to world-space rays. `CameraPose` wraps the existing `RigidFrame` as a named camera-from-world transform; there is no second rigid-transform convention.

Pinhole intrinsics and Brown-Conrady distortion operate in normalized camera coordinates. Planar refractive stacks trace rays with explicit interface geometry and refractive indices. Total internal reflection, parallel geometry, failed roots, and nonfinite values are statuses—not identity corrections.

A camera rig fixes camera order, world frame, length unit, calibration identities, validity, and optional timing offsets.

## Calibration

Calibration observations carry known world points, observed pixels, camera indices, support, and localization covariance. The calibration problem declares free/fixed parameters and gauge constraints. Robust nonlinear least squares returns initial/final rigs, per-observation residuals and weights, inliers, rank/conditioning, parameter covariance when observable, and holdout reconstruction evidence.

A low reprojection residual alone does not establish 3-D observability. Holdout points, camera-baseline geometry, and normal-matrix rank remain explicit.

## Detection

Point-particle detection uses mask-aware background correction/filtering, local extrema, nonmaximum suppression, and centroid/second-moment refinement. The fixed-capacity result records pixel position, localization covariance, integrated intensity, radius/size, validity, status, and overflow evidence. Capacity overflow never silently changes the scientific population.

## Multi-view association

Two-view one-to-one association uses geometric gates and explicit unmatched dummy decisions before Hungarian assignment. An absent match is not a fabricated image target.

For three or more cameras, compatible detection tuples are generated with fixed capacity. Each tuple records camera indices, support count, triangulation/reprojection diagnostics, appearance score, and ambiguity. A generic weighted set-packing solve prohibits reuse of any image detection. Exact and heuristic methods report different certification evidence.

## Triangulation

Weighted all-ray triangulation minimizes orthogonal distance to valid rays and reports per-camera residual, view count, rank, condition, and covariance when identifiable. Near-parallel or insufficient-view geometry fails instead of returning a large plausible coordinate. Optional robust refinement holds association fixed while downweighting inconsistent views.

## Temporal association

Streaming tracking predicts each active state and covariance, gates measurements, solves one-to-one association with explicit misses, updates matched tracks, ages misses, terminates expired tracks, and births unmatched particles into deterministic free slots. Track IDs are stable and never reused during one result.

Offline min-cost-flow refinement is a separate result. It may alter the association graph but never silently rewrites the streaming record.

## Smoothing and downstream data

Kalman filtering/smoothing is applied only after associations are frozen. It returns innovations, filtered/smoothed states, covariance, support, and status.

`TrackResult` remains canonical because it carries identity, observation provenance, association evidence, covariance, birth/death, and gaps. Conversion to `TrajectoryData` creates one case per track, increasing timestamps, invalid transitions across gaps, and reset boundaries. It does not discard the original tracking result.

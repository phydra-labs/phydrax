# Velocimetry

`phydrax.velocimetry` separates four scientific problems that are often conflated:

- Particle image velocimetry (PIV) estimates an Eulerian image-displacement field from particle-pattern correlations.
- Dense PIV estimates image displacement at pixel resolution; it is still not particle tracking.
- Particle tracking velocimetry (PTV) detects particles, reconstructs multi-camera positions, and infers identity-bearing trajectories.
- Shake-the-Box-style Lagrangian particle tracking refines predicted particles against residual camera images.

## Coordinates and quantities

Image arrays use `(row, column)`. Row increases down and column increases right. Image coordinates and image displacements therefore use `(row_down, column_right)` component order. Physical planar and camera/world coordinates use explicitly named right-handed frames.

An image displacement is not a physical velocity. PIV first estimates displacement in pixels. An explicit planar calibration maps the interrogation centres and displacement vectors to physical space; explicit positive frame timing then converts physical displacement to velocity.

Camera projection returns `(row, column)` positions. World points and reconstructed tracks use right-handed `(x, y, z)` coordinates. Results carry coordinate-frame, unit, source, plan, and calibration identities so that incompatible data cannot be combined silently.

## Measurement evidence

Every workflow keeps these concepts separate:

1. Raw measurement.
2. Geometric or image support.
3. Algorithm status.
4. Validation evidence.
5. Optional replacement or reconstruction.
6. Quality diagnostics.
7. Statistical uncertainty when an uncertainty model actually exists.

Zero is valid data. Invalid or missing observations use masks and status evidence, never a zero sentinel. PIV vector replacement creates a new result and replacement mask; it does not overwrite the measured field. Correlation peak ratio, signal-to-noise diagnostics, reprojection error, ray separation, and residual image loss are quality measures rather than covariance.

## Plans, preparation, and resources

Velocimetry follows the Phydrax lifecycle:

```text
immutable plan -> prepared fixed-shape resources -> runtime state/result
```

Preparation fixes interrogation grids, FFT sizes, memory batches, camera and particle capacities, candidate budgets, precision roles, retention policy, and deterministic tie rules. Runtime kernels operate on fixed-shape arrays with explicit validity masks. Capacity exhaustion is reported; detections, candidates, tracks, or images are never silently truncated.

## Classical PIV

The classical pipeline is:

```text
image pair
  -> supported interrogation windows
  -> mask-aware FFT correlation
  -> primary and alternate peaks
  -> subpixel refinement
  -> predictor/deformation passes
  -> validation evidence
  -> optional separate replacement
  -> planar calibration and timing
```

Linear correlation has nonperiodic lag topology; circular correlation is an explicit periodic option. Extended search areas increase measurable displacement without wraparound. Multipass deformation may use symmetric half-warps or a declared second-frame warp. The final result retains enough peak and support evidence to distinguish weak, ambiguous, clipped, and invalid vectors.

Ensemble PIV accumulates correlation evidence before peak selection. Residual particle-disparity analysis is reported as a resolution/error diagnostic unless a separate qualification establishes a calibrated uncertainty mapping.

## Cameras and PTV

A camera model provides two operations:

```text
world points -> projected image points
image points -> world-space rays
```

Native models include pinhole projection, Brown-Conrady distortion, fixed rigid camera poses, and planar refractive layer stacks. Calibration exposes gauge, observability, robust residual, rank, conditioning, and holdout evidence. A failed refractive or calibration solve never returns an identity correction.

Sparse PTV performs point-particle detection, geometric candidate generation, conflict-free multi-camera association, weighted robust triangulation, temporal association, and optional frozen-association smoothing. Two-view and adjacent-time one-to-one problems use explicit unmatched dummy decisions with Hungarian assignment. Multi-camera tuple conflicts use weighted set packing. Reconstructed tracks retain births, deaths, gaps, association provenance, and covariance.

`TrajectoryData` is a downstream adapter after identities are reconstructed. It is not the native detection, association, or tracking data model.

## Lagrangian image refinement

The high-density workflow is:

```text
predict tracks
  -> project and render predicted particles
  -> refine position and intensity against camera images
  -> subtract explained image evidence
  -> reconstruct particles from residual images
  -> associate new and continued particles
  -> promote or terminate tracks
```

The radiometric particle-image renderer is distinct from conservative particle-grid deposition. Continuous rendering and Shake refinement are differentiable while projection support, detections, candidate tuples, and identity assignments remain discrete.

## Learned dense PIV

The learned backend shares only the neutral dense image-displacement contract with classical PIV. It uses native mask-aware features, bounded local cost volumes, nonperiodic backward warping, component-correct multiscale flow resizing, and explicit supervised or photometric objectives. It does not import external architectures, kernels, weights, or confidence claims.

Training and qualification splits are made by scenario family rather than augmented frame. A trained artifact records normalization, coordinate convention, model configuration, corpus and split identities, precision evidence, and qualification identities.

## Interoperability

The native archive is canonical. External readers and writers return an adapter report that lists transformed, synthesized, unsupported, and dropped semantics. OpenPIV, PIVlab, xarray/pivpy, and OpenPTV-style formats are never treated as the internal data model.

PIV physical fields can adapt to a compatible `PreparedTensorGrid` or masked state-space observation sequence. PTV tracks can adapt to `TrajectoryData` with explicit reset boundaries at broken links. Invalid observations remain masks rather than zero-valued samples.

## Detailed guides

- [Classical PIV](guides_velocimetry_piv.md)
- [Camera calibration and PTV](guides_velocimetry_ptv.md)
- [Residual-image LPT](guides_velocimetry_stb.md)
- [Learned dense PIV](guides_learned_piv.md)
- [Persistence and interoperability](guides_velocimetry_interop.md)

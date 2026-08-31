# Residual-image Lagrangian particle tracking

The high-density tracking path refines predicted physical particles directly against camera images and reconstructs additional particles from residual evidence. It is separate from framewise sparse PTV.

## Radiometric image formation

The particle renderer projects valid 3-D points through a calibrated rig and deposits bounded Gaussian point-spread support into each camera image. Intensity, radius, exposure, gain, background, noise, and clipping are explicit policies. Rendering reports particles outside the sensor, invalid projections, clipped support, saturation, and resource overflow.

Radiometric rendering does not reuse conservative mass-deposition semantics. It may share fixed-route and deterministic scatter mechanics, but pixel intensity is an observation model rather than a conserved measure.

## Iterative particle reconstruction

IPR starts from masked residual images. It detects candidate image evidence, generates geometrically compatible camera tuples, reconstructs and refines 3-D candidates, rejects duplicates/ghosts, renders accepted particles, and updates residuals. The camera-subset policy and candidate counts are fixed resources. Reduced camera subsets require at least two views and retain their provenance.

## Shake refinement

Shake holds discrete camera support and particle identity fixed while optimizing continuous 3-D position and intensity against masked image residual. An accepted step must reduce the declared robust objective. The result preserves initial/final particle parameters, per-camera residuals, convergence, unsupported views, and ghost/repeat evidence.

The objective is differentiable with respect to continuous particle parameters while raster support-route changes remain discrete.

## Streaming state

One STB step performs:

```text
predict active tracks
-> render predictions
-> Shake predicted particles
-> subtract explained image signal
-> IPR on residual images
-> merge new particles
-> associate observations and tracks
-> promote, age, or terminate tracks
```

Short, active, and terminated tracks remain distinguishable. Image detections, reconstructed particles, and track identities use separate IDs. Capacity overflow is a result status rather than truncation.

The native implementation covers point tracers. Finite bubbles/fibres and online volume self-calibration require different image and calibration contracts and are intentionally outside this surface.

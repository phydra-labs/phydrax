# Classical particle image velocimetry

Classical PIV estimates displacement from the statistical motion of particle-image patterns. Its output is an Eulerian grid field; it does not identify or track individual particles.

## Workflow

1. Construct one `ImageGeometry2D` and an `ImagePair2D` with independent frame masks and positive timing.
2. Define the interrogation passes in a `PIVPlan`.
3. Prepare fixed window routes, lag domains, FFT shapes, memory batches, precision, and retention.
4. Run the prepared plan to obtain raw displacement, peak evidence, support, and status.
5. Apply validation without changing raw measurements.
6. Optionally create a separate replacement result.
7. Apply an affine or homographic planar calibration to obtain physical coordinates and velocity.

Image displacement uses `(delta_row_down, delta_column_right)`. A synthetic rightward translation therefore has positive column displacement. Physical output uses the calibration's explicitly named right-handed axes.

## Correlation topology

Linear and circular FFT correlation are different scientific choices.

- Linear correlation exposes nonwrapped signed lags and is the correctness-first choice for ordinary images.
- Circular correlation assumes periodic image-window topology and can wrap a displacement across the correlation boundary.
- Extended search uses a larger second-frame patch and declared lag bounds rather than wraparound.

Masks participate in support, means, energy, and overlap. A masked pixel is not a zero-intensity observation. Constant or insufficiently supported windows return failure evidence rather than a plausible peak.

Window batches are fixed at preparation. Runtime processes one batch, extracts fixed top-k candidates, and discards correlation planes unless retention requests them. This bounds memory independently of the number of interrogation windows.

## Peaks and subpixel displacement

The result retains primary and alternate candidates, integer lag, refined lag, score, ambiguity, curvature/width, and fit status. Gaussian and parabolic three-point fits operate only when their local neighbourhood is finite and admissible. A failed subpixel fit retains the integer candidate and an explicit fallback status.

A primary peak at the declared search boundary is not silently accepted as an interior optimum. Ties use a deterministic score/index policy without perturbing the scientific correlation values.

## Multipass deformation

Later passes interpolate the preceding predictor onto their interrogation and sampling coordinates. Symmetric deformation samples the two frames at opposite half-predictor offsets; second-frame-only deformation is a distinct declared option. Interpolation support and boundary loss contribute to the following pass's validity.

The measured residual displacement is composed with the predictor under one tested frame-0-to-frame-1 sign convention. Pass history is retained only under the selected retention policy.

## Validation and replacement

Validation criteria return separate Boolean evidence:

- finite and supported correlation;
- minimum supported-window fraction;
- search-boundary contact;
- peak ambiguity or weak signal;
- physical/component bounds;
- universal normalized-median residual.

The final usable mask combines enabled criteria, but their individual reasons remain available.

Replacement consumes a validated result and creates a new displacement plus replacement mask. Raw values, raw validity, and validation evidence remain unchanged. Filled values are excluded from measurement-accuracy metrics.

## Physical calibration

An affine calibration applies one constant pixel-to-physical Jacobian. A homographic calibration evaluates its local Jacobian at every interrogation centre. Physical velocity is that mapped displacement divided by explicit positive frame timing. Singular maps, invalid homography denominators, unsupported positions, and inconsistent frames/units fail closed.

## Ensemble and residual disparity

Ensemble PIV accumulates correlation evidence over a declared image-pair group and selects a peak once. It does not average already selected velocity vectors.

Residual disparity fully deforms the images using the final field and measures remaining local particle-pattern mismatch. It is a resolution/error diagnostic. It becomes statistical uncertainty only after an independently qualified calibration maps disparity to coverage.

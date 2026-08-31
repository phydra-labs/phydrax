# Learned dense particle-image displacement

The learned backend estimates dense image displacement. It shares `DenseDisplacementField2D` with imaging consumers but does not inherit classical PIV interrogation-grid, particle-track, or physical-velocity semantics.

## Geometry primitives

Backward warping and flow resizing are part of the scientific contract:

- Images are channel-last.
- Displacement components are `(delta_row_down, delta_column_right)`.
- Warping frame 1 toward frame 0 uses the declared frame-0-to-frame-1 displacement sign.
- Out-of-domain and masked samples return support validity.
- Nonperiodic padding is the default.
- Resizing scales row displacement by the height ratio and column displacement by the width ratio.

A local cost volume declares displacement radius, stride, offset order, feature normalization, and memory budget. Offset order is deterministic `(delta_row, delta_column)`.

## Native model

The native model uses a shared mask-aware convolutional feature pyramid, bounded local cost volumes, warped second-frame features, and coarse-to-fine residual refinement. It is independently implemented in Equinox/JAX and carries no imported kernels, graph definitions, weights, or architecture-specific file formats.

The output retains image geometry, valid support, model/plan identity, and inference status. Model scores or residuals are not reported as uncertainty.

## Training objectives

Training can combine independently normalized terms:

- supervised endpoint/vector error on valid truth;
- forward and reverse photometric residual;
- forward/backward consistency;
- optional declared smoothness.

Every term has its own valid count. A batch with no valid support fails rather than contributing zero loss. Incompressibility is not a default regularizer because image displacement need not satisfy a two-dimensional incompressible flow model.

## Scenario leakage and qualification

Splits operate on scenario family IDs. All temporal frames and augmentations derived from one latent flow/particle realization remain in one split. Qualification covers no motion, translation, affine deformation, spatial-frequency response, boundaries, dropout/occlusion, illumination, density, diameter, and noise.

A trained artifact records model structure, leaf weights, normalization, coordinate convention, corpus/split identities, precision evidence, selection state, and qualification result identities. The learned backend remains nondefault unless held-out qualification demonstrates value relative to classical PIV.

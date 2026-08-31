# Circuit calibration

Circuit calibration reuses Phydrax's existing optimization, posterior, likelihood, and
model-discrepancy contracts. The circuit package supplies only scattering-aware data
alignment and residual construction; it does not own another parameter, optimizer,
sampler, prior, covariance, or result hierarchy.

A calibration parameterization is a pure function from a real PyTree of controls to a
static-structure scattering network or grounded MNA circuit. This explicit function is
also how one parameter is shared by several components or datasets. Bounds and
constraints belong to `phydrax.optim`; transforms, priors, and posterior inference
belong to `phydrax.uq`.

Before a residual is formed, model and data must explicitly agree on frequency, ordered
ports and modes, power-wave convention, reference impedances, basis normalization, and
reference planes. Parsed file content and discrete selections are nondifferentiable.

The default complex residual stacks real and imaginary coordinates in a fixed order and
then applies an explicit real whitening factor. Magnitude/phase residuals are opt-in
because wrapped phase changes the objective geometry. Invalid or singular circuit solves
remain invalid and are handled by the selected native optimization or posterior failure
policy; they are not converted to an arbitrary finite penalty.

Model-form discrepancy can be introduced through the existing scalar, multi-output, or
functional Gaussian-process discrepancy machinery. Repeated-data identifiability gates
must pass before discrepancy-corrected parameter uniqueness or uncertainty claims are
published.

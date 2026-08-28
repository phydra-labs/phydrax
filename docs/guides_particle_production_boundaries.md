# Production boundary and stabilization tools

Production wall preprocessing separates feature classification, constrained
tangential relaxation, layer generation, and moment certification.
`WallMomentCertification` reports zeroth/first moment error, volume variation,
normal defect, and success. Sharp-feature classification is explicit rather than
an averaged-normal side effect.

`FreeSurfaceReconstructionPlan` promotes detector output to a local reconstructed
surface point, normal, signed distance, kernel support fraction, fit residual,
confidence, and success. `truncated_kernel_moments` derives local correction
moments. `ContactAnglePlan` applies a declared wall/interface angle.

Production stabilization includes stateful shock-viscosity coefficients, Balsara
shear limiting, audited pressure/energy jumps from density renormalization, and
free-surface-tangential particle shifting. Every accepted-step state is
checkpointable and every correction is observable.

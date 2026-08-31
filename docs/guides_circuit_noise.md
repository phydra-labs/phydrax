# Circuit noise

Circuit noise is represented by a spectral factor `L`, with covariance `Q = L L*`. Factor form guarantees positive semidefiniteness by construction and permits correlated low-rank sources without materializing a dense source covariance.

`propagate_descriptor_noise` applies the descriptor transfer response to the source factor and returns the output factor and covariance. Diagnostics report Hermitian defect, minimum covariance eigenvalue, positive-semidefinite eligibility, linear solve status, and finiteness.

`thermal_resistor_noise_factor` implements the Johnson–Nyquist current-noise factor using explicit resistance and absolute temperature. Noise reference or basis changes must be performed with the same explicit coordinate maps used for deterministic waves.

Frequency-domain noise does not imply periodic phase noise or transient stochastic behavior. Those analyses require separate harmonic-transfer or stochastic-DAE contracts.

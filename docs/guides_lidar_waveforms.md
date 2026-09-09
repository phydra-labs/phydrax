# LiDAR waveforms, atmosphere, and multipath

`WaveformSupport` is the product of acquisition rays, delay bins, bin widths, and receiver channels. `PulseResponse` retains a normalized emitted or receiver impulse response.

`HardSurfaceLidarWaveformPlan` converts exact dynamic surface ranges into round-trip delays, incidence/backscatter amplitudes, pulse shifts, receiver convolution, and an energy ledger. `LidarReturnExtractionPlan` performs explicit matched-filter peak selection and reports fixed return-capacity overflow; peak topology is nondifferentiable.

`AtmosphericLidarPlan` integrates nonnegative extinction and backscatter with two-way transmittance, overlap, spreading, and pulse delay. A single elastic channel does not identify unrestricted extinction and backscatter independently.

`SpecularLidarMultipathPlan` tallies bounded externally traced path lengths and powers. `TimeResolvedMultipleScatteringPlan` supplies reproducible stochastic time-bin tallies with explicit packet/event capacities; it does not claim pathwise gradients through interaction choices.

# Coherent wave Schlieren

`RefractivePhaseScreenPlan` converts optical-path difference to `exp(i k ΔL)` and reports phase magnitude, neighbor phase increments, finite state, and sampling adequacy.

`WaveSchlierenPlan` composes the phase screen, angular-spectrum propagation, an explicit complex Fourier-plane filter or knife edge, detector propagation, and square-law intensity. It retains input/detector/rejected power and boundary-leakage evidence.

`MultisliceRefractivePlan` alternates symmetric half-step diffraction with per-slice refractive phase. Slice count and padding are static. Evidence includes maximum slice phase, power drift, leakage, and finite status.

`ScalarHelmholtzContinuationPlan` provides a bounded periodic Lippmann–Schwinger continuation route for cases beyond multislice assumptions. It reports the actual equation residual and refuses nonperiodic support. Laboratory-scale vector Maxwell remains a separate microscale method.

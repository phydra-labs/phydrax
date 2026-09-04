# Skeletal-muscle fiber bundles

`SkeletalFiberBundlePlan` advances a fixed homogeneous bundle of one-dimensional
fibers using the complete Shorten 2007 fast-twitch reaction model and a no-flux
finite-difference monodomain diffusion term. Each fiber has a fixed node count;
fibers with different node counts belong to separate execution worksets rather than
one global padded state.

The state shape is `(fiber, node, 56)`. Only the surface membrane potential `vS`
diffuses. The remaining cellular states evolve locally through the pinned Shorten
reaction equations. Fiber lengths use mm, time uses ms, and diffusivity uses
`mm2/ms`.

`PrescribedFiberStimulusSchedule` owns a fixed pulse capacity and an explicit
`(pulse, fiber, node)` support mask. Pulses are left-closed and right-open. Candidate
windows may end on a pulse boundary but may not cross an unrepresented interior
boundary; such a candidate retains its evidence and rolls the whole bundle back.
Motor-unit firing rates are not converted to current.

The prepared runtime solves the coupled reaction-diffusion system with Phydrax's
Diffrax backend and Kvaerno5. It exposes membrane/t-tubule potential, cytosolic
calcium, the Shorten force-bearing `A_2` crossbridge concentration, and the applied
stimulus field. `A_2` is a biochemical concentration, not physical force or stress.

The spatial discretization uses mirrored no-flux endpoint values. Constant membrane
potential therefore has exactly zero diffusion contribution. Qualification checks
support selectivity, neighbor propagation, event alignment, finiteness, and complete
rollback. It does not claim endplate physiology, EMG, three-dimensional mechanics,
or MPI-scale performance.

## Sparse motor-unit territories

`MotorUnitTerritoryPlan` stores one motor-unit index and one endplate node for each
fiber. The representation scales as motor units plus fibers plus event slots; it does
not allocate a dense motor-unit-by-fiber-by-node tensor. Preparation requires every
fiber to be assigned and every declared motor unit to own at least one fiber.

`bind_events(...)` routes a fixed `(motor_unit, event_slot)` event block to a
`MotorUnitEndplateStimulus`. Event times may come from a qualified stochastic motor
unit model, but pulse amplitude, duration, and `stimulus_source_id` are explicit
inputs. This adapter performs no firing-rate-to-current conversion and makes no
universal neuromuscular-junction claim. Inactive event slots remain masked and do not
allocate or contribute current.

Run `examples/skeletal_muscle_motor_territories.py` to inspect the sparse routing
contract.

Run:

```text
python examples/skeletal_muscle_fibers.py
python tools/skeletal_muscle_fiber_qualification.py
python benchmarks/skeletal_muscle_fibers.py
```

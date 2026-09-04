# Skeletal-muscle proprioception

`MileusnicSpindle2006Plan` implements the three-branch feline muscle-spindle
model of Mileusnic, Brown, Lan, and Loeb, DOI
[`10.1152/jn.00868.2005`](https://doi.org/10.1152/jn.00868.2005).
It is a receptor/transducer model, not a reflex controller.

Inputs are fascicle length normalized by optimal fascicle length, its first and
second time derivatives, and dynamic/static gamma frequencies in pulses per
second. Whole musculotendon length is not a valid substitute. The state carries
bag1/bag2 fusimotor filters and tension/tension-rate for bag1, bag2, and chain
intrafusal branches. Outputs are Ia and group-II rates in impulses per second.
The source force unit is arbitrary and scale-invariant.

The runtime uses the source equations and Table-1 feline parameters with a
fixed-step RK4 transaction. The source did not publish solver settings or initial
numerical states, so `initialize()` computes a declared zero-velocity mechanical
equilibrium. The default maximum step is 0.1 ms. A failed state, input, parameter,
or step proposal rolls back the complete receptor state.

The model was fitted to cat soleus records and validated on cat medial
gastrocnemius data. It omits stiction, initial burst, and movement-history effects,
and it must not be represented as a human-generic spindle.

The companion Mileusnic–Loeb 2006 Golgi tendon-organ model is intentionally not
implemented. Its exact input is per-fiber tension plus collagen topology for one
receptor, not aggregate tendon force, and the printed nonlinear collagen exponent
is not unambiguous enough to reproduce the unpublished MATLAB/Simulink model.
No scalar tendon-force-to-Ib approximation or generic reflex law is substituted.

Run:

```text
python examples/skeletal_muscle_proprioception.py
python tools/skeletal_proprioception_qualification.py
python benchmarks/skeletal_muscle_proprioception.py
```

Differentiation is local to the fixed continuous spindle equations and integration
path. A future event-generating afferent or closed-loop circuit would require a
separate source identity, fixed event capacity, latency contract, and validation.

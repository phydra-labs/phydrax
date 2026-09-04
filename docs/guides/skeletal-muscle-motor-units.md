# Fuglevand–Winter–Patla 1993 motor units

`FuglevandWinterPatla1993Plan` implements the isometric recruitment,
rate-coding, stochastic discharge, nonlinear force-gain, and critically damped
twitch model in Fuglevand, Winter & Patla, “Models of Recruitment and Rate
Coding Organization in Motor-Unit Pools,” *Journal of Neurophysiology* 70
(1993), 2470–2488, [doi:10.1152/jn.1993.70.6.2470](https://doi.org/10.1152/jn.1993.70.6.2470).
The citation supports only that source model. It does not support fatigue,
dynamic contraction, synchronized discharge, or differentiability of event
topology.

For source rank $i=1,\ldots,n$, preparation evaluates

$$
RTE_i=\exp\!\left(\frac{\log RR}{n}i\right),\qquad
P_i=\exp\!\left(\frac{\log RP}{n}i\right),\qquad
T_i=T_L P_i^{-\log(R_T)/\log(R_P)}.
$$

A recruited unit has firing rate

$$
f_i=\min\{g_e(E-RTE_i)+MFR,\;PFR_i\},
$$

and its renewal interval is $\mu_i(1+cv\,z)$, where $z$ is a standard
normal variate truncated to the paper’s $[-3.9,3.9]$ support. The default
$cv=0.2$ reproduces the source setting. A discharge with preceding interval
$ISI$ uses normalized rate $x=T_i/ISI$ and the source gain

$$
g(x)=\begin{cases}
1,&x\le0.4,\\
\dfrac{[1-\exp(-2x^3)]/x}{[1-\exp(-2\,0.4^3)]/0.4},&x>0.4,
\end{cases}
$$

before adding the twitch

$$
h_i(t)=g(x)P_i\frac{t}{T_i}\exp(1-t/T_i),\qquad t\ge0.
$$

The exact critically damped state update stores two values per motor unit;
it does not retain an unbounded event history.

## Units and identities

- excitation and recruitment threshold: source arbitrary excitation unit;
- firing rate: Hz;
- time, contraction time, and event time: ms;
- twitch and total force: source arbitrary force unit;
- event axes: `(motor_unit, event_slot)`.

The result is the terminal force for this route. It is not D1 force, is not
combined with D1, and is not in newtons. A separately fitted physical
observation model may calibrate an explicitly normalized relative-force trace.

## Transaction and random semantics

Create a plan with a nonempty semantic `random_stream_id`, prepare it, and
initialize state. Each call receives a `FuglevandWinterPatla1993RandomInput`
containing a JAX key, the state’s exact `random_step`, and the same stream ID.
The same source state and random input replay bitwise. Successful commit
advances the random counter once. Any invalid input or event-capacity overflow
rolls back time, twitch state, renewal schedule, and random counter together.

The event buffer has fixed `(unit_count, event_capacity_per_unit)` shape and an
explicit mask. `EVENT_CAPACITY_OVERFLOW` is fail-closed; increase capacity or
shorten the step rather than silently dropping discharges.

Recruitment, event count, event time, and force-gain branch selection are
stop-gradient. Only smooth twitch evolution conditional on a fixed realized
event schedule has a local derivative. No global AD claim crosses an event.

Run the example and independent qualification surface with:

```console
python examples/skeletal_motor_units_fuglevand_1993.py
python tools/skeletal_motor_units_qualification.py
```

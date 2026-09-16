# Polynomial moment and SOS relaxations

`phydrax.optim.polynomial` compiles bounded polynomial optimization problems into the
existing Phydrax conic substrate. It owns no separate optimizer.

## Problem

A problem consists of a polynomial objective, polynomial equalities, and polynomial
nonnegative inequalities over a declared variable ordering. Compact-domain assumptions
must be explicit; a finite relaxation is not evidence that an unconstrained problem is
well posed.

## Planning

A `MomentRelaxationPlan` fixes:

- relaxation order;
- monomial ordering;
- the moment basis;
- each constraint's localizing basis;
- PSD block dimensions;
- equality rows;
- numeric dtype and scaling;
- storage and workspace limits;
- rank and flatness policies.

Planning computes dimensions and projected bytes before allocating the moment or
localizing matrices. Oversized relaxations fail with resource evidence.

## Conic lowering

The compiler lowers moment and localizing positivity to
`PositiveSemidefiniteCone` blocks and polynomial equalities to `ZeroCone` rows in a
canonical `ConicProgram`. Existing native, Clarabel, or CVXPY execution retains its
ordinary provider and original-coordinate audit semantics.

## Result claims

A relaxation result keeps separate:

- conic primal and dual status;
- provider lower bound;
- independently evaluated candidate upper bound;
- objective gap;
- moment and localizing PSD residuals;
- equality and original inequality residuals;
- moment spectra and adjacent-order rank evidence;
- flat-extension status;
- atom-extraction status.

A lower bound is not an optimizer. A flatness candidate is not an extracted measure.
A global-optimum claim requires a valid lower bound, a replayed feasible candidate, and
a gap inside the declared tolerance.

Atom extraction may use quotient-algebra multiplication operators only when the
flat-extension and conditioning policies admit it. Otherwise the result remains a
bound with explicit extraction failure.

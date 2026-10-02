# Delay operators

Delay shifts or clips a field's time coordinate. Variable delays and clipping
can destroy periodicity even when the source field is certified periodic.
The result therefore retains ordinary user metadata but drops source-bound
model construction certificates and port evidence. Bind a separately certified
construction if hard periodicity is required after the transformation.

::: phydrax.operators.delay_operator

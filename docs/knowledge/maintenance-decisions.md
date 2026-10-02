# Maintenance decisions (prototype policy)

The decision engine maps an advisory and a schedule to one of three actions. The thresholds are
**configurable demonstration values, not a certified maintenance procedure.**

| Action | Rule (defaults) |
|---|---|
| `GROUND_NOW` | P(failure during the next flight leg) ≥ 5% |
| `REPLACE_AT_NEXT_CHECK` | otherwise, P(failure before the next check) ≥ 10%, or RUL p10 ≤ 60 FH |
| `DEFER_AND_MONITOR` | otherwise; re-assess every flight |

Expected cost if deferred = P(failure before check) × unscheduled cost + (1 − P) × planned cost.
Cost inputs are illustrative assumptions (see `CostAssumptions` in the decision module).
In the demo, an overnight check occurs every 12 flight hours at a fictional station.

A work order is a **draft** generated from the advisory and decision. Real work orders need a
qualified technician, approved data and the operator's maintenance system.

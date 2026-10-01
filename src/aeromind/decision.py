"""Maintenance decisions from RUL quantiles: what to do, when, and at what risk.

PROTOTYPE DECISION POLICY. The thresholds and cost figures are configurable assumptions for
demonstration, not certified maintenance procedures or real airline data.

The RUL quantiles (p10, p50, p90) define a piecewise-linear failure-time CDF:
F(p10) = 0.1, F(p50) = 0.5, F(p90) = 0.9, extended linearly to F = 0 and F = 1 and clipped.
From it: the probability of failure before the next flight leg ends and before the next
scheduled maintenance opportunity.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from .alerts import ACTIONS, Advisory

GROUND_NOW = "GROUND_NOW"
REPLACE_AT_NEXT_CHECK = "REPLACE_AT_NEXT_CHECK"
DEFER = "DEFER_AND_MONITOR"


@dataclass
class Schedule:
    """Where the aircraft is in its operating day (hours are flight hours)."""

    hours_to_next_check: float = 10.0  # until the next overnight check where parts/labour are available
    leg_hours: float = 2.0  # length of the next flight leg
    check_station: str = "BLR"  # station of that check (fictional rotation)


@dataclass
class DecisionPolicy:
    """Prototype thresholds (configurable)."""

    ground_if_p_fail_next_leg: float = 0.05
    replace_if_p_fail_before_check: float = 0.10
    plan_horizon_hours: float = 60.0  # within this p10 RUL, plan a replacement even if risk is low


@dataclass
class CostAssumptions:
    """Illustrative cost assumptions in USD (labelled assumptions, not airline data)."""

    aog_cost_per_hour: float = 150_000.0  # figure used in the project deck; needs a citation
    unscheduled_aog_hours: float = 12.0  # downtime when the part fails in service
    unscheduled_extra_labour: float = 20_000.0
    planned_replacement_labour: float = 6_000.0
    part_cost: dict = field(default_factory=lambda: {
        "bearing_wear": 40_000.0, "oil_contamination": 8_000.0, "overheating": 25_000.0,
        "electrical_fault": 30_000.0, "pressure_leak": 5_000.0, "unclassified_anomaly": 15_000.0,
    })

    def unscheduled_cost(self, fault: str) -> float:
        return (self.aog_cost_per_hour * self.unscheduled_aog_hours + self.unscheduled_extra_labour
                + self.part_cost.get(fault, 15_000.0))

    def planned_cost(self, fault: str) -> float:
        return self.planned_replacement_labour + self.part_cost.get(fault, 15_000.0)


def failure_cdf(t: float, p10: float, p50: float, p90: float) -> float:
    """P(failure by flight hour ``t`` from now) under the piecewise-linear quantile CDF."""
    xs = [p10 - (p50 - p10) * 0.25, p10, p50, p90, p90 + (p90 - p50) * 0.25]
    ys = [0.0, 0.1, 0.5, 0.9, 1.0]
    xs = np.maximum.accumulate(np.asarray(xs, dtype=float) + np.arange(5) * 1e-9)
    return float(np.clip(np.interp(t, xs, ys), 0.0, 1.0))


@dataclass(frozen=True)
class Decision:
    action: str
    p_fail_next_leg: float
    p_fail_before_check: float
    hours_to_next_check: float
    window: str
    rationale: str
    expected_cost_if_deferred: float
    cost_if_planned: float
    policy_note: str = "Prototype decision policy, not a certified maintenance procedure"

    def to_dict(self) -> dict:
        return asdict(self)


def decide(advisory: Advisory, schedule: Schedule | None = None, policy: DecisionPolicy | None = None,
           costs: CostAssumptions | None = None) -> Decision:
    s, pol, c = schedule or Schedule(), policy or DecisionPolicy(), costs or CostAssumptions()
    q = (advisory.rul_hours_p10, advisory.rul_hours_p50, advisory.rul_hours_p90)
    p_leg = failure_cdf(s.leg_hours, *q)
    p_check = failure_cdf(s.hours_to_next_check, *q)
    unplanned, planned = c.unscheduled_cost(advisory.fault_type), c.planned_cost(advisory.fault_type)
    if p_leg >= pol.ground_if_p_fail_next_leg:
        action, window = GROUND_NOW, "Before next departure"
        why = f"P(failure during next {s.leg_hours:g} h leg) = {p_leg:.0%} ≥ {pol.ground_if_p_fail_next_leg:.0%}"
    elif p_check >= pol.replace_if_p_fail_before_check or q[0] <= pol.plan_horizon_hours:
        action = REPLACE_AT_NEXT_CHECK
        window = f"Next overnight check at {s.check_station}, in {s.hours_to_next_check:g} FH"
        why = (f"P(failure before next check) = {p_check:.0%}; conservative RUL p10 = {q[0]:.0f} FH"
               f" (plan within {pol.plan_horizon_hours:g} FH)")
    else:
        action, window = DEFER, f"Re-assess each flight; plan before {max(q[0] - pol.plan_horizon_hours, 0):.0f} FH"
        why = f"P(failure before next check) = {p_check:.1%}; RUL p10 = {q[0]:.0f} FH"
    return Decision(action, round(p_leg, 4), round(p_check, 4), s.hours_to_next_check, window, why,
                    round(p_check * unplanned + (1 - p_check) * planned), round(planned))


_wo_counter = itertools.count(1)


@dataclass(frozen=True)
class WorkOrder:
    number: str
    tail: str
    component: str
    fault: str
    confidence: float
    rul_hours: tuple[float, float, float]
    risk_before_check: float
    action: str
    window: str
    task: str
    part: str
    advisory_text: str
    acars: str

    def to_dict(self) -> dict:
        return asdict(self)


def work_order(tail: str, advisory: Advisory, decision: Decision) -> WorkOrder:
    from .acars import encode

    task, part = ACTIONS[advisory.fault_type]
    return WorkOrder(
        number=f"WO-{tail}-{next(_wo_counter):04d}", tail=tail, component=advisory.component,
        fault=advisory.fault_type, confidence=advisory.confidence,
        rul_hours=(advisory.rul_hours_p10, advisory.rul_hours_p50, advisory.rul_hours_p90),
        risk_before_check=decision.p_fail_before_check, action=decision.action, window=decision.window,
        task=task, part=part,
        advisory_text="; ".join(advisory.evidence) or ", ".join(advisory.contributing_signals),
        acars=encode(advisory, tail),
    )

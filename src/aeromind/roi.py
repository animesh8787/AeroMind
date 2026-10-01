"""Fleet maintenance simulation: reactive vs fixed-interval vs AeroMind.

Monte Carlo over a fleet and a horizon in flight hours. Component failures arrive per aircraft
from a Weibull time-to-failure per fault type; each failure is preceded by a degradation period.
AeroMind's behaviour is NOT assumed: the probability of detecting a degradation and the warning
time before failure are resampled from a measured ``evaluate()`` result (per fault mode), and its
false advisories come from the measured false-advisory rate.

Policies:
  reactive        replace on failure (unscheduled removal, AOG downtime).
  fixed_interval  replace at a scheduled check every ``interval_fh``; failures before that are unscheduled.
  aeromind        detected degradations are replaced at the first scheduled check before failure
                  (planned removal); if the warning is shorter than the time to that check the aircraft
                  is grounded early (planned part, shorter downtime); missed ones fail like reactive.

Every number here is ASSUMED unless it comes from the evaluation; ``Assumptions`` lists them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import FAULT_MODES
from .decision import CostAssumptions


@dataclass
class Assumptions:
    fleet_size: int = 30
    horizon_fh: float = 3_000.0  # flight hours per aircraft (about a year of narrow-body flying)
    weibull_shape: float = 2.5
    mean_life_fh: float = 1_500.0  # mean time to failure of the monitored component, any mode (ASSUMED)
    check_interval_fh: float = 12.0  # an overnight check every 12 FH
    fixed_interval_fh: float = 1_000.0  # hard-time replacement interval for the fixed-interval policy
    early_grounding_aog_hours: float = 4.0  # downtime when grounded early with the part pre-positioned
    false_advisory_inspection_cost: float = 1_500.0
    seed: int = 0
    costs: CostAssumptions = field(default_factory=CostAssumptions)

    def describe(self) -> dict:
        d = asdict(self)
        d["note"] = "All values are illustrative assumptions except AeroMind detection and lead times (measured)"
        return d


@dataclass
class PolicyResult:
    unscheduled_removals: int = 0
    scheduled_removals: int = 0
    early_groundings: int = 0
    aog_hours: float = 0.0
    parts_used: int = 0
    false_inspections: int = 0
    cost: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aog_hours"] = round(d["aog_hours"], 1)
        d["cost"] = round(d["cost"])
        return d


def _detection_model(evaluation: dict):
    """Per-mode (detection probability, lead-time samples in hours) from an evaluate() result."""
    model = {}
    for mode, r in evaluation["per_mode"].items():
        leads = r.get("lead_times_hours_first_alert") or []
        model[mode] = (r["detected"] / max(r["runs"], 1), np.asarray(leads, dtype=float))
    return model


def simulate(evaluation: dict, a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    # Separate streams: the failure history must not depend on detection settings (common random numbers).
    rng_fail, rng_det, rng_fix, rng_fa = (np.random.default_rng([a.seed, k]) for k in range(4))
    det = _detection_model(evaluation)
    c = a.costs
    scale = a.mean_life_fh / _weibull_mean_factor(a.weibull_shape)  # Weibull scale for that mean
    res = {p: PolicyResult() for p in ("reactive", "fixed_interval", "aeromind")}
    fa_per_fh = evaluation["false_advisories_per_1000_windows"] / 1000.0 / 0.5  # windows are 0.5 FH apart

    def failure_cost(mode):
        return c.unscheduled_cost(mode)

    for _ in range(a.fleet_size):
        # Same failure history for all three policies (common random numbers).
        events, t = [], 0.0
        while True:
            t += scale * rng_fail.weibull(a.weibull_shape)
            if t > a.horizon_fh:
                break
            mode = FAULT_MODES[rng_fail.integers(len(FAULT_MODES))]
            p_det, leads = det[mode]
            u, k = rng_det.random(), rng_det.random()  # always drawn, whatever the detection model
            detected = u < p_det and len(leads) > 0
            events.append((t, mode, detected, float(leads[int(k * len(leads))]) if detected else 0.0))

        # Reactive: every failure is unscheduled.
        for _, mode, _, _ in events:
            r = res["reactive"]
            r.unscheduled_removals += 1
            r.parts_used += 1
            r.aog_hours += c.unscheduled_aog_hours
            r.cost += failure_cost(mode)

        # Fixed interval: hard-time replacement resets the component; failures in between are unscheduled.
        r = res["fixed_interval"]
        n_sched = int(a.horizon_fh // a.fixed_interval_fh)
        r.scheduled_removals += n_sched
        r.parts_used += n_sched
        r.cost += n_sched * (c.planned_replacement_labour + np.mean(list(c.part_cost.values())))
        # Renewal: a failure only happens if its age since the last replacement reaches the Weibull draw.
        tt = 0.0
        while True:
            life = scale * rng_fix.weibull(a.weibull_shape)
            next_sched = (np.floor(tt / a.fixed_interval_fh) + 1) * a.fixed_interval_fh
            if tt + life < min(next_sched, a.horizon_fh):
                mode = FAULT_MODES[rng_fix.integers(len(FAULT_MODES))]
                tt += life
                r.unscheduled_removals += 1
                r.parts_used += 1
                r.aog_hours += c.unscheduled_aog_hours
                r.cost += failure_cost(mode)
            else:
                tt = next_sched
            if tt >= a.horizon_fh:
                break

        # AeroMind: act on detections, using the measured warning time.
        r = res["aeromind"]
        for t_fail, mode, detected, lead in events:
            if not detected:
                r.unscheduled_removals += 1
                r.aog_hours += c.unscheduled_aog_hours
                r.cost += failure_cost(mode)
            else:
                t_alert = t_fail - lead
                next_check = (np.floor(t_alert / a.check_interval_fh) + 1) * a.check_interval_fh
                if next_check < t_fail:
                    r.scheduled_removals += 1
                    r.cost += c.planned_cost(mode)
                else:
                    r.early_groundings += 1
                    r.aog_hours += a.early_grounding_aog_hours
                    r.cost += c.planned_cost(mode) + a.early_grounding_aog_hours * c.aog_cost_per_hour
            r.parts_used += 1
        n_fa = rng_fa.poisson(fa_per_fh * a.horizon_fh)
        r.false_inspections += n_fa
        r.cost += n_fa * a.false_advisory_inspection_cost

    out = {k: v.to_dict() for k, v in res.items()}
    base = out["reactive"]["cost"]
    for k in out:
        out[k]["cost_vs_reactive_pct"] = round(100 * (out[k]["cost"] - base) / base, 1) if base else None
    return {"policies": out, "assumptions": a.describe(),
            "measured_inputs": {m: {"detection_rate": round(p, 3), "lead_time_samples_h": leads.tolist()}
                                for m, (p, leads) in det.items()}
            | {"false_advisories_per_fh": fa_per_fh}}


def _weibull_mean_factor(k: float) -> float:
    from math import gamma

    return gamma(1 + 1 / k)


def sensitivity(evaluation: dict, base: Assumptions | None = None) -> list[dict]:
    """The same simulation under harsher conditions, so no single assumption carries the result."""
    import copy

    base = base or Assumptions()

    def degraded(det_scale: float, lead_scale: float, fa_per_1000: float | None = None) -> dict:
        ev = copy.deepcopy(evaluation)
        for r in ev["per_mode"].values():
            r["detected"] = int(round(r["detected"] * det_scale))
            r["lead_times_hours_first_alert"] = [v * lead_scale for v in r.get("lead_times_hours_first_alert", [])]
        if fa_per_1000 is not None:
            ev["false_advisories_per_1000_windows"] = fa_per_1000
        return ev

    low_aog = copy.deepcopy(base)
    low_aog.costs.aog_cost_per_hour = 10_000.0
    scenarios = [
        ("as measured on the simulator", evaluation, base),
        ("detection 67%, warning x0.1, 5 false advisories/1000 windows", degraded(4 / 6, 0.1, 5.0), base),
        ("as measured, AOG cost $10k/h instead of $150k/h", evaluation, low_aog),
        ("detection 67%, warning x0.1, 5 FA/1000, AOG $10k/h", degraded(4 / 6, 0.1, 5.0), low_aog),
    ]
    out = []
    for name, ev, a in scenarios:
        p = simulate(ev, a)["policies"]
        out.append({"scenario": name, **{f"{k}_cost_vs_reactive_pct": v["cost_vs_reactive_pct"] for k, v in p.items()},
                    "aeromind_aog_hours": p["aeromind"]["aog_hours"], "reactive_aog_hours": p["reactive"]["aog_hours"]})
    return out

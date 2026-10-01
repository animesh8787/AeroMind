from aeromind.config import FAULT_MODES
from aeromind.roi import Assumptions, sensitivity, simulate


def _eval(detected=6, lead=100.0, fa=0.0):
    return {"false_advisories_per_1000_windows": fa,
            "per_mode": {m: {"runs": 6, "detected": detected, "lead_times_hours_first_alert": [lead] * detected}
                         for m in FAULT_MODES}}


def test_policies_conserve_failures_and_respond_to_detection():
    a = Assumptions(fleet_size=10, horizon_fh=2000, seed=3)
    r = simulate(_eval(), a)["policies"]
    react, aero = r["reactive"], r["aeromind"]
    n_fail = react["unscheduled_removals"]
    assert n_fail > 0 and aero["parts_used"] == n_fail  # same failure history for both policies
    assert aero["unscheduled_removals"] == 0 and aero["aog_hours"] == 0 and aero["cost"] < react["cost"]
    blind = simulate(_eval(detected=0), a)["policies"]["aeromind"]
    assert blind["unscheduled_removals"] == n_fail and blind["cost"] == react["cost"]
    short = simulate(_eval(lead=1.0), a)["policies"]["aeromind"]
    assert short["early_groundings"] > 0  # warning shorter than the time to the next check


def test_sensitivity_lists_harsher_scenarios():
    rows = sensitivity(_eval(), Assumptions(fleet_size=5, horizon_fh=1500))
    assert len(rows) == 4 and rows[0]["reactive_cost_vs_reactive_pct"] == 0.0
    assert rows[1]["aeromind_cost_vs_reactive_pct"] > rows[0]["aeromind_cost_vs_reactive_pct"]

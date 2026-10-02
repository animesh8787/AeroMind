from aeromind.report.evidence_report import to_markdown


def test_markdown_renders_unavailable_datasets():
    r = {"generated": "now", "runtime_s": 1, "note": "n",
         "simulator": {"phase_aware_model_on_flight_phases": {
             "detected": {"bearing_wear": "6/6"}, "median_lead_time_h": {"bearing_wear": 100},
             "first_classified_correct": {"bearing_wear": "6/6"}, "rul_mae_h": {"bearing_wear": 5},
             "rul_p10_p90_coverage": {"bearing_wear": 0.8}, "false_advisories_per_1000_windows": 0,
             "bandwidth_reduction_factor": 600, "latency_ms_mean": 1, "latency_ms_max": 2}},
         "flight_phases": {"healthy_steady_model": {"over_threshold": 0.9, "gate_open": 0.9},
                           "healthy_phase_aware_model": {"over_threshold": 0.02, "gate_open": 0.0}},
         "sensor_health": {"false_sensor_faults_on_clean_runs": 0, "clean_windows": 10, "cases": {}},
         "acars": {"messages": 1, "max_chars": 100, "mean_chars": 100, "limit": 220, "all_fit": True},
         "roi": {"policies": {}, "sensitivity": []},
         "edge_benchmark": {"host": "h", "note": "cpu", "total_kb": 1, "rss_mb_after_load": 1,
                            "latency": {"anomaly": {"median_ms": 0.1}, "classifier": {"median_ms": 0.1},
                                        "rul": {"median_ms": 0.1}, "full_window_process": {"mean_ms": 1}}},
         "ims": {"unavailable": "no IMS"}, "cmapss_hgb_conformal": {"unavailable": "no C-MAPSS"}}
    md = to_markdown(r)
    assert "no IMS" in md and "no C-MAPSS" in md and "| bearing_wear | 6/6 |" in md

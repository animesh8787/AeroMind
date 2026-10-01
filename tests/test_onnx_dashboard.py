import json

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from aeromind.config import FAULT_MODES, HEALTHY
from aeromind.dashboard import Scenario, build_dashboard
from aeromind.features import trend_matrix
from aeromind.onnx_export import OnnxBundle, check_parity, export_onnx
from aeromind.pipeline import EdgePipeline
from aeromind.simulator import simulate_run
from aeromind.train import TrainConfig, collect_run, train


@pytest.fixture(scope="module")
def bundle():
    return train(TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=4, life_range=(200, 300), seed=11))


@pytest.fixture(scope="module")
def onnx_dir(bundle, tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    export_onnx(bundle, d)
    return d


def test_manifest_lists_files_and_classes(bundle, onnx_dir):
    m = json.loads((onnx_dir / "manifest.json").read_text())
    assert set(m["files"]) == {"anomaly", "classifier", "rul"}
    assert m["classes"] == [str(c) for c in bundle.classifier.model.classes_]
    assert len(m["feature_names"]) == 19


def test_onnx_matches_sklearn(bundle, onnx_dir):
    runs = [collect_run(m, 200, 8_100_000 + i) for i, m in enumerate((HEALTHY, *FAULT_MODES))]
    X = np.vstack([r.X for r in runs])
    T = np.vstack([trend_matrix(r.X, bundle.anomaly.score(r.X)) for r in runs])
    p = check_parity(bundle, OnnxBundle(onnx_dir), X, T)
    # float32 graphs: tiny numeric differences, identical decisions.
    assert p["class_probability_max_abs_diff"] < 1e-4
    assert p["predicted_class_agree"] == 1.0
    assert p["threshold_decisions_agree"] >= 0.999
    d_score = np.abs(bundle.anomaly.score(X) - OnnxBundle(onnx_dir).anomaly.score(X))
    assert np.median(d_score) < 1e-4
    d_rul = np.abs(bundle.rul.predict_batch(T) - OnnxBundle(onnx_dir).rul.predict_batch(T))
    assert np.median(d_rul) < 1e-2
    assert (d_rul > 1.0).mean() < 0.01  # rare float32 split flips only


def test_pipeline_advisories_match_across_backends(bundle, onnx_dir):
    def advisories(models):
        pipe = EdgePipeline(models)
        return [(a.window, a.fault_type, a.priority)
                for w, _ in simulate_run("bearing_wear", 260, 424242) if (a := pipe.process(w)) is not None]

    ref = advisories(bundle)
    assert ref and advisories(OnnxBundle(onnx_dir)) == ref


def test_tampered_model_is_rejected(onnx_dir, tmp_path):
    for f in onnx_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    (tmp_path / "rul.onnx").write_bytes((tmp_path / "rul.onnx").read_bytes() + b"\0")
    with pytest.raises(ValueError, match="checksum"):
        OnnxBundle(tmp_path)


@pytest.fixture(scope="module")
def trt_dir(bundle, tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("hummingbird.ml")
    d = tmp_path_factory.mktemp("onnx_trt")
    export_onnx(bundle, d, trees="hummingbird", strategy="gemm")
    return d


def test_hummingbird_parser_keeps_zero_leaves(bundle):
    pytest.importorskip("hummingbird.ml")
    from aeromind.onnx_export import _hist_gbdt_params

    for pred in [p for it in bundle.classifier.model._predictors for p in it]:
        params, n = _hist_gbdt_params(pred, {}), pred.nodes
        leaf = n["is_leaf"].astype(bool)
        assert [v[0] for v in params.values] == n["value"].tolist()
        assert all(params.lefts[i] == -1 for i in np.flatnonzero(leaf))
        assert all(params.lefts[i] == int(n["left"][i]) for i in np.flatnonzero(~leaf))


def test_hummingbird_graphs_use_only_tensorrt_operators(trt_dir):
    import onnx

    from aeromind.onnx_export import FILES, TENSORRT_OPERATORS, operators

    for name in FILES.values():
        m = onnx.load(trt_dir / name)
        assert {o.domain for o in m.opset_import} <= {"", "ai.onnx"}
        assert set(operators(m)) <= TENSORRT_OPERATORS
        squeezes = [n for n in m.graph.node if n.op_type == "Squeeze"]
        assert all(len(n.input) == 2 for n in squeezes)  # static axes


def test_hummingbird_matches_sklearn_including_single_rows(bundle, trt_dir):
    runs = [collect_run(m, 200, 8_200_000 + i) for i, m in enumerate((HEALTHY, *FAULT_MODES))]
    X = np.vstack([r.X for r in runs])
    T = np.vstack([trend_matrix(r.X, bundle.anomaly.score(r.X)) for r in runs])
    ob = OnnxBundle(trt_dir)
    p = check_parity(bundle, ob, X, T)
    assert p["class_probability_max_abs_diff"] < 1e-4
    assert p["predicted_class_agree"] == 1.0
    assert p["threshold_decisions_agree"] >= 0.999
    d_rul = np.abs(bundle.rul.predict_batch(T) - ob.rul.predict_batch(T))
    assert np.median(d_rul) < 1e-2 and (d_rul > 1.0).mean() < 0.01
    # The edge pipeline sends one row at a time.
    rows = np.array([ob.anomaly.score(x)[0] for x in X[:20]])
    np.testing.assert_allclose(rows, ob.anomaly.score(X[:20]), atol=1e-5)
    assert [ob.classifier.predict(x)[0] for x in X[:20]] == list(ob.classifier.predict_batch(X[:20]))


def test_hummingbird_pipeline_advisories_match(bundle, trt_dir):
    def advisories(models):
        pipe = EdgePipeline(models)
        return [(a.window, a.fault_type, a.priority)
                for w, _ in simulate_run("oil_contamination", 260, 434343) if (a := pipe.process(w)) is not None]

    ref = advisories(bundle)
    assert ref and advisories(OnnxBundle(trt_dir)) == ref


def test_dashboard_embeds_every_window(onnx_dir, tmp_path):
    out = build_dashboard(OnnxBundle(onnx_dir), tmp_path / "d.html", "test",
                          [Scenario("pressure_leak", 120, 7_100_000), Scenario(HEALTHY, 60, 7_100_001)])
    html = out.read_text()
    raw = html.split('<script id="aeromind-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(raw)
    leak, healthy = data["scenarios"]
    assert leak["n"] == len(leak["score"]) == len(leak["wave"]) == 120
    assert all(len(v) == 120 for v in leak["sensors"].values())
    assert healthy["truth_rul"] == [None] * 60
    assert "/*__AEROMIND_DATA__*/" not in html

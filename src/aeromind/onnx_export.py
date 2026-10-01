"""Export a trained ``ModelBundle`` to ONNX, and run it with ONNX Runtime.

Three self-contained float32 graphs, written directly with ``onnx.helper`` (no skl2onnx):

  anomaly.onnx     features (N,19)        -> score (N,), z (N,19)
                   scaler, Isolation Forest, autoencoder and the healthy calibration, all in-graph
  classifier.onnx  features (N,19)        -> probabilities (N,C)   classes listed in manifest.json
  rul.onnx         trend features (N,22)  -> rul_hours (N,3)       sorted p10 <= p50 <= p90, >= 0

Feature extraction, trend tracking, the persistence gate and advisory formatting stay in
host code (``pipeline.py``); only the learned models are exported.

Tree ensembles use the ONNX-ML ``TreeEnsembleRegressor`` operator. ONNX Runtime runs it, but
TensorRT does not support ONNX-ML operators, so on a Jetson the tree models would run on
the CPU through ONNX Runtime unless they are first rewritten as tensor operations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES, N_TREND_FEATURES
from .models.rul import QUANTILES
from .train import ModelBundle

FORMAT_VERSION = 1
OPSET = 17
ML_OPSET = 3
FILES = {"anomaly": "anomaly.onnx", "classifier": "classifier.onnx", "rul": "rul.onnx"}


# --------------------------------------------------------------------------- graph building


@dataclass
class _Trees:
    """Flat attribute lists for one ``TreeEnsembleRegressor`` node."""

    treeids: list[int]
    nodeids: list[int]
    featureids: list[int]
    values: list[float]
    modes: list[str]
    true_ids: list[int]
    false_ids: list[int]
    missing_true: list[int]
    t_treeids: list[int]
    t_nodeids: list[int]
    t_ids: list[int]
    t_weights: list[float]

    @classmethod
    def empty(cls) -> "_Trees":
        return cls(*([] for _ in range(12)))

    def add(self, tree_id, feature, threshold, left, right, is_leaf, leaf_value, missing_left, target=0):
        """Append one tree. A sample goes left when ``x[feature] <= threshold``."""
        for i in range(len(is_leaf)):
            self.treeids.append(tree_id)
            self.nodeids.append(i)
            if is_leaf[i]:
                self.featureids.append(0)
                self.values.append(0.0)
                self.modes.append("LEAF")
                self.true_ids.append(0)
                self.false_ids.append(0)
                self.missing_true.append(0)
                self.t_treeids.append(tree_id)
                self.t_nodeids.append(i)
                self.t_ids.append(target)
                self.t_weights.append(float(leaf_value[i]))
            else:
                self.featureids.append(int(feature[i]))
                self.values.append(float(threshold[i]))
                self.modes.append("BRANCH_LEQ")
                self.true_ids.append(int(left[i]))
                self.false_ids.append(int(right[i]))
                self.missing_true.append(int(bool(missing_left[i])))

    def node(self, inp: str, out: str, n_targets: int, base_values, name: str):
        from onnx import helper

        return helper.make_node(
            "TreeEnsembleRegressor", [inp], [out], domain="ai.onnx.ml", name=name,
            n_targets=n_targets, aggregate_function="SUM", post_transform="NONE",
            base_values=[float(v) for v in base_values],
            nodes_treeids=self.treeids, nodes_nodeids=self.nodeids, nodes_featureids=self.featureids,
            nodes_values=self.values, nodes_modes=self.modes,
            nodes_truenodeids=self.true_ids, nodes_falsenodeids=self.false_ids,
            nodes_missing_value_tracks_true=self.missing_true,
            target_treeids=self.t_treeids, target_nodeids=self.t_nodeids,
            target_ids=self.t_ids, target_weights=self.t_weights,
        )


def _hgb_trees(model) -> tuple[_Trees, int, np.ndarray]:
    """Trees of a fitted HistGradientBoosting model: one ONNX target per tree within an iteration."""
    trees = _Trees.empty()
    tid = 0
    for iteration in model._predictors:
        for k, pred in enumerate(iteration):
            n = pred.nodes
            if n["is_categorical"].any():
                raise NotImplementedError("categorical splits are not supported")
            trees.add(tid, n["feature_idx"], n["num_threshold"], n["left"], n["right"],
                      n["is_leaf"], n["value"], n["missing_go_to_left"], target=k)
            tid += 1
    n_targets = len(model._predictors[0])
    return trees, n_targets, np.asarray(model._baseline_prediction, dtype=float).ravel()


def _iforest_trees(forest) -> _Trees:
    """Each leaf holds its isolation path length; summing over trees gives sklearn's ``depths``."""
    trees = _Trees.empty()
    subsample = forest._max_features != forest.n_features_in_
    for t, (est, feats) in enumerate(zip(forest.estimators_, forest.estimators_features_)):
        tr = est.tree_
        is_leaf = tr.children_left == -1
        feature = np.where(is_leaf, 0, tr.feature)
        if subsample:
            feature = np.where(is_leaf, 0, np.asarray(feats)[feature])
        leaf_value = forest._decision_path_lengths[t] + forest._average_path_length_per_tree[t] - 1.0
        trees.add(t, feature, tr.threshold, tr.children_left, tr.children_right, is_leaf, leaf_value,
                  np.zeros(len(is_leaf), dtype=bool))
    return trees


def _model(nodes, inputs, outputs, initializers, name: str):
    from onnx import checker, helper

    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET),
                                                helper.make_opsetid("ai.onnx.ml", ML_OPSET)],
                          producer_name="aeromind")
    m.ir_version = 8
    checker.check_model(m, full_check=True)
    return m


def _const(name: str, value) -> "object":
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


def build_anomaly_graph(det) -> "object":
    from onnx import TensorProto, helper
    from sklearn.ensemble._iforest import _average_path_length

    nodes, inits = [], []

    def op(kind, ins, out, **kw):
        nodes.append(helper.make_node(kind, ins, [out], **kw))
        return out

    # Standardise.
    inits += [_const("sc_mean", det.scaler.mean_), _const("sc_scale", det.scaler.scale_)]
    op("Sub", ["features", "sc_mean"], "centred")
    xs = op("Div", ["centred", "sc_scale"], "z")

    # Isolation Forest: -score_samples = 2 ** (-depths / (n_trees * c(max_samples))).
    f = det.iforest
    denom = len(f.estimators_) * float(_average_path_length([f._max_samples])[0])
    nodes.append(_iforest_trees(f).node(xs, "if_depths", 1, [0.0], "iforest"))
    inits.append(_const("if_k", [-np.log(2.0) / denom]))
    op("Mul", ["if_depths", "if_k"], "if_exp")
    raw_if = op("Exp", ["if_exp"], "if_score")  # (N,1)

    # Autoencoder (tanh MLP, identity output): log of mean squared reconstruction error.
    net = det.autoencoder.net
    if net.activation != "tanh" or net.out_activation_ != "identity":
        raise NotImplementedError("only tanh hidden layers with an identity output are supported")
    h = xs
    for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        inits += [_const(f"W{i}", W), _const(f"b{i}", b)]
        op("MatMul", [h, f"W{i}"], f"mm{i}")
        h = op("Add", [f"mm{i}", f"b{i}"], f"a{i}")
        if i < len(net.coefs_) - 1:
            h = op("Tanh", [h], f"h{i}")
    op("Sub", [h, xs], "ae_diff")
    op("Mul", ["ae_diff", "ae_diff"], "ae_sq")
    op("ReduceMean", ["ae_sq"], "ae_mse", axes=[1], keepdims=1)
    inits.append(_const("eps", [1e-9]))
    op("Add", ["ae_mse", "eps"], "ae_mse_eps")
    raw_ae = op("Log", ["ae_mse_eps"], "ae_score")  # (N,1)

    # Calibration: 1.0 ~ 99th percentile of held-out healthy windows.
    (m1, s1), (m2, s2) = det._norm1, det._norm2
    inits += [_const("n1_med", m1), _const("n1_scale", s1), _const("n2_med", [m2]), _const("n2_scale", [s2])]
    op("Concat", [raw_if, raw_ae], "raw", axis=1)
    op("Sub", ["raw", "n1_med"], "raw_c")
    op("Div", ["raw_c", "n1_scale"], "raw_n")
    op("ReduceMean", ["raw_n"], "combined", axes=[1], keepdims=0)
    op("Sub", ["combined", "n2_med"], "comb_c")
    op("Div", ["comb_c", "n2_scale"], "score")

    f32 = TensorProto.FLOAT
    return _model(
        nodes,
        [helper.make_tensor_value_info("features", f32, ["N", N_FEATURES])],
        [helper.make_tensor_value_info("score", f32, ["N"]),
         helper.make_tensor_value_info("z", f32, ["N", N_FEATURES])],
        inits, "aeromind_anomaly",
    )


def build_classifier_graph(clf) -> "object":
    from onnx import TensorProto, helper

    model = clf.model
    trees, n_targets, base = _hgb_trees(model)
    if n_targets != len(model.classes_) or n_targets < 3:
        raise NotImplementedError("only multiclass (3+ classes) classifiers are supported")
    nodes = [trees.node("features", "logits", n_targets, base, "fault_trees"),
             helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)]
    f32 = TensorProto.FLOAT
    return _model(
        nodes,
        [helper.make_tensor_value_info("features", f32, ["N", N_FEATURES])],
        [helper.make_tensor_value_info("probabilities", f32, ["N", n_targets])],
        [], "aeromind_classifier",
    )


def build_rul_graph(rul) -> "object":
    from onnx import TensorProto, helper

    trees, bases = _Trees.empty(), []
    offset = 0
    for target, q in enumerate(QUANTILES):
        t, n, base = _hgb_trees(rul.models[q])
        assert n == 1
        # Renumber trees so all three quantile models live in one ensemble, one target each.
        t.treeids = [i + offset for i in t.treeids]
        t.t_treeids = [i + offset for i in t.t_treeids]
        t.t_ids = [target] * len(t.t_ids)
        offset = max(t.treeids) + 1
        for field in trees.__dataclass_fields__:
            getattr(trees, field).extend(getattr(t, field))
        bases.append(base[0])
    nodes = [
        trees.node("trend_features", "quantiles", len(QUANTILES), bases, "rul_trees"),
        # Same post-processing as RULEstimator.predict_batch: sort ascending, clip at 0.
        helper.make_node("Constant", [], ["k"], value=helper.make_tensor("kv", TensorProto.INT64, [1], [3])),
        helper.make_node("TopK", ["quantiles", "k"], ["sorted", "sorted_idx"], axis=1, largest=0, sorted=1),
        helper.make_node("Relu", ["sorted"], ["rul_hours"]),
    ]
    f32 = TensorProto.FLOAT
    return _model(
        nodes,
        [helper.make_tensor_value_info("trend_features", f32, ["N", N_TREND_FEATURES])],
        [helper.make_tensor_value_info("rul_hours", f32, ["N", len(QUANTILES)])],
        [], "aeromind_rul",
    )


def export_onnx(bundle: ModelBundle, out_dir: str | Path) -> dict:
    """Write the three graphs plus ``manifest.json`` to ``out_dir``. Returns the manifest."""
    import onnx
    import sklearn

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = {
        "anomaly": build_anomaly_graph(bundle.anomaly),
        "classifier": build_classifier_graph(bundle.classifier),
        "rul": build_rul_graph(bundle.rul),
    }
    files = {}
    for key, g in graphs.items():
        data = g.SerializeToString()
        (out / FILES[key]).write_bytes(data)
        files[key] = {"file": FILES[key], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "format_version": FORMAT_VERSION,
        "opset": {"ai.onnx": OPSET, "ai.onnx.ml": ML_OPSET},
        "dtype": "float32",
        "feature_names": FEATURE_NAMES,
        "n_trend_features": N_TREND_FEATURES,
        "classes": [str(c) for c in bundle.classifier.model.classes_],
        "quantiles": list(QUANTILES),
        "score_note": "anomaly score is calibrated so 1.0 ~ 99th percentile of healthy windows",
        "exported_from": {"scikit-learn": sklearn.__version__, "onnx": onnx.__version__},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- runtime


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # One thread: closer to a dedicated edge core, and steadier per-window latency.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_2d(x), dtype=np.float32)


class OnnxAnomaly:
    def __init__(self, path: Path, feature_names: list[str]):
        self._s, self._names = _session(path), feature_names

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._s.run(["score"], {"features": _f32(X)})[0].astype(np.float64)

    def explain(self, x: np.ndarray, k: int = 3) -> list[str]:
        z = np.abs(self._s.run(["z"], {"features": _f32(x)})[0][0])
        return [self._names[i] for i in np.argsort(z)[::-1][:k]]


class OnnxClassifier:
    def __init__(self, path: Path, classes: list[str]):
        self._s, self.classes_ = _session(path), np.asarray(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._s.run(["probabilities"], {"features": _f32(X)})[0].astype(np.float64)

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        proba = self.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return str(self.classes_[i]), float(proba[i])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class OnnxRUL:
    def __init__(self, path: Path):
        self._s = _session(path)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self._s.run(["rul_hours"], {"trend_features": _f32(X)})[0].astype(np.float64)

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        p10, p50, p90 = self.predict_batch(x)[0]
        return float(p10), float(p50), float(p90)


class OnnxBundle:
    """Drop-in replacement for ``ModelBundle`` inside ``EdgePipeline``, backed by ONNX Runtime."""

    def __init__(self, directory: str | Path):
        d = Path(directory)
        self.manifest = json.loads((d / "manifest.json").read_text())
        if self.manifest["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported ONNX bundle format {self.manifest['format_version']}")
        for info in self.manifest["files"].values():
            if hashlib.sha256((d / info["file"]).read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"{info['file']} does not match the checksum in manifest.json")
        self.anomaly = OnnxAnomaly(d / FILES["anomaly"], self.manifest["feature_names"])
        self.classifier = OnnxClassifier(d / FILES["classifier"], self.manifest["classes"])
        self.rul = OnnxRUL(d / FILES["rul"])

    @staticmethod
    def load(directory: str | Path) -> "OnnxBundle":
        return OnnxBundle(directory)


def check_parity(bundle: ModelBundle, onnx_bundle: OnnxBundle, X: np.ndarray, T: np.ndarray) -> dict:
    """Compare scikit-learn and ONNX outputs on feature rows ``X`` and trend rows ``T``."""
    s_ref, s_onx = bundle.anomaly.score(X), onnx_bundle.anomaly.score(X)
    p_ref = bundle.classifier.model.predict_proba(X)
    p_onx = onnx_bundle.classifier.predict_proba(X)
    r_ref, r_onx = bundle.rul.predict_batch(T), onnx_bundle.rul.predict_batch(T)
    return {
        "rows": len(X),
        "anomaly_score_max_abs_diff": float(np.max(np.abs(s_ref - s_onx))),
        "threshold_decisions_agree": float(np.mean((s_ref > 1.0) == (s_onx > 1.0))),
        "class_probability_max_abs_diff": float(np.max(np.abs(p_ref - p_onx))),
        "predicted_class_agree": float(np.mean(np.argmax(p_ref, 1) == np.argmax(p_onx, 1))),
        "rul_hours_max_abs_diff": float(np.max(np.abs(r_ref - r_onx))),
    }

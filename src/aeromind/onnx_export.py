"""Export a trained ``ModelBundle`` to ONNX, and run it with ONNX Runtime.

Three self-contained float32 graphs, written directly with ``onnx.helper`` (no skl2onnx):

  anomaly.onnx     features (N,19)        -> score (N,), z (N,19)
                   scaler, Isolation Forest, autoencoder and the healthy calibration, all in-graph
  classifier.onnx  features (N,19)        -> probabilities (N,C)   classes listed in manifest.json
  rul.onnx         trend features (N,22)  -> rul_hours (N,3)       sorted p10 <= p50 <= p90, >= 0

Feature extraction, trend tracking, the persistence gate and advisory formatting stay in
host code (``pipeline.py``); only the learned models are exported.

The tree ensembles (Isolation Forest, fault classifier, RUL quantiles) can be exported two ways:

  trees="onnx-ml"      ONNX-ML ``TreeEnsembleRegressor`` nodes. Compact, exact, ONNX Runtime only:
                       TensorRT does not implement ONNX-ML operators.
  trees="hummingbird"  Hummingbird compiles each ensemble to plain tensor operations (GEMM or
                       tree traversal), so every graph uses only standard ONNX operators that
                       TensorRT's ONNX parser accepts. Needs ``torch`` and ``hummingbird-ml``.

Both produce the same inputs, outputs and manifest, so ``OnnxBundle`` loads either.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
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
TREE_BACKENDS = ("onnx-ml", "hummingbird")
HB_STRATEGIES = ("gemm", "tree_trav")  # perf_tree_trav unrolls to hundreds of MB here

# Every operator a trees="hummingbird" export may contain. All are listed as supported by
# TensorRT's ONNX parser (onnx-tensorrt docs/operators.md). An export containing anything
# else is refused, so a graph that TensorRT cannot parse is caught here rather than on the device.
TENSORRT_OPERATORS = frozenset({
    "Add", "Cast", "Concat", "Constant", "ConstantOfShape", "Div", "Equal", "Exp", "Expand",
    "Gather", "GatherElements", "Gemm", "LessOrEqual", "Log", "MatMul", "Mul", "Neg", "Pow",
    "ReduceMean", "ReduceSum", "Relu", "Reshape", "Shape", "Softmax", "Squeeze", "Sub", "Tanh",
    "TopK", "Transpose", "Unsqueeze", "Where",
})


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
    opsets = [helper.make_opsetid("", OPSET)]
    if any(n.domain == "ai.onnx.ml" for n in nodes):
        opsets.append(helper.make_opsetid("ai.onnx.ml", ML_OPSET))
    m = helper.make_model(graph, opset_imports=opsets, producer_name="aeromind")
    m.ir_version = 8
    checker.check_model(m, full_check=True)
    return m


def _const(name: str, value) -> "object":
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


# --------------------------------------------------------------------------- Hummingbird (tensor ops)


def _hist_gbdt_params(trees, extra_config):
    """Replacement for Hummingbird 0.4.12's ``_get_parameters_hist_gbdt``.

    The original stores every leaf value or threshold that is exactly 0 as -1 (a sentinel left
    over from other model formats). Gradient-boosted leaves are often exactly 0, so converted
    models were wrong: here 57 classifier leaves and one RUL leaf. This version passes the
    values through unchanged and marks leaves only through their children.
    """
    from hummingbird.ml.operator_converters._tree_commons import TreeParameters

    n = trees.nodes
    if n["is_categorical"].any():
        raise NotImplementedError("categorical splits are not supported")
    leaf = n["is_leaf"].astype(bool)
    return TreeParameters(
        np.where(leaf, -1, n["left"].astype(np.int64)).tolist(),
        np.where(leaf, -1, n["right"].astype(np.int64)).tolist(),
        n["feature_idx"].astype(np.int64).tolist(),
        n["num_threshold"].astype(np.float64).tolist(),
        [[v] for v in n["value"].astype(np.float64).tolist()],
    )


@contextlib.contextmanager
def _fixed_hummingbird():
    import hummingbird.ml.operator_converters.sklearn.gbdt as gbdt

    original = gbdt._get_parameters_hist_gbdt
    gbdt._get_parameters_hist_gbdt = _hist_gbdt_params
    try:
        yield
    finally:
        gbdt._get_parameters_hist_gbdt = original


def _hb_onnx(model, n_features: int, strategy: str):
    """Compile one fitted tree model to an ONNX graph of tensor ops via Hummingbird.

    Input ``x`` (N, n_features); outputs ``o0``, ``o1``, ... in Hummingbird's order.
    """
    import hummingbird.ml as hb
    import onnx
    import torch

    if strategy not in HB_STRATEGIES:
        raise ValueError(f"strategy must be one of {HB_STRATEGIES}")
    sample = np.zeros((2, n_features), dtype=np.float32)
    with _fixed_hummingbird():
        module = hb.convert(model, "torch", sample, extra_config={"tree_implementation": strategy}).model
    module.eval()
    x = torch.from_numpy(sample)
    with torch.no_grad():
        out = module(x)
    n_out = len(out) if isinstance(out, (tuple, list)) else 1
    names = [f"o{i}" for i in range(n_out)]
    buf = io.BytesIO()
    # The TorchScript exporter: Hummingbird 0.4.12 modules do not trace under torch.export (dynamo).
    torch.onnx.export(module, x, buf, dynamo=False, opset_version=OPSET, input_names=["x"],
                      output_names=names, dynamic_axes={"x": {0: "N"}, **{o: {0: "N"} for o in names}})
    return _pin_squeeze_axes(onnx.load_from_string(buf.getvalue()), n_features)


def _pin_squeeze_axes(model, n_features: int):
    """Give every axis-less ``Squeeze`` the axes it squeezes for a batch larger than one.

    Hummingbird calls ``tensor.squeeze()``, which exports without axes and would also drop the
    batch dimension when N == 1 (the edge pipeline's case). TensorRT also wants static axes.
    """
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    targets = [n for n in model.graph.node if n.op_type == "Squeeze" and len(n.input) < 2]
    if not targets:
        return model
    probe = copy.deepcopy(model)
    for n in targets:
        probe.graph.output.append(helper.make_empty_tensor_value_info(n.input[0]))
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    shapes = dict(zip([o.name for o in sess.get_outputs()],
                      [a.shape for a in sess.run(None, {"x": np.zeros((3, n_features), np.float32)})]))
    for k, n in enumerate(targets):
        axes = [i for i, d in enumerate(shapes[n.input[0]]) if d == 1]
        name = f"squeeze_axes_{k}"
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name))
        n.input.append(name)
    return model


def _inline(sub, prefix: str, inputs: dict[str, str], outputs: dict[str, str]):
    """Nodes and initializers of ``sub`` renamed into a host graph, pruned to ``outputs``."""
    from onnx import compose

    g = compose.add_prefix(sub, prefix).graph
    rename = {prefix + k: v for k, v in inputs.items()} | {prefix + k: v for k, v in outputs.items()}
    needed, keep = set(outputs.values()), []
    for node in reversed([copy.deepcopy(n) for n in g.node]):
        node.input[:] = [rename.get(i, i) for i in node.input]
        node.output[:] = [rename.get(o, o) for o in node.output]
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    inits = [i for i in g.initializer if i.name in needed]
    return keep[::-1], inits


def _int_labels(clf):
    """Hummingbird only converts classifiers with integer labels; class names live in the manifest."""
    c = copy.copy(clf)
    c.classes_ = np.arange(len(clf.classes_))
    return c


def build_anomaly_graph(det, trees: str = "onnx-ml", strategy: str = "gemm") -> "object":
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

    f = det.iforest
    if trees == "onnx-ml":
        # -score_samples = 2 ** (-depths / (n_trees * c(max_samples))).
        denom = len(f.estimators_) * float(_average_path_length([f._max_samples])[0])
        nodes.append(_iforest_trees(f).node(xs, "if_depths", 1, [0.0], "iforest"))
        inits.append(_const("if_k", [-np.log(2.0) / denom]))
        op("Mul", ["if_depths", "if_k"], "if_exp")
        raw_if = op("Exp", ["if_exp"], "if_score")  # (N,1)
    else:
        # Hummingbird returns (label, decision_function); -score_samples = -(decision + offset_).
        sub_nodes, sub_inits = _inline(_hb_onnx(f, N_FEATURES, strategy), "hb_if_", {"x": xs}, {"o1": "if_decision"})
        nodes += sub_nodes
        inits += sub_inits + [_const("if_offset", [f.offset_])]
        op("Add", ["if_decision", "if_offset"], "if_samples")
        raw_if = op("Neg", ["if_samples"], "if_score")  # (N,1)

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


def build_classifier_graph(clf, trees: str = "onnx-ml", strategy: str = "gemm") -> "object":
    from onnx import TensorProto, helper

    model = clf.model
    n_targets = len(model._predictors[0])
    if n_targets != len(model.classes_) or n_targets < 3:
        raise NotImplementedError("only multiclass (3+ classes) classifiers are supported")
    inits = []
    if trees == "onnx-ml":
        ens, _, base = _hgb_trees(model)
        nodes = [ens.node("features", "logits", n_targets, base, "fault_trees"),
                 helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)]
    else:
        # Hummingbird returns (label, probabilities); softmax is already in its graph.
        nodes, inits = _inline(_hb_onnx(_int_labels(model), N_FEATURES, strategy), "hb_clf_",
                               {"x": "features"}, {"o1": "probabilities"})
    f32 = TensorProto.FLOAT
    return _model(
        nodes,
        [helper.make_tensor_value_info("features", f32, ["N", N_FEATURES])],
        [helper.make_tensor_value_info("probabilities", f32, ["N", n_targets])],
        inits, "aeromind_classifier",
    )


def build_rul_graph(rul, trees: str = "onnx-ml", strategy: str = "gemm") -> "object":
    from onnx import TensorProto, helper

    nodes, inits = [], []
    if trees == "onnx-ml":
        ens, bases = _Trees.empty(), []
        offset = 0
        for target, q in enumerate(QUANTILES):
            t, n, base = _hgb_trees(rul.models[q])
            assert n == 1
            # Renumber trees so all three quantile models live in one ensemble, one target each.
            t.treeids = [i + offset for i in t.treeids]
            t.t_treeids = [i + offset for i in t.t_treeids]
            t.t_ids = [target] * len(t.t_ids)
            offset = max(t.treeids) + 1
            for field in ens.__dataclass_fields__:
                getattr(ens, field).extend(getattr(t, field))
            bases.append(base[0])
        nodes.append(ens.node("trend_features", "quantiles", len(QUANTILES), bases, "rul_trees"))
    else:
        for k, q in enumerate(QUANTILES):
            n, i = _inline(_hb_onnx(rul.models[q], N_TREND_FEATURES, strategy), f"hb_q{k}_",
                           {"x": "trend_features"}, {"o0": f"q{k}"})
            nodes += n
            inits += i
        nodes.append(helper.make_node("Concat", [f"q{k}" for k in range(len(QUANTILES))], ["quantiles"], axis=1))
    nodes += [
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
        inits, "aeromind_rul",
    )


def operators(model) -> list[str]:
    """Sorted operator types in a graph, prefixed with their domain when it is not the default."""
    return sorted({f"{n.domain}:{n.op_type}" if n.domain else n.op_type for n in model.graph.node})


def export_onnx(bundle: ModelBundle, out_dir: str | Path, trees: str = "onnx-ml", strategy: str = "gemm") -> dict:
    """Write the three graphs plus ``manifest.json`` to ``out_dir``. Returns the manifest.

    ``trees`` picks how tree ensembles are expressed (see the module docstring); ``strategy``
    is Hummingbird's tree implementation and only applies to ``trees="hummingbird"``.
    """
    import onnx
    import sklearn

    if trees not in TREE_BACKENDS:
        raise ValueError(f"trees must be one of {TREE_BACKENDS}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = {
        "anomaly": build_anomaly_graph(bundle.anomaly, trees, strategy),
        "classifier": build_classifier_graph(bundle.classifier, trees, strategy),
        "rul": build_rul_graph(bundle.rul, trees, strategy),
    }
    if trees == "hummingbird":
        for key, g in graphs.items():
            extra = set(operators(g)) - TENSORRT_OPERATORS
            if extra:
                raise ValueError(f"{key} graph uses operators outside the TensorRT list: {sorted(extra)}")
    files = {}
    for key, g in graphs.items():
        data = g.SerializeToString()
        (out / FILES[key]).write_bytes(data)
        files[key] = {"file": FILES[key], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "operators": operators(g)}
    tree_info = {"backend": trees}
    if trees == "hummingbird":
        import hummingbird
        import torch

        tree_info |= {"strategy": strategy, "hummingbird": hummingbird.__version__, "torch": torch.__version__}
    manifest = {
        "format_version": FORMAT_VERSION,
        "trees": tree_info,
        "opset": {"ai.onnx": OPSET, **({"ai.onnx.ml": ML_OPSET} if trees == "onnx-ml" else {})},
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

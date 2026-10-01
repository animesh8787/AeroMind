"""Streaming edge pipeline: window in, (rarely) advisory out."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .alerts import ACTIONS, PRIORITY_RANK, Advisory
from .config import HEALTHY, HOURS_PER_WINDOW, RAW_BYTES_PER_WINDOW
from .features import SequenceTracker, TrendTracker, extract_features
from .simulator import SensorWindow
from .train import ModelBundle


@dataclass
class PipelineConfig:
    anomaly_threshold: float = 1.0  # score units; 1.0 ~ 99th percentile of healthy windows
    persist_n: int = 5  # alarm only if >= persist_n of the last persist_of windows exceed threshold
    persist_of: int = 8
    min_confidence: float = 0.6  # below this the fault type is reported as unclassified
    critical_hours: float = 20.0  # priority is based on the conservative (p10) RUL
    urgent_hours: float = 60.0
    reemit_windows: int = 20  # re-send an unchanged advisory at most this often


@dataclass(frozen=True)
class Assessment:
    """What the models concluded for a window that passed the persistence gate."""

    fault: str
    confidence: float
    rul_hours: tuple[float, float, float]  # p10, p50, p90
    priority: str


@dataclass
class PipelineStats:
    windows: int = 0
    advisories: int = 0
    advisory_bytes: int = 0
    latency_ms_total: float = 0.0
    latency_ms_max: float = 0.0

    @property
    def raw_bytes(self) -> int:
        return self.windows * RAW_BYTES_PER_WINDOW

    @property
    def reduction_factor(self) -> float:
        return self.raw_bytes / max(self.advisory_bytes, 1)

    @property
    def latency_ms_mean(self) -> float:
        return self.latency_ms_total / max(self.windows, 1)


@dataclass
class EdgePipeline:
    bundle: ModelBundle
    component: str = "ENG1-GEARBOX"
    cfg: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        self._trend = TrendTracker()
        # A sequence model (LSTM) reads the last seq_len windows instead of trend features.
        rul = self.bundle.rul
        self._seq = None
        if getattr(rul, "input_kind", "trend") == "sequence":
            self._seq = SequenceTracker(rul.seq_len)
            rul.predict_batch(np.zeros((1, rul.seq_len, rul.n_in)))  # load the network before the first window
        self._flags: deque[bool] = deque(maxlen=self.cfg.persist_of)
        self._last: tuple[int, str, str] | None = None  # (window, fault, priority) of last emission
        self.stats = PipelineStats()
        # Per-window introspection for dashboards; never part of the downlink.
        self.last_features = None  # feature vector of the latest window
        self.last_score = 0.0
        self.last_flags = 0  # windows over threshold among the last persist_of
        self.last_assessment: Assessment | None = None
        self.last_latency_ms = 0.0

    def _priority(self, rul_p10: float) -> str:
        if rul_p10 <= self.cfg.critical_hours:
            return "CRITICAL"
        if rul_p10 <= self.cfg.urgent_hours:
            return "URGENT"
        return "ADVISORY"

    def process(self, w: SensorWindow) -> Advisory | None:
        start = time.perf_counter()
        adv = self._process(w)
        ms = (time.perf_counter() - start) * 1000
        self.last_latency_ms = ms
        s = self.stats
        s.windows += 1
        s.latency_ms_total += ms
        s.latency_ms_max = max(s.latency_ms_max, ms)
        if adv is not None:
            s.advisories += 1
            s.advisory_bytes += adv.size_bytes
        return adv

    def _process(self, w: SensorWindow) -> Advisory | None:
        b, cfg = self.bundle, self.cfg
        x = self.last_features = extract_features(w)
        score = float(b.anomaly.score(x)[0])
        self.last_score = score
        self.last_assessment = None
        trend = self._trend.update(x, score)
        seq = self._seq.update(x, score) if self._seq is not None else None
        self._flags.append(score > cfg.anomaly_threshold)
        self.last_flags = sum(self._flags)

        if self.last_flags < cfg.persist_n:
            if not any(self._flags):
                self._last = None  # anomaly cleared: next alarm is reported afresh
            return None

        fault, conf = b.classifier.predict(trend[: len(x)])
        if fault == HEALTHY:
            return None
        if conf < cfg.min_confidence:
            fault = "unclassified_anomaly"
        p10, p50, p90 = b.rul.predict(trend if seq is None else seq)
        priority = self._priority(p10)
        self.last_assessment = Assessment(fault, conf, (p10, p50, p90), priority)

        if self._last is not None:
            last_w, last_fault, last_prio = self._last
            unchanged = fault == last_fault and PRIORITY_RANK[priority] <= PRIORITY_RANK[last_prio]
            if unchanged and w.t - last_w < cfg.reemit_windows:
                return None
        self._last = (w.t, fault, priority)

        action, parts = ACTIONS[fault]
        return Advisory(
            component=self.component,
            window=w.t,
            flight_hours=round(w.t * HOURS_PER_WINDOW, 1),
            fault_type=fault,
            confidence=round(conf, 3),
            anomaly_score=round(score, 2),
            rul_hours_p10=round(p10, 1),
            rul_hours_p50=round(p50, 1),
            rul_hours_p90=round(p90, 1),
            priority=priority,
            recommended_action=action,
            parts_logistics=parts,
            contributing_signals=tuple(b.anomaly.explain(x)),
        )

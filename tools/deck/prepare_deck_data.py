"""Collect the numbers for the evidence deck from artifacts/report/results.json (no hand-typed values)."""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
res = json.loads((root / "artifacts/report/results.json").read_text())
out = {"results": res}
cache = root / "data/ims/ims_test2_features.npz"
if cache.exists():
    from aeromind.datasets.ims import FEATURES

    z = np.load(cache)
    hours, snr = z["hours"], z["X"][:, :, FEATURES.index("bpfo_snr")]
    step = max(1, len(hours) // 120)
    out["ims_curve"] = {"hours": [round(float(h), 1) for h in hours[::step]],
                        "bpfo_snr": [[round(float(v), 1) for v in snr[::step, b]] for b in range(4)]}
dest = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "artifacts/deck_data.json"
dest.write_text(json.dumps(out))
print(f"wrote {dest}")

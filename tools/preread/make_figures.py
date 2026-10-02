"""Figures for the pre-read. All numbers are copied from docs/evidence-report.md."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "preread" / "figures"
NAVY, TEAL, GREY, AMBER, LIGHT = "#0B2545", "#13A89E", "#5B6770", "#E09F3E", "#EAF2F7"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc=NAVY, tc="white", fs=8.5, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    # on-aircraft band
    ax.add_patch(FancyBboxPatch((0.2, 3.35), 19.6, 5.6, boxstyle="round,pad=0.02,rounding_size=0.15", fc=LIGHT, ec=TEAL, lw=1.5))
    ax.text(0.5, 8.6, "ON THE AIRCRAFT (edge, no connectivity needed)", color=TEAL, fontsize=9, fontweight="bold", va="center")
    ax.add_patch(FancyBboxPatch((0.2, 0.15), 19.6, 2.75, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=GREY, lw=1.2))
    ax.text(0.5, 2.55, "ON THE GROUND (ground station)", color=GREY, fontsize=9, fontweight="bold", va="center")

    top = [("6 sensor\nfamilies", GREY), ("Sensor\nhealth\n(mask bad\nchannels)", NAVY), ("Features\n(21, phase-\naware)", NAVY),
           ("Anomaly\nscore", NAVY), ("Persistence\ngate\n(5 of 8)", NAVY), ("Fault\nclassifier", NAVY),
           ("RUL p10/\np50/p90\n+ conformal", NAVY), ("Physics\nevidence\n(BPFO, THD)", TEAL)]
    w, gap, x = 1.95, 0.36, 0.5
    xs = []
    for label, c in top:
        box(ax, x, 5.0, w, 2.7, label, fc=c, fs=7.6)
        xs.append(x)
        x += w + gap
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w + 0.02, 6.35, b - 0.02, 6.35)
    box(ax, 10.3, 3.65, 9.2, 0.95, "Advisory: JSON + ACARS line (max 141 of 220 chars)", fc=AMBER, tc=NAVY, fs=8)
    arrow(ax, xs[-1] + w / 2, 5.0, xs[-1] + w / 2, 4.62, AMBER)
    ax.text(0.5, 4.15, "Raw data never leaves the aircraft.\nOnly the advisory is downlinked: 545x less data.",
            fontsize=8.2, color=NAVY, va="center")

    bot = ["Decision engine\n(ground, replace,\ndefer)", "Work orders\n(parts, hours)", "Fleet ROI\n(Monte Carlo)", "Signed OTA\n(Ed25519, rollback)", "Federated\naveraging"]
    w2, x = 3.4, 0.5
    for i, label in enumerate(bot):
        box(ax, x, 0.45, w2, 1.5, label, fc="white", tc=NAVY, fs=8)
        ax.add_patch(FancyBboxPatch((x, 0.45), w2, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc="none", ec=NAVY, lw=1.2))
        x += w2 + 0.37
    arrow(ax, 14.9, 3.65, 14.9, 2.0, AMBER)
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cmapss():
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    const = [48.5, 52.4, 62.9, 61.9]
    trees = [19.377, 17.745, 22.255, 20.1]
    lstm = [16.003, 14.645, 16.11, 15.781]
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    xs = range(4)
    bw = 0.26
    for off, vals, col, lab in [(-bw, const, "#B8C2CC", "Constant baseline"), (0, trees, NAVY, "Gradient boosting + conformal"), (bw, lstm, TEAL, "LSTM + conformal")]:
        bars = ax.bar([x + off for x in xs], vals, bw, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=7, color=NAVY)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(subsets)
    ax.set_ylabel("RUL RMSE (cycles, lower is better)", fontsize=8)
    ax.set_ylim(0, 72)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title("NASA C-MAPSS, official test split (public benchmark data)", fontsize=9, color=NAVY, loc="left")
    fig.savefig(OUT / "cmapss_rmse.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def roi():
    labels = ["Reactive", "Fixed interval\n(as measured)", "AeroMind\n(as measured)", "AeroMind\n(harsh assumptions)"]
    vals = [0.0, -24.0, -98.6, -34.4]
    cols = ["#B8C2CC", "#8795A3", TEAL, NAVY]
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    bars = ax.bar(labels, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 4 if v < 0 else 2, f"{v:.1f}%", ha="center", va="top" if v < 0 else "bottom", fontsize=8, color=NAVY, fontweight="bold")
    ax.set_ylim(-112, 10)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Maintenance cost vs reactive", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Fleet ROI simulation (assumed costs and failure rates)", fontsize=9, color=NAVY, loc="left")
    fig.savefig(OUT / "roi.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    cmapss()
    roi()
    print("ok")

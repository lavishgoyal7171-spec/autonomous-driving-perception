"""Evaluate a pipeline run from its per-frame CSV log.

Metrics (as listed in the synopsis - "Evaluation and Testing"):
  * lane-boundary detection rate and stability (frame-to-frame offset jitter)
  * object-detection count and confidence
  * traffic-signal classification (accuracy + confusion matrix if ground truth exists)
  * tracking consistency (track lengths, number of IDs)
  * processing speed
  * decision behaviour (distribution, switches/min, accuracy if ground truth exists)

    python scripts/evaluate.py --log outputs/project_video_log.csv
    python scripts/evaluate.py --log outputs/synthetic_road_log.csv --gt data/synthetic_road_gt.csv
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DECISION_COLORS = {"MAINTAIN": "#3cb43c", "STEER LEFT": "#1e8cc8", "STEER RIGHT": "#1e8cc8",
                   "SLOW DOWN": "#ffaa00", "STOP": "#dc2828"}
SIGNAL_LEVEL = {"NONE": 0, "UNKNOWN": 0, "GREEN": 1, "YELLOW": 2, "RED": 3}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def confusion(true, pred, labels):
    m = np.zeros((len(labels), len(labels)), int)
    idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(true, pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def plot_confusion(m, labels, title, path):
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.1 * len(labels) + 1.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate(log_path, gt_path=None, fps=25.0, out_dir=None):
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        raise SystemExit("empty log")
    name = os.path.splitext(os.path.basename(log_path))[0].replace("_log", "")
    out_dir = out_dir or os.path.join("outputs", "eval", name)
    os.makedirs(out_dir, exist_ok=True)
    n = len(rows)
    frames = np.array([int(r["frame"]) for r in rows])
    t = frames / fps

    # ---------------------------------------------------------------- speed
    proc = np.array([fnum(r["proc_ms"]) for r in rows])[1:] if n > 1 else np.array([fnum(rows[0]["proc_ms"])])
    speed = {"mean_ms": float(np.mean(proc)), "median_ms": float(np.median(proc)),
             "p95_ms": float(np.percentile(proc, 95)), "fps_mean": float(1000 / np.mean(proc))}

    # ---------------------------------------------------------------- lane
    detected = np.array([int(r["lane_detected"]) for r in rows])
    both = np.array([r["lane_status"] == "BOTH" for r in rows])
    off = np.array([fnum(r["offset_norm"]) for r in rows])
    off_m = np.array([fnum(r["offset_m"]) for r in rows])
    d_off = np.diff(off)
    lane = {
        "detection_rate": float(detected.mean()),
        "both_lines_fresh_rate": float(both.mean()),
        "status_counts": dict(Counter(r["lane_status"] for r in rows)),
        "offset_m_mean": float(np.nanmean(off_m)) if np.isfinite(off_m).any() else None,
        "offset_m_std": float(np.nanstd(off_m)) if np.isfinite(off_m).any() else None,
        "jitter_mean_abs_diff": float(np.nanmean(np.abs(d_off))) if np.isfinite(d_off).any() else None,
    }

    # ---------------------------------------------------------------- objects / tracking
    ndet = np.array([int(r["num_detections"]) for r in rows])
    confs = np.array([fnum(r["mean_conf"]) for r in rows])
    id_frames = defaultdict(int)
    for r in rows:
        for tid in r["track_ids"].split():
            id_frames[tid] += 1
    lengths = np.array(list(id_frames.values())) if id_frames else np.array([0])
    objects = {
        "mean_detections_per_frame": float(ndet.mean()),
        "frames_with_detections": float((ndet > 0).mean()),
        "mean_confidence": float(np.nanmean(confs)) if np.isfinite(confs).any() else None,
        "unique_tracks": len(id_frames),
        "mean_track_length_frames": float(lengths.mean()),
        "median_track_length_frames": float(np.median(lengths)),
        "tracks_longer_than_1s": int((lengths >= fps).sum()),
    }

    # ---------------------------------------------------------------- signal + decision
    sig = [r["signal_state"] for r in rows]
    dec = [r["decision"] for r in rows]
    switches = sum(a != b for a, b in zip(dec, dec[1:]))
    raw_switches = sum(a != b for a, b in zip([r["raw_decision"] for r in rows],
                                              [r["raw_decision"] for r in rows][1:]))
    minutes = max(n / fps / 60.0, 1e-6)
    signal = {"state_counts": dict(Counter(sig))}
    decision = {"counts": dict(Counter(dec)), "switches_per_min": switches / minutes,
                "raw_switches_per_min": raw_switches / minutes,
                "top_reasons": Counter(r["reason"] for r in rows).most_common(6)}

    gt_rows = None
    if gt_path and os.path.exists(gt_path):
        gt = {int(g["frame"]): g for g in csv.DictReader(open(gt_path))}
        gt_rows = [gt.get(f) for f in frames]
        pairs = [(g, r) for g, r in zip(gt_rows, rows) if g]
        s_true, s_pred = [g["signal"] for g, _ in pairs], [r["signal_state"] for _, r in pairs]
        d_true, d_pred = [g["expected_decision"] for g, _ in pairs], [r["decision"] for _, r in pairs]
        signal["accuracy"] = float(np.mean([a == b for a, b in zip(s_true, s_pred)]))
        decision["accuracy"] = float(np.mean([a == b for a, b in zip(d_true, d_pred)]))
        # per-class recall for decisions
        decision["per_class_recall"] = {
            c: float(np.mean([p == c for tr, p in zip(d_true, d_pred) if tr == c]))
            for c in sorted(set(d_true))}
        g_off = np.array([fnum(g["offset_norm"]) for g, _ in pairs])
        p_off = np.array([fnum(r["offset_norm"]) for _, r in pairs])
        lane["offset_mae_lane_widths"] = float(np.nanmean(np.abs(g_off - p_off)))
        s_labels = ["GREEN", "YELLOW", "RED", "NONE"]
        plot_confusion(confusion(s_true, s_pred, s_labels), s_labels, "Traffic-signal classification",
                       os.path.join(out_dir, "confusion_signal.png"))
        d_labels = ["MAINTAIN", "STEER LEFT", "STEER RIGHT", "SLOW DOWN", "STOP"]
        plot_confusion(confusion(d_true, d_pred, d_labels), d_labels, "Driving decision",
                       os.path.join(out_dir, "confusion_decision.png"))

    summary = {"video": name, "frames": n, "speed": speed, "lane": lane, "objects": objects,
               "signal": signal, "decision": decision}

    # ---------------------------------------------------------------- timeline plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax = axes[0]
    ax.plot(t, off, color="#1e5aa0", lw=1.2, label="estimated offset")
    if gt_rows:
        ax.plot(t, [fnum(g["offset_norm"]) if g else np.nan for g in gt_rows], "k--", lw=1, label="ground truth")
    ax.axhspan(-0.15, 0.15, color="#3cb43c", alpha=0.08)
    ax.set_ylabel("offset (lane widths)")
    ax.set_title(f"{name}: per-frame perception timeline")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.step(t, [SIGNAL_LEVEL.get(s, 0) for s in sig], where="post", color="#c03030", label="detected")
    if gt_rows:
        ax.step(t, [SIGNAL_LEVEL.get(g["signal"], 0) if g else np.nan for g in gt_rows], where="post",
                color="k", ls="--", lw=1, label="ground truth")
    ax.set_yticks([0, 1, 2, 3], ["none", "GREEN", "YELLOW", "RED"])
    ax.set_ylabel("signal")
    ax.legend(loc="upper right")
    ax = axes[2]
    ax.plot(t, ndet, color="#555", lw=1)
    ax.set_ylabel("# objects")
    ax = axes[3]
    labels = list(DECISION_COLORS)
    for i, d in enumerate(dec):
        ax.axvspan(t[i], t[i] + 1 / fps, color=DECISION_COLORS.get(d, "#999"), lw=0)
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1] + 1 / fps)
    ax.set_ylabel("decision")
    ax.set_xlabel("time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=DECISION_COLORS[l]) for l in
               ["MAINTAIN", "STEER LEFT", "SLOW DOWN", "STOP"]]
    ax.legend(handles, ["MAINTAIN", "STEER", "SLOW DOWN", "STOP"], loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timeline.png"), dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- speed + tracks
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(proc, bins=40, color="#1e5aa0")
    axes[0].axvline(speed["mean_ms"], color="r", ls="--", label=f"mean {speed['mean_ms']:.1f} ms")
    axes[0].set_xlabel("processing time per frame (ms)")
    axes[0].set_ylabel("frames")
    axes[0].legend()
    axes[1].hist(lengths, bins=30, color="#3cb43c")
    axes[1].set_xlabel("track length (frames)")
    axes[1].set_ylabel("tracks")
    axes[1].set_title(f"{objects['unique_tracks']} tracks")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_tracks.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print_summary(summary)
    print(f"\nplots + summary.json written to {out_dir}/")
    return summary


def print_summary(s):
    pct = lambda v: f"{100 * v:.1f}%"
    print(f"\n=== Evaluation: {s['video']} ({s['frames']} frames) ===")
    sp = s["speed"]
    print(f"Speed     : {sp['mean_ms']:.1f} ms/frame (median {sp['median_ms']:.1f}, p95 {sp['p95_ms']:.1f}) "
          f"-> {sp['fps_mean']:.1f} FPS")
    ln = s["lane"]
    print(f"Lane      : detected {pct(ln['detection_rate'])}, both lines fresh {pct(ln['both_lines_fresh_rate'])}, "
          f"jitter {ln['jitter_mean_abs_diff'] or 0:.4f} lane-widths/frame")
    if "offset_mae_lane_widths" in ln:
        print(f"            offset MAE vs ground truth {ln['offset_mae_lane_widths']:.4f} lane widths")
    ob = s["objects"]
    conf = f"{ob['mean_confidence']:.2f}" if ob["mean_confidence"] is not None else "-"
    print(f"Objects   : {ob['mean_detections_per_frame']:.2f}/frame, mean conf {conf}, "
          f"{ob['unique_tracks']} tracks, mean length {ob['mean_track_length_frames']:.1f} frames")
    sg = s["signal"]
    print(f"Signal    : {sg['state_counts']}" + (f", accuracy {pct(sg['accuracy'])}" if "accuracy" in sg else ""))
    dc = s["decision"]
    print(f"Decision  : {dc['counts']}")
    print(f"            {dc['switches_per_min']:.1f} switches/min (raw rules: {dc['raw_switches_per_min']:.1f})"
          + (f", accuracy {pct(dc['accuracy'])}" if "accuracy" in dc else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--gt", help="ground-truth CSV (synthetic video)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", help="output directory")
    a = ap.parse_args()
    evaluate(a.log, a.gt, a.fps, a.out)

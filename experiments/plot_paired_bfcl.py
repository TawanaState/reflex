#!/usr/bin/env python3
"""Create a measured two-panel figure from a completed paired summary."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    if data["complete_pairs"] != 500:
        raise ValueError("Refusing to label a partial run as the primary 500-request figure")
    groups = data["groups"]
    names = ["atomic", "primitive", "freetext", "no_tool"]
    labels = ["Empty args", "Primitive args", "Longer text", "No tool"]
    points = [groups[name]["paired_mean_saved_ms"] for name in names]
    intervals = [groups[name]["paired_mean_saved_ms_ci95"] for name in names]
    errors = [[point - lo for point, (lo, hi) in zip(points, intervals)],
              [hi - point for point, (lo, hi) in zip(points, intervals)]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    left.axhline(0, color="#555555", lw=1)
    left.errorbar(range(4), points, yerr=errors, fmt="o", color="#1261a0", capsize=4)
    left.set_xticks(range(4), [f"{label}\n(n={groups[name]['n']})" for label, name in zip(labels, names)])
    left.set_ylabel("Paired mean wall time saved (ms)")
    left.set_title("Reflex vs official sampler")
    left.grid(axis="y", alpha=0.2)

    ablation = data["atomic_ablation"]
    for policy, label, color in [
        ("reflex_1", "1 draft", "#1261a0"),
        ("reflex_2", "2 drafts", "#2c8f62"),
        ("reflex_3", "3 drafts", "#9b6518"),
        ("base", "Official", "#333333"),
    ]:
        if policy not in ablation:
            continue
        arm = ablation[policy]
        right.scatter(arm["mean_ms"], arm["correct"] / arm["n"], s=60, color=color)
        right.annotate(f"{label}: {arm['correct']}/{arm['n']}",
                       (arm["mean_ms"], arm["correct"] / arm["n"]),
                       xytext=(5, 5), textcoords="offset points", fontsize=8)
    right.set_xlabel("Mean wall time on empty-argument calls (ms)")
    right.set_ylabel("Strict call accuracy")
    right.set_title("Atomic stability ablation (n=25)")
    right.grid(alpha=0.2)
    right.set_ylim(-0.04, 1.08)
    for suffix in (".png", ".svg"):
        output = args.summary.with_suffix(suffix)
        fig.savefig(output, dpi=180)
        print(output)


if __name__ == "__main__":
    main()

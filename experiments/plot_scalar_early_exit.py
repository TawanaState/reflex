#!/usr/bin/env python3
"""Plot only measured complete-request scalar-exit results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    if data["complete_three_arm_requests"] != 214:
        raise ValueError("Primary figure requires all 214 complete requests")
    groups = data["groups"]
    names = ["scalar_positive", "no_tool_scalar_distractor", "atomic_control", "open_text_control"]
    labels = ["Scalar gold", "No tool; scalar distractor", "Atomic control", "Open-text control"]
    values = [groups[name]["contrasts"]["scalar_2_vs_atomic_only"]["mean_saved_ms"] for name in names]
    bounds = [groups[name]["contrasts"]["scalar_2_vs_atomic_only"]["saved_ms_ci95"] for name in names]
    errors = [[value - lo for value, (lo, hi) in zip(values, bounds)],
              [hi - value for value, (lo, hi) in zip(values, bounds)]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    left, right = axes
    left.axhline(0, color="#555555", lw=1)
    left.errorbar(range(len(names)), values, yerr=errors, fmt="o", capsize=4, color="#145e9a")
    left.set_xticks(range(len(names)), [f"{label}\n(n={groups[name]['n']})" for label, name in zip(labels, names)])
    left.tick_params(axis="x", labelsize=8)
    left.set_ylabel("Paired mean wall time saved (ms)")
    left.set_title("Two-draft scalar gate vs atomic-only")
    left.grid(axis="y", alpha=0.2)

    for policy, color, label in [
        ("official", "#333333", "Official"),
        ("atomic_only", "#145e9a", "Atomic-only"),
        ("scalar_1", "#b66718", "Scalar: 1 draft"),
        ("scalar_2", "#288353", "Scalar: 2 drafts"),
    ]:
        arm = groups["scalar_positive"]["arms"][policy]
        right.scatter(arm["mean_ms"], arm["correct"] / arm["n"], s=70, color=color)
        right.annotate(f"{label} ({arm['correct']}/{arm['n']})",
                       (arm["mean_ms"], arm["correct"] / arm["n"]),
                       xytext=(5, 5), textcoords="offset points", fontsize=8)
    right.set_xlabel("Mean wall time on scalar gold calls (ms)")
    right.set_ylabel("Strict-call score")
    right.set_ylim(-0.04, 1.08)
    right.set_title("Positive-call latency and outcome")
    right.grid(alpha=0.2)
    for suffix in (".svg", ".png"):
        path = args.summary.with_suffix(suffix)
        fig.savefig(path, dpi=180)
        print(path)


if __name__ == "__main__":
    main()

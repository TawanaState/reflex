#!/usr/bin/env python3
"""Analyze complete same-request scalar-exit comparisons from a real trace."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.summarize_paired_bfcl import ci, percentile


def correct(row):
    return row["no_tool_ok"] if row["arg_bucket"] == "no_tool" else row["strict_call_ok"]


def signature(row):
    return row["pred_tool"], row["pred_args"], row["content"]


def summarize(rows, query_by_id):
    by_id = defaultdict(dict)
    for row in rows:
        if row["policy"] in by_id[row["id"]]:
            raise ValueError(f"Duplicate policy for {row['id']}")
        by_id[row["id"]][row["policy"]] = row
    groups = defaultdict(list)
    for pair in by_id.values():
        if {"official", "atomic_only", "scalar_2"} <= pair.keys():
            groups["overall"].append(pair)
            groups[pair["official"]["bucket"]].append(pair)
    out = {"complete_three_arm_requests": len(groups["overall"]), "groups": {}}
    contrasts = [
        ("atomic_only", "official"),
        ("scalar_2", "atomic_only"),
        ("scalar_2", "official"),
        ("scalar_1", "atomic_only"),
    ]
    for group, pairs in groups.items():
        result = {"n": len(pairs), "arms": {}, "contrasts": {}}
        for policy in ("official", "atomic_only", "scalar_1", "scalar_2"):
            arm = [p[policy] for p in pairs if policy in p]
            if not arm:
                continue
            result["arms"][policy] = {
                "n": len(arm), "correct": sum(correct(r) for r in arm),
                "mean_ms": statistics.mean(r["latency_ms"] for r in arm),
                "p50_ms": percentile([r["latency_ms"] for r in arm], 50),
                "p95_ms": percentile([r["latency_ms"] for r in arm], 95),
                "mean_forward_estimate": statistics.mean(r["forward_passes_estimate"] for r in arm),
                "atomic_exits": sum(r["early_kind"] == "atomic" for r in arm),
                "scalar_exits": sum(r["early_kind"] == "scalar" for r in arm),
                "correct_scalar_exits": sum(r["early_kind"] == "scalar" and correct(r) for r in arm),
                "execution_paths": dict(Counter(r["execution_path"] for r in arm)),
            }
        for treatment, control in contrasts:
            complete = [p for p in pairs if treatment in p and control in p]
            if not complete:
                continue
            saved = [p[control]["latency_ms"] - p[treatment]["latency_ms"] for p in complete]
            deltas = [int(correct(p[treatment])) - int(correct(p[control])) for p in complete]
            matched = [signature(p[treatment]) == signature(p[control]) for p in complete]
            labels = [query_by_id.get(p[control]["id"], p[control]["id"]) for p in complete]
            result["contrasts"][f"{treatment}_vs_{control}"] = {
                "n": len(complete),
                "mean_saved_ms": statistics.mean(saved),
                "median_saved_ms": statistics.median(saved),
                "saved_ms_ci95": ci(saved, labels),
                "relative_mean_latency_reduction": statistics.mean(saved) /
                    statistics.mean(p[control]["latency_ms"] for p in complete),
                "matching_content_and_call": sum(matched),
                "changed_ids": [p[control]["id"] for p, same in zip(complete, matched) if not same],
                "accuracy_improved": sum(d > 0 for d in deltas),
                "accuracy_regressed": sum(d < 0 for d in deltas),
                "both_correct_n": sum(correct(p[treatment]) and correct(p[control]) for p in complete),
            }
        out["groups"][group] = result
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    manifest_path = args.trace.with_suffix(".manifest.json")
    input_path = args.input or Path(json.loads(manifest_path.read_text())["input"])
    frozen = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    query_by_id = {r["id"]: " ".join(r["user_query"].casefold().split()) for r in frozen}
    summary = summarize(rows, query_by_id)
    summary["bootstrap_unit"] = "normalized exact query"
    output = args.trace.with_suffix(".summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

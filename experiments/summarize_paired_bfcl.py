#!/usr/bin/env python3
"""Summarize only complete interleaved pairs; bootstrap over request IDs."""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p / 100
    low, high = int(x), min(int(x) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (x - low)


def accuracy(row):
    return row["no_tool_ok"] if row["bucket"] == "no_tool" else row["strict_call_ok"]


def ci(values, labels=None, seed=23, samples=4000):
    if not values:
        return None
    rng = random.Random(seed)
    if labels is None:
        means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    else:
        clusters = defaultdict(list)
        for label, value in zip(labels, values):
            clusters[label].append(value)
        keys = list(clusters)
        means = [statistics.mean(v for key in rng.choices(keys, k=len(keys))
                                 for v in clusters[key]) for _ in range(samples)]
    return [percentile(means, 2.5), percentile(means, 97.5)]


def summarize(rows, query_by_id=None):
    by_id = defaultdict(dict)
    for row in rows:
        if row["policy"] in by_id[row["id"]]:
            raise ValueError(f"Duplicate policy for {row['id']}")
        by_id[row["id"]][row["policy"]] = row
    groups = {"overall": []}
    for pair in by_id.values():
        if "base" not in pair or "reflex_1" not in pair:
            continue
        bucket = pair["base"]["bucket"]
        groups.setdefault(bucket, []).append(pair)
        if bucket == "no_tool":
            subgroup = ("no_tool_atomic_distractor" if pair["base"]["source"] == "BFCL_v4_live_irrelevance"
                        else "no_tool_other")
            groups.setdefault(subgroup, []).append(pair)
        groups["overall"].append(pair)
    out = {"complete_pairs": len(groups["overall"]), "groups": {}}
    for group, pairs in groups.items():
        base = [p["base"] for p in pairs]
        reflex = [p["reflex_1"] for p in pairs]
        differences = [p["base"]["latency_ms"] - p["reflex_1"]["latency_ms"] for p in pairs]
        accuracy_differences = [int(accuracy(p["reflex_1"])) - int(accuracy(p["base"])) for p in pairs]
        both_correct_savings = [p["base"]["latency_ms"] - p["reflex_1"]["latency_ms"]
                                for p in pairs if accuracy(p["base"]) and accuracy(p["reflex_1"])]
        cluster_labels = ([query_by_id.get(p["base"]["id"], p["base"]["id"]) for p in pairs]
                          if query_by_id else None)
        out["groups"][group] = {
            "n": len(pairs),
            "matching_content_and_call": sum(
                (p["base"]["pred_tool"], p["base"]["pred_args"], p["base"]["content"])
                == (p["reflex_1"]["pred_tool"], p["reflex_1"]["pred_args"], p["reflex_1"]["content"])
                for p in pairs
            ),
            "base": {
                "accuracy": sum(accuracy(r) for r in base) / len(base),
                "name_accuracy": (sum(r["name_ok"] is True for r in base) / sum(r["name_ok"] is not None for r in base))
                    if any(r["name_ok"] is not None for r in base) else None,
                "mean_ms": statistics.mean(r["latency_ms"] for r in base),
                "p50_ms": percentile([r["latency_ms"] for r in base], 50),
                "p95_ms": percentile([r["latency_ms"] for r in base], 95),
                "mean_forward_estimate": statistics.mean(r["forward_passes_estimate"] for r in base),
                "execution_paths": dict(Counter(r["execution_path"] for r in base)),
            },
            "reflex_1": {
                "accuracy": sum(accuracy(r) for r in reflex) / len(reflex),
                "name_accuracy": (sum(r["name_ok"] is True for r in reflex) / sum(r["name_ok"] is not None for r in reflex))
                    if any(r["name_ok"] is not None for r in reflex) else None,
                "mean_ms": statistics.mean(r["latency_ms"] for r in reflex),
                "p50_ms": percentile([r["latency_ms"] for r in reflex], 50),
                "p95_ms": percentile([r["latency_ms"] for r in reflex], 95),
                "mean_forward_estimate": statistics.mean(r["forward_passes_estimate"] for r in reflex),
                "early_exits": sum(r["early_exit"] for r in reflex),
                "early_exits_correct": sum(r["early_exit"] and accuracy(r) for r in reflex),
                "execution_paths": dict(Counter(r["execution_path"] for r in reflex)),
            },
            "paired_mean_saved_ms": statistics.mean(differences),
            "paired_median_saved_ms": statistics.median(differences),
            "paired_saved_ms_iqr": [percentile(differences, 25), percentile(differences, 75)],
            "both_correct_n": len(both_correct_savings),
            "both_correct_mean_saved_ms": statistics.mean(both_correct_savings) if both_correct_savings else None,
            "paired_mean_saved_ms_ci95": ci(differences, cluster_labels),
            "relative_mean_latency_reduction": statistics.mean(differences) / statistics.mean(r["latency_ms"] for r in base),
            "accuracy_delta": statistics.mean(accuracy_differences),
            "accuracy_delta_ci95": (ci(accuracy_differences, cluster_labels)
                                     if any(accuracy_differences) else None),
            "accuracy_delta_interval_note": ("All observed paired accuracy differences are zero; "
                                             "a resampling interval would be degenerate and is not a risk bound")
                                             if not any(accuracy_differences) else None,
            "improved": sum(d > 0 for d in accuracy_differences),
            "regressed": sum(d < 0 for d in accuracy_differences),
        }
    atomic = groups.get("atomic", [])
    if atomic:
        ablation = {}
        for policy in ("reflex_1", "reflex_2", "reflex_3", "base"):
            arm = [p[policy] for p in atomic if policy in p]
            if not arm:
                continue
            ablation[policy] = {
                "n": len(arm), "correct": sum(accuracy(r) for r in arm),
                "mean_ms": statistics.mean(r["latency_ms"] for r in arm),
                "p50_ms": percentile([r["latency_ms"] for r in arm], 50),
                "p95_ms": percentile([r["latency_ms"] for r in arm], 95),
                "mean_forward_estimate": statistics.mean(r["forward_passes_estimate"] for r in arm),
                "early_exits": sum(r["early_exit"] for r in arm),
            }
        out["atomic_ablation"] = ablation
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--input", type=Path, help="Frozen dataset, used for duplicate-query cluster bootstrap")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    input_path = args.input
    manifest_path = args.trace.with_suffix(".manifest.json")
    if input_path is None and manifest_path.exists():
        input_path = Path(json.loads(manifest_path.read_text())["input"])
    query_by_id = None
    if input_path is not None:
        frozen = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
        query_by_id = {r["id"]: " ".join(r["user_query"].casefold().split()) for r in frozen}
    summary = summarize(rows, query_by_id)
    summary["bootstrap_unit"] = "normalized exact query" if query_by_id else "request ID"
    path = args.trace.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

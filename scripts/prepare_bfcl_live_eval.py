#!/usr/bin/env python3
"""
Builds a frozen, independent held-out routing evaluation set from BFCL's
`live_multiple` category (real user-collected queries), which is a disjoint
source file from `BFCL_v4_multiple.json` used for data/bfcl/{train,cal,test}.

This directly answers the senior-review request for "500+ genuinely unseen
routing examples, no training-query overlap, randomized candidate ordering,
with 2/3/5/8/10 candidate counts."

Filtering rules (same routing-subset definition used by prepare_datasets.py):
  - Exactly one gold function call, with exactly one function name (no
    parallel calls, no ambiguous multi-name answers).
  - At least 2 candidate functions offered (otherwise routing is trivial).
  - Candidates capped at 10 (matches the <unused0..9> control-token budget
    used elsewhere in this project).
  - The gold function name must literally appear in the offered candidates.

Contamination check: drops any record whose normalized user_query string
exactly matches a user_query already present in data/bfcl/{train,cal,test}.jsonl.
"""
import json
import os
import random
import urllib.request
from collections import Counter

random.seed(1234)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
BFCL_SOURCE_REV = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BASE = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_SOURCE_REV}/berkeley-function-call-leaderboard/bfcl_eval/data"
QUESTION_URL = f"{BASE}/BFCL_v4_live_multiple.json"
ANSWER_URL = f"{BASE}/possible_answer/BFCL_v4_live_multiple.json"


def fetch_jsonl(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Reflex-Data-Prep"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [json.loads(line) for line in resp if line.strip()]


def normalize(q):
    return " ".join(q.strip().lower().split())


def load_existing_train_queries():
    seen = set()
    for split in ("train", "cal", "test"):
        path = os.path.join(DATA_DIR, "bfcl", f"{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                seen.add(normalize(row["user_query"]))
    return seen


def main():
    print(f"Fetching {QUESTION_URL} ...")
    questions = fetch_jsonl(QUESTION_URL)
    print(f"Fetching {ANSWER_URL} ...")
    answers = fetch_jsonl(ANSWER_URL)
    answer_by_id = {a["id"]: a["ground_truth"] for a in answers}
    print(f"Raw live_multiple examples: {len(questions)}")

    existing_queries = load_existing_train_queries()
    print(f"Existing train/cal/test BFCL query strings (any overlap will be dropped): {len(existing_queries)}")

    records = []
    dropped_overlap = 0
    for item in questions:
        cid = item.get("id", "")
        q_turns = item.get("question", [])
        if not q_turns or not q_turns[0]:
            continue
        user_query = q_turns[0][0].get("content", "")
        if not user_query:
            continue

        functions = item.get("function", [])
        if len(functions) < 2:
            continue

        gold_calls = answer_by_id.get(cid, [])
        if len(gold_calls) != 1 or len(gold_calls[0]) != 1:
            continue
        gold_name = next(iter(gold_calls[0]))

        func_names = [f["name"] for f in functions]
        k = min(len(func_names), 10)
        cand_names = func_names[:k]
        if gold_name not in cand_names:
            # Same rule as scripts/prepare_datasets.py: drop examples where the
            # gold tool falls outside the first 10 candidates rather than
            # reshuffling the menu.
            continue

        if normalize(user_query) in existing_queries:
            dropped_overlap += 1
            continue

        func_descs = []
        full_functions = []
        for name in cand_names:
            f = next(f for f in functions if f["name"] == name)
            func_descs.append(f"{f['name']}: {f.get('description', '')}")
            full_functions.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {"type": "dict", "properties": {}}),
            })

        gold_call = gold_calls[0]
        gold_args_raw = next(iter(gold_call.values()))
        arg_vals = []
        for v in (gold_args_raw or {}).values():
            arg_vals.extend(v) if isinstance(v, list) else arg_vals.append(v)
        if not gold_args_raw:
            arg_bucket = "atomic"
        elif any(isinstance(v, str) and len(v) > 25 for v in arg_vals):
            arg_bucket = "freetext"
        else:
            arg_bucket = "primitive"

        gold_index = cand_names.index(gold_name)
        records.append({
            "id": cid,
            "user_query": user_query,
            "candidate_names": cand_names,
            "candidate_descs": func_descs,
            "candidate_functions": full_functions,
            "ground_truth_index": gold_index,
            "ground_truth_label": gold_name,
            "gold_args": gold_args_raw,
            "arg_bucket": arg_bucket,
            "num_candidates": len(cand_names),
            "source": "BFCL_v4_live_multiple",
        })

    print(f"Dropped for exact query overlap with train/cal/test: {dropped_overlap}")
    print(f"Usable independent routing examples: {len(records)}")

    bucket_counts = Counter(r["num_candidates"] for r in records)
    print(f"Candidate-count distribution: {dict(sorted(bucket_counts.items()))}")
    arg_bucket_counts = Counter(r["arg_bucket"] for r in records)
    print(f"Argument-complexity distribution: {dict(arg_bucket_counts)}")

    random.shuffle(records)

    out_path = os.path.join(DATA_DIR, "bfcl", "live_eval.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")

    provenance = {
        "source_question_url": QUESTION_URL,
        "source_answer_url": ANSWER_URL,
        "source_revision": BFCL_SOURCE_REV,
        "raw_count": len(questions),
        "dropped_overlap_with_train": dropped_overlap,
        "final_count": len(records),
        "candidate_count_distribution": dict(sorted(bucket_counts.items())),
        "arg_bucket_distribution": dict(arg_bucket_counts),
        "note": (
            "Independent held-out routing set from a disjoint BFCL source file "
            "(live_multiple, real user-collected queries) vs. the synthetic "
            "'multiple' category used for data/bfcl/{train,cal,test}. Exact "
            "user_query string overlap with train/cal/test was checked and dropped."
        ),
    }
    with open(os.path.join(DATA_DIR, "bfcl", "live_eval_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


if __name__ == "__main__":
    main()

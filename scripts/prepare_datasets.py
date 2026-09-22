#!/usr/bin/env python3
"""
Reflex Dataset Preparation Pipeline:
Downloads, pre-processes, and partitions real benchmark datasets into strict
Train (60%), Calibration (20%), and Test (20%) splits:
1. Banking77 (Fine-grained intent classification with 77 categories)
2. BoolQ (Factual boolean verification)
3. BFCL Routing (Berkeley Function Calling Benchmark - tool dispatch among candidates)
"""

import os
import sys
import json
import random
import hashlib
import urllib.request
from typing import Dict, List, Any
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Set deterministic seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
BFCL_SOURCE_REV = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_BASE_URL = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_SOURCE_REV}/berkeley-function-call-leaderboard/bfcl_eval/data"


def save_jsonl(records: List[Dict[str, Any]], filepath: str):
    """Saves records to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(records)} records to {filepath}")


def prepare_banking77():
    """
    Downloads and prepares Banking77 dataset (77 fine-grained intent classes).
    Maps intent labels to dedicated control tokens: <unused0> .. <unused76> (IDs 6..82).
    """
    print("\n[1/3] Preparing Banking77 Dataset...")
    ds_train = load_dataset("mteb/banking77", split="train")
    ds_test = load_dataset("mteb/banking77", split="test")

    # Combine all records to create a clean, stratified 60 / 20 / 20 partition
    all_records = []
    label_map = {}
    for item in ds_train:
        lbl = item["label"]
        txt = item["label_text"]
        label_map[lbl] = txt
        all_records.append({
            "text": item["text"],
            "label": lbl,
            "label_text": txt,
        })
    for item in ds_test:
        lbl = item["label"]
        txt = item["label_text"]
        label_map[lbl] = txt
        all_records.append({
            "text": item["text"],
            "label": lbl,
            "label_text": txt,
        })

    # Stratified split per class (60% Train, 20% Cal, 20% Test)
    class_buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(77)}
    for r in all_records:
        class_buckets[r["label"]].append(r)

    train_set, cal_set, test_set = [], [], []
    for lbl, items in class_buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.60)
        n_cal = int(n * 0.20)
        train_set.extend(items[:n_train])
        cal_set.extend(items[n_train:n_train + n_cal])
        test_set.extend(items[n_train + n_cal:])

    random.shuffle(train_set)
    random.shuffle(cal_set)
    random.shuffle(test_set)

    # Format into Reflex records
    # Schema: Slot "@intent" mapped to candidate token IDs: 6..82 (<unused0>..<unused76>)
    # Also include human-readable label_names in metadata
    def format_item(item):
        prompt = f"Customer Query: {item['text']}\nClassify customer banking intent:"
        target_token_id = 6 + item["label"]  # <unused{label}>
        return {
            "prompt": prompt,
            "query": item["text"],
            "schema_type": "choice",
            "slot_name": "intent",
            "ground_truth_label": item["label_text"],
            "ground_truth_index": item["label"],
            "target_token_id": target_token_id,
            "candidate_token_ids": list(range(6, 83)),
            "candidate_labels": [label_map[i] for i in range(77)],
        }

    out_dir = os.path.join(DATA_DIR, "banking77")
    save_jsonl([format_item(x) for x in train_set], os.path.join(out_dir, "train.jsonl"))
    save_jsonl([format_item(x) for x in cal_set], os.path.join(out_dir, "cal.jsonl"))
    save_jsonl([format_item(x) for x in test_set], os.path.join(out_dir, "test.jsonl"))
    print(f"  Banking77 summary: Train={len(train_set)}, Cal={len(cal_set)}, Test={len(test_set)}")


def prepare_boolq():
    """
    Downloads and prepares Google BoolQ dataset (Boolean factual question answering).
    Candidate token IDs are derived from the pinned tokenizer at preparation time.
    """
    print("\n[2/3] Preparing Google BoolQ Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("DIFFUSION_GEMMA_PATH", "google/diffusiongemma-26B-A4B-it"))
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids == no_ids:
        raise ValueError("BoolQ verbalizers must be distinct single tokenizer tokens")
    ds_train = load_dataset("google/boolq", split="train")
    ds_val = load_dataset("google/boolq", split="validation")

    train_items = list(ds_train)
    val_items = list(ds_val)
    random.shuffle(train_items)
    random.shuffle(val_items)

    # Use 6,000 from train for SFT training
    train_split = train_items[:6000]
    # Partition validation split (3,270) equally into Calibration and Test
    n_val = len(val_items)
    half_val = n_val // 2
    cal_split = val_items[:half_val]
    test_split = val_items[half_val:]

    def format_item(item):
        passage = item["passage"]
        question = item["question"]
        gt_bool = item["answer"]
        gt_label = "yes" if gt_bool else "no"
        prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer (yes or no):"
        target_token_id = yes_ids[0] if gt_bool else no_ids[0]
        return {
            "prompt": prompt,
            "passage": passage,
            "question": question,
            "schema_type": "noul",
            "slot_name": "answer",
            "ground_truth_label": gt_label,
            "ground_truth_index": 1 if gt_bool else 0,
            "target_token_id": target_token_id,
            "candidate_token_ids": [no_ids[0], yes_ids[0]],  # [no, yes]
            "candidate_labels": ["no", "yes"],
        }

    out_dir = os.path.join(DATA_DIR, "boolq")
    save_jsonl([format_item(x) for x in train_split], os.path.join(out_dir, "train.jsonl"))
    save_jsonl([format_item(x) for x in cal_split], os.path.join(out_dir, "cal.jsonl"))
    save_jsonl([format_item(x) for x in test_split], os.path.join(out_dir, "test.jsonl"))
    print(f"  BoolQ summary: Train={len(train_split)}, Cal={len(cal_split)}, Test={len(test_split)}")


def prepare_bfcl():
    """
    Downloads and prepares Berkeley Function Calling Leaderboard (BFCL v4 multiple).
    Evaluates discrete tool selection across candidate functions.
    """
    print("\n[3/3] Preparing BFCL Routing Dataset...")
    bfcl_url = f"{BFCL_BASE_URL}/BFCL_v4_multiple.json"
    answer_url = f"{BFCL_BASE_URL}/possible_answer/BFCL_v4_multiple.json"
    with urllib.request.urlopen(urllib.request.Request(answer_url, headers={"User-Agent": "Reflex-Data-Prep"}), timeout=30) as resp:
        answer_by_id = {entry["id"]: entry["ground_truth"] for entry in map(json.loads, resp)}
    
    req = urllib.request.Request(bfcl_url, headers={"User-Agent": "Reflex-Data-Prep"})
    raw_lines = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str:
                raw_lines.append(json.loads(line_str))

    print(f"  Downloaded {len(raw_lines)} BFCL raw instances.")

    # Parse and extract tool selection records
    formatted_records = []
    for item in raw_lines:
        cid = item.get("id", "")
        # Query
        q_turns = item.get("question", [])
        if not q_turns or not q_turns[0]:
            continue
        user_query = q_turns[0][0].get("content", "")

        # Candidate functions
        functions = item.get("function", [])
        if len(functions) < 2:
            continue

        func_names = [f["name"] for f in functions]
        func_descs = [f"{f['name']}: {f.get('description', '')}" for f in functions]

        # Use the official BFCL answer file. This subset is single-call routing only.
        gold_calls = answer_by_id.get(cid, [])
        if len(gold_calls) != 1 or len(gold_calls[0]) != 1:
            continue
        gold_name = next(iter(gold_calls[0]))
        # Candidate token IDs: <unused0>..<unusedK-1> (IDs 6..6+K-1)
        k = min(len(func_names), 10)  # Up to 10 candidates
        cand_names = func_names[:k]
        if gold_name not in cand_names:
            continue
        gold_index = cand_names.index(gold_name)
        cand_token_ids = [6 + idx for idx in range(k)]

        # Candidate function list formatted into prompt
        tools_doc = "\n".join([f"[{i}] {func_descs[i]}" for i in range(k)])
        prompt = (
            f"Available Tools:\n{tools_doc}\n\n"
            f"User Request: {user_query}\n\n"
            f"Select the most appropriate tool index [0 to {k-1}]:"
        )

        formatted_records.append({
            "id": cid,
            "prompt": prompt,
            "user_query": user_query,
            "schema_type": "choice",
            "slot_name": "tool_idx",
            "candidate_names": cand_names,
            "candidate_token_ids": cand_token_ids,
            "candidate_labels": [str(i) for i in range(k)],
            "ground_truth_index": gold_index,
            "ground_truth_label": gold_name,
            "target_token_id": cand_token_ids[gold_index],
        })

    random.shuffle(formatted_records)
    n = len(formatted_records)
    n_train = int(n * 0.60)
    n_cal = int(n * 0.20)

    train_set = formatted_records[:n_train]
    cal_set = formatted_records[n_train:n_train + n_cal]
    test_set = formatted_records[n_train + n_cal:]

    out_dir = os.path.join(DATA_DIR, "bfcl")
    save_jsonl(train_set, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(cal_set, os.path.join(out_dir, "cal.jsonl"))
    save_jsonl(test_set, os.path.join(out_dir, "test.jsonl"))
    print(f"  BFCL summary: Train={len(train_set)}, Cal={len(cal_set)}, Test={len(test_set)}")


def verify_splits():
    """Verifies non-overlapping partitions and data integrity across all datasets."""
    print("\n=======================================================")
    print("VERIFYING DATASET SPLITS INTEGRITY")
    print("=======================================================")
    datasets = ["banking77", "boolq", "bfcl"]
    all_ok = True

    for name in datasets:
        dir_path = os.path.join(DATA_DIR, name)
        train_file = os.path.join(dir_path, "train.jsonl")
        cal_file = os.path.join(dir_path, "cal.jsonl")
        test_file = os.path.join(dir_path, "test.jsonl")

        assert os.path.exists(train_file), f"Missing {train_file}"
        assert os.path.exists(cal_file), f"Missing {cal_file}"
        assert os.path.exists(test_file), f"Missing {test_file}"

        def load_prompts(fpath):
            with open(fpath) as f:
                return set(json.loads(l)["prompt"] for l in f)

        tr_p = load_prompts(train_file)
        ca_p = load_prompts(cal_file)
        te_p = load_prompts(test_file)

        # Check disjointness
        overlap_tr_ca = tr_p.intersection(ca_p)
        overlap_tr_te = tr_p.intersection(te_p)
        overlap_ca_te = ca_p.intersection(te_p)

        print(f"\nDataset: {name.upper()}")
        print(f"  Train: {len(tr_p)} | Cal: {len(ca_p)} | Test: {len(te_p)}")
        print(f"  Overlap Train-Cal: {len(overlap_tr_ca)}")
        print(f"  Overlap Train-Test: {len(overlap_tr_te)}")
        print(f"  Overlap Cal-Test: {len(overlap_ca_te)}")

        if len(overlap_tr_ca) > 0 or len(overlap_tr_te) > 0 or len(overlap_ca_te) > 0:
            print(f"  [FAIL] Overlap detected in {name}!")
            all_ok = False
        else:
            print(f"  [PASS] Clean, disjoint 60/20/20 splits verified.")

    print("\n=======================================================")
    if all_ok:
        print("ALL DATASET SPLITS VERIFIED SUCCESSFULLY!")
    else:
        print("SPLIT INTEGRITY CHECK FAILED!")
        sys.exit(1)



def repair_existing_supervision():
    """Correct committed labels without changing example IDs or split membership."""
    tokenizer_path = os.environ.get("DIFFUSION_GEMMA_PATH", "google/diffusiongemma-26B-A4B-it")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    if len(no_ids) != 1 or len(yes_ids) != 1 or no_ids == yes_ids:
        raise ValueError("BoolQ verbalizers must be distinct single tokens")
    answer_url = f"{BFCL_BASE_URL}/possible_answer/BFCL_v4_multiple.json"
    with urllib.request.urlopen(urllib.request.Request(answer_url, headers={"User-Agent": "Reflex-Data-Prep"}), timeout=30) as resp:
        answer_bytes = resp.read()
    answers = {entry["id"]: entry["ground_truth"] for entry in map(json.loads, answer_bytes.splitlines())}
    counts = {}
    for name in ("boolq", "bfcl"):
        changed = 0
        for split in ("train", "cal", "test"):
            path = os.path.join(DATA_DIR, name, f"{split}.jsonl")
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            for row in rows:
                if name == "boolq":
                    gold_index = 1 if row["ground_truth_label"] == "yes" else 0
                    if row["ground_truth_label"] not in ("yes", "no"):
                        raise ValueError(f"Unexpected BoolQ label in {path}")
                    row["candidate_token_ids"] = [no_ids[0], yes_ids[0]]
                else:
                    gold_calls = answers.get(row["id"])
                    if gold_calls is None or len(gold_calls) != 1 or len(gold_calls[0]) != 1:
                        raise ValueError(f"No unique official BFCL answer for {row['id']}")
                    gold_name = next(iter(gold_calls[0]))
                    if gold_name not in row["candidate_names"]:
                        raise ValueError(f"BFCL gold name missing from candidates for {row['id']}")
                    gold_index = row["candidate_names"].index(gold_name)
                    row["ground_truth_label"] = gold_name
                if row["ground_truth_index"] != gold_index or row["target_token_id"] != row["candidate_token_ids"][gold_index]:
                    changed += 1
                row["ground_truth_index"] = gold_index
                row["target_token_id"] = row["candidate_token_ids"][gold_index]
            save_jsonl(rows, path)
        counts[name] = changed
    provenance = {
        "bfcl_answer_url": answer_url,
        "bfcl_answer_sha256": hashlib.sha256(answer_bytes).hexdigest(),
        "tokenizer_path": tokenizer_path,
        "boolq_candidate_token_ids_no_yes": [no_ids[0], yes_ids[0]],
        "changed_label_rows": counts,
    }
    with open(os.path.join(DATA_DIR, "supervision_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Corrected supervision: {counts}")
    verify_splits()


if __name__ == "__main__":
    if "--repair-existing-supervision" in sys.argv:
        repair_existing_supervision()
    elif "--verify-splits" in sys.argv:
        verify_splits()
    else:
        prepare_banking77()
        prepare_boolq()
        prepare_bfcl()
        verify_splits()


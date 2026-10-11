"""scripts/evaluate_baselines.py

Streaming baseline evaluator for ClearScreenAI validation phase.
Evaluates:
  1. Normalized exact name/caption match
  2. Exact match over available alias values
  3. Transparent RapidFuzz name/alias similarity with threshold tuned on validation data

Evaluation Methodology:
- Uses deterministic pair-hash splits (Train 70%, Val 15%, Test 15%).
- Tunes RapidFuzz threshold ONLY on validation partition; evaluates test ONCE.
- Audits entity-ID overlap across partitions. Explicitly notes pair-level limitation
  and makes no claim of unseen-entity generalization.
- Reports metrics for (a) all rows, and (b) analyst-judged subset when verified.
- Privacy Guarantee: Never prints or commits raw entity names or row data.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    import rapidfuzz.fuzz as rf_fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    rf_fuzz = None
    RAPIDFUZZ_AVAILABLE = False


# Known relation schemas
KNOWN_RELATION_SCHEMAS = {
    "Occupancy",
    "Succession",
    "Ownership",
    "Family",
    "Directorship",
    "Associate",
    "Membership",
    "Employment",
    "UnknownLink",
    "CourtCaseParty",
    "Passport",
    "Identification",
}


def normalize_text(text: Optional[str]) -> str:
    """Normalizes a name or alias string.
    
    Performs NFKD unicode normalization, strips combining diacritical marks,
    case folds, strips punctuation, and collapses whitespace.
    """
    if not text:
        return ""
    # NFKD decompose
    decomposed = unicodedata.normalize("NFKD", str(text))
    # Strip combining diacritical marks (e.g. umlauts, accents)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Case fold
    text_folded = stripped.casefold()
    # Strip punctuation and special symbols
    no_punct = re.sub(r"[^\w\s]", " ", text_folded)
    # Collapse multiple whitespace characters
    return re.sub(r"\s+", " ", no_punct).strip()


def extract_names_and_aliases(entity_dict: Optional[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
    """Extracts (primary_names, all_names_and_aliases) from an entity dict."""
    if not isinstance(entity_dict, dict):
        return set(), set()
        
    primary_names: Set[str] = set()
    all_names: Set[str] = set()
    
    # Caption
    caption = entity_dict.get("caption")
    if caption:
        norm = normalize_text(str(caption))
        if norm:
            primary_names.add(norm)
            all_names.add(norm)
            
    # Properties
    props = entity_dict.get("properties")
    if isinstance(props, dict):
        # Primary names
        for val in props.get("name", []):
            norm = normalize_text(str(val))
            if norm:
                primary_names.add(norm)
                all_names.add(norm)
                
        # Aliases
        for alias_key in ("alias", "weakAlias", "previousName"):
            for val in props.get(alias_key, []):
                norm = normalize_text(str(val))
                if norm:
                    all_names.add(norm)
                    
    return primary_names, all_names


def calculate_similarity_score(left_names: Set[str], right_names: Set[str]) -> float:
    """Calculates max fuzzy similarity score between two name sets in [0.0, 1.0].
    
    Uses RapidFuzz token_sort_ratio if installed; falls back to token-sorted
    SequenceMatcher ratio with warning if RapidFuzz is not yet installed.
    """
    if not left_names or not right_names:
        return 0.0
        
    best_score = 0.0
    for l_name in left_names:
        for r_name in right_names:
            if l_name == r_name:
                return 1.0
            if RAPIDFUZZ_AVAILABLE and rf_fuzz is not None:
                score = rf_fuzz.token_sort_ratio(l_name, r_name) / 100.0
            else:
                # Deterministic pure standard-library fallback
                l_sorted = " ".join(sorted(l_name.split()))
                r_sorted = " ".join(sorted(r_name.split()))
                score = difflib.SequenceMatcher(None, l_sorted, r_sorted).ratio()
            if score > best_score:
                best_score = score
                if best_score >= 1.0:
                    return 1.0
    return best_score


def normalize_binary_label(row: Dict[str, Any]) -> Optional[int]:
    """Converts row label to 1 (positive) or 0 (negative)."""
    raw = row.get("label")
    if raw is None:
        raw = row.get("decision")
    if raw is None:
        raw = row.get("judgement")
    if raw is None:
        raw = row.get("match")
        
    if raw in (1, True, "1", "true", "positive", "POSITIVE", "match"):
        return 1
    if raw in (0, False, "0", "false", "negative", "NEGATIVE", "no_match", "mismatch"):
        return 0
    return None


def get_deterministic_split(pair_key: str, train_ratio: float = 0.70, val_ratio: float = 0.15) -> str:
    """Assigns row to train, val, or test based on SHA-256 hash of pair key."""
    h_int = int(hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:8], 16)
    ratio = h_int / 0xFFFFFFFF
    if ratio < train_ratio:
        return "train"
    elif ratio < (train_ratio + val_ratio):
        return "val"
    else:
        return "test"


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Computes precision, recall, F1, accuracy, confusion matrix, and PR-AUC."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Compute PR-AUC
    pr_auc = 0.0
    if scores is not None and len(scores) == len(y_true) and (tp + fn) > 0:
        # Sort by scores descending
        paired = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
        running_tp = 0
        running_fp = 0
        total_pos = tp + fn
        prev_recall = 0.0
        
        # Calculate trapezoidal area under PR curve
        for s, y in paired:
            if y == 1:
                running_tp += 1
            else:
                running_fp += 1
            cur_prec = running_tp / (running_tp + running_fp)
            cur_rec = running_tp / total_pos
            if cur_rec > prev_recall:
                pr_auc += cur_prec * (cur_rec - prev_recall)
                prev_recall = cur_rec
    else:
        # For discrete binary predictions without scores
        pr_auc = precision * recall

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "support": {
            "positive": tp + fn,
            "negative": fp + tn,
            "total": total,
        },
    }


def tune_fuzzy_threshold(val_records: List[Tuple[float, int]]) -> Tuple[float, float]:
    """Finds threshold in [0.50, 0.95] that maximizes F1 on validation partition.
    
    Returns (best_threshold, best_val_f1).
    """
    if not val_records:
        return 0.80, 0.0
        
    best_thresh = 0.80
    best_f1 = -1.0
    
    candidate_thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    
    for thresh in candidate_thresholds:
        preds = [1 if score >= thresh else 0 for score, _ in val_records]
        trues = [label for _, label in val_records]
        m = compute_metrics(trues, preds)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh
            
    return best_thresh, best_f1


def evaluate_benchmark(
    pairs_path: str,
    max_rows: Optional[int] = None,
    relation_schemas: Optional[Set[str]] = None,
    verify_analyst_filter: bool = False,
) -> Dict[str, Any]:
    """Streams pairs benchmark and evaluates non-neural baselines."""
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    target_relations = relation_schemas or KNOWN_RELATION_SCHEMAS
    is_gz = pairs_path.endswith(".gz")
    opener = gzip.open(pairs_path, "rt", encoding="utf-8", errors="replace") if is_gz else open(
        pairs_path, "rt", encoding="utf-8", errors="replace"
    )

    # Partition entity sets for overlap audit
    entity_ids_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    seen_pair_keys: Set[bytes] = set()

    # Collectors for (all_rows, analyst_judged_rows)
    # Structure for each subset:
    # {"val_fuzzy": [(score, y)], "test_exact_name": (y_true, y_pred), ...}
    def make_collector() -> Dict[str, Any]:
        return {
            "val_fuzzy": [],
            "test_exact_name_y_true": [],
            "test_exact_name_y_pred": [],
            "test_exact_alias_y_true": [],
            "test_exact_alias_y_pred": [],
            "test_fuzzy_y_true": [],
            "test_fuzzy_scores": [],
        }

    collectors = {
        "all_rows": make_collector(),
        "analyst_judged": make_collector(),
    }

    total_rows = 0
    duplicate_rows = 0
    skipped_rows = 0

    with opener as f:
        for line_num, line in enumerate(f, start=1):
            if max_rows and total_rows >= max_rows:
                break
                
            line_str = line.strip()
            if not line_str:
                continue
                
            try:
                row = json.loads(line_str)
            except json.JSONDecodeError:
                skipped_rows += 1
                continue
                
            if not isinstance(row, dict):
                skipped_rows += 1
                continue

            y = normalize_binary_label(row)
            if y is None:
                skipped_rows += 1
                continue

            # Entities
            left = row.get("left") if isinstance(row.get("left"), dict) else row.get("entity1", {})
            right = row.get("right") if isinstance(row.get("right"), dict) else row.get("entity2", {})

            lid = str(left.get("id")) if left.get("id") else f"anon_L_{line_num}"
            rid = str(right.get("id")) if right.get("id") else f"anon_R_{line_num}"

            # Deduplicate pair ordering
            pair_key_str = f"{min(lid, rid)}::{max(lid, rid)}"
            pair_key_bytes = hashlib.blake2b(pair_key_str.encode("utf-8"), digest_size=8).digest()
            if pair_key_bytes in seen_pair_keys:
                duplicate_rows += 1
                continue
            seen_pair_keys.add(pair_key_bytes)
            total_rows += 1

            # Deterministic split
            split = get_deterministic_split(pair_key_str)
            entity_ids_by_split[split].add(lid)
            entity_ids_by_split[split].add(rid)

            # Check relation schema
            l_sch = str(left.get("schema", ""))
            r_sch = str(right.get("schema", ""))
            row_sch = str(row.get("schema", ""))
            is_relation = (
                l_sch in target_relations or r_sch in target_relations or row_sch in target_relations
            )
            is_analyst_judged = not is_relation

            # Feature extraction (excluding IDs and labels)
            l_prim, l_all = extract_names_and_aliases(left)
            r_prim, r_all = extract_names_and_aliases(right)

            # Baseline 1: Exact Name/Caption Match
            exact_name_match = 1 if bool(l_prim and r_prim and (l_prim & r_prim)) else 0

            # Baseline 2: Exact Name/Alias Match
            exact_alias_match = 1 if bool(l_all and r_all and (l_all & r_all)) else 0

            # Baseline 3: RapidFuzz Similarity
            fuzzy_score = calculate_similarity_score(l_all, r_all)

            # Populate collectors
            subsets_to_feed = ["all_rows"]
            if is_analyst_judged:
                subsets_to_feed.append("analyst_judged")

            for sub in subsets_to_feed:
                c = collectors[sub]
                if split == "val":
                    c["val_fuzzy"].append((fuzzy_score, y))
                elif split == "test":
                    c["test_exact_name_y_true"].append(y)
                    c["test_exact_name_y_pred"].append(exact_name_match)
                    c["test_exact_alias_y_true"].append(y)
                    c["test_exact_alias_y_pred"].append(exact_alias_match)
                    c["test_fuzzy_y_true"].append(y)
                    c["test_fuzzy_scores"].append(fuzzy_score)

    # Partition Overlap Audit
    train_ids = entity_ids_by_split["train"]
    val_ids = entity_ids_by_split["val"]
    test_ids = entity_ids_by_split["test"]

    overlap_train_val = len(train_ids & val_ids)
    overlap_train_test = len(train_ids & test_ids)
    overlap_val_test = len(val_ids & test_ids)

    partition_audit = {
        "split_strategy": "deterministic_pair_hash_split",
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "evaluation_level": "pair_level",
        "unseen_entity_generalization_claimed": False,
        "limitation_disclosure": (
            "OpenSanctions Pairs flat snapshot contains repeated entities across pairs. "
            "Pair-level hash splitting results in entity overlap between partitions. "
            "Results measure pair-level matching performance and MUST NOT be claimed as "
            "unseen-entity generalization."
        ),
        "unique_entity_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "entity_id_overlap": {
            "train_val_overlap_count": overlap_train_val,
            "train_test_overlap_count": overlap_train_test,
            "val_test_overlap_count": overlap_val_test,
        },
    }

    # Evaluate baselines for a given collector
    def compute_subset_results(c: Dict[str, Any]) -> Dict[str, Any]:
        # Tune threshold on val
        tuned_threshold, val_f1 = tune_fuzzy_threshold(c["val_fuzzy"])
        
        # Test evaluation
        # Baseline A: Exact Name/Caption
        m_exact_name = compute_metrics(
            c["test_exact_name_y_true"],
            c["test_exact_name_y_pred"],
        )
        # Baseline B: Exact Name/Alias
        m_exact_alias = compute_metrics(
            c["test_exact_alias_y_true"],
            c["test_exact_alias_y_pred"],
        )
        # Baseline C: RapidFuzz Name/Alias
        fuzzy_preds = [1 if s >= tuned_threshold else 0 for s in c["test_fuzzy_scores"]]
        m_fuzzy = compute_metrics(
            c["test_fuzzy_y_true"],
            fuzzy_preds,
            scores=c["test_fuzzy_scores"],
        )

        return {
            "validation_tuning": {
                "val_samples_count": len(c["val_fuzzy"]),
                "selected_fuzzy_threshold": tuned_threshold,
                "validation_f1_at_threshold": round(val_f1, 4),
            },
            "test_evaluation": {
                "test_samples_count": len(c["test_exact_name_y_true"]),
                "baseline_1_exact_name_caption": m_exact_name,
                "baseline_2_exact_name_alias": m_exact_alias,
                "baseline_3_rapidfuzz_similarity": {
                    "scoring_method": "token_sort_ratio",
                    "rapidfuzz_library_active": RAPIDFUZZ_AVAILABLE,
                    "applied_threshold": tuned_threshold,
                    "metrics": m_fuzzy,
                },
            },
        }

    results_all = compute_subset_results(collectors["all_rows"])

    if verify_analyst_filter:
        results_analyst = compute_subset_results(collectors["analyst_judged"])
        analyst_status = "evaluated_verified_subset"
    else:
        results_analyst = None
        analyst_status = "not determined — review required"

    summary: Dict[str, Any] = {
        "metadata": {
            "benchmark_file": pairs_path,
            "evaluation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "max_rows_limit": max_rows,
            "total_pairs_processed": total_rows,
            "duplicate_pairs_discarded": duplicate_rows,
            "skipped_rows": skipped_rows,
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
        },
        "partition_audit": partition_audit,
        "results_all_rows": results_all,
        "analyst_judged_subset": {
            "status": analyst_status,
            "results": results_analyst,
            "note": (
                "Full-corpus scores include auto-propagated positive relation pairs. "
                "Do NOT present all_rows metrics alone as proof of matcher quality."
            ),
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming baseline evaluator for ClearScreenAI validation."
    )
    parser.add_argument(
        "--pairs",
        default="data/raw/pairs.json.gz",
        help="Path to labeled pairs JSONL/GZ benchmark (default: data/raw/pairs.json.gz)",
    )
    parser.add_argument(
        "--out",
        default="outputs/baseline_summary.json",
        help="Path to output baseline metrics JSON (default: outputs/baseline_summary.json)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum rows to parse (for test fixtures/smoke tests)",
    )
    parser.add_argument(
        "--relation-schemas",
        default=None,
        help="Comma-separated custom relation schemas to filter",
    )
    parser.add_argument(
        "--verify-analyst-filter",
        action="store_true",
        help="Explicitly mark relation-schema filter as verified and evaluate analyst subset",
    )

    args = parser.parse_args()

    rel_schemas = None
    if args.relation_schemas:
        rel_schemas = set(s.strip() for s in args.relation_schemas.split(",") if s.strip())

    print(f"[*] Starting baseline evaluation on: {args.pairs}")
    if not os.path.exists(args.pairs):
        print(f"[!] File not found: {args.pairs}")
        sys.exit(1)

    summary = evaluate_benchmark(
        pairs_path=args.pairs,
        max_rows=args.max_rows,
        relation_schemas=rel_schemas,
        verify_analyst_filter=args.verify_analyst_filter,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Evaluation completed. Summary written to: {args.out}")
    print("\n--- BASELINE RESULTS (TEST PARTITION) ---")
    test_eval = summary["results_all_rows"]["test_evaluation"]
    print(f"Evaluated Test Rows: {test_eval['test_samples_count']:,}")
    print(f"Baseline 1 (Exact Name/Caption): F1={test_eval['baseline_1_exact_name_caption']['f1']}, Precision={test_eval['baseline_1_exact_name_caption']['precision']}, Recall={test_eval['baseline_1_exact_name_caption']['recall']}")
    print(f"Baseline 2 (Exact Name/Alias):   F1={test_eval['baseline_2_exact_name_alias']['f1']}, Precision={test_eval['baseline_2_exact_name_alias']['precision']}, Recall={test_eval['baseline_2_exact_name_alias']['recall']}")
    fuzz_res = test_eval["baseline_3_rapidfuzz_similarity"]
    print(f"Baseline 3 (Fuzzy Match, Thresh={fuzz_res['applied_threshold']}): F1={fuzz_res['metrics']['f1']}, PR-AUC={fuzz_res['metrics']['pr_auc']}")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()

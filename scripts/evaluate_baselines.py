"""scripts/evaluate_baselines.py

Streaming baseline evaluator for ClearScreenAI validation phase.
Evaluates non-neural baselines with bounded memory:
  1. Normalized exact name/caption match
  2. Exact match over available alias values
  3. Transparent RapidFuzz name/alias similarity with threshold tuned on validation data

Methodology & Standards:
- Bounded RAM: uses temporary disk-backed SQLite tables for pair deduplication,
  entity partition overlap audits, and score tracking. Temporary files are always
  cleaned up on exit or error.
- Deterministic splits: Train 70%, Val 15%, Test 15% assigned by pair-key hash.
- Strict validation tuning: fuzzy similarity threshold is tuned ONLY on the validation partition;
  test set is evaluated exactly once at the fixed tuned threshold.
- Valid metrics: continuous ranking evaluated with standard Average Precision (AP);
  discrete binary predictions return null/None for PR-AUC/AP rather than arbitrary heuristics.
- Entity-ID overlap audit: explicitly reports overlap across splits; notes pair-level limitation
  and disclaims unseen-entity generalization.
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
import sqlite3
import sys
import tempfile
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    import rapidfuzz.fuzz as rf_fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    rf_fuzz = None
    RAPIDFUZZ_AVAILABLE = False


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
                l_sorted = " ".join(sorted(l_name.split()))
                r_sorted = " ".join(sorted(r_name.split()))
                score = difflib.SequenceMatcher(None, l_sorted, r_sorted).ratio()
            if score > best_score:
                best_score = score
                if best_score >= 1.0:
                    return 1.0
    return best_score


def normalize_binary_label(row: Dict[str, Any]) -> Optional[int]:
    """Converts row label/judgement to 1 (positive) or 0 (negative).
    
    Returns None for missing, malformed, or unrecognized labels.
    """
    raw = row.get("judgement")
    if raw is None:
        raw = row.get("label")
    if raw is None:
        raw = row.get("decision")
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


def compute_average_precision(y_true: List[int], scores: List[float]) -> Optional[float]:
    """Computes standard Average Precision (AP) for continuous prediction scores.
    
    AP = sum_n (R_n - R_{n-1}) * P_n across distinct score thresholds.
    Returns None if there are no positive samples in y_true.
    """
    if not y_true or not scores or len(y_true) != len(scores):
        return None
        
    total_positives = sum(y_true)
    if total_positives == 0:
        return None

    # Group true labels by distinct descending thresholds
    # to avoid threshold tie artifacts
    threshold_map: Dict[float, List[int]] = {}
    for s, y in zip(scores, y_true):
        threshold_map.setdefault(s, []).append(y)

    distinct_thresholds = sorted(threshold_map.keys(), reverse=True)
    
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    for thresh in distinct_thresholds:
        labels_at_thresh = threshold_map[thresh]
        pos_at_thresh = sum(labels_at_thresh)
        neg_at_thresh = len(labels_at_thresh) - pos_at_thresh
        
        tp += pos_at_thresh
        fp += neg_at_thresh
        
        precision = tp / (tp + fp)
        recall = tp / total_positives
        
        # Increment AP by area of this recall segment
        ap += precision * (recall - prev_recall)
        prev_recall = recall

    return round(ap, 4)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Computes precision, recall, F1, accuracy, confusion matrix, and average precision.
    
    For continuous scores, computes standard Average Precision (AP).
    For discrete binary predictions without continuous scores, returns None for average_precision.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Valid continuous ranking metric
    if scores is not None and len(scores) == len(y_true):
        avg_precision = compute_average_precision(y_true, scores)
    else:
        # Strictly return None rather than precision * recall heuristic
        avg_precision = None

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "average_precision": avg_precision,
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
) -> Dict[str, Any]:
    """Streams pairs benchmark and evaluates non-neural baselines with bounded memory."""
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    is_gz = pairs_path.endswith(".gz")
    opener = gzip.open(pairs_path, "rt", encoding="utf-8", errors="replace") if is_gz else open(
        pairs_path, "rt", encoding="utf-8", errors="replace"
    )

    # Temporary disk-backed SQLite database for deduplication, overlap audit, and evaluation records
    temp_dir = tempfile.mkdtemp(prefix="clearscreen_eval_")
    db_path = os.path.join(temp_dir, "eval_temp.db")
    db_conn: Optional[sqlite3.Connection] = None

    total_rows = 0
    duplicate_rows = 0
    malformed_rows = 0
    unlabeled_or_unknown_rows = 0

    try:
        db_conn = sqlite3.connect(db_path)
        cur = db_conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = OFF")
        
        # Tables for indexing
        cur.execute("CREATE TABLE seen_pairs (pair_hash BLOB PRIMARY KEY)")
        cur.execute("CREATE TABLE entity_partitions (entity_hash BLOB, split TEXT, PRIMARY KEY(entity_hash, split))")
        cur.execute(
            """CREATE TABLE eval_records (
                split TEXT,
                y INTEGER,
                exact_name INTEGER,
                exact_alias INTEGER,
                fuzzy_score REAL
            )"""
        )
        db_conn.commit()

        batch_pairs: List[Tuple[bytes,]] = []
        batch_entities: List[Tuple[bytes, str]] = []
        batch_eval: List[Tuple[str, int, int, int, float]] = []
        batch_size = 5000

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
                    malformed_rows += 1
                    continue
                    
                if not isinstance(row, dict):
                    malformed_rows += 1
                    continue

                y = normalize_binary_label(row)
                if y is None:
                    unlabeled_or_unknown_rows += 1
                    continue

                # Entities
                left = row.get("left") if isinstance(row.get("left"), dict) else row.get("entity1", {})
                right = row.get("right") if isinstance(row.get("right"), dict) else row.get("entity2", {})

                lid = str(left.get("id")) if left.get("id") else f"anon_L_{line_num}"
                rid = str(right.get("id")) if right.get("id") else f"anon_R_{line_num}"

                # Pair deduplication key
                pair_key_str = f"{min(lid, rid)}::{max(lid, rid)}"
                pair_hash = hashlib.blake2b(pair_key_str.encode("utf-8"), digest_size=16).digest()
                
                # Check for duplicate pair in database
                cur.execute("INSERT OR IGNORE INTO seen_pairs VALUES (?)", (pair_hash,))
                if cur.rowcount == 0:
                    duplicate_rows += 1
                    continue

                total_rows += 1

                # Deterministic split assignment
                split = get_deterministic_split(pair_key_str)
                
                # Record entity IDs by partition (hashed to avoid storing raw IDs)
                lid_hash = hashlib.sha256(lid.encode("utf-8")).digest()[:16]
                rid_hash = hashlib.sha256(rid.encode("utf-8")).digest()[:16]
                batch_entities.append((lid_hash, split))
                batch_entities.append((rid_hash, split))

                # Feature extraction
                l_prim, l_all = extract_names_and_aliases(left)
                r_prim, r_all = extract_names_and_aliases(right)

                exact_name = 1 if bool(l_prim and r_prim and (l_prim & r_prim)) else 0
                exact_alias = 1 if bool(l_all and r_all and (l_all & r_all)) else 0
                fuzzy_score = calculate_similarity_score(l_all, r_all)

                batch_eval.append((split, y, exact_name, exact_alias, fuzzy_score))

                if len(batch_eval) >= batch_size:
                    cur.executemany("INSERT OR IGNORE INTO entity_partitions VALUES (?, ?)", batch_entities)
                    cur.executemany("INSERT INTO eval_records VALUES (?, ?, ?, ?, ?)", batch_eval)
                    batch_entities.clear()
                    batch_eval.clear()

        if batch_eval:
            cur.executemany("INSERT OR IGNORE INTO entity_partitions VALUES (?, ?)", batch_entities)
            cur.executemany("INSERT INTO eval_records VALUES (?, ?, ?, ?, ?)", batch_eval)
            batch_entities.clear()
            batch_eval.clear()
            
        db_conn.commit()

        # Audit entity partition overlap via SQL
        cur.execute("SELECT count(DISTINCT entity_hash) FROM entity_partitions WHERE split = 'train'")
        unique_train = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT entity_hash) FROM entity_partitions WHERE split = 'val'")
        unique_val = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT entity_hash) FROM entity_partitions WHERE split = 'test'")
        unique_test = cur.fetchone()[0]

        cur.execute(
            """SELECT count(DISTINCT t1.entity_hash)
               FROM entity_partitions t1
               JOIN entity_partitions t2 ON t1.entity_hash = t2.entity_hash
               WHERE t1.split = 'train' AND t2.split = 'val'"""
        )
        overlap_train_val = cur.fetchone()[0]

        cur.execute(
            """SELECT count(DISTINCT t1.entity_hash)
               FROM entity_partitions t1
               JOIN entity_partitions t2 ON t1.entity_hash = t2.entity_hash
               WHERE t1.split = 'train' AND t2.split = 'test'"""
        )
        overlap_train_test = cur.fetchone()[0]

        cur.execute(
            """SELECT count(DISTINCT t1.entity_hash)
               FROM entity_partitions t1
               JOIN entity_partitions t2 ON t1.entity_hash = t2.entity_hash
               WHERE t1.split = 'val' AND t2.split = 'test'"""
        )
        overlap_val_test = cur.fetchone()[0]

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
                "train": unique_train,
                "val": unique_val,
                "test": unique_test,
            },
            "entity_id_overlap": {
                "train_val_overlap_count": overlap_train_val,
                "train_test_overlap_count": overlap_train_test,
                "val_test_overlap_count": overlap_val_test,
            },
        }

        # Retrieve validation records for threshold tuning
        cur.execute("SELECT fuzzy_score, y FROM eval_records WHERE split = 'val'")
        val_fuzzy_records = [(row[0], row[1]) for row in cur.fetchall()]
        tuned_threshold, val_f1 = tune_fuzzy_threshold(val_fuzzy_records)

        # Retrieve test records for evaluation
        cur.execute("SELECT y, exact_name, exact_alias, fuzzy_score FROM eval_records WHERE split = 'test'")
        test_rows = cur.fetchall()
        
        test_y = [r[0] for r in test_rows]
        test_exact_name = [r[1] for r in test_rows]
        test_exact_alias = [r[2] for r in test_rows]
        test_fuzzy_scores = [r[3] for r in test_rows]
        test_fuzzy_preds = [1 if s >= tuned_threshold else 0 for s in test_fuzzy_scores]

        # Compute metrics
        m_exact_name = compute_metrics(test_y, test_exact_name)
        m_exact_alias = compute_metrics(test_y, test_exact_alias)
        m_fuzzy = compute_metrics(test_y, test_fuzzy_preds, scores=test_fuzzy_scores)

        results = {
            "validation_tuning": {
                "val_samples_count": len(val_fuzzy_records),
                "selected_fuzzy_threshold": tuned_threshold,
                "validation_f1_at_threshold": round(val_f1, 4),
            },
            "test_evaluation": {
                "test_samples_count": len(test_y),
                "baseline_1_exact_name_caption": {
                    **m_exact_name,
                    "note": "Discrete binary predictions; average_precision is not applicable.",
                },
                "baseline_2_exact_name_alias": {
                    **m_exact_alias,
                    "note": "Discrete binary predictions; average_precision is not applicable.",
                },
                "baseline_3_rapidfuzz_similarity": {
                    "scoring_method": "token_sort_ratio",
                    "rapidfuzz_library_active": RAPIDFUZZ_AVAILABLE,
                    "applied_threshold": tuned_threshold,
                    "metrics": m_fuzzy,
                },
            },
        }

    finally:
        # Guarantee cleanup of temporary SQLite database
        if db_conn:
            try:
                db_conn.close()
            except Exception:
                pass
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass

    summary: Dict[str, Any] = {
        "metadata": {
            "benchmark_file": pairs_path,
            "evaluation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "max_rows_limit": max_rows,
            "total_pairs_processed": total_rows,
            "duplicate_pairs_discarded": duplicate_rows,
            "malformed_rows_skipped": malformed_rows,
            "unlabeled_or_unknown_rows_skipped": unlabeled_or_unknown_rows,
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
            "memory_management": "streaming_jsonl_with_disk_backed_sqlite_evaluation",
        },
        "partition_audit": partition_audit,
        "results": results,
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

    args = parser.parse_args()

    print(f"[*] Starting baseline evaluation on: {args.pairs}")
    if not os.path.exists(args.pairs):
        print(f"[!] File not found: {args.pairs}")
        sys.exit(1)

    summary = evaluate_benchmark(
        pairs_path=args.pairs,
        max_rows=args.max_rows,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Evaluation completed. Summary written to: {args.out}")
    print("\n--- BASELINE RESULTS (TEST PARTITION) ---")
    test_eval = summary["results"]["test_evaluation"]
    print(f"Evaluated Test Rows: {test_eval['test_samples_count']:,}")
    print(
        f"Baseline 1 (Exact Name/Caption): F1={test_eval['baseline_1_exact_name_caption']['f1']}, "
        f"Precision={test_eval['baseline_1_exact_name_caption']['precision']}, "
        f"Recall={test_eval['baseline_1_exact_name_caption']['recall']}"
    )
    print(
        f"Baseline 2 (Exact Name/Alias):   F1={test_eval['baseline_2_exact_name_alias']['f1']}, "
        f"Precision={test_eval['baseline_2_exact_name_alias']['precision']}, "
        f"Recall={test_eval['baseline_2_exact_name_alias']['recall']}"
    )
    fuzz_res = test_eval["baseline_3_rapidfuzz_similarity"]
    print(
        f"Baseline 3 (Fuzzy Match, Thresh={fuzz_res['applied_threshold']}): "
        f"F1={fuzz_res['metrics']['f1']}, "
        f"Average Precision (AP)={fuzz_res['metrics']['average_precision']}"
    )
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()

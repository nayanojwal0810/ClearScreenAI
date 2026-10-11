"""scripts/evaluate_baselines.py

Streaming baseline evaluator for ClearScreenAI validation phase.
Evaluates non-neural baselines with bounded memory and cursor-based processing:
  1. Normalized exact name/caption match
  2. Exact match over available alias values
  3. Transparent RapidFuzz name/alias similarity with validation-tuned threshold

Evaluation Scopes:
  - Primary Scope: Observations where both entities have official matchable schemas
    (Person, Company, Organization, LegalEntity, Security, CryptoWallet, Vessel, Airplane, PublicBody),
    excluding conflicting-label pair groups.
  - Secondary Scope: The broader benchmark (including contextual schemas such as Occupancy,
    Succession, Position, Ownership, Family), with the same conflict-exclusion policy.

Duplicate & Conflict Policy:
  - Deterministic splits assign all versions of an unordered pair key to the same partition.
  - Different entity payload versions of a pair are preserved.
  - Only exact duplicate observations (identical pair key, canonical payload hash, and label)
    are deduplicated.
  - Pair groups with conflicting positive/negative judgements are excluded from clean metrics,
    retained in diagnostic reporting, and never resolved arbitrarily.
  - Zero bulk fetching (no fetchall): threshold tuning, confusion matrices, and tie-aware
    Average Precision are computed via cursor/query-based SQLite operations.
  - Temporary disk-backed SQLite files are guaranteed removed on exit or error.
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


# Official OpenSanctions matchable schema types (https://www.opensanctions.org/reference/)
MATCHABLE_SCHEMAS: Set[str] = {
    "Person",
    "Company",
    "Organization",
    "LegalEntity",
    "Security",
    "CryptoWallet",
    "Vessel",
    "Airplane",
    "PublicBody",
}

# Known contextual/relational schemas
KNOWN_CONTEXTUAL_SCHEMAS: Set[str] = {
    "Occupancy",
    "Succession",
    "Position",
    "Directorship",
    "Ownership",
    "Family",
    "Associate",
    "Membership",
    "Employment",
    "Representation",
    "Documentation",
    "CourtCaseParty",
    "UnknownLink",
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
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    text_folded = stripped.casefold()
    no_punct = re.sub(r"[^\w\s]", " ", text_folded)
    return re.sub(r"\s+", " ", no_punct).strip()


def extract_names_and_aliases(entity_dict: Optional[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
    """Extracts (primary_names, all_names_and_aliases) from an entity dict."""
    if not isinstance(entity_dict, dict):
        return set(), set()
        
    primary_names: Set[str] = set()
    all_names: Set[str] = set()
    
    caption = entity_dict.get("caption")
    if caption:
        norm = normalize_text(str(caption))
        if norm:
            primary_names.add(norm)
            all_names.add(norm)
            
    props = entity_dict.get("properties")
    if isinstance(props, dict):
        for val in props.get("name", []):
            norm = normalize_text(str(val))
            if norm:
                primary_names.add(norm)
                all_names.add(norm)
                
        for alias_key in ("alias", "weakAlias", "previousName"):
            for val in props.get(alias_key, []):
                norm = normalize_text(str(val))
                if norm:
                    all_names.add(norm)
                    
    return primary_names, all_names


def hash_entity_payload(entity: Optional[Dict[str, Any]]) -> bytes:
    """Computes a 16-byte cryptographic hash of a canonicalized entity payload."""
    if not isinstance(entity, dict):
        return b""
    canon = json.dumps(entity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).digest()[:16]


def canonical_pair_payload_hash(
    lid: str,
    rid: str,
    left_entity: Optional[Dict[str, Any]],
    right_entity: Optional[Dict[str, Any]],
) -> bytes:
    """Computes a 16-byte cryptographic hash of an unordered pair payload."""
    h_left = hash_entity_payload(left_entity)
    h_right = hash_entity_payload(right_entity)
    if lid <= rid:
        combined = h_left + h_right
    else:
        combined = h_right + h_left
    return hashlib.sha256(combined).digest()[:16]


def calculate_similarity_score(left_names: Set[str], right_names: Set[str]) -> float:
    """Calculates max fuzzy similarity score between two name sets in [0.0, 1.0]."""
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
    """Converts row label/judgement to 1 (positive) or 0 (negative)."""
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
    """Assigns observation to train, val, or test based on SHA-256 hash of pair key.
    
    Guarantees that all observations sharing the same unordered pair key reside
    in the exact same partition.
    """
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
    
    AP = sum_n (R_n - R_{n-1}) * P_n across distinct score thresholds, correctly
    grouping tied scores. Returns None if there are no positive samples in y_true.
    """
    if not y_true or not scores or len(y_true) != len(scores):
        return None
        
    total_positives = sum(y_true)
    if total_positives == 0:
        return None

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
        
        ap += precision * (recall - prev_recall)
        prev_recall = recall

    return round(ap, 4)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Computes precision, recall, F1, accuracy, confusion matrix, and average precision."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    if scores is not None and len(scores) == len(y_true):
        avg_precision = compute_average_precision(y_true, scores)
    else:
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
    """Finds threshold in [0.50, 0.95] that maximizes F1 on validation partition."""
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


def tune_fuzzy_threshold_sql(
    cur: sqlite3.Cursor,
    is_matchable_filter: Optional[int],
) -> Tuple[float, float, int]:
    """Finds threshold in [0.50, 0.95] maximizing F1 on clean validation partition.
    
    Uses direct SQL queries without bulk data fetching.
    Returns (best_threshold, best_val_f1, total_val_clean_samples).
    """
    candidate_thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    best_thresh = 0.80
    best_f1 = -1.0
    
    scope_clause = ""
    params_base: List[Any] = []
    if is_matchable_filter is not None:
        scope_clause = "AND is_matchable = ?"
        params_base.append(is_matchable_filter)
        
    # Get total validation clean count
    count_sql = f"""
        SELECT count(*)
        FROM eval_records
        WHERE split = 'val'
          {scope_clause}
          AND pair_key_hash NOT IN (SELECT pair_key_hash FROM conflicting_pairs)
    """
    cur.execute(count_sql, params_base)
    total_val_samples = cur.fetchone()[0]
    
    if total_val_samples == 0:
        return 0.80, 0.0, 0

    query = f"""
        SELECT 
            sum(CASE WHEN fuzzy_score >= ? AND y = 1 THEN 1 ELSE 0 END),
            sum(CASE WHEN fuzzy_score >= ? AND y = 0 THEN 1 ELSE 0 END),
            sum(CASE WHEN fuzzy_score < ? AND y = 1 THEN 1 ELSE 0 END)
        FROM eval_records
        WHERE split = 'val'
          {scope_clause}
          AND pair_key_hash NOT IN (SELECT pair_key_hash FROM conflicting_pairs)
    """

    for thresh in candidate_thresholds:
        query_params = [thresh, thresh, thresh] + params_base
        cur.execute(query, query_params)
        res = cur.fetchone()
        tp = res[0] or 0
        fp = res[1] or 0
        fn = res[2] or 0
        
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, round(best_f1, 4), total_val_samples


def compute_confusion_matrix_sql(
    cur: sqlite3.Cursor,
    pred_field: str,
    threshold: Optional[float],
    is_matchable_filter: Optional[int],
) -> Dict[str, Any]:
    """Computes confusion matrix and classification metrics via direct SQL aggregation."""
    scope_clause = ""
    params: List[Any] = []
    
    if threshold is not None:
        pred_expr_1 = f"CASE WHEN {pred_field} >= ? THEN 1 ELSE 0 END"
        pred_expr_0 = f"CASE WHEN {pred_field} < ? THEN 1 ELSE 0 END"
        params.extend([threshold, threshold, threshold, threshold])
    else:
        pred_expr_1 = f"CASE WHEN {pred_field} = 1 THEN 1 ELSE 0 END"
        pred_expr_0 = f"CASE WHEN {pred_field} = 0 THEN 1 ELSE 0 END"

    if is_matchable_filter is not None:
        scope_clause = "AND is_matchable = ?"
        params.append(is_matchable_filter)

    query = f"""
        SELECT 
            sum(CASE WHEN {pred_expr_1} AND y = 1 THEN 1 ELSE 0 END) as tp,
            sum(CASE WHEN {pred_expr_1} AND y = 0 THEN 1 ELSE 0 END) as fp,
            sum(CASE WHEN {pred_expr_0} AND y = 1 THEN 1 ELSE 0 END) as fn,
            sum(CASE WHEN {pred_expr_0} AND y = 0 THEN 1 ELSE 0 END) as tn
        FROM eval_records
        WHERE split = 'test'
          {scope_clause}
          AND pair_key_hash NOT IN (SELECT pair_key_hash FROM conflicting_pairs)
    """
    cur.execute(query, params)
    res = cur.fetchone()
    tp = res[0] or 0
    fp = res[1] or 0
    fn = res[2] or 0
    tn = res[3] or 0
    
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "average_precision": None,
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


def compute_average_precision_cursor_sql(
    cur: sqlite3.Cursor,
    is_matchable_filter: Optional[int],
    batch_size: int = 1000,
) -> Optional[float]:
    """Computes standard Average Precision (AP) for continuous scores using cursor-based streaming.
    
    Correctly handles tied scores by grouping identical fuzzy_score thresholds in SQL.
    Processes rows in bounded batches without fetchall().
    """
    scope_clause = ""
    params_base: List[Any] = []
    if is_matchable_filter is not None:
        scope_clause = "AND is_matchable = ?"
        params_base.append(is_matchable_filter)

    # 1. Get total positives on clean test set
    cur.execute(
        f"""SELECT sum(y)
            FROM eval_records
            WHERE split = 'test'
              {scope_clause}
              AND pair_key_hash NOT IN (SELECT pair_key_hash FROM conflicting_pairs)""",
        params_base,
    )
    res = cur.fetchone()
    total_positives = res[0] if res and res[0] else 0
    if total_positives == 0:
        return None

    # 2. Stream distinct descending thresholds with grouped counts
    query = f"""
        SELECT 
            fuzzy_score,
            sum(CASE WHEN y = 1 THEN 1 ELSE 0 END) as pos_cnt,
            sum(CASE WHEN y = 0 THEN 1 ELSE 0 END) as neg_cnt
        FROM eval_records
        WHERE split = 'test'
          {scope_clause}
          AND pair_key_hash NOT IN (SELECT pair_key_hash FROM conflicting_pairs)
        GROUP BY fuzzy_score
        ORDER BY fuzzy_score DESC
    """
    cur.execute(query, params_base)

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for _, pos_cnt, neg_cnt in rows:
            tp += pos_cnt
            fp += neg_cnt
            precision = tp / (tp + fp)
            recall = tp / total_positives
            ap += precision * (recall - prev_recall)
            prev_recall = recall

    return round(ap, 4)


def evaluate_scope_sql(
    cur: sqlite3.Cursor,
    is_matchable_filter: Optional[int],
    scope_name: str,
) -> Dict[str, Any]:
    """Evaluates all baselines for a specific scope using bounded SQLite queries."""
    # 1. Validation tuning
    tuned_threshold, val_f1, val_samples_count = tune_fuzzy_threshold_sql(cur, is_matchable_filter)

    # 2. Baseline 1: Exact Name/Caption
    m_exact_name = compute_confusion_matrix_sql(
        cur, pred_field="exact_name", threshold=None, is_matchable_filter=is_matchable_filter
    )
    
    # 3. Baseline 2: Exact Name/Alias
    m_exact_alias = compute_confusion_matrix_sql(
        cur, pred_field="exact_alias", threshold=None, is_matchable_filter=is_matchable_filter
    )

    # 4. Baseline 3: RapidFuzz Similarity
    m_fuzzy = compute_confusion_matrix_sql(
        cur, pred_field="fuzzy_score", threshold=tuned_threshold, is_matchable_filter=is_matchable_filter
    )
    ap = compute_average_precision_cursor_sql(cur, is_matchable_filter=is_matchable_filter)
    m_fuzzy["average_precision"] = ap

    test_samples_count = m_exact_name["support"]["total"]

    return {
        "scope_name": scope_name,
        "validation_tuning": {
            "val_clean_samples_count": val_samples_count,
            "selected_fuzzy_threshold": tuned_threshold,
            "validation_f1_at_threshold": val_f1,
        },
        "test_evaluation": {
            "test_clean_samples_count": test_samples_count,
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


def evaluate_benchmark(
    pairs_path: str,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Streams pairs benchmark and evaluates baselines with bounded memory and two scopes."""
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    is_gz = pairs_path.endswith(".gz")
    opener = gzip.open(pairs_path, "rt", encoding="utf-8", errors="replace") if is_gz else open(
        pairs_path, "rt", encoding="utf-8", errors="replace"
    )

    temp_dir = tempfile.mkdtemp(prefix="clearscreen_eval_")
    db_path = os.path.join(temp_dir, "eval_temp.db")
    db_conn: Optional[sqlite3.Connection] = None

    total_source_rows = 0
    exact_duplicates_discarded = 0
    malformed_rows = 0
    unlabeled_or_unknown_rows = 0

    schema_distribution: Dict[str, int] = {}
    uncertain_schemas: Dict[str, int] = {}

    try:
        db_conn = sqlite3.connect(db_path)
        cur = db_conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = OFF")
        
        # Deduplication table for exact repeated observations:
        # (pair_key_hash, payload_hash, label)
        cur.execute(
            """CREATE TABLE seen_observations (
                pair_key_hash BLOB,
                payload_hash BLOB,
                label INTEGER,
                PRIMARY KEY (pair_key_hash, payload_hash, label)
            )"""
        )
        
        cur.execute(
            """CREATE TABLE entity_partitions (
                entity_hash BLOB,
                split TEXT,
                PRIMARY KEY(entity_hash, split)
            )"""
        )
        
        cur.execute(
            """CREATE TABLE eval_records (
                pair_key_hash BLOB,
                split TEXT,
                is_matchable INTEGER,
                y INTEGER,
                exact_name INTEGER,
                exact_alias INTEGER,
                fuzzy_score REAL
            )"""
        )
        db_conn.commit()

        batch_entities: List[Tuple[bytes, str]] = []
        batch_eval: List[Tuple[bytes, str, int, int, int, int, float]] = []
        batch_size = 5000

        with opener as f:
            for line_num, line in enumerate(f, start=1):
                if max_rows and total_source_rows >= max_rows:
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

                total_source_rows += 1

                # Entities
                left = row.get("left") if isinstance(row.get("left"), dict) else row.get("entity1", {})
                right = row.get("right") if isinstance(row.get("right"), dict) else row.get("entity2", {})

                lid = str(left.get("id")) if left.get("id") else f"anon_L_{line_num}"
                rid = str(right.get("id")) if right.get("id") else f"anon_R_{line_num}"

                l_sch = str(left.get("schema", "Unknown"))
                r_sch = str(right.get("schema", "Unknown"))

                # Track schema distributions
                schema_distribution[l_sch] = schema_distribution.get(l_sch, 0) + 1
                schema_distribution[r_sch] = schema_distribution.get(r_sch, 0) + 1

                for s in (l_sch, r_sch):
                    if s not in MATCHABLE_SCHEMAS and s not in KNOWN_CONTEXTUAL_SCHEMAS:
                        uncertain_schemas[s] = uncertain_schemas.get(s, 0) + 1

                # Determine matchable scope: both entities matchable
                is_matchable = 1 if (l_sch in MATCHABLE_SCHEMAS and r_sch in MATCHABLE_SCHEMAS) else 0

                # Pair keys and payload hashes
                pair_key_str = f"{min(lid, rid)}::{max(lid, rid)}"
                pair_key_bytes = pair_key_str.encode("utf-8")
                pair_key_h = hashlib.blake2b(pair_key_bytes, digest_size=16).digest()
                payload_h = canonical_pair_payload_hash(lid, rid, left, right)

                # Deduplicate ONLY exact repeated observations (pair_key, payload_hash, label)
                cur.execute(
                    "INSERT OR IGNORE INTO seen_observations VALUES (?, ?, ?)",
                    (pair_key_h, payload_h, y),
                )
                if cur.rowcount == 0:
                    exact_duplicates_discarded += 1
                    continue

                # Deterministic split assignment using unordered pair key
                split = get_deterministic_split(pair_key_str)
                
                # Record entity IDs by partition
                lid_h = hashlib.sha256(lid.encode("utf-8")).digest()[:16]
                rid_h = hashlib.sha256(rid.encode("utf-8")).digest()[:16]
                batch_entities.append((lid_h, split))
                batch_entities.append((rid_h, split))

                # Feature extraction
                l_prim, l_all = extract_names_and_aliases(left)
                r_prim, r_all = extract_names_and_aliases(right)

                exact_name = 1 if bool(l_prim and r_prim and (l_prim & r_prim)) else 0
                exact_alias = 1 if bool(l_all and r_all and (l_all & r_all)) else 0
                fuzzy_score = calculate_similarity_score(l_all, r_all)

                batch_eval.append((pair_key_h, split, is_matchable, y, exact_name, exact_alias, fuzzy_score))

                if len(batch_eval) >= batch_size:
                    cur.executemany("INSERT OR IGNORE INTO entity_partitions VALUES (?, ?)", batch_entities)
                    cur.executemany("INSERT INTO eval_records VALUES (?, ?, ?, ?, ?, ?, ?)", batch_eval)
                    batch_entities.clear()
                    batch_eval.clear()

        if batch_eval:
            cur.executemany("INSERT OR IGNORE INTO entity_partitions VALUES (?, ?)", batch_entities)
            cur.executemany("INSERT INTO eval_records VALUES (?, ?, ?, ?, ?, ?, ?)", batch_eval)
            batch_entities.clear()
            batch_eval.clear()
            
        db_conn.commit()

        # Create indexes for fast query evaluation
        cur.execute("CREATE INDEX idx_eval_records_lookup ON eval_records(split, is_matchable)")
        cur.execute("CREATE INDEX idx_eval_records_pair ON eval_records(pair_key_hash)")
        cur.execute("CREATE INDEX idx_eval_records_score ON eval_records(fuzzy_score)")

        # Identify conflicting label pair groups
        cur.execute("CREATE TABLE conflicting_pairs (pair_key_hash BLOB PRIMARY KEY)")
        cur.execute(
            """INSERT INTO conflicting_pairs (pair_key_hash)
               SELECT pair_key_hash
               FROM eval_records
               GROUP BY pair_key_hash
               HAVING count(DISTINCT y) > 1"""
        )
        db_conn.commit()

        # Audit conflicting label pairs
        cur.execute("SELECT count(*) FROM conflicting_pairs")
        conflicting_pairs_count = cur.fetchone()[0]

        cur.execute(
            """SELECT count(*)
               FROM eval_records
               WHERE pair_key_hash IN (SELECT pair_key_hash FROM conflicting_pairs)"""
        )
        conflicting_rows_total = cur.fetchone()[0]

        cur.execute(
            """SELECT count(*)
               FROM eval_records
               WHERE is_matchable = 1
                 AND pair_key_hash IN (SELECT pair_key_hash FROM conflicting_pairs)"""
        )
        conflicting_rows_matchable = cur.fetchone()[0]

        conflict_diagnostics = {
            "conflicting_pair_keys": conflicting_pairs_count,
            "conflicting_rows_total": conflicting_rows_total,
            "conflicting_rows_in_matchable_scope": conflicting_rows_matchable,
            "conflicting_rows_in_secondary_scope": conflicting_rows_total - conflicting_rows_matchable,
            "policy": (
                "Conflicting-label pair groups are excluded from clean evaluation metrics "
                "to avoid arbitrary label assignment, and retained here for diagnostics."
            ),
        }

        # Audit partition entity overlap via SQL
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
                "Pair-level hash splitting assigns all versions of the same pair key to the same partition, "
                "but entities connected across multiple distinct pair keys overlap across partitions. "
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

        # Evaluate Primary Scope: matchable schemas only
        primary_results = evaluate_scope_sql(
            cur=cur,
            is_matchable_filter=1,
            scope_name="primary_matchable_schemas",
        )

        # Evaluate Secondary Scope: broader dataset (all observations)
        secondary_results = evaluate_scope_sql(
            cur=cur,
            is_matchable_filter=None,
            scope_name="secondary_broader_dataset",
        )

    finally:
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
            "total_source_rows": total_source_rows,
            "exact_duplicates_discarded": exact_duplicates_discarded,
            "malformed_rows_skipped": malformed_rows,
            "unlabeled_or_unknown_rows_skipped": unlabeled_or_unknown_rows,
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
            "memory_management": "cursor_based_sqlite_evaluation_no_bulk_fetchall",
        },
        "conflict_diagnostics": conflict_diagnostics,
        "partition_audit": partition_audit,
        "scope_definitions": {
            "primary_scope": (
                "Observations where both entities have official matchable schemas: "
                f"{sorted(list(MATCHABLE_SCHEMAS))}. Excludes conflicting-label pair groups. "
                "Better represents direct sanctions screening."
            ),
            "secondary_scope": (
                "Broader benchmark including contextual and relationship-oriented schemas, "
                "with the same conflict-exclusion policy. Measures broader entity matching."
            ),
            "schema_distribution_across_dataset": schema_distribution,
            "uncertain_schemas_reported_for_review": uncertain_schemas,
        },
        "results": {
            "primary_scope_matchable": primary_results,
            "secondary_scope_broader": secondary_results,
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
    print("\n--- BASELINE RESULTS: PRIMARY SCOPE (MATCHABLE SCHEMAS) ---")
    p_test = summary["results"]["primary_scope_matchable"]["test_evaluation"]
    print(f"Test Clean Samples: {p_test['test_clean_samples_count']:,}")
    print(
        f"Baseline 1 (Exact Name/Caption): F1={p_test['baseline_1_exact_name_caption']['f1']}, "
        f"Precision={p_test['baseline_1_exact_name_caption']['precision']}, "
        f"Recall={p_test['baseline_1_exact_name_caption']['recall']}"
    )
    print(
        f"Baseline 2 (Exact Name/Alias):   F1={p_test['baseline_2_exact_name_alias']['f1']}, "
        f"Precision={p_test['baseline_2_exact_name_alias']['precision']}, "
        f"Recall={p_test['baseline_2_exact_name_alias']['recall']}"
    )
    p_fuzz = p_test["baseline_3_rapidfuzz_similarity"]
    print(
        f"Baseline 3 (Fuzzy Match, Thresh={p_fuzz['applied_threshold']}): "
        f"F1={p_fuzz['metrics']['f1']}, "
        f"Average Precision (AP)={p_fuzz['metrics']['average_precision']}"
    )

    print("\n--- BASELINE RESULTS: SECONDARY SCOPE (BROADER DATASET) ---")
    s_test = summary["results"]["secondary_scope_broader"]["test_evaluation"]
    print(f"Test Clean Samples: {s_test['test_clean_samples_count']:,}")
    print(
        f"Baseline 1 (Exact Name/Caption): F1={s_test['baseline_1_exact_name_caption']['f1']}, "
        f"Precision={s_test['baseline_1_exact_name_caption']['precision']}, "
        f"Recall={s_test['baseline_1_exact_name_caption']['recall']}"
    )
    print(
        f"Baseline 2 (Exact Name/Alias):   F1={s_test['baseline_2_exact_name_alias']['f1']}, "
        f"Precision={s_test['baseline_2_exact_name_alias']['precision']}, "
        f"Recall={s_test['baseline_2_exact_name_alias']['recall']}"
    )
    s_fuzz = s_test["baseline_3_rapidfuzz_similarity"]
    print(
        f"Baseline 3 (Fuzzy Match, Thresh={s_fuzz['applied_threshold']}): "
        f"F1={s_fuzz['metrics']['f1']}, "
        f"Average Precision (AP)={s_fuzz['metrics']['average_precision']}"
    )
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    main()

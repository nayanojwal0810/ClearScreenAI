"""scripts/audit_dataset.py

Streaming dataset audit for ClearScreenAI validation phase.
Audits the OpenSanctions Pairs snapshot and OFAC SDN Enhanced XML archive using
bounded memory and streaming processing.

Key design & privacy principles:
- Fully bounded RAM usage: uses streaming I/O and temporary disk-backed SQLite for
  pair-deduplication indexing and duplicate diagnostics. Temporary files are always
  cleaned up on exit or error.
- Cryptographic payload hashing: evaluates identical vs different entity payloads in
  repeated pair keys using hashes of canonicalized entity dictionaries.
- Identifier missingness: measures true entity properties (e.g. taxNumber, passportNumber,
  registrationNumber), explicitly excluding internal entity IDs.
- Primary evaluation target: reports all valid positive and negative judgement labels,
  as well as label breakdowns across schema pairs.
- Aggregate only: NEVER prints or persists raw entity names, birth dates, addresses,
  aliases, IDs, or raw JSON lines.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import zipfile
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# Documented business and personal identifier properties in FollowTheMoney / OpenSanctions.
# Note: Internal database entity IDs (entity.id) are explicitly EXCLUDED from this list.
IDENTIFIER_PROPERTY_FIELDS: Set[str] = {
    "idNumber",
    "registrationNumber",
    "taxNumber",
    "passportNumber",
    "innCode",
    "vatCode",
    "nationalId",
    "leiCode",
    "bic",
    "swiftBic",
    "serialNumber",
    "imoNumber",
    "ogrnCode",
    "kppCode",
}

# Expected snapshot benchmarks from OpenSanctions pairs snapshot (2025-12-09).
# Used strictly as reference checks, NEVER as hardcoded results.
EXPECTED_SNAPSHOT_ROWS = 755_540
EXPECTED_POSITIVE_RATIO = 0.769


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file using chunked streaming."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def stream_jsonl(filepath: str) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    """Streams lines from a plain or gzip JSONL file.
    
    Yields (line_number, parsed_dict or None, error_type or None).
    Never loads the full file into memory.
    """
    is_gz = filepath.endswith(".gz")
    opener = gzip.open(filepath, "rt", encoding="utf-8", errors="replace") if is_gz else open(
        filepath, "rt", encoding="utf-8", errors="replace"
    )
    
    with opener as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                if not isinstance(data, dict):
                    yield (line_num, None, "non_dict_json")
                else:
                    yield (line_num, data, None)
            except json.JSONDecodeError:
                yield (line_num, None, "json_decode_error")


def hash_entity_payload(entity: Optional[Dict[str, Any]]) -> bytes:
    """Computes a 16-byte cryptographic hash of a canonicalized entity payload.
    
    Ensures that identical entity contents produce identical hashes regardless
    of key ordering.
    """
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
    """Computes a 16-byte cryptographic hash of an unordered pair payload.
    
    Canonicalizes based on sorted entity IDs so (left=A, right=B) produces the same
    payload hash as (left=B, right=A) if the entity contents are identical.
    """
    h_left = hash_entity_payload(left_entity)
    h_right = hash_entity_payload(right_entity)
    if lid <= rid:
        combined = h_left + h_right
    else:
        combined = h_right + h_left
    return hashlib.sha256(combined).digest()[:16]


def extract_entity_info(row: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Extracts entity schema, id, caption, properties, and raw dict for 'left' or 'right'.
    
    Crucial: has_identifier measures ONLY business/personal identifier fields in properties.
    The internal record ID (entity.id) is explicitly not counted as an identifying attribute.
    """
    entity = row.get(side)
    if not isinstance(entity, dict):
        alt_key = "entity1" if side == "left" else "entity2"
        entity = row.get(alt_key)
    
    if not isinstance(entity, dict):
        return {
            "id": None,
            "schema": "Unknown",
            "has_caption": False,
            "has_properties": False,
            "has_identifier": False,
            "raw_entity": None,
        }
    
    ent_id = entity.get("id")
    schema = entity.get("schema")
    caption = entity.get("caption")
    properties = entity.get("properties")
    
    has_caption = bool(caption and str(caption).strip())
    has_props = isinstance(properties, dict) and len(properties) > 0
    
    # Check for actual business or personal identifier properties (taxNumber, passportNumber, etc.)
    # DO NOT count internal entity.id as an identifying attribute.
    has_identifier = False
    if isinstance(properties, dict):
        for field in IDENTIFIER_PROPERTY_FIELDS:
            val = properties.get(field)
            if val:
                if isinstance(val, list) and any(str(v).strip() for v in val):
                    has_identifier = True
                    break
                elif isinstance(val, (str, int)) and str(val).strip():
                    has_identifier = True
                    break
                
    return {
        "id": str(ent_id) if ent_id else None,
        "schema": str(schema) if schema else "Unknown",
        "has_caption": has_caption,
        "has_properties": has_props,
        "has_identifier": has_identifier,
        "raw_entity": entity,
    }


def normalize_label(row: Dict[str, Any]) -> str:
    """Normalizes label/decision/judgement into 'positive', 'negative', 'missing', or 'unknown'."""
    raw = row.get("judgement")
    if raw is None:
        raw = row.get("label")
    if raw is None:
        raw = row.get("decision")
    if raw is None:
        raw = row.get("match")
        
    if raw is None:
        return "missing"

    if raw in (1, True, "1", "true", "positive", "POSITIVE", "match"):
        return "positive"
    if raw in (0, False, "0", "false", "negative", "NEGATIVE", "no_match", "mismatch"):
        return "negative"
    return f"unknown_{str(raw).lower()}"


def audit_ofac_archive(ofac_path: str) -> Dict[str, Any]:
    """Audits the OFAC SDN archive file if present."""
    if not os.path.exists(ofac_path):
        dirname, basename = os.path.split(ofac_path)
        found_alt = None
        if os.path.exists(dirname):
            for entry in os.listdir(dirname):
                if entry.lower() == basename.lower():
                    found_alt = os.path.join(dirname, entry)
                    break
        if not found_alt:
            return {
                "path": ofac_path,
                "status": "missing",
                "message": f"Expected OFAC archive not found at '{ofac_path}'."
            }
        ofac_path = found_alt

    file_size = os.path.getsize(ofac_path)
    sha256 = compute_file_sha256(ofac_path)
    members = []
    
    if zipfile.is_zipfile(ofac_path):
        with zipfile.ZipFile(ofac_path, "r") as zf:
            for info in zf.infolist():
                members.append({
                    "filename": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                })
                
    return {
        "path": ofac_path,
        "status": "present",
        "file_size_bytes": file_size,
        "sha256": sha256,
        "archive_members": members,
    }


def run_audit(
    pairs_path: str,
    ofac_path: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Executes the streaming dataset audit with bounded memory and temporary disk-backed SQLite.
    
    Guarantees:
    - Bounded RAM usage: pair records and cryptographic payload hashes are stored
      in a temporary disk-backed SQLite database.
    - Temporary SQLite resources are guaranteed to be cleaned up on exit or error.
    - Zero raw entity names, birth dates, addresses, IDs, or lines are stored or output.
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    file_size_bytes = os.path.getsize(pairs_path)
    sha256_hash = compute_file_sha256(pairs_path)
    audit_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Aggregates
    total_parsed_rows = 0
    malformed_rows = 0
    parse_errors: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    left_schema_counts: Dict[str, int] = {}
    right_schema_counts: Dict[str, int] = {}
    schema_pair_counts: Dict[str, int] = {}
    schema_pair_label_counts: Dict[str, Dict[str, int]] = {}
    
    missing_captions = {"left": 0, "right": 0, "both": 0, "any": 0}
    empty_properties = {"left": 0, "right": 0, "both": 0, "any": 0}
    missing_identifiers = {"left": 0, "right": 0, "both": 0, "any": 0}
    
    rows_missing_ids = 0

    # Temporary disk-backed SQLite database for duplicate pair diagnostics
    temp_dir = tempfile.mkdtemp(prefix="clearscreen_audit_")
    db_path = os.path.join(temp_dir, "pair_diagnostics.db")
    db_conn: Optional[sqlite3.Connection] = None

    try:
        db_conn = sqlite3.connect(db_path)
        cur = db_conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = OFF")
        cur.execute(
            """CREATE TABLE pair_records (
                pair_key_hash BLOB,
                label TEXT,
                payload_hash BLOB
            )"""
        )
        db_conn.commit()

        batch_records: List[Tuple[bytes, str, bytes]] = []
        batch_size = 5000

        for line_num, row_data, err in stream_jsonl(pairs_path):
            if max_rows and total_parsed_rows >= max_rows:
                break
                
            if err or row_data is None:
                malformed_rows += 1
                err_key = err or "unknown_error"
                parse_errors[err_key] = parse_errors.get(err_key, 0) + 1
                continue
                
            total_parsed_rows += 1
            
            # Label counting
            lbl = normalize_label(row_data)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            
            # Schema and attribute extraction
            left_info = extract_entity_info(row_data, "left")
            right_info = extract_entity_info(row_data, "right")
            
            l_sch = left_info["schema"]
            r_sch = right_info["schema"]
            left_schema_counts[l_sch] = left_schema_counts.get(l_sch, 0) + 1
            right_schema_counts[r_sch] = right_schema_counts.get(r_sch, 0) + 1
            
            # Schema pair counting and label breakdown
            sch_pair = f"{min(l_sch, r_sch)} <-> {max(l_sch, r_sch)}"
            schema_pair_counts[sch_pair] = schema_pair_counts.get(sch_pair, 0) + 1
            
            if sch_pair not in schema_pair_label_counts:
                schema_pair_label_counts[sch_pair] = {
                    "positive": 0,
                    "negative": 0,
                    "unknown_or_missing": 0,
                    "total": 0,
                }
            schema_pair_label_counts[sch_pair]["total"] += 1
            if lbl in ("positive", "negative"):
                schema_pair_label_counts[sch_pair][lbl] += 1
            else:
                schema_pair_label_counts[sch_pair]["unknown_or_missing"] += 1
            
            # Attribute missingness: Captions
            left_no_cap = not left_info["has_caption"]
            right_no_cap = not right_info["has_caption"]
            if left_no_cap:
                missing_captions["left"] += 1
            if right_no_cap:
                missing_captions["right"] += 1
            if left_no_cap and right_no_cap:
                missing_captions["both"] += 1
            if left_no_cap or right_no_cap:
                missing_captions["any"] += 1
                
            # Attribute missingness: Properties
            left_no_props = not left_info["has_properties"]
            right_no_props = not right_info["has_properties"]
            if left_no_props:
                empty_properties["left"] += 1
            if right_no_props:
                empty_properties["right"] += 1
            if left_no_props and right_no_props:
                empty_properties["both"] += 1
            if left_no_props or right_no_props:
                empty_properties["any"] += 1
                
            # Attribute missingness: Identifiers (business / personal properties only)
            left_no_id = not left_info["has_identifier"]
            right_no_id = not right_info["has_identifier"]
            if left_no_id:
                missing_identifiers["left"] += 1
            if right_no_id:
                missing_identifiers["right"] += 1
            if left_no_id and right_no_id:
                missing_identifiers["both"] += 1
            if left_no_id or right_no_id:
                missing_identifiers["any"] += 1
                
            # Record duplicate diagnostics using temporary SQLite
            lid = left_info["id"]
            rid = right_info["id"]
            if lid and rid:
                pair_key_bytes = f"{min(lid, rid)}::{max(lid, rid)}".encode("utf-8")
                pair_key_h = hashlib.blake2b(pair_key_bytes, digest_size=16).digest()
                payload_h = canonical_pair_payload_hash(
                    lid, rid, left_info["raw_entity"], right_info["raw_entity"]
                )
                batch_records.append((pair_key_h, lbl, payload_h))
                if len(batch_records) >= batch_size:
                    cur.executemany("INSERT INTO pair_records VALUES (?, ?, ?)", batch_records)
                    batch_records.clear()
            else:
                rows_missing_ids += 1

        if batch_records:
            cur.executemany("INSERT INTO pair_records VALUES (?, ?, ?)", batch_records)
            batch_records.clear()

        # Index pair_key_hash for diagnostic query performance
        cur.execute("CREATE INDEX idx_pair_key_hash ON pair_records(pair_key_hash)")
        db_conn.commit()

        # 1. Total rows with IDs & unique unordered entity-pair keys
        total_with_ids = total_parsed_rows - rows_missing_ids
        cur.execute("SELECT count(DISTINCT pair_key_hash) FROM pair_records")
        unique_pair_keys = cur.fetchone()[0]

        # 2. Number of repeated pair keys and duplicate rows
        cur.execute(
            """SELECT count(*) FROM (
                SELECT pair_key_hash FROM pair_records
                GROUP BY pair_key_hash
                HAVING count(*) > 1
            )"""
        )
        repeated_pair_keys = cur.fetchone()[0]
        duplicate_rows = total_with_ids - unique_pair_keys

        # 3 & 4. Conflicting label pair keys and rows
        cur.execute(
            """SELECT
                count(*),
                coalesce(sum(cnt), 0)
            FROM (
                SELECT pair_key_hash, count(*) as cnt
                FROM pair_records
                GROUP BY pair_key_hash
                HAVING count(DISTINCT CASE WHEN label IN ('positive', 'negative') THEN label END) > 1
            )"""
        )
        conflicting_res = cur.fetchone()
        conflicting_keys_count = conflicting_res[0] if conflicting_res else 0
        conflicting_rows_count = conflicting_res[1] if conflicting_res else 0

        # 5. Number of repeated pairs with identical versus different entity payloads
        cur.execute(
            """SELECT
                coalesce(sum(CASE WHEN distinct_payloads = 1 THEN 1 ELSE 0 END), 0) as identical_keys,
                coalesce(sum(CASE WHEN distinct_payloads > 1 THEN 1 ELSE 0 END), 0) as different_keys,
                coalesce(sum(CASE WHEN distinct_payloads = 1 THEN row_cnt ELSE 0 END), 0) as identical_rows,
                coalesce(sum(CASE WHEN distinct_payloads > 1 THEN row_cnt ELSE 0 END), 0) as different_rows
            FROM (
                SELECT
                    pair_key_hash,
                    count(*) as row_cnt,
                    count(DISTINCT payload_hash) as distinct_payloads
                FROM pair_records
                GROUP BY pair_key_hash
                HAVING count(*) > 1
            )"""
        )
        payload_diag = cur.fetchone()
        identical_payload_keys = payload_diag[0] if payload_diag else 0
        different_payload_keys = payload_diag[1] if payload_diag else 0
        identical_payload_rows = payload_diag[2] if payload_diag else 0
        different_payload_rows = payload_diag[3] if payload_diag else 0

    finally:
        # Guarantee cleanup of temporary database and directory
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

    # Expected vs observed snapshot checks
    valid_positive = label_counts.get("positive", 0)
    valid_negative = label_counts.get("negative", 0)
    valid_labeled_rows = valid_positive + valid_negative
    unknown_or_missing_rows = sum(
        count for lbl, count in label_counts.items() if lbl not in ("positive", "negative")
    )

    expected_checks = {
        "expected_total_rows": EXPECTED_SNAPSHOT_ROWS,
        "observed_total_rows": total_parsed_rows,
        "rows_match": (total_parsed_rows == EXPECTED_SNAPSHOT_ROWS),
        "expected_positive_ratio": EXPECTED_POSITIVE_RATIO,
        "observed_positive_ratio": (
            valid_positive / valid_labeled_rows if valid_labeled_rows > 0 else 0.0
        ),
        "warnings": [],
    }
    
    if total_parsed_rows > 0 and total_parsed_rows != EXPECTED_SNAPSHOT_ROWS:
        expected_checks["warnings"].append(
            f"Observed rows ({total_parsed_rows}) differs from expected snapshot rows ({EXPECTED_SNAPSHOT_ROWS})."
        )
    pos_ratio = expected_checks["observed_positive_ratio"]
    if valid_labeled_rows > 1000 and abs(pos_ratio - EXPECTED_POSITIVE_RATIO) > 0.05:
        expected_checks["warnings"].append(
            f"Observed positive ratio ({pos_ratio:.3f}) deviates from expected reference ({EXPECTED_POSITIVE_RATIO:.3f})."
        )
    if unknown_or_missing_rows > 0:
        expected_checks["warnings"].append(
            f"Encountered {unknown_or_missing_rows} rows with unknown or missing labels."
        )

    duplicate_analysis = {
        "total_rows_with_pair_ids": total_with_ids,
        "unique_unordered_pair_keys": unique_pair_keys,
        "repeated_pair_keys": repeated_pair_keys,
        "duplicate_rows": duplicate_rows,
        "conflicting_label_pair_keys": conflicting_keys_count,
        "conflicting_label_rows": conflicting_rows_count,
        "repeated_pairs_identical_payload": identical_payload_keys,
        "repeated_pairs_different_payload": different_payload_keys,
        "repeated_rows_identical_payload": identical_payload_rows,
        "repeated_rows_different_payload": different_payload_rows,
        "note": (
            "Cryptographic hashes of canonicalized entity payloads were used for duplicate-content "
            "diagnostics. No raw entity content or identifiers are persisted."
        ),
    }

    summary: Dict[str, Any] = {
        "metadata": {
            "pairs_file": pairs_path,
            "file_size_bytes": file_size_bytes,
            "sha256": sha256_hash,
            "audit_timestamp_utc": audit_time,
            "max_rows_limit": max_rows,
            "memory_management": "streaming_jsonl_with_disk_backed_sqlite_indexing",
        },
        "row_counts": {
            "total_parsed_rows": total_parsed_rows,
            "malformed_rows": malformed_rows,
            "parse_errors": parse_errors,
            "valid_labeled_rows": valid_labeled_rows,
            "unknown_or_missing_label_rows": unknown_or_missing_rows,
        },
        "labels": {
            "counts": label_counts,
            "primary_targets": {"positive": valid_positive, "negative": valid_negative},
        },
        "schemas": {
            "left_schema_counts": left_schema_counts,
            "right_schema_counts": right_schema_counts,
            "schema_pair_counts": schema_pair_counts,
            "schema_pair_label_counts": schema_pair_label_counts,
        },
        "duplicate_pair_analysis": duplicate_analysis,
        "data_quality": {
            "missing_captions": missing_captions,
            "empty_properties": empty_properties,
            "missing_property_identifiers": {
                **missing_identifiers,
                "note": "Evaluated against business/personal identifier fields; entity.id explicitly excluded.",
                "identifier_fields_checked": sorted(list(IDENTIFIER_PROPERTY_FIELDS)),
            },
            "duplicate_pair_rows": duplicate_rows,
            "unique_pair_ids_seen": unique_pair_keys,
            "rows_missing_ids": rows_missing_ids,
            "duplicate_pair_analysis": duplicate_analysis,
        },
        "expected_vs_observed_checks": expected_checks,
    }

    if ofac_path:
        summary["ofac_sdn_archive"] = audit_ofac_archive(ofac_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming dataset audit for ClearScreenAI validation (aggregate-only)."
    )
    parser.add_argument(
        "--pairs",
        default="data/raw/pairs.json.gz",
        help="Path to labeled pairs JSONL/GZ benchmark (default: data/raw/pairs.json.gz)",
    )
    parser.add_argument(
        "--ofac-zip",
        default="data/raw/SDN_ENHANCED.ZIP",
        help="Path to OFAC SDN Enhanced ZIP archive (default: data/raw/SDN_ENHANCED.ZIP)",
    )
    parser.add_argument(
        "--out",
        default="outputs/audit_summary.json",
        help="Path to output aggregate JSON (default: outputs/audit_summary.json)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum rows to parse (for test fixtures/smoke tests)",
    )

    args = parser.parse_args()

    print(f"[*] Starting dataset audit on: {args.pairs}")
    if not os.path.exists(args.pairs):
        print(f"[!] File not found: {args.pairs}")
        sys.exit(1)

    summary = run_audit(
        pairs_path=args.pairs,
        ofac_path=args.ofac_zip,
        max_rows=args.max_rows,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Audit completed. Aggregate results written to: {args.out}")
    print("\n--- AGGREGATE SUMMARY ---")
    print(f"Total rows parsed: {summary['row_counts']['total_parsed_rows']:,}")
    print(f"Valid labeled rows: {summary['row_counts']['valid_labeled_rows']:,}")
    print(f"Unknown/missing labels: {summary['row_counts']['unknown_or_missing_label_rows']:,}")
    print(f"Labels: {summary['labels']['counts']}")
    
    dup_info = summary["duplicate_pair_analysis"]
    print(f"Unique pair keys: {dup_info['unique_unordered_pair_keys']:,}")
    print(f"Repeated pair keys: {dup_info['repeated_pair_keys']:,}")
    print(f"Duplicate rows: {dup_info['duplicate_rows']:,}")
    print(f"Conflicting-label pair keys: {dup_info['conflicting_label_pair_keys']:,} ({dup_info['conflicting_label_rows']:,} rows)")
    print(f"Repeated pairs with identical payload: {dup_info['repeated_pairs_identical_payload']:,} keys ({dup_info['repeated_rows_identical_payload']:,} rows)")
    print(f"Repeated pairs with different payload: {dup_info['repeated_pairs_different_payload']:,} keys ({dup_info['repeated_rows_different_payload']:,} rows)")
    
    if summary.get("ofac_sdn_archive"):
        print(f"OFAC SDN archive: {summary['ofac_sdn_archive'].get('status')}")
    warnings = summary["expected_vs_observed_checks"]["warnings"]
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()

"""scripts/audit_dataset.py

Streaming dataset audit for ClearScreenAI validation phase.
Audits OpenSanctions Pairs snapshot and OFAC SDN Enhanced XML archive without
loading full decompressed files into RAM.

Privacy Guarantee:
Reports ONLY aggregate statistics. NEVER prints or stores raw entity names,
birth dates, addresses, aliases, IDs, or raw JSON lines.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import sys
import zipfile
from typing import Any, Dict, Iterator, Optional, Set, Tuple


# Known relation/auto-propagated schemas in FollowTheMoney / OpenSanctions
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

# Expected snapshot benchmarks from OpenSanctions pairs snapshot (2025-12-09)
# Used strictly as reference checks, NEVER as hardcoded results.
EXPECTED_SNAPSHOT_ROWS = 755_540
EXPECTED_POSITIVE_RATIO = 0.769
EXPECTED_ANALYST_JUDGED_ROWS = 501_298
EXPECTED_RELATION_ROWS = 254_242


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


def extract_entity_info(row: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Extracts entity schema, id, caption, and properties for 'left' or 'right'."""
    entity = row.get(side)
    if not isinstance(entity, dict):
        alt_key = "entity1" if side == "left" else "entity2"
        entity = row.get(alt_key)
    
    if not isinstance(entity, dict):
        return {
            "id": None,
            "schema": None,
            "has_caption": False,
            "has_properties": False,
            "has_identifier": False,
        }
    
    ent_id = entity.get("id")
    schema = entity.get("schema")
    caption = entity.get("caption")
    properties = entity.get("properties")
    
    has_caption = bool(caption and str(caption).strip())
    has_props = isinstance(properties, dict) and len(properties) > 0
    
    # Check for identifier: either entity-level id or ID-like property fields
    has_id = bool(ent_id and str(ent_id).strip())
    if not has_id and isinstance(properties, dict):
        for id_prop in ("idNumber", "registrationNumber", "taxNumber", "passportNumber", "innCode"):
            if properties.get(id_prop):
                has_id = True
                break
                
    return {
        "id": str(ent_id) if ent_id else None,
        "schema": str(schema) if schema else "Unknown",
        "has_caption": has_caption,
        "has_properties": has_props,
        "has_identifier": has_id,
    }


def normalize_label(row: Dict[str, Any]) -> str:
    """Normalizes label/decision from row into a canonical string."""
    raw = row.get("label")
    if raw is None:
        raw = row.get("decision")
    if raw is None:
        raw = row.get("judgement")
    if raw is None:
        raw = row.get("match")
        
    if raw in (1, True, "1", "true", "positive", "POSITIVE", "match"):
        return "positive"
    if raw in (0, False, "0", "false", "negative", "NEGATIVE", "no_match", "mismatch"):
        return "negative"
    return str(raw).lower() if raw is not None else "missing"


def is_relation_row(
    left_schema: str,
    right_schema: str,
    row_schema: Optional[str] = None,
    relation_schemas: Optional[Set[str]] = None,
) -> bool:
    """Determines if a row represents a relation/auto-propagated pair schema."""
    check_set = relation_schemas or KNOWN_RELATION_SCHEMAS
    if left_schema in check_set or right_schema in check_set:
        return True
    if row_schema and row_schema in check_set:
        return True
    return False


def audit_ofac_archive(ofac_path: str) -> Dict[str, Any]:
    """Audits the OFAC SDN archive file if present."""
    if not os.path.exists(ofac_path):
        # Check case-insensitive variant in the same directory
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
    relation_schemas: Optional[Set[str]] = None,
    verify_analyst_filter: bool = False,
) -> Dict[str, Any]:
    """Executes the full streaming dataset audit.
    
    Processes the benchmark line-by-line and returns strictly aggregate statistics.
    Never stores or returns raw entity names, birth dates, addresses, IDs, or lines.
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    file_size_bytes = os.path.getsize(pairs_path)
    sha256_hash = compute_file_sha256(pairs_path)
    audit_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Tracking aggregates
    total_parsed_rows = 0
    malformed_rows = 0
    parse_errors: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    left_schema_counts: Dict[str, int] = {}
    right_schema_counts: Dict[str, int] = {}
    schema_pair_counts: Dict[str, int] = {}
    relation_schema_counts: Dict[str, int] = {}
    
    missing_captions = {"left": 0, "right": 0, "any": 0}
    empty_properties = {"left": 0, "right": 0, "any": 0}
    missing_identifiers = {"left": 0, "right": 0, "any": 0}
    
    seen_pair_hashes: Set[bytes] = set()
    duplicate_pair_rows = 0
    rows_missing_ids = 0
    
    analyst_judged_count = 0
    relation_pairs_count = 0
    
    target_relations = relation_schemas or KNOWN_RELATION_SCHEMAS

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
        row_schema = str(row_data.get("schema")) if row_data.get("schema") else None
        
        l_sch = left_info["schema"]
        r_sch = right_info["schema"]
        left_schema_counts[l_sch] = left_schema_counts.get(l_sch, 0) + 1
        right_schema_counts[r_sch] = right_schema_counts.get(r_sch, 0) + 1
        
        # Sorted schema pair for symmetry: (A, B) same as (B, A)
        sch_pair = f"{min(l_sch, r_sch)} <-> {max(l_sch, r_sch)}"
        schema_pair_counts[sch_pair] = schema_pair_counts.get(sch_pair, 0) + 1
        
        # Attribute missingness
        if not left_info["has_caption"]:
            missing_captions["left"] += 1
        if not right_info["has_caption"]:
            missing_captions["right"] += 1
        if not left_info["has_caption"] or not right_info["has_caption"]:
            missing_captions["any"] += 1
            
        if not left_info["has_properties"]:
            empty_properties["left"] += 1
        if not right_info["has_properties"]:
            empty_properties["right"] += 1
        if not left_info["has_properties"] or not right_info["has_properties"]:
            empty_properties["any"] += 1
            
        if not left_info["has_identifier"]:
            missing_identifiers["left"] += 1
        if not right_info["has_identifier"]:
            missing_identifiers["right"] += 1
        if not left_info["has_identifier"] or not right_info["has_identifier"]:
            missing_identifiers["any"] += 1
            
        # Duplicate detection using stable IDs
        lid = left_info["id"]
        rid = right_info["id"]
        if lid and rid:
            pair_key = f"{min(lid, rid)}::{max(lid, rid)}".encode("utf-8")
            h = hashlib.blake2b(pair_key, digest_size=8).digest()
            if h in seen_pair_hashes:
                duplicate_pair_rows += 1
            else:
                seen_pair_hashes.add(h)
        else:
            rows_missing_ids += 1
            
        # Relation schema tracking
        is_rel = is_relation_row(l_sch, r_sch, row_schema, target_relations)
        if is_rel:
            relation_pairs_count += 1
            for sch in (l_sch, r_sch, row_schema):
                if sch and sch in target_relations:
                    relation_schema_counts[sch] = relation_schema_counts.get(sch, 0) + 1
        else:
            analyst_judged_count += 1

    # Expected vs observed snapshot checks
    expected_checks = {
        "expected_total_rows": EXPECTED_SNAPSHOT_ROWS,
        "observed_total_rows": total_parsed_rows,
        "rows_match": (total_parsed_rows == EXPECTED_SNAPSHOT_ROWS),
        "expected_positive_ratio": EXPECTED_POSITIVE_RATIO,
        "observed_positive_ratio": (
            label_counts.get("positive", 0) / total_parsed_rows if total_parsed_rows > 0 else 0.0
        ),
        "warnings": [],
    }
    
    if total_parsed_rows > 0 and total_parsed_rows != EXPECTED_SNAPSHOT_ROWS:
        expected_checks["warnings"].append(
            f"Observed rows ({total_parsed_rows}) differs from expected snapshot rows ({EXPECTED_SNAPSHOT_ROWS})."
        )
    pos_ratio = expected_checks["observed_positive_ratio"]
    if total_parsed_rows > 1000 and abs(pos_ratio - EXPECTED_POSITIVE_RATIO) > 0.05:
        expected_checks["warnings"].append(
            f"Observed positive ratio ({pos_ratio:.3f}) deviates from expected ({EXPECTED_POSITIVE_RATIO:.3f})."
        )

    # Analyst-judged subset status
    if verify_analyst_filter:
        subset_info = {
            "status": "verified_by_user_flag",
            "analyst_judged_rows": analyst_judged_count,
            "relation_rows": relation_pairs_count,
            "filter_rule": "Excluded pairs where left, right, or row schema matches relation schemas.",
            "target_relation_schemas": sorted(list(target_relations)),
        }
    else:
        subset_info = {
            "status": "not determined — review required",
            "reason": (
                "Schema-based filter must be confirmed by human review against the published paper "
                "before excluding auto-propagated pairs. Use --verify-analyst-filter only after verification."
            ),
            "unverified_candidate_counts": {
                "candidate_subject_matching_rows": analyst_judged_count,
                "candidate_relation_rows": relation_pairs_count,
            },
            "candidate_relation_schemas": sorted(list(target_relations)),
        }

    summary: Dict[str, Any] = {
        "metadata": {
            "pairs_file": pairs_path,
            "file_size_bytes": file_size_bytes,
            "sha256": sha256_hash,
            "audit_timestamp_utc": audit_time,
            "max_rows_limit": max_rows,
        },
        "row_counts": {
            "total_parsed_rows": total_parsed_rows,
            "malformed_rows": malformed_rows,
            "parse_errors": parse_errors,
        },
        "labels": {
            "counts": label_counts,
        },
        "schemas": {
            "left_schema_counts": left_schema_counts,
            "right_schema_counts": right_schema_counts,
            "schema_pair_counts": schema_pair_counts,
            "detected_relation_schemas": relation_schema_counts,
        },
        "data_quality": {
            "missing_captions": missing_captions,
            "empty_properties": empty_properties,
            "missing_identifiers": missing_identifiers,
            "duplicate_pair_rows": duplicate_pair_rows,
            "unique_pair_ids_seen": len(seen_pair_hashes),
            "rows_missing_ids": rows_missing_ids,
        },
        "analyst_judged_subset": subset_info,
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
    parser.add_argument(
        "--relation-schemas",
        default=None,
        help="Comma-separated custom relation schemas to test or override",
    )
    parser.add_argument(
        "--verify-analyst-filter",
        action="store_true",
        help="Explicitly mark relation-schema filter as verified by human review",
    )
    
    args = parser.parse_args()

    rel_schemas = None
    if args.relation_schemas:
        rel_schemas = set(s.strip() for s in args.relation_schemas.split(",") if s.strip())

    print(f"[*] Starting dataset audit on: {args.pairs}")
    if not os.path.exists(args.pairs):
        print(f"[!] File not found: {args.pairs}")
        sys.exit(1)

    summary = run_audit(
        pairs_path=args.pairs,
        ofac_path=args.ofac_zip,
        max_rows=args.max_rows,
        relation_schemas=rel_schemas,
        verify_analyst_filter=args.verify_analyst_filter,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Audit completed. Aggregate results written to: {args.out}")
    print("\n--- AGGREGATE SUMMARY ---")
    print(f"Total rows parsed: {summary['row_counts']['total_parsed_rows']:,}")
    print(f"Malformed rows: {summary['row_counts']['malformed_rows']:,}")
    print(f"Labels: {summary['labels']['counts']}")
    print(f"Unique pair IDs: {summary['data_quality']['unique_pair_ids_seen']:,}")
    print(f"Duplicate pair rows: {summary['data_quality']['duplicate_pair_rows']:,}")
    print(f"Analyst-judged status: {summary['analyst_judged_subset']['status']}")
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

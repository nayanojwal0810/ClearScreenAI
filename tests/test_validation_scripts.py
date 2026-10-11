"""tests/test_validation_scripts.py

Unit tests for ClearScreenAI validation scaffolding.
Uses only compact synthetic test fixtures generated in-memory or in temp files.
DOES NOT copy or use any real records from OpenSanctions or OFAC.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest

# Import from scripts
from scripts.audit_dataset import (
    extract_entity_info,
    is_relation_row,
    normalize_label,
    run_audit,
    stream_jsonl,
)
from scripts.evaluate_baselines import (
    calculate_similarity_score,
    compute_metrics,
    evaluate_benchmark,
    extract_names_and_aliases,
    get_deterministic_split,
    normalize_binary_label,
    normalize_text,
    tune_fuzzy_threshold,
)


class TestValidationScripts(unittest.TestCase):
    """Test suite for dataset audit and baseline evaluation scaffolding."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gz_path = os.path.join(self.temp_dir.name, "synthetic_pairs.json.gz")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_synthetic_gz(self, rows_and_lines: list) -> str:
        """Helper to create a temporary .json.gz file with synthetic rows."""
        with gzip.open(self.gz_path, "wt", encoding="utf-8") as f:
            for item in rows_and_lines:
                if isinstance(item, str):
                    f.write(item + "\n")
                else:
                    f.write(json.dumps(item) + "\n")
        return self.gz_path

    def test_text_normalization(self) -> None:
        """Test Unicode NFKD, case-folding, and punctuation handling."""
        self.assertEqual(normalize_text("Müller, Hans"), "muller hans")
        self.assertEqual(normalize_text("  ACME   HOLDINGS,  LLC  "), "acme holdings llc")
        self.assertEqual(normalize_text("O'Connor-Smith"), "o connor smith")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(""), "")

    def test_extract_names_and_aliases(self) -> None:
        """Test extraction of primary names and all aliases from synthetic entity."""
        synthetic_entity = {
            "id": "syn-001",
            "caption": "Alpha Trading Corp",
            "properties": {
                "name": ["Alpha Trading", "ALPHA CORP"],
                "alias": ["ATC International"],
                "weakAlias": ["Alpha"],
            },
        }
        primary, all_names = extract_names_and_aliases(synthetic_entity)
        self.assertIn("alpha trading corp", primary)
        self.assertIn("alpha trading", primary)
        self.assertIn("alpha corp", primary)
        self.assertNotIn("atc international", primary)
        
        self.assertIn("atc international", all_names)
        self.assertIn("alpha", all_names)
        self.assertEqual(len(all_names), 5)

    def test_similarity_score(self) -> None:
        """Test fuzzy similarity on identical, reordered, typo, and distinct names."""
        s1 = {"john doe"}
        s2 = {"doe john"}
        s3 = {"jon doe"}
        s4 = {"completely different entity"}

        # Identical / reordered
        self.assertGreaterEqual(calculate_similarity_score(s1, s1), 0.99)
        self.assertGreaterEqual(calculate_similarity_score(s1, s2), 0.95)
        # Typo
        self.assertGreater(calculate_similarity_score(s1, s3), 0.70)
        # Unrelated
        self.assertLess(calculate_similarity_score(s1, s4), 0.40)

    def test_stream_jsonl_and_malformed_handling(self) -> None:
        """Test streaming reader with valid rows, empty lines, and malformed lines."""
        content = [
            {"label": "positive", "left": {"id": "1"}, "right": {"id": "2"}},
            "",  # empty line
            "INVALID_JSON_LINE{{{",  # malformed JSON
            12345,  # non-dict line
            {"label": "negative", "left": {"id": "3"}, "right": {"id": "4"}},
        ]
        path = self._write_synthetic_gz(content)
        
        results = list(stream_jsonl(path))
        self.assertEqual(len(results), 4)  # empty line is skipped silently
        
        # Line 1: valid dict
        self.assertEqual(results[0][0], 1)
        self.assertIsNotNone(results[0][1])
        self.assertIsNone(results[0][2])
        
        # Line 3: JSON decode error
        self.assertEqual(results[1][0], 3)
        self.assertIsNone(results[1][1])
        self.assertEqual(results[1][2], "json_decode_error")
        
        # Line 4: non dict
        self.assertEqual(results[2][0], 4)
        self.assertIsNone(results[2][1])
        self.assertEqual(results[2][2], "non_dict_json")
        
        # Line 5: valid dict
        self.assertEqual(results[3][0], 5)
        self.assertIsNotNone(results[3][1])
        self.assertIsNone(results[3][2])

    def test_audit_aggregate_output_and_privacy_guarantee(self) -> None:
        """Verify that audit computes aggregate metrics and contains NO raw entity content."""
        sensitive_name = "Synthetic Secret Target Name"
        sensitive_id = "syn_id_99998888"
        sensitive_dob = "1975-04-12"
        sensitive_addr = "123 Synthetic Fictitious Way"

        synthetic_data = [
            {
                "id": "pair_01",
                "label": "positive",
                "left": {
                    "id": sensitive_id,
                    "schema": "Person",
                    "caption": sensitive_name,
                    "properties": {
                        "birthDate": [sensitive_dob],
                        "address": [sensitive_addr],
                    },
                },
                "right": {
                    "id": "syn_id_2222",
                    "schema": "Person",
                    "caption": "Synthetic Alias Name",
                    "properties": {"name": ["Synthetic Alias Name"]},
                },
            },
            {
                "id": "pair_02",
                "label": "negative",
                "left": {
                    "id": "syn_id_3333",
                    "schema": "Company",
                    "caption": "Synthetic Co A",
                    "properties": {},
                },
                "right": {
                    "id": "syn_id_4444",
                    "schema": "Company",
                    "caption": "",  # missing caption
                    "properties": None,  # empty properties
                },
            },
            # Duplicate pair (same IDs as pair 01, reversed)
            {
                "id": "pair_03",
                "label": "positive",
                "left": {"id": "syn_id_2222", "schema": "Person", "caption": "Synthetic Alias Name"},
                "right": {"id": sensitive_id, "schema": "Person", "caption": sensitive_name, "properties": {"name": [sensitive_name]}},
            },
            # Malformed line
            "BAD_LINE_SYNTHETIC",
        ]

        path = self._write_synthetic_gz(synthetic_data)
        out_json = os.path.join(self.temp_dir.name, "test_audit_out.json")

        summary = run_audit(pairs_path=path, ofac_path=None, verify_analyst_filter=False)

        # Assert correct aggregates
        self.assertEqual(summary["row_counts"]["total_parsed_rows"], 3)
        self.assertEqual(summary["row_counts"]["malformed_rows"], 1)
        self.assertEqual(summary["labels"]["counts"].get("positive"), 2)
        self.assertEqual(summary["labels"]["counts"].get("negative"), 1)
        self.assertEqual(summary["data_quality"]["duplicate_pair_rows"], 1)
        self.assertEqual(summary["data_quality"]["missing_captions"]["right"], 1)
        self.assertEqual(summary["data_quality"]["empty_properties"]["right"], 1)

        # Privacy test: serialize summary to JSON string and ensure raw strings never leak
        serialized = json.dumps(summary)
        self.assertNotIn(sensitive_name, serialized)
        self.assertNotIn(sensitive_id, serialized)
        self.assertNotIn(sensitive_dob, serialized)
        self.assertNotIn(sensitive_addr, serialized)
        self.assertNotIn("BAD_LINE_SYNTHETIC", serialized)

    def test_analyst_judged_subset_switch(self) -> None:
        """Test the analyst-judged subset switch requirement."""
        data = [
            # Normal person match
            {
                "label": "positive",
                "left": {"id": "p1", "schema": "Person", "caption": "Alice"},
                "right": {"id": "p2", "schema": "Person", "caption": "Alice"},
            },
            # Auto-propagated relation schema
            {
                "label": "positive",
                "left": {"id": "p1", "schema": "Ownership", "caption": "Owns"},
                "right": {"id": "c1", "schema": "Company", "caption": "Company C"},
            },
        ]
        path = self._write_synthetic_gz(data)

        # 1. Unverified run: must mark status as review required
        unverified = run_audit(pairs_path=path, verify_analyst_filter=False)
        self.assertEqual(
            unverified["analyst_judged_subset"]["status"],
            "not determined — review required",
        )

        # 2. Verified run: excludes relation schema
        verified = run_audit(pairs_path=path, verify_analyst_filter=True)
        self.assertEqual(verified["analyst_judged_subset"]["status"], "verified_by_user_flag")
        self.assertEqual(verified["analyst_judged_subset"]["analyst_judged_rows"], 1)
        self.assertEqual(verified["analyst_judged_subset"]["relation_rows"], 1)

    def test_compute_metrics_and_tuning(self) -> None:
        """Test classification metrics computation and threshold tuning."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        scores = [0.9, 0.4, 0.2, 0.7]

        metrics = compute_metrics(y_true, y_pred, scores=scores)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

        # Tuning on validation data
        val_records = [
            (0.95, 1),
            (0.85, 1),
            (0.75, 1),
            (0.65, 0),
            (0.30, 0),
        ]
        thresh, val_f1 = tune_fuzzy_threshold(val_records)
        self.assertGreaterEqual(thresh, 0.70)
        self.assertEqual(val_f1, 1.0)

    def test_evaluate_baselines_synthetic(self) -> None:
        """Test streaming evaluate_baselines on synthetic dataset."""
        synthetic_pairs = [
            {
                "label": "positive",
                "left": {"id": "e1", "caption": "Sample Entity One", "properties": {"alias": ["SEO"]}},
                "right": {"id": "e2", "caption": "Sample Entity One", "properties": {"alias": ["SEO"]}},
            },
            {
                "label": "positive",
                "left": {"id": "e3", "caption": "Primary Name Only", "properties": {"alias": ["Alias Match"]}},
                "right": {"id": "e4", "caption": "Different Name", "properties": {"alias": ["Alias Match"]}},
            },
            {
                "label": "negative",
                "left": {"id": "e5", "caption": "Alpha Beta Gamma", "properties": {}},
                "right": {"id": "e6", "caption": "Zeta Theta Omega", "properties": {}},
            },
        ]
        path = self._write_synthetic_gz(synthetic_pairs)

        result = evaluate_benchmark(pairs_path=path)

        self.assertIn("partition_audit", result)
        self.assertEqual(result["partition_audit"]["evaluation_level"], "pair_level")
        self.assertFalse(result["partition_audit"]["unseen_entity_generalization_claimed"])
        self.assertIn("results_all_rows", result)
        self.assertIn("test_evaluation", result["results_all_rows"])

    def test_ofac_archive_synthetic(self) -> None:
        """Test OFAC zip audit logic on synthetic zip archive and missing file."""
        import zipfile
        from scripts.audit_dataset import audit_ofac_archive

        # 1. Missing archive
        missing_res = audit_ofac_archive(os.path.join(self.temp_dir.name, "nonexistent.zip"))
        self.assertEqual(missing_res["status"], "missing")

        # 2. Synthetic archive
        synthetic_zip = os.path.join(self.temp_dir.name, "test_sdn.zip")
        with zipfile.ZipFile(synthetic_zip, "w") as zf:
            zf.writestr("sdn_test.xml", b"<xml>dummy content</xml>")

        present_res = audit_ofac_archive(synthetic_zip)
        self.assertEqual(present_res["status"], "present")
        self.assertEqual(len(present_res["archive_members"]), 1)
        self.assertEqual(present_res["archive_members"][0]["filename"], "sdn_test.xml")
        self.assertGreater(present_res["file_size_bytes"], 0)

    def test_max_rows_limit(self) -> None:
        """Test that max_rows limits processing cleanly without error."""
        records = [
            {"label": "positive", "left": {"id": f"l{i}"}, "right": {"id": f"r{i}"}}
            for i in range(10)
        ]
        path = self._write_synthetic_gz(records)
        summary = run_audit(pairs_path=path, max_rows=3)
        self.assertEqual(summary["row_counts"]["total_parsed_rows"], 3)

    def test_deterministic_split(self) -> None:
        """Test that pair split is strictly deterministic across repeated runs."""
        pair_key = "id_alpha::id_beta"
        split_1 = get_deterministic_split(pair_key)
        split_2 = get_deterministic_split(pair_key)
        self.assertEqual(split_1, split_2)
        self.assertIn(split_1, ("train", "val", "test"))


if __name__ == "__main__":
    unittest.main()

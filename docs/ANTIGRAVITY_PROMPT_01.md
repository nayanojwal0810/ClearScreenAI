# Antigravity Prompt 01 — Validation scaffolding only

Copy everything under **Prompt** into Antigravity after you have cloned this repository and placed the data files locally. This is a bounded coding task. Do not let Antigravity run the full dataset analysis yet.

## Prompt

You are the coding implementer for ClearScreenAI. The human project owner has delegated technical decisions to the project lead. Follow this specification exactly; do not redesign the product or add unrelated features.

### Goal

Prepare reproducible, low-cost scripts for the **signal-only validation phase** of an educational sanctions-screening and investigation-support prototype. We have NOT passed validation. Do not build the final product, frontend, API, Docker setup, agent workflow, or production architecture now.

### Data files expected locally

- data/raw/pairs.json.gz — OpenSanctions Pairs snapshot, 755,540 labeled entity pairs, ~409 MB compressed, snapshot dated 2025-12-09.
- data/raw/SDN_ENHANCED.ZIP — current official OFAC SDN enhanced XML export.

If either file is missing, do not download it, do not create fake replacement data, and do not abort the coding scaffold. Report which file is missing and continue using only synthetic test fixtures.

### Scope: create only these files

1. scripts/audit_dataset.py
2. scripts/evaluate_baselines.py
3. tests/test_validation_scripts.py
4. requirements-validation.txt
5. Update docs/VALIDATION_PLAN.md only if a real implementation detail needs to be documented; do not change its gates or success criteria without asking the project lead.

Use Python 3.11-compatible code. Prefer the standard library; add RapidFuzz only for the fuzzy baseline. Do not add heavy ML packages in this task.

### Script A — full dataset audit

Implement a streaming gzip JSONL reader. It must process the file line by line and never load the entire decompressed benchmark into RAM.

When the human later runs it on the real dataset, it must report aggregate-only information:
- total parsed rows, malformed rows and parse errors;
- labels and counts;
- left/right schema counts and schema-pair counts;
- counts of missing captions, empty properties and missing identifiers;
- duplicate pair-row counts using stable record IDs where present;
- counts of possible relation/auto-propagated schemas (including Occupancy, Succession, Ownership, Family and Directorship if actually present);
- expected-vs-observed snapshot counts and warnings;
- input file size, SHA-256, and timestamp of the audit.

Do not print or store real entity names, birth dates, addresses, IDs, aliases, raw JSON lines, or other row-level data. Do not hardcode expected counts as actual results. Expected counts are checks only.

Create a documented switch or separate function for defining the primary analyst-judged matching subset. **Do not silently assume schema types or infer analyst-judged rows from the positive label.** Use the audit's observed schema counts and the published paper to identify relation rows. If there is any ambiguity, report the schema table and leave the primary subset marked “not determined — review required” rather than inventing a filter. The script must still audit all rows.

Aggregate outputs should go under outputs/ (already ignored by .gitignore). Never include raw records or names in these outputs.

### Script B — transparent baselines

Implement a streaming evaluator for:
- normalized exact name/caption match;
- exact match over available alias values;
- a transparent RapidFuzz name/alias similarity score, with the scoring method and thresholds documented.

Use the entity fields such as caption/name/alias where appropriate. Exclude internal record IDs and any field that directly encodes the label or linkage decision. Do not call any external API, LLM or hosted model.

The evaluator must be able to process the complete benchmark. It must not silently downsample, truncate, or stop after a fixed row limit. Provide an optional explicit limit only for synthetic test fixtures; the default real-data mode must mean all rows.

Use deterministic splits and tune a threshold only on validation data, never on the final test partition. Deduplicate pair ordering when identifying repeated rows. Audit entity-ID overlap across partitions. If the flat snapshot does not support a truly entity-disjoint split without discarding too much data, clearly label results as pair-level evaluation and record the limitation. Do not claim unseen-entity generalization.

Report precision, recall, F1, confusion matrix and PR-AUC if implemented correctly. Report metrics separately for (a) all rows and (b) the analyst-judged subset only when its schema-based filter has been verified. Since this benchmark includes auto-propagated positive relation pairs, never present metrics on all rows alone as proof of identity-matching quality. Do not fabricate results when no real data has been processed.

Write aggregate metrics to outputs/baseline_summary.json. Put any row-level error examples in a local file only if essential; keep that file ignored and never print raw names in terminal output.

### Tests

Write compact unit tests using tiny synthetic records created inside the tests. Test gzip/JSONL parsing, label counting, normalization, exact matching, fuzzy scoring, malformed-line handling, and no raw entity content in the aggregate audit output. Do not copy examples from the real dataset into tests.

### Resource and quota rules

- Do not download data or models.
- Do not install anything automatically.
- Do not run on the real 409 MB file.
- Do not run a model, notebook, GPU job, full-corpus audit, or full-corpus baseline in this task.
- Do not call an external LLM or API.
- Do not create a frontend, API server, Dockerfile, CI workflow, vector database, or deployment configuration.
- Do not invent a model-performance result or mark any validation gate as passed.

### Repository rules

- Check git status before making changes.
- The existing .gitignore should ignore data/, outputs/, model artifacts and secrets. Verify this.
- Never stage data files, generated outputs, logs containing names, model weights, embeddings, or secrets.
- Run only the tiny synthetic unit tests. If the test dependency is unavailable, use Python's built-in unittest or clearly report the missing dependency; do not spend time troubleshooting the environment.
- Inspect the final diff and check the staged file list before committing.
- Commit and push only the scripts, tests, requirements file and any justified documentation changes to nayanojwal0810/ClearScreenAI.
- If authentication, remote configuration or push fails, do not claim success. Report the exact failure and next command needed.

### Required final response

Stop after the scaffold task. Report:
1. files created/changed;
2. tests run and results;
3. the exact commit SHA and push status;
4. confirmation that local data was not staged;
5. the exact next command for the owner to run the full audit, for example:
   python scripts/audit_dataset.py --pairs data/raw/pairs.json.gz --ofac-zip data/raw/SDN_ENHANCED.ZIP --out outputs/audit_summary.json
6. Do not run that command. Wait for the owner to send the output before making further changes.

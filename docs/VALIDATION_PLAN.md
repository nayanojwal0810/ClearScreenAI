# ClearScreenAI — Signal-Only Validation Plan

**Status:** selected for validation; not yet validated. No model results are available.

## Product hypothesis

ClearScreenAI is an educational sanctions-screening and investigation-support prototype. It should (1) rank records that may refer to the same person or organization, (2) show which names and identifiers support or contradict a candidate match, and (3) retrieve official OFAC guidance to help a human reviewer understand the process. It must never declare someone legally sanctioned or make an automated blocking/clearing decision.

## Why this problem

- Sanctions screening is a real compliance workflow. OFAC explicitly describes fuzzy name search, weak aliases and the need to review additional identifying information.
- Entity matching has a real human-labeled benchmark: OpenSanctions Pairs, 755,540 pairs from 293 sources across 45 jurisdictions, with multilingual names and incomplete/noisy attributes.
- A neural matcher can be measured against exact-match and fuzzy-matching baselines. GenAI can have a separate measurable role: retrieve official guidance and prepare source-grounded reviewer briefs.
- The pipeline naturally demonstrates data ingestion, entity resolution, neural NLP, RAG, API design, evaluation, testing and deployment without requiring a large model to be trained from scratch.

## Data sources

### 1. Labeled matching benchmark — OpenSanctions Pairs

- **Direct download:** https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs/resolve/d79295c551dd36f60745aae4a2345f4c8542b037/pairs.json.gz?download=true
- **Dataset card:** https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs
- **Upstream documentation:** https://www.opensanctions.org/docs/opensource/pairs/
- Snapshot: 2025-12-09; approximately 409 MB compressed; 755,540 labeled pairs.
- Upstream documentation marks the dataset for non-commercial use. The Hugging Face card specifies CC BY-NC 4.0. Cite OpenSanctions and the snapshot. Use only for non-commercial study/portfolio work unless appropriate permission is obtained. Do not upload the dataset to this repository.
- The selected OpenSanctions Pairs snapshot contains 755,540 labeled entity pairs spanning several entity schema types (including Person, Organization, LegalEntity, Company, and relationship-oriented schemas). Validation evaluates the documented binary judgement labels (positive and negative) across all pairs without treating relationship schemas as automatically propagated or excluding them without reliable supporting evidence. Unknown or missing labels are counted and reported separately.
- Upstream docs say the old flat snapshot is no longer updated; use it only because it is a fixed, directly downloadable benchmark for reproducible validation. Do not present it as the current sanctions list.

### 2. Current official target list — OFAC SDN Enhanced XML

- **Official page/download center:** https://sanctionslist.ofac.treas.gov/Home/SdnList
- **Direct Enhanced ZIP endpoint:** https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN_ENHANCED.ZIP
- OFAC listed the compressed enhanced XML at about 6.46 MB as of 2026-10-09; size and content change as OFAC updates the list. Record download date, file size and SHA-256.
- Use as a current searchable reference list for a small end-to-end screening smoke test, not as a labeled customer benchmark.

### 3. Official retrieval corpus for the GenAI part

Use only these authoritative sources for the validation prototype:
- OFAC FAQ #5, assessing name matches: https://ofac.treasury.gov/faqs/5
- OFAC weak alias FAQ #122: https://ofac.treasury.gov/faqs/122
- OFAC weak alias location FAQ #123: https://ofac.treasury.gov/faqs/123
- OFAC FAQ #246 on how Sanctions List Search works: https://ofac.treasury.gov/faqs/246
- OFAC compliance framework PDF: https://ofac.treasury.gov/system/files/126/framework_ofac_cc.pdf

Record the source URL and title with every retrieved chunk. Do not invent legal advice; clearly label outputs as educational and refer users to qualified compliance professionals.

## Validation approach (full data, no random sampling for the initial signal)

### Gate 0 — file and data audit (CPU)

- Process every line in the full compressed benchmark using streaming; do not load the full decompressed file into memory.
- Confirm readable gzip/JSONL, total rows, unique labels, row counts by label, schema pairs, source fields, missingness, malformed rows, and duplicates.
- Expected snapshot-level checks: 755,540 rows; labels positive and negative; about 76.9% positive overall. Verify actual counts and report differences rather than forcing expected numbers.
- Identify and audit entity schemas (including Person, Organization, LegalEntity, Company, and relationship-oriented schemas) and schema-pair label distributions. Measure missingness of business and personal identifiers (e.g. taxNumber, registrationNumber, passportNumber) on both entities and the pair, explicitly excluding internal entity IDs. Investigate duplicate pair keys, checking for conflicting labels and comparing canonicalized entity payloads.
- Inspect a few rows only to validate parser behavior; this is a parser smoke test, not the model evaluation dataset.
- Keep raw data in ignored data/; do not commit data or generated row-level names.

### Gate 1 — non-neural baselines (full eligible corpus)

- **Evaluation Scopes**:
  - **Primary scope (direct sanctions screening)**: Observations where both entities use official matchable schema types (`Person`, `Company`, `Organization`, `LegalEntity`, `Security`, `CryptoWallet`, `Vessel`, `Airplane`, `PublicBody`), with conflicting-label pair groups excluded.
  - **Secondary scope (broader entity matching)**: The broader benchmark including contextual schemas (`Occupancy`, `Succession`, `Position`, `Directorship`, `Ownership`, etc.), with the same conflict-exclusion policy and reported schema distribution.
- **Duplicate & Conflict Policy**:
  - Assign observations to deterministic splits (Train 70%, Val 15%, Test 15%) using the unordered pair key, ensuring all versions of a pair key reside in the same partition.
  - Preserve distinct entity payload versions of repeated pair keys.
  - Deduplicate only exact repeated observations (identical pair key, canonical payload hash, and label).
  - Exclude conflicting-label pair groups (both positive and negative judgements on the same pair key) from primary clean metrics, retaining them in diagnostic reporting.
- Baseline A: Unicode/case/whitespace-normalized exact name and caption matching.
- Baseline B: Exact match over available alias values.
- Baseline C: Transparent character/token fuzzy match using RapidFuzz with threshold tuned on validation data only.
- Evaluation metrics: Precision, recall, F1, Average Precision (AP, tie-score aware), and confusion matrix. For discrete binary baselines without continuous scores, AP is recorded as not applicable.
- Report metrics separately for primary matchable scope and secondary broader scope. Pair-level evaluation limitation remains explicitly disclosed without claiming unseen-entity generalization.

### Gate 2 — neural signal (full eligible corpus; one controlled run)

- Preferred low-cost signal: a small multilingual pretrained transformer embedding model (e.g., sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) plus a small learned pair classifier using text/field comparison features. Reuse embeddings for repeated entity records where safe. No model training from scratch.
- Use all eligible labeled rows across train/validation/test partitions; do not randomly reduce the dataset for the reported signal. Batch and stream processing to control RAM. If full-data embedding or training is likely to exceed the local machine budget, create a Colab/Kaggle run command and STOP for user approval rather than launching a long job.
- Keep entity identifiers out of model features. Inspect whether repeated entities could leak across partitions. If a truly entity-disjoint split cannot be implemented for this flat snapshot, disclose that limitation; do not claim unseen-entity generalization.
- Compare the neural model to the strongest non-neural baseline on the same held-out records. Use a fixed validation threshold; evaluate test once. No hyperparameter sweeps or API-based LLM calls in validation.

### Gate 3 — GenAI/RAG signal (small, fixed test set)

- Create 10–15 representative compliance questions from the cited OFAC pages (weak aliases, how to assess a possible match, evidence needed, role of human review).
- Test retrieval against the full downloaded guidance corpus. Record top-k source relevance and exact source URL.
- If local/free generation is feasible, generate short reviewer briefs with explicit evidence citations and a clear uncertainty statement. No paid API calls and no large LLM. Verify that every material statement is supported by a retrieved passage; otherwise mark the RAG signal as not yet proven.
- This small curated question set is an evaluation set for document retrieval/generation, not a substitute for full benchmark evaluation.

## Go / no-go rule

Proceed to the full project only if:
1. The data is complete, parsable and legally usable for a non-commercial portfolio.
2. A defensible evaluation can be run without label leakage or hidden test tuning.
3. The neural model gives meaningful value versus the strongest simple baseline (target: at least +2 absolute F1 points on the test holdout, or a clearly better recall at the same review volume without materially worse precision). If not, inspect errors once; do not force the neural model.
4. RAG retrieves relevant official passages and produces source-grounded, uncertainty-aware summaries in the fixed test set.
5. Compute and deployment cost are realistic for a student project.

The numerical uplift target is a decision gate for this project, not an expected result or a claim about current performance. If any gate fails, stop and reconsider the problem/design.

## Resource and repository rules

- No API keys, paid API calls, full-size LLMs, hyperparameter sweeps, frontend, API, Docker or production build during signal-only validation.
- The first Antigravity task is to create and test scripts on tiny synthetic fixtures only. It must not start a full-data run automatically. It must give exact commands for full-data audit/baselines, commit only code/docs/tests, verify that data/ is ignored and untracked, push to GitHub, then stop.
- Full dataset audit and baseline runs must process all eligible rows. Any long neural run must be launched only after the user receives the exact command and explicitly runs it; inspect returned output before proceeding.
- No results have been run yet. Never put fabricated metrics in a report or resume.

### Implementation scripts and commands (Gates 0 & 1)

- **Audit script:** `scripts/audit_dataset.py` (streaming gzip JSONL, standard library only, aggregate-only output without storing raw entity content):
  ```bash
  python scripts/audit_dataset.py --pairs data/raw/pairs.json.gz --ofac-zip data/raw/SDN_ENHANCED.ZIP --out outputs/audit_summary.json
  ```
- **Baseline evaluation:** `scripts/evaluate_baselines.py` (streaming exact name, exact alias, RapidFuzz token similarity with val-only threshold tuning):
  ```bash
  python scripts/evaluate_baselines.py --pairs data/raw/pairs.json.gz --out outputs/baseline_summary.json
  ```
- **Synthetic unit tests:** `tests/test_validation_scripts.py`:
  ```bash
  python -m unittest tests/test_validation_scripts.py
  ```
- **Validation dependencies:** `requirements-validation.txt` (`rapidfuzz>=3.0.0`).

## Research references

- OpenSanctions Pairs paper: https://arxiv.org/abs/2603.11051
- OpenSanctions Pairs docs/licensing: https://www.opensanctions.org/docs/opensource/pairs/
- OFAC Sanctions List Service: https://ofac.treasury.gov/sanctions-list-service
- OFAC FAQ #5: https://ofac.treasury.gov/faqs/5
- OFAC FAQ #122: https://ofac.treasury.gov/faqs/122
- OFAC FAQ #246: https://ofac.treasury.gov/faqs/246

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
- The benchmark paper describes 501,298 analyst-judged matching-subject pairs plus 254,242 automatic relation-pair propagations that are positive by construction. Therefore report both (a) results on the analyst-judged matching-subject subset, and (b) full-corpus results separately. Do not present full-corpus scores alone as proof of matcher quality.
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
- Identify relation schemas that represent automatically propagated pairs (e.g., Occupancy, Succession, Ownership/Family/Directorship and related relation types). Report the count excluded from the primary analyst-judged subset and document the exact rule; do not infer this filter from labels alone.
- Inspect a few rows only to validate parser behavior; this is a parser smoke test, not the model evaluation dataset.
- Keep raw data in ignored data/; do not commit data or generated row-level names.

### Gate 1 — non-neural baselines (full eligible corpus)

- Baseline A: Unicode/case/whitespace-normalized exact name and alias matching.
- Baseline B: a transparent character/token fuzzy match using RapidFuzz.
- Where feasible, add a small tabular baseline using field-level similarities and conflicts (alias, date of birth, country/nationality, identifiers, entity type). Exclude source entity IDs and any other fields that reveal label or record linkage identity.
- Use one documented deterministic split and threshold selection on validation data only. Do not tune on the final test set. Do not report pair-row random splitting as entity-level generalization.
- Use precision, recall, F1, PR-AUC, confusion matrix, and false-positive/false-negative examples. Show both the analyst-judged subset and the full corpus; state that benchmark class prevalence does not represent real-world customer screening prevalence.

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
3. The neural model gives meaningful value versus the strongest simple baseline (target: at least +2 absolute F1 points on the analyst-judged holdout, or a clearly better recall at the same review volume without materially worse precision). If not, inspect errors once; do not force the neural model.
4. RAG retrieves relevant official passages and produces source-grounded, uncertainty-aware summaries in the fixed test set.
5. Compute and deployment cost are realistic for a student project.

The numerical uplift target is a decision gate for this project, not an expected result or a claim about current performance. If any gate fails, stop and reconsider the problem/design.

## Resource and repository rules

- No API keys, paid API calls, full-size LLMs, hyperparameter sweeps, frontend, API, Docker or production build during signal-only validation.
- The first Antigravity task is to create and test scripts on tiny synthetic fixtures only. It must not start a full-data run automatically. It must give exact commands for full-data audit/baselines, commit only code/docs/tests, verify that data/ is ignored and untracked, push to GitHub, then stop.
- Full dataset audit and baseline runs must process all eligible rows. Any long neural run must be launched only after the user receives the exact command and explicitly runs it; inspect returned output before proceeding.
- Store aggregate metrics and concise error examples only after review. Never commit source records, names, downloaded files, model weights, embeddings, API keys or generated row-level predictions.
- No results have been run yet. Never put fabricated metrics in a report or resume.

## Research references

- OpenSanctions Pairs paper: https://arxiv.org/abs/2603.11051
- OpenSanctions Pairs docs/licensing: https://www.opensanctions.org/docs/opensource/pairs/
- OFAC Sanctions List Service: https://ofac.treasury.gov/sanctions-list-service
- OFAC FAQ #5: https://ofac.treasury.gov/faqs/5
- OFAC FAQ #122: https://ofac.treasury.gov/faqs/122
- OFAC FAQ #246: https://ofac.treasury.gov/faqs/246

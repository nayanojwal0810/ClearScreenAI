# ClearScreenAI

**Evidence-grounded sanctions screening research prototype** — a student project exploring entity-name matching, analyst review, and retrieval-grounded compliance assistance.

> **Important:** This is an educational prototype, not a sanctions compliance product. It must not be used to approve, reject, block, or clear real customers or transactions. A possible match is not a determination that a person is sanctioned. Human review and authoritative official sources are required.

## Current phase

**Phase 0 — data acquisition and signal-only validation planning.** The repository is intentionally kept small. The first check will compare simple name-matching baselines on a controlled test set generated from official sanctions-list records. We will not claim model performance until experiments actually run.

## Dataset

- Primary source: [OFAC Sanctions List Service — SDN List](https://sanctionslist.ofac.treas.gov/Home/SdnList)
- Direct enhanced XML download (compressed): [SDN_ENHANCED.ZIP](https://sanctionslist.ofac.treas.gov/Home/SdnList?leng=eng)
- OFAC explains its data formats and the distinction between strong and weak aliases in its [official FAQ](https://ofac.treasury.gov/faqs/122).
- Dataset page and download layout can change. Prefer the official download page if the direct link does not start a download.

The official list is a list of sanctioned targets and aliases; it is **not** a ready-made labeled customer screening benchmark. Validation queries and non-matches must be generated separately and documented as synthetic.

## Planned validation (not yet executed)

1. Confirm the downloaded archive and file format.
2. Parse records and aliases locally; report counts, missing fields, and parsing errors.
3. Create deterministic synthetic screening queries from a fixed seed, with positive variants and difficult non-match names; keep entity-level train/test separation to avoid leakage.
4. Compare exact matching and character-similarity/fuzzy matching against a small pretrained transformer-based reranker only if label construction and runtime justify it.
5. Report precision, recall, F1, precision-recall AUC where meaningful, recall at a fixed review volume, and examples of false positives/false negatives. Do not treat synthetic performance as real-world compliance accuracy.
6. Separately test retrieval against official compliance guidance, measuring retrieval relevance and whether answers cite retrieved passages. No LLM-based component is assumed to pass merely because it produces fluent text.

## Data handling

Do not commit downloaded sanctions lists, generated datasets containing personal names, model weights, API keys, or other large/generated files. Keep local data under ignored `data/` directories. Commit code, configuration, tests, and aggregate evaluation results only after reviewing them.

## Status

No validation has been run yet. No metrics or performance claims are available.

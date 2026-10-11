# ClearScreenAI

**Evidence-grounded sanctions screening and investigation copilot** — a student portfolio project combining entity matching, Deep Learning, retrieval-grounded GenAI, and a human-review workflow.

> **Safety boundary:** This is an educational prototype, not a sanctions compliance product. It must not automatically approve, reject, block, or clear customers or transactions. A possible match is not proof that a person or organization is sanctioned. Final review belongs to qualified humans using authoritative sources.

## Current phase

**Signal-only validation planning.** No experiments have been run and no performance claims are available. See [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md) for source decisions, licensing, evaluation gates, resource limits, and the go/no-go criteria.

## Data sources

1. **OpenSanctions Pairs** — labeled entity pairs for matching-model evaluation, snapshot dated 2025-12-09. [Direct download](https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs/resolve/d79295c551dd36f60745aae4a2345f4c8542b037/pairs.json.gz?download=true) (~409 MB compressed; 755,540 pairs). Upstream docs: [OpenSanctions matcher training data](https://www.opensanctions.org/docs/opensource/pairs/). The source marks this dataset as non-commercial; cite OpenSanctions and review the license before use beyond a non-commercial portfolio.
2. **Official OFAC SDN list** — current reference list for a small application smoke test. [Official download page](https://sanctionslist.ofac.treas.gov/Home/SdnList) | [Direct Enhanced ZIP](https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN_ENHANCED.ZIP).
3. **Official OFAC guidance** — [FAQ #5 on assessing possible matches](https://ofac.treasury.gov/faqs/5), [FAQ #122 on weak aliases](https://ofac.treasury.gov/faqs/122), [FAQ #246 on how list search works](https://ofac.treasury.gov/faqs/246), and [A Framework for OFAC Compliance Commitments](https://ofac.treasury.gov/system/files/126/framework_ofac_cc.pdf). These sources form the small retrieval corpus for testing source-grounded answers.

The OpenSanctions Pairs snapshot is a labeled matching benchmark, not a current sanctions list. The current OFAC list is a reference source, not a labeled customer benchmark. The published benchmark also includes automatically propagated positive relation pairs; validation will report analyst-judged identity pairs separately so headline performance is not inflated.

## Data handling

Raw data belongs only in ignored local folders such as `data/raw/`. Do not commit downloaded data, raw entity records, generated record-level names, model weights, embeddings, API keys, or row-level predictions. Commit code, tests, source citations, and reviewed aggregate metrics only.

## Status

- Repository initialized with README and .gitignore.
- Signal-only validation plan committed.
- Dataset has not been run through any parser or model.
- No performance metrics have been measured.

# Component 1: Corpus Foundation

## Purpose

This component defines the fixed corpus package for Agentic Document Intelligence.
It creates a stable dependency contract for later components without implementing
retrieval, graph querying, SQL runtime execution, or frontend behavior.

## Fixed corpus contents

### Source-derived asset

- Document:
  - `Microsoft_FY2025_10K_Summary.pdf`
  - copied into this workspace under `corpus/sources/`
  - this is the only source document in v1

### Synthetic asset

- Dataset:
  - `microsoft_fy2025_analyst_dataset`
  - represented as three CSV tables:
    - `financial_performance_by_segment.csv`
    - `geographic_revenue_mix.csv`
    - `product_family_signals.csv`

The dataset is synthetic and designed for demo-quality text-to-SQL questions.
It is aligned with the Microsoft FY2025 filing narrative, but it is not claimed
to be a direct extraction from the PDF.

## What later components may assume

- The active corpus is declared only through `corpus_manifest.json`.
- The source document path and hash are stable unless the manifest changes.
- The synthetic dataset schema is fixed and machine-readable.
- Capability flags in the manifest are the source of truth for what the corpus supports.
- Downstream preprocessing outputs must conform to `preprocessing_contract.json`.

## What later components may not assume

- They may not infer active files by scanning directories.
- They may not invent table names or column names outside the dataset schema.
- They may not treat the synthetic dataset as source-derived evidence.
- They may not add new preprocessing artifact formats without updating the contract.

## Synthetic dataset design

### Table: `financial_performance_by_segment`

Purpose:

- support revenue trend analysis
- compare segment performance
- support operating income and margin questions

Key columns:

- `fiscal_year`
- `segment_name`
- `revenue_usd_millions`
- `operating_income_usd_millions`
- `operating_margin_pct`
- `yoy_revenue_growth_pct`
- `narrative_driver`

### Table: `geographic_revenue_mix`

Purpose:

- support geographic revenue mix and trend questions

Key columns:

- `fiscal_year`
- `geography`
- `revenue_usd_millions`
- `revenue_mix_pct`
- `yoy_revenue_growth_pct`

### Table: `product_family_signals`

Purpose:

- support analyst-style product and strategic-priority questions

Key columns:

- `fiscal_year`
- `product_family`
- `strategic_priority`
- `revenue_signal_index`
- `margin_profile`
- `ai_relevance_score`
- `commentary`

## Defined artifact layout

- `corpus/sources/`
  - immutable source files used by the corpus package
- `corpus/datasets/`
  - fixed SQL demo tables
- `corpus/metadata/`
  - manifest, metadata, and future machine-readable summaries
- `corpus/contracts/`
  - output contracts for later preprocessing stages
- `artifacts/validation/`
  - generated validation reports
- `artifacts/experiments/`
  - later experiment outputs per component

## Validation scope in Component 1

Validation checks cover:

- source document presence
- manifest parsing
- document hash consistency
- metadata completeness
- capability consistency
- dataset schema and column validation
- preprocessing contract completeness

This component does not validate retrieval quality, graph quality, or SQL answer quality yet.

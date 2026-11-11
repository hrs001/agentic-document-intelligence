# Component 4B: Retrieval Merge and Dedup Across Variants

## What this sub-component does

- takes the transformed retrieval executor output
- merges results across all transformed query variants
- deduplicates by `source_chunk_id` within each original sub-query
- keeps every provenance trail that matched the chunk

## Why this exists

Many transformed queries return the same child chunk. This step creates one clean candidate pool per original sub-query before:

- coverage scoring
- corrective HyDE triggering
- reranking
- MMR diversification

## Input

- `transformed_retrieval_executor_report.json`

## Output

- one merged result set per original sub-query
- each merged chunk keeps:
  - `best_score`
  - `match_count`
  - `provenance_list`
  - `variant_types`
  - `matched_query_texts`

## Active dedup policy

- dedup key: `source_chunk_id`
- dedup scope: within each `sub_query_id`
- score kept: highest score among duplicate hits

## Why sub-query scoped dedup is used

The same chunk can legitimately help multiple original sub-queries. Deduplicating globally would hide that signal and hurt later sub-query-specific rerank/MMR steps.

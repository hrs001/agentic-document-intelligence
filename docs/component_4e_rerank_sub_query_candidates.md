# Component 4E: Rerank Merged Candidates Against Original Sub-Query

## What this sub-component does

- takes the merged candidate pool for each original sub-query
- reranks those candidates against the original sub-query itself
- preserves merged provenance and retrieval history

## Why this exists

The retrieval pool is built from many transformed variants. Before MMR diversification, we want a cleaner ordering that is anchored back to the original sub-query, not to whichever transformed query happened to retrieve the chunk.

## Input

- `corrective_hyde_retry_report.json`

## Output

For each original sub-query:

- `candidate_count`
- `reranked_count`
- `reranked_matches`
  - original merged fields
  - plus `rerank_score`

## Current reranker

- `pinecone-rerank-v0`

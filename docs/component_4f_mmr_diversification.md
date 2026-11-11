# Component 4F: MMR Diversification on Reranked Sub-Query Candidates

## What this sub-component does

- takes the reranked candidate pool for each original sub-query
- uses existing chunk embeddings to measure similarity between candidates
- applies MMR to keep relevant chunks while reducing near-duplicates

## Inputs used

- reranked sub-query candidates
- existing embedding records
- existing Pinecone vectors

## Why this exists

The transformed retrieval path often returns several chunks that say almost the same thing. MMR helps preserve the best evidence while making the final evidence bundle more diverse and useful for answer synthesis.

## Current policy

- relevance signal: `rerank_score`
- diversity signal: cosine similarity between existing chunk vectors
- selection scope: per original sub-query

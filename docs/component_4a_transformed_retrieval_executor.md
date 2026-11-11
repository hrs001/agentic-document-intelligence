# Component 4A: Transformed Retrieval Executor

## What this sub-component does

- takes the transformed query bundle
- runs Pinecone hybrid retrieval for each query variant
- preserves provenance for every retrieved chunk

## Query variant types

- original sub-query
- multi-query rewrite
- step-back query

## What it does not do yet

- merge and dedup across variants
- coverage scoring
- corrective HyDE retrieval
- rerank and MMR

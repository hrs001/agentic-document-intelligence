# Component 3D: Query Decomposition Repair and Normalization

## What this sub-component does

- takes the first-pass `gpt-5-mini` decomposition result
- runs a lightweight gate to detect weak outputs
- calls the LLM again only when repair is likely needed
- returns a cleaned decomposition bundle for retrieval

## Repair triggers

- unresolved references
- possible under-splitting
- duplicated sub-queries
- overly generic retrieval phrasing
- multi-intent residue inside one sub-query

## Current design

- generic gate, not corpus-specific
- LLM repair for semantic normalization
- schema validation on repaired output

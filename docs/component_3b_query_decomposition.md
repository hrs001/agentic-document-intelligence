# Component 3B: Query Decomposition

## What this sub-component does

- inspects one sanitized user query
- decides whether the query should stay atomic or be split
- returns a deterministic decomposition result for downstream retrieval

## Current implementation

- cheap deterministic logic only
- splits on:
  - explicit question boundaries
  - clear multi-intent conjunctions
  - list-style financial asks
- does not split comparison-style queries

## Inputs

- one sanitized user query

## Outputs

- `needs_decomposition`
- `sub_queries`
- `reasoning_type`
- `decomposition_strategy`

## Current limits

- no LLM reasoning
- no domain-aware semantic decomposition
- no confidence scoring yet

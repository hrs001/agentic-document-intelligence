# Component 3E: Multi-Query Generation

## What this sub-component does

- generates a small set of alternative retrieval rewrites
- keeps the rewrite fanout tightly bounded
- tries to vary retrieval angle without changing the user's meaning

## Current constraints

- max 3 rewrites
- no placeholders
- no scope expansion
- no duplicate rewrites

## Evaluation goals

- diversity without drift
- stable output schema
- useful retrieval angles
